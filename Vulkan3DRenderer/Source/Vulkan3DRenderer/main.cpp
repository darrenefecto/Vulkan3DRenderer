#include <Vulkan3DRenderer/VulkanContext.h>
#include <Vulkan3DRenderer/Renderer.h>
#include <Vulkan3DRenderer/Mesh.h>
#include <Vulkan3DRenderer/Camera.h>
#include <iostream>
#include <filesystem>
#include <array>
#include <vector>
#include <cmath>

static Camera g_camera;
static bool g_captured = true;

// Everything the input callbacks need (window user pointer).
struct AppState {
    Renderer* renderer = nullptr;
};
static AppState g_app;

static const char* modeName(RenderMode mode) {
    switch (mode) {
    case RenderMode::Deferred: return "Deferred";
    case RenderMode::Forward:  return "Forward";
    case RenderMode::Custom:   return "Custom Forward";
    }
    return "Unknown";
}

static const char* tonemapName(int op) {
    switch (op) { case 0: return "None (clamp)"; case 1: return "Reinhard"; case 2: return "ACES filmic"; }
    return "?";
}

static void configureAssetWorkingDirectory(const char* executablePath) {
    namespace fs = std::filesystem;
    const auto hasAssets = [](const fs::path& root) {
        return fs::exists(root / "Assets" / "Shaders" / "deferred_lighting.frag.spv") &&
            fs::exists(root / "Assets" / "Skybox" / "px.png");
    };

    std::array<fs::path, 2> starts = {
        fs::current_path(), fs::absolute(executablePath).parent_path()
    };
    for (fs::path start : starts) {
        for (int depth = 0; depth < 8 && !start.empty(); ++depth) {
            std::error_code ec;
            if (hasAssets(start)) {
                fs::current_path(start, ec);
                if (!ec) {
                    std::cout << "[Assets] Working directory: " << start.string() << "\n";
                    return;
                }
            }
            const fs::path parent = start.parent_path();
            if (parent == start) break;
            start = parent;
        }
    }
    std::cerr << "[Assets] Project asset root not found; using current directory.\n";
}


static void cursorPosCallback(GLFWwindow*, double x, double y) {
    if (g_captured) g_camera.onMouseMove(x, y);
}
static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) {
        g_captured = !g_captured;
        glfwSetInputMode(w, GLFW_CURSOR, g_captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        if (g_captured) g_camera.firstMouse = true;
    }
    // HDR post-processing
    if (key == GLFW_KEY_T) {
        Renderer* r = g_app.renderer;
        r->tonemapOperator = (r->tonemapOperator + 1) % 3;
        std::cout << "[HDR] tonemap: " << tonemapName(r->tonemapOperator) << "\n";
    }
    if (key == GLFW_KEY_MINUS || key == GLFW_KEY_EQUAL) {
        Renderer* r = g_app.renderer;
        r->exposure = glm::clamp(r->exposure * (key == GLFW_KEY_MINUS ? 0.8f : 1.25f), 0.05f, 16.0f);
        std::cout << "[HDR] exposure: " << r->exposure << "\n";
    }
}
static void framebufferSizeCallback(GLFWwindow*, int, int) {
    if (g_app.renderer) g_app.renderer->framebufferResized = true;
}

// Demo scene lighting: one sun (with CSM), a ring of moving point lights,
// one spot light and one rectangular area light.
//
// Transparent materials use clustered lighting. Opaque Forward and Deferred
// paths use the light counts supported by their standard shaders.
static void setupLights(LightSystem& lights) {
    // --- sun ---
    lights.sun.enabled = true;
    lights.sun.direction = glm::normalize(glm::vec3(-0.6f, -1.0f, -0.35f));
    lights.sun.color = glm::vec3(1.0f, 0.96f, 0.88f);
    lights.sun.intensity = 3.0f;
    lights.sun.castShadow = true;      // 4-cascade CSM
    lights.sun.shadowDistance = 60.0f;

    // --- point lights ---
    PointLight warm;
    warm.position = glm::vec3(3.0f, 2.0f, 2.0f);
    warm.color = glm::vec3(1.0f, 0.55f, 0.25f);
    warm.intensity = 25.0f;
    warm.range = 14.0f;
    warm.castShadow = true;            // cube shadow map
    lights.pointLights.push_back(warm);

    PointLight cool;
    cool.position = glm::vec3(-3.0f, 1.0f, -2.0f);
    cool.color = glm::vec3(0.3f, 0.55f, 1.0f);
    cool.intensity = 15.0f;
    cool.range = 10.0f;
    cool.castShadow = false;
    lights.pointLights.push_back(cool);

    // extra unshadowed point lights to make light clustering visible
    for (int i = 0; i < 24; i++) {
        PointLight p;
        float a = float(i) / 24.0f * 6.2831853f;
        p.position = glm::vec3(cosf(a) * 6.0f, 0.5f + 2.5f * float(i % 3) / 2.0f, sinf(a) * 6.0f);
        p.color = glm::vec3(0.5f + 0.5f * cosf(a), 0.5f + 0.5f * cosf(a + 2.1f), 0.5f + 0.5f * cosf(a + 4.2f));
        p.intensity = 4.0f;
        p.range = 5.0f;
        p.castShadow = false;
        lights.pointLights.push_back(p);
    }

    // --- spot light (shadowed) ---
    SpotLight spot;
    spot.position = glm::vec3(0.0f, 5.0f, 3.0f);
    spot.direction = glm::normalize(glm::vec3(0.0f, -1.0f, -0.6f));
    spot.color = glm::vec3(1.0f, 0.95f, 0.8f);
    spot.intensity = 60.0f;
    spot.range = 25.0f;
    spot.innerAngleDeg = 16.0f;
    spot.outerAngleDeg = 24.0f;
    spot.castShadow = true;
    lights.spotLights.push_back(spot);

    // --- rectangular area light (shadowed) ---
    AreaLight panel;
    panel.position = glm::vec3(-2.5f, 3.0f, 2.5f);
    panel.right = glm::vec3(1.0f, 0.0f, 0.0f);
    panel.up = glm::normalize(glm::vec3(0.0f, 0.35f, -1.0f)); // tilted towards origin
    panel.width = 2.5f;
    panel.height = 1.2f;
    panel.color = glm::vec3(0.85f, 0.9f, 1.0f);
    panel.intensity = 10.0f;
    panel.range = 20.0f;
    panel.castShadow = true;
    panel.twoSided = true;
    lights.areaLights.push_back(panel);

    lights.ambientColor = glm::vec3(0.02f, 0.025f, 0.035f);
    lights.iblIntensity = 1.0f;
}

int main(int argc, char** argv) {
    std::string model1Path = argc > 1 ? argv[1] : "Assets/Models/model.obj";
    std::string model2Path = argc > 2 ? argv[2] : "Assets/Models/fire-extinguisher.glb";

    try {
        configureAssetWorkingDirectory(argc > 0 ? argv[0] : ".");
        if (!glfwInit()) throw std::runtime_error("GLFW init failed");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(1600, 900,
            "Vulkan Hybrid Renderer | T=Tonemap -/+=Exposure",
            nullptr, nullptr);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

        Context ctx;
        ctx.init(window, /*validationLayers=*/true);

        Renderer renderer;
        const std::array<std::string, 6> skyFaces = {
            "Assets/Skybox/px.png", "Assets/Skybox/nx.png",
            "Assets/Skybox/py.png", "Assets/Skybox/ny.png",
            "Assets/Skybox/pz.png", "Assets/Skybox/nz.png" };
        renderer.init(ctx, skyFaces);
        setupLights(renderer.lights);

        Model model1(RenderMode::Deferred);
        model1.load(ctx, model1Path);
        model1.createDescriptors(ctx, renderer.descriptorPool, renderer.materialLayout);

        Model model2(RenderMode::Forward);
        const bool model2Loaded = std::filesystem::exists(model2Path);
        if (model2Loaded) {
            model2.load(ctx, model2Path);
            model2.createDescriptors(ctx, renderer.descriptorPool, renderer.materialLayout);
        }
        else {
            std::cerr << "[Scene] Optional Forward model not found; continuing with the Deferred model only: "
                << model2Path << "\n";
        }
        std::vector<Model*> sceneModels{ &model1 };
        if (model2Loaded) sceneModels.push_back(&model2);

        // Custom objects can select both SPIR-V shaders in their constructor.
        // Model customModel(RenderMode::Custom,
        //     "Assets/Shaders/Custom/custom.vert.spv",
        //     "Assets/Shaders/Custom/custom.frag.spv");
        // customModel.load(ctx, "Assets/Models/model.obj");
        // For one mesh: model2.setMeshCustomShader(0, vertexSpv, fragmentSpv);

        g_app.renderer = &renderer;
        glfwSetWindowUserPointer(window, &g_app);

        std::cout << "model1: " << modeName(model1.meshes.empty() ? RenderMode::Deferred : model1.meshes[0].renderMode) << "\n";
        if (model2Loaded)
            std::cout << "model2: " << modeName(model2.meshes.empty() ? RenderMode::Forward : model2.meshes[0].renderMode) << "\n";
        std::cout << "HDR: tonemap = " << tonemapName(renderer.tonemapOperator)
            << ", exposure = " << renderer.exposure << "\n";
        std::cout << "Keys: T tonemap, -/= exposure\n";

        double last = glfwGetTime();
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            double now = glfwGetTime();
            float dt = float(now - last);
            last = now;
            if (g_captured) g_camera.processInput(window, dt);

            // subtle light animation to show shadows tracking the light
            if (!renderer.lights.pointLights.empty()) {
                float t = (float)now * 0.7f;
                renderer.lights.pointLights[0].position.x = 3.0f * cosf(t);
                renderer.lights.pointLights[0].position.z = 3.0f * sinf(t);
            }

            renderer.drawFrame(ctx, g_camera, sceneModels);
        }

        vkDeviceWaitIdle(ctx.device);
        model1.destroy(ctx);
        if (model2Loaded) model2.destroy(ctx);
        renderer.shutdown(ctx);
        ctx.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
