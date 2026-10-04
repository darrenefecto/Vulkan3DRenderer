#include <Vulkan3DRenderer/VulkanContext.h>
#include <Vulkan3DRenderer/Renderer.h>
#include <Vulkan3DRenderer/Mesh.h>
#include <Vulkan3DRenderer/Camera.h>
#include <iostream>
#include <array>
#include <cmath>

static Camera g_camera;
static bool g_captured = true;

static void cursorPosCallback(GLFWwindow*, double x, double y) {
    if (g_captured) g_camera.onMouseMove(x, y);
}
static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        g_captured = !g_captured;
        glfwSetInputMode(w, GLFW_CURSOR, g_captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        if (g_captured) g_camera.firstMouse = true;
    }
}
static void framebufferSizeCallback(GLFWwindow* w, int, int) {
    auto* renderer = (Renderer*)glfwGetWindowUserPointer(w);
    if (renderer) renderer->framebufferResized = true;
}

// Demo scene lighting: one sun (with CSM), two point lights (one shadowed),
// one spot light and one rectangular area light.
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
        if (!glfwInit()) throw std::runtime_error("GLFW init failed");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(1600, 900, "Vulkan PBR Renderer (CSM + Point/Spot/Area shadows)", nullptr, nullptr);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

        Context ctx;
        ctx.init(window, /*validationLayers=*/true);

        Renderer renderer;
        const std::array<std::string, 6> skyFaces = {
            "Assets/Skybox/px.jpg", "Assets/Skybox/nx.jpg",
            "Assets/Skybox/py.jpg", "Assets/Skybox/ny.jpg",
            "Assets/Skybox/pz.jpg", "Assets/Skybox/nz.jpg" };
        renderer.init(ctx, skyFaces);
        setupLights(renderer.lights);

        Model model1;
        model1.load(ctx, model1Path);
        model1.createDescriptors(ctx, renderer.descriptorPool, renderer.materialLayout);

        Model model2;
        model2.load(ctx, model2Path);
        model2.createDescriptors(ctx, renderer.descriptorPool, renderer.materialLayout);

        glfwSetWindowUserPointer(window, &renderer);

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

            renderer.drawFrame(ctx, g_camera, { &model1, &model2 });
        }

        vkDeviceWaitIdle(ctx.device);
        model1.destroy(ctx);
        model2.destroy(ctx);
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
