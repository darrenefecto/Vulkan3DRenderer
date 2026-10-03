#include <Vulkan3DRenderer/VulkanContext.h>
#include <Vulkan3DRenderer/Renderer.h>
#include <Vulkan3DRenderer/Mesh.h>
#include <Vulkan3DRenderer/Camera.h>
#include <iostream>
#include <array>

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

int main(int argc, char** argv) {
    std::string modelPath = argc > 1 ? argv[1] : "Assets/Models/model.obj";

    try {
        if (!glfwInit()) throw std::runtime_error("GLFW init failed");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        GLFWwindow* window = glfwCreateWindow(1600, 900, "Vulkan 1.4 Mesh Renderer", nullptr, nullptr);
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

        Model model;
        model.load(ctx, modelPath);
        model.createDescriptors(ctx, renderer.descriptorPool, renderer.materialLayout);

        glfwSetWindowUserPointer(window, &renderer);

        double last = glfwGetTime();
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            double now = glfwGetTime();
            float dt = float(now - last);
            last = now;
            if (g_captured) g_camera.processInput(window, dt);
            renderer.drawFrame(ctx, g_camera, model);
        }

        vkDeviceWaitIdle(ctx.device);
        model.destroy(ctx);
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