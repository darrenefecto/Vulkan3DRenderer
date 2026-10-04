#pragma once
#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

class Camera {
public:
    glm::vec3 position{ 0.0f, 1.5f, 4.0f };
    float yaw = -90.0f, pitch = -10.0f;
    float fov = 60.0f, nearClip = 0.05f, farClip = 1000.0f;
    float moveSpeed = 5.0f, sensitivity = 0.12f;
    bool firstMouse = true;

    glm::vec3 front() const {
        glm::vec3 f;
        f.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        f.y = sin(glm::radians(pitch));
        f.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        return glm::normalize(f);
    }
    glm::mat4 view() const { return glm::lookAt(position, position + front(), { 0, 1, 0 }); }
    glm::mat4 proj(float aspect) const {
        glm::mat4 p = glm::perspective(glm::radians(fov), aspect, nearClip, farClip);
        p[1][1] *= -1; // GLM -> Vulkan clip space
        return p;
    }
    void onMouseMove(double x, double y) {
        static double lastX = 0, lastY = 0;
        if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }
        yaw += float(x - lastX) * sensitivity;
        pitch += float(lastY - y) * sensitivity;
        lastX = x; lastY = y;
        pitch = glm::clamp(pitch, -89.0f, 89.0f);
    }
    void processInput(GLFWwindow* w, float dt) {
        float v = moveSpeed * dt * (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 4.0f : 1.0f);
        glm::vec3 f = front();
        glm::vec3 right = glm::normalize(glm::cross(f, { 0, 1, 0 }));
        if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) position += f * v;
        if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) position -= f * v;
        if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) position -= right * v;
        if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) position += right * v;
        if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_SPACE) == GLFW_PRESS)
            position.y += v;
        if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            position.y -= v;
    }
};
