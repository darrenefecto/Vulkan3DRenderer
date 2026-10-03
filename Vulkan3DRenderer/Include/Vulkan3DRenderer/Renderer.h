#pragma once
#include "VulkanContext.h"
#include "Texture.h"
#include "Mesh.h"
#include "Camera.h"
#include <array>
#include <string>

class Renderer {
public:
    static constexpr int MAX_FRAMES = 2;
    bool framebufferResized = false;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout materialLayout = VK_NULL_HANDLE; // used by Model::createDescriptors

    glm::vec3 lightDir = glm::normalize(glm::vec3(-0.6f, -1.0f, -0.35f));
    glm::vec3 lightColor{ 1.0f, 0.98f, 0.92f };
    float ambient = 0.12f;

    void init(Context& ctx, const std::array<std::string, 6>& skyboxFaces);
    void drawFrame(Context& ctx, Camera& camera, Model& model);
    void shutdown(Context& ctx);

private:
    struct GlobalUBO {
        glm::mat4 view, proj;
        glm::vec4 camPos, lightDir, lightColor, ambient;
    };
    struct Frame {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        VkBuffer ubo = VK_NULL_HANDLE;
        VkDeviceMemory uboMem = VK_NULL_HANDLE;
        void* uboMapped = nullptr;
        VkDescriptorSet globalSet = VK_NULL_HANDLE;
    };

    VkDescriptorSetLayout globalLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout skyboxSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout meshPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout skyboxPipeLayout = VK_NULL_HANDLE;
    VkPipeline meshPipeline = VK_NULL_HANDLE;
    VkPipeline skyboxPipeline = VK_NULL_HANDLE;

    Cubemap skybox;
    VkDescriptorSet skyboxSet = VK_NULL_HANDLE;

    Frame frames[MAX_FRAMES];
    std::vector<VkSemaphore> renderFinished;  // per swapchain image
    std::vector<VkFence> imagesInFlight;      // per swapchain image
    uint32_t currentFrame = 0;

    void createLayouts(Context& ctx);
    void createPipelines(Context& ctx);
    VkPipeline createPipeline(Context& ctx, const std::string& vertSpv, const std::string& fragSpv,
        VkPipelineLayout layout, bool skybox);
    void createFrames(Context& ctx);
    void createSkyboxResources(Context& ctx, const std::array<std::string, 6>& faces);
    void createPerImageSync(Context& ctx);
    void recordFrame(Context& ctx, Frame& frame, uint32_t imageIndex, Model& model);
    void recreate(Context& ctx);
};