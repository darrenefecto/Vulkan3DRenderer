#pragma once
#include "VulkanContext.h"
#include "Texture.h"
#include "Mesh.h"
#include "Camera.h"
#include "Light.h"
#include <array>
#include <string>
#include <vector>

class Renderer {
public:
    static constexpr int MAX_FRAMES = 2;
    bool framebufferResized = false;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout materialLayout = VK_NULL_HANDLE; // used by Model::createDescriptors

    // Scene lighting: add/edit lights from application code before drawFrame.
    LightSystem lights;

    void init(Context& ctx, const std::array<std::string, 6>& skyboxFaces);
    void drawFrame(Context& ctx, Camera& camera, const std::vector<Model*>& models);
    void shutdown(Context& ctx);

private:
    struct Frame {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        VkBuffer ubo = VK_NULL_HANDLE;
        VkDeviceMemory uboMem = VK_NULL_HANDLE;
        void* uboMapped = nullptr;
        VkDescriptorSet globalSet = VK_NULL_HANDLE;
    };

    // Depth-only array image: one sampled view + one renderable view per layer.
    struct ShadowMap2D {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView arrayView = VK_NULL_HANDLE;
        std::vector<VkImageView> layerViews;
        uint32_t size = 0, layers = 0;
    };
    struct ShadowMapCube {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView cubeArrayView = VK_NULL_HANDLE;   // CUBE_ARRAY, for sampling
        std::vector<VkImageView> faceViews;           // 2D, one per cube face
        uint32_t size = 0;
    };

    VkDescriptorSetLayout globalLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout skyboxSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout meshPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout skyboxPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout shadow2DPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout shadowCubePipeLayout = VK_NULL_HANDLE;
    VkPipeline meshPipeline = VK_NULL_HANDLE;
    VkPipeline skyboxPipeline = VK_NULL_HANDLE;
    VkPipeline shadow2DPipeline = VK_NULL_HANDLE;
    VkPipeline shadowCubePipeline = VK_NULL_HANDLE;

    Cubemap skybox;
    VkDescriptorSet skyboxSet = VK_NULL_HANDLE;

    // shadow resources (shared by all frames; same-queue barriers synchronize them)
    ShadowMap2D csmShadow;     // NUM_CASCADES layers
    ShadowMap2D spotShadow;    // MAX_SHADOW_SPOT layers
    ShadowMap2D areaShadow;    // MAX_SHADOW_AREA layers
    ShadowMapCube pointShadow; // MAX_SHADOW_POINT cubes
    VkSampler shadowSampler = VK_NULL_HANDLE;

    Frame frames[MAX_FRAMES];
    std::vector<VkSemaphore> renderFinished;  // per swapchain image
    std::vector<VkFence> imagesInFlight;      // per swapchain image
    uint32_t currentFrame = 0;

    void createLayouts(Context& ctx);
    void createPipelines(Context& ctx);
    VkPipeline createPipeline(Context& ctx, const std::string& vertSpv,
        const std::string& fragSpv, VkPipelineLayout layout, bool skybox);
    VkPipeline createShadowPipeline(Context& ctx, const std::string& vertSpv,
        const std::string& fragSpv, VkPipelineLayout layout);
    void createShadowCubePipeline(Context& ctx);
    void createFrames(Context& ctx);
    void createSkyboxResources(Context& ctx, const std::array<std::string, 6>& faces);
    void createShadowResources(Context& ctx);
    void createShadowMap2D(Context& ctx, ShadowMap2D& map, uint32_t size, uint32_t layers);
    void createPerImageSync(Context& ctx);
    void recordFrame(Context& ctx, Frame& frame, uint32_t imageIndex, const std::vector<Model*>& models,
        const GlobalUBO& ubo);
    void recordShadowPasses(Context& ctx, VkCommandBuffer cmd, const std::vector<Model*>& models,
        const GlobalUBO& ubo, VkDescriptorSet globalSet);
    void recordDepthPass2D(VkCommandBuffer cmd, VkImageView layerView, uint32_t size,
        const glm::mat4& viewProj, Model& model);
    void recordDepthPassCubeFace(VkCommandBuffer cmd, VkImageView faceView, uint32_t size,
        int vpIndex, Model& model);
    void drawMeshGeometry(VkCommandBuffer cmd, VkPipelineLayout layout, Model& model);
    void recreate(Context& ctx);
};
