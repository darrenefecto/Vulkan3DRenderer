#pragma once
#include "VulkanContext.h"
#include "Texture.h"
#include "Mesh.h"
#include "Camera.h"
#include "Light.h"
#include "RenderGraph.h"
#include "ClusteredLighting.h"
#include <array>
#include <string>
#include <vector>
#include <unordered_map>

// Hybrid renderer: opaque meshes use their selected Deferred, Forward, or
// Custom path; transparent materials use internal clustered Forward+ shading.

class Renderer {
public:
    static constexpr int MAX_FRAMES = 2;
    bool framebufferResized = false;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout materialLayout = VK_NULL_HANDLE; // used by Model::createDescriptors

    // Scene lighting: add/edit lights from application code before drawFrame.
    LightSystem lights;

    // --- HDR post-processing (composite pass) ---
    bool taaEnabled = true; // TAA camera reprojection; moving meshes need per-object velocities
    float taaHistoryWeight = 0.9f;
    // Conservative fallback: history cap while transparent meshes are present.
    // A future per-pixel reactive mask can replace this scene-wide protection.
    float taaTransparentHistoryCap = 0.45f;
    float exposure = 1.0f;     // linear exposure multiplier before tonemapping
    float gtaoRadius = 0.75f;  // World-space GTAO radius
    float gtaoStrength = 1.25f; // GTAO darkening strength
    int   tonemapOperator = 2; // 0 = none (clamp), 1 = Reinhard, 2 = ACES filmic

    void init(Context& ctx, const std::array<std::string, 6>& skyboxFaces);
    void drawFrame(Context& ctx, Camera& camera, const std::vector<Model*>& models);
    void shutdown(Context& ctx);

    // The frame graph managing passes + resource dependencies (public so the
    // app can inspect execution order via graph.dumpExecutionOrder()).
    RenderGraph graph;

private:
    struct Frame {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        VkBuffer ubo = VK_NULL_HANDLE;
        VkDeviceMemory uboMem = VK_NULL_HANDLE;
        void* uboMapped = nullptr;
        VkDescriptorSet globalSet = VK_NULL_HANDLE;

        // --- clustered lighting (per frame in flight) ---
        VkBuffer clusterLights = VK_NULL_HANDLE;       // host-visible ClusterLight[]
        VkDeviceMemory clusterLightsMem = VK_NULL_HANDLE;
        void* clusterLightsMapped = nullptr;
        VkBuffer clusterGrid = VK_NULL_HANDLE;         // device-local uvec2[]
        VkDeviceMemory clusterGridMem = VK_NULL_HANDLE;
        VkBuffer clusterIndices = VK_NULL_HANDLE;      // device-local uint[]
        VkDeviceMemory clusterIndicesMem = VK_NULL_HANDLE;
        VkDescriptorSet clusterSet = VK_NULL_HANDLE;
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

    // Screen-sized image owned by the renderer and imported into the graph.
    struct RTImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        RenderGraph::Handle rg = RenderGraph::INVALID_HANDLE;
    };

    VkDescriptorSetLayout globalLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout forwardAOLayout = VK_NULL_HANDLE; // set 2: final GTAO texture
    VkDescriptorSetLayout gbufferSampleLayout = VK_NULL_HANDLE; // G-Buffer plus AO samplers
    VkDescriptorSetLayout motionSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout taaSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssrSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssrTemporalSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout compositeSetLayout = VK_NULL_HANDLE;  // 1 HDR sampler
    VkDescriptorSetLayout skyboxSetLayout = VK_NULL_HANDLE;     // skybox cubemap
    VkDescriptorSetLayout clusterSetLayout = VK_NULL_HANDLE;    // 3 SSBOs

    VkPipelineLayout meshPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout shadow2DPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout shadowCubePipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout gbufferPipeLayout = VK_NULL_HANDLE;      // = mesh layout
    VkPipelineLayout deferredPipeLayout = VK_NULL_HANDLE;     // global + G-Buffer/AO samplers
    VkPipelineLayout forwardPlusPipeLayout = VK_NULL_HANDLE;  // global + material + clusters
    VkPipelineLayout geometryMotionPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout motionPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout taaPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout compositePipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout skyboxPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout clusterPipeLayout = VK_NULL_HANDLE;      // compute

    VkPipeline meshPipelineHDR = VK_NULL_HANDLE;       // classic forward -> HDR target
    VkPipeline shadow2DPipeline = VK_NULL_HANDLE;
    VkPipeline shadowCubePipeline = VK_NULL_HANDLE;
    VkPipeline gbufferPipeline = VK_NULL_HANDLE;       // MRT geometry pass
    VkPipeline forwardAOPrepassPipeline = VK_NULL_HANDLE; // forward/custom AO-only MRT
    VkPipeline gtaoPipeline = VK_NULL_HANDLE;          // Screen-space ambient occlusion
    VkPipeline gtaoBlurPipeline = VK_NULL_HANDLE;     // Edge-aware AO denoising
    VkPipeline deferredPipeline = VK_NULL_HANDLE;      // fullscreen lighting
    VkPipeline forwardPlusBlendPipeline = VK_NULL_HANDLE; // transparent materials
    std::unordered_map<std::string, VkPipeline> customForwardPipelines;
    VkPipeline geometryMotionPipeline = VK_NULL_HANDLE;
    VkPipeline transparentMotionPipeline = VK_NULL_HANDLE;
    VkPipeline motionPipeline = VK_NULL_HANDLE;
    VkPipeline taaPipeline = VK_NULL_HANDLE;
    VkPipeline ssrTracePipeline = VK_NULL_HANDLE;
    VkPipeline ssgiTracePipeline = VK_NULL_HANDLE;
    VkPipeline ssgiCombinePipeline = VK_NULL_HANDLE;
    VkPipeline ssgiTemporalPipeline = VK_NULL_HANDLE;
    VkPipeline ssrCombinePipeline = VK_NULL_HANDLE;
    VkPipeline ssrTemporalPipeline = VK_NULL_HANDLE;
    VkPipelineLayout ssrTraceLayout = VK_NULL_HANDLE;
    VkPipelineLayout ssrCombineLayout = VK_NULL_HANDLE;
    VkPipelineLayout ssgiDenoiseLayout = VK_NULL_HANDLE;
    VkPipelineLayout ssrTemporalLayout = VK_NULL_HANDLE;
    VkPipeline compositePipeline = VK_NULL_HANDLE;     // HDR -> swapchain
    VkPipeline skyboxPipeline = VK_NULL_HANDLE;         // environment background
    VkPipeline clusterPipeline = VK_NULL_HANDLE;       // compute

    // Environment cubemap: sampled as IBL by every shading shader and as the
    // skybox background by the deferred lighting pass (no dedicated pipeline).
    Cubemap skybox;
    Cubemap irradiance; // baked cosine-weighted irradiance / PI
    Cubemap specularPrefilter; // GGX convolution stored in mip chain
    Texture2D brdfLut; // linear BRDF split-sum lookup

    // shadow resources (shared by all frames; same-queue barriers synchronize them)
    ShadowMap2D csmShadow;     // NUM_CASCADES layers
    ShadowMap2D spotShadow;    // MAX_SHADOW_SPOT layers
    ShadowMap2D areaShadow;    // MAX_SHADOW_AREA layers
    ShadowMapCube pointShadow; // MAX_SHADOW_POINT cubes
    VkSampler shadowSampler = VK_NULL_HANDLE;
    VkSampler gbufferSampler = VK_NULL_HANDLE;         // linear/clamp, for gbuffer+HDR

    // --- render targets for the graph-driven paths (recreated on resize) ---
    RTImage gAlbedo;      // R8G8B8A8_SRGB        rgb = albedo, a = metallic
    RTImage gNormal;      // R16G16B16A16_SFLOAT  rgb = world N, a = roughness
    RTImage gEmissive;    // R16G16B16A16_SFLOAT  rgb = emissive, a = AO
    RTImage gDepth;       // D32_SFLOAT (sampled + attachment)
    RTImage gtaoRaw;      // R16_SFLOAT raw screen-space AO
    RTImage gtaoBlur;     // R16_SFLOAT intermediate denoised AO
    RTImage gtao;         // R16_SFLOAT final screen-space AO
    RTImage transparentMotionVectors; // Separate transparent velocity; opaque motion is never overwritten
    RTImage motionVectors; // RGBA16F: velocity.xy, previous depth, validity
    RTImage taaReactiveMask; // R16F: per-pixel accumulated transparent coverage
    RTImage taaHistory[2]; // HDR temporal ping-pong; no read/write feedback
    RTImage iblDiffuse; // deferred-only diffuse IBL for controlled SSGI replacement
    RTImage iblSpecular; // deferred-only indirect specular for SSR replacement
    RTImage ssgiCombined; // diffuse SSGI composite after SSR, before TAA
    RTImage ssgiIndirect; // current SSGI radiance/confidence
    RTImage ssgiDenoised; // edge-aware spatial filtered indirect radiance
    RTImage ssgiHistory[2]; // temporal indirect radiance/confidence
    RTImage ssrReflection; // HDR radiance + hit confidence
    RTImage ssrHistory[2]; // RGBA16F: radiance.rgb and reflection confidence
    RTImage ssrCombined; // composited HDR, separate from trace source
    RTImage sceneColor;   // R16G16B16A16_SFLOAT HDR

    VkDescriptorSet forwardAOSet = VK_NULL_HANDLE;
    VkDescriptorSet gbufferSampleSet = VK_NULL_HANDLE;
    VkDescriptorSet gtaoHorizontalSet = VK_NULL_HANDLE;
    VkDescriptorSet gtaoVerticalSet = VK_NULL_HANDLE;
    VkDescriptorSet motionSet = VK_NULL_HANDLE;
    VkDescriptorSet taaSets[2]{};
    VkDescriptorSet ssrTraceSet = VK_NULL_HANDLE;
    VkDescriptorSet ssrCombineSet = VK_NULL_HANDLE;
    VkDescriptorSet ssgiCombineSet = VK_NULL_HANDLE;
    VkDescriptorSet ssgiDenoiseSet = VK_NULL_HANDLE;
    VkDescriptorSet ssgiTemporalSets[2]{};
    VkPipeline ssgiDenoisePipeline = VK_NULL_HANDLE;
    VkDescriptorSet ssrTemporalSets[2]{};
    VkDescriptorSet compositeSet = VK_NULL_HANDLE;
    glm::mat4 previousViewProj{ 1.0f };
    glm::mat4 previousUnjitteredViewProj{ 1.0f }; // camera-cut detection independent of TAA jitter
    std::unordered_map<const Mesh*, glm::mat4> previousMeshTransforms;
    bool ssrEnabled = true;
    bool ssgiTraceEnabled = false; // opt-in until GPU validation and diffuse composition
    bool ssgiHistoryValid = false;
    uint32_t ssgiHistoryIndex = 0;
    bool ssrHistoryValid = false;
    uint32_t ssrHistoryIndex = 0;
    bool taaHistoryValid = false;
    bool taaWasEnabled = false;
    uint32_t taaHistoryIndex = 0; // most recently written
    uint32_t taaFrameNumber = 0;
    int lastSubmittedFrame = -1;
    VkDescriptorSet skyboxSet = VK_NULL_HANDLE;

    // render-graph handles for imported resources
    std::vector<RenderGraph::Handle> hSwapchain;   // per swapchain image
    RenderGraph::Handle hCsmShadow = RenderGraph::INVALID_HANDLE;
    RenderGraph::Handle hSpotShadow = RenderGraph::INVALID_HANDLE;
    RenderGraph::Handle hAreaShadow = RenderGraph::INVALID_HANDLE;
    RenderGraph::Handle hPointShadow = RenderGraph::INVALID_HANDLE;
    RenderGraph::Handle hFrameUBO[MAX_FRAMES]{};
    RenderGraph::Handle hClusterLights[MAX_FRAMES]{};
    RenderGraph::Handle hClusterGrid[MAX_FRAMES]{};
    RenderGraph::Handle hClusterIndices[MAX_FRAMES]{};

    Frame frames[MAX_FRAMES];
    std::vector<VkSemaphore> renderFinished;  // per swapchain image
    std::vector<VkFence> imagesInFlight;      // per swapchain image
    uint32_t currentFrame = 0;

    uint32_t clusterLightCount = 0;           // lights uploaded this frame

    // --- setup ---
    void createLayouts(Context& ctx);
    void createPipelines(Context& ctx);

    struct PipelineDesc {
        std::string vertSpv, fragSpv;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::vector<VkFormat> colorFormats;
        VkFormat depthFormat = VK_FORMAT_UNDEFINED;   // UNDEFINED = no depth
        bool vertexInput = true;
        bool depthTest = true, depthWrite = true;
        VkCompareOp depthCompare = VK_COMPARE_OP_LESS;
        bool blending = false;
        VkCullModeFlags cull = VK_CULL_MODE_BACK_BIT;
        bool depthBias = false;
    };
    VkPipeline createGraphicsPipeline(Context& ctx, const PipelineDesc& d);
    VkPipeline getCustomForwardPipeline(Context& ctx, const Mesh& mesh);

    VkPipeline createShadowPipeline(Context& ctx, const std::string& vertSpv,
        const std::string& fragSpv, VkPipelineLayout layout);
    void createShadowCubePipeline(Context& ctx);
    void createFrames(Context& ctx);
    void createSkyboxResources(Context& ctx, const std::array<std::string, 6>& faces);
    void createShadowResources(Context& ctx);
    void createShadowMap2D(Context& ctx, ShadowMap2D& map, uint32_t size, uint32_t layers);
    void createPerImageSync(Context& ctx);

    // --- deferred / clustered resources ---
    void createGraphResources(Context& ctx);      // gbuffer + HDR + depth
    void destroyGraphResources(Context& ctx);
    void createClusterResources(Context& ctx);    // SSBOs + compute pipeline
    void createGraphDescriptors(Context& ctx);
    void recordMotionPass(VkCommandBuffer cmd, VkExtent2D extent, const glm::mat4& invCurrent, const glm::mat4& prev);
    void recordGeometryMotionPass(VkCommandBuffer cmd, VkExtent2D extent,
        const std::vector<Model*>& models, VkDescriptorSet globalSet,
        const glm::mat4& previousVP, bool valid);
    void recordTransparentMotionPass(VkCommandBuffer cmd, VkExtent2D extent,
        const std::vector<Model*>& models, VkDescriptorSet globalSet,
        const glm::mat4& previousVP, bool valid);
    void recordSSRTemporalPass(VkCommandBuffer cmd, VkExtent2D extent, uint32_t dst, bool valid, const glm::mat4& currentToPreviousClip);
    void recordSSGIDenoisePass(VkCommandBuffer cmd, VkExtent2D extent, const glm::mat4& invProj);
    void recordSSGICombinePass(VkCommandBuffer cmd, VkExtent2D extent);
    void recordSSGITemporalPass(VkCommandBuffer cmd, VkExtent2D extent, uint32_t dst, bool valid, const glm::mat4& currentToPreviousClip);
    void recordSSGITracePass(VkCommandBuffer cmd, VkExtent2D extent, const glm::mat4& invProj);
    void recordSSRPass(VkCommandBuffer cmd, VkExtent2D extent, bool combine, const glm::mat4& invProj);
    void recordTAAPass(VkCommandBuffer cmd, VkExtent2D extent, uint32_t dst, bool valid, bool hasTransparency);    // gbuffer/composite/cluster sets
    void registerGraphResources(Context& ctx);    // import everything into the graph

    // --- frame recording ---
    void recordGraphFrame(Context& ctx, Frame& frame, uint32_t imageIndex,
        const std::vector<Model*>& models, const GlobalUBO& ubo,
        const Camera& camera);                                       // unified hybrid frame
    void buildGraph(Context& ctx, Frame& frame, uint32_t imageIndex,
        const std::vector<Model*>& models, const GlobalUBO& ubo,
        const Camera& camera);                                       // declares passes

    // All barriers are inserted by the render graph around this pass.
    void recordShadowPasses(Context& ctx, VkCommandBuffer cmd, const std::vector<Model*>& models,
        const GlobalUBO& ubo, VkDescriptorSet globalSet);
    void recordDepthPass2D(VkCommandBuffer cmd, VkImageView layerView, uint32_t size,
        const glm::mat4& viewProj, Model& model);
    void recordDepthPassCubeFace(VkCommandBuffer cmd, VkImageView faceView, uint32_t size,
        int vpIndex, Model& model);
    // Draws opaque meshes assigned to the requested mode.
    void drawMeshesForMode(VkCommandBuffer cmd, VkPipelineLayout layout,
        const std::vector<Model*>& models, RenderMode mode);
    void drawMeshGeometryFiltered(VkCommandBuffer cmd, VkPipelineLayout layout, Model& model,
        bool transparent);

    void recordGBufferPass(VkCommandBuffer cmd, const std::vector<Model*>& models,
        VkDescriptorSet globalSet, VkExtent2D extent);
    void recordForwardAOPrepass(VkCommandBuffer cmd, const std::vector<Model*>& models,
        VkDescriptorSet globalSet, VkExtent2D extent);
    void recordGTAOPass(VkCommandBuffer cmd, VkDescriptorSet globalSet,
        const glm::mat4& invProj, VkExtent2D extent);
    void recordGTAODenoisePass(VkCommandBuffer cmd, VkDescriptorSet globalSet,
        VkDescriptorSet inputSet, RTImage& output, const glm::mat4& invProj,
        VkExtent2D extent, glm::vec2 direction);
    void recordDeferredLightingPass(VkCommandBuffer cmd, VkDescriptorSet globalSet,
        const glm::mat4& invViewProj, VkExtent2D extent);
    void recordSkyboxPass(VkCommandBuffer cmd, VkDescriptorSet globalSet, VkExtent2D extent);
    void recordForwardPass(Context& ctx, VkCommandBuffer cmd,
        const std::vector<Model*>& models, Frame& frame, VkExtent2D extent);
    void recordTransparentPass(VkCommandBuffer cmd, const std::vector<Model*>& models,
        Frame& frame, const Camera& camera, VkExtent2D extent);
    void recordCompositePass(VkCommandBuffer cmd, uint32_t imageIndex, VkExtent2D extent);
    void recordClusterBuildPass(VkCommandBuffer cmd, Frame& frame, const Camera& camera,
        VkExtent2D extent);

    ClusterParams makeClusterParams(const Camera& camera, VkExtent2D extent) const;
    void updateClusterLights(Frame& frame);
    void recreate(Context& ctx);
};
