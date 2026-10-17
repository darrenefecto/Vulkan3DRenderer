#include <Vulkan3DRenderer/Renderer.h>
#include <fstream>
#include <iostream>
#include <cstring>

using namespace LightLimits;

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    size_t size = (size_t)f.tellg();
    std::vector<char> buf(size);
    f.seekg(0);
    f.read(buf.data(), size);
    return buf;
}

// Counts which hybrid passes are needed for the current scene.
struct ScenePathUsage {
    bool forward = false;
    bool custom = false;
    bool transparent = false;
    bool needsClusters() const { return transparent; }
};

static ScenePathUsage classifyScene(const std::vector<Model*>& models) {
    ScenePathUsage usage;
    for (Model* model : models) {
        if (!model) continue;
        for (const Mesh& mesh : model->meshes) {
            const Material& material = model->materials[mesh.materialIndex];
            if (material.transparent) {
                usage.transparent = true;
                continue;
            }
            switch (mesh.renderMode) {
            case RenderMode::Deferred: break;
            case RenderMode::Forward:  usage.forward = true; break;
            case RenderMode::Custom:   usage.custom = true; break;
            }
        }
    }
    return usage;
}

void Renderer::init(Context& ctx, const std::array<std::string, 6>& skyboxFaces) {
    createLayouts(ctx);

    VkDescriptorPoolSize sizes[3]{};
    sizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 512 };
    sizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024 };
    sizes[2] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = 1024;
    pi.poolSizeCount = 3;
    pi.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descriptorPool));

    createSkyboxResources(ctx, skyboxFaces);
    createShadowResources(ctx);
    createFrames(ctx);
    createGraphResources(ctx);
    createClusterResources(ctx);
    createGraphDescriptors(ctx);
    createPipelines(ctx);
    createPerImageSync(ctx);
    registerGraphResources(ctx);
}

void Renderer::createLayouts(Context& ctx) {
    {   // set 0: global UBO + env cubemap + 4 shadow maps
        VkDescriptorSetLayoutBinding b[6]{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        for (uint32_t i = 1; i < 6; i++)
            b[i] = { i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 6;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &globalLayout));
    }
    {   // set 1: material UBO + 5 PBR textures (albedo/normal/mr/emissive/ao)
        VkDescriptorSetLayoutBinding b[6]{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        for (uint32_t i = 1; i < 6; i++)
            b[i] = { i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 6;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &materialLayout));
    }
    {   // set 1 (deferred lighting): 4 gbuffer samplers
        VkDescriptorSetLayoutBinding b[4]{};
        for (uint32_t i = 0; i < 4; i++)
            b[i] = { i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                     VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 4;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &gbufferSampleLayout));
    }
    {   // set 0 (composite): HDR scene color sampler
        VkDescriptorSetLayoutBinding b{ 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                        VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 1;
        ci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &compositeSetLayout));
    }
    {   // skybox: set 1 cubemap sampler (skybox.frag)
        VkDescriptorSetLayoutBinding b{
            0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr
        };
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 1;
        ci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &skyboxSetLayout));
    }
    {   // clustered lighting: light list (ro) + cluster grid (rw) + index list (rw)
        VkDescriptorSetLayoutBinding b[3]{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                 VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        b[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                 VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        b[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                 VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 3;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &clusterSetLayout));
    }

    VkPushConstantRange meshPush{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) };
    {   // mesh pipeline layout: set0 global, set1 material, push model
        VkDescriptorSetLayout sets[2] = { globalLayout, materialLayout };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 2;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &meshPush;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &meshPipeLayout));
    }
    gbufferPipeLayout = meshPipeLayout; // alias: same interface (destroyed once)
    {   // 2D shadow depth: push { lightViewProj, model }
        VkPushConstantRange push{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) * 2 };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &shadow2DPipeLayout));
    }
    {   // cube shadow depth: set0 global (face matrices) + push { model, vpIndex }
        struct { glm::mat4 model; glm::ivec4 idx; } pushSizeProbe;
        VkPushConstantRange push{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                  0, (uint32_t)sizeof(pushSizeProbe) };
        VkDescriptorSetLayout sets[1] = { globalLayout };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 1;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &shadowCubePipeLayout));
    }
    {   // deferred lighting: set0 global, set1 gbuffer samplers, push { invViewProj, screen }
        VkDescriptorSetLayout sets[2] = { globalLayout, gbufferSampleLayout };
        VkPushConstantRange push{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::mat4) + sizeof(glm::vec4) };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 2;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &deferredPipeLayout));
    }
    {   // forward+: set0 global, set1 material, set2 cluster SSBOs
        // push: model (vertex, offset 0) + cluster params (fragment, offset 64)
        VkDescriptorSetLayout sets[3] = { globalLayout, materialLayout, clusterSetLayout };
        VkPushConstantRange pushes[2]{};
        pushes[0] = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) };
        pushes[1] = { VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4),
                      sizeof(ClusterParams) - sizeof(glm::mat4) };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 3;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 2;
        ci.pPushConstantRanges = pushes;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &forwardPlusPipeLayout));
    }
    {   // composite: set0 HDR sampler, push { exposure, tonemapOp }
        VkDescriptorSetLayout sets[1] = { compositeSetLayout };
        VkPushConstantRange push{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec4) };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 1;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &compositePipeLayout));
    }
    {   // skybox: set0 global UBO, set1 cubemap
        VkDescriptorSetLayout sets[2] = { globalLayout, skyboxSetLayout };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 2;
        ci.pSetLayouts = sets;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &skyboxPipeLayout));
    }
    {   // cluster build (compute): set0 cluster SSBOs, push full ClusterParams
        VkDescriptorSetLayout sets[1] = { clusterSetLayout };
        VkPushConstantRange push{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusterParams) };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 1;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &clusterPipeLayout));
    }
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

// Generic dynamic-rendering pipeline used by every pass.
VkPipeline Renderer::createGraphicsPipeline(Context& ctx, const PipelineDesc& d) {
    auto vertCode = readFile(d.vertSpv);
    auto fragCode = readFile(d.fragSpv);
    VkShaderModule vert = ctx.createShaderModule(vertCode);
    VkShaderModule frag = ctx.createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    auto binding = Vertex::binding();
    auto attrs = Vertex::attributes();
    VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    if (d.vertexInput) {
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
        vertexInput.pVertexAttributeDescriptions = attrs.data();
    }

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = d.cull;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    raster.depthBiasEnable = d.depthBias ? VK_TRUE : VK_FALSE;

    VkPipelineMultisampleStateCreateInfo msaa{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth.depthTestEnable = (d.depthFormat != VK_FORMAT_UNDEFINED && d.depthTest) ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = d.depthWrite ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp = d.depthCompare;

    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(d.colorFormats.size());
    for (auto& b : blendAttachments) {
        b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (d.blending) {
            b.blendEnable = VK_TRUE;
            b.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            b.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            b.colorBlendOp = VK_BLEND_OP_ADD;
            b.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            b.alphaBlendOp = VK_BLEND_OP_ADD;
        }
    }
    VkPipelineColorBlendStateCreateInfo blend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend.attachmentCount = (uint32_t)blendAttachments.size();
    blend.pAttachments = blendAttachments.data();

    std::vector<VkDynamicState> dynStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    if (d.depthBias) dynStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
    VkPipelineDynamicStateCreateInfo dynamic{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic.dynamicStateCount = (uint32_t)dynStates.size();
    dynamic.pDynamicStates = dynStates.data();

    VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering.colorAttachmentCount = (uint32_t)d.colorFormats.size();
    rendering.pColorAttachmentFormats = d.colorFormats.data();
    rendering.depthAttachmentFormat = d.depthFormat;

    VkGraphicsPipelineCreateInfo ci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    ci.pNext = &rendering;
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vertexInput;
    ci.pInputAssemblyState = &inputAssembly;
    ci.pViewportState = &viewport;
    ci.pRasterizationState = &raster;
    ci.pMultisampleState = &msaa;
    ci.pDepthStencilState = &depth;
    ci.pColorBlendState = &blend;
    ci.pDynamicState = &dynamic;
    ci.layout = d.layout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline));
    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
    return pipeline;
}

VkPipeline Renderer::getCustomForwardPipeline(Context& ctx, const Mesh& mesh) {
    if (mesh.customVertexShaderSpv.empty() || mesh.customFragmentShaderSpv.empty())
        throw std::runtime_error("Custom render mode requires vertex and fragment SPIR-V paths.");

    const std::string key = mesh.customVertexShaderSpv + "\n" + mesh.customFragmentShaderSpv;
    const auto found = customForwardPipelines.find(key);
    if (found != customForwardPipelines.end()) return found->second;

    PipelineDesc desc{};
    desc.vertSpv = mesh.customVertexShaderSpv;
    desc.fragSpv = mesh.customFragmentShaderSpv;
    desc.layout = meshPipeLayout;
    desc.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT };
    desc.depthFormat = VK_FORMAT_D32_SFLOAT;
    VkPipeline pipeline = createGraphicsPipeline(ctx, desc);
    customForwardPipelines.emplace(key, pipeline);
    return pipeline;
}

VkPipeline Renderer::createShadowPipeline(Context& ctx, const std::string& vertSpv,
    const std::string& fragSpv, VkPipelineLayout layout) {
    PipelineDesc d{};
    d.vertSpv = vertSpv;
    d.fragSpv = fragSpv;
    d.layout = layout;
    d.colorFormats = {}; // depth-only
    d.depthFormat = VK_FORMAT_D32_SFLOAT;
    d.depthBias = true;
    return createGraphicsPipeline(ctx, d);
}

void Renderer::createPipelines(Context& ctx) {
    // --- classic forward (per-mesh RenderMode::Forward), shaded linear into HDR ---
    {
        PipelineDesc d{};
        d.vertSpv = "Assets/Shaders/mesh.vert.spv";
        d.fragSpv = "Assets/Shaders/mesh.frag.spv";
        d.layout = meshPipeLayout;
        d.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT };
        d.depthFormat = VK_FORMAT_D32_SFLOAT;
        meshPipelineHDR = createGraphicsPipeline(ctx, d);
    }
    shadow2DPipeline = createShadowPipeline(ctx, "Assets/Shaders/shadow2d.vert.spv",
        "Assets/Shaders/shadow2d.frag.spv", shadow2DPipeLayout);
    createShadowCubePipeline(ctx);

    // --- G-Buffer (deferred / hybrid) ---
    {
        PipelineDesc d{};
        d.vertSpv = "Assets/Shaders/geometry.vert.spv";
        d.fragSpv = "Assets/Shaders/gbuffer.frag.spv";
        d.layout = gbufferPipeLayout;
        d.colorFormats = { VK_FORMAT_R8G8B8A8_SRGB,           // albedo + metallic
                           VK_FORMAT_R16G16B16A16_SFLOAT,     // normal + roughness
                           VK_FORMAT_R16G16B16A16_SFLOAT };   // emissive + AO
        d.depthFormat = VK_FORMAT_D32_SFLOAT;
        gbufferPipeline = createGraphicsPipeline(ctx, d);
    }
    // --- deferred lighting (fullscreen) ---
    {
        PipelineDesc d{};
        d.vertSpv = "Assets/Shaders/fullscreen.vert.spv";
        d.fragSpv = "Assets/Shaders/deferred_lighting.frag.spv";
        d.layout = deferredPipeLayout;
        d.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT };
        d.vertexInput = false;
        d.cull = VK_CULL_MODE_NONE; // Fullscreen triangles must not be culled.
        deferredPipeline = createGraphicsPipeline(ctx, d);
    }
    // --- dedicated skybox pass ---
    {
        PipelineDesc d{};
        d.vertSpv = "Assets/Shaders/skybox.vert.spv";
        d.fragSpv = "Assets/Shaders/skybox.frag.spv";
        d.layout = skyboxPipeLayout;
        d.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT };
        d.depthFormat = VK_FORMAT_D32_SFLOAT;
        d.vertexInput = false;
        d.cull = VK_CULL_MODE_NONE;
        d.depthTest = true;
        d.depthWrite = false;
        d.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL;
        skyboxPipeline = createGraphicsPipeline(ctx, d);
    }
    // --- transparent materials use blended clustered Forward+ shading ---
    {
        PipelineDesc d{};
        d.vertSpv = "Assets/Shaders/geometry.vert.spv";
        d.fragSpv = "Assets/Shaders/forwardplus.frag.spv";
        d.layout = forwardPlusPipeLayout;
        d.colorFormats = { VK_FORMAT_R16G16B16A16_SFLOAT };
        d.depthFormat = VK_FORMAT_D32_SFLOAT;
        d.depthWrite = false;
        d.blending = true;
        forwardPlusBlendPipeline = createGraphicsPipeline(ctx, d);
    }
    // --- composite (HDR -> swapchain, exposure + tonemap) ---
    {
        PipelineDesc d{};
        d.vertSpv = "Assets/Shaders/fullscreen.vert.spv";
        d.fragSpv = "Assets/Shaders/composite.frag.spv";
        d.layout = compositePipeLayout;
        d.colorFormats = { ctx.swapchainFormat };
        d.vertexInput = false;
        d.cull = VK_CULL_MODE_NONE; // Fullscreen triangles must not be culled.
        compositePipeline = createGraphicsPipeline(ctx, d);
    }
    // --- cluster light assignment (compute) ---
    {
        auto code = readFile("Assets/Shaders/cluster_build.comp.spv");
        VkShaderModule mod = ctx.createShaderModule(code);
        VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        ci.stage = stage;
        ci.layout = clusterPipeLayout;
        VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &clusterPipeline));
        vkDestroyShaderModule(ctx.device, mod, nullptr);
    }
}

// Cube shadow pipeline: vertex + fragment (fragment writes radial distance to depth).
void Renderer::createShadowCubePipeline(Context& ctx) {
    PipelineDesc d{};
    d.vertSpv = "Assets/Shaders/shadow_cube.vert.spv";
    d.fragSpv = "Assets/Shaders/shadow_cube.frag.spv";
    d.layout = shadowCubePipeLayout;
    d.colorFormats = {};
    d.depthFormat = VK_FORMAT_D32_SFLOAT;
    d.depthBias = true;
    shadowCubePipeline = createGraphicsPipeline(ctx, d);
}

// ---------------------------------------------------------------------------
// Skybox / shadows (unchanged from the forward renderer)
// ---------------------------------------------------------------------------

void Renderer::createSkyboxResources(Context& ctx, const std::array<std::string, 6>& faces) {
    try {
        skybox.createFromFiles(ctx, faces);
        std::cout << "[Skybox] Cubemap loaded with " << skybox.mipLevels << " mip levels.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "[Skybox] " << e.what() << "\nUsing fallback gray skybox.\n";
        skybox.createSolid(ctx, 60, 70, 90, 255);
    }
    // The cubemap is bound here for the dedicated skybox pass. It is also
    // bound in the global descriptor set for image-based lighting on materials.
}

void Renderer::createShadowMap2D(Context& ctx, ShadowMap2D& map, uint32_t size, uint32_t layers) {
    map.size = size;
    map.layers = layers;
    ctx.createImage(size, size, 1, layers, 0, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        map.image, map.memory);
    map.arrayView = ctx.createImageView(map.image, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_ASPECT_DEPTH_BIT, 1, VK_IMAGE_VIEW_TYPE_2D_ARRAY, layers);
    map.layerViews.resize(layers);
    for (uint32_t i = 0; i < layers; i++)
        map.layerViews[i] = ctx.createImageView(map.image, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_ASPECT_DEPTH_BIT, 1, VK_IMAGE_VIEW_TYPE_2D, 1, /*baseLayer=*/i);
    // start in SHADER_READ_ONLY so the per-frame transition is well-defined from frame 0
    ctx.transitionImageLayout(map.image, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, layers, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void Renderer::createShadowResources(Context& ctx) {
    createShadowMap2D(ctx, csmShadow, CSM_RESOLUTION, NUM_CASCADES);
    createShadowMap2D(ctx, spotShadow, SPOT_RESOLUTION, MAX_SHADOW_SPOT);
    createShadowMap2D(ctx, areaShadow, AREA_RESOLUTION, MAX_SHADOW_AREA);

    // point light cube map array
    pointShadow.size = POINT_RESOLUTION;
    ctx.createImage(POINT_RESOLUTION, POINT_RESOLUTION, 1, MAX_SHADOW_POINT * 6,
        VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        pointShadow.image, pointShadow.memory);
    pointShadow.cubeArrayView = ctx.createImageView(pointShadow.image, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_ASPECT_DEPTH_BIT, 1, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, MAX_SHADOW_POINT * 6);
    pointShadow.faceViews.resize(MAX_SHADOW_POINT * 6);
    for (uint32_t i = 0; i < MAX_SHADOW_POINT * 6; i++)
        pointShadow.faceViews[i] = ctx.createImageView(pointShadow.image, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_ASPECT_DEPTH_BIT, 1, VK_IMAGE_VIEW_TYPE_2D, 1, /*baseLayer=*/i);
    ctx.transitionImageLayout(pointShadow.image, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, MAX_SHADOW_POINT * 6, VK_IMAGE_ASPECT_DEPTH_BIT);

    // hardware-compare sampler (returns filtered visibility in [0,1])
    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // outside the map = fully lit
    si.compareEnable = VK_TRUE;
    si.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VK_CHECK(vkCreateSampler(ctx.device, &si, nullptr, &shadowSampler));

    // plain sampler for the gbuffer / HDR targets
    VkSamplerCreateInfo gi{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    gi.magFilter = VK_FILTER_LINEAR;
    gi.minFilter = VK_FILTER_LINEAR;
    gi.addressModeU = gi.addressModeV = gi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(ctx.device, &gi, nullptr, &gbufferSampler));
}

// ---------------------------------------------------------------------------
// Per-frame resources + clustered lighting buffers
// ---------------------------------------------------------------------------

void Renderer::createFrames(Context& ctx) {
    // command buffers
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = ctx.commandPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = MAX_FRAMES;
    VkCommandBuffer cmds[MAX_FRAMES];
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cai, cmds));

    // global descriptor sets
    VkDescriptorSetLayout layouts[MAX_FRAMES] = { globalLayout, globalLayout };
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = descriptorPool;
    dai.descriptorSetCount = MAX_FRAMES;
    dai.pSetLayouts = layouts;
    VkDescriptorSet sets[MAX_FRAMES];
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &dai, sets));

    VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES; i++) {
        Frame& f = frames[i];
        f.cmd = cmds[i];
        f.globalSet = sets[i];
        VK_CHECK(vkCreateSemaphore(ctx.device, &sci, nullptr, &f.imageAvailable));
        VK_CHECK(vkCreateFence(ctx.device, &fci, nullptr, &f.inFlight));

        ctx.createBuffer(sizeof(GlobalUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, f.ubo, f.uboMem);
        vkMapMemory(ctx.device, f.uboMem, 0, sizeof(GlobalUBO), 0, &f.uboMapped);

        VkDescriptorBufferInfo bi{ f.ubo, 0, sizeof(GlobalUBO) };
        VkDescriptorImageInfo envImg{ skybox.sampler, skybox.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo csmImg{ shadowSampler, csmShadow.arrayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo spotImg{ shadowSampler, spotShadow.arrayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo pointImg{ shadowSampler, pointShadow.cubeArrayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo areaImg{ shadowSampler, areaShadow.arrayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

        VkWriteDescriptorSet writes[6]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = f.globalSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &bi;

        const VkDescriptorImageInfo* imgs[5] = { &envImg, &csmImg, &spotImg, &pointImg, &areaImg };
        for (uint32_t b = 1; b < 6; b++) {
            writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[b].dstSet = f.globalSet;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[b].pImageInfo = imgs[b - 1];
        }
        vkUpdateDescriptorSets(ctx.device, 6, writes, 0, nullptr);
    }
}

void Renderer::createClusterResources(Context& ctx) {
    using namespace ClusterLimits;
    for (int i = 0; i < MAX_FRAMES; i++) {
        Frame& f = frames[i];
        ctx.createBuffer(sizeof(ClusterLight) * MAX_LIGHTS,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            f.clusterLights, f.clusterLightsMem);
        vkMapMemory(ctx.device, f.clusterLightsMem, 0, sizeof(ClusterLight) * MAX_LIGHTS, 0,
            &f.clusterLightsMapped);
        ctx.createBuffer(sizeof(glm::uvec2) * CLUSTER_COUNT,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.clusterGrid, f.clusterGridMem);
        ctx.createBuffer(sizeof(uint32_t) * CLUSTER_COUNT * MAX_PER_CLUSTER,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.clusterIndices, f.clusterIndicesMem);
    }
}

// ---------------------------------------------------------------------------
// G-Buffer / HDR targets (screen-sized, recreated on resize)
// ---------------------------------------------------------------------------

void Renderer::createGraphResources(Context& ctx) {
    VkExtent2D e = ctx.swapchainExtent;
    auto make = [&](RTImage& rt, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect) {
        rt.format = format;
        ctx.createImage(e.width, e.height, 1, 1, 0, format, usage, rt.image, rt.memory);
        rt.view = ctx.createImageView(rt.image, format, aspect, 1);
    };
    make(gAlbedo, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    make(gNormal, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    make(gEmissive, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    make(gDepth, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    make(sceneColor, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
}

void Renderer::destroyGraphResources(Context& ctx) {
    auto destroyRT = [&](RTImage& rt) {
        if (rt.view) vkDestroyImageView(ctx.device, rt.view, nullptr);
        if (rt.image) vkDestroyImage(ctx.device, rt.image, nullptr);
        if (rt.memory) vkFreeMemory(ctx.device, rt.memory, nullptr);
        rt = {};
    };
    destroyRT(gAlbedo);
    destroyRT(gNormal);
    destroyRT(gEmissive);
    destroyRT(gDepth);
    destroyRT(sceneColor);
}

// gbuffer sample set + composite set + per-frame cluster sets.
// Called from init and again after every resize (image views change).
void Renderer::createGraphDescriptors(Context& ctx) {
    using namespace ClusterLimits;

    if (gbufferSampleSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &gbufferSampleLayout;
        VK_CHECK(vkAllocateDescriptorSets(ctx.device, &ai, &gbufferSampleSet));
    }
    {
        VkDescriptorImageInfo infos[4] = {
            { gbufferSampler, gAlbedo.view,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { gbufferSampler, gNormal.view,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { gbufferSampler, gEmissive.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { gbufferSampler, gDepth.view,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        };
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < 4; i++) {
            writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet = gbufferSampleSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &infos[i];
        }
        vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);
    }

    if (compositeSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &compositeSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(ctx.device, &ai, &compositeSet));
    }
    {
        VkDescriptorImageInfo info{ gbufferSampler, sceneColor.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = compositeSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    }

    if (skyboxSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &skyboxSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(ctx.device, &ai, &skyboxSet));
    }
    {
        VkDescriptorImageInfo info{ skybox.sampler, skybox.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = skyboxSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    }

    for (int i = 0; i < MAX_FRAMES; i++) {
        Frame& f = frames[i];
        if (f.clusterSet == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            ai.descriptorPool = descriptorPool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &clusterSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(ctx.device, &ai, &f.clusterSet));
        }
        VkDescriptorBufferInfo lb{ f.clusterLights, 0, sizeof(ClusterLight) * MAX_LIGHTS };
        VkDescriptorBufferInfo gb{ f.clusterGrid, 0, sizeof(glm::uvec2) * CLUSTER_COUNT };
        VkDescriptorBufferInfo ib{ f.clusterIndices, 0, sizeof(uint32_t) * CLUSTER_COUNT * MAX_PER_CLUSTER };
        VkWriteDescriptorSet writes[3]{};
        const VkDescriptorBufferInfo* bufs[3] = { &lb, &gb, &ib };
        for (uint32_t b = 0; b < 3; b++) {
            writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[b].dstSet = f.clusterSet;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = bufs[b];
        }
        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
    }
}

void Renderer::createPerImageSync(Context& ctx) {
    size_t n = ctx.swapchainImages.size();
    renderFinished.resize(n);
    imagesInFlight.assign(n, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (size_t i = 0; i < n; i++)
        VK_CHECK(vkCreateSemaphore(ctx.device, &sci, nullptr, &renderFinished[i]));
}

// ---------------------------------------------------------------------------
// Render graph: one-time resource registration
// ---------------------------------------------------------------------------

void Renderer::registerGraphResources(Context& ctx) {
    auto imgDesc = [](const char* name, VkExtent2D e, VkFormat fmt, VkImageAspectFlags aspect,
        uint32_t layers = 1) {
            RenderGraph::ImageDesc d{};
            d.name = name;
            d.width = e.width;
            d.height = e.height;
            d.layers = layers;
            d.format = fmt;
            d.aspect = aspect;
            return d;
        };

    VkExtent2D e = ctx.swapchainExtent;

    // -----------------------------------------------------------------------
    // Swapchain images
    // The present target is selected per frame in buildGraph(), because
    // imageIndex changes every frame.
    // -----------------------------------------------------------------------
    hSwapchain.resize(ctx.swapchainImages.size());

    for (size_t i = 0; i < ctx.swapchainImages.size(); i++) {
        hSwapchain[i] = graph.importImage(
            imgDesc("swapchain", e, ctx.swapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT),
            ctx.swapchainImages[i],
            ctx.swapchainViews[i],
            VK_IMAGE_LAYOUT_UNDEFINED
        );
    }

    // -----------------------------------------------------------------------
    // Renderer-owned render targets
    // -----------------------------------------------------------------------
    gAlbedo.rg = graph.importImage(
        imgDesc("gAlbedo", e, gAlbedo.format, VK_IMAGE_ASPECT_COLOR_BIT),
        gAlbedo.image,
        gAlbedo.view,
        VK_IMAGE_LAYOUT_UNDEFINED
    );

    gNormal.rg = graph.importImage(
        imgDesc("gNormal", e, gNormal.format, VK_IMAGE_ASPECT_COLOR_BIT),
        gNormal.image,
        gNormal.view,
        VK_IMAGE_LAYOUT_UNDEFINED
    );

    gEmissive.rg = graph.importImage(
        imgDesc("gEmissive", e, gEmissive.format, VK_IMAGE_ASPECT_COLOR_BIT),
        gEmissive.image,
        gEmissive.view,
        VK_IMAGE_LAYOUT_UNDEFINED
    );

    gDepth.rg = graph.importImage(
        imgDesc("gDepth", e, gDepth.format, VK_IMAGE_ASPECT_DEPTH_BIT),
        gDepth.image,
        gDepth.view,
        VK_IMAGE_LAYOUT_UNDEFINED
    );

    sceneColor.rg = graph.importImage(
        imgDesc("sceneColor", e, sceneColor.format, VK_IMAGE_ASPECT_COLOR_BIT),
        sceneColor.image,
        sceneColor.view,
        VK_IMAGE_LAYOUT_UNDEFINED
    );

    // -----------------------------------------------------------------------
    // Shadow maps persist in SHADER_READ_ONLY between frames
    // -----------------------------------------------------------------------
    auto shadowInitial = [](VkImageLayout& l,
        VkPipelineStageFlags2& s,
        VkAccessFlags2& a) {
            l = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            s = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            a = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        };

    VkImageLayout l;
    VkPipelineStageFlags2 s;
    VkAccessFlags2 a;

    shadowInitial(l, s, a);

    hCsmShadow = graph.importImage(
        imgDesc(
            "csmShadow",
            { CSM_RESOLUTION, CSM_RESOLUTION },
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            NUM_CASCADES
        ),
        csmShadow.image,
        csmShadow.arrayView,
        l,
        s,
        a
    );

    hSpotShadow = graph.importImage(
        imgDesc(
            "spotShadow",
            { SPOT_RESOLUTION, SPOT_RESOLUTION },
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            MAX_SHADOW_SPOT
        ),
        spotShadow.image,
        spotShadow.arrayView,
        l,
        s,
        a
    );

    hAreaShadow = graph.importImage(
        imgDesc(
            "areaShadow",
            { AREA_RESOLUTION, AREA_RESOLUTION },
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            MAX_SHADOW_AREA
        ),
        areaShadow.image,
        areaShadow.arrayView,
        l,
        s,
        a
    );

    hPointShadow = graph.importImage(
        imgDesc(
            "pointShadow",
            { POINT_RESOLUTION, POINT_RESOLUTION },
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            MAX_SHADOW_POINT * 6
        ),
        pointShadow.image,
        pointShadow.cubeArrayView,
        l,
        s,
        a
    );

    // -----------------------------------------------------------------------
    // Per-frame buffers
    // -----------------------------------------------------------------------
    for (int i = 0; i < MAX_FRAMES; i++) {
        Frame& f = frames[i];

        hFrameUBO[i] = graph.importBuffer(
            "globalUBO",
            f.ubo,
            sizeof(GlobalUBO)
        );

        hClusterLights[i] = graph.importBuffer(
            "clusterLights",
            f.clusterLights,
            sizeof(ClusterLight) * ClusterLimits::MAX_LIGHTS
        );

        hClusterGrid[i] = graph.importBuffer(
            "clusterGrid",
            f.clusterGrid,
            sizeof(glm::uvec2) * ClusterLimits::CLUSTER_COUNT
        );

        hClusterIndices[i] = graph.importBuffer(
            "clusterIndices",
            f.clusterIndices,
            sizeof(uint32_t) *
            ClusterLimits::CLUSTER_COUNT *
            ClusterLimits::MAX_PER_CLUSTER
        );
    }
}

// ---------------------------------------------------------------------------
// Shadow pass recording
// ---------------------------------------------------------------------------

void Renderer::recordDepthPass2D(VkCommandBuffer cmd, VkImageView layerView, uint32_t size,
    const glm::mat4& viewProj, Model& model) {
    VkRenderingAttachmentInfo depthA{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthA.imageView = layerView;
    depthA.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthA.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthA.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthA.clearValue.depthStencil = { 1.0f, 0 };

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, { size, size } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 0;
    ri.pDepthAttachment = &depthA;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0, 0, (float)size, (float)size, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ { 0, 0 }, { size, size } };
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdSetDepthBias(cmd, 1.5f, 0.0f, 2.0f);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow2DPipeline);
    struct { glm::mat4 vp; glm::mat4 model; } push;
    push.vp = viewProj;
    for (auto& mesh : model.meshes) {
        push.model = mesh.transform;
        vkCmdPushConstants(cmd, shadow2DPipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
            0, sizeof(push), &push);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vb, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.ib, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, (uint32_t)mesh.indices.size(), 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
}

void Renderer::recordDepthPassCubeFace(VkCommandBuffer cmd, VkImageView faceView, uint32_t size,
    int vpIndex, Model& model) {
    VkRenderingAttachmentInfo depthA{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthA.imageView = faceView;
    depthA.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthA.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthA.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthA.clearValue.depthStencil = { 1.0f, 0 };

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, { size, size } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 0;
    ri.pDepthAttachment = &depthA;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0, 0, (float)size, (float)size, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ { 0, 0 }, { size, size } };
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdSetDepthBias(cmd, 1.5f, 0.0f, 2.0f);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowCubePipeline);
    struct { glm::mat4 model; glm::ivec4 idx; } push;
    push.idx = glm::ivec4(vpIndex, 0, 0, 0);
    for (auto& mesh : model.meshes) {
        push.model = mesh.transform;
        vkCmdPushConstants(cmd, shadowCubePipeLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vb, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.ib, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, (uint32_t)mesh.indices.size(), 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
}

// Renders all shadow maps. Layout transitions are handled by the render graph.
void Renderer::recordShadowPasses(Context& ctx, VkCommandBuffer cmd, const std::vector<Model*>& models,
    const GlobalUBO& ubo, VkDescriptorSet globalSet) {
    // ---- sun CSM: one depth pass per cascade ----
    if (ubo.counts.w > 0) {
        for (int c = 0; c < ubo.counts.w; c++) {
            for (Model* model : models) {
                if (!model) continue;

                recordDepthPass2D(cmd, csmShadow.layerViews[c], csmShadow.size, ubo.sunViewProj[c], *model);
            }
        }
    }

    // ---- spot lights ----
    {
        int si = 0;

        for (size_t i = 0;
            i < lights.spotLights.size() && i < (size_t)MAX_SPOT_LIGHTS && si < MAX_SHADOW_SPOT;
            i++) {

            if (!lights.spotLights[i].castShadow)
                continue;

            for (Model* model : models) {
                if (!model) continue;

                recordDepthPass2D(cmd, spotShadow.layerViews[si], spotShadow.size, ubo.spotViewProj[si], *model);
            }

            si++;
        }
    }

    // ---- area lights ----
    {
        int ai = 0;

        for (size_t i = 0;
            i < lights.areaLights.size() &&
            i < (size_t)MAX_AREA_LIGHTS &&
            ai < MAX_SHADOW_AREA;
            i++) {

            if (!lights.areaLights[i].castShadow)
                continue;

            for (Model* model : models) {
                if (!model) continue;

                recordDepthPass2D(cmd, areaShadow.layerViews[ai], areaShadow.size, ubo.areaViewProj[ai], *model);
            }

            ai++;
        }
    }

    // ---- point lights: 6 cube faces per shadow-casting light ----
    {
        int pi = 0;

        for (size_t i = 0;
            i < lights.pointLights.size() && i < (size_t)MAX_POINT_LIGHTS && pi < MAX_SHADOW_POINT;
            i++) {

            if (!lights.pointLights[i].castShadow)
                continue;

            if (pi == 0) {
                // cube vertex/fragment shaders read face matrices from the global UBO
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowCubePipeLayout, 0, 1, &globalSet, 0, nullptr);
            }

            for (int f = 0; f < 6; f++) {
                for (Model* model : models) {
                    if (!model) continue;

                    recordDepthPassCubeFace(cmd, pointShadow.faceViews[pi * 6 + f], pointShadow.size, pi * 6 + f, *model);
                }
            }

            pi++;
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry drawing helpers
// ---------------------------------------------------------------------------

// Draws every opaque mesh whose per-mesh render mode resolves to `mode`.
// Transparent meshes are skipped (they belong to the transparent pass).
void Renderer::drawMeshesForMode(VkCommandBuffer cmd, VkPipelineLayout layout,
    const std::vector<Model*>& models, RenderMode mode) {
    for (Model* model : models) {
        if (!model) continue;
        for (auto& mesh : model->meshes) {
            Material& mat = model->materials[mesh.materialIndex];
            if (mat.transparent) continue;
            if (mesh.renderMode != mode) continue;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                1, 1, &mat.set, 0, nullptr);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT,
                0, sizeof(glm::mat4), &mesh.transform);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vb, &offset);
            vkCmdBindIndexBuffer(cmd, mesh.ib, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, (uint32_t)mesh.indices.size(), 1, 0, 0, 0);
        }
    }
}

// Draws only opaque (transparent=false) or only transparent (true) meshes.
void Renderer::drawMeshGeometryFiltered(VkCommandBuffer cmd, VkPipelineLayout layout, Model& model,
    bool transparent) {
    for (auto& mesh : model.meshes) {
        Material& mat = model.materials[mesh.materialIndex];
        if (mat.transparent != transparent) continue;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
            1, 1, &mat.set, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT,
            0, sizeof(glm::mat4), &mesh.transform);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vb, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.ib, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, (uint32_t)mesh.indices.size(), 1, 0, 0, 0);
    }
}

// ---------------------------------------------------------------------------
// Graph pass record callbacks
// ---------------------------------------------------------------------------

void Renderer::recordGBufferPass(VkCommandBuffer cmd, const std::vector<Model*>& models,
    VkDescriptorSet globalSet, VkExtent2D extent) {

    VkRenderingAttachmentInfo cols[3]{};
    VkImageView views[3] = { gAlbedo.view, gNormal.view, gEmissive.view };
    for (int i = 0; i < 3; i++) {
        cols[i] = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        cols[i].imageView = views[i];
        cols[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        cols[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        cols[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        cols[i].clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    }
    VkRenderingAttachmentInfo depthA{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthA.imageView = gDepth.view;
    depthA.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthA.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthA.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthA.clearValue.depthStencil = { 1.0f, 0 };

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, extent };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 3;
    ri.pColorAttachments = cols;
    ri.pDepthAttachment = &depthA;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0, 0, (float)extent.width, (float)extent.height, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ { 0, 0 }, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gbufferPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gbufferPipeLayout,
        0, 1, &globalSet, 0, nullptr);
    drawMeshesForMode(cmd, gbufferPipeLayout, models, RenderMode::Deferred);

    vkCmdEndRendering(cmd);
}

void Renderer::recordDeferredLightingPass(VkCommandBuffer cmd, VkDescriptorSet globalSet,
    const glm::mat4& invViewProj, VkExtent2D extent) {
    VkRenderingAttachmentInfo color{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    color.imageView = sceneColor.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, extent };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0, 0, (float)extent.width, (float)extent.height, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ { 0, 0 }, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredPipeLayout,
        0, 1, &globalSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredPipeLayout,
        1, 1, &gbufferSampleSet, 0, nullptr);

    struct { glm::mat4 invViewProj; glm::vec4 screen; } push;
    push.invViewProj = invViewProj;
    push.screen = glm::vec4((float)extent.width, (float)extent.height,
        1.0f / extent.width, 1.0f / extent.height);
    vkCmdPushConstants(cmd, deferredPipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);

    vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle
    vkCmdEndRendering(cmd);
}

void Renderer::recordSkyboxPass(VkCommandBuffer cmd, VkDescriptorSet globalSet, VkExtent2D extent) {
    VkRenderingAttachmentInfo color{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    color.imageView = sceneColor.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depth{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depth.imageView = gDepth.view;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, extent };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depth;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0, 0, (float)extent.width, (float)extent.height, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ { 0, 0 }, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline);
    VkDescriptorSet sets[2] = { globalSet, skyboxSet };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeLayout,
        0, 2, sets, 0, nullptr);
    vkCmdDraw(cmd, 36, 1, 0, 0);

    vkCmdEndRendering(cmd);
}

void Renderer::recordClusterBuildPass(VkCommandBuffer cmd, Frame& frame, const Camera& camera,
    VkExtent2D extent) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clusterPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clusterPipeLayout,
        0, 1, &frame.clusterSet, 0, nullptr);
    ClusterParams params = makeClusterParams(camera, extent);
    vkCmdPushConstants(cmd, clusterPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(ClusterParams), &params);
    vkCmdDispatch(cmd, ClusterLimits::GRID_X, ClusterLimits::GRID_Y, ClusterLimits::GRID_Z);
}

// Classic forward meshes (RenderMode::Forward): the full UBO light rig in a
// single shader (limited to LightLimits light counts), rendered linear into
// the HDR target on top of the deferred result, sharing the G-Buffer depth.
void Renderer::recordForwardPass(Context& ctx, VkCommandBuffer cmd,
    const std::vector<Model*>& models, Frame& frame, VkExtent2D extent) {
    for (Model* model : models) {
        if (!model) continue;
        for (const Mesh& mesh : model->meshes) {
            const Material& material = model->materials[mesh.materialIndex];
            if (!material.transparent && mesh.renderMode == RenderMode::Custom)
                (void)getCustomForwardPipeline(ctx, mesh);
        }
    }
    VkRenderingAttachmentInfo color{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    color.imageView = sceneColor.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // on top of the deferred result
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthA{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthA.imageView = gDepth.view;
    depthA.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthA.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // shared G-Buffer depth
    depthA.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, extent };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depthA;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0, 0, (float)extent.width, (float)extent.height, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ { 0, 0 }, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineHDR);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeLayout,
        0, 1, &frame.globalSet, 0, nullptr);

    drawMeshesForMode(cmd, meshPipeLayout, models, RenderMode::Forward);

    for (Model* model : models) {
        if (!model) continue;
        for (auto& mesh : model->meshes) {
            Material& material = model->materials[mesh.materialIndex];
            if (material.transparent || mesh.renderMode != RenderMode::Custom) continue;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                getCustomForwardPipeline(ctx, mesh));
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeLayout,
                1, 1, &material.set, 0, nullptr);
            vkCmdPushConstants(cmd, meshPipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                0, sizeof(glm::mat4), &mesh.transform);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vb, &offset);
            vkCmdBindIndexBuffer(cmd, mesh.ib, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, (uint32_t)mesh.indices.size(), 1, 0, 0, 0);
        }
    }

    vkCmdEndRendering(cmd);
}

// Hybrid only: transparent surfaces, forward-shaded on top of the deferred
// result, testing against the G-Buffer depth (read-only).
void Renderer::recordTransparentPass(VkCommandBuffer cmd, const std::vector<Model*>& models,
    Frame& frame, const Camera& camera, VkExtent2D extent) {
    VkRenderingAttachmentInfo color{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    color.imageView = sceneColor.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // on top of the deferred result
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthA{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthA.imageView = gDepth.view;
    depthA.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    depthA.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthA.storeOp = VK_ATTACHMENT_STORE_OP_NONE;

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, extent };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depthA;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0, 0, (float)extent.width, (float)extent.height, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ { 0, 0 }, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardPlusBlendPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardPlusPipeLayout,
        0, 1, &frame.globalSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardPlusPipeLayout,
        2, 1, &frame.clusterSet, 0, nullptr);

    ClusterParams params = makeClusterParams(camera, extent);
    vkCmdPushConstants(cmd, forwardPlusPipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
        sizeof(glm::mat4), sizeof(ClusterParams) - sizeof(glm::mat4), &params.screen);

    for (Model* model : models)
        if (model) drawMeshGeometryFiltered(cmd, forwardPlusPipeLayout, *model, true);

    vkCmdEndRendering(cmd);
}

void Renderer::recordCompositePass(VkCommandBuffer cmd, uint32_t imageIndex, VkExtent2D extent) {
    VkRenderingAttachmentInfo color{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    color.imageView = imageIndex < hSwapchain.size() ? graph.image(hSwapchain[imageIndex]).view : VK_NULL_HANDLE;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, extent };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0, 0, (float)extent.width, (float)extent.height, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ { 0, 0 }, extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeLayout,
        0, 1, &compositeSet, 0, nullptr);

    // x = exposure, y = tonemap operator (0 = none, 1 = Reinhard, 2 = ACES)
    glm::vec4 params(exposure, (float)tonemapOperator, 0.0f, 0.0f);
    vkCmdPushConstants(cmd, compositePipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(glm::vec4), &params);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);
}

// ---------------------------------------------------------------------------
// Clustered lighting CPU side
// ---------------------------------------------------------------------------

ClusterParams Renderer::makeClusterParams(const Camera& camera, VkExtent2D extent) const {
    using namespace ClusterLimits;
    ClusterParams p{};
    p.view = camera.view();
    p.screen = glm::vec4((float)extent.width, (float)extent.height,
        extent.width / (float)GRID_X, extent.height / (float)GRID_Y);
    float tanY = tanf(glm::radians(camera.fov) * 0.5f);
    float tanX = tanY * (extent.width / (float)extent.height);
    p.zparams = glm::vec4(camera.nearClip, camera.farClip, tanX, tanY);
    p.dims = glm::ivec4(GRID_X, GRID_Y, GRID_Z, (int)clusterLightCount);
    return p;
}

void Renderer::updateClusterLights(Frame& frame) {
    clusterLightCount = buildClusterLights(lights,
        (ClusterLight*)frame.clusterLightsMapped, ClusterLimits::MAX_LIGHTS);
}

// ---------------------------------------------------------------------------
// Render graph assembly: declares WHAT renders, in which ORDER, with WHICH
// resources. Sorting, culling and all barriers are handled by the graph.
// ---------------------------------------------------------------------------

void Renderer::buildGraph(Context& ctx, Frame& frame, uint32_t imageIndex,
    const std::vector<Model*>& models, const GlobalUBO& ubo, const Camera& camera) {
    using U = RenderGraph::Usage;
    VkExtent2D extent = ctx.swapchainExtent;

    int fi = (int)(&frame - &frames[0]);
    const glm::mat4 invViewProj = glm::inverse(ubo.proj * ubo.view);

    graph.beginFrame();

    // The acquired swapchain image is the present target for this frame.
    RenderGraph::Handle swap = hSwapchain[imageIndex];
    graph.setPresentTarget(swap);

    // ---- shadows (all four maps in one pass; unused maps are left alone) ----
    graph.addPass("Shadows")
        .writeImage(hCsmShadow, U::DepthAttachmentWrite)
        .writeImage(hSpotShadow, U::DepthAttachmentWrite)
        .writeImage(hAreaShadow, U::DepthAttachmentWrite)
        .writeImage(hPointShadow, U::DepthAttachmentWrite)
        .record([&](VkCommandBuffer cmd) {
        recordShadowPasses(ctx, cmd, models, ubo, frame.globalSet);
            });

    // Which render paths does this frame's scene actually use? (per-mesh modes)
    const ScenePathUsage paths = classifyScene(models);

    // ---- cluster light assignment (compute) ----
    if (paths.needsClusters()) {
        graph.addPass("ClusterBuild")
            .readBuffer(hClusterLights[fi], U::StorageReadCompute)
            .writeBuffer(hClusterGrid[fi], U::StorageWriteCompute)
            .writeBuffer(hClusterIndices[fi], U::StorageWriteCompute)
            .record([&](VkCommandBuffer cmd) {
            recordClusterBuildPass(cmd, frame, camera, extent);
                });
    }

    // ---- G-Buffer + deferred lighting ----
    // Always declared: with no Deferred meshes the geometry pass simply draws
    // nothing, while the lighting pass still clears sceneColor, draws the
    // skybox background and provides the shared depth buffer for the forward
    // passes.
    {
        graph.addPass("GBuffer")
            .writeImage(gAlbedo.rg, U::ColorAttachmentWrite)
            .writeImage(gNormal.rg, U::ColorAttachmentWrite)
            .writeImage(gEmissive.rg, U::ColorAttachmentWrite)
            .writeImage(gDepth.rg, U::DepthAttachmentWrite)
            .readBuffer(hFrameUBO[fi], U::UniformRead)
            .record([&, extent](VkCommandBuffer cmd) {
            recordGBufferPass(cmd, models, frame.globalSet, extent);
                });

        graph.addPass("DeferredLighting")
            .readImage(gAlbedo.rg, U::SampledFragment)
            .readImage(gNormal.rg, U::SampledFragment)
            .readImage(gEmissive.rg, U::SampledFragment)
            .readImage(gDepth.rg, U::SampledFragment)
            .readImage(hCsmShadow, U::SampledFragment)
            .readImage(hSpotShadow, U::SampledFragment)
            .readImage(hAreaShadow, U::SampledFragment)
            .readImage(hPointShadow, U::SampledFragment)
            .readBuffer(hFrameUBO[fi], U::UniformRead)
            .writeImage(sceneColor.rg, U::ColorAttachmentWrite)
            .record([&, extent](VkCommandBuffer cmd) {
            recordDeferredLightingPass(cmd, frame.globalSet, invViewProj, extent);
                });
    }

    // ---- dedicated skybox background ----
    graph.addPass("Skybox")
        .readImage(gDepth.rg, U::DepthAttachmentRead)
        .writeImage(sceneColor.rg, U::ColorAttachmentWrite)
        .record([&, extent](VkCommandBuffer cmd) {
            recordSkyboxPass(cmd, frame.globalSet, extent);
        });

    // ---- classic and custom forward meshes (on top of deferred) ----
    if (paths.forward || paths.custom) {
        graph.addPass("Forward")
            .writeImage(sceneColor.rg, U::ColorAttachmentWrite)
            .writeImage(gDepth.rg, U::DepthAttachmentWrite) // test + write, loaded
            .readImage(hCsmShadow, U::SampledFragment)
            .readImage(hSpotShadow, U::SampledFragment)
            .readImage(hAreaShadow, U::SampledFragment)
            .readImage(hPointShadow, U::SampledFragment)
            .readBuffer(hFrameUBO[fi], U::UniformRead)
            .record([&, extent](VkCommandBuffer cmd) {
            recordForwardPass(ctx, cmd, models, frame, extent);
                });
    }

    // ---- transparent meshes (clustered forward, blended, depth read-only) ----
    if (paths.transparent) {
        graph.addPass("ForwardTransparent")
            .writeImage(sceneColor.rg, U::ColorAttachmentWrite)
            .readImage(gDepth.rg, U::DepthAttachmentRead)
            .readImage(hCsmShadow, U::SampledFragment)
            .readImage(hSpotShadow, U::SampledFragment)
            .readImage(hAreaShadow, U::SampledFragment)
            .readImage(hPointShadow, U::SampledFragment)
            .readBuffer(hFrameUBO[fi], U::UniformRead)
            .readBuffer(hClusterLights[fi], U::StorageReadFragment)
            .readBuffer(hClusterGrid[fi], U::StorageReadFragment)
            .readBuffer(hClusterIndices[fi], U::StorageReadFragment)
            .record([&, extent](VkCommandBuffer cmd) {
            recordTransparentPass(cmd, models, frame, camera, extent);
                });
    }

    // ---- composite to the swapchain (drives the whole graph) ----
    graph.addPass("Composite")
        .readImage(sceneColor.rg, U::SampledFragment)
        .writeImage(swap, U::ColorAttachmentWrite)
        .sideEffect()
        .record([&, extent, imageIndex](VkCommandBuffer cmd) {
        recordCompositePass(cmd, imageIndex, extent);
            });
}

void Renderer::recordGraphFrame(Context& ctx, Frame& frame, uint32_t imageIndex,
    const std::vector<Model*>& models, const GlobalUBO& ubo, const Camera& camera) {
    VkCommandBuffer cmd = frame.cmd;
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    buildGraph(ctx, frame, imageIndex, models, ubo, camera);
    graph.compile(ctx);   // topological sort + culling + transient creation
    graph.execute(cmd);   // automatic barriers + pass record callbacks

    VK_CHECK(vkEndCommandBuffer(cmd));
}

// ---------------------------------------------------------------------------
// Resize / frame / shutdown
// ---------------------------------------------------------------------------

void Renderer::recreate(Context& ctx) {
    ctx.recreateSwapchain();
    for (auto s : renderFinished) vkDestroySemaphore(ctx.device, s, nullptr);
    createPerImageSync(ctx);

    // size-dependent graph resources
    destroyGraphResources(ctx);
    createGraphResources(ctx);
    createGraphDescriptors(ctx); // re-point gbuffer/composite samplers

    // re-import resized images (tracked state resets to UNDEFINED)
    for (size_t i = 0; i < ctx.swapchainImages.size() && i < hSwapchain.size(); i++)
        graph.updateImportedImage(hSwapchain[i], ctx.swapchainImages[i], ctx.swapchainViews[i],
            ctx.swapchainExtent.width, ctx.swapchainExtent.height);
    auto reimport = [&](RenderGraph::Handle h, RTImage& rt) {
        graph.updateImportedImage(h, rt.image, rt.view,
            ctx.swapchainExtent.width, ctx.swapchainExtent.height);
    };
    reimport(gAlbedo.rg, gAlbedo);
    reimport(gNormal.rg, gNormal);
    reimport(gEmissive.rg, gEmissive);
    reimport(gDepth.rg, gDepth);
    reimport(sceneColor.rg, sceneColor);

    framebufferResized = false;
}

void Renderer::drawFrame(Context& ctx, Camera& camera, const std::vector<Model*>& models)
{
    Frame& f = frames[currentFrame];

    // wait until this frame slot is no longer in use
    vkWaitForFences(ctx.device, 1, &f.inFlight, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;

    VkResult r = vkAcquireNextImageKHR(ctx.device, ctx.swapchain, UINT64_MAX, f.imageAvailable, VK_NULL_HANDLE, &imageIndex);

    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate(ctx);
        return;
    }

    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR)
        VK_CHECK(r);

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(ctx.device, 1, &imagesInFlight[imageIndex], VK_TRUE, INT64_MAX);

    imagesInFlight[imageIndex] = f.inFlight;

    // update global UBO (camera + lights + shadow matrices)
    float aspect = ctx.swapchainExtent.width / (float)ctx.swapchainExtent.height;

    GlobalUBO ubo{};

    ubo.view = camera.view();
    ubo.proj = camera.proj(aspect);
    ubo.camPos = glm::vec4(camera.position, 1.0f);

    float envMaxLod = skybox.mipLevels > 0 ? float(skybox.mipLevels - 1) : 0.0f;

    lights.fillGlobalUBO(
        ubo,
        camera,
        aspect,
        envMaxLod
    );

    std::memcpy(f.uboMapped, &ubo, sizeof(ubo));

    // clustered paths: upload the compacted light list for this frame slot
    // (needed when any mesh is Forward+ or transparent)
    if (classifyScene(models).needsClusters())
        updateClusterLights(f);

    vkResetFences(ctx.device, 1, &f.inFlight);

    vkResetCommandBuffer(f.cmd, 0);

    recordGraphFrame(ctx, f, imageIndex, models, ubo, camera);

    // submit command buffer
    VkPipelineStageFlags waitStage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo si{
        VK_STRUCTURE_TYPE_SUBMIT_INFO
    };

    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &f.imageAvailable;
    si.pWaitDstStageMask = &waitStage;

    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cmd;

    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores =
        &renderFinished[imageIndex];

    VK_CHECK(
        vkQueueSubmit(
            ctx.queue,
            1,
            &si,
            f.inFlight
        )
    );

    // present rendered image
    VkPresentInfoKHR pi{
        VK_STRUCTURE_TYPE_PRESENT_INFO_KHR
    };

    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores =
        &renderFinished[imageIndex];

    pi.swapchainCount = 1;
    pi.pSwapchains = &ctx.swapchain;
    pi.pImageIndices = &imageIndex;

    r = vkQueuePresentKHR(
        ctx.queue,
        &pi
    );

    if (r == VK_ERROR_OUT_OF_DATE_KHR ||
        r == VK_SUBOPTIMAL_KHR ||
        framebufferResized)
    {
        recreate(ctx);
    }
    else if (r != VK_SUCCESS)
    {
        VK_CHECK(r);
    }

    currentFrame =
        (currentFrame + 1) % MAX_FRAMES;
}

void Renderer::shutdown(Context& ctx) {
    graph.shutdown(ctx); // destroys graph-owned transients (none imported are touched)

    auto destroyMap2D = [&](ShadowMap2D& m) {
        for (auto v : m.layerViews) vkDestroyImageView(ctx.device, v, nullptr);
        vkDestroyImageView(ctx.device, m.arrayView, nullptr);
        vkDestroyImage(ctx.device, m.image, nullptr);
        vkFreeMemory(ctx.device, m.memory, nullptr);
    };
    destroyMap2D(csmShadow);
    destroyMap2D(spotShadow);
    destroyMap2D(areaShadow);
    for (auto v : pointShadow.faceViews) vkDestroyImageView(ctx.device, v, nullptr);
    vkDestroyImageView(ctx.device, pointShadow.cubeArrayView, nullptr);
    vkDestroyImage(ctx.device, pointShadow.image, nullptr);
    vkFreeMemory(ctx.device, pointShadow.memory, nullptr);
    vkDestroySampler(ctx.device, shadowSampler, nullptr);
    vkDestroySampler(ctx.device, gbufferSampler, nullptr);

    destroyGraphResources(ctx);

    for (auto s : renderFinished) vkDestroySemaphore(ctx.device, s, nullptr);
    for (auto& f : frames) {
        vkDestroySemaphore(ctx.device, f.imageAvailable, nullptr);
        vkDestroyFence(ctx.device, f.inFlight, nullptr);
        vkDestroyBuffer(ctx.device, f.ubo, nullptr);
        vkFreeMemory(ctx.device, f.uboMem, nullptr);
        vkDestroyBuffer(ctx.device, f.clusterLights, nullptr);
        vkFreeMemory(ctx.device, f.clusterLightsMem, nullptr);
        vkDestroyBuffer(ctx.device, f.clusterGrid, nullptr);
        vkFreeMemory(ctx.device, f.clusterGridMem, nullptr);
        vkDestroyBuffer(ctx.device, f.clusterIndices, nullptr);
        vkFreeMemory(ctx.device, f.clusterIndicesMem, nullptr);
    }
    skybox.destroy(ctx);
    vkDestroyPipeline(ctx.device, meshPipelineHDR, nullptr);
    vkDestroyPipeline(ctx.device, shadow2DPipeline, nullptr);
    vkDestroyPipeline(ctx.device, shadowCubePipeline, nullptr);
    vkDestroyPipeline(ctx.device, gbufferPipeline, nullptr);
    vkDestroyPipeline(ctx.device, deferredPipeline, nullptr);
    vkDestroyPipeline(ctx.device, skyboxPipeline, nullptr);
    for (const auto& entry : customForwardPipelines)
        vkDestroyPipeline(ctx.device, entry.second, nullptr);
    customForwardPipelines.clear();
    vkDestroyPipeline(ctx.device, forwardPlusBlendPipeline, nullptr);
    vkDestroyPipeline(ctx.device, compositePipeline, nullptr);
    vkDestroyPipeline(ctx.device, clusterPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, meshPipeLayout, nullptr); // == gbufferPipeLayout
    vkDestroyPipelineLayout(ctx.device, shadow2DPipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, shadowCubePipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, deferredPipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, forwardPlusPipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, compositePipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, skyboxPipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, clusterPipeLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, globalLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, materialLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, gbufferSampleLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, compositeSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, skyboxSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, clusterSetLayout, nullptr);
}
