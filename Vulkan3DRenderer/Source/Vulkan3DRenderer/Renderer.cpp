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

void Renderer::init(Context& ctx, const std::array<std::string, 6>& skyboxFaces) {
    createLayouts(ctx);

    VkDescriptorPoolSize sizes[2]{};
    sizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 512 };
    sizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024 };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = 1024;
    pi.poolSizeCount = 2;
    pi.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descriptorPool));

    createSkyboxResources(ctx, skyboxFaces);
    createShadowResources(ctx);
    createFrames(ctx);
    createPipelines(ctx);
    createPerImageSync(ctx);
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
    {   // set 1 (skybox): cubemap sampler only
        VkDescriptorSetLayoutBinding b{};
        b = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 1;
        ci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &skyboxSetLayout));
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
    {   // skybox pipeline layout: set0 global, set1 cubemap
        VkDescriptorSetLayout sets[2] = { globalLayout, skyboxSetLayout };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 2;
        ci.pSetLayouts = sets;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &skyboxPipeLayout));
    }
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
}

VkPipeline Renderer::createPipeline(Context& ctx, const std::string& vertSpv,
    const std::string& fragSpv, VkPipelineLayout layout, bool isSkybox) {
    auto vertCode = readFile(vertSpv);
    auto fragCode = readFile(fragSpv);
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
    if (!isSkybox) {
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
    raster.cullMode = isSkybox ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaa{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = isSkybox ? VK_FALSE : VK_TRUE;
    depth.depthCompareOp = isSkybox ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    VkDynamicState dynStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynStates;

    // dynamic rendering (core in Vulkan 1.3+): no render pass objects
    VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &ctx.swapchainFormat;
    rendering.depthAttachmentFormat = ctx.depthFormat;

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
    ci.layout = layout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline));
    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
    return pipeline;
}

VkPipeline Renderer::createShadowPipeline(Context& ctx, const std::string& vertSpv,
    const std::string& fragSpv, VkPipelineLayout layout) {
    auto vertCode = readFile(vertSpv);
    auto fragCode = readFile(fragSpv);
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
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    vertexInput.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_BACK_BIT;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    raster.depthBiasEnable = VK_TRUE; // dynamic depth bias set per pass

    VkPipelineMultisampleStateCreateInfo msaa{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendStateCreateInfo blend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend.attachmentCount = 0; // depth-only

    VkDynamicState dynStates[3] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                    VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynamic{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic.dynamicStateCount = 3;
    dynamic.pDynamicStates = dynStates;

    VkFormat shadowFormat = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering.colorAttachmentCount = 0;
    rendering.depthAttachmentFormat = shadowFormat;

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
    ci.layout = layout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline));
    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
    return pipeline;
}

void Renderer::createPipelines(Context& ctx) {
    meshPipeline = createPipeline(ctx, "Assets/Shaders/mesh.vert.spv", "Assets/Shaders/mesh.frag.spv", meshPipeLayout, false);
    skyboxPipeline = createPipeline(ctx, "Assets/Shaders/skybox.vert.spv", "Assets/Shaders/skybox.frag.spv", skyboxPipeLayout, true);
    shadow2DPipeline = createShadowPipeline(ctx, "Assets/Shaders/shadow2d.vert.spv",
        "Assets/Shaders/shadow2d.frag.spv", shadow2DPipeLayout);
    createShadowCubePipeline(ctx);
}

// Cube shadow pipeline: vertex + fragment (fragment writes radial distance to depth).
void Renderer::createShadowCubePipeline(Context& ctx) {
    auto vertCode = readFile("Assets/Shaders/shadow_cube.vert.spv");
    auto fragCode = readFile("Assets/Shaders/shadow_cube.frag.spv");
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
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    vertexInput.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_BACK_BIT;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    raster.depthBiasEnable = VK_TRUE;

    VkPipelineMultisampleStateCreateInfo msaa{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendStateCreateInfo blend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend.attachmentCount = 0;

    VkDynamicState dynStates[3] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                    VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynamic{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic.dynamicStateCount = 3;
    dynamic.pDynamicStates = dynStates;

    VkFormat shadowFormat = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering.colorAttachmentCount = 0;
    rendering.depthAttachmentFormat = shadowFormat;

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
    ci.layout = shadowCubePipeLayout;

    VK_CHECK(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &ci, nullptr, &shadowCubePipeline));
    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
}

void Renderer::createSkyboxResources(Context& ctx, const std::array<std::string, 6>& faces) {
    try {
        skybox.createFromFiles(ctx, faces);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\nUsing fallback gray skybox.\n";
        skybox.createSolid(ctx, 60, 70, 90, 255);
    }
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = descriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &skyboxSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &ai, &skyboxSet));

    VkDescriptorImageInfo ii{ skybox.sampler, skybox.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = skyboxSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &ii;
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
}

// ---------------------------------------------------------------------------
// Shadow map resources
// ---------------------------------------------------------------------------

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
}

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

void Renderer::createPerImageSync(Context& ctx) {
    size_t n = ctx.swapchainImages.size();
    renderFinished.resize(n);
    imagesInFlight.assign(n, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (size_t i = 0; i < n; i++)
        VK_CHECK(vkCreateSemaphore(ctx.device, &sci, nullptr, &renderFinished[i]));
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

void Renderer::recordShadowPasses(Context& ctx, VkCommandBuffer cmd, const std::vector<Model*>& models,
    const GlobalUBO& ubo, VkDescriptorSet globalSet) {
    const VkImageAspectFlags depth = VK_IMAGE_ASPECT_DEPTH_BIT;

    auto toAttach = [&](VkImage img, uint32_t layers) {
        Context::cmdImageBarrier(cmd, img,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            depth, 1, layers);
        };

    auto toRead = [&](VkImage img, uint32_t layers) {
        Context::cmdImageBarrier(cmd, img,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            depth, 1, layers);
        };

    // ---- sun CSM: one depth pass per cascade ----
    if (ubo.counts.w > 0) {
        toAttach(csmShadow.image, NUM_CASCADES);

        for (int c = 0; c < ubo.counts.w; c++) {
            for (Model* model : models) {
                if (!model) continue;

                recordDepthPass2D(cmd, csmShadow.layerViews[c], csmShadow.size, ubo.sunViewProj[c], *model);
            }
        }

        toRead(csmShadow.image, NUM_CASCADES);
    }

    // ---- spot lights ----
    {
        int si = 0;

        for (size_t i = 0;
            i < lights.spotLights.size() && i < (size_t)MAX_SPOT_LIGHTS && si < MAX_SHADOW_SPOT;
            i++) {

            if (!lights.spotLights[i].castShadow)
                continue;

            if (si == 0)
                toAttach(spotShadow.image, MAX_SHADOW_SPOT);

            for (Model* model : models) {
                if (!model) continue;

                recordDepthPass2D(cmd, spotShadow.layerViews[si], spotShadow.size, ubo.spotViewProj[si], *model);
            }

            si++;
        }

        if (si > 0)
            toRead(spotShadow.image, MAX_SHADOW_SPOT);
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

            if (ai == 0)
                toAttach(areaShadow.image, MAX_SHADOW_AREA);

            for (Model* model : models) {
                if (!model) continue;

                recordDepthPass2D(cmd, areaShadow.layerViews[ai], areaShadow.size, ubo.areaViewProj[ai], *model);
            }

            ai++;
        }

        if (ai > 0)
            toRead(areaShadow.image, MAX_SHADOW_AREA);
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
                toAttach(pointShadow.image,MAX_SHADOW_POINT * 6);

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

        if (pi > 0)
            toRead(pointShadow.image, MAX_SHADOW_POINT * 6);
    }
}

// ---------------------------------------------------------------------------

void Renderer::drawMeshGeometry(VkCommandBuffer cmd, VkPipelineLayout layout, Model& model) {
    for (auto& mesh : model.meshes) {
        Material& mat = model.materials[mesh.materialIndex];
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

void Renderer::recordFrame(Context& ctx, Frame& frame, uint32_t imageIndex, const std::vector<Model*>& models,
    const GlobalUBO& ubo) {
    VkCommandBuffer cmd = frame.cmd;
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    // ---- shadow passes (before the main pass so maps are readable) ----
    recordShadowPasses(ctx, cmd, models, ubo, frame.globalSet);

    // swapchain image -> color attachment
    Context::cmdImageBarrier(cmd, ctx.swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo color{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    color.imageView = ctx.swapchainViews[imageIndex];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = { { 0.02f, 0.03f, 0.05f, 1.0f } };

    VkRenderingAttachmentInfo depthA{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthA.imageView = ctx.depthView;
    depthA.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthA.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthA.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthA.clearValue.depthStencil = { 1.0f, 0 };

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, ctx.swapchainExtent };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depthA;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0, 0, (float)ctx.swapchainExtent.width, (float)ctx.swapchainExtent.height, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ { 0, 0 }, ctx.swapchainExtent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // --- skybox ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeLayout,
        0, 1, &frame.globalSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeLayout,
        1, 1, &skyboxSet, 0, nullptr);
    vkCmdDraw(cmd, 36, 1, 0, 0);

    // --- meshes (PBR + all lights + shadows) ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeLayout,
        0, 1, &frame.globalSet, 0, nullptr);
    
    // draw every loaded model
    for (Model* model : models) {
        if (!model)
            continue;

        drawMeshGeometry(cmd, meshPipeLayout, *model);
    }

    vkCmdEndRendering(cmd);

    Context::cmdImageBarrier(cmd, ctx.swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE, 0);

    VK_CHECK(vkEndCommandBuffer(cmd));
}

void Renderer::recreate(Context& ctx) {
    ctx.recreateSwapchain();
    for (auto s : renderFinished) vkDestroySemaphore(ctx.device, s, nullptr);
    createPerImageSync(ctx);
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

    vkResetFences(ctx.device, 1, &f.inFlight);

    vkResetCommandBuffer(f.cmd, 0);

    recordFrame(ctx, f, imageIndex, models, ubo);

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

    for (auto s : renderFinished) vkDestroySemaphore(ctx.device, s, nullptr);
    for (auto& f : frames) {
        vkDestroySemaphore(ctx.device, f.imageAvailable, nullptr);
        vkDestroyFence(ctx.device, f.inFlight, nullptr);
        vkDestroyBuffer(ctx.device, f.ubo, nullptr);
        vkFreeMemory(ctx.device, f.uboMem, nullptr);
    }
    skybox.destroy(ctx);
    vkDestroyPipeline(ctx.device, meshPipeline, nullptr);
    vkDestroyPipeline(ctx.device, skyboxPipeline, nullptr);
    vkDestroyPipeline(ctx.device, shadow2DPipeline, nullptr);
    vkDestroyPipeline(ctx.device, shadowCubePipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, meshPipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, skyboxPipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, shadow2DPipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, shadowCubePipeLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, globalLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, materialLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, skyboxSetLayout, nullptr);
}
