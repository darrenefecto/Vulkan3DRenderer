#include <Vulkan3DRenderer/Renderer.h>
#include <fstream>
#include <iostream>
#include <cstring>

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
    sizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256 };
    sizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = 512;
    pi.poolSizeCount = 2;
    pi.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descriptorPool));

    createSkyboxResources(ctx, skyboxFaces);
    createFrames(ctx);
    createPipelines(ctx);
    createPerImageSync(ctx);
}

void Renderer::createLayouts(Context& ctx) {
    {   // set 0: per-frame global UBO
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 1;
        ci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &globalLayout));
    }
    {   // set 1: material UBO + albedo sampler
        VkDescriptorSetLayoutBinding b[2]{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        b[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 2;
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

    VkPushConstantRange push{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) };
    {   // mesh pipeline layout: set0 global, set1 material, push model
        VkDescriptorSetLayout sets[2] = { globalLayout, materialLayout };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 2;
        ci.pSetLayouts = sets;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &meshPipeLayout));
    }
    {   // skybox pipeline layout: set0 global, set1 cubemap
        VkDescriptorSetLayout sets[2] = { globalLayout, skyboxSetLayout };
        VkPipelineLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        ci.setLayoutCount = 2;
        ci.pSetLayouts = sets;
        VK_CHECK(vkCreatePipelineLayout(ctx.device, &ci, nullptr, &skyboxPipeLayout));
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

void Renderer::createPipelines(Context& ctx) {
    meshPipeline = createPipeline(ctx, "Assets/Shaders/mesh.vert.spv", "Assets/Shaders/mesh.frag.spv", meshPipeLayout, false);
    skyboxPipeline = createPipeline(ctx, "Assets/Shaders/skybox.vert.spv", "Assets/Shaders/skybox.frag.spv", skyboxPipeLayout, true);
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
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = f.globalSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &bi;
        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
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

void Renderer::recordFrame(Context& ctx, Frame& frame, uint32_t imageIndex, Model& model) {
    VkCommandBuffer cmd = frame.cmd;
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

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

    // --- meshes ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeLayout,
        0, 1, &frame.globalSet, 0, nullptr);
    for (auto& mesh : model.meshes) {
        Material& mat = model.materials[mesh.materialIndex];
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeLayout,
            1, 1, &mat.set, 0, nullptr);
        vkCmdPushConstants(cmd, meshPipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
            0, sizeof(glm::mat4), &mesh.transform);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vb, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.ib, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, (uint32_t)mesh.indices.size(), 1, 0, 0, 0);
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

void Renderer::drawFrame(Context& ctx, Camera& camera, Model& model) {
    Frame& f = frames[currentFrame];
    vkWaitForFences(ctx.device, 1, &f.inFlight, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult r = vkAcquireNextImageKHR(ctx.device, ctx.swapchain, UINT64_MAX,
        f.imageAvailable, VK_NULL_HANDLE, &imageIndex);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { recreate(ctx); return; }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) VK_CHECK(r);

    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(ctx.device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    imagesInFlight[imageIndex] = f.inFlight;

    // update global UBO
    float aspect = ctx.swapchainExtent.width / (float)ctx.swapchainExtent.height;
    GlobalUBO ubo{};
    ubo.view = camera.view();
    ubo.proj = camera.proj(aspect);
    ubo.camPos = glm::vec4(camera.position, 1.0f);
    ubo.lightDir = glm::vec4(glm::normalize(lightDir), 0.0f);
    ubo.lightColor = glm::vec4(lightColor, 1.0f);
    ubo.ambient = glm::vec4(ambient);
    std::memcpy(f.uboMapped, &ubo, sizeof(ubo));

    vkResetFences(ctx.device, 1, &f.inFlight);
    vkResetCommandBuffer(f.cmd, 0);
    recordFrame(ctx, f, imageIndex, model);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &f.imageAvailable;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &renderFinished[imageIndex];
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &si, f.inFlight));

    VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &renderFinished[imageIndex];
    pi.swapchainCount = 1;
    pi.pSwapchains = &ctx.swapchain;
    pi.pImageIndices = &imageIndex;
    r = vkQueuePresentKHR(ctx.queue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || framebufferResized)
        recreate(ctx);
    else if (r != VK_SUCCESS)
        VK_CHECK(r);

    currentFrame = (currentFrame + 1) % MAX_FRAMES;
}

void Renderer::shutdown(Context& ctx) {
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
    vkDestroyPipelineLayout(ctx.device, meshPipeLayout, nullptr);
    vkDestroyPipelineLayout(ctx.device, skyboxPipeLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, globalLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, materialLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, skyboxSetLayout, nullptr);
}