#include <Vulkan3DRenderer/RenderGraph.h>
#include <algorithm>
#include <iostream>
#include <queue>

// ---------------------------------------------------------------------------
// Usage -> resource state inference
// ---------------------------------------------------------------------------

bool RenderGraph::isWriteUsage(Usage u) {
    switch (u) {
    case Usage::ColorAttachmentWrite:
    case Usage::DepthAttachmentWrite:
    case Usage::StorageWriteCompute:
    case Usage::TransferDst:
        return true;
    default:
        return false;
    }
}

bool RenderGraph::isDepthUsage(Usage u) {
    return u == Usage::DepthAttachmentWrite || u == Usage::DepthAttachmentRead;
}

void RenderGraph::stateForUsage(Usage u,
    VkImageLayout& layout, VkPipelineStageFlags2& stage, VkAccessFlags2& access) {
    switch (u) {
    case Usage::ColorAttachmentWrite:
        layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
        break;
    case Usage::DepthAttachmentWrite:
        layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        break;
    case Usage::DepthAttachmentRead:
        layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        break;
    case Usage::SampledFragment:
        layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        break;
    case Usage::SampledCompute:
        layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        break;
    case Usage::StorageReadFragment:
        layout = VK_IMAGE_LAYOUT_GENERAL;
        stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        break;
    case Usage::StorageReadCompute:
        layout = VK_IMAGE_LAYOUT_GENERAL;
        stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        break;
    case Usage::StorageWriteCompute:
        layout = VK_IMAGE_LAYOUT_GENERAL;
        stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        break;
    case Usage::UniformRead:
        layout = VK_IMAGE_LAYOUT_UNDEFINED; // buffers: layout unused
        stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        access = VK_ACCESS_2_UNIFORM_READ_BIT;
        break;
    case Usage::TransferSrc:
        layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        access = VK_ACCESS_2_TRANSFER_READ_BIT;
        break;
    case Usage::TransferDst:
        layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        break;
    case Usage::Present:
        layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        stage = VK_PIPELINE_STAGE_2_NONE;
        access = 0;
        break;
    default:
        layout = VK_IMAGE_LAYOUT_GENERAL;
        stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        break;
    }
}

// ---------------------------------------------------------------------------
// Resource registration
// ---------------------------------------------------------------------------

RenderGraph::Handle RenderGraph::createImage(const ImageDesc& desc) {
    ImageResource r{};
    r.desc = desc;
    r.transient = true;
    r.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_images.push_back(r);
    m_transientsDirty = true;
    return (Handle)m_images.size() - 1;
}

RenderGraph::Handle RenderGraph::importImage(const ImageDesc& desc, VkImage image, VkImageView view,
    VkImageLayout initialLayout, VkPipelineStageFlags2 initialStage, VkAccessFlags2 initialAccess) {
    ImageResource r{};
    r.desc = desc;
    r.transient = false;
    r.image = image;
    r.view = view;
    r.layout = initialLayout;
    r.stage = initialStage;
    r.access = initialAccess;
    m_images.push_back(r);
    return (Handle)m_images.size() - 1;
}

RenderGraph::Handle RenderGraph::importBuffer(const std::string& name, VkBuffer buffer, VkDeviceSize size) {
    BufferResource r{};
    r.name = name;
    r.buffer = buffer;
    r.size = size;
    m_buffers.push_back(r);
    return (Handle)m_buffers.size() - 1;
}

void RenderGraph::updateImportedImage(Handle h, VkImage image, VkImageView view,
    uint32_t width, uint32_t height) {
    ImageResource& r = m_images[h];
    r.image = image;
    r.view = view;
    r.desc.width = width;
    r.desc.height = height;
    // swapchain images always come back from recreate in UNDEFINED
    r.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    r.stage = VK_PIPELINE_STAGE_2_NONE;
    r.access = 0;
    r.fresh = true;
}

void RenderGraph::setPresentTarget(Handle h, bool v) {
    m_images[h].presentTarget = v;
}

RenderGraph::PassBuilder RenderGraph::addPass(const std::string& name) {
    m_passes.push_back(Pass{});
    m_passes.back().name = name;
    return PassBuilder(*this, m_passes.back());
}

void RenderGraph::beginFrame() {
    m_passes.clear();
    m_execOrder.clear();
}

// ---------------------------------------------------------------------------
// Compile: dependency sort + culling + transient creation
// ---------------------------------------------------------------------------

void RenderGraph::compile(Context& ctx) {
    const uint32_t n = (uint32_t)m_passes.size();
    m_execOrder.clear();
    for (auto& p : m_passes) p.culled = false;

    // --- build dependencies -------------------------------------------------
    // dep[p] = bitmask of passes that must run before p
    std::vector<std::vector<uint32_t>> deps(n);
    std::vector<int> lastImageWriter(m_images.size(), -1);
    std::vector<int> lastBufferWriter(m_buffers.size(), -1);
    std::vector<std::vector<uint32_t>> imageReaders(m_images.size());
    std::vector<std::vector<uint32_t>> bufferReaders(m_buffers.size());

    auto addDep = [&](uint32_t pass, int earlier) {
        if (earlier < 0 || earlier == (int)pass) return;
        auto& d = deps[pass];
        if (std::find(d.begin(), d.end(), (uint32_t)earlier) == d.end())
            d.push_back((uint32_t)earlier);
    };

    for (uint32_t p = 0; p < n; p++) {
        for (auto& a : m_passes[p].imageAccesses) {
            if (a.resource == INVALID_HANDLE) continue;
            addDep(p, lastImageWriter[a.resource]);              // RAW
            if (a.write) {
                for (uint32_t r : imageReaders[a.resource]) addDep(p, (int)r); // WAR
                imageReaders[a.resource].clear();
                lastImageWriter[a.resource] = (int)p;
            }
            else {
                imageReaders[a.resource].push_back(p);
            }
        }
        for (auto& a : m_passes[p].bufferAccesses) {
            if (a.resource == INVALID_HANDLE) continue;
            addDep(p, lastBufferWriter[a.resource]);
            if (a.write) {
                for (uint32_t r : bufferReaders[a.resource]) addDep(p, (int)r);
                bufferReaders[a.resource].clear();
                lastBufferWriter[a.resource] = (int)p;
            }
            else {
                bufferReaders[a.resource].push_back(p);
            }
        }
    }

    // --- cull passes that don't feed a side-effect pass ---------------------
    std::vector<bool> needed(n, false);
    std::vector<uint32_t> stack;
    for (uint32_t p = 0; p < n; p++)
        if (m_passes[p].sideEffect) { needed[p] = true; stack.push_back(p); }
    while (!stack.empty()) {
        uint32_t p = stack.back(); stack.pop_back();
        for (uint32_t d : deps[p])
            if (!needed[d]) { needed[d] = true; stack.push_back(d); }
    }
    for (uint32_t p = 0; p < n; p++) m_passes[p].culled = !needed[p];

    // --- stable topological sort (Kahn, lowest declaration index first) -----
    std::vector<uint32_t> indeg(n, 0);
    std::vector<std::vector<uint32_t>> successors(n);
    for (uint32_t p = 0; p < n; p++) {
        if (!needed[p]) continue;
        for (uint32_t d : deps[p]) {
            if (!needed[d]) continue;
            successors[d].push_back(p);
            indeg[p]++;
        }
    }
    auto cmp = [](uint32_t a, uint32_t b) { return a > b; }; // min-heap
    std::priority_queue<uint32_t, std::vector<uint32_t>, decltype(cmp)> ready(cmp);
    for (uint32_t p = 0; p < n; p++)
        if (needed[p] && indeg[p] == 0) ready.push(p);
    while (!ready.empty()) {
        uint32_t p = ready.top(); ready.pop();
        m_execOrder.push_back(p);
        for (uint32_t s : successors[p])
            if (--indeg[s] == 0) ready.push(s);
    }

    createTransientImages(ctx);
}

// ---------------------------------------------------------------------------
// Transient image management
// ---------------------------------------------------------------------------

void RenderGraph::createTransientImages(Context& ctx) {
    for (uint32_t h = 0; h < (uint32_t)m_images.size(); h++) {
        ImageResource& r = m_images[h];
        if (!r.transient || r.image != VK_NULL_HANDLE) continue;

        // infer usage flags from every declared access
        VkImageUsageFlags usage = r.desc.extraUsage;
        for (auto& p : m_passes) {
            for (auto& a : p.imageAccesses) {
                if (a.resource != h) continue;
                switch (a.usage) {
                case Usage::ColorAttachmentWrite: usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; break;
                case Usage::DepthAttachmentWrite:
                case Usage::DepthAttachmentRead:  usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; break;
                case Usage::SampledFragment:
                case Usage::SampledCompute:       usage |= VK_IMAGE_USAGE_SAMPLED_BIT; break;
                case Usage::StorageReadFragment:
                case Usage::StorageReadCompute:
                case Usage::StorageWriteCompute:  usage |= VK_IMAGE_USAGE_STORAGE_BIT; break;
                case Usage::TransferSrc:          usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT; break;
                case Usage::TransferDst:          usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT; break;
                default: break;
                }
            }
        }

        ctx.createImage(r.desc.width, r.desc.height, 1, r.desc.layers, r.desc.createFlags,
            r.desc.format, usage, r.image, r.memory);
        r.view = ctx.createImageView(r.image, r.desc.format, r.desc.aspect, 1,
            r.desc.layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, r.desc.layers);
        r.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        r.stage = VK_PIPELINE_STAGE_2_NONE;
        r.access = 0;
        r.fresh = true;
    }
    m_transientsDirty = false;
}

void RenderGraph::destroyImageResource(Context& ctx, ImageResource& r) {
    if (r.view) vkDestroyImageView(ctx.device, r.view, nullptr);
    if (r.image) vkDestroyImage(ctx.device, r.image, nullptr);
    if (r.memory) vkFreeMemory(ctx.device, r.memory, nullptr);
    r.view = VK_NULL_HANDLE;
    r.image = VK_NULL_HANDLE;
    r.memory = VK_NULL_HANDLE;
}

void RenderGraph::destroyTransients(Context& ctx) {
    for (auto& r : m_images)
        if (r.transient) destroyImageResource(ctx, r);
    m_transientsDirty = true;
}

void RenderGraph::shutdown(Context& ctx) {
    destroyTransients(ctx);
}

// ---------------------------------------------------------------------------
// Execute: automatic barrier insertion + pass callbacks
// ---------------------------------------------------------------------------

void RenderGraph::emitImageBarrier(VkCommandBuffer cmd, ImageResource& r, Usage u) {
    VkImageLayout newLayout; VkPipelineStageFlags2 newStage; VkAccessFlags2 newAccess;
    stateForUsage(u, newLayout, newStage, newAccess);

    bool needBarrier = r.fresh
        || r.layout != newLayout
        || r.stage != newStage
        || r.access != newAccess
        || isWriteUsage(u)                                   // WAW / RAW always sync
        || (r.access & (VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
            VK_ACCESS_2_TRANSFER_WRITE_BIT |
            VK_ACCESS_2_MEMORY_WRITE_BIT)) != 0;

    if (!needBarrier) return;

    VkPipelineStageFlags2 srcStage = r.stage;
    VkAccessFlags2 srcAccess = r.access;
    VkImageLayout oldLayout = r.layout;
    if (r.fresh) {                      // first touch: no wait, just layout transition
        srcStage = VK_PIPELINE_STAGE_2_NONE;
        srcAccess = 0;
    }

    Context::cmdImageBarrier(cmd, r.image, oldLayout, newLayout,
        srcStage, srcAccess, newStage, newAccess, r.desc.aspect, 1, r.desc.layers);

    r.layout = newLayout;
    r.stage = newStage;
    r.access = newAccess;
    r.fresh = false;
}

void RenderGraph::emitBufferBarrier(VkCommandBuffer cmd, BufferResource& r, Usage u) {
    VkImageLayout unused; VkPipelineStageFlags2 newStage; VkAccessFlags2 newAccess;
    stateForUsage(u, unused, newStage, newAccess);

    bool prevWrite = (r.access & (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
        VK_ACCESS_2_TRANSFER_WRITE_BIT |
        VK_ACCESS_2_MEMORY_WRITE_BIT)) != 0;
    bool needBarrier = !r.fresh && (prevWrite || isWriteUsage(u)
        || r.stage != newStage || r.access != newAccess);
    if (needBarrier) {
        VkBufferMemoryBarrier2 b{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        b.srcStageMask = r.stage == VK_PIPELINE_STAGE_2_NONE ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : r.stage;
        b.srcAccessMask = r.access;
        b.dstStageMask = newStage;
        b.dstAccessMask = newAccess;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = r.buffer;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    r.stage = newStage;
    r.access = newAccess;
    r.fresh = false;
}

void RenderGraph::execute(VkCommandBuffer cmd) {
    for (uint32_t idx : m_execOrder) {
        Pass& p = m_passes[idx];
        for (auto& a : p.imageAccesses)  emitImageBarrier(cmd, m_images[a.resource], a.usage);
        for (auto& a : p.bufferAccesses) emitBufferBarrier(cmd, m_buffers[a.resource], a.usage);
        if (p.callback) p.callback(cmd);
    }
    // final transitions (e.g. swapchain -> PRESENT_SRC_KHR)
    for (auto& r : m_images) {
        if (r.presentTarget && r.layout != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR && !r.fresh)
            emitImageBarrier(cmd, r, Usage::Present);
    }
}

void RenderGraph::dumpExecutionOrder() const {
    std::cout << "[RenderGraph] execution order:";
    for (uint32_t idx : m_execOrder) std::cout << " " << m_passes[idx].name << " ->";
    std::cout << " end\n";
}
