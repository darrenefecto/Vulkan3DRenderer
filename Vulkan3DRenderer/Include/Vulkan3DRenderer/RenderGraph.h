#pragma once
#include "VulkanContext.h"
#include <functional>
#include <string>
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// RenderGraph
//
// A frame graph that owns the "what / in which order / with which resources"
// description of a frame:
//
//   * WHAT   - passes are declared with a record callback (the GPU work).
//   * ORDER  - passes are topologically sorted from their resource
//              dependencies (a stable sort keeps declaration order whenever
//              no dependency forces a reorder).
//   * RESOURCES - images/buffers are imported (owned elsewhere: swapchain,
//              shadow maps, per-frame UBOs) or transient (created/owned by
//              the graph from a description). Every access declares a Usage;
//              the graph infers layouts, pipeline stages and access masks and
//              inserts all VkImageMemoryBarrier2 / VkBufferMemoryBarrier2
//              automatically. Passes that contribute nothing to a
//              side-effect pass (e.g. a shadow pass nobody samples) are
//              culled.
//
// Typical use (per frame):
//      rg.beginFrame();                      // clears passes, keeps state
//      auto h = rg.addPass("GBuffer");
//      h.writeImage(albedo, Usage::ColorAttachmentWrite)
//       .writeImage(depth,  Usage::DepthAttachmentWrite)
//       .record([&](VkCommandBuffer cmd){ ... });
//      ...
//      rg.compile(ctx);                      // sort, cull, create transients
//      rg.execute(cmd);                      // barriers + callbacks
// ---------------------------------------------------------------------------

class RenderGraph {
public:
    using Handle = uint32_t;
    static constexpr Handle INVALID_HANDLE = ~0u;

    // How a pass touches a resource. Drives layout/stage/access inference.
    enum class Usage {
        None,
        ColorAttachmentWrite,   // render pass color output
        DepthAttachmentWrite,   // render pass depth output (test + write)
        DepthAttachmentRead,    // render pass depth input (test only, load)
        SampledFragment,        // texture() in a fragment shader
        SampledCompute,         // texture() in a compute shader
        StorageReadFragment,    // SSBO / storage image read in fragment
        StorageReadCompute,     // SSBO read in compute
        StorageWriteCompute,    // SSBO / storage image write in compute
        UniformRead,            // UBO read (any stage)
        TransferSrc,
        TransferDst,
        Present                 // final swapchain present
    };

    struct ImageDesc {
        std::string name;
        uint32_t width = 0, height = 0, layers = 1;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        VkImageCreateFlags createFlags = 0;
        VkImageUsageFlags extraUsage = 0;
    };

    struct ImageResource {
        ImageDesc desc;
        bool transient = false;             // created + owned by the graph
        bool presentTarget = false;         // auto-transition to PRESENT at frame end
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE; // transient only
        VkImageView view = VK_NULL_HANDLE;      // whole-resource view (all layers)
        // tracked execution state (persists across frames)
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 access = 0;
        bool fresh = true;                  // never used inside the graph yet
    };

    struct BufferResource {
        std::string name;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 access = 0;
        bool fresh = true;
    };

    struct Access {
        Handle resource = INVALID_HANDLE;
        Usage usage = Usage::None;
        bool write = false;
    };

    struct Pass {
        std::string name;
        std::vector<Access> imageAccesses;
        std::vector<Access> bufferAccesses;
        std::function<void(VkCommandBuffer)> callback;
        bool sideEffect = false;            // must execute even if output unused
        bool culled = false;
    };

    // ------------------------------------------------------------------
    // Resource registration
    // ------------------------------------------------------------------

    // Graph-owned image, created lazily in compile() from the desc.
    Handle createImage(const ImageDesc& desc);

    // Externally-owned image (swapchain, shadow maps, ...). The initial
    // layout/stage/access describe the state the resource is in BEFORE the
    // first frame touches it; afterwards the graph keeps tracking it.
    Handle importImage(const ImageDesc& desc, VkImage image, VkImageView view,
        VkImageLayout initialLayout,
        VkPipelineStageFlags2 initialStage = VK_PIPELINE_STAGE_2_NONE,
        VkAccessFlags2 initialAccess = 0);

    Handle importBuffer(const std::string& name, VkBuffer buffer, VkDeviceSize size);

    // Update handles after swapchain recreation (keeps tracked state).
    void updateImportedImage(Handle h, VkImage image, VkImageView view,
        uint32_t width, uint32_t height);

    // Marks an image as "transition to PRESENT_SRC_KHR when the frame ends".
    void setPresentTarget(Handle h, bool v = true);

    const ImageResource& image(Handle h) const { return m_images[h]; }
    ImageResource& image(Handle h) { return m_images[h]; }
    const BufferResource& buffer(Handle h) const { return m_buffers[h]; }
    VkImageView imageView(Handle h) const { return m_images[h].view; }

    // ------------------------------------------------------------------
    // Pass declaration
    // ------------------------------------------------------------------

    class PassBuilder {
    public:
        PassBuilder(RenderGraph& g, Pass& p) : g(g), p(p) {}
        PassBuilder& readImage(Handle h, Usage u) { p.imageAccesses.push_back({ h, u, false }); return *this; }
        PassBuilder& writeImage(Handle h, Usage u) { p.imageAccesses.push_back({ h, u, true }); return *this; }
        PassBuilder& readBuffer(Handle h, Usage u) { p.bufferAccesses.push_back({ h, u, false }); return *this; }
        PassBuilder& writeBuffer(Handle h, Usage u) { p.bufferAccesses.push_back({ h, u, true }); return *this; }
        PassBuilder& sideEffect(bool v = true) { p.sideEffect = v; return *this; }
        void record(std::function<void(VkCommandBuffer)> fn) { p.callback = std::move(fn); }
    private:
        RenderGraph& g;
        Pass& p;
    };

    PassBuilder addPass(const std::string& name);

    // ------------------------------------------------------------------
    // Frame lifecycle
    // ------------------------------------------------------------------

    void beginFrame();                  // clears passes (resource state persists)
    void compile(Context& ctx);         // topo-sort, cull, create transients
    void execute(VkCommandBuffer cmd);  // automatic barriers + pass callbacks

    // Destroy/recreate graph-owned images (call after swapchain resize).
    void destroyTransients(Context& ctx);
    void shutdown(Context& ctx);        // destroys transients; imported stay

    const std::vector<uint32_t>& executionOrder() const { return m_execOrder; }

    // Debug: prints the final execution order to stdout.
    void dumpExecutionOrder() const;

private:
    std::vector<ImageResource> m_images;
    std::vector<BufferResource> m_buffers;
    std::vector<Pass> m_passes;
    std::vector<uint32_t> m_execOrder;   // indices into m_passes, execution order
    bool m_transientsDirty = false;

    void createTransientImages(Context& ctx);
    void destroyImageResource(Context& ctx, ImageResource& r);

    static bool isWriteUsage(Usage u);
    static bool isDepthUsage(Usage u);
    static void stateForUsage(Usage u,
        VkImageLayout& layout, VkPipelineStageFlags2& stage, VkAccessFlags2& access);
    void emitImageBarrier(VkCommandBuffer cmd, ImageResource& r, Usage u);
    void emitBufferBarrier(VkCommandBuffer cmd, BufferResource& r, Usage u);
};
