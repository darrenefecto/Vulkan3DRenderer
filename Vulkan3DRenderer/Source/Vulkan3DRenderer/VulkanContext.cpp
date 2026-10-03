#include <Vulkan3DRenderer/VulkanContext.h>
#include <cstring>
#include <algorithm>
#include <iostream>

void Context::init(GLFWwindow* win, bool validation) {
    window = win;
    createInstance(validation);
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
    pickPhysicalDevice();
    createDevice();

    VkCommandPoolCreateInfo pi{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = queueFamily;
    VK_CHECK(vkCreateCommandPool(device, &pi, nullptr, &commandPool));

    createSwapchain();
    createDepthResources();
}

void Context::createInstance(bool validation) {
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "Vulkan Mesh Renderer";
    app.apiVersion = VK_API_VERSION_1_4; // needs Vulkan SDK/loader >= 1.4

    uint32_t glfwCount = 0;
    const char** glfwExt = glfwGetRequiredInstanceExtensions(&glfwCount);
    std::vector<const char*> extensions(glfwExt, glfwExt + glfwCount);

    std::vector<const char*> layers;
    if (validation) {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        vkEnumerateInstanceLayerProperties(&count, available.data());
        for (auto& l : available)
            if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0)
                layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo ci{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount = (uint32_t)layers.size();
    ci.ppEnabledLayerNames = layers.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
}

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR caps;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> modes;
};

static SwapchainSupport querySwapchain(VkPhysicalDevice d, VkSurfaceKHR surface) {
    SwapchainSupport s;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d, surface, &s.caps);
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(d, surface, &n, nullptr);
    s.formats.resize(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(d, surface, &n, s.formats.data());
    vkGetPhysicalDeviceSurfacePresentModesKHR(d, surface, &n, nullptr);
    s.modes.resize(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(d, surface, &n, s.modes.data());
    return s;
}

bool Context::isDeviceSuitable(VkPhysicalDevice d) const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(d, &props);
    if (props.apiVersion < VK_API_VERSION_1_3) return false; // we use 1.3-core features

    // queue family with graphics + present
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qf.data());
    bool foundQueue = false;
    for (uint32_t i = 0; i < qn; i++) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface, &present);
        if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
            const_cast<Context*>(this)->queueFamily = i;
            foundQueue = true;
            break;
        }
    }
    if (!foundQueue) return false;

    // swapchain extension
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(d, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(d, nullptr, &en, exts.data());
    bool hasSwapchain = false;
    for (auto& e : exts)
        if (std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) hasSwapchain = true;
    if (!hasSwapchain) return false;

    // required features: dynamic rendering + synchronization2 (core since 1.3)
    VkPhysicalDeviceVulkan13Features f13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.pNext = &f13;
    vkGetPhysicalDeviceFeatures2(d, &f2);
    if (!f13.dynamicRendering || !f13.synchronization2 || !f2.features.samplerAnisotropy) return false;

    auto sc = querySwapchain(d, surface);
    return !sc.formats.empty() && !sc.modes.empty();
}

void Context::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (!count) throw std::runtime_error("No Vulkan-capable GPU found");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    int bestScore = -1;
    for (auto d : devices) {
        if (!isDeviceSuitable(d)) continue;
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1000 :
            p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 100 : 10;
        if (score > bestScore) { bestScore = score; physicalDevice = d; }
    }
    if (!physicalDevice) throw std::runtime_error("No suitable GPU (needs dynamic rendering + sync2)");
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
    std::cout << "GPU: " << deviceProps.deviceName << " (Vulkan "
        << VK_VERSION_MAJOR(deviceProps.apiVersion) << "."
        << VK_VERSION_MINOR(deviceProps.apiVersion) << ")\n";
}

void Context::createDevice() {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan13Features f13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.pNext = &f13;
    f2.features.samplerAnisotropy = VK_TRUE;

    const char* devExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo ci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    ci.pNext = &f2;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qci;
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = devExts;
    VK_CHECK(vkCreateDevice(physicalDevice, &ci, nullptr, &device));
    vkGetDeviceQueue(device, queueFamily, 0, &queue);
}

void Context::createSwapchain() {
    auto sc = querySwapchain(physicalDevice, surface);

    VkSurfaceFormatKHR format = sc.formats[0];

    for (auto& f : sc.formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            format = f;
            break;
        }
    }

    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;

    for (auto m : sc.modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
            mode = m;
            break;
        }
    }

    VkExtent2D extent = sc.caps.currentExtent;

    if (extent.width == UINT32_MAX) {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);

        extent.width = std::clamp(
            (uint32_t)w,
            sc.caps.minImageExtent.width,
            sc.caps.maxImageExtent.width
        );

        extent.height = std::clamp(
            (uint32_t)h,
            sc.caps.minImageExtent.height,
            sc.caps.maxImageExtent.height
        );
    }

    uint32_t imageCount = sc.caps.minImageCount + 1;

    if (sc.caps.maxImageCount > 0) {
        imageCount = std::min(imageCount, sc.caps.maxImageCount);
    }

    VkSwapchainCreateInfoKHR ci{
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR
    };

    ci.surface = surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = format.format;
    ci.imageColorSpace = format.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = sc.caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(
        device,
        &ci,
        nullptr,
        &swapchain
    ));

    swapchainFormat = format.format;
    swapchainExtent = extent;

    uint32_t n = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &n, nullptr);

    swapchainImages.resize(n);

    vkGetSwapchainImagesKHR(
        device,
        swapchain,
        &n,
        swapchainImages.data()
    );

    swapchainViews.resize(n);

    for (uint32_t i = 0; i < n; i++) {
        swapchainViews[i] = createImageView(
            swapchainImages[i],
            swapchainFormat,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1
        );
    }
}

VkFormat Context::findDepthFormat() const {
    const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_X8_D24_UNORM_PACK32 };
    for (auto f : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, f, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) return f;
    }
    throw std::runtime_error("No depth format supported");
}

void Context::createDepthResources() {
    depthFormat = findDepthFormat();
    createImage(swapchainExtent.width, swapchainExtent.height, 1, 1, 0, depthFormat,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImage, depthMemory);
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT || depthFormat == VK_FORMAT_D24_UNORM_S8_UINT)
        aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    depthView = createImageView(depthImage, depthFormat, aspect, 1);
}

void Context::destroySwapchainResources() {
    vkDestroyImageView(device, depthView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthMemory, nullptr);
    for (auto v : swapchainViews) vkDestroyImageView(device, v, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void Context::recreateSwapchain() {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    while (w == 0 || h == 0) { glfwGetFramebufferSize(window, &w, &h); glfwWaitEvents(); }
    vkDeviceWaitIdle(device);
    destroySwapchainResources();
    createSwapchain();
    createDepthResources();
}

void Context::shutdown() {
    destroySwapchainResources();
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
}

// ---------------- helpers ----------------

VkCommandBuffer Context::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void Context::endSingleTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

uint32_t Context::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    throw std::runtime_error("No suitable memory type");
}

void Context::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
    VkBuffer& buffer, VkDeviceMemory& memory) const {
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bi, nullptr, &buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buffer, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &memory));
    vkBindBufferMemory(device, buffer, memory, 0);
}

void Context::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy region{ 0, 0, size };
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    endSingleTimeCommands(cmd);
}

void Context::createImage(uint32_t w, uint32_t h, uint32_t mipLevels, uint32_t layers,
    VkImageCreateFlags flags, VkFormat format, VkImageUsageFlags usage,
    VkImage& image, VkDeviceMemory& memory) const {
    VkImageCreateInfo ii{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent = { w, h, 1 };
    ii.mipLevels = mipLevels;
    ii.arrayLayers = layers;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = usage;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ii.flags = flags;
    VK_CHECK(vkCreateImage(device, &ii, nullptr, &image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, image, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &memory));
    vkBindImageMemory(device, image, memory, 0);
}

VkImageView Context::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect,
    uint32_t mipLevels, VkImageViewType type, uint32_t layers) const {
    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = image;
    vi.viewType = type;
    vi.format = format;
    vi.subresourceRange = { aspect, 0, mipLevels, 0, layers };
    VkImageView view;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &view));
    return view;
}

void Context::cmdImageBarrier(VkCommandBuffer cmd, VkImage image,
    VkImageLayout oldL, VkImageLayout newL,
    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
    VkImageAspectFlags aspect, uint32_t mipLevels, uint32_t layers) {
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;
    b.dstAccessMask = dstAccess;
    b.oldLayout = oldL;
    b.newLayout = newL;
    b.image = image;
    b.subresourceRange = { aspect, 0, mipLevels, 0, layers };
    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep); // synchronization2, core in Vulkan 1.3+
}

void Context::transitionImageLayout(VkImage image, VkImageLayout oldL, VkImageLayout newL,
    uint32_t mipLevels, uint32_t layers, VkImageAspectFlags aspect) {
    VkCommandBuffer cmd = beginSingleTimeCommands();
    cmdImageBarrier(cmd, image, oldL, newL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        aspect, mipLevels, layers);
    endSingleTimeCommands(cmd);
}

void Context::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t w, uint32_t h, uint32_t layers) {
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers };
    region.imageExtent = { w, h, 1 };
    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(cmd);
}

VkShaderModule Context::createShaderModule(const std::vector<char>& code) const {
    VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = code.size();
    ci.pCode = (const uint32_t*)code.data();
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &mod));
    return mod;
}