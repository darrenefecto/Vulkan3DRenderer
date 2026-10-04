#pragma once
#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <stdexcept>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) \
    throw std::runtime_error(std::string("Vulkan error ") + std::to_string((int)_r) + " : " #x); } while (0)

// Owns instance, device, swapchain, depth buffer and low-level helpers.
class Context {
public:
    GLFWwindow* window = nullptr;
    VkInstance           instance = VK_NULL_HANDLE;
    VkSurfaceKHR         surface = VK_NULL_HANDLE;
    VkPhysicalDevice     physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties deviceProps{};
    VkDevice             device = VK_NULL_HANDLE;
    uint32_t             queueFamily = 0;
    VkQueue              queue = VK_NULL_HANDLE;
    VkCommandPool        commandPool = VK_NULL_HANDLE;

    VkSwapchainKHR       swapchain = VK_NULL_HANDLE;
    VkFormat             swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D           swapchainExtent{};
    std::vector<VkImage>     swapchainImages;
    std::vector<VkImageView> swapchainViews;

    VkFormat       depthFormat = VK_FORMAT_D32_SFLOAT;
    VkImage        depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView    depthView = VK_NULL_HANDLE;

    void init(GLFWwindow* window, bool validationLayers);
    void recreateSwapchain();
    void shutdown();

    // --- helpers ---
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const;
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
        VkBuffer& buffer, VkDeviceMemory& memory) const;
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    void createImage(uint32_t w, uint32_t h, uint32_t mipLevels, uint32_t layers,
        VkImageCreateFlags flags, VkFormat format, VkImageUsageFlags usage,
        VkImage& image, VkDeviceMemory& memory) const;
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect,
        uint32_t mipLevels, VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D,
        uint32_t layers = 1, uint32_t baseLayer = 0, uint32_t baseMip = 0) const;
    void transitionImageLayout(VkImage image, VkImageLayout oldL, VkImageLayout newL,
        uint32_t mipLevels, uint32_t layers,
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t w, uint32_t h, uint32_t layers);
    VkShaderModule createShaderModule(const std::vector<char>& code) const;

    static void cmdImageBarrier(VkCommandBuffer cmd, VkImage image,
        VkImageLayout oldL, VkImageLayout newL,
        VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
        VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
        uint32_t mipLevels = 1, uint32_t layers = 1);

private:
    void createInstance(bool validation);
    void pickPhysicalDevice();
    void createDevice();
    void createSwapchain();
    void createDepthResources();
    void destroySwapchainResources();
    VkFormat findDepthFormat() const;
    bool isDeviceSuitable(VkPhysicalDevice d) const;
};
