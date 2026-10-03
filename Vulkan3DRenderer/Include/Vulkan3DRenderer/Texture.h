#pragma once
#include "VulkanContext.h"
#include <array>
#include <string>

class Texture2D {
public:
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
    uint32_t width = 0, height = 0, mipLevels = 1;

    void createFromFile(Context& ctx, const std::string& path);
    void createFromMemory(Context& ctx, const void* data, int byteLength); // png/jpg in memory
    void createSolid(Context& ctx, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void destroy(Context& ctx);

private:
    void createFromPixels(Context& ctx, const void* pixels, uint32_t w, uint32_t h);
    void generateMipmaps(Context& ctx);
};

class Cubemap {
public:
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    void createFromFiles(Context& ctx, const std::array<std::string, 6>& paths); // px nx py ny pz nz
    void createSolid(Context& ctx, uint8_t r, uint8_t g, uint8_t b, uint8_t a);  // fallback
    void destroy(Context& ctx);
};