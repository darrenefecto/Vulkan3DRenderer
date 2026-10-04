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

    // srgb=false for data textures (normal, metallic-roughness, AO)
    void createFromFile(Context& ctx, const std::string& path, bool srgb = true);
    void createFromMemory(Context& ctx, const void* data, int byteLength, bool srgb = true); // png/jpg in memory
    void createSolid(Context& ctx, uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb = true);
    void destroy(Context& ctx);

private:
    void createFromPixels(Context& ctx, const void* pixels, uint32_t w, uint32_t h, bool srgb);
    void generateMipmaps(Context& ctx);
};

class Cubemap {
public:
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    uint32_t mipLevels = 1;

    void createFromFiles(Context& ctx, const std::array<std::string, 6>& paths); // px nx py ny pz nz
    void createSolid(Context& ctx, uint8_t r, uint8_t g, uint8_t b, uint8_t a);  // fallback
    void destroy(Context& ctx);

private:
    void generateMipmaps(Context& ctx, uint32_t w, uint32_t h);
};
