#include <Vulkan3DRenderer/Texture.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

void Texture2D::createFromFile(Context& ctx, const std::string& path, bool srgb) {
    int w, h, channels;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) throw std::runtime_error("Failed to load texture: " + path);
    createFromPixels(ctx, pixels, (uint32_t)w, (uint32_t)h, srgb);
    stbi_image_free(pixels);
}

void Texture2D::createFromMemory(Context& ctx, const void* data, int byteLength, bool srgb) {
    int w, h, channels;
    stbi_uc* pixels = stbi_load_from_memory((const stbi_uc*)data, byteLength, &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) throw std::runtime_error("Failed to decode embedded texture");
    createFromPixels(ctx, pixels, (uint32_t)w, (uint32_t)h, srgb);
    stbi_image_free(pixels);
}

void Texture2D::createSolid(Context& ctx, uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb) {
    uint8_t px[4] = { r, g, b, a };
    createFromPixels(ctx, px, 1, 1, srgb);
}

void Texture2D::createFromPixels(Context& ctx, const void* pixels, uint32_t w, uint32_t h, bool srgb) {
    width = w; height = h;
    mipLevels = (uint32_t)std::floor(std::log2((double)std::max(w, h))) + 1;

    // SRGB blit support check (needed for mipmap generation)
    VkFormatProperties fpSrgb, fpUnorm;
    vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, VK_FORMAT_R8G8B8A8_SRGB, &fpSrgb);
    vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &fpUnorm);
    if (srgb && (fpSrgb.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        format = VK_FORMAT_R8G8B8A8_SRGB;
    else
        format = VK_FORMAT_R8G8B8A8_UNORM;

    VkDeviceSize size = (VkDeviceSize)w * h * 4;
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    ctx.createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
    void* mapped;
    vkMapMemory(ctx.device, stagingMem, 0, size, 0, &mapped);
    std::memcpy(mapped, pixels, size);
    vkUnmapMemory(ctx.device, stagingMem);

    ctx.createImage(w, h, mipLevels, 1, 0, format,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        image, memory);

    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        mipLevels, 1);
    ctx.copyBufferToImage(staging, image, w, h, 1);
    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);

    generateMipmaps(ctx); // leaves image in SHADER_READ_ONLY_OPTIMAL

    view = ctx.createImageView(image, format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.anisotropyEnable = VK_TRUE;
    si.maxAnisotropy = std::min(8.0f, ctx.deviceProps.limits.maxSamplerAnisotropy);
    si.maxLod = (float)mipLevels;
    VK_CHECK(vkCreateSampler(ctx.device, &si, nullptr, &sampler));
}

void Texture2D::generateMipmaps(Context& ctx) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    int32_t mipW = (int32_t)width, mipH = (int32_t)height;

    for (uint32_t i = 1; i < mipLevels; i++) {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.image = image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 };
        blit.srcOffsets[1] = { mipW, mipH, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };
        blit.dstOffsets[1] = { mipW > 1 ? mipW / 2 : 1, mipH > 1 ? mipH / 2 : 1, 1 };
        vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        if (mipW > 1) mipW /= 2;
        if (mipH > 1) mipH /= 2;
    }
    // last mip level
    VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, mipLevels - 1, 1, 0, 1 };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b);
    ctx.endSingleTimeCommands(cmd);
}

void Texture2D::destroy(Context& ctx) {
    vkDestroySampler(ctx.device, sampler, nullptr);
    vkDestroyImageView(ctx.device, view, nullptr);
    vkDestroyImage(ctx.device, image, nullptr);
    vkFreeMemory(ctx.device, memory, nullptr);
}

// ---------------- Cubemap ----------------

static bool fileExists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

void Cubemap::createFromFiles(Context& ctx, const std::array<std::string, 6>& paths) {
    std::array<stbi_uc*, 6> pixels{};
    int w = 0, h = 0;
    for (int i = 0; i < 6; i++) {
        std::string p = paths[i];
        if (!fileExists(p)) { // jpg <-> png fallback
            std::string alt = p.substr(0, p.find_last_of('.')) +
                (p.ends_with(".png") ? ".jpg" : ".png");
            if (fileExists(alt)) p = alt;
        }
        int fw, fh, ch;
        pixels[i] = stbi_load(p.c_str(), &fw, &fh, &ch, STBI_rgb_alpha);
        if (!pixels[i]) {
            for (int j = 0; j < i; j++) stbi_image_free(pixels[j]);
            throw std::runtime_error("Failed to load cubemap face: " + p);
        }
        if (i == 0) { w = fw; h = fh; }
        else if (fw != w || fh != h) throw std::runtime_error("Cubemap faces must be the same size");
    }

    mipLevels = (uint32_t)std::floor(std::log2((double)std::max(w, h))) + 1;

    VkDeviceSize faceSize = (VkDeviceSize)w * h * 4;
    VkDeviceSize total = faceSize * 6;
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    ctx.createBuffer(total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
    void* mapped;
    vkMapMemory(ctx.device, stagingMem, 0, total, 0, &mapped);
    for (int i = 0; i < 6; i++) {
        std::memcpy((char*)mapped + i * faceSize, pixels[i], faceSize);
        stbi_image_free(pixels[i]);
    }
    vkUnmapMemory(ctx.device, stagingMem);

    ctx.createImage((uint32_t)w, (uint32_t)h, mipLevels, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        image, memory);

    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        mipLevels, 6);

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    std::array<VkBufferImageCopy, 6> regions{};
    for (uint32_t i = 0; i < 6; i++) {
        regions[i].bufferOffset = i * faceSize;
        regions[i].imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, i, 1 };
        regions[i].imageExtent = { (uint32_t)w, (uint32_t)h, 1 };
    }
    vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, regions.data());
    ctx.endSingleTimeCommands(cmd);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);

    generateMipmaps(ctx, (uint32_t)w, (uint32_t)h); // leaves image in SHADER_READ_ONLY_OPTIMAL

    view = ctx.createImageView(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT,
        mipLevels, VK_IMAGE_VIEW_TYPE_CUBE, 6);

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.anisotropyEnable = VK_TRUE;
    si.maxAnisotropy = std::min(8.0f, ctx.deviceProps.limits.maxSamplerAnisotropy);
    si.maxLod = (float)mipLevels;
    VK_CHECK(vkCreateSampler(ctx.device, &si, nullptr, &sampler));
}

void Cubemap::generateMipmaps(Context& ctx, uint32_t w, uint32_t h) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    int32_t mipW = (int32_t)w, mipH = (int32_t)h;

    for (uint32_t i = 1; i < mipLevels; i++) {
        // previous mip of every face: DST -> SRC
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.image = image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 6 };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        for (uint32_t f = 0; f < 6; f++) {
            VkImageBlit blit{};
            blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, f, 1 };
            blit.srcOffsets[1] = { mipW, mipH, 1 };
            blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, f, 1 };
            blit.dstOffsets[1] = { mipW > 1 ? mipW / 2 : 1, mipH > 1 ? mipH / 2 : 1, 1 };
            vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        }

        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        if (mipW > 1) mipW /= 2;
        if (mipH > 1) mipH /= 2;
    }
    // last mip level of every face
    VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, mipLevels - 1, 1, 0, 6 };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b);
    ctx.endSingleTimeCommands(cmd);
}

void Cubemap::createSolid(Context& ctx, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mipLevels = 1;
    uint8_t px[4] = { r, g, b, a };
    VkDeviceSize total = 4 * 6;
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    ctx.createBuffer(total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
    void* mapped;
    vkMapMemory(ctx.device, stagingMem, 0, total, 0, &mapped);
    for (int i = 0; i < 6; i++) std::memcpy((char*)mapped + i * 4, px, 4);
    vkUnmapMemory(ctx.device, stagingMem);

    ctx.createImage(1, 1, 1, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, image, memory);
    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 6);
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    std::array<VkBufferImageCopy, 6> regions{};
    for (uint32_t i = 0; i < 6; i++) {
        regions[i].bufferOffset = i * 4;
        regions[i].imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, i, 1 };
        regions[i].imageExtent = { 1, 1, 1 };
    }
    vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, regions.data());
    ctx.endSingleTimeCommands(cmd);
    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);
    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 6);
    view = ctx.createImageView(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT,
        1, VK_IMAGE_VIEW_TYPE_CUBE, 6);
    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(ctx.device, &si, nullptr, &sampler));
}

void Cubemap::destroy(Context& ctx) {
    vkDestroySampler(ctx.device, sampler, nullptr);
    vkDestroyImageView(ctx.device, view, nullptr);
    vkDestroyImage(ctx.device, image, nullptr);
    vkFreeMemory(ctx.device, memory, nullptr);
}
