#include <Vulkan3DRenderer/Texture.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

void Texture2D::createFromFile(Context& ctx, const std::string& path, bool srgb) {
    int w, h, channels;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) throw std::runtime_error("Failed to load texture: " + path);
    try {
        createFromPixels(ctx, pixels, (uint32_t)w, (uint32_t)h, srgb);
    } catch (...) {
        stbi_image_free(pixels);
        throw;
    }
    stbi_image_free(pixels);
}

void Texture2D::createFromMemory(Context& ctx, const void* data, int byteLength, bool srgb) {
    int w, h, channels;
    stbi_uc* pixels = stbi_load_from_memory((const stbi_uc*)data, byteLength, &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) throw std::runtime_error("Failed to decode embedded texture");
    try {
        createFromPixels(ctx, pixels, (uint32_t)w, (uint32_t)h, srgb);
    } catch (...) {
        stbi_image_free(pixels);
        throw;
    }
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
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    try {
    ctx.createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
    void* mapped;
    VK_CHECK(vkMapMemory(ctx.device, stagingMem, 0, size, 0, &mapped));
    std::memcpy(mapped, pixels, size);
    vkUnmapMemory(ctx.device, stagingMem);

    ctx.createImage(w, h, mipLevels, 1, 0, format,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        image, memory);

    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        mipLevels, 1);
    ctx.copyBufferToImage(staging, image, w, h, 1);
    vkDestroyBuffer(ctx.device, staging, nullptr);
    staging = VK_NULL_HANDLE;
    vkFreeMemory(ctx.device, stagingMem, nullptr);
    stagingMem = VK_NULL_HANDLE;

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
    } catch (...) {
        if (staging != VK_NULL_HANDLE) vkDestroyBuffer(ctx.device, staging, nullptr);
        if (stagingMem != VK_NULL_HANDLE) vkFreeMemory(ctx.device, stagingMem, nullptr);
        destroy(ctx);
        throw;
    }
}

// Scoped owner for temporary upload resources, including exception paths.
namespace {
struct ScopedUploadBuffer {
    VkDevice device;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    explicit ScopedUploadBuffer(VkDevice d) : device(d) {}
    ScopedUploadBuffer(const ScopedUploadBuffer&) = delete;
    ScopedUploadBuffer& operator=(const ScopedUploadBuffer&) = delete;
    ~ScopedUploadBuffer() {
        if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
    }
};
}

void Texture2D::createFromRG32FFile(Context& ctx, const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open RG32F LUT: " + path);
    const auto fileSize = file.tellg();
    file.seekg(0);
    uint32_t dimensions[2]{};
    file.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
    const uint32_t w = dimensions[0], h = dimensions[1];
    if (!file || !w || !h || w > 4096 || h > 4096 ||
        fileSize != static_cast<std::streamoff>(sizeof(dimensions) + uint64_t(w) * h * 2 * sizeof(float)))
        throw std::runtime_error("Invalid RG32F LUT header/size: " + path);
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, VK_FORMAT_R32G32_SFLOAT, &props);
    if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0 ||
        (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0)
        throw std::runtime_error("RG32F linear sampling not supported");
    const VkDeviceSize byteSize = VkDeviceSize(w) * h * 2 * sizeof(float);
    std::vector<char> pixels(static_cast<size_t>(byteSize));
    file.read(pixels.data(), static_cast<std::streamsize>(byteSize));
    if (!file) throw std::runtime_error("Truncated RG32F LUT: " + path);
    ScopedUploadBuffer upload(ctx.device);
    try {
    ctx.createBuffer(byteSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, upload.buffer, upload.memory);
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx.device, upload.memory, 0, byteSize, 0, &mapped));
    std::memcpy(mapped, pixels.data(), static_cast<size_t>(byteSize));
    vkUnmapMemory(ctx.device, upload.memory);
    width = w; height = h; mipLevels = 1; format = VK_FORMAT_R32G32_SFLOAT;
    ctx.createImage(w, h, 1, 1, 0, format,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, image, memory);
    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 1);
    ctx.copyBufferToImage(upload.buffer, image, w, h, 1);
    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1);
    view = ctx.createImageView(image, format, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(ctx.device, &si, nullptr, &sampler));
    } catch (...) {
        destroy(ctx); // also handles partially created image, view and sampler
        throw;
    }
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
    sampler = VK_NULL_HANDLE;
    view = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    width = height = 0;
    mipLevels = 1;
}

// ---------------- Cubemap ----------------

static bool fileExists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

static std::string resolveAssetPath(const std::string& path) {
    namespace fs = std::filesystem;
    const fs::path requested(path);
    std::error_code ec;
    if (requested.is_absolute() && fs::exists(requested, ec)) return requested.string();
    if (fs::exists(requested, ec)) return fs::absolute(requested, ec).string();

    fs::path base = fs::current_path(ec);
    for (int depth = 0; depth < 8 && !base.empty(); ++depth) {
        const fs::path candidate = base / requested;
        ec.clear();
        if (fs::exists(candidate, ec)) return candidate.string();
        const fs::path parent = base.parent_path();
        if (parent == base) break;
        base = parent;
    }
    return path;
}

void Cubemap::createFromFiles(Context& ctx, const std::array<std::string, 6>& paths) {
    std::array<stbi_uc*, 6> pixels{};
    int w = 0, h = 0;
    for (int i = 0; i < 6; i++) {
        std::string p = resolveAssetPath(paths[i]);
        if (!fileExists(p)) {
            std::string alt = p.substr(0, p.find_last_of('.')) +
                (p.ends_with(".png") ? ".jpg" : ".png");
            alt = resolveAssetPath(alt);
            if (fileExists(alt)) p = alt;
        }
        int fw = 0, fh = 0, ch = 0;
        pixels[i] = stbi_load(p.c_str(), &fw, &fh, &ch, STBI_rgb_alpha);
        if (!pixels[i]) {
            for (stbi_uc* loaded : pixels) if (loaded) stbi_image_free(loaded);
            throw std::runtime_error("Failed to load cubemap face: " + p);
        }
        if (i == 0) { w = fw; h = fh; }
        else if (fw != w || fh != h) {
            for (stbi_uc* loaded : pixels) if (loaded) stbi_image_free(loaded);
            throw std::runtime_error("Cubemap faces must be the same size: " + p);
        }
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

void Cubemap::createFromMipFiles(Context& ctx, const std::string& directory, uint32_t levels) {
    if (!levels || levels > 16) throw std::runtime_error("Invalid prefiltered cubemap mip count");
    const char* faces[6] = { "px", "nx", "py", "ny", "pz", "nz" };
    std::vector<unsigned char> stagingBytes;
    std::vector<VkBufferImageCopy> copies;
    int base = 0;
    for (uint32_t mip = 0; mip < levels; ++mip) {
        int expected = 0;
        for (uint32_t face = 0; face < 6; ++face) {
            std::string path = resolveAssetPath(directory + "/mip" + std::to_string(mip) + "/" + faces[face] + ".png");
            int w = 0, h = 0, channels = 0;
            stbi_uc* data = stbi_load(path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
            if (!data) throw std::runtime_error("Missing prefiltered cubemap mip face: " + path);
            if (!base) base = w;
            expected = std::max(1, base >> mip);
            if (w != expected || h != expected) {
                stbi_image_free(data);
                throw std::runtime_error("Invalid prefiltered cubemap mip dimensions: " + path);
            }
            VkBufferImageCopy copy{};
            copy.bufferOffset = stagingBytes.size();
            copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, mip, face, 1 };
            copy.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };
            copies.push_back(copy);
            stagingBytes.insert(stagingBytes.end(), data, data + size_t(w) * h * 4);
            stbi_image_free(data);
        }
    }
    mipLevels = levels;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    ctx.createBuffer(stagingBytes.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx.device, stagingMem, 0, stagingBytes.size(), 0, &mapped));
    std::memcpy(mapped, stagingBytes.data(), stagingBytes.size());
    vkUnmapMemory(ctx.device, stagingMem);
    ctx.createImage(base, base, levels, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, image, memory);
    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, levels, 6);
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        uint32_t(copies.size()), copies.data());
    ctx.endSingleTimeCommands(cmd);
    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMem, nullptr);
    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, levels, 6);
    view = ctx.createImageView(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT,
        levels, VK_IMAGE_VIEW_TYPE_CUBE, 6);
    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = float(levels - 1);
    VK_CHECK(vkCreateSampler(ctx.device, &si, nullptr, &sampler));
}

// Offline HDR bake format: tightly packed little-endian float32 RGBA texels,
// one file per face/mip. Files are validated before Vulkan resources are created.
void Cubemap::createFromHDRRawMipFiles(Context& ctx, const std::string& directory, uint32_t levels) {
    if (!levels || levels > 16) throw std::runtime_error("Invalid HDR IBL mip count");
    const char* names[6] = { "px", "nx", "py", "ny", "pz", "nz" };
    std::vector<char> bytes;
    std::vector<VkBufferImageCopy> regions;
    uint32_t base = 0;
    for (uint32_t mip = 0; mip < levels; ++mip) {
        for (uint32_t f = 0; f < 6; ++f) {
            auto file = resolveAssetPath(directory + "/mip" + std::to_string(mip) + "/" + names[f] + ".rgba32f");
            std::ifstream stream(file, std::ios::binary | std::ios::ate);
            if (!stream) throw std::runtime_error("Missing HDR IBL face: " + file);
            const auto len = stream.tellg();
            if (len <= 0 || len % 16 != 0) throw std::runtime_error("Invalid HDR IBL data: " + file);
            if (mip == 0 && f == 0) {
                base = (uint32_t)std::sqrt(double(len) / 16.0);
                if (!base || (base & (base - 1)) || uint64_t(base) * base * 16 != uint64_t(len))
                    throw std::runtime_error("HDR IBL base dimension must be square and power of two");
                if (levels > uint32_t(std::log2(base)) + 1) throw std::runtime_error("Excessive HDR IBL mip count");
            }
            uint32_t side = std::max(1u, base >> mip);
            if (uint64_t(len) != uint64_t(side) * side * 16) throw std::runtime_error("HDR IBL mip size mismatch: " + file);
            VkBufferImageCopy region{};
            region.bufferOffset = bytes.size();
            region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, mip, f, 1 };
            region.imageExtent = { side, side, 1 };
            regions.push_back(region);
            const auto offset = bytes.size();
            bytes.resize(offset + size_t(len));
            stream.seekg(0);
            if (!stream.read(bytes.data() + offset, std::streamsize(len))) throw std::runtime_error("Cannot read HDR face: " + file);
        }
    }
    constexpr VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, fmt, &props);
    if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ||
        !(props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        throw std::runtime_error("Device cannot linearly sample RGBA32F cubemaps");
    ScopedUploadBuffer upload(ctx.device);
    try {
    ctx.createBuffer(bytes.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, upload.buffer, upload.memory);
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx.device, upload.memory, 0, bytes.size(), 0, &mapped));
    std::memcpy(mapped, bytes.data(), bytes.size());
    vkUnmapMemory(ctx.device, upload.memory);
    mipLevels = levels;
    ctx.createImage(base, base, levels, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, fmt,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, image, memory);
    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, levels, 6);
    auto cmd = ctx.beginSingleTimeCommands();
    vkCmdCopyBufferToImage(cmd, upload.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        uint32_t(regions.size()), regions.data());
    ctx.endSingleTimeCommands(cmd);
    ctx.transitionImageLayout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, levels, 6);
    view = ctx.createImageView(image, fmt, VK_IMAGE_ASPECT_COLOR_BIT, levels, VK_IMAGE_VIEW_TYPE_CUBE, 6);
    VkSamplerCreateInfo ci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    ci.magFilter = ci.minFilter = VK_FILTER_LINEAR;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.maxLod = float(levels - 1);
    VK_CHECK(vkCreateSampler(ctx.device, &ci, nullptr, &sampler));
    } catch (...) {
        destroy(ctx);
        throw;
    }
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
    sampler = VK_NULL_HANDLE;
    view = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    mipLevels = 1;
}
