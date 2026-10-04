#pragma once
#include "VulkanContext.h"
#include "Texture.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <unordered_map>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec4 tangent; // xyz = tangent, w = bitangent sign

    static VkVertexInputBindingDescription binding() {
        return { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    }
    static std::array<VkVertexInputAttributeDescription, 4> attributes() {
        return { {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex, pos) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex, normal) },
            { 2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(Vertex, uv) },
            { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent) },
        } };
    }
};

// Must match the std140 MaterialUBO in mesh.frag
struct MaterialData {
    glm::vec4 baseColor{ 1.0f };   // rgba base color factor
    glm::vec4 emissive{ 0.0f };    // rgb emissive factor
    // x = metallic, y = roughness, z = ao strength, w = has AO map
    glm::vec4 params{ 0.0f, 0.8f, 1.0f, 0.0f };
    // x = has albedo, y = has normal, z = has metallic-roughness, w = has emissive
    glm::vec4 flags{ 0.0f };
};

// PBR material: factors in `data`, optional textures for each channel.
struct Material {
    std::string name = "default";
    MaterialData data{};
    Texture2D* albedo = nullptr;             // sRGB
    Texture2D* normal = nullptr;             // linear, tangent-space
    Texture2D* metallicRoughness = nullptr;  // linear, g = roughness, b = metallic (glTF)
    Texture2D* emissive = nullptr;           // sRGB
    Texture2D* ao = nullptr;                 // linear, r channel
    VkBuffer ubo = VK_NULL_HANDLE;
    VkDeviceMemory uboMem = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    glm::mat4 transform{ 1.0f };
    uint32_t materialIndex = 0;
    VkBuffer vb = VK_NULL_HANDLE, ib = VK_NULL_HANDLE;
    VkDeviceMemory vbm = VK_NULL_HANDLE, ibm = VK_NULL_HANDLE;
};

class Model {
public:
    std::vector<Mesh> meshes;
    std::vector<Material> materials;

    void load(Context& ctx, const std::string& path);
    void createDescriptors(Context& ctx, VkDescriptorPool pool, VkDescriptorSetLayout layout);
    void destroy(Context& ctx);

private:
    Context* ctx = nullptr;
    std::string directory;
    std::unordered_map<std::string, std::unique_ptr<Texture2D>> textureCache;
    Texture2D defaultWhite;   // albedo / AO fallback
    Texture2D defaultNormal;  // flat (128,128,255)
    Texture2D defaultBlack;   // emissive fallback
    glm::vec3 aabbMin{ 1e30f }, aabbMax{ -1e30f };

    void processNode(const void* node, const void* scene, const glm::mat4& parent);
    void processMesh(const void* mesh, const glm::mat4& transform);
    void loadMaterial(int index, const void* aiMat, const void* scene);
    Texture2D* loadTexture(const std::string& path, bool srgb);
    Texture2D* loadEmbeddedTexture(const void* scene, int index, bool srgb);
    void createBuffers(Mesh& mesh);
};
