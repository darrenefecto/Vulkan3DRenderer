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

    static VkVertexInputBindingDescription binding() {
        return { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    }
    static std::array<VkVertexInputAttributeDescription, 3> attributes() {
        return { {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
            { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv) },
        } };
    }
};

// Must match the std140 MaterialUBO in mesh.frag (32 bytes)
struct MaterialData {
    glm::vec4 baseColor{ 1.0f };
    float hasTexture = 0.0f;
    float roughness = 0.8f;
    float metallic = 0.0f;
    float _pad = 0.0f;
};

struct Material {
    std::string name = "default";
    MaterialData data{};
    Texture2D* albedo = nullptr; // points into texture cache (or defaultWhite)
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
    Texture2D defaultWhite;
    glm::vec3 aabbMin{ 1e30f }, aabbMax{ -1e30f };

    void processNode(const void* node, const void* scene, const glm::mat4& parent);
    void processMesh(const void* mesh, const glm::mat4& transform);
    void loadMaterial(int index, const void* aiMat, const void* scene);
    Texture2D* loadTexture(const std::string& path);
    Texture2D* loadEmbeddedTexture(const void* scene, int index);
    void createBuffers(Mesh& mesh);
};