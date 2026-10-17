#include <Vulkan3DRenderer/Mesh.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#if __has_include(<assimp/GltfMaterial.h>)
#include <assimp/GltfMaterial.h>
#elif __has_include(<assimp/pbrmaterial.h>)
#include <assimp/pbrmaterial.h>
#endif
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include <cstdlib>

static glm::mat4 toGlm(const aiMatrix4x4& m) {
    return glm::mat4( // aiMatrix is row-major, glm column-major
        m.a1, m.b1, m.c1, m.d1,
        m.a2, m.b2, m.c2, m.d2,
        m.a3, m.b3, m.c3, m.d3,
        m.a4, m.b4, m.c4, m.d4);
}

void Model::load(Context& c, const std::string& path) {
    ctx = &c;
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs |
        aiProcess_CalcTangentSpace |
        aiProcess_JoinIdenticalVertices | aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph);
    if (!scene || !scene->mRootNode)
        throw std::runtime_error(std::string("Assimp failed: ") + importer.GetErrorString());

    size_t slash = path.find_last_of("/\\");
    directory = (slash == std::string::npos) ? "." : path.substr(0, slash);

    defaultWhite.createSolid(c, 255, 255, 255, 255, /*srgb=*/true);
    defaultNormal.createSolid(c, 128, 128, 255, 255, /*srgb=*/false); // flat normal
    defaultBlack.createSolid(c, 0, 0, 0, 255, /*srgb=*/false);

    materials.resize(scene->mNumMaterials);
    for (unsigned i = 0; i < scene->mNumMaterials; i++)
        loadMaterial((int)i, scene->mMaterials[i], scene);

    processNode(scene->mRootNode, scene, glm::mat4(1.0f));

    // Normalize the whole model to ~4 units, centered at origin
    glm::vec3 extent = aabbMax - aabbMin;
    glm::vec3 center = (aabbMax + aabbMin) * 0.5f;
    float scale = 4.0f / std::max({ extent.x, extent.y, extent.z, 1e-6f });
    glm::mat4 normalize = glm::scale(glm::mat4(1.0f), glm::vec3(scale)) *
        glm::translate(glm::mat4(1.0f), -center);
    for (auto& m : meshes) {
        m.transform = normalize * m.transform;
        createBuffers(m);
    }

    std::cout << "Loaded " << path << ": " << meshes.size() << " meshes, "
        << materials.size() << " materials, " << textureCache.size() << " textures\n";
}

void Model::processNode(const void* nodePtr, const void* scenePtr, const glm::mat4& parent) {
    auto* node = (const aiNode*)nodePtr;
    auto* scene = (const aiScene*)scenePtr;
    glm::mat4 world = parent * toGlm(node->mTransformation);
    for (unsigned i = 0; i < node->mNumMeshes; i++)
        processMesh(scene->mMeshes[node->mMeshes[i]], world);
    for (unsigned i = 0; i < node->mNumChildren; i++)
        processNode(node->mChildren[i], scene, world);
}

void Model::processMesh(const void* meshPtr, const glm::mat4& transform) {
    const ::aiMesh* meshData = static_cast<const ::aiMesh*>(meshPtr);

    Mesh mesh;
    mesh.renderMode = defaultRenderMode;
    mesh.customVertexShaderSpv = defaultCustomVertexShaderSpv;
    mesh.customFragmentShaderSpv = defaultCustomFragmentShaderSpv;
    mesh.transform = transform;
    mesh.materialIndex = meshData->mMaterialIndex;
    mesh.vertices.reserve(meshData->mNumVertices);

    for (unsigned i = 0; i < meshData->mNumVertices; i++) {
        Vertex v{};

        v.pos = {
            meshData->mVertices[i].x,
            meshData->mVertices[i].y,
            meshData->mVertices[i].z
        };

        if (meshData->HasNormals()) {
            v.normal = {
                meshData->mNormals[i].x,
                meshData->mNormals[i].y,
                meshData->mNormals[i].z
            };
        }

        if (meshData->HasTextureCoords(0)) {
            v.uv = {
                meshData->mTextureCoords[0][i].x,
                meshData->mTextureCoords[0][i].y
            };
        }

        if (meshData->HasTangentsAndBitangents()) {
            const aiVector3D& t = meshData->mTangents[i];
            const aiVector3D& b = meshData->mBitangents[i];
            const aiVector3D& n = meshData->mNormals[i];
            // bitangent sign: reconstruct handedness so the shader can do B = cross(N,T) * w
            aiVector3D ct = n ^ t; // cross(n, t)
            float w = ((ct * b) < 0.0f) ? -1.0f : 1.0f; // dot(cross(n,t), b)
            v.tangent = { t.x, t.y, t.z, w };
        }
        else {
            v.tangent = { 1.0f, 0.0f, 0.0f, 1.0f };
        }

        mesh.vertices.push_back(v);

        glm::vec4 wp = transform * glm::vec4(v.pos, 1.0f);
        aabbMin = glm::min(aabbMin, glm::vec3(wp));
        aabbMax = glm::max(aabbMax, glm::vec3(wp));
    }

    for (unsigned i = 0; i < meshData->mNumFaces; i++) {
        for (unsigned j = 0; j < meshData->mFaces[i].mNumIndices; j++) {
            mesh.indices.push_back(meshData->mFaces[i].mIndices[j]);
        }
    }

    meshes.push_back(std::move(mesh));
}

// Loads one texture slot; resolves external files and glTF embedded ("*N") textures.
Texture2D* Model::loadTexture(const std::string& path, bool srgb) {
    std::string key = path + (srgb ? "#srgb" : "#linear");
    auto it = textureCache.find(key);
    if (it != textureCache.end()) return it->second.get();
    auto tex = std::make_unique<Texture2D>();
    try {
        tex->createFromFile(*ctx, path, srgb);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << " (using default)\n";
        return nullptr;
    }
    Texture2D* raw = tex.get();
    textureCache[key] = std::move(tex);
    return raw;
}

Texture2D* Model::loadEmbeddedTexture(const void* scenePtr, int index, bool srgb) {
    auto* scene = (const aiScene*)scenePtr;
    if (index < 0 || index >= (int)scene->mNumTextures) return nullptr;
    const aiTexture* aiTex = scene->mTextures[index];
    std::string key = std::string("embedded_") + std::to_string(index) + (srgb ? "#srgb" : "#linear");
    auto it = textureCache.find(key);
    if (it != textureCache.end()) return it->second.get();

    auto tex = std::make_unique<Texture2D>();
    try {
        if (aiTex->mHeight == 0) { // compressed (typically PNG/JPEG) blob
            tex->createFromMemory(*ctx, aiTex->pcData, (int)aiTex->mWidth, srgb);
        }
        else {
            std::cerr << "Uncompressed embedded texture not supported, using default\n";
            return nullptr;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to load embedded texture *" << index << ": "
                  << e.what() << " (using default)\n";
        return nullptr;
    }
    Texture2D* raw = tex.get();
    textureCache[key] = std::move(tex);
    return raw;
}

void Model::loadMaterial(int index, const void* matPtr, const void* scenePtr) {
    auto* aiMat = static_cast<const aiMaterial*>(matPtr);
    Material& m = materials[index];
    m.name = aiMat->GetName().C_Str();

    // Resolve both external files and compressed textures embedded in a .glb.
    // Data maps must stay linear; only color maps are loaded as sRGB.
    auto resolveTexturePath = [&](const aiString& texturePath, bool srgb) -> Texture2D* {
        const char* p = texturePath.C_Str();
        if (p[0] == '*')
            return loadEmbeddedTexture(scenePtr, std::atoi(p + 1), srgb);

        std::string full = directory + "/" + std::string(p);
        std::replace(full.begin(), full.end(), '\\', '/');
        return loadTexture(full, srgb);
    };

    auto resolveTexture = [&](aiTextureType type, unsigned texIndex, bool srgb) -> Texture2D* {
        aiString texturePath;
        if (aiMat->GetTexture(type, texIndex, &texturePath) != AI_SUCCESS)
            return nullptr;
        return resolveTexturePath(texturePath, srgb);
    };

    auto resolveFirstTexture = [&](std::initializer_list<aiTextureType> types,
                                   bool srgb) -> Texture2D* {
        for (aiTextureType type : types) {
            if (Texture2D* texture = resolveTexture(type, 0, srgb))
                return texture;
        }
        return nullptr;
    };

    // --- PBR factors ---
    // Assimp's glTF importer writes the core glTF values to these generic keys.
    aiColor4D color;
    if (aiGetMaterialColor(aiMat, AI_MATKEY_COLOR_DIFFUSE, &color) == AI_SUCCESS)
        m.data.baseColor = { color.r, color.g, color.b, color.a };
    if (aiGetMaterialColor(aiMat, AI_MATKEY_BASE_COLOR, &color) == AI_SUCCESS)
        m.data.baseColor = { color.r, color.g, color.b, color.a }; // glTF baseColorFactor wins

    float f = 0.0f;
    if (aiGetMaterialFloat(aiMat, AI_MATKEY_METALLIC_FACTOR, &f) == AI_SUCCESS)
        m.data.params.x = glm::clamp(f, 0.0f, 1.0f);
    if (aiGetMaterialFloat(aiMat, AI_MATKEY_ROUGHNESS_FACTOR, &f) == AI_SUCCESS)
        m.data.params.y = glm::clamp(f, 0.0f, 1.0f);
    else if (aiGetMaterialFloat(aiMat, AI_MATKEY_SHININESS, &f) == AI_SUCCESS && f > 0.0f)
        m.data.params.y = glm::clamp(sqrtf(2.0f / (f + 2.0f)), 0.05f, 1.0f);

    if (aiGetMaterialColor(aiMat, AI_MATKEY_COLOR_EMISSIVE, &color) == AI_SUCCESS)
        m.data.emissive = { color.r, color.g, color.b, 1.0f };

    // --- Base color / albedo (sRGB) ---
    // Recent Assimp exposes glTF's baseColorTexture as BASE_COLOR; older
    // importers expose it through the legacy DIFFUSE slot as well.
    if (Texture2D* t = resolveFirstTexture(
            { aiTextureType_BASE_COLOR, aiTextureType_DIFFUSE }, true)) {
        m.albedo = t;
        m.data.flags.x = 1.0f;
    }

    // --- Tangent-space normal map (linear) ---
    if (Texture2D* t = resolveFirstTexture(
            { aiTextureType_NORMALS, aiTextureType_NORMAL_CAMERA }, false)) {
        m.normal = t;
        m.data.flags.y = 1.0f;
    }

    // --- glTF metallicRoughnessTexture (linear; G=roughness, B=metallic) ---
    // Older and current Assimp releases expose this texture under the
    // glTF-specific UNKNOWN key. Some releases also expose METALNESS and
    // DIFFUSE_ROUGHNESS aliases, so try those if the key lookup is unavailable.
    Texture2D* mrTexture = nullptr;
#ifdef AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE
    {
        aiString texturePath;
        if (aiMat->GetTexture(
                AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE,
                &texturePath) == AI_SUCCESS) {
            mrTexture = resolveTexturePath(texturePath, false);
        }
    }
#endif
    if (!mrTexture) {
        mrTexture = resolveFirstTexture(
            { aiTextureType_METALNESS, aiTextureType_DIFFUSE_ROUGHNESS }, false);
    }
    // Keep the direct fallback even on Assimp versions without GltfMaterial.h.
    if (!mrTexture)
        mrTexture = resolveTexture(aiTextureType_UNKNOWN, 0, false);
    if (mrTexture) {
        m.metallicRoughness = mrTexture;
        m.data.flags.z = 1.0f;
    }

    // --- Emissive map (sRGB) ---
    if (Texture2D* t = resolveFirstTexture(
            { aiTextureType_EMISSION_COLOR, aiTextureType_EMISSIVE }, true)) {
        m.emissive = t;
        m.data.flags.w = 1.0f;
    }

    // --- Occlusion map (linear; glTF uses the red channel) ---
    if (Texture2D* t = resolveFirstTexture(
            { aiTextureType_AMBIENT_OCCLUSION, aiTextureType_LIGHTMAP }, false)) {
        m.ao = t;
        m.data.params.w = 1.0f;
#ifdef AI_MATKEY_GLTF_TEXTURE_STRENGTH
        // glTF occlusionTexture.strength defaults to 1.0.
        float strength = 1.0f;
        if (aiMat->Get(AI_MATKEY_GLTF_TEXTURE_STRENGTH(aiTextureType_LIGHTMAP, 0), strength)
                == AI_SUCCESS) {
            m.data.params.z = glm::clamp(strength, 0.0f, 1.0f);
        }
#endif
    }

    // Neutral fallback textures keep the factors above authoritative whenever
    // a material does not supply the corresponding map.
    if (!m.albedo) m.albedo = &defaultWhite;
    if (!m.normal) m.normal = &defaultNormal;
    if (!m.metallicRoughness) m.metallicRoughness = &defaultWhite;
    if (!m.emissive) m.emissive = &defaultBlack;
    if (!m.ao) m.ao = &defaultWhite;

    // --- Transparency classification (opaque/deferred vs blended/Forward+) ---
    {
        float opacity = 1.0f;
        if (aiGetMaterialFloat(aiMat, AI_MATKEY_OPACITY, &f) == AI_SUCCESS)
            opacity = f;
        opacity *= m.data.baseColor.a;

        bool blendMode = false;
        float blendFunc = 0.0f;
        if (aiGetMaterialFloat(aiMat, AI_MATKEY_BLEND_FUNC, &blendFunc) == AI_SUCCESS)
            blendMode = (blendFunc != 0.0f); // 0 = aiBlendMode_Default

        bool gltfBlend = false;
#ifdef AI_MATKEY_GLTF_ALPHAMODE
        aiString alphaMode;
        if (aiGetMaterialString(aiMat, AI_MATKEY_GLTF_ALPHAMODE, &alphaMode) == AI_SUCCESS)
            gltfBlend = (std::strcmp(alphaMode.C_Str(), "BLEND") == 0);
#endif
        m.transparent = opacity < 0.999f || blendMode || gltfBlend;
    }

    // These per-material diagnostics are deliberately explicit: if a channel
    // is reported as absent here, it was not imported or resolved before the
    // Vulkan descriptor/shader stage.
    std::cout << "[Material] " << (m.name.empty() ? "<unnamed>" : m.name)
        << " | baseColor=" << (m.data.flags.x > 0.5f ? "texture" : "factor")
        << " | normal=" << (m.data.flags.y > 0.5f ? "texture" : "none")
        << " | metallicRoughness=" << (m.data.flags.z > 0.5f ? "texture" : "factors")
        << " | AO=" << (m.data.params.w > 0.5f ? "texture" : "none")
        << " | emissive=" << (m.data.flags.w > 0.5f ? "texture" : "factor/none")
        << " | metallic=" << m.data.params.x
        << " | roughness=" << m.data.params.y
        << "\n";
}

void Model::createBuffers(Mesh& mesh) {
    VkDeviceSize vbSize = sizeof(Vertex) * mesh.vertices.size();
    VkDeviceSize ibSize = sizeof(uint32_t) * mesh.indices.size();

    VkBuffer staging; VkDeviceMemory stagingMem;
    ctx->createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
    void* data;
    vkMapMemory(ctx->device, stagingMem, 0, vbSize, 0, &data);
    std::memcpy(data, mesh.vertices.data(), vbSize);
    vkUnmapMemory(ctx->device, stagingMem);
    ctx->createBuffer(vbSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mesh.vb, mesh.vbm);
    ctx->copyBuffer(staging, mesh.vb, vbSize);
    vkDestroyBuffer(ctx->device, staging, nullptr);
    vkFreeMemory(ctx->device, stagingMem, nullptr);

    ctx->createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging, stagingMem);
    vkMapMemory(ctx->device, stagingMem, 0, ibSize, 0, &data);
    std::memcpy(data, mesh.indices.data(), ibSize);
    vkUnmapMemory(ctx->device, stagingMem);
    ctx->createBuffer(ibSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mesh.ib, mesh.ibm);
    ctx->copyBuffer(staging, mesh.ib, ibSize);
    vkDestroyBuffer(ctx->device, staging, nullptr);
    vkFreeMemory(ctx->device, stagingMem, nullptr);
}

void Model::createDescriptors(Context& c, VkDescriptorPool pool, VkDescriptorSetLayout layout) {
    std::vector<VkDescriptorSetLayout> layouts(materials.size(), layout);
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = pool;
    ai.descriptorSetCount = (uint32_t)materials.size();
    ai.pSetLayouts = layouts.data();
    std::vector<VkDescriptorSet> sets(materials.size());
    VK_CHECK(vkAllocateDescriptorSets(c.device, &ai, sets.data()));

    for (size_t i = 0; i < materials.size(); i++) {
        Material& m = materials[i];
        m.set = sets[i];
        c.createBuffer(sizeof(MaterialData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m.ubo, m.uboMem);
        void* data;
        vkMapMemory(c.device, m.uboMem, 0, sizeof(MaterialData), 0, &data);
        std::memcpy(data, &m.data, sizeof(MaterialData));
        vkUnmapMemory(c.device, m.uboMem);

        VkDescriptorBufferInfo bi{ m.ubo, 0, sizeof(MaterialData) };
        VkDescriptorImageInfo images[5] = {
            { m.albedo->sampler,            m.albedo->view,            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { m.normal->sampler,            m.normal->view,            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { m.metallicRoughness->sampler, m.metallicRoughness->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { m.emissive->sampler,          m.emissive->view,          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { m.ao->sampler,                m.ao->view,                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        };

        VkWriteDescriptorSet writes[6]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = m.set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &bi;
        for (int t = 0; t < 5; t++) {
            writes[t + 1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[t + 1].dstSet = m.set;
            writes[t + 1].dstBinding = (uint32_t)(t + 1);
            writes[t + 1].descriptorCount = 1;
            writes[t + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[t + 1].pImageInfo = &images[t];
        }
        vkUpdateDescriptorSets(c.device, 6, writes, 0, nullptr);
    }
}

void Model::destroy(Context& c) {
    for (auto& m : meshes) {
        vkDestroyBuffer(c.device, m.vb, nullptr); vkFreeMemory(c.device, m.vbm, nullptr);
        vkDestroyBuffer(c.device, m.ib, nullptr); vkFreeMemory(c.device, m.ibm, nullptr);
    }
    for (auto& mat : materials) {
        vkDestroyBuffer(c.device, mat.ubo, nullptr);
        vkFreeMemory(c.device, mat.uboMem, nullptr);
    }
    for (auto& [path, tex] : textureCache) tex->destroy(c);
    defaultWhite.destroy(c);
    defaultNormal.destroy(c);
    defaultBlack.destroy(c);
}
