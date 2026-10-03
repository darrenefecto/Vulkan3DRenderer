#include <Vulkan3DRenderer/Mesh.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>

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
        aiProcess_JoinIdenticalVertices | aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph);
    if (!scene || !scene->mRootNode)
        throw std::runtime_error(std::string("Assimp failed: ") + importer.GetErrorString());

    size_t slash = path.find_last_of("/\\");
    directory = (slash == std::string::npos) ? "." : path.substr(0, slash);

    defaultWhite.createSolid(c, 255, 255, 255, 255);

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

void Model::loadMaterial(int index, const void* matPtr, const void* scenePtr) {
    auto* aiMat = (const aiMaterial*)matPtr;
    Material& m = materials[index];
    m.name = aiMat->GetName().C_Str();

    aiColor4D color;
    if (aiGetMaterialColor(aiMat, AI_MATKEY_COLOR_DIFFUSE, &color) == AI_SUCCESS)
        m.data.baseColor = { color.r, color.g, color.b, color.a };
    if (aiGetMaterialColor(aiMat, AI_MATKEY_BASE_COLOR, &color) == AI_SUCCESS)
        m.data.baseColor = { color.r, color.g, color.b, color.a };

    float shininess;
    if (aiGetMaterialFloat(aiMat, AI_MATKEY_SHININESS, &shininess) == AI_SUCCESS && shininess > 0.0f)
        m.data.roughness = glm::clamp(sqrtf(2.0f / (shininess + 2.0f)), 0.05f, 1.0f);

    aiString texPath;
    bool hasTex = aiMat->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == AI_SUCCESS ||
        aiMat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS;
    if (hasTex) {
        const char* p = texPath.C_Str();
        if (p[0] == '*') { // embedded texture
            m.albedo = loadEmbeddedTexture(scenePtr, atoi(p + 1));
        }
        else {
            std::string full = directory + "/" + std::string(p);
            std::replace(full.begin(), full.end(), '\\', '/');
            m.albedo = loadTexture(full);
        }
        if (m.albedo) m.data.hasTexture = 1.0f;
    }
    if (!m.albedo) m.albedo = &defaultWhite;
}

Texture2D* Model::loadTexture(const std::string& path) {
    auto it = textureCache.find(path);
    if (it != textureCache.end()) return it->second.get();
    auto tex = std::make_unique<Texture2D>();
    try {
        tex->createFromFile(*ctx, path);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << " (using white)\n";
        return nullptr;
    }
    Texture2D* raw = tex.get();
    textureCache[path] = std::move(tex);
    return raw;
}

Texture2D* Model::loadEmbeddedTexture(const void* scenePtr, int index) {
    auto* scene = (const aiScene*)scenePtr;
    if (index < 0 || index >= (int)scene->mNumTextures) return nullptr;
    const aiTexture* aiTex = scene->mTextures[index];
    std::string key = std::string("embedded_") + std::to_string(index);
    auto it = textureCache.find(key);
    if (it != textureCache.end()) return it->second.get();

    auto tex = std::make_unique<Texture2D>();
    if (aiTex->mHeight == 0) { // compressed (png/jpg) blob
        tex->createFromMemory(*ctx, aiTex->pcData, (int)aiTex->mWidth);
    }
    else {                   // raw BGRA data
        std::vector<uint8_t> rgba(aiTex->mWidth * aiTex->mHeight * 4);
        for (size_t i = 0; i < aiTex->mWidth * aiTex->mHeight; i++) {
            rgba[i * 4 + 0] = aiTex->pcData[i].r;
            rgba[i * 4 + 1] = aiTex->pcData[i].g;
            rgba[i * 4 + 2] = aiTex->pcData[i].b;
            rgba[i * 4 + 3] = aiTex->pcData[i].a;
        }
        // createFromPixels is private; go through a 1-call path via memory PNG not possible,
        // so just fail gracefully and let the material use white.
        std::cerr << "Uncompressed embedded texture not supported, using white\n";
        return nullptr;
    }
    Texture2D* raw = tex.get();
    textureCache[key] = std::move(tex);
    return raw;
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
        VkDescriptorImageInfo ii{ m.albedo->sampler, m.albedo->view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = m.set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &bi;
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = m.set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &ii;
        vkUpdateDescriptorSets(c.device, 2, writes, 0, nullptr);
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
}