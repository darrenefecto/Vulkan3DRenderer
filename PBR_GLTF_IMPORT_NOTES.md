# glTF / GLB PBR material import notes

`Mesh.cpp` now resolves the standard glTF metallic-roughness material channels through Assimp before creating the material descriptor sets.

Supported material inputs:

- `baseColorFactor` and `baseColorTexture` (sRGB)
- `metallicFactor` and `roughnessFactor`
- packed `metallicRoughnessTexture` (linear; G = roughness, B = metallic)
- `normalTexture` (linear, tangent-space)
- `occlusionTexture` (linear, red channel; occlusion strength is read when Assimp exposes it)
- `emissiveFactor` and `emissiveTexture` (emissive texture is sRGB)
- embedded textures in `.glb` and external image files referenced by `.gltf`

The existing fragment shaders already perform metallic-roughness PBR shading. This change repairs material import and adds one console line per material so you can confirm which channels were found. No shader recompilation should be needed for this patch because the shader sources and their active SPIR-V outputs already include the PBR sampling path.

Build the Visual Studio project and launch it with a `.glb`/`.gltf` model path. The console should print lines similar to:

```text
[Material] PaintedMetal | baseColor=texture | normal=texture | metallicRoughness=texture | AO=texture | emissive=factor/none | metallic=1 | roughness=0.4
```

Texture features not yet covered by the renderer include arbitrary `TEXCOORD_n` selection / `KHR_texture_transform`, full alpha-mask cutoffs from `alphaCutoff`, and advanced glTF material extensions such as clearcoat, sheen, transmission and volume.
