# Custom Forward Shaders

Compile the sample shaders with `compile_custom.bat`. The renderer loads SPIR-V files.

Create a custom object with:

```cpp
Model object(RenderMode::Custom,
    "Assets/Shaders/Custom/custom.vert.spv",
    "Assets/Shaders/Custom/custom.frag.spv");
object.load(ctx, "Assets/Models/model.obj");
object.createDescriptors(ctx, renderer.descriptorPool, renderer.materialLayout);
```

Custom shaders use the renderer's mesh pipeline layout:

- Vertex inputs: locations 0-3 are position, normal, UV, tangent.
- Set 0, binding 0: global uniform buffer; `view` then `proj` are the first two matrices.
- Vertex push constants: one `mat4 model` at offset 0.
- Set 1: material UBO and five texture bindings are available if needed.
- **Set 2, binding 0**: shared, denoised GTAO `sampler2D` (bound automatically). Read using `gl_FragCoord.xy / vec2(textureSize(gtaoTexture, 0))`; multiply **indirect/ambient/IBL** lighting only, not direct light or emission.
- Output: one HDR color attachment (`R16G16B16A16_SFLOAT`).

You own the custom shading logic. Transparent materials still use the internal clustered Forward+ pass.

## GTAO for opaque custom meshes

Opaque `RenderMode::Custom` meshes now participate in `ForwardAOPrepass` with the **standard material G-buffer** (depth + normals). The prepass runs before GTAO, and custom shading runs afterward with `LESS_OR_EQUAL` depth comparison and depth writes disabled. Custom shaders can read the same final GTAO as classic Forward, without allocating descriptor sets themselves. The sample `custom.frag` demonstrates this.

The prepass uses the renderer's regular material alpha cutoff and normal map, **not arbitrary vertex deformation, fragment discard, or modified depth in custom shaders**. If a custom shader changes geometry or coverage, its prepass will need a corresponding custom depth/normal implementation for correct AO and depth testing. Transparent materials still use Forward+ and do not write GTAO depth/normals.

Recompile `custom.frag` with `compile_custom.bat`, then reload/restart the renderer. User-provided custom shaders must explicitly sample the AO texture; descriptor binding alone cannot alter arbitrary shader lighting.
