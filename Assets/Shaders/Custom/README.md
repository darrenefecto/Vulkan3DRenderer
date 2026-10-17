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
- Output: one HDR color attachment (`R16G16B16A16_SFLOAT`).

You own the custom shading logic. Transparent materials still use the internal clustered Forward+ pass.
