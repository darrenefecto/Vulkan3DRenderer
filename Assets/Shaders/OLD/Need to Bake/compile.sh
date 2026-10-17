#!/usr/bin/env bash
# Compiles all GLSL shaders to SPIR-V next to this script (Assets/Shaders/).
# Requires glslc from the Vulkan SDK.
set -e
cd "$(dirname "$0")"

for src in mesh.vert mesh.frag skybox.vert skybox.frag \
           shadow2d.vert shadow2d.frag shadow_cube.vert shadow_cube.frag; do
    echo "glslc $src -> $src.spv"
    glslc --target-env=vulkan1.3 -O "$src" -o "$src.spv"
done
echo "Done."
