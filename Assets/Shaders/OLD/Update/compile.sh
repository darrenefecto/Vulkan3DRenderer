#!/usr/bin/env bash
# Compiles all GLSL shaders to SPIR-V next to the sources.
# Requires glslangValidator (Vulkan SDK). Run from the project root:
#   bash Assets/Shaders/compile.sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
GLSLANG="${GLSLANG:-glslangValidator}"

for f in geometry.vert gbuffer.frag fullscreen.vert deferred_lighting.frag \
         forwardplus.frag composite.frag cluster_build.comp; do
    echo "  $f"
    "$GLSLANG" -V --target-env vulkan1.3 -I"$DIR" "$DIR/$f" -o "$DIR/$f.spv"
done
echo "Done. (.spv files written to $DIR)"
