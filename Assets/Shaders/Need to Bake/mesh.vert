#version 460

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

// Only the prefix of the global UBO is needed here; offsets match the full
// block declared in mesh.frag (std140).
layout(set = 0, binding = 0, std140) uniform Global {
    mat4 view;
    mat4 proj;
} g;

layout(push_constant) uniform Push { mat4 model; } pc;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) out vec4 vTangent;
layout(location = 4) out float vViewDepth;

void main() {
    vec4 world = pc.model * vec4(inPos, 1.0);
    vWorldPos = world.xyz;

    // proper normal/tangent transform (handles non-uniform scale)
    mat3 nm = transpose(inverse(mat3(pc.model)));
    vNormal = nm * inNormal;
    vTangent = vec4(nm * inTangent.xyz, inTangent.w);

    vUV = inUV;
    vec4 viewPos = g.view * world;
    vViewDepth = -viewPos.z;
    gl_Position = g.proj * viewPos;
}
