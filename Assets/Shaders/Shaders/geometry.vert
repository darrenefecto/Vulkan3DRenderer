#version 450
// Shared geometry vertex shader for the G-Buffer and Forward+ pipelines.
// Interface-compatible with the legacy mesh.vert (same set/push layout).

layout(std140, set = 0, binding = 0) uniform GlobalView {
    mat4 view;
    mat4 proj;
} g;

layout(push_constant) uniform Push { mat4 model; } pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 uv;
layout(location = 3) out vec4 tangent;

void main() {
    vec4 wp = pc.model * vec4(inPos, 1.0);
    worldPos = wp.xyz;
    mat3 nrm = transpose(inverse(mat3(pc.model)));
    normal = normalize(nrm * inNormal);
    tangent = vec4(normalize(nrm * inTangent.xyz), inTangent.w);
    uv = inUV;
    gl_Position = g.proj * g.view * wp;
}
