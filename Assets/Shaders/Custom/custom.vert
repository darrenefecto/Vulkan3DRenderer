#version 450

layout(std140, set = 0, binding = 0) uniform GlobalView {
    mat4 view;
    mat4 proj;
} g;

layout(push_constant) uniform Push {
    mat4 model;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 uv;
layout(location = 3) out vec4 tangent;

void main() {
    vec4 world = pc.model * vec4(inPos, 1.0);
    worldPos = world.xyz;
    worldNormal = normalize(transpose(inverse(mat3(pc.model))) * inNormal);
    uv = inUV;
    tangent = inTangent;
    gl_Position = g.proj * g.view * world;
}
