#version 450

layout(push_constant) uniform Push { mat4 model; } push;

layout(set = 0, binding = 0) uniform Global {
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
} g;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 uv;

void main() {
    vec4 wp = push.model * vec4(inPos, 1.0);
    worldPos = wp.xyz;
    normal = mat3(push.model) * inNormal; // ok for uniform scale (we normalize the model)
    uv = inUV;
    gl_Position = g.proj * g.view * wp;
}