#version 460

// Depth-only pass for sun CSM cascades, spot lights and area lights.
layout(location = 0) in vec3 inPos;

layout(push_constant) uniform Push {
    mat4 lightViewProj;
    mat4 model;
} pc;

void main() {
    gl_Position = pc.lightViewProj * pc.model * vec4(inPos, 1.0);
}
