#version 450

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 tangent;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 n = normalize(worldNormal);
    float stripe = 0.65 + 0.35 * sin(uv.x * 32.0);
    outColor = vec4(abs(n) * stripe, 1.0);
}
