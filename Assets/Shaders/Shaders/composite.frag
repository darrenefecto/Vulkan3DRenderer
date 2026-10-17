#version 450
// Composite: HDR scene color -> exposure + tonemap -> swapchain image.
// The swapchain image is sRGB, so the hardware applies the linear->sRGB
// conversion; this shader only applies exposure and the tonemap curve.
layout(set = 0, binding = 0) uniform sampler2D sceneColor;

layout(push_constant) uniform Push {
    vec4 params; // x = exposure, y = tonemap operator (0 = none, 1 = Reinhard, 2 = ACES)
} pc;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

// ACES filmic tonemapping (Narkowicz approximation)
vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 tonemap(vec3 hdr, int op) {
    if (op == 1) return hdr / (1.0 + hdr);      // Reinhard
    if (op == 2) return aces(hdr);              // ACES filmic
    return clamp(hdr, 0.0, 1.0);                // none (clamp to LDR)
}

void main() {
    vec3 hdr = texture(sceneColor, uv).rgb * pc.params.x;
    outColor = vec4(tonemap(hdr, int(pc.params.y + 0.5)), 1.0);
}
