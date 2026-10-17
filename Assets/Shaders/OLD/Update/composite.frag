#version 450
// Composite: HDR scene color -> tonemapped swapchain image.
// The swapchain image is sRGB, so the hardware applies the linear->sRGB
// conversion; this shader only tonemaps.
layout(set = 0, binding = 0) uniform sampler2D sceneColor;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

// ACES filmic tonemapping (Narkowicz approximation)
vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(sceneColor, uv).rgb;
    outColor = vec4(aces(hdr), 1.0);
}
