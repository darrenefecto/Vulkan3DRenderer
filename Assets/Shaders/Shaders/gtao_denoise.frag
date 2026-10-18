#version 450
// Separable depth/normal-aware filter for GTAO.

layout(std140, set = 0, binding = 0) uniform GlobalView {
    mat4 view;
    mat4 proj;
} g;

layout(set = 1, binding = 1) uniform sampler2D gNormalRough;
layout(set = 1, binding = 3) uniform sampler2D gDepth;
layout(set = 1, binding = 5) uniform sampler2D sourceAO;

layout(push_constant) uniform Push {
    mat4 invProj;
    vec4 screen; // width, height, direction x, direction y
} pc;

layout(location = 0) in vec2 uv;
layout(location = 0) out float outAO;

vec3 reconstructViewPosition(ivec2 pixel, float depth, ivec2 size) {
    vec2 texcoord = (vec2(pixel) + vec2(0.5)) / vec2(size);
    vec4 clip = vec4(texcoord * 2.0 - 1.0, depth, 1.0);
    vec4 p = pc.invProj * clip;
    return p.xyz / max(abs(p.w), 1e-6) * sign(p.w);
}

void main() {
    ivec2 size = ivec2(pc.screen.xy);
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - 1);
    float centerDepth = texelFetch(gDepth, pixel, 0).r;
    if (centerDepth >= 0.999999) {
        outAO = 1.0;
        return;
    }

    vec3 centerPos = reconstructViewPosition(pixel, centerDepth, size);
    vec3 centerNormal = normalize(mat3(g.view) * normalize(texelFetch(gNormalRough, pixel, 0).xyz));
    const float kernel[5] = float[5](1.0, 4.0, 6.0, 4.0, 1.0);
    vec2 direction = normalize(pc.screen.zw);
    float sum = 0.0;
    float weightSum = 0.0;

    for (int i = -2; i <= 2; ++i) {
        ivec2 samplePixel = pixel + ivec2(round(direction * float(i)));
        if (any(lessThan(samplePixel, ivec2(0))) || any(greaterThanEqual(samplePixel, size)))
            continue;
        float sampleDepth = texelFetch(gDepth, samplePixel, 0).r;
        if (sampleDepth >= 0.999999)
            continue;

        vec3 samplePos = reconstructViewPosition(samplePixel, sampleDepth, size);
        vec3 sampleNormal = normalize(mat3(g.view) * normalize(texelFetch(gNormalRough, samplePixel, 0).xyz));
        float depthWeight = exp(-abs(samplePos.z - centerPos.z) * 4.0);
        float normalWeight = pow(max(dot(centerNormal, sampleNormal), 0.0), 8.0);
        float weight = kernel[i + 2] * depthWeight * normalWeight;
        sum += texelFetch(sourceAO, samplePixel, 0).r * weight;
        weightSum += weight;
    }

    outAO = weightSum > 1e-5 ? sum / weightSum : texelFetch(sourceAO, pixel, 0).r;
}
