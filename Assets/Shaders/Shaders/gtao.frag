#version 450
// Full-resolution screen-space horizon AO.

layout(std140, set = 0, binding = 0) uniform GlobalView {
    mat4 view;
    mat4 proj;
} g;

layout(set = 1, binding = 1) uniform sampler2D gNormalRough;
layout(set = 1, binding = 3) uniform sampler2D gDepth;

layout(push_constant) uniform Push {
    mat4 invProj;
    vec4 screen; // width, height, world radius, strength
} pc;

layout(location = 0) in vec2 uv;
layout(location = 0) out float outAO;

const int DIRECTION_COUNT = 8;
const int STEPS_PER_DIRECTION = 4;
const float PI = 3.14159265359;

vec3 reconstructViewPosition(ivec2 pixel, float depth, ivec2 size) {
    vec2 texcoord = (vec2(pixel) + vec2(0.5)) / vec2(size);
    vec4 clip = vec4(texcoord * 2.0 - 1.0, depth, 1.0);
    vec4 viewPosition = pc.invProj * clip;
    return viewPosition.xyz / max(abs(viewPosition.w), 1e-6) * sign(viewPosition.w);
}

float hashPixel(ivec2 pixel) {
    vec2 p = vec2(pixel);
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    ivec2 size = ivec2(pc.screen.xy);
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - 1);
    float centerDepth = texelFetch(gDepth, pixel, 0).r;
    if (centerDepth >= 0.999999) {
        outAO = 1.0;
        return;
    }

    vec3 P = reconstructViewPosition(pixel, centerDepth, size);
    vec3 worldNormal = normalize(texelFetch(gNormalRough, pixel, 0).xyz);
    vec3 N = normalize(mat3(g.view) * worldNormal);
    float radius = clamp(pc.screen.z, 0.05, 5.0);
    float radiusPixels = clamp(radius * abs(g.proj[1][1]) * float(size.y)
        / max(2.0 * abs(P.z), 0.05), 2.0, 64.0);
    float rotation = hashPixel(pixel) * (2.0 * PI);

    float occlusion = 0.0;
    for (int directionIndex = 0; directionIndex < DIRECTION_COUNT; ++directionIndex) {
        float angle = rotation + (float(directionIndex) / float(DIRECTION_COUNT)) * 2.0 * PI;
        vec2 direction = vec2(cos(angle), sin(angle));
        float horizonOcclusion = 0.0;

        for (int stepIndex = 1; stepIndex <= STEPS_PER_DIRECTION; ++stepIndex) {
            float stepFraction = float(stepIndex) / float(STEPS_PER_DIRECTION);
            ivec2 samplePixel = pixel + ivec2(round(direction * radiusPixels * stepFraction));
            if (any(lessThan(samplePixel, ivec2(0))) || any(greaterThanEqual(samplePixel, size)))
                continue;

            float sampleDepth = texelFetch(gDepth, samplePixel, 0).r;
            if (sampleDepth >= 0.999999)
                continue;

            vec3 Q = reconstructViewPosition(samplePixel, sampleDepth, size);
            vec3 delta = Q - P;
            float distanceToSample = length(delta);
            if (distanceToSample < 1e-4 || distanceToSample >= radius)
                continue;

            float hemisphere = max(dot(N, delta / distanceToSample) - 0.04, 0.0);
            float falloff = 1.0 - smoothstep(radius * 0.15, radius, distanceToSample);
            horizonOcclusion = max(horizonOcclusion, hemisphere * falloff);
        }
        occlusion += horizonOcclusion;
    }

    float visibility = 1.0 - clamp(occlusion / float(DIRECTION_COUNT) * 1.8 * pc.screen.w, 0.0, 0.95);
    outAO = clamp(visibility, 0.05, 1.0);
}
