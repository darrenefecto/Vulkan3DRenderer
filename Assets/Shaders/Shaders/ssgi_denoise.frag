#version 450
// Confidence-aware joint bilateral SSGI filter (5x5). Operates on temporal
// radiance, not the history buffer, to avoid recursively blurring feedback.
// Descriptor bindings match the existing SSR combine layout.
layout(set=0,binding=0) uniform sampler2D temporalGI;
layout(set=0,binding=1) uniform sampler2D normalRough;
layout(set=0,binding=3) uniform sampler2D depthTex;
layout(push_constant) uniform DenoiseParams { mat4 inverseProjection; } pc;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;

float viewDepth(ivec2 pixel, float deviceDepth, ivec2 dimensions) {
    vec2 pixelUV = (vec2(pixel) + vec2(0.5)) / vec2(dimensions);
    vec4 view = pc.inverseProjection * vec4(pixelUV * 2.0 - 1.0, deviceDepth, 1.0);
    if (abs(view.w) < 1e-7 || any(isnan(view)) || any(isinf(view))) return -1.0;
    return abs(view.z / view.w);
}
bool valid4(vec4 v) { return !any(isnan(v)) && !any(isinf(v)); }
void main() {
    ivec2 sz = textureSize(temporalGI, 0);
    ivec2 p = clamp(ivec2(gl_FragCoord.xy), ivec2(0), sz-ivec2(1));
    vec4 center = texelFetch(temporalGI, p, 0);
    if (!valid4(center)) center = vec4(0.0);
    center = max(center, vec4(0.0));
    float z0 = texelFetch(depthTex, p, 0).r;
    vec3 n0 = texelFetch(normalRough, p, 0).xyz;
    float n0len = length(n0);
    if (z0 <= 0.0 || z0 >= 0.999999 || n0len < 0.0001 ||
        isnan(z0) || isinf(z0) || any(isnan(n0)) || any(isinf(n0))) {
        color = center; return;
    }
    n0 /= n0len;
    float viewZ0 = viewDepth(p, z0, sz);
    if (viewZ0 <= 0.0 || isnan(viewZ0) || isinf(viewZ0)) { color = center; return; }
    vec3 sum = vec3(0.0);
    float total = 0.0;
    float confidenceSum = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            ivec2 q = clamp(p + ivec2(x, y), ivec2(0), sz-ivec2(1));
            vec4 sampleGI = texelFetch(temporalGI, q, 0);
            vec3 n = texelFetch(normalRough, q, 0).xyz;
            float z = texelFetch(depthTex, q, 0).r;
            if (!valid4(sampleGI) || sampleGI.a <= 0.0001 ||
                any(isnan(n)) || any(isinf(n)) || isnan(z) || isinf(z) ||
                z <= 0.0 || z >= 0.999999 || dot(n,n) < 0.000001) continue;
            n = normalize(n);
            float spatial = exp(-0.5 * float(x*x + y*y) / 2.0);
            // Relative depth difference: strong rejection at disocclusions.
            float viewZ = viewDepth(q, z, sz);
            if (viewZ <= 0.0 || isnan(viewZ) || isinf(viewZ)) continue;
            // Scale tolerance by physical view-space distance, not non-linear device Z.
            float depthWeight = exp(-abs(viewZ - viewZ0) / max(0.02, 0.035 * viewZ0));
            float normalWeight = pow(max(dot(n0, n), 0.0), 24.0);
            float confidence = clamp(sampleGI.a, 0.0, 1.0);
            float w = spatial * depthWeight * normalWeight * confidence;
            sum += max(sampleGI.rgb, vec3(0.0)) * w;
            confidenceSum += confidence * w;
            total += w;
        }
    }
    if (total <= 0.000001) { color = center; return; }
    color = vec4(sum / total, clamp(confidenceSum / total, 0.0, 1.0));
}
