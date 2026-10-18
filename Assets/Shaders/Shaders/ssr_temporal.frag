#version 450
// Temporal stabilization of SSR radiance and confidence BEFORE IBL replacement.
// The motion buffer encodes previousUV - currentUV; alpha indicates validity.
layout(set=0,binding=0) uniform sampler2D currentReflection;
layout(set=0,binding=1) uniform sampler2D previousReflection;
layout(set=0,binding=2) uniform sampler2D motionTex;
layout(set=0,binding=3) uniform sampler2D depthTex;
// TAA history alpha holds last frame's Vulkan device depth (when enabled).
layout(set=0,binding=4) uniform sampler2D previousDepthHistory;
layout(push_constant) uniform TemporalParams {
    vec4 params;
    mat4 currentToPreviousClip;
} temporal;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
bool finite4(vec4 x) { return !any(isnan(x)) && !any(isinf(x)); }
void main() {
    ivec2 size = textureSize(currentReflection, 0);
    ivec2 px = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size-ivec2(1));
    vec4 current = texelFetch(currentReflection, px, 0);
    if (!finite4(current)) current = vec4(0.0);
    current = max(current, vec4(0.0));
    // First frame, camera cut, or disabled TAA: copy the current SSR sample.
    // Do not sample velocity/depth/history attachments that may be unwritten.
    if (temporal.params.x < 0.5) {
        color = current; return;
    }
    vec4 mv = texelFetch(motionTex, px, 0);
    float depth = texelFetch(depthTex, px, 0).r;
    if (!finite4(mv) || mv.a < 0.5 ||
        isnan(depth) || isinf(depth) || depth >= 0.99999) {
        color = current; return;
    }
    vec2 oldUV = uv + mv.xy;
    if (any(lessThanEqual(oldUV, vec2(0.0))) ||
        any(greaterThanEqual(oldUV, vec2(1.0)))) {
        color = current; return;
    }
    // Reproject current depth into the PREVIOUS camera clip space. Comparing
    // raw depths across camera motion would reject valid history erroneously.
    if (temporal.params.z > 0.5) {
        float previousDepth = texture(previousDepthHistory, oldUV).a;
        vec4 oldClip = temporal.currentToPreviousClip *
            vec4(uv * 2.0 - 1.0, depth, 1.0);
        if (any(isnan(oldClip)) || any(isinf(oldClip)) || oldClip.w <= 0.000001) {
            color = current; return;
        }
        float expectedPreviousDepth = oldClip.z / oldClip.w;
        if (isnan(previousDepth) || isinf(previousDepth) ||
            previousDepth <= 0.0 || previousDepth >= 1.0 ||
            expectedPreviousDepth < 0.0 || expectedPreviousDepth > 1.0 ||
            abs(expectedPreviousDepth - previousDepth) > (0.0015 + 0.004 * expectedPreviousDepth)) {
            color = current; return;
        }
    }
    vec4 history = texture(previousReflection, oldUV);
    if (!finite4(history) || history.a <= 0.001 || current.a <= 0.001) {
        color = current; return;
    }
    // Neighborhood bound rejects large radiance changes (ghost trails).
    vec3 lo = current.rgb, hi = current.rgb;
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        vec3 v = texelFetch(currentReflection, clamp(px+ivec2(x,y), ivec2(0), size-ivec2(1)), 0).rgb;
        if (any(isnan(v)) || any(isinf(v))) continue;
        lo = min(lo, max(v, vec3(0.0)));
        hi = max(hi, max(v, vec3(0.0)));
    }
    // Reject when a hit disappears, confidence changes sharply, or velocity is large.
    float confidence = min(current.a, history.a);
    float consistency = 1.0 - smoothstep(0.08, 0.55, abs(current.a-history.a));
    float velocityFactor = 1.0 - smoothstep(0.005, 0.075, length(mv.xy));
    float weight = clamp(temporal.params.y * confidence * consistency * velocityFactor, 0.0, 0.88);
    vec3 clipped = clamp(max(history.rgb, vec3(0.0)), lo, hi);
    color = vec4(mix(current.rgb, clipped, weight), current.a);
}
