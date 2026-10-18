#version 450
// Temporal stabilization of SSGI radiance and confidence BEFORE IBL replacement.
// The motion buffer encodes previousUV - currentUV; alpha indicates validity.
layout(set=0,binding=0) uniform sampler2D currentIndirect;
layout(set=0,binding=1) uniform sampler2D previousIndirect;
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
    ivec2 size = textureSize(currentIndirect, 0);
    ivec2 px = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size-ivec2(1));
    vec4 current = texelFetch(currentIndirect, px, 0);
    if (!finite4(current)) current = vec4(0.0);
    current = max(current, vec4(0.0));
    // First frame, camera cut, or disabled TAA: copy the current SSGI sample.
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
    vec4 history = texture(previousIndirect, oldUV);
    if (!finite4(history) || history.a <= 0.001) {
        color = current; return;
    }
    // Neighborhood bound rejects large radiance changes (ghost trails).
    vec3 lo = current.rgb, hi = current.rgb;
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        vec3 v = texelFetch(currentIndirect, clamp(px+ivec2(x,y), ivec2(0), size-ivec2(1)), 0).rgb;
        if (any(isnan(v)) || any(isinf(v))) continue;
        lo = min(lo, max(v, vec3(0.0)));
        hi = max(hi, max(v, vec3(0.0)));
    }
    // Reject when a hit disappears, confidence changes sharply, or velocity is large.
    float currentConfidence = clamp(current.a, 0.0, 1.0);
    float historyConfidence = clamp(history.a, 0.0, 1.0);
    float consistency = 1.0 - smoothstep(0.12, 0.65, abs(currentConfidence-historyConfidence));
    float velocityFactor = 1.0 - smoothstep(0.002, 0.06, length(mv.xy));
    // Stable high-confidence hits accumulate strongly; rapid motion or a
    // confidence discontinuity falls back to the fresh ray result.
    float weight = clamp(temporal.params.y * historyConfidence * consistency *
                         velocityFactor, 0.0, 0.88);
    // A missing current hit may be stochastic rather than disocclusion, but
    // must not retain a bright historical bounce indefinitely.
    if (currentConfidence < 0.001) weight *= 0.35;
    // Widen a degenerate neighborhood slightly to retain small, valid GI
    // changes without allowing unbounded history ghosting.
    vec3 margin = max((hi - lo) * 0.15, vec3(0.002));
    vec3 clipped = clamp(max(history.rgb, vec3(0.0)), max(lo - margin, vec3(0.0)), hi + margin);
    float resolvedConfidence = mix(currentConfidence, historyConfidence, weight);
    color = vec4(mix(current.rgb, clipped, weight), clamp(resolvedConfidence, 0.0, 1.0));
}
