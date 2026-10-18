#version 450
// HDR temporal accumulation. Velocity convention: previous UV - current UV.
// Adaptive per-pixel history rejection and variance clipping limit trails.
// Transparent coverage comes from a separate Forward+ MRT attachment.
layout(set = 0, binding = 0) uniform sampler2D currentHDR;
layout(set = 0, binding = 1) uniform sampler2D motionVectors;
layout(set = 0, binding = 2) uniform sampler2D previousHDR;
layout(set = 0, binding = 3) uniform sampler2D currentDepth;
layout(set = 0, binding = 4) uniform sampler2D transparentCoverage;
layout(set = 0, binding = 5) uniform sampler2D transparentVelocity;
layout(push_constant) uniform Settings { vec4 settings; } pc;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

vec3 toYCoCg(vec3 c) {
    return vec3(0.25*c.r + 0.5*c.g + 0.25*c.b,
                0.5*c.r - 0.5*c.b,
               -0.25*c.r + 0.5*c.g - 0.25*c.b);
}
vec3 fromYCoCg(vec3 c) {
    return vec3(c.x+c.y-c.z, c.x+c.z, c.x-c.y-c.z);
}

bool finiteRGB(vec3 v) {
    return !any(isnan(v)) && !any(isinf(v));
}
vec3 safeHDR(vec3 v) {
    return finiteRGB(v) ? max(v, vec3(0.0)) : vec3(0.0);
}

void main() {
    ivec2 size = textureSize(currentHDR, 0);
    ivec2 center = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - ivec2(1));
    vec2 currentUV = (vec2(center) + 0.5) / vec2(size);
    vec3 current = safeHDR(texelFetch(currentHDR, center, 0).rgb);
    float rawDepthNow = texelFetch(currentDepth, center, 0).r;
    float depthNow = (isnan(rawDepthNow) || isinf(rawDepthNow))
        ? 1.0 : clamp(rawDepthNow, 0.0, 1.0);
    if (pc.settings.y < 0.5) { outColor = vec4(current, depthNow); return; }

    // Largest local velocity reduces leaking slow background history across
    // silhouettes. For accurate per-pixel reprojection, use the center velocity
    // except where a neighbour moves much faster.
    vec4 motion = texelFetch(motionVectors, center, 0);
    vec2 velocity = motion.xy;
    float coverage = pc.settings.z < 0.999 ?
        clamp(texelFetch(transparentCoverage, center, 0).r, 0.0, 1.0) : 0.0;
    vec4 translucent = vec4(0.0);
    bool useTransparentMotion = false;
    if (coverage > 0.05) {
        translucent = texelFetch(transparentVelocity, center, 0);
        useTransparentMotion = translucent.w > 0.5 &&
            !any(isnan(translucent.xy)) && !any(isinf(translucent.xy)) &&
            all(lessThan(abs(translucent.xy), vec2(2.0)));
        // A blended velocity is not the motion of either physical surface.
        // Reproject with the foremost transparent layer when it is dominant;
        // otherwise retain opaque reprojection and aggressively reject history.
        if (useTransparentMotion && coverage >= 0.5)
            velocity = translucent.xy;
    }
    // A corrupt velocity must never produce invalid texture coordinates.
    if (any(isnan(velocity)) || any(isinf(velocity)) ||
        any(greaterThan(abs(velocity), vec2(2.0)))) {
        outColor = vec4(current, depthNow); return;
    }
    float centerSpeed = length(velocity);
    vec2 fastest = velocity;
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        ivec2 p = clamp(center + ivec2(x, y), ivec2(0), size - ivec2(1));
        vec2 v = texelFetch(motionVectors, p, 0).xy;
        if (!any(isnan(v)) && !any(isinf(v)) &&
            all(lessThan(abs(v), vec2(2.01))) && length(v) > length(fastest)) fastest = v;
    }
    // Treat neighbouring fast motion as a reason to trust the current frame
    // more, rather than assigning that velocity to the background pixel.
    float edgeMotion = max(0.0, length(fastest) - centerSpeed);
    vec2 previousUV = currentUV + velocity;
    if (any(isnan(previousUV)) || any(isinf(previousUV)) ||
        any(lessThanEqual(previousUV, vec2(0.0))) ||
        any(greaterThanEqual(previousUV, vec2(1.0)))) {
        outColor = vec4(current, depthNow); return;
    }

    vec3 lo = vec3(1e20), hi = vec3(-1e20), mean = vec3(0.0), meanSq = vec3(0.0);
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        ivec2 p = clamp(center + ivec2(x, y), ivec2(0), size - ivec2(1));
        vec3 c = toYCoCg(safeHDR(texelFetch(currentHDR, p, 0).rgb));
        lo = min(lo, c); hi = max(hi, c);
        mean += c; meanSq += c*c;
    }
    mean /= 9.0; meanSq /= 9.0;
    vec3 sigma = sqrt(max(meanSq - mean*mean, vec3(0.0)));
    // Clamp to a 1.25 sigma AABB, bounded by actual neighborhood extrema.
    vec3 clipLo = max(lo, mean - 1.25*sigma);
    vec3 clipHi = min(hi, mean + 1.25*sigma);
    vec3 rawPrevious = texture(previousHDR, previousUV).rgb;
    // Never let invalid HDR history propagate through temporal accumulation.
    if (!finiteRGB(rawPrevious)) {
        outColor = vec4(current, depthNow);
        return;
    }
    vec3 previous = safeHDR(rawPrevious);
    vec3 history = fromYCoCg(clamp(toYCoCg(previous), clipLo, clipHi));
    history = max(history, vec3(0.0));

    // Do not reduce the history weight of the entire frame just because one
    // transparent draw call exists. Instead, estimate local reactivity from
    // changes in un-clipped history. This avoids destabilizing static opaque
    // surfaces elsewhere in the frame. RGB is linear HDR here.
    float feedback = clamp(pc.settings.x, 0.0, 0.97);
    float reactive = 0.0;
    if (pc.settings.z < 0.999) {
        // Luminance-normalized RGB difference also reacts to colored particles
        // whose brightness is similar to their background.
        vec3 delta = abs(current - previous);
        float energy = max(max(max(current.r, current.g), current.b),
                           max(max(previous.r, previous.g), previous.b));
        float colorDifference = max(max(delta.r, delta.g), delta.b) / max(energy, 0.1);
        // Clipped history disagreeing with the original history suggests
        // newly visible or changing high-frequency content.
        float clippingDifference = length(previous - history) /
            max(length(previous), 0.1);
        reactive = max(smoothstep(0.08, 0.45, colorDifference),
                       0.7 * smoothstep(0.04, 0.35, clippingDifference));
        // Scene-wide transparency flag is only permission for this heuristic.
        // The cap is applied proportionally, per pixel, not globally.
        // Per-pixel coverage is produced by the same transparent fragments as
        // the HDR color pass; no global history penalty for opaque pixels.
        reactive = max(reactive * coverage, coverage);
        feedback = min(feedback, mix(1.0, clamp(pc.settings.z, 0.0, 0.97), reactive));
    }
    // Geometric disocclusion: compare the OLD frame depth sampled at the
    // reprojected UV with the depth this surface WOULD have had in that frame.
    // Motion.b is the previous projected Vulkan device depth (not current depth).
    // This is valid for moving models as well as moving cameras.
    // RGB history is bilinearly filtered, but its alpha contains *device depth*.
    // Filtering depth across silhouettes interpolates unrelated surfaces and
    // causes false accept/reject decisions. Fetch the nearest actual depth texel.
    ivec2 historySize = textureSize(previousHDR, 0);
    ivec2 historyPixel = clamp(ivec2(previousUV * vec2(historySize)),
                               ivec2(0), historySize - ivec2(1));
    float previousDepth = texelFetch(previousHDR, historyPixel, 0).a;
    float expectedPreviousDepth = motion.z;
    if (useTransparentMotion && coverage >= 0.5) {
        // History alpha stores OPAQUE depth, while the selected velocity
        // belongs to the transparent surface. These depths are incomparable.
        // Keep only a small amount of color history for transparent layers.
        feedback = min(feedback, 0.20);
    } else if (motion.w < 0.5 || isnan(expectedPreviousDepth) || isinf(expectedPreviousDepth) ||
        isnan(previousDepth) || isinf(previousDepth) ||
        expectedPreviousDepth < 0.0 || expectedPreviousDepth > 1.0) {
        feedback = 0.0;
    } else {
        // Depth is non-linear: use an adaptive tolerance near the far plane.
        float tolerance = 0.0005 + 0.002 * max(expectedPreviousDepth, previousDepth);
        if (abs(previousDepth - expectedPreviousDepth) > tolerance)
            feedback = 0.0;
        if ((depthNow >= 0.99999) != (previousDepth >= 0.99999))
            feedback = 0.0;
    }
    // If transparent fragments were drawn but have no usable layer velocity,
    // do not accumulate opaque reprojection as if it belonged to the glass.
    if (coverage > 0.05 && !useTransparentMotion)
        feedback = min(feedback, mix(0.70, 0.05, coverage));
    feedback *= 1.0 - smoothstep(0.008, 0.08, centerSpeed);
    feedback *= 1.0 - 0.8*smoothstep(0.001, 0.025, edgeMotion);
    // Bright changes (including particles and newly uncovered areas) should
    // not drag stale accumulated radiance across subsequent frames.
    float lumNow = dot(current, vec3(0.2126, 0.7152, 0.0722));
    float lumPrev = dot(previous, vec3(0.2126, 0.7152, 0.0722));
    float relativeChange = abs(lumNow-lumPrev) / max(max(lumNow, lumPrev), 0.1);
    feedback *= 1.0 - 0.85*smoothstep(0.15, 0.8, relativeChange);
    outColor = vec4(mix(current, history, feedback), depthNow);
}
