#version 450
// Camera-only screen-space motion vectors in UV units: previousUV-currentUV.
// Sky pixels reproject a world-space ray (w=0): translation must not move the sky.
// Dynamic object and skinned motion require geometry velocity generation.
layout(set = 0, binding = 0) uniform sampler2D sceneDepth;
layout(push_constant) uniform Matrices {
    mat4 inverseCurrentViewProj;
    mat4 previousViewProj;
} pc;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outVelocity;
void main() {
    ivec2 p = clamp(ivec2(gl_FragCoord.xy), ivec2(0), textureSize(sceneDepth, 0) - ivec2(1));
    float depth = texelFetch(sceneDepth, p, 0).r;
    vec2 screenUV = (vec2(p) + 0.5) / vec2(textureSize(sceneDepth, 0));
    if (depth >= 0.999999) {
        // Reconstruct two points along the current camera ray. Subtracting them
        // eliminates camera translation, including for off-center projection.
        // Avoid z=1 here because infinite-far projection may yield world.w=0.
        vec2 ndc = screenUV * 2.0 - 1.0;
        vec4 a = pc.inverseCurrentViewProj * vec4(ndc, 0.0, 1.0);
        vec4 b = pc.inverseCurrentViewProj * vec4(ndc, 0.5, 1.0);
        if (abs(a.w) < 1e-7 || abs(b.w) < 1e-7 ||
            any(isnan(a)) || any(isinf(a)) ||
            any(isnan(b)) || any(isinf(b))) {
            outVelocity = vec4(0.0, 0.0, 1.0, 0.0); return;
        }
        vec3 ray = b.xyz / b.w - a.xyz / a.w;
        if (dot(ray, ray) < 1e-12 || any(isnan(ray)) || any(isinf(ray))) {
            outVelocity = vec4(0.0, 0.0, 1.0, 0.0); return;
        }
        // A direction has homogeneous w=0, so a translation between camera
        // frames cannot introduce bogus sky motion.
        vec4 oldClip = pc.previousViewProj * vec4(normalize(ray), 0.0);
        if (oldClip.w <= 1e-7 || any(isnan(oldClip)) || any(isinf(oldClip))) {
            outVelocity = vec4(0.0, 0.0, 1.0, 0.0); return;
        }
        vec2 oldUV = oldClip.xy / oldClip.w * 0.5 + 0.5;
        outVelocity = vec4(clamp(oldUV - screenUV, vec2(-2.0), vec2(2.0)),
                           1.0, 1.0);
        return;
    }
    vec4 world = pc.inverseCurrentViewProj * vec4(screenUV * 2.0 - 1.0, depth, 1.0);
    if (abs(world.w) < 1e-7 || any(isnan(world)) || any(isinf(world))) {
        outVelocity = vec4(0.0, 0.0, 1.0, 0.0); return;
    }
    world /= world.w;
    vec4 previousClip = pc.previousViewProj * vec4(world.xyz, 1.0);
    if (previousClip.w <= 1e-7 || any(isnan(previousClip)) || any(isinf(previousClip))) { outVelocity = vec4(0.0, 0.0, 1.0, 0.0); return; }
    vec2 previousUV = previousClip.xy / previousClip.w * 0.5 + 0.5;
    outVelocity = vec4(clamp(previousUV - screenUV, vec2(-2.0), vec2(2.0)),
                       clamp(previousClip.z / previousClip.w, 0.0, 1.0), 1.0);
}
