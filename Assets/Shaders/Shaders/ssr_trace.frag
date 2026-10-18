#version 450
// SSR trace (standalone integration stage). Source HDR MUST be a separate, fully
// resolved pre-SSR image. Never sample the color attachment being written.
// Vulkan NDC depth is [0,1]. G-buffer normals are world-space, roughness in alpha.
layout(std140, set=0, binding=0) uniform GlobalView { mat4 view; mat4 proj; } g;
layout(set=1, binding=0) uniform sampler2D sourceHDR;
layout(set=1, binding=1) uniform sampler2D gNormalRough;
layout(set=1, binding=3) uniform sampler2D gDepth;
layout(push_constant) uniform SSRSettings {
    mat4 invProj;
    vec4 trace; // x=max distance, y=thickness, z=stride, w=roughness cutoff
} pc;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 reflection; // linear HDR radiance + confidence

bool finite1(float x) { return !isnan(x) && !isinf(x); }
bool finite3(vec3 x) { return !any(isnan(x)) && !any(isinf(x)); }
bool validUV(vec2 x) { return all(greaterThan(x, vec2(0.001))) && all(lessThan(x, vec2(0.999))); }
vec3 reconstructView(vec2 p, float d) {
    vec4 h = pc.invProj * vec4(p * 2.0 - 1.0, d, 1.0);
    return h.xyz / h.w;
}
vec2 projectUV(vec3 p) {
    vec4 h = g.proj * vec4(p, 1.0);
    if (h.w <= 1e-5 || !finite1(h.w)) return vec2(-1.0);
    return h.xy / h.w * 0.5 + 0.5;
}
void main() {
    reflection = vec4(0.0);
    float d = texture(gDepth, uv).r;
    if (!finite1(d) || d <= 0.0 || d >= 0.999999) return;
    vec4 nr = texture(gNormalRough, uv);
    if (!finite3(nr.xyz) || !finite1(nr.a) || dot(nr.xyz, nr.xyz) < 1e-8) return;
    float cutoff = clamp(pc.trace.w, 0.02, 1.0);
    float roughness = clamp(nr.a, 0.0, 1.0);
    if (roughness >= cutoff) return;
    vec3 p = reconstructView(uv, d);
    if (!finite3(p) || dot(p, p) < 1e-8) return;
    vec3 n = normalize(mat3(g.view) * normalize(nr.xyz));
    // An outward-facing surface should point towards the camera in view space.
    // Backfaces and grazing reflections are unreliable in screen space.
    float facing = dot(n, normalize(-p));
    if (facing <= 0.03) return;
    vec3 ray = normalize(reflect(normalize(p), n));
    if (!finite3(ray) || dot(ray, n) <= 0.0) return;
    float stride = max(pc.trace.z, 0.01);
    float maxDistance = max(pc.trace.x, stride);
    float thickness = max(pc.trace.y, 0.005);
    // Start off the reflector to avoid self-intersection.
    vec3 start = p + n * max(thickness, 0.02);
    vec3 previous = start;
    float previousDelta = -1.0;
    // A hit must remain on a valid front-facing G-buffer surface.
    for (int i = 1; i <= 96; ++i) {
        float t = float(i) * stride;
        if (t > maxDistance) break;
        vec3 probe = start + ray * t;
        vec2 coord = projectUV(probe);
        if (!validUV(coord)) break;
        float sd = texture(gDepth, coord).r;
        if (!finite1(sd) || sd <= 0.0 || sd >= 0.999999) {
            previous = probe;
            previousDelta = -1.0;
            continue;
        }
        vec3 surface = reconstructView(coord, sd);
        if (!finite3(surface)) break;
        // Camera faces -Z: positive delta means probe is behind visible surface.
        float delta = surface.z - probe.z;
        // Require a front-to-back crossing; otherwise thick objects can produce
        // spurious reflections without a genuine screen-space intersection.
        if (previousDelta < 0.0 && delta >= 0.0 && delta < thickness) {
            vec3 low = previous;
            vec3 high = probe;
            for (int j = 0; j < 5; ++j) {
                vec3 mid = (low + high) * 0.5;
                vec2 midUV = projectUV(mid);
                if (!validUV(midUV)) break;
                float md = texture(gDepth, midUV).r;
                if (!finite1(md) || md <= 0.0 || md >= 0.999999) { low = mid; continue; }
                vec3 ms = reconstructView(midUV, md);
                if (!finite3(ms)) break;
                if (ms.z - mid.z >= 0.0) high = mid;
                else low = mid;
            }
            vec2 hitUV = projectUV(high);
            if (!validUV(hitUV)) return;
            // Reject depth discontinuities and untrustworthy normals at the hit.
            // Otherwise a ray may reflect sky/foreground color across a silhouette.
            float hitDepth = texture(gDepth, hitUV).r;
            if (!finite1(hitDepth) || hitDepth >= 0.999999 || hitDepth <= 0.0) return;
            vec3 hitSurface = reconstructView(hitUV, hitDepth);
            if (!finite3(hitSurface) || abs(hitSurface.z - high.z) > thickness * 1.5) return;
            vec3 hitNormalWS = texture(gNormalRough, hitUV).xyz;
            if (!finite3(hitNormalWS) || dot(hitNormalWS,hitNormalWS) < 1e-8) return;
            vec3 hitNormalVS = normalize(mat3(g.view) * normalize(hitNormalWS));
            if (dot(hitNormalVS, normalize(-hitSurface)) <= 0.02) return;
            vec3 hdr = texture(sourceHDR, hitUV).rgb;
            if (!finite3(hdr)) return;
            float edge = clamp(min(min(hitUV.x, hitUV.y), min(1.0-hitUV.x, 1.0-hitUV.y)) * 12.0, 0.0, 1.0);
            float distanceFade = 1.0 - clamp(t / maxDistance, 0.0, 1.0);
            float roughFade = 1.0 - smoothstep(cutoff * 0.5, cutoff, roughness);
            float grazingFade = smoothstep(0.03, 0.2, facing);
            // Down-weight unstable hits near a depth discontinuity. Sampling a
            // small cross in screen space prevents obvious silhouette leaks.
            vec2 texel = 1.0 / vec2(textureSize(gDepth, 0));
            float depthSupport = 1.0;
            for (int k = 0; k < 4; ++k) {
                vec2 offset = (k == 0 ? vec2(1,0) : k == 1 ? vec2(-1,0) :
                               k == 2 ? vec2(0,1) : vec2(0,-1)) * texel;
                vec2 q = hitUV + offset;
                if (!validUV(q)) { depthSupport *= 0.5; continue; }
                float nd = texture(gDepth, q).r;
                if (!finite1(nd) || nd <= 0.0 || nd >= 0.999999) {
                    depthSupport *= 0.5; continue;
                }
                vec3 neighbor = reconstructView(q, nd);
                if (!finite3(neighbor) || abs(neighbor.z - hitSurface.z) > max(thickness * 4.0, 0.12))
                    depthSupport *= 0.65;
            }
            reflection = vec4(max(hdr, vec3(0.0)), edge * distanceFade * roughFade * grazingFade * depthSupport);
            return;
        }
        previous = probe;
        previousDelta = delta;
    }
}
