#version 450
// Phase 7: blend local diffuse GI against deferred diffuse IBL (no additive double count).
layout(set=0,binding=0) uniform sampler2D sceneHDR;
layout(set=0,binding=1) uniform sampler2D normalRough;
layout(set=0,binding=2) uniform sampler2D albedoMetal;
layout(set=0,binding=3) uniform sampler2D sceneDepth;
layout(set=0,binding=4) uniform sampler2D indirectHDR;
layout(set=0,binding=5) uniform sampler2D diffuseIBL;
layout(set=0,binding=6) uniform sampler2D emissiveAO;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
void main() {
    vec3 base = texture(sceneHDR, uv).rgb;
    color = vec4(base, 1.0);
    float depth = texture(sceneDepth, uv).r;
    if (depth <= 0.0 || depth >= 0.999999) return;
    vec4 eo = texture(emissiveAO, uv);
    // Forward/custom geometry is not shaded by deferred IBL: preserve it.
    if (eo.a < 0.0) return;
    vec4 nr = texture(normalRough, uv);
    if (dot(nr.xyz, nr.xyz) < 1e-8) return;
    vec4 material = texture(albedoMetal, uv);
    vec4 gi = texture(indirectHDR, uv);
    vec3 ibl = texture(diffuseIBL, uv).rgb;
    if (any(isnan(gi)) || any(isinf(gi)) || any(isnan(ibl)) || any(isinf(ibl))) return;
    vec3 diffuse = clamp(material.rgb, vec3(0.0), vec3(1.0))
                 * (1.0 - clamp(material.a, 0.0, 1.0));
    float confidence = clamp(gi.a, 0.0, 1.0);
    // A conservative local replacement; confidence zero leaves the original image unchanged.
    float replacement = 0.35 * confidence;
    vec3 localGI = min(max(gi.rgb, vec3(0.0)) * diffuse * 0.31830988618, vec3(1.5));
    color.rgb = max(base - ibl * replacement + localGI * replacement, vec3(0.0));
}
