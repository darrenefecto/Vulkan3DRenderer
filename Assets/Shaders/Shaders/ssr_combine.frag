#version 450
// Replace the deferred indirect specular term, never the full HDR scene.
// Alpha of iblSpecular is a coverage flag: forward paths do not populate it.
layout(set=0,binding=0) uniform sampler2D sceneHDR;
layout(set=0,binding=1) uniform sampler2D normalRough;
layout(set=0,binding=2) uniform sampler2D albedoMetallic;
layout(set=0,binding=3) uniform sampler2D sceneDepth;
layout(set=0,binding=4) uniform sampler2D reflectionHDR;
layout(set=0,binding=5) uniform sampler2D iblSpecular;
layout(set=0,binding=6) uniform sampler2D emissiveAO;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
bool finite3(vec3 x) { return !any(isnan(x)) && !any(isinf(x)); }
void main() {
    ivec2 px = clamp(ivec2(gl_FragCoord.xy), ivec2(0), textureSize(sceneHDR,0)-ivec2(1));
    vec3 base = texelFetch(sceneHDR,px,0).rgb;
    if (!finite3(base)) { color=vec4(0.0,0.0,0.0,1.0); return; }
    vec4 hit = texelFetch(reflectionHDR,px,0);
    vec4 spec = texelFetch(iblSpecular,px,0);
    vec4 nr = texelFetch(normalRough,px,0);
    vec4 mat = texelFetch(albedoMetallic,px,0);
    float depth = texelFetch(sceneDepth,px,0).r;
    // Forward/custom pixels carry a negative AO marker in the G-buffer.
    // Their specular lighting was not authored by deferred MRT.
    if (texelFetch(emissiveAO,px,0).a < 0.0 ||
        depth >= 0.99999 || isnan(depth) || isinf(depth) ||
        spec.a < 0.5 || !finite3(spec.rgb) || !finite3(hit.rgb) ||
        isnan(hit.a) || isinf(hit.a) || isnan(nr.a) || isinf(nr.a) ||
        !finite3(mat.rgb) || isnan(mat.a) || isinf(mat.a)) {
        color=vec4(base,1.0); return;
    }
    float roughness = clamp(nr.a, 0.0, 1.0);
    float confidence = clamp(hit.a, 0.0, 1.0) *
                       (1.0 - smoothstep(0.15, 0.75, roughness));
    float metallic = clamp(mat.a, 0.0, 1.0);
    vec3 f0 = mix(vec3(0.04), clamp(mat.rgb,0.0,1.0), metallic);
    // SSR stores incident HDR radiance. The approximate Fresnel term gives
    // it a material response; a full split-sum BRDF match is future work.
    vec3 ssrSpecular = max(hit.rgb, vec3(0.0)) * f0;
    vec3 replacement = mix(max(spec.rgb,vec3(0.0)), ssrSpecular, confidence);
    color = vec4(max(base - max(spec.rgb,vec3(0.0)) + replacement, vec3(0.0)), 1.0);
}
