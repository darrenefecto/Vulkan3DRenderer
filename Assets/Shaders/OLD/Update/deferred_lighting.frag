#version 450
#extension GL_GOOGLE_include_directive : require
// Deferred lighting: reconstructs the surface from the G-Buffer and shades it
// with the full light rig (sun+CSM, points, spots, areas, IBL) + skybox.

#include "pbr_common.glsl"

layout(set = 1, binding = 0) uniform sampler2D gAlbedoMetallic;
layout(set = 1, binding = 1) uniform sampler2D gNormalRough;
layout(set = 1, binding = 2) uniform sampler2D gEmissiveAO;
layout(set = 1, binding = 3) uniform sampler2D gDepth;

layout(push_constant) uniform Push {
    mat4 invViewProj;
    vec4 screen; // w, h, 1/w, 1/h
} pc;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

vec3 worldPosFromDepth(float depth, vec2 texcoord) {
    vec4 ndc = vec4(texcoord * 2.0 - 1.0, depth, 1.0);
    vec4 wp = pc.invViewProj * ndc;
    return wp.xyz / wp.w;
}

void main() {
    float depth = texture(gDepth, uv).r;

    vec4 nr = texture(gNormalRough, uv);
    vec4 am = texture(gAlbedoMetallic, uv);
    vec4 eo = texture(gEmissiveAO, uv);

    vec3 worldPos = worldPosFromDepth(depth, uv);
    vec3 viewDir = normalize(g.camPos.xyz - worldPos);

    // sky where no geometry was written
    if (depth >= 0.99999) {
        vec3 dir = normalize(worldPos - g.camPos.xyz);
        outColor = vec4(textureLod(envMap, dir, 0.0).rgb, 1.0);
        return;
    }

    vec3 N = normalize(nr.xyz);
    float roughness = nr.w;
    vec3 albedo = am.rgb;
    float metallic = am.a;
    vec3 emissive = eo.rgb;
    float ao = eo.a;

    float viewDepth = -(g.view * vec4(worldPos, 1.0)).z;

    vec3 color = vec3(0.0);
    color += evalSun(N, viewDir, worldPos, viewDepth, albedo, metallic, roughness);
    for (int i = 0; i < g.counts.x; i++)
        color += evalPoint(i, N, viewDir, worldPos, albedo, metallic, roughness);
    for (int i = 0; i < g.counts.y; i++)
        color += evalSpot(i, N, viewDir, worldPos, albedo, metallic, roughness);
    for (int i = 0; i < g.counts.z; i++)
        color += evalArea(i, N, viewDir, worldPos, albedo, metallic, roughness);
    color += evalIBL(N, viewDir, albedo, metallic, roughness, ao);
    color += emissive;

    outColor = vec4(color, 1.0);
}
