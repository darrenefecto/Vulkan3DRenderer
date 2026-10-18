#version 450
#extension GL_GOOGLE_include_directive : require
// Deferred lighting: reconstructs the surface from the G-Buffer and shades it
// with the full light rig (sun+CSM, points, spots, areas, IBL) + skybox.

#include "pbr_common.glsl"

layout(set = 1, binding = 0) uniform sampler2D gAlbedoMetallic;
layout(set = 1, binding = 1) uniform sampler2D gNormalRough;
layout(set = 1, binding = 2) uniform sampler2D gEmissiveAO;
layout(set = 1, binding = 3) uniform sampler2D gDepth;
layout(set = 1, binding = 4) uniform sampler2D gtaoTexture;

layout(push_constant) uniform Push {
    mat4 invViewProj;
    vec4 screen; // w, h, 1/w, 1/h
} pc;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outIBLSpecular;
layout(location = 2) out vec4 outIBLDiffuse;

vec3 worldPosFromDepth(float depth, vec2 texcoord) {
    vec4 ndc = vec4(texcoord * 2.0 - 1.0, depth, 1.0);
    vec4 wp = pc.invViewProj * ndc;
    return wp.xyz / wp.w;
}

void main() {
    // All G-buffer attachments must use the same *integer pixel*.
    // Linear filtering of gEmissiveAO.a mixes negative Forward markers with
    // Deferred AO and can cause double shading or holes at geometry edges.
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), textureSize(gDepth, 0) - ivec2(1));
    float depth = texelFetch(gDepth, pixel, 0).r;
    vec4 nr = texelFetch(gNormalRough, pixel, 0);
    vec4 am = texelFetch(gAlbedoMetallic, pixel, 0);
    vec4 eo = texelFetch(gEmissiveAO, pixel, 0);

    // Forward/Custom geometry is present in depth and normals for GTAO,
    // but its shading runs later in the dedicated Forward pass.
    // Avoid double lighting (deferred + forward) at these pixels.
    if (depth < 0.99999 && eo.a < 0.0) {
        outIBLSpecular = vec4(0.0);
        outIBLDiffuse = vec4(0.0);
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 worldPos = worldPosFromDepth(depth, uv);

    // sky where no geometry was written
    if (depth >= 0.99999) {
        vec3 dir = normalize(worldPos - g.camPos.xyz);
        outIBLSpecular = vec4(0.0);
        outIBLDiffuse = vec4(0.0);
        outColor = vec4(textureLod(envMap, dir, 0.0).rgb, 1.0);
        return;
    }

    vec3 viewDir = normalize(g.camPos.xyz - worldPos);
    vec3 N = normalize(nr.xyz);
    float roughness = nr.w;
    vec3 albedo = am.rgb;
    float metallic = am.a;
    vec3 emissive = eo.rgb;
    // Read AO at the current pixel, matching the Forward renderpaths.
    // texelFetch avoids filtering across geometry silhouettes.
    ivec2 aoSize = textureSize(gtaoTexture, 0);
    ivec2 aoPixel = clamp(pixel, ivec2(0), aoSize - ivec2(1));
    float screenAO = clamp(texelFetch(gtaoTexture, aoPixel, 0).r, 0.0, 1.0);
    float ao = clamp(eo.a, 0.0, 1.0) * screenAO;

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
    // Match iblLighting() exactly, including Fresnel, AO and flat ambient.
    float ndv = clamp(dot(N, viewDir), 0.0, 1.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlickRoughness(ndv, F0, clamp(roughness, 0.045, 1.0));
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuseIBL = max(g.ambient.w, 0.0) * iblDiffuse(N, g.skyParams.x) * kd * albedo * ao
                    + g.ambient.rgb * albedo * kd * ao;
    outIBLDiffuse = vec4(max(diffuseIBL, vec3(0.0)), 1.0);
    outIBLSpecular = vec4(iblSpecularLighting(N, viewDir, albedo, metallic, roughness,
        ao, g.ambient.w, g.skyParams.x), 1.0);
    color += emissive;

    outColor = vec4(color, 1.0);
}
