#version 460
#extension GL_GOOGLE_include_directive : require

#define NUM_CASCADES 4
#define MAX_POINT 8
#define MAX_SPOT 8
#define MAX_AREA 4
#define MAX_SHADOW_POINT 4
#define MAX_SHADOW_SPOT 4
#define MAX_SHADOW_AREA 4

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 tangent;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0, std140) uniform Global {
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 ambient;                       // rgb = flat ambient, w = IBL intensity
    vec4 skyParams;                     // x = env max LOD
    vec4 sunDir;                        // xyz = direction towards sun, w = enabled
    vec4 sunColor;                      // rgb = color*intensity, w = castShadow
    mat4 sunViewProj[NUM_CASCADES];
    vec4 cascadeSplits;
    ivec4 counts;                       // x = points, y = spots, z = areas, w = cascades
    vec4 pointPos[MAX_POINT];           // xyz = pos, w = range
    vec4 pointColor[MAX_POINT];         // rgb = color*intensity, w = shadow index
    vec4 spotPos[MAX_SPOT];
    vec4 spotDir[MAX_SPOT];             // xyz = dir, w = shadow index
    vec4 spotColor[MAX_SPOT];           // rgb = color*intensity, w = innerCos
    vec4 spotParams[MAX_SPOT];          // x = outerCos
    vec4 areaPos[MAX_AREA];             // xyz = center, w = shadow index
    vec4 areaRight[MAX_AREA];           // xyz = right * halfWidth
    vec4 areaUp[MAX_AREA];              // xyz = up * halfHeight, w = twoSided
    vec4 areaColor[MAX_AREA];           // rgb = color*intensity, w = range
    mat4 spotViewProj[MAX_SHADOW_SPOT];
    mat4 areaViewProj[MAX_SHADOW_AREA];
    mat4 pointViewProj[MAX_SHADOW_POINT * 6];
    vec4 pointShadowPosFar[MAX_SHADOW_POINT];
} g;

layout(set = 0, binding = 1) uniform samplerCube envMap;
layout(set = 0, binding = 6) uniform samplerCube diffuseIrradianceMap;
layout(set = 0, binding = 7) uniform samplerCube specularPrefilterMap;
layout(set = 0, binding = 8) uniform sampler2D brdfLutMap;

layout(set = 2, binding = 0) uniform sampler2D gtaoTexture;
layout(set = 1, binding = 0, std140) uniform MaterialUBO {
    vec4 baseColor;
    vec4 emissive;
    vec4 params;
    vec4 flags;
} m;

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0)
        * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
#include "ibl_runtime.glsl"

void main() {
    vec3 N = normalize(worldNormal);
    vec3 V = normalize(g.camPos.xyz - worldPos);
    float stripe = 0.65 + 0.35 * sin(uv.x * 32.0);
    vec3 albedo = clamp(m.baseColor.rgb * stripe, 0.0, 1.0);
    ivec2 aoSize = textureSize(gtaoTexture, 0);
    ivec2 aoPixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), aoSize - ivec2(1));
    float gtao = clamp(texelFetch(gtaoTexture, aoPixel, 0).r, 0.0, 1.0);
    float materialAO = clamp(m.params.z, 0.0, 1.0);
    vec3 indirect = iblLighting(N, V, albedo, clamp(m.params.x, 0.0, 1.0),
                                clamp(m.params.y, 0.045, 1.0),
                                materialAO * clamp(gtao, 0.0, 1.0),
                                g.ambient.w, g.ambient.rgb, g.skyParams.x);
    outColor = vec4(indirect + m.emissive.rgb, 1.0);
}
