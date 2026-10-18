// Shared PBR + shadow sampling code for the deferred / forward+ pipelines.
// Requires: set 0 binding 0 = Global UBO, bindings 1..5 = env/shadow samplers.
#ifndef PBR_COMMON_GLSL
#define PBR_COMMON_GLSL

#extension GL_EXT_shadow_samplers : enable

#define MAX_POINT_LIGHTS 8
#define MAX_SPOT_LIGHTS  8
#define MAX_AREA_LIGHTS  4
#define MAX_SHADOW_POINT 4
#define MAX_SHADOW_SPOT  4
#define MAX_SHADOW_AREA  4
#define NUM_CASCADES     4

// Must match GlobalUBO (Light.h), std140.
layout(std140, set = 0, binding = 0) uniform Global {
    mat4 view;
    mat4 proj;
    vec4 camPos;        // xyz = position
    vec4 ambient;       // rgb = flat ambient, w = IBL intensity
    vec4 skyParams;     // x = env cubemap max LOD

    vec4 sunDir;        // xyz = direction TOWARDS the light, w = enabled
    vec4 sunColor;      // rgb = color*intensity, w = castShadow
    mat4 sunViewProj[NUM_CASCADES];
    vec4 cascadeSplits; // view-space depth at the end of each cascade

    ivec4 counts;       // x = points, y = spots, z = areas, w = active cascades

    vec4 pointPos[MAX_POINT_LIGHTS];   // xyz = pos, w = range
    vec4 pointColor[MAX_POINT_LIGHTS]; // rgb = color*intensity, w = shadow index (-1 = none)

    vec4 spotPos[MAX_SPOT_LIGHTS];     // xyz = pos, w = range
    vec4 spotDir[MAX_SPOT_LIGHTS];     // xyz = dir, w = shadow index (-1 = none)
    vec4 spotColor[MAX_SPOT_LIGHTS];   // rgb = color*intensity, w = innerCos
    vec4 spotParams[MAX_SPOT_LIGHTS];  // x = outerCos

    vec4 areaPos[MAX_AREA_LIGHTS];     // xyz = center, w = shadow index (-1 = none)
    vec4 areaRight[MAX_AREA_LIGHTS];   // xyz = right * halfWidth
    vec4 areaUp[MAX_AREA_LIGHTS];      // xyz = up * halfHeight, w = twoSided
    vec4 areaColor[MAX_AREA_LIGHTS];   // rgb = color*intensity, w = range

    mat4 spotViewProj[MAX_SHADOW_SPOT];
    mat4 areaViewProj[MAX_SHADOW_AREA];
    mat4 pointViewProj[MAX_SHADOW_POINT * 6];
    vec4 pointShadowPosFar[MAX_SHADOW_POINT]; // xyz = pos, w = far plane
} g;

layout(set = 0, binding = 1) uniform samplerCube envMap;
layout(set = 0, binding = 6) uniform samplerCube diffuseIrradianceMap;
layout(set = 0, binding = 7) uniform samplerCube specularPrefilterMap;
layout(set = 0, binding = 8) uniform sampler2D brdfLutMap;
layout(set = 0, binding = 2) uniform sampler2DArrayShadow csmShadowMap;
layout(set = 0, binding = 3) uniform sampler2DArrayShadow spotShadowMap;
layout(set = 0, binding = 4) uniform samplerCubeArrayShadow pointShadowMap;
layout(set = 0, binding = 5) uniform sampler2DArrayShadow areaShadowMap;

const float PI = 3.14159265359;

// ---------------------------------------------------------------------------
// Cook-Torrance GGX
// ---------------------------------------------------------------------------
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float ndh = max(dot(N, H), 0.0);
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}
float geometrySchlick(float ndv, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return ndv / (ndv * (1.0 - k) + k);
}
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlick(max(dot(N, V), 0.0), roughness) *
           geometrySchlick(max(dot(N, L), 0.0), roughness);
}
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 fr = max(vec3(1.0 - roughness), F0);
    return F0 + (fr - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 cookTorrance(vec3 N, vec3 V, vec3 L, vec3 radiance,
                  vec3 albedo, float metallic, float roughness) {
    // Imported/custom roughness can be zero or out of range. Avoid a GGX
    // singularity and use the same lower bound as the IBL shader.
    roughness = clamp(roughness, 0.045, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    if (dot(N, L) <= 0.0 || dot(N, V) <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    float ndf = distributionGGX(N, H, roughness);
    float g = geometrySmith(N, V, L, roughness);
    vec3 spec = ndf * g * F / max(4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0), 1e-6);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    float ndl = max(dot(N, L), 0.0);
    return (kd * albedo / PI + spec) * radiance * ndl;
}

// ---------------------------------------------------------------------------
// Shadows (hardware depth-compare samplers, same layout as the forward path)
// ---------------------------------------------------------------------------
float sunVisibility(vec3 worldPos, float viewDepth) {
    if (g.sunColor.w < 0.5) return 1.0;
    int c = g.counts.w - 1;
    for (int i = 0; i < g.counts.w; i++) {
        if (viewDepth <= g.cascadeSplits[i]) { c = i; break; }
    }
    vec4 lp = g.sunViewProj[c] * vec4(worldPos, 1.0);
    vec3 uvz = lp.xyz / lp.w;
    uvz.xy = uvz.xy * 0.5 + 0.5;
    if (uvz.x < 0.0 || uvz.x > 1.0 || uvz.y < 0.0 || uvz.y > 1.0 || uvz.z > 1.0) return 1.0;
    float bias = 0.0006;
    return texture(csmShadowMap, vec4(uvz.xy, float(c), uvz.z - bias));
}

float pointVisibility(int shadowIdx, vec3 worldPos) {
    vec3 toFrag = worldPos - g.pointShadowPosFar[shadowIdx].xyz;
    float dist = length(toFrag);
    float far = g.pointShadowPosFar[shadowIdx].w;
    float r = dist / far - 0.005;
    return texture(pointShadowMap, vec4(toFrag, float(shadowIdx)), r);
}

float spotVisibility(int shadowIdx, vec3 worldPos) {
    vec4 lp = g.spotViewProj[shadowIdx] * vec4(worldPos, 1.0);
    vec3 uvz = lp.xyz / lp.w;
    uvz.xy = uvz.xy * 0.5 + 0.5;
    if (uvz.x < 0.0 || uvz.x > 1.0 || uvz.y < 0.0 || uvz.y > 1.0 || uvz.z > 1.0) return 0.0;
    return texture(spotShadowMap, vec4(uvz.xy, float(shadowIdx), uvz.z - 0.0008));
}

float areaVisibility(int shadowIdx, vec3 worldPos) {
    vec4 lp = g.areaViewProj[shadowIdx] * vec4(worldPos, 1.0);
    vec3 uvz = lp.xyz / lp.w;
    uvz.xy = uvz.xy * 0.5 + 0.5;
    if (uvz.x < 0.0 || uvz.x > 1.0 || uvz.y < 0.0 || uvz.y > 1.0 || uvz.z > 1.0) return 1.0;
    return texture(areaShadowMap, vec4(uvz.xy, float(shadowIdx), uvz.z - 0.0008));
}

// ---------------------------------------------------------------------------
// Light evaluators
// ---------------------------------------------------------------------------
vec3 evalSun(vec3 N, vec3 V, vec3 worldPos, float viewDepth,
             vec3 albedo, float metallic, float roughness) {
    if (g.sunDir.w < 0.5) return vec3(0.0);
    vec3 L = normalize(g.sunDir.xyz);
    float vis = sunVisibility(worldPos, viewDepth);
    return cookTorrance(N, V, L, g.sunColor.rgb * vis, albedo, metallic, roughness);
}

vec3 evalPoint(int i, vec3 N, vec3 V, vec3 worldPos,
               vec3 albedo, float metallic, float roughness) {
    vec3 toLight = g.pointPos[i].xyz - worldPos;
    float dist = length(toLight);
    float range = g.pointPos[i].w;
    if (dist >= range) return vec3(0.0);
    vec3 L = toLight / max(dist, 1e-5);
    float att = pow(clamp(1.0 - dist / range, 0.0, 1.0), 2.0) / max(dist * dist, 0.01);
    float shadowIdx = g.pointColor[i].w;
    float vis = 1.0;
    if (shadowIdx >= 0.0) vis = pointVisibility(int(shadowIdx + 0.5), worldPos);
    return cookTorrance(N, V, L, g.pointColor[i].rgb * att * vis, albedo, metallic, roughness);
}

vec3 evalSpot(int i, vec3 N, vec3 V, vec3 worldPos,
              vec3 albedo, float metallic, float roughness) {
    vec3 toLight = g.spotPos[i].xyz - worldPos;
    float dist = length(toLight);
    float range = g.spotPos[i].w;
    if (dist >= range) return vec3(0.0);
    vec3 L = toLight / max(dist, 1e-5);
    float cosA = dot(-L, normalize(g.spotDir[i].xyz));
    float inner = g.spotColor[i].w;
    float outer = g.spotParams[i].x;
    float cone = clamp((cosA - outer) / max(inner - outer, 1e-5), 0.0, 1.0);
    float att = cone * cone * pow(clamp(1.0 - dist / range, 0.0, 1.0), 2.0) / max(dist * dist, 0.01);
    float shadowIdx = g.spotDir[i].w;
    float vis = 1.0;
    if (shadowIdx >= 0.5) vis = spotVisibility(int(shadowIdx + 0.5), worldPos);
    return cookTorrance(N, V, L, g.spotColor[i].rgb * att * vis, albedo, metallic, roughness);
}

// Rectangular area light, approximated by 5 point samples on the panel.
vec3 evalArea(int i, vec3 N, vec3 V, vec3 worldPos,
              vec3 albedo, float metallic, float roughness) {
    vec3 center = g.areaPos[i].xyz;
    vec3 R = g.areaRight[i].xyz;
    vec3 U = g.areaUp[i].xyz;
    bool twoSided = g.areaUp[i].w > 0.5;
    float range = g.areaColor[i].w;
    vec3 lightN = normalize(cross(normalize(R), normalize(U)));
    if (!twoSided && dot(lightN, worldPos - center) < 0.0) return vec3(0.0);

    float vis = 1.0;
    float shadowIdx = g.areaPos[i].w;
    if (shadowIdx >= 0.5) vis = areaVisibility(int(shadowIdx + 0.5), worldPos);

    vec3 radiance = g.areaColor[i].rgb * vis / 5.0;
    vec3 sum = vec3(0.0);
    for (int s = 0; s < 5; s++) {
        vec2 o = (s == 0) ? vec2(0.0) :
                 (s == 1) ? vec2(0.7, 0.0) :
                 (s == 2) ? vec2(-0.7, 0.0) :
                 (s == 3) ? vec2(0.0, 0.7) : vec2(0.0, -0.7);
        vec3 p = center + R * o.x + U * o.y;
        vec3 toLight = p - worldPos;
        float dist = length(toLight);
        if (dist >= range) continue;
        vec3 L = toLight / max(dist, 1e-5);
        float att = pow(clamp(1.0 - dist / range, 0.0, 1.0), 2.0) / max(dist * dist, 0.01);
        sum += cookTorrance(N, V, L, radiance * att, albedo, metallic, roughness);
    }
    return sum;
}

#include "ibl_runtime.glsl"

vec3 evalIBL(vec3 N, vec3 V, vec3 albedo, float metallic, float roughness, float ao) {
    return iblLighting(N, V, albedo, metallic, roughness, ao,
                       g.ambient.w, g.ambient.rgb, g.skyParams.x);
}
#endif // PBR_COMMON_GLSL
