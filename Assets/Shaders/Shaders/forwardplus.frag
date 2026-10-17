#version 450
#extension GL_GOOGLE_include_directive : require
// Forward+ fragment shader: sun + area lights come from the Global UBO,
// point/spot lights are fetched per-pixel from the cluster light list.

#include "pbr_common.glsl"

struct ClusterLight {
    vec4 posRange;   // xyz = world pos, w = range
    vec4 color;      // rgb = color*intensity, w = shadow index (-1 = none)
    vec4 dirInner;   // xyz = direction (spots), w = innerCos
    vec4 outerType;  // x = outerCos, y = type (0 = point, 1 = spot)
};

layout(std430, set = 2, binding = 0) readonly buffer LightBuffer {
    ClusterLight lights[];
};
layout(std430, set = 2, binding = 1) readonly buffer ClusterGrid {
    uvec2 grid[];
};
layout(std430, set = 2, binding = 2) readonly buffer LightIndices {
    uint lightIndices[];
};

layout(std140, set = 1, binding = 0) uniform MaterialUBO {
    vec4 baseColor;
    vec4 emissive;
    vec4 params;    // x = metallic, y = roughness, z = ao strength, w = has AO map
    vec4 flags;     // x = has albedo, y = has normal, z = has MR map, w = has emissive map
} mat;

layout(set = 1, binding = 1) uniform sampler2D albedoMap;
layout(set = 1, binding = 2) uniform sampler2D normalMap;
layout(set = 1, binding = 3) uniform sampler2D metallicRoughnessMap;
layout(set = 1, binding = 4) uniform sampler2D emissiveMap;
layout(set = 1, binding = 5) uniform sampler2D aoMap;

#define MAX_LIGHTS_PER_CLUSTER 64

layout(push_constant) uniform ClusterParams {
    layout(offset = 64) vec4 screen;  // w, h, tileW, tileH
    vec4 zparams;                     // near, far, tanHalfFovX, tanHalfFovY
    ivec4 dims;                       // gridX, gridY, gridZ, lightCount
} cp;

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 tangent;

layout(location = 0) out vec4 outColor;

vec3 evalClusterPoint(ClusterLight L, vec3 N, vec3 V, vec3 albedo, float metallic, float roughness) {
    vec3 toLight = L.posRange.xyz - worldPos;
    float dist = length(toLight);
    float range = L.posRange.w;
    if (dist >= range) return vec3(0.0);
    vec3 l = toLight / max(dist, 1e-5);
    float att = pow(clamp(1.0 - dist / range, 0.0, 1.0), 2.0) / max(dist * dist, 0.01);
    float vis = 1.0;
    if (L.color.w >= 0.0) vis = pointVisibility(int(L.color.w + 0.5), worldPos);
    return cookTorrance(N, V, l, L.color.rgb * att * vis, albedo, metallic, roughness);
}

vec3 evalClusterSpot(ClusterLight L, vec3 N, vec3 V, vec3 albedo, float metallic, float roughness) {
    vec3 toLight = L.posRange.xyz - worldPos;
    float dist = length(toLight);
    float range = L.posRange.w;
    if (dist >= range) return vec3(0.0);
    vec3 l = toLight / max(dist, 1e-5);
    float cosA = dot(-l, normalize(L.dirInner.xyz));
    float cone = clamp((cosA - L.outerType.x) / max(L.dirInner.w - L.outerType.x, 1e-5), 0.0, 1.0);
    float att = cone * cone * pow(clamp(1.0 - dist / range, 0.0, 1.0), 2.0) / max(dist * dist, 0.01);
    float vis = 1.0;
    if (L.color.w >= 0.0) vis = spotVisibility(int(L.color.w + 0.5), worldPos);
    return cookTorrance(N, V, l, L.color.rgb * att * vis, albedo, metallic, roughness);
}

void main() {
    vec4 albedoA = mat.baseColor;
    if (mat.flags.x > 0.5) albedoA *= texture(albedoMap, uv);
    if (albedoA.a < 0.1) discard;

    vec3 N = normalize(normal);
    if (mat.flags.y > 0.5) {
        vec3 T = normalize(tangent.xyz);
        T = normalize(T - N * dot(N, T));
        vec3 B = cross(N, T) * tangent.w;
        vec3 tn = texture(normalMap, uv).xyz * 2.0 - 1.0;
        N = normalize(mat3(T, B, N) * tn);
    }

    float metallic = mat.params.x;
    float roughness = mat.params.y;
    if (mat.flags.z > 0.5) {
        vec4 mr = texture(metallicRoughnessMap, uv);
        roughness *= mr.g;
        metallic *= mr.b;
    }
    vec3 emissive = mat.emissive.rgb;
    if (mat.flags.w > 0.5) emissive *= texture(emissiveMap, uv).rgb;
    float ao = 1.0;
    if (mat.params.w > 0.5) ao = mix(1.0, texture(aoMap, uv).r, mat.params.z);

    vec3 V = normalize(g.camPos.xyz - worldPos);
    float viewDepth = -(g.view * vec4(worldPos, 1.0)).z;
    if (viewDepth <= cp.zparams.x || viewDepth >= cp.zparams.y) {
        outColor = vec4(0.0);
        return;
    }

    vec3 albedo = albedoA.rgb;

    vec3 color = vec3(0.0);
    color += evalSun(N, V, worldPos, viewDepth, albedo, metallic, roughness);

    // --- clustered point/spot lights ---
    
    ivec3 cid;

    cid.x = clamp(
        int(gl_FragCoord.x / cp.screen.z),
        0,
        cp.dims.x - 1
    );

    // Vulkan fragment coordinates use the same top-left pixel space
    // used by the cluster build.
    cid.y = clamp(
        int(gl_FragCoord.y / cp.screen.w),
        0,
        cp.dims.y - 1
    );

    float zSlice =
        log(max(viewDepth, cp.zparams.x) / cp.zparams.x) /
        log(cp.zparams.y / cp.zparams.x) *
        float(cp.dims.z);

    cid.z = clamp(
        int(zSlice),
        0,
        cp.dims.z - 1
    );

    uint clusterIdx = uint(cid.x) + uint(cid.y) * uint(cp.dims.x)
                    + uint(cid.z) * uint(cp.dims.x) * uint(cp.dims.y);
    uvec2 tile = grid[clusterIdx];
    for (uint i = 0; i < tile.y; i++) {
        ClusterLight L = lights[lightIndices[tile.x + i]];
        if (L.outerType.y < 0.5)
            color += evalClusterPoint(L, N, V, albedo, metallic, roughness);
        else
            color += evalClusterSpot(L, N, V, albedo, metallic, roughness);
    }

    for (int i = 0; i < g.counts.z; i++)
        color += evalArea(i, N, V, worldPos, albedo, metallic, roughness);
    color += evalIBL(N, V, albedo, metallic, roughness, ao);
    color += emissive;

    outColor = vec4(color, albedoA.a);
}
