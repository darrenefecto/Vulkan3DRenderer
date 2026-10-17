#version 460

#define NUM_CASCADES 4
#define MAX_POINT 8
#define MAX_SPOT 8
#define MAX_AREA 4
#define MAX_SHADOW_POINT 4
#define MAX_SHADOW_SPOT 4
#define MAX_SHADOW_AREA 4

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vTangent;
layout(location = 4) in float vViewDepth;
layout(location = 0) out vec4 outColor;

// Must match GlobalUBO in Light.h exactly (std140).
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
layout(set = 0, binding = 2) uniform sampler2DArrayShadow csmShadow;
layout(set = 0, binding = 3) uniform sampler2DArrayShadow spotShadow;
layout(set = 0, binding = 4) uniform samplerCubeArrayShadow pointShadow;
layout(set = 0, binding = 5) uniform sampler2DArrayShadow areaShadow;

// Must match MaterialData in Mesh.h (std140).
layout(set = 1, binding = 0, std140) uniform MaterialUBO {
    vec4 baseColor;
    vec4 emissive;
    vec4 params;    // x = metallic, y = roughness, z = ao strength, w = has AO map
    vec4 flags;     // x = albedo, y = normal, z = metallic-roughness, w = emissive map
} m;
layout(set = 1, binding = 1) uniform sampler2D albedoMap;
layout(set = 1, binding = 2) uniform sampler2D normalMap;
layout(set = 1, binding = 3) uniform sampler2D metallicRoughnessMap;
layout(set = 1, binding = 4) uniform sampler2D emissiveMap;
layout(set = 1, binding = 5) uniform sampler2D aoMap;

const float PI = 3.14159265359;

// ---------------------------------------------------------------------------
// PBR (Cook-Torrance GGX)
// ---------------------------------------------------------------------------

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) *
        pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotX / (NdotX * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

// One analytic light contribution. radiance = color*intensity at the fragment.
vec3 evalLight(vec3 radiance, vec3 L, vec3 N, vec3 V, vec3 albedo,
               float metallic, float roughness, vec3 F0) {
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 1e-4);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

// Windowed inverse-square falloff (smoothly reaches 0 at range).
float distanceAttenuation(float dist, float range) {
    float x = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    return (x * x) / (dist * dist + 1.0);
}

// ---------------------------------------------------------------------------
// Shadow sampling
// ---------------------------------------------------------------------------

float sunShadowVis(vec3 worldPos, vec3 N, vec3 L) {
    int cascade = g.counts.w - 1;
    for (int i = 0; i < g.counts.w; i++) {
        if (vViewDepth < g.cascadeSplits[i]) { cascade = i; break; }
    }
    vec4 p = g.sunViewProj[cascade] * vec4(worldPos, 1.0);
    vec3 c = p.xyz / p.w;
    vec2 uv = c.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || c.z >= 1.0)
        return 1.0; // outside the cascade chain: lit (border sampler would also do this)
    // bias grows with cascade index (larger texels)
    float bias = max(0.0018 * (1.0 - dot(N, L)), 0.0004) * float(cascade + 1);
    vec2 texel = 1.0 / vec2(textureSize(csmShadow, 0).xy);
    float sum = 0.0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            sum += texture(csmShadow,
                vec4(uv + vec2(x, y) * texel, float(cascade), c.z - bias));
    return sum / 9.0;
}

float spotShadowVis(vec3 worldPos, int idx, vec3 N, vec3 L) {
    vec4 p = g.spotViewProj[idx] * vec4(worldPos, 1.0);
    vec3 c = p.xyz / p.w;
    vec2 uv = c.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || c.z >= 1.0)
        return 1.0;
    float bias = max(0.003 * (1.0 - dot(N, L)), 0.0008);
    vec2 texel = 1.0 / vec2(textureSize(spotShadow, 0).xy);
    float sum = 0.0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            sum += texture(spotShadow,
                vec4(uv + vec2(x, y) * texel, float(idx), c.z - bias));
    return sum / 9.0;
}

float areaShadowVis(vec3 worldPos, int idx, vec3 N, vec3 L) {
    vec4 p = g.areaViewProj[idx] * vec4(worldPos, 1.0);
    vec3 c = p.xyz / p.w;
    vec2 uv = c.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || c.z >= 1.0)
        return 1.0;
    float bias = max(0.003 * (1.0 - dot(N, L)), 0.0008);
    vec2 texel = 1.0 / vec2(textureSize(areaShadow, 0).xy);
    float sum = 0.0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            sum += texture(areaShadow,
                vec4(uv + vec2(x, y) * texel, float(idx), c.z - bias));
    return sum / 9.0;
}

float pointShadowVis(vec3 worldPos, int idx, vec3 N, vec3 L) {
    vec3 lightPos = g.pointShadowPosFar[idx].xyz;
    float far = g.pointShadowPosFar[idx].w;
    vec3 dir = worldPos - lightPos;
    float dist = length(dir);
    float bias = max(0.0035 * (1.0 - dot(N, L)), 0.0012);
    float compare = dist / far - bias;
    if (compare >= 1.0) return 1.0;
    vec3 d = normalize(dir);
    // small 5-tap PCF in a tangent frame around the sample direction
    vec3 t = normalize(cross(abs(d.y) > 0.99 ? vec3(1, 0, 0) : vec3(0, 1, 0), d));
    vec3 b = cross(d, t);
    float spread = 0.012;
    float sum = texture(pointShadow, vec4(d, float(idx)), compare);
    sum += texture(pointShadow, vec4(normalize(d + t * spread), float(idx)), compare);
    sum += texture(pointShadow, vec4(normalize(d - t * spread), float(idx)), compare);
    sum += texture(pointShadow, vec4(normalize(d + b * spread), float(idx)), compare);
    sum += texture(pointShadow, vec4(normalize(d - b * spread), float(idx)), compare);
    return sum / 5.0;
}

// ---------------------------------------------------------------------------
// Rect area light: most-representative-point specular + closest-point diffuse
// ---------------------------------------------------------------------------

// Closest point on the rect to a world position (used for diffuse).
vec3 areaClosestPoint(vec3 worldPos, vec3 center, vec3 right, vec3 up) {
    vec3 rN = normalize(right);
    vec3 uN = normalize(up);
    vec3 d = worldPos - center;
    vec2 uv = vec2(dot(d, rN), dot(d, uN));
    uv = clamp(uv, -vec2(length(right), length(up)), vec2(length(right), length(up)));
    return center + rN * uv.x + uN * uv.y;
}

// Karis "most representative point": intersect the reflection ray with the
// rect plane, clamp to the rect (used for specular).
vec3 areaMRP(vec3 worldPos, vec3 R, vec3 center, vec3 right, vec3 up) {
    vec3 n = normalize(cross(normalize(right), normalize(up)));
    vec3 p = center;
    float denom = dot(R, n);
    if (abs(denom) > 1e-4) {
        float t = dot(center - worldPos, n) / denom;
        p = worldPos + R * (t < 0.0 ? 1e5 : t); // behind: push far out, clamp brings it back
    }
    return areaClosestPoint(p, center, right, up);
}

// ---------------------------------------------------------------------------

vec3 getNormal() {
    vec3 N = normalize(vNormal);
    if (m.flags.y < 0.5) return N;
    vec3 mapN = texture(normalMap, vUV).xyz * 2.0 - 1.0;
    vec3 T = normalize(vTangent.xyz);
    T = normalize(T - N * dot(N, T));      // re-orthogonalize
    vec3 B = cross(N, T) * vTangent.w;     // handedness from the loader
    return normalize(mat3(T, B, N) * mapN);
}

vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    // ---- material fetch ----
    vec3 albedo = m.baseColor.rgb;
    if (m.flags.x > 0.5) albedo *= texture(albedoMap, vUV).rgb;

    vec2 mr = vec2(1.0);
    if (m.flags.z > 0.5) mr = texture(metallicRoughnessMap, vUV).gb; // glTF packing
    float roughness = clamp(m.params.y * mr.x, 0.04, 1.0);
    float metallic = clamp(m.params.x * mr.y, 0.0, 1.0);

    float ao = 1.0;
    if (m.params.w > 0.5) ao = mix(1.0, texture(aoMap, vUV).r, m.params.z);

    vec3 N = getNormal();
    vec3 V = normalize(g.camPos.xyz - vWorldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 Lo = vec3(0.0);

    // ---- sun (directional, CSM) ----
    if (g.sunDir.w > 0.5) {
        vec3 L = normalize(g.sunDir.xyz);
        float vis = (g.sunColor.w > 0.5) ? sunShadowVis(vWorldPos, N, L) : 1.0;
        Lo += evalLight(g.sunColor.rgb, L, N, V, albedo, metallic, roughness, F0) * vis;
    }

    // ---- point lights (cube shadow maps) ----
    for (int i = 0; i < g.counts.x; i++) {
        vec3 toLight = g.pointPos[i].xyz - vWorldPos;
        float dist = length(toLight);
        vec3 L = toLight / max(dist, 1e-5);
        float att = distanceAttenuation(dist, g.pointPos[i].w);
        if (att <= 0.0) continue;
        int shadowIdx = int(floor(g.pointColor[i].w + 0.5));
        float vis = (shadowIdx >= 0) ? pointShadowVis(vWorldPos, shadowIdx, N, L) : 1.0;
        Lo += evalLight(g.pointColor[i].rgb * att, L, N, V,
                        albedo, metallic, roughness, F0) * vis;
    }

    // ---- spot lights (2D shadow maps) ----
    for (int i = 0; i < g.counts.y; i++) {
        vec3 toLight = g.spotPos[i].xyz - vWorldPos;
        float dist = length(toLight);
        vec3 L = toLight / max(dist, 1e-5);
        float theta = dot(-L, normalize(g.spotDir[i].xyz));
        float innerCos = g.spotColor[i].w;
        float outerCos = g.spotParams[i].x;
        float cone = clamp((theta - outerCos) / max(innerCos - outerCos, 1e-4), 0.0, 1.0);
        cone *= cone;
        float att = distanceAttenuation(dist, g.spotPos[i].w) * cone;
        if (att <= 0.0) continue;
        int shadowIdx = int(floor(g.spotDir[i].w + 0.5));
        float vis = (shadowIdx >= 0) ? spotShadowVis(vWorldPos, shadowIdx, N, L) : 1.0;
        Lo += evalLight(g.spotColor[i].rgb * att, L, N, V,
                        albedo, metallic, roughness, F0) * vis;
    }

    // ---- rect area lights ----
    for (int i = 0; i < g.counts.z; i++) {
        vec3 center = g.areaPos[i].xyz;
        vec3 right = g.areaRight[i].xyz;   // scaled by half width
        vec3 up = g.areaUp[i].xyz;         // scaled by half height
        bool twoSided = g.areaUp[i].w > 0.5;
        float range = g.areaColor[i].w;
        vec3 areaN = normalize(cross(normalize(right), normalize(up)));

        vec3 closest = areaClosestPoint(vWorldPos, center, right, up);
        vec3 Ld = closest - vWorldPos;
        float distD = length(Ld);
        Ld /= max(distD, 1e-5);
        float facing = dot(-Ld, areaN);
        if (!twoSided && facing < 0.0) continue;
        facing = abs(facing);
        float attD = distanceAttenuation(distD, range) * facing;

        int shadowIdx = int(floor(g.areaPos[i].w + 0.5));
        float vis = (shadowIdx >= 0) ? areaShadowVis(vWorldPos, shadowIdx, N, Ld) : 1.0;

        // diffuse: light from the closest point on the rect
        if (attD > 0.0)
            Lo += evalLight(g.areaColor[i].rgb * attD, Ld, N, V,
                            albedo, metallic, roughness, F0) * vis;

        // specular: most representative point
        vec3 R = reflect(-V, N);
        vec3 mrp = areaMRP(vWorldPos, R, center, right, up);
        vec3 Ls = mrp - vWorldPos;
        float distS = length(Ls);
        Ls /= max(distS, 1e-5);
        float attS = distanceAttenuation(distS, range) * facing;
        if (attS > 0.0) {
            float NdotL = dot(N, Ls);
            if (NdotL > 0.0) {
                vec3 H = normalize(V + Ls);
                float NDF = distributionGGX(N, H, roughness);
                float G = geometrySmith(N, V, Ls, roughness);
                vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                vec3 spec = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 1e-4);
                Lo += spec * g.areaColor[i].rgb * attS * NdotL * vis;
            }
        }
    }

    // ---- ambient: IBL from the skybox cubemap + flat fallback ----
    vec3 R = reflect(-V, N);
    float maxLod = g.skyParams.x;
    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    vec3 irradiance = textureLod(envMap, N, maxLod).rgb;
    vec3 prefiltered = textureLod(envMap, R, roughness * maxLod).rgb;
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 ibl = (kD * irradiance * albedo + prefiltered * F) * g.ambient.w;
    vec3 ambient = (ibl + g.ambient.rgb * albedo) * ao;

    // ---- emissive ----
    vec3 emissive = m.emissive.rgb;
    if (m.flags.w > 0.5) emissive *= texture(emissiveMap, vUV).rgb;

    vec3 color = ambient + Lo + emissive;
    color = acesFilm(color); // HDR tonemap; sRGB swapchain encodes gamma
    outColor = vec4(color, 1.0);
}
