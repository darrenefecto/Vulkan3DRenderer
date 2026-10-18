#version 450
// G-Buffer pass: writes albedo+metallic, world normal+roughness, emissive+AO.

layout(std140, set = 1, binding = 0) uniform MaterialUBO {
    vec4 baseColor; // rgba base color factor
    vec4 emissive;  // rgb emissive factor
    vec4 params;    // x = metallic, y = roughness, z = ao strength, w = has AO map
    vec4 flags;     // x = has albedo, y = has normal, z = has MR map, w = has emissive map
} mat;

layout(set = 1, binding = 1) uniform sampler2D albedoMap;
layout(set = 1, binding = 2) uniform sampler2D normalMap;
layout(set = 1, binding = 3) uniform sampler2D metallicRoughnessMap;
layout(set = 1, binding = 4) uniform sampler2D emissiveMap;
layout(set = 1, binding = 5) uniform sampler2D aoMap;

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 tangent;

layout(location = 0) out vec4 outAlbedoMetallic; // rgb = albedo,  a = metallic
layout(location = 1) out vec4 outNormalRough;    // rgb = world N, a = roughness
layout(location = 2) out vec4 outEmissiveAO;     // rgb = emissive, a = AO

void main() {
    vec4 albedo = mat.baseColor;
    if (mat.flags.x > 0.5) albedo *= texture(albedoMap, uv);
    if (albedo.a < 0.1) discard; // alpha-tested foliage etc.

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
        vec4 mr = texture(metallicRoughnessMap, uv); // glTF: g = roughness, b = metallic
        roughness *= mr.g;
        metallic *= mr.b;
    }

    vec3 emissive = mat.emissive.rgb;
    if (mat.flags.w > 0.5) emissive *= texture(emissiveMap, uv).rgb;

    float ao = 1.0;
    if (mat.params.w > 0.5) ao = mix(1.0, texture(aoMap, uv).r, mat.params.z);

    outAlbedoMetallic = vec4(albedo.rgb, clamp(metallic, 0.0, 1.0));
    outNormalRough = vec4(N, clamp(roughness, 0.03, 1.0));
    // Negative AO marks Forward/Custom pixels for the deferred lighting shader.
    // Still preserve valid normals, depth and material cutout for GTAO.
    outEmissiveAO = vec4(emissive, -max(ao, 0.001));
}
