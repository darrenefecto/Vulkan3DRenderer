// Shared runtime IBL convolution. No extra descriptor sets: works with the
// existing linear-sampled sRGB environment cubemap (decoded by Vulkan).
// This is a bounded per-fragment alternative to offline irradiance/PMREM/LUT.
#ifndef RUNTIME_IBL_GLSL
#define RUNTIME_IBL_GLSL
#ifndef IBL_DIFFUSE_SAMPLES
#define IBL_DIFFUSE_SAMPLES 12
#endif
#ifndef IBL_SPECULAR_SAMPLES
#define IBL_SPECULAR_SAMPLES 16
#endif

float iblRadicalInverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
vec2 iblHammersley(uint i, uint n) {
    return vec2(float(i) / float(n), iblRadicalInverse(i));
}
mat3 iblBasis(vec3 n) {
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, n));
    return mat3(t, cross(n, t), n);
}
// Returns cosine-weighted hemisphere integral divided by PI (diffuse
// irradiance / PI), compatible with Lambertian kD * albedo.
// Offline cosine convolution stores irradiance / PI as an sRGB cubemap.
vec3 iblDiffuse(vec3 N, float maxLod) {
    return textureLod(diffuseIrradianceMap, normalize(N), 0.0).rgb;
}
vec3 iblSampleGGX(vec2 xi, vec3 N, float roughness) {
    float a = max(roughness * roughness, 0.0025);
    float a2 = a * a;
    float phi = 6.28318530718 * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    return iblBasis(N) * vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
// Environment GGX convolution about reflection direction R (V=R assumption).
// The source cubemap mip chain is only a sampling anti-aliasing aid; it is NOT
// mistakenly treated as a roughness-prefiltered radiance map.
// GGX convolution is baked into roughness-indexed mip levels.
vec3 iblPrefilter(vec3 R, float roughness, float maxLod) {
    float lod = clamp(roughness, 0.0, 1.0) * float(textureQueryLevels(specularPrefilterMap) - 1);
    return textureLod(specularPrefilterMap, normalize(R), lod).rgb;
}
vec2 iblEnvBRDF(float ndv, float roughness) {
    return texture(brdfLutMap, vec2(clamp(ndv, 0.0, 1.0), clamp(roughness, 0.0, 1.0))).rg;
}
// View/roughness-aware specular ambient occlusion (Filament-style heuristic).
// With rough surfaces specular AO tends toward diffuse AO; smooth surfaces
// retain more unoccluded environment reflections. AO=1 is always neutral.
// GTAO is screen-space and cannot provide directionally correct specular
// visibility: this is intentionally a cheap, bounded approximation.
float iblSpecularOcclusion(float ndv, float ao, float roughness) {
    float visibility = clamp(ao, 0.0, 1.0);
    float exponent = exp2(-16.0 * clamp(roughness, 0.045, 1.0) - 1.0);
    return clamp(pow(max(ndv + visibility, 0.0), exponent) - 1.0 + visibility, 0.0, 1.0);
}

// Exactly the specular term used by iblLighting(), shared with the
// deferred specular MRT so SSR can replace it without doubling energy.
vec3 iblSpecularLighting(vec3 N, vec3 V, vec3 albedo, float metallic,
                         float roughness, float ao, float intensity, float maxLod) {
    roughness = clamp(roughness, 0.045, 1.0);
    float ndv = clamp(dot(N, V), 0.0, 1.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlickRoughness(ndv, F0, roughness);
    vec3 prefiltered = iblPrefilter(reflect(-V, N), roughness, maxLod);
    vec2 brdf = iblEnvBRDF(ndv, roughness);
    float specularAO = iblSpecularOcclusion(ndv, clamp(ao, 0.0, 1.0), roughness);
    return max(intensity, 0.0) * prefiltered * (F0 * brdf.x + brdf.y) * specularAO;
}

vec3 iblLighting(vec3 N, vec3 V, vec3 albedo, float metallic,
                 float roughness, float ao, float intensity, vec3 flatAmbient,
                 float maxLod) {
    roughness = clamp(roughness, 0.045, 1.0);
    float ndv = clamp(dot(N, V), 0.0, 1.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlickRoughness(ndv, F0, roughness);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = iblDiffuse(N, maxLod) * kd * albedo;
    vec3 prefiltered = iblPrefilter(reflect(-V, N), roughness, maxLod);
    vec2 brdf = iblEnvBRDF(ndv, roughness);
    vec3 specular = prefiltered * (F0 * brdf.x + brdf.y);
    // Apply diffuse AO to diffuse/flat ambient only. Use view- and
    // roughness-dependent occlusion for specular IBL (not direct/emissive).
    float diffuseAO = clamp(ao, 0.0, 1.0);
    float specularAO = iblSpecularOcclusion(ndv, diffuseAO, roughness);
    // Legacy flat ambient is diffuse irradiance, not an additional unoccluded
    // material color. Apply the same Fresnel/metallic diffuse weight as IBL.
    // In particular, fully metallic surfaces cannot gain diffuse ambient light.
    return max(intensity, 0.0) * (diffuse * diffuseAO + specular * specularAO)
           + flatAmbient * albedo * kd * diffuseAO;
}
#endif
