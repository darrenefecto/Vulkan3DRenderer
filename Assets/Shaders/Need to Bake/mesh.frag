#version 450

layout(set = 0, binding = 0) uniform Global {
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
} g;

layout(set = 1, binding = 0) uniform MaterialUBO {
    vec4 baseColor;
    float hasTexture;
    float roughness;
    float metallic;
} mat;

layout(set = 1, binding = 1) uniform sampler2D albedoMap;

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 albedo = mat.baseColor;
    if (mat.hasTexture > 0.5)
        albedo *= texture(albedoMap, uv);

    vec3 N = normalize(normal);
    vec3 L = normalize(-g.lightDir.xyz);
    vec3 V = normalize(g.camPos.xyz - worldPos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float specPow = mix(8.0, 128.0, 1.0 - mat.roughness);
    float spec = pow(max(dot(N, H), 0.0), specPow) * (1.0 - mat.roughness);

    vec3 color = g.ambient.rgb * albedo.rgb
               + diff * g.lightColor.rgb * albedo.rgb
               + spec * g.lightColor.rgb;
    outColor = vec4(color, albedo.a);
}