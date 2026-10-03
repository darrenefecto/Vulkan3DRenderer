#version 450

layout(set = 0, binding = 0) uniform Global {
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
} g;

layout(location = 0) out vec3 dir;

const vec3 pos[36] = vec3[36](
    // +X
    vec3( 1,-1,-1), vec3( 1, 1,-1), vec3( 1, 1, 1),
    vec3( 1,-1,-1), vec3( 1, 1, 1), vec3( 1,-1, 1),
    // -X
    vec3(-1,-1, 1), vec3(-1, 1, 1), vec3(-1, 1,-1),
    vec3(-1,-1, 1), vec3(-1, 1,-1), vec3(-1,-1,-1),
    // +Y
    vec3(-1, 1,-1), vec3( 1, 1,-1), vec3( 1, 1, 1),
    vec3(-1, 1,-1), vec3( 1, 1, 1), vec3(-1, 1, 1),
    // -Y
    vec3(-1,-1, 1), vec3( 1,-1, 1), vec3( 1,-1,-1),
    vec3(-1,-1, 1), vec3( 1,-1,-1), vec3(-1,-1,-1),
    // +Z
    vec3(-1,-1, 1), vec3( 1,-1, 1), vec3( 1, 1, 1),
    vec3(-1,-1, 1), vec3( 1, 1, 1), vec3(-1, 1, 1),
    // -Z
    vec3( 1,-1,-1), vec3(-1,-1,-1), vec3(-1, 1,-1),
    vec3( 1,-1,-1), vec3(-1, 1,-1), vec3( 1, 1,-1)
);

void main() {
    vec3 p = pos[gl_VertexIndex];
    dir = p;
    mat4 viewNoTranslation = mat4(mat3(g.view)); // skybox follows the camera
    vec4 clip = g.proj * viewNoTranslation * vec4(p, 1.0);
    gl_Position = clip.xyww; // depth = 1.0 (far plane)
}