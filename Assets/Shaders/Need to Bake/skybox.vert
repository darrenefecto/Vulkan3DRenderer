#version 460

// Only the prefix of the global UBO is needed here.
layout(set = 0, binding = 0, std140) uniform Global {
    mat4 view;
    mat4 proj;
} g;

layout(location = 0) out vec3 vDir;

void main() {
    // unit cube, no vertex buffer
    vec3 pos[36] = vec3[](
        vec3(-1,-1,-1), vec3( 1, 1,-1), vec3( 1,-1,-1),
        vec3( 1, 1,-1), vec3(-1,-1,-1), vec3(-1, 1,-1),
        vec3(-1,-1, 1), vec3( 1,-1, 1), vec3( 1, 1, 1),
        vec3( 1, 1, 1), vec3(-1, 1, 1), vec3(-1,-1, 1),
        vec3(-1,-1,-1), vec3(-1,-1, 1), vec3(-1, 1, 1),
        vec3(-1, 1, 1), vec3(-1, 1,-1), vec3(-1,-1,-1),
        vec3( 1,-1,-1), vec3( 1, 1,-1), vec3( 1, 1, 1),
        vec3( 1, 1, 1), vec3( 1,-1, 1), vec3( 1,-1,-1),
        vec3(-1,-1,-1), vec3( 1,-1,-1), vec3( 1,-1, 1),
        vec3( 1,-1, 1), vec3(-1,-1, 1), vec3(-1,-1,-1),
        vec3(-1, 1,-1), vec3(-1, 1, 1), vec3( 1, 1, 1),
        vec3( 1, 1, 1), vec3( 1, 1,-1), vec3(-1, 1,-1)
    );
    vDir = pos[gl_VertexIndex];
    // strip translation from the view matrix; force depth to the far plane
    mat4 rotView = mat4(mat3(g.view));
    vec4 clip = g.proj * rotView * vec4(vDir, 1.0);
    gl_Position = clip.xyww;
}
