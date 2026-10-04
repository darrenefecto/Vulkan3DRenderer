#version 460

// Writes radial distance (normalized by the light's far plane) into depth so
// hardware depth-compare works with samplerCubeArrayShadow in mesh.frag.
layout(location = 0) in vec3 vWorldPos;

#define NUM_CASCADES 4
#define MAX_POINT 8
#define MAX_SPOT 8
#define MAX_AREA 4
#define MAX_SHADOW_POINT 4
#define MAX_SHADOW_SPOT 4
#define MAX_SHADOW_AREA 4

layout(set = 0, binding = 0, std140) uniform Global {
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 ambient;
    vec4 skyParams;
    vec4 sunDir;
    vec4 sunColor;
    mat4 sunViewProj[NUM_CASCADES];
    vec4 cascadeSplits;
    ivec4 counts;
    vec4 pointPos[MAX_POINT];
    vec4 pointColor[MAX_POINT];
    vec4 spotPos[MAX_SPOT];
    vec4 spotDir[MAX_SPOT];
    vec4 spotColor[MAX_SPOT];
    vec4 spotParams[MAX_SPOT];
    vec4 areaPos[MAX_AREA];
    vec4 areaRight[MAX_AREA];
    vec4 areaUp[MAX_AREA];
    vec4 areaColor[MAX_AREA];
    mat4 spotViewProj[MAX_SHADOW_SPOT];
    mat4 areaViewProj[MAX_SHADOW_AREA];
    mat4 pointViewProj[MAX_SHADOW_POINT * 6];
    vec4 pointShadowPosFar[MAX_SHADOW_POINT];
} g;

layout(push_constant) uniform Push {
    mat4 model;
    ivec4 idx; // x = lightIndex * 6 + face
} pc;

void main() {
    int lightIndex = pc.idx.x / 6;
    vec3 lightPos = g.pointShadowPosFar[lightIndex].xyz;
    float far = g.pointShadowPosFar[lightIndex].w;
    gl_FragDepth = length(vWorldPos - lightPos) / far;
}
