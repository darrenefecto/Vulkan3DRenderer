#version 450
layout(set = 0, binding = 0) uniform GlobalView { mat4 view; mat4 proj; } g;
// 128-byte push constants: current model and previous MVP.
layout(push_constant) uniform Push { mat4 model; mat4 previousMVP; } pc;
layout(location = 0) in vec3 inPos;
// Separate previous-frame vertex stream, supplied at binding 1.
layout(location = 4) in vec3 previousPos;
layout(location = 0) out vec3 velocityDepth;
void main() {
    vec4 cur = g.proj * g.view * pc.model * vec4(inPos, 1.0);
    vec4 prev = pc.previousMVP * vec4(previousPos, 1.0);
    vec2 curUV = cur.xy / max(abs(cur.w), 1e-7) * sign(cur.w) * 0.5 + 0.5;
    vec2 prevUV = prev.xy / max(abs(prev.w), 1e-7) * sign(prev.w) * 0.5 + 0.5;
    velocityDepth = vec3(clamp(prevUV - curUV, vec2(-2.0), vec2(2.0)),
                         clamp(prev.z / max(abs(prev.w), 1e-7) * sign(prev.w), 0.0, 1.0));
    gl_Position = cur;
}
