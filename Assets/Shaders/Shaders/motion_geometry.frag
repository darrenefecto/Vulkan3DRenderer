#version 450
// RG = previousUV-currentUV, B = predicted previous-frame device depth.
layout(location = 0) in vec3 velocityDepth;
layout(location = 0) out vec4 outVelocity;
void main() { outVelocity = vec4(velocityDepth, 1.0); }
