#version 450

// UI overlay fragment shader
// Outputs interpolated per-vertex color with alpha blending support.

layout(location = 0) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = fragColor;
}
