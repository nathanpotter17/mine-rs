#version 450

// UI overlay vertex shader — with texture support.
// Receives clip-space positions, texture UVs, and per-vertex RGBA color.
// CPU converts from normalized screen coords (0-1) to clip space.

layout(location = 0) in vec2 inPos;    // clip-space xy
layout(location = 1) in vec2 inUV;     // texture coordinates
layout(location = 2) in vec4 inColor;  // RGBA tint / solid color

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec4 fragColor;

void main() {
    gl_Position = vec4(inPos, 0.0, 1.0);
    fragUV    = inUV;
    fragColor = inColor;
}
