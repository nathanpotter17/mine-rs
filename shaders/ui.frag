#version 450

// UI overlay fragment shader — with texture support.
//
// Every UI quad is textured. Color-only elements sample a built-in 1×1 white
// pixel so the unified path is always:
//   outColor = texture(tex, uv) * vertexColor
//
// For text images with transparency, ensure the source PNG/BMP has a proper
// alpha channel — the alpha blending pipeline state handles compositing.

layout(binding = 0) uniform sampler2D uiTexture;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 texSample = texture(uiTexture, fragUV);
    outColor = texSample * fragColor;
}
