#version 450

layout(binding = 0) uniform ViewUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
} ubo;

layout(push_constant) uniform PushConstants {
    vec4 fog_color;
    float fog_start;
    float fog_end;
    vec2 sun_dir;
} pc;

// Compact vertex inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColorAO;        // R8G8B8A8_UNORM: .rgb = color, .a = AO
layout(location = 2) in uvec4 inNormalLight;    // R8G8B8A8_UINT: .r = normal_idx, .g = light_u8

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out float fragDist;
layout(location = 3) out float fragAO;
layout(location = 4) out float fragLight;

// Normal lookup table — matches Face enum order: Top, Bottom, North, South, East, West
const vec3 NORMAL_LUT[6] = vec3[6](
    vec3( 0.0,  1.0,  0.0),  // Top
    vec3( 0.0, -1.0,  0.0),  // Bottom
    vec3( 0.0,  0.0,  1.0),  // North
    vec3( 0.0,  0.0, -1.0),  // South
    vec3( 1.0,  0.0,  0.0),  // East
    vec3(-1.0,  0.0,  0.0)   // West
);

void main() {
    vec4 worldPos = vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;
    
    fragColor = inColorAO.rgb;
    fragAO = inColorAO.a;
    
    // Unpack normal from index
    uint normalIdx = clamp(inNormalLight.r, 0u, 5u);
    fragNormal = NORMAL_LUT[normalIdx];
    
    // Unpack light from u8 → 0.0-1.0
    fragLight = float(inNormalLight.g) / 255.0;
    
    fragDist = distance(worldPos.xyz, ubo.camera_pos.xyz);
}
