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

// Compact vertex inputs (20 bytes total — unchanged from pre-texture layout)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColorAO;        // R8G8B8A8_UNORM: .rgb = tint color, .a = AO
layout(location = 2) in uvec4 inNormalLight;    // R8G8B8A8_UINT: .r = normal_idx, .g = light_u8,
                                                //                .b = tile_index,  .a = uv_corner

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out float fragDist;
layout(location = 3) out float fragAO;
layout(location = 4) out float fragLight;
layout(location = 5) out vec2 fragUV;

// Normal lookup — matches Face enum: Top, Bottom, North, South, East, West
const vec3 NORMAL_LUT[6] = vec3[6](
    vec3( 0.0,  1.0,  0.0),
    vec3( 0.0, -1.0,  0.0),
    vec3( 0.0,  0.0,  1.0),
    vec3( 0.0,  0.0, -1.0),
    vec3( 1.0,  0.0,  0.0),
    vec3(-1.0,  0.0,  0.0)
);

// uv_corner: 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1)
const vec2 UV_CORNERS[4] = vec2[4](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
);

void main() {
    vec4 worldPos = vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;

    fragColor = inColorAO.rgb;
    fragAO = inColorAO.a;

    uint normalIdx = clamp(inNormalLight.r, 0u, 5u);
    fragNormal = NORMAL_LUT[normalIdx];

    fragLight = float(inNormalLight.g) / 255.0;

    // Decode tile atlas UV from packed .b (tile_index) and .a (uv_corner)
    // Atlas: 16x16 grid of tiles, each tile = 16x16 px in a 256x256 texture
    uint tileIndex = inNormalLight.b;
    uint uvCorner  = clamp(inNormalLight.a, 0u, 3u);

    float tileCol = float(tileIndex & 0xFu);       // % 16
    float tileRow = float(tileIndex >> 4u);         // / 16
    vec2 localUV  = UV_CORNERS[uvCorner];

    // Half-texel inset prevents atlas tile bleeding under bilinear filtering
    // Atlas = 256px, tile = 16px, half texel = 0.5/256
    const float HALF_TEXEL = 0.5 / 256.0;
    const float TILE_SIZE  = 1.0 / 16.0;

    fragUV = vec2(tileCol, tileRow) * TILE_SIZE
           + localUV * (TILE_SIZE - 2.0 * HALF_TEXEL)
           + HALF_TEXEL;

    fragDist = distance(worldPos.xyz, ubo.camera_pos.xyz);
}
