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

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in float inAO;
layout(location = 4) in float inLight;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out float fragDist;
layout(location = 3) out float fragAO;
layout(location = 4) out float fragLight;

void main() {
    vec4 worldPos = vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;
    
    fragColor = inColor;
    fragNormal = inNormal;
    fragDist = distance(worldPos.xyz, ubo.camera_pos.xyz);
    fragAO = inAO;
    fragLight = inLight;
}
