#version 450

layout(binding = 0) uniform ViewUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out float fragDist;

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;

    fragColor = inColor;
    // Transform normal by model rotation (no non-uniform scale, so mat3 is fine)
    fragNormal = normalize(mat3(pc.model) * inNormal);
    fragDist = distance(worldPos.xyz, ubo.camera_pos.xyz);
}
