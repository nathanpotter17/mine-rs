#version 450

layout(location = 0) in vec2 inPos;

layout(push_constant) uniform PushData {
    float aspect_ratio; // width / height
} pc;

void main() {
    vec2 pos = inPos;
    pos.x /= pc.aspect_ratio;
    gl_Position = vec4(pos, 0.0, 1.0);
}