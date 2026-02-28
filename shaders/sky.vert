#version 450

layout(binding = 0) uniform ViewUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
} ubo;

layout(location = 0) out vec3 fragDir;

void main() {
    // Generate fullscreen triangle from vertex index (3 verts, no buffer)
    // Produces NDC covering [-1,-1] to [3,3], GPU clips to screen
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 ndc = pos * 2.0 - 1.0;

    // Place at far plane so blocks draw in front
    gl_Position = vec4(ndc, 0.9999, 1.0);

    // Reconstruct world-space ray direction from clip coordinates
    // inv(proj) gives view-space direction, inv(view) rotates to world
    mat4 invProj = inverse(ubo.proj);
    mat4 invView = inverse(ubo.view);
    vec4 viewDir = invProj * vec4(ndc, 1.0, 1.0);
    viewDir.xyz /= viewDir.w;
    fragDir = mat3(invView) * viewDir.xyz;
}
