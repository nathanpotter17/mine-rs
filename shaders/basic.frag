#version 450

layout(push_constant) uniform PushConstants {
    vec4 fog_color;
    float fog_start;
    float fog_end;
    vec2 sun_dir;
} pc;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in float fragDist;

layout(location = 0) out vec4 outColor;

void main() {
    // Directional sun light
    vec3 sunDir = normalize(vec3(pc.sun_dir.x, 0.8, pc.sun_dir.y));
    
    // Lighting
    float ambient = 0.45;
    float diffuse = max(dot(fragNormal, sunDir), 0.0) * 0.55;
    
    // Slight sky light from above
    float sky = max(dot(fragNormal, vec3(0.0, 1.0, 0.0)), 0.0) * 0.15;
    
    float light = ambient + diffuse + sky;
    vec3 color = fragColor * light;
    
    // Distance fog
    float fogFactor = clamp((pc.fog_end - fragDist) / (pc.fog_end - pc.fog_start), 0.0, 1.0);
    color = mix(pc.fog_color.rgb, color, fogFactor);
    
    outColor = vec4(color, 1.0);
}
