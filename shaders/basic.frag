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
layout(location = 3) in float fragAO;
layout(location = 4) in float fragLight;

layout(location = 0) out vec4 outColor;

void main() {
    // Directional sun light
    vec3 sunDir = normalize(vec3(pc.sun_dir.x, 0.8, pc.sun_dir.y));
    
    // Base sun/sky lighting
    float ambient = 0.35;
    float diffuse = max(dot(fragNormal, sunDir), 0.0) * 0.50;
    float sky = max(dot(fragNormal, vec3(0.0, 1.0, 0.0)), 0.0) * 0.12;
    float sunLight = ambient + diffuse + sky;
    
    // Ambient occlusion: smoothly darkens corners and crevices
    // Map AO from [0..1] to [0.35..1.0] so fully occluded corners aren't pitch black
    float ao = mix(0.35, 1.0, fragAO);
    
    // Torch / point light contribution
    // fragLight is 0..1 from the light map. Warm-tinted torch glow.
    vec3 torchColor = vec3(1.0, 0.85, 0.6); // warm orange-white
    float torchStrength = fragLight * fragLight * 0.9; // quadratic falloff feels more natural
    
    // Combine: sun lighting modulated by AO, plus additive torch light
    vec3 litColor = fragColor * sunLight * ao;
    litColor += fragColor * torchColor * torchStrength;
    
    // Slight boost in very dark areas so you can still see something underground
    float minLight = 0.06;
    litColor = max(litColor, fragColor * minLight);
    
    // Distance fog
    float fogFactor = clamp((pc.fog_end - fragDist) / (pc.fog_end - pc.fog_start), 0.0, 1.0);
    vec3 color = mix(pc.fog_color.rgb, litColor, fogFactor);
    
    outColor = vec4(color, 1.0);
}
