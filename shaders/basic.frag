#version 450

layout(binding = 1) uniform sampler2D texAtlas;  // binding 1 = texture atlas

layout(push_constant) uniform PushConstants {
    vec4 fog_color;
    float fog_start;
    float fog_end;
    vec2 sun_dir;
} pc;

layout(location = 0) in vec3 fragColor;     // tint color from vertex
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in float fragDist;
layout(location = 3) in float fragAO;
layout(location = 4) in float fragLight;
layout(location = 5) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

void main() {
    // Sample texture atlas
    vec4 texSample = texture(texAtlas, fragUV);

    // Combine texture with per-vertex tint color
    // Tint allows biome variation / per-block-type coloring on top of the texture
    vec3 baseColor = texSample.rgb * fragColor;

    // Directional sun light
    vec3 sunDir = normalize(vec3(pc.sun_dir.x, 0.8, pc.sun_dir.y));

    // Base sun/sky lighting
    float ambient = 0.35;
    float diffuse = max(dot(fragNormal, sunDir), 0.0) * 0.50;
    float sky = max(dot(fragNormal, vec3(0.0, 1.0, 0.0)), 0.0) * 0.12;
    float sunLight = ambient + diffuse + sky;

    // Ambient occlusion: smoothly darkens corners and crevices
    float ao = mix(0.35, 1.0, fragAO);

    // Torch / point light contribution — warm-tinted glow (boosted)
    vec3 torchColor = vec3(1.0, 0.82, 0.55);
    float torchStrength = fragLight * fragLight * 1.5;

    // Combine: sun lighting modulated by AO, plus additive torch light
    vec3 litColor = baseColor * sunLight * ao;
    litColor += baseColor * torchColor * torchStrength;

    // Slight boost in very dark areas
    float minLight = 0.06;
    litColor = max(litColor, baseColor * minLight);

    // Distance fog
    float fogFactor = clamp((pc.fog_end - fragDist) / (pc.fog_end - pc.fog_start), 0.0, 1.0);
    vec3 color = mix(pc.fog_color.rgb, litColor, fogFactor);

    outColor = vec4(color, 1.0);
}
