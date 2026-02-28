#version 450

layout(binding = 1) uniform sampler2D texAtlas;  // binding 1 = texture atlas

layout(push_constant) uniform PushConstants {
    vec4 fog_color;
    float fog_start;
    float fog_end;
    float time_of_day;
    float sun_intensity;
} pc;

layout(location = 0) in vec3 fragColor;     // tint color from vertex
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in float fragDist;
layout(location = 3) in float fragAO;
layout(location = 4) in float fragLight;
layout(location = 5) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Shared sun direction computation — MUST match sky.frag
vec3 computeSunDir(float tod) {
    float angle = tod * 6.2831853;
    return normalize(vec3(cos(angle), sin(angle), 0.25));
}

// Reinhard tone mapping — prevents blown-out highlights
vec3 tonemap(vec3 color) {
    return color / (color + vec3(1.0));
}

void main() {
    // Sample texture atlas
    vec4 texSample = texture(texAtlas, fragUV);

    // Combine texture with per-vertex tint color
    vec3 baseColor = texSample.rgb * fragColor;

    // === Time-based directional sun light ===
    vec3 sunDir = computeSunDir(pc.time_of_day);
    float sunElev = sunDir.y;
    float intensity = pc.sun_intensity;

    // Sun color: warm at sunrise/sunset, neutral at noon
    float sunsetBand = smoothstep(-0.05, 0.15, sunElev)
                     * (1.0 - smoothstep(0.15, 0.45, sunElev));
    vec3 sunTint = mix(vec3(1.0, 0.6, 0.35), vec3(1.0, 0.98, 0.95),
                       clamp(sunElev * 3.0, 0.0, 1.0));
    sunTint = mix(sunTint, vec3(1.0, 0.5, 0.2), sunsetBand * 0.4);

    // Ambient: stronger during day, dim blue at night
    vec3 ambientColor = mix(vec3(0.04, 0.04, 0.08), vec3(0.40, 0.42, 0.50), intensity);
    float ambientStr = mix(0.10, 0.32, intensity);

    // Diffuse: sun contribution scaled by intensity
    float NdotL = max(dot(fragNormal, sunDir), 0.0);
    float diffuseStr = NdotL * 0.55 * intensity;

    // Sky fill: upward-facing surfaces catch scattered skylight
    float skyFill = max(dot(fragNormal, vec3(0.0, 1.0, 0.0)), 0.0) * 0.10 * intensity;

    // Combine sun lighting
    vec3 sunLight = ambientColor * ambientStr + sunTint * (diffuseStr + skyFill);

    // === Ambient Occlusion ===
    float ao = mix(0.40, 1.0, fragAO);

    // === Torch / point light — warm glow ===
    // FIX: Removed min() cap that killed torches. Reinhard tonemap handles
    // any overflow gracefully — no more blown-out whites, but torches stay visible.
    // Higher boost at night when sun_intensity is low.
    vec3 torchColor = vec3(1.0, 0.82, 0.55);
    float torchRaw = fragLight * fragLight;
    float torchBoost = mix(1.8, 1.2, intensity);  // 1.8x at night, 1.2x at noon
    float torchStrength = torchRaw * torchBoost;   // NO cap

    // === Combine lighting ===
    vec3 litColor = baseColor * sunLight * ao;
    litColor += baseColor * torchColor * torchStrength;

    // Slight minimum so nothing goes full black
    float minLight = mix(0.03, 0.05, intensity);
    litColor = max(litColor, baseColor * minLight);

    // === Tone map block color (compresses highlights, fixes blown whites) ===
    litColor = tonemap(litColor);

    // === Fog AFTER tonemap ===
    // fog_color is raw (not tonemapped) — matches the sky shader output space.
    // At full fog, block color = fog_color = sky horizon → seamless blend to sky.
    float fogFactor = clamp((pc.fog_end - fragDist) / (pc.fog_end - pc.fog_start), 0.0, 1.0);
    vec3 color = mix(pc.fog_color.rgb, litColor, fogFactor);

    outColor = vec4(color, 1.0);
}
