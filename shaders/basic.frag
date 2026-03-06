#version 450

layout(binding = 1) uniform sampler2D texAtlas;

layout(push_constant) uniform PushConstants {
    vec4 fog_color;
    float fog_start;
    float fog_end;
    float time_of_day;
    float sun_intensity;
} pc;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in float fragDist;
layout(location = 3) in float fragAO;
layout(location = 4) in float fragSkyLight;
layout(location = 5) in vec2 fragUV;
layout(location = 6) in float fragBlockLight;

layout(location = 0) out vec4 outColor;

// Shared sun direction — MUST match sky.frag
vec3 computeSunDir(float tod) {
    float angle = tod * 6.2831853;
    return normalize(vec3(cos(angle), sin(angle), 0.25));
}

// Softened ACES-inspired filmic tone mapping
vec3 tonemap(vec3 x) {
    // Slightly gentler than stock ACES — less crushed blacks, softer rolloff
    float a = 2.40;
    float b = 0.04;
    float c = 2.40;
    float d = 0.62;
    float e = 0.15;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec4 texSample = texture(texAtlas, fragUV);

    // Alpha test — foliage cutout (TallGrass, Leaves holes)
    if (texSample.a < 0.5) discard;

    vec3 baseColor = texSample.rgb * fragColor;

    // === Sun parameters ===
    vec3 sunDir = computeSunDir(pc.time_of_day);
    float sunElev = sunDir.y;
    float intensity = pc.sun_intensity;

    // Sun color: warm at horizon, white at zenith
    float sunsetBand = smoothstep(-0.05, 0.15, sunElev)
                     * (1.0 - smoothstep(0.15, 0.45, sunElev));
    vec3 sunTint = mix(vec3(1.0, 0.6, 0.35), vec3(1.0, 0.98, 0.95),
                       clamp(sunElev * 3.0, 0.0, 1.0));
    sunTint = mix(sunTint, vec3(1.0, 0.5, 0.2), sunsetBand * 0.5);

    // === Hemispherical ambient (sky+ground bounce) ===
    vec3 skyAmbient = mix(vec3(0.04, 0.04, 0.08), vec3(0.38, 0.44, 0.58), intensity);
    vec3 groundBounce = mix(vec3(0.03, 0.02, 0.03), vec3(0.14, 0.11, 0.07), intensity);
    float hemiBlend = fragNormal.y * 0.5 + 0.5;
    vec3 ambientColor = mix(groundBounce, skyAmbient, hemiBlend);

    // Sky light modulates ambient — caves/interiors get reduced sky ambient
    // Softer gamma curve for gentler shadow transitions
    float skyFactor = fragSkyLight * fragSkyLight;
    vec3 ambient = ambientColor * mix(0.10, 0.40, intensity) * max(skyFactor, 0.08);

    // === Directional sun diffuse ===
    float NdotL = max(dot(fragNormal, sunDir), 0.0);
    // Softened shadow: wider ramp so transition from shade to sun is gradual
    float shadowTerm = smoothstep(0.15, 1.0, fragSkyLight);
    float diffuse = NdotL * 0.60 * intensity * shadowTerm;

    // === Subsurface scattering (foliage translucency) ===
    float NdotL_back = max(dot(-fragNormal, sunDir), 0.0);
    float subsurface = NdotL_back * 0.15 * intensity * shadowTerm;

    // === Sky fill: scattered overhead light ===
    float skyFill = max(fragNormal.y, 0.0) * 0.12 * intensity * skyFactor;

    vec3 sunLight = ambient + sunTint * (diffuse + skyFill + subsurface);

    // === Ambient Occlusion — gentle contact darkening ===
    float ao = mix(0.42, 1.0, fragAO);
    float aoAmb = mix(0.35, 1.0, fragAO);
    // Blend: AO hits ambient harder, but overall softer than before
    float aoBlend = diffuse / max(diffuse + 0.20, 0.001);
    sunLight *= mix(aoAmb, ao, aoBlend);

    // === Block light (torches, emissives) — warm glow ===
    vec3 torchColor = vec3(1.0, 0.80, 0.50);
    float torchRaw = fragBlockLight * fragBlockLight;
    float torchBoost = mix(2.0, 0.9, intensity);  // brighter at night
    vec3 blockLighting = torchColor * torchRaw * torchBoost;

    // === Combine ===
    vec3 litColor = baseColor * sunLight + baseColor * blockLighting;

    // Night floor: never fully black — slightly lifted
    vec3 nightFloor = mix(vec3(0.02, 0.02, 0.04), vec3(0.035, 0.035, 0.035), intensity);
    litColor = max(litColor, baseColor * nightFloor);

    // === Tone map (softened ACES filmic) ===
    litColor = tonemap(litColor);

    // === Fog ===
    float fogFactor = clamp((pc.fog_end - fragDist) / (pc.fog_end - pc.fog_start), 0.0, 1.0);
    vec3 color = mix(pc.fog_color.rgb, litColor, fogFactor);

    outColor = vec4(color, 1.0);
}
