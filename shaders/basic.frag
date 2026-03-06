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

// Gentle filmic tonemap — lifted shadows, soft highlight rolloff
vec3 tonemap(vec3 x) {
    // Tuned for low contrast: raised toe (b=0.06), gentle shoulder
    float a = 2.20;
    float b = 0.06;
    float c = 2.20;
    float d = 0.65;
    float e = 0.16;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec4 texSample = texture(texAtlas, fragUV);

    // Alpha test — foliage cutout
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

    // === Hemispherical ambient (sky dome + ground bounce) ===
    // This is the PRIMARY light source for shadow-side faces, so it must be strong
    vec3 skyAmbient = mix(vec3(0.06, 0.06, 0.10), vec3(0.42, 0.48, 0.60), intensity);
    vec3 groundBounce = mix(vec3(0.04, 0.03, 0.04), vec3(0.18, 0.14, 0.09), intensity);
    float hemiBlend = fragNormal.y * 0.5 + 0.5;
    vec3 ambientColor = mix(groundBounce, skyAmbient, hemiBlend);

    // Sky light modulates ambient — generous floor so even caves aren't pitch black
    float skyFactor = pow(fragSkyLight, 1.5); // gentler curve than squaring
    vec3 ambient = ambientColor * mix(0.15, 0.50, intensity) * max(skyFactor, 0.12);

    // === Directional sun diffuse ===
    float NdotL = max(dot(fragNormal, sunDir), 0.0);
    // Wrap-around diffuse: back faces still catch ~20% of directional light
    // This is the key fix for crushed blacks on shadow-side block faces
    float wrapDiffuse = max(dot(fragNormal, sunDir) * 0.5 + 0.5, 0.0);
    float shadowTerm = smoothstep(0.1, 1.0, fragSkyLight);
    float diffuse = mix(wrapDiffuse * 0.18, NdotL * 0.55, 0.75) * intensity * shadowTerm;

    // === Subsurface scattering (foliage translucency) ===
    float NdotL_back = max(dot(-fragNormal, sunDir), 0.0);
    float subsurface = NdotL_back * 0.12 * intensity * shadowTerm;

    // === Sky fill: scattered overhead light ===
    float skyFill = max(fragNormal.y, 0.0) * 0.12 * intensity * skyFactor;

    vec3 sunLight = ambient + sunTint * (diffuse + skyFill + subsurface);

    // === Ambient Occlusion — subtle contact darkening only ===
    // Much gentler than before: min 0.55 means darkest corner is still 55% lit
    float ao = mix(0.55, 1.0, fragAO);
    sunLight *= ao;

    // === Block light (torches, emissives) — warm glow ===
    vec3 torchColor = vec3(1.0, 0.80, 0.50);
    float torchRaw = fragBlockLight * fragBlockLight;
    float torchBoost = mix(2.0, 0.9, intensity);
    vec3 blockLighting = torchColor * torchRaw * torchBoost;

    // === Combine ===
    vec3 litColor = baseColor * sunLight + baseColor * blockLighting;

    // Shadow floor: guarantees nothing goes truly black, even at night
    vec3 shadowFloor = mix(vec3(0.03, 0.03, 0.06), vec3(0.05, 0.05, 0.05), intensity);
    litColor = max(litColor, baseColor * shadowFloor);

    // === Tone map (gentle filmic) ===
    litColor = tonemap(litColor);

    // === Fog ===
    float fogFactor = clamp((pc.fog_end - fragDist) / (pc.fog_end - pc.fog_start), 0.0, 1.0);
    vec3 color = mix(pc.fog_color.rgb, litColor, fogFactor);

    outColor = vec4(color, 1.0);
}
