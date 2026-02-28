#version 450

layout(push_constant) uniform PushConstants {
    vec4 fog_color;       // offset 0
    float fog_start;      // offset 16
    float fog_end;        // offset 20
    float time_of_day;    // offset 24  (0.0=midnight, 0.5=noon)
    float sun_intensity;  // offset 28  (0.0=night, 1.0=full day)
} pc;

layout(location = 0) in vec3 fragDir;
layout(location = 0) out vec4 outColor;

// Shared sun direction computation — must match basic.frag
vec3 computeSunDir(float tod) {
    float angle = tod * 6.2831853;  // full rotation over 24h cycle
    return normalize(vec3(cos(angle), sin(angle), 0.25));
}

void main() {
    vec3 dir = normalize(fragDir);
    vec3 sunDir = computeSunDir(pc.time_of_day);
    float sunElev = sunDir.y;  // how high sun is above horizon

    // === Sky gradient ===
    float elevation = dir.y;

    // Day colors
    vec3 dayZenith  = vec3(0.22, 0.45, 0.85);
    vec3 dayHorizon = vec3(0.65, 0.78, 0.95);

    // Night colors
    vec3 nightZenith  = vec3(0.01, 0.01, 0.04);
    vec3 nightHorizon = vec3(0.03, 0.04, 0.08);

    // Blend day/night
    float dayFactor = pc.sun_intensity;
    vec3 zenith  = mix(nightZenith,  dayZenith,  dayFactor);
    vec3 horizon = mix(nightHorizon, dayHorizon, dayFactor);

    // Sunrise/sunset warm horizon band
    float sunsetBand = smoothstep(-0.05, 0.15, sunElev)
                     * (1.0 - smoothstep(0.15, 0.45, sunElev));
    vec3 sunsetColor = vec3(0.95, 0.45, 0.15);
    horizon = mix(horizon, sunsetColor, sunsetBand * 0.7);
    zenith  = mix(zenith, vec3(0.6, 0.3, 0.2), sunsetBand * 0.25);

    // Vertical gradient
    float t = clamp(elevation * 1.5, 0.0, 1.0);
    vec3 sky = mix(horizon, zenith, t);

    // Below horizon — gentle darkening that stays CLOSE to horizon color.
    // FIX: Old code transitioned to brown (0.22, 0.20, 0.18) which clashed
    // with fog. Now the ground color is derived from horizon with slight
    // desaturation, and the transition is much softer (factor 2.0 not 4.0).
    if (elevation < 0.0) {
        // Ground tint: desaturated, slightly darker version of horizon
        vec3 ground = horizon * 0.7 + vec3(0.05, 0.04, 0.03);
        float belowT = clamp(-elevation * 2.0, 0.0, 1.0);
        sky = mix(sky, ground, belowT * 0.6);  // never fully replaces, stays close to horizon
    }

    // === Sun (square!) ===
    // Build tangent frame around sun direction
    vec3 sunRight = normalize(cross(sunDir, vec3(0.001, 1.0, 0.0)));
    vec3 sunUp    = cross(sunRight, sunDir);

    // Project view direction onto sun's tangent plane
    float dotMain = dot(dir, sunDir);
    float projR   = dot(dir, sunRight);
    float projU   = dot(dir, sunUp);

    // Square shape: Chebyshev distance (L-infinity norm)
    float angularSize = 0.045;  // half-angle in radians
    float squareDist  = max(abs(projR), abs(projU));

    if (squareDist < angularSize && dotMain > 0.0 && sunElev > -0.08) {
        // Sun color: warm at horizon, white at zenith
        vec3 sunColor = mix(vec3(1.0, 0.55, 0.2), vec3(1.0, 0.95, 0.85),
                            clamp(sunElev * 3.0, 0.0, 1.0));

        // Slight edge darkening for a blocky but defined look
        float edgeFade = smoothstep(angularSize, angularSize * 0.75, squareDist);
        sky = mix(sky, sunColor * 1.8, edgeFade * clamp(sunElev + 0.1, 0.0, 1.0));
    }

    // Sun glow (circular, around the square)
    float glowDist = length(vec2(projR, projU));
    float glow = exp(-glowDist * 18.0) * pc.sun_intensity * 0.35;
    vec3 glowColor = mix(vec3(1.0, 0.4, 0.15), vec3(1.0, 0.85, 0.6),
                         clamp(sunElev * 2.0, 0.0, 1.0));
    sky += glowColor * glow;

    // Night stars (simple procedural)
    if (dayFactor < 0.3 && elevation > 0.05) {
        vec3 absDir = abs(dir * 200.0);
        float star = fract(sin(dot(floor(absDir.xz), vec2(12.9898, 78.233))) * 43758.5453);
        float starBright = step(0.997, star) * (1.0 - dayFactor / 0.3) * elevation;
        sky += vec3(starBright * 0.8);
    }

    // Clamp to prevent fireflies
    sky = clamp(sky, vec3(0.0), vec3(2.5));

    outColor = vec4(sky, 1.0);
}
