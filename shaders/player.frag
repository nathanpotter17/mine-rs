#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in float fragDist;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple sun-like directional light (matches approximate noon direction)
    vec3 lightDir = normalize(vec3(0.3, 0.8, 0.2));
    float NdotL = max(dot(fragNormal, lightDir), 0.0);

    // Ambient + diffuse
    float ambient = 0.40;
    float diffuse = NdotL * 0.55;

    // Sky fill — upward faces catch extra light
    float skyFill = max(fragNormal.y, 0.0) * 0.10;

    vec3 lit = fragColor * (ambient + diffuse + skyFill);

    // Subtle distance fog to match world (fade toward white-ish at distance)
    float fogFactor = smoothstep(80.0, 200.0, fragDist);
    vec3 fogColor = vec3(0.7, 0.75, 0.85);
    lit = mix(lit, fogColor, fogFactor * 0.5);

    outColor = vec4(lit, 1.0);
}
