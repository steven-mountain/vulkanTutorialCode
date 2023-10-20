#version 450
#extension GL_KHR_vulkan_glsl : enable
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 coord;

layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 1) uniform sampler2D texSampler;

void main() {
    outColor = texture(texSampler, coord);
}
