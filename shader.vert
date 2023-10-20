#version 450

layout(binding = 0) uniform UniformBufferObject{
    mat4 model;
    mat4 view;
    mat4 projection;
}ubo[];
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 texCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 coord;

void main() {
    gl_Position = ubo[0].projection * ubo[0].view * ubo[0].model * vec4(inPosition, 1.0);
    fragColor = inColor;
    coord = texCoord;
}