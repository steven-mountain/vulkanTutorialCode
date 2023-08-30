#version 450
layout(location = 0) in vec2 inPosition;
layout(set = 0, binding = 0) uniform UniformBufferObject{
    mat4 view;
    mat4 projection;
}ubo[];

layout(set = 1, binding = 0) uniform UboInstance{
    mat4 model;
}uboInstance[];

void main() {
    gl_Position = ubo[0].projection * ubo[0].view * uboInstance[0].model * vec4(inPosition, 0.0, 1.0);
}