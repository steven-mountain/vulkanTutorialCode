#version 450
layout(location = 0) in vec2 inPosition;
layout(binding = 0) uniform UniformBufferObject{
    mat4 view;
    mat4 projection;
}ubo[];

layout(binding = 1) uniform UboInstance{
    mat4 model;
}uboInstance[];

void main() {
    gl_Position = ubo[0].projection * ubo[0].view * uboInstance[0].model * vec4(inPosition, 0.0, 1.0);
//    gl_Position = vec4(inPosition, 0.0, 1.0);
}