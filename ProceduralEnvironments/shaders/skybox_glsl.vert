#version 450

layout(binding = 0) uniform MVPBuffer {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 mvp;
    mat4 modelInverse;
    mat4 viewInverse;
    mat4 projectionInverse;
} mvpBuffer;

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 outTexCoord;

void main() {
    outTexCoord = inPosition;
    mat4 viewNoTranslation = mvpBuffer.view;
    viewNoTranslation[3] = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 pos = mvpBuffer.projection * viewNoTranslation * vec4(inPosition, 1.0);
    gl_Position = pos.xyww;
}