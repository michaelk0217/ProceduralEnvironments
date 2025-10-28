// skybox.slang
struct MVPMatrices
{
    column_major float4x4 model;
    column_major float4x4 view;
    column_major float4x4 projection;
    column_major float4x4 mvp;
    column_major float4x4 modelInverse;
    column_major float4x4 viewInverse;
    column_major float4x4 projectionInverse;
};

// struct MVPMatrices
// {
//     float4x4 model;
//     float4x4 view;
//     float4x4 projection;
//     float4x4 mvp;
//     float4x4 modelInverse;
//     float4x4 viewInverse;
//     float4x4 projectionInverse;
// };

[[vk::binding(0, 0)]]
ConstantBuffer<MVPMatrices> mvpBuffer;

[[vk::binding(1, 0)]]
SamplerCube skyboxTexture;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 texCoord : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;

    output.texCoord = input.position;

    float4x4 viewNoTranslation = mvpBuffer.view;
    viewNoTranslation[3] = float4(0.0, 0.0, 0.0, 1.0);

    // float4 pos = mul(mvpBuffer.projection, mul(viewNoTranslation, float4(input.position, 1.0)));
    float4 pos = mul(mvpBuffer.projection, mul( float4(input.position, 1.0), viewNoTranslation));

    output.position = pos.xyww;

    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return skyboxTexture.SampleLevel(input.texCoord, 0);
}