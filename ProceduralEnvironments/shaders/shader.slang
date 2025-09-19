// shader.slang
struct MVPMatrices
{
    column_major float4x4 model;
    column_major float4x4 view;
    column_major float4x4 projection;
    column_major float4x4 mvp;
};

[[vk::binding(0, 0)]]
ConstantBuffer<MVPMatrices> mvpBuffer;

[[vk::binding(1, 0)]]
Texture2D<float> heightMap;

[[vk::binding(2, 0)]]
SamplerState sampler;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
    [[vk::location(3)]] float4 color : COLOR;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPos : WORLD_POS;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
    [[vk::location(3)]] float4 color : COLOR;
};


[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;

    float height = heightMap.SampleLevel(sampler, input.texCoord, 0) * 8.0;

    float3 displacedPosition = input.position;
    displacedPosition.y += height;

    float4 worldPosition = mul(mvpBuffer.model, float4(displacedPosition, 1.0));
    output.worldPos = worldPosition.xyz;

    output.position = mul(mvpBuffer.mvp, float4(displacedPosition, 1.0));

    output.normal = normalize(mul((float3x3)mvpBuffer.model, input.normal));

    output.texCoord = input.texCoord;
    output.color = input.color;

    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return input.color;
}