// shader.slang
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

[[vk::binding(0, 0)]]
ConstantBuffer<MVPMatrices> mvpBuffer;


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
    [[vk::location(1)]] float3 worldNormal : NORMAL;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
    [[vk::location(3)]] float4 color : COLOR;
};


[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.worldPos = (mul(mvpBuffer.model, float4(input.position, 1.0))).xyz;
    output.position = mul(mvpBuffer.mvp, float4(input.position, 1.0));
    output.worldNormal = normalize(mul((float3x3)mvpBuffer.model, input.normal));
    output.texCoord = input.texCoord;
    output.color = input.color;

    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{

    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));
    float3 normal = normalize(input.worldNormal);

    float diffuse = max(dot(normal, lightDir), 0.0);
    float ambient = 0.2;

    float3 finalColor = float3(0.07, 0.03, 0.00) * (diffuse + ambient);

    return float4(finalColor, 1.0);
    // return input.color;
}