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

[[vk::binding(1, 0)]]
Texture2D<float> heightMap;

[[vk::binding(2, 0)]]
SamplerState heightmapSampler;

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

    // ----- Noraml Calculation -----
    uint width, height;
    heightMap.GetDimensions(width, height);
    float2 texelSize = float2(1.0 / width, 1.0 / height);

    float heightScale = 2.0f;

    float h_center = heightMap.SampleLevel(heightmapSampler, input.texCoord, 0) * heightScale;
    float h_right = heightMap.SampleLevel(heightmapSampler, input.texCoord + float2(texelSize.x, 0), 0) * heightScale;
    float h_up = heightMap.SampleLevel(heightmapSampler, input.texCoord + (float2(0, texelSize.y)), 0) * heightScale;

    float gridSpacing = 10.0 / 400.0;

    float3 p_center = float3(input.position.x, h_center, input.position.z);
    float3 p_right = float3(input.position.x + gridSpacing, h_right, input.position.z);
    float3 p_up = float3(input.position.x, h_up, input.position.z + gridSpacing);

    float3 tangent = p_right - p_center;
    float3 bitangent = p_up - p_center;

    float3 normal = normalize(cross(bitangent, tangent));
    
    output.worldNormal = normalize(mul((float3x3)transpose(mvpBuffer.modelInverse), normal));

    float3 displacedPosition = p_center;

    float4 worldPosition = mul(mvpBuffer.model, float4(displacedPosition, 1.0));
    output.worldPos = worldPosition.xyz;

    output.position = mul(mvpBuffer.mvp, float4(displacedPosition, 1.0));


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
}