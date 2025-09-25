// height_normals

[[vk::binding(0, 0)]]
Sampler2D heightMap;
// Texture2D<float> heightmap;

// [[vk::binding(1, 0)]]
// SamplerState heightmapSampler;

[[vk::binding(1, 0)]]
RWTexture2D<float4> outNormalMap;

[push_constant]
cbuffer NormalMapParams {
    float strength;
    float _padding;
    float2 _padding2;
};

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint width, height;
    outNormalMap.GetDimensions(width, height);

    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
    {
        return;
    }


    // const float strength = 8.0f;

    float2 texelSize = 1.0f / float2(width, height);
    float2 uv = (float2(dispatchThreadID.xy) + 0.5f) * texelSize;
    float h_left = heightMap.SampleLevel(uv - float2(texelSize.x, 0), 0).r;
    float h_right = heightMap.SampleLevel(uv + float2(texelSize.x, 0), 0).r;
    float h_top = heightMap.SampleLevel(uv - float2(texelSize.x, 0), 0).r;
    float h_bottom = heightMap.SampleLevel(uv + float2(texelSize.x, 0), 0).r;


    float dx = (h_right - h_left) * strength;
    float dy = (h_bottom - h_top) * strength;

    float3 normal = normalize(float3(-dx, -dy, 1.0f));
    float3 encodedNormal = normal * 0.5f + 0.5f;

    outNormalMap[dispatchThreadID.xy] = float4(encodedNormal, 1.0f);
}
