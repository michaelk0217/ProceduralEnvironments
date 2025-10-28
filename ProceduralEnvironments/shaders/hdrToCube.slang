// hdrToCube.slang
[[vk::binding(0, 0)]]
Sampler2D equirectangularMap;

[[vk::binding(1, 0)]]
RWTexture2DArray<float4> cubeMap;

static const float PI = 3.14159265359;

float3 getDirectionForFace(uint3 id)
{
    uint width, height, layers;
    cubeMap.GetDimensions(width, height, layers);

    float2 uv = float2(id.xy) / float2(width, height);
    uv = uv * 2.0 - 1.0;

    float3 dir = float3(0.0, 0.0, 0.0);
    switch (id.z)
    {
    case 0: dir = float3(1.0, -uv.y, -uv.x); break; // +X
    case 1: dir = float3(-1.0, -uv.y, uv.x); break; // -X
    case 2: dir = float3(uv.x, 1.0, uv.y); break; // +Y
    case 3: dir = float3(uv.x, -1.0, -uv.y); break; // -Y
    case 4: dir = float3(uv.x, -uv.y, 1.0); break; // +Z
    case 5: dir = float3(-uv.x, -uv.y, -1.0); break; // -Z
    }

    return normalize(dir);
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float3 dir = getDirectionForFace(dispatchThreadID);

    float2 uv = float2(
        atan2(dir.z, dir.x) / (2.0 * PI) + 0.5,
        acos(dir.y) / PI
    );

    float4 color = equirectangularMap.SampleLevel(uv, 0);

    cubeMap[dispatchThreadID] = color;
}