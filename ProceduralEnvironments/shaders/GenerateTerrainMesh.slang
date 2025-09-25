

struct Vertex
{
    float3 pos;
    float3 normal;
    float2 texCoord;
    float4 color;
}

[[vk::binding(0, 0)]]
Sampler2D heightMap;

[[vk::binding(1, 0)]]
RWStructuredBuffer<Vertex> outVertices;

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> outIndices;

[push_constant]
cbuffer TerrainParams
{
    float terrainSideLength;
    float heightScale;
    uint gridResolution;
    float normalsStrength;
};


float getHeight(int2 coord) {
    coord = clamp(coord, int2(0, 0), int2(gridResolution - 1, gridResolution - 1));
    float2 uv = float2(coord) / float(gridResolution - 1);
    return heightMap.SampleLevel(uv, 0).r;
}

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= gridResolution || dispatchThreadID.y >= gridResolution)
    {
        return;
    }

    uint vertexIndex = dispatchThreadID.y * gridResolution + dispatchThreadID.x;

    float2 uv = float2(dispatchThreadID.xy) / (gridResolution - 1.0f);
    // float height = getHeight(int2(dispatchThreadID.xy));
    float height = heightMap.SampleLevel(uv, 0).r;

    float halfSide = terrainSideLength / 2.0f;
    float x = uv.x * terrainSideLength - halfSide;
    float z = uv.y * terrainSideLength - halfSide;

    Vertex v;
    v.pos = float3(x, height * heightScale, z);
    v.texCoord = uv;
    v.color = float4(uv.x, uv.y, 0.5, 1.0); // placeholder

    float h_left = getHeight(int2(dispatchThreadID.x - 1, dispatchThreadID.y));
    float h_right = getHeight(int2(dispatchThreadID.x + 1, dispatchThreadID.y));
    float h_down = getHeight(int2(dispatchThreadID.x, dispatchThreadID.y - 1));
    float h_up = getHeight(int2(dispatchThreadID.x, dispatchThreadID.y + 1));

    float dx = (h_right - h_left) * heightScale;
    float dz = (h_up - h_down) * heightScale;
    float pixelWidth = terrainSideLength / gridResolution;

    v.normal = normalize(float3(-dx, 2.0 * pixelWidth, -dz));

    outVertices[vertexIndex] = v;

    // Each thread (except those on the right/bottom edge) generates indices for the quad to its bottom-right
    if (dispatchThreadID.x < gridResolution - 1 && dispatchThreadID.y < gridResolution - 1)
    {
        uint quadIndex = (dispatchThreadID.y * (gridResolution - 1) + dispatchThreadID.x) * 6;

        uint topLeft = vertexIndex;
        uint topRight = topLeft + 1;
        uint bottomLeft = topLeft + gridResolution;
        uint bottomRight = bottomLeft + 1;

        // First triangle
        outIndices[quadIndex + 0] = topLeft;
        outIndices[quadIndex + 1] = bottomLeft;
        outIndices[quadIndex + 2] = topRight;

        // Second triangle
        outIndices[quadIndex + 3] = topRight;
        outIndices[quadIndex + 4] = bottomLeft;
        outIndices[quadIndex + 5] = bottomRight;
    }
}

