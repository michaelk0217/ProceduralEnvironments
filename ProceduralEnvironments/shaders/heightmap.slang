// heightmap.slang


[[vk::binding(0, 0)]]
RWTexture2D<float> outNoise;


[push_constant]
cbuffer HeightMapParams
{
    int seed;
    float2 offset;
    float frequency;
    int octaves;
    float lacunarity;
    float persistence;
    float noiseScale;
};



uint hash(int2 p_int, int seed)
{
    uint2 p = uint2(p_int);
    p += seed;
    p = p * uint2(1664525, 1013904223);
    p.x += p.y;
    p.y += p.x;
    p = p ^ (p >> 16);
    p.x += p.y;
    p.y += p.x;
    p = p ^ (p >> 16);
    return p.x;
}

float2 getGradient(uint n)
{
    switch (n & 3) 
    {
    case 0: return float2(1.0, 1.0);
    case 1: return float2(-1.0, 1.0);
    case 2: return float2(1.0, -1.0);
    case 3: return float2(-1.0, -1.0);
    default: return float2(0.0, 0.0); 
    }
}

float PerlinNoise(float2 p)
{
    float2 i = floor(p); // Integer grid cell coordinates
    float2 f = frac(p);  // Fractional coordinates within the cell

    // 6t^5 - 15t^4 + 10t^3
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    int2 pi = int2(i);

    uint h00 = hash(pi + int2(0, 0), seed);
    uint h10 = hash(pi + int2(1, 0), seed);
    uint h01 = hash(pi + int2(0, 1), seed);
    uint h11 = hash(pi + int2(1, 1), seed);


    float d00 = dot(getGradient(h00), f - float2(0.0, 0.0));
    float d10 = dot(getGradient(h10), f - float2(1.0, 0.0));
    float d01 = dot(getGradient(h01), f - float2(0.0, 1.0));
    float d11 = dot(getGradient(h11), f - float2(1.0, 1.0));

    return lerp(lerp(d00, d10, u.x),
                lerp(d01, d11, u.x),
                u.y);
}

float fractalNoise(float x, float y, float width, float height)
{
    
    float total = 0.0;
    float freq = frequency;
    float amplitude = 1.0;
    float maxAmplitude = 0.0; // Used for normalization

    

    float halfWidth = width / 2.0;
    float halfHeight = height / 2.0f;

    for (int i = 0; i < octaves; i++)
    {
        // float2 octaveOffset = rng_range(float2(x, y), -100.0f, 100.0f);
        float sampleX = (x - halfWidth) / noiseScale;
        float sampleY = (y - halfHeight) / noiseScale;
        

        float2 p = float2(sampleX, sampleY);

        total += PerlinNoise(p * freq) * amplitude;

        maxAmplitude += amplitude;

        freq *= lacunarity;
        amplitude *= persistence;
    }

    if (maxAmplitude == 0.0)
    {
        return 0.0;
    }
    return total / maxAmplitude;
}



[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint width, height;
    outNoise.GetDimensions(width, height);

    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
    {
        return;
    }

    float x = (float)dispatchThreadID.x + offset.x;
    float y = (float)dispatchThreadID.y + offset.y;

    float noiseValue = fractalNoise(x, y, width, height);
    
    float finalValue = (noiseValue + 1.0) / 2.0;

    outNoise[dispatchThreadID.xy] = finalValue;
}