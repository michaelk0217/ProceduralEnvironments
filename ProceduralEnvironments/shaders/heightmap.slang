// heightmap.slang
// struct HeightMapParams
// {
//     uint seed;
//     float2 offset;
//     float frequency;
//     uint octaves;
//     float lacunarity;
//     float persistence;
// };

[[vk::binding(0, 0)]]
RWTexture2D<float> outNoise;

// [[vk::binding(1, 0)]]
// ConstantBuffer<HeightMapParams> params;
[push_constant]
cbuffer HeightMapParams
{
    int seed;
    float2 offset;
    float frequency;
    int octaves;
    float lacunarity;
    float persistence;
    float _padding2;
};

uint hash(uint2 p, int seed)
{
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
    switch (n & 3) // Use the last 2 bits
    {
    case 0: return float2(1.0, 1.0);
    case 1: return float2(-1.0, 1.0);
    case 2: return float2(1.0, -1.0);
    case 3: return float2(-1.0, -1.0);
    default: return float2(0.0, 0.0); // Should not happen
    }
}

float PerlinNoise(float2 p)
{
    float2 i = floor(p); // Integer grid cell coordinates
    float2 f = frac(p);  // Fractional coordinates within the cell

    // Smooth the fractional coordinates using a quintic curve for better quality
    // 6t^5 - 15t^4 + 10t^3
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    uint2 pi = uint2(i);

    // Get a random hash for each of the four corners of the grid cell.
    uint h00 = hash(pi + uint2(0, 0), seed);
    uint h10 = hash(pi + uint2(1, 0), seed);
    uint h01 = hash(pi + uint2(0, 1), seed);
    uint h11 = hash(pi + uint2(1, 1), seed);

    // Calculate the influence of each corner's gradient on the point 'p'.
    // This is done by taking the dot product of the gradient and the distance vector.
    float d00 = dot(getGradient(h00), f - float2(0.0, 0.0));
    float d10 = dot(getGradient(h10), f - float2(1.0, 0.0));
    float d01 = dot(getGradient(h01), f - float2(0.0, 1.0));
    float d11 = dot(getGradient(h11), f - float2(1.0, 1.0));

    // Bilinearly interpolate the results using the smoothed fractional coordinates.
    return lerp(lerp(d00, d10, u.x),
                lerp(d01, d11, u.x),
                u.y);
}

// Fractal Noise (or Fractal Brownian Motion - fBm).
// This function sums multiple layers (octaves) of Perlin noise at different
// frequencies and amplitudes to create a more detailed and natural look.
float fractalNoise(float x, float y)
{
    float2 p = float2(x, y);
    float total = 0.0;
    float freq = frequency;
    float amplitude = 1.0;
    float maxAmplitude = 0.0; // Used for normalization

    // Loop through each octave
    for (int i = 0; i < octaves; i++)
    {
        // Add a layer of Perlin noise
        total += PerlinNoise(p * freq) * amplitude;

        // Keep track of the maximum possible amplitude to normalize the final result
        maxAmplitude += amplitude;

        // Increase frequency and decrease amplitude for the next octave
        freq *= lacunarity;
        amplitude *= persistence;
    }

    // Normalize to bring the result roughly into the [-1, 1] range
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

    float noiseValue = fractalNoise(x, y);
    
    float finalValue = (noiseValue + 1.0) / 2.0; // map noise from [-1, 1] to [0, 1]

    outNoise[dispatchThreadID.xy] = finalValue;
}