// heightmap.slang
struct HeightMapParams
{
    int seed;
    float2 offset;
    float frequency;
    int octaves;
    float lacunarity;
    float persistence;
};

[[vk::binding(0, 0)]]
RWTexture2D<float> outNoise;

[[vk::binding(1, 0)]]
ConstantBuffer<HeightMapParams> params;

uint hash(uint state)
{
    state ^= state >> 16;
    state *= 0x7feb352d;
    state ^= state >> 15;
    state *= 0x846ca68b;
    state ^= state >> 16;
    return state;
}

uint hash2(int ix, int iy)
{
    uint state = params.seed + ix * 73856093 + iy * 19349663;
    return hash(state);
}

// smooth interpolation (6t^5 - 15t^4 + 10t^3)
float fade(float t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

float2 getGradient(int h)
{
    float random = (float)(h & 0xFFFF) / 65535.0f;

    float angle = random * 2.0 * 3.14159265f;

    return float2(cos(angle), sin(angle));
}

float dotGridGradient(int ix, int iy, float x, float y)
{
    int h = hash2(ix, iy);
    float2 gradient = getGradient(h);

    float dx = x - ix;
    float dy = y - iy;

    return dx * gradient.x + dy * gradient.y;
}

float PerlinNoise(float x, float y)
{
    int x0 = (int)floor(x);
    int x1 = x0 + 1;
    int y0 = (int)floor(y);
    int y1 = y0 + 1;

    float sx = fade(x - x0);
    float sy = fade(y - y0);

    float n0 = dotGridGradient(x0, y0, x, y);
    float n1 = dotGridGradient(x1, y0, x, y);
    float n2 = dotGridGradient(x0, y1, x, y);
    float n3 = dotGridGradient(x1, y1, x, y);

    float ix0 = lerp(n0, n1, sx);
    float ix1 = lerp(n2, n3, sx);

    return lerp(ix0, ix1, sy);
}

float fractalNoise(float x, float y)
{
    float total = 0.0;
    float frequency = params.frequency;
    float amplitude = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < params.octaves; i++)
    {
        total += PerlinNoise(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= params.persistence; // amplitude decrease with each octave
        frequency *= params.lacunarity; // frequency increase with each octave
    }

    return total / maxValue;
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

    float x = (float)dispatchThreadID.x + params.offset.x;
    float y = (float)dispatchThreadID.y + params.offset.y;

    float noiseValue = fractalNoise(x, y);

    float finalValue = (noiseValue + 1.0) / 2.0; // map noise from [-1, 1] to [0, 1]

    outNoise[dispatchThreadID.xy] = finalValue;
}