[[vk::binding(0)]] StructuredBuffer<float> t;
[[vk::binding(1)]] RWStructuredBuffer<float> o;

struct Push { uint dim; float max_period; uint flip_sin_cos; float scale; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint half_dim = pc.dim / 2u;
    uint i = gid.x;
    uint row = gid.y;
    if (i >= half_dim) return;
    float freq = exp(-log(pc.max_period) * float(i) / float(half_dim));
    float angle = pc.scale * (t[row] * freq);
    // Fold the angle into one turn: GPU sin and cos are accurate only near zero.
    const float TURN = 6.283185307179586;
    angle = angle - TURN * round(angle / TURN);
    float s = sin(angle);
    float c = cos(angle);
    uint base = row * pc.dim;
    o[base + i] = pc.flip_sin_cos != 0u ? c : s;
    o[base + i + half_dim] = pc.flip_sin_cos != 0u ? s : c;
    if ((pc.dim & 1u) != 0u && i == 0u) o[base + pc.dim - 1u] = 0.0;
}

// pie:instantiate sinusoid_f32
