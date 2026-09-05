#include "common/bf16.slang"

[[vk::binding(0)]] StructuredBuffer<float> part_out;
[[vk::binding(1)]] StructuredBuffer<float> part_lse;
PIE_BUFFER_RW(2, out_);
#if defined(PIE_LSE)
[[vk::binding(3)]] RWStructuredBuffer<float> lse_out;
#endif

struct Push {
    int splits;
    int n_rows;
    int q_heads;
    int head_dim;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float PIE_SPLIT_NEG_INF = -3.0e38;

bool finite_(float x) {
    return abs(x) < 3.0e38;
}

bool already_written(float x) {
    return x > 3.0e38;
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint width = uint(pc.q_heads) * uint(pc.head_dim);
    uint e = gid.x;
    if (e >= width) return;
    uint row = gid.y;
    if (row >= uint(pc.n_rows)) return;

    uint head = e / uint(pc.head_dim);
    uint d = e - head * uint(pc.head_dim);
    uint rows = uint(pc.n_rows);
    uint heads = uint(pc.q_heads);

    if (already_written(part_lse[row * heads + head])) {
        return;
    }

    float merged_max = PIE_SPLIT_NEG_INF;
    for (int s = 0; s < pc.splits; s++) {
        float l = part_lse[(uint(s) * rows + row) * heads + head];
        if (finite_(l)) merged_max = max(merged_max, l);
    }

    if (!finite_(merged_max)) {
        out_[row * width + e] = PIE_STORE(0.0);
#if defined(PIE_LSE)
        if (d == 0u) lse_out[row * heads + head] = asfloat(0xff800000u);
#endif
        return;
    }

    float total = 0.0;
    float acc = 0.0;
    for (int s = 0; s < pc.splits; s++) {
        uint col = (uint(s) * rows + row) * heads + head;
        float l = part_lse[col];
        if (!finite_(l)) continue;
        float w = exp2(l - merged_max);
        total += w;
        acc += part_out[col * uint(pc.head_dim) + d] * w;
    }

    out_[row * width + e] = PIE_STORE(total == 0.0 ? 0.0 : acc / total);
#if defined(PIE_LSE)
    if (d == 0u) lse_out[row * heads + head] = merged_max + log2(total);
#endif
}

// pie:instantiate sdpa_split_fold_bf16
// pie:instantiate sdpa_split_fold_lse_bf16 PIE_LSE=1
