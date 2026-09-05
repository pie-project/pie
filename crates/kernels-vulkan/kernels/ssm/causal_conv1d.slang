#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

#define PIE_CONV_HIST_MAX 64

#if defined(PIE_COMMITTED)

PIE_BUFFER_RO(0, x);
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<int> replay;
[[vk::binding(3)]] StructuredBuffer<int> commit;
[[vk::binding(4)]] StructuredBuffer<int> slots;
PIE_BUFFER_RO(5, weight);
[[vk::binding(6)]] RWStructuredBuffer<float> state;
PIE_BUFFER_RW(7, y);
#elif defined(PIE_CHUNKED)
PIE_BUFFER_RO(0, x);
[[vk::binding(1)]] StructuredBuffer<int> indptr;
PIE_BUFFER_RO(2, weight);
[[vk::binding(3)]] RWStructuredBuffer<float> state;
[[vk::binding(4)]] StructuredBuffer<uint> slots;
PIE_BUFFER_RW(5, y);
#else
PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, weight);
[[vk::binding(2)]] RWStructuredBuffer<float> state;
[[vk::binding(3)]] StructuredBuffer<uint> slots;
PIE_BUFFER_RW(4, y);
#endif

struct Push {
#if defined(PIE_COMMITTED)
    int lane0;
#endif
    int channels;
    int conv_width;
    int dilation;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

float silu(float z) {
    return z / (1.0 + exp(-z));
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint r = gid.y;
    uint chans = uint(pc.channels);
    if (c >= chans) {
        return;
    }
    int width = pc.conv_width;
    int dil = pc.dilation;
    int hist = (width - 1) * dil + 1;
    uint tap0 = c * uint(width);

#if defined(PIE_COMMITTED)
    int begin = indptr[r];
    for (uint j = 0; j < r; j++) {
        begin += replay[uint(pc.lane0) + j];
    }
    int span = (indptr[r + 1] - indptr[r]) + replay[uint(pc.lane0) + r];
    if (span <= 0) {
        return;
    }
    int slot = slots[uint(pc.lane0) + r];
    if (slot < 0) {
        return;
    }
    uint slab = uint(slot) * uint(hist) * chans;
    float past[PIE_CONV_HIST_MAX];
    for (int s = 0; s < hist; s++) {
        past[s] = state[slab + uint(s) * chans + c];
    }
    for (int t = 0; t < span; t++) {
        float acc = 0.0;
        for (int k = 0; k < width; k++) {
            int src = t - (width - 1 - k) * dil;
            float tap = src < 0 ? past[hist + src]
                                : PIE_LOAD(x[uint(begin + src) * chans + c]);
            acc += tap * PIE_LOAD(weight[tap0 + uint(k)]);
        }
        y[uint(begin + t) * chans + c] = PIE_STORE(silu(acc));
    }
    int keep = min(commit[uint(pc.lane0) + r], span);
    if (keep <= 0) {
        return;
    }

    for (int s = 0; s < hist; s++) {
        int src = keep - hist + s;
        state[slab + uint(s) * chans + c] =
            src < 0 ? past[hist + src] : PIE_LOAD(x[uint(begin + src) * chans + c]);
    }
#elif defined(PIE_CHUNKED)
    int begin = indptr[r];
    int end = indptr[r + 1];
    if (end <= begin) {
        return;
    }
    int span = end - begin;
    uint slab = slots[uint(begin)] * uint(hist) * chans;

    float past[PIE_CONV_HIST_MAX];
    for (int s = 0; s < hist; s++) {
        past[s] = state[slab + uint(s) * chans + c];
    }
    for (int t = 0; t < span; t++) {
        float acc = 0.0;
        for (int k = 0; k < width; k++) {
            int src = t - (width - 1 - k) * dil;
            float tap = src < 0 ? past[hist + src]
                                : PIE_LOAD(x[uint(begin + src) * chans + c]);
            acc += tap * PIE_LOAD(weight[tap0 + uint(k)]);
        }
        y[uint(begin + t) * chans + c] = PIE_STORE(silu(acc));
    }
    for (int s = 0; s < hist; s++) {
        int src = span - hist + s;
        state[slab + uint(s) * chans + c] =
            src < 0 ? past[hist + src] : PIE_LOAD(x[uint(begin + src) * chans + c]);
    }
#else
    uint slab = slots[r] * uint(hist) * chans;
    float past[PIE_CONV_HIST_MAX];
    for (int s = 0; s < hist; s++) {
        past[s] = state[slab + uint(s) * chans + c];
    }
    float fresh = PIE_LOAD(x[r * chans + c]);
    float acc = 0.0;
    for (int k = 0; k + 1 < width; k++) {
        acc += past[k * dil + 1] * PIE_LOAD(weight[tap0 + uint(k)]);
    }
    acc += fresh * PIE_LOAD(weight[tap0 + uint(width - 1)]);
    y[r * chans + c] = PIE_STORE(silu(acc));
    for (int s = 0; s + 1 < hist; s++) {
        state[slab + uint(s) * chans + c] = past[s + 1];
    }
    state[slab + uint(hist - 1) * chans + c] = fresh;
#endif
}

// pie:instantiate causal_conv1d_bf16 PIE_GROUP_X=256
// pie:instantiate causal_conv1d_chunked_bf16 PIE_GROUP_X=256 PIE_CHUNKED=1
// pie:instantiate causal_conv1d_committed_bf16 PIE_GROUP_X=256 PIE_COMMITTED=1
