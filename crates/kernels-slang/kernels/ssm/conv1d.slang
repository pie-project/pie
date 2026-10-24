#include "prim/bf16.slang"
#include "prim/params.slang"

// The causal depthwise conv over a recurrent slot's history, SiLU after, a
// pair of channels (one bf16 word) a lane. A slot keeps the last `hist =
// (conv_width - 1) * dilation + 1` input rows of its request in f32, `hist`
// rows of `channels`, and the conv rolls them in place.
//
// The default form is one decode step a row (`slots[row]`). `PIE_CHUNKED`
// walks a request's rows of the query CSR `indptr` (its slot at its first
// row). `PIE_COMMITTED` walks a request's rows of a speculative window: its
// rows are the CSR's plus `replay[lane0 + r]` replayed ones before them, and
// the slot keeps the history as of row `commit[lane0 + r]` (none when that is
// zero; no slot when `slots[lane0 + r] < 0`).
//
// Weights are `[channels, conv_width]`, the taps of channel `c` at
// `c * conv_width`. The grid is (channel pairs, rows or requests).

#define HIST_MAX 64

#if defined(PIE_COMMITTED)
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<int> replay;
[[vk::binding(3)]] StructuredBuffer<int> commit;
[[vk::binding(4)]] StructuredBuffer<int> slots;
[[vk::binding(5)]] StructuredBuffer<uint> weight;
[[vk::binding(6)]] RWStructuredBuffer<float> state;
[[vk::binding(7)]] RWStructuredBuffer<uint> y;
#define PARAMS_AT 8
#elif defined(PIE_CHUNKED)
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<uint> weight;
[[vk::binding(3)]] RWStructuredBuffer<float> state;
[[vk::binding(4)]] StructuredBuffer<uint> slots;
[[vk::binding(5)]] RWStructuredBuffer<uint> y;
#define PARAMS_AT 6
#else
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<uint> weight;
[[vk::binding(2)]] RWStructuredBuffer<float> state;
[[vk::binding(3)]] StructuredBuffer<uint> slots;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;
#define PARAMS_AT 5
#endif

struct Params {
#if defined(PIE_COMMITTED)
    int lane0;
#endif
    int channels;
    int conv_width;
    int dilation;
};
PIE_PARAMS(PARAMS_AT, Params, params);

float silu(float z) {
    return z / (1.0 + exp(-z));
}

// Channels `c` and `c + 1` of row `row`.
float2 x_at(uint row, uint chans, uint c) {
    uint word = x[(row * chans + c) >> 1];
    return float2(bf16_to_f32(word & 0xffffu), bf16_to_f32(word >> 16));
}

void y_put(uint row, uint chans, uint c, float2 v) {
    y[(row * chans + c) >> 1] = pack_bf16(silu(v.x), silu(v.y));
}

// The slot's `hist` rows of channels `c`, `c + 1`.
void past_load(uint slab, int hist, uint chans, uint c, out float2 past[HIST_MAX]) {
    for (int s = 0; s < hist; s++) {
        uint at = slab + uint(s) * chans + c;
        past[s] = float2(state[at], state[at + 1u]);
    }
}

void past_put(uint slab, int s, uint chans, uint c, float2 v) {
    uint at = slab + uint(s) * chans + c;
    state[at] = v.x;
    state[at + 1u] = v.y;
}

#if defined(PIE_COMMITTED) || defined(PIE_CHUNKED)
// Rows `begin .. begin + span` convolved over the slot's history, which then
// rolls to the history as of row `keep` (none when `keep <= 0`).
void walk(uint slab, int begin, int span, int keep, uint chans, uint c) {
    int width = params.conv_width;
    int dil = params.dilation;
    int hist = (width - 1) * dil + 1;
    uint tap0 = c * uint(width);
    uint tap1 = tap0 + uint(width);
    float2 past[HIST_MAX];
    past_load(slab, hist, chans, c, past);
    for (int t = 0; t < span; t++) {
        float2 acc = float2(0.0);
        for (int k = 0; k < width; k++) {
            int src = t - (width - 1 - k) * dil;
            float2 tap = src < 0 ? past[hist + src] : x_at(uint(begin + src), chans, c);
            float2 w = float2(bf16_at(weight, tap0 + uint(k)), bf16_at(weight, tap1 + uint(k)));
            acc += tap * w;
        }
        y_put(uint(begin + t), chans, c, acc);
    }
    if (keep <= 0) {
        return;
    }
    for (int s = 0; s < hist; s++) {
        int src = keep - hist + s;
        past_put(slab, s, chans, c, src < 0 ? past[hist + src] : x_at(uint(begin + src), chans, c));
    }
}
#endif

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = 2u * gid.x;
    uint r = gid.y;
    uint chans = uint(params.channels);
    if (c >= chans) {
        return;
    }
    int width = params.conv_width;
    int hist = (width - 1) * params.dilation + 1;

#if defined(PIE_COMMITTED)
    uint lane = uint(params.lane0) + r;
    int begin = indptr[r];
    for (uint j = 0; j < r; j++) {
        begin += replay[uint(params.lane0) + j];
    }
    int span = (indptr[r + 1] - indptr[r]) + replay[lane];
    int slot = slots[lane];
    if (span <= 0 || slot < 0) {
        return;
    }
    walk(uint(slot) * uint(hist) * chans, begin, span, min(commit[lane], span), chans, c);
#elif defined(PIE_CHUNKED)
    int begin = indptr[r];
    int span = indptr[r + 1] - begin;
    if (span <= 0) {
        return;
    }
    walk(slots[uint(begin)] * uint(hist) * chans, begin, span, span, chans, c);
#else
    uint slab = slots[r] * uint(hist) * chans;
    float2 past[HIST_MAX];
    past_load(slab, hist, chans, c, past);
    uint tap0 = c * uint(width);
    uint tap1 = tap0 + uint(width);
    float2 fresh = x_at(r, chans, c);
    float2 acc = float2(0.0);
    for (int k = 0; k + 1 < width; k++) {
        float2 w = float2(bf16_at(weight, tap0 + uint(k)), bf16_at(weight, tap1 + uint(k)));
        acc += past[k * params.dilation + 1] * w;
    }
    float2 last = float2(bf16_at(weight, tap0 + uint(width - 1)), bf16_at(weight, tap1 + uint(width - 1)));
    acc += fresh * last;
    y_put(r, chans, c, acc);
    for (int s = 0; s + 1 < hist; s++) {
        past_put(slab, s, chans, c, past[s + 1]);
    }
    past_put(slab, hist - 1, chans, c, fresh);
#endif
}

// pie:instantiate ssm_conv1d_bf16 PIE_GROUP_X=128
// pie:instantiate ssm_conv1d_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1
// pie:instantiate ssm_conv1d_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1
