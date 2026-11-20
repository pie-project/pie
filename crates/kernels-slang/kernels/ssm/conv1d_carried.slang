#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/pool.slang"

// The causal depthwise conv's decode step over a lane's buffered rows then
// its own (`prim/carry.slang`), SiLU after, a pair of channels (one bf16
// word) a lane, a lane of the fire a grid row. Position `t` of the lane's
// run is a buffered row (`t < n`, its input read from the lane's rs buffer
// pages where it lies) or its own (`t = n`); positions below 0 are the
// committed history's rows (`ssm/conv1d.slang` keeps `hist = (conv_width -
// 1) * dilation + 1` of them, f32). The own row's output is its window's
// sum, from at most `(conv_width - 1) * dilation` buffered rows; the history
// is kept as it stands after `commit` rows of the run (none moved where
// that is zero), never past a row the lane has not committed. The own row's
// input lands in its buffer (`x_at`), whatever its state.

#define HIST_MAX 64

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<uint> weight;
[[vk::binding(2)]] StructuredBuffer<int> slots;
[[vk::binding(3)]] StructuredBuffer<int> carry;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;
// The history bank and the rs list, in the chunk pool's segments, as words.
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, 5)

struct Params {
    uint seg_words;
    uint state_origin;
    uint slot_stride;
    uint pieces;
    uint piece_first;
    uint piece_atoms;
    // The window's first lane in the fire's tables (`slots`, `carry`).
    int lane0;
    int channels;
    int conv_width;
    int dilation;
    // The fire row the window's row 0 is.
    uint row0;
    uint page_tokens;
    uint stride_words;
    // Where the input plane lies in a page, and the words a token of it.
    uint x_at;
    uint x_row;
};
PIE_PARAMS(9, Params, params);

PIE_POOL_READ(uint, params.seg_words)
PIE_POOL_WRITE(uint, params.seg_words)
PIE_STATE_PIECES

#include "prim/carry.slang"

float silu(float z) {
    return z / (1.0 + exp(-z));
}

float2 pair(uint word) {
    return float2(bf16_to_f32(word & 0xffffu), bf16_to_f32(word >> 16));
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = 2u * gid.x;
    uint r = gid.y;
    uint chans = uint(params.channels);
    if (c >= chans) {
        return;
    }
    uint lane = uint(params.lane0) + r;
    uint row = params.row0 + r;
    CarryLane cl = carry_lane(lane);
    uint own_word = x[(row * chans + c) >> 1];
    if (cl.at >= 0) {
        pie_pool_set(carry_word(cl, uint(cl.at), params.x_at, params.x_row) + (c >> 1), own_word);
    }
    if (pie_state_none(lane)) {
        return;
    }
    int width = params.conv_width;
    int dil = params.dilation;
    int hist = (width - 1) * dil + 1;
    int n = cl.n;
    int commit = min(cl.commit, n + 1);

    // Channel `c`'s history in the record (`ssm/conv1d.slang`'s column),
    // found once: a slot's history lies in one segment.
    uint slab = pie_piece_unit(lane, c) * params.slot_stride;
    uint pitch = min(params.piece_atoms, chans);
    uint col = c % params.piece_atoms;
    uint seg, first;
    pie_pool_find(params.state_origin + slab, seg, first);
    float2 past[HIST_MAX];
#define CONV_CARRIED_LOAD(buf_) \
    for (int s = 0; s < hist; s++) { \
        uint i = first + uint(s) * pitch + col; \
        past[s] = float2(asfloat(buf_[i]), asfloat(buf_[i + 1u])); \
    }
    PIE_POOL_BRANCH(seg, CONV_CARRIED_LOAD)
    float2 own = pair(own_word);

    // Run position `t`'s input pair.
#define CONV_AT(t_, out_) \
    { \
        int t = (t_); \
        if (t < 0) { \
            out_ = past[hist + t]; \
        } else if (t < n) { \
            uint w = carry_word(cl, carry_tok(cl, uint(t)), params.x_at, params.x_row) + (c >> 1); \
            out_ = pair(pie_pool_get(w)); \
        } else { \
            out_ = own; \
        } \
    }

    uint tap0 = c * uint(width);
    uint tap1 = tap0 + uint(width);
    float2 acc = float2(0.0);
    for (int k = 0; k < width; k++) {
        float2 tap;
        CONV_AT(n - (width - 1 - k) * dil, tap)
        float2 w = float2(bf16_at(weight, tap0 + uint(k)), bf16_at(weight, tap1 + uint(k)));
        acc += tap * w;
    }
    y[(row * chans + c) >> 1] = pack_bf16(silu(acc.x), silu(acc.y));
    if (commit <= 0) {
        return;
    }
    // Every position read before any is written: the window as of `commit`.
    float2 kept[HIST_MAX];
    for (int s = 0; s < hist; s++) {
        CONV_AT(commit - hist + s, kept[s])
    }
#define CONV_CARRIED_STORE(buf_) \
    for (int s = 0; s < hist; s++) { \
        uint i = first + uint(s) * pitch + col; \
        buf_[i] = asuint(kept[s].x); \
        buf_[i + 1u] = asuint(kept[s].y); \
    }
    PIE_POOL_BRANCH(seg, CONV_CARRIED_STORE)
}

// pie:instantiate ssm_conv1d_carried_bf16 PIE_GROUP_X=128
