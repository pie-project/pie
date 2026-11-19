#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/pool.slang"

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
// `PIE_REPLAY` is the decode step of a lane that carries rows (the lagged
// fold, `new-memory/designs/lagged-fold.md`): row `r` is lane `lane0 + r`'s
// one own row, and `lag` holds, a lane, `3 + carry` ints: its carried count
// `L` (at most `carry`), the rows `keep` of its `L + 1` whose history the
// slot keeps (none when 0), the rs-buffer row its own row lands in (none
// when negative), then its carried rows' rs-buffer rows in order. An
// rs-buffer row `u` is token `u % page_tokens` of rs page `u / page_tokens`,
// this conv's input row at pool word `page * page_words + plane_words +
// token * row_words`. The lane's history is loaded once, walked over its
// carried rows (read from the rs buffer, each one's output landed in
// `carried`, lane `l`'s row `j` at row `l * carry + j`, for the scan that
// replays them) and then its own row, and stored once, as of row `keep`.
// The pool is read as words here: the rs rows are bf16 pairs.
//
// `PIE_SNAPS` (with either): a request's lane also lands the history as of
// up to `PIE_RS_SNAPSHOTS` rows inside it in records of their own
// (`prim/pool.slang`, CONTRACT-BC 16).
//
// Weights are `[channels, conv_width]`, the taps of channel `c` at
// `c * conv_width`. The grid is (channel pairs, rows or requests).

#define HIST_MAX 64

#if defined(PIE_REPLAY)
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<uint> weight;
[[vk::binding(2)]] StructuredBuffer<uint> slots;
[[vk::binding(3)]] StructuredBuffer<int> lag;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;
[[vk::binding(5)]] RWStructuredBuffer<uint> carried;
#define SEGMENTS_AT 6
#elif defined(PIE_COMMITTED)
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<int> replay;
[[vk::binding(3)]] StructuredBuffer<int> commit;
[[vk::binding(4)]] StructuredBuffer<int> slots;
[[vk::binding(5)]] StructuredBuffer<uint> weight;
[[vk::binding(6)]] RWStructuredBuffer<uint> y;
#if defined(PIE_SNAPS)
[[vk::binding(7)]] StructuredBuffer<int> pie_snaps;
#define SEGMENTS_AT 8
#else
#define SEGMENTS_AT 7
#endif
#elif defined(PIE_CHUNKED)
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<uint> weight;
[[vk::binding(3)]] StructuredBuffer<uint> slots;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;
#if defined(PIE_SNAPS)
[[vk::binding(5)]] StructuredBuffer<int> pie_snaps;
#define SEGMENTS_AT 6
#else
#define SEGMENTS_AT 5
#endif
#else
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<uint> weight;
[[vk::binding(2)]] StructuredBuffer<uint> slots;
[[vk::binding(3)]] RWStructuredBuffer<uint> y;
#define SEGMENTS_AT 4
#endif
// The history bank, in the chunk pool's segments (`prim/pool.slang`).
#if defined(PIE_REPLAY)
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, SEGMENTS_AT)
#else
PIE_POOL_SEGMENTS(RWStructuredBuffer, float, SEGMENTS_AT)
#endif
#define PARAMS_AT (SEGMENTS_AT + 4)

struct Params {
    uint seg_words;
    uint state_origin;
    uint slot_stride;
    uint pieces;
    uint piece_first;
    uint piece_atoms;
#if defined(PIE_COMMITTED)
    int lane0;
#endif
    int channels;
    int conv_width;
    int dilation;
#if defined(PIE_REPLAY)
    int lane0;
    uint carry;
    uint page_tokens;
    uint page_words;
    uint plane_words;
    uint row_words;
#endif
#if defined(PIE_SNAPS)
    int snap_lane0;
#endif
};
PIE_PARAMS(PARAMS_AT, Params, params);
#if defined(PIE_REPLAY)
PIE_POOL_READ(uint, params.seg_words)
PIE_POOL_WRITE(uint, params.seg_words)
#else
PIE_POOL_READ(float, params.seg_words)
PIE_POOL_WRITE(float, params.seg_words)
PIE_POOL_VIEW(state, float, params.state_origin)
#endif
PIE_STATE_PIECES
#if defined(PIE_SNAPS)
PIE_STATE_SNAPSHOTS
#endif

// Where channel `c`'s history lies in the record of `slots` entry `at`: a
// piece is `[hist][min(piece_atoms, chans)]`, the channel axis innermost.
struct Column {
    uint slab;
    uint pitch;
    uint col;
};

Column column(uint at, uint chans, uint c) {
    Column out;
    out.slab = pie_piece_unit(at, c) * params.slot_stride;
    out.pitch = min(params.piece_atoms, chans);
    out.col = c % params.piece_atoms;
    return out;
}

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

// A history element as the pool holds it: a float, or its bits where the
// pool is read as words (`PIE_REPLAY`).
#if defined(PIE_REPLAY)
#define POOL_F(w_) asfloat(w_)
#define POOL_W(f_) asuint(f_)
#else
#define POOL_F(w_) (w_)
#define POOL_W(f_) (f_)
#endif

// The slot's `hist` rows of channels `c`, `c + 1` (column `at.col`).
// A slot's history lies in the slot, so in one segment: found once.
void past_load(Column at, int hist, out float2 past[HIST_MAX]) {
    uint seg, first;
    pie_pool_find(params.state_origin + at.slab, seg, first);
#define CONV_PAST_LOAD(buf_) \
    for (int s = 0; s < hist; s++) { \
        uint i = first + uint(s) * at.pitch + at.col; \
        past[s] = float2(POOL_F(buf_[i]), POOL_F(buf_[i + 1u])); \
    }
    PIE_POOL_BRANCH(seg, CONV_PAST_LOAD)
}

void past_put(Column at, int s, float2 v) {
    uint seg, first;
    pie_pool_find(params.state_origin + at.slab, seg, first);
    uint i = first + uint(s) * at.pitch + at.col;
#define CONV_PAST_PUT(buf_) \
    buf_[i] = POOL_W(v.x); \
    buf_[i + 1u] = POOL_W(v.y);
    PIE_POOL_BRANCH(seg, CONV_PAST_PUT)
}

#if defined(PIE_REPLAY)
// The pool word holding channels `c`, `c + 1` of rs-buffer row `unit`.
uint rs_word(int unit, uint c) {
    uint u = uint(unit);
    uint page = u / params.page_tokens;
    uint token = u % params.page_tokens;
    return page * params.page_words + params.plane_words + token * params.row_words + (c >> 1);
}
#endif

#if defined(PIE_COMMITTED) || defined(PIE_CHUNKED)
// Rows `begin .. begin + span` convolved over the slot's history, which then
// rolls to the history as of row `keep` (none when `keep <= 0`); snapshot
// lane `snap`'s records (none when negative) take the history as of their
// rows.
void walk(Column slab, int begin, int span, int keep, uint chans, uint c, int snap) {
    int width = params.conv_width;
    int dil = params.dilation;
    int hist = (width - 1) * dil + 1;
    uint tap0 = c * uint(width);
    uint tap1 = tap0 + uint(width);
    float2 past[HIST_MAX];
    past_load(slab, hist, past);
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
#if defined(PIE_SNAPS)
    for (uint n = 0u; n < PIE_RS_SNAPSHOTS; n++) {
        int at = pie_snap_row(snap, n);
        if (at <= 0 || at > span) {
            break;
        }
        Column into;
        into.slab = pie_snap_unit(uint(snap), n, c) * params.slot_stride;
        into.pitch = slab.pitch;
        into.col = slab.col;
        for (int s = 0; s < hist; s++) {
            int src = at - hist + s;
            past_put(into, s, src < 0 ? past[hist + src] : x_at(uint(begin + src), chans, c));
        }
    }
#endif
    if (keep <= 0) {
        return;
    }
    for (int s = 0; s < hist; s++) {
        int src = keep - hist + s;
        past_put(slab, s, src < 0 ? past[hist + src] : x_at(uint(begin + src), chans, c));
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

#if defined(PIE_REPLAY)
    uint lane = uint(params.lane0) + r;
    uint entry = lane * (3u + params.carry);
    int carries = lag[entry];
    int keep = lag[entry + 1u];
    int own = lag[entry + 2u];
    uint word = x[(r * chans + c) >> 1];
    float2 fresh = float2(bf16_to_f32(word & 0xffffu), bf16_to_f32(word >> 16));
    if (own >= 0) {
        pie_pool_set(rs_word(own, c), word);
    }
    Column slab = column(r, chans, c);
    float2 past[HIST_MAX];
    past_load(slab, hist, past);
    uint tap0 = c * uint(width);
    uint tap1 = tap0 + uint(width);
    float2 last = float2(bf16_at(weight, tap0 + uint(width - 1)), bf16_at(weight, tap1 + uint(width - 1)));
    for (int s = 0; s <= carries; s++) {
        float2 v = fresh;
        if (s < carries) {
            uint held = pie_pool_get(rs_word(lag[entry + 3u + uint(s)], c));
            v = float2(bf16_to_f32(held & 0xffffu), bf16_to_f32(held >> 16));
        }
        float2 acc = float2(0.0);
        for (int k = 0; k + 1 < width; k++) {
            float2 w = float2(bf16_at(weight, tap0 + uint(k)), bf16_at(weight, tap1 + uint(k)));
            acc += past[k * params.dilation + 1] * w;
        }
        acc += v * last;
        if (s < carries) {
            carried[((lane * params.carry + uint(s)) * chans + c) >> 1] = pack_bf16(silu(acc.x), silu(acc.y));
        } else {
            y_put(r, chans, c, acc);
        }
        for (int i = 0; i + 1 < hist; i++) {
            past[i] = past[i + 1];
        }
        past[hist - 1] = v;
        if (s + 1 == keep) {
            for (int i = 0; i < hist; i++) {
                past_put(slab, i, past[i]);
            }
        }
    }
#elif defined(PIE_COMMITTED)
    uint lane = uint(params.lane0) + r;
    int begin = indptr[r];
    for (uint j = 0; j < r; j++) {
        begin += replay[uint(params.lane0) + j];
    }
    int span = (indptr[r + 1] - indptr[r]) + replay[lane];
    if (span <= 0 || pie_state_none(lane)) {
        return;
    }
#if defined(PIE_SNAPS)
    int snap = pie_snap_lane(r);
#else
    int snap = -1;
#endif
    walk(column(lane, chans, c), begin, span, min(commit[lane], span), chans, c, snap);
#elif defined(PIE_CHUNKED)
    int begin = indptr[r];
    int span = indptr[r + 1] - begin;
    if (span <= 0) {
        return;
    }
#if defined(PIE_SNAPS)
    int snap = pie_snap_lane(r);
#else
    int snap = -1;
#endif
    walk(column(uint(begin), chans, c), begin, span, span, chans, c, snap);
#else
    Column slab = column(r, chans, c);
    float2 past[HIST_MAX];
    past_load(slab, hist, past);
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
        past_put(slab, s, past[s + 1]);
    }
    past_put(slab, hist - 1, fresh);
#endif
}

// pie:instantiate ssm_conv1d_bf16 PIE_GROUP_X=128
// pie:instantiate ssm_conv1d_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1
// pie:instantiate ssm_conv1d_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1
// pie:instantiate ssm_conv1d_chunked_snap_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_SNAPS=1
// pie:instantiate ssm_conv1d_replay_bf16 PIE_GROUP_X=128 PIE_REPLAY=1
// pie:instantiate ssm_conv1d_committed_snap_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_SNAPS=1
