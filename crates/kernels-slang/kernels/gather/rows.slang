#include "prim/bf16.slang"
#include "prim/params.slang"

// Row movement, in 32-bit words: a bf16 row moves two values a word, an f32
// or i32 row one, so every mover but the pool is blind to the element and
// the host states widths in words. A word a lane, a row a grid row (`y`).
// `PIE_<KERNEL>` picks the kernel:
//
// - `PIE_COPY`: a flat run of words.
// - `PIE_GATHER`: `out[i] = in[rows[i]]`; `PIE_SCATTER`: `out[rows[i]] =
//   in[i]`; `PIE_SCATTER_LIVE`: the same, skipping a negative row.
// - `PIE_POOL`: bf16 rows averaged in blocks of `block` (a word a lane, two
//   values).
// - `PIE_SELECT`: a `width`-word slice at `offset` of each `stride`-word row.
// - `PIE_PLACE`: a `width`-word row landed at `offset` of each `stride`-word row.
// - `PIE_SPLIT`: each row cut into a left and a right part; `PIE_SPLIT_QKV`:
//   into q, k and v; `PIE_SPLIT_Q_GATE`: each `2 * head`-word head into its
//   query half and its gate half.

#if defined(PIE_COPY)

[[vk::binding(0)]] StructuredBuffer<uint> src;
[[vk::binding(1)]] RWStructuredBuffer<uint> dst;

struct Params {
    int words;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x < uint(max(params.words, 0))) {
        dst[gid.x] = src[gid.x];
    }
}

#elif defined(PIE_GATHER) || defined(PIE_SCATTER) || defined(PIE_SCATTER_LIVE)

[[vk::binding(0)]] StructuredBuffer<uint> in_;
[[vk::binding(1)]] RWStructuredBuffer<uint> out_;
[[vk::binding(2)]] StructuredBuffer<int> rows;

struct Params {
    uint pitch;
    uint count;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint i = gid.y;
    uint pitch = params.pitch;
    if (c >= pitch || i >= params.count) {
        return;
    }
#if defined(PIE_GATHER)
    out_[i * pitch + c] = in_[uint(rows[i]) * pitch + c];
#else
    int at = rows[i];
#if defined(PIE_SCATTER_LIVE)
    if (at < 0) {
        return;
    }
#endif
    out_[uint(at) * pitch + c] = in_[i * pitch + c];
#endif
}

#elif defined(PIE_POOL)

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] RWStructuredBuffer<uint> y;

struct Params {
    uint words;
    uint block;
    uint rows;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint o = gid.y;
    if (c >= params.words || o >= params.rows) {
        return;
    }
    float2 acc = float2(0.0, 0.0);
    for (uint r = 0u; r < params.block; r++) {
        uint word = x[(o * params.block + r) * params.words + c];
        acc += float2(bf16_to_f32(word & 0xffffu), bf16_to_f32(word >> 16));
    }
    acc /= float(params.block);
    y[o * params.words + c] = pack_bf16(acc.x, acc.y);
}

#elif defined(PIE_SELECT)

[[vk::binding(0)]] StructuredBuffer<uint> table;
[[vk::binding(1)]] RWStructuredBuffer<uint> y;

struct Params {
    uint stride;
    uint offset;
    uint width;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    if (c < params.width) {
        y[gid.y * params.width + c] = table[gid.y * params.stride + params.offset + c];
    }
}

#elif defined(PIE_PLACE)

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] RWStructuredBuffer<uint> y;

struct Params {
    uint stride;
    uint offset;
    uint width;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    if (c < params.width) {
        y[gid.y * params.stride + params.offset + c] = x[gid.y * params.width + c];
    }
}

#elif defined(PIE_SPLIT)

[[vk::binding(0)]] StructuredBuffer<uint> src;
[[vk::binding(1)]] RWStructuredBuffer<uint> left;
[[vk::binding(2)]] RWStructuredBuffer<uint> right;

struct Params {
    uint left_width;
    uint right_width;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint row = gid.y;
    uint lw = params.left_width;
    uint rw = params.right_width;
    if (c >= lw + rw) {
        return;
    }
    uint v = src[row * (lw + rw) + c];
    if (c < lw) {
        left[row * lw + c] = v;
    } else {
        right[row * rw + c - lw] = v;
    }
}

#elif defined(PIE_SPLIT_QKV)

[[vk::binding(0)]] StructuredBuffer<uint> packed;
[[vk::binding(1)]] RWStructuredBuffer<uint> q;
[[vk::binding(2)]] RWStructuredBuffer<uint> k;
[[vk::binding(3)]] RWStructuredBuffer<uint> v;

struct Params {
    uint q_width;
    uint kv_width;
};
PIE_PARAMS(4, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint qw = params.q_width;
    uint kvw = params.kv_width;
    uint pw = qw + 2u * kvw;
    uint c = gid.x;
    uint row = gid.y;
    if (c >= pw) {
        return;
    }
    uint value = packed[row * pw + c];
    if (c < qw) {
        q[row * qw + c] = value;
    } else if (c < qw + kvw) {
        k[row * kvw + c - qw] = value;
    } else {
        v[row * kvw + c - qw - kvw] = value;
    }
}

#elif defined(PIE_SPLIT_Q_GATE)

[[vk::binding(0)]] StructuredBuffer<uint> qg;
[[vk::binding(1)]] RWStructuredBuffer<uint> q;
[[vk::binding(2)]] RWStructuredBuffer<uint> gate;

struct Params {
    uint head;
    uint qg_width;
    uint q_width;
};
PIE_PARAMS(3, Params, params);

// The grid is (head words, heads, rows).
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint hd = params.head;
    uint i = gid.x;
    uint h = gid.y;
    if (i >= hd) {
        return;
    }
    uint out_at = gid.z * params.q_width + h * hd + i;
    uint in_at = gid.z * params.qg_width + 2u * h * hd + i;
    q[out_at] = qg[in_at];
    gate[out_at] = qg[in_at + hd];
}

#else
#error "gather/rows.slang: no kernel selected"
#endif

// pie:instantiate gather_rows_copy_words PIE_COPY=1 PIE_GROUP_X=256
// pie:instantiate gather_rows_words PIE_GATHER=1 PIE_GROUP_X=256
// pie:instantiate gather_rows_scatter_words PIE_SCATTER=1 PIE_GROUP_X=256
// pie:instantiate gather_rows_scatter_live_words PIE_SCATTER_LIVE=1 PIE_GROUP_X=256
// pie:instantiate gather_rows_pool_bf16 PIE_POOL=1 PIE_GROUP_X=256
// pie:instantiate gather_rows_select_words PIE_SELECT=1 PIE_GROUP_X=256
// pie:instantiate gather_rows_place_words PIE_PLACE=1 PIE_GROUP_X=256
// pie:instantiate gather_rows_split_words PIE_SPLIT=1 PIE_GROUP_X=256
// pie:instantiate gather_rows_split_qkv_words PIE_SPLIT_QKV=1 PIE_GROUP_X=256
// pie:instantiate gather_rows_split_q_gate_words PIE_SPLIT_Q_GATE=1 PIE_GROUP_X=256
