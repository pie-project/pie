#include "common/bf16.slang"

// PTQ1_0 ternary decode-in-dot (Dtype::Ptq1_0, spelled `g128_t3_f16_n`).
//
// A block is 28 bytes for 128 weights (1.75 bpw): `qs[24]` (5 base-3 trits per
// byte) + `qh[2]` (4 trits per byte) + a trailing fp16 scale `d`. A weight is
// `value = (trit - 1) * d`, trit in {0,1,2} -> {-1,0,+1}. The scale is INLINE
// per block; there is NO separate scales plane. row_bytes = ceil(K/128) * 28.
//
// This is a verbatim translation of the Metal kernel
// `kernels-metal/kernels/linear/quant_ptq1_0.metal`, which itself mirrors the
// host oracle `checkpoint::codec::ptq1_0::decode_block` byte for byte. The
// subtle part is the *staging*: for a 24-byte `qs` the fork walks chunks
// {16,8} (the 32-chunk never fits), emitting elements in NATURAL order 0..127;
// the `qh` tail carries elements 120..127. We decode a block the same way,
// inside the dot, so `y[row] = sum_i (trit_i - 1) * d * x[block*128 + i]`.
//
// We form each decoded weight as `float(trit - 1) * d` and then multiply by the
// activation, exactly as the host forms `f32::from(trit-1) * d`. With a one-hot
// activation only one term survives, and the read-back (the f32-out variant) is
// bit-exact against the host decoder: an integer in {-1,0,1} times a float is
// exact, and half->float is exact for every finite scale.

#define PTQ1_0_BLOCK 128u
#define PTQ1_0_BLOCK_BYTES 28u
#define PTQ1_0_QS_LEN 24u

// A workgroup owns `PIE_RLANES` output rows for one activation vector; its 32
// lanes split that row's K-blocks and a groupshared tree reduces them (the
// same groupshared-reduction shape as `gemm/dense.slang`, which runs correctly
// on lavapipe — no subgroup/Wave ops).
#define PIE_KLANES 32u
#define PIE_RLANES 8u

[[vk::binding(0)]] StructuredBuffer<uint8_t> w;
PIE_BUFFER_RO(1, x);
#if defined(PIE_OUT_F32)
[[vk::binding(2)]] RWStructuredBuffer<float> y;
#else
PIE_BUFFER_RW(2, y);
#endif

struct Push {
    int out_vec_size;
    int in_vec_size;
    int vecs;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float partials[PIE_RLANES * PIE_KLANES];

// fp16 -> f32, bit-exact against the host `half::f16::to_f32`
// (`GGML_FP16_TO_FP32`). Shares the integer decode of `quant/kquant.slang`'s
// `gguf_f16`; scales are always finite, so this is exact.
float ptq1_0_f16(uint h) {
    uint s = (h & 0x8000u) << 16u;
    uint e = (h >> 10u) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u) {
        float v = float(m) * 5.9604645e-8;
        return (s != 0u) ? -v : v;
    }
    if (e == 31u) {
        return asfloat(s | 0x7f800000u | (m << 13u));
    }
    return asfloat(s | ((e + 112u) << 23u) | (m << 13u));
}

// One base-3 trit from a packed byte at place-weight `pw` (3^n), matching the
// host's `trit()` arithmetic exactly: `q = (byte * 3^n) & 0xff; (q * 3) >> 8`.
int ptq1_0_trit(uint byte, int pw) {
    int q = (int(byte) * pw) & 0xff;
    return (q * 3) >> 8;
}

// Dot one 28-byte block (`blk` is its byte offset in `w`) against the 128
// activations at element offset `xblk`, decoding weights in the fork's
// {16,8}+qh staging (natural element order).
float ptq1_0_block_dot(uint blk, uint xblk) {
    uint dlo = uint(w[blk + 26u]);
    uint dhi = uint(w[blk + 27u]);
    float d = ptq1_0_f16(dlo | (dhi << 8u));

    float acc = 0.0;
    uint idx = 0u;

    // qs chunk A: bytes 0..15, elements 0..79 (trit-slot `n` outer, byte inner).
    int pw = 1;
    for (uint n = 0u; n < 5u; n++) {
        for (uint mm = 0u; mm < 16u; mm++) {
            int t = ptq1_0_trit(uint(w[blk + mm]), pw);
            acc += float(t - 1) * d * PIE_LOAD(x[xblk + idx]);
            idx++;
        }
        pw *= 3;
    }

    // qs chunk B: bytes 16..23, elements 80..119.
    pw = 1;
    for (uint n = 0u; n < 5u; n++) {
        for (uint mm = 0u; mm < 8u; mm++) {
            int t = ptq1_0_trit(uint(w[blk + 16u + mm]), pw);
            acc += float(t - 1) * d * PIE_LOAD(x[xblk + idx]);
            idx++;
        }
        pw *= 3;
    }

    // qh tail: bytes 24..25, elements 120..127, 4 trits per byte.
    pw = 1;
    for (uint n = 0u; n < 4u; n++) {
        for (uint mm = 0u; mm < 2u; mm++) {
            int t = ptq1_0_trit(uint(w[blk + PTQ1_0_QS_LEN + mm]), pw);
            acc += float(t - 1) * d * PIE_LOAD(x[xblk + idx]);
            idx++;
        }
        pw *= 3;
    }

    return acc;
}

[numthreads(32, 8, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint klane = local.x;
    uint rlane = local.y;
    uint row = group.x * PIE_RLANES + rlane;
    uint vec = group.y;

    uint k = uint(pc.in_vec_size);
    uint n = uint(pc.out_vec_size);
    uint num_blocks = k / PTQ1_0_BLOCK;
    uint row_bytes = num_blocks * PTQ1_0_BLOCK_BYTES;

    bool live = row < n && vec < uint(pc.vecs);
    float acc = 0.0;
    if (live) {
        uint xrow = vec * k;
        for (uint bk = klane; bk < num_blocks; bk += PIE_KLANES) {
            uint blk = row * row_bytes + bk * PTQ1_0_BLOCK_BYTES;
            uint xblk = xrow + bk * PTQ1_0_BLOCK;
            acc += ptq1_0_block_dot(blk, xblk);
        }
    }

    uint slot = rlane * PIE_KLANES + klane;
    partials[slot] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = PIE_KLANES / 2u; step > 0u; step >>= 1u) {
        if (klane < step) {
            partials[slot] += partials[slot + step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (klane == 0u && live) {
#if defined(PIE_OUT_F32)
        y[vec * n + row] = partials[rlane * PIE_KLANES];
#else
        y[vec * n + row] = PIE_STORE(partials[rlane * PIE_KLANES]);
#endif
    }
}

// Production path: bf16 activation in, bf16 out.
// pie:instantiate ptq1_0_qmv_bfloat16
// Bit-exact read-back path: bf16 activation in, f32 out (the oracle test).
// pie:instantiate ptq1_0_qmv_bfloat16_f32 PIE_OUT_F32=1
