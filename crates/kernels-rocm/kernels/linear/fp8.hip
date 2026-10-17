#pragma once

// FP8 (W8A8) products on gfx942's FP8 matrix core: `e4m3tile128` weights
// (e4m3 codes under one f32 factor a 128x128 block, transformers' FP8
// release) against activations quantized to e4m3 codes under an f32 scale a
// 1x128 block (transformers' and vLLM's dynamic recipe), as dense
// projections and as expert banks' routed selects.
//
// The matrix core reads e4m3fnuz, not the OCP e4m3fn the checkpoints and
// the recipe use. The same bits hold half the value in fnuz (its bias is 8,
// not 7, subnormals too), except two: OCP's negative zero (0x80) is fnuz's
// NaN, and OCP's NaNs (0x7f, 0xff) are finite in fnuz. A weight's codes are
// put in the matrix core's order and made canonical once at load
// (`e4m3fn_shuffle`: negative zero as zero, the value it is, and a NaN code
// refused), the activation's are quantized so, and the matrix core reads
// both as they are: every product is a
// quarter of the OCP one, exactly, and the factor of four folds into each
// block's scale with the weight's factor and the activation's.
//
// Each 128-wide block's products sum on the matrix core in f32, then fold
// under s_w·s_a·4 into the row's sum: the W8A8 GEMM's sum, in its order up
// to f32 rounding.
//
// A tile: one item of up to MT activation rows (the routes of one expert,
// or rows of a dense projection) against 64 weight rows, four waves of 16.
// A stage is one 128-code block of K: the weight goes memory → registers
// (DEPTH stages ahead, whole-row reads of 128 bytes on eight lanes) → the
// wave's own LDS → its A operand; the item's activation codes and scales
// memory → registers → the block's LDS (double-buffered, one barrier a
// stage) → each wave's B operand. A lane's D cells are four neighbouring
// weight rows of one activation row, stored as one 8-byte write (or 16
// bytes of f32 partial for a slice of K).

#include "linear/moe_kit.hip"

namespace pie::linear {

typedef float fp8_f4 __attribute__((ext_vector_type(4)));

// The matrix core's order for a weight (n x k e4m3 codes, n a multiple of 16,
// k of 64), which its products read straight into the A operand: each strip
// of 16 rows is K / 64 units of 1 KB, unit u holding codes [64u, +64) of the
// strip's rows as 64 lanes of 16 bytes, lane l row l % 16's codes
// 8·(l / 16) + [0, 8) of the unit's first 32-code step, then of its second.
// A wave reads a strip's stage in whole 1 KB runs, and lane l holds what
// the matrix core's lane l multiplies.
__device__ __forceinline__ long long fp8_unit_at(int row, int col, int k) {
    const int strip = row / 16, r16 = row % 16, unit = col / 64, cu = col % 64;
    const int half = cu / 32, g = (cu % 32) / 8, e = cu % 8;
    return static_cast<long long>(strip) * 16 * k + unit * 1024 + (g * 16 + r16) * 16 + half * 8 + e;
}

// `src` (`rows` x k OCP e4m3fn codes, row-major) into `dst` in the matrix
// core's order (`fp8_unit_at`), made canonical as `e4m3fn_canonical` makes
// them (NaN codes counted into `nans`). A thread a 16-byte lane chunk.
__global__ __launch_bounds__(256) void e4m3fn_shuffle(const u8* __restrict__ src, u8* __restrict__ dst, int rows,
                                                      int k, unsigned int* __restrict__ nans) {
    const long long chunks = static_cast<long long>(rows) * k / 16;
    unsigned int found = 0;
    for (long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; q < chunks;
         q += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long strip = q / k;  // a strip is 16 rows x k codes, k chunks
        const int within = static_cast<int>(q % k);
        const int unit = within / 64, l = within % 64;
        const int row = static_cast<int>(strip) * 16 + l % 16, col = unit * 64 + 8 * (l / 16);
        u32 w[4];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint2 v = *reinterpret_cast<const uint2*>(src + static_cast<long long>(row) * k + col + 32 * half);
            const u32 x[2] = {v.x, v.y};
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const u32 mag = x[j] & 0x7F7F7F7Fu;
                found += __popc((mag + 0x01010101u) & 0x80808080u);
                w[2 * half + j] = x[j] & (((mag + 0x7F7F7F7Fu) & 0x80808080u) | 0x7F7F7F7Fu);
            }
        }
        *reinterpret_cast<uint4*>(dst + q * 16) = uint4{w[0], w[1], w[2], w[3]};
    }
    if (found != 0u) atomicAdd(nans, found);
}

__device__ __forceinline__ fp8_f4 mma_fp8(u32 a0, u32 a1, u32 b0, u32 b1, fp8_f4 c) {
    const long a = static_cast<long>((static_cast<unsigned long long>(a1) << 32) | a0);
    const long b = static_cast<long>((static_cast<unsigned long long>(b1) << 32) | b0);
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, 0, 0);
}

// A wave's 4-byte reads from `base + voff` landing as 256 contiguous bytes
// of LDS at `lds` (a wave-uniform LDS address), hand-issued: the compiler
// would wait on it before every read of LDS, and the tile waits it itself.
__device__ __forceinline__ void lds_dma4(const void* base, u32 voff, u32 lds) {
    asm volatile("s_mov_b32 m0, %2\n\tglobal_load_lds_dword %0, %1" ::"v"(voff), "s"(base), "s"(lds) : "memory", "m0");
}
__device__ __forceinline__ u32 lds_address(const void* p) {
    return __builtin_amdgcn_readfirstlane(static_cast<u32>(reinterpret_cast<size_t>(p)));
}

// The bits of e4m3fn value nearest `q` (|q| ≤ 448), ties to even: steps
// of 2^-9 below 2^-6, else three mantissa bits under the binade; zero is
// 0x00 whatever its sign, as the matrix core reads it.
__device__ __forceinline__ u32 e4m3fn_bits(float q) {
    const float a = fabsf(q);
    u32 bits;
    if (a < 0.015625f) {
        bits = static_cast<u32>(rintf(a * 512.f));
    } else {
        const int e = static_cast<int>((__float_as_uint(a) >> 23) & 0xFFu) - 127;
        const float step = __uint_as_float(static_cast<u32>(e - 3 + 127) << 23);
        const u32 mant = static_cast<u32>(rintf(a / step));  // 8 .. 16
        bits = mant == 16u ? static_cast<u32>(e + 8) << 3 : (static_cast<u32>(e + 7) << 3) | (mant - 8u);
    }
    return bits != 0u && q < 0.f ? bits | 0x80u : bits;
}

// The recipe's scale of a block whose amax is `amax`.
__device__ __forceinline__ float e4m3_block_scale(float amax) { return fmaxf(amax, 1e-10f) / 448.f; }

// The power of two e8m0 exponent `e` stands for: 2^(e − 127), 2^-127 (an
// f32 denormal) at e = 0.
__device__ __forceinline__ float fp8_e8m0(u32 e) { return __uint_as_float(e == 0u ? 0x400000u : e << 23); }

// y[route] = W·act[route / act_div] over routes order[begin .. begin +
// count) (rows begin .. begin + count without an order), count ≤ MT, for
// the weight rows [tile·64, +64) of `codes` / `factors` (n x k, k a multiple
// of 128); `act` the rows' e4m3 codes ([rows, k], `act_quant_e4m3_codes`)
// beside their scales ([rows, k / 128]). Stages [s0, s1) of K; with `part`
// the f32 partial lands there (rows of n by the item's own order) for a
// later sum, else y rounds once.
// SB 32: an `fp8b32` weight (ue8m0 exponents a 32x32 block, `factors` their
// bytes) against rows under ue8m0 exponents a 1x32 block (`act_scales`
// their bytes, `act_quant_e4m3_e8m0`): the same bytes a stage as the f32
// forms, each 32-code step's products folded under its own two exponents.
template <int MT, int DEPTH, int NB = 1, int SB = 0>
__device__ __forceinline__ void fp8_tile(const u8* __restrict__ act, const float* __restrict__ act_scales,
                                         const i32* __restrict__ order, int begin, int count,
                                         const u8* __restrict__ codes, const float* __restrict__ factors,
                                         bf16* __restrict__ out, float* __restrict__ part, int act_div, int n, int k,
                                         int tile, int s0, int s1) {
    static_assert(MT == 16 || MT == 32 || MT == 64, "an item of 16, 32 or 64 rows");
    constexpr int KC = 128, PITCH = KC + 16, WAVES = kGroupThreads / kWave, MB = MT / 16;
    constexpr int R = DEPTH + 1;
    constexpr int PA = MT * KC / 16;                               // 16-byte code pieces a stage, eight a row
    constexpr int NA = (PA + kGroupThreads - 1) / kGroupThreads;  // reads a thread
    constexpr int NL = 2 * NB;  // weight reads a lane a stage: two units of each of its NB strips
    constexpr int NAL = NA + 1, NWL = NL + 1;  // with the activation's scale, the weight's factor
    static_assert(PA % kGroupThreads == 0 || PA < kGroupThreads, "the activation stage splits evenly");
    static_assert(DEPTH >= 2 && NWL + (DEPTH - 1) * (NAL + NWL) < 64, "the reads in flight fit the count");
    __shared__ __align__(16) u8 xs[2][MT * PITCH];
    __shared__ float xsc[2][MT];

    const int tid = threadIdx.x, l = lane(), r16 = l % 16, g = l / 16;
    const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const int row0 = (tile * WAVES + wave) * 16 * NB;
    const int kbytes = k;  // a row's bytes
    const int kb = k / 128;  // scale blocks a row
    const int stages = min(s1, kbytes / KC) - s0;
    const int live_mb = __builtin_amdgcn_readfirstlane((count + 15) / 16);
    auto route = [&](int m) { return order != nullptr ? order[begin + m] : begin + m; };
    const moe_rsrc wr = batch_plane(codes, 1ull * n * kbytes);

    // The activation pieces a thread reads: row m's 16 codes `col` of the
    // stage; a row past the item reads the first and lands as zeros.
    const u8* src[NA];
    int col[NA], at[NA];
    bool live[NA], mine[NA];
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int piece = (tid + i * kGroupThreads) % PA;
        const int m = piece / (KC / 16);
        mine[i] = tid + i * kGroupThreads < PA;
        live[i] = m < count;
        src[i] = act + static_cast<long long>(route(live[i] ? m : 0) / act_div) * kbytes;
        col[i] = (piece % (KC / 16)) * 16;
        at[i] = m * PITCH + col[i];
    }
    const float* srow = act_scales + static_cast<long long>(route(min(tid % MT, max(count - 1, 0))) / act_div) * kb;
    // The weight in the matrix core's order (`fp8_unit_at`): the wave's NB
    // strips of 16 rows, a stage two 1 KB units of each, lane l's 16 bytes of
    // a unit its own A operand for two k-steps. A strip past the weight
    // reads the last.
    u32 wv[NB];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
        wv[nb] = static_cast<u32>(min(row0 + 16 * nb, n - 16) / 16) * static_cast<u32>(16 * kbytes) + 16 * l;
    // The wave's 16 rows lie in one 128-row block (one 32-row block under
    // SB 32): its factor a stage (four exponents).
    static_assert(SB == 0 || (SB == 32 && NB <= 2), "f32 128x128 factors, or ue8m0 exponents a 32x32 block");
    const float* frow = factors + static_cast<long long>(min(row0, n - 1) / (SB == 32 ? 32 : 128)) * kb;

    i32x4 w[R][NL];
    i32x4 a[R][NA];
    moe_i2 as[R], ws[R];
    fp8_f4 acc[NB][MB];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) acc[nb][mb] = fp8_f4{0.f, 0.f, 0.f, 0.f};

    // Stage s's reads: the activation and its scale, then the weight and its
    // factor; past the last the activation rereads the last and the weight
    // reads nothing.
    auto issue = [&](auto slot, int s) {
        constexpr int S = decltype(slot)::value;
        const int sc = s0 + min(s, stages - 1), sa = s0 + s;
        const bool on = s < stages;
#pragma unroll
        for (int i = 0; i < NA; ++i) gload16(a[S][i], src[i] + sc * KC + col[i]);
        gload_run<4>(as[S], srow + sc);
#pragma unroll
        for (int i = 0; i < NL; ++i) bload_nt(w[S][i], wr, on ? wv[i / 2] + (2 * sa + i % 2) * 1024 : kBatchVoid);
        gload_run<4>(ws[S], frow + sc);
    };
    auto store = [&](auto slot, int buf) {
        constexpr int S = decltype(slot)::value;
#pragma unroll
        for (int i = 0; i < NA; ++i)
            if (PA % kGroupThreads == 0 || mine[i])
                *reinterpret_cast<i32x4*>(xs[buf] + at[i]) = live[i] ? a[S][i] : i32x4{0, 0, 0, 0};
        if (tid < MT) xsc[buf][tid] = __int_as_float(as[S][0]);
    };
    // A = the weight (row r16, codes 32·kk + 8g of the stage), B = the
    // activation (row 16·mb + r16, the same codes): D cell (weight row 4g +
    // t, activation row 16·mb + r16).
    auto multiply = [&](auto slot, int buf) {
        constexpr int S = decltype(slot)::value;
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
            if (mb >= live_mb) continue;
            moe_u2 xb[4];
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
                xb[kk] = *reinterpret_cast<const moe_u2*>(xs[buf] + (16 * mb + r16) * PITCH + 32 * kk + 8 * g);
            if constexpr (SB == 32) {
                const u32 we = static_cast<u32>(ws[S][0]), ae = __float_as_uint(xsc[buf][16 * mb + r16]);
                float f[4];
#pragma unroll
                for (int kk = 0; kk < 4; ++kk)
                    f[kk] = 4.f * fp8_e8m0((we >> (8 * kk)) & 0xFFu) * fp8_e8m0((ae >> (8 * kk)) & 0xFFu);
#pragma unroll
                for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const i32x4& u = w[S][2 * nb + kk / 2];
                        const fp8_f4 p = mma_fp8(static_cast<u32>(u[2 * (kk % 2)]), static_cast<u32>(u[2 * (kk % 2) + 1]),
                                                 xb[kk][0], xb[kk][1], fp8_f4{0.f, 0.f, 0.f, 0.f});
                        // Two lanes' cells an instruction (v_pk_fma_f32).
                        acc[nb][mb] = __builtin_elementwise_fma(fp8_f4{f[kk], f[kk], f[kk], f[kk]}, p, acc[nb][mb]);
                    }
                continue;
            }
            const float f = 4.f * __int_as_float(ws[S][0]) * xsc[buf][16 * mb + r16];
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                fp8_f4 p = fp8_f4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const i32x4& u = w[S][2 * nb + kk / 2];
                    p = mma_fp8(static_cast<u32>(u[2 * (kk % 2)]), static_cast<u32>(u[2 * (kk % 2) + 1]), xb[kk][0],
                                xb[kk][1], p);
                }
#pragma unroll
                for (int t = 0; t < 4; ++t) acc[nb][mb][t] = fmaf(f, p[t], acc[nb][mb][t]);
            }
        }
    };

    Unroll<DEPTH, 0>::run([&](auto slot, int s) { issue(slot, s); }, 0);
    // Stage 0's activation: younger are its weight's reads and the DEPTH - 1
    // stages behind.
    vm_wait<NWL + (DEPTH - 1) * (NAL + NWL)>();
    tie(a[0]);
    batch_hold(as[0]);
    store(Const<0>{}, 0);
    lds_barrier();

    auto stage_at = [&](auto cur, int s) {
        constexpr int CU = decltype(cur)::value;
        constexpr int NEXT = (CU + 1) % R;
        if (s >= stages) return;
        // Stage s + 1's activation, and so (older) stage s's weight.
        vm_wait<NWL + (DEPTH - 2) * (NAL + NWL)>();
        tie(a[NEXT]);
        batch_hold(as[NEXT]);
        tie(w[CU]);
        batch_hold(ws[CU]);
        store(Const<NEXT>{}, (s + 1) & 1);
        multiply(Const<CU>{}, s & 1);
        issue(Const<(CU + DEPTH) % R>{}, s + DEPTH);
        lds_barrier();
    };
    for (int s = 0; s < stages; s += R) Unroll<R, 0>::run(stage_at, s);
    vm_wait<0>();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        tie(a[r]);
        tie(w[r]);
        batch_hold(as[r]);
        batch_hold(ws[r]);
    }

#pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
        const int m = 16 * mb + r16;
        if (mb >= live_mb || m >= count) continue;
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const int c = row0 + 16 * nb + 4 * g;
            if (part != nullptr) {
                float* q = part + static_cast<long long>(begin + m) * n;
                if (c + 3 < n && n % 4 == 0) {
                    *reinterpret_cast<fp8_f4*>(q + c) = acc[nb][mb];
                } else {
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        if (c + t < n) q[c + t] = acc[nb][mb][t];
                }
                continue;
            }
            bf16* o = out + static_cast<long long>(route(m)) * n;
            bf16 v[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) v[t] = from_f32<bf16>(acc[nb][mb][t]);
            if (c + 3 < n && n % 4 == 0) {
                *reinterpret_cast<moe_u2*>(o + c) = __builtin_bit_cast(moe_u2, v);
            } else {
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    if (c + t < n) o[c + t] = v[t];
            }
        }
    }
}

// The prefill tile on eight waves: 128 activation rows against 128 weight
// rows, waves of 64 weight rows x 32 activation rows, so a wave's sums and
// a stage's products fit beside two stages' reads in flight (set S holds a
// stage while it waits for the LDS, the other the stage after). Otherwise as
// `fp8_gemm_tile`.
// FP8 false: a bf16 projection or bank against bf16 rows on the bf16 matrix
// core, a stage 64 elements (the same 128 bytes a row), no scales to fold:
// the tile at the decode batches past 64 routes an expert that the batch
// legs read as many items.
template <bool FP8 = true>
__device__ __forceinline__ void fp8_gemm_tile8(const u8* __restrict__ act, const float* __restrict__ act_scales,
                                               const i32* __restrict__ order, int begin, int count,
                                               const u8* __restrict__ codes, const float* __restrict__ factors,
                                               bf16* __restrict__ out, int act_div, int n, int k, int tile, int s0,
                                               int s1, float* __restrict__ part) {
    constexpr int THREADS = 512, TM = 128, TN = 128, KC = 128, PITCH = KC + 16;  // KC: bytes a row a stage
    constexpr int NR = TN * KC / 16 / THREADS;  // 16-byte reads a thread a stage, each operand
    constexpr int PER = FP8 ? 2 * NR + 2 : 2 * NR;  // a stage's reads: codes, scale, codes, factor
    constexpr int EB = FP8 ? 1 : 2;                 // bytes an element
    __shared__ __align__(16) u8 wl[TN * PITCH];
    __shared__ __align__(16) u8 xl[TM * PITCH];
    __shared__ float xsc[TM + 1];  // the rows' scales, then the weight's factor
    const int tid = threadIdx.x, l = lane(), r16 = l % 16, g = l / 16;
    const int wave = tid / kWave, wm = wave / 2, wn = wave % 2;  // wm: 32 activation rows, wn: 64 weight rows
    const int row0 = tile * TN;
    const int kb = k * EB / KC;  // stages
    auto route = [&](int m) { return order != nullptr ? order[begin + m] : begin + m; };
    const moe_rsrc wr = batch_plane(codes, 1ull * EB * n * k);

    u32 wo[NR];
    const u8* xp[NR];
    bool xlive[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) {
        const int flat = THREADS * i + tid, r = flat / 8;
        if constexpr (FP8) {
            // The weight in the matrix core's order: chunk `flat` is lane
            // flat % 64 of unit (flat / 64) % 2 of strip flat / 128 (a strip
            // past the weight reads the last).
            const int strip = min(row0 / 16 + flat / 128, n / 16 - 1);
            wo[i] = static_cast<u32>(strip) * static_cast<u32>(16 * k) + (flat / 64 % 2) * 1024 + 16 * (flat % 64);
        } else {
            wo[i] = static_cast<u32>(min(row0 + r, n - 1)) * static_cast<u32>(EB * k) + 16 * (tid % 8);
        }
        xlive[i] = r < count;
        xp[i] = act + static_cast<long long>(route(xlive[i] ? r : 0) / act_div) * EB * k + 16 * (tid % 8);
    }
    const float* srow =
        FP8 ? act_scales + static_cast<long long>(route(min(tid % TM, max(count - 1, 0))) / act_div) * kb : nullptr;
    const float* frow = FP8 ? factors + static_cast<long long>(min(row0, n - 1) / 128) * kb : nullptr;

    i32x4 wv[2][NR], xv[2][NR];
    moe_i2 as[2], ws[2];
    const int last = min(s1, kb);
    auto issue = [&](auto set, int s) {
        constexpr int S = decltype(set)::value;
        const int sc = min(s, last - 1);
#pragma unroll
        for (int i = 0; i < NR; ++i) gload16(xv[S][i], xp[i] + sc * KC);
        if constexpr (FP8) gload_run<4>(as[S], srow + sc);
#pragma unroll
        for (int i = 0; i < NR; ++i) bload_nt(wv[S][i], wr, s < last ? wo[i] + s * (FP8 ? 2048 : KC) : kBatchVoid);
        if constexpr (FP8) gload_run<4>(ws[S], frow + sc);
    };
    auto store = [&](auto set) {
        constexpr int S = decltype(set)::value;
        tie(wv[S]);
        tie(xv[S]);
        batch_hold(as[S]);
        batch_hold(ws[S]);
#pragma unroll
        for (int i = 0; i < NR; ++i) {
            const int flat = THREADS * i + tid;
            if constexpr (FP8) {
                // Lane l's two 8-byte steps, back to their row's codes.
                const int l64 = flat % 64;
                u8* row = wl + (flat / 128 * 16 + l64 % 16) * PITCH + (flat / 64 % 2) * 64 + 8 * (l64 / 16);
                *reinterpret_cast<moe_u2*>(row) = moe_u2{static_cast<u32>(wv[S][i][0]), static_cast<u32>(wv[S][i][1])};
                *reinterpret_cast<moe_u2*>(row + 32) =
                    moe_u2{static_cast<u32>(wv[S][i][2]), static_cast<u32>(wv[S][i][3])};
            } else {
                *reinterpret_cast<i32x4*>(wl + flat / 8 * PITCH + 16 * (flat % 8)) = wv[S][i];
            }
            *reinterpret_cast<i32x4*>(xl + flat / 8 * PITCH + 16 * (flat % 8)) =
                xlive[i] ? xv[S][i] : i32x4{0, 0, 0, 0};
        }
        if constexpr (FP8) {
            if (tid < TM) xsc[tid] = __int_as_float(as[S][0]);
            if (tid == TM) xsc[TM] = __int_as_float(ws[S][0]);
        }
    };

    fp8_f4 acc[4][2];
#pragma unroll
    for (int nb = 0; nb < 4; ++nb)
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) acc[nb][mb] = fp8_f4{0.f, 0.f, 0.f, 0.f};

    auto compute = [&]() {
        if constexpr (!FP8) {
            // bf16: four elements a lane a 16-wide k-step, the same bytes.
#pragma unroll
            for (int kk = 0; kk < KC / 32; ++kk) {
                moe_u2 a[4], b[2];
#pragma unroll
                for (int nb = 0; nb < 4; ++nb)
                    a[nb] = *reinterpret_cast<const moe_u2*>(wl + (wn * 64 + nb * 16 + r16) * PITCH + 32 * kk + 8 * g);
#pragma unroll
                for (int mb = 0; mb < 2; ++mb)
                    b[mb] = *reinterpret_cast<const moe_u2*>(xl + (wm * 32 + mb * 16 + r16) * PITCH + 32 * kk + 8 * g);
#pragma unroll
                for (int nb = 0; nb < 4; ++nb)
#pragma unroll
                    for (int mb = 0; mb < 2; ++mb)
                        acc[nb][mb] = mma16x16<bf16>(a[nb][0], a[nb][1], b[mb][0], b[mb][1], acc[nb][mb]);
            }
            return;
        }
        const float wf = 4.f * xsc[TM];
        fp8_f4 p[4][2];
#pragma unroll
        for (int nb = 0; nb < 4; ++nb)
#pragma unroll
            for (int mb = 0; mb < 2; ++mb) p[nb][mb] = fp8_f4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < KC / 32; ++kk) {
            moe_u2 a[4], b[2];
#pragma unroll
            for (int nb = 0; nb < 4; ++nb)
                a[nb] = *reinterpret_cast<const moe_u2*>(wl + (wn * 64 + nb * 16 + r16) * PITCH + 32 * kk + 8 * g);
#pragma unroll
            for (int mb = 0; mb < 2; ++mb)
                b[mb] = *reinterpret_cast<const moe_u2*>(xl + (wm * 32 + mb * 16 + r16) * PITCH + 32 * kk + 8 * g);
#pragma unroll
            for (int nb = 0; nb < 4; ++nb)
#pragma unroll
                for (int mb = 0; mb < 2; ++mb) p[nb][mb] = mma_fp8(a[nb][0], a[nb][1], b[mb][0], b[mb][1], p[nb][mb]);
        }
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
            const float f = wf * xsc[wm * 32 + mb * 16 + r16];
#pragma unroll
            for (int nb = 0; nb < 4; ++nb)
#pragma unroll
                for (int t = 0; t < 4; ++t) acc[nb][mb][t] = fmaf(f, p[nb][mb][t], acc[nb][mb][t]);
        }
    };

    issue(Const<0>{}, s0);
    issue(Const<1>{}, s0 + 1);
    vm_wait<PER>();
    store(Const<0>{});
    lds_barrier();
    auto stage_at = [&](auto cur, int s) {
        constexpr int S = decltype(cur)::value;
        if (s >= last) return;
        issue(cur, s + 2);
        compute();
        lds_barrier();
        // Stage s + 1's reads; younger are stage s + 2's.
        vm_wait<PER>();
        store(Const<1 - S>{});
        lds_barrier();
    };
    for (int s = s0; s < last; s += 2) Unroll<2, 0>::run(stage_at, s);
    vm_wait<0>();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        tie(wv[r]);
        tie(xv[r]);
        batch_hold(as[r]);
        batch_hold(ws[r]);
    }

#pragma unroll
    for (int mb = 0; mb < 2; ++mb) {
        const int m = wm * 32 + mb * 16 + r16;
        if (m >= count) continue;
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const int c = row0 + wn * 64 + nb * 16 + 4 * g;
            if (part != nullptr) {
                float* q = part + static_cast<long long>(route(m)) * n;
                if (c + 3 < n && n % 4 == 0) {
                    *reinterpret_cast<fp8_f4*>(q + c) = acc[nb][mb];
                } else {
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        if (c + t < n) q[c + t] = acc[nb][mb][t];
                }
                continue;
            }
            bf16* o = out + static_cast<long long>(route(m)) * n;
            bf16 v[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) v[t] = from_f32<bf16>(acc[nb][mb][t]);
            if (c + 3 < n && n % 4 == 0) {
                *reinterpret_cast<moe_u2*>(o + c) = __builtin_bit_cast(moe_u2, v);
            } else {
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    if (c + t < n) o[c + t] = v[t];
            }
        }
    }
}

// 16 bytes of a partial one block lands and another sums, past the XCD's
// L2 (sc1, the aux bit 16: coherent at agent scope across the XCDs), as
// buffer stores and loads the compiler counts.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t shared_plane(const float* base, int bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<float*>(base), 0, bytes, 0x00020000);
}
__device__ __forceinline__ void store_shared16(__amdgpu_buffer_rsrc_t r, u32 off, fp8_f4 v) {
    __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(i32x4, v), r, off, 0, 16);
}
__device__ __forceinline__ fp8_f4 load_shared16(__amdgpu_buffer_rsrc_t r, u32 off) {
    return __builtin_bit_cast(fp8_f4, __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 16));
}

// The FP8 prefill tile: 128 activation rows against 256 weight rows, eight
// waves of 64 weight rows x 64 activation rows (wn along the weight, wm
// along the activation). The weight, in the matrix core's order, goes
// straight to each wave's registers (its four strips, two 1 KB units a
// stage: the A operands as they land); the activation codes and their scales
// go through NB stages of LDS, read by the card straight into it, 16-byte
// chunks XOR-swizzled by row. A stage is 128 codes of K; each 128-wide block
// (SB 0: f32 scales a 1x128 block, factors a 128x128 one) or each 32-code
// step (SB 32: ue8m0 exponents a 1x32 and a 32x32 block) folds into the
// sums under its scales. Tile (tm, tn) of y = act · codesᵀ, rows [m) live.
template <int SB, int NB, int WN, int WM, int NS, int MS>
__device__ __forceinline__ void fp8_prefill_tile(const u8* __restrict__ act, const u8* __restrict__ act_scales,
                                                 const u8* __restrict__ codes, const u8* __restrict__ factors,
                                                 bf16* __restrict__ y, int m, int n, int k, int tm, int tn,
                                                 int s_begin, int s_end, float* __restrict__ slot) {
    constexpr int MR = WM * MS * 16, KS = 128;          // activation rows a tile, codes a stage
    constexpr int XI = MR * KS / 256 / (WN * WM);       // activation DMA reads a wave a stage
    constexpr int FW = SB == 32 ? (NS + 1) / 2 : 1;     // factor words a wave a stage
    constexpr int PER = XI + 1 + 2 * NS + FW;           // reads a lane a stage
    constexpr int D = NB - 1;
    static_assert(XI * 256 * WN * WM == MR * KS, "a stage is whole wave reads");
    static_assert((D - 1) * PER < 64, "vmcnt counts to 63");
    __shared__ __align__(16) u8 xl[NB][MR * KS];
    __shared__ __align__(16) u32 sl[NB][MR];
    const int tid = threadIdx.x, l = lane(), r16 = l % 16, g = l / 16;
    const int wave = tid / kWave, wn = wave % WN, wm = wave / WN;
    const int m0 = tm * MR, n0 = tn * WN * NS * 16 + wn * NS * 16;
    const int stages = s_end - s_begin, sb = k / 32;  // scale bytes a row
    // 8-byte slots of a row's stage, slot e at e ^ (r % 16): a B read's 16
    // rows land in 16 distinct slots.
    auto at8 = [](int r, int e) { return r * KS + 8 * (e ^ (r & 15)); };

    // The activation: wave read j lands rows 2q, 2q + 1 of the stage (q =
    // wave·XI + j), lane l's dword at 4l, from the slot the swizzle puts
    // there.
    u32 xo[XI];
#pragma unroll
    for (int j = 0; j < XI; ++j) {
        const int q = wave * XI + j, r = 2 * q + l / 32, d = l % 32;
        const int row = min(m0 + r, m - 1);
        xo[j] = static_cast<u32>(row) * static_cast<u32>(k) + 8 * ((d / 2) ^ (r & 15)) + 4 * (d % 2);
    }
    // The scales: wave w lands rows 64·(w % (MR / 64)) .. +64 (waves past
    // the tile's rows land the same words again).
    static_assert(MR % 64 == 0, "the scales land 64 rows a wave");
    constexpr int SH = MR / 64;
    const u32 so = static_cast<u32>(min(m0 + 64 * (wave % SH) + l, m - 1)) * static_cast<u32>(sb);
    const u32 uw = static_cast<u32>(__builtin_amdgcn_readfirstlane(wave));
    const u32 xl0 = lds_address(xl[0]) + uw * XI * 256, sl0 = lds_address(sl[0]) + (uw % SH) * 256;
    // The weight: strip ns of the wave, lane l's 16 bytes of each unit.
    const moe_rsrc cr = batch_plane(codes, 1ull * n * k);
    u32 co[NS];
#pragma unroll
    for (int ns = 0; ns < NS; ++ns)
        co[ns] = static_cast<u32>(min(n0 / 16 + ns, n / 16 - 1)) * static_cast<u32>(16 * k) + 16 * l;
    constexpr int BLK = SB == 32 ? 32 : 128;
    const moe_rsrc fr = batch_plane(factors, 1ull * ((n + BLK - 1) / BLK) * sb);
    u32 fo[FW];
#pragma unroll
    for (int f = 0; f < FW; ++f) fo[f] = static_cast<u32>(min(n0 + 32 * f, n - 1) / BLK) * static_cast<u32>(sb);

    i32x4 wv[NB][2 * NS];
    moe_i2 wf[NB][FW];
    auto issue = [&](int sl_, int b) {
        const int s = s_begin + sl_;
#pragma unroll
        for (int j = 0; j < XI; ++j) lds_dma4(act, xo[j] + s * KS, xl0 + b * (MR * KS) + j * 256);
        lds_dma4(act_scales, so + 4 * s, sl0 + b * (MR * 4));
#pragma unroll
        for (int ns = 0; ns < NS; ++ns)
#pragma unroll
            for (int u = 0; u < 2; ++u) bload_kept(wv[b][2 * ns + u], cr, co[ns] + (2 * s + u) * 1024);
#pragma unroll
        for (int f = 0; f < FW; ++f) bload_run<4>(wf[b][f], fr, fo[f] + 4 * s);
    };

    fp8_f4 acc[NS][MS];
#pragma unroll
    for (int ns = 0; ns < NS; ++ns)
#pragma unroll
        for (int ms = 0; ms < MS; ++ms) acc[ns][ms] = fp8_f4{0.f, 0.f, 0.f, 0.f};
    int units[FW][MS];  // SB 32: each sum's biased exponents (`step`)
#pragma unroll
    for (int f = 0; f < FW; ++f)
#pragma unroll
        for (int ms = 0; ms < MS; ++ms) units[f][ms] = 254;

    auto step = [&](int s, int b) {
        tie(wv[b]);
#pragma unroll
        for (int f = 0; f < FW; ++f) batch_hold(wf[b][f]);
        const u8* xc = xl[b];
        u32 ae[MS];
#pragma unroll
        for (int ms = 0; ms < MS; ++ms) ae[ms] = sl[b][wm * MS * 16 + ms * 16 + r16];
        if constexpr (SB == 32) {
            // Each sum in units of its last step's scale, 4·2^(E − 254), E
            // the step's two biased exponents: a step's products chain onto
            // the sum once it moves to the new step's units, a power of two
            // and so exact.
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                moe_u2 bv[MS];
#pragma unroll
                for (int ms = 0; ms < MS; ++ms) {
                    const int r = wm * MS * 16 + ms * 16 + r16;
                    bv[ms] = *reinterpret_cast<const moe_u2*>(xc + at8(r, 4 * kk + g));
                }
#pragma unroll
                for (int f = 0; f < FW; ++f)
#pragma unroll
                    for (int ms = 0; ms < MS; ++ms) {
                        const int e = static_cast<int>(((static_cast<u32>(wf[b][f][0]) >> (8 * kk)) & 0xFFu) +
                                                       ((ae[ms] >> (8 * kk)) & 0xFFu));
                        const float to = __builtin_ldexpf(1.f, units[f][ms] - e);
                        units[f][ms] = e;
#pragma unroll
                        for (int ns = f * NS / FW; ns < (f + 1) * NS / FW; ++ns)
                            acc[ns][ms] = acc[ns][ms] * fp8_f4{to, to, to, to};
                    }
#pragma unroll
                for (int ns = 0; ns < NS; ++ns) {
                    const i32x4& u = wv[b][2 * ns + kk / 2];
#pragma unroll
                    for (int ms = 0; ms < MS; ++ms)
                        acc[ns][ms] = mma_fp8(static_cast<u32>(u[2 * (kk % 2)]), static_cast<u32>(u[2 * (kk % 2) + 1]),
                                              bv[ms][0], bv[ms][1], acc[ns][ms]);
                }
            }
        } else {
            fp8_f4 p[NS][MS];
#pragma unroll
            for (int ns = 0; ns < NS; ++ns)
#pragma unroll
                for (int ms = 0; ms < MS; ++ms) p[ns][ms] = fp8_f4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                moe_u2 bv[MS];
#pragma unroll
                for (int ms = 0; ms < MS; ++ms) {
                    const int r = wm * MS * 16 + ms * 16 + r16;
                    bv[ms] = *reinterpret_cast<const moe_u2*>(xc + at8(r, 4 * kk + g));
                }
#pragma unroll
                for (int ns = 0; ns < NS; ++ns) {
                    const i32x4& u = wv[b][2 * ns + kk / 2];
#pragma unroll
                    for (int ms = 0; ms < MS; ++ms)
                        p[ns][ms] = mma_fp8(static_cast<u32>(u[2 * (kk % 2)]), static_cast<u32>(u[2 * (kk % 2) + 1]),
                                            bv[ms][0], bv[ms][1], p[ns][ms]);
                }
            }
            const float wsc = 4.f * __int_as_float(wf[b][0][0]);
#pragma unroll
            for (int ms = 0; ms < MS; ++ms) {
                const float f = wsc * __uint_as_float(ae[ms]);
#pragma unroll
                for (int ns = 0; ns < NS; ++ns)
                    acc[ns][ms] = __builtin_elementwise_fma(fp8_f4{f, f, f, f}, p[ns][ms], acc[ns][ms]);
            }
        }
        // Stage s + 1 landed; the stages past it may still be in flight.
        if (s + D < stages) vm_wait<(D - 1) * PER>(); else vm_wait<0>();
        lds_barrier();
    };
#pragma unroll
    for (int d = 0; d < D; ++d)
        if (d < stages) issue(d, d);
    if (D < stages) vm_wait<(D - 1) * PER>(); else vm_wait<0>();
    lds_barrier();
    if constexpr (SB == 32) {
        // The sums start in their first step's units (zero in any).
#pragma unroll
        for (int f = 0; f < FW; ++f) batch_hold(wf[0][f]);
#pragma unroll
        for (int f = 0; f < FW; ++f)
#pragma unroll
            for (int ms = 0; ms < MS; ++ms)
                units[f][ms] = static_cast<int>((static_cast<u32>(wf[0][f][0]) & 0xFFu) +
                                                (sl[0][wm * MS * 16 + ms * 16 + r16] & 0xFFu));
    }
    for (int s0 = 0; s0 < stages; s0 += NB) {
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int s = s0 + b;
            if (s >= stages) break;
            if (s + D < stages) issue(s + D, (b + D) % NB);
            step(s, b);
        }
    }

    if (slot != nullptr) {
        // A slice of the tile's K for `dense_fp8_tiles`: its f32 partial, the
        // tile's own rows and columns.
        const __amdgpu_buffer_rsrc_t sr = shared_plane(slot, 4 * MR * WN * NS * 16);
#pragma unroll
        for (int ms = 0; ms < MS; ++ms)
#pragma unroll
            for (int ns = 0; ns < NS; ++ns) {
                const float f = SB == 32 ? __builtin_ldexpf(4.f, units[ns * FW / NS][ms] - 254) : 1.f;
                const int r = wm * MS * 16 + ms * 16 + r16, c = wn * NS * 16 + ns * 16 + 4 * g;
                store_shared16(sr, 4 * (r * (WN * NS * 16) + c), acc[ns][ms] * fp8_f4{f, f, f, f});
            }
        return;
    }
#pragma unroll
    for (int ms = 0; ms < MS; ++ms) {
        const int row = m0 + wm * MS * 16 + ms * 16 + r16;
        if (row >= m) continue;
        bf16* o = y + static_cast<long long>(row) * n;
#pragma unroll
        for (int ns = 0; ns < NS; ++ns) {
            const int c = n0 + ns * 16 + 4 * g;
            if (c >= n) continue;
            const float f = SB == 32 ? __builtin_ldexpf(4.f, units[ns * FW / NS][ms] - 254) : 1.f;
            bf16 v[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) v[t] = from_f32<bf16>(acc[ns][ms][t] * f);
            *reinterpret_cast<moe_u2*>(o + c) = __builtin_bit_cast(moe_u2, v);
        }
    }
}

// A dense FP8 projection's prefill rows, the tiles taken whole (tile-major
// along M within a weight tile) but for the last `rem` tiles, short of a
// round, whose K goes in `splits` slices of `per` stages a block: blocks [0,
// tiles − rem) a tile each (renumbered so that an XCD runs a contiguous
// range), then a slice each, round the XCDs. A slice's f32 partial lands in
// `part` ([rem, splits, tile rows x cols], the tile's own layout) past the
// L2s, and the last of a tile's slices to finish (by its ticket in
// `tickets`, zero between launches) sums them in slice order and lands the
// tile, so the sum's order is the slices' whatever order they finish in. A
// staged window bounds the rows.
template <int SB, int NB, int WN, int WM, int NS, int MS, int OCC>
__global__ __launch_bounds__(64 * WN * WM) __attribute__((amdgpu_waves_per_eu(OCC, OCC))) void dense_fp8_tiles(
    const u8* __restrict__ act, const u8* __restrict__ act_scales, const u8* __restrict__ codes,
    const u8* __restrict__ factors, bf16* __restrict__ y, float* __restrict__ part, u32* __restrict__ tickets, int m,
    int n, int k, int rem, int splits, int per, const u32* __restrict__ win) {
    constexpr int TMR = WM * MS * 16, TNR = WN * NS * 16, TS = TMR * TNR, T = 64 * WN * WM;
    // The tiles over the stated rows, as the launch dealt them; a staged
    // window's rows bound what lands, its tiles past them idle (every slice
    // of such a tile alike, so none waits on a ticket).
    const int tms = (m + TMR - 1) / TMR, tns = (n + TNR - 1) / TNR, tiles = tms * tns, dp = tiles - rem;
    if (win != nullptr) m = min(m, static_cast<int>(win[0]));
    const long long first = win != nullptr ? Window{win}.plane(0) : 0;
    const int live = dp + rem * splits;
    const int b = static_cast<int>(blockIdx.x);
    if (b >= live) return;
    // The whole tiles renumbered so that an XCD runs a contiguous range; the
    // slices after them as dealt, round the XCDs.
    int at = b;
    if (b < dp) {
        const int xcd = b % 8, slots = dp / 8, extra = dp % 8;
        at = xcd * slots + min(xcd, extra) + b / 8;
    }
    const int t = at < dp ? at : dp + (at - dp) / splits, split = at < dp ? 0 : (at - dp) % splits;
    const int stages = k / 128, tm = t % tms, tn = t / tms;
    if (tm * TMR >= m) return;
    act += first * k;
    act_scales += first * (k / 32);
    y += first * n;
    float* mine = at < dp ? nullptr : part + (static_cast<long long>(t - dp) * splits + split) * TS;
    fp8_prefill_tile<SB, NB, WN, WM, NS, MS>(act, act_scales, codes, factors, y, m, n, k, tm, tn,
                                             at < dp ? 0 : split * per, at < dp ? stages : min((split + 1) * per, stages),
                                             mine);
    if (at < dp) return;
    __shared__ int last;
    // Every lane's partial at the memory the XCDs share before the ticket
    // moves; the ticket there too (sc0 sc1).
    vm_wait<0>();
    __syncthreads();
    if (threadIdx.x == 0)
        last = __hip_atomic_fetch_add(&tickets[t - dp], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) ==
               static_cast<u32>(splits - 1);
    __syncthreads();
    if (!last) return;
    constexpr int PER_T = TS / (T * 4);  // 16-byte pieces a thread
    static_assert(PER_T * T * 4 == TS, "a tile is whole pieces a thread");
    fp8_f4 v[PER_T];
#pragma unroll
    for (int q = 0; q < PER_T; ++q) v[q] = fp8_f4{0.f, 0.f, 0.f, 0.f};
    for (int sp = 0; sp < splits; ++sp) {
        const __amdgpu_buffer_rsrc_t src = shared_plane(part + (static_cast<long long>(t - dp) * splits + sp) * TS, 4 * TS);
#pragma unroll
        for (int q = 0; q < PER_T; ++q) v[q] += load_shared16(src, 16 * (q * T + threadIdx.x));
    }
    const int r0 = tm * TMR, c0 = tn * TNR;
#pragma unroll
    for (int q = 0; q < PER_T; ++q) {
        const int e = (q * T + threadIdx.x) * 4, r = e / TNR, c = e % TNR;
        if (r0 + r >= m || c0 + c >= n) continue;
        bf16 o[4];
#pragma unroll
        for (int z = 0; z < 4; ++z) o[z] = from_f32<bf16>(v[q][z]);
        *reinterpret_cast<moe_u2*>(y + static_cast<long long>(r0 + r) * n + c0 + c) = __builtin_bit_cast(moe_u2, o);
    }
    if (threadIdx.x == 0) __hip_atomic_store(&tickets[t - dp], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// An MXFP4 bank (e2m1 nibble pairs, the low one first, under an e8m0
// exponent a 1x32 block, in the matrix core's lane order: `mxfp4_lanes`) on the bf16 matrix core: each lane
// reads 16 bytes of its row a stage, one block, and decodes them to bf16
// in registers under the block's exponent (exact: every e2m1 value times a
// power of two is a bf16). The k a matrix-core step takes is permuted alike
// in A and B: lane group g's step kk holds k 32g + 4kk .. +3 of the stage, so
// a lane's 32 elements are one block and one read. A stage is 128 k; the
// routes' rows go through LDS, 16-byte chunks XOR-swizzled by row. NW waves side by side
// along the rows, TN strips a wave, TM 16-route blocks.
template <int NW, int TN, int TM, int NB>
__device__ __forceinline__ void fp4_group_tile(const bf16* __restrict__ act, const i32* __restrict__ order,
                                               int begin, int count, const u8* __restrict__ codes,
                                               const u8* __restrict__ scales, bf16* __restrict__ out, int act_div,
                                               int n, int k, int tile) {
    constexpr int KS = 128, RB = 2 * KS;       // k a stage, bytes a route row a stage
    constexpr int NR = NW * TN * 16, MR = TM * 16;
    constexpr int XI = MR * RB / 256 / NW;     // activation reads a wave a stage
    static_assert(XI * NW * 256 == MR * RB, "a stage is whole wave reads");
    constexpr int SB = MR * RB;
    constexpr int PER = XI + 2 * TN;           // reads a thread a stage
    constexpr int D = NB - 1;                  // stages read ahead
    static_assert((D - 1) * PER < 64, "vmcnt counts to 63");
    __shared__ __align__(16) u8 xl[NB][SB];
    const int tid = threadIdx.x, l = lane(), r16 = l % 16, g = l / 16;
    const int wave = tid / kWave;
    const int row0 = tile * NR;
    const int stages = k / KS;
    auto route = [&](int m) { return order != nullptr ? order[begin + m] : begin + m; };
    auto xat = [&](int r, int c) { return r * RB + 16 * (c ^ (r & 7)); };

    // A strip's unit (128 k of 16 rows): 1 KB of codes, lane l's 16 bytes
    // at 16·l, and 64 bytes of exponents, lane l's at l. A strip past the
    // bank reads the last.
    const moe_rsrc cr = batch_plane(codes, 1ull * n * (k / 2));
    const moe_rsrc sr = batch_plane(scales, 1ull * n * (k / 32));
    u32 co[TN], so[TN];
#pragma unroll
    for (int nb = 0; nb < TN; ++nb) {
        const int strip = min(row0 / 16 + wave * TN + nb, n / 16 - 1);
        co[nb] = static_cast<u32>(strip) * static_cast<u32>(8 * k) + 16 * l;
        so[nb] = static_cast<u32>(strip) * static_cast<u32>(k / 2) + l;
    }
    const u8* xb = reinterpret_cast<const u8*>(act);
    u32 xo[XI];
#pragma unroll
    for (int j = 0; j < XI; ++j) {
        const int q = wave * XI + j;
        const int byte = q * 256 + 4 * l;
        const int r = byte / RB, cl = byte % RB / 16;
        const int c = cl ^ (r & 7);
        xo[j] = static_cast<u32>(route(r < count ? r : 0) / act_div) * static_cast<u32>(2 * k) + 16 * c + 4 * (l % 4);
    }
    const u32 uw = static_cast<u32>(__builtin_amdgcn_readfirstlane(wave));
    const u32 xl0 = lds_address(xl[0]) + uw * XI * 256;

    i32x4 wc[NB][TN];
    moe_i2 ws[NB][TN];
    auto issue = [&](int s, int b) {
#pragma unroll
        for (int j = 0; j < XI; ++j) lds_dma4(xb, xo[j] + s * RB, xl0 + b * SB + j * 256);
#pragma unroll
        for (int nb = 0; nb < TN; ++nb) {
            bload_nt(wc[b][nb], cr, co[nb] + s * 1024);
            bload_run<1>(ws[b][nb], sr, so[nb] + s * 64);
        }
    };

    mfma_cd acc[TN][TM];
#pragma unroll
    for (int nb = 0; nb < TN; ++nb)
#pragma unroll
        for (int mb = 0; mb < TM; ++mb) acc[nb][mb] = mfma_zero();

    // e2m1 magnitudes 0 .. 7 as bf16 (0, 0.5, 1, 1.5, 2, 3, 4, 6), two a
    // word, 0 and 1 in the first.
    constexpr u32 kBase[4] = {0x3F000000u, 0x3FC03F80u, 0x40404000u, 0x40C04080u};
    using u16x2 = unsigned short __attribute__((ext_vector_type(2)));
    auto step = [&](int s, const u8* xc, i32x4 (&wset)[TN], moe_i2 (&eset)[TN]) {
        tie(wset);
#pragma unroll
        for (int nb = 0; nb < TN; ++nb) batch_hold(eset[nb]);
        u32 pairs[TN][16];
#pragma unroll
        for (int nb = 0; nb < TN; ++nb) {
            // The block's table: each magnitude's bf16 under its exponent
            // (its exponent field moved by e - 127, a halfword at a time;
            // zero kept zero), then split into high and low bytes, 0..3 in
            // the first word.
            const u32 d16 = (((static_cast<u32>(eset[nb][0]) & 0xFFu) - 127u) << 7) & 0xFFFFu;
            const u16x2 dv = __builtin_bit_cast(u16x2, d16 * 0x10001u);
            u32 q[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) q[j] = __builtin_bit_cast(u32, __builtin_bit_cast(u16x2, kBase[j]) + dv);
            q[0] &= 0xFFFF0000u;
            const u32 hi[2] = {__builtin_amdgcn_perm(q[1], q[0], 0x07050301u), __builtin_amdgcn_perm(q[3], q[2], 0x07050301u)};
            const u32 lo[2] = {__builtin_amdgcn_perm(q[1], q[0], 0x06040200u), __builtin_amdgcn_perm(q[3], q[2], 0x06040200u)};
            // A word's byte j holds k j (low nibble) and k j + 4 (high), as
            // `mxfp4_lanes` packs them: one perm a bf16 pair.
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const u32 w = static_cast<u32>(wset[nb][d]);
                const u32 ml = w & 0x07070707u, mh = (w >> 4) & 0x07070707u;
                const u32 hl = __builtin_amdgcn_perm(hi[1], hi[0], ml) | ((w & 0x08080808u) << 4);
                const u32 ll = __builtin_amdgcn_perm(lo[1], lo[0], ml);
                const u32 hh = __builtin_amdgcn_perm(hi[1], hi[0], mh) | (w & 0x80808080u);
                const u32 lh = __builtin_amdgcn_perm(lo[1], lo[0], mh);
                pairs[nb][4 * d + 0] = __builtin_amdgcn_perm(hl, ll, 0x05010400u);  // k 0, 1
                pairs[nb][4 * d + 1] = __builtin_amdgcn_perm(hl, ll, 0x07030602u);  // k 2, 3
                pairs[nb][4 * d + 2] = __builtin_amdgcn_perm(hh, lh, 0x05010400u);  // k 4, 5
                pairs[nb][4 * d + 3] = __builtin_amdgcn_perm(hh, lh, 0x07030602u);  // k 6, 7
            }
        }
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            moe_u2 bv[TM];
#pragma unroll
            for (int mb = 0; mb < TM; ++mb) {
                const int r = mb * 16 + r16;
                bv[mb] = *reinterpret_cast<const moe_u2*>(xc + xat(r, 4 * g + kk / 2) + 8 * (kk % 2));
            }
#pragma unroll
            for (int nb = 0; nb < TN; ++nb)
#pragma unroll
                for (int mb = 0; mb < TM; ++mb)
                    acc[nb][mb] = mma16x16<bf16>(pairs[nb][2 * kk], pairs[nb][2 * kk + 1], bv[mb][0], bv[mb][1],
                                                 acc[nb][mb]);
        }
        // Stage s + 1 landed; the stages past it may still be in flight.
        if (s + D < stages) vm_wait<(D - 1) * PER>(); else vm_wait<0>();
        lds_barrier();
    };
#pragma unroll
    for (int d = 0; d < D; ++d)
        if (d < stages) issue(d, d);
    if (D < stages) vm_wait<(D - 1) * PER>(); else vm_wait<0>();
    lds_barrier();
    for (int s0 = 0; s0 < stages; s0 += NB) {
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int s = s0 + b;
            if (s >= stages) break;
            if (s + D < stages) issue(s + D, (b + D) % NB);
            step(s, xl[b], wc[b], ws[b]);
        }
    }

#pragma unroll
    for (int mb = 0; mb < TM; ++mb) {
        const int m = mb * 16 + r16;
        if (m >= count) continue;
        bf16* o = out + static_cast<long long>(route(m)) * n;
#pragma unroll
        for (int nb = 0; nb < TN; ++nb) {
            const int c = row0 + (wave * TN + nb) * 16 + 4 * g;
            bf16 v[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) v[t] = from_f32<bf16>(acc[nb][mb][t]);
            if (c + 3 < n && n % 4 == 0) {
                *reinterpret_cast<moe_u2*>(o + c) = __builtin_bit_cast(moe_u2, v);
            } else {
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    if (c + t < n) o[c + t] = v[t];
            }
        }
    }
}

// An MXFP4 plane of `rows` x `k` (codes [rows, k/2], exponents [rows,
// k/32], rows a multiple of 16, k of 128) into the matrix core's lane order:
// strip q (16 rows), unit u (128 k): 1 KB of codes, lane l's 16 bytes the
// bytes 16·(l / 16) .. +16 of row l % 16 in the unit (its 32 k, one block),
// and 64 bytes of exponents, lane l's the block l / 16 of row l % 16; each
// 4-byte word's codes k 0 .. 3 in its bytes' low nibbles, k 4 .. 7 in their
// high ones. A thread a 16-byte chunk of codes and its lane's exponent.
__global__ __launch_bounds__(256) void mxfp4_lanes(const u8* __restrict__ codes, u8* __restrict__ out_codes,
                                                   const u8* __restrict__ scales, u8* __restrict__ out_scales,
                                                   i32 rows, i32 k) {
    const long long units = k / 128, chunks = static_cast<long long>(rows / 16) * units * 64;
    for (long long c = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; c < chunks;
         c += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long strip = c / (units * 64), u = c / 64 % units;
        const int l = static_cast<int>(c % 64);
        const long long row = strip * 16 + l % 16;
        uint4 v = *reinterpret_cast<const uint4*>(codes + row * (k / 2) + u * 64 + 16 * (l / 16));
        // Each word's eight codes (k 2j low, 2j + 1 high in byte j) repacked
        // as k j low, j + 4 high.
        u32* wv = reinterpret_cast<u32*>(&v);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const u32 w = wv[d];
            u32 o = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const u32 code = (w >> (4 * j)) & 0xFu;
                o |= code << (j < 4 ? 8 * j : 8 * (j - 4) + 4);
            }
            wv[d] = o;
        }
        *reinterpret_cast<uint4*>(out_codes + 16 * c) = v;
        out_scales[c] = scales[row * (k / 32) + u * 4 + l / 16];
    }
}

// The same select by route, for a token's few routes: route r (blockIdx.x)
// its own item of one row against expert routes[r], unordered (grid
// (routes, ⌈n / (16·TN·NW)⌉)).
template <int NW, int TN, int NB>
__global__ __launch_bounds__(NW * 64) void moe_mxfp4_route(const bf16* __restrict__ act, const i32* __restrict__ routes,
                                                           const u8* __restrict__ codes, const u8* __restrict__ scales,
                                                           bf16* __restrict__ out, int act_div, int n, int k,
                                                           int num_experts) {
    const int r = static_cast<int>(blockIdx.x), e = routes[r];
    if (e < 0 || e >= num_experts) return;
    fp4_group_tile<NW, TN, 1, NB>(act, nullptr, r, 1, codes + static_cast<long long>(e) * n * (k / 2),
                                 scales + static_cast<long long>(e) * n * (k / 32), out, act_div, n, k, blockIdx.y);
}

// An MXFP4 bank's routed select on the bf16 matrix core: one work item of up
// to 16·TM routes of one expert a block row of NW·64 weight rows (grid (work
// cap, ⌈n / (64·NW)⌉), 64·NW threads), the blocks renumbered so an XCD runs
// a contiguous range, every item at one block row before the next.
template <int NW, int TM, int NB = 4, int TN = 4>
__global__ __launch_bounds__(NW * 64) void moe_mxfp4_group(const bf16* __restrict__ act, const i32* __restrict__ order,
                                                           const i32* __restrict__ offsets, const i32* __restrict__ work,
                                                           const u8* __restrict__ codes, const u8* __restrict__ scales,
                                                           bf16* __restrict__ out, int act_div, int n, int k,
                                                           int num_experts) {
    const int items = offsets[num_experts + 1];
    const int live = items * static_cast<int>(gridDim.y);
    const int b = static_cast<int>(blockIdx.x + gridDim.x * blockIdx.y);
    if (b >= live) return;
    const int xcd = b % 8, slots = live / 8, extra = live % 8;
    const int at = xcd * slots + min(xcd, extra) + b / 8;
    GroupItem item;
    if (!group_item(offsets, work, num_experts, 16 * TM, item, at % items)) return;
    const long long e = item.expert;
    fp4_group_tile<NW, TN, TM, NB>(act, order, item.begin, item.count, codes + e * n * (k / 2), scales + e * n * (k / 32), out,
                              act_div, n, k, at / items);
}

// A bf16 bank's routed select past 64 routes an expert: one work item of up
// to 128 routes of one expert a block row (grid (work cap, ⌈n / 128⌉)).
// Slice z of K (`per` stages) of `gridDim.z`; under slices each one's f32
// partial lands in `part` ([slices, routes, n], by route) for
// `fp8_mfma_finalize`.
__global__ __launch_bounds__(512) void moe_bf16_gemm8(const bf16* __restrict__ act, const i32* __restrict__ order,
                                                      const i32* __restrict__ offsets, const i32* __restrict__ work,
                                                      const bf16* __restrict__ bank, bf16* __restrict__ out,
                                                      float* __restrict__ part, int act_div, int n, int k,
                                                      int num_experts, int routes, int per) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, 128, item)) return;
    const int split = static_cast<int>(blockIdx.z);
    fp8_gemm_tile8<false>(reinterpret_cast<const u8*>(act), nullptr, order, item.begin, item.count,
                          reinterpret_cast<const u8*>(bank + static_cast<long long>(item.expert) * n * k), nullptr,
                          out, act_div, n, k, blockIdx.y, split * per, (split + 1) * per,
                          gridDim.z > 1 ? part + static_cast<long long>(split) * routes * n : nullptr);
}

__global__ __launch_bounds__(512) void moe_fp8_gemm8(const u8* __restrict__ act, const float* __restrict__ act_scales,
                                                     const i32* __restrict__ order, const i32* __restrict__ offsets,
                                                     const i32* __restrict__ work, const u8* __restrict__ codes,
                                                     const u8* __restrict__ factors, bf16* __restrict__ out,
                                                     int act_div, int n, int k, int num_experts,
                                                     const MoeGroupBases* __restrict__ bases,
                                                     unsigned int* __restrict__ group_hits) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, 128, item)) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(item.count));
    const u8* c = bases != nullptr ? bases->codes : codes;
    const u8* f = bases != nullptr ? bases->scales : factors;
    const long long blocks = static_cast<long long>((n + 127) / 128) * (k / 128);
    fp8_gemm_tile8(act, act_scales, order, item.begin, item.count, c + static_cast<long long>(item.expert) * n * k,
                   reinterpret_cast<const float*>(f) + static_cast<long long>(item.expert) * blocks, out, act_div, n,
                   k, blockIdx.y, 0, k / 128, nullptr);
}



// The fp8 stages in flight (three measured faster than four or more).
constexpr int kFp8MfmaDepth = 3;

// A dense `e4m3tile128` projection's decode rows (up to 64) as one
// expert every row routes to: rows [item·MT, +MT) an item, 64 weight rows a
// tile, one dimension of blocks dealt as `dense_batch_bf16` deals them (each
// tile's `splits` slices of K, `items` items each, the tiles round the eight
// XCDs). Under splits a slice's f32 partial lands in `part` ([splits, rows,
// n]) for `fp8_mfma_finalize`. A staged window bounds the rows, which the
// planes hold from its first.
template <int MT, int NB, int SB = 0>
__global__ __launch_bounds__(256) void dense_fp8_mfma(const u8* __restrict__ act, const float* __restrict__ act_scales,
                                                      const u8* __restrict__ codes,
                                                      const float* __restrict__ factors, bf16* __restrict__ y,
                                                      float* __restrict__ part, int m, int n, int k, int items,
                                                      int tiles, int splits, int per, const u32* __restrict__ win) {
    const int b = blockIdx.x, xcd = b % 8, slot = b / 8;
    const int item = slot % items, rest = slot / items, split = rest % splits;
    const int tile = rest / splits * 8 + xcd;
    if (tile >= tiles) return;
    const int rows = m;
    if (win != nullptr) m = min(m, static_cast<int>(win[0]));
    const int begin = item * MT;
    if (begin >= m) return;
    const long long first = win != nullptr ? Window{win}.plane(0) : 0;
    fp8_tile<MT, kFp8MfmaDepth, NB, SB>(act + first * k, act_scales + first * (k / 128), nullptr, begin, min(MT, m - begin),
                                codes, factors, y + first * n,
                                splits > 1 ? part + static_cast<long long>(split) * rows * n : nullptr, 1, n, k, tile,
                                split * per, (split + 1) * per);
}

// An `e4m3tile128` bank's routed select at decode batches: one
// work item of up to MT routes of one expert a block row (grid (work cap,
// ⌈n / 64⌉)).
template <int MT, int NB>
__global__ __launch_bounds__(256) void moe_fp8_mfma(const u8* __restrict__ act, const float* __restrict__ act_scales,
                                                    const i32* __restrict__ order,
                                                    const i32* __restrict__ offsets, const i32* __restrict__ work,
                                                    const u8* __restrict__ codes, const u8* __restrict__ factors,
                                                    bf16* __restrict__ out, int act_div, int n, int k,
                                                    int num_experts, const MoeGroupBases* __restrict__ bases,
                                                    unsigned int* __restrict__ group_hits) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item)) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(item.count));
    const u8* c = bases != nullptr ? bases->codes : codes;
    const u8* f = bases != nullptr ? bases->scales : factors;
    const long long blocks = static_cast<long long>((n + 127) / 128) * (k / 128);
    fp8_tile<MT, kFp8MfmaDepth, NB>(act, act_scales, order, item.begin, item.count, c + static_cast<long long>(item.expert) * n * k,
                                reinterpret_cast<const float*>(f) + static_cast<long long>(item.expert) * blocks, out,
                                nullptr, act_div, n, k, blockIdx.y, 0, k / 128);
}

// x's rows quantized as e4m3 codes under an f32 scale a GRAN-wide block,
// `amax / 448` (the amax floored at 1e-10), rounded to nearest even: the
// OCP bytes in `codes` ([rows, k]), negative zero as zero (the matrix core
// reads them as fnuz), each block's scale in `scales` ([rows, k / GRAN]).
// x's rows quantized as DeepSeek's recipe does (`act_quant_e4m3_values`:
// a ue8m0 scale a 32-wide block, 2^ceil(log2(max(amax, 1e-4) / 448))) into
// e4m3 codes as the matrix core reads them (zero 0x00) beside each block's
// exponent byte. A thread eight elements, four a block.
__global__ __launch_bounds__(256) void act_quant_e4m3_e8m0(const bf16* __restrict__ x, u8* __restrict__ codes,
                                                           u8* __restrict__ exps, i32 rows, i32 k,
                                                           const u32* __restrict__ win) {
    const Window window{win};
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int per = k / 8;
    if (t >= static_cast<long long>(rows) * per) return;
    const long long r = t / per;
    if (!window.live(r)) return;
    const long long row = window.plane(r);
    const long long at = row * k + (t % per) * 8;
    const Pack8<bf16> in = *reinterpret_cast<const Pack8<bf16>*>(x + at);
    float v[8];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = to_f32(in.v[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = fmaxf(amax, __shfl_xor(amax, 1, kWave));
    amax = fmaxf(amax, __shfl_xor(amax, 2, kWave));
    amax = fmaxf(amax, 1e-4f);
    const u32 bits = __float_as_uint(amax * (1.f / 448.f));
    const int e = static_cast<int>((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
    const float sc = __uint_as_float(static_cast<u32>(e + 127) << 23);
    u32 word[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 8; ++i) word[i / 4] |= e4m3fn_bits(fminf(fmaxf(v[i] / sc, -448.f), 448.f)) << (8 * (i % 4));
    *reinterpret_cast<uint2*>(codes + at) = uint2{word[0], word[1]};
    if (t % 4 == 0) exps[at / 32] = static_cast<u8>(e + 127);
}

template <int GRAN>
__global__ __launch_bounds__(256) void act_quant_e4m3_codes(const bf16* __restrict__ x, u8* __restrict__ codes,
                                                            float* __restrict__ scales, i32 rows, i32 k,
                                                            const u32* __restrict__ win) {
    static_assert(GRAN % kWave == 0, "a block spans the wavefront");
    constexpr int EACH = GRAN / kWave;
    const Window window{win};
    const int blocks = k / GRAN;
    const long long slot = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / kWave;
    if (slot >= static_cast<long long>(rows) * blocks) return;
    const long long r = slot / blocks;
    if (!window.live(r)) return;
    const long long row = window.plane(r);
    const int b = static_cast<int>(slot % blocks);
    const int l = lane();
    const long long at = row * k + static_cast<long long>(b) * GRAN + l * EACH;
    float v[EACH];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < EACH; ++i) {
        v[i] = to_f32(x[at + i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, kWave));
    const float sc = e4m3_block_scale(amax);
    u32 word = 0;
#pragma unroll
    for (int i = 0; i < EACH; ++i) {
        const u32 bits = e4m3fn_bits(fminf(fmaxf(v[i] / sc, -448.f), 448.f));
        word |= bits << (8 * (i % 4));
        if (i % 4 == 3) {
            *reinterpret_cast<u32*>(codes + at + i - 3) = word;
            word = 0;
        }
    }
    if constexpr (EACH % 4 != 0) {
#pragma unroll
        for (int i = EACH - EACH % 4; i < EACH; ++i) codes[at + i] = static_cast<u8>(word >> (8 * (i % 4)));
    }
    if (l == 0) scales[row * blocks + b] = sc;
}

// The slices' f32 partials of `dense_fp8_mfma` summed in slice order and
// rounded into y's live rows.
__global__ __launch_bounds__(256) void fp8_mfma_finalize(const float* __restrict__ part, bf16* __restrict__ y, int m,
                                                         int n, int splits, const u32* __restrict__ win) {
    const int rows = m;
    if (win != nullptr) m = min(m, static_cast<int>(win[0]));
    const long long first = win != nullptr ? Window{win}.plane(0) : 0;
    const long long total = static_cast<long long>(m) * n;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < splits; ++s) v += part[static_cast<long long>(s) * rows * n + i];
        y[first * n + i] = from_f32<bf16>(v);
    }
}

}  // namespace pie::linear
