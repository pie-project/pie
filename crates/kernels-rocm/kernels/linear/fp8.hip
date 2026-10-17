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
// CANON false: bytes put in order as they are (a bf16 weight, k its rows'
// bytes).
template <bool CANON = true>
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
                if constexpr (!CANON) {
                    w[2 * half + j] = x[j];
                    continue;
                }
                const u32 mag = x[j] & 0x7F7F7F7Fu;
                found += __popc((mag + 0x01010101u) & 0x80808080u);
                w[2 * half + j] = x[j] & (((mag + 0x7F7F7F7Fu) & 0x80808080u) | 0x7F7F7F7Fu);
            }
        }
        *reinterpret_cast<uint4*>(dst + q * 16) = uint4{w[0], w[1], w[2], w[3]};
    }
    if (CANON && found != 0u) atomicAdd(nans, found);
}

__device__ __forceinline__ fp8_f4 mma_fp8(u32 a0, u32 a1, u32 b0, u32 b1, fp8_f4 c) {
    const long a = static_cast<long>((static_cast<unsigned long long>(a1) << 32) | a0);
    const long b = static_cast<long>((static_cast<unsigned long long>(b1) << 32) | b0);
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, 0, 0);
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

// y[route] = W·act[route / act_div] over routes order[begin .. begin +
// count) (rows begin .. begin + count without an order), count ≤ MT, for
// the weight rows [tile·64, +64) of `codes` / `factors` (n x k, k a multiple
// of 128); `act` the rows' e4m3 codes ([rows, k], `act_quant_e4m3_codes`)
// beside their scales ([rows, k / 128]). Stages [s0, s1) of K; with `part`
// the f32 partial lands there (rows of n by the item's own order) for a
// later sum, else y rounds once. FP8 false: a bf16 weight in the matrix
// core's order (the same bytes: a row's 2k bytes as fp8_unit_at lays k
// codes, lane l's 8 bytes a k-step four elements of its row) against bf16
// rows, a stage 64 elements, nothing to fold (`act_scales`, `factors`
// unread).
template <int MT, int DEPTH, int NB = 1, bool FP8 = true>
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
    constexpr int NAL = FP8 ? NA + 1 : NA, NWL = FP8 ? NL + 1 : NL;  // with the activation's scale, the weight's factor
    static_assert(PA % kGroupThreads == 0 || PA < kGroupThreads, "the activation stage splits evenly");
    static_assert(DEPTH >= 2 && NWL + (DEPTH - 1) * (NAL + NWL) < 64, "the reads in flight fit the count");
    __shared__ __align__(16) u8 xs[2][MT * PITCH];
    __shared__ float xsc[2][MT];

    const int tid = threadIdx.x, l = lane(), r16 = l % 16, g = l / 16;
    const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const int row0 = (tile * WAVES + wave) * 16 * NB;
    const int kbytes = FP8 ? k : 2 * k;  // a row's bytes
    const int kb = k / 128;              // scale blocks a row (fp8)
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
    const float* srow =
        FP8 ? act_scales + static_cast<long long>(route(min(tid % MT, max(count - 1, 0))) / act_div) * kb : nullptr;
    // The weight in the matrix core's order (`fp8_unit_at`): the wave's NB
    // strips of 16 rows, a stage two 1 KB units of each, lane l's 16 bytes of
    // a unit its own A operand for two k-steps. A strip past the weight
    // reads the last.
    u32 wv[NB];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
        wv[nb] = static_cast<u32>(min(row0 + 16 * nb, n - 16) / 16) * static_cast<u32>(16 * kbytes) + 16 * l;
    // The wave's 16 rows lie in one 128-row block: its factor a stage.
    const float* frow = FP8 ? factors + static_cast<long long>(min(row0, n - 1) / 128) * kb : nullptr;

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
        if constexpr (FP8) gload_run<4>(as[S], srow + sc);
#pragma unroll
        for (int i = 0; i < NL; ++i) bload_nt(w[S][i], wr, on ? wv[i / 2] + (2 * sa + i % 2) * 1024 : kBatchVoid);
        if constexpr (FP8) gload_run<4>(ws[S], frow + sc);
    };
    auto store = [&](auto slot, int buf) {
        constexpr int S = decltype(slot)::value;
#pragma unroll
        for (int i = 0; i < NA; ++i)
            if (PA % kGroupThreads == 0 || mine[i])
                *reinterpret_cast<i32x4*>(xs[buf] + at[i]) = live[i] ? a[S][i] : i32x4{0, 0, 0, 0};
        if constexpr (FP8)
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
            if constexpr (!FP8) {
#pragma unroll
                for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const i32x4& u = w[S][2 * nb + kk / 2];
                        acc[nb][mb] = mma16x16<bf16>(static_cast<u32>(u[2 * (kk % 2)]),
                                                     static_cast<u32>(u[2 * (kk % 2) + 1]), xb[kk][0], xb[kk][1],
                                                     acc[nb][mb]);
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

__global__ __launch_bounds__(512) void dense_fp8_gemm8(const u8* __restrict__ act, const float* __restrict__ act_scales,
                                                       const u8* __restrict__ codes, const float* __restrict__ factors,
                                                       bf16* __restrict__ y, float* __restrict__ part, int m, int n,
                                                       int k, int per, const u32* __restrict__ win) {
    const int rows = m;
    if (win != nullptr) m = min(m, static_cast<int>(win[0]));
    const int begin = static_cast<int>(blockIdx.x) * 128;
    if (begin >= m) return;
    const long long first = win != nullptr ? Window{win}.plane(0) : 0;
    const int split = static_cast<int>(blockIdx.z);
    fp8_gemm_tile8(act + first * k, act_scales + first * (k / 128), nullptr, begin, min(128, m - begin), codes,
                   factors, y + first * n, 1, n, k, blockIdx.y, split * per, (split + 1) * per,
                   gridDim.z > 1 ? part + static_cast<long long>(split) * rows * n : nullptr);
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

// The bf16 prefill tile over a bank in the matrix core's order: up to TM·16
// routes of one expert against NW·TN·16 weight rows, NW waves side by side
// along the rows, each against every route. The weight goes straight from
// its order into each wave's registers (the wave's own TN strips, 16 bytes
// a lane a strip a 1 KB unit: the A operands as they land); the routes'
// rows go through NB stages of LDS, read by the card straight into it, 16-byte
// chunks XOR-swizzled by row so the B reads spread over the banks. Stages
// s + 1 .. s + NB - 1 of both are in flight while stage s multiplies, one
// barrier a stage. KS elements a stage; D cell (weight row 4g + t, route
// r16) of block (nb, mb).
template <int KS, int NW, int TN, int TM, int NB>
__device__ __forceinline__ void bf16_group_tile(const bf16* __restrict__ act, const i32* __restrict__ order,
                                                int begin, int count, const bf16* __restrict__ w,
                                                bf16* __restrict__ out, int act_div, int n, int k, int tile,
                                                int s_begin, int s_end, float* __restrict__ part) {
    static_assert(KS % 32 == 0, "a stage is whole 1 KB units a strip");
    constexpr int U = KS / 32;                 // units a strip a stage
    constexpr int RB = 2 * KS;                 // bytes a route row a stage
    constexpr int RPL = 128 / RB;              // rows a 128-byte line
    constexpr int CR = RB / 16;                // 16-byte chunks a row
    constexpr int NR = NW * TN * 16, MR = TM * 16;  // weight rows, routes
    constexpr int XI = MR * RB / 256 / NW;     // activation reads a wave a stage
    static_assert(XI * NW * 256 == MR * RB, "a stage is whole wave reads");
    constexpr int SB = MR * RB;                // LDS bytes a stage
    constexpr int D = NB - 1;                  // stages read ahead
    constexpr int PER = XI + TN * U;           // reads a thread a stage
    static_assert((D - 1) * PER < 64, "vmcnt counts to 63");
    __shared__ __align__(16) u8 xl[NB][SB];
    const int tid = threadIdx.x, l = lane(), r16 = l % 16, g = l / 16;
    const int wave = tid / kWave;
    const int row0 = tile * NR;
    const int kbytes = 2 * k, stages = s_end - s_begin;
    auto route = [&](int m) { return order != nullptr ? order[begin + m] : begin + m; };
    auto swz = [&](int r) { return (r / RPL) % CR; };
    auto xat = [&](int r, int c) { return r * RB + 16 * (c ^ swz(r)); };

    const moe_rsrc wr = batch_plane(w, 1ull * n * kbytes);
    u32 wo[TN];
#pragma unroll
    for (int nb = 0; nb < TN; ++nb) {
        const int strip = min(row0 / 16 + wave * TN + nb, n / 16 - 1);
        wo[nb] = static_cast<u32>(strip) * static_cast<u32>(16 * kbytes) + s_begin * U * 1024 + 16 * l;
    }
    const u8* xb = reinterpret_cast<const u8*>(act);
    u32 xo[XI];
#pragma unroll
    for (int j = 0; j < XI; ++j) {
        const int q = wave * XI + j;
        const int byte = q * 256 + 4 * l;
        const int r = byte / RB, cl = byte % RB / 16;
        const int c = cl ^ swz(r);
        xo[j] = static_cast<u32>(route(r < count ? r : 0) / act_div) * static_cast<u32>(kbytes) + s_begin * RB + 16 * c +
                4 * (l % 4);
    }
    const u32 uw = static_cast<u32>(__builtin_amdgcn_readfirstlane(wave));
    const u32 xl0 = lds_address(xl[0]) + uw * XI * 256;
    i32x4 wv[NB][TN * U];
    auto issue = [&](int s, int b, i32x4 (&wset)[TN * U]) {
#pragma unroll
        for (int nb = 0; nb < TN; ++nb)
#pragma unroll
            for (int u = 0; u < U; ++u) bload_kept(wset[nb * U + u], wr, wo[nb] + (s * U + u) * 1024);
#pragma unroll
        for (int j = 0; j < XI; ++j) lds_dma4(xb, xo[j] + s * RB, xl0 + b * SB + j * 256);
    };

    mfma_cd acc[TN][TM];
#pragma unroll
    for (int nb = 0; nb < TN; ++nb)
#pragma unroll
        for (int mb = 0; mb < TM; ++mb) acc[nb][mb] = mfma_zero();

    auto step = [&](int s, const u8* xc, i32x4 (&wset)[TN * U]) {
        tie(wset);
#pragma unroll
        for (int kk = 0; kk < KS / 16; ++kk) {
            moe_u2 bv[TM];
#pragma unroll
            for (int mb = 0; mb < TM; ++mb) {
                const int r = mb * 16 + r16;
                bv[mb] = *reinterpret_cast<const moe_u2*>(xc + xat(r, 2 * kk + g / 2) + 8 * (g % 2));
            }
#pragma unroll
            for (int nb = 0; nb < TN; ++nb) {
                const i32x4& wq = wset[nb * U + kk / 2];
                const u32 a0 = static_cast<u32>(wq[2 * (kk % 2)]), a1 = static_cast<u32>(wq[2 * (kk % 2) + 1]);
#pragma unroll
                for (int mb = 0; mb < TM; ++mb)
                    acc[nb][mb] = mma16x16<bf16>(a0, a1, bv[mb][0], bv[mb][1], acc[nb][mb]);
            }
        }
        if (s + D < stages) vm_wait<(D - 1) * PER>(); else vm_wait<0>();
        lds_barrier();
    };
#pragma unroll
    for (int d = 0; d < D; ++d)
        if (d < stages) issue(d, d, wv[d]);
    if (D < stages) vm_wait<(D - 1) * PER>(); else vm_wait<0>();
    lds_barrier();
    for (int s0 = 0; s0 < stages; s0 += NB) {
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int s = s0 + b;
            if (s >= stages) break;
            if (s + D < stages) issue(s + D, (b + D) % NB, wv[(b + D) % NB]);
            step(s, xl[b], wv[b]);
        }
    }

#pragma unroll
    for (int mb = 0; mb < TM; ++mb) {
        const int m = mb * 16 + r16;
        if (m >= count) continue;
        const long long at = static_cast<long long>(route(m)) * n;
        bf16* o = out + at;
#pragma unroll
        for (int nb = 0; nb < TN; ++nb) {
            const int c = row0 + (wave * TN + nb) * 16 + 4 * g;
            if (part != nullptr) {
                // A slice of K: its f32 partial, summed after.
                if (c < n) *reinterpret_cast<mfma_cd*>(part + at + c) = acc[nb][mb];
                continue;
            }
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

// A bf16 bank in the matrix core's order: its routed select past 64 routes
// an expert, one work item of up to 128 routes a block row of 64·NW weight
// rows (grid (work cap, ⌈n / (64·NW)⌉, slices), 64·NW threads). The card
// deals block b to XCD b % 8, each with its own L2: the blocks are
// renumbered so that an XCD runs a contiguous range, every item at one block
// row before the next, so the blocks an XCD holds at once read the one
// weight tile. Slice z of K (`per` stages of 64) of `gridDim.z`; under slices
// each one's f32 partial lands in `part` ([slices, routes, n], by route) for
// `fp8_mfma_finalize`.
template <int NW>
__global__ __launch_bounds__(NW * 64) void moe_bf16_group(const bf16* __restrict__ act, const i32* __restrict__ order,
                                                          const i32* __restrict__ offsets, const i32* __restrict__ work,
                                                          const bf16* __restrict__ bank, bf16* __restrict__ out,
                                                          float* __restrict__ part, int act_div, int n, int k,
                                                          int num_experts, int routes, int per) {
    const int items = offsets[num_experts + 1];
    const int live = items * static_cast<int>(gridDim.y);
    const int b = static_cast<int>(blockIdx.x + gridDim.x * blockIdx.y);
    if (b >= live) return;
    const int xcd = b % 8, slots = live / 8, extra = live % 8;
    const int at = xcd * slots + min(xcd, extra) + b / 8;
    GroupItem item;
    if (!group_item(offsets, work, num_experts, 128, item, at % items)) return;
    const int split = static_cast<int>(blockIdx.z);
    bf16_group_tile<64, NW, 4, 8, 2>(act, order, item.begin, item.count,
                                     bank + static_cast<long long>(item.expert) * n * k, out, act_div, n, k,
                                     at / items, split * per, min((split + 1) * per, k / 64),
                                     gridDim.z > 1 ? part + static_cast<long long>(split) * routes * n : nullptr);
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
template <int MT, int NB>
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
    fp8_tile<MT, kFp8MfmaDepth, NB>(act + first * k, act_scales + first * (k / 128), nullptr, begin, min(MT, m - begin),
                                codes, factors, y + first * n,
                                splits > 1 ? part + static_cast<long long>(split) * rows * n : nullptr, 1, n, k, tile,
                                split * per, (split + 1) * per);
}

// A bf16 bank in the matrix core's order (each expert's plane by
// `e4m3fn_shuffle<false>`): its routed select at decode batches, one work
// item of up to MT routes of one expert against 64·NB weight rows a block
// (grid (work cap, ⌈n / (64·NB)⌉)).
template <int MT, int NB = 1>
__global__ __launch_bounds__(256) void moe_bf16_stream(const bf16* __restrict__ act, const i32* __restrict__ order,
                                                       const i32* __restrict__ offsets, const i32* __restrict__ work,
                                                       const bf16* __restrict__ bank, bf16* __restrict__ out,
                                                       int act_div, int n, int k, int num_experts) {
    // An XCD (block b runs on XCD b % 8) takes a contiguous run of the live
    // blocks, an item's tiles in turn: the item's rows stay in its L2.
    const int items = offsets[num_experts + 1], ny = static_cast<int>(gridDim.y);
    const int live = items * ny;
    const int b = static_cast<int>(blockIdx.x + gridDim.x * blockIdx.y);
    if (b >= live) return;
    const int xcd = b % 8, slots = live / 8, extra = live % 8;
    const int at = xcd * slots + min(xcd, extra) + b / 8;
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item, at / ny)) return;
    fp8_tile<MT, kFp8MfmaDepth, NB, false>(reinterpret_cast<const u8*>(act), nullptr, order, item.begin, item.count,
                                           reinterpret_cast<const u8*>(bank + static_cast<long long>(item.expert) * n * k),
                                           nullptr, out, nullptr, act_div, n, k, at % ny, 0, 2 * k / 128);
}

// The same for one token: route r (blockIdx.x) its own item of one row
// against expert routes[r], unordered (grid (routes, ⌈n / (64·NB)⌉)).
template <int NB = 1>
__global__ __launch_bounds__(256) void moe_bf16_route(const bf16* __restrict__ act, const i32* __restrict__ routes,
                                                      const bf16* __restrict__ bank, bf16* __restrict__ out,
                                                      int act_div, int n, int k, int num_experts) {
    const int r = static_cast<int>(blockIdx.x), e = routes[r];
    if (e < 0 || e >= num_experts) return;
    fp8_tile<16, kFp8MfmaDepth, NB, false>(reinterpret_cast<const u8*>(act), nullptr, nullptr, r, 1,
                                           reinterpret_cast<const u8*>(bank + static_cast<long long>(e) * n * k),
                                           nullptr, out, nullptr, act_div, n, k, blockIdx.y, 0, 2 * k / 128);
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
