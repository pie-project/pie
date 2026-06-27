#pragma once

// Sampling-IR primitive kernel templates (Lane L2 / charlie).
//
// This header carries the **canonical CUDA-C device prelude** that every
// generated sampling-IR kernel is compiled with. It is stored as a single
// NVRTC-safe source string so that:
//
//   1. the codegen (W2) embeds it verbatim ahead of each emitted
//      `extern "C" __global__` entry point, and delta's NVRTC JIT compiles
//      the combined source for sm_89; and
//   2. the standalone primitive test (tests/sampling_ir_primitives_test.cu)
//      compiles the *exact same bytes* through NVRTC + the CUDA driver API,
//      so what we unit-test is byte-for-byte what we ship.
//
// NVRTC constraints honoured by the prelude:
//   * no `#include` of host headers — only device builtins
//     (expf/logf/fmaxf/fminf/fabsf/atomicAdd/__syncthreads/__int_as_float);
//   * only builtin scalar types (int / unsigned int / unsigned long long /
//     float) — no <cstdint>, no std::;
//   * one CUDA block per logical row; a fixed launch width of PIE_IR_BLOCK
//     (256) threads. Block reduction/scan/threshold helpers assume that.
//
// RNG PARITY (coordinated with hotel, transcribed from
// driver/cuda/src/kernels/sample_temp.cu): the Gumbel/uniform helpers
// reproduce the production temp kernel bit-for-bit so the IR path matches
// `launch_sample_temp_bf16`. Do not "improve" the constants.

namespace pie_cuda_driver::sampling_ir {

// Canonical NVRTC-safe device prelude. Exposed via primitive_prelude().
inline constexpr char kPrimitivePrelude[] = R"PIECUDA(
// ============================================================================
// Sampling-IR primitive device prelude (NVRTC-safe). DO NOT add #includes.
// ============================================================================
#define PIE_IR_BLOCK 256

__device__ __forceinline__ float pie_ir_pos_inf() { return __int_as_float(0x7f800000); }
__device__ __forceinline__ float pie_ir_neg_inf() { return __int_as_float(0xff800000); }

// ---------------------------------------------------------------------------
// RNG — SplitMix64 per-(seed,column) noise. Bit-parity with sample_temp.cu.
// ---------------------------------------------------------------------------

// Per-row seed preparation: the production temp kernel XORs the u32 wire seed
// with 0xA5A5A5A5 before mixing. Codegen/executor must feed seed_eff (NOT the
// raw u32) into the hash/gumbel helpers.
__device__ __forceinline__ unsigned long long pie_ir_seed_eff(unsigned int seed_u32) {
    return (unsigned long long)seed_u32 ^ 0xA5A5A5A5ULL;
}

__device__ __forceinline__ unsigned long long pie_ir_splitmix64(unsigned long long x) {
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    return x;
}

// RNG model (B), frozen: a draw is Rng{stream:u32, kind} over a per-fire AMBIENT
// seed S (= legacy per-row sample_seed, executor-supplied) — there is no seed
// operand. `stream` is a static stream-id that decorrelates independent draws by
// construction (e.g. lossless: accept = stream 0, resample = stream 1).
//
// Stream salt mixes the stream-id into S. CONTRACT (req-1, parity-locked):
//   salt(0) == 0  ->  stream 0 reproduces today's seed_eff = S ^ 0xA5A5A5A5
//   EXACTLY, so single-RNG programs are bit-parity with sample_temp / goldens
//   (the rng-model change alters NO token stream). salt(0)=0 falls out branch-
//   free: (u64)0 * golden = 0, and splitmix64(0) = 0.
__device__ __forceinline__ unsigned long long pie_ir_stream_salt(unsigned int stream) {
    return pie_ir_splitmix64((unsigned long long)stream * 0x9E3779B97F4A7C15ULL);
}

// seed_eff for RNG `stream` off the ambient seed S. stream 0 == pie_ir_seed_eff(S).
// Hotel's eval mixes the stream-id with this identical formula (req-1 co-owner).
__device__ __forceinline__ unsigned long long pie_ir_seed_eff_stream(unsigned int seed_u32,
                                                                     unsigned int stream) {
    return pie_ir_seed_eff(seed_u32) ^ pie_ir_stream_salt(stream);
}

// bf16 → f32. A bf16 value is exactly the high 16 bits of the f32, so the
// conversion is a zero-extend of the mantissa: shift left 16 and reinterpret.
// Bit-identical to __bfloat162float, but needs no cuda_bf16.h (NVRTC-safe).
// Codegen reads the intrinsic logits (passed as raw `const unsigned short*`,
// i.e. ws.logits reinterpreted) through this helper.
__device__ __forceinline__ float pie_ir_bf16_to_f32(unsigned short h) {
    return __uint_as_float((unsigned int)h << 16);
}

// Uniform in (0, 1). `seed_eff` = pie_ir_seed_eff(wire_seed); `j` = absolute
// vocab column index. High 24 bits, offset by 0.5 to avoid exact 0/1.
__device__ __forceinline__ float pie_ir_hash_uniform(unsigned long long seed_eff, int j) {
    unsigned long long x = seed_eff + 0x9E3779B97F4A7C15ULL * (unsigned long long)(j + 1);
    x = pie_ir_splitmix64(x);
    unsigned int bits = (unsigned int)(x >> 40);
    return ((float)bits + 0.5f) * (1.0f / 16777216.0f);
}

// Gumbel noise g_j = -log(-log(u)). Add to (logit / T) for Gumbel-max sampling.
__device__ __forceinline__ float pie_ir_gumbel(unsigned long long seed_eff, int j) {
    float u = pie_ir_hash_uniform(seed_eff, j);
    return -logf(-logf(u));
}

// ---------------------------------------------------------------------------
// Elementwise map ops (emitted inline by codegen; provided as helpers so the
// non-operator ones have one definition and the test can exercise them).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float pie_ir_exp(float x)   { return expf(x); }
__device__ __forceinline__ float pie_ir_log(float x)   { return logf(x); }
__device__ __forceinline__ float pie_ir_neg(float x)   { return -x; }
__device__ __forceinline__ float pie_ir_recip(float x) { return 1.0f / x; }
__device__ __forceinline__ float pie_ir_abs(float x)   { return fabsf(x); }
__device__ __forceinline__ float pie_ir_sign(float x)  { return (x > 0.0f) - (x < 0.0f); }

__device__ __forceinline__ float pie_ir_add(float a, float b) { return a + b; }
__device__ __forceinline__ float pie_ir_sub(float a, float b) { return a - b; }
__device__ __forceinline__ float pie_ir_mul(float a, float b) { return a * b; }
__device__ __forceinline__ float pie_ir_div(float a, float b) { return a / b; }
__device__ __forceinline__ float pie_ir_max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float pie_ir_min(float a, float b) { return fminf(a, b); }

__device__ __forceinline__ unsigned char pie_ir_gt(float a, float b) { return a >  b ? 1u : 0u; }
__device__ __forceinline__ unsigned char pie_ir_ge(float a, float b) { return a >= b ? 1u : 0u; }
__device__ __forceinline__ unsigned char pie_ir_eq(float a, float b) { return a == b ? 1u : 0u; }

// select(cond, a, b): cond!=0 -> a else b.
__device__ __forceinline__ float pie_ir_select(unsigned char cond, float a, float b) {
    return cond ? a : b;
}

// ---------------------------------------------------------------------------
// Block reduction of a PER-THREAD partial (for fused map→reduce: the caller
// computes each element's value inline in registers and accumulates a local
// partial, then reduces it here — no materialized input buffer). Every thread
// receives the result; trailing __syncthreads() keeps scratch reusable.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float pie_ir_block_sum_reduce(float local) {
    __shared__ float sbuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    sbuf[tid] = local; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sbuf[tid] += sbuf[tid + off];
        __syncthreads();
    }
    float r = sbuf[0]; __syncthreads();
    return r;
}
__device__ __forceinline__ float pie_ir_block_max_reduce(float local) {
    __shared__ float sbuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    sbuf[tid] = local; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sbuf[tid] = fmaxf(sbuf[tid], sbuf[tid + off]);
        __syncthreads();
    }
    float r = sbuf[0]; __syncthreads();
    return r;
}
__device__ __forceinline__ float pie_ir_block_min_reduce(float local) {
    __shared__ float sbuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    sbuf[tid] = local; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sbuf[tid] = fminf(sbuf[tid], sbuf[tid + off]);
        __syncthreads();
    }
    float r = sbuf[0]; __syncthreads();
    return r;
}
// Argmax of per-thread (value, index) partials; lowest index wins ties.
__device__ __forceinline__ int pie_ir_block_argmax_reduce(float local_best, int local_idx) {
    __shared__ float vbuf[PIE_IR_BLOCK];
    __shared__ int   ibuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    vbuf[tid] = local_best; ibuf[tid] = local_idx; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float ov = vbuf[tid + off];
            int   oi = ibuf[tid + off];
            if (ov > vbuf[tid] || (ov == vbuf[tid] && oi < ibuf[tid])) {
                vbuf[tid] = ov; ibuf[tid] = oi;
            }
        }
        __syncthreads();
    }
    int r = ibuf[0]; __syncthreads();
    return r;
}

// ---------------------------------------------------------------------------
// Block reductions (one block / row, PIE_IR_BLOCK threads). Every thread
// receives the result; helpers leave their shared scratch reusable (trailing
// __syncthreads()).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float pie_ir_block_sum(const float* __restrict__ row, int n) {
    __shared__ float sbuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int j = tid; j < n; j += PIE_IR_BLOCK) local += row[j];
    sbuf[tid] = local; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sbuf[tid] += sbuf[tid + off];
        __syncthreads();
    }
    float r = sbuf[0]; __syncthreads();
    return r;
}

__device__ __forceinline__ float pie_ir_block_max(const float* __restrict__ row, int n) {
    __shared__ float sbuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    float local = pie_ir_neg_inf();
    for (int j = tid; j < n; j += PIE_IR_BLOCK) local = fmaxf(local, row[j]);
    sbuf[tid] = local; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sbuf[tid] = fmaxf(sbuf[tid], sbuf[tid + off]);
        __syncthreads();
    }
    float r = sbuf[0]; __syncthreads();
    return r;
}

__device__ __forceinline__ float pie_ir_block_min(const float* __restrict__ row, int n) {
    __shared__ float sbuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    float local = pie_ir_pos_inf();
    for (int j = tid; j < n; j += PIE_IR_BLOCK) local = fminf(local, row[j]);
    sbuf[tid] = local; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sbuf[tid] = fminf(sbuf[tid], sbuf[tid + off]);
        __syncthreads();
    }
    float r = sbuf[0]; __syncthreads();
    return r;
}

// Argmax over the row; lowest index wins ties (matches sample_temp/argmax).
__device__ __forceinline__ int pie_ir_block_argmax(const float* __restrict__ row, int n) {
    __shared__ float vbuf[PIE_IR_BLOCK];
    __shared__ int   ibuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    float best = pie_ir_neg_inf();
    int   bidx = 0x7fffffff;
    for (int j = tid; j < n; j += PIE_IR_BLOCK) {
        float v = row[j];
        if (v > best || (v == best && j < bidx)) { best = v; bidx = j; }
    }
    vbuf[tid] = best; ibuf[tid] = bidx; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float ov = vbuf[tid + off];
            int   oi = ibuf[tid + off];
            if (ov > vbuf[tid] || (ov == vbuf[tid] && oi < ibuf[tid])) {
                vbuf[tid] = ov; ibuf[tid] = oi;
            }
        }
        __syncthreads();
    }
    int r = ibuf[0]; __syncthreads();
    return r;
}

// Count of row[j] >= thr across the block.
__device__ __forceinline__ int pie_ir_block_count_ge(const float* __restrict__ row, int n, float thr) {
    __shared__ int sbuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    int c = 0;
    for (int j = tid; j < n; j += PIE_IR_BLOCK) if (row[j] >= thr) ++c;
    sbuf[tid] = c; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sbuf[tid] += sbuf[tid + off];
        __syncthreads();
    }
    int r = sbuf[0]; __syncthreads();
    return r;
}

// Sum of row[j] over the j with row[j] >= thr (mass above threshold).
__device__ __forceinline__ float pie_ir_block_mass_ge(const float* __restrict__ row, int n, float thr) {
    __shared__ float sbuf[PIE_IR_BLOCK];
    int tid = threadIdx.x;
    float m = 0.0f;
    for (int j = tid; j < n; j += PIE_IR_BLOCK) if (row[j] >= thr) m += row[j];
    sbuf[tid] = m; __syncthreads();
    for (int off = PIE_IR_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sbuf[tid] += sbuf[tid + off];
        __syncthreads();
    }
    float r = sbuf[0]; __syncthreads();
    return r;
}

// ---------------------------------------------------------------------------
// Scan — block-wide inclusive scan over a row (Hillis-Steele per tile + carry).
// kind: 0 = cumsum, 1 = cumprod. All threads participate.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pie_ir_block_inclusive_scan(
        const float* __restrict__ in, float* __restrict__ out, int n, int is_prod) {
    __shared__ float tile[PIE_IR_BLOCK];
    __shared__ float carry;
    int tid = threadIdx.x;
    float identity = is_prod ? 1.0f : 0.0f;
    if (tid == 0) carry = identity;
    __syncthreads();
    for (int base = 0; base < n; base += PIE_IR_BLOCK) {
        int idx = base + tid;
        tile[tid] = (idx < n) ? in[idx] : identity;
        __syncthreads();
        for (int off = 1; off < PIE_IR_BLOCK; off <<= 1) {
            float t = (tid >= off) ? tile[tid - off] : identity;
            __syncthreads();
            tile[tid] = is_prod ? (tile[tid] * t) : (tile[tid] + t);
            __syncthreads();
        }
        float c = carry;
        float res = is_prod ? (tile[tid] * c) : (tile[tid] + c);
        if (idx < n) out[idx] = res;
        __syncthreads();
        if (tid == PIE_IR_BLOCK - 1) carry = is_prod ? (tile[tid] * c) : (tile[tid] + c);
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Iterative-pivot threshold (sort-free). Each returns a threshold τ; the mask
// is (value >= τ). 50 bisection iters → deterministic, parity-friendly.
// ---------------------------------------------------------------------------

// top-k: τ = k-th largest value of `val`. mask (val >= τ) keeps >= k tokens
// (ties at the boundary are kept). k<=0 -> keep none; k>=n -> keep all.
__device__ __forceinline__ float pie_ir_pivot_topk(const float* __restrict__ val, int n, int k) {
    if (k <= 0) {
        float hi = pie_ir_block_max(val, n);
        return hi + fmaxf(fabsf(hi) * 1e-4f, 1e-4f);   // τ above max -> mask empty
    }
    if (k >= n) return pie_ir_block_min(val, n);        // τ at min -> mask full
    float lo = pie_ir_block_min(val, n);
    float hi = pie_ir_block_max(val, n);
    // Invariant: count_ge(lo) >= k, count_ge(hi) < k. Nudge hi strictly above max.
    hi = hi + fmaxf(fabsf(hi) * 1e-4f, 1e-4f);
    for (int it = 0; it < 50; ++it) {
        float mid = 0.5f * (lo + hi);
        int cnt = pie_ir_block_count_ge(val, n, mid);
        if (cnt >= k) lo = mid; else hi = mid;
    }
    return lo;
}

// Order-preserving float<->uint32 mapping: larger float -> larger key. Lets a
// radix select on the bit pattern recover the k-th largest value exactly.
__device__ __forceinline__ unsigned int pie_ir_f2ord(float f) {
    unsigned int b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float pie_ir_ord2f(unsigned int u) {
    unsigned int b = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    return __uint_as_float(b);
}

// top-k via MSB-first radix select (4 × 8-bit passes) — finds the exact k-th
// largest value with 4 full-row histogram scans instead of the bisection's ~50
// count scans. Same semantics as pie_ir_pivot_topk: returns τ = k-th largest;
// mask (val >= τ) keeps >= k tokens (boundary ties kept). One block per row.
__device__ __forceinline__ float pie_ir_pivot_topk_radix(const float* __restrict__ val, int n, int k) {
    if (k <= 0) {
        float hi = pie_ir_block_max(val, n);
        return hi + fmaxf(fabsf(hi) * 1e-4f, 1e-4f);
    }
    if (k >= n) return pie_ir_block_min(val, n);

    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix;
    __shared__ int s_need;
    int tid = threadIdx.x;
    if (tid == 0) { s_prefix = 0u; s_need = k; }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - 8 * pass;
        unsigned int hi_mask = (shift + 8 >= 32) ? 0u : (0xFFFFFFFFu << (shift + 8));
        unsigned int prefix = s_prefix;
        for (int i = tid; i < 256; i += PIE_IR_BLOCK) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < n; i += PIE_IR_BLOCK) {
            unsigned int key = pie_ir_f2ord(val[i]);
            if ((key & hi_mask) == (prefix & hi_mask))
                atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        // Single thread walks buckets high→low to find the one holding the k-th.
        if (tid == 0) {
            int need = s_need;
            int acc = 0;
            int digit = 0;
            for (int d = 255; d >= 0; --d) {
                if (acc + (int)hist[d] >= need) { digit = d; break; }
                acc += (int)hist[d];
            }
            s_prefix = prefix | ((unsigned int)digit << shift);
            s_need = need - acc;
        }
        __syncthreads();
    }
    float r = pie_ir_ord2f(s_prefix);
    __syncthreads();
    return r;
}

// top-p (nucleus): keep the smallest set of highest-`prob` tokens whose mass
// >= p. τ = largest threshold with mass(prob >= τ) >= p. `prob` must be a
// normalized distribution. p<=0 -> keep top-1 region; p>=1 -> keep all.
__device__ __forceinline__ float pie_ir_pivot_topp(const float* __restrict__ prob, int n, float p) {
    float maxp = pie_ir_block_max(prob, n);
    if (p >= 1.0f) return pie_ir_block_min(prob, n);
    float lo = 0.0f;                                    // mass(>=0) = 1 >= p
    float hi = maxp + fmaxf(fabsf(maxp) * 1e-4f, 1e-6f);// mass(>=hi) = 0 < p
    for (int it = 0; it < 50; ++it) {
        float mid = 0.5f * (lo + hi);
        float mass = pie_ir_block_mass_ge(prob, n, mid);
        if (mass >= p) lo = mid; else hi = mid;
    }
    return lo;
}

// top-p (nucleus) via MSB-first radix select on prob MASS (4 × 8-bit passes) —
// finds τ = the nucleus boundary prob (largest threshold with mass(prob>=τ) >= p)
// with 4 full-row mass-histogram scans instead of pie_ir_pivot_topp's ~50
// bisection mass scans. Mass analogue of pie_ir_pivot_topk_radix (count → mass).
// Same semantics as pie_ir_pivot_topp; `prob` must be a normalized distribution.
// One block per row.
//
// PARITY NOTE: per-bucket mass is a float atomicAdd (order-non-deterministic), so
// the boundary token can shift by a ULP-level mass crossing vs the bisection.
// Gated vs the CPU golden (top-p token-stream is behavior-change-flagged anyway).
__device__ __forceinline__ float pie_ir_pivot_topp_radix(const float* __restrict__ prob, int n, float p) {
    if (p >= 1.0f) return pie_ir_block_min(prob, n);            // keep all
    if (p <= 0.0f) return pie_ir_block_max(prob, n);            // keep top-1
    if (p >= pie_ir_block_sum(prob, n)) return pie_ir_block_min(prob, n);  // unreachable mass -> keep all

    __shared__ float massHist[256];
    __shared__ unsigned int s_prefix;
    __shared__ float s_need;                 // remaining mass to reach p
    int tid = threadIdx.x;
    if (tid == 0) { s_prefix = 0u; s_need = p; }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - 8 * pass;
        unsigned int hi_mask = (shift + 8 >= 32) ? 0u : (0xFFFFFFFFu << (shift + 8));
        unsigned int prefix = s_prefix;
        for (int i = tid; i < 256; i += PIE_IR_BLOCK) massHist[i] = 0.0f;
        __syncthreads();
        for (int i = tid; i < n; i += PIE_IR_BLOCK) {
            float pv = prob[i];
            unsigned int key = pie_ir_f2ord(pv);
            if ((key & hi_mask) == (prefix & hi_mask))
                atomicAdd(&massHist[(key >> shift) & 0xFFu], pv);
        }
        __syncthreads();
        // Single thread walks buckets high→low to the one holding the cumulative-
        // mass crossing of `need` (guaranteed non-empty: the total-mass guard above
        // ensures the crossing exists, so we never recurse into an empty tail).
        if (tid == 0) {
            float need = s_need;
            float acc = 0.0f;
            int digit = 0;
            for (int d = 255; d >= 0; --d) {
                if (acc + massHist[d] >= need) { digit = d; break; }
                acc += massHist[d];
            }
            s_prefix = prefix | ((unsigned int)digit << shift);
            s_need = need - acc;
        }
        __syncthreads();
    }
    float r = pie_ir_ord2f(s_prefix);
    __syncthreads();
    return r;
}
// the threshold scales with the row max either way.
//
// PARITY NOTE (hotel): for the temp+min-p program that must bit-match
// sample_temp.cu, codegen emits the *logit-space* form instead —
// keep = Ge(logits, max_logit + Log(min_p)) — composed from pie_ir_block_max +
// pie_ir_log + pie_ir_ge. Algebraically identical but avoids exp/log rounding
// flipping the boundary token. This prob-space helper is for standalone/generic
// min-p (parity target = golden, not the kernel), where either form is fine.
__device__ __forceinline__ float pie_ir_pivot_minp(const float* __restrict__ prob, int n, float min_p) {
    return min_p * pie_ir_block_max(prob, n);
}

// Write a bool/byte mask for (val >= thr) over a row. Elementwise.
__device__ __forceinline__ void pie_ir_write_ge_mask(
        const float* __restrict__ val, unsigned char* __restrict__ mask, int n, float thr) {
    for (int j = threadIdx.x; j < n; j += PIE_IR_BLOCK) mask[j] = (val[j] >= thr) ? 1u : 0u;
}

// ---------------------------------------------------------------------------
// Sort (descending) — shared-memory bitonic over one row. Produces sorted
// values + their original indices. n must be <= PIE_IR_SORT_MAX; the array is
// padded to the next power of two with -inf (index -1). Larger rows are a W2
// concern (top-k/top-p are sort-free above and do not need this).
// ---------------------------------------------------------------------------
#define PIE_IR_SORT_MAX 1024

__device__ __forceinline__ void pie_ir_block_sort_desc(
        const float* __restrict__ in, float* __restrict__ out_val,
        int* __restrict__ out_idx, int n) {
    __shared__ float sval[PIE_IR_SORT_MAX];
    __shared__ int   sidx[PIE_IR_SORT_MAX];
    int tid = threadIdx.x;

    int np = 1;
    while (np < n) np <<= 1;                 // next pow2 >= n

    for (int i = tid; i < np; i += PIE_IR_BLOCK) {
        if (i < n) { sval[i] = in[i]; sidx[i] = i; }
        else       { sval[i] = pie_ir_neg_inf(); sidx[i] = -1; }
    }
    __syncthreads();

    for (int ksz = 2; ksz <= np; ksz <<= 1) {
        for (int j = ksz >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < np; i += PIE_IR_BLOCK) {
                int ixj = i ^ j;
                if (ixj > i) {
                    // descending: larger values toward lower indices
                    bool ascending = ((i & ksz) != 0);
                    float a = sval[i],   b = sval[ixj];
                    int   ai = sidx[i],  bi = sidx[ixj];
                    bool swap = ascending ? (a > b) : (a < b);
                    if (a == b) { /* stable-ish: keep order, no swap */ }
                    if (swap) {
                        sval[i] = b; sval[ixj] = a;
                        sidx[i] = bi; sidx[ixj] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < n; i += PIE_IR_BLOCK) { out_val[i] = sval[i]; out_idx[i] = sidx[i]; }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Indexing — gather / scatter / gather-row. Grid-stride over the output.
//
// SENTINEL POLICY (frozen by manager): the *semantic* contract is PRE-MASK —
// programs must pass in-range, non-negative indices; sentinels are masked in
// the IR graph (e.g. select(valid, idx, 0)). The bounds checks below are a
// memory-safety BACKSTOP ONLY against malformed inferlet bytecode — never a
// semantic programs may rely on: scatter drops OOB/negative writes, gather
// fills 0. `src_len` / `base_len` / `nrows` carry the valid range.
// ---------------------------------------------------------------------------

// dst[i] = src[idx[i]] for i in [0, n); invalid idx -> 0 (safety fill).
__device__ __forceinline__ void pie_ir_gather(
        const float* __restrict__ src, int src_len, const int* __restrict__ idx,
        float* __restrict__ dst, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = g; i < n; i += stride) {
        int s = idx[i];
        dst[i] = (s >= 0 && s < src_len) ? src[s] : 0.0f;
    }
}

// dst[c] = src[row * ncols + c] for c in [0, ncols). The reduce-result-used-
// as-index barrier consumer (e.g. j = reduce-sum(...); gather-row(resid, j)).
// Invalid `row` (<0 or >= nrows) -> dst filled with 0 (safety fill).
__device__ __forceinline__ void pie_ir_gather_row(
        const float* __restrict__ src, int nrows, int row, int ncols,
        float* __restrict__ dst) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    bool valid = (row >= 0 && row < nrows);
    const float* srow = valid ? (src + (long long)row * ncols) : (const float*)0;
    for (int c = g; c < ncols; c += stride) dst[c] = valid ? srow[c] : 0.0f;
}

// base[idx[i]] += vals[i] (atomic; duplicate indices accumulate). OOB/neg idx
// dropped (safety backstop).
__device__ __forceinline__ void pie_ir_scatter_add(
        float* __restrict__ base, int base_len, const int* __restrict__ idx,
        const float* __restrict__ vals, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = g; i < n; i += stride) {
        int b = idx[i];
        if (b >= 0 && b < base_len) atomicAdd(&base[b], vals[i]);
    }
}

// base[idx[i]] = vals[i]. (Duplicate indices: last writer wins, unspecified.)
// OOB/neg idx dropped (safety backstop).
__device__ __forceinline__ void pie_ir_scatter_set(
        float* __restrict__ base, int base_len, const int* __restrict__ idx,
        const float* __restrict__ vals, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = g; i < n; i += stride) {
        int b = idx[i];
        if (b >= 0 && b < base_len) base[b] = vals[i];
    }
}
)PIECUDA";

// Accessor used by the codegen (W2) and the JIT (delta) to obtain the prelude.
inline const char* primitive_prelude() { return kPrimitivePrelude; }

}  // namespace pie_cuda_driver::sampling_ir
