//===-- topk_softmax.cuh - the softmax and sigmoid routers ----------------===//
//
// Nine `__global__` templates, two `__device__` bodies and the block-wide
// argmax they share. No launcher: `topk_softmax.cu` includes this file and
// keeps the `<<<>>>`s, the environment variable and the exceptions, so every
// kernel here has exactly ONE definition that nvcc and NVRTC both read.
//
// # Five kernels became nine entry points, and no body was copied
//
// The `.cu` held five `__global__`s and fired ten launches. The extra five
// are one kernel's run-time ladder: `topk_softmax_warp` keeps the experts in
// registers, `PER_LANE` of them per lane, and the launcher picks 1, 2, 4, 8
// or 16 from `num_experts` — five instantiations of one template, chosen on
// the host.
//
// A row names ONE instantiation and supplies exactly ONE template argument,
// the element type. So a `PER_LANE` that the row cannot state is a form the
// JIT cannot reach. The ladder is preserved by factoring the maths into
// `topk_softmax_warp_body<T, PerLane>` — a `__device__` function, defined
// once — and giving each rung a thin `__global__` of its own. Five entry
// points, one body; the alternative was five copies of a 90-line kernel,
// which is the drift `new-horizon.md` §10.10 names.
//
// The same trick retires the `FUSED_GEMV` bool: `topk_softmax` and
// `router_topk_softmax` are the two forms the launchers already fired, now
// spelled as names rather than as a template argument nothing could state.
//
// # Two kernels became one
//
// `topk_sigmoid_bias_bf16_kernel` and `topk_sigmoid_bias_fp32_kernel` were
// the same 40 lines twice, differing in one load. They are now one template
// over `T`. That is the whole point of the migration: the ahead-of-time build
// pays a translation unit per instantiation and so grew a second copy rather
// than a second row, and the two had already drifted in their comments.
//
// # Why this family names its own `f32`
//
// The prelude specialises `Elem` for `bf16` and `f16` and names no
// fp32 element type — nothing in `norm` needed one. The fp32 router does:
// some checkpoints keep the router in fp32 while everything around it is
// bf16, precisely because a 256-way argmax over bf16 logits has ties that
// fp32 does not. `f32` and `Logit` are therefore declared HERE, in this
// family's namespace, and the rows say `pie::moe::f32`. A `f32`
// and an `Elem<float>` belong in the prelude, which is not this family's
// file — that is a change to report, not to make.
//
// # One row out of nine, and the block width is the whole reason
//
// Only `apply_per_expert_scale` is rowed. Its launcher is `ceil(N*K / 256)`
// blocks of 256, which is `LaunchRule::Elementwise` over a `[N, K]`
// rectangle exactly — same block, same count, same rounding, and `total`
// comes off `Source::OutElements`.
//
// The other eight are carried as device text and left unrowed, because
// every one of them has a block width that is part of the algorithm rather
// than a tiling choice:
//
//   * the block forms — `topk_softmax`, `router_topk_softmax`,
//     `topk_sigmoid_bias` — fire 64 threads, and `block_argmax`
//     `static_assert`s on `kSoftmaxBlock` because its reduction is a
//     fixed-depth tree over exactly two warps. `Rms` would hand it 256.
//   * the five warp rungs fire ONE warp and reduce with `__shfl_xor_sync`
//     over 32 lanes. A second warp does not make them wrong — it makes them
//     compute the same answer twice and race on the store.
//
// `topk_sigmoid` in `topk_sigmoid.cuh` shows what it would take: its
// staging loop was changed to step by `blockDim.x`, which made every block
// width correct and `LaunchRule::Rms` exact. The same edit here would mean
// rewriting a warp reduction into a block reduction — a different kernel,
// with its own parity evidence to produce, and not a row's business.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::moe {

// The scalar layer is the PRELUDE's, named here so `i32` keeps its
// meaning inside `kernels::moe` once this nested namespace shadows it.

/// The fp32 element type, named because the prelude does not name one.
///
/// A row's element is a path under `kernels`, so an fp32 row has to have
/// something to point at. See this file's header for why it is here and not
/// in `pie_device.cuh`.
using f32 = float;

/// How a router reads one logit as a float.
///
/// `Elem` would do this and is specialised for `bf16` and `f16` only;
/// an fp32 router loads a float and converts nothing, and adding that
/// specialisation to the prelude is not this family's edit to make. Two
/// lines here, and the fp32 and bf16 routers stay one kernel.
template <class T>
struct Logit {
    static __device__ __forceinline__ float to_f32(T v) { return Elem<T>::to_f32(v); }
};

template <>
struct Logit<f32> {
    static __device__ __forceinline__ float to_f32(f32 v) { return v; }
};

/// Threads per block for every block-form router here, and the stride their
/// staging loops step by. `block_argmax` static_asserts on it.
constexpr int kSoftmaxBlock = 64;

/// The widest router the static shared slabs hold.
///
/// Qwen3.6-35B-A3B routes through 256 experts; Kimi K2.6 through 384. 512
/// floats is 2 KB and covers both. The launchers THROW above it rather than
/// clamping — a clamp is a routing change nothing reports.
///
/// `[[maybe_unused]]` because a NVRTC unit instantiates only the templates
/// its rows name, and every template that reads this bound is a block-form
/// router with no row — so to the front end the constant is "declared but
/// never referenced". Saying so per symbol beats a `--diag-suppress` that
/// would also hide the warning about a constant nothing reads at all.
[[maybe_unused]] constexpr int kSoftmaxMaxExperts = 512;

/// Block-wide argmax over `scores[0..num_experts)`, ties resolved to the
/// LOWEST index — the same winner a serial `for (j) if (s[j] > best)` scan
/// picks, so routing decisions stay bit-identical.
///
/// The serial form cost K * num_experts iterations on thread 0 while the
/// other 63 threads idled: at 256 experts and K = 8 that is 2048 dependent
/// shared-memory reads, which measured 21 us per layer (7% of a Qwen3.6
/// decode step). Strided scan plus a log-depth reduction is ~10 steps.
__device__ inline void block_argmax(
    const float* __restrict__ scores,
    int num_experts,
    float floor_value,
    float* __restrict__ value_buf,
    int* __restrict__ index_buf,
    float& best_value,
    int& best_index)
{
    const int tid = threadIdx.x;
    // Strictly above `floor_value`, matching the serial scan's seed: an
    // already-picked expert is excluded by writing the floor back into
    // `scores`, and the floor itself must never win.
    float local_v = floor_value;
    int local_i = -1;
    for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
        const float v = scores[j];
        if (v > local_v) {
            local_v = v;
            local_i = j;
        }
    }
    value_buf[tid] = local_v;
    index_buf[tid] = local_i;
    __syncthreads();
    // A shared-memory tree over all kSoftmaxBlock lanes costs log2(BLOCK)
    // __syncthreads PER ROUND, and there are K rounds. Fold the upper
    // warp once, then finish inside warp 0 with shuffles, which need no
    // barrier at all: 2 barriers per round instead of 8.
    static_assert(kSoftmaxBlock == 64, "block_argmax folds exactly one upper warp");
    if (tid < 32) {
        float v = value_buf[tid];
        int i = index_buf[tid];
        // A strided scan gives thread t the indices t, t+BLOCK, ..., so the
        // lower index of a tie is not always in the lower lane: compare
        // indices explicitly rather than relying on lane order. This keeps
        // the winner identical to a serial `if (s[j] > best)` scan.
        auto take = [](float& v, int& i, float ov, int oi) {
            if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) {
                v = ov;
                i = oi;
            }
        };
        take(v, i, value_buf[tid + 32], index_buf[tid + 32]);
        for (int off = 16; off > 0; off >>= 1) {
            take(v, i,
                 __shfl_down_sync(0xffffffffu, v, off),
                 __shfl_down_sync(0xffffffffu, i, off));
        }
        if (tid == 0) {
            value_buf[0] = v;
            index_buf[0] = i;
        }
    }
    __syncthreads();
    best_value = value_buf[0];
    best_index = index_buf[0];
    // No trailing barrier: every caller syncs after acting on the winner,
    // which is what orders the next round's `value_buf` writes.
}

/// One block per token. Phase 1: thread-local max-reduce + exp+sum-reduce
/// for softmax. Phase 2: K iterations of argmax-with-exclusion to pick the
/// top-K probs. Phase 3: thread 0 renormalizes and writes back.
///
/// `FusedGemv` computes the router logits here instead of reading them: one
/// warp per expert, walking that expert's weight row. The router is a
/// [num_experts, hidden] projection of ONE token, so as a standalone GEMV it
/// is 32 blocks on 132 SMs and costs what a launch costs whatever it does --
/// the tuner measured 5.5 us for 0.18 MB. Folding it into the consumer that
/// was going to read its output anyway removes the launch and the round trip
/// through HBM, and the logits never leave shared memory.
///
/// A `__device__` body rather than a `__global__` template because a row
/// states one template argument and this needs two; the two entry points
/// below are what the rows name.
template <class T, bool FusedGemv>
__device__ __forceinline__ void topk_softmax_body(
    const T* __restrict__ logits,   // FusedGemv: the weight
    const T* __restrict__ act,      // FusedGemv only
    const T* __restrict__ bias,     // FusedGemv only, may be null
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row =
        FusedGemv ? logits
                  : logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[kSoftmaxMaxExperts];
    __shared__ float buf[kSoftmaxBlock];
    __shared__ int ibuf[kSoftmaxBlock];

    // 1. Stage row into shared memory + find max.
    float local_max = -flt_max();
    if constexpr (FusedGemv) {
        const T* x = act + static_cast<long long>(n) * hidden;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        constexpr int kWarps = kSoftmaxBlock / 32;
        for (int e = warp; e < num_experts; e += kWarps) {
            const T* w = row + static_cast<long long>(e) * hidden;
            float acc = 0.f;
            for (int i = lane; i < hidden; i += 32) {
                acc += Logit<T>::to_f32(w[i]) * Logit<T>::to_f32(x[i]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) {
                if (bias != nullptr) acc += Logit<T>::to_f32(bias[e]);
                probs[e] = acc;
            }
        }
        __syncthreads();
        for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
            if (probs[j] > local_max) local_max = probs[j];
        }
    } else {
    for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
        const float v = Logit<T>::to_f32(row[j]);
        probs[j] = v;
        if (v > local_max) local_max = v;
    }
    }
    buf[tid] = local_max;
    __syncthreads();
    for (int off = kSoftmaxBlock / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] = fmaxf(buf[tid], buf[tid + off]);
        __syncthreads();
    }
    const float row_max = buf[0];
    __syncthreads();

    // 2. exp + sum.
    float local_sum = 0.f;
    for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
        const float e = expf(probs[j] - row_max);
        probs[j] = e;
        local_sum += e;
    }
    buf[tid] = local_sum;
    __syncthreads();
    for (int off = kSoftmaxBlock / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_Z = 1.f / buf[0];
    __syncthreads();

    // 3. Normalize in shared mem, then K block-wide argmaxes with exclusion.
    for (int j = tid; j < num_experts; j += kSoftmaxBlock) probs[j] *= inv_Z;
    __syncthreads();

    i32* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float w_sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -1.f;
        int best_i = -1;
        block_argmax(probs, num_experts, -1.f, buf, ibuf, best_v, best_i);
        // **A SLOT WITH NO EXPERT LEFT WEIGHS NOTHING**, and taking `best_v`
        // here took the FLOOR. `block_argmax` seeds `best_value` at the floor
        // and wins only strictly above it, so once every expert has been
        // excluded it answers `(-1.f, -1)` — and `-1.f` went into `out_w` and
        // into the normaliser.
        //
        // At `K == num_experts + 1` that is exactly fatal: the real weights are
        // normalised probabilities summing to 1, so `w_sum` is `1 - 1 == 0`,
        // `inv_w` is `+inf`, and EVERY weight in the row — including the real
        // ones — comes out infinite. One more spare slot and they come out
        // negated instead. Nothing faults either way.
        //
        // The other two routers in this file already answer zero here: the
        // warp rungs by construction, since `expf(-flt_max() - row_max)` is
        // `0`, and `topk_sqrt_softplus_body` by writing `best_i >= 0 ?
        // probs[best_i] : 0.f` in as many words. This one was the odd one out.
        const float w = best_i >= 0 ? best_v : 0.f;
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = w;
            if (best_i >= 0) probs[best_i] = -1.f;  // exclude on next pass
        }
        w_sum += w;
        __syncthreads();
    }
    if (tid == 0) {
        const float inv_w = 1.f / w_sum;
        for (int k = 0; k < K; ++k) out_w[k] *= inv_w;
    }
}

/// Top-K softmax over router logits already in memory.
///
/// `act`, `bias` and `hidden` are the fused form's and unread here; the
/// launcher passes nulls and a zero, which is what it always passed.
template <class T>
__global__ void topk_softmax(
    const T* __restrict__ logits,
    const T* __restrict__ act,
    const T* __restrict__ bias,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden)
{
    topk_softmax_body<T, false>(logits, act, bias, topk_idx, topk_w,
                                num_experts, K, hidden);
}

/// The router projection and its top-K, fused: `logits` is the router WEIGHT
/// here and the logits are computed into shared memory and never stored.
template <class T>
__global__ void router_topk_softmax(
    const T* __restrict__ router_weight,
    const T* __restrict__ act,
    const T* __restrict__ bias,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden)
{
    topk_softmax_body<T, true>(router_weight, act, bias, topk_idx, topk_w,
                               num_experts, K, hidden);
}

/// Single-warp top-K softmax: no shared memory, no __syncthreads.
///
/// The block form above pays three block-wide reduction trees (max, sum, and
/// K argmaxes) through shared memory. At BLOCK=64 each tree is 6 rounds and
/// every round carries a __syncthreads, so routing one decode token through
/// 32 experts runs ~36 barriers to do 32 exponentials. Measured on B200 with
/// graph replay: 4.39 us/call against a 0.54 us empty-kernel floor, and it is
/// called once per layer per token -- 105 us of a ~2.4 ms decode step.
///
/// When the experts fit in a warp's registers (`PerLane` values per lane) the
/// same reductions are __shfl_xor, which need no barrier and no shared
/// traffic. Ties still resolve to the LOWEST index, so routing decisions stay
/// identical to the block form -- that is a correctness requirement, not a
/// nicety: a different expert choice is a different model.
///
/// This is not a Blackwell path; warp shuffles are universal and this helps
/// every architecture equally, so it is not gated on compute capability.
template <class T, int PerLane>
__device__ __forceinline__ void topk_softmax_warp_body(
    const T* __restrict__ logits,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K)
{
    const int n = blockIdx.x;
    const int lane = static_cast<int>(threadIdx.x);
    const T* row = logits + static_cast<long long>(n) * num_experts;

    // The full softmax is never needed. The block form computes probs over
    // all experts, takes the top K, then renormalises them by their own sum --
    // so the partition function divides out exactly:
    //
    //   w_k = (e^{v_k - m} / Z) / sum_j (e^{v_j - m} / Z)
    //       =  e^{v_k - m}      / sum_j e^{v_j - m}      over the K winners
    //
    // and exp is monotonic, so the K winners are the same whether it is
    // applied or not. Selecting on the RAW logits therefore gives identical
    // routing while costing K exponentials instead of num_experts, and drops
    // a whole warp-sum reduction. At E=32 that is 32 expf calls saved.
    int idx[PerLane];
    float val[PerLane];
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
        idx[i] = lane + i * 32;
        val[i] = idx[i] < num_experts ? Logit<T>::to_f32(row[idx[i]])
                                      : -flt_max();
    }

    // K rounds of warp argmax with exclusion. Every lane ends each round
    // holding the winner, so the running sum needs no broadcast.
    i32* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float best_w[8];
    int best_e[8];
    for (int k = 0; k < K; ++k) {
        // Seeded below every representable logit, matching block_argmax's
        // "strictly above the floor" rule now that the scores are raw logits
        // rather than non-negative probabilities.
        float bv = -flt_max();
        int bi = -1;
#pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            if (val[i] > bv || (val[i] == bv && idx[i] < bi)) {
                bv = val[i];
                bi = idx[i];
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
#pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            if (idx[i] == bi) val[i] = -flt_max();  // exclude on the next round
        }
        best_w[k] = bv;
        best_e[k] = bi;
    }
    if (lane == 0) {
        // best_w[0] is the row max by construction, so it is the shift that
        // keeps the exponentials in range -- the same one the block form
        // subtracts. Copy it out first: writing best_w[0] on the k=0
        // iteration would leave every later term shifted by exp(0)=1 instead
        // of by the max.
        const float row_max = best_w[0];
        float w_sum = 0.f;
        for (int k = 0; k < K; ++k) {
            best_w[k] = expf(best_w[k] - row_max);
            w_sum += best_w[k];
        }
        const float inv_w = 1.f / w_sum;
        for (int k = 0; k < K; ++k) {
            out_idx[k] = best_e[k];
            out_w[k] = best_w[k] * inv_w;
        }
    }
}

// The ladder, one entry point per rung. The launcher picks by expert count:
// 32, 64, 128, 256, and 512 -- which is `kSoftmaxMaxExperts`, and 16 values
// per lane is 32 registers of scores plus indices, which still leaves the
// warp room. Each is three lines around a shared body, so the arithmetic
// exists once however many rungs there are.
template <class T>
__global__ void topk_softmax_warp_x1(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    topk_softmax_warp_body<T, 1>(logits, topk_idx, topk_w, num_experts, K);
}

template <class T>
__global__ void topk_softmax_warp_x2(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    topk_softmax_warp_body<T, 2>(logits, topk_idx, topk_w, num_experts, K);
}

template <class T>
__global__ void topk_softmax_warp_x4(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    topk_softmax_warp_body<T, 4>(logits, topk_idx, topk_w, num_experts, K);
}

template <class T>
__global__ void topk_softmax_warp_x8(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    topk_softmax_warp_body<T, 8>(logits, topk_idx, topk_w, num_experts, K);
}

template <class T>
__global__ void topk_softmax_warp_x16(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    topk_softmax_warp_body<T, 16>(logits, topk_idx, topk_w, num_experts, K);
}

/// `topk_w[t] *= per_expert_scale[topk_idx[t]]` over the flat `[N, K]` table.
///
/// Flat pointwise, one thread per routed slot — the one kernel in this file
/// whose launch a ported rule states exactly.
template <class T>
__global__ void apply_per_expert_scale(
    const i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const T* __restrict__ per_expert_scale,
    int total)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int e = topk_idx[t];
    // An unrouted slot has no scale to apply and its weight is already the
    // zero the router gave it, so leaving it alone is the whole right answer.
    // Reading `per_expert_scale[-1]` instead is one element before the table
    // and `0 * NaN` is `NaN`, which the combine would fold into the token.
    if (e < 0) return;
    const float s = Logit<T>::to_f32(per_expert_scale[e]);
    topk_w[t] *= s;
}

/// Sigmoid routing with a per-expert correction bias, over bf16 or fp32
/// logits.
///
/// One template where the `.cu` had two kernels. The bias enters the CHOICE
/// and not the WEIGHT — `choice` is ranked, `probs` is published — which is
/// the DeepSeek-style correction; swapping the two reweights every expert by
/// its own bias and still produces plausible text.
///
/// `correction_bias` is read unconditionally: this entry point is the one a
/// checkpoint WITH a bias uses, and the launchers pass a real pointer.
template <class T>
__global__ void topk_sigmoid_bias(
    const T* __restrict__ logits,
    const float* __restrict__ correction_bias,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts,
    int K,
    int normalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row = logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[kSoftmaxMaxExperts];
    __shared__ float choice[kSoftmaxMaxExperts];
    __shared__ float buf[kSoftmaxBlock];
    __shared__ int ibuf[kSoftmaxBlock];

    for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
        const float z = Logit<T>::to_f32(row[j]);
        const float p = 1.f / (1.f + __expf(-z));
        probs[j] = p;
        choice[j] = p + correction_bias[j];
    }
    __syncthreads();

    i32* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -flt_max();
        int best_i = -1;
        block_argmax(choice, num_experts, -flt_max(), buf, ibuf, best_v, best_i);
        const float weight = best_i >= 0 ? probs[best_i] : 0.f;
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = weight;
            if (best_i >= 0) choice[best_i] = -flt_max();
        }
        sum += weight;
        __syncthreads();
    }
    if (tid == 0) {
        const float scale =
            normalize ? (routed_scaling_factor / (sum + 1e-20f))
                      : routed_scaling_factor;
        for (int k = 0; k < K; ++k) {
            out_w[k] *= scale;
        }
    }
}

}  // namespace pie::moe
