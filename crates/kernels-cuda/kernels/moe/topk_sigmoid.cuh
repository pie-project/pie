//===-- topk_sigmoid.cuh - the sigmoid router's top-k ---------------------===//
//
// One `__global__` template and the two constants its shared arrays are
// sized by. No host function, no `<<<>>>`, no entry point -- and since §43,
// no `topk_sigmoid.cu` either: that file held only the launcher, the row
// `moe::topk_sigmoid_bf16` is in `JIT_DISPATCHED` so the shim never
// emitted an entry for it, and nothing else called it. The kernel has
// exactly ONE definition in the tree and now only one compiler reads it.
//
// # Why the split, and what a copy would have cost
//
// `new-horizon.md` §10.10 names the failure this arrangement prevents, and it
// is the expensive one: a `__global__` copied into a `.cuh` while the `.cu`
// keeps its own leaves the archive holding one kernel and the JIT the other,
// with every test passing on whichever half it exercises. `norm/altup_aux`
// did exactly that for a release. So the text MOVED; nothing here is a second
// copy of anything.
//
// # The kernel is a template, and only in the element type
//
// The original was `_bf16` and only `_bf16`, because an ahead-of-time build
// spends a translation unit per instantiation and nobody was going to spend
// one on a second router. Under a JIT the element type is the row's, so the
// kernel is written over `T` and reaches its scalar layer through
// `Elem<T>` — which is what `norm/elementwise.cuh` established and
// what makes an fp16 router a line in a table rather than a build-time
// budget.
//
// # The block width became the rule's, and that is why there is a row
//
// The expert bound stays the kernel's: `scores`, `orig_scores` and `taken`
// are static `__shared__` arrays sized by `kSigmoidMaxExperts`, and the
// launcher REFUSES a wider router rather than clamping it. The block width
// did not stay the kernel's. The staging loops used to step by
// `kSigmoidBlock`, which pinned the launch to 128 threads and matched no
// ported rule; they now step by `blockDim.x`, which is the same arithmetic
// per element at 128 and correct at any width. `LaunchRule::Rms` — one block
// per row, 256 threads — therefore states this launch exactly, and the row
// in `families/moe.rs` is the whole of what the launcher used to decide.
//
// The 32 bytes `Rms` hands the launch as dynamic shared memory are never
// read: this kernel reduces on thread 0 and does not call
// `block_sum`, which is what that allowance is for.
//
// # Why the constants are named for this file
//
// `MAX_EXPERTS` is what the `.cu` called it, and three files in this family
// called it that with three meanings' worth of arithmetic behind it. They now
// share one namespace — `pie::moe` — where a `constexpr int MAX_EXPERTS`
// per header is a redefinition the moment two of them meet in one translation
// unit. The names are per-header for that reason, not for decoration.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::moe {

// The scalar layer is the PRELUDE's. Named here so the kernel below reads as
// it always did, and so that `i32` inside `kernels::moe` keeps
// meaning what it meant once this nested namespace exists to shadow it.

/// The widest router this kernel's static shared arrays hold.
///
/// Qwen3.6-35B-A3B routes through 256 experts and Kimi K2.6 through 384; 512
/// covers both in 2 KB per array. The launcher refuses a wider router rather
/// than truncating it, because a silent truncation is a routing decision
/// nothing reports.
constexpr int kSigmoidMaxExperts = 512;

/// Sigmoid routing with an optional per-expert correction bias: the top `K`
/// experts of each row, and their UNBIASED gate values as weights.
///
/// One block per row, `E` experts staged in shared memory. The bias enters
/// the CHOICE and not the WEIGHT — `scores` is what the scan ranks and
/// `orig_scores` is what it publishes — which is the DeepSeek-style
/// correction, and swapping the two silently reweights every expert by its
/// own bias.
///
/// `correction_bias` may be null: a family without one states no fourth
/// operand and the row's `Source::Or(&Weight(0), &Lit(Null))` supplies the
/// null, which is the same reading the ahead-of-time launcher took.
template <class T>
__global__ void topk_sigmoid(
    const T* __restrict__ logits,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const float* __restrict__ correction_bias,
    int E,
    int K,
    bool renormalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row = logits + static_cast<long long>(n) * E;
    __shared__ float scores[kSigmoidMaxExperts];
    __shared__ float orig_scores[kSigmoidMaxExperts];

    for (int e = tid; e < E; e += blockDim.x) {
        const float x = Elem<T>::to_f32(row[e]);
        const float s = 1.f / (1.f + expf(-x));
        orig_scores[e] = s;
        scores[e] = correction_bias != nullptr ? s + correction_bias[e] : s;
    }
    __syncthreads();

    // `taken` rather than poisoning `scores` with -flt_max(): the poison value
    // is indistinguishable from a genuine score, so a row containing NaN
    // (every comparison against which is false) or K > E would leave the scan
    // with no winner. That used to fall out as `best_i == -1`, which then
    // wrote `scores[-1]` -- an out-of-bounds shared write -- and published
    // expert -1 into `topk_idx`, where the MoE pointer builder turned it into
    // a negative weight offset and the failure finally surfaced as an illegal
    // address inside a batched GEMM, far from its cause.
    __shared__ bool taken[kSigmoidMaxExperts];
    for (int e = tid; e < E; e += blockDim.x) taken[e] = false;
    __syncthreads();

    if (tid == 0) {
        i32* idx = topk_idx + static_cast<long long>(n) * K;
        float* w = topk_w + static_cast<long long>(n) * K;
        float sum = 0.f;
        const int picks = K < E ? K : E;
        for (int k = 0; k < picks; ++k) {
            int best_i = -1;
            float best_v = -flt_max();
            for (int e = 0; e < E; ++e) {
                if (taken[e]) continue;
                const float v = scores[e];
                // Seeding from the first untaken expert keeps a winner even
                // when every remaining score is NaN; for ordinary rows this is
                // the same first-maximum the strict `>` scan already produced.
                if (best_i < 0 || v > best_v) {
                    best_v = v;
                    best_i = e;
                }
            }
            idx[k] = best_i;
            w[k] = orig_scores[best_i];
            sum += orig_scores[best_i];
            taken[best_i] = true;
        }
        // Only reachable when a checkpoint asks for more routes than it has
        // experts. Repeating the last expert would double-count it in the
        // weighted sum, so these slots are parked on expert 0 with zero weight.
        for (int k = picks; k < K; ++k) {
            idx[k] = 0;
            w[k] = 0.f;
        }
        const float scale = renormalize && sum > 0.f
            ? routed_scaling_factor / sum
            : routed_scaling_factor;
        for (int k = 0; k < K; ++k) w[k] *= scale;
    }
}

}  // namespace pie::moe
