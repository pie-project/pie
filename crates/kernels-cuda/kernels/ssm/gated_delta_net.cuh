//===-- gated_delta_net.cuh - GDN's fourteen recurrence kernels ---------===//
//
// Gated Delta Net's recurrence: two decode steps and their GQA, fused, SMEM
// and FLA variants, plus five chunked-prefill kernels. Fourteen `__global__`
// templates and four `__device__` helpers, with no host function and no
// `<<<>>>`. `gated_delta_net.cu` includes this file and keeps only its
// launchers — one definition of each kernel in the tree, which is what
// `tests/device_sources.rs` enforces after `norm/altup_aux` shipped two
// for a release with every test green on whichever half it exercised.
//
// # This header carried NO rows for three separate reasons, and all three are
// # gone
//
// The refusal recorded here read: **(1)** the grid is `(requests, heads)` and
// `Dims` carries only `rows`, `width` and `in_width`; **(2)** these are
// `<StateT, bool KLast>` and `<StateT, int BV, int BK_MAX>` templates and
// `DeviceKernel::instantiation()` emits exactly one type argument; **(3)**
// `slot_stride_elems` is a `long long` and `Args::bind` answered
// `ArgError::Unsupported` for `Ty::I64`. It closed *"ready for the day a rule
// can say `(rows, heads)` and the binder can marshal an `I64`"*.
//
// That day arrived in one commit and every clause of it is now false:
//
//  1. `Dims` gained `q_heads`, `kv_heads`, `head_dim`, `rotary_dims`,
//     `n_experts` and `experts_per_token`, and `LaunchRule::RecurrentScan`
//     states `grid(rows, kv_heads)`, `block(128)`,
//     `smem = 2 * head_dim * sizeof(float)` — the four `<<<>>>` fields of
//     `recurrent_gated_delta_step` and its three siblings, unchanged.
//     `runtime::launch::recurrent_scan` cites those launchers by name.
//  2. `elem` is pasted between the angle brackets and NVRTC parses C++ there,
//     so it carries an ARGUMENT LIST. `crate::device::args` has the eight
//     measured cases. Only the FIRST argument is prefixed, which is what the
//     two aliases below exist for.
//  3. `Ty::I64` binds, through its own `ArgValue` kept distinct from `Usize`
//     because a signed stride and an unsigned size are not the same claim.
//
// THIRTEEN of the fourteen are rows now — `families::ssm::GATED_DELTA_NET`.
// Five were rows when the paragraph below was written and eight arrived with
// §57, which is the section that also deleted the launchers: the refusals
// the paragraph records were all refusals of a LAUNCH RULE, and
// `LaunchRule::Unstated` plus a driver-owned `kernels::Launch` in
// `kernels-cuda/src/x/ssm.rs` is exactly the answer to "no rule
// states this". Read the next paragraph as the history of what each
// row had to route AROUND, not as a list of what is still missing. The one
// kernel with no row is the `_fused` pair's, and that is not a vocabulary
// refusal either: `qwen_gdn_fused_step_enabled()` was `constexpr false`, so
// no launcher in either archive ever launched it.
//
// `recurrent_step_batched_gqa_smem` is reached on a SHAPE the fire states
// (`V_d == 128 && K_d == 128`), which `Term` cannot spell until §26.10(b)'s
// `Term::IntIs` lands and which selects an arm that has no row and no
// `LaunchRule` — it opens `grid(ceil(V_d/BV), R, V_h)` on
// `K_d*BV*sizeof(bf16) + 2*K_d*sizeof(float)`, which is not `RecurrentScan`
// in either shape or size. It used to be reached through an ENVIRONMENT
// VARIABLE; §30 measured that arm against the one below it, found them
// **byte-identical at eight shapes on both results**, and deleted the
// variable rather than relocating it. What makes that identity hold is the
// `__float2bfloat16` in this kernel's phase 2 — it rounds `state*g` where
// the legacy kernel's HBM round trip rounds it, and it is not a redundant
// conversion to be optimised away. `..._fla` is a three-axis grid; the
// `_fused` pair and the chunked
// prefills each want a second head width (`K_d * V_d`) or a chunk axis that
// `Dims` does not carry. Their launchers stay.
//
// **`KLast` is `false` in every row, and that is a value read off the
// launcher.** `gated_delta_net.cu` picks the arm with
// `if (qwen_gdn_k_last_state_enabled())`, which is `constexpr bool ... {
// return false; }` at file scope: one arm is compiled and the other is dead.
// So a row spelling `false` states the instantiation the archive ships, on
// the same footing as `dsv4_hc.cu`'s `constexpr int BLOCK = 256` that
// `norm::hc_pre_postprocess_bf16`'s `elem` spells. If that `constexpr` ever
// returns `true`, these five rows are stale and the fix is five `elem`
// strings — which is why the coupling is written here, where the reader
// flipping the switch will be.
//
// # Why this is still on `<cuda_bf16.h>` and not the prelude
//
// §10.5 records the prelude conversion for this file as REVERTED, and the
// reason is `recurrent_step_batched_gqa_smem` below: it stages the bf16 state
// slab as `__nv_bfloat162` and moves it with `__floats2bfloat162_rn` and
// `__bfloat1622float2`. The prelude has no packed-half type — its `bf16` is a
// scalar struct with an explicit `operator float()` — so converting would mean
// rewriting the kernel that bought +32% end-to-end on Qwen/Qwen3.5-4B
// (6924 -> 9166 tok/s), and §8 says a changed `__global__` body needs its own
// parity evidence. Compiling is not measuring arithmetic.
//
// That is also why the seven pre-recurrence kernels went to a SEPARATE
// header, `ssm/gated_delta_net_prep.cuh`: NVRTC compiles a unit whole, so one
// unresolved packed-half intrinsic here would take that unit's five rows down
// with it. The file already had the seam — two anonymous namespaces with a
// hundred lines of launcher between them — and the split follows it.
//
// Whether this header compiles under NVRTC against the crate's carried
// `cuda_bf16.h` shim was UNMEASURED until it was measured, and the answer was
// no: `gated_delta_net.cuh(73): catastrophic error: could not open source
// file "cstdint"`. `<cuda_bf16.h>` on the line above resolved — the shim set
// carried in the Rust binary answers it — but NVRTC ships no standard library
// at all, which §13's probe put at 0 of 31 headers, and `<cstdint>` is one of
// the 31. So the include is gone and the fourteen kernels below take their
// integer names from the prelude, as the other fifty converted files do.
//
// That measurement is what made the unit possible: NVRTC compiles a unit
// whole, so the five rows below rest on this whole file resolving under the
// carried shim set rather than on the five kernels they name.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>

// The prelude, for its integer names only. `u8` is the per-row persistence
// mask's element type, `u32` and `i32` the index arrays', and `usize` the
// pointer-width unsigned the alignment check casts through — each spelled as
// the COMPILER's own type, which is what makes them the same types
// `<cstdint>` was handing out and this an ABI-neutral substitution rather
// than a widening. The bf16 half of this file is still `__nv_bfloat16` and
// `__floats2bfloat162_rn` — §10.5 records that conversion as REVERTED, and
// nothing here reverses it.
#include "prelude/device.cuh"

namespace pie::ssm {

// The integer layer is the PRELUDE's. Named here rather than qualified at
// every use so the fourteen signatures below read as they did, and repeated
// verbatim in `gated_delta_net_prep.cuh` because a using-declaration at
// namespace scope may be declared as many times as a unit sees it — the two
// headers land in the same namespace in the same translation unit, and that
// is legal and intended.

/// The two `StateT`s the recurrence is instantiated for, SPELLABLE.
///
/// `DeviceKernel::instantiation()` glues `::pie::` to
/// the front of `elem`'s first argument, so a row's state type has to resolve
/// INSIDE this tree's namespaces -- and `float` and `__nv_bfloat16` are both
/// at global scope. Without these two lines the rows in
/// `families::ssm::GATED_DELTA_NET` could name the grid, the block, the
/// shared memory and the `KLast` flag, and still not name the type.
///
/// `state_bf16` is `__nv_bfloat16` and NOT the prelude's `bf16`,
/// which is a different type: `state_load`/`state_store` below are
/// template-specialised on `__nv_bfloat16` and a `bf16` state would
/// fall into the generic `static_cast` primary template instead. Same two
/// bytes, a different rounding path, and nothing reports the substitution --
/// which is why the alias names the compiler's type rather than a convenient
/// one. `gated_delta_net.cu` casts to `__nv_bfloat16*` at every call site and
/// keeps compiling: an alias declares no new type.
using f32 = float;
using state_bf16 = __nv_bfloat16;

/// `recurrent_step_batched_gqa_smem`'s ONE `BV`, spellable for the same
/// reason the two aliases above are.
///
/// That kernel is `template <int BV>` — a single NON-TYPE parameter — and
/// `elem` is the whole argument list with `::pie::`
/// glued to its FIRST token. A row spelling `elem: "128"` emits
/// `recurrent_step_batched_gqa_smem<::pie::128>`, which
/// is not a C++ token sequence at all; the prefix reaches the first argument
/// and there is no second one to hide behind. `chunk_gated_delta_prefill_
/// batched_fla<StateT, BV, BK_MAX>` escapes this only because its first
/// argument is a type — `"pie::ssm::f32, 128, 128"` prefixes `f32` and
/// leaves the two integers alone.
///
/// `DeviceKernel::PLAIN` is not the way out either: it emits no angle
/// brackets, and `kernels-cuda/tests/layers.rs`'s
/// `every_row_spells_a_qualified_instantiation` asserts a plain row's
/// instantiation carries neither `<` nor `>`.
///
/// So the number gets a name. `gated_delta_net.cu:243` held it as
/// `constexpr int BV = 128;` beside the launch; this is that same constant,
/// moved to where a row can reach it, and the launcher's grid
/// (`ceil(V_d / BV)` on `grid.x`) is stated against it in
/// `kernels-cuda/src/x/ssm.rs` as `SMEM_BV`. Change one and both rows
/// and that grid are stale together, which is why they cite this line.
constexpr int gqa_smem_bv = 128;

template <typename StateT>
__device__ __forceinline__ float state_load(const StateT* p) {
    return static_cast<float>(*p);
}

template <>
__device__ __forceinline__ float state_load<__nv_bfloat16>(
    const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename StateT>
__device__ __forceinline__ void state_store(StateT* p, float v) {
    *p = static_cast<StateT>(v);
}

template <>
__device__ __forceinline__ void state_store<__nv_bfloat16>(
    __nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

template <bool KLast>
__device__ __forceinline__ long long state_offset(
    int k_idx, int v_idx, int K_d, int V_d) {
    if constexpr (KLast) {
        return (long long)v_idx * K_d + k_idx;
    } else {
        return (long long)k_idx * V_d + v_idx;
    }
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return __shfl_sync(0xffffffffu, x, 0);
}

// One block per (request, head). Threads parallelize over v_idx in
// [0, V_d). Each thread loops over k_idx in [0, K_d) twice (once for
// the kv_mem accumulation, once for the post-update output).
//
// Shared memory layout: q[K_d] + k[K_d] fp32. Caller passes shmem of
// size 2*K_d*sizeof(float).
// Per-row refinement of the pass-level `write_state`. A mixed fire folds some
// rows and leaves others buffered, and the two differ ONLY in whether the
// recurrence persists -- same initial state, same outputs. A null mask means
// the pass is uniform, which is the overwhelmingly common case.
__device__ __forceinline__ bool row_persists(
    const u8* __restrict__ mask, int r) {
    return mask == nullptr || mask[r] != 0;
}

template <typename StateT, bool KLast>
__global__ void recurrent_step(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;

    const long long bh = (long long)b * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    state += bh * (long long)K_d * V_d;
    out   += bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    // Load q/k into shmem cooperatively.
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    // Phase 1: state *= g, kv_mem[v] = Σ_k state[k, v] * k[k].
    // Output of this phase: kv_mem[v] in register `kv_mem` for each
    // thread that owns its v_idx.
    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t   = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        // Phase 2: state[k, v] += k[k] * delta; out[v] = Σ_k state[k,v]*q[k].
        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) + sk[k_idx] * delta;
            state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out[v_idx] = out_v;
    }
}

// Multi-request batched chunked prefill. One block per (request, head);
// the block walks its T_r tokens sequentially (per-token state
// dependency), accumulating the recurrence into the request's state
// slab. Same per-token math as `recurrent_step`.
template <typename StateT, bool KLast>
__global__ void chunk_gated_delta_prefill_batched(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm + bh * K_d;
        const float* k_h = k_norm + bh * K_d;
        const float* v_h = v      + bh * V_d;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];
        float* out_bh = out + bh * V_d;

        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h[i];
            sk[i] = k_h[i];
        }
        __syncthreads();

        for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
            float kv_mem = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = state_load(state + off) * g_h;
                state_store(state + off, s);
                kv_mem += s * sk[k_idx];
            }

            const float v_t   = v_h[v_idx];
            const float delta = (v_t - kv_mem) * beta_h;

            float out_v = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = state_load(state + off) + sk[k_idx] * delta;
                state_store(state + off, s);
                out_v += s * sq[k_idx];
            }
            out_bh[v_idx] = out_v;
        }
        // State must be globally visible before next-token's reads;
        // __syncthreads ensures the block sees its own writes — adjacent
        // blocks (different r or h) are independent.
        __syncthreads();
    }
}

template <typename StateT, bool KLast>
__global__ void chunk_gated_delta_prefill_batched_cached(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float s_state[];
    const int state_elems = K_d * V_d;
    for (int i = threadIdx.x; i < state_elems; i += blockDim.x) {
        s_state[i] = state_load(state + i);
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm + bh * K_d;
        const float* k_h = k_norm + bh * K_d;
        const float* v_h = v      + bh * V_d;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];
        float* out_bh = out + bh * V_d;

        for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
            float kv_mem = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = s_state[off] * g_h;
                s_state[off] = s;
                kv_mem += s * k_h[k_idx];
            }

            const float delta = (v_h[v_idx] - kv_mem) * beta_h;
            float out_v = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = s_state[off] + k_h[k_idx] * delta;
                s_state[off] = s;
                out_v += s * q_h[k_idx];
            }
            out_bh[v_idx] = out_v;
        }
    }

    // Frozen verify (write_state=false): produce outputs but persist nothing,
    // leaving the committed slot at its pre-verify value (advanced later by the
    // repair forward).
    if (write_state && row_persists(write_state_mask, r)) {
        __syncthreads();
        for (int i = threadIdx.x; i < state_elems; i += blockDim.x) {
            state_store(state + i, s_state[i]);
        }
    }
}

template <typename StateT, bool KLast>
__global__ void chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask)
{
    constexpr int WARPS = 4;
    constexpr int MAX_K_PER_LANE = 8;  // supports K_d <= 256 with 32 lanes
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int v_tile = blockIdx.z * WARPS;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v_idx = v_tile + warp;
    if (warp >= WARPS || v_idx >= V_d) return;

    const int repeat = V_h / K_h;
    const int qk_h = h / repeat;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    float s_vals[MAX_K_PER_LANE];
    int k_vals[MAX_K_PER_LANE];
    int n_k = 0;
    for (int k_idx = lane; k_idx < K_d && n_k < MAX_K_PER_LANE; k_idx += 32) {
        k_vals[n_k] = k_idx;
        s_vals[n_k] = state_load(
            state + state_offset<KLast>(k_idx, v_idx, K_d, V_d));
        ++n_k;
    }

    for (int t = 0; t < T; ++t) {
        const long long qk_bh = ((long long)(t0 + t) * K_h + qk_h);
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm_kh + qk_bh * K_d;
        const float* k_h = k_norm_kh + qk_bh * K_d;
        const float* v_h = v + vh * V_d;
        const float g_h = __expf(g_log[vh]);
        const float beta_h = beta[vh];

        float kv_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] * g_h;
                s_vals[i] = s;
                kv_part += s * k_h[k_idx];
            }
        }
        const float kv_mem = warp_sum(kv_part);
        const float delta = (v_h[v_idx] - kv_mem) * beta_h;

        float out_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] + k_h[k_idx] * delta;
                s_vals[i] = s;
                out_part += s * q_h[k_idx];
            }
        }
        const float out_v = warp_sum(out_part);
        if (lane == 0) {
            out[vh * (long long)V_d + v_idx] = out_v;
        }
    }

    // Frozen verify (write_state=false): walk the state in registers for
    // correct draft outputs but persist nothing, leaving the committed slot at
    // its pre-verify value (the implicit speculative snapshot). The repair
    // forward over [input|accepted] advances it afterward.
    if (write_state && row_persists(write_state_mask, r)) {
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                state_store(
                    state + state_offset<KLast>(
                        k_vals[i], v_idx, K_d, V_d),
                    s_vals[i]);
            }
        }
    }
}

template <typename StateT, bool KLast>
__global__ void chunk_gated_delta_prefill_batched_warp_tiled_gqa_ilp2(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask)
{
    constexpr int WARPS = 4;
    constexpr int ILP_V = 2;
    constexpr int TILE_V = WARPS * ILP_V;
    constexpr int MAX_K_PER_LANE = 8;  // supports K_d <= 256 with 32 lanes
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v0 = blockIdx.z * TILE_V + warp * ILP_V;
    const int v1 = v0 + 1;
    if (warp >= WARPS || v0 >= V_d) return;
    const bool has_v1 = v1 < V_d;

    const int repeat = V_h / K_h;
    const int qk_h = h / repeat;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    float s0[MAX_K_PER_LANE];
    float s1[MAX_K_PER_LANE];
    int k_vals[MAX_K_PER_LANE];
    int n_k = 0;
    for (int k_idx = lane; k_idx < K_d && n_k < MAX_K_PER_LANE; k_idx += 32) {
        k_vals[n_k] = k_idx;
        s0[n_k] = state_load(
            state + state_offset<KLast>(k_idx, v0, K_d, V_d));
        s1[n_k] = has_v1
            ? state_load(state + state_offset<KLast>(k_idx, v1, K_d, V_d))
            : 0.f;
        ++n_k;
    }

    for (int t = 0; t < T; ++t) {
        const long long qk_bh = ((long long)(t0 + t) * K_h + qk_h);
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm_kh + qk_bh * K_d;
        const float* k_h = k_norm_kh + qk_bh * K_d;
        const float* v_h = v + vh * V_d;
        const float g_h = __expf(g_log[vh]);
        const float beta_h = beta[vh];

        float kv_part0 = 0.f;
        float kv_part1 = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float k_val = k_h[k_idx];
                const float s_v0 = s0[i] * g_h;
                s0[i] = s_v0;
                kv_part0 += s_v0 * k_val;
                if (has_v1) {
                    const float s_v1 = s1[i] * g_h;
                    s1[i] = s_v1;
                    kv_part1 += s_v1 * k_val;
                }
            }
        }
        const float kv_mem0 = warp_sum(kv_part0);
        const float kv_mem1 = has_v1 ? warp_sum(kv_part1) : 0.f;
        const float delta0 = (v_h[v0] - kv_mem0) * beta_h;
        const float delta1 = has_v1 ? (v_h[v1] - kv_mem1) * beta_h : 0.f;

        float out_part0 = 0.f;
        float out_part1 = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float k_val = k_h[k_idx];
                const float q_val = q_h[k_idx];
                const float new_s0 = s0[i] + k_val * delta0;
                s0[i] = new_s0;
                out_part0 += new_s0 * q_val;
                if (has_v1) {
                    const float new_s1 = s1[i] + k_val * delta1;
                    s1[i] = new_s1;
                    out_part1 += new_s1 * q_val;
                }
            }
        }
        const float out_v0 = warp_sum(out_part0);
        const float out_v1 = has_v1 ? warp_sum(out_part1) : 0.f;
        if (lane == 0) {
            out[vh * (long long)V_d + v0] = out_v0;
            if (has_v1) out[vh * (long long)V_d + v1] = out_v1;
        }
    }

    // Frozen verify (write_state=false): persist nothing — see the non-ILP2
    // GQA kernel above.
    if (write_state && row_persists(write_state_mask, r)) {
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                state_store(
                    state + state_offset<KLast>(k_vals[i], v0, K_d, V_d),
                    s0[i]);
                if (has_v1) {
                    state_store(
                        state + state_offset<KLast>(k_vals[i], v1, K_d, V_d),
                        s1[i]);
                }
            }
        }
    }
}

// Batched variant with slot indirection. State for request r lives at
// `state_base + slot_ids[r] * slot_stride_elems`. Otherwise the
// per-(request, head) compute is identical to `recurrent_step`.
template <typename StateT, bool KLast>
__global__ void recurrent_step_batched(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const long long bh = (long long)r * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t   = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) + sk[k_idx] * delta;
            state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out_bh[v_idx] = out_v;
    }
}

template <typename StateT, bool KLast>
__global__ void recurrent_step_batched_gqa(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) + sk[k_idx] * delta;
            state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out_bh[v_idx] = out_v;
    }
}

// Fused recurrent-step kernel: same per-token math as
// `recurrent_step` / `recurrent_step_batched`, but
// reorganized to halve the state slab HBM traffic. The original
// kernel reads state, scales by g and writes back, then reads state
// again to apply delta and writes again (4 ops per element). The
// fused variant:
//
//   1. Reads state ONCE into a per-thread register cache `s_cache[K_d]`.
//   2. Accumulates sum_s_sk = Σ_k s[k]*sk[k] (proxy for kv_mem)
//      and  sum_s_sq = Σ_k s[k]*sq[k] (proxy for partial out_v).
//   3. Computes kv_mem = g * sum_s_sk, delta = (v - kv_mem) * beta.
//   4. Computes out_v = g * sum_s_sq + delta * sum_sk_sq, where
//      sum_sk_sq is a per-block constant precomputed once in shmem.
//   5. Writes the updated state once: state[k,v] = s_cache[k]*g + sk[k]*delta.
//
// Memory traffic per (head, batch, v_idx): K_d state reads + K_d
// state writes (1R+1W) vs the original's 2R+2W. Register footprint:
// K_d floats per thread, fine for K_d up to 256 on H100 (255-reg cap).
//
// Output equivalence:
//   final state[k,v] = s_initial[k,v] * g + sk[k] * delta
//   out_v          = Σ_k (s_initial[k,v]*g + sk[k]*delta) * sq[k]
//                  = g * Σ_k s_initial*sq + delta * Σ_k sk*sq
//                  = g * sum_s_sq + delta * sum_sk_sq
// — exactly the value the original kernel computes via its second
// state read. The analytical decomposition introduces no extra FLOPs
// (same 3 K_d FMAs per element) while saving half the state I/O.
//
// Template `K_D_MAX` bounds the per-thread state cache. We
// dispatch at launch time on the actual K_d.
template <typename StateT, bool KLast, int K_D_MAX>
__global__ void recurrent_step_batched_fused(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const long long bh = (long long)r * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;
    // sum_sk_sq is a per-block scalar (same value for every v_idx
    // since sk·sq depends only on the head). Reduce it cooperatively
    // and broadcast via shared memory; saves K_d FMAs per thread.
    float* sm_scalars = smem + 2 * K_d;  // [sum_sk_sq]

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    // Cooperative reduction of sum_sk_sq across the block.
    float partial = 0.f;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        partial += sk[i] * sq[i];
    }
    // Warp + block reduce.
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_xor_sync(0xffffffffu, partial, offset);
    }
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        float w = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
        for (int offset = 16; offset > 0; offset /= 2) {
            w += __shfl_xor_sync(0xffffffffu, w, offset);
        }
        if (threadIdx.x == 0) sm_scalars[0] = w;
    }
    __syncthreads();
    const float sum_sk_sq = sm_scalars[0];

    // Per-thread state cache. K_D_MAX bounds the static array; we
    // only ever touch [0, K_d). Sized for the worst case across
    // instantiations (currently K_d <= 256 for Qwen3.5 family).
    float s_cache[K_D_MAX];

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        // Pass 1: read state, cache, accumulate kv_mem & out_v partials.
        float sum_s_sk = 0.f;
        float sum_s_sq = 0.f;
        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off);
            s_cache[k_idx] = s;
            sum_s_sk += s * sk[k_idx];
            sum_s_sq += s * sq[k_idx];
        }

        const float kv_mem = g_h * sum_s_sk;
        const float v_t    = v_h[v_idx];
        const float delta  = (v_t - kv_mem) * beta_h;
        // out_v = g * Σ s*sq + delta * Σ sk*sq, an algebraic
        // rewrite of the original Phase-2 reduction.
        const float out_v  = g_h * sum_s_sq + delta * sum_sk_sq;
        out_bh[v_idx] = out_v;

        // Pass 2: write updated state.
        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s_new = s_cache[k_idx] * g_h + sk[k_idx] * delta;
            state_store(state + off, s_new);
        }
    }
}

template <typename StateT, bool KLast, int K_D_MAX>
__global__ void recurrent_step_batched_gqa_fused(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;
    float* sm_scalars = smem + 2 * K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float partial = 0.f;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        partial += sk[i] * sq[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_xor_sync(0xffffffffu, partial, offset);
    }
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        float w = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
        for (int offset = 16; offset > 0; offset /= 2) {
            w += __shfl_xor_sync(0xffffffffu, w, offset);
        }
        if (threadIdx.x == 0) sm_scalars[0] = w;
    }
    __syncthreads();
    const float sum_sk_sq = sm_scalars[0];

    float s_cache[K_D_MAX];

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float sum_s_sk = 0.f;
        float sum_s_sq = 0.f;
        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off);
            s_cache[k_idx] = s;
            sum_s_sk += s * sk[k_idx];
            sum_s_sq += s * sq[k_idx];
        }

        const float kv_mem = g_h * sum_s_sk;
        const float v_t    = v_h[v_idx];
        const float delta  = (v_t - kv_mem) * beta_h;
        const float out_v  = g_h * sum_s_sq + delta * sum_sk_sq;
        out_bh[v_idx] = out_v;

        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s_new = s_cache[k_idx] * g_h + sk[k_idx] * delta;
            state_store(state + off, s_new);
        }
    }
}

// ── FLA-style recurrent step (KLast=false / V-last storage only) ──
// C++ port of fla-org/flash-linear-attention's
// fused_recurrent_gated_delta_rule_fwd_kernel for T=1 decode.
//
// Grid: (NV, R, V_h) where NV = ceil(V_d / BV).
// Block: BV threads — each owns one V column of the [K_d, BV] state
//        tile, keeping the per-column state vector in registers and
//        sharing the q/k vectors via shmem.
//
// Microbench on H100 PCIe at R=511, V_h=K_d=V_d=128:
//   legacy kernel    1.92 ms/call  (567 GB/s, 19% of HBM3 peak)
//   FLA-port BV=64   1.22 ms/call  (894 GB/s, +37%, bit-identical)
//
// Only handles KLast=false (V-last) — KLast=true is non-coalesced
// and is the layout the production default already moved off.
template <typename StateT, int BV, int BK_MAX>
__global__ void recurrent_step_batched_fla(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    const long long bh = (long long)r * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float bh_state[BK_MAX];
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] = state_load(state + (long long)k_idx * V_d + v_idx);
    }
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] *= g_h;
    }
    float kv_mem = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        kv_mem += bh_state[k_idx] * sk[k_idx];
    }
    const float v_t   = v_h[v_idx];
    const float delta = (v_t - kv_mem) * beta_h;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] += sk[k_idx] * delta;
    }
    float out_v = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        out_v += bh_state[k_idx] * sq[k_idx];
    }
    out_bh[v_idx] = out_v;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        state_store(state + (long long)k_idx * V_d + v_idx, bh_state[k_idx]);
    }
}

template <typename StateT, int BV, int BK_MAX>
__global__ void recurrent_step_batched_gqa_fla(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float bh_state[BK_MAX];
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] = state_load(state + (long long)k_idx * V_d + v_idx);
    }
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] *= g_h;
    }
    float kv_mem = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        kv_mem += bh_state[k_idx] * sk[k_idx];
    }
    const float v_t   = v_h[v_idx];
    const float delta = (v_t - kv_mem) * beta_h;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] += sk[k_idx] * delta;
    }
    float out_v = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        out_v += bh_state[k_idx] * sq[k_idx];
    }
    out_bh[v_idx] = out_v;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        state_store(state + (long long)k_idx * V_d + v_idx, bh_state[k_idx]);
    }
}

// ── SMEM read-only step kernel (KLast=false / V-last only) ───────
// Stages the V-last state slab into SMEM ONCE (read-only thereafter),
// computes both analytical phases reading from SMEM, and writes the
// updated state straight to HBM from phase 2 — no SMEM writebacks.
// Wins over the legacy 2R+2W kernel by halving HBM round-trips on the
// state slab; wins over the naive SMEM-staging variant by halving
// SMEM traffic (the binding bottleneck: the kernel is SMEM-read-
// bandwidth bound, not HBM bound). State is read once per phase from
// SMEM; phase 2 recomputes `s*g` (one extra FMA) rather than reading
// a stored scaled value, so the only SMEM writes are the one-time
// stage. q/k go into SMEM once with a [k][v_local] layout so adjacent
// threads touch adjacent offsets (coalesced).
//
// Microbench at R=511, V_h=32, K_d=V_d=128, BF16 state on H100 PCIe:
//   ref bf16 v-last (2R+2W)        2406 us  445 GB/s
//   FLA burst regs                 3632 us  296 GB/s (register spill)
//   cp.async SMEM regs             1989 us  539 GB/s
//   SMEM-staging (RMW writebacks)  1660 us  646 GB/s
//   fp32 SMEM (2x read bytes)      2691 us  398 GB/s
//   SMEM read-only (this)          1579 us  679 GB/s  ← best
//
// Precision: fp32 accumulate, rounded to BF16 once on the final
// store (same scheme as the FLA chunked-prefill path, which is the
// default). Not bit-identical to the legacy per-element-BF16-round
// kernel, but strictly less quantization.
template <int BV>
__global__ void recurrent_step_batched_gqa_smem(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* v_h = v + vh * V_d;
    const float g_h = __expf(g_log[vh]);
    const float beta_h = beta[vh];

    __nv_bfloat16* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ __nv_bfloat16 smem_smem_step[];
    __nv_bfloat16* s_state = smem_smem_step;          // read-only after stage
    float* sq = (float*)(smem_smem_step + K_d * BV);
    float* sk = sq + K_d;

    // Stage state HBM → SMEM. Adjacent threads at fixed k load
    // adjacent v indices → coalesced HBM reads; SMEM layout is
    // [k][v_local] → coalesced SMEM writes.
    //
    // Load a tile into registers BEFORE storing any of it. Writing each
    // element to SMEM as it arrives makes every load depend on the
    // previous store's address computation, so only a couple of loads are
    // ever in flight — fine at R=512 where thousands of blocks hide the
    // latency for each other, ruinous at R=1 where 32 blocks are all the
    // parallelism there is. Decoupling the two halves lets a whole tile
    // of loads issue at once. Measured on A100 (V_h=32, K_d=V_d=128):
    //   R=1   19.7 -> 10.8 us      R=8   29.6 -> 16.1 us
    //   R=2   22.9 -> 11.7 us      R=64 138.0 -> 96.5 us
    // bf16 stays bf16 the whole way, so the staged values are identical.
    // When one block covers the whole v axis, SMEM's [k][v] layout is
    // byte-for-byte the HBM tile's, so staging is a flat copy -- and a flat
    // copy can move 16 bytes per thread instead of 2. The scalar path below
    // has each warp touch 32 adjacent bf16, a 64-byte transaction: half a
    // cache line, and the reason this kernel sustained ~1100 GB/s where
    // flashinfer's equivalent reaches ~1450 GB/s on the same shape.
    const bool vec_tile =
        (BV == V_d) && ((V_d & 7) == 0) &&
        ((reinterpret_cast<usize>(state) & 15) == 0);
    const int n_vec = (K_d * V_d) >> 3;
    if (vec_tile) {
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(state);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(s_state);
        for (int i = threadIdx.x; i < n_vec; i += BV) dst[i] = src[i];
    } else {
        constexpr int kStageTile = 16;
        __nv_bfloat16 staged[kStageTile];
        int k = 0;
        for (; k + kStageTile <= K_d; k += kStageTile) {
            #pragma unroll
            for (int u = 0; u < kStageTile; ++u) {
                staged[u] = state[(long long)(k + u) * V_d + v_idx];
            }
            #pragma unroll
            for (int u = 0; u < kStageTile; ++u) {
                s_state[(k + u) * BV + threadIdx.x] = staged[u];
            }
        }
        for (; k < K_d; ++k) {
            s_state[k * BV + threadIdx.x] =
                state[(long long)k * V_d + v_idx];
        }
    }
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    // Phase 1: kv_mem = Σ (state*g) * sk.  No SMEM writes.
    float kv_mem = 0.f;
    for (int k = 0; k < K_d; ++k) {
        float s = __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h;
        kv_mem += s * sk[k];
    }
    const float delta = (v_h[v_idx] - kv_mem) * beta_h;

    // Phase 2: recompute s = state*g + sk*delta; accumulate out_v;
    // write the updated state straight to HBM (skips SMEM writeback).
    //
    // `state*g` is rounded to bf16 before delta is added, which looks
    // gratuitous when the value is already in a register — but it is what
    // makes this kernel a drop-in for the legacy one. The legacy kernel
    // STORES state*g to HBM in its first phase and RELOADS it in its second,
    // so its phase-2 base is bf16-rounded; carrying full fp32 through here
    // instead is more accurate per step and still changes the greedy
    // trajectory, because argmax turns any perturbation into a different
    // token the moment two logits are close. Qwen3.5-0.8B diverged from the
    // HF reference trajectory at the SECOND decoded token that way. Kernel
    // selection is an implementation detail and must not be observable in
    // the output, so match the legacy rounding exactly.
    float out_v = 0.f;
    for (int k = 0; k < K_d; ++k) {
        const float sg = __bfloat162float(__float2bfloat16(
            __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h));
        float s = sg + sk[k] * delta;
        out_v += s * sq[k];
        // Each thread owns column `threadIdx.x` for every k, so rewriting
        // its own SMEM slot races with nothing. Costing a SMEM round trip
        // to make the HBM store a flat vectorised copy is a good trade:
        // the store is the second half of this kernel's HBM traffic.
        if (vec_tile) {
            s_state[k * BV + threadIdx.x] = __float2bfloat16(s);
        } else {
            state[(long long)k * V_d + v_idx] = __float2bfloat16(s);
        }
    }
    out_bh[v_idx] = out_v;
    if (vec_tile) {
        __syncthreads();
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(s_state);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(state);
        for (int i = threadIdx.x; i < n_vec; i += BV) dst[i] = src[i];
    }
}

// ── FLA-style chunked prefill kernel (KLast=false / V-last storage) ──
// State stays in per-thread registers across the whole T-token chunk,
// only one HBM round-trip per (request, head). The reference
// `chunk_gated_delta_prefill_batched` does 2R+2W per state
// element PER TOKEN — i.e. T-fold blowup of state HBM traffic.
//
// Microbench at production shapes (R=512, V_h=K_d=V_d=128, T=36 tokens
// per request) on H100 PCIe:
//   ref (per-token HBM)  47.5 ms/call  (~38 GB of state IO per call)
//   v1 FLA BV=128         5.27 ms/call (~1 GB,  9.0x faster, bit-identical)
//   v1 FLA BV=64          5.57 ms/call (8.5x)
//   v1 FLA BV=32          6.32 ms/call (7.5x)
//
// Picked BV=128 (matches the legacy block shape — 128 threads per
// (request, head) block, one v_idx per thread, state cached in
// 128 floats of registers). Output is bit-identical to the legacy
// kernel — the math is exactly the same per-token recurrence, only
// the in-block state staging changes.
template <typename StateT, int BV, int BK_MAX>
__global__ void chunk_gated_delta_prefill_batched_fla(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const int* __restrict__ commit_len,
    const u8* __restrict__ write_state_mask)
{
    // GQA: q/k are stored compact in K_h heads; g/beta/v/out are V_h heads.
    // Block head h (0..V_h) maps to its K-head h_k = h / (V_h/K_h). When
    // K_h == V_h (no GQA) this is the identity h_k = h, so callers that pass
    // the already-expanded V_h layout with K_h = V_h are unaffected.
    const int gqa_repeat = V_h / K_h;
    // Grid: (NV, R, V_h) where NV = ceil(V_d / BV). Each block owns
    // one V-tile of BV columns. Lowering BV from V_d (single block per
    // (r, h)) to smaller values raises grid parallelism, reduces
    // per-block register pressure, and lets the SM scheduler hide
    // memory latency across more in-flight warps.
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int h_k = h / gqa_repeat;   // K-head feeding this V-head (GQA)
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    const int t0 = static_cast<int>(qo_indptr[r]);
    int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    // Boundary-write (recurrent-only commit-advance): the in-projection ran
    // over the full [input|drafts] window (matching the verify's GEMM tiling),
    // but we only fold the confirmed prefix [input|accepted] into the committed
    // state. Walk just commit_len[r] tokens; the final writeback then lands at
    // the accepted boundary. Rejected drafts (later positions) are never folded.
    if (commit_len != nullptr) {
        const int c = commit_len[r];
        if (c < T) T = c;
    }
    if (T <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;

    // Load the state column [0..K_d, v_idx] into a per-thread register cache,
    // PACKED 2 K-values per register as __nv_bfloat162. This halves the
    // register footprint of the dominant array (128 floats -> 64 bf162),
    // raising occupancy (~19% -> ~31%) so the SM can hide the state-I/O
    // latency that bounds this kernel. All arithmetic still accumulates in
    // fp32 (the packed state is unpacked to float2 per access); only the
    // per-token state value is held in bf16 — which matches the bf16 storage
    // precision the committed slot already uses.
    __nv_bfloat162 bh_state[BK_MAX / 2];
    #pragma unroll
    for (int j = 0; j < BK_MAX / 2; ++j) {
        const int k0 = 2 * j;
        if (k0 >= K_d) break;
        const int k1 = k0 + 1;
        const float s0 = state_load(state + (long long)k0 * V_d + v_idx);
        const float s1 = (k1 < K_d)
            ? state_load(state + (long long)k1 * V_d + v_idx)
            : 0.f;
        bh_state[j] = __floats2bfloat162_rn(s0, s1);
    }

    // COMMIT-LEN-GATED rounding (verified on the 4090): the SAME kernel
    // serves two ops with DIFFERENT bit-exactness references:
    //   * commit_len == nullptr (plain PREFILL, K=0 & K=2 initial): fold N fresh
    //     tokens into a reset state. The HF reference (T0 GDN_GOLDEN) matches the
    //     DOUBLE-round trajectory here — single-round diverges (T0 glitch, the
    //     warp-tiled-bug signature). So plain prefill KEEPS double-round.
    //   * commit_len != nullptr (COMMIT-ADVANCE replay [input|accepted]): must
    //     bit-match the K=0 decode-step kernel (single bf16 round/token) so the
    //     spec-verify is lossless → SINGLE-round (fixes T1 K=0==K=2).
    // Verified: gating → T0 HF-exact (golden) AND T1 K=0==K=2 both green; ungated
    // single-round gave T1 green but T0 red (both glitch); double-round-only gave
    // T0 green but T1 red. (:1625 is shared prefill+commit-advance.)
    const bool single_round = (commit_len != nullptr);

    // Walk T tokens; state stays in registers.
    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];

        // Reload q/k into shmem for this token. q/k are compact in K_h heads.
        const long long bh_qk = (long long)(t0 + t) * K_h + h_k;
        const float* q_h_t = q_norm + bh_qk * K_d;
        const float* k_h_t = k_norm + bh_qk * K_d;
        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h_t[i];
            sk[i] = k_h_t[i];
        }
        __syncthreads();

        // Phase 1: accumulate kv_mem = Σ (state*g)·sk (fp32). single_round leaves
        // bh_state untouched (Phase 2 recomputes state*g from the original);
        // double_round re-packs the g-scaled state into bh_state (the extra round).
        float kv_mem = 0.f;
        #pragma unroll
        for (int j = 0; j < BK_MAX / 2; ++j) {
            const int k0 = 2 * j;
            if (k0 >= K_d) break;
            const int k1 = k0 + 1;
            float2 s = __bfloat1622float2(bh_state[j]);
            s.x *= g_h;
            if (k1 < K_d) s.y *= g_h;
            if (!single_round) bh_state[j] = __floats2bfloat162_rn(s.x, s.y);
            kv_mem += s.x * sk[k0];
            if (k1 < K_d) kv_mem += s.y * sk[k1];
        }
        const float v_t   = v[bh * V_d + v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        // Phase 2: state = state*g + k·δ, accumulate out_v (fp32). single_round
        // recomputes state*g fresh from the ORIGINAL bh_state (one round total);
        // double_round reloads the already-g-scaled-and-rounded bh_state from
        // Phase 1 and adds k·δ (a second round) — matches HF for the plain prefill.
        float out_v = 0.f;
        #pragma unroll
        for (int j = 0; j < BK_MAX / 2; ++j) {
            const int k0 = 2 * j;
            if (k0 >= K_d) break;
            const int k1 = k0 + 1;
            float2 s = __bfloat1622float2(bh_state[j]);
            float sx, sy;
            if (single_round) {
                sx = s.x * g_h + sk[k0] * delta;
                sy = (k1 < K_d) ? (s.y * g_h + sk[k1] * delta) : s.y;
            } else {
                // bh_state already holds round(state*g) from Phase 1.
                sx = s.x + sk[k0] * delta;
                sy = (k1 < K_d) ? (s.y + sk[k1] * delta) : s.y;
            }
            bh_state[j] = __floats2bfloat162_rn(sx, sy);
            out_v += sx * sq[k0];
            if (k1 < K_d) out_v += sy * sq[k1];
        }
        out[bh * V_d + v_idx] = out_v;
        __syncthreads();
    }

    // Single state writeback at the (possibly commit_len-clamped) chunk end.
    // ACTIVATION REPLAY: honor write_state. Frozen verify (write_state=false)
    // persists nothing — it emits draft logits against the correct committed
    // state (the model's true ~54%/c256, ~65%/c1 acceptance; the not-frozen 90%
    // is corrupt co-drift). On accept, the commit-advance replays the recurrence
    // over just the accepted prefix (write_state=true) — lossless, cheap. This is
    // the SoTA (Snakes-and-Ladders / STree) and the right design for the low-
    // concurrency regime MTP targets.
    if (!write_state || !row_persists(write_state_mask, r)) return;
    #pragma unroll
    for (int j = 0; j < BK_MAX / 2; ++j) {
        const int k0 = 2 * j;
        if (k0 >= K_d) break;
        const int k1 = k0 + 1;
        const float2 s = __bfloat1622float2(bh_state[j]);
        state_store(state + (long long)k0 * V_d + v_idx, s.x);
        if (k1 < K_d) state_store(state + (long long)k1 * V_d + v_idx, s.y);
    }
}

// Same kernel for GQA variant (only the (q, k) indexing differs —
// they're loaded once per token via the shared shmem so the structural
// loop is identical to the non-GQA path; only the per-token bh index
// changes).
template <typename StateT, int BV, int BK_MAX>
__global__ void chunk_gated_delta_prefill_batched_gqa_fla(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;

    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;

    float bh_state[BK_MAX];
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] = state_load(state + (long long)k_idx * V_d + v_idx);
    }

    for (int t = 0; t < T; ++t) {
        const long long qh = ((long long)(t0 + t) * K_h + h_k) * K_d;
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float  g_h = __expf(g_log[vh]);
        const float  beta_h = beta[vh];

        const float* q_h_t = q_norm_kh + qh;
        const float* k_h_t = k_norm_kh + qh;
        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h_t[i];
            sk[i] = k_h_t[i];
        }
        __syncthreads();

        float kv_mem = 0.f;
        #pragma unroll
        for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
            if (k_idx >= K_d) break;
            bh_state[k_idx] *= g_h;
            kv_mem += bh_state[k_idx] * sk[k_idx];
        }
        const float v_t   = v[vh * V_d + v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        #pragma unroll
        for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
            if (k_idx >= K_d) break;
            bh_state[k_idx] += sk[k_idx] * delta;
            out_v += bh_state[k_idx] * sq[k_idx];
        }
        out[vh * V_d + v_idx] = out_v;
        __syncthreads();
    }

    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        state_store(state + (long long)k_idx * V_d + v_idx, bh_state[k_idx]);
    }
}

}  // namespace pie::ssm
