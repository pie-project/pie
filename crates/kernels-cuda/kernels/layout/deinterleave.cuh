//===-- deinterleave.cuh - the packed-bank splits and the row concat --===//
//
// Seven `__global__` templates, six of which a row instantiates.
// `deinterleave.cu` includes this file and keeps its seven launchers,
// so exactly ONE definition of each kernel exists in the tree -- a split and
// not a copy. `norm/altup_aux` shipped a release with two definitions of six
// kernels; they agreed on the day they were written, each stayed right for
// whichever half of the tests exercised it, and the drift was invisible until
// one half was edited.
//
// # What `T` means, and what it cost to make it mean that
//
// The element type, and nothing else -- `elementwise.cuh`'s rule. Every one
// of these is a pure copy or a pure interleave, so nothing here converts to
// float and `Elem<T>` is not needed; a `T` that is two bytes wide is a `T`
// these kernels move correctly. The ahead-of-time build named them `_bf16`
// because it had to pick its instantiations at build time and every
// instantiation cost a translation unit's worth of `cicc`. Under the JIT a
// second numeric format costs a ROW, which is the measurement
// `norm_device.rs` records for `residual_add_f16`.
//
// # The extents that vanished
//
// `deinterleave_rows` and `concat_rows` used to take a row count and guard on
// it. Both are gone: `LaunchRule::RouteRows` IS "one block per row", so the
// grid covers exactly the rows and a bound check on `blockIdx.x` is a test
// the launch rule already made. `altup_aux.cuh` states the same thing about
// `compute_rms`. The ahead-of-time launchers stopped passing them and did not
// otherwise change -- they launched `<<<I, ...>>>` and `<<<N, ...>>>` already,
// so the guard could never fire.
//
// `deinterleave_vec` KEPT its `I`, because `LaunchRule::Elementwise` rounds
// the element count up to a block and the tail threads have to be told to
// stop. The distinction is the one `norm::tanh` records: an extent a rule
// recovers is not an operand, and an extent a rule ROUNDS is.
//
// # The two that were not templates, and are now
//
// `split_q_gate` and `repeat_interleave_heads` state a `dim3(rows, heads)`
// grid at `head_dim < 128 ? 64 : 128`, and this header used to record that no
// `LaunchRule` spelled a head axis. Four of them do now:
// `Rule::PerHeadElementwise` is `grid [rows, q_heads]` at
// `clamp(head_dim, 32, 128)`, which is these launchers' axes in these
// launchers' order and their block at every width they choose -- 128 at 128
// and above, `head_dim` itself below it, and the clamp's 32 under a head
// narrower than a warp, where the surplus lanes fail `i < head_dim` on the
// first iteration and this file declares no shared array for them to touch.
//
// So the geometry stopped being the blocker and being NAMEABLE became it:
// `DeviceKernel::instantiation()` emits `path<Elem>`, one type argument, and
// a plain `__global__` answers `nvrtcGetLoweredName` with nothing. Both are
// templates now, for the same reason and at the same cost as their five
// siblings -- `T` is the element type, both are pure copies, and the
// launchers instantiate them at `bf16` and emit what they emitted.
//
// Only `split_q_gate` carries a row. `repeat_interleave_heads_bf16` is
// declared in `deinterleave.hpp`, defined in `deinterleave.cu`, and called
// from NOWHERE -- no model text, no driver-internal table row, no sibling
// `.cu` -- so a row for it would be a contract naming a caller that does not
// exist, which is the refusal `layout/geometry.cuh`'s two kernels already
// carry. The template is worth having anyway: the day a caller appears the
// row costs a line and no C++.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

/// gpt-oss packs gate and up ROW BY ROW, so splitting them is a parity
/// deinterleave and not a slice.
///
/// No row count and no bound check on `blockIdx.x`: `LaunchRule::RouteRows` is
/// one block per row, so the grid covers the rows exactly and a guard would
/// test what the rule already promised.
template <class T>
__global__ void deinterleave_rows(
    const T* __restrict__ fused,
    T* __restrict__       gate_out,
    T* __restrict__       up_out,
    int H)
{
    const int row = blockIdx.x;
    const T* gate_src = fused + (2 * row    ) * H;
    const T* up_src   = fused + (2 * row + 1) * H;
    T* gate_dst = gate_out + row * H;
    T* up_dst   = up_out   + row * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        gate_dst[j] = gate_src[j];
        up_dst[j]   = up_src[j];
    }
}

/// The flat form of the split: one thread per output element.
///
/// `I` SURVIVES where `deinterleave_rows` lost its row count, because
/// `LaunchRule::Elementwise` rounds the element count up to a whole block and
/// the tail threads have to be told to stop. An extent a rule recovers is not
/// an operand; an extent a rule ROUNDS is.
template <class T>
__global__ void deinterleave_vec(
    const T* __restrict__ fused,
    T* __restrict__       gate_out,
    T* __restrict__       up_out,
    int I)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    gate_out[i] = fused[2 * i];
    up_out[i]   = fused[2 * i + 1];
}

/// Full attention's `q_proj` packs the query and the per-token output gate PER
/// HEAD -- `[N, heads, 2*head_dim]`, query first -- so this is strided by head
/// rather than a halves cut like `split_gate_up`.
///
/// `N` and `num_heads` stay parameters although `PerHeadElementwise` recovers
/// both: the kernel declares them and guards on them, and dropping a parameter
/// a `__global__` declares is not a shorter row, it is a `void**` one entry
/// short.
template <class T>
__global__ void split_q_gate(
    const T* __restrict__ packed,  // [N, num_heads, 2*head_dim]
    T* __restrict__ q_out,         // [N, num_heads, head_dim]
    T* __restrict__ gate_out,      // [N, num_heads, head_dim]
    int N, int num_heads, int head_dim)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    if (n >= N || h >= num_heads) return;

    const int twod = 2 * head_dim;
    const T* row = packed + ((long long)n * num_heads + h) * twod;
    T* q_row     = q_out   + ((long long)n * num_heads + h) * head_dim;
    T* gate_row  = gate_out + ((long long)n * num_heads + h) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_row[i]    = row[i];
        gate_row[i] = row[head_dim + i];
    }
}

/// `[N, left] ++ [N, right] -> [N, left+right]`, one block per row.
template <class T>
__global__ void concat_rows(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    int left_dim, int right_dim)
{
    const int n = blockIdx.x;
    const int total_dim = left_dim + right_dim;
    const T* l = left + (long long)n * left_dim;
    const T* r = right + (long long)n * right_dim;
    T* o = out + (long long)n * total_dim;
    for (int i = threadIdx.x; i < total_dim; i += blockDim.x) {
        o[i] = (i < left_dim) ? l[i] : r[i - left_dim];
    }
}

/// Qwen's GDN bank: `[N, 2*v_h] -> [N, v_h] x 2`, halves rather than parity.
template <class T>
__global__ void split_qwen_gdn_ba(
    const T* __restrict__ ba,
    T* __restrict__ b_out,
    T* __restrict__ a_out,
    int v_h)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* ba_row = ba + (long long)n * (2 * v_h);
    T* b_row = b_out + (long long)n * v_h;
    T* a_row = a_out + (long long)n * v_h;
    for (int i = tid; i < v_h; i += blockDim.x) {
        b_row[i] = ba_row[i];
        a_row[i] = ba_row[v_h + i];
    }
}

/// The inverse of `concat_rows`: one packed row out to two.
template <class T>
__global__ void split_rows(
    const T* __restrict__ src,
    T* __restrict__ left,
    T* __restrict__ right,
    int left_dim, int right_dim)
{
    const int n = blockIdx.x;
    const int total = left_dim + right_dim;
    const T* row = src + (long long)n * total;
    T* l = left + (long long)n * left_dim;
    T* r = right + (long long)n * right_dim;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        if (i < left_dim) {
            l[i] = row[i];
        } else {
            r[i - left_dim] = row[i];
        }
    }
}

/// A grouped-query broadcast: `[N, kv_heads, head_dim]` read out over
/// `[N, q_heads, head_dim]`, each key head repeated `q_heads / kv_heads`
/// times.
///
/// Templated for the reader and for the day a caller exists. Nothing in the
/// tree calls `repeat_interleave_heads_bf16` -- see this file's header -- so
/// it carries no row, and the launcher below is its only instantiation.
template <class T>
__global__ void repeat_interleave_heads(
    const T* __restrict__ in,
    T* __restrict__ out,
    int N, int kv_heads, int q_heads, int head_dim)
{
    const int n = blockIdx.x;
    const int qh = blockIdx.y;
    if (n >= N || qh >= q_heads) return;
    const int repeat = q_heads / kv_heads;
    const int kh = qh / repeat;
    const T* src =
        in + ((long long)n * kv_heads + kh) * head_dim;
    T* dst =
        out + ((long long)n * q_heads + qh) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace pie::layout
