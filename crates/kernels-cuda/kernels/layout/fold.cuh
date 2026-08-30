#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

/// **THE POOLING FOLD: `side²` ROWS AVERAGED INTO ONE**
/// (`.wiki/alto/multimodal.md` §6.5, §7.4).
///
/// `y[j] = (sum over r in [j*block, (j+1)*block) of x[r]) / block`, where
/// `block = side * side`. One block of threads per OUTPUT row, striding the
/// width; the accumulation is f32 whatever the element is, because
/// `Gemma4VisionPooler` pools through an f32 matmul and a bf16 running sum
/// over nine rows would round nine times.
///
/// **WHY A 2-D POOL IS A 1-D REDUCTION HERE.** gemma4 averages each `k x k`
/// square of the patch GRID. That is a row reduction exactly when the
/// submission orders an image's patches so each square is contiguous — which
/// is the statute §2 already makes for qwen's 2x2 spatial merge
/// ("merge-block-major patch ordering in the submission"), at `k` instead of
/// 2. So this kernel asks the submission for the ordering and asks the fire
/// for nothing: no position stream, no image indptr, no grid width. The
/// alternative — pooling BY position, the way `_avg_pool_by_positions` does
/// with a one-hot — materializes a `[soft_tokens, patches]` matrix to express
/// an `O(patches)` reduction, and would be a different matrix per rung.
///
/// **AND WHY IMAGE BOUNDARIES NEED NO INDPTR.** `get_aspect_ratio_preserving
/// _size` rounds an image's height and width DOWN to a multiple of
/// `pooling_kernel_size * patch_size`, so every image's patch run is a whole
/// number of `block` rows -- and a run of whole blocks laid after another run
/// of whole blocks never has a block straddling the two. Two images pool as
/// one concatenation, which is the golden's own claim.
///
/// **THE DIVISOR IS `block` AND NOT THE COUNT OF LIVE ROWS**, transcribed
/// from `_avg_pool_by_positions`: it builds `one_hot / k_squared` and the
/// padding patches it zeroed still sit in the denominator. A block that
/// contains a zeroed row therefore answers what transformers answers.
///
/// `side == 1` is the identity, and it is a real case: the pooler skips
/// itself when the patch count already equals the soft-token count.
///
/// **COMPACTING**, like `merge_rows` below: `y` holds `x.rows / block` rows
/// and the rest of the destination is untouched. What says "this row has no
/// destination" downstream is a `-1` in `patch_routes` and the scatter that
/// honours it (`layout.scatter_live_rows`, §8.6) — not a rule this kernel
/// could enforce, because it does not own the route vector.
template <class T>
__global__ void pool_rows(
    const T* __restrict__ x,
    T* __restrict__ y,
    int width,
    int block)
{
    const int out = blockIdx.x;
    const long long base = static_cast<long long>(out) * block * width;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float acc = 0.f;
        for (int r = 0; r < block; ++r) {
            acc += Elem<T>::to_f32(x[base + static_cast<long long>(r) * width + i]);
        }
        y[static_cast<long long>(out) * width + i] =
            Elem<T>::from_f32(acc / static_cast<float>(block));
    }
}


/// **THE MERGING FOLD: `side²` ROWS CONCATENATED INTO ONE**
/// (`.wiki/alto/multimodal.md` §8.1, §8.3).
///
/// qwen's spatial merger. `y[j]` is rows `[j*block, (j+1)*block)` of `x` laid
/// end to end, so `y` is `x.rows / block` rows of `block * width` — which is
/// `Qwen3_5VisionPatchMerger.forward`'s opening `x.view(-1, hidden_size *
/// spatial_merge_size**2)`, and why `merger.linear_fc1.weight` is
/// `[4*hidden, 4*hidden]` on a 768-wide tower.
///
/// **AND ON A DENSE RECTANGLE IT IS THE IDENTITY COPY**, which §2's "the
/// reshape is a view" was right about and §8.1 doubted only because it assumed
/// the row count had to survive. Row-major `[rows, width]` and row-major
/// `[rows/block, block*width]` put the same element at the same offset: this
/// kernel writes `out * block * width` elements from the same index it read
/// them at. It is a NODE rather than an alias because the IR has no way to
/// give one value two types, and a compiler that later folds it into a
/// placement alias changes nothing about what any text says.
///
/// Written as a copy rather than a `cudaMemcpyAsync` because it is a graph
/// node in a captured region, and one launch is what the walk dispatches.
template <class T>
__global__ void merge_rows(
    const T* __restrict__ x,
    T* __restrict__ y,
    int width,
    int block)
{
    const int out = blockIdx.x;
    const long long units = static_cast<long long>(block) * width;
    const T* src = x + static_cast<long long>(out) * units;
    T* dst = y + static_cast<long long>(out) * units;

    for (long long i = threadIdx.x; i < units; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}
