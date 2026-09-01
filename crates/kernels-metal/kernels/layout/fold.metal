#include <metal_stdlib>
using namespace metal;

/// **THE POOLING FOLD: `side^2` ROWS AVERAGED INTO ONE** — the Metal mirror
/// of `kernels-cuda`'s `::pie::layout::pool_rows`
/// (`.wiki/alto/multimodal.md` §6.5, §7.4).
///
/// `y[j] = (sum over r in [j*block, (j+1)*block) of x[r]) / block`, where
/// `block = side * side`. One thread per OUTPUT element — `tid.x` the column,
/// `tid.y` the output row — which is this plane's row grid rather than the
/// twin's block-per-row stride, and the same arithmetic either way.
///
/// **THE ACCUMULATION IS f32 WHATEVER THE ELEMENT IS.**
/// `Gemma4VisionPooler` pools through an f32 matmul, and a bf16 running sum
/// over nine rows would round nine times.
///
/// **WHY A 2-D POOL IS A 1-D REDUCTION HERE.** gemma4 averages each `k x k`
/// square of the patch GRID; that is a row reduction exactly when the
/// submission orders an image's patches so each square is contiguous, which
/// is §2's merge-block-major statute at `k` instead of 2. So this kernel asks
/// the submission for the ordering and asks the fire for nothing: no position
/// stream, no image indptr, no grid width.
///
/// **THE DIVISOR IS `block` AND NOT THE COUNT OF LIVE ROWS**, transcribed
/// from `_avg_pool_by_positions`: it builds `one_hot / k_squared` and the
/// padding patches it zeroed still sit in the denominator.
template <typename T>
[[kernel]] void pool_rows(
    const device T* x         [[buffer(0)]],
    device T* y               [[buffer(1)]],
    const constant int& width [[buffer(2)]],
    const constant int& block [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t c = size_t(tid.x);
  const size_t out = size_t(tid.y);
  const size_t base = out * size_t(block) * size_t(width);

  float acc = 0.0f;
  for (int r = 0; r < block; ++r) {
    acc += float(x[base + size_t(r) * size_t(width) + c]);
  }
  y[out * size_t(width) + c] = T(acc / float(block));
}

#define instantiate_pool_rows(name, itype)                            \
  template [[host_name("pool_rows_" #name)]]                          \
  [[kernel]] void pool_rows<itype>(                                   \
      const device itype*, device itype*, const constant int&,        \
      const constant int&, uint2);

instantiate_pool_rows(bfloat16, bfloat)

/// **THE MERGING FOLD: `side^2` ROWS CONCATENATED INTO ONE** — the Metal
/// mirror of `::pie::layout::merge_rows` (`.wiki/alto/multimodal.md` §8.1,
/// §8.3).
///
/// `y[j]` is rows `[j*block, (j+1)*block)` of `x` laid end to end, so `y` is
/// `x.rows / block` rows of `block * width` — which is
/// `Qwen3_5VisionPatchMerger.forward`'s opening
/// `x.view(-1, hidden_size * spatial_merge_size**2)`, and why
/// `merger.linear_fc1.weight` is `[4*hidden, 4*hidden]` on a 768-wide tower.
///
/// **AND ON A DENSE RECTANGLE IT IS THE IDENTITY COPY.** Row-major
/// `[rows, width]` and row-major `[rows/block, block*width]` put the same
/// element at the same offset: this kernel writes the element it read at the
/// index it read it. It is a NODE rather than an alias because the IR has no
/// way to give one value two types, and a compiler that later folds it into a
/// placement alias changes nothing about what any text says.
///
/// The grid is the DESTINATION's: `tid.x` walks the merged row's
/// `block * width` columns, `tid.y` the merged rows.
template <typename T>
[[kernel]] void merge_rows(
    const device T* x          [[buffer(0)]],
    device T* y                [[buffer(1)]],
    const constant int& merged [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(merged) + size_t(tid.x);
  y[i] = x[i];
}

#define instantiate_merge_rows(name, itype)                           \
  template [[host_name("merge_rows_" #name)]]                         \
  [[kernel]] void merge_rows<itype>(                                  \
      const device itype*, device itype*, const constant int&, uint2);

instantiate_merge_rows(bfloat16, bfloat)
