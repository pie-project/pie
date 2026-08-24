// The routed projection against a DENSE expert bank.
//
// `quant/qmv.metal`'s `qmv_routed` and `qmv_routed_bias` already do this
// against a QUANTIZED one, and they are where a gpt-oss or a Qwen3-MoE
// checkpoint lands: the bank is codes plus a scale plane, the dot product is
// a dequantise-and-accumulate, and the whole kernel is shaped around how many
// packs a lane can hold. `Moe::matmul_select` declares its bank
// `Const<Tensor<T>>` -- ONE address at the element the activation is in --
// so none of that applies and none of it should be paid for. This is the
// same routing over an ordinary bf16 stack.
//
// The bank is `[E, N, K]` with the EXPERT ON AXIS 0, so expert `e`'s row `r`
// begins at `(e * N + r) * K` and the expert axis needs no stride of its own.
// Folding the expert into the element offset instead is the classic way to
// read expert 0's weights for every expert, and it produces text.
//
// One simdgroup per output row, 32 lanes striding the reduction, one
// `simd_sum` at the end. That is `pie::moe::moe_decode_gemv_body`'s shape and
// it is chosen for the same reason: at one row per route the projection reads
// a whole `[N, K]` slice per route and is a bandwidth problem before it is an
// arithmetic one, so what matters is that the 32 lanes of a simdgroup read 32
// contiguous elements and nothing else.
//
// NO VECTORISED TWIN. cuda's reads `float4` per lane and this one could read
// `bfloat4`; whether that is worth a second entrypoint is a measurement, and
// the machine this was written on has no Metal device to take it on. A
// vectorised path landed on a guess is a second body to keep agreeing with
// the first for a speedup nobody has seen.

#include <metal_stdlib>

using namespace metal;

/// `y[route, n] = Σ_k bank[routes[route], n, k] · x[row(route), k]`.
///
/// THE ROUTE IS THE OUTPUT ROW. `y` is one row per (token, slot) pair in the
/// order the router chose them, which is what `Moe::matmul_select` states with
/// `y = [per(routes), bank.axis(1)]`.
///
/// The ACTIVATION's row is not the route's, and the two strides are how a
/// caller says which. A gate or up projection reads the one shared row a
/// token's norm produced, so its slot stride is 0 and every slot of a token
/// reads the same activation; a down projection reads the `[rows, k, I]`
/// stack the activation before it wrote, so its slot stride is `I` and its
/// row stride is `k * I`. Reading slot 0 for every expert is not a crash --
/// it is k copies of the first expert's activation, which survives all the
/// way to a plausible wrong token.
///
/// A NEGATIVE ID ZEROES THE ROW rather than returning early. `routes` is a
/// device read and every lane of the simdgroup reads the SAME element of it
/// (the route is `gid.y`, which is uniform across a simdgroup here), so the
/// branch really is uniform and an early return would be safe. Writing zero
/// is what makes the result defined: the fold that follows multiplies this
/// row by a weight and adds it, and an untouched row is whatever the arena
/// held, which in bf16 can be inf.
[[kernel]] void select_gemv(
    const device bfloat* x    [[buffer(0)]],
    const device bfloat* bank [[buffer(1)]],
    const device int* routes  [[buffer(2)]],
    device bfloat* y          [[buffer(3)]],
    const constant uint& in_width      [[buffer(4)]],
    const constant uint& out_width     [[buffer(5)]],
    const constant uint& slots_per_row [[buffer(6)]],
    const constant uint& x_row_stride  [[buffer(7)]],
    const constant uint& x_slot_stride [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint out_row = gid.x >> 5;
  const uint route = gid.y;
  const size_t at = size_t(route) * size_t(out_width) + size_t(out_row);
  const int e = routes[route];
  if (e < 0) {
    if (lane == 0) y[at] = bfloat(0);
    return;
  }

  const device bfloat* w =
      bank + (size_t(uint(e)) * size_t(out_width) + size_t(out_row)) * size_t(in_width);
  const uint k = slots_per_row < 1u ? 1u : slots_per_row;
  const device bfloat* a =
      x + size_t(route / k) * size_t(x_row_stride) + size_t(route % k) * size_t(x_slot_stride);

  float acc = 0.0f;
  for (uint i = lane; i < in_width; i += 32u) {
    acc += float(w[i]) * float(a[i]);
  }
  acc = simd_sum(acc);
  if (lane == 0) y[at] = static_cast<bfloat>(acc);
}
