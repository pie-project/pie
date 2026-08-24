// The DENSE token-embedding gather: `y[n, :] = table[ids[n], :]`.
//
// This is `layout.embed` at a `dense` bank, which is the seed statement of
// every tower — the one point in the whole declaration floor whose result is
// sized without reading an operand's rectangle (`[fire, table.axis(1)]`), and
// therefore the only thing that can start one.
//
// # This is NOT `embed_gather.metal`
//
// That file gathers a row out of a TIED 4-BIT table and affine-dequantizes it,
// and it takes three operands this statement does not carry (codes, scales,
// biases). While `layout.embed` was unclaimed, `kernels_metal::CANON` routed
// the claim to it BY NAME and the walk then refused at the fire, because the
// staging between `embed(ids, table, vocab, y)` and
// `layout::embed_gather_mb_4bit` does not exist. This file is the other half
// of that pair: the bf16 table, gathered. The CANON row is now unreachable —
// `model_compiler::program::call_for` asks the claim table first — and
// `driver-metal`'s `the_walk_refuses_a_canon_symbol_by_name` is the test that
// pinned the refusal it replaces.
//
// # The vocab clamp is not decoration
//
// `row = (raw >= 0 && raw < vocab) ? raw : 0`, the same guard
// `kernels-cuda/kernels/layout/embed.cuh` states and for the same reason: the
// ids arrive from a wire payload, and an out-of-range one is an OOB read into
// the largest tensor in the model rather than a wrong answer. Out of vocab
// reads row 0.
//
// # Not vectorised, and cuda's header says what that costs
//
// The cuda kernel gathers eight bf16 through one 16-byte load when `hidden % 8
// == 0` and both pointers are 16-byte aligned, because the row it reads is a
// random offset into that largest tensor and the access is a cold TLB miss
// whose latency only a wide grid hides. The same widening is open here and is
// deliberately not taken yet: the alignment test is a host-side question this
// plane has no `Source` for, and an unmeasurable widening on a plane with no
// device to measure it on is a guess. One thread per element, one grid over
// `[hidden, rows]`.
//
// Launch: dispatchThreads grid=(hidden, rows, 1), tg=(256, 1, 1) —
// `elementwise_rows`, one thread per output element.
//
// # UNVERIFIED
//
// Written without a Metal toolchain or an Apple device. Never compiled, never
// run, no number compared against anything.

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void embed(
    const device int* ids      [[buffer(0)]],  // [rows]
    const device T* table      [[buffer(1)]],  // [vocab, hidden]
    device T* y                [[buffer(2)]],  // [rows, hidden]
    const constant int& hidden [[buffer(3)]],
    const constant int& vocab  [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int c = int(tid.x);
  // The grid is rounded up to whole threadgroups, so the tail runs over the
  // end of a row. Metal does not report that; it reads the next row.
  if (c >= hidden) {
    return;
  }
  const size_t n = size_t(tid.y);
  const int raw = ids[n];
  const int row = (raw >= 0 && raw < vocab) ? raw : 0;
  y[n * size_t(hidden) + size_t(c)] =
      table[size_t(row) * size_t(hidden) + size_t(c)];
}

#define instantiate_embed_dense(name, itype)                                \
  template [[host_name("embed_" #name)]]                                    \
  [[kernel]] void embed<itype>(                                             \
      const device int*, const device itype*, device itype*,                \
      const constant int&, const constant int&, uint2);

instantiate_embed_dense(bfloat16, bfloat)
