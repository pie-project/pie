// Compact the rows a fire will actually SAMPLE.
//
//   out[i, :] = in[rows[i], :]      for i in [0, count)
//
// A prefill computes every row of the model body, but only one row per request
// is read: the last one, whose logits the sampler turns into the next token.
// Running the LM head over all of them costs `hidden * vocab` per wasted row --
// on gemma4 that is 1536x262144, two orders of magnitude more than the layer it
// follows -- and allocates a `[rows, vocab]` buffer to throw away.
//
// So the tail runs on the gathered rows instead: this copies them dense, and
// everything after it is `[count, *]` rather than `[rows, *]`.
//
// bind::RowGather = { In=0, Out=1, Rows=2, Params=3 }.

#include <metal_stdlib>
using namespace metal;

struct RowGatherParams {
  uint width;  // elements per row
  uint count;  // rows to gather
};

template <typename T>
[[kernel]] void row_gather(
    const device T* in            [[buffer(0)]],
    device T* out                 [[buffer(1)]],
    const device uint* rows       [[buffer(2)]],
    constant RowGatherParams& p   [[buffer(3)]],
    uint2 tid                     [[thread_position_in_grid]]) {
  const uint c = tid.x;
  const uint i = tid.y;
  if (c >= p.width || i >= p.count) return;
  out[size_t(i) * size_t(p.width) + size_t(c)] =
      in[size_t(rows[i]) * size_t(p.width) + size_t(c)];
}

#define instantiate_row_gather(name, itype)                \
  template [[host_name("row_gather_" #name)]]              \
  [[kernel]] void row_gather<itype>(                       \
      const device itype*, device itype*,                  \
      const device uint*, constant RowGatherParams&, uint2);

instantiate_row_gather(bfloat16, bfloat)
