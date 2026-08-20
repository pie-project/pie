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
// bind::RowGather = { In=0, Out=1, Rows=2 }; Width=3 and Count=4, each a scalar
// of its own.

#include <metal_stdlib>
using namespace metal;

// TWO SCALARS, NOT A STRUCT, AND THE ORDER IS THE ABI.
//
// This took `constant RowGatherParams& p [[buffer(3)]]` out of the shared
// header `layout/row_gather_params.h`, `{ unsigned int width; unsigned int
// count; }` -- and on this plane a packed slot IS the buffer, so a trailing
// scalar landed in the same argument: the statement stated `[width]`, the
// driver appended the request count as `Ty::InPacked`, and the struct read
// `[width, count]` off the staged run with no buffer 4 in sight.
//
// Both words are ordinary scalars now. `width` is a `Const<uint>` mark on the
// routine, which is word 0 of the same statement run the struct's first field
// read, and the body passes the derived request count straight after it, so
// `driver-metal`'s `lay_out` gives each its own argument slot and its own
// `setBytes`. The order between them is what has to be right -- a swap binds
// the count as the pitch and gathers whole rows out of the wrong place, and
// both are plausible `uint`s -- so the SIGNATURE is where it is stated, once,
// rather than in a header the shader and the host each read separately.
// `layout/row_gather_params.h` is gone with the struct; `kernels-vulkan` and
// `kernels-wgpu` spell the same pair in the same order in a push block and a
// `@group(1)` uniform.

template <typename T>
[[kernel]] void row_gather(
    const device T* in            [[buffer(0)]],
    device T* out                 [[buffer(1)]],
    const device uint* rows       [[buffer(2)]],
    const constant uint& width    [[buffer(3)]],
    const constant uint& count    [[buffer(4)]],
    uint2 tid                     [[thread_position_in_grid]]) {
  const uint c = tid.x;
  const uint i = tid.y;
  if (c >= width || i >= count) return;
  out[size_t(i) * size_t(width) + size_t(c)] =
      in[size_t(rows[i]) * size_t(width) + size_t(c)];
}

#define instantiate_row_gather(name, itype)                \
  template [[host_name("row_gather_" #name)]]              \
  [[kernel]] void row_gather<itype>(                       \
      const device itype*, device itype*,                  \
      const device uint*, const constant uint&,            \
      const constant uint&, uint2);

instantiate_row_gather(bfloat16, bfloat)
