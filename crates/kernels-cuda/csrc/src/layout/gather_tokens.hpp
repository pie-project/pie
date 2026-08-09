#pragma once

// `gather_tokens`: the driver-side op behind `compact`. Packs
// live token runs densely into fresh page slots per a host-given plan, so the
// working set reclaims token-space waste (frozen fork tails, H2O-style
// eviction). A STANDALONE streaming-copy kernel, NOT an attention-kernel
// modification (open question 5).
//
// The default KV page layout is NHD `[num_pages, page_size, num_kv_heads,
// head_dim]`, so a run of `len` consecutive tokens WITHIN a page is a single
// CONTIGUOUS span (`len · num_kv_heads · head_dim` elements). `compact` already
// splits any run that would straddle a destination page boundary, so every op
// here copies a contiguous src span to a contiguous dst span — the op is a
// batched device-to-device memcpy, targeting `cudaMemcpy` bandwidth.

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::layout {

// `GatherTokenOp` is DEFINED in `layout/gather_tokens.cuh`, beside the two
// `__global__`s that read it -- a kernel parameter type has to be visible to
// whatever compiles the kernel, and NVRTC cannot open this file: it ships
// neither `<cstdint>` nor `<cuda_runtime.h>`, and the `stdlib_probe` measured
// zero of thirty-one standard headers answering.
//
// So this file forward-declares and does not restate. It is compiled by a
// PLAIN host C++ compiler -- `build.rs` generates `shim.cpp`, which includes
// every family header and knows nothing of `__global__` -- so it may not
// include the device header, and an incomplete type is all a `const
// GatherTokenOp*` parameter needs. A second definition spelled
// `std::uint32_t` would be the same struct until someone reordered a field,
// and then every op in the plan would be read wrong by one half of the tree.
namespace device {
struct GatherTokenOp;
}
using device::GatherTokenOp;

// Pack the plan `ops` densely for ONE layer's paged K/V (bf16, NHD). `k_pages`
// and `v_pages` are `[num_pages, page_size, num_kv_heads, head_dim]` bf16 (as
// `std::uint16_t`). Both K and V are copied per op. Safe to run on the copy
// stream off the decode path (the copies ride behind the grace period). A
// per-layer call; the caller loops layers (or batches them via `num_layers` +
// `layer_stride_elems` below).

// Multi-layer variant: `k_pages`/`v_pages` point at layer 0; layer L starts at
// `layer_stride_elems * L` elements (typically `num_pages · page_size ·
// num_kv_heads · head_dim`). One launch copies every op for every layer.

}  // namespace pie_cuda_driver::kernels::layout
