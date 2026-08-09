#pragma once
// decode_dispatch.hpp — the decode DAG's launch geometry, which now lives with
// the kernels that define it.
//
// Every shape below was read off a kernel's `[[thread_position_in_grid]]`
// contract, so it is the KERNEL's knowledge and moved to
// `crates/kernels-metal/kernels/<family>.h` with §7 step 3 of
// .wiki/kernel-metal-refactor.md. This header is what keeps the call sites
// reading: `pie::metal::qmv_dispatch(...)` is still `pie::metal::qmv_dispatch`.
//
// Nothing here is a decision. When a caller needs one -- which tile, whether to
// batch, whether to tile the paged attention -- that stays in this driver,
// because it reads `DeviceTuning`. See `decode_dispatch_mb.hpp`.

#include "decode_abi.hpp"
#include "mtl4_context.hpp"  // Grid, Threadgroup

#include "pie/kernels/attn.h"
#include "pie/kernels/layout.h"
#include "pie/kernels/mlp.h"
#include "pie/kernels/norm.h"
#include "pie/kernels/quant.h"
#include "pie/kernels/rope.h"

namespace pie::metal {

using pie::kernels::attn::attn_gate_dispatch;
using pie::kernels::attn::kv_append_dispatch;
using pie::kernels::attn::q_split_dispatch;
using pie::kernels::attn::sdpa_dispatch;
using pie::kernels::layout::embed_dispatch;
using pie::kernels::mlp::silu_mul_dispatch;
using pie::kernels::norm::gated_rms_dispatch;
using pie::kernels::norm::residual_dispatch;
using pie::kernels::norm::rms_dispatch;
using pie::kernels::quant::qmv_dispatch;
using pie::kernels::rope::rope_dispatch;

}  // namespace pie::metal
