#pragma once

// Memoized `mx::compile` harness for fused decode regions.
//
// `mx::compile` fuses a graph-building function into one traced+optimized kernel
// sequence, collapsing the per-op dispatch overhead that dominates batch=1
// decode (the "loose pointwise glue" -- norms, residuals, scales, gated
// activations -- around the matmuls). The compiled callable must be created
// ONCE and replayed; this harness owns that lifecycle so model graphs request a
// compiled region without managing cache state (the ops/ vs model/ boundary:
// arch logic stays in the caller's region fn, compile mechanism + cache here).
//
// ┌─ CRITICAL: the region fn MUST be a FUNCTION POINTER (a capture-less lambda
// │  or named free function -- a capture-less lambda converts to a fn ptr
// │  automatically). Do NOT pass a *capturing* lambda: MLX's std::function
// │  compile path crashes on prefill/decode shape alternation (verified). So
// │  constants cannot be captured -- pass them as inputs or bake them per-fn:
// │    * arrays (weights, activations)  -> via `inputs` (replayed across layers;
// │      identical shapes => clean cache hits, distinct weight data is fine).
// │    * scalar runtime values (scale, softcap) -> pass as 1-element scalar
// │      `Tensor` inputs and read them as arrays inside the region.
// │    * float constants a fused op needs as a C++ float (e.g. rms_norm eps)
// │      -> write them as literals in the region body (capture-less). Region
// │      fns are authored per-arch, so the arch's eps is a natural literal.
// │    * discrete variants (e.g. SiLU vs GELU-tanh activation) -> use a
// │      SEPARATE region fn per variant. Different fns => different compiled
// │      instances, which is exactly the desired "constant baked into identity".
// └─ The region fn MUST be host-readback-free: no `.eval()`, `.item()`,
//    `to_host`, or control flow on array *contents* (breaks compile capture).
//
// `mx::compile` keys internally on input shapes, so one cached instance per
// region transparently retraces across decode/prefill shapes.
//
// Usage (from a per-arch model graph layer fn):
//
//   // gemma4 FFN core: pre-norm -> gate/up -> geglu-tanh -> down (eps literal).
//   static std::vector<Tensor> gemma4_ffn(const std::vector<Tensor>& in) {
//       Tensor n = ops::rms_norm(in[0], in[1], 1e-6f, /*plus_one=*/false);
//       Tensor g = ops::linear(in[2], n);
//       Tensor u = ops::linear(in[3], n);
//       return { ops::linear(in[4], ops::geglu(g, u, /*tanh_approx=*/true)) };
//   }
//   Tensor ffn = ops::compiled("gemma4.ffn",
//       {hidden, ffn_norm_w, gate_w, up_w, down_w}, gemma4_ffn)[0];

#include <string>
#include <vector>

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// A compiled region: a pure, host-readback-free graph builder taking the region
// inputs (in declaration order) and returning its outputs. Must be a function
// pointer (named fn or capture-less lambda) -- see the header note above.
using CompiledRegion = std::vector<Tensor> (*)(const std::vector<Tensor>&);

// Trace+cache (first call per region fn) then replay a compiled region. `key`
// is a human-readable label for diagnostics; the cache is keyed on the region
// fn pointer (so distinct variants -- distinct fns -- get distinct instances).
std::vector<Tensor> compiled(const std::string& key,
                             const std::vector<Tensor>& inputs,
                             CompiledRegion fn);

}  // namespace pie_metal_driver::ops
