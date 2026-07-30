#pragma once

/// What Gemma-4 binds (`model/registry.cpp` rows).
///
/// The only family the encode pipeline can scope a load to. An encode-scoped
/// rank declares the vision and audio towers and nothing else, so it never
/// allocates the language model — and because that is a *declaration* rather
/// than a filter applied to a finished contract, the plan it compiles has no
/// trace of the tensors it skipped.

#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "model/contract.hpp"

namespace pie_cuda_driver::model {

/// Fold `1/sqrt(H)` into Gemma-4's per-layer router scale.
///
/// The router pipeline is `rmsnorm_no_scale(x) * scale * (1/sqrt(H))` followed
/// by a linear. Multiplying the two constants together at load time lets the
/// forward collapse the first three steps into one rmsnorm-with-weight call,
/// and `scale` is a `[H]` vector so the fold costs nothing to store.
///
/// Bind used to do this by hand: copy the vector to the host, walk it
/// reinterpreting each bf16 as fp32, multiply, and upload the result into a
/// `DeviceTensor` the weights struct then had to own and keep alive. That is
/// arithmetic the plan cannot see, so it was re-done on every load, it had to
/// be kept out of the cache key by hand, and the unscaled original stayed
/// resident beside the scaled copy for the process's lifetime. Said here, it is
/// one `Scale` node the loader schedules, checks and caches like anything else.
///
/// `H` is read off the tensor rather than off the config: the fold has to use
/// the same `H` the forward's rmsnorm does, and that one is this vector's
/// length by construction.
///
/// **Rounding.** The kernel rounds to nearest-even; the host loop it replaced
/// truncated (`f32_bits >> 16`), biasing every element down by up to one bf16
/// ULP. The new result is strictly closer to the exact product, but it is not
/// bit-identical, and a near-tie in top-k expert selection can land on the
/// other expert. Any bit-exact Gemma-4 MoE baseline recorded before this needs
/// re-recording.
///
/// Scoped to `decoder_layer_prefix()` for the reason the fused-projection pass
/// is: Gemma-4 nests its decoder under the vision and audio towers, and a
/// suffix match alone would fold a tower's tensor of the same name.
inline void gemma4_fold_router_scale(ContractBuilder& b) {
    constexpr std::string_view kSuffix = ".router.scale";
    // `tensors()` holds source names, so the bound prefix has to be mapped
    // through `source_name` the way the fused-projection pass does.
    const std::string layers = b.source_name(b.decoder_layer_prefix());
    for (const SourceTensor& raw : b.tensors()) {
        if (!contract_detail::starts_with(raw.name, layers) ||
            !contract_detail::ends_with(raw.name, kSuffix)) {
            continue;
        }
        const std::vector<std::int64_t> shape = contract_detail::shape_of(raw);
        if (shape.size() != 1 || shape[0] <= 0) {
            continue;
        }
        // Computed exactly as the forward would have: fp32 `sqrt`, then a
        // reciprocal. Reordering this to `rsqrtf` or to fp64 would change the
        // last bit of every router logit.
        const float inv_sqrt_h = 1.f / std::sqrt(static_cast<float>(shape[0]));
        b.define(b.output_name(raw.name),
                 b.contract().scale(b.contract().src(std::string(raw.name)), inv_sqrt_h),
                 raw.encoding, shape);
        b.consume(raw.id);
    }
}

/// gemma4, gemma4_text.
inline void author_gemma4_contract(ContractBuilder& b) {
    b.allow_encode_scope();
    // The decoder is nested; the vision and audio towers are not, and they
    // have `self_attn.q_proj.weight` of their own.
    b.decoder_layer_prefix("model.language_model.layers.");
    gemma4_fold_router_scale(b);
    author_dense_contract(b);
}

}  // namespace pie_cuda_driver::model
