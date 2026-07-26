#pragma once

/// The driver's half of the load contract (architecture.md §10.2.1, §12 row
/// 7b-4b).
///
/// Every family binder already knows exactly which tensors it will bind and what
/// shape each one has — `llama_like`'s `must(engine, p + "self_attn.o_proj.weight")`
/// and its `Hq = num_attention_heads * head_dim` are that knowledge, spelled
/// imperatively and read *after* the load. Because it is only ever expressed
/// after the fact, nothing checks it against the plan: an `arch/` pass that
/// invents an output name no binder asks for, or computes a shape the binder
/// disagrees with, fails at bind time with a missing-weight error, or worse
/// binds a wrongly-shaped buffer and produces garbage.
///
/// A `TensorContract` is the same knowledge stated *before* the load, so
/// `pie_loader_verify` compares two independently-derived answers instead of
/// comparing the plan with itself.
///
/// # Declaring less than you know is fine
///
/// `demand(name)` with no shape asks only that the tensor exist. That is the
/// honest declaration for a weight whose runtime layout the binder reads off the
/// loaded tensor rather than predicting (a repacked Marlin or MXFP4 buffer, for
/// instance). Encoding is deliberately not declarable: which quantization a
/// weight lands in is the *loader's* decision under the runtime quant policy,
/// and binders probe `->dtype()` afterwards.
///
/// # Ownership
///
/// The builder owns its names and shapes, and `slice()` lends a borrowed POD
/// view of them. The contract must outlive the `pie_loader_compile` /
/// `pie_loader_verify` call that reads the slice, and must not be mutated while
/// a slice is outstanding — `demand` may reallocate.

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "pie_loader.h"

namespace pie_loader {

class TensorContract {
public:
    /// Demand a tensor whose per-rank shape the driver knows.
    ///
    /// `shape` is what *this rank* expects, already divided by the TP size: a
    /// column-sharded projection on rank 1 of 4 declares `hidden / 4`, not
    /// `hidden`. Declaring the model-wide shape would report a mismatch on
    /// every sharded weight.
    TensorContract& demand(std::string name, std::vector<std::int64_t> shape) {
        return push(std::move(name), std::move(shape), false);
    }

    /// Demand only that a tensor exist.
    TensorContract& demand(std::string name) { return push(std::move(name), {}, false); }

    /// Demand a tensor whose absence the binder tolerates.
    ///
    /// The motivating case is `lm_head.weight` under `tie_word_embeddings`,
    /// where the binder falls back to the embedding table. Without this, a
    /// correct tied-embedding load would be reported as a violation.
    TensorContract& demand_optional(std::string name, std::vector<std::int64_t> shape = {}) {
        return push(std::move(name), std::move(shape), true);
    }

    /// How many tensors were demanded.
    std::size_t size() const { return demands_.size(); }
    bool empty() const { return demands_.empty(); }

    /// A borrowed view for [`PieLoaderRequest::demands`]. Invalidated by any
    /// later `demand`.
    pie_loader::PieLoaderTensorDemandSlice slice() const {
        return {demands_.empty() ? nullptr : demands_.data(), demands_.size()};
    }

private:
    TensorContract& push(std::string name, std::vector<std::int64_t> shape, bool optional) {
        // `deque` rather than `vector`: the POD demands point into these, and a
        // vector would dangle every previously-pushed pointer when it grew.
        const std::string& stored_name = names_.emplace_back(std::move(name));
        const std::vector<std::int64_t>& stored_shape = shapes_.emplace_back(std::move(shape));
        demands_.push_back(pie_loader::PieLoaderTensorDemand{
            .name = {reinterpret_cast<const std::uint8_t*>(stored_name.data()),
                     stored_name.size()},
            .shape = {stored_shape.empty() ? nullptr : stored_shape.data(),
                      stored_shape.size()},
            .optional = optional,
        });
        return *this;
    }

    std::deque<std::string> names_;
    std::deque<std::vector<std::int64_t>> shapes_;
    std::vector<pie_loader::PieLoaderTensorDemand> demands_;
};

}  // namespace pie_loader
