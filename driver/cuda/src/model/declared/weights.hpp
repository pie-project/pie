#pragma once

// The declared executor's WEIGHT VOCABULARY — shared by every family.
//
// A traced op names its weight ("layer.3.attn_norm", "final_norm"); which
// DEVICE TENSOR that name is, is the family's contract and nothing else.
// This header holds the two halves of that sentence:
//
//   * the name GRAMMAR, which is `forward/src/family.rs`'s and identical
//     everywhere (it was copied byte-for-byte into each family executor);
//   * the one indirection an interpreter arm may use — a BINDER the family
//     supplies, answering name -> tensor.
//
// Why it matters beyond de-duplication: an arm that reaches into a family
// weights struct (`layer.attn_norm` vs `layer.attn_norm_pre` — the same
// traced name, two field names) is an arm that CANNOT be shared, and that
// is how one executor became two. Behind a binder the arms stop knowing
// which family they serve, which is the precondition for one interpreter.

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>

#include "tensor.hpp"

namespace pie_cuda_driver::model::declared {

// A plan weight name split into its layer index and field: "layer.3.qkv" →
// {3, "qkv"}; prologue/epilogue names ("embed", "final_norm") keep layer -1.
// Anything the parse cannot place throws, loudly, because a name the
// resolver does not know means the trace and this executor have drifted.
struct ParsedWeightName {
    int layer = -1;
    std::string_view field;
};

[[noreturn]] inline void throw_unknown_weight(std::string_view name) {
    throw std::runtime_error(
        "declared forward: unknown weight name '" + std::string(name) +
        "' (trace vocabulary is forward/src/family.rs's)");
}

inline ParsedWeightName parse_weight_name(std::string_view name) {
    constexpr std::string_view prefix = "layer.";
    if (name.substr(0, prefix.size()) != prefix) {
        return ParsedWeightName{-1, name};
    }
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos) throw_unknown_weight(name);
    int layer = -1;
    const char* first = name.data() + prefix.size();
    const char* last = name.data() + dot;
    const auto [ptr, ec] = std::from_chars(first, last, layer);
    if (ec != std::errc() || ptr != last || layer < 0) {
        throw_unknown_weight(name);
    }
    return ParsedWeightName{layer, name.substr(dot + 1)};
}

// The family's half: a traced name, resolved against ITS weights. A plain
// function pointer + context rather than std::function — this sits on the
// per-op path, and the call it replaces was a chain of string compares.
struct WeightBinder {
    using Fn = const DeviceTensor* (*)(const void* ctx,
                                       const ParsedWeightName& parsed,
                                       std::string_view name);

    Fn fn = nullptr;
    const void* ctx = nullptr;

    // Null when the family knows the name but the checkpoint left it
    // unbound (an optional bias, a fusion the deployment did not take).
    const DeviceTensor* find(std::string_view name) const {
        return fn(ctx, parse_weight_name(name), name);
    }

    // Same question asked positionally: "this layer's `gate_proj`". The
    // trace states ONE packed name (`gate_up`) and the driver decides
    // whether the binding materialised it fused; the split halves are then
    // asked for by field, with the traced name kept only for the throw.
    const DeviceTensor* find_field(int layer, std::string_view field,
                                   std::string_view name) const {
        return fn(ctx, ParsedWeightName{layer, field}, name);
    }

    const DeviceTensor& require_field(int layer, std::string_view field,
                                      std::string_view name) const {
        const DeviceTensor* tensor = find_field(layer, field, name);
        if (tensor == nullptr) {
            throw std::runtime_error(
                "declared forward: weight '" + std::string(name) +
                "' needs this family's '" + std::string(field) +
                "', which is not bound");
        }
        return *tensor;
    }

    // The arms' form: a name the trace STATED must be bound, and a missing
    // one names itself in the throw.
    const DeviceTensor& require(std::string_view name) const {
        const DeviceTensor* tensor = find(name);
        if (tensor == nullptr) {
            throw std::runtime_error(
                "declared forward: weight '" + std::string(name) +
                "' is named by the trace but not bound");
        }
        return *tensor;
    }
};

}  // namespace pie_cuda_driver::model::declared
