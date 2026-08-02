#pragma once

/// The facts and policy vocabulary the boot sends across the request
/// boundary — all that remains of `model/contract.hpp` after the harvest.
///
/// The builder, the family authors and the fourteen-node authoring DSL that
/// used to live beside these types are gone: authoring happens on the far
/// side of `pie_loader_compile_model` (`plan/model-in-rust.md` §2), proven
/// byte-equal to the deleted C++ authors by the §8-3 differential — 17
/// synthetic cases, ten real checkpoints, and a dual boot on real hardware
/// whose plans matched byte for byte. What the driver still owns is what
/// only it can know: the facts it parsed and the policy it decided.

#include <cstdint>
#include <string>
#include <string_view>

namespace pie_cuda_driver::model {

/// Which half of a multimodal checkpoint to load.
///
/// A driver concept, not a loader one: a contract that should not load the
/// vision tower simply does not declare it, and the *author* decides what to
/// declare.
enum class Component : std::uint32_t {
    Full = 0,
    Text = 1,
    Encode = 2,
};

/// How this driver wants MXFP4 expert weights lowered.
enum class Mxfp4MoePolicy : std::uint32_t {
    RoutedDecode = 0,
    NativeGemm = 1,
    EagerBf16 = 2,
};

/// What the caller asked for, before the device has had its say. The
/// resolution — including the per-family overrides — lives with the author,
/// and the resolved [`Mxfp4MoePolicy`] comes back through
/// `pie_loader_compile_model`'s out-parameter.
enum class Mxfp4MoeRequest : std::uint32_t {
    Auto = 0,
    RoutedDecode = 1,
    NativeGemm = 2,
    EagerBf16 = 3,
};

/// The config facts a family needs beyond the checkpoint itself.
///
/// A short list, and it grows only when a *partition* of a shape has to be
/// known that the checkpoint does not record. Everything else — which
/// tensors exist, what shape they are, how they are encoded — is in the
/// checkpoint, and the author reads it there.
struct ModelFacts {
    std::string model_type;
    /// `quantization_config.quant_method`, empty for an unquantized checkpoint.
    std::string quant_method;
    std::uint32_t num_hidden_layers = 0;
    std::uint32_t num_experts = 0;
    std::uint32_t head_dim = 0;
    /// `mamba_n_groups`, zero for a family without a Mamba mixer.
    std::uint32_t mamba_groups = 0;
};

/// Resolve the caller's runtime-quantization request against what the device
/// can execute.
///
/// A request the device cannot run is dropped rather than refused: it is a
/// preference, and the checkpoint's own format is always a valid answer.
/// `fp8_native` is a fact this driver measured off `cudaDeviceProp`, so this
/// driver is what applies it.
inline std::string_view resolve_runtime_quant(std::string_view request, bool fp8_native) {
    return request == "fp8" && !fp8_native ? std::string_view() : request;
}

}  // namespace pie_cuda_driver::model
