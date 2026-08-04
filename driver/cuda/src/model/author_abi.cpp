// The author-only entry: run a family's contract hook against a checkpoint
// and hand the result back, without booting a device.
//
// `pie model optimize` needs the answer to "what contract would this driver
// write for this checkpoint and this policy?" so it can precompute the
// load-time work offline — and nothing about *answering* needs a GPU:
// authoring reads the checkpoint's tensor table and the parsed config, and
// both are host facts (`contract.hpp`'s own opening statement). What the
// load path adds around the same builder is the device — `cudaSetDevice`,
// the capability probes, the plan compile, the executor — and that is
// exactly what this entry leaves out. The two device-derived *parameters*
// the builder does consume (`fp8_native`, `native_mxfp4_moe`) are taken from
// the caller, who queried a device or read a config; this file never asks
// CUDA anything.
//
// Not part of the frozen `pie_driver_abi.h` surface: the caller is the Rust
// side of this same repository (worker), version-locked by the build, so the
// contract here is the source and not a header.

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "loader/load_plan.hpp"
#include "model/config.hpp"
#include "model/contract.hpp"
#include "model/registry.hpp"
#include "pie_loader/model_contract.hpp"
#include "pie_loader/source_checkpoint.hpp"

namespace {

constexpr int32_t kOk = 0;
constexpr int32_t kInvalidArgument = 1;
constexpr int32_t kUnsupportedModel = 2;
constexpr int32_t kAuthorFailed = 3;

}  // namespace

/// Author the load contract for `snapshot_dir` under the given policy.
///
/// On success writes an opaque owner to `out_owner` and a borrowed view of
/// the contract to `out_view`; the view is valid until
/// `pie_cuda_author_release(owner)`. On failure returns nonzero and logs the
/// reason to stderr, like every other entry in this driver.
extern "C" int32_t pie_cuda_author_contract(
    const char* snapshot_dir,
    const char* runtime_quant,
    uint32_t mxfp4_moe_request,
    uint32_t component,
    int32_t stream_routed_experts,
    uint32_t tp_rank,
    uint32_t tp_size,
    int32_t fp8_native,
    int32_t native_mxfp4_moe,
    void** out_owner,
    pie_loader::PieLoaderModelContractView* out_view)
{
    using pie_cuda_driver::model::Component;
    using pie_cuda_driver::model::Mxfp4MoeRequest;

    if (snapshot_dir == nullptr || out_owner == nullptr || out_view == nullptr) {
        std::cerr << "[pie-driver-cuda] author_contract: null argument\n";
        return kInvalidArgument;
    }
    if (tp_size == 0 || tp_rank >= tp_size) {
        std::cerr << "[pie-driver-cuda] author_contract: tp " << tp_rank << "/"
                  << tp_size << " is not a rank of a world\n";
        return kInvalidArgument;
    }
    if (mxfp4_moe_request > static_cast<uint32_t>(Mxfp4MoeRequest::EagerBf16) ||
        component > static_cast<uint32_t>(Component::Encode)) {
        std::cerr << "[pie-driver-cuda] author_contract: unknown policy enum\n";
        return kInvalidArgument;
    }

    try {
        const auto hf = pie_cuda_driver::parse_hf_config(
            std::string(snapshot_dir) + "/config.json");

        const auto* arch = pie_cuda_driver::model::find_arch_entry(hf.model_type);
        if (arch == nullptr || !arch->author_contract) {
            std::cerr << "[pie-driver-cuda] author_contract: unsupported model_type '"
                      << hf.model_type << "'\n";
            return kUnsupportedModel;
        }

        std::string open_error;
        pie_loader::Checkpoint checkpoint =
            pie_loader::Checkpoint::open(snapshot_dir, &open_error);
        if (!checkpoint) {
            std::cerr << "[pie-driver-cuda] author_contract: cannot read checkpoint: "
                      << open_error << "\n";
            return kInvalidArgument;
        }

        // The same target the load path builds, minus the device probes the
        // caller answered for us.
        pie_loader::DeviceTarget target = pie_cuda_driver::cuda_device_target();
        target.tp_rank = tp_rank;
        target.tp_size = tp_size;
        target.native_mxfp4_moe = native_mxfp4_moe != 0;

        const pie_cuda_driver::model::ModelFacts facts{
            .model_type = hf.model_type,
            .quant_method = hf.quant_method,
            .num_hidden_layers =
                static_cast<std::uint32_t>(std::max(0, hf.num_hidden_layers)),
            .num_experts = static_cast<std::uint32_t>(std::max(0, hf.num_experts)),
            .head_dim = static_cast<std::uint32_t>(std::max(0, hf.head_dim)),
            .mamba_groups =
                static_cast<std::uint32_t>(std::max(0, hf.mamba_n_groups)),
        };

        auto contract = std::make_unique<pie_loader::ModelContract>();
        {
            pie_cuda_driver::model::ContractBuilder builder(
                checkpoint, facts, target,
                pie_cuda_driver::model::resolve_runtime_quant(
                    runtime_quant == nullptr ? "" : runtime_quant, fp8_native != 0),
                static_cast<Mxfp4MoeRequest>(mxfp4_moe_request),
                static_cast<Component>(component),
                stream_routed_experts != 0,
                *contract);
            arch->author_contract(builder);
            builder.finish();
        }

        *out_view = contract->view();
        *out_owner = contract.release();
        return kOk;
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] author_contract: " << e.what() << "\n";
        return kAuthorFailed;
    } catch (...) {
        std::cerr << "[pie-driver-cuda] author_contract: unknown exception\n";
        return kAuthorFailed;
    }
}

extern "C" void pie_cuda_author_release(void* owner) {
    delete static_cast<pie_loader::ModelContract*>(owner);
}
