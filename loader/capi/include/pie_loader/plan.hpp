#pragma once

// C++ ownership layer over the loader's generated C ABI.
//
// This header used to contain a second, hand-written copy of every `PieLoader*`
// POD type plus a ~460-line nlohmann parser that turned the loader's JSON output
// back into those types. Both are gone: `pie_loader.h` is generated from the
// Rust definitions by `cargo run -p pie-loader-cbindgen`, so the two sides can
// no longer disagree, and the plan now arrives as POD that needs no parsing
// (see `loader/architecture.md` §5).
//
// What remains here is the part C++ still has to supply: RAII ownership of a
// plan whose memory belongs to Rust, and the small helpers that read the POD.
//
// It deliberately states nothing about any particular backend. Which tile-map
// transforms a driver's kernels implement is that driver's claim to make, so
// the masks live next to the kernels that honour them; the cross-check in §9 is
// one-sided on purpose, and a mask defined here would make it circular.

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pie_loader.h"

namespace pie_loader {

// The old header spelled buffer runs with their own slice type; the generated
// header has one `u32` slice. Keep the descriptive name.
using PieLoaderBufferIdSlice = PieLoaderU32Slice;

// `PieLoaderPlan` opens with exactly the members the old `LoadPlanView` had, in
// the same order and under the same names, so every consumer that walked a
// `LoadPlanView` walks a plan unchanged.
using LoadPlanView = PieLoaderPlan;

inline constexpr std::uint32_t kTileMapCast = PIE_LOADER_TILE_MAP_CAST;
inline constexpr std::uint32_t kTileMapDecode = PIE_LOADER_TILE_MAP_DECODE;
inline constexpr std::uint32_t kTileMapEncode = PIE_LOADER_TILE_MAP_ENCODE;
inline constexpr std::uint32_t kTileMapTranscode = PIE_LOADER_TILE_MAP_TRANSCODE;
inline constexpr std::uint32_t kTileMapReblock = PIE_LOADER_TILE_MAP_REBLOCK;
inline constexpr std::uint32_t kTileMapRepack = PIE_LOADER_TILE_MAP_REPACK;
inline constexpr std::uint32_t kTileMapScale = PIE_LOADER_TILE_MAP_SCALE;

/// Everything the loader reported about one call, owned and freed by C++.
class LoadPlanDiagnostics {
  public:
    LoadPlanDiagnostics() = default;
    explicit LoadPlanDiagnostics(PieLoaderDiagnostics* raw) noexcept : raw_(raw) {}
    LoadPlanDiagnostics(const LoadPlanDiagnostics&) = delete;
    LoadPlanDiagnostics& operator=(const LoadPlanDiagnostics&) = delete;
    LoadPlanDiagnostics(LoadPlanDiagnostics&& other) noexcept
        : raw_(std::exchange(other.raw_, nullptr)) {}
    LoadPlanDiagnostics& operator=(LoadPlanDiagnostics&& other) noexcept {
        if (this != &other) {
            reset();
            raw_ = std::exchange(other.raw_, nullptr);
        }
        return *this;
    }
    ~LoadPlanDiagnostics() { reset(); }

    void reset() noexcept {
        if (raw_ != nullptr) {
            pie_loader_release_diagnostics(raw_);
            raw_ = nullptr;
        }
    }

    /// Join every record into one message, for throwing.
    std::string text() const {
        if (raw_ == nullptr || raw_->len == 0) return {};
        std::string out;
        for (std::size_t i = 0; i < raw_->len; ++i) {
            const auto& item = raw_->items[i];
            if (!out.empty()) out += "; ";
            if (item.message.ptr != nullptr) {
                out.append(
                    reinterpret_cast<const char*>(item.message.ptr),
                    item.message.len);
            }
        }
        return out;
    }

    bool empty() const noexcept { return raw_ == nullptr || raw_->len == 0; }
    PieLoaderDiagnostics** slot() noexcept { return &raw_; }

  private:
    PieLoaderDiagnostics* raw_ = nullptr;
};

/// An owned compiled plan.
///
/// The bytes belong to Rust; this class is only responsible for calling
/// `pie_loader_release` exactly once. Ownership is unique and move-only, which
/// matches how the old parser-backed `LoadPlan` behaved.
class LoadPlan {
  public:
    LoadPlan() = default;
    LoadPlan(const LoadPlan&) = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;
    LoadPlan(LoadPlan&& other) noexcept
        : plan_(std::exchange(other.plan_, nullptr)) {}
    LoadPlan& operator=(LoadPlan&& other) noexcept {
        if (this != &other) {
            reset();
            plan_ = std::exchange(other.plan_, nullptr);
        }
        return *this;
    }
    ~LoadPlan() { reset(); }

    void reset() noexcept {
        if (plan_ != nullptr) {
            pie_loader_release(plan_);
            plan_ = nullptr;
        }
    }

    /// Author this model's contract on the Rust side and compile it, in one
    /// call — the §6 request boundary of `plan/model-in-rust.md`, and the only
    /// way to obtain a plan. The facts and policy cross as a handful of
    /// scalars; no contract crosses at all, and nothing here lets a driver
    /// hand one over: what gets built is decided by the author registry on
    /// the far side. Nothing in the *loader* dispatches on the model_type the
    /// facts carry — a family the registry has never heard of is a
    /// diagnostic asking for an author, not a silent guess.
    ///
    /// `out_mxfp4_moe` receives the author's resolved policy as its wire
    /// value: the bind path branches on the author's answer, and a family may
    /// override the device rule, so the answer is handed back rather than
    /// recomputed.
    static LoadPlan compile_model(const PieLoaderModelRequest& request,
                                  std::uint32_t* out_mxfp4_moe) {
        PieLoaderPlan* raw = nullptr;
        LoadPlanDiagnostics diags;
        const PieLoaderStatus status =
            pie_loader_compile_model(&request, &raw, out_mxfp4_moe, diags.slot());
        if (status != PieLoaderStatus::Ok) {
            throw std::runtime_error(
                "load plan: rust-author compile failed (" + status_name(status) +
                "): " + diags.text());
        }
        if (raw == nullptr) {
            throw std::runtime_error("load plan: compile returned no plan");
        }
        return LoadPlan(raw);
    }

    /// Re-check the plan against the contract its model request authors.
    ///
    /// The verifier re-authors from the request on the far side and holds
    /// the plan to it, so the author's determinism is in scope along with
    /// everything `verify` checks.
    void verify_model(const PieLoaderModelRequest& request) const {
        LoadPlanDiagnostics diags;
        const PieLoaderStatus status =
            pie_loader_verify_model(plan_, &request, diags.slot());
        if (status != PieLoaderStatus::Ok) {
            throw std::runtime_error(
                "load plan: model verification failed (" + status_name(status) +
                "): " + diags.text());
        }
    }

    explicit operator bool() const noexcept { return plan_ != nullptr; }

    const LoadPlanView& view() const {
        if (plan_ == nullptr) throw std::runtime_error("load plan: empty");
        return *plan_;
    }

    PieLoaderBackendKind backend() const { return view().target.backend; }
    bool native_mxfp4_moe() const { return view().target.native_mxfp4_moe; }
    std::uint32_t preferred_alignment() const {
        return view().target.preferred_alignment;
    }
    std::uint64_t max_tile_bytes() const { return view().target.max_tile_bytes; }
    std::uint32_t tile_map_mask() const { return view().target.tile_map_mask; }
    std::uint64_t compiler_version() const { return view().compiler_version; }

    static std::string status_name(PieLoaderStatus status) {
        switch (status) {
        case PieLoaderStatus::Ok: return "ok";
        case PieLoaderStatus::InvalidRequest: return "invalid request";
        case PieLoaderStatus::InvalidCheckpoint: return "invalid checkpoint";
        case PieLoaderStatus::ContractViolation: return "contract violation";
        case PieLoaderStatus::Internal: return "internal error";
        }
        return "unknown status";
    }

  private:
    explicit LoadPlan(PieLoaderPlan* plan) noexcept : plan_(plan) {}

    PieLoaderPlan* plan_ = nullptr;
};

inline std::string bytes_to_string(PieLoaderBytes bytes) {
    if (bytes.ptr == nullptr || bytes.len == 0) return {};
    return std::string(
        reinterpret_cast<const char*>(bytes.ptr),
        reinterpret_cast<const char*>(bytes.ptr) + bytes.len);
}

inline std::vector<std::int64_t> i64_slice_to_vector(PieLoaderI64Slice shape) {
    if (shape.ptr == nullptr || shape.len == 0) return {};
    return {shape.ptr, shape.ptr + shape.len};
}

inline std::vector<std::uint32_t> buffer_id_slice_to_vector(
    PieLoaderBufferIdSlice ids) {
    if (ids.ptr == nullptr || ids.len == 0) return {};
    return {ids.ptr, ids.ptr + ids.len};
}

inline std::vector<std::int64_t> extent_shape(
    const PieLoaderStridedExtentView& extent) {
    std::vector<std::int64_t> shape;
    shape.reserve(extent.dims.len);
    for (std::size_t i = 0; i < extent.dims.len; ++i) {
        shape.push_back(extent.dims.ptr[i].count);
    }
    return shape;
}

inline bool compact_extent(const PieLoaderStridedExtentView& extent) {
    std::int64_t stride = static_cast<std::int64_t>(extent.element_bytes);
    for (std::size_t i = extent.dims.len; i > 0; --i) {
        const auto& dim = extent.dims.ptr[i - 1];
        if (dim.src_stride != stride || dim.dst_stride != stride) return false;
        stride *= dim.count;
    }
    return true;
}

inline std::uint64_t extent_bytes(
    const PieLoaderStridedExtentView& extent,
    const char* context) {
    std::uint64_t elements = 1;
    for (std::size_t i = 0; i < extent.dims.len; ++i) {
        const auto count = extent.dims.ptr[i].count;
        if (count < 0) {
            throw std::runtime_error(std::string(context) + ": negative extent");
        }
        const auto ucount = static_cast<std::uint64_t>(count);
        if (ucount != 0 &&
            elements > std::numeric_limits<std::uint64_t>::max() / ucount) {
            throw std::runtime_error(std::string(context) + ": extent overflow");
        }
        elements *= ucount;
    }
    if (extent.element_bytes != 0 &&
        elements > std::numeric_limits<std::uint64_t>::max() /
                       extent.element_bytes) {
        throw std::runtime_error(std::string(context) + ": extent overflow");
    }
    return elements * extent.element_bytes;
}

class LoadPlanIndex {
  public:
    explicit LoadPlanIndex(std::string context)
        : context_(std::move(context)) {}

    void reset(const LoadPlanView& plan) {
        instr_by_id_.clear();
        buffer_by_id_.clear();
        tensor_by_id_.clear();
        source_by_id_.clear();
        for (std::size_t i = 0; i < plan.instrs.len; ++i) {
            instr_by_id_.emplace(plan.instrs.ptr[i].id, &plan.instrs.ptr[i]);
        }
        for (std::size_t i = 0; i < plan.buffers.len; ++i) {
            buffer_by_id_.emplace(plan.buffers.ptr[i].id, &plan.buffers.ptr[i]);
        }
        for (std::size_t i = 0; i < plan.tensors.len; ++i) {
            tensor_by_id_.emplace(plan.tensors.ptr[i].id, &plan.tensors.ptr[i]);
        }
        for (std::size_t i = 0; i < plan.sources.len; ++i) {
            const auto* source = &plan.sources.ptr[i];
            source_by_id_.emplace(source->id, source);
        }
    }

    const PieLoaderStorageInstrView& instruction(std::uint32_t id) const {
        const auto it = instr_by_id_.find(id);
        if (it == instr_by_id_.end()) {
            throw std::runtime_error(context_ + ": instruction id out of range");
        }
        return *it->second;
    }
    const PieLoaderBufferDeclView& buffer(std::uint32_t id) const {
        const auto it = buffer_by_id_.find(id);
        if (it == buffer_by_id_.end()) {
            throw std::runtime_error(context_ + ": buffer id out of range");
        }
        return *it->second;
    }
    const PieLoaderTensorDeclView& tensor(std::uint32_t id) const {
        const auto it = tensor_by_id_.find(id);
        if (it == tensor_by_id_.end()) {
            throw std::runtime_error(context_ + ": tensor id out of range");
        }
        return *it->second;
    }
    const PieLoaderSourceTensorView& source(std::uint32_t id) const {
        const auto it = source_by_id_.find(id);
        if (it == source_by_id_.end()) {
            throw std::runtime_error(context_ + ": source tensor id out of range");
        }
        return *it->second;
    }

  private:
    std::string context_;
    std::unordered_map<std::uint32_t, const PieLoaderStorageInstrView*> instr_by_id_;
    std::unordered_map<std::uint32_t, const PieLoaderBufferDeclView*> buffer_by_id_;
    std::unordered_map<std::uint32_t, const PieLoaderTensorDeclView*> tensor_by_id_;
    std::unordered_map<std::uint32_t, const PieLoaderSourceTensorView*> source_by_id_;
};

}  // namespace pie_loader
