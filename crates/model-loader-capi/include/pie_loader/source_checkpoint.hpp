#pragma once

/// The checkpoint, as a value a contract can be written against.
///
/// `plan = compile(source_facts, program, target)`. This is `source_facts`: an
/// RAII handle over the parsed tensor table, opened once and passed to the
/// compile so that the contract the driver writes and the plan the loader
/// builds are provably about the same parse.
///
/// A driver needs this before it can author anything. Whether a checkpoint
/// ships a fused `qkv_proj` or three separate projections, whether the MoE
/// experts are stacked or per-expert, what the packed extents of an AWQ weight
/// are — every one of those is read here and none of them is in `config.json`.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pie_loader.h"
#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"

namespace pie_loader {

/// One tensor as the checkpoint declares it.
struct SourceTensor {
    std::string_view name;
    std::uint32_t id = 0;
    std::uint32_t file_id = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t span_bytes = 0;
    std::span<const std::int64_t> shape;
    PieLoaderEncodingSpec encoding{};
};

/// An opened checkpoint. Move-only; closes on destruction.
class Checkpoint {
public:
    Checkpoint() = default;

    Checkpoint(Checkpoint&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

    Checkpoint& operator=(Checkpoint&& other) noexcept {
        if (this != &other) {
            close();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    ~Checkpoint() { close(); }

    /// Parse the tensor table of every weight file under `snapshot_dir`.
    ///
    /// Returns an empty handle on failure, with the reason written to `error`
    /// if one was passed.
    static Checkpoint open(std::string_view snapshot_dir, std::string* error = nullptr) {
        PieLoaderCheckpoint* handle = nullptr;
        LoadPlanDiagnostics diags;
        const PieLoaderStatus status =
            pie_loader_open_checkpoint(borrow(snapshot_dir), &handle, diags.slot());
        if (status != PieLoaderStatus::Ok) {
            if (error != nullptr) {
                *error = diags.text();
            }
            return Checkpoint();
        }
        return Checkpoint(handle);
    }

    explicit operator bool() const { return handle_ != nullptr; }
    const PieLoaderCheckpoint* get() const { return handle_; }

    /// Every tensor in the checkpoint, in the order the reader found them.
    std::vector<SourceTensor> tensors() const {
        std::vector<SourceTensor> out;
        if (handle_ == nullptr) {
            return out;
        }
        out.reserve(handle_->tensors.len);
        for (std::size_t i = 0; i < handle_->tensors.len; ++i) {
            const PieLoaderRawTensorView& raw = handle_->tensors.ptr[i];
            out.push_back(SourceTensor{
                .name = text(raw.name),
                .id = raw.id,
                .file_id = raw.file_id,
                .file_offset = raw.file_offset,
                .span_bytes = raw.span_bytes,
                .shape = {raw.shape.ptr, raw.shape.len},
                .encoding = raw.encoding,
            });
        }
        return out;
    }

    /// Look one tensor up by name.
    ///
    /// The question a contract author asks constantly — "does this checkpoint
    /// have a fused QKV?" is `find("...qkv_proj.weight").has_value()` — so it is
    /// worth having rather than making every caller scan `tensors()`.
    std::optional<SourceTensor> find(std::string_view name) const {
        if (handle_ == nullptr) {
            return std::nullopt;
        }
        for (std::size_t i = 0; i < handle_->tensors.len; ++i) {
            const PieLoaderRawTensorView& raw = handle_->tensors.ptr[i];
            if (text(raw.name) == name) {
                return SourceTensor{
                    .name = text(raw.name),
                    .id = raw.id,
                    .file_id = raw.file_id,
                    .file_offset = raw.file_offset,
                    .span_bytes = raw.span_bytes,
                    .shape = {raw.shape.ptr, raw.shape.len},
                    .encoding = raw.encoding,
                };
            }
        }
        return std::nullopt;
    }

    bool has(std::string_view name) const { return find(name).has_value(); }

private:
    explicit Checkpoint(PieLoaderCheckpoint* handle) : handle_(handle) {}

    void close() {
        if (handle_ != nullptr) {
            pie_loader_close_checkpoint(handle_);
            handle_ = nullptr;
        }
    }

    static std::string_view text(const PieLoaderBytes& bytes) {
        return bytes.ptr == nullptr
                   ? std::string_view{}
                   : std::string_view(reinterpret_cast<const char*>(bytes.ptr), bytes.len);
    }

    PieLoaderCheckpoint* handle_ = nullptr;
};

}  // namespace pie_loader
