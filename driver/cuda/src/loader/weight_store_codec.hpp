#pragma once

// Weight-store codec: write a materialized WeightStore to an artifact file, and
// restore one from it. A general conversion/load primitive for finished device
// weights — it knows nothing about caches, directories, keys, or eviction. The
// artifact cache (model/weight_artifact_cache.hpp) is one consumer: it owns
// that policy and uses serialize/restore as opaque file conversion.
//
// The byte format is not here. It used to be — PIEWSTOR, a hand-rolled binary
// layout with its own checksum, its own alignment arithmetic and its own
// truncation checks. An artifact is a file format, and pie has one: the loader
// writes these as `.zt` (loader/src/weight_store.rs), so placement, digests and
// a validating reader all come from the same place the checkpoints do. What is
// left here is the part only the driver can do, which is the device copies.
//
// Memory model: the WeightStore is a set of *owned* DeviceTensors (the storage
// arena + any separately-owned buffers) and *view* DeviceTensors that point into
// one of those owned buffers. Each owned buffer's bytes are stored once, and
// each view is recorded as its owning root and a byte offset — restoring a view
// as a copy would double the memory the views exist to save. The full TensorDecl
// and the quant_meta map travel too, so the rebuilt store passes finalize()
// validation unchanged.
//
// serialize() copies each root off the device in chunks and hands them over as
// they arrive: the host never holds more than one chunk of a store that is tens
// of gigabytes. restore() streams the artifact's mapped bytes to the device
// through the shared pinned staged-H2D engine (loader/staged_h2d.hpp) — the same
// path the cold materialize uses — and verifies each payload's digest
// concurrently with the copy.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"

#include "loader/dtype_map.hpp"
#include "loader/tensor_spec.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

#if __has_include(<cuda_runtime.h>)
#define PIE_CUDA_WEIGHT_CODEC_HAS_CUDA 1
#include <cuda_runtime.h>
#include "loader/staged_h2d.hpp"
#else
#define PIE_CUDA_WEIGHT_CODEC_HAS_CUDA 0
#endif

namespace pie_cuda_driver::weight_codec {

// This whole file is one ABI's caller, and every name it reaches for is from
// that ABI. The directive is scoped to this namespace, so it pulls nothing into
// anyone else's.
using namespace pie_loader;

// How much of a root is copied back from the device at a time. Sized to amortize
// the per-copy latency without pinning a large host buffer; nothing in the file
// depends on it, so it can change freely.
inline constexpr std::uint64_t kChunkBytes = 64ull * 1024ull * 1024ull;

namespace detail {

// The artifact writer, freed exactly once however the function leaves.
class Writer {
  public:
    Writer() = default;
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    ~Writer() { pie_loader_weight_store_discard(raw_); }

    bool open(const std::filesystem::path& path, const std::string& key)
    {
        pie_loader::LoadPlanDiagnostics diags;
        const auto status = pie_loader_weight_store_create(
            pie_loader::borrow(path.string()), pie_loader::borrow(key), &raw_,
            diags.slot());
        if (status != pie_loader::PieLoaderStatus::Ok) {
            std::fprintf(stderr, "[pie-driver-cuda] weight codec: %s\n",
                         diags.text().c_str());
            return false;
        }
        return true;
    }

    // Publish the artifact. The handle is consumed either way.
    bool publish()
    {
        pie_loader::LoadPlanDiagnostics diags;
        const auto status =
            pie_loader_weight_store_publish(std::exchange(raw_, nullptr), diags.slot());
        if (status != pie_loader::PieLoaderStatus::Ok) {
            std::fprintf(stderr, "[pie-driver-cuda] weight codec: %s\n",
                         diags.text().c_str());
            return false;
        }
        return true;
    }

    // Report and swallow a failed call: every one of them is a reason to skip
    // persisting, never a reason to fail the load that produced the store.
    bool ok(pie_loader::PieLoaderStatus status, const char* what)
    {
        if (status == pie_loader::PieLoaderStatus::Ok) {
            return true;
        }
        std::fprintf(stderr, "[pie-driver-cuda] weight codec: %s: %s\n", what,
                     pie_loader::bytes_to_string(pie_loader_weight_store_error(raw_))
                         .c_str());
        return false;
    }

    PieLoaderWeightWriter* raw() { return raw_; }

  private:
    PieLoaderWeightWriter* raw_ = nullptr;
};

// An opened artifact, closed exactly once however the function leaves.
class Store {
  public:
    Store() = default;
    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;
    ~Store() { pie_loader_weight_store_close(raw_); }

    bool open(const std::filesystem::path& path, const std::string& key)
    {
        pie_loader::LoadPlanDiagnostics diags;
        const auto status = pie_loader_weight_store_open(
            pie_loader::borrow(path.string()), pie_loader::borrow(key), &raw_,
            diags.slot());
        if (status != pie_loader::PieLoaderStatus::Ok) {
            // A miss, not a fault: a stale key, another schema and a file that
            // is not an artifact all mean the same thing to the caller.
            return false;
        }
        return true;
    }

    const PieLoaderWeightStore& get() const { return *raw_; }
    const PieLoaderWeightStore* raw() const { return raw_; }

  private:
    PieLoaderWeightStore* raw_ = nullptr;
};

}  // namespace detail

// Serialize `store` (keyed by `key`) to `path`, which is created and renamed
// into place. Best-effort: returns false (without throwing for recoverable
// cases) if the store doesn't fit the owned-buffer + views model or a call
// fails, so the caller can simply skip persisting.
inline bool serialize_weight_store(const WeightStore& store,
                                   const std::string& key,
                                   const std::filesystem::path& path)
{
#if !PIE_CUDA_WEIGHT_CODEC_HAS_CUDA
    (void)store; (void)key; (void)path;
    return false;
#else
    struct OwnedRoot {
        const TensorRecord* rec;
        std::uintptr_t base;
        std::uint64_t nbytes;
    };
    struct ViewEntry {
        const TensorRecord* rec;
        const OwnedRoot* root;  // null for a view with no bytes at all
        std::uint64_t byte_offset;
    };

    std::vector<OwnedRoot> owned;
    for (auto it = store.begin(); it != store.end(); ++it) {
        const TensorRecord& rec = it->second;
        if (!rec.has_spec) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight codec: declining — tensor '%s' has "
                "no TensorDecl\n", it->first.c_str());
            return false;
        }
        const DeviceTensor& t = rec.tensor;
        if (t.owns_memory() && t.data() != nullptr) {
            owned.push_back(OwnedRoot{
                &rec, reinterpret_cast<std::uintptr_t>(t.data()),
                static_cast<std::uint64_t>(t.nbytes())});
        }
    }

    std::vector<ViewEntry> views;
    for (auto it = store.begin(); it != store.end(); ++it) {
        const TensorRecord& rec = it->second;
        const DeviceTensor& t = rec.tensor;
        if (t.owns_memory() && t.data() != nullptr) {
            continue;  // already an owned root
        }
        const OwnedRoot* root = nullptr;
        std::uint64_t byte_offset = 0;
        if (t.data() != nullptr) {
            const auto vbase = reinterpret_cast<std::uintptr_t>(t.data());
            const auto vbytes = static_cast<std::uint64_t>(t.nbytes());
            for (const auto& candidate : owned) {
                if (vbase >= candidate.base &&
                    vbase + vbytes <= candidate.base + candidate.nbytes) {
                    root = &candidate;
                    byte_offset = vbase - candidate.base;
                    break;
                }
            }
            if (root == nullptr) {
                std::fprintf(stderr,
                    "[pie-driver-cuda] weight codec: declining — view '%s' has "
                    "no owned backing buffer in the store\n", it->first.c_str());
                return false;
            }
        }
        views.push_back(ViewEntry{&rec, root, byte_offset});
    }

    detail::Writer writer;
    if (!writer.open(path, key)) {
        return false;
    }

    std::vector<char> host(static_cast<std::size_t>(kChunkBytes));
    for (const auto& root : owned) {
        const TensorDecl& spec = root.rec->spec;
        const auto shape = spec.shape;
        const PieLoaderWeightTensor decl{
            .name = pie_loader::borrow(spec.name),
            .dtype = dtype_to_rust(spec.dtype),
            .runtime_dtype = pie_loader::borrow(dtype_name(spec.dtype)),
            .shape = PieLoaderI64Slice{shape.data(), shape.size()},
        };
        if (!writer.ok(pie_loader_weight_store_begin_tensor(writer.raw(), &decl),
                       spec.name.c_str())) {
            return false;
        }
        const auto* src = static_cast<const std::uint8_t*>(root.rec->tensor.data());
        std::uint64_t done = 0;
        while (done < root.nbytes) {
            const std::size_t n = static_cast<std::size_t>(
                std::min<std::uint64_t>(host.size(), root.nbytes - done));
            const cudaError_t e = cudaMemcpy(host.data(), src + done, n,
                                             cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) {
                std::fprintf(stderr,
                    "[pie-driver-cuda] weight codec: D2H failed for '%s': %s\n",
                    spec.name.c_str(), cudaGetErrorString(e));
                return false;
            }
            if (!writer.ok(pie_loader_weight_store_write(
                               writer.raw(),
                               reinterpret_cast<const std::uint8_t*>(host.data()), n),
                           spec.name.c_str())) {
                return false;
            }
            done += n;
        }
        if (!writer.ok(pie_loader_weight_store_end_tensor(writer.raw()),
                       spec.name.c_str())) {
            return false;
        }
    }

    for (const auto& view : views) {
        const TensorDecl& spec = view.rec->spec;
        const auto shape = spec.shape;
        const PieLoaderWeightView decl{
            .name = pie_loader::borrow(spec.name),
            .root = view.root != nullptr
                        ? pie_loader::borrow(view.root->rec->spec.name)
                        : PieLoaderBytes{nullptr, 0},
            .byte_offset = view.byte_offset,
            .dtype = dtype_to_rust(spec.dtype),
            .runtime_dtype = pie_loader::borrow(dtype_name(spec.dtype)),
            .shape = PieLoaderI64Slice{shape.data(), shape.size()},
            .backing = pie_loader::borrow(spec.backing_tensor),
        };
        if (!writer.ok(pie_loader_weight_store_add_view(writer.raw(), &decl),
                       spec.name.c_str())) {
            return false;
        }
    }

    for (const auto& kv : store.quant_meta_map()) {
        const QuantMeta& meta = kv.second;
        const PieLoaderWeightQuant decl{
            .name = pie_loader::borrow(kv.first),
            .kind = pie_loader::borrow(quant_kind_name(meta.kind)),
            .scale = pie_loader::borrow(meta.scale_name),
            .zero_point = pie_loader::borrow(meta.zero_point_name),
            .group_size = static_cast<std::int32_t>(meta.group_size),
            .channel_axis = static_cast<std::int32_t>(meta.channel_axis),
        };
        if (!writer.ok(pie_loader_weight_store_add_quant(writer.raw(), &decl),
                       kv.first.c_str())) {
            return false;
        }
    }

    return writer.publish();
#endif
}

// Restore a store from the artifact at `path`. Returns false on a key or format
// mismatch or on corruption (a "miss" the caller falls back from); the payloads
// are streamed to the device through `pool` straight from the mapped file, and
// each one's digest is verified concurrently with the copy when `verify` is set.
inline bool restore_weight_store(const std::filesystem::path& path,
                                 const std::string& expected_key, bool verify,
                                 WeightStoreBuilder& builder,
                                 PinnedLanePool& pool)
{
#if !PIE_CUDA_WEIGHT_CODEC_HAS_CUDA
    (void)path; (void)expected_key; (void)verify; (void)builder; (void)pool;
    return false;
#else
    detail::Store store;
    if (!store.open(path, expected_key)) {
        return false;
    }
    const PieLoaderWeightStore& artifact = store.get();

    // Allocate the roots and queue one staged copy each, sourced straight from
    // the mapped file. The shared engine pins + pipelines the H2D.
    std::unordered_map<std::string, void*> root_base;
    std::vector<StagedCopy> copies;
    copies.reserve(artifact.tensors.len);
    root_base.reserve(artifact.tensors.len);
    for (std::size_t i = 0; i < artifact.tensors.len; ++i) {
        const PieLoaderWeightTensorView& raw = artifact.tensors.ptr[i];
        const std::string name = pie_loader::bytes_to_string(raw.name);
        TensorDecl spec;
        spec.name = name;
        if (!dtype_by_name(
                std::string_view(reinterpret_cast<const char*>(raw.runtime_dtype.ptr),
                                 raw.runtime_dtype.len),
                spec.dtype)) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight codec: '%s' names a type this build "
                "does not have — discarding restore\n", name.c_str());
            return false;
        }
        spec.shape = pie_loader::i64_slice_to_vector(raw.shape);
        spec.ownership = TensorOwnershipKind::Owned;

        DeviceTensor t = DeviceTensor::allocate(spec.dtype, spec.shape);
        if (t.nbytes() != raw.nbytes) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight codec: nbytes mismatch for '%s' — "
                "discarding restore\n", name.c_str());
            return false;
        }
        copies.push_back(StagedCopy{t.data(), raw.data, raw.nbytes});
        root_base.emplace(name, t.data());
        builder.insert(name, std::move(t), spec);
    }

    // Verify on the CPU concurrently with the DMA: the copy is PCIe-bound while
    // the digest is a memory read, and both touch the same warm pages.
    std::thread verifier;
    bool digests_ok = true;
    if (verify) {
        const PieLoaderWeightStore* raw = store.raw();
        const std::size_t count = artifact.tensors.len;
        verifier = std::thread([raw, count, &digests_ok] {
            for (std::size_t i = 0; i < count && digests_ok; ++i) {
                bool ok = false;
                if (pie_loader_weight_store_verify(raw, i, &ok) !=
                        pie_loader::PieLoaderStatus::Ok ||
                    !ok) {
                    digests_ok = false;
                }
            }
        });
    }
    try {
        staged_pinned_h2d(pool, copies);
    } catch (...) {
        if (verifier.joinable()) {
            verifier.join();  // never leave the verify thread joinable (would std::terminate)
        }
        throw;
    }
    if (verifier.joinable()) {
        verifier.join();
    }
    if (verify && !digests_ok) {
        std::fprintf(stderr,
            "[pie-driver-cuda] weight codec: digest mismatch — discarding restore\n");
        return false;
    }

    // Reconstruct the views as windows into their roots, with their spec.
    for (std::size_t i = 0; i < artifact.views.len; ++i) {
        const PieLoaderWeightView& raw = artifact.views.ptr[i];
        TensorDecl spec;
        spec.name = pie_loader::bytes_to_string(raw.name);
        if (!dtype_by_name(
                std::string_view(reinterpret_cast<const char*>(raw.runtime_dtype.ptr),
                                 raw.runtime_dtype.len),
                spec.dtype)) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight codec: view '%s' names a type this "
                "build does not have — discarding restore\n", spec.name.c_str());
            return false;
        }
        spec.shape = pie_loader::i64_slice_to_vector(raw.shape);
        spec.ownership = TensorOwnershipKind::BorrowedView;
        spec.backing_tensor = pie_loader::bytes_to_string(raw.backing);

        void* base = nullptr;
        const std::string root = pie_loader::bytes_to_string(raw.root);
        if (!root.empty()) {
            const auto it = root_base.find(root);
            if (it == root_base.end()) {
                std::fprintf(stderr,
                    "[pie-driver-cuda] weight codec: view '%s' names root '%s', "
                    "which the artifact does not hold — discarding restore\n",
                    spec.name.c_str(), root.c_str());
                return false;
            }
            base = static_cast<std::uint8_t*>(it->second) + raw.byte_offset;
        }
        builder.insert(spec.name,
                       DeviceTensor::view(base, spec.dtype, spec.shape),
                       spec);
    }

    // Rebuild quant metadata (pointers resolved by name from the populated store).
    for (std::size_t i = 0; i < artifact.quants.len; ++i) {
        const PieLoaderWeightQuant& raw = artifact.quants.ptr[i];
        QuantMeta meta;
        if (!quant_kind_by_name(
                std::string_view(reinterpret_cast<const char*>(raw.kind.ptr),
                                 raw.kind.len),
                meta.kind)) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight codec: unknown scale kind — "
                "discarding restore\n");
            return false;
        }
        meta.scale_name = pie_loader::bytes_to_string(raw.scale);
        meta.zero_point_name = pie_loader::bytes_to_string(raw.zero_point);
        meta.group_size = raw.group_size;
        meta.channel_axis = raw.channel_axis;
        if (!meta.scale_name.empty()) {
            auto it = builder.find(meta.scale_name);
            if (it != builder.end()) {
                meta.scale = &it->second.tensor;
            }
        }
        if (!meta.zero_point_name.empty()) {
            auto it = builder.find(meta.zero_point_name);
            if (it != builder.end()) {
                meta.zero_point = &it->second.tensor;
            }
        }
        builder.set_quant_meta(pie_loader::bytes_to_string(raw.name), std::move(meta));
    }

    builder.finalize();
    return true;
#endif
}

}  // namespace pie_cuda_driver::weight_codec
