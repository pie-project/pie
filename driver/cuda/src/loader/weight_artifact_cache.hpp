#pragma once

// Materialized-weight artifact cache (WEIGHT_LOADER_TODO.md A3.1).
//
// The materialized weights are a deterministic function of the compile cache
// key (checkpoint files + config + quant scheme + TP layout + ABI version), yet
// today they are recomputed every boot — for FP8->MXFP4 models that is hundreds
// of seconds of dequant/re-encode. This cache snapshots the materialized device
// memory + the full WeightStore manifest after the first load, keyed by that
// authoritative key, so a warm boot reloads the finished tensors straight into
// device memory and skips compile + materialize entirely.
//
// Memory model it relies on: the WeightStore is a set of *owned* DeviceTensors
// (the storage arena, plus any separately-allocated owned buffers) and *view*
// DeviceTensors that point into one of those owned buffers (see
// rust_storage_executor.hpp). The cache stores each owned buffer's bytes once
// and records each view as (owning-buffer index, byte offset). The full
// TensorDecl spec (incl. ownership + backing_tensor) and the quant_meta map are
// serialized so the rebuilt store passes WeightStore::finalize() validation
// unchanged. If a view's backing buffer is not an owned entry in the store, the
// write path safely declines (no cache written).
//
// OFF BY DEFAULT — strictly opt-in via PIE_CUDA_WEIGHT_CACHE_DIR. With the env
// unset/empty the cache never reads or writes (weight_artifact_cache_dir()
// returns empty and both paths are skipped), so it costs zero disk. Even when
// enabled, the write declines if free space < blob size + margin (the artifact
// is the size of the materialized weights — tens to hundreds of GB). Each owned
// blob carries a fast word-wise checksum verified on reload (skip with
// PIE_CUDA_WEIGHT_CACHE_NO_VERIFY); key/format-version mismatch => treated as miss.
//
// VALIDATION STATUS: the dense single-arena path (e.g. gemma) is validated on a
// local box (snapshot+reload is bit-identical — guaranteed by the per-blob
// checksum — and serves correctly). The quant_meta / multi-owned-buffer (FP8
// MoE) path is implemented but its load-time win must be measured on the 4xB200
// box; the whole feature is gated behind the env flag.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "model/weight_store.hpp"
#include "loader/tensor_spec.hpp"
#include "tensor.hpp"

#if __has_include(<cuda_runtime.h>)
#define PIE_CUDA_WEIGHT_ARTIFACT_CACHE_HAS_CUDA 1
#else
#define PIE_CUDA_WEIGHT_ARTIFACT_CACHE_HAS_CUDA 0
#endif
#if PIE_CUDA_WEIGHT_ARTIFACT_CACHE_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace pie_cuda_driver {

// Returns the configured artifact cache directory, or empty if the feature is
// disabled (PIE_CUDA_WEIGHT_CACHE_DIR unset/empty).
inline std::filesystem::path weight_artifact_cache_dir()
{
    const char* dir = std::getenv("PIE_CUDA_WEIGHT_CACHE_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return {};
    }
    return std::filesystem::path(dir);
}

namespace detail {

inline constexpr char kWeightArtifactMagic[8] =
    {'P', 'I', 'E', 'W', 'C', 'A', 'C', '3'};
inline constexpr std::uint32_t kWeightArtifactFormatVersion = 3;
inline constexpr std::uint64_t kWeightArtifactChunkBytes = 64ull * 1024ull * 1024ull;

inline constexpr std::uint64_t kBlobHashSeed = 0xcbf29ce484222325ull;
// Streaming integrity hash over blob bytes. A byte-wise FNV1a here measured at
// ~0.75 GiB/s — 90% of cache-reload time — so this folds 32 bytes/step across 4
// independent accumulator lanes (breaks the multiply dependency chain) with an
// 8-byte then 1-byte tail. It is boundary-DEPENDENT (the per-call seed fold), so
// write and read MUST chunk identically; they do (same kWeightArtifactChunkBytes
// formula + same per-root nbytes). The format version bumps when this changes.
// Host-endian, non-cryptographic — fine for a same-machine corruption check.
inline std::uint64_t blob_hash_update(std::uint64_t seed, const void* data, std::size_t len)
{
    constexpr std::uint64_t prime = 0x100000001b3ull;
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t h0 = seed;
    std::uint64_t h1 = seed ^ 0x9e3779b97f4a7c15ull;
    std::uint64_t h2 = seed ^ 0xff51afd7ed558ccdull;
    std::uint64_t h3 = seed ^ 0xc4ceb9fe1a85ec53ull;
    std::size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        std::uint64_t w0, w1, w2, w3;
        std::memcpy(&w0, p + i, 8);
        std::memcpy(&w1, p + i + 8, 8);
        std::memcpy(&w2, p + i + 16, 8);
        std::memcpy(&w3, p + i + 24, 8);
        h0 = (h0 ^ w0) * prime;
        h1 = (h1 ^ w1) * prime;
        h2 = (h2 ^ w2) * prime;
        h3 = (h3 ^ w3) * prime;
    }
    std::uint64_t h = h0 ^ h1 ^ h2 ^ h3;
    for (; i + 8 <= len; i += 8) {
        std::uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * prime;
    }
    for (; i < len; ++i) {
        h = (h ^ p[i]) * prime;
    }
    return h;
}

// --- little binary writers/readers over std::ostream/istream ---------------
template <typename T>
inline void put_scalar(std::ostream& os, const T& v)
{
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
inline T get_scalar(std::istream& is)
{
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}
inline void put_str(std::ostream& os, const std::string& s)
{
    put_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(s.size()));
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}
inline std::string get_str(std::istream& is)
{
    const auto len = get_scalar<std::uint32_t>(is);
    std::string s(len, '\0');
    is.read(s.data(), static_cast<std::streamsize>(len));
    return s;
}
inline void put_shape(std::ostream& os, const std::vector<std::int64_t>& shape)
{
    put_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(shape.size()));
    for (const auto d : shape) {
        put_scalar<std::int64_t>(os, d);
    }
}
inline std::vector<std::int64_t> get_shape(std::istream& is)
{
    const auto rank = get_scalar<std::uint32_t>(is);
    std::vector<std::int64_t> shape(rank);
    for (std::uint32_t i = 0; i < rank; ++i) {
        shape[i] = get_scalar<std::int64_t>(is);
    }
    return shape;
}

// Full TensorDecl so the rebuilt store passes validate_tensor_records()
// (ownership + backing_tensor are required for view/alias tensors).
inline void put_spec(std::ostream& os, const TensorDecl& s)
{
    put_str(os, s.name);
    put_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(s.dtype));
    put_shape(os, s.shape);
    put_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(s.layout));
    put_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(s.ownership));
    put_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(s.parallel));
    put_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(s.quant.format));
    put_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(s.quant.granularity));
    put_scalar<std::int32_t>(os, static_cast<std::int32_t>(s.quant.group_size));
    put_scalar<std::int32_t>(os, static_cast<std::int32_t>(s.quant.channel_axis));
    put_str(os, s.quant.scale_tensor);
    put_str(os, s.quant.zero_point_tensor);
    put_str(os, s.backing_tensor);
    put_scalar<std::int32_t>(os, static_cast<std::int32_t>(s.view_axis));
    put_scalar<std::int64_t>(os, s.view_start);
    put_scalar<std::int64_t>(os, s.view_length);
}
inline TensorDecl get_spec(std::istream& is)
{
    TensorDecl s;
    s.name = get_str(is);
    s.dtype = static_cast<DType>(get_scalar<std::uint8_t>(is));
    s.shape = get_shape(is);
    s.layout = static_cast<TensorLayoutKind>(get_scalar<std::uint8_t>(is));
    s.ownership = static_cast<TensorOwnershipKind>(get_scalar<std::uint8_t>(is));
    s.parallel = static_cast<TensorParallelKind>(get_scalar<std::uint8_t>(is));
    s.quant.format = static_cast<QuantFormat>(get_scalar<std::uint8_t>(is));
    s.quant.granularity = static_cast<QuantGranularity>(get_scalar<std::uint8_t>(is));
    s.quant.group_size = get_scalar<std::int32_t>(is);
    s.quant.channel_axis = get_scalar<std::int32_t>(is);
    s.quant.scale_tensor = get_str(is);
    s.quant.zero_point_tensor = get_str(is);
    s.backing_tensor = get_str(is);
    s.view_axis = get_scalar<std::int32_t>(is);
    s.view_start = get_scalar<std::int64_t>(is);
    s.view_length = get_scalar<std::int64_t>(is);
    return s;
}

}  // namespace detail

// Writes a materialized-weight cache file for `store` keyed by `cache_key` into
// `dir`. Best-effort: returns false (without throwing for recoverable cases) if
// the store doesn't fit the owned-buffer + views model, leaving normal loading
// unaffected. Returns true once the file is durably renamed into place.
inline bool write_weight_artifact_cache(
    const WeightStore& store,
    const std::string& cache_key,
    const std::filesystem::path& dir)
{
#if !PIE_CUDA_WEIGHT_ARTIFACT_CACHE_HAS_CUDA
    (void)store; (void)cache_key; (void)dir;
    return false;
#else
    using namespace detail;

    struct OwnedRoot {
        const TensorRecord* rec;
        std::uintptr_t base;
        std::uint64_t nbytes;
        std::uint64_t blob_offset;
        std::uint64_t checksum;
    };
    struct ViewEntry {
        const TensorRecord* rec;
        std::uint64_t root_index;
        std::uint64_t byte_offset;
    };

    // 1. Classify into owned roots and views; require a full spec on each.
    std::vector<OwnedRoot> owned;
    for (auto it = store.begin(); it != store.end(); ++it) {
        const TensorRecord& rec = it->second;
        if (!rec.has_spec) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight cache: declining — tensor '%s' has "
                "no TensorDecl\n", it->first.c_str());
            return false;
        }
        const DeviceTensor& t = rec.tensor;
        if (t.owns_memory() && t.data() != nullptr) {
            owned.push_back(OwnedRoot{
                &rec, reinterpret_cast<std::uintptr_t>(t.data()),
                static_cast<std::uint64_t>(t.nbytes()), 0, 0});
        }
    }

    std::vector<ViewEntry> views;
    for (auto it = store.begin(); it != store.end(); ++it) {
        const TensorRecord& rec = it->second;
        const DeviceTensor& t = rec.tensor;
        if (t.owns_memory() && t.data() != nullptr) {
            continue;  // already an owned root
        }
        const auto vbase = reinterpret_cast<std::uintptr_t>(t.data());
        const auto vbytes = static_cast<std::uint64_t>(t.nbytes());
        std::uint64_t root_index = UINT64_MAX;
        std::uint64_t byte_offset = 0;
        if (t.data() != nullptr) {
            for (std::uint64_t i = 0; i < owned.size(); ++i) {
                const auto& r = owned[i];
                if (vbase >= r.base && vbase + vbytes <= r.base + r.nbytes) {
                    root_index = i;
                    byte_offset = vbase - r.base;
                    break;
                }
            }
            if (root_index == UINT64_MAX) {
                std::fprintf(stderr,
                    "[pie-driver-cuda] weight cache: declining — view '%s' has "
                    "no owned backing buffer in the store\n",
                    it->first.c_str());
                return false;
            }
        }
        views.push_back(ViewEntry{&rec, root_index, byte_offset});
    }

    // 2. Assign blob offsets for owned roots.
    std::uint64_t blob_cursor = 0;
    for (auto& r : owned) {
        r.blob_offset = blob_cursor;
        blob_cursor += r.nbytes;
    }
    const std::uint64_t blob_section_bytes = blob_cursor;

    // 3. Write metadata + blobs to a temp file, then atomically rename.
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    // Don't risk filling the disk: require room for the blob section + a small
    // margin, else decline. The cache is best-effort, so a skipped write never
    // breaks loading — it just falls back to recompute next boot.
    {
        std::error_code space_ec;
        const auto space = std::filesystem::space(dir, space_ec);
        const std::uint64_t need = blob_section_bytes + (256ull << 20);  // +256 MiB
        if (!space_ec && space.available < need) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight cache: declining write — need %.1f GiB "
                "but only %.1f GiB free in %s (set a dir with more space, or "
                "unset PIE_CUDA_WEIGHT_CACHE_DIR)\n",
                static_cast<double>(need) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(space.available) / (1024.0 * 1024.0 * 1024.0),
                dir.string().c_str());
            return false;
        }
    }

    const auto final_path = dir / (cache_key + ".weights");
    const auto tmp_path = dir / (cache_key + ".weights.tmp");
    {
        std::ofstream os(tmp_path, std::ios::binary | std::ios::trunc);
        if (!os) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight cache: cannot open %s for write\n",
                tmp_path.string().c_str());
            return false;
        }

        os.write(kWeightArtifactMagic, sizeof(kWeightArtifactMagic));
        put_scalar<std::uint32_t>(os, kWeightArtifactFormatVersion);
        put_str(os, cache_key);
        put_scalar<std::uint64_t>(os, owned.size());
        put_scalar<std::uint64_t>(os, views.size());
        const auto& qmap = store.quant_meta_map();
        put_scalar<std::uint64_t>(os, qmap.size());
        put_scalar<std::uint64_t>(os, blob_section_bytes);

        std::vector<std::streampos> checksum_pos(owned.size());
        for (std::uint64_t i = 0; i < owned.size(); ++i) {
            const auto& r = owned[i];
            put_spec(os, r.rec->spec);
            put_scalar<std::uint64_t>(os, r.nbytes);
            put_scalar<std::uint64_t>(os, r.blob_offset);
            checksum_pos[i] = os.tellp();
            put_scalar<std::uint64_t>(os, std::uint64_t{0});  // checksum placeholder
        }
        for (const auto& v : views) {
            put_spec(os, v.rec->spec);
            put_scalar<std::uint64_t>(os, v.root_index);
            put_scalar<std::uint64_t>(os, v.byte_offset);
        }
        for (const auto& kv : qmap) {
            const QuantMeta& m = kv.second;
            put_str(os, kv.first);
            put_scalar<std::uint8_t>(os, static_cast<std::uint8_t>(m.kind));
            put_str(os, m.scale_name);
            put_str(os, m.zero_point_name);
            put_scalar<std::int32_t>(os, static_cast<std::int32_t>(m.group_size));
            put_scalar<std::int32_t>(os, static_cast<std::int32_t>(m.channel_axis));
        }

        // Blob section: D2H each owned buffer in chunks, write + checksum.
        std::vector<char> host(static_cast<std::size_t>(
            std::min<std::uint64_t>(kWeightArtifactChunkBytes,
                                    std::max<std::uint64_t>(blob_section_bytes, 1))));
        for (std::uint64_t i = 0; i < owned.size(); ++i) {
            auto& r = owned[i];
            std::uint64_t sum = kBlobHashSeed;
            std::uint64_t done = 0;
            const auto* src = static_cast<const std::uint8_t*>(r.rec->tensor.data());
            while (done < r.nbytes) {
                const std::size_t n = static_cast<std::size_t>(
                    std::min<std::uint64_t>(host.size(), r.nbytes - done));
                const cudaError_t e = cudaMemcpy(
                    host.data(), src + done, n, cudaMemcpyDeviceToHost);
                if (e != cudaSuccess) {
                    std::fprintf(stderr,
                        "[pie-driver-cuda] weight cache: D2H failed for '%s': %s\n",
                        r.rec->spec.name.c_str(), cudaGetErrorString(e));
                    os.close();
                    std::filesystem::remove(tmp_path, ec);
                    return false;
                }
                sum = blob_hash_update(sum, host.data(), n);
                os.write(host.data(), static_cast<std::streamsize>(n));
                done += n;
            }
            r.checksum = sum;
        }

        for (std::uint64_t i = 0; i < owned.size(); ++i) {
            os.seekp(checksum_pos[i]);
            put_scalar<std::uint64_t>(os, owned[i].checksum);
        }
        os.seekp(0, std::ios::end);
        os.flush();
        if (!os) {
            std::fprintf(stderr, "[pie-driver-cuda] weight cache: write error\n");
            os.close();
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
    }

    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        std::fprintf(stderr,
            "[pie-driver-cuda] weight cache: rename failed: %s\n",
            ec.message().c_str());
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    return true;
#endif
}

// Attempts to populate `builder`'s store from a cache file keyed by `cache_key`
// in `dir`. Returns true on a verified hit (store populated + finalized); false
// on miss / key mismatch / corruption (caller falls back to normal load). May
// throw on a hard error mid-populate; callers wrap in try/catch and discard the
// partially-populated store on throw.
inline bool read_weight_artifact_cache(
    WeightStoreBuilder& builder,
    const std::string& cache_key,
    const std::filesystem::path& dir)
{
#if !PIE_CUDA_WEIGHT_ARTIFACT_CACHE_HAS_CUDA
    (void)builder; (void)cache_key; (void)dir;
    return false;
#else
    using namespace detail;

    const auto path = dir / (cache_key + ".weights");
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return false;
    }
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        return false;
    }

    char magic[8];
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, kWeightArtifactMagic, sizeof(magic)) != 0) {
        return false;
    }
    if (get_scalar<std::uint32_t>(is) != kWeightArtifactFormatVersion) {
        return false;
    }
    if (get_str(is) != cache_key) {
        return false;  // stale/foreign cache; treat as miss
    }
    const auto num_owned = get_scalar<std::uint64_t>(is);
    const auto num_views = get_scalar<std::uint64_t>(is);
    const auto num_quant = get_scalar<std::uint64_t>(is);
    const auto blob_section_bytes = get_scalar<std::uint64_t>(is);

    struct OwnedHdr { TensorDecl spec; std::uint64_t nbytes, blob_offset, checksum; };
    std::vector<OwnedHdr> owned(num_owned);
    for (auto& o : owned) {
        o.spec = get_spec(is);
        o.nbytes = get_scalar<std::uint64_t>(is);
        o.blob_offset = get_scalar<std::uint64_t>(is);
        o.checksum = get_scalar<std::uint64_t>(is);
    }
    struct ViewHdr { TensorDecl spec; std::uint64_t root_index, byte_offset; };
    std::vector<ViewHdr> views(num_views);
    for (auto& v : views) {
        v.spec = get_spec(is);
        v.root_index = get_scalar<std::uint64_t>(is);
        v.byte_offset = get_scalar<std::uint64_t>(is);
    }
    struct QuantHdr {
        std::string name; std::uint8_t kind; std::string scale_name;
        std::string zero_point_name; std::int32_t group_size, channel_axis;
    };
    std::vector<QuantHdr> quants(num_quant);
    for (auto& q : quants) {
        q.name = get_str(is);
        q.kind = get_scalar<std::uint8_t>(is);
        q.scale_name = get_str(is);
        q.zero_point_name = get_str(is);
        q.group_size = get_scalar<std::int32_t>(is);
        q.channel_axis = get_scalar<std::int32_t>(is);
    }
    if (!is) {
        return false;  // truncated metadata
    }
    const std::streampos blob_section_pos = is.tellg();

    // Allocate + load owned roots; remember device base pointers by index.
    // Per-phase timing (read / verify / h2d) is accumulated so the dominant
    // cost is measured rather than guessed before optimizing the reload.
    const bool verify =
        std::getenv("PIE_CUDA_WEIGHT_CACHE_NO_VERIFY") == nullptr;
    const bool profile = []{
        const char* p = std::getenv("PIE_WEIGHT_LOADER_PROFILE");
        return p != nullptr && p[0] != '\0' && p[0] != '0';
    }();
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };
    double read_ms = 0, verify_ms = 0, h2d_ms = 0;
    std::uint64_t total_bytes = 0;

    std::vector<void*> root_base(num_owned, nullptr);
    // Same chunk-size formula as the write path so the boundary-dependent
    // blob hash chains identically (see blob_hash_update).
    std::vector<char> host(static_cast<std::size_t>(
        std::min<std::uint64_t>(kWeightArtifactChunkBytes,
                                std::max<std::uint64_t>(blob_section_bytes, 1))));
    for (std::uint64_t i = 0; i < num_owned; ++i) {
        const auto& o = owned[i];
        DeviceTensor t = DeviceTensor::allocate(o.spec.dtype, o.spec.shape);
        if (t.nbytes() != o.nbytes) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight cache: nbytes mismatch for '%s' — "
                "discarding cache\n", o.spec.name.c_str());
            return false;
        }
        is.seekg(blob_section_pos + static_cast<std::streamoff>(o.blob_offset));
        std::uint64_t sum = kBlobHashSeed;
        std::uint64_t done = 0;
        auto* dst = static_cast<std::uint8_t*>(t.data());
        while (done < o.nbytes) {
            const std::size_t n = static_cast<std::size_t>(
                std::min<std::uint64_t>(host.size(), o.nbytes - done));
            auto tr0 = clock::now();
            is.read(host.data(), static_cast<std::streamsize>(n));
            read_ms += ms(clock::now() - tr0);
            if (!is) {
                std::fprintf(stderr,
                    "[pie-driver-cuda] weight cache: short read for '%s'\n",
                    o.spec.name.c_str());
                return false;
            }
            if (verify) {
                auto tv0 = clock::now();
                sum = blob_hash_update(sum, host.data(), n);
                verify_ms += ms(clock::now() - tv0);
            }
            auto th0 = clock::now();
            const cudaError_t e = cudaMemcpy(
                dst + done, host.data(), n, cudaMemcpyHostToDevice);
            h2d_ms += ms(clock::now() - th0);
            if (e != cudaSuccess) {
                std::fprintf(stderr,
                    "[pie-driver-cuda] weight cache: H2D failed for '%s': %s\n",
                    o.spec.name.c_str(), cudaGetErrorString(e));
                return false;
            }
            done += n;
        }
        if (verify && sum != o.checksum) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight cache: checksum mismatch for '%s' — "
                "discarding cache\n", o.spec.name.c_str());
            return false;
        }
        total_bytes += o.nbytes;
        root_base[i] = t.data();
        builder.insert(o.spec.name, std::move(t), o.spec);
    }
    if (profile) {
        const double gib = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
        const double total = read_ms + verify_ms + h2d_ms;
        std::fprintf(stderr,
            "[pie-driver-cuda] weight cache reload: %.2f GiB  read=%.0fms "
            "verify=%.0fms(%s) h2d=%.0fms  serial_sum=%.0fms (%.2f GiB/s eff)\n",
            gib, read_ms, verify_ms, verify ? "on" : "off", h2d_ms, total,
            total > 0 ? gib / (total / 1000.0) : 0.0);
    }

    // Reconstruct views into their owned roots (physical), with their spec.
    for (const auto& v : views) {
        void* base = nullptr;
        if (v.root_index != UINT64_MAX && v.root_index < num_owned) {
            base = static_cast<std::uint8_t*>(root_base[v.root_index]) + v.byte_offset;
        }
        builder.insert(v.spec.name,
                       DeviceTensor::view(base, v.spec.dtype, v.spec.shape),
                       v.spec);
    }

    // Rebuild quant metadata (pointers resolved by name from the populated store).
    for (const auto& q : quants) {
        QuantMeta meta;
        meta.kind = static_cast<QuantMeta::Kind>(q.kind);
        meta.scale_name = q.scale_name;
        meta.zero_point_name = q.zero_point_name;
        meta.group_size = q.group_size;
        meta.channel_axis = q.channel_axis;
        if (!q.scale_name.empty()) {
            auto it = builder.find(q.scale_name);
            if (it != builder.end()) {
                meta.scale = &it->second.tensor;
            }
        }
        if (!q.zero_point_name.empty()) {
            auto it = builder.find(q.zero_point_name);
            if (it != builder.end()) {
                meta.zero_point = &it->second.tensor;
            }
        }
        builder.set_quant_meta(q.name, std::move(meta));
    }

    builder.finalize();
    return true;
#endif
}

}  // namespace pie_cuda_driver
