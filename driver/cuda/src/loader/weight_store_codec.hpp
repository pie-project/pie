#pragma once

// Weight-store codec (PIEWSTOR format): serialize a materialized WeightStore to a
// byte stream, and restore one from a byte buffer. A general conversion/load
// primitive for finished device weights — it owns the on-disk byte format and the
// integrity checksum but knows nothing about caches, directories, keys, or
// eviction. The artifact cache (model/weight_artifact_cache.hpp) is one consumer:
// it owns that policy and uses serialize/restore as opaque byte conversion.
//
// Memory model: the WeightStore is a set of *owned* DeviceTensors (the storage
// arena + any separately-owned buffers) and *view* DeviceTensors that point into
// one of those owned buffers. We store each owned buffer's bytes once and record
// each view as (owning-root index, byte offset). The full TensorDecl spec and
// the quant_meta map are serialized so the rebuilt store passes finalize()
// validation unchanged.
//
// restore() streams the owned blobs to the device through the shared pinned
// staged-H2D engine (loader/staged_h2d.hpp) — the same path the cold materialize
// uses — and verifies each blob's checksum concurrently with the copy.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <istream>
#include <ostream>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include "model/weight_store.hpp"
#include "loader/tensor_spec.hpp"
#include "tensor.hpp"

#if __has_include(<cuda_runtime.h>)
#define PIE_CUDA_WEIGHT_CODEC_HAS_CUDA 1
#include <cuda_runtime.h>
#include "loader/staged_h2d.hpp"
#else
#define PIE_CUDA_WEIGHT_CODEC_HAS_CUDA 0
#endif

namespace pie_cuda_driver::weight_codec {

inline constexpr char kMagic[8] = {'P', 'I', 'E', 'W', 'S', 'T', 'O', 'R'};
inline constexpr std::uint32_t kFormatVersion = 3;
// Blob checksum fold granularity. Boundary-DEPENDENT (see blob_hash_update), so
// serialize and restore MUST fold in identical chunks; both use this constant.
inline constexpr std::uint64_t kChunkBytes = 64ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kBlobHashSeed = 0xcbf29ce484222325ull;

// Streaming integrity hash over blob bytes — folds 32 bytes/step across 4 lanes
// (breaks the multiply dependency chain) with an 8-byte then 1-byte tail.
// Host-endian, non-cryptographic: a same-machine corruption check, not security.
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

// Fold a whole owned blob into its checksum using the canonical chunking.
inline std::uint64_t blob_checksum(const std::uint8_t* data, std::uint64_t nbytes)
{
    std::uint64_t sum = kBlobHashSeed;
    std::uint64_t done = 0;
    while (done < nbytes) {
        const std::size_t n =
            static_cast<std::size_t>(std::min<std::uint64_t>(kChunkBytes, nbytes - done));
        sum = blob_hash_update(sum, data + done, n);
        done += n;
    }
    return sum;
}

namespace detail {

// --- little binary writers over std::ostream -------------------------------
template <typename T>
inline void put_scalar(std::ostream& os, const T& v)
{
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
inline void put_str(std::ostream& os, const std::string& s)
{
    put_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(s.size()));
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}
inline void put_shape(std::ostream& os, const std::vector<std::int64_t>& shape)
{
    put_scalar<std::uint32_t>(os, static_cast<std::uint32_t>(shape.size()));
    for (const auto d : shape) {
        put_scalar<std::int64_t>(os, d);
    }
}
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

// --- seekable read-only stream over an in-memory buffer (the mmap'd file) ---
// Lets restore parse the (small) manifest with the same readers the writer used,
// while the (huge) blob section is accessed directly through the buffer pointer.
struct membuf : std::streambuf {
    membuf(const char* base, std::size_t n)
    {
        char* p = const_cast<char*>(base);
        setg(p, p, p + n);
    }
    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode) override
    {
        char* target = (dir == std::ios_base::beg)   ? eback() + off
                       : (dir == std::ios_base::cur) ? gptr() + off
                                                     : egptr() + off;
        if (target < eback() || target > egptr()) {
            return pos_type(off_type(-1));
        }
        setg(eback(), target, egptr());
        return pos_type(target - eback());
    }
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override
    {
        return seekoff(off_type(pos), std::ios_base::beg, which);
    }
};
struct imemstream : virtual membuf, std::istream {
    imemstream(const char* base, std::size_t n)
        : membuf(base, n), std::istream(static_cast<std::streambuf*>(this)) {}
};

template <typename T>
inline T get_scalar(std::istream& is)
{
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}
inline std::string get_str(std::istream& is)
{
    const auto len = get_scalar<std::uint32_t>(is);
    std::string s(len, '\0');
    is.read(s.data(), static_cast<std::streamsize>(len));
    return s;
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

// Serialize `store` (keyed by `key`) to `out`. Best-effort: returns false
// (without throwing for recoverable cases) if the store doesn't fit the
// owned-buffer + views model, so the caller can simply skip persisting. `out`
// must be seekable (the per-blob checksums are back-patched after the D2H pass).
inline bool serialize_weight_store(const WeightStore& store,
                                   const std::string& key,
                                   std::ostream& out)
{
#if !PIE_CUDA_WEIGHT_CODEC_HAS_CUDA
    (void)store; (void)key; (void)out;
    return false;
#else
    using namespace detail;

    struct OwnedRoot {
        const TensorRecord* rec;
        std::uintptr_t base;
        std::uint64_t nbytes;
        std::uint64_t blob_offset;
    };
    struct ViewEntry {
        const TensorRecord* rec;
        std::uint64_t root_index;
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
                static_cast<std::uint64_t>(t.nbytes()), 0});
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
                    "[pie-driver-cuda] weight codec: declining — view '%s' has "
                    "no owned backing buffer in the store\n", it->first.c_str());
                return false;
            }
        }
        views.push_back(ViewEntry{&rec, root_index, byte_offset});
    }

    std::uint64_t blob_cursor = 0;
    for (auto& r : owned) {
        r.blob_offset = blob_cursor;
        blob_cursor += r.nbytes;
    }
    const std::uint64_t blob_section_bytes = blob_cursor;

    out.write(kMagic, sizeof(kMagic));
    put_scalar<std::uint32_t>(out, kFormatVersion);
    put_str(out, key);
    put_scalar<std::uint64_t>(out, owned.size());
    put_scalar<std::uint64_t>(out, views.size());
    const auto& qmap = store.quant_meta_map();
    put_scalar<std::uint64_t>(out, qmap.size());
    put_scalar<std::uint64_t>(out, blob_section_bytes);

    std::vector<std::streampos> checksum_pos(owned.size());
    for (std::uint64_t i = 0; i < owned.size(); ++i) {
        const auto& r = owned[i];
        put_spec(out, r.rec->spec);
        put_scalar<std::uint64_t>(out, r.nbytes);
        put_scalar<std::uint64_t>(out, r.blob_offset);
        checksum_pos[i] = out.tellp();
        put_scalar<std::uint64_t>(out, std::uint64_t{0});  // checksum placeholder
    }
    for (const auto& v : views) {
        put_spec(out, v.rec->spec);
        put_scalar<std::uint64_t>(out, v.root_index);
        put_scalar<std::uint64_t>(out, v.byte_offset);
    }
    for (const auto& kv : qmap) {
        const QuantMeta& m = kv.second;
        put_str(out, kv.first);
        put_scalar<std::uint8_t>(out, static_cast<std::uint8_t>(m.kind));
        put_str(out, m.scale_name);
        put_str(out, m.zero_point_name);
        put_scalar<std::int32_t>(out, static_cast<std::int32_t>(m.group_size));
        put_scalar<std::int32_t>(out, static_cast<std::int32_t>(m.channel_axis));
    }

    // Blob section: D2H each owned buffer in chunks, write + checksum.
    std::vector<std::uint64_t> checksums(owned.size());
    std::vector<char> host(static_cast<std::size_t>(
        std::min<std::uint64_t>(kChunkBytes, std::max<std::uint64_t>(blob_section_bytes, 1))));
    for (std::uint64_t i = 0; i < owned.size(); ++i) {
        std::uint64_t sum = kBlobHashSeed;
        std::uint64_t done = 0;
        const auto* src = static_cast<const std::uint8_t*>(owned[i].rec->tensor.data());
        while (done < owned[i].nbytes) {
            const std::size_t n = static_cast<std::size_t>(
                std::min<std::uint64_t>(host.size(), owned[i].nbytes - done));
            const cudaError_t e = cudaMemcpy(host.data(), src + done, n,
                                             cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) {
                std::fprintf(stderr,
                    "[pie-driver-cuda] weight codec: D2H failed for '%s': %s\n",
                    owned[i].rec->spec.name.c_str(), cudaGetErrorString(e));
                return false;
            }
            sum = blob_hash_update(sum, host.data(), n);
            out.write(host.data(), static_cast<std::streamsize>(n));
            done += n;
        }
        checksums[i] = sum;
    }
    for (std::uint64_t i = 0; i < owned.size(); ++i) {
        out.seekp(checksum_pos[i]);
        put_scalar<std::uint64_t>(out, checksums[i]);
    }
    out.seekp(0, std::ios::end);
    return static_cast<bool>(out);
#endif
}

// Restore a store from a contiguous byte buffer `data` (e.g. a mmap'd file),
// which must remain valid for the duration of the call. Returns
// false on magic/version/key mismatch or corruption (a "miss" the caller falls
// back from); the blobs are streamed to device through `pool` and each blob's
// checksum is verified concurrently with the copy when `verify` is set.
inline bool restore_weight_store(const std::uint8_t* data, std::size_t size,
                                 const std::string& expected_key, bool verify,
                                 WeightStoreBuilder& builder,
                                 PinnedLanePool& pool)
{
#if !PIE_CUDA_WEIGHT_CODEC_HAS_CUDA
    (void)data; (void)size; (void)expected_key; (void)verify; (void)builder; (void)pool;
    return false;
#else
    using namespace detail;

    imemstream is(reinterpret_cast<const char*>(data), size);
    char magic[8];
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
        return false;
    }
    if (get_scalar<std::uint32_t>(is) != kFormatVersion) {
        return false;
    }
    if (get_str(is) != expected_key) {
        return false;  // key mismatch — stale/foreign store
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
    const std::uint64_t blob_section_pos =
        static_cast<std::uint64_t>(is.tellg());
    if (blob_section_pos + blob_section_bytes > size) {
        return false;  // truncated blob section
    }
    const std::uint8_t* blobs = data + blob_section_pos;

    // Allocate owned roots and queue one staged copy each, sourced straight from
    // the (mmap'd) blob section. The shared engine pins + pipelines the H2D.
    std::vector<void*> root_base(num_owned, nullptr);
    std::vector<StagedCopy> copies;
    copies.reserve(num_owned);
    for (std::uint64_t i = 0; i < num_owned; ++i) {
        const auto& o = owned[i];
        if (o.blob_offset + o.nbytes > blob_section_bytes) {
            return false;
        }
        DeviceTensor t = DeviceTensor::allocate(o.spec.dtype, o.spec.shape);
        if (t.nbytes() != o.nbytes) {
            std::fprintf(stderr,
                "[pie-driver-cuda] weight codec: nbytes mismatch for '%s' — "
                "discarding restore\n", o.spec.name.c_str());
            return false;
        }
        copies.push_back(StagedCopy{t.data(), blobs + o.blob_offset, o.nbytes});
        root_base[i] = t.data();
        builder.insert(o.spec.name, std::move(t), o.spec);
    }

    // Verify checksums on the CPU concurrently with the DMA: the copy is PCIe-
    // bound while the hash is a memory read, and both touch the same warm pages.
    std::thread verifier;
    bool checksum_ok = true;
    if (verify) {
        verifier = std::thread([&] {
            for (std::uint64_t i = 0; i < num_owned && checksum_ok; ++i) {
                if (blob_checksum(blobs + owned[i].blob_offset, owned[i].nbytes)
                        != owned[i].checksum) {
                    checksum_ok = false;
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
    if (verify && !checksum_ok) {
        std::fprintf(stderr,
            "[pie-driver-cuda] weight codec: checksum mismatch — discarding restore\n");
        return false;
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

}  // namespace pie_cuda_driver::weight_codec
