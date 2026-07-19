#pragma once

// Materialized-weight artifact cache (driver policy layer).
//
// The materialized weights are a deterministic function of the compile cache
// key (checkpoint files + config + quant scheme + TP layout + ABI version), yet
// they are otherwise recomputed every boot — for FP8->MXFP4 models that is a
// large transcode cost. This cache snapshots the finished device weights after
// the first load, keyed by that authoritative key, so a warm boot reloads them
// straight into device memory and skips compile + materialize.
//
// This file owns only the *policy*: where the artifact lives (the cache dir +
// key->path), the durable write (free-space guard + temp file + atomic rename),
// and the read (mmap the file, restore through the shared staged-H2D engine).
// The byte format + integrity checksum live in the loader codec
// (loader/weight_store_codec.hpp); this layer treats them as an opaque stream.
//
// OFF BY DEFAULT — strictly opt-in via PIE_CUDA_WEIGHT_CACHE_DIR. With the env
// unset/empty the cache never reads or writes (zero disk). Even when enabled,
// the write declines if free space < blob size + margin (the artifact is the
// size of the materialized weights — tens to hundreds of GB). Each owned blob
// carries a fast checksum verified on reload (skip with
// PIE_CUDA_WEIGHT_CACHE_NO_VERIFY); key/format-version mismatch => miss.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "model/weight_store.hpp"
#include "loader/weight_store_codec.hpp"
#include "loader/loader_config.hpp"

#if __has_include(<cuda_runtime.h>)
#define PIE_CUDA_WEIGHT_ARTIFACT_CACHE_HAS_CUDA 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define PIE_CUDA_WEIGHT_ARTIFACT_CACHE_HAS_CUDA 0
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

// Writes a materialized-weight cache file for `store` keyed by `cache_key` into
// `dir`. Best-effort: returns false (without throwing for recoverable cases) if
// the store doesn't serialize or there isn't room, leaving normal loading
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
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    // Don't risk filling the disk: require room for the materialized blobs + a
    // small margin, else decline. A skipped write never breaks loading — it
    // just falls back to recompute next boot.
    {
        std::error_code space_ec;
        const auto space = std::filesystem::space(dir, space_ec);
        const std::uint64_t need = store.total_bytes() + (256ull << 20);  // +256 MiB
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
        if (!weight_codec::serialize_weight_store(store, cache_key, os)) {
            os.close();
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
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
    const auto path = dir / (cache_key + ".weights");
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return false;
    }

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return false;
    }
    const std::size_t size = static_cast<std::size_t>(st.st_size);
    void* map = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) {
        return false;
    }
    ::madvise(map, size, MADV_SEQUENTIAL);

    const bool verify =
        std::getenv("PIE_CUDA_WEIGHT_CACHE_NO_VERIFY") == nullptr;
    const bool profile = []{
        const char* p = std::getenv("PIE_LOAD_EXECUTOR_PROFILE");
        return p != nullptr && p[0] != '\0' && p[0] != '0';
    }();
    const auto t0 = std::chrono::steady_clock::now();

    // A local lane pool sized like the cold reader path; restore streams the
    // blobs through it (pinned + pipelined) the same way materialize does.
    PinnedLanePool pool(std::max<std::size_t>(loader_config::reader_lane_count(), 1),
                        loader_config::reader_buf_bytes());
    bool ok = false;
    try {
        ok = weight_codec::restore_weight_store(
            static_cast<const std::uint8_t*>(map), size, cache_key, verify,
            builder, pool);
    } catch (...) {
        ::munmap(map, size);
        throw;
    }
    const auto t1 = std::chrono::steady_clock::now();

    if (ok && profile) {
        const double gib = static_cast<double>(builder.store().total_bytes()) /
            (1024.0 * 1024.0 * 1024.0);
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr,
            "[pie-driver-cuda] weight cache reload: %.2f GiB in %.0fms "
            "(%.2f GiB/s, verify=%s)\n",
            gib, ms, ms > 0 ? gib / (ms / 1000.0) : 0.0, verify ? "on" : "off");
    }

    ::munmap(map, size);
    return ok;
#endif
}

}  // namespace pie_cuda_driver
