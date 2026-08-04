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
// key->path) and whether writing one is worth the disk. The file itself --
// format, placement, digests, the temp-file-and-rename that publishes it -- is
// the loader codec's (loader/weight_store_codec.hpp), which this layer drives by
// path and never looks inside.
//
// Located by `[model] weight_cache_dir`, which the worker resolves to
// $PIE_HOME/models when the operator leaves it empty. Empty here (the driver
// was told nothing) disables the cache entirely: zero reads, zero writes. The
// write declines if free space < blob size + margin, because the artifact is
// the size of the materialized weights — tens to hundreds of GB. Every owned
// blob carries a fast checksum, always verified on reload; a key or
// format-version mismatch is a miss.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "config.hpp"

#include "model/weight_store.hpp"
#include "loader/weight_store_codec.hpp"
#include "loader/loader_config.hpp"

#if __has_include(<cuda_runtime.h>)
#define PIE_CUDA_WEIGHT_ARTIFACT_CACHE_HAS_CUDA 1
#else
#define PIE_CUDA_WEIGHT_ARTIFACT_CACHE_HAS_CUDA 0
#endif

namespace pie_cuda_driver {

// Returns the configured artifact cache directory, or empty if the feature is
// disabled (the driver was given no directory).
inline std::filesystem::path weight_artifact_cache_dir()
{
    const std::string& dir = pie_cuda_driver::weight_cache_dir();
    return dir.empty() ? std::filesystem::path{} : std::filesystem::path(dir);
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
                "but only %.1f GiB free in %s (point [model] weight_cache_dir "
                "at a disk with more space)\n",
                static_cast<double>(need) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(space.available) / (1024.0 * 1024.0 * 1024.0),
                dir.string().c_str());
            return false;
        }
    }

    return weight_codec::serialize_weight_store(
        store, cache_key, dir / (cache_key + ".weights"));
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

    // Always verify. A silently-corrupt weight artifact produces garbage
    // tokens with no error, which is not a trade any operator should be
    // offered for a few seconds of load time.
    constexpr bool verify = true;
    constexpr bool profile = false;
    const auto t0 = std::chrono::steady_clock::now();

    // A local lane pool sized like the cold reader path; restore streams the
    // payloads through it (pinned + pipelined) the same way materialize does.
    PinnedLanePool pool(std::max<std::size_t>(loader_config::reader_lane_count(), 1),
                        loader_config::reader_buf_bytes());
    const bool ok = weight_codec::restore_weight_store(
        path, cache_key, verify, builder, pool);
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

    return ok;
#endif
}

}  // namespace pie_cuda_driver
