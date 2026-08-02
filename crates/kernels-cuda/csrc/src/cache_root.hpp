#pragma once

// Where every on-disk cache in this driver derives its path from.
//
// One mutable global, published once by the shell out of `[cache] dir` in the
// TOML (normally `$PIE_HOME/cache`) and read by each cache that needs a place
// to put a file. Empty means the engine did not say, which happens when a
// driver runs against a hand-written TOML; the caches then fall back to their
// own XDG derivation.
//
// ## Why this is a file, and why the file is here
//
// It was four lines in the shell's `config.hpp`, which is the natural home for
// something read out of a config file. But `config.hpp` also pulls
// `batch/planner_calibration.hpp` and `store/kv_cache_format.hpp`, and
// `ops/tuning_cache.hpp` -- a kernel-side autotuning memo -- needs the cache
// root and nothing else. Including the shell's config header to get a string
// was the single edge in either direction between `kernels`/`ops` and the rest
// of the driver.
//
// So the four lines moved rather than the tangle staying. This is not a kernel
// concern and does not pretend to be; it sits on the kernels side because that
// is the lower of the two crates, and a header both ends include has to be
// somewhere the lower one can see.
//
// `inline` function-local statics, so the two archives share one instance
// after link rather than each getting its own -- which is exactly the failure
// this would have if it were a plain `extern` global defined on one side.

#include <string>

namespace pie_cuda_driver {

/// The cache root, writable. Only the config loader should assign to it.
inline std::string& mutable_cache_dir() {
    static std::string dir;
    return dir;
}

/// The cache root, as every cache reads it. Empty = fall back to XDG.
inline const std::string& cache_dir() { return mutable_cache_dir(); }

}  // namespace pie_cuda_driver
