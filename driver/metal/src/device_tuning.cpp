// Per-device tuning: the table, and the env override that measured it.
//
// The query itself is Objective-C (Metal for the family, IOKit for the core
// count) and lives in `device_tuning_apple.mm`; this translation unit is
// plain C++ and holds the part that is worth reading.

#include "device_tuning.hpp"

#include <cstdlib>

namespace pie::metal {

namespace {

int env_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    char* end = nullptr;
    const long v = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || v <= 0) return fallback;
    return int(v);
}

DeviceTuning tuning_for(const DeviceInfo& info) {
    DeviceTuning t;  // the M1 measurements, unchanged for any device not below
    switch (info.apple_family) {
        case 9:
            // M3/M4 generation. Measured on an M4 Pro (20 cores) with
            // gemma-4-E4B at concurrency 8 -- the batch where the two
            // settings actually diverge -- same binary via
            // PIE_METAL_QMM_MIN_BATCH, arms alternated, quiet host:
            // 138.90 tok/s at 12 against 144.04 at 8, +3.7%. See
            // `device_tuning.hpp` for the individual runs. Applied by FAMILY
            // and not by core count -- the
            // crossover is set by per-core matrix throughput, which the
            // family names and the core count does not.
            t.qmm_min_batch = 8;
            break;
        default:
            break;
    }
    t.qmm_min_batch = env_int("PIE_METAL_QMM_MIN_BATCH", t.qmm_min_batch);
    return t;
}

}  // namespace

#if !defined(__APPLE__)
// No Metal device to ask. Every field stays 0, which selects the defaults --
// the same constants this driver shipped before the tuning layer existed.
DeviceInfo query_device_info() { return DeviceInfo{}; }
#endif

const DeviceInfo& device_info() {
    static const DeviceInfo info = query_device_info();
    return info;
}

const DeviceTuning& device_tuning() {
    static const DeviceTuning t = tuning_for(device_info());
    return t;
}

int qmm_min_batch() { return device_tuning().qmm_min_batch; }

}  // namespace pie::metal
