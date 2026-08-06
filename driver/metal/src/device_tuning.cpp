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

bool env_bool(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    // Unlike `env_int`, zero is a VALUE here and not a rejected one.
    return !(raw[0] == '0' && raw[1] == '\0');
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
            // Same machine, same reasoning, a different constant: with 20
            // cores rather than 32 the wide tile's smaller grid fills the
            // machine sooner, so the tile crossover moves down. Measured with
            // `roofline_probe`; the table is in `device_tuning.hpp`.
            t.qmm_bn_crossover_tg = 96;
            // Both crossovers, because 8 is what this device measured and
            // shipped with while there was one of them, and a mixture ran it
            // too. Splitting the constant may not move a device that was
            // measured before the split existed: the M4's routed half is
            // unmeasured, and leaving it at 12 here would be a change dressed
            // as a default.
            t.qmm_min_batch_moe = 8;
            break;
        case 8:
            // M2 generation. Measured on an M2 Max (38 cores), one binary via
            // the env overrides, arms alternated within each batch, three
            // reps: eight is the first batch where the GEMM beats the GEMV on
            // all four dense checkpoints (Llama-1B +17.6%, Llama-3B +19.2%,
            // Qwen3-1.7B +14.4%, gemma-4-E2B +4.6%) and the first with no
            // regression on any of them.
            //
            // The ROUTED crossover stays at the M1 number, and that is the
            // finding rather than an omission: at the same batches the GEMV
            // still wins on every mixture measured -- Qwen3-30B by 8%,
            // gemma-4-26B by 12%, gpt-oss-20B by nothing either way. See
            // `DeviceTuning::qmm_min_batch_moe` for the runs.
            //
            // Only this one constant. The rest of the table is still the M1's
            // on this machine because nothing here has measured them, and a
            // family entry that guessed at the others would be worse than the
            // default it replaced.
            t.qmm_min_batch = 8;
            break;
        default:
            break;
    }
    // Every tuned constant gets an override, and for the same reason the first
    // one did: measuring a crossover means running the same binary twice with
    // different answers, and a rebuild between arms is a different binary.
    t.qmm_min_batch = env_int("PIE_METAL_QMM_MIN_BATCH", t.qmm_min_batch);
    // The dense override carries the routed one unless it is named separately.
    // A sweep that moved only the dense number on a mixture would measure a
    // model that never changed path and read the flat curve as the crossover
    // not mattering.
    t.qmm_min_batch_moe =
        env_int("PIE_METAL_QMM_MIN_BATCH_MOE",
                env_int("PIE_METAL_QMM_MIN_BATCH", t.qmm_min_batch_moe));
    t.qmm_bn_crossover_tg =
        env_int("PIE_METAL_QMM_BN_CROSSOVER_TG", t.qmm_bn_crossover_tg);
    t.moe_tile_mid_per = env_int("PIE_METAL_MOE_TILE_MID_PER", t.moe_tile_mid_per);
    t.moe_tile_wide_per = env_int("PIE_METAL_MOE_TILE_WIDE_PER", t.moe_tile_wide_per);
    t.fp16_qmm = env_bool("PIE_METAL_FP16_QMM", t.fp16_qmm);
    t.sdpa_tile_min_rows_per_request =
        env_int("PIE_METAL_SDPA_TILE_MIN_ROWS", t.sdpa_tile_min_rows_per_request);
    t.moe_batch_min_per_expert =
        env_int("PIE_METAL_MOE_BATCH_MIN_PER_EXPERT", t.moe_batch_min_per_expert);
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

int qmm_min_batch(bool is_moe) {
    const DeviceTuning& t = device_tuning();
    return is_moe ? t.qmm_min_batch_moe : t.qmm_min_batch;
}
int qmm_bn_crossover_tg() { return device_tuning().qmm_bn_crossover_tg; }
int moe_tile_mid_per() { return device_tuning().moe_tile_mid_per; }
int moe_tile_wide_per() { return device_tuning().moe_tile_wide_per; }
bool fp16_qmm() { return device_tuning().fp16_qmm; }
int sdpa_tile_min_rows_per_request() {
    return device_tuning().sdpa_tile_min_rows_per_request;
}
int moe_batch_min_per_expert() { return device_tuning().moe_batch_min_per_expert; }

}  // namespace pie::metal
