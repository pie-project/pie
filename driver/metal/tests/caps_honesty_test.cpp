// Truthful caps gate (metal_ptir_plan.md Phase 1, §12 "Caps honesty" /
// §5.5). Pure host (registers no program, runs no forward, needs no
// checkpoint or Apple/Metal — `pie_metal_create` only parses `config.json`
// at `initialize()`), so this binary always builds and runs.
//
// Phase 1a's Metal forward is exactly ONE resident linear-sequence
// RawMetalDecoder (a fixed `max_ctx_ = 4096` KV/GDN ring), not the runtime's
// multi-tenant paged pool — this proves the driver stops echoing the config
// file's generic multi-request capacity once it knows the checkpoint needs
// the forward (GDN-hybrid / qwen3.6 target), instead of advertising a
// concurrency/context budget it cannot structurally honor. Every cap is a
// MIN against the ring's 4096-token capacity (never an inflation of a
// smaller config/model limit), and the page count is a CEIL division so a
// non-power-of-two `kv_page_size` still covers the full ring.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "entry.hpp"

namespace {

int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

void notify_cb(void*, std::uint64_t, std::uint64_t) {}

std::string caps_for(const std::string& config_toml_path) {
    PieDriverCreateDesc create{};
    create.abi_version = PIE_DRIVER_ABI_VERSION;
    create.config_bytes.ptr =
        reinterpret_cast<const std::uint8_t*>(config_toml_path.data());
    create.config_bytes.len = config_toml_path.size();
    create.runtime.abi_version = PIE_DRIVER_ABI_VERSION;
    create.runtime.ctx = nullptr;
    create.runtime.notify = notify_cb;
    PieDriverCaps caps{};
    PieDriver* driver = pie_metal_create(&create, &caps);
    if (driver == nullptr) return {};
    std::string json(reinterpret_cast<const char*>(caps.json_bytes), caps.json_len);
    pie_metal_destroy(driver);
    return json;
}

// Writes a scratch HF snapshot dir (config.json only) + a scratch driver
// config.toml pointing at it, runs `pie_metal_create`, and returns the caps
// JSON. Cleans up both scratch paths before returning.
std::string caps_for_scenario(const std::string& tag, const std::string& config_json_body,
                              std::uint32_t kv_page_size, std::uint32_t total_pages,
                              std::uint32_t max_forward_tokens,
                              std::uint32_t max_forward_requests) {
    const std::string hf_dir = "caps_honesty_" + tag + "_hf";
    std::filesystem::create_directories(hf_dir);
    {
        std::ofstream f(hf_dir + "/config.json", std::ios::trunc);
        f << config_json_body;
    }
    const std::string config_path = "caps_honesty_" + tag + ".generated.toml";
    {
        std::ofstream f(config_path, std::ios::trunc);
        f << "[model]\nhf_path = \"" << hf_dir << "\"\nbackend = \"metal:0\"\n"
          << "[batching]\nkv_page_size = " << kv_page_size << "\ntotal_pages = " << total_pages
          << "\nmax_forward_tokens = " << max_forward_tokens
          << "\nmax_forward_requests = " << max_forward_requests
          << "\n[runtime]\nverbose = false\n";
    }
    const std::string caps = caps_for(config_path);
    std::remove((hf_dir + "/config.json").c_str());
    std::filesystem::remove(hf_dir);
    std::remove(config_path.c_str());
    return caps;
}

}  // namespace

int main() {
    std::printf("[caps honesty]\n");

    // A hybrid (GDN) HF config with LARGE config/model limits — every
    // Phase 1a cap must clamp DOWN to the 4096-token ring.
    {
        const std::string caps = caps_for_scenario(
            "hybrid_large",
            R"({"vocab_size": 248320, "max_position_embeddings": 262144,)"
            R"( "architectures": ["Qwen3NextForCausalLM"],)"
            R"( "layer_types": ["linear_attention", "full_attention"]})",
            /*kv_page_size=*/32, /*total_pages=*/1024, /*max_forward_tokens=*/10240,
            /*max_forward_requests=*/512);
        expect(!caps.empty(), "pie_metal_create with a hybrid config succeeds");
        expect(caps.find("\"rs_cache_required\":true") != std::string::npos,
              "rs_cache_required reports true for a GDN-hybrid checkpoint");
        expect(caps.find("\"max_forward_requests\":4") != std::string::npos,
              "max_forward_requests is truthfully capped to the four actually "
              "allocated/bound paged slots, not the config's 512 (" + caps + ")");
        expect(caps.find("\"rs_cache_slots\":4") != std::string::npos,
              "rs_cache_slots reports 4 (Phase 1b: kPhase1bRsSlots genuinely "
              "addressable GDN state slots, see heap_layout.hpp plan_heap sizing "
              "by max_slots) (" + caps + ")");
        expect(caps.find("\"rs_cache_slot_bytes\":22413312") != std::string::npos,
              "rs_cache_slot_bytes reports the real per-slot GDN state size: 18 "
              "GDN layers * (2*conv_stride(6144*4*4=98304) + recur_stride"
              "(16*128*128*4=1048576)) = 18*1245184=22413312 bytes (" + caps + ")");
        expect(caps.find("\"total_pages\":128") != std::string::npos,
              "total_pages caps to ceil(4096/32)=128, not the config's 1024 (" + caps + ")");
        expect(caps.find("\"max_page_refs\":512") != std::string::npos,
              "max_page_refs covers four CSR segments over the 128-page pool (" + caps + ")");
        expect(caps.find("\"max_forward_tokens\":64") != std::string::npos,
              "max_forward_tokens caps down to the 64 buffered prompt/decode rows, not the "
              "config's 10240 (" + caps + ")");
        expect(caps.find("\"max_model_len\":4096") != std::string::npos,
              "max_model_len caps down to the 4096-token ring, not the HF "
              "config's 262144 (" + caps + ")");
    }

    // A hybrid config whose LIMITS ARE ALREADY SMALLER than the 4096-token
    // ring — capping must never INFLATE a smaller config/model limit.
    {
        const std::string caps = caps_for_scenario(
            "hybrid_small",
            R"({"vocab_size": 248320, "max_position_embeddings": 1024,)"
            R"( "architectures": ["Qwen3NextForCausalLM"],)"
            R"( "layer_types": ["linear_attention", "full_attention"]})",
            /*kv_page_size=*/32, /*total_pages=*/1024, /*max_forward_tokens=*/2048,
            /*max_forward_requests=*/512);
        expect(!caps.empty(), "pie_metal_create with a small-limits hybrid config succeeds");
        expect(caps.find("\"max_forward_tokens\":64") != std::string::npos,
              "max_forward_tokens caps to the buffered prompt/decode-row limit 64 (" + caps + ")");
        expect(caps.find("\"max_model_len\":1024") != std::string::npos,
              "max_model_len keeps the smaller HF config value 1024, not inflated "
              "to 4096 (" + caps + ")");
    }

    // A hybrid config whose kv_page_size does NOT evenly divide 4096 — the
    // page count must be a CEIL division (41), not a floor (40), or the
    // last partial page's tokens would be unreachable.
    {
        const std::string caps = caps_for_scenario(
            "hybrid_ceil",
            R"({"vocab_size": 248320, "max_position_embeddings": 262144,)"
            R"( "architectures": ["Qwen3NextForCausalLM"],)"
            R"( "layer_types": ["linear_attention", "full_attention"]})",
            /*kv_page_size=*/100, /*total_pages=*/1024, /*max_forward_tokens=*/10240,
            /*max_forward_requests=*/512);
        expect(!caps.empty(), "pie_metal_create with a non-dividing kv_page_size succeeds");
        expect(caps.find("\"total_pages\":41") != std::string::npos,
              "total_pages is ceil(4096/100)=41, not the floor 40 (" + caps + ")");
    }

    // A plain (non-hybrid) config — Phase 1a caps stay the generic config
    // values (no forward-needing checkpoint is claimed to be loadable, but
    // there is also nothing to truthfully constrain yet).
    {
        const std::string caps = caps_for_scenario(
            "plain",
            R"({"vocab_size": 32000, "max_position_embeddings": 8192,)"
            R"( "architectures": ["LlamaForCausalLM"]})",
            /*kv_page_size=*/32, /*total_pages=*/1024, /*max_forward_tokens=*/10240,
            /*max_forward_requests=*/512);
        expect(!caps.empty(), "pie_metal_create with a plain config succeeds");
        expect(caps.find("\"rs_cache_required\":false") != std::string::npos,
              "rs_cache_required reports false for a non-hybrid checkpoint");
        expect(caps.find("\"max_forward_requests\":512") != std::string::npos,
              "max_forward_requests keeps the config value when no forward-needing "
              "checkpoint is targeted (" + caps + ")");
        expect(caps.find("\"max_forward_tokens\":10240") != std::string::npos,
              "max_forward_tokens keeps the config value (no forward-needing "
              "checkpoint targeted) (" + caps + ")");
        expect(caps.find("\"max_model_len\":8192") != std::string::npos,
              "max_model_len keeps the HF config value (no forward-needing "
              "checkpoint targeted) (" + caps + ")");
        expect(caps.find("\"total_pages\":1024") != std::string::npos,
              "total_pages keeps the config value (no forward-needing checkpoint "
              "targeted) (" + caps + ")");
    }

    std::printf("\n==== caps_honesty_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
