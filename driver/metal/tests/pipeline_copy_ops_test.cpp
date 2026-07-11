// Phase 1b/3 paged-KV bridge driver-level ABI gate for the three control
// operations (metal_ptir_plan.md Phase 1b: "these ABI control operations
// must settle terminal/completion and notify exactly once on success/
// failure ... never return success without doing work"). No checkpoint/
// Apple/Metal dependency needed for these assertions: every path exercised
// here fails BEFORE any real Metal work would happen (either because the
// checkpoint has no GDN/linear-attention layers at all, because the
// requested cross-domain copy has no swap-pool backing in this build, or
// because the fixture checkpoint dir is deliberately incomplete — no
// safetensors weights — so `MetalExecutor::setup` throws before touching
// a GPU), so this binary always builds and runs.
//
// Proves, for `pie_metal_copy_kv` / `pie_metal_copy_state` / `pie_metal_
// resize_pool`:
//   * copy_kv over a cross-domain request (anything other than same-domain
//     PIE_MEMORY_DOMAIN_METAL_SHARED) is UNSUPPORTED — there is no
//     host-pinned swap pool in this build — never publishes the terminal
//     cell, never notifies.
//   * resize_pool on a non-hybrid checkpoint (no paged KV pool at all,
//     since MetalExecutor::setup refuses non-hybrid architectures entirely)
//     is UNSUPPORTED with the same no-terminal/no-notify contract.
//   * copy_state on a NON-hybrid (no GDN layers) checkpoint is UNSUPPORTED
//     with no terminal publish / no notify (there is no recurrent state to
//     copy).
//   * copy_state/resize_pool on a hybrid checkpoint whose directory has no
//     real weights fail driver-error-safely (the underlying
//     SafetensorsView throws; the extern "C" boundary's catch(...)
//     converts it, it does not crash) — still no terminal publish / no
//     notify.
//   * the shared ABI completion validator (`validate_completion`) is
//     enforced BEFORE any of the above: an invalid completion (wait_id==0)
//     is rejected synchronously, before the driver method (or even a null-
//     driver check) ever runs.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <pie_driver_abi.h>

namespace {

int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

struct NotifyState {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> log;
};

void notify_cb(void* ctx, std::uint64_t wait_id, std::uint64_t epoch) {
    static_cast<NotifyState*>(ctx)->log.emplace_back(wait_id, epoch);
}

// Writes a scratch HF snapshot dir (config.json only, deliberately NO
// safetensors weights) + a scratch driver config.toml pointing at it, and
// returns a live `PieDriver*` (never null: `initialize()` only parses
// config.json, it never opens the checkpoint). Caller must
// `pie_metal_destroy` it.
PieDriver* create_driver(const std::string& tag, const std::string& config_json_body,
                         NotifyState& notify_state) {
    const std::string hf_dir = "copy_ops_" + tag + "_hf";
    std::filesystem::create_directories(hf_dir);
    {
        std::ofstream f(hf_dir + "/config.json", std::ios::trunc);
        f << config_json_body;
    }
    const std::string config_path = "copy_ops_" + tag + ".generated.toml";
    {
        std::ofstream f(config_path, std::ios::trunc);
        f << "[model]\nhf_path = \"" << hf_dir << "\"\nbackend = \"metal:0\"\n"
          << "[batching]\nkv_page_size = 32\ntotal_pages = 128\n"
          << "max_forward_tokens = 4096\nmax_forward_requests = 1\n"
          << "[runtime]\nverbose = false\n";
    }
    PieDriverCreateDesc create{};
    create.abi_version = PIE_DRIVER_ABI_VERSION;
    create.config_bytes.ptr =
        reinterpret_cast<const std::uint8_t*>(config_path.data());
    create.config_bytes.len = config_path.size();
    create.runtime.abi_version = PIE_DRIVER_ABI_VERSION;
    create.runtime.ctx = &notify_state;
    create.runtime.notify = notify_cb;
    PieDriverCaps caps{};
    PieDriver* driver = pie_metal_create(&create, &caps);
    std::remove((hf_dir + "/config.json").c_str());
    std::filesystem::remove(hf_dir);
    std::remove(config_path.c_str());
    return driver;
}

constexpr char kNonHybridConfig[] =
    R"({"vocab_size": 32000, "max_position_embeddings": 8192,)"
    R"( "architectures": ["LlamaForCausalLM"]})";
constexpr char kHybridConfig[] =
    R"({"vocab_size": 248320, "max_position_embeddings": 262144,)"
    R"( "architectures": ["Qwen3NextForCausalLM"],)"
    R"( "layer_types": ["linear_attention", "full_attention"]})";

PieCompletion make_completion(std::uint64_t wait_id, std::uint64_t epoch,
                              PieTerminalCell* cell) {
    PieCompletion c{};
    c.wait_id = wait_id;
    c.target_epoch = epoch;
    c.terminal_cell = cell;
    return c;
}

}  // namespace

int main() {
    std::printf("[copy_kv / copy_state / resize_pool ABI gate]\n");

    // ── copy_kv: a cross-domain request (no host-pinned swap pool exists
    //    in this build) is UNSUPPORTED regardless of checkpoint
    //    architecture — rejected before ever touching the executor/pool. ──
    {
        NotifyState notify_state;
        PieDriver* driver = create_driver("kv_hybrid", kHybridConfig, notify_state);
        expect(driver != nullptr, "pie_metal_create (hybrid, for copy_kv) succeeds");
        PieTerminalCell cell{};
        cell.outcome = PIE_TERMINAL_OUTCOME_PENDING;
        PieKvCopyDesc copy{};
        copy.abi_version = PIE_DRIVER_ABI_VERSION;
        copy.src_domain = PIE_MEMORY_DOMAIN_CUDA_DEVICE;  // cross-domain: no swap pool exists
        copy.dst_domain = PIE_MEMORY_DOMAIN_CUDA_DEVICE;
        const std::int32_t rc =
            pie_metal_copy_kv(driver, &copy, make_completion(11, 1, &cell));
        expect(rc == PIE_STATUS_UNSUPPORTED,
              "copy_kv rejects a cross-domain request (no host-pinned swap pool) (rc=" +
              std::to_string(rc) + ")");
        expect(cell.outcome == PIE_TERMINAL_OUTCOME_PENDING,
              "copy_kv never publishes the terminal cell on the UNSUPPORTED path");
        expect(notify_state.log.empty(), "copy_kv never notifies on the UNSUPPORTED path");
        pie_metal_destroy(driver);
    }

    // ── resize_pool on a NON-hybrid checkpoint: UNSUPPORTED (MetalExecutor
    //    ::setup refuses non-hybrid architectures entirely, so there is no
    //    paged KV pool to resize — rejected before ever touching the
    //    executor). ──
    {
        NotifyState notify_state;
        PieDriver* driver = create_driver("resize_plain", kNonHybridConfig, notify_state);
        expect(driver != nullptr, "pie_metal_create (non-hybrid, for resize_pool) succeeds");
        PieTerminalCell cell{};
        cell.outcome = PIE_TERMINAL_OUTCOME_PENDING;
        PiePoolResizeDesc resize{};
        resize.abi_version = PIE_DRIVER_ABI_VERSION;
        resize.pool_id = 0;
        resize.target_pages = 256;
        const std::int32_t rc =
            pie_metal_resize_pool(driver, &resize, make_completion(12, 1, &cell));
        expect(rc == PIE_STATUS_UNSUPPORTED,
              "resize_pool on a non-hybrid checkpoint returns PIE_STATUS_UNSUPPORTED (rc=" +
              std::to_string(rc) + ")");
        expect(cell.outcome == PIE_TERMINAL_OUTCOME_PENDING,
              "resize_pool never publishes the terminal cell on the non-hybrid UNSUPPORTED path");
        expect(notify_state.log.empty(),
              "resize_pool never notifies on the non-hybrid UNSUPPORTED path");
        pie_metal_destroy(driver);
    }

    // ── resize_pool on a hybrid checkpoint whose directory has no real
    //    weights: the lazy executor setup throws inside SafetensorsView;
    //    the extern "C" boundary's catch(...) must convert this to
    //    PIE_STATUS_DRIVER_ERROR safely (no crash), matching copy_state's
    //    identical hybrid-no-checkpoint precedent. ──
    {
        NotifyState notify_state;
        PieDriver* driver = create_driver("resize_hybrid_nockpt", kHybridConfig, notify_state);
        expect(driver != nullptr, "pie_metal_create (hybrid, for resize_pool) succeeds");
        PieTerminalCell cell{};
        cell.outcome = PIE_TERMINAL_OUTCOME_PENDING;
        PiePoolResizeDesc resize{};
        resize.abi_version = PIE_DRIVER_ABI_VERSION;
        resize.pool_id = 0;
        resize.target_pages = 256;
        const std::int32_t rc =
            pie_metal_resize_pool(driver, &resize, make_completion(13, 1, &cell));
        expect(rc == PIE_STATUS_DRIVER_ERROR || rc == PIE_STATUS_UNSUPPORTED,
              "resize_pool on a hybrid checkpoint with no real weights fails safely, never "
              "crashes (rc=" + std::to_string(rc) + ")");
        expect(cell.outcome == PIE_TERMINAL_OUTCOME_PENDING,
              "resize_pool never publishes the terminal cell on the setup-failure path");
        expect(notify_state.log.empty(),
              "resize_pool never notifies on the setup-failure path");
        pie_metal_destroy(driver);
    }

    // ── copy_state on a NON-hybrid checkpoint: UNSUPPORTED (no GDN state to
    //    copy at all — rejected before any executor/checkpoint touch). ──
    {
        NotifyState notify_state;
        PieDriver* driver = create_driver("state_plain", kNonHybridConfig, notify_state);
        expect(driver != nullptr, "pie_metal_create (non-hybrid, for copy_state) succeeds");
        PieTerminalCell cell{};
        cell.outcome = PIE_TERMINAL_OUTCOME_PENDING;
        PieStateCopyDesc copy{};
        copy.abi_version = PIE_DRIVER_ABI_VERSION;
        const PieStateCopyRange range{0, 1, 0, 0, 0};
        copy.slot_ranges.ptr = &range;
        copy.slot_ranges.len = 1;
        const std::int32_t rc =
            pie_metal_copy_state(driver, &copy, make_completion(13, 1, &cell));
        expect(rc == PIE_STATUS_UNSUPPORTED,
              "copy_state on a non-hybrid checkpoint returns PIE_STATUS_UNSUPPORTED (rc=" +
              std::to_string(rc) + ")");
        expect(cell.outcome == PIE_TERMINAL_OUTCOME_PENDING,
              "copy_state never publishes the terminal cell on the non-hybrid UNSUPPORTED path");
        expect(notify_state.log.empty(),
              "copy_state never notifies on the non-hybrid UNSUPPORTED path");
        pie_metal_destroy(driver);
    }

    // ── copy_state on a hybrid checkpoint whose directory has no real
    //    weights: the lazy executor setup throws inside SafetensorsView;
    //    the extern "C" boundary's catch(...) must convert this to
    //    PIE_STATUS_DRIVER_ERROR safely (no crash), still without ever
    //    publishing the terminal cell or notifying. ──
    {
        NotifyState notify_state;
        PieDriver* driver = create_driver("state_hybrid_nockpt", kHybridConfig, notify_state);
        expect(driver != nullptr, "pie_metal_create (hybrid, for copy_state) succeeds");
        PieTerminalCell cell{};
        cell.outcome = PIE_TERMINAL_OUTCOME_PENDING;
        PieStateCopyDesc copy{};
        copy.abi_version = PIE_DRIVER_ABI_VERSION;
        const PieStateCopyRange range{0, 1, 0, 0, 0};
        copy.slot_ranges.ptr = &range;
        copy.slot_ranges.len = 1;
        const std::int32_t rc =
            pie_metal_copy_state(driver, &copy, make_completion(14, 1, &cell));
        expect(rc == PIE_STATUS_DRIVER_ERROR || rc == PIE_STATUS_UNSUPPORTED,
              "copy_state on a hybrid checkpoint with no real weights fails safely, "
              "never crashes (rc=" + std::to_string(rc) + ")");
        expect(cell.outcome == PIE_TERMINAL_OUTCOME_PENDING,
              "copy_state never publishes the terminal cell on the setup-failure path");
        expect(notify_state.log.empty(),
              "copy_state never notifies on the setup-failure path");
        pie_metal_destroy(driver);
    }

    // ── Shared ABI validator: an invalid completion (wait_id==0) is
    //    rejected before the driver method (or even a null-driver check)
    //    ever runs — proven by passing driver=nullptr too. ──
    {
        PieTerminalCell cell{};
        cell.outcome = PIE_TERMINAL_OUTCOME_PENDING;
        PieStateCopyDesc copy{};
        copy.abi_version = PIE_DRIVER_ABI_VERSION;
        copy.slot_ranges.ptr = nullptr;
        copy.slot_ranges.len = 0;
        PieCompletion bad = make_completion(/*wait_id=*/0, /*epoch=*/0, &cell);
        const std::int32_t rc = pie_metal_copy_state(nullptr, &copy, bad);
        expect(rc == PIE_STATUS_INVALID_ARGUMENT,
              "an invalid completion (wait_id==0) is rejected before any driver dispatch, "
              "even with driver==nullptr (rc=" + std::to_string(rc) + ")");
    }

    std::printf("\n==== ptir_copy_ops_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
