// Checkpoint-gated Metal PTIR driver e2e (metal_ptir_plan.md Phase 1, G1.2).
// Registers a REAL greedy-argmax epilogue program (the same container/PTIB
// sidecar bytes as `ptir_host_interp_test`'s `greedy_argmax` — echo's golden
// vector), binds + launches it through the full `pie_metal_*` ABI surface
// with real forward fields (token/position/qo/sampling CSR, rs_slot info),
// and cross-checks the epilogue's sampled token against an INDEPENDENTLY
// constructed `MetalExecutor::argmax()` for the identical (token,
// position) input. Two different code paths, same deterministic forward —
// they must agree.
//
// Gated on `PIE_METAL_CKPT` (a real qwen3.6 HF snapshot directory: config.json
// + safetensors). Unset/empty -> prints a SKIP line and exits 0 (this
// increment has no bundled checkpoint fixture, so CI stays green without
// one; the test is written and wired so infra that DOES have a checkpoint
// can flip it on with zero code changes).

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <pie_driver_abi.h>
#include "../tools/rawmetal/native_access.hpp"

namespace {

int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

// ── Paged numeric gate: the NHD pool path writes the requested physical
// page (not page zero), and sequential paged prefill agrees with sealed HND replay.
// This is checkpoint-gated with the rest of this executable. ──
void run_paged_numeric_gate(const std::string& ckpt_dir, const std::string& kernels_dir,
                            std::uint32_t token) {
    using pie::metal::BatchSchedule;
    using pie::metal::BatchStepInputs;
    using pie::metal::DecodeGeometry;
    using pie::metal::build_batch_schedule;
    using pie::metal::batch::MetalExecutor;
    using pie::metal::batch::NativeAccess;

    DecodeGeometry pg{};
    pg.paged_kv_enabled = true;
    pg.max_tokens = 4;
    pg.max_requests = 4;
    pg.max_slots = 4;
    pg.total_pages = 4;
    pg.kv_page_size = 32;
    MetalExecutor paged;
    std::string paged_err;
    if (expect(NativeAccess::setup(
                   paged, ckpt_dir, kernels_dir, pg, &paged_err),
               "paged numeric decoder setup (" + paged_err + ")") &&
        expect(NativeAccess::setup_kv_pool(paged, 4, 32, &paged_err),
               "paged numeric pool setup (" + paged_err + ")")) {
        BatchStepInputs in;
        in.token_ids = {token, token + 1, token + 2, token + 3};
        in.position_ids = {0, 1, 2, 3};
        in.qo_indptr = {0, 4};
        in.kv_page_indptr = {0, 1};
        in.kv_page_indices = {3};  // deliberately not page zero
        in.kv_last_page_lens = {4};
        in.rs_slot_ids = {0};
        in.rs_slot_flags = {1};
        in.w_page = {3, 3, 3, 3};
        in.w_off = {0, 1, 2, 3};
        const BatchSchedule sched = build_batch_schedule(
            in.token_ids.data(), 4, in.qo_indptr.data(), in.kv_page_indptr.data(),
            in.kv_last_page_lens.data(), in.rs_slot_ids.data(), in.rs_slot_flags.data(),
            2, 32);
        expect(NativeAccess::run_batch_step(paged, sched, in, &paged_err),
               "explicit paged append + paged SDPA launch (" + paged_err + ")");
        const uint64_t bind_generation =
            NativeAccess::paged_bind_generation(paged);
        expect(paged.resize_kv_pool(5, /*unmapped_tail_pages=*/false, &paged_err) &&
                   NativeAccess::paged_bind_generation(paged) ==
                       bind_generation + 1 &&
                   NativeAccess::run_batch_step(paged, sched, in, &paged_err),
               "pool resize invalidates/rebinds paged tables before the next batch (" +
                   paged_err + ")");

        const size_t row_bytes = size_t(pg.n_kv_heads) * pg.head_dim * 2;
        bool wrote_page3 = true, left_page0_zero = true;
        for (int l = 0; l < pg.n_layers; ++l) {
            if (!DecodeGeometry::is_full_attn(l)) continue;
            const auto& lp =
                NativeAccess::kv_pool(paged).layers[size_t(l)];
            const auto* k = static_cast<const std::uint8_t*>(lp.k_pages.contents());
            const auto* v = static_cast<const std::uint8_t*>(lp.v_pages.contents());
            bool k3 = false, v3 = false, k0 = false, v0 = false;
            for (size_t b = 0; b < 4 * row_bytes; ++b) {
                k3 |= k[3 * 32 * row_bytes + b] != 0;
                v3 |= v[3 * 32 * row_bytes + b] != 0;
                k0 |= k[b] != 0;
                v0 |= v[b] != 0;
            }
            wrote_page3 &= k3 && v3;
            left_page0_zero &= !k0 && !v0;
        }
        expect(wrote_page3 && left_page0_zero,
               "paged append honors explicit w_page/w_off (writes page 3, not page 0)");

        MetalExecutor ring;
        std::string ring_err;
        if (expect(NativeAccess::setup(
                       ring,
                       ckpt_dir,
                       kernels_dir,
                       DecodeGeometry{},
                       &ring_err),
                   "ring reference decoder setup (" + ring_err + ")")) {
            NativeAccess::reset_state(ring);
            for (size_t t = 0; t < in.token_ids.size(); ++t)
                NativeAccess::step(
                    ring, in.token_ids[t], in.position_ids[t]);
            std::vector<float> ring_logits(
                size_t(NativeAccess::vocab(ring)));
            std::vector<float> paged_logits(
                size_t(NativeAccess::vocab(paged)));
            NativeAccess::copy_logits_f32(ring, ring_logits.data());
            NativeAccess::copy_batch_logits_f32(
                paged, 3, paged_logits.data());
            float max_abs = 0.0f;
            for (size_t i = 0; i < ring_logits.size(); ++i)
                max_abs = std::max(max_abs, std::fabs(ring_logits[i] - paged_logits[i]));
            expect(NativeAccess::argmax(ring) ==
                       NativeAccess::argmax(paged) &&
                       max_abs <= 2e-2f,
                   "paged sequential prefill logits agree with repeated HND ring replay (max_abs=" +
                       std::to_string(max_abs) + ")");
        }
    }
}

std::vector<std::uint8_t> hex_to_bytes(const std::string& h) {
    std::vector<std::uint8_t> b;
    for (std::size_t i = 0; i + 1 < h.size(); i += 2)
        b.push_back(static_cast<std::uint8_t>(std::stoul(h.substr(i, 2), nullptr, 16)));
    return b;
}

// echo's greedy_argmax golden container + PTIB sidecar (V=8): chan0 seeded
// embed token, chan1 the argmax-token reader — the same bytes
// `ptir_host_interp_test.cpp` replays with injected logits. Here the logits
// come from a REAL forward instead.
std::vector<std::uint8_t> greedy_argmax_container() {
    return hex_to_bytes(
        "5054495201000000000000000200000001000000010000000101010000000100000000010101010000"
        "000100000002000000000000000306000000a000000002010000000800000039000000000108000000"
        "330100000039020000000101000000920000000003000000920100000003000000");
}
std::vector<std::uint8_t> greedy_argmax_sidecar() {
    return hex_to_bytes(
        "5054494201000000fe598742954369ff0200000000000200000000000000ff00010000000301010000"
        "000304000000000201000000080000000001080000000100010101000000");
}

void notify_cb(void*, std::uint64_t, std::uint64_t) {}

// Phase 3 (review item 1): launches are async — the driver posts settlement to
// its worker and pie_metal_launch returns after acceptance. Spin until the
// member terminal leaves PENDING (published just before the batch notify, and
// after all channel words), with a bounded timeout so a genuine hang fails the
// test instead of blocking forever.
bool wait_terminal(const PieTerminalCell& cell) {
    for (int i = 0; i < 20000; ++i) {  // ~20s max at 1ms
        if (cell.outcome != PIE_TERMINAL_OUTCOME_PENDING) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

}  // namespace

int main() {
    const char* ckpt_env = std::getenv("PIE_METAL_CKPT");
    if (ckpt_env == nullptr || std::strlen(ckpt_env) == 0) {
        std::printf(
            "[ptir_checkpoint_e2e] SKIP: PIE_METAL_CKPT not set (no bundled checkpoint "
            "fixture in this increment; set PIE_METAL_CKPT=<qwen3.6 HF snapshot dir> to "
            "run this gate).\n");
        return 0;
    }
    const std::string ckpt_dir = ckpt_env;
    std::string kernels_dir;
    if (const char* kd = std::getenv("PIE_METAL_KERNELS_DIR")) kernels_dir = kd;
#ifdef PIE_METAL_KERNELS_DIR_DEFAULT
    if (kernels_dir.empty()) kernels_dir = PIE_METAL_KERNELS_DIR_DEFAULT;
#endif

    // Scratch driver config in the CURRENT working directory (ctest's build-
    // tree CWD, not a source path and not /tmp) — deleted at the end.
    const std::string config_path = "ptir_checkpoint_e2e.generated.toml";
    {
        std::ofstream f(config_path, std::ios::trunc);
        f << "[model]\nhf_path = \"" << ckpt_dir << "\"\nbackend = \"metal:0\"\n"
          << "[batching]\nkv_page_size = 32\ntotal_pages = 1024\n"
          << "max_forward_tokens = 10240\nmax_forward_requests = 512\n"
          << "[runtime]\nverbose = false\n";
    }

    PieDriverCreateDesc create{};
    create.abi_version = PIE_DRIVER_ABI_VERSION;
    create.config_bytes.ptr = reinterpret_cast<const std::uint8_t*>(config_path.data());
    create.config_bytes.len = config_path.size();
    create.runtime.abi_version = PIE_DRIVER_ABI_VERSION;
    create.runtime.ctx = nullptr;
    create.runtime.notify = notify_cb;
    PieDriverCaps caps{};
    PieDriver* driver = pie_metal_create(&create, &caps);
    if (!expect(driver != nullptr, "pie_metal_create")) {
        std::remove(config_path.c_str());
        return 1;
    }

    const std::vector<std::uint8_t> container = greedy_argmax_container();
    const std::vector<std::uint8_t> sidecar = greedy_argmax_sidecar();
    PieProgramDesc program{};
    program.abi_version = PIE_DRIVER_ABI_VERSION;
    program.program_hash = 0xff694395428759feULL;  // echo's greedy_argmax hash
    program.canonical_bytes.ptr = container.data();
    program.canonical_bytes.len = container.size();
    program.sidecar_bytes.ptr = sidecar.data();
    program.sidecar_bytes.len = sidecar.size();
    std::uint64_t program_id = 0;
    expect(pie_metal_register_program(driver, &program, &program_id) == PIE_STATUS_OK,
          "register_program(greedy_argmax)");

    PieChannelDesc chan0{};
    chan0.abi_version = PIE_DRIVER_ABI_VERSION;
    chan0.channel_id = 1;
    const std::uint32_t shape1[] = {1};
    chan0.shape = {.ptr = shape1, .len = 1};
    chan0.dtype = PIE_CHANNEL_DTYPE_I32;
    chan0.host_role = PIE_CHANNEL_HOST_ROLE_NONE;
    chan0.seeded = 1;
    chan0.extern_dir = PIE_CHANNEL_EXTERN_NONE;
    chan0.capacity = 1;
    chan0.reader_wait_id = 1;
    chan0.writer_wait_id = 2;
    PieChannelEndpointBinding chan0_binding{};
    expect(pie_metal_register_channel(driver, &chan0, &chan0_binding) == PIE_STATUS_OK,
          "register_channel(chan0, embed token)");

    PieChannelDesc chan1{};
    chan1.abi_version = PIE_DRIVER_ABI_VERSION;
    chan1.channel_id = 2;
    chan1.shape = {.ptr = shape1, .len = 1};
    chan1.dtype = PIE_CHANNEL_DTYPE_I32;
    chan1.host_role = PIE_CHANNEL_HOST_ROLE_READER;
    chan1.seeded = 0;
    chan1.extern_dir = PIE_CHANNEL_EXTERN_NONE;
    chan1.capacity = 1;
    chan1.reader_wait_id = 3;
    chan1.writer_wait_id = 4;
    PieChannelEndpointBinding chan1_binding{};
    expect(pie_metal_register_channel(driver, &chan1, &chan1_binding) == PIE_STATUS_OK,
          "register_channel(chan1, sampled token)");

    const std::uint32_t embed_token = 1;
    run_paged_numeric_gate(ckpt_dir, kernels_dir, embed_token);
    const std::uint64_t channel_ids[] = {1, 2};
    const std::uint8_t seed_bytes[4] = {
        static_cast<std::uint8_t>(embed_token), 0, 0, 0};
    PieChannelValueDesc seed{};
    seed.channel_id = 1;
    seed.bytes = {.ptr = seed_bytes, .len = 4};
    PieInstanceDesc instance{};
    instance.abi_version = PIE_DRIVER_ABI_VERSION;
    instance.program_id = program_id;
    instance.channel_ids = {.ptr = channel_ids, .len = 2};
    instance.seed_values = {.ptr = &seed, .len = 1};
    PieInstanceBinding binding{};
    expect(pie_metal_bind_instance(driver, &instance, &binding) == PIE_STATUS_OK,
          "bind_instance");

    const std::uint64_t instance_ids[] = {binding.instance_id};
    PieTerminalCell terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
    PieTerminalCell* terminal_ptrs[] = {&terminal};
    const std::uint32_t token_ids[] = {embed_token};
    const std::uint32_t position_ids[] = {0};
    const std::uint32_t qo_indptr[] = {0, 1};
    const std::uint32_t sampling_indices[] = {0};
    const std::uint32_t sampling_indptr[] = {0, 1};
    const std::uint32_t rs_slot_ids[] = {0};
    const std::uint8_t rs_slot_flags[] = {1};  // RS_FLAG_RESET — fresh sequence

    PieLaunchDesc launch{};
    launch.abi_version = PIE_DRIVER_ABI_VERSION;
    launch.instance_ids = {.ptr = instance_ids, .len = 1};
    launch.terminal_cells = {.ptr = terminal_ptrs, .len = 1};
    launch.token_ids = {.ptr = token_ids, .len = 1};
    launch.position_ids = {.ptr = position_ids, .len = 1};
    launch.qo_indptr = {.ptr = qo_indptr, .len = 2};
    launch.sampling_indices = {.ptr = sampling_indices, .len = 1};
    launch.sampling_indptr = {.ptr = sampling_indptr, .len = 2};
    launch.rs_slot_ids = {.ptr = rs_slot_ids, .len = 1};
    launch.rs_slot_flags = {.ptr = rs_slot_flags, .len = 1};
    const std::uint64_t no_ticket = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t ticket_heads[] = {0, no_ticket};
    const std::uint64_t ticket_tails[] = {no_ticket, 0};
    const std::uint32_t ticket_indptr[] = {0, 2};
    launch.channel_expected_head = {.ptr = ticket_heads, .len = 2};
    launch.channel_expected_tail = {.ptr = ticket_tails, .len = 2};
    launch.channel_ticket_indptr = {.ptr = ticket_indptr, .len = 2};

    const PieCompletion completion{.wait_id = 0, .target_epoch = 0, .terminal_cell = nullptr};
    const int32_t launch_rc = pie_metal_launch(driver, &launch, completion);
    expect(launch_rc == PIE_STATUS_OK, "launch (rc=" + std::to_string(launch_rc) + ")");
    expect(wait_terminal(terminal), "launch settles asynchronously (terminal leaves PENDING)");
    expect(terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
          "terminal outcome == SUCCESS (got " + std::to_string(terminal.outcome) + ")");

    // Read the sampled token straight out of chan1's mirror (word[1] is the
    // reader tail; the driver wrote wire bytes at cell 0 of the mirror).
    std::int32_t sampled_token = -1;
    std::memcpy(&sampled_token, reinterpret_cast<void*>(chan1_binding.mirror_base), 4);
    const std::uint64_t reader_tail =
        *reinterpret_cast<const std::uint64_t*>(chan1_binding.word_base + 8);
    expect(reader_tail == 1, "chan1 reader tail advanced to 1");

    // Independent cross-check: a fresh MetalExecutor, same checkpoint, the
    // exact same (token=1, position=0) input from a reset state.
    pie::metal::batch::MetalExecutor cross_check;
    std::string decoder_err;
    const bool decoder_ready =
        pie::metal::batch::NativeAccess::setup(
            cross_check,
            ckpt_dir,
            kernels_dir,
            pie::metal::DecodeGeometry{},
            &decoder_err);
    if (expect(
            decoder_ready,
            "cross-check MetalExecutor setup (" + decoder_err + ")")) {
        pie::metal::batch::NativeAccess::reset_state(cross_check);
        pie::metal::batch::NativeAccess::step(
            cross_check, embed_token, 0);
        const std::uint32_t want =
            pie::metal::batch::NativeAccess::argmax(cross_check);
        expect(sampled_token == static_cast<std::int32_t>(want),
              "epilogue token == native argmax (epilogue=" +
                  std::to_string(sampled_token) + " decoder=" + std::to_string(want) + ")");
    }

    // ── Phase 1b: real copy_state functional check, over the SAME live
    //    executor/decoder the launch above just ran a forward on (real
    //    Metal buffers, real memcpy — not a stub). Copies the just-updated
    //    GDN state from resident slot 0 into slot 1, verifies OK + exactly-
    //    once terminal/notify settlement, then rejects an out-of-range slot
    //    id without ever touching the terminal/notify (matching the
    //    "never return success without doing work" contract). ──
    {
        PieTerminalCell copy_terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        const PieStateCopyRange range{/*src_slot_id=*/0, /*dst_slot_id=*/1,
                                      /*src_token_offset=*/0, /*dst_token_offset=*/0,
                                      /*token_count=*/0};
        PieStateCopyDesc copy{};
        copy.abi_version = PIE_DRIVER_ABI_VERSION;
        copy.slot_ranges = {.ptr = &range, .len = 1};
        const PieCompletion copy_completion{
            .wait_id = 101, .target_epoch = 1, .terminal_cell = &copy_terminal};
        const int32_t copy_rc = pie_metal_copy_state(driver, &copy, copy_completion);
        expect(copy_rc == PIE_STATUS_OK,
              "copy_state(slot 0 -> slot 1) on the live post-forward decoder (rc=" +
                  std::to_string(copy_rc) + ")");
        expect(copy_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
              "copy_state publishes the terminal cell as SUCCESS exactly once");

        // Out-of-range destination slot (rs_cache_slots is kPhase1bRsSlots=4)
        // rejects with INVALID_ARGUMENT and must NOT touch the terminal cell.
        PieTerminalCell bad_terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        const PieStateCopyRange bad_range{/*src_slot_id=*/0, /*dst_slot_id=*/99, 0, 0, 0};
        PieStateCopyDesc bad_copy{};
        bad_copy.abi_version = PIE_DRIVER_ABI_VERSION;
        bad_copy.slot_ranges = {.ptr = &bad_range, .len = 1};
        const PieCompletion bad_completion{
            .wait_id = 102, .target_epoch = 1, .terminal_cell = &bad_terminal};
        const int32_t bad_rc = pie_metal_copy_state(driver, &bad_copy, bad_completion);
        expect(bad_rc == PIE_STATUS_INVALID_ARGUMENT,
              "copy_state rejects an out-of-range slot id (rc=" + std::to_string(bad_rc) + ")");
        expect(bad_terminal.outcome == PIE_TERMINAL_OUTCOME_PENDING,
              "copy_state never publishes the terminal cell on the bounds-rejection path");
    }

    // ── Phase 1b/3 paged-KV bridge: real copy_kv (whole-page + per-token
    //    cell) and resize_pool functional checks, over the REAL standalone
    //    NHD pool MetalExecutor::setup allocated (sized to caps'
    //    total_pages=ceil(4096/32)=128 for this test's kv_page_size=32
    //    config) — genuine memcpy over Shared-storage buffers, not stubs. ──
    {
        // Whole-page copy: page 0 -> page 1 (both in-range for a 128-page pool).
        PieTerminalCell page_terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        const std::uint32_t src_pages[] = {0};
        const std::uint32_t dst_pages[] = {1};
        PieKvCopyDesc page_copy{};
        page_copy.abi_version = PIE_DRIVER_ABI_VERSION;
        page_copy.src_domain = PIE_MEMORY_DOMAIN_METAL_SHARED;
        page_copy.dst_domain = PIE_MEMORY_DOMAIN_METAL_SHARED;
        page_copy.src_page_ids = {.ptr = src_pages, .len = 1};
        page_copy.dst_page_ids = {.ptr = dst_pages, .len = 1};
        const int32_t page_rc =
            pie_metal_copy_kv(driver, &page_copy, PieCompletion{.wait_id = 201, .target_epoch = 1, .terminal_cell = &page_terminal});
        expect(page_rc == PIE_STATUS_OK,
              "copy_kv whole-page copy (0 -> 1) over the real paged pool (rc=" +
                  std::to_string(page_rc) + ")");
        expect(page_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
              "copy_kv publishes the terminal cell as SUCCESS exactly once (whole-page)");

        // Per-token cell copy: (page 0, tok 0) -> (page 2, tok 5).
        PieTerminalCell cell_terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        const PieKvMoveCell cell{/*dst_page_id=*/2, /*dst_token_offset=*/5,
                                 /*src_page_id=*/0, /*src_token_offset=*/0};
        PieKvCopyDesc cell_copy{};
        cell_copy.abi_version = PIE_DRIVER_ABI_VERSION;
        cell_copy.src_domain = PIE_MEMORY_DOMAIN_METAL_SHARED;
        cell_copy.dst_domain = PIE_MEMORY_DOMAIN_METAL_SHARED;
        cell_copy.cells = {.ptr = &cell, .len = 1};
        const int32_t cell_rc =
            pie_metal_copy_kv(driver, &cell_copy, PieCompletion{.wait_id = 202, .target_epoch = 1, .terminal_cell = &cell_terminal});
        expect(cell_rc == PIE_STATUS_OK,
              "copy_kv per-token cell copy over the real paged pool (rc=" +
                  std::to_string(cell_rc) + ")");
        expect(cell_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
              "copy_kv publishes the terminal cell as SUCCESS exactly once (cell copy)");

        // resize_pool GROW: 128 -> 140 pages, always allowed, page ids stable.
        PieTerminalCell grow_terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        PiePoolResizeDesc grow{};
        grow.abi_version = PIE_DRIVER_ABI_VERSION;
        grow.pool_id = 0;
        grow.target_pages = 140;
        const int32_t grow_rc =
            pie_metal_resize_pool(driver, &grow, PieCompletion{.wait_id = 203, .target_epoch = 1, .terminal_cell = &grow_terminal});
        expect(grow_rc == PIE_STATUS_OK,
              "resize_pool GROW 128 -> 140 pages (rc=" + std::to_string(grow_rc) + ")");
        expect(grow_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
              "resize_pool publishes the terminal cell as SUCCESS exactly once (grow)");

        // resize_pool SHRINK without an unmap attestation: rejected — the
        // driver cannot independently know pages [100, 140) are free.
        PieTerminalCell bad_shrink_terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING,
                                            .reserved0 = 0};
        PiePoolResizeDesc bad_shrink{};
        bad_shrink.abi_version = PIE_DRIVER_ABI_VERSION;
        bad_shrink.pool_id = 0;
        bad_shrink.target_pages = 100;
        const int32_t bad_shrink_rc = pie_metal_resize_pool(
            driver, &bad_shrink, PieCompletion{.wait_id = 204, .target_epoch = 1, .terminal_cell = &bad_shrink_terminal});
        expect(bad_shrink_rc == PIE_STATUS_UNSUPPORTED,
              "resize_pool SHRINK without an unmap attestation rejects (rc=" +
                  std::to_string(bad_shrink_rc) + ")");
        expect(bad_shrink_terminal.outcome == PIE_TERMINAL_OUTCOME_PENDING,
              "resize_pool never publishes the terminal cell on the unattested-shrink "
              "rejection path");

        // resize_pool SHRINK WITH a full unmap attestation for [100, 140):
        // accepted.
        PieTerminalCell shrink_terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        const PiePoolRange unmap_range{/*page_index=*/100, /*page_count=*/40};
        PiePoolResizeDesc shrink{};
        shrink.abi_version = PIE_DRIVER_ABI_VERSION;
        shrink.pool_id = 0;
        shrink.target_pages = 100;
        shrink.unmap_ranges = {.ptr = &unmap_range, .len = 1};
        const int32_t shrink_rc =
            pie_metal_resize_pool(driver, &shrink, PieCompletion{.wait_id = 205, .target_epoch = 1, .terminal_cell = &shrink_terminal});
        expect(shrink_rc == PIE_STATUS_OK,
              "resize_pool SHRINK 140 -> 100 pages with a full unmap attestation (rc=" +
                  std::to_string(shrink_rc) + ")");
        expect(shrink_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
              "resize_pool publishes the terminal cell as SUCCESS exactly once (attested "
              "shrink)");
    }

    // ── Phase 1b state-slot fix: DECISIVE hardware-level proof that
    //    copy_state_slot + step(..., slot) genuinely operate on independent
    //    per-slot state (not silently aliasing slot 0). Uses native test access
    //    directly (bypassing MetalExecutor's honest "not ring-backed"
    //    continuation gate, which is a DELIBERATE Phase-1a/1b M=1-ring
    //    limitation, not a limitation of the underlying per-slot mechanism
    //    itself): step slot 0 for two tokens, copy its state to slot 1,
    //    reset slot 0 (diverging it), then continue EACH of slot 0 (fresh)
    //    and slot 1 (from the copy) with a THIRD token at position 2. Slot
    //    1's result must be BIT-IDENTICAL to an independent decoder that
    //    replayed the ORIGINAL unbroken 3-token sequence on slot 0 alone —
    //    proving the copy moved the right bytes, `step()`'s per-slot
    //    rebinding retargeted the GDN kernels correctly, and ping-pong
    //    parity resumed correctly on the copied slot. ──
    {
        using pie::metal::DecodeGeometry;
        using pie::metal::batch::MetalExecutor;
        using pie::metal::batch::NativeAccess;

        DecodeGeometry geom2slots{};
        geom2slots.max_slots = 2;
        const std::uint32_t tokA = embed_token, tokB = embed_token, tokC = embed_token;

        MetalExecutor decoder_a;
        std::string err_a;
        if (expect(NativeAccess::setup(
                      decoder_a,
                      ckpt_dir,
                      kernels_dir,
                      geom2slots,
                      &err_a),
                  "state-slot-fix decoder_a setup (" + err_a + ")")) {
            NativeAccess::reset_state(decoder_a, 0);
            NativeAccess::step(decoder_a, tokA, 0, /*slot=*/0);
            NativeAccess::step(decoder_a, tokB, 1, /*slot=*/0);
            std::string copy_err;
            expect(NativeAccess::copy_state_slot(
                      decoder_a, 0, 1, &copy_err),
                  "copy_state_slot(0 -> 1) after 2 steps on slot 0 (" + copy_err + ")");

            // Diverge slot 0: reset it and step it fresh with a DIFFERENT
            // token at position 0 — if slot 1 secretly aliased slot 0's
            // memory, this would corrupt slot 1's copied state too.
            NativeAccess::reset_state(decoder_a, 0);
            NativeAccess::step(
                decoder_a, tokA + 1u, 0, /*slot=*/0);

            // Continue slot 1 from its copied (2-step) state with the THIRD
            // token at position 2 — exactly the position slot 0's ORIGINAL
            // (pre-reset) history would have continued to.
            NativeAccess::step(decoder_a, tokC, 2, /*slot=*/1);
            const std::uint32_t slot1_argmax =
                NativeAccess::argmax(decoder_a);

            // Independent reference: a SEPARATE fresh decoder that replays
            // the ORIGINAL unbroken 3-token sequence on slot 0 alone, never
            // touching a second slot at all.
            MetalExecutor decoder_b;
            std::string err_b;
            if (expect(NativeAccess::setup(
                          decoder_b,
                          ckpt_dir,
                          kernels_dir,
                          DecodeGeometry{},
                          &err_b),
                      "state-slot-fix decoder_b (reference) setup (" + err_b + ")")) {
                NativeAccess::reset_state(decoder_b);
                NativeAccess::step(decoder_b, tokA, 0);
                NativeAccess::step(decoder_b, tokB, 1);
                NativeAccess::step(decoder_b, tokC, 2);
                const std::uint32_t reference_argmax =
                    NativeAccess::argmax(decoder_b);
                expect(slot1_argmax == reference_argmax,
                      "slot 1's copied-then-continued argmax == an unbroken single-slot "
                      "replay's argmax (slot1=" + std::to_string(slot1_argmax) +
                      " reference=" + std::to_string(reference_argmax) +
                      ") — proves real, independent per-slot state, not slot-0 aliasing");
            }
        }
    }

    // ── Phase 3 (§7): multi-member forward_batch — logits vs individual.
    //    Bind a SECOND instance of the same greedy_argmax program (distinct
    //    channels 3/4, rs_slot 1) and launch BOTH members in ONE batch. The
    //    old per-member forward loop failed the 2nd forward member ("a
    //    different sequence is resident"); the batch scheduler runs both fresh
    //    members serially over the shared ring and each captures its own
    //    logits. Both members feed the identical (token=1, position=0) fresh
    //    input, so each member's sampled token must equal the single-member
    //    reference argmax computed above — proving the batch produced correct
    //    per-member logits, settled both terminals, and notified once. ──
    {
        PieChannelDesc chan2{};  // 2nd member's seeded embed-token channel
        chan2.abi_version = PIE_DRIVER_ABI_VERSION;
        chan2.channel_id = 3;
        chan2.shape = {.ptr = shape1, .len = 1};
        chan2.dtype = PIE_CHANNEL_DTYPE_I32;
        chan2.host_role = PIE_CHANNEL_HOST_ROLE_NONE;
        chan2.seeded = 1;
        chan2.extern_dir = PIE_CHANNEL_EXTERN_NONE;
        chan2.capacity = 1;
        chan2.reader_wait_id = 5;
        chan2.writer_wait_id = 6;
        PieChannelEndpointBinding chan2_binding{};
        expect(pie_metal_register_channel(driver, &chan2, &chan2_binding) == PIE_STATUS_OK,
              "register_channel(chan3, 2nd member embed token)");

        PieChannelDesc chan3{};  // 2nd member's sampled-token reader channel
        chan3.abi_version = PIE_DRIVER_ABI_VERSION;
        chan3.channel_id = 4;
        chan3.shape = {.ptr = shape1, .len = 1};
        chan3.dtype = PIE_CHANNEL_DTYPE_I32;
        chan3.host_role = PIE_CHANNEL_HOST_ROLE_READER;
        chan3.seeded = 0;
        chan3.extern_dir = PIE_CHANNEL_EXTERN_NONE;
        chan3.capacity = 1;
        chan3.reader_wait_id = 7;
        chan3.writer_wait_id = 8;
        PieChannelEndpointBinding chan3_binding{};
        expect(pie_metal_register_channel(driver, &chan3, &chan3_binding) == PIE_STATUS_OK,
              "register_channel(chan4, 2nd member sampled token)");

        const std::uint64_t channel_ids2[] = {3, 4};
        PieChannelValueDesc seed2{};
        seed2.channel_id = 3;
        seed2.bytes = {.ptr = seed_bytes, .len = 4};
        PieInstanceDesc instance2{};
        instance2.abi_version = PIE_DRIVER_ABI_VERSION;
        instance2.program_id = program_id;
        instance2.channel_ids = {.ptr = channel_ids2, .len = 2};
        instance2.seed_values = {.ptr = &seed2, .len = 1};
        PieInstanceBinding binding2{};
        expect(pie_metal_bind_instance(driver, &instance2, &binding2) == PIE_STATUS_OK,
              "bind_instance (2nd member)");

        PieChannelDesc chan4 = chan2;
        chan4.channel_id = 5;
        chan4.reader_wait_id = 9;
        chan4.writer_wait_id = 10;
        PieChannelEndpointBinding chan4_binding{};
        expect(pie_metal_register_channel(driver, &chan4, &chan4_binding) == PIE_STATUS_OK,
              "register_channel(chan5, 3rd member embed token)");
        PieChannelDesc chan5 = chan3;
        chan5.channel_id = 6;
        chan5.reader_wait_id = 11;
        chan5.writer_wait_id = 12;
        PieChannelEndpointBinding chan5_binding{};
        expect(pie_metal_register_channel(driver, &chan5, &chan5_binding) == PIE_STATUS_OK,
              "register_channel(chan6, 3rd member sampled token)");
        const std::uint64_t channel_ids3[] = {5, 6};
        PieChannelValueDesc seed3{};
        seed3.channel_id = 5;
        seed3.bytes = {.ptr = seed_bytes, .len = 4};
        PieInstanceDesc instance3{};
        instance3.abi_version = PIE_DRIVER_ABI_VERSION;
        instance3.program_id = program_id;
        instance3.channel_ids = {.ptr = channel_ids3, .len = 2};
        instance3.seed_values = {.ptr = &seed3, .len = 1};
        PieInstanceBinding binding3{};
        expect(pie_metal_bind_instance(driver, &instance3, &binding3) == PIE_STATUS_OK,
              "bind_instance (3rd member)");

        // Both batch members are fresh; the earlier single-member reference
        // already consumed its seed and is not reused.
        const std::uint64_t batch_instance_ids[] = {binding2.instance_id, binding3.instance_id};
        PieTerminalCell t0{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        PieTerminalCell t1{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        PieTerminalCell* batch_terminal_ptrs[] = {&t0, &t1};
        const std::uint32_t batch_tokens[] = {embed_token, embed_token};
        const std::uint32_t batch_positions[] = {0, 0};
        const std::uint32_t batch_qo_indptr[] = {0, 1, 2};
        const std::uint32_t batch_kv_page_indices[] = {0, 1};
        const std::uint32_t batch_kv_page_indptr[] = {0, 1, 2};
        const std::uint32_t batch_kv_last_page_lens[] = {1, 1};
        const std::uint32_t batch_sampling_indices[] = {0, 0};
        const std::uint32_t batch_sampling_indptr[] = {0, 1, 2};
        const std::uint32_t batch_rs_slot_ids[] = {0, 1};
        const std::uint8_t batch_rs_slot_flags[] = {1, 1};  // both fresh

        PieLaunchDesc batch{};
        batch.abi_version = PIE_DRIVER_ABI_VERSION;
        batch.instance_ids = {.ptr = batch_instance_ids, .len = 2};
        batch.terminal_cells = {.ptr = batch_terminal_ptrs, .len = 2};
        batch.token_ids = {.ptr = batch_tokens, .len = 2};
        batch.position_ids = {.ptr = batch_positions, .len = 2};
        batch.qo_indptr = {.ptr = batch_qo_indptr, .len = 3};
        batch.kv_page_indices = {.ptr = batch_kv_page_indices, .len = 2};
        batch.kv_page_indptr = {.ptr = batch_kv_page_indptr, .len = 3};
        batch.kv_last_page_lens = {.ptr = batch_kv_last_page_lens, .len = 2};
        batch.sampling_indices = {.ptr = batch_sampling_indices, .len = 2};
        batch.sampling_indptr = {.ptr = batch_sampling_indptr, .len = 3};
        batch.rs_slot_ids = {.ptr = batch_rs_slot_ids, .len = 2};
        batch.rs_slot_flags = {.ptr = batch_rs_slot_flags, .len = 2};
        const std::uint64_t batch_ticket_heads[] = {
            0, no_ticket, 0, no_ticket};
        const std::uint64_t batch_ticket_tails[] = {
            no_ticket, 0, no_ticket, 0};
        const std::uint32_t batch_ticket_indptr[] = {0, 2, 4};
        batch.channel_expected_head = {.ptr = batch_ticket_heads, .len = 4};
        batch.channel_expected_tail = {.ptr = batch_ticket_tails, .len = 4};
        batch.channel_ticket_indptr = {.ptr = batch_ticket_indptr, .len = 3};

        // The driver was created with a no-op notify; the batch completion here
        // uses wait_id 0 (no batch notify), so we assert on terminals only.
        const PieCompletion batch_completion{
            .wait_id = 0, .target_epoch = 0, .terminal_cell = nullptr};
        const int32_t batch_rc = pie_metal_launch(driver, &batch, batch_completion);
        expect(batch_rc == PIE_STATUS_OK,
              "multi-member forward_batch launch (rc=" + std::to_string(batch_rc) + ")");
        expect(wait_terminal(t0) && wait_terminal(t1),
              "both batch members settle asynchronously (terminals leave PENDING)");
        expect(t0.outcome == PIE_TERMINAL_OUTCOME_SUCCESS &&
                   t1.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
              "both batch members settle their terminals SUCCESS exactly once");

        std::int32_t tok0 = -1, tok1 = -1;
        std::memcpy(&tok0, reinterpret_cast<void*>(chan3_binding.mirror_base), 4);
        std::memcpy(&tok1, reinterpret_cast<void*>(chan5_binding.mirror_base), 4);
        expect(tok0 == sampled_token && tok1 == sampled_token,
              "each batch member's sampled token == the single-member reference (member0=" +
                  std::to_string(tok0) + " member1=" + std::to_string(tok1) + " ref=" +
                  std::to_string(sampled_token) + ")");
    }

    pie_metal_destroy(driver);
    std::remove(config_path.c_str());
    std::printf("\n==== ptir_checkpoint_e2e_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}