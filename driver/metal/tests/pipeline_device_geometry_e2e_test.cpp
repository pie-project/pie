// Device-geometry (Phase 2, C3) driver-level poison/settlement gate
// (metal_ptir_plan.md §9 G2.3-style: "not-ready descriptor channel fails the
// fire and poisons per D4"). Drives the FULL `pie_metal_*` ABI surface (no
// checkpoint/Apple/Metal dependency needed: the not-ready descriptor channel
// fails `resolve_fire_geometry` BEFORE `MetalExecutor` is ever touched), so
// this binary always builds and runs.
//
// Registers two programs in ONE launch batch:
//   - A: device-geometry (a channel-bound `Positions` port + an
//     Intrinsic(Logits) epilogue value, so it IS forward-needing) whose
//     positions channel is NEVER produced — the resolver must fail it
//     (W1.6, no dummy-run), poisoning ONLY this member.
//   - B: an ordinary channel-plane passthrough (C1, take -> put) with no
//     forward dependency at all.
// Expects: A's terminal outcome is FAILED and its attached channel is
// poisoned + its wait id notified; B's terminal outcome is SUCCESS and its
// own put/notify settle exactly as if A were not in the batch — proving
// per-member failure isolation (D4) and that settlement/notify ordering for
// the REST of the batch is unaffected by one member's descriptor-resolve
// failure.

#include <cstdint>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <pie_driver_abi.h>
#include "pie_native/ptir/container.hpp"

namespace {

int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

// Phase 3 (review item 1): launches are async — the driver publishes
// terminals/words/notifies from its worker thread. Thread-safe sink + a
// wait(wait_id, epoch) for the batch notify (published last).
struct NotifyState {
    mutable std::mutex mu;
    std::condition_variable cv;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> log;
    void record(std::uint64_t wait_id, std::uint64_t epoch) {
        {
            std::lock_guard<std::mutex> lock(mu);
            log.emplace_back(wait_id, epoch);
        }
        cv.notify_all();
    }
    bool contains(std::uint64_t wait_id, std::uint64_t epoch) const {
        std::lock_guard<std::mutex> lock(mu);
        return contains_locked(wait_id, epoch);
    }
    std::size_t count() const {
        std::lock_guard<std::mutex> lock(mu);
        return log.size();
    }
    void wait(std::uint64_t wait_id, std::uint64_t epoch) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return contains_locked(wait_id, epoch); });
    }

  private:
    bool contains_locked(std::uint64_t wait_id, std::uint64_t epoch) const {
        for (const auto& [w, e] : log)
            if (w == wait_id && e == epoch) return true;
        return false;
    }
};

void notify_cb(void* ctx, std::uint64_t wait_id, std::uint64_t epoch) {
    static_cast<NotifyState*>(ctx)->record(wait_id, epoch);
}

void push_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value));
    out.push_back(static_cast<std::uint8_t>(value >> 8));
}
void push_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

// Program A: 1 channel (positions descriptor target, never produced), 1
// channel-bound Positions port (=> device geometry), 1 epilogue op reading
// Intrinsic(Logits) (=> needs_forward()) whose result is never consumed —
// the resolve fails long before `step()` would run, so this never matters.
// The channel is host_role=NONE (a realistic device-geometry descriptor
// channel is purely device/channel-plane-produced — a host WRITER role
// would make the launch's writer-availability check reject the whole batch
// upfront, since positions is a CONSUMING port; that is a different, correct
// rejection path, not the one this test exercises).
std::vector<std::uint8_t> device_geometry_container() {
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    push_u16(out, PTIR_VERSION);
    push_u16(out, 0);
    push_u32(out, 0);  // names
    push_u32(out, 1);  // channels
    push_u32(out, 1);  // ports
    push_u32(out, 1);  // stages
    // channel 0: U32 vector[1], capacity 1, host_role=NONE, not seeded.
    out.push_back(PTIR_DT_U32);
    out.push_back(1);  // rank
    push_u32(out, 1);
    push_u32(out, 1);  // capacity
    out.push_back(PIE_CHANNEL_HOST_ROLE_NONE);
    out.push_back(0);  // seeded
    // port 0: Positions, channel-bound to chan 0.
    out.push_back(2);  // kPortPositions
    out.push_back(0);  // src == channel
    push_u32(out, 0);  // chan
    // stage 0 (epilogue): one INTRINSIC_VAL(Logits) op, result unused.
    out.push_back(PTIR_STAGE_EPILOGUE);
    push_u32(out, 1);  // ops
    out.push_back(PTIR_OP_INTRINSIC_VAL);
    push_u16(out, 0);            // intr == Logits (PTIR_INTR_LOGITS)
    out.push_back(PTIR_DT_F32);  // dtype
    out.push_back(1);            // shape rank
    push_u32(out, 1);            // shape dims[0]
    return out;
}

std::vector<std::uint8_t> device_geometry_sidecar(const std::vector<std::uint8_t>& container) {
    const std::uint64_t hash =
        pie_native::ptir::container::fnv1a64(container.data(), container.size());
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    push_u16(out, PTIB_VERSION);
    push_u16(out, 0);
    for (int b = 0; b < 8; ++b) out.push_back(static_cast<std::uint8_t>(hash >> (b * 8)));
    push_u32(out, 1);            // channel classes
    out.push_back(0);            // chan0 class = FullRing
    push_u32(out, 0);            // readiness entries (none needed — no stages take/put chan0)
    push_u32(out, 1);            // stages
    out.push_back(PTIR_STAGE_EPILOGUE);
    push_u32(out, 1);            // 1 SSA value in this stage (the intrinsic)
    out.push_back(PTIR_DT_F32);
    out.push_back(1);
    push_u32(out, 1);
    return out;
}

// Program B: a plain channel-plane passthrough — chan0 (seeded Writer) ->
// take -> chan1 (Reader) put. No ports, no forward dependency at all.
std::vector<std::uint8_t> passthrough_container() {
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    push_u16(out, PTIR_VERSION);
    push_u16(out, 0);
    push_u32(out, 0);  // names
    push_u32(out, 2);  // channels
    push_u32(out, 0);  // ports
    push_u32(out, 1);  // stages
    for (int c = 0; c < 2; ++c) {
        out.push_back(PTIR_DT_U32);
        out.push_back(1);  // rank
        push_u32(out, 1);
        push_u32(out, 1);  // capacity
        out.push_back(c == 0 ? PIE_CHANNEL_HOST_ROLE_WRITER : PIE_CHANNEL_HOST_ROLE_READER);
        out.push_back(c == 0 ? 1 : 0);  // chan0 seeded
    }
    out.push_back(PTIR_STAGE_EPILOGUE);
    push_u32(out, 2);  // ops
    out.push_back(PTIR_OP_CHAN_TAKE);
    push_u32(out, 0);
    out.push_back(PTIR_OP_CHAN_PUT);
    push_u32(out, 1);
    push_u32(out, 0);  // put the take's result (value id 0)
    return out;
}

std::vector<std::uint8_t> passthrough_sidecar(const std::vector<std::uint8_t>& container) {
    const std::uint64_t hash =
        pie_native::ptir::container::fnv1a64(container.data(), container.size());
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    push_u16(out, PTIB_VERSION);
    push_u16(out, 0);
    for (int b = 0; b < 8; ++b) out.push_back(static_cast<std::uint8_t>(hash >> (b * 8)));
    push_u32(out, 2);  // channel classes
    out.push_back(0);
    out.push_back(0);
    push_u32(out, 2);  // readiness entries
    push_u32(out, 0);  // chan0
    out.push_back(PTIR_STAGE_EPILOGUE);
    out.push_back(0);  // NeedsFull (take)
    push_u32(out, 1);  // chan1
    out.push_back(PTIR_STAGE_EPILOGUE);
    out.push_back(1);  // NeedsEmpty (put)
    push_u32(out, 1);  // stages
    out.push_back(PTIR_STAGE_EPILOGUE);
    push_u32(out, 1);  // 1 SSA value (the take's result)
    out.push_back(PTIR_DT_U32);
    out.push_back(1);
    push_u32(out, 1);
    return out;
}

}  // namespace

int main() {
    NotifyState notify_state;

    // A self-contained scratch config in the CURRENT working directory
    // (ctest's build-tree CWD, not a source path and not /tmp) — no real
    // checkpoint is needed (member A's resolve fails before any forward is
    // attempted; member B needs no forward at all). Deleted at the end.
    const std::string config_path = "ptir_device_geometry_e2e.generated.toml";
    {
        std::ofstream f(config_path, std::ios::trunc);
        f << "[model]\nhf_path = \"\"\nbackend = \"metal:0\"\n"
          << "[batching]\nkv_page_size = 32\ntotal_pages = 1024\n"
          << "max_forward_tokens = 10240\nmax_forward_requests = 512\n"
          << "[runtime]\nverbose = false\n";
    }

    PieDriverCreateDesc create{};
    create.abi_version = PIE_DRIVER_ABI_VERSION;
    create.config_bytes.ptr = reinterpret_cast<const std::uint8_t*>(config_path.data());
    create.config_bytes.len = config_path.size();
    create.runtime.abi_version = PIE_DRIVER_ABI_VERSION;
    create.runtime.ctx = &notify_state;
    create.runtime.notify = notify_cb;
    PieDriverCaps caps{};
    PieDriver* driver = pie_metal_create(&create, &caps);
    if (!expect(driver != nullptr, "pie_metal_create")) {
        std::remove(config_path.c_str());
        return 1;
    }

    // -- register program A (device geometry) --
    const std::vector<std::uint8_t> a_container = device_geometry_container();
    const std::vector<std::uint8_t> a_sidecar = device_geometry_sidecar(a_container);
    PieProgramDesc a_program{};
    a_program.abi_version = PIE_DRIVER_ABI_VERSION;
    a_program.program_hash = 0xA0000001ULL;
    a_program.canonical_bytes = {.ptr = a_container.data(), .len = a_container.size()};
    a_program.sidecar_bytes = {.ptr = a_sidecar.data(), .len = a_sidecar.size()};
    std::uint64_t a_program_id = 0;
    expect(pie_metal_register_program(driver, &a_program, &a_program_id) == PIE_STATUS_OK,
          "register program A (device geometry)");

    const std::uint32_t shape1[] = {1};
    PieChannelDesc a_chan0{};
    a_chan0.abi_version = PIE_DRIVER_ABI_VERSION;
    a_chan0.channel_id = 201;
    a_chan0.shape = {.ptr = shape1, .len = 1};
    a_chan0.dtype = PIE_CHANNEL_DTYPE_U32;
    a_chan0.host_role = PIE_CHANNEL_HOST_ROLE_NONE;
    a_chan0.seeded = 0;
    a_chan0.extern_dir = PIE_CHANNEL_EXTERN_NONE;
    a_chan0.capacity = 1;
    a_chan0.reader_wait_id = 11;
    a_chan0.writer_wait_id = 12;
    PieChannelEndpointBinding a_chan0_binding{};
    expect(pie_metal_register_channel(driver, &a_chan0, &a_chan0_binding) == PIE_STATUS_OK,
          "register program A's positions channel (never produced)");

    const std::uint64_t a_channel_ids[] = {201};
    PieInstanceDesc a_instance{};
    a_instance.abi_version = PIE_DRIVER_ABI_VERSION;
    a_instance.program_id = a_program_id;
    a_instance.channel_ids = {.ptr = a_channel_ids, .len = 1};
    PieInstanceBinding a_binding{};
    expect(pie_metal_bind_instance(driver, &a_instance, &a_binding) == PIE_STATUS_OK,
          "bind program A instance");

    // -- a SECOND device-geometry instance of the SAME program (its own
    //    channel), to prove "at most one device-geometry program per batch"
    //    (metal_ptir_plan.md §6) rejects synchronously BEFORE any member is
    //    processed — neither instance's terminal cell is touched. --
    PieChannelDesc a2_chan0{};
    a2_chan0.abi_version = PIE_DRIVER_ABI_VERSION;
    a2_chan0.channel_id = 291;
    a2_chan0.shape = {.ptr = shape1, .len = 1};
    a2_chan0.dtype = PIE_CHANNEL_DTYPE_U32;
    a2_chan0.host_role = PIE_CHANNEL_HOST_ROLE_NONE;
    a2_chan0.seeded = 0;
    a2_chan0.extern_dir = PIE_CHANNEL_EXTERN_NONE;
    a2_chan0.capacity = 1;
    a2_chan0.reader_wait_id = 91;
    a2_chan0.writer_wait_id = 92;
    PieChannelEndpointBinding a2_chan0_binding{};
    expect(pie_metal_register_channel(driver, &a2_chan0, &a2_chan0_binding) == PIE_STATUS_OK,
          "register a second device-geometry instance's channel");
    const std::uint64_t a2_channel_ids[] = {291};
    PieInstanceDesc a2_instance{};
    a2_instance.abi_version = PIE_DRIVER_ABI_VERSION;
    a2_instance.program_id = a_program_id;
    a2_instance.channel_ids = {.ptr = a2_channel_ids, .len = 1};
    PieInstanceBinding a2_binding{};
    expect(pie_metal_bind_instance(driver, &a2_instance, &a2_binding) == PIE_STATUS_OK,
          "bind a second device-geometry instance");
    {
        const std::uint64_t two_dg_instance_ids[] = {a_binding.instance_id, a2_binding.instance_id};
        PieTerminalCell t0{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        PieTerminalCell t1{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
        PieTerminalCell* two_dg_terminal_ptrs[] = {&t0, &t1};
        PieLaunchDesc two_dg_launch{};
        two_dg_launch.abi_version = PIE_DRIVER_ABI_VERSION;
        two_dg_launch.instance_ids = {.ptr = two_dg_instance_ids, .len = 2};
        two_dg_launch.terminal_cells = {.ptr = two_dg_terminal_ptrs, .len = 2};
        const PieCompletion two_dg_completion{
            .wait_id = 0, .target_epoch = 0, .terminal_cell = nullptr};
        const int32_t two_dg_rc = pie_metal_launch(driver, &two_dg_launch, two_dg_completion);
        expect(two_dg_rc == PIE_STATUS_INVALID_ARGUMENT,
              "two device-geometry programs in one batch rejects with INVALID_ARGUMENT (rc=" +
                  std::to_string(two_dg_rc) + ")");
        expect(t0.outcome == PIE_TERMINAL_OUTCOME_PENDING && t1.outcome == PIE_TERMINAL_OUTCOME_PENDING,
              "neither instance's terminal cell is touched by the rejected batch");
    }
    expect(pie_metal_close_instance(driver, a2_binding.instance_id) == PIE_STATUS_OK,
          "close the second device-geometry instance (done with the batch-limit check)");

    // -- register program B (ordinary channel-plane passthrough) --
    const std::vector<std::uint8_t> b_container = passthrough_container();
    const std::vector<std::uint8_t> b_sidecar = passthrough_sidecar(b_container);
    PieProgramDesc b_program{};
    b_program.abi_version = PIE_DRIVER_ABI_VERSION;
    b_program.program_hash = 0xB0000001ULL;
    b_program.canonical_bytes = {.ptr = b_container.data(), .len = b_container.size()};
    b_program.sidecar_bytes = {.ptr = b_sidecar.data(), .len = b_sidecar.size()};
    std::uint64_t b_program_id = 0;
    expect(pie_metal_register_program(driver, &b_program, &b_program_id) == PIE_STATUS_OK,
          "register program B (passthrough)");

    PieChannelDesc b_chan0{};
    b_chan0.abi_version = PIE_DRIVER_ABI_VERSION;
    b_chan0.channel_id = 202;
    b_chan0.shape = {.ptr = shape1, .len = 1};
    b_chan0.dtype = PIE_CHANNEL_DTYPE_U32;
    b_chan0.host_role = PIE_CHANNEL_HOST_ROLE_WRITER;
    b_chan0.seeded = 1;
    b_chan0.extern_dir = PIE_CHANNEL_EXTERN_NONE;
    b_chan0.capacity = 1;
    b_chan0.reader_wait_id = 21;
    b_chan0.writer_wait_id = 22;
    PieChannelEndpointBinding b_chan0_binding{};
    expect(pie_metal_register_channel(driver, &b_chan0, &b_chan0_binding) == PIE_STATUS_OK,
          "register program B's writer channel");

    PieChannelDesc b_chan1{};
    b_chan1.abi_version = PIE_DRIVER_ABI_VERSION;
    b_chan1.channel_id = 203;
    b_chan1.shape = {.ptr = shape1, .len = 1};
    b_chan1.dtype = PIE_CHANNEL_DTYPE_U32;
    b_chan1.host_role = PIE_CHANNEL_HOST_ROLE_READER;
    b_chan1.seeded = 0;
    b_chan1.extern_dir = PIE_CHANNEL_EXTERN_NONE;
    b_chan1.capacity = 1;
    b_chan1.reader_wait_id = 23;
    b_chan1.writer_wait_id = 24;
    PieChannelEndpointBinding b_chan1_binding{};
    expect(pie_metal_register_channel(driver, &b_chan1, &b_chan1_binding) == PIE_STATUS_OK,
          "register program B's reader channel");

    const std::uint64_t b_channel_ids[] = {202, 203};
    const std::uint8_t seed_bytes[4] = {7, 0, 0, 0};
    PieChannelValueDesc b_seed{};
    b_seed.channel_id = 202;
    b_seed.bytes = {.ptr = seed_bytes, .len = 4};
    PieInstanceDesc b_instance{};
    b_instance.abi_version = PIE_DRIVER_ABI_VERSION;
    b_instance.program_id = b_program_id;
    b_instance.channel_ids = {.ptr = b_channel_ids, .len = 2};
    b_instance.seed_values = {.ptr = &b_seed, .len = 1};
    PieInstanceBinding b_binding{};
    expect(pie_metal_bind_instance(driver, &b_instance, &b_binding) == PIE_STATUS_OK,
          "bind program B instance (chan0 seeded 7)");

    // -- launch BOTH in one batch: A's descriptor channel is never produced --
    const std::uint64_t instance_ids[] = {a_binding.instance_id, b_binding.instance_id};
    PieTerminalCell a_terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
    PieTerminalCell b_terminal{.outcome = PIE_TERMINAL_OUTCOME_PENDING, .reserved0 = 0};
    PieTerminalCell* terminal_ptrs[] = {&a_terminal, &b_terminal};

    PieLaunchDesc launch{};
    launch.abi_version = PIE_DRIVER_ABI_VERSION;
    launch.instance_ids = {.ptr = instance_ids, .len = 2};
    launch.terminal_cells = {.ptr = terminal_ptrs, .len = 2};

    const PieCompletion completion{.wait_id = 99, .target_epoch = 1, .terminal_cell = nullptr};
    const int32_t launch_rc = pie_metal_launch(driver, &launch, completion);
    expect(launch_rc == PIE_STATUS_OK, "launch accepts the batch (rc=" + std::to_string(launch_rc) + ")");
    // Async (review item 1): the batch settles on the worker; wait for the
    // batch notify (published last) before asserting settlement.
    notify_state.wait(99, 1);

    // -- A: poisoned (not-ready descriptor channel, no dummy-run) --
    expect(a_terminal.outcome == PIE_TERMINAL_OUTCOME_FAILED,
          "A's terminal outcome == FAILED (got " + std::to_string(a_terminal.outcome) + ")");
    const std::uint64_t a_poison =
        *reinterpret_cast<const std::uint64_t*>(a_chan0_binding.word_base + 16);  // word[2] = poison
    expect(a_poison != 0, "A's positions channel word is poisoned (nonzero epoch)");
    // chan0 is host_role=NONE (a realistic device-geometry descriptor
    // channel), so `poison_instance` has no reader/writer wait id to notify
    // for it (`notify()` itself no-ops on wait_id == 0) — the channel's own
    // poison WORD is the only observable here; B's notify below is the
    // meaningful "did the batch keep settling/notifying correctly" check.
    expect(!notify_state.contains(0, a_poison),
          "no spurious wait_id==0 notification is queued for A's NONE-role channel");

    // -- B: settles normally, unaffected by A's failure in the same batch --
    expect(b_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
          "B's terminal outcome == SUCCESS (got " + std::to_string(b_terminal.outcome) + ")");
    std::uint32_t taken = 0;
    std::memcpy(&taken, reinterpret_cast<void*>(b_chan1_binding.mirror_base), 4);
    expect(taken == 7, "B's reader channel took the seeded value 7 through (got " +
                          std::to_string(taken) + ")");
    const std::uint64_t b_reader_tail =
        *reinterpret_cast<const std::uint64_t*>(b_chan1_binding.word_base + 8);  // word[1] = tail
    expect(b_reader_tail == 1, "B's reader tail advanced to 1");
    expect(notify_state.contains(b_chan1.reader_wait_id, 1),
          "B's reader wait id was notified (unaffected by A's poison)");

    // -- the batch-level completion still notifies exactly once --
    expect(notify_state.contains(completion.wait_id, completion.target_epoch),
          "batch completion notified exactly once despite A's per-member failure");

    pie_metal_destroy(driver);
    std::remove(config_path.c_str());
    std::printf("\n==== ptir_device_geometry_e2e_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
