#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <pie_driver_abi.h>
#include "pie_native/ptir/container.hpp"  // fnv1a64 (the PTIB sidecar's container hash)

namespace {

// Phase 3 (review item 1): launches are now ASYNC — the driver posts the
// forward/settlement to its executor worker and `pie_metal_launch` returns
// after acceptance. Terminals/words/notifies are published later, from the
// worker thread. So this notify sink is thread-safe (the worker calls
// `notify_cb` off the caller thread) and exposes `wait(wait_id, epoch)` so the
// test blocks for a launch's BATCH notify — published last, after words +
// terminals + per-channel notifies — before asserting settlement.
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
    std::size_t count() const {
        std::lock_guard<std::mutex> lock(mu);
        return log.size();
    }
    bool contains(std::uint64_t wait_id, std::uint64_t epoch) const {
        std::lock_guard<std::mutex> lock(mu);
        return contains_locked(wait_id, epoch);
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mu);
        log.clear();
    }
    bool empty() const {
        std::lock_guard<std::mutex> lock(mu);
        return log.empty();
    }
    std::pair<std::uint64_t, std::uint64_t> back() const {
        std::lock_guard<std::mutex> lock(mu);
        return log.back();
    }
    // Block until (wait_id, epoch) has been recorded — the async launch's batch
    // notify, which the driver publishes last.
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

bool expect(bool cond, const char* msg) {
    if (!cond) std::fprintf(stderr, "FAIL: %s\n", msg);
    return cond;
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

PieTerminalCell pending_terminal() {
    return PieTerminalCell{
        .outcome = PIE_TERMINAL_OUTCOME_PENDING,
        .reserved0 = 0,
    };
}

std::vector<std::uint8_t> mixed_role_program(std::uint32_t capacity = 1) {
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    push_u16(out, 1);
    push_u16(out, 0);
    push_u32(out, 0);  // names
    push_u32(out, 3);  // channels
    push_u32(out, 0);  // ports
    push_u32(out, 0);  // stages
    for (std::uint8_t role : {std::uint8_t{1}, std::uint8_t{0}, std::uint8_t{2}}) {
        out.push_back(2);  // u32
        out.push_back(1);  // rank
        push_u32(out, 1);
        push_u32(out, capacity);
        out.push_back(role);
        out.push_back(role == 2 ? 1 : 0);
    }
    return out;
}

// Channel-plane container: c0 = host Writer u32[1], c1 = host Reader u32[1],
// epilogue `v0 = c0.take(); v1 = const 1u; v2 = v0 + v1; c1.put(v2)`.
std::vector<std::uint8_t> add_one_program(bool seeded_writer) {
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
        out.push_back(c == 0 ? PTIR_HOST_WRITER : PTIR_HOST_READER);
        out.push_back(c == 0 && seeded_writer ? 1 : 0);
    }
    out.push_back(PTIR_STAGE_EPILOGUE);
    push_u32(out, 4);  // ops
    out.push_back(PTIR_OP_CHAN_TAKE);
    push_u32(out, 0);
    out.push_back(PTIR_OP_CONST);
    out.push_back(PTIR_DT_U32);
    push_u32(out, 1);
    out.push_back(PTIR_OP_ADD);
    push_u32(out, 0);
    push_u32(out, 1);
    out.push_back(PTIR_OP_CHAN_PUT);
    push_u32(out, 1);
    push_u32(out, 2);
    return out;
}

// The matching PTIB sidecar: classes, the derived readiness table (c0 take →
// NeedsFull, c1 leading put → NeedsEmpty, both at the epilogue), and the
// epilogue's three SSA value types.
std::vector<std::uint8_t> add_one_sidecar(const std::vector<std::uint8_t>& container) {
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
    push_u32(out, 0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    out.push_back(0);  // NeedsFull
    push_u32(out, 1);
    out.push_back(PTIR_STAGE_EPILOGUE);
    out.push_back(1);  // NeedsEmpty
    push_u32(out, 1);  // stages
    out.push_back(PTIR_STAGE_EPILOGUE);
    push_u32(out, 3);  // value types: u32[1], u32 scalar, u32[1]
    out.push_back(PTIR_DT_U32);
    out.push_back(1);
    push_u32(out, 1);
    out.push_back(PTIR_DT_U32);
    out.push_back(0);
    out.push_back(PTIR_DT_U32);
    out.push_back(1);
    push_u32(out, 1);
    return out;
}

std::vector<std::uint8_t> extern_program(std::uint8_t direction) {
    constexpr char name[] = "shared";
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    push_u16(out, 2);
    push_u16(out, 0);
    push_u32(out, 1);  // names
    push_u32(out, 1);  // channels
    push_u32(out, 0);  // ports
    push_u32(out, 0);  // stages
    push_u32(out, 1);  // externs
    push_u16(out, sizeof(name) - 1);
    out.insert(out.end(), name, name + sizeof(name) - 1);
    out.push_back(2);  // u32
    out.push_back(1);  // rank
    push_u32(out, 1);
    push_u32(out, 1);  // capacity
    out.push_back(0);  // host none
    out.push_back(0);  // unseeded
    push_u16(out, 0);  // name index
    out.push_back(direction);
    push_u32(out, 0);  // dense channel
    return out;
}

}  // namespace

int main() {
    const std::string config_path = "../dev.toml";
    NotifyState notify{};
    PieDriverCreateDesc create{};
    create.abi_version = PIE_DRIVER_ABI_VERSION;
    create.config_bytes.ptr =
        reinterpret_cast<const std::uint8_t*>(config_path.data());
    create.config_bytes.len = config_path.size();
    create.runtime.abi_version = PIE_DRIVER_ABI_VERSION;
    create.runtime.ctx = &notify;
    create.runtime.notify = notify_cb;

    PieDriverCaps caps{};
    PieDriverCreateDesc bad_create = create;
    bad_create.abi_version += 1;
    if (!expect(pie_metal_create(&bad_create, &caps) == nullptr,
                "create rejects wrong ABI")) return 1;
    bad_create = create;
    bad_create.runtime.abi_version += 1;
    if (!expect(pie_metal_create(&bad_create, &caps) == nullptr,
                "create rejects callback ABI")) return 1;
    bad_create = create;
    bad_create.runtime.notify = nullptr;
    if (!expect(pie_metal_create(&bad_create, &caps) == nullptr,
                "create rejects missing notify")) return 1;
    bad_create = create;
    bad_create.config_bytes = {.ptr = nullptr, .len = 1};
    if (!expect(pie_metal_create(&bad_create, &caps) == nullptr,
                "create rejects null config bytes")) return 1;
    if (!expect(pie_metal_create(&create, nullptr) == nullptr,
                "create rejects null caps output")) return 1;

    PieDriver* driver = pie_metal_create(&create, &caps);
    if (!expect(driver != nullptr, "driver create")) return 1;

    const auto program_bytes = mixed_role_program();
    PieProgramDesc program{};
    program.abi_version = PIE_DRIVER_ABI_VERSION;
    program.program_hash = 0x1234;
    program.canonical_bytes.ptr = program_bytes.data();
    program.canonical_bytes.len = program_bytes.size();
    std::uint64_t program_id = 0;
    const auto oversized_capacity_bytes = mixed_role_program(8);
    PieProgramDesc oversized_capacity_program = program;
    oversized_capacity_program.program_hash += 1;
    oversized_capacity_program.canonical_bytes = {
        .ptr = oversized_capacity_bytes.data(),
        .len = oversized_capacity_bytes.size(),
    };
    if (!expect(pie_metal_register_program(
                    driver,
                    &oversized_capacity_program,
                    &program_id) == PIE_STATUS_INVALID_ARGUMENT,
                "register rejects unsupported channel capacity")) return 1;
    PieProgramDesc bad_program = program;
    bad_program.abi_version += 1;
    if (!expect(pie_metal_register_program(driver, &bad_program, &program_id) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "register rejects wrong ABI")) return 1;
    if (!expect(pie_metal_register_program(driver, &program, nullptr) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "register rejects null output")) return 1;
    bad_program = program;
    bad_program.sidecar_bytes = {.ptr = nullptr, .len = 1};
    if (!expect(pie_metal_register_program(driver, &bad_program, &program_id) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "register rejects null sidecar bytes")) return 1;
    bad_program = program;
    bad_program.canonical_bytes = {
        .ptr = reinterpret_cast<const std::uint8_t*>(1),
        .len = std::numeric_limits<std::size_t>::max(),
    };
    if (!expect(pie_metal_register_program(driver, &bad_program, &program_id) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "register rejects byte extent overflow")) return 1;
    if (!expect(pie_metal_register_program(driver, &program, &program_id) == PIE_STATUS_OK,
                "register_program")) return 1;
    PieProgramDesc cached_program = program;
    cached_program.canonical_bytes = {};
    std::uint64_t cached_program_id = 0;
    if (!expect(pie_metal_register_program(
                    driver, &cached_program, &cached_program_id) == PIE_STATUS_OK &&
                    cached_program_id == program_id,
                "hash-only cached registration")) return 1;
    cached_program.program_hash += 1;
    if (!expect(pie_metal_register_program(
                    driver, &cached_program, &cached_program_id) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "cache miss requires canonical bytes")) return 1;

    const std::uint64_t channel_ids[] = {11, 22, 33};
    const std::uint32_t scalar_shape[] = {1};
    PieChannelDesc channel_descs[3]{};
    for (std::size_t i = 0; i < 3; ++i) {
        channel_descs[i].abi_version = PIE_DRIVER_ABI_VERSION;
        channel_descs[i].channel_id = channel_ids[i];
        channel_descs[i].shape = {.ptr = scalar_shape, .len = 1};
        channel_descs[i].dtype = PIE_CHANNEL_DTYPE_U32;
        channel_descs[i].capacity = 1;
        channel_descs[i].reader_wait_id = 101 + i;
        channel_descs[i].writer_wait_id = 201 + i;
    }
    channel_descs[0].host_role = PIE_CHANNEL_HOST_ROLE_WRITER;
    channel_descs[1].host_role = PIE_CHANNEL_HOST_ROLE_NONE;
    channel_descs[2].host_role = PIE_CHANNEL_HOST_ROLE_READER;
    channel_descs[2].seeded = 1;
    PieChannelEndpointBinding endpoints[3]{};
    PieChannelDesc bad_channel = channel_descs[0];
    bad_channel.abi_version += 1;
    if (!expect(pie_metal_register_channel(driver, &bad_channel, &endpoints[0]) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "channel rejects wrong ABI")) return 1;
    bad_channel = channel_descs[0];
    bad_channel.reader_wait_id = bad_channel.writer_wait_id;
    if (!expect(pie_metal_register_channel(driver, &bad_channel, &endpoints[0]) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "channel rejects duplicate wait ids")) return 1;
    for (std::size_t i = 0; i < 3; ++i) {
        if (!expect(pie_metal_register_channel(
                        driver, &channel_descs[i], &endpoints[i]) == PIE_STATUS_OK,
                    "register_channel")) return 1;
        if (!expect(endpoints[i].channel_id == channel_ids[i] &&
                        endpoints[i].mirror_base != 0 &&
                        endpoints[i].word_base != 0 &&
                        endpoints[i].cell_bytes == sizeof(std::uint32_t) &&
                        endpoints[i].capacity == 1 &&
                        endpoints[i].head_word_index == 0 &&
                        endpoints[i].tail_word_index == 1 &&
                        endpoints[i].poison_word_index == 2 &&
                        endpoints[i].closed_word_index == 3,
                    "endpoint binding layout")) return 1;
    }
    if (!expect(pie_metal_register_channel(
                    driver, &channel_descs[0], &endpoints[0]) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "duplicate channel registration rejected")) return 1;
    const std::uint8_t seed_bytes[] = {0xAA, 0xBB, 0xCC, 0xDD};
    const PieChannelValueDesc seeds[] = {{
        .channel_id = 33,
        .bytes = {.ptr = seed_bytes, .len = sizeof(seed_bytes)},
    }};
    PieInstanceDesc instance{};
    instance.abi_version = PIE_DRIVER_ABI_VERSION;
    instance.program_id = program_id;
    instance.channel_ids.ptr = channel_ids;
    instance.channel_ids.len = 3;
    instance.seed_values.ptr = seeds;
    instance.seed_values.len = 1;

    PieInstanceBinding binding{};
    PieInstanceDesc bad_instance = instance;
    bad_instance.abi_version += 1;
    if (!expect(pie_metal_bind_instance(driver, &bad_instance, &binding) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "bind rejects wrong ABI")) return 1;
    if (!expect(pie_metal_bind_instance(driver, &instance, nullptr) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "bind rejects null output")) return 1;
    bad_instance = instance;
    bad_instance.channel_ids.len = 2;
    if (!expect(pie_metal_bind_instance(driver, &bad_instance, &binding) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "bind rejects channel count mismatch")) return 1;
    PieChannelValueDesc bad_seed = seeds[0];
    bad_seed.bytes = {.ptr = nullptr, .len = 4};
    bad_instance = instance;
    bad_instance.seed_values = {.ptr = &bad_seed, .len = 1};
    if (!expect(pie_metal_bind_instance(driver, &bad_instance, &binding) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "bind rejects nested seed bytes")) return 1;
    bad_seed = {
        .channel_id = 11,
        .bytes = {.ptr = seed_bytes, .len = sizeof(seed_bytes)},
    };
    bad_instance = instance;
    bad_instance.seed_values = {.ptr = &bad_seed, .len = 1};
    if (!expect(pie_metal_bind_instance(driver, &bad_instance, &binding) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "bind rejects seed for non-seeded channel")) return 1;
    bad_instance = instance;
    bad_instance.seed_values = {
        .ptr = reinterpret_cast<const PieChannelValueDesc*>(
            alignof(PieChannelValueDesc)),
        .len = std::numeric_limits<std::size_t>::max(),
    };
    if (!expect(pie_metal_bind_instance(driver, &bad_instance, &binding) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "bind rejects slice byte overflow")) return 1;
    if (!expect(pie_metal_bind_instance(driver, &instance, &binding) == PIE_STATUS_OK,
                "bind_instance")) return 1;
    if (!expect(binding.instance_id != 0, "instance identity returned")) return 1;
    if (!expect(pie_metal_close_channel(driver, 33) == PIE_STATUS_INVALID_ARGUMENT,
                "close_channel rejects live attachment")) return 1;

    // Seeds are per-instance interp state (D2), never bind-time ring
    // publications: the reader ring stays untouched until a fire produces.
    auto* mirror = reinterpret_cast<const std::uint8_t*>(endpoints[2].mirror_base);
    auto* words = reinterpret_cast<const std::uint64_t*>(endpoints[2].word_base);
    const std::uint8_t zero_cell[4] = {0, 0, 0, 0};
    if (!expect(std::memcmp(mirror, zero_cell, sizeof(zero_cell)) == 0,
                "bind leaves the reader ring untouched")) return 1;
    if (!expect(words[1] == 0, "bind publishes no reader tail")) return 1;

    PieInstanceDesc duplicate_instance = instance;
    duplicate_instance.requested_instance_id = binding.instance_id;
    PieInstanceBinding duplicate_binding{};
    if (!expect(pie_metal_bind_instance(
                    driver, &duplicate_instance, &duplicate_binding) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "duplicate bind preserves original instance")) return 1;

    const std::uint64_t instance_ids[] = {binding.instance_id};
    PieLaunchDesc launch{};
    launch.abi_version = PIE_DRIVER_ABI_VERSION;
    launch.instance_ids.ptr = instance_ids;
    launch.instance_ids.len = 1;
    PieTerminalCell launch_terminal = pending_terminal();
    PieTerminalCell* const launch_terminal_ptrs[] = {&launch_terminal};
    launch.terminal_cells = {.ptr = launch_terminal_ptrs, .len = 1};
    PieTerminalCell rejected_terminal = pending_terminal();
    const PieCompletion launch_completion{
        .wait_id = 77,
        .target_epoch = 5,
        .terminal_cell = nullptr,
    };
    const PieCompletion rejected_completion{
        .wait_id = 88,
        .target_epoch = 6,
        .terminal_cell = &rejected_terminal,
    };
    const std::uint32_t empty_csr[] = {0};
    PieLaunchDesc device_geometry_launch{};
    device_geometry_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    device_geometry_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    device_geometry_launch.terminal_cells = {.ptr = launch_terminal_ptrs, .len = 1};
    device_geometry_launch.masks.request_indptr = {
        .ptr = empty_csr,
        .len = 1,
    };
    device_geometry_launch.masks.word_indptr = {
        .ptr = empty_csr,
        .len = 1,
    };
    if (!expect(pie_metal_launch(driver, &device_geometry_launch, launch_completion) ==
                    PIE_STATUS_UNSUPPORTED,
                "device-geometry empty mask encoding")) return 1;

    PieLaunchDesc bad_launch = launch;
    bad_launch.abi_version += 1;
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "launch rejects wrong ABI")) return 1;
    bad_launch = launch;
    bad_launch.single_token_mode = 2;
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects invalid bool")) return 1;
    bad_launch = launch;
    bad_launch.reserved_flags[3] = 1;
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects reserved flags")) return 1;
    const std::uint32_t token[] = {1};
    const std::uint32_t position[] = {0};
    const std::uint32_t bad_qo_indptr[] = {0, 0};
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    bad_launch.terminal_cells = {.ptr = launch_terminal_ptrs, .len = 1};
    bad_launch.token_ids = {.ptr = token, .len = 1};
    bad_launch.position_ids = {.ptr = position, .len = 1};
    bad_launch.qo_indptr = {.ptr = bad_qo_indptr, .len = 2};
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects nonterminal qo CSR")) return 1;
    const std::uint32_t qo_indptr[] = {0, 1};
    bad_launch.qo_indptr = {.ptr = qo_indptr, .len = 2};
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "model launch requires KV and sampling CSRs")) return 1;
    const std::uint32_t kv_page[] = {0};
    const std::uint32_t kv_indptr[] = {0, 1};
    const std::uint32_t kv_last_len[] = {1};
    const std::uint32_t bad_sampling_index[] = {1};
    const std::uint32_t sampling_indptr[] = {0, 1};
    bad_launch.kv_page_indices = {.ptr = kv_page, .len = 1};
    bad_launch.kv_page_indptr = {.ptr = kv_indptr, .len = 2};
    bad_launch.kv_last_page_lens = {.ptr = kv_last_len, .len = 1};
    bad_launch.sampling_indices = {
        .ptr = bad_sampling_index,
        .len = 1,
    };
    bad_launch.sampling_indptr = {.ptr = sampling_indptr, .len = 2};
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects out-of-range sampling row")) return 1;
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {
        .ptr = reinterpret_cast<const std::uint64_t*>(alignof(std::uint64_t)),
        .len = std::numeric_limits<std::size_t>::max(),
    };
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects count overflow")) return 1;
    const std::uint32_t rs_slot[] = {0};
    const std::uint8_t bad_rs_flag[] = {4};
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    bad_launch.terminal_cells = {.ptr = launch_terminal_ptrs, .len = 1};
    bad_launch.rs_slot_ids = {.ptr = rs_slot, .len = 1};
    bad_launch.rs_slot_flags = {.ptr = bad_rs_flag, .len = 1};
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects invalid recurrent-state flags")) return 1;
    const std::uint32_t oversized_rs_slot[] = {
        static_cast<std::uint32_t>(std::numeric_limits<int>::max()) + 1u,
    };
    const std::uint8_t valid_rs_flag[] = {0};
    bad_launch.rs_slot_ids = {.ptr = oversized_rs_slot, .len = 1};
    bad_launch.rs_slot_flags = {.ptr = valid_rs_flag, .len = 1};
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects recurrent slot cast overflow")) return 1;
    const std::uint32_t image_grid[] = {1, 1, 1};
    const std::uint32_t image_anchor[] = {0};
    const std::uint32_t image_pixel_indptr[] = {0, 0};
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    bad_launch.terminal_cells = {.ptr = launch_terminal_ptrs, .len = 1};
    bad_launch.image_indptr = {.ptr = kv_indptr, .len = 2};
    bad_launch.image_grids = {.ptr = image_grid, .len = 3};
    bad_launch.image_anchor_positions = {.ptr = image_anchor, .len = 1};
    bad_launch.image_anchor_rows = {.ptr = image_anchor, .len = 1};
    bad_launch.image_pixel_indptr = {
        .ptr = image_pixel_indptr,
        .len = 2,
    };
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects image anchor outside token rows")) return 1;
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    bad_launch.terminal_cells = {.ptr = launch_terminal_ptrs, .len = 1};
    bad_launch.token_ids = {
        .ptr = reinterpret_cast<const std::uint32_t*>(1),
        .len = 1,
    };
    if (!expect(pie_metal_launch(driver, &bad_launch, launch_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects unaligned slices")) return 1;
    if (!expect(notify.count() == 0, "rejected launches do not notify")) return 1;

    PieKvCopyDesc bad_kv{};
    bad_kv.abi_version = PIE_DRIVER_ABI_VERSION + 1;
    if (!expect(pie_metal_copy_kv(driver, &bad_kv, rejected_completion) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "KV copy rejects wrong ABI")) return 1;
    bad_kv.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_kv.src_domain = 99;
    bad_kv.dst_domain = PIE_MEMORY_DOMAIN_HOST_PINNED;
    if (!expect(pie_metal_copy_kv(driver, &bad_kv, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "KV copy rejects invalid domain")) return 1;
    const std::uint32_t page[] = {1};
    bad_kv.src_domain = PIE_MEMORY_DOMAIN_HOST_PINNED;
    bad_kv.src_page_ids = {.ptr = page, .len = 1};
    if (!expect(pie_metal_copy_kv(driver, &bad_kv, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "KV copy rejects parallel count mismatch")) return 1;

    PieStateCopyDesc bad_state{};
    bad_state.abi_version = PIE_DRIVER_ABI_VERSION + 1;
    if (!expect(pie_metal_copy_state(driver, &bad_state, rejected_completion) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "state copy rejects wrong ABI")) return 1;
    const PieStateCopyRange overflowing_state{
        .src_token_offset = std::numeric_limits<std::uint32_t>::max(),
        .token_count = 1,
    };
    bad_state.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_state.slot_ranges = {.ptr = &overflowing_state, .len = 1};
    if (!expect(pie_metal_copy_state(driver, &bad_state, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "state copy rejects range overflow")) return 1;

    PiePoolResizeDesc bad_resize{};
    bad_resize.abi_version = PIE_DRIVER_ABI_VERSION + 1;
    if (!expect(pie_metal_resize_pool(driver, &bad_resize, rejected_completion) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "resize rejects wrong ABI")) return 1;
    const PiePoolRange overflowing_range{
        .page_index = std::numeric_limits<std::uint64_t>::max(),
        .page_count = 1,
    };
    bad_resize.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_resize.map_ranges = {.ptr = &overflowing_range, .len = 1};
    if (!expect(pie_metal_resize_pool(driver, &bad_resize, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "resize rejects range overflow")) return 1;
    bad_resize = {};
    bad_resize.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_resize.target_pages =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1;
    if (!expect(pie_metal_resize_pool(driver, &bad_resize, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "resize rejects target count overflow")) return 1;
    if (!expect(notify.count() == 0, "rejected typed operations do not notify")) return 1;

    if (!expect(pie_metal_launch(driver, &launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects operation terminal cell")) return 1;
    // The mixed-role program ships no PTIB sidecar, so it is not executable
    // on the host interpreter — launch rejects UNSUPPORTED without touching
    // any ring state.
    if (!expect(pie_metal_launch(driver, &launch, launch_completion) == PIE_STATUS_UNSUPPORTED,
                "launch of a sidecar-less program is unsupported")) return 1;
    if (!expect(notify.count() == 0, "unsupported launch does not notify")) return 1;
    if (!expect(words[0] == 0, "reader head unchanged")) return 1;
    if (!expect(words[1] == 0, "reader tail unchanged")) return 1;
    if (!expect(words[2] == 0, "poison stays clear")) return 1;
    if (!expect(words[3] == 0, "closed stays clear")) return 1;
    if (!expect(launch_terminal.outcome == PIE_TERMINAL_OUTCOME_PENDING,
                "unsupported launch keeps member terminal pending")) return 1;
    if (!expect(rejected_terminal.outcome == PIE_TERMINAL_OUTCOME_PENDING,
                "rejected launch keeps operation terminal pending")) return 1;

    PieTerminalCell copy_terminal = pending_terminal();
    const PieCompletion copy_completion{
        .wait_id = 91,
        .target_epoch = 2,
        .terminal_cell = &copy_terminal,
    };
    PieKvCopyDesc kv_copy{};
    kv_copy.abi_version = PIE_DRIVER_ABI_VERSION;
    kv_copy.src_domain = PIE_MEMORY_DOMAIN_HOST_PINNED;
    kv_copy.dst_domain = PIE_MEMORY_DOMAIN_HOST_PINNED;
    if (!expect(pie_metal_copy_kv(driver, &kv_copy, copy_completion) == PIE_STATUS_UNSUPPORTED,
                "copy_kv unsupported")) return 1;
    if (!expect(copy_terminal.outcome == PIE_TERMINAL_OUTCOME_PENDING,
                "unsupported copy_kv keeps terminal pending")) return 1;

    PieTerminalCell state_terminal = pending_terminal();
    const PieCompletion state_completion{
        .wait_id = 92,
        .target_epoch = 3,
        .terminal_cell = &state_terminal,
    };
    PieStateCopyDesc state_copy{};
    state_copy.abi_version = PIE_DRIVER_ABI_VERSION;
    if (!expect(
            pie_metal_copy_state(driver, &state_copy, state_completion) == PIE_STATUS_UNSUPPORTED,
            "copy_state unsupported")) return 1;
    if (!expect(state_terminal.outcome == PIE_TERMINAL_OUTCOME_PENDING,
                "unsupported copy_state keeps terminal pending")) return 1;

    PieTerminalCell resize_terminal = pending_terminal();
    const PieCompletion resize_completion{
        .wait_id = 93,
        .target_epoch = 4,
        .terminal_cell = &resize_terminal,
    };
    PiePoolResizeDesc resize{};
    resize.abi_version = PIE_DRIVER_ABI_VERSION;
    if (!expect(
            pie_metal_resize_pool(driver, &resize, resize_completion) == PIE_STATUS_UNSUPPORTED,
            "resize unsupported")) return 1;
    if (!expect(resize_terminal.outcome == PIE_TERMINAL_OUTCOME_PENDING,
                "unsupported resize keeps terminal pending")) return 1;
    if (!expect(notify.count() == 0, "unsupported operations never notify")) return 1;

    if (!expect(pie_metal_close_instance(driver, binding.instance_id) == PIE_STATUS_OK,
                "close_instance")) return 1;
    for (const std::uint64_t channel_id : channel_ids) {
        if (!expect(pie_metal_close_channel(driver, channel_id) == PIE_STATUS_OK,
                   "close_channel")) return 1;
        if (!expect(pie_metal_close_channel(driver, channel_id) == PIE_STATUS_CLOSED,
                   "closed channel stays closed")) return 1;
    }

    // ── real execution: put → launch → publish/notify per plan §4.3/§4.4 ──
    {
        const auto exec_bytes = add_one_program(false);
        const auto exec_sidecar = add_one_sidecar(exec_bytes);
        PieProgramDesc exec_program{};
        exec_program.abi_version = PIE_DRIVER_ABI_VERSION;
        exec_program.program_hash = 0x3001;
        exec_program.canonical_bytes = {.ptr = exec_bytes.data(), .len = exec_bytes.size()};
        exec_program.sidecar_bytes = {.ptr = exec_sidecar.data(), .len = exec_sidecar.size()};
        std::uint64_t exec_program_id = 0;
        if (!expect(pie_metal_register_program(driver, &exec_program, &exec_program_id) ==
                        PIE_STATUS_OK,
                    "register add_one program")) return 1;

        const std::uint64_t exec_channel_ids[] = {55, 66};
        PieChannelDesc exec_descs[2]{};
        for (std::size_t i = 0; i < 2; ++i) {
            exec_descs[i].abi_version = PIE_DRIVER_ABI_VERSION;
            exec_descs[i].channel_id = exec_channel_ids[i];
            exec_descs[i].shape = {.ptr = scalar_shape, .len = 1};
            exec_descs[i].dtype = PIE_CHANNEL_DTYPE_U32;
            exec_descs[i].capacity = 1;
            exec_descs[i].reader_wait_id = 501 + i * 10;
            exec_descs[i].writer_wait_id = 502 + i * 10;
        }
        exec_descs[0].host_role = PIE_CHANNEL_HOST_ROLE_WRITER;
        exec_descs[1].host_role = PIE_CHANNEL_HOST_ROLE_READER;
        PieChannelEndpointBinding exec_endpoints[2]{};
        for (std::size_t i = 0; i < 2; ++i) {
            if (!expect(pie_metal_register_channel(driver, &exec_descs[i],
                                                   &exec_endpoints[i]) == PIE_STATUS_OK,
                        "register add_one channel")) return 1;
        }
        PieInstanceDesc exec_instance{};
        exec_instance.abi_version = PIE_DRIVER_ABI_VERSION;
        exec_instance.program_id = exec_program_id;
        exec_instance.channel_ids = {.ptr = exec_channel_ids, .len = 2};
        PieInstanceBinding exec_binding{};
        if (!expect(pie_metal_bind_instance(driver, &exec_instance, &exec_binding) ==
                        PIE_STATUS_OK,
                    "bind add_one instance")) return 1;

        auto* writer_mirror = reinterpret_cast<std::uint8_t*>(exec_endpoints[0].mirror_base);
        auto* writer_words = reinterpret_cast<std::uint64_t*>(exec_endpoints[0].word_base);
        auto* reader_mirror = reinterpret_cast<const std::uint8_t*>(exec_endpoints[1].mirror_base);
        auto* reader_words = reinterpret_cast<std::uint64_t*>(exec_endpoints[1].word_base);

        const std::uint64_t exec_instance_ids[] = {exec_binding.instance_id};
        PieTerminalCell exec_terminal = pending_terminal();
        PieTerminalCell* const exec_terminal_ptrs[] = {&exec_terminal};
        PieLaunchDesc exec_launch{};
        exec_launch.abi_version = PIE_DRIVER_ABI_VERSION;
        exec_launch.instance_ids = {.ptr = exec_instance_ids, .len = 1};
        exec_launch.terminal_cells = {.ptr = exec_terminal_ptrs, .len = 1};
        const PieCompletion exec_completion{
            .wait_id = 777,
            .target_epoch = 9,
            .terminal_cell = nullptr,
        };

        // §4.3 availability: a fire taking from an empty host-writer ring
        // rejects synchronously — no epoch, no poison, no notify.
        if (!expect(pie_metal_launch(driver, &exec_launch, exec_completion) ==
                        PIE_STATUS_INVALID_ARGUMENT,
                    "launch without a host put rejects")) return 1;
        if (!expect(notify.count() == 0, "availability rejection does not notify")) return 1;
        if (!expect(exec_terminal.outcome == PIE_TERMINAL_OUTCOME_PENDING,
                    "availability rejection keeps terminal pending")) return 1;

        // Host put: wire bytes at `tail % cap1`, then the release tail store.
        const std::uint32_t put_value = 7;
        std::memcpy(writer_mirror, &put_value, sizeof(put_value));
        writer_words[1] = 1;
        if (!expect(pie_metal_launch(driver, &exec_launch, exec_completion) == PIE_STATUS_OK,
                    "put → launch succeeds")) return 1;
        // Async (review item 1): launch returns after acceptance; the worker
        // publishes words/terminal/notifies. Wait for the batch notify (last).
        notify.wait(777, 9);
        std::uint32_t produced = 0;
        std::memcpy(&produced, reader_mirror, sizeof(produced));
        if (!expect(writer_words[0] == 1, "consumed writer head published")) return 1;
        if (!expect(reader_words[1] == 1, "produced reader tail published")) return 1;
        if (!expect(produced == 8, "epilogue computed take + 1")) return 1;
        if (!expect(exec_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
                    "member terminal settles success")) return 1;
        if (!expect(notify.contains(502, 1), "writer wake carries the new head")) return 1;
        if (!expect(notify.contains(511, 1), "reader wake carries the new tail")) return 1;
        if (!expect(!notify.empty() && notify.back() == std::make_pair(
                        std::uint64_t{777}, std::uint64_t{9}),
                    "batch notify lands last, exactly once")) return 1;

        // Failure settlement (D4): a second fire with a fresh put but a full
        // reader ring (no host take) poisons and fails the member terminal.
        notify.clear();
        exec_terminal = pending_terminal();
        const std::uint32_t second_value = 20;
        std::memcpy(writer_mirror + exec_endpoints[0].cell_bytes, &second_value,
                    sizeof(second_value));
        writer_words[1] = 2;
        if (!expect(pie_metal_launch(driver, &exec_launch, exec_completion) == PIE_STATUS_OK,
                    "accepted fire settles even when publication fails")) return 1;
        notify.wait(777, 9);
        if (!expect(exec_terminal.outcome == PIE_TERMINAL_OUTCOME_FAILED,
                    "publication failure fails the member terminal")) return 1;
        if (!expect(reader_words[2] != 0, "reader poison word published")) return 1;
        if (!expect(notify.contains(511, reader_words[2]),
                    "reader wake carries the poison epoch")) return 1;
        if (!expect(!notify.empty() && notify.back() == std::make_pair(
                        std::uint64_t{777}, std::uint64_t{9}),
                    "failed fire still notifies the batch slot last")) return 1;

        // ── Driver-level async acceptance (review items 1/5): a fresh add_one
        //    instance on its own channels. `pie_metal_launch` returns after
        //    acceptance; `close_instance` then QUEUES BEHIND the launch job on
        //    the executor worker (FIFO), so by the time close returns the launch
        //    has fully settled — its terminal is SUCCESS and its batch notify
        //    was delivered exactly once, all published asynchronously from the
        //    worker (never from the launch call). This is the driver-level
        //    counterpart to executor_worker_test's mechanism-level proof. ──
        {
            const std::uint64_t async_channel_ids[] = {155, 166};
            PieChannelDesc async_descs[2]{};
            for (std::size_t i = 0; i < 2; ++i) {
                async_descs[i].abi_version = PIE_DRIVER_ABI_VERSION;
                async_descs[i].channel_id = async_channel_ids[i];
                async_descs[i].shape = {.ptr = scalar_shape, .len = 1};
                async_descs[i].dtype = PIE_CHANNEL_DTYPE_U32;
                async_descs[i].capacity = 1;
                async_descs[i].reader_wait_id = 1550 + i;
                async_descs[i].writer_wait_id = 1560 + i;
            }
            async_descs[0].host_role = PIE_CHANNEL_HOST_ROLE_WRITER;
            async_descs[1].host_role = PIE_CHANNEL_HOST_ROLE_READER;
            PieChannelEndpointBinding async_endpoints[2]{};
            for (std::size_t i = 0; i < 2; ++i) {
                if (!expect(pie_metal_register_channel(driver, &async_descs[i],
                                                       &async_endpoints[i]) == PIE_STATUS_OK,
                            "register async add_one channel")) return 1;
            }
            PieInstanceDesc async_instance{};
            async_instance.abi_version = PIE_DRIVER_ABI_VERSION;
            async_instance.program_id = exec_program_id;
            async_instance.channel_ids = {.ptr = async_channel_ids, .len = 2};
            PieInstanceBinding async_binding{};
            if (!expect(pie_metal_bind_instance(driver, &async_instance, &async_binding) ==
                            PIE_STATUS_OK,
                        "bind async add_one instance")) return 1;

            auto* async_writer_mirror =
                reinterpret_cast<std::uint8_t*>(async_endpoints[0].mirror_base);
            auto* async_writer_words =
                reinterpret_cast<std::uint64_t*>(async_endpoints[0].word_base);
            const std::uint32_t async_put = 41;
            std::memcpy(async_writer_mirror, &async_put, sizeof(async_put));
            async_writer_words[1] = 1;

            const std::uint64_t async_instance_ids[] = {async_binding.instance_id};
            PieTerminalCell async_terminal = pending_terminal();
            PieTerminalCell* const async_terminal_ptrs[] = {&async_terminal};
            PieLaunchDesc async_launch{};
            async_launch.abi_version = PIE_DRIVER_ABI_VERSION;
            async_launch.instance_ids = {.ptr = async_instance_ids, .len = 1};
            async_launch.terminal_cells = {.ptr = async_terminal_ptrs, .len = 1};
            const PieCompletion async_completion{.wait_id = 1777, .target_epoch = 4,
                                                 .terminal_cell = nullptr};
            notify.clear();
            if (!expect(pie_metal_launch(driver, &async_launch, async_completion) == PIE_STATUS_OK,
                        "async launch returns OK after acceptance (no wait)")) return 1;
            // Close WITHOUT waiting on the completion: close is a worker job
            // enqueued strictly behind the launch job (FIFO), so it cannot
            // return until the launch has settled.
            if (!expect(pie_metal_close_instance(driver, async_binding.instance_id) ==
                            PIE_STATUS_OK,
                        "close queues behind the in-flight launch")) return 1;
            if (!expect(async_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
                        "launch settled asynchronously before close returned (terminal SUCCESS)"))
                return 1;
            if (!expect(notify.contains(1777, 4),
                        "batch completion notified exactly once from the worker")) return 1;
            for (const std::uint64_t id : async_channel_ids) {
                if (!expect(pie_metal_close_channel(driver, id) == PIE_STATUS_OK,
                            "close async add_one channel")) return 1;
            }
            notify.clear();
        }

        if (!expect(pie_metal_close_instance(driver, exec_binding.instance_id) ==
                        PIE_STATUS_OK,
                    "close add_one instance")) return 1;
        for (const std::uint64_t id : exec_channel_ids) {
            if (!expect(pie_metal_close_channel(driver, id) == PIE_STATUS_OK,
                        "close add_one channel")) return 1;
        }
    }

    // ── seed credit: a seeded Writer channel's first take spends the seed
    //    without a host put and without a head-word publish (§4.3) ──
    {
        const auto seeded_bytes = add_one_program(true);
        const auto seeded_sidecar = add_one_sidecar(seeded_bytes);
        PieProgramDesc seeded_program{};
        seeded_program.abi_version = PIE_DRIVER_ABI_VERSION;
        seeded_program.program_hash = 0x3002;
        seeded_program.canonical_bytes = {.ptr = seeded_bytes.data(), .len = seeded_bytes.size()};
        seeded_program.sidecar_bytes = {.ptr = seeded_sidecar.data(),
                                        .len = seeded_sidecar.size()};
        std::uint64_t seeded_program_id = 0;
        if (!expect(pie_metal_register_program(driver, &seeded_program, &seeded_program_id) ==
                        PIE_STATUS_OK,
                    "register seeded add_one program")) return 1;

        const std::uint64_t seeded_channel_ids[] = {77, 88};
        PieChannelDesc seeded_descs[2]{};
        for (std::size_t i = 0; i < 2; ++i) {
            seeded_descs[i].abi_version = PIE_DRIVER_ABI_VERSION;
            seeded_descs[i].channel_id = seeded_channel_ids[i];
            seeded_descs[i].shape = {.ptr = scalar_shape, .len = 1};
            seeded_descs[i].dtype = PIE_CHANNEL_DTYPE_U32;
            seeded_descs[i].capacity = 1;
            seeded_descs[i].reader_wait_id = 601 + i * 10;
            seeded_descs[i].writer_wait_id = 602 + i * 10;
        }
        seeded_descs[0].host_role = PIE_CHANNEL_HOST_ROLE_WRITER;
        seeded_descs[0].seeded = 1;
        seeded_descs[1].host_role = PIE_CHANNEL_HOST_ROLE_READER;
        PieChannelEndpointBinding seeded_endpoints[2]{};
        for (std::size_t i = 0; i < 2; ++i) {
            if (!expect(pie_metal_register_channel(driver, &seeded_descs[i],
                                                   &seeded_endpoints[i]) == PIE_STATUS_OK,
                        "register seeded channel")) return 1;
        }
        const std::uint32_t seed_value = 41;
        const PieChannelValueDesc seeded_seeds[] = {{
            .channel_id = 77,
            .bytes = {.ptr = reinterpret_cast<const std::uint8_t*>(&seed_value),
                      .len = sizeof(seed_value)},
        }};
        PieInstanceDesc seeded_instance{};
        seeded_instance.abi_version = PIE_DRIVER_ABI_VERSION;
        seeded_instance.program_id = seeded_program_id;
        seeded_instance.channel_ids = {.ptr = seeded_channel_ids, .len = 2};
        seeded_instance.seed_values = {.ptr = seeded_seeds, .len = 1};
        PieInstanceBinding seeded_binding{};
        if (!expect(pie_metal_bind_instance(driver, &seeded_instance, &seeded_binding) ==
                        PIE_STATUS_OK,
                    "bind seeded instance")) return 1;

        auto* seeded_writer_words =
            reinterpret_cast<const std::uint64_t*>(seeded_endpoints[0].word_base);
        auto* seeded_reader_mirror =
            reinterpret_cast<const std::uint8_t*>(seeded_endpoints[1].mirror_base);
        const std::uint64_t seeded_instance_ids[] = {seeded_binding.instance_id};
        PieTerminalCell seeded_terminal = pending_terminal();
        PieTerminalCell* const seeded_terminal_ptrs[] = {&seeded_terminal};
        PieLaunchDesc seeded_launch{};
        seeded_launch.abi_version = PIE_DRIVER_ABI_VERSION;
        seeded_launch.instance_ids = {.ptr = seeded_instance_ids, .len = 1};
        seeded_launch.terminal_cells = {.ptr = seeded_terminal_ptrs, .len = 1};
        const PieCompletion seeded_completion{
            .wait_id = 888,
            .target_epoch = 3,
            .terminal_cell = nullptr,
        };
        notify.clear();
        if (!expect(pie_metal_launch(driver, &seeded_launch, seeded_completion) ==
                        PIE_STATUS_OK,
                    "seed credit satisfies the first fire without a put")) return 1;
        notify.wait(888, 3);
        std::uint32_t seeded_out = 0;
        std::memcpy(&seeded_out, seeded_reader_mirror, sizeof(seeded_out));
        if (!expect(seeded_out == 42, "seed value flowed through the program")) return 1;
        if (!expect(seeded_writer_words[0] == 0,
                    "seed spend publishes no writer head")) return 1;
        if (!expect(!notify.contains(602, 0) && !notify.contains(602, 1),
                    "seed spend sends no writer wake")) return 1;
        if (!expect(seeded_terminal.outcome == PIE_TERMINAL_OUTCOME_SUCCESS,
                    "seeded fire settles success")) return 1;

        // The credit is spent: a second fire needs a real host put.
        seeded_terminal = pending_terminal();
        if (!expect(pie_metal_launch(driver, &seeded_launch, seeded_completion) ==
                        PIE_STATUS_INVALID_ARGUMENT,
                    "spent seed credit no longer satisfies availability")) return 1;

        if (!expect(pie_metal_close_instance(driver, seeded_binding.instance_id) ==
                        PIE_STATUS_OK,
                    "close seeded instance")) return 1;
        for (const std::uint64_t id : seeded_channel_ids) {
            if (!expect(pie_metal_close_channel(driver, id) == PIE_STATUS_OK,
                        "close seeded channel")) return 1;
        }
        notify.clear();
    }

    const auto export_bytes = extern_program(1);
    const auto import_bytes = extern_program(0);
    PieProgramDesc export_program{};
    export_program.abi_version = PIE_DRIVER_ABI_VERSION;
    export_program.program_hash = 0x2001;
    export_program.canonical_bytes = {
        .ptr = export_bytes.data(),
        .len = export_bytes.size(),
    };
    PieProgramDesc import_program = export_program;
    import_program.program_hash = 0x2002;
    import_program.canonical_bytes = {
        .ptr = import_bytes.data(),
        .len = import_bytes.size(),
    };
    std::uint64_t export_program_id = 0;
    std::uint64_t import_program_id = 0;
    if (!expect(pie_metal_register_program(
                   driver, &export_program, &export_program_id) == PIE_STATUS_OK,
                "register extern export program")) return 1;
    if (!expect(pie_metal_register_program(
                   driver, &import_program, &import_program_id) == PIE_STATUS_OK,
                "register extern import program")) return 1;

    const std::uint64_t shared_channel_id = 44;
    const std::uint8_t shared_name[] = {'s', 'h', 'a', 'r', 'e', 'd'};
    PieChannelDesc shared_channel{};
    shared_channel.abi_version = PIE_DRIVER_ABI_VERSION;
    shared_channel.channel_id = shared_channel_id;
    shared_channel.shape = {.ptr = scalar_shape, .len = 1};
    shared_channel.dtype = PIE_CHANNEL_DTYPE_U32;
    shared_channel.extern_dir = PIE_CHANNEL_EXTERN_EXPORT;
    shared_channel.capacity = 1;
    shared_channel.reader_wait_id = 401;
    shared_channel.writer_wait_id = 402;
    shared_channel.extern_name = {
        .ptr = shared_name,
        .len = sizeof(shared_name),
    };
    PieChannelEndpointBinding shared_endpoint{};
    if (!expect(pie_metal_register_channel(
                   driver, &shared_channel, &shared_endpoint) == PIE_STATUS_OK,
                "register shared endpoint")) return 1;
    PieInstanceDesc export_instance{};
    export_instance.abi_version = PIE_DRIVER_ABI_VERSION;
    export_instance.program_id = export_program_id;
    export_instance.requested_instance_id = 100;
    export_instance.channel_ids = {.ptr = &shared_channel_id, .len = 1};
    PieInstanceBinding export_binding{};
    if (!expect(pie_metal_bind_instance(
                   driver, &export_instance, &export_binding) == PIE_STATUS_OK,
                "bind extern exporter")) return 1;
    PieInstanceDesc import_instance = export_instance;
    import_instance.program_id = import_program_id;
    import_instance.requested_instance_id = 101;
    PieInstanceBinding import_binding{};
    if (!expect(pie_metal_bind_instance(
                   driver, &import_instance, &import_binding) == PIE_STATUS_OK,
                "bind extern importer")) return 1;
    PieInstanceDesc duplicate_export = export_instance;
    duplicate_export.requested_instance_id = 102;
    PieInstanceBinding duplicate_export_binding{};
    if (!expect(pie_metal_bind_instance(
                   driver, &duplicate_export, &duplicate_export_binding) ==
                   PIE_STATUS_INVALID_ARGUMENT,
                "reject duplicate extern direction")) return 1;
    if (!expect(pie_metal_close_channel(driver, shared_channel_id) ==
                   PIE_STATUS_INVALID_ARGUMENT,
                "shared endpoint rejects live close")) return 1;
    if (!expect(pie_metal_close_instance(driver, export_binding.instance_id) ==
                   PIE_STATUS_OK,
                "close exporter")) return 1;
    if (!expect(pie_metal_close_instance(driver, import_binding.instance_id) ==
                   PIE_STATUS_OK,
                "close importer")) return 1;
    if (!expect(pie_metal_close_channel(driver, shared_channel_id) ==
                   PIE_STATUS_OK,
                "close shared endpoint")) return 1;

    pie_metal_destroy(driver);
    std::puts("metal_direct_stub_test: OK");
    return 0;
}
