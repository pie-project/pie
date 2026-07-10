#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "entry.hpp"

namespace {

struct NotifyState {
    std::uint32_t count = 0;
    std::uint64_t last_wait_id = 0;
    std::uint64_t last_epoch = 0;
};

void notify_cb(void* ctx, std::uint64_t wait_id, std::uint64_t epoch) {
    auto* state = static_cast<NotifyState*>(ctx);
    state->count += 1;
    state->last_wait_id = wait_id;
    state->last_epoch = epoch;
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
    const PieChannelWait waits[] = {
        {.reader_wait_id = 101, .writer_wait_id = 201},
        {.reader_wait_id = 102, .writer_wait_id = 202},
        {.reader_wait_id = 103, .writer_wait_id = 203},
    };
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
    instance.channel_waits.ptr = waits;
    instance.channel_waits.len = 3;
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
    bad_instance.channel_waits = {.ptr = nullptr, .len = 3};
    if (!expect(pie_metal_bind_instance(driver, &bad_instance, &binding) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "bind rejects null waits")) return 1;
    bad_instance = instance;
    bad_instance.channel_waits.len = 2;
    if (!expect(pie_metal_bind_instance(driver, &bad_instance, &binding) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "bind rejects parallel count mismatch")) return 1;
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
    if (!expect(binding.channel_count == 1, "reader-only channel_count")) return 1;
    if (!expect(binding.word_count == 4, "reader-only word_count")) return 1;
    if (!expect(binding.channels.ptr != nullptr && binding.channels.len == 1,
                "binding slice")) return 1;
    if (!expect(binding.channels.ptr[0].channel_id == 33, "reader channel id")) return 1;
    if (!expect(binding.channels.ptr[0].head_word_index == 1 &&
                    binding.channels.ptr[0].tail_word_index == 2 &&
                    binding.channels.ptr[0].poison_word_index == 3,
                "reader word layout")) return 1;

    auto* mirror = reinterpret_cast<const std::uint8_t*>(binding.mirror_base);
    auto* words = reinterpret_cast<const std::uint64_t*>(binding.word_base);
    if (!expect(std::memcmp(mirror, seed_bytes, sizeof(seed_bytes)) == 0,
                "seed mirrored")) return 1;
    if (!expect(words[2] == 1, "seed tail published")) return 1;

    const std::uint8_t put_bytes[] = {0x10, 0x11, 0x12, 0x13};
    const PieChannelValueDesc puts[] = {{
        .channel_id = 11,
        .bytes = {.ptr = put_bytes, .len = sizeof(put_bytes)},
    }};
    const std::uint64_t instance_ids[] = {binding.instance_id};
    const std::uint32_t host_put_indptr[] = {0, 1};
    PieLaunchDesc launch{};
    launch.abi_version = PIE_DRIVER_ABI_VERSION;
    launch.instance_ids.ptr = instance_ids;
    launch.instance_ids.len = 1;
    launch.ptir_host_put_values.ptr = puts;
    launch.ptir_host_put_values.len = 1;
    launch.host_put_indptr.ptr = host_put_indptr;
    launch.host_put_indptr.len = 2;
    const PieCompletion completion{.wait_id = 77, .target_epoch = 5};
    const PieCompletion rejected_completion{.wait_id = 88, .target_epoch = 6};
    const std::uint32_t empty_csr[] = {0};
    PieLaunchDesc device_geometry_launch{};
    device_geometry_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    device_geometry_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    device_geometry_launch.masks.request_indptr = {
        .ptr = empty_csr,
        .len = 1,
    };
    device_geometry_launch.masks.word_indptr = {
        .ptr = empty_csr,
        .len = 1,
    };
    if (!expect(pie_metal_launch(driver, &device_geometry_launch, {}) ==
                    PIE_STATUS_OK,
                "device-geometry empty mask encoding")) return 1;

    PieLaunchDesc bad_launch = launch;
    bad_launch.abi_version += 1;
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "launch rejects wrong ABI")) return 1;
    bad_launch = launch;
    bad_launch.single_token_mode = 2;
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects invalid bool")) return 1;
    bad_launch = launch;
    bad_launch.reserved_flags[3] = 1;
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects reserved flags")) return 1;
    const std::uint32_t bad_host_put_indptr[] = {0, 2};
    bad_launch = launch;
    bad_launch.host_put_indptr = {
        .ptr = bad_host_put_indptr,
        .len = 2,
    };
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects malformed host-put CSR")) return 1;
    PieChannelValueDesc bad_put = puts[0];
    bad_put.bytes = {.ptr = nullptr, .len = 4};
    bad_launch = launch;
    bad_launch.ptir_host_put_values = {.ptr = &bad_put, .len = 1};
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects nested host-put bytes")) return 1;
    bad_put = {
        .channel_id = 33,
        .bytes = {.ptr = put_bytes, .len = sizeof(put_bytes)},
    };
    bad_launch = launch;
    bad_launch.ptir_host_put_values = {.ptr = &bad_put, .len = 1};
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects host put to non-writer channel")) return 1;
    const std::uint32_t token[] = {1};
    const std::uint32_t position[] = {0};
    const std::uint32_t bad_qo_indptr[] = {0, 0};
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    bad_launch.token_ids = {.ptr = token, .len = 1};
    bad_launch.position_ids = {.ptr = position, .len = 1};
    bad_launch.qo_indptr = {.ptr = bad_qo_indptr, .len = 2};
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects nonterminal qo CSR")) return 1;
    const std::uint32_t qo_indptr[] = {0, 1};
    bad_launch.qo_indptr = {.ptr = qo_indptr, .len = 2};
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
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
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects out-of-range sampling row")) return 1;
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {
        .ptr = reinterpret_cast<const std::uint64_t*>(alignof(std::uint64_t)),
        .len = std::numeric_limits<std::size_t>::max(),
    };
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects count overflow")) return 1;
    const std::uint32_t rs_slot[] = {0};
    const std::uint8_t bad_rs_flag[] = {4};
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    bad_launch.rs_slot_ids = {.ptr = rs_slot, .len = 1};
    bad_launch.rs_slot_flags = {.ptr = bad_rs_flag, .len = 1};
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects invalid recurrent-state flags")) return 1;
    const std::uint32_t oversized_rs_slot[] = {
        static_cast<std::uint32_t>(std::numeric_limits<int>::max()) + 1u,
    };
    const std::uint8_t valid_rs_flag[] = {0};
    bad_launch.rs_slot_ids = {.ptr = oversized_rs_slot, .len = 1};
    bad_launch.rs_slot_flags = {.ptr = valid_rs_flag, .len = 1};
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects recurrent slot cast overflow")) return 1;
    const std::uint32_t image_grid[] = {1, 1, 1};
    const std::uint32_t image_anchor[] = {0};
    const std::uint32_t image_pixel_indptr[] = {0, 0};
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    bad_launch.image_indptr = {.ptr = kv_indptr, .len = 2};
    bad_launch.image_grids = {.ptr = image_grid, .len = 3};
    bad_launch.image_anchor_positions = {.ptr = image_anchor, .len = 1};
    bad_launch.image_anchor_rows = {.ptr = image_anchor, .len = 1};
    bad_launch.image_pixel_indptr = {
        .ptr = image_pixel_indptr,
        .len = 2,
    };
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects image anchor outside token rows")) return 1;
    bad_launch = {};
    bad_launch.abi_version = PIE_DRIVER_ABI_VERSION;
    bad_launch.instance_ids = {.ptr = instance_ids, .len = 1};
    bad_launch.token_ids = {
        .ptr = reinterpret_cast<const std::uint32_t*>(1),
        .len = 1,
    };
    if (!expect(pie_metal_launch(driver, &bad_launch, rejected_completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects unaligned slices")) return 1;
    if (!expect(notify.count == 0, "rejected launches do not notify")) return 1;

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
    if (!expect(notify.count == 0, "rejected typed operations do not notify")) return 1;

    if (!expect(pie_metal_launch(driver, &launch, completion) == PIE_STATUS_OK,
                "launch")) return 1;
    if (!expect(notify.count == 1 && notify.last_wait_id == 77 && notify.last_epoch == 5,
                "notify once")) return 1;
    if (!expect(words[0] == 2, "pacing word")) return 1;
    if (!expect(std::memcmp(mirror + binding.channels.ptr[0].mirror_offset,
                            seed_bytes, sizeof(seed_bytes)) == 0,
                "reader seed remains visible")) return 1;
    if (!expect(words[3] == 0, "poison stays clear")) return 1;

    if (!expect(pie_metal_close_instance(driver, binding.instance_id) == PIE_STATUS_OK,
                "close_instance")) return 1;
    pie_metal_destroy(driver);
    std::puts("metal_direct_stub_test: OK");
    return 0;
}
