#include <cstdint>
#include <cstdio>
#include <limits>

#include <cuda_runtime.h>
#include <pie_driver_abi.h>

#include "entry_validation.hpp"
#include "kernels/slab_scatter.hpp"
#include "pie_native/abi_validation.hpp"
#include "pie_native/ptir/fire_geometry.hpp"
#include "store/recurrent_state_cache.hpp"

namespace {

void notify_cb(void*, std::uint64_t, std::uint64_t) {}

bool expect(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

// v14: the launch entry takes a sealed frame; validator tests wrap one step
// in a fixed-roster frame.
constexpr std::size_t kTestRosterLen = 16;
const std::uint32_t kRosterRows[] = {0, 1, 2, 3, 4, 5, 6, 7};
const std::uint64_t kRosterIds[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

int launch_step(PieDriver* driver,
                const PieStepDesc& step,
                PieCompletion completion) {
    PieFrameDesc frame{};
    frame.abi_version = PIE_DRIVER_ABI_VERSION;
    frame.instance_ids = {.ptr = kRosterIds, .len = kTestRosterLen};
    frame.steps = {.ptr = &step, .len = 1};
    return pie_cuda_launch(driver, &frame, completion);
}

}  // namespace

int main() {
    const std::uint8_t config_byte = 0;
    PieDriverCreateDesc create{};
    create.abi_version = PIE_DRIVER_ABI_VERSION;
    create.config_bytes = {.ptr = &config_byte, .len = 1};
    create.runtime.abi_version = PIE_DRIVER_ABI_VERSION;
    create.runtime.notify = notify_cb;
    PieDriverCaps caps{};

    PieDriverCreateDesc bad_create = create;
    bad_create.abi_version += 1;
    if (!expect(pie_cuda_create(&bad_create, &caps) == nullptr,
                "create rejects wrong ABI")) return 1;
    bad_create = create;
    bad_create.runtime.notify = nullptr;
    if (!expect(pie_cuda_create(&bad_create, &caps) == nullptr,
                "create rejects missing notify")) return 1;
    if (!expect(pie_cuda_create(&create, nullptr) == nullptr,
                "create rejects null caps output")) return 1;

    auto* driver = reinterpret_cast<PieDriver*>(std::uintptr_t{1});
    const std::uint8_t program_byte = 1;
    std::uint64_t program_id = 0;
    PieProgramDesc program{};
    program.abi_version = PIE_DRIVER_ABI_VERSION;
    program.canonical_bytes = {.ptr = &program_byte, .len = 1};

    PieProgramDesc bad_program = program;
    bad_program.abi_version += 1;
    if (!expect(pie_cuda_register_program(driver, &bad_program, &program_id) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "register rejects wrong ABI")) return 1;
    bad_program = program;
    bad_program.sidecar_bytes = {.ptr = nullptr, .len = 1};
    if (!expect(pie_cuda_register_program(driver, &bad_program, &program_id) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "register rejects invalid sidecar")) return 1;
    if (!expect(pie_cuda_register_program(driver, &program, nullptr) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "register rejects null output")) return 1;

    const std::uint64_t channel_id = 3;
    const std::uint32_t shape = 1;
    PieChannelEndpointBinding endpoint{};
    PieChannelDesc channel{};
    channel.abi_version = PIE_DRIVER_ABI_VERSION + 1;
    channel.channel_id = channel_id;
    channel.shape = {.ptr = &shape, .len = 1};
    channel.capacity = 1;
    channel.reader_wait_id = 4;
    channel.writer_wait_id = 5;
    if (!expect(pie_cuda_register_channel(driver, &channel, &endpoint) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "channel rejects wrong ABI")) return 1;
    channel.abi_version = PIE_DRIVER_ABI_VERSION;
    channel.writer_wait_id = channel.reader_wait_id;
    if (!expect(pie_cuda_register_channel(driver, &channel, &endpoint) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "channel rejects duplicate wait ids")) return 1;

    PieInstanceBinding binding{};
    PieInstanceDesc instance{};
    instance.abi_version = PIE_DRIVER_ABI_VERSION;
    instance.channel_ids = {.ptr = &channel_id, .len = 1};

    PieInstanceDesc bad_instance = instance;
    bad_instance.abi_version += 1;
    if (!expect(pie_cuda_bind_instance(driver, &bad_instance, &binding) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "bind rejects wrong ABI")) return 1;
    PieChannelValueDesc bad_seed{
        .channel_id = channel_id,
        .bytes = {.ptr = nullptr, .len = 1},
    };
    bad_instance = instance;
    bad_instance.seed_values = {.ptr = &bad_seed, .len = 1};
    if (!expect(pie_cuda_bind_instance(driver, &bad_instance, &binding) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "bind rejects nested seed bytes")) return 1;

    const PieCompletion completion{.wait_id = 9, .target_epoch = 2};
    PieStepDesc launch{};
    launch.single_token_mode = 2;
    if (!expect(launch_step(driver, launch, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects invalid bool")) return 1;
    launch = {};
    launch.reserved_flags[0] = 1;
    if (!expect(launch_step(driver, launch, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects reserved flags")) return 1;
    const std::uint64_t instance_id = 1;
    const std::uint32_t token = 1;
    const std::uint32_t position = 0;
    const std::uint32_t bad_qo_indptr[] = {0, 0};
    launch = {};
    launch.roster_rows = {.ptr = kRosterRows, .len = 1};
    launch.token_ids = {.ptr = &token, .len = 1};
    launch.position_ids = {.ptr = &position, .len = 1};
    launch.qo_indptr = {.ptr = bad_qo_indptr, .len = 2};
    if (!expect(launch_step(driver, launch, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects malformed CSR")) return 1;
    const std::uint32_t qo_indptr[] = {0, 1};
    launch.qo_indptr = {.ptr = qo_indptr, .len = 2};
    if (!expect(launch_step(driver, launch, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "model launch requires KV and sampling CSRs")) return 1;
    const std::uint32_t sampling_index = 0;
    const std::uint32_t sampling_indptr[] = {0, 1};
    launch = {};
    launch.roster_rows = {.ptr = kRosterRows, .len = 1};
    launch.sampling_indices = {.ptr = &sampling_index, .len = 1};
    launch.sampling_indptr = {.ptr = sampling_indptr, .len = 2};
    if (!expect(launch_step(driver, launch, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects sampling rows without query geometry")) return 1;
    const std::uint32_t empty_kv_indptr[] = {0, 0};
    launch = {};
    launch.roster_rows = {.ptr = kRosterRows, .len = 1};
    launch.kv_page_indptr = {.ptr = empty_kv_indptr, .len = 2};
    if (!expect(launch_step(driver, launch, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects KV CSR without last-page lengths")) return 1;
    launch = {};
    launch.roster_rows = {
        .ptr = reinterpret_cast<const std::uint32_t*>(alignof(std::uint32_t)),
        .len = std::numeric_limits<std::size_t>::max(),
    };
    if (!expect(launch_step(driver, launch, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "launch rejects count overflow")) return 1;
    const std::uint32_t resolved_rs_slots[] = {2, 3};
    const std::uint8_t resolved_rs_flags[] = {
        PIE_RS_FLAG_RESET, 0};
    const std::uint32_t resolved_fold_lens[] = {0, 0};
    const std::uint32_t resolved_buffer_slots[] = {1, 2};
    const std::uint32_t resolved_buffer_indptr[] = {0, 1, 2};
    PieTerminalCell resolved_terminal{};
    PieTerminalCell* resolved_terminals[] = {&resolved_terminal};
    PieStepDesc device_geometry_rs{};
    device_geometry_rs.roster_rows = {.ptr = kRosterRows, .len = 1};
    device_geometry_rs.terminal_cells = {
        .ptr = resolved_terminals,
        .len = 1,
    };
    device_geometry_rs.rs_slot_ids = {
        .ptr = resolved_rs_slots,
        .len = 2,
    };
    device_geometry_rs.rs_slot_flags = {
        .ptr = resolved_rs_flags,
        .len = 2,
    };
    device_geometry_rs.rs_fold_lens = {
        .ptr = resolved_fold_lens,
        .len = 2,
    };
    device_geometry_rs.rs_buffer_slot_ids = {
        .ptr = resolved_buffer_slots,
        .len = 2,
    };
    device_geometry_rs.rs_buffer_slot_indptr = {
        .ptr = resolved_buffer_indptr,
        .len = 3,
    };
    if (!expect(
            pie_native::abi::validate_step_desc(&device_geometry_rs, kTestRosterLen) ==
                    PIE_STATUS_OK &&
                pie_cuda_driver::abi::validate_launch_resources(
                    device_geometry_rs, 4, 16, 8, 8,
                    pie_cuda_driver::abi::MultimodalLimits{}, nullptr, 0) ==
                    PIE_STATUS_OK,
            "solo instance accepts B=2 resolved folded RS rows")) {
        return 1;
    }
    const std::uint32_t bad_buffer_indptr[] = {0, 2, 1};
    device_geometry_rs.rs_buffer_slot_indptr = {
        .ptr = bad_buffer_indptr,
        .len = 3,
    };
    if (!expect(
            pie_native::abi::validate_step_desc(&device_geometry_rs, kTestRosterLen) ==
                PIE_STATUS_INVALID_ARGUMENT,
            "deferred buffered RS CSR still rejects non-monotonic input")) {
        return 1;
    }
    device_geometry_rs.rs_buffer_slot_indptr = {
        .ptr = resolved_buffer_indptr,
        .len = 3,
    };
    device_geometry_rs.rs_slot_flags.len = 1;
    if (!expect(
            pie_native::abi::validate_step_desc(&device_geometry_rs, kTestRosterLen) ==
                PIE_STATUS_INVALID_ARGUMENT,
            "resolved folded RS rejects mismatched ids/flags")) {
        return 1;
    }
    device_geometry_rs.rs_slot_flags.len = 2;
    device_geometry_rs.rs_fold_lens.len = 1;
    if (!expect(
            pie_native::abi::validate_step_desc(&device_geometry_rs, kTestRosterLen) ==
                PIE_STATUS_INVALID_ARGUMENT,
            "resolved folded RS rejects mismatched fold lengths")) {
        return 1;
    }

    PieKvCopyDesc kv{};
    kv.abi_version = PIE_DRIVER_ABI_VERSION + 1;
    if (!expect(pie_cuda_copy_kv(driver, &kv, completion) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "KV copy rejects wrong ABI")) return 1;
    kv.abi_version = PIE_DRIVER_ABI_VERSION;
    kv.src_domain = 99;
    kv.dst_domain = PIE_MEMORY_DOMAIN_HOST_PINNED;
    if (!expect(pie_cuda_copy_kv(driver, &kv, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "KV copy rejects invalid domain")) return 1;

    PieStateCopyDesc state{};
    state.abi_version = PIE_DRIVER_ABI_VERSION + 1;
    if (!expect(pie_cuda_copy_state(driver, &state, completion) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "state copy rejects wrong ABI")) return 1;
    const PieStateCopyRange overflowing_state{
        .src_token_offset = std::numeric_limits<std::uint32_t>::max(),
        .token_count = 1,
    };
    state.abi_version = PIE_DRIVER_ABI_VERSION;
    state.slot_ranges = {.ptr = &overflowing_state, .len = 1};
    if (!expect(pie_cuda_copy_state(driver, &state, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "state copy rejects range overflow")) return 1;

    PiePoolResizeDesc resize{};
    resize.abi_version = PIE_DRIVER_ABI_VERSION + 1;
    if (!expect(pie_cuda_resize_pool(driver, &resize, completion) ==
                    PIE_STATUS_BAD_ABI_VERSION,
                "resize rejects wrong ABI")) return 1;
    const PiePoolRange overflowing_range{
        .page_index = std::numeric_limits<std::uint64_t>::max(),
        .page_count = 1,
    };
    resize.abi_version = PIE_DRIVER_ABI_VERSION;
    resize.map_ranges = {.ptr = &overflowing_range, .len = 1};
    if (!expect(pie_cuda_resize_pool(driver, &resize, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "resize rejects range overflow")) return 1;
    resize = {};
    resize.abi_version = PIE_DRIVER_ABI_VERSION;
    resize.target_pages =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1;
    if (!expect(pie_cuda_resize_pool(driver, &resize, completion) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "resize rejects target count overflow")) return 1;

    const std::uint32_t page_id[] = {4};
    const std::uint32_t page_indptr[] = {0, 1};
    const std::uint32_t last_page_len[] = {1};
    const pie_cuda_driver::abi::MultimodalLimits no_multimodal{};
    PieStepDesc resource_launch{};
    resource_launch.roster_rows = {.ptr = kRosterRows, .len = 1};
    resource_launch.kv_page_indices = {.ptr = page_id, .len = 1};
    resource_launch.kv_page_indptr = {.ptr = page_indptr, .len = 2};
    resource_launch.kv_last_page_lens = {
        .ptr = last_page_len,
        .len = 1,
    };
    if (!expect(pie_cuda_driver::abi::validate_launch_resources(
                    resource_launch, 4, 16, 0, 0, no_multimodal,
                    nullptr, 0) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "CUDA launch rejects out-of-range KV page")) return 1;

    const std::uint32_t valid_page_id[] = {3};
    const std::uint32_t oversized_last_page_len[] = {17};
    resource_launch.kv_page_indices = {.ptr = valid_page_id, .len = 1};
    resource_launch.kv_last_page_lens = {
        .ptr = oversized_last_page_len,
        .len = 1,
    };
    const std::uint32_t unit_query_indptr[] = {0, 1};
    resource_launch.qo_indptr = {
        .ptr = unit_query_indptr,
        .len = 2,
    };
    if (!expect(pie_cuda_driver::abi::validate_launch_resources(
                    resource_launch, 4, 16, 0, 0, no_multimodal,
                    nullptr, 0) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "CUDA launch rejects oversized last page")) return 1;
    const std::uint32_t oversized_query_indptr[] = {0, 2};
    resource_launch.kv_last_page_lens = {
        .ptr = last_page_len,
        .len = 1,
    };
    resource_launch.qo_indptr = {
        .ptr = oversized_query_indptr,
        .len = 2,
    };
    if (!expect(pie_cuda_driver::abi::validate_launch_resources(
                    resource_launch, 4, 16, 0, 0, no_multimodal,
                    nullptr, 0) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "CUDA launch rejects KV shorter than query span")) return 1;
    const std::uint32_t out_of_range_rs_slot[] = {2};
    resource_launch.kv_page_indices = {};
    resource_launch.kv_page_indptr = {};
    resource_launch.kv_last_page_lens = {};
    resource_launch.rs_slot_ids = {
        .ptr = out_of_range_rs_slot,
        .len = 1,
    };
    if (!expect(pie_cuda_driver::abi::validate_launch_resources(
                    resource_launch, 4, 16, 2, 0, no_multimodal,
                    nullptr, 0) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "CUDA launch rejects out-of-range recurrent slot")) return 1;
    const std::uint32_t image_grid[] = {1, 1, 1};
    const std::uint32_t image_anchor_row[] = {0};
    const std::uint32_t image_pixel_indptr[] = {0, 3072};
    PieStepDesc image_launch{};
    image_launch.token_ids = {.ptr = &token, .len = 1};
    image_launch.image_grids = {.ptr = image_grid, .len = 3};
    image_launch.image_anchor_rows = {
        .ptr = image_anchor_row,
        .len = 1,
    };
    image_launch.image_pixel_indptr = {
        .ptr = image_pixel_indptr,
        .len = 2,
    };
    pie_cuda_driver::abi::MultimodalLimits gemma4_limits{};
    gemma4_limits.gemma4_pool_kernel = 1;
    gemma4_limits.gemma4_position_table = 16;
    if (!expect(pie_cuda_driver::abi::validate_launch_resources(
                    image_launch, 4, 16, 0, 0, gemma4_limits,
                    nullptr, 0) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "Gemma4 launch requires two positions per patch")) return 1;

    // C2: the deferred-geometry exemption is keyed on the OWNING
    // INSTANCE's ACK'd class, never on "this batch is PTIR". A
    // device-resolved row may ship empty wire pages (the driver resolves
    // its geometry in-graph); a Host-class row with query rows and no
    // pages is exactly the engine bug the pages/last-len rule catches.
    const std::uint32_t empty_page_indptr[] = {0, 0};
    const std::uint32_t zero_last_page_len[] = {0};
    const std::uint32_t one_program_row_indptr[] = {0, 1};
    PieStepDesc deferred_launch{};
    deferred_launch.roster_rows = {.ptr = kRosterRows, .len = 1};
    deferred_launch.ptir_program_row_indptr = {
        .ptr = one_program_row_indptr,
        .len = 2,
    };
    deferred_launch.kv_page_indptr = {.ptr = empty_page_indptr, .len = 2};
    deferred_launch.kv_last_page_lens = {
        .ptr = zero_last_page_len,
        .len = 1,
    };
    deferred_launch.qo_indptr = {.ptr = unit_query_indptr, .len = 2};
    const std::uint32_t envelope_class[] = {
        PIE_GEOMETRY_CLASS_DECODE_ENVELOPE};
    const std::uint32_t host_class[] = {PIE_GEOMETRY_CLASS_HOST};
    if (!expect(pie_cuda_driver::abi::validate_launch_resources(
                    deferred_launch, 4, 16, 0, 0, no_multimodal,
                    envelope_class, 1) == PIE_STATUS_OK,
                "device-resolved row defers its wire geometry")) return 1;
    if (!expect(pie_cuda_driver::abi::validate_launch_resources(
                    deferred_launch, 4, 16, 0, 0, no_multimodal,
                    host_class, 1) == PIE_STATUS_INVALID_ARGUMENT,
                "host-class row without pages rejects")) return 1;

    pie_native::ptir::FireGeometry geometry;
    geometry.token_ids = {1};
    geometry.position_ids = {0};
    geometry.qo_indptr = {0, 1};
    geometry.kv_page_indices = {0};
    geometry.kv_page_indptr = {0, 1};
    geometry.kv_last_page_lens = {1};
    geometry.sampling_indices = {0};
    geometry.sampling_indptr = {0, 1};
    geometry.w_page = {0};
    geometry.w_off = {0};
    geometry.has_kv_family = true;
    geometry.has_write_desc = true;
    if (!expect(pie_native::ptir::validate_fire_geometry(
                    geometry, 4, 16),
                "resolved device geometry accepts valid descriptor")) return 1;
    // Device extent kv_len=20 with one live row: the lane's first live
    // write sits at 19, and the write descriptor resolves to exactly that.
    geometry.kv_page_indices = {0, 1};
    geometry.kv_page_indptr = {0, 2};
    geometry.kv_last_page_lens = {4};
    geometry.w_page = {1};
    geometry.w_off = {3};
    if (!expect(pie_native::ptir::validate_kv_write_containment(
                    geometry, 16, 19, 20),
                "resolved write accepts the containment floor")) return 1;
    if (!expect(!pie_native::ptir::validate_kv_write_containment(
                    geometry, 16, 25, 26),
                "resolved write rejects a token below the exact declaration")) return 1;
    geometry.w_off = {2};
    if (!expect(!pie_native::ptir::validate_kv_write_containment(
                    geometry, 16, 25, 26),
                "resolved write rejects a token below the device extent")) return 1;
    geometry.w_off = {3};
    if (!expect(!pie_native::ptir::validate_kv_write_containment(
                    geometry, 16, 18, 19),
                "resolved write rejects the exclusive upper bound")) return 1;
    geometry.token_ids = {std::numeric_limits<std::uint32_t>::max()};
    if (!expect(pie_native::ptir::validate_kv_write_containment(
                    geometry, 16, 25, 26),
                "sentinel rows bypass the exact declaration bounds")) return 1;
    geometry.token_ids = {1};
    geometry.kv_page_indices = {0};
    geometry.kv_page_indptr = {0, 1};
    geometry.kv_last_page_lens = {1};
    geometry.w_page = {0};
    geometry.w_off = {0};
    geometry.structured_mask = {
        pie_native::ptir::StructuredMaskKind::Causal, 1, 0, 0};
    if (!expect(pie_native::ptir::validate_fire_geometry(
                   geometry, 4, 16),
                "structured extent validates without a dense mask")) return 1;
    geometry.structured_mask = {
        pie_native::ptir::StructuredMaskKind::SlidingWindow, 1, 0, 0};
    if (!expect(pie_native::ptir::validate_fire_geometry(
                   geometry, 4, 16),
                "zero-width sliding descriptor is valid")) return 1;
    geometry.structured_mask = {
        pie_native::ptir::StructuredMaskKind::SinkWindow, 1, 99, 0};
    if (!expect(pie_native::ptir::validate_fire_geometry(
                   geometry, 4, 16),
                "sink extent may exceed key length")) return 1;
    geometry.structured_mask.key_len = 0;
    if (!expect(!pie_native::ptir::validate_fire_geometry(
                   geometry, 4, 16),
                "structured extent rejects independently of dense fallback")) {
        return 1;
    }
    geometry.structured_mask = {};
    pie_native::ptir::FireGeometry idle_geometry;
    idle_geometry.qo_indptr = {0, 0};
    idle_geometry.sampling_indptr = {0, 0};
    idle_geometry.structured_mask = {
        pie_native::ptir::StructuredMaskKind::Causal, 16, 0, 0};
    if (!expect(pie_native::ptir::validate_fire_geometry(
                    idle_geometry, 4, 16),
                "idle structured geometry does not index absent KV arrays")) {
        return 1;
    }
    geometry.w_page[0] = 4;
    if (!expect(!pie_native::ptir::validate_fire_geometry(
                    geometry, 4, 16),
                "resolved device geometry rejects out-of-range write")) return 1;
    geometry.w_page[0] = 0;
    geometry.token_ids.push_back(2);
    geometry.position_ids.push_back(1);
    geometry.qo_indptr[1] = 2;
    geometry.w_page.push_back(0);
    geometry.w_off.push_back(1);
    if (!expect(!pie_native::ptir::validate_fire_geometry(
                    geometry, 4, 16),
                "resolved device geometry rejects short KV extent")) return 1;

    auto state_cache = pie_cuda_driver::RecurrentStateCache::allocate(
        {true}, 2, 2, 1, 1, 1, 2, 2);
    const std::size_t conv_bytes = state_cache.conv_slot_stride_bytes();
    cudaMemset(state_cache.conv_state(0, 0), 0x7f, conv_bytes);
    cudaMemset(state_cache.conv_state(0, 1), 0x7f, conv_bytes);
    cudaMemset(
        state_cache.mtp_pending_hidden(0), 0x7f,
        2 * sizeof(std::uint16_t));
    cudaMemset(
        state_cache.mtp_pending_hidden(1), 0x7f,
        2 * sizeof(std::uint16_t));
    const std::int32_t reset_slots[] = {-1, 1};
    const std::uint8_t reset_fresh[] = {1, 1};
    std::int32_t* reset_slots_d = nullptr;
    std::uint8_t* reset_fresh_d = nullptr;
    cudaMalloc(&reset_slots_d, sizeof(reset_slots));
    cudaMalloc(&reset_fresh_d, sizeof(reset_fresh));
    cudaMemcpy(
        reset_slots_d, reset_slots, sizeof(reset_slots),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        reset_fresh_d, reset_fresh, sizeof(reset_fresh),
        cudaMemcpyHostToDevice);
    state_cache.reset_slots_if_fresh(
        reset_slots_d, reset_fresh_d, 2);
    std::uint8_t slot0 = 0;
    std::uint8_t slot1 = 0xff;
    cudaMemcpy(
        &slot0, state_cache.conv_state(0, 0), 1,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        &slot1, state_cache.conv_state(0, 1), 1,
        cudaMemcpyDeviceToHost);
    if (!expect(
            slot0 == 0x7f && slot1 == 0,
            "device reset ignores negative sentinel slots")) return 1;

    const std::uint8_t copy_src[] = {0xaa, 0xbb};
    const std::uint8_t copy_initial[] = {0x11, 0x22};
    std::uint8_t* copy_src_d = nullptr;
    std::uint8_t* copy_dst_d = nullptr;
    cudaMalloc(&copy_src_d, sizeof(copy_src));
    cudaMalloc(&copy_dst_d, sizeof(copy_initial));
    cudaMemcpy(
        copy_src_d, copy_src, sizeof(copy_src),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        copy_dst_d, copy_initial, sizeof(copy_initial),
        cudaMemcpyHostToDevice);
    pie_cuda_driver::launch_copy_if_valid_slot(
        copy_src_d, copy_dst_d, 1, reset_slots_d, 0, nullptr);
    pie_cuda_driver::launch_copy_if_valid_slot(
        copy_src_d + 1, copy_dst_d + 1, 1, reset_slots_d, 1, nullptr);
    std::uint8_t copy_result[2]{};
    cudaMemcpy(
        copy_result, copy_dst_d, sizeof(copy_result),
        cudaMemcpyDeviceToHost);
    if (!expect(
            copy_result[0] == 0x11 && copy_result[1] == 0xbb,
            "buffer writes ignore negative sentinel slots")) return 1;
    cudaFree(copy_dst_d);
    cudaFree(copy_src_d);
    cudaFree(reset_fresh_d);
    cudaFree(reset_slots_d);

    const PieKvMoveCell bad_cell{
        .dst_page_id = 0,
        .dst_token_offset = 16,
        .src_page_id = 0,
        .src_token_offset = 0,
    };
    PieKvCopyDesc resource_copy{};
    resource_copy.src_domain = PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    resource_copy.dst_domain = PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    resource_copy.cells = {.ptr = &bad_cell, .len = 1};
    if (!expect(pie_cuda_driver::abi::validate_kv_copy_resources(
                    resource_copy, 0, 4, 2, 16, true) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "CUDA KV copy rejects cell offset")) return 1;
    resource_copy.src_domain = PIE_MEMORY_DOMAIN_HOST_PINNED;
    if (!expect(pie_cuda_driver::abi::validate_kv_copy_resources(
                    resource_copy, 0, 4, 2, 16, true) ==
                    PIE_STATUS_UNSUPPORTED,
                "CUDA KV copy preflights incompatible cells")) return 1;
    const PieKvMoveCell valid_cell{};
    resource_copy.src_domain = PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    resource_copy.dst_domain = PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    resource_copy.cells = {.ptr = &valid_cell, .len = 1};
    if (!expect(pie_cuda_driver::abi::validate_kv_copy_resources(
                    resource_copy, 0, 4, 2, 16, false) ==
                    PIE_STATUS_UNSUPPORTED,
                "CUDA KV copy preflights quantized cell moves")) return 1;
    const std::uint32_t out_of_range_page[] = {4};
    const std::uint32_t host_page[] = {0};
    resource_copy.cells = {};
    resource_copy.src_domain = PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    resource_copy.dst_domain = PIE_MEMORY_DOMAIN_HOST_PINNED;
    resource_copy.src_page_ids = {
        .ptr = out_of_range_page,
        .len = 1,
    };
    resource_copy.dst_page_ids = {.ptr = host_page, .len = 1};
    if (!expect(pie_cuda_driver::abi::validate_kv_copy_resources(
                    resource_copy, 0, 4, 2, 16, true) ==
                    PIE_STATUS_INVALID_ARGUMENT,
                "CUDA KV copy rejects out-of-range page")) return 1;

    const PieStateCopyRange state_ranges[] = {
        {.src_slot_id = 0, .dst_slot_id = 1},
        {.src_slot_id = 1,
         .dst_slot_id = 0,
         .src_token_offset = 1},
    };
    PieStateCopyDesc resource_state{};
    resource_state.slot_ranges = {.ptr = state_ranges, .len = 2};
    if (!expect(pie_cuda_driver::abi::validate_state_copy_resources(
                    resource_state, 2) == PIE_STATUS_UNSUPPORTED,
                "CUDA state copy preflights every range")) return 1;

    std::puts("cuda_entry_validation_test: OK");
    return 0;
}
