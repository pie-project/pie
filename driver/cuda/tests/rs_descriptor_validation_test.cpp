#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <pie_driver_abi.h>

#include "pie_native/abi_validation.hpp"
#include "pipeline/batch_compose.hpp"

namespace {

int failures = 0;

// v14: step members reference the frame roster by index; validator tests use
// identity rows under a fixed roster size.
constexpr std::size_t kTestRosterLen = 16;
const std::uint32_t kRosterRows[] = {0, 1, 2, 3, 4, 5, 6, 7};

void expect(bool condition, const char* message) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

}  // namespace

int main() {
    const std::uint64_t instance = 1;
    PieTerminalCell terminal{};
    PieTerminalCell* terminals[] = {&terminal};
    const std::uint32_t folded_slots[] = {3, 4};
    const std::uint8_t folded_flags[] = {0, PIE_RS_FLAG_RESET};
    const std::uint32_t fold_lens[] = {0, 0};
    const std::uint32_t buffer_slots[] = {7, 8};
    const std::uint32_t buffer_indptr[] = {0, 1, 2};
    PieStepDesc launch{};
    launch.roster_rows = {.ptr = kRosterRows, .len = 1};
    launch.terminal_cells = {.ptr = terminals, .len = 1};
    launch.rs_slot_ids = {.ptr = folded_slots, .len = 2};
    launch.rs_slot_flags = {.ptr = folded_flags, .len = 2};
    launch.rs_fold_lens = {.ptr = fold_lens, .len = 2};
    launch.rs_buffer_slot_ids = {.ptr = buffer_slots, .len = 2};
    launch.rs_buffer_slot_indptr = {.ptr = buffer_indptr, .len = 3};
    const std::uint32_t empty_qo[] = {0};
    launch.qo_indptr = {.ptr = empty_qo, .len = 1};
    expect(
        pie_native::abi::validate_step_desc(&launch, kTestRosterLen) == PIE_STATUS_OK,
        "empty wire geometry accepts a self-consistent deferred B=2 RS CSR");

    const std::uint32_t non_monotonic[] = {0, 2, 1};
    launch.rs_buffer_slot_indptr = {.ptr = non_monotonic, .len = 3};
    expect(
        pie_native::abi::validate_step_desc(&launch, kTestRosterLen) ==
            PIE_STATUS_INVALID_ARGUMENT,
        "deferred RS CSR still validates monotonicity");

    const std::uint64_t mixed_instances[] = {1, 2};
    PieTerminalCell mixed_terminal0{};
    PieTerminalCell mixed_terminal1{};
    PieTerminalCell* mixed_terminals[] = {
        &mixed_terminal0, &mixed_terminal1};
    const std::uint32_t mixed_qo[] = {0, 0};
    const std::uint32_t mixed_kv_ptr[] = {0, 0};
    const std::uint32_t mixed_last[] = {0};
    const std::uint32_t mixed_sample_ptr[] = {0, 0};
    const std::uint32_t mixed_program_rows[] = {0, 0, 1};
    const std::uint32_t mixed_slots[] = {10, 11, 12};
    const std::uint8_t mixed_flags[] = {
        PIE_RS_FLAG_FOLD, PIE_RS_FLAG_FOLD, 0};
    const std::uint32_t mixed_fold_lens[] = {1, 2, 0};
    const std::uint32_t mixed_buffer_slots[] = {20, 21, 22};
    const std::uint32_t mixed_buffer_ptr[] = {0, 1, 2, 3};
    PieStepDesc mixed_launch{};
    mixed_launch.roster_rows = {.ptr = kRosterRows, .len = 2};
    mixed_launch.terminal_cells = {
        .ptr = mixed_terminals, .len = 2};
    mixed_launch.qo_indptr = {.ptr = mixed_qo, .len = 2};
    mixed_launch.kv_page_indptr = {
        .ptr = mixed_kv_ptr, .len = 2};
    mixed_launch.kv_last_page_lens = {
        .ptr = mixed_last, .len = 1};
    mixed_launch.sampling_indptr = {
        .ptr = mixed_sample_ptr, .len = 2};
    mixed_launch.ptir_program_row_indptr = {
        .ptr = mixed_program_rows, .len = 3};
    mixed_launch.rs_slot_ids = {.ptr = mixed_slots, .len = 3};
    mixed_launch.rs_slot_flags = {.ptr = mixed_flags, .len = 3};
    mixed_launch.rs_fold_lens = {
        .ptr = mixed_fold_lens, .len = 3};
    mixed_launch.rs_buffer_slot_ids = {
        .ptr = mixed_buffer_slots, .len = 3};
    mixed_launch.rs_buffer_slot_indptr = {
        .ptr = mixed_buffer_ptr, .len = 4};
    expect(
        pie_native::abi::validate_step_desc(&mixed_launch, kTestRosterLen) ==
            PIE_STATUS_OK,
        "mixed wire+descriptor launch defers RS outer cardinality");
    mixed_launch.ptir_program_row_indptr = {};
    expect(
        pie_native::abi::validate_step_desc(&mixed_launch, kTestRosterLen) ==
            PIE_STATUS_INVALID_ARGUMENT,
        "wire-only launch still enforces exact RS outer cardinality");
    mixed_launch.ptir_program_row_indptr = {
        .ptr = mixed_program_rows, .len = 3};
    const std::uint8_t inconsistent_fold_flags[] = {
        0, PIE_RS_FLAG_FOLD, 0};
    mixed_launch.rs_slot_flags = {
        .ptr = inconsistent_fold_flags, .len = 3};
    expect(
        pie_native::abi::validate_step_desc(&mixed_launch, kTestRosterLen) ==
            PIE_STATUS_INVALID_ARGUMENT,
        "shared ABI rejects RS fold flags/lens disagreement");
    mixed_launch.rs_slot_flags = {.ptr = mixed_flags, .len = 3};

    const std::uint64_t hash = 9;
    const std::uint32_t row_attribution[] = {0, 0};
    pie_native::LaunchView view{};
    view.ptir_program_hashes =
        pie_native::slice_from_u64(&hash, 1);
    view.ptir_program_row_indptr =
        pie_native::slice_from_u32(row_attribution, 2);
    view.rs_slot_ids =
        pie_native::slice_from_u32(folded_slots, 2);
    view.rs_slot_flags =
        pie_native::slice_from_u8(folded_flags, 2);
    view.rs_buffer_slot_ids =
        pie_native::slice_from_u32(buffer_slots, 2);
    view.rs_buffer_slot_indptr =
        pie_native::slice_from_u32(buffer_indptr, 3);

    pie_native::ptir::ResolvedPrograms resolved;
    resolved.per_program.resize(1);
    resolved.is_device_geometry = {1};
    resolved.device_count = 1;
    auto& geometry = resolved.per_program[0];
    geometry.token_ids = {10, 11};
    geometry.position_ids = {0, 0};
    geometry.qo_indptr = {0, 1, 2};
    geometry.kv_page_indices = {0, 1};
    geometry.kv_page_indptr = {0, 1, 2};
    geometry.kv_last_page_lens = {1, 1};
    geometry.sampling_indices = {0, 0};
    geometry.sampling_indptr = {0, 1, 2};
    geometry.has_kv_family = true;

    pie_cuda_driver::pipeline::ComposedBatch composed;
    std::string error;
    expect(
        pie_cuda_driver::pipeline::compose_forward_batch(
            view, resolved, 16, composed, &error) &&
            composed.rs_slot_ids.size() == 2 &&
            composed.rs_buffer_slot_indptr ==
                std::vector<std::uint32_t>({0, 1, 2}),
        "CUDA composition validates folded and buffered RS against resolved B");

    const std::uint32_t short_indptr[] = {0, 2};
    view.rs_buffer_slot_indptr =
        pie_native::slice_from_u32(short_indptr, 2);
    expect(
        !pie_cuda_driver::pipeline::compose_forward_batch(
            view, resolved, 16, composed, &error),
        "CUDA composition rejects buffered RS whose outer count misses resolved B");

    const std::uint64_t mixed_hashes[] = {9, 10};
    const std::uint32_t mixed_wire_token[] = {33};
    const std::uint32_t mixed_wire_position[] = {0};
    const std::uint32_t mixed_wire_qo[] = {0, 1};
    const std::uint32_t mixed_wire_pages[] = {3};
    const std::uint32_t mixed_wire_page_ptr[] = {0, 1};
    const std::uint32_t mixed_wire_last[] = {1};
    const std::uint32_t mixed_wire_sample[] = {0};
    const std::uint32_t mixed_wire_sample_ptr[] = {0, 1};
    pie_native::LaunchView mixed_view{};
    mixed_view.ptir_program_hashes =
        pie_native::slice_from_u64(mixed_hashes, 2);
    mixed_view.ptir_program_row_indptr =
        pie_native::slice_from_u32(mixed_program_rows, 3);
    mixed_view.token_ids =
        pie_native::slice_from_u32(mixed_wire_token, 1);
    mixed_view.position_ids =
        pie_native::slice_from_u32(mixed_wire_position, 1);
    mixed_view.qo_indptr =
        pie_native::slice_from_u32(mixed_wire_qo, 2);
    mixed_view.kv_page_indices =
        pie_native::slice_from_u32(mixed_wire_pages, 1);
    mixed_view.kv_page_indptr =
        pie_native::slice_from_u32(mixed_wire_page_ptr, 2);
    mixed_view.kv_last_page_lens =
        pie_native::slice_from_u32(mixed_wire_last, 1);
    mixed_view.sampling_indices =
        pie_native::slice_from_u32(mixed_wire_sample, 1);
    mixed_view.sampling_indptr =
        pie_native::slice_from_u32(mixed_wire_sample_ptr, 2);
    mixed_view.rs_slot_ids =
        pie_native::slice_from_u32(mixed_slots, 3);
    mixed_view.rs_slot_flags =
        pie_native::slice_from_u8(mixed_flags, 3);
    mixed_view.rs_fold_lens =
        pie_native::slice_from_u32(mixed_fold_lens, 3);
    mixed_view.rs_buffer_slot_ids =
        pie_native::slice_from_u32(mixed_buffer_slots, 3);
    mixed_view.rs_buffer_slot_indptr =
        pie_native::slice_from_u32(mixed_buffer_ptr, 4);

    pie_native::ptir::ResolvedPrograms mixed_resolved;
    mixed_resolved.per_program.resize(2);
    mixed_resolved.is_device_geometry = {1, 0};
    mixed_resolved.device_count = 1;
    auto& mixed_geometry = mixed_resolved.per_program[0];
    mixed_geometry.token_ids = {41, 42};
    mixed_geometry.position_ids = {0, 0};
    mixed_geometry.qo_indptr = {0, 1, 2};
    mixed_geometry.kv_page_indices = {0, 1};
    mixed_geometry.kv_page_indptr = {0, 1, 2};
    mixed_geometry.kv_last_page_lens = {1, 1};
    mixed_geometry.sampling_indices = {0, 0};
    mixed_geometry.sampling_indptr = {0, 1, 2};
    mixed_geometry.has_kv_family = true;
    expect(
        pie_cuda_driver::pipeline::compose_forward_batch(
            mixed_view, mixed_resolved, 16, composed, &error) &&
            composed.rs_slot_ids ==
                std::vector<std::uint32_t>({12, 10, 11}) &&
            composed.rs_fold_lens ==
                std::vector<std::uint32_t>({0, 1, 2}) &&
            composed.rs_buffer_slot_ids ==
                std::vector<std::uint32_t>({22, 20, 21}) &&
            composed.rs_buffer_slot_indptr ==
                std::vector<std::uint32_t>({0, 1, 2, 3}),
        "mixed composition reorders ids, flags, folds, and buffer CSR together");

    const std::uint32_t fold_plan_slots[] = {1, 2};
    const std::uint8_t fold_plan_flags[] = {
        PIE_RS_FLAG_FOLD, PIE_RS_FLAG_FOLD};
    const std::uint32_t fold_plan_lens[] = {3, 1};
    const std::uint32_t fold_plan_buffers[] = {5, 6, 7};
    const std::uint32_t fold_plan_buffer_ptr[] = {0, 2, 3};
    const std::uint32_t fold_plan_qo[] = {0, 0, 0};
    pie_cuda_driver::pipeline::RsExecutionPlan fold_plan;
    expect(
        pie_cuda_driver::pipeline::plan_rs_execution(
            fold_plan_slots,
            fold_plan_flags,
            fold_plan_lens,
            fold_plan_buffers,
            fold_plan_buffer_ptr,
            fold_plan_qo,
            true,
            true,
            2,
            fold_plan,
            &error) &&
            fold_plan.mode ==
                pie_cuda_driver::RsExecutionMode::BufferFold &&
            fold_plan.fold_qo_indptr ==
                std::vector<std::uint32_t>({0, 3, 4}) &&
            fold_plan.fold_tokens == 4,
        "fold execution plan uses per-request commit lengths");
    const std::uint8_t mixed_mode_flags[] = {PIE_RS_FLAG_FOLD, 0};
    const std::uint32_t mixed_mode_lens[] = {3, 0};
    expect(
        !pie_cuda_driver::pipeline::plan_rs_execution(
            fold_plan_slots,
            mixed_mode_flags,
            mixed_mode_lens,
            fold_plan_buffers,
            fold_plan_buffer_ptr,
            fold_plan_qo,
            true,
            true,
            2,
            fold_plan,
            &error),
        "fold preflight rejects mixed state paths before mutation");
    const std::uint8_t write_flags[] = {0, PIE_RS_FLAG_RESET};
    const std::uint32_t write_lens[] = {0, 0};
    const std::uint32_t write_qo[] = {0, 2, 3};
    const std::uint32_t write_buffers[] = {8, 9};
    const std::uint32_t write_buffer_ptr[] = {0, 1, 2};
    expect(
        pie_cuda_driver::pipeline::plan_rs_execution(
            fold_plan_slots,
            write_flags,
            write_lens,
            write_buffers,
            write_buffer_ptr,
            write_qo,
            true,
            true,
            2,
            fold_plan,
            &error) &&
            fold_plan.mode ==
                pie_cuda_driver::RsExecutionMode::BufferWrite,
        "buffer scatter preflight selects the non-folding state path");
    expect(
        !pie_cuda_driver::pipeline::plan_rs_execution(
            fold_plan_slots,
            write_flags,
            write_lens,
            write_buffers,
            write_buffer_ptr,
            write_qo,
            true,
            false,
            0,
            fold_plan,
            &error),
        "buffered RS is rejected before mutation on unsupported models");

    const pie_native::ptir::StructuredMaskDescriptor unset_masks[2]{};
    const pie_native::ptir::StructuredMaskDescriptor explicit_masks[2] = {
        {pie_native::ptir::StructuredMaskKind::SlidingWindow, 8, 0, 4},
        {pie_native::ptir::StructuredMaskKind::SlidingWindow, 8, 0, 4},
    };
    const pie_native::ptir::StructuredMaskDescriptor mixed_masks[2] = {
        {},
        {pie_native::ptir::StructuredMaskKind::SlidingWindow, 8, 0, 4},
    };
    expect(
        pie_cuda_driver::pipeline::structured_mask_coverage(unset_masks) ==
                pie_cuda_driver::pipeline::StructuredMaskCoverage::None &&
            pie_cuda_driver::pipeline::structured_mask_coverage(
                explicit_masks) ==
                pie_cuda_driver::pipeline::StructuredMaskCoverage::Complete &&
            pie_cuda_driver::pipeline::structured_mask_coverage(mixed_masks) ==
                pie_cuda_driver::pipeline::StructuredMaskCoverage::Mixed,
        "wire rows preserve model-native windows unless PTIR masks are explicit");

    std::printf("rs_descriptor_validation_test: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
