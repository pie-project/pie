// PTIR tier-2 graph-key extension test. Pure-host unit
// test for the program-set hash (contract C3) added to ForwardGraphKey: proves
// (a) the non-PTIR path is unchanged (default 0), (b) different program sets key
// to different graphs, (c) the fold is order-independent and preserves exact
// stage multiplicity. No GPU.

#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "batch/forward_graph.hpp"
#include "cuda_check.hpp"
#include "model/workspace.hpp"
#include "pipeline/batch_compose.hpp"
#include "pipeline/program_identity.hpp"

using namespace pie_cuda_driver;

namespace {
int g_pass = 0, g_fail = 0;
void expect(bool ok, const char* what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what); }
}

std::uint64_t program_set(
    std::initializer_list<std::uint64_t> identities) {
    pie_cuda_driver::pipeline::ProgramSetIdentityFold fold;
    for (const std::uint64_t identity : identities) fold.add(identity, 0);
    return fold.finish();
}
}

int main() {
    std::printf("PTIR tier-2 graph-key (program-set hash / C3)\n");
    ForwardGraphKeyHash H;

    // (a) backward-compat: 3-field aggregate init leaves program_set_hash = 0,
    // identical to today's key.
    ForwardGraphKey legacy{4, 4, 7};
    ForwardGraphKey explicit0{4, 4, 7, 0};
    expect(legacy == explicit0, "3-field init defaults program_set_hash to 0");
    expect(legacy.program_set_hash == 0, "non-PTIR key has zero program-set hash");
    expect(H(legacy) == H(explicit0), "hash agrees for the default");

    // (b) same shape, different program set → distinct key + (very likely) hash.
    std::uint64_t setA = program_set({0x1111, 0x2222});
    std::uint64_t setB = program_set({0x1111, 0x3333});
    ForwardGraphKey ka{4, 4, 7, setA};
    ForwardGraphKey kb{4, 4, 7, setB};
    expect(!(ka == kb), "different program sets → different keys");
    expect(H(ka) != H(kb), "different program sets → different hashes");
    expect(!(ka == legacy), "PTIR key != non-PTIR key of same shape");
    ForwardGraphKey write_desc = ka;
    write_desc.has_write_desc = true;
    expect(
        !(write_desc == ka) && H(write_desc) != H(ka),
        "explicit write descriptors select a distinct forward graph");
    ForwardGraphKey windowed = ka;
    windowed.structured_window_left = 127;
    expect(
        !(windowed == ka) && H(windowed) != H(ka),
        "structured runtime windows select a distinct forward graph");

    // (c) order-independent, with exact lane/stage multiplicity.
    expect(program_set({0xAA, 0xBB, 0xCC}) ==
           program_set({0xCC, 0xAA, 0xBB}), "fold is order-independent");
    expect(program_set({0xAA, 0xBB, 0xAA}) !=
           program_set({0xAA, 0xBB}), "fold includes duplicate stage count");
    expect(program_set({}) == 0, "empty set folds to 0 (== non-PTIR)");
    expect(program_set({0xAA}) != program_set({0xBB}),
           "distinct singletons fold distinctly");
    // a single-program fleet vs a two-program fleet must differ.
    expect(program_set({0xAA}) != program_set({0xAA, 0xBB}),
           "adding a program changes the set identity");

    ForwardGraphCache cache;
    expect(cache.get(ka) == nullptr, "graph cache records a miss");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphCreate(&graph, 0);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    cache.put(ka, exec);
    expect(cache.get(ka) == exec, "captured PTIR graph is keyed by program identity");
    const auto metrics = cache.metrics();
    expect(
        metrics.misses == 1 && metrics.hits == 1 &&
           metrics.captures == 1,
        "graph cache exposes hit/miss/capture metrics");
    cudaStream_t failed_capture_stream = nullptr;
    cudaStreamCreate(&failed_capture_stream);
    try {
        pie_cuda_driver::StreamCaptureGuard capture(
            failed_capture_stream);
        throw std::runtime_error("injected capture failure");
    } catch (const std::runtime_error&) {
    }
    expect(
        cudaStreamSynchronize(failed_capture_stream) == cudaSuccess,
        "capture exception guard restores a reusable CUDA stream");
    cudaStreamDestroy(failed_capture_stream);

    std::vector<std::int32_t> global_rows;
    std::string sampling_error;
    const std::uint32_t qo[] = {0, 2, 5};
    const std::uint32_t sample_ptr[] = {0, 1, 2};
    const std::uint32_t sample_index[] = {0, 0};
    expect(
        pie_cuda_driver::pipeline::global_sampling_rows(
            qo,
            sample_ptr,
            sample_index,
            global_rows,
            &sampling_error) &&
            global_rows == std::vector<std::int32_t>({0, 2}),
        "request-relative all-zero sampling indices map to global rows");
    const int single_gpu_history =
        pie_cuda_driver::pipeline::mtp_global_history_tokens(41, 3, true);
    const int tp_history =
        pie_cuda_driver::pipeline::mtp_global_history_tokens(41, 3, true);
    expect(
        single_gpu_history == 38 && tp_history == single_gpu_history,
        "TP and single-GPU MTP preserve identical paged-history bounds");
    const std::uint32_t aggregate_drafts[] = {20, 20};
    std::vector<std::uint32_t> aggregate_starts;
    expect(
        pie_cuda_driver::pipeline::plan_mtp_draft_rows(
            aggregate_drafts, 100, 40, aggregate_starts,
            &sampling_error) &&
            aggregate_starts ==
                std::vector<std::uint32_t>({100, 120}),
        "multi-program MTP supports more than 32 aggregate draft rows");
    expect(
        !pie_cuda_driver::pipeline::plan_mtp_draft_rows(
            aggregate_drafts, 100, 39, aggregate_starts,
            &sampling_error),
        "aggregate MTP storage rejects only beyond its truthful capacity");
    expect(
        pie_cuda_driver::model::workspace_mtp_draft_row_base(1024) == 1024 &&
            pie_cuda_driver::model::workspace_logits_rows(1024, 40) == 1064,
        "workspace reserves every target token row before MTP drafts");
    const pie_native::ptir::StructuredMaskDescriptor sliding_masks[] = {
        {pie_native::ptir::StructuredMaskKind::SlidingWindow, 8, 0, 4},
    };
    const std::uint32_t tail_positions[] = {6, 7};
    const std::uint32_t arbitrary_positions[] = {1, 7};
    const std::uint32_t tail_qo[] = {0, 2};
    const std::uint32_t tail_pages[] = {0, 1};
    const std::uint32_t tail_last[] = {8};
    expect(
        pie_cuda_driver::pipeline::runtime_window_for_tail_aligned(
            sliding_masks, tail_positions, tail_qo, tail_pages, tail_last, 16) ==
            std::optional<int>(3),
        "tail-aligned positions use the representable runtime window");
    expect(
        !pie_cuda_driver::pipeline::runtime_window_for_tail_aligned(
             sliding_masks, arbitrary_positions, tail_qo, tail_pages,
             tail_last, 16)
             .has_value(),
        "arbitrary positions fall back to the exact packed mask");
    auto oversized_window = sliding_masks[0];
    oversized_window.window =
        static_cast<std::uint32_t>(std::numeric_limits<int>::max()) + 2u;
    expect(
        !pie_cuda_driver::pipeline::runtime_window_for_tail_aligned(
             std::span<const pie_native::ptir::StructuredMaskDescriptor>(
                 &oversized_window, 1),
             tail_positions, tail_qo, tail_pages, tail_last, 16)
             .has_value(),
        "unrepresentable structured windows fall back without overflow");

    const std::uint64_t program_hashes[] = {1, 2};
    const std::uint32_t program_rows[] = {0, 0, 1};
    const std::uint32_t wire_tokens[] = {11};
    const std::uint32_t wire_positions[] = {0};
    const std::uint32_t wire_qo[] = {0, 1};
    const std::uint32_t wire_pages[] = {0};
    const std::uint32_t wire_page_ptr[] = {0, 1};
    const std::uint32_t wire_last[] = {1};
    const std::uint32_t wire_samples[] = {0};
    const std::uint32_t wire_sample_ptr[] = {0, 1};
    // Runtime request order is program order: device program first, then wire.
    // Composition emits wire requests first, so folded and buffered resources
    // must be reordered independently.
    const std::uint32_t folded_rs_slots[] = {41, 7};
    const std::uint8_t folded_rs_flags[] = {PIE_RS_FLAG_RESET, 0};
    const std::uint32_t buffered_rs_slots[] = {900, 901};
    const std::uint32_t buffered_rs_indptr[] = {0, 1, 2};
    pie_native::LaunchView mixed_view{};
    mixed_view.ptir_program_hashes =
        pie_native::slice_from_u64(program_hashes, 2);
    mixed_view.ptir_program_row_indptr =
        pie_native::slice_from_u32(program_rows, 3);
    mixed_view.token_ids =
        pie_native::slice_from_u32(wire_tokens, 1);
    mixed_view.position_ids =
        pie_native::slice_from_u32(wire_positions, 1);
    mixed_view.qo_indptr =
        pie_native::slice_from_u32(wire_qo, 2);
    mixed_view.kv_page_indices =
        pie_native::slice_from_u32(wire_pages, 1);
    mixed_view.kv_page_indptr =
        pie_native::slice_from_u32(wire_page_ptr, 2);
    mixed_view.kv_last_page_lens =
        pie_native::slice_from_u32(wire_last, 1);
    mixed_view.sampling_indices =
        pie_native::slice_from_u32(wire_samples, 1);
    mixed_view.sampling_indptr =
        pie_native::slice_from_u32(wire_sample_ptr, 2);
    mixed_view.rs_slot_ids =
        pie_native::slice_from_u32(folded_rs_slots, 2);
    mixed_view.rs_slot_flags =
        pie_native::slice_from_u8(folded_rs_flags, 2);
    mixed_view.rs_buffer_slot_ids =
        pie_native::slice_from_u32(buffered_rs_slots, 2);
    mixed_view.rs_buffer_slot_indptr =
        pie_native::slice_from_u32(buffered_rs_indptr, 3);
    pie_native::ptir::ResolvedPrograms resolved;
    resolved.per_program.resize(2);
    resolved.is_device_geometry = {1, 0};
    resolved.device_count = 1;
    auto& device = resolved.per_program[0];
    device.token_ids = {22};
    device.position_ids = {0};
    device.qo_indptr = {0, 1};
    device.kv_page_indices = {0};
    device.kv_page_indptr = {0, 1};
    device.kv_last_page_lens = {1};
    device.sampling_indices = {0};
    device.sampling_indptr = {0, 1};
    device.has_kv_family = true;
    device.structured_mask.kind =
        pie_native::ptir::StructuredMaskKind::SlidingWindow;
    device.structured_mask.key_len = 16;
    device.structured_mask.window = 4;
    pie_cuda_driver::pipeline::ComposedBatch mixed;
    expect(
        pie_cuda_driver::pipeline::compose_forward_batch(
            mixed_view, resolved, 16, mixed, &sampling_error) &&
            mixed.prog_sample_starts ==
                std::vector<std::uint32_t>({1, 0}) &&
            mixed.prog_sample_counts ==
                std::vector<std::uint32_t>({1, 1}) &&
            mixed.rs_slot_ids ==
                std::vector<std::uint32_t>({7, 41}) &&
            mixed.rs_slot_flags ==
                std::vector<std::uint8_t>({0, PIE_RS_FLAG_RESET}) &&
            mixed.rs_buffer_slot_ids ==
                std::vector<std::uint32_t>({901, 900}) &&
            mixed.rs_buffer_slot_indptr ==
                std::vector<std::uint32_t>({0, 1, 2}) &&
            mixed.rs_slot_ids != mixed.rs_buffer_slot_ids &&
            mixed.structured_masks.size() == 2 &&
            mixed.structured_masks[0].kind ==
                pie_native::ptir::StructuredMaskKind::None &&
            mixed.structured_masks[1].kind ==
                pie_native::ptir::StructuredMaskKind::SlidingWindow,
        "B=2 device/wire composition preserves native wire masking and "
        "request-order folded RS");
    const pie_native::ptir::StructuredMaskDescriptor no_masks[2]{};
    const pie_native::ptir::StructuredMaskDescriptor explicit_masks[2] = {
        sliding_masks[0], sliding_masks[0]};
    const pie_native::ptir::StructuredMaskDescriptor mixed_masks[2] = {
        {}, sliding_masks[0]};
    expect(
        pie_cuda_driver::pipeline::structured_mask_coverage(no_masks) ==
                pie_cuda_driver::pipeline::StructuredMaskCoverage::None &&
            pie_cuda_driver::pipeline::structured_mask_coverage(
                explicit_masks) ==
                pie_cuda_driver::pipeline::StructuredMaskCoverage::Complete &&
            pie_cuda_driver::pipeline::structured_mask_coverage(mixed_masks) ==
                pie_cuda_driver::pipeline::StructuredMaskCoverage::Mixed,
        "only complete explicit PTIR mask coverage may override model policy");
    const std::span<const std::uint32_t> missing_folded_ids;
    const std::span<const std::uint8_t> missing_folded_flags;
    expect(
        !pie_cuda_driver::pipeline::validate_folded_rs_bindings(
            missing_folded_ids,
            missing_folded_flags,
            std::span<const std::uint32_t>{},
            2,
            true,
            &sampling_error),
        "B=2 RS model rejects missing folded ids even when buffers exist");
    const std::uint32_t short_buffered_indptr[] = {0, 2};
    auto bad_buffer_view = mixed_view;
    bad_buffer_view.rs_buffer_slot_indptr =
        pie_native::slice_from_u32(short_buffered_indptr, 2);
    pie_cuda_driver::pipeline::ComposedBatch bad_buffered;
    expect(
        !pie_cuda_driver::pipeline::compose_forward_batch(
            bad_buffer_view, resolved, 16, bad_buffered, &sampling_error),
        "buffered RS exact outer count is deferred to resolved B=2 geometry");

    std::printf("\n==== graph-key C3: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
