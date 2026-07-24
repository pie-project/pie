#include "batch/frame.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "batch/brle.hpp"
#include "batch/fire_timing.hpp"
#include "batch/forward.hpp"
#include "batch/tp.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "kernels/argmax.hpp"
#include "kernels/graph_pad.hpp"
#include "kernels/pack_dense_mask.hpp"
#include "model/loaded_model.hpp"
#include "model/stage_hooks.hpp"
#include "store/kv_cache.hpp"
#include "store/recurrent_state_cache.hpp"
#include "model/workspace.hpp"
#include "ops/gemm.hpp"
#include "pipeline/batch_compose.hpp"
#include "pipeline/dispatch.hpp"

namespace pie_cuda_driver {

namespace {

int tensor_rows(const DeviceTensor& t) {
    if (t.shape().empty()) return 0;
    return static_cast<int>(t.shape()[0]);
}

struct MtpDraftWork {
    std::size_t program = 0;
    std::uint32_t drafts = 0;
    std::uint32_t request = 0;
    std::uint32_t anchor_row = 0;
    std::uint32_t seed_row = 0;
    std::uint32_t source_position = 0;
    std::uint32_t draft_start = 0;
};

struct MtpDraftPlan {
    std::vector<MtpDraftWork> work;
    std::vector<std::uint32_t> draft_starts;
    std::size_t total_drafts = 0;
    std::uint32_t vocab = 0;
};

MtpDraftPlan preflight_mtp_draft_logits(
    BatchEngine& engine,
    const pipeline::ComposedBatch& composed,
    std::span<const std::int32_t> sampled_model_rows,
    std::span<const std::uint32_t> draft_counts) {
    MtpDraftPlan plan;
    if (engine.ws.mtp_draft_row_base < 0 ||
        engine.ws.mtp_draft_row_capacity < 0) {
        throw std::runtime_error("MTP workspace layout is negative");
    }
    const std::uint32_t draft_base =
        static_cast<std::uint32_t>(engine.ws.mtp_draft_row_base);
    std::string layout_error;
    if (!pipeline::plan_mtp_draft_rows(
            draft_counts,
            draft_base,
            static_cast<std::uint32_t>(
                std::max(0, engine.ws.mtp_draft_row_capacity)),
            plan.draft_starts,
            &layout_error,
            model::Workspace::kMtpDraftRowsPerProgram)) {
        throw std::runtime_error(layout_error);
    }
    for (std::size_t program = 0; program < draft_counts.size(); ++program) {
        const std::uint32_t drafts = draft_counts[program];
        if (drafts == 0) continue;
        if (program >= composed.prog_request_starts.size() ||
            program >= composed.prog_request_counts.size() ||
            composed.prog_request_counts[program] != 1) {
            throw std::runtime_error(
                "MtpLogits requires one attributed forward request per program");
        }
        if (drafts > static_cast<std::uint32_t>(
                std::max(0, engine.system_drafter.max_drafts))) {
            throw std::runtime_error(
                "MtpLogits draft-row requirement exceeds the production layout");
        }
        const std::uint32_t request =
            composed.prog_request_starts[program];
        if (request + 1 >= composed.qo_indptr.size() ||
            composed.qo_indptr[request + 1] <=
                composed.qo_indptr[request]) {
            throw std::runtime_error(
                "MtpLogits request has no anchor hidden row");
        }
        if (program >= composed.prog_sample_starts.size() ||
            program >= composed.prog_sample_counts.size() ||
            composed.prog_sample_counts[program] == 0) {
            throw std::runtime_error(
                "MtpLogits request has no target seed-logit row");
        }
        const std::uint32_t sample =
            composed.prog_sample_starts[program] +
            composed.prog_sample_counts[program] - 1;
        if (sample >= sampled_model_rows.size() ||
            sampled_model_rows[sample] < 0) {
            throw std::runtime_error(
                "MtpLogits target seed row is outside the model output");
        }
        const std::uint32_t anchor =
            composed.qo_indptr[request + 1] - 1;
        if (anchor >= composed.position_ids.size()) {
            throw std::runtime_error(
                "MtpLogits anchor position is outside the forward geometry");
        }
        if (sampled_model_rows[sample] >= tensor_rows(engine.ws.logits)) {
            throw std::runtime_error(
                "MtpLogits target seed row exceeds logits storage");
        }
        const std::uint64_t last_position =
            static_cast<std::uint64_t>(composed.position_ids[anchor]) +
            static_cast<std::uint64_t>(
                std::max(0, engine.system_drafter.draft_position_offset)) +
            drafts - 1;
        if (last_position >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            request >
                static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            anchor >
                static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("MTP scalar staging exceeds i32");
        }
        plan.work.push_back({
            program,
            drafts,
            request,
            anchor,
            static_cast<std::uint32_t>(sampled_model_rows[sample]),
            composed.position_ids[anchor],
            plan.draft_starts[program],
        });
    }
    if (!plan.work.empty() && !engine.system_drafter.draft_step) {
        throw std::runtime_error(
            "MtpLogits is valid for this program but the model has no native draft head");
    }
    if (plan.work.empty()) return plan;
    for (const MtpDraftWork& item : plan.work) {
        if (item.drafts >
            std::numeric_limits<std::size_t>::max() - plan.total_drafts) {
            throw std::runtime_error("aggregate MTP scalar demand overflows");
        }
        plan.total_drafts += item.drafts;
    }
    const std::size_t scalar_capacity = std::min({
        engine.inputs.mtp_positions_host.size(),
        engine.inputs.mtp_hidden_rows_host.size(),
        engine.inputs.mtp_request_ids_host.size(),
    });
    if (plan.total_drafts > scalar_capacity) {
        throw std::runtime_error(
            "aggregate MTP scalar demand exceeds persistent staging capacity");
    }
    const int model_vocab = engine.loaded_model.hf_config().vocab_size;
    if (model_vocab <= 0) {
        throw std::runtime_error("MTP model vocabulary is invalid");
    }
    plan.vocab = static_cast<std::uint32_t>(model_vocab);
    if (!plan.work.empty() &&
        (plan.vocab == 0 ||
         engine.ws.mtp_row0_save.numel() < plan.vocab ||
         engine.inputs.tokens.size() < 1 ||
         engine.inputs.positions.size() < 1 ||
         engine.inputs.sample_idx.size() < 1 ||
         engine.inputs.mtp_request_ids.size() < 1)) {
        throw std::runtime_error(
            "MTP persistent staging capacity is incomplete");
    }
    const std::uint32_t logits_rows = static_cast<std::uint32_t>(
        std::max(0, tensor_rows(engine.ws.logits)));
    for (const MtpDraftWork& item : plan.work) {
        if (item.draft_start > logits_rows ||
            item.drafts > logits_rows - item.draft_start) {
            throw std::runtime_error(
                "MTP draft rows exceed physical logits storage");
        }
    }
    // Every draft step writes its one-row logits result to row zero and its
    // hidden result to ws.y row zero. Preserve target row zero and process a
    // request anchored at hidden row zero first, before another chain can
    // overwrite that anchor.
    std::stable_sort(
        plan.work.begin(), plan.work.end(),
        [](const MtpDraftWork& left, const MtpDraftWork& right) {
            return (left.anchor_row == 0) > (right.anchor_row == 0);
        });
    return plan;
}

void enqueue_mtp_draft_logits(
    BatchEngine& engine,
    const MtpDraftPlan& plan) {
    if (plan.work.empty()) return;
    auto* logits = static_cast<std::uint16_t*>(engine.ws.logits.data());
    const std::uint32_t vocab = plan.vocab;
    cudaStream_t stream = engine.cublas.stream();
    static thread_local cudaEvent_t scalar_copies_done = nullptr;
    if (scalar_copies_done == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(
            &scalar_copies_done, cudaEventDisableTiming));
    } else {
        CUDA_CHECK(cudaEventSynchronize(scalar_copies_done));
    }
    std::size_t scalar_index = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        engine.ws.mtp_row0_save.data(),
        logits,
        static_cast<std::size_t>(vocab) * sizeof(std::uint16_t),
        cudaMemcpyDeviceToDevice,
        stream));
    for (const MtpDraftWork& item : plan.work) {
        kernels::launch_argmax_bf16(
            logits + static_cast<std::size_t>(item.seed_row) * vocab,
            reinterpret_cast<std::int32_t*>(engine.inputs.sampled.data()),
            1,
            vocab,
            stream);
        for (std::uint32_t draft = 0; draft < item.drafts; ++draft) {
            const std::int32_t position = static_cast<std::int32_t>(
                item.source_position +
                std::max(0, engine.system_drafter.draft_position_offset) +
                draft);
            const std::int32_t hidden_row = draft == 0
                ? static_cast<std::int32_t>(item.anchor_row)
                : 0;
            const std::int32_t request =
                static_cast<std::int32_t>(item.request);
            engine.inputs.mtp_positions_host[scalar_index] = position;
            engine.inputs.mtp_hidden_rows_host[scalar_index] = hidden_row;
            engine.inputs.mtp_request_ids_host[scalar_index] = request;
            CUDA_CHECK(cudaMemcpyAsync(
                engine.inputs.tokens.data(),
                engine.inputs.sampled.data(),
                sizeof(std::int32_t),
                cudaMemcpyDeviceToDevice,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                engine.inputs.positions.data(),
                engine.inputs.mtp_positions_host.data() + scalar_index,
                sizeof(position),
                cudaMemcpyHostToDevice,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                engine.inputs.sample_idx.data(),
                engine.inputs.mtp_hidden_rows_host.data() + scalar_index,
                sizeof(hidden_row),
                cudaMemcpyHostToDevice,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                engine.inputs.mtp_request_ids.data(),
                engine.inputs.mtp_request_ids_host.data() + scalar_index,
                sizeof(request),
                cudaMemcpyHostToDevice,
                stream));
            const bool prefix_global =
                engine.system_drafter.draft_global_cache_uses_prefix_position;
            const int max_global_tokens =
                pipeline::mtp_global_history_tokens(
                    position, draft, prefix_global);
            if (engine.tp_comm != nullptr) {
                tp_cpu_gate_notify(engine.tp_cpu_gate_key);
                tp_broadcast_mtp_step(
                    *engine.tp_comm,
                    engine.inputs,
                    1,
                    static_cast<int>(draft),
                    max_global_tokens,
                    stream);
            }
            engine.system_drafter.draft_step(
                engine.ws,
                engine.kv_cache,
                engine.cublas,
                reinterpret_cast<const std::int32_t*>(
                    engine.inputs.tokens.data()),
                reinterpret_cast<const std::int32_t*>(
                    engine.inputs.positions.data()),
                engine.inputs.sample_idx.data(),
                engine.inputs.mtp_request_ids.data(),
                engine.inputs.kv_page_indices.data(),
                engine.inputs.kv_page_indptr.data(),
                engine.inputs.kv_last_page_lens.data(),
                nullptr,
                1,
                static_cast<int>(draft),
                max_global_tokens);
            kernels::launch_argmax_bf16(
                logits,
                reinterpret_cast<std::int32_t*>(
                    engine.inputs.sampled.data()),
                1,
                vocab,
                stream);
            CUDA_CHECK(cudaMemcpyAsync(
                logits +
                    static_cast<std::size_t>(item.draft_start + draft) *
                        vocab,
                logits,
                static_cast<std::size_t>(vocab) * sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice,
                stream));
            ++scalar_index;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(
        logits,
        engine.ws.mtp_row0_save.data(),
        static_cast<std::size_t>(vocab) * sizeof(std::uint16_t),
        cudaMemcpyDeviceToDevice,
        stream));
    CUDA_CHECK(cudaEventRecord(scalar_copies_done, stream));
}

// Correlated host-stage timing across the three phase calls (fire-timing
// diagnostic; plain members set as the phases run — the emission fires when
// the PreparedStep dies, which also covers exception unwinds). Leg
// semantics under the frame split: the prepare legs (begin/resolve/compose/
// wire_finalize/plan) span FramePrepare; `h2d_prepare_us` spans StepEnqueue
// entry → parameter-block commits + attention-plan hook; `host_total_us` is
// the sum of the two exclusive phase spans (the wall gap between them holds
// sibling steps' work by design).
struct StepTiming {
    fire_timing::Clock::time_point t0{};
    fire_timing::Clock::time_point begin_end{};
    pie_cuda_driver::pipeline::StagedLaunch::BeginBreakdown begin_breakdown{};
    fire_timing::Clock::time_point resolve_end{};
    fire_timing::Clock::time_point compose_end{};
    fire_timing::Clock::time_point wire_parse_end{};
    fire_timing::Clock::time_point prepare_end{};
    fire_timing::Clock::time_point enqueue_start{};
    fire_timing::Clock::time_point h2d_end{};
    fire_timing::Clock::time_point forward_enqueue_end{};
    fire_timing::Clock::time_point settlement_enqueue_end{};
    pipeline::Dispatch::FinishBreakdown finish_breakdown{};
    std::int64_t finish_groups = -1;
    std::int64_t finish_grouped_lanes = -1;
    std::int64_t finish_body_launches = -1;
    std::int64_t finish_shared_exclusions = -1;
    int requests = 0;
    int tokens = 0;
    int images = 0;
    std::uint64_t wave_id = 0;
    std::size_t fire_count = 0;
    std::uint64_t membership_hash = 0;
    int uncaught = 0;
    bool enabled = false;

    ~StepTiming() noexcept {
        if (!enabled) return;
        try {
            const auto now = fire_timing::Clock::now();
            std::ostringstream output;
            output << R"({"schema":1,"source":"cuda","event":"cuda_submit")"
                   << R"(,"wave_id":)"
                   << wave_id
                   << R"(,"fire_count":)"
                   << fire_count
                   << R"(,"membership_hash":)"
                   << membership_hash
                   << R"(,"status":")"
                   << (std::uncaught_exceptions() > uncaught
                           ? "error"
                           : "enqueued")
                   << R"(","requests":)" << requests
                   << R"(,"tokens":)" << tokens
                   << R"(,"images":)" << images;
            if (wire_parse_end != fire_timing::Clock::time_point{} &&
                begin_end != fire_timing::Clock::time_point{}) {
                output << R"(,"wire_compose_us":)"
                       << fire_timing::duration_us(begin_end, wire_parse_end);
            }
            if (begin_end != fire_timing::Clock::time_point{}) {
                output << R"(,"begin_us":)"
                       << fire_timing::duration_us(t0, begin_end);
            }
            if (begin_breakdown.pass_a_us >= 0) {
                output << R"(,"begin_prologue_us":)"
                       << begin_breakdown.prologue_us
                       << R"(,"begin_pass_a_us":)"
                       << begin_breakdown.pass_a_us
                       << R"(,"begin_tickets_us":)"
                       << begin_breakdown.tickets_us
                       << R"(,"begin_pass_c_us":)"
                       << begin_breakdown.pass_c_us
                       << R"(,"begin_pull_validate_us":)"
                       << begin_breakdown.pull_validate_us;
            }
            if (begin_end != fire_timing::Clock::time_point{} &&
                resolve_end != fire_timing::Clock::time_point{}) {
                output << R"(,"descriptor_resolve_us":)"
                       << fire_timing::duration_us(
                              begin_end, resolve_end);
            }
            if (resolve_end != fire_timing::Clock::time_point{} &&
                compose_end != fire_timing::Clock::time_point{}) {
                output << R"(,"geometry_compose_us":)"
                       << fire_timing::duration_us(
                              resolve_end, compose_end);
            }
            if (compose_end != fire_timing::Clock::time_point{} &&
                wire_parse_end != fire_timing::Clock::time_point{}) {
                output << R"(,"wire_finalize_us":)"
                       << fire_timing::duration_us(
                              compose_end, wire_parse_end);
            }
            if (wire_parse_end != fire_timing::Clock::time_point{} &&
                prepare_end != fire_timing::Clock::time_point{}) {
                output << R"(,"plan_us":)"
                       << fire_timing::duration_us(
                              wire_parse_end, prepare_end);
            }
            if (enqueue_start != fire_timing::Clock::time_point{} &&
                h2d_end != fire_timing::Clock::time_point{}) {
                output << R"(,"h2d_prepare_us":)"
                       << fire_timing::duration_us(enqueue_start, h2d_end);
            }
            if (h2d_end != fire_timing::Clock::time_point{} &&
                forward_enqueue_end !=
                fire_timing::Clock::time_point{}) {
                output << R"(,"forward_enqueue_us":)"
                       << fire_timing::duration_us(
                              h2d_end, forward_enqueue_end);
            }
            if (forward_enqueue_end !=
                    fire_timing::Clock::time_point{} &&
                settlement_enqueue_end !=
                fire_timing::Clock::time_point{}) {
                output << R"(,"settlement_enqueue_us":)"
                       << fire_timing::duration_us(
                              forward_enqueue_end,
                              settlement_enqueue_end);
            }
            if (finish_breakdown.epilogue_us >= 0) {
                output << R"(,"finish_epilogue_us":)"
                       << finish_breakdown.epilogue_us
                       << R"(,"finish_settle_lock_us":)"
                       << finish_breakdown.settle_lock_us
                       << R"(,"finish_settle_prep_us":)"
                       << finish_breakdown.settle_prep_us;
            }
            if (finish_breakdown.epilogue_assemble_us >= 0) {
                output << R"(,"finish_epilogue_assemble_us":)"
                       << finish_breakdown.epilogue_assemble_us
                       << R"(,"finish_epilogue_group_us":)"
                       << finish_breakdown.epilogue_group_us
                       << R"(,"finish_epilogue_execute_us":)"
                       << finish_breakdown.epilogue_execute_us;
            }
            if (finish_breakdown.epilogue_exec_build_us >= 0) {
                output << R"(,"finish_exec_build_us":)"
                       << finish_breakdown.epilogue_exec_build_us
                       << R"(,"finish_exec_workspace_us":)"
                       << finish_breakdown.epilogue_exec_workspace_us
                       << R"(,"finish_exec_upload_us":)"
                       << finish_breakdown.epilogue_exec_upload_us
                       << R"(,"finish_exec_launch_us":)"
                       << finish_breakdown.epilogue_exec_launch_us;
            }
            if (finish_groups >= 0) {
                output << R"(,"finish_groups":)" << finish_groups
                       << R"(,"finish_grouped_lanes":)"
                       << finish_grouped_lanes
                       << R"(,"finish_body_launches":)"
                       << finish_body_launches
                       << R"(,"finish_shared_exclusions":)"
                       << finish_shared_exclusions;
            }
            std::int64_t host_total = 0;
            if (prepare_end != fire_timing::Clock::time_point{}) {
                host_total += fire_timing::duration_us(t0, prepare_end);
            }
            if (enqueue_start != fire_timing::Clock::time_point{}) {
                host_total +=
                    fire_timing::duration_us(enqueue_start, now);
            }
            output << R"(,"host_total_us":)" << host_total << '}';
            fire_timing::write(output.str());
        } catch (...) {
        }
    }
};

}  // namespace

// All host-prepared state one step carries across prepare → enqueue →
// settle. Vectors own the geometry the dispatch view and the forward CSR
// pointers alias; the Impl is heap-held so those addresses are stable
// under PreparedStep moves.
struct PreparedStep::Impl {
    const pie_native::LaunchView* view = nullptr;
    std::unique_ptr<pipeline::StagedLaunch> staged;

    // Resolution + composition.
    pipeline::ResolvedPrograms rpg;
    pipeline::ComposedBatch composed;
    bool dg_resolved = false;
    bool composed_ready = false;
    std::vector<std::uint32_t> prog_sample_csr;
    std::vector<std::uint32_t> program_token_starts;
    pie_native::LaunchView dispatch_view{};

    // Shape + mode flags.
    int R = 0;
    int N = 0;
    int num_sampling = 0;
    bool is_pure_decode = false;
    bool empty_step = false;
    bool settle_plain = false;   // empty or fold settle: finish(nullptr, 0)
    int forward_R = 0;           // final (post-pad)
    int forward_N = 0;
    int fR_real = 0;             // pre-pad
    int fN_real = 0;
    bool has_attention_stages = false;
    bool has_decode_envelopes = false;
    bool use_fixed_decode = false;
    pipeline::FixedDecodeDeviceBuffers fixed_buffers{};
    pipeline::DecodeEnvelopeDeviceBuffers envelope_buffers{};
    bool compact_logits = false;

    // Masks.
    bool have_custom_mask = false;
    bool has_write_desc = false;
    int mask_bytes = 0;
    int mask_indptr_count = 0;
    int structured_window_left = -2;
    bool pack_structured = false;
    int pack_structured_lanes = 0;
    bool pack_dense = false;
    int pack_dense_lanes = 0;
    int pack_dense_stride = 0;
    std::size_t pack_dense_bytes = 0;

    // RS.
    pipeline::RsExecutionPlan rs_plan;
    bool rs_is_fold = false;
    bool rs_is_write = false;
    bool use_slots = false;
    std::span<const std::uint32_t> rs_slot_view;
    std::span<const std::uint8_t> rs_flag_view;
    std::span<const std::uint32_t> rs_fold_len_view;
    std::span<const std::uint32_t> rs_buf_id_view;
    std::span<const std::uint32_t> rs_buf_indptr_view;
    std::vector<std::int32_t> slot_ids_h;
    std::vector<std::uint8_t> is_fresh_h;

    // Sampling / MTP.
    std::vector<std::int32_t> sample_rows;
    std::vector<std::uint32_t> compact_logit_rows;
    const std::uint32_t* direct_logit_rows = nullptr;
    std::vector<std::uint32_t> mtp_draft_counts;
    MtpDraftPlan mtp_plan;

    // Multimodal side-channels (host pointers into the wire view).
    const float* img_pixels_h = nullptr;
    std::span<const std::uint32_t> img_pix_byte_indptr;
    std::span<const std::uint32_t> img_patch_pos;
    std::span<const std::uint32_t> img_anchor;
    int img_num_images = 0;
    std::span<const std::uint32_t> img_grids;
    std::vector<std::uint32_t> mrope_positions_storage;
    const float* aud_features_h = nullptr;
    std::span<const std::uint32_t> aud_feat_byte_indptr;
    std::span<const std::uint32_t> aud_anchor;
    int aud_num_clips = 0;
    PrecomputedEmbeddingInputs precomputed_embeddings{};

    // Forward geometry: the final host CSR pointers (post envelope-bound
    // override, post pad redirect) and the pre-override wire-forward CSR.
    ForwardInputViews forward_inputs{};
    const std::uint32_t* h_qo_forward = nullptr;
    const std::uint32_t* h_kvpi_forward = nullptr;
    const std::uint32_t* h_kvpp_forward = nullptr;
    const std::uint32_t* h_kvlpl_forward = nullptr;
    const std::uint32_t* h_kvpp_wire = nullptr;
    std::vector<std::uint32_t> plan_page_counts;
    std::vector<std::uint32_t> plan_kv_page_indptr;
    std::vector<std::uint32_t> plan_kv_last_lens;
    int graph_pad_requests = 0;
    int pad_real_mask_bytes = 0;
    std::vector<std::uint32_t> pad_qo_indptr;
    std::vector<std::uint32_t> pad_kv_page_indices;
    std::vector<std::uint32_t> pad_kv_page_indptr;
    std::vector<std::uint32_t> pad_kv_last_page_lens;

    // Staged parameter-block uploads (host memcpy done at prepare; the
    // async H2D commits run at enqueue in the original per-fire order).
    bool wire_refill = false;
    DeviceBuffer<std::uint32_t>::StagedUpload up_tokens{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_positions{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_qo{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_kvpi{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_kvpp{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_kvlpl{};
    DeviceBuffer<std::uint8_t>::StagedUpload up_custom_mask{};
    DeviceBuffer<std::int32_t>::StagedUpload up_mask_indptr{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_klen{};
    DeviceBuffer<kernels::StructuredMaskParams>::StagedUpload
        up_struct_masks{};
    DeviceBuffer<std::uint8_t>::StagedUpload up_dense_mask{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_dense_klen{};
    DeviceBuffer<std::int32_t>::StagedUpload up_dense_indptr{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_w_page{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_w_off{};
    DeviceBuffer<std::int32_t>::StagedUpload up_slot_ids{};
    DeviceBuffer<std::uint8_t>::StagedUpload up_is_fresh{};
    DeviceBuffer<std::uint8_t>::StagedUpload up_rs_flags{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_rs_fold_lens{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_rs_buf_indptr{};
    DeviceBuffer<std::uint32_t>::StagedUpload up_rs_buf_ids{};
    DeviceBuffer<std::int32_t>::StagedUpload up_sample_idx{};

    // Diagnostics. Declared last so its emission (at destruction) runs
    // while every other member is still alive.
    bool ir_trace = false;
    StepTiming timing;
};

PreparedStep::PreparedStep() : impl_(std::make_unique<Impl>()) {}
PreparedStep::~PreparedStep() = default;
PreparedStep::PreparedStep(PreparedStep&&) noexcept = default;
PreparedStep& PreparedStep::operator=(PreparedStep&&) noexcept = default;

void prepare_step(
    BatchEngine& engine,
    const pie_native::LaunchView& view,
    PreparedStep& step) {
    PreparedStep::Impl& s = *step.impl();
    s.view = &view;
    const bool dbg_fire = fire_timing::full();
    s.timing.enabled = dbg_fire;
    s.timing.uncaught = std::uncaught_exceptions();
    s.timing.t0 = dbg_fire ? fire_timing::Clock::now()
                           : fire_timing::Clock::time_point{};
    if (dbg_fire) {
        const auto ids = view.logical_fire_ids.as<std::uint64_t>();
        s.timing.fire_count = ids.size();
        s.timing.membership_hash = fire_timing::membership_hash(ids);
    }

    s.ir_trace = std::getenv("PIE_SAMPLING_IR_TRACE") != nullptr;
    if (s.ir_trace) {
        std::cerr << "[ir-trace] fire entry req_id=0\n";
        std::cerr.flush();
    }
    if (std::getenv("PIE_PTIR_TRACE")) {
        std::fprintf(stderr,
                     "[ptir-serve] entry: ptir_hashes=%zu tokens=%zu\n",
                     view.ptir_program_hashes.size(), view.token_ids.size());
    }

    auto& ws = engine.ws;
    auto& kv_cache = engine.kv_cache;
    auto& pi = engine.inputs;

    // Multimodal side-channels from the view (frame-lifetime memory).
    s.img_pixels_h =
        reinterpret_cast<const float*>(view.image_pixels.data());
    s.img_pix_byte_indptr = view.image_pixel_indptr.as<std::uint32_t>();
    s.img_patch_pos       = view.image_patch_positions.as<std::uint32_t>();
    s.img_anchor          = view.image_anchor_rows.as<std::uint32_t>();
    s.img_num_images      = static_cast<int>(s.img_anchor.size());
    s.timing.images       = s.img_num_images;
    s.img_grids           = view.image_grids.as<std::uint32_t>();
    const auto img_mrope_pos    = view.image_mrope_positions.as<std::uint32_t>();
    const auto img_mrope_indptr = view.image_mrope_indptr.as<std::uint32_t>();
    s.aud_features_h =
        reinterpret_cast<const float*>(view.audio_features.data());
    s.aud_feat_byte_indptr = view.audio_feature_indptr.as<std::uint32_t>();
    s.aud_anchor           = view.audio_anchor_rows.as<std::uint32_t>();
    s.aud_num_clips        = static_cast<int>(s.aud_anchor.size());
    s.precomputed_embeddings = PrecomputedEmbeddingInputs{
        .rows_h = view.embed_rows.data(),
        .byte_indptr_h = view.embed_indptr.data(),
        .shapes_h = view.embed_shapes.data(),
        .dtypes_h = view.embed_dtypes.data(),
        .anchor_rows_h = view.embed_anchor_rows.data(),
        .num_blocks = static_cast<int>(view.embed_dtypes.size()),
    };

    const auto tok_view_orig   = view.token_ids.as<std::uint32_t>();
    const auto pos_view_orig   = view.position_ids.as<std::uint32_t>();
    const auto qo_view_orig    = view.qo_indptr.as<std::uint32_t>();
    const auto kvpi_view_wire  = view.kv_page_indices.as<std::uint32_t>();
    const auto kvpp_view_wire  = view.kv_page_indptr.as<std::uint32_t>();
    const auto kvlpl_view_orig = view.kv_last_page_lens.as<std::uint32_t>();
    const auto sidx_view_orig  = view.sampling_indices.as<std::uint32_t>();
    const auto sptr_view_orig  = view.sampling_indptr.as<std::uint32_t>();

    s.staged = engine.dispatch->begin_host(view, engine.cublas.stream());
    if (dbg_fire) {
        s.timing.begin_end = fire_timing::Clock::now();
    }

    // ── W1.1: pre-forward device-geometry descriptor resolution ──────
    // EVERY device-geometry PTIR program in the batch ships EMPTY wire
    // geometry; the driver reads its port channels at prepare time and
    // COMPOSES the resolved geometries with the wire programs' launch
    // slices into one flat forward batch (batch_compose.hpp). A not-ready
    // descriptor channel fails the frame (W1.6). Pure-wire batches resolve
    // nothing and use the wire geometry unchanged.
    if (!view.ptir_program_hashes.empty()) {
        std::string dg_err;
        s.dg_resolved = engine.dispatch->resolve_descriptors(
            view,
            static_cast<std::uint32_t>(kv_cache.page_size()),
            static_cast<std::uint32_t>(kv_cache.num_pages()),
            s.rpg,
            &dg_err,
            engine.forward_fn.supports_runtime_window,
            s.staged.get(),
            engine.graph_cache != nullptr &&
                engine.forward_fn.graph_safe &&
                engine.tp_comm == nullptr);
        if (!s.dg_resolved && !dg_err.empty()) {
            throw std::runtime_error(dg_err);
        }
        if (s.dg_resolved) {
            // v1 mask scope: a dense device mask (AttnMask channel)
            // composes only SOLO — the runtime scheduler batches such
            // fires alone; fail loud if the contract is violated.
            if (s.rpg.per_program.size() > 1) {
                for (std::size_t p = 0; p < s.rpg.per_program.size(); ++p) {
                    if (s.rpg.is_device_geometry[p] &&
                        s.rpg.per_program[p].has_mask) {
                        throw pipeline::RetryableLaunchError(
                            "ptir: dense device mask in a multi-program "
                            "batch requires solo retry");
                    }
                }
            }
        }
        if (dbg_fire) s.timing.resolve_end = fire_timing::Clock::now();
        std::string compose_err;
        if (!pipeline::compose_forward_batch(
                view, s.rpg,
                static_cast<std::uint32_t>(kv_cache.page_size()),
                s.composed, &compose_err)) {
            throw std::runtime_error(compose_err);
        }
        s.composed_ready = true;
        if (dbg_fire) s.timing.compose_end = fire_timing::Clock::now();
        s.prog_sample_csr.assign(
            s.composed.prog_sample_counts.size() + 1, 0);
        for (std::size_t program = 0;
             program < s.composed.prog_sample_counts.size();
             ++program) {
            s.prog_sample_csr[program + 1] =
                s.prog_sample_csr[program] +
                s.composed.prog_sample_counts[program];
        }
        if (s.rpg.device_composed) {
            constexpr std::uint32_t unavailable =
                std::numeric_limits<std::uint32_t>::max();
            std::fill(
                s.composed.prog_kv_lens.begin(),
                s.composed.prog_kv_lens.end(),
                unavailable);
            std::fill(
                s.composed.prog_page_counts.begin(),
                s.composed.prog_page_counts.end(),
                unavailable);
            std::fill(
                s.composed.prog_key_lens.begin(),
                s.composed.prog_key_lens.end(),
                unavailable);
        }
    }
    if (dbg_fire &&
        s.timing.resolve_end == fire_timing::Clock::time_point{}) {
        s.timing.resolve_end = fire_timing::Clock::now();
        s.timing.compose_end = s.timing.resolve_end;
    }
    // A dense device mask is always solo. Its geometry may come from the
    // descriptor resolver or remain on the ordinary wire path.
    const pipeline::FireGeometry* solo_fg =
        (s.dg_resolved && s.rpg.per_program.size() == 1 &&
         (s.rpg.is_device_geometry[0] ||
          s.rpg.per_program[0].has_mask ||
          static_cast<bool>(s.rpg.per_program[0].structured_mask)))
            ? &s.rpg.per_program[0]
            : nullptr;
    bool use_structured_mask = false;
    bool pack_structured_mask = false;
    std::vector<pie_native::ptir::StructuredMaskDescriptor>
        effective_structured_masks = s.composed.structured_masks;
    const auto mask_coverage = pipeline::structured_mask_coverage(
        effective_structured_masks);
    if (mask_coverage ==
        pipeline::StructuredMaskCoverage::Mixed) {
        throw pipeline::RetryableLaunchError(
            "explicit PTIR masks cannot share one runtime override with "
            "ordinary wire requests");
    }
    const auto effective_first = std::find_if(
        effective_structured_masks.begin(),
        effective_structured_masks.end(),
        [](const auto& descriptor) {
            return static_cast<bool>(descriptor);
        });
    // Composed geometry takes precedence over the borrowed launch
    // slices. No request/speculation carrier is expanded in the driver.
    const int R = static_cast<int>(
        s.composed_ready ? s.composed.qo_indptr.size()
                         : qo_view_orig.size()) - 1;
    s.R = R;

    const std::span<const std::uint32_t> tok_view   = s.composed_ready ? std::span<const std::uint32_t>(s.composed.token_ids)         : tok_view_orig;
    const std::span<const std::uint32_t> pos_view   = s.composed_ready ? std::span<const std::uint32_t>(s.composed.position_ids)      : pos_view_orig;
    const std::span<const std::uint32_t> qo_view    = s.composed_ready ? std::span<const std::uint32_t>(s.composed.qo_indptr)         : qo_view_orig;
    const std::span<const std::uint32_t> kvpi_view  = s.composed_ready ? std::span<const std::uint32_t>(s.composed.kv_page_indices)   : kvpi_view_wire;
    const std::span<const std::uint32_t> kvpp_view  = s.composed_ready ? std::span<const std::uint32_t>(s.composed.kv_page_indptr)    : kvpp_view_wire;
    const std::span<const std::uint32_t> kvlpl_view = s.composed_ready ? std::span<const std::uint32_t>(s.composed.kv_last_page_lens) : kvlpl_view_orig;
    const std::span<const std::uint32_t> sidx_view  = s.composed_ready ? std::span<const std::uint32_t>(s.composed.sampling_indices)  : sidx_view_orig;
    const std::span<const std::uint32_t> sptr_view  = s.composed_ready ? std::span<const std::uint32_t>(s.composed.sampling_indptr)   : sptr_view_orig;
    if (effective_first != effective_structured_masks.end() &&
        engine.forward_fn.supports_runtime_window) {
        const auto window = pipeline::runtime_window_for_tail_aligned(
            effective_structured_masks,
            pos_view,
            qo_view,
            kvpp_view,
            kvlpl_view,
            static_cast<std::uint32_t>(kv_cache.page_size()));
        if (window.has_value()) {
            use_structured_mask = true;
            s.structured_window_left = *window;
        } else {
            pack_structured_mask = true;
        }
    }
    s.dispatch_view = view;
    if (s.composed_ready) {
        s.dispatch_view.rs_slot_ids = pie_native::slice_from_u32(
            s.composed.rs_slot_ids.data(), s.composed.rs_slot_ids.size());
        s.dispatch_view.rs_slot_flags = pie_native::slice_from_u8(
            s.composed.rs_slot_flags.data(),
            s.composed.rs_slot_flags.size());
        s.dispatch_view.rs_fold_lens = pie_native::slice_from_u32(
            s.composed.rs_fold_lens.data(), s.composed.rs_fold_lens.size());
        s.dispatch_view.rs_buffer_slot_ids = pie_native::slice_from_u32(
            s.composed.rs_buffer_slot_ids.data(),
            s.composed.rs_buffer_slot_ids.size());
        s.dispatch_view.rs_buffer_slot_indptr = pie_native::slice_from_u32(
            s.composed.rs_buffer_slot_indptr.data(),
            s.composed.rs_buffer_slot_indptr.size());
        s.dispatch_view.sampling_indices = pie_native::slice_from_u32(
            sidx_view.data(), sidx_view.size());
        s.dispatch_view.sampling_indptr = pie_native::slice_from_u32(
            s.prog_sample_csr.data(), s.prog_sample_csr.size());
        s.dispatch_view.ptir_sample_starts = pie_native::slice_from_u32(
            s.composed.prog_sample_starts.data(),
            s.composed.prog_sample_starts.size());
        s.dispatch_view.ptir_sample_counts = pie_native::slice_from_u32(
            s.composed.prog_sample_counts.data(),
            s.composed.prog_sample_counts.size());
        s.dispatch_view.ptir_row_counts = pie_native::slice_from_u32(
            s.composed.prog_row_counts.data(),
            s.composed.prog_row_counts.size());
        s.dispatch_view.ptir_token_counts = pie_native::slice_from_u32(
            s.composed.prog_token_counts.data(),
            s.composed.prog_token_counts.size());
        s.dispatch_view.ptir_kv_lens = pie_native::slice_from_u32(
            s.composed.prog_kv_lens.data(),
            s.composed.prog_kv_lens.size());
        s.dispatch_view.ptir_page_counts = pie_native::slice_from_u32(
            s.composed.prog_page_counts.data(),
            s.composed.prog_page_counts.size());
        s.dispatch_view.ptir_query_lens = pie_native::slice_from_u32(
            s.composed.prog_query_lens.data(),
            s.composed.prog_query_lens.size());
        s.dispatch_view.ptir_key_lens = pie_native::slice_from_u32(
            s.composed.prog_key_lens.data(),
            s.composed.prog_key_lens.size());
    }
    s.program_token_starts.assign(view.ptir_program_hashes.size(), 0);
    if (s.composed_ready) {
        for (std::size_t program = 0;
             program < s.program_token_starts.size();
             ++program) {
            const std::uint32_t request =
                s.composed.prog_request_starts[program];
            if (request >= s.composed.qo_indptr.size()) {
                throw std::runtime_error(
                    "PTIR program token attribution is outside composed geometry");
            }
            s.program_token_starts[program] =
                s.composed.qo_indptr[request];
        }
    }
    // NOTE: update_launch_geometry is DEFERRED to enqueue_step, after the
    // Prologue phase — the Prologue historically executed against the
    // pre-resolution wave state (begin ran before resolve/compose), and
    // its stage bindings (including the predicated commit logic) must keep
    // seeing exactly those inputs.
    s.mtp_draft_counts = engine.dispatch->mtp_draft_rows(s.dispatch_view);

    if (dbg_fire) s.timing.wire_parse_end = fire_timing::Clock::now();

    const int N = static_cast<int>(tok_view.size());
    const int num_sampling = static_cast<int>(sidx_view.size());
    s.N = N;
    s.num_sampling = num_sampling;
    s.timing.requests = R;
    s.timing.tokens = N;
    if (s.ir_trace) {
        std::cerr << "[ir-trace] fire shape req_id=0"
                  << " N=" << N << " R=" << R
                  << " num_sampling=" << num_sampling << "\n";
        std::cerr.flush();
    }

    // Qwen3-VL: assemble the per-token [N,3] M-RoPE positions. Text rows
    // carry (p,p,p) from the 1-D `pos_view`; image-token rows are
    // overwritten with the staged 3-axis (t,h,w) positions for each image.
    if (s.img_num_images > 0 && !img_mrope_pos.empty()) {
        s.mrope_positions_storage.resize(static_cast<std::size_t>(N) * 3);
        for (int t = 0; t < N; ++t) {
            const std::uint32_t p =
                t < static_cast<int>(pos_view.size()) ? pos_view[t] : 0u;
            s.mrope_positions_storage[3 * t + 0] = p;
            s.mrope_positions_storage[3 * t + 1] = p;
            s.mrope_positions_storage[3 * t + 2] = p;
        }
        for (int im = 0; im < s.img_num_images; ++im) {
            const std::uint32_t anchor_row = s.img_anchor[im];
            const std::uint32_t lo =
                im < static_cast<int>(img_mrope_indptr.size()) - 1
                    ? img_mrope_indptr[im] : 0u;
            const std::uint32_t hi =
                im + 1 < static_cast<int>(img_mrope_indptr.size())
                    ? img_mrope_indptr[im + 1] : 0u;
            const std::uint32_t n_tok = (hi - lo) / 3u;
            for (std::uint32_t j = 0; j < n_tok; ++j) {
                const int row = static_cast<int>(anchor_row + j);
                if (row < 0 || row >= N) continue;
                s.mrope_positions_storage[3 * row + 0] = img_mrope_pos[lo + 3 * j + 0];
                s.mrope_positions_storage[3 * row + 1] = img_mrope_pos[lo + 3 * j + 1];
                s.mrope_positions_storage[3 * row + 2] = img_mrope_pos[lo + 3 * j + 2];
            }
        }
    }

    // Detect pure decode so the model can choose its decode kernel.
    const std::uint32_t* h_qo = qo_view.data();
    bool is_pure_decode = (R > 0);
    for (int r = 0; r < R; ++r) {
        if (h_qo[r + 1] - h_qo[r] != 1u) is_pure_decode = false;
    }

    s.rs_slot_view = s.composed_ready
        ? std::span<const std::uint32_t>(s.composed.rs_slot_ids)
        : view.rs_slot_ids.as<std::uint32_t>();
    s.rs_flag_view = s.composed_ready
        ? std::span<const std::uint8_t>(s.composed.rs_slot_flags)
        : view.rs_slot_flags.as<std::uint8_t>();
    s.rs_fold_len_view = s.composed_ready
        ? std::span<const std::uint32_t>(s.composed.rs_fold_lens)
        : view.rs_fold_lens.as<std::uint32_t>();
    std::string rs_binding_error;
    if (!pipeline::validate_folded_rs_bindings(
            s.rs_slot_view,
            s.rs_flag_view,
            s.rs_fold_len_view,
            static_cast<std::size_t>(std::max(R, 0)),
            engine.rs_cache != nullptr,
            &rs_binding_error)) {
        throw std::runtime_error(rs_binding_error);
    }
    s.use_slots = !s.rs_slot_view.empty();
    // Ph7 RS working-set buffered-activation channel (single-role per
    // pass, v1): FOLD gathers+replays from the buffered pool; an rs-output
    // write pass scatters in-proj slabs during the main forward.
    s.rs_buf_id_view = s.composed_ready
        ? std::span<const std::uint32_t>(s.composed.rs_buffer_slot_ids)
        : view.rs_buffer_slot_ids.as<std::uint32_t>();
    s.rs_buf_indptr_view = s.composed_ready
        ? std::span<const std::uint32_t>(s.composed.rs_buffer_slot_indptr)
        : view.rs_buffer_slot_indptr.as<std::uint32_t>();
    if (!pipeline::plan_rs_execution(
            s.rs_slot_view,
            s.rs_flag_view,
            s.rs_fold_len_view,
            s.rs_buf_id_view,
            s.rs_buf_indptr_view,
            qo_view,
            engine.rs_cache != nullptr,
            engine.rs_cache != nullptr &&
                engine.rs_cache->rs_buffer_pool_enabled(),
            engine.rs_cache != nullptr
                ? static_cast<std::uint32_t>(
                      engine.rs_cache->rs_buffer_page_tokens())
                : 0,
            s.rs_plan,
            &rs_binding_error)) {
        throw std::runtime_error(rs_binding_error);
    }
    s.rs_is_fold = s.rs_plan.mode == RsExecutionMode::BufferFold;
    s.rs_is_write = s.rs_plan.mode == RsExecutionMode::BufferWrite;
    if (s.rs_is_fold && !sidx_view.empty()) {
        throw std::runtime_error(
            "buffered RS fold is state-only and cannot sample logits");
    }

    // Anatomical stages run through explicit model hooks.
    s.forward_inputs = make_forward_input_views(
        tok_view, pos_view, qo_view, kvpi_view, kvpp_view, kvlpl_view, R);
    if (s.rs_is_fold) {
        s.forward_inputs.qo_indptr =
            std::span<const std::uint32_t>(s.rs_plan.fold_qo_indptr);
        s.forward_inputs.total_tokens =
            static_cast<int>(s.rs_plan.fold_tokens);
    }
    s.fN_real = s.forward_inputs.total_tokens;
    s.fR_real = s.forward_inputs.num_requests;
    s.forward_N = s.fN_real;
    s.forward_R = s.fR_real;
    s.h_qo_forward = s.forward_inputs.qo_indptr.data();
    s.h_kvpi_forward = s.forward_inputs.kv_page_indices.data();
    s.h_kvpp_forward = s.forward_inputs.kv_page_indptr.data();
    s.h_kvlpl_forward = s.forward_inputs.kv_last_page_lens.data();
    s.h_kvpp_wire = s.h_kvpp_forward;
    if (s.fN_real == 0 || s.fR_real <= 0) {
        // Nothing to forward: the wave still runs its channel phases and
        // settles plainly (Prologue/Epilogue at enqueue/settle).
        s.empty_step = true;
        s.settle_plain = true;
        if (dbg_fire) s.timing.prepare_end = fire_timing::Clock::now();
        return;
    }
    if (s.fN_real > engine.max_workspace_tokens) {
        std::cerr << "[pie-driver-cuda] batch tokens=" << s.fN_real
                  << " exceeds workspace=" << engine.max_workspace_tokens
                  << "\n";
        throw std::runtime_error("forward batch exceeds workspace capacity");
    }
    if (s.rs_is_fold) {
        is_pure_decode = std::all_of(
            s.rs_fold_len_view.begin(), s.rs_fold_len_view.end(),
            [](std::uint32_t length) { return length == 1; });
    }
    s.is_pure_decode = is_pure_decode;

    // Stage the persistent-input refill for this step's parameter block.
    // Same device addresses every fire — required for graph-replay safety;
    // the commits run at StepEnqueue in the original order.
    s.wire_refill = !s.rpg.device_composed;
    if (s.wire_refill) {
        s.up_tokens = pi.tokens.stage_from_host(s.forward_inputs.tokens);
        s.up_positions =
            pi.positions.stage_from_host(s.forward_inputs.positions);
        s.up_qo = pi.qo_indptr.stage_from_host(s.forward_inputs.qo_indptr);
        s.up_kvpi = pi.kv_page_indices.stage_from_host(
            s.forward_inputs.kv_page_indices);
        s.up_kvpp = pi.kv_page_indptr.stage_from_host(
            s.forward_inputs.kv_page_indptr);
        s.up_kvlpl = pi.kv_last_page_lens.stage_from_host(
            s.forward_inputs.kv_last_page_lens);
    }

    // BRLE attention masks. For any batch that isn't pure causal, decode
    // a packed bitmap here (host) and route through the flashinfer kCustom
    // path. A decode-shaped batch carrying a per-cell custom mask (§6.2
    // beam fire) must ALSO build the mask; a normal decode batch carries
    // no mask (`fmask_view` empty) so it is unaffected.
    const auto fmask_view  = view.flattened_masks.as<std::uint32_t>();
    const auto mskptr_view = view.mask_indptr.as<std::uint32_t>();
    if (!fmask_view.empty()) {
        const bool resolved_custom_wire =
            s.dg_resolved && view.has_user_mask;
        if (resolved_custom_wire &&
            view.ptir_program_hashes.size() > 1) {
            throw std::runtime_error(
                "ptir: host-derived masks on device geometry require a "
                "solo program");
        }
        const auto qo_span = std::span<const std::uint32_t>(
            resolved_custom_wire ? qo_view.data() : qo_view_orig.data(),
            resolved_custom_wire ? qo_view.size() : qo_view_orig.size());
        const auto kvpp_span = std::span<const std::uint32_t>(
            resolved_custom_wire
                ? kvpp_view.data()
                : kvpp_view_wire.data(),
            resolved_custom_wire
                ? kvpp_view.size()
                : kvpp_view_wire.size());
        const auto kvlpl_span = std::span<const std::uint32_t>(
            resolved_custom_wire
                ? kvlpl_view.data()
                : kvlpl_view_orig.data(),
            resolved_custom_wire
                ? kvlpl_view.size()
                : kvlpl_view_orig.size());
        bool pure_causal =
            pie_cuda_driver::brle::is_pure_causal(
                fmask_view, mskptr_view,
                qo_span, kvpp_span, kvlpl_span,
                kv_cache.page_size());
        if (!pure_causal && !view.has_user_mask && is_pure_decode) {
            std::vector<std::uint32_t> logical_kv_lens;
            pure_causal =
                pie_cuda_driver::brle::causal_prefix_lengths(
                    fmask_view,
                    mskptr_view,
                    qo_span,
                    kvpp_span,
                    kvlpl_span,
                    kv_cache.page_size(),
                    logical_kv_lens);
        }
        if (!pure_causal) {
            if (!s.dg_resolved || resolved_custom_wire) {
                auto decoded = pie_cuda_driver::brle::decode(
                    fmask_view, mskptr_view,
                    qo_span, kvpp_span, kvlpl_span,
                    kv_cache.page_size());
                s.up_custom_mask = pi.custom_mask.stage_from_host(
                    std::span<const std::uint8_t>(decoded.packed));
                s.up_mask_indptr = pi.custom_mask_indptr.stage_from_host(
                    std::span<const std::int32_t>(decoded.mask_indptr));
                s.mask_bytes = static_cast<int>(decoded.packed.size());
                s.mask_indptr_count =
                    static_cast<int>(decoded.mask_indptr.size());
                s.have_custom_mask = true;
            }
        }
    }

    if (pack_structured_mask && !s.have_custom_mask) {
        const int lanes = static_cast<int>(qo_view.size()) - 1;
        std::vector<std::uint32_t> klen(
            static_cast<std::size_t>(lanes), 0);
        std::vector<std::int32_t> mindptr(
            static_cast<std::size_t>(lanes) + 1, 0);
        std::vector<kernels::StructuredMaskParams> masks(
            static_cast<std::size_t>(lanes));
        const std::uint32_t page =
            static_cast<std::uint32_t>(kv_cache.page_size());
        for (int lane = 0; lane < lanes; ++lane) {
            const std::uint32_t pages =
                kvpp_view[lane + 1] - kvpp_view[lane];
            klen[lane] = pages == 0
                ? 0
                : (pages - 1) * page + kvlpl_view[lane];
            const std::uint32_t queries =
                qo_view[lane + 1] - qo_view[lane];
            const std::uint64_t bits =
                static_cast<std::uint64_t>(queries) * klen[lane];
            if (bits > static_cast<std::uint64_t>(
                           std::numeric_limits<std::int32_t>::max()) *
                    8 ||
                mindptr[lane] >
                    std::numeric_limits<std::int32_t>::max() -
                        static_cast<std::int64_t>((bits + 7) / 8)) {
                throw std::runtime_error(
                    "structured attention mask exceeds packed ABI");
            }
            mindptr[lane + 1] = mindptr[lane] +
                static_cast<std::int32_t>((bits + 7) / 8);
            const auto& descriptor =
                effective_structured_masks[lane];
            masks[lane] = {
                static_cast<std::uint32_t>(descriptor.kind),
                descriptor.window,
                descriptor.sink,
            };
        }
        const std::size_t packed_bytes =
            static_cast<std::size_t>(mindptr.back());
        if (packed_bytes > pi.custom_mask.size() ||
            mindptr.size() > pi.custom_mask_indptr.size() ||
            klen.size() > pi.structured_mask_klen.size() ||
            masks.size() > pi.structured_masks.size()) {
            throw std::runtime_error(
                "structured attention mask exceeds persistent capacity");
        }
        s.up_klen = pi.structured_mask_klen.stage_from_host(
            std::span<const std::uint32_t>(klen));
        s.up_struct_masks = pi.structured_masks.stage_from_host(
            std::span<const kernels::StructuredMaskParams>(masks));
        s.up_mask_indptr = pi.custom_mask_indptr.stage_from_host(
            std::span<const std::int32_t>(mindptr));
        s.pack_structured = true;
        s.pack_structured_lanes = lanes;
        s.have_custom_mask = true;
        s.mask_bytes = static_cast<int>(packed_bytes);
        s.mask_indptr_count = lanes + 1;
    }

    // ── W1.3: device-geometry AttnMask → FlashInfer packed custom mask ──
    // A device-geometry fire may carry a DENSE [lanes, stride] per-cell
    // mask on its AttnMask descriptor port (resolved into fg.mask). Staged
    // here; packed to the bit-packed custom mask at enqueue so the standard
    // custom-mask forward path consumes it like a BRLE-decoded wire mask.
    if (solo_fg != nullptr && solo_fg->has_mask &&
        !solo_fg->mask.empty() && !use_structured_mask) {
        const pipeline::FireGeometry& fg = *solo_fg;
        const int lanes = static_cast<int>(qo_view.size()) - 1;
        const int total_q =
            lanes > 0 ? static_cast<int>(qo_view[lanes]) : 0;
        if (lanes > 0 && total_q > 0 &&
            fg.mask.size() % static_cast<std::size_t>(total_q) == 0) {
            const int stride = static_cast<int>(
                fg.mask.size() / static_cast<std::size_t>(total_q));
            const std::uint32_t page =
                static_cast<std::uint32_t>(kv_cache.page_size());
            std::vector<std::uint32_t> klen(
                static_cast<std::size_t>(lanes), 0);
            std::vector<std::int32_t> mindptr(
                static_cast<std::size_t>(lanes) + 1, 0);
            for (int l = 0; l < lanes; ++l) {
                const bool resolved_geometry =
                    !fg.kv_page_indptr.empty();
                const std::uint32_t np = resolved_geometry
                    ? ((l + 1 < static_cast<int>(fg.kv_page_indptr.size()))
                           ? fg.kv_page_indptr[l + 1] -
                                 fg.kv_page_indptr[l]
                           : 0u)
                    : kvpp_view[l + 1] - kvpp_view[l];
                const std::uint32_t lpl = resolved_geometry
                    ? ((l < static_cast<int>(
                                  fg.kv_last_page_lens.size()))
                           ? fg.kv_last_page_lens[l]
                           : 0u)
                    : kvlpl_view[l];
                klen[l] = np == 0 ? 0u : (np - 1) * page + lpl;
                const std::uint32_t qo_len =
                    qo_view[l + 1] - qo_view[l];
                const std::uint64_t bits =
                    static_cast<std::uint64_t>(qo_len) * klen[l];
                mindptr[l + 1] = mindptr[l] +
                    static_cast<std::int32_t>((bits + 7u) / 8u);
            }
            const std::size_t packed_bytes =
                static_cast<std::size_t>(mindptr[lanes]);
            if (packed_bytes > 0 &&
                packed_bytes <= pi.custom_mask.size() &&
                static_cast<std::size_t>(lanes) + 1 <=
                    pi.custom_mask_indptr.size() &&
                fg.mask.size() <= pi.dense_mask.size() &&
                klen.size() <= pi.structured_mask_klen.size()) {
                s.up_dense_mask = pi.dense_mask.stage_from_host(
                    std::span<const std::uint8_t>(fg.mask));
                s.up_dense_klen = pi.structured_mask_klen.stage_from_host(
                    std::span<const std::uint32_t>(klen));
                s.up_dense_indptr = pi.custom_mask_indptr.stage_from_host(
                    std::span<const std::int32_t>(mindptr));
                s.pack_dense = true;
                s.pack_dense_lanes = lanes;
                s.pack_dense_stride = stride;
                s.pack_dense_bytes = packed_bytes;
                s.have_custom_mask = true;
                s.mask_bytes = static_cast<int>(packed_bytes);
                s.mask_indptr_count = lanes + 1;
            } else if (packed_bytes > 0) {
                throw std::runtime_error(
                    "dense attention mask exceeds persistent capacity");
            }
        }
    }

    // Explicit KV-write descriptor upload (device-geometry WSlot/WOff, B2).
    if (s.dg_resolved && s.composed.has_write_desc &&
        !s.composed.w_page.empty()) {
        if (s.composed.w_page.size() != s.composed.w_off.size() ||
            s.composed.w_page.size() > pi.w_page.size()) {
            throw std::runtime_error(
                "ptir: composed write descriptor exceeds persistent "
                "input capacity");
        }
        if (!s.rpg.device_composed) {
            s.up_w_page = pi.w_page.stage_from_host(
                std::span<const std::uint32_t>(s.composed.w_page));
            s.up_w_off = pi.w_off.stage_from_host(
                std::span<const std::uint32_t>(s.composed.w_off));
        }
        s.has_write_desc = true;
    }

    // Graph-lattice padding decision (V6 iterations 8–9): eligible
    // pure-decode waves pad to `forward_graph_request_bucket` so
    // off-lattice widths never capture throwaway single-use graphs.
    s.has_attention_stages =
        engine.dispatch->launch_has_attention_stages(s.dispatch_view);
    {
        const bool eligible = forward_graph_replay_eligible(
            engine,
            is_pure_decode,
            s.have_custom_mask,
            s.rs_is_write,
            s.rs_is_fold,
            s.has_write_desc,
            s.structured_window_left,
            s.use_slots,
            nullptr,
            s.fR_real,
            s.img_num_images,
            s.aud_num_clips,
            s.has_attention_stages);
        if (eligible && engine.graph_pad_page >= 0) {
            const int max_requests = std::min(
                engine.max_forward_requests, engine.max_workspace_tokens);
            const int bucket =
                forward_graph_request_bucket(s.fR_real, max_requests);
            const int padding = bucket - s.fR_real;
            const std::size_t padded_tokens =
                static_cast<std::size_t>(s.fN_real) +
                static_cast<std::size_t>(std::max(padding, 0));
            const bool fits = padding > 0 &&
                padded_tokens <= pi.tokens.size() &&
                padded_tokens <= pi.row_valid.size() &&
                padded_tokens <=
                    static_cast<std::size_t>(tensor_rows(ws.logits)) &&
                static_cast<std::size_t>(bucket) + 1 <=
                    pi.qo_indptr.size() &&
                static_cast<std::size_t>(bucket) + 1 <=
                    pi.kv_page_indptr.size() &&
                static_cast<std::size_t>(bucket) <=
                    pi.kv_last_page_lens.size() &&
                (!s.have_custom_mask ||
                 (static_cast<std::size_t>(s.mask_bytes + padding) <=
                      pi.custom_mask.size() &&
                  static_cast<std::size_t>(bucket) + 1 <=
                      pi.custom_mask_indptr.size()));
            if (fits) {
                s.graph_pad_requests = padding;
            }
        }
    }

    // Linear-attention rs_cache slots. Runtime owns slot assignment;
    // RS-capable models must receive one slot id per request.
    if (s.use_slots) {
        const int slot_count = R;
        if (s.rs_flag_view.size() > pi.rs_slot_flags.size() ||
            s.rs_fold_len_view.size() > pi.rs_fold_lens.size() ||
            s.rs_buf_id_view.size() > pi.rs_buffer_slot_ids.size() ||
            s.rs_buf_indptr_view.size() >
                pi.rs_buffer_slot_indptr.size()) {
            throw std::runtime_error(
                "RS metadata exceeds persistent input capacity");
        }
        s.slot_ids_h.resize(slot_count);
        s.is_fresh_h.resize(slot_count);
        for (int r = 0; r < R; ++r) {
            s.slot_ids_h[r] =
                static_cast<std::int32_t>(s.rs_slot_view[r]);
            s.is_fresh_h[r] = (s.rs_flag_view[r] & PIE_RS_FLAG_RESET)
                                  ? 1u
                                  : 0u;
        }
        for (int r = R; r < slot_count; ++r) {
            s.slot_ids_h[r] = engine.graph_pad_slot;
            s.is_fresh_h[r] = 0u;
        }
        s.up_slot_ids = pi.slot_ids.stage_from_host(
            std::span<const std::int32_t>(s.slot_ids_h));
        s.up_is_fresh = pi.is_fresh.stage_from_host(
            std::span<const std::uint8_t>(s.is_fresh_h));
        s.up_rs_flags = pi.rs_slot_flags.stage_from_host(s.rs_flag_view);
        if (!s.rs_fold_len_view.empty()) {
            s.up_rs_fold_lens =
                pi.rs_fold_lens.stage_from_host(s.rs_fold_len_view);
        }
        if (!s.rs_buf_indptr_view.empty()) {
            s.up_rs_buf_indptr = pi.rs_buffer_slot_indptr.stage_from_host(
                s.rs_buf_indptr_view);
        }
        if (!s.rs_buf_id_view.empty()) {
            s.up_rs_buf_ids =
                pi.rs_buffer_slot_ids.stage_from_host(s.rs_buf_id_view);
        }
    }

    if (!s.rs_is_fold &&
        (sptr_view.size() != static_cast<std::size_t>(R + 1) ||
         sptr_view.back() != sidx_view.size())) {
        throw std::runtime_error(
            "sampling CSR does not match launched instances");
    }
    std::string sampling_error;
    if (!s.rs_is_fold && !pipeline::global_sampling_rows(
            qo_view,
            sptr_view,
            sidx_view,
            s.sample_rows,
            &sampling_error)) {
        throw std::runtime_error(sampling_error);
    }
    if (s.sample_rows.size() > pi.sample_idx.size()) {
        throw std::runtime_error(
            "sampling rows exceed persistent input capacity");
    }
    if (!s.rs_is_fold && N > tensor_rows(ws.logits)) {
        throw std::runtime_error("forward batch exceeds logits workspace");
    }
    if (!s.sample_rows.empty() && !s.rpg.device_composed) {
        s.up_sample_idx = pi.sample_idx.stage_from_host(
            std::span<const std::int32_t>(s.sample_rows));
    }
    s.mtp_plan = preflight_mtp_draft_logits(
        engine, s.composed, s.sample_rows, s.mtp_draft_counts);
    s.compact_logits =
        !is_pure_decode &&
        s.mtp_plan.work.empty() &&
        num_sampling > 0 &&
        num_sampling < s.fN_real;
    if (s.rs_is_fold && !s.mtp_plan.work.empty()) {
        throw std::runtime_error(
            "state-only buffered RS fold cannot produce MTP drafts");
    }
    if (s.rs_is_fold && s.has_attention_stages) {
        throw std::runtime_error(
            "state-only buffered RS fold cannot execute anatomical PTIR "
            "attention stages");
    }
    if (!tp_rs_metadata_shape_valid(
            s.rs_plan.mode,
            static_cast<std::size_t>(s.fR_real),
            s.rs_slot_view.size(),
            s.rs_flag_view.size(),
            s.rs_fold_len_view.size(),
            (s.rs_is_write || s.rs_is_fold) ? s.rs_buf_id_view.size() : 0,
            (s.rs_is_write || s.rs_is_fold)
                ? s.rs_buf_indptr_view.size()
                : 0)) {
        throw std::runtime_error(
            "RS metadata cannot be represented by the TP payload");
    }
    s.has_decode_envelopes = s.composed_ready &&
        engine.dispatch->has_decode_envelopes(s.dispatch_view);

    // Stage the device-composition lane tables NOW: they read live
    // registry ring cursors, which are only valid between this step's
    // begin_host and the next step's (the frame driver prepares steps in
    // wave order). The enqueue halves claim the upload arena and launch
    // the compose kernel at the step's stream position.
    if (s.has_decode_envelopes) {
        bool staged_ok = false;
        if (engine.graph_pad_page >= 0 && s.rpg.device_composed) {
            s.use_fixed_decode = true;
            s.fixed_buffers = pipeline::FixedDecodeDeviceBuffers{
                .token_ids = pi.tokens.data(),
                .position_ids = pi.positions.data(),
                .qo_indptr = pi.qo_indptr.data(),
                .kv_page_indices = pi.kv_page_indices.data(),
                .kv_page_indptr = pi.kv_page_indptr.data(),
                .kv_last_page_lens = pi.kv_last_page_lens.data(),
                .w_page = pi.w_page.data(),
                .w_off = pi.w_off.data(),
                .row_valid = pi.row_valid.data(),
                .rs_slot_ids =
                    s.use_slots ? pi.slot_ids.data() : nullptr,
                .sample_indices = pi.sample_idx.data(),
                .token_capacity = pi.tokens.size(),
                .request_capacity = pi.kv_last_page_lens.size(),
                .page_capacity = pi.kv_page_indices.size(),
                .dummy_page = static_cast<std::uint32_t>(
                    engine.graph_pad_page),
            };
            std::string fixed_error;
            staged_ok = engine.dispatch->stage_fixed_decode(
                s.dispatch_view,
                static_cast<std::uint32_t>(kv_cache.page_size()),
                static_cast<std::uint32_t>(kv_cache.num_pages()),
                s.fixed_buffers,
                &fixed_error,
                *s.staged);
            if (!staged_ok && !fixed_error.empty()) {
                throw std::runtime_error(fixed_error);
            }
        } else if (engine.graph_pad_page >= 0) {
            s.envelope_buffers = pipeline::DecodeEnvelopeDeviceBuffers{
                .token_ids = pi.tokens.data(),
                .position_ids = pi.positions.data(),
                .kv_page_indices = pi.kv_page_indices.data(),
                .kv_page_indptr = pi.kv_page_indptr.data(),
                .kv_last_page_lens = pi.kv_last_page_lens.data(),
                .row_valid = pi.row_valid.data(),
                .rs_slot_ids =
                    s.use_slots ? pi.slot_ids.data() : nullptr,
                .dummy_page = static_cast<std::uint32_t>(
                    engine.graph_pad_page),
                .page_size = static_cast<std::uint32_t>(
                    kv_cache.page_size()),
            };
            std::string envelope_error;
            staged_ok = engine.dispatch->stage_decode_envelopes(
                s.dispatch_view,
                s.program_token_starts,
                s.composed.prog_request_starts,
                std::span<const std::uint32_t>(
                    s.h_kvpp_wire,
                    static_cast<std::size_t>(s.fR_real) + 1),
                s.envelope_buffers,
                &envelope_error,
                *s.staged);
            if (!staged_ok && !envelope_error.empty()) {
                throw std::runtime_error(envelope_error);
            }
        }
        if (!staged_ok) {
            throw std::runtime_error(
                "decode envelope composition is unavailable");
        }
    }

    // Plan attention from the WS envelope upper bound, never from
    // placeholder wire geometry: device-resolved lanes carry their real KV
    // lengths only on device, so XQA bucket selection and FlashInfer plans
    // must cover the host-known reserved envelope or long-KV attention
    // truncates silently.
    if (s.has_decode_envelopes &&
        engine.dispatch->envelope_plan_page_bounds(
            s.dispatch_view,
            s.composed.prog_request_starts,
            std::span<const std::uint32_t>(
                s.h_kvpp_wire,
                static_cast<std::size_t>(s.fR_real) + 1),
            s.plan_page_counts)) {
        const auto page = static_cast<std::uint32_t>(
            kv_cache.page_size());
        s.plan_kv_page_indptr.assign(
            static_cast<std::size_t>(s.fR_real) + 1, 0);
        for (int request = 0; request < s.fR_real; ++request) {
            s.plan_kv_page_indptr[request + 1] =
                s.plan_kv_page_indptr[request] +
                s.plan_page_counts[request];
        }
        s.plan_kv_last_lens.assign(
            static_cast<std::size_t>(s.fR_real), page);
        s.h_kvpp_forward = s.plan_kv_page_indptr.data();
        s.h_kvlpl_forward = s.plan_kv_last_lens.data();
    }

    // Apply the graph-lattice padding decided above. Padded COPIES of the
    // host CSRs feed the flashinfer plan / TP broadcast / graph key; the
    // pad-fill kernel (StepEnqueue) writes the pad lanes' device rows
    // coherently (a host-side copy left device-composed rows stale and
    // hung the attention kernel; V6 iteration 8).
    if (s.graph_pad_requests > 0) {
        s.pad_real_mask_bytes = s.mask_bytes;
        if (s.have_custom_mask) {
            if (s.mask_indptr_count != s.fR_real + 1) {
                throw std::runtime_error(
                    "custom attention mask CSR does not match graph padding");
            }
            s.mask_bytes += s.graph_pad_requests;
            s.mask_indptr_count += s.graph_pad_requests;
        }
        s.pad_qo_indptr.assign(
            s.h_qo_forward, s.h_qo_forward + s.fR_real + 1);
        s.pad_kv_page_indptr.assign(
            s.h_kvpp_forward, s.h_kvpp_forward + s.fR_real + 1);
        const std::size_t pages = s.pad_kv_page_indptr.back();
        s.pad_kv_page_indices.assign(
            s.h_kvpi_forward, s.h_kvpi_forward + pages);
        s.pad_kv_last_page_lens.assign(
            s.h_kvlpl_forward, s.h_kvlpl_forward + s.fR_real);
        for (int r = 0; r < s.graph_pad_requests; ++r) {
            s.pad_qo_indptr.push_back(s.pad_qo_indptr.back() + 1);
            s.pad_kv_page_indices.push_back(
                static_cast<std::uint32_t>(engine.graph_pad_page));
            s.pad_kv_page_indptr.push_back(
                s.pad_kv_page_indptr.back() + 1);
            s.pad_kv_last_page_lens.push_back(1);
        }
        s.h_qo_forward = s.pad_qo_indptr.data();
        s.h_kvpi_forward = s.pad_kv_page_indices.data();
        s.h_kvpp_forward = s.pad_kv_page_indptr.data();
        s.h_kvlpl_forward = s.pad_kv_last_page_lens.data();
        s.forward_R = s.fR_real + s.graph_pad_requests;
        s.forward_N = s.fN_real + s.graph_pad_requests;
    }

    // Compact-logit direct rows for settlement (the epilogue slices each
    // program's logits base from the per-program gathered-row offsets).
    s.direct_logit_rows =
        reinterpret_cast<const std::uint32_t*>(s.sample_rows.data());
    if (s.compact_logits) {
        s.compact_logit_rows.resize(
            static_cast<std::size_t>(num_sampling));
        std::iota(
            s.compact_logit_rows.begin(),
            s.compact_logit_rows.end(),
            std::uint32_t{0});
        s.direct_logit_rows = s.compact_logit_rows.data();
    }
    s.settle_plain = s.rs_is_fold;
    if (dbg_fire) s.timing.prepare_end = fire_timing::Clock::now();
}

void enqueue_step(BatchEngine& engine, PreparedStep& step) {
    PreparedStep::Impl& s = *step.impl();
    const bool dbg_fire = s.timing.enabled;
    if (dbg_fire) s.timing.enqueue_start = fire_timing::Clock::now();

    auto& pi = engine.inputs;
    auto& cublas = engine.cublas;

    engine.dispatch->begin_enqueue(*s.staged);
    // Original wave order: the Prologue (begin_enqueue) runs against the
    // pre-resolution state; the resolved geometry lands on the wave only
    // now, before every later phase (attention hooks, Epilogue, settle).
    engine.dispatch->update_launch_geometry(
        *s.staged, s.dispatch_view, s.program_token_starts);
    if (s.empty_step) {
        if (dbg_fire) {
            s.timing.begin_breakdown = s.staged->begin_breakdown();
            s.timing.h2d_end = fire_timing::Clock::now();
            s.timing.forward_enqueue_end = s.timing.h2d_end;
        }
        return;
    }

    // Parameter-block commits, in the original per-fire order.
    if (s.wire_refill) {
        pi.tokens.commit_staged(s.up_tokens);
        pi.positions.commit_staged(s.up_positions);
        pi.qo_indptr.commit_staged(s.up_qo);
        pi.kv_page_indices.commit_staged(s.up_kvpi);
        pi.kv_page_indptr.commit_staged(s.up_kvpp);
        pi.kv_last_page_lens.commit_staged(s.up_kvlpl);
    }
    pi.custom_mask.commit_staged(s.up_custom_mask);
    pi.custom_mask_indptr.commit_staged(s.up_mask_indptr);
    if (s.pack_structured) {
        pi.structured_mask_klen.commit_staged(s.up_klen);
        pi.structured_masks.commit_staged(s.up_struct_masks);
        kernels::launch_pack_structured_mask(
            pi.positions.data(),
            pi.structured_mask_klen.data(),
            pi.qo_indptr.data(),
            pi.custom_mask_indptr.data(),
            pi.structured_masks.data(),
            pi.custom_mask.data(),
            s.pack_structured_lanes,
            cublas.stream());
    }
    if (s.pack_dense) {
        pi.dense_mask.commit_staged(s.up_dense_mask);
        pi.structured_mask_klen.commit_staged(s.up_dense_klen);
        pi.custom_mask_indptr.commit_staged(s.up_dense_indptr);
        CUDA_CHECK(cudaMemsetAsync(pi.custom_mask.data(), 0,
                                   s.pack_dense_bytes, cublas.stream()));
        kernels::launch_pack_dense_mask(
            pi.dense_mask.data(),
            pi.structured_mask_klen.data(),
            pi.qo_indptr.data(),
            pi.custom_mask_indptr.data(), pi.custom_mask.data(),
            s.pack_dense_lanes, s.pack_dense_stride, cublas.stream());
    }
    if (s.has_write_desc && !s.rpg.device_composed) {
        pi.w_page.commit_staged(s.up_w_page);
        pi.w_off.commit_staged(s.up_w_off);
    }
    if (s.use_slots) {
        pi.slot_ids.commit_staged(s.up_slot_ids);
        pi.is_fresh.commit_staged(s.up_is_fresh);
        pi.rs_slot_flags.commit_staged(s.up_rs_flags);
        pi.rs_fold_lens.commit_staged(s.up_rs_fold_lens);
        pi.rs_buffer_slot_indptr.commit_staged(s.up_rs_buf_indptr);
        pi.rs_buffer_slot_ids.commit_staged(s.up_rs_buf_ids);
    }
    pi.sample_idx.commit_staged(s.up_sample_idx);
    if (engine.rs_cache != nullptr) {
        engine.rs_cache->set_verify_frozen(false);
    }

    if (!s.rpg.device_composed) {
        CUDA_CHECK(cudaMemsetAsync(
            pi.row_valid.data(), 1,
            static_cast<std::size_t>(s.fN_real),
            cublas.stream()));
    }
    if (s.has_decode_envelopes) {
        std::string compose_error;
        const bool enqueued = s.use_fixed_decode
            ? engine.dispatch->enqueue_fixed_decode(
                  s.fixed_buffers, &compose_error, *s.staged)
            : engine.dispatch->enqueue_decode_envelopes(
                  s.envelope_buffers, &compose_error, *s.staged);
        if (!enqueued) {
            throw std::runtime_error(
                compose_error.empty()
                    ? "decode envelope composition is unavailable"
                    : compose_error);
        }
    }

    if (s.graph_pad_requests > 0) {
        launch_graph_pad_rows(
            reinterpret_cast<std::uint32_t*>(pi.qo_indptr.data()),
            reinterpret_cast<std::uint32_t*>(pi.kv_page_indptr.data()),
            reinterpret_cast<std::uint32_t*>(pi.kv_page_indices.data()),
            reinterpret_cast<std::uint32_t*>(pi.kv_last_page_lens.data()),
            reinterpret_cast<std::uint32_t*>(pi.tokens.data()),
            reinterpret_cast<std::uint32_t*>(pi.positions.data()),
            reinterpret_cast<std::uint8_t*>(pi.row_valid.data()),
            s.have_custom_mask ? pi.custom_mask.data() : nullptr,
            s.have_custom_mask ? pi.custom_mask_indptr.data() : nullptr,
            s.pad_real_mask_bytes,
            s.fR_real,
            s.fN_real,
            s.graph_pad_requests,
            static_cast<std::uint32_t>(engine.graph_pad_page),
            cublas.stream());
    }

    // TP fan-out. Rank 0 broadcasts the per-fire payload (header +
    // refilled persistent_inputs) to every follower; the all-reduces
    // inside `forward_fn.body` then synchronise the ranks layer-by-layer.
    if (engine.tp_comm != nullptr) {
        tp_cpu_gate_notify(engine.tp_cpu_gate_key);
        const int tp_kv_indices_count = s.rs_is_fold
            ? 0
            : static_cast<int>(s.h_kvpp_forward[s.forward_R]);
        tp_broadcast_inputs(*engine.tp_comm, pi,
                            s.forward_N, s.forward_R, s.is_pure_decode,
                            tp_kv_indices_count,
                            engine.required_kv_pages,
                            s.rs_is_fold ? 0 : s.mask_bytes,
                            s.rs_is_fold ? 0 : s.mask_indptr_count,
                            /*has_slot_ids=*/s.use_slots,
                            !s.rs_is_fold && s.has_write_desc,
                            s.compact_logits ? s.num_sampling : 0,
                            s.structured_window_left,
                            s.rs_plan.mode,
                            static_cast<int>(s.rs_fold_len_view.size()),
                            static_cast<int>(s.rs_buf_id_view.size()),
                            /*stream=*/nullptr);
    }

    // ── attention-plan hook ─────────────────────────────────────────
    // Runs at enqueue, not prepare: the plan's device commit targets the
    // single stable attention int workspace (a graph-replay invariant),
    // so it is inherently ordered between this step's neighbours on the
    // stream. Its host half is slot-ring staged and non-blocking. Hoisting
    // the host half needs a per-step plan snapshot through the model
    // layer — recorded follow-up; intra-frame decode plans are step-
    // invariant, so the end state is plan-once-per-frame.
    if (!s.rs_is_fold) {
        engine.attn_ws.begin_plan_update();
        engine.forward_fn.invoke_prepare(
            engine.attn_ws,
            ForwardFn::PrepareInputs{
                .qo_indptr_h = s.h_qo_forward,
                .kv_page_indices_h = s.h_kvpi_forward,
                .kv_page_indices_d =
                    reinterpret_cast<const std::uint32_t*>(
                        pi.kv_page_indices.data()),
                .kv_page_indptr_h = s.h_kvpp_forward,
                .kv_page_indptr_d =
                    reinterpret_cast<const std::uint32_t*>(
                        pi.kv_page_indptr.data()),
                .kv_last_page_lens_h = s.h_kvlpl_forward,
                .kv_last_page_lens_d =
                    reinterpret_cast<const std::uint32_t*>(
                        pi.kv_last_page_lens.data()),
                .total_tokens = s.forward_N,
                .num_requests = s.forward_R,
                .is_pure_decode = s.is_pure_decode,
                .have_custom_mask = s.have_custom_mask,
                .runtime_window_left = s.structured_window_left,
            });
        engine.attn_ws.end_plan_update(cublas.stream());
    }
    if (dbg_fire) s.timing.h2d_end = fire_timing::Clock::now();

    // ── Forward pass ────────────────────────────────────────────────
    StepProfileTimer verify_timer(
        "verify", cublas.stream(), s.forward_N, s.forward_R);
    if (s.ir_trace) {
        std::cerr << "[ir-trace] forward-begin req_id=0"
                  << " forward_N=" << s.forward_N
                  << " forward_R=" << s.forward_R << "\n";
        std::cerr.flush();
    }
    auto dump_rs = [&](const char* tag) {
        if (!std::getenv("PIE_RS_TRACE") || engine.rs_cache == nullptr ||
            !s.use_slots || s.R < 1) return;
        const int slot = static_cast<int>(s.rs_slot_view[0]);
        std::uint32_t rw[4] = {0, 0, 0, 0}, cw[4] = {0, 0, 0, 0};
        cudaMemcpy(rw, engine.rs_cache->recurrent_state_raw(0, slot),
                   sizeof(rw), cudaMemcpyDeviceToHost);
        cudaMemcpy(cw, engine.rs_cache->conv_state(0, slot),
                   sizeof(cw), cudaMemcpyDeviceToHost);
        std::cerr << "[rs-trace] " << tag << " slot=" << slot
                  << " bf16=" << engine.rs_cache->recurrent_state_bf16()
                  << " N=" << s.N << " rs_is_fold=" << s.rs_is_fold
                  << " rs_is_write=" << s.rs_is_write << std::hex
                  << " recur16B=" << rw[0] << "," << rw[1] << "," << rw[2] << "," << rw[3]
                  << " conv16B=" << cw[0] << "," << cw[1] << "," << cw[2] << "," << cw[3]
                  << std::dec << "\n";
    };
    dump_rs("PRE ");

    struct StageHookContext {
        pipeline::Dispatch* dispatch = nullptr;
        pipeline::StagedLaunch* launch = nullptr;
    } stage_hook_context{
        engine.dispatch,
        s.staged.get(),
    };
    const model::StageHooks stage_hooks{
        .context = &stage_hook_context,
        .execute = [](
            void* opaque,
            model::StageHookPoint point,
            const void* query_data,
            std::uint32_t query_rows,
            std::uint32_t query_columns,
            std::uint32_t layer,
            cudaStream_t stream,
            bool query_is_f32) {
            auto& context =
                *static_cast<StageHookContext*>(opaque);
            context.dispatch->execute_attention_phase(
                *context.launch,
                static_cast<std::uint8_t>(point),
                query_data,
                query_rows,
                query_columns,
                layer,
                stream,
                query_is_f32);
        },
    };
    run_forward_dispatch(
        engine, ForwardDispatchInputs{
            .forward_R = s.forward_R,
            .forward_N = s.forward_N,
            .num_sampling = s.num_sampling,
            .is_pure_decode = s.is_pure_decode,
            .have_custom_mask = s.have_custom_mask,
            .compact_logits = s.compact_logits,
            .structured_window_left = s.structured_window_left,
            .has_write_desc = s.has_write_desc,
            .use_slots = s.use_slots,
            .h_qo_forward = s.h_qo_forward,
            .h_kvpi_forward = s.h_kvpi_forward,
            .h_kvpp_forward = s.h_kvpp_forward,
            .h_kvlpl_forward = s.h_kvlpl_forward,
            .slot_ids_h_data = s.slot_ids_h.data(),
            .is_fresh_h_data = s.is_fresh_h.data(),
            .rs_slot_flags_h = s.rs_flag_view.data(),
            .rs_buffer_slot_ids_h =
                (s.rs_is_write || s.rs_is_fold)
                    ? s.rs_buf_id_view.data()
                    : nullptr,
            .rs_buffer_slot_indptr_h =
                (s.rs_is_write || s.rs_is_fold)
                    ? s.rs_buf_indptr_view.data()
                    : nullptr,
            .rs_fold_lens_h = !s.rs_fold_len_view.empty()
                ? s.rs_fold_len_view.data()
                : nullptr,
            .rs_fold_lens_d = !s.rs_fold_len_view.empty()
                ? reinterpret_cast<const std::int32_t*>(
                      pi.rs_fold_lens.data())
                : nullptr,
            .rs_buffer_write = s.rs_is_write,
            .rs_buffer_fold = s.rs_is_fold,
            .image_pixels_h = s.img_pixels_h,
            .image_pixel_byte_indptr_h = s.img_pix_byte_indptr.data(),
            .image_patch_positions_h = s.img_patch_pos.data(),
            .image_anchor_rows_h = s.img_anchor.data(),
            .num_images = s.img_num_images,
            .image_grids_h = s.img_grids.data(),
            .mrope_positions_h = s.mrope_positions_storage.empty()
                ? nullptr : s.mrope_positions_storage.data(),
            .num_mrope_positions = static_cast<int>(
                s.mrope_positions_storage.size() / 3),
            .audio_features_h = s.aud_features_h,
            .audio_feature_byte_indptr_h = s.aud_feat_byte_indptr.data(),
            .audio_anchor_rows_h = s.aud_anchor.data(),
            .num_clips = s.aud_num_clips,
            .precomputed_embeddings = s.precomputed_embeddings,
            .stage_hooks =
                s.has_attention_stages ? &stage_hooks : nullptr,
        });
    dump_rs("POST");
    if (s.ir_trace) {
        std::cerr << "[ir-trace] forward-returned req_id=0\n";
        std::cerr.flush();
    }
    verify_timer.finish(cublas.stream());
    if (dbg_fire) {
        s.timing.begin_breakdown = s.staged->begin_breakdown();
        s.timing.forward_enqueue_end = fire_timing::Clock::now();
    }
    if (!s.rs_is_fold) {
        enqueue_mtp_draft_logits(engine, s.mtp_plan);
    }
}

void settle_step(
    BatchEngine& engine,
    const PieRuntimeCallbacks& runtime,
    PieCompletion completion,
    PreparedStep& step) {
    PreparedStep::Impl& s = *step.impl();
    const bool dbg_fire = s.timing.enabled;
    s.timing.wave_id = completion.wait_id;
    const pipeline::DispatchStats stats_before = dbg_fire
        ? engine.dispatch->stats()
        : pipeline::DispatchStats{};
    if (s.settle_plain) {
        engine.dispatch->finish(
            *s.staged, s.dispatch_view, nullptr, 0,
            engine.cublas.stream(), &runtime, completion);
    } else {
        const std::uint32_t vocab = static_cast<std::uint32_t>(
            engine.loaded_model.hf_config().vocab_size);
        engine.dispatch->finish(
            *s.staged, s.dispatch_view, nullptr, vocab,
            engine.cublas.stream(),
            &runtime, completion,
            static_cast<const std::uint16_t*>(engine.ws.logits.data()),
            s.direct_logit_rows,
            s.mtp_plan.draft_starts,
            s.mtp_draft_counts,
            static_cast<std::uint32_t>(tensor_rows(engine.ws.logits)),
            engine.inputs.row_valid.data(),
            s.program_token_starts,
            dbg_fire ? &s.timing.finish_breakdown : nullptr);
    }
    if (dbg_fire) {
        s.timing.settlement_enqueue_end = fire_timing::Clock::now();
        const pipeline::DispatchStats stats_after =
            engine.dispatch->stats();
        s.timing.finish_groups = static_cast<std::int64_t>(
            stats_after.generated_fused_groups -
            stats_before.generated_fused_groups);
        s.timing.finish_grouped_lanes = static_cast<std::int64_t>(
            stats_after.grouped_lanes - stats_before.grouped_lanes);
        s.timing.finish_body_launches = static_cast<std::int64_t>(
            stats_after.grouped_body_op_launches -
            stats_before.grouped_body_op_launches);
        s.timing.finish_shared_exclusions = static_cast<std::int64_t>(
            stats_after.shared_slot_exclusions -
            stats_before.shared_slot_exclusions);
    }
}

void abort_step(BatchEngine& engine, PreparedStep& step) noexcept {
    PreparedStep::Impl* s = step.impl();
    if (s == nullptr || s->staged == nullptr) return;
    engine.dispatch->abort(*s->staged, engine.cublas.stream());
}

}  // namespace pie_cuda_driver
