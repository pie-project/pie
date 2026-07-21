#include "batch/compose.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <exception>
#include <iostream>
#include <limits>
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

}  // namespace

void handle_fire_batch(
    std::uint32_t req_id,
    const pie_native::LaunchView& view,
    BatchEngine& engine,
    const PieRuntimeCallbacks& runtime,
    PieCompletion completion)
{
    using clock = std::chrono::steady_clock;
    const bool trace_fire_timing = fire_timing::full();
    const auto t_entry = trace_fire_timing
        ? clock::now()
        : clock::time_point{};

    // Diagnostic trace for the direct launch path.
    const bool ir_trace = std::getenv("PIE_SAMPLING_IR_TRACE") != nullptr;
    if (ir_trace) {
        std::cerr << "[ir-trace] fire entry req_id=" << req_id << "\n";
        std::cerr.flush();
    }
    if (std::getenv("PIE_PTIR_TRACE")) {
        std::fprintf(stderr, "[ptir-serve] entry req_id=%u: ptir_hashes=%zu tokens=%zu\n",
                     req_id, view.ptir_program_hashes.size(), view.token_ids.size());
    }

    // Local references for the most-touched BatchEngine members.
    auto& ws                   = engine.ws;
    auto& kv_cache             = engine.kv_cache;
    auto& attn_ws              = engine.attn_ws;
    auto& cublas                = engine.cublas;
    auto& pi                   = engine.inputs;  // persistent input slabs
    auto& forward_fn           = engine.forward_fn;
    const int max_workspace_tokens = engine.max_workspace_tokens;

    // Track whether the custom-mask path was populated this fire so the
    // forward kernel knows whether to consume `pi.custom_mask`. Sizes are
    // stashed alongside so the TP broadcast knows how many bytes to fan
    // out to followers.
    bool have_custom_mask = false;
    bool has_write_desc = false;
    int mask_bytes = 0;
    int mask_indptr_count = 0;

    // Multimodal (gemma4 vision): image side-channel from the view. Declared
    // before the try so it's in scope at the forward dispatch (which is after
    // the try/catch). `image_pixels` is f32 pixel_values stored as bytes.
    const float* img_pixels_h =
        reinterpret_cast<const float*>(view.image_pixels.data());
    const auto img_pix_byte_indptr = view.image_pixel_indptr.as<std::uint32_t>();
    const auto img_patch_pos       = view.image_patch_positions.as<std::uint32_t>();
    const auto img_anchor          = view.image_anchor_rows.as<std::uint32_t>();
    const int img_num_images       = static_cast<int>(img_anchor.size());
    // Qwen3-VL M-RoPE: per-image (t,h,w) grids + per-image-token 3-axis
    // positions. Assembled into a full `[N, 3]` per-token array below.
    const auto img_grids           = view.image_grids.as<std::uint32_t>();
    const auto img_mrope_pos       = view.image_mrope_positions.as<std::uint32_t>();
    const auto img_mrope_indptr    = view.image_mrope_indptr.as<std::uint32_t>();
    // Storage for the assembled per-token [N,3] M-RoPE positions (filled
    // only when this fire carries Qwen3-VL mrope image data).
    std::vector<std::uint32_t> mrope_positions_storage;

    // Multimodal (gemma4 audio): log-mel side-channel from the view.
    // `audio_features` is f32 log-mel stored as bytes.
    const float* aud_features_h =
        reinterpret_cast<const float*>(view.audio_features.data());
    const auto aud_feat_byte_indptr = view.audio_feature_indptr.as<std::uint32_t>();
    const auto aud_anchor           = view.audio_anchor_rows.as<std::uint32_t>();
    const int aud_num_clips         = static_cast<int>(aud_anchor.size());
    const PrecomputedEmbeddingInputs precomputed_embeddings{
        .rows_h = view.embed_rows.data(),
        .byte_indptr_h = view.embed_indptr.data(),
        .shapes_h = view.embed_shapes.data(),
        .dtypes_h = view.embed_dtypes.data(),
        .anchor_rows_h = view.embed_anchor_rows.data(),
        .num_blocks = static_cast<int>(view.embed_dtypes.size()),
    };

    // Correlated host-stage timing. The scope guard also covers early returns
    // and exceptions; every record is keyed by the logical fire ids already
    // present in the launch ABI.
    int dbg_R = 0, dbg_N = 0;
    const bool dbg_fire = trace_fire_timing;
    const auto dbg_logical_fire_ids =
        view.logical_fire_ids.as<std::uint64_t>();
    struct FireTimer {
        fire_timing::Clock::time_point t0;
        fire_timing::Clock::time_point begin_end{};
        pie_cuda_driver::pipeline::StagedLaunch::BeginBreakdown begin_breakdown{};
        fire_timing::Clock::time_point resolve_end{};
        fire_timing::Clock::time_point compose_end{};
        fire_timing::Clock::time_point wire_parse_end{};
        fire_timing::Clock::time_point plan_end{};
        fire_timing::Clock::time_point h2d_end{};
        fire_timing::Clock::time_point forward_enqueue_end{};
        fire_timing::Clock::time_point settlement_enqueue_end{};
        pipeline::Dispatch::FinishBreakdown finish_breakdown{};
        std::int64_t finish_groups = -1;
        std::int64_t finish_grouped_lanes = -1;
        std::int64_t finish_body_launches = -1;
        std::int64_t finish_shared_exclusions = -1;
        const int& requests;
        const int& tokens;
        int images;
        std::uint64_t wave_id;
        std::size_t fire_count;
        std::uint64_t membership_hash;
        int uncaught;
        bool enabled;

        ~FireTimer() noexcept {
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
                    // Exclusive leg (W6 measurement fix): begin_end ->
                    // wire_parse_end. This used to span from t0 and
                    // double-counted begin_us inside it.
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
                    plan_end != fire_timing::Clock::time_point{}) {
                    output << R"(,"plan_us":)"
                           << fire_timing::duration_us(
                                  wire_parse_end, plan_end);
                }
                if (plan_end != fire_timing::Clock::time_point{} &&
                    h2d_end != fire_timing::Clock::time_point{}) {
                    output << R"(,"h2d_prepare_us":)"
                           << fire_timing::duration_us(plan_end, h2d_end);
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
                output << R"(,"host_total_us":)"
                       << fire_timing::duration_us(t0, now) << '}';
                fire_timing::write(output.str());
            } catch (...) {
            }
        }
    } dbg_ft{
        .t0 = t_entry,
        .requests = dbg_R,
        .tokens = dbg_N,
        .images = img_num_images,
        .wave_id = completion.wait_id,
        .fire_count = dbg_logical_fire_ids.size(),
        .membership_hash = dbg_fire
            ? fire_timing::membership_hash(dbg_logical_fire_ids)
            : 0,
        .uncaught = std::uncaught_exceptions(),
        .enabled = dbg_fire,
    };

    std::unique_ptr<pipeline::StagedLaunch> staged_launch;
    try {
        const auto tok_view_orig   = view.token_ids.as<std::uint32_t>();
        const auto pos_view_orig   = view.position_ids.as<std::uint32_t>();
        const auto qo_view_orig    = view.qo_indptr.as<std::uint32_t>();
        const auto kvpi_view_wire = view.kv_page_indices.as<std::uint32_t>();
        const auto kvpp_view_wire = view.kv_page_indptr.as<std::uint32_t>();
        const auto kvlpl_view_orig = view.kv_last_page_lens.as<std::uint32_t>();

        const auto sidx_view_orig  = view.sampling_indices.as<std::uint32_t>();
        const auto sptr_view_orig  = view.sampling_indptr.as<std::uint32_t>();

        staged_launch = engine.dispatch->begin(view, cublas.stream());
        if (dbg_fire) {
            dbg_ft.begin_end = clock::now();
            dbg_ft.begin_breakdown = staged_launch->begin_breakdown();
        }

        // ── W1.1: pre-forward device-geometry descriptor resolution ──────
        // EVERY device-geometry PTIR program in the batch (WSlot/WOff write
        // descriptors + a channel-bound [B, P>1] Pages port — the runtime's
        // `detect_device_geometry` mirror) ships EMPTY wire geometry; the
        // driver reads its port channels at fire time and COMPOSES the
        // resolved geometries with the wire programs' launch slices into one
        // flat forward batch (batch_compose.hpp) — no program-specific
        // assembly (owner constraint §3.1). A not-ready descriptor channel
        // fails the fire (W1.6). Pure-wire batches resolve nothing
        // (dg_resolved = false, empty *err) and use the wire geometry
        // unchanged.
        pipeline::ResolvedPrograms rpg;
        pipeline::ComposedBatch composed;
        // Per-PROGRAM offsets of each program's sampled rows within the
        // gathered logits buffer (`n_prog + 1` entries) — what
        // `Dispatch::finish` slices each program's logits base from.
        std::vector<std::uint32_t> prog_sample_csr;
        bool dg_resolved = false;
        bool composed_ready = false;
        if (!view.ptir_program_hashes.empty()) {
            std::string dg_err;
            dg_resolved = engine.dispatch->resolve_descriptors(
                view,
                static_cast<std::uint32_t>(kv_cache.page_size()),
                static_cast<std::uint32_t>(kv_cache.num_pages()),
                rpg,
                &dg_err,
                engine.forward_fn.supports_runtime_window,
                staged_launch.get(),
                engine.graph_cache != nullptr &&
                    engine.forward_fn.graph_safe &&
                    engine.tp_comm == nullptr);
            if (!dg_resolved && !dg_err.empty()) {
                throw std::runtime_error(dg_err);
            }
            if (dg_resolved) {
                // v1 mask scope: a dense device mask (AttnMask channel)
                // composes only SOLO — the runtime scheduler batches such
                // fires alone; fail loud if the contract is violated.
                if (rpg.per_program.size() > 1) {
                    for (std::size_t p = 0; p < rpg.per_program.size(); ++p) {
                        if (rpg.is_device_geometry[p] &&
                            rpg.per_program[p].has_mask) {
                            throw pipeline::RetryableLaunchError(
                                "ptir: dense device mask in a multi-program "
                                "batch requires solo retry");
                        }
                    }
                }
            }
            if (dbg_fire) dbg_ft.resolve_end = clock::now();
            std::string compose_err;
            if (!pipeline::compose_forward_batch(
                    view, rpg,
                    static_cast<std::uint32_t>(kv_cache.page_size()),
                    composed, &compose_err)) {
                throw std::runtime_error(compose_err);
            }
            composed_ready = true;
            if (dbg_fire) dbg_ft.compose_end = clock::now();
            prog_sample_csr.assign(
                composed.prog_sample_counts.size() + 1, 0);
            for (std::size_t program = 0;
                 program < composed.prog_sample_counts.size();
                 ++program) {
                prog_sample_csr[program + 1] =
                    prog_sample_csr[program] +
                    composed.prog_sample_counts[program];
            }
            if (rpg.device_composed) {
                constexpr std::uint32_t unavailable =
                    std::numeric_limits<std::uint32_t>::max();
                std::fill(
                    composed.prog_kv_lens.begin(),
                    composed.prog_kv_lens.end(),
                    unavailable);
                std::fill(
                    composed.prog_page_counts.begin(),
                    composed.prog_page_counts.end(),
                    unavailable);
                std::fill(
                    composed.prog_key_lens.begin(),
                    composed.prog_key_lens.end(),
                    unavailable);
            }
        }
        if (dbg_fire && dbg_ft.resolve_end == clock::time_point{}) {
            dbg_ft.resolve_end = clock::now();
            dbg_ft.compose_end = dbg_ft.resolve_end;
        }
        // Only the SOLO device-geometry fire may carry a dense device mask;
        // its resolved geometry equals the composed batch.
        const pipeline::FireGeometry* solo_fg =
            (dg_resolved && rpg.per_program.size() == 1 &&
             rpg.is_device_geometry[0])
                ? &rpg.per_program[0]
                : nullptr;
        int structured_window_left = -2;
        bool use_structured_mask = false;
        bool pack_structured_mask = false;
        std::vector<pie_native::ptir::StructuredMaskDescriptor>
            effective_structured_masks = composed.structured_masks;
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
            composed_ready ? composed.qo_indptr.size() : qo_view_orig.size()) - 1;

        const std::span<const std::uint32_t> tok_view   = composed_ready ? std::span<const std::uint32_t>(composed.token_ids)         : tok_view_orig;
        const std::span<const std::uint32_t> pos_view   = composed_ready ? std::span<const std::uint32_t>(composed.position_ids)      : pos_view_orig;
        const std::span<const std::uint32_t> qo_view    = composed_ready ? std::span<const std::uint32_t>(composed.qo_indptr)         : qo_view_orig;
        const std::span<const std::uint32_t> kvpi_view  = composed_ready ? std::span<const std::uint32_t>(composed.kv_page_indices)   : kvpi_view_wire;
        const std::span<const std::uint32_t> kvpp_view  = composed_ready ? std::span<const std::uint32_t>(composed.kv_page_indptr)    : kvpp_view_wire;
        const std::span<const std::uint32_t> kvlpl_view = composed_ready ? std::span<const std::uint32_t>(composed.kv_last_page_lens) : kvlpl_view_orig;
        const std::span<const std::uint32_t> sidx_view  = composed_ready ? std::span<const std::uint32_t>(composed.sampling_indices)  : sidx_view_orig;
        const std::span<const std::uint32_t> sptr_view  = composed_ready ? std::span<const std::uint32_t>(composed.sampling_indptr)   : sptr_view_orig;
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
                structured_window_left = *window;
            } else {
                pack_structured_mask = true;
            }
        }
        pie_native::LaunchView dispatch_view = view;
        if (composed_ready) {
            dispatch_view.rs_slot_ids = pie_native::slice_from_u32(
                composed.rs_slot_ids.data(), composed.rs_slot_ids.size());
            dispatch_view.rs_slot_flags = pie_native::slice_from_u8(
                composed.rs_slot_flags.data(), composed.rs_slot_flags.size());
            dispatch_view.rs_fold_lens = pie_native::slice_from_u32(
                composed.rs_fold_lens.data(), composed.rs_fold_lens.size());
            dispatch_view.rs_buffer_slot_ids = pie_native::slice_from_u32(
                composed.rs_buffer_slot_ids.data(),
                composed.rs_buffer_slot_ids.size());
            dispatch_view.rs_buffer_slot_indptr = pie_native::slice_from_u32(
                composed.rs_buffer_slot_indptr.data(),
                composed.rs_buffer_slot_indptr.size());
            dispatch_view.sampling_indices = pie_native::slice_from_u32(
                sidx_view.data(), sidx_view.size());
            dispatch_view.sampling_indptr = pie_native::slice_from_u32(
                prog_sample_csr.data(), prog_sample_csr.size());
            dispatch_view.ptir_sample_starts = pie_native::slice_from_u32(
                composed.prog_sample_starts.data(),
                composed.prog_sample_starts.size());
            dispatch_view.ptir_sample_counts = pie_native::slice_from_u32(
                composed.prog_sample_counts.data(),
                composed.prog_sample_counts.size());
            dispatch_view.ptir_row_counts = pie_native::slice_from_u32(
                composed.prog_row_counts.data(),
                composed.prog_row_counts.size());
            dispatch_view.ptir_token_counts = pie_native::slice_from_u32(
                composed.prog_token_counts.data(),
                composed.prog_token_counts.size());
            dispatch_view.ptir_kv_lens = pie_native::slice_from_u32(
                composed.prog_kv_lens.data(),
                composed.prog_kv_lens.size());
            dispatch_view.ptir_page_counts = pie_native::slice_from_u32(
                composed.prog_page_counts.data(),
                composed.prog_page_counts.size());
            dispatch_view.ptir_query_lens = pie_native::slice_from_u32(
                composed.prog_query_lens.data(),
                composed.prog_query_lens.size());
            dispatch_view.ptir_key_lens = pie_native::slice_from_u32(
                composed.prog_key_lens.data(),
                composed.prog_key_lens.size());
        }
        std::vector<std::uint32_t> program_token_starts(
            view.ptir_program_hashes.size(), 0);
        if (composed_ready) {
            for (std::size_t program = 0;
                 program < program_token_starts.size();
                 ++program) {
                const std::uint32_t request =
                    composed.prog_request_starts[program];
                if (request >= composed.qo_indptr.size()) {
                    throw std::runtime_error(
                        "PTIR program token attribution is outside composed geometry");
                }
                program_token_starts[program] =
                    composed.qo_indptr[request];
            }
        }
        engine.dispatch->update_launch_geometry(
            *staged_launch, dispatch_view, program_token_starts);
        const std::vector<std::uint32_t> mtp_draft_counts =
            engine.dispatch->mtp_draft_rows(dispatch_view);

        const auto t_wire_parse_end = dbg_fire
            ? clock::now()
            : clock::time_point{};
        dbg_ft.wire_parse_end = t_wire_parse_end;

        const int N = static_cast<int>(tok_view.size());
        const int num_sampling = static_cast<int>(sidx_view.size());
        dbg_R = R; dbg_N = N;
        if (ir_trace) {
            std::cerr << "[ir-trace] fire shape req_id=" << req_id
                      << " N=" << N << " R=" << R
                      << " num_sampling=" << num_sampling << "\n";
            std::cerr.flush();
        }

        // Qwen3-VL: assemble the per-token [N,3] M-RoPE positions. Text rows
        // carry (p,p,p) from the 1-D `pos_view`; image-token rows are
        // overwritten with the staged 3-axis (t,h,w) positions for each image
        // (image i's rows start at batch row `img_anchor[i]`). Built only when
        // image mrope data is present (image prefills never carry spec drafts).
        if (img_num_images > 0 && !img_mrope_pos.empty()) {
            mrope_positions_storage.resize(static_cast<std::size_t>(N) * 3);
            for (int t = 0; t < N; ++t) {
                const std::uint32_t p =
                    t < static_cast<int>(pos_view.size()) ? pos_view[t] : 0u;
                mrope_positions_storage[3 * t + 0] = p;
                mrope_positions_storage[3 * t + 1] = p;
                mrope_positions_storage[3 * t + 2] = p;
            }
            for (int im = 0; im < img_num_images; ++im) {
                const std::uint32_t anchor_row = img_anchor[im];
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
                    mrope_positions_storage[3 * row + 0] = img_mrope_pos[lo + 3 * j + 0];
                    mrope_positions_storage[3 * row + 1] = img_mrope_pos[lo + 3 * j + 1];
                    mrope_positions_storage[3 * row + 2] = img_mrope_pos[lo + 3 * j + 2];
                }
            }
        }

        // Detect pure decode so the model can choose its decode kernel.
        const std::uint32_t* h_kvpp  = kvpp_view.data();
        const std::uint32_t* h_kvlpl = kvlpl_view.data();
        const std::uint32_t* h_qo    = qo_view.data();
        bool is_pure_decode = (R > 0);
        for (int r = 0; r < R; ++r) {
            if (h_qo[r + 1] - h_qo[r] != 1u) is_pure_decode = false;
        }

        const std::span<const std::uint32_t> rs_slot_view = composed_ready
            ? std::span<const std::uint32_t>(composed.rs_slot_ids)
            : view.rs_slot_ids.as<std::uint32_t>();
        const std::span<const std::uint8_t> rs_flag_view = composed_ready
            ? std::span<const std::uint8_t>(composed.rs_slot_flags)
            : view.rs_slot_flags.as<std::uint8_t>();
        const std::span<const std::uint32_t> rs_fold_len_view =
            composed_ready
                ? std::span<const std::uint32_t>(composed.rs_fold_lens)
                : view.rs_fold_lens.as<std::uint32_t>();
        std::string rs_binding_error;
        if (!pipeline::validate_folded_rs_bindings(
                rs_slot_view,
                rs_flag_view,
                rs_fold_len_view,
                static_cast<std::size_t>(std::max(R, 0)),
                engine.rs_cache != nullptr,
                &rs_binding_error)) {
            throw std::runtime_error(rs_binding_error);
        }
        const bool use_slots = !rs_slot_view.empty();
        // Ph7 RS working-set buffered-activation channel. Single-role per pass
        // (v1): a FOLD pass (FOLD-bit=2 set) gathers+replays from the buffered
        // pool into recurrent_state (separate fold-replay dispatch below); an
        // rs-output write pass (FOLD-bit clear + buffered slabs present)
        // scatters in-proj [mixed_qkv|a|b] to the pool during the main forward.
        const std::span<const std::uint32_t> rs_buf_id_view = composed_ready
            ? std::span<const std::uint32_t>(composed.rs_buffer_slot_ids)
            : view.rs_buffer_slot_ids.as<std::uint32_t>();
        const std::span<const std::uint32_t> rs_buf_indptr_view =
            composed_ready
                ? std::span<const std::uint32_t>(
                      composed.rs_buffer_slot_indptr)
                : view.rs_buffer_slot_indptr.as<std::uint32_t>();
        pipeline::RsExecutionPlan rs_plan;
        if (!pipeline::plan_rs_execution(
                rs_slot_view,
                rs_flag_view,
                rs_fold_len_view,
                rs_buf_id_view,
                rs_buf_indptr_view,
                qo_view,
                engine.rs_cache != nullptr,
                engine.rs_cache != nullptr &&
                    engine.rs_cache->rs_buffer_pool_enabled(),
                engine.rs_cache != nullptr
                    ? static_cast<std::uint32_t>(
                          engine.rs_cache->rs_buffer_page_tokens())
                    : 0,
                rs_plan,
                &rs_binding_error)) {
            throw std::runtime_error(rs_binding_error);
        }
        const bool rs_is_fold =
            rs_plan.mode == RsExecutionMode::BufferFold;
        const bool rs_is_write =
            rs_plan.mode == RsExecutionMode::BufferWrite;
        if (rs_is_fold && !sidx_view.empty()) {
            throw std::runtime_error(
                "buffered RS fold is state-only and cannot sample logits");
        }

        // Anatomical stages run through explicit model hooks. Boundary-only
        // PTIR stays outside the captured model body, whose decode geometry may
        // be padded to a reusable graph bucket below.
        ForwardInputViews forward_inputs = make_forward_input_views(
            tok_view, pos_view, qo_view, kvpi_view, kvpp_view, kvlpl_view, R);
        if (rs_is_fold) {
            forward_inputs.qo_indptr =
                std::span<const std::uint32_t>(rs_plan.fold_qo_indptr);
            forward_inputs.total_tokens =
                static_cast<int>(rs_plan.fold_tokens);
        }
        int forward_N = forward_inputs.total_tokens;
        int forward_R = forward_inputs.num_requests;
        const std::uint32_t* h_qo_forward =
            forward_inputs.qo_indptr.data();
        const std::uint32_t* h_kvpi_forward =
            forward_inputs.kv_page_indices.data();
        const std::uint32_t* h_kvpp_forward =
            forward_inputs.kv_page_indptr.data();
        const std::uint32_t* h_kvlpl_forward =
            forward_inputs.kv_last_page_lens.data();
        if (forward_N == 0 || forward_R <= 0) {
            engine.dispatch->finish(
                *staged_launch, dispatch_view, nullptr, 0,
                engine.cublas.stream(), &runtime, completion);
            if (dbg_fire) {
                dbg_ft.settlement_enqueue_end = clock::now();
            }
            return;
        }
        if (forward_N > max_workspace_tokens) {
            std::cerr << "[pie-driver-cuda] batch tokens=" << forward_N
                      << " exceeds workspace=" << max_workspace_tokens << "\n";
            throw std::runtime_error("forward batch exceeds workspace capacity");
        }
        if (rs_is_fold) {
            is_pure_decode = std::all_of(
                rs_fold_len_view.begin(), rs_fold_len_view.end(),
                [](std::uint32_t length) { return length == 1; });
        }

        // Refill persistent device buffers with this fire's wire inputs.
        // Same device addresses every fire — required for graph-replay
        // safety; cheap (single async memcpy each) on its own.
        if (!rpg.device_composed) {
            pi.tokens.copy_from_host(forward_inputs.tokens);
            pi.positions.copy_from_host(forward_inputs.positions);
            pi.qo_indptr.copy_from_host(forward_inputs.qo_indptr);
            pi.kv_page_indices.copy_from_host(
                forward_inputs.kv_page_indices);
            pi.kv_page_indptr.copy_from_host(
                forward_inputs.kv_page_indptr);
            pi.kv_last_page_lens.copy_from_host(
                forward_inputs.kv_last_page_lens);
        }

        // BRLE attention masks. For any batch that isn't pure causal, decode +
        // upload a packed bitmap and route through the flashinfer kCustom path.
        // NOTE: `is_pure_decode` is intentionally NOT gated here. A decode-shaped
        // batch (qo_len==1/req) that carries a per-cell custom mask — e.g. the
        // §6.2 beam fire, whose kvm expresses fork-freeze mid-page holes the
        // decode/xqa kernels can't — must ALSO build the mask; the forward then
        // routes it through the custom-mask prefill kernel (llama_like's
        // `has_custom_mask` gate). This is the previously-noted "route decode
        // through the prefill kernel for custom-mask inferlets" fix; a normal
        // decode batch carries no mask (`fmask_view` empty) so it is unaffected.
        // Wire BRLE masks are indexed by the WIRE request layout, so the
        // causality check and decode run against the WIRE spans (identical to
        // the selected spans on a pure-wire batch). A composed batch never
        // carries a custom wire mask (the runtime scheduler keeps custom-mask
        // wire fires and device-geometry fires apart — fail loud otherwise);
        // pure-causal wire masks are simply dropped, as before.
        const auto fmask_view  = view.flattened_masks.as<std::uint32_t>();
        const auto mskptr_view = view.mask_indptr.as<std::uint32_t>();
        if (!fmask_view.empty()) {
            const auto qo_span = std::span<const std::uint32_t>(
                qo_view_orig.data(), qo_view_orig.size());
            const auto kvpp_span = std::span<const std::uint32_t>(
                kvpp_view_wire.data(), kvpp_view_wire.size());
            const auto kvlpl_span = std::span<const std::uint32_t>(
                kvlpl_view_orig.data(), kvlpl_view_orig.size());
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
                if (dg_resolved) {
                    // A MULTI-program batch cannot honor wire BRLE masks
                    // (they index the wire request layout; the scheduler
                    // batches mask-carrying fires solo — fail loud if not).
                    // On a SOLO device-geometry fire the wire rows are
                    // engine-SYNTHESIZED causal (a guest's mask is the DENSE
                    // channel mask, packed below) and simply drop — the
                    // resolved geometry runs the standard causal path.
                    if (view.has_user_mask &&
                        view.ptir_program_hashes.size() > 1) {
                        throw std::runtime_error(
                            "ptir: custom wire masks cannot co-batch with "
                            "device-geometry programs (scheduler contract "
                            "violated)");
                    }
                } else {
                    auto decoded = pie_cuda_driver::brle::decode(
                        fmask_view, mskptr_view,
                        qo_span, kvpp_span, kvlpl_span,
                        kv_cache.page_size());
                    pi.custom_mask.copy_from_host(
                        std::span<const std::uint8_t>(decoded.packed));
                    pi.custom_mask_indptr.copy_from_host(
                        std::span<const std::int32_t>(decoded.mask_indptr));
                    mask_bytes = static_cast<int>(decoded.packed.size());
                    mask_indptr_count = static_cast<int>(decoded.mask_indptr.size());
                    have_custom_mask = true;
                }
            }
        }

        if (pack_structured_mask && !have_custom_mask) {
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
            pi.structured_mask_klen.copy_from_host(klen);
            pi.structured_masks.copy_from_host(masks);
            pi.custom_mask_indptr.copy_from_host(mindptr);
            kernels::launch_pack_structured_mask(
                pi.positions.data(),
                pi.structured_mask_klen.data(),
                pi.qo_indptr.data(),
                pi.custom_mask_indptr.data(),
                pi.structured_masks.data(),
                pi.custom_mask.data(),
                lanes,
                cublas.stream());
            have_custom_mask = true;
            mask_bytes = static_cast<int>(packed_bytes);
            mask_indptr_count = lanes + 1;
        }

        // ── W1.3: device-geometry AttnMask → FlashInfer packed custom mask ──
        // A device-geometry fire may carry a DENSE [lanes, stride] per-cell mask
        // on its AttnMask descriptor port (resolved into fg.mask). Pack it to
        // FlashInfer's bit-packed custom mask (launch_pack_dense_mask) INTO
        // pi.custom_mask, so the standard custom-mask forward path consumes it
        // exactly like a BRLE-decoded wire mask. DORMANT unless a device-geometry
        // program binds an AttnMask channel (fg.has_mask); the guest producer is
        // W2.1. Correctness is validated once a real device-geometry fire exists.
        if (solo_fg != nullptr && solo_fg->has_mask &&
            !solo_fg->mask.empty() && !use_structured_mask) {
            const pipeline::FireGeometry& fg = *solo_fg;
            const int lanes = static_cast<int>(qo_view.size()) - 1;
            // Total query rows = qo_indptr.back(). For a 1-query/lane decode this
            // equals `lanes`; for a variable-length prefill a single lane carries
            // N query rows, so the dense mask is [TOTAL_Q, STRIDE] (one row per
            // QUERY token), STRIDE = mask.size()/TOTAL_Q.
            const int total_q =
                lanes > 0 ? static_cast<int>(qo_view[lanes]) : 0;
            if (lanes > 0 && total_q > 0 &&
                fg.mask.size() % static_cast<std::size_t>(total_q) == 0) {
                const int stride =
                    static_cast<int>(fg.mask.size() / static_cast<std::size_t>(total_q));
                const std::uint32_t page =
                    static_cast<std::uint32_t>(kv_cache.page_size());
                // Per-lane physical KV span klen[l] from the resolved page geometry,
                // and the packed byte-offset CSR (ceil(qo_len[l]·klen[l]/8) per lane).
                std::vector<std::uint32_t> klen(static_cast<std::size_t>(lanes), 0);
                std::vector<std::int32_t> mindptr(static_cast<std::size_t>(lanes) + 1, 0);
                for (int l = 0; l < lanes; ++l) {
                    const std::uint32_t np =
                        (l + 1 < static_cast<int>(fg.kv_page_indptr.size()))
                            ? fg.kv_page_indptr[l + 1] - fg.kv_page_indptr[l] : 0u;
                    const std::uint32_t lpl =
                        (l < static_cast<int>(fg.kv_last_page_lens.size()))
                            ? fg.kv_last_page_lens[l] : 0u;
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
                    static_cast<std::size_t>(lanes) + 1 <= pi.custom_mask_indptr.size()) {
                    auto kvm_dev = DeviceBuffer<std::uint8_t>::from_bytes(
                        std::span<const std::uint8_t>(fg.mask));
                    auto klen_dev = DeviceBuffer<std::uint32_t>::from_host(
                        std::span<const std::uint32_t>(klen));
                    auto qo_dev = DeviceBuffer<std::uint32_t>::from_host(
                        std::span<const std::uint32_t>(qo_view.data(), qo_view.size()));
                    pi.custom_mask_indptr.copy_from_host(
                        std::span<const std::int32_t>(mindptr));
                    CUDA_CHECK(cudaMemsetAsync(pi.custom_mask.data(), 0,
                                               packed_bytes, cublas.stream()));
                    kernels::launch_pack_dense_mask(
                        kvm_dev.data(), klen_dev.data(), qo_dev.data(),
                        pi.custom_mask_indptr.data(), pi.custom_mask.data(),
                        lanes, stride, cublas.stream());
                    have_custom_mask = true;
                    mask_bytes = static_cast<int>(packed_bytes);
                    mask_indptr_count = lanes + 1;
                }
            }
        }

        // Explicit KV-write descriptor upload (device-geometry WSlot/WOff, B2).
        // Parallels the mask pack above: when any composed program bound
        // WSlot/WOff ports, the composition carries per-TOKEN physical page
        // ids + offsets for EVERY batch row (device-geometry rows from their
        // translated descriptors; wire rows synthesized to their standard
        // append target — `has_write_desc` routes the whole forward's
        // per-layer KV append through launch_write_kv_explicit_bf16, so every
        // row needs a target). Beam fork/freeze correctness: a frozen fork's
        // cell is not overwritten (a sibling's mask hides it).
        if (dg_resolved && composed.has_write_desc && !composed.w_page.empty()) {
            if (composed.w_page.size() != composed.w_off.size() ||
                composed.w_page.size() > pi.w_page.size()) {
                throw std::runtime_error(
                    "ptir: composed write descriptor exceeds persistent "
                    "input capacity");
            }
            if (!rpg.device_composed) {
                pi.w_page.copy_from_host(
                    std::span<const std::uint32_t>(composed.w_page));
                pi.w_off.copy_from_host(
                    std::span<const std::uint32_t>(composed.w_off));
            }
            has_write_desc = true;
        }

        // Graph-lattice padding decision (V6 iterations 8–9). The prewarm
        // captures forward graphs for every request-lattice bucket, but an
        // off-lattice pure-decode width (a cohort swap's 46, 53, …) used to
        // miss the cache and capture a throwaway single-use graph (~1 ms on
        // the submit thread; ~50 per c64 run). Eligible waves now pad to
        // `forward_graph_request_bucket`. Pad lanes are in-band-invalid
        // (`row_valid = 0` — the KV-write kernels in split_packed.cu skip
        // them) and their device CSR rows are written COHERENTLY by the
        // pad-fill kernel below; stale device rows are NOT acceptable for
        // CSRs (a stale indptr below the real total wraps the per-row page
        // count negative — the iteration-8 hang). Shares
        // run_forward_dispatch's eligibility predicate so padded waves
        // always match cached keys.
        const bool has_attention_stages =
            engine.dispatch->launch_has_attention_stages(dispatch_view);
        int graph_pad_requests = 0;
        {
            // `is_fresh` is passed as nullptr: for RS models the shared
            // predicate then answers conservatively (no padding, exact-shape
            // keys — exactly today's behavior), while every non-RS model
            // pads. RS-model padding can follow once RS graphs matter.
            const bool eligible = forward_graph_replay_eligible(
                engine,
                is_pure_decode,
                have_custom_mask,
                rs_is_write,
                rs_is_fold,
                has_write_desc,
                structured_window_left,
                use_slots,
                nullptr,
                forward_R,
                img_num_images,
                aud_num_clips,
                has_attention_stages);
            if (eligible && engine.graph_pad_page >= 0) {
                const int max_requests = std::min(
                    engine.max_forward_requests, engine.max_workspace_tokens);
                const int bucket =
                    forward_graph_request_bucket(forward_R, max_requests);
                const int padding = bucket - forward_R;
                const std::size_t padded_tokens =
                    static_cast<std::size_t>(forward_N) +
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
                        pi.kv_last_page_lens.size();
                if (fits) {
                    graph_pad_requests = padding;
                }
            }
        }

        // Linear-attention rs_cache slots. Runtime owns slot assignment;
        // RS-capable models must receive one slot id per request.
        std::vector<std::int32_t> slot_ids_h;
        std::vector<std::uint8_t> is_fresh_h;
        if (use_slots) {
            const int slot_count = R;
            if (rs_flag_view.size() > pi.rs_slot_flags.size() ||
                rs_fold_len_view.size() > pi.rs_fold_lens.size() ||
                rs_buf_id_view.size() > pi.rs_buffer_slot_ids.size() ||
                rs_buf_indptr_view.size() >
                    pi.rs_buffer_slot_indptr.size()) {
                throw std::runtime_error(
                    "RS metadata exceeds persistent input capacity");
            }
            slot_ids_h.resize(slot_count);
            is_fresh_h.resize(slot_count);
            for (int r = 0; r < R; ++r) {
                slot_ids_h[r] = static_cast<std::int32_t>(rs_slot_view[r]);
                is_fresh_h[r] = (rs_flag_view[r] & PIE_RS_FLAG_RESET)
                                    ? 1u
                                    : 0u;
            }
            for (int r = R; r < slot_count; ++r) {
                slot_ids_h[r] = engine.graph_pad_slot;
                is_fresh_h[r] = 0u;
            }
            pi.slot_ids.copy_from_host(std::span<const std::int32_t>(slot_ids_h));
            pi.is_fresh.copy_from_host(std::span<const std::uint8_t>(is_fresh_h));
            std::copy(
                rs_flag_view.begin(), rs_flag_view.end(),
                pi.rs_slot_flags_host.data());
            pi.rs_slot_flags.copy_from_host(
                std::span<const std::uint8_t>(
                    pi.rs_slot_flags_host.data(), rs_flag_view.size()));
            if (!rs_fold_len_view.empty()) {
                std::copy(
                    rs_fold_len_view.begin(), rs_fold_len_view.end(),
                    pi.rs_fold_lens_host.data());
                pi.rs_fold_lens.copy_from_host(
                    std::span<const std::uint32_t>(
                        pi.rs_fold_lens_host.data(),
                        rs_fold_len_view.size()));
            }
            if (!rs_buf_indptr_view.empty()) {
                std::copy(
                    rs_buf_indptr_view.begin(), rs_buf_indptr_view.end(),
                    pi.rs_buffer_slot_indptr_host.data());
                pi.rs_buffer_slot_indptr.copy_from_host(
                    std::span<const std::uint32_t>(
                        pi.rs_buffer_slot_indptr_host.data(),
                        rs_buf_indptr_view.size()));
            }
            if (!rs_buf_id_view.empty()) {
                std::copy(
                    rs_buf_id_view.begin(), rs_buf_id_view.end(),
                    pi.rs_buffer_slot_ids_host.data());
                pi.rs_buffer_slot_ids.copy_from_host(
                    std::span<const std::uint32_t>(
                        pi.rs_buffer_slot_ids_host.data(),
                        rs_buf_id_view.size()));
            }
        }

        if (!rs_is_fold &&
            (sptr_view.size() != static_cast<std::size_t>(R + 1) ||
             sptr_view.back() != sidx_view.size())) {
            throw std::runtime_error("sampling CSR does not match launched instances");
        }
        std::vector<std::int32_t> sample_rows;
        std::string sampling_error;
        if (!rs_is_fold && !pipeline::global_sampling_rows(
                qo_view,
                sptr_view,
                sidx_view,
                sample_rows,
                &sampling_error)) {
            throw std::runtime_error(sampling_error);
        }
        if (sample_rows.size() > pi.sample_idx.size()) {
            throw std::runtime_error("sampling rows exceed persistent input capacity");
        }
        if (!rs_is_fold && N > tensor_rows(ws.logits)) {
            throw std::runtime_error("forward batch exceeds logits workspace");
        }
        if (!sample_rows.empty() && !rpg.device_composed) {
            pi.sample_idx.copy_from_host(
                std::span<const std::int32_t>(sample_rows));
        }
        const MtpDraftPlan mtp_draft_plan =
            preflight_mtp_draft_logits(
                engine, composed, sample_rows, mtp_draft_counts);
        if (rs_is_fold && !mtp_draft_plan.work.empty()) {
            throw std::runtime_error(
                "state-only buffered RS fold cannot produce MTP drafts");
        }
        if (engine.rs_cache != nullptr) {
            engine.rs_cache->set_verify_frozen(false);
        }

        const auto t_plan_end = dbg_fire
            ? clock::now()
            : clock::time_point{};
        dbg_ft.plan_end = t_plan_end;
        if (rs_is_fold && has_attention_stages) {
            throw std::runtime_error(
                "state-only buffered RS fold cannot execute anatomical PTIR "
                "attention stages");
        }
        if (!tp_rs_metadata_shape_valid(
                rs_plan.mode,
                static_cast<std::size_t>(forward_R),
                rs_slot_view.size(),
                rs_flag_view.size(),
                rs_fold_len_view.size(),
                (rs_is_write || rs_is_fold) ? rs_buf_id_view.size() : 0,
                (rs_is_write || rs_is_fold)
                    ? rs_buf_indptr_view.size()
                    : 0)) {
            throw std::runtime_error(
                "RS metadata cannot be represented by the TP payload");
        }

        if (!rpg.device_composed) {
            CUDA_CHECK(cudaMemsetAsync(
                pi.row_valid.data(), 1,
                static_cast<std::size_t>(forward_N),
                cublas.stream()));
        }
        const bool has_decode_envelopes = composed_ready &&
            engine.dispatch->has_decode_envelopes(dispatch_view);
        if (has_decode_envelopes) {
            bool enqueued = false;
            if (engine.graph_pad_page >= 0 && rpg.device_composed) {
                std::string fixed_error;
                enqueued = engine.dispatch->enqueue_fixed_decode(
                    dispatch_view,
                    static_cast<std::uint32_t>(kv_cache.page_size()),
                    static_cast<std::uint32_t>(kv_cache.num_pages()),
                    pipeline::FixedDecodeDeviceBuffers{
                        .token_ids = pi.tokens.data(),
                        .position_ids = pi.positions.data(),
                        .qo_indptr = pi.qo_indptr.data(),
                        .kv_page_indices = pi.kv_page_indices.data(),
                        .kv_page_indptr = pi.kv_page_indptr.data(),
                        .kv_last_page_lens =
                            pi.kv_last_page_lens.data(),
                        .w_page = pi.w_page.data(),
                        .w_off = pi.w_off.data(),
                        .row_valid = pi.row_valid.data(),
                        .rs_slot_ids =
                            use_slots ? pi.slot_ids.data() : nullptr,
                        .sample_indices = pi.sample_idx.data(),
                        .token_capacity = pi.tokens.size(),
                        .request_capacity =
                            pi.kv_last_page_lens.size(),
                        .page_capacity = pi.kv_page_indices.size(),
                        .dummy_page = static_cast<std::uint32_t>(
                            engine.graph_pad_page),
                    },
                    &fixed_error,
                    *staged_launch);
                if (!enqueued && !fixed_error.empty()) {
                    throw std::runtime_error(fixed_error);
                }
            } else if (engine.graph_pad_page >= 0) {
                std::string envelope_error;
                enqueued = engine.dispatch->enqueue_decode_envelopes(
                    dispatch_view,
                    program_token_starts,
                    composed.prog_request_starts,
                    std::span<const std::uint32_t>(
                        h_kvpp_forward,
                        static_cast<std::size_t>(forward_R) + 1),
                    pipeline::DecodeEnvelopeDeviceBuffers{
                        .token_ids = pi.tokens.data(),
                        .position_ids = pi.positions.data(),
                        .kv_page_indices = pi.kv_page_indices.data(),
                        .kv_page_indptr = pi.kv_page_indptr.data(),
                        .kv_last_page_lens =
                            pi.kv_last_page_lens.data(),
                        .row_valid = pi.row_valid.data(),
                        .rs_slot_ids =
                            use_slots ? pi.slot_ids.data() : nullptr,
                        .dummy_page = static_cast<std::uint32_t>(
                            engine.graph_pad_page),
                        .page_size = static_cast<std::uint32_t>(
                            kv_cache.page_size()),
                    },
                    &envelope_error,
                    *staged_launch);
                if (!enqueued && !envelope_error.empty()) {
                    throw std::runtime_error(envelope_error);
                }
            }
            if (!enqueued) {
                throw std::runtime_error(
                    "decode envelope composition is unavailable");
            }
        }

        // Plan attention from the WS envelope upper bound, never from
        // placeholder wire geometry: device-resolved lanes carry their real
        // KV lengths only on device, so XQA bucket selection and FlashInfer
        // plans must cover the host-known reserved envelope or long-KV
        // attention truncates silently.
        std::vector<std::uint32_t> plan_page_counts;
        std::vector<std::uint32_t> plan_kv_page_indptr;
        std::vector<std::uint32_t> plan_kv_last_lens;
        if (has_decode_envelopes &&
            engine.dispatch->envelope_plan_page_bounds(
                dispatch_view,
                composed.prog_request_starts,
                std::span<const std::uint32_t>(
                    h_kvpp_forward,
                    static_cast<std::size_t>(forward_R) + 1),
                plan_page_counts)) {
            const auto page = static_cast<std::uint32_t>(
                kv_cache.page_size());
            plan_kv_page_indptr.assign(
                static_cast<std::size_t>(forward_R) + 1, 0);
            for (int request = 0; request < forward_R; ++request) {
                plan_kv_page_indptr[request + 1] =
                    plan_kv_page_indptr[request] +
                    plan_page_counts[request];
            }
            plan_kv_last_lens.assign(
                static_cast<std::size_t>(forward_R), page);
            h_kvpp_forward = plan_kv_page_indptr.data();
            h_kvlpl_forward = plan_kv_last_lens.data();
        }

        // Apply the graph-lattice padding decided above. Padded COPIES of
        // the host CSRs feed the flashinfer plan / TP broadcast / graph
        // key, while the pad-fill kernel writes the pad lanes' device rows
        // coherently — the page CSR continues from the DEVICE-resident
        // total, which for a device-composed wave the host does not know
        // (a host-side copy left those rows stale and hung the attention
        // kernel; V6 iteration 8). Runs on the launch stream after every
        // real-row write, before the forward.
        std::vector<std::uint32_t> pad_qo_indptr;
        std::vector<std::uint32_t> pad_kv_page_indices;
        std::vector<std::uint32_t> pad_kv_page_indptr;
        std::vector<std::uint32_t> pad_kv_last_page_lens;
        if (graph_pad_requests > 0) {
            pad_qo_indptr.assign(
                h_qo_forward, h_qo_forward + forward_R + 1);
            pad_kv_page_indptr.assign(
                h_kvpp_forward, h_kvpp_forward + forward_R + 1);
            const std::size_t pages = pad_kv_page_indptr.back();
            pad_kv_page_indices.assign(
                h_kvpi_forward, h_kvpi_forward + pages);
            pad_kv_last_page_lens.assign(
                h_kvlpl_forward, h_kvlpl_forward + forward_R);
            for (int r = 0; r < graph_pad_requests; ++r) {
                pad_qo_indptr.push_back(pad_qo_indptr.back() + 1);
                pad_kv_page_indices.push_back(
                    static_cast<std::uint32_t>(engine.graph_pad_page));
                pad_kv_page_indptr.push_back(
                    pad_kv_page_indptr.back() + 1);
                pad_kv_last_page_lens.push_back(1);
            }
            launch_graph_pad_rows(
                reinterpret_cast<std::uint32_t*>(pi.qo_indptr.data()),
                reinterpret_cast<std::uint32_t*>(pi.kv_page_indptr.data()),
                reinterpret_cast<std::uint32_t*>(pi.kv_page_indices.data()),
                reinterpret_cast<std::uint32_t*>(pi.kv_last_page_lens.data()),
                reinterpret_cast<std::uint32_t*>(pi.tokens.data()),
                reinterpret_cast<std::uint32_t*>(pi.positions.data()),
                reinterpret_cast<std::uint8_t*>(pi.row_valid.data()),
                forward_R,
                forward_N,
                graph_pad_requests,
                static_cast<std::uint32_t>(engine.graph_pad_page),
                cublas.stream());
            h_qo_forward = pad_qo_indptr.data();
            h_kvpi_forward = pad_kv_page_indices.data();
            h_kvpp_forward = pad_kv_page_indptr.data();
            h_kvlpl_forward = pad_kv_last_page_lens.data();
            forward_R += graph_pad_requests;
            forward_N += graph_pad_requests;
        }

        // TP fan-out. Rank 0 broadcasts the per-fire payload (header +
        // refilled persistent_inputs) to every follower so they can run
        // the same forward kernels against an identical view of inputs.
        // The all-reduces inside `forward_fn.body` then synchronise the
        // ranks layer-by-layer. The header includes the forward variant so
        // CUDA graph capture/replay stays lockstep across ranks.
        if (engine.tp_comm != nullptr) {
            tp_cpu_gate_notify(engine.tp_cpu_gate_key);
            const int tp_kv_indices_count = rs_is_fold
                ? 0
                : static_cast<int>(h_kvpp_forward[forward_R]);
            tp_broadcast_inputs(*engine.tp_comm, pi,
                                forward_N, forward_R, is_pure_decode,
                                tp_kv_indices_count,
                                engine.required_kv_pages,
                                rs_is_fold ? 0 : mask_bytes,
                                rs_is_fold ? 0 : mask_indptr_count,
                                /*has_slot_ids=*/use_slots,
                                !rs_is_fold && has_write_desc,
                                /*logit_rows=*/0,
                                structured_window_left,
                                rs_plan.mode,
                                static_cast<int>(rs_fold_len_view.size()),
                                static_cast<int>(rs_buf_id_view.size()),
                                /*stream=*/nullptr);
        }

        // ── prepare hook ────────────────────────────────────────
        // Always run the per-arch prepare phase first (when present).
        // For graph-capable archs this updates pinned host / device
        // plan state for the captured body to read. Lives outside any
        // capture region so the host work re-runs every fire.
        if (!rs_is_fold) {
            attn_ws.begin_plan_update();
            forward_fn.invoke_prepare(
                attn_ws,
                ForwardFn::PrepareInputs{
                    .qo_indptr_h = h_qo_forward,
                    .kv_page_indices_h = h_kvpi_forward,
                    .kv_page_indices_d =
                        reinterpret_cast<const std::uint32_t*>(
                            pi.kv_page_indices.data()),
                    .kv_page_indptr_h = h_kvpp_forward,
                    .kv_page_indptr_d =
                        reinterpret_cast<const std::uint32_t*>(
                            pi.kv_page_indptr.data()),
                    .kv_last_page_lens_h =
                        h_kvlpl_forward,
                    .kv_last_page_lens_d =
                        reinterpret_cast<const std::uint32_t*>(
                            pi.kv_last_page_lens.data()),
                    .total_tokens = forward_N,
                    .num_requests = forward_R,
                    .is_pure_decode = is_pure_decode,
                    .runtime_window_left = structured_window_left,
                });
            attn_ws.end_plan_update(cublas.stream());
        }

        const auto t_h2d_end = dbg_fire
            ? clock::now()
            : clock::time_point{};
        dbg_ft.h2d_end = t_h2d_end;

        // ── Forward pass ────────────────────────────────────────
        StepProfileTimer verify_timer(
            "verify", cublas.stream(), forward_N, forward_R);
        if (ir_trace) {
            std::cerr << "[ir-trace] forward-begin req_id=" << req_id
                      << " forward_N=" << forward_N
                      << " forward_R=" << forward_R << "\n";
            std::cerr.flush();
        }
        auto dump_rs = [&](const char* tag) {
            if (!std::getenv("PIE_RS_TRACE") || engine.rs_cache == nullptr ||
                !use_slots || R < 1) return;
            const int slot = static_cast<int>(rs_slot_view[0]);
            std::uint32_t rw[4] = {0, 0, 0, 0}, cw[4] = {0, 0, 0, 0};
            cudaMemcpy(rw, engine.rs_cache->recurrent_state_raw(0, slot),
                       sizeof(rw), cudaMemcpyDeviceToHost);
            cudaMemcpy(cw, engine.rs_cache->conv_state(0, slot),
                       sizeof(cw), cudaMemcpyDeviceToHost);
            std::cerr << "[rs-trace] " << tag << " req_id=" << req_id
                      << " slot=" << slot
                      << " bf16=" << engine.rs_cache->recurrent_state_bf16()
                      << " N=" << N << " rs_is_fold=" << rs_is_fold
                      << " rs_is_write=" << rs_is_write << std::hex
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
            staged_launch.get(),
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
                .forward_R = forward_R,
                .forward_N = forward_N,
                .num_sampling = num_sampling,
                .is_pure_decode = is_pure_decode,
                .have_custom_mask = have_custom_mask,
                .structured_window_left = structured_window_left,
                .has_write_desc = has_write_desc,
                .use_slots = use_slots,
                .h_qo_forward = h_qo_forward,
                .h_kvpi_forward = h_kvpi_forward,
                .h_kvpp_forward = h_kvpp_forward,
                .h_kvlpl_forward = h_kvlpl_forward,
                .slot_ids_h_data = slot_ids_h.data(),
                .is_fresh_h_data = is_fresh_h.data(),
                .rs_buffer_slot_ids_h =
                    (rs_is_write || rs_is_fold)
                        ? pi.rs_buffer_slot_ids_host.data()
                        : nullptr,
                .rs_buffer_slot_indptr_h =
                    (rs_is_write || rs_is_fold)
                        ? pi.rs_buffer_slot_indptr_host.data()
                        : nullptr,
                .rs_fold_lens_h = !rs_fold_len_view.empty()
                    ? pi.rs_fold_lens_host.data()
                    : nullptr,
                .rs_fold_lens_d = !rs_fold_len_view.empty()
                    ? reinterpret_cast<const std::int32_t*>(
                          pi.rs_fold_lens.data())
                    : nullptr,
                .rs_buffer_write = rs_is_write,
                .rs_buffer_fold = rs_is_fold,
                .image_pixels_h = img_pixels_h,
                .image_pixel_byte_indptr_h = img_pix_byte_indptr.data(),
                .image_patch_positions_h = img_patch_pos.data(),
                .image_anchor_rows_h = img_anchor.data(),
                .num_images = img_num_images,
                .image_grids_h = img_grids.data(),
                .mrope_positions_h = mrope_positions_storage.empty()
                    ? nullptr : mrope_positions_storage.data(),
                .num_mrope_positions = static_cast<int>(
                    mrope_positions_storage.size() / 3),
                .audio_features_h = aud_features_h,
                .audio_feature_byte_indptr_h = aud_feat_byte_indptr.data(),
                .audio_anchor_rows_h = aud_anchor.data(),
                .num_clips = aud_num_clips,
                .precomputed_embeddings = precomputed_embeddings,
                .stage_hooks =
                    has_attention_stages ? &stage_hooks : nullptr,
            });
        dump_rs("POST");
        if (rs_is_fold) {
            verify_timer.finish(cublas.stream());
            if (dbg_fire) {
                dbg_ft.forward_enqueue_end = clock::now();
            }
            engine.dispatch->finish(
                *staged_launch, dispatch_view, nullptr, 0,
                engine.cublas.stream(), &runtime, completion);
            if (dbg_fire) {
                dbg_ft.settlement_enqueue_end = clock::now();
            }
            return;
        }
        if (ir_trace) {
            std::cerr << "[ir-trace] forward-returned req_id=" << req_id << "\n";
            std::cerr.flush();
        }
        verify_timer.finish(cublas.stream());
        const auto t_kernel_launch_end = dbg_fire
            ? clock::now()
            : clock::time_point{};
        dbg_ft.forward_enqueue_end = t_kernel_launch_end;

        if (view.ptir_program_hashes.empty()) {
            throw std::runtime_error(
                "legacy sampler launches are removed; direct PTIR launch required");
        }
        const std::uint32_t vocab = static_cast<std::uint32_t>(
            engine.loaded_model.hf_config().vocab_size);
        enqueue_mtp_draft_logits(engine, mtp_draft_plan);
        // The epilogue phase slices each program's logits base from
        // `sampling_indptr[p]` — hand it the PER-PROGRAM gathered-row offsets
        // (`n_prog + 1` entries), not the per-request sampling CSR (the two
        // coincided only while every batched program was exactly one wire
        // request).
        const pipeline::DispatchStats stats_before = dbg_fire
            ? engine.dispatch->stats()
            : pipeline::DispatchStats{};
        engine.dispatch->finish(
            *staged_launch, dispatch_view, nullptr, vocab,
            engine.cublas.stream(),
            &runtime, completion,
            static_cast<const std::uint16_t*>(engine.ws.logits.data()),
            reinterpret_cast<const std::uint32_t*>(sample_rows.data()),
            mtp_draft_plan.draft_starts,
            mtp_draft_counts,
            static_cast<std::uint32_t>(tensor_rows(engine.ws.logits)),
            pi.row_valid.data(),
            program_token_starts,
            dbg_fire ? &dbg_ft.finish_breakdown : nullptr);
        if (dbg_fire) {
            dbg_ft.settlement_enqueue_end = clock::now();
            const pipeline::DispatchStats stats_after =
                engine.dispatch->stats();
            dbg_ft.finish_groups = static_cast<std::int64_t>(
                stats_after.generated_fused_groups -
                stats_before.generated_fused_groups);
            dbg_ft.finish_grouped_lanes = static_cast<std::int64_t>(
                stats_after.grouped_lanes - stats_before.grouped_lanes);
            dbg_ft.finish_body_launches = static_cast<std::int64_t>(
                stats_after.grouped_body_op_launches -
                stats_before.grouped_body_op_launches);
            dbg_ft.finish_shared_exclusions = static_cast<std::int64_t>(
                stats_after.shared_slot_exclusions -
                stats_before.shared_slot_exclusions);
        }
        return;

    } catch (const std::exception& e) {
        if (staged_launch != nullptr) {
            engine.dispatch->abort(
                *staged_launch, engine.cublas.stream());
        }
        std::cerr << "[pie-driver-cuda] fire_batch failed for req_id="
                  << req_id << ": " << e.what() << "\n";
        throw;
    }
}

}  // namespace pie_cuda_driver
