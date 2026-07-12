#include "batch/compose.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "batch/brle.hpp"
#include "batch/forward.hpp"
#include "batch/logits.hpp"
#include "batch/tp.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "kernels/pack_dense_mask.hpp"
#include "model/loaded_model.hpp"
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

}  // namespace

void handle_fire_batch(
    std::uint32_t req_id,
    const pie_native::LaunchView& view,
    BatchEngine& engine,
    const PieRuntimeCallbacks& runtime,
    PieCompletion completion)
{
    using clock = std::chrono::steady_clock;
    const auto t_entry = clock::now();

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

    // Env-gated per-fire timing (PIE_FIRE_TIMING=1): logs tokens/requests/images
    // and wall duration of the whole fire. Scope guard fires on every return.
    int dbg_R = 0, dbg_N = 0;
    const bool dbg_fire = std::getenv("PIE_FIRE_TIMING") != nullptr;
    struct FireTimer {
        std::chrono::steady_clock::time_point t0; const int& R; const int& N;
        int nimg; std::uint32_t rid; bool en;
        ~FireTimer() {
            if (!en) return;
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            std::cerr << "[fire] req=" << rid << " R=" << R << " N=" << N
                      << " imgs=" << nimg << " " << ms << "ms\n";
        }
    } dbg_ft{t_entry, dbg_R, dbg_N, img_num_images, req_id, dbg_fire};

    try {
        const auto tok_view_orig   = view.token_ids.as<std::uint32_t>();
        const auto pos_view_orig   = view.position_ids.as<std::uint32_t>();
        const auto qo_view_orig    = view.qo_indptr.as<std::uint32_t>();
        const auto kvpi_view_wire = view.kv_page_indices.as<std::uint32_t>();
        const auto kvpp_view_wire = view.kv_page_indptr.as<std::uint32_t>();
        const auto kvlpl_view_orig = view.kv_last_page_lens.as<std::uint32_t>();

        const auto sidx_view_orig  = view.sampling_indices.as<std::uint32_t>();
        const auto sptr_view_orig  = view.sampling_indptr.as<std::uint32_t>();

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
        // `Dispatch::run` slices each program's logits base from.
        std::vector<std::uint32_t> prog_sample_offsets;
        bool dg_resolved = false;
        bool composed_ready = false;
        if (!view.ptir_program_hashes.empty()) {
            std::string dg_err;
            dg_resolved = engine.dispatch->resolve_descriptors(
                view,
                static_cast<std::uint32_t>(kv_cache.page_size()),
                static_cast<std::uint32_t>(kv_cache.num_pages()),
                rpg,
                &dg_err);
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
                            throw std::runtime_error(
                                "ptir: dense device mask in a multi-program "
                                "batch (scheduler contract violated)");
                        }
                    }
                }
            }
            std::string compose_err;
            if (!pipeline::compose_forward_batch(
                    view, rpg,
                    static_cast<std::uint32_t>(kv_cache.page_size()),
                    composed, &compose_err)) {
                throw std::runtime_error(compose_err);
            }
            composed_ready = true;
            prog_sample_offsets = composed.prog_sample_offsets;
        }
        // Only the SOLO device-geometry fire may carry a dense device mask;
        // its resolved geometry equals the composed batch.
        const pipeline::FireGeometry* solo_fg =
            (dg_resolved && rpg.per_program.size() == 1 &&
             rpg.is_device_geometry[0])
                ? &rpg.per_program[0]
                : nullptr;

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

        const auto t_wire_parse_end = clock::now();

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

        if (N == 0 || R <= 0) {
            engine.dispatch->run(
                view, nullptr, 0, engine.cublas.stream(), &runtime, completion);
            return;
        }
        if (N > max_workspace_tokens) {
            std::cerr << "[pie-driver-cuda] batch tokens=" << N
                      << " exceeds workspace=" << max_workspace_tokens << "\n";
            throw std::runtime_error("forward batch exceeds workspace capacity");
        }

        // Detect pure decode so the model can choose its decode kernel.
        const std::uint32_t* h_kvpp  = kvpp_view.data();
        const std::uint32_t* h_kvlpl = kvlpl_view.data();
        const std::uint32_t* h_qo    = qo_view.data();
        bool is_pure_decode = (R > 0);
        for (int r = 0; r < R; ++r) {
            if (h_qo[r + 1] - h_qo[r] != 1u) is_pure_decode = false;
        }

        const auto rs_slot_view = view.rs_slot_ids.as<std::uint32_t>();
        const auto rs_flag_view = view.rs_slot_flags.as<std::uint8_t>();
        bool use_slots =
            R > 0 && rs_slot_view.size() == static_cast<std::size_t>(R);
        if (engine.rs_cache != nullptr && R > 0 && !use_slots) {
            throw std::runtime_error(
                "rs_cache forward missing runtime-assigned slot ids");
        }
        // Ph7 RS working-set buffered-activation channel. Single-role per pass
        // (v1): a FOLD pass (FOLD-bit=2 set) gathers+replays from the buffered
        // pool into recurrent_state (separate fold-replay dispatch below); an
        // rs-output write pass (FOLD-bit clear + buffered slabs present)
        // scatters in-proj [mixed_qkv|a|b] to the pool during the main forward.
        const auto rs_buf_id_view = view.rs_buffer_slot_ids.as<std::uint32_t>();
        const auto rs_buf_indptr_view = view.rs_buffer_slot_indptr.as<std::uint32_t>();
        const bool rs_is_fold = use_slots && std::any_of(
            rs_flag_view.begin(), rs_flag_view.end(),
            [](std::uint8_t v) { return (v & 2u) != 0; });
        const bool rs_is_write =
            use_slots && !rs_is_fold && rs_buf_id_view.size() > 0;

        // Direct PTIR launches keep the forward geometry exact. The program runs
        // after the model and therefore cannot use the legacy graph variants that
        // fused sampling into the captured forward.
        ForwardInputViews forward_inputs = make_forward_input_views(
            tok_view, pos_view, qo_view, kvpi_view, kvpp_view, kvlpl_view, R);
        const int forward_N = forward_inputs.total_tokens;
        const int forward_R = forward_inputs.num_requests;
        const std::uint32_t* h_qo_forward = forward_inputs.qo_indptr.data();
        const std::uint32_t* h_kvpi_forward =
            forward_inputs.kv_page_indices.data();
        const std::uint32_t* h_kvpp_forward =
            forward_inputs.kv_page_indptr.data();
        const std::uint32_t* h_kvlpl_forward =
            forward_inputs.kv_last_page_lens.data();

        // Refill persistent device buffers with this fire's wire inputs.
        // Same device addresses every fire — required for graph-replay
        // safety; cheap (single async memcpy each) on its own.
        pi.tokens.copy_from_host(forward_inputs.tokens);
        pi.positions.copy_from_host(forward_inputs.positions);
        pi.qo_indptr.copy_from_host(forward_inputs.qo_indptr);
        pi.kv_page_indices.copy_from_host(forward_inputs.kv_page_indices);
        pi.kv_page_indptr.copy_from_host(forward_inputs.kv_page_indptr);
        pi.kv_last_page_lens.copy_from_host(forward_inputs.kv_last_page_lens);

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
            if (!pie_cuda_driver::brle::is_pure_causal(
                    fmask_view, mskptr_view,
                    qo_span, kvpp_span, kvlpl_span,
                    kv_cache.page_size())) {
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

        // ── W1.3: device-geometry AttnMask → FlashInfer packed custom mask ──
        // A device-geometry fire may carry a DENSE [lanes, stride] per-cell mask
        // on its AttnMask descriptor port (resolved into fg.mask). Pack it to
        // FlashInfer's bit-packed custom mask (launch_pack_dense_mask) INTO
        // pi.custom_mask, so the standard custom-mask forward path consumes it
        // exactly like a BRLE-decoded wire mask. DORMANT unless a device-geometry
        // program binds an AttnMask channel (fg.has_mask); the guest producer is
        // W2.1. Correctness is validated once a real device-geometry fire exists.
        if (solo_fg != nullptr && solo_fg->has_mask && !solo_fg->mask.empty()) {
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
            pi.w_page.copy_from_host(
                std::span<const std::uint32_t>(composed.w_page));
            pi.w_off.copy_from_host(
                std::span<const std::uint32_t>(composed.w_off));
            has_write_desc = true;
        }

        // Linear-attention rs_cache slots. Runtime owns slot assignment;
        // RS-capable models must receive one slot id per request.
        std::vector<std::int32_t> slot_ids_h;
        std::vector<std::uint8_t> is_fresh_h;
        if (use_slots) {
            const int slot_count = R;
            slot_ids_h.resize(slot_count);
            is_fresh_h.resize(slot_count);
            for (int r = 0; r < R; ++r) {
                slot_ids_h[r] = static_cast<std::int32_t>(rs_slot_view[r]);
                is_fresh_h[r] = (r < static_cast<int>(rs_flag_view.size()) &&
                                 (rs_flag_view[r] & 1u))
                                    ? 1u
                                    : 0u;
            }
            for (int r = R; r < slot_count; ++r) {
                slot_ids_h[r] = engine.graph_pad_slot;
                is_fresh_h[r] = 0u;
            }
            pi.slot_ids.copy_from_host(std::span<const std::int32_t>(slot_ids_h));
            pi.is_fresh.copy_from_host(std::span<const std::uint8_t>(is_fresh_h));
        }

        if (engine.rs_cache != nullptr) {
            engine.rs_cache->set_verify_frozen(false);
        }

        if (sptr_view.size() != static_cast<std::size_t>(R + 1) ||
            sptr_view.back() != sidx_view.size()) {
            throw std::runtime_error("sampling CSR does not match launched instances");
        }
        std::vector<std::int32_t> sample_rows;
        sample_rows.reserve(sidx_view.size());
        const std::uint32_t* h_sptr = sptr_view.data();
        const std::uint32_t* h_sidx = sidx_view.data();
        for (int r = 0; r < R; ++r) {
            const std::uint32_t qo_begin = h_qo[r];
            const std::uint32_t qo_len = h_qo[r + 1] - qo_begin;
            for (std::uint32_t k = h_sptr[r]; k < h_sptr[r + 1]; ++k) {
                if (h_sidx[k] >= qo_len) {
                    throw std::runtime_error("sampling row exceeds request query span");
                }
                sample_rows.push_back(
                    static_cast<std::int32_t>(qo_begin + h_sidx[k]));
            }
        }
        if (sample_rows.size() > pi.sample_idx.size()) {
            throw std::runtime_error("sampling rows exceed persistent input capacity");
        }
        if (N > tensor_rows(ws.logits)) {
            throw std::runtime_error("forward batch exceeds logits workspace");
        }
        if (!sample_rows.empty()) {
            pi.sample_idx.copy_from_host(
                std::span<const std::int32_t>(sample_rows));
        }

        const auto t_plan_end = clock::now();

        // TP fan-out. Rank 0 broadcasts the per-fire payload (header +
        // refilled persistent_inputs) to every follower so they can run
        // the same forward kernels against an identical view of inputs.
        // The all-reduces inside `forward_fn.body` then synchronise the
        // ranks layer-by-layer. The header includes the forward variant so
        // CUDA graph capture/replay stays lockstep across ranks.
        if (engine.tp_comm != nullptr) {
            tp_cpu_gate_notify(engine.tp_cpu_gate_key);
            tp_broadcast_inputs(*engine.tp_comm, pi,
                                forward_N, forward_R, is_pure_decode,
                                static_cast<int>(
                                    forward_inputs.kv_page_indices.size()),
                                mask_bytes, mask_indptr_count,
                                /*has_slot_ids=*/use_slots,
                                /*tp_greedy_argmax=*/false,
                                /*logit_rows=*/0,
                                /*stream=*/nullptr);
        }

        // ── prepare hook ────────────────────────────────────────
        // Always run the per-arch prepare phase first (when present).
        // For graph-capable archs this updates pinned host / device
        // plan state for the captured body to read. Lives outside any
        // capture region so the host work re-runs every fire.
        forward_fn.invoke_prepare(
            attn_ws,
            ForwardFn::PrepareInputs{
                .qo_indptr_h = h_qo_forward,
                .kv_page_indices_h = h_kvpi_forward,
                .kv_page_indices_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_page_indices.data()),
                .kv_page_indptr_h = h_kvpp_forward,
                .kv_page_indptr_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_page_indptr.data()),
                .kv_last_page_lens_h =
                    h_kvlpl_forward,
                .kv_last_page_lens_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_last_page_lens.data()),
                .total_tokens = forward_N,
                .num_requests = forward_R,
                .is_pure_decode = is_pure_decode,
            });

        forward_fn.invoke_set_logits_argmax_only(false);
        forward_fn.invoke_set_fused_argmax_output(nullptr);
        const auto t_h2d_end = clock::now();

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

        run_forward_dispatch(
            engine, ForwardDispatchInputs{
                .forward_R = forward_R,
                .forward_N = forward_N,
                .num_sampling = num_sampling,
                .is_pure_decode = is_pure_decode,
                .have_custom_mask = have_custom_mask,
                .has_write_desc = has_write_desc,
                .use_slots = use_slots,
                .h_qo_forward = h_qo_forward,
                .h_kvpi_forward = h_kvpi_forward,
                .h_kvpp_forward = h_kvpp_forward,
                .h_kvlpl_forward = h_kvlpl_forward,
                .slot_ids_h_data = slot_ids_h.data(),
                .is_fresh_h_data = is_fresh_h.data(),
                .rs_buffer_slot_ids_h = rs_is_write ? rs_buf_id_view.data() : nullptr,
                .rs_buffer_slot_indptr_h = rs_is_write ? rs_buf_indptr_view.data() : nullptr,
                .rs_buffer_write = rs_is_write,
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
            });
        dump_rs("POST");
        if (ir_trace) {
            std::cerr << "[ir-trace] forward-returned req_id=" << req_id << "\n";
            std::cerr.flush();
        }
        verify_timer.finish(cublas.stream());
        const auto t_kernel_launch_end = clock::now();
        static_cast<void>(t_entry);
        static_cast<void>(t_wire_parse_end);
        static_cast<void>(t_plan_end);
        static_cast<void>(t_h2d_end);
        static_cast<void>(t_kernel_launch_end);

        if (view.ptir_program_hashes.empty()) {
            throw std::runtime_error(
                "legacy sampler launches are removed; direct PTIR launch required");
        }
        const std::uint32_t vocab = static_cast<std::uint32_t>(
            engine.loaded_model.hf_config().vocab_size);
        const void* ptir_logits =
            gather_selected_logits_f32(engine, num_sampling, vocab);
        // The post-forward dispatch slices each program's logits base from
        // `sampling_indptr[p]` — hand it the PER-PROGRAM gathered-row offsets
        // (`n_prog + 1` entries), not the per-request sampling CSR (the two
        // coincided only while every batched program was exactly one wire
        // request).
        pie_native::LaunchView dispatch_view = view;
        dispatch_view.sampling_indices = pie_native::slice_from_u32(
            sidx_view.data(), sidx_view.size());
        dispatch_view.sampling_indptr = pie_native::slice_from_u32(
            prog_sample_offsets.data(), prog_sample_offsets.size());
        engine.dispatch->run(
            dispatch_view, ptir_logits, vocab, engine.cublas.stream(),
            &runtime, completion);
        return;

    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] fire_batch failed for req_id="
                  << req_id << ": " << e.what() << "\n";
        throw;
    }
}

}  // namespace pie_cuda_driver
