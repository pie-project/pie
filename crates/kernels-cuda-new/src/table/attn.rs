//! Attention: the paged dispatches, the KV writes, MLA, DSA and the sinks.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::{Cap, KernelSig, Prepare, Source, operands};

#[rustfmt::skip]

/// The head count, which nobody carries: a packed row over the head dim.
///
/// Named because two rows share it and a subtree written twice is a
/// subtree that can drift.
const HEAD_DIM: Source = Source::KvLayerField("head_dim");
const PACKED_W: Source = Source::Width(&Source::In(0));
/// The two KV banks the packed row carries beside q.
const KV_BANKS: Source = Source::Mul(
    &Source::Lit(kernels::Lit::I32(2)),
    &Source::Mul(&Source::KvLayerField("num_kv_heads"), &HEAD_DIM),
);

const PACKED_HEADS_IN: Source =
    Source::Div(&Source::Width(&Source::In(0)), &Source::CtxNonZero("head_dim"));
const PACKED_HEADS_OUT: Source =
    Source::Div(&Source::Width(&Source::Out(0)), &Source::CtxNonZero("head_dim"));

pub static KERNELS: &[KernelSig] = &[
    kernel!(flashinfer_decode "attn::dispatch_attention_flashinfer_decode",
        needs = Prepare::DecodePlan, sink = Some("kv.pages"),
        depth_prefix_plan = true,
        // THREE ARITIES, ONE ROW. `[q]` leaves the output to the guard,
        // `[q, o]` states it, and `[q, o, lse]` states the log-sum-exp
        // too because the fire CONSUMES it downstream (gpt-oss's sink
        // rescale reads it). `Or` is how one row serves all three: a slot
        // that is there wins, and a slot that is not falls through to the
        // context's. Nothing here knows arities exist.
        operands = operands![
            cache: DecodePlanCache <- Source::AttnPlan("decode"),
            q: Buf <- Source::In(0),
            kv_layer: KvCacheLayerView <- Source::KvLayerView,
            o: BufMut <- Source::Or(&Source::Out(0), &Source::Attn("o_out")),
            kv_page_indices_d: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr_d: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens_d: U32s <- Source::Attn("kv_last_page_lens_d"),
            workspace: AttentionWorkspaceView <- Source::Attn("workspace"),
            stream: Stream <- Source::Ctx("stream"),
            window_left: I32 <- Source::AttnWindow,
            logits_soft_cap: F32 <- Source::Attn("logits_soft_cap"),
            sm_scale: F32 <- Source::Attn("sm_scale"),
            lse_out: F32sMut <- Source::Or(&Source::Out(1), &Source::Attn("lse_out_d")),
        ]),
    kernel!(flashinfer_decode_capture "attn::dispatch_attention_flashinfer_decode_capture",
        needs = Prepare::DecodePlan, sink = Some("kv.pages"),
        // The decode row with two more operands. `AttnNonZero` on both
        // score buffers is the hand arm's "the fire published no score
        // buffers" refusal, one layer earlier: a fire that wants no
        // scores leaves them null, and a branch that would launch into
        // them declines instead.
        operands = operands![
            cache: DecodePlanCache <- Source::AttnPlan("decode"),
            q: Buf <- Source::In(0),
            kv_layer: KvCacheLayerView <- Source::KvLayerView,
            o: BufMut <- Source::Or(&Source::Out(0), &Source::Attn("o_out")),
            kv_page_indices_d: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr_d: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens_d: U32s <- Source::Attn("kv_last_page_lens_d"),
            workspace: AttentionWorkspaceView <- Source::Attn("workspace"),
            stream: Stream <- Source::Ctx("stream"),
            score_out: F32sMut <- Source::AttnNonZero("score_out"),
            score_indptr_d: I32s <- Source::AttnNonZero("score_indptr_d"),
            window_left: I32 <- Source::AttnWindow,
            logits_soft_cap: F32 <- Source::Attn("logits_soft_cap"),
            sm_scale: F32 <- Source::Attn("sm_scale"),
            lse_out: F32sMut <- Source::Or(&Source::Out(1), &Source::Attn("lse_out_d")),
        ]),
    kernel!(flashinfer_prefill "attn::dispatch_attention_flashinfer_prefill_bf16",
        needs = Prepare::PrefillPlan, sink = Some("kv.pages"),
        // The PAGES loose rather than the view whole, which is what this
        // launcher takes. `prefill_workspace` and not `workspace`: a
        // FlashInfer plan writes its schedule into the workspace it was
        // raised against, so a prefill reading the decode plan's is one
        // clobbering the other.
        operands = operands![
            cache: PrefillPlanCache <- Source::AttnPlan("prefill"),
            q: Buf <- Source::In(0),
            k_pages: BufMut <- Source::KvKeys,
            v_pages: BufMut <- Source::KvValues,
            o: BufMut <- Source::Or(&Source::Out(0), &Source::Attn("o_out")),
            qo_indptr_d: U32s <- Source::Attn("qo_indptr_d"),
            kv_page_indices_d: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr_d: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens_d: U32s <- Source::Attn("kv_last_page_lens_d"),
            workspace: AttentionWorkspaceView <- Source::Attn("prefill_workspace"),
            stream: Stream <- Source::Ctx("stream"),
            logits_soft_cap: F32 <- Source::Attn("logits_soft_cap"),
            sm_scale: F32 <- Source::Attn("sm_scale"),
            lse_out: F32sMut <- Source::Attn("lse_out_d"),
        ]),
    // The plan-free prefill wrapper: it builds an R-shaped plan on the
    // way in, so it owes its caller nothing and cannot be handed a row
    // window — `whole`, and `FireWide` for the same reason XQA is.
    kernel!(flashinfer_prefill_planless "attn::attention_flashinfer_prefill",
        whole = true, needs = Prepare::FireWide, sink = Some("kv.pages"),
        operands = operands![
            q: Buf <- Source::In(0),
            kv_layer: KvCacheLayerView <- Source::KvLayerView,
            o: BufMut <- Source::Out(0),
            qo_indptr_d: U32s <- Source::Attn("qo_indptr_d"),
            kv_page_indices_d: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr_d: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens_d: U32s <- Source::Attn("kv_last_page_lens_d"),
            qo_indptr_h: U32s <- Source::Attn("qo_indptr_h"),
            kv_page_indptr_h: U32s <- Source::Attn("kv_page_indptr_h"),
            total_tokens: I32 <- Source::Rows,
            num_requests: I32 <- Source::Attn("num_requests"),
            // The head COUNT, which nobody carries: the query's width
            // over the cache's head dim.
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::In(0)),
                &Source::KvLayerField("head_dim"),
            ),
            workspace: AttentionWorkspaceView <- Source::Attn("workspace"),
            stream: Stream <- Source::Ctx("stream"),
            window_left: I32 <- Source::AttnWindow,
            logits_soft_cap: F32 <- Source::Attn("logits_soft_cap"),
            sm_scale: F32 <- Source::Attn("sm_scale"),
            lse_out: F32sMut <- Source::Attn("lse_out_d"),
        ]),
    // Head dims flashinfer's prefill template rejects (gemma-4's 512)
    // take a naive paged kernel instead. No plan at all; fire-shaped.
    kernel!(attention_naive_paged "attn::attention_naive_paged",
        whole = true, sink = Some("kv.pages"),
        operands = operands![
            q: Buf <- Source::In(0),
            kv_layer: KvCacheLayerView <- Source::KvLayerView,
            o: BufMut <- Source::Out(0),
            qo_indptr_d: U32s <- Source::Attn("qo_indptr_d"),
            kv_page_indices_d: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr_d: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens_d: U32s <- Source::Attn("kv_last_page_lens_d"),
            total_tokens: I32 <- Source::Rows,
            num_requests: I32 <- Source::Attn("num_requests"),
            num_pages_in_batch: I32 <- Source::Attn("num_pages_in_batch"),
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::In(0)),
                &Source::KvLayerField("head_dim"),
            ),
            stream: Stream <- Source::Ctx("stream"),
            window_left: I32 <- Source::AttnWindow,
            sm_scale: F32 <- Source::Attn("sm_scale"),
        ]),
    kernel!(flashinfer_prefill_capture "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
        needs = Prepare::PrefillPlan, sink = Some("kv.pages"),
        operands = operands![
            cache: PrefillPlanCache <- Source::AttnPlan("prefill"),
            q: Buf <- Source::In(0),
            k_pages: BufMut <- Source::KvKeys,
            v_pages: BufMut <- Source::KvValues,
            o: BufMut <- Source::Or(&Source::Out(0), &Source::Attn("o_out")),
            qo_indptr_d: U32s <- Source::Attn("qo_indptr_d"),
            kv_page_indices_d: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr_d: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens_d: U32s <- Source::Attn("kv_last_page_lens_d"),
            workspace: AttentionWorkspaceView <- Source::Attn("prefill_workspace"),
            stream: Stream <- Source::Ctx("stream"),
            score_out: F32sMut <- Source::AttnNonZero("score_out"),
            folded_out: F32sMut <- Source::Attn("folded_out"),
            score_indptr_d: I32s <- Source::AttnNonZero("score_indptr_d"),
            // The OBSERVATION window, not the attention one --
            // deliberately NOT `AttnWindow`. The launcher refuses `<= 0`
            // and `window_left` is -1 on a family that attends the whole
            // context, so the same number reads as "no window" to one
            // layer and "invalid" to the other.
            window: I32 <- Source::Attn("score_window"),
            logits_soft_cap: F32 <- Source::Attn("logits_soft_cap"),
            sm_scale: F32 <- Source::Attn("sm_scale"),
            lse_out: F32sMut <- Source::Attn("lse_out_d"),
        ]),
    kernel!(flashinfer_custom "attn::dispatch_attention_flashinfer_prefill_custom",
        needs = Prepare::CustomPlan, sink = Some("kv.pages"),
        // The mask rides the CONTEXT, not the statement, for the reason
        // the score sink does: the predicate is folded, so one exec serves
        // the fire that stages a mask and the fire that does not, and the
        // address recorded now must still be right when it goes true.
        operands = operands![
            cache: PrefillPlanCache <- Source::AttnPlan("prefill"),
            q: Buf <- Source::In(0),
            kv_layer: KvCacheLayerView <- Source::KvLayerView,
            o: BufMut <- Source::Or(&Source::Out(0), &Source::Attn("o_out")),
            qo_indptr_d: U32s <- Source::Attn("qo_indptr_d"),
            kv_page_indices_d: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr_d: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens_d: U32s <- Source::Attn("kv_last_page_lens_d"),
            mask_d: U8s <- Source::AttnNonZero("mask_d"),
            mask_indptr_d: I32s <- Source::AttnNonZero("mask_indptr_d"),
            workspace: AttentionWorkspaceView <- Source::Attn("prefill_workspace"),
            stream: Stream <- Source::Ctx("stream"),
            logits_soft_cap: F32 <- Source::Attn("logits_soft_cap"),
            sm_scale: F32 <- Source::Attn("sm_scale"),
            lse_out: F32sMut <- Source::Attn("lse_out_d"),
        ]),
    // XQA: its prepare is fire-wide (R-shaped), so the kernel cannot be
    // given a row window — `whole`. And no capture variant of it
    // exists, so it cannot publish scores — `lacks Scores`. Both are
    // hand-written rules today: the first is the model body's
    // `window_one && c.xqa_decode` test, the second a C++ throw.
    kernel!(xqa_decode "attn::attention_xqa_decode_bf16_prepared",
        whole = true, needs = Prepare::FireWide, lacks = &[Cap::Scores],
        operands = operands![
            q: Buf, k_pages: BufMut, v_pages: BufMut, o: BufMut, num_requests: I32,
            num_q_heads: I32, num_kv_heads: I32, head_dim: I32, page_size: I32,
            max_pages_per_seq: I32, workspace: AttentionWorkspaceView,
            stream: Stream, sm_scale: F32,
        ]),
    kernel!(qkv_decode_fused "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Attn("q_out"),
            k_pages: BufMut <- Source::KvLayerField("k_pages"),
            v_pages: BufMut <- Source::KvLayerField("v_pages"),
            q_weight: Buf <- Source::Weight(0),
            k_weight: Buf <- Source::Weight(1),
            positions: I32s <- Source::Positions,
            rope_table: F32s <- Source::In(1),
            kv_page_indices: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens: U32s <- Source::Attn("kv_last_page_lens_d"),
            w_page: U32s <- Source::Attn("w_page_d"),
            w_off: U32s <- Source::Attn("w_off_d"),
            row_valid: U8s <- Source::Attn("row_valid_d"),
            num_requests: I32 <- Source::Rows,
            // THE PACKED ROW HOLDS Q, K AND V END TO END, so the q heads
            // are what is left after the two kv banks come off.
            num_q_heads: I32 <- Source::Div(&Source::Sub(&PACKED_W, &KV_BANKS), &HEAD_DIM),
            num_kv_heads: I32 <- Source::KvLayerField("num_kv_heads"),
            head_dim: I32 <- HEAD_DIM,
            page_size: I32 <- Source::KvLayerField("page_size"),
            hnd_layout: Bool <- Source::KvLayerField("hnd_layout"),
            theta: F32 <- Source::CtxByLayer("theta"),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The EXPLICIT append: the fire states each token's destination page
    // and offset instead of deriving them from the CSR. Only a fire that
    // computed those carries them, which is what `AttnNonZero` tests —
    // the hand arm made the same null check and returned `NoAttnCtx`
    // saying "the fire published no write descriptors".
    kernel!(write_kv_explicit "attn::write_kv_explicit_bf16",
        operands = operands![
            layer: KvCacheLayerView <- Source::KvLayerView,
            k_curr: Buf <- Source::In(0),
            v_curr: Buf <- Source::In(1),
            w_page: U32s <- Source::AttnNonZero("w_page_d"),
            w_off: U32s <- Source::AttnNonZero("w_off_d"),
            B: I32 <- Source::Rows,
            stream: Stream <- Source::Ctx("stream"),
            row_valid: U8s <- Source::Attn("row_valid_d"),
        ]),
    // The paged KV append, fired once per layer of every fire — and the
    // first row to bind a `KvCacheLayerView`. `Source::KvKeys` and
    // `KvValues` spell a cache as two device pointers, which is METAL's
    // shape and which this emitter refuses; CUDA's launcher takes the
    // view whole, so `Source::KvLayerView` is the spelling it can answer.
    kernel!(write_kv_to_pages "attn::write_kv_to_pages",
        operands = operands![
            layer: KvCacheLayerView <- Source::KvLayerView,
            k_curr: Buf <- Source::In(0),
            v_curr: Buf <- Source::In(1),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            kv_page_indices: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens: U32s <- Source::Attn("kv_last_page_lens_d"),
            total_tokens: I32 <- Source::Rows,
            num_requests: I32 <- Source::Attn("num_requests"),
            stream: Stream <- Source::Ctx("stream"),
            row_valid: U8s <- Source::Attn("row_valid_d"),
            first_token: I32 <- Source::Attn("first_token"),
        ]),
    kernel!(write_kv_explicit_devwin "attn::write_kv_explicit_bf16_devwin",
        whole = true, sink = Some("kv.pages"),
        operands = operands![
            layer: KvCacheLayerView, k_curr: Buf, v_curr: Buf, w_page: U32s,
            w_off: U32s, win_d: U32s, n_max: I32, stream: Stream, row_valid: U8s,
        ]),
    // The pair is what `head_dim_padded` COSTS; stating it turns
    // `if (c.head_dim_padded)` in the model body into a fact the trace
    // carries. Row-shaped -- each token's heads pad independently.
    // THE PACKED SIDE IS WHICHEVER END IS `head_dim` WIDE, which is the
    // input on the way in and the output on the way out. The head count
    // divides out of the packed side either way, and the padded head dim
    // is the other side over that count.
    kernel!(pad_head_dim "attn::pad_head_dim_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            padded: BufMut <- Source::Out(0),
            num_tokens: I32 <- Source::Rows,
            num_heads: I32 <- PACKED_HEADS_IN,
            head_dim: I32 <- Source::CtxNonZero("head_dim"),
            head_dim_padded: I32 <- Source::Div(
                &Source::Width(&Source::Out(0)),
                &PACKED_HEADS_IN,
            ),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(strip_head_dim "attn::strip_head_dim_bf16",
        operands = operands![
            padded: Buf <- Source::In(0),
            packed: BufMut <- Source::Out(0),
            num_tokens: I32 <- Source::Rows,
            num_heads: I32 <- PACKED_HEADS_OUT,
            head_dim: I32 <- Source::CtxNonZero("head_dim"),
            head_dim_padded: I32 <- Source::Div(
                &Source::Width(&Source::In(0)),
                &PACKED_HEADS_OUT,
            ),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // `attn::merge_attention_states_bf16` WAS HERE — the KV-split's other
    // half. Deleted by `new-horizon.md` §38: its whole consumer set was
    // `dsl::cuda::merge_attention_states`, which nothing called. The
    // launcher (`attn/attention_merge_states.cu:31`) stays — it holds no
    // `<<<>>>` at all, it dispatches into the vendored
    // `flashinfer/attention/cascade.cuh`, and `examples/vendor_probe.rs:199`
    // cites it by file and line as the proof that that text compiles here
    // (96,176 B, 8 of 8 symbols). §31.4's precedent exactly: the probe is
    // what makes re-adding this row cheap, and the row was never how to get
    // there.
    // Rewrites `[R+1]` indptr arrays, so a row window would compact the wrong
    // requests' page lists.
    kernel!(compact_page_csr "attn::compact_page_csr", whole = true,
        operands = operands![
            page_indices_in: U32s, page_indptr_in: U32s, last_page_lens_in: U32s,
            keep: U8s, scratch_counts: U32sMut, keep_stride: U32,
            num_requests: I32, page_indices_out: U32sMut, page_indptr_out: U32sMut,
            last_page_lens_out: U32sMut, stream: Stream,
        ]),
    kernel!(attn_score_fold_heads "attn::attn_score_fold_heads", whole = true,
        operands = operands![
            scores: F32s, score_indptr_d: I32s, kv_page_indptr_d: U32s,
            kv_last_page_lens_d: U32s, page_size: I32, num_requests: I32,
            num_q_heads: I32, folded: F32sMut, stream: Stream,
        ]),
    // MLA's absorb pair -- cuBLAS ops rather than raw launches, which is why
    // a launcher is "anything that issues DEVICE work" and not "anything
    // taking a cudaStream_t". `scripts/kernel-vocabulary-audit.py` learned
    // that the hard way.
    // MLA's two absorptions. Both take the WHOLE `kv_b_proj` bank and
    // slice it themselves, so the bank is a weight and the four widths
    // are the shapes around it: `heads` and `kv_lora_rank` are the
    // result's trailing extents, `qk_nope_dim` is the operand's, and
    // `v_head_dim` rides the param channel because this statement's
    // result does not carry it.
    kernel!(mla_absorb_q_to_latent "gemm::mla_absorb_q_to_latent_bf16",
        operands = operands![
            // NO `handle: CublasHandle` — §45. `execution::RUST_SERVED`
            // names both absorbs, so their bodies are `driver-cuda`'s
            // `bind::service::gemm_mla_absorb_*` and the handle comes off
            // the dispatch context rather than out of the row.
            q_nope: Buf <- Source::In(0),
            kv_b_proj: Buf <- Source::Weight(0),
            q_latent: BufMut <- Source::Out(0),
            tokens: I32 <- Source::Rows,
            heads: I32 <- Source::Param(0),
            qk_nope_dim: I32 <- Source::Param(1),
            v_head_dim: I32 <- Source::Param(2),
            kv_lora_rank: I32 <- Source::Param(3),
        ]),
    // The mirror: the latent goes in, `v_head_dim` comes out, and
    // `qk_nope_dim` is the param this direction lacks a shape for.
    kernel!(mla_absorb_latent_to_v "gemm::mla_absorb_latent_to_v_bf16",
        operands = operands![
            attn_latent: Buf <- Source::In(0),
            kv_b_proj: Buf <- Source::Weight(0),
            attn_v: BufMut <- Source::Out(0),
            tokens: I32 <- Source::Rows,
            heads: I32 <- Source::Param(0),
            qk_nope_dim: I32 <- Source::Param(1),
            v_head_dim: I32 <- Source::Param(2),
            kv_lora_rank: I32 <- Source::Param(3),
        ]),
    // MTP drafts several tokens per step and repairs on rejection, which
    // needs an attention that sees a HISTORY buffer beside the pages (the
    // drafted tokens are not committed -- committing them before acceptance
    // is the thing MTP must not do) and a per-slot pending-hidden shuffle.
    // All three address through `slot_ids` or `qo_indptr`.
    //
    // `attn::attention_mtp_paged_history_bf16` WAS the fourth. Deleted by
    // `new-horizon.md` §38: its whole consumer set was
    // `dsl::cuda::attention_mtp_paged_history`, which nothing called. The
    // launcher stays, and the reason is arithmetic rather than caution --
    // `attention_naive.cu:80`'s three-way host choice is the ONLY caller of
    // `attention_mtp_history_bf16` (`:52`), so deleting it would orphan two
    // launchers and two `<<<>>>`, not one, and move `EXPECTED` off 401.
    // Both are `NoRow` entries in `driver-cuda/tests/launch_abi.rs`.
    // Both walk `src_indptr[R+1]`. The window view is how sliding-window
    // attention is expressed without a second cache -- the window is a VIEW
    // over the same pages.
    kernel!(build_window_page_view "attn::build_window_page_view", whole = true,
        operands = operands![
            src_indices: U32s, src_indptr: U32s, keep_pages: I32,
            dst_indptr: U32sMut, dst_indices: U32sMut, R: I32, stream: Stream,
        ]),
    kernel!(build_full_split_view "attn::build_full_split_view", whole = true,
        operands = operands![
            src_indptr: U32s, src_last_page_len: U32s, splits: I32, page_size: I32,
            dst_indptr: U32sMut, dst_indices: U32sMut, dst_last: U32sMut,
            src_indices: U32s, stream: Stream,
        ]),
    // A SECOND KV cache beside the fine-grained one, holding one entry per
    // `ratio` tokens. Every query attends both and the outputs are merged by
    // their log-sum-exps -- exact, not an approximation: the same algebra
    // flashinfer's own KV-split merge uses.
    kernel!(dsv4_boundary_meta_decode "attn::dsv4_boundary_meta_decode",
        operands = operands![
            positions: I32s, out_pos: I32sMut, out_req: I32sMut, out_rope: I32sMut,
            n: I32, ratio: I32, stream: Stream, row_valid: U8s,
        ]),
    // The prefill form. A SECOND row rather than a wider first one: the decode
    // launcher is what a CUDA-graph-captured decode calls, and giving it two
    // more operands would make every capture carry a `qo_indptr` it does not
    // read. The kernels differ in one line -- the request index -- and the
    // tables say so by naming both.
    kernel!(dsv4_boundary_meta_paged "attn::dsv4_boundary_meta_paged",
        operands = operands![
            positions: I32s, qo_indptr: U32s,
            out_pos: I32sMut, out_req: I32sMut, out_rope: I32sMut,
            n: I32, num_requests: I32, ratio: I32, stream: Stream, row_valid: U8s,
        ]),
    // Both address through `kv_page_indptr` and the boundary arrays.
    kernel!(dsv4_compress_gather_paged "attn::dsv4_compress_gather_paged_bf16", whole = true,
        operands = operands![
            state_kv: Buf, state_score: Buf, ape: F32s, boundary_pos: I32s,
            boundary_req: I32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            out: BufMut, num_entries: I32, head_dim: I32, ratio: I32, coff: I32,
            page_size: I32, stream: Stream,
        ]),
    kernel!(dsv4_store_comp_entries "attn::dsv4_store_comp_entries_bf16", whole = true,
        operands = operands![
            entries: Buf, comp_kv_pages: BufMut, boundary_pos: I32s,
            boundary_req: I32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            num_entries: I32, head_dim: I32, page_size: I32, stream: Stream,
        ]),
    // `qo_indptr` + `kv_page_indptr`, like every other paged attention here.
    // No capture variant, so it cannot publish scores; it does publish an LSE,
    // which is what the combine below consumes.
    kernel!(attention_compressed_paged "attn::attention_compressed_paged_bf16",
        whole = true, lacks = &[Cap::Scores],
        operands = operands![
            q: Buf, comp_kv_pages: Buf, o: BufMut, lse_out: F32sMut,
            positions: I32s, qo_indptr: U32s, kv_page_indices: U32s,
            kv_page_indptr: U32s, req_of_token: I32s, total_tokens: I32,
            num_q_heads: I32, head_dim: I32, ratio: I32, page_size: I32,
            sm_scale: F32, stream: Stream,
        ]),
    // Two attention halves and their LSEs, merged. Four operands, two
    // results, and the head geometry off the first result --
    // `[Tokens, heads, head_dim]`.
    kernel!(combine_attn_outputs "attn::combine_attn_outputs_bf16",
        operands = operands![
            o1: Buf <- Source::In(0),
            lse1: F32s <- Source::In(1),
            o2: Buf <- Source::In(2),
            lse2: F32s <- Source::In(3),
            o_out: BufMut <- Source::Out(0),
            lse_out: F32sMut <- Source::Out(1),
            N: I32 <- Source::Rows,
            num_heads: I32 <- Source::Param(0),
            head_dim: I32 <- Source::Param(1),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // FlashInfer publishes its LSE in log2 and the combine works in ln. A
    // unit conversion, stated so a reader never has to guess which base an
    // LSE is in.
    // The rebase is in place on the value it names: `Out(0)` is the
    // statement's result and `In(0)` is the same buffer, so the element
    // count is the result's own extent.
    kernel!(lse_log2_to_ln "attn::lse_log2_to_ln",
        operands = operands![
            lse: F32sMut <- Source::Out(0),
            n: I32 <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // UNSOURCED, and `B` is the whole reason: how many blocks the prefix
    // holds is the BLOCKS operand's row width over the RESULT's -- an
    // operand-over-operand ratio, where every `*WidthOver` variant
    // divides by a CONTEXT field. A row that guessed a param would launch
    // the right kernel over the wrong rectangle.
    kernel!(attn_res_blend "attn::attn_res_blend_bf16",
        operands = operands![
            prefix: Buf <- Source::In(0),
            blocks: Buf <- Source::In(1),
            norm_weight: Buf <- Source::In(2),
            proj_weight: Buf <- Source::In(3),
            out: BufMut <- Source::Out(0),
            T: I32 <- Source::Rows,
            // AN OPERAND OVER AN OPERAND, not a plan dimension. `B` is
            // how many blocks the packed input holds, which is its width
            // divided by the output's — the two are in the same
            // statement, so the row can say it.
            B: I32 <- Source::Div(&Source::Width(&Source::In(1)), &Source::Width(&Source::Out(0))),
            H: I32 <- Source::OutWidth(0),
            block_rows: I32 <- Source::Rows,
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The unfused counterpart of `mla_prepare`. `tokens` is their only
    // extent, so unlike the fused prepare they are NOT `whole` -- which is
    // the reason a deployment might bind them instead.
    // One latent operand split in two: the normed `kv_c` and the
    // rope-carrying `k_pe`. Both widths are the results' own, and the
    // source stride is the operand's row width -- which is what makes
    // the fused q/kv binding readable without being told.
    kernel!(kimi_split_kv_a_norm "attn::kimi_split_kv_a_norm_bf16",
        operands = operands![
            kv_a: Buf <- Source::In(0),
            norm_weight: Buf <- Source::Weight(0),
            kv_c: BufMut <- Source::Out(0),
            k_pe: BufMut <- Source::Out(1),
            tokens: I32 <- Source::Rows,
            kv_lora_rank: I32 <- Source::OutWidth(0),
            qk_rope_dim: I32 <- Source::OutWidth(1),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
            src_row_stride: I32 <- Source::InWidth(0),
        ]),
    // The query's half of the same split. `[Tokens, heads, dim]` on both
    // results, so every extent is a result's own.
    kernel!(kimi_split_q_b "attn::kimi_split_q_b_bf16",
        operands = operands![
            q_b: Buf <- Source::In(0),
            q_nope: BufMut <- Source::Out(0),
            q_pe: BufMut <- Source::Out(1),
            tokens: I32 <- Source::Rows,
            heads: I32 <- Source::Param(0),
            qk_nope_dim: I32 <- Source::Param(1),
            qk_rope_dim: I32 <- Source::Param(2),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // glm5 attends SPARSELY: a small side network scores every (query, key)
    // pair and only the top-k keys per query are attended.
    kernel!(dsa_index_q_rope "attn::dsa_index_q_rope_bf16",
        operands = operands![
            idx_q: BufMut, positions: I32s, tokens: I32, n_heads: I32,
            head_dim: I32, rope_dim: I32, theta: F32, stream: Stream,
        ]),
    kernel!(dsa_index_knorm_rope "attn::dsa_index_knorm_rope_bf16",
        operands = operands![
            idx_k: BufMut, k_norm_weight: Buf, k_norm_bias: Buf, positions: I32s,
            tokens: I32, head_dim: I32, rope_dim: I32, theta: F32, eps: F32,
            stream: Stream,
        ]),
    // `whole`, and here the reason is the ALGEBRA rather than the addressing:
    // query `i` scores keys `0..=i`, so a row window starting anywhere but
    // zero cannot see the keys it must rank against.
    // The indexer's three operands and its mask. `n_heads` and
    // `head_dim` are `idx_q`'s own trailing extents -- it is
    // `[Tokens, n_heads, head_dim]` -- and the top-k rides the param
    // channel, because it is a load-time number no shape carries.
    kernel!(dsa_index_topk_mask "attn::dsa_index_topk_mask", whole = true,
        operands = operands![
            idx_q: Buf <- Source::In(0),
            idx_k: Buf <- Source::In(1),
            idx_w: Buf <- Source::In(2),
            mask: U8sMut <- Source::Out(0),
            tokens: I32 <- Source::Rows,
            n_heads: I32 <- Source::Param(0),
            head_dim: I32 <- Source::Param(1),
            topk: I32 <- Source::Param(2),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // deepseek_v4, glm5 and kimi_k3 attend through a compressed KV: a
    // `kv_lora_rank`-wide latent row plus a small rope-carrying companion,
    // with the heads reconstructed on the way in. A different attention
    // algebra, not a different head count.
    //
    // The two paged statements are `whole` because they address through
    // `qo_indptr` / `kv_page_indptr` / `kv_last_page_lens`, which are
    // R-shaped: a row window would leave that arithmetic pointing at the
    // wrong request. The dispatch is not -- like the flashinfer dispatches,
    // it reads a plan built over the whole fire and still covers a row range.
    kernel!(mla_prepare "attn::mla_prepare_bf16", whole = true,
        operands = operands![
            layer: MlaCacheLayerView, kv_a: Buf, kv_a_norm_weight: Buf, q_b: Buf,
            kv_c: BufMut, k_pe: BufMut, q_nope: BufMut, q_pe: BufMut,
            positions: I32s, qo_indptr: U32s, kv_page_indices: U32s,
            kv_page_indptr: U32s, kv_last_page_lens: U32s, total_tokens: I32,
            num_requests: I32, heads: I32, qk_nope_head_dim: I32, eps: F32,
            theta: F32, interleaved: Bool, kv_a_row_stride: I32,
            yarn: YarnOriginalParams, stream: Stream, row_valid: U8s,
        ]),
    kernel!(write_mla_to_pages "attn::write_mla_to_pages", whole = true,
        operands = operands![
            layer: MlaCacheLayerView, ckv_curr: Buf, kpe_curr: Buf,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s, total_tokens: I32, num_requests: I32,
            stream: Stream, row_valid: U8s,
        ]),
    // No capture variant of this dispatch exists, so it cannot publish the
    // score matrix an `attn.out` observer asks for. It does publish an LSE,
    // which is a different thing and not what the capability names.
    kernel!(attention_mla "attn::dispatch_attention_mla_bf16",
        needs = Prepare::MlaPlan, lacks = &[Cap::Scores],
        operands = operands![
            cache: MlaPlanCache, q_nope: Buf, q_pe: Buf, layer: MlaCacheLayerView,
            o: BufMut, kv_page_indices_d: U32s, workspace: AttentionWorkspaceView,
            stream: Stream, lse_out: F32sMut, qo_indptr_d: U32s,
            kv_page_indptr_d: U32s, kv_last_page_lens_d: U32s, index_mask: U8s,
            index_mask_stride: I32,
        ]),
    // Caps the logits WHERE THEY LIE — one buffer, no destination. Which
    // `Buffers::assign` was already relying on ("the logit softcap
    // accumulates into the logits it was handed", where it widens a
    // seam's pin over an alias set) while this row said nothing, so the
    // set had one member and the widening reached nothing. The head
    // wrote the logits into the arena and the cap ran over `ws.logits`,
    // which is where the sampler then read an uncapped previous fire.
    kernel!(logit_softcap "attn::logit_softcap_bf16",
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            cap: F32 <- Source::CtxNonZero("final_logit_softcap"),
            n: Usize <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // Six statements in one launch; the only value that survives is q.
    kernel!(qkv_packed_post "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
        sink = Some("kv.pages"),
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            // The NATIVE pages, not the bf16 mirrors: this one writes the
            // cache in whatever the cache is.
            k_pages: BufMut <- Source::KvLayerField("k_pages"),
            v_pages: BufMut <- Source::KvLayerField("v_pages"),
            q_weight: Buf <- Source::Weight(0),
            k_weight: Buf <- Source::Weight(1),
            positions: I32s <- Source::Positions,
            kv_page_indices: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens: U32s <- Source::Attn("kv_last_page_lens_d"),
            row_valid: U8s <- Source::Attn("row_valid_d"),
            num_rows: I32 <- Source::Rows,
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::Out(0)),
                &Source::KvLayerField("head_dim"),
            ),
            num_kv_heads: I32 <- Source::KvLayerField("num_kv_heads"),
            head_dim: I32 <- Source::KvLayerField("head_dim"),
            page_size: I32 <- Source::KvLayerField("page_size"),
            hnd_layout: Bool <- Source::KvLayerField("hnd_layout"),
            theta: F32 <- Source::CtxByLayer("theta"),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // Rescales the attention output IN PLACE against the per-head sink
    // logit; the LSE is read-only. gpt-oss's sink layers state it right
    // after the dispatch, so `attn.out` observes the RESCALED result.
    // The LSE is the dispatch's second RESULT, which only a sink layer
    // declares — so it is operand 1 here and traced, not a scratch the
    // executor remembers handing the dispatch.
    kernel!(attention_sink_rescale "attn::attention_sink_rescale_bf16",
        in_place = &[(0, 0)],
        operands = operands![
            o: BufMut <- Source::Out(0),
            lse: F32s <- Source::In(1),
            sinks: Buf <- Source::Weight(0),
            N: I32 <- Source::Rows,
            num_q_heads: I32 <- Source::Ctx("num_q_heads"),
            head_dim: I32 <- Source::Ctx("head_dim"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(mtp_shift_hidden "attn::mtp_shift_hidden_bf16", whole = true,
        operands = operands![
            target_hidden: Buf, pending_hidden: Buf, qo_indptr: U32s,
            slot_ids: I32s, out: BufMut, total_tokens: I32, num_requests: I32,
            hidden_size: I32, stream: Stream,
        ]),
    kernel!(mtp_update_pending_hidden "attn::mtp_update_pending_hidden_bf16", whole = true,
        operands = operands![
            target_hidden: Buf, pending_hidden: BufMut, qo_indptr: U32s,
            slot_ids: I32s, num_requests: I32, hidden_size: I32, stream: Stream,
        ]),
    // Reads and writes the cache and states no operand at all: every
    // argument is the layer's view or the fire's page table.
    kernel!(dequant "attn::dequant_kv_cache_layer_to_bf16_active",
        operands = operands![
            layer: KvCacheLayerView <- Source::KvLayerView,
            kv_page_indices: U32s <- Source::Attn("kv_page_indices_d"),
            num_pages_in_batch: I32 <- Source::Attn("num_pages_in_batch"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
];
