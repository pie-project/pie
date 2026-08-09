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

// `PACKED_HEADS_IN` and `PACKED_HEADS_OUT` WERE HERE, and went with the two
// rows that read them: `attn::pad_head_dim_bf16` and
// `attn::strip_head_dim_bf16` crossed into `crate::x::attn`. They were two
// constants and not one expression written twice because the pad and the
// strip read them off OPPOSITE ends, and a copy that drifted would count
// heads on the padded side — where the divisor is `head_dim_padded`, so the
// count comes out short and the launch covers a prefix of the heads. The two
// divisions are now the `PAD_HEAD_DIM` and `STRIP_HEAD_DIM` bind bodies.

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
    // XQA's row IS DELETED and its symbol has crossed to fn-world:
    // `crate::x::xqa::XQA_DECODE_BF16_PREPARED`, declared beside the five
    // `Unit`s that compile `attn/attention_xqa_mha.cuh` and the
    // `KVCacheList<true>` mirror that made their parameter list
    // expressible. What the row said, kept because the contract cannot
    // carry it: *"its prepare is fire-wide (R-shaped), so the kernel cannot
    // be given a row window — `whole`. And no capture variant of it exists,
    // so it cannot publish scores — `lacks Scores`. Both are hand-written
    // rules today: the first is the model body's `window_one &&
    // c.xqa_decode` test, the second a C++ throw."* `whole`, `needs` and
    // `lacks` are `Contract` fields and survive verbatim; the C++ throw does
    // not, because the file that threw it is deleted.
    //
    // THIS DELETION IS WHAT PERMITTED THE SIX `.cu` TO GO. The symbol
    // stated `operands`, was in neither `device::JIT_DISPATCHED` nor
    // `execution::RUST_SERVED`, and therefore kept a `pie_k_xqa_decode`
    // entry out of `abi::emit_c_shim` — of which
    // `src/attn/attention_xqa.cu` was the definition. Deleting the file
    // without deleting this row is a link error; deleting this row without
    // adding the contract is `Route::Unknown` at model load. Three atomic
    // edits, and this is the third.
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
    // `attn::pad_head_dim_bf16` and `attn::strip_head_dim_bf16` CROSSED INTO
    // FN-WORLD — `crate::x::attn`'s `PAD_HEAD_DIM` and `STRIP_HEAD_DIM`.
    // Stating the pair still turns `if (c.head_dim_padded)` in the model body
    // into a fact the trace carries; what the contracts no longer state is
    // the binding instruction, because the `fn` binds its own arguments.
    // `attn::merge_attention_states_bf16` WAS HERE — the KV-split's other
    // half. Deleted by `new-horizon.md` §38: its whole consumer set was
    // `dsl::cuda::merge_attention_states`, which nothing called.
    //
    // THE TABLE ROW STAYS DELETED AND THE DEVICE TEXT CAME BACK. Those are
    // two different things and this block is the record of both, because for
    // one pass the tree behaved as though they were one.
    //
    // `dsl::cuda::merge_attention_states` (`model-compiler/src/dsl.rs:3532`)
    // still exists and is still called by nothing — `tests/consumer.rs:63`
    // and `examples/migration_status.rs:926` both say so, in the same words:
    // zero callers, zero goldens, zero `pie_k_*`, zero `lower.rs` arms, no
    // peel stem, no fact gate. §38's argument about the CONSUMER SET was
    // correct and nothing here revises it. A table row is a thing
    // `model-compiler` can name in a statement; nothing names this one, so
    // there is no row.
    //
    // What §38 could not see is that the FA2 lattice's split path calls this
    // fold from INSIDE upstream's dispatch (`prefill.cuh:4350-4352`,
    // `decode.cuh:822-824`) rather than through the DSL. The C++ that ran was
    // compiled into `driver-cuda/csrc/attn/attention_flashinfer.cu`, and
    // closing the FA2 seams deleted that file — so
    // `fire/flashinfer_fa2.rs` had to set `disable_split_kv: true` and
    // split-KV prefill was off for a pass. That was a real performance
    // regression on short prompts and small batches.
    //
    // IT IS BACK, as `crate::families::cascade` — one unit,
    // `csrc/src/cascade/merge_states.cuh`, ten rows over the VENDORED
    // `cascade.cuh` — and `driver-cuda/src/fire/merge_states.rs`, the Rust
    // host program. `unit.rs`'s `DEMANDS` names it `Headers::LibraryAndVendor`,
    // the second of two entries.
    //
    // NOT the vendored `cascade.cuh`, which this comment used to claim, AND
    // THAT IS STILL A LIVE DISTINCTION. This crate carries
    // `csrc/vendor/flashinfer/attention/cascade.cuh`, but no `-I` anywhere in
    // the repository puts it in front of a C++ compiler:
    // `kernels-cuda/csrc/CMakeLists.txt`'s include list names
    // `${flashinfer_SOURCE_DIR}` — the CPM checkout — and never `csrc/vendor`.
    // The deleted launcher read the fetched copy; the vendored copy is
    // NVRTC's alone, reachable only through `Headers::LibraryAndVendor`. The
    // two copies being byte-for-byte the same upstream text is what has kept
    // the distinction invisible, and it is the distinction that decides
    // whether deleting `kernels-cuda` frees `csrc/vendor` — it does not. The
    // new unit points at the VENDORED copy, which is the whole point of the
    // return trip: no include path, no CPM, `carried.rs` hands NVRTC the
    // bytes.
    //
    // TWO CORRECTIONS TO THE SPECIFICATION THIS BLOCK USED TO BE.
    //
    // FIRST, IT NAMED THE WRONG LAUNCHER. The spec described `MergeStates`
    // (`cascade.cuh:637-668`) and its `num_index_sets >= seq_len` arm. That
    // launcher is real, it is ported (`fire/merge_states.rs::merge_states`),
    // and it is NOT the one the FA2 split path calls. Both batched dispatches
    // call `VariableLengthMergeStates` (`cascade.cuh:686-736`);
    // `MergeStates` is reached only from the SINGLE-request paths
    // (`prefill.cuh:2559`, `decode.cuh:739`), where every row was split into
    // the same number of chunks. The difference is correctness, not speed:
    // `MergeStatesKernel` folds one `num_index_sets` for every row
    // (`:221`), while `PersistentVariableLengthMergeStatesKernel` reads each
    // row's own count as `indptr[pos + 1] - indptr[pos]` (`:401`). A batch of
    // unequal KV lengths folded with a uniform count reads another row's
    // partials. Implementing only what this block specified and flipping
    // `disable_split_kv` would have been silent corruption.
    //
    // SECOND, IT WAS RIGHT ABOUT THE MISSING VOCABULARY AND THE MEASUREMENTS,
    // AND ALL OF THAT SURVIVES:
    //
    //   * Exactly one kernel fires, never both and never in sequence, so
    //     there is no intermediate buffer. The host decides an empty-work
    //     guard and one arm — `num_index_sets >= seq_len` (`:644`) picks the
    //     large-index-set kernel.
    //   * Shared memory is 8,704 B at head dim 64/128/256 and 16,896 B at
    //     512, all under 48 KB, so the `cudaFuncSetAttribute` at `:656` and
    //     `:715` is a no-op nothing has to express.
    //     `families::cascade::smem_bytes` re-derives both figures and a test
    //     pins them.
    //   * MISSING VOCABULARY: none, and that was a retraction of two
    //     entries. The arm had been written down as unstateable because it
    //     compares TWO operands while every `Term` is unary and `Source`'s
    //     combinators stop at `Ne`; the geometry had been written down as
    //     unstateable because both arms take a computed 2-D block
    //     `(HEAD_DIM / vec_size, bdy)`. Neither survived the rule that host
    //     composition is Rust. Both are now written: the comparison is an
    //     `if` in `fire/merge_states.rs` and the block is a `Launch` literal.
    //     A `LaunchRule` is for a table-driven row, not for a Rust walk.
    //   * Nothing crosses by value. `MergeStatesKernel` takes four pointers
    //     and three `uint32_t` (`:213-216`), the large one four and two
    //     (`:275-281`), the persistent one five and two plus a nullable
    //     device `uint32_t*` (`:366-371`) — every one of which `ArgValue`
    //     binds today. That made it, as this block predicted, the cheapest
    //     available proof that the whole shape works, and it needed no
    //     `params_layout.py` probe.
    //   * The header gate was the thing that ordered the work, and it was
    //     already clear. NVRTC sees only the vendored tree; the CPM checkout
    //     is on no NVRTC path. `csrc/vendor/flashinfer/attention/cascade.cuh`
    //     IS vendored — unlike the sm90 prefill and
    //     `comm/custom_all_reduce.cu`, whose headers are CPM-only and for
    //     which `csrc/vendor` has no `attention/hopper/` and no `comm/`
    //     directory at all.
    //   * `examples/vendor_probe.rs`' `MERGE` candidate compiled this header
    //     to 96,176 B with 8 of 8 symbols resolving, and that measurement is
    //     what made the return trip cheap. §31.4's precedent exactly: the
    //     probe is how you get there, and the row was never how.
    //
    // The one claim in the old block that has decayed: it cited
    // `attn/attention_merge_states.cu:31` as a surviving launcher. That file
    // is gone with the rest of `kernels-cuda/csrc/src/attn/`.
    // `examples/vendor_probe.rs:200` cites it too and is stale for the same
    // reason; the probe still runs, because it reads the vendored header and
    // never that file.
    // Rewrites `[R+1]` indptr arrays, so a row window would compact the wrong
    // requests' page lists.
    // ── `attn::split_qkv_bf16_devwin`, MOVED HERE FROM
    // `table::driver_internal` ──────────────────────────────────────────
    //
    // `model-compiler/src/lower.rs:1503` lowers the peel's tail region to
    // this symbol, so a statement names it and `driver_internal`'s stated
    // membership rule had stopped describing it. The move is also what makes
    // `execution::RUST_SERVED` legal for the row, which is what frees
    // `attn/split_packed.cu`. Every `Source` is carried across unchanged.
    // The DEVICE-WINDOW twin. Its own stated symbol, so there is no
    // ambiguity for a binder to resolve — the peel's tail region states
    // this one and the plain body states the other. `CtxNonZero` on the
    // window is the arm's null check: a fire that published no peel
    // window is not one this launcher can run for.
    kernel!(split_qkv_devwin "attn::split_qkv_bf16_devwin",
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            k_out: BufMut <- Source::Out(1),
            v_out: BufMut <- Source::Out(2),
            win_d: U32s <- Source::CtxNonZero("peel_window"),
            n_max: I32 <- Source::Ctx("rows_total"),
            q_dim: I32 <- Source::OutWidth(0),
            kv_dim: I32 <- Source::OutWidth(1),
            stream: Stream <- Source::Ctx("stream"),
        ]),

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
    // ── `attn::build_window_page_view` AND `attn::build_full_split_view` ─────
    //
    // BOTH ROWS ARE DELETED, with their two `dsl::cuda` wrappers and the two
    // launchers in `attn/kv_paged.cu`. The Rust is
    // `driver-cuda/src/fire/kv_paged.rs::build_window_page_view` and
    // `::build_full_split_view`.
    //
    // WHY DELETION AND NOT `RUST_SERVED`. Every operand of both rows was
    // `Source::Unbound` — `src_indptr` is the page table's CSR, `keep_pages`
    // is a model window divided by a page size, `splits` is a driver plan's
    // piece count, and no model text names any of them — so `crate::abi`
    // skipped each row WHOLE and neither ever generated a dispatch. §60.7
    // establishes that `RUST_SERVED` on an unsourced row is legitimate, and
    // it would work here; the reason it is not used is that it needs a
    // classification first (`every_taken_over_row_was_classified_first`) and
    // §58 says a single launch with no choice and no loop should carry none.
    // A row nothing binds, with a wrapper nothing calls, is §54's case: the
    // row and the wrapper go together.
    //
    // THE SWEEP. `crates/model/src`: no hit for either symbol string OR
    // either wrapper name — the two tokens were swept separately, because a
    // sweep for one has reported a live symbol as uncalled before.
    // `model-compiler/src/dsl.rs`: the two wrappers, deleted in the same
    // change. `lower.rs::semantic()`: no mapping. Hand `ffi::pie_k_*` arms:
    // none. C++: the only hit is
    // `kernels-cuda-new/csrc/src/attn/kv_paged.cuh`, which is the device text.
    //
    // The DEVICE rows stay — `LaunchRule::Single` and `SingleWarp`,
    // `families/attn.rs` — because a family row is a claim about what a
    // kernel IS. The warp-width argument for `SingleWarp` is carried into the
    // Rust's doc comment rather than left behind here.

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
    // `attn::dsv4_compress_gather_paged_bf16` AND
    // `attn::dsv4_store_comp_entries_bf16` CROSSED INTO FN-WORLD as
    // `crate::x::attn`'s `DSV4_COMPRESS_GATHER_PAGED` and
    // `DSV4_STORE_COMP_ENTRIES`, both UNBOUND, with the whole of
    // `attn/dsv4_compress.cuh` as `x::attn::dsv4_compress`'s unit.
    //
    // `whole = true` and no `sink` travelled unchanged. Nine paged rows above
    // state `sink = Some("kv.pages")` and these two never did, because they
    // write the COMPRESSED cache and this vocabulary has one name for the
    // fine-grained one. That absence is a statement and it is kept as one.
    //
    // WHY UNBOUND, and it is not a `Cx` gap. `dsl::cuda`'s two wrappers
    // (`dsl.rs:4684`, `:4702`) record ONE and TWO inputs and no parameters
    // for kernels that read twelve and eight operands, so `state_kv`,
    // `state_score`, `ape`, `boundary_req`, `ratio` and `coff` have no
    // operand to be bound from. Both symbols are on
    // `driver-cuda/tests/executor_bind.rs`' UNARMED list and both were in
    // `device::JIT_DISPATCHED`, so neither had a shim entry, and both rows
    // were unsourced, so `abi` skipped them WHOLE and generated no dispatch.
    // Nothing fired them here and nothing fires them there; what moved is the
    // device text and a written host program waiting on a statement.
    //
    // THREE OTHER ROWS OF THIS ROOT STAY, immediately below and above:
    // `dsv4_boundary_meta_decode`, `dsv4_boundary_meta_paged` and
    // `attention_compressed_paged` are served by
    // `driver-cuda/src/fire/dsv4_compress.rs` through `bind::service`, which
    // is `qkv_decode_fused`'s arrangement exactly. They fire the `_dev`
    // symbols of the unit that just moved, by name, through `hand::fire` —
    // so the §60.6 symbol split is carried VERBATIM into fn-world and
    // renaming any of the three is a panic at the first deepseek_v4 fire.
    //
    // THERE WERE FOUR. `combine_attn_outputs` has since crossed, and the
    // difference between it and these three is worth one sentence because it
    // is the whole of what makes a bind possible: its row sourced every
    // operand from the statement, and each of theirs needs a value the
    // statement does not carry.
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
    // `attn::combine_attn_outputs_bf16` CROSSED INTO FN-WORLD —
    // `crate::x::attn`'s `COMBINE_ATTN_OUTPUTS`, a contract with a bind.
    //
    // The row stated ten sources and every one of them came out of the
    // statement: `In(0..3)`, `Out(0..1)`, `Rows`, `Param(0..1)` and the
    // stream. It was still not a row the executor could fire, because the
    // BLOCK width is `clamp(head_dim, 32, 256)` and no `LaunchRule` says
    // that -- `PerHeadElementwise` clamps into `[32, 128]` and would run the
    // same bytes in half the threads, a slower kernel and never a wrong
    // answer, which is why it could never be caught. So the row lived beside
    // an `execution::WALKED` entry whose whole content was
    // `Control::Supplies { the BLOCK width }`.
    //
    // A `fn` supplies its own geometry, so the classification had nothing
    // left to describe and both artefacts go together. The clamp is now four
    // lines from the launch, in `x::attn::combine_attn`, with the divergence
    // from `SINK_BLOCK_MAX` stated where a reader of the launch will find it.
    // `attn::lse_log2_to_ln` and `attn::attn_res_blend_bf16` CROSSED INTO
    // FN-WORLD — `crate::x::attn`'s `LSE_LOG2_TO_LN` and `ATTN_RES_BLEND`.
    // The rebase is still in place on the value it names, which is what
    // `in_place` says and what the contract keeps. `attn_res_blend`'s `B` is
    // still AN OPERAND OVER AN OPERAND and not a plan dimension — the blocks
    // operand's row width over the result's, two widths of one statement —
    // and the bind reads exactly that, `in_width(1) / out_width(0)`.
    // The unfused counterpart of `mla_prepare`. `tokens` is their only
    // extent, so unlike the fused prepare they are NOT `whole` -- which is
    // the reason a deployment might bind them instead.
    // `attn::kimi_split_kv_a_norm_bf16` and `attn::kimi_split_q_b_bf16`
    // CROSSED INTO FN-WORLD — `crate::x::attn`'s `KIMI_SPLIT_KV_A_NORM` and
    // `KIMI_SPLIT_Q_B`. The first FULL crossing of this family since
    // `softcap`: unit, contracts and binds, so the rows go.
    //
    // What the two paragraphs above said is kept where it is now checked.
    // `kimi_split_kv_a_norm`'s widths are still THE RESULTS' OWN and its
    // source stride still the operand's row width — the bind reads
    // `out_width(0)`, `out_width(1)` and `in_width(0)`, in that order — and
    // it now also REFUSES a stride narrower than the two halves it is being
    // asked to read out of, which three independent `Source`s could not
    // compare. `kimi_split_q_b`'s three trailing extents are still `Param`s
    // and not widths, because `q_b` is one operand of width
    // `heads * (nope + rope)` and no division recovers three numbers.
    //
    // AND THE CROSSING CLOSES A MEASURED WRONG ANSWER. The row above
    // described the LAUNCHER, which formed the device kernel's `total` from
    // `Rows` and the three `Param`s. The JIT has no launcher, so it sized
    // its grid from `LaunchRule::Elementwise` — `rows * out_width(0)` —
    // which for a kernel that splits one wide tensor into two narrow ones is
    // short by `rope / (nope + rope)`. At 6 rows of 8 heads, nope 128, rope
    // 64 it wrote four rows of six. The `fn` forms the product itself, which
    // is what the launcher did, and `device.rs`'s hold on
    // `attn::kimi_split_q_b_bf16` is lifted with it.
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
    // `attn::dispatch_attention_mla_bf16` CROSSED INTO FN-WORLD as
    // `crate::x::attn::ATTENTION_MLA`, carrying `needs = Prepare::MlaPlan` and
    // `lacks = &[Cap::Scores]` verbatim. Its two arms are now BOTH Rust —
    // `x::attn::mla_fa2` is FlashInfer's cooperative kernel with its own unit
    // and a `fire_ex`, and `driver-cuda/src/fire/mla_naive.rs` is the sm_100
    // pair — which is what let the row go, since a row loses its shim entry
    // whole or not at all. `csrc/src/attn/attention_mla.cu` went with it, and
    // with that file the last nvcc-compiled `<<<>>>` in the workspace.
    //
    // It crossed UNBOUND, and the reason is in the contract's `none:` arm:
    // four of its fourteen operands are facts `Cx` does not state — the MLA
    // cache layer, the plan handle, the attention workspace and `sm_scale` —
    // which is the same sentence `executor_bind.rs:1519` was already using to
    // explain why nothing arms it. The row never fired; a row that never fires
    // and holds an nvcc translation unit hostage is worth strictly less than
    // the contract that replaces it.
    // `attn::logit_softcap_bf16` CROSSED INTO FN-WORLD as
    // `crate::x::attn::LOGIT_SOFTCAP`, once `Facts::final_logit_softcap()`
    // landed to source its cap. The row's argument travelled with it: it caps
    // the logits WHERE THEY LIE — one buffer, no destination, which
    // `Buffers::assign` was already relying on ("the logit softcap
    // accumulates into the logits it was handed", where it widens a seam's
    // pin over an alias set) while this row said nothing, so the set had one
    // member and the widening reached nothing. The head wrote the logits into
    // the arena and the cap ran over `ws.logits`, which is where the sampler
    // then read an uncapped previous fire. `in_place` is on the contract.
    // `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` CROSSED INTO
    // FN-WORLD as `crate::x::attn::QKV_PACKED_POST`, with a REAL bind: every
    // one of its twenty-one sources is a `Cx` query that already existed, and
    // ten of them are two — `Cx::kv_layer` carries the five `KvLayerField`
    // spellings plus `hnd`, and `Cx::plan` carries the three CSR arrays with
    // `row_valid`. Six statements in one launch and the only value that
    // survives is q; the rest is the `sink`, which travelled to the contract.
    //
    // `num_q_heads` still comes off the RESULT: `packed` is `[N, q + 2·kv]`
    // and cannot say where the cut falls, `q_out` is `[N, q]` and can.
    //
    // Its DECODE sibling `attn::qkv_decode_qk_norm_rope_write_kv_bf16` stays
    // right here, and that is the `attn::split_qkv_bf16_devwin` arrangement:
    // its host program is `driver-cuda/src/fire/qkv_fused.rs`, served through
    // `bind::service`, so the row is what a trace names and the crossing was
    // the device text alone. Both kernels' text is now
    // `crate::x::attn::qkv_fused`'s unit, `#rope` and `#norope` suffixes
    // carried verbatim because that file fires by name.
    // `attn::attention_sink_rescale_bf16` CROSSED INTO FN-WORLD —
    // `crate::x::attn`'s `ATTENTION_SINK_RESCALE`. gpt-oss's sink layers
    // still state it right after the dispatch, so `attn.out` observes the
    // RESCALED result, and the LSE is still the dispatch's SECOND result —
    // operand 1, a value only a sink layer declares, and not a scratch the
    // executor remembers handing the dispatch.
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
