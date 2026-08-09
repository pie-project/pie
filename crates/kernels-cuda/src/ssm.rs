//! Linear attention and state-space recurrences: GDN, KDA, mamba, and the
//! causal conv that feeds them.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::operands;
use kernels::Lit;
use kernels::Source;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // The other mamba scan: nemotron_h takes FlashInfer's SSU on sm90+ and
    // its own batched kernel elsewhere.
    kernel!(flashinfer_mamba_ssu "ssm::flashinfer_mamba_ssu_bf16", whole = true,
        returns = "bool",
        operands = operands![
            conv_out: U16s,
            dt: U16s,
            a: F32s,
            d: U16s,
            dt_bias: U16s,
            state_base: U16sMut,
            slot_ids: I32s,
            y: U16sMut,
            batch: I32,
            num_heads: I32,
            head_dim: I32,
            state_size: I32,
            num_groups: I32,
            conv_dim: I32,
            intermediate: I32,
            state_cache_size: I32,
            stream: Stream,
        ]),
    // Unbatched twins of the `_batched` forms below -- a legacy parity
    // entrypoint and a single-request fast path. Not `whole`, for the reason
    // the batched ones are not: their `B` is the batch, not a window into it.
    // The `_state_bf16` pairing is a precision BINDING a deployment states,
    // the same way the batched rows spell it.
    kernel!(gdn_step_single "ssm::recurrent_gated_delta_step",
        operands = operands![
            q_norm: F32s,
            k_norm: F32s,
            v: F32s,
            g_log: F32s,
            beta: F32s,
            state: F32sMut,
            out: F32sMut,
            b: I32,
            v_h: I32,
            k_d: I32,
            v_d: I32,
            stream: Stream,
        ]),
    kernel!(gdn_step_single_state_bf16 "ssm::recurrent_gated_delta_step_state_bf16",
        operands = operands![
            q_norm: F32s,
            k_norm: F32s,
            v: F32s,
            g_log: F32s,
            beta: F32s,
            state: BufMut,
            out: F32sMut,
            b: I32,
            v_h: I32,
            k_d: I32,
            v_d: I32,
            stream: Stream,
        ]),
    kernel!(gdn_prefill_single "ssm::chunk_gated_delta_prefill",
        operands = operands![
            q_norm: F32s,
            k_norm: F32s,
            v: F32s,
            g_log: F32s,
            beta: F32s,
            state: F32sMut,
            out: F32sMut,
            t: I32,
            v_h: I32,
            k_d: I32,
            v_d: I32,
            chunk_size: I32,
            stream: Stream,
        ]),
    kernel!(gdn_prefill_single_state_bf16 "ssm::chunk_gated_delta_prefill_state_bf16",
        operands = operands![
            q_norm: F32s,
            k_norm: F32s,
            v: F32s,
            g_log: F32s,
            beta: F32s,
            state: BufMut,
            out: F32sMut,
            t: I32,
            v_h: I32,
            k_d: I32,
            v_d: I32,
            chunk_size: I32,
            stream: Stream,
        ]),
    kernel!(causal_conv1d_prefill_single "ssm::causal_conv1d_prefill_bf16",
        operands = operands![
            x: Buf,
            weight: Buf,
            bias: Buf,
            y: BufMut,
            state_out: BufMut,
            n: I32,
            c: I32,
            k: I32,
            stream: Stream,
        ]),
    // The third linear-attention shape here, and not a variant of the other
    // two: mamba carries a `[head_dim, state_size]` slab per head and
    // advances it with a scalar `dA` from a per-token `dt` -- a selective
    // scan, not a delta rule. A different state SHAPE, which is why none of
    // the GDN or KDA rows stand in for it.
    // Every extent comes off an operand, which is unusual for this
    // module: a three-way cut states all three destinations, so the
    // widths that say where the cuts fall are the results' own. Nothing
    // here needs the GDN context that blocks the rest of `ssm`.
    kernel!(nemotron_mamba_split "ssm::nemotron_mamba_split_bf16",
        operands = operands![
            projected: Buf <- Source::In(0),
            gate: BufMut <- Source::Out(0),
            conv_in: BufMut <- Source::Out(1),
            dt: BufMut <- Source::Out(2),
            n: I32 <- Source::Rows,
            projection_dim: I32 <- Source::InWidth(0),
            intermediate: I32 <- Source::OutWidth(0),
            conv_dim: I32 <- Source::OutWidth(1),
            num_heads: I32 <- Source::OutWidth(2),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The only row in this module that needs nothing from the GDN
    // context BUT a scalar — no slab, no aux operand, no attention
    // context. Which is why it is the one `Source::Gdn` gets on its own:
    // the rest of `ssm` wants per-layer state slabs and operands the
    // trace does not state, and naming a field does not reach those.
    kernel!(nemotron_prepare_mamba_params "ssm::nemotron_prepare_mamba_params",
        operands = operands![
            a_log: Buf <- Source::Weight(0),
            d: Buf <- Source::Weight(1),
            dt_bias: Buf <- Source::Weight(2),
            a: F32sMut <- Source::Out(0),
            d_f32: F32sMut <- Source::Out(1),
            dt_bias_f32: F32sMut <- Source::Out(2),
            num_heads: I32 <- Source::Gdn("v_h"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // SOURCED, and the reason it could not be before is worth keeping:
    // `dt_bias` is a FOREIGN value — the split's raw table, which this
    // statement does not carry as an arg — so it needed `Source::Aux`,
    // and `Aux` was in the vocabulary and in the emitter while the
    // runtime helper both of them call, `join_aux`, had never been
    // written. Nothing noticed, because an emitter arm no row reaches
    // emits no code and a missing symbol that is never emitted never
    // reaches a compiler.
    //
    // The symptom was a branch that read correctly "and never fired".
    // It could not have fired: the crate did not build, and the binary
    // under test was the previous one. Recorded because it is the
    // SECOND stale-build mistaken for a live mystery in this table.
    //
    // `Aux(3)` is `dt_bias` in the join's order
    // `[dt_raw, a, d, dt_bias, dt_pre, da_pre]` — the same index the
    // hand arm spelled `aux_slot(3, resolver)`. An aux operand does not
    // raise the arity guard, which is right: it is not one of the
    // statement's args, and a row demanding it as one would decline
    // every site.
kernel!(nemotron_prepare_mamba_dt_da "ssm::nemotron_prepare_mamba_dt_da",
        operands = operands![
            dt: Buf <- Source::In(0),
            a: F32s <- Source::In(1),
            dt_bias: F32s <- Source::Aux(3),
            dt_out: F32sMut <- Source::Out(0),
            da_out: F32sMut <- Source::Out(1),
            n: I32 <- Source::Rows,
            num_heads: I32 <- Source::InWidth(0),
            time_step_min: F32 <- Source::Lit(Lit::F32(0.0)),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // `whole` for both reasons this table collects: it addresses through
    // `slot_ids` and `qo_indptr`, and the scan carries state token to token,
    // so a row window would resume from the wrong slab.
    kernel!(nemotron_mamba_ssm "ssm::nemotron_mamba_ssm_batched_bf16", whole = true,
        operands = operands![
            conv_out: Buf,
            dt: Buf,
            a: F32s,
            d: F32s,
            dt_bias: F32s,
            dt_precomputed: F32s,
            da_precomputed: F32s,
            ssm_state_base: BufMut,
            slot_ids: I32s,
            qo_indptr: U32s,
            y: BufMut,
            r: I32,
            num_heads: I32,
            head_dim: I32,
            state_size: I32,
            n_groups: I32,
            conv_dim: I32,
            intermediate: I32,
            time_step_min: F32,
            sequence_prefill: Bool,
            stream: Stream,
        ]),
    // Advances a slot's conv window in place; a row window advances the
    // wrong slots.
    kernel!(causal_conv1d_update "ssm::causal_conv1d_update_bf16", whole = true,
        operands = operands![
            x: Buf,
            weight: Buf,
            bias: Buf,
            state: BufMut,
            y: BufMut,
            c: I32,
            k: I32,
            stream: Stream,
        ]),
    // kimi_k3's linear-attention half. The gated delta rule qwen3_5 runs,
    // with the decay per KEY CHANNEL rather than per head -- which is why
    // these exist beside the GDN kernels instead of reusing them with a
    // broadcast.
    // Two operands, two weights, two results — and the head geometry is
    // the results' own: `gate_out` is `[Tokens, h * d]` and `beta_out` is
    // `[Tokens, h]`, so `h` is the second result's width and `d` the
    // first's divided by it. The row says both directly off the shapes.
    kernel!(kda_gate_beta "ssm::kda_gate_beta_bf16",
        operands = operands![
            raw_g: Buf <- Source::In(0),
            raw_beta: Buf <- Source::In(1),
            a_log: F32s <- Source::Weight(0),
            dt_bias: F32s <- Source::Weight(1),
            gate_out: F32sMut <- Source::Out(0),
            beta_out: F32sMut <- Source::Out(1),
            t: I32 <- Source::Rows,
            h: I32 <- Source::OutWidth(1),
            d: I32 <- Source::Param(0),
            lower_bound: F32 <- Source::Lit(Lit::F32(0.0)),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // `slot_ids` is indexed `0..R` against the fire's request order, so a row
    // window would advance the wrong slots.
    kernel!(kda_recurrent_step "ssm::kda_recurrent_step_batched", whole = true,
        operands = operands![
            q_norm: F32s,
            k_norm: F32s,
            v: F32s,
            gate: F32s,
            beta: F32s,
            state_base: F32sMut,
            slot_ids: I32s,
            slot_stride_elems: I64,
            out: F32sMut,
            r: I32,
            h: I32,
            d: I32,
            stream: Stream,
        ]),
    // `whole` twice over: it walks windows out of `qo_indptr`, and the
    // recurrence has a strict per-token state dependency -- a row window
    // would start the scan from the wrong state, which is a different answer
    // rather than a misaddressed one.
    kernel!(kda_prefill "ssm::kda_prefill_batched", whole = true,
        operands = operands![
            q_norm: F32s,
            k_norm: F32s,
            v: F32s,
            gate: F32s,
            beta: F32s,
            state_base: F32sMut,
            slot_ids: I32s,
            qo_indptr: U32s,
            slot_stride_elems: I64,
            out: F32sMut,
            r: I32,
            h: I32,
            d: I32,
            stream: Stream,
        ]),
    // The gated output norm: the recurrence's fp32 output, the gate, one
    // weight, one bf16 result. `h` and `d` ride the param channel — the
    // result is `[Tokens, h * d]` and only their product is a shape.
    kernel!(kda_o_norm_gated "ssm::kda_o_norm_gated_bf16",
        operands = operands![
            o: F32s <- Source::In(0),
            g: Buf <- Source::In(1),
            weight: F32s <- Source::Weight(0),
            out: BufMut <- Source::Out(0),
            t: I32 <- Source::Rows,
            h: I32 <- Source::Param(0),
            d: I32 <- Source::Param(1),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(gdn_conv_update "ssm::causal_conv1d_update_batched_bf16",
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Weight(0),
            // The checkpoint may ship none, which is a fact about the
            // checkpoint rather than drift -- so null, not a refusal.
            bias: Buf <- Source::WeightSuffix("_bias"),
            // The STATEMENT'S layer's conv window. Absent three ways (no
            // GDN context, no layer stated, no slab there) and all three
            // decline the branch.
            state_base: BufMut <- Source::GdnSlab("conv_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            slot_stride_elems: I64 <- Source::Gdn("conv_stride_elems"),
            y: BufMut <- Source::Out(0),
            r: I32 <- Source::Rows,
            c: I32 <- Source::Gdn("conv_dim"),
            k: I32 <- Source::Gdn("conv_k"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(gdn_conv_prefill "ssm::causal_conv1d_prefill_batched_bf16",
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Weight(0),
            bias: Buf <- Source::WeightSuffix("_bias"),
            y: BufMut <- Source::Out(0),
            state_out_base: BufMut <- Source::GdnSlab("conv_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            // The prefill walks the fire's qo CSR, so its request count
            // and its offsets are the ATTENTION context's -- the same
            // pair the attention dispatches take.
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            slot_stride_elems: I64 <- Source::Gdn("conv_stride_elems"),
            r: I32 <- Source::Attn("num_requests"),
            c: I32 <- Source::Gdn("conv_dim"),
            k: I32 <- Source::Gdn("conv_k"),
            stream: Stream <- Source::Ctx("stream"),
            write_state: Bool <- Source::Gdn("write_state"),
            // Spec-decode's per-row commit lengths and its write mask. No
            // fire threads either yet, and the hand arm passed the same
            // two nulls -- stated here so the absence is the ROW's rather
            // than an arm's.
            commit_len: I32s <- Source::Lit(Lit::Null),
            write_state_mask: U8s <- Source::Lit(Lit::Null),
        ]),
    kernel!(gdn_step "ssm::recurrent_gated_delta_step_batched",
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: F32sMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Attn("num_requests"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(gdn_step_gqa "ssm::recurrent_gated_delta_step_batched_gqa",
        operands = operands![
            q_norm_kh: F32s <- Source::In(0),
            k_norm_kh: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: F32sMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Attn("num_requests"),
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(gdn_step_state_bf16 "ssm::recurrent_gated_delta_step_batched_state_bf16",
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            // RESTATED IN CUDA'S VOCABULARY. These four were `Ctx("rs_slab")`,
            // `Ctx("rs_slot_ids")`, `Ctx("rs_slot_stride")` and three
            // `InDim`s -- Metal's spellings, which this generator refuses:
            // a DIM is the plan's and not the binder's, and the recurrent
            // slab is a per-LAYER entry rather than a context field. The
            // GDN context carries every one of them, and `GdnSlab` reaches
            // the layer's.
            state_base: BufMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Rows,
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(gdn_step_gqa_state_bf16 "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16",
        operands = operands![
            q_norm_kh: F32s <- Source::In(0),
            k_norm_kh: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: BufMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Attn("num_requests"),
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(gdn_prefill_fla "ssm::chunk_gated_delta_prefill_batched",
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: F32sMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Attn("num_requests"),
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
            write_state: Bool <- Source::Gdn("write_state"),
            commit_len: I32s <- Source::Lit(Lit::Null),
            write_state_mask: U8s <- Source::Lit(Lit::Null),
        ]),
    kernel!(gdn_prefill_fla_state_bf16 "ssm::chunk_gated_delta_prefill_batched_state_bf16",
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: BufMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Attn("num_requests"),
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
            write_state: Bool <- Source::Gdn("write_state"),
            commit_len: I32s <- Source::Lit(Lit::Null),
            write_state_mask: U8s <- Source::Lit(Lit::Null),
        ]),
    kernel!(gdn_prefill_cached "ssm::chunk_gated_delta_prefill_batched_cached",
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: F32sMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Attn("num_requests"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
            write_state: Bool <- Source::Gdn("write_state"),
            write_state_mask: U8s <- Source::Lit(Lit::Null),
        ]),
    kernel!(gdn_prefill_cached_state_bf16
        "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16",
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: BufMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Attn("num_requests"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
            write_state: Bool <- Source::Gdn("write_state"),
            write_state_mask: U8s <- Source::Lit(Lit::Null),
        ]),
    kernel!(gdn_prefill_warp_tiled_gqa "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: F32sMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Attn("num_requests"),
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
            write_state: Bool <- Source::Gdn("write_state"),
            write_state_mask: U8s <- Source::Lit(Lit::Null),
        ]),
    kernel!(gdn_prefill_warp_tiled_gqa_state_bf16
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
        operands = operands![
            q_norm: F32s <- Source::In(0),
            k_norm: F32s <- Source::In(1),
            v: F32s <- Source::In(2),
            g_log: F32s <- Source::In(3),
            beta: F32s <- Source::In(4),
            state_base: BufMut <- Source::GdnSlab("recurrent_state"),
            slot_ids: I32s <- Source::Gdn("slot_ids_d"),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            slot_stride_elems: I64 <- Source::Gdn("state_stride_elems"),
            out: F32sMut <- Source::ResultOrRegion(0),
            r: I32 <- Source::Attn("num_requests"),
            k_h: I32 <- Source::Gdn("k_h"),
            v_h: I32 <- Source::Gdn("v_h"),
            k_d: I32 <- Source::Gdn("k_d"),
            v_d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
            write_state: Bool <- Source::Gdn("write_state"),
            write_state_mask: U8s <- Source::Lit(Lit::Null),
        ]),
    // The head geometry off the two VALUES: the compact operand is
    // `[Tokens, key_heads, key_dim]` and the repeated result is
    // `[Tokens, value_heads, key_dim]`, so all three counts are dims the
    // statement already carries. It states its result since the repeat
    // stopped being output-less.
    kernel!(repeat_interleave_heads "ssm::repeat_interleave_heads_fp32",
        operands = operands![
            in_: F32s <- Source::In(0),
            out: F32sMut <- Source::ResultOrRegion(0),
            n: I32 <- Source::Rows,
            k_h: I32 <- Source::Gdn("k_h"),
            // The repeated head count and the head width -- `OutDim(0, 1)`
            // and `OutDim(0, 2)` on Metal, where a value's dims are the
            // binder's to read. Here they are the GDN context's, which is
            // the same two numbers from the place that computes them.
            v_h: I32 <- Source::Gdn("v_h"),
            d: I32 <- Source::Gdn("v_d"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // KDA's arithmetic is fp32 throughout, so operands living in bf16 in the
    // workspace cross explicitly. Launches, so the trace records them.
    // Every argument is the statement's: one operand, one result, the
    // fire's rows and the result's row width. The two scalars are the
    // KDA convention — an unscaled L2 norm at the context's epsilon.
    kernel!(l2norm_scale_to_f32 "ssm::l2norm_scale_bf16_to_fp32",
        operands = operands![
            x: Buf <- Source::In(0),
            y: F32sMut <- Source::Out(0),
            n: I32 <- Source::Rows,
            hidden: I32 <- Source::OutWidth(0),
            scale: F32 <- Source::Lit(Lit::F32(1.0)),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The two casts, and the first rows whose every argument the
    // statement already carries: one operand, one result, and an element
    // count that is the result's own extent.
    kernel!(bf16_to_f32 "ssm::bf16_to_fp32",
        operands = operands![
            x: Buf <- Source::In(0),
            y: F32sMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(f32_to_bf16 "ssm::fp32_to_bf16",
        operands = operands![
            x: F32s <- Source::In(0),
            y: BufMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(zamba_rmsnorm_gated "ssm::zamba_rmsnorm_gated_bf16",
        operands = operands![
            x: Buf,
            gate: Buf,
            weight: Buf,
            y: BufMut,
            n: I32,
            hidden: I32,
            gate_stride: I32,
            group_size: I32,
            eps: F32,
            stream: Stream,
        ]),
    kernel!(build_nemotron_moe_ptrs_aligned "ssm::build_nemotron_moe_ptrs_aligned_bf16",
        whole = true,
        operands = operands![
            expert_ids: I32s,
            up_weight_ptrs: BufArray,
            down_weight_ptrs: BufArray,
            aligned_in: Buf,
            aligned_up: BufMut,
            aligned_act: BufMut,
            aligned_out: BufMut,
            a_up_ptrs: BufArrayOut,
            b_up_ptrs: BufArrayOut,
            c_up_ptrs: BufArrayOutMut,
            a_down_ptrs: BufArrayOut,
            b_down_ptrs: BufArrayOut,
            c_down_ptrs: BufArrayOutMut,
            max_blocks: I32,
            block_size: I32,
            hidden: I32,
            intermediate: I32,
            stream: Stream,
        ]),
    kernel!(build_nemotron_moe_ptrs_decode "ssm::build_nemotron_moe_ptrs_decode_batched_bf16",
        whole = true,
        operands = operands![
            topk_idx: I32s,
            topk_w: F32s,
            up_weight_ptrs: BufArray,
            down_weight_ptrs: BufArray,
            norm_x: Buf,
            expert_up: BufMut,
            expert_act: BufMut,
            expert_out: BufMut,
            a_up_ptrs: BufArrayOut,
            b_up_ptrs: BufArrayOut,
            c_up_ptrs: BufArrayOutMut,
            a_down_ptrs: BufArrayOut,
            b_down_ptrs: BufArrayOut,
            c_down_ptrs: BufArrayOutMut,
            weights_out: F32sMut,
            n: I32,
            top_k: I32,
            hidden: I32,
            intermediate: I32,
            stream: Stream,
        ]),
];
