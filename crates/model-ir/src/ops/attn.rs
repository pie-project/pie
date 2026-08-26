use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// Ops where tokens interact, or where a sequence cache — kv pages, ssm
/// recurrent state, the indexer's key cache, the compressor's pooled entries —
/// is touched. Plans are explicit ops: the `Plan*` variants define `Struct`
/// values from declared geometry inputs, and every variant that walks a cache
/// takes the plan it was built from — `cache` is the pool pointer, nothing
/// more. The append ops carry their write addressing
/// (`write_page`/`write_offset`) the same way.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attention {
    /// Defines `Struct(AttnDecodePlan)`. Host work; runs in the prepare phase.
    PlanDecode {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        kv_len: ValueId,
        plan: ValueId,
    },
    /// Defines `Struct(AttnPrefillPlan)`.
    PlanPrefill {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        kv_len: ValueId,
        plan: ValueId,
    },
    Decode {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        o: ValueId,
    },
    Prefill {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: ValueId,
    },
    /// Prefill against a query-provided mask instead of the causal one;
    /// the op names the `mask` it applies, not the driver.
    Masked {
        q: ValueId,
        plan: ValueId,
        mask: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        o: ValueId,
    },
    DecodeLse {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        o: ValueId,
        lse: ValueId,
    },
    PrefillLse {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: ValueId,
        lse: ValueId,
    },
    /// Folds attention-sink mass into `o` using its log-sum-exp.
    Sink {
        o: ValueId,
        lse: ValueId,
        sink: ValueId,
        head_dim: u32,
        o_out: ValueId,
    },
    MergeLse {
        o1: ValueId,
        lse1: ValueId,
        o2: ValueId,
        lse2: ValueId,
        heads: u32,
        head_dim: u32,
        o: ValueId,
        lse: ValueId,
    },
    LogitSoftcap {
        x: ValueId,
        cap: f32,
        x_out: ValueId,
    },
    KvAppend {
        k: ValueId,
        v: ValueId,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
    },
    /// Appends one plane shared as both k and v.
    KvAppendShared {
        plane: ValueId,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
    },

    // Multi-head latent attention. Same plan discipline as the paged variants
    // above: one `MlaPlan` op defines the struct, the four cache-walking
    // variants take it, and `MlaKvAppend` carries its write addressing. The
    // absorb/split variants are pure math and take nothing but tensors.
    /// Defines `Struct(MlaPlan)`, shared by decode and prefill.
    MlaPlan {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        kv_len: ValueId,
        plan: ValueId,
    },
    /// Splits `kv_a` into the rmsnormed compressed latent and the rope plane.
    MlaLatents {
        kv_a: ValueId,
        weight: ValueId,
        eps: f32,
        kv_lora_rank: u32,
        kv_c: ValueId,
        k_pe: ValueId,
    },
    MlaLatentsRope {
        kv_a: ValueId,
        positions: ValueId,
        weight: ValueId,
        eps: f32,
        kv_lora_rank: u32,
        rope_dim: u32,
        theta: f32,
        kv_c: ValueId,
        k_pe: ValueId,
    },
    MlaSplitQB {
        q_b: ValueId,
        heads: u32,
        nope_dim: u32,
        rope_dim: u32,
        q_nope: ValueId,
        q_pe: ValueId,
    },
    /// Absorbs `kv_b`'s up-projection into q, mapping heads into latent space.
    MlaAbsorbQ {
        q_nope: ValueId,
        kv_b: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        q_latent: ValueId,
    },
    MlaAbsorbOut {
        latent: ValueId,
        kv_b: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
        o: ValueId,
    },
    MlaKvAppend {
        kv_c: ValueId,
        k_pe: ValueId,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
    },
    MlaDecode {
        q: ValueId,
        plan: ValueId,
        q_pe: ValueId,
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: ValueId,
    },
    MlaPrefill {
        q: ValueId,
        plan: ValueId,
        q_pe: ValueId,
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: ValueId,
    },
    /// Decode over the sparse `selection` produced by `IndexTopk`.
    MlaDecodeSelected {
        q: ValueId,
        plan: ValueId,
        q_pe: ValueId,
        selection: ValueId,
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: ValueId,
    },
    MlaPrefillSelected {
        q: ValueId,
        plan: ValueId,
        q_pe: ValueId,
        selection: ValueId,
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: ValueId,
    },

    // Recurrent-state mixers: causal conv, gated delta nets, KDA. `state` is
    // the recurrent cache — storage only, updated in place by the kernel.
    SsmCausalConv1d {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        y: ValueId,
    },
    /// Prefill form: walks the fire's ambient request boundaries.
    SsmCausalConv1dChunked {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        y: ValueId,
    },
    /// Folds `ba` with dt bias and A-log into per-head decay gates.
    SsmGdnPrep {
        ba: ValueId,
        dt_bias: ValueId,
        a_log: ValueId,
        gates: ValueId,
    },
    SsmGatedDelta {
        qkv: ValueId,
        z: ValueId,
        gates: ValueId,
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: ValueId,
    },
    SsmGatedDeltaChunked {
        qkv: ValueId,
        z: ValueId,
        gates: ValueId,
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: ValueId,
    },
    SsmKdaStep {
        mixed: ValueId,
        f: ValueId,
        b: ValueId,
        dt_bias: ValueId,
        a_log: ValueId,
        state: ValueId,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: ValueId,
    },
    SsmKdaChunked {
        mixed: ValueId,
        f: ValueId,
        b: ValueId,
        dt_bias: ValueId,
        a_log: ValueId,
        state: ValueId,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: ValueId,
    },

    // The sparse-attention indexer: a small key cache (`keys`) scored against
    // queries to select which pages the main attention will read.
    IndexLayernormRope {
        k: ValueId,
        positions: ValueId,
        weight: ValueId,
        bias: ValueId,
        eps: f32,
        rope_dim: u32,
        theta: f32,
        k_out: ValueId,
    },
    IndexRope {
        q: ValueId,
        positions: ValueId,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
        q_out: ValueId,
    },
    /// Scores `q` against the cached keys; `selection` is the top-k page ids.
    IndexTopk {
        q: ValueId,
        weights: ValueId,
        keys: ValueId,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        selection: ValueId,
    },
    IndexKvAppend {
        k: ValueId,
        keys: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
    },

    // Pooled (compressed) attention: every `ratio` tokens close a boundary
    // whose pooled entry lands in its own cache. Boundary outputs are
    // token-shaped — over-allocated with a sentinel in the non-boundary rows —
    // so no counted dim exists and the shapes stay trace-time facts.
    /// `row_valid` masks graph-padding rows out of the boundary math.
    PoolBoundaryDecode {
        positions: ValueId,
        row_valid: ValueId,
        ratio: u32,
        boundary_pos: ValueId,
        boundary_req: ValueId,
    },
    PoolBoundaryPrefill {
        positions: ValueId,
        row_valid: ValueId,
        ratio: u32,
        boundary_pos: ValueId,
        boundary_req: ValueId,
    },
    /// Pools the closing window out of the kv cache into per-boundary entries.
    PoolGather {
        boundary_pos: ValueId,
        boundary_req: ValueId,
        pages: ValueId,
        head_dim: u32,
        ratio: u32,
        entries: ValueId,
    },
    PoolKvAppend {
        entries: ValueId,
        boundary_pos: ValueId,
        boundary_req: ValueId,
        pool: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
    },
    /// Attends each token over the pooled entries of its own request:
    /// `request_of_token` maps tokens to lanes, and `entries` is the pool
    /// cache space itself — not `PoolGather`'s tensor, which reaches it
    /// through `PoolKvAppend`.
    PoolLse {
        q: ValueId,
        positions: ValueId,
        request_of_token: ValueId,
        entries: ValueId,
        ratio: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        o: ValueId,
        lse: ValueId,
    },
}

impl Operands for Attention {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::PlanDecode { kv_indptr, kv_indices, last_page_len, kv_len, .. } => {
                sink.extend([*kv_indptr, *kv_indices, *last_page_len, *kv_len]);
            }
            Self::PlanPrefill { kv_indptr, kv_indices, last_page_len, kv_len, .. } => {
                sink.extend([*kv_indptr, *kv_indices, *last_page_len, *kv_len]);
            }
            Self::Decode { q, plan, cache, .. } => sink.extend([*q, *plan, *cache]),
            Self::Prefill { q, plan, cache, .. } => sink.extend([*q, *plan, *cache]),
            Self::Masked { q, plan, mask, cache, .. } => sink.extend([*q, *plan, *mask, *cache]),
            Self::DecodeLse { q, plan, cache, .. } => sink.extend([*q, *plan, *cache]),
            Self::PrefillLse { q, plan, cache, .. } => sink.extend([*q, *plan, *cache]),
            // The `sink` input field is bound as `sink_id`: its name collides with
            // the `sink` parameter this arm pushes into.
            Self::Sink { o, lse, sink: sink_id, .. } => sink.extend([*o, *lse, *sink_id]),
            Self::MergeLse { o1, lse1, o2, lse2, .. } => sink.extend([*o1, *lse1, *o2, *lse2]),
            Self::LogitSoftcap { x, .. } => sink.push(*x),
            Self::KvAppend { k, v, cache, write_page, write_offset } => {
                sink.extend([*k, *v, *cache, *write_page, *write_offset]);
            }
            Self::KvAppendShared { plane, cache, write_page, write_offset } => {
                sink.extend([*plane, *cache, *write_page, *write_offset]);
            }
            Self::MlaPlan { kv_indptr, kv_indices, last_page_len, kv_len, .. } => {
                sink.extend([*kv_indptr, *kv_indices, *last_page_len, *kv_len]);
            }
            Self::MlaLatents { kv_a, weight, .. } => sink.extend([*kv_a, *weight]),
            Self::MlaLatentsRope { kv_a, positions, weight, .. } => {
                sink.extend([*kv_a, *positions, *weight]);
            }
            Self::MlaSplitQB { q_b, .. } => sink.push(*q_b),
            Self::MlaAbsorbQ { q_nope, kv_b, .. } => sink.extend([*q_nope, *kv_b]),
            Self::MlaAbsorbOut { latent, kv_b, .. } => sink.extend([*latent, *kv_b]),
            Self::MlaKvAppend { kv_c, k_pe, cache, write_page, write_offset } => {
                sink.extend([*kv_c, *k_pe, *cache, *write_page, *write_offset]);
            }
            Self::MlaDecode { q, plan, q_pe, cache, .. } => {
                sink.extend([*q, *plan, *q_pe, *cache]);
            }
            Self::MlaPrefill { q, plan, q_pe, cache, .. } => {
                sink.extend([*q, *plan, *q_pe, *cache]);
            }
            Self::MlaDecodeSelected { q, plan, q_pe, selection, cache, .. } => {
                sink.extend([*q, *plan, *q_pe, *selection, *cache]);
            }
            Self::MlaPrefillSelected { q, plan, q_pe, selection, cache, .. } => {
                sink.extend([*q, *plan, *q_pe, *selection, *cache]);
            }
            Self::SsmCausalConv1d { x, weight, state, .. } => sink.extend([*x, *weight, *state]),
            Self::SsmCausalConv1dChunked { x, weight, state, .. } => {
                sink.extend([*x, *weight, *state]);
            }
            Self::SsmGdnPrep { ba, dt_bias, a_log, .. } => sink.extend([*ba, *dt_bias, *a_log]),
            Self::SsmGatedDelta { qkv, z, gates, state, .. } => {
                sink.extend([*qkv, *z, *gates, *state]);
            }
            Self::SsmGatedDeltaChunked { qkv, z, gates, state, .. } => {
                sink.extend([*qkv, *z, *gates, *state]);
            }
            Self::SsmKdaStep { mixed, f, b, dt_bias, a_log, state, .. } => {
                sink.extend([*mixed, *f, *b, *dt_bias, *a_log, *state]);
            }
            Self::SsmKdaChunked { mixed, f, b, dt_bias, a_log, state, .. } => {
                sink.extend([*mixed, *f, *b, *dt_bias, *a_log, *state]);
            }
            Self::IndexLayernormRope { k, positions, weight, bias, .. } => {
                sink.extend([*k, *positions, *weight, *bias]);
            }
            Self::IndexRope { q, positions, .. } => sink.extend([*q, *positions]),
            Self::IndexTopk { q, weights, keys, .. } => sink.extend([*q, *weights, *keys]),
            Self::IndexKvAppend { k, keys, write_page, write_offset } => {
                sink.extend([*k, *keys, *write_page, *write_offset]);
            }
            Self::PoolBoundaryDecode { positions, row_valid, .. } => {
                sink.extend([*positions, *row_valid]);
            }
            Self::PoolBoundaryPrefill { positions, row_valid, .. } => {
                sink.extend([*positions, *row_valid]);
            }
            Self::PoolGather { boundary_pos, boundary_req, pages, .. } => {
                sink.extend([*boundary_pos, *boundary_req, *pages]);
            }
            Self::PoolKvAppend {
                entries,
                boundary_pos,
                boundary_req,
                pool,
                write_page,
                write_offset,
            } => {
                sink.extend([
                    *entries,
                    *boundary_pos,
                    *boundary_req,
                    *pool,
                    *write_page,
                    *write_offset,
                ]);
            }
            Self::PoolLse { q, positions, request_of_token, entries, .. } => {
                sink.extend([*q, *positions, *request_of_token, *entries]);
            }
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::PlanDecode { plan, .. } => sink.push(*plan),
            Self::PlanPrefill { plan, .. } => sink.push(*plan),
            Self::Decode { o, .. } => sink.push(*o),
            Self::Prefill { o, .. } => sink.push(*o),
            Self::Masked { o, .. } => sink.push(*o),
            Self::DecodeLse { o, lse, .. } => sink.extend([*o, *lse]),
            Self::PrefillLse { o, lse, .. } => sink.extend([*o, *lse]),
            Self::Sink { o_out, .. } => sink.push(*o_out),
            Self::MergeLse { o, lse, .. } => sink.extend([*o, *lse]),
            Self::LogitSoftcap { x_out, .. } => sink.push(*x_out),
            Self::KvAppend { .. } => {}
            Self::KvAppendShared { .. } => {}
            Self::MlaPlan { plan, .. } => sink.push(*plan),
            Self::MlaLatents { kv_c, k_pe, .. } => sink.extend([*kv_c, *k_pe]),
            Self::MlaLatentsRope { kv_c, k_pe, .. } => sink.extend([*kv_c, *k_pe]),
            Self::MlaSplitQB { q_nope, q_pe, .. } => sink.extend([*q_nope, *q_pe]),
            Self::MlaAbsorbQ { q_latent, .. } => sink.push(*q_latent),
            Self::MlaAbsorbOut { o, .. } => sink.push(*o),
            Self::MlaKvAppend { .. } => {}
            Self::MlaDecode { o, .. } => sink.push(*o),
            Self::MlaPrefill { o, .. } => sink.push(*o),
            Self::MlaDecodeSelected { o, .. } => sink.push(*o),
            Self::MlaPrefillSelected { o, .. } => sink.push(*o),
            Self::SsmCausalConv1d { y, .. } => sink.push(*y),
            Self::SsmCausalConv1dChunked { y, .. } => sink.push(*y),
            Self::SsmGdnPrep { gates, .. } => sink.push(*gates),
            Self::SsmGatedDelta { y, .. } => sink.push(*y),
            Self::SsmGatedDeltaChunked { y, .. } => sink.push(*y),
            Self::SsmKdaStep { y, .. } => sink.push(*y),
            Self::SsmKdaChunked { y, .. } => sink.push(*y),
            Self::IndexLayernormRope { k_out, .. } => sink.push(*k_out),
            Self::IndexRope { q_out, .. } => sink.push(*q_out),
            Self::IndexTopk { selection, .. } => sink.push(*selection),
            Self::IndexKvAppend { .. } => {}
            Self::PoolBoundaryDecode { boundary_pos, boundary_req, .. } => {
                sink.extend([*boundary_pos, *boundary_req]);
            }
            Self::PoolBoundaryPrefill { boundary_pos, boundary_req, .. } => {
                sink.extend([*boundary_pos, *boundary_req]);
            }
            Self::PoolGather { entries, .. } => sink.push(*entries),
            Self::PoolKvAppend { .. } => {}
            Self::PoolLse { o, lse, .. } => sink.extend([*o, *lse]),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::PlanDecode { .. } => {}
            Self::PlanPrefill { .. } => {}
            Self::Decode { .. } => {}
            Self::Prefill { .. } => {}
            Self::Masked { .. } => {}
            Self::DecodeLse { .. } => {}
            Self::PrefillLse { .. } => {}
            Self::Sink { o_out, o, .. } => sink.push((*o_out, *o)),
            Self::MergeLse { .. } => {}
            Self::LogitSoftcap { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::KvAppend { .. } => {}
            Self::KvAppendShared { .. } => {}
            Self::MlaPlan { .. } => {}
            Self::MlaLatents { .. } => {}
            Self::MlaLatentsRope { .. } => {}
            Self::MlaSplitQB { .. } => {}
            Self::MlaAbsorbQ { .. } => {}
            Self::MlaAbsorbOut { .. } => {}
            Self::MlaKvAppend { .. } => {}
            Self::MlaDecode { .. } => {}
            Self::MlaPrefill { .. } => {}
            Self::MlaDecodeSelected { .. } => {}
            Self::MlaPrefillSelected { .. } => {}
            Self::SsmCausalConv1d { .. } => {}
            Self::SsmCausalConv1dChunked { .. } => {}
            Self::SsmGdnPrep { .. } => {}
            Self::SsmGatedDelta { .. } => {}
            Self::SsmGatedDeltaChunked { .. } => {}
            Self::SsmKdaStep { .. } => {}
            Self::SsmKdaChunked { .. } => {}
            Self::IndexLayernormRope { k_out, k, .. } => sink.push((*k_out, *k)),
            Self::IndexRope { q_out, q, .. } => sink.push((*q_out, *q)),
            Self::IndexTopk { .. } => {}
            Self::IndexKvAppend { .. } => {}
            Self::PoolBoundaryDecode { .. } => {}
            Self::PoolBoundaryPrefill { .. } => {}
            Self::PoolGather { .. } => {}
            Self::PoolKvAppend { .. } => {}
            Self::PoolLse { .. } => {}
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::PlanDecode { .. } => "attention.plan_decode",
            Self::PlanPrefill { .. } => "attention.plan_prefill",
            Self::Decode { .. } => "attention.decode",
            Self::Prefill { .. } => "attention.prefill",
            Self::Masked { .. } => "attention.masked",
            Self::DecodeLse { .. } => "attention.decode_lse",
            Self::PrefillLse { .. } => "attention.prefill_lse",
            Self::Sink { .. } => "attention.sink",
            Self::MergeLse { .. } => "attention.merge_lse",
            Self::LogitSoftcap { .. } => "attention.logit_softcap",
            Self::KvAppend { .. } => "attention.kv_append",
            Self::KvAppendShared { .. } => "attention.kv_append_shared",
            Self::MlaPlan { .. } => "attention.mla_plan",
            Self::MlaLatents { .. } => "attention.mla_latents",
            Self::MlaLatentsRope { .. } => "attention.mla_latents_rope",
            Self::MlaSplitQB { .. } => "attention.mla_split_q_b",
            Self::MlaAbsorbQ { .. } => "attention.mla_absorb_q",
            Self::MlaAbsorbOut { .. } => "attention.mla_absorb_out",
            Self::MlaKvAppend { .. } => "attention.mla_kv_append",
            Self::MlaDecode { .. } => "attention.mla_decode",
            Self::MlaPrefill { .. } => "attention.mla_prefill",
            Self::MlaDecodeSelected { .. } => "attention.mla_decode_selected",
            Self::MlaPrefillSelected { .. } => "attention.mla_prefill_selected",
            Self::SsmCausalConv1d { .. } => "attention.ssm_causal_conv1d",
            Self::SsmCausalConv1dChunked { .. } => "attention.ssm_causal_conv1d_chunked",
            Self::SsmGdnPrep { .. } => "attention.ssm_gdn_prep",
            Self::SsmGatedDelta { .. } => "attention.ssm_gated_delta",
            Self::SsmGatedDeltaChunked { .. } => "attention.ssm_gated_delta_chunked",
            Self::SsmKdaStep { .. } => "attention.ssm_kda_step",
            Self::SsmKdaChunked { .. } => "attention.ssm_kda_chunked",
            Self::IndexLayernormRope { .. } => "attention.index_layernorm_rope",
            Self::IndexRope { .. } => "attention.index_rope",
            Self::IndexTopk { .. } => "attention.index_topk",
            Self::IndexKvAppend { .. } => "attention.index_kv_append",
            Self::PoolBoundaryDecode { .. } => "attention.pool_boundary_decode",
            Self::PoolBoundaryPrefill { .. } => "attention.pool_boundary_prefill",
            Self::PoolGather { .. } => "attention.pool_gather",
            Self::PoolKvAppend { .. } => "attention.pool_kv_append",
            Self::PoolLse { .. } => "attention.pool_lse",
        }
    }
}
