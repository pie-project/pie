use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attention {
    PlanDecode {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        kv_len: ValueId,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        window: Option<u32>,
        plan: ValueId,
    },
    PlanPrefill {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        kv_len: ValueId,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        window: Option<u32>,
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
    DecodeRel {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        bias: ValueId,
        window: Option<u32>,
        head_dim: u32,
        extent: u32,
        sm_scale: f32,
        log_floor: u32,
        log_alpha: f32,
        o: ValueId,
    },
    PrefillRel {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        bias: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        extent: u32,
        sm_scale: f32,
        log_floor: u32,
        log_alpha: f32,
        o: ValueId,
    },
    Masked {
        q: ValueId,
        plan: ValueId,
        mask: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        causal: bool,
        sm_scale: f32,
        o: ValueId,
    },
    Dense {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        segments: ValueId,
        head_dim: u32,
        sm_scale: f32,
        o: ValueId,
    },
    Ragged {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        q_indptr: ValueId,
        kv_indptr: ValueId,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        mask: RaggedMask,
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
    KvAppendShared {
        plane: ValueId,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
    },

    MlaPlan {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        kv_len: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        plan: ValueId,
    },
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

    SsmCausalConv1d {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        dilation: u32,
        y: ValueId,
    },
    ShortConv {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        y: ValueId,
    },
    ShortConvChunked {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        y: ValueId,
    },
    SsmCausalConv1dChunked {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        dilation: u32,
        y: ValueId,
    },
    BlockDynConv {
        x: ValueId,
        coeff: ValueId,
        base: ValueId,
        side: u32,
        taps: u32,
        group: u32,
        y: ValueId,
    },
    SelectorWalk {
        cand: ValueId,
        unary: ValueId,
        hp: Option<ValueId>,
        tokens: ValueId,
        pred: ValueId,
        succ: ValueId,
        first: u32,
        picks: ValueId,
    },
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
        gate_floor: f32,
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
        gate_floor: f32,
        y: ValueId,
    },

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
    IndexTopk {
        q: ValueId,
        weights: ValueId,
        keys: ValueId,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        ratio: u32,
        selection: ValueId,
    },
    IndexKvAppend {
        k: ValueId,
        keys: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
    },

    PoolBoundaryDecode {
        positions: ValueId,
        row_valid: ValueId,
        ratio: u32,
        boundary_pos: ValueId,
        boundary_req: ValueId,
        boundary_rope: ValueId,
    },
    PoolBoundaryPrefill {
        positions: ValueId,
        row_valid: ValueId,
        ratio: u32,
        boundary_pos: ValueId,
        boundary_req: ValueId,
        boundary_rope: ValueId,
    },
    PoolStateWrite {
        kv: ValueId,
        score: ValueId,
        pages: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
        head_dim: u32,
        ratio: u32,
    },
    PoolGather {
        boundary_pos: ValueId,
        boundary_req: ValueId,
        pages: ValueId,
        ape: Option<ValueId>,
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
    PoolLseSelected {
        q: ValueId,
        positions: ValueId,
        request_of_token: ValueId,
        selection: ValueId,
        entries: ValueId,
        ratio: u32,
        top_k: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        o: ValueId,
        lse: ValueId,
    },

    PleNgramIds {
        ids: ValueId,
        state: ValueId,
        eos: u32,
        mults: Vec<u64>,
        primes: Vec<u64>,
        offsets: Vec<u64>,
        heads_per_ngram: u32,
        ngram_ids: ValueId,
    },
    PleNgramIdsChunked {
        ids: ValueId,
        state: ValueId,
        eos: u32,
        mults: Vec<u64>,
        primes: Vec<u64>,
        offsets: Vec<u64>,
        heads_per_ngram: u32,
        ngram_ids: ValueId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RaggedMask {
    None,
    GroupBlockDiagonal,
    ReferenceSelfOnly { q_tags: ValueId, kv_tags: ValueId },
    RelativeBias { table: ValueId, max_len: u32 },
}

impl Operands for Attention {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::PlanDecode {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                ..
            } => {
                sink.extend([*kv_indptr, *kv_indices, *last_page_len, *kv_len]);
            }
            Self::PlanPrefill {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                ..
            } => {
                sink.extend([*kv_indptr, *kv_indices, *last_page_len, *kv_len]);
            }
            Self::Decode { q, plan, cache, .. } => sink.extend([*q, *plan, *cache]),
            Self::Prefill { q, plan, cache, .. } => sink.extend([*q, *plan, *cache]),
            Self::Masked {
                q,
                plan,
                mask,
                cache,
                ..
            } => sink.extend([*q, *plan, *mask, *cache]),
            Self::Dense {
                q, k, v, segments, ..
            } => sink.extend([*q, *k, *v, *segments]),
            Self::Ragged {
                q,
                k,
                v,
                q_indptr,
                kv_indptr,
                mask,
                ..
            } => {
                sink.extend([*q, *k, *v, *q_indptr, *kv_indptr]);
                match mask {
                    RaggedMask::ReferenceSelfOnly { q_tags, kv_tags } => {
                        sink.extend([*q_tags, *kv_tags]);
                    }
                    RaggedMask::RelativeBias { table, .. } => sink.push(*table),
                    RaggedMask::None | RaggedMask::GroupBlockDiagonal => {}
                }
            }
            Self::DecodeLse { q, plan, cache, .. } => sink.extend([*q, *plan, *cache]),
            Self::PrefillLse { q, plan, cache, .. } => sink.extend([*q, *plan, *cache]),
            Self::DecodeRel {
                q,
                plan,
                cache,
                bias,
                ..
            } => {
                sink.extend([*q, *plan, *cache, *bias]);
            }
            Self::PrefillRel {
                q,
                plan,
                cache,
                bias,
                ..
            } => {
                sink.extend([*q, *plan, *cache, *bias]);
            }
            Self::Sink {
                o,
                lse,
                sink: sink_id,
                ..
            } => sink.extend([*o, *lse, *sink_id]),
            Self::MergeLse {
                o1, lse1, o2, lse2, ..
            } => sink.extend([*o1, *lse1, *o2, *lse2]),
            Self::LogitSoftcap { x, .. } => sink.push(*x),
            Self::KvAppend {
                k,
                v,
                cache,
                write_page,
                write_offset,
            } => {
                sink.extend([*k, *v, *cache, *write_page, *write_offset]);
            }
            Self::KvAppendShared {
                plane,
                cache,
                write_page,
                write_offset,
            } => {
                sink.extend([*plane, *cache, *write_page, *write_offset]);
            }
            Self::MlaPlan {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                ..
            } => {
                sink.extend([*kv_indptr, *kv_indices, *last_page_len, *kv_len]);
            }
            Self::MlaLatents { kv_a, weight, .. } => sink.extend([*kv_a, *weight]),
            Self::MlaLatentsRope {
                kv_a,
                positions,
                weight,
                ..
            } => {
                sink.extend([*kv_a, *positions, *weight]);
            }
            Self::MlaSplitQB { q_b, .. } => sink.push(*q_b),
            Self::MlaAbsorbQ { q_nope, kv_b, .. } => sink.extend([*q_nope, *kv_b]),
            Self::MlaAbsorbOut { latent, kv_b, .. } => sink.extend([*latent, *kv_b]),
            Self::MlaKvAppend {
                kv_c,
                k_pe,
                cache,
                write_page,
                write_offset,
            } => {
                sink.extend([*kv_c, *k_pe, *cache, *write_page, *write_offset]);
            }
            Self::MlaDecode {
                q,
                plan,
                q_pe,
                cache,
                ..
            } => {
                sink.extend([*q, *plan, *q_pe, *cache]);
            }
            Self::MlaPrefill {
                q,
                plan,
                q_pe,
                cache,
                ..
            } => {
                sink.extend([*q, *plan, *q_pe, *cache]);
            }
            Self::MlaDecodeSelected {
                q,
                plan,
                q_pe,
                selection,
                cache,
                ..
            } => {
                sink.extend([*q, *plan, *q_pe, *selection, *cache]);
            }
            Self::MlaPrefillSelected {
                q,
                plan,
                q_pe,
                selection,
                cache,
                ..
            } => {
                sink.extend([*q, *plan, *q_pe, *selection, *cache]);
            }
            Self::SsmCausalConv1d {
                x, weight, state, ..
            } => sink.extend([*x, *weight, *state]),
            Self::ShortConv {
                x, weight, state, ..
            } => sink.extend([*x, *weight, *state]),
            Self::ShortConvChunked {
                x, weight, state, ..
            } => sink.extend([*x, *weight, *state]),
            Self::SsmCausalConv1dChunked {
                x, weight, state, ..
            } => {
                sink.extend([*x, *weight, *state]);
            }
            Self::BlockDynConv { x, coeff, base, .. } => sink.extend([*x, *coeff, *base]),
            Self::SelectorWalk {
                cand,
                unary,
                hp,
                tokens,
                pred,
                succ,
                ..
            } => {
                sink.extend([*cand, *unary]);
                sink.extend(hp.iter().copied());
                sink.extend([*tokens, *pred, *succ]);
            }
            Self::SsmGdnPrep {
                ba, dt_bias, a_log, ..
            } => sink.extend([*ba, *dt_bias, *a_log]),
            Self::SsmGatedDelta {
                qkv,
                z,
                gates,
                state,
                ..
            } => {
                sink.extend([*qkv, *z, *gates, *state]);
            }
            Self::SsmGatedDeltaChunked {
                qkv,
                z,
                gates,
                state,
                ..
            } => {
                sink.extend([*qkv, *z, *gates, *state]);
            }
            Self::SsmKdaStep {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                ..
            } => {
                sink.extend([*mixed, *f, *b, *dt_bias, *a_log, *state]);
            }
            Self::SsmKdaChunked {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                state,
                ..
            } => {
                sink.extend([*mixed, *f, *b, *dt_bias, *a_log, *state]);
            }
            Self::IndexLayernormRope {
                k,
                positions,
                weight,
                bias,
                ..
            } => {
                sink.extend([*k, *positions, *weight, *bias]);
            }
            Self::IndexRope { q, positions, .. } => sink.extend([*q, *positions]),
            Self::IndexTopk {
                q, weights, keys, ..
            } => sink.extend([*q, *weights, *keys]),
            Self::IndexKvAppend {
                k,
                keys,
                write_page,
                write_offset,
            } => {
                sink.extend([*k, *keys, *write_page, *write_offset]);
            }
            Self::PoolBoundaryDecode {
                positions,
                row_valid,
                ..
            } => {
                sink.extend([*positions, *row_valid]);
            }
            Self::PoolBoundaryPrefill {
                positions,
                row_valid,
                ..
            } => {
                sink.extend([*positions, *row_valid]);
            }
            Self::PoolStateWrite {
                kv,
                score,
                pages,
                write_page,
                write_offset,
                ..
            } => {
                sink.extend([*kv, *score, *pages, *write_page, *write_offset]);
            }
            Self::PoolGather {
                boundary_pos,
                boundary_req,
                pages,
                ape,
                ..
            } => {
                sink.extend([*boundary_pos, *boundary_req, *pages]);
                sink.extend(ape.iter().copied());
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
            Self::PoolLse {
                q,
                positions,
                request_of_token,
                entries,
                ..
            } => {
                sink.extend([*q, *positions, *request_of_token, *entries]);
            }
            Self::PoolLseSelected {
                q,
                positions,
                request_of_token,
                selection,
                entries,
                ..
            } => {
                sink.extend([*q, *positions, *request_of_token, *selection, *entries]);
            }
            Self::PleNgramIds { ids, state, .. } => sink.extend([*ids, *state]),
            Self::PleNgramIdsChunked { ids, state, .. } => sink.extend([*ids, *state]),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::PlanDecode { plan, .. } => sink.push(*plan),
            Self::PlanPrefill { plan, .. } => sink.push(*plan),
            Self::Decode { o, .. } => sink.push(*o),
            Self::Prefill { o, .. } => sink.push(*o),
            Self::Masked { o, .. } => sink.push(*o),
            Self::Dense { o, .. } => sink.push(*o),
            Self::Ragged { o, .. } => sink.push(*o),
            Self::DecodeLse { o, lse, .. } => sink.extend([*o, *lse]),
            Self::PrefillLse { o, lse, .. } => sink.extend([*o, *lse]),
            Self::DecodeRel { o, .. } => sink.push(*o),
            Self::PrefillRel { o, .. } => sink.push(*o),
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
            Self::ShortConv { y, .. } => sink.push(*y),
            Self::ShortConvChunked { y, .. } => sink.push(*y),
            Self::SsmCausalConv1dChunked { y, .. } => sink.push(*y),
            Self::BlockDynConv { y, .. } => sink.push(*y),
            Self::SelectorWalk { picks, .. } => sink.push(*picks),
            Self::SsmGdnPrep { gates, .. } => sink.push(*gates),
            Self::SsmGatedDelta { y, .. } => sink.push(*y),
            Self::SsmGatedDeltaChunked { y, .. } => sink.push(*y),
            Self::SsmKdaStep { y, .. } => sink.push(*y),
            Self::SsmKdaChunked { y, .. } => sink.push(*y),
            Self::IndexLayernormRope { k_out, .. } => sink.push(*k_out),
            Self::IndexRope { q_out, .. } => sink.push(*q_out),
            Self::IndexTopk { selection, .. } => sink.push(*selection),
            Self::IndexKvAppend { .. } => {}
            Self::PoolBoundaryDecode {
                boundary_pos,
                boundary_req,
                boundary_rope,
                ..
            } => {
                sink.extend([*boundary_pos, *boundary_req, *boundary_rope]);
            }
            Self::PoolBoundaryPrefill {
                boundary_pos,
                boundary_req,
                boundary_rope,
                ..
            } => {
                sink.extend([*boundary_pos, *boundary_req, *boundary_rope]);
            }
            Self::PoolStateWrite { .. } => {}
            Self::PoolGather { entries, .. } => sink.push(*entries),
            Self::PoolKvAppend { .. } => {}
            Self::PoolLse { o, lse, .. } => sink.extend([*o, *lse]),
            Self::PoolLseSelected { o, lse, .. } => sink.extend([*o, *lse]),
            Self::PleNgramIds { ngram_ids, .. } => sink.push(*ngram_ids),
            Self::PleNgramIdsChunked { ngram_ids, .. } => sink.push(*ngram_ids),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::PlanDecode { .. } => {}
            Self::PlanPrefill { .. } => {}
            Self::Decode { .. } => {}
            Self::Prefill { .. } => {}
            Self::Masked { .. } => {}
            Self::Dense { .. } => {}
            Self::Ragged { .. } => {}
            Self::DecodeLse { .. } => {}
            Self::PrefillLse { .. } => {}
            Self::DecodeRel { .. } => {}
            Self::PrefillRel { .. } => {}
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
            Self::ShortConv { .. } => {}
            Self::ShortConvChunked { .. } => {}
            Self::BlockDynConv { .. } => {}
            Self::SelectorWalk { .. } => {}
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
            Self::PoolStateWrite { .. } => {}
            Self::PoolGather { .. } => {}
            Self::PoolKvAppend { .. } => {}
            Self::PoolLse { .. } => {}
            Self::PoolLseSelected { .. } => {}
            Self::PleNgramIds { .. } => {}
            Self::PleNgramIdsChunked { .. } => {}
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::PlanDecode { .. } => "attention.plan_decode",
            Self::PlanPrefill { .. } => "attention.plan_prefill",
            Self::Decode { .. } => "attention.decode",
            Self::Prefill { .. } => "attention.prefill",
            Self::Masked { .. } => "attention.masked",
            Self::Dense { .. } => "attention.dense",
            Self::Ragged { .. } => "attention.ragged",
            Self::DecodeLse { .. } => "attention.decode_lse",
            Self::PrefillLse { .. } => "attention.prefill_lse",
            Self::DecodeRel { .. } => "attention.decode_rel",
            Self::PrefillRel { .. } => "attention.prefill_rel",
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
            Self::ShortConv { .. } => "attention.short_conv",
            Self::ShortConvChunked { .. } => "attention.short_conv_chunked",
            Self::BlockDynConv { .. } => "attention.block_dyn_conv",
            Self::SelectorWalk { .. } => "attention.selector_walk",
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
            Self::PoolStateWrite { .. } => "attention.pool_state_write",
            Self::PoolGather { .. } => "attention.pool_gather",
            Self::PoolKvAppend { .. } => "attention.pool_kv_append",
            Self::PoolLse { .. } => "attention.pool_lse",
            Self::PoolLseSelected { .. } => "attention.pool_lse_selected",
            Self::PleNgramIds { .. } => "attention.ple_ngram_ids",
            Self::PleNgramIdsChunked { .. } => "attention.ple_ngram_ids_chunked",
        }
    }
}
