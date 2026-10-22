use serde::{Deserialize, Serialize};

use crate::Op;
use crate::check::Expect;
use crate::operands::Part;
use crate::ops::conv::VoxelSegment;
use crate::value::ValueId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[op(group = "attn")]
pub enum Attn {
    PlanDecode {
        #[input(I32)]
        kv_indptr: ValueId,
        #[input(I32)]
        kv_indices: ValueId,
        #[input(I32)]
        last_page_len: ValueId,
        #[input(I32)]
        kv_len: ValueId,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        window: Option<u32>,
        #[output]
        plan: ValueId,
    },
    PlanPrefill {
        #[input(I32)]
        kv_indptr: ValueId,
        #[input(I32)]
        kv_indices: ValueId,
        #[input(I32)]
        last_page_len: ValueId,
        #[input(I32)]
        kv_len: ValueId,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        window: Option<u32>,
        #[output]
        plan: ValueId,
    },
    Decode {
        #[input]
        q: ValueId,
        #[input(DECODE_PLAN)]
        plan: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
    },
    Prefill {
        #[input]
        q: ValueId,
        #[input(PREFILL_PLAN)]
        plan: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
    },
    DecodeRel {
        #[input]
        q: ValueId,
        #[input(DECODE_PLAN)]
        plan: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[input(F32)]
        bias: ValueId,
        window: Option<u32>,
        head_dim: u32,
        extent: u32,
        sm_scale: f32,
        log_floor: u32,
        log_alpha: f32,
        #[output]
        o: ValueId,
    },
    PrefillRel {
        #[input]
        q: ValueId,
        #[input(PREFILL_PLAN)]
        plan: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[input(F32)]
        bias: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        extent: u32,
        sm_scale: f32,
        log_floor: u32,
        log_alpha: f32,
        #[output]
        o: ValueId,
    },
    Masked {
        #[input]
        q: ValueId,
        #[input(PREFILL_PLAN)]
        plan: ValueId,
        #[input]
        mask: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        causal: bool,
        sm_scale: f32,
        #[output]
        o: ValueId,
    },
    MaskedLse {
        #[input]
        q: ValueId,
        #[input(PREFILL_PLAN)]
        plan: ValueId,
        #[input]
        mask: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        causal: bool,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: ValueId,
    },
    Dense {
        #[input]
        q: ValueId,
        #[input]
        k: ValueId,
        #[input]
        v: ValueId,
        #[input(I32)]
        segments: ValueId,
        head_dim: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
    },
    Ragged {
        #[input]
        q: ValueId,
        #[input]
        k: ValueId,
        #[input]
        v: ValueId,
        #[input(I32)]
        q_indptr: ValueId,
        #[input(I32)]
        kv_indptr: ValueId,
        #[part]
        mask: RaggedMask,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
    },
    /// Bidirectional attention within each clip's voxel rows (or blocks of
    /// its frames), as `grid` (i32) lays them out.
    Voxel {
        #[input]
        q: ValueId,
        #[input]
        k: ValueId,
        #[input]
        v: ValueId,
        #[input(I32)]
        grid: ValueId,
        segment: VoxelSegment,
        sm_scale: f32,
        #[output]
        o: ValueId,
    },
    DecodeLse {
        #[input]
        q: ValueId,
        #[input(DECODE_PLAN)]
        plan: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: ValueId,
    },
    PrefillLse {
        #[input]
        q: ValueId,
        #[input(PREFILL_PLAN)]
        plan: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        /// Each row's least window position (i32 a row, from
        /// `attn.replay_floor`): the window reads no key below it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[input(I32)]
        floor: Option<ValueId>,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: ValueId,
    },
    DecodeSelected {
        #[input]
        q: ValueId,
        #[input(DECODE_PLAN)]
        plan: ValueId,
        #[input(I32)]
        selection: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        ratio: u32,
        /// The selection holds one run of blocks per kv head, each head's
        /// queries reading its own; else one run every head shares.
        #[serde(default)]
        per_head: bool,
        #[output]
        o: ValueId,
    },
    PrefillSelected {
        #[input]
        q: ValueId,
        #[input(PREFILL_PLAN)]
        plan: ValueId,
        #[input(I32)]
        selection: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        ratio: u32,
        #[serde(default)]
        per_head: bool,
        #[output]
        o: ValueId,
    },
    Sink {
        #[input]
        o: ValueId,
        #[input(F32)]
        lse: ValueId,
        sink: ValueId,
        head_dim: u32,
        #[output(alias = o)]
        o_out: ValueId,
    },
    MergeLse {
        #[input]
        o1: ValueId,
        #[input(F32)]
        lse1: ValueId,
        #[input]
        o2: ValueId,
        #[input(F32)]
        lse2: ValueId,
        heads: u32,
        head_dim: u32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: ValueId,
    },
    LogitSoftcap {
        #[input]
        x: ValueId,
        cap: f32,
        #[output(alias = x)]
        x_out: ValueId,
    },
    KvAppend {
        #[input]
        k: ValueId,
        #[input]
        v: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
    },
    KvAppendShared {
        #[input]
        plane: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
    },
    MlaPlan {
        #[input(I32)]
        kv_indptr: ValueId,
        #[input(I32)]
        kv_indices: ValueId,
        #[input(I32)]
        last_page_len: ValueId,
        #[input(I32)]
        kv_len: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        #[output]
        plan: ValueId,
    },
    MlaLatents {
        #[input]
        kv_a: ValueId,
        #[input]
        weight: ValueId,
        eps: f32,
        kv_lora_rank: u32,
        #[output]
        kv_c: ValueId,
        #[output]
        k_pe: ValueId,
    },
    MlaLatentsRope {
        #[input]
        kv_a: ValueId,
        #[input(I32)]
        positions: ValueId,
        #[input]
        weight: ValueId,
        eps: f32,
        kv_lora_rank: u32,
        rope_dim: u32,
        theta: f32,
        #[output]
        kv_c: ValueId,
        #[output]
        k_pe: ValueId,
    },
    MlaSplitQB {
        #[input]
        q_b: ValueId,
        heads: u32,
        nope_dim: u32,
        rope_dim: u32,
        #[output]
        q_nope: ValueId,
        #[output]
        q_pe: ValueId,
    },
    MlaAbsorbQ {
        #[input]
        q_nope: ValueId,
        #[input]
        kv_b: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        #[output]
        q_latent: ValueId,
    },
    MlaAbsorbOut {
        #[input]
        latent: ValueId,
        #[input]
        kv_b: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
        #[output]
        o: ValueId,
    },
    MlaKvAppend {
        #[input]
        kv_c: ValueId,
        #[input]
        k_pe: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
    },
    MlaDecode {
        #[input]
        q: ValueId,
        #[input(MLA_PLAN)]
        plan: ValueId,
        #[input]
        q_pe: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        /// The tokens a query reads back through, itself included; none
        /// reads every row.
        #[serde(default)]
        window: Option<u32>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: Option<ValueId>,
    },
    MlaPrefill {
        #[input]
        q: ValueId,
        #[input(MLA_PLAN)]
        plan: ValueId,
        #[input]
        q_pe: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        /// The tokens a query reads back through, itself included; none
        /// reads every row.
        #[serde(default)]
        window: Option<u32>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: Option<ValueId>,
    },
    MlaDecodeSelected {
        #[input]
        q: ValueId,
        #[input(MLA_PLAN)]
        plan: ValueId,
        #[input]
        q_pe: ValueId,
        #[input(I32)]
        selection: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: Option<ValueId>,
    },
    MlaPrefillSelected {
        #[input]
        q: ValueId,
        #[input(MLA_PLAN)]
        plan: ValueId,
        #[input]
        q_pe: ValueId,
        #[input(I32)]
        selection: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: Option<ValueId>,
    },
    IndexLayernormRope {
        #[input]
        k: ValueId,
        #[input(I32)]
        positions: ValueId,
        #[input]
        weight: ValueId,
        #[input]
        bias: ValueId,
        eps: f32,
        rope_dim: u32,
        theta: f32,
        #[output(alias = k)]
        k_out: ValueId,
    },
    IndexRope {
        #[input]
        q: ValueId,
        #[input(I32)]
        positions: ValueId,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
        #[output(alias = q)]
        q_out: ValueId,
    },
    IndexTopk {
        #[input]
        q: ValueId,
        #[input(CACHE)]
        keys: ValueId,
        #[input]
        weights: Option<ValueId>,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        ratio: u32,
        #[serde(default)]
        score: IndexScore,
        #[output(I32)]
        selection: ValueId,
    },
    IndexKvAppend {
        #[input]
        k: ValueId,
        #[input(CACHE)]
        keys: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
    },
    IndexBlockMean {
        #[input(I32)]
        boundary_pos: ValueId,
        #[input(I32)]
        boundary_req: ValueId,
        #[input(CACHE)]
        keys: ValueId,
        head_dim: u32,
        ratio: u32,
        #[output]
        entries: ValueId,
    },
    PoolBoundaryDecode {
        #[input(I32)]
        positions: ValueId,
        #[input(U8)]
        row_valid: ValueId,
        ratio: u32,
        #[output(I32)]
        boundary_pos: ValueId,
        #[output(I32)]
        boundary_req: ValueId,
        #[output(I32)]
        boundary_rope: ValueId,
    },
    PoolBoundaryPrefill {
        #[input(I32)]
        positions: ValueId,
        #[input(U8)]
        row_valid: ValueId,
        ratio: u32,
        #[output(I32)]
        boundary_pos: ValueId,
        #[output(I32)]
        boundary_req: ValueId,
        #[output(I32)]
        boundary_rope: ValueId,
    },
    PoolStateWrite {
        #[input]
        kv: ValueId,
        #[input]
        score: ValueId,
        #[input(CACHE)]
        pages: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
        head_dim: u32,
        ratio: u32,
    },
    PoolGather {
        #[input(I32)]
        boundary_pos: ValueId,
        #[input(I32)]
        boundary_req: ValueId,
        #[input(CACHE)]
        pages: ValueId,
        #[input(F32)]
        ape: Option<ValueId>,
        head_dim: u32,
        ratio: u32,
        #[output]
        entries: ValueId,
    },
    PoolKvAppend {
        #[input]
        entries: ValueId,
        #[input(I32)]
        boundary_pos: ValueId,
        #[input(I32)]
        boundary_req: ValueId,
        #[input(CACHE)]
        pool: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
    },
    PoolLse {
        #[input]
        q: ValueId,
        #[input(I32)]
        positions: ValueId,
        #[input(I32)]
        request_of_token: ValueId,
        #[input(CACHE)]
        entries: ValueId,
        ratio: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: ValueId,
    },
    PoolLseSelected {
        #[input]
        q: ValueId,
        #[input(I32)]
        positions: ValueId,
        #[input(I32)]
        request_of_token: ValueId,
        #[input(I32)]
        selection: ValueId,
        #[input(CACHE)]
        entries: ValueId,
        ratio: u32,
        top_k: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: ValueId,
    },
    /// Bounded replay's floor, a lane's least position whose window keys a
    /// replaying layer holds: with `mark` the rows are ones it skips and the
    /// lane's state moves past them; without, each row reads its lane's floor
    /// (0 if it never skipped). `floor` is i32 a row.
    ReplayFloor {
        #[input(I32)]
        positions: ValueId,
        #[input(CACHE)]
        state: ValueId,
        mark: bool,
        #[output(I32)]
        floor: ValueId,
    },
    /// Each row's lane's last key position in `cache` (i32 a row): its kv
    /// extent less one, whatever row of the lane it is.
    KvEnds {
        #[input(I32)]
        request_of_token: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[output(I32)]
        ends: ValueId,
    },
    /// DeepSeek's window attention with its sinks, each row reading the
    /// `window` positions that end at `positions` (not necessarily its own:
    /// a drafted block's rows each read the whole block), its output's tail
    /// not turned back.
    WindowSink {
        #[input]
        q: ValueId,
        #[input(I32)]
        positions: ValueId,
        #[input(I32)]
        request_of_token: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        sink: ValueId,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
    },
}

/// How `attn.index_topk` scores a block of keys.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexScore {
    /// `Σ_h relu(q_h·k)·w_h` against the one pooled key that stands for each
    /// closed block; one selection every head shares.
    #[default]
    Pooled,
    /// `max_s q_h·k_s` over the keys of each block the row sees, per head: a
    /// run of `top_k` blocks for each head, ascending, the row's own block
    /// always in it, and every visible block when they fit.
    BlockMax,
    /// `Pooled`'s ranking, its `top_k` closed blocks expanded to their raw
    /// tokens (`ratio` a block), then the row's own open block's: `top_k *
    /// ratio + ratio - 1` token indices a row, unused -1 (GLM-5.3-Flash).
    PooledTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RaggedMask {
    None,
    GroupBlockDiagonal,
    ReferenceSelfOnly { q_tags: ValueId, kv_tags: ValueId },
    RelativeBias { table: ValueId, max_len: u32 },
}

impl Part for RaggedMask {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            RaggedMask::ReferenceSelfOnly { q_tags, kv_tags } => sink.extend([*q_tags, *kv_tags]),
            RaggedMask::RelativeBias { table, .. } => sink.push(*table),
            RaggedMask::None | RaggedMask::GroupBlockDiagonal => {}
        }
    }
    fn outputs(&self, _sink: &mut Vec<ValueId>) {}
    fn ports(&self, sink: &mut Vec<(&'static str, ValueId, Expect)>) {
        match self {
            RaggedMask::ReferenceSelfOnly { q_tags, kv_tags } => {
                sink.push(("q_tags", *q_tags, Expect::I32));
                sink.push(("kv_tags", *kv_tags, Expect::I32));
            }
            RaggedMask::RelativeBias { table, .. } => sink.push(("table", *table, Expect::F32)),
            RaggedMask::None | RaggedMask::GroupBlockDiagonal => {}
        }
    }
}
