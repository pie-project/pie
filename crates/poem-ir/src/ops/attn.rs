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
    /// Each row's query against its lane's paged keys, as one token's. A
    /// selection (`selection`, its run of blocks as `selected` lays them out)
    /// reads only those blocks; a relative bias (`bias`, as `rel` scales it)
    /// adds each key's learned offset; `lse` writes the rows' log-sum-exp.
    /// `mask` is the fire's [`crate::RuntimeInput::Mask`]: per-lane data,
    /// each lane's mask or its word stating none, read as
    /// [`Attn::Prefill`]'s is.
    Decode {
        #[input]
        q: ValueId,
        #[input(DECODE_PLAN)]
        plan: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[input]
        mask: Option<ValueId>,
        #[input(I32)]
        selection: Option<ValueId>,
        #[input(F32)]
        bias: Option<ValueId>,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        selected: Option<Blocks>,
        rel: Option<Relative>,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: Option<ValueId>,
    },
    /// Each row's query against its lane's paged keys, causal where `causal`, masked by `mask`
    /// (the fire's [`crate::RuntimeInput::Mask`], per-lane data: a lane stating no mask reads
    /// every key its bounds give, in the same body as a lane stating one). A
    /// selection, a relative bias and `lse` as [`Attn::Decode`]'s; `floor`,
    /// each row's least window position (i32 a row, from
    /// `attn.replay_floor`), below which the window reads no key.
    Prefill {
        #[input]
        q: ValueId,
        #[input(PREFILL_PLAN)]
        plan: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[input]
        mask: Option<ValueId>,
        #[input(I32)]
        selection: Option<ValueId>,
        #[input(F32)]
        bias: Option<ValueId>,
        #[input(I32)]
        floor: Option<ValueId>,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        causal: bool,
        sm_scale: f32,
        selected: Option<Blocks>,
        rel: Option<Relative>,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: Option<ValueId>,
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
    /// Each capturing lane's attention mass a key of the load's context
    /// ([`crate::Dim::ContextTimes`]`(1)`), averaged over its last `observe`
    /// query rows and over every query head, added to `into` (an earlier
    /// layer's, the same row) or, with none, to a zeroed row: one row a
    /// capturing lane ([`crate::Dim::Capturing`]). The last layer's is the
    /// mass summed over the layers, a key.
    ScoreCapture {
        #[input]
        q: ValueId,
        #[input(PREFILL_PLAN)]
        plan: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        head_dim: u32,
        kv_heads: u32,
        /// The model's query heads, every rank's: what the mass a key is
        /// averaged over, where a rank attends only its share of them.
        q_heads: u32,
        sm_scale: f32,
        observe: u32,
        #[input(F32)]
        into: Option<ValueId>,
        #[output(F32)]
        scores: ValueId,
    },
    /// Each capturing lane's key row `keys` summed a kv page of the load's:
    /// one row a capturing lane, a mass a page ([`crate::Dim::ContextPages`]).
    ScorePages {
        #[input(F32)]
        keys: ValueId,
        #[output(F32)]
        pages: ValueId,
    },
    Sink {
        #[input]
        o: ValueId,
        #[input(F32)]
        lse: ValueId,
        #[input]
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
    /// Each row's key and value written at its page and offset; without
    /// `v`, `k` is one plane a pool reads as both (a shared-kv latent).
    KvAppend {
        #[input]
        k: ValueId,
        #[input]
        v: Option<ValueId>,
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
        /// The latent rows a query reads, where a selection picks them;
        /// none reads its whole window.
        #[input(I32)]
        selection: Option<ValueId>,
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
        /// The latent rows a query reads, where a selection picks them;
        /// none reads its whole window.
        #[input(I32)]
        selection: Option<ValueId>,
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
    /// Where the rows close entries of a pool compressing `ratio` positions
    /// each, for one token's rows (`decode`) or a prompt's.
    PoolBoundary {
        #[input(I32)]
        positions: ValueId,
        #[input(U8)]
        row_valid: ValueId,
        ratio: u32,
        decode: bool,
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
        /// The `top_k` pool entries each row reads, where a selection picks
        /// them; none reads every closed entry.
        #[input(I32)]
        selection: Option<ValueId>,
        #[input(CACHE)]
        entries: ValueId,
        ratio: u32,
        top_k: Option<u32>,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
        #[output(F32)]
        lse: ValueId,
    },
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
        #[input]
        sink: ValueId,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        #[output]
        o: ValueId,
    },
}

/// The blocks a selection reads, `ratio` keys each: one run every head
/// shares, or (`per_head`) one run per kv head, each head's queries reading
/// its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Blocks {
    pub ratio: u32,
    pub per_head: bool,
}

/// A learned relative bias over `extent` positions: a key's distance past
/// `2^log_floor` falls into log buckets `log_alpha` wide.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Relative {
    pub extent: u32,
    pub log_floor: u32,
    pub log_alpha: f32,
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
    /// Each row attends rows of its own group, narrowed by the mask its lane
    /// states (`forward.mask` on a kv-less lane), a lane stating none
    /// attending its whole group. Only an attention the model states so reads
    /// a lane's mask: a lane's mask over any other is refused.
    Stated,
    RelativeBias {
        table: ValueId,
        max_len: u32,
    },
}

impl Part for RaggedMask {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            RaggedMask::RelativeBias { table, .. } => sink.push(*table),
            RaggedMask::None | RaggedMask::GroupBlockDiagonal | RaggedMask::Stated => {}
        }
    }
    fn outputs(&self, _sink: &mut Vec<ValueId>) {}
    fn ports(&self, sink: &mut Vec<(&'static str, ValueId, Expect)>) {
        match self {
            RaggedMask::RelativeBias { table, .. } => sink.push(("table", *table, Expect::F32)),
            RaggedMask::None | RaggedMask::GroupBlockDiagonal | RaggedMask::Stated => {}
        }
    }
}
