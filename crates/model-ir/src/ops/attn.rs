use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// Ops where tokens interact, or where a sequence cache — kv pages, ssm
/// recurrent state, the indexer's key cache, the compressor's pooled entries —
/// is touched. Plans are explicit ops: the `Plan*` variants define `Struct`
/// values from declared geometry inputs and from the reading they are carved
/// for, and every variant that walks a cache takes the plan it was built from
/// — `cache` is the pool pointer, nothing
/// more. The append ops carry their write addressing
/// (`write_page`/`write_offset`) the same way.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attention {
    /// Defines `Struct(AttnDecodePlan)`. Host work; runs in the prepare phase.
    /// A schedule is carved for ONE reading, stated here; every launch that
    /// reads it restates its share and the shell refuses a disagreement.
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
    /// Defines `Struct(AttnPrefillPlan)`. Carries the same reading
    /// [`Attention::PlanDecode`] does, on the same terms.
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
    /// Prefill against a query-provided mask instead of the causal one;
    /// the op names the `mask` it applies, not the engine.
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
    /// **BIDIRECTIONAL ATTENTION OVER THE PATCH WINDOW, BLOCK-DIAGONAL PER
    /// IMAGE** — the vision towers' one real kernel (multimodal §2).
    ///
    /// It is an `Attention` by rule 1 and by nothing else it shares with the
    /// paged family: no cache, no plan struct, no page table, no append, no
    /// mask ladder, no log-sum-exp. `segments` is the PATCH axis's own indptr
    /// (`RuntimeInput::PatchSegments`, `i32`, `[Dim::ImagesPlus(1)]`): patch
    /// row `n` attends over the rows of the image whose span contains it, in
    /// both directions, and over nothing else — which is the same batching
    /// grammar the token axis runs, recursed one axis down with IMAGES where
    /// the lanes were.
    ///
    /// `q`, `k`, `v` and `o` are patch rectangles, so this op's row axis is
    /// [`RowAxis::Patches`](crate::RowAxis::Patches) and the capture-unit
    /// partition reads that off the shapes rather than off the op's name.
    Dense {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        segments: ValueId,
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
    /// Defines `Struct(MlaPlan)`, shared by decode and prefill. `heads` and
    /// `kv_lora_rank` are the absorbed reading: the latent kernels size their
    /// output at `heads × kv_lora_rank`, which is what `MlaDecode`/`MlaPrefill`
    /// restate.
    MlaPlan {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        kv_len: ValueId,
        heads: u32,
        kv_lora_rank: u32,
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
    //
    // `dilation` spreads the taps: tap `j` reads `dilation · j` positions
    // back, so the state keeps `(conv_width − 1) · dilation` columns. Every
    // GDN mixer states 1; qwen4's PLE convolution states its `ngram_size`.
    SsmCausalConv1d {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        dilation: u32,
        y: ValueId,
    },
    /// Prefill form: walks the fire's ambient request boundaries.
    SsmCausalConv1dChunked {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        dilation: u32,
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
    /// Scores `q` against the cached keys; `selection` is the top-k key ids.
    ///
    /// **`ratio` IS WHICH CACHED ROWS ARE KEYS, AND IT IS THE DIFFERENCE
    /// BETWEEN TWO FAMILIES.** glm_5's indexer keys one row per TOKEN — a
    /// `k_proj` layernormed and roped into the index cache at the token's own
    /// cell — and states `ratio = 1`: the visible keys are `pos + 1`, key `j`
    /// lives at position `j`, and the ids this op publishes are token
    /// positions. dsv4-flash's indexer keys one row per COMPRESSED BLOCK: its
    /// keys are its own compressor's pooled entries (`v4mlx/compressor.py`'s
    /// `compressor_prefill(..., rotate=True)` returns `[b, s // ratio, d]`),
    /// stored at the boundary cell `(c + 1) * ratio - 1` exactly as
    /// `PoolKvAppend` stores the attention compressor's, so it states its
    /// compressor's own `ratio`: the visible keys are `(pos + 1) / ratio`, key
    /// `c` lives at position `(c + 1) * ratio - 1`, and the ids this op
    /// publishes are COMPRESSED ROW indices — which is exactly what
    /// [`Self::PoolLseSelected`] walks.
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

    // Pooled (compressed) attention: every `ratio` tokens close a boundary
    // whose pooled entry lands in its own cache. Boundary outputs are
    // token-shaped — over-allocated with a sentinel in the non-boundary rows —
    // so no counted dim exists and the shapes stay trace-time facts.
    /// `row_valid` masks graph-padding rows out of the boundary math.
    ///
    /// **`boundary_rope` IS NOT `boundary_pos`.** The pooled entry is CACHED
    /// at the cell its window closes on (`boundary_pos`, the block's LAST
    /// token — `(c+1)·ratio - 1`, which is the cell `PoolLse` reads back),
    /// but it is ROPED at the compressed row's own position
    /// (`(p / ratio) · ratio` — the block's FIRST token, the reference's
    /// `rows = arange(0, cutoff, ratio)` in `v4mlx/compressor.py`). Two
    /// different positions for one entry, which is why this op publishes two
    /// columns and not one; a text that roped at `boundary_pos` — or at the
    /// raw token positions, which agree with it on every boundary row — is
    /// off by `ratio - 1` on every compressed key it caches. Non-boundary
    /// rows carry `0` here (they carry `-1` in `boundary_pos`, the sentinel
    /// the append and the gather test), because a rope operand is an angle
    /// and not a cell and the row is dropped before it is stored.
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
    /// **THE COMPRESSOR'S ROLLING STATE, WRITTEN.**
    ///
    /// `kv` is the learned compressor's `wkv · x` and `score` its `wgate · x`,
    /// both `[tokens, coff · head_dim]`. They do not land in a fire-shaped
    /// rectangle: [`Self::PoolGather`] addresses the state at the SOURCE
    /// cache's own paged slot, because a pooling window closing in this fire
    /// straddles tokens written in earlier ones. So this op scatters each row
    /// into the cell `write_page`/`write_offset` name for it — the same cell
    /// [`Self::KvAppendShared`] writes the latent into, in the same fire —
    /// and the slabs it writes are the shell's, not a value the trace names.
    ///
    /// Without it the gather pools zeros, which is what it did for as long as
    /// the compressor's four planes were interned.
    PoolStateWrite {
        kv: ValueId,
        score: ValueId,
        pages: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
        head_dim: u32,
        ratio: u32,
    },
    /// Pools the closing window out of the kv cache into per-boundary entries.
    ///
    /// `ape` is the compressor's intra-block ABSOLUTE POSITION plane
    /// (`[ratio, coff · head_dim]`, f32), folded into the gate logits before
    /// the softmax. `None` for a parameter-free mean pool — the toy rows —
    /// which is the `ape == nullptr` path both shaders already carry.
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
    /// [`Self::PoolLse`] over a SELECTION instead of over the whole visible
    /// prefix: the same reader, the same online softmax and the same base-2
    /// log-sum-exp the cascade merge folds, walking `selection[t · top_k + n]`
    /// — the compressed-row ids [`Self::IndexTopk`] published, ascending, with
    /// `-1` padding a row that saw fewer keys than its budget.
    ///
    /// **THIS IS THE NSA FINE BRANCH, AND THE BRANCH IT NARROWS IS THE
    /// COMPRESSED ONE.** The reference oracle's attention is one softmax over
    /// `concat(sliding window over the per-token latent, EVERY visible
    /// compressed row)` with the per-head sink in the denominator — which is
    /// what `prefill_lse` at `window` merged with `pool_lse` and closed by
    /// `sink` already computes. The window is a fixed 128; the only key set
    /// that grows with the context is the compressed one (`nvis = (pos + 1) /
    /// ratio`), so the indexer's `index_topk` budget is what caps THAT, and
    /// the ratio-128 layers carry no indexer because `S / 128` needs no
    /// capping. A row whose visible count is inside the budget selects the
    /// identity, so this op reduces to `PoolLse` exactly.
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

    // The PLE n-gram hasher (qwen4). An `Attention` by rule 1's second
    // clause alone: it touches a sequence cache — the last `mults.len() − 1`
    // token ids of each lane, kept so a decode step can hash the window its
    // fire cannot see. Everything else about it is per-token integer math.
    /// `ngram_ids[r, g·heads_per_ngram + h]` = the hashed (g+2)-gram id of
    /// token `r` under head `h`'s own prime: ids at `r, r−1, …` (eos where
    /// the window crosses a sequence start) are multiplied by `mults`,
    /// xor-folded, and reduced modulo `primes[·]` plus `offsets[·]` — the
    /// reference's splitmix64-seeded hash, with the derived constants stated
    /// on the node because no checkpoint plane needs to be read to know them.
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
    /// Prefill form: walks the fire's ambient request boundaries, as the
    /// chunked convolution does.
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
            Self::Dense { q, k, v, segments, .. } => sink.extend([*q, *k, *v, *segments]),
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
            Self::PoolStateWrite { kv, score, pages, write_page, write_offset, .. } => {
                sink.extend([*kv, *score, *pages, *write_page, *write_offset]);
            }
            Self::PoolGather { boundary_pos, boundary_req, pages, ape, .. } => {
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
            Self::PoolLse { q, positions, request_of_token, entries, .. } => {
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
