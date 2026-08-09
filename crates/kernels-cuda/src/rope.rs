//! Rotary position encoding, and the norm+rope fusions that feed attention.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::KernelSig;
use kernels::{kernel, operands};
use kernels::Lit;
use kernels::Source;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(rope_standard_table "rope::rope_standard_table",
        operands = operands![
            positions: I32s <- Source::Positions,
            table: F32sMut <- Source::Out(0),
            num_tokens: I32 <- Source::Rows,
            head_dim: I32 <- Source::Ctx("head_dim"),
            theta: F32 <- Source::CtxNonZero("rope_theta"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // Not in the table until the ABI pilot; the vocabulary audit has been
    // counting it as undeclared. `interleaved` is where GLM and the MLA rope
    // dims differ from Llama/Qwen -- a load-time checkpoint fact reaching the
    // kernel as an argument rather than as a second symbol, which is the one
    // place this family does it that way.
    kernel!(rope "rope::rope_bf16",
        // Rotates q and k WHERE THEY LIE -- `BufMut` on both, and no
        // destination to give them another. Unstated while the only
        // caller was the SEMANTIC `OpKind::Rope`, whose alias
        // `kernels::semantic_in_place` carried; `cuda::rope` states this
        // symbol now, and a host that assigns addresses reads the pair
        // list off the row.
        in_place = &[(0, 0), (1, 1)],
        // WHERE EACH ARGUMENT COMES FROM, so the arm is generated
        // rather than written. `interleaved` is a literal because no
        // statement and no context carries it: the families that pass
        // `true` (GLM, MLA) are not declared, and a row that pretended
        // otherwise would be guessing on their behalf.
        operands = operands![
            q: BufMut <- Source::Out(0),
            k: BufMut <- Source::Out(1),
            positions: I32s <- Source::Positions,
            num_tokens: I32 <- Source::Rows,
            num_q_heads: I32 <- Source::Ctx("num_q_heads"),
            num_kv_heads: I32 <- Source::Ctx("num_kv_heads"),
            head_dim: I32 <- Source::Ctx("head_dim"),
            theta: F32 <- Source::CtxNonZero("rope_theta"),
            stream: Stream <- Source::Ctx("stream"),
            interleaved: Bool <- Source::Lit(Lit::Bool(false)),
        ]),
    // Norms AND rotates q and k where they lie -- `BufMut` on both, and
    // no destination to give them another. The `_rounded` twin below has
    // said so since gemma-4's conversion; this row had not, and
    // llama_like states it 84 times per decode text.
    // THE DEVWIN CAUTION THIS ROW CARRIED WAS ABOUT A HAZARD THAT DOES
    // NOT EXIST. It read: llama_like fires a different launcher for this
    // same stated symbol on a peel's tail, so a generated branch would
    // take the plain form on a fire that wanted the windowed one. It
    // does not — `dsl::cuda::qk_rmsnorm_rope_devwin` records
    // `rope::qk_rmsnorm_rope_bf16_devwin`, a SECOND symbol, so the two
    // forms never arrive under one key and there is nothing for a guard
    // to disambiguate. The hand arm this replaces fired the plain
    // launcher unconditionally, which is the same behaviour and is what
    // said the caution was stale.
    kernel!(qk_rmsnorm_rope "rope::qk_rmsnorm_rope_bf16",
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut <- Source::Out(0),
            k: BufMut <- Source::Out(1),
            q_weight: Buf <- Source::Weight(0),
            k_weight: Buf <- Source::Weight(1),
            positions: I32s <- Source::Positions,
            num_tokens: I32 <- Source::Rows,
            num_q_heads: I32 <- Source::Div(&Source::Width(&Source::Out(0)), &Source::CtxNonZero("head_dim")),
            num_kv_heads: I32 <- Source::Div(&Source::Width(&Source::Out(1)), &Source::CtxNonZero("head_dim")),
            head_dim: I32 <- Source::Ctx("head_dim"),
            theta: F32 <- Source::CtxByLayer("theta"),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // A hooked pure-decode fire is graph-CAPTURED and its hook split rides a
    // DEVICE word (`win_d`), not a host row range. All four are `whole`, and
    // for a reason no other `whole` row here gives: the window is not a
    // number the lowering knows, so it cannot be a rectangle at all.
    kernel!(qk_rmsnorm_rope_devwin "rope::qk_rmsnorm_rope_bf16_devwin", whole = true,
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut, k: BufMut,
            q_weight: Buf, k_weight: Buf,
            positions: I32s,
            win_d: U32s,
            n_max: I32, num_q_heads: I32, num_kv_heads: I32, head_dim: I32,
            theta: F32, eps: F32,
            stream: Stream,
        ]),
    // YaRN and original-YaRN interpolate frequencies differently; which a
    // checkpoint wants is a load-time fact, so they are two rows.
    kernel!(rope_yarn "rope::rope_yarn_bf16",
        operands = operands![
            q: BufMut, k: BufMut,
            positions: I32s,
            num_tokens: I32, num_q_heads: I32, num_kv_heads: I32, head_dim: I32,
            theta: F32,
            factor: F32, low_freq_factor: F32, high_freq_factor: F32,
            original_max_position: I32,
            stream: Stream,
        ]),
    // MROPE takes `[num_tokens, 3]` positions -- a (t, h, w) triple, because
    // a vision model's tokens sit in a grid. Not the plain qk_rmsnorm_rope
    // with a different theta.
    kernel!(qk_rmsnorm_mrope "rope::qk_rmsnorm_mrope_bf16",
        operands = operands![
            q: BufMut, k: BufMut,
            q_weight: Buf, k_weight: Buf,
            positions: I32s,
            num_tokens: I32, num_q_heads: I32, num_kv_heads: I32, head_dim: I32,
            theta: F32, eps: F32,
            mrope_section_t: I32, mrope_section_h: I32, mrope_section_w: I32,
            stream: Stream,
        ]),
    // Ropes the LAST `rope_dim` channels rather than the first. A different
    // statement from `rope_partial_q_only`, not a flag on it: which end of
    // the channel axis carries position is a property of the checkpoint.
    kernel!(rope_partial_last "rope::rope_partial_last_bf16",
        operands = operands![
            q: BufMut, k: BufMut,
            positions: I32s,
            num_tokens: I32, num_q_heads: I32, num_kv_heads: I32,
            head_dim: I32, rotary_dim: I32,
            theta: F32,
            stream: Stream,
            inverse: Bool, interleaved: Bool,
            yarn_factor: F32, yarn_beta_fast: F32, yarn_beta_slow: F32,
            yarn_original_max_position: I32,
        ]),
    // Q-only rotation: a KV-shared layer's K was rotated at its source
    // layer. One operand is the statement.
    //
    // Rotates q and k WHERE THEY LIE — two aliases, which is what the
    // pair list exists for. A q-only site states one operand and the
    // second pair falls outside its arity, which `Buffers` skips.
    kernel!(rope_partial_q_only "rope::rope_partial_bf16",
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut <- Source::Out(0),
            // A Q-ONLY SITE STATES ONE RESULT and the launcher takes q
            // for k with `num_kv_heads = 0`. That used to read as arity
            // the row could not name; it is an `Or`, and what decides is
            // whether the second result is there.
            k: BufMut <- Source::Or(&Source::Out(1), &Source::Out(0)),
            positions: I32s <- Source::Positions,
            num_tokens: I32 <- Source::Rows,
            // The head COUNTS off the cache's head dim rather than the
            // ctx's, because a KV-shared layer's q and k disagree.
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::Out(0)),
                &Source::KvLayerField("head_dim"),
            ),
            // ZERO when there is no second result, which is the q-only
            // form's whole signal to the launcher.
            num_kv_heads: I32 <- Source::Or(
                &Source::Div(
                    &Source::Width(&Source::Out(1)),
                    &Source::KvLayerField("head_dim"),
                ),
                &Source::Lit(Lit::I32(0)),
            ),
            head_dim: I32 <- Source::KvLayerField("head_dim"),
            rotary_dim: I32 <- Source::RotaryWidth,
            theta: F32 <- Source::CtxByLayer("theta"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The other row the audit was counting as undeclared. It is `rope_partial`
    // with `positions` shifted by a host constant, and the delta sits between
    // `positions` and the extents rather than at the end -- which is exactly
    // the kind of fact a hand-written binding gets wrong and a generated one
    // cannot.
    kernel!(rope_partial_position_delta "rope::rope_partial_bf16_position_delta",
        operands = operands![
            q: BufMut, k: BufMut,
            positions: I32s,
            position_delta: I32,
            num_tokens: I32, num_q_heads: I32, num_kv_heads: I32,
            head_dim: I32, rotary_dim: I32,
            theta: F32,
            stream: Stream,
        ]),
    // gemma-4 rounds where qwen3_5 does not, and bf16 rounding is which
    // numbers come out — so the symbol IS the statement.
    kernel!(qk_rmsnorm_rope_rounded "rope::qk_rmsnorm_rope_bf16_rounded",
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut <- Source::Out(0),
            // A Q-ONLY SITE STATES ONE RESULT and no k weight, and the
            // launcher reads the nulls as "there is no k". Same `Or` the
            // partial rotation makes, and the head count comes off k's
            // OWN width so that one expression answers both forms.
            k: BufMut <- Source::Or(&Source::Out(1), &Source::Lit(Lit::Null)),
            q_weight: Buf <- Source::Weight(0),
            k_weight: Buf <- Source::Or(&Source::Weight(1), &Source::Lit(Lit::Null)),
            positions: I32s <- Source::Positions,
            num_tokens: I32 <- Source::Rows,
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::Out(0)),
                &Source::KvLayerField("head_dim"),
            ),
            num_kv_heads: I32 <- Source::Or(
                &Source::Div(
                    &Source::Width(&Source::Out(1)),
                    &Source::KvLayerField("head_dim"),
                ),
                &Source::Lit(Lit::I32(0)),
            ),
            head_dim: I32 <- Source::KvLayerField("head_dim"),
            theta: F32 <- Source::CtxByLayer("theta"),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // YaRN, as its paper spells it. A deployment's scaling is a load-time
    // config answer, so it picks a kernel here rather than an argument.
    kernel!(rope_yarn_original "rope::rope_yarn_original_bf16",
        in_place = &[(0, 0), (1, 1)],
        operands = operands![
            q: BufMut <- Source::Out(0),
            k: BufMut <- Source::Out(1),
            positions: I32s <- Source::Ctx("positions"),
            num_tokens: I32 <- Source::Rows,
            // Heads, not width: the kernel rotates per head, and how many
            // fit in a row is the row width over the head dim.
            num_q_heads: I32 <- Source::Div(&Source::Width(&Source::Out(0)), &Source::CtxNonZero("head_dim")),
            num_kv_heads: I32 <- Source::Div(&Source::Width(&Source::Out(1)), &Source::CtxNonZero("head_dim")),
            head_dim: I32 <- Source::Ctx("head_dim"),
            theta: F32 <- Source::Ctx("rope_theta"),
            // YaRN's four scalars, in the order the config states them.
            // `Ctx` names a FIELD PATH, so an index is as nameable as a
            // name -- which is what keeps four near-identical sources
            // from having to exist.
            factor: F32 <- Source::Ctx("yarn[0]"),
            beta_fast: F32 <- Source::Ctx("yarn[1]"),
            beta_slow: F32 <- Source::Ctx("yarn[2]"),
            attention_factor: F32 <- Source::Ctx("yarn[3]"),
            original_max_position: I32 <- Source::Ctx("yarn_original_max"),
            stream: Stream <- Source::Ctx("stream"),
            interleaved: Bool <- Source::Ctx("rope_interleaved"),
        ]),
    kernel!(rope_write_kv "rope::rope_write_kv_bf16", whole = true, sink = Some("kv.pages"),
        operands = operands![
            q: BufMut, k: BufMut, v: Buf,
            positions: I32s,
            k_pages: BufMut, v_pages: BufMut,
            qo_indptr: U32s,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            row_valid: U8s | null,
            num_tokens: I32, num_requests: I32, page_size: I32,
            num_q_heads: I32, num_kv_heads: I32, head_dim: I32,
            theta: F32,
            hnd_layout: Bool,
            stream: Stream,
            interleaved: Bool,
        ]),
];
