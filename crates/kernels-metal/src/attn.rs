//! Attention, and the KV writes that feed it.
//!
//! The head dim is an axis everywhere here and its points are the checkpoint
//! geometries the tree is actually compiled for -- 64 (llama-3.2, gpt-oss),
//! 128 (llama, qwen), 256 (qwen3.5), 512 (gemma4 full-attn). A checkpoint
//! whose width is not a point of the axis has no pipeline, which used to be a
//! runtime PSO failure naming a string and is now a fact the table states.

#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, I32s, Routine, U8s, U32s, Usize};

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[
    ("attn/gate.metal", "gate_bfloat16"),
    ("attn/kv_write.metal", "kv_append_bfloat16"),
    ("attn/kv_write.metal", "kv_append_paged_bfloat16"),
    ("attn/logit_softcap.metal", "logit_softcap_bfloat16"),
    ("attn/gate.metal", "q_gate_split_bfloat16"),
    ("attn/sdpa_paged.metal", "sdpa_paged_decode_bfloat16_d_128"),
    (
        "attn/sdpa_paged.metal",
        "sdpa_paged_decode_bfloat16_d_128_p32",
    ),
    ("attn/sdpa_paged.metal", "sdpa_paged_decode_bfloat16_d_256"),
    ("attn/sdpa_paged.metal", "sdpa_paged_decode_bfloat16_d_512"),
    ("attn/sdpa_paged.metal", "sdpa_paged_decode_bfloat16_d_64"),
    (
        "attn/sdpa_paged.metal",
        "sdpa_paged_decode_bfloat16_d_64_p32",
    ),
    (
        "attn/sdpa_paged.metal",
        "sdpa_paged_decode_bfloat16_d_64_p32_sg8",
    ),
    (
        "attn/sdpa_paged.metal",
        "sdpa_paged_decode_sink_bfloat16_d_64",
    ),
    ("attn/sdpa_paged_mma.metal", "sdpa_paged_mma_bfloat16_d_64"),
    (
        "attn/sdpa_paged_mma.metal",
        "sdpa_paged_mma_sink_bfloat16_d_64",
    ),
    ("attn/sdpa_paged.metal", "sdpa_paged_tiled_bfloat16_d_128"),
    ("attn/sdpa_paged.metal", "sdpa_paged_tiled_bfloat16_d_256"),
    ("attn/sdpa_paged.metal", "sdpa_paged_tiled_bfloat16_d_512"),
    ("attn/sdpa_paged.metal", "sdpa_paged_tiled_bfloat16_d_64"),
    (
        "attn/sdpa_paged.metal",
        "sdpa_paged_tiled_sink_bfloat16_d_64",
    ),
    (
        "attn/sdpa_paged.metal",
        "sdpa_paged_tiled_strided_bfloat16_d_256",
    ),
    (
        "attn/sdpa_vector.metal",
        "sdpa_vector_decode_bfloat16_d_128",
    ),
    (
        "attn/sdpa_vector.metal",
        "sdpa_vector_decode_bfloat16_d_256",
    ),
    ("attn/sdpa_vector.metal", "sdpa_vector_decode_bfloat16_d_64"),
    (
        "attn/sdpa_sliding.metal",
        "sdpa_vector_decode_sink_bfloat16_d_64",
    ),
    (
        "attn/sdpa_sliding.metal",
        "sdpa_vector_decode_swa_bfloat16_d_256",
    ),
    (
        "attn/sdpa_sliding.metal",
        "sdpa_vector_decode_swa_bfloat16_d_512",
    ),
    ("attn/split_qkv.metal", "split_qkv_bf16"),
];

/// The head widths `sdpa_paged.metal` is compiled for.
pub const PAGED_DIMS: [i32; 4] = [64, 128, 256, 512];

/// The head widths `sdpa_vector.metal` is compiled for. 512 is not one: a
/// whole-history vector decode at that width would hold a 512-float
/// accumulator per lane, and the paged form is what serves it.
pub const VECTOR_DIMS: [i32; 3] = [64, 128, 256];

/// The two widths the sliding-window vector decode carries.
pub const SWA_DIMS: [i32; 2] = [256, 512];

/// The four paged single-pass decode instantiations, in [`PAGED_DIMS`] order.
///
/// The three PAGE-SHAPE tails are deliberately absent, and the row above says
/// why at length: `_p32` and `_p32_sg8` set `FAST_FULL`, which deletes the
/// window and all three mask operands from the body, and `_p32_sg8` is `BN =
/// 8` -- a 256-thread threadgroup where every point here is 1024. Naming one
/// from this signature would bind four operands the shader does not read and,
/// for the last, launch it at four times the simdgroups its threadgroup
/// arrays are sized for. A caller that wants one has to bring its own launch
/// and its own shorter argument list; until it does, their absence here is
/// the row's refusal made executable.
pub const PAGED_DECODE: [&str; 4] = [
    "sdpa_paged_decode_bfloat16_d_64",
    "sdpa_paged_decode_bfloat16_d_128",
    "sdpa_paged_decode_bfloat16_d_256",
    "sdpa_paged_decode_bfloat16_d_512",
];

/// The four paged tiled prefill instantiations, in [`PAGED_DIMS`] order.
pub const PAGED_TILED: [&str; 4] = [
    "sdpa_paged_tiled_bfloat16_d_64",
    "sdpa_paged_tiled_bfloat16_d_128",
    "sdpa_paged_tiled_bfloat16_d_256",
    "sdpa_paged_tiled_bfloat16_d_512",
];

/// The three whole-history vector decodes, in [`VECTOR_DIMS`] order.
pub const VECTOR_DECODE: [&str; 3] = [
    "sdpa_vector_decode_bfloat16_d_64",
    "sdpa_vector_decode_bfloat16_d_128",
    "sdpa_vector_decode_bfloat16_d_256",
];

/// The two sliding-window vector decodes, in [`SWA_DIMS`] order.
pub const VECTOR_SWA: [&str; 2] = [
    "sdpa_vector_decode_swa_bfloat16_d_256",
    "sdpa_vector_decode_swa_bfloat16_d_512",
];

/// Which instantiation a head width names.
///
/// A width off the axis is a [`Refusal::Narrow`] here rather than a nil
/// `newFunctionWithName:` inside a fire. That distinction is the whole reason
/// the tables are written out: the failure it replaces happened after the plan
/// was accepted and the pipelines batch-compiled, with a string in the message
/// and nothing to act on.
fn head_point(head_dim: i32, points: &[i32]) -> Result<usize, Refusal> {
    points
        .iter()
        .position(|&p| p == head_dim)
        .ok_or(Refusal::Narrow {
            what: "a head width no shader is compiled for",
            at: i64::from(head_dim),
        })
}

/// The rows of a tile, which is also the simdgroup count of the scalar form.
const TILE: u32 = 32;

/// Every single-pass and tiled attention threadgroup: 32 simdgroups of 32.
///
/// Public because it is a claim about the compiled SHADER and not only about
/// this dispatch: `device_kernels.rs` asks each pipeline whether it admits
/// this many threads, and a mirrored literal there could drift from the
/// number actually launched.
pub const BIG_GROUP: [u32; 3] = [1024, 1, 1];

/// The matrix-unit tiling's threadgroup, and it is not a knob: a simdgroup
/// owns EIGHT query rows there, so the same 32-row tile is four simdgroups.
/// `max_total_threads_per_threadgroup(128)` is declared on the shader, and a
/// 1024-wide dispatch of it is refused by the device rather than run slowly.
///
/// Public for the same reason as [`BIG_GROUP`].
pub const MMA_GROUP: [u32; 3] = [128, 1, 1];

/// Single-pass attention: one threadgroup per `(query head, row)`.
fn vector_grid(q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let heads = positive(q_heads, "query heads")?;
    let rows = positive(rows, "rows")?;
    let x = heads.checked_mul(BIG_GROUP[0]).ok_or(Refusal::Grid {
        what: "query heads * the threadgroup width",
        at: i64::from(heads) * i64::from(BIG_GROUP[0]),
    })?;
    Ok([x, rows, 1])
}

/// Tiled attention: one threadgroup per `(query head, TILE of query rows)`.
///
/// Rounded UP on the row axis and the kernel guards its own tail. A truncating
/// count drops the last partial tile, and those are whole TOKENS -- a prefill
/// of 33 rows would attend 32 and leave the thirty-third holding whatever the
/// arena had.
fn tiled_grid(q_heads: i32, rows: i32, group: u32) -> Result<[u32; 3], Refusal> {
    let heads = positive(q_heads, "query heads")?;
    let rows = positive(rows, "rows")?;
    let x = heads.checked_mul(group).ok_or(Refusal::Grid {
        what: "query heads * the threadgroup width",
        at: i64::from(heads) * i64::from(group),
    })?;
    Ok([x, rows.div_ceil(TILE), 1])
}

/// One threadgroup per `(head, token)`, the head's own channels wide.
fn head_grid(head_dim: i32, heads: i32, depth: i32) -> Result<[u32; 3], Refusal> {
    Ok([
        positive(head_dim, "the head width")?,
        positive(heads, "heads")?,
        positive(depth, "tokens")?,
    ])
}

/// The threadgroup a [`head_grid`] dispatch wants: the head's channels.
const fn head_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 1, 1]
}

/// An extent that has to be there, as a `u32`.
fn positive(v: i32, what: &'static str) -> Result<u32, Refusal> {
    if v <= 0 {
        return Err(Refusal::Empty { what });
    }
    Ok(v.unsigned_abs())
}

/// The elementwise threadgroup this file shares with `mlp` and `norm`.
const GROUP_X: u32 = 256;

const PAGED_FILE: &str = "attn/sdpa_paged.metal";

/// The matrix-unit tiling is its OWN translation unit, not a specialization
/// of the file above. Naming `PAGED_FILE` for it would ask the library for a
/// function the module does not carry, and `newFunctionWithName:` answers
/// that with nil at pipeline time.
const MMA_FILE: &str = "attn/sdpa_paged_mma.metal";
const VECTOR_FILE: &str = "attn/sdpa_vector.metal";

const SPLIT_FILE: &str = "attn/split_qkv.metal";

/// `attn/logit_softcap.metal`, and NOT `attn/softcap.metal`, which is what
/// this said until a test read the directory. Metal answers a missing module
/// with nil rather than an error, so the wrong spelling would have survived
/// every check that does not touch a device.
const SOFTCAP_FILE: &str = "attn/logit_softcap.metal";
const GATE_FILE: &str = "attn/gate.metal";

/// The packed QKV projection taken apart into three tensors.
///
/// Three results, which is what makes the argument order load-bearing: a
/// statement writing three states them all after its inputs, and the shader
/// declares `packed, q, k, v` before its params block.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty width or row count.
pub fn split_qkv_bf16(
    ctx: &Ctx<'_>,
    packed: Buf,
    q: BufMut,
    k: BufMut,
    v: BufMut,
    params: Buf,
    packed_width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "split_qkv_bf16",
            file: SPLIT_FILE,
            lanes: crate::routine::elementwise_rows(*packed_width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[packed.v(), q.v(), k.v(), v.v(), params.v()],
    )
}

/// `attn *= sigmoid(gate)`, in place over the whole attention output.
///
/// The ROW AXIS is stated here and `LaunchRule::PerHeadElementwise` does not
/// state it. `gate.metal`'s own header says the launch is
/// `grid=(n_q*head_dim, rows, 1)` and the rule gives `(n_q*head_dim, 1, 1)`,
/// so a prefill through this symbol gates its first token and leaves every
/// other one ungated -- finite, plausible, and wrong. It is the third instance
/// of one defect in this port: `gated_rms` had the same missing z and
/// `LaunchRule::RouterLane`'s doc records the same missing y, already fixed.
///
/// `row_stride` is 0 for the packed single row a decode hands over, and the
/// body reads the grid's own width in that case. Both tensors live in one
/// arena at one pitch, so one number covers the read and the in-place write.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty width or row count.
pub fn gate(
    ctx: &Ctx<'_>,
    attn: BufMut,
    gate: Buf,
    row_stride: i32,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gate_bfloat16",
            file: GATE_FILE,
            lanes: crate::routine::elementwise_rows(*width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[attn.v(), gate.v(), row_stride.v()],
    )
}

/// The interleaved `[rows, n_q, 2, head_dim]` projection split into a query
/// and its gate.
///
/// Two pitches and not one, unlike [`gate`]: the source is twice as wide per
/// head as either result, so `qg_row_stride` and `out_row_stride` are
/// different numbers even when both tensors are packed.
///
/// # Errors
///
/// See [`head_grid`].
pub fn q_gate_split(
    ctx: &Ctx<'_>,
    qg: Buf,
    q_out: BufMut,
    gate_out: BufMut,
    head_dim: i32,
    qg_row_stride: i32,
    out_row_stride: i32,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = head_grid(head_dim, *q_heads, *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "q_gate_split_bfloat16",
            file: GATE_FILE,
            lanes,
            group: head_group(lanes),
        },
        &[
            qg.v(),
            q_out.v(),
            gate_out.v(),
            head_dim.v(),
            qg_row_stride.v(),
            out_row_stride.v(),
        ],
    )
}

/// One token's keys and values written into a contiguous cache.
///
/// ONE token, which is why the depth is 1 and not an argument: the contiguous
/// cache is the whole-history decode's, and a prefill uses the paged form.
///
/// # Errors
///
/// See [`head_grid`].
pub fn kv_append(
    ctx: &Ctx<'_>,
    k_new: Buf,
    v_new: Buf,
    k_cache: BufMut,
    v_cache: BufMut,
    pos: I32s,
    head_dim: i32,
    k_head_stride: Usize,
    k_seq_stride: Usize,
    heads: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = head_grid(head_dim, *heads, 1)?;
    ctx.dispatch(
        Fire {
            entrypoint: "kv_append_bfloat16",
            file: KV_FILE,
            lanes,
            group: head_group(lanes),
        },
        &[
            k_new.v(),
            v_new.v(),
            k_cache.v(),
            v_cache.v(),
            pos.v(),
            head_dim.v(),
            k_head_stride.v(),
            k_seq_stride.v(),
        ],
    )
}

/// [`kv_append`] into a paged pool, one token per grid.z.
///
/// # The ring slots
///
/// This entrypoint's argument table is the paged pool's, and six of its
/// sixteen slots are ones this kernel does not read: they belong to the wider
/// binding the pool's other users share, and a Metal argument table is a
/// contiguous run.
///
/// They are taken SEPARATELY -- `ring_4`, `ring_6`..`ring_9`, `ring_11`,
/// `ring_15` -- and not as one repeated handle the way `ssm.rs`'s scan and
/// `moe.rs`'s routed GEMM take a single `pad`. The row already names them one
/// by one, and every other backend states sixteen; folding them into one
/// argument would leave three ports declaring a different call than this one
/// for the same kernel, which is the disagreement the cross-backend gate
/// exists to catch. Where a caller has nothing for a slot it passes the same
/// handle six times at the call site, and that is its choice to make rather
/// than one this signature makes for it.
///
/// # Errors
///
/// See [`head_grid`].
pub fn kv_append_paged(
    ctx: &Ctx<'_>,
    k_new: Buf,
    v_new: Buf,
    k_pages: BufMut,
    v_pages: BufMut,
    ring_4: Buf,
    head_dim: i32,
    ring_6: Buf,
    ring_7: Buf,
    ring_8: Buf,
    ring_9: Buf,
    page_size: i32,
    ring_11: Buf,
    n_kv_heads: i32,
    w_page: U32s,
    w_off: U32s,
    ring_15: Buf,
    tokens: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = head_grid(head_dim, n_kv_heads, *tokens)?;
    ctx.dispatch(
        Fire {
            entrypoint: "kv_append_paged_bfloat16",
            file: KV_FILE,
            lanes,
            group: head_group(lanes),
        },
        &[
            k_new.v(),
            v_new.v(),
            k_pages.v(),
            v_pages.v(),
            ring_4.v(),
            head_dim.v(),
            ring_6.v(),
            ring_7.v(),
            ring_8.v(),
            ring_9.v(),
            page_size.v(),
            ring_11.v(),
            n_kv_heads.v(),
            w_page.v(),
            w_off.v(),
            ring_15.v(),
        ],
    )
}

/// gemma's `tanh` cap on the logits, elementwise over the vocabulary.
///
/// # Errors
///
/// See [`elementwise`](crate::routine::elementwise).
pub fn logit_softcap(
    ctx: &Ctx<'_>,
    logits: Buf,
    out: BufMut,
    params: Buf,
    n: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "logit_softcap_bfloat16",
            file: SOFTCAP_FILE,
            lanes: crate::routine::elementwise(*n, 1)?,
            group: [GROUP_X, 1, 1],
        },
        &[logits.v(), out.v(), params.v()],
    )
}

/// `attn/kv_write.metal`, which is not what the two append routines are
/// named after. The third path in this module to have been spelled from the
/// routine rather than read off the directory, and the third the disk sweep
/// caught.
const KV_FILE: &str = "attn/kv_write.metal";

/// Paged single-pass attention for a decode: one threadgroup per `(query
/// head, row)`, with the whole of that row's history walked inside it.
///
/// `sinks` is bound and not read; [`sdpa_paged_decode_sink`] is the symbol
/// that reads it, and the slot is positional so it holds an address either
/// way. gpt-oss is the model with sinks -- a per-head learned logit that joins
/// the softmax with no value behind it.
///
/// # The three page tails this cannot name
///
/// `sdpa_paged.metal` also instantiates `_d_64_p32`, `_d_128_p32` and
/// `_d_64_p32_sg8`, and none of them is in [`PAGED_DECODE`]. They set
/// `FAST_FULL`, which makes `kv_start` unconditionally 0 and puts the mask
/// test behind `if constexpr (!FAST_FULL)` -- so `window`, `attention_mask`,
/// `attention_mask_stride` and `attention_mask_enabled` are all bound and
/// none is read, and a sliding layer served by one is a FULL-attention layer.
/// The last is `BN = 8`: a 256-thread threadgroup where every point here is
/// 1024, whose threadgroup arrays are sized for eight simdgroups and would be
/// indexed with a `simd_gid` running to 31. A caller that wants one has to
/// bring its own launch and its own shorter argument list; until one does,
/// their absence here is the row's refusal made executable.
///
/// # Errors
///
/// [`Refusal::Narrow`] from [`head_point`] for a width no shader carries, and
/// see [`vector_grid`].
pub fn sdpa_paged_decode(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: PAGED_DECODE[head_point(*head_dim, &PAGED_DIMS)?],
            file: PAGED_FILE,
            lanes: vector_grid(*q_heads, *rows)?,
            group: BIG_GROUP,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
        ],
    )
}

/// [`sdpa_paged_decode`] with the sink logit in the softmax.
///
/// ONE head width, and that is the checkpoint's rather than a gap: gpt-oss is
/// the only model in the tree with sinks and its heads are 64 wide.
///
/// # Errors
///
/// See [`sdpa_paged_decode`].
pub fn sdpa_paged_decode_sink(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: PAGED_DECODE_SINK[head_point(*head_dim, &SINK_DIMS)?],
            file: PAGED_FILE,
            lanes: vector_grid(*q_heads, *rows)?,
            group: BIG_GROUP,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
        ],
    )
}

/// Paged attention for a prefill: one threadgroup per `(query head, TILE of
/// 32 query rows)`, a simdgroup to a row.
///
/// `n_rows` is an ARGUMENT and the grid's row axis is derived from it, because
/// the two are different numbers: the grid counts TILES and the kernel needs
/// the row count to guard the last partial one.
///
/// # Errors
///
/// See [`sdpa_paged_decode`] and [`tiled_grid`].
pub fn sdpa_paged_tiled(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: PAGED_TILED[head_point(*head_dim, &PAGED_DIMS)?],
            file: PAGED_FILE,
            lanes: tiled_grid(*q_heads, n_rows, BIG_GROUP[0])?,
            group: BIG_GROUP,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
        ],
    )
}

/// [`sdpa_paged_tiled`] with the sink logit in the softmax.
///
/// # Errors
///
/// See [`sdpa_paged_tiled`].
pub fn sdpa_paged_tiled_sink(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: PAGED_TILED_SINK[head_point(*head_dim, &SINK_DIMS)?],
            file: PAGED_FILE,
            lanes: tiled_grid(*q_heads, n_rows, BIG_GROUP[0])?,
            group: BIG_GROUP,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
        ],
    )
}

/// [`sdpa_paged_tiled`] over a query and an output that are not packed.
///
/// The two pitches are separate numbers: a fused QKV projection leaves q at
/// the packed width's stride while the attention output is its own tensor.
///
/// # Errors
///
/// See [`sdpa_paged_tiled`].
pub fn sdpa_paged_tiled_strided(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    n_rows: i32,
    q_row_pitch: i32,
    o_row_pitch: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: PAGED_TILED_STRIDED[head_point(*head_dim, &STRIDED_DIMS)?],
            file: PAGED_FILE,
            lanes: tiled_grid(*q_heads, n_rows, BIG_GROUP[0])?,
            group: BIG_GROUP,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
            q_row_pitch.v(),
            o_row_pitch.v(),
        ],
    )
}

/// [`sdpa_paged_tiled`] on the matrix unit.
///
/// The same 32-row tile and a different threadgroup: a simdgroup owns EIGHT
/// query rows here instead of one, so the tile is four simdgroups and 128
/// threads. That width is declared on the shader with
/// `max_total_threads_per_threadgroup(128)`, so it is not a tuning knob -- a
/// 1024-wide dispatch of it is refused by the device.
///
/// # Errors
///
/// See [`sdpa_paged_tiled`].
pub fn sdpa_paged_mma(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: PAGED_MMA[head_point(*head_dim, &SINK_DIMS)?],
            file: MMA_FILE,
            lanes: tiled_grid(*q_heads, n_rows, MMA_GROUP[0])?,
            group: MMA_GROUP,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
        ],
    )
}

/// [`sdpa_paged_mma`] with the sink logit in the softmax.
///
/// # Errors
///
/// See [`sdpa_paged_mma`].
pub fn sdpa_paged_mma_sink(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: PAGED_MMA_SINK[head_point(*head_dim, &SINK_DIMS)?],
            file: MMA_FILE,
            lanes: tiled_grid(*q_heads, n_rows, MMA_GROUP[0])?,
            group: MMA_GROUP,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
        ],
    )
}

/// Whole-history attention over a CONTIGUOUS cache, one threadgroup per
/// `(query head, row)`.
///
/// Four strides and not two: the key and value planes are separate tensors
/// with separate head and sequence pitches, and a cache written by
/// [`kv_append`] can have them differ.
///
/// 512 is not a point of the axis here. A whole-history decode at that width
/// would hold a 512-float accumulator per lane; the paged form is what serves
/// it.
///
/// # Errors
///
/// [`Refusal::Narrow`] from [`head_point`], and see [`vector_grid`].
pub fn sdpa_vector_decode(
    ctx: &Ctx<'_>,
    queries: Buf,
    keys: Buf,
    values: Buf,
    out: BufMut,
    gqa_factor: i32,
    n: i32,
    k_head_stride: Usize,
    k_seq_stride: Usize,
    v_head_stride: Usize,
    v_seq_stride: Usize,
    scale: f32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: VECTOR_DECODE[head_point(*head_dim, &VECTOR_DIMS)?],
            file: VECTOR_FILE,
            lanes: vector_grid(*q_heads, *rows)?,
            group: BIG_GROUP,
        },
        &[
            queries.v(),
            keys.v(),
            values.v(),
            out.v(),
            gqa_factor.v(),
            n.v(),
            k_head_stride.v(),
            k_seq_stride.v(),
            v_head_stride.v(),
            v_seq_stride.v(),
            scale.v(),
        ],
    )
}

/// [`sdpa_vector_decode`] over a sliding window.
///
/// The window is an OPERAND and not a flag, which is this port's rule: a
/// per-fire choice the C++ made at encode time becomes data on the dispatch.
///
/// Two row pitches the contiguous form does not have -- gemma reads its query
/// out of a wider buffer than it writes -- and two head widths, which are the
/// two the sliding models have.
///
/// # Errors
///
/// See [`sdpa_vector_decode`].
pub fn sdpa_vector_decode_swa(
    ctx: &Ctx<'_>,
    queries: Buf,
    keys: Buf,
    values: Buf,
    out: BufMut,
    gqa_factor: i32,
    n: i32,
    k_head_stride: Usize,
    k_seq_stride: Usize,
    v_head_stride: Usize,
    v_seq_stride: Usize,
    scale: f32,
    window: i32,
    q_row_stride: i32,
    o_row_stride: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: VECTOR_SWA[head_point(*head_dim, &SWA_DIMS)?],
            file: SLIDING_FILE,
            lanes: vector_grid(*q_heads, *rows)?,
            group: BIG_GROUP,
        },
        &[
            queries.v(),
            keys.v(),
            values.v(),
            out.v(),
            gqa_factor.v(),
            n.v(),
            k_head_stride.v(),
            k_seq_stride.v(),
            v_head_stride.v(),
            v_seq_stride.v(),
            scale.v(),
            window.v(),
            q_row_stride.v(),
            o_row_stride.v(),
        ],
    )
}

/// [`sdpa_vector_decode_swa`] with the sink logit in the softmax.
///
/// `sinks` is bound LAST, at buffer 14, and that is worth stating because it
/// is not where the paged forms put it: those carry the sink plane in the
/// slot the unsinked twin leaves empty, ahead of nothing. Here it is appended
/// past the two pitches, so the sliding pair shares a prefix instead.
///
/// # Errors
///
/// See [`sdpa_vector_decode`].
pub fn sdpa_vector_decode_sink(
    ctx: &Ctx<'_>,
    queries: Buf,
    keys: Buf,
    values: Buf,
    out: BufMut,
    gqa_factor: i32,
    n: i32,
    k_head_stride: Usize,
    k_seq_stride: Usize,
    v_head_stride: Usize,
    v_seq_stride: Usize,
    scale: f32,
    window: i32,
    q_row_stride: i32,
    o_row_stride: i32,
    sinks: Buf,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: VECTOR_SINK[head_point(*head_dim, &SINK_DIMS)?],
            file: SLIDING_FILE,
            lanes: vector_grid(*q_heads, *rows)?,
            group: BIG_GROUP,
        },
        &[
            queries.v(),
            keys.v(),
            values.v(),
            out.v(),
            gqa_factor.v(),
            n.v(),
            k_head_stride.v(),
            k_seq_stride.v(),
            v_head_stride.v(),
            v_seq_stride.v(),
            scale.v(),
            window.v(),
            q_row_stride.v(),
            o_row_stride.v(),
            sinks.v(),
        ],
    )
}

/// The one head width the sink and matrix-unit forms are compiled at.
///
/// gpt-oss is the only model in the tree with sinks and its heads are 64
/// wide, and the MMA tiling was written against the same checkpoint. A width
/// off this axis is a refusal and not a fallback: there is no wider
/// instantiation to fall back to.
pub const SINK_DIMS: [i32; 1] = [64];

/// The one head width the unpacked tiled prefill is compiled at.
pub const STRIDED_DIMS: [i32; 1] = [256];

/// The paged decode with sinks, at [`SINK_DIMS`].
pub const PAGED_DECODE_SINK: [&str; 1] = ["sdpa_paged_decode_sink_bfloat16_d_64"];

/// The tiled prefill with sinks, at [`SINK_DIMS`].
pub const PAGED_TILED_SINK: [&str; 1] = ["sdpa_paged_tiled_sink_bfloat16_d_64"];

/// The tiled prefill over unpacked tensors, at [`STRIDED_DIMS`].
pub const PAGED_TILED_STRIDED: [&str; 1] = ["sdpa_paged_tiled_strided_bfloat16_d_256"];

/// The matrix-unit tiling, at [`SINK_DIMS`].
pub const PAGED_MMA: [&str; 1] = ["sdpa_paged_mma_bfloat16_d_64"];

/// The matrix-unit tiling with sinks, at [`SINK_DIMS`].
pub const PAGED_MMA_SINK: [&str; 1] = ["sdpa_paged_mma_sink_bfloat16_d_64"];

/// The vector decode with sinks, at [`SINK_DIMS`].
pub const VECTOR_SINK: [&str; 1] = ["sdpa_vector_decode_sink_bfloat16_d_64"];

const SLIDING_FILE: &str = "attn/sdpa_sliding.metal";

/// The family, in the order the rows above state it.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(split_qkv_bf16),
    // In place on the attention output, and a real alias: the trace states the
    // tensor it gates and the kernel binds it once because they are the same
    // bytes.
    crate::routine!(gate, in_place = &[(0, 0)]),
    crate::routine!(kv_append),
    crate::routine!(kv_append_paged),
    crate::routine!(logit_softcap),
    crate::routine!(q_gate_split),
    crate::routine!(sdpa_paged_decode),
    crate::routine!(sdpa_paged_decode_sink),
    crate::routine!(sdpa_paged_mma),
    crate::routine!(sdpa_paged_mma_sink),
    crate::routine!(sdpa_paged_tiled),
    crate::routine!(sdpa_paged_tiled_sink),
    crate::routine!(sdpa_paged_tiled_strided),
    crate::routine!(sdpa_vector_decode),
    crate::routine!(sdpa_vector_decode_sink),
    crate::routine!(sdpa_vector_decode_swa),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do.
    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// [`gate`] covers every row, and the row that carried it did not.
    ///
    /// `gate.metal`'s header states `grid = (n_q * head_dim, rows, 1)` and its
    /// body indexes `tgpos.y` for the row. `LaunchRule::PerHeadElementwise`
    /// built `attn_gate(q_heads, head_dim)`, which is
    /// `(q_heads * head_dim, 1, 1)` -- so a prefill of 512 tokens gated token
    /// zero and left the other 511 with an ungated attention output, while a
    /// decode, being one row, looked correct throughout.
    ///
    /// This is the third launch in this port found stating fewer axes than
    /// its shader reads, after `RouterLane`'s mixture prefill and
    /// `GatedRms`'s. All three were invisible to a decode for the same reason.
    #[test]
    fn a_gate_covers_every_row_of_a_prefill() {
        let seen = Seen::default();
        gate(&seen, BufMut(1), Buf(2), 4096, Env(4096), Env(512)).expect("a launch");
        let calls = seen.0.borrow();
        let (fire, _) = &calls[0];
        assert_eq!(
            fire.lanes[1], 512,
            "the row axis the shader indexes must be the row count"
        );
        assert_eq!(fire.lanes[0], 4096, "and the width the head axis covers");
    }

    /// The paged pool's six unread slots are bound where the pool puts them.
    ///
    /// `w_page` and `w_off` are what this kernel actually writes through, and
    /// they sit at 13 and 14. Closing the ring's gaps would slide them to 7
    /// and 8, where the kernel reads two ring pointers -- and Metal validates
    /// no binding, so the token would be appended at whatever address those
    /// held.
    #[test]
    fn a_paged_append_binds_the_rings_slots_it_does_not_read() {
        let seen = Seen::default();
        let ring = Buf(99);
        kv_append_paged(
            &seen,
            Buf(1),
            Buf(2),
            BufMut(3),
            BufMut(4),
            ring,
            256,
            ring,
            ring,
            ring,
            ring,
            16,
            ring,
            8,
            U32s(11),
            U32s(12),
            ring,
            Env(1),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
        let (_, args) = &calls[0];
        assert_eq!(args.len(), 16, "the pool's whole argument table");
        assert_eq!(args[13], U32s(11).v(), "the destination page at thirteen");
        assert_eq!(args[14], U32s(12).v(), "and its offset at fourteen");
        for slot in [4, 6, 7, 8, 9, 11, 15] {
            assert_eq!(args[slot], ring.v(), "slot {slot} still holds an address");
        }
    }

    /// A head width off the compiled axis is refused, not rounded.
    ///
    /// `head_point` is the whole of the specialization: an entrypoint name is
    /// a `&'static str` picked out of a table, and there is no name for 96.
    /// Rounding to 128 would attend over 32 channels of the next head.
    #[test]
    fn a_head_width_no_shader_carries_is_refused() {
        assert!(head_point(96, &PAGED_DIMS).is_err());
        assert!(head_point(32, &PAGED_DIMS).is_err());
        assert!(
            head_point(512, &VECTOR_DIMS).is_err(),
            "a whole-history decode at 512 would hold a 512-float accumulator \
             per lane; the paged form is what serves that width"
        );
        assert_eq!(head_point(512, &PAGED_DIMS), Ok(3));
        assert_eq!(head_point(64, &PAGED_DIMS), Ok(0));
        assert_eq!(head_point(256, &PAGED_DIMS), Ok(2));
        assert_eq!(head_point(64, &SINK_DIMS), Ok(0));
        assert!(
            head_point(128, &SINK_DIMS).is_err(),
            "the sink forms are compiled at one width and gpt-oss is the model"
        );
    }

    /// A tiled prefill counts TILES on the row axis and hands the kernel the
    /// row count separately.
    ///
    /// The two are different numbers whenever the rows do not divide by 32,
    /// and the kernel needs both: the grid to know how many groups to run,
    /// the argument to guard the last one's tail.
    #[test]
    fn a_tiled_prefill_launches_a_group_per_tile_and_states_the_rows() {
        let seen = Seen::default();
        sdpa_paged_tiled(
            &seen,
            Buf(1),
            Buf(2),
            Buf(3),
            BufMut(4),
            4,
            I32s(5),
            I32s(6),
            U32s(7),
            U32s(8),
            16,
            8,
            0.125,
            U8s(9),
            0,
            U8s(10),
            0,
            Buf(11),
            100,
            Env(64),
            Env(32),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(
            fire.lanes[1], 4,
            "a hundred rows is four tiles of thirty-two, the last one partial"
        );
        assert_eq!(args[17], 100.v(), "and the kernel is told the hundred");
    }

    /// The matrix-unit tiling launches the 128 threads it was compiled for.
    ///
    /// `max_total_threads_per_threadgroup(128)` is on the shader, so this is
    /// not a tuning choice: a 1024-wide dispatch of it is refused by the
    /// device at pipeline creation.
    #[test]
    fn the_matrix_unit_tiling_launches_its_declared_width() {
        let seen = Seen::default();
        sdpa_paged_mma(
            &seen,
            Buf(1),
            Buf(2),
            Buf(3),
            BufMut(4),
            4,
            I32s(5),
            I32s(6),
            U32s(7),
            U32s(8),
            16,
            8,
            0.125,
            U8s(9),
            0,
            U8s(10),
            0,
            Buf(11),
            64,
            Env(64),
            Env(32),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
        let (fire, _) = &calls[0];
        assert_eq!(fire.group, MMA_GROUP, "the width the shader declares");
        assert_eq!(
            fire.lanes[1], 2,
            "sixty-four rows is two tiles, and a tile is still thirty-two"
        );
        assert_eq!(
            fire.lanes[0],
            32 * MMA_GROUP[0],
            "and the head axis is scaled by the narrower group, not by 1024"
        );
    }

    /// The three page tails are named by no routine here.
    ///
    /// `_d_64_p32`, `_d_128_p32` and `_d_64_p32_sg8` are compiled and
    /// unreachable, and that is the row's refusal: the first two ignore
    /// `window` and the mask, the third is a 256-thread group. A table that
    /// held them would let a sliding layer be served by a FULL-attention
    /// kernel with no diagnostic.
    #[test]
    fn no_routine_can_name_a_page_tail() {
        for name in PAGED_DECODE
            .iter()
            .chain(PAGED_DECODE_SINK.iter())
            .chain(PAGED_TILED.iter())
            .chain(VECTOR_DECODE.iter())
        {
            assert!(!name.contains("_p32"), "{name} sets FAST_FULL");
            assert!(!name.contains("_sg8"), "{name} is a 256-thread group");
        }
    }

    /// Every attention routine binds as many arguments as its entrypoint
    /// declares buffers, and names the file that carries it.
    ///
    /// Neither is checked anywhere else. The cross-backend gate compares
    /// SIGNATURES -- the parameter list -- so a parameter accepted and never
    /// bound is invisible to it, and `sdpa_paged_tiled` shipped for exactly
    /// as long as it took to write this: `n_rows` was taken, documented, and
    /// dropped, which would have left the kernel reading whatever the last
    /// dispatch left at buffer 17 as its row guard. The file is the same kind
    /// of silence -- the matrix-unit pair is its own translation unit, and
    /// asking `sdpa_paged.metal` for it yields nil at pipeline creation.
    #[test]
    fn every_attention_routine_binds_its_entrypoints_whole_table() {
        let seen = Seen::default();
        let (b, m, i, u, u8s) = (Buf(1), BufMut(2), I32s(3), U32s(4), U8s(5));
        sdpa_paged_decode(
            &seen,
            b,
            b,
            b,
            m,
            4,
            i,
            i,
            u,
            u,
            16,
            8,
            0.125,
            u8s,
            0,
            u8s,
            0,
            b,
            Env(64),
            Env(32),
            Env(1),
        )
        .expect("a launch");
        sdpa_paged_tiled(
            &seen,
            b,
            b,
            b,
            m,
            4,
            i,
            i,
            u,
            u,
            16,
            8,
            0.125,
            u8s,
            0,
            u8s,
            0,
            b,
            64,
            Env(64),
            Env(32),
        )
        .expect("a launch");
        sdpa_paged_tiled_strided(
            &seen,
            b,
            b,
            b,
            m,
            4,
            i,
            i,
            u,
            u,
            16,
            8,
            0.125,
            u8s,
            0,
            u8s,
            0,
            b,
            64,
            8192,
            4096,
            Env(256),
            Env(32),
        )
        .expect("a launch");
        sdpa_paged_mma(
            &seen,
            b,
            b,
            b,
            m,
            4,
            i,
            i,
            u,
            u,
            16,
            8,
            0.125,
            u8s,
            0,
            u8s,
            0,
            b,
            64,
            Env(64),
            Env(32),
        )
        .expect("a launch");
        sdpa_vector_decode(
            &seen,
            b,
            b,
            b,
            m,
            4,
            128,
            Usize(6),
            Usize(7),
            Usize(8),
            Usize(9),
            0.125,
            Env(64),
            Env(32),
            Env(1),
        )
        .expect("a launch");
        sdpa_vector_decode_swa(
            &seen,
            b,
            b,
            b,
            m,
            4,
            128,
            Usize(6),
            Usize(7),
            Usize(8),
            Usize(9),
            0.125,
            512,
            8192,
            4096,
            Env(256),
            Env(32),
            Env(1),
        )
        .expect("a launch");

        // The buffer counts the shaders declare, in dispatch order.
        let want = [
            (17, PAGED_FILE),
            (18, PAGED_FILE),
            (20, PAGED_FILE),
            (18, MMA_FILE),
            (11, VECTOR_FILE),
            (14, SLIDING_FILE),
        ];
        let calls = seen.0.borrow();
        for (call, (arity, file)) in calls.iter().zip(want) {
            assert_eq!(
                call.1.len(),
                arity,
                "{} binds a partial table",
                call.0.entrypoint
            );
            assert_eq!(call.0.file, file, "{} names a file", call.0.entrypoint);
        }
        assert_eq!(calls.len(), want.len(), "every form above was fired");
    }
}
