//! Attention, and the KV writes that feed it.
//!
//! The head dim is an axis everywhere here and its points are the checkpoint
//! geometries the tree is actually compiled for -- 64 (llama-3.2, gpt-oss),
//! 128 (llama, qwen), 256 (qwen3.5), 512 (gemma4 full-attn). A checkpoint
//! whose width is not a point of the axis has no pipeline, which used to be a
//! runtime pipeline-creation failure naming a string it could not find, and
//! is now a fact the table states.

#![allow(clippy::too_many_arguments)]

use kernels::KernelSig;
use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, I32s, Routine, U8s, U32s, Usize};

/// The entrypoints this family's crossed routines spell, now that their
/// rows are gone. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &[
    "split_qkv_bf16",
    "gate_bfloat16",
    "kv_append_bfloat16",
    "kv_append_paged_bfloat16",
    "logit_softcap_bfloat16",
    "q_gate_split_bfloat16",
    "sdpa_paged_decode_bfloat16_d_64",
    "sdpa_paged_decode_bfloat16_d_128",
    "sdpa_paged_decode_bfloat16_d_256",
    "sdpa_paged_decode_bfloat16_d_512",
    "sdpa_paged_decode_bfloat16_d_64_p32",
    "sdpa_paged_decode_bfloat16_d_128_p32",
    "sdpa_paged_decode_bfloat16_d_64_p32_sg8",
    "sdpa_paged_decode_sink_bfloat16_d_64",
    "sdpa_paged_mma_bfloat16_d_64",
    "sdpa_paged_mma_sink_bfloat16_d_64",
    "sdpa_paged_tiled_bfloat16_d_64",
    "sdpa_paged_tiled_bfloat16_d_128",
    "sdpa_paged_tiled_bfloat16_d_256",
    "sdpa_paged_tiled_bfloat16_d_512",
    "sdpa_paged_tiled_sink_bfloat16_d_64",
    "sdpa_paged_tiled_strided_bfloat16_d_256",
    "sdpa_vector_decode_bfloat16_d_64",
    "sdpa_vector_decode_bfloat16_d_128",
    "sdpa_vector_decode_bfloat16_d_256",
    "sdpa_vector_decode_sink_bfloat16_d_64",
    "sdpa_vector_decode_swa_bfloat16_d_256",
    "sdpa_vector_decode_swa_bfloat16_d_512",
];

pub static KERNELS: &[KernelSig] = &[];

/// The four head widths this tree compiles attention for.
///
/// `deployment::ATTN_HEAD_DIMS` is the same four, and they are the checkpoint
/// geometries and not a range: 64 (llama-3.2, gpt-oss), 128 (llama, qwen),
/// 256 (qwen3.5), 512 (gemma4 full-attn). A width off this list has no module,
/// and refusing it here is the difference between a named refusal and
/// `vkCreateComputePipelines` faulting on a string it could not find.
const PAGED_DIMS: [i32; 4] = [64, 128, 256, 512];

/// The three the dense, unpaged decode is compiled for. 512 is missing because
/// gemma-4's wide layers are paged.
const VECTOR_DIMS: [i32; 3] = [64, 128, 256];

/// The two the sliding window is compiled for. A window is a gemma statement,
/// and gemma's two widths are these.
const SWA_DIMS: [i32; 2] = [256, 512];

/// `sdpa_paged_decode`, by head width.
///
/// Four entries where the row beside it states SEVEN points. The other three
/// -- `_d_64_p32`, `_d_128_p32`, `_d_64_p32_sg8` -- are deliberately not
/// reachable from here, and `tests/gpu.rs`'s
/// `the_page_shape_tails_are_one_real_variant_and_one_bare_name` is where the
/// measurement lives. In short: `_p32` sets `PIE_FAST_FULL`, which pins the
/// key run's start to zero, so a caller who asked for a window would get FULL
/// attention and no error; and `_p32_sg8` is byte-identical to `_p32`, so a
/// caller who asked for a short group would get the ordinary one. Both
/// failures are silent and neither has a caller -- no text names a tail and
/// the driver's reachability census lists none -- so a routine that could
/// spell them would only be a way to reach them by mistake.
const PAGED_DECODE: [&str; 4] = [
    "sdpa_paged_decode_bfloat16_d_64",
    "sdpa_paged_decode_bfloat16_d_128",
    "sdpa_paged_decode_bfloat16_d_256",
    "sdpa_paged_decode_bfloat16_d_512",
];

/// `sdpa_paged_tiled`, by head width. All four are real and all four are
/// reachable.
const PAGED_TILED: [&str; 4] = [
    "sdpa_paged_tiled_bfloat16_d_64",
    "sdpa_paged_tiled_bfloat16_d_128",
    "sdpa_paged_tiled_bfloat16_d_256",
    "sdpa_paged_tiled_bfloat16_d_512",
];

/// `sdpa_vector_decode`, by head width.
const VECTOR_DECODE: [&str; 3] = [
    "sdpa_vector_decode_bfloat16_d_64",
    "sdpa_vector_decode_bfloat16_d_128",
    "sdpa_vector_decode_bfloat16_d_256",
];

/// `sdpa_vector_decode_swa`, by head width.
const VECTOR_SWA: [&str; 2] = [
    "sdpa_vector_decode_swa_bfloat16_d_256",
    "sdpa_vector_decode_swa_bfloat16_d_512",
];

/// The query rows one tiled workgroup covers, and its lane count on each axis.
///
/// `sdpa_paged.slang` and `sdpa_paged_mma.slang` both declare
/// `[numthreads(32, 32, 1)]` under `PIE_TILED`, and the tile height is the
/// same 32: the staged key run is shared across 32 query rows, which is the
/// whole reason the tiled form exists beside the decode one.
const TILE: u32 = 32;

/// Which point of an axis a head width is.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a width the shader tree does not carry. `Narrow`
/// rather than `Wide` even at 1024, because the axis is a LIST and not an
/// interval: 512 is compiled and 384 is not, so "too large" is not what is
/// wrong with either.
fn head_point(head_dim: i32, points: &[i32]) -> Result<usize, Refusal> {
    points
        .iter()
        .position(|d| *d == head_dim)
        .ok_or(Refusal::Narrow {
            what: "the head width",
            at: i64::from(head_dim),
        })
}

/// The vector shape: one workgroup per (query head, row), one lane per head
/// dimension.
///
/// `sdpa_vector.slang`, `sdpa_sliding.slang` and `sdpa_paged.slang`'s decode
/// branch all declare `[numthreads(PIE_HEAD_DIM, 1, 1)]` and all three read
/// `gl_NumWorkGroups.x` back as the query head count, so the x extent in
/// THREADS is the product and the driver's `div_ceil` gives the head count
/// back exactly. It gives it back exactly only because the head width IS the
/// workgroup width -- which is why `head_dim` is a point of the axis here
/// rather than a number the caller chooses.
///
/// # Errors
///
/// [`Refusal::Empty`] for a head count or row count that is zero or negative,
/// and [`Refusal::Grid`] if the x product does not fit a `u32`.
fn vector_grid(head_dim: i32, q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if q_heads <= 0 {
        return Err(Refusal::Empty {
            what: "query heads",
        });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let x = q_heads
        .unsigned_abs()
        .checked_mul(head_dim.unsigned_abs())
        .ok_or(Refusal::Grid {
            what: "query heads * the head width",
            at: i64::from(q_heads) * i64::from(head_dim),
        })?;
    Ok([x, rows.unsigned_abs(), 1])
}

/// The tiled shape: one 32 x 32 workgroup per (query head, block of 32 rows).
///
/// The row count is rounded UP to whole tiles, which is what the `n_rows`
/// push scalar is for: the last tile's rows past the end have to know they are
/// past it, and the grid cannot tell them.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty head or row count, and [`Refusal::Grid`]
/// if either axis does not fit a `u32` once multiplied by the tile.
fn tiled_grid(q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if q_heads <= 0 {
        return Err(Refusal::Empty {
            what: "query heads",
        });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let x = q_heads
        .unsigned_abs()
        .checked_mul(TILE)
        .ok_or(Refusal::Grid {
            what: "query heads * the tile's lane count",
            at: i64::from(q_heads) * i64::from(TILE),
        })?;
    let y = rows
        .unsigned_abs()
        .div_ceil(TILE)
        .checked_mul(TILE)
        .ok_or(Refusal::Grid {
            what: "rows rounded up to whole tiles",
            at: i64::from(rows),
        })?;
    Ok([x, y, 1])
}

/// The KV write shape: one lane per head dimension, one row per head, one
/// plane per token.
///
/// # Errors
///
/// [`Refusal::Empty`] for any of the three being zero or negative.
fn head_grid(head_dim: i32, heads: i32, depth: i32) -> Result<[u32; 3], Refusal> {
    if head_dim <= 0 {
        return Err(Refusal::Empty {
            what: "the head width",
        });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if depth <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    Ok([
        head_dim.unsigned_abs(),
        heads.unsigned_abs(),
        depth.unsigned_abs(),
    ])
}

/// One packed row cut into three, at two boundaries the caller does not state.
///
/// `q_width` and `kv_width` ride in `params` rather than in the push block --
/// the shader reads them out of a `SplitQkvParams` struct -- so this signature
/// cannot check them and does not pretend to. What it does state is
/// `packed_width`, which is `q_width + 2 * kv_width` and is the extent the
/// grid needs; the shader recomputes the same sum from its own copy and
/// guards on it, so a `packed_width` that disagreed would leave a tail of the
/// row uncopied rather than write out of bounds.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty row or an empty rectangle.
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
            lanes: crate::routine::elementwise_rows(*packed_width, *rows)?,
        },
        &[packed.v(), q.v(), k.v(), v.v(), params.v()],
    )
}

/// qwen3.5's attention gate: `attn *= sigmoid(gate)`, in place.
///
/// `row_stride` is a real argument and not a convenience. When it is zero the
/// shader falls back to `gl_NumWorkGroups.x * 256`, which is the width ROUNDED
/// UP to whole workgroups -- so the fallback is the true row pitch only when
/// the width is already a multiple of 256, and is silently too large
/// otherwise. Pass the pitch unless the row is known to be aligned.
///
/// The tail past the row is guarded by the output's own descriptor range
/// (`GetDimensions`), not by anything here, which is why this body has no
/// width to check against a length.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle.
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
            lanes: crate::routine::elementwise_rows(*width, *rows)?,
        },
        &[attn.v(), gate.v(), row_stride.v()],
    )
}

/// qwen3.5's `[query|gate]` split: one interleaved head row cut in two.
///
/// `q_heads` is grid-only and load-bearing: the shader reads it back as
/// `gl_NumWorkGroups.y` to compute the default row pitches. The y axis is one
/// workgroup per head because `[numthreads]` is `(256, 1, 1)`, so the count
/// comes back exactly.
///
/// # Errors
///
/// Whatever [`head_grid`] refuses, with the head COUNT on the y axis rather
/// than a token depth.
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
    ctx.dispatch(
        Fire {
            entrypoint: "q_gate_split_bfloat16",
            lanes: head_grid(head_dim, *q_heads, *rows)?,
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

/// The contiguous KV append: one token into the slot `pos` names.
///
/// The two strides are the POOL's, in elements, and they are
/// [`Usize`] because the shader declares them `uint2`. That is also why this
/// is the family that made `driver-vulkan::encode::words` align an extent to
/// its own width: `Push { int head_dim; uint2 k_head_stride; uint2
/// k_seq_stride; }` leaves a four-byte hole after `head_dim`, and a packer
/// that concatenated would push twenty bytes where the module declares
/// twenty-four.
///
/// One token, hence the `1` on the z axis: `pos` is read at index 0 and the
/// destination row is the same for every head.
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
    ctx.dispatch(
        Fire {
            entrypoint: "kv_append_bfloat16",
            lanes: head_grid(head_dim, *heads, 1)?,
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

/// The paged KV append: many tokens, each into the page and offset its own
/// entry names.
///
/// Sixteen arguments, six of which reach the device.
///
/// `kv_write.slang`'s paged branch reads only the four KV planes and the two
/// write tables; bindings 4 through 9 and 11 are the Metal ring ABI's slots
/// and no Vulkan module compiles them. slangc emits no `OpDecorate Binding`
/// for a global it did not compile, so the module's own answer is
/// `{0, 1, 2, 3, 10, 11}` -- six used slots inside a twelve-wide layout.
///
/// The LAYOUT keeps its holes: `Device::build` sizes the descriptor set over
/// `0..declared.bindings`, so `w_page` still lands at descriptor 10. The CALL
/// does not: `Device::slots` skips every unused index while writing
/// descriptors, and `encode::dispatch` refuses any list whose length is not
/// `declared.bindings - holes()`. So the six placeholders that would push
/// `w_page` to the tenth argument are exactly the six that make the dispatch
/// too long.
///
/// The SIGNATURE keeps them anyway, under `_` names. It is the operand list
/// the trace fills and the row states, and `shader_backends_agree.rs` holds
/// the two against each other while the kernel is stated twice. What a body
/// may do is decline to forward one -- a subsequence, never a reordering,
/// which `tests/routines.rs` measures both halves of.
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
    _ring_4: Buf,
    head_dim: i32,
    _ring_6: Buf,
    _ring_7: Buf,
    _ring_8: Buf,
    _ring_9: Buf,
    page_size: i32,
    _ring_11: Buf,
    n_kv_heads: i32,
    w_page: U32s,
    w_off: U32s,
    _ring_15: Buf,
    tokens: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "kv_append_paged_bfloat16",
            lanes: head_grid(head_dim, n_kv_heads, *tokens)?,
        },
        &[
            k_new.v(),
            v_new.v(),
            k_pages.v(),
            v_pages.v(),
            head_dim.v(),
            page_size.v(),
            n_kv_heads.v(),
            w_page.v(),
            w_off.v(),
        ],
    )
}

/// gemma's final logit softcap: `cap * tanh(x / cap)`.
///
/// The cap rides in `params` with a trailing word nothing reads, so the only
/// extent this body states is the element count.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty readout.
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
            lanes: crate::routine::elementwise(*n, 1)?,
        },
        &[logits.v(), out.v(), params.v()],
    )
}

/// Paged decode attention: one query row per workgroup row, walking the pages
/// its request owns.
///
/// Ten buffers, and six of them are the FIRE's tables rather than the
/// statement's values -- the positions, which request owns each token, the
/// page CSR, the mask and its enable. `sdpa_paged.slang` declares `sinks` at
/// binding 10 unconditionally, but nothing reads it without `PIE_WITH_SINK`
/// and slangc decorates no binding for a global it compiled out, so this form
/// takes ten arguments and [`sdpa_paged_decode_sink`] -- whose module really
/// does decorate 10 -- takes eleven. One text, two signatures, and the
/// module's decoration set is the only thing that says which is which.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a head width off [`PAGED_DIMS`], and whatever
/// [`vector_grid`] refuses.
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
    _sinks: Buf,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: PAGED_DECODE[head_point(*head_dim, &PAGED_DIMS)?],
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
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
        ],
    )
}

/// Paged decode with attention sinks.
///
/// A sink is a per-head learned logit that joins the softmax with no value
/// behind it -- gpt-oss's, and the one point compiled is `d_64` because that
/// is gpt-oss's head width. The signature is [`sdpa_paged_decode`]'s
/// unchanged, with `sinks` actually read.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever
/// [`vector_grid`] refuses.
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
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            entrypoint: "sdpa_paged_decode_sink_bfloat16_d_64",
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
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

/// Paged prefill attention: a tile of 32 query rows against one staged run of
/// keys.
///
/// The difference from [`sdpa_paged_decode`] is not the arithmetic but the
/// sharing. The decode form gives one workgroup to each (head, query row) and
/// each walks the whole key run alone; this one stages the run once for 32
/// rows, which is why it exists and why `n_rows` appears: the grid rounds up
/// to whole tiles, so the last tile has rows past the end and only this scalar
/// can tell them so.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a head width off [`PAGED_DIMS`], and whatever
/// [`tiled_grid`] refuses.
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
    _sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: PAGED_TILED[head_point(*head_dim, &PAGED_DIMS)?],
            lanes: tiled_grid(*q_heads, n_rows)?,
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
            n_rows.v(),
        ],
    )
}

/// Paged prefill with attention sinks. One point, `d_64`.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever [`tiled_grid`]
/// refuses.
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
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            entrypoint: "sdpa_paged_tiled_sink_bfloat16_d_64",
            lanes: tiled_grid(*q_heads, n_rows)?,
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

/// Paged prefill whose query and output rows have their own pitches.
///
/// TWO more push scalars than [`sdpa_paged_tiled`] and they are the whole
/// difference: `PIE_STRIDED` adds `q_row_pitch` and `o_row_pitch` to the end
/// of the block, so a checkpoint that reads its queries out of a wider buffer
/// than it writes has somewhere to say so. One point, `d_256`, which is
/// qwen3.5's.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 256, and whatever
/// [`tiled_grid`] refuses.
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
    _sinks: Buf,
    n_rows: i32,
    q_row_pitch: i32,
    o_row_pitch: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    head_point(*head_dim, &[256])?;
    ctx.dispatch(
        Fire {
            entrypoint: "sdpa_paged_tiled_strided_bfloat16_d_256",
            lanes: tiled_grid(*q_heads, n_rows)?,
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
            n_rows.v(),
            q_row_pitch.v(),
            o_row_pitch.v(),
        ],
    )
}

/// The cooperative-matrix prefill.
///
/// The same eighteen operands and the same 32-row tile as
/// [`sdpa_paged_tiled`]: the module's push block is field for field identical.
/// What differs is inside -- the `Q.K^T` and `P.V` inner products go through
/// `coopMatMulAdd`.
///
/// The tier does NOT change the entrypoint. `sdpa_paged_mma.slang` names
/// `sdpa_paged_mma_bfloat16_d_64` on both its plain and its `@coopmat`
/// instantiate lines, and which module a name resolves to is the driver's
/// choice from the device's capability tiers, not this body's. That is why a
/// [`Fire`] carries a name and no tier.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever [`tiled_grid`]
/// refuses.
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
    _sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            entrypoint: "sdpa_paged_mma_bfloat16_d_64",
            lanes: tiled_grid(*q_heads, n_rows)?,
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
            n_rows.v(),
        ],
    )
}

/// The cooperative-matrix prefill with attention sinks.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever [`tiled_grid`]
/// refuses.
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
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            entrypoint: "sdpa_paged_mma_sink_bfloat16_d_64",
            lanes: tiled_grid(*q_heads, n_rows)?,
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

/// The dense, unpaged decode: one contiguous cache, addressed by four strides.
///
/// The four strides are [`Usize`] because `sdpa_vector.slang` declares them
/// `uint2`. Here they need no alignment hole -- two `int`s precede them, so
/// the first already starts on an eight-byte boundary -- which is exactly why
/// `driver-vulkan::encode::words` pads before an extent rather than after one.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a head width off [`VECTOR_DIMS`], and whatever
/// [`vector_grid`] refuses.
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
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
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

/// The dense decode over a SLIDING window.
///
/// The window is an operand and not a flag -- the port's rule that a per-fire
/// choice the C++ made at encode time becomes data on the dispatch -- and it
/// comes with two row pitches the contiguous form does not have, because
/// gemma reads its query out of a wider buffer than it writes.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a head width off [`SWA_DIMS`], and whatever
/// [`vector_grid`] refuses.
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
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
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

/// The dense decode with attention sinks.
///
/// It lives in `sdpa_sliding.slang` rather than in `sdpa_vector.slang`, which
/// is why it takes the SLIDING signature: the window and the two row pitches
/// are in the push block whether or not a caller wants a window, and `sinks`
/// takes binding 4, which the windowed form leaves undeclared. So this is
/// [`sdpa_vector_decode_swa`] plus one buffer, and not
/// [`sdpa_vector_decode`] plus one buffer.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever
/// [`vector_grid`] refuses.
pub fn sdpa_vector_decode_sink(
    ctx: &Ctx<'_>,
    queries: Buf,
    keys: Buf,
    values: Buf,
    out: BufMut,
    sinks: Buf,
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
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            entrypoint: "sdpa_vector_decode_sink_bfloat16_d_64",
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
        },
        &[
            queries.v(),
            keys.v(),
            values.v(),
            out.v(),
            sinks.v(),
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

/// The sixteen, in the order the rows above name them.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(split_qkv_bf16),
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

    type Call = (String, [u32; 3], Vec<ArgValue>);

    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    fn one(seen: &Seen) -> Call {
        let calls = seen.0.borrow();
        assert_eq!(calls.len(), 1, "expected exactly one dispatch");
        calls[0].clone()
    }

    /// Buffer handles, so a bound list reads back as the names it came from.
    fn handles(call: &Call) -> Vec<u32> {
        call.2
            .iter()
            .filter_map(|v| match v {
                ArgValue::Buffer { handle, .. } => Some(*handle),
                _ => None,
            })
            .collect()
    }

    /// One dispatch of the paged decode, with everything but the swept
    /// arguments held still.
    fn decode(seen: &Seen, head_dim: i32, q_heads: i32, rows: i32) -> Result<(), Refusal> {
        sdpa_paged_decode(
            seen,
            Buf(0),
            Buf(1),
            Buf(2),
            BufMut(3),
            4,
            I32s(4),
            I32s(5),
            U32s(6),
            U32s(7),
            32,
            2,
            0.125,
            U8s(8),
            0,
            U8s(9),
            0,
            Buf(10),
            Env(head_dim),
            Env(q_heads),
            Env(rows),
        )
    }

    /// The vector shape multiplies the head width INTO the x extent, and the
    /// tiled shape rounds the rows up to whole tiles.
    ///
    /// Two different mistakes, and both are silent. The lane counts this crate
    /// states are in THREADS -- `driver-vulkan::encode` does the `div_ceil`
    /// into workgroups -- so an x extent of the head COUNT would divide by the
    /// 128-wide workgroup to one group and attend one head out of five, and
    /// `gl_NumWorkGroups.x`, which the shader reads back as the head count,
    /// would say so too late. On the tiled side a y extent of the row count
    /// would launch one workgroup where 33 rows need two, and the 32 rows past
    /// the first tile would keep whatever the output buffer held.
    #[test]
    fn the_vector_extent_is_heads_times_the_head_width_and_a_tile_is_rounded_up() {
        let seen = Seen::default();
        sdpa_vector_decode(
            &seen,
            Buf(0),
            Buf(1),
            Buf(2),
            BufMut(3),
            4,
            17,
            Usize(1),
            Usize(2),
            Usize(3),
            Usize(4),
            0.125,
            Env(128),
            Env(5),
            Env(3),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(call.0, "sdpa_vector_decode_bfloat16_d_128");
        assert_eq!(call.1, [640, 3, 1]);

        let seen = Seen::default();
        sdpa_paged_tiled(
            &seen,
            Buf(0),
            Buf(1),
            Buf(2),
            BufMut(3),
            4,
            I32s(4),
            I32s(5),
            U32s(6),
            U32s(7),
            32,
            2,
            0.125,
            U8s(8),
            0,
            U8s(9),
            0,
            Buf(10),
            33,
            Env(64),
            Env(5),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(call.0, "sdpa_paged_tiled_bfloat16_d_64");
        assert_eq!(
            call.1,
            [160, 64, 1],
            "33 rows are two tiles, and a tile is 32 lanes on each axis"
        );
    }

    /// The paged append passes the six buffers its module uses, and the two
    /// write tables are the last of them.
    ///
    /// The tempting reading of "`w_page` is at binding 10" is that ten
    /// buffers must precede it. They must precede it in the LAYOUT, and they
    /// do -- `Device::build` sizes the set over `0..declared.bindings` and the
    /// holes keep their slots. They must not precede it in the CALL:
    /// `Device::slots` skips every unused index when it writes descriptors,
    /// and `encode::dispatch` refuses a buffer list whose length is not
    /// `declared.bindings - holes()`. Padding this call to thirteen is a
    /// `Refusal::Arity` on the device, not a wasted descriptor.
    ///
    /// `every_routine_binds_the_buffers_its_module_uses_and_no_others` in
    /// `tests/routines.rs` is what measures the six against the compiled
    /// module; this test only fixes the ORDER, which SPIR-V cannot state.
    #[test]
    fn the_paged_appends_write_tables_are_its_last_two_buffers() {
        let seen = Seen::default();
        kv_append_paged(
            &seen,
            Buf(0),
            Buf(1),
            BufMut(2),
            BufMut(3),
            Buf(4),
            64,
            Buf(5),
            Buf(6),
            Buf(7),
            Buf(8),
            32,
            Buf(9),
            4,
            U32s(10),
            U32s(11),
            Buf(12),
            Env(5),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(call.0, "kv_append_paged_bfloat16");
        assert_eq!(call.1, [64, 4, 5], "one lane per head element, per token");
        assert_eq!(
            call.2.len(),
            9,
            "six buffers and three scalars, and the six are what the module \
             decorates"
        );
        assert_eq!(
            &call.2[4..],
            &[
                ArgValue::I32(64),
                ArgValue::I32(32),
                ArgValue::I32(4),
                ArgValue::Buffer {
                    handle: 10,
                    writes: false
                },
                ArgValue::Buffer {
                    handle: 11,
                    writes: false
                },
            ],
            "the write tables come after the four KV planes, and land at \
             descriptors 10 and 11 because `Device::slots` skips the holes"
        );
    }

    #[test]
    fn the_contiguous_append_is_one_token_deep_and_its_strides_are_extents() {
        let seen = Seen::default();
        kv_append(
            &seen,
            Buf(0),
            Buf(1),
            BufMut(2),
            BufMut(3),
            I32s(4),
            128,
            Usize(4096),
            Usize(128),
            Env(8),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(call.1, [128, 8, 1]);
        assert_eq!(
            &call.2[5..],
            &[
                ArgValue::I32(128),
                ArgValue::Usize(4096),
                ArgValue::Usize(128)
            ],
            "the strides push as eight-byte extents"
        );
    }

    /// The sinked dense decode takes the SLIDING signature, not the vector
    /// one.
    ///
    /// It is compiled out of `sdpa_sliding.slang`, so its push block carries
    /// `window`, `q_row_stride` and `o_row_stride` whether or not a caller
    /// wants a window, and `sinks` occupies binding 4 -- a binding the
    /// windowed form leaves undeclared. Giving it `sdpa_vector_decode`'s
    /// eleven arguments plus a sink would push a block three words short of
    /// what the module declares, which `Device::dispatch` refuses, and would
    /// bind the sink where the module's first stride word is not.
    #[test]
    fn the_sinked_dense_decode_is_the_sliding_signature_and_not_the_vector_one() {
        let seen = Seen::default();
        sdpa_vector_decode_sink(
            &seen,
            Buf(0),
            Buf(1),
            Buf(2),
            BufMut(3),
            Buf(4),
            4,
            17,
            Usize(1),
            Usize(2),
            Usize(3),
            Usize(4),
            0.125,
            512,
            256,
            256,
            Env(64),
            Env(5),
            Env(3),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(handles(&call), vec![0, 1, 2, 3, 4], "the sink is binding 4");
        assert_eq!(
            call.2.len() - handles(&call).len(),
            10,
            "gqa, n, four strides, scale, window and two row pitches"
        );

        let seen = Seen::default();
        sdpa_vector_decode_swa(
            &seen,
            Buf(0),
            Buf(1),
            Buf(2),
            BufMut(3),
            4,
            17,
            Usize(1),
            Usize(2),
            Usize(3),
            Usize(4),
            0.125,
            512,
            256,
            256,
            Env(256),
            Env(5),
            Env(3),
        )
        .unwrap();
        let swa = one(&seen);
        assert_eq!(
            handles(&swa),
            vec![0, 1, 2, 3],
            "and the windowed form has none"
        );
        assert_eq!(
            swa.2.len() - handles(&swa).len(),
            10,
            "with the same scalars"
        );
    }

    /// A head width off the axis is refused BY NAME, at every width the axis
    /// does not carry.
    ///
    /// The point of refusing rather than formatting a name is that an unbuilt
    /// module is not an error on this backend: `vkCreateComputePipelines`
    /// faults on it with the validation layer silent. 512 is the case worth
    /// having: it is a real width this tree compiles the PAGED kernels for, so
    /// a reader could reasonably expect the dense decode to take it, and the
    /// dense decode has no `d_512` module.
    #[test]
    fn a_head_width_the_tree_does_not_carry_is_refused_by_name() {
        for width in [1, 32, 96, 384, 1024] {
            assert_eq!(
                decode(&Seen::default(), width, 5, 3),
                Err(Refusal::Narrow {
                    what: "the head width",
                    at: i64::from(width)
                }),
                "the paged decode has no module at {width}"
            );
        }
        let seen = Seen::default();
        assert_eq!(
            sdpa_vector_decode(
                &seen,
                Buf(0),
                Buf(1),
                Buf(2),
                BufMut(3),
                4,
                17,
                Usize(1),
                Usize(2),
                Usize(3),
                Usize(4),
                0.125,
                Env(512),
                Env(5),
                Env(3),
            ),
            Err(Refusal::Narrow {
                what: "the head width",
                at: 512
            }),
            "512 is paged-only: gemma-4's wide layers do not take this path"
        );
    }

    /// No head width spells a page-shape tail.
    ///
    /// The row beside `sdpa_paged_decode` states seven points and this routine
    /// reaches four. The three it does not are not an oversight: `_p32` sets
    /// `PIE_FAST_FULL`, which pins the key run's start to zero, so a caller
    /// who asked for a window would silently get FULL attention; and
    /// `_p32_sg8` is byte-identical to `_p32`, so a caller who asked for a
    /// short group would silently get the ordinary one.
    /// `tests/gpu.rs::the_page_shape_tails_are_one_real_variant_and_one_bare_name`
    /// is where both halves are measured against the compiled modules. This
    /// asserts the consequence: there is no way to reach either from here.
    #[test]
    fn no_head_width_reaches_a_page_shape_tail() {
        for width in PAGED_DIMS {
            let seen = Seen::default();
            decode(&seen, width, 5, 3).unwrap();
            let name = one(&seen).0;
            assert_eq!(name, format!("sdpa_paged_decode_bfloat16_d_{width}"));
            assert!(
                !name.contains("_p32"),
                "`{name}` is a page-shape tail, and one of the three is a bare \
                 name that would launch the ordinary workgroup"
            );
        }
    }

    /// An empty extent is refused rather than launched, on both shapes.
    ///
    /// `vkCmdDispatch(0, 1, 1)` is legal Vulkan that runs nothing and reports
    /// success, so a zero here is an attention output that keeps whatever the
    /// buffer held. Zero arrives honestly -- a turn with no rows to prefill --
    /// which is why this is a value the caller reads and not a panic.
    #[test]
    fn an_empty_head_count_or_row_count_is_refused() {
        assert_eq!(
            decode(&Seen::default(), 64, 0, 3),
            Err(Refusal::Empty {
                what: "query heads"
            })
        );
        assert_eq!(
            decode(&Seen::default(), 64, 5, 0),
            Err(Refusal::Empty { what: "rows" })
        );
        assert_eq!(
            tiled_grid(5, 0),
            Err(Refusal::Empty { what: "rows" }),
            "and a tile of no rows is not one tile"
        );
    }
}
