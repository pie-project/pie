//! Attention, and the KV writes that feed it.
//!
//! The head dim is an axis everywhere here and its points are the checkpoint
//! geometries the tree is actually compiled for -- 64 (llama-3.2, gpt-oss),
//! 128 (llama, qwen), 256 (qwen3.5), 512 (gemma4 full-attn). A checkpoint
//! whose width is not a point of the axis has no pipeline, which used to be a
//! runtime pipeline-creation failure naming a string it could not find, and
//! is now a fact the table states.

#![allow(clippy::too_many_arguments)]

use kernels_macros::routine;
use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, Usize, bf16, elementwise, elementwise_rows, keys};
use kernels::KernelSig;
use kernels::BindMut;
use kernels::routine::Refusal;


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
    "sdpa_paged_decode_split_bfloat16_d_64",
    "sdpa_paged_decode_split_bfloat16_d_128",
    "sdpa_paged_decode_split_bfloat16_d_256",
    "sdpa_paged_decode_split_bfloat16_d_512",
    "sdpa_paged_decode_combine_bfloat16_d_64",
    "sdpa_paged_decode_combine_bfloat16_d_128",
    "sdpa_paged_decode_combine_bfloat16_d_256",
    "sdpa_paged_decode_combine_bfloat16_d_512",
    "sdpa_paged_decode_combine_sink_bfloat16_d_64",
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

/// The split pass of the flash decode, by head width.
///
/// One table and not two: the sink and the sinkless forms share these
/// modules, because a sink joins the softmax ONCE over the whole key range
/// and the split pass sees only a slice of it. See [`PAGED_COMBINE`].
const PAGED_SPLIT: [&str; 4] = [
    "sdpa_paged_decode_split_bfloat16_d_64",
    "sdpa_paged_decode_split_bfloat16_d_128",
    "sdpa_paged_decode_split_bfloat16_d_256",
    "sdpa_paged_decode_split_bfloat16_d_512",
];

/// The fold pass, by head width. The sink form is a fifth module at 64 and is
/// named where it is used, the same way `sdpa_paged_decode_sink` names its
/// one point.
const PAGED_COMBINE: [&str; 4] = [
    "sdpa_paged_decode_combine_bfloat16_d_64",
    "sdpa_paged_decode_combine_bfloat16_d_128",
    "sdpa_paged_decode_combine_bfloat16_d_256",
    "sdpa_paged_decode_combine_bfloat16_d_512",
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

/// How many ways a decode splits its key range, given the history and the
/// shape.
///
/// # Why there is a rule at all
///
/// A decode's grid is `(query head, row)` and nothing else, so qwen3-0.6b at
/// one row is SIXTEEN workgroups on a 128-SM card. `tests/sdpa_bench.rs`
/// priced what that costs: at a 384-key history one row and thirty-two rows
/// both take about 68 us -- thirty-two times the work for twelve percent more
/// time -- so the kernel was never bandwidth-bound, it was waiting, and the
/// only axis left to make workgroups out of is the key range.
///
/// Splitting it `S` ways multiplies the grid by `S` and costs a second,
/// cheap dispatch that folds the `S` partials. Measured, one row of
/// qwen3-0.6b, microseconds for the WHOLE attention (both dispatches):
///
/// | history | S=1 | 2 | 4 | 8 | 16 | 32 | 64 |
/// |---|---|---|---|---|---|---|---|
/// | 24 | 6.14 | 5.79 | 5.09 | **4.80** | 5.12 | 6.88 | 9.28 |
/// | 128 | 24.13 | 14.34 | 9.22 | 6.75 | **6.18** | 7.01 | 9.66 |
/// | 384 | 67.58 | 36.51 | 20.19 | 12.29 | 9.09 | **8.42** | 10.78 |
/// | 1024 | 178.59 | 91.81 | 48.26 | 26.27 | 16.32 | **12.35** | 13.31 |
///
/// Three things are read off that table and they are the whole rule:
///
///   - the best `S` GROWS with the history, because what a split has to hide
///     is its own slice's latency and a slice too short has nothing to hide;
///   - it stops growing at 32, because past that the fold's `S` reads per
///     output dimension start costing more than the split saves;
///   - and it must fall with the row count, because rows multiply the grid
///     too: at 32 rows the single-pass kernel is already 4,096-workgroup
///     work and `S = 4` (55.1 us against 76.0) is as much as it wants.
///
/// So: aim for about 2,048 workgroups, allow one split per 8 keys of
/// history, cap at 32, and round DOWN to a power of two.
///
/// # Why the history is bucketed by the caller
///
/// `S` decides a grid, a grid is recorded into a command buffer, and
/// `driver-vulkan::replay` re-submits a recorded decode across tokens. A
/// rule reading the exact history would change the grid every token and the
/// replay would have to notice every time. The caller therefore hands a
/// history ROUNDED UP TO A POWER OF TWO, which for the life of a sequence
/// changes a handful of times, and folds that same bucketed number into the
/// replay key. See `driver-vulkan::replay::Key::state`.
///
/// Returns 1 when the split is not worth its fold, and the caller then fires
/// the single-pass [`sdpa_paged_decode`] path unchanged.
#[must_use]
pub fn decode_splits(history_bucket: i32, q_heads: i32, rows: i32) -> i32 {
    /// Workgroups the split pass aims at, across every head and row.
    const TARGET_GROUPS: i64 = 2048;
    /// Keys a split must have to be worth being a split.
    const KEYS_PER_SPLIT: i64 = 8;
    /// Past this the fold costs more than the split saves.
    const MOST: i64 = 32;

    if history_bucket <= 0 || q_heads <= 0 || rows <= 0 {
        return 1;
    }
    // The A/B switch the numbers in this doc-comment were taken with, read
    // ONCE: this is called per layer per step, and `var_os` allocates.
    // Setting it puts every decode back on the single-pass path, which is how
    // "4.07 ms/token" and "1.66 ms/token" were measured against each other on
    // one machine on one afternoon rather than across two commits.
    static UNSPLIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *UNSPLIT.get_or_init(|| std::env::var_os("PIE_NO_FLASH_DECODE").is_some()) {
        return 1;
    }
    let base = i64::from(q_heads) * i64::from(rows);
    let want = (TARGET_GROUPS / base)
        .min(i64::from(history_bucket) / KEYS_PER_SPLIT)
        .min(MOST);
    if want < 2 {
        return 1;
    }
    // Down to a power of two, which is what keeps the breakpoints few: `S`
    // then changes only where the bucketed history does.
    1 << (63 - want.leading_zeros() as i64).min(30)
}

/// The split pass's shape: [`vector_grid`] with a third axis of splits.
///
/// # Errors
///
/// Whatever [`vector_grid`] refuses, and [`Refusal::Empty`] for a split count
/// that is not positive.
fn split_grid(head_dim: i32, q_heads: i32, rows: i32, splits: i32) -> Result<[u32; 3], Refusal> {
    if splits <= 0 {
        return Err(Refusal::Empty { what: "splits" });
    }
    let [x, y, _] = vector_grid(head_dim, q_heads, rows)?;
    Ok([x, y, splits.unsigned_abs()])
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
#[routine]
pub fn split_qkv_bf16(
    ctx: &Ctx<'_>,
    packed: In<Tensor<bf16>>,
    q: Out<Tensor<bf16>>,
    k: Out<Tensor<bf16>>,
    v: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let packed_width = packed.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("split_qkv_bf16", ctx.best()), "split_qkv_bf16").apply(elementwise_rows(packed_width, rows)?),
        &[packed.arg(), q.arg(), k.arg(), v.arg(), params],
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
#[routine]
pub fn gate(
    ctx: &Ctx<'_>,
    attn: InOut<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    // THE STATEMENT'S STRIDE, WHICH WAS `Param<0, i32>`. A row stride is the
    // rectangle the text laid out, not something this batch made, so it fails
    // `ask`'s own test and no driver answers `keys::RowStride`.
    row_stride: Const<i32>) -> Result<(), Refusal> {
    let width = attn.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("gate_bfloat16", ctx.best()), "gate_bfloat16").apply(elementwise_rows(width, rows)?),
        &[attn.arg(), gate.arg(), row_stride.arg()],
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
#[routine]
pub fn q_gate_split(
    ctx: &Ctx<'_>,
    qg: In<Tensor<bf16>>,
    q_out: Out<Tensor<bf16>>,
    gate_out: Out<Tensor<bf16>>,
    head_dim: Const<i32>,
    // THE TWO STRIDES, WHICH WERE `Param<1>` AND `Param<2>`. A row stride is
    // the rectangle the text laid out -- two fires of one deployment stride the
    // same way -- so both fail `ask`'s own test and no driver answers
    // `keys::QgRowStride` or `keys::OutRowStride`. The split refused
    // `Unstated` on all three planes while they were asks.
    qg_row_stride: Const<i32>,
    out_row_stride: Const<i32>,
    q_heads: Const<i32>) -> Result<(), Refusal> {
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("q_gate_split_bfloat16", ctx.best()), "q_gate_split_bfloat16").apply(head_grid(*head_dim, *q_heads, rows)?),
        &[
            qg.arg(),
            q_out.arg(),
            gate_out.arg(),
            head_dim.arg(),
            qg_row_stride.arg(),
            out_row_stride.arg(),
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
#[routine]
pub fn kv_append(
    ctx: &Ctx<'_>,
    k_new: In<Tensor<bf16>>,
    v_new: In<Tensor<bf16>>,
    head_dim: Const<i32>,
    heads: Const<i32>) -> Result<(), Refusal> {
    let k_cache = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let v_cache = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    let pos = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let k_head_stride = ctx.ask::<Usize, keys::KvHeadStride>()?;
    let k_seq_stride = ctx.ask::<Usize, keys::KvSeqStride>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("kv_append_bfloat16", ctx.best()), "kv_append_bfloat16").apply(head_grid(*head_dim, *heads, 1)?),
        &[
            k_new.arg(),
            v_new.arg(),
            k_cache.arg_mut(),
            v_cache.arg_mut(),
            pos.arg(),
            head_dim.arg(),
            k_head_stride.arg(),
            k_seq_stride.arg(),
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
#[routine]
pub fn kv_append_paged(
    ctx: &Ctx<'_>,
    k_new: In<Tensor<bf16>>,
    v_new: In<Tensor<bf16>>,
    head_dim: Const<i32>,
    n_kv_heads: Const<i32>) -> Result<(), Refusal> {
    // THE POOL'S NUMBER, ASKED FOR. `page_size` is a property of the
    // allocation this driver made, not of the model the text describes,
    // so no builder has it to state -- it was `Held<keys::KvPageSize>`
    // before the marks and it is an ask now.
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;

    let k_pages = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let v_pages = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    let _ring_4 = ctx.absent()?;
    let _ring_6 = ctx.absent()?;
    let _ring_7 = ctx.absent()?;
    let _ring_8 = ctx.absent()?;
    let _ring_9 = ctx.absent()?;
    let _ring_11 = ctx.absent()?;
    let w_page = ctx.ask::<Tensor<u32>, keys::KvWritePage>()?;
    let w_off = ctx.ask::<Tensor<u32>, keys::KvWriteOffset>()?;
    let _ring_15 = ctx.absent()?;
    let tokens = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("kv_append_paged_bfloat16", ctx.best()), "kv_append_paged_bfloat16").apply(head_grid(*head_dim, *n_kv_heads, tokens)?),
        &[
            k_new.arg(),
            v_new.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            head_dim.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            w_page.arg(),
            w_off.arg(),
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
#[routine]
pub fn logit_softcap(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    // THE ELEMENT COUNT, DERIVED RATHER THAN ASKED. HEAD spelled it
    // `Reckoned<Times<Say<Width>, Say<Rows>>>` -- a product of two facts,
    // not a fact -- and the migration turned it into `keys::Elements`,
    // which no driver answers. Both halves are on the operand's own
    // rectangle, so the body multiplies them.
    let n = out.rows.saturating_mul(out.width);
    ctx.fire(
        Fire::at(crate::routine::module_path("logit_softcap_bfloat16", ctx.best()), "logit_softcap_bfloat16").apply(elementwise(n, 1)?),
        &[logits.arg(), out.arg(), params],
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
#[routine]
pub fn sdpa_paged_decode(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>) -> Result<(), Refusal> {
    // THE POOL'S NUMBER, ASKED FOR. `page_size` is a property of the
    // allocation this driver made, not of the model the text describes,
    // so no builder has it to state -- it was `Held<keys::KvPageSize>`
    // before the marks and it is an ask now.
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;

    let k_pages = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let v_pages = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let gqa_factor = if *n_kv_heads > 0 { *q_heads / *n_kv_heads } else { 0 };
    let position_ids = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let req_of_token = ctx.ask::<Tensor<i32>, keys::RequestOfToken>()?;
    let kv_page_indices = ctx.ask::<Tensor<u32>, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<Tensor<u32>, keys::KvPageIndptr>()?;
    let attention_mask = ctx.ask::<Tensor<u8>, keys::AttentionMask>()?;
    let attention_mask_stride = ctx.ask::<u32, keys::AttentionMaskStride>()?;
    let attention_mask_enabled = ctx.ask::<Tensor<u8>, keys::AttentionMaskEnabled>()?;
    let _sinks = ctx.absent()?;
    let partials = ctx.ask::<Tensor<f32>, keys::AttnPartials>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let splits = ctx.ask::<i32, keys::AttnSplits>()?;
    let at = head_point(*head_dim, &PAGED_DIMS)?;
    if splits <= 1 {
        return ctx.fire(
            Fire::at(crate::routine::module_path(PAGED_DECODE[at], ctx.best()), PAGED_DECODE[at]).apply(vector_grid(*head_dim, *q_heads, rows)?),
            &[
                queries.arg(),
                k_pages.arg_mut(),
                v_pages.arg_mut(),
                out.arg(),
                gqa_factor.arg(),
                position_ids.arg(),
                req_of_token.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                page_size.arg(),
                n_kv_heads.arg(),
                scale.arg(),
                attention_mask.arg(),
                attention_mask_stride.arg(),
                attention_mask_enabled.arg(),
                window.arg(),
            ],
        );
    }
    flash_decode(
        ctx,
        Flash {
            split: PAGED_SPLIT[at],
            combine: PAGED_COMBINE[at],
            sinks: None,
        },
        queries.ptr,
        k_pages,
        v_pages,
        out.ptr,
        gqa_factor,
        position_ids,
        req_of_token,
        kv_page_indices,
        kv_page_indptr,
        page_size,
        *n_kv_heads,
        *scale,
        attention_mask,
        attention_mask_stride,
        attention_mask_enabled,
        *window,
        partials,
        *head_dim,
        *q_heads,
        rows,
        splits,
    )
}

/// Which two modules a flash decode fires, and whether the fold merges a
/// sink.
///
/// A struct because [`flash_decode`] would otherwise take twenty-two
/// positional arguments of which three are strings, and the three that say
/// WHICH ARITHMETIC belong together.
struct Flash {
    /// The pass that walks a slice of the keys.
    split: &'static str,
    /// The pass that folds the slices.
    combine: &'static str,
    /// The per-head sink logit, and the fold module that reads it. `None` is
    /// the ordinary decode.
    sinks: Option<(Tensor<bf16>, &'static str)>,
}

/// The two dispatches of a flash decode.
///
/// Shared by the sink and sinkless forms, which differ in the fold module and
/// in one binding -- the split pass is byte-identical between them, because
/// the sink is merged once in the fold.
///
/// # The split pass binds no output
///
/// `sdpa_paged.slang` declares `out_` at binding 3 for every decode variant,
/// and the split body never writes it, so slangc drops it and the module's
/// descriptor set carries a HOLE there. The argument list below therefore
/// goes `queries, k, v, positions, ...` with no `out` -- which is what
/// `Declared::holes` and `encode`'s arity check are counting.
///
/// # Errors
///
/// Whatever [`split_grid`] or [`vector_grid`] refuses.
fn flash_decode(
    ctx: &Ctx<'_>,
    which: Flash,
    queries: Tensor<bf16>,
    k_pages: Tensor<bf16>,
    v_pages: Tensor<bf16>,
    out: Tensor<bf16>,
    gqa_factor: i32,
    position_ids: Tensor<i32>,
    req_of_token: Tensor<i32>,
    kv_page_indices: Tensor<u32>,
    kv_page_indptr: Tensor<u32>,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: Tensor<u8>,
    attention_mask_stride: u32,
    attention_mask_enabled: Tensor<u8>,
    window: i32,
    partials: Tensor<f32>,
    head_dim: i32,
    q_heads: i32,
    rows: i32,
    splits: i32) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(crate::routine::module_path(which.split, ctx.best()), which.split).apply(split_grid(head_dim, q_heads, rows, splits)?),
        // In the order the SIGNATURE takes them, minus the two the split
        // pass does not touch. `encode` splits buffers from scalars on its
        // own; what has to hold here is that neither list is reordered
        // relative to the signature -- see `tests/routines.rs`.
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            partials.arg_mut(),
        ],
    )?;
    // The fold reads what the split wrote, which the driver turns into one
    // barrier between the two: they share `partials` and the first writes it.
    // `partials` is handed over WRITABLE here although the fold only reads
    // it. The flag is not documentation: `driver-vulkan` decides its barriers
    // from it, and a conservative "writes" between two dispatches that share
    // the buffer costs one execution barrier it was going to need anyway --
    // where narrowing it to a read and getting the direction wrong is a race.
    // `arg_mut`, and the whole decode depends on it. `out` arrives here as a
    // bare `Tensor<bf16>` -- the caller's `Out<Tensor<bf16>>` mark was spent
    // at the signature and cannot travel through a helper's positional
    // parameter -- so the direction has to be restated at the bind. Bound
    // with `arg()` the fold's own output is not in the driver's write set,
    // `driver-vulkan::device::hazards` sees no write-then-read between this
    // dispatch and the projection that consumes the attention output, and no
    // barrier is emitted: the projection reads the previous step's numbers
    // wherever the card has not caught up. It is a race, so it is fluent and
    // wrong rather than a crash, and it costs one barrier per layer per
    // decode step to be right.
    let mut args = vec![out.arg_mut()];
    let entrypoint = match which.sinks {
        Some((sinks, module)) => {
            args.push(sinks.arg());
            module
        }
        None => which.combine,
    };
    args.push(partials.arg_mut());
    args.push(splits.arg());
    ctx.fire(
        Fire::at(crate::routine::module_path(entrypoint, ctx.best()), entrypoint).apply(vector_grid(head_dim, q_heads, rows)?),
        &args,
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
#[routine]
pub fn sdpa_paged_decode_sink(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    sinks: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    q_heads: Const<i32>) -> Result<(), Refusal> {
    // THE POOL'S NUMBER, ASKED FOR. `page_size` is a property of the
    // allocation this driver made, not of the model the text describes,
    // so no builder has it to state -- it was `Held<keys::KvPageSize>`
    // before the marks and it is an ask now.
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;

    let k_pages = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let v_pages = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let gqa_factor = if *n_kv_heads > 0 { *q_heads / *n_kv_heads } else { 0 };
    let position_ids = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let req_of_token = ctx.ask::<Tensor<i32>, keys::RequestOfToken>()?;
    let kv_page_indices = ctx.ask::<Tensor<u32>, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<Tensor<u32>, keys::KvPageIndptr>()?;
    let attention_mask = ctx.ask::<Tensor<u8>, keys::AttentionMask>()?;
    let attention_mask_stride = ctx.ask::<u32, keys::AttentionMaskStride>()?;
    let attention_mask_enabled = ctx.ask::<Tensor<u8>, keys::AttentionMaskEnabled>()?;
    let partials = ctx.ask::<Tensor<f32>, keys::AttnPartials>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let splits = ctx.ask::<i32, keys::AttnSplits>()?;
    head_point(*head_dim, &[64])?;
    if splits > 1 {
        return flash_decode(
            ctx,
            Flash {
                split: PAGED_SPLIT[0],
                combine: PAGED_COMBINE[0],
                sinks: Some((*sinks, "sdpa_paged_decode_combine_sink_bfloat16_d_64")),
            },
            queries.ptr,
            k_pages,
            v_pages,
            out.ptr,
            gqa_factor,
            position_ids,
            req_of_token,
            kv_page_indices,
            kv_page_indptr,
            page_size,
            *n_kv_heads,
            *scale,
            attention_mask,
            attention_mask_stride,
            attention_mask_enabled,
            *window,
            partials,
            *head_dim,
            *q_heads,
            rows,
            splits,
        );
    }
    ctx.fire(
        Fire::at(crate::routine::module_path("sdpa_paged_decode_sink_bfloat16_d_64", ctx.best()), "sdpa_paged_decode_sink_bfloat16_d_64").apply(vector_grid(*head_dim, *q_heads, rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks.arg(),
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
#[routine]
pub fn sdpa_paged_tiled(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>) -> Result<(), Refusal> {
    // THE POOL'S NUMBER, ASKED FOR. `page_size` is a property of the
    // allocation this driver made, not of the model the text describes,
    // so no builder has it to state -- it was `Held<keys::KvPageSize>`
    // before the marks and it is an ask now.
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;

    let k_pages = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let v_pages = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let gqa_factor = if *n_kv_heads > 0 { *q_heads / *n_kv_heads } else { 0 };
    let position_ids = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let req_of_token = ctx.ask::<Tensor<i32>, keys::RequestOfToken>()?;
    let kv_page_indices = ctx.ask::<Tensor<u32>, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<Tensor<u32>, keys::KvPageIndptr>()?;
    let attention_mask = ctx.ask::<Tensor<u8>, keys::AttentionMask>()?;
    let attention_mask_stride = ctx.ask::<u32, keys::AttentionMaskStride>()?;
    let attention_mask_enabled = ctx.ask::<Tensor<u8>, keys::AttentionMaskEnabled>()?;
    let _sinks = ctx.absent()?;
    let n_rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path(PAGED_TILED[head_point(*head_dim, &PAGED_DIMS)?], ctx.best()), PAGED_TILED[head_point(*head_dim, &PAGED_DIMS)?]).apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            n_rows.arg(),
        ],
    )
}

/// Paged prefill with attention sinks. One point, `d_64`.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever [`tiled_grid`]
/// refuses.
#[routine]
pub fn sdpa_paged_tiled_sink(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    sinks: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    q_heads: Const<i32>) -> Result<(), Refusal> {
    // THE POOL'S NUMBER, ASKED FOR. `page_size` is a property of the
    // allocation this driver made, not of the model the text describes,
    // so no builder has it to state -- it was `Held<keys::KvPageSize>`
    // before the marks and it is an ask now.
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;

    let k_pages = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let v_pages = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let gqa_factor = if *n_kv_heads > 0 { *q_heads / *n_kv_heads } else { 0 };
    let position_ids = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let req_of_token = ctx.ask::<Tensor<i32>, keys::RequestOfToken>()?;
    let kv_page_indices = ctx.ask::<Tensor<u32>, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<Tensor<u32>, keys::KvPageIndptr>()?;
    let attention_mask = ctx.ask::<Tensor<u8>, keys::AttentionMask>()?;
    let attention_mask_stride = ctx.ask::<u32, keys::AttentionMaskStride>()?;
    let attention_mask_enabled = ctx.ask::<Tensor<u8>, keys::AttentionMaskEnabled>()?;
    let n_rows = ctx.ask::<i32, keys::Rows>()?;
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(crate::routine::module_path("sdpa_paged_tiled_sink_bfloat16_d_64", ctx.best()), "sdpa_paged_tiled_sink_bfloat16_d_64").apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks.arg(),
            n_rows.arg(),
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
#[routine]
pub fn sdpa_paged_tiled_strided(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>) -> Result<(), Refusal> {
    // THE POOL'S NUMBER, ASKED FOR. `page_size` is a property of the
    // allocation this driver made, not of the model the text describes,
    // so no builder has it to state -- it was `Held<keys::KvPageSize>`
    // before the marks and it is an ask now.
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;

    let k_pages = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let v_pages = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let gqa_factor = if *n_kv_heads > 0 { *q_heads / *n_kv_heads } else { 0 };
    let position_ids = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let req_of_token = ctx.ask::<Tensor<i32>, keys::RequestOfToken>()?;
    let kv_page_indices = ctx.ask::<Tensor<u32>, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<Tensor<u32>, keys::KvPageIndptr>()?;
    let attention_mask = ctx.ask::<Tensor<u8>, keys::AttentionMask>()?;
    let attention_mask_stride = ctx.ask::<u32, keys::AttentionMaskStride>()?;
    let attention_mask_enabled = ctx.ask::<Tensor<u8>, keys::AttentionMaskEnabled>()?;
    let _sinks = ctx.absent()?;
    let n_rows = ctx.ask::<i32, keys::Rows>()?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<5>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::QRowPitch`, which no driver answers.
    let q_row_pitch = ctx.param(5)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<6>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::ORowPitch`, which no driver answers.
    let o_row_pitch = ctx.param(6)?;
    head_point(*head_dim, &[256])?;
    ctx.fire(
        Fire::at(crate::routine::module_path("sdpa_paged_tiled_strided_bfloat16_d_256", ctx.best()), "sdpa_paged_tiled_strided_bfloat16_d_256").apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            n_rows.arg(),
            q_row_pitch.arg(),
            o_row_pitch.arg(),
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
#[routine]
pub fn sdpa_paged_mma(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>) -> Result<(), Refusal> {
    // THE POOL'S NUMBER, ASKED FOR. `page_size` is a property of the
    // allocation this driver made, not of the model the text describes,
    // so no builder has it to state -- it was `Held<keys::KvPageSize>`
    // before the marks and it is an ask now.
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;

    let k_pages = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let v_pages = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let gqa_factor = if *n_kv_heads > 0 { *q_heads / *n_kv_heads } else { 0 };
    let position_ids = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let req_of_token = ctx.ask::<Tensor<i32>, keys::RequestOfToken>()?;
    let kv_page_indices = ctx.ask::<Tensor<u32>, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<Tensor<u32>, keys::KvPageIndptr>()?;
    let attention_mask = ctx.ask::<Tensor<u8>, keys::AttentionMask>()?;
    let attention_mask_stride = ctx.ask::<u32, keys::AttentionMaskStride>()?;
    let attention_mask_enabled = ctx.ask::<Tensor<u8>, keys::AttentionMaskEnabled>()?;
    let _sinks = ctx.absent()?;
    let n_rows = ctx.ask::<i32, keys::Rows>()?;
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(crate::routine::module_path("sdpa_paged_mma_bfloat16_d_64", ctx.best()), "sdpa_paged_mma_bfloat16_d_64").apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            n_rows.arg(),
        ],
    )
}

/// The cooperative-matrix prefill with attention sinks.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever [`tiled_grid`]
/// refuses.
#[routine]
pub fn sdpa_paged_mma_sink(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    sinks: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    q_heads: Const<i32>) -> Result<(), Refusal> {
    // THE POOL'S NUMBER, ASKED FOR. `page_size` is a property of the
    // allocation this driver made, not of the model the text describes,
    // so no builder has it to state -- it was `Held<keys::KvPageSize>`
    // before the marks and it is an ask now.
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;

    let k_pages = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let v_pages = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let gqa_factor = if *n_kv_heads > 0 { *q_heads / *n_kv_heads } else { 0 };
    let position_ids = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let req_of_token = ctx.ask::<Tensor<i32>, keys::RequestOfToken>()?;
    let kv_page_indices = ctx.ask::<Tensor<u32>, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<Tensor<u32>, keys::KvPageIndptr>()?;
    let attention_mask = ctx.ask::<Tensor<u8>, keys::AttentionMask>()?;
    let attention_mask_stride = ctx.ask::<u32, keys::AttentionMaskStride>()?;
    let attention_mask_enabled = ctx.ask::<Tensor<u8>, keys::AttentionMaskEnabled>()?;
    let n_rows = ctx.ask::<i32, keys::Rows>()?;
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(crate::routine::module_path("sdpa_paged_mma_sink_bfloat16_d_64", ctx.best()), "sdpa_paged_mma_sink_bfloat16_d_64").apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks.arg(),
            n_rows.arg(),
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
#[routine]
pub fn sdpa_vector_decode(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>) -> Result<(), Refusal> {
    let keys = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let values = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let n_kv_heads = ctx.ask::<i32, keys::NumKvHeads>()?;
    let gqa_factor = if n_kv_heads > 0 { *q_heads / n_kv_heads } else { 0 };
    let n = out.width;
    let k_head_stride = ctx.ask::<Usize, keys::KvHeadStride>()?;
    let k_seq_stride = ctx.ask::<Usize, keys::KvSeqStride>()?;
    let v_head_stride = ctx.ask::<Usize, keys::KvHeadStride>()?;
    let v_seq_stride = ctx.ask::<Usize, keys::KvSeqStride>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path(VECTOR_DECODE[head_point(*head_dim, &VECTOR_DIMS)?], ctx.best()), VECTOR_DECODE[head_point(*head_dim, &VECTOR_DIMS)?]).apply(vector_grid(*head_dim, *q_heads, rows)?),
        &[
            queries.arg(),
            keys.arg(),
            values.arg(),
            out.arg(),
            gqa_factor.arg(),
            n.arg(),
            k_head_stride.arg(),
            k_seq_stride.arg(),
            v_head_stride.arg(),
            v_seq_stride.arg(),
            scale.arg(),
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
#[routine]
pub fn sdpa_vector_decode_swa(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    // THE TWO PITCHES, WHICH WERE `Param<4>` AND `Param<5>`. A row stride is
    // the rectangle the text laid out -- two fires of one deployment stride
    // the same way -- so it fails `ask`'s own test, and no driver answers
    // `keys::QRowStride` or `keys::ORowStride`. Every sliding and sinked
    // decode refused `Unstated` while they were asks.
    q_row_stride: Const<i32>,
    o_row_stride: Const<i32>) -> Result<(), Refusal> {
    let keys = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let values = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let n_kv_heads = ctx.ask::<i32, keys::NumKvHeads>()?;
    let gqa_factor = if n_kv_heads > 0 { *q_heads / n_kv_heads } else { 0 };
    let n = out.width;
    let k_head_stride = ctx.ask::<Usize, keys::KvHeadStride>()?;
    let k_seq_stride = ctx.ask::<Usize, keys::KvSeqStride>()?;
    let v_head_stride = ctx.ask::<Usize, keys::KvHeadStride>()?;
    let v_seq_stride = ctx.ask::<Usize, keys::KvSeqStride>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path(VECTOR_SWA[head_point(*head_dim, &SWA_DIMS)?], ctx.best()), VECTOR_SWA[head_point(*head_dim, &SWA_DIMS)?]).apply(vector_grid(*head_dim, *q_heads, rows)?),
        &[
            queries.arg(),
            keys.arg(),
            values.arg(),
            out.arg(),
            gqa_factor.arg(),
            n.arg(),
            k_head_stride.arg(),
            k_seq_stride.arg(),
            v_head_stride.arg(),
            v_seq_stride.arg(),
            scale.arg(),
            window.arg(),
            q_row_stride.arg(),
            o_row_stride.arg(),
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
#[routine]
pub fn sdpa_vector_decode_sink(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    sinks: Const<Tensor<bf16>>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    // THE TWO PITCHES, WHICH WERE `Param<4>` AND `Param<5>`. A row stride is
    // the rectangle the text laid out -- two fires of one deployment stride
    // the same way -- so it fails `ask`'s own test, and no driver answers
    // `keys::QRowStride` or `keys::ORowStride`. Every sliding and sinked
    // decode refused `Unstated` while they were asks.
    q_row_stride: Const<i32>,
    o_row_stride: Const<i32>) -> Result<(), Refusal> {
    let keys = ctx.ask::<Tensor<bf16>, keys::KvKeys>()?;
    let values = ctx.ask::<Tensor<bf16>, keys::KvValues>()?;
    // GQA'S FACTOR, DERIVED FROM THE TWO COUNTS THE STATEMENT CARRIES.
    // It was `Param<0, i32>` -- a scalar the DSL stated -- and the migration
    // read it as a fire's fact and turned it into an ask no driver answers,
    // so every paged attention refused `Unstated`. It is neither: it is
    // `q_heads / n_kv_heads` and both are already `Const<i32>` here, so the
    // one number is derived from the two rather than stated a third time.
    let n_kv_heads = ctx.ask::<i32, keys::NumKvHeads>()?;
    let gqa_factor = if n_kv_heads > 0 { *q_heads / n_kv_heads } else { 0 };
    let n = out.width;
    let k_head_stride = ctx.ask::<Usize, keys::KvHeadStride>()?;
    let k_seq_stride = ctx.ask::<Usize, keys::KvSeqStride>()?;
    let v_head_stride = ctx.ask::<Usize, keys::KvHeadStride>()?;
    let v_seq_stride = ctx.ask::<Usize, keys::KvSeqStride>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(crate::routine::module_path("sdpa_vector_decode_sink_bfloat16_d_64", ctx.best()), "sdpa_vector_decode_sink_bfloat16_d_64").apply(vector_grid(*head_dim, *q_heads, rows)?),
        &[
            queries.arg(),
            keys.arg(),
            values.arg(),
            out.arg(),
            sinks.arg(),
            gqa_factor.arg(),
            n.arg(),
            k_head_stride.arg(),
            k_seq_stride.arg(),
            v_head_stride.arg(),
            v_seq_stride.arg(),
            scale.arg(),
            window.arg(),
            q_row_stride.arg(),
            o_row_stride.arg(),
        ],
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    type Call = (String, [u32; 3], Vec<ArgValue>);

    struct Seen {
        calls: RefCell<Vec<Call>>,
        rows: Cell<i32>,
        gqa_factor: Cell<i32>,
        row_stride: Cell<i32>,
        qg_row_stride: Cell<i32>,
        out_row_stride: Cell<i32>,
        k_head_stride: Cell<u32>,
        k_seq_stride: Cell<u32>,
        attention_mask_stride: Cell<u32>,
        q_row_pitch: Cell<i32>,
        o_row_pitch: Cell<i32>,
        q_row_stride: Cell<i32>,
        o_row_stride: Cell<i32>,
        attn_splits: Cell<i32>,
        elements: Cell<i32>,
        kv_keys: Cell<u32>,
        kv_values: Cell<u32>,
        positions: Cell<u32>,
        request_of_token: Cell<u32>,
        kv_page_indices: Cell<u32>,
        kv_page_indptr: Cell<u32>,
        attention_mask: Cell<u32>,
        attention_mask_enabled: Cell<u32>,
        attn_partials: Cell<u32>,
        kv_write_page: Cell<u32>,
        kv_write_offset: Cell<u32>,
        params_handle: Cell<u32>,
        /// THE STATEMENT\'S SCALAR RUN, for a body that reads a word by
        /// index. Empty means "4096 at every slot", which is a plausible
        /// stride for the rows these tests build; a case that means a
        /// particular tiling or split count sets its own.
        words: RefCell<Vec<i32>>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                rows: Cell::new(3),
                gqa_factor: Cell::new(2),
                row_stride: Cell::new(256),
                qg_row_stride: Cell::new(512),
                out_row_stride: Cell::new(256),
                k_head_stride: Cell::new(4096),
                k_seq_stride: Cell::new(128),
                attention_mask_stride: Cell::new(77),
                q_row_pitch: Cell::new(512),
                o_row_pitch: Cell::new(256),
                q_row_stride: Cell::new(512),
                o_row_stride: Cell::new(256),
                attn_splits: Cell::new(1),
                elements: Cell::new(16),
                kv_keys: Cell::new(1),
                kv_values: Cell::new(2),
                positions: Cell::new(3),
                request_of_token: Cell::new(4),
                kv_page_indices: Cell::new(5),
                kv_page_indptr: Cell::new(6),
                attention_mask: Cell::new(7),
                attention_mask_enabled: Cell::new(8),
                attn_partials: Cell::new(9),
                kv_write_page: Cell::new(10),
                kv_write_offset: Cell::new(11),
                params_handle: Cell::new(900),
                words: RefCell::default(),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            // THE POOL'S PAGE SIZE AND THE CACHE'S HEAD COUNT, both asked
            // for by the paged bodies: the first is a property of the
            // allocation the driver made and the second is the geometry
            // `gqa_factor` is derived from. Constants rather than fields:
            // no case in this file varies either, and a `Cell` nobody sets
            // reads as a knob that does nothing. 32 is what
            // `the_paged_appends_write_tables_are_its_last_two_buffers`
            // pins by name, and it predates the ask -- `page_size` was a
            // parameter that test supplied directly before it moved into
            // the body.
            if source == <keys::KvPageSize as Fact>::SOURCE {
                return Ok(ArgValue::I32(32));
            }
            if source == <keys::NumKvHeads as Fact>::SOURCE {
                return Ok(ArgValue::I32(8));
            }
            if source == <keys::GqaFactor as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.gqa_factor.get()));
            }
            if source == <keys::KvHeadStride as Fact>::SOURCE {
                return Ok(ArgValue::Usize(u64::from(self.k_head_stride.get())));
            }
            if source == <keys::KvSeqStride as Fact>::SOURCE {
                return Ok(ArgValue::Usize(u64::from(self.k_seq_stride.get())));
            }
            if source == <keys::AttentionMaskStride as Fact>::SOURCE {
                return Ok(ArgValue::U32(self.attention_mask_stride.get()));
            }
            if source == <keys::QRowStride as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.q_row_stride.get()));
            }
            if source == <keys::ORowStride as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.o_row_stride.get()));
            }
            if source == <keys::AttnSplits as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.attn_splits.get()));
            }
            if source == <keys::KvKeys as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.kv_keys.get(), writes: false, rows: 0, width: 0 });
            }
            if source == <keys::KvValues as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.kv_values.get(), writes: false, rows: 0, width: 0 });
            }
            if source == <keys::Positions as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.positions.get(), writes: false, rows: 0, width: 0 });
            }
            if source == <keys::RequestOfToken as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.request_of_token.get(), writes: false, rows: 0, width: 0 });
            }
            if source == <keys::KvPageIndices as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.kv_page_indices.get(), writes: false, rows: 0, width: 0 });
            }
            if source == <keys::KvPageIndptr as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.kv_page_indptr.get(), writes: false, rows: 0, width: 0 });
            }
            if source == <keys::AttentionMask as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.attention_mask.get(), writes: false, rows: 0, width: 0 });
            }
            if source == <keys::AttentionMaskEnabled as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.attention_mask_enabled.get(), writes: false, rows: 0, width: 0 });
            }
            if source == <keys::AttnPartials as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.attn_partials.get(), writes: true, rows: 0, width: 0 });
            }
            if source == <keys::KvWritePage as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.kv_write_page.get(), writes: false, rows: 0, width: 0 });
            }
            if source == <keys::KvWriteOffset as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: self.kv_write_offset.get(), writes: false, rows: 0, width: 0 });
            }
            // THE STATEMENT'S OWN SCALARS, which a body reads by index when its
            // params run is a struct and no `Const` mark can name a word inside
            // it -- see `Asks::param`. The probe answers a number that is
            // plausible for every reader: a stride wide enough for the rows
            // these tests build, and a positive tiling.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                return Ok(ArgValue::I32(
                    self.words.borrow().get(usize::from(n)).copied().unwrap_or(4096),
                ));
            }
            if source == kernels::Source::Slot(kernels::Kind::Params, 0) {
                return Ok(ArgValue::Buffer { handle: self.params_handle.get(), writes: false, rows: 0, width: 0 });
            }
            if matches!(ty, kernels::Ty::Buf) {
                return Ok(ArgValue::Buffer { handle: 900, writes: false, rows: 0, width: 0 });
            }
            // Anything else is refused: a probe that invented an answer to a
            // fact it does not know would let a body pass under test while
            // the same fact went unanswered on a real driver.
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    fn one(seen: &Seen) -> Call {
        let calls = seen.calls.borrow();
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

    fn decode(seen: &Seen, head_dim: i32, q_heads: i32, rows: i32) -> Result<(), Refusal> {
        seen.rows.set(rows);
        seen.attn_splits.set(1);
        sdpa_paged_decode(
            seen,
            In { ptr: Tensor::<bf16>::new(0), rows, width: head_dim * q_heads },
            Out { ptr: Tensor::<bf16>::new(3), rows, width: head_dim * q_heads },
            Const::new(2),
            Const::new(0.125),
            Const::new(0),
            Const::new(head_dim),
            Const::new(q_heads),
        )
    }

    fn splitting(
        seen: &Seen,
        head_dim: i32,
        q_heads: i32,
        rows: i32,
        splits: i32,
    ) -> Result<(), Refusal> {
        seen.rows.set(rows);
        seen.attn_splits.set(splits);
        sdpa_paged_decode(
            seen,
            In { ptr: Tensor::<bf16>::new(0), rows, width: head_dim * q_heads },
            Out { ptr: Tensor::<bf16>::new(3), rows, width: head_dim * q_heads },
            Const::new(2),
            Const::new(0.125),
            Const::new(0),
            Const::new(head_dim),
            Const::new(q_heads),
        )
    }

    #[test]
    fn a_split_decode_fires_the_split_grid_then_a_flat_fold() {
        let seen = Seen::default();
        splitting(&seen, 128, 16, 1, 8).expect("a legal split decode");
        let calls = seen.calls.borrow().clone();
        assert_eq!(calls.len(), 2, "a split pass and a fold pass");
        assert_eq!(calls[0].0, "sdpa_paged_decode_split_bfloat16_d_128");
        assert_eq!(calls[0].1, [16 * 128, 1, 8], "heads * width, rows, splits");
        assert_eq!(calls[1].0, "sdpa_paged_decode_combine_bfloat16_d_128");
        assert_eq!(calls[1].1, [16 * 128, 1, 1], "the fold has no split axis");
    }

    #[test]
    fn one_split_is_the_original_single_dispatch() {
        let seen = Seen::default();
        splitting(&seen, 128, 16, 1, 1).expect("a legal decode");
        let call = one(&seen);
        assert_eq!(call.0, "sdpa_paged_decode_bfloat16_d_128");
    }

    #[test]
    fn the_split_rule_grows_with_the_history_and_falls_with_the_rows() {
        assert_eq!(decode_splits(0, 16, 1), 1, "no history stated");
        assert_eq!(decode_splits(32, 16, 1), 4, "32 keys, four ways");
        assert_eq!(decode_splits(128, 16, 1), 16);
        assert_eq!(decode_splits(512, 16, 1), 32, "capped");
        assert_eq!(decode_splits(8192, 16, 1), 32, "still capped");
        assert_eq!(decode_splits(1024, 16, 8), 16);
        assert_eq!(decode_splits(1024, 16, 32), 4);
        assert_eq!(decode_splits(1024, 16, 512), 1, "a prefill needs no help");
        assert_eq!(decode_splits(8, 16, 1), 1, "a split would hold one key");
    }

    #[test]
    fn the_vector_extent_is_heads_times_the_head_width_and_a_tile_is_rounded_up() {
        let seen = Seen::default();
        seen.rows.set(3);
        seen.k_head_stride.set(1);
        seen.k_seq_stride.set(2);
        sdpa_vector_decode(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 3, width: 17 },
            Out { ptr: Tensor::<bf16>::new(3), rows: 3, width: 17 },
            Const::new(0.125),
            Const::new(128),
            Const::new(5),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(call.0, "sdpa_vector_decode_bfloat16_d_128");
        assert_eq!(call.1, [640, 3, 1]);

        let seen = Seen::default();
        seen.rows.set(33);
        sdpa_paged_tiled(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 33, width: 64 * 5 },
            Out { ptr: Tensor::<bf16>::new(3), rows: 33, width: 64 * 5 },
            Const::new(2),
            Const::new(0.125),
            Const::new(0),
            Const::new(64),
            Const::new(5),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(call.0, "sdpa_paged_tiled_bfloat16_d_64");
        assert_eq!(call.1, [160, 64, 1], "33 rows are two tiles, and a tile is 32 lanes on each axis");
    }

    #[test]
    fn the_paged_appends_write_tables_are_its_last_two_buffers() {
        let seen = Seen::default();
        seen.rows.set(5);
        seen.kv_keys.set(2);
        seen.kv_values.set(3);
        seen.kv_write_page.set(10);
        seen.kv_write_offset.set(11);
        kv_append_paged(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 5, width: 64 * 4 },
            In { ptr: Tensor::<bf16>::new(1), rows: 5, width: 64 * 4 },
            Const::new(64),
            Const::new(4),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(call.0, "kv_append_paged_bfloat16");
        assert_eq!(call.1, [64, 4, 5], "one lane per head element, per token");
        assert_eq!(call.2.len(), 9, "six buffers and three scalars, and the six are what the module decorates");
        assert_eq!(
            &call.2[4..],
            &[
                ArgValue::I32(64),
                ArgValue::I32(32),
                ArgValue::I32(4),
                ArgValue::Buffer { handle: 10, writes: false, rows: 0, width: 0 },
                ArgValue::Buffer { handle: 11, writes: false, rows: 0, width: 0 },
            ],
            "the write tables come after the four KV planes, and land at descriptors 10 and 11 because `Device::slots` skips the holes"
        );
    }

    #[test]
    fn the_contiguous_append_is_one_token_deep_and_its_strides_are_extents() {
        let seen = Seen::default();
        seen.positions.set(4);
        seen.k_head_stride.set(4096);
        seen.k_seq_stride.set(128);
        kv_append(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 1, width: 128 * 8 },
            In { ptr: Tensor::<bf16>::new(1), rows: 1, width: 128 * 8 },
            Const::new(128),
            Const::new(8),
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

    #[test]
    fn the_sinked_dense_decode_is_the_sliding_signature_and_not_the_vector_one() {
        let seen = Seen::default();
        seen.rows.set(3);
        seen.q_row_stride.set(256);
        seen.o_row_stride.set(256);
        sdpa_vector_decode_sink(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 3, width: 256 },
            Out { ptr: Tensor::<bf16>::new(3), rows: 3, width: 256 },
            Const::new(Tensor::<bf16>::new(4)),
            Const::new(0.125),
            Const::new(512),
            Const::new(64),
            Const::new(5),
        Const::new(4096), Const::new(4096))
        .unwrap();
        let call = one(&seen);
        assert_eq!(handles(&call), vec![0, 1, 2, 3, 4], "the sink is binding 4");
        assert_eq!(call.2.len() - handles(&call).len(), 10, "gqa, n, four strides, scale, window and two row pitches");

        let seen = Seen::default();
        seen.rows.set(3);
        seen.q_row_stride.set(256);
        seen.o_row_stride.set(256);
        sdpa_vector_decode_swa(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 3, width: 256 },
            Out { ptr: Tensor::<bf16>::new(3), rows: 3, width: 256 },
            Const::new(0.125),
            Const::new(512),
            Const::new(256),
            Const::new(5),
        Const::new(4096), Const::new(4096))
        .unwrap();
        let swa = one(&seen);
        assert_eq!(handles(&swa), vec![0, 1, 2, 3], "and the windowed form has none");
        assert_eq!(swa.2.len() - handles(&swa).len(), 10, "with the same scalars");
    }

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
        seen.rows.set(3);
        assert_eq!(
            sdpa_vector_decode(
                &seen,
                In { ptr: Tensor::<bf16>::new(0), rows: 3, width: 17 },
                Out { ptr: Tensor::<bf16>::new(3), rows: 3, width: 17 },
                Const::new(0.125),
                Const::new(512),
                Const::new(5),
            ),
            Err(Refusal::Narrow {
                what: "the head width",
                at: 512
            }),
            "512 is paged-only: gemma-4's wide layers do not take this path"
        );
    }

    #[test]
    fn no_head_width_reaches_a_page_shape_tail() {
        for width in PAGED_DIMS {
            let seen = Seen::default();
            decode(&seen, width, 5, 3).unwrap();
            let name = one(&seen).0;
            assert_eq!(name, format!("sdpa_paged_decode_bfloat16_d_{width}"));
            assert!(
                !name.contains("_p32"),
                "`{name}` is a page-shape tail, and one of the three is a bare name that would launch the ordinary workgroup"
            );
        }
    }

    #[test]
    fn an_empty_head_count_or_row_count_is_refused() {
        assert_eq!(
            decode(&Seen::default(), 64, 0, 3),
            Err(Refusal::Empty { what: "query heads" })
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

