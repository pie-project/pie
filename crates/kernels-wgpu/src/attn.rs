//! Attention, and the KV writes that feed it.
//!
//! The head dim is an axis everywhere here and its points are the checkpoint
//! geometries the tree is actually compiled for -- 64 (llama-3.2, gpt-oss),
//! 128 (llama, qwen), 256 (qwen3.5), 512 (gemma4 full-attn). A checkpoint
//! whose width is not a point of the axis has no pipeline, which used to be a
//! runtime PSO failure naming a string and is now a fact the table states.

use kernels_macros::routine;
use kernels::BindMut;

// ── R4, THE NOTATION NO CENSUS IN THIS REFACTOR HAS MEASURED ──
//
// This table and `kernels-vulkan/src/attn.rs` are mirrors, operand for
// operand, and between the two crates they author `kernels::Source` values
// 637 times through `kernels::operands!`'s `$src:expr` -- ten times the whole
// `#[source(..)]` surface in `kernels-cuda` that every census in this refactor
// was counting. The argument for the thirty conversions in this file, the
// reason `.wiki/kilimanjaro2.md` §3.11 is unsafe here until they happen, the
// count that came in at 88 and is 60, and the type evidence for all seven
// keys are written ONCE, in the vulkan mirror's header. It is not repeated
// here because a mirror that argues with itself in two places drifts.
//
// The two paragraphs below that name `keys::AttentionMaskStride` in prose are
// the entire difference between a naive census of this file and a true one,
// and they are left standing: the sentence they carry is load-bearing and the
// miscount is the instrument's problem, not theirs.
//
// What those paragraphs say has changed. They called the mask's pitch this
// table's DIVERGENCE from `kernels-metal`; it is now the agreement. All three
// planes ask the fire, and metal's fire answers zero because metal stages no
// mask -- which is a fact about that driver, stated in that driver, rather
// than a literal frozen into six signature rows.

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, Usize, bf16, keys};
use kernels::routine::Refusal;
use kernels::shader::{elementwise, elementwise_rows};

/// The head widths `sdpa_paged_*` is built for.
const PAGED_DIMS: [i32; 4] = [64, 128, 256, 512];

/// The head widths `sdpa_vector_decode` is built for.
const VECTOR_DIMS: [i32; 3] = [64, 128, 256];

/// The head widths the sliding-window decode is built for.
const SWA_DIMS: [i32; 2] = [256, 512];

/// `sdpa_paged_decode`, by head width.
///
/// Four entries where the row beside it states SEVEN points. The other three
/// -- `_d_64_p32`, `_d_128_p32`, `_d_64_p32_sg8` -- are deliberately not
/// reachable from here, and `kernels-vulkan`'s
/// `the_page_shape_tails_are_one_real_variant_and_one_bare_name` is where the
/// measurement lives, and this crate's
/// `the_entrypoints_that_ignore_the_window_are_exactly_the_ones_named_p32` is
/// the same claim asked of this tree. In short: `_p32` sets `PIE_FAST_FULL`, which pins the
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

/// The y extent of a tiled attention workgroup, from `@workgroup_size(32, 8)`.
///
/// Separate from `TILE` because they are two different numbers that happen
/// to be one on the sibling: the tile is 32 ROWS of work and the workgroup is
/// eight lanes tall, each sweeping four of them.
const TILE_LANES: u32 = 8;

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
    // ONE LANE PER PAIR, and this is where `kernels-vulkan`'s arithmetic does
    // NOT transfer. Its shaders run one lane per channel and it multiplies by
    // `head_dim`; every wgpu decode shader here is
    // `@workgroup_size(PIE_PAIRS)` with `PIE_PAIRS = head_dim / 2`, because a
    // lane packs two bf16 channels into a word. `driver-wgpu::geometry`'s
    // `Rule::SdpaVector` says the same thing as
    // `module.local.at(0) * dims.q_heads`, and refuses outright when the
    // module's width and the fire's head width disagree.
    //
    // Copying vulkan's line would ask for twice the lanes: the second half of
    // every head would run with `d_out` past the row and, because the bodies
    // guard on `arrayLength`, write nothing and report success.
    if head_dim % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "the head width is not a whole number of bf16 pairs",
            at: i64::from(head_dim),
        });
    }
    let x = q_heads
        .unsigned_abs()
        .checked_mul(head_dim.unsigned_abs() / 2)
        .ok_or(Refusal::Grid {
            what: "query heads * the head width in pairs",
            at: i64::from(q_heads) * i64::from(head_dim) / 2,
        })?;
    Ok([x, rows.unsigned_abs(), 1])
}

/// The paged decode shape: [`vector_grid`], with the key block on y.
///
/// `sdpa_paged.wgsl`'s decode arm is the one shader on this grid whose
/// workgroup is not flat. It spends the invocations WebGPU guarantees beyond
/// the head's pairs on KEYS -- `PIE_KB = 256 / PIE_PAIRS`, so eight at a
/// 64-wide head and one at a 512-wide one -- because the arm dispatches one
/// workgroup per query head and nothing else, and a 32-head decode was
/// therefore asking a whole GPU for 1024 invocations.
///
/// The y EXTENT is multiplied by that block so the y GROUPS stay one per row,
/// which is what the body reads `workgroup_id.y` as.
/// `driver-wgpu::geometry`'s `Rule::SdpaVector` says the same thing from the
/// module's own `@workgroup_size`, and the two are cross-checked.
///
/// # Errors
///
/// Whatever [`vector_grid`] refuses, and [`Refusal::Grid`] if the y product
/// does not fit a `u32`.
fn paged_decode_grid(head_dim: i32, q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let g = vector_grid(head_dim, q_heads, rows)?;
    // 512 / head_dim is 256 / (head_dim / 2), and `vector_grid` has already
    // refused an odd head width, so the pairs divide 256 exactly for every
    // instantiated point.
    let keys = (512 / head_dim.unsigned_abs()).max(1);
    let y = g[1].checked_mul(keys).ok_or(Refusal::Grid {
        what: "rows * the decode key block",
        at: i64::from(g[1]) * i64::from(keys),
    })?;
    Ok([g[0], y, g[2]])
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
    // The tile is 32 ROWS and the workgroup is 32 x 8, so a whole tile is
    // `TILE_LANES` lanes on y and not `TILE`. `kernels-vulkan` multiplies by
    // its own 32 because its module is square; `driver-wgpu::geometry`'s
    // `Rule::SdpaTiled` states `module.local.at(1) * rows.div_ceil(32)`.
    let y = rows
        .unsigned_abs()
        .div_ceil(TILE)
        .checked_mul(TILE_LANES)
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

/// One packed row cut into three, at two boundaries the statement states.
///
/// `q_width` and `kv_width` used to ride in `params` -- the shader read them
/// out of a `SplitQkvParams` struct -- so this signature could not name either
/// one and this paragraph said so. They are MARKS now, which is the same two
/// words of the same statement run reached by index instead of by field, and
/// the routine names them in the order the struct laid them out.
///
/// What the extent still comes from is `packed_width`, taken off the operand
/// rather than summed from the pair: it is `q_width + 2 * kv_width` and it is
/// what the grid needs. The shader recomputes that sum from the marks and
/// guards on it, so a `packed_width` that disagreed would leave a tail of the
/// row uncopied rather than write out of bounds. The two are not checked
/// against each other here, because the operand's width is the rectangle the
/// arena allocated and the marks are what the text said -- a disagreement is a
/// fact about the plan, not about this fire.
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
    v: Out<Tensor<bf16>>,
    // THE TWO BOUNDARIES, WHICH WERE `SplitQkvParams`'s two fields. They rode
    // a staged block because the row named `params: Buf`, and the cost was
    // that no signature could name either width -- so this doc said, truly,
    // that it could not check them. Both are `Const<u32>` marks now, at words
    // 0 and 1 of the same statement run the struct was staged from, IN THE
    // STRUCT'S ORDER because it is the statement's: `q_width` first and
    // `kv_width` second. Swapping the two would cut both boundaries inside a
    // neighbouring projection rather than refuse, which is why the order is
    // written down here rather than left to the reader to infer from the
    // shader.
    q_width: Const<u32>,
    kv_width: Const<u32>) -> Result<(), Refusal> {
    let packed_width = packed.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("attn/split_qkv.wgsl", "split_qkv_bf16").apply(elementwise_rows(packed_width, rows)?),
        &[packed.arg(), q.arg(), k.arg(), v.arg(), q_width.arg(), kv_width.arg()],
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
        Fire::at("attn/gate.wgsl", "gate_bfloat16").apply(elementwise_rows(width, rows)?),
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
/// Whatever `head_grid` refuses, with the head COUNT on the y axis rather
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
        Fire::at("attn/gate.wgsl", "q_gate_split_bfloat16").apply(head_grid(*head_dim, *q_heads, rows)?),
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
/// See `head_grid`.
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
        Fire::at("attn/kv_write.wgsl", "kv_append_bfloat16").apply(head_grid(*head_dim, *heads, 1)?),
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
/// See `head_grid`.
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
        Fire::at("attn/kv_write.wgsl", "kv_append_paged_bfloat16").apply(head_grid(*head_dim, *n_kv_heads, tokens)?),
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
/// The cap is the only scalar this kernel reads, so it is a MARK rather than a
/// struct: `attn/logit_softcap.wgsl` takes it as the one field of its
/// `@group(1)` uniform block. It used to ride a `SoftcapParams { cap, unused }`
/// storage buffer -- MLX's layout, ported twice -- whose second word existed
/// only to hold the struct's size. Word 0 of the statement's run is the same
/// number either way; the mark reaches it by index instead of by field.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty readout.
#[routine]
pub fn logit_softcap(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    cap: Const<f32>) -> Result<(), Refusal> {
    // THE ELEMENT COUNT, DERIVED RATHER THAN ASKED. HEAD spelled it
    // `Reckoned<Times<Say<Width>, Say<Rows>>>` -- a product of two facts,
    // not a fact -- and the migration turned it into `keys::Elements`,
    // which no driver answers. Both halves are on the operand's own
    // rectangle, so the body multiplies them.
    let n = out.rows.saturating_mul(out.width);
    ctx.fire(
        Fire::at("attn/logit_softcap.wgsl", "logit_softcap_bfloat16").apply(elementwise(n, 1)?),
        &[logits.arg(), out.arg(), cap.arg()],
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
/// [`Refusal::Narrow`] for a head width off `PAGED_DIMS`, and whatever
/// `vector_grid` refuses.
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

    let sinks = ctx.absent()?;
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
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("attn/sdpa_paged.wgsl", PAGED_DECODE[head_point(*head_dim, &PAGED_DIMS)?]).apply(paged_decode_grid(*head_dim, *q_heads, rows)?),
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
            sinks,
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
/// `vector_grid` refuses.
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
    let rows = ctx.ask::<i32, keys::Rows>()?;
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at("attn/sdpa_paged.wgsl", "sdpa_paged_decode_sink_bfloat16_d_64").apply(paged_decode_grid(*head_dim, *q_heads, rows)?),
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
/// [`Refusal::Narrow`] for a head width off `PAGED_DIMS`, and whatever
/// `tiled_grid` refuses.
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

    let sinks = ctx.absent()?;
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
    ctx.fire(
        Fire::at("attn/sdpa_paged.wgsl", PAGED_TILED[head_point(*head_dim, &PAGED_DIMS)?]).apply(tiled_grid(*q_heads, n_rows)?),
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
            sinks,
            n_rows.arg(),
        ],
    )
}

/// Paged prefill with attention sinks. One point, `d_64`.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever `tiled_grid`
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
        Fire::at("attn/sdpa_paged.wgsl", "sdpa_paged_tiled_sink_bfloat16_d_64").apply(tiled_grid(*q_heads, n_rows)?),
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
/// `tiled_grid` refuses.
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

    let sinks = ctx.absent()?;
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
        Fire::at("attn/sdpa_paged.wgsl", "sdpa_paged_tiled_strided_bfloat16_d_256").apply(tiled_grid(*q_heads, n_rows)?),
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
            sinks,
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
/// [`Refusal::Narrow`] for any head width but 64, and whatever `tiled_grid`
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

    let sinks = ctx.absent()?;
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
        Fire::at("attn/sdpa_paged_mma.wgsl", "sdpa_paged_mma_bfloat16_d_64").apply(tiled_grid(*q_heads, n_rows)?),
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
            sinks,
            n_rows.arg(),
        ],
    )
}

/// The cooperative-matrix prefill with attention sinks.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever `tiled_grid`
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
        Fire::at("attn/sdpa_paged_mma.wgsl", "sdpa_paged_mma_sink_bfloat16_d_64").apply(tiled_grid(*q_heads, n_rows)?),
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
/// [`Refusal::Narrow`] for a head width off `VECTOR_DIMS`, and whatever
/// `vector_grid` refuses.
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
        Fire::at("attn/sdpa_vector.wgsl", VECTOR_DECODE[head_point(*head_dim, &VECTOR_DIMS)?]).apply(vector_grid(*head_dim, *q_heads, rows)?),
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
/// [`Refusal::Narrow`] for a head width off `SWA_DIMS`, and whatever
/// `vector_grid` refuses.
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
        Fire::at("attn/sdpa_sliding.wgsl", VECTOR_SWA[head_point(*head_dim, &SWA_DIMS)?]).apply(vector_grid(*head_dim, *q_heads, rows)?),
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
/// `vector_grid` refuses.
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
        Fire::at("attn/sdpa_sliding.wgsl", "sdpa_vector_decode_sink_bfloat16_d_64").apply(vector_grid(*head_dim, *q_heads, rows)?),
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

    /// The two grids that are NOT `kernels-vulkan`'s, pinned as numbers.
    ///
    /// This family's bodies were ported from vulkan's and its seven family
    /// tests were NOT: two of them assert `q_heads * head_dim` for the vector
    /// extent and a square tile for the tiled one, and both are wrong here.
    /// Shipping them would have been seven green tests asserting the sibling's
    /// arithmetic about this backend's shaders.
    ///
    /// What replaces them is this, plus `tests/routines.rs`, which drives all
    /// sixteen bodies over every head width their tables carry and checks each
    /// dispatch's buffer count against the parsed `naga` module.
    #[test]
    fn the_two_grids_that_are_not_vulkans_are_the_ones_the_driver_states() {
        // ONE LANE PER PAIR: every decode shader is
        // `@workgroup_size(PIE_PAIRS)` with `PIE_PAIRS = head_dim / 2`, so a
        // 64-wide head over 8 query heads is 8 * 32 lanes and not 8 * 64.
        // `driver-wgpu::geometry`'s `Rule::SdpaVector` says
        // `module.local.at(0) * dims.q_heads` for the same reason.
        assert_eq!(vector_grid(64, 8, 7).expect("a real point"), [256, 7, 1]);
        assert_eq!(vector_grid(128, 4, 1).expect("a real point"), [256, 1, 1]);

        // THE TILE IS 32 ROWS AND THE WORKGROUP IS 32 x 8. Vulkan multiplies
        // the row-tile count by its own square 32; here y is `TILE_LANES`.
        // `Rule::SdpaTiled` states `module.local.at(1) * rows.div_ceil(32)`.
        assert_eq!(tiled_grid(8, 32).expect("one whole tile"), [256, 8, 1]);
        assert_eq!(tiled_grid(8, 33).expect("two tiles"), [256, 16, 1]);
        assert_eq!(tiled_grid(8, 1).expect("a partial tile"), [256, 8, 1]);

        // And the head width has to be a whole number of pairs, because half
        // a pair is a lane that would read one channel and write two.
        assert!(matches!(vector_grid(65, 8, 7), Err(Refusal::Narrow { .. })));
    }

    /// An empty extent is refused rather than dispatched.
    #[test]
    fn an_empty_head_count_or_row_count_is_refused() {
        assert!(matches!(vector_grid(64, 0, 7), Err(Refusal::Empty { .. })));
        assert!(matches!(vector_grid(64, 8, 0), Err(Refusal::Empty { .. })));
        assert!(matches!(tiled_grid(0, 32), Err(Refusal::Empty { .. })));
        assert!(matches!(tiled_grid(8, 0), Err(Refusal::Empty { .. })));
        assert!(matches!(head_grid(0, 8, 7), Err(Refusal::Empty { .. })));
        assert!(matches!(head_grid(64, 8, 0), Err(Refusal::Empty { .. })));
    }

    /// A head width the tree does not carry is refused by NAME.
    ///
    /// The bodies index a literal spelling table with the head width, so an
    /// unknown one must not reach it: `PAGED_DIMS` carries four and a fifth
    /// would panic on the index rather than refuse.
    #[test]
    fn a_head_width_the_tree_does_not_carry_is_refused_by_name() {
        assert!(head_point(96, &PAGED_DIMS).is_err());
        assert!(head_point(512, &VECTOR_DIMS).is_err());
        assert!(head_point(64, &SWA_DIMS).is_err());
        // And the ones it does carry resolve to their own index.
        assert_eq!(head_point(64, &PAGED_DIMS).expect("carried"), 0);
        assert_eq!(head_point(512, &PAGED_DIMS).expect("carried"), 3);
        assert_eq!(head_point(256, &SWA_DIMS).expect("carried"), 0);
    }
}
