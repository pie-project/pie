//! Attention, and the KV writes that feed it.
//!
//! The head dim is an axis everywhere here and its points are the checkpoint
//! geometries the tree is actually compiled for -- 64 (llama-3.2, gpt-oss),
//! 128 (llama, qwen), 256 (qwen3.5), 512 (gemma4 full-attn). A checkpoint
//! whose width is not a point of the axis has no pipeline, which used to be a
//! runtime PSO failure naming a string and is now a fact the table states.


use kernels::Grid;
use kernels::BindMut;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, Usize, bf16, elementwise, elementwise_rows, keys};

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
        Fire::at(SPLIT_FILE, "split_qkv_bf16").apply(Grid::of(elementwise_rows(packed_width, rows)?, [GROUP_X, 1, 1])),
        &[packed.arg(), q.arg(), k.arg(), v.arg(), params],
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
        Fire::at(GATE_FILE, "gate_bfloat16").apply(Grid::of(elementwise_rows(width, rows)?, [GROUP_X, 1, 1])),
        &[attn.arg(), gate.arg(), row_stride.arg()],
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
    let lanes = head_grid(*head_dim, *q_heads, rows)?;
    ctx.fire(
        Fire::at(GATE_FILE, "q_gate_split_bfloat16").apply(Grid::of(lanes, head_group(lanes))),
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

/// One token's keys and values written into a contiguous cache.
///
/// ONE token, which is why the depth is 1 and not an argument: the contiguous
/// cache is the whole-history decode's, and a prefill uses the paged form.
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
    let lanes = head_grid(*head_dim, *heads, 1)?;
    ctx.fire(
        Fire::at(KV_FILE, "kv_append_bfloat16").apply(Grid::of(lanes, head_group(lanes))),
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
    let ring_4 = ctx.absent()?;
    let ring_6 = ctx.absent()?;
    let ring_7 = ctx.absent()?;
    let ring_8 = ctx.absent()?;
    let ring_9 = ctx.absent()?;
    let ring_11 = ctx.absent()?;
    let w_page = ctx.ask::<Tensor<u32>, keys::KvWritePage>()?;
    let w_off = ctx.ask::<Tensor<u32>, keys::KvWriteOffset>()?;
    let ring_15 = ctx.absent()?;
    let tokens = ctx.ask::<i32, keys::Rows>()?;
    let lanes = head_grid(*head_dim, *n_kv_heads, tokens)?;
    ctx.fire(
        Fire::at(KV_FILE, "kv_append_paged_bfloat16").apply(Grid::of(lanes, head_group(lanes))),
        &[
            k_new.arg(),
            v_new.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            ring_4,
            head_dim.arg(),
            ring_6,
            ring_7,
            ring_8,
            ring_9,
            page_size.arg(),
            ring_11,
            n_kv_heads.arg(),
            w_page.arg(),
            w_off.arg(),
            ring_15,
        ],
    )
}

/// gemma's `tanh` cap on the logits, elementwise over the vocabulary.
///
/// # Errors
///
/// See [`elementwise`](crate::routine::elementwise).
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
        Fire::at(SOFTCAP_FILE, "logit_softcap_bfloat16").apply(Grid::of(elementwise(n, 1)?, [GROUP_X, 1, 1])),
        &[logits.arg(), out.arg(), params],
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
    let sinks = ctx.absent()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(PAGED_FILE, PAGED_DECODE[head_point(*head_dim, &PAGED_DIMS)?]).apply(Grid::of(vector_grid(*q_heads, rows)?, BIG_GROUP)),
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

/// [`sdpa_paged_decode`] with the sink logit in the softmax.
///
/// ONE head width, and that is the checkpoint's rather than a gap: gpt-oss is
/// the only model in the tree with sinks and its heads are 64 wide.
///
/// # Errors
///
/// See [`sdpa_paged_decode`].
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
    ctx.fire(
        Fire::at(PAGED_FILE, PAGED_DECODE_SINK[head_point(*head_dim, &SINK_DIMS)?]).apply(Grid::of(vector_grid(*q_heads, rows)?, BIG_GROUP)),
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
    let sinks = ctx.absent()?;
    let n_rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(PAGED_FILE, PAGED_TILED[head_point(*head_dim, &PAGED_DIMS)?]).apply(Grid::of(tiled_grid(*q_heads, n_rows, BIG_GROUP[0])?, BIG_GROUP)),
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

/// [`sdpa_paged_tiled`] with the sink logit in the softmax.
///
/// # Errors
///
/// See [`sdpa_paged_tiled`].
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
    ctx.fire(
        Fire::at(PAGED_FILE, PAGED_TILED_SINK[head_point(*head_dim, &SINK_DIMS)?]).apply(Grid::of(tiled_grid(*q_heads, n_rows, BIG_GROUP[0])?, BIG_GROUP)),
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

/// [`sdpa_paged_tiled`] over a query and an output that are not packed.
///
/// The two pitches are separate numbers: a fused QKV projection leaves q at
/// the packed width's stride while the attention output is its own tensor.
///
/// # Errors
///
/// See [`sdpa_paged_tiled`].
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
    let sinks = ctx.absent()?;
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
    ctx.fire(
        Fire::at(PAGED_FILE, PAGED_TILED_STRIDED[head_point(*head_dim, &STRIDED_DIMS)?]).apply(Grid::of(tiled_grid(*q_heads, n_rows, BIG_GROUP[0])?, BIG_GROUP)),
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
    let sinks = ctx.absent()?;
    let n_rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(MMA_FILE, PAGED_MMA[head_point(*head_dim, &SINK_DIMS)?]).apply(Grid::of(tiled_grid(*q_heads, n_rows, MMA_GROUP[0])?, MMA_GROUP)),
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

/// [`sdpa_paged_mma`] with the sink logit in the softmax.
///
/// # Errors
///
/// See [`sdpa_paged_mma`].
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
    ctx.fire(
        Fire::at(MMA_FILE, PAGED_MMA_SINK[head_point(*head_dim, &SINK_DIMS)?]).apply(Grid::of(tiled_grid(*q_heads, n_rows, MMA_GROUP[0])?, MMA_GROUP)),
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
        Fire::at(VECTOR_FILE, VECTOR_DECODE[head_point(*head_dim, &VECTOR_DIMS)?]).apply(Grid::of(vector_grid(*q_heads, rows)?, BIG_GROUP)),
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
        Fire::at(SLIDING_FILE, VECTOR_SWA[head_point(*head_dim, &SWA_DIMS)?]).apply(Grid::of(vector_grid(*q_heads, rows)?, BIG_GROUP)),
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
#[routine]
pub fn sdpa_vector_decode_sink(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    scale: Const<f32>,
    window: Const<i32>,
    sinks: Const<Tensor<bf16>>,
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
        Fire::at(SLIDING_FILE, VECTOR_SINK[head_point(*head_dim, &SINK_DIMS)?]).apply(Grid::of(vector_grid(*q_heads, rows)?, BIG_GROUP)),
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
            sinks.arg(),
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do, and answers the
    /// facts this file's TESTED bodies ask for: the KV-cache pair and its
    /// four strides, the paged pool's index/mask/page-size family, the write
    /// pair a paged append binds, the two strided pitches
    /// `sdpa_paged_tiled_strided` adds, and `Rows` -- which every one of them
    /// asks under that one name whether the caller thinks of it as rows, a
    /// token count, or a tile count. `gate`'s `row_stride` is separate from
    /// all of that. None of `split_qkv_bf16`, `q_gate_split`,
    /// `logit_softcap` or the three `_sink` forms is fired by a test in this
    /// module, so their own facts (`params`, `QgRowStride`, `OutRowStride`,
    /// `Elements`) have no Cell here -- adding one nothing exercises would be
    /// a guess this probe cannot stand behind.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        page_size: Cell<i32>,
        k_pages: Cell<u32>,
        v_pages: Cell<u32>,
        gqa_factor: Cell<i32>,
        positions: Cell<u32>,
        request_of_token: Cell<u32>,
        kv_page_indices: Cell<u32>,
        kv_page_indptr: Cell<u32>,
        attention_mask: Cell<u32>,
        attention_mask_stride: Cell<u32>,
        attention_mask_enabled: Cell<u32>,
        rows: Cell<i32>,
        row_stride: Cell<i32>,
        kv_head_stride: Cell<u32>,
        kv_seq_stride: Cell<u32>,
        kv_write_page: Cell<u32>,
        kv_write_offset: Cell<u32>,
        q_row_pitch: Cell<i32>,
        o_row_pitch: Cell<i32>,
        q_row_stride: Cell<i32>,
        o_row_stride: Cell<i32>,
        absent_handle: Cell<u32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                page_size: Cell::new(16),
                k_pages: Cell::new(700),
                v_pages: Cell::new(701),
                gqa_factor: Cell::new(2),
                positions: Cell::new(702),
                request_of_token: Cell::new(703),
                kv_page_indices: Cell::new(704),
                kv_page_indptr: Cell::new(705),
                attention_mask: Cell::new(706),
                attention_mask_stride: Cell::new(0),
                attention_mask_enabled: Cell::new(707),
                rows: Cell::new(4),
                row_stride: Cell::new(4096),
                kv_head_stride: Cell::new(128),
                kv_seq_stride: Cell::new(64),
                kv_write_page: Cell::new(11),
                kv_write_offset: Cell::new(12),
                q_row_pitch: Cell::new(512),
                o_row_pitch: Cell::new(256),
                q_row_stride: Cell::new(512),
                o_row_stride: Cell::new(256),
                absent_handle: Cell::new(99),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(&self, _ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            // THE STATEMENT'S OWN SCALARS, read by index where the params run
            // is the shader's struct and no `Const` mark can name a word in it
            // -- see `Asks::param`. A stride wide enough for the rows these
            // cases build.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                let _ = n;
                return Ok(ArgValue::I32(4096));
            }
            if source == <keys::KvPageSize as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.page_size.get()));
            }
            if source == <keys::KvKeys as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.k_pages.get()));
            }
            if source == <keys::KvValues as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.v_pages.get()));
            }
            // THE CACHE'S HEAD COUNT, which the paged bodies derive
            // `gqa_factor` from now rather than being told it. The probe's
            // own `gqa_factor` cell stays: `sdpa_vector_*` still asks.
            if source == <keys::NumKvHeads as Fact>::SOURCE {
                return Ok(ArgValue::I32(8));
            }
            if source == <keys::GqaFactor as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.gqa_factor.get()));
            }
            if source == <keys::Positions as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.positions.get()));
            }
            if source == <keys::RequestOfToken as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.request_of_token.get()));
            }
            if source == <keys::KvPageIndices as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.kv_page_indices.get()));
            }
            if source == <keys::KvPageIndptr as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.kv_page_indptr.get()));
            }
            if source == <keys::AttentionMask as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.attention_mask.get()));
            }
            if source == <keys::AttentionMaskStride as Fact>::SOURCE {
                return Ok(ArgValue::U32(self.attention_mask_stride.get()));
            }
            if source == <keys::AttentionMaskEnabled as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.attention_mask_enabled.get()));
            }
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            if source == <keys::KvHeadStride as Fact>::SOURCE {
                return Ok(ArgValue::Usize(u64::from(self.kv_head_stride.get())));
            }
            if source == <keys::KvSeqStride as Fact>::SOURCE {
                return Ok(ArgValue::Usize(u64::from(self.kv_seq_stride.get())));
            }
            if source == <keys::KvWritePage as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.kv_write_page.get()));
            }
            if source == <keys::KvWriteOffset as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.kv_write_offset.get()));
            }
            if source == <keys::QRowStride as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.q_row_stride.get()));
            }
            if source == <keys::ORowStride as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.o_row_stride.get()));
            }
            if source == kernels::Source::Lit(kernels::Lit::Null) {
                return Ok(ArgValue::Buffer(self.absent_handle.get()));
            }
            // Anything else is refused: a probe that invented an answer to a
            // fact it does not know would let a body pass under test while
            // the same fact went unanswered on a real driver.
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls.borrow_mut().push((fire, args.to_vec()));
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
        seen.rows.set(512);
        gate(
            &seen,
            InOut { ptr: Tensor::<bf16>::new(1), rows: 0, width: 4096 },
            In::new(Tensor::<bf16>::new(2)),
            // The statement's stride, which the probe used to answer.
            Const::new(4096))
        .expect("a launch");
        let calls = seen.calls.borrow();
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
        let ring = Tensor::<bf16>::new(99);
        kv_append_paged(
            &seen,
            In::new(Tensor::<bf16>::new(1)),
            In::new(Tensor::<bf16>::new(2)),
            Const::new(256),
            Const::new(8))
        .expect("a launch");
        let calls = seen.calls.borrow();
        let (_, args) = &calls[0];
        assert_eq!(args.len(), 16, "the pool's whole argument table");
        assert_eq!(args[13], Tensor::<u32>::new(11).arg(), "the destination page at thirteen");
        assert_eq!(args[14], Tensor::<u32>::new(12).arg(), "and its offset at fourteen");
        for slot in [4, 6, 7, 8, 9, 11, 15] {
            assert_eq!(args[slot], ring.arg(), "slot {slot} still holds an address");
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
        seen.rows.set(100);
        sdpa_paged_tiled(
            &seen,
            In::new(Tensor::<bf16>::new(1)),
            Out::new(Tensor::<bf16>::new(4)),
            Const::new(8),
            Const::new(0.125),
            Const::new(0),
            Const::new(64),
            Const::new(32))
        .expect("a launch");
        let calls = seen.calls.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(
            fire.lanes[1], 4,
            "a hundred rows is four tiles of thirty-two, the last one partial"
        );
        assert_eq!(args[17], 100.arg(), "and the kernel is told the hundred");
    }

    /// The matrix-unit tiling launches the 128 threads it was compiled for.
    ///
    /// `max_total_threads_per_threadgroup(128)` is on the shader, so this is
    /// not a tuning choice: a 1024-wide dispatch of it is refused by the
    /// device at pipeline creation.
    #[test]
    fn the_matrix_unit_tiling_launches_its_declared_width() {
        let seen = Seen::default();
        seen.rows.set(64);
        sdpa_paged_mma(
            &seen,
            In::new(Tensor::<bf16>::new(1)),
            Out::new(Tensor::<bf16>::new(4)),
            Const::new(8),
            Const::new(0.125),
            Const::new(0),
            Const::new(64),
            Const::new(32))
        .expect("a launch");
        let calls = seen.calls.borrow();
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
        let (b, m, _i, _u, _u8s) = (Tensor::<bf16>::new(1), Tensor::<bf16>::new(2), Tensor::<i32>::new(3), Tensor::<u32>::new(4), Tensor::<u8>::new(5));
        sdpa_paged_decode(
            &seen,
            In::new(b),
            Out::new(m),
            Const::new(8),
            Const::new(0.125),
            Const::new(0),
            Const::new(64),
            Const::new(32))
        .expect("a launch");
        sdpa_paged_tiled(
            &seen,
            In::new(b),
            Out::new(m),
            Const::new(8),
            Const::new(0.125),
            Const::new(0),
            Const::new(64),
            Const::new(32))
        .expect("a launch");
        sdpa_paged_tiled_strided(
            &seen,
            In::new(b),
            Out::new(m),
            Const::new(8),
            Const::new(0.125),
            Const::new(0),
            Const::new(256),
            Const::new(32))
        .expect("a launch");
        sdpa_paged_mma(
            &seen,
            In::new(b),
            Out::new(m),
            Const::new(8),
            Const::new(0.125),
            Const::new(0),
            Const::new(64),
            Const::new(32))
        .expect("a launch");
        sdpa_vector_decode(
            &seen,
            In::new(b),
            Out::new(m),
            Const::new(0.125),
            Const::new(64),
            Const::new(32))
        .expect("a launch");
        sdpa_vector_decode_swa(
            &seen,
            In::new(b),
            Out::new(m),
            Const::new(0.125),
            Const::new(512),
            Const::new(256),
            Const::new(32), Const::new(4096), Const::new(4096))
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
        let calls = seen.calls.borrow();
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
