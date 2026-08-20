//! Gated DeltaNet: the recurrent state kernels and their prep.
//!
//! `gdn` is an algorithm and not a model, so it takes no model qualifier --
//! the same call the CUDA table makes for `delta_attn_kda` and `indexer_dsa`.

// The fused core binds eleven buffers and its slotted form twelve, and then
// eleven SCALARS apiece: the twelfth and thirteenth buffers were one
// `constant GdnCoreParams&`, and the routines below name its eleven fields as
// `Const` marks instead. Gathering them back into a struct would restate that
// binding order somewhere a shader cannot check, which is the thing this
// refactor removes -- `ssm/gdn_params.h` was exactly that restatement, in
// triplicate across the three shader planes, and it is deleted.

use kernels::Grid;
use kernels::BindMut;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, keys};


/// The dk lanes one simdgroup reduces over, and the x extent of every dispatch
/// in this family.
///
/// Not a tile and not a tunable. `gdn_core.metal:121` closes its l2 norm with
/// `simd_sum` over the whole simdgroup, and `gdn_prep.metal:450`'s
/// `gdn_row_sum` walks an xor tree that reaches lane 16. An x narrower than a
/// simdgroup makes both read lanes that were never dispatched -- finite, in
/// bounds and wrong -- and a wider one walks `n_per_t * dk_idx` off the end
/// of the head it was given.
/// `Dk / 32` is the per-lane run every body in both shaders computes, so 32 is
/// also what makes `n_per_t` come out right.
const SIMD: u32 = 32;

/// The dv tile one core threadgroup spans.
///
/// `device_gdn.rs:363` and `:560` both state `[32, 4, 1]`. For [`gdn_core`] it
/// is load-bearing rather than occupancy: `gdn_core.metal:112` computes the
/// whole dv-independent q/k path on the `tpit.y == 0` simdgroup and broadcasts
/// it across the tile through the barrier at `:137`, so the tile is exactly
/// how much of that kernel's redundancy the threadgroup share removes. For
/// [`gdn_core_recurrent`] there is nothing shared and the four is occupancy,
/// but it is the same number the split pair was measured at.
const CORE_DV_TILE: u32 = 4;

/// The shader the two fused cores are compiled from.
const CORE_FILE: &str = "ssm/gdn_core.metal";

/// The shader the other six are compiled from, prefill included.
const PREP_FILE: &str = "ssm/gdn_prep.metal";

/// The `(row, v-head)` axis every decode kernel here walks on z.
///
/// All three bodies open the same three lines -- `n = tpig.z`,
/// `b_idx = n / Hv`, `hv_idx = n % Hv` at `gdn_core.metal:76`,
/// `gdn_prep.metal:83` and `gdn_prep.metal:181` -- so the z extent is the
/// PRODUCT and `Hv` is whatever the packed `GdnCoreParams` says. One function
/// rather than a multiplication at seven call sites because the two numbers
/// have to be the same two: a z built from a head count the parameter block does
/// not carry splits every row at the wrong place, which lands inside the
/// activations and reads a plausible number.
///
/// # Errors
///
/// [`Refusal::Empty`] for either extent at zero. A fire with no rows arrives
/// honestly; a dispatch of no threads does not, because it runs nothing,
/// reports success and leaves `core_out` holding the previous token's answer.
/// [`Refusal::Grid`] if the product does not fit a `u32`.
fn head_rows(rows: i32, v_heads: i32) -> Result<u32, Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    let n = u64::from(rows.unsigned_abs()) * u64::from(v_heads.unsigned_abs());
    u32::try_from(n).map_err(|_| Refusal::Grid {
        what: "rows * v_heads",
        at: i64::try_from(n).unwrap_or(i64::MAX),
    })
}

/// The grid a per-dv core takes: one simdgroup per `(row, v-head, v-channel)`.
///
/// `[32, Dv, rows * Hv]`, which is what `device_gdn.rs:362` fires for the
/// fused core and `:559` for the split one.
///
/// # Errors
///
/// See [`head_rows`], plus [`Refusal::Empty`] for a `v_dim` of zero -- the y
/// axis is the only thing that covers the value channels, and a zero there is
/// a dispatch of nothing at all.
fn core_grid(rows: i32, v_heads: i32, v_dim: i32) -> Result<[u32; 3], Refusal> {
    let z = head_rows(rows, v_heads)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    Ok([SIMD, v_dim.unsigned_abs(), z])
}

/// The grid a prep takes: one simdgroup per `(row, v-head)` and NO dv axis.
///
/// `[32, 1, rows * Hv]` -- `device_gdn.rs:542`. The one on y is the whole
/// reason the split exists, so it is written here rather than left as a
/// literal beside each call.
///
/// # Errors
///
/// See [`head_rows`].
fn prep_grid(rows: i32, v_heads: i32) -> Result<[u32; 3], Refusal> {
    Ok([SIMD, 1, head_rows(rows, v_heads)?])
}

/// The threadgroup a core dispatch takes, DERIVED from the grid it was handed.
///
/// x is the grid's own x rather than a second literal 32, and that is the
/// point. [`gdn_core`] and [`gdn_core_slotted`] read their dk lane off
/// `thread_position_in_threadgroup.x` (`gdn_core.metal:85`) where every other
/// kernel in the family reads `thread_position_in_grid.x`. The two are the
/// same number only while there is ONE threadgroup on x, which is only true
/// while `group.x == grid.x`. A body that states both can state them
/// differently; a body that derives one from the other cannot.
const fn core_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], CORE_DV_TILE, 1]
}

/// The threadgroup a one-simdgroup dispatch takes: the grid's x, and one of
/// everything else.
///
/// Derived for the same reason as [`core_group`], and it matches
/// `device_gdn.rs:543`, `:1271` and `:1304`.
const fn simd_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 1, 1]
}

/// The nine `(LANES, VROWS)` points `gdn_core_recurrent_prefill` is compiled
/// for, in the census's sorted order.
///
/// A LITERAL table, the way `layout::EMBED_GATHER` is one and for the same
/// reason: `format!("..._l_{lanes}_v_{vrows}")` would spell `_l_32_v_1` as
/// readily as `_l_32_v_4`, and Metal does not find out until
/// `newFunctionWithName:` returns nil at RUN time -- inside a fire, after the
/// plan was accepted and after everything before it in the plan has already
/// been encoded. A point the tree does not carry cannot be spelled at all.
///
/// Unlike the affine axis this is not a PRODUCT: nine of the sixteen
/// `{4,8,16,32} x {1,2,4,8}` combinations exist and the missing seven are not
/// a rectangle. `(32, 1)` is the one that looks cheapest and is not compiled,
/// which is why [`scan_point`] is a match over pairs rather than two
/// independent lookups multiplied together.
static GDN_SCAN: [&str; 9] = [
    "gdn_core_recurrent_prefill_bfloat16_l_16_v_1",
    "gdn_core_recurrent_prefill_bfloat16_l_16_v_2",
    "gdn_core_recurrent_prefill_bfloat16_l_16_v_4",
    "gdn_core_recurrent_prefill_bfloat16_l_32_v_2",
    "gdn_core_recurrent_prefill_bfloat16_l_32_v_4",
    "gdn_core_recurrent_prefill_bfloat16_l_32_v_8",
    "gdn_core_recurrent_prefill_bfloat16_l_4_v_1",
    "gdn_core_recurrent_prefill_bfloat16_l_8_v_1",
    "gdn_core_recurrent_prefill_bfloat16_l_8_v_2",
];

/// The point in the scan's tiling axis a spelling exists for.
///
/// `driver-metal`'s `Tuning` defaults to `gdn_scan_lanes: 32,
/// gdn_scan_rows: 4` (`layout/tuning.rs:117`) and reads both from the
/// environment, so an unlisted pair is one `PIE_METAL_GDN_SCAN_LANES=2` away
/// and is a value that reaches here rather than a typo somebody would catch.
///
/// # Errors
///
/// [`Refusal::Narrow`] naming whichever half was not one of the nine. The lane
/// width is checked first because it is the one that changes the reduction
/// -- `gdn_row_sum`'s xor tree is written for it -- and the row count second,
/// carrying its own value so a caller can step down to a tiling that exists
/// rather than fault.
fn scan_point(lanes: i32, vrows: i32) -> Result<usize, Refusal> {
    match (lanes, vrows) {
        (16, 1) => Ok(0),
        (16, 2) => Ok(1),
        (16, 4) => Ok(2),
        (32, 2) => Ok(3),
        (32, 4) => Ok(4),
        (32, 8) => Ok(5),
        (4, 1) => Ok(6),
        (8, 1) => Ok(7),
        (8, 2) => Ok(8),
        (4 | 8 | 16 | 32, _) => Err(Refusal::Narrow {
            what: "scan rows per lane group, at this lane width",
            at: i64::from(vrows),
        }),
        _ => Err(Refusal::Narrow {
            what: "scan lane width",
            at: i64::from(lanes),
        }),
    }
}

/// The scan's grid: the dv axis FOLDED by the tiling, one head per z.
///
/// `LANES` lanes own one dv row, so `32/LANES` rows share the simdgroup and
/// each walks `VROWS` of them -- `gdn_prep.metal:489` and `:492`. What is left
/// of `Dv` is y, which is `device_gdn.rs:1264`'s `per_y` and `:1265`'s
/// `div_ceil` exactly.
///
/// It rounds UP and that is safe rather than sloppy: `gdn_prep.metal:497`
/// returns early for a `dv_base` past the end and `:501` masks the short last
/// group with `vn = min(VROWS, Dv - dv_base)`, because clamping would make two
/// lane groups own one state row and the scan is a read-modify-write.
/// Rounding DOWN would leave the tail of every head's state unscanned while
/// the rest of the prompt advanced over it.
///
/// # Errors
///
/// [`Refusal::Empty`] for a zero extent or a tiling that would not divide.
/// [`scan_point`] has already refused any tiling the tree does not carry by
/// the time a body calls this, but the division is here, so the guard is here
/// too.
fn scan_grid(v_dim: i32, v_heads: i32, lanes: i32, vrows: i32) -> Result<[u32; 3], Refusal> {
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    if lanes <= 0 || vrows <= 0 {
        return Err(Refusal::Empty {
            what: "the scan tiling",
        });
    }
    let per_y = (SIMD / lanes.unsigned_abs()) * vrows.unsigned_abs();
    if per_y == 0 {
        return Err(Refusal::Empty {
            what: "the scan tiling",
        });
    }
    Ok([
        SIMD,
        v_dim.unsigned_abs().div_ceil(per_y),
        v_heads.unsigned_abs(),
    ])
}

/// The whole gated-delta-net core in ONE dispatch: conv1d, silu, the q/k
/// l2 norm, the gates, the recurrent step and the convolution writeback.
///
/// This signature is the FIRST statement of this binding order that has ever
/// existed. The row above states no operands, `driver-metal` has no arm for
/// any `gdn_*` symbol, and `text_conformance.rs:1496` records *"no plan in
/// this workspace names a GDN symbol"* for all eight. It was read off
/// `ssm/gdn_core.metal:187`, buffer by buffer.
///
/// `conv_state` is [`Buf`] and `new_conv_state` is [`Buf`] because the two
/// cannot be one allocation: `convsilu` reads the `Kc`-tap history while the
/// writeback shifts it, and the redundant dv threadgroups interleave those
/// reads and writes. The shader's own header calls that out; the types are
/// where a caller that tried to alias them fails to compile.
///
/// The geometry is ELEVEN SCALARS at buffers 11 through 21, one `setBytes`
/// apiece, and it is where `Dk`, `Hk`, `Kc` and the three region offsets
/// arrive. It was one packed `constant GdnCoreParams&` at buffer 11 -- the
/// shape `layout::ple_combine` used to take `PleCombineParams` in before its
/// one live scalar became a `Const<f32>` mark -- and the cost was that this
/// signature could name none of them: `v_heads` and `v_dim` had to be ASKED
/// for beside the block because the grid needs both, so one number was stated
/// twice and only the caller could make the two agree. They are the same
/// marks the shader reads now, so there is one statement of each.
///
/// # Errors
///
/// See [`core_grid`].
#[routine]
pub fn gdn_core(
    ctx: &Ctx<'_>,
    mixed: In<Tensor<bf16>>,
    core_out: Out<Tensor<bf16>>,
    conv_w: Const<Tensor<bf16>>,
    conv_b: Const<Tensor<bf16>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<bf16>>,
    a_gate: In<Tensor<bf16>>,
    b_gate: In<Tensor<bf16>>,
    // THE ELEVEN, IN THE ORDER THE STATEMENT PLACES THEM. A `Const` mark's
    // slot is its POSITION among the scalar marks of this signature, so the
    // list below IS `model-dsl`'s `GdnShape::params` and a pair swapped here
    // reads its neighbour's number rather than refusing. The two swaps most
    // likely to be made are also the two that stay silent longest: `Dk` and
    // `Dv` are 128 and 128 on every GDN checkpoint the tree has seen, and
    // `Hk` and `Hv` differ only under group-query, so either transposition
    // runs and answers.
    k_dim: Const<i32>,
    v_dim: Const<i32>,
    k_heads: Const<i32>,
    v_heads: Const<i32>,
    conv_dim: Const<i32>,
    conv_k: Const<i32>,
    q_off: Const<i32>,
    k_off: Const<i32>,
    v_off: Const<i32>,
    eps: Const<f32>,
    inv_sqrt_dk: Const<f32>) -> Result<(), Refusal> {
    // THE ELEVEN GEOMETRY MARKS, WHICH WERE `GdnCoreParams`' ELEVEN FIELDS.
    //
    // A block stood here saying *"BACK TO AN ASK, BECAUSE THIS ROUTINE'S
    // PARAMS RUN IS A STRUCT"*, and it was true of the arrangement it
    // described: the body forwarded `ctx.params()` whole, the shader read the
    // run as `GdnCoreParams` rather than as a scalar run, so slot 0 was the
    // struct's FIRST FIELD and a `Const` derived onto it would have read that
    // field's bits. Nothing in this signature could name a word of the run,
    // which is why the head count and the value width came back as asks even
    // though the statement was already carrying both.
    //
    // The marks name all eleven now. They are the same eleven words of the
    // same `Lowered::params` run the struct was staged from, reached by index
    // instead of by field, and the entrypoint takes each as its own
    // `const constant&` argument -- one `setBytes` apiece -- instead of one
    // `constant GdnCoreParams&` at a single buffer.
    //
    // So `keys::VHeads` and `keys::VDim` are gone from this body: `Asks`'
    // own test for whether a number is a fact -- *"two fires of the same
    // model, on the same deployment, can see different answers here"* -- is
    // FALSE of a head count and of a head width, which are checkpoint
    // configuration and reach the shader from this same run anyway. It stays
    // TRUE of every table below, which is why every one of them is still an
    // ask: the conv and recurrent slabs and the seat map are what the fire
    // threaded from the previous step, and no text can state them.
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let grid = core_grid(rows, *v_heads, *v_dim)?;
    ctx.fire(
        Fire::at(CORE_FILE, "gdn_core_bfloat16").apply(Grid::of(grid, core_group(grid))),
        &[
            mixed.arg(),
            conv_state.arg(),
            rstate.arg_mut(),
            core_out.arg(),
            conv_w.arg(),
            conv_b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            a_gate.arg(),
            b_gate.arg(),
            new_conv_state.arg_mut(),
            k_dim.arg(),
            v_dim.arg(),
            k_heads.arg(),
            v_heads.arg(),
            conv_dim.arg(),
            conv_k.arg(),
            q_off.arg(),
            k_off.arg(),
            v_off.arg(),
            eps.arg(),
            inv_sqrt_dk.arg(),
        ],
    )
}

/// [`gdn_core`] over rows whose persistent state is not where the row is.
///
/// One buffer more, at 11 -- where the params block used to sit, since the
/// scalars now follow every buffer -- and ONLY the state accesses remap
/// through it: the conv slab and the recurrent slab take `slot_ids[b_idx]`
/// while `mixed`, `core_out`, `a_gate` and `b_gate` stay token-major
/// (`gdn_core.metal:90`). That asymmetry is the whole of the slotted seam and
/// it is why the two symbols take the same eleven buffers in the same eleven
/// slots.
///
/// A separate SYMBOL rather than this one handed a null, because `SLOTTED` is
/// a template parameter: the sealed form never reads the pointer and compiles
/// to a byte-identical pipeline, which `gdn_core.metal:185` records as *"264
/// holds"*.
///
/// `slot_ids` is declared `const device uint*`, which is the less common of
/// the two live MSL spellings [`crate::routine`]'s `SPELLING` counts -- 44
/// against 62 for `uint32_t*`. Same type, one [`U32s`].
///
/// # Errors
///
/// See [`core_grid`].
#[routine]
pub fn gdn_core_slotted(
    ctx: &Ctx<'_>,
    mixed: In<Tensor<bf16>>,
    core_out: Out<Tensor<bf16>>,
    conv_w: Const<Tensor<bf16>>,
    conv_b: Const<Tensor<bf16>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<bf16>>,
    a_gate: In<Tensor<bf16>>,
    b_gate: In<Tensor<bf16>>,
    // THE ELEVEN, IN THE ORDER THE STATEMENT PLACES THEM. A `Const` mark's
    // slot is its POSITION among the scalar marks of this signature, so the
    // list below IS `model-dsl`'s `GdnShape::params` and a pair swapped here
    // reads its neighbour's number rather than refusing. The two swaps most
    // likely to be made are also the two that stay silent longest: `Dk` and
    // `Dv` are 128 and 128 on every GDN checkpoint the tree has seen, and
    // `Hk` and `Hv` differ only under group-query, so either transposition
    // runs and answers.
    k_dim: Const<i32>,
    v_dim: Const<i32>,
    k_heads: Const<i32>,
    v_heads: Const<i32>,
    conv_dim: Const<i32>,
    conv_k: Const<i32>,
    q_off: Const<i32>,
    k_off: Const<i32>,
    v_off: Const<i32>,
    eps: Const<f32>,
    inv_sqrt_dk: Const<f32>) -> Result<(), Refusal> {
    // THE ELEVEN GEOMETRY MARKS, WHICH WERE `GdnCoreParams`' ELEVEN FIELDS.
    //
    // A block stood here saying *"BACK TO AN ASK, BECAUSE THIS ROUTINE'S
    // PARAMS RUN IS A STRUCT"*, and it was true of the arrangement it
    // described: the body forwarded `ctx.params()` whole, the shader read the
    // run as `GdnCoreParams` rather than as a scalar run, so slot 0 was the
    // struct's FIRST FIELD and a `Const` derived onto it would have read that
    // field's bits. Nothing in this signature could name a word of the run,
    // which is why the head count and the value width came back as asks even
    // though the statement was already carrying both.
    //
    // The marks name all eleven now. They are the same eleven words of the
    // same `Lowered::params` run the struct was staged from, reached by index
    // instead of by field, and the entrypoint takes each as its own
    // `const constant&` argument -- one `setBytes` apiece -- instead of one
    // `constant GdnCoreParams&` at a single buffer.
    //
    // So `keys::VHeads` and `keys::VDim` are gone from this body: `Asks`'
    // own test for whether a number is a fact -- *"two fires of the same
    // model, on the same deployment, can see different answers here"* -- is
    // FALSE of a head count and of a head width, which are checkpoint
    // configuration and reach the shader from this same run anyway. It stays
    // TRUE of every table below, which is why every one of them is still an
    // ask: the conv and recurrent slabs and the seat map are what the fire
    // threaded from the previous step, and no text can state them.
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let grid = core_grid(rows, *v_heads, *v_dim)?;
    ctx.fire(
        Fire::at(CORE_FILE, "gdn_core_slotted_bfloat16").apply(Grid::of(grid, core_group(grid))),
        &[
            mixed.arg(),
            conv_state.arg(),
            rstate.arg_mut(),
            core_out.arg(),
            conv_w.arg(),
            conv_b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            a_gate.arg(),
            b_gate.arg(),
            new_conv_state.arg_mut(),
            slot_ids.arg(),
            k_dim.arg(),
            v_dim.arg(),
            k_heads.arg(),
            v_heads.arg(),
            conv_dim.arg(),
            conv_k.arg(),
            q_off.arg(),
            k_off.arg(),
            v_off.arg(),
            eps.arg(),
            inv_sqrt_dk.arg(),
        ],
    )
}

/// The dv-INDEPENDENT half of the core, computed exactly once per
/// `(row, v-head)`.
///
/// The grid has no dv axis -- `[32, 1, rows * Hv]` -- and that is the entire
/// reason this kernel exists. [`gdn_core`] recomputes the q/k path once per dv
/// TILE, which its threadgroup share caps at four rather than removing; a y of
/// `v_dim` here would put the redundancy back AND have `v_dim` threadgroups
/// write the same `pre_q`, `pre_k` and q/k `new_conv_state` channels over each
/// other.
///
/// The order is not [`gdn_core`]'s with three buffers dropped. `conv_w` and
/// `conv_b` move up to 2 and 3, the three scratch outputs take 8, 9 and 10,
/// and there is no `rstate` and no `core_out` at all -- this pass touches
/// neither. Read off `ssm/gdn_prep.metal:153`.
///
/// # Errors
///
/// See [`head_rows`].
#[routine]
pub fn gdn_prep(
    ctx: &Ctx<'_>,
    mixed: In<Tensor<bf16>>,
    conv_w: Const<Tensor<bf16>>,
    conv_b: Const<Tensor<bf16>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<bf16>>,
    a_gate: In<Tensor<bf16>>,
    b_gate: In<Tensor<bf16>>,
    pre_q: Out<Tensor<f32>>,
    pre_k: Out<Tensor<f32>>,
    pre_gate: Out<Tensor<f32>>,
    // THE ELEVEN, IN THE ORDER THE STATEMENT PLACES THEM. A `Const` mark's
    // slot is its POSITION among the scalar marks of this signature, so the
    // list below IS `model-dsl`'s `GdnShape::params` and a pair swapped here
    // reads its neighbour's number rather than refusing. The two swaps most
    // likely to be made are also the two that stay silent longest: `Dk` and
    // `Dv` are 128 and 128 on every GDN checkpoint the tree has seen, and
    // `Hk` and `Hv` differ only under group-query, so either transposition
    // runs and answers.
    k_dim: Const<i32>,
    v_dim: Const<i32>,
    k_heads: Const<i32>,
    v_heads: Const<i32>,
    conv_dim: Const<i32>,
    conv_k: Const<i32>,
    q_off: Const<i32>,
    k_off: Const<i32>,
    v_off: Const<i32>,
    eps: Const<f32>,
    inv_sqrt_dk: Const<f32>) -> Result<(), Refusal> {
    // THE ELEVEN GEOMETRY MARKS, WHICH WERE `GdnCoreParams`' ELEVEN FIELDS.
    //
    // A block stood here saying *"BACK TO AN ASK, BECAUSE THIS ROUTINE'S
    // PARAMS RUN IS A STRUCT"*, and it was true of the arrangement it
    // described: the body forwarded `ctx.params()` whole, the shader read the
    // run as `GdnCoreParams` rather than as a scalar run, so slot 0 was the
    // struct's FIRST FIELD and a `Const` derived onto it would have read that
    // field's bits. Nothing in this signature could name a word of the run,
    // which is why the head count and the value width came back as asks even
    // though the statement was already carrying both.
    //
    // The marks name all eleven now. They are the same eleven words of the
    // same `Lowered::params` run the struct was staged from, reached by index
    // instead of by field, and the entrypoint takes each as its own
    // `const constant&` argument -- one `setBytes` apiece -- instead of one
    // `constant GdnCoreParams&` at a single buffer.
    //
    // So `keys::VHeads` and `keys::VDim` are gone from this body: `Asks`'
    // own test for whether a number is a fact -- *"two fires of the same
    // model, on the same deployment, can see different answers here"* -- is
    // FALSE of a head count and of a head width, which are checkpoint
    // configuration and reach the shader from this same run anyway. It stays
    // TRUE of every table below, which is why every one of them is still an
    // ask: the conv and recurrent slabs and the seat map are what the fire
    // threaded from the previous step, and no text can state them.
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let grid = prep_grid(rows, *v_heads)?;
    ctx.fire(
        Fire::at(PREP_FILE, "gdn_prep_bfloat16").apply(Grid::of(grid, simd_group(grid))),
        &[
            mixed.arg(),
            conv_state.arg(),
            conv_w.arg(),
            conv_b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            a_gate.arg(),
            b_gate.arg(),
            pre_q.arg(),
            pre_k.arg(),
            pre_gate.arg(),
            new_conv_state.arg_mut(),
            k_dim.arg(),
            v_dim.arg(),
            k_heads.arg(),
            v_heads.arg(),
            conv_dim.arg(),
            conv_k.arg(),
            q_off.arg(),
            k_off.arg(),
            v_off.arg(),
            eps.arg(),
            inv_sqrt_dk.arg(),
        ],
    )
}

/// [`gdn_prep`] with the state slabs reached through a slot map.
///
/// `slot_ids` lands at 12 here and at 11 in [`gdn_core_slotted`] and 10 in
/// [`gdn_core_recurrent_slotted`]: one buffer, three slots, because each
/// entrypoint numbers its own list. That is exactly the kind of fact a shared
/// binding struct would have had to get right three times -- and each of the
/// three is one lower than it was, because the `GdnCoreParams` pointer that
/// stood in front of it is eleven scalars that come after every buffer.
///
/// # Errors
///
/// See [`head_rows`].
#[routine]
pub fn gdn_prep_slotted(
    ctx: &Ctx<'_>,
    mixed: In<Tensor<bf16>>,
    conv_w: Const<Tensor<bf16>>,
    conv_b: Const<Tensor<bf16>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<bf16>>,
    a_gate: In<Tensor<bf16>>,
    b_gate: In<Tensor<bf16>>,
    pre_q: Out<Tensor<f32>>,
    pre_k: Out<Tensor<f32>>,
    pre_gate: Out<Tensor<f32>>,
    // THE ELEVEN, IN THE ORDER THE STATEMENT PLACES THEM. A `Const` mark's
    // slot is its POSITION among the scalar marks of this signature, so the
    // list below IS `model-dsl`'s `GdnShape::params` and a pair swapped here
    // reads its neighbour's number rather than refusing. The two swaps most
    // likely to be made are also the two that stay silent longest: `Dk` and
    // `Dv` are 128 and 128 on every GDN checkpoint the tree has seen, and
    // `Hk` and `Hv` differ only under group-query, so either transposition
    // runs and answers.
    k_dim: Const<i32>,
    v_dim: Const<i32>,
    k_heads: Const<i32>,
    v_heads: Const<i32>,
    conv_dim: Const<i32>,
    conv_k: Const<i32>,
    q_off: Const<i32>,
    k_off: Const<i32>,
    v_off: Const<i32>,
    eps: Const<f32>,
    inv_sqrt_dk: Const<f32>) -> Result<(), Refusal> {
    // THE ELEVEN GEOMETRY MARKS, WHICH WERE `GdnCoreParams`' ELEVEN FIELDS.
    //
    // A block stood here saying *"BACK TO AN ASK, BECAUSE THIS ROUTINE'S
    // PARAMS RUN IS A STRUCT"*, and it was true of the arrangement it
    // described: the body forwarded `ctx.params()` whole, the shader read the
    // run as `GdnCoreParams` rather than as a scalar run, so slot 0 was the
    // struct's FIRST FIELD and a `Const` derived onto it would have read that
    // field's bits. Nothing in this signature could name a word of the run,
    // which is why the head count and the value width came back as asks even
    // though the statement was already carrying both.
    //
    // The marks name all eleven now. They are the same eleven words of the
    // same `Lowered::params` run the struct was staged from, reached by index
    // instead of by field, and the entrypoint takes each as its own
    // `const constant&` argument -- one `setBytes` apiece -- instead of one
    // `constant GdnCoreParams&` at a single buffer.
    //
    // So `keys::VHeads` and `keys::VDim` are gone from this body: `Asks`'
    // own test for whether a number is a fact -- *"two fires of the same
    // model, on the same deployment, can see different answers here"* -- is
    // FALSE of a head count and of a head width, which are checkpoint
    // configuration and reach the shader from this same run anyway. It stays
    // TRUE of every table below, which is why every one of them is still an
    // ask: the conv and recurrent slabs and the seat map are what the fire
    // threaded from the previous step, and no text can state them.
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let grid = prep_grid(rows, *v_heads)?;
    ctx.fire(
        Fire::at(PREP_FILE, "gdn_prep_slotted_bfloat16").apply(Grid::of(grid, simd_group(grid))),
        &[
            mixed.arg(),
            conv_state.arg(),
            conv_w.arg(),
            conv_b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            a_gate.arg(),
            b_gate.arg(),
            pre_q.arg(),
            pre_k.arg(),
            pre_gate.arg(),
            new_conv_state.arg_mut(),
            slot_ids.arg(),
            k_dim.arg(),
            v_dim.arg(),
            k_heads.arg(),
            v_heads.arg(),
            conv_dim.arg(),
            conv_k.arg(),
            q_off.arg(),
            k_off.arg(),
            v_off.arg(),
            eps.arg(),
            inv_sqrt_dk.arg(),
        ],
    )
}

/// The per-dv half: the recurrent read-modify-write, with q, k and the gates
/// read BACK from [`gdn_prep`]'s scratch instead of recomputed.
///
/// The same grid as [`gdn_core`] -- `[32, Dv, rows * Hv]`, `device_gdn.rs:559`
/// -- over ten buffers instead of eleven. `A_log`, `dt_bias`, `a_gate` and
/// `b_gate` are gone, `pre_q`, `pre_k` and `pre_gate` take 6, 7 and 8, and
/// `new_conv_state` moves from 10 to 9. The first six slots agree with
/// [`gdn_core`]'s and nothing after them does, which is why the two cannot
/// share an argument list however alike they look -- and the eleven geometry
/// scalars start at 10 here where they start at 11 there, for the same
/// reason.
///
/// The scratch is [`Buf`] here and [`Buf`] in the prep, so its direction is
/// in the type rather than in a comment.
///
/// The pair is ORDERED and no signature can say so: this reads what the prep
/// wrote, the edge is the driver's, and `device_gdn.rs`'s
/// `the_split_gdn_pair_is_the_fused_kernel_to_the_bit` fails when the two are
/// encoded the other way round or when the prep is dropped.
///
/// # Errors
///
/// See [`core_grid`].
#[routine]
pub fn gdn_core_recurrent(
    ctx: &Ctx<'_>,
    mixed: In<Tensor<bf16>>,
    core_out: Out<Tensor<bf16>>,
    conv_w: Const<Tensor<bf16>>,
    conv_b: Const<Tensor<bf16>>,
    pre_q: In<Tensor<f32>>,
    pre_k: In<Tensor<f32>>,
    pre_gate: In<Tensor<f32>>,
    // THE ELEVEN, IN THE ORDER THE STATEMENT PLACES THEM. A `Const` mark's
    // slot is its POSITION among the scalar marks of this signature, so the
    // list below IS `model-dsl`'s `GdnShape::params` and a pair swapped here
    // reads its neighbour's number rather than refusing. The two swaps most
    // likely to be made are also the two that stay silent longest: `Dk` and
    // `Dv` are 128 and 128 on every GDN checkpoint the tree has seen, and
    // `Hk` and `Hv` differ only under group-query, so either transposition
    // runs and answers.
    k_dim: Const<i32>,
    v_dim: Const<i32>,
    k_heads: Const<i32>,
    v_heads: Const<i32>,
    conv_dim: Const<i32>,
    conv_k: Const<i32>,
    q_off: Const<i32>,
    k_off: Const<i32>,
    v_off: Const<i32>,
    eps: Const<f32>,
    inv_sqrt_dk: Const<f32>) -> Result<(), Refusal> {
    // THE ELEVEN GEOMETRY MARKS, WHICH WERE `GdnCoreParams`' ELEVEN FIELDS.
    //
    // A block stood here saying *"BACK TO AN ASK, BECAUSE THIS ROUTINE'S
    // PARAMS RUN IS A STRUCT"*, and it was true of the arrangement it
    // described: the body forwarded `ctx.params()` whole, the shader read the
    // run as `GdnCoreParams` rather than as a scalar run, so slot 0 was the
    // struct's FIRST FIELD and a `Const` derived onto it would have read that
    // field's bits. Nothing in this signature could name a word of the run,
    // which is why the head count and the value width came back as asks even
    // though the statement was already carrying both.
    //
    // The marks name all eleven now. They are the same eleven words of the
    // same `Lowered::params` run the struct was staged from, reached by index
    // instead of by field, and the entrypoint takes each as its own
    // `const constant&` argument -- one `setBytes` apiece -- instead of one
    // `constant GdnCoreParams&` at a single buffer.
    //
    // So `keys::VHeads` and `keys::VDim` are gone from this body: `Asks`'
    // own test for whether a number is a fact -- *"two fires of the same
    // model, on the same deployment, can see different answers here"* -- is
    // FALSE of a head count and of a head width, which are checkpoint
    // configuration and reach the shader from this same run anyway. It stays
    // TRUE of every table below, which is why every one of them is still an
    // ask: the conv and recurrent slabs and the seat map are what the fire
    // threaded from the previous step, and no text can state them.
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let grid = core_grid(rows, *v_heads, *v_dim)?;
    ctx.fire(
        Fire::at(PREP_FILE, "gdn_core_recurrent_bfloat16").apply(Grid::of(grid, core_group(grid))),
        &[
            mixed.arg(),
            conv_state.arg(),
            rstate.arg_mut(),
            core_out.arg(),
            conv_w.arg(),
            conv_b.arg(),
            pre_q.arg(),
            pre_k.arg(),
            pre_gate.arg(),
            new_conv_state.arg_mut(),
            k_dim.arg(),
            v_dim.arg(),
            k_heads.arg(),
            v_heads.arg(),
            conv_dim.arg(),
            conv_k.arg(),
            q_off.arg(),
            k_off.arg(),
            v_off.arg(),
            eps.arg(),
            inv_sqrt_dk.arg(),
        ],
    )
}

/// [`gdn_core_recurrent`] with the state slabs reached through a slot map.
///
/// `slot_ids` at 10, where the packed `params` block used to be -- the lowest
/// of the three slots the same buffer takes across this family, and the eleven
/// geometry scalars follow it at 11 through 21.
///
/// # Errors
///
/// See [`core_grid`].
#[routine]
pub fn gdn_core_recurrent_slotted(
    ctx: &Ctx<'_>,
    mixed: In<Tensor<bf16>>,
    core_out: Out<Tensor<bf16>>,
    conv_w: Const<Tensor<bf16>>,
    conv_b: Const<Tensor<bf16>>,
    pre_q: In<Tensor<f32>>,
    pre_k: In<Tensor<f32>>,
    pre_gate: In<Tensor<f32>>,
    // THE ELEVEN, IN THE ORDER THE STATEMENT PLACES THEM. A `Const` mark's
    // slot is its POSITION among the scalar marks of this signature, so the
    // list below IS `model-dsl`'s `GdnShape::params` and a pair swapped here
    // reads its neighbour's number rather than refusing. The two swaps most
    // likely to be made are also the two that stay silent longest: `Dk` and
    // `Dv` are 128 and 128 on every GDN checkpoint the tree has seen, and
    // `Hk` and `Hv` differ only under group-query, so either transposition
    // runs and answers.
    k_dim: Const<i32>,
    v_dim: Const<i32>,
    k_heads: Const<i32>,
    v_heads: Const<i32>,
    conv_dim: Const<i32>,
    conv_k: Const<i32>,
    q_off: Const<i32>,
    k_off: Const<i32>,
    v_off: Const<i32>,
    eps: Const<f32>,
    inv_sqrt_dk: Const<f32>) -> Result<(), Refusal> {
    // THE ELEVEN GEOMETRY MARKS, WHICH WERE `GdnCoreParams`' ELEVEN FIELDS.
    //
    // A block stood here saying *"BACK TO AN ASK, BECAUSE THIS ROUTINE'S
    // PARAMS RUN IS A STRUCT"*, and it was true of the arrangement it
    // described: the body forwarded `ctx.params()` whole, the shader read the
    // run as `GdnCoreParams` rather than as a scalar run, so slot 0 was the
    // struct's FIRST FIELD and a `Const` derived onto it would have read that
    // field's bits. Nothing in this signature could name a word of the run,
    // which is why the head count and the value width came back as asks even
    // though the statement was already carrying both.
    //
    // The marks name all eleven now. They are the same eleven words of the
    // same `Lowered::params` run the struct was staged from, reached by index
    // instead of by field, and the entrypoint takes each as its own
    // `const constant&` argument -- one `setBytes` apiece -- instead of one
    // `constant GdnCoreParams&` at a single buffer.
    //
    // So `keys::VHeads` and `keys::VDim` are gone from this body: `Asks`'
    // own test for whether a number is a fact -- *"two fires of the same
    // model, on the same deployment, can see different answers here"* -- is
    // FALSE of a head count and of a head width, which are checkpoint
    // configuration and reach the shader from this same run anyway. It stays
    // TRUE of every table below, which is why every one of them is still an
    // ask: the conv and recurrent slabs and the seat map are what the fire
    // threaded from the previous step, and no text can state them.
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let grid = core_grid(rows, *v_heads, *v_dim)?;
    ctx.fire(
        Fire::at(PREP_FILE, "gdn_core_recurrent_slotted_bfloat16").apply(Grid::of(grid, core_group(grid))),
        &[
            mixed.arg(),
            conv_state.arg(),
            rstate.arg_mut(),
            core_out.arg(),
            conv_w.arg(),
            conv_b.arg(),
            pre_q.arg(),
            pre_k.arg(),
            pre_gate.arg(),
            new_conv_state.arg_mut(),
            slot_ids.arg(),
            k_dim.arg(),
            v_dim.arg(),
            k_heads.arg(),
            v_heads.arg(),
            conv_dim.arg(),
            conv_k.arg(),
            q_off.arg(),
            k_off.arg(),
            v_off.arg(),
            eps.arg(),
            inv_sqrt_dk.arg(),
        ],
    )
}

/// The prep over a WHOLE PROMPT: `n_scan` tokens in one dispatch instead of
/// one dispatch per token.
///
/// Its z is `n_scan * Hv` (`device_gdn.rs:1270`) where the decode prep's is
/// `rows * Hv`, and the tokens really are parallel: a prompt's conv window is
/// the prompt itself once `t >= Kc - 1`, and only the last scanned token
/// carries the history forward (`gdn_prep.metal:426`). A prompt walked token
/// by token serialized the pair behind a barrier per token -- 34 tokens by 18
/// layers by 2 kernels is 1224 dispatches in a strict chain, measured at 25 ms
/// of a 60 ms prefill.
///
/// It takes `slot_ids` unconditionally and has no unslotted twin, because it
/// reads `slot_ids[0]` and nothing else (`gdn_prep.metal:373`): one prompt is
/// one slot.
///
/// `row_pitch` and `n_scan` are separate `constant int&` operands at 24 and
/// 25, AFTER the eleven geometry scalars at 13 through 23. They were at 14 and
/// 15 after a packed `params` block at 12 -- one dispatch carrying a struct
/// and two loose scalars -- and they stay LAST because the body passes them
/// last and a Metal argument index is the position in that list.
/// [`gdn_core_recurrent_prefill`] puts the same two at 22 and 23, so the two
/// prefill kernels share neither numbering nor a binding order, and a body
/// that copied one into the other would write the pitch into the middle of
/// the geometry.
///
/// Neither is a `Const` mark, and that is the line this migration did not
/// cross. `row_pitch` is `mixed`'s own row -- the in-projection's
/// width in activation elements, which is what the shader indexes it by
/// (`gdn_prep.metal:393`) and nothing else. NOT "the widest tensor in the
/// prefill scratch layout", which this said until the layout stopped having
/// one: `pre_gate` is WIDER than `mixed` wherever a stack's value width
/// reaches its key width -- qwen3-next asks 2*Hv + Hv*Dv = 8320 floats of it
/// against a conv_dim of 8192 -- so every scratch row is packed at its own
/// width and only `mixed` is measured by this number. `n_scan` is the fire's
/// token count. Neither is a number a text states.
///
/// # Errors
///
/// [`Refusal::Empty`] naming `n_scan` before [`head_rows`] can name it
/// `rows` -- the two are the same product and a caller that scanned nothing
/// wants to be told which one it was. Also [`Refusal::Empty`] for a
/// `row_pitch` of zero, which is not the same mistake: every token's scratch
/// row would land on token zero's, the scan would read token zero's q and k
/// for the whole prompt, and the prompt would come back a plausible length.
#[routine]
pub fn gdn_prep_prefill(
    ctx: &Ctx<'_>,
    mixed: In<Tensor<bf16>>,
    conv_w: Const<Tensor<bf16>>,
    conv_b: Const<Tensor<bf16>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<bf16>>,
    a_gate: In<Tensor<bf16>>,
    b_gate: In<Tensor<bf16>>,
    pre_q: Out<Tensor<f32>>,
    pre_k: Out<Tensor<f32>>,
    pre_gate: Out<Tensor<f32>>,
    // THE ELEVEN, IN THE ORDER THE STATEMENT PLACES THEM. A `Const` mark's
    // slot is its POSITION among the scalar marks of this signature, so the
    // list below IS `model-dsl`'s `GdnShape::params` and a pair swapped here
    // reads its neighbour's number rather than refusing. The two swaps most
    // likely to be made are also the two that stay silent longest: `Dk` and
    // `Dv` are 128 and 128 on every GDN checkpoint the tree has seen, and
    // `Hk` and `Hv` differ only under group-query, so either transposition
    // runs and answers.
    k_dim: Const<i32>,
    v_dim: Const<i32>,
    k_heads: Const<i32>,
    v_heads: Const<i32>,
    conv_dim: Const<i32>,
    conv_k: Const<i32>,
    q_off: Const<i32>,
    k_off: Const<i32>,
    v_off: Const<i32>,
    eps: Const<f32>,
    inv_sqrt_dk: Const<f32>) -> Result<(), Refusal> {
    // THE ELEVEN GEOMETRY MARKS, WHICH WERE `GdnCoreParams`' ELEVEN FIELDS.
    //
    // A block stood here saying *"BACK TO AN ASK, BECAUSE THIS ROUTINE'S
    // PARAMS RUN IS A STRUCT"*, and it was true of the arrangement it
    // described: the body forwarded `ctx.params()` whole, the shader read the
    // run as `GdnCoreParams` rather than as a scalar run, so slot 0 was the
    // struct's FIRST FIELD and a `Const` derived onto it would have read that
    // field's bits. Nothing in this signature could name a word of the run,
    // which is why the head count and the value width came back as asks even
    // though the statement was already carrying both.
    //
    // The marks name all eleven now. They are the same eleven words of the
    // same `Lowered::params` run the struct was staged from, reached by index
    // instead of by field, and the entrypoint takes each as its own
    // `const constant&` argument -- one `setBytes` apiece -- instead of one
    // `constant GdnCoreParams&` at a single buffer.
    //
    // So `keys::VHeads` and `keys::VDim` are gone from this body: `Asks`'
    // own test for whether a number is a fact -- *"two fires of the same
    // model, on the same deployment, can see different answers here"* -- is
    // FALSE of a head count and of a head width, which are checkpoint
    // configuration and reach the shader from this same run anyway. It stays
    // TRUE of every table below, which is why every one of them is still an
    // ask: the conv and recurrent slabs and the seat map are what the fire
    // threaded from the previous step, and no text can state them.
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    let row_pitch = mixed.width;
    let n_scan = ctx.ask::<i32, keys::Rows>()?;
    if n_scan <= 0 {
        return Err(Refusal::Empty { what: "n_scan" });
    }
    if row_pitch <= 0 {
        return Err(Refusal::Empty { what: "row_pitch" });
    }
    let grid = prep_grid(n_scan, *v_heads)?;
    ctx.fire(
        Fire::at(PREP_FILE, "gdn_prep_prefill_bfloat16").apply(Grid::of(grid, simd_group(grid))),
        &[
            mixed.arg(),
            conv_state.arg(),
            conv_w.arg(),
            conv_b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            a_gate.arg(),
            b_gate.arg(),
            pre_q.arg(),
            pre_k.arg(),
            pre_gate.arg(),
            new_conv_state.arg_mut(),
            slot_ids.arg(),
            k_dim.arg(),
            v_dim.arg(),
            k_heads.arg(),
            v_heads.arg(),
            conv_dim.arg(),
            conv_k.arg(),
            q_off.arg(),
            k_off.arg(),
            v_off.arg(),
            eps.arg(),
            inv_sqrt_dk.arg(),
            row_pitch.arg(),
            n_scan.arg(),
        ],
    )
}

/// The scan: the whole prompt's recurrence walked inside one kernel, with the
/// state living in registers for the duration.
///
/// Nine entrypoints over one template, and [`scan_point`] picks the spelling.
/// `lanes` and `vrows` are `Const<i32>` MARKS at words 11 and 12 of this
/// entrypoint's own thirteen-word run: `model-dsl`'s
/// `gdn_core_recurrent_prefill` pushes the tile onto `GdnShape::params`'
/// eleven, so the text really does name them, twice -- once there and once
/// spelled into the symbol this body composes. They were `Env` while the run
/// was a struct with no slot to spare, and `Asks::param` existed to read them
/// by index because of it.
///
/// # The five pad slots
///
/// This entrypoint declares 2, 3, 6, 7, 8, 10 and then 11 through 23, and
/// **nothing at 0, 1, 4, 5 or 9**. The BUFFER numbering is
/// [`gdn_core_recurrent`]'s, kept so the two can be encoded against one
/// argument table; the scan simply needs none of `mixed`, `conv_state`,
/// `conv_w`, `conv_b` or `new_conv_state`, because the prep already did every
/// convolution and wrote the history forward.
///
/// A Metal argument table is a contiguous run, so the five holes still have to
/// hold an address: `device_gdn.rs:1206`'s `fill` gives them all `core_out`'s
/// address and then overwrites the ones it binds. A routine's argument list is
/// positional -- the index in the list IS the buffer slot -- so `pad` is taken
/// once and bound at each hole. Skipping them would slide `slot_ids` into slot
/// 6 and hand the scan a seat table where it reads q.
///
/// Past the holes it is twenty-four slots, not fourteen: `slot_ids` sits at
/// 10, where the `GdnCoreParams` pointer used to, the eleven geometry scalars
/// follow at 11 through 21, and `row_pitch` and `n_scan` are last at 22 and
/// 23. Metal allows 31, so this is the widest entrypoint in the family and
/// still seven short of the ceiling.
///
/// `pad` is [`Buf`]: nothing dereferences it, and a read-only handle is the
/// weakest claim that can fill a slot. NOTHING READS ITS SHAPE EITHER -- see
/// the body's note on `row_pitch`, which used to and is the reason this
/// sentence is now two.
///
/// # Errors
///
/// [`Refusal::Narrow`] from [`scan_point`] for a tiling the shader tree does
/// not carry, and [`Refusal::Empty`] from [`scan_grid`] for a zero extent or
/// from the two prefill scalars, as [`gdn_prep_prefill`] -- except that
/// `row_pitch` here is `pre_q`'s row and not `mixed`'s, because `mixed` is
/// not an operand this entrypoint has.
///
/// The tiling is checked FIRST: an entrypoint Metal has no `[[host_name]]`
/// for makes `newFunctionWithName:` return nil at run time, inside a fire,
/// after the plan was accepted -- so it has to be a refusal here, where the
/// caller can still step down to a tiling that exists.
#[routine]
pub fn gdn_core_recurrent_prefill(
    ctx: &Ctx<'_>,
    pad: In<Tensor<bf16>>,
    core_out: Out<Tensor<bf16>>,
    pre_q: In<Tensor<f32>>,
    pre_k: In<Tensor<f32>>,
    pre_gate: In<Tensor<f32>>,
    // THE ELEVEN, IN THE ORDER THE STATEMENT PLACES THEM. A `Const` mark's
    // slot is its POSITION among the scalar marks of this signature, so the
    // list below IS `model-dsl`'s `GdnShape::params` and a pair swapped here
    // reads its neighbour's number rather than refusing. The two swaps most
    // likely to be made are also the two that stay silent longest: `Dk` and
    // `Dv` are 128 and 128 on every GDN checkpoint the tree has seen, and
    // `Hk` and `Hv` differ only under group-query, so either transposition
    // runs and answers.
    k_dim: Const<i32>,
    v_dim: Const<i32>,
    k_heads: Const<i32>,
    v_heads: Const<i32>,
    conv_dim: Const<i32>,
    conv_k: Const<i32>,
    q_off: Const<i32>,
    k_off: Const<i32>,
    v_off: Const<i32>,
    eps: Const<f32>,
    inv_sqrt_dk: Const<f32>,
    // THE TILING, WHICH IS WORDS 11 AND 12 OF THE SAME RUN. `model-dsl`'s
    // `gdn_core_recurrent_prefill` pushes `tile.0` and `tile.1` onto
    // `GdnShape::params`' eleven, so the statement's run is thirteen words
    // long for this entrypoint alone and these two are the last of them.
    // Being marks, they take slots 11 and 12 by POSITION -- which is exactly
    // where the run puts them -- and the shader never sees either: the pair
    // is spelled into the entrypoint's own NAME and reaches the module as
    // `PIE_LANES`/`PIE_VROWS` at compile time, so `scan_point` below turns
    // them into a symbol rather than into an argument.
    lanes: Const<i32>,
    vrows: Const<i32>) -> Result<(), Refusal> {
    // THE ELEVEN GEOMETRY MARKS, WHICH WERE `GdnCoreParams`' ELEVEN FIELDS.
    //
    // A block stood here saying *"BACK TO AN ASK, BECAUSE THIS ROUTINE'S
    // PARAMS RUN IS A STRUCT"*, and it was true of the arrangement it
    // described: the body forwarded `ctx.params()` whole, the shader read the
    // run as `GdnCoreParams` rather than as a scalar run, so slot 0 was the
    // struct's FIRST FIELD and a `Const` derived onto it would have read that
    // field's bits. Nothing in this signature could name a word of the run,
    // which is why the head count and the value width came back as asks even
    // though the statement was already carrying both.
    //
    // The marks name all eleven now. They are the same eleven words of the
    // same `Lowered::params` run the struct was staged from, reached by index
    // instead of by field, and the entrypoint takes each as its own
    // `const constant&` argument -- one `setBytes` apiece -- instead of one
    // `constant GdnCoreParams&` at a single buffer.
    //
    // So `keys::VHeads` and `keys::VDim` are gone from this body: `Asks`'
    // own test for whether a number is a fact -- *"two fires of the same
    // model, on the same deployment, can see different answers here"* -- is
    // FALSE of a head count and of a head width, which are checkpoint
    // configuration and reach the shader from this same run anyway. It stays
    // TRUE of every table below, which is why every one of them is still an
    // ask: the conv and recurrent slabs and the seat map are what the fire
    // threaded from the previous step, and no text can state them.
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    // `row_pitch` RIDES SLOT 22 AND THIS SHADER NEVER READS IT
    // (`gdn_prep.metal:533`). It rode slot 12 while the eleven geometry
    // fields were one `constant GdnCoreParams&` at slot 11; they are eleven
    // marks now and every loose scalar after them moved up by ten.
    // Every row the scan walks is packed at its own
    // width and the kernel reckons all three off the parameter block --
    // `qk_pitch = Hv*Dk`, `g_pitch = 2*Hv + Hv*Dv`, `o_pitch = Hv*Dv`. The
    // operand exists because the numbering is [`gdn_core_recurrent`]'s and a
    // Metal argument table is a contiguous run, not because anything reads it.
    //
    // It said `pad.width`, which is the shape of dependency `232892260`
    // named: a routine reading a fact off an operand that does not carry it.
    // `pad` is the FILLER for this entrypoint's five holes -- "nothing
    // dereferences it, and a read-only handle is the weakest claim that can
    // fill a slot" -- so its width is whatever a text happened to place at
    // input 0. MEASURED, on Qwen3.6-27B through `model-dsl`, which pads with
    // `mixed`: `pad.width` is 10240, the in-projection's `conv_dim`, and
    // `pre_q.width` is 6144 = Hv*Dk = 48*128. `device_gdn.rs`'s harness fills
    // the same holes with `core_out` instead, whose row is Hv*Dv. Different
    // numbers, one dispatch, one answer -- which is the measurement that this
    // operand is not a pitch anybody reads.
    //
    // The refusal below is what made that matter. An operand whose statement
    // gives no width answers zero -- a weight binds `width: 0` by
    // construction, and `Holds::in_width` falls back to it -- so a text that
    // filled the holes with such a value would make EVERY prefill scan on
    // that stack refuse `Empty { row_pitch }`, over a number no shader reads.
    // So it is stated from `pre_q`, which this kernel does read, and read at
    // exactly this pitch.
    let row_pitch = pre_q.width;
    let n_scan = ctx.ask::<i32, keys::Rows>()?;
    // THE TILING, READ AS TWO MARKS RATHER THAN BY INDEX.
    //
    // `let lanes = ctx.param(11)?;` and `let vrows = ctx.param(12)?;` stood
    // here, and `Asks::param`'s own doc named this routine as the reason that
    // method exists at all: a body forwarding its params run as a STRUCT has
    // no slots to spare, because slots 0..10 are the struct's fields and not
    // this body's to take. (They had been `Param<11, i32>` and
    // `Param<12, i32>` before that, and the migration between the two turned
    // them into `keys::Lanes` and `keys::Vrows`, which no driver answers, so
    // every prefill scan refused `Unstated`.)
    //
    // With the eleven fields stated as marks the numbering works out on its
    // own: the marks are positional, the eleven come first in
    // `GdnShape::params`' order, and these two follow at 11 and 12 -- the
    // same two words `ctx.param` was reaching for, now named.
    let point = scan_point(*lanes, *vrows)?;
    if n_scan <= 0 {
        return Err(Refusal::Empty { what: "n_scan" });
    }
    if row_pitch <= 0 {
        return Err(Refusal::Empty { what: "row_pitch" });
    }
    let grid = scan_grid(*v_dim, *v_heads, *lanes, *vrows)?;
    ctx.fire(
        Fire::at(PREP_FILE, GDN_SCAN[point]).apply(Grid::of(grid, simd_group(grid))),
        &[
            pad.arg(),
            pad.arg(),
            rstate.arg_mut(),
            core_out.arg(),
            pad.arg(),
            pad.arg(),
            pre_q.arg(),
            pre_k.arg(),
            pre_gate.arg(),
            pad.arg(),
            slot_ids.arg(),
            k_dim.arg(),
            v_dim.arg(),
            k_heads.arg(),
            v_heads.arg(),
            conv_dim.arg(),
            conv_k.arg(),
            q_off.arg(),
            k_off.arg(),
            v_off.arg(),
            eps.arg(),
            inv_sqrt_dk.arg(),
            row_pitch.arg(),
            n_scan.arg(),
        ],
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do, and answers the
    /// five facts this file's bodies still ask for: the four state buffers
    /// (`ConvState`, `RecurrentState`, `NewConvState`, `RecurrentSlots`) and
    /// `Rows` -- also read as `n_scan`, since a scan over a whole prompt
    /// counts tokens on the same axis a decode counts rows. Every Cell has a
    /// default a test that does not care can ignore, and a test that does care
    /// overrides it before firing.
    ///
    /// WHAT IT NO LONGER ANSWERS is the interesting half. It carried cells for
    /// `v_heads`, `v_dim`, `lanes` and `vrows`, a `params_handle` for the
    /// staged block, and a `words` run standing in for the statement's
    /// scalars, because every body here forwarded `ctx.params()` as a STRUCT
    /// and could name no word inside it: the head count came back through
    /// `keys::VHeads`, and the scan's tiling through `Asks::param`, reading
    /// words 11 and 12 by index. All six are `Const` marks now, so a CALL
    /// states them and this probe has nothing to say about any of them --
    /// including `Slot(Kind::Params, 0)`, which no body in this file reaches
    /// for any more.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        conv_state: Cell<u32>,
        rstate: Cell<u32>,
        new_conv_state: Cell<u32>,
        slot_ids: Cell<u32>,
        rows: Cell<i32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                conv_state: Cell::new(501),
                rstate: Cell::new(502),
                new_conv_state: Cell::new(503),
                slot_ids: Cell::new(504),
                rows: Cell::new(ROWS),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(
            &self,
            _ty: kernels::Ty,
            source: kernels::Source,
        ) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            if source == <keys::ConvState as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.conv_state.get()));
            }
            if source == <keys::RecurrentState as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.rstate.get()));
            }
            if source == <keys::NewConvState as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.new_conv_state.get()));
            }
            if source == <keys::RecurrentSlots as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.slot_ids.get()));
            }
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
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

    /// `device_gdn.rs`'s own fixture, so the grids below are comparable to the
    /// literals that test hands the device.
    const ROWS: i32 = 2;
    const HV: i32 = 4;
    const DV: i32 = 8;
    /// `T_SCAN` and `ROW_PITCH` there.
    const N_SCAN: i32 = 5;
    const ROW_PITCH: i32 = 1100;
    /// What the scan's FILLER slot happens to be as wide as, which is not a
    /// pitch and is deliberately not [`ROW_PITCH`]. On Qwen3.6-27B the two
    /// really do differ -- `model-dsl` pads with `mixed`, 10240 wide, against
    /// a `pre_q` row of 6144 -- so this fixture keeps them apart too.
    const PAD_WIDTH: i32 = 1717;

    /// The rest of the geometry every call in this module states, in
    /// `GdnShape::params` order and minus the two numbers a test varies.
    ///
    /// They are CONSTANTS and not probe cells, which is what the migration to
    /// `Const` marks changed here: the eleven words used to reach the shader
    /// as a `GdnCoreParams` block the body forwarded whole, so a test that
    /// wanted a particular head count set it on the probe; they are arguments
    /// now, so a call states its own geometry.
    ///
    /// Nothing here dispatches a shader, so only the ORDER is load-bearing --
    /// but a fixture that could not describe a real checkpoint would be a bad
    /// place to read the order off, so these are consistent with `HV` and
    /// `DV`: one key head 128 wide, four value heads 8 wide, and the three
    /// region offsets laid over `2*Hk*Dk + Hv*Dv` channels.
    const K_DIM: i32 = 128;
    /// See [`K_DIM`].
    const K_HEADS: i32 = 1;
    /// See [`K_DIM`].
    const CONV_K: i32 = 4;
    /// See [`K_DIM`].
    const CONV_DIM: i32 = 2 * K_HEADS * K_DIM + HV * DV;
    /// See [`K_DIM`].
    const Q_OFF: i32 = 0;
    /// See [`K_DIM`].
    const K_OFF: i32 = K_HEADS * K_DIM;
    /// See [`K_DIM`].
    const V_OFF: i32 = 2 * K_HEADS * K_DIM;
    /// See [`K_DIM`].
    const EPS: f32 = 1e-6;
    /// See [`K_DIM`].
    const INV_SQRT_DK: f32 = 0.088_388_35;

    /// The eleven scalars a body forwards, as this module expects to read them
    /// back off a recorded call.
    ///
    /// `v_heads` and `v_dim` are parameters because they are the two a test
    /// varies; the other nine are the constants above. `eps` and
    /// `inv_sqrt_dk` come back as `ArgValue::F32` and not `I32`, which is the
    /// one thing a run of eleven words could not have said: a `Const<f32>`
    /// resolves through `Kind::ParamF32` and binds the float READING of the
    /// same bits.
    fn geometry(v_heads: i32, v_dim: i32) -> Vec<ArgValue> {
        vec![
            ArgValue::I32(K_DIM),
            ArgValue::I32(v_dim),
            ArgValue::I32(K_HEADS),
            ArgValue::I32(v_heads),
            ArgValue::I32(CONV_DIM),
            ArgValue::I32(CONV_K),
            ArgValue::I32(Q_OFF),
            ArgValue::I32(K_OFF),
            ArgValue::I32(V_OFF),
            ArgValue::F32(EPS),
            ArgValue::F32(INV_SQRT_DK),
        ]
    }

    /// A decode core dispatches the grid `device_gdn.rs` fires.
    ///
    /// `[32, Dv, rows * Hv]` and `[32, 4, 1]` -- `device_gdn.rs:362` and
    /// `:363` for the fused core, `:559` and `:560` for the split one. Three
    /// separate symbols computing one geometry is exactly the arrangement
    /// where a transcription drifts, and until this file existed the geometry
    /// was written down only in that test.
    #[test]
    fn the_three_cores_dispatch_the_grid_the_device_test_fires() {
        let seen = Seen::default();
        core(&seen, ROWS, HV, DV).expect("a launch");
        core_slotted(&seen, HV, DV).expect("a launch");
        recurrent(&seen, HV, DV).expect("a launch");

        for (fire, _) in seen.calls.borrow().iter() {
            assert_eq!(
                fire.lanes,
                [32, 8, 8],
                "`{}`: a simdgroup per (row, v-head, v-channel), and the z is \
                 the PRODUCT -- the shader splits it back with `n / Hv` and \
                 `n % Hv`",
                fire.entrypoint
            );
            assert_eq!(
                fire.group,
                [32, 4, 1],
                "`{}`: four dv rows to a threadgroup",
                fire.entrypoint
            );
        }
    }

    /// The dk lane is the SAME number in the grid and in the threadgroup, for
    /// every kernel in the family.
    ///
    /// `gdn_core` and `gdn_core_slotted` read that lane off
    /// `thread_position_in_threadgroup.x` (`gdn_core.metal:85`) and the other
    /// six off `thread_position_in_grid.x`. The two agree only while there is
    /// one threadgroup on x, which is only true while `group.x == grid.x`, and
    /// a fused core given a wider grid than group would compute head zero's
    /// q and k `grid.x / 32` times over and never touch the rest.
    ///
    /// Both numbers are [`SIMD`] because the reductions are a simdgroup's:
    /// `simd_sum` at `gdn_core.metal:121`, `simd_shuffle_xor` at
    /// `gdn_prep.metal:450`.
    #[test]
    fn the_dk_lane_is_one_number_in_the_grid_and_the_threadgroup() {
        let seen = Seen::default();
        every_kernel(&seen);
        for (fire, _) in seen.calls.borrow().iter() {
            assert_eq!(
                fire.lanes[0], fire.group[0],
                "`{}` states a grid x and a group x that disagree",
                fire.entrypoint
            );
            assert_eq!(
                fire.lanes[0], SIMD,
                "`{}` is not a simdgroup wide",
                fire.entrypoint
            );
        }
    }

    /// The prep comes OFF the dv axis and the recurrent half keeps it.
    ///
    /// `[32, 1, rows * Hv]` against `[32, Dv, rows * Hv]` --
    /// `device_gdn.rs:542` and `:559`. This is the whole of what the split
    /// buys: the q/k path runs once per head rather than once per dv tile. A
    /// prep given the core's grid would also have `Dv` threadgroups writing
    /// one head's `pre_q`, `pre_k` and q/k convolution history on top of each
    /// other.
    #[test]
    fn the_prep_has_no_dv_axis_and_the_recurrent_half_does() {
        let seen = Seen::default();
        prep(&seen, HV, DV).expect("a launch");
        recurrent(&seen, HV, DV).expect("a launch");

        let calls = seen.calls.borrow();
        assert_eq!(calls[0].0.lanes, [32, 1, 8]);
        assert_eq!(calls[0].0.group, [32, 1, 1]);
        assert_eq!(calls[1].0.lanes, [32, 8, 8]);
        assert_eq!(calls[1].0.group, [32, 4, 1]);
    }

    /// The prefill prep counts TOKENS on z where the decode prep counts rows.
    ///
    /// `[32, 1, n_scan * Hv]` -- `device_gdn.rs:1270`. A prompt walked one
    /// token at a time serialized the pair behind a barrier per token, which
    /// is the 1224-dispatch chain the shader header measured at 25 ms of a
    /// 60 ms prefill.
    #[test]
    fn the_prefill_prep_puts_the_whole_prompt_on_z() {
        let seen = Seen::default();
        prep_prefill(&seen, N_SCAN, ROW_PITCH).expect("a launch");
        let calls = seen.calls.borrow();
        assert_eq!(calls[0].0.lanes, [32, 1, 20], "five tokens by four heads");
        assert_eq!(calls[0].0.group, [32, 1, 1]);
        assert_eq!(calls[0].0.entrypoint, "gdn_prep_prefill_bfloat16");
    }

    /// The scan's y is `Dv` folded by the tiling, for all nine of them.
    ///
    /// `(32 / LANES) * VROWS` rows per y, rounded UP -- `device_gdn.rs:1264`
    /// and `:1265`, recomputed here from the same two numbers. Rounding down
    /// would leave the tail of every head's state unscanned while the rest of
    /// the prompt advanced over it; the shader is written for the excess
    /// instead, returning early at `gdn_prep.metal:497` and masking the short
    /// last group at `:501`.
    #[test]
    fn the_scan_folds_dv_by_its_tiling_and_rounds_up() {
        for (lanes, vrows) in [
            (4, 1),
            (8, 1),
            (8, 2),
            (16, 1),
            (16, 2),
            (16, 4),
            (32, 2),
            (32, 4),
            (32, 8),
        ] {
            let seen = Seen::default();
            scan(&seen, DV, lanes, vrows).expect("a launch");
            let want = u32::try_from(DV)
                .unwrap()
                .div_ceil(u32::try_from((32 / lanes) * vrows).unwrap());
            let calls = seen.calls.borrow();
            assert_eq!(
                calls[0].0.lanes,
                [32, want, 4],
                "l_{lanes}_v_{vrows} covers Dv={DV} in {want} on y, with the \
                 heads on z"
            );
            assert_eq!(calls[0].0.group, [32, 1, 1]);
            assert_eq!(
                calls[0].0.entrypoint,
                format!("gdn_core_recurrent_prefill_bfloat16_l_{lanes}_v_{vrows}"),
                "the spelling is picked from the table, not pasted"
            );
        }

        // A `Dv` the tiling does not divide: 130 over 8 is 17 groups, of which
        // the last carries two rows and masks six.
        let seen = Seen::default();
        scan(&seen, 130, 32, 8).expect("a launch");
        assert_eq!(seen.calls.borrow()[0].0.lanes[1], 17);
    }

    /// A `(lanes, vrows)` the shader tree does not carry is refused, and
    /// nothing is encoded on the way to refusing.
    ///
    /// `(32, 1)` is the falsifier worth having: it is the cheapest-looking
    /// tiling in the family, it is what a caller would reach for after
    /// `(32, 2)`, and it is not compiled. Nine of sixteen points exist and the
    /// seven that do not are not a rectangle, which is why the lookup is a
    /// match over pairs.
    ///
    /// The refusal has to be HERE. Metal resolves an entrypoint by name at run
    /// time, so a spelling with no `[[host_name]]` makes `newFunctionWithName:`
    /// return nil inside a fire, after the plan was accepted and after
    /// everything before it in the plan has been encoded.
    #[test]
    fn a_tiling_the_shader_tree_does_not_carry_is_refused() {
        let seen = Seen::default();
        assert_eq!(
            scan(&seen, DV, 32, 1),
            Err(Refusal::Narrow {
                what: "scan rows per lane group, at this lane width",
                at: 1
            }),
            "32 lanes are compiled for 2, 4 and 8 rows and not for 1"
        );
        assert_eq!(
            scan(&seen, DV, 2, 1),
            Err(Refusal::Narrow {
                what: "scan lane width",
                at: 2
            }),
            "and the lane width is named first, because it is the one that \
             changes the reduction"
        );
        assert_eq!(
            scan(&seen, DV, 4, 2),
            Err(Refusal::Narrow {
                what: "scan rows per lane group, at this lane width",
                at: 2
            })
        );
        assert!(seen.calls.borrow().is_empty(), "and none of them was encoded");

        // Every one of the nine that DOES exist names a spelling in the table,
        // and the table is the shader census's nine.
        assert_eq!(GDN_SCAN.len(), 9);
        for name in GDN_SCAN {
            assert!(
                name.starts_with("gdn_core_recurrent_prefill_bfloat16_l_"),
                "{name}"
            );
        }
    }

    /// The scan binds `pad` at every slot its entrypoint does not declare, and
    /// the twelve it does declare land where the shader put them.
    ///
    /// `gdn_prep.metal` declares 2, 3, 6, 7, 8, 10 and then 11..23, and
    /// **nothing at 0, 1, 4, 5 or 9** -- the buffer numbering is
    /// [`gdn_core_recurrent`]'s, kept so both can be encoded against one
    /// argument table. A routine's argument list is positional, so closing the
    /// holes would slide `slot_ids` into slot 6 and hand the scan a seat table
    /// where it reads q.
    ///
    /// The geometry moved DOWN by one here and everywhere else in the family:
    /// `slot_ids` sits at 10, where the `GdnCoreParams` pointer used to, and
    /// the eleven scalars that replaced that pointer follow it at 11..21 with
    /// `row_pitch` and `n_scan` last at 22 and 23. Nothing reaches buffer 31,
    /// which is Metal's ceiling.
    ///
    /// `rstate` and `core_out` come back `ArgValue::BufferMut`, which is the
    /// claim worth making rather than assuming: `rstate` is ASKED as a bare
    /// `Tensor<f32>` and bound through `arg_mut`, `core_out` is an
    /// `Out<Tensor<bf16>>` bound through `Bind`, and the two reach the same
    /// constructor by different routes. `driver-metal`'s `touches` reads that
    /// variant to learn which buffers a dispatch writes, and the scan writes
    /// both.
    #[test]
    fn the_scan_binds_a_pad_at_every_slot_its_entrypoint_does_not_declare() {
        let seen = Seen::default();
        seen.rstate.set(2);
        seen.slot_ids.set(10);
        scan(&seen, DV, 32, 4).expect("a launch");
        let calls = seen.calls.borrow();
        let args = &calls[0].1;
        assert_eq!(args.len(), 24, "buffer 23 is the last one declared");
        for at in [0, 1, 4, 5, 9] {
            assert_eq!(
                args[at],
                ArgValue::Buffer(90),
                "slot {at} is a pad: the entrypoint declares nothing there and \
                 a Metal argument table is a contiguous run"
            );
        }
        assert_eq!(
            args[2..4],
            [ArgValue::BufferMut(2), ArgValue::BufferMut(3)],
            "rstate then core_out"
        );
        assert_eq!(
            args[6..9],
            [
                ArgValue::Buffer(6),
                ArgValue::Buffer(7),
                ArgValue::Buffer(8)
            ],
            "pre_q, pre_k, pre_gate -- read back, not recomputed"
        );
        assert_eq!(
            args[10],
            ArgValue::Buffer(10),
            "the seat map, at the slot the params pointer used to hold"
        );
        assert_eq!(
            args[11..22],
            geometry(HV, DV)[..],
            "the eleven geometry scalars, in `GdnShape::params` order, where \
             one `constant GdnCoreParams&` used to stand"
        );
        assert_eq!(
            args[22..],
            [ArgValue::I32(ROW_PITCH), ArgValue::I32(N_SCAN)],
            "and the two loose scalars last, at 22 and 23 -- where \
             `gdn_prep_prefill` puts them at 24 and 25"
        );
    }

    /// THE SCAN'S `row_pitch` IS A ROW IT READS, NOT THE FILLER'S.
    ///
    /// This body said `let row_pitch = pad.width`, which is the dependency
    /// `232892260` named twice over: a routine taking a fact off an operand
    /// that does not carry it. `pad` is the handle bound into the five holes
    /// this entrypoint leaves at slots 0, 1, 4, 5 and 9 -- `model-dsl` fills
    /// them with `mixed`, `device_gdn.rs`'s harness with `core_out`, and the
    /// routine's own doc says nothing dereferences it. So the number stated
    /// at slot 12 was whichever tensor a caller happened to put at input 0.
    ///
    /// The shader never reads that scalar (`gdn_prep.metal:533`: every row is
    /// packed at its own width and all three pitches are reckoned off the
    /// parameter block), so the wrong value cost no answer. The REFUSAL cost
    /// one: `row_pitch <= 0` refuses the fire, and an operand whose statement
    /// gives no width answers zero, so a text that filled the holes with such
    /// a value would have stopped every hybrid prefill on this backend over a
    /// number no kernel reads.
    ///
    /// Asserted as the relation rather than against a literal: the scalar at
    /// slot 22 follows `pre_q`'s row and does NOT follow the pad's, with the
    /// two widths held apart so that one answer cannot pass for the other.
    ///
    /// SLOT 22, WHICH WAS 12. The eleven geometry fields were one
    /// `constant GdnCoreParams&` at slot 11 and are eleven marks now, so the
    /// two loose scalars after them moved up by ten. Reading 12 read `v_dim`
    /// -- a `Const` this fixture holds FIXED across all three fires -- so
    /// every relation below held trivially and the one that must not hold
    /// was the one that failed. A slot index written down beside a dispatch
    /// list is a restatement of it, and this is the drift that produces.
    #[test]
    fn the_scans_row_pitch_follows_pre_q_and_not_whatever_fills_its_holes() {
        let seen = Seen::default();
        // Same fire twice, changing ONLY the filler's width.
        scan_with(&seen, DV, 32, 4, PAD_WIDTH, ROW_PITCH).expect("a launch");
        scan_with(&seen, DV, 32, 4, PAD_WIDTH * 3, ROW_PITCH).expect("a launch");
        // And once more, changing only `pre_q`'s.
        scan_with(&seen, DV, 32, 4, PAD_WIDTH, ROW_PITCH + 64).expect("a launch");

        let calls = seen.calls.borrow();
        let pitch_of = |call: &Call| call.1[22];
        assert_eq!(
            pitch_of(&calls[0]),
            pitch_of(&calls[1]),
            "the pad's width is not the scan's pitch, and tripling it must \
             move nothing"
        );
        assert_ne!(
            pitch_of(&calls[0]),
            pitch_of(&calls[2]),
            "`pre_q`'s row is, and widening it must move exactly it"
        );
        assert_eq!(pitch_of(&calls[2]), ArgValue::I32(ROW_PITCH + 64));
        assert_ne!(
            pitch_of(&calls[0]),
            ArgValue::I32(PAD_WIDTH),
            "and the filler's width is not what reached slot 22"
        );
    }

    /// A scan whose `pre_q` states no row is refused, and the refusal names
    /// `row_pitch` rather than the operand.
    ///
    /// Zero is what [`kernels::bind`] answers for an operand a statement
    /// carries no width for, so this is the shape a malformed statement
    /// arrives in. It is refused BEFORE the dispatch for the reason every
    /// extent in this family is: a fire that runs nothing reports success and
    /// leaves `core_out` holding the last token's answer.
    #[test]
    fn a_scan_whose_pre_q_states_no_row_is_refused_by_name() {
        let seen = Seen::default();
        assert_eq!(
            scan_with(&seen, DV, 32, 4, PAD_WIDTH, 0),
            Err(Refusal::Empty { what: "row_pitch" })
        );
        assert!(seen.calls.borrow().is_empty(), "and it was never encoded");
    }

    /// One slot map, three slots.
    ///
    /// `slot_ids` is buffer 11 in `gdn_core_slotted`, 12 in `gdn_prep_slotted`
    /// and `gdn_prep_prefill`, and 10 in `gdn_core_recurrent_slotted`, because
    /// each entrypoint numbers its own list. Every one of those is ONE LOWER
    /// than it was, because the `GdnCoreParams` pointer that sat in front of
    /// it is eleven scalars now and they come after every buffer.
    ///
    /// It is the same buffer carrying the same meaning in all four, which is
    /// exactly the shape a shared binding struct gets wrong: only the STATE
    /// accesses remap through it (`gdn_core.metal:90`), so a misplaced one
    /// still reads activations that are there.
    #[test]
    fn the_slot_map_lands_where_each_entrypoint_declares_it() {
        let seen = Seen::default();
        seen.slot_ids.set(77);
        core_slotted(&seen, HV, DV).expect("a launch");
        prep_slotted(&seen, HV, DV).expect("a launch");
        recurrent_slotted(&seen, HV, DV).expect("a launch");
        prep_prefill(&seen, N_SCAN, ROW_PITCH).expect("a launch");

        let calls = seen.calls.borrow();
        // The slot, and the whole list's length after it: three of the four
        // end at the eleven geometry scalars and the prefill prep carries
        // `row_pitch` and `n_scan` past them.
        for ((at, wide), (fire, args)) in [(11usize, 23usize), (12, 24), (10, 22), (12, 26)]
            .iter()
            .zip(calls.iter())
        {
            assert_eq!(
                args[*at],
                ArgValue::Buffer(77),
                "`{}` takes the slot map at {at}",
                fire.entrypoint
            );
            assert_eq!(
                args.len(),
                *wide,
                "`{}` binds {wide} slots",
                fire.entrypoint
            );
            assert_eq!(
                args[at + 1..at + 12],
                geometry(HV, DV)[..],
                "`{}` puts the eleven geometry scalars straight after the \
                 seat map, in `GdnShape::params` order",
                fire.entrypoint
            );
        }
        assert_eq!(
            calls[3].1[24..],
            [ArgValue::I32(ROW_PITCH), ArgValue::I32(N_SCAN)],
            "and those two are the prefill prep's, at 24 and 25 -- the scan \
             puts the same pair at 22 and 23"
        );
    }

    /// Every extent is refused BY NAME when it is zero, and nothing is
    /// encoded.
    ///
    /// A dispatch of no threads runs nothing and reports success, which leaves
    /// `core_out` holding the previous token's answer and `rstate` un-advanced
    /// -- a wrong answer and not a missing one. The names are separate because
    /// the mistakes are: no rows is an empty fire, no `v_heads` is a model
    /// whose head count never reached the driver, and no `v_dim` is a value
    /// width that did not either.
    #[test]
    fn a_zero_extent_is_refused_by_name_and_never_dispatched() {
        let seen = Seen::default();
        assert_eq!(core(&seen, 0, HV, DV), Err(Refusal::Empty { what: "rows" }));
        assert_eq!(
            core(&seen, ROWS, 0, DV),
            Err(Refusal::Empty { what: "v_heads" })
        );
        assert_eq!(
            core(&seen, ROWS, HV, 0),
            Err(Refusal::Empty { what: "v_dim" })
        );
        assert_eq!(
            prep_prefill(&seen, 0, ROW_PITCH),
            Err(Refusal::Empty { what: "n_scan" }),
            "named `n_scan` and not `rows`: they are the same product and a \
             caller that scanned nothing wants to be told which"
        );
        assert_eq!(
            prep_prefill(&seen, N_SCAN, 0),
            Err(Refusal::Empty { what: "row_pitch" }),
            "a zero pitch stacks every token's scratch on token zero's, and \
             the scan would read token zero's q and k for the whole prompt"
        );
        assert_eq!(scan(&seen, 0, 32, 4), Err(Refusal::Empty { what: "v_dim" }));
        assert!(seen.calls.borrow().is_empty(), "and none of them was encoded");
    }

    /// Fire every kernel in the family once, for the assertions that hold over
    /// all eight.
    fn every_kernel(seen: &Seen) {
        core(seen, ROWS, HV, DV).expect("a launch");
        core_slotted(seen, HV, DV).expect("a launch");
        prep(seen, HV, DV).expect("a launch");
        prep_slotted(seen, HV, DV).expect("a launch");
        recurrent(seen, HV, DV).expect("a launch");
        recurrent_slotted(seen, HV, DV).expect("a launch");
        prep_prefill(seen, N_SCAN, ROW_PITCH).expect("a launch");
        scan(seen, DV, 32, 4).expect("a launch");
    }

    /// The fused core over one fixture, so a test that varies the extents says
    /// only what it varies.
    ///
    /// `rows` is asked in the body and set on the probe; the head count and
    /// the value width are STATED on the call, at words 3 and 1 of the
    /// statement's run, where they used to be probe cells the body asked for.
    fn core(seen: &Seen, rows: i32, v_heads: i32, v_dim: i32) -> Result<(), Refusal> {
        seen.rows.set(rows);
        gdn_core(
            seen,
            In::new(Tensor::<bf16>::new(1)),
            Out::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<bf16>::new(5)),
            Const::new(Tensor::<bf16>::new(6)),
            Const::new(Tensor::<f32>::new(7)),
            Const::new(Tensor::<bf16>::new(8)),
            In::new(Tensor::<bf16>::new(9)),
            In::new(Tensor::<bf16>::new(10)),
            Const::new(K_DIM),
            Const::new(v_dim),
            Const::new(K_HEADS),
            Const::new(v_heads),
            Const::new(CONV_DIM),
            Const::new(CONV_K),
            Const::new(Q_OFF),
            Const::new(K_OFF),
            Const::new(V_OFF),
            Const::new(EPS),
            Const::new(INV_SQRT_DK))
    }

    /// [`gdn_core_slotted`] over the same fixture.
    fn core_slotted(seen: &Seen, v_heads: i32, v_dim: i32) -> Result<(), Refusal> {
        gdn_core_slotted(
            seen,
            In::new(Tensor::<bf16>::new(1)),
            Out::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<bf16>::new(5)),
            Const::new(Tensor::<bf16>::new(6)),
            Const::new(Tensor::<f32>::new(7)),
            Const::new(Tensor::<bf16>::new(8)),
            In::new(Tensor::<bf16>::new(9)),
            In::new(Tensor::<bf16>::new(10)),
            Const::new(K_DIM),
            Const::new(v_dim),
            Const::new(K_HEADS),
            Const::new(v_heads),
            Const::new(CONV_DIM),
            Const::new(CONV_K),
            Const::new(Q_OFF),
            Const::new(K_OFF),
            Const::new(V_OFF),
            Const::new(EPS),
            Const::new(INV_SQRT_DK))
    }

    /// [`gdn_prep`] over the same fixture.
    fn prep(seen: &Seen, v_heads: i32, v_dim: i32) -> Result<(), Refusal> {
        gdn_prep(
            seen,
            In::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(3)),
            Const::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<f32>::new(5)),
            Const::new(Tensor::<bf16>::new(6)),
            In::new(Tensor::<bf16>::new(7)),
            In::new(Tensor::<bf16>::new(8)),
            Out::new(Tensor::<f32>::new(9)),
            Out::new(Tensor::<f32>::new(10)),
            Out::new(Tensor::<f32>::new(11)),
            Const::new(K_DIM),
            Const::new(v_dim),
            Const::new(K_HEADS),
            Const::new(v_heads),
            Const::new(CONV_DIM),
            Const::new(CONV_K),
            Const::new(Q_OFF),
            Const::new(K_OFF),
            Const::new(V_OFF),
            Const::new(EPS),
            Const::new(INV_SQRT_DK))
    }

    /// [`gdn_prep_slotted`] over the same fixture.
    fn prep_slotted(seen: &Seen, v_heads: i32, v_dim: i32) -> Result<(), Refusal> {
        gdn_prep_slotted(
            seen,
            In::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(3)),
            Const::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<f32>::new(5)),
            Const::new(Tensor::<bf16>::new(6)),
            In::new(Tensor::<bf16>::new(7)),
            In::new(Tensor::<bf16>::new(8)),
            Out::new(Tensor::<f32>::new(9)),
            Out::new(Tensor::<f32>::new(10)),
            Out::new(Tensor::<f32>::new(11)),
            Const::new(K_DIM),
            Const::new(v_dim),
            Const::new(K_HEADS),
            Const::new(v_heads),
            Const::new(CONV_DIM),
            Const::new(CONV_K),
            Const::new(Q_OFF),
            Const::new(K_OFF),
            Const::new(V_OFF),
            Const::new(EPS),
            Const::new(INV_SQRT_DK))
    }

    /// [`gdn_core_recurrent`] over the same fixture.
    fn recurrent(seen: &Seen, v_heads: i32, v_dim: i32) -> Result<(), Refusal> {
        gdn_core_recurrent(
            seen,
            In::new(Tensor::<bf16>::new(1)),
            Out::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<bf16>::new(5)),
            Const::new(Tensor::<bf16>::new(6)),
            In::new(Tensor::<f32>::new(7)),
            In::new(Tensor::<f32>::new(8)),
            In::new(Tensor::<f32>::new(9)),
            Const::new(K_DIM),
            Const::new(v_dim),
            Const::new(K_HEADS),
            Const::new(v_heads),
            Const::new(CONV_DIM),
            Const::new(CONV_K),
            Const::new(Q_OFF),
            Const::new(K_OFF),
            Const::new(V_OFF),
            Const::new(EPS),
            Const::new(INV_SQRT_DK))
    }

    /// [`gdn_core_recurrent_slotted`] over the same fixture.
    fn recurrent_slotted(seen: &Seen, v_heads: i32, v_dim: i32) -> Result<(), Refusal> {
        gdn_core_recurrent_slotted(
            seen,
            In::new(Tensor::<bf16>::new(1)),
            Out::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<bf16>::new(5)),
            Const::new(Tensor::<bf16>::new(6)),
            In::new(Tensor::<f32>::new(7)),
            In::new(Tensor::<f32>::new(8)),
            In::new(Tensor::<f32>::new(9)),
            Const::new(K_DIM),
            Const::new(v_dim),
            Const::new(K_HEADS),
            Const::new(v_heads),
            Const::new(CONV_DIM),
            Const::new(CONV_K),
            Const::new(Q_OFF),
            Const::new(K_OFF),
            Const::new(V_OFF),
            Const::new(EPS),
            Const::new(INV_SQRT_DK))
    }

    /// The prefill prep: `n_scan` is set on the probe as `Rows` (the same fact
    /// a decode counts rows with), and `row_pitch` rides `mixed`'s own `width`
    /// rather than a separate ask, exactly as the body reads it.
    fn prep_prefill(seen: &Seen, n_scan: i32, row_pitch: i32) -> Result<(), Refusal> {
        seen.rows.set(n_scan);
        gdn_prep_prefill(
            seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: row_pitch },
            Const::new(Tensor::<bf16>::new(3)),
            Const::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<f32>::new(5)),
            Const::new(Tensor::<bf16>::new(6)),
            In::new(Tensor::<bf16>::new(7)),
            In::new(Tensor::<bf16>::new(8)),
            Out::new(Tensor::<f32>::new(9)),
            Out::new(Tensor::<f32>::new(10)),
            Out::new(Tensor::<f32>::new(11)),
            Const::new(K_DIM),
            Const::new(DV),
            Const::new(K_HEADS),
            Const::new(HV),
            Const::new(CONV_DIM),
            Const::new(CONV_K),
            Const::new(Q_OFF),
            Const::new(K_OFF),
            Const::new(V_OFF),
            Const::new(EPS),
            Const::new(INV_SQRT_DK))
    }

    /// The scan, with the pad handle a number no real buffer in these tests
    /// takes, so a slot holding it is unmistakable.
    ///
    /// `n_scan` is fixed at [`N_SCAN`] (the body reads it off the probe's
    /// `Rows`) and `row_pitch` rides `pre_q`'s own `width`, exactly as the
    /// body reads it. The pad is given a DIFFERENT width -- one no pitch in
    /// this family could be -- so that a body that went back to reading the
    /// filler's shape is a failing assertion rather than a passing one.
    /// `lanes` and `vrows` are STATED: they are words 11 and 12 of this
    /// entrypoint's own thirteen-word statement run -- `model-dsl` pushes the
    /// tile onto `GdnShape::params` -- and the body takes them as `Const`
    /// marks where it used to read them by index through `Asks::param`.
    fn scan(seen: &Seen, v_dim: i32, lanes: i32, vrows: i32) -> Result<(), Refusal> {
        scan_with(seen, v_dim, lanes, vrows, PAD_WIDTH, ROW_PITCH)
    }

    /// [`scan`], with the two operand widths said out loud.
    fn scan_with(
        seen: &Seen,
        v_dim: i32,
        lanes: i32,
        vrows: i32,
        pad_width: i32,
        pre_q_width: i32,
    ) -> Result<(), Refusal> {
        seen.rows.set(N_SCAN);
        gdn_core_recurrent_prefill(
            seen,
            In { ptr: Tensor::<bf16>::new(90), rows: 0, width: pad_width },
            Out::new(Tensor::<bf16>::new(3)),
            In { ptr: Tensor::<f32>::new(6), rows: 0, width: pre_q_width },
            In::new(Tensor::<f32>::new(7)),
            In::new(Tensor::<f32>::new(8)),
            Const::new(K_DIM),
            Const::new(v_dim),
            Const::new(K_HEADS),
            Const::new(HV),
            Const::new(CONV_DIM),
            Const::new(CONV_K),
            Const::new(Q_OFF),
            Const::new(K_OFF),
            Const::new(V_OFF),
            Const::new(EPS),
            Const::new(INV_SQRT_DK),
            Const::new(lanes),
            Const::new(vrows))
    }
}
