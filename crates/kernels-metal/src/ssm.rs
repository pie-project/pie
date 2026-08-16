//! Gated DeltaNet: the recurrent state kernels and their prep.
//!
//! `gdn` is an algorithm and not a model, so it takes no model qualifier --
//! the same call the CUDA table makes for `delta_attn_kda` and `indexer_dsa`.

// The fused core binds twelve buffers and its slotted form thirteen, which the
// row below already counted before any of them was written down. Gathering
// them into a struct would restate that binding order somewhere a shader
// cannot check, which is the thing this refactor removes.
#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, F32s, F32sMut, Fire, Routine, U32s};

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
    ("ssm/gdn_core.metal", "gdn_core_bfloat16"),
    ("ssm/gdn_prep.metal", "gdn_core_recurrent_bfloat16"),
    (
        "ssm/gdn_prep.metal",
        "gdn_core_recurrent_prefill_bfloat16_l_16_v_1",
    ),
    (
        "ssm/gdn_prep.metal",
        "gdn_core_recurrent_prefill_bfloat16_l_16_v_2",
    ),
    (
        "ssm/gdn_prep.metal",
        "gdn_core_recurrent_prefill_bfloat16_l_16_v_4",
    ),
    (
        "ssm/gdn_prep.metal",
        "gdn_core_recurrent_prefill_bfloat16_l_32_v_2",
    ),
    (
        "ssm/gdn_prep.metal",
        "gdn_core_recurrent_prefill_bfloat16_l_32_v_4",
    ),
    (
        "ssm/gdn_prep.metal",
        "gdn_core_recurrent_prefill_bfloat16_l_32_v_8",
    ),
    (
        "ssm/gdn_prep.metal",
        "gdn_core_recurrent_prefill_bfloat16_l_4_v_1",
    ),
    (
        "ssm/gdn_prep.metal",
        "gdn_core_recurrent_prefill_bfloat16_l_8_v_1",
    ),
    (
        "ssm/gdn_prep.metal",
        "gdn_core_recurrent_prefill_bfloat16_l_8_v_2",
    ),
    ("ssm/gdn_prep.metal", "gdn_core_recurrent_slotted_bfloat16"),
    ("ssm/gdn_core.metal", "gdn_core_slotted_bfloat16"),
    ("ssm/gdn_prep.metal", "gdn_prep_bfloat16"),
    ("ssm/gdn_prep.metal", "gdn_prep_prefill_bfloat16"),
    ("ssm/gdn_prep.metal", "gdn_prep_slotted_bfloat16"),
];

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
/// `conv_state` is [`Buf`] and `new_conv_state` is [`BufMut`] because the two
/// cannot be one allocation: `convsilu` reads the `Kc`-tap history while the
/// writeback shifts it, and the redundant dv threadgroups interleave those
/// reads and writes. The shader's own header calls that out; the types are
/// where a caller that tried to alias them fails to compile.
///
/// `params` is the packed `GdnCoreParams` at buffer 11 -- eleven four-byte
/// fields, the way `layout::ple_combine` takes `PleCombineParams` -- and it is
/// where `Dk`, `Hk`, `Kc` and the three region offsets arrive. `v_heads` and
/// `v_dim` are arguments as WELL, because the grid needs both and the kernel
/// reads both out of that block: two statements of one number, and only the
/// caller can make them agree.
///
/// # Errors
///
/// See [`core_grid`].
pub fn gdn_core(
    ctx: &Ctx<'_>,
    mixed: Buf,
    conv_state: F32s,
    rstate: F32sMut,
    core_out: BufMut,
    conv_w: Buf,
    conv_b: Buf,
    a_log: F32s,
    dt_bias: Buf,
    a_gate: Buf,
    b_gate: Buf,
    new_conv_state: F32sMut,
    params: Buf,
    rows: Env<i32>,
    v_heads: Env<i32>,
    v_dim: Env<i32>,
) -> Result<(), Refusal> {
    let grid = core_grid(*rows, *v_heads, *v_dim)?;
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_core_bfloat16",
            file: CORE_FILE,
            lanes: grid,
            group: core_group(grid),
        },
        &[
            mixed.v(),
            conv_state.v(),
            rstate.v(),
            core_out.v(),
            conv_w.v(),
            conv_b.v(),
            a_log.v(),
            dt_bias.v(),
            a_gate.v(),
            b_gate.v(),
            new_conv_state.v(),
            params.v(),
        ],
    )
}

/// [`gdn_core`] over rows whose persistent state is not where the row is.
///
/// One buffer more, at 12, and ONLY the state accesses remap through it: the
/// conv slab and the recurrent slab take `slot_ids[b_idx]` while `mixed`,
/// `core_out`, `a_gate` and `b_gate` stay token-major (`gdn_core.metal:90`).
/// That asymmetry is the whole of the slotted seam and it is why the two
/// symbols take the same twelve buffers in the same twelve slots.
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
pub fn gdn_core_slotted(
    ctx: &Ctx<'_>,
    mixed: Buf,
    conv_state: F32s,
    rstate: F32sMut,
    core_out: BufMut,
    conv_w: Buf,
    conv_b: Buf,
    a_log: F32s,
    dt_bias: Buf,
    a_gate: Buf,
    b_gate: Buf,
    new_conv_state: F32sMut,
    params: Buf,
    slot_ids: U32s,
    rows: Env<i32>,
    v_heads: Env<i32>,
    v_dim: Env<i32>,
) -> Result<(), Refusal> {
    let grid = core_grid(*rows, *v_heads, *v_dim)?;
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_core_slotted_bfloat16",
            file: CORE_FILE,
            lanes: grid,
            group: core_group(grid),
        },
        &[
            mixed.v(),
            conv_state.v(),
            rstate.v(),
            core_out.v(),
            conv_w.v(),
            conv_b.v(),
            a_log.v(),
            dt_bias.v(),
            a_gate.v(),
            b_gate.v(),
            new_conv_state.v(),
            params.v(),
            slot_ids.v(),
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
pub fn gdn_prep(
    ctx: &Ctx<'_>,
    mixed: Buf,
    conv_state: F32s,
    conv_w: Buf,
    conv_b: Buf,
    a_log: F32s,
    dt_bias: Buf,
    a_gate: Buf,
    b_gate: Buf,
    pre_q: F32sMut,
    pre_k: F32sMut,
    pre_gate: F32sMut,
    new_conv_state: F32sMut,
    params: Buf,
    rows: Env<i32>,
    v_heads: Env<i32>,
) -> Result<(), Refusal> {
    let grid = prep_grid(*rows, *v_heads)?;
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_prep_bfloat16",
            file: PREP_FILE,
            lanes: grid,
            group: simd_group(grid),
        },
        &[
            mixed.v(),
            conv_state.v(),
            conv_w.v(),
            conv_b.v(),
            a_log.v(),
            dt_bias.v(),
            a_gate.v(),
            b_gate.v(),
            pre_q.v(),
            pre_k.v(),
            pre_gate.v(),
            new_conv_state.v(),
            params.v(),
        ],
    )
}

/// [`gdn_prep`] with the state slabs reached through a slot map.
///
/// `slot_ids` lands at 13 here and at 12 in [`gdn_core_slotted`] and 11 in
/// [`gdn_core_recurrent_slotted`]: one buffer, three slots, because each
/// entrypoint numbers its own list. That is exactly the kind of fact a shared
/// binding struct would have had to get right three times.
///
/// # Errors
///
/// See [`head_rows`].
pub fn gdn_prep_slotted(
    ctx: &Ctx<'_>,
    mixed: Buf,
    conv_state: F32s,
    conv_w: Buf,
    conv_b: Buf,
    a_log: F32s,
    dt_bias: Buf,
    a_gate: Buf,
    b_gate: Buf,
    pre_q: F32sMut,
    pre_k: F32sMut,
    pre_gate: F32sMut,
    new_conv_state: F32sMut,
    params: Buf,
    slot_ids: U32s,
    rows: Env<i32>,
    v_heads: Env<i32>,
) -> Result<(), Refusal> {
    let grid = prep_grid(*rows, *v_heads)?;
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_prep_slotted_bfloat16",
            file: PREP_FILE,
            lanes: grid,
            group: simd_group(grid),
        },
        &[
            mixed.v(),
            conv_state.v(),
            conv_w.v(),
            conv_b.v(),
            a_log.v(),
            dt_bias.v(),
            a_gate.v(),
            b_gate.v(),
            pre_q.v(),
            pre_k.v(),
            pre_gate.v(),
            new_conv_state.v(),
            params.v(),
            slot_ids.v(),
        ],
    )
}

/// The per-dv half: the recurrent read-modify-write, with q, k and the gates
/// read BACK from [`gdn_prep`]'s scratch instead of recomputed.
///
/// The same grid as [`gdn_core`] -- `[32, Dv, rows * Hv]`, `device_gdn.rs:559`
/// -- over eleven buffers instead of twelve. `A_log`, `dt_bias`, `a_gate` and
/// `b_gate` are gone, `pre_q`, `pre_k` and `pre_gate` take 6, 7 and 8, and
/// `new_conv_state` moves from 10 to 9. The first six slots agree with
/// [`gdn_core`]'s and nothing after them does, which is why the two cannot
/// share an argument list however alike they look.
///
/// The scratch is [`Buf`] here and [`BufMut`] in the prep, so its direction is
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
pub fn gdn_core_recurrent(
    ctx: &Ctx<'_>,
    mixed: Buf,
    conv_state: F32s,
    rstate: F32sMut,
    core_out: BufMut,
    conv_w: Buf,
    conv_b: Buf,
    pre_q: F32s,
    pre_k: F32s,
    pre_gate: F32s,
    new_conv_state: F32sMut,
    params: Buf,
    rows: Env<i32>,
    v_heads: Env<i32>,
    v_dim: Env<i32>,
) -> Result<(), Refusal> {
    let grid = core_grid(*rows, *v_heads, *v_dim)?;
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_core_recurrent_bfloat16",
            file: PREP_FILE,
            lanes: grid,
            group: core_group(grid),
        },
        &[
            mixed.v(),
            conv_state.v(),
            rstate.v(),
            core_out.v(),
            conv_w.v(),
            conv_b.v(),
            pre_q.v(),
            pre_k.v(),
            pre_gate.v(),
            new_conv_state.v(),
            params.v(),
        ],
    )
}

/// [`gdn_core_recurrent`] with the state slabs reached through a slot map.
///
/// `slot_ids` at 11, immediately after `params` at 10 -- the lowest of the
/// three slots the same buffer takes across this family.
///
/// # Errors
///
/// See [`core_grid`].
pub fn gdn_core_recurrent_slotted(
    ctx: &Ctx<'_>,
    mixed: Buf,
    conv_state: F32s,
    rstate: F32sMut,
    core_out: BufMut,
    conv_w: Buf,
    conv_b: Buf,
    pre_q: F32s,
    pre_k: F32s,
    pre_gate: F32s,
    new_conv_state: F32sMut,
    params: Buf,
    slot_ids: U32s,
    rows: Env<i32>,
    v_heads: Env<i32>,
    v_dim: Env<i32>,
) -> Result<(), Refusal> {
    let grid = core_grid(*rows, *v_heads, *v_dim)?;
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_core_recurrent_slotted_bfloat16",
            file: PREP_FILE,
            lanes: grid,
            group: core_group(grid),
        },
        &[
            mixed.v(),
            conv_state.v(),
            rstate.v(),
            core_out.v(),
            conv_w.v(),
            conv_b.v(),
            pre_q.v(),
            pre_k.v(),
            pre_gate.v(),
            new_conv_state.v(),
            params.v(),
            slot_ids.v(),
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
/// `row_pitch` and `n_scan` are separate `constant int&` operands at 14 and
/// 15, AFTER the packed `params` at 12 -- one dispatch carrying a packed
/// struct and two loose scalars. [`gdn_core_recurrent_prefill`] puts the same
/// two at 12 and 13, so the two prefill kernels share neither numbering nor a
/// binding order, and a body that copied one into the other would write the
/// pitch where the parameter block goes.
///
/// Both are [`Env`]: `row_pitch` is the prefill scratch layout's widest
/// tensor in activation elements (`gdn_prep.metal:329`) and `n_scan` is the
/// fire's token count. Neither is a number a text states.
///
/// # Errors
///
/// [`Refusal::Empty`] naming `n_scan` before [`head_rows`] can name it
/// `rows` -- the two are the same product and a caller that scanned nothing
/// wants to be told which one it was. Also [`Refusal::Empty`] for a
/// `row_pitch` of zero, which is not the same mistake: every token's scratch
/// row would land on token zero's, the scan would read token zero's q and k
/// for the whole prompt, and the prompt would come back a plausible length.
pub fn gdn_prep_prefill(
    ctx: &Ctx<'_>,
    mixed: Buf,
    conv_state: F32s,
    conv_w: Buf,
    conv_b: Buf,
    a_log: F32s,
    dt_bias: Buf,
    a_gate: Buf,
    b_gate: Buf,
    pre_q: F32sMut,
    pre_k: F32sMut,
    pre_gate: F32sMut,
    new_conv_state: F32sMut,
    params: Buf,
    slot_ids: U32s,
    row_pitch: Env<i32>,
    n_scan: Env<i32>,
    v_heads: Env<i32>,
) -> Result<(), Refusal> {
    if *n_scan <= 0 {
        return Err(Refusal::Empty { what: "n_scan" });
    }
    if *row_pitch <= 0 {
        return Err(Refusal::Empty { what: "row_pitch" });
    }
    let grid = prep_grid(*n_scan, *v_heads)?;
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_prep_prefill_bfloat16",
            file: PREP_FILE,
            lanes: grid,
            group: simd_group(grid),
        },
        &[
            mixed.v(),
            conv_state.v(),
            conv_w.v(),
            conv_b.v(),
            a_log.v(),
            dt_bias.v(),
            a_gate.v(),
            b_gate.v(),
            pre_q.v(),
            pre_k.v(),
            pre_gate.v(),
            new_conv_state.v(),
            params.v(),
            slot_ids.v(),
            row_pitch.v(),
            n_scan.v(),
        ],
    )
}

/// The scan: the whole prompt's recurrence walked inside one kernel, with the
/// state living in registers for the duration.
///
/// Nine entrypoints over one template, and [`scan_point`] picks the spelling.
/// `lanes` and `vrows` are [`Env`] for the same reason `layout`'s `group` and
/// `bits` are: they are an instantiation axis the environment chooses --
/// `driver-metal`'s `Tuning` defaults to 32 and 4 and reads both from the
/// environment -- and no text names them.
///
/// # The five pad slots
///
/// This entrypoint declares buffers at 2, 3, 6, 7, 8, 10, 11, 12 and 13 and
/// **nothing at 0, 1, 4, 5 or 9** (`ssm/gdn_prep.metal:474`). The numbering is
/// [`gdn_core_recurrent`]'s, kept so the two can be encoded against one
/// argument table; the scan simply needs none of `mixed`, `conv_state`,
/// `conv_w`, `conv_b` or `new_conv_state`, because the prep already did every
/// convolution and wrote the history forward.
///
/// A Metal argument table is a contiguous run, so the five holes still have to
/// hold an address: `device_gdn.rs:1206`'s `fill` gives all fourteen
/// `core_out`'s address and then overwrites the nine it binds. A routine's
/// argument list is positional -- the index in the list IS the buffer slot --
/// so `pad` is taken once and bound at each hole. Skipping them would slide
/// `params` into slot 6 and hand the scan its own geometry where it reads q.
///
/// `pad` is [`Buf`]: nothing dereferences it, and a read-only handle is the
/// weakest claim that can fill a slot.
///
/// # Errors
///
/// [`Refusal::Narrow`] from [`scan_point`] for a tiling the shader tree does
/// not carry, and [`Refusal::Empty`] from [`scan_grid`] for a zero extent or
/// from the two prefill scalars, as [`gdn_prep_prefill`].
///
/// The tiling is checked FIRST: an entrypoint Metal has no `[[host_name]]`
/// for makes `newFunctionWithName:` return nil at run time, inside a fire,
/// after the plan was accepted -- so it has to be a refusal here, where the
/// caller can still step down to a tiling that exists.
pub fn gdn_core_recurrent_prefill(
    ctx: &Ctx<'_>,
    pad: Buf,
    rstate: F32sMut,
    core_out: BufMut,
    pre_q: F32s,
    pre_k: F32s,
    pre_gate: F32s,
    params: Buf,
    slot_ids: U32s,
    row_pitch: Env<i32>,
    n_scan: Env<i32>,
    v_heads: Env<i32>,
    v_dim: Env<i32>,
    lanes: Env<i32>,
    vrows: Env<i32>,
) -> Result<(), Refusal> {
    let point = scan_point(*lanes, *vrows)?;
    if *n_scan <= 0 {
        return Err(Refusal::Empty { what: "n_scan" });
    }
    if *row_pitch <= 0 {
        return Err(Refusal::Empty { what: "row_pitch" });
    }
    let grid = scan_grid(*v_dim, *v_heads, *lanes, *vrows)?;
    ctx.dispatch(
        Fire {
            entrypoint: GDN_SCAN[point],
            file: PREP_FILE,
            lanes: grid,
            group: simd_group(grid),
        },
        &[
            pad.v(),
            pad.v(),
            rstate.v(),
            core_out.v(),
            pad.v(),
            pad.v(),
            pre_q.v(),
            pre_k.v(),
            pre_gate.v(),
            pad.v(),
            params.v(),
            slot_ids.v(),
            row_pitch.v(),
            n_scan.v(),
        ],
    )
}

/// This family's routines.
///
/// All eight, and none of them states `in_place` though the five that carry
/// `rstate` look like they should. `rstate` IS read and written -- `gdn_core.metal:149`
/// loads the row and `:160` stores it back -- but `in_place` names
/// `(input, output)` pairs of TRACE OPERANDS that must be given one address,
/// and the recurrent state arrives as a single slab the kernel rolls. There is
/// no input operand to alias it to. What makes it in place is that the buffer
/// is [`BufMut`], which the signature already says.
///
/// `conv_state` and `new_conv_state` are the opposite case and the reason the
/// distinction matters: they are two buffers that must NOT be one, and the
/// pair of types says so.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(gdn_core),
    crate::routine!(gdn_core_recurrent),
    crate::routine!(gdn_core_recurrent_prefill),
    crate::routine!(gdn_core_recurrent_slotted),
    crate::routine!(gdn_core_slotted),
    crate::routine!(gdn_prep),
    crate::routine!(gdn_prep_prefill),
    crate::routine!(gdn_prep_slotted),
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

    /// `device_gdn.rs`'s own fixture, so the grids below are comparable to the
    /// literals that test hands the device.
    const ROWS: i32 = 2;
    const HV: i32 = 4;
    const DV: i32 = 8;
    /// `T_SCAN` and `ROW_PITCH` there.
    const N_SCAN: i32 = 5;
    const ROW_PITCH: i32 = 1100;

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
        gdn_core(
            &seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            Buf(8),
            Buf(9),
            Buf(10),
            F32sMut(11),
            Buf(12),
            Env(ROWS),
            Env(HV),
            Env(DV),
        )
        .expect("a launch");
        gdn_core_slotted(
            &seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            Buf(8),
            Buf(9),
            Buf(10),
            F32sMut(11),
            Buf(12),
            U32s(13),
            Env(ROWS),
            Env(HV),
            Env(DV),
        )
        .expect("a launch");
        gdn_core_recurrent(
            &seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            F32s(8),
            F32s(9),
            F32sMut(10),
            Buf(11),
            Env(ROWS),
            Env(HV),
            Env(DV),
        )
        .expect("a launch");

        for (fire, _) in seen.0.borrow().iter() {
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
        for (fire, _) in seen.0.borrow().iter() {
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
        gdn_prep(
            &seen,
            Buf(1),
            F32s(2),
            Buf(3),
            Buf(4),
            F32s(5),
            Buf(6),
            Buf(7),
            Buf(8),
            F32sMut(9),
            F32sMut(10),
            F32sMut(11),
            F32sMut(12),
            Buf(13),
            Env(ROWS),
            Env(HV),
        )
        .expect("a launch");
        gdn_core_recurrent(
            &seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            F32s(8),
            F32s(9),
            F32sMut(10),
            Buf(11),
            Env(ROWS),
            Env(HV),
            Env(DV),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
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
        let calls = seen.0.borrow();
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
            let calls = seen.0.borrow();
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
        assert_eq!(seen.0.borrow()[0].0.lanes[1], 17);
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
        assert!(seen.0.borrow().is_empty(), "and none of them was encoded");

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
    /// the nine it does declare land where the shader put them.
    ///
    /// `gdn_prep.metal:474` declares 2, 3, 6, 7, 8, 10, 11, 12, 13 and nothing
    /// at 0, 1, 4, 5 or 9 -- the numbering is `gdn_core_recurrent`'s, kept so
    /// both can be encoded against one argument table. A routine's argument
    /// list is positional, so closing the holes would slide `params` into slot
    /// 6 and hand the scan its own geometry where it reads q.
    #[test]
    fn the_scan_binds_a_pad_at_every_slot_its_entrypoint_does_not_declare() {
        let seen = Seen::default();
        scan(&seen, DV, 32, 4).expect("a launch");
        let calls = seen.0.borrow();
        let args = &calls[0].1;
        assert_eq!(args.len(), 14, "buffer 13 is the last one declared");
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
            [ArgValue::Buffer(2), ArgValue::Buffer(3)],
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
            args[10..],
            [
                ArgValue::Buffer(10),
                ArgValue::Buffer(11),
                ArgValue::I32(ROW_PITCH),
                ArgValue::I32(N_SCAN)
            ],
            "params, slot_ids, and the two loose scalars at 12 and 13 -- where \
             `gdn_prep_prefill` puts them at 14 and 15"
        );
    }

    /// One slot map, three slots.
    ///
    /// `slot_ids` is buffer 12 in `gdn_core_slotted`, 13 in `gdn_prep_slotted`
    /// and `gdn_prep_prefill`, and 11 in `gdn_core_recurrent_slotted`, because
    /// each entrypoint numbers its own list. It is the same buffer carrying
    /// the same meaning in all four, which is exactly the shape a shared
    /// binding struct gets wrong: only the STATE accesses remap through it
    /// (`gdn_core.metal:90`), so a misplaced one still reads activations that
    /// are there.
    #[test]
    fn the_slot_map_lands_where_each_entrypoint_declares_it() {
        let seen = Seen::default();
        gdn_core_slotted(
            &seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            Buf(8),
            Buf(9),
            Buf(10),
            F32sMut(11),
            Buf(12),
            U32s(77),
            Env(ROWS),
            Env(HV),
            Env(DV),
        )
        .expect("a launch");
        gdn_prep_slotted(
            &seen,
            Buf(1),
            F32s(2),
            Buf(3),
            Buf(4),
            F32s(5),
            Buf(6),
            Buf(7),
            Buf(8),
            F32sMut(9),
            F32sMut(10),
            F32sMut(11),
            F32sMut(12),
            Buf(13),
            U32s(77),
            Env(ROWS),
            Env(HV),
        )
        .expect("a launch");
        gdn_core_recurrent_slotted(
            &seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            F32s(8),
            F32s(9),
            F32sMut(10),
            Buf(11),
            U32s(77),
            Env(ROWS),
            Env(HV),
            Env(DV),
        )
        .expect("a launch");
        prep_prefill(&seen, N_SCAN, ROW_PITCH).expect("a launch");

        let calls = seen.0.borrow();
        // The slot, and the whole list's length after it: three of the four
        // end AT the slot map and the prefill prep carries `row_pitch` and
        // `n_scan` past it.
        for ((at, wide), (fire, args)) in [(12usize, 13usize), (13, 14), (11, 12), (13, 16)]
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
        }
        assert_eq!(
            calls[3].1[14..],
            [ArgValue::I32(ROW_PITCH), ArgValue::I32(N_SCAN)],
            "and those two are the prefill prep's, at 14 and 15 -- the scan \
             puts the same pair at 12 and 13"
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
        assert!(seen.0.borrow().is_empty(), "and none of them was encoded");
    }

    /// Fire every kernel in the family once, for the assertions that hold over
    /// all eight.
    fn every_kernel(seen: &Seen) {
        core(seen, ROWS, HV, DV).expect("a launch");
        gdn_core_slotted(
            seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            Buf(8),
            Buf(9),
            Buf(10),
            F32sMut(11),
            Buf(12),
            U32s(13),
            Env(ROWS),
            Env(HV),
            Env(DV),
        )
        .expect("a launch");
        gdn_prep(
            seen,
            Buf(1),
            F32s(2),
            Buf(3),
            Buf(4),
            F32s(5),
            Buf(6),
            Buf(7),
            Buf(8),
            F32sMut(9),
            F32sMut(10),
            F32sMut(11),
            F32sMut(12),
            Buf(13),
            Env(ROWS),
            Env(HV),
        )
        .expect("a launch");
        gdn_prep_slotted(
            seen,
            Buf(1),
            F32s(2),
            Buf(3),
            Buf(4),
            F32s(5),
            Buf(6),
            Buf(7),
            Buf(8),
            F32sMut(9),
            F32sMut(10),
            F32sMut(11),
            F32sMut(12),
            Buf(13),
            U32s(14),
            Env(ROWS),
            Env(HV),
        )
        .expect("a launch");
        gdn_core_recurrent(
            seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            F32s(8),
            F32s(9),
            F32sMut(10),
            Buf(11),
            Env(ROWS),
            Env(HV),
            Env(DV),
        )
        .expect("a launch");
        gdn_core_recurrent_slotted(
            seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            F32s(8),
            F32s(9),
            F32sMut(10),
            Buf(11),
            U32s(12),
            Env(ROWS),
            Env(HV),
            Env(DV),
        )
        .expect("a launch");
        prep_prefill(seen, N_SCAN, ROW_PITCH).expect("a launch");
        scan(seen, DV, 32, 4).expect("a launch");
    }

    /// The fused core over one fixture, so a test that varies the extents says
    /// only what it varies.
    fn core(seen: &Seen, rows: i32, v_heads: i32, v_dim: i32) -> Result<(), Refusal> {
        gdn_core(
            seen,
            Buf(1),
            F32s(2),
            F32sMut(3),
            BufMut(4),
            Buf(5),
            Buf(6),
            F32s(7),
            Buf(8),
            Buf(9),
            Buf(10),
            F32sMut(11),
            Buf(12),
            Env(rows),
            Env(v_heads),
            Env(v_dim),
        )
    }

    /// The prefill prep, likewise.
    fn prep_prefill(seen: &Seen, n_scan: i32, row_pitch: i32) -> Result<(), Refusal> {
        gdn_prep_prefill(
            seen,
            Buf(1),
            F32s(2),
            Buf(3),
            Buf(4),
            F32s(5),
            Buf(6),
            Buf(7),
            Buf(8),
            F32sMut(9),
            F32sMut(10),
            F32sMut(11),
            F32sMut(12),
            Buf(13),
            U32s(77),
            Env(row_pitch),
            Env(n_scan),
            Env(HV),
        )
    }

    /// The scan, with the pad handle a number no real buffer in these tests
    /// takes, so a slot holding it is unmistakable.
    fn scan(seen: &Seen, v_dim: i32, lanes: i32, vrows: i32) -> Result<(), Refusal> {
        gdn_core_recurrent_prefill(
            seen,
            Buf(90),
            F32sMut(2),
            BufMut(3),
            F32s(6),
            F32s(7),
            F32s(8),
            Buf(10),
            U32s(11),
            Env(ROW_PITCH),
            Env(N_SCAN),
            Env(HV),
            Env(v_dim),
            Env(lanes),
            Env(vrows),
        )
    }
}
