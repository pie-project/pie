//! Gated DeltaNet: the recurrent state kernels and their prep.
//!
//! `gdn` is an algorithm and not a model, so it takes no model qualifier --
//! the same call the CUDA table makes for `delta_attn_kda` and `indexer_dsa`.
#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{keys, Ask, Bind, Block, Buf, BufMut, Ctx, Env, F32s, F32sMut, Fire, Held, Param, Routine, U32s};
use crate::routine::{InSlot, OutSlot, Weight};

/// The entrypoints this family's crossed routines spell, now that their
/// rows are gone. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &[
    "gdn_core_bfloat16",
    "gdn_core_recurrent_bfloat16",
    "gdn_core_recurrent_prefill_bfloat16_l_16_v_1",
    "gdn_core_recurrent_prefill_bfloat16_l_16_v_2",
    "gdn_core_recurrent_prefill_bfloat16_l_16_v_4",
    "gdn_core_recurrent_prefill_bfloat16_l_32_v_2",
    "gdn_core_recurrent_prefill_bfloat16_l_32_v_4",
    "gdn_core_recurrent_prefill_bfloat16_l_32_v_8",
    "gdn_core_recurrent_prefill_bfloat16_l_4_v_1",
    "gdn_core_recurrent_prefill_bfloat16_l_8_v_1",
    "gdn_core_recurrent_prefill_bfloat16_l_8_v_2",
    "gdn_core_recurrent_slotted_bfloat16",
    "gdn_core_slotted_bfloat16",
    "gdn_prep_bfloat16",
    "gdn_prep_prefill_bfloat16",
    "gdn_prep_slotted_bfloat16",
];

// Eight entrypoints here name a shader and cannot be reached, and the rows
// that used to say so are gone. The reason is not the rows and did not leave
// with them, so it is kept:
//
// The sixteen modules behind them are compiled into every native build and
// load on this device. What is missing is a SEAM, not memory and not a
// signature. The tracer's GDN vocabulary emits THREE ops -- `causal_conv1d`,
// then `gdn_prep` over `(q, k, v, g, beta)`, then a step or prefill scan --
// where `gdn_core` here is ONE dispatch with the convolution, the prep and the
// recurrence fused. So a `dsl::vulkan::gdn_*` cannot be added the way a second
// attention name was; it needs a `TraceBuilder` row, a lowering and a
// statement. `gdn_prep` and `gdn_core_recurrent` split the work one seam
// further and are the closer pair to start from.
//
// The widths were read out of the SPIR-V and are not the obstacle: twelve
// buffers for `gdn_core`, thirteen for `gdn_core_slotted` and `gdn_prep`,
// fourteen for the prefill and slotted preps, eleven for
// `gdn_core_recurrent`, seven for each of the nine prefill shapes. The first
// two are the same twelve and thirteen `kernels-metal` names.
//
// What is upstream of the count is that three of the buffers (`conv_state`,
// `rstate`, `new_conv_state`) are recurrent STATE, and a guessed `Source` is
// worse than an absent one -- an absent one refuses, a wrong one binds.

/// The nine `(LANES, VROWS)` shapes the prefill scan is compiled for, in the
/// order the row's axis names them.
///
/// A literal table and not a `format!`, for the reason
/// `layout::EMBED_GATHER` is one: an unbuilt module is not an error on this
/// backend. `vkCreateComputePipelines` SIGSEGVs on a name that was never
/// compiled and the validation layer says nothing, so a composed name is a
/// crash and a written one is a link error.
///
/// Not every pair of a plausible `LANES` and a plausible `VROWS` is here.
/// `(32, 1)` and `(4, 2)` are as sensible-looking as any of the nine and
/// neither exists, which is what [`scan_point`] is for.
static SCAN: [&str; 9] = [
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

/// Which of the nine compiled scan shapes a `(lanes, vrows)` pair is.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a pair that is not one of the nine. The two
/// numbers are not independent -- the tiling axis is a LIST of nine points and
/// not a product of three lane widths and four row counts -- so this is a
/// match over pairs rather than two lookups multiplied together. `(32, 1)`
/// reads as an obvious member of a product and is not compiled.
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
        _ => Err(Refusal::Narrow {
            what: "no gdn scan is compiled for this (LANES, VROWS)",
            at: i64::from(lanes) * 100 + i64::from(vrows),
        }),
    }
}

/// The grid every gdn kernel except the prefill scan fires on.
///
/// `[32, dv, rows]`, in THREADS. The x extent is 32 because the reduction is
/// over exactly 32 lanes: `row_sum32` halves a 32-wide shared-memory window
/// and the `st[v][32]` register slabs are 32 deep, so this is the shader's
/// shape and not a tuning choice. `dv` is one lane row per value channel and
/// `rows` is one z group per `(request, value head)` pair.
///
/// `tests/gpu.rs` fires these at `[1, dv / 4, b * hv]` WORKGROUPS, which is
/// the same grid seen from the other side of `[numthreads(32, 4, 1)]`. The
/// `/ 4` there is exact division and would drop the tail of a `dv` that four
/// does not divide; stated in threads, the division is `driver-vulkan`'s
/// `div_ceil` and the tail is covered.
///
/// # Errors
///
/// [`Refusal::Empty`] for a zero or negative extent. A gdn launch over no
/// value channels or no heads is a launch that reads uninitialised recurrent
/// state and writes it back.
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

/// The grid a per-dv core takes: one warp per `(row, v-head, v-channel)`.
///
/// `[32, Dv, rows * Hv]`, in THREADS. The z axis is a PRODUCT and this body
/// takes both of its factors rather than the product, which is the shape
/// `kernels-metal` states and the one the cross-backend gate settled on. A
/// body that took the product could not tell `rows * v_heads` from any other
/// pair with the same product, and the two are not interchangeable to the
/// shader: it recovers `hv = z % Hv` and `row = z / Hv`.
///
/// # Errors
///
/// See [`head_rows`], plus [`Refusal::Empty`] for a `v_dim` of zero -- the y
/// axis is the only thing that covers the value channels, and a zero there is
/// a dispatch of nothing at all.
fn gdn_grid(rows: i32, v_heads: i32, v_dim: i32) -> Result<[u32; 3], Refusal> {
    let z = head_rows(rows, v_heads)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    Ok([WARP, v_dim.unsigned_abs(), z])
}

/// The grid a prep takes: one warp per `(row, v-head)` and NO dv axis.
///
/// # Errors
///
/// See [`head_rows`].
fn prep_grid(rows: i32, v_heads: i32) -> Result<[u32; 3], Refusal> {
    Ok([WARP, 1, head_rows(rows, v_heads)?])
}

/// The lanes a GDN dispatch puts on x, every one of them.
const WARP: u32 = 32;

/// The fused GDN core: convolution, gates, the delta rule and the state
/// writeback in one dispatch.
///
/// Three of its buffers are recurrent STATE. `conv_state` and `rstate` arrive
/// holding the previous step's values, `rstate` is updated in place, and
/// `new_conv_state` receives the rolled convolution history -- which is a
/// SEPARATE buffer from `conv_state` on purpose, because the shader is still
/// reading the old taps while it writes the new ones. Binding one buffer to
/// both is the ping-pong collapsing, and the failure is silent.
///
/// This is the one dispatch the split pair
/// ([`gdn_prep`] then [`gdn_core_recurrent`]) reproduces bit for bit.
///
/// # Errors
///
/// See [`gdn_grid`].
pub fn gdn_core(
    ctx: &Ctx<'_>,
    mixed: InSlot<0, Buf>,
    conv_state: Held<keys::ConvState, F32s>,
    rstate: Held<keys::RecurrentState, F32sMut>,
    core_out: OutSlot<0, BufMut>,
    conv_w: Weight<0, Buf>,
    conv_b: Weight<1, Buf>,
    a_log: Weight<2, F32s>,
    dt_bias: Weight<3, Buf>,
    a_gate: InSlot<1, Buf>,
    b_gate: InSlot<2, Buf>,
    new_conv_state: Held<keys::NewConvState, F32sMut>,
    params: Block<Buf>,
    rows: Ask<keys::Rows, i32>,
    v_heads: Ask<keys::VHeads, i32>,
    v_dim: Ask<keys::VDim, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_core_bfloat16",
            lanes: gdn_grid(*rows, *v_heads, *v_dim)?,
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

/// [`gdn_core`] with the slot map it addresses its state through.
///
/// The plain form takes the z group's request index as the state slot; this
/// one reads `slot_ids[b]`, so a batch whose requests occupy scattered slots
/// of a shared state arena addresses them where they actually are. Exactly one
/// extra buffer, in the one position the shader declares it.
///
/// # Errors
///
/// See [`gdn_grid`].
pub fn gdn_core_slotted(
    ctx: &Ctx<'_>,
    mixed: InSlot<0, Buf>,
    conv_state: Ask<keys::ConvState, F32s>,
    rstate: Ask<keys::RecurrentState, F32sMut>,
    core_out: OutSlot<0, BufMut>,
    conv_w: Weight<0, Buf>,
    conv_b: Weight<1, Buf>,
    a_log: Weight<2, F32s>,
    dt_bias: Weight<3, Buf>,
    a_gate: InSlot<1, Buf>,
    b_gate: InSlot<2, Buf>,
    new_conv_state: Ask<keys::NewConvState, F32sMut>,
    params: Block<Buf>,
    slot_ids: Ask<keys::RecurrentSlots, U32s>,
    rows: Ask<keys::Rows, i32>,
    v_heads: Ask<keys::VHeads, i32>,
    v_dim: Ask<keys::VDim, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_core_slotted_bfloat16",
            lanes: gdn_grid(*rows, *v_heads, *v_dim)?,
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

/// The prep half of the split pair: everything a value channel would redo.
///
/// Every value channel of a head recomputes the same convolution, the same
/// pair of L2 norms and the same gates, so the split stages them once into
/// three f32 scratch slabs and [`gdn_core_recurrent`] reads them back. The
/// slabs are `F32sMut` here and `F32s` there, which is the direction of the
/// seam written into the types.
///
/// The two halves also SPLIT the convolution writeback: this one rolls the q
/// and k channels and the recurrent half rolls the v channels, so
/// `new_conv_state` is whole only if both ran. That is why it is threaded from
/// this dispatch into the next rather than allocated twice.
///
/// The grid is one z group per `(request, value head)` pair and a single lane
/// row -- `[32, 1, rows]` -- because a prep has no value channel to spread
/// over: producing the shared work once is the whole point.
///
/// # Errors
///
/// See [`gdn_grid`].
pub fn gdn_prep(
    ctx: &Ctx<'_>,
    mixed: InSlot<0, Buf>,
    conv_state: Held<keys::ConvState, F32s>,
    conv_w: Weight<0, Buf>,
    conv_b: Weight<1, Buf>,
    a_log: Weight<2, F32s>,
    dt_bias: Weight<3, Buf>,
    a_gate: InSlot<1, Buf>,
    b_gate: InSlot<2, Buf>,
    pre_q: OutSlot<0, F32sMut>,
    pre_k: OutSlot<1, F32sMut>,
    pre_gate: OutSlot<2, F32sMut>,
    new_conv_state: Held<keys::NewConvState, F32sMut>,
    params: Block<Buf>,
    rows: Ask<keys::Rows, i32>,
    v_heads: Ask<keys::VHeads, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_prep_bfloat16",
            lanes: prep_grid(*rows, *v_heads)?,
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

/// [`gdn_prep`] with the slot map, for the same reason [`gdn_core_slotted`]
/// has one.
///
/// # Errors
///
/// See [`gdn_grid`].
pub fn gdn_prep_slotted(
    ctx: &Ctx<'_>,
    mixed: InSlot<0, Buf>,
    conv_state: Held<keys::ConvState, F32s>,
    conv_w: Weight<0, Buf>,
    conv_b: Weight<1, Buf>,
    a_log: Weight<2, F32s>,
    dt_bias: Weight<3, Buf>,
    a_gate: InSlot<1, Buf>,
    b_gate: InSlot<2, Buf>,
    pre_q: OutSlot<0, F32sMut>,
    pre_k: OutSlot<1, F32sMut>,
    pre_gate: OutSlot<2, F32sMut>,
    new_conv_state: Held<keys::NewConvState, F32sMut>,
    params: Block<Buf>,
    slot_ids: Held<keys::RecurrentSlots, U32s>,
    rows: Ask<keys::Rows, i32>,
    v_heads: Ask<keys::VHeads, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_prep_slotted_bfloat16",
            lanes: prep_grid(*rows, *v_heads)?,
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

/// The prep over a whole PROMPT rather than one token per request.
///
/// Two things separate it from [`gdn_prep_slotted`], and both are on the
/// signature. It always takes a slot map -- the shader declares `slot_ids`
/// under `PIE_SLOTTED` OR `PIE_PREFILL`, so the prefill has one whether or not
/// the deployment is slotted -- and it takes `row_pitch` and `n_scan` as PUSH
/// constants, because a prompt is a strided run of tokens rather than one row.
///
/// `rows` is `tokens * Hv` here, not `requests * Hv`: the prefill walks a
/// single prompt and every token in it needs its own convolution.
///
/// # Errors
///
/// See [`gdn_grid`].
pub fn gdn_prep_prefill(
    ctx: &Ctx<'_>,
    mixed: InSlot<0, Buf>,
    conv_state: Ask<keys::ConvState, F32s>,
    conv_w: Weight<0, Buf>,
    conv_b: Weight<1, Buf>,
    a_log: Weight<2, F32s>,
    dt_bias: Weight<3, Buf>,
    a_gate: InSlot<1, Buf>,
    b_gate: InSlot<2, Buf>,
    pre_q: OutSlot<0, F32sMut>,
    pre_k: OutSlot<1, F32sMut>,
    pre_gate: OutSlot<2, F32sMut>,
    new_conv_state: Ask<keys::NewConvState, F32sMut>,
    params: Block<Buf>,
    slot_ids: Ask<keys::RecurrentSlots, U32s>,
    row_pitch: Ask<keys::InWidth, i32>,
    n_scan: Ask<keys::Rows, i32>,
    rows: Env<i32>,
    v_heads: Ask<keys::VHeads, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_prep_prefill_bfloat16",
            lanes: prep_grid(*rows, *v_heads)?,
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

/// The recurrent half of the split pair, reading the scratch [`gdn_prep`]
/// staged.
///
/// The same eleven-buffer rearrangement the prep is: `rstate` and `core_out`
/// come back (the prep has neither) and the three scratch slabs arrive as
/// `F32s` rather than the gate weights the fused kernel derives them from.
/// Reusing the fused kernel's operand vector here would bind `conv_w` where
/// this shader reads `rstate` and still run.
///
/// # Errors
///
/// See [`gdn_grid`].
pub fn gdn_core_recurrent(
    ctx: &Ctx<'_>,
    mixed: InSlot<0, Buf>,
    conv_state: Held<keys::ConvState, F32s>,
    rstate: Held<keys::RecurrentState, F32sMut>,
    core_out: OutSlot<0, BufMut>,
    conv_w: Weight<0, Buf>,
    conv_b: Weight<1, Buf>,
    pre_q: InSlot<1, F32s>,
    pre_k: InSlot<2, F32s>,
    pre_gate: InSlot<3, F32s>,
    new_conv_state: Held<keys::NewConvState, F32sMut>,
    params: Block<Buf>,
    rows: Ask<keys::Rows, i32>,
    v_heads: Ask<keys::VHeads, i32>,
    v_dim: Ask<keys::VDim, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_core_recurrent_bfloat16",
            lanes: gdn_grid(*rows, *v_heads, *v_dim)?,
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

/// [`gdn_core_recurrent`] with the slot map, for the same reason
/// [`gdn_core_slotted`] has one.
///
/// # Errors
///
/// See [`gdn_grid`].
pub fn gdn_core_recurrent_slotted(
    ctx: &Ctx<'_>,
    mixed: InSlot<0, Buf>,
    conv_state: Held<keys::ConvState, F32s>,
    rstate: Held<keys::RecurrentState, F32sMut>,
    core_out: OutSlot<0, BufMut>,
    conv_w: Weight<0, Buf>,
    conv_b: Weight<1, Buf>,
    pre_q: InSlot<1, F32s>,
    pre_k: InSlot<2, F32s>,
    pre_gate: InSlot<3, F32s>,
    new_conv_state: Held<keys::NewConvState, F32sMut>,
    params: Block<Buf>,
    slot_ids: Held<keys::RecurrentSlots, U32s>,
    rows: Ask<keys::Rows, i32>,
    v_heads: Ask<keys::VHeads, i32>,
    v_dim: Ask<keys::VDim, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "gdn_core_recurrent_slotted_bfloat16",
            lanes: gdn_grid(*rows, *v_heads, *v_dim)?,
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

/// The prefill SCAN: the recurrence walked over a whole prompt in one
/// dispatch, carrying the state forward token by token in registers.
///
/// Seven buffers and not eleven, because a scan reads no weights: the
/// convolution and the gates were done once by [`gdn_prep_prefill`] and this
/// half only walks the delta rule. It is the one shape in the family whose
/// state never leaves the workgroup between tokens, which is what makes a
/// prompt one dispatch instead of `n_scan` of them.
///
/// # The tiling is a schedule, and one half of it is not
///
/// `lanes` and `vrows` pick which compiled shape fires. `VROWS` is how many
/// independent value rows a lane group carries and touches nothing that is
/// summed; `LANES` is the WIDTH of the lane reduction and so its association,
/// and `tests/gpu.rs` measures exactly that -- two tilings hold bit-identical
/// recurrent state precisely when their `LANES` agree. So these two arguments
/// are not interchangeable knobs, and only one of them is free.
///
/// The grid follows from the tiling: a workgroup covers `(32 / LANES) * VROWS`
/// value rows, so the y extent is the number of such groups. `[numthreads(32,
/// 1, 1)]` under `PIE_SCAN` -- not the `(32, 4, 1)` the rest of the family
/// uses -- which is why y is a count of groups here and a count of value
/// channels in [`gdn_core`].
///
/// # Errors
///
/// [`Refusal::Narrow`] for a `(lanes, vrows)` that is not compiled (see
/// [`scan_point`]), and [`Refusal::Empty`] for an empty extent.
pub fn gdn_core_recurrent_prefill(
    ctx: &Ctx<'_>,
    rstate: Ask<keys::RecurrentState, F32sMut>,
    core_out: OutSlot<0, BufMut>,
    pre_q: InSlot<1, F32s>,
    pre_k: InSlot<2, F32s>,
    pre_gate: InSlot<3, F32s>,
    params: Block<Buf>,
    slot_ids: Ask<keys::RecurrentSlots, U32s>,
    row_pitch: Ask<keys::InWidth, i32>,
    n_scan: Ask<keys::Rows, i32>,
    lanes: Param<11, i32>,
    vrows: Param<12, i32>,
    dv: Env<i32>,
    hv: Env<i32>,
) -> Result<(), Refusal> {
    let point = scan_point(*lanes, *vrows)?;
    if *dv <= 0 {
        return Err(Refusal::Empty { what: "dv" });
    }
    if *hv <= 0 {
        return Err(Refusal::Empty { what: "hv" });
    }
    let per_group = (32 / lanes.unsigned_abs()) * vrows.unsigned_abs();
    ctx.dispatch(
        Fire {
            entrypoint: SCAN[point],
            lanes: [32, dv.unsigned_abs().div_ceil(per_group), hv.unsigned_abs()],
        },
        &[
            rstate.v(),
            core_out.v(),
            pre_q.v(),
            pre_k.v(),
            pre_gate.v(),
            params.v(),
            slot_ids.v(),
            row_pitch.v(),
            n_scan.v(),
        ],
    )
}

/// The eight, in the order the rows above name them.
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

    /// A slotted form is the plain one plus a slot map, in that order.
    ///
    /// Three pairs in this family differ by exactly one buffer, and the shader
    /// declares it LAST in all three -- binding 12 of `gdn_core`, 11 of the
    /// recurrent half, 13 of the prep. That is worth checking rather than
    /// reading, because the two members of a pair are written a hundred lines
    /// apart and the slotted one is the copy: a slot map that drifted to the
    /// wrong end binds `slot_ids` where the shader reads `params` and reads a
    /// slot index out of a struct of extents. Every buffer in it is a valid
    /// handle, so nothing refuses.
    #[test]
    fn a_slotted_form_is_the_plain_one_with_the_slot_map_last() {
        let seen = Seen::default();
        gdn_core(
            &seen,
            InSlot::new(Buf(0)),
            Held::new(F32s(1)),
            Held::new(F32sMut(2)),
            OutSlot::new(BufMut(3)),
            Weight::new(Buf(4)),
            Weight::new(Buf(5)),
            Weight::new(F32s(6)),
            Weight::new(Buf(7)),
            InSlot::new(Buf(8)),
            InSlot::new(Buf(9)),
            Held::new(F32sMut(10)),
            Block::new(Buf(11)),
            Ask::new(2),
            Ask::new(4),
            Ask::new(64),
        )
        .expect("a launch");
        gdn_core_slotted(
            &seen,
            InSlot::new(Buf(0)),
            Ask::new(F32s(1)),
            Ask::new(F32sMut(2)),
            OutSlot::new(BufMut(3)),
            Weight::new(Buf(4)),
            Weight::new(Buf(5)),
            Weight::new(F32s(6)),
            Weight::new(Buf(7)),
            InSlot::new(Buf(8)),
            InSlot::new(Buf(9)),
            Ask::new(F32sMut(10)),
            Block::new(Buf(11)),
            Ask::new(U32s(12)),
            Ask::new(2),
            Ask::new(4),
            Ask::new(64),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        let (plain, slotted) = (handles(&calls[0]), handles(&calls[1]));
        assert_eq!(
            slotted,
            [plain.as_slice(), &[12]].concat(),
            "the slotted gdn core is not the plain one with a slot map \
             appended, and the shader declares `slot_ids` after `params`"
        );
        assert_eq!(
            calls[0].1, calls[1].1,
            "a slot map changes where the state IS and not how much of it \
             there is, so the two grids are the same"
        );
    }

    /// The fused core spreads over value channels; a prep does not.
    ///
    /// This is the split's whole reason. The prep computes what every value
    /// channel of a head would recompute -- one convolution, one pair of L2
    /// norms, one set of gates -- so it fires a SINGLE lane row and the
    /// recurrent half fires `Dv` of them. A prep given the core's grid would
    /// recompute the shared work `Dv` times and write the same three scratch
    /// slabs `Dv` times over, which is not wrong and is the entire cost the
    /// split was written to remove.
    #[test]
    fn the_prep_fires_one_lane_row_and_the_recurrent_half_fires_every_value_channel() {
        let seen = Seen::default();
        gdn_prep(
            &seen,
            InSlot::new(Buf(0)),
            Held::new(F32s(1)),
            Weight::new(Buf(2)),
            Weight::new(Buf(3)),
            Weight::new(F32s(4)),
            Weight::new(Buf(5)),
            InSlot::new(Buf(6)),
            InSlot::new(Buf(7)),
            OutSlot::new(F32sMut(8)),
            OutSlot::new(F32sMut(9)),
            OutSlot::new(F32sMut(10)),
            Held::new(F32sMut(11)),
            Block::new(Buf(12)),
            Ask::new(2),
            Ask::new(4),
        )
        .expect("a launch");
        gdn_core_recurrent(
            &seen,
            InSlot::new(Buf(0)),
            Held::new(F32s(1)),
            Held::new(F32sMut(2)),
            OutSlot::new(BufMut(3)),
            Weight::new(Buf(4)),
            Weight::new(Buf(5)),
            InSlot::new(F32s(6)),
            InSlot::new(F32s(7)),
            InSlot::new(F32s(8)),
            Held::new(F32sMut(9)),
            Block::new(Buf(10)),
            Ask::new(2),
            Ask::new(4),
            Ask::new(64),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        assert_eq!(
            (calls[0].1, calls[1].1),
            ([32, 1, 8], [32, 64, 8]),
            "the prep is one lane row over 8 (request, head) pairs and the \
             recurrent half is 64 value channels over the same 8"
        );
    }

    /// A scan tiling names its own module, and only nine pairs do.
    ///
    /// The nine are a LIST and not a product: `(32, 1)` and `(4, 2)` sit
    /// squarely inside the grid the nine suggest and neither is compiled. A
    /// composed name would be a `vkCreateComputePipelines` fault with the
    /// validation layer silent, so the refusal has to happen here.
    #[test]
    fn a_scan_fires_the_tiling_its_two_numbers_name_and_only_nine_pairs_exist() {
        let seen = Seen::default();
        for (lanes, vrows) in [(4, 1), (32, 8)] {
            gdn_core_recurrent_prefill(
                &seen,
                Ask::new(F32sMut(0)),
                OutSlot::new(BufMut(1)),
                InSlot::new(F32s(2)),
                InSlot::new(F32s(3)),
                InSlot::new(F32s(4)),
                Block::new(Buf(5)),
                Ask::new(U32s(6)),
                Ask::new(256),
                Ask::new(7),
                Param::new(lanes),
                Param::new(vrows),
                Env(64),
                Env(4),
            )
            .expect("a compiled tiling is a launch");
        }
        let calls = seen.0.borrow();
        assert_eq!(
            (calls[0].0.as_str(), calls[1].0.as_str()),
            (
                "gdn_core_recurrent_prefill_bfloat16_l_4_v_1",
                "gdn_core_recurrent_prefill_bfloat16_l_32_v_8",
            ),
            "a tiling fires the module its own two numbers spell"
        );

        for (lanes, vrows) in [(32, 1), (4, 2), (64, 1), (16, 8)] {
            assert!(
                scan_point(lanes, vrows).is_err(),
                "({lanes}, {vrows}) is not one of the nine compiled tilings, \
                 and it reads like one"
            );
        }
        assert_eq!(
            SCAN.len(),
            9,
            "nine tilings, and `scan_point` maps onto all of them"
        );
    }

    /// A scan's y extent is groups of value rows, not value rows.
    ///
    /// Under `PIE_SCAN` the shader is `[numthreads(32, 1, 1)]` and NOT the
    /// `(32, 4, 1)` the rest of the family uses, so one workgroup carries
    /// `(32 / LANES) * VROWS` value rows and the grid counts groups. Reading
    /// this as `Dv` -- which is what every other kernel in the file means by
    /// its y extent -- launches `LANES / 32 * VROWS` times too many groups,
    /// and every extra one walks the whole prompt writing `core_out` rows that
    /// are already written.
    ///
    /// The rounding is the second half: `Dv = 100` under `(16, 4)` is eight
    /// rows a group and thirteen groups, the last of which is partial. The
    /// shader carries that -- `vn = min(VROWS, Dv - dv_base)` -- so the tail
    /// must be LAUNCHED, which a truncating division would not do.
    #[test]
    fn a_scans_y_extent_counts_groups_of_value_rows_and_rounds_up() {
        let seen = Seen::default();
        for (lanes, vrows, dv) in [(16, 4, 128), (16, 4, 100), (32, 2, 128), (4, 1, 128)] {
            gdn_core_recurrent_prefill(
                &seen,
                Ask::new(F32sMut(0)),
                OutSlot::new(BufMut(1)),
                InSlot::new(F32s(2)),
                InSlot::new(F32s(3)),
                InSlot::new(F32s(4)),
                Block::new(Buf(5)),
                Ask::new(U32s(6)),
                Ask::new(256),
                Ask::new(7),
                Param::new(lanes),
                Param::new(vrows),
                Env(dv),
                Env(4),
            )
            .expect("a launch");
        }
        let calls = seen.0.borrow();
        let ys: Vec<u32> = calls.iter().map(|c| c.1[1]).collect();
        assert_eq!(
            ys,
            // 8 rows a group over 128 and over 100; 2 rows a group; 8 rows.
            vec![16, 13, 64, 16],
            "the y extent is `Dv` divided by `(32 / LANES) * VROWS`, rounded \
             UP -- 100 value rows in groups of eight is thirteen groups and \
             the last is partial"
        );
        assert!(
            calls.iter().all(|c| c.1[0] == 32 && c.1[2] == 4),
            "the lane width is the shader's 32 and z is one group per value \
             head, whatever the tiling"
        );
    }

    /// An empty extent is refused rather than launched.
    ///
    /// A gdn dispatch over no value channels or no heads is not a no-op: the
    /// grid collapses, nothing runs, and `rstate` keeps the PREVIOUS step's
    /// values while the caller goes on believing the step happened. The
    /// recurrence then continues from a state one token stale, which decodes
    /// as plausible text.
    #[test]
    fn an_empty_extent_is_refused_by_every_shape_in_the_family() {
        let seen = Seen::default();
        assert!(
            gdn_grid(0, 8, 64).is_err()
                && gdn_grid(2, 0, 64).is_err()
                && gdn_grid(2, 4, 0).is_err()
                && gdn_grid(-1, 8, 64).is_err(),
            "no value channels, no heads, or a negative extent is not a launch"
        );
        assert!(
            gdn_core_recurrent_prefill(
                &seen,
                Ask::new(F32sMut(0)),
                OutSlot::new(BufMut(1)),
                InSlot::new(F32s(2)),
                InSlot::new(F32s(3)),
                InSlot::new(F32s(4)),
                Block::new(Buf(5)),
                Ask::new(U32s(6)),
                Ask::new(256),
                Ask::new(7),
                Param::new(16),
                Param::new(4),
                Env(0),
                Env(4),
            )
            .is_err(),
            "a scan over no value channels is refused too, and its grid is \
             computed a different way from the rest of the family"
        );
        assert!(
            seen.0.borrow().is_empty(),
            "a refusal does not dispatch first"
        );
    }
}
