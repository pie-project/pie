use kernels::BindMut;
use kernels::Grid;
use kernels::points::Scalar;
use kernels::routine::Refusal;
use kernels_macros::routine;

use crate::plane::{self, Handle};
use crate::routine::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use crate::views::RecurrentState;
use kernels::raises::Struct;

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

fn core_grid(rows: i32, v_heads: i32, v_dim: i32) -> Result<[u32; 3], Refusal> {
    let z = head_rows(rows, v_heads)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    Ok([32, v_dim.unsigned_abs(), z])
}

fn prep_grid(rows: i32, v_heads: i32) -> Result<[u32; 3], Refusal> {
    Ok([32, 1, head_rows(rows, v_heads)?])
}

const fn core_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 4, 1]
}

const fn simd_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 1, 1]
}

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
    let per_y = (32 / lanes.unsigned_abs()) * vrows.unsigned_abs();
    if per_y == 0 {
        return Err(Refusal::Empty {
            what: "the scan tiling",
        });
    }
    Ok([
        32,
        v_dim.unsigned_abs().div_ceil(per_y),
        v_heads.unsigned_abs(),
    ])
}

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
    rsv: In<Struct<RecurrentState>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let conv_state = rsv.conv_state;
    let rstate = rsv.state;
    let new_conv_state = rsv.new_conv_state;
    let rows = *rows;
    let grid = core_grid(rows, *v_heads, *v_dim)?;
    ctx.fire(
        Fire::at("ssm/gdn_core.metal", "gdn_core_bfloat16").apply(Grid::of(grid, core_group(grid))),
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
    rsv: In<Struct<RecurrentState>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let conv_state = rsv.conv_state;
    let rstate = rsv.state;
    let new_conv_state = rsv.new_conv_state;
    let slot_ids = rsv.slots;
    let rows = *rows;
    let grid = core_grid(rows, *v_heads, *v_dim)?;
    ctx.fire(
        Fire::at("ssm/gdn_core.metal", "gdn_core_slotted_bfloat16")
            .apply(Grid::of(grid, core_group(grid))),
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
    rsv: In<Struct<RecurrentState>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let conv_state = rsv.conv_state;
    let new_conv_state = rsv.new_conv_state;
    let rows = *rows;
    let grid = prep_grid(rows, *v_heads)?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.metal", "gdn_prep_bfloat16").apply(Grid::of(grid, simd_group(grid))),
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

// NOT INLINED, AND NOT THIS POINT ANY MORE. W10 rewrote `ssm.gdn_prep` into
// one launch over the packed `[b | a]` projection; this routine is the
// pre-W10 one (conv + norm + widen + gates against a recurrent view, five
// rectangles out). The `canon` stays because the legacy driver still fires
// it under that name; `impl Ssm` below claims the point with the launch the
// declaration actually states.
#[routine(canon = "ssm.gdn_prep")]
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
    rsv: In<Struct<RecurrentState>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let conv_state = rsv.conv_state;
    let new_conv_state = rsv.new_conv_state;
    let slot_ids = rsv.slots;
    let rows = *rows;
    let grid = prep_grid(rows, *v_heads)?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.metal", "gdn_prep_slotted_bfloat16")
            .apply(Grid::of(grid, simd_group(grid))),
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
    rsv: In<Struct<RecurrentState>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let conv_state = rsv.conv_state;
    let rstate = rsv.state;
    let new_conv_state = rsv.new_conv_state;
    let rows = *rows;
    let grid = core_grid(rows, *v_heads, *v_dim)?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.metal", "gdn_core_recurrent_bfloat16")
            .apply(Grid::of(grid, core_group(grid))),
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
    rsv: In<Struct<RecurrentState>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let conv_state = rsv.conv_state;
    let rstate = rsv.state;
    let new_conv_state = rsv.new_conv_state;
    let slot_ids = rsv.slots;
    let rows = *rows;
    let grid = core_grid(rows, *v_heads, *v_dim)?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.metal", "gdn_core_recurrent_slotted_bfloat16")
            .apply(Grid::of(grid, core_group(grid))),
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
    rsv: In<Struct<RecurrentState>>,
    n_scan: Const<i32>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let conv_state = rsv.conv_state;
    let new_conv_state = rsv.new_conv_state;
    let slot_ids = rsv.slots;
    let row_pitch = mixed.width;
    let n_scan = *n_scan;
    if n_scan <= 0 {
        return Err(Refusal::Empty { what: "n_scan" });
    }
    if row_pitch <= 0 {
        return Err(Refusal::Empty { what: "row_pitch" });
    }
    let grid = prep_grid(n_scan, *v_heads)?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.metal", "gdn_prep_prefill_bfloat16")
            .apply(Grid::of(grid, simd_group(grid))),
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

#[routine]
pub fn gdn_core_recurrent_prefill(
    ctx: &Ctx<'_>,
    pad: In<Tensor<bf16>>,
    core_out: Out<Tensor<bf16>>,
    pre_q: In<Tensor<f32>>,
    pre_k: In<Tensor<f32>>,
    pre_gate: In<Tensor<f32>>,
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
    lanes: Const<i32>,
    vrows: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    n_scan: Const<i32>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let rstate = rsv.state;
    let slot_ids = rsv.slots;

    let row_pitch = pre_q.width;
    let n_scan = *n_scan;

    let point = scan_point(*lanes, *vrows)?;
    if n_scan <= 0 {
        return Err(Refusal::Empty { what: "n_scan" });
    }
    if row_pitch <= 0 {
        return Err(Refusal::Empty { what: "row_pitch" });
    }
    let grid = scan_grid(*v_dim, *v_heads, *lanes, *vrows)?;
    ctx.fire(
        Fire::at(
            "ssm/gdn_prep.metal",
            [
                "gdn_core_recurrent_prefill_bfloat16_l_16_v_1",
                "gdn_core_recurrent_prefill_bfloat16_l_16_v_2",
                "gdn_core_recurrent_prefill_bfloat16_l_16_v_4",
                "gdn_core_recurrent_prefill_bfloat16_l_32_v_2",
                "gdn_core_recurrent_prefill_bfloat16_l_32_v_4",
                "gdn_core_recurrent_prefill_bfloat16_l_32_v_8",
                "gdn_core_recurrent_prefill_bfloat16_l_4_v_1",
                "gdn_core_recurrent_prefill_bfloat16_l_8_v_1",
                "gdn_core_recurrent_prefill_bfloat16_l_8_v_2",
            ][point],
        )
        .apply(Grid::of(grid, simd_group(grid))),
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

/// The threadgroup a per-head prologue runs in.
///
/// One lane per value head, capped at the widest threadgroup Metal will take.
/// The grid is `[v_heads, rows, 1]` and Metal dispatches EXACTLY the threads
/// asked for, so the cap only splits the row across threadgroups — it never
/// leaves a head unwritten.
fn head_lanes(v_heads: i32) -> Result<u32, Refusal> {
    if v_heads <= 0 {
        return Err(Refusal::Empty { what: "v_heads" });
    }
    Ok(v_heads.unsigned_abs().min(256))
}

/// The `Ssm` family, claimed — one point of seven, and the one is the one
/// W10 rewrote.
///
/// Six points stay on the floor's default body, and the six absences are two
/// seams and a family:
///
/// * `ssm.causal_conv1d` / `ssm.causal_conv1d_chunked` — SEAM: THE
///   CONVOLUTION IS NOT A LAUNCH ON THIS PLANE. Both `gdn_prep*` and both
///   `gdn_core*` run their own depthwise conv inline (over the channels each
///   one owns) and write their half of `new_conv_state` as they go; no
///   `.metal` entrypoint takes a `[C, K]` bank, a conv-state ring and a
///   stated width and does that and nothing else. Cuda's
///   `causal_conv1d_update_batched` / `_prefill_batched` are the shape these
///   want.
/// * `ssm.gated_delta` / `ssm.gated_delta_chunked` — SEAM: the metal
///   recurrences read the PRE-W10 STAGING. `gdn_core_recurrent_slotted` and
///   `gdn_core_recurrent_prefill` take `pre_q`, `pre_k` and `pre_gate` — the
///   three f32 scratch planes `gdn_prep_slotted` wrote — while the points
///   state the packed post-convolution `qkv`, the gate row and the packed
///   `[g_log | beta]` decay row, and expect the cut from those to the
///   recurrence's compact planes to happen INSIDE the launch. That cut is
///   cuda's `qwen_gdn_v_gates` (`GdnShape::stage`) and this tree has no
///   counterpart; a body that offset into the packed rows by hand would be
///   claiming a row stride of `v_heads` for bytes whose stride is
///   `2 * v_heads` — true at one token, false at two, which is the exact
///   defect W10 was written to remove.
/// * `ssm.kda_step` / `ssm.kda_chunked` — SEAM: kimi's delta attention, and
///   the `.metal` tree carries no KDA kernel in any form.
#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {
    /// Qwen's gated-delta prologue: the packed `[b | a]` projection in, the
    /// packed `[g_log | beta]` decay row out.
    ///
    /// ONE LAUNCH, AND EXACTLY THE DECLARATION'S SLOTS — which is what makes
    /// this a claim rather than a `canon` row. `gdn_prep_slotted` above wears
    /// the same point's name and is a different statement: it takes the
    /// post-mixer row this declaration has no slot for, reaches a recurrent
    /// view this declaration does not name, and writes five rectangles where
    /// this one states one.
    ///
    /// `qwen_gdn_ba_gates` is the arithmetic with the packing kept, ported
    /// from `kernels-cuda`'s `ssm/gated_delta_net_prep.cuh` slot for slot. It
    /// reads the projection as the matmul wrote it and writes the decay row
    /// as the two recurrence points read it.
    ///
    /// # `v_heads` is read, not stated
    ///
    /// The declaration states no scalar, and it does not need to: the operand
    /// IS `[b | a]`, so the value-head count is half its width. A `Const`
    /// restating it could disagree with the rectangle it divides.
    fn gdn_prep<T: Scalar>(
        &self,
        ba: In<Handle<T>>,
        dt_bias: Const<Handle<T>>,
        a_log: Const<Handle<f32>>,
        gates: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`ssm.gdn_prep`, at an element this plane does not stamp";
        let ba = plane::input::<T, bf16>(ba, WHAT)?;
        let gates = plane::result::<f32, f32>(gates, "`ssm.gdn_prep`'s decay row")?;
        if ba.width % 2 != 0 {
            return Err(Refusal::Narrow {
                what: "the `[b | a]` projection's row, which halves into the value heads",
                at: i64::from(ba.width),
            });
        }
        let v_heads = ba.width / 2;
        // THE RESULT IS THE OPERAND'S SHAPE ON f32 — the width rule says so
        // and the kernel strides both by the same `2 * v_heads`, so a
        // rectangle that disagreed would be written past rather than
        // partially.
        if gates.width != ba.width || gates.rows != ba.rows {
            return Err(Refusal::Narrow {
                what: "the fused `[g_log | beta]` row, against the projection it is derived from",
                at: i64::from(gates.width),
            });
        }
        let lanes = head_lanes(v_heads)?;
        let rows = u32::try_from(ba.rows).map_err(|_| Refusal::Empty { what: "rows" })?;
        if rows == 0 {
            return Err(Refusal::Empty { what: "rows" });
        }
        self.fire(
            Fire::at("ssm/gdn_prep.metal", "qwen_gdn_ba_gates_bfloat16").apply(Grid::of(
                [v_heads.unsigned_abs(), rows, 1],
                [lanes, 1, 1],
            )),
            &[
                ba.arg(),
                plane::weight::<f32, f32>(a_log, "`ssm.gdn_prep`'s decay bank")?.arg(),
                plane::weight::<T, bf16>(dt_bias, WHAT)?.arg(),
                gates.arg(),
                v_heads.arg(),
            ],
        )
    }
}
