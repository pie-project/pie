use kernels::BindMut;
use kernels_macros::routine;

use crate::routine::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use crate::views::RecurrentState;
use kernels::raises::Struct;
use kernels::routine::Refusal;

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

fn gdn_grid(rows: i32, v_heads: i32, v_dim: i32) -> Result<[u32; 3], Refusal> {
    let z = head_rows(rows, v_heads)?;
    if v_dim <= 0 {
        return Err(Refusal::Empty { what: "v_dim" });
    }
    Ok([32, v_dim.unsigned_abs(), z])
}

fn prep_grid(rows: i32, v_heads: i32) -> Result<[u32; 3], Refusal> {
    Ok([32, 1, head_rows(rows, v_heads)?])
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
    ctx.fire(
        Fire::at("ssm/gdn_core.wgsl", "gdn_core_bfloat16").apply(gdn_grid(rows, *v_heads, *v_dim)?),
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
    ctx.fire(
        Fire::at("ssm/gdn_core.wgsl", "gdn_core_slotted_bfloat16")
            .apply(gdn_grid(rows, *v_heads, *v_dim)?),
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
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_prep_bfloat16").apply(prep_grid(rows, *v_heads)?),
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
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_prep_slotted_bfloat16")
            .apply(prep_grid(rows, *v_heads)?),
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
    let rows = n_scan;
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_prep_prefill_bfloat16")
            .apply(prep_grid(rows, *v_heads)?),
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
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_core_recurrent_bfloat16")
            .apply(gdn_grid(rows, *v_heads, *v_dim)?),
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
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_core_recurrent_slotted_bfloat16")
            .apply(gdn_grid(rows, *v_heads, *v_dim)?),
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
pub fn gdn_core_recurrent_prefill(
    ctx: &Ctx<'_>,
    #[allow(unused_variables)] pad: In<Tensor<bf16>>,
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
    let hv = *v_heads;
    let dv = *v_dim;
    let rstate = rsv.state;
    let slot_ids = rsv.slots;
    let row_pitch = pre_q.width;
    let n_scan = *n_scan;

    let point = scan_point(*lanes, *vrows)?;
    if dv <= 0 {
        return Err(Refusal::Empty { what: "dv" });
    }
    if hv <= 0 {
        return Err(Refusal::Empty { what: "hv" });
    }
    let per_group = (32 / lanes.unsigned_abs()) * vrows.unsigned_abs();
    ctx.fire(
        Fire::at(
            "ssm/gdn_prep.wgsl",
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
        .apply([32, dv.unsigned_abs().div_ceil(per_group), hv.unsigned_abs()]),
        &[
            rstate.arg_mut(),
            core_out.arg(),
            pre_q.arg(),
            pre_k.arg(),
            pre_gate.arg(),
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
