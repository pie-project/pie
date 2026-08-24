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

/// The `Ssm` family: NONE of seven points lands, and the cause is one shape
/// disagreement rather than seven missing shaders.
///
/// # This plane's GDN is fused where the declaration is cut
///
/// `kernels::points::Ssm` cuts qwen3.5's gated-delta mixer into three
/// statements, each declaring what it reads and nothing else:
///
/// | point | operands |
/// | --- | --- |
/// | `ssm.causal_conv1d` | the packed `qkv` row, ONE conv weight, the state |
/// | `ssm.gdn_prep` | the packed `[b \| a]` projection, `dt_bias`, `a_log` |
/// | `ssm.gated_delta{,_chunked}` | the POST-CONV `qkv`, `z`, the packed `[g_log \| beta]`, the state |
///
/// `ssm/gdn_core.wgsl` and `ssm/gdn_prep.wgsl` do not cut there. `gdn_core`
/// is ONE launch that convolves, l2-normalises, cooks the gates and runs the
/// recurrence; `gdn_prep` is the first three of those and `gdn_core_recurrent`
/// the last. Both halves take `conv_w`, `conv_b` and the conv state, and both
/// take the WHOLE post-projection row with `q_off`/`k_off`/`v_off` to cut it
/// by — nine scalars where the points state four.
///
/// So there is no arrangement of these shaders that answers a point:
///
/// * `ssm.causal_conv1d` has no shader at all. The convolution exists here
///   only fused into a launch that also runs the recurrence, and a point that
///   fired `gdn_core` would be running the mixer twice.
/// * `ssm.gdn_prep` declares three slots and NOT ONE OF THEM is the packed
///   row `gdn_prep_bfloat16` reads. `gdn_prep_slotted` carries
///   `#[routine(canon = "ssm.gdn_prep")]` today and that canon is a LIE
///   against the current declaration — the same lie
///   `qwen_gdn_post_conv_prep_bf16` told on cuda, which W10 removed by
///   writing a shader that takes the declaration's slots. The canon is left
///   in place because the legacy driver still reaches it by name.
/// * `ssm.gated_delta{,_chunked}` declare a post-conv `qkv` and read the
///   gates packed. `gdn_core_recurrent*` read THREE compact f32 planes
///   (`pre_q`, `pre_k`, `pre_gate`) that only `gdn_prep` produces, and take
///   the conv weights again on top. A body could fire the pair — prep then
///   scan — but it would have to invent `pre_q`/`pre_k`/`pre_gate`, which is
///   scratch this plane cannot allocate (`Ctx` is `dyn Encode`; there is no
///   `Ctx::scratch` behind it), and it would still have no conv weight,
///   because this point declares none.
/// * `ssm.kda_step` and `ssm.kda_chunked` are kimi's KDA rule and no `.wgsl`
///   in this tree mentions it.
///
/// # SEAM — what closes it, and the choice is real
///
///  1. **Shaders that take the declaration's slots.** A `qwen_gdn_ba_gates`
///     twin (packed `[b | a]` in, packed `[g_log | beta]` out, `v_heads` read
///     off half the operand's width) and a scan that reads the packed row and
///     the packed gates. This is what W10 did on cuda and its GATE is worth
///     restating: the packed cut has to be the KERNEL's, because the two
///     halves are `2 * v_heads` apart and a host that offsets by `v_heads` is
///     right at exactly one token.
///  2. **Tier-2.** Make the fused core an inherent method on this plane and
///     have the text gate on `inputs.wgpu()` with a tier-1 else. Cheap, and
///     it costs the text its plane-agnosticism for qwen3.5.
///  3. **Scratch on `Encode`.** Would let a body fire prep-then-scan, and is
///     the wrong shape for the same reason cuda's `Ctx::scratch` is a named
///     device slab and not an arena: the three planes are alive only between
///     the two launches, so they are the BODY's and not the plan's.
///
/// (1) is the one that matches the declaration and the one every other plane
/// has converged on. It is also the largest: three shaders, and the
/// rounding-trajectory law from W2 applies — a prefill tail must round the
/// way the decode step rounds, or the second decoded token diverges.
#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {}
