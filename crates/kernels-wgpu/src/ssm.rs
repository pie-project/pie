//! Gated DeltaNet: the recurrent state kernels and their prep.
//!
//! `gdn` is an algorithm and not a model, so it takes no model qualifier --
//! the same call the CUDA table makes for `delta_attn_kda` and `indexer_dsa`.

use kernels_macros::routine;
use kernels::KernelSig;
use kernels::BindMut;

pub static KERNELS: &[KernelSig] = &[
    // 1 in gdn_core.wgsl
    // 1 in gdn_prep.wgsl
    // 9 in gdn_prep.wgsl
    // 1 in gdn_prep.wgsl
    // 1 in gdn_core.wgsl
    // 1 in gdn_prep.wgsl
    // 1 in gdn_prep.wgsl
    // 1 in gdn_prep.wgsl
];
/// The entrypoints of this family's routines whose ROWS have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. Not every kernel here has crossed its
/// arm — this family still states rows for the ones that have not — so this
/// is the retired SUBSET rather than the whole family, and
/// `a_retired_familys_stated_entrypoints_are_what_its_bodies_fire` compares
/// it against the bodies that fire them.
///
/// See [`crate::sample::ENTRYPOINTS`] for why a retired row's entrypoints
/// have to be stated at all.
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

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, keys};
use kernels::routine::Refusal;

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
/// neither exists, which is what `scan_point` is for.
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
/// See `head_rows`, plus [`Refusal::Empty`] for a `v_dim` of zero -- the y
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
/// See `head_rows`.
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
/// See `gdn_grid`.
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
    b_gate: In<Tensor<bf16>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let v_heads = ctx.ask::<i32, keys::VHeads>()?;
    let v_dim = ctx.ask::<i32, keys::VDim>()?;
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let params = ctx.params()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("ssm/gdn_core.wgsl", "gdn_core_bfloat16").apply(gdn_grid(rows, v_heads, v_dim)?),
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
            params,
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
/// See `gdn_grid`.
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
    b_gate: In<Tensor<bf16>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let v_heads = ctx.ask::<i32, keys::VHeads>()?;
    let v_dim = ctx.ask::<i32, keys::VDim>()?;
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let params = ctx.params()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("ssm/gdn_core.wgsl", "gdn_core_slotted_bfloat16").apply(gdn_grid(rows, v_heads, v_dim)?),
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
            params,
            slot_ids.arg(),
        ],
    )
}

/// The prep half of the split pair: everything a value channel would redo.
///
/// Every value channel of a head recomputes the same convolution, the same
/// pair of L2 norms and the same gates, so the split stages them once into
/// three f32 scratch slabs and [`gdn_core_recurrent`] reads them back. The
/// slabs are `F32s` here and `F32s` there, which is the direction of the
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
/// See `gdn_grid`.
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
    pre_gate: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let v_heads = ctx.ask::<i32, keys::VHeads>()?;
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let params = ctx.params()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_prep_bfloat16").apply(prep_grid(rows, v_heads)?),
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
            params,
        ],
    )
}

/// [`gdn_prep`] with the slot map, for the same reason [`gdn_core_slotted`]
/// has one.
///
/// # Errors
///
/// See `gdn_grid`.
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
    pre_gate: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let v_heads = ctx.ask::<i32, keys::VHeads>()?;
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let params = ctx.params()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_prep_slotted_bfloat16").apply(prep_grid(rows, v_heads)?),
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
            params,
            slot_ids.arg(),
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
/// See `gdn_grid`.
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
    pre_gate: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let v_heads = ctx.ask::<i32, keys::VHeads>()?;
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let params = ctx.params()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    let row_pitch = mixed.width;
    let n_scan = ctx.ask::<i32, keys::Rows>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_prep_prefill_bfloat16").apply(prep_grid(rows, v_heads)?),
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
            params,
            slot_ids.arg(),
            row_pitch.arg(),
            n_scan.arg(),
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
/// See `gdn_grid`.
#[routine]
pub fn gdn_core_recurrent(
    ctx: &Ctx<'_>,
    mixed: In<Tensor<bf16>>,
    core_out: Out<Tensor<bf16>>,
    conv_w: Const<Tensor<bf16>>,
    conv_b: Const<Tensor<bf16>>,
    pre_q: In<Tensor<f32>>,
    pre_k: In<Tensor<f32>>,
    pre_gate: In<Tensor<f32>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let v_heads = ctx.ask::<i32, keys::VHeads>()?;
    let v_dim = ctx.ask::<i32, keys::VDim>()?;
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let params = ctx.params()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_core_recurrent_bfloat16").apply(gdn_grid(rows, v_heads, v_dim)?),
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
            params,
        ],
    )
}

/// [`gdn_core_recurrent`] with the slot map, for the same reason
/// [`gdn_core_slotted`] has one.
///
/// # Errors
///
/// See `gdn_grid`.
#[routine]
pub fn gdn_core_recurrent_slotted(
    ctx: &Ctx<'_>,
    mixed: In<Tensor<bf16>>,
    core_out: Out<Tensor<bf16>>,
    conv_w: Const<Tensor<bf16>>,
    conv_b: Const<Tensor<bf16>>,
    pre_q: In<Tensor<f32>>,
    pre_k: In<Tensor<f32>>,
    pre_gate: In<Tensor<f32>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let v_heads = ctx.ask::<i32, keys::VHeads>()?;
    let v_dim = ctx.ask::<i32, keys::VDim>()?;
    let conv_state = ctx.ask::<Tensor<f32>, keys::ConvState>()?;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let new_conv_state = ctx.ask::<Tensor<f32>, keys::NewConvState>()?;
    let params = ctx.params()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", "gdn_core_recurrent_slotted_bfloat16").apply(gdn_grid(rows, v_heads, v_dim)?),
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
            params,
            slot_ids.arg(),
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
/// `scan_point`), and [`Refusal::Empty`] for an empty extent.
#[routine]
pub fn gdn_core_recurrent_prefill(
    ctx: &Ctx<'_>,
    // THE STATEMENT'S INPUT 0, WHICH THIS SCAN DOES NOT READ. The text places
    // `[mixed, pre_q, pre_k, pre_gate]` and the scan takes the three prepared
    // planes only; the mark is here because the slot is a POSITION now, and
    // without it `pre_q` would bind `mixed`.
    #[allow(unused_variables)]
    pad: In<Tensor<bf16>>,
    core_out: Out<Tensor<bf16>>,
    pre_q: In<Tensor<f32>>,
    pre_k: In<Tensor<f32>>,
    pre_gate: In<Tensor<f32>>) -> Result<(), Refusal> {
    // BACK TO ASKS, AS METAL'S TWIN. This routine's params run is a struct the
    // shader reads by field, so slot 0 is not `dv`'s to take -- and the two
    // planes must ask the binder the same questions, which is what
    // `kernels`'s `two_backends_that_crossed_the_same_kernel_agree_on_its_signature` says.
    let hv = ctx.ask::<i32, keys::VHeads>()?;
    let dv = ctx.ask::<i32, keys::VDim>()?;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let params = ctx.params()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
    let row_pitch = pre_q.width;
    let n_scan = ctx.ask::<i32, keys::Rows>()?;
    // THE TILING, READ OUT OF THE STATEMENT'S OWN RUN. These were
    // `Param<11, i32>` and `Param<12, i32>` -- words of the struct this body
    // forwards whole -- and the migration turned them into `keys::Lanes` and
    // `keys::Vrows`, which no driver answers, so every prefill scan refused
    // `Unstated`. They cannot be `Const` marks either: the run is the
    // shader's layout and slots 0..10 are its fields, not this body's.
    let lanes = ctx.param(11)?;
    let vrows = ctx.param(12)?;
    let point = scan_point(lanes, vrows)?;
    if dv <= 0 {
        return Err(Refusal::Empty { what: "dv" });
    }
    if hv <= 0 {
        return Err(Refusal::Empty { what: "hv" });
    }
    let per_group = (32 / lanes.unsigned_abs()) * vrows.unsigned_abs();
    ctx.fire(
        Fire::at("ssm/gdn_prep.wgsl", SCAN[point]).apply([
                32,
                dv.unsigned_abs().div_ceil(per_group),
                hv.unsigned_abs(),
            ]),
        &[
            rstate.arg_mut(),
            core_out.arg(),
            pre_q.arg(),
            pre_k.arg(),
            pre_gate.arg(),
            params,
            slot_ids.arg(),
            row_pitch.arg(),
            n_scan.arg(),
        ],
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The three grids, checked against this tree's OWN shaders.
    ///
    /// This family's rows state no operands and no `launch` rule, so there is
    /// neither a row nor a `driver-wgpu::geometry` arm to compare a grid
    /// against — the only authority is `ssm/gdn_{core,prep}.wgsl`. All three
    /// happen to be `kernels-vulkan`'s numbers, and that is a conclusion
    /// rather than an assumption: each was read off the shader first.
    ///
    /// * `gdn_core.wgsl` and the `PIE_RECURRENT` arm are
    ///   `@workgroup_size(32, 4)`. `wid.z` is `rows * v_heads`, `lid.x` the
    ///   32-lane key sweep, and `gid.y` the value channel — guarded by
    ///   `dv_idx < p.Dv`, which is what makes rounding y up to a whole four
    ///   safe.
    /// * the prep arm is `@workgroup_size(32)`, one workgroup per
    ///   `(row, head)`.
    /// * the `PIE_SCAN` arm is `@workgroup_size(32)` and divides its 32 lanes
    ///   into `32 / PIE_LANES` value groups, which is why `per_group` is
    ///   `(32 / lanes) * vrows`.
    #[test]
    fn the_three_grids_are_what_the_shaders_index() {
        // 7 rows x 8 value heads = 56 workgroups on z, 64 value channels on
        // y, one 32-lane sweep on x.
        assert_eq!(gdn_grid(7, 8, 64).expect("a real shape"), [32, 64, 56]);
        assert_eq!(prep_grid(7, 8).expect("a real shape"), [32, 1, 56]);

        // Empty extents refuse rather than dispatch a grid of nothing.
        assert!(matches!(gdn_grid(0, 8, 64), Err(Refusal::Empty { .. })));
        assert!(matches!(gdn_grid(7, 0, 64), Err(Refusal::Empty { .. })));
        assert!(matches!(gdn_grid(7, 8, 0), Err(Refusal::Empty { .. })));
        assert!(matches!(prep_grid(7, 0), Err(Refusal::Empty { .. })));
    }

    /// A `(LANES, VROWS)` pair the tree compiles no scan for is refused.
    ///
    /// The body indexes `SCAN` with it, so an unknown pair must not reach the
    /// index. Nine are compiled and the ninth is not `(32, 1)` -- the table is
    /// not a product, and guessing its shape is how a body spells a name that
    /// is not there.
    #[test]
    fn a_scan_shape_the_tree_does_not_carry_is_refused_by_name() {
        assert!(scan_point(32, 1).is_err());
        assert!(scan_point(64, 2).is_err());
        assert!(scan_point(16, 8).is_err());
        for (lanes, vrows) in [
            (16, 1),
            (16, 2),
            (16, 4),
            (32, 2),
            (32, 4),
            (32, 8),
            (4, 1),
            (8, 1),
            (8, 2),
        ] {
            let at = scan_point(lanes, vrows).expect("a compiled shape");
            assert!(at < SCAN.len());
            assert!(
                SCAN[at].ends_with(&format!("_l_{lanes}_v_{vrows}")),
                "`{}` is not the spelling for ({lanes}, {vrows})",
                SCAN[at]
            );
        }
    }
}
