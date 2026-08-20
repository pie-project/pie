//! Gated DeltaNet: the recurrent state kernels and their prep.
//!
//! `gdn` is an algorithm and not a model, so it takes no model qualifier --
//! the same call the CUDA table makes for `delta_attn_kda` and `indexer_dsa`.

use kernels_macros::routine;
use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, keys};
use kernels::BindMut;
use kernels::routine::Refusal;

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
// The widths were read out of the SPIR-V and are not the obstacle. They were
// twelve descriptors for `gdn_core`, thirteen for `gdn_core_slotted` and
// `gdn_prep`, fourteen for the prefill and slotted preps, eleven for
// `gdn_core_recurrent` and seven for each of the nine prefill shapes; every
// one of those counted a `GdnCoreParams` storage block that is now a push
// range instead, so each is one LOWER -- eleven, twelve, thirteen, ten and
// six -- and each still matches `kernels-metal`'s buffer count for the same
// symbol, because both planes dropped the same block.
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
    // instead of by field, and the shader reads them out of a forty-four-byte
    // push range instead of out of a storage descriptor -- inside the 128
    // bytes `Device::max_push` guarantees, which is what makes the move legal
    // rather than merely tidier.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("gdn_core_bfloat16", ctx.best()), "gdn_core_bfloat16").apply(gdn_grid(rows, *v_heads, *v_dim)?),
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
    // instead of by field, and the shader reads them out of a forty-four-byte
    // push range instead of out of a storage descriptor -- inside the 128
    // bytes `Device::max_push` guarantees, which is what makes the move legal
    // rather than merely tidier.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("gdn_core_slotted_bfloat16", ctx.best()), "gdn_core_slotted_bfloat16").apply(gdn_grid(rows, *v_heads, *v_dim)?),
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
/// See [`gdn_grid`].
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
    // instead of by field, and the shader reads them out of a forty-four-byte
    // push range instead of out of a storage descriptor -- inside the 128
    // bytes `Device::max_push` guarantees, which is what makes the move legal
    // rather than merely tidier.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("gdn_prep_bfloat16", ctx.best()), "gdn_prep_bfloat16").apply(prep_grid(rows, *v_heads)?),
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

/// [`gdn_prep`] with the slot map, for the same reason [`gdn_core_slotted`]
/// has one.
///
/// # Errors
///
/// See [`gdn_grid`].
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
    // instead of by field, and the shader reads them out of a forty-four-byte
    // push range instead of out of a storage descriptor -- inside the 128
    // bytes `Device::max_push` guarantees, which is what makes the move legal
    // rather than merely tidier.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("gdn_prep_slotted_bfloat16", ctx.best()), "gdn_prep_slotted_bfloat16").apply(prep_grid(rows, *v_heads)?),
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

/// The prep over a whole PROMPT rather than one token per request.
///
/// Two things separate it from [`gdn_prep_slotted`], and both are on the
/// signature. It always takes a slot map -- the shader declares `slot_ids`
/// under `PIE_SLOTTED` OR `PIE_PREFILL`, so the prefill has one whether or not
/// the deployment is slotted -- and it takes `row_pitch` and `n_scan`, because
/// a prompt is a strided run of tokens rather than one row.
///
/// Those two used to be the WHOLE of this entrypoint's push block, beside a
/// storage descriptor carrying the geometry: one dispatch holding a range and
/// a descriptor for what is one run of scalars. With the geometry stated as
/// marks they are simply the twelfth and thirteenth fields of a single
/// fifty-two-byte push range, in the order this body passes them, and
/// `Device::max_push` guarantees 128.
///
/// `rows` is `tokens * Hv` here, not `requests * Hv`: the prefill walks a
/// single prompt and every token in it needs its own convolution.
///
/// # Errors
///
/// See [`gdn_grid`].
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
    // instead of by field, and the shader reads them out of a forty-four-byte
    // push range instead of out of a storage descriptor -- inside the 128
    // bytes `Device::max_push` guarantees, which is what makes the move legal
    // rather than merely tidier.
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
    // `Env<i32>` WITH NO KEY STOOD HERE and claimed no source at all, so this
    // routine could never be bound from its column: the wrapper only kept a
    // bare scalar from being read as an operand. It is the fire's token count,
    // which is what [`gdn_prep_slotted`] beside it already asks for.
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let row_pitch = mixed.width;
    let n_scan = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("gdn_prep_prefill_bfloat16", ctx.best()), "gdn_prep_prefill_bfloat16").apply(prep_grid(rows, *v_heads)?),
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

/// The recurrent half of the split pair, reading the scratch [`gdn_prep`]
/// staged.
///
/// The same ten-descriptor rearrangement the prep is: `rstate` and `core_out`
/// come back (the prep has neither) and the three scratch slabs arrive as
/// `F32s` rather than the gate weights the fused kernel derives them from.
/// Reusing the fused kernel's operand vector here would bind `conv_w` where
/// this shader reads `rstate` and still run.
///
/// TEN AND NOT ELEVEN, because the eleventh was the `GdnCoreParams` storage
/// block. It is a forty-four-byte push range now, so every binding after where
/// it stood moved down by one -- `slot_ids` from 11 to 10 in the slotted
/// twin.
///
/// # Errors
///
/// See [`gdn_grid`].
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
    // instead of by field, and the shader reads them out of a forty-four-byte
    // push range instead of out of a storage descriptor -- inside the 128
    // bytes `Device::max_push` guarantees, which is what makes the move legal
    // rather than merely tidier.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("gdn_core_recurrent_bfloat16", ctx.best()), "gdn_core_recurrent_bfloat16").apply(gdn_grid(rows, *v_heads, *v_dim)?),
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

/// [`gdn_core_recurrent`] with the slot map, for the same reason
/// [`gdn_core_slotted`] has one.
///
/// # Errors
///
/// See [`gdn_grid`].
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
    // instead of by field, and the shader reads them out of a forty-four-byte
    // push range instead of out of a storage descriptor -- inside the 128
    // bytes `Device::max_push` guarantees, which is what makes the move legal
    // rather than merely tidier.
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
    ctx.fire(
        Fire::at(crate::routine::module_path("gdn_core_recurrent_slotted_bfloat16", ctx.best()), "gdn_core_recurrent_slotted_bfloat16").apply(gdn_grid(rows, *v_heads, *v_dim)?),
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

/// The prefill SCAN: the recurrence walked over a whole prompt in one
/// dispatch, carrying the state forward token by token in registers.
///
/// Six descriptors and not ten, because a scan reads no weights: the
/// convolution and the gates were done once by [`gdn_prep_prefill`] and this
/// half only walks the delta rule. It was seven while `GdnCoreParams` was a
/// storage block; the geometry and the tiling are thirteen `Const` marks now,
/// of which the eleven geometry words reach the module -- in a push range that
/// with `row_pitch` and `n_scan` behind them comes to fifty-two bytes -- while
/// the two tiling words reach it only as the entrypoint's NAME. It is the one
/// shape in the family whose state
/// never leaves the workgroup between tokens, which is what makes a prompt one
/// dispatch instead of `n_scan` of them.
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
    // THE HEAD COUNT AND THE VALUE WIDTH, OFF THE MARKS. Both were asks --
    // *"BACK TO ASKS, AS THE OTHER TWO PLANES"* -- because this routine
    // forwarded its params run as a STRUCT the shader read by field, so slot
    // 0 was not `dv`'s to take. All three planes state the same eleven marks
    // now, which is what `kernels`'s
    // `two_backends_that_crossed_the_same_kernel_agree_on_its_signature`
    // still holds them to.
    let hv = *v_heads;
    let dv = *v_dim;
    let rstate = ctx.ask::<Tensor<f32>, keys::RecurrentState>()?;
    let slot_ids = ctx.ask::<Tensor<u32>, keys::RecurrentSlots>()?;
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
    if dv <= 0 {
        return Err(Refusal::Empty { what: "dv" });
    }
    if hv <= 0 {
        return Err(Refusal::Empty { what: "hv" });
    }
    let per_group = (32 / lanes.unsigned_abs()) * vrows.unsigned_abs();
    ctx.fire(
        Fire::at(crate::routine::module_path(SCAN[point], ctx.best()), SCAN[point]).apply([32, dv.unsigned_abs().div_ceil(per_group), hv.unsigned_abs()]),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    type Call = (String, [u32; 3], Vec<ArgValue>);

    /// The geometry every call in this module states, in `GdnShape::params`
    /// order, minus the two numbers a test may vary.
    ///
    /// They are CONSTANTS and not probe cells, which is the whole of what the
    /// migration to `Const` marks changed here: the eleven words used to reach
    /// the shader as a `GdnCoreParams` block the body forwarded whole, so a
    /// test that wanted a particular head count had to set it on the probe and
    /// hope the driver would have answered the same; they are arguments now,
    /// so a call states its own geometry and this module owns it outright.
    ///
    /// The numbers are consistent for the default fixture (`Hv = 4`,
    /// `Dv = 64`): one key head 128 wide, four value heads 64 wide, and the
    /// three region offsets laid out over `2*Hk*Dk + Hv*Dv` channels. Nothing
    /// here dispatches a shader, so only the offsets' ORDER is load-bearing --
    /// but a fixture that could not describe a real checkpoint would be a bad
    /// place to read the order off.
    const K_DIM: i32 = 128;
    /// See [`K_DIM`].
    const K_HEADS: i32 = 1;
    /// See [`K_DIM`].
    const CONV_K: i32 = 4;
    /// See [`K_DIM`].
    const CONV_DIM: i32 = 2 * K_HEADS * K_DIM + 4 * 64;
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

    /// An `Encode` that remembers, and answers the facts this family's bodies
    /// still ask for.
    ///
    /// `rows` backs every routine's `ctx.ask::<i32, keys::Rows>()` (the
    /// prefill scan asks it twice, once as itself and once as `n_scan` --
    /// see [`gdn_core_recurrent_prefill`] -- so one field serves both). The
    /// four recurrent-state facts -- `conv_state`, `rstate`,
    /// `new_conv_state`, `recurrent_slots` -- are buffers a real driver would
    /// thread from the previous step and this probe answers with a distinct,
    /// fixed handle apiece, so a test that inspects the bound list (see
    /// `handles`) can tell them apart and two calls that ask the same fact see
    /// the same answer.
    ///
    /// WHAT IT NO LONGER ANSWERS is the interesting half. It carried cells for
    /// `v_heads`, `v_dim`, `lanes` and `vrows`, and a `words` run standing in
    /// for the statement's scalars, because every body here forwarded
    /// `ctx.params()` as a STRUCT and could name no word inside it: the head
    /// count came back through `keys::VHeads`, and the scan's tiling through
    /// `Asks::param`, reading words 11 and 12 by index. All six are marks now,
    /// so a call states them and the probe has nothing to say about any of
    /// them. `ctx.params()` itself is gone with them, so there is no arm for
    /// `Slot(Kind::Params, 0)` either.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        rows: Cell<i32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                rows: Cell::new(1),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(
            &self,
            ty: kernels::Ty,
            source: kernels::Source,
        ) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            if source == <keys::ConvState as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: 501, writes: false, rows: 0, width: 0 });
            }
            if source == <keys::RecurrentState as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: 502, writes: true, rows: 0, width: 0 });
            }
            if source == <keys::NewConvState as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: 503, writes: true, rows: 0, width: 0 });
            }
            if source == <keys::RecurrentSlots as Fact>::SOURCE {
                return Ok(ArgValue::Buffer { handle: 504, writes: false, rows: 0, width: 0 });
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
                .push((fire.entrypoint.to_owned(), fire.lanes, args.to_vec()));
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

    /// [`gdn_core`] over this module's fixture.
    fn core(seen: &Seen, v_heads: i32, v_dim: i32) -> Result<(), Refusal> {
        gdn_core(
            seen,
            In::new(Tensor::<bf16>::new(0)),
            Out::new(Tensor::<bf16>::new(3)),
            Const::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<bf16>::new(5)),
            Const::new(Tensor::<f32>::new(6)),
            Const::new(Tensor::<bf16>::new(7)),
            In::new(Tensor::<bf16>::new(8)),
            In::new(Tensor::<bf16>::new(9)),
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

    /// [`gdn_core_slotted`] over the same fixture, so the pair below differs
    /// by the slot map and by nothing else.
    fn core_slotted(seen: &Seen, v_heads: i32, v_dim: i32) -> Result<(), Refusal> {
        gdn_core_slotted(
            seen,
            In::new(Tensor::<bf16>::new(0)),
            Out::new(Tensor::<bf16>::new(3)),
            Const::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<bf16>::new(5)),
            Const::new(Tensor::<f32>::new(6)),
            Const::new(Tensor::<bf16>::new(7)),
            In::new(Tensor::<bf16>::new(8)),
            In::new(Tensor::<bf16>::new(9)),
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
            In::new(Tensor::<bf16>::new(0)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            Const::new(Tensor::<f32>::new(4)),
            Const::new(Tensor::<bf16>::new(5)),
            In::new(Tensor::<bf16>::new(6)),
            In::new(Tensor::<bf16>::new(7)),
            Out::new(Tensor::<f32>::new(8)),
            Out::new(Tensor::<f32>::new(9)),
            Out::new(Tensor::<f32>::new(10)),
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
            In::new(Tensor::<bf16>::new(0)),
            Out::new(Tensor::<bf16>::new(3)),
            Const::new(Tensor::<bf16>::new(4)),
            Const::new(Tensor::<bf16>::new(5)),
            In::new(Tensor::<f32>::new(6)),
            In::new(Tensor::<f32>::new(7)),
            In::new(Tensor::<f32>::new(8)),
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

    /// [`gdn_core_recurrent_prefill`], whose thirteenth and twelfth words are
    /// the tiling.
    ///
    /// `width: 256` on `pre_q` is `row_pitch`, read off that mark directly
    /// rather than asked; its `rows` is unused by this body. `lanes` and
    /// `vrows` are stated here because they are words 11 and 12 of the
    /// statement's own run -- `model-dsl` pushes them onto `GdnShape::params`
    /// for this entrypoint alone -- and the body turns them into a compiled
    /// SPELLING rather than forwarding them.
    fn scan(seen: &Seen, v_heads: i32, v_dim: i32, lanes: i32, vrows: i32) -> Result<(), Refusal> {
        gdn_core_recurrent_prefill(
            seen,
            // `pad` stands in for the statement's input 0, which this scan
            // does not read -- see the routine's own doc comment.
            In::new(Tensor::<bf16>::new(0)),
            Out::new(Tensor::<bf16>::new(1)),
            In { ptr: Tensor::<f32>::new(2), rows: 0, width: 256 },
            In::new(Tensor::<f32>::new(3)),
            In::new(Tensor::<f32>::new(4)),
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
            Const::new(INV_SQRT_DK),
            Const::new(lanes),
            Const::new(vrows))
    }

    /// A slotted form is the plain one plus a slot map, in that order.
    ///
    /// Three pairs in this family differ by exactly one buffer, and the shader
    /// declares it LAST in all three -- binding 11 of `gdn_core`, 10 of the
    /// recurrent half, 12 of the prep, each one lower than it used to be
    /// because the `GdnCoreParams` binding that sat in front of it is gone.
    /// That is worth checking rather than reading, because the two members of
    /// a pair are written a hundred lines apart and the slotted one is the
    /// copy: a slot map that drifted to the wrong end binds `slot_ids` where
    /// the shader reads a value plane and reads a slot index out of fp32
    /// scratch. Every buffer in it is a valid handle, so nothing refuses.
    #[test]
    fn a_slotted_form_is_the_plain_one_with_the_slot_map_last() {
        let seen = Seen::default();
        core(&seen, 4, 64).expect("a launch");
        core_slotted(&seen, 4, 64).expect("a launch");

        let calls = seen.calls.borrow();
        let (plain, slotted) = (handles(&calls[0]), handles(&calls[1]));
        assert_eq!(
            slotted,
            // 504 is this probe's fixed `RecurrentSlots` handle -- both
            // calls ask the same recurrent-state facts in the same order, so
            // `plain` is exactly the prefix `slotted` adds one handle to.
            [plain.as_slice(), &[504]].concat(),
            "the slotted gdn core is not the plain one with a slot map \
             appended, and the shader declares `slot_ids` last"
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
        // Both dispatches ask the same `rows`, which is the "8" the
        // assertion below and its doc comment both mean.
        seen.rows.set(4);
        // Two value heads over four rows is the eight on z; the calls STATE
        // the head count now, at word 3 of the statement's run, rather than
        // asking a probe cell for it.
        prep(&seen, 2, 64).expect("a launch");
        recurrent(&seen, 2, 64).expect("a launch");

        let calls = seen.calls.borrow();
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
        seen.rows.set(7);
        for (lanes, vrows) in [(4, 1), (32, 8)] {
            scan(&seen, 4, 64, lanes, vrows).expect("a compiled tiling is a launch");
        }
        let calls = seen.calls.borrow();
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
        seen.rows.set(7);
        for (lanes, vrows, dv) in [(16, 4, 128), (16, 4, 100), (32, 2, 128), (4, 1, 128)] {
            // The value width this case is about: a mark on the call now,
            // where it used to be a cell the body asked this probe for.
            scan(&seen, 4, dv, lanes, vrows).expect("a launch");
        }
        let calls = seen.calls.borrow();
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
        // A compiled tiling, so this call reaches its own `dv <= 0` check
        // instead of refusing `Narrow` at `scan_point` first -- and a value
        // width of zero, which is the refusal it exercises. Both are stated
        // on the call: they were a probe cell apiece while the body asked
        // for them, its params run being a struct with no slot to spare.
        assert!(
            scan(&seen, 4, 0, 16, 4).is_err(),
            "a scan over no value channels is refused too, and its grid is \
             computed a different way from the rest of the family"
        );
        assert!(
            seen.calls.borrow().is_empty(),
            "a refusal does not dispatch first"
        );
    }
}
