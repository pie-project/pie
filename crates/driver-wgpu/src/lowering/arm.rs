//! The routine plane's driver side: an arm per crossed kernel.
//!
//! # What an arm is for
//!
//! A `kernels-wgpu` routine BODY states the entrypoint, the module and the
//! grid, and takes its operands as typed arguments. What it cannot do is find
//! them: a body is handed `Buf`s and `i32`s, and turning a traced statement
//! into that argument list is the driver's half. An [`Arm`] is that half, one
//! per kernel, and it is the only thing the routine path needs that the table
//! path expressed as a `KernelSig`'s `operands` column.
//!
//! ```text
//!   table path:    row.operands ──► reorder ──► slots
//!   routine path:  arm          ──► ArgValue list ──► body ──► Fire + args
//! ```
//!
//! # Why this is a skeleton with one arm in it
//!
//! `refactor-bigplan.md` §7 Stage 2: *"the first commit in a backend is the
//! driver-side arm registry skeleton, with `ROUTINES` empty and behaviour
//! unchanged. It compiles, both suites stay green, and every family port
//! afterwards has somewhere to land."* This backend crossed its bodies first
//! — all ninety-nine of them — so the skeleton arrives late and lands into a
//! tree that already has somewhere to come FROM.
//!
//! The one arm is `sample::argmax_logits`, and it is first for the reason
//! `kernels-metal` gives for the same choice: its row states no `launch`
//! rule, so [`crate::geometry::lanes`] has always refused it `Unstated` and
//! **the table path has never dispatched this kernel at all**. There is no
//! prior behaviour for the crossing to preserve, which makes it the cheapest
//! place a seam can be proven.
//!
//! # What it does not do yet
//!
//! Dispatch. Nothing calls [`arm_for`] from the launch path; the fork in
//! [`crate::dispatch::plan_one`] is the next commit, and it needs one thing
//! this file cannot supply on its own — see [`Handles::params_block`].

use kernels::routine::Refusal;
use kernels_wgpu::routine::ArgValue;
use model_compiler::lower::Arg;

use crate::binding::FireTable;
use crate::dispatch::Geometry;

/// The numbers a body takes as `Env` arguments.
///
/// Every one is a fact about the FIRE or the model, not about the statement,
/// which is what `Provenance::Env` means: the kernel never reads them and
/// they size the grid. A body asks for the ones it needs by name and this is
/// where they come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Facts {
    /// How many rows this launch covers — its rectangle's height.
    pub rows: u32,
    /// The row width the launch writes.
    pub width: u32,
    /// The row width it reads, where the two differ.
    pub in_width: u32,
    /// How many requests the fire holds, which is not its row count.
    pub requests: u32,
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// The head width.
    pub head_dim: u32,
    /// How many channels of each head rotate.
    pub rotary_dims: u32,
    /// Experts in the mixture, or zero.
    pub n_experts: u32,
    /// How many of them each token picks.
    pub experts_per_token: u32,
    /// The norm axis, where a row holds more than one.
    pub axis: u32,
    /// The affine group size, off the SYMBOL.
    pub group: i32,
    /// The affine bit width, off the SYMBOL.
    pub bits: i32,
    /// The tile shape, off the symbol's `_bm_M_bn_N` suffix.
    pub tile_m: i32,
    /// See [`Self::tile_m`].
    pub tile_n: i32,
}

/// One thing a body asked for, in the order it asked.
///
/// Not just an `Arg`: a body may want the parameter BLOCK, which is the
/// packed scalar run and has no operand, or one of the fire's TABLES, which
/// the resolver holds and the statement never names. Each takes a place in
/// the body's argument list, so each takes a handle here, and a variant that
/// skipped one would point every handle after it at the wrong buffer.
#[derive(Clone)]
pub enum Asked {
    /// An operand the statement carries.
    Operand(Arg),
    /// The packed scalar run.
    Params,
    /// A buffer the FIRE holds — the rope frequencies, the sampling indices.
    /// `driver-metal::lowering::arm::Handles::table` is the same ask, and the
    /// table path reaches these through `kernels::Source::RopeFrequencies`
    /// and its siblings.
    Table(FireTable),
    /// A binding the module DECLARES and this statement does not fill.
    ///
    /// `naga` deletes nothing, so a variant that never reads a buffer still
    /// declares it and the layout still has an entry at that number. The row
    /// said `Source::Unbound` and `reorder` answered `Slot::Nothing`; this is
    /// that answer for an arm, and `bind` skips it rather than binding a
    /// buffer the shader would not read.
    Unbound,
    /// One LAYER of the KV cache, keys or values.
    ///
    /// Not a `FireTable`: the pool is per-layer state and the layer is the
    /// rectangle's, so the ask carries no number and `plan` supplies it from
    /// `launch.layers.start` — exactly as `reorder` does for
    /// `kernels::Source::KvKeys`. An arm that carried its own layer could
    /// disagree with the rectangle it is planning.
    Kv {
        /// Values rather than keys.
        values: bool,
    },
}

/// A `_key_value` pair out of an entrypoint's axis suffix.
///
/// On this backend the affine point is in the NAME —
/// `affine_qmv_fast_bfloat16_gs_64_b_4` — where metal takes it as a launch
/// fact and carries it in its own `Facts`. A row's `axes` used to generate
/// these names, so reading one back is reading the row's own statement; the
/// table is gone and the name is not.
fn suffix(symbol: &str, key: &str) -> i32 {
    let mut parts = symbol.split('_');
    while let Some(part) = parts.next() {
        if part == key {
            return parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        }
    }
    0
}

// NO `group` OR `bits`, and that is a real difference from
// `driver-metal::lowering::arm::Facts`.
//
// Metal's fire carries the affine point because its kernels take it as a
// launch fact. On this backend the point is in the ENTRYPOINT NAME --
// `affine_qmv_fast_bfloat16_gs_64_b_4` -- and a body picks the whole spelling
// out of a literal table with it, which is `kernels-wgpu::layout`'s
// `affine_point` and the four tables beside it. `dispatch::Geometry` has no
// such field, so inventing one here would be a second statement of something
// the name already says.
//
// The arms that need it (`layout`'s gathers, `moe`'s routed GEMMs, all of
// `quant`) will read it where the table path does: off `Declared`, which is
// the module the shell resolved. That is the fork's to wire, not this
// struct's to carry.

/// Everything a launch knows about itself, as a body's arguments want it.
///
/// # Errors
///
/// Infallible: every field is read off the launch or the fire, and a launch
/// with no rows is refused before this by [`crate::geometry`].
#[must_use]
pub fn facts(
    symbol: &str,
    rows: u32,
    fire: Geometry,
    requests: u32,
    width: u32,
    in_width: u32,
) -> Facts {
    Facts {
        group: suffix(symbol, "gs"),
        bits: suffix(symbol, "b"),
        tile_m: suffix(symbol, "bm"),
        tile_n: suffix(symbol, "bn"),
        rows,
        width,
        in_width,
        requests,
        q_heads: fire.q_heads,
        kv_heads: fire.kv_heads,
        head_dim: fire.head_dim,
        rotary_dims: fire.rotary_dims,
        n_experts: fire.n_experts,
        experts_per_token: fire.experts_per_token,
        axis: fire.head_dim,
    }
}

/// The launch's operands, handed out as the handles a body binds.
///
/// A body's `Buf` is an opaque handle; this is what mints them. Each call
/// appends to [`Self::taken`] and returns the handle that indexes it, so the
/// ORDER a body asks in is the order the driver binds — which is the whole
/// point, and is what `every_routine_binds_a_buffer_for_every_binding_its_
/// module_declares` measures against the shader.
pub struct Handles<'a> {
    /// The statement's arguments, as the lowering states them.
    args: &'a [Arg],
    /// Which of them are inputs, outputs and weights, in that order.
    ins: Vec<usize>,
    outs: Vec<usize>,
    weights: Vec<usize>,
    /// What the body has asked for so far, in the order it asked.
    ///
    /// `None` is the PARAMETER BLOCK, which is not one of the statement's
    /// operands and has no `Arg`: it is the packed scalar run, and the driver
    /// stages it. It takes a place in this list because it takes a place in
    /// the body's argument list, and a handle that skipped it would point one
    /// operand short of what the body meant.
    taken: Vec<Asked>,
    /// The handle the parameter block was minted as, if the body asked.
    block: Option<u32>,
    /// The statement's own scalar run.
    scalars: &'a [u32],
    /// The fire's own numbers, as the resolver answered them.
    numbers: std::collections::BTreeMap<crate::binding::FireNumber, u32>,
}

impl<'a> Handles<'a> {
    /// The operands of one launch, ready to be asked for.
    ///
    /// Takes the operand SLICE and the op's result count, because that is all
    /// it reads, and the narrower argument is what lets this be tested
    /// without building a lowering.
    ///
    /// `results` is how many of the non-weight operands are OUTPUTS, and the
    /// rule is `driver-metal::lowering::arm::split`'s: weights are their own
    /// list, and of what is left the LAST `results` are the outputs. It is
    /// shared semantics rather than a wgpu choice — the lowering states a
    /// statement's reads before its writes — so the two backends read one
    /// convention the same way.
    ///
    /// A first draft here guessed instead: *"the first `Arena` is the
    /// output"*. `argmax_logits` has two of each and it refused its own
    /// statement.
    ///
    /// # The ORDER an arm asks in does not matter
    ///
    /// Worth stating because it is not obvious and it makes one class of
    /// sabotage look like a defect when it is not. A handle is an index into
    /// [`Self::taken`], assigned when the operand is ASKED for, and the body
    /// hands the handles back in its own order — so an arm that reads its
    /// second input before its first produces different indices AND a
    /// differently ordered `taken`, and the two shifts cancel exactly.
    ///
    /// What matters is which operand each NAME binds to. `gate = input(0)`
    /// against `gate = input(1)` is a real defect and
    /// `the_routine_path_plans_what_the_table_path_planned` fails on it;
    /// reordering the two `let`s is not, and it does not.
    #[must_use]
    pub fn over(args: &'a [Arg], results: usize) -> Self {
        Self::with_scalars(args, results, &[])
    }

    /// [`Self::over`] with the statement's scalar run, which some arms read.
    ///
    /// A row said this with `grid_param`: it named a param index and the grid
    /// read it. `norm` is where it starts to matter — `RmsParams.axis_size` is
    /// word 1 and it is NOT the row width, because a QK-norm packs
    /// `width / axis` reductions into each row and a grid sized per row
    /// normalizes head 0 and leaves the rest as the projection wrote them.
    /// Fully written, never reported.
    #[must_use]
    pub fn with_scalars(args: &'a [Arg], results: usize, scalars: &'a [u32]) -> Self {
        Self::with_numbers(args, results, scalars, &std::collections::BTreeMap::new())
    }

    /// [`Self::with_scalars`] with the fire's own numbers, which some arms read.
    #[must_use]
    pub fn with_numbers(
        args: &'a [Arg],
        results: usize,
        scalars: &'a [u32],
        numbers: &std::collections::BTreeMap<crate::binding::FireNumber, u32>,
    ) -> Self {
        let mut out = Self::build(args, results, scalars);
        out.numbers = numbers.clone();
        out
    }

    fn build(args: &'a [Arg], results: usize, scalars: &'a [u32]) -> Self {
        let widthed: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| !matches!(a, Arg::Weight(_)))
            .map(|(i, _)| i)
            .collect();
        let weights: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| matches!(a, Arg::Weight(_)))
            .map(|(i, _)| i)
            .collect();
        let results = results.min(widthed.len());
        let (ins, outs) = widthed.split_at(widthed.len() - results);
        Self {
            args,
            ins: ins.to_vec(),
            outs: outs.to_vec(),
            weights,
            taken: Vec::new(),
            block: None,
            scalars,
            numbers: std::collections::BTreeMap::new(),
        }
    }

    /// The `n`th INPUT, as a handle.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] when the statement has no such input, which is a
    /// disagreement between the arm and the trace rather than a caller's
    /// mistake.
    pub fn input(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self.ins.get(n).ok_or(Refusal::Empty {
            what: "an input operand the arm asked for",
        })?;
        Ok(self.take(at))
    }

    /// The `n`th OUTPUT, as a handle.
    ///
    /// # Errors
    ///
    /// As [`Self::input`].
    pub fn output(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self.outs.get(n).ok_or(Refusal::Empty {
            what: "an output operand the arm asked for",
        })?;
        Ok(self.take(at))
    }

    /// The `n`th WEIGHT, as a handle.
    ///
    /// # Errors
    ///
    /// As [`Self::input`].
    pub fn weight(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self.weights.get(n).ok_or(Refusal::Empty {
            what: "a weight operand the arm asked for",
        })?;
        Ok(self.take(at))
    }

    /// A scalar the STATEMENT may state, falling back to the fire's number.
    ///
    /// Zero is treated as ABSENT, exactly as `dims_of`'s `.filter(|n| *n > 0)`
    /// does: a grid axis of zero launches nothing, which is a silent no-op.
    /// `driver-metal::lowering::arm::stated` is the same function.
    #[must_use]
    pub fn stated(&self, i: usize, fire: u32) -> i32 {
        self.scalars
            .get(i)
            .copied()
            .filter(|n| *n > 0)
            .unwrap_or(fire)
            .cast_signed()
    }

    /// A scalar the statement MUST state, by index into its run.
    ///
    /// Unlike [`Self::stated`] there is no fire-wide fallback: a row pitch is
    /// a fact about how the trace laid this tensor out and nothing else knows
    /// it. A missing one is a disagreement between the arm and the trace.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] when the statement's run is shorter than `i`.
    pub fn param(&self, i: usize) -> Result<i32, Refusal> {
        self.scalars
            .get(i)
            .copied()
            .map(u32::cast_signed)
            .ok_or(Refusal::Empty {
                what: "a scalar operand the arm asked for",
            })
    }

    /// A scalar the statement states, read as an `f32`.
    ///
    /// The run is a `Vec<u32>` and a float rides it as its bit pattern, which
    /// is how `Source::ParamF32` reads it too.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] when the statement's run is shorter than `i`.
    pub fn param_f32(&self, i: usize) -> Result<f32, Refusal> {
        self.scalars
            .get(i)
            .copied()
            .map(f32::from_bits)
            .ok_or(Refusal::Empty {
                what: "a float scalar the arm asked for",
            })
    }

    /// The parameter block, as a handle that STANDS FOR the staged run.
    ///
    /// **The one thing this file cannot answer on its own.** A statement's
    /// scalars are packed into a block, and on this backend that block is
    /// usually a `@group(1)` uniform the encoder builds from the dispatch's
    /// own scalar list — so it has no address at plan time and no place in
    /// the buffer list at all. Where a shader declares it as a `@group(0)`
    /// storage entry instead, it does have a slot, and the two cases are told
    /// apart by the module rather than by the row.
    ///
    /// `kernels-metal` met this crossing its own `mlp` — *"the parameter
    /// block was a buffer with no address"* — and answered it with a handle
    /// that stands for the staged run, resolved to a packed slot at lay-out
    /// rather than to an address. This returns the same kind of handle and
    /// the resolution is the fork's to write, which is why nothing calls it
    /// yet.
    pub fn params_block(&mut self) -> ArgValue {
        let handle = u32::try_from(self.taken.len()).expect("a small operand count");
        self.taken.push(Asked::Params);
        self.block = Some(handle);
        ArgValue::Buffer(handle)
    }

    /// The handle the parameter block was minted as, if the body asked.
    #[must_use]
    pub const fn block(&self) -> Option<u32> {
        self.block
    }

    /// What the body asked for, in the order it asked.
    #[must_use]
    pub fn asked(&self) -> &[Asked] {
        &self.taken
    }

    /// A buffer the FIRE holds, as a handle.
    ///
    /// The statement does not name these — a rope table and a sampling index
    /// run belong to the fire, not to the op — so they cannot come out of
    /// `args`. The table path reaches the same buffers through
    /// `kernels::Source::RopeFrequencies` and its siblings, and this is that
    /// door for an arm.
    /// One LAYER of the KV cache, keys or values, as a handle.
    ///
    /// The layer is NOT a parameter: it is the rectangle's, and `plan` reads
    /// it off `launch.layers.start` when it resolves the ask. See
    /// [`Asked::Kv`].
    /// A binding the module declares and the statement does not fill.
    ///
    /// The body forwards a handle for it — the argument list has to match the
    /// module's numbering — and nothing is bound there. See [`Asked::Unbound`].
    pub fn unbound(&mut self) -> ArgValue {
        let handle = u32::try_from(self.taken.len()).expect("a small operand count");
        self.taken.push(Asked::Unbound);
        ArgValue::Buffer(handle)
    }

    /// One LAYER of the KV cache, keys or values, as a handle.
    ///
    /// The layer is NOT a parameter: it is the rectangle's, and `plan` reads
    /// it off `launch.layers.start`. See [`Asked::Kv`].
    pub fn kv(&mut self, values: bool) -> ArgValue {
        let handle = u32::try_from(self.taken.len()).expect("a small operand count");
        self.taken.push(Asked::Kv { values });
        ArgValue::Buffer(handle)
    }

    /// One of the FIRE's own numbers — a pool shape or a mask pitch.
    ///
    /// A statement does not carry these and must not: the mask's pitch is
    /// whatever the DRIVER made the widest row, and `driver-metal` reading a
    /// text scalar there is the live defect `DRIFTED["sdpa_paged_decode"]`
    /// names. Absent is ZERO, which is what the table path's `derived` does,
    /// and a zero page size refuses at the grid rather than reading garbage.
    #[must_use]
    pub fn fire_number(&self, which: crate::binding::FireNumber) -> u32 {
        self.numbers.get(&which).copied().unwrap_or(0)
    }

    /// A buffer the FIRE holds, as a handle.
    ///
    /// The statement does not name these — a rope table and a sampling index
    /// run belong to the fire, not to the op — so they cannot come out of
    /// `args`. The table path reaches the same buffers through
    /// `kernels::Source::RopeFrequencies` and its siblings.
    pub fn table(&mut self, which: FireTable) -> ArgValue {
        let handle = u32::try_from(self.taken.len()).expect("a small operand count");
        self.taken.push(Asked::Table(which));
        ArgValue::Buffer(handle)
    }

    /// The OPERANDS the body asked for, in the order it asked.
    ///
    /// The parameter block is not one: it is dropped here, so the indices of
    /// this slice are not handles. Use [`Self::asked`] where they must be.
    #[must_use]
    pub fn taken(&self) -> Vec<Arg> {
        self.taken
            .iter()
            .filter_map(|a| match a {
                Asked::Operand(arg) => Some(arg.clone()),
                Asked::Params | Asked::Table(_) | Asked::Kv { .. } | Asked::Unbound => None,
            })
            .collect()
    }

    /// Whether the body asked for its parameter block.
    #[must_use]
    pub const fn wants_block(&self) -> bool {
        self.block.is_some()
    }

    fn take(&mut self, at: usize) -> ArgValue {
        let handle = u32::try_from(self.taken.len()).expect("a small operand count");
        self.taken.push(Asked::Operand(self.args[at].clone()));
        ArgValue::Buffer(handle)
    }
}

/// One kernel's operand resolution.
///
/// Returns the argument list its BODY takes, which the body then turns into a
/// `Fire` and a bound list. The two halves are deliberately separate: this
/// one knows the trace and nothing about the shader, and the body knows the
/// shader and nothing about the trace.
pub type Arm = fn(&mut Handles<'_>, Facts) -> Result<Vec<ArgValue>, Refusal>;

/// `sample::argmax_logits`.
///
/// The first arm, and the only one, for the reason this module's docs give:
/// its row states no launch rule, so nothing has ever dispatched it through
/// the table and there is no prior behaviour to preserve.
///
/// # Errors
///
/// [`Refusal::Empty`] when the statement does not carry the four operands the
/// body takes.
pub fn argmax_logits(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let logits = o.input(0)?;
    let next_token = o.output(0)?;
    let params = o.input(1)?;
    let eos_flag = o.output(1)?;
    Ok(vec![
        logits,
        next_token,
        params,
        eos_flag,
        ArgValue::U32(f.rows),
    ])
}

/// `ptir::copy_logits_bf16`.
///
/// The second arm, and dark for the same reason the first is: this backend's
/// channel-plane interpreter never dispatches it, so no text names the symbol
/// and arming it cannot change what any model computes.
///
/// # `vocab` is the launch's WIDTH
///
/// The body takes `vocab` and halves it, because `logits_copy.wgsl` packs two
/// bf16 into a `u32` and one lane owns one word. What arrives here is the row
/// width the launch writes, which is the vocabulary — the halving is the
/// SHADER's fact and stays in the body, and `kernels-metal`'s row states the
/// unhalved `[vocab, rows, 1]` for the same kernel and is equally right about
/// its own shader. That split is `refactor-bigplan.md` §2.
///
/// # Errors
///
/// [`Refusal::Empty`] when the statement does not carry the three operands
/// the body takes.
pub fn copy_logits_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let source = o.input(0)?;
    let destination = o.output(0)?;
    let params = o.input(1)?;
    Ok(vec![
        source,
        destination,
        params,
        ArgValue::U32(f.width),
        ArgValue::U32(f.rows),
    ])
}

/// `mlp::silu_mul`.
///
/// **The first LIVE arm on this backend.** Every gated MLP names this symbol,
/// so unlike `sample` and `ptir` a mistake here changes what a real model
/// computes — which is why `the_routine_path_plans_what_the_table_path_planned`
/// exists and derives every field of its dispatch twice.
///
/// The one kernel in `mlp/gated.wgsl` that reads no parameter block: it needs
/// no scalar the grid does not give it. That is what makes it armable before
/// the block's resolution is written, and it is the same reason
/// `driver-vulkan` gives at its own crossing.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn silu_mul(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let gate = o.input(0)?;
    let up = o.input(1)?;
    let out = o.output(0)?;
    Ok(vec![
        gate,
        up,
        out,
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `mlp::geglu_tanh`, `mlp::geglu_tanh_strided` and `mlp::gptoss_swiglu`.
///
/// One arm for three routines because the STATEMENT is the same shape for all
/// three — two inputs, one result, one parameter block — and arms are about
/// statements. What differs is the activation, which is the body's business,
/// and the block's fields, which are the trace's: the driver forwards the
/// scalar run without knowing that gemma's third word is a gate pitch and
/// gpt-oss's is a clamp. `driver-vulkan`'s `gated` is the same arm.
///
/// # The block is STORAGE here
///
/// `gated.wgsl` declares `@group(0) @binding(3) var<storage> params`, so the
/// block takes a place in the numbering and its contents are the statement's
/// own scalar run — which is what the row said too, as `Param(0): Buf`. That
/// is a different thing from the `@group(1)` uniform most of this backend's
/// kernels read, and `bind` tells them apart by which one the body asked for.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn gated(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let gate = o.input(0)?;
    let up = o.input(1)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        gate,
        up,
        out,
        params,
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `mlp::geglu_tanh`. See [`gated`].
///
/// # Errors
///
/// See [`gated`].
pub fn geglu_tanh(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    gated(o, f)
}

/// `mlp::geglu_tanh_strided`. See [`gated`].
///
/// # Errors
///
/// See [`gated`].
pub fn geglu_tanh_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    gated(o, f)
}

/// `mlp::gptoss_swiglu`. See [`gated`].
///
/// # Errors
///
/// See [`gated`].
pub fn gptoss_swiglu(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    gated(o, f)
}

// ---------------------------------------------------------------------------
// layout
// ---------------------------------------------------------------------------

/// The four quantized embedding gathers, which differ by two flags.
///
/// `scaled` appends gemma's embedding scale; `many` appends the row count.
/// `driver-metal`'s `embed_gather` is the same arm with the same two flags.
///
/// The three quantized planes are WEIGHTS and the token ids are the FIRE's —
/// no statement names them, which is why `Handles::table` exists.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand or scalar the statement does not carry.
pub fn embed_gather(
    o: &mut Handles<'_>,
    f: Facts,
    scaled: bool,
    many: bool,
) -> Result<Vec<ArgValue>, Refusal> {
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    let id = o.table(FireTable::TokenIds);
    let out = o.output(0)?;
    let hidden = o.param(0)?;
    let mut args = vec![w, scales, biases, id, out, ArgValue::I32(hidden)];
    if scaled {
        args.push(ArgValue::F32(o.param_f32(1)?));
    }
    if many {
        args.push(ArgValue::I32(f.rows.cast_signed()));
    }
    args.push(ArgValue::I32(f.group));
    args.push(ArgValue::I32(f.bits));
    Ok(args)
}

/// `layout::embed_gather_4bit`. See [`embed_gather`].
///
/// # Errors
///
/// See [`embed_gather`].
pub fn embed_gather_4bit(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    embed_gather(o, f, false, false)
}

/// `layout::embed_gather_mb_4bit`. See [`embed_gather`].
///
/// # Errors
///
/// See [`embed_gather`].
pub fn embed_gather_mb_4bit(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    embed_gather(o, f, false, true)
}

/// `layout::embed_gather_scaled_4bit`. See [`embed_gather`].
///
/// # Errors
///
/// See [`embed_gather`].
pub fn embed_gather_scaled_4bit(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    embed_gather(o, f, true, false)
}

/// `layout::embed_gather_scaled_mb_4bit`. See [`embed_gather`].
///
/// # Errors
///
/// See [`embed_gather`].
pub fn embed_gather_scaled_mb_4bit(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    embed_gather(o, f, true, true)
}

/// `layout::ple_combine`: gemma's per-layer embedding, folded in.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn ple_combine(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let proj = o.input(0)?;
    let token = o.input(1)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        proj,
        token,
        out,
        params,
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `layout::row_gather`: the readout rows, gathered into request order.
///
/// The row indices are the FIRE's `SamplingIndices` and the count is the
/// request count, not the rectangle's rows: a request contributes one readout
/// row however many tokens it carried.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn row_gather(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let input = o.input(0)?;
    let out = o.output(0)?;
    let rows = o.table(FireTable::SamplingIndices);
    let params = o.params_block();
    Ok(vec![
        input,
        out,
        rows,
        params,
        ArgValue::U32(f.requests),
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

// ---------------------------------------------------------------------------
// norm
// ---------------------------------------------------------------------------

/// The four buffers every RMS form shares, and the axis it reduces over.
///
/// `axis` is `RmsParams.axis_size` — word 1 of the staged block, which is what
/// `grid_param = Some(1)` read. It is NOT the row width: a QK-norm packs
/// `width / axis` reductions into each row, and a grid sized per ROW
/// normalizes head 0 and leaves the rest as the projection wrote them. Fully
/// written, never reported — `driver-metal`'s `rms` says the same, and
/// `refactor-bigplan.md` §8d is the class.
///
/// Two rows stated no `grid_param` and got `axis = width` from the fallback:
/// the fused-residual forms, where the norm spans its row and the two numbers
/// coincide, so reading the field is the same answer with a reason.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn rms(o: &mut Handles<'_>, f: Facts, weighted: bool) -> Result<Rms, Refusal> {
    let x = o.input(0)?;
    let w = weighted.then(|| o.weight(0)).transpose()?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(Rms {
        x,
        w,
        out,
        params,
        width: f.width.cast_signed(),
        axis: o.stated(1, f.width),
        rows: f.rows.cast_signed(),
    })
}

/// What every RMS arm has in common. See [`rms`].
pub struct Rms {
    x: ArgValue,
    w: Option<ArgValue>,
    out: ArgValue,
    params: ArgValue,
    width: i32,
    axis: i32,
    rows: i32,
}

impl Rms {
    /// `x, w, out, params` — the shader's order, which is not the trace's.
    pub fn head(&self) -> Vec<ArgValue> {
        let mut v = vec![self.x.clone()];
        if let Some(w) = &self.w {
            v.push(w.clone());
        }
        v.push(self.out.clone());
        v.push(self.params.clone());
        v
    }
}

/// `norm::rms_single_row`. See [`rms`].
///
/// # Errors
///
/// See [`rms`].
pub fn rms_single_row(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let r = rms(o, f, true)?;
    let mut v = r.head();
    v.extend([
        ArgValue::I32(r.width),
        ArgValue::I32(r.axis),
        ArgValue::I32(r.rows),
    ]);
    Ok(v)
}

/// `norm::vnorm_single_row`: gemma's value norm, which has no gain.
///
/// The absent weight is the whole difference from [`rms_single_row`], and it
/// renumbers the buffers — `out` is 1 here and 2 there.
///
/// # Errors
///
/// See [`rms`].
pub fn vnorm_single_row(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let r = rms(o, f, false)?;
    let mut v = r.head();
    v.extend([
        ArgValue::I32(r.width),
        ArgValue::I32(r.axis),
        ArgValue::I32(r.rows),
    ]);
    Ok(v)
}

/// `norm::rms_strided_row`: the same norm over rows a pitch apart.
///
/// This backend's body takes `row_pitch` and `rows` and no width or axis — the
/// shader reads those from the block — which is a real difference from
/// `kernels-metal`'s signature for the same kernel and the reason this is
/// written against the wgpu body rather than copied from metal's arm.
///
/// # Errors
///
/// See [`rms`], plus [`Refusal::Empty`] when the statement states no pitch.
pub fn rms_strided_row(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let pitch = o.param(0)?;
    let r = rms(o, f, true)?;
    let mut v = r.head();
    v.extend([ArgValue::I32(pitch), ArgValue::I32(r.rows)]);
    Ok(v)
}

/// `norm::rms_strided_head_row`: a strided QK-norm, `heads` per row.
///
/// `heads` is `width / axis`, which is how many reductions a row holds.
///
/// # Errors
///
/// See [`rms_strided_row`].
pub fn rms_strided_head_row(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let pitch = o.param(0)?;
    let r = rms(o, f, true)?;
    let heads = if r.axis > 0 { r.width / r.axis } else { 1 };
    let mut v = r.head();
    v.extend([
        ArgValue::I32(pitch),
        ArgValue::I32(heads),
        ArgValue::I32(r.rows),
    ]);
    Ok(v)
}

/// `norm::rms_residual`: the norm with its residual folded in.
///
/// # Errors
///
/// See [`rms`].
pub fn rms_residual(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let r = rms(o, f, true)?;
    let mut v = r.head();
    v.extend([
        residual,
        ArgValue::I32(r.width),
        ArgValue::I32(r.axis),
        ArgValue::I32(r.rows),
    ]);
    Ok(v)
}

/// `norm::rms_residual_scaled`: [`rms_residual`] with a per-layer scale.
///
/// # Errors
///
/// See [`rms`].
pub fn rms_residual_scaled(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let scale = o.input(2)?;
    let r = rms(o, f, true)?;
    let mut v = r.head();
    v.extend([
        residual,
        scale,
        ArgValue::I32(r.width),
        ArgValue::I32(r.axis),
        ArgValue::I32(r.rows),
    ]);
    Ok(v)
}

/// `norm::gated_rms`: the gated per-head norm, `heads` of them per row.
///
/// The pair of numbers is the POOL's, not the projection's: `LaunchRule::GatedRms`
/// said `kv_heads` by `head_dim`, and a grid built from the query heads would
/// normalize past the end of the row. `refactor-bigplan.md` §8d records the
/// defect this exact kernel had on this backend — a prefill normalized only
/// its first row.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn gated_rms(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let x = o.input(0)?;
    let z = o.input(1)?;
    let w = o.weight(0)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        x,
        z,
        w,
        out,
        params,
        ArgValue::I32(f.kv_heads.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `norm::gated_rms_strided`: [`gated_rms`] over rows a pitch apart.
///
/// # Errors
///
/// See [`gated_rms`], plus [`Refusal::Empty`] for a missing pitch.
pub fn gated_rms_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let pitch = o.param(0)?;
    let x = o.input(0)?;
    let z = o.input(1)?;
    let w = o.weight(0)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        x,
        z,
        w,
        out,
        params,
        ArgValue::I32(pitch),
        ArgValue::I32(f.kv_heads.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `norm::layer_scalar_mul`: a per-layer scalar, read from a WEIGHT.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn layer_scalar_mul(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let x = o.input(0)?;
    let scalar = o.weight(0)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        x,
        scalar,
        out,
        params,
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `norm::residual_add`: the one norm-family kernel with no parameter block.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn residual_add(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let x = o.input(0)?;
    let residual = o.input(1)?;
    let out = o.output(0)?;
    Ok(vec![
        x,
        residual,
        out,
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `norm::residual_add_strided`: [`residual_add`] over rows a pitch apart.
///
/// # Errors
///
/// See [`residual_add`], plus [`Refusal::Empty`] for a missing pitch.
pub fn residual_add_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let pitch = o.param(0)?;
    let x = o.input(0)?;
    let residual = o.input(1)?;
    let out = o.output(0)?;
    Ok(vec![
        x,
        residual,
        out,
        ArgValue::I32(pitch),
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `norm::add_bias`: IN PLACE, so the result is the first buffer.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn add_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let out = o.output(0)?;
    let bias = o.weight(0)?;
    Ok(vec![
        out,
        bias,
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

// ---------------------------------------------------------------------------
// rope
// ---------------------------------------------------------------------------

/// The rotation every neox form shares: the tensor, the positions, and the
/// three axes its grid is built from.
///
/// # The tensor is an OUTPUT, and asking for it as an input refuses everything
///
/// Every entrypoint in `rope/neox.wgsl` takes ONE tensor — `@group(0)
/// @binding(0) var<storage, read_write> x` — read and written, which is why
/// the bodies here take a single `BufMut` and no `Buf` beside it. The
/// statement matches: `model-dsl`'s `rope_one` records `vec![x.id]` and states
/// no separate result, precisely so the result binds the buffer the shader
/// mutates. [`crate::lowering::routine::results`] then counts the body's one
/// writable argument, so [`Handles::with_scalars`] splits that single widthed
/// operand into ZERO inputs and one output. `o.input(0)` here would refuse
/// every rotation in the tree; `o.output(0)` is the tensor.
///
/// It is also why a statement carrying q and k has to be two launches. One
/// launch rotating both was the shape this family had, and the second tensor
/// was never turned at all — invisible at position zero, where rope is the
/// identity.
///
/// # `position` belongs to the FIRE
///
/// No statement names it, exactly as `row_gather`'s indices are not named:
/// the rows said `Source::Positions` and an arm says [`FireTable::Positions`].
///
/// # `head_dim` and `rotary` are the statement's, with the fire behind them
///
/// This is what `head_param` and `grid_param = Some(3)` said, and the fallback
/// is not a nicety: gemma-4 rotates a quarter of each full-attention head and
/// all of each sliding one over the same tensor, so no fire-wide
/// `rotary_dims` is right for both layer kinds, while every single-shape
/// deployment states nothing and means the fire's number. [`Handles::stated`]
/// treats zero as absent for the reason `dims_of` filters it: a grid axis of
/// zero launches nothing and reports success.
///
/// Reading it ONCE also repairs something the table path had wrong. There the
/// shader was handed the raw `Source::Param(2)` while `dims_of` handed the
/// GRID `stated(head_param).unwrap_or(fire.head_dim)` — so a statement that
/// stated no head width dispatched a kernel told `head_dim = 0` over a grid
/// sized by the fire, and `pie_theta` divided by it. One number now, or
/// `rope::rope_grid`'s refusal.
///
/// # Why `head_dim_at` is a parameter
///
/// Because `inv_freq` takes `base`'s place in the scalar run. The geometric
/// forms state `scale, base, head_dim, rotary` and the `_freqs` pair states
/// `scale, head_dim, mscale, rotary` — the row's own `ParamF32(0)`,
/// `Param(1)`, `ParamF32(2)` — so the head width is param 2 in one shape and
/// param 1 in the other. Transcribing the geometric index into the frequency
/// arms would read YaRN's `mscale` as a head width: a plausible small number,
/// a wrong grid, and a rotation that still completes.
///
/// # Errors
///
/// [`Refusal::Empty`] when the statement carries no tensor, or no `scale` —
/// the one scalar every spelling of this family states first.
pub fn neox(o: &mut Handles<'_>, f: Facts, head_dim_at: usize) -> Result<Rotation, Refusal> {
    let x = o.output(0)?;
    let position = o.table(FireTable::Positions);
    Ok(Rotation {
        x,
        position,
        scale: o.param_f32(0)?,
        head_dim: o.stated(head_dim_at, f.head_dim),
        rotary: o.stated(3, f.rotary_dims),
        width: f.width.cast_signed(),
        rows: f.rows.cast_signed(),
    })
}

/// What every neox arm has in common. See [`neox`].
pub struct Rotation {
    x: ArgValue,
    position: ArgValue,
    scale: f32,
    head_dim: i32,
    rotary: i32,
    width: i32,
    rows: i32,
}

/// `rope::neox_decode`: the geometric ladder, ONE row.
///
/// `base` is `log2(theta)`, not `theta`: `pie_theta` raises two to it, and
/// handing it the period rotates by a frequency ladder that is wrong from the
/// second channel on. The trace states it that way — `theta.log2().to_bits()`
/// — so the arm only has to read param 1 as the float it is.
///
/// The body's `rows` is not here at all: the decode entrypoint is the
/// single-row shape and its grid states `1`.
///
/// # Errors
///
/// [`Refusal::Empty`] for the tensor, or for `scale`/`base`, which are the two
/// scalars no deployment leaves out.
pub fn neox_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let base = o.param_f32(1)?;
    let r = neox(o, f, 2)?;
    Ok(vec![
        r.x,
        r.position,
        ArgValue::F32(r.scale),
        ArgValue::F32(base),
        ArgValue::I32(r.head_dim),
        ArgValue::I32(r.rotary),
        ArgValue::I32(r.width),
    ])
}

/// `rope::neox_mb`: [`neox_decode`] over one row per token.
///
/// The row count is the only difference in the argument list, and it is the
/// whole difference in behaviour: `neox_mb_bfloat16` is the symbol every
/// non-rescaled deployment in this tree actually names, for a decode of four
/// requests as much as for a prefill, because the class does not answer how
/// many rows a fire has.
///
/// # Errors
///
/// See [`neox_decode`].
pub fn neox_mb(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let base = o.param_f32(1)?;
    let r = neox(o, f, 2)?;
    Ok(vec![
        r.x,
        r.position,
        ArgValue::F32(r.scale),
        ArgValue::F32(base),
        ArgValue::I32(r.head_dim),
        ArgValue::I32(r.rotary),
        ArgValue::I32(r.width),
        ArgValue::I32(r.rows),
    ])
}

/// `rope::neox_prop_decode`: gemma's ladder over a PROPORTIONAL slice.
///
/// The statement is [`neox_decode`]'s to the scalar — its row states the same
/// `ParamF32(0)`, `ParamF32(1)`, `Param(2)` and the same `grid_param` — so the
/// arm is that arm. What differs is inside `pie_theta`: the proportional
/// exponent divides by the WHOLE head while only `rotary` channels turn, where
/// the geometric one divides by the rotated half. Same operands, different
/// angles, which is a body's business and not an arm's.
///
/// # Errors
///
/// See [`neox_decode`].
pub fn neox_prop_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    neox_decode(o, f)
}

/// `rope::neox_prop_mb`: gemma's PREFILL rotation.
///
/// Its `kernel!` row is BARE — no operands and no launch rule — so
/// [`crate::geometry`] has always refused it `Unstated` and the table path has
/// never dispatched this symbol at all. Arming it cannot change what any
/// model computes today; it can only stop refusing a gemma prefill.
///
/// The binding order is [`neox_mb`]'s because the shader's is: one
/// `rope/neox.wgsl` declares `x`, `position` and a `Params` of
/// `scale, base, head_dim` for every non-freqs, non-strided spelling. The
/// trace agrees — its geometric branch packs `scale`, `log2(theta)`,
/// `head_dim` and the rotary width whichever proportional or geometric symbol
/// it names — so the proportional forms read their scalars exactly where the
/// geometric ones do.
///
/// # Errors
///
/// See [`neox_decode`].
pub fn neox_prop_mb(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    neox_mb(o, f)
}

/// `rope::neox_freqs_decode`: the ladder READ rather than raised.
///
/// llama-3's piecewise interpolation and YaRN's rescaling are neither of them
/// a base, so no exponent can express either and the frequencies arrive as a
/// buffer — the fire's [`FireTable::RopeFrequencies`], staged by
/// `crate::rope::words` at load. `base` is therefore ABSENT rather than
/// ignored: this shape's `Params` struct is `scale, head_dim, mscale` and has
/// no slot for one, which is the same reason `head_dim` sits one earlier in
/// the run.
///
/// `mscale` is YaRN's attention-temperature correction. It rides the rotation
/// rather than a dispatch of its own because rotation is linear, and it is
/// `1.0` for every llama-3 deployment, whose rescaling is entirely in the
/// frequencies.
///
/// `inv_freq` is asked for AFTER the tensor and the positions so the handles
/// run 0, 1, 2 in the body's own order — which is also the shader's binding
/// order, `x`, `position`, `inv_freq`. Reading metal's buffer numbers instead,
/// where `inv_freq` is 3 and `head_dim` is 4, is how a rotation ends up
/// reading the frequency table's address as its head width.
///
/// # Errors
///
/// [`Refusal::Empty`] for the tensor, `scale`, or `mscale`.
pub fn neox_freqs_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mscale = o.param_f32(2)?;
    let r = neox(o, f, 1)?;
    let inv_freq = o.table(FireTable::RopeFrequencies);
    Ok(vec![
        r.x,
        r.position,
        ArgValue::F32(r.scale),
        inv_freq,
        ArgValue::I32(r.head_dim),
        ArgValue::F32(mscale),
        ArgValue::I32(r.rotary),
        ArgValue::I32(r.width),
    ])
}

/// `rope::neox_freqs_mb`: [`neox_freqs_decode`] over one row per token.
///
/// The rotation a llama-3.1, llama-3.2 or any YaRN deployment takes, and the
/// only rescaled symbol this tree's texts name. Its row was bare once, so a
/// statement had nothing to name and named the DECODE symbol instead: a
/// single-row kernel over a multi-row grid, rotating row zero and leaving
/// every row after it as the projection wrote it. Rope is the identity at
/// position zero, so row zero agreed with the reference and the failure was
/// silent.
///
/// # Errors
///
/// See [`neox_freqs_decode`].
pub fn neox_freqs_mb(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mscale = o.param_f32(2)?;
    let r = neox(o, f, 1)?;
    let inv_freq = o.table(FireTable::RopeFrequencies);
    Ok(vec![
        r.x,
        r.position,
        ArgValue::F32(r.scale),
        inv_freq,
        ArgValue::I32(r.head_dim),
        ArgValue::F32(mscale),
        ArgValue::I32(r.rotary),
        ArgValue::I32(r.width),
        ArgValue::I32(r.rows),
    ])
}

/// `rope::neox_strided`: the ladder over rows that do not tile.
///
/// A DIFFERENT kernel from [`neox_mb`], not a flag on it: `neox_strided`'s
/// `Params` is `scale, base, head_dim, row_pitch`, so it binds one scalar more
/// and walks rows by a pitch the contiguous body derives from the width. A
/// packed QKV projection is where that arises — q and k share a buffer, so
/// rotating q means striding over k — and a prefill's scratch rows are a
/// uniform pitch apart that is wider than `n_head * head_dim`, which is
/// exactly the case the packed stride walks into the next row for.
///
/// # The pitch is param 4, NOT param 0
///
/// Worth stating because the rest of this file reads a pitch at zero:
/// `rms_strided_row`, `gated_rms_strided` and `residual_add_strided` all say
/// `o.param(0)?`, since a norm's run begins with it. This family's run does
/// not. Params 0..3 are the rotation's fixed preamble — `scale`, `base`,
/// `head_dim`, `rotary`, which the rows spell as `ParamF32(0)`, `ParamF32(1)`,
/// `Param(2)` and `grid_param = Some(3)` — so the pitch can only be the scalar
/// after them. `driver-metal` and `driver-vulkan` read param 4 here for the
/// same reason. Reading param 0 instead would hand the shader `scale`'s BIT
/// PATTERN as a row stride: a huge number, every row after the first read from
/// far outside the tensor.
///
/// A statement that carries no pitch is REFUSED rather than given the row
/// width, since a pitch equal to the width is precisely the case this kernel
/// is not for.
///
/// Its row is bare like `neox_prop_mb`'s, so nothing has dispatched this
/// symbol either.
///
/// # Errors
///
/// [`Refusal::Empty`] for the tensor, `scale`, `base` or `row_pitch`.
pub fn neox_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let base = o.param_f32(1)?;
    let row_pitch = o.param(4)?;
    let r = neox(o, f, 2)?;
    Ok(vec![
        r.x,
        r.position,
        ArgValue::F32(r.scale),
        ArgValue::F32(base),
        ArgValue::I32(r.head_dim),
        ArgValue::I32(row_pitch),
        ArgValue::I32(r.rotary),
        ArgValue::I32(r.width),
        ArgValue::I32(r.rows),
    ])
}

// ---------------------------------------------------------------------------
// ssm
// ---------------------------------------------------------------------------
//
// Gated DeltaNet, and the one family whose shape is a CHOICE rather than a
// kernel: `gdn_core` fuses the whole step, and `gdn_prep` plus
// `gdn_core_recurrent` split the same arithmetic in two so the work every value
// channel would redo is staged once into three f32 scratch slabs. The two paths
// must agree bit for bit, which is why the seam is written into the types —
// `pre_q`/`pre_k`/`pre_gate` are `F32sMut` in the prep and `F32s` in the
// recurrent half — and why an arm that crossed those wires would produce a
// model that runs.
//
// `_slotted` is exactly one more buffer: the shader reads `slot_ids[b]` instead
// of taking the z group's request index as its state slot, which is what lets a
// batch address scattered slots of a shared arena. `_prefill` is not a suffix
// with one meaning: on `gdn_prep` it is *slotted, plus a row pitch and a scan
// length*, and on `gdn_core_recurrent` it is a DIFFERENT KERNEL — seven buffers
// and no convolution at all, because the scan carries its state in registers
// across the whole prompt and reads no weights.
//
// # Where the recurrent state comes from on this backend
//
// `driver-metal` and `driver-vulkan` reach `conv_state`, `rstate` and
// `new_conv_state` through `Handles::slab`, because on those backends the GDN
// slabs are implicit per-layer state — `model_ir::trace::StateStore::
// RecurrentState`, the KV cache's sibling — and no traced value stands for
// them. This backend's `Handles` has no such door and `FireTable` names no GDN
// slab, so the three arrive as the statement's own operands: the read state as
// an INPUT and the two written ones as OUTPUTS.
//
// That is not a workaround, it is what `lowering::routine::results` already
// says out loud: it counts every writable `kernels::Ty` and not just `BufMut`
// *"because `ssm`'s bodies take `F32sMut` for their recurrent state and a count
// that missed those would split a statement in the wrong place"*. A `gdn_core`
// statement therefore has THREE results — `rstate`, `core_out`,
// `new_conv_state` — and a `gdn_prep` statement four. Every arm below maps the
// writable buffers to `output(0..)` and the readable non-weight buffers to
// `input(0..)` in the order the BODY lists them, which is the only ordering
// either half states.

/// The four gate weights every fused or prep GDN dispatch reads.
///
/// Weights 2 through 5 of the statement, and the two halves of the split pair
/// disagree about whether they are wanted at all: `gdn_core` and `gdn_prep`
/// derive the gates themselves, and `gdn_core_recurrent` reads no gate weight
/// at all — the prep already folded them into `pre_gate` — and takes only the
/// convolution pair. So this is a helper and not a prologue: asking for it in
/// the recurrent arm would bind `a_log` where that shader reads `pre_q`.
///
/// `a_log` is `F32s` and the other three are `Buf`; both spellings unpack from
/// the same `ArgValue::Buffer`, so the distinction lives in the body and the
/// shader rather than here. The order is the one `driver-metal::lowering::arm::
/// gates` and `driver-vulkan::arm::gates` state, and it is the same order
/// because it is the same lowering underneath all three.
///
/// # Errors
///
/// [`Refusal::Empty`] for a weight the statement does not carry — a GDN
/// statement that names four weights instead of six is a disagreement between
/// the arm and the trace, not a caller's mistake.
fn gates(o: &mut Handles<'_>) -> Result<[ArgValue; 4], Refusal> {
    let a_log = o.weight(2)?;
    let dt_bias = o.weight(3)?;
    let a_gate = o.weight(4)?;
    let b_gate = o.weight(5)?;
    Ok([a_log, dt_bias, a_gate, b_gate])
}

/// The fused core, with or without its slot map.
///
/// Three buffers of this signature are recurrent STATE and they are not
/// interchangeable. `conv_state` arrives holding the previous step's taps and
/// `new_conv_state` receives the rolled ones — a SEPARATE buffer on purpose,
/// because the shader is still reading the old taps while it writes the new —
/// so the arm must ask for a read handle and a write handle and never the same
/// one twice. `rstate` is updated in place and is both.
///
/// `slotted` appends `slot_ids` immediately after the parameter block, which is
/// where the shader declares it: after every buffer and before the three grid
/// numbers. `driver-metal` reached the same position by `Vec::insert(12, ..)`
/// and records that inserting at thirteen — one place later, after `rows` —
/// shipped for a while, with the table arriving where the row count belongs.
/// Appending in the body's own order is why that index does not appear here.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry, which for
/// this family includes the three state buffers: see the module note on why
/// they are operands on this backend and slabs on the other two.
pub fn core(o: &mut Handles<'_>, f: Facts, slotted: bool) -> Result<Vec<ArgValue>, Refusal> {
    let mixed = o.input(0)?;
    let conv_state = o.input(1)?;
    let rstate = o.output(0)?;
    let core_out = o.output(1)?;
    let conv_w = o.weight(0)?;
    let conv_b = o.weight(1)?;
    let g = gates(o)?;
    let new_conv_state = o.output(2)?;
    let params = o.params_block();
    let slot_ids = slotted.then(|| o.input(2)).transpose()?;
    let mut v = vec![mixed, conv_state, rstate, core_out, conv_w, conv_b];
    v.extend(g);
    v.extend([new_conv_state, params]);
    v.extend(slot_ids);
    v.extend([
        ArgValue::I32(f.rows.cast_signed()),
        ArgValue::I32(f.kv_heads.cast_signed()),
        ArgValue::I32(f.head_dim.cast_signed()),
    ]);
    Ok(v)
}

/// `ssm::gdn_core`: convolution, gates, the delta rule and the writeback in one
/// dispatch. See [`self::core`].
///
/// The three grid numbers are the FIRE's and not the statement's. `v_heads` is
/// `kv_heads` and `v_dim` is `head_dim`, which is how a GDN launch spells its
/// value pool on every backend — `driver-metal` and `driver-vulkan` pass the
/// same two fields for the same two arguments. The body wants both factors of
/// its z extent rather than their product, because the shader recovers
/// `hv = z % Hv` and `row = z / Hv` and cannot tell one factorisation from
/// another.
///
/// # Errors
///
/// See [`self::core`].
pub fn gdn_core(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    core(o, f, false)
}

/// `ssm::gdn_core_slotted`: [`gdn_core`] addressing its state through a slot
/// map. See [`self::core`].
///
/// # Errors
///
/// See [`self::core`].
pub fn gdn_core_slotted(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    core(o, f, true)
}

/// The prep half of the split pair, in its three forms.
///
/// It writes FOUR buffers and that is the sharpest case in the family for the
/// result count: `pre_q`, `pre_k`, `pre_gate` and `new_conv_state` are all
/// `F32sMut` and none is a `BufMut`, so a count that read only `BufMut` would
/// say this statement has ZERO results, split it as all-inputs, and refuse the
/// first `output(0)` below. `lowering::routine::results` counts every writable
/// type for exactly this reason — see the NOTES at the end of this file for
/// what that does and does not settle.
///
/// The convolution writeback is SPLIT across the pair — this half rolls the q
/// and k channels and [`recurrent`] rolls the v channels — so `new_conv_state`
/// is whole only if both dispatches ran against the same buffer. Nothing here
/// can check that; what it can do is ask for it as this statement's last
/// result rather than inventing a second one.
///
/// `prefill` is the pitched form: it always carries a slot map (the shader
/// declares `slot_ids` under `PIE_SLOTTED` OR `PIE_PREFILL`, so a prefill has
/// one whether or not the deployment is slotted) and it states `row_pitch` and
/// `n_scan` itself, because a prompt is a strided run of tokens rather than one
/// row per request and no fire fact knows how the trace laid it out.
///
/// The two scalars are asked for FIRST, before any handle is minted. Ask order
/// does not change what gets bound — a handle is an index assigned at ask time
/// and the body hands them back in its own order — so this is only about a
/// refusal being cheap: a prefill statement with no pitch declines before the
/// operands are walked.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry, or for the
/// pitch or scan length a prefill does not state.
pub fn prep(
    o: &mut Handles<'_>,
    f: Facts,
    slotted: bool,
    prefill: bool,
) -> Result<Vec<ArgValue>, Refusal> {
    let row_pitch = prefill.then(|| o.param(0)).transpose()?;
    let n_scan = prefill.then(|| o.param(1)).transpose()?;
    let mixed = o.input(0)?;
    let conv_state = o.input(1)?;
    let conv_w = o.weight(0)?;
    let conv_b = o.weight(1)?;
    let g = gates(o)?;
    let pre_q = o.output(0)?;
    let pre_k = o.output(1)?;
    let pre_gate = o.output(2)?;
    let new_conv_state = o.output(3)?;
    let params = o.params_block();
    let slot_ids = (slotted || prefill).then(|| o.input(2)).transpose()?;
    let mut v = vec![mixed, conv_state, conv_w, conv_b];
    v.extend(g);
    v.extend([pre_q, pre_k, pre_gate, new_conv_state, params]);
    v.extend(slot_ids);
    v.extend(row_pitch.map(ArgValue::I32));
    v.extend(n_scan.map(ArgValue::I32));
    v.extend([
        ArgValue::I32(f.rows.cast_signed()),
        ArgValue::I32(f.kv_heads.cast_signed()),
    ]);
    Ok(v)
}

/// `ssm::gdn_prep`: everything a value channel would otherwise redo, staged
/// once. See [`prep`].
///
/// Two grid numbers and not three: a prep has no value channel to spread over,
/// so its grid is `[32, 1, rows * v_heads]` and the body never asks for
/// `v_dim`. Producing the shared work once is the whole point of the split, and
/// an arm that passed a third number anyway would be one value too long for a
/// signature that has nowhere to put it.
///
/// # Errors
///
/// See [`prep`].
pub fn gdn_prep(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    prep(o, f, false, false)
}

/// `ssm::gdn_prep_slotted`: [`gdn_prep`] with the slot map. See [`prep`].
///
/// # Errors
///
/// See [`prep`].
pub fn gdn_prep_slotted(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    prep(o, f, true, false)
}

/// `ssm::gdn_prep_prefill`: the prep over a whole PROMPT. See [`prep`].
///
/// `rows` is the launch's rectangle height, which for a prefill is the TOKEN
/// count rather than the request count — every token of the prompt needs its
/// own convolution — so the z extent `rows * v_heads` is `tokens * Hv`, which
/// is what the body's own doc says it wants.
///
/// This backend's body takes `rows` and takes `row_pitch`/`n_scan` as TRACE
/// scalars; `kernels-metal`'s takes neither a `rows` nor a trace scalar, and
/// marks both pitch numbers `Env<i32>`. The two are not the same signature and
/// this arm is written against the wgpu one.
///
/// # Errors
///
/// See [`prep`].
pub fn gdn_prep_prefill(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    prep(o, f, true, true)
}

/// The recurrent half of the split pair, with or without its slot map.
///
/// Eleven buffers where the fused core has twelve, and the eleven are not the
/// twelve minus one: `rstate` and `core_out` come back where the prep had
/// neither, the four gate weights are GONE, and the three scratch slabs arrive
/// as reads in the positions `gdn_core` gives to `a_log`, `dt_bias` and
/// `a_gate`. The first six match exactly, which is what makes the rest
/// dangerous — reusing the fused arm's vector here would bind `a_log` where
/// this shader reads `pre_q` and `b_gate`, a read-only gate weight, where it
/// writes `new_conv_state`. Buffers of the right kind at the wrong offsets,
/// which dispatches and answers.
///
/// # Errors
///
/// See [`self::core`].
pub fn recurrent(o: &mut Handles<'_>, f: Facts, slotted: bool) -> Result<Vec<ArgValue>, Refusal> {
    let mixed = o.input(0)?;
    let conv_state = o.input(1)?;
    let rstate = o.output(0)?;
    let core_out = o.output(1)?;
    let conv_w = o.weight(0)?;
    let conv_b = o.weight(1)?;
    let pre_q = o.input(2)?;
    let pre_k = o.input(3)?;
    let pre_gate = o.input(4)?;
    let new_conv_state = o.output(2)?;
    let params = o.params_block();
    let slot_ids = slotted.then(|| o.input(5)).transpose()?;
    let mut v = vec![
        mixed,
        conv_state,
        rstate,
        core_out,
        conv_w,
        conv_b,
        pre_q,
        pre_k,
        pre_gate,
        new_conv_state,
        params,
    ];
    v.extend(slot_ids);
    v.extend([
        ArgValue::I32(f.rows.cast_signed()),
        ArgValue::I32(f.kv_heads.cast_signed()),
        ArgValue::I32(f.head_dim.cast_signed()),
    ]);
    Ok(v)
}

/// `ssm::gdn_core_recurrent`: the scan over what [`gdn_prep`] staged. See
/// [`recurrent`].
///
/// # Errors
///
/// See [`self::core`].
pub fn gdn_core_recurrent(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    recurrent(o, f, false)
}

/// `ssm::gdn_core_recurrent_slotted`: [`gdn_core_recurrent`] with the slot map.
/// See [`recurrent`].
///
/// # Errors
///
/// See [`self::core`].
pub fn gdn_core_recurrent_slotted(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    recurrent(o, f, true)
}

/// `ssm::gdn_core_recurrent_prefill`: the prompt-long SCAN.
///
/// The one member of the family that is not a rearrangement of its neighbours.
/// Seven buffers and not eleven: the convolution and the gates were done once
/// by [`gdn_prep_prefill`], so this half reads no `mixed`, no `conv_w`, no
/// `conv_b` and no conv state at all, and it writes no `new_conv_state`. Two
/// results only — `rstate` and `core_out` — which is why `output(2)` would
/// refuse here and is asked for by every other arm in this family.
///
/// # `lanes` and `vrows` come from the STATEMENT, and the body calls them `Env`
///
/// The scan is compiled for nine `(LANES, VROWS)` shapes and the body picks its
/// entrypoint spelling out of a literal table with the pair. They are a
/// decomposition choice rather than a shape — `VROWS` is how many independent
/// value rows a lane group carries and `LANES` is the WIDTH of the reduction,
/// so only one of the two is free and two tilings hold bit-identical state
/// exactly when their `LANES` agree — which is why `driver-metal` and
/// `driver-vulkan` both read them out of the statement's scalar run at indices
/// 2 and 3, right behind the pitch and the scan length.
///
/// This body marks both `Env<i32>`, and this backend's [`Facts`] carries no
/// such fact: no `lanes`, no `vrows`, and an arm never sees the symbol the
/// suffix `_l_16_v_2` is written on. So they are read where the other two
/// backends read them, from the same statement, at the same two indices. The
/// disagreement is safe to have got wrong in one direction only: `scan_point`
/// refuses a pair it has no compiled module for rather than rounding to one it
/// has, so a wrong index surfaces as [`Refusal::Narrow`] instead of a scan at
/// the wrong tiling. If the `Env` marking is meant literally, the fix is a
/// `lanes`/`vrows` pair on `Facts` fed by `suffix(symbol, "l")` and
/// `suffix(symbol, "v")` exactly as `tile_m`/`tile_n` are fed today, and this
/// arm's two `param` calls become two field reads.
///
/// # The tail is not metal's tail
///
/// This body ends `lanes, vrows, dv, hv` and `kernels-metal`'s ends `v_heads,
/// v_dim, lanes, vrows`. Transcribing the metal arm would put the lane width
/// where the value width belongs, which `scan_point` would then read as a
/// tiling — a refusal for most shapes and a compiled point for a few.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry or for any of
/// the four scalars it does not state, and [`Refusal::Narrow`] from the body
/// for a `(lanes, vrows)` this tree compiles no scan for.
pub fn gdn_core_recurrent_prefill(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_pitch = o.param(0)?;
    let n_scan = o.param(1)?;
    let lanes = o.param(2)?;
    let vrows = o.param(3)?;
    let rstate = o.output(0)?;
    let core_out = o.output(1)?;
    let pre_q = o.input(0)?;
    let pre_k = o.input(1)?;
    let pre_gate = o.input(2)?;
    let params = o.params_block();
    let slot_ids = o.input(3)?;
    Ok(vec![
        rstate,
        core_out,
        pre_q,
        pre_k,
        pre_gate,
        params,
        slot_ids,
        ArgValue::I32(row_pitch),
        ArgValue::I32(n_scan),
        ArgValue::I32(lanes),
        ArgValue::I32(vrows),
        ArgValue::I32(f.head_dim.cast_signed()),
        ArgValue::I32(f.kv_heads.cast_signed()),
    ])
}

// ---- moe ----
//
// A section for `crates/driver-wgpu/src/lowering/arm.rs`: it uses that file's
// own imports (`kernels::routine::Refusal`, `kernels_wgpu::routine::ArgValue`,
// `Handles`, `Facts`, `Arm`) and adds none.
//
// Routing is a SORT. The router picks `experts_per_token` experts per token,
// `route_sort` orders the `(token, slot)` pairs so that every expert's rows are
// contiguous, the multiply runs once per tile against whichever expert that
// tile owns, and `combine_sorted` puts the results back where the tokens were.
// Six buffers carry the permutation between those steps -- `perm`, `inv`,
// `row_expert`, `tile_expert`, `pad`, `expert_ids` -- and every one of them is
// a traced operand, which is why this family asks for no `FireTable` at all.
//
// # Where the numbers live on THIS backend
//
// Two places, and the split is not the one metal makes. `moe/params.inc.wgsl`
// declares `RouterParams`, `ExpertCombineParams` and `MoeRouteParams` as
// `@group(0)` STORAGE blocks, and its own doc says why: the routing params are
// built by the host plan rather than carried in the statement, so they are
// already in device memory when the launch is assembled. Five routines take
// them as `params: Buf` -- both routers, the sort, the gather and the combine --
// and every one of those arms calls `Handles::params_block()` at exactly the
// position the body takes it, because `bind` reads the block's slot off its
// POSITION in the body's buffer list (`route_sort` puts an operand AFTER it, at
// binding 5, which is the case that makes position the only workable answer).
//
// The two shared-expert forms take their scalars as ordinary arguments instead:
// `route.wgsl`'s shared arm declares `@group(1) @binding(0) var<uniform> params`
// and the body passes `width` -- and, strided, `row_pitch` -- so those arms ask
// for no block and the driver stages a uniform.
//
// # The affine point is in the NAME here
//
// `qmm_t_routed`'s body indexes a literal table with
// `AFFINE_QMM[affine_qmm_point(*group, *bits, *tile_m, *tile_n)?]`, and those
// four are `Env` -- they come off the SYMBOL's `_gs_64_b_4_bm_32_bn_32` suffix,
// which `facts` already parsed into `Facts::group`, `bits`, `tile_m`, `tile_n`.
// Metal carries the same four as launch facts and `driver-metal`'s arms read
// them off its own `Facts`; wgpu reads them off the name. A symbol that names no
// point parses as zero and the BODY refuses it `Refusal::Narrow` before it
// dispatches, which is the loud failure -- there is no fallback tile and there
// must not be one, because a tile that is not the `tile_rows` the sort was given
// makes `tile_expert` a list indexed by something else.
//
// # Two routines this backend cannot arm yet
//
// `router_topk` and `qmv_routed` each take a `Buf` for a binding their shader
// DECLARES, never READS, and no statement supplies -- `per_expert_scale` at
// `route.wgsl`'s binding 4, `bias` at `qmv_routed.wgsl`'s binding 5. Both are
// positional slots kept so the scaled/biased twin can share the template.
// `driver-metal` fills them with `Handles::state(None)` and `driver-vulkan`
// with `Handles::unbound()`; this backend's `Handles` mints handles only for
// operands, the parameter block and fire tables, so an arm here has nothing
// truthful to put there. Re-asking a neighbouring weight WOULD compile and
// WOULD run -- the shader never reads the slot -- but it binds an address where
// the table path binds nothing, which is a plan that differs from the one
// `the_routine_path_plans_what_the_table_path_planned` compares against. Both
// stems are therefore registered with `arm: None`, which keeps them on the
// table path where they work today. See the note at the foot of this file.

/// `k` and `n`: the contraction and the output width every routed GEMM is told.
///
/// Both are the STATEMENT's, not the fire's: `n` is one expert's output width
/// and the rectangle this launch writes is the sorted stack, whose width is the
/// same number only when `experts_per_token` is one.
///
/// # Errors
///
/// [`Refusal::Empty`] when the statement's run is shorter than two words.
fn kn(o: &Handles<'_>) -> Result<(i32, i32), Refusal> {
    Ok((o.param(0)?, o.param(1)?))
}

/// `moe::router_topk_scaled`: the experts each token picks, with a per-expert
/// gain applied to the logits.
///
/// The gain is a second INPUT rather than a weight: it is a traced value, and
/// the unscaled twin is the same shader with the same five bindings reading
/// only four of them.
///
/// `n_experts` and `experts_per_token` do NOT appear here even though `Facts`
/// carries both. They are `RouterParams`' first two words and the body takes
/// the block as a pointer, so stating them again as arguments would be a second
/// copy of a number the shader already reads -- `driver-metal`'s arm passes
/// `n_experts` because ITS kernel takes it as a launch argument, and that is a
/// difference in the kernels rather than in the traces.
///
/// # Errors
///
/// [`Refusal::Empty`] for the logits, the gain or either result.
pub fn router_topk_scaled(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let logits = o.input(0)?;
    let expert_ids = o.output(0)?;
    let expert_weights = o.output(1)?;
    let params = o.params_block();
    let per_expert_scale = o.input(1)?;
    Ok(vec![
        logits,
        expert_ids,
        expert_weights,
        params,
        per_expert_scale,
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `moe::route_sort`: the `(token, slot)` pairs put in expert order.
///
/// FOUR results, which is the most any routine on this backend writes: `perm`
/// is where each sorted slot came from, `row_expert` is which expert owns each
/// sorted row, `tile_expert` is the same per GEMM tile -- which is what lets the
/// tiled multiply pick a weight matrix per workgroup without reading the routing
/// again -- and `inv` is where each pair went, which the combine reads back.
///
/// The parameter block sits at slot 4, BEFORE `inv`. That is the whole reason
/// an arm cannot hand a body an address for it and hands a HANDLE instead: this
/// family's ABI puts an operand after the block, so the block's binding number
/// is a position in the body's list rather than a tail.
///
/// The one arm here that reads no fact at all. `route_sort` launches one
/// workgroup whatever the row count -- the histogram is fire-wide -- so its body
/// takes no `Env` and there is nothing for `Facts` to answer.
///
/// # Errors
///
/// [`Refusal::Empty`] for the expert ids or any of the four results.
pub fn route_sort(o: &mut Handles<'_>, _f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let expert_ids = o.input(0)?;
    let perm = o.output(0)?;
    let row_expert = o.output(1)?;
    let tile_expert = o.output(2)?;
    let params = o.params_block();
    let inv = o.output(3)?;
    Ok(vec![expert_ids, perm, row_expert, tile_expert, params, inv])
}

/// `moe::route_gather`: the activation rows copied into expert order.
///
/// # `padded` is the SORTED extent and not the fire's token count
///
/// The sort rounds each expert's run up to a whole tile so that the multiply's
/// tiles never straddle two experts, which makes the gathered rectangle TALLER
/// than the fire's rectangle -- up to `experts_per_token` times as tall, plus
/// the padding. The statement states it as its fifth scalar, which is what the
/// row said with `rows_param = Some(4)` and what `MoeRouteParams::padded` is;
/// `driver-vulkan`'s arm spells the same index and says the same thing.
///
/// Given the fire's count instead, the gather ran over a quarter of its own
/// output at `top_k = 4` and left the rest whatever the arena held. Nothing
/// reports that: the dispatch succeeds, the tail experts read stale bytes, and
/// the combine weights them in.
///
/// `Handles::stated` falls back to the fire's rows for a trace that does not
/// state it, and treats a stated ZERO as absent for the reason `dims_of` does --
/// a grid axis of zero launches nothing and reports success.
///
/// # Errors
///
/// [`Refusal::Empty`] for the activation, the result or the permutation.
pub fn route_gather(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let padded = o.stated(4, f.rows);
    let x = o.input(0)?;
    let out = o.output(0)?;
    let perm = o.input(1)?;
    let params = o.params_block();
    Ok(vec![
        x,
        out,
        perm,
        params,
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(padded),
    ])
}

/// `moe::combine_sorted`: the experts' results weighted and summed back onto
/// their tokens.
///
/// The scatter that undoes [`route_gather`]'s gather, and its rows ARE the
/// fire's: one output row per token, however tall the sorted stack was. `inv`
/// is where each pair landed, so this reads a position rather than searching
/// for one, and -- like [`route_sort`] -- it sits AFTER the parameter block.
///
/// # Errors
///
/// [`Refusal::Empty`] for the expert results, the routing weights, the
/// destination or the inverse permutation.
pub fn combine_sorted(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let y = o.input(0)?;
    let expert_weights = o.input(1)?;
    let out = o.output(0)?;
    let params = o.params_block();
    let inv = o.input(2)?;
    Ok(vec![
        y,
        expert_weights,
        out,
        params,
        inv,
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `moe::shared_expert_combine`: `routed + sigmoid(gate) * shared`, the
/// always-on expert folded into the routed sum.
///
/// The one shape in this family with no parameter block: `route.wgsl`'s shared
/// arm reads a `@group(1)` uniform built from the scalars the BODY passes, so
/// the arm states `width` as a value and asks for no slot. `width` is the
/// statement's first scalar and the fire's row width is the fallback, which for
/// this kernel is the same number by construction -- the combine writes the
/// hidden state -- so the fallback is an answer with a reason rather than a
/// guess.
///
/// `out` MAY alias `routed`, which the driver does not need to know: an alias
/// is two names for one address and the binding is by address.
///
/// # Errors
///
/// [`Refusal::Empty`] for any of the three inputs or the result.
pub fn shared_expert_combine(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let width = o.stated(0, f.width);
    let routed = o.input(0)?;
    let shared = o.input(1)?;
    let gate = o.input(2)?;
    let out = o.output(0)?;
    Ok(vec![
        routed,
        shared,
        gate,
        out,
        ArgValue::U32(width.cast_unsigned()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `moe::shared_expert_combine_strided`: [`shared_expert_combine`] over rows a
/// `row_pitch` apart.
///
/// A DIFFERENT kernel, not a flag: `PIE_STRIDED` gives the uniform a second
/// field and moves the gate's index -- the contiguous arm reads `gate[r]`, one
/// value per row, and the strided arm reads `gate[r * row_pitch]`, because
/// `qmv_out_size` answers 1 for the shared gate projection and its single
/// column is written a full pitch apart like every other projection's. Handing
/// a strided rectangle to the contiguous body blends every row but row 0 with a
/// garbage weight, which is why the two stems are both registered.
///
/// The pitch is the statement's or a refusal, with no fire-wide fallback: it is
/// a fact about how the trace laid this tensor out and nothing else knows it.
/// No text names this symbol today, so this arm has never planned a fire.
///
/// # Errors
///
/// See [`shared_expert_combine`], plus [`Refusal::Empty`] for the pitch.
pub fn shared_expert_combine_strided(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let width = o.stated(0, f.width);
    let row_pitch = o.param(1)?;
    let routed = o.input(0)?;
    let shared = o.input(1)?;
    let gate = o.input(2)?;
    let out = o.output(0)?;
    Ok(vec![
        routed,
        shared,
        gate,
        out,
        ArgValue::U32(width.cast_unsigned()),
        ArgValue::I32(row_pitch),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `moe::qmv_routed_bias`: the routed affine matvec, with a per-output bias.
///
/// # `x` is NOT the gathered rectangle
///
/// A matvec runs at DECODE, where there is one token per request and
/// `experts_per_token` slots each, so the activation stays in token order and
/// the kernel steps through the slots itself. That is what `x_slot_stride`,
/// `x_row_stride` and `slots_per_row` are for, and it is why this form reads
/// `expert_ids` -- the sort's `row_expert`, "which expert does sorted row `p`
/// read" -- where the tiled form reads `tile_expert`.
///
/// # Four weights, and the last one is not the codec's
///
/// `biases` at slot 2 is the AFFINE codec's zero-point plane, one value per
/// group; `bias` at slot 7 is the projection's additive bias, one value per
/// output row. They are different tensors and the statement carries both, which
/// is why this asks `weight(2)` and `weight(3)` rather than reusing one --
/// `model-dsl::metal::routed_qmv` appends `{name}.bias` to the weight list only
/// for the biased arm, and handing the row three weights left the kernel's
/// `bias` naming a tensor the statement does not have.
///
/// # Errors
///
/// [`Refusal::Empty`] for a weight, operand or scalar the statement does not
/// carry.
pub fn qmv_routed_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let in_vec_size = o.param(0)?;
    let out_vec_size = o.param(1)?;
    let x_slot_stride = o.param(2)?;
    let x_row_stride = o.param(3)?;
    let slots_per_row = o.param(4)?;
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let bias = o.weight(3)?;
    let expert_ids = o.input(1)?;
    Ok(vec![
        w,
        scales,
        biases,
        x,
        y,
        ArgValue::I32(in_vec_size),
        ArgValue::I32(out_vec_size),
        bias,
        expert_ids,
        ArgValue::I32(x_slot_stride),
        ArgValue::I32(x_row_stride),
        ArgValue::I32(slots_per_row),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `moe::mxfp4_qmv_routed_bias`: [`qmv_routed_bias`] over gpt-oss's MXFP4
/// expert banks.
///
/// TWO weights and not three -- codes and shared exponents -- so what the affine
/// form calls `biases` is here the projection bias, and the trace says so by
/// putting it at `Weight(2)` where the affine statement puts it at `Weight(3)`.
/// `model-dsl` appends `{name}.bias` for this repr whatever `biased` was asked,
/// because MXFP4 has no unbiased twin.
///
/// # The dropped slot
///
/// The body takes `_biases` and does not forward it: `moe/qmv_routed.wgsl`
/// wraps the affine zero-point plane in `//#if !defined(PIE_MXFP4)`, so the
/// mxfp4 instantiation declares SIX `@group(0)` bindings where the row states
/// seven, and there is no slot to fill. The argument still has to BE a buffer
/// handle -- `Buf::unpack` refuses every other variant -- so this passes the
/// `scales` handle a second time. It mints nothing, it reaches no bind group
/// (the body drops it before `ctx.dispatch`), and `Handles::asked()` stays
/// exactly the six buffers the module declares. A `Handles::unbound()` would
/// say it properly; see the note at the foot of this file.
///
/// # Errors
///
/// See [`qmv_routed_bias`].
pub fn mxfp4_qmv_routed_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let in_vec_size = o.param(0)?;
    let out_vec_size = o.param(1)?;
    let x_slot_stride = o.param(2)?;
    let x_row_stride = o.param(3)?;
    let slots_per_row = o.param(4)?;
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let bias = o.weight(2)?;
    let expert_ids = o.input(1)?;
    Ok(vec![
        w,
        scales,
        // The slot the mxfp4 arm never declares and the body never forwards.
        scales,
        x,
        y,
        ArgValue::I32(in_vec_size),
        ArgValue::I32(out_vec_size),
        bias,
        expert_ids,
        ArgValue::I32(x_slot_stride),
        ArgValue::I32(x_row_stride),
        ArgValue::I32(slots_per_row),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `moe::qmm_t_routed`: the routed affine GEMM over the sorted rectangle.
///
/// `tile_expert` tells each workgroup which expert's weights to read, which is
/// what makes ONE dispatch serve every expert. It is read at the tile's `y`, so
/// the tiling is not a knob the launch may round: `tile_m` has to be the
/// `tile_rows` the sort was given, and it comes off the symbol.
///
/// # `pad` is `Input(1)` and this backend does not bind it
///
/// The statement carries three inputs -- `x`, `pad`, `tile_expert` -- where
/// `pad` is the sort's padded row count read on the DEVICE, because the host
/// does not know it: it depends on the routing. Metal binds `pad` five times
/// over, to fill the argument-table holes its entrypoint leaves between slot 6
/// and `tile_expert` at 12. `moe/qmm_t_routed.wgsl` declares six dense bindings
/// and takes its extent through the grid, so nothing binds `pad` here.
///
/// The index is skipped rather than renumbered: `tile_expert` is `Input(2)`
/// because that is where the TRACE puts it, and asking `input(1)` would hand
/// the GEMM a device-side row count where it reads which expert owns a tile.
///
/// # Errors
///
/// [`Refusal::Empty`] for a weight, operand or scalar the statement does not
/// carry. A symbol naming no affine point parses as zeros and the BODY refuses
/// it [`Refusal::Narrow`], which is where that check belongs -- the table of
/// fifty-four spellings is the body's.
pub fn qmm_t_routed(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let tile_expert = o.input(2)?;
    Ok(vec![
        w,
        scales,
        biases,
        x,
        y,
        tile_expert,
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(f.rows.cast_signed()),
        ArgValue::I32(f.group),
        ArgValue::I32(f.bits),
        ArgValue::I32(f.tile_m),
        ArgValue::I32(f.tile_n),
    ])
}

/// `moe::qmm_t_routed_fp16`: [`qmm_t_routed`] with the activation precast to
/// fp16.
///
/// Group 64 at four bits, compiled in -- `AffineQ::group_size` is a constant
/// there, so a second point would name an instantiation that dequantises at 64
/// whatever it claims -- so the body takes neither number and this arm states
/// only the tile.
///
/// # Errors
///
/// See [`qmm_t_routed`].
pub fn qmm_t_routed_fp16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let tile_expert = o.input(2)?;
    Ok(vec![
        w,
        scales,
        biases,
        x,
        y,
        tile_expert,
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(f.rows.cast_signed()),
        ArgValue::I32(f.tile_m),
        ArgValue::I32(f.tile_n),
    ])
}

/// `moe::mxfp4_qmm_t_routed_bias`: the routed GEMM over MXFP4 expert banks.
///
/// The binding order is NOT the affine one with a hole in it, which is this
/// family's difference from its own matvec: `qmm_t_routed.wgsl`'s `PIE_MXFP4`
/// arm RENUMBERS -- `exponents` at 1, `x` at 2, `y` at 3, `bias` at 4,
/// `tile_expert` at 5 -- six dense bindings against affine's six, with the
/// output bias occupying the slot affine spends on a zero-point plane. So the
/// body takes six buffers and not seven, and the affine order here would bind
/// the bias where the GEMM reads its activations.
///
/// # Errors
///
/// See [`qmm_t_routed`]; the tile is the only point this one has and a symbol
/// naming none is [`Refusal::Narrow`] from the body.
pub fn mxfp4_qmm_t_routed_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let w = o.weight(0)?;
    let exponents = o.weight(1)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let bias = o.weight(2)?;
    let tile_expert = o.input(2)?;
    Ok(vec![
        w,
        exponents,
        x,
        y,
        bias,
        tile_expert,
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(f.rows.cast_signed()),
        ArgValue::I32(f.tile_m),
        ArgValue::I32(f.tile_n),
    ])
}

// ---- attn ----
//
// The family where the KV cache stops being an operand, and the family this
// backend can arm the least of. Every SDPA form and both KV appends read
// `k_pages`/`v_pages` out of the driver's own pool -- no traced value stands
// for them, so no `Arg` names them -- and `Handles` on this backend can ask
// for an operand, the packed scalar run, or one of `FireTable`'s eleven
// buffers, and for nothing else. `FireTable` has no KV cache entry (the table
// path reaches it through `kernels::Source::KvKeys`, which `binding::reorder`
// answers from `Resolve::kv(layer, values)`), `Asked` has no variant that
// could carry one, and `Facts` carries no `layer` to ask with. Three separate
// absences, each of which alone would be enough.
//
// So THREE of the sixteen are armed here -- the packed-QKV split, the
// query/gate split and the softcap -- and they are exactly the three that
// touch neither the pool nor a `FireNumber`. The other thirteen are listed in
// the registry with `arm: None` so that the longest-match lookup still sends
// their symbols to the table path, and the prose after the registry says for
// each one exactly what was missing.

/// `attn::split_qkv_bf16`: one packed projection row cut into three.
///
/// The one statement in this family that writes THREE results, and the reason
/// the row's `Out` indices were called load-bearing: `q`, `k` and `v` are
/// results 0, 1 and 2 in the order the lowering states them, and two swapped
/// here is a model that runs at full speed and attends over the wrong planes.
/// Nothing downstream can see it -- all three are storage buffers of the same
/// element type, and only their extents differ.
///
/// # The block is STORAGE, and the only extent beside it is the INPUT's
///
/// `split_qkv.wgsl` declares `@group(0) @binding(4) var<storage, read_write>
/// params: SplitQkvParams` and no `@group(1)` at all, so this is the storage
/// kind of parameter block -- it takes a place in the buffer numbering, and
/// its bytes are the statement's own scalar run. `mlp`'s three gated
/// activations are the same shape and [`gated`] says so at length.
///
/// The extent is `in_width` and NOT `width`. A statement that reads one buffer
/// and writes three has no single output that spells its grid, so the width
/// that sizes it is the first widthed operand -- which is what
/// `dispatch::dims` means by *"the FIRST widthed operand, which is the first
/// input"* and what `geometry`'s `Rule::SplitPacked` reads as
/// `[dims.in_width, rows, 1]`. The shader recomputes `q + 2 * kv` from its own
/// copy of the widths and guards on it, so a `packed_width` that disagreed
/// would leave a tail of every row uncopied rather than run off the end --
/// silent again, which is why this reads the same field the table path read.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn split_qkv_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let packed = o.input(0)?;
    let q = o.output(0)?;
    let k = o.output(1)?;
    let v = o.output(2)?;
    let params = o.params_block();
    Ok(vec![
        packed,
        q,
        k,
        v,
        params,
        ArgValue::I32(f.in_width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `attn::q_gate_split`: qwen3.5's `[query|gate]` block cut in two.
///
/// [`split_qkv_bf16`]'s problem with two results instead of three, and with
/// stated scalars instead of a block: `gate.wgsl` declares `struct Params {
/// head_dim, qg_row_stride, out_row_stride }` at `@group(1) @binding(0)`,
/// which is the UNIFORM kind, so the three numbers are passed as scalars and
/// the driver packs them. They are words 0, 1 and 2 of the statement's run in
/// that order, which is what the shader's struct says and what
/// `driver-metal`'s and `driver-vulkan`'s arms for the same kernel read.
///
/// # This is the family's `argmax_logits`
///
/// The row states no `launch` rule -- `kernel!(q_gate_split "q_gate_split",
/// axes = &[BF16])`, nothing else -- so `geometry::lanes` has always refused
/// it `Unstated` and the table path has never dispatched it. Arming it
/// therefore changes no fire that runs today and is the cheapest place in this
/// family a seam can be proven, which is the reason `sample::argmax_logits`
/// went first on this backend. The operand reading is not a guess either: one
/// input, two results, no `in_place` pair, so no index here is ambiguous.
///
/// # `head_dim` falls back and the two pitches do not
///
/// The head width reaches the SHADER and the GRID both -- the body builds
/// `head_grid(head_dim, q_heads, rows)` out of the same argument -- so one
/// number has to serve both, and a statement that states none should get the
/// fire's rather than dispatch a kernel told `head_dim = 0` over a grid sized
/// by the fire. That is [`Handles::stated`]'s whole reason and
/// `driver-metal`'s rope arms record the defect it prevents.
///
/// A row pitch is the other kind of fact: it says how the trace laid this
/// tensor out and nothing else knows it, so an absent one is a disagreement
/// between this arm and the trace and is refused. The shader does fall back --
/// `if (params.qg_row_stride > 0)`, else `row * n_q * hd * 2u` -- but the
/// fallback is the packed layout, and a windowed one that reached it would
/// read every row after the first from the wrong offset.
///
/// # `q_heads` is grid-only and load-bearing
///
/// The shader reads `num_workgroups.y` back as the query-head count to
/// compute its default pitches, so the y extent is not merely how much work
/// gets launched: an extent short by a head tells the kernel the model has one
/// fewer.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry, or for
/// either pitch it does not state.
pub fn q_gate_split(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let head_dim = o.stated(0, f.head_dim);
    let qg_row_stride = o.param(1)?;
    let out_row_stride = o.param(2)?;
    let qg = o.input(0)?;
    let q_out = o.output(0)?;
    let gate_out = o.output(1)?;
    Ok(vec![
        qg,
        q_out,
        gate_out,
        ArgValue::I32(head_dim),
        ArgValue::I32(qg_row_stride),
        ArgValue::I32(out_row_stride),
        ArgValue::I32(f.q_heads.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `attn::logit_softcap`: gemma's `cap * tanh(x / cap)` on the readout.
///
/// The cap rides in the block with a trailing word nothing reads --
/// `SoftcapParams` is `cap` then `n` -- and `logit_softcap.wgsl` declares it
/// `@group(0) @binding(2) var<storage, read_write>`, so this is the storage
/// kind of block again and the statement's own run is its bytes.
///
/// The only extent the body states is the ELEMENT COUNT, which is
/// `width * rows` and not the width: the row's rule is
/// `LaunchRule::Elementwise` and `geometry`'s reading of it is
/// `[dims.width * rows, 1, 1]`, one flat axis over the whole rectangle. The
/// multiply saturates rather than wrapping, for `binding::extent`'s reason: an
/// absurd plan should reach the body's own `elementwise` and be refused there
/// -- `u32::MAX` casts to `-1`, which is `Refusal::Empty` -- rather than
/// become a small number that launches over part of the readout and reports
/// success.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn logit_softcap(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let logits = o.input(0)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        logits,
        out,
        params,
        ArgValue::I32(f.width.saturating_mul(f.rows).cast_signed()),
    ])
}

// ---- quant ----
//
// PATH NOTE. The brief asked for `/tmp/agent-quant-arms.rs`. This runtime
// refuses every write under `/tmp`, so the file is written into the working
// directory instead. It is not in any crate's `src/`, so nothing compiles it
// and no repo file was modified. Nothing in this session ran cargo.
//
// Written against `crates/kernels-wgpu/src/quant.rs` — the signature AND the
// `ctx.dispatch` list of every one of its thirty-one `pub fn`s — and against
// `crates/kernels-wgpu/kernels/quant/{qmm_t,qmv,transcode}.wgsl`, which is
// what settles the divergence question at the end. `driver-metal`'s arms are
// the operand-provenance authority (which traced value is `x`, which is the
// residual, where the bias sits in the weight list) and NOT the argument-list
// authority: metal's signatures carry a `pad` this backend does not have.
//
// Thirty-one routines, twenty-eight armed. The shape is the same throughout:
// three weights (the codes and the two affine planes), then the activation,
// then the result, then `k` and `n` — the two scalars every statement of this
// family carries. What varies is what rides after: a bias, a residual, a row
// pitch, a split, or a precast activation that replaces `x` entirely.
//
// THE AFFINE POINT IS IN THE NAME ON THIS BACKEND. Metal takes `group` and
// `bits` as launch facts off its `Geometry`; here a body indexes a static
// spelling table — `QMM_T[qmm_point(*group, *bits, *bm, *bn)?]` — and the
// numbers come from `Facts::{group, bits, tile_m, tile_n}`, which
// `arm::suffix` parses off the SYMBOL's `_gs_64_b_4_bm_32_bn_32` tail. So
// there is no `tile()` helper here answering metal's `Refusal::Unstated`: a
// symbol that names no tile parses as zero, and zero is not a point on
// `TILES`, so `point()` refuses it `Refusal::Narrow { what: "the row tile" }`
// inside the body. A wrong tile cannot be reached by falling back, which is
// the property metal's `tile()` had to write code to get.

// No `use` block: these land in `crates/driver-wgpu/src/lowering/arm.rs`
// beside the `norm` and `layout` arms, which already imports
// `kernels::routine::Refusal` and `kernels_wgpu::routine::ArgValue` and
// declares `Facts`, `Handles` and `Arm`. If the family goes into a submodule
// instead it needs those two `use`s plus `use super::{Arm, Facts, Handles};`.

/// The quantized weight triple every routine in this family opens with.
///
/// `w`, `scales`, `biases` — the packed codes and the two affine terms that
/// decode them. Weights 0, 1 and 2 in every row of this family that states
/// any, and a `_bias` form's bias is weight 3 AFTER them, which is the order
/// the lowering lists a statement's weights in.
///
/// # Errors
///
/// [`Refusal::Empty`] when the statement carries fewer than three weights.
fn codec(o: &mut Handles<'_>) -> Result<[ArgValue; 3], Refusal> {
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    Ok([w, scales, biases])
}

/// The four numbers a split-K multiply is told about its own partition.
///
/// The row pitch, how much of `k` one partition covers, how far apart the
/// partials are, and how many there are — words 2, 3, 4 and 5. FOUR, where
/// `driver-metal::lowering::arm::split_k` returns three: metal's signature
/// takes no `row_stride` for the split-K GEMMs and this backend's does, which
/// is one of the seventeen disagreements the report below sets out. Reading
/// metal's indices here would hand `k_partition_size` to `row_stride` and
/// walk off the end of the partials.
///
/// None of the four is a shape the fire knows. The driver that chose to split
/// states all four and a statement that does not carry them is refused.
///
/// # Errors
///
/// [`Refusal::Empty`] for any of the four the statement does not state.
pub fn split_k(o: &Handles<'_>) -> Result<[ArgValue; 4], Refusal> {
    Ok([
        ArgValue::I32(o.param(2)?),
        ArgValue::I32(o.param(3)?),
        ArgValue::I32(o.param(4)?),
        ArgValue::I32(o.param(5)?),
    ])
}

/// The precast GEMM's opening: the codec, the result, and the half-precision
/// activation.
///
/// There is no `x`. The activation was cast to `float16` by
/// [`cast_qmm_input_bfloat16_to_float16`] into a buffer of its own and THAT is
/// the statement's first input, which is why `y` comes before it here and
/// after it everywhere else. `quant/qmm_t.wgsl` numbers a precast variant
/// `w`=0, `scales`=1, `biases`=2, `out_`=3, `half_in`=4 — dense, because it
/// declares only what it reads — and this list is that numbering.
///
/// AND NO PAD. `driver-metal`'s `precast` pushes `c[0]` a second time to fill
/// buffer 3, a slot every `_fp16_precast` MSL entrypoint leaves undeclared
/// because it keeps `affine_qmm_t`'s numbering where 3 is `x`. WGSL has no
/// such hole: see the report at the end of this file.
///
/// # Errors
///
/// [`Refusal::Empty`] for a weight, the result, or the staged activation.
pub fn precast(o: &mut Handles<'_>) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let half_in = o.input(0)?;
    let y = o.output(0)?;
    let mut v = c.to_vec();
    v.extend([y, half_in]);
    Ok(v)
}

/// The five GEMMs whose codec and tile are compiled into the symbol.
///
/// `qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2` and its four siblings are
/// one point each, so their bodies name a literal entrypoint and take neither
/// `group` nor `bits` nor a tile — only `m`. The arm is [`qmm_t`]'s without
/// the four axis facts.
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_fixed(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (k, n) = kn(o)?;
    let mut v = c.to_vec();
    v.extend([
        x,
        y,
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `quant::qmm_t`: the tiled GEMM against affine-quantized weights.
///
/// The one row of this family with a full `operands` column, so it is also
/// the one arm whose order can be checked against something other than a
/// signature: `w, scales, biases, x, y, k, n` is exactly what
/// `kernel!(qmm_t "affine_qmm_t", operands = ...)` states.
///
/// The four axis facts after `n` are not operands and not launch facts — they
/// pick the SPELLING, `QMM_T[qmm_point(group, bits, bm, bn)?]`, out of
/// fifty-four. A pair that is off any axis is refused by the body rather than
/// rounded, because g64/b8 and g128/b4 pack to identical shapes and a module
/// chosen for the wrong pair unpacks fluent nonsense.
///
/// # Errors
///
/// [`Refusal::Empty`] for a weight, operand or scalar the statement does not
/// carry. The body then refuses [`Refusal::Narrow`] for a symbol whose suffix
/// names a codec point or tile the shader tree does not carry, which is what
/// a missing `_bm_`/`_bn_` parses to.
pub fn qmm_t(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (k, n) = kn(o)?;
    let mut v = c.to_vec();
    v.extend([
        x,
        y,
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(f.group),
        ArgValue::I32(f.bits),
        ArgValue::I32(f.tile_m),
        ArgValue::I32(f.tile_n),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `quant::qmm_t_bias`: [`qmm_t`] with the projection's bias added.
///
/// The bias is WEIGHT 3 — after the codec's three, which is the order a
/// statement lists its weights in — and it lands at index 5 of the argument
/// list, between `y` and `k`, because `quant/qmm_t.wgsl` binds `extra` at 5
/// for a non-precast variant. Metal inserts at 5 as well and for a different
/// reason: its list has a pad at 3, so its `y` is at 4.
///
/// # Errors
///
/// See [`qmm_t`], plus [`Refusal::Empty`] when the statement carries no
/// fourth weight.
pub fn qmm_t_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let bias = o.weight(3)?;
    let mut v = qmm_t(o, f)?;
    v.insert(5, bias);
    Ok(v)
}

/// `quant::qmm_t_residual`: [`qmm_t`] with the block residual folded in.
///
/// The residual lands AFTER `k` and `n` rather than beside the activation.
/// That is this body's own order — `w, scales, biases, x, y, k, n, residual`
/// in its dispatch list — and it is the fold's convention throughout the
/// tree: a conditional binding comes last so that folding does not renumber
/// what every form shares. The shader still binds it at 5, because the
/// scalars are not in the buffer numbering here; `bind` splits them out.
///
/// # Errors
///
/// See [`qmm_t`], plus [`Refusal::Empty`] for a statement with one input.
pub fn qmm_t_residual(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let mut v = qmm_t(o, f)?;
    v.insert(7, residual);
    Ok(v)
}

/// `quant::qmm_t_fp16_precast`: [`qmm_t`] over an activation already cast to
/// `float16`.
///
/// Group 64 at four bits only — the precast family is stamped at one codec
/// point, which is why its axes state `GROUP_64, BITS_4` and its signature
/// takes neither. The tile is still a choice, so `bm` and `bn` still come off
/// the symbol.
///
/// # Errors
///
/// See [`precast`]; the body then refuses [`Refusal::Narrow`] for a tile the
/// tree does not carry.
pub fn qmm_t_fp16_precast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let mut v = precast(o)?;
    v.extend([
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(f.tile_m),
        ArgValue::I32(f.tile_n),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `quant::qmm_t_bias_fp16_precast`: [`qmm_t_fp16_precast`] with a bias.
///
/// Index FOUR, between `y` and `half_in`, which is where the shader declares
/// `extra` for a precast variant. `driver-metal` inserts the same operand at
/// 5 and both are right for their own backend: metal's list has a pad at 3.
/// Copying metal's index here would bind the bias where the body passes the
/// staged activation.
///
/// # Errors
///
/// See [`qmm_t_fp16_precast`], plus [`Refusal::Empty`] for a missing fourth
/// weight.
pub fn qmm_t_bias_fp16_precast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let bias = o.weight(3)?;
    let mut v = qmm_t_fp16_precast(o, f)?;
    v.insert(4, bias);
    Ok(v)
}

/// `quant::qmm_t_residual_fp16_precast`: [`qmm_t_fp16_precast`] with the block
/// residual folded in.
///
/// Index four, for [`qmm_t_bias_fp16_precast`]'s reason: `extra` is one
/// binding and no variant has both a bias and a residual.
///
/// # Errors
///
/// See [`qmm_t_fp16_precast`], plus [`Refusal::Empty`] for a statement with
/// one input.
pub fn qmm_t_residual_fp16_precast(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let mut v = qmm_t_fp16_precast(o, f)?;
    v.insert(4, residual);
    Ok(v)
}

/// `quant::qmm_t_splitk`: the GEMM with the contraction cut into partitions.
///
/// The output is the `[split_k, M, N]` PARTIAL block and not the projection —
/// [`qmm_splitk_reduce`] is the second half — but it is still the statement's
/// first result, so it is `output(0)` here as everywhere else.
///
/// Stamped at `_bn_32` alone, so the column tile is not a choice: the body
/// reads `WIDE_BN` for its grid and the argument list carries only `bm`.
///
/// # Errors
///
/// See [`qmm_t`], plus [`Refusal::Empty`] for any of [`split_k`]'s four.
pub fn qmm_t_splitk(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let x = o.input(0)?;
    let out = o.output(0)?;
    let (k, n) = kn(o)?;
    let split = split_k(o)?;
    let mut v = c.to_vec();
    v.extend([x, out, ArgValue::I32(k), ArgValue::I32(n)]);
    v.extend(split);
    v.extend([
        ArgValue::I32(f.group),
        ArgValue::I32(f.bits),
        ArgValue::I32(f.tile_m),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `quant::qmm_t_splitk_f32`: [`qmm_t_splitk`] accumulating into `float32`.
///
/// The partials are `f32` rather than `bf16`, which is a binding TYPE and not
/// an argument list, so the arm is the same one.
///
/// # Errors
///
/// See [`qmm_t_splitk`].
pub fn qmm_t_splitk_f32(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    qmm_t_splitk(o, f)
}

/// `quant::qmm_t_splitk_fp16_precast`: [`qmm_t_splitk`] over a precast
/// activation.
///
/// # Errors
///
/// See [`qmm_t_splitk`] and [`precast`].
pub fn qmm_t_splitk_fp16_precast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let split = split_k(o)?;
    let mut v = precast(o)?;
    v.extend([ArgValue::I32(k), ArgValue::I32(n)]);
    v.extend(split);
    v.extend([ArgValue::I32(f.tile_m), ArgValue::I32(f.rows.cast_signed())]);
    Ok(v)
}

/// `quant::qmm_t_splitk_fp16_precast_f32`: [`qmm_t_splitk_fp16_precast`]
/// accumulating into `float32`.
///
/// # Errors
///
/// See [`qmm_t_splitk`].
pub fn qmm_t_splitk_fp16_precast_f32(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    qmm_t_splitk_fp16_precast(o, f)
}

/// `quant::qmm_t_strided`: the GEMM over an activation whose rows do not tile.
///
/// A packed projection is the case, as with every `_strided` form in this
/// tree: the pitch spans more than the row the multiply reads. It is the
/// statement's THIRD scalar or a refusal — a pitch is a fact about how the
/// trace laid this tensor out and no fire-wide number is one, which is
/// `rope::neox_strided`'s reason and `Handles::param`'s.
///
/// # Errors
///
/// See [`qmm_t`], plus [`Refusal::Empty`] for the pitch.
pub fn qmm_t_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_stride = o.param(2)?;
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (k, n) = kn(o)?;
    let mut v = c.to_vec();
    v.extend([
        x,
        y,
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(row_stride),
        ArgValue::I32(f.group),
        ArgValue::I32(f.bits),
        ArgValue::I32(f.tile_m),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `quant::qmm_t_strided_residual`: [`qmm_t_strided`] with the residual folded
/// in.
///
/// The residual is at index FIVE — right after `y`, before `k` — which is
/// this body's own dispatch order and the shader's `extra` at binding 5. It
/// does not delegate to [`qmm_t_strided`] and insert, because the two lists
/// differ in where the fold sits relative to the scalars and an insert index
/// is exactly the kind of thing that survives a rename.
///
/// This is a place worth being explicit about: `driver-metal`'s
/// `qmm_t_strided_residual` binds the residual at 7 and its own comment
/// records that it said 5 "for as long as it has existed", which bound the
/// residual pointer over `K`. Metal's 7 is right on metal — its argument
/// table numbers scalars in the same run as buffers — and 5 is right here,
/// where scalars are a uniform block and buffers are numbered densely.
///
/// # Errors
///
/// See [`qmm_t_strided`], plus [`Refusal::Empty`] for a statement with one
/// input.
pub fn qmm_t_strided_residual(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_stride = o.param(2)?;
    let c = codec(o)?;
    let x = o.input(0)?;
    let residual = o.input(1)?;
    let y = o.output(0)?;
    let (k, n) = kn(o)?;
    let mut v = c.to_vec();
    v.extend([
        x,
        y,
        residual,
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(row_stride),
        ArgValue::I32(f.group),
        ArgValue::I32(f.bits),
        ArgValue::I32(f.tile_m),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `quant::qmm_t_strided_fp16_precast`: [`qmm_t_strided`] over a precast
/// activation.
///
/// # Errors
///
/// See [`qmm_t_strided`] and [`precast`].
pub fn qmm_t_strided_fp16_precast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_stride = o.param(2)?;
    let (k, n) = kn(o)?;
    let mut v = precast(o)?;
    v.extend([
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(row_stride),
        ArgValue::I32(f.tile_m),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `quant::qmm_t_strided_fp16_precast_residual`:
/// [`qmm_t_strided_fp16_precast`] with the residual folded in.
///
/// Index four, where every precast variant declares `extra`.
///
/// # Errors
///
/// See [`qmm_t_strided_fp16_precast`], plus [`Refusal::Empty`] for a
/// statement with one input.
pub fn qmm_t_strided_fp16_precast_residual(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let mut v = qmm_t_strided_fp16_precast(o, f)?;
    v.insert(4, residual);
    Ok(v)
}

/// `quant::qmm_splitk_reduce`: the sum of a split multiply's partials.
///
/// The second half of a split-K pair and the only routine in the family that
/// reads no weights: the partials are the statement's input and the
/// projection is its result. The RESULT IS FIRST in the argument list —
/// `reduce_y` is binding 0 and `partial` is 1 — which is the reverse of every
/// GEMM above and the reason this is written out rather than shaped from one.
///
/// Five scalars: `k`, `n`, the row pitch, the partition stride and the
/// partition count, words 0..=4 of the run. See the report at the end of this
/// file — this backend's WGSL declares three of the five, which is a
/// `kernels-wgpu` question and not an arm's.
///
/// # Errors
///
/// [`Refusal::Empty`] for the partials, the result, or any of the five
/// scalars.
pub fn qmm_splitk_reduce(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let partial = o.input(0)?;
    let y = o.output(0)?;
    let (k, n) = kn(o)?;
    Ok(vec![
        y,
        partial,
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(o.param(2)?),
        ArgValue::I32(o.param(3)?),
        ArgValue::I32(o.param(4)?),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `quant::qmm_splitk_reduce_f32`: [`qmm_splitk_reduce`] over `float32`
/// partials.
///
/// # Errors
///
/// See [`qmm_splitk_reduce`].
pub fn qmm_splitk_reduce_f32(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    qmm_splitk_reduce(o, f)
}

/// `quant::cast_qmm_input_bfloat16_to_float16`: the cast the precast GEMMs
/// read.
///
/// `count` is how many elements to cast and it is the STATEMENT's rather than
/// the fire's: a cast covers exactly the activation the multiply after it
/// will read, which need not be the whole rectangle. It is word 3, after the
/// `k`, `n`, `row_stride` prefix every statement of this family states —
/// which is why this arm reads four scalars for a kernel whose shader uses
/// one.
///
/// No weights and no codec: this is a format change, not a multiply.
///
/// # Errors
///
/// [`Refusal::Empty`] for the tensor, the result, or any of the four scalars.
pub fn cast_qmm_input_bfloat16_to_float16(
    o: &mut Handles<'_>,
    _f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let cast_in = o.input(0)?;
    let half_out = o.output(0)?;
    let (k, n) = kn(o)?;
    Ok(vec![
        cast_in,
        half_out,
        ArgValue::I32(k),
        ArgValue::I32(n),
        ArgValue::I32(o.param(2)?),
        ArgValue::I32(o.param(3)?),
    ])
}

/// `quant::cast_qmm_input_strided_bfloat16_to_float16`:
/// [`cast_qmm_input_bfloat16_to_float16`] over rows a pitch apart.
///
/// One argument longer, and the extra one is a FACT rather than a scalar: the
/// row count sizes the grid (`elementwise_rows(k, rows)`) where the packed
/// form counts elements. Pushing onto the packed arm is right here because
/// the two lists really do share a prefix on this backend — they do not on
/// metal, whose shared argument table gives the strided form `k` at 5 and
/// `row_stride` at 8.
///
/// # Errors
///
/// See [`cast_qmm_input_bfloat16_to_float16`].
pub fn cast_qmm_input_strided_bfloat16_to_float16(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let mut v = cast_qmm_input_bfloat16_to_float16(o, f)?;
    v.push(ArgValue::I32(f.rows.cast_signed()));
    Ok(v)
}

/// `quant::qmv_fast`: the matvec, `vecs` rows at a time.
///
/// `in_vec_size` and `out_vec_size` are the statement's first two scalars —
/// the same words `k` and `n` ride for a GEMM — and `vecs` is the row count,
/// which is what made the multi-row form a generalisation of the single-row
/// one rather than a different kernel.
///
/// # Errors
///
/// [`Refusal::Empty`] for a weight, operand or scalar the statement does not
/// carry; the body then refuses [`Refusal::Narrow`] for a codec point the
/// tree does not carry.
pub fn qmv_fast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (in_vec, out_vec) = kn(o)?;
    let mut v = c.to_vec();
    v.extend([
        x,
        y,
        ArgValue::I32(in_vec),
        ArgValue::I32(out_vec),
        ArgValue::I32(f.group),
        ArgValue::I32(f.bits),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `quant::qmv_fast_residual`: [`qmv_fast`] with the residual folded in.
///
/// Index seven — after the two extents, which is where the body passes it and
/// where `qmv.wgsl` binds `extra` at 5 once the scalars are split out.
///
/// # Errors
///
/// See [`qmv_fast`], plus [`Refusal::Empty`] for a statement with one input.
pub fn qmv_fast_residual(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let mut v = qmv_fast(o, f)?;
    v.insert(7, residual);
    Ok(v)
}

/// `quant::qmv_tail`: the matvec for an output width the fast form's
/// decomposition does not divide.
///
/// Stamped at group 64 only, so the signature takes `bits` and not `group`
/// and the body indexes `QMV_TAIL[bits_point(bits)?]` — two spellings, not
/// six. `f.group` is still parsed off the symbol and still ignored, which is
/// the difference between "the deployment has no group size" and "this kernel
/// does not choose on it".
///
/// # Errors
///
/// See [`qmv_fast`].
pub fn qmv_tail(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (in_vec, out_vec) = kn(o)?;
    let mut v = c.to_vec();
    v.extend([
        x,
        y,
        ArgValue::I32(in_vec),
        ArgValue::I32(out_vec),
        ArgValue::I32(f.bits),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `quant::qmv_tail_bias`: [`qmv_tail`] with the projection's bias.
///
/// Index five, between `y` and the two extents — `qmv.wgsl` binds `extra` at
/// 5 and the bias is weight 3. Metal inserts at 6 because of its pad.
///
/// # Errors
///
/// See [`qmv_fast`], plus [`Refusal::Empty`] for a missing fourth weight.
pub fn qmv_tail_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let bias = o.weight(3)?;
    let mut v = qmv_tail(o, f)?;
    v.insert(5, bias);
    Ok(v)
}

/// `quant::qmv_wide_strided`: the matvec over a wide contraction with rows a
/// pitch apart.
///
/// `m` is an ARGUMENT here rather than an `Env`, and it is the only routine in
/// the family where the row count is both: the grid covers QUARTERS of it
/// (`qmv_grid(quarters(m), out_vec_size)`), so the threads of a partial last
/// quarter need the number to know they are past the end. `qmv.wgsl`'s
/// `PIE_WIDE_STRIDED` block is `in_vec_size, out_vec_size, row_stride, m` —
/// four words, and this list passes exactly those four.
///
/// `bits` comes LAST, after `m`, which is the one place in this family where
/// an axis fact is not adjacent to its siblings.
///
/// # Errors
///
/// See [`qmv_fast`], plus [`Refusal::Empty`] for the pitch.
pub fn qmv_wide_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_stride = o.param(2)?;
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (in_vec, out_vec) = kn(o)?;
    let mut v = c.to_vec();
    v.extend([
        x,
        y,
        ArgValue::I32(in_vec),
        ArgValue::I32(out_vec),
        ArgValue::I32(row_stride),
        ArgValue::I32(f.rows.cast_signed()),
        ArgValue::I32(f.bits),
    ]);
    Ok(v)
}

/// `quant::qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4`. See [`qmm_fixed`].
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    qmm_fixed(o, f)
}

/// `quant::qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2`. See
/// [`qmm_fixed`].
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    qmm_fixed(o, f)
}

/// `quant::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2`. See
/// [`qmm_fixed`].
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    qmm_fixed(o, f)
}

/// `quant::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1`. See
/// [`qmm_fixed`].
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    qmm_fixed(o, f)
}

/// `quant::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4`. See [`qmm_fixed`].
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    qmm_fixed(o, f)
}

// ===========================================================================

/// `attn::gate`: the attention gate, `attn *= sigmoid(gate)`, in place.
///
/// Two buffers and three scalars, in the body's order: the tensor being
/// gated (which is also the result), the gate, the row pitch, and then the
/// rectangle. `width` and `rows` are `Env<i32>`, so they are spent on the
/// grid rather than bound, and the shader's one `@group(1)` field is the
/// pitch alone -- four bytes, which is what `attn/gate.wgsl`'s `Params`
/// declares and what `Declared::uniform_bytes` will ask for.
///
/// # The tensor is `output(0)` and the gate is `input(1)`
///
/// `routine!(gate, in_place = &[(0, 0)])` says output 0 and input 0 name the
/// same address. In the trace that address is the attention result: the
/// statement is `SigmoidGateMul` with `inputs = [x, gate]` and one result,
/// `lower::walk` pushes every input and then every output, and in-place
/// changes where the OUTPUT is written, never the argument list. So the
/// launch carries three widthed args, `[x, gate, out]`.
///
/// [`Handles::over`] splits that with `results = 1` -- the body has one
/// writable parameter, `attn: BufMut` -- so the LAST widthed arg is the
/// output and the first two are inputs: `ins = [x, gate]`, `outs = [out]`.
/// The gate is therefore `input(1)`, and `input(0)` is the tensor `output(0)`
/// already aliases. Asking for `input(0)` here would bind one address at both
/// `@binding(0)` and `@binding(1)` and compute `attn *= sigmoid(attn)` --
/// arithmetic that runs, reports success, and never reads the gate, with no
/// arity or shape check able to see it because the two operands share a shape
/// by construction.
///
/// [`add_bias`] is the precedent rather than the analogy: same
/// `in_place = &[(0, 0)]`, and it takes the tensor from `o.output(0)` and
/// never asks `input(0)`, because for an aliased pair those are the same
/// bytes and the output is the one the body writes through. `gate` is that
/// shape with one extra input that is NOT aliased, and that extra input is
/// the gate.
///
/// Metal (`driver-metal/src/lowering/arm.rs:2485`) and Vulkan
/// (`driver-vulkan/src/arm.rs:2549`) both read `o.input(0)` here. The operand
/// order is `model-ir`'s and not a backend's, so that is a defect in those
/// two arms and not a difference between the backends; it has never fired
/// because both of their rows are bare too, which both of their doc comments
/// say in as many words.
///
/// # The pitch is stated or nothing
///
/// `o.param(0)` with no fire-wide fallback, following both siblings and this
/// file's own rule for `residual_add_strided` and
/// `shared_expert_combine_strided`: a row pitch is a fact about how the trace
/// laid this tensor out and nothing else in the fire knows it. `f.width`
/// would be a guess wherever the gated rectangle is a window into a packed
/// block, and a zero hands the shader its own fallback of one workgroup's
/// span per row, which is the true pitch only when the row is already a whole
/// number of workgroups. A statement that carries no run gets a refusal,
/// which is the same darkness the bare row produces today and not a
/// regression from it.
///
/// # Errors
///
/// [`Refusal::Empty`] when the statement states no scalar at 0, or when it
/// carries no second input or no output.
pub fn gate(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_stride = o.param(0)?;
    let attn = o.output(0)?;
    let gate_in = o.input(1)?;
    Ok(vec![
        attn,
        gate_in,
        ArgValue::I32(row_stride),
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `moe::router_topk`: the unscaled router.
///
/// The fifth binding is `per_expert_scale`, which this variant DECLARES and
/// does not read — `router_topk_scaled` is the symbol that means it. The row
/// listed it with no `Source` and `reorder` answered `Slot::Nothing`;
/// [`Handles::unbound`] is that answer here. The body forwards a handle for it
/// because the module's numbering is positional, and `bind` puts no buffer
/// there.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn router_topk(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let logits = o.input(0)?;
    let expert_ids = o.output(0)?;
    let expert_weights = o.output(1)?;
    let params = o.params_block();
    let scale = o.unbound();
    Ok(vec![
        logits,
        expert_ids,
        expert_weights,
        params,
        scale,
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// `moe::qmv_routed`: the unbiased routed matvec.
///
/// `bias` is the binding this variant declares and does not read —
/// `affine_qmv_routed_bias` is the symbol that means it — so it is
/// [`Handles::unbound`] for the reason [`router_topk`] gives.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand or scalar the statement does not carry.
pub fn qmv_routed(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let in_vec_size = o.param(0)?;
    let out_vec_size = o.param(1)?;
    let x_slot_stride = o.param(2)?;
    let x_row_stride = o.param(3)?;
    let slots_per_row = o.param(4)?;
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let bias = o.unbound();
    let expert_ids = o.input(1)?;
    Ok(vec![
        w,
        scales,
        biases,
        x,
        y,
        ArgValue::I32(in_vec_size),
        ArgValue::I32(out_vec_size),
        bias,
        expert_ids,
        ArgValue::I32(x_slot_stride),
        ArgValue::I32(x_row_stride),
        ArgValue::I32(slots_per_row),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

// ---------------------------------------------------------------------------
// attn
// ---------------------------------------------------------------------------

/// The `n`th value the STATEMENT reads, wherever the split put it.
///
/// **For the two KV appends and for nothing else.** `Handles::over` splits a
/// launch's widthed operands by the ROUTINE's writable-argument count --
/// `routine::results` counts `Ty::BufMut` and its siblings -- and both append
/// bodies declare the CACHE planes `BufMut`. That is two writable arguments
/// against a statement that produces no value at all: `TraceBuilder::kv_append`
/// is `push(OpKind::KvAppend { layer }, vec![k, v], vec![])`, and `model-ir`'s
/// own `OpInfo::outputs` doc says it in as many words -- *"SplitQkv produces
/// three, KvAppend none"*. So `widthed = [k_new, v_new]`, `results = 2`, the
/// split takes the last two as OUTPUTS, and `o.input(0)` refuses a statement
/// that is perfectly well formed.
///
/// The table path reaches the same two buffers from the other side and gets
/// the same answer: `binding::runs` derives its result count from the row's
/// highest `Out(i)`, both append rows name none, so `results = 0` and
/// `ins = [k_new, v_new]`. Two splits, opposite ends, one pair of buffers in
/// one order -- which is why this reads INPUT FIRST and falls back to OUTPUT
/// rather than the other way round: on a statement that does state results the
/// input side is the right one, and on these two it is empty.
///
/// `driver-metal` answers this with a probe run (`asked_results` re-runs the
/// arm and counts what it asked for); this backend's `plan` has no probe, so
/// the arm carries the knowledge instead. `input` mints nothing when it
/// refuses -- it fails on `self.ins.get(n)` before `take` -- so the fallback
/// costs no handle and cannot shift the ones after it.
///
/// # Errors
///
/// [`Refusal::Empty`] when the statement carries neither an `n`th input nor an
/// `n`th output, which is a disagreement between this arm and the trace.
fn read(o: &mut Handles<'_>, n: usize) -> Result<ArgValue, Refusal> {
    if let Ok(got) = o.input(n) {
        return Ok(got);
    }
    o.output(n)
}

/// This driver's pool is PAGED, and a contiguous-stride kernel walks it wrong.
///
/// `binding::scalars` refuses any row naming `Source::KvHeadStride` or
/// `Source::KvSeqStride` with [`crate::binding::Misplaced::Contiguous`] --
/// *"operand N is a contiguous KV stride, and this driver's pool is paged: the
/// row would read real memory at the wrong tokens"* -- and that refusal covers
/// `kv_append`, `sdpa_vector_decode` and `sdpa_vector_decode_swa` today. The
/// routine path has no such gate: `Handles::fire_number` answers a head stride
/// and a sequence stride happily, because `resources::Shape::number` CAN
/// derive both from a paged pool. Arming those three without this check would
/// convert a loud blanket refusal into a dispatch that binds real memory and
/// attends to the wrong tokens, which is the exact failure the table path's
/// comment was written about.
///
/// So the check moves into the arm, one notch narrower than blanket: a page
/// size the fire answers with is a paged pool and is refused, and a fire whose
/// pool has none is served. On this driver that is the same set --
/// `model::kv::Pool` allocates `[page, token, head, dim]` for every fire, which
/// is why `llama_like` states `let paged = true` unconditionally -- so nothing
/// that runs today changes behaviour. What changes is that the reason is now
/// stated where the kernel is chosen instead of where the row is read.
///
/// # Errors
///
/// [`Refusal::Absent`] when the fire's pool states a page size.
pub fn contiguous_pool(o: &Handles<'_>) -> Result<(), Refusal> {
    if o.fire_number(crate::binding::FireNumber::KvPageSize) != 0 {
        return Err(Refusal::Absent {
            what: "a contiguous KV cache: this fire's pool is paged, and a \
                   head/sequence stride over it addresses the wrong tokens",
        });
    }
    Ok(())
}

/// `attn::kv_append`: one token into the contiguous slot `pos` names.
///
/// The row is `In(0)`, `In(1)`, `KvKeys`, `KvValues`, `Positions`, `Param(0)`,
/// `KvHeadStride`, `KvSeqStride`, and this is that list with the two cache
/// planes coming from [`Handles::kv`] and the positions from the fire. The
/// LAYER is not a parameter: `plan` takes it from `launch.layers.start`, the
/// same number `reorder` uses for `Source::KvKeys`.
///
/// `head_dim` is [`Handles::stated`] and not `param(0)?` because the row says
/// `head_param = Some(0)`: the same number reaches the SHADER and the GRID, so
/// a statement that states none must get the fire's rather than hand the
/// kernel a zero over a grid the fire sized. `heads` is `Env<i32>` -- the
/// contiguous row states no `heads_param`, so the fire's count is what
/// `geometry`'s `Rule::PerHead` reads too.
///
/// The two strides are the POOL's, eight bytes wide because `kv_write.wgsl`
/// declares them `vec2<u32>`; `routine::bind` aligns a `Usize` to eight, which
/// is what puts them at offsets 8 and 16 where the shader reads them and not
/// at 4 and 12 where a concatenating packer would.
///
/// # This form is refused on a paged pool
///
/// See [`contiguous_pool`]. The body's own `head_grid(head_dim, heads, 1)` also
/// pins the token axis at ONE -- the shader reads `pos[0]` and every head
/// writes the same destination row -- where `Rule::PerHead` gives the table
/// path `[head_dim, kv_heads, rows]`. The two agree at one row, which is the
/// only shape this kernel is written for, and disagree above it; that is a
/// second reason a multi-token fire must not reach this symbol.
///
/// # Errors
///
/// [`Refusal::Absent`] on a paged pool; [`Refusal::Empty`] for an operand the
/// statement does not carry.
pub fn kv_append(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    contiguous_pool(o)?;
    let head_dim = o.stated(0, f.head_dim);
    let k_new = read(o, 0)?;
    let v_new = read(o, 1)?;
    let k_cache = o.kv(false);
    let v_cache = o.kv(true);
    let pos = o.table(FireTable::Positions);
    Ok(vec![
        k_new,
        v_new,
        k_cache,
        v_cache,
        pos,
        ArgValue::I32(head_dim),
        ArgValue::Usize(u64::from(
            o.fire_number(crate::binding::FireNumber::KvHeadStride),
        )),
        ArgValue::Usize(u64::from(
            o.fire_number(crate::binding::FireNumber::KvSeqStride),
        )),
        ArgValue::I32(f.kv_heads.cast_signed()),
    ])
}

/// `attn::kv_append_paged`: many tokens, each into the page and offset its own
/// entry names.
///
/// SIXTEEN arguments of which six reach the device, and the sixteen are the
/// row's sixteen in order. Seven of them -- 4, 6, 7, 8, 9, 11 and 15 -- are the
/// Metal ring ABI's slots that `kv_write.wgsl`'s paged arm does not compile;
/// the row lists them as gaps rather than closing them, because an operand list
/// is positional and closing one would shift `w_page` and `w_off` onto the
/// scalars behind them.
///
/// [`Handles::unbound`] is the ask for such a slot. A `Buf` parameter refuses
/// anything that is not a buffer handle -- `shader::Arg::unpack` answers
/// `Refusal::Kind` -- so the argument has to BE a handle, and `unbound` mints
/// one that `plan` resolves to `Placed::Nothing` and `bind` binds nothing at.
/// It costs a handle and no buffer, which is exactly what `reorder`'s
/// `Slot::Nothing` costs on the other path. This body then drops all seven
/// anyway (its dispatch forwards nine values, six of them buffers), so nothing
/// here can reach a bind group even if the ask were wrong.
///
/// # Both halves of the head shape are the STATEMENT's
///
/// `head_param = Some(0)` and `heads_param = Some(1)`, and the row's own
/// comment says why: a grid built from the fire's `[256, 16]` where the
/// statement said `[512, 4]` left the top half of every gemma-4 KV head
/// unwritten and put heads 4..15 in the next token's rows. The body computes
/// `head_grid(head_dim, n_kv_heads, tokens)` from these two arguments, so the
/// same numbers size the grid here that `dims_of` reads for `Rule::PerHead`
/// there -- which is what makes the two paths' grids equal.
///
/// The page size is the POOL's and comes from the fire, not the statement.
/// Absent is zero, matching `scalars`' `resolver.number(want).unwrap_or(0)`
/// exactly; a zero page size is refused downstream rather than turned into a
/// second opinion here.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn kv_append_paged(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let head_dim = o.stated(0, f.head_dim);
    let n_kv_heads = o.stated(1, f.kv_heads);
    let page_size = o.fire_number(crate::binding::FireNumber::KvPageSize);
    let k_new = read(o, 0)?;
    let v_new = read(o, 1)?;
    let k_pages = o.kv(false);
    let v_pages = o.kv(true);
    let w_page = o.table(FireTable::KvWritePage);
    let w_off = o.table(FireTable::KvWriteOffset);
    Ok(vec![
        k_new,
        v_new,
        k_pages,
        v_pages,
        o.unbound(),
        ArgValue::I32(head_dim),
        o.unbound(),
        o.unbound(),
        o.unbound(),
        o.unbound(),
        ArgValue::I32(page_size.cast_signed()),
        o.unbound(),
        ArgValue::I32(n_kv_heads),
        w_page,
        w_off,
        o.unbound(),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// The seventeen values every `sdpa_paged_*` body opens with.
///
/// One shape for six routines plus the strided seventh, because the six
/// signatures are identical up to `sinks` and differ only in their tails. The
/// order is the shader's argument table and the row's operand list, which are
/// the same list: `sdpa_paged.wgsl` and `sdpa_paged_mma.wgsl` declare bindings
/// 0..=10 in this order and their `Params` struct is `gqa_factor, page_size,
/// n_kv_heads, scale, attention_mask_stride, window` in this order.
pub struct Paged {
    queries: ArgValue,
    k_pages: ArgValue,
    v_pages: ArgValue,
    out: ArgValue,
    gqa_factor: i32,
    position_ids: ArgValue,
    req_of_token: ArgValue,
    kv_page_indices: ArgValue,
    kv_page_indptr: ArgValue,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: ArgValue,
    attention_mask_stride: u32,
    attention_mask_enabled: ArgValue,
    window: i32,
    sinks: ArgValue,
}

impl Paged {
    /// The arguments up to and including `sinks`, which every paged form takes
    /// before its own tail.
    pub fn head(&self) -> Vec<ArgValue> {
        vec![
            self.queries,
            self.k_pages,
            self.v_pages,
            self.out,
            ArgValue::I32(self.gqa_factor),
            self.position_ids,
            self.req_of_token,
            self.kv_page_indices,
            self.kv_page_indptr,
            ArgValue::I32(self.page_size),
            ArgValue::I32(self.n_kv_heads),
            ArgValue::F32(self.scale),
            self.attention_mask,
            ArgValue::U32(self.attention_mask_stride),
            self.attention_mask_enabled,
            ArgValue::I32(self.window),
            self.sinks,
        ]
    }
}

/// What a paged attention statement carries, read once.
///
/// `sinks` is handed in rather than read here because it is the one operand
/// the six forms disagree about: the sinked three take `Weight(0)` and the
/// others take a slot nothing fills. See [`paged_sink`] and [`paged_plain`].
///
/// # `n_kv_heads` is the STATEMENT's second scalar and not `f.kv_heads`
///
/// gemma-3 carries four 512-wide KV heads in its full-attention layers and
/// sixteen 256-wide ones in its sliding layers, so no fire-wide number is right
/// for both; the row says `Param(1)` and `driver-metal`'s arm says the same
/// thing with the same reason attached. `gqa_factor` and `scale` are the same
/// kind of fact -- `dsl::metal::sdpa` states all five words as
/// `[gqa_factor, kv_heads, scale.to_bits(), 0, window]` -- and all three are
/// asked for strictly, because a zero in any of them is not a neutral value: a
/// zero `gqa_factor` divides by zero, a zero head count attends over nothing,
/// and a zero scale makes every logit zero and every distribution uniform,
/// which is finite, varied and wrong.
///
/// # The mask pitch is the FIRE's, and that is this table's divergence
///
/// The row says `Source::AttentionMaskStride` where `kernels-metal` says
/// `Param(3)` -- `DELIBERATE` in `kernels/tests/entrypoints.rs` records it --
/// and `Handles::fire_number` is the arm's door to the same number `reorder`
/// would have packed. A mask rectangle is as wide as the widest row of the fire
/// that supplied it, so no statement can know the pitch; `dsl::metal::sdpa`
/// duly states a literal `0` at word 3, which is why reading it there is the
/// live defect metal's `DRIFTED["sdpa_paged_decode"]` names and why a
/// user-supplied mask works on this backend.
///
/// # The window is word 4, and an absent one is zero
///
/// `o.param(4).unwrap_or(0)` and not `param(4)?`: the table path packs
/// `stated.get(4).copied().unwrap_or(0)` for the same operand, and the shader
/// gates on `params.window > 0`, so zero IS "no window" in both. Refusing here
/// would turn a short run into a dead rectangle where the row makes it a full
/// attention. `driver-metal` writes `-1`, which the same comparison reads the
/// same way.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand or a scalar the statement does not carry.
pub fn paged(o: &mut Handles<'_>, sinks: ArgValue) -> Result<Paged, Refusal> {
    let gqa_factor = o.param(0)?;
    let n_kv_heads = o.param(1)?;
    let scale = o.param_f32(2)?;
    let window = o.param(4).unwrap_or(0);
    let page_size = o
        .fire_number(crate::binding::FireNumber::KvPageSize)
        .cast_signed();
    let attention_mask_stride = o.fire_number(crate::binding::FireNumber::AttentionMaskStride);
    let queries = o.input(0)?;
    let k_pages = o.kv(false);
    let v_pages = o.kv(true);
    let out = o.output(0)?;
    let position_ids = o.table(FireTable::Positions);
    let req_of_token = o.table(FireTable::RequestOfToken);
    let kv_page_indices = o.table(FireTable::KvPageIndices);
    let kv_page_indptr = o.table(FireTable::KvPageIndptr);
    let attention_mask = o.table(FireTable::AttentionMask);
    let attention_mask_enabled = o.table(FireTable::AttentionMaskEnabled);
    Ok(Paged {
        queries,
        k_pages,
        v_pages,
        out,
        gqa_factor,
        position_ids,
        req_of_token,
        kv_page_indices,
        kv_page_indptr,
        page_size,
        n_kv_heads,
        scale,
        attention_mask,
        attention_mask_stride,
        attention_mask_enabled,
        window,
        sinks,
    })
}

/// [`paged`] with the per-head sink logit the gpt-oss forms read.
///
/// `Weight(0)`, which is the only weight this family names: `dsl::metal::sdpa`
/// puts `layer.N.attn_sinks` in the statement's weight list when the
/// deployment has one, and no other paged operand is a weight.
///
/// # Errors
///
/// See [`paged`], plus [`Refusal::Empty`] when the statement carries no weight.
pub fn paged_sink(o: &mut Handles<'_>) -> Result<Paged, Refusal> {
    let sinks = o.weight(0)?;
    paged(o, sinks)
}

/// [`paged`] for a form whose module declares `sinks` and never reads it.
///
/// `sdpa_paged.wgsl` declares `@group(0) @binding(10) var<storage, read_write>
/// sinks` unconditionally and reads it only under `PIE_WITH_SINK`, so the
/// non-sink entrypoints carry a DECLARED-AND-UNREAD binding. The body still
/// forwards an argument for it -- its parameter list is the sinked one's -- and
/// the row still lists the operand with no `Source`, which `reorder` answers
/// `Slot::Nothing`. [`Handles::unbound`] is that answer for an arm, and it is
/// what makes these four armable: `reflect::of_module` reports binding 10
/// unreachable, `Device::build` builds the layout from the REACHABLE set, and
/// `check_bindable` refuses a list whose length is not the slot count. An arm
/// that put a real buffer there would bind eleven where the pipeline has ten.
///
/// # Errors
///
/// See [`paged`].
pub fn paged_plain(o: &mut Handles<'_>) -> Result<Paged, Refusal> {
    let sinks = o.unbound();
    paged(o, sinks)
}

/// `attn::sdpa_paged_decode`: one query row per workgroup row, walking the
/// pages its request owns.
///
/// [`paged_plain`]'s seventeen, then the three `Env` numbers the body spends on
/// `vector_grid(head_dim, q_heads, rows)`. `geometry`'s `Rule::SdpaVector`
/// builds the same extent from the same three facts -- `module.local.at(0) *
/// dims.q_heads` on x, rows on y -- so the two paths' grids are equal by
/// construction rather than by coincidence.
///
/// # Errors
///
/// See [`paged`].
pub fn sdpa_paged_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = paged_plain(o)?;
    let mut v = s.head();
    v.extend([
        ArgValue::I32(f.head_dim.cast_signed()),
        ArgValue::I32(f.q_heads.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// `attn::sdpa_paged_decode_sink`: [`sdpa_paged_decode`] with gpt-oss's
/// per-head sink logit.
///
/// The same seventeen with `sinks` filled from `Weight(0)`, and the same tail.
/// One point is compiled, `_d_64`, because that is gpt-oss's head width, and
/// the body refuses any other by name.
///
/// # Errors
///
/// See [`paged_sink`].
pub fn sdpa_paged_decode_sink(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = paged_sink(o)?;
    let mut v = s.head();
    v.extend([
        ArgValue::I32(f.head_dim.cast_signed()),
        ArgValue::I32(f.q_heads.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ]);
    Ok(v)
}

/// The tail the four TILED forms share: `n_rows`, then the two grid numbers.
///
/// `n_rows` is `f.rows` -- the RECTANGLE's row span -- and not the fire's. The
/// row sources it `Source::Rows`, which `scalars` answers from
/// `FireNumber::Rows` (what the pool last staged), and `plan` does not hand
/// that number to an arm. The rectangle's is the right one anyway and this is
/// the place to say so: the grid rounds rows up to whole 32-row tiles and this
/// scalar is what tells the partial last tile where the end is, so it has to be
/// the same count the grid was built from. `geometry`'s `SdpaTiled` uses
/// `rows.div_ceil(32)` off the LAUNCH, and the body uses
/// `tiled_grid(q_heads, n_rows)` off this argument; passing the fire's would
/// let a rectangle narrower than its fire write past its own end.
pub fn tiled_tail(f: Facts) -> [ArgValue; 3] {
    [
        ArgValue::I32(f.rows.cast_signed()),
        ArgValue::I32(f.head_dim.cast_signed()),
        ArgValue::I32(f.q_heads.cast_signed()),
    ]
}

/// `attn::sdpa_paged_tiled`: a tile of 32 query rows against one staged run of
/// keys.
///
/// Not different arithmetic from [`sdpa_paged_decode`] -- different SHARING.
/// The decode form gives one workgroup to each (head, row) and each walks the
/// whole key run alone; this one stages the run once for 32 rows. Hence the
/// eighteenth operand, and hence [`tiled_tail`].
///
/// # Errors
///
/// See [`paged`].
pub fn sdpa_paged_tiled(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = paged_plain(o)?;
    let mut v = s.head();
    v.extend(tiled_tail(f));
    Ok(v)
}

/// `attn::sdpa_paged_tiled_sink`: [`sdpa_paged_tiled`] with the sink logit.
///
/// # Errors
///
/// See [`paged_sink`].
pub fn sdpa_paged_tiled_sink(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = paged_sink(o)?;
    let mut v = s.head();
    v.extend(tiled_tail(f));
    Ok(v)
}

/// `attn::sdpa_paged_tiled_strided`: the tiled prefill whose query and output
/// rows have their own pitches.
///
/// TWO more scalars than [`sdpa_paged_tiled`] and they are the whole
/// difference: `PIE_STRIDED` appends `q_row_pitch` and `o_row_pitch` to the
/// end of the `Params` block so a checkpoint that reads its queries out of a
/// wider buffer than it writes has somewhere to say so. One point, `_d_256`,
/// which is qwen3.5's.
///
/// # This row is BARE, so the two pitches are read from the body and the shape
///
/// `kernel!(sdpa_paged_tiled_strided "sdpa_paged_tiled_strided", axes = ...)`
/// states no operands, so there is no `Source` list to translate and nothing
/// for `the_routine_path_plans_what_the_table_path_planned` to compare against:
/// `kernels_wgpu::sig` answers, but with an empty operand list the row path
/// falls back to positional binding. The seventeen ahead of the pitches are
/// `sdpa_paged_tiled`'s exactly -- the signature is that one plus two `i32`s --
/// and the pitches follow `n_rows`, so they are words 5 and 6 of the statement's
/// run, one and two past the window at 4. `driver-metal`'s arm for the same
/// kernel reads the same two indices.
///
/// They are `param(_)?` and not `stated(_, _)`: a row pitch says how the trace
/// laid this tensor out and nothing else in the fire knows it, which is this
/// file's rule for `q_gate_split`, `residual_add_strided` and
/// `shared_expert_combine_strided` alike. The shader has no fallback for either
/// -- unlike `gate.wgsl` -- so a zero would read every row after the first from
/// offset zero.
///
/// # Errors
///
/// See [`paged`], plus [`Refusal::Empty`] for either pitch the statement does
/// not state.
pub fn sdpa_paged_tiled_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let q_row_pitch = o.param(5)?;
    let o_row_pitch = o.param(6)?;
    let s = paged_plain(o)?;
    let mut v = s.head();
    v.extend([
        ArgValue::I32(f.rows.cast_signed()),
        ArgValue::I32(q_row_pitch),
        ArgValue::I32(o_row_pitch),
        ArgValue::I32(f.head_dim.cast_signed()),
        ArgValue::I32(f.q_heads.cast_signed()),
    ]);
    Ok(v)
}

/// `attn::sdpa_paged_mma`: the cooperative-matrix prefill.
///
/// The same eighteen operands and the same 32-row tile as
/// [`sdpa_paged_tiled`] -- the `Params` struct is field for field identical --
/// so it is the same arm with the same tail. What differs is inside the shader,
/// where `Q.K^T` and `P.V` go through the matrix unit, and in the grid rule's
/// NAME: `Rule::SdpaMma` shares `SdpaTiled`'s arm in `geometry`, so even that
/// arithmetic is one expression.
///
/// The row states these operands rather than leaving them bare, and the reason
/// is worth carrying: while it was unstated the plan supplied FIVE scalars into
/// a uniform block the shader reads SEVEN fields of.
///
/// # Errors
///
/// See [`paged`].
pub fn sdpa_paged_mma(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = paged_plain(o)?;
    let mut v = s.head();
    v.extend(tiled_tail(f));
    Ok(v)
}

/// `attn::sdpa_paged_mma_sink`: [`sdpa_paged_mma`] with the sink logit.
///
/// # Errors
///
/// See [`paged_sink`].
pub fn sdpa_paged_mma_sink(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = paged_sink(o)?;
    let mut v = s.head();
    v.extend(tiled_tail(f));
    Ok(v)
}

/// What a DENSE decode statement carries: one contiguous cache, four strides.
///
/// The three vector forms share every operand up to `scale` and differ in their
/// tails, so this is the shared read. `sinks` is not part of it: the sinked
/// form takes it FIFTH, immediately after `out` and before `gqa_factor`, which
/// is the one place in this family where a sink is not the last buffer -- see
/// [`sdpa_vector_decode_sink`].
///
/// # `n` is `Param(1)`, which the paged rows read as `n_kv_heads`
///
/// The contiguous row calls word 1 `n`, the key count; the paged rows call the
/// same word `n_kv_heads`. `llama_like`'s text says so where it pins `paged =
/// true`: *"one statement cannot supply both correctly"*. The arm reproduces
/// the row it is the arm for and the collision stays the text's to avoid.
///
/// The four strides are the POOL's, `Usize` because `sdpa_vector.slang`
/// declares them `uint2`, and the head/sequence pair is repeated for K and V:
/// the row names `KvHeadStride`, `KvSeqStride`, `KvHeadStride`, `KvSeqStride`
/// in that order, because one pool shape serves both planes -- which is why
/// this holds one pair and [`Vector::head`] passes it twice.
pub struct Vector {
    queries: ArgValue,
    keys: ArgValue,
    values: ArgValue,
    out: ArgValue,
    gqa_factor: i32,
    n: i32,
    k_head_stride: u64,
    k_seq_stride: u64,
    scale: f32,
}

impl Vector {
    /// The arguments up to and including `scale`, with `sinks` spliced in
    /// after `out` where the form has one.
    pub fn head(&self, sinks: Option<ArgValue>) -> Vec<ArgValue> {
        let mut v = vec![self.queries, self.keys, self.values, self.out];
        v.extend(sinks);
        v.extend([
            ArgValue::I32(self.gqa_factor),
            ArgValue::I32(self.n),
            ArgValue::Usize(self.k_head_stride),
            ArgValue::Usize(self.k_seq_stride),
            ArgValue::Usize(self.k_head_stride),
            ArgValue::Usize(self.k_seq_stride),
            ArgValue::F32(self.scale),
        ]);
        v
    }
}

/// Read one dense decode statement. See [`Vector`].
///
/// # Errors
///
/// [`Refusal::Absent`] on a paged pool (see [`contiguous_pool`]);
/// [`Refusal::Empty`] for an operand or scalar the statement does not carry.
pub fn vector(o: &mut Handles<'_>) -> Result<Vector, Refusal> {
    contiguous_pool(o)?;
    let gqa_factor = o.param(0)?;
    let n = o.param(1)?;
    let scale = o.param_f32(2)?;
    let k_head_stride = u64::from(o.fire_number(crate::binding::FireNumber::KvHeadStride));
    let k_seq_stride = u64::from(o.fire_number(crate::binding::FireNumber::KvSeqStride));
    let queries = o.input(0)?;
    let keys = o.kv(false);
    let values = o.kv(true);
    let out = o.output(0)?;
    Ok(Vector {
        queries,
        keys,
        values,
        out,
        gqa_factor,
        n,
        k_head_stride,
        k_seq_stride,
        scale,
    })
}

/// The three `Env` numbers every vector form ends with.
fn vector_tail(f: Facts) -> [ArgValue; 3] {
    [
        ArgValue::I32(f.head_dim.cast_signed()),
        ArgValue::I32(f.q_heads.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ]
}

/// `attn::sdpa_vector_decode`: the dense, unpaged decode.
///
/// Eleven operands, dense 0..=10, and the row is where the WIDTHS live: a
/// driver handing a four-byte slot to an eight-byte read gives the kernel the
/// next scalar as this one's high half.
///
/// # Errors
///
/// See [`vector`].
pub fn sdpa_vector_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = vector(o)?;
    let mut v = s.head(None);
    v.extend(vector_tail(f));
    Ok(v)
}

/// `attn::sdpa_vector_decode_swa`: the dense decode over a SLIDING window.
///
/// The window is an operand and not a flag -- the port's rule that a per-fire
/// choice the C++ made at encode time becomes data on the dispatch -- and it
/// brings two row pitches the contiguous form has no fields for, because gemma
/// reads its query out of a wider buffer than it writes.
///
/// # The three tail scalars are words 3, 4 and 5, which the PAGED rows are not
///
/// This row states `window <- Param(3)`, `q_row_stride <- Param(4)`,
/// `o_row_stride <- Param(5)`, while every `sdpa_paged_*` row reads word 3 as
/// the mask pitch and word 4 as the window. The two indexings collide, exactly
/// as `n`/`n_kv_heads` collides at word 1: a statement written for one family
/// hands the other its window as a row pitch. The arm reproduces its own row;
/// the collision is a fact about the two rows and is named again in the closing
/// prose.
///
/// The pitches are strict for [`sdpa_paged_tiled_strided`]'s reason. The window
/// is strict HERE, unlike the paged forms: there the row's word is a documented
/// literal zero that means "no window", while a sliding statement that states
/// no window has nothing to slide and would silently become a full attention
/// over the whole cache.
///
/// # Errors
///
/// See [`vector`], plus [`Refusal::Empty`] for the window or either pitch.
pub fn sdpa_vector_decode_swa(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let window = o.param(3)?;
    let q_row_stride = o.param(4)?;
    let o_row_stride = o.param(5)?;
    let s = vector(o)?;
    let mut v = s.head(None);
    v.extend([
        ArgValue::I32(window),
        ArgValue::I32(q_row_stride),
        ArgValue::I32(o_row_stride),
    ]);
    v.extend(vector_tail(f));
    Ok(v)
}

/// `attn::sdpa_vector_decode_sink`: the dense decode with attention sinks.
///
/// **[`sdpa_vector_decode_swa`] plus one buffer, not [`sdpa_vector_decode`]
/// plus one.** The entrypoint is instantiated in `attn/sdpa_sliding.wgsl` with
/// `PIE_WITH_SINK=1`, so it takes the SLIDING `Params` -- the window and both
/// row pitches are in the block whether or not a caller wants a window -- and
/// `sinks` takes `@group(0) @binding(4)`, which the windowed form leaves
/// undeclared. That binding number is why `sinks` is the FIFTH argument and not
/// the last: the buffer list is dense in binding order, so the sink plane sits
/// between `out` at 3 and the first scalar.
///
/// This row is BARE, like `sdpa_paged_tiled_strided`'s and `gate`'s, so the
/// reading is the sibling row's plus the shader's binding table rather than a
/// `Source` list. `Weight(0)` for the sink follows every other sinked form in
/// the family and `dsl::metal::sdpa`, which puts `layer.N.attn_sinks` in the
/// statement's weight list.
///
/// # Errors
///
/// See [`vector`], plus [`Refusal::Empty`] for the missing weight, window or
/// either pitch.
pub fn sdpa_vector_decode_sink(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let window = o.param(3)?;
    let q_row_stride = o.param(4)?;
    let o_row_stride = o.param(5)?;
    let sinks = o.weight(0)?;
    let s = vector(o)?;
    let mut v = s.head(Some(sinks));
    v.extend([
        ArgValue::I32(window),
        ArgValue::I32(q_row_stride),
        ArgValue::I32(o_row_stride),
    ]);
    v.extend(vector_tail(f));
    Ok(v)
}

/// The three TRANSCODES, which share a shape: planes in, planes out, and a
/// pair of scalars the shader reads from its `@group(1)` uniform.
///
/// `count` is how many groups or blocks the run covers and `size` is how wide
/// one is — `transcode.wgsl` calls them `groups`/`group_size` for the affine
/// encoders and `blocks`/`block_size` for the mxfp4 dequant, and they are the
/// statement's first two words either way.
///
/// These rows state NO operands, so nothing checked the pair until now: the
/// bodies took a `_params: Buf` they could not use and forwarded no scalars at
/// all, which meant the block arrived empty and the shader read zero groups —
/// a loop that runs no iterations and reports success. The bodies forward the
/// pair now and this is what supplies it.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand or scalar the statement does not carry.
pub fn transcode(o: &mut Handles<'_>, outs: usize) -> Result<Vec<ArgValue>, Refusal> {
    let count = o.param(0)?;
    let size = o.param(1)?;
    let mut args = vec![o.input(0)?];
    for n in 0..outs {
        args.push(o.output(n)?);
    }
    args.push(ArgValue::I32(count));
    args.push(ArgValue::I32(size));
    Ok(args)
}

/// `quant::encode_u4_bf16`. See [`transcode`].
///
/// # Errors
///
/// See [`transcode`].
pub fn encode_u4_bf16(o: &mut Handles<'_>, _f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    transcode(o, 3)
}

/// `quant::encode_u4_f32`. See [`transcode`].
///
/// # Errors
///
/// See [`transcode`].
pub fn encode_u4_f32(o: &mut Handles<'_>, _f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    transcode(o, 3)
}

/// `quant::mxfp4_dequant_bf16`: two planes in, one out. See [`transcode`].
///
/// The second INPUT is the exponent plane, which the affine encoders do not
/// have, so this cannot use [`transcode`] unchanged.
///
/// # Errors
///
/// See [`transcode`].
pub fn mxfp4_dequant_bf16(o: &mut Handles<'_>, _f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let blocks = o.param(0)?;
    let block_size = o.param(1)?;
    let payload = o.input(0)?;
    let exponents = o.input(1)?;
    let out = o.output(0)?;
    Ok(vec![
        payload,
        exponents,
        out,
        ArgValue::I32(blocks),
        ArgValue::I32(block_size),
    ])
}

/// `mlp::silu_mul_strided`: [`silu_mul`] over rows a pitch apart.
///
/// The fleet's LAST unarmed kernel, and it was unarmed because every backend
/// had inherited metal's conclusion that it cannot take a positional argument
/// list. On this backend `gated.wgsl` numbers its three buffers densely and
/// puts the pitch in a uniform of its own, so it can.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand or scalar the statement does not carry.
pub fn silu_mul_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let pitch = o.param(0)?;
    let gate = o.input(0)?;
    let up = o.input(1)?;
    let out = o.output(0)?;
    Ok(vec![
        gate,
        up,
        out,
        ArgValue::I32(pitch),
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// One crossed kernel: the entrypoint STEM a plan spells it with, and the arm
/// that feeds it.
///
/// # Why a stem and not a name
///
/// A plan names `silu_mul_bfloat16`, not `silu_mul`. A row could be found from
/// that because `kernels::sig_in` falls back to an axis match, but the fork
/// sits ABOVE the row lookup — that is what lets rows be deleted — so it has
/// to answer from the symbol alone. This is the same answer
/// `driver-vulkan::arm::Crossed` and `driver-metal`'s `crossed` reached.
///
/// Matching by name alone worked for exactly the two DARK families and hid
/// this: `argmax_logits` and `copy_logits_bf16` have no axis, so their symbol
/// IS their stem. `silu_mul` was the first live arm and it was never reached —
/// `the_routine_path_plans_what_the_table_path_planned` compared zero
/// rectangles and said so, which is why that test asserts a floor.
pub struct Crossed {
    /// The longest prefix of a symbol that names this kernel.
    ///
    /// Usually the routine's own name, and NOT always: `moe`'s routed GEMMs
    /// are spelled `affine_qmm_t_routed_bfloat16_gs_64_b_4` and the routine is
    /// called `qmm_t_routed`, because on this backend the quantization scheme
    /// is a prefix the row name never carried. [`Self::routine`] is how those
    /// two are told apart.
    pub stem: &'static str,
    /// The routine this stem names, where it is not the stem itself.
    pub routine: Option<&'static str>,
    /// The arm, or `None` for a stem that is CLAIMED but not yet armed.
    pub arm: Option<Arm>,
}

/// The crossed kernels, by stem.
///
/// # Longest match, and why an unarmed stem still belongs here
///
/// Stems nest. `silu_mul` is a prefix of `silu_mul_strided_bfloat16`, and a
/// first match would hand a STRIDED rectangle to the contiguous body: it binds
/// real buffers, dispatches, and reads every row from the wrong offset.
/// Nothing downstream would notice — the shapes are identical and both are
/// storage buffers. So `silu_mul_strided` is listed with no arm, which makes
/// it the longer match and sends its symbols back to the table path where
/// they still belong.
///
/// `every_entrypoint_is_claimed_by_the_stem_that_owns_it` checks that over the
/// whole census rather than over the pairs anyone thought of.
static LIVE: &[Crossed] = &[
    Crossed {
        stem: "argmax_logits",
        routine: None,
        arm: Some(argmax_logits as Arm),
    },
    Crossed {
        stem: "copy_logits_bf16",
        routine: None,
        arm: Some(copy_logits_bf16 as Arm),
    },
    Crossed {
        stem: "silu_mul",
        routine: None,
        arm: Some(silu_mul as Arm),
    },
    // CLAIMED, NOT ARMED. `mlp/gated.wgsl`'s strided variant walks rows by a
    // pitch the contiguous body does not take, and it keeps `silu_mul` from
    // claiming its symbols by prefix.
    Crossed {
        stem: "silu_mul_strided",
        routine: None,
        arm: Some(silu_mul_strided as Arm),
    },
    Crossed {
        stem: "geglu_tanh",
        routine: None,
        arm: Some(geglu_tanh as Arm),
    },
    Crossed {
        stem: "geglu_tanh_strided",
        routine: None,
        arm: Some(geglu_tanh_strided as Arm),
    },
    Crossed {
        stem: "gptoss_swiglu",
        routine: None,
        arm: Some(gptoss_swiglu as Arm),
    },
    // norm. Longest-match matters throughout: `rms_residual` nests inside
    // `rms_residual_scaled`, `rms_strided_row` inside `rms_strided_head_row`,
    // `gated_rms` inside `gated_rms_strided`, and `residual_add` inside
    // `residual_add_strided` — four pairs where a first match would hand a
    // rectangle to a body that binds one fewer buffer and reads the wrong
    // pitch. `every_entrypoint_is_claimed_by_the_stem_that_owns_it` checks it
    // over the whole census.
    Crossed {
        stem: "rms_single_row",
        routine: None,
        arm: Some(rms_single_row as Arm),
    },
    Crossed {
        stem: "vnorm_single_row",
        routine: None,
        arm: Some(vnorm_single_row as Arm),
    },
    Crossed {
        stem: "rms_strided_row",
        routine: None,
        arm: Some(rms_strided_row as Arm),
    },
    Crossed {
        stem: "rms_strided_head_row",
        routine: None,
        arm: Some(rms_strided_head_row as Arm),
    },
    Crossed {
        stem: "rms_residual",
        routine: None,
        arm: Some(rms_residual as Arm),
    },
    Crossed {
        stem: "rms_residual_scaled",
        routine: None,
        arm: Some(rms_residual_scaled as Arm),
    },
    Crossed {
        stem: "gated_rms",
        routine: None,
        arm: Some(gated_rms as Arm),
    },
    Crossed {
        stem: "gated_rms_strided",
        routine: None,
        arm: Some(gated_rms_strided as Arm),
    },
    Crossed {
        stem: "layer_scalar_mul",
        routine: None,
        arm: Some(layer_scalar_mul as Arm),
    },
    Crossed {
        stem: "residual_add",
        routine: None,
        arm: Some(residual_add as Arm),
    },
    Crossed {
        stem: "residual_add_strided",
        routine: None,
        arm: Some(residual_add_strided as Arm),
    },
    Crossed {
        stem: "add_bias",
        routine: None,
        arm: Some(add_bias as Arm),
    },
    // layout
    Crossed {
        stem: "embed_gather_4bit",
        routine: None,
        arm: Some(embed_gather_4bit as Arm),
    },
    Crossed {
        stem: "embed_gather_mb_4bit",
        routine: None,
        arm: Some(embed_gather_mb_4bit as Arm),
    },
    Crossed {
        stem: "embed_gather_scaled_4bit",
        routine: None,
        arm: Some(embed_gather_scaled_4bit as Arm),
    },
    Crossed {
        stem: "embed_gather_scaled_mb_4bit",
        routine: None,
        arm: Some(embed_gather_scaled_mb_4bit as Arm),
    },
    Crossed {
        stem: "ple_combine",
        routine: None,
        arm: Some(ple_combine as Arm),
    },
    Crossed {
        stem: "row_gather",
        routine: None,
        arm: Some(row_gather as Arm),
    },
    // quant
    Crossed {
        stem: "affine_qmm_t",
        routine: Some("qmm_t"),
        arm: Some(qmm_t as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_bias",
        routine: Some("qmm_t_bias"),
        arm: Some(qmm_t_bias as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_bias_fp16_precast",
        routine: Some("qmm_t_bias_fp16_precast"),
        arm: Some(qmm_t_bias_fp16_precast as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_fp16_precast",
        routine: Some("qmm_t_fp16_precast"),
        arm: Some(qmm_t_fp16_precast as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_residual",
        routine: Some("qmm_t_residual"),
        arm: Some(qmm_t_residual as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_residual_fp16_precast",
        routine: Some("qmm_t_residual_fp16_precast"),
        arm: Some(qmm_t_residual_fp16_precast as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_splitk",
        routine: Some("qmm_t_splitk"),
        arm: Some(qmm_t_splitk as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_splitk_f32",
        routine: Some("qmm_t_splitk_f32"),
        arm: Some(qmm_t_splitk_f32 as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_splitk_fp16_precast",
        routine: Some("qmm_t_splitk_fp16_precast"),
        arm: Some(qmm_t_splitk_fp16_precast as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_splitk_fp16_precast_f32",
        routine: Some("qmm_t_splitk_fp16_precast_f32"),
        arm: Some(qmm_t_splitk_fp16_precast_f32 as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_strided",
        routine: Some("qmm_t_strided"),
        arm: Some(qmm_t_strided as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_strided_residual",
        routine: Some("qmm_t_strided_residual"),
        arm: Some(qmm_t_strided_residual as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_strided_fp16_precast",
        routine: Some("qmm_t_strided_fp16_precast"),
        arm: Some(qmm_t_strided_fp16_precast as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_strided_fp16_precast_residual",
        routine: Some("qmm_t_strided_fp16_precast_residual"),
        arm: Some(qmm_t_strided_fp16_precast_residual as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4"),
        arm: Some(qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4 as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2"),
        arm: Some(qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2 as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2"),
        arm: Some(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2 as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1"),
        arm: Some(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1 as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4"),
        arm: Some(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4 as Arm),
    },
    Crossed {
        stem: "qmm_splitk_reduce",
        routine: None,
        arm: Some(qmm_splitk_reduce as Arm),
    },
    Crossed {
        stem: "qmm_splitk_reduce_f32",
        routine: None,
        arm: Some(qmm_splitk_reduce_f32 as Arm),
    },
    Crossed {
        stem: "cast_qmm_input_bfloat16_to_float16",
        routine: None,
        arm: Some(cast_qmm_input_bfloat16_to_float16 as Arm),
    },
    Crossed {
        stem: "cast_qmm_input_strided_bfloat16_to_float16",
        routine: None,
        arm: Some(cast_qmm_input_strided_bfloat16_to_float16 as Arm),
    },
    Crossed {
        stem: "affine_qmv_fast",
        routine: Some("qmv_fast"),
        arm: Some(qmv_fast as Arm),
    },
    Crossed {
        stem: "affine_qmv_fast_residual",
        routine: Some("qmv_fast_residual"),
        arm: Some(qmv_fast_residual as Arm),
    },
    Crossed {
        stem: "affine_qmv_tail",
        routine: Some("qmv_tail"),
        arm: Some(qmv_tail as Arm),
    },
    Crossed {
        stem: "affine_qmv_tail_bias",
        routine: Some("qmv_tail_bias"),
        arm: Some(qmv_tail_bias as Arm),
    },
    Crossed {
        stem: "affine_qmv_wide_strided",
        routine: Some("qmv_wide_strided"),
        arm: Some(qmv_wide_strided as Arm),
    },
    Crossed {
        stem: "affine_encode_u4_bf16",
        routine: Some("encode_u4_bf16"),
        arm: Some(encode_u4_bf16 as Arm),
    },
    Crossed {
        stem: "affine_encode_u4_f32",
        routine: Some("encode_u4_f32"),
        arm: Some(encode_u4_f32 as Arm),
    },
    Crossed {
        stem: "mxfp4_dequant_bf16",
        routine: None,
        arm: Some(mxfp4_dequant_bf16 as Arm),
    },
    // attn
    Crossed {
        stem: "split_qkv_bf16",
        routine: None,
        arm: Some(split_qkv_bf16 as Arm),
    },
    Crossed {
        stem: "gate",
        routine: None,
        arm: Some(gate as Arm),
    },
    Crossed {
        stem: "kv_append",
        routine: None,
        arm: Some(kv_append as Arm),
    },
    Crossed {
        stem: "kv_append_paged",
        routine: None,
        arm: Some(kv_append_paged as Arm),
    },
    Crossed {
        stem: "logit_softcap",
        routine: None,
        arm: Some(logit_softcap as Arm),
    },
    Crossed {
        stem: "q_gate_split",
        routine: None,
        arm: Some(q_gate_split as Arm),
    },
    Crossed {
        stem: "sdpa_paged_decode",
        routine: None,
        arm: Some(sdpa_paged_decode as Arm),
    },
    Crossed {
        stem: "sdpa_paged_decode_sink",
        routine: None,
        arm: Some(sdpa_paged_decode_sink as Arm),
    },
    Crossed {
        stem: "sdpa_paged_mma",
        routine: None,
        arm: Some(sdpa_paged_mma as Arm),
    },
    Crossed {
        stem: "sdpa_paged_mma_sink",
        routine: None,
        arm: Some(sdpa_paged_mma_sink as Arm),
    },
    Crossed {
        stem: "sdpa_paged_tiled",
        routine: None,
        arm: Some(sdpa_paged_tiled as Arm),
    },
    Crossed {
        stem: "sdpa_paged_tiled_sink",
        routine: None,
        arm: Some(sdpa_paged_tiled_sink as Arm),
    },
    Crossed {
        stem: "sdpa_paged_tiled_strided",
        routine: None,
        arm: Some(sdpa_paged_tiled_strided as Arm),
    },
    Crossed {
        stem: "sdpa_vector_decode",
        routine: None,
        arm: Some(sdpa_vector_decode as Arm),
    },
    Crossed {
        stem: "sdpa_vector_decode_sink",
        routine: None,
        arm: Some(sdpa_vector_decode_sink as Arm),
    },
    Crossed {
        stem: "sdpa_vector_decode_swa",
        routine: None,
        arm: Some(sdpa_vector_decode_swa as Arm),
    },
    // ssm
    Crossed {
        stem: "gdn_core",
        routine: None,
        arm: Some(gdn_core as Arm),
    },
    Crossed {
        stem: "gdn_core_recurrent",
        routine: None,
        arm: Some(gdn_core_recurrent as Arm),
    },
    Crossed {
        stem: "gdn_core_recurrent_prefill",
        routine: None,
        arm: Some(gdn_core_recurrent_prefill as Arm),
    },
    Crossed {
        stem: "gdn_core_recurrent_slotted",
        routine: None,
        arm: Some(gdn_core_recurrent_slotted as Arm),
    },
    Crossed {
        stem: "gdn_core_slotted",
        routine: None,
        arm: Some(gdn_core_slotted as Arm),
    },
    Crossed {
        stem: "gdn_prep",
        routine: None,
        arm: Some(gdn_prep as Arm),
    },
    Crossed {
        stem: "gdn_prep_prefill",
        routine: None,
        arm: Some(gdn_prep_prefill as Arm),
    },
    Crossed {
        stem: "gdn_prep_slotted",
        routine: None,
        arm: Some(gdn_prep_slotted as Arm),
    },
    // moe
    Crossed {
        stem: "router_topk",
        routine: None,
        arm: Some(router_topk as Arm),
    },
    Crossed {
        stem: "router_topk_scaled",
        routine: None,
        arm: Some(router_topk_scaled as Arm),
    },
    Crossed {
        stem: "route_sort",
        routine: None,
        arm: Some(route_sort as Arm),
    },
    Crossed {
        stem: "route_gather",
        routine: None,
        arm: Some(route_gather as Arm),
    },
    Crossed {
        stem: "combine_sorted",
        routine: None,
        arm: Some(combine_sorted as Arm),
    },
    Crossed {
        stem: "shared_expert_combine",
        routine: None,
        arm: Some(shared_expert_combine as Arm),
    },
    Crossed {
        stem: "shared_expert_combine_strided",
        routine: None,
        arm: Some(shared_expert_combine_strided as Arm),
    },
    Crossed {
        stem: "affine_qmv_routed",
        routine: Some("qmv_routed"),
        arm: Some(qmv_routed as Arm),
    },
    Crossed {
        stem: "affine_qmv_routed_bias",
        routine: Some("qmv_routed_bias"),
        arm: Some(qmv_routed_bias as Arm),
    },
    Crossed {
        stem: "mxfp4_qmv_routed_bias",
        routine: None,
        arm: Some(mxfp4_qmv_routed_bias as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_routed",
        routine: Some("qmm_t_routed"),
        arm: Some(qmm_t_routed as Arm),
    },
    Crossed {
        stem: "affine_qmm_t_routed_fp16",
        routine: Some("qmm_t_routed_fp16"),
        arm: Some(qmm_t_routed_fp16 as Arm),
    },
    Crossed {
        stem: "mxfp4_qmm_t_routed_bias",
        routine: None,
        arm: Some(mxfp4_qmm_t_routed_bias as Arm),
    },
    // rope
    Crossed {
        stem: "neox_decode",
        routine: None,
        arm: Some(neox_decode as Arm),
    },
    Crossed {
        stem: "neox_mb",
        routine: None,
        arm: Some(neox_mb as Arm),
    },
    Crossed {
        stem: "neox_freqs_decode",
        routine: None,
        arm: Some(neox_freqs_decode as Arm),
    },
    Crossed {
        stem: "neox_freqs_mb",
        routine: None,
        arm: Some(neox_freqs_mb as Arm),
    },
    Crossed {
        stem: "neox_prop_decode",
        routine: None,
        arm: Some(neox_prop_decode as Arm),
    },
    Crossed {
        stem: "neox_prop_mb",
        routine: None,
        arm: Some(neox_prop_mb as Arm),
    },
    Crossed {
        stem: "neox_strided",
        routine: None,
        arm: Some(neox_strided as Arm),
    },
];

/// The stem and arm for a symbol, if this backend has crossed AND armed it.
///
/// `None` is the ordinary answer today and means *"take the table path"*.
#[must_use]
pub fn crossed(symbol: &str) -> Option<(&'static str, Arm)> {
    let found = LIVE
        .iter()
        .filter(|c| {
            symbol
                .strip_prefix(c.stem)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
        })
        .max_by_key(|c| c.stem.len())?;
    Some((found.routine.unwrap_or(found.stem), found.arm?))
}

/// The arm for a symbol, if this backend has crossed and armed it.
#[must_use]
pub fn arm_for(symbol: &str) -> Option<Arm> {
    crossed(symbol).map(|(_, arm)| arm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entrypoint is claimed by the stem that OWNS it, or by none.
    ///
    /// The nesting trap over the whole census rather than over the pairs
    /// anyone thought of. `silu_mul` is a prefix of `silu_mul_strided_bfloat16`
    /// and the two are different kernels; a claim there would hand a strided
    /// rectangle to the contiguous body, which binds real buffers, dispatches,
    /// and reads every row from the wrong offset. Both are storage buffers of
    /// the same length, so nothing downstream would see it.
    ///
    /// What makes this checkable without a second list: an entrypoint belongs
    /// to the ROW or ROUTINE whose name is its longest prefix, and
    /// `kernels-wgpu` already states both. So the claim is only correct when
    /// the claiming stem IS that owner.
    #[test]
    fn every_entrypoint_is_claimed_by_the_stem_that_owns_it() {
        let owners: Vec<&str> = kernels_wgpu::KERNELS
            .iter()
            .map(|k| k.name)
            .chain(kernels_wgpu::routines().into_iter().map(|r| r.name))
            .collect();

        let mut claimed = 0u32;
        for point in kernels_wgpu::entrypoints() {
            let Some((stem, _)) = crossed(&point) else {
                continue;
            };
            claimed += 1;
            // WORD-BOUNDED, not prefix. `moe`'s routed GEMMs are spelled
            // `affine_qmm_t_routed_bfloat16_gs_64_b_4` and the routine is
            // `qmm_t_routed`: the quantization scheme is a PREFIX the routine
            // name never carried, so an owner is a name that appears bounded
            // by underscores rather than one the symbol starts with.
            // `kernels-metal::kernel_of` had exactly this defect and it cost
            // 363 of 479 entrypoints.
            let bounded = |name: &str| {
                let mut from = 0;
                while let Some(at) = point[from..].find(name) {
                    let start = from + at;
                    let end = start + name.len();
                    let before = start == 0 || point.as_bytes()[start - 1] == b'_';
                    let after = end == point.len() || point.as_bytes()[end] == b'_';
                    if before && after {
                        return true;
                    }
                    from = start + 1;
                }
                false
            };
            let owner = owners
                .iter()
                .filter(|n| bounded(n))
                .max_by_key(|n| n.len())
                .unwrap_or_else(|| panic!("`{point}` is named by no row and no routine"));
            assert_eq!(
                stem, *owner,
                "`{point}` is claimed by `{stem}` and owned by `{owner}`. A \
                 stem that claims a sibling's symbols sends them to the wrong \
                 body, which binds, dispatches and answers wrongly",
            );
        }
        assert!(
            claimed > 0,
            "no entrypoint is claimed, so this test compared nothing"
        );
    }

    /// The two dark families and the first live one are armed.
    ///
    /// `sample` and `ptir` are the two symbols no text on this backend names,
    /// so arming them could not change what any model computes — which is why
    /// they went first, and the same pair `kernels-metal` and `kernels-vulkan`
    /// crossed first for the same reason.
    ///
    /// The number is asserted rather than bounded so that the next family to
    /// land here is a deliberate edit. `arm_for` returning `None` is what
    /// keeps every other kernel on the table path, so a name added here
    /// CHANGES how a real fire is planned and should never arrive quietly.
    #[test]
    fn the_armed_stems_are_the_ones_registered_and_nothing_else() {
        assert!(arm_for("argmax_logits").is_some());
        assert!(arm_for("copy_logits_bf16").is_some());
        // EVERY kernel is armed now, so what this test can still falsify is
        // the LOOKUP, not the roster: a symbol resolves to the longest stem
        // that ends on a word boundary, and to nothing otherwise.
        assert!(arm_for("affine_qmv_routed_bfloat16_gs_64_b_4").is_some());
        assert!(arm_for("sdpa_paged_decode_bfloat16_d_128").is_some());
        // THE AFFINE TRAP, pinned. The quantization scheme is a PREFIX the
        // routine's name never carries, so `qmv_routed` is what the body is
        // called and `affine_qmv_routed_...` is what a plan spells. A lookup
        // matching on the routine name would find nothing here, which is the
        // defect `kernels-metal::kernel_of` shipped for 363 of its 479
        // entrypoints — and which this crate then reproduced twice.
        assert!(arm_for("qmv_routed_bfloat16_gs_64_b_4").is_none());
        // Armed, and reached through the SYMBOL a plan actually spells.
        assert!(arm_for("silu_mul").is_some());
        assert!(arm_for("silu_mul_bfloat16").is_some());
        assert!(arm_for("argmax_logits_bfloat16").is_some());
        // THE NESTING TRAP, and it is now a live one. `silu_mul` is a PREFIX
        // of `silu_mul_strided` and both are armed, so a lookup that took the
        // first match would plan the strided kernel with the contiguous body
        // — three buffers where the shader wants three and a pitch, and a flat
        // grid where it wants rows. Asserting `is_some()` would not catch
        // that; the STEM is what has to be checked.
        for (symbol, stem) in [
            ("silu_mul_strided_bfloat16", "silu_mul_strided"),
            ("silu_mul_bfloat16", "silu_mul"),
            ("geglu_tanh_strided_bfloat16", "geglu_tanh_strided"),
            ("geglu_tanh_bfloat16", "geglu_tanh"),
            ("residual_add_strided_bfloat16", "residual_add_strided"),
            ("residual_add_bfloat16", "residual_add"),
        ] {
            let (found, _) = crossed(symbol).expect("armed");
            assert_eq!(
                found, stem,
                "`{symbol}` resolved to `{found}`, not `{stem}`: the lookup                  stopped at a prefix instead of taking the longest stem"
            );
        }
        // A stem may not end mid-word.
        assert!(arm_for("silu_multiply").is_none());
        // And a name no backend has.
        assert!(arm_for("not_a_kernel").is_none());

        // COUNTING BY ROUTINE NAME IS THE TRAP ITSELF. `arm_for(r.name)` is
        // `None` for all 30 routines whose plan-spelling carries a scheme
        // prefix — `qmv_routed` is the body, `affine_qmv_routed_...` is the
        // symbol — so that measure reads 70 of 100 while every kernel is in
        // fact reachable. It is the SYMBOL that has to be counted, because the
        // symbol is what `plan_one` is handed.
        let points = kernels_wgpu::entrypoints();
        let orphans: Vec<&String> = points.iter().filter(|p| crossed(p).is_none()).collect();
        assert!(
            orphans.is_empty(),
            "{} of {} entrypoints have no arm, starting with {:?}. THE TABLE \
             IS EMPTY, so this is no longer a count that may grow at leisure: \
             an entrypoint no stem claims cannot be planned at all, by any \
             path. It was 481 unclaimed when the refactor began.",
            orphans.len(),
            points.len(),
            &orphans[..orphans.len().min(4)],
        );
    }

    /// No stem is registered twice, and the lookup could not say if one were.
    ///
    /// `LIVE` held `affine_qmm_t_routed` and `affine_qmm_t_routed_fp16` TWICE
    /// each -- once with `arm: None` from when the family had crossed but its
    /// arm had not landed, and again with `arm: Some` when it did. Nothing
    /// failed, because `crossed` picks with `max_by_key(|c| c.stem.len())` and
    /// Rust's `max_by_key` returns the LAST maximum, so the armed row won on a
    /// tie-break that is a documented property of the standard library rather
    /// than anything this file says. Reverse the two and every routed GEMM
    /// silently loses its arm.
    ///
    /// A duplicate stem has no correct resolution, so it is refused here
    /// rather than ordered around.
    #[test]
    fn no_stem_is_registered_twice() {
        let mut seen: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
        for c in LIVE {
            *seen.entry(c.stem).or_default() += 1;
        }
        let twice: Vec<(&&str, &usize)> = seen.iter().filter(|(_, n)| **n > 1).collect();
        assert!(
            twice.is_empty(),
            "these stems are registered more than once, and which `Crossed` \
             answers for them is decided by `max_by_key`'s tie-break and not \
             by this file: {twice:?}"
        );
    }

    /// A body asks in its own order, and that order is what gets bound.
    ///
    /// `argmax_logits` takes `logits, next_token, params, eos_flag` — an
    /// input, an output, an input, an output — so the handles it is given
    /// must be 0, 1, 2, 3 in the order ASKED and not in the order the trace
    /// states. That is the whole reason `Handles` mints handles on demand
    /// instead of returning the statement's own indices.
    #[test]
    fn handles_are_minted_in_the_order_the_body_asks() {
        let args: Vec<Arg> = (0..4)
            .map(|n| Arg::Arena {
                at: n * 64,
                width: 1,
                bytes: 2,
            })
            .collect();
        let mut o = Handles::over(&args, 2);
        let f = facts("x", 7, Geometry::default(), 1, 1024, 1024);
        let args = argmax_logits(&mut o, f).expect("four operands");
        // The handle SEQUENCE is 0, 1, 2, 3 by construction -- they are
        // minted in order -- so asserting it proves nothing on its own. What
        // it does pin is that the body asked four times and got four
        // distinct slots.
        assert_eq!(
            args,
            vec![
                ArgValue::Buffer(0),
                ArgValue::Buffer(1),
                ArgValue::Buffer(2),
                ArgValue::Buffer(3),
                ArgValue::U32(7),
            ]
        );

        // THE CLAIM. `argmax_logits` binds `logits, next_token, params,
        // eos_flag`, which is in, OUT, in, out; the lowering states its reads
        // before its writes, so the statement is `logits, params, next_token,
        // eos_flag`. Handle 1 must therefore be the statement's THIRD operand
        // and handle 2 its second -- the reorder is the whole reason this
        // type exists, and it is invisible in the handle numbers.
        let at = |arg: &Arg| match arg {
            Arg::Arena { at, .. } => *at,
            _ => unreachable!("this fixture states only arena operands"),
        };
        let taken: Vec<u32> = o
            .taken()
            .iter()
            .map(at)
            .map(u32::try_from)
            .map(|v| v.expect("a small offset"))
            .collect();
        assert_eq!(
            taken,
            vec![0, 128, 64, 192],
            "the body's order was not applied: handle 1 should be the \
             statement's third operand and handle 2 its second"
        );
        assert!(!o.wants_block(), "argmax reads no packed parameter block");
    }

    /// A statement short of an operand is refused, not indexed past.
    #[test]
    fn a_statement_the_arm_cannot_fill_is_refused() {
        let args = [Arg::Arena {
            at: 0,
            width: 1,
            bytes: 2,
        }];
        let mut o = Handles::over(&args, 2);
        let f = facts("x", 1, Geometry::default(), 1, 1024, 1024);
        assert!(matches!(
            argmax_logits(&mut o, f),
            Err(Refusal::Empty { .. })
        ));
    }
}
