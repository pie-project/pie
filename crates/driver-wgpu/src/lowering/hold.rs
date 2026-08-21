//! What a statement HOLDS, and how the shared binder reaches into it.
//!
//! A `kernels-wgpu` routine BODY states the entrypoint, the module and the
//! grid, and takes its operands as typed arguments. What it cannot do is find
//! them: a body is handed `Buf`s and `i32`s, and turning a traced statement
//! into that argument list is the driver's half. [`Handles`] is what holds
//! the raw material for that -- the launch's resolved operands, the
//! statement's scalars, and this driver's own tables and pools -- and it
//! mints a handle per ask so a bound argument list is a list of indices the
//! encoder can fill. [`Facts`] is the fire's geometry, which is what a source
//! names when it names something no statement carries.
//!
//! The reaching itself is not here. It is `kernels::bind`, shared with metal
//! and vulkan, driven by the `sources` column each routine already states.
//! [`super::bind`] is this plane's half of that: the `Holds` impl that turns
//! a key into one of the things above.
//!
//! # The arms that stood here
//!
//! This file was 5,277 lines and most of them were ARMS: one function per
//! crossed kernel, reading the statement positionally -- `o.input(0)`,
//! `o.param(3)`, `o.fire_number(KvPageSize)` -- and handing a body the list
//! it wanted. A hundred kernels, a hundred arms, and every one of them a
//! SECOND spelling of an order the routine's own signature already stated.
//!
//! Two spellings agree until they do not. Reading the column with the shared
//! binder and comparing the two lists found three that had silently parted:
//! `copy_logits_bf16` bound an arena operand where the packed run belonged,
//! `router_topk_scaled` read input 1 for a weight, and
//! `mxfp4_qmv_routed_bias` handed a shader seven arguments for six bindings.
//! Deleting the arms did not fix those; it left the plane with only the
//! spelling that was right. See `super::bind`'s header for the three.
//!
//! What remains here is what an arm was NOT: the statement, the fire, the
//! tables, and which kernels this driver crosses at all.

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
    /// The recurrent head count, from [`Geometry::recurrent`].
    ///
    /// [`Geometry::recurrent`]: crate::dispatch::Geometry::recurrent
    pub v_heads: u32,
    /// The recurrent head width. See [`Self::v_heads`].
    pub v_dim: u32,
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
#[derive(Clone, Debug)]
pub enum Asked {
    /// An operand the statement carries, and WHICH of the statement's it is.
    ///
    /// The index is what lets a caller read `Lowered::arg_rows` beside it: an
    /// operand's own row space is not always the launch's rectangle, and the
    /// body asks in its own order rather than the statement's, so the index
    /// cannot be recovered from the ask list's position.
    Operand(usize, Arg),
    /// The packed scalar run.
    Params,
    /// A buffer the FIRE holds — the rope frequencies, the sampling indices.
    /// `driver-metal::lowering::hold::Handles::table` is the same ask, and the
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
    /// A per-layer RECURRENT slab, by the name the kernel knows it as.
    ///
    /// The LAYER is not carried, for [`Asked::Kv`]'s reason: `plan` reads it
    /// off the rectangle, so an arm cannot disagree with the launch it is
    /// planning.
    Slab(&'static str),
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
// `driver-metal::lowering::hold::Facts`.
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
        v_heads: fire.recurrent().0,
        v_dim: fire.recurrent().1,
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
///
/// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
/// measurement now, not a live proof: the crate lost `tests/` and every
/// in-file `mod tests` when the three shader planes moved their numbers to
/// the fire that reads them, and nothing in this workspace re-runs it. What
/// it reported is still why the sentence above says what it says; what is
/// gone is the thing that would notice if it stopped being true.
pub struct Handles<'a> {
    /// The statement's arguments, as the lowering states them.
    args: &'a [Arg],
    /// Which of them are inputs, outputs and weights, in that order.
    ins: Vec<usize>,
    outs: Vec<usize>,
    /// The highest `output(i)` index asked for, plus one.
    ///
    /// A statement's results cannot be counted from the routine's SIGNATURE:
    /// a recurrent slab is `F32sMut` exactly as a result is, and no type tells
    /// them apart. The arm can, and does — a result is what it asks
    /// `output(i)` for and a slab is what it asks `slab(..)` for — so `plan`
    /// runs the arm once over an [`Handles::undivided`] statement purely to
    /// read this, then splits by it. `driver-metal` reached the same shape
    /// after the same defect: a `gdn_prep` statement was read as carrying no
    /// input and two results, so the arm asked for `input(0)` and was told the
    /// statement does not carry one, on every fire of every layer.
    results_asked: usize,
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
    /// rule is `driver-metal::lowering::hold::split`'s: weights are their own
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
    /// `every_launchs_scalars_land_where_its_module_reads_them` fails on it;
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
            results_asked: 0,
        }
    }

    /// Every widthed operand offered as BOTH an input and an output.
    ///
    /// For the counting pass only. `output(i)` and `input(i)` both resolve, so
    /// an arm can be run purely to see how many results it asks for, without
    /// a split that might already be wrong.
    #[must_use]
    pub fn undivided(args: &'a [Arg], scalars: &'a [u32]) -> Self {
        let mut out = Self::build(args, 0, scalars);
        out.outs.clone_from(&out.ins);
        out
    }

    /// How many `output(i)` asks this run made, as a COUNT of results.
    ///
    /// The highest index asked plus one, not the number of asks: an arm may
    /// ask for the same result twice, and it is the arity of the statement's
    /// tail that a split needs.
    #[must_use]
    pub fn asked_results(&self) -> usize {
        self.results_asked
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

    /// Elements per row of the statement's `n`th INPUT.
    ///
    /// THE HALF A HANDLE DOES NOT CARRY, and the reason it is asked for at
    /// all: `Tensor<E>` gives a mark the rectangle as well as the buffer, so
    /// `bind`'s `shaped` asks both questions where it used to ask one. A
    /// backend that answers only the first binds every operand at width ZERO,
    /// and every body reading `x.width` refuses `Empty`.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] when the statement carries no such input.
    pub fn in_width(&self, n: usize) -> Result<i32, Refusal> {
        self.width_at(self.ins.get(n).copied(), "an input operand the arm asked for")
    }

    /// Elements per row of the statement's `n`th RESULT. See [`Self::in_width`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] when the statement declares no such result.
    pub fn out_width(&self, n: usize) -> Result<i32, Refusal> {
        self.width_at(self.outs.get(n).copied(), "a result the arm asked for")
    }

    /// One operand's row width, off the argument the lowering stated.
    ///
    /// A WEIGHT HAS NONE, and answers zero rather than refusing: its extents
    /// are the checkpoint's and the statement does not restate them.
    fn width_at(&self, at: Option<usize>, what: &'static str) -> Result<i32, Refusal> {
        let at = at.ok_or(Refusal::Empty { what })?;
        let width = match self.args.get(at).ok_or(Refusal::Empty { what })? {
            Arg::Arena { width, .. } | Arg::Named { width, .. } => *width,
            // As a weight: neither states a rectangle this launch measures.
            Arg::Weight(_) | Arg::Raised { .. } => 0,
        };
        i32::try_from(width).map_err(|_| Refusal::Wide {
            what: "an operand's row width",
            at: i64::from(width),
            max: i64::from(i32::MAX),
        })
    }

    /// The key of the raised view the statement placed as its `n`th INPUT.
    ///
    /// `In<Struct<T>>` claims an input slot like any other mark, so the view
    /// operand is counted among [`Self::input`]'s — but it is never TAKEN:
    /// no handle is minted for it, because the carrier crosses as an address
    /// rather than a binding. `lowering::views` asks this to learn WHICH
    /// view to build.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] when the statement has no such input, and
    /// [`Refusal::Unstated`] when the operand there is not a raise — a
    /// statement and a signature that disagree about what sits at `n`.
    pub fn raised_key(&self, n: usize) -> Result<String, Refusal> {
        let at = *self.ins.get(n).ok_or(Refusal::Empty {
            what: "an input operand the signature marks as a raised view",
        })?;
        match self.args.get(at) {
            Some(Arg::Raised { key, .. }) => Ok(key.clone()),
            _ => Err(Refusal::Unstated {
                what: "a raised view where the statement placed an ordinary operand",
            }),
        }
    }

    /// The `n`th OUTPUT, as a handle.
    ///
    /// # Errors
    ///
    /// As [`Self::input`].
    pub fn output(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        self.results_asked = self.results_asked.max(n + 1);
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
    /// `driver-metal::lowering::hold::stated` is the same function.
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

    /// The per-layer RECURRENT slab this kernel carries between fires.
    ///
    /// Unlike [`Handles::kv`] this can REFUSE, and the refusal is the point:
    /// this driver allocates no slabs, so a body asking for one declines by
    /// name rather than being handed a null carry. A scan given a null carry
    /// reads zero, writes nothing back, and answers fluently and wrongly --
    /// see [`crate::binding::Resolve::slab`].
    ///
    /// It is a handle rather than an `Option`, so an arm cannot accidentally
    /// bind nothing where a carry belongs.
    pub fn slab(&mut self, which: &'static str) -> ArgValue {
        let handle = u32::try_from(self.taken.len()).expect("a small operand count");
        self.taken.push(Asked::Slab(which));
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
                Asked::Operand(_, arg) => Some(arg.clone()),
                Asked::Params
                | Asked::Table(_)
                | Asked::Kv { .. }
                | Asked::Slab(_)
                | Asked::Unbound => None,
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
        self.taken.push(Asked::Operand(at, self.args[at].clone()));
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

/// One crossed kernel: the entrypoint STEM a plan spells it with, and the arm
/// that feeds it.
///
/// # Why a stem and not a name
///
/// A plan names `silu_mul_bfloat16`, not `silu_mul`. A row could be found from
/// that because `kernels::sig_in` falls back to an axis match, but the fork
/// sits ABOVE the row lookup — that is what lets rows be deleted — so it has
/// to answer from the symbol alone. This is the same answer
/// `driver-vulkan::hold::Crossed` and `driver-metal`'s `crossed` reached.
///
/// Matching by name alone worked for exactly the two DARK families and hid
/// this: `argmax_logits` and `copy_logits_bf16` have no axis, so their symbol
/// IS their stem. `silu_mul` was the first live arm and it was never reached —
/// `every_launchs_scalars_land_where_its_module_reads_them` compared zero
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
pub(crate) static LIVE: &[Crossed] = &[
    Crossed {
        stem: "argmax_logits",
        routine: None,
    },
    Crossed {
        stem: "copy_logits_bf16",
        routine: None,
    },
    Crossed {
        stem: "silu_mul",
        routine: None,
    },
    // CLAIMED, NOT ARMED. `mlp/gated.wgsl`'s strided variant walks rows by a
    // pitch the contiguous body does not take, and it keeps `silu_mul` from
    // claiming its symbols by prefix.
    Crossed {
        stem: "silu_mul_strided",
        routine: None,
    },
    Crossed {
        stem: "geglu_tanh",
        routine: None,
    },
    Crossed {
        stem: "geglu_tanh_strided",
        routine: None,
    },
    Crossed {
        stem: "gptoss_swiglu",
        routine: None,
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
    },
    Crossed {
        stem: "vnorm_single_row",
        routine: None,
    },
    Crossed {
        stem: "rms_strided_row",
        routine: None,
    },
    Crossed {
        stem: "rms_strided_head_row",
        routine: None,
    },
    Crossed {
        stem: "rms_residual",
        routine: None,
    },
    Crossed {
        stem: "rms_residual_scaled",
        routine: None,
    },
    Crossed {
        stem: "gated_rms",
        routine: None,
    },
    Crossed {
        stem: "gated_rms_strided",
        routine: None,
    },
    Crossed {
        stem: "layer_scalar_mul",
        routine: None,
    },
    Crossed {
        stem: "residual_add",
        routine: None,
    },
    Crossed {
        stem: "residual_add_strided",
        routine: None,
    },
    Crossed {
        stem: "add_bias",
        routine: None,
    },
    // layout
    Crossed {
        stem: "embed_gather_4bit",
        routine: None,
    },
    Crossed {
        stem: "embed_gather_mb_4bit",
        routine: None,
    },
    Crossed {
        stem: "embed_gather_scaled_4bit",
        routine: None,
    },
    Crossed {
        stem: "embed_gather_scaled_mb_4bit",
        routine: None,
    },
    Crossed {
        stem: "ple_combine",
        routine: None,
    },
    Crossed {
        stem: "row_gather",
        routine: None,
    },
    // quant
    Crossed {
        stem: "affine_qmm_t",
        routine: Some("qmm_t"),
    },
    Crossed {
        stem: "affine_qmm_t_bias",
        routine: Some("qmm_t_bias"),
    },
    Crossed {
        stem: "affine_qmm_t_bias_fp16_precast",
        routine: Some("qmm_t_bias_fp16_precast"),
    },
    Crossed {
        stem: "affine_qmm_t_fp16_precast",
        routine: Some("qmm_t_fp16_precast"),
    },
    Crossed {
        stem: "affine_qmm_t_residual",
        routine: Some("qmm_t_residual"),
    },
    Crossed {
        stem: "affine_qmm_t_residual_fp16_precast",
        routine: Some("qmm_t_residual_fp16_precast"),
    },
    Crossed {
        stem: "affine_qmm_t_splitk",
        routine: Some("qmm_t_splitk"),
    },
    Crossed {
        stem: "affine_qmm_t_splitk_f32",
        routine: Some("qmm_t_splitk_f32"),
    },
    Crossed {
        stem: "affine_qmm_t_splitk_fp16_precast",
        routine: Some("qmm_t_splitk_fp16_precast"),
    },
    Crossed {
        stem: "affine_qmm_t_splitk_fp16_precast_f32",
        routine: Some("qmm_t_splitk_fp16_precast_f32"),
    },
    Crossed {
        stem: "affine_qmm_t_strided",
        routine: Some("qmm_t_strided"),
    },
    Crossed {
        stem: "affine_qmm_t_strided_residual",
        routine: Some("qmm_t_strided_residual"),
    },
    Crossed {
        stem: "affine_qmm_t_strided_fp16_precast",
        routine: Some("qmm_t_strided_fp16_precast"),
    },
    Crossed {
        stem: "affine_qmm_t_strided_fp16_precast_residual",
        routine: Some("qmm_t_strided_fp16_precast_residual"),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4"),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2"),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2"),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1"),
    },
    Crossed {
        stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
        routine: Some("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4"),
    },
    Crossed {
        stem: "qmm_splitk_reduce",
        routine: None,
    },
    Crossed {
        stem: "qmm_splitk_reduce_f32",
        routine: None,
    },
    Crossed {
        stem: "cast_qmm_input_bfloat16_to_float16",
        routine: None,
    },
    Crossed {
        stem: "cast_qmm_input_strided_bfloat16_to_float16",
        routine: None,
    },
    Crossed {
        stem: "affine_qmv_fast",
        routine: Some("qmv_fast"),
    },
    Crossed {
        stem: "affine_qmv_fast_residual",
        routine: Some("qmv_fast_residual"),
    },
    Crossed {
        stem: "affine_qmv_tail",
        routine: Some("qmv_tail"),
    },
    Crossed {
        stem: "affine_qmv_tail_bias",
        routine: Some("qmv_tail_bias"),
    },
    Crossed {
        stem: "affine_qmv_wide_strided",
        routine: Some("qmv_wide_strided"),
    },
    Crossed {
        stem: "affine_encode_u4_bf16",
        routine: Some("encode_u4_bf16"),
    },
    Crossed {
        stem: "affine_encode_u4_f32",
        routine: Some("encode_u4_f32"),
    },
    Crossed {
        stem: "mxfp4_dequant_bf16",
        routine: None,
    },
    // attn
    Crossed {
        stem: "split_qkv_bf16",
        routine: None,
    },
    Crossed {
        stem: "gate",
        routine: None,
    },
    Crossed {
        stem: "kv_append",
        routine: None,
    },
    Crossed {
        stem: "kv_append_paged",
        routine: None,
    },
    Crossed {
        stem: "logit_softcap",
        routine: None,
    },
    Crossed {
        stem: "q_gate_split",
        routine: None,
    },
    Crossed {
        stem: "sdpa_paged_decode",
        routine: None,
    },
    Crossed {
        stem: "sdpa_paged_decode_sink",
        routine: None,
    },
    Crossed {
        stem: "sdpa_paged_mma",
        routine: None,
    },
    Crossed {
        stem: "sdpa_paged_mma_sink",
        routine: None,
    },
    Crossed {
        stem: "sdpa_paged_tiled",
        routine: None,
    },
    Crossed {
        stem: "sdpa_paged_tiled_sink",
        routine: None,
    },
    Crossed {
        stem: "sdpa_paged_tiled_strided",
        routine: None,
    },
    Crossed {
        stem: "sdpa_vector_decode",
        routine: None,
    },
    Crossed {
        stem: "sdpa_vector_decode_sink",
        routine: None,
    },
    Crossed {
        stem: "sdpa_vector_decode_swa",
        routine: None,
    },
    // ssm
    Crossed {
        stem: "gdn_core",
        routine: None,
    },
    Crossed {
        stem: "gdn_core_recurrent",
        routine: None,
    },
    Crossed {
        stem: "gdn_core_recurrent_prefill",
        routine: None,
    },
    Crossed {
        stem: "gdn_core_recurrent_slotted",
        routine: None,
    },
    Crossed {
        stem: "gdn_core_slotted",
        routine: None,
    },
    Crossed {
        stem: "gdn_prep",
        routine: None,
    },
    Crossed {
        stem: "gdn_prep_prefill",
        routine: None,
    },
    Crossed {
        stem: "gdn_prep_slotted",
        routine: None,
    },
    // moe
    Crossed {
        stem: "router_topk",
        routine: None,
    },
    Crossed {
        stem: "router_topk_scaled",
        routine: None,
    },
    Crossed {
        stem: "route_sort",
        routine: None,
    },
    Crossed {
        stem: "route_gather",
        routine: None,
    },
    Crossed {
        stem: "combine_sorted",
        routine: None,
    },
    Crossed {
        stem: "shared_expert_combine",
        routine: None,
    },
    Crossed {
        stem: "shared_expert_combine_strided",
        routine: None,
    },
    Crossed {
        stem: "affine_qmv_routed",
        routine: Some("qmv_routed"),
    },
    Crossed {
        stem: "affine_qmv_routed_bias",
        routine: Some("qmv_routed_bias"),
    },
    Crossed {
        stem: "mxfp4_qmv_routed_bias",
        routine: None,
    },
    Crossed {
        stem: "affine_qmm_t_routed",
        routine: Some("qmm_t_routed"),
    },
    Crossed {
        stem: "affine_qmm_t_routed_fp16",
        routine: Some("qmm_t_routed_fp16"),
    },
    Crossed {
        stem: "mxfp4_qmm_t_routed_bias",
        routine: None,
    },
    // rope
    Crossed {
        stem: "neox_decode",
        routine: None,
    },
    Crossed {
        stem: "neox_mb",
        routine: None,
    },
    Crossed {
        stem: "neox_freqs_decode",
        routine: None,
    },
    Crossed {
        stem: "neox_freqs_mb",
        routine: None,
    },
    Crossed {
        stem: "neox_prop_decode",
        routine: None,
    },
    Crossed {
        stem: "neox_prop_mb",
        routine: None,
    },
    Crossed {
        stem: "neox_strided",
        routine: None,
    },
];

/// The stem and arm for a symbol, if this backend has crossed AND armed it.
///
/// `None` is the ordinary answer today and means *"take the table path"*.
#[must_use]
pub fn crossed(symbol: &str) -> Option<&'static str> {
    let found = LIVE
        .iter()
        .filter(|c| {
            symbol
                .strip_prefix(c.stem)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
        })
        .max_by_key(|c| c.stem.len())?;
    Some(found.routine.unwrap_or(found.stem))
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
    /// to the ROUTINE whose name is its longest prefix, and `kernels-wgpu`
    /// states those. So the claim is only correct when the claiming stem IS
    /// that owner.
    #[test]
    fn every_entrypoint_is_claimed_by_the_stem_that_owns_it() {
        let owners: Vec<&str> = kernels_wgpu::routines()
            .into_iter()
            .map(|r| r.name)
            .collect();

        let mut claimed = 0u32;
        for point in kernels_wgpu::entrypoints() {
            let Some(stem) = crossed(&point) else {
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
        assert!(crossed("argmax_logits").is_some());
        assert!(crossed("copy_logits_bf16").is_some());
        // EVERY kernel is armed now, so what this test can still falsify is
        // the LOOKUP, not the roster: a symbol resolves to the longest stem
        // that ends on a word boundary, and to nothing otherwise.
        assert!(crossed("affine_qmv_routed_bfloat16_gs_64_b_4").is_some());
        assert!(crossed("sdpa_paged_decode_bfloat16_d_128").is_some());
        // THE AFFINE TRAP, pinned. The quantization scheme is a PREFIX the
        // routine's name never carries, so `qmv_routed` is what the body is
        // called and `affine_qmv_routed_...` is what a plan spells. A lookup
        // matching on the routine name would find nothing here, which is the
        // defect `kernels-metal::kernel_of` shipped for 363 of its 479
        // entrypoints — and which this crate then reproduced twice.
        assert!(crossed("qmv_routed_bfloat16_gs_64_b_4").is_none());
        // Armed, and reached through the SYMBOL a plan actually spells.
        assert!(crossed("silu_mul").is_some());
        assert!(crossed("silu_mul_bfloat16").is_some());
        assert!(crossed("argmax_logits_bfloat16").is_some());
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
            let found = crossed(symbol).expect("armed");
            assert_eq!(
                found, stem,
                "`{symbol}` resolved to `{found}`, not `{stem}`: the lookup \
                 stopped at a prefix instead of taking the longest stem"
            );
        }
        // A stem may not end mid-word.
        assert!(crossed("silu_multiply").is_none());
        // And a name no backend has.
        assert!(crossed("not_a_kernel").is_none());

        // COUNTING BY ROUTINE NAME IS THE TRAP ITSELF. `crossed(r.name)` is
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
    /// Bind one routine by name, the way `routine::plan` does.
    ///
    /// These two tests date from when an arm was a function this module
    /// held, so they called it directly. The arms are gone and the reading
    /// they checked is the signature's; what they check about `Handles` --
    /// that a handle is minted per ask, and that a short statement refuses
    /// rather than indexing past -- is unchanged and worth keeping.
    fn bound(
        name: &str,
        o: &mut Handles<'_>,
        f: Facts,
    ) -> Result<Vec<ArgValue>, Refusal> {
        let routine = kernels_wgpu::routines()
            .into_iter()
            .find(|r| r.name == name)
            .expect("a routine by that name");
        crate::lowering::bind::bind(routine.args, routine.sources, o, f)
    }

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
        let args = bound("argmax_logits", &mut o, f).expect("four operands");
        // The handle SEQUENCE is 0, 1, 2, 3 by construction -- they are
        // minted in order -- so asserting it proves nothing on its own. What
        // it does pin is that the body asked four times and got four
        // distinct slots.
        // BOUND AS `Shaped`, AND THE ROW COUNT IS NOT HERE. Each operand now
        // carries the rectangle the statement gave it -- that is what a mark
        // binds on this plane -- and the `U32(7)` that used to close the run
        // was the row count, which the body asks the fire for.
        assert_eq!(
            args,
            vec![
                ArgValue::Shaped { handle: 0, rows: 7, width: 1 },
                ArgValue::Shaped { handle: 1, rows: 7, width: 1 },
                ArgValue::Shaped { handle: 2, rows: 7, width: 1 },
                ArgValue::Shaped { handle: 3, rows: 7, width: 1 },
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
            bound("argmax_logits", &mut o, f),
            Err(Refusal::Empty { .. })
        ));
    }
}
