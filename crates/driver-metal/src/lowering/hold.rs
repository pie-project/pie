//! What a statement HOLDS, and how a binder reaches into it.
//!
//! A routine's signature says what it takes and in what order, and its
//! `sources` column says where each of those comes from. [`Handles`] is what
//! turns a source into a value: it holds the launch's resolved operands, the
//! statement's scalars, and the driver's own pools, and it mints a handle per
//! ask so that a bound argument list is a list of indices the encoder can
//! fill. [`Facts`] is the fire's geometry, which is what a source names when
//! it names a fact.
//!
//! # This file used to be `arm.rs`
//!
//! It was ninety-one functions, one per routine, each saying in Rust where
//! every argument of that routine comes from. They were the second statement
//! of something a signature could say once, and STAGES 2 through 6 of
//! `.wiki/kilimanjaro4.md` moved all of it into the signatures -- measured
//! off running arms rather than read off them, one class of source at a
//! time, gated at every step against the arm it was copied from.
//!
//! At `99 of 99` the arms were saying nothing the rows did not, and three
//! thousand lines of them went. What is left is the plumbing they all shared,
//! which was never the duplicated part: [`Facts`], [`Handles`], [`Staged`],
//! and [`split`], which cuts a launch's flat operand list into inputs,
//! results and weights.
//!
//! # What a source may not do
//!
//! Compute a grid. Everything here is operand plumbing and nothing else: the
//! numbers a launch is built from reach the routine as arguments and the
//! routine states its own rectangle. A binder that did arithmetic beyond what
//! a `Source` spells would put the second opinion back -- see
//! `.wiki/kernel-x/refactor-bigplan.md` §6, *"the QMM tile is chosen in
//! `model/` and again in `launch.rs`, compared nowhere"*.

use kernels::routine::Refusal;
use kernels_metal::routine::{
    Buf, BufMut,
};

use model_compiler::lower::{Arg, Launch};

use crate::lowering::Geometry;
use crate::lowering::executor::{BoundArg, FireTable, Resolver, Slice};

/// The environment a launch runs in: the fire's geometry, plus what only
/// this launch knows.
///
/// # It used to be a snapshot, and the snapshot was a third statement
///
/// Every number here was a `u32` field, filled once per launch from
/// [`Geometry`] and read by ninety-one arms. That made three places a fact
/// existed -- the geometry that holds it, the field that copies it, and the
/// key that names it -- and adding one meant editing all three plus every
/// synthetic `Facts` in the tests.
///
/// It holds the geometry now and DERIVES the rest, which is Kilimanjaro III
/// F5's cursor. A new fact off the geometry costs one arm in
/// [`bind::geometry`](crate::lowering::bind), and the tests do not change at
/// all: `Geometry` is `Default`, so a synthetic statement names the fields it
/// cares about and inherits the rest.
///
/// What stays as a field is what the geometry cannot answer: the rectangle's
/// row count, the operand widths, the tile and the quantisation point the
/// SYMBOL names, and the layer the statement runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Facts {
    /// The fire's shape, which every derived fact below reads.
    pub geometry: Geometry,
    /// Rows the rectangle covers.
    pub rows: u32,
    /// Elements per row of the operand that sizes the launch — the last
    /// widthed operand, which is the last result.
    pub width: u32,
    /// Elements per row of the first widthed operand, the first input.
    pub in_width: u32,
    /// The GEMM tile the trace chose, as `(bm, bn)`, or `None` for a
    /// statement whose symbol names none.
    ///
    /// The third axis fact that lives only in the SYMBOL, after `group` and
    /// `bits`, and the one with the least excuse: `_bm_64_bn_32` is a
    /// decision the compiler made about this deployment's shapes, and the
    /// table path recovered it by parsing the string back
    /// (`launch::named_tile`). A routine composes the string instead, so the
    /// numbers have to arrive as numbers -- and the spelling check in
    /// `plan_routine` is what holds the two readings together.
    pub tile: Option<(u32, u32)>,
    /// This statement's affine group size and bits per weight.
    ///
    /// NOT [`Geometry::group`] and [`Geometry::bits`]: a router gate
    /// projection on a checkpoint whose gates arrived at their own width
    /// states a different point from the fire's, which `facts_of` resolves.
    pub point: (u32, u32),
    /// The layer this statement runs in, for the state lookups.
    ///
    /// A rolled trace states a span and an unrolled one states a layer; the
    /// span's first is the answer either way, because a rolled statement
    /// reaches a plan once per layer WITH it.
    pub layer: u16,
    /// Requests the fire serves.
    ///
    /// Not an extent of any rectangle: it is the number `RowGatherParams`
    /// carries as its second field, and the one statement that needs it takes
    /// it as `InPacked` rather than as a lane count.
    pub requests: u32,
}

impl Facts {
    /// Query heads.
    #[must_use]
    pub const fn q_heads(&self) -> u32 {
        self.geometry.q_heads
    }

    /// THIS LAYER'S key/value head count, not the fire's.
    ///
    /// gemma-4 is the one stack that states two, and its full-attention
    /// layers are twice as wide per head over a quarter the KV heads -- so a
    /// fire-wide pair is wrong for whichever kind of layer it is not, and a
    /// routine composing `sdpa_paged_decode_bfloat16_d_<width>` from it
    /// spells a symbol the trace did not state.
    #[must_use]
    pub const fn kv_heads(&self) -> u32 {
        self.geometry.heads_at(self.layer as u32).0
    }

    /// Elements per head, at this layer. See [`Self::kv_heads`].
    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        self.geometry.heads_at(self.layer as u32).1
    }

    /// A linear-attention layer's value heads, which is a different number
    /// from [`Self::kv_heads`] on a hybrid and the same on everything else.
    #[must_use]
    pub const fn v_heads(&self) -> u32 {
        self.geometry.recurrent_at().0
    }

    /// See [`Self::v_heads`].
    #[must_use]
    pub const fn v_dim(&self) -> u32 {
        self.geometry.recurrent_at().1
    }

    /// Channels a rope rotates.
    #[must_use]
    pub const fn rotary_dims(&self) -> u32 {
        self.geometry.rotary_dims
    }

    /// Experts a mixture holds.
    #[must_use]
    pub const fn n_experts(&self) -> u32 {
        self.geometry.n_experts
    }

    /// Experts a token is routed to.
    #[must_use]
    pub const fn experts_per_token(&self) -> u32 {
        self.geometry.experts_per_token
    }

    /// This statement's affine group size. See [`Self::point`].
    #[must_use]
    pub const fn group(&self) -> u32 {
        self.point.0
    }

    /// This statement's bits per weight. See [`Self::point`].
    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.point.1
    }
}

/// The statement's operands and scalars, and the handles an arm builds from
/// them.
///
/// An arm names a value -- `o.input(0)`, `o.weight(2)`, `o.kv_keys()` -- and
/// gets back a handle it can pass to a routine. The handle is an index into
/// [`Handles::bound`], which is what the planner resolves a body's
/// [`ArgValue::Buffer`] through, and the indices are assigned HERE rather than
/// fixed by the trace's order: a fire table and a KV page are not operands and
/// have no place in it.
pub struct Handles<'a> {
    /// Every handle an arm has asked for, in the order it asked.
    bound: Vec<BoundArg>,
    /// The statement's widthed operands that are INPUTS, as indices into the
    /// launch's bound arguments.
    ins: &'a [usize],
    /// The same, for its RESULTS.
    outs: &'a [usize],
    /// The same, for the weights it names.
    weights: &'a [usize],
    /// What the launch bound, in the trace's order.
    args: &'a [BoundArg],
    /// The statement's own scalar run.
    params: &'a [Option<u32>],
    /// What answers for the things the STATEMENT does not carry: a weight by
    /// name, a fire's position table, a layer's KV pages.
    ///
    /// A `dyn` because [`Arm`] is a plain function pointer and cannot be
    /// generic. The cost is one virtual call per table, once per LOWERING --
    /// `lowering::cached` plans a fire's rectangles once and replays them --
    /// so it is not on any encode path.
    resolver: &'a mut dyn Resolver,
    /// The handle [`Handles::params_block`] minted, if an arm asked for one.
    block: Option<u32>,
    /// The words that block holds. See [`Handles::params_block`].
    words: Vec<u32>,
    /// The highest result index an arm asked for, plus one. Nothing in
    /// dispatch reads it; see [`Handles::asked_results`].
    asked: usize,
}

/// A launch's staged scalars, and which handle stands for them.
///
/// The planner needs both: the words to write, and the one handle that must
/// become a pointer INTO them rather than an address of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Staged<'a> {
    /// The handle an arm minted for the packed block, if it minted one.
    pub block: Option<u32>,
    /// The words the block holds, in the struct's own field order.
    pub words: &'a [u32],
}

/// A handle that addresses nothing.
///
/// What an unresolvable fire table or an absent KV pool answers, which is the
/// same answer the table path's `reorder` gives and for the same reason: the
/// pool is state a fire may legitimately not have, and a statement that asks
/// for one it does not have is a trace mismatch the binder reports elsewhere.
const NOTHING: BoundArg = BoundArg {
    slice: Slice {
        address: 0,
        bytes: 0,
    },
    width: 0,
};

impl<'a> Handles<'a> {
    /// The handles for one launch.
    #[must_use]
    pub fn new(
        args: &'a [BoundArg],
        ins: &'a [usize],
        outs: &'a [usize],
        weights: &'a [usize],
        params: &'a [Option<u32>],
        resolver: &'a mut dyn Resolver,
    ) -> Self {
        Self {
            bound: Vec::new(),
            ins,
            outs,
            weights,
            args,
            params,
            resolver,
            block: None,
            words: Vec::new(),
            asked: 0,
        }
    }

    /// How many of the statement's widthed operands this arm treated as
    /// RESULTS: the highest index it passed to [`Handles::output`] or
    /// [`Handles::output_read`], plus one.
    ///
    /// THIS IS NO LONGER HOW DISPATCH LEARNS THE NUMBER. It was: the number
    /// `split` needs could not be read off a signature, because
    /// `attn::kv_append_paged` declares two `BufMut` and both are the KV
    /// POOL, which is state and not a traced result -- counting writable
    /// arguments took that statement's two inputs for results and left the
    /// arm nothing to read. So the arm ran twice, once purely to be counted.
    ///
    /// A signature can answer it now. `OutSlot<0, BufMut>` is a result and
    /// `Env<BufMut>` is not, and `dispatch` counts the `Side::Declared`
    /// entries of a row it already holds.
    ///
    /// What survives is this, as the CHECK. The row's count and this number
    /// are not the same shape -- one is a count of parameters, the other a
    /// highest index plus one -- and they agree only while each result index
    /// appears in a signature exactly once.
    /// `routine::tests::every_arm_binds_the_slot_its_signature_states` runs
    /// every arm and asserts they do. Delete this and that assertion has
    /// nothing to hold the row against; the cost it carries is one `usize`
    /// and one `max` per ask, which is not the cost that was worth removing.
    #[must_use]
    pub fn asked_results(&self) -> usize {
        self.asked
    }

    /// The staged scalars and the handle that points at them.
    #[must_use]
    pub fn staged(&self) -> Staged<'_> {
        Staged {
            block: self.block,
            words: &self.words,
        }
    }

    /// What the planner resolves a body's handles through.
    #[must_use]
    pub fn bound(&self) -> &[BoundArg] {
        &self.bound
    }

    /// Take a handle for `slice`, whatever it is.
    fn take(&mut self, bound: BoundArg) -> u32 {
        let at = u32::try_from(self.bound.len()).unwrap_or(u32::MAX);
        self.bound.push(bound);
        at
    }

    fn pick(&mut self, at: Option<&usize>, what: &'static str) -> Result<u32, Refusal> {
        let at = *at.ok_or(Refusal::Absent { what })?;
        let bound = *self.args.get(at).ok_or(Refusal::Absent { what })?;
        Ok(self.take(bound))
    }

    /// The statement's `i`-th input.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement has fewer, which is an arm
    /// asking for an operand its trace does not carry.
    pub fn input(&mut self, i: usize) -> Result<Buf, Refusal> {
        let at = self.ins.get(i).copied();
        self.pick(at.as_ref(), "an input the statement does not carry")
            .map(Buf)
    }

    /// The statement's `i`-th result.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement has fewer.
    pub fn output(&mut self, i: usize) -> Result<BufMut, Refusal> {
        self.asked = self.asked.max(i + 1);
        let at = self.outs.get(i).copied();
        self.pick(at.as_ref(), "a result the statement does not carry")
            .map(BufMut)
    }

    /// The statement's `i`-th result, read rather than written.
    ///
    /// For the routines that take their own output as an input -- a residual
    /// added in place, a gate applied to the tensor it gates. The aliasing is
    /// stated on the routine as `in_place`, and this is the arm honouring it.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement has fewer.
    pub fn output_read(&mut self, i: usize) -> Result<Buf, Refusal> {
        self.asked = self.asked.max(i + 1);
        let at = self.outs.get(i).copied();
        self.pick(at.as_ref(), "a result the statement does not carry")
            .map(Buf)
    }

    /// The `i`-th weight the statement names.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement names fewer.
    pub fn weight(&mut self, i: usize) -> Result<Buf, Refusal> {
        let at = self.weights.get(i).copied();
        self.pick(at.as_ref(), "a weight the statement does not name")
            .map(Buf)
    }

    /// One of the FIRE's tables: the token ids, the positions, the sampled
    /// rows, the KV page directory.
    ///
    /// Not the statement's, which is why a row had to name these with a
    /// `Source` of their own and why an arm asks for them by name rather than
    /// by index. They are this fire's data -- what is being run -- where an
    /// operand is the model's structure.
    ///
    /// A table the fire does not have answers [`NOTHING`], which is the
    /// answer the table path's `reorder` gives and for the same reason: a
    /// decode has no sampling indices and a statement that asks for them in
    /// one is a trace mismatch the binder reports elsewhere.
    pub fn table(&mut self, which: FireTable) -> u32 {
        let slice = self.resolver.fire(which);
        let bound = slice.map_or(NOTHING, |slice| BoundArg { slice, width: 0 });
        self.take(bound)
    }

    /// A layer's KV cache, keys or values.
    pub fn kv(&mut self, layer: u16, values: bool) -> u32 {
        let slice = self.resolver.kv(layer, values);
        let bound = slice.map_or(NOTHING, |slice| BoundArg { slice, width: 0 });
        self.take(bound)
    }

    /// This layer's entry in a per-layer GDN slab.
    ///
    /// The recurrent state and the conv window: state, like the KV cache, so
    /// no traced value stands for it.
    ///
    /// Unlike [`Handles::kv`] this REFUSES when the driver has none rather
    /// than binding [`NOTHING`], and the difference is not fussiness. A
    /// missing scale is a legitimate absence -- an unquantized tensor has
    /// none -- so binding nothing is the honest answer. A recurrent state is
    /// not: a scan handed a null carry reads zero, writes nothing back, and
    /// returns a fluent result that is wrong in a way no output check
    /// catches.
    ///
    /// The refusal earns its keep on a rig rather than in production. This
    /// read "this backend allocates no slabs today, so every `ssm` arm
    /// declines here", which stopped being true when the recurrent pool
    /// landed and left the sentence describing a driver that no longer
    /// existed. What it caught instead was `device_real_weights`'s MLX
    /// gates, which passed `slabs: None` and so refused every hybrid
    /// checkpoint at its first `gdn_*` statement -- loudly, by name, with
    /// the op that owed the slab. A rig hole that announces itself is the
    /// whole point of refusing rather than binding nothing.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] when the driver holds no slab for this layer.
    pub fn slab(&mut self, layer: u16, which: &'static str) -> Result<u32, Refusal> {
        let slice = self.resolver.slab(layer, which).ok_or(Refusal::Unstated {
            what: "a GDN slab: this driver allocates none",
        })?;
        Ok(self.take(BoundArg { slice, width: 0 }))
    }

    /// A number the driver keeps for the KV pool -- a stride, a page size.
    ///
    /// Not a handle: these reach a kernel as scalars, and a routine that needs
    /// one takes it as an argument.
    pub fn pooled(&mut self, which: FireTable) -> Option<u32> {
        self.resolver.pool(which)
    }

    /// A handle for a slice the driver resolved itself -- a fire table, a KV
    /// page range.
    pub fn state(&mut self, slice: Option<Slice>) -> Buf {
        let bound = slice.map_or(NOTHING, |slice| BoundArg { slice, width: 0 });
        Buf(self.take(bound))
    }

    /// The same, written.
    pub fn state_mut(&mut self, slice: Option<Slice>) -> BufMut {
        let bound = slice.map_or(NOTHING, |slice| BoundArg { slice, width: 0 });
        BufMut(self.take(bound))
    }

    /// The statement's scalars, as the one packed struct a kernel reads them
    /// through.
    ///
    /// Seventeen of this backend's rows spell a parameter block: `constant
    /// GegluParams&`, `constant RouterParams&` -- a buffer argument whose
    /// bytes are the statement's own scalar run, in order, and every one of
    /// them reads it from `Source::Slot(Kind::Param, 0)`, meaning the whole run. Metal
    /// declares them as buffers because MSL has no push-constant, which is
    /// why they occupy an argument slot at all.
    ///
    /// The address is not known here: the encoder allocates one staging
    /// region per fire and writes each dispatch's run into it. So this mints
    /// a handle that STANDS FOR the run, and the planner turns it into a
    /// packed [`ParamSlot`](crate::lowering::dispatch::ParamSlot) rather than
    /// an address.
    ///
    /// A statement carrying no scalars still gets one word. The shader
    /// dereferences the pointer whether or not it reads a field -- and an
    /// argument slot left unbound holds whatever address the previous
    /// dispatch put there, which is a wild read rather than a dead one.
    pub fn params_block(&mut self) -> Buf {
        self.words = self.params.iter().map(|p| p.unwrap_or(0)).collect();
        if self.words.is_empty() {
            self.words.push(0);
        }
        let at = self.take(NOTHING);
        self.block = Some(at);
        Buf(at)
    }

    /// The statement's `i`-th scalar, as the signed number a kernel reads.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement's run is shorter, or the slot is
    /// empty -- a trace that stated a scalar's position and not its value.
    pub fn param(&self, i: usize) -> Result<i32, Refusal> {
        let held = self
            .params
            .get(i)
            .copied()
            .flatten()
            .ok_or(Refusal::Absent {
                what: "a scalar the statement does not carry",
            })?;
        Ok(held.cast_signed())
    }

    /// The statement's `i`-th scalar, as the float a kernel reads.
    ///
    /// The trace carries every scalar as a `u32` and the BITS are the value,
    /// so this reinterprets rather than converting -- `1.0f32` rides as
    /// `0x3f80_0000`, and a conversion would hand the kernel 1065353216.0.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement's run is shorter.
    pub fn param_f32(&self, i: usize) -> Result<f32, Refusal> {
        let held = self
            .params
            .get(i)
            .copied()
            .flatten()
            .ok_or(Refusal::Absent {
                what: "a scalar the statement does not carry",
            })?;
        Ok(f32::from_bits(held))
    }
}

/// The widthed operands of a launch split into inputs and results, and the
/// weights it names.
///
/// The trace concatenates inputs, then results, then weights, and the binder
/// keeps that order. `results` is how many of the widthed ones are results,
/// which the signature states: a result wears `OutSlot<N, _>` and the KV pool
/// wears `Env<_>`, so `Side::Declared` separates a traced result from state
/// the driver holds even though both are `BufMut`.
#[must_use]
pub fn split(args: &[Arg], results: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let (widthed, weights) = undivided(args);
    let results = results.min(widthed.len());
    let (ins, outs) = widthed.split_at(widthed.len() - results);
    (ins.to_vec(), outs.to_vec(), weights)
}

/// The same two lists, with the cut between inputs and results NOT yet made.
///
/// [`split`] makes the cut; this is the part before it, which is just the
/// binder's order restated -- everything with a width in trace order, then
/// the weights. It was public because a probe run used to be handed it: every
/// widthed operand answered as both an input and a result so that an arm
/// could be run purely to be counted. Nothing outside this module asks for it
/// now.
fn undivided(args: &[Arg]) -> (Vec<usize>, Vec<usize>) {
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
    (widthed, weights)
}

/// The environment a launch runs in, from the fire's geometry and the
/// rectangle.
///
/// No `sig`. The four `*_param` columns are gone from this reading: a routine
/// that needs a number the fire cannot answer for every layer takes it as an
/// argument, which is what `head_param`, `heads_param`, `grid_param` and
/// `rows_param` were indices standing in for.
#[must_use]
pub fn facts(
    launch: &Launch,
    geometry: Geometry,
    requests: u32,
    width: u32,
    in_width: u32,
    tile: Option<(u32, u32)>,
    point: (u32, u32),
) -> Facts {
    Facts {
        geometry,
        rows: launch.rows.end - launch.rows.start,
        width,
        in_width,
        tile,
        // THIS STATEMENT'S point, resolved by `facts_of` -- the fire's for
        // every statement but a router gate projection on a checkpoint whose
        // gates arrived at their own width.
        point,
        // `layers.start` because a rectangle is one layer's: `plan_launch` is
        // reached per layer and the range is the peel's, not a span of
        // differing shapes.
        layer: launch.layers.start,
        requests,
    }
}
