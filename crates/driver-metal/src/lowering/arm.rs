//! What a statement supplies a routine, per routine.
//!
//! `.wiki/kernel-x/refactor-bigplan.md` §7 Stage 4, and §6's *"the `operands`
//! column, as code"*. A routine's signature says what it takes and in what
//! order; it does not say where any of it comes from. That is a fact about the
//! STATEMENT — the third input, the second weight, the fire's position table,
//! the layer's KV pages — and an arm is where it is written.
//!
//! # Why this is a function per routine and not a table
//!
//! It was a table. `KernelSig::operands` is that table, and the whole of
//! `.wiki/kernel-x/refactor-bigplan.md` §8a is the class of defect it admits:
//! a row states a `Source` for every slot, the slots are positional, and a
//! row that names one too few or one too many binds every argument after it
//! one place off. Nothing checks a row against the kernel it describes,
//! because a row is data and the kernel is a shader in another language.
//!
//! An arm is a CALL. It hands its values to a `pub fn` whose parameters have
//! types and names, so an argument list one short does not compile. That is
//! the trade the refactor makes: the same information, moved from a place
//! where only a device could disagree with it to a place where the compiler
//! does.
//!
//! # What an arm may not do
//!
//! Compute a grid. Every arm here is operand plumbing and nothing else: the
//! numbers a launch is built from reach the routine as arguments and the
//! routine states its own rectangle. An arm that did arithmetic would put the
//! second opinion back — see `.wiki/kernel-x/refactor-bigplan.md` §6, *"the
//! QMM tile is chosen in `model/` and again in `launch.rs`, compared
//! nowhere"*.

use kernels::routine::Refusal;
use kernels_metal::routine::{
    ArgValue, Bind, Buf, BufMut, Env, F32s, F32sMut, I32s, InPacked, U8s, U32s, Usize,
};

use model_compiler::lower::{Arg, Launch};

use crate::lowering::Geometry;
use crate::lowering::executor::{BoundArg, FireTable, Resolver, Slice};

/// The extents a routine takes from its environment rather than its
/// statement.
///
/// These are the numbers `Dims` carried for the launch rules, minus the four
/// `*_param` overrides: a routine that needs a per-statement head width takes
/// it as an ARGUMENT, off the scalar run, because its signature can name it.
/// The columns existed to let a row point at a scalar it could not name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Facts {
    /// Rows the rectangle covers.
    pub rows: u32,
    /// Elements per row of the operand that sizes the launch — the last
    /// widthed operand, which is the last result.
    pub width: u32,
    /// Elements per row of the first widthed operand, the first input.
    pub in_width: u32,
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Channels a rope rotates.
    pub rotary_dims: u32,
    /// Experts a mixture holds.
    pub n_experts: u32,
    /// Experts a token is routed to.
    pub experts_per_token: u32,
    /// The deployment's affine group size and bits per weight. See
    /// [`Geometry::group`](crate::lowering::dispatch::Geometry::group).
    pub group: u32,
    /// See [`Facts::group`].
    pub bits: u32,
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
    /// it as [`InPacked`] rather than as a lane count.
    pub requests: u32,
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
        }
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
    /// catches. This backend allocates no slabs today, so every `ssm` arm
    /// declines here, which is what keeps the family's dispatch honestly
    /// dark instead of quietly broken.
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
    /// them reads it from `Source::Param(0)`, meaning the whole run. Metal
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
/// which the ROUTINE knows -- it is the count of `BufMut` in its signature --
/// where the table path read it off the row's `Out` sources.
#[must_use]
pub fn split(args: &[Arg], results: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
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
    (ins.to_vec(), outs.to_vec(), weights)
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
) -> Facts {
    Facts {
        rows: launch.rows.end - launch.rows.start,
        tile,
        layer: launch.layers.start,
        requests,
        width,
        in_width,
        q_heads: geometry.q_heads,
        kv_heads: geometry.kv_heads,
        head_dim: geometry.head_dim,
        rotary_dims: geometry.rotary_dims,
        n_experts: geometry.n_experts,
        experts_per_token: geometry.experts_per_token,
        group: geometry.group,
        bits: geometry.bits,
    }
}

/// One routine's operand plumbing: the values it takes, in its own order.
///
/// `Env` arguments are appended by the arm from [`Facts`], because they are
/// not values a body receives through [`ArgValue`] at all -- they reach it as
/// typed parameters and the erased body reconstructs them positionally.
pub type Arm = fn(&mut Handles<'_>, Facts) -> Result<Vec<ArgValue>, Refusal>;

/// `sample::argmax_logits`.
///
/// Four buffers in the shader's own order -- logits, the token it writes, the
/// packed params, the EOS flag -- and the row count off the rectangle. The
/// trace states two inputs and two results, and the interleaving is the
/// kernel's: this is the arm that would have to be got wrong, and it is
/// checked by the routine's parameter list rather than by a device.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand the statement does not carry.
pub fn argmax_logits(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let logits = o.input(0)?;
    let next_token = o.output(0)?;
    let params = o.input(1)?;
    let eos_flag = o.output(1)?;
    let _ = Env(f.rows);
    Ok(vec![
        logits.v(),
        next_token.v(),
        params.v(),
        eos_flag.v(),
        f.rows.v(),
    ])
}

/// `mlp::geglu_tanh`, `mlp::geglu_tanh_strided` and `mlp::gptoss_swiglu`.
///
/// One arm for three routines because the STATEMENT is the same shape for all
/// three -- two inputs, one result, one parameter block -- and the arms are
/// about statements. What differs between them is the activation, which is
/// the body's business, and the block's fields, which are the trace's: the
/// driver forwards the scalar run without knowing that gemma's third word is
/// a gate pitch and gpt-oss's is a clamp.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand the statement does not carry.
fn gated(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let gate = o.input(0)?;
    let up = o.input(1)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        gate.v(),
        up.v(),
        out.v(),
        params.v(),
        f.width.cast_signed().v(),
        f.rows.cast_signed().v(),
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

/// `mlp::silu_mul`.
///
/// The same statement without the block: this kernel is the only one in
/// `mlp/gated.metal` that reads no parameter struct, because it needs no
/// scalar the grid does not give it.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand the statement does not carry.
pub fn silu_mul(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let gate = o.input(0)?;
    let up = o.input(1)?;
    let out = o.output(0)?;
    Ok(vec![
        gate.v(),
        up.v(),
        out.v(),
        f.width.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `layout::embed_gather_mb_4bit`, and the three siblings it shares a
/// statement shape with.
///
/// Three weights -- the packed table, its scales, its biases -- the FIRE's
/// token ids, and one result. `hidden` is the statement's first scalar and
/// `embed_scale` its second, which is gemma multiplying its embeddings by
/// `sqrt(hidden)`: a number the text states and not one a kernel could know.
///
/// `scaled` and `mb` are the caller's, because they are the routine's: a
/// single-row gather over a four-row fire writes one row and reports success,
/// which is measured in `dsl::embed_gather`'s own note.
///
/// # Errors
///
/// [`Refusal::Absent`] for a weight the statement does not name or a scalar
/// its run does not carry.
fn embed_gather(o: &mut Handles<'_>, f: Facts, scaled: bool) -> Result<Vec<ArgValue>, Refusal> {
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    let id = I32s(o.table(FireTable::TokenIds));
    let out = o.output(0)?;
    let hidden = o.param(0)?;
    let mut args = vec![w.v(), scales.v(), biases.v(), id.v(), out.v(), hidden.v()];
    if scaled {
        args.push(o.param_f32(1)?.v());
    }
    args.push(f.rows.cast_signed().v());
    args.push(f.group.cast_signed().v());
    args.push(f.bits.cast_signed().v());
    Ok(args)
}

/// `layout::embed_gather_4bit` -- the single-row form.
///
/// It takes no row count: its grid is one row by construction. So the `rows`
/// its `mb` sibling carries is dropped here rather than passed as a one.
///
/// # Errors
///
/// See [`embed_gather`].
pub fn embed_gather_4bit(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mut args = embed_gather(o, f, false)?;
    args.remove(6);
    Ok(args)
}

/// `layout::embed_gather_mb_4bit`. See [`embed_gather`].
///
/// # Errors
///
/// See [`embed_gather`].
pub fn embed_gather_mb_4bit(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    embed_gather(o, f, false)
}

/// `layout::embed_gather_scaled_4bit`. See [`embed_gather`] and
/// [`embed_gather_4bit`].
///
/// # Errors
///
/// See [`embed_gather`].
pub fn embed_gather_scaled_4bit(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mut args = embed_gather(o, f, true)?;
    args.remove(7);
    Ok(args)
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
    embed_gather(o, f, true)
}

/// `layout::ple_combine`.
///
/// gemma's PLE join, over the whole `[n_layers, ple_dim]` block. The scale is
/// the JOIN's -- two streams averaged in the root-mean-square sense -- so it
/// rides the packed block with the element count rather than arriving here.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand the statement does not carry.
pub fn ple_combine(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let proj = o.input(0)?;
    let token = o.input(1)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        proj.v(),
        token.v(),
        out.v(),
        params.v(),
        f.width.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `layout::row_gather`: one distribution per REQUEST out of one row per
/// TOKEN.
///
/// `count` is the request count and it is [`InPacked`] -- the second FIELD of
/// `RowGatherParams`, not an operand. There is no buffer 4 in that shader, so
/// the word is appended to the block's run and takes no argument slot; see
/// `lay_out`'s `InPacked` arm.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand the statement does not carry.
pub fn row_gather(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let input = o.input(0)?;
    let out = o.output(0)?;
    let rows = U32s(o.table(FireTable::SamplingIndices));
    let params = o.params_block();
    let count = InPacked(f.requests);
    Ok(vec![
        input.v(),
        out.v(),
        rows.v(),
        params.v(),
        count.v(),
        f.width.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// A scalar the STATEMENT may state and the fire can otherwise answer.
///
/// `kernel!` said this with `grid_param`/`head_param`: a row names a param
/// index and the grid reads it, falling back to the fire's geometry when the
/// statement does not carry one. The fallback is not a nicety -- gemma-4
/// rotates a quarter of each full-attention head and all of each sliding one,
/// so no fire-wide `rotary_dims` is right for both layers, while every
/// single-shape deployment states nothing and means the fire's number.
///
/// Zero is treated as absent, exactly as `dims_of`'s `.filter(|n| *n > 0)`
/// does: a grid axis of zero launches nothing, which is a silent no-op.
fn stated(o: &Handles<'_>, i: usize, fire: u32) -> i32 {
    o.param(i)
        .ok()
        .filter(|n| *n > 0)
        .unwrap_or(fire.cast_signed())
}

/// The rotation every neox form shares: the tensor, the positions, and the
/// three axes its grid is built from.
///
/// In place -- `x` is buffer 0 and it is both the input and the result --
/// which is why there is one `BufMut` and no `Buf` beside it, and why a
/// statement carrying q and k must be two launches rather than one.
///
/// `head_dim` is read from the statement and falls back to the fire, and the
/// fallback matters more here than [`stated`]'s doc suggests: the table path
/// passed the raw param to the SHADER while giving the GRID the fallback, so
/// a statement missing it dispatched a kernel told `head_dim = 0` over a grid
/// sized by the fire. One number now, or [`rope_grid`]'s refusal.
///
/// [`rope_grid`]: kernels_metal::rope
fn neox(o: &mut Handles<'_>, f: Facts, head_dim_at: usize) -> Result<Rotation, Refusal> {
    let x = o.output(0)?;
    let position = I32s(o.table(FireTable::Positions));
    Ok(Rotation {
        x,
        position,
        scale: o.param_f32(0)?,
        head_dim: stated(o, head_dim_at, f.head_dim),
        rotary: stated(o, 3, f.rotary_dims),
        width: f.width.cast_signed(),
        rows: f.rows.cast_signed(),
    })
}

/// What every neox arm has in common. See [`neox`].
struct Rotation {
    x: BufMut,
    position: I32s,
    scale: f32,
    head_dim: i32,
    rotary: i32,
    width: i32,
    rows: i32,
}

/// `rope::neox_decode`: the geometric ladder, one row.
///
/// # Errors
///
/// [`Refusal::Absent`] for the tensor or for `scale`/`base`, which are the
/// two scalars no deployment leaves out.
pub fn neox_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let base = o.param_f32(1)?;
    let r = neox(o, f, 2)?;
    Ok(vec![
        r.x.v(),
        r.position.v(),
        r.scale.v(),
        base.v(),
        r.head_dim.v(),
        r.rotary.v(),
        r.width.v(),
    ])
}

/// `rope::neox_mb`: [`neox_decode`] over one row per token.
///
/// # Errors
///
/// See [`neox_decode`].
pub fn neox_mb(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let base = o.param_f32(1)?;
    let r = neox(o, f, 2)?;
    Ok(vec![
        r.x.v(),
        r.position.v(),
        r.scale.v(),
        base.v(),
        r.head_dim.v(),
        r.rotary.v(),
        r.width.v(),
        r.rows.v(),
    ])
}

/// `rope::neox_prop_decode`: gemma's ladder over a proportional slice.
///
/// # Errors
///
/// See [`neox_decode`].
pub fn neox_prop_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    neox_decode(o, f)
}

/// `rope::neox_prop_mb`: gemma's PREFILL rotation.
///
/// Its row states no operands, so the table path has never dispatched it --
/// a gemma prefill refused here. The binding order is [`neox_mb`]'s because
/// the shader's is: `rope_neox_prop_mb` declares `x`, `position`, `scale`,
/// `base`, `head_dim`.
///
/// # Errors
///
/// See [`neox_decode`].
pub fn neox_prop_mb(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    neox_mb(o, f)
}

/// `rope::neox_freqs_decode`: the ladder READ rather than raised.
///
/// llama-3's piecewise rescaling and YaRN's are neither of them a base, so
/// the frequencies arrive as a buffer the fire stages. `base` is absent
/// rather than ignored -- this entrypoint has no slot for one -- and
/// `head_dim` sits one earlier in the statement because of it.
///
/// # Errors
///
/// [`Refusal::Absent`] for the tensor, `scale`, or `mscale`.
pub fn neox_freqs_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mscale = o.param_f32(2)?;
    let r = neox(o, f, 1)?;
    let inv_freq = Buf(o.table(FireTable::RopeFrequencies));
    Ok(vec![
        r.x.v(),
        r.position.v(),
        r.scale.v(),
        inv_freq.v(),
        r.head_dim.v(),
        mscale.v(),
        r.rotary.v(),
        r.width.v(),
    ])
}

/// `rope::neox_freqs_mb`: [`neox_freqs_decode`] over one row per token, and
/// the rotation a llama-3.1, llama-3.2 or YaRN PREFILL takes.
///
/// # Errors
///
/// See [`neox_freqs_decode`].
pub fn neox_freqs_mb(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mscale = o.param_f32(2)?;
    let r = neox(o, f, 1)?;
    let inv_freq = Buf(o.table(FireTable::RopeFrequencies));
    Ok(vec![
        r.x.v(),
        r.position.v(),
        r.scale.v(),
        inv_freq.v(),
        r.head_dim.v(),
        mscale.v(),
        r.rotary.v(),
        r.width.v(),
        r.rows.v(),
    ])
}

/// `rope::neox_strided`: the ladder over rows that do not tile.
///
/// A packed QKV projection is where that arises -- q and k share a buffer, so
/// rotating q means striding over k -- and `row_pitch` is the stride. Its row
/// states no operands either, so like [`neox_prop_mb`] nothing has dispatched
/// it; unlike `neox_prop_mb` the pitch is a number no other form carries, and
/// it is read from param 4 because params 0..3 are this family's fixed
/// preamble. A statement that does not carry one is REFUSED rather than
/// given the row width, since a pitch equal to the width is precisely the
/// case this kernel is not for.
///
/// # Errors
///
/// [`Refusal::Absent`] for the tensor, `scale`, `base` or `row_pitch`.
pub fn neox_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let base = o.param_f32(1)?;
    let row_pitch = o.param(4)?;
    let r = neox(o, f, 2)?;
    Ok(vec![
        r.x.v(),
        r.position.v(),
        r.scale.v(),
        base.v(),
        r.head_dim.v(),
        row_pitch.v(),
        r.rotary.v(),
        r.width.v(),
        r.rows.v(),
    ])
}

/// The four buffers every RMS form shares, and the axis it reduces over.
///
/// `axis` is `RmsParams.axis_size` -- word 1 of the staged block, which is
/// exactly what `grid_param = Some(1)` read. It is not the row width: a
/// QK-norm packs `width / axis` reductions into each row, and a grid sized
/// per ROW normalizes head 0 and leaves the rest as the projection wrote
/// them. Fully written, never reported.
///
/// Two of the rows below stated no `grid_param` and got `axis = width` from
/// the fallback. They are the fused-residual forms, where the norm spans its
/// row and the two numbers coincide -- so reading the field is the same
/// answer with a reason.
fn rms(o: &mut Handles<'_>, f: Facts, weighted: bool) -> Result<Rms, Refusal> {
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
        axis: stated(o, 1, f.width),
        rows: f.rows.cast_signed(),
    })
}

/// What every RMS arm has in common. See [`rms`].
struct Rms {
    x: Buf,
    w: Option<Buf>,
    out: BufMut,
    params: Buf,
    width: i32,
    axis: i32,
    rows: i32,
}

impl Rms {
    /// `x, w, out, params` -- the shader's order, which is not the trace's.
    fn head(&self) -> Vec<ArgValue> {
        let mut v = vec![self.x.v()];
        if let Some(w) = self.w {
            v.push(w.v());
        }
        v.push(self.out.v());
        v.push(self.params.v());
        v
    }
}

/// `norm::rms_single_row`: one threadgroup per axis.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand or weight the statement does not carry.
pub fn rms_single_row(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let r = rms(o, f, true)?;
    let mut v = r.head();
    let tail: [ArgValue; 3] = [r.width.v(), r.axis.v(), r.rows.v()];
    v.extend(tail);
    Ok(v)
}

/// `norm::vnorm_single_row`: gemma's value norm, which has no gain.
///
/// The absent weight is the whole difference from [`rms_single_row`], and it
/// renumbers the buffers -- `out` is 1 here and 2 there.
///
/// # Errors
///
/// See [`rms_single_row`].
pub fn vnorm_single_row(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let r = rms(o, f, false)?;
    let mut v = r.head();
    let tail: [ArgValue; 3] = [r.width.v(), r.axis.v(), r.rows.v()];
    v.extend(tail);
    Ok(v)
}

/// `norm::rms_residual`: the block residual folded into the epilogue.
///
/// The residual is buffer 4, AFTER the params struct, because folding must
/// not renumber the four every form shares.
///
/// # Errors
///
/// See [`rms_single_row`].
pub fn rms_residual(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let r = rms(o, f, true)?;
    let mut v = r.head();
    let tail: [ArgValue; 4] = [residual.v(), r.width.v(), r.axis.v(), r.rows.v()];
    v.extend(tail);
    Ok(v)
}

/// `norm::rms_residual_scaled`: [`rms_residual`] with a per-layer gain.
///
/// # Errors
///
/// See [`rms_single_row`].
pub fn rms_residual_scaled(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let scale = o.input(2)?;
    let r = rms(o, f, true)?;
    let mut v = r.head();
    let tail: [ArgValue; 5] = [residual.v(), scale.v(), r.width.v(), r.axis.v(), r.rows.v()];
    v.extend(tail);
    Ok(v)
}

/// `norm::rms_strided_row`: one norm per row, rows a pitch apart.
///
/// Its row was bare, so this is the first statement of its bindings. The
/// pitch is the fire's row width and the axis is the block's -- the shader's
/// own comment says the rows are "the widest tensor in the layout" apart,
/// which is what `sizing_width` measures, while the reduction is `axis_size`
/// wide. A packed QKV projection is the case: the pitch spans q, k and v and
/// the norm spans one of them.
///
/// # Errors
///
/// See [`rms_single_row`].
pub fn rms_strided_row(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let r = rms(o, f, true)?;
    let mut v = r.head();
    let tail: [ArgValue; 3] = [r.width.v(), r.axis.v(), r.rows.v()];
    v.extend(tail);
    Ok(v)
}

/// `norm::rms_strided_head_row`: the per-head q/k norms over a whole prompt.
///
/// `heads` is `width / axis`, which is how many reductions a row holds --
/// the same quotient `rms_grid` folds into its flat lane count, given its own
/// grid axis here because the rows are not contiguous.
///
/// # Errors
///
/// See [`rms_single_row`], plus [`Refusal::Empty`] for an axis of zero, which
/// would make the head count a division by it.
pub fn rms_strided_head_row(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let r = rms(o, f, true)?;
    if r.axis <= 0 {
        return Err(Refusal::Empty { what: "axis" });
    }
    let heads = r.width / r.axis;
    let mut v = r.head();
    let tail: [ArgValue; 4] = [r.width.v(), r.axis.v(), heads.v(), r.rows.v()];
    v.extend(tail);
    Ok(v)
}

/// `norm::gated_rms`: the gated per-head norm, `heads` of them per row.
///
/// A bare row, and the axes come from the fire because a value head's shape
/// is the pool's: `LaunchRule::GatedRms` said the same pair, `kv_heads` by
/// `head_dim`. Its own grid had no row axis at all, which is the defect the
/// routine's header records -- one dispatch normalized the first token of a
/// prompt and no other.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand or weight the statement does not carry.
pub fn gated_rms(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let x = o.input(0)?;
    let z = o.input(1)?;
    let w = o.weight(0)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        x.v(),
        z.v(),
        w.v(),
        out.v(),
        params.v(),
        f.head_dim.cast_signed().v(),
        f.kv_heads.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `norm::gated_rms_strided`: [`gated_rms`] over rows a pitch apart, which is
/// the prefill form.
///
/// # Errors
///
/// See [`gated_rms`].
pub fn gated_rms_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mut v = gated_rms(o, f)?;
    // Before the three `Env`, after the five buffers.
    v.insert(5, f.width.cast_signed().v());
    Ok(v)
}

/// `norm::layer_scalar_mul`: gemma's per-layer scale.
///
/// `params` is bound and not read -- the entrypoint declares it and the body
/// bounds itself with the grid instead.
///
/// # Errors
///
/// [`Refusal::Absent`] for the tensor or the scalar weight.
pub fn layer_scalar_mul(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let x = o.input(0)?;
    let scalar = o.weight(0)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        x.v(),
        scalar.v(),
        out.v(),
        params.v(),
        f.width.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `norm::residual_add`: `out = x + residual`, elementwise.
///
/// # Errors
///
/// [`Refusal::Absent`] for either input or the result.
pub fn residual_add(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let x = o.input(0)?;
    let residual = o.input(1)?;
    let out = o.output(0)?;
    Ok(vec![
        x.v(),
        residual.v(),
        out.v(),
        f.width.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `norm::residual_add_strided`: [`residual_add`] over rows a pitch apart.
///
/// A bare row, and unlike the strided norms there is no params struct to hold
/// the pitch and no fire-wide number that is it -- a pitch equal to the row
/// width is the case [`residual_add`] serves. So it is the statement's first
/// scalar or a refusal, as in `rope::neox_strided`.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand or for the pitch.
pub fn residual_add_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_pitch = o.param(0)?;
    let mut v = residual_add(o, f)?;
    v.insert(3, row_pitch.v());
    Ok(v)
}

/// `norm::add_bias`: the Qwen-2 family's attention biases, added in place.
///
/// The width is an ARGUMENT and not an `Env` -- the kernel indexes with
/// `tid.y * width + tid.x` and the grid only carries the extent -- and it is
/// the result's own width, which the trace holds and no text repeats.
///
/// # Errors
///
/// [`Refusal::Absent`] for the tensor or the bias weight.
pub fn add_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let out = o.output(0)?;
    let bias = o.weight(0)?;
    Ok(vec![
        out.v(),
        bias.v(),
        f.width.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

// ---------------------------------------------------------------- quant ---
//
// Thirty-one routines and FIVE rows that carry operands: `qmm_t`,
// `qmm_t_residual`, `qmv_fast`, `qmv_fast_residual` and `qmv_tail`. The other
// twenty-six are bare -- the precast pair-ups, every split-K form, every
// strided form, both transcoders -- so nothing has dispatched them and there
// is no prior behaviour for their arms to preserve. They are written from the
// signatures, which is the only statement of their bindings there has ever
// been.
//
// The shape is the same throughout: three weights, then the activation, then
// the result, then `k` and `n` as the two scalars every statement carries.
// What varies is what rides after -- a residual, a bias, a stride, a split.

/// The quantized weight triple, which every routine in this family opens
/// with.
///
/// `w`, `scales`, `biases` -- the codes and the two affine terms that decode
/// them. Weights 0, 1 and 2 in every row of this family that states any.
fn codec(o: &mut Handles<'_>) -> Result<[ArgValue; 3], Refusal> {
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    Ok([w.v(), scales.v(), biases.v()])
}

/// The GEMM tile this statement's symbol named, or a refusal.
///
/// There is no fallback and there cannot be one. The tile is not a property
/// of the deployment that a fire-wide number could stand in for -- it is a
/// choice about THIS multiply's shape, and the two substitutions tried
/// before were both measured wrong against a real checkpoint: handing the
/// GEMM's symbol the matvec's grid made a prefill entirely NaN, and rounding
/// an axis up made it finite and wrong.
fn tile(f: Facts) -> Result<(i32, i32), Refusal> {
    let (bm, bn) = f.tile.ok_or(Refusal::Unstated {
        what: "a GEMM tile: the symbol names none and no fire-wide number is one",
    })?;
    Ok((bm.cast_signed(), bn.cast_signed()))
}

/// `k` and `n`: the contraction and the output width, which every multiply
/// here is told.
fn kn(o: &Handles<'_>) -> Result<(i32, i32), Refusal> {
    Ok((o.param(0)?, o.param(1)?))
}

/// `quant::qmm_t`: the tiled GEMM against affine-quantized weights.
///
/// # Errors
///
/// [`Refusal::Absent`] for a weight, operand or scalar the statement does not
/// carry, and [`Refusal::Unstated`] for a symbol that names no tile.
pub fn qmm_t(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (k, n) = kn(o)?;
    let (bm, bn) = tile(f)?;
    let mut v = c.to_vec();
    let tail: [ArgValue; 9] = [
        x.v(),
        y.v(),
        k.v(),
        n.v(),
        f.group.cast_signed().v(),
        f.bits.cast_signed().v(),
        bm.v(),
        bn.v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `quant::qmm_t_bias`: [`qmm_t`] with the projection's bias added.
///
/// The bias is weight 3 -- after the codec's three, which is the order a
/// statement lists them in and the order `split` preserves.
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let bias = o.weight(3)?;
    let mut v = qmm_t(o, f)?;
    // After `y`, before `k`: the shader binds it at buffer 5.
    v.insert(5, bias.v());
    Ok(v)
}

/// `quant::qmm_t_residual`: [`qmm_t`] with the block residual folded in.
///
/// The residual lands AFTER `k` and `n` rather than beside the activation,
/// which is the fold's convention throughout this tree: a conditional
/// binding comes last so that folding does not renumber what every form
/// shares.
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_residual(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let mut v = qmm_t(o, f)?;
    v.insert(7, residual.v());
    Ok(v)
}

/// The precast GEMM's opening: the codec, the result, and the half-precision
/// activation.
///
/// There is no `x`. The activation was cast to `float16` by
/// [`cast_qmm_input_bfloat16_to_float16`] into a buffer of its own, and THAT
/// is what the statement's first input is -- which is why `y` comes before it
/// here and after it everywhere else.
fn precast(o: &mut Handles<'_>) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let half_in = o.input(0)?;
    let y = o.output(0)?;
    let mut v = c.to_vec();
    let tail: [ArgValue; 2] = [y.v(), half_in.v()];
    v.extend(tail);
    Ok(v)
}

/// `quant::qmm_t_fp16_precast`: [`qmm_t`] over an activation already cast to
/// `float16`.
///
/// Only group 64 at four bits: the precast family is compiled for one codec
/// point, which is why its axes state `GROUP_64, BITS_4` and its signature
/// takes neither.
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_fp16_precast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let (bm, bn) = tile(f)?;
    let mut v = precast(o)?;
    let tail: [ArgValue; 5] = [k.v(), n.v(), bm.v(), bn.v(), f.rows.cast_signed().v()];
    v.extend(tail);
    Ok(v)
}

/// `quant::qmm_t_bias_fp16_precast`: [`qmm_t_fp16_precast`] with a bias.
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_bias_fp16_precast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let bias = o.weight(3)?;
    let mut v = qmm_t_fp16_precast(o, f)?;
    // Between `y` and `half_in`: buffer 4.
    v.insert(4, bias.v());
    Ok(v)
}

/// `quant::qmm_t_residual_fp16_precast`: [`qmm_t_fp16_precast`] with the
/// block residual folded in.
///
/// # Errors
///
/// See [`qmm_t`].
pub fn qmm_t_residual_fp16_precast(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let mut v = qmm_t_fp16_precast(o, f)?;
    v.insert(4, residual.v());
    Ok(v)
}

/// The four numbers a split-K multiply is told about its own partition.
///
/// None of them is a shape the fire knows: how far apart the output's rows
/// are, how much of `k` one partition covers, how far apart the partials are,
/// and how many there are. The driver that chooses to split states all four,
/// and a statement that does not carry them is refused -- a zero split is a
/// dispatch that reduces nothing.
fn split_k(o: &Handles<'_>) -> Result<[ArgValue; 4], Refusal> {
    Ok([
        o.param(2)?.v(),
        o.param(3)?.v(),
        o.param(4)?.v(),
        o.param(5)?.v(),
    ])
}

/// `quant::qmm_t_splitk`: the GEMM with the contraction cut into partitions.
///
/// # Errors
///
/// See [`qmm_t`], plus [`Refusal::Absent`] for any of the four partition
/// numbers.
pub fn qmm_t_splitk(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let x = o.input(0)?;
    let out = o.output(0)?;
    let (k, n) = kn(o)?;
    let split = split_k(o)?;
    let (bm, _) = tile(f)?;
    let mut v = c.to_vec();
    let tail: [ArgValue; 4] = [x.v(), out.v(), k.v(), n.v()];
    v.extend(tail);
    v.extend(split);
    let tail: [ArgValue; 4] = [
        f.group.cast_signed().v(),
        f.bits.cast_signed().v(),
        bm.v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `quant::qmm_t_splitk_f32`: [`qmm_t_splitk`] accumulating into `float32`.
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
/// See [`qmm_t_splitk`].
pub fn qmm_t_splitk_fp16_precast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let split = split_k(o)?;
    let (bm, _) = tile(f)?;
    let mut v = precast(o)?;
    let tail: [ArgValue; 2] = [k.v(), n.v()];
    v.extend(tail);
    v.extend(split);
    let tail: [ArgValue; 2] = [bm.v(), f.rows.cast_signed().v()];
    v.extend(tail);
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

/// `quant::qmm_t_strided`: the GEMM over an activation whose rows do not
/// tile.
///
/// A packed projection is the case, as with every other `_strided` form here:
/// the pitch spans more than the row the multiply reads. It is the
/// statement's third scalar or a refusal, for the reason
/// `rope::neox_strided` gives.
///
/// # Errors
///
/// See [`qmm_t`], plus [`Refusal::Absent`] for the pitch.
pub fn qmm_t_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_stride = o.param(2)?;
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (k, n) = kn(o)?;
    let (bm, _) = tile(f)?;
    let mut v = c.to_vec();
    let tail: [ArgValue; 9] = [
        x.v(),
        y.v(),
        k.v(),
        n.v(),
        row_stride.v(),
        f.group.cast_signed().v(),
        f.bits.cast_signed().v(),
        bm.v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `quant::qmm_t_strided_residual`: [`qmm_t_strided`] with the residual
/// folded in.
///
/// The residual is buffer 5, BEFORE `k` -- unlike [`qmm_t_residual`], whose
/// is buffer 7. The two orders are the shaders' and there is no rule that
/// derives one from the other, which is exactly the kind of fact a
/// signature exists to state.
///
/// # Errors
///
/// See [`qmm_t_strided`].
pub fn qmm_t_strided_residual(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let mut v = qmm_t_strided(o, f)?;
    v.insert(5, residual.v());
    Ok(v)
}

/// `quant::qmm_t_strided_fp16_precast`: [`qmm_t_strided`] over a precast
/// activation.
///
/// # Errors
///
/// See [`qmm_t_strided`].
pub fn qmm_t_strided_fp16_precast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_stride = o.param(2)?;
    let (k, n) = kn(o)?;
    let (bm, _) = tile(f)?;
    let mut v = precast(o)?;
    let tail: [ArgValue; 5] = [
        k.v(),
        n.v(),
        row_stride.v(),
        bm.v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `quant::qmm_t_strided_fp16_precast_residual`:
/// [`qmm_t_strided_fp16_precast`] with the residual folded in.
///
/// # Errors
///
/// See [`qmm_t_strided`].
pub fn qmm_t_strided_fp16_precast_residual(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let mut v = qmm_t_strided_fp16_precast(o, f)?;
    v.insert(4, residual.v());
    Ok(v)
}

/// `quant::qmm_splitk_reduce`: the sum of a split multiply's partials.
///
/// The second half of a split-K pair and the only routine here that reads no
/// weights: the partials are the statement's input and the result is the
/// sum over them.
///
/// # Errors
///
/// [`Refusal::Absent`] for the partials, the result, or any of the five
/// scalars.
pub fn qmm_splitk_reduce(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let partial = o.input(0)?;
    let y = o.output(0)?;
    let (k, n) = kn(o)?;
    Ok(vec![
        y.v(),
        partial.v(),
        k.v(),
        n.v(),
        o.param(2)?.v(),
        o.param(3)?.v(),
        o.param(4)?.v(),
        f.rows.cast_signed().v(),
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
/// `count` is how many elements to cast, and it is the statement's rather
/// than the fire's: a cast covers exactly the activation the multiply after
/// it will read, which need not be the whole rectangle.
///
/// # Errors
///
/// [`Refusal::Absent`] for the tensor, the result, or any of the four
/// scalars.
pub fn cast_qmm_input_bfloat16_to_float16(
    o: &mut Handles<'_>,
    _f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let cast_in = o.input(0)?;
    let half_out = o.output(0)?;
    let (k, n) = kn(o)?;
    Ok(vec![
        cast_in.v(),
        half_out.v(),
        k.v(),
        n.v(),
        o.param(2)?.v(),
        o.param(3)?.v(),
    ])
}

/// `quant::cast_qmm_input_strided_bfloat16_to_float16`:
/// [`cast_qmm_input_bfloat16_to_float16`] over rows a pitch apart.
///
/// # Errors
///
/// See [`cast_qmm_input_bfloat16_to_float16`].
pub fn cast_qmm_input_strided_bfloat16_to_float16(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let mut v = cast_qmm_input_bfloat16_to_float16(o, f)?;
    v.push(f.rows.cast_signed().v());
    Ok(v)
}

/// `quant::qmv_fast`: the matvec, one row at a time.
///
/// `in_vec_size` and `out_vec_size` are the statement's two scalars and the
/// grid is `vecs` by the output width -- where `vecs` is the row count, which
/// is what made the multi-row form a generalisation of the single-row one
/// rather than a different kernel.
///
/// # Errors
///
/// [`Refusal::Absent`] for a weight, operand or scalar the statement does not
/// carry.
pub fn qmv_fast(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (in_vec, out_vec) = kn(o)?;
    let mut v = c.to_vec();
    let tail: [ArgValue; 7] = [
        x.v(),
        y.v(),
        in_vec.v(),
        out_vec.v(),
        f.group.cast_signed().v(),
        f.bits.cast_signed().v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `quant::qmv_fast_residual`: [`qmv_fast`] with the residual folded in.
///
/// # Errors
///
/// See [`qmv_fast`].
pub fn qmv_fast_residual(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let residual = o.input(1)?;
    let mut v = qmv_fast(o, f)?;
    v.insert(7, residual.v());
    Ok(v)
}

/// `quant::qmv_tail`: the matvec for an output width the fast form's
/// decomposition does not divide.
///
/// Group 64 only, so the signature takes `bits` and not `group`.
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
    let tail: [ArgValue; 6] = [
        x.v(),
        y.v(),
        in_vec.v(),
        out_vec.v(),
        f.bits.cast_signed().v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `quant::qmv_tail_bias`: [`qmv_tail`] with the projection's bias.
///
/// # Errors
///
/// See [`qmv_fast`].
pub fn qmv_tail_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let bias = o.weight(3)?;
    let mut v = qmv_tail(o, f)?;
    v.insert(5, bias.v());
    Ok(v)
}

/// `quant::qmv_wide_strided`: the matvec over a wide contraction with rows a
/// pitch apart.
///
/// `m` is an ARGUMENT here rather than an `Env`: the kernel is told the row
/// count because its grid covers quarters of it, so the threads of a partial
/// last quarter need the number to know they are past the end.
///
/// # Errors
///
/// See [`qmv_fast`], plus [`Refusal::Absent`] for the pitch.
pub fn qmv_wide_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_stride = o.param(2)?;
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (in_vec, out_vec) = kn(o)?;
    let mut v = c.to_vec();
    let tail: [ArgValue; 7] = [
        x.v(),
        y.v(),
        in_vec.v(),
        out_vec.v(),
        row_stride.v(),
        f.rows.cast_signed().v(),
        f.bits.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// The five GEMMs whose tile is compiled into the symbol rather than chosen.
///
/// `qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2` and its four siblings are
/// one codec point and one tile each, so they take neither `group` nor `bits`
/// nor a tile -- the name is a constant and the body states it. The arm is
/// [`qmm_t`]'s without the four axis facts.
fn qmm_fixed(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let c = codec(o)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let (k, n) = kn(o)?;
    let mut v = c.to_vec();
    let tail: [ArgValue; 5] = [x.v(), y.v(), k.v(), n.v(), f.rows.cast_signed().v()];
    v.extend(tail);
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

/// `quant::encode_u4_bf16`: the affine encoder, which writes a codec rather
/// than reading one.
///
/// Three results and not one: the codes, the scales and the biases are what
/// the GEMMs above take as weights 0, 1 and 2, so this is where a
/// runtime-quantized tensor comes from. `groups` is how many affine groups
/// the input holds, which is its extent over the group size.
///
/// # Errors
///
/// [`Refusal::Absent`] for the input or any of the three results, and
/// [`Refusal::Empty`] for a group size of zero.
pub fn encode_u4_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let input = o.input(0)?;
    let codes = o.output(0)?;
    let scales = o.output(1)?;
    let biases = o.output(2)?;
    let params = o.params_block();
    if f.group == 0 {
        return Err(Refusal::Empty { what: "the group" });
    }
    let groups = f.width.saturating_mul(f.rows) / f.group;
    Ok(vec![
        input.v(),
        codes.v(),
        scales.v(),
        biases.v(),
        params.v(),
        groups.cast_signed().v(),
    ])
}

/// `quant::encode_u4_f32`: [`encode_u4_bf16`] over `float32` input.
///
/// # Errors
///
/// See [`encode_u4_bf16`].
pub fn encode_u4_f32(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    encode_u4_bf16(o, f)
}

/// `quant::mxfp4_dequant_bf16`: gpt-oss's MXFP4 blocks expanded to
/// `bfloat16`.
///
/// The payload and the shared exponents are two separate tensors, which is
/// what makes this a different shape from the affine codec above: a block's
/// scale is an exponent byte rather than a scale-and-bias pair. Blocks of
/// 32, so the count is the extent over the group.
///
/// # Errors
///
/// [`Refusal::Absent`] for either input or the result, and [`Refusal::Empty`]
/// for a group size of zero.
pub fn mxfp4_dequant_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let payload = o.input(0)?;
    let exponents = o.input(1)?;
    let out = o.output(0)?;
    let params = o.params_block();
    if f.group == 0 {
        return Err(Refusal::Empty { what: "the group" });
    }
    let blocks = f.width.saturating_mul(f.rows) / f.group;
    Ok(vec![
        payload.v(),
        exponents.v(),
        out.v(),
        params.v(),
        blocks.cast_signed().v(),
    ])
}

// ----------------------------------------------------------------- attn ---
//
// The family where the KV cache stops being an operand. Every SDPA form
// reads `k_pages` and `v_pages` from the driver's own pool -- no traced value
// stands for them, which is why `Handles::kv` exists and why the layer had to
// become a fact. Everything else the paged forms read is a FIRE table: which
// pages a request holds, where each token sits, whether a mask is armed.
//
// Nine of the sixteen are one shape with three folds on it -- a sink, a
// window, a pitch -- so `Sdpa` states the shape once and the arms differ by
// what they insert.

/// The paged attention shape: what every `sdpa_paged_*` routine is told.
///
/// The order is the shader's argument table and it is the same in all six
/// paged forms. `sinks` is the only weight any of them names, and the
/// non-sink forms still bind it -- to nothing -- because the shader declares
/// the pointer either way and an unbound slot holds the last dispatch's
/// address.
struct Sdpa {
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: i32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
}

impl Sdpa {
    /// The arguments up to and including `sinks`, which every paged form
    /// opens with.
    fn head(&self) -> Vec<ArgValue> {
        vec![
            self.queries.v(),
            self.k_pages.v(),
            self.v_pages.v(),
            self.out.v(),
            self.gqa_factor.v(),
            self.position_ids.v(),
            self.req_of_token.v(),
            self.kv_page_indices.v(),
            self.kv_page_indptr.v(),
            self.page_size.v(),
            self.n_kv_heads.v(),
            self.scale.v(),
            self.attention_mask.v(),
            self.attention_mask_stride.cast_unsigned().v(),
            self.attention_mask_enabled.v(),
            self.window.v(),
            self.sinks.v(),
        ]
    }
}

/// What a paged attention statement carries, read once.
///
/// `n_kv_heads` is the statement's second scalar and NOT `f.kv_heads`: gemma-3
/// carries four 512-wide KV heads in its full-attention layers and sixteen
/// 256-wide ones in its sliding layers, so no fire-wide number is right for
/// both, and the head count is per-statement for exactly that reason.
///
/// The page size is the pool's -- the driver chose it -- so it comes through
/// `pooled` rather than from the statement.
fn paged(o: &mut Handles<'_>, f: Facts, sinks: Buf) -> Result<Sdpa, Refusal> {
    let gqa_factor = o.param(0)?;
    let n_kv_heads = o.param(1)?;
    let scale = o.param_f32(2)?;
    let attention_mask_stride = o.param(3).unwrap_or(0);
    let window = o.param(4).unwrap_or(-1);
    let queries = o.input(0)?;
    let k_pages = o.kv(f.layer, false);
    let v_pages = o.kv(f.layer, true);
    let out = o.output(0)?;
    let position_ids = o.table(FireTable::Positions);
    let req_of_token = o.table(FireTable::RequestOfToken);
    let kv_page_indices = o.table(FireTable::KvPageIndices);
    let kv_page_indptr = o.table(FireTable::KvPageIndptr);
    let attention_mask = o.table(FireTable::AttentionMask);
    let attention_mask_enabled = o.table(FireTable::AttentionMaskEnabled);
    let page_size = o
        .pooled(FireTable::KvPageSize)
        .ok_or(Refusal::Unstated {
            what: "the KV page size: the pool has none",
        })?
        .cast_signed();
    Ok(Sdpa {
        queries,
        k_pages: Buf(k_pages),
        v_pages: Buf(v_pages),
        out,
        gqa_factor,
        position_ids: I32s(position_ids),
        req_of_token: I32s(req_of_token),
        kv_page_indices: U32s(kv_page_indices),
        kv_page_indptr: U32s(kv_page_indptr),
        page_size,
        n_kv_heads,
        scale,
        attention_mask: U8s(attention_mask),
        attention_mask_stride,
        attention_mask_enabled: U8s(attention_mask_enabled),
        window,
        sinks,
    })
}

/// The same, with the attention sinks the gpt-oss forms read.
///
/// Weight 0, which is the only weight this family names.
fn paged_sink(o: &mut Handles<'_>, f: Facts) -> Result<Sdpa, Refusal> {
    let sinks = o.weight(0)?;
    paged(o, f, sinks)
}

/// A paged form that binds no sinks still binds the slot.
///
/// [`NOTHING`] rather than a stale address: the shader dereferences the
/// pointer whether or not the branch that reads it is taken.
fn no_sinks(o: &mut Handles<'_>) -> Buf {
    o.state(None)
}

/// `attn::sdpa_paged_decode`: one query row against a request's pages.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand or scalar the statement does not carry,
/// and [`Refusal::Unstated`] when the pool states no page size.
pub fn sdpa_paged_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let sinks = no_sinks(o);
    let s = paged(o, f, sinks)?;
    let mut v = s.head();
    let tail: [ArgValue; 3] = [
        f.head_dim.cast_signed().v(),
        f.q_heads.cast_signed().v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `attn::sdpa_paged_decode_sink`: [`sdpa_paged_decode`] with gpt-oss's
/// per-head sink logit.
///
/// # Errors
///
/// See [`sdpa_paged_decode`], plus [`Refusal::Absent`] for the sinks.
pub fn sdpa_paged_decode_sink(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = paged_sink(o, f)?;
    let mut v = s.head();
    let tail: [ArgValue; 3] = [
        f.head_dim.cast_signed().v(),
        f.q_heads.cast_signed().v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `attn::sdpa_paged_tiled`: prefill attention, a tile of query rows at a
/// time.
///
/// `n_rows` is an ARGUMENT and not just the grid's extent: the grid opens
/// whole tiles, so the threads of a partial last tile are past the end and
/// only this number tells them so.
///
/// # Errors
///
/// See [`sdpa_paged_decode`].
pub fn sdpa_paged_tiled(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let sinks = no_sinks(o);
    let s = paged(o, f, sinks)?;
    let mut v = s.head();
    let tail: [ArgValue; 3] = [
        f.rows.cast_signed().v(),
        f.head_dim.cast_signed().v(),
        f.q_heads.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `attn::sdpa_paged_tiled_sink`: [`sdpa_paged_tiled`] with sinks.
///
/// # Errors
///
/// See [`sdpa_paged_decode`].
pub fn sdpa_paged_tiled_sink(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = paged_sink(o, f)?;
    let mut v = s.head();
    let tail: [ArgValue; 3] = [
        f.rows.cast_signed().v(),
        f.head_dim.cast_signed().v(),
        f.q_heads.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `attn::sdpa_paged_tiled_strided`: [`sdpa_paged_tiled`] over queries and
/// results a pitch apart.
///
/// The packed-QKV case: `q` is a window into a `[rows, q+k+v]` block, so its
/// pitch is wider than what attention reads. Both pitches are the
/// statement's, and a statement that does not carry them is refused rather
/// than defaulted to the width -- a pitch equal to the width is the case this
/// kernel is not for.
///
/// # Errors
///
/// See [`sdpa_paged_decode`], plus [`Refusal::Absent`] for either pitch.
pub fn sdpa_paged_tiled_strided(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let q_row_pitch = o.param(5)?;
    let o_row_pitch = o.param(6)?;
    let sinks = no_sinks(o);
    let s = paged(o, f, sinks)?;
    let mut v = s.head();
    let tail: [ArgValue; 5] = [
        f.rows.cast_signed().v(),
        q_row_pitch.v(),
        o_row_pitch.v(),
        f.head_dim.cast_signed().v(),
        f.q_heads.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `attn::sdpa_paged_mma`: prefill attention through the simdgroup matrix
/// units.
///
/// The same arguments as [`sdpa_paged_tiled`] and a different grid -- which is
/// the whole difference, and the reason both exist: the MMA form wants a
/// query tile deep enough to fill a matrix and the tiled form does not.
///
/// # Errors
///
/// See [`sdpa_paged_decode`].
pub fn sdpa_paged_mma(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    sdpa_paged_tiled(o, f)
}

/// `attn::sdpa_paged_mma_sink`: [`sdpa_paged_mma`] with sinks.
///
/// # Errors
///
/// See [`sdpa_paged_decode`].
pub fn sdpa_paged_mma_sink(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    sdpa_paged_tiled_sink(o, f)
}

/// The contiguous-cache decode shape: the KV strides the pool keeps.
///
/// Not paged. This is the flat `[layer, head, seq, dim]` cache, so the kernel
/// walks it with two strides instead of a page directory -- and the strides
/// are the POOL's, resolved from the driver rather than stated, because a
/// stride is the pool's shape and a trace that guessed one would be right for
/// a deployment and silently wrong for the next.
struct Vector {
    queries: Buf,
    keys: Buf,
    values: Buf,
    out: BufMut,
    gqa_factor: i32,
    n: i32,
    k_head_stride: Usize,
    k_seq_stride: Usize,
    scale: f32,
}

impl Vector {
    /// Through `scale`, which is where the three forms start to differ.
    ///
    /// The V strides repeat the K ones: the two halves of the cache have the
    /// same shape, and the row said so by naming `KvHeadStride` twice.
    fn head(&self) -> Vec<ArgValue> {
        vec![
            self.queries.v(),
            self.keys.v(),
            self.values.v(),
            self.out.v(),
            self.gqa_factor.v(),
            self.n.v(),
            self.k_head_stride.v(),
            self.k_seq_stride.v(),
            self.k_head_stride.v(),
            self.k_seq_stride.v(),
            self.scale.v(),
        ]
    }
}

/// What a contiguous-cache decode statement carries.
fn vector(o: &mut Handles<'_>, f: Facts) -> Result<Vector, Refusal> {
    let gqa_factor = o.param(0)?;
    let n = o.param(1)?;
    let scale = o.param_f32(2)?;
    let queries = o.input(0)?;
    let keys = o.kv(f.layer, false);
    let values = o.kv(f.layer, true);
    let out = o.output(0)?;
    let k_head_stride = o.pooled(FireTable::KvHeadStride).ok_or(Refusal::Unstated {
        what: "the KV head stride: the pool has none",
    })?;
    let k_seq_stride = o.pooled(FireTable::KvSeqStride).ok_or(Refusal::Unstated {
        what: "the KV sequence stride: the pool has none",
    })?;
    Ok(Vector {
        queries,
        keys: Buf(keys),
        values: Buf(values),
        out,
        gqa_factor,
        n,
        k_head_stride: Usize(u64::from(k_head_stride)),
        k_seq_stride: Usize(u64::from(k_seq_stride)),
        scale,
    })
}

/// `attn::sdpa_vector_decode`: one query row against a contiguous cache.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand or scalar the statement does not carry,
/// and [`Refusal::Unstated`] when the pool states no strides.
pub fn sdpa_vector_decode(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let s = vector(o, f)?;
    let mut v = s.head();
    let tail: [ArgValue; 3] = [
        f.head_dim.cast_signed().v(),
        f.q_heads.cast_signed().v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `attn::sdpa_vector_decode_swa`: [`sdpa_vector_decode`] over a sliding
/// window, with pitched rows.
///
/// # Errors
///
/// See [`sdpa_vector_decode`], plus [`Refusal::Absent`] for the window or
/// either pitch.
pub fn sdpa_vector_decode_swa(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let window = o.param(3)?;
    let q_row_stride = o.param(4)?;
    let o_row_stride = o.param(5)?;
    let s = vector(o, f)?;
    let mut v = s.head();
    let tail: [ArgValue; 6] = [
        window.v(),
        q_row_stride.v(),
        o_row_stride.v(),
        f.head_dim.cast_signed().v(),
        f.q_heads.cast_signed().v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `attn::sdpa_vector_decode_sink`: [`sdpa_vector_decode_swa`] with sinks.
///
/// # Errors
///
/// See [`sdpa_vector_decode`].
pub fn sdpa_vector_decode_sink(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let window = o.param(3)?;
    let q_row_stride = o.param(4)?;
    let o_row_stride = o.param(5)?;
    let sinks = o.weight(0)?;
    let s = vector(o, f)?;
    let mut v = s.head();
    let tail: [ArgValue; 7] = [
        window.v(),
        q_row_stride.v(),
        o_row_stride.v(),
        sinks.v(),
        f.head_dim.cast_signed().v(),
        f.q_heads.cast_signed().v(),
        f.rows.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `attn::kv_append`: the new keys and values written into a contiguous
/// cache.
///
/// The one routine here that WRITES the cache, which is why `k_cache` and
/// `v_cache` are `BufMut` and why this dispatch has to precede the attention
/// that reads them. `pos` is the fire's position table: where in each
/// request's sequence this token lands.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand or the head width, and
/// [`Refusal::Unstated`] when the pool states no strides.
pub fn kv_append(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let head_dim = stated(o, 0, f.head_dim);
    let k_new = o.input(0)?;
    let v_new = o.input(1)?;
    let k_cache = BufMut(o.kv(f.layer, false));
    let v_cache = BufMut(o.kv(f.layer, true));
    let pos = I32s(o.table(FireTable::Positions));
    let k_head_stride = o.pooled(FireTable::KvHeadStride).ok_or(Refusal::Unstated {
        what: "the KV head stride: the pool has none",
    })?;
    let k_seq_stride = o.pooled(FireTable::KvSeqStride).ok_or(Refusal::Unstated {
        what: "the KV sequence stride: the pool has none",
    })?;
    Ok(vec![
        k_new.v(),
        v_new.v(),
        k_cache.v(),
        v_cache.v(),
        pos.v(),
        head_dim.v(),
        Usize(u64::from(k_head_stride)).v(),
        Usize(u64::from(k_seq_stride)).v(),
        f.kv_heads.cast_signed().v(),
    ])
}

/// `attn::kv_append_paged`: the same, into the paged pool.
///
/// Six of this routine's arguments are named `ring_*` and every one of them
/// is bound to nothing. They are the elastic ring's slots -- the shader
/// declares them, the row bound none, and the addresses have to be written
/// anyway for the reason [`no_sinks`] gives. Naming them for what they are
/// rather than eliding them is what keeps the argument table honest.
///
/// `w_page` and `w_off` are the fire's write directory: which page each token
/// lands in and where inside it.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand, the head width or the head count, and
/// [`Refusal::Unstated`] when the pool states no page size.
pub fn kv_append_paged(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let head_dim = stated(o, 0, f.head_dim);
    let n_kv_heads = stated(o, 1, f.kv_heads);
    let k_new = o.input(0)?;
    let v_new = o.input(1)?;
    let k_pages = BufMut(o.kv(f.layer, false));
    let v_pages = BufMut(o.kv(f.layer, true));
    let ring_4 = o.state(None);
    let ring_6 = o.state(None);
    let ring_7 = o.state(None);
    let ring_8 = o.state(None);
    let ring_9 = o.state(None);
    let ring_11 = o.state(None);
    let w_page = U32s(o.table(FireTable::KvWritePage));
    let w_off = U32s(o.table(FireTable::KvWriteOffset));
    let ring_15 = o.state(None);
    let page_size = o
        .pooled(FireTable::KvPageSize)
        .ok_or(Refusal::Unstated {
            what: "the KV page size: the pool has none",
        })?
        .cast_signed();
    Ok(vec![
        k_new.v(),
        v_new.v(),
        k_pages.v(),
        v_pages.v(),
        ring_4.v(),
        head_dim.v(),
        ring_6.v(),
        ring_7.v(),
        ring_8.v(),
        ring_9.v(),
        page_size.v(),
        ring_11.v(),
        n_kv_heads.v(),
        w_page.v(),
        w_off.v(),
        ring_15.v(),
        f.rows.cast_signed().v(),
    ])
}

/// `attn::split_qkv_bf16`: one packed projection cut into three.
///
/// A fused QKV projection writes `[rows, q+k+v]` and attention wants three
/// rectangles, so this copies rather than computes. The widths are in the
/// params block because there are three of them and they are not derivable
/// from anything the fire knows.
///
/// # Errors
///
/// [`Refusal::Absent`] for the packed block or any of the three results.
pub fn split_qkv_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let packed = o.input(0)?;
    let q = o.output(0)?;
    let k = o.output(1)?;
    let v_out = o.output(2)?;
    let params = o.params_block();
    Ok(vec![
        packed.v(),
        q.v(),
        k.v(),
        v_out.v(),
        params.v(),
        f.in_width.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `attn::gate`: gpt-oss's output gate, applied in place.
///
/// A bare row: nothing has dispatched this. `attn` is the result being gated
/// and `gate` the sigmoid's argument, and the pitch is the statement's
/// because a gated attention output is usually a window into a packed block.
///
/// # Errors
///
/// [`Refusal::Absent`] for either operand or the pitch.
pub fn gate(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_stride = o.param(0)?;
    let attn = o.output(0)?;
    let gate_in = o.input(0)?;
    Ok(vec![
        attn.v(),
        gate_in.v(),
        row_stride.v(),
        f.width.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `attn::q_gate_split`: the queries and their gate cut out of one block.
///
/// Also bare. gpt-oss projects `q` and its gate together and this separates
/// them, which is [`split_qkv_bf16`]'s problem with two results instead of
/// three -- and stated scalars instead of a block, because there are only
/// three numbers.
///
/// # Errors
///
/// [`Refusal::Absent`] for the block, either result, or any of the three
/// scalars.
pub fn q_gate_split(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let head_dim = stated(o, 0, f.head_dim);
    let qg_row_stride = o.param(1)?;
    let out_row_stride = o.param(2)?;
    let qg = o.input(0)?;
    let q_out = o.output(0)?;
    let gate_out = o.output(1)?;
    Ok(vec![
        qg.v(),
        q_out.v(),
        gate_out.v(),
        head_dim.v(),
        qg_row_stride.v(),
        out_row_stride.v(),
        f.q_heads.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `attn::logit_softcap`: gemma-2's `tanh` clamp on the attention logits.
///
/// # Errors
///
/// [`Refusal::Absent`] for the logits or the result.
pub fn logit_softcap(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let logits = o.input(0)?;
    let out = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        logits.v(),
        out.v(),
        params.v(),
        f.width.saturating_mul(f.rows).cast_signed().v(),
    ])
}

// ------------------------------------------------------------------ moe ---
//
// Routing is a sort. The router picks experts per token, `route_sort` orders
// the rows so that every expert's rows are contiguous, the multiply runs once
// per tile against whichever expert that tile belongs to, and `combine_sorted`
// puts the results back where the tokens were. Six buffers carry the
// permutation between those steps -- `perm`, `inv`, `row_expert`,
// `tile_expert`, `pad`, `expert_ids` -- and every one of them is a traced
// operand, which is why this family needs no state accessors at all.

/// `moe::router_topk`: the experts each token goes to, and how much of each.
///
/// `per_expert_scale` is bound to nothing here. The shader declares the
/// pointer and the unscaled router does not read it, but a slot left unbound
/// holds the previous dispatch's address -- see [`no_sinks`].
///
/// # Errors
///
/// [`Refusal::Absent`] for the logits or either result.
pub fn router_topk(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let logits = o.input(0)?;
    let expert_ids = o.output(0)?;
    let expert_weights = o.output(1)?;
    let params = o.params_block();
    let per_expert_scale = o.state(None);
    Ok(vec![
        logits.v(),
        expert_ids.v(),
        expert_weights.v(),
        params.v(),
        per_expert_scale.v(),
        f.n_experts.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `moe::router_topk_scaled`: [`router_topk`] with a per-expert gain applied
/// to the logits.
///
/// The gain is a second INPUT rather than a weight: it is a traced value.
///
/// # Errors
///
/// See [`router_topk`], plus [`Refusal::Absent`] for the gain.
pub fn router_topk_scaled(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let logits = o.input(0)?;
    let per_expert_scale = o.input(1)?;
    let expert_ids = o.output(0)?;
    let expert_weights = o.output(1)?;
    let params = o.params_block();
    Ok(vec![
        logits.v(),
        expert_ids.v(),
        expert_weights.v(),
        params.v(),
        per_expert_scale.v(),
        f.n_experts.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `moe::route_sort`: the rows put in expert order.
///
/// Four results, which is the most any routine in this backend writes:
/// `perm` is where each sorted slot came from, `inv` is where each token
/// went, `row_expert` is which expert owns each sorted row, and
/// `tile_expert` is the same per GEMM tile -- which is what lets the tiled
/// multiply pick a weight matrix per threadgroup without reading the routing
/// again.
///
/// # Errors
///
/// [`Refusal::Absent`] for the expert ids or any of the four results.
pub fn route_sort(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let expert_ids = o.input(0)?;
    let perm = o.output(0)?;
    let row_expert = o.output(1)?;
    let tile_expert = o.output(2)?;
    let params = o.params_block();
    let inv = o.output(3)?;
    Ok(vec![
        expert_ids.v(),
        perm.v(),
        row_expert.v(),
        tile_expert.v(),
        params.v(),
        inv.v(),
        f.n_experts.cast_signed().v(),
    ])
}

/// `moe::route_gather`: the activation rows copied into expert order.
///
/// `padded` is the SORTED extent and not the token count: the sort rounds
/// each expert's run up to a tile so the multiply's tiles never straddle two
/// experts, which makes the gathered rectangle taller than the fire's. The
/// statement states it -- the row said so with `rows_param = Some(4)` -- and
/// the fire's count is the fallback for a trace that does not.
///
/// # Errors
///
/// [`Refusal::Absent`] for the activation, the result or the permutation.
pub fn route_gather(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let padded = stated(o, 4, f.rows);
    let x = o.input(0)?;
    let out = o.output(0)?;
    let perm = o.input(1)?;
    let params = o.params_block();
    Ok(vec![
        x.v(),
        out.v(),
        perm.v(),
        params.v(),
        f.width.cast_signed().v(),
        padded.v(),
    ])
}

/// `moe::combine_sorted`: the experts' results weighted and summed back onto
/// their tokens.
///
/// The scatter that undoes [`route_gather`]'s gather. `inv` is where each
/// token's slots went, so this reads them rather than searching.
///
/// # Errors
///
/// [`Refusal::Absent`] for the results, the weights, the destination or the
/// inverse permutation.
pub fn combine_sorted(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let y = o.input(0)?;
    let expert_weights = o.input(1)?;
    let out = o.output(0)?;
    let params = o.params_block();
    let inv = o.input(2)?;
    Ok(vec![
        y.v(),
        expert_weights.v(),
        out.v(),
        params.v(),
        inv.v(),
        f.width.cast_signed().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `moe::shared_expert_combine`: the always-on expert folded into the routed
/// sum.
///
/// # Errors
///
/// [`Refusal::Absent`] for any of the three inputs, the result, or the width.
pub fn shared_expert_combine(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let width = stated(o, 0, f.width);
    let routed = o.input(0)?;
    let shared = o.input(1)?;
    let gate_in = o.input(2)?;
    let out = o.output(0)?;
    Ok(vec![
        routed.v(),
        shared.v(),
        gate_in.v(),
        out.v(),
        width.cast_unsigned().v(),
        f.rows.cast_signed().v(),
    ])
}

/// `moe::shared_expert_combine_strided`: [`shared_expert_combine`] over rows
/// a pitch apart.
///
/// A bare row: this has never dispatched. The pitch is the statement's or a
/// refusal, for the reason `rope::neox_strided` gives.
///
/// # Errors
///
/// See [`shared_expert_combine`], plus [`Refusal::Absent`] for the pitch.
pub fn shared_expert_combine_strided(
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<Vec<ArgValue>, Refusal> {
    let row_pitch = o.param(1)?;
    let mut v = shared_expert_combine(o, f)?;
    v.insert(5, row_pitch.v());
    Ok(v)
}

/// The routed matvec's shape: the codec, the activation, and the slot strides
/// that walk it.
///
/// `x` here is NOT the gathered rectangle. A matvec runs at decode, where
/// there is one token per request and `experts_per_token` slots each, so the
/// activation stays in token order and the kernel steps through the slots
/// itself -- which is what `x_slot_stride`, `x_row_stride` and
/// `slots_per_row` are for, and why this form reads `expert_ids` where the
/// tiled form reads `tile_expert`.
struct Routed {
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    in_vec_size: i32,
    out_vec_size: i32,
    expert_ids: Buf,
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
}

impl Routed {
    /// The arguments with `bias` spliced in at the shader's slot 7.
    fn with_bias(&self, bias: Buf) -> Vec<ArgValue> {
        vec![
            self.w.v(),
            self.scales.v(),
            self.biases.v(),
            self.x.v(),
            self.y.v(),
            self.in_vec_size.v(),
            self.out_vec_size.v(),
            bias.v(),
            self.expert_ids.v(),
            self.x_slot_stride.v(),
            self.x_row_stride.v(),
            self.slots_per_row.v(),
        ]
    }
}

/// What a routed matvec statement carries.
fn routed(o: &mut Handles<'_>) -> Result<Routed, Refusal> {
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
    let expert_ids = o.input(1)?;
    Ok(Routed {
        w,
        scales,
        biases,
        x,
        y,
        in_vec_size,
        out_vec_size,
        expert_ids,
        x_slot_stride,
        x_row_stride,
        slots_per_row,
    })
}

/// `moe::qmv_routed`: the routed matvec, no projection bias.
///
/// The bias slot is bound to nothing rather than skipped -- see
/// [`router_topk`].
///
/// # Errors
///
/// [`Refusal::Absent`] for a weight, operand or scalar the statement does not
/// carry.
pub fn qmv_routed(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let r = routed(o)?;
    let bias = o.state(None);
    let mut v = r.with_bias(bias);
    v.push(f.rows.cast_signed().v());
    Ok(v)
}

/// `moe::qmv_routed_bias`: [`qmv_routed`] with a per-expert projection bias.
///
/// # Errors
///
/// See [`qmv_routed`].
pub fn qmv_routed_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let bias = o.weight(3)?;
    let r = routed(o)?;
    let mut v = r.with_bias(bias);
    v.push(f.rows.cast_signed().v());
    Ok(v)
}

/// `moe::mxfp4_qmv_routed_bias`: [`qmv_routed_bias`] over gpt-oss's MXFP4
/// experts.
///
/// Two weights and not three -- codes and shared exponents -- so what the
/// affine form calls `biases` is here the projection bias, and the row said
/// so by naming `Weight(2)` for `bias` where the affine row names
/// `Weight(3)`.
///
/// # Errors
///
/// See [`qmv_routed`].
pub fn mxfp4_qmv_routed_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let in_vec_size = o.param(0)?;
    let out_vec_size = o.param(1)?;
    let x_slot_stride = o.param(2)?;
    let x_row_stride = o.param(3)?;
    let slots_per_row = o.param(4)?;
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.state(None);
    let x = o.input(0)?;
    let y = o.output(0)?;
    let bias = o.weight(2)?;
    let expert_ids = o.input(1)?;
    Ok(vec![
        w.v(),
        scales.v(),
        biases.v(),
        x.v(),
        y.v(),
        in_vec_size.v(),
        out_vec_size.v(),
        bias.v(),
        expert_ids.v(),
        x_slot_stride.v(),
        x_row_stride.v(),
        slots_per_row.v(),
        f.rows.cast_signed().v(),
    ])
}

/// `moe::qmm_t_routed`: the routed GEMM over the sorted rectangle.
///
/// `pad` is the sort's padded row count, read on the device: how many sorted
/// rows there actually are, which the host does not know because it depends
/// on the routing. `tile_expert` tells each threadgroup which expert's
/// weights to read.
///
/// # Errors
///
/// [`Refusal::Absent`] for a weight, operand or scalar the statement does not
/// carry, and [`Refusal::Unstated`] for a symbol that names no tile.
pub fn qmm_t_routed(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let (bm, bn) = tile(f)?;
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let pad = o.input(1)?;
    let tile_expert = o.input(2)?;
    Ok(vec![
        w.v(),
        scales.v(),
        biases.v(),
        x.v(),
        y.v(),
        pad.v(),
        tile_expert.v(),
        k.v(),
        n.v(),
        f.rows.cast_signed().v(),
        f.group.cast_signed().v(),
        f.bits.cast_signed().v(),
        bm.v(),
        bn.v(),
    ])
}

/// `moe::qmm_t_routed_fp16`: [`qmm_t_routed`] at one codec point.
///
/// Group 64 at four bits, compiled in, so the signature takes neither.
///
/// # Errors
///
/// See [`qmm_t_routed`].
pub fn qmm_t_routed_fp16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let (bm, bn) = tile(f)?;
    let w = o.weight(0)?;
    let scales = o.weight(1)?;
    let biases = o.weight(2)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let pad = o.input(1)?;
    let tile_expert = o.input(2)?;
    Ok(vec![
        w.v(),
        scales.v(),
        biases.v(),
        x.v(),
        y.v(),
        pad.v(),
        tile_expert.v(),
        k.v(),
        n.v(),
        f.rows.cast_signed().v(),
        bm.v(),
        bn.v(),
    ])
}

/// `moe::mxfp4_qmm_t_routed_bias`: the routed GEMM over MXFP4 experts.
///
/// # Errors
///
/// See [`qmm_t_routed`].
pub fn mxfp4_qmm_t_routed_bias(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let (k, n) = kn(o)?;
    let (bm, bn) = tile(f)?;
    let w = o.weight(0)?;
    let exponents = o.weight(1)?;
    let x = o.input(0)?;
    let y = o.output(0)?;
    let bias = o.weight(2)?;
    let pad = o.input(1)?;
    let tile_expert = o.input(2)?;
    Ok(vec![
        w.v(),
        exponents.v(),
        pad.v(),
        x.v(),
        y.v(),
        bias.v(),
        tile_expert.v(),
        k.v(),
        n.v(),
        f.rows.cast_signed().v(),
        bm.v(),
        bn.v(),
    ])
}

// ------------------------------------------------------------------ ssm ---
//
// The recurrent family, and the one that is dark end to end: all eight rows
// are bare, no `Source::GdnSlab` has ever been bound on this backend, and
// `Resolver::slab` answers `None` until a driver allocates the slabs. Writing
// the arms anyway is what makes the gap a MISSING ALLOCATION rather than a
// missing plumbing -- when the pool lands, these dispatch.
//
// The shape is a two-step: `gdn_prep` computes the gated projections into
// three scratch buffers, `gdn_core_recurrent` scans them. `gdn_core` fuses
// both for the decode case. `_slotted` means the state index comes from a
// slot table rather than the row number, which is what lets requests share a
// slab.

/// The gate weights and biases every GDN routine reads, in signature order.
///
/// `a_log` and `dt_bias` are the two named weights a prep statement carries
/// -- the pair that made this row hand-written in the first place, because
/// there was nowhere to put a second name.
fn gates(o: &mut Handles<'_>) -> Result<[ArgValue; 4], Refusal> {
    let a_log = F32s(o.weight(2)?.0);
    let dt_bias = o.weight(3)?;
    let a_gate = o.weight(4)?;
    let b_gate = o.weight(5)?;
    Ok([a_log.v(), dt_bias.v(), a_gate.v(), b_gate.v()])
}

/// `ssm::gdn_prep`: the gated deltanet's projections, before the scan.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand or weight the statement does not carry.
pub fn gdn_prep(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mixed = o.input(0)?;
    let conv_state = F32s(o.slab(f.layer, "conv_state")?);
    let conv_w = o.weight(0)?;
    let conv_b = o.weight(1)?;
    let g = gates(o)?;
    let pre_q = F32sMut(o.output(0)?.0);
    let pre_k = F32sMut(o.output(1)?.0);
    let pre_gate = F32sMut(o.output(2)?.0);
    let new_conv_state = F32sMut(o.slab(f.layer, "new_conv_state")?);
    let params = o.params_block();
    let mut v = vec![mixed.v(), conv_state.v(), conv_w.v(), conv_b.v()];
    v.extend(g);
    let tail: [ArgValue; 7] = [
        pre_q.v(),
        pre_k.v(),
        pre_gate.v(),
        new_conv_state.v(),
        params.v(),
        f.rows.cast_signed().v(),
        f.kv_heads.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `ssm::gdn_prep_slotted`: [`gdn_prep`] with the state index read from a
/// slot table.
///
/// # Errors
///
/// See [`gdn_prep`].
pub fn gdn_prep_slotted(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let slot_ids = U32s(o.input(1)?.0);
    let mut v = gdn_prep(o, f)?;
    v.insert(13, slot_ids.v());
    Ok(v)
}

/// `ssm::gdn_prep_prefill`: [`gdn_prep_slotted`] over a pitched block, many
/// scan rows at once.
///
/// `n_scan` is how many rows the scan will walk and `row_pitch` how far apart
/// they sit -- both the statement's, because a prefill's rectangle is a
/// window into a packed projection.
///
/// # Errors
///
/// See [`gdn_prep`], plus [`Refusal::Absent`] for the pitch or the scan
/// count.
pub fn gdn_prep_prefill(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_pitch = o.param(0)?;
    let n_scan = o.param(1)?;
    let slot_ids = U32s(o.input(1)?.0);
    let mixed = o.input(0)?;
    let conv_state = F32s(o.slab(f.layer, "conv_state")?);
    let conv_w = o.weight(0)?;
    let conv_b = o.weight(1)?;
    let g = gates(o)?;
    let pre_q = F32sMut(o.output(0)?.0);
    let pre_k = F32sMut(o.output(1)?.0);
    let pre_gate = F32sMut(o.output(2)?.0);
    let new_conv_state = F32sMut(o.slab(f.layer, "new_conv_state")?);
    let params = o.params_block();
    let mut v = vec![mixed.v(), conv_state.v(), conv_w.v(), conv_b.v()];
    v.extend(g);
    let tail: [ArgValue; 9] = [
        pre_q.v(),
        pre_k.v(),
        pre_gate.v(),
        new_conv_state.v(),
        params.v(),
        slot_ids.v(),
        row_pitch.v(),
        n_scan.v(),
        f.kv_heads.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `ssm::gdn_core`: the fused prep-and-scan, one token per request.
///
/// # Errors
///
/// See [`gdn_prep`].
pub fn gdn_core(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mixed = o.input(0)?;
    let conv_state = F32s(o.slab(f.layer, "conv_state")?);
    let rstate = F32sMut(o.slab(f.layer, "recurrent_state")?);
    let core_out = o.output(0)?;
    let conv_w = o.weight(0)?;
    let conv_b = o.weight(1)?;
    let g = gates(o)?;
    let new_conv_state = F32sMut(o.slab(f.layer, "new_conv_state")?);
    let params = o.params_block();
    let mut v = vec![
        mixed.v(),
        conv_state.v(),
        rstate.v(),
        core_out.v(),
        conv_w.v(),
        conv_b.v(),
    ];
    v.extend(g);
    let tail: [ArgValue; 5] = [
        new_conv_state.v(),
        params.v(),
        f.rows.cast_signed().v(),
        f.kv_heads.cast_signed().v(),
        f.head_dim.cast_signed().v(),
    ];
    v.extend(tail);
    Ok(v)
}

/// `ssm::gdn_core_slotted`: [`gdn_core`] with the state index read from a
/// slot table.
///
/// # Errors
///
/// See [`gdn_prep`].
pub fn gdn_core_slotted(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let slot_ids = U32s(o.input(1)?.0);
    let mut v = gdn_core(o, f)?;
    v.insert(13, slot_ids.v());
    Ok(v)
}

/// `ssm::gdn_core_recurrent`: the scan over what [`gdn_prep`] wrote.
///
/// # Errors
///
/// See [`gdn_prep`].
pub fn gdn_core_recurrent(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let mixed = o.input(0)?;
    let conv_state = F32s(o.slab(f.layer, "conv_state")?);
    let rstate = F32sMut(o.slab(f.layer, "recurrent_state")?);
    let core_out = o.output(0)?;
    let conv_w = o.weight(0)?;
    let conv_b = o.weight(1)?;
    let pre_q = F32s(o.input(1)?.0);
    let pre_k = F32s(o.input(2)?.0);
    let pre_gate = F32s(o.input(3)?.0);
    let new_conv_state = F32sMut(o.slab(f.layer, "new_conv_state")?);
    let params = o.params_block();
    Ok(vec![
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
        f.rows.cast_signed().v(),
        f.kv_heads.cast_signed().v(),
        f.head_dim.cast_signed().v(),
    ])
}

/// `ssm::gdn_core_recurrent_slotted`: [`gdn_core_recurrent`] with the state
/// index read from a slot table.
///
/// # Errors
///
/// See [`gdn_prep`].
pub fn gdn_core_recurrent_slotted(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let slot_ids = U32s(o.input(4)?.0);
    let mut v = gdn_core_recurrent(o, f)?;
    v.insert(11, slot_ids.v());
    Ok(v)
}

/// `ssm::gdn_core_recurrent_prefill`: the scan over a prefill's many rows.
///
/// `pad` is the padded row count, read on the device: a prefill's scan length
/// depends on how the requests packed, which the host does not know when it
/// builds the fire. `lanes` and `vrows` are how the scan divides the value
/// dimension, and both are the statement's -- they are a decomposition
/// choice, not a shape.
///
/// # Errors
///
/// [`Refusal::Absent`] for an operand or any of the four scalars.
pub fn gdn_core_recurrent_prefill(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let row_pitch = o.param(0)?;
    let n_scan = o.param(1)?;
    let lanes = o.param(2)?;
    let vrows = o.param(3)?;
    let pad = o.input(0)?;
    let rstate = F32sMut(o.slab(f.layer, "recurrent_state")?);
    let core_out = o.output(0)?;
    let pre_q = F32s(o.input(1)?.0);
    let pre_k = F32s(o.input(2)?.0);
    let pre_gate = F32s(o.input(3)?.0);
    let params = o.params_block();
    let slot_ids = U32s(o.input(4)?.0);
    Ok(vec![
        pad.v(),
        rstate.v(),
        core_out.v(),
        pre_q.v(),
        pre_k.v(),
        pre_gate.v(),
        params.v(),
        slot_ids.v(),
        row_pitch.v(),
        n_scan.v(),
        f.kv_heads.cast_signed().v(),
        f.head_dim.cast_signed().v(),
        lanes.v(),
        vrows.v(),
    ])
}

// ----------------------------------------------------------------- ptir ---

/// `ptir::copy_logits_bf16`: one rectangle of logits copied into another.
///
/// The whole of the PTIR family on this backend, and dark: nothing in the
/// compiler emits it yet. It exists because a program that computes logits
/// and a program that consumes them may not share a buffer.
///
/// # Errors
///
/// [`Refusal::Absent`] for the source or the destination.
pub fn copy_logits_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let source = o.input(0)?;
    let destination = o.output(0)?;
    let params = o.params_block();
    Ok(vec![
        source.v(),
        destination.v(),
        params.v(),
        f.width.v(),
        f.rows.v(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resolver that answers nothing: these fixtures are about statements,
    /// and a statement's operands are already bound before an arm runs.
    struct Unresolved;

    impl Resolver for Unresolved {
        fn weight(&mut self, _: &str) -> Option<Slice> {
            None
        }
        fn named(&mut self, _: model_ir::trace::ValueId) -> Option<Slice> {
            None
        }
    }

    fn handle(address: u64, width: u32) -> BoundArg {
        BoundArg {
            slice: Slice { address, bytes: 64 },
            width,
        }
    }

    /// The split is the trace's order, and a weight is not a widthed operand.
    #[test]
    fn a_statement_splits_into_inputs_results_and_weights() {
        let args = [
            Arg::Weight("q_proj".into()),
            Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            },
            Arg::Arena {
                at: 16,
                width: 8,
                bytes: 2,
            },
            Arg::Weight("k_proj".into()),
        ];
        let (ins, outs, weights) = split(&args, 1);
        assert_eq!(ins, vec![1], "the widthed operand before the last");
        assert_eq!(outs, vec![2], "and the last widthed one is the result");
        assert_eq!(weights, vec![0, 3], "in the order the trace names them");
    }

    /// A statement with no results keeps every widthed operand an input.
    ///
    /// `kv_append` is the shape: it writes the POOL and produces no traced
    /// value. The table path defaulted this count to one, which took the last
    /// input for a result -- the V pages filled with zero, every layer, and
    /// the attention that read them answered without failing.
    #[test]
    fn a_statement_that_writes_only_state_has_no_results() {
        let args = [
            Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            },
            Arg::Arena {
                at: 16,
                width: 8,
                bytes: 2,
            },
        ];
        let (ins, outs, _) = split(&args, 0);
        assert_eq!(ins, vec![0, 1]);
        assert!(outs.is_empty());
    }

    /// A handle is assigned when an arm ASKS, so a fire table and an operand
    /// share one space.
    #[test]
    fn handles_are_numbered_in_the_order_an_arm_asks() {
        let args = [handle(0x100, 8), handle(0x200, 8)];
        let (ins, outs, weights) = split(
            &[
                Arg::Arena {
                    at: 0,
                    width: 8,
                    bytes: 2,
                },
                Arg::Arena {
                    at: 16,
                    width: 8,
                    bytes: 2,
                },
            ],
            1,
        );
        let mut r = Unresolved;
        let mut o = Handles::new(&args, &ins, &outs, &weights, &[], &mut r);
        let a = o.input(0).expect("an input");
        let table = o.state(Some(Slice {
            address: 0x900,
            bytes: 32,
        }));
        let b = o.output(0).expect("a result");
        assert_eq!(a, Buf(0));
        assert_eq!(table, Buf(1), "the table takes the next handle, not a slot");
        assert_eq!(b, BufMut(2));
        assert_eq!(o.bound()[1].slice.address, 0x900);
    }

    /// A float scalar is REINTERPRETED out of the trace's `u32`, not
    /// converted.
    ///
    /// Every scalar rides as bits. A conversion would hand a kernel expecting
    /// `1.0` the number 1065353216.0, which is a scale no attention would
    /// survive and no type would catch.
    #[test]
    fn a_float_scalar_keeps_its_bits() {
        let args: [BoundArg; 0] = [];
        let params = [Some(1.0f32.to_bits()), Some(7)];
        let mut r = Unresolved;
        let o = Handles::new(&args, &[], &[], &[], &params, &mut r);
        assert!((o.param_f32(0).expect("a float") - 1.0).abs() < f32::EPSILON);
        assert_eq!(o.param(1), Ok(7));
        assert!(o.param(2).is_err(), "past the statement's run");
    }

    /// An arm asking past its statement refuses rather than binding nothing.
    ///
    /// The table path answered a missing operand with a zero address, and
    /// `mxfp4_qmv_routed_bias` read an additive bias off it for every expert
    /// logit without a word.
    #[test]
    fn an_arm_reaching_past_its_statement_is_refused() {
        let args = [handle(0x100, 8)];
        let mut r = Unresolved;
        let mut o = Handles::new(&args, &[0], &[], &[], &[], &mut r);
        assert!(o.input(0).is_ok());
        assert!(matches!(o.input(1), Err(Refusal::Absent { .. })));
        assert!(matches!(o.output(0), Err(Refusal::Absent { .. })));
        assert!(matches!(o.weight(0), Err(Refusal::Absent { .. })));
    }
}
