//! One launch of a plan, turned into the four things a dispatch needs.
//!
//! [`binding`](crate::binding) answers where an operand lives, and
//! [`geometry`](crate::geometry) answers how many workgroups a rule wants.
//! Neither knows about the other, and a plan states neither directly: it
//! states a rectangle, a symbol, and a run of operands. This file is the
//! join. It turns one [`Launch`] into a [`Dispatch`] — a module name, its
//! bound operands, its push bytes, and its grid — and nothing here needs a
//! device, so the whole join can be measured against real plans on a machine
//! with no GPU in it.
//!
//! # Why the grid needs the statement and not just the fire
//!
//! A [`Rule`] evaluates against [`Dims`], and most of `Dims` is fire-wide:
//! head count, head width, expert count. Most, not all. `KernelSig` carries
//! three indices — `grid_param`, `head_param`, `heads_param` — that each name
//! a scalar in the STATEMENT's own run, and a row that names one is saying
//! its extent varies per layer in a way no fire-wide number can express.
//!
//! Gemma-4 is the case that forces it: its full-attention layers rotate a
//! quarter of each head and its sliding layers rotate all of one, and they
//! carry four 512-wide KV heads against sixteen 256-wide ones. A driver
//! reading the fire's `rotary_dims` describes neither. `driver-metal` learned
//! this and states so at [`dims_of`]'s counterpart; the reading is
//! transcribed rather than rediscovered.
//!
//! A row naming a param its statement does not carry falls back to the fire
//! rather than to zero, because a grid of zero is a dispatch that runs
//! nothing, returns success, and leaves the output holding whatever it was
//! born with. That failure mode is the one this crate refuses hardest —
//! [`Device::run`](crate::device::Device::run) refuses a zero grid outright.

use crate::binding::{Arena, Params, Resolve, Slot, Unbindable, descriptors, reorder, scalars};
use crate::device::Bound;
use crate::geometry::{Dims, Module, Rule, Ungeometric, groups};
use kernels::KernelSig;
use model_compiler::lower::{Arg, Launch, Lowered};

/// The fire-wide shape a plan is executed at.
///
/// Everything a [`Rule`] needs that a single statement does not state. A
/// statement may override three of these; see [`dims_of`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Geometry {
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Channels a partial rope rotates.
    pub rotary_dims: u32,
    /// Experts the router scores.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
}

/// One recordable dispatch: everything a command buffer needs, and nothing
/// that needs a command buffer to compute.
#[derive(Clone, Debug, PartialEq)]
pub struct Dispatch<'a> {
    /// The entrypoint to run.
    ///
    /// Borrowed from [`Lowered::kernels`] on the plan-ordered path, which
    /// costs nothing and is what it has always been. Owned when a ROUTINE
    /// composed it: a body that varies over an instantiation axis builds the
    /// whole spelling itself — `affine_qmm_t_bf16_gs_128_b_4` — and the
    /// `String` it built lives in the body's frame, which ends before the
    /// dispatch is recorded. `Cow` is the smaller of the two prices; the
    /// other was making every body's entrypoint `&'static str`, which forbids
    /// exactly the axis instantiation half this table needs.
    pub symbol: std::borrow::Cow<'a, str>,
    /// The operands, in binding order over the module's non-hole slots.
    pub buffers: Vec<Bound<'a>>,
    /// Which of [`Self::buffers`] the shader may WRITE through, in the same
    /// order and of the same length.
    ///
    /// Read off the kernel row's operand types -- [`kernels::Ty::BufMut`] is
    /// "the launcher may write through this" and everything else is a read --
    /// which is the only place the distinction exists. SPIR-V does not carry
    /// it usefully: `slangc` decorates a buffer `NonWritable` only when the
    /// shader said `readonly`, and this tree's shaders mostly do not.
    ///
    /// What it is FOR is the barrier between two dispatches. A fire is a few
    /// hundred rectangles over one arena and the recording used to put a
    /// full pipeline barrier between every pair of them, which on this card
    /// is 8 microseconds each -- measured, 3.8 milliseconds of a 7.2
    /// millisecond decode. Most neighbouring pairs do not touch the same
    /// bytes, and this is what lets the recorder tell which ones do.
    ///
    /// A row that states no operands has no answer here, so every slot is
    /// marked written: the coarse reading is the safe one, and it is what
    /// this driver did for every slot of every launch before this existed.
    pub writes: Vec<bool>,
    /// Where this launch's scalars go, and the bytes to put there.
    ///
    /// Not a `Vec<u8>`, because "the bytes to push" is only half the answer:
    /// the module decides whether they ride push constants or a struct in a
    /// storage buffer, and the reachable symbols split almost evenly on it.
    /// A dispatch that flattened both into one byte run would be a dispatch
    /// whose caller has to ask the shader the same question again.
    pub params: Params,
    /// Where in [`Self::buffers`] the caller's scalar block goes.
    ///
    /// [`Params::Block`] states a BINDING index; this states an index into
    /// the dense list [`Device::run`](crate::device::Device::run) takes,
    /// which skips the module's descriptor holes. The two agree on every
    /// module in this tree -- no module that reads its scalars from a buffer
    /// has a hole -- and [`plan_one`] refuses rather than assume it.
    pub block_at: Option<usize>,
    /// Workgroups in each dimension. Never contains a zero; see
    /// [`Undispatchable::Empty`].
    pub groups: [u32; 3],
    /// Which traced op produced this, for a refusal to point at.
    pub op: u32,
}

/// Why a launch could not be turned into a dispatch.
#[derive(Clone, Debug, PartialEq)]
pub enum Undispatchable {
    /// The plan names a symbol no row in the kernel table states.
    Unknown {
        /// The symbol the plan named.
        symbol: String,
    },
    /// The row's slots and the module's bindings do not line up.
    Layout(
        /// Which reading disagreed.
        crate::binding::Unlayoutable,
    ),
    /// An operand could not be bound.
    Operand {
        /// Which operand of the launch, counting from zero.
        at: usize,
        /// Why.
        why: Unbindable,
    },
    /// The scalars the plan states do not fit where the module reads them.
    Scalars {
        /// How many the plan states.
        stated: usize,
        /// How many the module's block or push range has room for.
        room: usize,
    },
    /// The row names a scalar this driver cannot work out. See
    /// [`crate::binding::Misplaced::Unresolved`].
    Unresolved {
        /// The symbol whose row named it.
        symbol: String,
        /// Which operand, counting from zero.
        at: usize,
        /// The operand's name, as the row spells it.
        name: &'static str,
        /// The `kernels::Source` variant, rendered.
        source: String,
    },
    /// The row walks the KV cache with strides and no page table, and this
    /// driver's pool is paged. See [`crate::binding::Misplaced::Contiguous`].
    Contiguous {
        /// The symbol whose row named it.
        symbol: String,
        /// Which operand, counting from zero.
        at: usize,
        /// The operand's name, as the row spells it.
        name: &'static str,
    },
    /// The rule and dims do not describe a grid.
    Ungeometric {
        /// Why.
        why: Ungeometric,
    },
    /// The rectangle sits under a guard nothing has evaluated.
    ///
    /// Not a defect: a conditional arm is a launch a caller records only
    /// after deciding the branch. Refused rather than recorded because
    /// recording every arm would RUN every arm.
    Conditional {
        /// The symbol the guarded rectangle names.
        symbol: String,
        /// Which conditional region, as an index into `Lowered::conds`.
        cond: u32,
    },
    /// The plan states a different number of operands than the module binds.
    ///
    /// Almost always the plan being short of a resource the driver is
    /// expected to own -- a paged KV cache, a page table -- rather than a
    /// mistake. Refused because binding fewer descriptors than a shader
    /// reads is what `VUID-vkCmdDispatch-None-08114` exists to catch, and
    /// binding more would put a tensor at a slot on the theory that nothing
    /// looks there.
    Arity {
        /// The symbol whose module disagrees.
        symbol: String,
        /// How many operands the plan states.
        stated: usize,
        /// How many bindings the module really has: its layout less its
        /// holes.
        module: usize,
    },
    /// The grid computed to zero in some dimension.
    ///
    /// Legal Vulkan and always a defect: it runs nothing, reports success,
    /// and leaves the output holding whatever it held before. Refused here
    /// rather than at the device so that the traced op is still in hand to
    /// name.
    Empty {
        /// The grid that would have been dispatched.
        groups: [u32; 3],
    },
    /// A crossed ROUTINE would not launch this statement.
    ///
    /// The routine path's single refusal, carrying whatever the arm, the body
    /// or the encoder said. Distinct from every variant above because those
    /// are the TABLE path's vocabulary -- a row that named an operand it does
    /// not have, a rule with no grid -- and a routine has none of those parts
    /// to get wrong. What it can refuse instead is an extent that came out
    /// empty, an entrypoint this build did not produce, and an argument list
    /// the module's bindings do not fit.
    Refused {
        /// The symbol the plan named, which is not necessarily the entrypoint
        /// the body asked for.
        symbol: String,
        /// What the routine said.
        why: kernels::routine::Refusal,
    },
}

impl std::fmt::Display for Undispatchable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { symbol } => write!(f, "no kernel row states `{symbol}`"),
            Self::Refused { symbol, why } => {
                write!(f, "the routine for `{symbol}` refused: {why:?}")
            }
            Self::Layout(why) => write!(f, "{why}"),
            Self::Operand { at, why } => write!(f, "operand {at}: {why}"),
            Self::Scalars { stated, room } => {
                write!(f, "{stated} scalars stated, room for {room}")
            }
            Self::Unresolved {
                symbol,
                at,
                name,
                source,
            } => write!(
                f,
                "`{symbol}` operand {at} (`{name}`) is sourced from {source}, \
                 which this driver does not know how to work out"
            ),
            Self::Contiguous { symbol, at, name } => write!(
                f,
                "`{symbol}` operand {at} (`{name}`) is a contiguous KV stride, \
                 and this driver's pool is paged: the row would read real \
                 memory at the wrong tokens"
            ),
            Self::Conditional { symbol, cond } => {
                write!(f, "`{symbol}` sits under unevaluated guard {cond}")
            }
            Self::Arity {
                symbol,
                stated,
                module,
            } => write!(
                f,
                "`{symbol}` states {stated} operands and its module binds {module}"
            ),
            Self::Ungeometric { why } => write!(f, "no grid: {why}"),
            Self::Empty { groups } => {
                write!(
                    f,
                    "a grid of {groups:?} would run nothing and report success"
                )
            }
        }
    }
}

impl std::error::Error for Undispatchable {}

/// The row widths this launch's operands state, in the trace's order.
fn widths<'a>(lowered: &'a Lowered, launch: &Launch) -> impl DoubleEndedIterator<Item = u32> + 'a {
    lowered.args[launch.args.start as usize..launch.args.end as usize]
        .iter()
        .filter_map(|arg| match arg {
            Arg::Arena { width, .. } | Arg::Named { width, .. } => Some(*width),
            Arg::Weight(_) => None,
        })
}

/// The scalar at `index` of this launch's own run, when it states one there
/// and it is not zero.
///
/// Zero is filtered because it is what an unstated slot reads as, and a zero
/// extent is a grid of nothing.
fn stated(lowered: &Lowered, launch: &Launch, index: Option<u8>) -> Option<u32> {
    let i = index?;
    let at = launch.params.start as usize + i as usize;
    if at >= launch.params.end as usize {
        return None;
    }
    lowered.params.get(at).copied().filter(|n| *n > 0)
}

/// The dims one launch evaluates its rule at.
///
/// `sig` is the launch's own kernel row, and it is here because a row may say
/// that its rule's extent comes from the STATEMENT rather than from the fire.
/// See this module's own documentation for the case that forces it.
///
/// # What each of the three overrides is held by
///
/// `grid_param` fires 2220 times over this tree's six texts -- the number the
/// walk pins as `overridden`, and it was 710 when this note was first written
/// against three of them. A driver that read only the fire would
/// normalise over the wrong width every one of those times and produce
/// numbers rather than an error, which is what
/// `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`
/// pins with a workgroup total: dropping this line changes that number and
/// nothing else in the suite.
///
/// `head_param` and `heads_param` fired ZERO times when this was written, and
/// this note said so and said the two lines below were transcribed from
/// `driver-metal` and carried untested. Half of that stopped being true the
/// day the walk began handing `plan_one` the model's real geometry: across
/// the six texts, 1056 rows now state a head width and 352 a head count.
///
/// The other half stayed true for longer than it looked, and the way it hid
/// is worth keeping. Counting rows that STATE a head shape does not witness
/// these lines USING it. Deleting either override left the whole suite green,
/// because for these six texts the stated value EQUALS the fire's -- the
/// overrides are no-ops here, and a no-op cannot be caught by a test that
/// only checks the answer. The model that separates them is gemma-4, whose
/// full-attention layers carry four 512-wide KV heads against its sliding
/// layers' sixteen 256-wide ones, and it is not one of the texts measured.
///
/// So the walk now hands the same launch a geometry that disagrees on
/// purpose and requires the answer to come from the row. Deleting either
/// line fails. Recorded rather than tidied away because "it is counted" read
/// as "it is checked" for two commits, and those are different claims.
#[must_use]
pub fn dims_of(sig: &KernelSig, lowered: &Lowered, launch: &Launch, fire: Geometry) -> Dims {
    // The last widthed operand, which is the launch's last OUTPUT: what sizes
    // the rectangle for nearly every rule.
    let width = widths(lowered, launch).next_back().unwrap_or(0);
    let extent = stated(lowered, launch, sig.grid_param);
    Dims {
        rows: launch.rows.end - launch.rows.start,
        width,
        // The FIRST widthed operand, which is the first input. What sizes a
        // statement that reads one packed buffer and writes several, since
        // there no single output spells the grid.
        in_width: widths(lowered, launch).next().unwrap_or(0),
        q_heads: fire.q_heads,
        kv_heads: stated(lowered, launch, sig.heads_param).unwrap_or(fire.kv_heads),
        head_dim: stated(lowered, launch, sig.head_param).unwrap_or(fire.head_dim),
        // The SAME stated number, read by whichever rule asked for it: a
        // rope's rotated channels, a norm's reduction axis. A row names one
        // scalar as its extent and its rule knows which dimension that is.
        axis: extent.unwrap_or(width),
        rotary_dims: extent.unwrap_or(fire.rotary_dims),
        n_experts: fire.n_experts,
        experts_per_token: fire.experts_per_token,
    }
}

/// The SPIR-V a launch will run, as the two things this file needs from it.
///
/// Grouped because they always travel together and always come from the same
/// built pipeline: the workgroup size and tile decide the GRID, and the
/// bindings and holes decide the ARITY. Separating them at the call site
/// would let a caller pass a module's geometry with another module's layout,
/// which is a mistake nothing downstream could detect.
// No `Debug`: `KernelSig` has none, and printing a whole kernel row beside a
// workgroup size would say nothing a reader of a refusal wants.
#[derive(Clone, Copy)]
pub struct Built<'a> {
    /// Workgroup size and tile, from the entrypoint name and the module.
    pub module: Module,
    /// What the module binds.
    pub declared: &'a crate::spirv::Declared,
    /// The kernel table's row for this symbol, when the caller already has it.
    ///
    /// `None` means "look it up", which is what a test wants and what this
    /// did for everyone before the field existed. A fire passes `Some`
    /// because the lookup is a LINEAR scan of the whole table -- see
    /// [`kernels::sig_in`], which compares every symbol and then, for a
    /// specialised name like `affine_qmm_t_bf16_gs_128_b_4`, walks the axis
    /// points of every row again -- and a decode of 452 rectangles over nine
    /// distinct symbols was doing it 452 times. Measured on this card: 0.23
    /// ms a fire, against a plan phase of 0.85 and a whole step of 6.7.
    ///
    /// The same argument, and the same cache, as the SPIR-V reflection beside
    /// it in `serve::fire`: once per SYMBOL, not once per launch.
    pub sig: Option<&'static KernelSig>,
}

/// Where a launch's operands are to be found.
///
/// Named `Sources` rather than the `Into` it started as, because a public
/// type called `Into` shadows `std::convert::Into` at every use site that
/// imports it.
#[derive(Clone, Copy, Debug)]
pub struct Sources<'a, R: Resolve> {
    /// The buffer the plan's offsets are into, and how much of it the plan
    /// was allowed to place in.
    pub arena: Arena<'a>,
    /// What holds the weights and the seam values.
    pub resolver: &'a R,
    /// The device's `minStorageBufferOffsetAlignment`.
    pub min_offset: u64,
}

/// The row in `table` whose symbol the plan names.
///
/// [`kernels::sig_in`], not an equality test. A plan names the symbol it will
/// dispatch, and that symbol carries the variant suffixes the specialisation
/// axes append -- `_gs_64_b_4` for an affine group, `_bm_16_bn_32` for a
/// routed tile, `_d_128` for a head width. The TABLE states the axes, not
/// the points, so exact equality finds a row for the few symbols that have no
/// axis and nothing for the rest.
///
/// Measured before it was fixed: over the texts of the day, exact matching found a
/// row for 432 launches and failed on 3030, across sixteen distinct symbols
/// that all exist and all have modules built for them. `sig_in` finds all of
/// them.
fn row(table: &'static [KernelSig], symbol: &str) -> Option<&'static KernelSig> {
    kernels::sig_in(table, symbol)
}

/// Turn one launch into a dispatch.
///
/// `table` is `kernels_vulkan::KERNELS` in every caller; it is a parameter so
/// that this module depends on the kernel *vocabulary* rather than on one
/// table, which is what lets a test state its own rows.
///
/// `module` describes the SPIR-V that will run: its workgroup size, and the
/// tile its name encodes. `declared` describes what that SPIR-V binds. Both
/// come from a built pipeline, so this function is the last step before a
/// command buffer and the first one that needs the shader to exist.
///
/// # Errors
///
/// [`Undispatchable`] in every case, naming the traced op through the
/// [`Dispatch`] it did not manage to build.
pub fn plan_one<'a, R: Resolve>(
    lowered: &'a Lowered,
    launch: &Launch,
    table: &'static [KernelSig],
    built: Built<'_>,
    sources: Sources<'a, R>,
    fire: Geometry,
) -> Result<Dispatch<'a>, Undispatchable> {
    let Built {
        module,
        declared,
        sig,
    } = built;
    let Sources {
        arena,
        resolver,
        min_offset,
    } = sources;
    let symbol = lowered.kernels[launch.kernel as usize].as_str();
    // A conditional rectangle's guard was NOT answered by the lowering, and
    // this walk has no way to answer it -- recording every arm would run
    // every arm. `driver-metal` refuses here for the same reason.
    if launch.cond != Launch::NO_COND {
        return Err(Undispatchable::Conditional {
            symbol: symbol.to_owned(),
            cond: launch.cond,
        });
    }
    let sig = match sig {
        Some(s) => s,
        None => row(table, symbol).ok_or_else(|| Undispatchable::Unknown {
            symbol: symbol.to_owned(),
        })?,
    };

    // Not `bind`, which hands the descriptors over in the order the TRACE
    // states -- inputs, outputs, weights. The shader binds them in the order
    // its kernel row states, and for 2898 of this tree's 3992 rectangles
    // those differ. `rms_single_row` is the plainest: `norm/rms.slang` is
    // `0=x, 1=w, 2=out`, its row is `In(0), Weight(0), Out(0)`, and the trace
    // hands over `In(0), Out(0), Weight(0)` -- so positionally the norm reads
    // its own output as the weight and writes over the weight.
    let slots = reorder(sig, lowered, launch, arena, resolver, min_offset)
        .map_err(|(at, why)| Undispatchable::Operand { at, why })?;

    // Which slot the shader may write, read here rather than after
    // `descriptors` because this is the only point where a slot's index is
    // still the index of the OPERAND that produced it.
    //
    // Sound because `descriptors` never drops a `Slot::Buffer`: it drops
    // repeated `Slot::Params` and pops trailing `Slot::Nothing`, and refuses
    // outright rather than pop anything else. So the buffers below are
    // `slots`' buffers, in `slots`' order.
    let writes: Vec<bool> = if sig.operands.is_empty() {
        slots
            .iter()
            .filter(|s| matches!(s, Slot::Buffer(_)))
            .map(|_| true)
            .collect()
    } else {
        slots
            .iter()
            .zip(sig.operands)
            .filter(|(s, _)| matches!(s, Slot::Buffer(_)))
            .map(|(_, o)| o.ty == kernels::Ty::BufMut)
            .collect()
    };

    // Where the scalars go is the MODULE's decision, not the plan's, and
    // `binding::params` is what reads it off the SPIR-V. Passed through
    // rather than flattened: half the reachable symbols want a push block
    // and half want a struct in a buffer of their own, and the plan states
    // the same run of `u32` either way.
    let placed = scalars(sig, lowered, launch, declared, resolver).map_err(|why| match why {
        crate::binding::Misplaced::Count { stated, push, .. } => {
            Undispatchable::Scalars { stated, room: push }
        }
        crate::binding::Misplaced::Contiguous { at, name } => Undispatchable::Contiguous {
            symbol: symbol.to_owned(),
            at,
            name,
        },
        crate::binding::Misplaced::Unresolved { at, name, source } => Undispatchable::Unresolved {
            symbol: symbol.to_owned(),
            at,
            name,
            source,
        },
    })?;

    // The plan's operands, PLUS the slot a parameter block takes, have to be
    // the module's real bindings -- the layout less its holes, which is the
    // arity [`Device::run`](crate::device::Device::run) enforces. Checked
    // here so that a mismatch is refused with the traced op in hand instead
    // of becoming a `Dispatch` no device accepts.
    //
    // The `+ 1` is the whole reason this comes after the scalars and not
    // before. Six of the reachable symbols read their parameters as a struct
    // in a storage buffer, and that buffer is a BINDING the plan never
    // mentions: `router_topk` states three operands against a four-binding
    // module and the fourth is its own scalar block. Checking arity first
    // refused 1439 rectangles across nine symbols that are all perfectly
    // dispatchable -- the measurement that produced this line.
    let laid = descriptors(slots, &placed, declared).map_err(Undispatchable::Layout)?;
    if laid.len() != declared.bindings as usize {
        return Err(Undispatchable::Arity {
            symbol: symbol.to_owned(),
            stated: laid.len(),
            module: declared.bindings as usize,
        });
    }

    // The dense list, which is what a descriptor set is written from: the
    // holes take no write, so they take no entry.
    let block_at = laid
        .iter()
        .position(|s| matches!(s, Slot::Params))
        .map(|at| {
            laid[..at]
                .iter()
                .filter(|s| !matches!(s, Slot::Nothing))
                .count()
        });
    let buffers: Vec<Bound<'a>> = laid
        .into_iter()
        .filter_map(|s| match s {
            Slot::Buffer(b) => Some(b),
            Slot::Params | Slot::Nothing => None,
        })
        .collect();
    let real = declared.bindings as usize - declared.holes();
    if buffers.len() + usize::from(block_at.is_some()) != real {
        return Err(Undispatchable::Arity {
            symbol: symbol.to_owned(),
            stated: buffers.len() + usize::from(block_at.is_some()),
            module: real,
        });
    }
    // The claim `writes` rests on: that laying the slots out neither drops,
    // adds nor moves a buffer. Checked rather than trusted, because a
    // disagreement here is not a crash -- it is a barrier decided from some
    // other operand's writability, which is a race, which is a plausible
    // number.
    if writes.len() != buffers.len() {
        return Err(Undispatchable::Arity {
            symbol: symbol.to_owned(),
            stated: writes.len(),
            module: buffers.len(),
        });
    }

    let dims = dims_of(sig, lowered, launch, fire);
    let groups =
        groups(sig.launch, dims, module).map_err(|why| Undispatchable::Ungeometric { why })?;
    if groups.contains(&0) {
        return Err(Undispatchable::Empty { groups });
    }

    Ok(Dispatch {
        symbol: symbol.into(),
        buffers,
        writes,
        params: placed,
        block_at,
        groups,
        op: launch.op,
    })
}

/// The rule a symbol's row states, for a caller that needs the grid before it
/// needs the operands.
///
/// # Errors
///
/// [`Undispatchable::Unknown`] when no row states the symbol.
pub fn rule_of(table: &'static [KernelSig], symbol: &str) -> Result<Rule, Undispatchable> {
    row(table, symbol)
        .map(|k| k.launch)
        .ok_or_else(|| Undispatchable::Unknown {
            symbol: symbol.to_owned(),
        })
}
