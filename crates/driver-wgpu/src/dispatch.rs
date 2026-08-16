//! One launch of a plan, turned into the four things a dispatch needs.
//!
//! [`binding`](crate::binding) answers where an operand lives, and
//! [`geometry`](crate::geometry) answers how many workgroups a rule wants.
//! Neither knows about the other, and a plan states neither directly: it states
//! a rectangle, a symbol, and a run of operands. This file is the join. It
//! turns one [`Launch`] into a [`Dispatch`] — a module name, its bound
//! operands, its parameter bytes, and its grid — and nothing here needs a
//! device, so the whole join can be measured against real plans on a machine
//! with no adapter in it.
//!
//! # Why the grid needs the statement and not just the fire
//!
//! A [`Rule`] evaluates against [`Dims`], and most of `Dims` is fire-wide: head
//! count, head width, expert count. Most, not all. `KernelSig` carries three
//! indices — `grid_param`, `head_param`, `heads_param` — that each name a
//! scalar in the STATEMENT's own run, and a row that names one is saying its
//! extent varies per layer in a way no fire-wide number can express.
//!
//! Gemma-4 is the case that forces it: its full-attention layers rotate a
//! quarter of each head and its sliding layers rotate all of one, and they
//! carry four 512-wide KV heads against sixteen 256-wide ones. A driver reading
//! the fire's `rotary_dims` describes neither. `driver-metal` learned this and
//! states so at [`dims_of`]'s counterpart; the reading is transcribed rather
//! than rediscovered.
//!
//! A row naming a param its statement does not carry falls back to the fire
//! rather than to zero, because a grid of zero is a dispatch that runs nothing,
//! returns success, and leaves the output holding whatever it was born with.
//! That failure mode is the one this crate refuses hardest —
//! [`Undispatchable::Empty`] is where.
//!
//! # The device limit is deliberately not applied here
//!
//! [`plan_one`] calls [`crate::geometry::groups`] and not `groups_within`,
//! because a `Dispatch` is what the KERNEL needs and the limit is a property of
//! the adapter that will run it. The device half checks
//! [`Dispatch::groups`] against its own
//! `max_compute_workgroups_per_dimension` at encode time, where it has the
//! number. Folding the limit in here would mean a plan that this build can
//! reason about only where it can also run.

use crate::binding::{
    Arena, Bound, ParamSlot, Params, Resolve, Slot, Unbindable, descriptors, reorder, scalars,
};
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

/// One recordable dispatch: everything a command encoder needs, and nothing
/// that needs an encoder to compute.
#[derive(Debug)]
pub struct Dispatch<'a, B> {
    /// The entrypoint to run, borrowed from [`Lowered::kernels`].
    pub symbol: &'a str,
    /// The operands, in `@group(0)` binding order over the module's non-hole
    /// slots.
    pub buffers: Vec<Bound<'a, B>>,
    /// Where this launch's scalars go, and the bytes to put there.
    ///
    /// Not a `Vec<u8>`, because "the bytes" is only half the answer: a row's
    /// scalars ride the `@group(1)` uniform block, and a row that names a `Buf`
    /// param rides a `@group(0)` storage entry instead. A dispatch that
    /// flattened both into one byte run would be a dispatch whose caller has to
    /// ask the row the same question again.
    pub params: Params,
    /// Where in [`Self::buffers`] the caller's parameter struct goes.
    ///
    /// [`ParamSlot::Storage`] states a BINDING index; this states an index into
    /// the dense list the device half writes a bind group from, which skips the
    /// module's unread bindings. The two agree on every module in this tree,
    /// and [`plan_one`] refuses rather than assume it.
    ///
    /// `None` for the ordinary case where the scalars are the uniform block:
    /// that buffer is a bind group of its own and takes no place in this list.
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
        /// How many the module's uniform block has room for.
        room: usize,
    },
    /// The row names a scalar the driver cannot work out.
    ///
    /// Forwarded from [`crate::binding::Misplaced::Unresolved`], which says
    /// why this is a refusal and not a zero. Carries the symbol as well as the
    /// operand, because the row is where the repair is: either the source is
    /// one `binding::scalars` should derive, or the row is naming the wrong
    /// one.
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
    /// A crossed routine could not be planned.
    ///
    /// Carries the reason as a STRING rather than the `Unplanned` itself,
    /// because `Unplanned` holds an `Unbindable` and this enum is compared
    /// for equality across the suite.
    Routine {
        /// The entrypoint.
        symbol: String,
        /// What the routine plane said.
        why: String,
    },
    /// A body stated a number of dispatches this path cannot carry.
    ///
    /// One launch, one `Dispatch`. A two-pass reduction is two entrypoints
    /// over one statement and would need the launch path to widen first.
    Multiple {
        /// The entrypoint.
        symbol: String,
        /// How many the body asked for.
        stated: usize,
    },
    /// The rule and dims do not describe a grid.
    Ungeometric {
        /// Why.
        why: Ungeometric,
    },
    /// The rectangle sits under a guard nothing has evaluated.
    ///
    /// Not a defect: a conditional arm is a launch a caller records only after
    /// deciding the branch. Refused rather than recorded because recording
    /// every arm would RUN every arm.
    Conditional {
        /// The symbol the guarded rectangle names.
        symbol: String,
        /// Which conditional region, as an index into `Lowered::conds`.
        cond: u32,
    },
    /// The plan states a different number of operands than the module binds.
    ///
    /// Almost always the plan being short of a resource the driver is expected
    /// to own -- a paged KV cache, a page table -- rather than a mistake.
    /// Refused because `wgpu` validates a bind group against its layout entry
    /// for entry, so a set that is short is rejected at encode time with
    /// nothing about the launch in the message, and one that is long would put
    /// a tensor at a slot on the theory that nothing looks there.
    Arity {
        /// The symbol whose module disagrees.
        symbol: String,
        /// How many operands the plan states.
        stated: usize,
        /// How many bindings the module really has: its layout less its unread
        /// slots.
        module: usize,
    },
    /// The grid computed to zero in some dimension.
    ///
    /// Legal WebGPU and always a defect: it runs nothing, reports success, and
    /// leaves the output holding whatever it held before. Refused here rather
    /// than at the device so that the traced op is still in hand to name.
    Empty {
        /// The grid that would have been dispatched.
        groups: [u32; 3],
    },
}

impl std::fmt::Display for Undispatchable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { symbol } => write!(f, "no kernel row states `{symbol}`"),
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
            Self::Routine { symbol, why } => {
                write!(f, "`{symbol}` could not be planned as a routine: {why}")
            }
            Self::Multiple { symbol, stated } => write!(
                f,
                "`{symbol}`'s body stated {stated} dispatches and one launch \
                 carries one"
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

/// The scalar at `index` of this launch's own run, when it states one there and
/// it is not zero.
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
/// `grid_param` fires 1788 times over the Vulkan port's six texts, and a driver
/// that read only the fire would normalise over the wrong width every one of
/// those times and produce numbers rather than an error.
///
/// `head_param` and `heads_param` are the two whose absence hid for longer than
/// it looked, and the way it hid is worth keeping. Counting rows that STATE a
/// head shape does not witness these lines USING it: for those texts the stated
/// value EQUALS the fire's, so the overrides are no-ops there and a no-op
/// cannot be caught by a test that only checks the answer. The model that
/// separates them is gemma-4, whose full-attention layers carry four 512-wide
/// KV heads against its sliding layers' sixteen 256-wide ones. A walk that
/// hands the same launch a geometry disagreeing on purpose is what makes
/// deleting either line fail; recorded rather than tidied away because "it is
/// counted" read as "it is checked" for two commits, and those are different
/// claims.
#[must_use]
pub fn dims_of(sig: &KernelSig, lowered: &Lowered, launch: &Launch, fire: Geometry) -> Dims {
    // The last widthed operand, which is the launch's last OUTPUT: what sizes
    // the rectangle for nearly every rule.
    let width = widths(lowered, launch).next_back().unwrap_or(0);
    let extent = stated(lowered, launch, sig.grid_param);
    Dims {
        // The ROW COUNT the statement's own rectangle has, when the fire's row
        // window is not it. A mixture's sorted stack is one row per ROUTE, and
        // there are `tokens * experts_per_token` of those, so `route_gather`
        // states its rows as a param -- and given the fire's count instead it
        // gathers a quarter of its own output at `top_k = 4` and leaves the
        // rest whatever the arena held. See `kernels::KernelSig::rows_param`.
        rows: stated(lowered, launch, sig.rows_param)
            .unwrap_or(launch.rows.end - launch.rows.start),
        width,
        // The FIRST widthed operand, which is the first input. What sizes a
        // statement that reads one packed buffer and writes several, since
        // there no single output spells the grid.
        in_width: widths(lowered, launch).next().unwrap_or(0),
        q_heads: fire.q_heads,
        kv_heads: stated(lowered, launch, sig.heads_param).unwrap_or(fire.kv_heads),
        head_dim: stated(lowered, launch, sig.head_param).unwrap_or(fire.head_dim),
        // The SAME stated number, read by whichever rule asked for it: a rope's
        // rotated channels, a norm's reduction axis. A row names one scalar as
        // its extent and its rule knows which dimension that is.
        axis: extent.unwrap_or(width),
        rotary_dims: extent.unwrap_or(fire.rotary_dims),
        n_experts: fire.n_experts,
        experts_per_token: fire.experts_per_token,
    }
}

/// The module a launch will run, as the two things this file needs from it.
///
/// Grouped because they always travel together and always come from the same
/// parsed source: the workgroup size and tile decide the GRID, and the bindings
/// and unread slots decide the ARITY. Separating them at the call site would
/// let a caller pass a module's geometry with another module's layout, which is
/// a mistake nothing downstream could detect.
// `KernelSig` has no `Debug`, so this cannot derive one either.
#[derive(Clone, Copy)]
pub struct Built<'a> {
    /// Workgroup size and tile, from the entrypoint name and the module.
    pub module: Module,
    /// What the module binds.
    pub declared: &'a crate::reflect::Declared,
    /// The table row this symbol resolves to, if the caller has it.
    ///
    /// `None` means "look it up", which is what a test does and what any
    /// caller that has one symbol in hand should do. A caller with a LOOP over
    /// launches should pass it: `row` is `sig_in`, which walks the table for
    /// an exact match and then walks it AGAIN matching axis points, and a
    /// decode fires 452 launches over ten distinct symbols.
    ///
    /// `driver-vulkan` carries the same field for the same reason and states
    /// it in the same words -- "once per SYMBOL, not once per launch" -- which
    /// is where this was found: its `serve::fire` caches the reflection and the
    /// row together, and this backend had ported only the first half.
    ///
    /// # What it is worth here, which is not much
    ///
    /// The plan phase of a real decode: **2.0 ms to 1.3 ms**, interleaved, over
    /// 452 launches. That is 0.7 ms of a ~28 ms step -- about 2.5%, and well
    /// under this machine's run-to-run spread, which the same binary shows as
    /// 33 to 45 ms. It is kept because it is free and strictly less work, and
    /// because the alternative was to leave a `sig_in` walk in a loop after
    /// discovering the sibling had removed exactly that; it is NOT kept on a
    /// measured end-to-end win, and this says so rather than letting the next
    /// reader assume one.
    pub sig: Option<&'static KernelSig>,
}

/// Where a launch's operands are to be found.
///
/// Named `Sources` rather than the `Into` it started as in the Vulkan port,
/// because a public type called `Into` shadows `std::convert::Into` at every
/// use site that imports it.
#[derive(Debug)]
pub struct Sources<'a, R: Resolve> {
    /// The buffer the plan's offsets are into, and how much of it the plan was
    /// allowed to place in.
    pub arena: Arena<'a, R::Buffer>,
    /// What holds the weights and the seam values.
    pub resolver: &'a R,
    /// The device's `min_storage_buffer_offset_alignment`.
    pub min_offset: u64,
}

impl<R: Resolve> Clone for Sources<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Resolve> Copy for Sources<'_, R> {}

/// The row in `table` whose symbol the plan names.
///
/// [`kernels::sig_in`], not an equality test. A plan names the symbol it will
/// dispatch, and that symbol carries the variant suffixes the specialisation
/// axes append -- `_gs_64_b_4` for an affine group, `_bm_16_bn_32` for a routed
/// tile, `_d_128` for a head width. The TABLE states the axes, not the points,
/// so exact equality finds a row for the few symbols that have no axis and
/// nothing for the rest.
///
/// Measured on the Vulkan side before it was fixed: over the texts of the day,
/// exact matching found a row for 432 launches and failed on 3030, across
/// sixteen distinct symbols that all exist and all have modules built for them.
/// `sig_in` finds all of them.
fn row(table: &'static [KernelSig], symbol: &str) -> Option<&'static KernelSig> {
    kernels::sig_in(table, symbol)
}

/// `row`, for a caller that wants to resolve a symbol ONCE and hand the
/// answer to every launch that names it. See [`Built::sig`].
#[must_use]
pub fn row_of(table: &'static [KernelSig], symbol: &str) -> Option<&'static KernelSig> {
    row(table, symbol)
}

/// Turn one launch into a dispatch.
///
/// `table` is `kernels_wgpu::KERNELS` in every caller; it is a parameter so
/// that this module depends on the kernel *vocabulary* rather than on one
/// table, which is what lets a test state its own rows.
///
/// `built.module` describes the WGSL that will run: its workgroup size, and the
/// tile its name encodes. `built.declared` describes what that WGSL binds. Both
/// come from [`crate::reflect`], which parses the source `kernels-wgpu` holds --
/// so unlike its Vulkan counterpart this function needs no compiled artefact
/// and no device, and a whole plan can be planned on any machine.
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
) -> Result<Dispatch<'a, R::Buffer>, Undispatchable> {
    let Built { declared, .. } = built;
    let Sources {
        arena,
        resolver,
        min_offset,
    } = sources;
    let symbol = lowered.kernels[launch.kernel as usize].as_str();
    // A conditional rectangle's guard was NOT answered by the lowering, and
    // this walk has no way to answer it -- recording every arm would run every
    // arm. `driver-metal` refuses here for the same reason.
    if launch.cond != Launch::NO_COND {
        return Err(Undispatchable::Conditional {
            symbol: symbol.to_owned(),
            cond: launch.cond,
        });
    }
    // THE FORK. A symbol this backend has both crossed and ARMED is planned
    // by its routine: the arm finds the operands, the body states the module,
    // the entrypoint and the lanes, and neither reads the row's `operands` or
    // `launch` columns. One symbol qualifies today.
    //
    // ABOVE THE ROW LOOKUP, which is the point. An armed symbol needs no
    // `KernelSig` at all: the arm finds the operands, the body states the
    // grid, and the two widths a body's `Facts` want come off the LOWERING --
    // the last widthed operand is what the launch writes and the first is
    // what it reads. So a family whose arms are all in can have its rows
    // DELETED, which is Stage 3 and is what the countdown counts.
    if let Some((routine, arm)) = crate::lowering::routine::armed(symbol) {
        let facts = crate::lowering::arm::facts(
            symbol,
            launch.rows.end - launch.rows.start,
            fire,
            lowered.n_requests,
            widths(lowered, launch).next_back().unwrap_or(0),
            widths(lowered, launch).next().unwrap_or(0),
        );
        let mut planned = crate::lowering::routine::plan(
            routine,
            arm,
            lowered,
            launch,
            declared,
            Sources {
                arena,
                resolver,
                min_offset,
            },
            facts,
        )
        .map_err(|why| Undispatchable::Routine {
            symbol: symbol.to_owned(),
            why: why.to_string(),
        })?;
        // A body MAY state more than one dispatch -- a two-pass reduction is
        // two entrypoints over one statement -- and this function returns
        // one. Nothing in this tree does it, so the shape stays narrow and
        // the day something does, it is a named refusal rather than a
        // silently dropped pass.
        if planned.len() != 1 {
            return Err(Undispatchable::Multiple {
                symbol: symbol.to_owned(),
                stated: planned.len(),
            });
        }
        return Ok(planned.remove(0));
    }

    plan_by_row(lowered, launch, table, built, sources, fire)
}

/// One rectangle planned from its ROW, with the fork behind it.
///
/// The table path, split out of [`plan_one`] so it can be CALLED — by
/// `the_routine_path_plans_what_the_table_path_planned`, which derives every
/// field of a live kernel's dispatch twice and compares. There is no other way
/// to reach this once a symbol is armed, because the fork above answers first
/// and by design.
///
/// # Errors
///
/// Every variant of [`Undispatchable`] except `Routine` and `Multiple`, which
/// only the routine plane raises.
pub fn plan_by_row<'a, R: Resolve>(
    lowered: &'a Lowered,
    launch: &Launch,
    table: &'static [KernelSig],
    built: Built<'_>,
    sources: Sources<'a, R>,
    fire: Geometry,
) -> Result<Dispatch<'a, R::Buffer>, Undispatchable> {
    let symbol = lowered.kernels[launch.kernel as usize].as_str();
    let Built {
        module,
        declared,
        sig: stated,
    } = built;
    let Sources {
        arena,
        resolver,
        min_offset,
    } = sources;
    let sig = match stated {
        Some(sig) => sig,
        None => row(table, symbol).ok_or_else(|| Undispatchable::Unknown {
            symbol: symbol.to_owned(),
        })?,
    };

    // Not `bind`, which hands the operands over in the order the TRACE states
    // -- inputs, outputs, weights. The shader binds them in the order its
    // kernel row states, and for 2898 of the Vulkan port's 3992 rectangles
    // those differ. `rms_single_row` is the plainest: `norm/rms.wgsl` is
    // `0=x, 1=w, 2=out`, its row is `In(0), Weight(0), Out(0)`, and the trace
    // hands over `In(0), Out(0), Weight(0)` -- so positionally the norm reads
    // its own output as the weight and writes over the weight. A `wgpu` bind
    // group typed by the layout accepts every one of those shuffles, because
    // all four entries are storage buffers.
    let slots = reorder(sig, lowered, launch, arena, resolver, min_offset)
        .map_err(|(at, why)| Undispatchable::Operand { at, why })?;

    // Where the scalars go is the ROW's decision on this backend, which is the
    // simplification the launch ABI buys: `kernels_wgpu::bindings` states which
    // operand takes which entry, so a row that names a `Buf` param has already
    // said where its struct is and the reflection is a check on that rather
    // than the source of it. An UNSTATED row still falls back to the module.
    let placed = scalars(sig, lowered, launch, declared, resolver).map_err(|why| match why {
        crate::binding::Misplaced::Count {
            stated, uniform, ..
        } => Undispatchable::Scalars {
            stated,
            room: uniform,
        },
        crate::binding::Misplaced::Unresolved { at, name, source } => Undispatchable::Unresolved {
            symbol: symbol.to_owned(),
            at,
            name,
            source,
        },
        crate::binding::Misplaced::Contiguous { at, name } => Undispatchable::Contiguous {
            symbol: symbol.to_owned(),
            at,
            name,
        },
    })?;

    // The plan's operands, PLUS the slot a parameter STRUCT takes, have to be
    // the module's real bindings. Checked here so that a mismatch is refused
    // with the traced op in hand instead of becoming a `Dispatch` no device
    // accepts.
    //
    // The uniform block is deliberately not counted: it is `@group(1)`, a bind
    // group of its own, so it takes no place in the `@group(0)` numbering. That
    // is the whole reason this crate does not need `driver-vulkan`'s `+ 1`
    // correction -- there a scalar block is a descriptor in the same set, and
    // checking arity before placing the scalars refused 1439 rectangles across
    // nine symbols that are all perfectly dispatchable.
    let laid = descriptors(slots, &placed, declared).map_err(Undispatchable::Layout)?;
    if laid.len() != declared.bindings as usize {
        return Err(Undispatchable::Arity {
            symbol: symbol.to_owned(),
            stated: laid.len(),
            module: declared.bindings as usize,
        });
    }

    // The dense list, which is what a bind group is written from: a slot the
    // row leaves EMPTY takes no entry, and its binding number is simply absent
    // from the layout. WebGPU allows that where Vulkan does not, because a
    // `BindGroupLayoutEntry` carries an explicit `binding: u32` -- omitting one
    // shifts nothing.
    let block_at = laid
        .iter()
        .position(|s| matches!(s, Slot::Params))
        .map(|at| {
            laid[..at]
                .iter()
                .filter(|s| !matches!(s, Slot::Nothing))
                .count()
        });
    let buffers: Vec<Bound<'a, R::Buffer>> = laid
        .into_iter()
        .filter_map(|s| match s {
            Slot::Buffer(b) => Some(b),
            Slot::Params | Slot::Nothing => None,
        })
        .collect();
    // Every binding the SHADER READS has to be filled, and this is the count
    // form of that claim. It is an INEQUALITY, and the direction is the whole
    // content of the line.
    //
    // `driver-vulkan` asserts equality here -- `buffers + block == bindings -
    // holes` -- and that equality is a Vulkan coincidence, not a rule. There a
    // "hole" is a binding number with NO declaration, because `glslc` deletes
    // the declaration of a buffer a variant never reads; the descriptor set
    // still needs a slot at every number up to the highest, the plan has
    // nothing to put in it, and the row says `Unbound` at exactly those
    // positions. So `Nothing`-slots and holes are the same slots, and equality
    // holds.
    //
    // `naga` deletes nothing. A hole here means DECLARED AND NOT READ BY THIS
    // ENTRY POINT, which is a different fact: the binding exists, the row names
    // a real tensor for it, and the driver binds one. Measured over the
    // complete tree, 19 entrypoints have such a slot -- `kv_append_paged` keeps
    // six ring-ABI placeholders, and every `sdpa_paged_*` declares an
    // attention-sink buffer its non-sink variants do not read -- and 13 of them
    // belong to rows that state operands. Carrying the equality across refused
    // all thirteen, including every `sdpa_paged_decode` and `router_topk`: the
    // paged decode step and every mixture layer, which is to say most of what a
    // model runs.
    //
    // So the check that survives is the useful half. A dispatch may bind MORE
    // than the shader reads -- that is a layout entry the body ignores, and it
    // costs a descriptor. It may not bind FEWER, because an entry the shader
    // reads and the group does not carry is a `wgpu` validation failure at
    // encode time, with a number in the message and no launch.
    //
    // Dormant today, and stated rather than deleted for that reason: it can
    // only fire when a slot comes back `Nothing`, and no row in this table
    // states `Source::Unbound`. `descriptors` refuses the same thing per slot
    // and by NAME (`Unlayoutable::Unfilled`), which is the better message; this
    // is the arithmetic backstop for a row that grows a gap.
    let reads = declared.bindings as usize - declared.holes();
    if buffers.len() + usize::from(block_at.is_some()) < reads {
        return Err(Undispatchable::Arity {
            symbol: symbol.to_owned(),
            stated: buffers.len() + usize::from(block_at.is_some()),
            module: reads,
        });
    }

    let dims = dims_of(sig, lowered, launch, fire);

    let groups =
        groups(sig.launch, dims, module).map_err(|why| Undispatchable::Ungeometric { why })?;
    if groups.contains(&0) {
        return Err(Undispatchable::Empty { groups });
    }

    Ok(Dispatch {
        symbol,
        buffers,
        params: placed,
        block_at,
        groups,
        op: launch.op,
    })
}

impl<B> Dispatch<'_, B> {
    /// Which bind group the parameter buffer belongs to, if there is one.
    ///
    /// A convenience the device half would otherwise write at its one call
    /// site, and it is here because getting it wrong is silent: binding the
    /// uniform block into `@group(0)` would shift every storage entry after it
    /// and `wgpu` would accept the set if the kinds happened to line up.
    #[must_use]
    pub fn param_slot(&self) -> Option<ParamSlot> {
        match &self.params {
            Params::Block { at, .. } => Some(*at),
            Params::None => None,
        }
    }
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::binding::Placeholder;
    use crate::geometry::Local;
    use model_ir::trace::ValueId;

    #[derive(Default)]
    struct Store {
        weights: BTreeMap<String, Placeholder>,
        /// One buffer standing in for every piece of fire state: the cache
        /// sides and the per-fire tables. `None` by default, so a test that
        /// wants the refusal still gets it.
        state: Option<Placeholder>,
    }

    /// A row that states no operands, so that the plan-order fallback can be
    /// exercised end to end.
    ///
    /// A table of this module's own, which is exactly what `table` being a
    /// parameter is for. It has to be invented, and the reason is a fact about
    /// this backend's table worth stating plainly: **all 56 rows that state no
    /// operands also state no launch RULE**, so no real one of them reaches
    /// the grid. `driver-metal`'s fallback is about the operand ORDER and this
    /// crate carries it -- see
    /// `an_unstated_rows_operands_come_from_the_plan_and_its_grid_does_not` --
    /// but a row that says nothing about its grid cannot be launched by
    /// guessing one, and [`crate::geometry::Ungeometric::Unstated`] is where
    /// that is refused.
    static PLAIN: &[KernelSig] = &[KernelSig {
        name: "plan_order",
        symbol: "plan_order",
        file: None,
        launch: Rule::Elementwise,
        whole: false,
        lacks: &[],
        sink: None,
        in_place: &[],
        depth_prefix_plan: false,
        args: &[],
        operands: &[],
        axes: &[],
        grid_param: None,
        head_param: None,
        heads_param: None,
        rows_param: None,
    }];

    /// A row with every column at rest, for a test to state the two or three
    /// its claim is about.
    ///
    /// SIXTEEN columns, of which any one check reads a handful. The rest used
    /// to be borrowed from a real row -- `..*base`, over a
    /// `KERNELS.iter().find(..)` -- and that idiom died with the table: the
    /// find returns `None` now and every test using it panicked in its setup
    /// rather than in its assertion. This is that base STATED instead of
    /// borrowed, which is also the honest shape, because none of the borrowed
    /// columns was ever the thing under test.
    ///
    /// `Rule::Unstated` and no operands are the resting values on purpose:
    /// both are the "the row has not said" reading, so a test that forgets to
    /// state one is refused by name rather than launched over a default
    /// somebody picked.
    const BARE: KernelSig = KernelSig {
        name: "bare",
        symbol: "bare",
        file: None,
        launch: Rule::Unstated,
        whole: false,
        lacks: &[],
        sink: None,
        in_place: &[],
        depth_prefix_plan: false,
        args: &[],
        operands: &[],
        axes: &[],
        grid_param: None,
        head_param: None,
        heads_param: None,
        rows_param: None,
    };

    impl Resolve for Store {
        type Buffer = Placeholder;
        fn weight(&self, name: &str) -> Option<&Placeholder> {
            self.weights.get(name)
        }
        fn named(&self, _value: ValueId) -> Option<&Placeholder> {
            None
        }
        // A resolver that holds every piece of fire state, so a row naming the
        // cache or a table is answered rather than refused for the driver not
        // having built one. `Placeholder` is a size and nothing else, which is
        // all binding reads.
        fn kv(&self, _layer: u16, _values: bool) -> Option<&Placeholder> {
            self.state.as_ref()
        }
        fn number(&self, _which: crate::binding::FireNumber) -> Option<u32> {
            Some(16)
        }
        fn table(&self, _which: crate::binding::FireTable) -> Option<&Placeholder> {
            self.state.as_ref()
        }
    }

    /// A module declaring `bindings` storage entries and a uniform block with
    /// `uniform` members.
    fn declared(bindings: u32, uniform: &[u32]) -> crate::reflect::Declared {
        crate::reflect::Declared {
            local: [256, 1, 1],
            bindings,
            used: vec![true; bindings as usize],
            reads_workgroup_count: false,
            grid_axes: [true, false, false],
            uniform_offsets: uniform.to_vec(),
            uniform_bytes: uniform
                .iter()
                .map(|o| (o + 4).next_multiple_of(16))
                .max()
                .unwrap_or(0),
            block_bytes: vec![None; bindings as usize],
        }
    }

    /// A plan of one rectangle over `symbol`, with `args` operands.
    /// An ARMED symbol is planned by its routine, and the row's columns are
    /// not what decided the answer.
    ///
    /// The fork's whole claim, and it is now as strong as it can get:
    /// `argmax_logits` HAS NO ROW. `sample`'s `kernel!` rows came off when its
    /// arm landed, so `kernels_wgpu::sig` returns `None` and there is nothing
    /// for the table path to read — not a row without a rule, no row. A grid
    /// coming back is therefore proof the routine path ran, because the other
    /// path cannot even begin.
    ///
    /// Its ENTRYPOINT is still stated, by `kernels_wgpu::RETIRED`, so every
    /// sweep still compiles and covers the shader. Losing the row is not
    /// losing the kernel.
    ///
    /// The first assertion below pins that, so if a row is ever restored this
    /// test says the reasoning changed rather than passing on the old one.
    ///
    /// The grid is `256` lanes over one workgroup of 256, which is
    /// `sample/argmax.wgsl`'s `@workgroup_size(256)` and the body's own
    /// `rows` extent.
    #[test]
    fn an_armed_symbol_is_planned_by_its_routine_and_not_by_its_row() {
        let lowered = plan(
            "argmax_logits",
            vec![
                arena_arg(0),
                arena_arg(1024),
                arena_arg(2048),
                arena_arg(4096),
            ],
            Vec::new(),
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(0, &[]);

        // The table path cannot do this: there is no row to read.
        assert!(
            kernels_wgpu::sig("argmax_logits").is_none(),
            "a row for `argmax_logits` is back, so a grid no longer proves \
             which path answered"
        );

        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            kernels_wgpu::KERNELS,
            Built {
                module: Module {
                    local: Local([256, 1, 1]),
                    tile: None,
                },
                declared: &d,
                sig: None,
            },
            Sources {
                arena: Arena {
                    buffer: &buf,
                    bytes: 1 << 20,
                },
                resolver: &store,
                min_offset: 256,
            },
            Geometry::default(),
        )
        .expect("the routine path planned it");

        assert_eq!(got.symbol, "argmax_logits");
        // The body states `[GROUP_X, rows, 1]` lanes -- one workgroup of 256
        // reduces ONE ROW over the vocabulary -- so four rows is four
        // workgroups on y and one on x. I guessed `[1, 1, 1]` writing this
        // and the fork answered `[1, 4, 1]`; the body was right and the guess
        // was a grid that would have reduced row zero four times.
        assert_eq!(got.groups, [1, 4, 1]);
        // Four operands, in the order the BODY asked -- logits, next_token,
        // params, eos_flag -- which is in, out, in, out against a statement
        // that states its reads first.
        assert_eq!(got.buffers.len(), 4);
        assert_eq!(got.op, 11);
    }

    /// The second armed symbol halves its width, and the fork carries it.
    ///
    /// `ptir::copy_logits_bf16` is the only place on this backend where the
    /// grid is not a function of the operand width the way every row states
    /// it: `logits_copy.wgsl` packs two bf16 into a `u32`, so one lane owns
    /// one WORD and the x extent is half the vocabulary. `kernels-metal`'s row
    /// for the same kernel states the unhalved `[vocab, rows, 1]` and is right
    /// about ITS shader, which has a 16-bit type.
    ///
    /// So this is the fact worth a test of its own: a width arriving whole and
    /// a grid coming back halved. Nothing else would notice if the body
    /// dropped the `/ 2` — the dispatch would simply write the top half of
    /// every row and leave the rest, which is a wrong answer that still runs.
    #[test]
    fn the_logits_copy_grid_is_half_the_width_it_was_given() {
        // NOT `arena_arg`, whose width is 64: 64 lanes and 32 both fit one
        // workgroup of 256, so the halved and unhalved grids are the SAME
        // `[1, rows, 1]` and the first version of this test passed either
        // way. 1024 is the smallest round width where they differ -- 4
        // workgroups against 2 -- which is the whole point of it.
        let wide = |at: usize| Arg::Arena {
            at,
            width: 1024,
            bytes: 2,
        };
        let lowered = plan(
            "copy_logits_bf16",
            vec![wide(0), wide(4096), wide(8192)],
            Vec::new(),
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(0, &[]);

        assert!(
            kernels_wgpu::sig("copy_logits_bf16").is_none(),
            "a row for `copy_logits_bf16` is back, so a grid no longer proves \
             which path answered"
        );

        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            kernels_wgpu::KERNELS,
            Built {
                module: Module {
                    local: Local([256, 1, 1]),
                    tile: None,
                },
                declared: &d,
                sig: None,
            },
            Sources {
                arena: Arena {
                    buffer: &buf,
                    bytes: 1 << 20,
                },
                resolver: &store,
                min_offset: 256,
            },
            Geometry::default(),
        )
        .expect("the routine path planned it");

        assert_eq!(got.symbol, "copy_logits_bf16");
        // 1024 / 2 = 512 words over a workgroup of 256, so TWO on x. Four on
        // y is the launch's rows. Unhalved it would be four on x, and the
        // shader would read past the row it was given.
        assert_eq!(got.groups, [2, 4, 1]);
        // source, destination, params -- in, out, in, against a statement
        // that states its reads first. The two U32 arguments the arm appends
        // are widths, not buffers, and do not bind.
        assert_eq!(got.buffers.len(), 3);
    }

    /// An odd vocabulary is refused by the BODY, through the fork.
    ///
    /// The body's own header requires `vocab` to be even, because an odd pitch
    /// starts the next row inside the previous row's last word. That refusal
    /// is a fact about the shader that no row had a column for, and this is
    /// the first time this backend can state one: the table path would have
    /// dispatched an odd width and written the rows into each other.
    #[test]
    fn an_odd_vocabulary_is_refused_where_the_row_had_no_column_for_it() {
        let lowered = plan(
            "copy_logits_bf16",
            vec![
                Arg::Arena {
                    at: 0,
                    width: 63,
                    bytes: 2,
                },
                Arg::Arena {
                    at: 1024,
                    width: 63,
                    bytes: 2,
                },
                Arg::Arena {
                    at: 2048,
                    width: 63,
                    bytes: 2,
                },
            ],
            Vec::new(),
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(0, &[]);
        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            kernels_wgpu::KERNELS,
            Built {
                module: Module {
                    local: Local([256, 1, 1]),
                    tile: None,
                },
                declared: &d,
                sig: None,
            },
            Sources {
                arena: Arena {
                    buffer: &buf,
                    bytes: 1 << 20,
                },
                resolver: &store,
                min_offset: 256,
            },
            Geometry::default(),
        );
        let Err(Undispatchable::Routine { symbol, why }) = got else {
            panic!("expected the body to refuse an odd width, got {got:?}");
        };
        assert_eq!(symbol, "copy_logits_bf16");
        assert!(
            why.contains("vocab"),
            "the refusal should name the width it refused: {why}"
        );
    }

    /// A routine that cannot be planned is `Undispatchable::Routine`, and it
    /// carries what the routine plane said.
    ///
    /// The armed symbol against a statement of no operands: the arm asks for
    /// its first input and the statement has none, so the refusal comes back
    /// from the arm rather than from anything the table path would have said.
    #[test]
    fn a_routine_that_cannot_be_planned_is_refused_by_name() {
        let lowered = plan("argmax_logits", Vec::new(), Vec::new());
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(0, &[]);
        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            kernels_wgpu::KERNELS,
            Built {
                module: Module {
                    local: Local([256, 1, 1]),
                    tile: None,
                },
                declared: &d,
                sig: None,
            },
            Sources {
                arena: Arena {
                    buffer: &buf,
                    bytes: 1 << 20,
                },
                resolver: &store,
                min_offset: 256,
            },
            Geometry::default(),
        );
        let Err(Undispatchable::Routine { symbol, why }) = got else {
            panic!("expected a routine refusal, got {got:?}");
        };
        assert_eq!(symbol, "argmax_logits");
        assert!(
            why.contains("an input"),
            "the refusal should carry what the arm said, got `{why}`"
        );
    }

    fn plan(symbol: &str, args: Vec<Arg>, params: Vec<u32>) -> Lowered {
        Lowered {
            launches: vec![Launch {
                kernel: 0,
                rows: 0..4,
                layers: 0..1,
                op: 11,
                args: 0..args.len() as u32,
                params: 0..params.len() as u32,
                peel: None,
                cond: Launch::NO_COND,
            }],
            kernels: vec![symbol.to_owned()],
            rectangles: 1,
            arena_bytes: 1 << 20,
            value_offset: Vec::new(),
            value_owner: Vec::new(),
            epilogue_gather: usize::MAX,
            epilogue_norm: usize::MAX,
            args,
            structural: Vec::new(),
            residue: Vec::new(),
            params,
            n_requests: 1,
            conds: Vec::new(),
            readout: None,
        }
    }

    /// A row whose scalars the module has no room for is refused by name.
    ///
    /// From the census in `tests/citations.rs`: `Undispatchable::Scalars` is
    /// one of the refusals this crate builds and no test named. It is the
    /// translation of `binding::Misplaced::Count`, which IS tested one layer
    /// down -- so what this adds is the three-arm `match` between them, where a
    /// swapped arm compiles and hands back a refusal describing a different
    /// fault entirely.
    ///
    /// `Misplaced::Count`'s own doc says why the underlying defect has no
    /// symptom: WGSL requires every access to be bounds-checked, so a block
    /// short of what the shader reads returns ZEROS rather than faulting, and
    /// a missing pitch is "a plausible number that no layer and no assertion
    /// will object to". The refusal is the only thing that ever says so, which
    /// is the argument for asserting it by name rather than by `is_err`.
    ///
    /// `shared_expert_combine_strided` was the row: the smallest stated one
    /// whose buffers a default resolver could satisfy -- four arena operands,
    /// `In` and `Out` only, no weights, no fire tables, no cache -- and it
    /// stated two scalars. A module declaring room for one is the mismatch.
    #[test]
    fn a_module_with_no_room_for_the_rows_scalars_says_how_many_of_each() {
        // Through `plan_by_row`, not `plan_one`: this is a claim about the
        // TABLE path's refusal, and every family is armed now, so `plan_one`
        // would answer from the routine plane and the refusal under test would
        // never be reached.
        //
        // SYNTHESIZED, and the row is now synthesized WHOLE. It was
        // `shared_expert_combine_strided`, then a real row's columns with an
        // invented operand list spread over them (`..*base`); the table is
        // empty, so there is no base to spread and no row to pick. The claim
        // needs no particular family to make it: `scalars` refuses a run that
        // does not fit the module's block -- two scalars against a block of
        // one -- and a row is data, so a test that needs one may write one.
        let two: &'static [kernels::Operand] = Box::leak(
            vec![
                kernels::Operand {
                    name: "x",
                    ty: kernels::Ty::Buf,
                    source: kernels::Source::In(0),
                    nullable: false,
                },
                kernels::Operand {
                    name: "out",
                    ty: kernels::Ty::BufMut,
                    source: kernels::Source::Out(0),
                    nullable: false,
                },
                kernels::Operand {
                    name: "a",
                    ty: kernels::Ty::I32,
                    source: kernels::Source::Param(0),
                    nullable: false,
                },
                kernels::Operand {
                    name: "b",
                    ty: kernels::Ty::I32,
                    source: kernels::Source::Param(1),
                    nullable: false,
                },
            ]
            .into_boxed_slice(),
        );
        let synthetic: &'static KernelSig = Box::leak(Box::new(KernelSig {
            name: "scalar_room",
            symbol: "scalar_room",
            // A rule, because the premise below has to PLAN: a row left at
            // `Rule::Unstated` would be refused for its grid and the refusal
            // under test would never be reached. `Elementwise` over a 64-wide
            // rectangle of four rows is one workgroup.
            launch: Rule::Elementwise,
            operands: two,
            ..BARE
        }));
        let symbol = synthetic
            .entrypoints()
            .into_iter()
            .next()
            .expect("the row has entrypoints");
        let lowered = plan(
            &symbol,
            vec![
                arena_arg(0),
                arena_arg(1024),
                arena_arg(2048),
                arena_arg(4096),
            ],
            vec![3, 5],
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();

        let plan_against = |uniform: &[u32]| {
            let d = declared(2, uniform);
            plan_by_row(
                &lowered,
                &lowered.launches[0],
                kernels_wgpu::KERNELS,
                Built {
                    module: Module {
                        local: Local([256, 1, 1]),
                        tile: None,
                    },
                    declared: &d,
                    sig: Some(synthetic),
                },
                Sources {
                    arena: Arena {
                        buffer: &buf,
                        bytes: 1 << 20,
                    },
                    resolver: &store,
                    min_offset: 256,
                },
                fire(),
            )
            .map(|_| ())
        };

        // The premise: with room for both, this rectangle plans. Without it
        // the refusal below could be about anything.
        plan_against(&[0, 4]).expect("a module with room for both scalars");

        let why = plan_against(&[0]).expect_err("a module with room for one");
        assert!(
            matches!(why, Undispatchable::Scalars { stated: 2, room: 1 }),
            "a row of two scalars against a block of one came back as `{why}`, \
             which does not say that the shader would have read a zero"
        );
    }

    fn arena_arg(at: usize) -> Arg {
        Arg::Arena {
            at,
            width: 64,
            bytes: 2,
        }
    }

    fn fire() -> Geometry {
        Geometry {
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            rotary_dims: 128,
            n_experts: 64,
            experts_per_token: 8,
        }
    }

    /// A whole rectangle becomes a dispatch, in the ROW's binding order.
    ///
    /// `rms_single_row` is the case the reorder exists for: the shader is
    /// `x, w, out, params`, the trace hands over `in, out, weight`, and binding
    /// positionally makes the norm read its own output as the gain.
    #[test]
    fn a_rectangle_binds_in_the_rows_order_and_not_the_traces() {
        // `rms_single_row` was the case this was written for and `norm` has
        // retired; `combine_sorted` stood in after it -- its row was
        // `In(0), In(1), Out(0), Param(0), In(2)` against a trace stating its
        // three reads before its write -- and `moe` retired too.
        //
        // SYNTHESIZED, and now synthesized WHOLE: the invented operand list
        // used to be spread over a real row's remaining columns (`..*base`),
        // and there is no longer a row to take them from. The claim is that
        // `reorder` binds in the ROW's order and not the trace's, and the
        // shape that shows it is a row stating `In(0), Out(0), In(1)` against
        // a trace that hands over its reads and then its write: positionally
        // the shader would be given the second read where its output belongs
        // and would write the answer into a buffer nobody reads. Both are
        // storage buffers of the same length, so nothing downstream sees it.
        let shuffled: &'static [kernels::Operand] = Box::leak(
            vec![
                kernels::Operand {
                    name: "a",
                    ty: kernels::Ty::Buf,
                    source: kernels::Source::In(0),
                    nullable: false,
                },
                kernels::Operand {
                    name: "out",
                    ty: kernels::Ty::BufMut,
                    source: kernels::Source::Out(0),
                    nullable: false,
                },
                kernels::Operand {
                    name: "b",
                    ty: kernels::Ty::Buf,
                    source: kernels::Source::In(1),
                    nullable: false,
                },
            ]
            .into_boxed_slice(),
        );
        let sig: &'static KernelSig = Box::leak(Box::new(KernelSig {
            // A symbol no stem in `lowering::arm::LIVE` claims, so `plan_one`
            // takes the table path. An armed one would be answered by its
            // routine and this row would never be read.
            name: "row_order",
            symbol: "row_order",
            launch: Rule::Elementwise,
            operands: shuffled,
            ..BARE
        }));
        let symbol = sig
            .entrypoints()
            .into_iter()
            .next()
            .expect("the row has entrypoints");
        // The trace's order: three reads, then the write.
        let lowered = plan(
            &symbol,
            vec![
                arena_arg(0),
                arena_arg(1024),
                arena_arg(2048),
                arena_arg(4096),
            ],
            vec![7, 8, 9, 10, 11],
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();

        let d = declared(3, &[]);
        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            kernels_wgpu::KERNELS,
            Built {
                module: Module {
                    local: Local([256, 1, 1]),
                    tile: None,
                },
                declared: &d,
                sig: Some(sig),
            },
            Sources {
                arena: Arena {
                    buffer: &buf,
                    bytes: 1 << 20,
                },
                resolver: &store,
                min_offset: 256,
            },
            fire(),
        )
        .expect("the rectangle dispatches");

        assert_eq!(got.op, 11, "the traced op travels with the dispatch");
        // Three buffers for three slots: this row states no parameter operand,
        // so nothing takes a `@group(0)` entry the plan did not name.
        assert_eq!(got.buffers.len(), 3);
        assert_eq!(
            got.block_at, None,
            "this synthetic row states no params struct"
        );

        // In(0), Out(0), In(1) -- the ROW's order. Positionally the trace
        // would have given 0, 1024, 2048: its two reads and then its write.
        assert_eq!(got.buffers[0].offset(), 0);
        assert_eq!(got.buffers[1].offset(), 4096, "the OUTPUT, second");
        assert_eq!(got.buffers[2].offset(), 1024, "and the second input, last");
        assert!(got.groups.iter().all(|n| *n >= 1));
    }

    /// An UNSTATED row is launchable, from the plan's own order.
    ///
    /// The fallback `driver-metal` carries and the Vulkan notes insist on: 56
    /// rows over 292 entrypoints state no operands, and they include
    /// `affine_qmm_t`, `sdpa_paged_tiled`, `gdn_core` and `argmax_logits` --
    /// most of what a model runs. Treating them as unlaunchable would be
    /// treating the backend as unusable.
    #[test]
    fn an_unstated_row_dispatches_from_the_plans_own_argument_order() {
        let lowered = plan("plan_order", vec![arena_arg(0), arena_arg(512)], Vec::new());
        let buf = Placeholder(1 << 20);
        let store = Store::default();

        let d = declared(2, &[]);
        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            PLAIN,
            Built {
                module: Module::new([256, 1, 1]),
                declared: &d,
                sig: None,
            },
            Sources {
                arena: Arena {
                    buffer: &buf,
                    bytes: 1 << 20,
                },
                resolver: &store,
                min_offset: 256,
            },
            fire(),
        )
        .expect("an unstated row is not an unbindable one");
        assert_eq!(
            got.buffers.iter().map(Bound::offset).collect::<Vec<_>>(),
            [0, 512],
            "the plan's order, which is the only order there is"
        );
        assert_eq!(got.params, Params::None);
        assert_eq!(got.block_at, None);
    }

    // RETIRED: THE TABLE IS EMPTY, so the walk has nothing to walk.
    //
    // It asserted the whole of `.wiki/new-driver/vulkan.md` §13's claim over
    // the set rather than over one row, and against the real WGSL rather than
    // against a `Declared` a test invented: `kernels-wgpu` embeds its shader
    // tree, so `naga` handed this walk the same module a fire would dispatch,
    // with no build product, no adapter and no fixture. For every entrypoint
    // of every row that stated NO operands -- 56 rows over 292 entrypoints
    // when the port landed, `affine_qmm_t_bias`, `sdpa_paged_tiled`,
    // `gdn_core` and `argmax_logits` among them, which is most of what a model
    // runs -- it built one arena operand per `@group(0)` binding the module
    // declares and exactly as many scalar words as the module's uniform block
    // has members, and demanded two things at once: that every one of them got
    // all the way through `reorder`'s plan-order fallback, `params_from`'s
    // module-driven placement and `descriptors`' layout check, and that every
    // one then stopped at the GRID, by name, with `Ungeometric::Unstated`,
    // because in this table an operand-less row was also a rule-less one. Two
    // different gaps with two different fixes; conflating them would have sent
    // the next reader to `operands` when the repair was `launch`.
    //
    // IT DID NOT BECOME TRUE. IT STOPPED LOOKING. The `for sig in KERNELS`
    // walks nothing, `wrong` is empty because it was never appended to, and
    // the closing `assert_eq!(bound, 7)` -- which existed precisely so the
    // denominator could not drift in silence -- is the only line that still
    // fires, against a count of zero. That refusal to go vacuous is what
    // retires it rather than leaving it green over an empty set.
    //
    // The fallback itself is live and reachable, and two tests in this file
    // hold it: `an_unstated_row_dispatches_from_the_plans_own_argument_order`
    // binds a rectangle in the plan's own order through `PLAIN`, and
    // `an_unstated_rows_operands_come_from_the_plan_and_its_grid_does_not`
    // takes the same rectangle through `reorder` and then all the way to the
    // `Ungeometric::Unstated` refusal. Both state their row now instead of
    // picking one, which is the only thing the emptying really took.
    //
    // What a REAL launch's operands come from is no longer a row at all:
    // `crate::lowering::arm`'s `Handles::{input, output, weight}` mint them in
    // the order the body asks and record each as an `Asked`, and
    // `crate::lowering::routine::bind` turns those handles into `Placed`
    // bind-group entries against the module's own `Declared`. Its per-symbol
    // cover is `arm.rs`'s `handles_are_minted_in_the_order_the_body_asks` and
    // `a_statement_the_arm_cannot_fill_is_refused`; its whole-corpus cover is
    // `crates/driver-wgpu/tests/arena.rs`'s
    // `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`,
    // which puts every rectangle of every real lowering through `plan_one`.
    // Neither is a census over entrypoints: the only sweep that still touches
    // all 481 is `crates/driver-wgpu/tests/device.rs`'s
    // `every_entrypoint_in_the_tree_builds_a_pipeline_on_this_adapter`, and
    // building a pipeline is not binding one. That much of the coverage this
    // walk gave is simply gone, and saying so is better than naming a
    // replacement that does less.

    /// An unstated row binds, and is refused for its GRID.
    ///
    /// The precise version of the claim `.wiki/new-driver/vulkan.md` §13
    /// makes. A row with no operands -- there were 56 of them here, spanning
    /// `affine_qmm_t_bias`, `sdpa_paged_tiled`, `gdn_core` and
    /// `argmax_logits`, which is most of what a model runs -- is NOT
    /// unbindable: its operands come from the plan's own order and every one
    /// of them resolves. Every one of those 56 also stated no launch rule, so
    /// what stopped them was the grid, by name, at the last step. That is a
    /// different sentence from "unstated rows are unlaunchable" and it points
    /// at a different fix: fill in `launch`, not `operands`.
    ///
    /// SYNTHESIZED, because the table is empty and there is no unstated row
    /// left to pick. The row it picked was replaced four times as each became
    /// armable, and the last replacement is the one that says why picking was
    /// never the point: the claim is about `reorder` reading an EMPTY operand
    /// list and `groups` reading `Rule::Unstated`, and both of those are
    /// columns, not kernels. `BARE` is exactly a row that has said neither, so
    /// this states it rather than hunting for one.
    #[test]
    fn an_unstated_rows_operands_come_from_the_plan_and_its_grid_does_not() {
        let sig: &'static KernelSig = Box::leak(Box::new(KernelSig {
            name: "no_rule",
            symbol: "no_rule",
            ..BARE
        }));
        assert!(
            sig.operands.is_empty() && sig.launch == Rule::Unstated,
            "the row under test has to have said neither, or this checks \
             something else"
        );
        let lowered = plan(sig.symbol, vec![arena_arg(0), arena_arg(512)], Vec::new());
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };

        // The operands, on their own: the fallback resolves both, in the
        // plan's order, with nothing from the row.
        let slots = reorder(sig, &lowered, &lowered.launches[0], arena, &store, 256)
            .expect("the plan's order is an order");
        assert_eq!(slots.len(), 2);
        assert!(slots.iter().all(|s| matches!(s, Slot::Buffer(_))));

        // And the whole dispatch, which gets past the binding and stops at the
        // grid. The row is handed over rather than looked up, and the table
        // passed in is empty to say so: a lookup would answer
        // `Undispatchable::Unknown` and this test would never reach a grid to
        // be refused for.
        let d = declared(2, &[]);
        assert_eq!(
            plan_one(
                &lowered,
                &lowered.launches[0],
                &[],
                Built {
                    module: Module::new([256, 1, 1]),
                    declared: &d,
                    sig: Some(sig),
                },
                Sources {
                    arena,
                    resolver: &store,
                    min_offset: 256,
                },
                fire(),
            )
            .expect_err("no rule, no grid"),
            Undispatchable::Ungeometric {
                why: Ungeometric::Unstated
            }
        );
    }

    // RETIRED: THE TABLE IS EMPTY, so the walk has nothing to walk.
    //
    // It was the other half of the retired sweep over unstated entrypoints
    // above, and the one that found the defect it then existed to keep out.
    // Over every entrypoint of every row that STATED operands -- 189
    // entrypoints of 44 rows when it was written, 24 of 12 by the end --
    // against the real WGSL each expands to, it
    // asserted that the row's operand order, the module's `@group(0)` binding
    // count, the uniform block's member count and the launch rule's grid all
    // agree, computed from two places that never consult each other: the
    // `kernel!` rows in `kernels-wgpu/src/*.rs` and the shaders in
    // `kernels-wgpu/kernels/`. It also pinned the refusals as a SET rather
    // than asserting none, because two readings genuinely disagreed and that
    // was `kernels-wgpu`'s to settle rather than this crate's to paper over.
    //
    // THE DEFECT IT CAUGHT, kept because the reasoning outlives the walk.
    // `driver-vulkan` closes `plan_one` with `buffers + block == bindings -
    // holes()`, an EQUALITY, and that is a Vulkan coincidence rather than a
    // rule: `glslc` deletes the declaration of a buffer a variant never reads,
    // so a hole is a binding number with nothing at it, the row says `Unbound`
    // at exactly those positions, and the two counts move together. `naga`
    // deletes nothing. A hole here is a binding that EXISTS and that this
    // entry point does not read -- `sdpa_paged_decode` declares an
    // attention-sink buffer its non-sink variants ignore, `kv_append_paged`
    // keeps six ring-ABI placeholders, `router_topk`, `affine_qmv_routed` and
    // `geglu_tanh` each keep one -- so the row names a real tensor for it and
    // the driver binds one. Carrying the equality across refused thirteen
    // entrypoints with `Undispatchable::Arity`: every `sdpa_paged_decode`
    // variant and `router_topk`, which is the paged decode step and every
    // mixture layer. `plan_one` asks `>=` because of this walk, and the
    // inequality and its argument are still stated at that line.
    //
    // THE SIX IT REFUSED ON PURPOSE were `kv_append_bfloat16` and the five
    // `sdpa_vector_decode*` entrypoints: three rows that walk the KV cache
    // with a head stride and a sequence stride and no page table, against a
    // pool this driver lays out as `[page, token, head, dim]`. Handing them
    // those two numbers makes the launch SUCCEED against the wrong tokens --
    // real memory, no fault, fluent text -- and they were listed by name so a
    // fourth row growing the same dependency would fail here rather than join
    // a filter.
    //
    // IT DID NOT BECOME TRUE. IT STOPPED LOOKING. `planned` counts to zero,
    // `refused` is empty because nothing was ever pushed onto it, and the
    // `assert_eq!(planned, 18)` that existed to keep the denominator honest is
    // the one line left with anything to say. Worse than vacuous: the
    // two-places-that-never-consult-each-other premise cannot be restored,
    // because one of the two descriptions has been deleted. A routine states
    // its own module and entrypoint next to the arguments it binds, so there
    // is no second, positional description of the launch left to disagree with
    // the shader -- which is the point of the crossing and also exactly why
    // this check has nothing to compare.
    //
    // WHERE EACH HALF LIVES NOW.
    //
    // * The operands and the bind group: `crate::lowering::arm`'s `Handles`
    //   mints them in the order the body asks and records each as an `Asked`,
    //   and `crate::lowering::routine::bind` lays them out as `Placed` against
    //   the module's own `Declared`, refusing by name through `Unplanned::`
    //   `{Handle, Operand, Blocks, NoCache, Absent}`.
    // * The scalars against the block's room: `Unplanned::Scalars`, checked by
    //   `routine.rs`'s `scalars_wider_than_the_modules_block_are_refused_by_name`.
    // * The grid that must not be empty: `Unplanned::Refused` through the
    //   body's own extent, checked by `a_body_that_computed_a_zero_extent_is_refused`.
    // * The contiguous-cache refusal: `crate::lowering::arm::contiguous_pool`,
    //   which narrows the blanket table-path refusal by one notch -- a fire
    //   whose pool states a page size is refused, a fire whose pool has none is
    //   served -- and whose doc names the same three families this walk pinned.
    // * The sweep itself: `crates/driver-wgpu/tests/arena.rs`'s
    //   `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`,
    //   which puts every rectangle of every real lowering through `plan_one`,
    //   pins the refusals as a set and asserts no grid contains a zero. It is
    //   a walk over LAUNCHES rather than over entrypoints, so a shader no text
    //   names is not covered by it; the only sweep that still touches all 481
    //   is `crates/driver-wgpu/tests/device.rs`'s
    //   `every_entrypoint_in_the_tree_builds_a_pipeline_on_this_adapter`, and
    //   building a pipeline is not planning a dispatch.

    /// A guarded rectangle is refused, not recorded.
    #[test]
    fn a_conditional_rectangle_is_refused_rather_than_run() {
        let mut lowered = plan("plan_order", vec![arena_arg(0)], Vec::new());
        lowered.launches[0].cond = 3;
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(1, &[]);
        assert_eq!(
            plan_one(
                &lowered,
                &lowered.launches[0],
                PLAIN,
                Built {
                    module: Module::new([256, 1, 1]),
                    declared: &d,
                    sig: None,
                },
                Sources {
                    arena: Arena {
                        buffer: &buf,
                        bytes: 1 << 20
                    },
                    resolver: &store,
                    min_offset: 256,
                },
                fire(),
            )
            .expect_err("guarded"),
            Undispatchable::Conditional {
                symbol: "plan_order".into(),
                cond: 3
            }
        );
    }

    /// A symbol no row covers is named rather than guessed at.
    #[test]
    fn a_symbol_no_row_states_is_refused_by_name() {
        let lowered = plan("no_such_kernel_bfloat16", vec![arena_arg(0)], Vec::new());
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(1, &[]);
        assert_eq!(
            plan_one(
                &lowered,
                &lowered.launches[0],
                kernels_wgpu::KERNELS,
                Built {
                    module: Module::new([256, 1, 1]),
                    declared: &d,
                    sig: None,
                },
                Sources {
                    arena: Arena {
                        buffer: &buf,
                        bytes: 1 << 20
                    },
                    resolver: &store,
                    min_offset: 256,
                },
                fire(),
            )
            .expect_err("unknown"),
            Undispatchable::Unknown {
                symbol: "no_such_kernel_bfloat16".into()
            }
        );
    }

    /// The row's own scalar overrides the fire's number.
    ///
    /// `grid_param` is the one that fires 1788 times over real texts, and a
    /// driver that read only the fire would normalise over the wrong width
    /// every one of them and produce numbers rather than an error.
    #[test]
    fn a_row_that_names_its_extent_takes_it_from_the_statement() {
        // `rms_single_row` was this test's row until `norm` retired, then
        // `qmv_routed` until `moe` did. SYNTHESIZED now, and it loses nothing:
        // `dims_of` reads exactly two columns to answer this -- `grid_param`
        // and the widths the LOWERING states -- so the row was never more than
        // a carrier for `Some(1)`. This is `qmv_routed`'s shape transcribed:
        // `Rule::RoutedQmv`, and its extent at param 1, which is
        // `out_vec_size` -- the matvec's row axis, and NOT the output
        // rectangle's width, because a routed projection writes a whole
        // token's `k` results end to end.
        let sig = KernelSig {
            name: "stated_axis",
            symbol: "stated_axis",
            launch: Rule::RoutedQmv,
            grid_param: Some(1),
            ..BARE
        };
        // The statement's run: index 1 is 256 -- a head-wide axis under a
        // 64-wide rectangle.
        let lowered = plan(
            sig.symbol,
            vec![arena_arg(0), arena_arg(1024), Arg::Weight("gain".into())],
            vec![0, 256, 0, 0, 0],
        );
        let dims = dims_of(&sig, &lowered, &lowered.launches[0], fire());
        assert_eq!(dims.axis, 256, "the row's own scalar, not the row width");
        assert_eq!(dims.width, 64, "the last widthed operand");

        // And a statement that does not carry it falls back to the fire rather
        // than to zero, because a zero extent is a grid of nothing.
        let bare = plan(
            sig.symbol,
            vec![arena_arg(0), arena_arg(1024), Arg::Weight("gain".into())],
            Vec::new(),
        );
        assert_eq!(dims_of(&sig, &bare, &bare.launches[0], fire()).axis, 64);
    }

    /// The head overrides come from the statement when it has them.
    ///
    /// The two that were counted and not checked for two commits, so they are
    /// checked here against a fire that DISAGREES on purpose -- a geometry
    /// equal to the stated value would pass with either line deleted.
    #[test]
    fn a_row_that_names_its_head_shape_takes_it_from_the_statement() {
        // A row of this file's own, with the two head indices set. It was
        // always synthesized in part -- no real row ever stated either index,
        // and the point is to check that the LINES read them rather than to
        // find a row that does -- and the columns it borrowed from
        // `qmv_routed` were never read by `dims_of`. The table is empty, so
        // the borrowing half goes and the stating half stays.
        let sig = KernelSig {
            name: "stated_heads",
            symbol: "stated_heads",
            launch: Rule::RoutedQmv,
            head_param: Some(2),
            heads_param: Some(3),
            ..BARE
        };
        let lowered = plan(
            sig.symbol,
            vec![arena_arg(0)],
            // index 2 is a 512-wide head, index 3 is four of them -- gemma-4's
            // full-attention shape, against a fire stating 128 and 8.
            vec![0, 0, 512, 4],
        );
        let dims = dims_of(&sig, &lowered, &lowered.launches[0], fire());
        assert_eq!(dims.head_dim, 512, "the statement's, not the fire's 128");
        assert_eq!(dims.kv_heads, 4, "the statement's, not the fire's 8");
    }

    /// A rule a caller needs before it has the operands.
    ///
    /// `engine`'s wgpu backend is that caller: `every_launch_fits` asks
    /// [`rule_of`] for a symbol's rule so it can size the grid a prefill would
    /// dispatch, long before anything binds an operand.
    ///
    /// SYNTHESIZED, and the qualification is worth stating plainly: THE
    /// SHIPPED TABLE ANSWERS NOTHING NOW. `kernels_wgpu::KERNELS` is empty, so
    /// `rule_of(KERNELS, ..)` is `Undispatchable::Unknown` for every symbol
    /// this backend can dispatch, and the caller above skips every launch it
    /// asks about. That is a fact about the TABLE, and the table's own tests
    /// own it; what is checked here is the two things `rule_of` itself
    /// promises, which are unchanged and which `table` is a parameter
    /// precisely so a test can state.
    ///
    /// The row carries AXES on purpose. `rule_of` goes through
    /// [`kernels::sig_in`] and not through an equality test, and that is the
    /// half with a defect behind it: exact matching found a row for 432 of the
    /// Vulkan port's 3992 launches and reported "no kernel row" for sixteen
    /// symbols that all exist and all have modules. So the symbol asked about
    /// below is a POINT of the row's axes and not the row's own name, which is
    /// what a driver holds.
    #[test]
    fn a_symbols_rule_is_answerable_from_the_table_alone() {
        // `kernels-wgpu`'s `BF16` and `GROUP_64`/`BITS_4` spelled out, because
        // `crate::axes` is that crate's and this row is this file's.
        const AXES: &[kernels::Axis] = &[
            kernels::Axis {
                what: "activation dtype",
                points: &["_bfloat16"],
            },
            kernels::Axis {
                what: "affine group and width",
                points: &["_gs_64_b_4"],
            },
        ];
        let table: &'static [KernelSig] = Box::leak(
            vec![KernelSig {
                name: "rule_lookup",
                symbol: "rule_lookup",
                launch: Rule::RoutedQmv,
                axes: AXES,
                ..BARE
            }]
            .into_boxed_slice(),
        );
        let symbol = table[0]
            .entrypoints()
            .into_iter()
            .next()
            .expect("the row names its points");
        assert_eq!(
            symbol, "rule_lookup_bfloat16_gs_64_b_4",
            "the spelling a driver holds is the base plus every axis point"
        );
        assert_eq!(rule_of(table, &symbol).expect("stated"), Rule::RoutedQmv);
        // And the base itself, which is what a model text states.
        assert_eq!(
            rule_of(table, "rule_lookup").expect("stated"),
            Rule::RoutedQmv
        );
        // A symbol no row covers is named rather than rounded to the nearest
        // row -- including a half-spelled one, which is the case `covers_point`
        // peels its axes from the END to refuse.
        assert!(rule_of(table, "no_such_kernel").is_err());
        assert!(rule_of(table, "rule_lookup_bfloat16").is_err());
    }

    /// A launch whose module binds more than the plan states is refused.
    #[test]
    fn a_module_that_binds_more_than_the_plan_states_is_refused_by_arity() {
        let lowered = plan("plan_order", vec![arena_arg(0)], Vec::new());
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        // The module wants three and the plan carries one.
        let d = declared(3, &[]);
        let err = plan_one(
            &lowered,
            &lowered.launches[0],
            PLAIN,
            Built {
                module: Module::new([256, 1, 1]),
                declared: &d,
                sig: None,
            },
            Sources {
                arena: Arena {
                    buffer: &buf,
                    bytes: 1 << 20,
                },
                resolver: &store,
                min_offset: 256,
            },
            fire(),
        )
        .expect_err("short");
        assert_eq!(
            err,
            Undispatchable::Arity {
                symbol: "plan_order".into(),
                stated: 1,
                module: 3
            }
        );
    }

    /// An operand the resolver does not hold names the slot it would have gone
    /// in.
    #[test]
    fn an_unresolvable_operand_is_refused_with_its_slot() {
        let lowered = plan(
            "plan_order",
            vec![arena_arg(0), Arg::Weight("absent".into())],
            Vec::new(),
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(2, &[]);
        let err = plan_one(
            &lowered,
            &lowered.launches[0],
            PLAIN,
            Built {
                module: Module::new([256, 1, 1]),
                declared: &d,
                sig: None,
            },
            Sources {
                arena: Arena {
                    buffer: &buf,
                    bytes: 1 << 20,
                },
                resolver: &store,
                min_offset: 256,
            },
            fire(),
        )
        .expect_err("absent");
        assert_eq!(
            err,
            Undispatchable::Operand {
                at: 1,
                why: Unbindable::UnknownWeight("absent".into())
            }
        );
    }

    /// An empty grid never reaches a device.
    ///
    /// The refusal this crate makes hardest, because the alternative reports
    /// success: `dispatch_workgroups(0, 1, 1)` is legal WebGPU, runs nothing,
    /// and leaves the output holding whatever it was born with.
    #[test]
    fn a_grid_that_would_run_nothing_is_refused() {
        // A zero-width operand: `Rule::Elementwise` is `width * rows`, so the
        // grid is zero on x.
        let lowered = plan(
            "plan_order",
            vec![Arg::Arena {
                at: 0,
                width: 0,
                bytes: 2,
            }],
            Vec::new(),
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(1, &[]);
        let err = plan_one(
            &lowered,
            &lowered.launches[0],
            PLAIN,
            Built {
                module: Module::new([256, 1, 1]),
                declared: &d,
                sig: None,
            },
            Sources {
                arena: Arena {
                    buffer: &buf,
                    bytes: 1 << 20,
                },
                resolver: &store,
                min_offset: 256,
            },
            fire(),
        )
        .expect_err("empty");
        // The operand is refused first, because a zero-width rectangle is also
        // a zero-length binding -- and WebGPU has no such binding. Either
        // refusal is a refusal; what matters is that nothing dispatches.
        assert!(
            matches!(
                err,
                Undispatchable::Empty { .. } | Undispatchable::Operand { .. }
            ),
            "a rectangle of nothing produced {err:?}"
        );
    }
}
