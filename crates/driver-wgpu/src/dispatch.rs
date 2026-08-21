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
//! A grid evaluates against facts that are mostly FIRE-WIDE — head count, head
//! width, expert count. Mostly, not all. `KernelSig` carried three indices —
//! `grid_param`, `head_param`, `heads_param` — that each named a scalar in the
//! STATEMENT's own run, because a kernel's extent can vary per layer in a way
//! no fire-wide number expresses.
//!
//! Gemma-4 is the case that forces it: its full-attention layers rotate a
//! quarter of each head and its sliding layers rotate all of one, and they
//! carry four 512-wide KV heads against sixteen 256-wide ones. A driver
//! reading the fire's `rotary_dims` describes neither.
//!
//! A ROUTINE says this without an index: the fact is an argument, and
//! [`crate::lowering::hold::Handles::stated`] is how an arm supplies it — the
//! statement's scalar where there is one, the fire's where there is not, and
//! never a zero. A grid of zero is a dispatch that runs nothing, returns
//! success, and leaves the output holding whatever it was born with, which is
//! the failure mode this crate refuses hardest.
//!
//! # The device limit is deliberately not applied here
//!
//! [`plan_one`] plans a grid without applying the adapter's ceiling,
//! because a `Dispatch` is what the KERNEL needs and the limit is a property of
//! the adapter that will run it. The device half checks
//! [`Dispatch::groups`] against its own
//! `max_compute_workgroups_per_dimension` at encode time, where it has the
//! number. Folding the limit in here would mean a plan that this build can
//! reason about only where it can also run.

use crate::binding::{Arena, Bound, ParamSlot, Params, Resolve};
use crate::geometry::Module;
use model_compiler::lower::{Arg, Launch, Lowered};

/// The fire-wide shape a plan is executed at.
///
/// Everything a grid needs that a single statement does not state. A statement
/// may override any of these by carrying its own scalar; see
/// [`crate::lowering::hold::Handles::stated`].
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
    /// Recurrent value heads, or `0` for a stack with none.
    ///
    /// A THIRD pair beside [`Self::kv_heads`] and [`Self::head_dim`], and not
    /// a third spelling of them: a gated-deltanet layer has no keys and
    /// values in the attention sense at all, and its head count and width are
    /// what size the recurrent slab, the gated norm and the scan grid.
    ///
    /// Before this pair existed the GDN arms read the numbers off the
    /// STATEMENT's scalars while their signatures asked the fire for
    /// `VHeads`/`VDim`, so the two readings of the same launch disagreed
    /// about where the shape lives. The signatures are the ones metal states
    /// and runs on a device; this makes the fire able to answer them.
    pub v_heads: u32,
    /// See [`Self::v_heads`].
    pub v_dim: u32,
    /// Experts the router scores.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
}

impl Geometry {
    /// This stack's RECURRENT head count and width.
    ///
    /// Falls back to the attention pair when no recurrent one is stated,
    /// character-for-character `driver_metal`'s `recurrent_at`. A fire whose
    /// only shape IS the recurrent shape -- which is every GDN fixture that
    /// predates the pair -- keeps reading the numbers it always did.
    #[must_use]
    pub const fn recurrent(&self) -> (u32, u32) {
        if self.v_heads > 0 && self.v_dim > 0 {
            (self.v_heads, self.v_dim)
        } else {
            (self.kv_heads, self.head_dim)
        }
    }
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
    /// Workgroups in each dimension.
    ///
    /// Never contains a zero, and that used to be a named refusal here. A
    /// routine builds its grid from `kernels::shader`'s helpers, which refuse
    /// an empty extent as `Refusal::Grid` before a `Fire` is ever stated, so
    /// the zero is caught a plane earlier and arrives as
    /// [`Undispatchable::Routine`].
    pub groups: [u32; 3],
    /// The `@group(1)` uniform block, when the dispatch ALSO has a storage
    /// one.
    ///
    /// Empty for every kernel whose scalars ride one place or the other, which
    /// until qwen3.5's prompt scan was all of them: [`Self::params`] carries a
    /// `ParamSlot::Uniform` block or a `ParamSlot::Storage` one and a dispatch
    /// needed no way to say BOTH.
    ///
    /// `ssm/gdn_prep.wgsl`'s prefill pair needs both, and the split is the
    /// ABI's rather than a convenience: `GdnCoreParams` is a `@group(0)`
    /// storage buffer because every GDN row states it as a pointer to numbers
    /// the host already assembled, while `row_pitch` and `n_scan` are the
    /// RECTANGLE's and reach the shader as the two fields of its `@group(1)`
    /// block. Appending them to the storage run instead -- which is what
    /// happened while nothing needed the distinction -- leaves the uniform
    /// empty, and `Device::check_bindable` refuses a short block by name
    /// because a zero pitch is a plausible number.
    pub uniform: Vec<u8>,
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
}

impl std::fmt::Display for Undispatchable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { symbol } => write!(f, "no armed stem claims `{symbol}`"),
            Self::Conditional { symbol, cond } => {
                write!(f, "`{symbol}` sits under unevaluated guard {cond}")
            }
            Self::Routine { symbol, why } => {
                write!(f, "`{symbol}` could not be planned as a routine: {why}")
            }
            Self::Multiple { symbol, stated } => write!(
                f,
                "`{symbol}`'s body stated {stated} dispatches and one launch \
                 carries one"
            ),
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
            // A raise has no row width to contribute; see `Arg::Raised`.
            Arg::Weight(_) | Arg::Raised { .. } => None,
        })
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
    // THERE IS NO FORK ANY MORE. Every symbol is planned by its ROUTINE: the
    // arm finds the operands, the body states the module, the entrypoint and
    // the lanes, and nothing reads a row.
    //
    // This stood above a row lookup for the length of the refactor, and the
    // ordering was the point -- an armed symbol needs no `KernelSig` at all,
    // so a family whose arms were all in could have its rows deleted. All ten
    // families are in, on all three backends, and `plan_by_row` is gone with
    // the columns it read.
    let Some(routine) = crate::lowering::routine::armed(symbol) else {
        // Not "fall back to the table": there is no table. A symbol no stem
        // claims cannot be planned by any path, and
        // `arm::the_armed_stems_are_the_ones_registered_and_nothing_else`
        // asserts that none exists by walking all 481 entrypoints. This is
        // what that assertion would look like at run time if it were ever
        // wrong.
        return Err(Undispatchable::Unknown {
            symbol: symbol.to_owned(),
        });
    };
    {
        let facts = crate::lowering::hold::facts(
            symbol,
            launch.rows.end - launch.rows.start,
            fire,
            lowered.n_requests,
            widths(lowered, launch).next_back().unwrap_or(0),
            widths(lowered, launch).next().unwrap_or(0),
        );
        let mut planned = crate::lowering::routine::plan(
            routine,
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
        Ok(planned.remove(0))
    }
}

/// [`plan_one`], for a routine that may state MORE than one dispatch.
///
/// The same work and the same refusals, without the narrowing at the end: a
/// body states a list and this hands the list back. `plan_one` is the caller
/// that wants exactly one and says so; a shell that records dispatches wants
/// all of them.
///
/// # Why two passes over one statement is a routine's business and not a
/// plan's
///
/// Because the second pass is a property of the DEVICE. `attn`'s split decode
/// cuts a row's key range into slices so that a fire too narrow to fill this
/// GPU has more workgroups, and then merges them -- and the same statement on
/// a machine with fewer cores, or on a driver that does not split, is one
/// dispatch. Putting the pair in the authored trace would put one backend's
/// occupancy in a model description every backend reads.
///
/// # Errors
///
/// As [`plan_one`], less [`Undispatchable::Multiple`], which this cannot
/// raise.
pub fn plan_all<'a, R: Resolve>(
    lowered: &'a Lowered,
    launch: &Launch,
    built: Built<'_>,
    sources: Sources<'a, R>,
    fire: Geometry,
) -> Result<Vec<Dispatch<'a, R::Buffer>>, Undispatchable> {
    let Built { declared, .. } = built;
    let Sources {
        arena,
        resolver,
        min_offset,
    } = sources;
    let symbol = lowered.kernels[launch.kernel as usize].as_str();
    if launch.cond != Launch::NO_COND {
        return Err(Undispatchable::Conditional {
            symbol: symbol.to_owned(),
            cond: launch.cond,
        });
    }
    let Some(routine) = crate::lowering::routine::armed(symbol) else {
        return Err(Undispatchable::Unknown {
            symbol: symbol.to_owned(),
        });
    };
    let facts = crate::lowering::hold::facts(
        symbol,
        launch.rows.end - launch.rows.start,
        fire,
        lowered.n_requests,
        widths(lowered, launch).next_back().unwrap_or(0),
        widths(lowered, launch).next().unwrap_or(0),
    );
    crate::lowering::routine::plan(
        routine,
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

    /// An ARMED symbol is planned by its routine, and nothing reads a row.
    ///
    /// `argmax_logits` HAS NO ROW: `sample`'s rows came off when its arm
    /// landed, and the table they were in is empty. A grid coming back is
    /// therefore the routine plane's answer, because there is no other
    /// answer left — [`plan_one`] finds the arm or refuses by name.
    ///
    /// Its ENTRYPOINT is still stated, by `kernels_wgpu::RETIRED`, so every
    /// sweep still compiles and covers the shader. Losing the row is not
    /// losing the kernel.
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
            // ONE SCALAR, `rows`: after the marks migration `argmax_logits`
            // takes its rectangle as a `Const<u32>` on the statement, so a
            // fixture with an empty params run refuses in the body before the
            // dispatch is planned. Matches the launch's own row count.
            vec![4],
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(0, &[]);

        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            Built {
                module: Module {
                    local: Local([256, 1, 1]),
                    tile: None,
                },
                declared: &d,
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

        // THE ENTRYPOINT, NOT THE NAME THAT WAS PLANNED. The statement says
        // `argmax_logits` and `sample.rs`'s body fires
        // `Fire::at("sample/argmax.wgsl", "argmax_logits_bfloat16")`, so the
        // planned symbol is the second of those. That is the whole point of
        // the routine plane being asked: a row could only ever hand back the
        // name it was filed under, and a body picks the module its arguments
        // are actually shaped for. This assertion used to read `argmax_logits`
        // and was weaker for it -- it could not tell the two planes apart,
        // which is the one thing this test exists to do.
        assert_eq!(got.symbol, "argmax_logits_bfloat16");
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

    /// The second armed symbol halves its width, and the routine carries it.
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
            // ONE SCALAR, `rows`: after the marks migration `copy_logits_bf16`
            // reads its rectangle from the statement's params run rather than
            // from the fire. Matches the launch's row count.
            vec![4],
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(0, &[]);

        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            Built {
                module: Module {
                    local: Local([256, 1, 1]),
                    tile: None,
                },
                declared: &d,
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
        assert_eq!(got.buffers.len(), 3);
        assert_eq!(got.block_at, None);
    }

    /// An odd vocabulary is refused by the BODY.
    ///
    /// The body's own header requires `vocab` to be even, because an odd pitch
    /// starts the next row inside the previous row's last word. That refusal
    /// is a fact about the shader that no row had a column for, and this is
    /// the first time this backend can state one: a positional row would have
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
            // ONE SCALAR, `rows`: after the marks migration `copy_logits_bf16`
            // takes its rectangle as a `Const<u32>` on the statement. Without
            // it the arm refuses in `param(0)` before it can look at `vocab`,
            // and this test would report the wrong refusal.
            vec![4],
        );
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(0, &[]);
        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            Built {
                module: Module {
                    local: Local([256, 1, 1]),
                    tile: None,
                },
                declared: &d,
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
    /// from the arm rather than from the binding it never reached.
    #[test]
    fn a_routine_that_cannot_be_planned_is_refused_by_name() {
        let lowered = plan("argmax_logits", Vec::new(), Vec::new());
        let buf = Placeholder(1 << 20);
        let store = Store::default();
        let d = declared(0, &[]);
        let got = plan_one(
            &lowered,
            &lowered.launches[0],
            Built {
                module: Module {
                    local: Local([256, 1, 1]),
                    tile: None,
                },
                declared: &d,
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
            // A hand-built lowering states no per-argument rows; zero is "no opinion".
            arg_rows: Vec::new(),
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
            // A fixture states no attention schedule to raise.
            preps: Vec::new(),
            readout: None,
        }
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
            ..Default::default()
        }
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
    // The fallback was live and reachable when this was written, and two
    // tests in this file held it:
    // `an_unstated_row_dispatches_from_the_plans_own_argument_order` bound a
    // rectangle in the plan's own order through `PLAIN`, and
    // `an_unstated_rows_operands_come_from_the_plan_and_its_grid_does_not`
    // took the same rectangle through `reorder` and then all the way to the
    // `Ungeometric::Unstated` refusal. Both have since gone with
    // `plan_by_row`, and the block at the end of this module is where that is
    // written down: nothing reaches `reorder` from here any more.
    //
    // What a REAL launch's operands come from is no longer a row at all:
    // `crate::lowering::hold`'s `Handles::{input, output, weight}` mint them in
    // the order the body asks and record each as an `Asked`, and
    // `crate::lowering::routine::bind` turns those handles into `Placed`
    // bind-group entries against the module's own `Declared`. Its per-symbol
    // cover is `hold.rs`'s `handles_are_minted_in_the_order_the_body_asks` and
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
    // * The operands and the bind group: `crate::lowering::hold`'s `Handles`
    //   mints them in the order the body asks and records each as an `Asked`,
    //   and `crate::lowering::routine::bind` lays them out as `Placed` against
    //   the module's own `Declared`, refusing by name through `Unplanned::`
    //   `{Handle, Operand, Blocks, NoCache, Absent}`.
    // * The scalars against the block's room: `Unplanned::Scalars`, checked by
    //   `routine.rs`'s `scalars_wider_than_the_modules_block_are_refused_by_name`.
    // * The grid that must not be empty: `Unplanned::Refused` through the
    //   body's own extent, checked by `a_body_that_computed_a_zero_extent_is_refused`.
    // * The contiguous-cache refusal: `crate::lowering::hold::contiguous_pool`,
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
    ///
    /// Before the arm is looked for, which is why the symbol below is one no
    /// stem claims: a guard nothing has answered is refused on its own terms
    /// rather than as a symbol the routine plane does not know.
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
                Built {
                    module: Module::new([256, 1, 1]),
                    declared: &d,
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

    /// A symbol no arm claims is named rather than guessed at.
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
                Built {
                    module: Module::new([256, 1, 1]),
                    declared: &d,
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

    // RETIRED: THE ROW PATH HAS NO CALLER. `plan_one` does not fork, so ten
    // tests here were driving `plan_by_row`, `dims_of` and `rule_of` -- three
    // functions nothing production-side can reach -- over a table with
    // nothing in it.
    //
    // Two fixtures went with them. `static PLAIN` was a one-row table of an
    // operand-less `Rule::Elementwise` row, invented so the plan-order
    // fallback could be driven end to end; `const BARE` was sixteen
    // `KernelSig` columns at rest, so a test could state the two its claim
    // was about. Both were written when the shipped table stopped being able
    // to supply an example, which is the same fact these deletions are, one
    // layer up.
    //
    // THE PLAN-ORDER FALLBACK, which was `.wiki/new-driver/vulkan.md` §13's
    // claim and carried 56 rows over 292 entrypoints.
    // `an_unstated_row_dispatches_from_the_plans_own_argument_order` bound a
    // rectangle whose row stated no operands and got the plan's own two
    // offsets back in the plan's own order, with `Params::None` and no
    // `block_at`.
    // `an_unstated_rows_operands_come_from_the_plan_and_its_grid_does_not`
    // took the same rectangle through `reorder` to two `Slot::Buffer`s and
    // then the whole way to `Undispatchable::Ungeometric` carrying
    // `Ungeometric::Unstated`. The pair separated "an unstated row is
    // unlaunchable" from "an unstated row binds and then has no grid", which
    // are two different repairs: `launch`, not `operands`.
    //
    // THE ROW'S ORDER AGAINST THE TRACE'S.
    // `a_rectangle_binds_in_the_rows_order_and_not_the_traces` stated
    // `In(0), Out(0), In(1)` against a trace handing over its reads and then
    // its write, and required the OUTPUT second. Bound positionally the
    // shader is given the second read where its output belongs and writes the
    // answer into a buffer nobody reads -- both are storage buffers of the
    // same length, so nothing downstream sees it.
    //
    // THE ROW'S SCALARS AGAINST THE MODULE'S BLOCK.
    // `a_module_with_no_room_for_the_rows_scalars_says_how_many_of_each` put
    // a row of two scalars against a uniform block of one member and required
    // `Undispatchable::Scalars { stated: 2, room: 1 }` by name, having first
    // planned the same rectangle against a block of two so the refusal could
    // not be about something else. By name because the underlying defect is
    // silent: WGSL bounds-checks every access, so a short block reads ZEROS
    // and a missing pitch is a plausible number.
    //
    // THE STATEMENT'S OWN NUMBERS AGAINST THE FIRE'S -- `dims_of`'s three
    // indices, each checked against a fire that DISAGREED on purpose.
    // `a_row_that_names_its_extent_takes_it_from_the_statement` required
    // `grid_param` to take the extent from the statement's param run (256)
    // and not the rectangle's width (64), and a statement that does not carry
    // it to fall back to the FIRE rather than to zero, because a zero extent
    // is a grid that runs nothing and reports success.
    // `a_row_that_names_its_head_shape_takes_it_from_the_statement` did the
    // same for `head_param` and `heads_param` with gemma-4's full-attention
    // shape -- a 512-wide head and four of them, against a fire stating 128
    // and 8.
    //
    // THE RULE, ANSWERED BEFORE ANYTHING BINDS.
    // `a_symbols_rule_is_answerable_from_the_table_alone` asked `rule_of` for
    // a row's `LaunchRule` with no operands, no arena and no resolver, which
    // is what `engine`'s `every_launch_fits` needs to size a prefill it has
    // not planned. It went through `kernels::sig_in` rather than an equality
    // test, and pinned the axis peeling that is the reason why: the fully
    // spelled `rule_lookup_bfloat16_gs_64_b_4` and the base `rule_lookup`
    // both answer, and a half-spelled `rule_lookup_bfloat16` is refused
    // rather than rounded to the nearest row. Exact matching found a row for
    // 432 of the Vulkan port's 3992 launches.
    //
    // THE THREE REFUSALS THE ROW PATH RAISED ON ITS WAY.
    // `a_module_that_binds_more_than_the_plan_states_is_refused_by_arity`
    // required `Undispatchable::Arity { stated: 1, module: 3 }`;
    // `an_unresolvable_operand_is_refused_with_its_slot` required
    // `Undispatchable::Operand` carrying `at: 1` and
    // `Unbindable::UnknownWeight`, so a missing weight names the slot it
    // would have gone in; and `a_grid_that_would_run_nothing_is_refused` sent
    // a zero-width rectangle in and required `Empty` or `Operand` -- either
    // refusal, so long as nothing dispatched.
    //
    // WHERE THE CLAIMS LIVE NOW.
    //
    // * The operand order and the bind group: `crate::lowering::hold`'s
    //   `Handles::{input, output, weight}` mint one handle per thing the BODY
    //   asks for, in the order it asks, recording each as an `Asked`, and
    //   `crate::lowering::routine::bind` lays them out as `Placed` against
    //   the module's own `Declared`. Held by `hold.rs`'s
    //   `handles_are_minted_in_the_order_the_body_asks` and
    //   `a_statement_the_arm_cannot_fill_is_refused`, and by `routine.rs`'s
    //   `a_handle_past_what_the_arm_minted_is_refused_by_name` and
    //   `an_operand_the_fire_does_not_hold_is_refused_by_name`.
    // * The scalars against the block's room: `Unplanned::Scalars`, held by
    //   `routine.rs`'s
    //   `scalars_wider_than_the_modules_block_are_refused_by_name`.
    // * The grid that must not be empty: the body states its own extent and
    //   `routine.rs`'s `a_body_that_computed_a_zero_extent_is_refused` pins
    //   the refusal; `crates/driver-wgpu/tests/arena.rs`'s
    //   `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`
    //   asserts no grid of any real rectangle contains a zero.
    // * The statement's numbers beating the fire's:
    //   `crate::lowering::hold::facts` builds `Facts` from the launch and the
    //   fire, and `Handles::stated` is `dims_of`'s fallback transcribed --
    //   the statement's scalar, zero treated as absent, the fire's number
    //   otherwise. Its doc says so at the line.
    // * The symbol lookup: `crate::lowering::routine::armed` over
    //   `hold::crossed`, whose spelling rules are held by `hold.rs`'s
    //   `the_armed_stems_are_the_ones_registered_and_nothing_else` and
    //   `every_entrypoint_is_claimed_by_the_stem_that_owns_it`, and whose
    //   join is held by `routine.rs`'s
    //   `an_armed_symbol_is_reached_through_the_spelling_a_plan_uses` --
    //   which pins the same two traps `sig_in`'s axis peeling was written
    //   for, a quantization suffix no routine name carries and a stem that is
    //   a strict prefix of another.
    //
    // WHAT IS LOST, and it is not small.
    //
    // `Undispatchable::{Arity, Operand, Scalars, Empty, Ungeometric}` were
    // built and named ONLY here. Nothing constructs them now: every refusal
    // the routine plane raises arrives as `Undispatchable::Routine` carrying
    // a rendered string, so a caller that used to tell an arity mistake from
    // an unresolvable weight from a block too small by MATCHING now gets one
    // variant and prose. The conditions are still checked one layer down as
    // `Unplanned::*`; what is gone is this enum's ability to say which.
    // `tests/citations.rs`'s census counts variants that are declared, so
    // those five have to be listed as unnamed there or deleted outright.
    //
    // The plan-order fallback has no successor at all. `Handles::input` is
    // not the same claim -- a body asks for the statement's operands one at a
    // time, by index, and no code anywhere now takes a whole argument list
    // and binds it in the order it arrived. That was `binding::reorder`, and
    // `reorder` is retiring with these tests.
    //
    // `rule_of`'s promise -- a grid rule for a symbol, with nothing bound --
    // has no successor either, and it had a real caller: a routine states its
    // lanes only when it is PLANNED, against a statement, so
    // `engine`'s `every_launch_fits` cannot ask what it used to ask. That
    // caller has to be rewritten around `plan_one` or dropped; it is not a
    // gap a test on this side can close.
    //
    // The three-index override survives as `Handles::stated` and is used by
    // the arms that need it, but no test in this crate now pins the RULE that
    // a statement's number beats the fire's. `dims_of` had one line per index
    // and two tests holding all three against a fire chosen to disagree.
    // Gemma-4 is the case that motivated them -- four 512-wide KV heads
    // against sixteen 256-wide ones in one checkpoint -- and it is now
    // asserted per body or not at all.
}
