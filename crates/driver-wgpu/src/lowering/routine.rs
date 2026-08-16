//! The routine plane: running a crossed body to get a dispatch.
//!
//! # The fork
//!
//! ```text
//!   table path:    row.operands ──► reorder ──► slots ──► Dispatch
//!   routine path:  arm ──► handles ──► BODY ──► Fire + args ──► Dispatch
//! ```
//!
//! [`super::arm`] does the first half: it finds a statement's operands and
//! hands them to a body as opaque handles. This file does the second: it runs
//! the body, which states the module, the entrypoint and the LANES, and turns
//! what comes back into the same [`Dispatch`] the table path builds.
//!
//! A body dispatches through [`Encode`], so [`Planner`] is an `Encode` that
//! records instead of submitting — the same shape `kernels-wgpu`'s own
//! `tests/routines.rs` uses to check bodies without a device, and the same
//! shape `driver-metal::lowering::routine::Planner` has.
//!
//! # Why a body may state more than one dispatch
//!
//! [`Encode::dispatch`] can be called more than once — a two-pass reduction
//! is two entrypoints over one statement — so this collects a `Vec` where the
//! table path returned exactly one. Nothing in this tree does it yet, and a
//! plane that carried only one would have to be reworked the day something
//! does.

use std::cell::RefCell;

use kernels::routine::{Provenance, Refusal, Routine};
use kernels_wgpu::routine::{ArgValue, Encode, Fire, Wgpu};
use model_compiler::lower::Launch;

use crate::binding::{Bound, ParamSlot, Params};
use crate::dispatch::Dispatch;
use crate::reflect::Declared;

/// Why a body's dispatch could not become one.
///
/// No `Eq`: `Unbindable` has none, because the extents it carries come from a
/// plan and are compared for equality nowhere.
#[derive(Clone, Debug, PartialEq)]
pub enum Unplanned {
    /// The body refused.
    Refused(Refusal),
    /// The body named a handle no arm minted.
    ///
    /// A body binds what its arm handed it, so this is the two halves of the
    /// routine plane disagreeing rather than anything a caller did.
    Handle {
        /// The handle it named.
        handle: u32,
        /// How many the arm minted.
        minted: usize,
    },
    /// The body's scalars do not fit the module's uniform block.
    Scalars {
        /// How many bytes the body's scalars pack to.
        stated: usize,
        /// How many the module declares.
        room: u32,
    },
    /// An operand the body asked for is not in the fire.
    Operand {
        /// Which of the body's asks, in the order it asked.
        at: usize,
        /// What binding said.
        why: crate::binding::Unbindable,
    },
    /// The body stated no dispatch at all.
    ///
    /// Distinct from a refusal: the body returned `Ok` having asked for
    /// nothing, so a caller that treated it as done would run a launch that
    /// writes nothing and reports success.
    Silent,
}

impl std::fmt::Display for Unplanned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(why) => write!(f, "the body refused: {why:?}"),
            Self::Handle { handle, minted } => write!(
                f,
                "the body bound handle {handle} and its arm minted {minted}"
            ),
            Self::Scalars { stated, room } => write!(
                f,
                "the body's scalars pack to {stated} bytes and the module's \
                 uniform block is {room}"
            ),
            Self::Operand { at, why } => {
                write!(f, "the body's operand {at} is not in the fire: {why}")
            }
            Self::Silent => write!(f, "the body stated no dispatch"),
        }
    }
}

impl std::error::Error for Unplanned {}

/// One dispatch a body asked for, before it is bound to buffers.
///
/// The buffers are HANDLES here rather than [`Bound`]s, which is what keeps
/// this type free of the arena's lifetime: a `Planner` collects what the body
/// said, and [`bind`] is where handles become ranges.
///
/// No `Eq`: `ArgValue` carries an `f32`, so the scalars can only be compared
/// partially. That is the type being honest rather than a gap -- two dispatch
/// lists that differ by a NaN are not equal and should not claim to be.
#[derive(Clone, Debug, PartialEq)]
pub struct Stated {
    /// The shader file the body named.
    pub module: String,
    /// The entrypoint it named.
    pub entrypoint: String,
    /// The lanes it asked for, BEFORE the division into workgroups.
    pub lanes: [u32; 3],
    /// Its buffer operands, as handles its arm minted.
    pub handles: Vec<u32>,
    /// Its scalars, packed in the order the body passed them.
    pub scalars: Vec<ArgValue>,
}

/// An [`Encode`] that records what a body asks for.
#[derive(Default)]
pub struct Planner {
    out: RefCell<Vec<Stated>>,
}

impl Planner {
    /// A planner with nothing recorded.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// What the body asked for, in the order it asked.
    #[must_use]
    pub fn stated(self) -> Vec<Stated> {
        self.out.into_inner()
    }
}

impl Encode for Planner {
    fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
        // A body with nothing to do should have refused. Zero lanes become
        // `dispatch_workgroups(0, 1, 1)`, which is legal WebGPU that runs
        // nothing and reports success, so the output keeps whatever it held
        // and the model answers from stale bytes. `Refusal::Empty` is a body
        // that NOTICED; this is a body that computed an extent and got zero.
        if fire.lanes.contains(&0) {
            return Err(Refusal::Grid {
                what: "the lanes a routine asked for",
                at: 0,
            });
        }
        let mut handles = Vec::new();
        let mut scalars = Vec::new();
        for arg in args {
            match arg {
                ArgValue::Buffer(h) => handles.push(*h),
                other => scalars.push(*other),
            }
        }
        self.out.borrow_mut().push(Stated {
            module: fire.module.to_owned(),
            entrypoint: fire.entrypoint.to_owned(),
            lanes: fire.lanes,
            handles,
            scalars,
        });
        Ok(())
    }
}

/// Run one crossed routine's body over the arguments its arm produced.
///
/// # Errors
///
/// [`Unplanned::Refused`] when the body refuses, [`Unplanned::Silent`] when
/// it returns `Ok` without dispatching.
pub fn state(routine: &'static Routine<Wgpu>, args: &[ArgValue]) -> Result<Vec<Stated>, Unplanned> {
    let planner = Planner::new();
    (routine.body)(&planner, args).map_err(Unplanned::Refused)?;
    let out = planner.stated();
    if out.is_empty() {
        return Err(Unplanned::Silent);
    }
    Ok(out)
}

/// Turn one stated dispatch into the [`Dispatch`] the shell submits.
///
/// `bounds` is what the arm resolved, indexed by the handles it minted.
///
/// # Errors
///
/// [`Unplanned::Handle`] for a handle no arm minted; [`Unplanned::Scalars`]
/// when the packed scalars do not fit the module's uniform block.
pub fn bind<'a, B>(
    stated: &Stated,
    bounds: &[Bound<'a, B>],
    declared: &Declared,
    symbol: &'a str,
    launch: &Launch,
) -> Result<Dispatch<'a, B>, Unplanned> {
    let mut buffers = Vec::with_capacity(stated.handles.len());
    for &h in &stated.handles {
        let at = h as usize;
        let bound = bounds.get(at).ok_or(Unplanned::Handle {
            handle: h,
            minted: bounds.len(),
        })?;
        buffers.push(*bound);
    }

    // The scalars, packed in the order the body passed them and aligned the
    // way WGSL requires. `Encoder::block` carries the same rule for the
    // device path and states vulkan's measurement as its justification: a
    // `Usize` is eight-aligned, so a run that concatenated would put the
    // second field where the shader does not read it.
    let mut bytes: Vec<u8> = Vec::new();
    for value in &stated.scalars {
        let (width, run): (usize, [u8; 8]) = match value {
            ArgValue::I32(v) => (4, {
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&v.to_le_bytes());
                b
            }),
            ArgValue::U32(v) => (4, {
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&v.to_le_bytes());
                b
            }),
            ArgValue::F32(v) => (4, {
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&v.to_le_bytes());
                b
            }),
            ArgValue::Usize(v) => (8, v.to_le_bytes()),
            ArgValue::Buffer(_) => unreachable!("buffers were split out above"),
        };
        while !bytes.len().is_multiple_of(width) {
            bytes.push(0);
        }
        bytes.extend_from_slice(&run[..width]);
    }

    let params = if bytes.is_empty() {
        Params::None
    } else {
        if bytes.len() > declared.uniform_bytes as usize {
            return Err(Unplanned::Scalars {
                stated: bytes.len(),
                room: declared.uniform_bytes,
            });
        }
        // Padded to what the module declares, for the reason
        // `Device::check_bindable` refuses a short block: WGSL reads the
        // struct's whole extent and a short binding reads zeros past its end,
        // which is a plausible number rather than an error.
        bytes.resize(declared.uniform_bytes as usize, 0);
        Params::Block {
            bytes,
            at: ParamSlot::Uniform,
        }
    };

    let local = declared.local;
    Ok(Dispatch {
        symbol,
        buffers,
        params,
        // `ParamSlot::Uniform` is a bind group of its own and takes no place
        // in the buffer list, which is what `None` says here.
        block_at: None,
        groups: [
            stated.lanes[0].div_ceil(local[0].max(1)),
            stated.lanes[1].div_ceil(local[1].max(1)),
            stated.lanes[2].div_ceil(local[2].max(1)),
        ],
        op: launch.op,
    })
}

/// Plan one launch through the ROUTINE path.
///
/// The whole fork in one function: find the operands with the arm, run the
/// body, resolve what it asked for, and bind. `plan_one` calls this when
/// [`armed`] answers and takes the table path otherwise.
///
/// # Why it returns a `Vec`
///
/// A body may state more than one dispatch — a two-pass reduction is two
/// entrypoints over one statement — where the table path returned exactly
/// one. Nothing in this tree does it yet, and a plane that carried only one
/// would have to be reworked the day something does.
///
/// # Errors
///
/// [`Unplanned`] for the body's own refusals; [`crate::binding::Unbindable`] wrapped in
/// [`Unplanned::Operand`] when an operand the body asked for cannot be found
/// in the fire.
pub fn plan<'a, R: crate::binding::Resolve>(
    routine: &'static Routine<Wgpu>,
    arm: super::arm::Arm,
    lowered: &'a model_compiler::lower::Lowered,
    launch: &Launch,
    declared: &Declared,
    sources: crate::dispatch::Sources<'a, R>,
    facts: super::arm::Facts,
) -> Result<Vec<Dispatch<'a, R::Buffer>>, Unplanned> {
    let crate::dispatch::Sources {
        arena,
        resolver,
        min_offset,
    } = sources;
    let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
    let mut handles = super::arm::Handles::over(args, results(routine));
    let taken_args = arm(&mut handles, facts).map_err(Unplanned::Refused)?;
    let stated = state(routine, &taken_args)?;

    // The operands the BODY asked for, resolved in the order it asked. Not
    // the statement's order: `Handles` minted a handle per ask, and this
    // walks the same list.
    let mut bounds = Vec::with_capacity(handles.taken().len());
    for (at, arg) in handles.taken().iter().enumerate() {
        let bound = crate::binding::resolve(arg, launch, arena, resolver, min_offset)
            .map_err(|why| Unplanned::Operand { at, why })?;
        bounds.push(bound);
    }

    let symbol = lowered.kernels[launch.kernel as usize].as_str();
    stated
        .iter()
        .map(|one| bind(one, &bounds, declared, symbol, launch))
        .collect()
}

/// The routine and the arm for a symbol, if this backend has crossed it AND
/// armed it.
///
/// Both halves or neither: a body with no arm cannot be given its operands,
/// and an arm with no body has nothing to hand them to. 99 routines have
/// crossed and one is armed, so this answers `None` for almost everything —
/// which is what keeps the table path serving every real fire today.
#[must_use]
pub fn armed(symbol: &str) -> Option<(&'static Routine<Wgpu>, super::arm::Arm)> {
    // By STEM, not by name. A plan spells `silu_mul_bfloat16` and the routine
    // is called `silu_mul`; the registry knows which prefix of a symbol names
    // a kernel, and it is the only thing that can, because this fork answers
    // before any row is looked up.
    let (stem, arm) = super::arm::crossed(symbol)?;
    let routine = kernels_wgpu::routines()
        .into_iter()
        .find(|r| r.name == stem)?;
    Some((routine, arm))
}

/// How many of a statement's widthed operands are RESULTS.
///
/// **The routine says**, by counting the writable types in its own signature.
/// The table path read the same fact off a row's `Out` sources — the same
/// number stated in a place the compiler cannot check against the kernel,
/// which is the whole reason this plane exists.
///
/// `driver-metal` counts only `BufMut`. This counts every writable type in
/// `kernels::Ty`, because `ssm`'s bodies take `F32sMut` for their recurrent
/// state and a count that missed those would split a statement in the wrong
/// place — silently, since `split` just takes the last `n`.
#[must_use]
pub fn results(routine: &Routine<Wgpu>) -> usize {
    use kernels::Ty;
    routine
        .args
        .iter()
        .filter(|(ty, _)| {
            matches!(
                ty,
                Ty::BufMut
                    | Ty::F32sMut
                    | Ty::I32sMut
                    | Ty::U32sMut
                    | Ty::U8sMut
                    | Ty::U16sMut
                    | Ty::I8sMut
                    | Ty::BufArrayMut
                    | Ty::BufArrayOut
                    | Ty::BufArrayOutMut
            )
        })
        .count()
}

/// What a body takes that is NOT an operand.
///
/// `Provenance::Env` marks the arguments that size the grid and that the
/// kernel never reads. They are why an arm can produce a shorter list than
/// the body's arity: the arm supplies operands, the `Facts` supply these.
#[must_use]
pub fn env_count(routine: &Routine<Wgpu>) -> usize {
    routine
        .args
        .iter()
        .filter(|(_, prov)| *prov == Provenance::Env)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A module that declares nothing but a workgroup and a block size.
    ///
    /// `Declared` has no `Default` on purpose — every field is a fact read
    /// off a shader — so a test that needs one writes it out.
    fn declared(uniform_bytes: u32, local: [u32; 3]) -> Declared {
        Declared {
            local,
            bindings: 0,
            reads_workgroup_count: false,
            grid_axes: [true, false, false],
            uniform_offsets: Vec::new(),
            uniform_bytes,
            used: Vec::new(),
            block_bytes: Vec::new(),
        }
    }

    fn fire<'a>(module: &'a str, entrypoint: &'a str, lanes: [u32; 3]) -> Fire<'a> {
        Fire {
            module,
            entrypoint,
            lanes,
        }
    }

    /// A planner records what the body said, splitting buffers from scalars
    /// by ARGVALUE VARIANT.
    ///
    /// That is the same rule `driver-wgpu::encode::Encoder` uses on the
    /// device path, and it is the reason a body's argument list needs no
    /// second statement of which position is which: a handle is a handle.
    #[test]
    fn a_planner_splits_buffers_from_scalars_by_variant() {
        let p = Planner::new();
        p.dispatch(
            fire("sample/argmax.wgsl", "argmax_logits_bfloat16", [256, 1, 1]),
            &[
                ArgValue::Buffer(0),
                ArgValue::Buffer(1),
                ArgValue::U32(7),
                ArgValue::Buffer(2),
                ArgValue::I32(-3),
            ],
        )
        .expect("a real grid");
        let out = p.stated();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].handles, vec![0, 1, 2]);
        assert_eq!(out[0].scalars, vec![ArgValue::U32(7), ArgValue::I32(-3)]);
        assert_eq!(out[0].lanes, [256, 1, 1]);
        assert_eq!(out[0].entrypoint, "argmax_logits_bfloat16");
    }

    /// A zero-lane dispatch is refused, not recorded.
    ///
    /// `dispatch_workgroups(0, 1, 1)` is legal WebGPU that runs nothing and
    /// reports success, so this is the difference between a launch that did
    /// nothing and a launch that says so.
    #[test]
    fn a_body_that_computed_a_zero_extent_is_refused() {
        let p = Planner::new();
        for lanes in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
            assert!(matches!(
                p.dispatch(fire("m.wgsl", "e", lanes), &[]),
                Err(Refusal::Grid { .. })
            ));
        }
        assert!(p.stated().is_empty(), "a refused dispatch was recorded");
    }

    /// Scalars are packed with WGSL's alignment, not concatenated.
    ///
    /// A `Usize` is eight-aligned. `driver-vulkan` measured the same rule the
    /// hard way — a packer that concatenated pushed twenty bytes against a
    /// twenty-four byte range — and `encode::Encoder::block` carries it for
    /// the device path.
    #[test]
    fn a_usize_scalar_is_eight_aligned_in_the_block() {
        let stated = Stated {
            module: "m.wgsl".to_owned(),
            entrypoint: "e".to_owned(),
            lanes: [1, 1, 1],
            handles: Vec::new(),
            scalars: vec![ArgValue::I32(1), ArgValue::Usize(2)],
        };
        let declared = declared(16, [1, 1, 1]);
        let launch = Launch {
            op: 0,
            ..sample_launch()
        };
        let d: Dispatch<'_, ()> = bind(&stated, &[], &declared, "e", &launch).expect("it packs");
        let Params::Block { bytes, .. } = d.params else {
            panic!("scalars should make a block");
        };
        // 4 bytes of i32, 4 of padding, 8 of usize, then padded to the
        // module's own 16.
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[..4], &1i32.to_le_bytes());
        assert_eq!(&bytes[4..8], &[0, 0, 0, 0], "the usize was not aligned");
        assert_eq!(&bytes[8..16], &2u64.to_le_bytes());
    }

    /// A handle no arm minted is a named refusal, not a panic.
    #[test]
    fn a_handle_past_what_the_arm_minted_is_refused_by_name() {
        let stated = Stated {
            module: "m.wgsl".to_owned(),
            entrypoint: "e".to_owned(),
            lanes: [1, 1, 1],
            handles: vec![0, 4],
            scalars: Vec::new(),
        };
        let declared = declared(0, [1, 1, 1]);
        let launch = sample_launch();
        let out: Result<Dispatch<'_, ()>, _> = bind(&stated, &[], &declared, "e", &launch);
        assert_eq!(
            out.unwrap_err(),
            Unplanned::Handle {
                handle: 0,
                minted: 0
            }
        );
    }

    /// A body that refuses is `Unplanned::Refused` and carries WHICH refusal.
    ///
    /// `state` runs the real crossed body, so this is `argmax_logits`'s own
    /// guard answering: a zero row count is a grid of nothing, and the body
    /// says so rather than dispatching it.
    #[test]
    fn a_body_that_refuses_is_reported_with_its_own_refusal() {
        let (routine, _) = armed("argmax_logits").expect("the one armed symbol");
        let args = vec![
            ArgValue::Buffer(0),
            ArgValue::Buffer(1),
            ArgValue::Buffer(2),
            ArgValue::Buffer(3),
            ArgValue::U32(0),
        ];
        let out = state(routine, &args);
        assert!(
            matches!(out, Err(Unplanned::Refused(_))),
            "a zero row count should reach the body's own refusal, got {out:?}"
        );
    }

    /// A body that dispatches nothing is `Unplanned::Silent`, not success.
    ///
    /// Distinct from a refusal on purpose: the body returned `Ok` having
    /// asked for nothing, and a caller that took that for done would run a
    /// launch that writes nothing and reports success — which is the same
    /// failure a zero grid is, arrived at from the other side.
    #[test]
    fn a_body_that_states_no_dispatch_is_not_taken_for_done() {
        // An untouched planner has nothing to state, and that is exactly the
        // condition `state` raises `Unplanned::Silent` on.
        assert!(Planner::new().stated().is_empty());

        // And a body that REFUSED must not have recorded one either -- the
        // two are told apart by which error comes back, not by the
        // recording, so both have to be empty for that distinction to mean
        // anything.
        let (routine, _) = armed("argmax_logits").expect("the one armed symbol");
        let planner = Planner::new();
        (routine.body)(
            &planner,
            &[
                ArgValue::Buffer(0),
                ArgValue::Buffer(1),
                ArgValue::Buffer(2),
                ArgValue::Buffer(3),
                ArgValue::U32(0),
            ],
        )
        .expect_err("a zero row count refuses");
        assert!(
            planner.stated().is_empty(),
            "a refused body recorded a dispatch"
        );

        // And `state` turns an empty recording into `Unplanned::Silent`,
        // driven through a routine whose body dispatches nothing and says it
        // went fine. Written out because no real body does this and the day
        // one does, it must not be taken for done.
        fn says_nothing(_: &kernels_wgpu::routine::Ctx<'_>, _: &[ArgValue]) -> Result<(), Refusal> {
            Ok(())
        }
        static QUIET: Routine<Wgpu> = Routine {
            name: "says_nothing",
            args: &[],
            spelling: &[],
            body: says_nothing,
            whole: false,
            depth_prefix_plan: false,
            in_place: &[],
        };
        assert_eq!(state(&QUIET, &[]), Err(Unplanned::Silent));
    }

    /// Scalars wider than the module's block are refused by NAME.
    ///
    /// WGSL reads the struct's whole extent, and `Device::check_bindable`
    /// refuses a short binding — so the two directions are both covered: too
    /// LITTLE room here, too little block there.
    #[test]
    fn scalars_wider_than_the_modules_block_are_refused_by_name() {
        let stated = Stated {
            module: "m.wgsl".to_owned(),
            entrypoint: "e".to_owned(),
            lanes: [1, 1, 1],
            handles: Vec::new(),
            scalars: vec![ArgValue::Usize(1), ArgValue::Usize(2)],
        };
        let launch = sample_launch();
        let out: Result<Dispatch<'_, ()>, _> =
            bind(&stated, &[], &declared(8, [1, 1, 1]), "e", &launch);
        assert_eq!(
            out.unwrap_err(),
            Unplanned::Scalars {
                stated: 16,
                room: 8
            }
        );
    }

    /// A routine states its own result count, and `argmax` states two.
    ///
    /// Falsified by counting only `Ty::BufMut` as metal does: `ssm`'s bodies
    /// take `F32sMut` for their recurrent state, and a count that missed
    /// those would split a statement in the wrong place — silently, because
    /// `Handles::over` just takes the last `n` as outputs.
    #[test]
    fn a_routine_states_how_many_of_its_operands_are_results() {
        let (argmax, _) = armed("argmax_logits").expect("the one armed symbol");
        assert_eq!(results(argmax), 2, "next_token and eos_flag");

        // The case metal's narrower count would miss.
        let gdn = kernels_wgpu::routines()
            .into_iter()
            .find(|r| r.name == "gdn_core")
            .expect("ssm has crossed");
        let mut_bufs = gdn
            .args
            .iter()
            .filter(|(ty, _)| *ty == kernels::Ty::BufMut)
            .count();
        assert!(
            results(gdn) > mut_bufs,
            "`gdn_core` writes through types that are not `BufMut`, so a \
             count of `BufMut` alone is short: {} against {}",
            mut_bufs,
            results(gdn)
        );
    }

    /// An operand the body asked for and the fire has not is a NAMED
    /// refusal, pointing at which ask.
    ///
    /// `at` is the index in the order the BODY asked, not the statement's,
    /// because that is the number a reader can act on: it names the
    /// `o.input(n)` line in the arm.
    ///
    /// Driven through `plan` end to end — arm, body, resolve — with a
    /// statement whose operands name a weight the resolver does not hold.
    #[test]
    fn an_operand_the_fire_does_not_hold_is_refused_by_name() {
        use crate::binding::{Arena, Resolve};
        use model_compiler::lower::{Arg, Lowered};

        /// A buffer that is a size and nothing else.
        #[derive(Debug, PartialEq)]
        struct Sized(u64);
        impl crate::binding::Allocation for Sized {
            fn size(&self) -> u64 {
                self.0
            }
        }

        /// A resolver that holds nothing at all.
        struct Empty;
        impl Resolve for Empty {
            type Buffer = Sized;
            fn weight(&self, _: &str) -> Option<&Sized> {
                None
            }
            fn named(&self, _: model_ir::trace::ValueId) -> Option<&Sized> {
                None
            }
            fn kv(&self, _: u16, _: bool) -> Option<&Sized> {
                None
            }
            fn table(&self, _: crate::binding::FireTable) -> Option<&Sized> {
                None
            }
        }

        let (routine, arm) = armed("argmax_logits").expect("the one armed symbol");
        // Four ARENA operands, against an arena of no bytes. Weights would
        // not do: `Handles` puts them in their own list, so `argmax`'s asks
        // for two inputs and two outputs would refuse before anything reached
        // the resolver -- which is what the first draft of this test did, and
        // it caught the wrong refusal.
        let lowered = Lowered {
            args: (0..4)
                .map(|n| Arg::Arena {
                    at: n * 64,
                    width: 1,
                    bytes: 2,
                })
                .collect(),
            kernels: vec!["argmax_logits".to_owned()],
            launches: Vec::new(),
            rectangles: 0,
            arena_bytes: 0,
            value_offset: Vec::new(),
            value_owner: Vec::new(),
            epilogue_gather: 0,
            epilogue_norm: 0,
            structural: Vec::new(),
            residue: Vec::new(),
            params: Vec::new(),
            n_requests: 1,
            conds: Vec::new(),
            readout: None,
        };
        let launch = Launch {
            args: 0..4,
            ..sample_launch()
        };
        let out = plan(
            routine,
            arm,
            &lowered,
            &launch,
            &declared(0, [1, 1, 1]),
            crate::dispatch::Sources {
                arena: Arena {
                    buffer: &Sized(0),
                    bytes: 0,
                },
                resolver: &Empty,
                min_offset: 256,
            },
            super::super::arm::facts(7, crate::dispatch::Geometry::default(), 1, 1024, 1024),
        );
        assert!(
            matches!(out, Err(Unplanned::Operand { at: 0, .. })),
            "expected the FIRST ask to be named, got {out:?}"
        );
    }

    /// The armed symbols are crossed, and reached through the spelling a
    /// plan actually uses.
    #[test]
    fn an_armed_symbol_is_reached_through_the_spelling_a_plan_uses() {
        assert!(armed("argmax_logits").is_some());
        assert!(armed("copy_logits_bf16").is_some());
        // The stem lookup: the routine is `silu_mul` and the plan says this.
        assert!(armed("silu_mul_bfloat16").is_some());
        assert_eq!(
            armed("silu_mul_bfloat16").expect("armed").0.name,
            "silu_mul",
            "the symbol resolves to the routine its STEM names"
        );
        // Claimed by a longer stem with no arm.
        assert!(armed("silu_mul_strided_bfloat16").is_none());
        // Crossed, not armed: its body exists and the driver still plans it
        // from its row.
        assert!(armed("rms_single_row_bfloat16").is_none());
        assert!(armed("not_a_kernel").is_none());
        let both = kernels_wgpu::routines()
            .into_iter()
            .filter(|r| armed(r.name).is_some())
            .count();
        assert_eq!(both, 3);
    }

    fn sample_launch() -> Launch {
        Launch {
            kernel: 0,
            rows: 0..1,
            layers: 0..1,
            op: 0,
            args: 0..0,
            params: 0..0,
            peel: None,
            cond: Launch::NO_COND,
        }
    }
}
