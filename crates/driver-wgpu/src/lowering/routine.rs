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
    /// The body bound the parameter block TWICE in one dispatch.
    ///
    /// A block is one buffer and one bind-group position, so a second is not
    /// a shape this crate can stage. Unreachable today — an arm mints one
    /// handle for it — and refused by name rather than by `buffers` silently
    /// coming out one entry short.
    Blocks,
    /// The body wants a per-layer RECURRENT slab and this driver holds none.
    ///
    /// Separate from [`Unplanned::NoCache`] because the consequence differs:
    /// a KV cache the driver lacks is a deployment that cannot attend, while a
    /// recurrent carry it lacks is a scan that would read zero, write nothing
    /// back, and answer fluently and wrongly. Both refuse; only one of them
    /// would have been quiet.
    ///
    /// Every `ssm` arm declines here today, on this backend and on
    /// `driver-metal`, because neither allocates a slab. That is what keeps
    /// the gated DeltaNet honestly DARK rather than silently wrong, and
    /// `tests/hybrid_probe.rs` reads the decline by name.
    NoSlab {
        /// Which operand of the body asked.
        at: usize,
        /// The rectangle's layer.
        layer: u16,
        /// The slab's name, as the kernel knows it.
        which: &'static str,
    },
    /// The body asked for a KV LAYER this fire does not hold.
    ///
    /// The table path refuses the same thing as `Unbindable::NoKvCache`; this
    /// is that refusal with the body's own argument position in hand.
    NoCache {
        /// The body's argument position.
        at: usize,
        /// Which layer.
        layer: u16,
        /// Values rather than keys.
        values: bool,
    },
    /// The body asked for a fire TABLE this fire does not hold.
    ///
    /// A rope frequency run or a sampling index list the resolver has no
    /// buffer for. The table path refuses the same thing as
    /// `Unbindable::NoDriverResource`, and this is that refusal with the
    /// body's own argument position in hand.
    Absent {
        /// The body's argument position.
        at: usize,
        /// Which table it wanted.
        what: crate::binding::FireTable,
    },
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
            Self::Blocks => write!(f, "the body bound two parameter blocks"),
            Self::NoSlab { at, layer, which } => write!(
                f,
                "the body's operand {at} wants layer {layer}'s `{which}` slab, \
                 which this driver allocates none of"
            ),
            Self::NoCache { at, layer, values } => write!(
                f,
                "the body's operand {at} wants layer {layer}'s {}, which this \
                 fire does not hold",
                if *values { "values" } else { "keys" }
            ),
            Self::Absent { at, what } => write!(
                f,
                "the body's operand {at} wants the fire's {what:?}, which this \
                 fire does not hold"
            ),
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

/// What one of the body's handles resolved to.
///
/// Three answers where an `Option` gave two. The parameter BLOCK and an
/// UNBOUND binding both bind no buffer and mean opposite things: the first
/// takes a bind-group position and the second takes none.
#[derive(Clone, Copy)]
pub enum Placed<'a, B> {
    /// A real buffer.
    Buffer(Bound<'a, B>),
    /// The packed scalar run.
    Params,
    /// A binding the module declares and this statement does not fill.
    Nothing,
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
    bounds: &[Placed<'a, B>],
    scalars: &[u32],
    declared: &Declared,
    symbol: &'a str,
    launch: &Launch,
) -> Result<Dispatch<'a, B>, Unplanned> {
    // The PARAMETER BLOCK's place in the body's buffer list, if it asked for
    // one. Its handle resolves to no `Bound` — it is the packed scalar run and
    // the driver stages it — so it takes a position and no entry, which is
    // exactly what `Dispatch::block_at` means.
    let mut storage_block = None;
    let mut buffers = Vec::with_capacity(stated.handles.len());
    for (n, &h) in stated.handles.iter().enumerate() {
        let at = h as usize;
        let bound = bounds.get(at).ok_or(Unplanned::Handle {
            handle: h,
            minted: bounds.len(),
        })?;
        match bound {
            Placed::Buffer(bound) => buffers.push(*bound),
            // A binding the module declares and nothing fills: it takes a
            // position and no entry, exactly as `reorder`'s `Slot::Nothing`
            // does. NOT the params block — telling the two apart is why this
            // is an enum and not an `Option`, and conflating them made
            // `router_topk` refuse itself as two blocks.
            Placed::Nothing => {}
            Placed::Params => {
                if storage_block.is_some() {
                    return Err(Unplanned::Blocks);
                }
                storage_block = Some(n);
            }
        }
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

    // A STORAGE block, when the body asked for one.
    //
    // Two different things are called "the parameter block" on this backend
    // and the module decides which. A `@group(1)` uniform is built from the
    // scalars the BODY passed and is a bind group of its own; a `@group(0)`
    // storage entry is the statement's own scalar run, staged as a buffer, and
    // it takes a place in the numbering. `mlp`'s three gated activations are
    // the second kind — `gated.wgsl` declares `@group(0) @binding(3)
    // var<storage> params` — and the table path resolves them the same way,
    // through a row that states `Param(0): Buf`.
    if let Some(at) = storage_block {
        // The statement's run FIRST, then whatever the body passed beside the
        // block. That order is the table path's: `scalars` builds one run by
        // walking the row, taking the whole tail at a `Param(_): Buf` and
        // appending each derived number after it — `row_gather`'s request
        // count is exactly such a number, and its shader reads it as the last
        // field of the struct.
        //
        // Most bodies append nothing: an `Env` argument computes the grid and
        // never reaches `ctx.dispatch`, so `mlp`'s three gated activations
        // pass no scalars at all and the run is the statement's alone.
        let mut run: Vec<u8> = scalars.iter().flat_map(|w| w.to_le_bytes()).collect();
        // WHERE THE BODY'S SCALARS GO, when the module declares a uniform
        // block BESIDE the storage one. They are two different things and the
        // shader reads them from two different places: `ssm/gdn_prep.wgsl`'s
        // prefill pair takes `GdnCoreParams` as the `@group(0)` pointer the
        // statement assembled and `row_pitch`/`n_scan` as the two fields of
        // its `@group(1)` block. Appending them to the storage run leaves the
        // uniform empty, which `Device::check_bindable` refuses by name.
        //
        // Every other kernel here declares one or the other, which is why the
        // run took the whole of both until now: `row_gather`'s derived request
        // count really is the last field of its storage struct.
        let mut uniform = Vec::new();
        if declared.uniform_bytes > 0 {
            if bytes.len() > declared.uniform_bytes as usize {
                return Err(Unplanned::Scalars {
                    stated: bytes.len(),
                    room: declared.uniform_bytes,
                });
            }
            uniform = bytes;
            uniform.resize(declared.uniform_bytes as usize, 0);
        } else {
            run.extend_from_slice(&bytes);
        }
        let local = declared.local;
        return Ok(Dispatch {
            symbol,
            buffers,
            params: Params::Block {
                bytes: run,
                at: ParamSlot::Storage(u32::try_from(at).expect("a small binding number")),
            },
            uniform,
            block_at: Some(at),
            groups: [
                stated.lanes[0].div_ceil(local[0].max(1)),
                stated.lanes[1].div_ceil(local[1].max(1)),
                stated.lanes[2].div_ceil(local[2].max(1)),
            ],
            op: launch.op,
        });
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
        // The uniform rides `params` on this path; the field is for the
        // dispatch that has a storage block TOO.
        uniform: Vec::new(),
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
    lowered: &'a model_compiler::lower::Lowered,
    launch: &Launch,
    declared: &Declared,
    sources: crate::dispatch::Sources<'a, R>,
    facts: super::hold::Facts,
) -> Result<Vec<Dispatch<'a, R::Buffer>>, Unplanned> {
    let crate::dispatch::Sources {
        arena,
        resolver,
        min_offset,
    } = sources;
    let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
    let scalars = &lowered.params[launch.params.start as usize..launch.params.end as usize];
    // The fire's own numbers, asked for once and handed to the arm as a map.
    // A `Resolve` answers each independently and an arm that took the resolver
    // could ask for anything; this keeps the ask to the four an arm may want.
    let mut numbers = std::collections::BTreeMap::new();
    for which in [
        crate::binding::FireNumber::KvPageSize,
        crate::binding::FireNumber::KvHeadStride,
        crate::binding::FireNumber::KvSeqStride,
        crate::binding::FireNumber::AttentionMaskStride,
    ] {
        if let Some(n) = resolver.number(which) {
            numbers.insert(which, n);
        }
    }
    // TWO PASSES, and the first one exists because a TYPE cannot tell a
    // statement's result from a recurrent slab: both are `F32sMut`. Counting
    // the routine's writable arguments -- which `results` does and which this
    // used -- overcounts by one for every GDN kernel, because
    // `new_conv_state` is writable and is not one of the statement's outputs.
    // The split then steals an input, and the arm asks for a `b_gate` the
    // statement "does not carry".
    //
    // The SIGNATURE can tell them apart, because it says which is which: a
    // result is an `OutSlot` and a slab is a `Held<NewConvState, _>`. So the
    // binder runs once over an undivided statement purely to count the asks,
    // and once over the split that count implies. `driver-metal` reached the
    // same shape after the same defect and records it at its own call site.
    let asked = {
        let mut probe = super::hold::Handles::undivided(args, scalars);
        // A refusal HERE is not the answer: the counting pass may fail for a
        // reason the split pass would too, and the split pass is the one whose
        // refusal names the operand. So this one is swallowed and the count it
        // reached is used, which is the number of results it managed to ask
        // for before stopping.
        let _ = super::bind::bind(routine.args, routine.sources, &mut probe, facts);
        probe.asked_results()
    };
    let mut handles = super::hold::Handles::with_numbers(args, asked, scalars, &numbers);
    let taken_args = super::bind::bind(routine.args, routine.sources, &mut handles, facts)
        .map_err(Unplanned::Refused)?;
    let stated = state(routine, &taken_args)?;

    // The operands the BODY asked for, resolved in the order it asked. Not
    // the statement's order: `Handles` minted a handle per ask, and this
    // walks the same list. `None` is the parameter block, which has no `Arg`
    // to resolve — it holds its handle's place so the ones after it are right.
    let mut bounds = Vec::with_capacity(handles.asked().len());
    for (at, arg) in handles.asked().iter().enumerate() {
        match arg {
            super::hold::Asked::Operand(arg) => {
                let bound = crate::binding::resolve(arg, launch, arena, resolver, min_offset)
                    .map_err(|why| Unplanned::Operand { at, why })?;
                bounds.push(Placed::Buffer(bound));
            }
            super::hold::Asked::Params => bounds.push(Placed::Params),
            super::hold::Asked::Unbound => bounds.push(Placed::Nothing),
            super::hold::Asked::Kv { values } => {
                // The layer is the RECTANGLE's, read here rather than carried
                // by the arm: `reorder` takes `launch.layers.start` for
                // `Source::KvKeys` and an arm holding its own number could
                // disagree with the launch it is planning.
                let layer = launch.layers.start;
                let held = resolver.kv(layer, *values).ok_or(Unplanned::NoCache {
                    at,
                    layer,
                    values: *values,
                })?;
                bounds.push(Placed::Buffer(crate::binding::Bound::whole(held)));
            }
            super::hold::Asked::Slab(which) => {
                // The rectangle's layer, for `Asked::Kv`'s reason.
                let layer = launch.layers.start;
                let held =
                    resolver
                        .slab(layer, which)
                        .ok_or(Unplanned::NoSlab { at, layer, which })?;
                bounds.push(Placed::Buffer(crate::binding::Bound::whole(held)));
            }
            super::hold::Asked::Table(which) => {
                let held = resolver
                    .table(*which)
                    .ok_or(Unplanned::Absent { at, what: *which })?;
                bounds.push(Placed::Buffer(crate::binding::Bound::whole(held)));
            }
        }
    }

    let symbol = lowered.kernels[launch.kernel as usize].as_str();
    stated
        .iter()
        .map(|one| bind(one, &bounds, scalars, declared, symbol, launch))
        .collect()
}

/// The routine and the arm for a symbol, if this backend has crossed it AND
/// armed it.
///
/// Both halves or neither: a body with no arm cannot be given its operands,
/// and an arm with no body has nothing to hand them to. The table is empty and
/// every family has crossed, so `None` is no longer a fallback to the table
/// path — it is a symbol nothing can plan, and
/// `an_armed_symbol_is_reached_through_the_spelling_a_plan_uses` is what says
/// the census holds none.
#[must_use]
pub fn armed(symbol: &str) -> Option<&'static Routine<Wgpu>> {
    // By STEM, not by name. A plan spells `silu_mul_bfloat16` and the routine
    // is called `silu_mul`; the registry knows which prefix of a symbol names
    // a kernel, and it is the only thing that can, because this fork answers
    // before any row is looked up.
    let stem = super::hold::crossed(symbol)?;
    kernels_wgpu::routines().into_iter().find(|r| r.name == stem)
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
        let d: Dispatch<'_, ()> =
            bind(&stated, &[], &[], &declared, "e", &launch).expect("it packs");
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
        let out: Result<Dispatch<'_, ()>, _> = bind(&stated, &[], &[], &declared, "e", &launch);
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
        let routine = armed("argmax_logits").expect("the one armed symbol");
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
    /// A body that wants a KV LAYER this fire lacks is `Unplanned::NoCache`.
    ///
    /// The pool is per-LAYER state and the layer is the rectangle's, not the
    /// arm's: `plan` reads `launch.layers.start` when it resolves the ask, the
    /// same number `reorder` takes for `kernels::Source::KvKeys`. An arm
    /// carrying its own layer could disagree with the launch it is planning,
    /// and a paged decode reading the wrong layer's keys is a model that runs
    /// and answers from the wrong context.
    #[test]
    fn a_kv_layer_this_fire_does_not_hold_is_refused_by_name() {
        let args = [model_compiler::lower::Arg::Arena {
            at: 0,
            width: 64,
            bytes: 2,
        }];
        let mut o = super::super::hold::Handles::over(&args, 1);
        let _ = o.output(0).expect("the one operand");
        let keys = o.kv(false);
        let values = o.kv(true);
        assert!(
            matches!(keys, ArgValue::Buffer(1)) && matches!(values, ArgValue::Buffer(2)),
            "keys and values take their own handles: {keys:?}, {values:?}"
        );
        assert!(
            matches!(
                o.asked().get(2),
                Some(super::super::hold::Asked::Kv { values: true })
            ),
            "and the second is recorded as the VALUES half"
        );

        let why = Unplanned::NoCache {
            at: 1,
            layer: 7,
            values: false,
        };
        let said = why.to_string();
        assert!(
            said.contains("operand 1") && said.contains("layer 7") && said.contains("keys"),
            "the refusal names the position, the layer AND which half: {said}"
        );
    }

    /// A body that wants a fire TABLE this fire lacks is `Unplanned::Absent`.
    ///
    /// The rope frequencies, the sampling indices and the token ids belong to
    /// the FIRE, not to the statement, so an arm asks for them by name and the
    /// resolver may simply not hold one. The table path refuses the same thing
    /// as `Unbindable::NoDriverResource`; this is that refusal with the body's
    /// own argument position in hand, which is what a reader needs in order to
    /// know WHICH ask went unanswered.
    #[test]
    fn a_fire_table_this_fire_does_not_hold_is_refused_by_name() {
        let args = [model_compiler::lower::Arg::Arena {
            at: 0,
            width: 64,
            bytes: 2,
        }];
        let mut o = super::super::hold::Handles::over(&args, 1);
        let _ = o.output(0).expect("the one operand");
        let table = o.table(crate::binding::FireTable::RopeFrequencies);
        assert!(
            matches!(table, ArgValue::Buffer(1)),
            "the table takes the handle after the operand, got {table:?}"
        );
        assert!(
            matches!(
                o.asked().get(1),
                Some(super::super::hold::Asked::Table(
                    crate::binding::FireTable::RopeFrequencies
                ))
            ),
            "and it is recorded as a TABLE, which is what `plan` resolves \
             through the resolver rather than through the arena"
        );

        // The refusal `plan` raises when the resolver holds no such buffer.
        // Built here rather than driven through a fire, because a `Resolve`
        // that answers `None` for one table and real buffers for the rest is
        // more scaffolding than the claim is worth.
        let why = Unplanned::Absent {
            at: 1,
            what: crate::binding::FireTable::RopeFrequencies,
        };
        let said = why.to_string();
        assert!(
            said.contains("operand 1") && said.contains("RopeFrequencies"),
            "the refusal names the position AND the table: {said}"
        );
    }

    /// Every gated-DeltaNet arm declines its recurrent slab, BY NAME.
    ///
    /// The family is DARK on this backend and on `driver-metal`, and this is
    /// what makes that a statement rather than an accident. Neither driver
    /// allocates a slab, so `Resolve::slab` answers `None` and `plan` refuses
    /// with [`Unplanned::NoSlab`].
    ///
    /// **The alternative is what this backend used to do.** Its arms took
    /// `conv_state` from `input(1)` and `new_conv_state` from `output(3)`,
    /// because `Handles` had no slab door — and the statement `model-dsl`
    /// actually emits carries FOUR weights and THREE outputs, so `input(1)` is
    /// `a_gate` and `output(3)` does not exist. `gates` read `a_gate`/`b_gate`
    /// as `weight(4)`/`weight(5)`, which `driver-metal`'s own comment records
    /// as a defect it fixed: a weight handle there binds a per-head buffer the
    /// shader strides BY TOKEN, so row zero reads the right gate and every row
    /// after it reads past the end.
    ///
    /// A refusal is the only honest answer while the carry does not exist: a
    /// scan handed a null one reads zero, writes nothing back, and returns a
    /// fluent result no output check catches.
    #[test]
    fn every_gdn_arm_declines_the_slab_this_driver_does_not_hold() {
        let mut declined = Vec::new();
        for point in kernels_wgpu::entrypoints() {
            if !point.starts_with("gdn_") {
                continue;
            }
            let Some(stem) = super::super::hold::crossed(&point) else {
                panic!("`{point}` is claimed by no crossed stem");
            };
            let routine = kernels_wgpu::routines()
                .into_iter()
                .find(|r| r.name == stem)
                .unwrap_or_else(|| panic!("`{stem}` names no routine"));
            let args: Vec<model_compiler::lower::Arg> = (0..8)
                .map(|n| model_compiler::lower::Arg::Arena {
                    at: n * 256,
                    width: 256,
                    bytes: 256,
                })
                .collect();
            // A WHOLE `GdnCoreParams` and the scan's tile after it, because
            // that is what the statements carry and the arms read: `Dv` at 1
            // and `Hv` at 3 size the grid, and the prefill scan's `(lanes,
            // vrows)` at 11 and 12 pick which compiled shape fires. Four ones
            // left every arm refusing, and the guard below is what said so
            // rather than the suite going quietly green over nothing.
            let mut o = super::super::hold::Handles::with_scalars(
                &args,
                3,
                // Dk, Dv, Hk, Hv, conv_dim, Kc, q_off, k_off, v_off, eps,
                // inv_sqrt_dk, then lanes and vrows.
                &[128, 128, 4, 4, 1024, 4, 0, 512, 1024, 0, 0, 32, 4],
            );
            let facts = super::super::hold::facts(
                &point,
                4,
                crate::dispatch::Geometry {
                    q_heads: 4,
                    kv_heads: 4,
                    head_dim: 64,
                    rotary_dims: 64,
                    n_experts: 0,
                    experts_per_token: 0,
                    ..Default::default()
                },
                4,
                256,
                256,
            );
            // The arm may refuse for its own reasons; what matters is that
            // when it SUCCEEDS it recorded a slab ask, so `plan` will refuse.
            if crate::lowering::bind::bind(routine.args, routine.sources, &mut o, facts).is_ok() {
                let asked = o.asked();
                let slab = asked
                    .iter()
                    .position(|a| matches!(a, super::super::hold::Asked::Slab(_)));
                let at = slab.unwrap_or_else(|| {
                    panic!(
                        "`{point}`'s arm filled every operand without asking \
                         for a recurrent slab, which means it took the carry \
                         from somewhere that is not one"
                    )
                });
                // The refusal `plan` will raise for that ask, constructed here
                // so this test NAMES it: `Resolve::slab` answers `None` on
                // this driver, so the ask cannot become a binding.
                let why = Unplanned::NoSlab {
                    at,
                    layer: 0,
                    which: "conv_state",
                };
                assert!(
                    why.to_string().contains("allocates none of"),
                    "the refusal should say the driver holds no slab: {why}"
                );
                declined.push(point.clone());
            }
        }
        assert!(
            !declined.is_empty(),
            "no gdn arm was exercised, so this checked nothing"
        );
    }

    /// A body that binds two parameter blocks is `Unplanned::Blocks`.
    ///
    /// One block is one buffer and one bind-group position. A second would
    /// leave `buffers` an entry short of what the body meant, and the shader
    /// would read some other operand's bytes as its scalars — a dispatch that
    /// runs and answers wrongly. Unreachable through `Handles`, which mints
    /// one handle for the block, so this drives `bind` directly.
    #[test]
    fn two_parameter_blocks_in_one_dispatch_are_refused_by_name() {
        let stated = Stated {
            module: "m".to_owned(),
            entrypoint: "e".to_owned(),
            lanes: [1, 1, 1],
            handles: vec![0, 1],
            scalars: Vec::new(),
        };
        let out: Result<Dispatch<'_, ()>, _> = bind(
            &stated,
            &[Placed::Params, Placed::Params],
            &[7],
            &declared(0, [1, 1, 1]),
            "e",
            &sample_launch(),
        );
        assert!(
            matches!(out, Err(Unplanned::Blocks)),
            "expected two blocks to be refused, got {out:?}"
        );
    }

    // RETIRED with `Unplanned::Both`, which turned out to be the wrong rule.
    //
    // It asserted that a body binding a storage block AND passing scalars is
    // refused, on the reasoning that the block IS the statement's run.
    // `layout::row_gather` does exactly that on purpose -- its request count
    // is a FIELD of the struct, and the table path appends the same number to
    // the same run -- so the refusal was rejecting a correct body. `bind` now
    // appends the body's scalars to the statement's run, and the 24
    // `row_gather` rectangles of
    // `every_launchs_scalars_land_where_its_module_reads_them` agree with the
    // row on every byte.

    /// A body that stated no dispatch asked for nothing, and a caller that
    /// took that for done would run a launch that writes nothing and reports
    /// success — which is the same failure a zero grid is, arrived at from
    /// the other side.
    #[test]
    fn a_body_that_states_no_dispatch_is_not_taken_for_done() {
        // An untouched planner has nothing to state, and that is exactly the
        // condition `state` raises `Unplanned::Silent` on.
        assert!(Planner::new().stated().is_empty());

        // And a body that REFUSED must not have recorded one either -- the
        // two are told apart by which error comes back, not by the
        // recording, so both have to be empty for that distinction to mean
        // anything.
        let routine = armed("argmax_logits").expect("the one armed symbol");
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
            // Empty beside an empty `args`, which is the pair's own rule: all
            // three are DERIVED from the body's arguments, and this body has
            // none.
            sides: &[],
            sources: &[],
            spelling: &[],
            body: says_nothing,
            whole: false,
            depth_prefix_plan: false,
            in_place: &[],
            derived: &[],
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
            bind(&stated, &[], &[], &declared(8, [1, 1, 1]), "e", &launch);
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
        let argmax = armed("argmax_logits").expect("the one armed symbol");
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

        let routine = armed("argmax_logits").expect("the one armed symbol");
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
            super::super::hold::facts("x", 7, crate::dispatch::Geometry::default(), 1, 1024, 1024),
        );
        assert!(
            matches!(out, Err(Unplanned::Operand { at: 0, .. })),
            "expected the FIRST ask to be named, got {out:?}"
        );
    }

    /// The armed symbols are crossed, and reached through the spelling a
    /// plan actually uses.
    ///
    /// [`armed`] is [`super::super::hold::crossed`] plus one lookup: the
    /// registry answers a ROUTINE NAME, and this finds the routine carrying
    /// it. `hold.rs`'s `the_armed_stems_are_the_ones_registered_and_nothing_
    /// else` owns the first half over the whole census. What is only
    /// checkable here is the JOIN, and it fails quietly: a stem whose
    /// `routine:` override names a body `kernels_wgpu::routines()` does not
    /// carry answers `None` with no error anywhere, and now that the table is
    /// empty a symbol that answers `None` is a symbol NO path can plan.
    ///
    /// Both traps below are live, and a lookup that falls into either still
    /// returns `Some` and still plans a real dispatch:
    ///
    /// * `affine_qmv_routed_bfloat16_gs_64_b_4` is spelled with a
    ///   QUANTIZATION SCHEME its routine's name never carries -- the body is
    ///   `qmv_routed` -- so a lookup that matched on the routine's own name
    ///   would find nothing here. That is `kernels-metal::kernel_of`'s defect,
    ///   which cost 363 of 479 entrypoints, and this crate reproduced it
    ///   twice.
    ///
    /// * `silu_mul` is a strict prefix of `silu_mul_strided` and both are
    ///   armed, so a first-match lookup hands a STRIDED rectangle to the
    ///   contiguous body: a flat grid where the shader wants rows, and every
    ///   row past the first read from the wrong offset. Both operands are
    ///   storage buffers of the same length, so nothing downstream sees it.
    ///
    /// This test used to assert `armed("silu_mul_strided_bfloat16").is_none()`
    /// and to count the routines whose own NAME is claimable. The first was a
    /// fact about the roster -- `silu_mul_strided` was the fleet's last
    /// unarmed kernel -- and stopped being true the day it was armed; the
    /// second is the trap itself wearing a number, and is kept below only as
    /// the inequality it always was.
    #[test]
    fn an_armed_symbol_is_reached_through_the_spelling_a_plan_uses() {
        // The symbol a plan spells, and the routine whose body must run.
        for (symbol, body) in [
            ("argmax_logits", "argmax_logits"),
            ("argmax_logits_bfloat16", "argmax_logits"),
            ("copy_logits_bf16", "copy_logits_bf16"),
            // The nesting trap, in both directions: the shorter stem must not
            // claim the longer symbol, and the longer must not shadow the
            // shorter.
            ("silu_mul_bfloat16", "silu_mul"),
            ("silu_mul_strided_bfloat16", "silu_mul_strided"),
            // The scheme prefix, which the routine's name never carries.
            ("affine_qmv_routed_bfloat16_gs_64_b_4", "qmv_routed"),
            (
                "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
                "mxfp4_qmv_routed_bias",
            ),
            ("sdpa_paged_decode_bfloat16_d_128", "sdpa_paged_decode"),
        ] {
            let routine = armed(symbol)
                .unwrap_or_else(|| panic!("`{symbol}` is a symbol this backend plans"));
            assert_eq!(
                routine.name, body,
                "`{symbol}` reached `{}`, and its body is `{body}`: a lookup \
                 that lands on the wrong routine binds real buffers, \
                 dispatches, and answers from the wrong shape",
                routine.name
            );
        }

        // A ROUTINE'S OWN NAME IS NOT A SPELLING ANY PLAN USES. `qmv_routed`
        // is what the body is called and `affine_qmv_routed_...` is what
        // `lowered.kernels` holds; nothing states the former, so answering it
        // would mean the lookup had started matching on names instead of on
        // symbols -- and the same change would silently unclaim all 30
        // routines whose spelling carries a prefix.
        assert!(
            armed("qmv_routed").is_none(),
            "a routine's own name is not a symbol, and answering it means the \
             lookup is matching on the body rather than on the plan"
        );
        assert!(
            armed("qmv_routed_bfloat16_gs_64_b_4").is_none(),
            "the scheme is part of the spelling: `affine_` and `mxfp4_` name \
             different weight layouts through one body"
        );
        // A stem may not end mid-word.
        assert!(armed("silu_multiply").is_none());
        // And a name no backend has.
        assert!(armed("not_a_kernel").is_none());

        // THE JOIN, over the census. Every symbol the registry claims has to
        // arrive at a routine that EXISTS: `find(|r| r.name == stem)` is an
        // `Option` and its `None` is indistinguishable, from the outside, from
        // a symbol nobody armed. A `routine:` override with a typo in it, or a
        // body renamed in `kernels-wgpu` without its registration, lands here
        // and nowhere else.
        let points = kernels_wgpu::entrypoints();
        let lost: Vec<&String> = points
            .iter()
            .filter(|p| super::super::hold::crossed(p).is_some() && armed(p).is_none())
            .collect();
        assert!(
            lost.is_empty(),
            "{} entrypoints are claimed by an armed stem whose routine this \
             crate cannot find, starting with {:?}. The registry names a body \
             `kernels_wgpu::routines()` does not carry, and the symbol is \
             unplannable by any path.",
            lost.len(),
            &lost[..lost.len().min(4)],
        );

        // COUNTING BY ROUTINE NAME IS THE TRAP ITSELF, so it is asserted as
        // the inequality and not as a number: a routine spelled with a scheme
        // prefix cannot answer to its own name, and a suite that measured
        // coverage this way would read the fleet as a third short while every
        // kernel was in fact reachable. The count that means something is the
        // one over SYMBOLS, and it is the sweep above.
        let routines = kernels_wgpu::routines();
        let by_name = routines.iter().filter(|r| armed(r.name).is_some()).count();
        assert!(
            by_name < routines.len(),
            "{by_name} of {} routines answer to their own name. If that ever \
             becomes all of them, either every scheme prefix has left the \
             spellings or the lookup has started matching on names -- and the \
             second is the defect this test is named for",
            routines.len()
        );
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
