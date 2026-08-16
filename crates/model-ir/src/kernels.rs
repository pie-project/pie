//! Kernel signatures, from the compiler's side.
//!
//! Contracts live beside their kernels ([`kernels_cuda::sigs`], and for Metal
//! the routines behind [`kernels_metal::declared`]); [`Stated`] is the shape
//! both planes answer in. What lives here is compiler-side: [`Backend`] and
//! [`check_plan`]. The tables are consumed with `default-features = false`,
//! so reading a contract builds no `.cu`.

use crate::trace::{ForwardPlan, Op, OpKind};

pub use kernels::{Cap, KernelSig};
pub use kernels_cuda::sigs;
/// Every entrypoint Metal can dispatch, and so everything a text may launch.
pub use kernels_metal::entrypoints as metal_entrypoints;

/// Which backend's kernels a lowered trace states.
///
/// Two variants, four execution shells: this names the surface a text was
/// written against, not the device that runs it. Vulkan and WGPU execute
/// plans traced by `llama_like_metal`, so they are checked against Metal's
/// statements — sound only while the three tables stay equal, which
/// `kernels/tests/shader_backends_agree.rs` holds. Hence no `Backend::Vulkan`:
/// no family name could construct one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// `kernels-cuda`'s table.
    Cuda,
    /// `kernels-metal`'s table.
    Metal,
}

impl Backend {
    /// `None` for a semantic trace, which states no kernels at all.
    pub fn of_family(family: &str) -> Option<Backend> {
        let mut parts = family.split('.').skip(1);
        match parts.next() {
            Some("cuda") => Some(Backend::Cuda),
            Some("metal") => Some(Backend::Metal),
            _ => None,
        }
    }
}

/// What a backend states about one symbol, with the plane forgotten — a CUDA
/// row and a Metal [`kernels::routine::Declared`] answer here alike.
#[derive(Debug, Clone, Copy)]
pub struct Stated {
    /// Consumes its whole operand, not a row range.
    pub whole: bool,
    /// Pairs the depth-prefix plan, and its workspace, on a union's tail
    /// layers. A kernel property, not an op's.
    pub depth_prefix_plan: bool,
    /// `(output, input)` pairs that must be given the same address.
    ///
    /// Output first. Both are `u32`, so a swap is silent.
    pub in_place: &'static [(u32, u32)],
    /// The routine's arguments, in signature order, and who supplies each.
    /// Derived from the `fn`, which is what makes the arity rule cost no new
    /// column. Empty means the plane states no signature, and the rule then
    /// has nothing to compare.
    pub args: &'static [(kernels::Ty, kernels::routine::Provenance)],
    /// Which side of the statement each argument sits on, in [`Self::args`]
    /// order. A `*mut` in an input slot is an operand the statement places,
    /// not a result it forgot. See [`kernels::routine::Side`].
    pub sides: &'static [kernels::routine::Side],
}

impl Stated {
    /// The pointers a statement must place: `(reads, writes)`, before the
    /// in-place slack and optional ceiling the caller adds.
    ///
    /// `(0, 0)` both when the plane states no signature and when a routine
    /// binds nothing; check [`Self::args`] for emptiness to tell them apart.
    #[must_use]
    pub fn places(&self) -> (usize, usize) {
        use kernels::routine::Provenance;

        let (mut reads, mut writes) = (0, 0);
        for (ty, prov) in self.args {
            match (ty.binds(), prov) {
                (kernels::Binds::Reads, Provenance::Trace) => reads += 1,
                (kernels::Binds::Writes, Provenance::Trace) => writes += 1,
                _ => {}
            }
        }
        (reads, writes)
    }
}

/// What `backend` states about `symbol`, or `None` if nothing does.
///
/// Metal resolves through [`kernels_metal::kernel_of`], so a text spelling a
/// base name and a lowering carrying an instantiated point agree.
#[must_use]
pub fn stated_in(backend: Backend, symbol: &str) -> Option<Stated> {
    match backend {
        Backend::Cuda => kernels::sig_in(sigs(), symbol).map(|k| Stated {
            whole: k.whole,
            depth_prefix_plan: k.depth_prefix_plan,
            in_place: k.in_place,
            args: k.args,
            sides: k.sides,
        }),
        Backend::Metal => {
            // Memoised: `declared()` rebuilds its `Vec` on every call, and
            // this is asked once per launched op of every model that loads.
            static ROUTINES: std::sync::OnceLock<
                std::collections::BTreeMap<&'static str, kernels::routine::Declared>,
            > = std::sync::OnceLock::new();
            let routines = ROUTINES.get_or_init(|| {
                kernels_metal::declared()
                    .into_iter()
                    .map(|d| (d.name, d))
                    .collect()
            });

            let name = kernels_metal::kernel_of(symbol)?;
            routines
                .get(name)
                .map(|d| Stated {
                    whole: d.whole,
                    depth_prefix_plan: d.depth_prefix_plan,
                    in_place: d.in_place,
                    args: d.args,
                    sides: d.sides,
                })
                // Metal's `silu_mul_strided` has no routine: its entrypoint
                // leaves a buffer slot empty, so it has no positional
                // argument list. `driver-metal` refuses to lower it.
                .or(Some(Stated {
                    whole: false,
                    depth_prefix_plan: false,
                    in_place: &[],
                    args: &[],
                    sides: &[],
                }))
        }
    }
}

/// The row for one CUDA symbol, for readers that want the row itself: the
/// CUDA emitter and the table tests.
///
/// Anything the compiler decides goes to [`stated_in`] instead — this answers
/// `None` for every Metal symbol, which does not mean nothing declares it.
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    kernels::sig_in(sigs(), symbol)
}

/// Symbols the arity rule cannot hold. Empty: each former entry encoded an
/// operation in its operand count, and was fixed by splitting the symbol.
pub const ARITY_EXCEPTIONS: &[&str] = &[];

/// For one statement: does it place what the routine reads and writes?
///
/// `None` when it agrees, when the symbol is excepted, or when the plane
/// states no signature.
///
/// An in-place pair is slack on the read side rather than a subtraction,
/// because a routine may spell the alias as one `*mut` parameter or as a
/// `*const`/`*mut` pair: placed reads run from `inputs + weights - in_place`
/// to `inputs + weights`. A [`Provenance::Either`] argument raises the
/// ceiling by one and the floor by nothing. `in_value_region` says the guard
/// owns the result and the launch records none, so the write floor drops to
/// zero — the ceiling still holds, since declaring more results than the
/// routine writes is wrong in a region too.
fn arity_problem(
    k: &Stated,
    kernel: &str,
    weights: usize,
    op: &Op,
    in_value_region: bool,
) -> Option<String> {
    use kernels::routine::{Provenance, Side};

    if k.args.is_empty() || ARITY_EXCEPTIONS.contains(&kernel) {
        return None;
    }
    let (mut reads, mut writes) = (0usize, 0usize);
    let (mut opt_reads, mut opt_writes) = (0usize, 0usize);
    for (i, (ty, prov)) in k.args.iter().enumerate() {
        // The slot answers first, the pointer only where no slot does:
        // `norm::residual_add_rmsnorm`'s `hidden: In<0, *mut T>` is an
        // operand written through, not an undeclared result.
        let binds = match k.sides.get(i) {
            Some(Side::Placed) => kernels::Binds::Reads,
            Some(Side::Declared) => kernels::Binds::Writes,
            Some(Side::OfType) | None => ty.binds(),
        };
        // An `Env` argument is the fire's by declaration, whatever its type:
        // a page index the runtime always has and no trace ever names.
        match (binds, prov) {
            (_, Provenance::Env) => {}
            (kernels::Binds::Reads, Provenance::Trace) => reads += 1,
            (kernels::Binds::Writes, Provenance::Trace) => writes += 1,
            (kernels::Binds::Reads, Provenance::Either) => opt_reads += 1,
            (kernels::Binds::Writes, Provenance::Either) => opt_writes += 1,
            (kernels::Binds::Nothing, _) => {}
        }
    }
    // Read side only, and the asymmetry is deliberate: a pair is
    // `(output, input)`, so its output half is already in `op.outputs` and
    // the write side compares flat. Parameters that write STATE rather than a
    // result -- `rope::rope_write_kv_bf16`'s `k_pages` -- are counted on
    // neither side; the vocabulary cannot yet declare them.
    let aliased = k.in_place.len().min(op.inputs.len());
    let placed = op.inputs.len() + weights;
    let read_problem = if reads > placed || reads + opt_reads < placed - aliased {
        Some(format!(
            "`{kernel}` reads {reads} pointer{} but the statement places {placed} \
             ({} input{}, {weights} weight{}{}); a builder and an arm that disagree about \
             operand COUNT disagree about every position after the first",
            plural(reads),
            op.inputs.len(),
            plural(op.inputs.len()),
            plural(weights),
            if aliased > 0 {
                format!(", {aliased} of them possibly in place")
            } else {
                String::new()
            },
        ))
    } else {
        None
    };
    let declared = op.outputs.len();
    let floor = if in_value_region { 0 } else { writes };
    let write_problem = if declared < floor || declared > writes + opt_writes {
        Some(format!(
            "`{kernel}` writes {}{writes} pointer{} but the statement declares {declared} \
             result{}",
            if opt_writes > 0 { "at least " } else { "" },
            plural(writes),
            plural(declared),
        ))
    } else {
        None
    };

    match (read_problem, write_problem) {
        (None, None) => None,
        (Some(r), None) => Some(r),
        (None, Some(w)) => Some(w),
        (Some(r), Some(w)) => Some(format!("{r}; and separately, {w}")),
    }
}

/// `""` or `"s"`.
const fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// Load-time check of a traced form against the kernel table, catching three
/// failures that would otherwise surface at runtime:
///
/// 1. a `whole` kernel stated inside a [`OpKind::Peel`] region, which gives
///    it a row window it cannot honour;
/// 2. a launched symbol nothing declares;
/// 3. operand counts that disagree with the routine's.
///
/// The table is [`Backend::of_family`]'s answer, which is the authoring
/// surface, not the executing device — see [`Backend`] before trusting this
/// for Vulkan or WGPU. Reads are compared against `inputs + weights`
/// together, since a weight and an input are both `const T*`.
///
/// Returns the failures rather than panicking, so a caller can name the
/// family it was loading.
pub fn check_plan(plan: &ForwardPlan) -> Vec<String> {
    let mut problems = Vec::new();
    let backend = Backend::of_family(&plan.family);
    // Countdowns over the flat op list; both kinds' regions are consecutive
    // and sit immediately after the op.
    let mut peeled = 0usize;
    // A value-producing guard owns its outputs, so the launches inside it
    // record none and their writes must not be compared against their own
    // `outputs`. Stays zero for a guard that declares nothing.
    let mut regioned = 0usize;
    for op in &plan.ops {
        let inside_peel = peeled > 0;
        let inside_value_region = regioned > 0;
        peeled = peeled.saturating_sub(1);
        regioned = regioned.saturating_sub(1);
        match &op.kind {
            OpKind::Peel {
                prefix_ops,
                tail_ops,
                ..
            } => {
                peeled = peeled.max(*prefix_ops as usize + *tail_ops as usize);
            }
            OpKind::Guard { arms, else_ops } if !op.outputs.is_empty() => {
                let span = arms.iter().map(|a| a.ops as usize).sum::<usize>()
                    + *else_ops as usize;
                regioned = regioned.max(span);
            }
            OpKind::Launch {
                kernel, weights, ..
            } => match backend.and_then(|b| stated_in(b, kernel)) {
                None => problems.push(format!(
                    "{}: launches `{kernel}`, which no {} kernel declares",
                    plan.family,
                    match backend {
                        Some(b) => format!("{b:?}").to_lowercase(),
                        // A semantic trace states no kernels.
                        None => "backend's".to_string(),
                    }
                )),
                Some(k) if k.whole && inside_peel => problems.push(format!(
                    "{}: `{kernel}` is declared `whole` but is stated inside a Peel \
                     region, which gives it a row window it cannot honour",
                    plan.family
                )),
                Some(k) => {
                    // A `scale.` name is a constant riding the weight slot,
                    // not a pointer: it reaches the arm through
                    // `DispatchCtx::scales`, so counting it as an operand
                    // would make `norm::scalar_mul_bf16` look one short.
                    let bound = weights.iter().filter(|w| !w.starts_with("scale.")).count();
                    if let Some(why) =
                        arity_problem(&k, kernel, bound, op, inside_value_region)
                    {
                        problems.push(format!("{}: {why}", plan.family));
                    }
                }
            },
            _ => {}
        }
    }
    problems
}

/// Which outputs a stated kernel writes over which inputs. Takes the plan
/// because the answer is the backend's.
pub fn in_place_pairs(plan: &ForwardPlan, kernel: &str) -> &'static [(u32, u32)] {
    Backend::of_family(&plan.family)
        .and_then(|b| stated_in(b, kernel))
        .map_or(&[][..], |s| s.in_place)
}

/// Which outputs a semantic op writes over which inputs.
///
/// Takes no backend by claim, not convenience: these follow from what the
/// kind means, so a backend that disagreed would not be implementing it.
pub fn semantic_in_place(kind: &OpKind) -> &'static [(u32, u32)] {
    match kind {
        // Rope rotates in place; the trace's SSA names for rotated q and k
        // are two names for one buffer.
        OpKind::Rope { .. } => &[(0, 0), (1, 1)],
        // `C = A·Bᵀ + C`, so C is read as well as written and the residual it
        // folds must be C. Only when folded.
        OpKind::Matmul { beta_one: true, .. } => &[(0, 1)],
        // `attn_out *= sigmoid(gate)`; the gate is read-only.
        OpKind::SigmoidGateMul => &[(0, 0)],
        // `x[r, :] += bias`, and the kernel has no destination parameter.
        OpKind::AddBias { .. } => &[(0, 0)],
        // `stream += branch`: the arms take a destination and an addend with
        // no separate source. `norm::residual_add_bf16`'s row says the same,
        // but rows are consulted only for `OpKind::Launch`, so without this
        // `alias_owners` lets the sum land wherever the arena pointed.
        //
        // Safe for a backend whose kernel is not in place: elementwise
        // `out[i] = in0[i] + in1[i]` still holds when `out` aliases `in0`.
        OpKind::ResidualAdd => &[(0, 0)],
        _ => &[],
    }
}
