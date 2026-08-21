//! Kernel signatures, from the compiler's side.
//!
//! Contracts live beside their kernels ([`kernels_cuda::sigs`], and for Metal
//! the routines behind [`kernels_metal::declared`]); [`Stated`] is the shape
//! both planes answer in. What lives here is compiler-side: [`Backend`] and
//! [`check_plan`]. The tables are consumed with `default-features = false`,
//! so reading a contract builds no `.cu`.

use crate::trace::{ForwardPlan, Op, OpKind};

pub use kernels::{Cap, KernelSig};
// The two a `Stated::sources` entry is made of. `Stated` already exposes the
// column publicly, so a consumer that wants to READ it needed a second path
// to the same crate to name what it found; these are that path.
pub use kernels::{Kind, Source};
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
    /// The routine's arguments, in signature order, and who supplies each.
    /// Derived from the `fn`, which is what makes the arity rule cost no new
    /// column. Empty means the plane states no signature, and the rule then
    /// has nothing to compare.
    pub args: &'static [kernels::Ty],
    /// Which side of the statement each argument sits on, in [`Self::args`]
    /// order. A `*mut` in an input slot is an operand the statement places,
    /// Where each argument comes from, in [`Self::args`] order.
    ///
    /// Carries the aliasing now: a [`kernels::Source::Alias`] is one address
    /// wearing an operand slot and a result slot, which is what `in_place`
    /// used to say on the row. [`Self::in_place`] reads them back out.
    pub sources: &'static [Option<kernels::Source>],
    /// Each parameter's name and NULLABILITY, in [`Self::args`] order.
    ///
    /// A nullable operand is an OPTIONAL one, and that is the whole of why
    /// this column is carried here: `Provenance::Either` used to say it on a
    /// wrapper, and when `Env` and its provenance went the claim went with
    /// them — so every routine with a bias plane the statement may omit
    /// started reading one operand more than the text places.
    ///
    /// Empty where the plane states no column, which reads as *"nothing is
    /// optional"* — the same answer an all-required row gives.
    pub derived: &'static [kernels::Derived],
}

/// What one signature asks the statement to place, required and optional apart.
///
/// `Provenance::Either` was the old name for the optional half, and it lived
/// on a WRAPPER (`Unbound<T>`) rather than on the operand. It is the carrier's
/// fact — a `MaybeConst<T>` is a plane that may not exist — so it is read off
/// the same column the names come from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pointers {
    /// Pointers the statement MUST place.
    pub reads: usize,
    /// Pointers the statement MAY place: a nullable operand binds null.
    pub optional_reads: usize,
    /// Results the statement must declare.
    pub writes: usize,
    /// Results it may declare.
    pub optional_writes: usize,
}

impl Stated {
    /// The `(output, input)` pairs that must be given the same address.
    ///
    /// DERIVED, where it used to be stated. The pairs came off the row as
    /// `in_place = &[(0, 0)]`, forty-five of them, written forty lines from
    /// the parameters they indexed; they come off the SIGNATURE now, through
    /// the [`kernels::routine::InOut`] mark that wears both slots.
    ///
    /// Output first, as the readers expect.
    #[must_use]
    pub fn in_place(&self) -> Vec<(u32, u32)> {
        self.sources
            .iter()
            .filter_map(|s| match s {
                Some(kernels::Source::Alias(i, o)) => Some((u32::from(*o), u32::from(*i))),
                _ => None,
            })
            .collect()
    }
}

impl Stated {
    /// The pointers a statement must place: `(reads, writes)`, before the
    /// in-place slack and optional ceiling the caller adds.
    ///
    /// # Read off the SOURCE column, which is now the only column that says
    ///
    /// It used to read `args`' `Provenance` beside a `Side`, and both are
    /// deleted: with `Env` out of the parameter list every parameter is the
    /// statement's, so the provenance column had one value at every row, and
    /// the mark says which side. The `Source` a mark resolves to carries the
    /// same fact and cannot drift from it — `Slot(Kind::In, _)` and
    /// `Slot(Kind::Weight, _)` are placed, `Slot(Kind::Out, _)` is declared,
    /// and `Alias(_, _)` is one address wearing both.
    ///
    /// THE PARAMS RUN IS COUNTED SEPARATELY and not here: a `Const` scalar is
    /// the statement's too, but it rides the `params` field rather than the
    /// operand list, so counting it as a pointer would make every routine that
    /// takes one read one operand more than the statement carries. See
    /// [`Self::scalars`].
    ///
    /// `(0, 0)` both when the plane states no signature and when a routine
    /// binds nothing; check [`Self::args`] for emptiness to tell them apart.
    #[must_use]
    pub fn places(&self) -> (usize, usize) {
        let p = self.pointers();
        (p.reads, p.writes)
    }

    /// [`Self::places`], with the optional half kept apart.
    #[must_use]
    pub fn pointers(&self) -> Pointers {
        use kernels::{Kind, Source};

        let mut p = Pointers::default();
        for (at, source) in self.sources.iter().enumerate() {
            // A NULLABLE OPERAND IS AN OPTIONAL ONE. The carrier says it —
            // `Const<Tensor<MaybeConst<T>>>` is a bias plane the export may
            // not have — and `#[routine]` reads it off the syntax into this
            // column. A statement that omits one is short by design, so it is
            // counted where a short count is allowed.
            let optional = self.derived.get(at).is_some_and(|d| d.nullable);
            match source {
                // A WEIGHT IS A READ, positional since the named half of the
                // old chain retired with the semantic ops.
                Some(Source::Slot(Kind::In | Kind::Weight, _)) => {
                    if optional {
                        p.optional_reads += 1;
                    } else {
                        p.reads += 1;
                    }
                }
                Some(Source::Slot(Kind::Out, _)) => {
                    if optional {
                        p.optional_writes += 1;
                    } else {
                        p.writes += 1;
                    }
                }
                // ONE ADDRESS IN TWO SLOTS, counted on the WRITE side only.
                // The statement placed it once; counting it on both would make
                // every aliasing routine read one operand more than there is.
                // The input slot it also wears is an ALIASING fact, which
                // `in_place` reads out of the same source.
                Some(Source::Alias(..)) => p.writes += 1,
                _ => {}
            }
        }
        p
    }

    /// How many scalars the statement must carry in its `params` run.
    ///
    /// THE COUNT THE PARAMS RUN NEVER HAD. `Param<N, T>` was deleted and the
    /// eighteen scalars that used it became `fact!(stated ..)` keys resolving
    /// through `Source::Named`, which no driver answers — so a statement that
    /// passed two scalars to a routine that takes three bound a zero at run
    /// time instead of being refused at plan time. `Const<i32>` gives the run
    /// a mark again, and a mark gives it an arity.
    ///
    /// The highest slot claimed plus one, not the number of `Const` marks:
    /// `Kind::Param` and `Kind::ParamF32` are two READINGS of one channel, and
    /// a routine may read a slot twice.
    #[must_use]
    pub fn scalars(&self) -> usize {
        use kernels::{Kind, Source};

        self.sources
            .iter()
            .filter_map(|s| match s {
                Some(Source::Slot(Kind::Param | Kind::ParamF32, n)) => Some(usize::from(*n) + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
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
            args: k.args,
            sources: k.sources,
            derived: k.derived,
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
                    args: d.args,
                    sources: d.sources,
                    derived: d.derived,
                })
                // Metal's `silu_mul_strided` has no routine: its entrypoint
                // leaves a buffer slot empty, so it has no positional
                // argument list. `driver-metal` refuses to lower it.
                .or(Some(Stated {
                    whole: false,
                    depth_prefix_plan: false,
                    args: &[],
                    sources: &[],
                    derived: &[],
                }))
        }
    }
}

/// The canon claim `backend`'s row for `symbol` makes, if any — the
/// reverse of [`canon_symbol`], for readers that hold a symbol and want
/// its role (the seam rules).
#[must_use]
pub fn claim_of(backend: Backend, symbol: &str) -> Option<&'static str> {
    match backend {
        Backend::Cuda => kernels::sig_in(sigs(), symbol).and_then(|k| k.canon),
        Backend::Metal => {
            let name = kernels_metal::kernel_of(symbol)?;
            kernels_metal::declared()
                .into_iter()
                .find(|d| d.name == name)
                .and_then(|d| d.canon)
        }
    }
}

/// The symbol `backend` CLAIMS for one canon point (`"rmsnorm"`,
/// `"rmsnorm.gemma"`), or `None` when nothing claims it.
///
/// THE tier-1 resolution: `dsl::rmsnorm` reads the family's backend off the
/// trace and asks here, so the symbol table lives on the routines
/// (`#[routine(canon = ..)]`) and nowhere in the DSL. A claim shared by two
/// routines of one backend is a bug the tables' tests refuse; first match
/// answers here.
#[must_use]
pub fn canon_symbol(backend: Backend, claim: &str) -> Option<&'static str> {
    match backend {
        Backend::Cuda => sigs()
            .iter()
            .find(|k| k.canon == Some(claim))
            .map(|k| k.symbol),
        Backend::Metal => {
            static CLAIMS: std::sync::OnceLock<
                std::collections::BTreeMap<String, &'static str>,
            > = std::sync::OnceLock::new();
            CLAIMS
                .get_or_init(|| {
                    kernels_metal::declared()
                        .into_iter()
                        .filter_map(|d| {
                            d.canon.map(|c| (c.to_string(), d.name))
                        })
                        .collect()
                })
                .get(claim)
                .copied()
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


/// Is `symbol`'s params run one the DSL has not finished stating?
///
/// A metal statement names an INSTANTIATED POINT, and a routine is stamped
/// over a product of axes; [`kernels_metal::kernel_of`] is the same map
/// [`stated_in`] uses to find the signature, so the allowlist is read through
/// it and one entry covers one routine however many points it has.
#[must_use]

/// For one statement: does it place what the routine reads and writes?
///
/// `None` when it agrees, when the symbol is excepted, or when the plane
/// states no signature.
///
/// An in-place pair is slack on the read side rather than a subtraction,
/// because a routine may spell the alias as one `*mut` parameter or as a
/// `*const`/`*mut` pair: placed reads run from `inputs + weights - in_place`
/// to `inputs + weights`. `in_value_region` says the guard owns the result and
/// the launch records none, so the write floor drops to zero — the ceiling
/// still holds, since declaring more results than the routine writes is wrong
/// in a region too.
///
/// # There is no optional half any more
///
/// `Provenance::Either` used to raise the read ceiling by one and the floor by
/// nothing, for the nullable spellings and for `keys::Unstated`. Both are gone:
/// every parameter of a columned routine is the statement's, and a parameter
/// nothing supplies is a bare pointer the launcher fills itself, which claims
/// no slot and so is not counted at all. What is left is one count per side,
/// read off the same `Source` column the binder walks.
///
/// # And the params run is checked the same way now
///
/// [`Stated::scalars`] is the arity the signature declares for the statement's
/// `params`, which nothing checked before — the run lost its mark when
/// `Param<N, T>` was deleted, so a statement that passed two scalars to a
/// routine taking three bound a zero at run time. `Const<i32>` gives it back,
/// and a statement short of what the signature claims is refused here.
fn arity_problem(
    k: &Stated,
    kernel: &str,
    weights: usize,
    op: &Op,
    in_value_region: bool,
) -> Option<String> {
    if k.args.is_empty() || ARITY_EXCEPTIONS.contains(&kernel) {
        return None;
    }
    let p = k.pointers();
    let (reads, writes) = (p.reads, p.writes);
    let (opt_reads, opt_writes) = (p.optional_reads, p.optional_writes);
    // Read side only, and the asymmetry is deliberate: a pair is
    // `(output, input)`, so its output half is already in `op.outputs` and
    // the write side compares flat. Parameters that write STATE rather than a
    // result -- `rope::rope_write_kv_bf16`'s `k_pages` -- are counted on
    // neither side; the vocabulary cannot yet declare them.
    let aliased = k.in_place().len().min(op.inputs.len());
    let placed = op.inputs.len() + weights;
    let read_problem = if reads > placed || reads + opt_reads < placed.saturating_sub(aliased) {
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
                kernel,
                weights,
                params,
                ..
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
                    // would make `norm::scalar_mul` look one short.
                    let bound = weights.iter().filter(|w| !w.starts_with("scale.")).count();
                    if let Some(why) =
                        arity_problem(&k, kernel, bound, op, inside_value_region)
                    {
                        problems.push(format!("{}: {why}", plan.family));
                    }
                    // THE PARAMS RUN, CHECKED THE WAY THE OPERANDS ALREADY
                    // WERE. Nothing checked it before: the run lost its mark
                    // when `Param<N, T>` was deleted, so a statement short of
                    // what the signature claims bound a zero at run time and
                    // the fire ran on it. `Const<i32>` gives the run an arity,
                    // and this is where a statement short of it is refused --
                    // at plan time, naming both counts.
                    //
                    // # ONLY WHERE THE STATEMENT HAS STARTED STATING THEM
                    //
                    // A run of length zero is not a short run: it is a symbol
                    // whose scalars a hand ARM supplies, which is most of them
                    // while the DSL stage of this migration is outstanding.
                    // `model-ir` cannot tell the two apart -- whether a symbol
                    // is bound from its column or fired by an arm is recorded
                    // in `driver-cuda/src/bind/arms/`, which this crate does
                    // not see and must not.
                    //
                    // So the check is aimed at the failure it can actually
                    // establish, which is also the one §5.8 of
                    // `.wiki/migration.md` names: a statement that carries SOME
                    // scalars and not all of them. `dequant_fp8_e4m3`'s
                    // `vec![0, rows, cols]` is that shape -- a placeholder hole
                    // held open so `rows` would land at index 1 -- and a
                    // builder that adds a `Const` without adding its value is
                    // caught here rather than at the fire.
                    //
                    // WHAT COMPLETES IT: when `model-dsl` states every `Const`
                    // for a symbol, drop the `!params.is_empty()` guard and the
                    // check becomes total.
                    let wants = k.scalars();
                    if !params.is_empty() && params.len() < wants {
                        problems.push(format!(
                            "{}: `{kernel}` takes {wants} scalar{} in its params run but the \
                             statement carries {}; a missing scalar is not an absence, it is \
                             a zero at that slot",
                            plan.family,
                            plural(wants),
                            params.len(),
                        ));
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
pub fn in_place_pairs(plan: &ForwardPlan, kernel: &str) -> Vec<(u32, u32)> {
    Backend::of_family(&plan.family)
        .and_then(|b| stated_in(b, kernel))
        .map(|s| s.in_place())
        .unwrap_or_default()
}

/// Which outputs a semantic op writes over which inputs.
///
/// RETIRED with the semantic vocabulary: every statement is a
/// [`OpKind::Launch`] now, and aliasing comes off the routine's own column
/// ([`in_place_pairs`]). Kept because `Select` still launches nothing and
/// callers still ask; it answers empty for everything.
pub fn semantic_in_place(kind: &OpKind) -> &'static [(u32, u32)] {
    let _ = kind;
    &[]
}
