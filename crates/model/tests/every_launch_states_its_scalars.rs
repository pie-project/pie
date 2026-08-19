//! **Does every launch a real model lowers carry the scalars its routine declares?**
//!
//! `Const<i32>` is a promise: *the statement carries this number, positionally,
//! at its slot in the params run*. `check_plan` checks that promise, but only
//! where a statement has begun to keep it — its rule is guarded by
//! `!params.is_empty()`, so a statement carrying NO scalars against a routine
//! declaring three is excused rather than refused (see `.wiki/migration.md`
//! §11.12, "the check is much less total than it looks").
//!
//! That guard is why the CUDA e2e path died at its FIRST kernel with
//! `layout::embed_bf16: the fire does not carry a statement parameter`, and
//! why nothing in the workspace saw it coming: every unit test in `kernels-cuda`
//! checks the routine's own arithmetic, and every unit test in `model-ir`
//! checks a hand-built plan. Neither lowers a real declaration and asks the
//! table what it wants.
//!
//! This does. For every catalog row that traces on CUDA, in every fire class,
//! it lowers the plan and compares each launch's params run against the arity
//! its routine's derived column declares — including the launches that come
//! from SEMANTIC ops, which is the half `check_plan` structurally cannot see:
//! only `OpKind::Launch` carries a `params` vector at all, so an
//! `OpKind::Embed` lowers to `layout::embed_bf16` with an empty run no matter
//! what that routine asks for.
//!
//! A row here is a routine whose scalars nothing states. The fix is one of
//! two, decided per routine and recorded in the migration notes: teach the
//! builder to state them, or -- where the op that fires it has no params
//! channel at all and so no trace text can keep the promise -- move the fact
//! into the body as an `ask`.

use std::collections::BTreeMap;

use model::catalog::{self, Deployed};
use model_compiler::lower::{Fire, Row, lower};
use model_ir::kernels::{Backend, stated_in};
use model_ir::trace::FireClass;

/// A decode-shaped fire: `n` rows, every one sampled.
fn sampled(n: usize) -> Vec<Row> {
    vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ]
}

/// Every fire class a row is traced in. A scalar can be missing in one shape
/// and stated in another -- prefill and decode reach different symbols.
const CLASSES: &[FireClass] = &[FireClass::Decode, FireClass::Prefill];

/// Stated names that stand for a CHOICE rather than a routine.
///
/// `attn::write_kv_to_pages` is declared by an `untraced!` row so `check_plan`
/// can measure a model text against a name every text spells; the driver's
/// `Boot::route` resolves it to one of two real routines from the KV dtype the
/// boot settled. The declaration row has no `Const` at all, so measuring the
/// stated name alone reports a clean run for a launch that refuses — which is
/// exactly what happened: the audit passed while the sixth launch of every
/// fire died on five unstated scalars.
///
/// Both alternatives are listed because the audit cannot know which boot will
/// win, and a scalar unstated for either is unstated.
const RESOLVES_TO: &[(&str, &[&str])] = &[(
    "attn::write_kv_to_pages",
    &[
        "attn::write_kv_to_pages_bf16",
        "attn::write_kv_to_pages_quantised",
    ],
)];

/// Every routine a stated name can actually run.
fn concrete(symbol: &str) -> Vec<&str> {
    RESOLVES_TO
        .iter()
        .find(|(s, _)| *s == symbol)
        .map_or_else(|| vec![symbol], |(_, to)| to.to_vec())
}

/// Which routines are missing which scalars, over every row that traces.
///
/// Keyed by symbol rather than by row: the defect belongs to the routine and
/// its builder, and reporting it once per model would print the same repair
/// two hundred times.
fn shortfalls() -> BTreeMap<String, (usize, usize, Vec<String>)> {
    let mut out: BTreeMap<String, (usize, usize, Vec<String>)> = BTreeMap::new();
    for v in catalog::catalog() {
        for &class in CLASSES {
            let Ok(plan) = v.trace(class, Deployed::single()) else {
                continue;
            };
            // Four rows: enough that a per-request axis is not one, small
            // enough that a lowering is cheap over the whole catalog.
            let Ok(low) = lower(&plan, &sampled(4), Fire::default()) else {
                continue;
            };
            for l in &low.launches {
                let Some(stated) = low.kernels.get(l.kernel as usize) else {
                    continue;
                };
                for kernel in concrete(stated) {
                    let Some(k) = stated_in(Backend::Cuda, kernel) else {
                        continue;
                    };
                    let wants = k.scalars();
                    let has = l.params.len();
                    if has < wants {
                        let e = out
                            .entry(kernel.to_string())
                            .or_insert_with(|| (wants, has, Vec::new()));
                        // The SHORTEST run seen wins: that is the one that refuses.
                        e.1 = e.1.min(has);
                        let is_launch = matches!(
                            plan.ops.get(l.op as usize).map(|o| &o.kind),
                            Some(model_ir::trace::OpKind::Launch { .. })
                        );
                        let who = format!(
                            "{} {:?} [{}]",
                            v.id(),
                            class,
                            if is_launch { "STATED" } else { "SEMANTIC" }
                        );
                        if !e.2.contains(&who) {
                            e.2.push(who);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Routines whose shortfall is OLDER than the four marks, with the reason.
///
/// An allowlist is a claim about each entry, not a mute button, so each one
/// says what it is waiting for and can be checked against HEAD.
const PRE_EXISTING: &[(&str, &str)] = &[
    // `s` was `Param<0, f32>` before the marks changed too, and
    // `dsl::cuda::scalar_mul`'s `by: Option<f32>` has always had `None`
    // callers: gemma-2's per-layer `query_scale` and gemma-3n's
    // `laurel_scale` name a `scale.<name>` WEIGHT and derive no number, so
    // the run has been empty since before this migration began. The repair
    // is a fact on the two families' catalog rows — gemma-2-27b's
    // `query_pre_attn_scalar` is 144, which is neither the head dim nor the
    // hidden size over the head count, so nothing already on the row implies
    // it — and that is a catalog change, not a marks one.
    ("norm::scalar_mul_bf16", "gemma-2 / gemma-3n state a scale WEIGHT and no number; \
                               pre-dates the four marks (HEAD: `Param<0, f32>`, same empty run)"),
];

/// The gate: nothing a real model fires may want a scalar its statement does
/// not carry.
#[test]
fn every_launch_carries_the_scalars_its_routine_declares() {
    let bad: BTreeMap<_, _> = shortfalls()
        .into_iter()
        .filter(|(sym, _)| !PRE_EXISTING.iter().any(|(s, _)| s == sym))
        .collect();
    assert!(
        bad.is_empty(),
        "{} routine(s) are fired with a params run shorter than their signature \
         declares. Each would refuse at the fire with \"the fire does not carry \
         a statement parameter\":\n{}",
        bad.len(),
        bad.iter()
            .map(|(sym, (wants, has, who))| {
                format!(
                    "  {sym}: wants {wants} scalar(s), the statement carries {has} \
                     (e.g. {})",
                    who.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The same question asked of the OPERANDS, which is a different failure with
/// the same cause.
///
/// A mark says which slot of which kind a parameter is, and `resolve` numbers
/// them by position — so `InOut` is not a spelling convenience: it claims the
/// statement placed an INPUT there as well as a result, and derives
/// `Source::Alias(i, o)` that reads input `i`. Put it on a result the
/// statement never paired with an input and the launch refuses with "the fire
/// does not carry an input operand", which is what `mlp::chunked_swiglu_bf16`
/// did at the tenth launch of every llama fire: its result is HALF the width
/// of its operand, so it can never have been the same buffer, and HEAD
/// spelled it `Out<0, T>`.
///
/// Counting is the whole check. A statement places `op.inputs` inputs and
/// `outs` results; a column asking for index `n` of either needs `n + 1`.
#[test]
fn every_launch_places_the_operands_its_routine_reaches_for() {
    use model_ir::kernels::{Kind, Source};

    let mut bad: BTreeMap<String, (String, Vec<String>)> = BTreeMap::new();
    for v in catalog::catalog() {
        for &class in CLASSES {
            let Ok(plan) = v.trace(class, Deployed::single()) else {
                continue;
            };
            let Ok(low) = lower(&plan, &sampled(4), Fire::default()) else {
                continue;
            };
            for l in &low.launches {
                let Some(stated) = low.kernels.get(l.kernel as usize) else {
                    continue;
                };
                let Some(op) = plan.ops.get(l.op as usize) else {
                    continue;
                };
                let n_in = op.inputs.len();
                let n_out = if op.outputs.is_empty() && !op.dest.is_empty() {
                    op.dest.len()
                } else {
                    op.outputs.len()
                };
                for kernel in concrete(stated) {
                    let Some(k) = stated_in(Backend::Cuda, kernel) else {
                        continue;
                    };
                    for s in k.sources {
                        let short = match s {
                            Some(Source::Alias(i, o)) => {
                                (usize::from(*i) >= n_in).then(|| {
                                    format!(
                                        "`InOut` reaches for input {i} (and result {o}) \
                                         where the statement places {n_in} input(s)"
                                    )
                                })
                            }
                            Some(Source::Slot(Kind::In, n)) => (usize::from(*n) >= n_in)
                                .then(|| {
                                    format!(
                                        "reaches for input {n} where the statement \
                                         places {n_in}"
                                    )
                                }),
                            Some(Source::Slot(Kind::Out, n)) => (usize::from(*n) >= n_out)
                                .then(|| {
                                    format!(
                                        "reaches for result {n} where the statement \
                                         declares {n_out}"
                                    )
                                }),
                            _ => None,
                        };
                        if let Some(why) = short {
                            let e = bad
                                .entry(kernel.to_string())
                                .or_insert_with(|| (why, Vec::new()));
                            let who = format!("{} {:?}", v.id(), class);
                            if !e.1.contains(&who) {
                                e.1.push(who);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} routine(s) reach for an operand their statement does not place:\n{}",
        bad.len(),
        bad.iter()
            .map(|(sym, (why, who))| format!(
                "  {sym}: {why} (e.g. {})",
                who.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
            ))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// An excuse that has stopped being needed is a lie about the code, so the
/// allowlist is checked in the other direction too.
#[test]
fn no_pre_existing_entry_has_gone_stale() {
    let bad = shortfalls();
    let stale: Vec<_> = PRE_EXISTING
        .iter()
        .filter(|(s, _)| !bad.contains_key(*s))
        .map(|(s, _)| *s)
        .collect();
    assert!(
        stale.is_empty(),
        "these no longer fall short and should leave `PRE_EXISTING`: {stale:?}"
    );
}

/// Not vacuous: the catalog really does trace and lower on CUDA here, so an
/// empty complaint list means "checked and clean" rather than "checked
/// nothing".
#[test]
fn the_catalog_actually_lowers_on_cuda() {
    let mut lowered = 0usize;
    let mut launches = 0usize;
    for v in catalog::catalog() {
        for &class in CLASSES {
            let Ok(plan) = v.trace(class, Deployed::single()) else {
                continue;
            };
            if let Ok(low) = lower(&plan, &sampled(4), Fire::default()) {
                lowered += 1;
                launches += low.launches.len();
            }
        }
    }
    assert!(
        lowered >= 20 && launches >= 1_000,
        "the sweep lowered {lowered} plan(s) and {launches} launch(es); it is \
         measuring nothing"
    );
}
