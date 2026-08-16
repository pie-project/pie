//! THE SEAM VOCABULARY — the named extension points a declaration states
//! and a lowering reads.
//!
//! It sat inside the authoring module (`dsl::seam`) while there was one
//! crate, and that was the single real back edge in the whole toolchain:
//! [`crate::trace::TraceBuilder::finish`] calls [`check_plan`] and
//! `model-compiler`'s lowering reads [`OUT`], so both the producer of the
//! traced form and its consumer reached into a module whose other 7,700
//! lines are authoring surface. Nothing here names a `Val`, a weight
//! handle or any other authoring type — a seam is IR, so this is where it
//! lives, and moving it is what let the three crates layer at all.
//!
//! What did NOT come along is the RECORDER (`dsl::seam(..)`, the free
//! function that lowers a seam statement onto a tape): it takes `&Val` and
//! is therefore surface. The split is exactly the declaration/recording
//! line — the words are here, the act of writing them is in `model-dsl`.
//!
//! V2 (north-star-dsl.md "V2 — THE REDESIGN"): the seam vocabulary.
//!
//! A seam is a named, typed, identity-by-default extension point in the
//! model text — the ONE surface behind what were three mechanisms (the
//! two [`crate::trace::HookStage`]s, the `HasLora` guard arm, and the
//! dispatch-side prologue/epilogue stages). At THIS rung only the
//! surface unifies: each seam lowers to exactly the op(s) the pre-seam
//! text recorded, and the goldens pin that byte-identity. The IR's own
//! Seam op and the signature-table ABI are later rungs; what changes
//! now is that the model text states extension points in one vocabulary
//! instead of naming mechanisms.

/// What an attachment at a seam MAY do. Caps are the seam's
/// interface; whether a given deployment can service a cap is a
/// dispatch-table fact refused at load (the XQA-has-no-capture
/// contract's future home — enforcement lands with the signature
/// ABI). The vocabulary documents the gradient: pure expressions
/// innermost, observation mid-body, full PTIR at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    /// Pure value rewrite `|x, y| y'` (the adapter family).
    Transform,
    /// Read the seam's value from an attached program.
    Observe,
    /// Read the attention scores the capturing dispatch published.
    Scores,
    /// Narrow the page list the stated attention kernel consumes.
    PageMaskSink,
    /// Device puts (embeds, channels) — boundary-only.
    Put,
    /// Draw samples from the logits — boundary-only.
    Sample,
    /// Emit host-visible outputs — boundary-only.
    Emit,
}

/// A seam's SIGNATURE (`.wiki/tart/dsl.md` ①): the stable NAME the
/// request surface keys on (`fwd.adapter("attn.qv", ..)`,
/// `fwd.attach(..)`), what an attachment SEES, what it MAY do, and —
/// for the seams that have one — where it sits and where its output
/// lands.
///
/// `after` / `before` and `sink` are the two lines the doc singles
/// out as carrying what is today only a comment. They are not
/// documentation here: [`check_plan`] reads `after` / `before`.
pub struct Def {
    pub name: &'static str,
    /// The value roles an attachment observes or rewrites, in
    /// operand order.
    pub sees: &'static [&'static str],
    pub caps: &'static [Cap],
    /// The seam's POSITION rule, for seams whose arithmetic depends
    /// on it. `after` names the op kinds that must have produced the
    /// values it sees; `before` names the op kinds that must not yet
    /// have consumed them.
    pub position: Option<Position>,
    /// Where a sink-writing attachment's output lands.
    pub sink: Option<&'static str>,
}

/// A seam's position rule, stated as op-kind names
/// ([`crate::trace::OpKind`]'s discriminants, plus `"Launch:<symbol>"`
/// for stated kernels).
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub after: &'static [&'static str],
    pub before: &'static [&'static str],
}

/// Pre-attention observation seam: sees the just-projected q; a
/// page-mask-sink attachment narrows the page list the SAME stated
/// attention kernel consumes as substituted arguments (today's
/// `OnAttnProj`).
pub const ATTN_Q: Def = Def {
    name: "attn.q",
    sees: &["q"],
    caps: &[Cap::Observe, Cap::PageMaskSink],
    position: None,
    // Where Quest's `attn_page_mask` lands. Hardcoded in
    // `emit_cuda::emit_masked_pages_bracket` today; declared here,
    // consumed when the launch ABI flattens (migration step 6).
    sink: Some("attention.pages"),
};

/// Post-attention observation seam: sees the scores the (possibly
/// capturing) dispatch published through the sideband (today's
/// `OnAttn`).
pub const ATTN_OUT: Def = Def {
    name: "attn.out",
    sees: &["a"],
    caps: &[Cap::Observe, Cap::Scores],
    position: None,
    sink: None,
};

/// The adapter value seam over the raw q/v projections — pure
/// expressions of `(x, y)`, `fwd.adapter`'s site family (today's
/// `HasLora` guard arm).
/// THE POSITION RULE IS THE POINT: the correction lands on the raw
/// projections, before bias, norms, rope and the KV append. Applying
/// it after rope is DIFFERENT ARITHMETIC — the bug the first live
/// A/B caught. It was a comment until now.
pub const ATTN_QV: Def = Def {
    name: "attn.qv",
    sees: &["q", "v"],
    caps: &[Cap::Transform],
    position: Some(Position {
        after: &["Matmul", "SplitQkv"],
        before: &["AddBias", "Rmsnorm", "Rope", "KvAppend", "Launch"],
    }),
    sink: None,
};

/// Entry boundary seam (prologue's home). Boundary attachments
/// never enter row signatures — they cause no divergence — which is
/// why their dispatch-side lowering needs no trace op at any rung.
pub const IN: Def = Def {
    name: "in",
    sees: &[],
    caps: &[Cap::Put, Cap::Emit],
    position: None,
    sink: None,
};

/// Exit boundary seam (epilogue's home).
pub const OUT: Def = Def {
    name: "out",
    sees: &["logits"],
    caps: &[Cap::Observe, Cap::Sample, Cap::Put, Cap::Emit],
    position: None,
    sink: None,
};

/// LOAD-TIME check of the seams a text stated.
///
/// One rule today, and it is the one whose violation is silent:
/// [`ATTN_QV`]'s position. The adapter's delta must land on the base
/// projection, not on base + bias, and not after rope — so between
/// the ops that PRODUCE the values the seam sees and the seam's own
/// statement, nothing may consume them. A live A/B caught exactly
/// this once; the rule stops being a comment here.
pub fn check_plan(plan: &crate::trace::ForwardPlan) -> Vec<String> {
    let mut problems = Vec::new();
    for stmt in &plan.seams {
        let Some(def) = by_name(&stmt.seam) else {
            problems.push(format!(
                "{}: states seam `{}`, which no seam! signature declares",
                plan.family, stmt.seam
            ));
            continue;
        };
        let (Some(pos), Some(at)) = (def.position, stmt.op) else {
            continue;
        };
        let at = at as usize;
        // The values this statement sees are the inputs of the op
        // it carries (the adapter's guard opens at `at`; its
        // correction launch is the next op and names q and v).
        let Some(seen) = plan.ops.get(at + 1).map(|op| op.inputs.clone()) else {
            continue;
        };
        for &v in &seen {
            let produced_at = plan.ops.iter().position(|op| op.outputs.contains(&v));
            match produced_at {
                None => problems.push(format!(
                    "{}: seam `{}` sees value {v}, which no op produces",
                    plan.family, def.name
                )),
                Some(from) => {
                    let producer = kind_name(&plan.ops[from].kind);
                    if !pos.after.contains(&producer) {
                        problems.push(format!(
                            "{}: seam `{}` must sit after {:?}, but value {v} \
                             comes from {producer}",
                            plan.family, def.name, pos.after
                        ));
                    }
                    for (i, op) in plan.ops.iter().enumerate().take(at).skip(from + 1) {
                        if !op.inputs.contains(&v) {
                            continue;
                        }
                        let consumer = kind_name(&op.kind);
                        if pos.before.contains(&consumer) {
                            problems.push(format!(
                                "{}: seam `{}` must sit before {consumer}, but op \
                                 {i} consumes value {v} first — different \
                                 arithmetic, not a reordering",
                                plan.family, def.name
                            ));
                        }
                    }
                }
            }
        }
    }
    problems
}

/// Every seam a model text may state.
pub const ALL: &[&Def] = &[&IN, &ATTN_QV, &ATTN_Q, &ATTN_OUT, &OUT];

pub fn by_name(name: &str) -> Option<&'static Def> {
    ALL.iter().copied().find(|d| d.name == name)
}

fn kind_name(kind: &crate::trace::OpKind) -> &'static str {
    use crate::trace::OpKind as K;
    match kind {
        K::Embed { .. } => "Embed",
        K::Matmul { .. } => "Matmul",
        K::SplitQkv { .. } => "SplitQkv",
        K::Rope { .. } => "Rope",
        K::Rmsnorm { .. } => "Rmsnorm",
        K::AddBias { .. } => "AddBias",
        K::KvAppend { .. } => "KvAppend",
        K::Launch { .. } => "Launch",
        K::Guard { .. } => "Guard",
        K::Peel { .. } => "Peel",
        K::HookSite { .. } => "HookSite",
        _ => "other",
    }
}

#[cfg(test)]
mod seam_tests {
    use crate::seam;
    use crate::trace::{ForwardPlan, GuardArm, GuardPred, Op, OpKind, SeamStatement};

    fn op(kind: OpKind, inputs: Vec<u32>, outputs: Vec<u32>) -> Op {
        Op {
            kind,
            inputs,
            outputs,
            layer: Some(0),
        }
    }

    fn matmul() -> OpKind {
        OpKind::Matmul {
            weight: "layer.0.q_proj".to_string(),
            beta_one: false,
            selector: None,
        }
    }

    fn lora() -> OpKind {
        OpKind::Launch {
            kernel: "gemm::lora_qkv_correction".to_string(),
            weights: vec![],
            state: None,
            params: vec![],
            param_extents: vec![],
        }
    }

    fn guard() -> OpKind {
        OpKind::Guard {
            arms: vec![GuardArm {
                pred: GuardPred::HasLora,
                ops: 1,
            }],
            else_ops: 0,
        }
    }

    /// `q` and `v` from projections, seam immediately after: the shape
    /// every live text has.
    fn well_placed() -> Vec<Op> {
        vec![
            op(matmul(), vec![], vec![1]),
            op(matmul(), vec![], vec![2]),
            op(guard(), vec![], vec![]),
            op(lora(), vec![1, 2], vec![]),
        ]
    }

    fn plan(ops: Vec<Op>) -> ForwardPlan {
        ForwardPlan {
            family: "test".to_string(),
            values: vec![],
            ops,
            depth_window: false,
            seams: vec![SeamStatement {
                seam: "attn.qv".to_string(),
                layer: Some(0),
                op: Some(2),
                // This fixture exists for the adapter POSITION rule, which
                // reads `layer`/`op` and never the exposed set.
                values: vec![],
            }],
        }
    }

    /// The adapter's position rule FIRES. Without this the live traces'
    /// clean check proves only that the walk found nothing to look at.
    #[test]
    fn the_adapter_position_rule_is_not_vacuous() {
        assert!(seam::check_plan(&plan(well_placed())).is_empty());

        // A bias consuming q BEFORE the seam: the delta would land on
        // base + bias. This is the shape the live A/B caught.
        let mut ops = well_placed();
        ops.insert(
            2,
            op(
                OpKind::AddBias {
                    weight: "layer.0.q_bias".to_string(),
                },
                vec![1],
                vec![3],
            ),
        );
        let mut p = plan(ops);
        p.seams[0].op = Some(3);
        let problems = seam::check_plan(&p);
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(problems[0].contains("AddBias"), "{}", problems[0]);

        // A seam placed after rope: different arithmetic, and the
        // producer is no longer a projection.
        let ops = vec![
            op(matmul(), vec![], vec![1]),
            op(matmul(), vec![], vec![2]),
            op(
                OpKind::Rope {
                    kind: crate::trace::RopeKind::Standard,
                    partial: None,
                },
                vec![1],
                vec![3],
            ),
            op(guard(), vec![], vec![]),
            op(lora(), vec![3, 2], vec![]),
        ];
        let mut p = plan(ops);
        p.seams[0].op = Some(3);
        let problems = seam::check_plan(&p);
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(problems[0].contains("Rope"), "{}", problems[0]);
    }

    /// Every seam a text may state is declared, and its `sees` arity is
    /// what the statement passes.
    #[test]
    fn the_seam_table_is_complete() {
        for d in seam::ALL {
            assert_eq!(seam::by_name(d.name).map(|x| x.name), Some(d.name));
        }
        assert_eq!(seam::ATTN_QV.sees, &["q", "v"]);
        assert!(
            seam::ATTN_Q.sink.is_some(),
            "the page-mask sink is declared"
        );
    }
}
