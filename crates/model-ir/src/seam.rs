//! Seam vocabulary: named extension points stated by model text and
//! read by lowerings.
//!

/// What an attachment at a seam may do.
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

/// A seam signature: name, observed value roles, capabilities, position
/// rule, and optional sink.
pub struct Def {
    pub name: &'static str,
    /// Value roles in operand order.
    pub sees: &'static [&'static str],
    pub caps: &'static [Cap],
    /// Position rule for seams whose arithmetic depends on placement.
    pub position: Option<Position>,
    /// Where a sink-writing attachment's output lands.
    pub sink: Option<&'static str>,
}

/// A seam position rule, stated as op-kind names.
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub after: &'static [&'static str],
    pub before: &'static [&'static str],
}

/// Pre-attention observation seam over q; a page-mask sink narrows the
/// page list consumed by the same stated attention kernel.
pub const ATTN_Q: Def = Def {
    name: "attn.q",
    sees: &["q"],
    caps: &[Cap::Observe, Cap::PageMaskSink],
    position: None,
    // Quest's `attn_page_mask` lands in the attention page-list sink.
    sink: Some("attention.pages"),
};

/// Post-attention observation seam over published scores.
pub const ATTN_OUT: Def = Def {
    name: "attn.out",
    sees: &["a"],
    caps: &[Cap::Observe, Cap::Scores],
    position: None,
    sink: None,
};

/// Adapter value seam over raw q/v projections.
/// The correction must land before bias, norms, rope and KV append;
/// after rope is different arithmetic.
pub const ATTN_QV: Def = Def {
    name: "attn.qv",
    sees: &["q", "v"],
    caps: &[Cap::Transform],
    // CANON ROLES, not op kinds: the semantic vocabulary retired, so a
    // producer is discriminated by what its LAUNCH claims. `after` admits
    // the projection GEMMs (any `gemm::` symbol — the quantized rows claim
    // no role) and the packed split; `before` refuses the transforms whose
    // consumption would make the correction different arithmetic.
    position: Some(Position {
        after: &["matmul", "split_qkv"],
        before: &["add_bias", "rmsnorm", "rope", "kv_append"],
    }),
    sink: None,
};

/// Entry boundary seam; boundary attachments do not enter row signatures.
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

/// Load-time check of stated seams.
/// The silent case is [`ATTN_QV`]: its delta must land on the base
/// projection before consumers such as bias or rope.
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
        // Adapter guard at `at`; correction launch at `at + 1` names q and v.
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
                    if !admits(plan, &plan.ops[from].kind, pos.after) {
                        problems.push(format!(
                            "{}: seam `{}` must sit after {:?}, but value {v} \
                             comes from {}",
                            plan.family,
                            def.name,
                            pos.after,
                            kind_name(&plan.ops[from].kind),
                        ));
                    }
                    for (i, op) in plan.ops.iter().enumerate().take(at).skip(from + 1) {
                        if !op.inputs.contains(&v) {
                            continue;
                        }
                        if admits(plan, &op.kind, pos.before) {
                            problems.push(format!(
                                "{}: seam `{}` must sit before {:?}, but op \
                                 {i} consumes value {v} first — different \
                                 arithmetic, not a reordering",
                                plan.family, def.name, pos.before
                            ));
                        }
                    }
                }
            }
        }
    }
    problems
}

/// Whether `kind` matches one of `roles` — canon roles, matched against
/// what the op's launch CLAIMS.
///
/// Three admits per role, in order: the backend's claim for the kernel
/// (axis points count by their role prefix — `matmul.acc` is a `matmul`),
/// the backend-less `canon::<role>` spelling, and — for `matmul` alone —
/// the `gemm::` namespace, because the quantized projection rows claim no
/// role and a per-symbol list here would be the drift this file refuses.
fn admits(plan: &crate::trace::ForwardPlan, kind: &crate::trace::OpKind, roles: &[&str]) -> bool {
    let crate::trace::OpKind::Launch { kernel, .. } = kind else {
        return roles.contains(&kind_name(kind));
    };
    let claim = crate::kernels::Backend::of_family(&plan.family)
        .and_then(|b| crate::kernels::claim_of(b, kernel));
    for role in roles {
        if let Some(c) = claim
            && (c == *role || c.split('.').next() == Some(role))
        {
            return true;
        }
        if kernel.strip_prefix("canon::") == Some(role) {
            return true;
        }
        if *role == "matmul" && kernel.starts_with("gemm::") {
            return true;
        }
    }
    false
}

/// Every seam a model text may state.
pub const ALL: &[&Def] = &[&IN, &ATTN_QV, &ATTN_Q, &ATTN_OUT, &OUT];

pub fn by_name(name: &str) -> Option<&'static Def> {
    ALL.iter().copied().find(|d| d.name == name)
}

fn kind_name(kind: &crate::trace::OpKind) -> &'static str {
    use crate::trace::OpKind as K;
    match kind {
        K::Launch { .. } => "Launch",
        K::Guard { .. } => "Guard",
        K::Peel { .. } => "Peel",
        K::HookSite { .. } => "HookSite",
        K::Select { .. } => "Select",
        K::LmHead { .. } => "LmHead",
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
            dest: Vec::new(),
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
            peel_slots: None,
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

    /// Projection outputs, seam immediately after.
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
                // The adapter position rule ignores this exposed set.
                values: vec![],
            }],
        }
    }

    /// The adapter's position rule is not vacuous.
    #[test]
    fn the_adapter_position_rule_is_not_vacuous() {
        assert!(seam::check_plan(&plan(well_placed())).is_empty());

        // Bias consumes q before the seam: delta would land on base + bias.
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

        // After rope is different arithmetic and a different producer.
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

    /// Stated seams are declared with the expected arity.
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
