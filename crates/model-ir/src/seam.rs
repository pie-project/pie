#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    Transform,

    Observe,

    Scores,

    PageMaskSink,

    Put,

    Sample,

    Emit,
}

pub struct Def {
    pub name: &'static str,

    pub sees: &'static [&'static str],
    pub caps: &'static [Cap],

    pub position: Option<Position>,

    pub sink: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub after: &'static [&'static str],
    pub before: &'static [&'static str],
}

pub const ATTN_Q: Def = Def {
    name: "attn.q",
    sees: &["q"],
    caps: &[Cap::Observe, Cap::PageMaskSink],
    position: None,

    sink: Some("attention.pages"),
};

pub const ATTN_OUT: Def = Def {
    name: "attn.out",
    sees: &["a"],
    caps: &[Cap::Observe, Cap::Scores],
    position: None,
    sink: None,
};

pub const ATTN_QV: Def = Def {
    name: "attn.qv",
    sees: &["q", "v"],
    caps: &[Cap::Transform],

    position: Some(Position {
        after: &["gemm.matmul", "gemm.matmul_acc", "matmul", "layout.split_qkv"],
        before: &[
            "norm.add_bias",
            "norm.rmsnorm",
            "norm.rmsnorm_no_scale",
            "rmsnorm",
            "rope",
            "rope.full",
            // THE SAME ADMITTED SET UNDER TWO SPELLINGS. `admits` matches a
            // routine's claim either whole or by its first dotted segment,
            // so the bare entry used to cover the core append and the
            // `kv_append.mla` / `.index` / `.pool` sub-families at once. The
            // core's claim is `attention.kv_append` now — under `attention`,
            // where the bare entry no longer reaches it — so it is spelled
            // whole, and the bare one stays for the three sub-families that
            // still read their role there. `attention` itself is NOT the
            // entry: it would admit the attention statement the seam is
            // required to sit before.
            "attention.kv_append",
            "kv_append",
        ],
    }),
    sink: None,
};

pub const RECURRENT: Def = Def {
    name: "recurrent",
    sees: &["mixed"],
    caps: &[Cap::Observe],
    position: None,
    sink: None,
};

pub const IN: Def = Def {
    name: "in",
    sees: &[],
    caps: &[Cap::Put, Cap::Emit],
    position: None,
    sink: None,
};

pub const OUT: Def = Def {
    name: "out",
    sees: &["logits"],
    caps: &[Cap::Observe, Cap::Sample, Cap::Put, Cap::Emit],
    position: None,
    sink: None,
};

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
                            producer_name(&plan.ops[from].kind),
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

pub const ALL: &[&Def] = &[&IN, &ATTN_QV, &ATTN_Q, &ATTN_OUT, &RECURRENT, &OUT];

pub fn by_name(name: &str) -> Option<&'static Def> {
    ALL.iter().copied().find(|d| d.name == name)
}

fn producer_name(kind: &crate::trace::OpKind) -> String {
    match kind {
        crate::trace::OpKind::Launch { kernel, .. } => kernel.clone(),
        other => kind_name(other).to_string(),
    }
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
