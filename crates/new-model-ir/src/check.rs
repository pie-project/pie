//! The trace-time validator (design §3). Everything monomorphization and the
//! old string tables used to guarantee is checked here, once, right after
//! every trace. Faults, never panics — these messages are the developer's
//! first error surface, so each reads like a sentence and names the node, the
//! op, and the value it is about. All faults are collected in one pass over
//! the plan; nothing stops at the first.

use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};

use crate::{Def, Dim, Dtype, Operands, Plan, StructKind, Ty, ValueId};

/// Where an out-of-range `ValueId` was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Site {
    Input { node: usize, op: &'static str },
    Output { node: usize, op: &'static str },
    Alias { node: usize, op: &'static str },
    MergeArm { merge: ValueId },
    Seam { seam: String },
}

/// A one-word summary of a `Def`, carried into faults instead of the def
/// itself so messages stay small and sentence-shaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefKind {
    Input,
    Weight(u32),
    Cache(u32),
    Op(u32),
    Merge,
}

/// One broken validator rule, with enough context to act on.
#[derive(Debug, Clone, PartialEq)]
pub enum Fault {
    /// A referenced id does not index `plan.values`.
    OutOfRange { site: Site, id: ValueId, len: usize },
    /// A node produces an id whose decl does not say `Def::Op(that node)`.
    ForeignOutput { node: usize, op: &'static str, id: ValueId, declared: DefKind },
    /// One id, two producers (or twice from the same node).
    DoubleOutput { id: ValueId, first: usize, first_op: &'static str, second: usize, second_op: &'static str },
    /// `Def::Op(i)` names a node that does not output this id.
    PhantomDef { id: ValueId, node: usize, op: &'static str },
    /// `Def::Op(i)` names a node index past the end of `plan.nodes`.
    DefNodeOutOfRange { id: ValueId, node: usize, len: usize },
    /// An input (directly, or through a merge arm) is defined at or after its
    /// consumer in program order.
    UseBeforeDef { node: usize, op: &'static str, input: ValueId, arm: Option<ValueId>, def_node: usize },
    /// `Def::Weight(k)` with `k` past the end of `plan.params`.
    WeightOutOfRange { id: ValueId, index: u32, len: usize },
    /// `Def::Cache(k)` with `k` past the end of `plan.caches`.
    CacheOutOfRange { id: ValueId, index: u32, len: usize },
    /// A weight's shape must be all-`Const`; this axis is symbolic.
    SymbolicWeight { id: ValueId, axis: usize, dim: Dim },
    /// A symbolic dim anywhere but axis 0.
    SymbolicAxis { id: ValueId, axis: usize, dim: Dim },
    /// An alias pair whose `out` is not among the node's outputs.
    AliasOutUnknown { node: usize, op: &'static str, out: ValueId },
    /// An alias pair whose `in` is not among the node's inputs.
    AliasInUnknown { node: usize, op: &'static str, input: ValueId },
    /// An in-place overwrite between two differently-typed values.
    AliasTyMismatch { node: usize, op: &'static str, out: ValueId, input: ValueId, out_ty: Ty, in_ty: Ty },
    /// A struct value defined by anything but an op.
    StructDef { id: ValueId, kind: StructKind, def: DefKind },
    /// A struct value used as a merge arm.
    StructArm { merge: ValueId, arm: ValueId },
    /// A merge with fewer than two arms.
    MergeTooFew { id: ValueId, arms: usize },
    /// A merge arm whose type differs from the merge's own.
    MergeArmTy { merge: ValueId, arm: ValueId, merge_ty: Ty, arm_ty: Ty },
}

/// Validate a traced plan against the §3 rules, collecting every fault.
pub fn check(plan: &Plan) -> Result<(), Vec<Fault>> {
    let mut faults = Vec::new();
    let len = plan.values.len();
    let in_range = |id: ValueId| (id.0 as usize) < len;

    // Program order: each node's ports, gathered once through `Operands`.
    // `owner` remembers who produced each id; `matched` marks ids whose decl
    // agrees with their producer, so the value sweep below can tell a
    // phantom `Def::Op` from a mere disagreement.
    let mut owner: Vec<Option<usize>> = vec![None; len];
    let mut matched = vec![false; len];
    let (mut ins, mut outs, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
    let mut seen = HashSet::new();

    for (j, node) in plan.nodes.iter().enumerate() {
        let op = node.op.name();
        ins.clear();
        outs.clear();
        pairs.clear();
        node.op.inputs(&mut ins);
        node.op.outputs(&mut outs);
        node.op.aliases(&mut pairs);

        for &id in &outs {
            if !in_range(id) {
                faults.push(Fault::OutOfRange { site: Site::Output { node: j, op }, id, len });
                continue;
            }
            match owner[id.0 as usize] {
                Some(first) => faults.push(Fault::DoubleOutput {
                    id, first, first_op: plan.nodes[first].op.name(), second: j, second_op: op,
                }),
                None => owner[id.0 as usize] = Some(j),
            }
            match &plan.values[id.0 as usize].def {
                Def::Op(i) if *i as usize == j => matched[id.0 as usize] = true,
                other => faults.push(Fault::ForeignOutput { node: j, op, id, declared: DefKind::of(other) }),
            }
        }

        for &id in &ins {
            if !in_range(id) {
                faults.push(Fault::OutOfRange { site: Site::Input { node: j, op }, id, len });
                continue;
            }
            // Use-after-def, chasing merge arms (a merge is data, so each arm
            // must itself be settled before the consumer fires).
            seen.clear();
            available(plan, id, id, j, op, &mut seen, &mut faults);
        }

        for &(out, input) in &pairs {
            for (id, side) in [(out, Site::Alias { node: j, op }), (input, Site::Alias { node: j, op })] {
                if !in_range(id) {
                    faults.push(Fault::OutOfRange { site: side, id, len });
                }
            }
            if in_range(out) && !outs.contains(&out) {
                faults.push(Fault::AliasOutUnknown { node: j, op, out });
            }
            if in_range(input) && !ins.contains(&input) {
                faults.push(Fault::AliasInUnknown { node: j, op, input });
            }
            if in_range(out) && in_range(input) {
                let (out_ty, in_ty) = (&plan.values[out.0 as usize].ty, &plan.values[input.0 as usize].ty);
                if out_ty != in_ty {
                    faults.push(Fault::AliasTyMismatch {
                        node: j, op, out, input, out_ty: out_ty.clone(), in_ty: in_ty.clone(),
                    });
                }
            }
        }
    }

    for (idx, decl) in plan.values.iter().enumerate() {
        let id = ValueId(idx as u32);
        let struct_kind = match &decl.ty {
            Ty::Struct(kind) => Some(*kind),
            Ty::Tensor { .. } => None,
        };
        // Struct values are defined only by plan-building ops (§6).
        if let (Some(kind), false) = (struct_kind, matches!(decl.def, Def::Op(_))) {
            faults.push(Fault::StructDef { id, kind, def: DefKind::of(&decl.def) });
        }
        match &decl.def {
            Def::Input(_) => {}
            Def::Weight(k) => {
                if *k as usize >= plan.params.len() {
                    faults.push(Fault::WeightOutOfRange { id, index: *k, len: plan.params.len() });
                }
                if let Ty::Tensor { shape, .. } = &decl.ty {
                    for (axis, &dim) in shape.iter().enumerate() {
                        if !matches!(dim, Dim::Const(_)) {
                            faults.push(Fault::SymbolicWeight { id, axis, dim });
                        }
                    }
                }
            }
            Def::Cache(k) => {
                if *k as usize >= plan.caches.len() {
                    faults.push(Fault::CacheOutOfRange { id, index: *k, len: plan.caches.len() });
                }
            }
            Def::Op(i) => {
                if *i as usize >= plan.nodes.len() {
                    faults.push(Fault::DefNodeOutOfRange { id, node: *i as usize, len: plan.nodes.len() });
                } else if !matched[idx] {
                    faults.push(Fault::PhantomDef { id, node: *i as usize, op: plan.nodes[*i as usize].op.name() });
                }
            }
            Def::Merge(arms) => {
                if arms.len() < 2 {
                    faults.push(Fault::MergeTooFew { id, arms: arms.len() });
                }
                for &(arm, _) in arms {
                    if !in_range(arm) {
                        faults.push(Fault::OutOfRange { site: Site::MergeArm { merge: id }, id: arm, len });
                        continue;
                    }
                    let arm_ty = &plan.values[arm.0 as usize].ty;
                    if struct_kind.is_none() && matches!(arm_ty, Ty::Struct(_)) {
                        faults.push(Fault::StructArm { merge: id, arm });
                    }
                    if arm_ty != &decl.ty {
                        faults.push(Fault::MergeArmTy {
                            merge: id, arm, merge_ty: decl.ty.clone(), arm_ty: arm_ty.clone(),
                        });
                    }
                }
            }
        }
        // Symbolic dims live only at axis 0 (weights answer to the stricter
        // all-`Const` rule above; don't fault them twice).
        if !matches!(decl.def, Def::Weight(_)) {
            if let Ty::Tensor { shape, .. } = &decl.ty {
                for (axis, &dim) in shape.iter().enumerate().skip(1) {
                    if !matches!(dim, Dim::Const(_)) {
                        faults.push(Fault::SymbolicAxis { id, axis, dim });
                    }
                }
            }
        }
    }

    for seam in &plan.seams {
        for &id in &seam.values {
            if !in_range(id) {
                faults.push(Fault::OutOfRange { site: Site::Seam { seam: seam.seam.clone() }, id, len });
            }
        }
    }

    if faults.is_empty() { Ok(()) } else { Err(faults) }
}

/// `check`, keeping the plan on success — for the tail of a trace pipeline.
pub fn checked(plan: Plan) -> Result<Plan, Vec<Fault>> {
    check(&plan)?;
    Ok(plan)
}

/// Is `id` settled before `plan.nodes[node]` fires? Direct op outputs must
/// come from an earlier node; merges are chased arm by arm (`root` keeps the
/// operand actually consumed for the message; `seen` breaks merge cycles).
fn available(
    plan: &Plan, root: ValueId, id: ValueId, node: usize, op: &'static str,
    seen: &mut HashSet<u32>, faults: &mut Vec<Fault>,
) {
    match &plan.values[id.0 as usize].def {
        Def::Op(i) if (*i as usize) < plan.nodes.len() && *i as usize >= node => {
            faults.push(Fault::UseBeforeDef {
                node, op, input: root, arm: (id != root).then_some(id), def_node: *i as usize,
            });
        }
        Def::Merge(arms) => {
            for &(arm, _) in arms {
                if (arm.0 as usize) < plan.values.len() && seen.insert(arm.0) {
                    available(plan, root, arm, node, op, seen, faults);
                }
            }
        }
        _ => {} // Input / Weight / Cache: always bound before the first node.
    }
}

impl DefKind {
    fn of(def: &Def) -> Self {
        match def {
            Def::Input(_) => DefKind::Input,
            Def::Weight(k) => DefKind::Weight(*k),
            Def::Cache(k) => DefKind::Cache(*k),
            Def::Op(i) => DefKind::Op(*i),
            Def::Merge(_) => DefKind::Merge,
        }
    }
}

/// "v42" — the spelling every message uses for a value.
struct V(ValueId);

impl Display for V {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0.0)
    }
}

/// "bf16[tokens, 4096]" / "struct AttnDecodePlan" — a type, said briefly.
struct T<'a>(&'a Ty);

impl Display for T<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            Ty::Tensor { shape, dtype } => {
                let name = match dtype {
                    Dtype::Bf16 => "bf16", Dtype::F16 => "f16", Dtype::F32 => "f32",
                    Dtype::I32 => "i32", Dtype::U32 => "u32", Dtype::U8 => "u8",
                };
                write!(f, "{name}[")?;
                for (i, dim) in shape.iter().enumerate() {
                    if i > 0 { f.write_str(", ")?; }
                    write!(f, "{}", D(*dim))?;
                }
                f.write_str("]")
            }
            Ty::Struct(kind) => write!(f, "struct {kind:?}"),
        }
    }
}

struct D(Dim);

impl Display for D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            Dim::Const(n) => write!(f, "{n}"),
            Dim::Tokens => f.write_str("tokens"),
            Dim::TokensTimes(k) => write!(f, "tokens*{k}"),
            Dim::Lanes => f.write_str("lanes"),
            Dim::LanesPlus(k) => write!(f, "lanes+{k}"),
        }
    }
}

impl Display for Site {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Site::Input { node, op } => write!(f, "node {node} ({op}): input"),
            Site::Output { node, op } => write!(f, "node {node} ({op}): output"),
            Site::Alias { node, op } => write!(f, "node {node} ({op}): alias"),
            Site::MergeArm { merge } => write!(f, "merge {}: arm", V(*merge)),
            Site::Seam { seam } => write!(f, "seam \"{seam}\": value"),
        }
    }
}

impl Display for DefKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DefKind::Input => f.write_str("a runtime input"),
            DefKind::Weight(k) => write!(f, "weight {k}"),
            DefKind::Cache(k) => write!(f, "cache {k}"),
            DefKind::Op(i) => write!(f, "the output of node {i}"),
            DefKind::Merge => f.write_str("a merge"),
        }
    }
}

impl Display for Fault {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Fault::OutOfRange { site, id, len } => {
                write!(f, "{site} {} is out of range — the plan declares {len} values", V(*id))
            }
            Fault::ForeignOutput { node, op, id, declared } => {
                write!(f, "node {node} ({op}): output {} is declared as {declared}, not as this node's output", V(*id))
            }
            Fault::DoubleOutput { id, first, first_op, second, second_op: _ } if first == second => {
                write!(f, "node {first} ({first_op}): {} appears twice among its outputs — one id, one definition", V(*id))
            }
            Fault::DoubleOutput { id, first, first_op, second, second_op } => {
                write!(f, "{} is output by both node {first} ({first_op}) and node {second} ({second_op}) — a value has exactly one definition", V(*id))
            }
            Fault::PhantomDef { id, node, op } => {
                write!(f, "{} is declared as the output of node {node} ({op}), but that node does not produce it — was `#[out]` forgotten?", V(*id))
            }
            Fault::DefNodeOutOfRange { id, node, len } => {
                write!(f, "{} is declared as the output of node {node}, but the plan has {len} nodes", V(*id))
            }
            Fault::UseBeforeDef { node, op, input, arm, def_node } => {
                match arm {
                    Some(a) => write!(f, "node {node} ({op}): input {} reaches merge arm {}, defined by ", V(*input), V(*a))?,
                    None => write!(f, "node {node} ({op}): input {} is defined by ", V(*input))?,
                }
                if def_node == node {
                    f.write_str("this very node")
                } else {
                    write!(f, "node {def_node}, later in program order")
                }
            }
            Fault::WeightOutOfRange { id, index, len } => {
                write!(f, "{} names weight {index}, but the plan declares {len} params", V(*id))
            }
            Fault::CacheOutOfRange { id, index, len } => {
                write!(f, "{} names cache {index}, but the plan declares {len} caches", V(*id))
            }
            Fault::SymbolicWeight { id, axis, dim } => {
                write!(f, "weight {}: axis {axis} is {} — a weight's shape is all-const", V(*id), D(*dim))
            }
            Fault::SymbolicAxis { id, axis, dim } => {
                write!(f, "{}: axis {axis} is {} — symbolic dims live only at axis 0", V(*id), D(*dim))
            }
            Fault::AliasOutUnknown { node, op, out } => {
                write!(f, "node {node} ({op}): alias names {} as an output, but the node does not produce it", V(*out))
            }
            Fault::AliasInUnknown { node, op, input } => {
                write!(f, "node {node} ({op}): alias names {} as an input, but the node does not consume it", V(*input))
            }
            Fault::AliasTyMismatch { node, op, out, input, out_ty, in_ty } => {
                write!(f, "node {node} ({op}): {} overwrites {} in place, but {} is not {}", V(*out), V(*input), T(out_ty), T(in_ty))
            }
            Fault::StructDef { id, kind, def } => {
                write!(f, "{} is a struct ({kind:?}) defined as {def} — struct values come only from plan-building ops", V(*id))
            }
            Fault::StructArm { merge, arm } => {
                write!(f, "merge {}: arm {} is struct-typed — a struct value never passes through a merge", V(*merge), V(*arm))
            }
            Fault::MergeTooFew { id, arms } => {
                write!(f, "merge {} has {arms} arm(s); a merge needs at least two", V(*id))
            }
            Fault::MergeArmTy { merge, arm, merge_ty, arm_ty } => {
                write!(f, "merge {}: arm {} is {}, but the merge is {}", V(*merge), V(*arm), T(arm_ty), T(merge_ty))
            }
        }
    }
}

impl std::error::Error for Fault {}
