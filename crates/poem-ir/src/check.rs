pub mod classes;

pub use classes::{fact_width, resolve_classes};

use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};

use crate::ops::Elemwise;
use crate::{
    Def, Dim, Dtype, InputAxis, Operands, Operation, RuntimeInput, StructKind, Trace, Ty, ValueId,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Site {
    Input { node: usize, op: &'static str },
    Output { node: usize, op: &'static str },
    Alias { node: usize, op: &'static str },
    MergeArm { merge: ValueId },
    Named { output: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefKind {
    Input,
    Weight(u32),
    Cache(u32),
    Op(u32),
    Merge,
}

/// The operand a type check names: the field the op declares it on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Port(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    Struct(&'static [StructKind]),
    Cache,
    Tensor(Dtype),
}

impl Expect {
    pub const I32: Self = Self::Tensor(Dtype::I32);
    pub const F32: Self = Self::Tensor(Dtype::F32);
    pub const U8: Self = Self::Tensor(Dtype::U8);
    pub const CACHE: Self = Self::Cache;
    pub const DECODE_PLAN: Self = Self::Struct(&[StructKind::AttnDecodePlan]);
    pub const PREFILL_PLAN: Self = Self::Struct(&[StructKind::AttnPrefillPlan]);
    pub const MLA_PLAN: Self = Self::Struct(&[StructKind::MlaPlan]);
}

#[derive(Debug, Clone, PartialEq)]
pub enum Fault {
    OutOfRange {
        site: Site,
        id: ValueId,
        len: usize,
    },
    ForeignOutput {
        node: usize,
        op: &'static str,
        id: ValueId,
        declared: DefKind,
    },
    DoubleOutput {
        id: ValueId,
        first: usize,
        first_op: &'static str,
        second: usize,
        second_op: &'static str,
    },
    PhantomDef {
        id: ValueId,
        node: usize,
        op: &'static str,
    },
    DefNodeOutOfRange {
        id: ValueId,
        node: usize,
        len: usize,
    },
    UseBeforeDef {
        node: usize,
        op: &'static str,
        input: ValueId,
        arm: Option<ValueId>,
        def_node: usize,
    },
    WeightOutOfRange {
        id: ValueId,
        index: u32,
        len: usize,
    },
    CacheOutOfRange {
        id: ValueId,
        index: u32,
        len: usize,
    },
    GuestInput {
        id: ValueId,
        why: String,
    },
    UndeclaredInput {
        name: String,
    },
    SymbolicWeight {
        id: ValueId,
        axis: usize,
        dim: Dim,
    },
    SymbolicAxis {
        id: ValueId,
        axis: usize,
        dim: Dim,
    },
    AliasOutUnknown {
        node: usize,
        op: &'static str,
        out: ValueId,
    },
    AliasInUnknown {
        node: usize,
        op: &'static str,
        input: ValueId,
    },
    AliasTyMismatch {
        node: usize,
        op: &'static str,
        out: ValueId,
        input: ValueId,
        out_ty: Ty,
        in_ty: Ty,
    },
    FoldThenRead {
        fold: usize,
        fold_op: &'static str,
        input: ValueId,
        node: usize,
        op: &'static str,
        arm: Option<ValueId>,
    },
    StructDef {
        id: ValueId,
        kind: StructKind,
        def: DefKind,
    },
    StructArm {
        merge: ValueId,
        arm: ValueId,
    },
    MergeTooFew {
        id: ValueId,
        arms: usize,
    },
    MergeArmTy {
        merge: ValueId,
        arm: ValueId,
        merge_ty: Ty,
        arm_ty: Ty,
    },
    PortKind {
        node: usize,
        op: &'static str,
        port: Port,
        id: ValueId,
        want: Expect,
        ty: Ty,
        def: DefKind,
    },
    PortDtype {
        node: usize,
        op: &'static str,
        port: Port,
        id: ValueId,
        want: Dtype,
        ty: Ty,
    },
    HadamardBlock {
        node: usize,
        op: &'static str,
        id: ValueId,
        block: u32,
        last: Option<Dim>,
        why: HadamardWhy,
    },
    HadamardSignDtype {
        node: usize,
        op: &'static str,
        x: ValueId,
        signs: ValueId,
        act: Dtype,
        sign: Dtype,
    },
}

fn guest_input(
    trace: &Trace,
    at: usize,
    input: u32,
    axis: InputAxis,
    width: u32,
    dtype: Dtype,
    ty: &Ty,
) -> Option<String> {
    let Some(named) = trace.inputs.get(input as usize) else {
        return Some(format!(
            "reads guest input {input}, and the trace names {}",
            trace.inputs.len()
        ));
    };
    let name = &named.name;
    if !matches!(dtype, Dtype::F32 | Dtype::Bf16 | Dtype::I32) {
        return Some(format!(
            "declares input `{name}` as {dtype:?}, not f32, bf16 or i32"
        ));
    }
    let want = Ty::Tensor {
        shape: vec![axis.dim(), Dim::Const(u64::from(width))],
        dtype,
    };
    if *ty != want {
        return Some(format!(
            "declares input `{name}` as {ty:?}, which is not {} by {width} {dtype:?}",
            axis.name()
        ));
    }
    let again = trace.values.iter().enumerate().any(|(other, decl)| {
        other != at && matches!(decl.def, Def::Input(RuntimeInput::Guest { id, .. }) if id == input)
    });
    again.then(|| format!("declares input `{name}`, which another value declares too"))
}

/// The smallest and largest Hadamard block the Metal butterfly instantiates.
pub const HADAMARD_MIN: u32 = 64;
pub const HADAMARD_MAX: u32 = 1024;

/// Which Hadamard invariant a [`Fault::HadamardBlock`] flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HadamardWhy {
    /// The block size is not a power of two in the supported span.
    Block,
    /// The activation's last dim is not a whole number of blocks.
    Width,
    /// The sign diagonal's last dim is not a whole number of blocks.
    Signs,
}

pub fn check(trace: &Trace) -> Result<(), Vec<Fault>> {
    let mut faults = Vec::new();
    let len = trace.values.len();
    let in_range = |id: ValueId| (id.0 as usize) < len;

    let mut owner: Vec<Option<usize>> = vec![None; len];
    let mut matched = vec![false; len];
    let mut folded: Vec<Option<(usize, &'static str)>> = vec![None; len];
    let (mut ins, mut outs, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
    let mut named = Vec::new();
    let mut seen = HashSet::new();

    for (j, node) in trace.nodes.iter().enumerate() {
        let op = node.op.name();
        ins.clear();
        outs.clear();
        pairs.clear();
        node.op.inputs(&mut ins);
        node.op.outputs(&mut outs);
        node.op.aliases(&mut pairs);

        for &id in &outs {
            if !in_range(id) {
                faults.push(Fault::OutOfRange {
                    site: Site::Output { node: j, op },
                    id,
                    len,
                });
                continue;
            }
            match owner[id.0 as usize] {
                Some(first) => faults.push(Fault::DoubleOutput {
                    id,
                    first,
                    first_op: trace.nodes[first].op.name(),
                    second: j,
                    second_op: op,
                }),
                None => owner[id.0 as usize] = Some(j),
            }
            match &trace.values[id.0 as usize].def {
                Def::Op(i) if *i as usize == j => matched[id.0 as usize] = true,
                other => faults.push(Fault::ForeignOutput {
                    node: j,
                    op,
                    id,
                    declared: DefKind::of(other),
                }),
            }
        }

        for &id in &ins {
            if !in_range(id) {
                faults.push(Fault::OutOfRange {
                    site: Site::Input { node: j, op },
                    id,
                    len,
                });
                continue;
            }
            seen.clear();
            available(trace, id, id, j, op, &mut seen, &mut faults);
        }

        for &(out, input) in &pairs {
            for (id, side) in [
                (out, Site::Alias { node: j, op }),
                (input, Site::Alias { node: j, op }),
            ] {
                if !in_range(id) {
                    faults.push(Fault::OutOfRange {
                        site: side,
                        id,
                        len,
                    });
                }
            }
            if in_range(out) && !outs.contains(&out) {
                faults.push(Fault::AliasOutUnknown { node: j, op, out });
            }
            if in_range(input) && !ins.contains(&input) {
                faults.push(Fault::AliasInUnknown { node: j, op, input });
            }
            if in_range(out) && in_range(input) {
                let (out_ty, in_ty) = (
                    &trace.values[out.0 as usize].ty,
                    &trace.values[input.0 as usize].ty,
                );
                if out_ty != in_ty {
                    faults.push(Fault::AliasTyMismatch {
                        node: j,
                        op,
                        out,
                        input,
                        out_ty: out_ty.clone(),
                        in_ty: in_ty.clone(),
                    });
                }
                folded[input.0 as usize].get_or_insert((j, op));
            }
        }

        named.clear();
        node.op.ports(&mut named);
        let ports = named.iter().map(|&(name, id, want)| (Port(name), id, want));
        for (port, id, want) in ports {
            if !in_range(id) {
                continue;
            }
            let decl = &trace.values[id.0 as usize];
            let wrong_kind = match want {
                Expect::Struct(kinds) => !matches!(&decl.ty, Ty::Struct(k) if kinds.contains(k)),
                Expect::Cache => !matches!(decl.def, Def::Cache(_)),
                Expect::Tensor(dtype) => {
                    if !matches!(&decl.ty, Ty::Tensor { dtype: d, .. } if *d == dtype) {
                        faults.push(Fault::PortDtype {
                            node: j,
                            op,
                            port,
                            id,
                            want: dtype,
                            ty: decl.ty.clone(),
                        });
                    }
                    false
                }
            };
            if wrong_kind {
                faults.push(Fault::PortKind {
                    node: j,
                    op,
                    port,
                    id,
                    want,
                    ty: decl.ty.clone(),
                    def: DefKind::of(&decl.def),
                });
            }
        }

        if let Operation::Elemwise(Elemwise::Hadamard {
            x, block, signs, ..
        }) = &node.op
        {
            let block = *block;
            // The block must be a power of two in the kernel's supported span.
            if !block.is_power_of_two() || !(HADAMARD_MIN..=HADAMARD_MAX).contains(&block) {
                faults.push(Fault::HadamardBlock {
                    node: j,
                    op,
                    id: *x,
                    block,
                    last: None,
                    why: HadamardWhy::Block,
                });
            } else {
                // The transformed row's last dim must tile into whole blocks.
                if in_range(*x)
                    && let Ty::Tensor { shape, .. } = &trace.values[x.0 as usize].ty
                {
                    let last = shape.last().copied();
                    if !matches!(last, Some(Dim::Const(d)) if d % u64::from(block) == 0) {
                        faults.push(Fault::HadamardBlock {
                            node: j,
                            op,
                            id: *x,
                            block,
                            last,
                            why: HadamardWhy::Width,
                        });
                    }
                }
                // A present sign diagonal repeats block-wise, so its own last
                // dim must itself be a whole number of blocks.
                if let Some(s) = signs
                    && in_range(*s)
                    && let Ty::Tensor { shape, dtype: sign } = &trace.values[s.0 as usize].ty
                {
                    let last = shape.last().copied();
                    if !matches!(last, Some(Dim::Const(d)) if d % u64::from(block) == 0) {
                        faults.push(Fault::HadamardBlock {
                            node: j,
                            op,
                            id: *s,
                            block,
                            last,
                            why: HadamardWhy::Signs,
                        });
                    }
                    // The signs multiply the activation on load, so the kernel
                    // reads both through one element; mismatched dtypes are
                    // rejected here rather than at launch.
                    if in_range(*x)
                        && let Ty::Tensor { dtype: act, .. } = &trace.values[x.0 as usize].ty
                        && act != sign
                    {
                        faults.push(Fault::HadamardSignDtype {
                            node: j,
                            op,
                            x: *x,
                            signs: *s,
                            act: *act,
                            sign: *sign,
                        });
                    }
                }
            }
        }
    }

    if folded.iter().any(Option::is_some) {
        for (k, node) in trace.nodes.iter().enumerate() {
            ins.clear();
            node.op.inputs(&mut ins);
            for &id in &ins {
                if !in_range(id) {
                    continue;
                }
                seen.clear();
                intact(trace, &folded, id, id, k, &mut seen, &mut faults);
            }
        }
    }

    for (input, named) in trace.inputs.iter().enumerate() {
        let declared = trace.values.iter().any(|decl| {
            matches!(decl.def, Def::Input(RuntimeInput::Guest { id, .. }) if id as usize == input)
        });
        if !declared {
            faults.push(Fault::UndeclaredInput {
                name: named.name.clone(),
            });
        }
    }

    for (idx, decl) in trace.values.iter().enumerate() {
        let id = ValueId(idx as u32);
        let struct_kind = match &decl.ty {
            Ty::Struct(kind) => Some(*kind),
            Ty::Tensor { .. } => None,
        };
        if let (Some(kind), false) = (struct_kind, matches!(decl.def, Def::Op(_))) {
            faults.push(Fault::StructDef {
                id,
                kind,
                def: DefKind::of(&decl.def),
            });
        }
        match &decl.def {
            Def::Input(RuntimeInput::Guest {
                id: input,
                axis,
                width,
                dtype,
            }) => {
                let why = guest_input(trace, idx, *input, *axis, *width, *dtype, &decl.ty);
                if let Some(why) = why {
                    faults.push(Fault::GuestInput { id, why });
                }
            }
            Def::Input(_) => {}
            Def::Weight(k) => {
                if *k as usize >= trace.params.len() {
                    faults.push(Fault::WeightOutOfRange {
                        id,
                        index: *k,
                        len: trace.params.len(),
                    });
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
                if *k as usize >= trace.caches.len() {
                    faults.push(Fault::CacheOutOfRange {
                        id,
                        index: *k,
                        len: trace.caches.len(),
                    });
                }
            }
            Def::Op(i) => {
                if *i as usize >= trace.nodes.len() {
                    faults.push(Fault::DefNodeOutOfRange {
                        id,
                        node: *i as usize,
                        len: trace.nodes.len(),
                    });
                } else if !matched[idx] {
                    faults.push(Fault::PhantomDef {
                        id,
                        node: *i as usize,
                        op: trace.nodes[*i as usize].op.name(),
                    });
                }
            }
            Def::Merge(arms) => {
                if arms.len() < 2 {
                    faults.push(Fault::MergeTooFew {
                        id,
                        arms: arms.len(),
                    });
                }
                for &(arm, _) in arms {
                    if !in_range(arm) {
                        faults.push(Fault::OutOfRange {
                            site: Site::MergeArm { merge: id },
                            id: arm,
                            len,
                        });
                        continue;
                    }
                    let arm_ty = &trace.values[arm.0 as usize].ty;
                    if struct_kind.is_none() && matches!(arm_ty, Ty::Struct(_)) {
                        faults.push(Fault::StructArm { merge: id, arm });
                    }
                    if arm_ty != &decl.ty {
                        faults.push(Fault::MergeArmTy {
                            merge: id,
                            arm,
                            merge_ty: decl.ty.clone(),
                            arm_ty: arm_ty.clone(),
                        });
                    }
                }
            }
        }
        if !matches!(decl.def, Def::Weight(_))
            && let Ty::Tensor { shape, .. } = &decl.ty
        {
            for (axis, &dim) in shape.iter().enumerate().skip(1) {
                if !matches!(dim, Dim::Const(_)) {
                    faults.push(Fault::SymbolicAxis { id, axis, dim });
                }
            }
        }
    }

    for output in &trace.outputs {
        for id in output.values() {
            if !in_range(id) {
                faults.push(Fault::OutOfRange {
                    site: Site::Named {
                        output: output.name.clone(),
                    },
                    id,
                    len,
                });
            }
        }
    }

    if faults.is_empty() {
        Ok(())
    } else {
        Err(faults)
    }
}

pub fn checked(trace: Trace) -> Result<Trace, Vec<Fault>> {
    check(&trace)?;
    Ok(trace)
}

fn available(
    trace: &Trace,
    root: ValueId,
    id: ValueId,
    node: usize,
    op: &'static str,
    seen: &mut HashSet<u32>,
    faults: &mut Vec<Fault>,
) {
    match &trace.values[id.0 as usize].def {
        Def::Op(i) if (*i as usize) < trace.nodes.len() && *i as usize >= node => {
            faults.push(Fault::UseBeforeDef {
                node,
                op,
                input: root,
                arm: (id != root).then_some(id),
                def_node: *i as usize,
            });
        }
        Def::Merge(arms) => {
            for &(arm, _) in arms {
                if (arm.0 as usize) < trace.values.len() && seen.insert(arm.0) {
                    available(trace, root, arm, node, op, seen, faults);
                }
            }
        }
        _ => {}
    }
}

fn intact(
    trace: &Trace,
    folded: &[Option<(usize, &'static str)>],
    root: ValueId,
    id: ValueId,
    at: usize,
    seen: &mut HashSet<u32>,
    faults: &mut Vec<Fault>,
) {
    let by = &trace.nodes[at];
    if let Some((fold, fold_op)) = folded[id.0 as usize]
        && fold < at
        && by.guard.implies(&trace.nodes[fold].guard)
    {
        faults.push(Fault::FoldThenRead {
            fold,
            fold_op,
            input: id,
            node: at,
            op: by.op.name(),
            arm: (id != root).then_some(root),
        });
    }
    if let Def::Merge(arms) = &trace.values[id.0 as usize].def {
        for &(arm, _) in arms {
            if (arm.0 as usize) < trace.values.len() && seen.insert(arm.0) {
                intact(trace, folded, root, arm, at, seen, faults);
            }
        }
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

pub(crate) struct V(pub(crate) ValueId);

impl Display for V {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0.0)
    }
}

struct N(Dtype);

impl Display for N {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.0.spelling())
    }
}

struct T<'a>(&'a Ty);

impl Display for T<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            Ty::Tensor { shape, dtype } => {
                write!(f, "{}[", N(*dtype))?;
                for (i, dim) in shape.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
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
            Dim::Readouts => f.write_str("readouts"),
            Dim::Patches => f.write_str("patches"),
            Dim::Images => f.write_str("images"),
            Dim::ImagesPlus(k) => write!(f, "images+{k}"),
            Dim::Voxels => f.write_str("voxels"),
            Dim::VoxelsTimes(k) => write!(f, "voxels*{k}"),
            Dim::Clips => f.write_str("clips"),
            Dim::ClipsPlus(k) => write!(f, "clips+{k}"),
        }
    }
}

impl Display for Port {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`", self.0)
    }
}

impl Display for Site {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Site::Input { node, op } => write!(f, "node {node} ({op}): input"),
            Site::Output { node, op } => write!(f, "node {node} ({op}): output"),
            Site::Alias { node, op } => write!(f, "node {node} ({op}): alias"),
            Site::MergeArm { merge } => write!(f, "merge {}: arm", V(*merge)),
            Site::Named { output } => write!(f, "output \"{output}\": value"),
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
                write!(
                    f,
                    "{site} {} is out of range — the plan declares {len} values",
                    V(*id)
                )
            }
            Fault::ForeignOutput {
                node,
                op,
                id,
                declared,
            } => {
                write!(
                    f,
                    "node {node} ({op}): output {} is declared as {declared}, not as this node's output",
                    V(*id)
                )
            }
            Fault::DoubleOutput {
                id,
                first,
                first_op,
                second,
                second_op: _,
            } if first == second => {
                write!(
                    f,
                    "node {first} ({first_op}): {} appears twice among its outputs — one id, one definition",
                    V(*id)
                )
            }
            Fault::DoubleOutput {
                id,
                first,
                first_op,
                second,
                second_op,
            } => {
                write!(
                    f,
                    "{} is output by both node {first} ({first_op}) and node {second} ({second_op}) — a value has exactly one definition",
                    V(*id)
                )
            }
            Fault::PhantomDef { id, node, op } => {
                write!(
                    f,
                    "{} is declared as the output of node {node} ({op}), but that node does not produce it — was `Out` forgotten?",
                    V(*id)
                )
            }
            Fault::DefNodeOutOfRange { id, node, len } => {
                write!(
                    f,
                    "{} is declared as the output of node {node}, but the plan has {len} nodes",
                    V(*id)
                )
            }
            Fault::UseBeforeDef {
                node,
                op,
                input,
                arm,
                def_node,
            } => {
                match arm {
                    Some(a) => write!(
                        f,
                        "node {node} ({op}): input {} reaches merge arm {}, defined by ",
                        V(*input),
                        V(*a)
                    )?,
                    None => write!(f, "node {node} ({op}): input {} is defined by ", V(*input))?,
                }
                if def_node == node {
                    f.write_str("this very node")
                } else {
                    write!(f, "node {def_node}, later in program order")
                }
            }
            Fault::WeightOutOfRange { id, index, len } => {
                write!(
                    f,
                    "{} names weight {index}, but the plan declares {len} params",
                    V(*id)
                )
            }
            Fault::GuestInput { id, why } => write!(f, "{} {why}", V(*id)),
            Fault::UndeclaredInput { name } => {
                write!(f, "the trace names input `{name}`, which no value declares")
            }
            Fault::CacheOutOfRange { id, index, len } => {
                write!(
                    f,
                    "{} names cache {index}, but the plan declares {len} caches",
                    V(*id)
                )
            }
            Fault::SymbolicWeight { id, axis, dim } => {
                write!(
                    f,
                    "weight {}: axis {axis} is {} — a weight's shape is all-const",
                    V(*id),
                    D(*dim)
                )
            }
            Fault::SymbolicAxis { id, axis, dim } => {
                write!(
                    f,
                    "{}: axis {axis} is {} — symbolic dims live only at axis 0",
                    V(*id),
                    D(*dim)
                )
            }
            Fault::AliasOutUnknown { node, op, out } => {
                write!(
                    f,
                    "node {node} ({op}): alias names {} as an output, but the node does not produce it",
                    V(*out)
                )
            }
            Fault::AliasInUnknown { node, op, input } => {
                write!(
                    f,
                    "node {node} ({op}): alias names {} as an input, but the node does not consume it",
                    V(*input)
                )
            }
            Fault::AliasTyMismatch {
                node,
                op,
                out,
                input,
                out_ty,
                in_ty,
            } => {
                write!(
                    f,
                    "node {node} ({op}): {} overwrites {} in place, but {} is not {}",
                    V(*out),
                    V(*input),
                    T(out_ty),
                    T(in_ty)
                )
            }
            Fault::FoldThenRead {
                fold,
                fold_op,
                input,
                node,
                op,
                arm,
            } => {
                write!(
                    f,
                    "node {fold} ({fold_op}) overwrites {} in place, and node {node} ({op}) reads it afterwards",
                    V(*input)
                )?;
                if let Some(a) = arm {
                    write!(f, " through merge {}", V(*a))?;
                }
                f.write_str(" — an in-place fold is the last read of its operand; fold onto a copy")
            }
            Fault::StructDef { id, kind, def } => {
                write!(
                    f,
                    "{} is a struct ({kind:?}) defined as {def} — struct values come only from plan-building ops",
                    V(*id)
                )
            }
            Fault::StructArm { merge, arm } => {
                write!(
                    f,
                    "merge {}: arm {} is struct-typed — a struct value never passes through a merge",
                    V(*merge),
                    V(*arm)
                )
            }
            Fault::MergeTooFew { id, arms } => {
                write!(
                    f,
                    "merge {} has {arms} arm(s); a merge needs at least two",
                    V(*id)
                )
            }
            Fault::MergeArmTy {
                merge,
                arm,
                merge_ty,
                arm_ty,
            } => {
                write!(
                    f,
                    "merge {}: arm {} is {}, but the merge is {}",
                    V(*merge),
                    V(*arm),
                    T(arm_ty),
                    T(merge_ty)
                )
            }
            Fault::PortKind {
                node,
                op,
                port,
                id,
                want,
                ty,
                def,
            } => {
                write!(f, "node {node} ({op}): {port} {} ", V(*id))?;
                match want {
                    Expect::Struct(kinds) => {
                        f.write_str("must be a ")?;
                        for (i, kind) in kinds.iter().enumerate() {
                            if i > 0 {
                                f.write_str(" or ")?;
                            }
                            write!(f, "struct {kind:?}")?;
                        }
                        write!(f, ", but it is {}", T(ty))
                    }
                    Expect::Cache => write!(f, "must name cache storage, but it is {def}"),
                    Expect::Tensor(dtype) => {
                        write!(f, "must be a {} tensor, but it is {}", N(*dtype), T(ty))
                    }
                }
            }
            Fault::PortDtype {
                node,
                op,
                port,
                id,
                want,
                ty,
            } => {
                write!(
                    f,
                    "node {node} ({op}): {port} {} is pinned to {}, but it is {}",
                    V(*id),
                    N(*want),
                    T(ty)
                )
            }
            Fault::HadamardBlock {
                node,
                op,
                id,
                block,
                last,
                why,
            } => match why {
                HadamardWhy::Block => write!(
                    f,
                    "node {node} ({op}): a Hadamard block of {block} is not a power of two in \
                     [{HADAMARD_MIN}, {HADAMARD_MAX}]; the butterfly turns 64, 128, 256, 512 \
                     and 1024-vectors",
                ),
                HadamardWhy::Width => write!(
                    f,
                    "node {node} ({op}): {} has last dim {last:?}, but a Hadamard block of \
                     {block} needs a last dim that is a multiple of {block}",
                    V(*id)
                ),
                HadamardWhy::Signs => write!(
                    f,
                    "node {node} ({op}): the sign diagonal {} has last dim {last:?}, but it \
                     repeats block-wise and so needs a last dim that is a multiple of {block}",
                    V(*id)
                ),
            },
            Fault::HadamardSignDtype {
                node,
                op,
                x,
                signs,
                act,
                sign,
            } => {
                write!(
                    f,
                    "node {node} ({op}): the sign diagonal {} is {}, but the activation {} it \
                     rides is {}; the signs multiply on the activation's element",
                    V(*signs),
                    N(*sign),
                    V(*x),
                    N(*act)
                )
            }
        }
    }
}

impl std::error::Error for Fault {}
