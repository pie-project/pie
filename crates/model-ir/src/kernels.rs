use crate::trace::{ForwardPlan, Op, OpKind};

pub use kernels::{Cap, KernelSig};

pub use kernels::{Kind, Source};
pub use kernels_cuda::sigs;

pub use kernels_metal::entrypoints as metal_entrypoints;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Backend {
    Cuda,

    Metal,
}

impl Backend {
    pub fn of_family(family: &str) -> Option<Backend> {
        let mut parts = family.split('.').skip(1);
        match parts.next() {
            Some("cuda") => Some(Backend::Cuda),
            Some("metal") => Some(Backend::Metal),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Stated {
    pub whole: bool,

    pub depth_prefix_plan: bool,

    pub args: &'static [kernels::Ty],

    pub sources: &'static [Option<kernels::Source>],

    pub derived: &'static [kernels::Derived],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pointers {
    pub reads: usize,

    pub optional_reads: usize,

    pub writes: usize,

    pub optional_writes: usize,
}

impl Stated {
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
    #[must_use]
    pub fn places(&self) -> (usize, usize) {
        let p = self.pointers();
        (p.reads, p.writes)
    }

    #[must_use]
    pub fn pointers(&self) -> Pointers {
        use kernels::{Kind, Source};

        let mut p = Pointers::default();
        for (at, source) in self.sources.iter().enumerate() {
            let optional = self.derived.get(at).is_some_and(|d| d.nullable);
            match source {
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

                Some(Source::Alias(..)) => p.writes += 1,
                _ => {}
            }
        }
        p
    }

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

/// Every family a cuda `#[claims]` block answers, one line per migrated
/// family. A family's line is its `*_CLAIMS` slice and nothing else, so
/// migrating one is adding one line here; the macro grows an aggregate when
/// there are enough of them to want one.
const CUDA_CLAIMS: &[&[&str]] = &[
    kernels_cuda::norm::NORM_CLAIMS,
    kernels_cuda::rope::ROPE_CLAIMS,
    kernels_cuda::mlp::MLP_CLAIMS,
    kernels_cuda::gemm::GEMM_CLAIMS,
    kernels_cuda::dist::DIST_CLAIMS,
    kernels_cuda::moe::MOE_CLAIMS,
    kernels_cuda::layout::LAYOUT_CLAIMS,
    // `GATE_CLAIMS` reads from `mlp` rather than a module of its own: the
    // impl lives beside the other gate kernel firing the same C++ namespace.
    kernels_cuda::mlp::GATE_CLAIMS,
    kernels_cuda::ssm::SSM_CLAIMS,
    // `ATTENTION_CLAIMS` reads from `attn` and not `attention`: the impl
    // lives in the module the fa2 core is filed under, which is where the
    // four delegations' neighbours already are.
    kernels_cuda::attn::ATTENTION_CLAIMS,
    // The three latent/paged families answer from `attn`, where their
    // delegates live — `mla`'s two absorbs reach across into `gemm::absorb`
    // from there, since a family is one impl block and its points may fire
    // out of two modules. `POOL_CLAIMS` is EMPTY and still listed: cuda
    // implements the family and overrides nothing, which is a measurement
    // and not an omission.
    kernels_cuda::attn::MLA_CLAIMS,
    kernels_cuda::attn::INDEX_CLAIMS,
    kernels_cuda::attn::POOL_CLAIMS,
    // `HC_CLAIMS` reads from `norm`, beside the five hyper-connection
    // kernels firing the same C++ namespace — the `GATE_CLAIMS` shape.
    kernels_cuda::norm::HC_CLAIMS,
];

/// The points a plane's `#[claims]` impl blocks answer — baker's claim
/// table, consulted ahead of the routine `canon` attributes. One slice per
/// migrated family, concatenated; a family lands by adding its line, and the
/// macro grows an aggregate when the list is long enough to want one.
#[must_use]
pub fn point_claims(backend: Backend) -> &'static [&'static str] {
    static CUDA: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    match backend {
        Backend::Cuda => CUDA.get_or_init(|| CUDA_CLAIMS.concat()),
        Backend::Metal => &[],
    }
}

/// The plane symbol whose `#[routine(canon = ..)]` answers a claim, or none.
///
/// There is no second walk behind this one. `canon::DEFAULTS` used to say
/// that an unanswered `lm_head` falls back on `matmul`; a delegation is a
/// claim now, written in the plane's own `#[claims]` block where it can be
/// read and where `point_claims` reports it, so a claim nothing spells is
/// simply a backlog row.
#[must_use]
pub fn canon_symbol(backend: Backend, claim: &str) -> Option<&'static str> {
    match backend {
        Backend::Cuda => sigs()
            .iter()
            .find(|k| k.canon == Some(claim))
            .map(|k| k.symbol),
        Backend::Metal => {
            static CLAIMS: std::sync::OnceLock<std::collections::BTreeMap<String, &'static str>> =
                std::sync::OnceLock::new();
            CLAIMS
                .get_or_init(|| {
                    kernels_metal::declared()
                        .into_iter()
                        .filter_map(|d| d.canon.map(|c| (c.to_string(), d.name)))
                        .collect()
                })
                .get(claim)
                .copied()
        }
    }
}

#[must_use]
pub fn out_shape(
    rule: kernels::OutRule,
    input_shapes: &[&crate::trace::Shape],
    input_dtypes: &[crate::trace::DType],
    params: &[u32],
) -> Option<(crate::trace::Shape, crate::trace::DType)> {
    use crate::trace::{Dim, Shape};
    use kernels::{OutRule, OutWidth};
    let width_of = |shape: &Shape| -> Option<u32> {
        match shape.0.last()? {
            Dim::Const(w) => Some(*w),
            _ => None,
        }
    };
    match rule {
        OutRule::Unstated => None,
        OutRule::Like { of } => {
            let at = usize::from(of);
            Some(((*input_shapes.get(at)?).clone(), *input_dtypes.get(at)?))
        }
        OutRule::Shaped { rows_of, width } => {
            let at = usize::from(rows_of);
            let base = *input_shapes.get(at)?;
            let dtype = *input_dtypes.get(at)?;
            let w = match width {
                OutWidth::Half { of } => width_of(input_shapes.get(usize::from(of))?)? / 2,
                OutWidth::Of { of } => width_of(input_shapes.get(usize::from(of))?)?,

                OutWidth::Weight { .. } => return None,
                OutWidth::Param { of } => *params.get(usize::from(of))?,
            };
            let mut dims = base.0[..base.0.len().saturating_sub(1)].to_vec();
            dims.push(Dim::Const(w));
            Some((Shape(dims), dtype))
        }
        OutRule::Split { of, dim_param } => {
            let at = usize::from(of);
            let base = *input_shapes.get(at)?;
            let dtype = *input_dtypes.get(at)?;
            let total = width_of(base)?;
            let d = *params.get(usize::from(dim_param))?;
            if d == 0 || total % d != 0 {
                return None;
            }
            let mut dims = base.0[..base.0.len().saturating_sub(1)].to_vec();
            dims.push(Dim::Const(total / d));
            dims.push(Dim::Const(d));
            Some((Shape(dims), dtype))
        }
    }
}

pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    kernels::sig_in(sigs(), symbol)
}

pub const ARITY_EXCEPTIONS: &[&str] = &[];

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

const fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

pub fn check_plan(plan: &ForwardPlan) -> Vec<String> {
    let mut problems = Vec::new();
    let backend = Backend::of_family(&plan.family);

    for b in &plan.runtime {
        if kernels::runtime::tier1(&b.name).is_none() && !b.name.contains('.') {
            problems.push(format!(
                "{}: names runtime value `{}`, which is neither in the tier-1 \
                 vocabulary nor spelled as a plane's dotted key",
                plan.family, b.name
            ));
        }
    }

    let mut peeled = 0usize;

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
                let span = arms.iter().map(|a| a.ops as usize).sum::<usize>() + *else_ops as usize;
                regioned = regioned.max(span);
            }
            OpKind::Launch {
                kernel,
                weights,
                params,
                ..
            } => match backend.and_then(|b| stated_in(b, kernel)) {
                None if backend.is_none()
                    && (kernel.starts_with("canon::")
                        || stated_in(Backend::Cuda, kernel).is_some()
                        || stated_in(Backend::Metal, kernel).is_some()) => {}
                None => problems.push(format!(
                    "{}: launches `{kernel}`, which no {} kernel declares",
                    plan.family,
                    match backend {
                        Some(b) => format!("{b:?}").to_lowercase(),

                        None => "backend's".to_string(),
                    }
                )),
                Some(k) if k.whole && inside_peel => problems.push(format!(
                    "{}: `{kernel}` is declared `whole` but is stated inside a Peel \
                     region, which gives it a row window it cannot honour",
                    plan.family
                )),
                Some(k) => {
                    let bound = weights.iter().filter(|w| !w.starts_with("scale.")).count();
                    if let Some(why) = arity_problem(&k, kernel, bound, op, inside_value_region) {
                        problems.push(format!("{}: {why}", plan.family));
                    }

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

pub fn in_place_pairs(plan: &ForwardPlan, kernel: &str) -> Vec<(u32, u32)> {
    Backend::of_family(&plan.family)
        .and_then(|b| stated_in(b, kernel))
        .map(|s| s.in_place())
        .unwrap_or_default()
}

pub fn semantic_in_place(kind: &OpKind) -> &'static [(u32, u32)] {
    let _ = kind;
    &[]
}
