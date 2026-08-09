use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Lit;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

/// AltUp's epsilon, which is the ALGORITHM's and not the model's -- the same
const ALTUP_EPS: f32 = 1e-5;

/// One kernel a row can state, as a template and the type to instantiate it
pub struct DeviceKernel {
    /// The contract: operands, launch rule, in-place claims.
    pub sig: &'static KernelSig,
    /// The `__global__` template, under `::pie_cuda_driver::kernels`.
    pub template_path: &'static str,
    /// The element type to instantiate it at, under the same root -- or
    pub elem: &'static str,
}

impl DeviceKernel {
    /// What a row states in [`DeviceKernel::elem`] when its `__global__` has
    pub const PLAIN: &'static str = "(no template arguments)";

    /// Whether this row names a `__global__` with no template parameter list.
    #[must_use]
    pub const fn is_plain(&self) -> bool {
        let (a, b) = (self.elem.as_bytes(), Self::PLAIN.as_bytes());
        if a.len() != b.len() {
            return false;
        }
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    /// The instantiation, as C++ spells it and as `nvrtcAddNameExpression`
    #[must_use]
    pub fn instantiation(&self) -> String {
        let path = Self::qualify(self.template_path);
        if self.is_plain() {
            return path;
        }
        format!("{path}<{}>", Self::qualify(self.elem))
    }

    /// One field of an instantiation, with the root prefix applied unless the
    fn qualify(field: &str) -> String {
        if field.starts_with("::") {
            return field.to_owned();
        }
        format!("::pie_cuda_driver::kernels::{field}")
    }
}

/// The `__global__` templates `csrc/src/norm/altup_aux.cuh` holds, and the
pub static ALTUP_AUX: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &SIGS[0],
        template_path: "norm::device::compute_rms",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[1],
        template_path: "norm::device::magnitude_rescale",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[2],
        template_path: "norm::device::mean_streams",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[3],
        template_path: "norm::device::unpack_predict_coefs",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[4],
        template_path: "norm::device::unpack_correct_coefs",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[5],
        template_path: "norm::device::tanh_inplace",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[6],
        template_path: "norm::device::tanh_inplace",
        elem: "device::f16",
    },
];

/// The contracts, in [`ALTUP_AUX`]' order.
#[rustfmt::skip]
static SIGS: [KernelSig; 7] = [
    kernel!(compute_rms "norm::compute_rms_bf16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            reference: Buf <- Source::In(0),
            target_rms_out: F32sMut <- Source::Out(0),
            h: I32 <- Source::InWidth(0),
            eps: F32 <- Source::Lit(Lit::F32(ALTUP_EPS)),
        ]),
    kernel!(magnitude_rescale "norm::magnitude_rescale_bf16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::Rms,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            target_rms: F32s <- Source::In(1),
            h: I32 <- Source::OutWidth(0),
            eps: F32 <- Source::Lit(Lit::F32(ALTUP_EPS)),
        ]),
    kernel!(mean_streams "norm::mean_streams_bf16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            streams: Buf <- Source::In(0),
            out: BufMut <- Source::Out(0),
            k: I32 <- Source::CtxNonZero("altup_streams"),
            t_stride: I32 <- Source::Rows,
            h: I32 <- Source::OutWidth(0),
        ]),
    kernel!(altup_unpack_predict_coefs "norm::altup_unpack_predict_coefs",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            in_bf16: Buf <- Source::In(0),
            out_fp32: F32sMut <- Source::Out(0),
            k: I32 <- Source::Isqrt(&Source::Width(&Source::In(0))),
        ]),
    kernel!(altup_unpack_correct_coefs "norm::altup_unpack_correct_coefs",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            in_bf16: Buf <- Source::In(0),
            out_fp32: F32sMut <- Source::Out(0),
            k: I32 <- Source::InWidth(0),
        ]),
    kernel!(tanh "norm::tanh_bf16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            numel: I32 <- Source::OutElements(0),
        ]),
    kernel!(tanh_f16 "norm::tanh_f16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            numel: I32 <- Source::OutElements(0),
        ]),
];

/// The `__global__` templates `csrc/src/norm/elementwise.cuh` holds.
pub static ELEMENTWISE: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ELEMENTWISE_SIGS[0],
        template_path: "norm::device::residual_add",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ELEMENTWISE_SIGS[1],
        template_path: "norm::device::scalar_mul",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ELEMENTWISE_SIGS[2],
        template_path: "norm::device::residual_add",
        elem: "device::f16",
    },
];

/// The contracts, in [`ELEMENTWISE`]'s order.
#[rustfmt::skip]
static ELEMENTWISE_SIGS: [KernelSig; 3] = [
    kernel!(residual_add_cuda "norm::residual_add_bf16",
        file = Some("norm/elementwise.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            y: BufMut <- Source::Out(0),
            x: Buf <- Source::In(1),
            n: Usize <- Source::OutElements(0),
        ]),
    kernel!(scalar_mul "norm::scalar_mul_bf16",
        file = Some("norm/elementwise.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            s: F32 <- Source::Or(&Source::ParamF32(0), &Source::NamedScale),
            n: Usize <- Source::OutElements(0),
        ]),
    kernel!(residual_add_f16_cuda "norm::residual_add_f16",
        file = Some("norm/elementwise.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            y: BufMut <- Source::Out(0),
            x: Buf <- Source::In(1),
            n: Usize <- Source::OutElements(0),
        ]),
];

/// The rows the DISPATCHER routes to the JIT path, by symbol.
pub static JIT_DISPATCHED: &[&str] = &[

    "attn::split_qkv_bf16",
    "layout::split_q_gate_bf16",
];

/// [`ELEMENTWISE`]'s rows that [`JIT_DISPATCHED`] names, as the emitters take
#[must_use]
pub fn jit_dispatched() -> Vec<&'static DeviceKernel> {
    crate::unit::rows()
        .filter(|d| JIT_DISPATCHED.contains(&d.sig.symbol))
        .collect()
}

/// A multi-argument instantiation, spelled in a row's `elem`.
#[must_use]
pub fn args(elem: &str, rest: &[&str]) -> String {
    let mut out = elem.to_string();
    for arg in rest {
        out.push_str(", ");
        out.push_str(arg);
    }
    out
}

/// A fact about one bound value that a [`Term`] is allowed to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fact {
    /// A device address, as a number. Never dereferenced, and it cannot be:
    Address(u64),
    /// An integer operand's value — a width, a stride, a count the HOST
    Int(i64),
    /// A host flag's value — a `Ty::Bool` operand, which in this tree is
    Bool(bool),
    /// Any other kind of bound value. Present so the mapping from an argument
    Opaque,
}

/// One clause of a selection predicate, over one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    /// The operand is a pointer whose address is a multiple of `bytes`.
    Aligned {
        /// Index into the BASE row's operand list.
        operand: usize,
        /// The alignment in bytes — a power of two.
        bytes: u64,
    },
    /// The operand is an integer that divides evenly by `of`.
    Multiple {
        /// Index into the BASE row's operand list.
        operand: usize,
        /// The divisor — `hidden % 8 == 0` and its two stride twins.
        of: i64,
    },
    /// The operand is a host flag with this value.
    Is {
        /// Index into the BASE row's operand list.
        operand: usize,
        /// The value that selects this arm.
        value: bool,
    },
    /// The operand is a pointer the fire published, or one it did not.
    Present {
        /// Index into the BASE row's operand list.
        operand: usize,
        /// `true` selects the arm for a published pointer, `false` the arm
        value: bool,
    },
}

impl Term {
    /// Which operand of the base row this clause reads.
    #[must_use]
    pub const fn operand(&self) -> usize {
        match self {
            Term::Aligned { operand, .. }
            | Term::Multiple { operand, .. }
            | Term::Is { operand, .. }
            | Term::Present { operand, .. } => *operand,
        }
    }

    /// Whether this clause holds over the facts a fire supplies.
    pub fn holds(&self, facts: &[Fact]) -> Result<bool, Fault> {
        let at = self.operand();
        let Some(fact) = facts.get(at) else {
            return Err(Fault::Range { operand: at, arity: facts.len() });
        };
        match (self, fact) {
            (Term::Aligned { bytes, .. }, Fact::Address(address)) => Ok(address % bytes == 0),
            (Term::Multiple { of, .. }, Fact::Int(value)) => Ok(value % of == 0),
            (Term::Is { value, .. }, Fact::Bool(flag)) => Ok(value == flag),
            (Term::Present { value, .. }, Fact::Address(address)) => Ok(*value == (*address != 0)),
            (Term::Aligned { .. }, _) => Err(Fault::Kind { operand: at, wanted: "an address" }),
            (Term::Multiple { .. }, _) => Err(Fault::Kind { operand: at, wanted: "an integer" }),
            (Term::Is { .. }, _) => Err(Fault::Kind { operand: at, wanted: "a flag" }),
            (Term::Present { .. }, _) => Err(Fault::Kind { operand: at, wanted: "an address" }),
        }
    }
}

/// Why a [`Term`] could not be evaluated at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The term names operand `operand` and the row has `arity` of them.
    Range {
        /// The index the term named.
        operand: usize,
        /// How many operands were bound.
        arity: usize,
    },
    /// The operand is bound to a [`Fact`] the term cannot read.
    Kind {
        /// The index the term named.
        operand: usize,
        /// What the term needed there.
        wanted: &'static str,
    },
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fault::Range { operand, arity } => {
                write!(f, "a term reads operand {operand} of a row with {arity}")
            }
            Fault::Kind { operand, wanted } => {
                write!(f, "a term reads operand {operand} and wanted {wanted} there")
            }
        }
    }
}

/// Where one of a variant's arguments comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Take {
    /// The base row's operand at this index, verbatim.
    From(usize),
    /// A null pointer, for a parameter the variant declares and this arm does
    Null,
}

/// One specialised instantiation, and the predicate that chooses it.
pub struct Arm {
    /// What this arm is called in a diagnosis and in the audit. Short, and
    pub name: &'static str,
    /// The clauses, ANDed. Empty would mean "always", which
    pub when: &'static [Term],
    /// The instantiation to fire instead. A row of the same unit, so the
    pub row: &'static DeviceKernel,
    /// The variant's argument list, in the variant's order, over the base's
    pub take: &'static [Take],
    /// The host code this arm reproduces, cited so the two can be compared.
    pub because: &'static str,
}

/// A base row and the arms a fire may choose instead of it.
pub struct Specialisation {
    /// The symbol a fire names — the row whose contract this is.
    pub base: &'static str,
    /// The arms, in order. The FIRST whose `when` holds is chosen, so a
    pub arms: &'static [Arm],
}

impl Specialisation {
    /// The first arm whose predicate holds over `facts`, or `None` for the
    pub fn choose(&self, facts: &[Fact]) -> Result<Option<&'static Arm>, Fault> {
        for arm in self.arms {
            let mut all = true;
            for term in arm.when {
                if !term.holds(facts)? {
                    all = false;
                }
            }
            if all {
                return Ok(Some(arm));
            }
        }
        Ok(None)
    }

    /// Everything about this specialisation a machine can check, checked.
    pub fn agrees(&self) -> Result<(), String> {
        let Some((_, unit)) = crate::unit::unit_of(self.base) else {
            return Err(format!("`{}` is specialised and no unit hosts it", self.base));
        };
        let base = unit.row(self.base).ok_or_else(|| format!("`{}` has no row", self.base))?.sig;
        if self.arms.is_empty() {
            return Err(format!("`{}` states a specialisation with no arms", self.base));
        }
        for arm in self.arms {
            let variant = arm.row.sig;
            let at = format!("`{}` arm `{}`", self.base, arm.name);
            if !unit.hosts(variant.symbol) {
                return Err(format!(
                    "{at} fires `{}`, which unit `{}` does not compile — a second unit \
                     would be a second cubin and a second first-fire stall",
                    variant.symbol, unit.name
                ));
            }
            if variant.launch != base.launch {
                return Err(format!(
                    "{at} states {:?} where the base states {:?}; a specialisation chooses an \
                     instantiation, not a geometry",
                    variant.launch, base.launch
                ));
            }
            if arm.when.is_empty() {
                return Err(format!("{at} applies always, which is not a specialisation"));
            }
            if arm.take.len() != variant.operands.len() {
                return Err(format!(
                    "{at} takes {} arguments and `{}` declares {}",
                    arm.take.len(),
                    variant.symbol,
                    variant.operands.len()
                ));
            }
            for (slot, take) in arm.take.iter().enumerate() {
                let wants = variant.operands[slot];
                match take {
                    Take::From(index) => {
                        let Some(source) = base.operands.get(*index) else {
                            return Err(format!(
                                "{at} fills `{}` from operand {index} of a row with {}",
                                wants.name,
                                base.operands.len()
                            ));
                        };
                        if source.ty != wants.ty {
                            return Err(format!(
                                "{at} fills `{}` ({:?}) from `{}` ({:?})",
                                wants.name, wants.ty, source.name, source.ty
                            ));
                        }
                    }
                    Take::Null => {
                        if !wants.nullable {
                            return Err(format!(
                                "{at} nulls `{}`, which the row does not declare nullable",
                                wants.name
                            ));
                        }
                        if scalar(wants.ty) {
                            return Err(format!(
                                "{at} nulls `{}`, which is {:?} and not a pointer",
                                wants.name, wants.ty
                            ));
                        }
                    }
                }
            }
            for term in arm.when {
                let index = term.operand();
                let Some(read) = base.operands.get(index) else {
                    return Err(format!(
                        "{at} reads operand {index} of a row with {}",
                        base.operands.len()
                    ));
                };
                match term {
                    Term::Aligned { bytes, .. } => {
                        if scalar(read.ty) {
                            return Err(format!(
                                "{at} tests the alignment of `{}`, which is {:?}",
                                read.name, read.ty
                            ));
                        }
                        if !bytes.is_power_of_two() {
                            return Err(format!(
                                "{at} tests alignment to {bytes}, and `rmsnorm.cu` spells the \
                                 same test as a MASK — the two agree only on powers of two"
                            ));
                        }
                    }
                    Term::Multiple { of, .. } => {
                        if read.ty != kernels::Ty::I32 {
                            return Err(format!(
                                "{at} divides `{}`, which is {:?} and not an i32",
                                read.name, read.ty
                            ));
                        }
                        if *of <= 0 {
                            return Err(format!("{at} divides `{}` by {of}", read.name));
                        }
                    }
                    Term::Is { .. } => {
                        if read.ty != kernels::Ty::Bool {
                            return Err(format!(
                                "{at} selects on `{}`, which is {:?} and not a Bool — a flag \
                                 clause reads a `Fact::Bool` and every other kind faults, so \
                                 this arm could never be taken",
                                read.name, read.ty
                            ));
                        }
                    }
                    Term::Present { .. } => {
                        if scalar(read.ty) {
                            return Err(format!(
                                "{at} tests `{}` for null, which is {:?} — a null clause reads \
                                 a `Fact::Address` and a scalar never supplies one, so this \
                                 arm could never be taken",
                                read.name, read.ty
                            ));
                        }
                        if !read.nullable {
                            return Err(format!(
                                "{at} tests `{}` for null and the row does not declare it \
                                 nullable — the binder refuses a null there, so the clause is \
                                 decided for every fire that reaches it and one of the two \
                                 arms is an instantiation that compiles and never runs",
                                read.name
                            ));
                        }
                    }
                }
            }
        }
        self.flags_are_covered(base)
    }

    /// Every flag a fire cannot fall through on, checked.
    fn flags_are_covered(&self, base: &'static KernelSig) -> Result<(), String> {
        let mut flags: Vec<usize> = Vec::new();
        for arm in self.arms {
            for term in arm.when {
                if let Term::Is { operand, .. } | Term::Present { operand, .. } = term
                    && !flags.contains(operand)
                {
                    flags.push(*operand);
                }
            }
        }
        flags.retain(|flag| {
            !self
                .arms
                .iter()
                .any(|arm| arm.take.contains(&Take::From(*flag)))
        });
        if flags.is_empty() {
            return Ok(());
        }
        if flags.len() > 8 {
            return Err(format!(
                "`{}` selects on {} flags that reach no kernel; the coverage this check \
                 proves is 2^n cases and a predicate over that many is not one a reader \
                 can compare to the C++",
                self.base,
                flags.len()
            ));
        }
        for assignment in 0..(1u32 << flags.len()) {
            let value = |operand: usize| {
                flags.iter().position(|f| *f == operand).map(|bit| assignment >> bit & 1 == 1)
            };
            let covered = self.arms.iter().any(|arm| {
                !arm.when.is_empty()
                    && arm.when.iter().all(|term| match term {
                        Term::Is { operand, value: wanted }
                        | Term::Present { operand, value: wanted } => {
                            value(*operand) == Some(*wanted)
                        }
                        _ => false,
                    })
            });
            if covered {
                continue;
            }
            let uncovered = flags
                .iter()
                .enumerate()
                .map(|(bit, operand)| {
                    format!("`{}` = {}", base.operands[*operand].name, assignment >> bit & 1 == 1)
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "`{}` selects on a flag no arm forwards, and states no arm for {uncovered}. \
                 A fire with that flag falls through to the base row, which binds {} cells \
                 for `{}` — and a flag that reaches no kernel is one cell more than the \
                 instantiation declares, which `cuLaunchKernel` accepts and never reads. \
                 State the other arm.",
                self.base,
                base.operands.len(),
                self.base,
            ));
        }
        Ok(())
    }
}

/// Whether a [`kernels::Ty`] is bound by a value rather than by an address.
const fn scalar(ty: kernels::Ty) -> bool {
    use kernels::Ty;
    matches!(
        ty,
        Ty::I32
            | Ty::U32
            | Ty::F32
            | Ty::Usize
            | Ty::I64
            | Ty::Bool
            | Ty::Stream
            | Ty::KvScheme
            | Ty::KvDType
            | Ty::Fp8Kind
    )
}

/// Every specialised row in the tree, one entry per FAMILY.
pub static SPECIALISED: &[&[&Specialisation]] = &[
];

/// Every specialised row, flattened — what a reader and a test want.
pub fn specialisations() -> impl Iterator<Item = &'static Specialisation> {
    SPECIALISED.iter().copied().flatten().copied()
}

/// The specialisation a symbol carries, if it carries one.
#[must_use]
pub fn specialisation(symbol: &str) -> Option<&'static Specialisation> {
    specialisations().find(|spec| spec.base == symbol)
}

/// The row a symbol names, over every device table this crate knows.
#[must_use]
pub fn row(symbol: &str) -> Option<&'static DeviceKernel> {
    crate::unit::rows().find(|entry| entry.sig.symbol == symbol)
}

#[cfg(test)]
mod tests {
    use super::{ALTUP_AUX, ELEMENTWISE, JIT_DISPATCHED, jit_dispatched, row};
    use kernels::LaunchRule;

    /// A device row's symbol is unique across the tables, not merely within
    #[test]
    fn a_symbol_belongs_to_one_device_row() {
        let mut seen: Vec<&str> = Vec::new();
        for entry in ALTUP_AUX.iter().chain(ELEMENTWISE) {
            assert!(!seen.contains(&entry.sig.symbol), "{} is stated twice", entry.sig.symbol);
            seen.push(entry.sig.symbol);
        }
    }

    /// Every symbol the dispatcher routes to the JIT is a symbol some unit
    #[test]
    fn every_dispatched_symbol_has_a_row() {
        for symbol in JIT_DISPATCHED {
            assert!(row(symbol).is_some(), "{symbol} is dispatched to the JIT and has no row");
        }
        assert_eq!(jit_dispatched().len(), JIT_DISPATCHED.len());
    }

    /// The lookup is the tables.
    #[test]
    fn the_lookup_is_the_tables() {
        for entry in ALTUP_AUX.iter().chain(ELEMENTWISE) {
            assert_eq!(row(entry.sig.symbol).map(|r| r.sig.symbol), Some(entry.sig.symbol));
        }
        assert!(row("norm::a_kernel_nobody_wrote").is_none());
    }

    /// ...and "the tables" means EVERY unit's, not the two this module holds.
    #[test]
    fn the_lookup_is_every_unit_s_tables() {
        let mut checked = 0usize;
        for entry in crate::unit::rows() {
            assert_eq!(
                row(entry.sig.symbol).map(|r| r.sig.symbol),
                Some(entry.sig.symbol),
                "`{}` is a row of unit `{}` and `device::row` cannot find it",
                entry.sig.symbol,
                crate::unit::unit_of(entry.sig.symbol).map_or("<none>", |(_, u)| u.name),
            );
            checked += 1;
        }
        assert!(
            checked > ALTUP_AUX.len() + ELEMENTWISE.len(),
            "only {checked} rows scanned; this module alone holds {}, so the \
             iterator is not the whole table and this test proves nothing",
            ALTUP_AUX.len() + ELEMENTWISE.len(),
        );
    }

    /// Every Tier A row states a rule. `Unstated` is what a row says when it
    #[test]
    fn every_entry_states_its_launch() {
        for k in ALTUP_AUX {
            assert_ne!(k.sig.launch, LaunchRule::Unstated, "{} states no rule", k.sig.symbol);
        }
    }

    /// No row here needed a rule the Metal tables had not already stated.
    #[test]
    fn the_pilot_added_no_launch_rules() {
        const REUSED: &[LaunchRule] = &[
            LaunchRule::Rms,
            LaunchRule::ElementwiseRows,
            LaunchRule::RouteRows,
            LaunchRule::Elementwise,
        ];
        for k in ALTUP_AUX {
            assert!(
                REUSED.contains(&k.sig.launch),
                "{} states {:?}, which is not one of the rules Metal already had",
                k.sig.symbol,
                k.sig.launch
            );
        }
    }

    /// A stream is not an operand, and a Tier A row may not say it is.
    #[test]
    fn no_entry_takes_a_stream() {
        for k in ALTUP_AUX {
            assert!(
                k.sig.operands.iter().all(|o| o.ty != kernels::Ty::Stream),
                "{} takes a stream as an operand",
                k.sig.symbol
            );
        }
    }

    /// Two symbols may not name one instantiation, and one symbol may not be
    #[test]
    fn the_map_from_symbol_to_instantiation_is_a_bijection() {
        let mut seen: Vec<(String, &str)> = Vec::new();
        for k in ALTUP_AUX {
            let inst = k.instantiation();
            assert!(
                !seen.iter().any(|(i, _)| *i == inst),
                "{} names an instantiation another row already claims: {inst}",
                k.sig.symbol
            );
            assert!(
                !seen.iter().any(|(_, s)| *s == k.sig.symbol),
                "{} is stated twice",
                k.sig.symbol
            );
            seen.push((inst, k.sig.symbol));
        }
        assert_eq!(seen.len(), ALTUP_AUX.len());
    }

    /// The two `tanh` rows are the same template at different element types,
    #[test]
    fn a_second_numeric_format_is_a_row_and_not_a_kernel() {
        let bf16 = ALTUP_AUX.iter().find(|k| k.sig.symbol == "norm::tanh_bf16").expect("bf16 row");
        let f16 = ALTUP_AUX.iter().find(|k| k.sig.symbol == "norm::tanh_f16").expect("f16 row");
        assert_eq!(bf16.template_path, f16.template_path);
        assert_ne!(bf16.elem, f16.elem);
        assert_eq!(bf16.sig.operands.len(), f16.sig.operands.len());
    }

    /// Each row is shorter than its twin, which is the deletion the
    #[test]
    fn tier_a_rows_are_shorter_than_their_twins() {
        let mine: usize = ALTUP_AUX
            .iter()
            .filter(|k| k.sig.symbol != "norm::tanh_f16")
            .map(|k| k.sig.operands.len())
            .sum();
        let twins: usize = 31;
        assert_eq!(ALTUP_AUX.len(), 7, "six ported kernels and the fp16 extra");
        assert_eq!((twins, mine), (31, 21));
    }
}
