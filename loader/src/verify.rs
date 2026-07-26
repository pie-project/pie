//! Does this plan honour its contract?
//!
//! Verification is deliberately not a second compiler. It takes the plan
//! exactly as the driver sees it and asks questions that can be answered from
//! the plan alone plus the filesystem — which is what makes it a *second
//! opinion* rather than a restatement (`architecture.md` §8). The compiler and
//! this module share no code: if `compile` had a bug that made it emit a plan
//! that does not produce a declared tensor, nothing in `compile` would notice,
//! because the same wrong belief produced both halves.
//!
//! The driver-side counterpart in `loaded_model.cpp` compares
//! `covered_contract_count` against `runtime_tensor_count`, but both are
//! assigned from `view.tensors.len` — it is literally `if (x != x)`. The real
//! check lives here, where the two sides being compared were derived
//! separately.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::load_plan::{LoadPlan, StorageInstr};
use crate::types::Encoding;

/// The plan, reduced to what verification can read.
///
/// Both callers build one: the CLI from a Rust [`LoadPlan`], and the FFI
/// boundary from the `PieLoaderPlan` the C++ driver is holding. The FFI case is
/// the one that matters — verifying the *marshalled* view means a bug in the
/// marshalling is in scope, which it would not be if verification re-read the
/// Rust plan the driver never sees.
pub struct PlanView<'a> {
    pub version: u32,
    pub compiler_version: u64,
    pub files: Vec<FileView<'a>>,
    pub sources: Vec<SourceView<'a>>,
    pub tensors: Vec<TensorView<'a>>,
    pub instr_count: usize,
    pub schedule: Vec<u32>,
    /// Names the instruction stream finalizes, in stream order. Duplicates are
    /// kept: finalizing the same name twice is one of the things being checked.
    pub finalized: Vec<&'a str>,
    /// Every instruction that reads the checkpoint, and the file it reads.
    pub reads: Vec<ReadView>,
}

pub struct ReadView {
    pub instr: u32,
    pub file_id: u32,
}

pub struct FileView<'a> {
    pub id: u32,
    pub path: &'a str,
    pub size_bytes: u64,
}

pub struct SourceView<'a> {
    pub name: &'a str,
    pub file_id: u32,
    pub offset_bytes: u64,
    pub span_bytes: u64,
}

pub struct TensorView<'a> {
    pub name: &'a str,
    pub shape: Vec<i64>,
    /// Owned, and always normalized: the plan and the contract are built by
    /// different code, so two spellings of the same quantization must not read
    /// as a disagreement.
    pub encoding: Encoding,
}

impl<'a> TensorView<'a> {
    pub fn new(name: &'a str, shape: &[i64], encoding: &Encoding) -> Self {
        Self {
            name,
            shape: shape.to_vec(),
            encoding: crate::types::normalize_encoding(encoding),
        }
    }
}

impl<'a> From<&'a LoadPlan> for PlanView<'a> {
    fn from(plan: &'a LoadPlan) -> Self {
        Self {
            version: plan.version,
            compiler_version: plan.compiler_version,
            files: plan
                .files
                .iter()
                .map(|file| FileView {
                    id: file.id.0,
                    path: file.path.as_str(),
                    size_bytes: file.size_bytes,
                })
                .collect(),
            sources: plan
                .sources
                .iter()
                .map(|source| SourceView {
                    name: source.name.as_str(),
                    file_id: source.file_id.0,
                    offset_bytes: source.file_offset,
                    span_bytes: source.span_bytes,
                })
                .collect(),
            tensors: plan
                .tensors
                .iter()
                .map(|tensor| {
                    TensorView::new(
                        tensor.name.as_str(),
                        tensor.shape.as_slice(),
                        &tensor.encoding,
                    )
                })
                .collect(),
            instr_count: plan.instrs.len(),
            schedule: plan.schedule.iter().map(|id| id.0).collect(),
            finalized: plan
                .instrs
                .iter()
                .filter_map(|instr| match instr {
                    StorageInstr::Finalize { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .collect(),
            reads: plan
                .instrs
                .iter()
                .filter_map(|instr| {
                    let (instr, file_id) = match instr {
                        StorageInstr::ExtentWrite { id, source, .. }
                        | StorageInstr::BulkExtentWrite { id, source, .. } => {
                            (id.0, source.file_id.0)
                        }
                        StorageInstr::SlabScatter { id, file_id, .. } => (id.0, file_id.0),
                        StorageInstr::TileMap {
                            id,
                            source: Some(source),
                            ..
                        } => (id.0, source.file_id.0),
                        _ => return None,
                    };
                    Some(ReadView { instr, file_id })
                })
                .collect(),
        }
    }
}

/// What the runtime demands, independently of how the plan proposes to deliver
/// it: a name, a shape, and an encoding per tensor.
///
/// The point of passing this separately is that it is derived by different code
/// than the plan. `arch/` decides *what* the model needs; `frontend` ->
/// `optimizer` -> `planner` decide *how*. A compiler bug that loses a dimension
/// somewhere in the second half cannot also corrupt the first half, so the two
/// disagreeing is a real signal.
pub struct ContractView<'a> {
    pub tensors: Vec<TensorDemand<'a>>,
}

/// One tensor the runtime demands.
///
/// `shape` and `encoding` are optional because the two demanders know different
/// amounts. `arch/` authored the tensor, so it states all three. A *driver*
/// declaring what it will bind knows the name always and the shape usually, but
/// generally not the encoding: which quantization a weight ends up in is the
/// loader's decision under the runtime quant policy, and the binder probes
/// `->dtype()` afterwards rather than demanding one. Demanding an encoding the
/// driver merely guessed would turn a correct plan into a violation, so an
/// unstated field is `None` and is not checked.
pub struct TensorDemand<'a> {
    pub name: &'a str,
    pub shape: Option<Vec<i64>>,
    pub encoding: Option<Encoding>,
    /// Absence is not a violation. Some weights are legitimately conditional —
    /// `lm_head.weight` is dropped when `tie_word_embeddings` is set, and the
    /// binder falls back to the embedding table.
    pub optional: bool,
}

impl<'a> TensorDemand<'a> {
    /// A demand that pins all three fields. Used for a contract `arch/` authored.
    pub fn exact(name: &'a str, shape: &[i64], encoding: &Encoding) -> Self {
        Self {
            name,
            shape: Some(shape.to_vec()),
            encoding: Some(crate::types::normalize_encoding(encoding)),
            optional: false,
        }
    }

    /// A demand that pins only what the demander actually knows.
    pub fn declared(name: &'a str, shape: Option<Vec<i64>>, optional: bool) -> Self {
        Self {
            name,
            shape,
            encoding: None,
            optional,
        }
    }
}

impl<'a> ContractView<'a> {
    /// Read a contract as the rank that will execute it sees it.
    pub fn of(contract: &'a crate::contract::ModelContract) -> Self {
        Self {
            tensors: contract
                .tensors
                .iter()
                .map(|tensor| {
                    TensorDemand::exact(tensor.name.as_str(), &tensor.shape, &tensor.encoding)
                })
                .collect(),
        }
    }
}

/// One reason a plan was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    /// The runtime tensor at fault, when the violation is about one.
    pub tensor: Option<String>,
    pub message: String,
}

impl Violation {
    fn plan(message: impl Into<String>) -> Self {
        Self {
            tensor: None,
            message: message.into(),
        }
    }

    fn tensor(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            tensor: Some(name.into()),
            message: message.into(),
        }
    }
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.tensor {
            Some(tensor) => write!(f, "'{tensor}': {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

/// What a plan was found to guarantee. Returned only when nothing was wrong,
/// so holding one is evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certificate {
    pub tensors: usize,
    pub instructions: usize,
    pub files: usize,
    /// Bytes the plan reads out of the checkpoint.
    pub source_bytes: u64,
}

impl fmt::Display for Certificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "verified: {} tensors, {} instructions, {} files, {} source bytes",
            self.tensors, self.instructions, self.files, self.source_bytes
        )
    }
}

/// Check a plan against the contract it declares.
///
/// The contract a plan carries is its `tensors` list: those are the names and
/// types a driver will look up after the load. Everything below asks whether
/// the instruction stream actually delivers them.
pub fn verify(
    plan: &PlanView<'_>,
    contract: Option<&ContractView<'_>>,
) -> Result<Certificate, Vec<Violation>> {
    let mut found = Vec::new();
    check_versions(plan, &mut found);
    check_schedule(plan, &mut found);
    check_coverage(plan, &mut found);
    check_files(plan, &mut found);
    if let Some(contract) = contract {
        check_contract(plan, contract, &mut found);
    }
    if !found.is_empty() {
        return Err(found);
    }
    Ok(Certificate {
        tensors: plan.tensors.len(),
        instructions: plan.instr_count,
        files: plan.files.len(),
        source_bytes: plan.sources.iter().map(|source| source.span_bytes).sum(),
    })
}

fn check_versions(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
    if plan.version != crate::load_plan::LOAD_PLAN_VERSION {
        found.push(Violation::plan(format!(
            "plan version {} does not match loader version {}",
            plan.version,
            crate::load_plan::LOAD_PLAN_VERSION
        )));
    }
    let expected = crate::load_plan::compiler_version();
    if plan.compiler_version != expected {
        found.push(Violation::plan(format!(
            "plan compiler version {:#x} does not match loader version {expected:#x}",
            plan.compiler_version
        )));
    }
}

/// The schedule must be a permutation of the instructions.
///
/// Anything else means an instruction is dropped or run twice, and both are
/// silent: the load completes and a buffer holds whatever it held before.
fn check_schedule(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
    if plan.schedule.len() != plan.instr_count {
        found.push(Violation::plan(format!(
            "schedule has {} entries but the plan has {} instructions",
            plan.schedule.len(),
            plan.instr_count
        )));
    }
    let mut seen = vec![false; plan.instr_count];
    for id in &plan.schedule {
        let index = *id as usize;
        match seen.get_mut(index) {
            None => found.push(Violation::plan(format!(
                "schedule references instruction {index}, but the plan has {}",
                plan.instr_count
            ))),
            Some(true) => found.push(Violation::plan(format!(
                "instruction {index} is scheduled more than once"
            ))),
            Some(slot) => *slot = true,
        }
    }
    for (index, scheduled) in seen.iter().enumerate() {
        if !scheduled {
            found.push(Violation::plan(format!(
                "instruction {index} is never scheduled"
            )));
        }
    }
}

/// Every declared tensor must be finalized, under its declared name, exactly
/// once — and nothing else may be.
///
/// A `TensorDecl` with no `Finalize` is a weight the driver will look up and
/// not find. A `Finalize` with no `TensorDecl` is a name the driver was never
/// told to expect. Both are the plan disagreeing with itself.
fn check_coverage(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
    let mut finalized: HashMap<&str, usize> = HashMap::new();
    for name in &plan.finalized {
        *finalized.entry(*name).or_default() += 1;
    }

    let mut declared: HashSet<&str> = HashSet::new();
    for tensor in &plan.tensors {
        if !declared.insert(tensor.name) {
            found.push(Violation::tensor(
                tensor.name,
                "is declared more than once, so a driver's lookup is ambiguous",
            ));
        }
        match finalized.get(tensor.name) {
            None => found.push(Violation::tensor(
                tensor.name,
                "is declared but never finalized, so the load would leave it absent",
            )),
            Some(1) => {}
            Some(count) => found.push(Violation::tensor(
                tensor.name,
                format!("is finalized {count} times; the last write silently wins"),
            )),
        }
    }

    for name in finalized.keys() {
        if !declared.contains(name) {
            found.push(Violation::tensor(
                *name,
                "is finalized but never declared, so nothing will look it up",
            ));
        }
    }
}

/// Check the plan's file table against the filesystem.
///
/// This is the check that a cached plan is still about the checkpoint it was
/// compiled from. Every offset in the plan is a byte position inside one of
/// these files, so a file that has been replaced, truncated, or removed since
/// the compile turns the plan's offsets into reads of unrelated bytes — which
/// is far worse than a load failure, because it succeeds.
///
/// Size is a weak proxy for content, but a cheap one, and it catches the case
/// that actually happens: a re-download or a re-quantization producing a
/// different checkpoint under the same path.
fn check_files(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
    for (index, file) in plan.files.iter().enumerate() {
        if file.id as usize != index {
            found.push(Violation::plan(format!(
                "file table entry {index} declares id {}; ids must equal their \
                 index because `file_id` is used as a table offset",
                file.id
            )));
        }
        match std::fs::metadata(file.path) {
            Ok(meta) if meta.len() != file.size_bytes => found.push(Violation::plan(format!(
                "{} is {} bytes; the plan was compiled against {} bytes",
                file.path,
                meta.len(),
                file.size_bytes
            ))),
            Ok(_) => {}
            Err(err) => found.push(Violation::plan(format!(
                "{} is unreadable: {err}",
                file.path
            ))),
        }
    }

    for read in &plan.reads {
        if read.file_id as usize >= plan.files.len() {
            found.push(Violation::plan(format!(
                "instruction {} reads from file {}, but the plan declares {} files",
                read.instr,
                read.file_id,
                plan.files.len()
            )));
        }
    }

    for source in &plan.sources {
        let Some(file) = plan.files.get(source.file_id as usize) else {
            found.push(Violation::tensor(
                source.name,
                format!(
                    "reads from file {}, but the plan declares {} files",
                    source.file_id,
                    plan.files.len()
                ),
            ));
            continue;
        };
        let end = source.offset_bytes.saturating_add(source.span_bytes);
        if end > file.size_bytes {
            found.push(Violation::tensor(
                source.name,
                format!(
                    "reads bytes [{}, {end}) of {}, which is {} bytes long",
                    source.offset_bytes, file.path, file.size_bytes
                ),
            ));
        }
    }
}

/// Check the plan against what the runtime actually demands.
///
/// This is §8.2's coverage question in full: not "is the plan self-consistent"
/// but "does it produce what will be asked for, in the form it will be asked
/// for?". Shape and encoding are checked as well as presence, because a tensor
/// that exists at the wrong shape fails later, further away, and as garbage
/// output rather than a missing symbol.
fn check_contract(plan: &PlanView<'_>, contract: &ContractView<'_>, found: &mut Vec<Violation>) {
    let planned: HashMap<&str, &TensorView<'_>> = plan
        .tensors
        .iter()
        .map(|tensor| (tensor.name, tensor))
        .collect();

    for demanded in &contract.tensors {
        let Some(planned) = planned.get(demanded.name) else {
            if !demanded.optional {
                found.push(Violation::tensor(
                    demanded.name,
                    "is demanded by the contract but the plan does not declare it",
                ));
            }
            continue;
        };
        if let Some(shape) = &demanded.shape
            && planned.shape != *shape
        {
            found.push(Violation::tensor(
                demanded.name,
                format!(
                    "is planned as {:?} but the contract demands {:?}",
                    planned.shape, shape
                ),
            ));
        }
        if let Some(encoding) = &demanded.encoding
            && planned.encoding != *encoding
        {
            found.push(Violation::tensor(
                demanded.name,
                format!(
                    "is planned as {:?} but the contract demands {:?}",
                    planned.encoding, encoding
                ),
            ));
        }
    }

    // The converse is not a violation by symmetry: the loader publishes views
    // (`nemotron.rs` republishes packed experts under their original names) that
    // a given driver may never bind. Producing more than was demanded costs a
    // name, not correctness. Producing less is the failure.
}

/// How much of a declared contract a plan delivers.
///
/// [`check_contract`] answers *is the plan wrong*; this answers *how much did it
/// build*, which is what the driver reports and gates on. Splitting them keeps
/// the count out of the violation path, where an early return would skew it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContractCoverage {
    pub covered: usize,
    pub demanded: usize,
}

impl ContractCoverage {
    /// Count demands the plan declares a tensor for.
    ///
    /// An optional demand the plan skipped is dropped from *both* sides rather
    /// than only the numerator. Counting it in the denominator would reject
    /// every tied-embedding checkpoint: `lm_head.weight` is optional precisely
    /// because the binder falls back to the embedding table when HF omits it,
    /// and the plan then rightly does not produce it.
    pub fn measure<'a>(
        planned: impl IntoIterator<Item = &'a str>,
        demands: impl IntoIterator<Item = (&'a str, bool)>,
    ) -> Self {
        let planned: HashSet<&str> = planned.into_iter().collect();
        let mut out = Self::default();
        for (name, optional) in demands {
            let present = planned.contains(name);
            if !present && optional {
                continue;
            }
            out.demanded += 1;
            if present {
                out.covered += 1;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_plan::{StorageTarget, compiler_version};
    use crate::types::{BufferId, Encoding, InstrId, TensorDecl, TensorId};

    fn decl(name: &str) -> TensorDecl {
        TensorDecl {
            id: TensorId(0),
            name: name.to_string(),
            shape: vec![1],
            encoding: Encoding::Raw(crate::types::DType::U8),
            alignment: 1,
        }
    }

    fn plan_with(tensors: Vec<TensorDecl>, instrs: Vec<StorageInstr>) -> LoadPlan {
        let schedule = (0..instrs.len() as u32).map(InstrId).collect();
        LoadPlan {
            compiler_version: compiler_version(),
            tensors,
            instrs,
            schedule,
            ..LoadPlan::empty(StorageTarget::default())
        }
    }

    fn finalize(index: u32, name: &str) -> StorageInstr {
        StorageInstr::Finalize {
            id: InstrId(index),
            tensor: BufferId(index),
            name: name.to_string(),
        }
    }

    #[test]
    fn a_plan_that_finalizes_what_it_declares_verifies() {
        let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
        let certificate = verify(&PlanView::from(&plan), None).expect("should verify");
        assert_eq!(certificate.tensors, 1);
        assert_eq!(certificate.instructions, 1);
    }

    #[test]
    fn a_declared_tensor_that_is_never_finalized_is_a_violation() {
        let plan = plan_with(vec![decl("w")], Vec::new());
        let violations = verify(&PlanView::from(&plan), None).unwrap_err();
        assert_eq!(violations[0].tensor.as_deref(), Some("w"));
        assert!(violations[0].message.contains("never finalized"));
    }

    #[test]
    fn finalizing_a_name_nobody_declared_is_a_violation() {
        let plan = plan_with(Vec::new(), vec![finalize(0, "ghost")]);
        let violations = verify(&PlanView::from(&plan), None).unwrap_err();
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("never declared"))
        );
    }

    #[test]
    fn finalizing_the_same_tensor_twice_is_a_violation() {
        let plan = plan_with(vec![decl("w")], vec![finalize(0, "w"), finalize(1, "w")]);
        let violations = verify(&PlanView::from(&plan), None).unwrap_err();
        assert!(violations.iter().any(|v| v.message.contains("2 times")));
    }

    #[test]
    fn an_instruction_left_out_of_the_schedule_is_a_violation() {
        let mut plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
        plan.schedule.clear();
        let violations = verify(&PlanView::from(&plan), None).unwrap_err();
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("never scheduled"))
        );
    }

    #[test]
    fn scheduling_an_instruction_twice_is_a_violation() {
        let mut plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
        plan.schedule.push(InstrId(0));
        let violations = verify(&PlanView::from(&plan), None).unwrap_err();
        assert!(
            violations
                .iter()
                .any(|v| v.message.contains("more than once"))
        );
    }

    fn contract(tensors: &[(&'static str, Vec<i64>, Encoding)]) -> ContractView<'static> {
        // Leaked so the view's borrows outlive the call; a test process is
        // exactly where that is the cheap answer.
        ContractView {
            tensors: tensors
                .iter()
                .map(|(name, shape, encoding)| {
                    TensorDemand::exact(name, Box::leak(shape.clone().into_boxed_slice()), encoding)
                })
                .collect(),
        }
    }

    #[test]
    fn a_plan_that_delivers_the_contract_verifies() {
        let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
        let demanded = contract(&[("w", vec![1], Encoding::Raw(crate::types::DType::U8))]);
        verify(&PlanView::from(&plan), Some(&demanded)).expect("should verify");
    }

    #[test]
    fn a_contract_tensor_the_plan_omits_is_a_violation() {
        let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
        let demanded = contract(&[
            ("w", vec![1], Encoding::Raw(crate::types::DType::U8)),
            ("bias", vec![1], Encoding::Raw(crate::types::DType::U8)),
        ]);
        let violations = verify(&PlanView::from(&plan), Some(&demanded)).unwrap_err();
        assert_eq!(violations[0].tensor.as_deref(), Some("bias"));
        assert!(violations[0].message.contains("does not declare it"));
    }

    #[test]
    fn a_plan_delivering_the_wrong_shape_is_a_violation() {
        let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
        let demanded = contract(&[("w", vec![2, 3], Encoding::Raw(crate::types::DType::U8))]);
        let violations = verify(&PlanView::from(&plan), Some(&demanded)).unwrap_err();
        assert!(violations[0].message.contains("contract demands [2, 3]"));
    }

    #[test]
    fn a_plan_delivering_the_wrong_encoding_is_a_violation() {
        let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
        let demanded = contract(&[("w", vec![1], Encoding::Raw(crate::types::DType::BF16))]);
        let violations = verify(&PlanView::from(&plan), Some(&demanded)).unwrap_err();
        assert!(violations[0].message.contains("BF16"));
    }

    /// Publishing more than was demanded is not a failure: the loader
    /// republishes views (`arch/nemotron.rs`) that a given driver may not bind.
    #[test]
    fn a_plan_delivering_more_than_the_contract_demands_verifies() {
        let plan = plan_with(
            vec![decl("w"), decl("view")],
            vec![finalize(0, "w"), finalize(1, "view")],
        );
        let demanded = contract(&[("w", vec![1], Encoding::Raw(crate::types::DType::U8))]);
        verify(&PlanView::from(&plan), Some(&demanded)).expect("should verify");
    }

    #[test]
    fn a_stale_compiler_version_is_a_violation() {
        let mut plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
        plan.compiler_version ^= 1;
        let violations = verify(&PlanView::from(&plan), None).unwrap_err();
        assert!(violations[0].message.contains("compiler version"));
    }

    /// An optional demand the plan skipped must leave the coverage ratio whole.
    /// Counting it in the denominator alone would reject every tied-embedding
    /// checkpoint, where `lm_head.weight` is legitimately absent.
    #[test]
    fn an_absent_optional_demand_does_not_dent_coverage() {
        let planned = ["model.embed_tokens.weight", "model.norm.weight"];
        let demands = [
            ("model.embed_tokens.weight", false),
            ("lm_head.weight", true),
            ("model.norm.weight", false),
        ];
        let coverage = ContractCoverage::measure(planned, demands);
        assert_eq!(coverage.covered, 2);
        assert_eq!(coverage.demanded, 2);
    }

    #[test]
    fn a_present_optional_demand_counts_on_both_sides() {
        let planned = [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
        ];
        let demands = [
            ("model.embed_tokens.weight", false),
            ("lm_head.weight", true),
            ("model.norm.weight", false),
        ];
        let coverage = ContractCoverage::measure(planned, demands);
        assert_eq!(coverage.covered, 3);
        assert_eq!(coverage.demanded, 3);
    }

    /// The shortfall this exists to report.
    #[test]
    fn a_missing_required_demand_is_a_shortfall() {
        let planned = ["model.embed_tokens.weight"];
        let demands = [
            ("model.embed_tokens.weight", false),
            ("model.layers.0.mlp.down_proj.weight", false),
        ];
        let coverage = ContractCoverage::measure(planned, demands);
        assert_eq!(coverage.covered, 1);
        assert_eq!(coverage.demanded, 2);
    }

    /// A driver that has not adopted the declaration reports `0 / 0`, which the
    /// driver-side gate reads as "nothing to check" rather than as a shortfall.
    #[test]
    fn declaring_nothing_is_whole_coverage() {
        let coverage = ContractCoverage::measure(["a", "b"], []);
        assert_eq!(coverage.covered, 0);
        assert_eq!(coverage.demanded, 0);
    }
}
