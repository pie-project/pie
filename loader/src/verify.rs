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
//! There is no driver-side counterpart. `loaded_model.cpp` used to compare a
//! covered count against a demanded one, which was `if (x != x)` twice over:
//! first because both were assigned from `view.tensors.len`, and then, once the
//! contract gave them separate origins, because `check_contract` below throws
//! on the only way they could differ. A driver that calls `verify` has already
//! been told; counting again on the far side of the FFI only looked like a
//! check.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::types::{Encoding, Visibility};

/// The plan, reduced to what verification can read.
///
/// Both callers build one: the CLI from a Rust [`LoadPlan`](crate::plan::LoadPlan), and the FFI
/// boundary from the `PieLoaderPlan` the C++ driver is holding. The FFI case is
/// the one that matters — verifying the *marshalled* view means a bug in the
/// marshalling is in scope, which it would not be if verification re-read the
/// Rust plan the driver never sees.
pub struct PlanView<'a> {
    pub compiler_version: u64,
    pub files: Vec<FileView<'a>>,
    pub sources: Vec<SourceView<'a>>,
    pub tensors: Vec<TensorView<'a>>,
    pub instr_count: usize,
    pub schedule: Vec<u32>,
    /// Names the instruction stream finalizes, in stream order. Duplicates are
    /// kept: finalizing the same name twice is one of the things being checked.
    pub finalized: Vec<&'a str>,
    /// Every instruction that reads the checkpoint, and the bytes it reads.
    ///
    /// The range is carried, not just the file, because the plan's *sources*
    /// and the plan's *reads* are different claims. A source says which bytes a
    /// tensor occupies; a read says which bytes an instruction will hand to the
    /// executor, and the compiler derives the second from the first through
    /// offsets it computes. Checking only sources leaves every derivation
    /// unchecked.
    pub reads: Vec<ReadView>,
}

pub struct ReadView {
    pub instr: u32,
    pub file_id: u32,
    pub file_offset: u64,
    pub span_bytes: u64,
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
    /// Whether the driver binds this name. An [`Visibility::Internal`]
    /// declaration is a name the contract needed for itself, so it is typed and
    /// planned like any other but never finalized.
    pub visibility: Visibility,
}

impl<'a> TensorView<'a> {
    pub fn new(name: &'a str, shape: &[i64], encoding: &Encoding, visibility: Visibility) -> Self {
        Self {
            name,
            shape: shape.to_vec(),
            encoding: crate::types::normalize_encoding(encoding),
            visibility,
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
/// Every demand is read off a contract the driver authored, so the encoding is
/// always pinned: a contract states the encoding of every tensor it defines.
/// The shape is pinned only when the contract declared one — a contract may
/// decline to predict a shape (`TensorContract::inferred`), and inventing one
/// here would turn the loader's own inference into the thing being verified.
///
/// There is no `optional`. It existed for a driver that declared the tensors it
/// would bind without authoring them, and could only guess whether a weight it
/// named would be present; a contract does not guess, because the author read
/// the checkpoint's tensor table before writing it. A tied-embedding checkpoint
/// yields a contract that does not declare `lm_head.weight` at all.
pub struct TensorDemand<'a> {
    pub name: &'a str,
    pub shape: Option<Vec<i64>>,
    pub encoding: Option<Encoding>,
}

impl<'a> TensorDemand<'a> {
    /// A demand read off a contract the loader itself is about to execute.
    pub fn authored(name: &'a str, shape: Option<&[i64]>, encoding: &Encoding) -> Self {
        Self {
            name,
            shape: shape.map(<[i64]>::to_vec),
            encoding: Some(crate::types::normalize_encoding(encoding)),
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
                    TensorDemand::authored(
                        tensor.name.as_str(),
                        tensor.shape.as_deref(),
                        &tensor.encoding,
                    )
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
    /// A violation about the plan as a whole. `pub` because the ABI crate's
    /// marshalled-view verifier reports marshalling failures through the same
    /// vocabulary.
    pub fn plan(message: impl Into<String>) -> Self {
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
    check_compiler_version(plan, &mut found);
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

/// The plan was compiled by *this* build of the loader.
///
/// A tautology on the compile path — the plan came from the same library a
/// moment ago — and kept anyway, because `compiler_version` is the field that
/// stops being one the day a plan reaches `verify` from anywhere but a
/// `compile` call in the same process. The `version` field that used to sit
/// beside it was removed: a monotonic layout number cannot say anything a
/// source hash does not, and it said it about a struct C++ reads by layout.
fn check_compiler_version(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
    let expected = crate::plan::compiler_version();
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

/// Every public declaration must be finalized, under its declared name, exactly
/// once — and nothing else may be.
///
/// A public `TensorDecl` with no `Finalize` is a weight the driver will look up
/// and not find. A `Finalize` with no `TensorDecl` is a name the driver was
/// never told to expect. Both are the plan disagreeing with itself.
///
/// The declaration table and the bind table are not the same set, which is what
/// [`Visibility`] says: an internal declaration is a name later expressions
/// resolve through, and finalizing it would put in the driver's hands the very
/// tensor the contract asked to keep.
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
        match (tensor.visibility, finalized.get(tensor.name)) {
            (Visibility::Public, None) => found.push(Violation::tensor(
                tensor.name,
                "is declared but never finalized, so the load would leave it absent",
            )),
            (Visibility::Public, Some(1)) | (Visibility::Internal, None) => {}
            (Visibility::Internal, Some(_)) => found.push(Violation::tensor(
                tensor.name,
                "is internal but finalized, so the driver would bind a name the \
                 contract asked to keep to itself",
            )),
            (Visibility::Public, Some(count)) => found.push(Violation::tensor(
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
        let Some(file) = plan.files.get(read.file_id as usize) else {
            found.push(Violation::plan(format!(
                "instruction {} reads from file {}, but the plan declares {} files",
                read.instr,
                read.file_id,
                plan.files.len()
            )));
            continue;
        };
        let end = read.file_offset.saturating_add(read.span_bytes);
        if end > file.size_bytes {
            found.push(Violation::plan(format!(
                "instruction {} reads bytes [{}, {end}) of {}, which is {} bytes long",
                read.instr, read.file_offset, file.path, file.size_bytes
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
/// This is `architecture.md` §8.2's coverage question in full: not "is the plan
/// self-consistent" but "does it produce what will be asked for, in the form it
/// will be asked for?". Shape and encoding are checked as well as presence,
/// because a tensor that exists at the wrong shape fails later, further away,
/// and as garbage output rather than a missing symbol.
fn check_contract(plan: &PlanView<'_>, contract: &ContractView<'_>, found: &mut Vec<Violation>) {
    let planned: HashMap<&str, &TensorView<'_>> = plan
        .tensors
        .iter()
        .map(|tensor| (tensor.name, tensor))
        .collect();

    for demanded in &contract.tensors {
        let Some(planned) = planned.get(demanded.name) else {
            found.push(Violation::tensor(
                demanded.name,
                "is demanded by the contract but the plan does not declare it",
            ));
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
            && !encoding_matches(&planned.encoding, encoding)
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

    // The converse is not a violation by symmetry: a contract may publish views
    // alongside the buffer they borrow — packed MoE experts republished under
    // their original names, as `tests/storage_compiler.rs` pins for Nemotron-H —
    // and a given driver may never bind them. Producing more than was demanded
    // costs a name, not correctness. Producing less is the failure.
}

/// Compare two encodings over the fields a plan can actually carry.
///
/// A [`PlanView`] can arrive two ways: from a plan the loader still holds in
/// typed form, or rebuilt from the POD arena the driver received. The POD
/// `TensorDecl` carries `scheme`, `dtype`, `bits_per_element` and `group_size`
/// and nothing else, because the rest of [`QuantSpec`] had no reader on the far
/// side — the scale tensor's axis, dtype and granularity are stated on the
/// `QuantAttachment`, which is where the executor looks for them.
///
/// So verification compares what crossed the boundary. Holding a plan to a
/// field that cannot be expressed in it would make the check fail for a
/// contract that is in fact satisfied, and comparing the two representations
/// asymmetrically would make the answer depend on which side of the FFI the
/// caller stood on.
fn encoding_matches(planned: &Encoding, demanded: &Encoding) -> bool {
    match (planned, demanded) {
        (Encoding::Quant(planned), Encoding::Quant(demanded)) => {
            planned.scheme == demanded.scheme
                && planned.logical_dtype == demanded.logical_dtype
                && planned.bits_per_element == demanded.bits_per_element
                && planned.group_size == demanded.group_size
        }
        _ => planned == demanded,
    }
}

// The verifier's tests live in `capi/tests/verify_marshalled.rs`: every one
// checks the verdict through the marshalled view, which is the ABI crate's.
