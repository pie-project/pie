//! Does this plan honour its contract?
//!
//! Verification is deliberately not a second compiler: it takes the plan
//! exactly as the engine sees it and asks questions answerable from the plan
//! and the filesystem alone, sharing no code with the compiler that built it.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::types::{Encoding, Visibility};

/// The plan, reduced to what verification can read.
///
/// A view rather than the [`LoadPlan`](crate::plan::LoadPlan) itself, so
/// verification can read a plan it did not build — `view_of` makes one from a
/// compiled plan, and a stored dump reduces to the same struct.
pub struct PlanView<'a> {
    pub files: Vec<FileView<'a>>,
    pub sources: Vec<SourceView<'a>>,
    pub tensors: Vec<TensorView<'a>>,
    pub instr_count: usize,
    pub schedule: Vec<u32>,
    /// Names the instruction stream finalizes, in stream order. Duplicates are
    /// kept: finalizing the same name twice is one of the things being checked.
    pub finalized: Vec<&'a str>,
    /// Every instruction that reads the checkpoint, and the bytes it reads.
    /// Carried separately from `sources`: a source says which bytes a tensor
    /// occupies, a read says which bytes an instruction hands the executor,
    /// and the compiler derives the second from the first.
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
    /// Whether the engine binds this name. An [`Visibility::Internal`]
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

/// What the runtime demands, independently of how the plan proposes to
/// deliver it: a name, a shape, and an encoding per tensor. Derived by
/// different code than the plan, so the two disagreeing is a real signal.
pub struct ContractView<'a> {
    pub tensors: Vec<TensorDemand<'a>>,
}

/// One tensor the runtime demands.
///
/// The encoding is always pinned — a contract states the encoding of every
/// tensor it defines. The shape is pinned only when the contract declared
/// one; a sharded tensor's shape is also left unpinned, since a plan holds
/// only this rank's band and that is checked elsewhere, against the same
/// declaration.
///
/// There is no `optional`: a contract does not guess whether a weight is
/// present, since the author read the checkpoint's tensor table before
/// writing it.
///
/// [`Expr::Shard`]: crate::contract::Expr::Shard
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
    /// Read a contract as the rank that will execute it sees it: every tensor
    /// it declares, and of each declaration the part that is about the rank.
    pub fn of(contract: &'a crate::contract::ModelContract) -> Self {
        Self {
            tensors: contract
                .tensors
                .iter()
                .map(|tensor| {
                    let shape = if tensor.expr.is_sharded() {
                        None
                    } else {
                        tensor.shape.as_deref()
                    };
                    TensorDemand::authored(tensor.name.as_str(), shape, &tensor.encoding)
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
    /// A violation about the plan as a whole.
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
/// types an engine will look up after the load. Everything below asks whether
/// the instruction stream actually delivers them.
pub fn verify(
    plan: &PlanView<'_>,
    contract: Option<&ContractView<'_>>,
) -> Result<Certificate, Vec<Violation>> {
    let mut found = Vec::new();
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

/// Every public declaration must be finalized, under its declared name,
/// exactly once — and nothing else may be. An internal declaration is a name
/// later expressions resolve through; finalizing it would hand the engine a
/// tensor the contract asked to keep internal.
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
                "is declared more than once, so an engine's lookup is ambiguous",
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
                "is internal but finalized, so the engine would bind a name the \
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

/// Check the plan's file table against the filesystem: is a cached plan
/// still about the checkpoint it was compiled from? Size is a weak but cheap
/// proxy for content, and it catches a re-download producing a different
/// checkpoint under the same path.
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

/// The bytes an instruction touches past `file_offset`: the strided
/// footprint the executor reads (`read_extent`), not the logical count it
/// produces. The two part on a broadcast source — a stride-0 dimension
/// reads one stretch and writes it several times — and the file-bounds
/// check is about what is read.
fn physical_span(source: &crate::plan::SourceExtent) -> u64 {
    let mut normalized = source.stride.clone();
    let base = normalized.base_offset;
    normalized.base_offset = 0;
    match crate::executor::walk::physical_source_bytes(&normalized) {
        Ok(len) => base.saturating_add(len),
        Err(_) => source.span_bytes,
    }
}

/// Check the plan against what the runtime actually demands: not "is the plan
/// self-consistent" but "does it produce what will be asked for?". Shape and
/// encoding are checked as well as presence.
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

    // The converse is not a violation: a plan may declare tensors a given
    // engine never binds. Producing more than demanded costs a name, not
    // correctness; producing less is the failure.
}

/// Compare two encodings over the fields a plan can actually carry: `scheme`,
/// `dtype`, `bits_per_element`, `group_size`. Holding a plan to a field it
/// cannot express would fail a contract that is in fact satisfied.
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

/// The verification view of a compiled plan, read straight off it.
#[must_use]
pub fn view_of(plan: &crate::plan::LoadPlan) -> PlanView<'_> {
    use crate::plan::StorageInstr;

    let files = plan
        .files
        .iter()
        .map(|f| FileView {
            id: f.id.0,
            path: &f.path,
            size_bytes: f.size_bytes,
        })
        .collect();
    let sources = plan
        .sources
        .iter()
        .map(|s| SourceView {
            name: &s.name,
            file_id: s.file_id.0,
            offset_bytes: s.file_offset,
            span_bytes: s.span_bytes,
        })
        .collect();
    let tensors = plan
        .tensors
        .iter()
        .map(|t| TensorView::new(&t.name, &t.shape, &t.encoding, t.visibility))
        .collect();

    let mut finalized = Vec::new();
    let mut reads = Vec::new();
    for instr in &plan.instrs {
        // Every arm named rather than a wildcard: an instruction that grows a
        // source and is not added here would make the plan look unread.
        match instr {
            StorageInstr::Finalize { name, .. } => finalized.push(name.as_str()),
            StorageInstr::ExtentWrite { id, source, .. }
            | StorageInstr::BulkExtentWrite { id, source, .. }
            | StorageInstr::GatherWrite { id, source, .. } => reads.push(ReadView {
                instr: id.0,
                file_id: source.file_id.0,
                file_offset: source.file_offset,
                span_bytes: physical_span(source),
            }),
            StorageInstr::TileMap { id, source, .. } => {
                if let Some(source) = source {
                    reads.push(ReadView {
                        instr: id.0,
                        file_id: source.file_id.0,
                        file_offset: source.file_offset,
                        span_bytes: physical_span(source),
                    });
                }
            }
            StorageInstr::Allocate { .. }
            | StorageInstr::CreateView { .. }
            | StorageInstr::Fill { .. } => {}
        }
    }

    PlanView {
        files,
        sources,
        tensors,
        instr_count: plan.instrs.len(),
        schedule: plan.schedule.iter().map(|i| i.0).collect(),
        finalized,
        reads,
    }
}

/// Verify a plan AND every instance of every group it carries.
///
/// A group's plan is compiled at index 0, so verifying it alone checks one
/// instance out of `arity` and leaves the other bindings unchecked. Rewriting
/// the template's reads with each instance's binding and rerunning the
/// file-bounds check is the cheapest way to check every instance.
///
/// # Errors
///
/// Every violation found, in plan-then-group order.
pub fn verify_plan(
    plan: &crate::plan::LoadPlan,
    contract: Option<&ContractView<'_>>,
) -> Result<Certificate, Vec<Violation>> {
    let certificate = verify(&view_of(plan), contract)?;
    let mut found = Vec::new();
    for group in &plan.groups {
        verify_group(group, &mut found);
    }
    if found.is_empty() {
        Ok(certificate)
    } else {
        Err(found)
    }
}

fn verify_group(group: &crate::plan::GroupPlan, found: &mut Vec<Violation>) {
    let name = &group.name;
    let mut template = view_of(&group.plan);
    if let Err(mut violations) = verify(&template, None) {
        for violation in &mut violations {
            violation.message = format!("group '{name}': {}", violation.message);
        }
        found.append(&mut violations);
        return;
    }
    let per = template.reads.len();
    if group.bindings.len() != group.arity as usize || group.bindings.iter().any(|b| b.len() != per)
    {
        found.push(Violation::plan(format!(
            "group '{name}': {} binding sets for {} instances of a plan with {per} reads",
            group.bindings.len(),
            group.arity
        )));
        return;
    }

    // Mutated in place: every read's file and offset is overwritten on each
    // pass, so there is nothing to restore between them.
    for (index, bindings) in group.bindings.iter().enumerate() {
        for (read, binding) in template.reads.iter_mut().zip(bindings) {
            if read.instr != binding.instr.0 {
                found.push(Violation::plan(format!(
                    "group '{name}' index {index}: binding names instruction {} where \
                     the plan reads at instruction {}",
                    binding.instr.0, read.instr
                )));
                return;
            }
            read.file_id = binding.file_id.0;
            read.file_offset = binding.file_offset;
        }
        if let Err(violations) = verify(&template, None) {
            for violation in violations {
                found.push(Violation::plan(format!(
                    "group '{name}' index {index}: {}",
                    violation.message
                )));
            }
            return;
        }
    }
}
