use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::types::{Encoding, Visibility};

pub struct PlanView<'a> {
    pub files: Vec<FileView<'a>>,
    pub sources: Vec<SourceView<'a>>,
    pub tensors: Vec<TensorView<'a>>,
    pub instr_count: usize,
    pub schedule: Vec<u32>,
    pub finalized: Vec<&'a str>,
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
    pub encoding: Encoding,
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

pub struct ContractView<'a> {
    pub tensors: Vec<TensorDemand<'a>>,
}

pub struct TensorDemand<'a> {
    pub name: &'a str,
    pub shape: Option<Vec<i64>>,
    pub encoding: Option<Encoding>,
}

impl<'a> TensorDemand<'a> {
    pub fn authored(name: &'a str, shape: Option<&[i64]>, encoding: &Encoding) -> Self {
        Self {
            name,
            shape: shape.map(<[i64]>::to_vec),
            encoding: Some(crate::types::normalize_encoding(encoding)),
        }
    }
}

impl<'a> ContractView<'a> {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub tensor: Option<String>,
    pub message: String,
}

impl Violation {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certificate {
    pub tensors: usize,
    pub instructions: usize,
    pub files: usize,
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

fn physical_span(source: &crate::plan::SourceExtent) -> u64 {
    let mut normalized = source.stride.clone();
    let base = normalized.base_offset;
    normalized.base_offset = 0;
    match crate::executor::walk::physical_source_bytes(&normalized) {
        Ok(len) => base.saturating_add(len),
        Err(_) => source.span_bytes,
    }
}

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
}

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
