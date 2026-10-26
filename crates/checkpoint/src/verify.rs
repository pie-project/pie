use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::plan::TileMapKind;
use crate::types::{Encoding, Visibility};

#[derive(Default)]
pub struct PlanView<'a> {
    pub files: Vec<FileView<'a>>,
    pub sources: Vec<SourceView<'a>>,
    pub tensors: Vec<TensorView<'a>>,
    pub instr_count: usize,
    pub schedule: Vec<u32>,
    pub finalized: Vec<&'a str>,
    pub reads: Vec<ReadView>,
    pub target_alignment: u32,
    pub buffers: Vec<BufferView>,
    pub finalized_buffers: Vec<(&'a str, u32)>,
    pub writes: Vec<WriteView>,
    pub fills: Vec<FillView>,
    pub views: Vec<SliceView>,
    pub tile_maps: Vec<TileMapView<'a>>,
}

pub struct ReadView {
    pub instr: u32,
    pub file_id: u32,
    pub file_offset: u64,
    pub span_bytes: u64,
}

impl ReadView {
    fn of(instr: u32, source: &crate::plan::SourceExtent) -> Self {
        Self {
            instr,
            file_id: source.file_id.0,
            file_offset: source.file_offset,
            span_bytes: physical_span(source),
        }
    }
}

pub struct BufferView {
    pub id: u32,
    pub tensor: Option<u32>,
    pub bytes: u64,
    pub alignment: u32,
    pub persistent_offset: Option<u64>,
    pub scratch_offset: Option<u64>,
}

pub struct WriteView {
    pub instr: u32,
    pub step: Option<usize>,
    pub buffer: u32,
    pub start: u64,
    pub end: u64,
}

impl WriteView {
    fn new(instr: u32, step: Option<usize>, buffer: u32, start: u64, end: u64) -> Self {
        Self { instr, step, buffer, start, end }
    }
}

pub struct FillView {
    pub instr: u32,
    pub step: Option<usize>,
    pub buffer: u32,
}

pub struct SliceView {
    pub instr: u32,
    pub input: u32,
    pub output: u32,
    pub offset: u64,
    pub extent_bytes: u64,
}

pub struct TileMapView<'a> {
    pub instr: u32,
    pub kind: TileMapKind,
    pub has_source: bool,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub dest_buffer: Option<u32>,
    pub scale_blocks: Vec<i64>,
    pub kernel: Option<&'a str>,
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
    check_fill_order(plan, &mut found);
    check_coverage(plan, &mut found);
    check_byte_partition(plan, &mut found);
    check_arena_layout(plan, &mut found);
    check_scales(plan, &mut found);
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

fn check_fill_order(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
    if plan.fills.is_empty() {
        return;
    }
    let mut filled: HashMap<u32, usize> = HashMap::new();
    for fill in &plan.fills {
        if let Some(step) = fill.step {
            let entry = filled.entry(fill.buffer).or_insert(step);
            *entry = (*entry).max(step);
        }
    }
    for write in &plan.writes {
        if let (Some(write_step), Some(&fill_step)) = (write.step, filled.get(&write.buffer))
            && fill_step > write_step
        {
            found.push(Violation::plan(format!(
                "buffer {} is written at step {write_step} (instruction {}) but not zeroed until step {fill_step}",
                write.buffer, write.instr
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

const MAX_VIEW_HOPS: usize = 16;

fn check_byte_partition(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
    if plan.buffers.is_empty() && plan.finalized_buffers.is_empty() {
        return;
    }
    let buffer_by_id: HashMap<u32, &BufferView> =
        plan.buffers.iter().map(|b| (b.id, b)).collect();
    let view_by_output: HashMap<u32, &SliceView> =
        plan.views.iter().map(|v| (v.output, v)).collect();
    let filled_buffers: HashSet<u32> = plan.fills.iter().map(|f| f.buffer).collect();

    let mut writes_by_buffer: HashMap<u32, Vec<(u64, u64, u32)>> = HashMap::new();
    for write in &plan.writes {
        if let Some(buf) = buffer_by_id.get(&write.buffer)
            && write.end > buf.bytes
        {
            found.push(Violation::plan(format!(
                "instruction {} writes bytes [{}, {}) of buffer {}, which is {} bytes long",
                write.instr, write.start, write.end, write.buffer, buf.bytes
            )));
        }
        if write.end > write.start {
            writes_by_buffer
                .entry(write.buffer)
                .or_default()
                .push((write.start, write.end, write.instr));
        }
    }

    for &(name, tensor_buf) in &plan.finalized_buffers {
        let mut root_buf = tensor_buf;
        let mut window_offset = 0_u64;
        let mut view_extent: Option<u64> = None;
        for _ in 0..MAX_VIEW_HOPS {
            let Some(view) = view_by_output.get(&root_buf) else {
                break;
            };
            if view.input == root_buf {
                break;
            }
            window_offset = window_offset.saturating_add(view.offset);
            if view_extent.is_none() {
                view_extent = Some(view.extent_bytes);
            }
            root_buf = view.input;
        }
        let Some(backing) = buffer_by_id.get(&root_buf) else {
            found.push(Violation::tensor(
                name,
                format!("finalizes undeclared buffer {tensor_buf}"),
            ));
            continue;
        };
        let window_len = view_extent.unwrap_or(backing.bytes);
        let window_start = window_offset;
        let window_end = window_start.saturating_add(window_len);
        if window_end <= window_start || window_end > backing.bytes {
            continue;
        }

        let mut intervals: Vec<(u64, u64)> = writes_by_buffer
            .get(&root_buf)
            .map(|list| {
                list.iter()
                    .filter(|&&(s, e, _)| s < window_end && e > window_start)
                    .map(|&(s, e, _)| (s.max(window_start), e.min(window_end)))
                    .collect()
            })
            .unwrap_or_default();
        intervals.sort_unstable();

        for pair in intervals.windows(2) {
            if pair[0].1 > pair[1].0 {
                found.push(Violation::tensor(
                    name,
                    format!(
                        "overlapping write intervals [{}, {}) and [{}, {}) in buffer {root_buf}",
                        pair[0].0, pair[0].1, pair[1].0, pair[1].1
                    ),
                ));
                break;
            }
        }

        if !filled_buffers.contains(&root_buf) {
            let mut cursor = window_start;
            let mut gap_reported = false;
            for &(s, e) in &intervals {
                if s > cursor {
                    found.push(Violation::tensor(
                        name,
                        format!(
                            "unwritten byte hole [{cursor}, {s}) in buffer {root_buf} ([{window_start}, {window_end}))"
                        ),
                    ));
                    gap_reported = true;
                    break;
                }
                cursor = cursor.max(e);
            }
            if !gap_reported && cursor < window_end {
                found.push(Violation::tensor(
                    name,
                    format!(
                        "unwritten byte hole [{cursor}, {window_end}) in buffer {root_buf} ([{window_start}, {window_end}))"
                    ),
                ));
            }
        }
    }
}

fn check_arena_layout(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
    let buffer_by_id: HashMap<u32, &BufferView> =
        plan.buffers.iter().map(|b| (b.id, b)).collect();
    let view_by_output: HashMap<u32, &SliceView> =
        plan.views.iter().map(|v| (v.output, v)).collect();

    let mut claimed: HashMap<u32, u32> = HashMap::new();
    let mut spans: Vec<(u64, u64, u32)> = Vec::new();
    for buffer in &plan.buffers {
        if let Some(tensor) = buffer.tensor
            && let Some(first) = claimed.insert(tensor, buffer.id)
        {
            found.push(Violation::plan(format!(
                "buffers {first} and {} both claim tensor {tensor}",
                buffer.id
            )));
        }
        let Some(offset) = buffer.persistent_offset else {
            continue;
        };
        let alignment = u64::from(buffer.alignment.max(plan.target_alignment).max(1));
        if offset % alignment != 0 {
            found.push(Violation::plan(format!(
                "persistent buffer {} base offset {offset} violates operand alignment {alignment}",
                buffer.id
            )));
        }
        let end = offset.saturating_add(buffer.bytes);
        spans.push((offset, end, buffer.id));
    }
    spans.sort_unstable_by_key(|span| span.0);
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            found.push(Violation::plan(format!(
                "persistent buffers {} and {} overlap in the arena: [{}, {}) vs [{}, {})",
                pair[0].2, pair[1].2, pair[0].0, pair[0].1, pair[1].0, pair[1].1
            )));
        }
    }

    for view in &plan.views {
        let Some(backing) = buffer_by_id.get(&view.input) else {
            found.push(Violation::plan(format!(
                "CreateView instruction {} references missing backing buffer {}",
                view.instr, view.input
            )));
            continue;
        };
        let end = view.offset.saturating_add(view.extent_bytes);
        if end > backing.bytes {
            found.push(Violation::plan(format!(
                "CreateView window [{}, {end}) escapes backing buffer {} ({} bytes)",
                view.offset, backing.id, backing.bytes
            )));
        }
    }

    for tm in &plan.tile_maps {
        let Some(kernel) = tm.kernel else {
            continue;
        };
        if tm.has_source {
            found.push(Violation::plan(format!(
                "{:?} names kernel `{kernel}` but reads the checkpoint directly without a staged buffer",
                tm.kind
            )));
        }
        for &operand in tm
            .inputs
            .iter()
            .chain(&tm.outputs)
            .chain(tm.dest_buffer.iter())
        {
            if !in_arena_view(operand, &buffer_by_id, &view_by_output) {
                found.push(Violation::plan(format!(
                    "{:?} names kernel `{kernel}` but operand buffer {operand} has no arena offset",
                    tm.kind
                )));
            }
        }
    }
}

fn in_arena_view(
    mut id: u32,
    buffer_by_id: &HashMap<u32, &BufferView>,
    view_by_output: &HashMap<u32, &SliceView>,
) -> bool {
    for _ in 0..MAX_VIEW_HOPS {
        let Some(decl) = buffer_by_id.get(&id) else {
            return false;
        };
        if decl.persistent_offset.is_some() || decl.scratch_offset.is_some() {
            return true;
        }
        let Some(view) = view_by_output.get(&id) else {
            return false;
        };
        if view.input == id {
            return false;
        }
        id = view.input;
    }
    false
}

fn check_scales(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
    for tm in &plan.tile_maps {
        if tm.kind != TileMapKind::Scale || tm.scale_blocks.is_empty() {
            continue;
        }
        let wanted = 1 + usize::from(!tm.has_source);
        if tm.inputs.len() != wanted {
            found.push(Violation::plan(format!(
                "per-group Scale (instruction {}) has {} input operands, expected {wanted} (payload then factors)",
                tm.instr,
                tm.inputs.len()
            )));
        }
        for &block in &tm.scale_blocks {
            if block <= 0 {
                found.push(Violation::plan(format!(
                    "per-group Scale (instruction {}) has non-positive block size {block}",
                    tm.instr
                )));
            }
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
        let on_disk = match ztensor::memfs::len(file.path) {
            Some(mounted) => Ok(mounted),
            None => std::fs::metadata(file.path).map(|meta| meta.len()),
        };
        match on_disk {
            Ok(len) if len != file.size_bytes => found.push(Violation::plan(format!(
                "{} is {len} bytes; the plan was compiled against {} bytes",
                file.path, file.size_bytes
            ))),
            Ok(_) => {}
            Err(err) => found.push(Violation::plan(format!(
                "{} is unreadable: {err}",
                file.path
            ))),
        }
    }

    check_reads(plan, found);

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

fn check_reads(plan: &PlanView<'_>, found: &mut Vec<Violation>) {
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

fn extent_bytes(extent: &crate::extent::Extent) -> u64 {
    extent
        .dims
        .iter()
        .try_fold(u64::from(extent.element_bytes), |acc, dim| {
            u64::try_from(dim.count).ok()?.checked_mul(acc)
        })
        .unwrap_or(0)
}

fn dest_span(dest: &crate::plan::DestExtent) -> (u64, u64) {
    let start = dest.offset.saturating_add(dest.stride.base_offset);
    let len = extent_bytes(&dest.stride);
    (start, start.saturating_add(len))
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
    let buffers: Vec<BufferView> = plan
        .buffers
        .iter()
        .map(|b| BufferView {
            id: b.id.0,
            tensor: b.tensor.map(|t| t.0),
            bytes: b.bytes,
            alignment: b.alignment,
            persistent_offset: b.persistent_offset,
            scratch_offset: b.scratch_offset,
        })
        .collect();

    let buf_bytes: HashMap<u32, u64> = buffers.iter().map(|b| (b.id, b.bytes)).collect();
    let mut persistent_windows: Vec<(u64, u64, u32)> = buffers
        .iter()
        .filter_map(|b| {
            let start = b.persistent_offset?;
            (b.bytes > 0).then_some((start, start.saturating_add(b.bytes), b.id))
        })
        .collect();
    persistent_windows.sort_unstable_by_key(|w| w.0);
    let windows_disjoint = persistent_windows
        .windows(2)
        .all(|pair| pair[0].1 <= pair[1].0);

    let mut step_of: HashMap<u32, usize> = HashMap::with_capacity(plan.schedule.len());
    for (step, instr_id) in plan.schedule.iter().enumerate() {
        step_of.entry(instr_id.0).or_insert(step);
    }

    let mut finalized = Vec::new();
    let mut finalized_buffers = Vec::new();
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    let mut fills = Vec::new();
    let mut views = Vec::new();
    let mut tile_maps = Vec::new();

    for instr in &plan.instrs {
        match instr {
            StorageInstr::Finalize { id: _, tensor, name } => {
                finalized.push(name.as_str());
                finalized_buffers.push((name.as_str(), tensor.0));
            }
            StorageInstr::ExtentWrite { id, source, dest }
            | StorageInstr::GatherWrite { id, source, dest, .. } => {
                reads.push(ReadView::of(id.0, source));
                let (start, end) = dest_span(dest);
                writes.push(WriteView::new(
                    id.0,
                    step_of.get(&id.0).copied(),
                    dest.buffer.0,
                    start,
                    end,
                ));
            }
            StorageInstr::BulkExtentWrite { id, source, dest_offset } => {
                reads.push(ReadView::of(id.0, source));
                let w_start = *dest_offset;
                let w_end = w_start.saturating_add(source.span_bytes);
                let step = step_of.get(&id.0).copied();
                let first = if windows_disjoint {
                    persistent_windows.partition_point(|w| w.1 <= w_start)
                } else {
                    0
                };
                for &(p_start, p_end, buf_id) in &persistent_windows[first..] {
                    if windows_disjoint && p_start >= w_end {
                        break;
                    }
                    if w_start < p_end && p_start < w_end {
                        writes.push(WriteView::new(
                            id.0,
                            step,
                            buf_id,
                            w_start.max(p_start) - p_start,
                            w_end.min(p_end) - p_start,
                        ));
                    }
                }
            }
            StorageInstr::TileMap {
                id,
                kind,
                source,
                dest,
                inputs,
                outputs,
                transform,
                ..
            } => {
                if let Some(source) = source {
                    reads.push(ReadView::of(id.0, source));
                }
                let step = step_of.get(&id.0).copied();
                if let Some(dest) = dest {
                    let (start, end) = dest_span(dest);
                    writes.push(WriteView::new(id.0, step, dest.buffer.0, start, end));
                }
                for out_buf in outputs {
                    if dest.as_ref().is_none_or(|d| d.buffer != *out_buf) {
                        let len = buf_bytes.get(&out_buf.0).copied().unwrap_or(0);
                        writes.push(WriteView::new(id.0, step, out_buf.0, 0, len));
                    }
                }
                tile_maps.push(TileMapView {
                    instr: id.0,
                    kind: *kind,
                    has_source: source.is_some(),
                    inputs: inputs.iter().map(|b| b.0).collect(),
                    outputs: outputs.iter().map(|b| b.0).collect(),
                    dest_buffer: dest.as_ref().map(|d| d.buffer.0),
                    scale_blocks: transform.scale_blocks.clone(),
                    kernel: transform.kernel.as_deref(),
                });
            }
            StorageInstr::Fill { id, buffer } => {
                fills.push(FillView {
                    instr: id.0,
                    step: step_of.get(&id.0).copied(),
                    buffer: buffer.0,
                });
            }
            StorageInstr::CreateView {
                id,
                input,
                output,
                view,
            } => {
                let offset = view.offset.saturating_add(view.stride.base_offset);
                let extent_bytes = extent_bytes(&view.stride);
                views.push(SliceView {
                    instr: id.0,
                    input: input.0,
                    output: output.0,
                    offset,
                    extent_bytes,
                });
            }
            StorageInstr::Allocate { .. } => {}
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
        target_alignment: plan.target.preferred_alignment,
        buffers,
        finalized_buffers,
        writes,
        fills,
        views,
        tile_maps,
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
        let mut instance_violations = Vec::new();
        check_reads(&template, &mut instance_violations);
        if !instance_violations.is_empty() {
            for violation in instance_violations {
                found.push(Violation::plan(format!(
                    "group '{name}' index {index}: {}",
                    violation.message
                )));
            }
            return;
        }
    }
}
