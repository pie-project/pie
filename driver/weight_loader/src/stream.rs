//! Deferred expert-load plans for SSD streaming.
//!
//! When `StorageTarget.stream_routed_experts` is set, routed-expert tensors are
//! excluded from the resident `schedule` and instead described by a
//! [`StreamPlan`]: one reusable instruction template (slot-relative
//! `ExtentWrite`s) plus per-(layer, expert) source bindings. At decode time the
//! driver executes the template into a cache slot — the same IR as boot loads,
//! with a different destination binding.
//!
//! Arch-specific naming (which tensors stream, section order, binding grid)
//! lives in [`crate::stream_arch`] plugins, registered per model on `ArchProfile`.

use crate::error::CompileError;
use crate::source::{CheckpointMetadata, RawTensor};
use crate::storage::{
    DestExtent, DimSpec, SourceExtent, StorageInstr, StorageProgram, StreamBinding, StreamPlan,
    StridedExtent,
};
use crate::types::{BufferId, InstrId, TensorId};

/// Arch plugin for building a [`StreamPlan`].
///
/// Each streaming-capable model registers one of these on its `ArchProfile`.
/// The generic builder never hardcodes section names or tensor-name patterns.
///
/// [`Self::collect_bindings`] may slice fused `[E, …]` banks into per-expert
/// extents (GPT-OSS) or map one checkpoint tensor per cell (DeepSeek-V4).
#[derive(Clone, Copy, Debug)]
pub struct StreamArchDesc {
    /// Ordered section labels; length = `sections_per_expert`.
    pub sections: &'static [&'static str],
    /// Which checkpoint tensors are streamed (and skipped from the resident ABI).
    pub is_streamed: fn(&str) -> bool,
    /// Build flat bindings `[L * E * sections.len()]` in layer-major, then
    /// expert, then section order.
    pub collect_bindings: fn(
        &CheckpointMetadata,
        num_layers: u32,
        num_experts: u32,
    ) -> Result<Vec<StreamBinding>, CompileError>,
}

const SECTION_ALIGN: u64 = 256;

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

fn compact_stride(span_bytes: u64) -> StridedExtent {
    StridedExtent {
        base_offset: 0,
        element_bytes: 1,
        dims: vec![DimSpec {
            count: span_bytes as i64,
            src_stride: 1,
            dst_stride: 1,
        }],
    }
}

/// Helper for arches where each grid cell is a distinct checkpoint tensor.
///
/// `is_streamed` / `parse` identify tensors; each matching tensor fills exactly
/// one `(layer, expert, section)` cell. Used by DeepSeek-V4 and unit tests.
pub fn collect_bindings_from_named_tensors(
    metadata: &CheckpointMetadata,
    num_layers: u32,
    num_experts: u32,
    sections_len: usize,
    is_streamed: fn(&str) -> bool,
    parse: fn(&str) -> Option<(u32, u32, usize)>,
) -> Result<Vec<StreamBinding>, CompileError> {
    let expected = (num_layers as usize) * (num_experts as usize) * sections_len;
    let mut grid: Vec<Option<&RawTensor>> = vec![None; expected];
    let mut found = 0usize;
    for raw in &metadata.tensors {
        if !is_streamed(&raw.name) {
            continue;
        }
        let Some((layer, expert, section)) = parse(&raw.name) else {
            return Err(CompileError::InvalidInput(format!(
                "stream_routed_experts: cannot parse expert tensor name '{}'",
                raw.name
            )));
        };
        if layer >= num_layers || expert >= num_experts {
            return Err(CompileError::InvalidInput(format!(
                "stream_routed_experts: tensor '{}' is outside the \
                 {num_layers}x{num_experts} expert grid",
                raw.name
            )));
        }
        if section >= sections_len {
            return Err(CompileError::InvalidInput(format!(
                "stream_routed_experts: tensor '{}' has section index {section} \
                 but arch defines only {sections_len} sections",
                raw.name
            )));
        }
        let idx = ((layer as usize) * (num_experts as usize) + (expert as usize)) * sections_len
            + section;
        if grid[idx].is_some() {
            return Err(CompileError::InvalidInput(format!(
                "stream_routed_experts: duplicate expert tensor '{}'",
                raw.name
            )));
        }
        grid[idx] = Some(raw);
        found += 1;
    }
    if found != expected {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: expected {expected} routed-expert tensors \
             (layers×experts×sections), found {found}"
        )));
    }
    Ok(grid
        .into_iter()
        .map(|slot| {
            let raw = slot.expect("grid filled");
            StreamBinding {
                file_id: raw.file_id,
                file_offset: raw.file_offset,
                span_bytes: raw.span_bytes,
            }
        })
        .collect())
}

/// Attach a deferred expert-load plan to `program`. Appends template
/// `ExtentWrite` instructions (not on `schedule`) and fills `program.stream`.
pub fn attach_stream_plan(
    program: &mut StorageProgram,
    metadata: &CheckpointMetadata,
    num_hidden_layers: u32,
    num_experts: u32,
    arch: StreamArchDesc,
) -> Result<(), CompileError> {
    if num_hidden_layers == 0 || num_experts == 0 {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: invalid expert grid {num_hidden_layers}x{num_experts}"
        )));
    }
    if arch.sections.is_empty() {
        return Err(CompileError::InvalidInput(
            "stream_routed_experts: StreamArchDesc has zero sections".to_string(),
        ));
    }
    let num_layers = num_hidden_layers;
    let num_experts_u = num_experts;
    let sections = arch.sections.len();
    let expected = (num_layers as usize) * (num_experts_u as usize) * sections;

    let bindings = (arch.collect_bindings)(metadata, num_layers, num_experts_u)?;
    if bindings.len() != expected {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: collect_bindings returned {} entries, \
             expected {expected} (layers×experts×sections)",
            bindings.len()
        )));
    }

    // Uniform section sizes from expert (0, 0).
    let mut section_bytes = vec![0u64; sections];
    for s in 0..sections {
        section_bytes[s] = bindings[s].span_bytes;
    }
    for layer in 0..num_layers as usize {
        for expert in 0..num_experts_u as usize {
            for s in 0..sections {
                let idx = (layer * (num_experts_u as usize) + expert) * sections + s;
                let span = bindings[idx].span_bytes;
                if span != section_bytes[s] {
                    return Err(CompileError::InvalidInput(format!(
                        "stream_routed_experts: non-uniform section size at \
                         layer={layer} expert={expert} section={s} \
                         ({} vs {} bytes)",
                        span, section_bytes[s]
                    )));
                }
            }
        }
    }

    let mut section_offsets = vec![0u64; sections];
    let mut offset = 0u64;
    for s in 0..sections {
        section_offsets[s] = offset;
        offset = align_up(offset + section_bytes[s], SECTION_ALIGN);
    }
    let slot_bytes = offset;

    // File path table indexed by FileId.
    let max_file_id = metadata
        .files
        .iter()
        .map(|f| f.id.0)
        .max()
        .unwrap_or(0);
    let mut files = vec![String::new(); (max_file_id as usize) + 1];
    for f in &metadata.files {
        files[f.id.0 as usize] = f.path.clone();
    }

    // Template: one ExtentWrite per section, dest offset slot-relative.
    // Prototype sources come from expert (0,0); execute time overrides from
    // bindings. BufferId(u32::MAX) marks "cache slot base" for the executor.
    let mut template = Vec::with_capacity(sections);
    let mut next_instr = program
        .instrs
        .iter()
        .map(|instr| match instr {
            StorageInstr::Allocate { id, .. }
            | StorageInstr::ExtentWrite { id, .. }
            | StorageInstr::BulkExtentWrite { id, .. }
            | StorageInstr::SlabScatter { id, .. }
            | StorageInstr::TileMap { id, .. }
            | StorageInstr::CreateView { id, .. }
            | StorageInstr::Attach { id, .. }
            | StorageInstr::Release { id, .. }
            | StorageInstr::Finalize { id, .. } => id.0,
        })
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    for s in 0..sections {
        let proto = &bindings[s];
        let id = InstrId(next_instr);
        next_instr += 1;
        let span = section_bytes[s];
        let stride = compact_stride(span);
        program.instrs.push(StorageInstr::ExtentWrite {
            id,
            source: SourceExtent {
                file_id: proto.file_id,
                tensor_id: TensorId(u32::MAX),
                file_offset: proto.file_offset,
                span_bytes: span,
                stride: stride.clone(),
            },
            dest: DestExtent {
                buffer: BufferId(u32::MAX),
                offset: section_offsets[s],
                stride,
            },
        });
        template.push(id);
    }

    program.stream = StreamPlan {
        template,
        files,
        num_layers,
        num_experts: num_experts_u,
        sections_per_expert: sections as u32,
        bindings,
        slot_bytes,
        section_offsets,
        section_bytes,
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{CheckpointFile, CheckpointMetadata, RawTensor};
    use crate::storage::{StorageProgram, StorageTarget};
    use crate::types::{CheckpointFormat, DType, Encoding, FileId, Layout, TensorId};

    const FAKE_SECTIONS: &[&str] = &["a.weight", "b.weight"];

    fn fake_is_streamed(name: &str) -> bool {
        name.starts_with("layers.")
            && name.contains(".experts.")
            && (name.ends_with(".a.weight") || name.ends_with(".b.weight"))
    }

    fn fake_parse(name: &str) -> Option<(u32, u32, usize)> {
        let rest = name.strip_prefix("layers.")?;
        let (layer_str, rest) = rest.split_once('.')?;
        let rest = rest.strip_prefix("experts.")?;
        let (expert_str, section) = rest.split_once('.')?;
        let layer: u32 = layer_str.parse().ok()?;
        let expert: u32 = expert_str.parse().ok()?;
        let section_idx = FAKE_SECTIONS.iter().position(|s| *s == section)?;
        Some((layer, expert, section_idx))
    }

    fn fake_collect(
        metadata: &CheckpointMetadata,
        num_layers: u32,
        num_experts: u32,
    ) -> Result<Vec<StreamBinding>, CompileError> {
        collect_bindings_from_named_tensors(
            metadata,
            num_layers,
            num_experts,
            FAKE_SECTIONS.len(),
            fake_is_streamed,
            fake_parse,
        )
    }

    const FAKE_ARCH: StreamArchDesc = StreamArchDesc {
        sections: FAKE_SECTIONS,
        is_streamed: fake_is_streamed,
        collect_bindings: fake_collect,
    };

    #[test]
    fn attach_stream_plan_is_arch_agnostic() {
        // 1 layer × 2 experts × 2 sections.
        let mut tensors = Vec::new();
        let mut id = 0u32;
        let mut offset = 0u64;
        for layer in 0..1u32 {
            for expert in 0..2u32 {
                for (s, section) in FAKE_SECTIONS.iter().enumerate() {
                    let span = 100u64 + s as u64;
                    tensors.push(RawTensor {
                        id: TensorId(id),
                        name: format!("layers.{layer}.experts.{expert}.{section}"),
                        file_id: FileId(0),
                        file_offset: offset,
                        span_bytes: span,
                        shape: vec![span as i64],
                        encoding: Encoding::Raw(DType::U8),
                        layout: Layout::dense(1),
                    });
                    id += 1;
                    offset += span;
                }
            }
        }
        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "shard.bin".into(),
                size_bytes: offset,
                format: CheckpointFormat::Safetensors,
            }],
            tensors,
        };
        let mut program = StorageProgram::empty(StorageTarget::default());
        attach_stream_plan(&mut program, &metadata, 1, 2, FAKE_ARCH).unwrap();

        let stream = &program.stream;
        assert_eq!(stream.num_layers, 1);
        assert_eq!(stream.num_experts, 2);
        assert_eq!(stream.sections_per_expert, 2);
        assert_eq!(stream.section_bytes, vec![100, 101]);
        assert_eq!(stream.bindings.len(), 4);
        assert_eq!(stream.template.len(), 2);
        // 100 + align_up to 256 + 101 → slot covers both sections.
        assert!(stream.slot_bytes >= 100 + 101);
        assert_eq!(stream.section_offsets[0], 0);
        assert_eq!(stream.section_offsets[1], 256);
    }
}
