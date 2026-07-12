use serde::{Deserialize, Serialize};

use crate::optimizer::OptimizerReport;
use crate::types::{
    BackendKind, BufferId, FileId, InstrId, Layout, Mxfp4MoePolicy, QuantScheme, RepackSpec,
    TensorDecl, TensorId,
};

pub const STORAGE_PROGRAM_VERSION: u32 = 4;

pub fn compiler_version() -> u64 {
    env!("PIE_WL_COMPILER_HASH").parse::<u64>().unwrap_or(0)
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub persistent_bytes: u64,
    pub temporary_peak_bytes: u64,
    pub transform_scratch_peak_bytes: u64,
    pub checkpoint_read_bytes: u64,
    pub device_write_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageTarget {
    pub backend: BackendKind,
    pub tp_rank: u32,
    pub tp_size: u32,
    pub max_tile_bytes: u64,
    pub preferred_alignment: u32,
    pub mxfp4_moe: Mxfp4MoePolicy,
    pub native_mxfp4_moe: bool,
}

impl Default for StorageTarget {
    fn default() -> Self {
        Self {
            backend: BackendKind::Unknown,
            tp_rank: 0,
            tp_size: 1,
            max_tile_bytes: 0,
            preferred_alignment: 1,
            mxfp4_moe: Mxfp4MoePolicy::RoutedDecode,
            native_mxfp4_moe: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferDecl {
    pub id: BufferId,
    pub tensor: Option<TensorId>,
    pub bytes: u64,
    pub alignment: u32,
    pub temporary: bool,
    pub persistent_offset: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceTensorDecl {
    pub id: TensorId,
    pub name: String,
    pub file_id: FileId,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub shape: Vec<i64>,
    pub encoding: crate::types::Encoding,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StridedExtent {
    pub base_offset: u64,
    pub element_bytes: u32,
    pub dims: Vec<DimSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimSpec {
    pub count: i64,
    pub src_stride: i64,
    pub dst_stride: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceExtent {
    pub file_id: FileId,
    pub tensor_id: TensorId,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub stride: StridedExtent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DestExtent {
    pub buffer: BufferId,
    pub offset: u64,
    pub stride: StridedExtent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlabPlacement {
    pub src_offset: u64,
    pub dest_offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileMapKind {
    Cast,
    Decode,
    Encode,
    Transcode,
    Reblock,
    Reorder,
    Repack,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileSpec {
    pub max_tile_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformSpec {
    pub from: Option<QuantScheme>,
    pub to: Option<QuantScheme>,
    pub repack: RepackSpec,
    pub scratch_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataSpec {
    pub kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageInstr {
    Allocate {
        id: InstrId,
        buffer: BufferId,
    },
    ExtentWrite {
        id: InstrId,
        source: SourceExtent,
        dest: DestExtent,
    },
    BulkExtentWrite {
        id: InstrId,
        source: SourceExtent,
        dest_offset: u64,
    },
    SlabScatter {
        id: InstrId,
        file_id: FileId,
        file_offset: u64,
        span_bytes: u64,
        placements: Vec<SlabPlacement>,
    },
    TileMap {
        id: InstrId,
        kind: TileMapKind,
        source: Option<SourceExtent>,
        dest: Option<DestExtent>,
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
        tile: TileSpec,
        transform: TransformSpec,
    },
    CreateView {
        id: InstrId,
        input: BufferId,
        output: BufferId,
        view: DestExtent,
        layout: Layout,
    },
    Attach {
        id: InstrId,
        tensor: BufferId,
        metadata: Vec<BufferId>,
        spec: MetadataSpec,
    },
    Release {
        id: InstrId,
        buffer: BufferId,
    },
    Finalize {
        id: InstrId,
        tensor: BufferId,
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageProgram {
    pub version: u32,
    pub compiler_version: u64,
    pub target: StorageTarget,
    pub optimizer: OptimizerReport,
    pub sources: Vec<SourceTensorDecl>,
    pub tensors: Vec<TensorDecl>,
    pub buffers: Vec<BufferDecl>,
    pub instrs: Vec<StorageInstr>,
    pub schedule: Vec<InstrId>,
    pub memory: MemoryPlan,
}

impl StorageProgram {
    pub fn empty(target: StorageTarget) -> Self {
        Self {
            version: STORAGE_PROGRAM_VERSION,
            compiler_version: compiler_version(),
            target,
            optimizer: OptimizerReport::default(),
            sources: Vec::new(),
            tensors: Vec::new(),
            buffers: Vec::new(),
            instrs: Vec::new(),
            schedule: Vec::new(),
            memory: MemoryPlan::default(),
        }
    }

    pub fn summary(&self) -> StorageProgramSummary {
        let mut s = StorageProgramSummary::default();
        s.tensor_count = self.tensors.len();
        s.buffer_count = self.buffers.len();
        s.schedule_len = self.schedule.len();
        s.persistent_bytes = self.memory.persistent_bytes;
        s.checkpoint_read_bytes = self.memory.checkpoint_read_bytes;
        s.device_write_bytes = self.memory.device_write_bytes;
        for instr in &self.instrs {
            match instr {
                StorageInstr::Allocate { .. } => s.allocate_count += 1,
                StorageInstr::ExtentWrite { source, .. } => {
                    s.extent_write_count += 1;
                    s.extent_write_bytes += source.span_bytes;
                }
                StorageInstr::BulkExtentWrite { source, .. } => {
                    s.bulk_extent_write_count += 1;
                    s.bulk_extent_write_bytes += source.span_bytes;
                }
                StorageInstr::SlabScatter {
                    placements,
                    span_bytes,
                    ..
                } => {
                    s.slab_scatter_count += 1;
                    s.slab_scatter_placement_count += placements.len();
                    s.slab_scatter_span_bytes += span_bytes;
                    s.slab_scatter_payload_bytes += placements.iter().map(|p| p.bytes).sum::<u64>();
                }
                StorageInstr::TileMap { .. } => s.tile_map_count += 1,
                StorageInstr::CreateView { .. } => s.create_view_count += 1,
                StorageInstr::Attach { .. } => s.attach_count += 1,
                StorageInstr::Release { .. } => s.release_count += 1,
                StorageInstr::Finalize { .. } => s.finalize_count += 1,
            }
        }
        s
    }
}

#[derive(Clone, Debug, Default)]
pub struct StorageProgramSummary {
    pub tensor_count: usize,
    pub buffer_count: usize,
    pub schedule_len: usize,
    pub persistent_bytes: u64,
    pub checkpoint_read_bytes: u64,
    pub device_write_bytes: u64,
    pub allocate_count: usize,
    pub extent_write_count: usize,
    pub extent_write_bytes: u64,
    pub bulk_extent_write_count: usize,
    pub bulk_extent_write_bytes: u64,
    pub slab_scatter_count: usize,
    pub slab_scatter_placement_count: usize,
    pub slab_scatter_span_bytes: u64,
    pub slab_scatter_payload_bytes: u64,
    pub tile_map_count: usize,
    pub create_view_count: usize,
    pub attach_count: usize,
    pub release_count: usize,
    pub finalize_count: usize,
}

impl std::fmt::Display for StorageProgramSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tensors={} buffers={} schedule={} \
             alloc={} extent_write={} bulk={} slab={} ({}placements, {:.1}MiB payload, {:.1}MiB span) \
             tile_map={} view={} finalize={} \
             persistent={:.1}MiB read={:.1}MiB write={:.1}MiB",
            self.tensor_count,
            self.buffer_count,
            self.schedule_len,
            self.allocate_count,
            self.extent_write_count,
            self.bulk_extent_write_count,
            self.slab_scatter_count,
            self.slab_scatter_placement_count,
            self.slab_scatter_payload_bytes as f64 / (1024.0 * 1024.0),
            self.slab_scatter_span_bytes as f64 / (1024.0 * 1024.0),
            self.tile_map_count,
            self.create_view_count,
            self.finalize_count,
            self.persistent_bytes as f64 / (1024.0 * 1024.0),
            self.checkpoint_read_bytes as f64 / (1024.0 * 1024.0),
            self.device_write_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}
