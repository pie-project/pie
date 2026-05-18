use serde::{Deserialize, Serialize};

use crate::optimizer::OptimizerReport;
use crate::types::{
    BackendKind, BufferId, FileId, InstrId, Layout, Mxfp4MoePolicy, QuantScheme, TensorDecl,
    TensorId,
};

pub const STORAGE_PROGRAM_VERSION: u32 = 1;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileMapKind {
    Cast,
    Decode,
    Encode,
    Transcode,
    Reblock,
    Reorder,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileSpec {
    pub max_tile_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformSpec {
    pub from: Option<QuantScheme>,
    pub to: Option<QuantScheme>,
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
    pub target: StorageTarget,
    pub optimizer: OptimizerReport,
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
            target,
            optimizer: OptimizerReport::default(),
            tensors: Vec::new(),
            buffers: Vec::new(),
            instrs: Vec::new(),
            schedule: Vec::new(),
            memory: MemoryPlan::default(),
        }
    }
}
