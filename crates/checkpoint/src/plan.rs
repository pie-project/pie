use serde::{Deserialize, Serialize};

use crate::contract::UnaryOp;
use crate::error::Result;
use crate::types::{
    BackendKind, BufferId, CheckpointFormat, DType, Encoding, FileId, InstrId, QuantGranularity,
    QuantScheme, RepackSpec, ScaleForm, TensorDecl, TensorId,
};

pub mod build;
pub mod constant;
pub(crate) mod geometry;
pub mod group;
pub mod index;
pub mod pass;
pub mod passes;

pub use crate::extent::{Dim, Extent};
pub use passes::tile::{
    CONVERT_TILE_MAP_MASK, CUDA_TILE_MAP_MASK, HOST_TILE_MAP_MASK, METAL_TILE_MAP_MASK,
    VULKAN_TILE_MAP_MASK, WGPU_TILE_MAP_MASK,
};

pub const TILE_MAP_CAST: u32 = 1 << 0;
pub const TILE_MAP_DECODE: u32 = 1 << 1;
pub const TILE_MAP_ENCODE: u32 = 1 << 2;
pub const TILE_MAP_TRANSCODE: u32 = 1 << 3;
pub const TILE_MAP_REBLOCK: u32 = 1 << 4;
pub const TILE_MAP_REPACK: u32 = 1 << 6;
pub const TILE_MAP_SCALE: u32 = 1 << 7;
pub const TILE_MAP_BIAS: u32 = 1 << 8;
pub const TILE_MAP_UNARY: u32 = 1 << 9;

pub fn compile(
    metadata: &crate::file::Metadata,
    contract: &crate::contract::ModelContract,
    target: StorageTarget,
) -> Result<LoadPlan> {
    compile_through(metadata, contract, target, pass::run_all)
}

pub fn compile_streaming(
    metadata: &crate::file::Metadata,
    contract: &crate::contract::ModelContract,
    target: StorageTarget,
) -> Result<LoadPlan> {
    compile_through(metadata, contract, target, pass::run_arenaless)
}

fn compile_through(
    metadata: &crate::file::Metadata,
    contract: &crate::contract::ModelContract,
    target: StorageTarget,
    passes: fn(&mut LoadPlan) -> Result<Vec<pass::PassStats>>,
) -> Result<LoadPlan> {
    let (metadata, contract) = constant::mounted(metadata, contract)?;
    let (metadata, contract) = (metadata.as_ref(), contract.as_ref());
    let rewritten =
        crate::contract::rewrite::coalesce_direct_row_shards(contract, metadata, &target)?;
    let mut plan = build::build(metadata, &rewritten, target.clone())?;
    plan.passes = passes(&mut plan)?;
    plan.groups = group::compile_all(metadata, contract, &target)?;
    Ok(plan)
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub persistent_bytes: u64,
    #[serde(default)]
    pub scratch_bytes: u64,
    pub temporary_peak_bytes: u64,
    pub transform_scratch_peak_bytes: u64,
    pub checkpoint_read_bytes: u64,
    pub device_write_bytes: u64,
}

impl MemoryPlan {
    #[must_use]
    pub fn arena_bytes(&self) -> u64 {
        self.persistent_bytes.saturating_add(self.scratch_bytes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageTarget {
    pub backend: BackendKind,
    pub tp_rank: u32,
    pub tp_size: u32,
    pub max_tile_bytes: u64,
    pub preferred_alignment: u32,
    pub tile_map_mask: u32,
    pub native_mxfp4_moe: bool,
}

impl StorageTarget {
    #[must_use]
    pub fn for_backend(backend: BackendKind, tp_rank: u32, tp_size: u32) -> Self {
        Self {
            backend,
            tp_rank,
            tp_size: tp_size.max(1),
            preferred_alignment: 256,
            max_tile_bytes: 64 * 1024 * 1024,
            tile_map_mask: passes::tile::compilable_tile_maps(backend),
            native_mxfp4_moe: false,
        }
    }
}

impl Default for StorageTarget {
    fn default() -> Self {
        Self {
            backend: BackendKind::Unknown,
            tp_rank: 0,
            tp_size: 1,
            max_tile_bytes: 0,
            preferred_alignment: 1,
            tile_map_mask: HOST_TILE_MAP_MASK,
            native_mxfp4_moe: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferDecl {
    pub id: BufferId,
    pub tensor: Option<TensorId>,
    pub ty: crate::contract::TensorType,
    pub bytes: u64,
    pub alignment: u32,
    pub temporary: bool,
    pub persistent_offset: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scratch_offset: Option<u64>,
}

impl BufferDecl {
    #[must_use]
    pub fn arena_offset(&self) -> Option<u64> {
        self.persistent_offset.or(self.scratch_offset)
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.ty.encoding.dtype()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointFileDecl {
    pub id: FileId,
    pub path: String,
    pub size_bytes: u64,
    pub format: CheckpointFormat,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantAttachment {
    pub tensor: TensorId,
    pub scale_tensor: TensorId,
    pub zero_point_tensor: Option<TensorId>,
    pub granularity: QuantGranularity,
    pub group_size: u32,
    pub channel_axis: u32,
    pub scale_form: ScaleForm,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceExtent {
    pub file_id: FileId,
    pub tensor_id: TensorId,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub stride: Extent,
    pub dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DestExtent {
    pub buffer: BufferId,
    pub offset: u64,
    pub stride: Extent,
}

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileMapKind {
    Cast,
    Decode,
    Encode,
    Transcode,
    Reblock,
    Repack,
    Scale,
    Bias,
    Unary,
}

impl TileMapKind {
    pub const fn capability_bit(self) -> u32 {
        match self {
            Self::Cast => TILE_MAP_CAST,
            Self::Decode => TILE_MAP_DECODE,
            Self::Encode => TILE_MAP_ENCODE,
            Self::Transcode => TILE_MAP_TRANSCODE,
            Self::Reblock => TILE_MAP_REBLOCK,
            Self::Repack => TILE_MAP_REPACK,
            Self::Scale => TILE_MAP_SCALE,
            Self::Bias => TILE_MAP_BIAS,
            Self::Unary => TILE_MAP_UNARY,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileSpec {
    pub max_tile_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformSpec {
    pub from: Option<QuantScheme>,
    pub to: Option<QuantScheme>,
    pub repack: Option<RepackSpec>,
    pub scratch_bytes: u64,
    pub metadata_source: Option<TensorId>,
    pub scale_factor_bits: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub bias_bits: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scale_blocks: Vec<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unary: Option<UnaryOp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageInstr {
    Allocate {
        id: InstrId,
        buffer: BufferId,
    },
    Fill {
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
    GatherWrite {
        id: InstrId,
        source: SourceExtent,
        dest: DestExtent,
        gather: GatherSpec,
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
    },
    Finalize {
        id: InstrId,
        tensor: BufferId,
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatherSpec {
    pub indices: Vec<i64>,
    pub block_bytes: u64,
    pub rows: u64,
    pub src_row_bytes: u64,
}

impl GatherSpec {
    pub fn dst_row_bytes(&self) -> u64 {
        self.indices.len() as u64 * self.block_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadPlan {
    pub target: StorageTarget,
    pub passes: Vec<pass::PassStats>,
    pub files: Vec<CheckpointFileDecl>,
    pub sources: Vec<SourceTensorDecl>,
    pub tensors: Vec<TensorDecl>,
    pub buffers: Vec<BufferDecl>,
    pub instrs: Vec<StorageInstr>,
    pub schedule: Vec<InstrId>,
    pub memory: MemoryPlan,
    pub attachments: Vec<QuantAttachment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupPlan {
    pub name: String,
    pub arity: u32,
    pub plan: LoadPlan,
    pub bindings: Vec<Vec<SourceBinding>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceBinding {
    pub instr: InstrId,
    pub file_id: FileId,
    pub tensor_id: TensorId,
    pub file_offset: u64,
}

impl LoadPlan {
    pub fn empty(target: StorageTarget) -> Self {
        Self {
            target,
            passes: Vec::new(),
            files: Vec::new(),
            sources: Vec::new(),
            tensors: Vec::new(),
            buffers: Vec::new(),
            instrs: Vec::new(),
            schedule: Vec::new(),
            memory: MemoryPlan::default(),
            attachments: Vec::new(),
            groups: Vec::new(),
        }
    }

    #[must_use]
    pub fn ties_embeddings(&self) -> bool {
        self.tensors
            .iter()
            .any(|tensor| tensor.name == TIED_EMBEDDING_NAME)
    }

    #[must_use]
    pub fn mxfp4_tensor_names(&self) -> std::collections::HashSet<String> {
        self.tensors
            .iter()
            .filter(|t| {
                matches!(
                    &t.encoding,
                    Encoding::Quant(spec) if spec.is_mxfp4()
                )
            })
            .map(|t| t.name.clone())
            .collect()
    }

    #[must_use]
    pub fn affine_points(&self) -> Vec<(u32, u32)> {
        let mut points: Vec<(u32, u32)> = self
            .tensors
            .iter()
            .filter_map(|t| match &t.encoding {
                Encoding::Quant(spec) => spec.affine_point(),
                _ => None,
            })
            .collect();
        points.sort_unstable();
        points.dedup();
        points
    }

    #[must_use]
    pub fn affine_by_name(&self) -> std::collections::HashMap<String, (u32, u32)> {
        self.tensors
            .iter()
            .filter_map(|t| match &t.encoding {
                Encoding::Quant(spec) => spec.affine_point().map(|p| (t.name.clone(), p)),
                _ => None,
            })
            .collect()
    }

    #[must_use]
    pub fn affine_point_of(&self, name: &str) -> Option<(u32, u32)> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .and_then(|t| match &t.encoding {
                Encoding::Quant(spec) => spec.affine_point(),
                _ => None,
            })
    }

    #[must_use]
    pub fn affine_point_witnesses(&self) -> Vec<((u32, u32), String)> {
        let mut out: Vec<((u32, u32), String)> = Vec::new();
        for t in &self.tensors {
            let Encoding::Quant(spec) = &t.encoding else {
                continue;
            };
            let Some(point) = spec.affine_point() else {
                continue;
            };
            if !out.iter().any(|(p, _)| *p == point) {
                out.push((point, t.name.clone()));
            }
        }
        out.sort_unstable_by_key(|(p, _)| *p);
        out
    }
}

pub const TIED_EMBEDDING_NAME: &str = "shared_embedding.weight";
