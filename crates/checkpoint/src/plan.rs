//! The plan: what the loader emits, and the passes that shape it.
//!
//! `plan/build.rs` turns a checked contract into instructions; `plan/passes/`
//! rewrites them into what the executor wants.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::types::{
    BackendKind, BufferId, CheckpointFormat, DType, Encoding, FileId, InstrId, QuantGranularity,
    QuantScheme, RepackSpec, ScaleForm, TensorDecl, TensorId,
};

pub mod build;
pub(crate) mod geometry;
pub mod group;
pub mod index;
pub mod pass;
pub mod passes;

pub use crate::extent::{Dim, Extent};
pub use passes::tile::{
    CONVERT_TILE_MAP_MASK, CUDA_TILE_MAP_MASK, HOST_TILE_MAP_MASK, METAL_TILE_MAP_MASK,
};

/// Which tile-map transforms a target's kernels implement.
pub const TILE_MAP_CAST: u32 = 1 << 0;
pub const TILE_MAP_DECODE: u32 = 1 << 1;
pub const TILE_MAP_ENCODE: u32 = 1 << 2;
pub const TILE_MAP_TRANSCODE: u32 = 1 << 3;
pub const TILE_MAP_REBLOCK: u32 = 1 << 4;
// 1 << 5 was `Reorder`, now unused; the bit stays reserved so the numbering
// below it is stable.
pub const TILE_MAP_REPACK: u32 = 1 << 6;
pub const TILE_MAP_SCALE: u32 = 1 << 7;
pub const TILE_MAP_BIAS: u32 = 1 << 8;

/// Compile a contract into the plan that satisfies it: rewrite the
/// contract, build the instructions, run the passes, decide tiling.
pub fn compile(
    metadata: &crate::file::Metadata,
    contract: &crate::contract::ModelContract,
    target: StorageTarget,
) -> Result<LoadPlan> {
    compile_through(metadata, contract, target, pass::run_all)
}

/// Compile the same contract for an execution that has no arena —
/// [`Execution::streaming`](crate::executor::Execution::streaming), where
/// every buffer is freed at its last use. Same everything but the schedule
/// (see [`pass::Pass::for_arena`]); not a key — [`compile`]'s plan is what
/// a warm-cache identity hashes.
///
/// # Errors
///
/// As [`compile`].
pub fn compile_streaming(
    metadata: &crate::file::Metadata,
    contract: &crate::contract::ModelContract,
    target: StorageTarget,
) -> Result<LoadPlan> {
    compile_through(metadata, contract, target, pass::run_arenaless)
}

/// The pipeline both entry points are, over the pass list each names.
fn compile_through(
    metadata: &crate::file::Metadata,
    contract: &crate::contract::ModelContract,
    target: StorageTarget,
    passes: fn(&mut LoadPlan) -> Result<Vec<pass::PassStats>>,
) -> Result<LoadPlan> {
    let rewritten =
        crate::contract::rewrite::coalesce_direct_row_shards(contract, metadata, &target)?;
    let mut plan = build::build(metadata, &rewritten, target.clone())?;
    // The pipeline ends with `lower-backend-tiling`, so tiling/fusion/kernel
    // fields are never observed as placeholders.
    plan.passes = passes(&mut plan)?;
    // Compiled from the unrewritten contract: each group is rewritten on
    // its own inside `group::compile_all`.
    plan.groups = group::compile_all(metadata, contract, &target)?;
    Ok(plan)
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub persistent_bytes: u64,
    /// Device bytes the arena reserves for transform operands, beyond the
    /// resident tensors — the largest single staged operand, not their sum.
    #[serde(default)]
    pub scratch_bytes: u64,
    pub temporary_peak_bytes: u64,
    pub transform_scratch_peak_bytes: u64,
    pub checkpoint_read_bytes: u64,
    pub device_write_bytes: u64,
}

impl MemoryPlan {
    /// What the caller has to allocate: resident tensors plus staging, one allocation with one base offset.
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
    /// The target a backend asks for, stated once rather than repeated
    /// elsewhere. `native_mxfp4_moe` is the one field not set from the
    /// backend, a per-request capability the caller varies.
    #[must_use]
    pub fn for_backend(backend: BackendKind, tp_rank: u32, tp_size: u32) -> Self {
        Self {
            backend,
            tp_rank,
            tp_size: tp_size.max(1),
            // What cuBLAS wants, and what `cudaMalloc` itself guarantees.
            preferred_alignment: 256,
            // How much host staging one load-time transform may take at once.
            max_tile_bytes: 64 * 1024 * 1024,
            tile_map_mask: passes::tile::compilable_tile_maps(backend),
            // Means "has a native MXFP4 GEMM", not "reads MXFP4".
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
    /// What this buffer's bytes are: shape and encoding, reachable without
    /// [`tensor`](Self::tensor), which an intermediate operand lacks.
    pub ty: crate::contract::TensorType,
    pub bytes: u64,
    pub alignment: u32,
    pub temporary: bool,
    pub persistent_offset: Option<u64>,
    /// Where this buffer sits in the arena's scratch region, if staging.
    /// Ask [`arena_offset`](Self::arena_offset) for either offset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scratch_offset: Option<u64>,
}

impl BufferDecl {
    /// Where this buffer lives in the arena, resident or staging, or `None`
    /// for a host-owned one.
    #[must_use]
    pub fn arena_offset(&self) -> Option<u64> {
        self.persistent_offset.or(self.scratch_offset)
    }

    /// The dtype one element reads as — the logical one for a quantized encoding.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.ty.encoding.dtype()
    }
}

/// A file the plan reads from, indexed by `SourceTensorDecl::file_id`.
/// Paths are stored exactly as opened, so the plan is non-relocatable.
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

/// A quantized tensor and the tensor holding its scales — separate
/// runtime tensors the engine must be told belong together.
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
    /// The type these bytes are read as, not always the checkpoint's own:
    /// [`Expr::Transmute`] can say a `U8`-stored tensor reads as `E8M0`.
    ///
    /// [`Expr::Transmute`]: crate::contract::Expr::Transmute
    pub dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DestExtent {
    pub buffer: BufferId,
    pub offset: u64,
    pub stride: Extent,
}

/// Serde helper: a zero addend is skipped, so plans written before `Bias`
/// existed still serialize byte-identically.
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
    /// The kernel this transform ends in, when it ends in one. `None`
    /// rather than zeros: a non-repacking transform has no rows to fake.
    pub repack: Option<RepackSpec>,
    pub scratch_bytes: u64,
    /// The checkpoint tensor holding this transform's input block scales,
    /// rather than the executor guessing a `_scale_inv` name.
    pub metadata_source: Option<TensorId>,
    /// The multiplier for a [`TileMapKind::Scale`], as [`f32::to_bits`]
    /// (bits, since `f32` has no total equality); zero on other kinds.
    pub scale_factor_bits: u32,
    /// The addend for a [`TileMapKind::Bias`], as [`f32::to_bits`]; zero on
    /// every other kind.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub bias_bits: u32,
    /// Elements of the operand per factor, per axis; empty when uniform.
    /// DeepSeek-style FP8 is `[128, 128]`, row-wise is `[1, 32]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scale_blocks: Vec<i64>,
    /// The backend entry point this transform runs as. Filled in by
    /// [`passes::tile::lower`](crate::plan::passes::tile::lower); `None`
    /// means it runs on the host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageInstr {
    Allocate {
        id: InstrId,
        buffer: BufferId,
    },
    /// Zero `buffer` before anything writes into it — a padded
    /// destination's holes are zeroed once and left alone, not copied.
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
    /// The gather lowering: permute `source` block by block, write `dest`
    /// dense — reorders bytes rather than merely striding them.
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

/// The table a [`StorageInstr::GatherWrite`] walks: destination block `i`
/// reads source block `indices[i]`, repeated over `rows`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatherSpec {
    /// Source block per destination block, in blocks, not bytes.
    pub indices: Vec<i64>,
    /// Bytes in one block: one element's worth on the innermost axis.
    pub block_bytes: u64,
    /// How many times the table repeats.
    pub rows: u64,
    /// Bytes between consecutive source rows.
    pub src_row_bytes: u64,
}

impl GatherSpec {
    /// Bytes between consecutive destination rows. Derived, not stored.
    pub fn dst_row_bytes(&self) -> u64 {
        self.indices.len() as u64 * self.block_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadPlan {
    pub target: StorageTarget,
    /// What each plan pass did.
    pub passes: Vec<pass::PassStats>,
    pub files: Vec<CheckpointFileDecl>,
    pub sources: Vec<SourceTensorDecl>,
    pub tensors: Vec<TensorDecl>,
    pub buffers: Vec<BufferDecl>,
    pub instrs: Vec<StorageInstr>,
    pub schedule: Vec<InstrId>,
    pub memory: MemoryPlan,
    /// Quantized tensors paired with the tensors holding their scales.
    pub attachments: Vec<QuantAttachment>,
    /// Interchangeable sets of tensors, each compiled once; empty and elided for a contract with no group.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupPlan>,
}

/// One plan, `arity` instances, differing only in which bytes they read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupPlan {
    pub name: String,
    pub arity: u32,
    /// The program one instance runs, compiled at index 0 — a whole
    /// [`LoadPlan`], since an instance is a self-contained load.
    pub plan: LoadPlan,
    /// `bindings[i]` is what instance `i` reads instead of
    /// [`plan`](Self::plan), indexed by instance then by instruction.
    pub bindings: Vec<Vec<SourceBinding>>,
}

/// Where one instruction's bytes come from, for one group instance —
/// everything else about the read is in the template.
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

    /// Does this plan publish one tensor for the embedding and output projection?
    #[must_use]
    pub fn ties_embeddings(&self) -> bool {
        self.tensors
            .iter()
            .any(|tensor| tensor.name == TIED_EMBEDDING_NAME)
    }

    /// The names of every tensor this plan leaves in MXFP4. A set, not a
    /// flag: a checkpoint need not be uniform.
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

    /// Every distinct affine point this plan's tensors arrive at:
    /// `(group_size, bits_per_element)`. Not a single point: `mlx_lm` can
    /// publish a routed stack at 4 bits and its router gate at 8.
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

    /// Every affine tensor's point, by name — for an engine answering
    /// several names without matching [`Encoding`].
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

    /// The affine point one named tensor arrives at, or `None` if absent, raw, or MXFP4.
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

    /// Each distinct affine point beside one witness tensor (the first), sorted.
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

/// The one name a contract publishes when the embedding and the output
/// projection are the same tensor. This crate does not emit it — several
/// contract authors in `crates/models` do.
pub const TIED_EMBEDDING_NAME: &str = "shared_embedding.weight";

