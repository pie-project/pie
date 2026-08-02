//! The plan: what the loader emits, and the passes that shape it.
//!
//! `build` turns a checked contract into instructions; `passes` rewrites them
//! into the form the executor wants; `mod.rs` is only the vocabulary the two
//! share.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::types::{
    BackendKind, BufferId, CheckpointFormat, DType, FileId, InstrId, QuantGranularity, QuantScheme,
    RepackSpec, ScaleForm, TensorDecl, TensorId,
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
///
/// Defined here, not in `ffi/`: the plan is the thing that has transforms, and
/// the ABI is a view of the plan. The arrow used to point the other way — the
/// compiler imported `crate::ffi::types::PIE_LOADER_TILE_MAP_*` — which
/// made the core depend on its own serialization format. `ffi/types.rs` now
/// restates these under the C names and a `const` assertion pins the two
/// together, because cbindgen emits literals and cannot follow a path.
pub const TILE_MAP_CAST: u32 = 1 << 0;
pub const TILE_MAP_DECODE: u32 = 1 << 1;
pub const TILE_MAP_ENCODE: u32 = 1 << 2;
pub const TILE_MAP_TRANSCODE: u32 = 1 << 3;
pub const TILE_MAP_REBLOCK: u32 = 1 << 4;
// 1 << 5 was `Reorder`, which no contract could reach and which the CUDA
// transcode engine dispatched to `reblock_tile_map` anyway — one transform
// under two names. The bit stays reserved so the numbering below it is stable.
pub const TILE_MAP_REPACK: u32 = 1 << 6;
pub const TILE_MAP_SCALE: u32 = 1 << 7;

/// Transform chains a backend can collapse into one kernel.
pub const FUSION_FP8_TO_MXFP4: u32 = 1 << 0;

/// Compile a contract into the plan that satisfies it.
///
/// The whole pipeline, and short enough to read: rewrite the contract, build
/// the instructions, run the passes, decide the backend's tiling.
pub fn compile(
    metadata: &crate::checkpoint::CheckpointMetadata,
    contract: &crate::contract::ModelContract,
    target: StorageTarget,
) -> Result<LoadPlan> {
    let rewritten =
        crate::contract::rewrite::coalesce_direct_row_shards(contract, metadata, &target)?;
    let mut plan = build::build(metadata, &rewritten, target.clone())?;
    plan.passes = pass::run_all(&mut plan)?;
    // Runs last, so a plan is never observable in a state where its tiling and
    // fusion fields are still placeholders.
    passes::tile::lower(&mut plan);
    // Compiled from the *unrewritten* contract, because `groups` is not what
    // the row-shard rewrite looks at; each group is rewritten on its own inside
    // `group::compile_all`, where the sub-contract it applies to exists.
    plan.groups = group::compile_all(metadata, contract, &target)?;
    Ok(plan)
}

pub fn compiler_version() -> u64 {
    env!("PIE_LOADER_COMPILER_HASH").parse::<u64>().unwrap_or(0)
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
    pub tile_map_mask: u32,
    pub native_mxfp4_moe: bool,
    /// Which fused transform chains the backend has kernels for.
    ///
    /// A capability, not a preference: the driver both knows whether the fused
    /// kernels are built and owns the opt-out that used to be
    /// `PIE_CUDA_DISABLE_FUSED_TRANSCODE`. Reading it here rather than in the
    /// executor is what makes the choice part of the plan, and therefore part
    /// of the plan hash (`architecture.md` §8.1).
    pub fusion_mask: u32,
    /// The dtype this target's encode kernels dequantize *through*.
    ///
    /// An Encode that does not have a direct kernel goes source → scratch →
    /// destination, and the scratch width is what decides how many rows fit in
    /// the tile budget. CUDA's is BF16. It is a device fact — it is which
    /// kernels were compiled — so it is stated, not assumed.
    pub encode_scratch_dtype: DType,
    /// Row granularity of the block scales this target's encode path consumes,
    /// or `0` if it has none.
    ///
    /// A block-scaled source carries one scale per `[block_scale_rows, N]`
    /// tile, so slicing the dequant by an arbitrary row count would cut a scale
    /// block in half. Rather than round the tile down to a multiple — which
    /// would be a second rule to keep in sync with the kernel — such a source is
    /// simply not tiled.
    pub block_scale_rows: u32,
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
            fusion_mask: 0,
            encode_scratch_dtype: DType::BF16,
            block_scale_rows: 0,
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

/// A file the plan reads from.
///
/// `SourceTensorDecl::file_id` indexes this table. Before it existed, the table
/// was an *unwritten* contract: the loader enumerated the shards one way and
/// the driver re-enumerated them another, and nothing checked that the two
/// agreed on which file index 3 was. Both sides sorted, so they did agree — but
/// by coincidence of two implementations, not by construction
/// (`architecture.md` §6).
///
/// Paths are stored exactly as the loader opened them, which makes the plan
/// non-relocatable. That is the honest representation: the plan was compiled
/// against the headers of *these* files, and a plan that pointed somewhere else
/// would be describing a checkpoint it never read.
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

/// A quantized tensor and the tensor holding its scales.
///
/// The two are separate runtime tensors — the driver materializes both and then
/// has to know they belong together in order to attach the quant metadata its
/// kernels read.
///
/// An affine scheme needs a third: its groups are offset as well as scaled, and
/// `zero_point_tensor` names the tensor holding the offsets. It is `None` for
/// every symmetric scheme, which is to say for everything whose `scale_form` is
/// not [`ScaleForm::Bf16AffineFactors`].
///
/// Every entry here is recorded by whoever declared the scale tensor, at the
/// point of declaring it: `plan/build.rs::quant_metadata_outputs` for scales the
/// loader creates, and [`Scales`](crate::contract::Scales) for scales the
/// checkpoint shipped. Neither involves inspecting a name.
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
    /// The type these bytes are read as, which is not always the type the
    /// checkpoint declares: [`Expr::Transmute`] exists precisely to say that a
    /// tensor stored as `U8` is to be read as `E8M0`. Looking the dtype up
    /// from `tensor_id` instead would discard that, and the executor would be
    /// asked for a cast from the storage type it was told to stop believing.
    ///
    /// Only the dtype can differ. A transmute preserves the byte count
    /// (`contract::infer`), so an extent's quantization scheme is still
    /// whatever `PieLoaderPlan::sources[tensor_id]` says.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileMapKind {
    Cast,
    Decode,
    Encode,
    Transcode,
    Reblock,
    Repack,
    Scale,
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
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileSpec {
    pub max_tile_bytes: u64,
    /// Rows of the output the driver transforms per launch; `0` means the whole
    /// tensor in one pass.
    ///
    /// A *decision*, unlike `max_tile_bytes`, which is only a budget. The driver
    /// used to turn the budget into a row count while executing
    /// (`transcode_engine.hpp::encode_rows_per_tile`), which left the plan
    /// silent about how it would actually run. Filled in by
    /// [`crate::plan::passes::tile::lower`].
    pub rows_per_tile: u32,
}

/// A transform chain the backend collapsed into one kernel.
///
/// Recorded rather than inferred: fusing FP8 → MXFP4 skips a BF16 round-trip
/// through HBM, and while it is bit-identical to the two-step path, it is a
/// different kernel sequence. A plan that does not say which one it means
/// cannot claim to determine execution (`architecture.md` §8.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformFusion {
    #[default]
    None,
    /// Encode an FP8 source directly to MXFP4.
    Fp8ToMxfp4,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformSpec {
    pub from: Option<QuantScheme>,
    pub to: Option<QuantScheme>,
    /// The kernel this transform ends in, when it ends in one.
    ///
    /// `None` rather than a layout of that name: a transform that does not
    /// repack has no rows, no columns and no kernel, and a struct of zeros
    /// standing for that is a value every reader has to know to disbelieve.
    pub repack: Option<RepackSpec>,
    pub scratch_bytes: u64,
    pub fusion: TransformFusion,
    /// The checkpoint tensor holding this transform's *input* block scales.
    ///
    /// A block-scaled FP8 source is two tensors on disk: the payload and a
    /// sibling of per-group factors. Only the payload is named by the contract,
    /// so the executor used to rebuild the sibling's name by appending
    /// `_scale_inv` and look it up — the same guess-the-loader's-answer
    /// anti-pattern `attachments` removed from the output side
    /// (`architecture.md` §12 row 10). The loader reads the tensor table, so
    /// the loader answers.
    pub metadata_source: Option<TensorId>,
    /// The multiplier for a [`TileMapKind::Scale`], as [`f32::to_bits`]; zero
    /// on every other kind.
    ///
    /// Bits for the reason [`Expr::Scale`](crate::contract::Expr::Scale) gives:
    /// a plan is compared and hashed, and `f32` has no total equality. The
    /// executor's own multiply is done in `f32`, so the loader hands over the
    /// same 32 bits the contract named rather than a widened value that would
    /// have to be narrowed again.
    pub scale_factor_bits: u32,
    /// Elements of the operand per factor, on each axis, for a per-block
    /// [`TileMapKind::Scale`]; empty when the factor is the uniform constant in
    /// [`scale_factor_bits`](TransformSpec::scale_factor_bits).
    ///
    /// Non-empty is what tells the executor to read its factors from the extra
    /// input buffer instead, so the two cases cannot be confused for one
    /// another by a field left unset.
    ///
    /// One entry per axis, so a two-dimensional block scale — which is what a
    /// DeepSeek-style FP8 checkpoint ships — is `[128, 128]` and the ordinary
    /// row-wise case is `[1, 32]`. Derived here rather than restated by the
    /// author: [`ScaleFactor::PerBlock`](crate::contract::ScaleFactor::PerBlock)
    /// carries only the factors, and the blocking is the ratio of the two
    /// inferred shapes. Both executors can recompute it and check this against
    /// what they see.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scale_blocks: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageInstr {
    Allocate {
        id: InstrId,
        buffer: BufferId,
    },
    /// Zero `buffer` before anything writes into it.
    ///
    /// A padded destination has holes, and a hole is not copied zeros — it is
    /// a destination that was zeroed once and then not written to. That is why
    /// `Lowering::cost` prices padding as one fill rather than one copy per
    /// band: dropping the holes lets the data on either side fold together,
    /// which for a head-dim pad is the difference between `2·n_heads` copies
    /// and one.
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
pub struct LoadPlan {
    pub compiler_version: u64,
    pub target: StorageTarget,
    /// What each plan pass did. Replaces the old `optimizer` report, which
    /// described a no-op pass over an IR that no longer exists.
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
    /// Interchangeable sets of tensors, each compiled once.
    ///
    /// Empty for every contract that declares no group, and elided when it is:
    /// a plan recorded before groups existed still reads, and one compiled from
    /// a contract without groups still records identically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupPlan>,
}

/// One plan, `arity` instances, differing only in which bytes they read.
///
/// See [`plan::group`](crate::plan::group) for what that sentence is worth and
/// how it is proved. The driver decides what a group is *for*; the plan only
/// says the instances are substitutable and where each one's bytes live.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupPlan {
    pub name: String,
    pub arity: u32,
    /// The program one instance runs, compiled at index 0.
    ///
    /// A whole [`LoadPlan`] rather than a bare instruction list: an instance is
    /// a self-contained load, with its own buffers and its own memory
    /// accounting, and the driver runs it with the executor it already has.
    pub plan: LoadPlan,
    /// `bindings[i]` is what instance `i` reads instead of what
    /// [`plan`](Self::plan) says. Indexed by instance, then by the
    /// source-naming instructions of `plan` in order.
    pub bindings: Vec<Vec<SourceBinding>>,
}

/// Where one instruction's bytes come from, for one instance of a group.
///
/// Exactly the three fields that locate bytes in a checkpoint. Everything else
/// about the read -- how many bytes, with what stride, read as what dtype --
/// is in the template, because a group whose instances disagreed about any of
/// those would not have compiled.
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
            compiler_version: compiler_version(),
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
}
