//! The plan: what the loader emits, and the passes that shape it.
//!
//! `plan/build.rs` turns a checked contract into instructions; `plan/passes/`
//! rewrites them into the form the executor wants; this file is only the
//! vocabulary the two share.

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
pub mod spans;

pub use crate::extent::{Dim, Extent};
pub use passes::tile::{
    CONVERT_TILE_MAP_MASK, CUDA_TILE_MAP_MASK, HOST_TILE_MAP_MASK, METAL_TILE_MAP_MASK,
};

/// Which tile-map transforms a target's kernels implement.
///
/// Defined here because the plan is the thing that has transforms. The arrow
/// used to point the other way — the compiler imported
/// `crate::ffi::types::PIE_LOADER_TILE_MAP_*`, which made the core depend on
/// its own serialization format. The C ABI that justified that dependency has
/// since gone, and these are now simply where they belong: beside the target
/// that advertises them.
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
    // The pipeline ends with `lower-backend-tiling`, so a plan is never
    // observable in a state where its tiling, fusion and kernel fields are
    // still placeholders — and the validators after it get to see what it
    // decided.
    plan.passes = pass::run_all(&mut plan)?;
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
    /// Device bytes the arena reserves BEYOND the resident tensors, for the
    /// operands of transforms the device runs itself.
    ///
    /// The arena used to be defined as "the resident tensors, laid out", so
    /// anything that was not a resident tensor was host memory by
    /// construction — which is why no transform reading a file or an
    /// intermediate could ever have its operands on the device, and why every
    /// load-time kernel in this tree was unreachable
    /// (`.wiki/fix/loader.md` §3.3).
    ///
    /// Bounded by the transform, not by the model: staging buffers are reused
    /// across a schedule that runs one instruction at a time, so this is the
    /// largest single staged operand and not their sum.
    #[serde(default)]
    pub scratch_bytes: u64,
    pub temporary_peak_bytes: u64,
    pub transform_scratch_peak_bytes: u64,
    pub checkpoint_read_bytes: u64,
    pub device_write_bytes: u64,
}

impl MemoryPlan {
    /// What the caller has to allocate.
    ///
    /// The resident tensors and the staging region behind them, which is one
    /// number because they are one allocation: every offset in the plan —
    /// `persistent_offset`, `scratch_offset`, `BulkExtentWrite::dest_offset` —
    /// is measured from the same base.
    ///
    /// A method rather than a wider `persistent_bytes`, because that field
    /// answers a question three call sites still ask on its own: how much of
    /// the arena holds tensors that outlive the load.
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

impl StorageTarget {
    /// The target a backend asks for, stated once.
    ///
    /// Four sites used to write this literal — both drivers, `pie model build`
    /// and `pie model import` — and every one of them repeated `256`, `64 MiB`
    /// and `BF16`. Repetition was not the worst of it: each also restated the
    /// backend's `tile_map_mask`, so a driver and
    /// [`passes::tile`](crate::plan::passes::tile) each held an opinion about
    /// which transforms that device implements, and a test compared the two
    /// instead of there being one.
    ///
    /// The mask comes from [`passes::tile::tile_map_mask`], which is the
    /// loader's model of the backend and now the only statement of it. That
    /// inverts the old rule — the driver was the authority and the loader
    /// checked it — and the reason is that the loader is where the consequence
    /// lands: it decides which plans compile, and it owns the host fallback
    /// every claimed transform has to have.
    ///
    /// The fields NOT here are the ones a caller genuinely varies:
    /// `native_mxfp4_moe` and `fusion_mask` are per-request capabilities, and
    /// `block_scale_rows` belongs to an encode path. Each starts at the
    /// conservative answer and is set by the caller that knows better.
    #[must_use]
    pub fn for_backend(backend: BackendKind, tp_rank: u32, tp_size: u32) -> Self {
        Self {
            backend,
            tp_rank,
            tp_size: tp_size.max(1),
            // What cuBLAS wants for a matrix operand and what `cudaMalloc`
            // itself guarantees, so a view into the arena is as aligned as its
            // own allocation would have been. Metal's buffers want no less.
            preferred_alignment: 256,
            // How much host staging one load-time transform may take at once.
            max_tile_bytes: 64 * 1024 * 1024,
            tile_map_mask: passes::tile::compilable_tile_maps(backend),
            // FALSE, and the name is the trap. `native_mxfp4_moe` does not mean
            // "reads MXFP4"; it means "has a native MXFP4 *GEMM*", which in
            // gpt-oss's contract selects a Marlin REPACK of the expert banks —
            // work this tree did not port. A driver whose GEMM reads the stored
            // banks directly wants the other branch, which is this one.
            native_mxfp4_moe: false,
            // No fused transcode kernels in this tree.
            fusion_mask: 0,
            // The dtype the encode kernels dequantize through, which decides
            // how many rows of scratch fit in the tile budget.
            encode_scratch_dtype: DType::BF16,
            block_scale_rows: 0,
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
    /// What this buffer's bytes ARE: the shape and encoding of the thing it
    /// holds, stated on the buffer itself.
    ///
    /// This used to be reachable only through [`tensor`](Self::tensor), and
    /// that indirection was the loader's largest silent defect. `tensor` says
    /// "this buffer IS that declared tensor", which an intermediate is not:
    /// the decoded operand a re-encode reads is nobody's tensor, so it had no
    /// type, so the compiler could not pick a kernel for a transform reading
    /// it and the executor could not even run one — `contract: buffer 2 has no
    /// tensor type` was a `Cast` of an internal tensor failing outright, on
    /// the host as well as the device (`.wiki/fix/loader.md` §3.3).
    ///
    /// The builder always had the answer. Every buffer is allocated from a
    /// [`TensorDecl`], `declared` or not; all that was missing was writing the
    /// type down when the buffer was not going to be bound by name.
    ///
    /// `tensor` stays, and stays `Option`, for the two things it actually
    /// means: what to publish this buffer as, and which declaration to
    /// finalize. Typing no longer asks it.
    pub ty: crate::contract::TensorType,
    pub bytes: u64,
    pub alignment: u32,
    pub temporary: bool,
    pub persistent_offset: Option<u64>,
    /// Where this buffer sits in the arena's SCRATCH region, if it is
    /// staging.
    ///
    /// Separate from [`persistent_offset`](Self::persistent_offset) because
    /// the two answer different questions, and three passes depend on the
    /// difference. `persistent_offset` means "this is a resident tensor, laid
    /// out" — [`spans::publish_spans`] publishes exactly those, and
    /// `rewrite::extent_write_as_bulk` turns exactly those writes into
    /// arena-absolute `BulkExtentWrite`s that `hoist_bulk_extent_writes` then
    /// moves to the front of the schedule. A staging buffer must be neither:
    /// it is not a tensor anyone names, and its write must stay where the
    /// transform that reads it is, because scratch is REUSED and a hoisted
    /// write would land in a slot another transform is still reading.
    ///
    /// Both are arena offsets, so anything asking merely "where is this
    /// buffer" asks [`arena_offset`](Self::arena_offset) and gets one answer.
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

    /// The dtype one element reads as — the logical one for a quantized
    /// encoding, which is what a transform over it sees.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.ty.encoding.dtype()
    }
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
    /// The backend entry point this transform runs as, when the target has one
    /// for these exact operands.
    ///
    /// Filled in by [`passes::tile::lower`](crate::plan::passes::tile::lower)
    /// against the target's own kernel table, and `None` when no row covers
    /// the operands — a dtype pair with no cast kernel, a uniform `Scale`
    /// where the kernel wants a per-group operand, an MXFP4 `Encode` whose
    /// width is not a multiple of its block. Those run on the host, and the
    /// plan now SAYS which ones will.
    ///
    /// This is the field that lets a backing stop deciding. A capability bit
    /// is per KIND and a kernel is per SHAPE, so a backing asked "can you do
    /// `Cast`" could only answer for the kind and then decline the operands at
    /// launch — which made "the loader transforms on the GPU" a claim you had
    /// to instrument a load to check. Naming the row moves that answer to
    /// compile time, where the tensor is still in hand to name in the refusal,
    /// and leaves the backing a lookup.
    ///
    /// A `String` rather than a `&'static str` because a plan is serialized and
    /// cached. The symbol is checked against the target's table when it is
    /// chosen, so a plan cannot name a row that does not exist; a plan read
    /// back from an older cache is refused by the compiler hash before it gets
    /// here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
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

    /// Does this plan publish ONE tensor for the embedding and the output
    /// projection?
    ///
    /// Decided once, by the contract, and read back here rather than decided
    /// a second time. The rule is that a shipped `lm_head` beats whatever the
    /// config says, and the config can be wrong in both directions:
    /// Qwen3.5-35B-A3B is a multimodal wrapper spelling `tie_word_embeddings`
    /// at the TOP level, outside the `text_config` its family parses, so the
    /// facts default to tied; Qwen3-0.6B says `tie_word_embeddings: true` and
    /// then ships an `lm_head.weight` anyway. Either way the contract staged
    /// `embed_tokens` and `lm_head` while the DAG asked for
    /// `shared_embedding`, and the load stopped on "unstaged weight
    /// shared_embedding.weight" — two opinions about one fact.
    ///
    /// The plan's own tensor list is the only opinion that cannot be wrong in
    /// a way the binding survives, so it is the one every family follows. It
    /// It is a method here rather than a helper in a driver because a driver
    /// asking the question had to spell `shared_embedding.weight` to ask it
    /// — a name four contract authors in `crates/model` produce and no
    /// driver owns. See [`TIED_EMBEDDING_NAME`] for what this does and does
    /// not close.
    #[must_use]
    pub fn ties_embeddings(&self) -> bool {
        self.tensors
            .iter()
            .any(|tensor| tensor.name == TIED_EMBEDDING_NAME)
    }

    /// The names of every tensor this plan leaves in MXFP4.
    ///
    /// A plan's job is to get bytes onto a device unchanged; what they MEAN
    /// is the binder's business, and this is how the plan tells it. A set of
    /// names rather than a flag because **a checkpoint need not be uniform**:
    /// `mlx-community/gpt-oss-20b-MXFP4-Q4` names 98 tensors as affine/64/4
    /// in its `quantization` block and leaves the expert banks out, so those
    /// take the top-level default — mxfp4, group 32. The block holds 122
    /// entries, not 98; the other 24 name the `mlp.router` gates at 64/**8**.
    /// So the one checkpoint carries THREE formats, and a set of names is the
    /// only shape of answer that can carry that.
    ///
    /// Reading a bank with the dense format is not a near miss. Every scale
    /// comes from the wrong offset and bf16 garbage is NaN more often than
    /// not: measured, a fire bound every name, ran all 484 statements, and
    /// produced NaNs from the first routed projection of layer 0 onward while
    /// every structural gate passed.
    ///
    /// A driver computing this itself has to match on `Encoding::Quant` and
    /// on the `QuantScheme` variant — two of this crate's enums, read
    /// structurally, in a crate that should be reading answers.
    #[must_use]
    pub fn mxfp4_tensor_names(&self) -> std::collections::HashSet<String> {
        self.tensors
            .iter()
            .filter(|t| {
                matches!(
                    &t.encoding,
                    Encoding::Quant(spec) if spec.scheme == QuantScheme::Mxfp4E2M1E8M0
                )
            })
            .map(|t| t.name.clone())
            .collect()
    }

    /// Every DISTINCT affine point this plan's tensors arrive at.
    ///
    /// `(group_size, bits_per_element)` for each tensor whose encoding is
    /// affine — which is to say quantized and not MXFP4, since a bank in
    /// MXFP4 takes its own kernel at its own group and is not read at an
    /// affine point at all.
    ///
    /// # Why a SET, and why the plan is the one that knows
    ///
    /// For the same reason [`Self::mxfp4_tensor_names`] is a set of names:
    /// **a checkpoint need not be uniform.** `mlx_lm` publishes a routed
    /// stack at 4 bits and its ROUTER GATE at 8, because the gate is a
    /// small tensor whose error the whole mixture inherits — every token
    /// routed to almost the right experts. That is not a fault. It is a
    /// fluent model answering wrongly, measured at cosine 0.84 against the
    /// reference logits.
    ///
    /// A driver cannot see this. It is handed the checkpoint's
    /// `config.json` point — ONE `(group, bits)` — and builds one kernel
    /// set from it, so a second point in the tensors is read at the first
    /// and nothing anywhere says so. The plan is where the per-tensor
    /// `QuantSpec` lives, so the plan is what can be asked.
    ///
    /// Sorted, so a caller comparing two plans or printing a refusal gets
    /// a stable answer rather than a hash order.
    #[must_use]
    pub fn affine_points(&self) -> Vec<(u32, u32)> {
        let mut points: Vec<(u32, u32)> = self
            .tensors
            .iter()
            .filter_map(|t| match &t.encoding {
                Encoding::Quant(spec)
                    if spec.scheme != QuantScheme::Mxfp4E2M1E8M0 && spec.group_size > 0 =>
                {
                    Some((spec.group_size, u32::from(spec.bits_per_element)))
                }
                _ => None,
            })
            .collect();
        points.sort_unstable();
        points.dedup();
        points
    }

    /// Each distinct affine point beside ONE tensor that arrives at it.
    ///
    /// [`Self::affine_points`] is a count, and a count is what a refusal
    /// cannot act on: an operator told a checkpoint "arrives at 2 affine
    /// points (g64/b4, g64/b8)" learns that it is refused and nothing about
    /// what to do next. `gpt-oss-20b-MXFP4-Q4`'s second point belongs to its
    /// 24 `mlp.router` gates and to nothing else, and a message that says so
    /// is the difference between a dead end and a named one.
    ///
    /// The witness is the FIRST tensor at each point in declaration order,
    /// which for every checkpoint this has been run against is the earliest
    /// layer's — a name an operator can look up in the index.
    ///
    /// Sorted by point, for the same reason [`Self::affine_points`] is.
    /// Every affine tensor's point, by name.
    ///
    /// [`Self::affine_point_of`] answers one name in a scan of the whole
    /// declaration list; a driver that must answer several — and that must
    /// not read [`Encoding`] and [`QuantScheme`] structurally to do it, which
    /// is the coupling `mxfp4_tensor_names` exists to avoid — takes the map
    /// once and asks it.
    #[must_use]
    pub fn affine_by_name(&self) -> std::collections::HashMap<String, (u32, u32)> {
        self.tensors
            .iter()
            .filter_map(|t| match &t.encoding {
                Encoding::Quant(spec)
                    if spec.scheme != QuantScheme::Mxfp4E2M1E8M0 && spec.group_size > 0 =>
                {
                    Some((
                        t.name.clone(),
                        (spec.group_size, u32::from(spec.bits_per_element)),
                    ))
                }
                _ => None,
            })
            .collect()
    }

    /// The affine point ONE named tensor arrives at, if it is affine.
    ///
    /// The by-name form of [`Self::affine_points`], for a driver that must
    /// know not just how many points a checkpoint holds but WHICH tensor
    /// holds which. `driver-metal` puts two names to this — the expert bank
    /// and the router gate — and a checkpoint that answers a third point for
    /// anything else is one it refuses.
    ///
    /// `None` for a tensor that is absent, raw, or MXFP4: none of the three
    /// is read at an affine point.
    #[must_use]
    pub fn affine_point_of(&self, name: &str) -> Option<(u32, u32)> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .and_then(|t| match &t.encoding {
                Encoding::Quant(spec)
                    if spec.scheme != QuantScheme::Mxfp4E2M1E8M0 && spec.group_size > 0 =>
                {
                    Some((spec.group_size, u32::from(spec.bits_per_element)))
                }
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
            if spec.scheme == QuantScheme::Mxfp4E2M1E8M0 || spec.group_size == 0 {
                continue;
            }
            let point = (spec.group_size, u32::from(spec.bits_per_element));
            if !out.iter().any(|(p, _)| *p == point) {
                out.push((point, t.name.clone()));
            }
        }
        out.sort_unstable_by_key(|(p, _)| *p);
        out
    }
}

/// The one name a contract publishes when the embedding and the output
/// projection are the same tensor.
///
/// **This crate does not emit it.** Four contract authors in `crates/model`
/// do — `llama_3`, `qwen_3_5` and `gemma_4` each `format!` it — and the
/// dependency runs `model` → `model-loader`, so nothing links their spelling
/// to this one. A constant here does not fix that; what it fixes is the
/// smaller thing, that a **driver** asking whether a plan ties its
/// embeddings no longer has to spell the name to find out.
///
/// The larger gap is real and stays open: rename the tied tensor in those
/// authors and this constant goes quietly stale.
pub const TIED_EMBEDDING_NAME: &str = "shared_embedding.weight";

#[cfg(test)]
mod plan_query_tests {
    use super::*;
    use crate::types::{QuantSpec, Visibility};

    fn decl(name: &str, encoding: Encoding) -> TensorDecl {
        TensorDecl {
            id: TensorId(0),
            name: name.to_string(),
            shape: vec![32, 8],
            encoding,
            alignment: 256,
            visibility: Visibility::Public,
        }
    }

    #[test]
    fn tied_embeddings_are_read_off_the_plan_not_the_config() {
        let mut plan = LoadPlan::empty(StorageTarget::default());
        assert!(!plan.ties_embeddings());
        plan.tensors
            .push(decl(TIED_EMBEDDING_NAME, Encoding::Raw(DType::BF16)));
        assert!(plan.ties_embeddings());
    }

    fn affine(name: &str, group: u32, bits: u8) -> TensorDecl {
        decl(
            name,
            Encoding::Quant(QuantSpec {
                scheme: QuantScheme::MlxAffineU4,
                logical_dtype: DType::BF16,
                bits_per_element: bits,
                group_size: group,
                channel_axis: None,
            }),
        )
    }

    /// The router gate published at a width the rest of the stack is not.
    ///
    /// `mlx_lm` does this deliberately: the gate is a small tensor whose
    /// error the WHOLE mixture inherits, so it is published at 8 bits
    /// inside a 4-bit stack. A driver handed one `(group, bits)` off
    /// `config.json` reads it at 4 and the mixture routes each token to
    /// almost the right experts — cosine 0.84 against the reference
    /// logits, and not one NaN to notice it by.
    ///
    /// The plan knew all along; nothing asked. This is the asking.
    #[test]
    fn a_stack_that_publishes_its_router_gate_wider_says_so() {
        let mut plan = LoadPlan::empty(StorageTarget::default());
        assert!(
            plan.affine_points().is_empty(),
            "an empty plan arrives at no affine point"
        );

        plan.tensors.push(affine("layer.0.q_proj.weight", 64, 4));
        plan.tensors.push(affine("layer.0.k_proj.weight", 64, 4));
        assert_eq!(
            plan.affine_points(),
            vec![(64, 4)],
            "a uniform stack is ONE point, however many tensors carry it"
        );

        plan.tensors.push(affine("layer.0.router.gate", 64, 8));
        assert_eq!(
            plan.affine_points(),
            vec![(64, 4), (64, 8)],
            "the gate's width is a second point, and sorted so a refusal \
             prints the same way twice"
        );

        // An MXFP4 bank is NOT a second affine point: it takes its own
        // kernel at its own group and is never read at one. Without this
        // exclusion every gpt-oss checkpoint would refuse.
        plan.tensors.push(decl(
            "layer.0.experts.gate_up",
            Encoding::Quant(QuantSpec {
                scheme: QuantScheme::Mxfp4E2M1E8M0,
                logical_dtype: DType::BF16,
                bits_per_element: 0,
                group_size: 32,
                channel_axis: None,
            }),
        ));
        assert_eq!(
            plan.affine_points(),
            vec![(64, 4), (64, 8)],
            "an mxfp4 bank is not read at an affine point"
        );

        // Nor is an unquantized tensor.
        plan.tensors
            .push(decl("layer.0.norm.weight", Encoding::Raw(DType::BF16)));
        assert_eq!(plan.affine_points(), vec![(64, 4), (64, 8)]);

        // AND WHICH TENSOR MADE IT SO. The count above says a driver cannot
        // serve this checkpoint; only the witness says the obstacle is the
        // router gate, which is the sentence that names the next piece of
        // work rather than ending the conversation.
        assert_eq!(
            plan.affine_point_witnesses(),
            vec![
                ((64, 4), "layer.0.q_proj.weight".to_string()),
                ((64, 8), "layer.0.router.gate".to_string()),
            ],
            "each point beside the FIRST tensor declared at it, and neither \
             the mxfp4 bank nor the raw norm is a witness to anything"
        );
    }

    #[test]
    fn a_mixed_checkpoint_names_only_the_banks_it_left_in_mxfp4() {
        // The case this exists for: gpt-oss-20b-MXFP4-Q4 quantizes 98 dense
        // tensors as affine/64/4 and leaves the expert banks at the top-level
        // mxfp4 default. A flag would have to pick one answer for both.
        let mut plan = LoadPlan::empty(StorageTarget::default());
        plan.tensors
            .push(decl("layer.0.mlp.weight", Encoding::Raw(DType::BF16)));
        plan.tensors.push(decl(
            "layer.0.experts.gate_up",
            Encoding::Quant(
                QuantSpec {
                    scheme: QuantScheme::Mxfp4E2M1E8M0,
                    logical_dtype: DType::BF16,
                    bits_per_element: 0,
                    group_size: 0,
                    channel_axis: None,
                }
                .normalized(),
            ),
        ));
        let names = plan.mxfp4_tensor_names();
        assert_eq!(names.len(), 1, "only the bank is mxfp4: {names:?}");
        assert!(names.contains("layer.0.experts.gate_up"));
    }
}
