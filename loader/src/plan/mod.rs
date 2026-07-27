//! The plan: what the loader emits, and the passes that shape it.
//!
//! `build` turns a checked contract into instructions; `passes` rewrites them
//! into the form the executor wants; `mod.rs` is only the vocabulary the two
//! share.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::types::{
    BackendKind, BufferId, CheckpointFormat, DType, Encoding, FileId, InstrId, QuantScheme,
    RepackSpec, TensorDecl, TensorId,
};

pub mod build;
pub(crate) mod geometry;
pub mod index;
pub mod pass;
pub mod passes;

pub use crate::extent::{Dim, Extent};
pub use passes::tile::{CUDA_TILE_MAP_MASK, HOST_TILE_MAP_MASK, METAL_TILE_MAP_MASK};

/// Which tile-map transforms a target's kernels implement.
///
/// Defined here, not in `ffi/`: the plan is the thing that has transforms, and
/// the ABI is a view of the plan. The arrow used to point the other way —
/// `load_plan.rs` imported `crate::ffi::types::PIE_LOADER_TILE_MAP_*` — which
/// made the core depend on its own serialization format. `ffi/types.rs` now
/// restates these under the C names and a `const` assertion pins the two
/// together, because cbindgen emits literals and cannot follow a path.
pub const TILE_MAP_CAST: u32 = 1 << 0;
pub const TILE_MAP_DECODE: u32 = 1 << 1;
pub const TILE_MAP_ENCODE: u32 = 1 << 2;
pub const TILE_MAP_TRANSCODE: u32 = 1 << 3;
pub const TILE_MAP_REBLOCK: u32 = 1 << 4;
pub const TILE_MAP_REORDER: u32 = 1 << 5;
pub const TILE_MAP_REPACK: u32 = 1 << 6;

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
    let contract =
        crate::contract::rewrite::coalesce_direct_row_shards(contract, metadata, &target)?;
    let mut plan = build::build(metadata, &contract, target)?;
    plan.passes = pass::run_all(&mut plan)?;
    // Runs last, so a plan is never observable in a state where its tiling and
    // fusion fields are still placeholders.
    passes::tile::lower(&mut plan);
    plan.attachments = derive_quant_attachments(&plan.tensors);
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
    /// executor is what makes the choice part of the plan, and therefore part of
    /// the plan hash (§8.1).
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
/// was an *unwritten* contract: the loader enumerated the shards one way and the
/// driver re-enumerated them another, and nothing checked that the two agreed on
/// which file index 3 was. Both sides sorted, so they did agree — but by
/// coincidence of two implementations, not by construction (§6).
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

/// How a scale tensor's entries map onto the tensor they scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantGranularity {
    /// One scale per row of `channel_axis`.
    PerChannel,
    /// One scale per `group_size` elements along the axis after `channel_axis`.
    PerGroup,
}

/// What the driver's kernels expect a scale tensor to hold by the time they read
/// it.
///
/// The distinction is not derivable from the scale tensor itself — its dtype says
/// how the bytes are stored, not how the kernel wants them — so the loader, which
/// chose the encoding, states it. The driver used to infer it from
/// `group_size == 32`, which was true only because MXFP4 is the one scheme with
/// that group size today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleForm {
    /// Consumed as raw E8M0 exponent bytes. The MXFP4 GEMM, the dequant kernels
    /// and `make_expert_weight_view` all require U8 and assert on anything else.
    RawE8M0,
    /// Consumed as F32 multipliers. Whatever the scales were stored as (E8M0
    /// bytes, BF16, or F32 already) is expanded before the GEMM sees them.
    F32Factors,
}

/// A quantized tensor and the tensor holding its scales.
///
/// The two are separate runtime tensors — the driver materializes both and then
/// has to know they belong together in order to attach the quant metadata its
/// kernels read. That pairing used to be re-derived driver-side by matching name
/// suffixes against the plan's tensor list, which guessed at a fact the loader
/// already had: `frontend.rs` names the scale tensor when it creates it, and the
/// [`QuantSpec`](crate::types::QuantSpec) it came from carries the granularity.
/// Stating it in the plan removes the guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantAttachment {
    pub tensor: TensorId,
    pub scale_tensor: TensorId,
    pub granularity: QuantGranularity,
    pub group_size: u32,
    pub channel_axis: u32,
    pub scale_form: ScaleForm,
}

/// Pair every quantized runtime tensor with the runtime tensor holding its
/// scales.
///
/// Two shapes of pairing exist, and they are found differently:
///
/// * **The loader quantized it.** The tensor carries [`Encoding::Quant`], and
///   `frontend.rs::quant_metadata_outputs` created the scale tensor and chose
///   its name. The name formula below is that function's, restated — which is
///   why the two must move together.
/// * **The checkpoint shipped it quantized.** The tensor is a plain
///   [`DType::F8E4M3`] passthrough whose scales came out of the checkpoint
///   under the checkpoint's own name, so there is nothing more authoritative to
///   consult than the name.
///
/// The granularity, group size and channel axis are fixed per shape rather than
/// read from the [`QuantSpec`](crate::types::QuantSpec): they describe the
/// *scale tensor's* layout, which the encode kernels fix, not the quantized
/// tensor's.
pub fn derive_quant_attachments(tensors: &[TensorDecl]) -> Vec<QuantAttachment> {
    use std::collections::HashMap;

    let by_name: HashMap<&str, TensorId> = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.id))
        .collect();

    let mut out = Vec::new();
    for tensor in tensors {
        let mut attach = |scale: &str, granularity, group_size, channel_axis, scale_form| {
            let Some(&scale_tensor) = by_name.get(scale) else {
                return false;
            };
            out.push(QuantAttachment {
                tensor: tensor.id,
                scale_tensor,
                granularity,
                group_size,
                channel_axis,
                scale_form,
            });
            true
        };

        if let Encoding::Quant(spec) = &tensor.encoding {
            match spec.scheme {
                QuantScheme::Fp8E4M3 | QuantScheme::Int8Symmetric => {
                    // `[rows]` of F32, one per output channel.
                    attach(
                        &format!("{}_scale_inv", tensor.name),
                        QuantGranularity::PerChannel,
                        0,
                        0,
                        ScaleForm::F32Factors,
                    );
                }
                QuantScheme::Mxfp4E2M1E8M0 => {
                    // `[rows, cols/32]` of E8M0 bytes, one per 32-element block
                    // along K, and the MXFP4 kernels want those bytes as they
                    // are.
                    attach(
                        &format!("{}_scale", tensor.name),
                        QuantGranularity::PerGroup,
                        32,
                        1,
                        ScaleForm::RawE8M0,
                    );
                }
                _ => {}
            }
            continue;
        }

        // A checkpoint-quantized weight copied through untouched. Block-scaled
        // FP8 (DeepSeek-V3 and its descendants) is the only such format the
        // driver attaches metadata for, and it spells the scales either way.
        if tensor.dtype() != DType::F8E4M3 {
            continue;
        }
        let Some(base) = tensor.name.strip_suffix(".weight") else {
            continue;
        };
        let _ = attach(
            &format!("{}_scale_inv", tensor.name),
            QuantGranularity::PerGroup,
            128,
            0,
            ScaleForm::F32Factors,
        ) || attach(
            &format!("{base}.scale"),
            QuantGranularity::PerGroup,
            128,
            0,
            ScaleForm::F32Factors,
        );
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceExtent {
    pub file_id: FileId,
    pub tensor_id: TensorId,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub stride: Extent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DestExtent {
    pub buffer: BufferId,
    pub offset: u64,
    pub stride: Extent,
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

impl TileMapKind {
    pub const fn capability_bit(self) -> u32 {
        match self {
            Self::Cast => TILE_MAP_CAST,
            Self::Decode => TILE_MAP_DECODE,
            Self::Encode => TILE_MAP_ENCODE,
            Self::Transcode => TILE_MAP_TRANSCODE,
            Self::Reblock => TILE_MAP_REBLOCK,
            Self::Reorder => TILE_MAP_REORDER,
            Self::Repack => TILE_MAP_REPACK,
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
/// cannot claim to determine execution (§8.1).
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
    pub repack: RepackSpec,
    pub scratch_bytes: u64,
    pub fusion: TransformFusion,
    /// The checkpoint tensor holding this transform's *input* block scales.
    ///
    /// A block-scaled FP8 source is two tensors on disk: the payload and a
    /// sibling of per-group factors. Only the payload is named by the contract,
    /// so the executor used to rebuild the sibling's name by appending
    /// `_scale_inv` and look it up — the same guess-the-loader's-answer
    /// anti-pattern `attachments` removed from the output side (§12 row 10).
    /// The loader reads the tensor table, so the loader answers.
    pub metadata_source: Option<TensorId>,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Axis, QuantScheme, QuantSpec, TensorId};

    fn tensor(id: u32, name: &str, encoding: Encoding) -> TensorDecl {
        TensorDecl {
            id: TensorId(id),
            name: name.to_string(),
            shape: vec![8, 32],
            encoding,
            alignment: 256,
        }
    }

    fn quant(scheme: QuantScheme, group_size: u32) -> Encoding {
        Encoding::Quant(QuantSpec {
            scheme,
            logical_dtype: DType::BF16,
            bits_per_element: 0,
            group_size,
            channel_axis: Some(Axis(0)),
            scale_dtype: Some(DType::U8),
            zero_point_dtype: None,
            block_shape: Vec::new(),
        })
    }

    /// The MXFP4 GEMM asserts its scale operand is U8, so a plan that asks the
    /// driver to expand these to F32 makes the kernel reject the load. Nothing
    /// else pins this: the synthetic MXFP4 goldens are checkpoint-quantized
    /// passthroughs, which produce no attachment at all.
    #[test]
    fn mxfp4_scales_stay_raw_e8m0() {
        let tensors = vec![
            tensor(0, "w.weight", quant(QuantScheme::Mxfp4E2M1E8M0, 32)),
            tensor(1, "w.weight_scale", Encoding::Raw(DType::U8)),
        ];
        let attachments = derive_quant_attachments(&tensors);
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].scale_form, ScaleForm::RawE8M0);
        assert_eq!(attachments[0].group_size, 32);
    }

    #[test]
    fn runtime_quantized_scales_are_f32_factors() {
        let tensors = vec![
            tensor(0, "w.weight", quant(QuantScheme::Fp8E4M3, 0)),
            tensor(1, "w.weight_scale_inv", Encoding::Raw(DType::F32)),
        ];
        let attachments = derive_quant_attachments(&tensors);
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].scale_form, ScaleForm::F32Factors);
        assert_eq!(attachments[0].granularity, QuantGranularity::PerChannel);
    }

    /// Block-scaled FP8 arrives already quantized, so the pairing is found by
    /// name rather than by encoding — but its scales still reach the GEMM as
    /// F32 multipliers.
    #[test]
    fn checkpoint_block_scaled_fp8_is_f32_factors() {
        let tensors = vec![
            tensor(0, "w.weight", Encoding::Raw(DType::F8E4M3)),
            tensor(1, "w.weight_scale_inv", Encoding::Raw(DType::F32)),
        ];
        let attachments = derive_quant_attachments(&tensors);
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].scale_form, ScaleForm::F32Factors);
        assert_eq!(attachments[0].group_size, 128);
    }
}
