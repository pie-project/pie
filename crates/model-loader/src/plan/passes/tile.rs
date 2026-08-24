//! Backend tile-map lowering.
//!
//! `architecture.md` §9 draws the line this file sits on. A fact the driver can
//! *measure* is data and travels in [`StorageTarget`]; a rule that differs per
//! backend and cannot be parameterized is code and lives here. It is the same
//! split LLVM draws between target features and `TargetLowering`.
//!
//! Everything here answers a question the C++ driver used to answer at run
//! time. That mattered less for being wrong than for being *unrecorded*: the
//! C++ transcode engine (`driver-cuda/csrc/src/loader/transcode_engine.hpp`,
//! deleted in `2cc4e5e4d` — cited as provenance, not as a path to open) chose a
//! tile size and a fusion strategy while executing, so one plan could run two
//! different kernel sequences and nothing in the plan said which
//! (`architecture.md` §8.1).
//! Deciding here puts the answer in the plan, which puts it in the plan hash,
//! which makes "the plan determines execution" true rather than aspirational.
//!
//! It used to be a `trait Backend` with three implementations, of which two
//! returned a constant and the third was one function. A trait is the right
//! shape for a decision with several *independent* implementations; this is one
//! decision with one implementation and two abstentions, so it is a `match`.

use crate::plan::index::PlanIndex;
use crate::plan::{
    FUSION_FP8_TO_MXFP4, LoadPlan, SourceExtent, StorageInstr, StorageTarget, TILE_MAP_BIAS,
    TILE_MAP_CAST, TILE_MAP_DECODE, TILE_MAP_ENCODE, TILE_MAP_REBLOCK, TILE_MAP_SCALE, TileMapKind,
    TransformFusion,
};
use crate::types::{BackendKind, BufferId, DType, Encoding, QuantScheme};

/// The kernel rows a load may run on the device, by table symbol.
///
/// Named here rather than reached for out of `kernels-cuda`, because the
/// plan for a CUDA target is compiled on machines that have no CUDA at all —
/// the import path does it, and so does every test. A plan is a claim about
/// what a device will do, and making that claim must not require the device's
/// toolchain.
///
/// What keeps them honest is the other side: with `feature = "cuda"` on,
/// `executor::cuda` calls exactly these as the typed `x::quant` host programs
/// they are, so a symbol that stopped existing fails that build rather than
/// becoming a plan nothing can run.
///
/// THE FOUR ARE NOT ALL THE SAME KIND OF NAME. The first two are `kernels-
/// cuda` ROUTINES and a test resolves each against `kernels_cuda::routine`.
/// The two quantisers are not routines and must not be looked for there: no
/// trace states a load-time weight transform, so they are plain `unsafe fn`s
/// and these strings are the LOADER's own vocabulary — the word a plan
/// carries from `tile` to `executor::cuda`'s dispatch, and nothing wider.
///
/// THE FIRST TWO CARRY NO DTYPE, and the constants' own names do. That is
/// not a mismatch: both routines are GENERIC (`cast_fp32_to<T>`,
/// `scale_rows<T>`), so the registry symbol is the generic one and the
/// instantiation is chosen where the routine is called, which for this
/// loader is `bf16` and always has been. The constant is named for the
/// instantiation the loader pins; the string is named for the row it must
/// resolve against. They were both spelled `..._bf16` while the routines
/// were monomorphic, and the string is what had to change when they stopped
/// being.
///
/// The typed call and the string are two halves of one claim, and only
/// together: the call is checked by the compiler and does not know what this
/// constant says; the constant is what a plan carries and the compiler cannot
/// read it. `executor::cuda`'s test is where the two meet.
pub const CUDA_CAST_FP32_TO_BF16: &str = "quant::cast_fp32_to";
pub const CUDA_SCALE_ROWS_BF16: &str = "quant::scale_rows";
pub const CUDA_QUANTIZE_BF16_TO_MXFP4: &str = "quant::quantize_bf16_to_mxfp4_e2m1_per_block";
pub const CUDA_QUANTIZE_BF16_TO_FP8: &str = "quant::quantize_bf16_to_fp8_e4m3_per_channel";

/// The transforms `driver/cuda`'s kernels implement.
///
/// `Repack` and `Reblock` were here and are not, and the correction matters
/// because this constant is load-bearing twice over: it decides which plans
/// COMPILE for a CUDA target, and — since a transform reaching the executor
/// with no host implementation has nowhere to go — which of those can run.
///
/// Both were what the C++ `transcode_engine.hpp` had device kernels for. This
/// tree did not port them, and `Repack` has no host implementation either, so
/// claiming it meant a checkpoint needing one compiled cleanly and then failed
/// at execution with nothing but a kind to name. Refusing it here refuses it
/// with the TENSOR named.
///
/// The driver used to restate a narrower mask of its own and a test compared
/// the two. There is nothing to compare now: this is the one statement, and
/// `StorageTarget::for_backend` is how a driver gets it.
///
/// What a CUDA arena actually launches is narrower still and is not a mask:
/// the plan names a kernel per instruction, and the backing looks it up.
///
/// `TILE_MAP_DECODE` is here because an archive may hold a self-contained
/// block. It is not a device kernel and does not claim to be: none of these
/// three drivers implements `run_tile_map`, so a decode in one of their plans
/// runs on the HOST, streaming into the arena — slower than a load whose bytes
/// are already plain, and the price of an archive that kept its source
/// packing. The three masks below say the same thing about their own backends;
/// this is the copy that states it.
pub const CUDA_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_ENCODE | TILE_MAP_SCALE | TILE_MAP_DECODE;

/// The transforms a plan for a Metal target may CARRY.
///
/// `SCALE` decodes a block-scaled scheme to values and `CAST` re-encodes them as
/// the affine-U4 that driver's matvecs read, which is what lets it load the
/// published MXFP4 gpt-oss checkpoint directly. There is no repacking or
/// reblocking, so those bits stay clear.
///
/// It said "the transforms `driver/metal`'s load-time kernels implement,
/// mirrored in C++ as `kMetalTileMapMask`". Three things were wrong with that.
/// The C++ was deleted in `2cc4e5e4d` and `kMetalTileMapMask` appears nowhere
/// but in the sentence claiming to mirror it. `driver-metal`'s backing
/// implements neither [`ArenaBacking::runs_named_kernels`] nor `run_tile_map`,
/// so it takes the default — no — and EVERY transform in a Metal plan runs on
/// the host. And this is the compile-time question anyway: what a device
/// actually runs is a property of the backing you were handed, which is what
/// `runs_named_kernels` is. This constant is the first, and no longer wears
/// the name of the second.
///
/// [`ArenaBacking::runs_named_kernels`]: crate::executor::arena::ArenaBacking::runs_named_kernels
pub const METAL_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_ENCODE | TILE_MAP_SCALE | TILE_MAP_DECODE;

/// The transforms a Vulkan plan may carry.
///
/// The same three Metal's does, and for the same reason rather than by
/// imitation: `driver-vulkan` implements neither
/// [`ArenaBacking::runs_named_kernels`] nor `run_tile_map` either, so every
/// transform in a Vulkan plan runs on the host, and this is the compile-time
/// question of which a plan may CONTAIN.
///
/// Equal to Metal's today and written out rather than aliased to it. They are
/// two answers that happen to agree, and the day a load-time compute kernel
/// is added to one of these drivers is the day an alias would quietly change
/// the other.
///
/// [`ArenaBacking::runs_named_kernels`]: crate::executor::arena::ArenaBacking::runs_named_kernels
pub const VULKAN_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_ENCODE | TILE_MAP_SCALE | TILE_MAP_DECODE;

/// The transforms `host_executor` implements. Not a device capability: it is
/// what a plan compiled for no device may carry, which is the reference the
/// device answers are checked against.
pub const HOST_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_REBLOCK | TILE_MAP_SCALE | TILE_MAP_BIAS;

/// The mask offline conversion compiles against: everything `replay` runs plus
/// `Encode` and `Decode`, which the host executor implements for exactly that
/// command.
///
/// A separate constant rather than a wider [`HOST_TILE_MAP_MASK`] because the
/// two answer different questions. HOST is the verification surface — what a
/// `replay` of a *device* plan may be asked to reproduce — and widening it
/// would quietly change which plans `replay` accepts. Conversion is its own
/// target: a plan compiled against this mask is meant to be executed on the
/// host and its output written back to a checkpoint, not compared against a
/// device.
///
/// `Decode` was here and in no device mask, on the grounds that the schemes it
/// covers carry their scales inside the payload (GGUF blocks), which no device
/// kernel reads. The first half is still true and the conclusion was not: a
/// device mask says which transforms a plan for that backend may CARRY, not
/// which its kernels run, and a decode in a Cuda/Metal/Vulkan plan runs on the
/// host like every other transform those drivers do not implement. Refusing it
/// here refused an archive that kept its source packing, which is a much
/// larger thing than it was meant to protect — so the three device masks
/// admit it now, and `validate_bound_encodings` enforces the fact that
/// actually matters: a device is never HANDED a block.
pub const CONVERT_TILE_MAP_MASK: u32 = HOST_TILE_MAP_MASK | TILE_MAP_ENCODE | TILE_MAP_DECODE;

/// Which transforms a plan compiled for `backend` may CARRY.
///
/// A compile-time property, and the only statement of it: `StorageTarget::for_backend`
/// is how a driver gets this, and the lowering below refuses an instruction
/// whose kind falls outside it with the TENSOR named rather than leaving it to
/// fail at dispatch with nothing but a kind.
///
/// It is deliberately not "what the device runs". That is a property of the
/// [`ArenaBacking`](crate::executor::arena::ArenaBacking) the caller handed
/// over — `runs_named_kernels`, one bit — and it can be narrower than this on
/// any given load: device transforms turned off, a build without the `cuda`
/// feature, a driver that supplied a plain `&mut [u8]`. Every transform this
/// admits also has a host implementation, which is the property
/// `mask_tests::every_device_transform_has_a_host_implementation` pins, and it
/// is what makes the narrower case a slower load rather than a failed one.
///
/// A paragraph here used to specify a cross-check against
/// `PieLoaderTargetSpec::tile_map_mask` in `storage_target`, calling it
/// load-bearing, thirty-five lines below a paragraph saying there was nothing
/// left to compare. Neither name exists anywhere in the tree. The paragraph
/// above was the true one.
pub fn compilable_tile_maps(backend: BackendKind) -> u32 {
    match backend {
        BackendKind::Cuda => CUDA_TILE_MAP_MASK,
        BackendKind::Metal => METAL_TILE_MAP_MASK,
        BackendKind::Vulkan => VULKAN_TILE_MAP_MASK,
        BackendKind::Unknown => HOST_TILE_MAP_MASK,
    }
}

#[cfg(test)]
mod mask_tests {
    use super::*;

    /// Every transform a device plan may carry has a host implementation.
    ///
    /// This lived in `driver-cuda`, comparing the driver's mask against
    /// `CONVERT_TILE_MAP_MASK`. It comes here with the mask it was checking,
    /// and it is the property that survives rather than the drift check beside
    /// it — that one compared two statements of one fact, and there is one
    /// statement now.
    ///
    /// What it buys: turning the device path off — `PIE_LOADER_DEVICE_TRANSFORMS=0`,
    /// a build without the `cuda` feature, or a plan that named no kernel for
    /// an instruction — always lands somewhere that can finish the load. It is
    /// also what lets one plan run into a host arena and a device arena and
    /// the bytes be compared, which is the only check a load-time kernel gets.
    #[test]
    fn every_transform_a_backend_may_lower_has_a_host_implementation() {
        for backend in [
            BackendKind::Cuda,
            BackendKind::Metal,
            BackendKind::Vulkan,
            BackendKind::Unknown,
        ] {
            assert_eq!(
                compilable_tile_maps(backend) & !crate::plan::CONVERT_TILE_MAP_MASK,
                0,
                "{backend:?} may lower a transform with no host implementation, \
                 which leaves the load with no fallback and no reference to \
                 check a device answer against"
            );
        }
    }

    /// Vulkan is a backend of its own and not the unknown one.
    ///
    /// It matters because `Unknown` is the arm a missing backend falls into,
    /// and its mask does NOT admit `TILE_MAP_ENCODE` -- a quantised
    /// checkpoint compiled against it produces a plan that carries no encode
    /// instruction, which loads and is wrong. `driver-vulkan` asked for
    /// Metal's target for exactly this reason before it had one of its own,
    /// with a note saying so; this is that note as a check.
    #[test]
    fn the_vulkan_target_admits_what_a_quantised_load_needs() {
        let vulkan = compilable_tile_maps(BackendKind::Vulkan);
        assert_ne!(
            vulkan & TILE_MAP_ENCODE,
            0,
            "a Vulkan plan may not carry an encode, so a quantised checkpoint \
             compiles to a plan that silently skips it"
        );
        assert_ne!(
            vulkan,
            compilable_tile_maps(BackendKind::Unknown),
            "the Vulkan arm is the fallback one, which is what having an arm \
             was supposed to stop"
        );
    }

    /// A kind this crate can name a kernel for is a kind its mask claims.
    ///
    /// The other direction is fine and expected — `Reblock` is in the host
    /// mask with no device kernel anywhere. What must not happen is
    /// `cuda_kernel` returning a row for a kind `validate_target_support`
    /// would have refused, because then the plan could never contain the
    /// instruction the selection was written for.
    #[test]
    fn a_kind_the_cuda_selector_can_answer_is_a_kind_the_mask_allows() {
        for kind in [TileMapKind::Cast, TileMapKind::Scale, TileMapKind::Encode] {
            assert_ne!(
                CUDA_TILE_MAP_MASK & kind.capability_bit(),
                0,
                "`cuda_kernel` selects a row for {kind:?}, which the CUDA mask \
                 does not allow into a plan"
            );
        }
    }
}

/// Everything the lowering rule is allowed to see about one `TileMap`.
///
/// Deliberately not the instruction itself. The plan walking — resolving a
/// source tensor's dtype, recovering a flat buffer's logical shape — happens
/// once, below, so the rule contains only the rule. A rule that needed a fact
/// not listed here is a signal that the fact belongs in [`StorageTarget`], not
/// that this struct should grow a plan reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileMapFacts {
    pub kind: TileMapKind,
    pub transform_from: Option<QuantScheme>,
    pub transform_to: Option<QuantScheme>,
    /// The dtype the transform reads: the source tensor's for a checkpoint
    /// source, the input buffer's tensor dtype otherwise. `None` when neither
    /// resolves, which forces the conservative answer everywhere below.
    pub source_dtype: Option<DType>,
    pub has_source: bool,
    /// Whether the source extent is one contiguous run. A strided source cannot
    /// be sliced by rows without re-deriving the stride per tile, which is why
    /// the driver refused to tile those.
    pub compact_source: bool,
    /// Declared 2-D shape of the primary output, or `None` if it is not 2-D.
    ///
    /// Read from the *instruction*, not the target: the build may narrow the
    /// budget for one instruction, and a rule that consulted the target would
    /// silently ignore that.
    pub shape: Option<(u64, u64)>,
    pub max_tile_bytes: u64,
    /// The dtype the transform WRITES, when the primary output is unquantized.
    ///
    /// `None` for a quantized destination, whose scheme is
    /// [`transform_to`](Self::transform_to) instead — the two are not
    /// interchangeable, and a `Cast` is exactly the kind for which both sides
    /// are plain dtypes.
    pub dest_dtype: Option<DType>,
    /// Whether the transform rewrites its input where it lies.
    ///
    /// The per-row scale kernel multiplies IN PLACE, so a plan whose
    /// destination is a different buffer is not one it can run. Known here
    /// because the instruction names both buffers.
    pub in_place: bool,
    /// Whether a [`TileMapKind::Scale`] reads per-group factors from an
    /// operand, rather than multiplying by the uniform constant in
    /// `scale_factor_bits`.
    pub blocked_scale: bool,
    /// Whether every operand resolves to a span of the arena.
    ///
    /// A kernel runs on a device and a device reaches the arena, so this is a
    /// precondition of naming one rather than a preference. It is a FACT and
    /// not a check because the answer decides the plan: an operand the arena
    /// does not hold means `kernel = None`, which is the plan saying the host
    /// runs this one — visibly, in the field a reader looks at — instead of
    /// the executor discovering it at launch and quietly doing the same thing.
    pub operands_in_arena: bool,
}

/// What the lowering decided. Written into the instruction verbatim.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileLowering {
    /// Rows of the output the driver transforms per launch. `0` means "no
    /// tiling" — do the whole tensor in one pass — which is both the answer for
    /// instructions that cannot be tiled and the answer when one tile covers
    /// everything.
    pub rows_per_tile: u32,
    pub fusion: TransformFusion,
    /// The backend entry point these operands run as, or `None` for the host.
    ///
    /// See [`TransformSpec::kernel`](crate::plan::TransformSpec::kernel) for
    /// why the decision belongs here rather than in the backing that launches
    /// it.
    pub kernel: Option<&'static str>,
}

/// Fill in every backend decision the plan carries.
///
/// Runs over a finished plan, so a decision may depend on anything the plan
/// says. It only ever *writes* decision fields — never adds, removes, or
/// reorders instructions — which is what lets a backend that declines to decide
/// leave the plan bit-identical.
///
/// It is the LAST [`Stage::Rewrite`] pass, and being a pass at all is what
/// makes `validate-kernel-operands` possible: a check runs after every
/// rewrite, so a validator can only speak about `TransformSpec::kernel` if the
/// thing that fills it in has already run. It was a bare function called after
/// the pipeline, which put the plan's most consequential decision outside the
/// only machinery that checks the plan.
///
/// [`Stage::Rewrite`]: crate::plan::pass::Stage::Rewrite
pub fn lower(plan: &mut LoadPlan) -> usize {
    let target = plan.target.clone();
    let index = PlanIndex::new(plan);

    // Resolved up front: the write loop holds `&mut` on an instruction while
    // these lookups still need to read buffer and tensor declarations.
    let facts: Vec<Option<TileMapFacts>> = plan
        .instrs
        .iter()
        .map(|instr| tile_map_facts(plan, &index, instr))
        .collect();

    let mut named = 0;
    for (instr, facts) in plan.instrs.iter_mut().zip(facts) {
        let (
            Some(facts),
            StorageInstr::TileMap {
                tile, transform, ..
            },
        ) = (facts, instr)
        else {
            continue;
        };
        let lowering = lower_tile_map(&facts, &target);
        tile.rows_per_tile = lowering.rows_per_tile;
        transform.fusion = lowering.fusion;
        transform.kernel = lowering.kernel.map(str::to_string);
        named += usize::from(lowering.kernel.is_some());
    }
    named
}

/// [`lower`], as the pipeline runs it. The count is the kernels it named.
pub(super) fn lower_backend_tiling(plan: &mut LoadPlan) -> crate::error::Result<usize> {
    Ok(lower(plan))
}

fn lower_tile_map(facts: &TileMapFacts, target: &StorageTarget) -> TileLowering {
    // Nothing to name a row for: the operands are not where a kernel could
    // read them. Stated once here rather than in each backend's table, because
    // it is true of every device and of no host.
    let kernel = |chosen| facts.operands_in_arena.then_some(chosen).flatten();
    match target.backend {
        BackendKind::Cuda => TileLowering {
            kernel: kernel(cuda_kernel(facts)),
            ..cuda_encode(facts, target)
        },
        // Neither Metal nor Vulkan runs a transform: the host executor
        // derives its own tiling from `max_tile_bytes` at run time, which it
        // is allowed to do precisely because it is not the thing whose
        // execution the plan is supposed to determine (`architecture.md`
        // §10.3).
        BackendKind::Metal | BackendKind::Vulkan | BackendKind::Unknown => TileLowering::default(),
    }
}

/// Which row this target would run these operands as, or `None` for the host.
///
/// The kernel table, asked a question about operands that do not exist yet.
/// [`stage_device_transforms`] rewrites a transform so its operands are on the
/// device, and it has to know BEFORE rewriting whether the rewrite buys
/// anything — so it builds the facts the rewritten instruction would have and
/// asks here. Restating the table there instead is how the driver and the
/// loader came to hold two opinions about one device
/// (`.wiki/fix/loader.md` §3.1); there is one table and this is the way in.
///
/// [`stage_device_transforms`]: super::stage::stage_device_transforms
pub(crate) fn kernel_for(facts: &TileMapFacts, target: &StorageTarget) -> Option<&'static str> {
    lower_tile_map(facts, target).kernel
}

/// Everything the lowering rule reads about one instruction, extracted from
/// the plan.
///
/// `pub(crate)` for the staging pass, which asks [`kernel_for`] about a
/// modified copy of what this returns.
pub(crate) fn facts_of(
    plan: &LoadPlan,
    index: &PlanIndex,
    instr: &StorageInstr,
) -> Option<TileMapFacts> {
    tile_map_facts(plan, index, instr)
}

/// Which kernel row runs these operands, or `None` for the host.
///
/// **This is the table that used to live in the driver**, as a `run_tile_map`
/// that took the operands and answered `Ok(false)` when it had no kernel for
/// them. Two things were wrong with that, and neither was the code:
///
/// * a capability bit is per KIND and a kernel is per SHAPE, so the mask could
///   claim `Cast` and the launch still decline an F16 source. The decline was
///   correct and invisible — the load finished, the bytes were right, and the
///   transform had quietly run on the host.
/// * the facts it decided on are all in the plan. Deciding at launch meant
///   deciding without the tensor's name, so a refusal could say "this backing
///   has no kernel for these bytes" where the compiler can say which tensor.
///
/// Every rule below is the driver's own, moved and not rewritten. What changes
/// is when it runs.
fn cuda_kernel(facts: &TileMapFacts) -> Option<&'static str> {
    match facts.kind {
        // The one cast the table implements. Any other dtype pair is refused,
        // not approximated: a cast with no kernel must never become a copy,
        // because the bytes would be the source's representation under the
        // destination's name and no later stage can detect that.
        TileMapKind::Cast => (facts.source_dtype == Some(DType::F32)
            && facts.dest_dtype == Some(DType::BF16))
        .then_some(CUDA_CAST_FP32_TO_BF16),
        // `scale_rows` multiplies IN PLACE and reads its factors from an
        // operand. A uniform factor has no operand to read and the table has no
        // scalar-multiply row; a destination that is not the source is not what
        // the kernel does.
        TileMapKind::Scale => (facts.blocked_scale
            && facts.in_place
            && facts.source_dtype == Some(DType::BF16)
            && facts.dest_dtype == Some(DType::BF16)
            && facts.shape.is_some())
        .then_some(CUDA_SCALE_ROWS_BF16),
        // No CUDA row, and the mask says so too. A bias is what an import
        // needs to reconcile a checkpoint format's constant with pie's
        // kernel, and an import runs on the host; a device plan that asks for
        // one is a contract that wandered, and it should be refused by name
        // rather than answered by a host fallback nobody asked for.
        TileMapKind::Bias => None,
        // Runtime quantization. Both rows want a bf16 source and a 2-D shape;
        // the MXFP4 one additionally wants a width that is a whole number of
        // its 32-element block, and refuses to guess otherwise.
        //
        // Every other target, including the fused FP8->MXFP4 that
        // `transcode.cu` implements, stays on the host: reaching it needs the
        // SOURCE's block scales, which an `Encode` whose input is already a
        // bf16 buffer does not have.
        TileMapKind::Encode => {
            let (_, cols) = facts.shape?;
            if facts.source_dtype != Some(DType::BF16) {
                return None;
            }
            match facts.transform_to {
                Some(QuantScheme::Mxfp4E2M1E8M0) => {
                    (cols % 32 == 0).then_some(CUDA_QUANTIZE_BF16_TO_MXFP4)
                }
                Some(QuantScheme::Fp8E4M3) => Some(CUDA_QUANTIZE_BF16_TO_FP8),
                _ => None,
            }
        }
        TileMapKind::Decode
        | TileMapKind::Transcode
        | TileMapKind::Reblock
        | TileMapKind::Repack => None,
    }
}

/// CUDA's one rule, ported from the deleted C++ `transcode_engine.hpp`.
///
/// The behaviour is deliberately unchanged from the C++ it replaces — this
/// moved *where* the decision happens, not *what* it decides, so the existing
/// kernel parity tests stayed valid as the safety net (`architecture.md` §8.1).
/// The citation is provenance for a rule ported verbatim; the file is gone, and
/// this is now the only statement of it.
fn cuda_encode(facts: &TileMapFacts, target: &StorageTarget) -> TileLowering {
    if facts.kind != TileMapKind::Encode {
        return TileLowering::default();
    }
    TileLowering {
        rows_per_tile: encode_rows_per_tile(facts, target),
        fusion: encode_fusion(facts, target),
        // Not this function's question. `lower_tile_map` asks `cuda_kernel`
        // and overwrites; stating `None` here rather than answering twice is
        // what keeps the two decisions from having to agree.
        ..TileLowering::default()
    }
}

/// Rows of the output the driver transforms per launch, or `0` for "all at
/// once".
///
/// Ported including its use of the *logical* dtype width for a quantized
/// source. That is not the true on-disk row size, but reproducing the
/// arithmetic exactly is what keeps this a pure relocation; changing the budget
/// is a separate question.
///
/// Every number this reads is now stated by the target. The rule that is left —
/// scratch is the source plus the dequant buffer, unless they are the same
/// dtype — is the part that cannot be a field, because it is *how the kernel is
/// written*, not what the device is.
fn encode_rows_per_tile(facts: &TileMapFacts, target: &StorageTarget) -> u32 {
    let Some((rows, cols)) = facts.shape else {
        return 0;
    };
    // A strided source would need its stride re-derived per tile, so only a
    // contiguous run qualifies. An Encode with no source reads a device buffer,
    // which is contiguous by construction.
    if facts.has_source && !facts.compact_source {
        return 0;
    }
    let Some(source_dtype) = facts.source_dtype else {
        return 0;
    };
    // A block-scaled source carries one scale per `[block_scale_rows, N]` tile,
    // so slicing the dequant by an arbitrary row count would cut a scale block
    // in half. GLM-5.1's expert weights at [2048, 6144] fit in ~50 MB of
    // scratch, so refusing to tile them costs nothing.
    if target.block_scale_rows != 0 && source_dtype.is_block_scaled() {
        return 0;
    }
    let scratch_dtype = target.encode_scratch_dtype;
    let source_row_bytes = cols.saturating_mul(source_dtype.bytes());
    let scratch_row_bytes = cols.saturating_mul(scratch_dtype.bytes());
    let scratch_per_row = if source_dtype == scratch_dtype {
        scratch_row_bytes
    } else {
        source_row_bytes.saturating_add(scratch_row_bytes)
    };
    let rows_per_tile = rows_under_budget(rows, scratch_per_row, facts.max_tile_bytes);
    // One tile covering everything is the untiled case; say so, rather than
    // making the driver compare a row count against the shape to find out.
    if u64::from(rows_per_tile) >= rows {
        0
    } else {
        rows_per_tile
    }
}

/// Whether to transcode FP8 straight to MXFP4, skipping the BF16 HBM
/// round-trip.
///
/// Bit-identical to the two-step path and kernel parity-tested. What changed is
/// that the opt-out is now a bit in `StorageTarget::fusion_mask` — a compile
/// input — instead of `PIE_CUDA_DISABLE_FUSED_TRANSCODE` read inside the
/// executor. An environment variable that silently selects different kernels
/// for the same plan is exactly the thing `architecture.md` §8.1 objects to; as
/// a target field it produces a *different plan*, which is the honest
/// representation of different execution.
///
/// The *chain* stays here rather than becoming data, and that is the
/// `architecture.md` §9 line: which two steps `Fp8ToMxfp4` collapses, and the
/// proof that collapsing them is bit-identical, is the loader's model of the
/// transform. Whether the kernel exists is the driver's, and that is the bit.
fn encode_fusion(facts: &TileMapFacts, target: &StorageTarget) -> TransformFusion {
    let fusable = target.fusion_mask & FUSION_FP8_TO_MXFP4 != 0
        && facts.transform_to == Some(QuantScheme::Mxfp4E2M1E8M0)
        && facts.has_source
        && facts.source_dtype == Some(DType::F8E4M3);
    if fusable {
        TransformFusion::Fp8ToMxfp4
    } else {
        TransformFusion::None
    }
}

/// Tile a transform by rows under a byte budget.
///
/// Separate because it is arithmetic rather than policy: given the cost of one
/// row and the budget for one tile, this is the row count. What differs per
/// backend is `scratch_per_row` and whether tiling is legal at all.
fn rows_under_budget(rows: u64, scratch_per_row: u64, max_tile_bytes: u64) -> u32 {
    if max_tile_bytes == 0 || scratch_per_row == 0 {
        return clamp_rows(rows);
    }
    let per_tile = (max_tile_bytes / scratch_per_row).max(1);
    clamp_rows(rows.min(per_tile))
}

fn clamp_rows(rows: u64) -> u32 {
    u32::try_from(rows).unwrap_or(u32::MAX)
}

fn extent_is_compact(extent: &crate::extent::Extent) -> bool {
    let mut stride = i64::from(extent.element_bytes);
    for dim in extent.dims.iter().rev() {
        if dim.src_stride != stride || dim.dst_stride != stride {
            return false;
        }
        match stride.checked_mul(dim.count) {
            Some(next) => stride = next,
            None => return false,
        }
    }
    true
}

/// The dtype a transform sees, which for a quantized encoding is the logical
/// one. This is the value the driver reads off `PieLoaderSourceTensorView`, so
/// the two must agree.
fn encoding_dtype(encoding: &Encoding) -> DType {
    match encoding {
        Encoding::Raw(dtype) => *dtype,
        Encoding::Quant(spec) => spec.logical_dtype,
    }
}

fn tile_map_facts(
    plan: &LoadPlan,
    index: &PlanIndex,
    instr: &StorageInstr,
) -> Option<TileMapFacts> {
    let StorageInstr::TileMap {
        kind,
        source,
        dest,
        inputs,
        outputs,
        tile,
        transform,
        ..
    } = instr
    else {
        return None;
    };
    Some(TileMapFacts {
        kind: *kind,
        transform_from: transform.from,
        transform_to: transform.to,
        source_dtype: source_dtype(plan, index, source.as_ref(), inputs),
        has_source: source.is_some(),
        compact_source: source
            .as_ref()
            .is_none_or(|source| extent_is_compact(&source.stride)),
        shape: outputs
            .first()
            .and_then(|buffer| logical_shape(plan, *buffer)),
        max_tile_bytes: tile.max_tile_bytes,
        dest_dtype: outputs.first().and_then(|buffer| raw_dtype(plan, *buffer)),
        in_place: rewrites_in_place(plan, source.as_ref(), inputs, outputs, dest.as_ref()),
        blocked_scale: !transform.scale_blocks.is_empty(),
        operands_in_arena: inputs
            .iter()
            .chain(outputs)
            .chain(dest.as_ref().map(|dest| &dest.buffer))
            .all(|buffer| in_arena(plan, *buffer)),
    })
}

/// Whether a buffer resolves to a span of the arena, through views.
///
/// The same walk the executor's `resolve` does — a window on a resident buffer
/// IS in the arena, and reading only the buffer's own offset reports every
/// alias as absent.
fn in_arena(plan: &LoadPlan, id: BufferId) -> bool {
    let mut id = id;
    for _ in 0..MAX_VIEW_HOPS {
        let Ok(decl) = plan.buffer(id) else {
            return false;
        };
        if decl.arena_offset().is_some() {
            return true;
        }
        let base = plan.instrs.iter().find_map(|instr| match instr {
            StorageInstr::CreateView { input, output, .. } if *output == id => Some(*input),
            _ => None,
        });
        match base {
            Some(base) => id = base,
            None => return false,
        }
    }
    false
}

/// How deep a chain of views may go before the walk gives up; the same guard
/// `passes::arena` and `passes::validate` use, for the same reason.
const MAX_VIEW_HOPS: usize = 16;

/// The dtype behind a buffer, when it is unquantized.
///
/// `None` for a quantized destination rather than its logical dtype: the two
/// are different claims, and a rule that read "bf16" off an MXFP4 output would
/// pick a kernel for bytes that are not there.
fn raw_dtype(plan: &LoadPlan, buffer: BufferId) -> Option<DType> {
    match plan.buffer(buffer).ok()?.ty.encoding {
        Encoding::Raw(dtype) => Some(dtype),
        Encoding::Quant(_) => None,
    }
}

/// Whether the transform's destination is the same bytes as its input.
///
/// Answers the question the executor answers with `op.src != op.dst`, and
/// answers it by BUFFER IDENTITY: the same buffer, covered whole. It used to
/// resolve both sides to arena spans and compare those, which made the answer
/// depend on where a pass had put things — true only after
/// `assign-persistent-offsets`, and `None` for an operand the arena had not
/// placed. Identity is the same answer wherever it is asked, which is what
/// lets `stage-device-transforms` ask it before the placement exists.
///
/// A checkpoint source is never in place: its bytes are on disk.
fn rewrites_in_place(
    plan: &LoadPlan,
    source: Option<&SourceExtent>,
    inputs: &[BufferId],
    outputs: &[BufferId],
    dest: Option<&crate::plan::DestExtent>,
) -> bool {
    if source.is_some() {
        return false;
    }
    let Some(&src) = inputs.first() else {
        return false;
    };
    match dest {
        Some(dest) => {
            let Ok(decl) = plan.buffer(dest.buffer) else {
                return false;
            };
            dest.buffer == src
                && dest.offset + dest.stride.base_offset == 0
                && extent_bytes(&dest.stride) == decl.bytes
        }
        None => outputs.first() == Some(&src),
    }
}

/// The bytes one extent covers: the product of its counts, times the width of
/// the contiguous inner block.
fn extent_bytes(extent: &crate::extent::Extent) -> u64 {
    extent
        .dims
        .iter()
        .try_fold(1u64, |n, d| u64::try_from(d.count).ok().map(|c| n * c))
        .unwrap_or(0)
        * u64::from(extent.element_bytes)
}

fn source_dtype(
    plan: &LoadPlan,
    index: &PlanIndex,
    source: Option<&SourceExtent>,
    inputs: &[BufferId],
) -> Option<DType> {
    if let Some(source) = source {
        return index
            .source(plan, source.tensor_id)
            .map(|decl| encoding_dtype(&decl.encoding));
    }
    // The BUFFER's own type, not its tensor's. This lookup used to go through
    // `BufferDecl::tensor`, so an operand that was not a bound tensor — every
    // intermediate a transform chain produces — typed as `None`, no kernel row
    // was named, and the transform ran on the host no matter what the device
    // could do (`.wiki/fix/loader.md` §3.2).
    plan.buffer(*inputs.first()?)
        .ok()
        .map(|decl| encoding_dtype(&decl.ty.encoding))
}

/// The declared 2-D shape behind a buffer.
///
/// MXFP4 outputs are allocated flat (`u8[bytes]`), so the buffer's own size
/// says nothing about rows and columns; the logical shape lives on the buffer's
/// declared type. Same recovery `encode_tile_map` did in C++.
fn logical_shape(plan: &LoadPlan, buffer: BufferId) -> Option<(u64, u64)> {
    match plan.buffer(buffer).ok()?.ty.shape.as_slice() {
        [rows, cols] => Some((u64::try_from(*rows).ok()?, u64::try_from(*cols).ok()?)),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
