//! Every reason an emitter can refuse to produce a kernel. The
//! [`Display`](core::fmt::Display) text crosses the C ABI to the runtime and
//! is what an operator reads in a log; `every_refusal_renders` pins it.

use core::fmt;

/// Which emitter refused, where the message names one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitterKind {
    /// CUDA, one op per kernel.
    CudaSingleton,
    /// CUDA, a whole generated region per kernel.
    CudaFused,
    /// CUDA, the generated `Order` library regions — `top_k` and `sort_desc`.
    CudaOrder,
    /// CUDA, the generated scan library region — `cumsum` and `cumprod`.
    CudaScan,
    /// Metal M1 readiness — the pre-launch channel check.
    MetalReadiness,
    /// Metal M1 commit — the post-launch ring advance.
    MetalCommit,
    /// Metal M2 fused, bound directly to channel cells.
    MetalFused,
}

impl EmitterKind {
    /// The name this emitter goes by in a refusal.
    fn label(self) -> &'static str {
        match self {
            EmitterKind::CudaSingleton => "CUDA singleton",
            EmitterKind::CudaFused => "CUDA fused",
            EmitterKind::CudaOrder => "CUDA order",
            EmitterKind::CudaScan => "CUDA scan",
            EmitterKind::MetalReadiness => "readiness kernel",
            EmitterKind::MetalCommit => "commit kernel",
            EmitterKind::MetalFused => "fused region",
        }
    }
}

/// Every reason source emission can decline to produce a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmitError {
    // --- malformed request ---
    /// The requested entry point cannot be spelled in C or MSL.
    EntryNameNotCIdentifier(EmitterKind),
    /// No singleton body exists for this op tag.
    UnsupportedSingletonOpcode {
        /// The op tag with no singleton body.
        tag: u8,
    },

    // --- unsupported, but well-formed ---
    /// More channels than the backend can bind directly. Metal binds each
    /// channel to its own buffer index, so the limit is a hard argument-table
    /// ceiling rather than a tuning choice.
    ChannelLimitExceeded {
        /// The [`EmitterKind`] that hit the ceiling.
        emitter: EmitterKind,
        /// The direct-binding channel ceiling the plan exceeded.
        limit: usize,
    },
    /// The op reaches a kernel boundary Metal has no lowering for.
    UnsupportedKernelBoundary,
    /// Metal binds only the logits buffer for intrinsics, so an intrinsic
    /// wanting anything else has nowhere to read from.
    UnbindableIntrinsic {
        /// The offending intrinsic's `IntrinsicId` wire tag.
        intrinsic: u16,
    },
    /// The op reaches a sink boundary Metal has no lowering for.
    UnsupportedSinkBoundary,
    /// The CUDA fused emitter was handed a library region, which is served by
    /// a hand-written kernel rather than generated source.
    FusedRequiresGeneratedRegion,
    /// A generated region reached a boundary op, which cannot be generated.
    /// Carries the library op's own name (its tagged-enum label).
    GeneratedRegionHasBoundary {
        /// The library op's tagged-enum name, e.g. `top_k` or `matmul`.
        library_op: &'static str,
    },

    // --- invalid plan: region structure ---
    /// A region's node list indexes past the stage's ops.
    RegionNodeOutOfRange(RegionForm),
    /// A region's node indices are not strictly increasing.
    RegionNodesUnordered(RegionForm),
    /// A region input indexes past the stage's values.
    RegionInputOutOfRange,
    /// A region output indexes past the stage's values.
    RegionOutputOutOfRange,
    /// A region sink indexes past the stage's values.
    RegionSinkOutOfRange,
    /// A library region's ABI record is not one this backend serves.
    LibraryRegionAbiInvalid(RegionForm),
    /// A channel root binding indexes past the stage's channels.
    ChannelRootBindingOutOfRange,
    /// A channel sink binding indexes past the stage's channels.
    ChannelSinkBindingOutOfRange,

    // --- invalid plan: singleton partition ---
    /// The singleton plan's identity does not match the stage it came with.
    SingletonPlanIdentityInvalid,
    /// The singleton partition is not one region per normalized op.
    SingletonPartitionArityMismatch,
    /// The plan asked for whole-stage fallback but every op in it is one the
    /// backend supports, so there is no cause to report.
    WholeStageFallbackWithoutCause,
    /// A singleton region's node does not match its position.
    SingletonRegionOrderingMismatch,

    // --- invalid plan: normalized ops and values ---
    /// A normalized value's type is not one the backend can lay out.
    NormalizedValueTypeInvalid,
    /// A normalized value's element count does not fit in `u32`.
    NormalizedValueShapeOverflow,
    /// A normalized op has a different operand count than its table arity.
    NormalizedOpArityMismatch,
    /// A normalized op's result ids fall outside the stage's value space.
    NormalizedOpResultRangeInvalid,
    /// A normalized operand names a value that is not already defined —
    /// the SSA dominance property, violated.
    NormalizedOperandNotPriorValue,
    /// A pivot predicate's value-id payload is outside the value space.
    PivotPredicatePayloadOutOfRange,
    /// A normalized channel operand names a slot the stage does not bind.
    NormalizedChannelSlotInvalid,
    /// The stage's value table and its ops' results disagree on count.
    NormalizedValueLayoutMismatch(ValueLayoutSite),
}

/// Which region form a structural refusal came from. The Metal emitters run
/// the same checks over four region shapes and each names itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionForm {
    /// An M2 fused region.
    Fused,
    /// An M3 grouped-fused region.
    GroupedFused,
    /// An M3 grouped nucleus library region.
    GroupedNucleus,
    /// An M3 grouped TopK library region.
    GroupedTopK,
    /// The CUDA `Order` library regions — `top_k` and `sort_desc` — emitted
    /// into the fused slot.
    CudaOrder,
    /// The CUDA scan library region — `cumsum` and `cumprod` — emitted into
    /// the fused slot.
    CudaScan,
    /// A region the validator inspects without naming its form.
    Unnamed,
}

/// Which side found the value layout inconsistent. The two spell it
/// differently and both strings are pinned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueLayoutSite {
    /// The CUDA fused emitter, comparing the stage's layout to its ops.
    CudaFusedStage,
    /// The Metal validator, comparing normalized values to op results.
    MetalNormalized,
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmitError::EntryNameNotCIdentifier(emitter) => {
                write!(f, "{} entry name is not a C identifier", emitter.label())
            }
            EmitError::UnsupportedSingletonOpcode { tag } => {
                write!(f, "unsupported CUDA singleton opcode tag {tag}")
            }
            EmitError::ChannelLimitExceeded { emitter, limit } => write!(
                f,
                "{} exceeds the {limit}-channel direct-binding limit",
                emitter.label()
            ),
            EmitError::UnsupportedKernelBoundary => {
                f.write_str("unsupported Metal semantic kernel boundary")
            }
            EmitError::UnbindableIntrinsic { intrinsic } => write!(
                f,
                "Metal binds only the logits buffer for intrinsics; \
                 intrinsic id {intrinsic} has no binding"
            ),
            EmitError::UnsupportedSinkBoundary => {
                f.write_str("unsupported Metal semantic sink boundary")
            }
            EmitError::FusedRequiresGeneratedRegion => {
                f.write_str("fused CUDA emitter requires a non-library generated region")
            }
            EmitError::GeneratedRegionHasBoundary { library_op } => write!(
                f,
                "generated region contains a non-generated boundary ({library_op})"
            ),
            EmitError::RegionNodeOutOfRange(form) => match form {
                RegionForm::Fused => f.write_str("fused region node out of range"),
                RegionForm::GroupedFused => f.write_str("grouped fused region node out of range"),
                RegionForm::GroupedTopK => f.write_str("TopK library node is out of range"),
                RegionForm::CudaOrder => f.write_str("CUDA order library node is out of range"),
                RegionForm::CudaScan => f.write_str("CUDA scan library node is out of range"),
                RegionForm::GroupedNucleus | RegionForm::Unnamed => {
                    f.write_str("region node out of range")
                }
            },
            EmitError::RegionNodesUnordered(form) => match form {
                RegionForm::Fused => f.write_str("fused region nodes are not strictly ordered"),
                RegionForm::GroupedFused => {
                    f.write_str("grouped fused region nodes are not strictly ordered")
                }
                RegionForm::GroupedTopK => f.write_str("TopK library node is invalid"),
                RegionForm::CudaOrder => f.write_str("CUDA order library node is invalid"),
                RegionForm::CudaScan => f.write_str("CUDA scan library node is invalid"),
                RegionForm::GroupedNucleus | RegionForm::Unnamed => {
                    f.write_str("region node indices are not strictly ordered")
                }
            },
            EmitError::RegionInputOutOfRange => f.write_str("region input out of range"),
            EmitError::RegionOutputOutOfRange => f.write_str("region output out of range"),
            EmitError::RegionSinkOutOfRange => f.write_str("region sink out of range"),
            EmitError::LibraryRegionAbiInvalid(form) => match form {
                RegionForm::GroupedFused => f.write_str("grouped library region ABI is invalid"),
                RegionForm::GroupedNucleus => f.write_str("invalid grouped nucleus library region"),
                RegionForm::GroupedTopK => f.write_str("invalid grouped TopK library region"),
                RegionForm::CudaOrder => f.write_str("invalid CUDA order library region"),
                RegionForm::CudaScan => f.write_str("invalid CUDA scan library region"),
                RegionForm::Fused | RegionForm::Unnamed => {
                    f.write_str("library region ABI is invalid")
                }
            },
            EmitError::ChannelRootBindingOutOfRange => {
                f.write_str("fused channel root binding out of range")
            }
            EmitError::ChannelSinkBindingOutOfRange => {
                f.write_str("fused channel sink binding out of range")
            }
            EmitError::SingletonPlanIdentityInvalid => {
                f.write_str("invalid singleton plan identity")
            }
            EmitError::SingletonPartitionArityMismatch => {
                f.write_str("singleton partition must contain one region per normalized op")
            }
            EmitError::WholeStageFallbackWithoutCause => f.write_str(
                "singleton plan requests whole-stage fallback without an identifiable unsupported op",
            ),
            EmitError::SingletonRegionOrderingMismatch => {
                f.write_str("singleton region/node ordering mismatch")
            }
            EmitError::NormalizedValueTypeInvalid => f.write_str("invalid normalized value type"),
            EmitError::NormalizedValueShapeOverflow => {
                f.write_str("normalized value shape product exceeds u32")
            }
            EmitError::NormalizedOpArityMismatch => f.write_str("normalized op arity mismatch"),
            EmitError::NormalizedOpResultRangeInvalid => {
                f.write_str("normalized op result range is invalid")
            }
            EmitError::NormalizedOperandNotPriorValue => {
                f.write_str("normalized SSA operand is not a prior value")
            }
            EmitError::PivotPredicatePayloadOutOfRange => {
                f.write_str("pivot predicate payload is out of range")
            }
            EmitError::NormalizedChannelSlotInvalid => {
                f.write_str("normalized channel slot is invalid")
            }
            EmitError::NormalizedValueLayoutMismatch(site) => match site {
                ValueLayoutSite::CudaFusedStage => {
                    f.write_str("fused stage value layout does not match normalized ops")
                }
                ValueLayoutSite::MetalNormalized => {
                    f.write_str("normalized value layout does not match op results")
                }
            },
        }
    }
}

