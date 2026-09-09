use core::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitterKind {
    CudaSingleton,
    CudaFused,
    CudaOrder,
    CudaScan,
    MetalReadiness,
    MetalCommit,
    MetalFused,
}

impl EmitterKind {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmitError {
    EntryNameNotCIdentifier(EmitterKind),
    UnsupportedSingletonOpcode { tag: u8 },

    ChannelLimitExceeded { emitter: EmitterKind, limit: usize },
    UnsupportedKernelBoundary,
    UnbindableIntrinsic { intrinsic: u16 },
    UnsupportedSinkBoundary,
    FusedRequiresGeneratedRegion,
    GeneratedRegionHasBoundary { library_op: &'static str },

    RegionNodeOutOfRange(RegionForm),
    RegionNodesUnordered(RegionForm),
    RegionInputOutOfRange,
    RegionOutputOutOfRange,
    RegionSinkOutOfRange,
    LibraryRegionAbiInvalid(RegionForm),
    ChannelRootBindingOutOfRange,
    ChannelSinkBindingOutOfRange,

    SingletonPlanIdentityInvalid,
    SingletonPartitionArityMismatch,
    WholeStageFallbackWithoutCause,
    SingletonRegionOrderingMismatch,

    NormalizedValueTypeInvalid,
    NormalizedValueShapeOverflow,
    NormalizedOpArityMismatch,
    NormalizedOpResultRangeInvalid,
    NormalizedOperandNotPriorValue,
    PivotPredicatePayloadOutOfRange,
    NormalizedChannelSlotInvalid,
    NormalizedValueLayoutMismatch(ValueLayoutSite),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionForm {
    Fused,
    GroupedFused,
    GroupedNucleus,
    GroupedTopK,
    CudaOrder,
    CudaScan,
    Unnamed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueLayoutSite {
    CudaFusedStage,
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
