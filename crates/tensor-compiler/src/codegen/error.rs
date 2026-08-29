//! Why an emitter refused — the complete list, in one enum.
//!
//! Emission is total in the sense that it never panics on a plan: every path
//! that cannot produce source returns instead, and the reason travels to the
//! host as a string in [`EmittedKernel::error`]. That string is ABI — the
//! runtime copies it across the C boundary (`crates/runtime/src/engine/abi.rs`)
//! and a human reads it out of an engine log — so the [`Display`](core::fmt::Display) impl here is
//! the one definition of that text, and `every_refusal_renders` pins all of
//! it.
//!
//! ## Why a type and not a `String`
//!
//! Refusals must not be `format!` calls written at their throw sites. That
//! shape leaves the vocabulary existing only as string literals scattered
//! across the emitters, and it fails in two ways at once.
//!
//! One refusal reachable from two places becomes two literals — `library
//! region ABI is invalid` is raised by both the Metal validator and the fused
//! emitter — and nothing makes them agree, even though the host parses one
//! ABI. Worse, a message that quotes a limit tends to quote it as a digit:
//! `fused region exceeds the 12-channel direct-binding limit` written by hand
//! keeps saying `12` after [`METAL_M2_MAX_FUSED_CHANNELS`] moves, so the
//! refusal describes a ceiling the code no longer enforces.
//!
//! Naming each refusal collapses it to one arm and lets the limits
//! interpolate, so both classes of drift become impossible rather than
//! unlikely.
//!
//! ## Shape of the space
//!
//! The variants are grouped by what a caller learns from them:
//!
//! * *Malformed request* — the entry name is not a C identifier, the opcode
//!   has no singleton body. The caller built the request wrong.
//! * *Unsupported* — the plan is well-formed but outside what this backend
//!   emits (a channel count past a direct-binding limit, a semantic boundary
//!   with no Metal lowering). A different backend or a re-planned stage may
//!   succeed.
//! * *Invalid plan* — the [`CompiledStage`] contradicts itself: an index out
//!   of range, an ordering violated, a layout that does not match the ops it
//!   describes. These are the validator's findings and they mean a planner
//!   bug, not a guest one.
//!
//! Nothing branches on the grouping today; it is how the list is read.
//!
//! [`EmittedKernel::error`]: crate::codegen::program::EmittedKernel::error
//! [`CompiledStage`]: https://github.com/pie-project/pie/tree/dev/tensor-compiler's plan
//! [`METAL_M2_MAX_FUSED_CHANNELS`]: crate::codegen::metal::METAL_M2_MAX_FUSED_CHANNELS

use core::fmt;

/// Which emitter refused, where the message names one. Kept separate from the
/// variants so the two entry-name refusals and the three channel-limit
/// refusals are each a single arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitterKind {
    /// CUDA, one op per kernel.
    CudaSingleton,
    /// CUDA, a whole generated region per kernel.
    CudaFused,
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
    ///
    /// Carries the library op's own name rather than just the fact, because
    /// the fact alone is unactionable: three curated inferlets refused here
    /// and the only way to learn WHICH op the partitioner had folded in was
    /// to add a print. The name is the tagged enum's own label, passed as a
    /// `&'static str` so this file keeps its one dependency on `core::fmt`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// The rendered text is ABI: it crosses the C boundary in
    /// `EmittedKernel::error` and an engine log is where it is read. Pinning
    /// every variant means a reworded message is a diff in this file rather
    /// than a silent change in what an operator sees.
    #[test]
    fn every_refusal_renders() {
        let cases: &[(EmitError, &str)] = &[
            (
                EmitError::EntryNameNotCIdentifier(EmitterKind::CudaSingleton),
                "CUDA singleton entry name is not a C identifier",
            ),
            (
                EmitError::EntryNameNotCIdentifier(EmitterKind::CudaFused),
                "CUDA fused entry name is not a C identifier",
            ),
            (
                EmitError::UnsupportedSingletonOpcode { tag: 0xB3 },
                "unsupported CUDA singleton opcode tag 179",
            ),
            (
                EmitError::ChannelLimitExceeded {
                    emitter: EmitterKind::MetalReadiness,
                    limit: 29,
                },
                "readiness kernel exceeds the 29-channel direct-binding limit",
            ),
            (
                EmitError::ChannelLimitExceeded {
                    emitter: EmitterKind::MetalCommit,
                    limit: 29,
                },
                "commit kernel exceeds the 29-channel direct-binding limit",
            ),
            (
                EmitError::ChannelLimitExceeded {
                    emitter: EmitterKind::MetalFused,
                    limit: 12,
                },
                "fused region exceeds the 12-channel direct-binding limit",
            ),
            (
                EmitError::UnsupportedKernelBoundary,
                "unsupported Metal semantic kernel boundary",
            ),
            (
                EmitError::UnsupportedSinkBoundary,
                "unsupported Metal semantic sink boundary",
            ),
            (
                EmitError::UnbindableIntrinsic { intrinsic: 7 },
                "Metal binds only the logits buffer for intrinsics; intrinsic id 7 has no binding",
            ),
            (
                EmitError::WholeStageFallbackWithoutCause,
                "singleton plan requests whole-stage fallback without an identifiable unsupported op",
            ),
            (
                EmitError::FusedRequiresGeneratedRegion,
                "fused CUDA emitter requires a non-library generated region",
            ),
            (
                EmitError::GeneratedRegionHasBoundary {
                    library_op: "top_k",
                },
                "generated region contains a non-generated boundary (top_k)",
            ),
            (
                EmitError::RegionNodeOutOfRange(RegionForm::Fused),
                "fused region node out of range",
            ),
            (
                EmitError::RegionNodeOutOfRange(RegionForm::GroupedFused),
                "grouped fused region node out of range",
            ),
            (
                EmitError::RegionNodeOutOfRange(RegionForm::GroupedTopK),
                "TopK library node is out of range",
            ),
            (
                EmitError::RegionNodeOutOfRange(RegionForm::Unnamed),
                "region node out of range",
            ),
            (
                EmitError::RegionNodesUnordered(RegionForm::Fused),
                "fused region nodes are not strictly ordered",
            ),
            (
                EmitError::RegionNodesUnordered(RegionForm::GroupedTopK),
                "TopK library node is invalid",
            ),
            (
                EmitError::RegionNodesUnordered(RegionForm::Unnamed),
                "region node indices are not strictly ordered",
            ),
            (
                EmitError::RegionInputOutOfRange,
                "region input out of range",
            ),
            (
                EmitError::RegionOutputOutOfRange,
                "region output out of range",
            ),
            (EmitError::RegionSinkOutOfRange, "region sink out of range"),
            (
                EmitError::LibraryRegionAbiInvalid(RegionForm::Unnamed),
                "library region ABI is invalid",
            ),
            (
                EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedFused),
                "grouped library region ABI is invalid",
            ),
            (
                EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedNucleus),
                "invalid grouped nucleus library region",
            ),
            (
                EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedTopK),
                "invalid grouped TopK library region",
            ),
            (
                EmitError::ChannelRootBindingOutOfRange,
                "fused channel root binding out of range",
            ),
            (
                EmitError::ChannelSinkBindingOutOfRange,
                "fused channel sink binding out of range",
            ),
            (
                EmitError::SingletonPlanIdentityInvalid,
                "invalid singleton plan identity",
            ),
            (
                EmitError::SingletonPartitionArityMismatch,
                "singleton partition must contain one region per normalized op",
            ),
            (
                EmitError::SingletonRegionOrderingMismatch,
                "singleton region/node ordering mismatch",
            ),
            (
                EmitError::NormalizedValueTypeInvalid,
                "invalid normalized value type",
            ),
            (
                EmitError::NormalizedValueShapeOverflow,
                "normalized value shape product exceeds u32",
            ),
            (
                EmitError::NormalizedOpArityMismatch,
                "normalized op arity mismatch",
            ),
            (
                EmitError::NormalizedOpResultRangeInvalid,
                "normalized op result range is invalid",
            ),
            (
                EmitError::NormalizedOperandNotPriorValue,
                "normalized SSA operand is not a prior value",
            ),
            (
                EmitError::PivotPredicatePayloadOutOfRange,
                "pivot predicate payload is out of range",
            ),
            (
                EmitError::NormalizedChannelSlotInvalid,
                "normalized channel slot is invalid",
            ),
            (
                EmitError::NormalizedValueLayoutMismatch(ValueLayoutSite::CudaFusedStage),
                "fused stage value layout does not match normalized ops",
            ),
            (
                EmitError::NormalizedValueLayoutMismatch(ValueLayoutSite::MetalNormalized),
                "normalized value layout does not match op results",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(&error.to_string(), expected);
        }
    }

    /// The channel-limit refusal must read its ceiling from the constant it
    /// enforces. Hard-coding the digit lets the constant move and leaves the
    /// message quoting a limit nothing checks.
    #[test]
    fn the_channel_limit_message_reads_the_constant() {
        let rendered = EmitError::ChannelLimitExceeded {
            emitter: EmitterKind::MetalFused,
            limit: crate::codegen::metal::METAL_M2_MAX_FUSED_CHANNELS,
        }
        .to_string();
        assert!(
            rendered.contains(&crate::codegen::metal::METAL_M2_MAX_FUSED_CHANNELS.to_string()),
            "the limit is interpolated, not transcribed: {rendered}"
        );
    }
}
