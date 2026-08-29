//! The region ABI checks the CUDA fused emitter gates on.
//!
//! `validate_generated_region` is what decides whether a region can be emitted
//! at all, and `second_party_region_supported` names the one second-party
//! kernel this backend launches.

use crate::codegen::error::{EmitError, RegionForm, ValueLayoutSite};
use crate::codegen::wellformed::{ops_valid, region_ranges_valid, value_types_valid};

use crate::plan::{CompiledStage, LibraryOp, Region, RegionKind, library_op_for_tag};
use tensor_ir::op::Op;
use tensor_ir::registry::Stage;
use tensor_ir::types::DType;

/// `second_party_region_supported` — `envelope_dot` is the only second-party
/// kernel this backend launches, so anything else fails at bind rather than
/// reaching a runtime throw mid-fire.
///
/// It is deliberately not type-checked the way `cuda.identity` is: its argument
/// is the query and its result is a per-page score, shapes unrelated by
/// construction, so arity is the only structural claim that holds.
pub fn second_party_region_supported(stage: &CompiledStage, region: &Region) -> bool {
    if region.nodes.len() != 1 {
        return false;
    }
    let node = region.nodes[0].index();
    let Some(op) = stage.normalized.ops.get(node) else {
        return false;
    };
    let value_types = &stage.normalized.value_types;

    // Configuration sinks. Two first-party shapes are launchable:
    //
    // `attn_page_mask(mask)` — one argument, no result. The mask is a
    // per-page vector over the request's page list, so the only structural
    // claim that holds is rank 1; its extent is the program's own page
    // ceiling, which the runtime checks against the lane's page count.
    //
    // `lora(a, b, sites)` — exactly three arguments, no result, prologue only
    // (pass-wide: the whole forward consumes it). The A/B extents are the
    // adapter's own trace-known geometry and `sites` is a constant over the
    // model's site vocabulary — none of them relate to a shape this gate
    // knows, so arity is the structural claim that holds.
    //
    // Everything else is refused here rather than mid-fire.
    if let Op::SinkCall { name, args } = op {
        let Some(sink) = stage.normalized.names.get(*name as usize) else {
            return false;
        };
        if !region.outputs.is_empty() || region.inputs.len() != args.len() {
            return false;
        }
        return match sink.as_str() {
            "attn_page_mask" => {
                if args.len() != 1 {
                    return false;
                }
                let Some(mask) = value_types.get(region.inputs[0] as usize) else {
                    return false;
                };
                mask.dims.len() == 1 && stage.normalized.stage == Stage::OnAttnProj
            }
            // 3 args = low-rank (A, B, SITES); 2 args = the SCALE form
            // (L, SITES) — IA3 (the adapter per-form rung).
            "lora" => {
                (args.len() == 3 || args.len() == 2) && stage.normalized.stage == Stage::Prologue
            }
            _ => false,
        };
    }

    let Op::KernelCall { name, args, .. } = op else {
        return false;
    };
    let Some(kernel) = stage.normalized.names.get(*name as usize) else {
        return false;
    };
    if kernel != "envelope_dot" || args.len() != 1 || region.outputs.len() != 1 {
        return false;
    }
    // The score is a per-page f32 vector. A different rank or dtype means the
    // program disagrees with the kernel's ABI.
    let Some(result) = value_types.get(region.outputs[0] as usize) else {
        return false;
    };
    if result.dtype != DType::F32 || result.dims.len() != 1 {
        return false;
    }
    matches!(stage.normalized.stage, Stage::OnAttnProj | Stage::OnAttn)
}

/// `validate_generated_region`.
///
/// Also the gate `region_analysis` asks before declaring a region bindable, so
/// everything emission refuses has to be refused here too — a region that
/// passes analysis and then fails emission reaches the engine as an error
/// kernel instead of a tier-0 fallback.
///
/// A `RegionKind::Library` region is NOT refused for being one. `Library` says
/// the plan RECOGNIZED a dataflow a backend may have a kernel for, not that
/// every backend has it: Metal's `grouped_library` returns `None` for the ones
/// it has not written and falls through to its generated emitter, and this
/// backend has written none at all. Refusing them here emitted nothing for the
/// region, the engine read `Slot::Refused` as "the host declined on purpose"
/// and skipped it, and the sampler's whole chain silently never ran — every
/// nonzero temperature published token 0. What actually decides emittability is
/// the per-node check below, which is unchanged: a multi-op lift like
/// `NucleusSample` wraps ordinary ops and emits, while a single-op lift wraps
/// the library op itself and still does not. `.wiki/migration.md` §11.21.
pub fn validate_generated_region(stage: &CompiledStage, region: &Region) -> Result<(), EmitError> {
    if region.nodes.is_empty() {
        return Err(EmitError::FusedRequiresGeneratedRegion);
    }
    // A library CLAIM still has to be true. Nothing downstream re-derives it —
    // a backend that owns the kernel takes the plan's word — so a generated
    // region relabelled `Library(Scan)` would be emitted here as the ordinary
    // ops it holds and launched there as a scan over them. The single-op lifts
    // are a tag lookup; `NucleusSample` is the one multi-op lift, and its
    // arity is what `compile::nucleus` builds (13 nodes, and the plain
    // `[logits, top_p, state]` or the temperature-folded five).
    if let RegionKind::Library(claimed) = region.kind {
        let honest = if claimed == LibraryOp::NucleusSample {
            region.nodes.len() == 13
                && (region.inputs.len() == 3 || region.inputs.len() == 5)
                && region.outputs.len() == 1
        } else {
            region.nodes.len() == 1
                && library_op_for_tag(stage.normalized.ops[region.nodes[0].index()].tag())
                    == Some(claimed)
        };
        if !honest {
            return Err(EmitError::FusedRequiresGeneratedRegion);
        }
    }
    value_types_valid(stage)?;
    ops_valid(stage, ValueLayoutSite::CudaFusedStage)?;
    region_ranges_valid(stage, region, RegionForm::Fused)?;
    for &node in &region.nodes {
        let op = &stage.normalized.ops[node.index()];
        // Asked through the same classifier `region_kind_for_node` used to
        // build the region, so the check cannot disagree with the decision it
        // is checking. The variant list this replaces named `kernel_call` and
        // `sink_call` only, and would have let a fused `top_k`, `sort_desc`,
        // `cumsum`, `cumprod` or `matmul` through to an emitter with no arm
        // for it.
        if let Some(library) = library_op_for_tag(op.tag()) {
            return Err(EmitError::GeneratedRegionHasBoundary {
                library_op: library.name(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::second_party_region_supported;
    use crate::plan::{LibraryOp, RegionKind, compile_bound};
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
    use tensor_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use tensor_ir::op::Op;
    use tensor_ir::registry::{ModelProfile, Stage};
    use tensor_ir::types::{DType, Shape};
    use tensor_ir::validate::bind;

    /// A prologue `lora` container: three peeked channels feeding the sink,
    /// with an optional argument dropped to model a malformed call.
    fn lora_container(args: usize) -> TraceContainer {
        let chan = |shape| ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(DType::F32),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        };
        TraceContainer {
            names: vec!["lora".to_string()],
            channels: vec![
                chan(Shape::new(&[2, 2, 4]).unwrap()), // A [num_layers, R, d]
                chan(Shape::new(&[2, 4, 2]).unwrap()), // B [num_layers, d_out, R]
                chan(Shape::vector(4)),                // SITES
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Prologue,
                ops: vec![
                    Op::ChanRead(0),
                    Op::ChanRead(1),
                    Op::ChanRead(2),
                    Op::SinkCall {
                        name: 0,
                        args: (0..args as u32).collect(),
                    },
                ],
            }],
            externs: Vec::new(),
        }
    }

    /// The lone second-party region of the container's prologue stage,
    /// answered by the gate.
    fn prologue_sink_supported(container: TraceContainer) -> bool {
        let bound = bind(container, ModelProfile::dummy()).expect("the lora prologue binds");
        let stages = compile_bound(&bound);
        let stage = stages
            .iter()
            .find(|s| s.normalized.stage == Stage::Prologue)
            .expect("prologue stage");
        let region = stage
            .fused
            .regions
            .iter()
            .find(|r| matches!(r.kind, RegionKind::Library(LibraryOp::SecondParty)))
            .expect("the sink call partitions into its own second-party region");
        second_party_region_supported(stage, region)
    }

    /// The adapter sink's two forms by arity: 3 = low-rank
    /// `(A, B, SITES)`, 2 = SCALE `(L, SITES)` (IA3). Any other arity is
    /// a program disagreeing with the sink's ABI and is refused at bind
    /// rather than mid-fire.
    #[test]
    fn lora_region_gate_holds_the_form_arities() {
        assert!(prologue_sink_supported(lora_container(3)));
        assert!(prologue_sink_supported(lora_container(2)));
        assert!(!prologue_sink_supported(lora_container(1)));
    }
}
