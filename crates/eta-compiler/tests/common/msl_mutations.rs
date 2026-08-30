//! The plan mutations the `validate_singleton_plan` corpus is built from. They
//! were transcribed from `kMutations` / `mutate()` in an in-driver C++ oracle
//! that has since been deleted, and are now the definition rather than a copy
//! of one.
//!
//! `validate_singleton_plan` rejects things a well-formed compiler plan never
//! produces, so almost all of its error paths are only reachable by damaging a
//! valid plan. Both dumps apply exactly these mutations, in this order.
//!
//! Nothing damages a symbolic extent role, a dtype or an op tag: those are
//! closed enums in the Rust plan types, so the matching C++ branches are
//! unreachable here and are deliberately left uncovered.
//!
//! `singleton_plan_rejection.rs` is what reads this. The oracle comparison in
//! `metal_msl_golden.rs` deliberately does not (its reasons are there), which
//! left this file compiled by nothing at all for long enough to bit-rot past a
//! `NodeIndex` newtype -- 245 lines describing a test that was not running.
#![allow(dead_code)]

use eta_compiler::plan::{
    ChannelSink, ChannelSlot, CompiledStage, Dimension, LibraryOp, NodeIndex, PartitionKind,
    RegionKind, SymbolicType,
};
use eta_ir::op::Op;
use eta_ir::types::{Dtype, Predicate};

pub const MUTATIONS: &[&str] = &[
    "none",
    "zero_signature_hash",
    "flip_signature_byte",
    "singleton_kind",
    "whole_stage_fallback",
    "drop_last_singleton_region",
    "singleton_region_node",
    "swap_singleton_region_nodes",
    "unordered_region_nodes",
    "region_input_out_of_range",
    "region_output_out_of_range",
    "region_sink_out_of_range",
    "library_scan_claim",
    "extra_value_type",
    "drop_last_value_type",
    "drop_last_value_type_and_refs",
    "rank5_value_type",
    "zero_static_dim",
    "overflow_static_dims",
    "reverse_ops",
    "bad_channel_slot",
    "pivot_payload_out_of_range",
    "rename_names",
    "clear_names",
];

/// Apply one mutation in place. `false` when it does not apply to this plan.
pub fn mutate(stage: &mut CompiledStage, mutation: &str) -> bool {
    match mutation {
        "none" => true,
        "zero_signature_hash" => {
            if stage.signature.hash == 0 {
                return false;
            }
            stage.signature.hash = 0;
            true
        }
        "flip_signature_byte" => {
            if stage.signature.canonical_bytes.is_empty() {
                return false;
            }
            stage.signature.canonical_bytes[0] ^= 1;
            true
        }
        "singleton_kind" => {
            stage.singleton.kind = PartitionKind::Fused;
            true
        }
        "whole_stage_fallback" => {
            stage.singleton.whole_stage_fallback = true;
            true
        }
        "drop_last_singleton_region" => {
            if stage.singleton.regions.is_empty() {
                return false;
            }
            stage.singleton.regions.pop();
            true
        }
        "singleton_region_node" => {
            if stage.singleton.regions.is_empty() || stage.singleton.regions[0].nodes.is_empty() {
                return false;
            }
            stage.singleton.regions[0].nodes[0] = NodeIndex(stage.normalized.ops.len() as u32);
            true
        }
        "swap_singleton_region_nodes" => {
            if stage.singleton.regions.len() < 2
                || stage.singleton.regions[0].nodes.is_empty()
                || stage.singleton.regions[1].nodes.is_empty()
            {
                return false;
            }
            let first = stage.singleton.regions[0].nodes[0];
            stage.singleton.regions[0].nodes[0] = stage.singleton.regions[1].nodes[0];
            stage.singleton.regions[1].nodes[0] = first;
            true
        }
        "unordered_region_nodes" => {
            if stage.fused.regions.is_empty() || stage.fused.regions[0].nodes.len() < 2 {
                return false;
            }
            stage.fused.regions[0].nodes.swap(0, 1);
            true
        }
        "region_input_out_of_range" => {
            if stage.fused.regions.is_empty() {
                return false;
            }
            let value = stage.normalized.value_types.len() as u32;
            stage.fused.regions[0].inputs.push(value);
            true
        }
        "region_output_out_of_range" => {
            if stage.fused.regions.is_empty() {
                return false;
            }
            let value = stage.normalized.value_types.len() as u32;
            stage.fused.regions[0].outputs.push(value);
            true
        }
        "region_sink_out_of_range" => {
            if stage.fused.regions.is_empty() {
                return false;
            }
            let channel_slot = ChannelSlot(stage.normalized.channel_bindings.len() as u32);
            stage.fused.regions[0].sinks.push(ChannelSink {
                channel_slot,
                value: 0,
            });
            true
        }
        "library_scan_claim" => {
            if stage.fused.regions.is_empty() {
                return false;
            }
            stage.fused.regions[0].kind = RegionKind::Library(LibraryOp::Scan);
            true
        }
        "extra_value_type" => {
            stage.normalized.value_types.push(SymbolicType {
                dtype: Dtype::F32,
                dims: vec![Dimension::Static(1)],
            });
            true
        }
        "drop_last_value_type" => {
            if stage.normalized.value_types.is_empty() {
                return false;
            }
            stage.normalized.value_types.pop();
            true
        }
        "drop_last_value_type_and_refs" => {
            if stage.normalized.value_types.is_empty() {
                return false;
            }
            stage.normalized.value_types.pop();
            let dropped = stage.normalized.value_types.len() as u32;
            for partition in [&mut stage.singleton, &mut stage.fused] {
                for region in &mut partition.regions {
                    region.inputs.retain(|value| *value != dropped);
                    region.outputs.retain(|value| *value != dropped);
                    region.sinks.retain(|sink| sink.value != dropped);
                }
            }
            true
        }
        "rank5_value_type" => {
            if stage.normalized.value_types.is_empty() {
                return false;
            }
            stage.normalized.value_types[0].dims = vec![Dimension::Static(2); 5];
            true
        }
        "zero_static_dim" => {
            for value_type in &mut stage.normalized.value_types {
                for dimension in &mut value_type.dims {
                    if matches!(dimension, Dimension::Static(_)) {
                        *dimension = Dimension::Static(0);
                        return true;
                    }
                }
            }
            false
        }
        "overflow_static_dims" => {
            if stage.normalized.value_types.is_empty() {
                return false;
            }
            stage.normalized.value_types[0].dims = vec![Dimension::Static(65536); 3];
            true
        }
        "reverse_ops" => {
            if stage.normalized.ops.len() < 2 {
                return false;
            }
            stage.normalized.ops.reverse();
            true
        }
        "bad_channel_slot" => {
            let slot = stage.normalized.channel_bindings.len() as u32;
            for op in &mut stage.normalized.ops {
                match op {
                    Op::ChanTake(chan) | Op::ChanRead(chan) | Op::ChanPut { chan, .. } => {
                        *chan = slot;
                        return true;
                    }
                    _ => {}
                }
            }
            false
        }
        "pivot_payload_out_of_range" => {
            let payload = stage.normalized.value_types.len() as u32;
            for op in &mut stage.normalized.ops {
                if let Op::PivotThreshold { predicate, .. } = op {
                    *predicate = match predicate {
                        Predicate::RankLe(_) => Predicate::RankLe(payload),
                        Predicate::CummassLe(_) => Predicate::CummassLe(payload),
                        Predicate::ProbGe(_) => Predicate::ProbGe(payload),
                    };
                    return true;
                }
            }
            false
        }
        "rename_names" => {
            if stage.normalized.names.is_empty() {
                return false;
            }
            for name in &mut stage.normalized.names {
                *name = "metal.bogus".into();
            }
            true
        }
        "clear_names" => {
            if stage.normalized.names.is_empty() {
                return false;
            }
            stage.normalized.names.clear();
            true
        }
        other => panic!("unknown mutation {other}"),
    }
}
