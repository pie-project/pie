//! The launch package — a program in the shape a driver executes it.
//!
//! This is the whole of what crosses the host→driver boundary for a program,
//! and it deliberately is not PTIR. A driver reading it never sees a container,
//! a sidecar, a wire format, or an identity to re-check: the compiler already
//! decided all of that (`ptir-refactor.md` §2.3). What it gets is the value
//! table, the channels and ports to allocate and bind, the per-stage op DAGs to
//! launch, and the per-stage plan the emitted kernels were generated from.
//!
//! Two things that used to be re-derived on the far side are folded in here,
//! because they are decisions about the program rather than about the machine:
//!
//! * each stage's **graph-cache identity** ([`pie_plan::stage_identity`]), and
//! * the **grouped static plan** — which runtime extents the stage depends on,
//!   which values bind through a channel, which intrinsics it reads, and
//!   whether the grouped launch path can cover it at all.
//!
//! The op projection is [`crate::op_view::OpView`], the same one the emitters
//! read, so the kernel and the description of the kernel cannot drift.

use pie_driver_abi::plan::{
    LaunchChannel, LaunchChannelRule, LaunchOp, LaunchPackage, LaunchPlanValue, LaunchPort,
    LaunchPut, LaunchRegion, LaunchStage, LaunchStagePlan, LaunchValue,
};
use pie_ir::DType;
use pie_ir::container::{ExternDir, HostRole, PortSource};
use pie_ir::op::{intrinsic_tags, tags};
use pie_ir::types::ValueType;
use pie_ir::validate::{BoundTrace, Direction};
use pie_plan::{
    CompiledStage, Dimension, LibraryOp, Region, RegionKind, RegionPartition, SymbolicType,
    stage_identity,
};

use crate::op_view::OpView;

// Op tags and enum values shared with the generated ABI header
// (`compiler/codegen/include/ptir_abi.h`). Named rather than inlined so the
// mapping below reads like the container's own op table.

/// Value sources, mirroring `PIE_VALUE_*`.
const VALUE_CONST: u8 = 0;
const VALUE_INTRINSIC: u8 = 1;
const VALUE_CHANNEL_TAKE: u8 = 2;
const VALUE_CHANNEL_READ: u8 = 3;
const VALUE_OP_RESULT: u8 = 4;

/// Channel flags, mirroring `PIE_CHANNEL_*`.
const CHANNEL_SEEDED: u8 = 1 << 0;
const CHANNEL_HOST_VISIBLE: u8 = 1 << 1;
const CHANNEL_HOST_READER: u8 = 1 << 2;

/// Readiness directions, mirroring `PIE_READINESS_*`.
const READINESS_UNTOUCHED: u8 = 0;
const READINESS_NEEDS_FULL: u8 = 1;
const READINESS_NEEDS_EMPTY: u8 = 2;

/// Region kinds, mirroring `PIE_REGION_GENERATED` / `PIE_REGION_LIBRARY`.
const REGION_GENERATED: u8 = 0;
const REGION_LIBRARY: u8 = 1;

/// A dimension is a literal extent, mirroring `PIE_EXTENT_STATIC`.
const EXTENT_STATIC: u8 = 0xff;
/// Highest runtime extent key the grouped path understands.
const EXTENT_KEY_MAX: u8 = 6;

/// Stage flags, mirroring `PIE_STAGE_REQUIRES_*` / `PIE_STAGE_GROUPED_VALID`.
const REQUIRES_QUERY: u32 = 1 << 0;
const REQUIRES_LAYER: u32 = 1 << 1;
const REQUIRES_ATTN_SCORE: u32 = 1 << 2;
const REQUIRES_KERNEL_CALL: u32 = 1 << 3;
const REQUIRES_PAGE_MASK: u32 = 1 << 4;
const REQUIRES_MTP_ROWS: u32 = 1 << 5;
const GROUPED_VALID: u32 = 1 << 6;

/// No channel. Mirrors the `-1` a wire op carries for a non-channel op.
const NO_CHANNEL: u32 = u32::MAX;

/// Build the launch package for a bound trace and its compiled stages.
///
/// `stages` is `pie_plan::compile_bound(bound)`, in container order — the
/// caller already has it, because the emitters need the same value.
pub fn build(bound: &BoundTrace, stages: &[CompiledStage]) -> LaunchPackage {
    LaunchPackage {
        values: lower_values(bound),
        channels: lower_channels(bound),
        ports: lower_ports(bound),
        names: bound.container.names.clone(),
        stages: lower_stages(bound),
        plans: stages.iter().map(lower_plan).collect(),
    }
}

// ── the trace half: what to allocate, bind, and launch ──

/// The SSA value table, in the same global numbering the stage bodies use:
/// stage `i`'s local value `v` is global `Σ|stage_types[..i]| + v`.
fn lower_values(bound: &BoundTrace) -> Vec<LaunchValue> {
    let mut values = Vec::new();
    let mut base = 0u32;
    for (stage_index, program) in bound.container.stages.iter().enumerate() {
        let types = &bound.stage_types[stage_index];
        let mut local = 0u32;
        let mut push = |local: u32, mut value: LaunchValue| {
            let value_type = types
                .get(local as usize)
                .copied()
                .unwrap_or(ValueType::scalar(DType::F32));
            value.id = base + local;
            value.dtype = value_type.dtype as u8;
            value.shape = value_type.shape.dims().to_vec();
            values.push(value);
        };
        for op in &program.ops {
            let view = OpView::of(op);
            match view.tag {
                tags::CHAN_TAKE | tags::CHAN_READ => {
                    push(
                        local,
                        LaunchValue {
                            source: if view.tag == tags::CHAN_TAKE {
                                VALUE_CHANNEL_TAKE
                            } else {
                                VALUE_CHANNEL_READ
                            },
                            channel: view.chan as u32,
                            ..LaunchValue::default()
                        },
                    );
                    local += 1;
                }
                tags::CONST => {
                    push(
                        local,
                        LaunchValue {
                            source: VALUE_CONST,
                            literal_bits: view.lit_bits,
                            ..LaunchValue::default()
                        },
                    );
                    local += 1;
                }
                tags::INTRINSIC_VAL => {
                    push(
                        local,
                        LaunchValue {
                            source: VALUE_INTRINSIC,
                            intrinsic: view.intr as u8,
                            ..LaunchValue::default()
                        },
                    );
                    local += 1;
                }
                tags::CHAN_PUT | tags::SINK_CALL => {}
                _ => {
                    for result in 0..view.results {
                        push(
                            local + result,
                            LaunchValue {
                                source: VALUE_OP_RESULT,
                                ..LaunchValue::default()
                            },
                        );
                    }
                    local += view.results;
                }
            }
        }
        base += types.len() as u32;
    }
    values
}

fn lower_channels(bound: &BoundTrace) -> Vec<LaunchChannel> {
    bound
        .container
        .channels
        .iter()
        .enumerate()
        .map(|(index, decl)| {
            let mut flags = 0u8;
            if decl.seeded {
                flags |= CHANNEL_SEEDED;
            }
            if decl.host_role != HostRole::None {
                flags |= CHANNEL_HOST_VISIBLE;
            }
            if decl.host_role == HostRole::Reader {
                flags |= CHANNEL_HOST_READER;
            }
            let extern_decl = bound
                .container
                .externs
                .iter()
                .find(|entry| entry.chan as usize == index);
            // First-touch, in pass order. The sets a stage ships say *whether*
            // a channel is taken and put; only this says which one comes first,
            // and an `InPlace` channel is both.
            let readiness = bound
                .readiness
                .iter()
                .find(|entry| entry.chan as usize == index)
                .map_or(READINESS_UNTOUCHED, |entry| match entry.dir {
                    Direction::NeedsFull => READINESS_NEEDS_FULL,
                    Direction::NeedsEmpty => READINESS_NEEDS_EMPTY,
                });
            LaunchChannel {
                id: index as u32,
                capacity: decl.capacity,
                // The program-side element type, with a late-bound activation
                // dtype already materialized — the driver allocates cells from
                // this and never sees `ACT`.
                dtype: bound.channel_types[index].dtype as u8,
                flags,
                extern_dir: extern_decl.map_or(-1, |entry| match entry.dir {
                    ExternDir::Import => 0,
                    ExternDir::Export => 1,
                }),
                readiness,
                shape: decl.shape.dims().to_vec(),
                extern_name: extern_decl
                    .and_then(|entry| bound.container.names.get(entry.name as usize))
                    .map(|name| name.as_bytes().to_vec())
                    .unwrap_or_default(),
            }
        })
        .collect()
}

fn lower_ports(bound: &BoundTrace) -> Vec<LaunchPort> {
    bound
        .container
        .ports
        .iter()
        .map(|binding| match binding.source {
            PortSource::Channel(chan) => LaunchPort {
                port: binding.port as u8,
                is_const: false,
                channel: chan,
                ..LaunchPort::default()
            },
            PortSource::Const {
                dtype,
                ref shape,
                ref data,
            } => LaunchPort {
                port: binding.port as u8,
                is_const: true,
                const_dtype: dtype as u8,
                const_shape: shape.dims().to_vec(),
                const_data: data.clone(),
                ..LaunchPort::default()
            },
        })
        .collect()
}

/// The stage bodies, with every operand remapped from stage-local to the
/// global numbering of [`lower_values`]. Ops that define no value — `chan_put`
/// — become stage effects instead of ops, exactly as the execution model wants
/// them: a put is committed at pass end, not launched.
fn lower_stages(bound: &BoundTrace) -> Vec<LaunchStage> {
    let mut stages = Vec::new();
    let mut base = 0u32;
    for (stage_index, program) in bound.container.stages.iter().enumerate() {
        let types = &bound.stage_types[stage_index];
        let global = |local: u32| base + local;
        let result_type = |local: u32| {
            types
                .get(local as usize)
                .copied()
                .unwrap_or(ValueType::scalar(DType::F32))
        };

        let mut stage = LaunchStage {
            kind: program.stage as u8,
            ..LaunchStage::default()
        };
        let mut local = 0u32;
        for op in &program.ops {
            let view = OpView::of(op);
            match view.tag {
                tags::CHAN_TAKE | tags::CHAN_READ => {
                    if view.tag == tags::CHAN_TAKE {
                        stage.takes.push(view.chan as u32);
                    } else {
                        stage.reads.push(view.chan as u32);
                    }
                    local += 1;
                }
                tags::CONST | tags::INTRINSIC_VAL => local += 1,
                tags::CHAN_PUT => stage.puts.push(LaunchPut {
                    channel: view.chan as u32,
                    value: global(view.args[0]),
                }),
                _ => {
                    let value_type = result_type(local);
                    stage.ops.push(LaunchOp {
                        code: view.tag as u16,
                        result_count: view.results as u16,
                        result_id: global(local),
                        intrinsic: view.intr,
                        lit_dtype: view.lit_dtype,
                        dtype: value_type.dtype as u8,
                        pred_tag: view.pred_tag,
                        rng_kind: view.kind,
                        lit_bits: view.lit_bits,
                        // Every predicate tag carries a value id, so it is
                        // remapped like any other operand rather than left as
                        // an immediate.
                        pred_payload: global(view.pred_payload),
                        channel: NO_CHANNEL,
                        name_index: u32::from(view.name_idx),
                        imm: view.imm,
                        imm2: view.imm2,
                        imm3: view.imm3,
                        args: view.args.iter().map(|arg| global(*arg)).collect(),
                        shape: value_type.shape.dims().to_vec(),
                    });
                    local += view.results;
                }
            }
        }
        stages.push(stage);
        base += types.len() as u32;
    }
    stages
}

// ── the plan half: what the emitted kernels were generated from ──

fn lower_plan(stage: &CompiledStage) -> LaunchStagePlan {
    let normalized = &stage.normalized;
    let ops: Vec<OpView> = OpView::of_all(&normalized.ops);
    let grouped = GroupedPlan::derive(
        &ops,
        &normalized.value_types,
        normalized.channel_bindings.len(),
    );

    LaunchStagePlan {
        signature_hash: stage.signature.hash,
        identity: stage_identity(stage),
        flags: grouped.flags,
        mtp_rows: grouped.mtp_rows,
        ops: ops.iter().map(lower_plan_op).collect(),
        source_ops: normalized.source_ops.clone(),
        value_types: normalized
            .value_types
            .iter()
            .map(lower_plan_value)
            .collect(),
        channel_bindings: normalized.channel_bindings.clone(),
        names: normalized.names.clone(),
        singleton: lower_partition(&stage.singleton),
        fused: lower_partition(&stage.fused),
        used_extents: grouped.used_extents,
        channel_rules: grouped.channel_rules,
        error: grouped.error,
    }
}

/// A normalized op keeps its stage-local numbering: the plan's ops index the
/// plan's own value table, which is what the emitted kernels bind against.
fn lower_plan_op(view: &OpView) -> LaunchOp {
    LaunchOp {
        code: view.tag as u16,
        result_count: view.results as u16,
        result_id: 0,
        intrinsic: view.intr,
        lit_dtype: view.lit_dtype,
        dtype: view.dtype,
        pred_tag: view.pred_tag,
        rng_kind: view.kind,
        lit_bits: view.lit_bits,
        pred_payload: view.pred_payload,
        channel: if view.chan < 0 {
            NO_CHANNEL
        } else {
            view.chan as u32
        },
        name_index: u32::from(view.name_idx),
        imm: view.imm,
        imm2: view.imm2,
        imm3: view.imm3,
        args: view.args.clone(),
        shape: view.shape.clone(),
    }
}

fn lower_plan_value(value_type: &SymbolicType) -> LaunchPlanValue {
    let mut extents = Vec::with_capacity(value_type.dims.len());
    let mut dims = Vec::with_capacity(value_type.dims.len());
    for dimension in &value_type.dims {
        match *dimension {
            Dimension::Static(extent) => {
                extents.push(EXTENT_STATIC);
                dims.push(extent);
            }
            Dimension::Symbolic(extent) => {
                extents.push(extent as u8);
                dims.push(0);
            }
        }
    }
    LaunchPlanValue {
        dtype: value_type.dtype as u8,
        extents,
        dims,
    }
}

fn lower_partition(partition: &RegionPartition) -> Vec<LaunchRegion> {
    partition.regions.iter().map(lower_region).collect()
}

fn lower_region(region: &Region) -> LaunchRegion {
    let (kind, library) = match region.kind {
        RegionKind::Generated => (REGION_GENERATED, 0),
        RegionKind::Library(op) => (REGION_LIBRARY, library_tag(op)),
    };
    LaunchRegion {
        kind,
        library,
        schedule: region.schedule as u8,
        nodes: region.nodes.clone(),
        inputs: region.inputs.clone(),
        outputs: region.outputs.clone(),
        sinks: region
            .sinks
            .iter()
            .map(|sink| LaunchPut {
                channel: sink.channel_slot,
                value: sink.value,
            })
            .collect(),
    }
}

fn library_tag(op: LibraryOp) -> u8 {
    op as u8
}

/// The grouped launch path's view of a stage: which runtime extents it
/// depends on, which values bind through a channel, which intrinsics it reads,
/// and whether the path can cover it at all.
///
/// This used to be `GroupedStageStaticPlan`'s constructor in the CUDA driver.
/// It is a decision about the program, so it is made once, here.
#[derive(Default)]
struct GroupedPlan {
    flags: u32,
    mtp_rows: u32,
    used_extents: Vec<u8>,
    channel_rules: Vec<LaunchChannelRule>,
    error: String,
}

impl GroupedPlan {
    fn derive(ops: &[OpView], value_types: &[SymbolicType], channel_count: usize) -> Self {
        let mut plan = GroupedPlan {
            flags: GROUPED_VALID,
            ..GroupedPlan::default()
        };
        let mut seen = [false; (EXTENT_KEY_MAX as usize) + 1];
        for value_type in value_types {
            for dimension in &value_type.dims {
                let Dimension::Symbolic(extent) = *dimension else {
                    continue;
                };
                let extent = extent as u8;
                if extent > EXTENT_KEY_MAX {
                    return plan.invalid("stage uses unsupported runtime extents");
                }
                if !seen[extent as usize] {
                    seen[extent as usize] = true;
                    plan.used_extents.push(extent);
                }
            }
        }

        let mut value_bases = Vec::with_capacity(ops.len());
        let mut next_value = 0u32;
        for op in ops {
            value_bases.push(next_value);
            next_value += op.results;
        }

        for (node, op) in ops.iter().enumerate() {
            // A second-party kernel is not a grouped-coverable tag — only the
            // fused path can launch one — but this plan is the shared
            // lane-binding metadata the fused path resolves through, so it
            // describes the op rather than rejecting it. `requires_query` is
            // set because the kernel consumes the lane's post-rope query row.
            if op.tag == tags::KERNEL_CALL {
                plan.flags |= REQUIRES_QUERY | REQUIRES_KERNEL_CALL;
                continue;
            }
            // `attn_page_mask` is grouped-walkable but only fused-executable.
            // Described here for the same reason.
            if op.tag == tags::SINK_CALL {
                plan.flags |= REQUIRES_PAGE_MASK;
            }
            if !grouped_supported_tag(op.tag) {
                return plan.invalid("stage contains an unsupported grouped op");
            }
            if op.tag == tags::INTRINSIC_VAL {
                match op.intr {
                    intrinsic_tags::QUERY => plan.flags |= REQUIRES_QUERY,
                    intrinsic_tags::LAYER => plan.flags |= REQUIRES_LAYER,
                    intrinsic_tags::ATTN_SCORE => plan.flags |= REQUIRES_ATTN_SCORE,
                    intrinsic_tags::MTP_LOGITS => {
                        let value = value_bases[node] as usize;
                        let rows = match value_types.get(value).map(|ty| ty.dims.as_slice()) {
                            Some([Dimension::Static(rows), _]) => *rows,
                            _ => return plan.invalid("MtpLogits has no static draft-row layout"),
                        };
                        if plan.flags & REQUIRES_MTP_ROWS != 0 && plan.mtp_rows != rows {
                            return plan.invalid(
                                "MtpLogits stages declare incompatible draft-row layouts",
                            );
                        }
                        plan.flags |= REQUIRES_MTP_ROWS;
                        plan.mtp_rows = rows;
                    }
                    intrinsic_tags::LOGITS => {}
                    _ => return plan.invalid("stage uses an unsupported intrinsic"),
                }
            }

            let value = match op.tag {
                tags::CHAN_TAKE | tags::CHAN_READ => value_bases[node],
                tags::CHAN_PUT if !op.args.is_empty() => op.args[0],
                _ => continue,
            };
            if value as usize >= value_types.len()
                || op.chan < 0
                || op.chan as usize >= channel_count
            {
                return plan.invalid("channel value is outside the grouped plan");
            }
            plan.channel_rules.push(LaunchChannelRule {
                value,
                local: op.chan as u32,
            });
        }
        plan
    }

    /// The stage cannot take the grouped path. The flags derived so far still
    /// describe it — the fused path reads them — so only the validity bit and
    /// the reason change.
    fn invalid(mut self, reason: &str) -> Self {
        self.flags &= !GROUPED_VALID;
        self.error = reason.to_string();
        self
    }
}

/// Whether a generic grouped op can cover this tag. The list is the grouped
/// interpreter's own coverage: everything except `kernel_call`, which names a
/// second-party kernel only the fused path can launch.
/// Ops the grouped runtime has no body for. Everything else in
/// [`pie_ir::op::OP_TABLE`] is emittable.
///
/// This used to be a list of raw hex ranges (`0x01..=0x07 | 0x10..=0x20 | ...`).
/// The gaps between those ranges are exactly where new op tags go, so a new op
/// was silently classified unsupported — no compile error, no test failure,
/// just a stage that stopped taking the grouped path. Asking the op table
/// closes the gaps; `grouped_classification_is_exhaustive` makes a new op
/// force a decision here.
const GROUPED_UNSUPPORTED: &[u8] = &[tags::KERNEL_CALL];

fn grouped_supported_tag(tag: u8) -> bool {
    pie_ir::op::spec(tag).is_some() && !GROUPED_UNSUPPORTED.contains(&tag)
}

#[cfg(test)]
mod grouped_coverage {
    use super::*;

    /// A tripwire, not a tautology: bump the count only together with a
    /// decision about whether the grouped runtime can emit the new op, and
    /// with `GROUPED_UNSUPPORTED` updated if it cannot.
    #[test]
    fn grouped_classification_is_exhaustive() {
        assert_eq!(
            pie_ir::op::OP_TABLE.len(),
            55,
            "a new op landed in OP_TABLE — decide whether the grouped runtime \
             has a body for it, update GROUPED_UNSUPPORTED if not, then bump \
             this count"
        );
        for tag in GROUPED_UNSUPPORTED {
            assert!(
                pie_ir::op::spec(*tag).is_some(),
                "GROUPED_UNSUPPORTED names a tag that is not an op"
            );
            assert!(!grouped_supported_tag(*tag));
        }
        assert!(
            !grouped_supported_tag(0xFF),
            "a non-op tag is not supported"
        );
    }
}
