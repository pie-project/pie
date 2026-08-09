//! Borrowed C view of a [`LaunchPackage`].
//!
//! The owned package already holds every leaf buffer — argument lists, shapes,
//! node lists, name bytes — so those slices point straight into it and are
//! never copied. What has to be materialised is the *record* arrays, because a
//! `Vec<LaunchOp>` is not a `Vec<PieLaunchOp>`; those live in this struct for
//! as long as the descriptor does.
//!
//! Nesting is safe for the same reason it is in [`super::abi`]: moving a `Vec`
//! moves its header, not its heap buffer, so a pointer taken into an inner
//! vector stays valid when the outer one grows.

use ::driver::local::{
    PieBytes, PieBytesSlice, PieLaunchChannel, PieLaunchChannelRule, PieLaunchChannelRuleSlice,
    PieLaunchChannelSlice, PieLaunchOp, PieLaunchOpSlice, PieLaunchPackage, PieLaunchPlanValue,
    PieLaunchPlanValueSlice, PieLaunchPort, PieLaunchPortSlice, PieLaunchPut, PieLaunchPutSlice,
    PieLaunchRegion, PieLaunchRegionSlice, PieLaunchStage, PieLaunchStagePlan,
    PieLaunchStagePlanSlice, PieLaunchStageSlice, PieLaunchValue, PieLaunchValueSlice, PieU8Slice,
    PieU32Slice,
};
use ::driver::plan::{LaunchOp, LaunchPackage, LaunchRegion, LaunchStagePlan};

fn bytes(slice: &[u8]) -> PieBytes {
    PieBytes {
        ptr: slice.as_ptr(),
        len: slice.len(),
    }
}
fn u8s(slice: &[u8]) -> PieU8Slice {
    PieU8Slice {
        ptr: slice.as_ptr(),
        len: slice.len(),
    }
}
fn u32s(slice: &[u32]) -> PieU32Slice {
    PieU32Slice {
        ptr: slice.as_ptr(),
        len: slice.len(),
    }
}

/// Per-stage and per-plan record arrays, kept alive alongside the raw package.
#[derive(Default)]
struct Arena {
    ops: Vec<Vec<PieLaunchOp>>,
    puts: Vec<Vec<PieLaunchPut>>,
    sinks: Vec<Vec<PieLaunchPut>>,
    regions: Vec<Vec<PieLaunchRegion>>,
    plan_values: Vec<Vec<PieLaunchPlanValue>>,
    rules: Vec<Vec<PieLaunchChannelRule>>,
    names: Vec<Vec<PieBytes>>,
    source_ops: Vec<Vec<u32>>,
    source_op_counts: Vec<Vec<u32>>,
}

impl Arena {
    fn push_ops(&mut self, ops: &[LaunchOp]) -> PieLaunchOpSlice {
        let records: Vec<PieLaunchOp> = ops
            .iter()
            .map(|op| PieLaunchOp {
                code: op.code,
                result_count: op.result_count,
                result_id: op.result_id,
                intrinsic: op.intrinsic,
                lit_dtype: op.lit_dtype,
                dtype: op.dtype,
                lit_bits: op.lit_bits,
                pred_tag: op.pred_tag,
                rng_kind: op.rng_kind,
                reserved0: 0,
                reserved1: 0,
                pred_payload: op.pred_payload,
                channel: op.channel,
                name_index: op.name_index,
                imm: op.imm,
                imm2: op.imm2,
                imm3: op.imm3,
                args: u32s(&op.args),
                shape: u32s(&op.shape),
            })
            .collect();
        let slice = PieLaunchOpSlice {
            ptr: records.as_ptr(),
            len: records.len(),
        };
        self.ops.push(records);
        slice
    }

    fn push_puts(&mut self, puts: &[::driver::plan::LaunchPut]) -> PieLaunchPutSlice {
        let records: Vec<PieLaunchPut> = puts
            .iter()
            .map(|put| PieLaunchPut {
                channel: put.channel,
                value: put.value,
            })
            .collect();
        let slice = PieLaunchPutSlice {
            ptr: records.as_ptr(),
            len: records.len(),
        };
        self.puts.push(records);
        slice
    }

    fn push_regions(&mut self, regions: &[LaunchRegion]) -> PieLaunchRegionSlice {
        let sinks: Vec<PieLaunchPutSlice> = regions
            .iter()
            .map(|region| {
                let records: Vec<PieLaunchPut> = region
                    .sinks
                    .iter()
                    .map(|sink| PieLaunchPut {
                        channel: sink.channel,
                        value: sink.value,
                    })
                    .collect();
                let slice = PieLaunchPutSlice {
                    ptr: records.as_ptr(),
                    len: records.len(),
                };
                self.sinks.push(records);
                slice
            })
            .collect();
        let records: Vec<PieLaunchRegion> = regions
            .iter()
            .zip(sinks)
            .map(|(region, sinks)| PieLaunchRegion {
                kind: region.kind,
                library: region.library,
                schedule: region.schedule,
                reserved0: 0,
                reserved1: 0,
                nodes: u32s(&region.nodes),
                inputs: u32s(&region.inputs),
                outputs: u32s(&region.outputs),
                sinks,
            })
            .collect();
        let slice = PieLaunchRegionSlice {
            ptr: records.as_ptr(),
            len: records.len(),
        };
        self.regions.push(records);
        slice
    }

    fn push_names(&mut self, names: &[String]) -> PieBytesSlice {
        let records: Vec<PieBytes> = names.iter().map(|name| bytes(name.as_bytes())).collect();
        let slice = PieBytesSlice {
            ptr: records.as_ptr(),
            len: records.len(),
        };
        self.names.push(records);
        slice
    }

    fn push_plan(&mut self, plan: &LaunchStagePlan) -> PieLaunchStagePlan {
        let ops = self.push_ops(&plan.ops);
        let singleton = self.push_regions(&plan.singleton);
        let fused = self.push_regions(&plan.fused);
        let names = self.push_names(&plan.names);

        let flat: Vec<u32> = plan.source_ops.iter().flatten().copied().collect();
        let counts: Vec<u32> = plan
            .source_ops
            .iter()
            .map(|sources| sources.len() as u32)
            .collect();
        let source_ops = u32s(&flat);
        let source_op_counts = u32s(&counts);
        self.source_ops.push(flat);
        self.source_op_counts.push(counts);

        let values: Vec<PieLaunchPlanValue> = plan
            .value_types
            .iter()
            .map(|value| PieLaunchPlanValue {
                dtype: value.dtype,
                reserved0: 0,
                reserved1: 0,
                extents: u8s(&value.extents),
                dims: u32s(&value.dims),
            })
            .collect();
        let value_types = PieLaunchPlanValueSlice {
            ptr: values.as_ptr(),
            len: values.len(),
        };
        self.plan_values.push(values);

        let rules: Vec<PieLaunchChannelRule> = plan
            .channel_rules
            .iter()
            .map(|rule| PieLaunchChannelRule {
                value: rule.value,
                local: rule.local,
            })
            .collect();
        let channel_rules = PieLaunchChannelRuleSlice {
            ptr: rules.as_ptr(),
            len: rules.len(),
        };
        self.rules.push(rules);

        PieLaunchStagePlan {
            signature_hash: plan.signature_hash,
            identity: plan.identity,
            flags: plan.flags,
            mtp_rows: plan.mtp_rows,
            ops,
            source_ops,
            source_op_counts,
            value_types,
            channel_bindings: u32s(&plan.channel_bindings),
            names,
            singleton,
            fused,
            used_extents: u8s(&plan.used_extents),
            channel_rules,
            error: bytes(plan.error.as_bytes()),
        }
    }
}

/// A [`LaunchPackage`] with its C record arrays materialised.
pub struct LaunchPackageBorrow<'a> {
    _package: &'a LaunchPackage,
    _arena: Arena,
    _values: Vec<PieLaunchValue>,
    _channels: Vec<PieLaunchChannel>,
    _ports: Vec<PieLaunchPort>,
    _names: Vec<PieBytes>,
    _stages: Vec<PieLaunchStage>,
    _plans: Vec<PieLaunchStagePlan>,
    raw: PieLaunchPackage,
}

impl<'a> LaunchPackageBorrow<'a> {
    pub fn new(package: &'a LaunchPackage) -> Self {
        let mut arena = Arena::default();

        let values: Vec<PieLaunchValue> = package
            .values
            .iter()
            .map(|value| PieLaunchValue {
                id: value.id,
                source: value.source,
                dtype: value.dtype,
                intrinsic: value.intrinsic,
                reserved1: 0,
                channel: value.channel,
                literal_bits: value.literal_bits,
                reserved0: 0,
                shape: u32s(&value.shape),
            })
            .collect();
        let channels: Vec<PieLaunchChannel> = package
            .channels
            .iter()
            .map(|channel| PieLaunchChannel {
                id: channel.id,
                capacity: channel.capacity,
                dtype: channel.dtype,
                flags: channel.flags,
                extern_dir: channel.extern_dir,
                readiness: channel.readiness,
                reserved1: 0,
                shape: u32s(&channel.shape),
                extern_name: bytes(&channel.extern_name),
            })
            .collect();
        let ports: Vec<PieLaunchPort> = package
            .ports
            .iter()
            .map(|port| PieLaunchPort {
                port: port.port,
                is_const: u8::from(port.is_const),
                const_dtype: port.const_dtype,
                reserved0: 0,
                channel: port.channel,
                const_shape: u32s(&port.const_shape),
                const_data: bytes(&port.const_data),
            })
            .collect();
        let names: Vec<PieBytes> = package
            .names
            .iter()
            .map(|name| bytes(name.as_bytes()))
            .collect();
        let stages: Vec<PieLaunchStage> = package
            .stages
            .iter()
            .map(|stage| PieLaunchStage {
                kind: stage.kind,
                reserved0: 0,
                reserved1: 0,
                reserved2: 0,
                ops: arena.push_ops(&stage.ops),
                puts: arena.push_puts(&stage.puts),
                takes: u32s(&stage.takes),
                reads: u32s(&stage.reads),
            })
            .collect();
        let plans: Vec<PieLaunchStagePlan> = package
            .plans
            .iter()
            .map(|plan| arena.push_plan(plan))
            .collect();

        let raw = PieLaunchPackage {
            values: PieLaunchValueSlice {
                ptr: values.as_ptr(),
                len: values.len(),
            },
            channels: PieLaunchChannelSlice {
                ptr: channels.as_ptr(),
                len: channels.len(),
            },
            ports: PieLaunchPortSlice {
                ptr: ports.as_ptr(),
                len: ports.len(),
            },
            names: PieBytesSlice {
                ptr: names.as_ptr(),
                len: names.len(),
            },
            stages: PieLaunchStageSlice {
                ptr: stages.as_ptr(),
                len: stages.len(),
            },
            plans: PieLaunchStagePlanSlice {
                ptr: plans.as_ptr(),
                len: plans.len(),
            },
        };

        Self {
            _package: package,
            _arena: arena,
            _values: values,
            _channels: channels,
            _ports: ports,
            _names: names,
            _stages: stages,
            _plans: plans,
            raw,
        }
    }

    pub fn as_raw(&self) -> PieLaunchPackage {
        self.raw
    }
}
