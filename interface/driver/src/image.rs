//! A relocatable byte image of a [`LaunchPackage`], for driver *fixtures*.
//!
//! The native driver tests register programs, so they need a
//! [`PieLaunchPackage`]. They cannot run the compiler, and the compiler cannot
//! emit C++ initializers because the fixtures are written *after* the C++ is
//! built. So the package is written out as its own `#[repr(C)]` bytes plus a
//! relocation table, and the C++ side (`pie/driver/testing/image.hpp`) turns it
//! back into a package with one pass of pointer fixups.
//!
//! This is not a wire format and nothing in production reads it: the engine
//! hands drivers a [`PieLaunchPackage`] built in memory. It exists so the driver
//! test surface can be fed the same table without a PTIR decoder — and it is
//! only sound because Rust and C++ already agree on every struct's layout, which
//! `interface/driver/tests/header_layout.rs` gates.

use core::mem::{align_of, offset_of, size_of};

use crate::local::{
    PIE_DRIVER_ABI_VERSION, PieBytes, PieBytesSlice, PieLaunchChannel, PieLaunchChannelRule,
    PieLaunchChannelRuleSlice, PieLaunchChannelSlice, PieLaunchOp, PieLaunchOpSlice,
    PieLaunchPackage, PieLaunchPlanValue, PieLaunchPlanValueSlice, PieLaunchPort,
    PieLaunchPortSlice, PieLaunchPut, PieLaunchPutSlice, PieLaunchRegion, PieLaunchRegionSlice,
    PieLaunchStage, PieLaunchStagePlan, PieLaunchStagePlanSlice, PieLaunchStageSlice,
    PieLaunchValue, PieLaunchValueSlice, PieU8Slice, PieU32Slice,
};
use crate::plan::{LaunchOp, LaunchPackage, LaunchPut, LaunchRegion, LaunchStagePlan};

/// `"PIELPKG\0"`, little-endian.
pub const IMAGE_MAGIC: u64 = 0x0047_4b50_4c45_4950;

/// Fixed image header. Followed by `blob_len` bytes of struct image, then
/// `reloc_count` little-endian `u64` byte offsets into that image, each naming a
/// pointer field to be shifted by the load address.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImageHeader {
    pub magic: u64,
    pub abi_version: u64,
    pub blob_len: u64,
    pub reloc_count: u64,
    /// Offset of the root [`PieLaunchPackage`] within the image.
    pub root: u64,
}

/// Encode `package` as a relocatable image.
pub fn encode(package: &LaunchPackage) -> Vec<u8> {
    let mut image = Image::default();
    let root = image.write_package(package);
    let header = ImageHeader {
        magic: IMAGE_MAGIC,
        abi_version: u64::from(PIE_DRIVER_ABI_VERSION),
        blob_len: image.blob.len() as u64,
        reloc_count: image.relocs.len() as u64,
        root: root as u64,
    };
    let mut out =
        Vec::with_capacity(size_of::<ImageHeader>() + image.blob.len() + image.relocs.len() * 8);
    out.extend_from_slice(&header.magic.to_le_bytes());
    out.extend_from_slice(&header.abi_version.to_le_bytes());
    out.extend_from_slice(&header.blob_len.to_le_bytes());
    out.extend_from_slice(&header.reloc_count.to_le_bytes());
    out.extend_from_slice(&header.root.to_le_bytes());
    out.extend_from_slice(&image.blob);
    for reloc in &image.relocs {
        out.extend_from_slice(&reloc.to_le_bytes());
    }
    out
}

struct Image {
    blob: Vec<u8>,
    relocs: Vec<u64>,
}

impl Default for Image {
    fn default() -> Self {
        // Offset 0 is reserved. A slice descriptor encodes its offset in the
        // pointer field, so data at offset 0 would be indistinguishable from
        // the null that marks an empty slice -- and the relocation pass would
        // skip it, handing the consumer a null pointer with a nonzero length.
        Self {
            blob: vec![0; 8],
            relocs: Vec::new(),
        }
    }
}

/// A pointer whose numeric value is a blob offset. Never dereferenced; the
/// consumer adds the load address to it.
fn offset_ptr<T>(offset: usize) -> *const T {
    offset as *const T
}

/// Record a relocation for `field` of every element of an array that landed at
/// `base`, skipping the null (empty-slice) ones.
macro_rules! reloc {
    ($image:expr, $base:expr, $items:expr, $ty:ty, $($field:ident),+ $(,)?) => {
        for (index, item) in $items.iter().enumerate() {
            let record = $base + index * size_of::<$ty>();
            $(
                if !item.$field.ptr.is_null() {
                    $image.relocs.push((record + offset_of!($ty, $field)) as u64);
                }
            )+
        }
    };
}

impl Image {
    /// Append `items`' raw `#[repr(C)]` bytes and return their offset. Empty
    /// arrays get no storage: their slice descriptor stays null.
    fn put<T: Copy>(&mut self, items: &[T]) -> usize {
        if items.is_empty() {
            return usize::MAX;
        }
        let align = align_of::<T>().max(8);
        while self.blob.len() % align != 0 {
            self.blob.push(0);
        }
        let offset = self.blob.len();
        let bytes = unsafe {
            core::slice::from_raw_parts(items.as_ptr().cast::<u8>(), size_of::<T>() * items.len())
        };
        self.blob.extend_from_slice(bytes);
        offset
    }

    fn u32s(&mut self, items: &[u32]) -> PieU32Slice {
        let offset = self.put(items);
        PieU32Slice {
            ptr: if items.is_empty() {
                core::ptr::null()
            } else {
                offset_ptr(offset)
            },
            len: items.len(),
        }
    }

    fn u8s(&mut self, items: &[u8]) -> PieU8Slice {
        let offset = self.put(items);
        PieU8Slice {
            ptr: if items.is_empty() {
                core::ptr::null()
            } else {
                offset_ptr(offset)
            },
            len: items.len(),
        }
    }

    fn bytes(&mut self, items: &[u8]) -> PieBytes {
        let offset = self.put(items);
        PieBytes {
            ptr: if items.is_empty() {
                core::ptr::null()
            } else {
                offset_ptr(offset)
            },
            len: items.len(),
        }
    }

    fn name_table(&mut self, names: &[String]) -> PieBytesSlice {
        let records: Vec<PieBytes> = names
            .iter()
            .map(|name| self.bytes(name.as_bytes()))
            .collect();
        let base = self.put(&records);
        if records.is_empty() {
            return PieBytesSlice {
                ptr: core::ptr::null(),
                len: 0,
            };
        }
        for (index, record) in records.iter().enumerate() {
            if !record.ptr.is_null() {
                self.relocs
                    .push((base + index * size_of::<PieBytes>()) as u64);
            }
        }
        PieBytesSlice {
            ptr: offset_ptr(base),
            len: records.len(),
        }
    }

    fn puts(&mut self, puts: &[LaunchPut]) -> PieLaunchPutSlice {
        let records: Vec<PieLaunchPut> = puts
            .iter()
            .map(|put| PieLaunchPut {
                channel: put.channel,
                value: put.value,
            })
            .collect();
        let base = self.put(&records);
        PieLaunchPutSlice {
            ptr: if records.is_empty() {
                core::ptr::null()
            } else {
                offset_ptr(base)
            },
            len: records.len(),
        }
    }

    fn ops(&mut self, ops: &[LaunchOp]) -> PieLaunchOpSlice {
        let records: Vec<PieLaunchOp> = ops
            .iter()
            .map(|op| PieLaunchOp {
                code: op.code,
                result_count: op.result_count,
                result_id: op.result_id,
                intrinsic: op.intrinsic,
                lit_dtype: op.lit_dtype,
                dtype: op.dtype,
                pred_tag: op.pred_tag,
                rng_kind: op.rng_kind,
                reserved0: 0,
                lit_bits: op.lit_bits,
                pred_payload: op.pred_payload,
                channel: op.channel,
                name_index: op.name_index,
                imm: op.imm,
                imm2: op.imm2,
                imm3: op.imm3,
                reserved1: 0,
                args: self.u32s(&op.args),
                shape: self.u32s(&op.shape),
            })
            .collect();
        let base = self.put(&records);
        if records.is_empty() {
            return PieLaunchOpSlice {
                ptr: core::ptr::null(),
                len: 0,
            };
        }
        reloc!(self, base, &records, PieLaunchOp, args, shape);
        PieLaunchOpSlice {
            ptr: offset_ptr(base),
            len: records.len(),
        }
    }

    fn regions(&mut self, regions: &[LaunchRegion]) -> PieLaunchRegionSlice {
        let records: Vec<PieLaunchRegion> = regions
            .iter()
            .map(|region| PieLaunchRegion {
                kind: region.kind,
                library: region.library,
                schedule: region.schedule,
                reserved0: 0,
                reserved1: 0,
                nodes: self.u32s(&region.nodes),
                inputs: self.u32s(&region.inputs),
                outputs: self.u32s(&region.outputs),
                sinks: self.puts(&region.sinks),
            })
            .collect();
        let base = self.put(&records);
        if records.is_empty() {
            return PieLaunchRegionSlice {
                ptr: core::ptr::null(),
                len: 0,
            };
        }
        reloc!(
            self,
            base,
            &records,
            PieLaunchRegion,
            nodes,
            inputs,
            outputs,
            sinks
        );
        PieLaunchRegionSlice {
            ptr: offset_ptr(base),
            len: records.len(),
        }
    }

    fn plan(&mut self, plan: &LaunchStagePlan) -> PieLaunchStagePlan {
        let value_records: Vec<PieLaunchPlanValue> = plan
            .value_types
            .iter()
            .map(|value| PieLaunchPlanValue {
                dtype: value.dtype,
                reserved0: 0,
                reserved1: 0,
                extents: self.u8s(&value.extents),
                dims: self.u32s(&value.dims),
            })
            .collect();
        let value_base = self.put(&value_records);
        let value_types = if value_records.is_empty() {
            PieLaunchPlanValueSlice {
                ptr: core::ptr::null(),
                len: 0,
            }
        } else {
            reloc!(
                self,
                value_base,
                &value_records,
                PieLaunchPlanValue,
                extents,
                dims
            );
            PieLaunchPlanValueSlice {
                ptr: offset_ptr(value_base),
                len: value_records.len(),
            }
        };

        let rule_records: Vec<PieLaunchChannelRule> = plan
            .channel_rules
            .iter()
            .map(|rule| PieLaunchChannelRule {
                value: rule.value,
                local: rule.local,
            })
            .collect();
        let rule_base = self.put(&rule_records);

        let flat_sources: Vec<u32> = plan.source_ops.iter().flatten().copied().collect();
        let source_counts: Vec<u32> = plan
            .source_ops
            .iter()
            .map(|sources| sources.len() as u32)
            .collect();

        PieLaunchStagePlan {
            signature_hash: plan.signature_hash,
            identity: plan.identity,
            flags: plan.flags,
            mtp_rows: plan.mtp_rows,
            ops: self.ops(&plan.ops),
            source_ops: self.u32s(&flat_sources),
            source_op_counts: self.u32s(&source_counts),
            value_types,
            channel_bindings: self.u32s(&plan.channel_bindings),
            names: self.name_table(&plan.names),
            singleton: self.regions(&plan.singleton),
            fused: self.regions(&plan.fused),
            used_extents: self.u8s(&plan.used_extents),
            channel_rules: PieLaunchChannelRuleSlice {
                ptr: if rule_records.is_empty() {
                    core::ptr::null()
                } else {
                    offset_ptr(rule_base)
                },
                len: rule_records.len(),
            },
            error: self.bytes(plan.error.as_bytes()),
        }
    }

    fn write_package(&mut self, package: &LaunchPackage) -> usize {
        let value_records: Vec<PieLaunchValue> = package
            .values
            .iter()
            .map(|value| PieLaunchValue {
                id: value.id,
                source: value.source,
                dtype: value.dtype,
                reserved1: 0,
                intrinsic: value.intrinsic,
                channel: value.channel,
                literal_bits: value.literal_bits,
                reserved0: 0,
                shape: self.u32s(&value.shape),
            })
            .collect();
        let value_base = self.put(&value_records);
        let values = if value_records.is_empty() {
            PieLaunchValueSlice {
                ptr: core::ptr::null(),
                len: 0,
            }
        } else {
            reloc!(self, value_base, &value_records, PieLaunchValue, shape);
            PieLaunchValueSlice {
                ptr: offset_ptr(value_base),
                len: value_records.len(),
            }
        };

        let channel_records: Vec<PieLaunchChannel> = package
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
                shape: self.u32s(&channel.shape),
                extern_name: self.bytes(&channel.extern_name),
            })
            .collect();
        let channel_base = self.put(&channel_records);
        let channels = if channel_records.is_empty() {
            PieLaunchChannelSlice {
                ptr: core::ptr::null(),
                len: 0,
            }
        } else {
            reloc!(
                self,
                channel_base,
                &channel_records,
                PieLaunchChannel,
                shape,
                extern_name
            );
            PieLaunchChannelSlice {
                ptr: offset_ptr(channel_base),
                len: channel_records.len(),
            }
        };

        let port_records: Vec<PieLaunchPort> = package
            .ports
            .iter()
            .map(|port| PieLaunchPort {
                port: port.port,
                is_const: u8::from(port.is_const),
                const_dtype: port.const_dtype,
                reserved0: 0,
                channel: port.channel,
                const_shape: self.u32s(&port.const_shape),
                const_data: self.bytes(&port.const_data),
            })
            .collect();
        let port_base = self.put(&port_records);
        let ports = if port_records.is_empty() {
            PieLaunchPortSlice {
                ptr: core::ptr::null(),
                len: 0,
            }
        } else {
            reloc!(
                self,
                port_base,
                &port_records,
                PieLaunchPort,
                const_shape,
                const_data
            );
            PieLaunchPortSlice {
                ptr: offset_ptr(port_base),
                len: port_records.len(),
            }
        };

        let names = self.name_table(&package.names);

        let stage_records: Vec<PieLaunchStage> = package
            .stages
            .iter()
            .map(|stage| PieLaunchStage {
                kind: stage.kind,
                reserved0: 0,
                reserved1: 0,
                reserved2: 0,
                ops: self.ops(&stage.ops),
                puts: self.puts(&stage.puts),
                takes: self.u32s(&stage.takes),
                reads: self.u32s(&stage.reads),
            })
            .collect();
        let stage_base = self.put(&stage_records);
        let stages = if stage_records.is_empty() {
            PieLaunchStageSlice {
                ptr: core::ptr::null(),
                len: 0,
            }
        } else {
            reloc!(
                self,
                stage_base,
                &stage_records,
                PieLaunchStage,
                ops,
                puts,
                takes,
                reads
            );
            PieLaunchStageSlice {
                ptr: offset_ptr(stage_base),
                len: stage_records.len(),
            }
        };

        let plan_records: Vec<PieLaunchStagePlan> =
            package.plans.iter().map(|plan| self.plan(plan)).collect();
        let plan_base = self.put(&plan_records);
        let plans = if plan_records.is_empty() {
            PieLaunchStagePlanSlice {
                ptr: core::ptr::null(),
                len: 0,
            }
        } else {
            reloc!(
                self,
                plan_base,
                &plan_records,
                PieLaunchStagePlan,
                ops,
                source_ops,
                source_op_counts,
                value_types,
                channel_bindings,
                names,
                singleton,
                fused,
                used_extents,
                channel_rules,
                error,
            );
            PieLaunchStagePlanSlice {
                ptr: offset_ptr(plan_base),
                len: plan_records.len(),
            }
        };

        let root = [PieLaunchPackage {
            values,
            channels,
            ports,
            names,
            stages,
            plans,
        }];
        let root_base = self.put(&root);
        reloc!(
            self,
            root_base,
            &root,
            PieLaunchPackage,
            values,
            channels,
            ports,
            names,
            stages,
            plans,
        );
        root_base
    }
}
