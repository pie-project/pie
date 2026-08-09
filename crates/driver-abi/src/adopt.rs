//! The `local` → `plan` direction: a C launch package, adopted into an owned one.
//!
//! This crate carries the same program in two shapes. [`local::PieLaunchPackage`]
//! is what actually crosses the C ABI — flat `#[repr(C)]` records whose every
//! list is a `{ptr, len}` pair into memory the host owns, valid only for the
//! duration of the registration call. [`plan::LaunchPackage`] is the owned
//! counterpart, `Vec`s and `String`s a driver can keep for the life of the
//! program. This module is the only thing that turns the first into the second.
//!
//! # What was missing without it
//!
//! A driver's `register_program` entry point receives a
//! [`local::PieProgramDesc`] and nothing else. Every Rust consumer of a launch
//! package — `driver_pipeline::adopt_launch_package`, the region and lane
//! derivations the CUDA and Metal shells share, the emitted-kernel index —
//! reads [`plan::LaunchPackage`]. With no bridge between them a Rust driver
//! could not register a PTIR program at all: it held the package in the one
//! shape nothing could read. The C++ driver had the bridge
//! (`pie::driver::launch::adopt`, in `driver/launch/program.hpp`) and Rust did
//! not.
//!
//! # Why it lives here and not in a driver crate
//!
//! Both representations are defined in this crate, so a conversion between
//! them is this crate's business — putting it in a driver would mean each
//! driver either writing its own copy or depending on whichever one wrote it
//! first, and a per-driver copy of a field-for-field mapping is a per-driver
//! opportunity to drop a field.
//!
//! It is also why only *this* direction is here. The reverse
//! (`engine/src/driver/launch_abi.rs`) cannot be: materialising a C view means
//! keeping the record arrays alive somewhere for as long as the descriptor is
//! read, so it needs an owner with a lifetime — a borrow type the caller holds.
//! Adoption allocates and hands the result back, so it needs nothing but the
//! two type definitions, both of which are here.
//!
//! # `launch_abi.rs` is the authority on the mapping
//!
//! That file builds a `PieLaunchPackage` from a `LaunchPackage`, field by
//! field, and this module is it read backwards. A field left at its default
//! here is not a compile error and not a test failure anywhere else — it is a
//! program that runs with a channel unbound, a region's schedule mis-chosen or
//! an op's third immediate lost — so the round-trip test below asserts a whole
//! non-default package against itself rather than spot-checking members.
//!
//! The C++ predecessor is not the model, for the reason that test exists: its
//! `Trace` stopped at values, channels, names, ports and stages, never looked
//! at `plans` at all, and its `Op` had no field for an op's `intrinsic`, its
//! `lit_dtype`/`lit_bits` or its `channel` — so those were dropped silently,
//! because a destination that lacks a field cannot complain about one.
//! Adopting into [`plan`][crate::plan]'s own types means the destination has a
//! field for everything the source carries, and the compiler says so.
//!
//! # Nothing here may panic
//!
//! Adoption runs inside a driver's `extern "C"` registration entry point. A
//! panic that reaches an `extern "C"` frame **aborts the process** — it does
//! not unwind into the host and it cannot become a `PieStatus` — so the whole
//! host, every other program and every other device context would die because
//! one name had a stray byte or one count was too large. Hence
//! [`String::from_utf8_lossy`] rather than `from_utf8().unwrap()`, a clamp
//! rather than a slice index in `adopt_source_ops`, and a null check rather
//! than a bare [`std::slice::from_raw_parts`] in `slice_of`.
//!
//! [`local::PieLaunchPackage`]: crate::local::PieLaunchPackage
//! [`local::PieProgramDesc`]: crate::local::PieProgramDesc
//! [`plan::LaunchPackage`]: crate::plan::LaunchPackage

use crate::local::{
    PieBytes, PieBytesSlice, PieEmittedKernelSlice, PieLaunchChannelRuleSlice,
    PieLaunchChannelSlice, PieLaunchOpSlice, PieLaunchPackage, PieLaunchPlanValueSlice,
    PieLaunchPortSlice, PieLaunchPutSlice, PieLaunchRegionSlice, PieLaunchStagePlan,
    PieLaunchStagePlanSlice, PieLaunchStageSlice, PieLaunchValueSlice, PieRegionAnalysisSlice,
    PieU8Slice, PieU32Slice,
};
use crate::plan::{
    DirectArgmax, EmittedKernel, LaunchChannel, LaunchChannelRule, LaunchOp, LaunchPackage,
    LaunchPlanValue, LaunchPort, LaunchPut, LaunchRegion, LaunchStage, LaunchStagePlan,
    LaunchValue, RegionAnalysis,
};

/// A borrowed ABI array as a Rust slice, empty when there is nothing to read.
///
/// [`std::slice::from_raw_parts`] is undefined behaviour on a null pointer
/// *even at length zero*: it requires a non-null, aligned, dereferenceable
/// address, and the empty slice is spelled with a dangling-but-aligned pointer,
/// not with null. Every `Pie*Slice` in this crate documents `ptr` as "may be
/// null only when `len == 0`", so an empty table arrives as exactly the pointer
/// that call rejects, and the guard is not defensive — it is the common case.
///
/// The length is checked as well as the pointer, so that a null pointer paired
/// with a nonzero length reads as empty rather than as a wild access. That pair
/// is malformed, and [`local::validate_program_desc`] rejects it — but a driver
/// that adopts before it validates, or that skips validation, must not get
/// undefined behaviour out of the difference.
///
/// # Safety
///
/// When `ptr` is non-null and `len` is nonzero, `ptr` must point at `len`
/// initialised, correctly-aligned `T`s that stay live and unaliased for the
/// returned slice's lifetime. The lifetime is unbounded, so callers must not
/// let the slice outlive the array the host owns.
///
/// [`local::validate_program_desc`]: crate::local::validate_program_desc
unsafe fn slice_of<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: the caller guarantees `len` live, aligned, initialised `T`s
        // at `ptr`; the null and empty cases took the branch above.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// A borrowed `u8` table, copied.
///
/// # Safety
///
/// `slice` must describe a live, correctly-aligned array (see [`slice_of`]).
unsafe fn u8s(slice: PieU8Slice) -> Vec<u8> {
    // SAFETY: the caller guarantees the pointer/len pair describes a live array.
    unsafe { slice_of(slice.ptr, slice.len) }.to_vec()
}

/// A borrowed `u32` table, copied.
///
/// # Safety
///
/// `slice` must describe a live, correctly-aligned array (see [`slice_of`]).
unsafe fn u32s(slice: PieU32Slice) -> Vec<u32> {
    // SAFETY: the caller guarantees the pointer/len pair describes a live array.
    unsafe { slice_of(slice.ptr, slice.len) }.to_vec()
}

/// A borrowed byte span, copied.
///
/// Stays bytes. [`LaunchChannel::extern_name`] and [`LaunchPort::const_data`]
/// are `Vec<u8>` on the owned side because one is a rendezvous key compared for
/// equality and the other is a const payload decoded per its dtype: neither is
/// text, and neither would survive the lossy decode [`string`] applies.
///
/// # Safety
///
/// `bytes` must describe a live array (see [`slice_of`]).
unsafe fn bytes(bytes: PieBytes) -> Vec<u8> {
    // SAFETY: the caller guarantees the pointer/len pair describes a live array.
    unsafe { slice_of(bytes.ptr, bytes.len) }.to_vec()
}

/// A borrowed byte span, decoded as text.
///
/// Lossy on purpose. `from_utf8().unwrap()` would panic on a byte the host
/// should never have sent, and a panic unwinding out of the `extern "C"`
/// registration call that leads here aborts the process rather than returning a
/// status code — one malformed kernel name would take down the host and every
/// other program on the device. A replacement character in a name instead fails
/// later, locally, as a kernel that does not link.
///
/// # Safety
///
/// `text` must describe a live array (see [`slice_of`]).
unsafe fn string(text: PieBytes) -> String {
    // SAFETY: the caller guarantees the pointer/len pair describes a live array.
    let raw = unsafe { slice_of(text.ptr, text.len) };
    String::from_utf8_lossy(raw).into_owned()
}

/// A borrowed name table, decoded as text.
///
/// # Safety
///
/// `slice` and every [`PieBytes`] in it must describe live arrays.
unsafe fn strings(slice: PieBytesSlice) -> Vec<String> {
    // SAFETY: the caller guarantees the table and each span within it are live.
    unsafe {
        slice_of(slice.ptr, slice.len)
            .iter()
            .map(|name| string(*name))
            .collect()
    }
}

/// One op table, adopted.
///
/// # Safety
///
/// `ops` and the `args`/`shape` arrays of every record in it must describe live
/// arrays.
unsafe fn adopt_ops(ops: PieLaunchOpSlice) -> Vec<LaunchOp> {
    // SAFETY: the caller guarantees the table and the two arrays hanging off
    // each record are live for the duration of the call.
    unsafe {
        slice_of(ops.ptr, ops.len)
            .iter()
            .map(|op| LaunchOp {
                code: op.code,
                result_count: op.result_count,
                result_id: op.result_id,
                intrinsic: op.intrinsic,
                lit_dtype: op.lit_dtype,
                dtype: op.dtype,
                pred_tag: op.pred_tag,
                rng_kind: op.rng_kind,
                lit_bits: op.lit_bits,
                pred_payload: op.pred_payload,
                channel: op.channel,
                name_index: op.name_index,
                imm: op.imm,
                imm2: op.imm2,
                imm3: op.imm3,
                args: u32s(op.args),
                shape: u32s(op.shape),
            })
            .collect()
    }
}

/// One `(channel, value)` table, adopted.
///
/// # Safety
///
/// `puts` must describe a live array.
unsafe fn adopt_puts(puts: PieLaunchPutSlice) -> Vec<LaunchPut> {
    // SAFETY: the caller guarantees the table is live; the records are POD.
    unsafe {
        slice_of(puts.ptr, puts.len)
            .iter()
            .map(|put| LaunchPut {
                channel: put.channel,
                value: put.value,
            })
            .collect()
    }
}

/// One region partition, adopted.
///
/// # Safety
///
/// `regions` and the node/input/output/sink arrays of every record in it must
/// describe live arrays.
unsafe fn adopt_regions(regions: PieLaunchRegionSlice) -> Vec<LaunchRegion> {
    // SAFETY: the caller guarantees the table and every array reachable from a
    // record in it are live for the duration of the call.
    unsafe {
        slice_of(regions.ptr, regions.len)
            .iter()
            .map(|region| LaunchRegion {
                kind: region.kind,
                library: region.library,
                schedule: region.schedule,
                nodes: u32s(region.nodes),
                inputs: u32s(region.inputs),
                outputs: u32s(region.outputs),
                sinks: adopt_puts(region.sinks),
            })
            .collect()
    }
}

/// One normalized value-type table, adopted.
///
/// # Safety
///
/// `values` and the `extents`/`dims` arrays of every record in it must describe
/// live arrays.
unsafe fn adopt_plan_values(values: PieLaunchPlanValueSlice) -> Vec<LaunchPlanValue> {
    // SAFETY: the caller guarantees the table and the two arrays hanging off
    // each record are live for the duration of the call.
    unsafe {
        slice_of(values.ptr, values.len)
            .iter()
            .map(|value| LaunchPlanValue {
                dtype: value.dtype,
                extents: u8s(value.extents),
                dims: u32s(value.dims),
            })
            .collect()
    }
}

/// One lane-binding rule table, adopted.
///
/// # Safety
///
/// `rules` must describe a live array.
unsafe fn adopt_channel_rules(rules: PieLaunchChannelRuleSlice) -> Vec<LaunchChannelRule> {
    // SAFETY: the caller guarantees the table is live; the records are POD.
    unsafe {
        slice_of(rules.ptr, rules.len)
            .iter()
            .map(|rule| LaunchChannelRule {
                value: rule.value,
                local: rule.local,
            })
            .collect()
    }
}

/// The ragged source-op map, un-flattened.
///
/// The owned side is a `Vec<Vec<u32>>`; the ABI carries it as one flat array
/// plus a per-op length, so this inverts `launch_abi.rs`'s
/// `flatten().copied().collect()` and its parallel `counts`. An entry covering
/// no source op is a legal empty run and must survive as an empty inner vector,
/// which is why the cursor advances by the count rather than by the run's
/// contents.
///
/// Counts that overrun the flat array are **clamped**, not trusted. They can
/// only appear in a package the host built wrong, and the natural spelling —
/// `flat[cursor..cursor + count]` — would panic on exactly that input, which
/// this module cannot afford (see the module docs). A short final run is a
/// visible, local wrongness; an abort is not.
///
/// # Safety
///
/// `flat` and `counts` must describe live arrays.
unsafe fn adopt_source_ops(flat: PieU32Slice, counts: PieU32Slice) -> Vec<Vec<u32>> {
    // SAFETY: the caller guarantees both pointer/len pairs describe live arrays.
    let (flat, counts) = unsafe {
        (
            slice_of(flat.ptr, flat.len),
            slice_of(counts.ptr, counts.len),
        )
    };
    let mut out = Vec::with_capacity(counts.len());
    let mut cursor = 0usize;
    for &count in counts {
        let end = cursor.saturating_add(count as usize).min(flat.len());
        out.push(flat[cursor..end].to_vec());
        cursor = end;
    }
    out
}

/// One stage's launch plan, adopted.
///
/// # Safety
///
/// Every pointer/len pair reachable from `plan` must describe a live,
/// correctly-aligned array for the duration of the call.
unsafe fn adopt_plan(plan: &PieLaunchStagePlan) -> LaunchStagePlan {
    // SAFETY: the caller guarantees every array reachable from the plan — its
    // ops and their operands, both region partitions and their sinks, the
    // value types, the name table and the error text — is live.
    unsafe {
        LaunchStagePlan {
            signature_hash: plan.signature_hash,
            identity: plan.identity,
            flags: plan.flags,
            mtp_rows: plan.mtp_rows,
            ops: adopt_ops(plan.ops),
            source_ops: adopt_source_ops(plan.source_ops, plan.source_op_counts),
            value_types: adopt_plan_values(plan.value_types),
            channel_bindings: u32s(plan.channel_bindings),
            names: strings(plan.names),
            singleton: adopt_regions(plan.singleton),
            fused: adopt_regions(plan.fused),
            used_extents: u8s(plan.used_extents),
            channel_rules: adopt_channel_rules(plan.channel_rules),
            error: string(plan.error),
        }
    }
}

/// The per-stage launch plan table, adopted.
///
/// # Safety
///
/// `plans` and every array reachable from a record in it must describe live
/// arrays.
unsafe fn adopt_plans(plans: PieLaunchStagePlanSlice) -> Vec<LaunchStagePlan> {
    // SAFETY: the caller guarantees the table and everything reachable from a
    // plan in it are live for the duration of the call.
    unsafe {
        let plans = slice_of(plans.ptr, plans.len);
        // Sized from the *checked* slice rather than from the raw length: a
        // null pointer paired with a huge length would otherwise reserve for a
        // table that does not exist and abort on the allocation failure.
        let mut out = Vec::with_capacity(plans.len());
        for plan in plans {
            out.push(adopt_plan(plan));
        }
        out
    }
}

/// The SSA value table, adopted.
///
/// # Safety
///
/// `values` and the `shape` of every record in it must describe live arrays.
unsafe fn adopt_values(values: PieLaunchValueSlice) -> Vec<LaunchValue> {
    // SAFETY: the caller guarantees the table and each record's shape are live.
    unsafe {
        slice_of(values.ptr, values.len)
            .iter()
            .map(|value| LaunchValue {
                id: value.id,
                source: value.source,
                dtype: value.dtype,
                intrinsic: value.intrinsic,
                channel: value.channel,
                literal_bits: value.literal_bits,
                shape: u32s(value.shape),
            })
            .collect()
    }
}

/// The channel table, adopted.
///
/// # Safety
///
/// `channels` and the `shape`/`extern_name` of every record in it must describe
/// live arrays.
unsafe fn adopt_channels(channels: PieLaunchChannelSlice) -> Vec<LaunchChannel> {
    // SAFETY: the caller guarantees the table, each record's shape and each
    // record's rendezvous name are live.
    unsafe {
        slice_of(channels.ptr, channels.len)
            .iter()
            .map(|channel| LaunchChannel {
                id: channel.id,
                capacity: channel.capacity,
                dtype: channel.dtype,
                flags: channel.flags,
                extern_dir: channel.extern_dir,
                readiness: channel.readiness,
                shape: u32s(channel.shape),
                extern_name: bytes(channel.extern_name),
            })
            .collect()
    }
}

/// The descriptor-port table, adopted.
///
/// `is_const` narrows a C byte to a `bool`, and **any** nonzero byte is true:
/// the field is documented as "nonzero when the port was const-folded", not as
/// one. Testing `== 1` would read a host that wrote `0xff` as a channel-bound
/// port and consume a channel the program never bound.
///
/// # Safety
///
/// `ports` and the `const_shape`/`const_data` of every record in it must
/// describe live arrays.
unsafe fn adopt_ports(ports: PieLaunchPortSlice) -> Vec<LaunchPort> {
    // SAFETY: the caller guarantees the table and each record's const shape and
    // payload are live.
    unsafe {
        slice_of(ports.ptr, ports.len)
            .iter()
            .map(|port| LaunchPort {
                port: port.port,
                is_const: port.is_const != 0,
                const_dtype: port.const_dtype,
                channel: port.channel,
                const_shape: u32s(port.const_shape),
                const_data: bytes(port.const_data),
            })
            .collect()
    }
}

/// The stage table, adopted.
///
/// # Safety
///
/// `stages` and every array reachable from a record in it must describe live
/// arrays.
unsafe fn adopt_stages(stages: PieLaunchStageSlice) -> Vec<LaunchStage> {
    // SAFETY: the caller guarantees the table, each stage's ops and their
    // operand arrays, and each stage's puts/takes/reads are live.
    unsafe {
        slice_of(stages.ptr, stages.len)
            .iter()
            .map(|stage| LaunchStage {
                kind: stage.kind,
                ops: adopt_ops(stage.ops),
                puts: adopt_puts(stage.puts),
                takes: u32s(stage.takes),
                reads: u32s(stage.reads),
            })
            .collect()
    }
}

/// Adopt the launch package a host shipped across the C ABI.
///
/// A field copy, in the sense the C++ predecessor's comment meant: every
/// decision was made on the host, and nothing here validates, re-derives or
/// rejects. The result owns all of its buffers, so the host's arrays may be
/// freed the moment this returns — which is what lets a driver keep the package
/// (and the plan it derives from it) for the life of the program.
///
/// Named `adopt_package` rather than `adopt_launch_package` because the latter
/// already means something else one layer up: `driver_pipeline`'s
/// `adopt_launch_package` takes an *owned* package and returns an `ExecPlan`.
/// Two functions with one name on either side of a boundary a driver crosses in
/// a single statement would read as a typo at the call site rather than as two
/// steps.
///
/// # Safety
///
/// Every pointer/len pair reachable from `package` — the six top-level tables,
/// the arrays inside each of their records, and the arrays inside those — must
/// describe a live, correctly-aligned, initialised array for the duration of
/// the call. A null pointer is tolerated at any depth and reads as empty; a
/// pointer to freed or misaligned memory is not.
pub unsafe fn adopt_package(package: &PieLaunchPackage) -> LaunchPackage {
    // SAFETY: the caller guarantees every array reachable from the package is
    // live and aligned for this call; nothing below retains a borrow of one.
    unsafe {
        LaunchPackage {
            values: adopt_values(package.values),
            channels: adopt_channels(package.channels),
            ports: adopt_ports(package.ports),
            names: strings(package.names),
            stages: adopt_stages(package.stages),
            plans: adopt_plans(package.plans),
        }
    }
}

/// Adopt the host-emitted kernel table a [`PieProgramDesc`] carries.
///
/// A separate call because it is a separate field of the descriptor: the
/// kernels are the host's code generation for this driver's backend, joined to
/// the package on `(kind, stage_index, region_index)`. A driver with no code
/// generation is shipped an empty table and gets an empty vector.
///
/// # Safety
///
/// `slice`, and the `entry_name`/`source`/`error` spans of every record in it,
/// must describe live, correctly-aligned arrays for the duration of the call.
///
/// [`PieProgramDesc`]: crate::local::PieProgramDesc
pub unsafe fn adopt_emitted_kernels(slice: PieEmittedKernelSlice) -> Vec<EmittedKernel> {
    // SAFETY: the caller guarantees the table and the three text spans hanging
    // off each record are live.
    unsafe {
        slice_of(slice.ptr, slice.len)
            .iter()
            .map(|kernel| EmittedKernel {
                kind: kernel.kind,
                stage_index: kernel.stage_index,
                region_index: kernel.region_index,
                entry_name: string(kernel.entry_name),
                source: string(kernel.source),
                error: string(kernel.error),
            })
            .collect()
    }
}

/// Adopt the per-region analysis table a [`PieProgramDesc`] carries.
///
/// An empty vector is not the same as "no rewrite is legal anywhere": it means
/// the host said nothing about any region. A driver reads the flags per region
/// rather than assuming a default, so adoption returns exactly as many records
/// as the host sent and invents none.
///
/// # Safety
///
/// `slice`, and the `direct_argmax`/`skipped` arrays of every record in it,
/// must describe live, correctly-aligned arrays for the duration of the call.
///
/// [`PieProgramDesc`]: crate::local::PieProgramDesc
pub unsafe fn adopt_region_analysis(slice: PieRegionAnalysisSlice) -> Vec<RegionAnalysis> {
    // SAFETY: the caller guarantees the table and the two arrays hanging off
    // each record are live.
    unsafe {
        slice_of(slice.ptr, slice.len)
            .iter()
            .map(|analysis| RegionAnalysis {
                stage_index: analysis.stage_index,
                region_index: analysis.region_index,
                flags: analysis.flags,
                direct_argmax: slice_of(analysis.direct_argmax.ptr, analysis.direct_argmax.len)
                    .iter()
                    .map(|argmax| DirectArgmax {
                        node: argmax.node,
                        source_value: argmax.source_value,
                        intrinsic: argmax.intrinsic,
                        requires_single_row: argmax.requires_single_row,
                    })
                    .collect(),
                skipped: u32s(analysis.skipped),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::local::{
        PIE_CHANNEL_HOST_READER, PIE_CHANNEL_HOST_VISIBLE, PIE_CHANNEL_SEEDED, PIE_EXTENT_STATIC,
        PIE_KERNEL_FUSED, PIE_KERNEL_SINGLETON, PIE_NO_CHANNEL, PIE_READINESS_NEEDS_EMPTY,
        PIE_READINESS_NEEDS_FULL, PIE_REGION_GENERATED, PIE_REGION_GENERATED_VALID,
        PIE_REGION_LIBRARY, PIE_REGION_SECOND_PARTY_SUPPORTED, PIE_STAGE_GROUPED_VALID,
        PIE_STAGE_REQUIRES_MTP_ROWS, PIE_STAGE_REQUIRES_QUERY, PIE_VALUE_CHANNEL_READ,
        PIE_VALUE_CHANNEL_TAKE, PIE_VALUE_CONST, PIE_VALUE_INTRINSIC, PieDirectArgmax,
        PieDirectArgmaxSlice, PieEmittedKernel, PieLaunchChannel, PieLaunchChannelRule,
        PieLaunchOp, PieLaunchPlanValue, PieLaunchPort, PieLaunchPut, PieLaunchRegion,
        PieLaunchStage, PieLaunchValue, PieRegionAnalysis,
    };

    fn u8s_of(slice: &[u8]) -> PieU8Slice {
        PieU8Slice {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }

    fn u32s_of(slice: &[u32]) -> PieU32Slice {
        PieU32Slice {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }

    fn bytes_of(slice: &[u8]) -> PieBytes {
        PieBytes {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }

    /// The C view of an owned package, materialised by hand.
    ///
    /// Deliberately not the engine's `LaunchPackageBorrow`, which does exactly
    /// this: `engine` depends on this crate, so reaching back for it is not
    /// available at any price. Writing the direction out here also keeps the
    /// two independent — the assertion is that two separately written mappings
    /// agree, not that one function is its own inverse.
    ///
    /// Every record is a full struct literal with no `..Default::default()`, so
    /// a field added to `local.rs` fails to compile here rather than quietly
    /// round-tripping as a zero.
    ///
    /// The nesting is sound for the reason `launch_abi.rs` states: moving a
    /// `Vec` moves its header and not its heap buffer, so a pointer taken into
    /// an inner vector stays valid when the outer one grows.
    struct CView<'a> {
        _package: &'a LaunchPackage,
        values: Vec<PieLaunchValue>,
        channels: Vec<PieLaunchChannel>,
        ports: Vec<PieLaunchPort>,
        names: Vec<PieBytes>,
        stages: Vec<PieLaunchStage>,
        plans: Vec<PieLaunchStagePlan>,
        ops: Vec<Vec<PieLaunchOp>>,
        puts: Vec<Vec<PieLaunchPut>>,
        sinks: Vec<Vec<PieLaunchPut>>,
        regions: Vec<Vec<PieLaunchRegion>>,
        plan_values: Vec<Vec<PieLaunchPlanValue>>,
        rules: Vec<Vec<PieLaunchChannelRule>>,
        name_tables: Vec<Vec<PieBytes>>,
        source_ops: Vec<Vec<u32>>,
        source_op_counts: Vec<Vec<u32>>,
    }

    impl<'a> CView<'a> {
        fn new(package: &'a LaunchPackage) -> Self {
            let mut view = CView {
                _package: package,
                values: Vec::new(),
                channels: Vec::new(),
                ports: Vec::new(),
                names: Vec::new(),
                stages: Vec::new(),
                plans: Vec::new(),
                ops: Vec::new(),
                puts: Vec::new(),
                sinks: Vec::new(),
                regions: Vec::new(),
                plan_values: Vec::new(),
                rules: Vec::new(),
                name_tables: Vec::new(),
                source_ops: Vec::new(),
                source_op_counts: Vec::new(),
            };
            view.values = package
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
                    shape: u32s_of(&value.shape),
                })
                .collect();
            view.channels = package
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
                    shape: u32s_of(&channel.shape),
                    extern_name: bytes_of(&channel.extern_name),
                })
                .collect();
            view.ports = package
                .ports
                .iter()
                .map(|port| PieLaunchPort {
                    port: port.port,
                    is_const: u8::from(port.is_const),
                    const_dtype: port.const_dtype,
                    reserved0: 0,
                    channel: port.channel,
                    const_shape: u32s_of(&port.const_shape),
                    const_data: bytes_of(&port.const_data),
                })
                .collect();
            view.names = package
                .names
                .iter()
                .map(|name| bytes_of(name.as_bytes()))
                .collect();
            let stages: Vec<PieLaunchStage> = package
                .stages
                .iter()
                .map(|stage| PieLaunchStage {
                    kind: stage.kind,
                    reserved0: 0,
                    reserved1: 0,
                    reserved2: 0,
                    ops: view.push_ops(&stage.ops),
                    puts: view.push_puts(&stage.puts),
                    takes: u32s_of(&stage.takes),
                    reads: u32s_of(&stage.reads),
                })
                .collect();
            view.stages = stages;
            let plans: Vec<PieLaunchStagePlan> = package
                .plans
                .iter()
                .map(|plan| view.push_plan(plan))
                .collect();
            view.plans = plans;
            view
        }

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
                    args: u32s_of(&op.args),
                    shape: u32s_of(&op.shape),
                })
                .collect();
            let slice = PieLaunchOpSlice {
                ptr: records.as_ptr(),
                len: records.len(),
            };
            self.ops.push(records);
            slice
        }

        fn push_puts(&mut self, puts: &[LaunchPut]) -> PieLaunchPutSlice {
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

        fn push_sinks(&mut self, sinks: &[LaunchPut]) -> PieLaunchPutSlice {
            let records: Vec<PieLaunchPut> = sinks
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
        }

        fn push_regions(&mut self, regions: &[LaunchRegion]) -> PieLaunchRegionSlice {
            let sinks: Vec<PieLaunchPutSlice> = regions
                .iter()
                .map(|region| self.push_sinks(&region.sinks))
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
                    nodes: u32s_of(&region.nodes),
                    inputs: u32s_of(&region.inputs),
                    outputs: u32s_of(&region.outputs),
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

        fn push_names(&mut self, names: &'a [String]) -> PieBytesSlice {
            let records: Vec<PieBytes> =
                names.iter().map(|name| bytes_of(name.as_bytes())).collect();
            let slice = PieBytesSlice {
                ptr: records.as_ptr(),
                len: records.len(),
            };
            self.name_tables.push(records);
            slice
        }

        fn push_plan(&mut self, plan: &'a LaunchStagePlan) -> PieLaunchStagePlan {
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
            let source_ops = u32s_of(&flat);
            let source_op_counts = u32s_of(&counts);
            self.source_ops.push(flat);
            self.source_op_counts.push(counts);

            let values: Vec<PieLaunchPlanValue> = plan
                .value_types
                .iter()
                .map(|value| PieLaunchPlanValue {
                    dtype: value.dtype,
                    reserved0: 0,
                    reserved1: 0,
                    extents: u8s_of(&value.extents),
                    dims: u32s_of(&value.dims),
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
                channel_bindings: u32s_of(&plan.channel_bindings),
                names,
                singleton,
                fused,
                used_extents: u8s_of(&plan.used_extents),
                channel_rules,
                error: bytes_of(plan.error.as_bytes()),
            }
        }

        fn raw(&self) -> PieLaunchPackage {
            PieLaunchPackage {
                values: PieLaunchValueSlice {
                    ptr: self.values.as_ptr(),
                    len: self.values.len(),
                },
                channels: PieLaunchChannelSlice {
                    ptr: self.channels.as_ptr(),
                    len: self.channels.len(),
                },
                ports: PieLaunchPortSlice {
                    ptr: self.ports.as_ptr(),
                    len: self.ports.len(),
                },
                names: PieBytesSlice {
                    ptr: self.names.as_ptr(),
                    len: self.names.len(),
                },
                stages: PieLaunchStageSlice {
                    ptr: self.stages.as_ptr(),
                    len: self.stages.len(),
                },
                plans: PieLaunchStagePlanSlice {
                    ptr: self.plans.as_ptr(),
                    len: self.plans.len(),
                },
            }
        }
    }

    /// An op with every scalar field distinct, so a mapping that crosses two of
    /// them shows up as a mismatch rather than as a coincidence.
    fn op(seed: u32) -> LaunchOp {
        LaunchOp {
            code: 0x0100 + seed as u16,
            result_count: 2,
            result_id: 0x0200 + seed,
            intrinsic: 0x0300 + seed as u16,
            lit_dtype: 4,
            dtype: 5,
            pred_tag: 6,
            rng_kind: 1,
            lit_bits: 0x0700_0000 + seed,
            pred_payload: 0x0800_0000 + seed,
            channel: 0x0900 + seed,
            name_index: seed,
            imm: 0x0a00 + seed,
            imm2: 0x0b00 + seed,
            imm3: 0x0c00 + seed,
            args: vec![0x0d00 + seed, 0x0e00 + seed],
            shape: vec![2, 3 + seed],
        }
    }

    fn generated_region() -> LaunchRegion {
        LaunchRegion {
            kind: PIE_REGION_GENERATED,
            library: 0,
            schedule: 1,
            nodes: vec![0, 1, 2],
            inputs: vec![7],
            outputs: vec![9, 10],
            sinks: vec![LaunchPut {
                channel: 11,
                value: 9,
            }],
        }
    }

    fn library_region() -> LaunchRegion {
        LaunchRegion {
            kind: PIE_REGION_LIBRARY,
            library: 4,
            schedule: 3,
            nodes: vec![3],
            inputs: vec![9, 10],
            outputs: vec![12],
            sinks: Vec::new(),
        }
    }

    /// A package with every field of every record set to a distinct non-default
    /// value: two stages with different op counts, a plan whose `source_ops`
    /// runs are ragged and include an empty one, regions of both kinds, a const
    /// port and a channel-bound one, a private channel and an exported one, and
    /// a stage plan carrying an error string.
    fn full_package() -> LaunchPackage {
        LaunchPackage {
            values: vec![
                LaunchValue {
                    id: 7,
                    source: PIE_VALUE_CHANNEL_TAKE,
                    dtype: 3,
                    intrinsic: 0,
                    channel: 11,
                    literal_bits: 0,
                    shape: vec![2, 3],
                },
                LaunchValue {
                    id: 8,
                    source: PIE_VALUE_CONST,
                    dtype: 2,
                    intrinsic: 0,
                    channel: PIE_NO_CHANNEL,
                    literal_bits: 0xdead_beef,
                    shape: Vec::new(),
                },
                LaunchValue {
                    id: 9,
                    source: PIE_VALUE_INTRINSIC,
                    dtype: 4,
                    intrinsic: 6,
                    channel: PIE_NO_CHANNEL,
                    literal_bits: 0,
                    shape: vec![128],
                },
                LaunchValue {
                    id: 10,
                    source: PIE_VALUE_CHANNEL_READ,
                    dtype: 1,
                    intrinsic: 0,
                    channel: 12,
                    literal_bits: 0,
                    shape: vec![1],
                },
            ],
            channels: vec![
                LaunchChannel {
                    id: 11,
                    capacity: 4,
                    dtype: 3,
                    flags: PIE_CHANNEL_SEEDED | PIE_CHANNEL_HOST_VISIBLE,
                    extern_dir: -1,
                    readiness: PIE_READINESS_NEEDS_FULL,
                    shape: vec![2, 3],
                    extern_name: Vec::new(),
                },
                LaunchChannel {
                    id: 12,
                    capacity: 1,
                    dtype: 1,
                    flags: PIE_CHANNEL_HOST_VISIBLE | PIE_CHANNEL_HOST_READER,
                    extern_dir: 1,
                    readiness: PIE_READINESS_NEEDS_EMPTY,
                    shape: vec![1],
                    extern_name: b"rendezvous/out".to_vec(),
                },
            ],
            ports: vec![
                LaunchPort {
                    port: 3,
                    is_const: true,
                    const_dtype: 2,
                    channel: PIE_NO_CHANNEL,
                    const_shape: vec![1, 2],
                    const_data: vec![0xde, 0xad, 0xbe, 0xef],
                },
                LaunchPort {
                    port: 6,
                    is_const: false,
                    const_dtype: 0,
                    channel: 12,
                    const_shape: Vec::new(),
                    const_data: Vec::new(),
                },
            ],
            names: vec![
                "program::second_party".to_string(),
                "program::sink".to_string(),
            ],
            stages: vec![
                LaunchStage {
                    kind: 0,
                    ops: vec![op(1), op(2), op(3)],
                    puts: vec![
                        LaunchPut {
                            channel: 11,
                            value: 7,
                        },
                        LaunchPut {
                            channel: 12,
                            value: 8,
                        },
                    ],
                    takes: vec![11],
                    reads: vec![12],
                },
                LaunchStage {
                    kind: 3,
                    ops: vec![op(4)],
                    puts: Vec::new(),
                    takes: Vec::new(),
                    reads: vec![11, 12],
                },
            ],
            plans: vec![
                LaunchStagePlan {
                    signature_hash: 0x1122_3344_5566_7788,
                    identity: 0x99aa_bbcc_ddee_ff00,
                    flags: PIE_STAGE_GROUPED_VALID | PIE_STAGE_REQUIRES_QUERY,
                    mtp_rows: 0,
                    ops: vec![op(5), op(6), op(7)],
                    // Ragged, and the middle run is empty: a normalized op that
                    // covers no source op must survive as an empty inner vector
                    // rather than shifting every run after it.
                    source_ops: vec![vec![0, 1], Vec::new(), vec![2]],
                    value_types: vec![
                        LaunchPlanValue {
                            dtype: 3,
                            extents: vec![PIE_EXTENT_STATIC, 2],
                            dims: vec![16, 0],
                        },
                        LaunchPlanValue {
                            dtype: 1,
                            extents: vec![0],
                            dims: vec![0],
                        },
                    ],
                    channel_bindings: vec![11, 12],
                    names: vec!["stage::kernel_a".to_string()],
                    singleton: vec![generated_region()],
                    fused: vec![library_region(), generated_region()],
                    used_extents: vec![0, 2, 5],
                    channel_rules: vec![
                        LaunchChannelRule { value: 4, local: 0 },
                        LaunchChannelRule { value: 6, local: 1 },
                    ],
                    error: String::new(),
                },
                LaunchStagePlan {
                    signature_hash: 0xfeed_face_dead_beef,
                    identity: 0x0102_0304_0506_0708,
                    flags: PIE_STAGE_REQUIRES_MTP_ROWS,
                    mtp_rows: 3,
                    ops: vec![op(8)],
                    source_ops: vec![vec![0]],
                    value_types: Vec::new(),
                    channel_bindings: Vec::new(),
                    names: Vec::new(),
                    singleton: Vec::new(),
                    fused: Vec::new(),
                    used_extents: Vec::new(),
                    channel_rules: Vec::new(),
                    error: "grouped path rejected: unsupported intrinsic".to_string(),
                },
            ],
        }
    }

    #[test]
    fn adoption_round_trips_a_package_with_every_field_set() {
        let package = full_package();
        let view = CView::new(&package);
        let raw = view.raw();
        // SAFETY: `view` owns every record array `raw` points at, borrows the
        // package the leaf arrays live in, and outlives this call.
        let adopted = unsafe { adopt_package(&raw) };
        assert_eq!(
            adopted, package,
            "adoption must be the exact inverse of the engine's materialisation; \
             a field left at its default here is a program that runs with a \
             channel unbound or an immediate lost, and nothing else would catch it"
        );
    }

    #[test]
    fn every_scalar_field_of_an_op_lands_in_its_own_counterpart_and_not_a_neighbour() {
        // The round trip cannot tell a mapping that swaps two fields
        // consistently in both directions from a correct one, so the C record
        // here is written by hand with a distinct small value per field.
        let args = [21u32, 22, 23];
        let shape = [24u32, 25];
        let records = [PieLaunchOp {
            code: 1,
            result_count: 2,
            result_id: 3,
            intrinsic: 4,
            lit_dtype: 5,
            dtype: 6,
            pred_tag: 7,
            rng_kind: 8,
            reserved0: 0,
            lit_bits: 9,
            pred_payload: 10,
            channel: 11,
            name_index: 12,
            imm: 13,
            imm2: 14,
            imm3: 15,
            reserved1: 0,
            args: u32s_of(&args),
            shape: u32s_of(&shape),
        }];
        let slice = PieLaunchOpSlice {
            ptr: records.as_ptr(),
            len: records.len(),
        };
        // SAFETY: `records`, `args` and `shape` are live locals for this call.
        let adopted = unsafe { adopt_ops(slice) };
        assert_eq!(
            adopted,
            vec![LaunchOp {
                code: 1,
                result_count: 2,
                result_id: 3,
                intrinsic: 4,
                lit_dtype: 5,
                dtype: 6,
                pred_tag: 7,
                rng_kind: 8,
                lit_bits: 9,
                pred_payload: 10,
                channel: 11,
                name_index: 12,
                imm: 13,
                imm2: 14,
                imm3: 15,
                args: vec![21, 22, 23],
                shape: vec![24, 25],
            }]
        );
    }

    #[test]
    fn a_regions_three_adjacent_bytes_are_not_shuffled_between_kind_library_and_schedule() {
        // `kind`, `library` and `schedule` are three consecutive `u8`s that mean
        // entirely different things — which partition served the region, which
        // vendor call, and which launch shape. The round trip would accept a
        // mapping that permuted them consistently in both directions, so the
        // record here is written by hand.
        let nodes = [3u32, 4];
        let inputs = [5u32];
        let outputs = [6u32, 7];
        let sinks = [PieLaunchPut {
            channel: 8,
            value: 9,
        }];
        let records = [PieLaunchRegion {
            kind: PIE_REGION_LIBRARY,
            library: 4,
            schedule: 2,
            reserved0: 0,
            reserved1: 0,
            nodes: u32s_of(&nodes),
            inputs: u32s_of(&inputs),
            outputs: u32s_of(&outputs),
            sinks: PieLaunchPutSlice {
                ptr: sinks.as_ptr(),
                len: sinks.len(),
            },
        }];
        let slice = PieLaunchRegionSlice {
            ptr: records.as_ptr(),
            len: records.len(),
        };
        // SAFETY: the records and every array they point at are live locals.
        let adopted = unsafe { adopt_regions(slice) };
        assert_eq!(
            adopted,
            vec![LaunchRegion {
                kind: PIE_REGION_LIBRARY,
                library: 4,
                schedule: 2,
                nodes: vec![3, 4],
                inputs: vec![5],
                outputs: vec![6, 7],
                sinks: vec![LaunchPut {
                    channel: 8,
                    value: 9
                }],
            }]
        );
    }

    #[test]
    fn null_slices_adopt_as_empty_vectors_rather_than_dereferencing_null() {
        // Every pointer in a default descriptor is null. `from_raw_parts` is
        // undefined behaviour on one even at length zero, so this is the case
        // the guard in `slice_of` exists for, and it is what a driver sees for
        // any table the host did not populate.
        let raw = PieLaunchPackage::default();
        // SAFETY: every pointer is null with a zero length, which `slice_of`
        // answers without a dereference.
        let adopted = unsafe { adopt_package(&raw) };
        assert_eq!(adopted, LaunchPackage::default());

        // SAFETY: as above.
        let kernels = unsafe { adopt_emitted_kernels(PieEmittedKernelSlice::default()) };
        assert!(kernels.is_empty());
        // SAFETY: as above.
        let analysis = unsafe { adopt_region_analysis(PieRegionAnalysisSlice::default()) };
        assert!(analysis.is_empty());
    }

    #[test]
    fn a_null_pointer_with_a_nonzero_length_is_empty_rather_than_a_wild_read() {
        // `validate_slice_ptr` rejects this pair, so a validated package never
        // contains one. Adoption still has to survive it: a driver may adopt
        // before it validates, or not validate at all, and the difference
        // between the two orders must not be undefined behaviour.
        let raw = PieLaunchPackage {
            values: PieLaunchValueSlice {
                ptr: std::ptr::null(),
                len: 4,
            },
            ..PieLaunchPackage::default()
        };
        // SAFETY: the only non-empty-looking table has a null pointer, which
        // `slice_of` refuses to dereference.
        let adopted = unsafe { adopt_package(&raw) };
        assert!(adopted.values.is_empty());
    }

    #[test]
    fn a_name_that_is_not_utf8_is_decoded_lossily_rather_than_panicking_across_the_abi() {
        // 0x80 is a continuation byte with no lead byte. `from_utf8().unwrap()`
        // would panic here, and a panic reaching the `extern "C"` frame that
        // called adoption aborts the host process instead of returning a status
        // code.
        let name = [b'f', b'o', 0x80, b'o'];
        let spans = [bytes_of(&name)];
        let raw = PieLaunchPackage {
            names: PieBytesSlice {
                ptr: spans.as_ptr(),
                len: spans.len(),
            },
            ..PieLaunchPackage::default()
        };
        // SAFETY: `name` and `spans` are live locals for the call.
        let adopted = unsafe { adopt_package(&raw) };
        assert_eq!(adopted.names, vec!["fo\u{fffd}o".to_string()]);
    }

    #[test]
    fn a_source_op_count_that_overruns_the_flattened_array_is_clamped_rather_than_panicking() {
        // A host that sent counts summing past the flat array is broken, but
        // `flat[cursor..cursor + count]` would turn its bug into an abort of a
        // process that was executing other programs correctly.
        let flat = [5u32, 6, 7];
        let counts = [1u32, 9, 4];
        // SAFETY: both arrays are live locals for the call.
        let adopted = unsafe { adopt_source_ops(u32s_of(&flat), u32s_of(&counts)) };
        assert_eq!(adopted, vec![vec![5], vec![6, 7], Vec::new()]);
    }

    #[test]
    fn a_const_port_flag_is_true_for_any_nonzero_byte_not_only_for_one() {
        // The field is documented as "nonzero when the port was const-folded".
        // A `== 1` test would read a host's 0xff as a channel-bound port and
        // consume a channel the program never bound.
        let records = [PieLaunchPort {
            port: 3,
            is_const: 0xff,
            const_dtype: 2,
            reserved0: 0,
            channel: PIE_NO_CHANNEL,
            const_shape: PieU32Slice::default(),
            const_data: PieBytes::default(),
        }];
        let slice = PieLaunchPortSlice {
            ptr: records.as_ptr(),
            len: records.len(),
        };
        // SAFETY: `records` is a live local for the call.
        let adopted = unsafe { adopt_ports(slice) };
        assert!(adopted[0].is_const);
    }

    #[test]
    fn the_emitted_kernel_table_adopts_a_kernel_and_a_refusal_with_every_field() {
        let entry = b"stage0_region0";
        let source = b"__global__ void k() {}";
        let why = b"too many bound channels";
        let records = [
            PieEmittedKernel {
                kind: PIE_KERNEL_SINGLETON,
                stage_index: 1,
                region_index: 2,
                reserved0: 0,
                entry_name: bytes_of(entry),
                source: bytes_of(source),
                error: PieBytes::default(),
            },
            PieEmittedKernel {
                kind: PIE_KERNEL_FUSED,
                stage_index: 3,
                region_index: 4,
                reserved0: 0,
                entry_name: PieBytes::default(),
                source: PieBytes::default(),
                error: bytes_of(why),
            },
        ];
        let slice = PieEmittedKernelSlice {
            ptr: records.as_ptr(),
            len: records.len(),
        };
        // SAFETY: the records and every span they point at are live locals.
        let adopted = unsafe { adopt_emitted_kernels(slice) };
        assert_eq!(
            adopted,
            vec![
                EmittedKernel {
                    kind: PIE_KERNEL_SINGLETON,
                    stage_index: 1,
                    region_index: 2,
                    entry_name: "stage0_region0".to_string(),
                    source: "__global__ void k() {}".to_string(),
                    error: String::new(),
                },
                EmittedKernel {
                    kind: PIE_KERNEL_FUSED,
                    stage_index: 3,
                    region_index: 4,
                    entry_name: String::new(),
                    source: String::new(),
                    // An empty source with a populated error is a deliberate
                    // refusal — a driver reads it as one only if the text
                    // survives adoption.
                    error: "too many bound channels".to_string(),
                },
            ]
        );
    }

    #[test]
    fn the_region_analysis_table_adopts_its_rewrites_and_its_skipped_nodes() {
        let argmax = [
            PieDirectArgmax {
                node: 7,
                source_value: 8,
                intrinsic: 9,
                requires_single_row: 1,
                reserved0: 0,
            },
            PieDirectArgmax {
                node: 10,
                source_value: 11,
                intrinsic: 12,
                requires_single_row: 0,
                reserved0: 0,
            },
        ];
        let skipped = [1u32, 2, 3];
        let records = [PieRegionAnalysis {
            stage_index: 5,
            region_index: 6,
            flags: PIE_REGION_SECOND_PARTY_SUPPORTED | PIE_REGION_GENERATED_VALID,
            reserved0: 0,
            direct_argmax: PieDirectArgmaxSlice {
                ptr: argmax.as_ptr(),
                len: argmax.len(),
            },
            skipped: u32s_of(&skipped),
        }];
        let slice = PieRegionAnalysisSlice {
            ptr: records.as_ptr(),
            len: records.len(),
        };
        // SAFETY: the records and both arrays they point at are live locals.
        let adopted = unsafe { adopt_region_analysis(slice) };
        assert_eq!(
            adopted,
            vec![RegionAnalysis {
                stage_index: 5,
                region_index: 6,
                flags: PIE_REGION_SECOND_PARTY_SUPPORTED | PIE_REGION_GENERATED_VALID,
                direct_argmax: vec![
                    DirectArgmax {
                        node: 7,
                        source_value: 8,
                        intrinsic: 9,
                        requires_single_row: 1,
                    },
                    DirectArgmax {
                        node: 10,
                        source_value: 11,
                        intrinsic: 12,
                        requires_single_row: 0,
                    },
                ],
                skipped: vec![1, 2, 3],
            }]
        );
    }

    #[test]
    fn a_program_desc_adopts_into_the_registration_a_driver_keeps() {
        // The point of the module: what a driver is handed at
        // `register_program` becomes the owned types every Rust consumer of a
        // launch package reads. Nothing else in this crate joins the two ends.
        let package = full_package();
        let view = CView::new(&package);
        let kernels = [PieEmittedKernel {
            kind: PIE_KERNEL_FUSED,
            stage_index: 0,
            region_index: 0,
            reserved0: 0,
            entry_name: bytes_of(b"stage0_fused0"),
            source: bytes_of(b"__global__ void k() {}"),
            error: PieBytes::default(),
        }];
        let desc = crate::local::PieProgramDesc {
            program_hash: 0x5555_6666_7777_8888,
            emitter_version: 36,
            emitted_kernels: PieEmittedKernelSlice {
                ptr: kernels.as_ptr(),
                len: kernels.len(),
            },
            launch: view.raw(),
            ..crate::local::PieProgramDesc::default()
        };
        // SAFETY: `view` and `kernels` own everything `desc` points at and both
        // outlive these calls.
        let registration = unsafe {
            crate::plan::ProgramRegistration {
                program_hash: desc.program_hash,
                emitted_kernels: adopt_emitted_kernels(desc.emitted_kernels),
                emitter_version: desc.emitter_version,
                region_analysis: adopt_region_analysis(desc.region_analysis),
                launch: adopt_package(&desc.launch),
                reference_ptir: Vec::new(),
            }
        };
        assert_eq!(registration.launch, package);
        assert_eq!(registration.emitted_kernels.len(), 1);
        assert!(registration.region_analysis.is_empty());
    }
}
