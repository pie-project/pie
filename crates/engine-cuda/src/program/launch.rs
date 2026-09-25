use eta_compiler::codegen::launch::{LaunchRegion, LaunchStagePlan};
use eta_compiler::plan::{LibraryOp, RegionKind};
use eta_exec::{
    Extents, LANE_HEADER_BYTES, LANE_RECORD_BYTES, LaneChannelSlot, LaneHeader, LaneRecord,
    LaneShape, Layout, Lifetime, NO_TICKET, OpParams, OpRuntime, SCRATCH_ALIGN, ValueDesc,
    describe, layout, layout_reusing,
};
use eta_ir::Dtype;
use eta_ir::container::HostRole;
use eta_ir::op::{IntrinsicId, tags};
use eta_ir::types::{MAX_RANK, name_or_unknown};

use std::sync::Arc;

use kernels_cuda::channel::MAX_RING;

use crate::device::Buffer;
use crate::error::{Fault, Result};

use super::compile::{Module, Region};
use super::endpoint::Endpoint;

const FUSED_ARITY: usize = 18;

pub const ORDER_ROW_BLOCKS: u32 = 32;

pub const INTRINSIC_SLOTS: usize = IntrinsicId::SLOTS as usize;

pub const INTRINSIC_STORAGE_F32: u32 = 0;

pub const INTRINSIC_STORAGE_RAW_BF16: u32 = 1;

pub const INTRINSIC_STORAGE_ROW_POINTERS: u32 = 2;

pub const INTRINSIC_STORAGE_RAW_I32: u32 = 3;

pub const BOOL_STORAGE_NATIVE_BYTES: u32 = 0;

pub const BOOL_STORAGE_WIRE_PACKED: u32 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct CudaOpParams {
    pub tag: u32,
    pub a0: u32,
    pub a1: u32,
    pub a2: u32,
    pub o0: u32,
    pub o1: u32,
    pub imm: u32,
    pub imm2: u32,
    pub imm3: u32,
    pub kind: u32,
    pub pred_tag: u32,
    pub lit_dtype: u32,
    pub lit_bits: u32,
    pub channel_slot: u32,
    pub intr: u32,
    pub sink_bytes: u32,

    pub intrinsic_dtype: u32,
    pub bool_storage: u32,
    pub intrinsic_row_stride: u32,
    pub intrinsic_row_offset: u32,
    pub rng_seed: u64,
}

const _: () = assert!(size_of::<CudaOpParams>() == 88);

const _: () = {
    assert!(std::mem::offset_of!(CudaOpParams, tag) == 0);
    assert!(std::mem::offset_of!(CudaOpParams, a0) == 4);
    assert!(std::mem::offset_of!(CudaOpParams, a1) == 8);
    assert!(std::mem::offset_of!(CudaOpParams, a2) == 12);
    assert!(std::mem::offset_of!(CudaOpParams, o0) == 16);
    assert!(std::mem::offset_of!(CudaOpParams, o1) == 20);
    assert!(std::mem::offset_of!(CudaOpParams, imm) == 24);
    assert!(std::mem::offset_of!(CudaOpParams, imm2) == 28);
    assert!(std::mem::offset_of!(CudaOpParams, imm3) == 32);
    assert!(std::mem::offset_of!(CudaOpParams, kind) == 36);
    assert!(std::mem::offset_of!(CudaOpParams, pred_tag) == 40);
    assert!(std::mem::offset_of!(CudaOpParams, lit_dtype) == 44);
    assert!(std::mem::offset_of!(CudaOpParams, lit_bits) == 48);
    assert!(std::mem::offset_of!(CudaOpParams, channel_slot) == 52);
    assert!(std::mem::offset_of!(CudaOpParams, intr) == 56);
    assert!(std::mem::offset_of!(CudaOpParams, sink_bytes) == 60);
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_dtype) == 64);
    assert!(std::mem::offset_of!(CudaOpParams, bool_storage) == 68);
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_row_stride) == 72);
    assert!(std::mem::offset_of!(CudaOpParams, intrinsic_row_offset) == 76);
    assert!(std::mem::offset_of!(CudaOpParams, rng_seed) == 80);
};

impl CudaOpParams {
    #[must_use]
    pub const fn widen(shared: OpParams) -> CudaOpParams {
        CudaOpParams {
            tag: shared.tag,
            a0: shared.a0,
            a1: shared.a1,
            a2: shared.a2,
            o0: shared.o0,
            o1: shared.o1,
            imm: shared.imm,
            imm2: shared.imm2,
            imm3: shared.imm3,
            kind: shared.kind,
            pred_tag: shared.pred_tag,
            lit_dtype: shared.lit_dtype,
            lit_bits: shared.lit_bits,
            channel_slot: shared.channel_slot,
            intr: shared.intr,
            sink_bytes: shared.sink_bytes,
            intrinsic_dtype: INTRINSIC_STORAGE_F32,
            bool_storage: BOOL_STORAGE_NATIVE_BYTES,
            intrinsic_row_stride: 0,
            intrinsic_row_offset: 0,
            rng_seed: 0,
        }
    }
}

#[must_use]
pub fn native_cell_bytes(dtype: Dtype, numel: usize) -> usize {
    if dtype == Dtype::Bool {
        numel.max(1)
    } else {
        numel.max(1) * 4
    }
}

pub fn wire_to_native(dtype: Dtype, numel: usize, wire: &[u8]) -> Result<Vec<u8>> {
    let numel = numel.max(1);
    let want = eta_exec::wire_cell_bytes(dtype, numel);
    if wire.len() != want {
        return Err(Fault::program(
            "program::launch",
            format!(
                "a {} wire cell of {numel} lane(s) is {want} bytes and {} were offered",
                name_or_unknown(dtype),
                wire.len()
            ),
        ));
    }
    if dtype != Dtype::Bool {
        return Ok(wire.to_vec());
    }
    Ok((0..numel)
        .map(|i| u8::from(wire[i / 8] >> (i % 8) & 1 == 1))
        .collect())
}

pub fn native_to_wire(dtype: Dtype, numel: usize, native: &[u8]) -> Result<Vec<u8>> {
    let numel = numel.max(1);
    let want = native_cell_bytes(dtype, numel);
    if native.len() != want {
        return Err(Fault::program(
            "program::launch",
            format!(
                "a {} native cell of {numel} lane(s) is {want} bytes and {} were offered",
                name_or_unknown(dtype),
                native.len()
            ),
        ));
    }
    if dtype != Dtype::Bool {
        return Ok(native.to_vec());
    }
    let mut out = vec![0u8; eta_exec::wire_cell_bytes(dtype, numel)];
    for (i, &byte) in native.iter().enumerate().take(numel) {
        if byte != 0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelShape {
    pub numel: usize,
    pub dtype: Dtype,
    pub capacity: u32,
}

impl ChannelShape {
    #[must_use]
    pub fn of(declared: &eta_compiler::codegen::launch::LaunchChannel) -> ChannelShape {
        ChannelShape {
            numel: declared
                .shape
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1),
            dtype: eta_exec::concrete_dtype(declared.dtype),
            capacity: declared.capacity.max(1),
        }
    }

    #[must_use]
    pub const fn ring(&self) -> u64 {
        self.capacity as u64 + 1
    }

    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        native_cell_bytes(self.dtype, self.numel)
    }
}

#[derive(Debug)]
pub struct Rings {
    cells: Vec<Buffer>,
    shapes: Vec<ChannelShape>,
    endpoints: Vec<Option<Arc<Endpoint>>>,
    registry: Buffer,
    device: kernels_cuda::channel::Rings,
}

impl Rings {
    pub fn allocate(
        shapes: &[ChannelShape],
        endpoints: Vec<Option<Arc<Endpoint>>>,
    ) -> Result<Rings> {
        if endpoints.len() != shapes.len() {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "{} channel shape(s) and {} endpoint slot(s): the two are indexed by \
                     the same dense channel number",
                    shapes.len(),
                    endpoints.len()
                ),
            ));
        }
        let mut cells = Vec::with_capacity(shapes.len());
        for (index, shape) in shapes.iter().enumerate() {
            let bytes = shape.cell_bytes();
            if bytes == 0 {
                return Err(Fault::program(
                    "program::launch",
                    format!("channel {index}'s cell is zero bytes, so it can never be ready"),
                ));
            }
            let shared = endpoints
                .get(index)
                .and_then(Option::as_ref)
                .and_then(|endpoint| endpoint.device_cells());
            cells.push(match shared {
                Some(_) => Buffer::zeroed(0)?,
                None => Buffer::zeroed(bytes * shape.ring() as usize)?,
            });
        }

        let slots = shapes.len();
        let words = slots * size_of::<u32>();
        let full_at = 3 * words;
        let mut registry = Buffer::zeroed(full_at + slots * MAX_RING as usize)?;
        let cap1: Vec<u8> = shapes
            .iter()
            .flat_map(|shape| {
                u32::try_from(shape.ring())
                    .unwrap_or(u32::MAX)
                    .to_le_bytes()
            })
            .collect();
        registry.write(2 * words as u64, &cap1)?;
        let device = kernels_cuda::channel::Rings::new(
            registry.at(full_at as u64)?,
            registry.ptr(),
            registry.at(words as u64)?,
            registry.at(2 * words as u64)?,
            u32::try_from(slots).unwrap_or(u32::MAX),
        );

        Ok(Rings {
            cells,
            shapes: shapes.to_vec(),
            endpoints,
            registry,
            device,
        })
    }

    #[must_use]
    pub const fn device(&self) -> kernels_cuda::channel::Rings {
        self.device
    }

    #[must_use]
    pub fn endpoint(&self, channel: usize) -> Option<&Arc<Endpoint>> {
        self.endpoints.get(channel).and_then(Option::as_ref)
    }

    pub fn seed_registry(&mut self, cursors: &[Cursor]) -> Result<()> {
        let slots = self.shapes.len();
        let words = slots * size_of::<u32>();
        let mut head = Vec::with_capacity(words);
        let mut tail = Vec::with_capacity(words);
        let mut full = vec![0u8; slots * MAX_RING as usize];
        for (channel, shape) in self.shapes.iter().enumerate() {
            let ring = shape.ring();
            let cursor = cursors
                .get(channel)
                .copied()
                .unwrap_or(Cursor { head: 0, tail: 0 });
            head.extend_from_slice(&((cursor.head % ring) as u32).to_le_bytes());
            tail.extend_from_slice(&((cursor.tail % ring) as u32).to_le_bytes());
            for sequence in cursor.head..cursor.tail {
                let at = kernels_cuda::channel::Rings::full_at(
                    u32::try_from(channel).unwrap_or(u32::MAX),
                    (sequence % ring) as u32,
                ) as usize;
                if let Some(byte) = full.get_mut(at) {
                    *byte = 1;
                }
            }
        }
        self.registry.write(0, &head)?;
        self.registry.write(words as u64, &tail)?;
        self.registry.write((3 * words) as u64, &full)?;
        Ok(())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    #[must_use]
    pub fn shape(&self, channel: usize) -> Option<ChannelShape> {
        self.shapes.get(channel).copied()
    }

    pub fn cell_address(&self, channel: usize, sequence: u64) -> Result<u64> {
        let shape = self.shape_of(channel)?;
        let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
        match self.shared_slab(channel) {
            Some(base) => Ok(base + at),
            None => self.cells[channel].at(at),
        }
    }

    pub fn feed_address(&self, channel: usize, sequence: u64) -> Result<u64> {
        let shape = self.shape_of(channel)?;
        if shape.dtype == Dtype::Bool {
            return Err(Fault::program(
                "program::launch",
                format!("channel {channel} is a bool ring, which no float port reads"),
            ));
        }
        let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
        if let Some(base) = self.shared_slab(channel) {
            return Ok(base + at);
        }
        if let Some(endpoint) = self.endpoint(channel) {
            let cell = (sequence % u64::from(endpoint.cap1())) * u64::from(endpoint.wire_bytes());
            return Ok(endpoint.mirror_device() + cell);
        }
        self.cells[channel].at(at)
    }

    #[must_use]
    pub fn shared_slab(&self, channel: usize) -> Option<u64> {
        self.endpoints
            .get(channel)
            .and_then(Option::as_ref)
            .and_then(|endpoint| endpoint.device_cells())
    }

    pub fn write_cell(&mut self, channel: usize, sequence: u64, native: &[u8]) -> Result<()> {
        let shape = self.shape_of(channel)?;
        if native.len() != shape.cell_bytes() {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "channel {channel}'s native cell is {} bytes and {} were offered",
                    shape.cell_bytes(),
                    native.len()
                ),
            ));
        }
        if let Some(base) = self.shared_slab(channel) {
            let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
            crate::device::write_raw(base + at, native)?;
            if let Some(endpoint) = self.endpoint(channel)
                && !endpoint.write_cell(sequence, native)
            {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "channel {channel}'s shared shadow cell is {} bytes and {} were \
                         offered",
                        endpoint.wire_bytes(),
                        native.len()
                    ),
                ));
            }
            return Ok(());
        }
        if let Some(endpoint) = self.endpoint(channel) {
            let wire = native_to_wire(shape.dtype, shape.numel, native)?;
            if !endpoint.write_cell(sequence, &wire) {
                return Err(Fault::program(
                    "program::launch",
                    format!(
                        "channel {channel}'s mirror cell is {} bytes and {} were offered",
                        endpoint.wire_bytes(),
                        wire.len()
                    ),
                ));
            }
            return Ok(());
        }
        let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
        self.cells[channel].write(at, native)
    }

    pub fn read_cell(&self, channel: usize, sequence: u64) -> Result<Vec<u8>> {
        let shape = self.shape_of(channel)?;
        if let Some(endpoint) = self.endpoint(channel)
            && endpoint.role() == HostRole::None
        {
            return Ok(endpoint.read_cell(sequence));
        }
        if let Some(endpoint) = self.endpoint(channel) {
            return wire_to_native(shape.dtype, shape.numel, &endpoint.read_cell(sequence));
        }
        let at = (sequence % shape.ring()) * shape.cell_bytes() as u64;
        let mut out = vec![0u8; shape.cell_bytes()];
        self.cells[channel].read(at, &mut out)?;
        Ok(out)
    }

    pub(crate) fn shape_of(&self, channel: usize) -> Result<ChannelShape> {
        self.shapes.get(channel).copied().ok_or_else(|| {
            Fault::program(
                "program::launch",
                format!("channel {channel} is not one this instance carries"),
            )
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    pub head: u64,
    pub tail: u64,
}
#[derive(Debug)]
pub struct Prepared {
    retired: Vec<Buffer>,
    table: Buffer,
    table_host: Vec<u8>,
    descriptors: Buffer,
    descriptor_row: Vec<u8>,
    descriptors_host: Vec<u8>,
    params: Buffer,
    offsets: Buffer,
    scratch: Buffer,
    pending: Buffer,
    intrinsic_bases: Buffer,
    intrinsic_modes: Buffer,
    intrinsic_widths: Buffer,
    intrinsic_strides: Buffer,
    intrinsic_offsets: Buffer,
    intrinsic_bases_host: Vec<u64>,
    intrinsic_modes_host: Vec<u32>,
    intrinsic_widths_host: Vec<u32>,
    intrinsic_strides_host: Vec<u32>,
    intrinsic_offsets_host: Vec<u32>,
    channel_count: u32,
    value_count: u32,
    scratch_stride: u32,
    temporary_offset: u32,
    temporary_bytes: u32,
    region_rows: Vec<u32>,
    lanes: u32,
    lane_cap: u32,
    filled: u32,
    bindings: Vec<u32>,
}

#[must_use]
pub(crate) fn shell_launches(region: &LaunchRegion) -> bool {
    region.kind != RegionKind::Library(LibraryOp::SecondParty)
}

#[must_use]
fn read_only_by_skipped_regions(plan: &LaunchStagePlan) -> Vec<bool> {
    let values = plan.value_types.len();

    let mut producer = vec![usize::MAX; values];
    let mut result_base = 0u32;
    for (node, op) in plan.ops.iter().enumerate() {
        for result in 0..u32::from(op.result_count) {
            if let Some(slot) = producer.get_mut((result_base + result) as usize) {
                *slot = node;
            }
        }
        result_base += u32::from(op.result_count);
    }

    let mut skipped = vec![false; plan.ops.len()];
    let mut unread = vec![false; values];
    for region in plan.fused.iter().filter(|region| !shell_launches(region)) {
        for &node in &region.nodes {
            if let Some(node) = skipped.get_mut(node as usize) {
                *node = true;
            }
        }
        for &value in region.inputs.iter().chain(
            region
                .nodes
                .iter()
                .filter_map(|&node| plan.ops.get(node as usize))
                .flat_map(|op| op.args.iter()),
        ) {
            if let Some(value) = unread.get_mut(value as usize) {
                *value = true;
            }
        }
    }

    for (node, op) in plan.ops.iter().enumerate() {
        if skipped.get(node).copied().unwrap_or(false) {
            continue;
        }
        let predicate = (op.tag == tags::PIVOT_THRESHOLD).then_some(op.pred_payload);
        for value in op.args.iter().copied().chain(predicate) {
            if let Some(value) = unread.get_mut(value as usize) {
                *value = false;
            }
        }
    }
    for region in plan.fused.iter().filter(|region| shell_launches(region)) {
        for sink in &region.sinks {
            if let Some(value) = unread.get_mut(sink.value as usize) {
                *value = false;
            }
        }
    }

    for (value, drop) in unread.iter_mut().enumerate() {
        let tag = producer
            .get(value)
            .and_then(|&node| plan.ops.get(node))
            .map(|op| op.tag);
        if !matches!(tag, Some(tags::CHAN_READ | tags::CHAN_TAKE)) {
            *drop = false;
        }
    }
    unread
}

pub fn describe_values(plan: &LaunchStagePlan, extents: Extents) -> Result<Vec<ValueDesc>> {
    let empty = read_only_by_skipped_regions(plan);
    plan.value_types
        .iter()
        .enumerate()
        .map(|(value, declared)| {
            let described = describe(declared, &extents).map_err(|why| {
                Fault::program(
                    "program::launch",
                    format!("a value's shape does not resolve against this fire: {why:?}"),
                )
            })?;
            Ok(if empty.get(value).copied().unwrap_or(false) {
                ValueDesc {
                    len: 0,
                    rows: 0,
                    last: 0,
                    dims: [0; MAX_RANK],
                    ..described
                }
            } else {
                described
            })
        })
        .collect()
}

fn value_lifetimes(plan: &LaunchStagePlan, rows: &[u32]) -> Option<Vec<Lifetime>> {
    let mut step_of = vec![u32::MAX; plan.ops.len()];
    let mut region_of = vec![u32::MAX; plan.ops.len()];
    let mut region_first = Vec::with_capacity(plan.fused.len());
    let mut region_last = Vec::with_capacity(plan.fused.len());
    let mut step = 0u32;
    for (index, region) in plan.fused.iter().enumerate() {
        let per_node = region.kind == RegionKind::Generated;
        region_first.push(step);
        for &node in &region.nodes {
            *step_of.get_mut(node as usize)? = step;
            *region_of.get_mut(node as usize)? = u32::try_from(index).ok()?;
            if per_node {
                step = step.checked_add(1)?;
            }
        }
        if !per_node {
            step = step.checked_add(1)?;
        }
        region_last.push(step.saturating_sub(1));
    }
    if step_of.contains(&u32::MAX) {
        return None;
    }
    let region_of_step = |step: u32| -> u32 {
        region_first
            .partition_point(|&first| first <= step)
            .saturating_sub(1) as u32
    };
    let class_of = |region: u32, value: u32| -> u64 {
        let Some(fused) = plan.fused.get(region as usize) else {
            return Lifetime::SEQUENTIAL;
        };
        if rows.get(region as usize).copied().unwrap_or(1) <= 1 {
            return Lifetime::SEQUENTIAL;
        }
        let geometry = fused
            .row_value
            .and_then(|witness| plan.value_types.get(witness as usize))
            .and_then(|ty| eta_compiler::plan::value_rows(&ty.axes));
        let Some(geometry) = geometry else {
            return 0;
        };
        let Some(ty) = plan.value_types.get(value as usize) else {
            return 0;
        };
        let elem: u64 = if ty.dtype == Dtype::Bool { 1 } else { 4 };
        let alias = fused.row_alias;
        if ty.axes.len() >= 2
            && eta_compiler::plan::value_rows(&ty.axes)
                .is_some_and(|shape| eta_compiler::plan::same_rows(shape, geometry, alias))
        {
            let width = match ty.axes.last() {
                Some(eta_compiler::plan::Dimension::Static(width)) => u64::from(*width),
                _ => return 0,
            };
            (1u64 << 40) | (width * elem)
        } else if eta_compiler::plan::is_row_vector(&ty.axes, geometry.0, geometry.1, alias) {
            (2u64 << 40) | elem
        } else {
            0
        }
    };
    let values = plan.value_types.len();
    let mut lifetimes = vec![
        Lifetime {
            dead: false,
            def: 0,
            last: 0,
            reusable: false,
            launch_def: 0,
            launch_last: 0,
            class_def: Lifetime::SEQUENTIAL,
            class_last: Lifetime::SEQUENTIAL,
        };
        values
    ];
    let mut alias_of = vec![u32::MAX; values];
    let bump = |lifetimes: &mut Vec<Lifetime>, alias_of: &[u32], mut value: u32, step: u32| {
        for _ in 0..=alias_of.len() {
            let Some(life) = lifetimes.get_mut(value as usize) else {
                return;
            };
            life.last = life.last.max(step);
            match alias_of.get(value as usize).copied() {
                Some(source) if source != u32::MAX => value = source,
                _ => return,
            }
        }
    };
    let mut result_base = 0u32;
    for (node, op) in plan.ops.iter().enumerate() {
        let step = step_of[node];
        let written_in_full = !matches!(
            op.tag,
            tags::PIVOT_THRESHOLD | tags::CHAN_READ | tags::CHAN_TAKE
        );
        for result in 0..u32::from(op.result_count) {
            if let Some(life) = lifetimes.get_mut((result_base + result) as usize) {
                life.def = step;
                life.last = life.last.max(step);
                life.reusable = written_in_full;
                life.launch_def = region_of[node];
            }
        }
        if op.tag == tags::RESHAPE
            && let Some(&source) = op.args.first()
            && let Some(slot) = alias_of.get_mut(result_base as usize)
        {
            *slot = source;
        }
        result_base += u32::from(op.result_count);
        let predicate = (op.tag == tags::PIVOT_THRESHOLD).then_some(op.pred_payload);
        let region = region_of[node];
        let generated = plan
            .fused
            .get(region as usize)
            .is_some_and(|fused| fused.kind == RegionKind::Generated);
        for value in op.args.iter().copied().chain(predicate) {
            bump(&mut lifetimes, &alias_of, value, step);
            if generated
                && lifetimes
                    .get(value as usize)
                    .is_some_and(|life| life.launch_def == region && life.last >= life.def)
                && let Some(&last) = region_last.get(region as usize)
            {
                bump(&mut lifetimes, &alias_of, value, last);
            }
        }
    }
    for (region, fused) in plan.fused.iter().enumerate() {
        let step = region_last[region];
        let named = fused
            .inputs
            .iter()
            .chain(fused.outputs.iter())
            .copied()
            .chain(fused.sinks.iter().map(|sink| sink.value));
        for value in named {
            bump(&mut lifetimes, &alias_of, value, step);
        }
    }
    for (value, life) in lifetimes.iter_mut().enumerate() {
        let value = value as u32;
        life.launch_last = region_of_step(life.last);
        life.class_def = class_of(life.launch_def, value);
        life.class_last = class_of(life.launch_last, value);
        if life.class_last == 0 {
            life.last = life.last.max(region_last[life.launch_last as usize]);
        }
    }
    for fused in &plan.fused {
        for &value in &fused.spent {
            if let Some(life) = lifetimes.get_mut(value as usize) {
                life.dead = true;
            }
        }
    }
    Some(lifetimes)
}

fn region_rows(plan: &LaunchStagePlan, descriptors: &[ValueDesc]) -> Vec<u32> {
    plan.fused
        .iter()
        .map(|region| {
            let rows = region
                .row_value
                .and_then(|value| descriptors.get(value as usize))
                .map_or(1, |descriptor| descriptor.rows.max(1));
            match region.kind {
                RegionKind::Generated => rows,
                RegionKind::Library(LibraryOp::TopK) if selects(plan, region) => rows,
                RegionKind::Library(LibraryOp::TopK | LibraryOp::Sort) => {
                    rows.min(ORDER_ROW_BLOCKS)
                }
                RegionKind::Library(_) => 1,
            }
        })
        .collect()
}

fn selects(plan: &LaunchStagePlan, region: &LaunchRegion) -> bool {
    region
        .nodes
        .first()
        .and_then(|&node| plan.ops.get(node as usize))
        .is_some_and(|op| {
            op.tag == tags::TOP_K
                && (1..=eta_compiler::codegen::cuda::order::TOP_K_SELECT_MAX).contains(&op.imm)
        })
}

fn temporary_floor(plan: &LaunchStagePlan, descriptors: &[ValueDesc], rows: &[u32]) -> u64 {
    let align = |bytes: u64| bytes.next_multiple_of(u64::from(SCRATCH_ALIGN));
    let widest = descriptors
        .iter()
        .map(|d| u64::from(d.last))
        .max()
        .unwrap_or(1)
        .max(1);
    plan.fused
        .iter()
        .zip(rows)
        .map(|(region, &blocks)| {
            if blocks <= 1 {
                return 0;
            }
            let per_block = match region.kind {
                RegionKind::Generated => align(2 * widest.div_ceil(32) * 4 + 4096),
                RegionKind::Library(LibraryOp::TopK) if selects(plan, region) => 0,
                RegionKind::Library(LibraryOp::TopK | LibraryOp::Sort) => align(8 * widest + 256),
                RegionKind::Library(_) => 0,
            };
            per_block * u64::from(blocks)
        })
        .max()
        .unwrap_or(0)
}

fn lay_out(plan: &LaunchStagePlan, descriptors: &[ValueDesc]) -> Result<Layout> {
    let rows = region_rows(plan, descriptors);
    let lifetimes = value_lifetimes(plan, &rows);
    let floor = temporary_floor(plan, descriptors, &rows);
    match lifetimes {
        Some(lifetimes) => layout_reusing(descriptors, &lifetimes, floor),
        None => layout(descriptors),
    }
    .map_err(|why| {
        Fault::program(
            "program::launch",
            format!("this fire's scratch does not fit: {why:?}"),
        )
    })
}

pub fn scratch_bytes(plan: &LaunchStagePlan, extents: Extents) -> Result<u64> {
    let descriptors = describe_values(plan, extents)?;
    lay_out(plan, &descriptors).map(|layout| layout.total)
}

pub fn scratch_offsets(plan: &LaunchStagePlan, extents: Extents) -> Result<Vec<u64>> {
    let descriptors = describe_values(plan, extents)?;
    lay_out(plan, &descriptors).map(|layout| layout.values)
}

impl Prepared {
    pub fn build(
        plan: &LaunchStagePlan,
        shapes: &[ChannelShape],
        extents: Extents,
        lanes: u32,
        stream: *mut core::ffi::c_void,
    ) -> Result<Prepared> {
        let channel_count = u32::try_from(plan.channel_bindings.len())
            .map_err(|_| Fault::program("program::launch", "more channels than a u32 can count"))?;
        let value_count = u32::try_from(plan.value_types.len())
            .map_err(|_| Fault::program("program::launch", "more values than a u32 can count"))?;

        let descriptors = describe_values(plan, extents)?;
        let scratch_layout = lay_out(plan, &descriptors)?;
        let scratch_stride = u32::try_from(scratch_layout.total)
            .map_err(|_| Fault::program("program::launch", "a scratch stride past a u32"))?;
        let temporary_offset = u32::try_from(scratch_layout.temporary)
            .map_err(|_| Fault::program("program::launch", "a temporary offset past a u32"))?;
        let temporary_bytes = u32::try_from(scratch_layout.temporary_bytes)
            .map_err(|_| Fault::program("program::launch", "a temporary arena past a u32"))?;
        let region_rows = region_rows(plan, &descriptors);

        let mut records = Vec::with_capacity(plan.ops.len());
        let mut result_base = 0u32;
        for op in &plan.ops {
            let mut record =
                CudaOpParams::widen(OpParams::of(op, result_base, OpRuntime::default()));
            if let (true, Some(channel)) = (op.tag == tags::CHAN_PUT, op.channel) {
                let dense = plan
                    .channel_bindings
                    .get(channel as usize)
                    .copied()
                    .ok_or_else(|| {
                        Fault::program(
                            "program::launch",
                            format!(
                                "a put names stage-local channel {channel}, which the plan \
                                 does not bind"
                            ),
                        )
                    })?;
                let shape = shapes.get(dense as usize).copied().ok_or_else(|| {
                    Fault::program(
                        "program::launch",
                        format!(
                            "a put targets channel {dense}, which this instance does not carry"
                        ),
                    )
                })?;
                record.sink_bytes = u32::try_from(shape.cell_bytes()).map_err(|_| {
                    Fault::program("program::launch", "a channel cell past what a u32 counts")
                })?;
            }
            records.push(record);
            result_base += u32::from(op.result_count);
        }

        let mut params = Buffer::zeroed(
            (records.len() * size_of::<CudaOpParams>()).max(size_of::<CudaOpParams>()),
        )?;
        params.write(0, &records_bytes(&records))?;

        let offset_bytes: Vec<u8> = scratch_layout
            .values
            .iter()
            .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut offsets = Buffer::zeroed(offset_bytes.len().max(size_of::<u32>()))?;
        offsets.write(0, &offset_bytes)?;

        let mut prepared = Prepared {
            retired: Vec::new(),
            table: Buffer::zeroed(0)?,
            table_host: Vec::new(),
            descriptors: Buffer::zeroed(0)?,
            descriptor_row: descriptors.iter().flat_map(record_bytes).collect(),
            descriptors_host: Vec::new(),
            params,
            offsets,
            scratch: Buffer::zeroed(0)?,
            pending: Buffer::zeroed(0)?,
            intrinsic_bases: Buffer::zeroed(0)?,
            intrinsic_modes: Buffer::zeroed(0)?,
            intrinsic_widths: Buffer::zeroed(0)?,
            intrinsic_strides: Buffer::zeroed(0)?,
            intrinsic_offsets: Buffer::zeroed(0)?,
            intrinsic_bases_host: Vec::new(),
            intrinsic_modes_host: Vec::new(),
            intrinsic_widths_host: Vec::new(),
            intrinsic_strides_host: Vec::new(),
            intrinsic_offsets_host: Vec::new(),
            channel_count,
            value_count,
            scratch_stride,
            temporary_offset,
            temporary_bytes,
            region_rows,
            lanes: 0,
            lane_cap: u32::MAX,
            filled: 0,
            bindings: plan.channel_bindings.clone(),
        };
        prepared.grow(extents, lanes.max(1), stream)?;
        Ok(prepared)
    }

    fn grow(&mut self, extents: Extents, lanes: u32, stream: *mut core::ffi::c_void) -> Result<()> {
        if lanes <= self.lanes {
            return Ok(());
        }
        // The buffers the last growth outgrew were kept for the fires still
        // reading them; freeing waits for the device, so it happens here,
        // once per growth, rather than hoarding a generation per doubling
        // until this batch is dropped.
        self.retired.clear();
        let lanes = lanes
            .checked_next_power_of_two()
            .unwrap_or(lanes)
            .min(self.lane_ceiling())
            .max(lanes);
        let total = u64::from(self.scratch_stride) * u64::from(lanes);
        if total > eta_exec::SCRATCH_MAX_BYTES {
            return Err(Fault::Ceiling {
                what: "a wave's fused-region scratch",
                need: total,
                have: eta_exec::SCRATCH_MAX_BYTES,
            });
        }
        let shape = LaneShape::of(lanes, self.channel_count);
        let table_bytes = shape
            .bytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| Fault::program("program::launch", "a lane table past what fits"))?;
        let mut table_host = vec![0u8; table_bytes];
        write_record(
            &mut table_host,
            0,
            &LaneHeader {
                abi_version: eta_exec::LANE_ABI_VERSION,
                lane_count: lanes,
                channel_slots_per_lane: self.channel_count,
                flags: 0,
            },
        );
        for lane in 0..lanes {
            let at = shape
                .record_offset(lane)
                .and_then(|at| usize::try_from(at).ok())
                .ok_or_else(|| Fault::program("program::launch", "a lane record past the table"))?;
            write_record(
                &mut table_host,
                at,
                &LaneRecord {
                    kv_len: extents.kv_len,
                    page_count: extents.page_count,
                    row_count: extents.row_count,
                    token_count: extents.token_count,
                    sampled_rows: extents.sampled_rows,
                    query_len: extents.query_len,
                    key_len: extents.key_len,
                    channel_slot_offset: shape.slot_index(lane).ok_or_else(|| {
                        Fault::program("program::launch", "a lane's slot row past the table")
                    })?,
                    ..LaneRecord::default()
                },
            );
        }
        let table = Buffer::zeroed_on(stream, table_bytes)?;
        self.retired.push(std::mem::replace(&mut self.table, table));
        self.table_host = table_host;

        let row = self.descriptor_row.len().max(1);
        let mut descriptors = Buffer::zeroed_on(stream, row * lanes as usize)?;
        if !self.descriptor_row.is_empty() {
            self.descriptors_host = self
                .descriptor_row
                .iter()
                .copied()
                .cycle()
                .take(self.descriptor_row.len() * lanes as usize)
                .collect();
            descriptors.stage(stream, 0, &self.descriptors_host)?;
        }
        self.retired
            .push(std::mem::replace(&mut self.descriptors, descriptors));

        let scratch_bytes = (self.scratch_stride as usize * lanes as usize)
            .max(usize::try_from(SCRATCH_ALIGN).unwrap_or(256));
        // A program arrives after the load declared its pool, so this is the
        // one allocation the pool's fit could only reserve for, not count.
        // When it does not fit, the scratch this growth outgrew is freed
        // first (the free waits for the fires reading it) and the growth is
        // tried once more; past that the lanes are capped under this width
        // and the wave flies narrower, and only one lane failing is final,
        // since a frame past static admission has no retry to answer.
        let scratch = match Buffer::zeroed_on(stream, scratch_bytes) {
            Err(Fault::OutOfMemory { .. }) => {
                self.scratch = Buffer::zeroed(0)?;
                self.lanes = 0;
                Buffer::zeroed_on(stream, scratch_bytes).map_err(|fault| match fault {
                    Fault::OutOfMemory { need, have } if lanes > 1 => {
                        self.lane_cap = lanes / 2;
                        eprintln!(
                            "engine-cuda: a guest program's scratch does not fit beside this \
                             load at {lanes} lanes ({need} bytes wanted, {have} free); its \
                             fires fly at most {} lanes wide",
                            self.lane_cap
                        );
                        fault
                    }
                    Fault::OutOfMemory { need, have } => Fault::Residency(format!(
                        "a guest program's scratch does not fit beside this load: {} bytes a \
                         lane at {lanes} lanes wants {need}, and the device has {have} free once \
                         the weights, the declared kv pool and everything else the load holds \
                         are resident. Lower `[engine] max_forward_requests`, or state `[engine] \
                         max_total_pages` under the count the load derived.",
                        self.scratch_stride
                    )),
                    other => other,
                })?
            }
            other => other?,
        };
        self.retired
            .push(std::mem::replace(&mut self.scratch, scratch));
        self.retired.push(std::mem::replace(
            &mut self.pending,
            Buffer::zeroed_on(
                stream,
                (self.channel_count as usize * lanes as usize).max(1),
            )?,
        ));

        let words = INTRINSIC_SLOTS * lanes as usize;
        self.intrinsic_bases_host.resize(words, 0u64);
        self.intrinsic_modes_host.resize(words, 0u32);
        self.intrinsic_widths_host.resize(words, 0u32);
        self.intrinsic_strides_host.resize(words, 0u32);
        self.intrinsic_offsets_host.resize(words, 0u32);
        for (slot, bytes) in [
            (&mut self.intrinsic_bases, words * size_of::<u64>()),
            (&mut self.intrinsic_modes, words * size_of::<u32>()),
            (&mut self.intrinsic_widths, words * size_of::<u32>()),
            (&mut self.intrinsic_strides, words * size_of::<u32>()),
            (&mut self.intrinsic_offsets, words * size_of::<u32>()),
        ] {
            self.retired
                .push(std::mem::replace(slot, Buffer::zeroed_on(stream, bytes)?));
        }
        self.lanes = lanes;
        Ok(())
    }

    pub fn begin(
        &mut self,
        extents: Extents,
        lanes: u32,
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        self.filled = 0;
        self.grow(extents, lanes, stream)
    }

    pub fn stage_lane(&mut self, rings: &Rings, cursors: &[Cursor], commit: u64) -> Result<u32> {
        let lane = self.filled;
        if lane >= self.lanes {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "a boundary staged lane {lane} of a batch cut for {}: `begin` is what \
                     sizes it and it was told a smaller number",
                    self.lanes
                ),
            ));
        }
        let shape = LaneShape::of(self.lanes, self.channel_count);
        let record_at = shape
            .record_offset(lane)
            .and_then(|at| usize::try_from(at).ok())
            .ok_or_else(|| Fault::program("program::launch", "a lane record past the table"))?;
        let commit_at = record_at + std::mem::offset_of!(LaneRecord, commit_slot);
        self.table_host[commit_at..commit_at + size_of::<u64>()]
            .copy_from_slice(&commit.to_le_bytes());

        for (local, &dense) in self.bindings.iter().enumerate() {
            let channel = dense as usize;
            let cursor = cursors.get(channel).copied().ok_or_else(|| {
                Fault::program(
                    "program::launch",
                    format!(
                        "stage-local channel {local} binds channel {dense}, which this \
                         instance does not carry"
                    ),
                )
            })?;
            let slot_at = shape
                .slot_offset(lane, u32::try_from(local).unwrap_or(u32::MAX))
                .and_then(|at| usize::try_from(at).ok())
                .ok_or_else(|| {
                    Fault::program("program::launch", "a channel slot past the table")
                })?;
            write_record(
                &mut self.table_host,
                slot_at,
                &LaneChannelSlot {
                    committed_cell: rings.cell_address(channel, cursor.head)?,
                    pending_cell: rings.cell_address(channel, cursor.tail)?,
                    expected_head: NO_TICKET,
                    expected_tail: NO_TICKET,
                },
            );
        }
        let row = lane as usize * INTRINSIC_SLOTS;
        self.intrinsic_bases_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.intrinsic_modes_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.intrinsic_widths_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.intrinsic_strides_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.intrinsic_offsets_host[row..row + INTRINSIC_SLOTS].fill(0);
        self.filled += 1;
        Ok(lane)
    }

    pub fn commit_lanes(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.filled == 0 {
            return Ok(());
        }
        write_record(
            &mut self.table_host,
            0,
            &LaneHeader {
                abi_version: eta_exec::LANE_ABI_VERSION,
                lane_count: self.filled,
                channel_slots_per_lane: self.channel_count,
                flags: 0,
            },
        );
        self.table.stage(stream, 0, &self.table_host)?;
        self.intrinsic_bases
            .stage(stream, 0, &slice_bytes(&self.intrinsic_bases_host))?;
        self.intrinsic_modes
            .stage(stream, 0, &slice_bytes(&self.intrinsic_modes_host))?;
        self.intrinsic_widths
            .stage(stream, 0, &slice_bytes(&self.intrinsic_widths_host))?;
        self.intrinsic_strides
            .stage(stream, 0, &slice_bytes(&self.intrinsic_strides_host))?;
        self.intrinsic_offsets
            .stage(stream, 0, &slice_bytes(&self.intrinsic_offsets_host))?;
        let lanes = self.filled as usize;
        self.pending
            .zero_span_on(stream, 0, lanes * self.channel_count as usize)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_intrinsic(
        &mut self,
        lane: u32,
        intrinsic: IntrinsicId,
        base: u64,
        storage: u32,
        width: u32,
        row_stride: u32,
        row_offset: u32,
    ) -> Result<()> {
        let slot = intrinsic as usize;
        if slot >= INTRINSIC_SLOTS {
            return Err(Fault::program(
                "program::launch",
                format!(
                    "intrinsic {slot} is past the {INTRINSIC_SLOTS}-slot pitch the side \
                     tables are indexed with"
                ),
            ));
        }
        if lane >= self.lanes {
            return Err(Fault::program(
                "program::launch",
                format!("lane {lane} is past a batch cut for {}", self.lanes),
            ));
        }
        let at = lane as usize * INTRINSIC_SLOTS + slot;
        self.intrinsic_bases_host[at] = base;
        self.intrinsic_modes_host[at] = storage;
        self.intrinsic_widths_host[at] = width;
        self.intrinsic_strides_host[at] = row_stride;
        self.intrinsic_offsets_host[at] = row_offset;
        Ok(())
    }

    pub fn launch_region(&self, region: &Region, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.filled == 0 {
            return Ok(());
        }
        let mut args = Args::new();
        args.ptr(self.table.ptr())
            .ptr(self.lane_records_ptr())
            .ptr(self.channel_slots_ptr())
            .ptr(self.descriptors.ptr())
            .ptr(self.params.ptr())
            .ptr(self.offsets.ptr())
            .ptr(self.scratch.ptr())
            .u32(self.value_count)
            .u32(self.scratch_stride)
            .u32(self.temporary_offset)
            .ptr(self.pending.ptr())
            .ptr(self.intrinsic_bases.ptr())
            .ptr(self.intrinsic_modes.ptr())
            .ptr(self.intrinsic_widths.ptr())
            .ptr(self.intrinsic_strides.ptr())
            .ptr(self.intrinsic_offsets.ptr());
        let rows = self
            .region_rows
            .get(region.region_index as usize)
            .copied()
            .unwrap_or(1)
            .max(1);
        let stride = (self.temporary_bytes / rows) & !(SCRATCH_ALIGN as u32 - 1);
        args.u32(rows).u32(stride);
        launch(
            &region.module,
            self.filled.saturating_mul(rows),
            region.module.block_threads(),
            &mut args,
            FUSED_ARITY,
            stream,
        )
    }

    #[must_use]
    pub fn lane_ceiling(&self) -> u32 {
        if self.scratch_stride == 0 {
            return self.lane_cap;
        }
        u32::try_from(eta_exec::SCRATCH_MAX_BYTES / u64::from(self.scratch_stride))
            .unwrap_or(u32::MAX)
            .min(self.lane_cap)
            .max(1)
    }

    #[must_use]
    pub const fn filled(&self) -> u32 {
        self.filled
    }

    #[must_use]
    pub const fn lanes(&self) -> u32 {
        self.lanes
    }

    #[must_use]
    pub const fn channel_count(&self) -> u32 {
        self.channel_count
    }

    fn lane_records_ptr(&self) -> u64 {
        self.table.ptr() + LANE_HEADER_BYTES
    }

    fn channel_slots_ptr(&self) -> u64 {
        self.table.ptr() + LANE_HEADER_BYTES + u64::from(self.lanes) * LANE_RECORD_BYTES
    }
}

fn write_record<T: Copy>(into: &mut [u8], at: usize, record: &T) {
    // SAFETY: `T` is a `#[repr(C)]` mirror of a device struct with every field
    // written by the caller, so reading it as bytes reads only initialised
    // memory.
    let bytes = unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(record).cast::<u8>(), size_of::<T>())
    };
    into[at..at + bytes.len()].copy_from_slice(bytes);
}

pub(super) fn record_bytes<T: Copy>(record: &T) -> Vec<u8> {
    // SAFETY: as `write_record` — a fully-initialised `#[repr(C)]` mirror.
    unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(record).cast::<u8>(), size_of::<T>()).to_vec()
    }
}

pub(super) fn slice_bytes<T: Copy>(records: &[T]) -> Vec<u8> {
    // SAFETY: as `record_bytes`, over a contiguous run of them.
    unsafe {
        std::slice::from_raw_parts(
            records.as_ptr().cast::<u8>(),
            std::mem::size_of_val(records),
        )
        .to_vec()
    }
}

fn records_bytes(params: &[CudaOpParams]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(params));
    for record in params {
        bytes.extend_from_slice(&record_bytes(record));
    }
    bytes
}

#[derive(Default)]
pub struct Args {
    #[allow(clippy::vec_box)]
    storage: Vec<Box<u64>>,
    slots: Vec<*mut std::ffi::c_void>,
}

impl Args {
    #[must_use]
    pub fn new() -> Args {
        Args::default()
    }

    pub fn ptr(&mut self, pointer: u64) -> &mut Args {
        self.scalar(pointer)
    }

    pub fn u32(&mut self, value: u32) -> &mut Args {
        self.scalar(u64::from(value))
    }

    fn scalar(&mut self, value: u64) -> &mut Args {
        let mut cell = Box::new(value);
        let at: *mut u64 = &raw mut *cell;
        self.storage.push(cell);
        self.slots.push(at.cast());
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

pub fn launch(
    module: &Module,
    grid: u32,
    block: u32,
    args: &mut Args,
    expected: usize,
    stream: *mut core::ffi::c_void,
) -> Result<()> {
    if args.len() != expected {
        return Err(Fault::program(
            "cuLaunchKernel",
            format!(
                "`{}` takes {expected} arguments and {} were bound; CUDA reads the rest \
                 from whatever follows the array",
                module.entry_name(),
                args.len()
            ),
        ));
    }
    if grid == 0 {
        return Err(Fault::program(
            "cuLaunchKernel",
            format!(
                "`{}` was launched with an empty grid, which launches nothing and \
                 returns success",
                module.entry_name()
            ),
        ));
    }
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys as dr;

        // SAFETY: the function is live for this borrow, and `args` holds every
        // scalar the pointer array points at for the duration of the call.
        let code = unsafe {
            dr::cuLaunchKernel(
                module.function(),
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                stream.cast(),
                args.slots.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(Fault::Device {
                call: "cuLaunchKernel",
                code: code as i32,
            });
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (block, stream);
        Err(Fault::Runtimeless)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_bool_is_packed_and_the_round_trip_is_lossless() {
        for (dtype, numel) in [
            (Dtype::F32, 3),
            (Dtype::I32, 5),
            (Dtype::U32, 1),
            (Dtype::Bool, 11),
        ] {
            let native = native_cell_bytes(dtype, numel);
            let wire = eta_exec::wire_cell_bytes(dtype, numel);
            if dtype == Dtype::Bool {
                assert_eq!(native, numel, "a byte per lane on the device");
                assert_eq!(wire, numel.div_ceil(8), "a bit per lane on the wire");
            } else {
                assert_eq!(native, wire, "four bytes either way");
            }
            let bytes: Vec<u8> = (0..wire).map(|i| (i as u8).wrapping_mul(37) | 1).collect();
            let there = wire_to_native(dtype, numel, &bytes).expect("one wire cell");
            assert_eq!(there.len(), native);
            let back = native_to_wire(dtype, numel, &there).expect("one native cell");
            if dtype == Dtype::Bool {
                for lane in 0..numel {
                    assert_eq!(
                        back[lane / 8] >> (lane % 8) & 1,
                        bytes[lane / 8] >> (lane % 8) & 1,
                        "lane {lane} did not survive the round trip"
                    );
                }
            } else {
                assert_eq!(back, bytes);
            }
        }
    }
}
