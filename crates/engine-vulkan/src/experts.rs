use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;

use checkpoint::plan::{LoadPlan, StorageInstr};
use engine::load::{Residency, Tiers};
use model_ir::{Def, Linear, Operation, Trace, ValueId};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use kernels_vulkan::Tensor;

const RING: usize = 2;

const MAX_ROUTES: u64 = 1 << 18;

const COPY_THREADS: usize = 16;

pub struct Mapping {
    at: NonNull<u8>,
    len: usize,
}

unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl std::fmt::Debug for Mapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapping").field("len", &self.len).finish()
    }
}

impl Mapping {
    pub fn open(path: &std::path::Path) -> Result<Mapping> {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path).map_err(|e| Fault::Device {
            call: "open",
            why: format!("{}: {e}", path.display()),
        })?;
        let len = usize::try_from(
            file.metadata()
                .map_err(|e| Fault::Device {
                    call: "fstat",
                    why: e.to_string(),
                })?
                .len(),
        )
        .map_err(|_| Fault::Device {
            call: "mmap",
            why: "the artifact is longer than the address space".into(),
        })?;
        if len == 0 {
            return Err(Fault::Device {
                call: "mmap",
                why: format!("{} is empty", path.display()),
            });
        }

        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if at == libc::MAP_FAILED {
            return Err(Fault::Device {
                call: "mmap",
                why: std::io::Error::last_os_error().to_string(),
            });
        }
        Ok(Mapping {
            at: NonNull::new(at.cast::<u8>()).expect("mmap returned a non-null address"),
            len,
        })
    }

    #[must_use]
    pub fn bytes(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let offset = usize::try_from(offset).ok()?;
        let len = usize::try_from(len).ok()?;
        let end = offset.checked_add(len)?;
        (end <= self.len)
            .then(|| unsafe { std::slice::from_raw_parts(self.at.as_ptr().add(offset), len) })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.at.as_ptr().cast(), self.len) };
    }
}

#[derive(Debug, Clone)]
pub struct HostPlane {
    runs: Vec<(u64, u64, u64)>,

    pub bytes: u64,

    pub per: u64,

    pub rows: u32,
}

impl HostPlane {
    #[must_use]
    pub fn new(runs: Vec<(u64, u64, u64)>, bytes: u64, per: u64, rows: u32) -> HostPlane {
        HostPlane {
            runs,
            bytes,
            per,
            rows,
        }
    }

    fn copy_row(&self, map: &Mapping, row: u64, into: *mut u8) -> Result<()> {
        let start = row * self.per;
        let end = start + self.per;
        let mut at = start;
        for &(virt, file, len) in &self.runs {
            let run_end = virt + len;
            if run_end <= at {
                continue;
            }
            if virt > at {
                break;
            }
            let take = (run_end.min(end)) - at;
            let src = map.bytes(file + (at - virt), take).ok_or(Fault::Ceiling {
                what: "bytes of the mapped artifact",
                need: file + (at - virt) + take,
                have: map.len() as u64,
            })?;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    into.add((at - start) as usize),
                    take as usize,
                );
            }
            at += take;
            if at >= end {
                break;
            }
        }
        if at < end {
            return Err(Fault::Ceiling {
                what: "bytes of a host-tier plane's runs",
                need: end,
                have: at,
            });
        }
        Ok(())
    }

    pub fn agrees(&self, map: &Mapping, bytes: &[u8]) -> bool {
        if bytes.len() as u64 != self.bytes {
            return false;
        }
        let page = 4096u64;
        let mut probes: Vec<u64> = (0..self.bytes).step_by(1 << 20).collect();
        probes.push(self.bytes.saturating_sub(page));
        for start in probes {
            let len = page.min(self.bytes - start);
            let mut got = vec![0u8; len as usize];
            let mut at = start;
            let end = start + len;
            for &(virt, file, run_len) in &self.runs {
                let run_end = virt + run_len;
                if run_end <= at {
                    continue;
                }
                if virt > at {
                    return false;
                }
                let take = run_end.min(end) - at;
                let Some(src) = map.bytes(file + (at - virt), take) else {
                    return false;
                };
                got[(at - start) as usize..(at - start + take) as usize].copy_from_slice(src);
                at += take;
                if at >= end {
                    break;
                }
            }
            if at < end || got != bytes[start as usize..end as usize] {
                return false;
            }
        }
        true
    }

    #[must_use]
    pub fn runs(&self) -> &[(u64, u64, u64)] {
        &self.runs
    }
}

pub fn identity_runs(
    plan: &LoadPlan,
    name: &str,
    bytes: u64,
) -> std::result::Result<Vec<(u64, u64, u64)>, String> {
    let mut finalized = None;
    let mut views = std::collections::HashMap::new();
    for instr in &plan.instrs {
        match instr {
            StorageInstr::Finalize {
                tensor, name: n, ..
            } if n == name => {
                finalized = Some(*tensor);
            }
            StorageInstr::CreateView {
                input,
                output,
                view,
                ..
            } => {
                views.insert(*output, (*input, view.offset + view.stride.base_offset));
            }
            _ => {}
        }
    }
    let mut root = finalized.ok_or_else(|| format!("the plan never finalizes `{name}`"))?;
    let mut base = 0u64;
    while let Some(&(input, offset)) = views.get(&root) {
        base += offset;
        root = input;
    }
    let arena = plan
        .buffers
        .iter()
        .find(|b| b.id == root)
        .and_then(|b| b.arena_offset());
    let mut runs: Vec<(u64, u64, u64)> = Vec::new();
    for instr in &plan.instrs {
        let (source, virt) = match instr {
            StorageInstr::ExtentWrite { source, dest, .. } if dest.buffer == root => {
                if !dest.stride.is_dense() {
                    return Err(format!("`{name}` is written through a strided extent"));
                }
                (source, dest.offset + dest.stride.base_offset)
            }
            StorageInstr::BulkExtentWrite {
                source,
                dest_offset,
                ..
            } => match arena {
                Some(at) if *dest_offset >= at && *dest_offset < at + bytes + base => {
                    (source, *dest_offset - at)
                }
                _ => continue,
            },
            StorageInstr::GatherWrite { dest, .. } if dest.buffer == root => {
                return Err(format!("`{name}` is gathered, not copied through"));
            }
            StorageInstr::TileMap { outputs, .. } if outputs.contains(&root) => {
                return Err(format!("`{name}` is transformed, not copied through"));
            }
            _ => continue,
        };
        if !source.stride.is_dense() {
            return Err(format!("`{name}` is read through a strided extent"));
        }
        let len = source
            .stride
            .dims
            .iter()
            .try_fold(u64::from(source.stride.element_bytes), |n, d| {
                u64::try_from(d.count).ok().and_then(|c| n.checked_mul(c))
            })
            .ok_or_else(|| format!("`{name}` has an extent that does not size"))?;
        if source.file_id.0 != 0 && plan.files.len() > 1 {
            return Err(format!("`{name}` is read from a second file"));
        }
        let file = source.file_offset + source.stride.base_offset;

        if virt + len <= base || virt >= base + bytes {
            continue;
        }
        let skip = base.saturating_sub(virt);
        let start = virt + skip - base;
        let take = (virt + len).min(base + bytes) - (virt + skip);
        runs.push((start, file + skip, take));
    }
    runs.sort_unstable();
    let mut end = 0u64;
    for &(virt, _, len) in &runs {
        if virt != end {
            return Err(format!(
                "`{name}` is not covered without gaps: a run starts at {virt}, the last ended at {end}"
            ));
        }
        end = virt + len;
    }
    if end != bytes {
        return Err(format!("`{name}` covers {end} bytes of {bytes}"));
    }
    Ok(runs)
}

#[derive(Debug)]
pub struct HostTier {
    map: Mapping,
    planes: BTreeMap<u32, HostPlane>,

    _phantom: Buffer,
    bytes: u64,
}

impl HostTier {
    pub fn new(
        device: &Context,
        handles: &Handles,
        map: Mapping,
        planes: Vec<HostPlane>,
    ) -> Result<(HostTier, Vec<u32>)> {
        const STRIDE: u64 = 256;
        let phantom = Buffer::host(device, STRIDE * planes.len().max(1) as u64)?;
        let mut minted = Vec::with_capacity(planes.len());
        let mut table = BTreeMap::new();
        let mut bytes = 0u64;
        for (at, plane) in planes.into_iter().enumerate() {
            let handle = handles.bind(&phantom, at as u64 * STRIDE, STRIDE)?;
            bytes += plane.bytes;
            table.insert(handle, plane);
            minted.push(handle);
        }
        Ok((
            HostTier {
                map,
                planes: table,
                _phantom: phantom,
                bytes,
            },
            minted,
        ))
    }

    #[must_use]
    pub fn plane(&self, handle: u32) -> Option<&HostPlane> {
        self.planes.get(&handle)
    }

    #[must_use]
    pub fn map(&self) -> &Mapping {
        &self.map
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Codes,
    Scales,
    Biases,
    Dense,
}

impl Kind {
    const ALL: [Kind; 4] = [Kind::Codes, Kind::Scales, Kind::Biases, Kind::Dense];

    const fn at(self) -> usize {
        match self {
            Kind::Codes => 0,
            Kind::Scales => 1,
            Kind::Biases => 2,
            Kind::Dense => 3,
        }
    }
}

#[derive(Debug)]
pub struct Seats {
    slabs: Vec<(Kind, Buffer)>,

    seat: [Option<(u32, u64)>; 4],
    capacity: u32,
    routes: Buffer,
    routes_handle: u32,

    staging: Vec<Buffer>,
    kind_off: [u64; 4],
    ring: Cell<usize>,
    pending: Cell<usize>,
}

impl Seats {
    pub fn reserve(
        device: &Context,
        handles: &Handles,
        widest: [u64; 4],
        capacity: u32,
    ) -> Result<Option<Seats>> {
        if widest.iter().all(|&b| b == 0) || capacity == 0 {
            return Ok(None);
        }
        let mut slabs = Vec::new();
        let mut seat = [None; 4];
        let mut kind_off = [0u64; 4];
        let mut slot_bytes = 0u64;
        for kind in Kind::ALL {
            let per = widest[kind.at()];
            if per == 0 {
                continue;
            }
            let cap = per * u64::from(capacity);
            let buffer = Buffer::zeroed(device, cap)?;
            let handle = handles.bind(&buffer, 0, cap)?;
            slabs.push((kind, buffer));
            seat[kind.at()] = Some((handle, per));
            kind_off[kind.at()] = slot_bytes;
            slot_bytes += cap;
        }
        let mut staging = Vec::with_capacity(RING);
        for _ in 0..RING {
            staging.push(Buffer::with(
                device,
                slot_bytes,
                crate::device::alloc::Memory::Staging,
            )?);
        }
        let routes = Buffer::zeroed(device, MAX_ROUTES * 4)?;
        let routes_handle = handles.bind(&routes, 0, MAX_ROUTES * 4)?;
        Ok(Some(Seats {
            slabs,
            seat,
            capacity,
            routes,
            routes_handle,
            staging,
            kind_off,
            ring: Cell::new(0),
            pending: Cell::new(0),
        }))
    }

    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        self.slabs.iter().map(|(_, b)| b.bytes()).sum::<u64>() + self.routes.bytes()
    }

    #[must_use]
    pub fn routes(&self, routes: Tensor) -> Tensor {
        Tensor::new(
            self.routes_handle,
            routes.rows,
            routes.width,
            model_ir::Dtype::I32,
        )
    }

    pub fn write_routes(&self, seat_of: &[i32]) -> Result<()> {
        if seat_of.len() as u64 > MAX_ROUTES {
            return Err(Fault::Ceiling {
                what: "route entries in one gather",
                need: seat_of.len() as u64,
                have: MAX_ROUTES,
            });
        }
        let mut bytes = Vec::with_capacity(seat_of.len() * 4);
        for s in seat_of {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        self.routes.write_shared(0, &bytes)
    }

    pub fn gather(
        &self,
        frame: &Frame,
        map: &Mapping,
        planes: &[(Kind, Tensor, &HostPlane)],
        unique: &[i32],
    ) -> Result<Vec<Tensor>> {
        if unique.len() as u32 > self.capacity {
            return Err(Fault::Ceiling {
                what: "distinct experts one gather seats",
                need: unique.len() as u64,
                have: u64::from(self.capacity),
            });
        }
        if self.pending.get() >= RING {
            frame.flush()?;
            self.pending.set(0);
        }
        let slot = self.ring.get();
        self.ring.set((slot + 1) % RING);
        self.pending.set(self.pending.get() + 1);
        let staging = &self.staging[slot];
        let base = staging.mapped_ptr().ok_or(Fault::Device {
            call: "gather",
            why: "the staging slot is not host-mapped".into(),
        })?;
        let mut out = Vec::with_capacity(planes.len());
        for &(kind, like, plane) in planes {
            let (handle, per) = self.seat[kind.at()].ok_or(Fault::Device {
                call: "gather",
                why: format!("no seat slab was reserved for {kind:?} planes"),
            })?;
            if plane.per > per {
                return Err(Fault::Ceiling {
                    what: "bytes of one expert row against the seat slab",
                    need: plane.per,
                    have: per,
                });
            }
            let region = base + self.kind_off[kind.at()] as usize;
            copy_rows(map, plane, unique, region)?;
            let len = plane.per * unique.len() as u64;
            let (_, slab) = self
                .slabs
                .iter()
                .find(|(k, _)| *k == kind)
                .expect("a seated kind has a slab");
            record_copy(frame, staging, self.kind_off[kind.at()], slab, 0, len)?;
            out.push(Tensor::new(handle, self.capacity, like.width, like.dtype));
        }
        Ok(out)
    }
}

#[derive(Debug)]
pub struct Pump {
    slots: Vec<(Buffer, u32)>,

    per: u64,

    staging: Vec<Buffer>,
    ring: Cell<usize>,
    pending: Cell<usize>,

    seated: std::cell::RefCell<BTreeMap<u32, u32>>,
}

const PUMP_MIN: usize = 3;

const PUMP_MAX: usize = 64;

impl Pump {
    pub fn reserve(
        device: &Context,
        handles: &Handles,
        widest: u64,
        ring: u64,
    ) -> Result<Option<Pump>> {
        if widest == 0 {
            return Ok(None);
        }
        let count = usize::try_from(ring / widest.max(1))
            .unwrap_or(PUMP_MIN)
            .clamp(PUMP_MIN, PUMP_MAX);
        let mut slots = Vec::with_capacity(count);
        let mut staging = Vec::with_capacity(count);
        for _ in 0..count {
            let buffer = Buffer::zeroed(device, widest)?;
            let handle = handles.bind(&buffer, 0, widest)?;
            slots.push((buffer, handle));
            staging.push(Buffer::with(
                device,
                widest,
                crate::device::alloc::Memory::Staging,
            )?);
        }
        Ok(Some(Pump {
            slots,
            per: widest,
            staging,
            ring: Cell::new(0),
            pending: Cell::new(0),
            seated: std::cell::RefCell::new(BTreeMap::new()),
        }))
    }

    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        self.slots.iter().map(|(b, _)| b.bytes()).sum()
    }

    pub fn stage(
        &self,
        frame: &Frame,
        map: &Mapping,
        plane: &HostPlane,
        like: Tensor,
    ) -> Result<Tensor> {
        if let Some(&handle) = self.seated.borrow().get(&like.buf) {
            return Ok(Tensor::new(handle, like.rows, like.width, like.dtype));
        }
        if plane.bytes > self.per {
            return Err(Fault::Ceiling {
                what: "bytes of one spilled dense plane against the pump slot",
                need: plane.bytes,
                have: self.per,
            });
        }
        if self.pending.get() >= self.slots.len() {
            frame.flush()?;
            self.pending.set(0);
            self.seated.borrow_mut().clear();
        }
        let at = self.ring.get();
        self.ring.set((at + 1) % self.slots.len());
        self.pending.set(self.pending.get() + 1);
        let staging = &self.staging[at];
        let base = staging.mapped_ptr().ok_or(Fault::Device {
            call: "pump",
            why: "the staging slot is not host-mapped".into(),
        })?;
        copy_plane(map, plane, base)?;
        let (slab, handle) = &self.slots[at];
        record_copy(frame, staging, 0, slab, 0, plane.bytes)?;
        self.seated.borrow_mut().insert(like.buf, *handle);
        Ok(Tensor::new(*handle, like.rows, like.width, like.dtype))
    }
}

fn copy_plane(map: &Mapping, plane: &HostPlane, into: usize) -> Result<()> {
    for &(virt, file, len) in plane.runs() {
        let src = map.bytes(file, len).ok_or(Fault::Ceiling {
            what: "bytes of the mapped artifact",
            need: file + len,
            have: map.len() as u64,
        })?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                (into + virt as usize) as *mut u8,
                len as usize,
            );
        }
    }
    Ok(())
}

fn copy_rows(map: &Mapping, plane: &HostPlane, unique: &[i32], into: usize) -> Result<()> {
    if unique.is_empty() {
        return Ok(());
    }
    let per = plane.per as usize;
    let threads = COPY_THREADS.min(unique.len());
    let chunk = unique.len().div_ceil(threads);
    let faults: Vec<Result<()>> = std::thread::scope(|scope| {
        unique
            .chunks(chunk)
            .enumerate()
            .map(|(at, rows)| {
                scope.spawn(move || {
                    for (i, &row) in rows.iter().enumerate() {
                        let seat = at * chunk + i;
                        if row < 0 || row as u32 >= plane.rows {
                            return Err(Fault::Ceiling {
                                what: "expert ids a route names",
                                need: u64::try_from(row).unwrap_or(u64::MAX),
                                have: u64::from(plane.rows),
                            });
                        }

                        plane.copy_row(map, row as u64, (into + seat * per) as *mut u8)?;
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().expect("a copy worker does not panic"))
            .collect()
    });
    faults.into_iter().collect()
}

#[cfg(feature = "vulkan")]
fn record_copy(
    frame: &Frame,
    source: &Buffer,
    source_at: u64,
    into: &Buffer,
    into_at: u64,
    len: u64,
) -> Result<()> {
    use ash::vk;
    if len == 0 {
        return Ok(());
    }
    source.span(source_at, len)?;
    into.span(into_at, len)?;
    let (source, into) = (source.slab(), into.slab());
    let d = &frame.core().device;
    let cmd = frame.cmd();

    unsafe {
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[before],
            &[],
            &[],
        );
        d.cmd_copy_buffer(
            cmd,
            source.buffer,
            into.buffer,
            &[vk::BufferCopy::default()
                .src_offset(source_at)
                .dst_offset(into_at)
                .size(len)],
        );
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[after],
            &[],
            &[],
        );
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn record_copy(
    _frame: &Frame,
    _source: &Buffer,
    _source_at: u64,
    _into: &Buffer,
    _into_at: u64,
    _len: u64,
) -> Result<()> {
    Err(Fault::Deviceless)
}

#[must_use]
pub fn row_bytes(dtype: model_ir::Dtype, width: u32) -> u64 {
    let width = u64::from(width);
    match dtype {
        model_ir::Dtype::Mxfp4 | model_ir::Dtype::U8g64 => width,
        model_ir::Dtype::U4g64 | model_ir::Dtype::U4g32 | model_ir::Dtype::U4g64tiled => {
            width.div_ceil(2)
        }
        model_ir::Dtype::U2g32 | model_ir::Dtype::U2g64 | model_ir::Dtype::U2g128 => {
            width.div_ceil(4)
        }
        other => width.saturating_mul(model_compiler::arena::elem_bytes(other).unwrap_or(0)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Device,

    Host,
}

#[derive(Debug, Default, Clone)]
pub struct Plan {
    routed: Vec<usize>,

    router_of: BTreeMap<usize, u32>,

    gathered: BTreeMap<usize, u32>,

    rows_per_token: u32,
}

impl Plan {
    pub fn of(trace: &Trace) -> Result<Plan> {
        let mut routed = BTreeSet::new();
        let mut router_of = BTreeMap::new();
        let mut gathered = BTreeMap::new();
        let mut rows_per_token = 0u32;
        for node in &trace.nodes {
            if let Operation::Layout(model_ir::Layout::EmbedConcat { ids, table, .. }) = &node.op
                && let Ok(at) = weight_of(trace, *table)
            {
                gathered.insert(at, ids.0);
            }

            if let Operation::Attention(
                model_ir::Attention::PleNgramIds {
                    heads_per_ngram, ..
                }
                | model_ir::Attention::PleNgramIdsChunked {
                    heads_per_ngram, ..
                },
            ) = &node.op
            {
                rows_per_token = rows_per_token.max(*heads_per_ngram);
            }
            let Operation::Linear(op) = &node.op else {
                continue;
            };
            let (routes, indexed): (ValueId, Vec<ValueId>) = match op {
                Linear::MoeMatmulSelect { bank, routes, .. }
                | Linear::MoeMatmulSelectQuant { bank, routes, .. } => (*routes, vec![*bank]),
                Linear::MoeMatmulSelectBias {
                    bank, bias, routes, ..
                } => (*routes, vec![*bank, *bias]),
                Linear::MoeBiasSum { bias, routes, .. } => (*routes, vec![*bias]),
                _ => continue,
            };
            for id in indexed {
                let at = weight_of(trace, id)?;
                routed.insert(at);
                router_of.insert(at, routes.0);
            }
        }
        Ok(Plan {
            routed: routed.into_iter().collect(),
            router_of,
            gathered,
            rows_per_token,
        })
    }

    #[must_use]
    pub fn gathered(&self) -> Vec<usize> {
        self.gathered.keys().copied().collect()
    }

    #[must_use]
    pub fn is_gathered(&self, param: usize) -> bool {
        self.gathered.contains_key(&param)
    }

    #[must_use]
    pub fn gathered_demand(&self, max_tokens: u32) -> u32 {
        if self.gathered.is_empty() {
            return 0;
        }
        max_tokens.saturating_mul(self.rows_per_token.max(1))
    }

    #[must_use]
    pub fn is_routed(&self, param: usize) -> bool {
        self.router_of.contains_key(&param)
    }

    #[must_use]
    pub fn routed(&self) -> &[usize] {
        &self.routed
    }

    pub fn tiers(
        &self,
        bytes: &[u64],
        planes: &BTreeMap<usize, Vec<usize>>,
        pinned: &[bool],
        device_cap: u64,
        residency: Residency,
    ) -> Result<Layout> {
        let mut tier = vec![Tier::Device; bytes.len()];

        let mut by_router: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        let mut grouped = vec![false; bytes.len()];
        for (&at, &router) in self.router_of.iter().chain(self.gathered.iter()) {
            let members = by_router.entry(router).or_default();
            members.push(at);
            members.extend(planes.get(&at).into_iter().flatten().copied());
        }
        let groups: Vec<Vec<usize>> = by_router.into_values().collect();
        for members in &groups {
            for &m in members {
                grouped[m] = true;
            }
        }
        let dense: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !grouped[*at])
            .map(|(_, b)| *b)
            .sum();
        let cap = match residency.device_weight_budget {
            Some(budget) => budget.min(device_cap),
            None => device_cap,
        };

        let mut dense = dense;
        let mut spilled = 0u64;
        let mut pump = 0u64;
        let mut slot = 0u64;
        if dense > cap {
            let ring = cap / 4;

            let mut order: Vec<usize> = (0..bytes.len())
                .filter(|at| !grouped[*at] && !pinned[*at])
                .collect();
            order.sort_by_key(|&at| bytes[at]);
            let mut room = cap;
            for at in order {
                if bytes[at] > ring {
                    continue;
                }
                slot = slot.max(bytes[at]);
                pump = slot * PUMP_MIN as u64;
                room = cap.saturating_sub(ring.max(pump));
                if dense <= room {
                    break;
                }
                tier[at] = Tier::Host;
                dense -= bytes[at];
                spilled += bytes[at];
            }
            if dense > room {
                return Err(Fault::Ceiling {
                    what: "bytes of device-local memory for the dense weight planes that \
                           must stay resident — every plane a rotating slot can hold is \
                           already pumped from the artifact, and the rest are each wider \
                           than one slot",
                    need: dense + pump,
                    have: cap,
                });
            }
            if spilled == 0 {
                pump = 0;
                slot = 0;
            } else {
                pump = ring.max(slot * PUMP_MIN as u64);
            }
        }
        let mut device = dense + pump;
        let mut host = 0u64;
        for members in &groups {
            let group: u64 = members.iter().map(|&m| bytes[m]).sum();
            if device + group <= cap {
                device += group;
            } else {
                for &m in members {
                    tier[m] = Tier::Host;
                }
                host += group;
            }
        }
        let tiers = Tiers {
            device,
            host,

            spilled,
            sourced: spilled > 0,
        };
        residency
            .admit_tiers(tiers)
            .map_err(|why| Fault::Residency(why.to_string()))?;
        Ok(Layout {
            tier,
            tiers,
            slot,
            ring: pump,
        })
    }
}

pub struct Gathered {
    seats: Seats,

    key: Cell<Option<(u32, u32, u32)>>,

    unique: RefCell<Vec<i32>>,
}

impl std::fmt::Debug for Gathered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gathered")
            .field("capacity", &self.seats.capacity())
            .finish()
    }
}

impl Gathered {
    pub fn reserve(
        device: &Context,
        handles: &Handles,
        widest: [u64; 4],
        capacity: u32,
    ) -> Result<Option<Gathered>> {
        Ok(
            Seats::reserve(device, handles, widest, capacity)?.map(|seats| Gathered {
                seats,
                key: Cell::new(None),
                unique: RefCell::new(Vec::new()),
            }),
        )
    }

    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        self.seats.device_bytes()
    }

    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.seats.capacity()
    }

    pub fn stage(
        &self,
        handles: &Handles,
        host: &HostTier,
        bank: kernels_vulkan::Bank,
        ids: Tensor,
    ) -> Result<Option<(kernels_vulkan::Bank, Tensor)>> {
        if host.plane(bank.codes.buf).is_none() {
            return Ok(None);
        }
        let key = (ids.buf, ids.rows, ids.width);
        if self.key.get() != Some(key) {
            let n = u64::from(ids.rows) * u64::from(ids.width);
            crate::probe::with_frame(Frame::flush)
                .ok_or_else(|| Fault::Residency("no frame is recording on this thread".into()))??;
            let raw = handles.read(ids.buf, n * 4)?;
            let mut unique: Vec<i32> = Vec::new();
            let mut seat_of: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
            let mut seated = Vec::with_capacity(raw.len() / 4);
            for word in raw.chunks_exact(4) {
                let row = i32::from_le_bytes([word[0], word[1], word[2], word[3]]);

                if row < 0 {
                    seated.push(-1);
                    continue;
                }
                let seat = *seat_of.entry(row).or_insert_with(|| {
                    unique.push(row);
                    unique.len() as i32 - 1
                });
                seated.push(seat);
            }
            self.seats.write_routes(&seated)?;
            self.key.set(Some(key));
            *self.unique.borrow_mut() = unique;
        }
        let unique = self.unique.borrow();
        let codes = host.plane(bank.codes.buf).expect("checked above");
        let scales = host.plane(bank.scales.buf).ok_or_else(|| {
            Fault::Residency("a gathered bank's scales are not on the host tier".into())
        })?;
        let mut wanted: Vec<(Kind, Tensor, &HostPlane)> = vec![
            (Kind::Codes, bank.codes, codes),
            (Kind::Scales, bank.scales, scales),
        ];
        if let Some(b) = bank.biases {
            let biases = host.plane(b.buf).ok_or_else(|| {
                Fault::Residency("a gathered bank's zero points are not on the host tier".into())
            })?;
            wanted.push((Kind::Biases, b, biases));
        }
        let seated = crate::probe::with_frame(|frame| {
            self.seats.gather(frame, host.map(), &wanted, &unique)
        })
        .ok_or_else(|| Fault::Residency("no frame is recording on this thread".into()))??;
        let mut out = bank;
        let mut seated = seated.into_iter();
        out.codes = seated.next().expect("the codes seat");
        out.scales = seated.next().expect("the scales seat");
        if bank.biases.is_some() {
            out.biases = seated.next();
        }
        Ok(Some((out, self.seats.routes(ids))))
    }
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub tier: Vec<Tier>,
    pub tiers: Tiers,

    pub slot: u64,

    pub ring: u64,
}

impl Layout {
    #[must_use]
    pub fn resident(bytes: &[u64]) -> Layout {
        Layout {
            tier: vec![Tier::Device; bytes.len()],
            tiers: Tiers {
                device: bytes.iter().sum(),
                host: 0,
                spilled: 0,
                sourced: false,
            },
            slot: 0,
            ring: 0,
        }
    }

    #[must_use]
    pub fn streams(&self) -> bool {
        self.tiers.host > 0 || self.tiers.spilled > 0
    }
}

fn weight_of(trace: &Trace, id: ValueId) -> Result<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Ok(*w as usize),
        _ => Err(Fault::Param {
            name: format!("value {}", id.0),
            why: "is read at a routed matmul's expert-indexed port and is not a weight",
        }),
    }
}
