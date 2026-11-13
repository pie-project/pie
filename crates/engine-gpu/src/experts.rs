#![allow(unsafe_code)]
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use checkpoint::plan::{LoadPlan, StorageInstr};
use engine::load::Tiers;
use poem_ir::{Def, Operation, Trace, ValueId};

use crate::backend::{Backend, Buffer as _, Frame as _};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use kernels_slang::Tensor;

const RING: usize = 2;

const MAX_ROUTES: u64 = 1 << 18;

const COPY_THREADS: usize = 16;

#[cfg(not(target_arch = "wasm32"))]
pub struct Mapping {
    map: memmap2::Mmap,
}

#[cfg(target_arch = "wasm32")]
pub struct Mapping {
    never: std::convert::Infallible,
}

impl std::fmt::Debug for Mapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapping").field("len", &self.len()).finish()
    }
}

#[cfg(target_arch = "wasm32")]
impl Mapping {
    pub fn open(path: &std::path::Path) -> Result<Mapping> {
        Err(Fault::Device {
            call: "mmap",
            why: format!("{}: a browser host maps no file", path.display()),
        })
    }

    #[must_use]
    pub fn bytes(&self, _offset: u64, _len: u64) -> Option<&[u8]> {
        match self.never {}
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self.never {}
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self.never {}
    }

    pub fn release(&self) {
        match self.never {}
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Mapping {
    pub fn open(path: &std::path::Path) -> Result<Mapping> {
        let file = std::fs::File::open(path).map_err(|e| Fault::Device {
            call: "open",
            why: format!("{}: {e}", path.display()),
        })?;
        let len = file
            .metadata()
            .map_err(|e| Fault::Device {
                call: "fstat",
                why: e.to_string(),
            })?
            .len();
        if len == 0 {
            return Err(Fault::Device {
                call: "mmap",
                why: format!("{} is empty", path.display()),
            });
        }
        // SAFETY: the artifact is opened read-only and must not be written by
        // another process for as long as this mapping lives.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Fault::Device {
            call: "mmap",
            why: e.to_string(),
        })?;
        Ok(Mapping { map })
    }

    #[must_use]
    pub fn bytes(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let offset = usize::try_from(offset).ok()?;
        let len = usize::try_from(len).ok()?;
        let end = offset.checked_add(len)?;
        (end <= self.map.len()).then(|| &self.map[offset..end])
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Lets go of the mapping's resident pages: what was copied out of the
    /// artifact (into pool slots) is read again only to fill a slot or stage
    /// a miss, from the page cache, so a load whose weights lie in the pool
    /// keeps no host copy of them resident.
    pub fn release(&self) {
        // SAFETY: a read-only private file mapping; dropping its pages makes
        // the next read fault them back in from the file, unchanged.
        let _ = unsafe {
            self.map
                .unchecked_advise(memmap2::UncheckedAdvice::DontNeed)
        };
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
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

    /// Row `row` of this plane, `per` bytes, into `out`.
    pub fn row_into(&self, map: &Mapping, row: u64, out: &mut [u8]) -> Result<()> {
        if (out.len() as u64) < self.per {
            return Err(Fault::Ceiling {
                what: "bytes a host-tier row lands in",
                need: self.per,
                have: out.len() as u64,
            });
        }
        self.copy_row(map, row, out.as_mut_ptr())
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

            // SAFETY: `into` is this row's `per` bytes of a live staging view
            // (`copy_row_at` seats row `seat` at `seat * per` of a stage sized
            // `per * unique.len()`), and `at - start + take <= end - start = per`.
            // `src` is `take` bytes of the mapping, which no view aliases.
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
pub struct HostTier<B: Backend> {
    map: Mapping,
    planes: BTreeMap<u32, HostPlane>,

    _phantom: B::Buffer,
    bytes: u64,
}

impl<B: Backend> HostTier<B> {
    pub fn new(
        device: &B::Context,
        handles: &Handles<B>,
        map: Mapping,
        planes: Vec<HostPlane>,
    ) -> Result<(HostTier<B>, Vec<u32>)> {
        const STRIDE: u64 = 256;
        let phantom = B::Buffer::host(
            device,
            STRIDE * planes.len().max(1) as u64,
            crate::books::owners::EXPERT_TIER,
        )?;
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
}

impl Kind {
    const ALL: [Kind; 3] = [Kind::Codes, Kind::Scales, Kind::Biases];

    const fn at(self) -> usize {
        match self {
            Kind::Codes => 0,
            Kind::Scales => 1,
            Kind::Biases => 2,
        }
    }
}

#[derive(Debug)]
pub struct Seats<B: Backend> {
    slabs: Vec<(Kind, B::Buffer)>,

    seat: [Option<(u32, u64)>; 3],
    capacity: u32,
    routes: B::Buffer,
    routes_handle: u32,

    staging: Vec<B::Buffer>,
    kind_off: [u64; 3],
    ring: Cell<usize>,
    pending: Cell<usize>,
}

impl<B: Backend> Seats<B> {
    pub fn reserve(
        device: &B::Context,
        handles: &Handles<B>,
        widest: [u64; 3],
        capacity: u32,
    ) -> Result<Option<Seats<B>>> {
        if widest.iter().all(|&b| b == 0) || capacity == 0 {
            return Ok(None);
        }
        let mut slabs = Vec::new();
        let mut seat = [None; 3];
        let mut kind_off = [0u64; 3];
        let mut slot_bytes = 0u64;
        for kind in Kind::ALL {
            let per = widest[kind.at()];
            if per == 0 {
                continue;
            }
            let cap = per * u64::from(capacity);
            let buffer = B::Buffer::zeroed(device, cap, crate::books::owners::EXPERT_SEATS)?;
            let handle = handles.bind(&buffer, 0, cap)?;
            slabs.push((kind, buffer));
            seat[kind.at()] = Some((handle, per));
            kind_off[kind.at()] = slot_bytes;
            slot_bytes += cap;
        }
        let mut staging = Vec::with_capacity(RING);
        for _ in 0..RING {
            staging.push(B::Buffer::with(
                device,
                slot_bytes,
                crate::backend::Memory::Staging,
                crate::books::owners::EXPERT_SEATS,
            )?);
        }
        let routes = B::Buffer::zeroed(device, MAX_ROUTES * 4, crate::books::owners::EXPERT_SEATS)?;
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
            poem_ir::Dtype::I32,
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
        frame: &B::Frame,
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
            let len = plane.per * unique.len() as u64;
            let at = self.kind_off[kind.at()];

            B::stage_rows(staging, at, len.next_multiple_of(B::COPY_GRAIN), |base| {
                copy_rows(map, plane, unique, base)
            })?;
            let (_, slab) = self
                .slabs
                .iter()
                .find(|(k, _)| *k == kind)
                .expect("a seated kind has a slab");
            B::copy_unretained(
                frame,
                staging,
                at,
                slab,
                0,
                len.next_multiple_of(B::COPY_GRAIN),
            )?;
            out.push(Tensor::new(handle, self.capacity, like.width, like.dtype));
        }
        Ok(out)
    }
}

fn copy_row_at(map: &Mapping, plane: &HostPlane, seat: usize, row: i32, into: usize) -> Result<()> {
    if row < 0 || row as u32 >= plane.rows {
        return Err(Fault::Ceiling {
            what: "expert ids a route names",
            need: u64::try_from(row).unwrap_or(u64::MAX),
            have: u64::from(plane.rows),
        });
    }
    plane.copy_row(
        map,
        row as u64,
        (into + seat * plane.per as usize) as *mut u8,
    )
}

#[cfg(target_arch = "wasm32")]
fn copy_rows(map: &Mapping, plane: &HostPlane, unique: &[i32], into: usize) -> Result<()> {
    let _ = COPY_THREADS;
    for (seat, &row) in unique.iter().enumerate() {
        copy_row_at(map, plane, seat, row, into)?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn copy_rows(map: &Mapping, plane: &HostPlane, unique: &[i32], into: usize) -> Result<()> {
    if unique.is_empty() {
        return Ok(());
    }
    let threads = COPY_THREADS.min(unique.len());
    let chunk = unique.len().div_ceil(threads);
    let faults: Vec<Result<()>> = std::thread::scope(|scope| {
        unique
            .chunks(chunk)
            .enumerate()
            .map(|(at, rows)| {
                scope.spawn(move || {
                    for (i, &row) in rows.iter().enumerate() {
                        copy_row_at(map, plane, at * chunk + i, row, into)?;
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

#[must_use]
pub fn row_bytes(dtype: poem_ir::Dtype, width: u32) -> u64 {
    let width = u64::from(width);
    match dtype {
        poem_ir::Dtype::Mxfp4 | poem_ir::Dtype::U8g64 => width,
        poem_ir::Dtype::U4g64 | poem_ir::Dtype::U4g32 | poem_ir::Dtype::U4g64tiled => {
            width.div_ceil(2)
        }
        poem_ir::Dtype::U2g32 | poem_ir::Dtype::U2g64 | poem_ir::Dtype::U2g128 => width.div_ceil(4),
        other => width.saturating_mul(poem_compiler::arena::elem_bytes(other).unwrap_or(0)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Device,

    Host,
}

/// The gathered tables a load seats a fire's rows of (hashed n-gram
/// tables the host keeps), and the rows a token reads of them.
#[derive(Debug, Default, Clone)]
pub struct Plan {
    gathered: BTreeMap<usize, u32>,

    rows_per_token: u32,
}

impl Plan {
    pub fn of(trace: &Trace) -> Result<Plan> {
        let mut gathered = BTreeMap::new();
        let mut rows_per_token = 0u32;
        for node in &trace.nodes {
            if let Operation::Gather(poem_ir::Gather::Embed {
                ids,
                table,
                weights: None,
                concat: true,
                ..
            }) = &node.op
                && let Ok(at) = weight_of(trace, *table)
            {
                gathered.insert(at, ids.0);
            }

            if let Operation::Gather(
                poem_ir::Gather::NgramIds {
                    heads_per_ngram, ..
                }
                | poem_ir::Gather::NgramIdsChunked {
                    heads_per_ngram, ..
                },
            ) = &node.op
            {
                rows_per_token = rows_per_token.max(*heads_per_ngram);
            }
        }
        Ok(Plan {
            gathered,
            rows_per_token,
        })
    }

    #[must_use]
    pub fn gathered(&self) -> Vec<usize> {
        self.gathered.keys().copied().collect()
    }

    #[must_use]
    pub fn gathered_demand(&self, max_tokens: u32) -> u32 {
        if self.gathered.is_empty() {
            return 0;
        }
        max_tokens.saturating_mul(self.rows_per_token.max(1))
    }

    /// Where each plane lies; `streamed` (the tables and routed banks the
    /// weight cache streams, [`crate::store::rows`],
    /// [`crate::store::experts`]) on the host, their rows and experts read
    /// from the artifact into pool slots.
    pub fn tiers(
        &self,
        bytes: &[u64],
        planes: &BTreeMap<usize, Vec<usize>>,
        streamed: &[usize],
        device_cap: u64,
    ) -> Result<Layout> {
        let mut tier = vec![Tier::Device; bytes.len()];
        for &at in streamed {
            tier[at] = Tier::Host;
        }

        let mut by_router: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        let mut grouped = vec![false; bytes.len()];
        for (&at, &router) in &self.gathered {
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
        for &at in streamed {
            grouped[at] = true;
        }
        let dense: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !grouped[*at])
            .map(|(_, b)| *b)
            .sum();
        // Dense weights always live on the device: past it, nothing makes
        // room for them.
        let cap = device_cap;
        if dense > cap {
            return Err(Fault::Residency(format!(
                "this deployment's dense weights take {dense} bytes, and the device holds {cap} \
                 for them under the load's allowance once the working set and one sequence's \
                 kv pages and state slots are down. Dense weights always live in device \
                 memory (only routed experts and token-id tables leave it, through the weight \
                 cache), so serve a quantized deployment of this model, split it across \
                 devices with tensor parallelism, raise `[engine.memory] budget_device`, or \
                 shorten `[engine] max_context`."
            )));
        }
        let mut device = dense;
        let mut host: u64 = streamed.iter().map(|&at| bytes[at]).sum();
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
        let tiers = Tiers { device, host };
        admit_streaming(tiers)?;
        Ok(Layout { tier, tiers })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn admit_streaming(_tiers: Tiers) -> Result<()> {
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn admit_streaming(tiers: Tiers) -> Result<()> {
    if tiers.host == 0 {
        return Ok(());
    }
    Err(Fault::Residency(format!(
        "this load would stream {} bytes of routed experts from the artifact between \
         steps, and a browser host streams nothing: serve a smaller checkpoint",
        tiers.host
    )))
}

pub struct Gathered<B: Backend> {
    seats: Seats<B>,

    key: Cell<Option<(u32, u32, u32)>>,

    unique: RefCell<Vec<i32>>,
}

impl<B: Backend> std::fmt::Debug for Gathered<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gathered")
            .field("capacity", &self.seats.capacity())
            .finish()
    }
}

impl<B: Backend> Gathered<B> {
    pub fn reserve(
        device: &B::Context,
        handles: &Handles<B>,
        widest: [u64; 3],
        capacity: u32,
    ) -> Result<Option<Gathered<B>>> {
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
        handles: &Handles<B>,
        host: &HostTier<B>,
        bank: kernels_slang::Bank,
        ids: Tensor,
    ) -> Result<Option<(kernels_slang::Bank, Tensor)>> {
        if host.plane(bank.codes.buf).is_none() {
            return Ok(None);
        }
        let key = (ids.buf, ids.rows, ids.width);
        if self.key.get() != Some(key) {
            let n = u64::from(ids.rows) * u64::from(ids.width);
            B::with_frame(B::Frame::flush)
                .ok_or_else(|| Fault::Residency("no frame is recording on this thread".into()))??;
            let raw = handles.read(ids.buf, n * 4)?;
            let mut unique: Vec<i32> = Vec::new();
            let mut seat_of: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
            let mut seated = Vec::with_capacity(raw.len() / 4);
            for word in raw.as_chunks::<4>().0 {
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
        let seated = B::with_frame(|frame| self.seats.gather(frame, host.map(), &wanted, &unique))
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
}

impl Layout {
    #[must_use]
    pub fn resident(bytes: &[u64]) -> Layout {
        Layout {
            tier: vec![Tier::Device; bytes.len()],
            tiers: Tiers {
                device: bytes.iter().sum(),
                host: 0,
            },
        }
    }

    #[must_use]
    pub fn streams(&self) -> bool {
        self.tiers.host > 0
    }
}

fn weight_of(trace: &Trace, id: ValueId) -> Result<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Ok(*w as usize),
        _ => Err(Fault::Param {
            name: format!("value {}", id.0),
            why: "is read as a gathered table and is not a weight",
        }),
    }
}
