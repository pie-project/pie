pub mod kv;

use engine::transfer::KvRun;
use kernels_slang::{KvPool, Tensor};

/// A layer's recurrent cache: the state bank, each row's slot in it, and the
/// convolution window before and after the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecurrentPool {
    pub state: Tensor,

    pub slots: Tensor,

    pub conv_state: Tensor,

    pub new_conv_state: Tensor,
}
use poem_ir::{CacheRow, Dtype, Trace};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles, Memory};
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable};
use crate::store::kv::{Facts, Paging};

impl From<poem_exec::store::Fault> for Fault {
    fn from(fault: poem_exec::store::Fault) -> Fault {
        match fault {
            poem_exec::store::Fault::Ceiling { what, need, have } => {
                Fault::Ceiling { what, need, have }
            }
            poem_exec::store::Fault::Unbound { what } => Fault::Unbound { what },
            poem_exec::store::Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            } => Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            },
        }
    }
}

const STATE_DTYPE: Dtype = Dtype::F32;

fn state_dtype(declared: Dtype) -> Dtype {
    match declared {
        Dtype::I32 => Dtype::I32,
        _ => STATE_DTYPE,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Kv {
        head_dim: u32,
        kv_heads: u32,
        dtype: Dtype,

        plane_bytes: u64,

        values_at: u64,

        values_width: u64,
        values_bytes: u64,
    },

    State {
        stride: u64,
        dtype: Dtype,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct TableSeat {
    pub page_indptr: Tensor,

    pub page_indices: Tensor,

    pub last_page_lens: Tensor,

    pub row_valid: Tensor,
}

#[derive(Debug, Clone)]
pub struct Seats {
    pub lanes: u32,

    pub rows: u32,

    pub pages: u32,

    pub page_tables: Vec<TableSeat>,

    pub slot_ids: Tensor,

    pub slot_of_row: Tensor,
}

#[derive(Debug)]
pub struct Pools {
    slabs: Vec<Buffer>,
    shapes: Vec<Shape>,
    paging: Paging,
    /// The kv pages the slabs hold, up to what `paging` declares: a slab
    /// cannot give memory back, so a smaller pool is smaller slabs.
    pages: u64,

    watermark: engine::frame::Demand,

    disk: Option<Spill>,
    /// No host tier here, so parked rs rows go to disk too.
    parked: Option<Spill>,
    parked_pages: Option<Spill>,
}

impl Pools {
    pub fn reserve(
        device: &Context,
        trace: &Trace,
        paging: Paging,
        facts: &Facts,
    ) -> Result<Pools> {
        let mut slabs = Vec::with_capacity(trace.caches.len());
        let mut shapes = Vec::with_capacity(trace.caches.len());

        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    ..
                } => {
                    let planes = split(name, planes)?;
                    let width = planes.keys;

                    let restated = facts
                        .rows
                        .get(index)
                        .copied()
                        .flatten()
                        .filter(|seat| seat.kv_heads != 0);
                    if let Some(seat) = restated {
                        let heads = u64::from(seat.kv_heads) * u64::from(seat.head_dim);
                        if heads != width {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, whose row is {width} wide while its \
                                     consumers state {} heads of {}",
                                    seat.kv_heads, seat.head_dim
                                ),
                            });
                        }
                    }

                    let head_dim = restated.map_or(width, |seat| u64::from(seat.head_dim));
                    let kv_heads = restated.map_or(1, |seat| u64::from(seat.kv_heads));
                    let element = elem_bytes(name, *dtype)?;
                    let cells = paging.pages() * u64::from(paging.page_size);
                    let plane = cells * width * element;
                    let values_bytes = cells * planes.values * element;
                    let own_values = !planes.shared && planes.values != 0;
                    slabs.push(Buffer::zeroed(
                        device,
                        plane + if own_values { values_bytes } else { 0 },
                    )?);
                    shapes.push(Shape::Kv {
                        head_dim: u32::try_from(head_dim).unwrap_or(u32::MAX),
                        kv_heads: u32::try_from(kv_heads).unwrap_or(u32::MAX),
                        dtype: *dtype,
                        plane_bytes: plane,
                        values_at: if own_values { plane } else { 0 },
                        values_width: planes.values,

                        values_bytes: if planes.values == 0 {
                            plane
                        } else {
                            values_bytes
                        },
                    });
                }
                CacheRow::State {
                    name, slab, dtype, ..
                } => {
                    let stride: u64 = slab.iter().product();
                    let dtype = state_dtype(*dtype);
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, dtype)?;
                    slabs.push(Buffer::zeroed(device, bytes)?);
                    shapes.push(Shape::State { stride, dtype });
                }
            }
            debug_assert_eq!(slabs.len(), index + 1, "one allocation per cache row");
        }
        Ok(Pools {
            slabs,
            shapes,
            pages: paging.pages(),
            paging,
            watermark: engine::frame::Demand::ZERO,
            disk: None,
            parked: None,
            parked_pages: None,
        })
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|shape| match shape {
                Shape::State { stride, dtype } => stride * u64::from(elem_size(*dtype)),
                Shape::Kv { .. } => 0,
            })
            .sum()
    }

    pub fn read_slot(&self, slot: u32) -> Result<Vec<u8>> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        let mut out = Vec::new();
        for (slab, shape) in self.slabs.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            let mut span = vec![0u8; usize::try_from(bytes).unwrap_or(0)];
            slab.read(u64::from(slot) * bytes, &mut span)?;
            out.extend_from_slice(&span);
        }
        Ok(out)
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::State { .. }))
    }

    #[must_use]
    pub fn watermark(&self) -> engine::frame::Demand {
        self.watermark
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slabs.iter().map(Buffer::bytes).sum()
    }

    pub fn table(&self, handles: &Handles, seats: &Seats) -> Result<CacheTable> {
        let mut rows = Vec::with_capacity(self.shapes.len());
        for (slab, shape) in self.slabs.iter().zip(&self.shapes) {
            rows.push(match *shape {
                Shape::Kv {
                    head_dim,
                    kv_heads,
                    dtype,
                    plane_bytes,
                    values_at,
                    values_width,
                    values_bytes,
                } => {
                    // The one page table every kv row reads here.
                    let seat = seats.page_tables.first().ok_or_else(|| Fault::Unbound {
                        what: "the page table, which this fire staged none of".to_string(),
                    })?;
                    let cells = self.pages * u64::from(self.paging.page_size);
                    let plane = |at: u64, bytes: u64, width: u64| -> Result<Tensor> {
                        Ok(Tensor::new(
                            handles.bind(slab, at, bytes)?,
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        ))
                    };
                    CachePool::Kv(KvPool {
                        keys: plane(0, plane_bytes, u64::from(kv_heads) * u64::from(head_dim))?,
                        values: plane(values_at, values_bytes, values_width)?,
                        page_indices: seat.page_indices,
                        page_indptr: seat.page_indptr,
                        page_size: narrow(u64::from(self.paging.page_size)),

                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                    })
                }
                Shape::State { stride, dtype } => {
                    let bytes = stride * u64::from(self.paging.slots) * u64::from(elem_size(dtype));

                    let bank = Tensor::new(
                        handles.bind(slab, 0, bytes)?,
                        self.paging.slots,
                        u32::try_from(stride).unwrap_or(u32::MAX),
                        dtype,
                    );
                    CachePool::Recurrent(RecurrentPool {
                        state: bank,
                        slots: seats.slot_of_row,
                        conv_state: bank,
                        new_conv_state: bank,
                    })
                }
            });
        }
        Ok(CacheTable(rows))
    }

    pub fn clear(&mut self, slot: u32) -> Result<()> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        for (slab, shape) in self.slabs.iter_mut().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            slab.zero_span(u64::from(slot) * bytes, bytes)?;
        }
        Ok(())
    }

    pub fn copy_slot(&mut self, frame: &mut Frame, src: u32, dst: u32) -> Result<()> {
        let have = u64::from(self.paging.slots);
        for slot in [src, dst] {
            if u64::from(slot) >= have {
                return Err(Fault::Ceiling {
                    what: "recurrent slots",
                    need: u64::from(slot) + 1,
                    have,
                });
            }
        }
        if src == dst {
            return Ok(());
        }
        self.admit(engine::frame::Demand {
            kv_pages: 0,
            state_slots: src.max(dst) + 1,
            workspace: 0,
        })?;
        for (slab, shape) in self.slabs.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            frame.copy(
                slab,
                u64::from(src) * bytes,
                slab,
                u64::from(dst) * bytes,
                bytes,
            )?;
        }
        Ok(())
    }

    pub fn copy_kv(&mut self, frame: &mut Frame, moves: &[KvRun]) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }
        let page_size = u64::from(self.paging.page_size);
        let mut highest = 0u32;
        for span in moves {
            for (page, token) in [
                (span.src_page, span.src_token),
                (span.dst_page, span.dst_token),
            ] {
                let end = u64::from(token) + u64::from(span.tokens);
                if end > page_size {
                    return Err(Fault::Ceiling {
                        what: "token slots in one kv page",
                        need: end,
                        have: page_size,
                    });
                }
                highest = highest.max(page.saturating_add(1));
            }
        }

        self.admit(engine::frame::Demand {
            kv_pages: highest,
            state_slots: 0,
            workspace: 0,
        })?;
        for (slab, plane, cell) in self.kv_planes() {
            for span in moves {
                if span.tokens == 0 {
                    continue;
                }
                let bytes = u64::from(span.tokens) * cell;
                let at = |page: u32, token: u32| {
                    plane + (u64::from(page) * page_size + u64::from(token)) * cell
                };
                let (src, dst) = (
                    at(span.src_page, span.src_token),
                    at(span.dst_page, span.dst_token),
                );
                if src == dst {
                    continue;
                }

                slab.span(src, bytes)?;
                slab.span(dst, bytes)?;
                frame.copy(slab, src, slab, dst, bytes)?;
            }
        }
        Ok(())
    }

    /// Every kv plane as its slab, the offset its page 0 starts at in the
    /// slab as it stands now, and the bytes one token takes in it.
    fn kv_planes(&self) -> Vec<(&Buffer, u64, u64)> {
        let mut planes = Vec::new();
        for (slab, shape) in self.slabs.iter().zip(&self.shapes) {
            let Shape::Kv {
                head_dim,
                kv_heads,
                dtype,
                values_at,
                values_width,
                ..
            } = *shape
            else {
                continue;
            };
            let element = u64::from(elem_size(dtype));
            planes.push((slab, 0, u64::from(kv_heads) * u64::from(head_dim) * element));
            if values_at != 0 {
                planes.push((slab, values_at, values_width * element));
            }
        }
        planes
    }

    fn state_planes(&self) -> Vec<(&Buffer, u64, u64)> {
        self.slabs
            .iter()
            .zip(&self.shapes)
            .filter_map(|(slab, shape)| match *shape {
                Shape::State { stride, dtype } => {
                    Some((slab, 0, stride * u64::from(elem_size(dtype))))
                }
                Shape::Kv { .. } => None,
            })
            .collect()
    }

    pub fn seat_disk(&mut self, dir: &std::path::Path, budget: u64, page_bytes: u64) -> Result<()> {
        let stride = |bytes: u64| bytes.next_multiple_of(engine::disk::ALIGN as u64);
        let (state, page) = (self.state_slot_bytes(), page_bytes);
        let pair = stride(state) + stride(page);
        let rows = u64::from(self.paging.slots).min(budget.checked_div(pair).unwrap_or(0));
        self.parked = Spill::open(dir, "r0.rs", state, rows * stride(state))?;
        self.parked_pages = Spill::open(dir, "r0.rsb", page, rows * stride(page))?;
        let kv = budget - rows * pair;
        self.disk = Spill::open(dir, "r0.kv", self.kv_page_bytes(), kv)?;
        Ok(())
    }

    pub fn unseat_disk(&mut self) {
        self.disk = None;
        self.parked = None;
        self.parked_pages = None;
    }

    #[must_use]
    pub fn disk_pages(&self) -> u32 {
        Spill::slots(self.disk.as_ref())
    }

    #[must_use]
    pub fn parked_slots(&self) -> u32 {
        Spill::slots(self.parked.as_ref())
    }

    #[must_use]
    pub fn parked_pages(&self) -> u32 {
        Spill::slots(self.parked_pages.as_ref())
    }

    /// Copies whole kv pages between the device pool and the slot file,
    /// `pages[i]` with `slots[i]`.
    pub fn spill_kv(
        &mut self,
        device: &Context,
        to_disk: bool,
        pages: &[u32],
        slots: &[u32],
    ) -> Result<()> {
        Spill::room(self.disk.as_ref(), "disk kv pages", slots)?;
        self.admit(engine::frame::Demand {
            kv_pages: pages.iter().map(|&page| page + 1).max().unwrap_or(0),
            state_slots: 0,
            workspace: 0,
        })?;
        let Some(disk) = self.disk.as_ref() else {
            return Ok(());
        };
        let page_size = u64::from(self.paging.page_size);
        let planes: Vec<(&Buffer, u64, u64)> = self
            .kv_planes()
            .into_iter()
            .map(|(slab, at, cell)| (slab, at, page_size * cell))
            .collect();
        disk.copy(device, &planes, to_disk, pages, slots)
    }

    pub fn park_state(
        &mut self,
        device: &Context,
        out: bool,
        device_slots: &[u32],
        rows: &[u32],
    ) -> Result<()> {
        let have = u64::from(self.paging.slots);
        if let Some(&slot) = device_slots.iter().find(|&&slot| u64::from(slot) >= have) {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have,
            });
        }
        Spill::room(self.parked.as_ref(), "parked rs rows", rows)?;
        self.admit(engine::frame::Demand {
            kv_pages: 0,
            state_slots: device_slots.iter().map(|&slot| slot + 1).max().unwrap_or(0),
            workspace: 0,
        })?;
        match self.parked.as_ref() {
            Some(parked) => parked.copy(device, &self.state_planes(), out, device_slots, rows),
            None => Ok(()),
        }
    }

    pub fn park_pages(
        &self,
        device: &Context,
        out: bool,
        pages: &[u32],
        rows: &[u32],
        buffer: (&Buffer, u64),
    ) -> Result<()> {
        Spill::room(self.parked_pages.as_ref(), "parked rs buffer pages", rows)?;
        match self.parked_pages.as_ref() {
            Some(parked) => parked.copy(device, &[(buffer.0, 0, buffer.1)], out, pages, rows),
            None => Ok(()),
        }
    }
}

/// The most one staging copy of a spill carries. Pressure is what brings a
/// spill on, so a stage does not grow with the request.
const STAGE_BYTES: usize = 32 << 20;

#[derive(Debug)]
struct Spill {
    file: engine::disk::SlotFile,
}

impl Spill {
    fn open(
        dir: &std::path::Path,
        name: &str,
        slot_bytes: u64,
        budget: u64,
    ) -> Result<Option<Spill>> {
        let file =
            engine::disk::SlotFile::open(dir, name, slot_bytes, budget).map_err(slot_file)?;
        Ok(file.map(|file| Spill { file }))
    }

    fn slots(spill: Option<&Spill>) -> u32 {
        spill.map_or(0, |spill| spill.file.slots())
    }

    fn room(spill: Option<&Spill>, what: &'static str, slots: &[u32]) -> Result<()> {
        let have = u64::from(Spill::slots(spill));
        match slots.iter().find(|&&slot| u64::from(slot) >= have) {
            Some(&slot) => Err(Fault::Ceiling {
                what,
                need: u64::from(slot) + 1,
                have,
            }),
            None => Ok(()),
        }
    }

    /// Returns once every row is on disk or on the device. The stages live
    /// for the call, so an idle tier holds no memory. No drain: every frame
    /// was submitted before the lane took this copy, and the queue runs
    /// submissions in order, so a row leaving reads what those frames wrote
    /// and a row arriving is there for the frame after.
    fn copy(
        &self,
        device: &Context,
        planes: &[(&Buffer, u64, u64)],
        to_disk: bool,
        ids: &[u32],
        slots: &[u32],
    ) -> Result<()> {
        let stride = self.file.stride();
        let batch = (STAGE_BYTES / stride).max(1).min(ids.len());
        if batch == 0 {
            return Ok(());
        }
        let kind = if to_disk {
            Memory::Readback
        } else {
            Memory::Staging
        };
        let mut stage = Buffer::with(device, (batch * stride) as u64, kind)?;
        // A multiple of the stride from an `ALIGN` boundary, so each slot in
        // it is one aligned stage.
        let mut host = engine::disk::Aligned::zeroed(batch * stride);
        for (ids, slots) in ids.chunks(batch).zip(slots.chunks(batch)) {
            let staged = &mut host.as_mut_slice()[..ids.len() * stride];
            if !to_disk {
                for (&slot, row) in slots.iter().zip(staged.chunks_exact_mut(stride)) {
                    self.file.read(slot, row).map_err(slot_file)?;
                }
                stage.write(0, staged)?;
            }
            let mut frame = device.frame()?;
            for (row, &id) in ids.iter().enumerate() {
                let mut offset = (row * stride) as u64;
                for &(slab, plane, bytes) in planes {
                    let on_device = plane + u64::from(id) * bytes;
                    if to_disk {
                        frame.copy(slab, on_device, &stage, offset, bytes)?;
                    } else {
                        frame.copy(&stage, offset, slab, on_device, bytes)?;
                    }
                    offset += bytes;
                }
            }
            frame.commit()?;
            if to_disk {
                stage.read(0, staged)?;
                for (&slot, row) in slots.iter().zip(staged.chunks_exact(stride)) {
                    self.file.write(slot, row).map_err(slot_file)?;
                }
            }
        }
        Ok(())
    }
}

fn slot_file(error: std::io::Error) -> Fault {
    Fault::Device {
        call: "slot file",
        why: error.to_string(),
    }
}

impl Pools {
    /// The bytes one kv page takes across every plane.
    #[must_use]
    pub fn kv_page_bytes(&self) -> u64 {
        u64::from(self.paging.page_size)
            * self
                .kv_planes()
                .iter()
                .map(|&(_, _, cell)| cell)
                .sum::<u64>()
    }

    /// Moves the kv into slabs of `pages` pages, keeping what the lower pages
    /// hold. The new slabs are all made before any copy, so a refused one
    /// leaves the pool as it stood. The caller has drained the frames.
    pub fn resize(&mut self, device: &Context, pages: u64) -> Result<()> {
        let pages = pages.clamp(1, self.paging.pages());
        if pages == self.pages {
            return Ok(());
        }
        let page_size = u64::from(self.paging.page_size);
        let (cells, kept) = (pages * page_size, pages.min(self.pages) * page_size);
        let mut moved = Vec::new();
        for (at, shape) in self.shapes.iter().enumerate() {
            let Shape::Kv {
                head_dim,
                kv_heads,
                dtype,
                values_at,
                values_width,
                ..
            } = *shape
            else {
                continue;
            };
            let element = u64::from(elem_size(dtype));
            let keys = u64::from(kv_heads) * u64::from(head_dim) * element;
            let values = values_width * element;
            let own_values = values_at != 0;
            let bytes = cells * keys + if own_values { cells * values } else { 0 };
            moved.push((at, keys, values, own_values, Buffer::zeroed(device, bytes)?));
        }
        let mut frame = device.frame()?;
        for (at, keys, values, own_values, next) in &moved {
            let slab = &self.slabs[*at];
            frame.copy(slab, 0, next, 0, kept * keys)?;
            if *own_values {
                let Shape::Kv { values_at, .. } = self.shapes[*at] else {
                    continue;
                };
                frame.copy(slab, values_at, next, cells * keys, kept * values)?;
            }
        }
        frame.commit()?;
        for (at, keys, values, own_values, next) in moved {
            self.slabs[at] = next;
            let Shape::Kv {
                plane_bytes,
                values_at,
                values_bytes,
                ..
            } = &mut self.shapes[at]
            else {
                continue;
            };
            *plane_bytes = cells * keys;
            *values_at = if own_values { *plane_bytes } else { 0 };
            *values_bytes = if values == 0 {
                *plane_bytes
            } else {
                cells * values
            };
        }
        self.pages = pages;
        Ok(())
    }

    /// Checks a frame's or copy's reach against the pool and lifts the
    /// watermark over it.
    pub fn admit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        if demand.state_slots > self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(demand.state_slots),
                have: u64::from(self.paging.slots),
            });
        }
        if u64::from(demand.kv_pages) > self.pages {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(demand.kv_pages),
                have: self.pages,
            });
        }

        if demand.workspace > 0 {
            return Err(Fault::Ceiling {
                what: "pool workspace bytes",
                need: demand.workspace,
                have: 0,
            });
        }
        self.watermark = self.watermark.union(demand);
        Ok(())
    }

    pub fn trim(&mut self, hint: engine::frame::Demand) {
        self.watermark = engine::frame::Demand {
            kv_pages: self.watermark.kv_pages.min(hint.kv_pages),
            state_slots: self.watermark.state_slots.min(hint.state_slots),
            workspace: self.watermark.workspace.min(hint.workspace),
        };
    }
}

/// The slab each cache row declares at `paging`, one device reservation a row.
pub fn pool_slabs(trace: &Trace, paging: Paging) -> Result<Vec<u64>> {
    let mut slabs = Vec::with_capacity(trace.caches.len());
    for row in &trace.caches {
        match row {
            CacheRow::Kv {
                name,
                planes,
                dtype,
                ..
            } => {
                let planes = split(name, planes)?;
                let element = elem_bytes(name, *dtype)?;
                let cells = paging.pages() * u64::from(paging.page_size);
                let width = if planes.shared {
                    planes.keys
                } else {
                    planes.keys + planes.values
                };
                slabs.push(cells * width * element);
            }
            CacheRow::State {
                name, slab, dtype, ..
            } => {
                let stride: u64 = slab.iter().product();
                let dtype = state_dtype(*dtype);
                slabs.push(
                    stride
                        .saturating_mul(u64::from(paging.slots))
                        .saturating_mul(elem_bytes(name, dtype)?),
                );
            }
        }
    }
    Ok(slabs)
}

struct Planes {
    keys: u64,
    values: u64,
    shared: bool,
}

fn split(name: &str, planes: &[u64]) -> Result<Planes> {
    match planes {
        [shared] => Ok(Planes {
            keys: *shared,
            values: *shared,
            shared: true,
        }),
        [keys, values] => Ok(Planes {
            keys: *keys,
            values: *values,
            shared: false,
        }),
        other => Err(Fault::Unbound {
            what: format!(
                "cache `{name}`, which declares {} plane(s) — this shell cuts kv \
                 pages into one shared plane or into a key half and a value half, \
                 and knows no other form",
                other.len()
            ),
        }),
    }
}

fn elem_bytes(name: &str, dtype: Dtype) -> Result<u64> {
    poem_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Unbound {
        what: format!("cache `{name}`, stored as {dtype:?}, which has no element size"),
    })
}

fn elem_size(dtype: Dtype) -> u32 {
    poem_compiler::arena::elem_bytes(dtype).unwrap_or(1) as u32
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

#[cfg(all(test, feature = "wgpu", not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    const PAGING: Paging = Paging {
        page_size: 16,
        pages_per_slot: 1,
        slots: 2,
        pages: 4,
        window: None,
    };

    fn device() -> Option<Context> {
        if !crate::device::present() {
            eprintln!("no wgpu adapter on this machine; skipping");
            return None;
        }
        Some(
            Context::bind(
                &crate::DeviceBoot::default(),
                engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
            )
            .expect("a wgpu device"),
        )
    }

    /// A pool of one kv row with a key and a value plane of `cell` bytes a
    /// token, filled with a ramp, and those bytes.
    fn pools_of(device: &Context, paging: Paging, cell: u64) -> (Pools, Vec<u8>) {
        let plane = paging.pages() * u64::from(paging.page_size) * cell;
        let mut pools = Pools {
            slabs: vec![Buffer::zeroed(device, 2 * plane).expect("a kv slab")],
            shapes: vec![Shape::Kv {
                head_dim: u32::try_from(cell / 4).expect("narrow"),
                kv_heads: 1,
                dtype: STATE_DTYPE,
                plane_bytes: plane,
                values_at: plane,
                values_width: cell / 4,
                values_bytes: plane,
            }],
            pages: paging.pages(),
            paging,
            watermark: engine::frame::Demand::ZERO,
            disk: None,
            parked: None,
            parked_pages: None,
        };
        let bytes: Vec<u8> = (0..2 * plane).map(|at| (at % 251) as u8).collect();
        pools.slabs[0].write(0, &bytes).expect("uploaded");
        (pools, bytes)
    }

    fn slab(pools: &Pools, at: usize) -> Vec<u8> {
        let mut back = vec![0u8; usize::try_from(pools.slabs[at].bytes()).expect("small")];
        pools.slabs[at].read(0, &mut back).expect("read back");
        back
    }

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("pie-wgpu-{name}-{}", std::process::id()))
    }

    #[test]
    fn kv_pages_and_rs_slots_come_back_from_disk_where_they_are_restored() {
        let Some(device) = device() else { return };
        let cell = 3 * 4u64;
        let row = 8 * 4u64;
        let page = 64u64;
        let (mut pools, bytes) = pools_of(&device, PAGING, cell);
        let plane = bytes.len() as u64 / 2;
        pools
            .slabs
            .push(Buffer::zeroed(&device, 2 * row).expect("a state slab"));
        pools.shapes.push(Shape::State {
            stride: 8,
            dtype: STATE_DTYPE,
        });
        let dir = scratch_dir("disk");
        pools.seat_disk(&dir, 1 << 20, page).expect("a slot file");
        pools
            .spill_kv(&device, true, &[1, 3], &[7, 8])
            .expect("suspended");
        pools.slabs[0].zero_span(0, 2 * plane).expect("cleared");
        pools
            .spill_kv(&device, false, &[2, 0], &[7, 8])
            .expect("restored");
        let back = slab(&pools, 0);
        let page_of = plane / 4;
        for base in [0, plane] {
            let at = |id: u64| (base + id * page_of) as usize..(base + (id + 1) * page_of) as usize;
            assert_eq!(back[at(2)], bytes[at(1)]);
            assert_eq!(back[at(0)], bytes[at(3)]);
            assert!(back[at(1)].iter().all(|&b| b == 0));
        }

        let mut pages = Buffer::zeroed(&device, 2 * page).expect("a buffer slab");
        pools.slabs[1]
            .write(row, &bytes[..row as usize])
            .expect("slot 1 uploaded");
        pages
            .write(page, &bytes[row as usize..(row + page) as usize])
            .expect("page 1 uploaded");
        pools.park_state(&device, true, &[1], &[0]).expect("parked");
        // Page 1 parks in row 0 of its own file, beside the state in row 0.
        pools
            .park_pages(&device, true, &[1], &[0], (&pages, page))
            .expect("parked");
        pools.slabs[1].zero_span(0, 2 * row).expect("cleared");
        pages.zero_span(0, 2 * page).expect("cleared");
        pools
            .park_state(&device, false, &[0], &[0])
            .expect("unparked");
        pools
            .park_pages(&device, false, &[0], &[0], (&pages, page))
            .expect("unparked");
        let state = slab(&pools, 1);
        let mut parked = vec![0u8; (2 * page) as usize];
        pages.read(0, &mut parked).expect("read back");
        let (row, page) = (row as usize, page as usize);
        assert_eq!(state[..row], bytes[..row]);
        assert!(state[row..].iter().all(|&b| b == 0));
        assert_eq!(parked[..page], bytes[row..row + page]);
        assert!(parked[page..].iter().all(|&b| b == 0));
        drop(pools);
        std::fs::remove_dir_all(dir).expect("the slot file removed");
    }

    #[test]
    fn a_page_spilled_comes_back_across_a_resize_of_the_pool() {
        let Some(device) = device() else { return };
        let (mut pools, bytes) = pools_of(&device, PAGING, 3 * 4);
        let plane = bytes.len() / 2;
        let page = plane / 4;
        let dir = scratch_dir("resize");
        pools.seat_disk(&dir, 1 << 20, 0).expect("a slot file");
        pools
            .spill_kv(&device, true, &[3], &[0])
            .expect("the page the shrink drops");

        pools.resize(&device, 2).expect("shrunk");
        assert!(
            pools.spill_kv(&device, true, &[2], &[1]).is_err(),
            "a page past the slabs as they stand"
        );
        // The values moved with the shrink: this spill reads them there.
        pools
            .spill_kv(&device, true, &[1], &[1])
            .expect("a page of the smaller slabs");

        pools.resize(&device, 4).expect("grown");
        pools.slabs[0]
            .zero_span(0, bytes.len() as u64)
            .expect("cleared");
        pools
            .spill_kv(&device, false, &[3, 2], &[0, 1])
            .expect("restored");
        let back = slab(&pools, 0);
        for base in [0, plane] {
            let at = |id: usize| base + id * page..base + (id + 1) * page;
            assert_eq!(back[at(3)], bytes[at(3)], "spilled before the shrink");
            assert_eq!(back[at(2)], bytes[at(1)], "spilled while small");
            assert!(back[at(0)].iter().all(|&b| b == 0));
            assert!(back[at(1)].iter().all(|&b| b == 0));
        }
        drop(pools);
        std::fs::remove_dir_all(dir).expect("the slot file removed");
    }

    #[test]
    fn a_slot_write_that_fails_leaves_the_page_on_the_device() {
        let Some(device) = device() else { return };
        let (mut pools, bytes) = pools_of(&device, PAGING, 3 * 4);
        let dir = scratch_dir("refused");
        pools.seat_disk(&dir, 1 << 20, 0).expect("a slot file");
        let page_size = u64::from(PAGING.page_size);
        let planes: Vec<(&Buffer, u64, u64)> = pools
            .kv_planes()
            .into_iter()
            .map(|(slab, at, cell)| (slab, at, page_size * cell))
            .collect();
        // Past `room`, so the refusal is the file's, after a slot went in.
        let past = pools.disk_pages();
        let disk = pools.disk.as_ref().expect("a slot file");
        let failed = disk
            .copy(&device, &planes, true, &[1, 2], &[0, past])
            .expect_err("a slot the file refuses");
        assert!(failed.to_string().contains("slot file"), "{failed}");
        drop(planes);
        assert_eq!(slab(&pools, 0), bytes, "the pages stay where they were");

        pools
            .spill_kv(&device, true, &[2], &[0])
            .expect("the tier still serves");
        pools.slabs[0]
            .zero_span(0, bytes.len() as u64)
            .expect("cleared");
        pools
            .spill_kv(&device, false, &[2], &[0])
            .expect("restored");
        let (plane, page) = (bytes.len() / 2, bytes.len() / 8);
        let back = slab(&pools, 0);
        for base in [0, plane] {
            let at = base + 2 * page..base + 3 * page;
            assert_eq!(back[at.clone()], bytes[at]);
        }
        drop(pools);
        std::fs::remove_dir_all(dir).expect("the slot file removed");
    }

    #[test]
    fn a_spill_wider_than_one_stage_lands_every_page() {
        let Some(device) = device() else { return };
        // Five pages of 12 MiB across both planes take three stages.
        let paging = Paging {
            page_size: 1 << 19,
            pages_per_slot: 1,
            slots: 1,
            pages: 5,
            window: None,
        };
        let (mut pools, bytes) = pools_of(&device, paging, 12);
        assert_eq!(STAGE_BYTES / pools.kv_page_bytes() as usize, 2);
        let plane = bytes.len() / 2;
        let dir = scratch_dir("stages");
        pools.seat_disk(&dir, 64 << 20, 0).expect("a slot file");
        let order = [0, 1, 2, 3, 4];
        pools
            .spill_kv(&device, true, &order, &order)
            .expect("suspended");
        pools.slabs[0]
            .zero_span(0, bytes.len() as u64)
            .expect("cleared");
        let reversed = [4, 3, 2, 1, 0];
        pools
            .spill_kv(&device, false, &reversed, &order)
            .expect("restored");
        let back = slab(&pools, 0);
        let cut = plane / 5;
        for base in [0, plane] {
            for id in 0..5 {
                let mirror = 4 - id;
                assert!(
                    back[base + id * cut..base + (id + 1) * cut]
                        == bytes[base + mirror * cut..base + (mirror + 1) * cut],
                    "page {id}"
                );
            }
        }
        drop(pools);
        std::fs::remove_dir_all(dir).expect("the slot file removed");
    }
}
