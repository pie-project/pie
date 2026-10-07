pub mod kv;

use engine::transfer::KvCopy;
use kernels_vulkan::{KvPool, RecurrentPool, Tensor};
use poem_ir::{CacheRow, Dtype, Trace};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles, Pending};
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
        space: u32,
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
pub struct SpaceSeat {
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

    pub spaces: Vec<SpaceSeat>,

    pub slot_ids: Tensor,

    pub slot_of_row: Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub src_page: u32,

    pub src_token: u32,

    pub dst_page: u32,

    pub dst_token: u32,

    pub tokens: u32,
}

impl Move {
    pub fn plan(copy: &KvCopy, page_size: u32) -> std::result::Result<Vec<Move>, String> {
        if copy.src_page_ids.len() != copy.dst_page_ids.len() {
            return Err(format!(
                "src_page_ids has {} entries and dst_page_ids {}",
                copy.src_page_ids.len(),
                copy.dst_page_ids.len()
            ));
        }
        let mut moves: Vec<Move> = Vec::with_capacity(copy.src_page_ids.len() + copy.moves.len());

        for (src, dst) in copy.src_page_ids.iter().zip(&copy.dst_page_ids) {
            moves.push(Move {
                src_page: *src,
                src_token: 0,
                dst_page: *dst,
                dst_token: 0,
                tokens: page_size,
            });
        }

        for (at, cell) in copy.moves.iter().enumerate() {
            if cell.src_token_offset >= page_size || cell.dst_token_offset >= page_size {
                return Err(format!(
                    "kv move {at} names token offsets {}/{} in pages of {page_size} tokens",
                    cell.src_token_offset, cell.dst_token_offset
                ));
            }

            if cell.src_page_id == cell.dst_page_id
                && cell.src_token_offset == cell.dst_token_offset
            {
                continue;
            }
            let run = moves.last_mut().filter(|run| {
                run.src_page == cell.src_page_id
                    && run.dst_page == cell.dst_page_id
                    && run.src_token + run.tokens == cell.src_token_offset
                    && run.dst_token + run.tokens == cell.dst_token_offset
                    && run.src_token + run.tokens < page_size
            });
            match run {
                Some(run) => run.tokens += 1,
                None => moves.push(Move {
                    src_page: cell.src_page_id,
                    src_token: cell.src_token_offset,
                    dst_page: cell.dst_page_id,
                    dst_token: cell.dst_token_offset,
                    tokens: 1,
                }),
            }
        }
        for run in &moves {
            if run.src_page != run.dst_page {
                continue;
            }
            let (lo, hi) = (
                u32::min(run.src_token, run.dst_token),
                u32::max(run.src_token, run.dst_token),
            );
            if hi - lo < run.tokens {
                return Err(format!(
                    "a kv move of {} tokens reads page {} from token {} and writes the same \
                     page at token {} — the two ends overlap, and a blit whose regions \
                     overlap is undefined rather than a shift",
                    run.tokens, run.src_page, run.src_token, run.dst_token
                ));
            }
        }
        Ok(moves)
    }
}

#[derive(Debug)]
pub struct Pools {
    slabs: Vec<Buffer>,
    shapes: Vec<Shape>,
    paging: Paging,

    watermark: engine::frame::Demand,

    state_scratch: Option<Buffer>,

    disk: Option<DiskKv>,
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
                    space,
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
                        space: *space,
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
                CacheRow::State { name, slab, dtype } => {
                    let stride: u64 = slab.iter().product();
                    let dtype = state_dtype(*dtype);
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, dtype)?;
                    slabs.push(Buffer::zeroed(device, bytes)?);
                    shapes.push(Shape::State { stride, dtype });
                }
            }
            debug_assert_eq!(slabs.len(), index + 1, "one allocation per cache row");
        }
        let widest = shapes
            .iter()
            .map(|shape| match shape {
                Shape::State { stride, dtype } => stride * u64::from(elem_size(*dtype)),
                Shape::Kv { .. } => 0,
            })
            .max()
            .unwrap_or(0);
        let state_scratch = (widest > 0)
            .then(|| Buffer::zeroed(device, widest))
            .transpose()?;
        Ok(Pools {
            slabs,
            shapes,
            paging,
            watermark: engine::frame::Demand::ZERO,
            state_scratch,
            disk: None,
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
                    space,
                    head_dim,
                    kv_heads,
                    dtype,
                    plane_bytes,
                    values_at,
                    values_width,
                    values_bytes,
                } => {
                    let seat = seats
                        .spaces
                        .get(space as usize)
                        .ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "cache space {space}, for which this fire wrote no \
                                     geometry"
                            ),
                        })?;
                    let cells = self.paging.pages() * u64::from(self.paging.page_size);
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
        engine::frame::Supply::commit(
            self,
            engine::frame::Demand {
                kv_pages: 0,
                state_slots: src.max(dst) + 1,
                workspace: 0,
            },
        )?;
        let Some(scratch) = self.state_scratch.as_ref() else {
            return Ok(());
        };
        for (slab, shape) in self.slabs.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            frame.copy(slab, u64::from(src) * bytes, scratch, 0, bytes)?;
            frame.copy(scratch, 0, slab, u64::from(dst) * bytes, bytes)?;
        }
        Ok(())
    }

    pub fn copy_kv(&mut self, frame: &mut Frame, moves: &[Move]) -> Result<()> {
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

        engine::frame::Supply::commit(
            self,
            engine::frame::Demand {
                kv_pages: highest,
                state_slots: 0,
                workspace: 0,
            },
        )?;
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

    /// Every kv plane as its slab, the offset its page 0 starts at, and the
    /// bytes one token takes in it.
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

    /// Opens the slot file under `dir` with `budget` bytes of kv pages, and
    /// the host-visible stage copies to and from it go through; a budget
    /// short of one page seats none.
    pub fn seat_disk(
        &mut self,
        device: &Context,
        dir: &std::path::Path,
        budget: u64,
    ) -> Result<()> {
        let page_bytes = u64::from(self.paging.page_size)
            * self
                .kv_planes()
                .iter()
                .map(|&(_, _, cell)| cell)
                .sum::<u64>();
        let file = engine::disk::SlotFile::open(dir, 0, page_bytes, budget).map_err(slot_file)?;
        self.disk = match file {
            Some(file) => {
                let stage = Buffer::host(device, (file.stride() + engine::disk::ALIGN) as u64)?;
                let mapped = stage.mapped_ptr().ok_or(Fault::Deviceless)?;
                Some(DiskKv {
                    file,
                    stage,
                    at: mapped.next_multiple_of(engine::disk::ALIGN) - mapped,
                    landing: None,
                })
            }
            None => None,
        };
        Ok(())
    }

    #[must_use]
    pub fn disk_pages(&self) -> u32 {
        self.disk.as_ref().map_or(0, |disk| disk.file.slots())
    }

    /// Copies whole kv pages between the device pool and the slot file,
    /// `pages[i]` with `slots[i]`, through the stage a page at a time. A
    /// write returns once the bytes the fires before it left are on disk; a
    /// read returns once its copies onto the device are submitted.
    pub fn spill_kv(
        &mut self,
        device: &Context,
        to_disk: bool,
        pages: &[u32],
        slots: &[u32],
    ) -> Result<()> {
        let have = u64::from(self.disk_pages());
        if let Some(&slot) = slots.iter().find(|&&slot| u64::from(slot) >= have) {
            return Err(Fault::Ceiling {
                what: "disk kv pages",
                need: u64::from(slot) + 1,
                have,
            });
        }
        engine::frame::Supply::commit(
            self,
            engine::frame::Demand {
                kv_pages: pages.iter().map(|&page| page + 1).max().unwrap_or(0),
                state_slots: 0,
                workspace: 0,
            },
        )?;
        let Some(mut disk) = self.disk.take() else {
            return Ok(());
        };
        let copied = self.spill_through(&mut disk, device, to_disk, pages, slots);
        self.disk = Some(disk);
        copied
    }

    fn spill_through(
        &self,
        disk: &mut DiskKv,
        device: &Context,
        to_disk: bool,
        pages: &[u32],
        slots: &[u32],
    ) -> Result<()> {
        let page_size = u64::from(self.paging.page_size);
        let planes = self.kv_planes();
        let stride = disk.file.stride();
        let base = disk.stage.mapped_ptr().ok_or(Fault::Deviceless)? + disk.at;
        for (&page, &slot) in pages.iter().zip(slots) {
            if let Some(landing) = disk.landing.take() {
                landing.wait()?;
            }
            // SAFETY: the stage stays mapped while `disk` lives, and no copy
            // in flight touches it now (waited above, or committed below).
            let staged = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, stride) };
            if !to_disk {
                disk.file.read(slot, staged).map_err(slot_file)?;
            }
            let mut frame = device.frame()?;
            let mut offset = disk.at as u64;
            for &(slab, plane, cell) in &planes {
                let bytes = page_size * cell;
                let on_device = plane + u64::from(page) * bytes;
                if to_disk {
                    frame.copy(slab, on_device, &disk.stage, offset, bytes)?;
                } else {
                    frame.copy(&disk.stage, offset, slab, on_device, bytes)?;
                }
                offset += bytes;
            }
            if to_disk {
                frame.commit()?;
                disk.file.write(slot, staged).map_err(slot_file)?;
            } else {
                disk.landing = Some(frame.commit_async(None)?);
            }
        }
        Ok(())
    }
}

/// The slot file kv is suspended into, and the host-visible buffer a copy to
/// or from it is staged through, one page in slot layout.
#[derive(Debug)]
struct DiskKv {
    file: engine::disk::SlotFile,
    stage: Buffer,
    /// Where the stage's first slot sits past the mapping, which need not
    /// start on the boundary direct I/O asks for.
    at: usize,
    /// A copy out of the stage may still be in flight.
    landing: Option<Pending>,
}

fn slot_file(error: std::io::Error) -> Fault {
    Fault::Device {
        call: "kv slot file",
        why: error.to_string(),
    }
}

impl engine::frame::Supply for Pools {
    type Error = Fault;

    fn commit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        if demand.state_slots > self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(demand.state_slots),
                have: u64::from(self.paging.slots),
            });
        }
        let pages = self.paging.pages();
        if u64::from(demand.kv_pages) > pages {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(demand.kv_pages),
                have: pages,
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

    fn trim(&mut self, hint: engine::frame::Demand) {
        self.watermark = engine::frame::Demand {
            kv_pages: self.watermark.kv_pages.min(hint.kv_pages),
            state_slots: self.watermark.state_slots.min(hint.state_slots),
            workspace: self.watermark.workspace.min(hint.workspace),
        };
    }
}

pub fn pool_demand(trace: &Trace, paging: Paging) -> Result<u64> {
    let mut bytes: u64 = 0;
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
                bytes = bytes.saturating_add(cells * width * element);
            }
            CacheRow::State { name, slab, dtype } => {
                let stride: u64 = slab.iter().product();
                let dtype = state_dtype(*dtype);
                bytes = bytes.saturating_add(
                    stride
                        .saturating_mul(u64::from(paging.slots))
                        .saturating_mul(elem_bytes(name, dtype)?),
                );
            }
        }
    }
    Ok(bytes)
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

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;

    #[test]
    fn kv_pages_come_back_from_disk_where_they_are_restored() {
        if !crate::device::present() {
            eprintln!("no Vulkan device on this machine; skipping");
            return;
        }
        let device = Context::bind(&crate::DeviceBoot::default()).expect("a Vulkan device");
        let paging = Paging {
            page_size: 16,
            pages_per_slot: 1,
            slots: 1,
            pages: 4,
            window: None,
        };
        let cell = 3 * 4u64;
        let plane = paging.pages() * u64::from(paging.page_size) * cell;
        let mut pools = Pools {
            slabs: vec![Buffer::zeroed(&device, 2 * plane).expect("a kv slab")],
            shapes: vec![Shape::Kv {
                space: 0,
                head_dim: 3,
                kv_heads: 1,
                dtype: STATE_DTYPE,
                plane_bytes: plane,
                values_at: plane,
                values_width: 3,
                values_bytes: plane,
            }],
            paging,
            watermark: engine::frame::Demand::ZERO,
            state_scratch: None,
            disk: None,
        };
        let bytes: Vec<u8> = (0..2 * plane).map(|at| (at % 251) as u8).collect();
        let mut staged = Buffer::host(&device, 2 * plane).expect("a host buffer");
        staged.write(0, &bytes).expect("staged");
        let mut frame = device.frame().expect("a frame");
        frame
            .copy(&staged, 0, &pools.slabs[0], 0, 2 * plane)
            .expect("uploaded");
        frame.commit().expect("committed");
        let dir = std::env::temp_dir().join(format!("pie-vulkan-kv-{}", std::process::id()));
        pools
            .seat_disk(&device, &dir, 1 << 20)
            .expect("a slot file");
        pools
            .spill_kv(&device, true, &[1, 3], &[7, 8])
            .expect("suspended");
        let mut frame = device.frame().expect("a frame");
        frame
            .fill_zero(&pools.slabs[0], 0, 2 * plane)
            .expect("cleared");
        frame.commit().expect("committed");
        pools
            .spill_kv(&device, false, &[2, 0], &[7, 8])
            .expect("restored");
        let mut frame = device.frame().expect("a frame");
        frame
            .copy(&pools.slabs[0], 0, &staged, 0, 2 * plane)
            .expect("downloaded");
        frame.commit().expect("committed");
        let mut back = vec![0u8; bytes.len()];
        staged.read(0, &mut back).expect("read back");
        let page = plane / 4;
        for base in [0, plane] {
            let at = |id: u64| (base + id * page) as usize..(base + (id + 1) * page) as usize;
            assert_eq!(back[at(2)], bytes[at(1)]);
            assert_eq!(back[at(0)], bytes[at(3)]);
            assert!(back[at(1)].iter().all(|&b| b == 0));
        }
        drop(pools);
        std::fs::remove_dir_all(dir).expect("the slot file removed");
    }
}
