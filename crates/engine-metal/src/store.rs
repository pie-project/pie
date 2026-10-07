pub mod accounting;
pub mod kv;

use engine::transfer::KvCopy;
use kernels_metal::{KvPool, RecurrentPool, Tensor};
use poem_ir::{CacheRow, Dtype, Trace};

use crate::device::ctx::Frame;
use crate::device::elastic;
use crate::device::{Buffer, Context, Handles};
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

/// One cache row's layout over the planes.
///
/// A kv row is a key plane and a value plane; when the trace declares one
/// shared plane both indices name it. A state row is one plane of slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Kv {
        space: u32,
        head_dim: u32,
        kv_heads: u32,
        dtype: Dtype,
        keys: usize,
        values: usize,
        values_width: u64,
    },
    State {
        stride: u64,
        dtype: Dtype,
        plane: usize,
    },
}

/// One allocation the pool binds: fixed for its whole life, or a sparse
/// buffer whose memory is mapped up to the pool's watermark.
#[derive(Debug)]
enum Plane {
    Fixed(Buffer),
    #[cfg(target_vendor = "apple")]
    Elastic {
        /// The binding view over the sparse buffer; encoders and blits use
        /// it like any other buffer.
        view: Buffer,
        store: elastic::Elastic,
    },
}

impl Plane {
    fn buffer(&self) -> &Buffer {
        match self {
            Plane::Fixed(buffer) => buffer,
            #[cfg(target_vendor = "apple")]
            Plane::Elastic { view, .. } => view,
        }
    }

    fn bytes(&self) -> u64 {
        self.buffer().bytes()
    }

    /// The bytes with memory behind them: all of a fixed plane, what is
    /// mapped of an elastic one.
    fn committed(&self) -> u64 {
        match self {
            Plane::Fixed(buffer) => buffer.bytes(),
            #[cfg(target_vendor = "apple")]
            Plane::Elastic { store, .. } => store.committed(),
        }
    }

    fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        match self {
            Plane::Fixed(buffer) => buffer.read(offset, into),
            #[cfg(target_vendor = "apple")]
            Plane::Elastic { store, .. } => store.read_into(offset, into),
        }
    }

    /// The caller has drained every frame that touches these bytes.
    fn write(&mut self, offset: u64, from: &[u8]) -> Result<()> {
        match self {
            Plane::Fixed(buffer) => buffer.write(offset, from),
            #[cfg(target_vendor = "apple")]
            // SAFETY: the pool's contract, restated on every host write:
            // nothing on the GPU reads a page the host is writing.
            Plane::Elastic { store, .. } => unsafe { store.write_from(offset, from) },
        }
    }

    /// The caller has drained every frame that touches these bytes.
    fn zero(&mut self, offset: u64, len: u64) -> Result<()> {
        match self {
            Plane::Fixed(buffer) => buffer.zero_span(offset, len),
            #[cfg(target_vendor = "apple")]
            // SAFETY: as `write`.
            Plane::Elastic { store, .. } => unsafe { store.zero(offset, len) },
        }
    }
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

/// How the pool holds its memory: every Metal 4 device gets the elastic
/// pool, and a device that cannot map sparse buffer tiles keeps the fixed
/// allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backing {
    /// Every plane allocated whole at load.
    Fixed,
    /// Every plane a placement sparse buffer, mapped up to the watermark a
    /// frame commits and released back down on `release`.
    Elastic,
}

#[derive(Debug)]
pub struct Pools {
    planes: Vec<Plane>,
    shapes: Vec<Shape>,
    paging: Paging,
    watermark: engine::frame::Demand,
    disk: Option<DiskKv>,
    /// The budget every elastic plane draws on; `None` for a fixed pool.
    arena: Option<elastic::Arena>,
}

/// The slot file kv is suspended into, and the page a copy to or from it
/// gathers the planes in, in slot layout.
#[derive(Debug)]
struct DiskKv {
    file: engine::disk::SlotFile,
    stage: engine::disk::Aligned,
}

/// What one cache row needs of its planes, as bytes per token cell.
struct RowPlan {
    shape: Shape,
    /// Each plane this row allocates: its full length.
    planes: Vec<u64>,
}

impl Pools {
    pub fn reserve(
        device: &Context,
        trace: &Trace,
        paging: Paging,
        facts: &Facts,
    ) -> Result<Pools> {
        let mut planes = Vec::with_capacity(2 * trace.caches.len());
        let mut shapes = Vec::with_capacity(trace.caches.len());
        let rows = Pools::plan(trace, paging, facts)?;

        #[cfg(target_vendor = "apple")]
        let sparse = device.sparse()?;

        let arena = {
            #[cfg(target_vendor = "apple")]
            {
                sparse.as_ref().map(|_| {
                    let total: u64 = rows
                        .iter()
                        .flat_map(|row| row.planes.iter())
                        .map(|&bytes| elastic::pages_up(bytes))
                        .fold(0u64, u64::saturating_add);
                    elastic::Arena::new(total)
                })
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                None
            }
        };

        for row in rows {
            for &bytes in &row.planes {
                #[cfg(target_vendor = "apple")]
                if let (Some(sparse), Some(arena)) = (&sparse, &arena)
                    && bytes > 0
                {
                    let store = elastic::create(sparse, arena, bytes)?;
                    planes.push(Plane::Elastic {
                        view: store.view(),
                        store,
                    });
                    continue;
                }
                planes.push(Plane::Fixed(Buffer::zeroed(device, bytes)?));
            }
            shapes.push(row.shape);
        }
        Ok(Pools {
            planes,
            shapes,
            paging,
            watermark: engine::frame::Demand::ZERO,
            disk: None,
            arena,
        })
    }

    /// Each cache row's shape and the planes it takes, before any is made.
    fn plan(trace: &Trace, paging: Paging, facts: &Facts) -> Result<Vec<RowPlan>> {
        let mut rows = Vec::with_capacity(trace.caches.len());
        let mut next = 0usize;
        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    space,
                    head_dim: trace_head_dim,
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
                    // Per-head geometry: a restated seat is authoritative; else fall
                    // back to the trace row's own head_dim (matching `pool_demand`), so
                    // KvU4 planes size per head at ANY head_dim rather than from the
                    // 256 anchor, and `plan`/`reserve` agree with the demand estimate.
                    // A pre-field trace (head_dim 0) leaves it unknown: `row_stride`
                    // uses the format anchor and `table` the whole-row width, as before.
                    let row_head_dim = (*trace_head_dim != 0).then_some(*trace_head_dim);
                    let head_dim = restated
                        .map(|seat| u64::from(seat.head_dim))
                        .or(row_head_dim.map(u64::from))
                        .unwrap_or(width);
                    let kv_heads = restated
                        .map(|seat| u64::from(seat.kv_heads))
                        .or(row_head_dim.map(|d| width / u64::from(d)))
                        .unwrap_or(1);
                    let block = restated.map(|seat| seat.head_dim).or(row_head_dim);
                    let cells = paging.pages() * u64::from(paging.page_size);
                    let plane = cells * row_stride(name, *dtype, width, block)?;
                    let values_bytes = if planes.values == 0 {
                        0
                    } else {
                        cells * row_stride(name, *dtype, planes.values, block)?
                    };
                    let own_values = !planes.shared && planes.values != 0;
                    let keys = next;
                    let values = if own_values { next + 1 } else { next };
                    let mut lengths = vec![plane];
                    if own_values {
                        lengths.push(values_bytes);
                    }
                    next += lengths.len();
                    rows.push(RowPlan {
                        shape: Shape::Kv {
                            space: *space,
                            head_dim: u32::try_from(head_dim).unwrap_or(u32::MAX),
                            kv_heads: u32::try_from(kv_heads).unwrap_or(u32::MAX),
                            dtype: *dtype,
                            keys,
                            values,
                            values_width: planes.values,
                        },
                        planes: lengths,
                    });
                }
                CacheRow::State { name, slab, dtype } => {
                    let stride: u64 = slab.iter().product();
                    let dtype = state_dtype(*dtype);
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, dtype)?;
                    rows.push(RowPlan {
                        shape: Shape::State {
                            stride,
                            dtype,
                            plane: next,
                        },
                        planes: vec![bytes],
                    });
                    next += 1;
                }
            }
        }
        Ok(rows)
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|shape| match shape {
                Shape::State { stride, dtype, .. } => stride * u64::from(elem_size(*dtype)),
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
        for shape in &self.shapes {
            let Shape::State {
                stride,
                dtype,
                plane,
            } = *shape
            else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            let mut span = vec![0u8; usize::try_from(bytes).unwrap_or(0)];
            self.planes[plane].read(u64::from(slot) * bytes, &mut span)?;
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

    /// Every plane's full length: what the pool would take if every page
    /// and slot were in use.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.planes.iter().map(Plane::bytes).sum()
    }

    /// The bytes with memory behind them right now.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.planes.iter().map(Plane::committed).sum()
    }

    /// The most that has had memory behind it at once.
    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        match &self.arena {
            Some(arena) => arena.budget().high_water,
            None => self.bytes(),
        }
    }

    #[must_use]
    pub fn is_elastic(&self) -> bool {
        self.arena.is_some()
    }

    /// The unit the elastic budget is quoted in; zero for a fixed pool.
    #[must_use]
    pub fn elastic_page_bytes(&self) -> u64 {
        if self.arena.is_some() {
            elastic::PAGE
        } else {
            0
        }
    }

    /// The elastic budget, in those units; zero for a fixed pool.
    #[must_use]
    pub fn elastic_budget_pages(&self) -> u64 {
        self.arena
            .as_ref()
            .map_or(0, |arena| elastic::pages_for_bytes(arena.budget().total))
    }

    pub fn table(&self, handles: &Handles, seats: &Seats) -> Result<CacheTable> {
        let mut rows = Vec::with_capacity(self.shapes.len());
        for shape in &self.shapes {
            rows.push(match *shape {
                Shape::Kv {
                    space,
                    head_dim,
                    kv_heads,
                    dtype,
                    keys,
                    values,
                    values_width,
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
                    let plane = |at: usize, width: u64| -> Result<Tensor> {
                        let buffer = self.planes[at].buffer();
                        Ok(Tensor::new(
                            handles.bind(buffer, 0, buffer.bytes())?,
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        ))
                    };
                    CachePool::Kv(KvPool {
                        keys: plane(keys, u64::from(kv_heads) * u64::from(head_dim))?,
                        values: plane(values, values_width)?,
                        page_indices: seat.page_indices,
                        page_indptr: seat.page_indptr,
                        page_size: narrow(u64::from(self.paging.page_size)),
                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                    })
                }
                Shape::State {
                    stride,
                    dtype,
                    plane,
                } => {
                    let bytes = stride * u64::from(self.paging.slots) * u64::from(elem_size(dtype));
                    let bank = Tensor::new(
                        handles.bind(self.planes[plane].buffer(), 0, bytes)?,
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
        if !self.has_state() {
            return Ok(());
        }
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        engine::frame::Supply::commit(
            self,
            engine::frame::Demand {
                kv_pages: 0,
                state_slots: slot.saturating_add(1),
                workspace: 0,
            },
        )?;
        for at in 0..self.shapes.len() {
            let Shape::State {
                stride,
                dtype,
                plane,
            } = self.shapes[at]
            else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            self.planes[plane].zero(u64::from(slot) * bytes, bytes)?;
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
        for (plane, cell) in self.kv_planes()? {
            let slab = self.planes[plane].buffer();
            for span in moves {
                if span.tokens == 0 {
                    continue;
                }
                let bytes = u64::from(span.tokens) * cell;
                let at =
                    |page: u32, token: u32| (u64::from(page) * page_size + u64::from(token)) * cell;
                let (src, dst) = (
                    at(span.src_page, span.src_token),
                    at(span.dst_page, span.dst_token),
                );
                if src == dst {
                    continue;
                }
                slab.span(src, bytes)?;
                slab.span(dst, bytes)?;
                frame.copy(slab.slab(), src, slab.slab(), dst, bytes)?;
            }
        }
        Ok(())
    }

    /// Every kv plane as its index and the bytes one token takes in it.
    fn kv_planes(&self) -> Result<Vec<(usize, u64)>> {
        let mut planes = Vec::new();
        for shape in &self.shapes {
            let Shape::Kv {
                head_dim,
                kv_heads,
                dtype,
                keys,
                values,
                values_width,
                ..
            } = *shape
            else {
                continue;
            };
            // Whole-slot cell strides. For a packed dtype these are the packed
            // row bytes (codes + inline scale), NOT width * element — a slot's
            // packed cells are contiguous, so a byte-blit still moves them, but
            // only when the cell size is the packed stride.
            let keys_cell = row_stride(
                "kv migration",
                dtype,
                u64::from(kv_heads) * u64::from(head_dim),
                Some(head_dim),
            )?;
            planes.push((keys, keys_cell));
            if values != keys {
                let values_cell = if values_width == 0 {
                    0
                } else {
                    row_stride("kv migration", dtype, values_width, Some(head_dim))?
                };
                planes.push((values, values_cell));
            }
        }
        Ok(planes)
    }

    fn page_bytes(&self) -> Result<u64> {
        let page_size = u64::from(self.paging.page_size);
        Ok(self
            .kv_planes()?
            .iter()
            .map(|&(_, cell)| page_size * cell)
            .sum())
    }

    /// Opens `dir`'s slot file with `budget` bytes of kv pages; a budget
    /// short of one page seats none.
    pub fn seat_disk(&mut self, dir: &std::path::Path, budget: u64) -> Result<u32> {
        let file =
            engine::disk::SlotFile::open(dir, 0, self.page_bytes()?, budget).map_err(slot_file)?;
        self.disk = file.map(|file| DiskKv {
            stage: engine::disk::Aligned::zeroed(file.stride()),
            file,
        });
        Ok(self.disk.as_ref().map_or(0, |disk| disk.file.slots()))
    }

    /// Copies whole kv pages between the pool and the slot file, `pages[i]`
    /// with `slots[i]`, one page through the stage at a time. The planes are
    /// host-addressable memory, so the caller has drained every command
    /// buffer that touches them.
    pub fn spill_kv(&mut self, to_disk: bool, pages: &[u32], slots: &[u32]) -> Result<()> {
        let have = self.disk.as_ref().map_or(0, |disk| disk.file.slots());
        if let Some(&slot) = slots.iter().find(|&&slot| slot >= have) {
            return Err(Fault::Ceiling {
                what: "disk kv pages",
                need: u64::from(slot) + 1,
                have: u64::from(have),
            });
        }
        let highest = pages.iter().map(|&page| page + 1).max().unwrap_or(0);
        engine::frame::Supply::commit(
            self,
            engine::frame::Demand {
                kv_pages: highest,
                state_slots: 0,
                workspace: 0,
            },
        )?;
        let Some(mut disk) = self.disk.take() else {
            return Ok(());
        };
        let copied = self.spill_through(&mut disk, to_disk, pages, slots);
        self.disk = Some(disk);
        copied
    }

    fn spill_through(
        &mut self,
        disk: &mut DiskKv,
        to_disk: bool,
        pages: &[u32],
        slots: &[u32],
    ) -> Result<()> {
        let page_size = u64::from(self.paging.page_size);
        let planes = self.kv_planes()?;
        for (&page, &slot) in pages.iter().zip(slots) {
            let staged = disk.stage.as_mut_slice();
            if !to_disk {
                disk.file.read(slot, staged).map_err(slot_file)?;
            }
            let mut offset = 0;
            for &(plane, cell) in &planes {
                let bytes = page_size * cell;
                let on_device = u64::from(page) * bytes;
                let on_host = &mut staged[offset..offset + bytes as usize];
                if to_disk {
                    self.planes[plane].read(on_device, on_host)?;
                } else {
                    self.planes[plane].write(on_device, on_host)?;
                }
                offset += bytes as usize;
            }
            if to_disk {
                disk.file.write(slot, staged).map_err(slot_file)?;
            }
        }
        Ok(())
    }

    pub fn copy_state(&mut self, frame: &mut Frame, moves: &[(u32, u32)]) -> Result<()> {
        if moves.is_empty() || !self.has_state() {
            return Ok(());
        }
        let mut highest = 0u32;
        for &(src, dst) in moves {
            for slot in [src, dst] {
                if slot >= self.paging.slots {
                    return Err(Fault::Ceiling {
                        what: "recurrent slots",
                        need: u64::from(slot) + 1,
                        have: u64::from(self.paging.slots),
                    });
                }
                highest = highest.max(slot.saturating_add(1));
            }
        }
        engine::frame::Supply::commit(
            self,
            engine::frame::Demand {
                kv_pages: 0,
                state_slots: highest,
                workspace: 0,
            },
        )?;
        for shape in &self.shapes {
            let Shape::State {
                stride,
                dtype,
                plane,
            } = *shape
            else {
                continue;
            };
            let slab = self.planes[plane].buffer();
            let bytes = stride * u64::from(elem_size(dtype));
            for &(src, dst) in moves {
                if src == dst {
                    continue;
                }
                let (from, to) = (u64::from(src) * bytes, u64::from(dst) * bytes);
                slab.span(from, bytes)?;
                slab.span(to, bytes)?;
                frame.copy(slab.slab(), from, slab.slab(), to, bytes)?;
            }
        }
        Ok(())
    }

    /// The bytes each plane needs mapped to serve `demand`, rounded up to
    /// the growth unit and capped at the plane.
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    fn wanted(&self, demand: engine::frame::Demand) -> Vec<u64> {
        let mut want = vec![0u64; self.planes.len()];
        let page_size = u64::from(self.paging.page_size);
        for shape in &self.shapes {
            match *shape {
                Shape::Kv {
                    head_dim,
                    kv_heads,
                    dtype,
                    keys,
                    values,
                    values_width,
                    ..
                } => {
                    let element = u64::from(elem_size(dtype));
                    let cells = u64::from(demand.kv_pages) * page_size;
                    let key_bytes = cells * u64::from(kv_heads) * u64::from(head_dim) * element;
                    want[keys] = want[keys].max(key_bytes);
                    let value_bytes = cells * values_width * element;
                    want[values] = want[values].max(value_bytes);
                }
                Shape::State {
                    stride,
                    dtype,
                    plane,
                } => {
                    let bytes =
                        u64::from(demand.state_slots) * stride * u64::from(elem_size(dtype));
                    want[plane] = want[plane].max(bytes);
                }
            }
        }
        want.iter_mut()
            .zip(&self.planes)
            .for_each(|(want, plane)| *want = elastic::pages_up(*want).min(plane.bytes()));
        want
    }

    /// Map memory under every elastic plane up to what `demand` needs.
    /// Nothing to do for a fixed pool.
    fn grow(&mut self, demand: engine::frame::Demand) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if self.arena.is_none() {
                return Ok(());
            }
            let want = self.wanted(demand);
            let mut targets: Vec<elastic::Target<'_>> = self
                .planes
                .iter_mut()
                .zip(want)
                .filter_map(|(plane, bytes)| match plane {
                    Plane::Elastic { store, .. } => Some(elastic::Target {
                        buffer: store,
                        bytes,
                    }),
                    Plane::Fixed(_) => None,
                })
                .collect();
            elastic::grow_all(&mut targets)?;
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = demand;
            Ok(())
        }
    }

    /// Give back the memory above what the watermark needs.
    ///
    /// The watermark is what `trim` has lowered it to; this is the half of
    /// a trim that touches the device, kept apart from `Supply::trim`
    /// because it has a precondition that trait cannot state: the caller
    /// has drained every frame that could read the pages released. A fixed
    /// pool releases nothing and answers `Ok`.
    pub fn release(&mut self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if self.arena.is_none() {
                return Ok(());
            }
            let want = self.wanted(self.watermark);
            let mut targets: Vec<elastic::Target<'_>> = self
                .planes
                .iter_mut()
                .zip(want)
                .filter_map(|(plane, bytes)| match plane {
                    Plane::Elastic { store, .. } => Some(elastic::Target {
                        buffer: store,
                        bytes,
                    }),
                    Plane::Fixed(_) => None,
                })
                .collect();
            // SAFETY: the caller's contract, stated above — no frame in
            // flight reads these planes.
            unsafe { elastic::shrink_all(&mut targets) }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(())
        }
    }
}

impl engine::frame::Supply for Pools {
    type Error = Fault;

    fn commit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        if self.has_state() && demand.state_slots > self.paging.slots {
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
        let union = self.watermark.union(demand);
        if union != self.watermark {
            self.grow(union)?;
            self.watermark = union;
        }
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
                head_dim,
                ..
            } => {
                let planes = split(name, planes)?;
                let cells = paging.pages() * u64::from(paging.page_size);
                // The demand estimate runs before consumers state a seat, but the
                // trace row now carries the cache's head_dim, so a packed KvU4
                // cache is sized exactly per head at ANY head_dim (matching what
                // `reserve` computes from the seat), not from the 256 anchor. A
                // row from a trace serialized before this field carries head_dim
                // 0, which `row_stride` treats as "unknown" and falls back to the
                // format anchor — the old behavior.
                let block = (*head_dim != 0).then_some(*head_dim);
                let keys = cells * row_stride(name, *dtype, planes.keys, block)?;
                let values = if planes.shared || planes.values == 0 {
                    0
                } else {
                    cells * row_stride(name, *dtype, planes.values, block)?
                };
                bytes = bytes.saturating_add(keys.saturating_add(values));
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

fn slot_file(error: std::io::Error) -> Fault {
    Fault::Device {
        call: "kv slot file",
        why: error.to_string(),
    }
}

fn elem_bytes(name: &str, dtype: Dtype) -> Result<u64> {
    poem_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Unbound {
        what: format!("cache `{name}`, stored as {dtype:?}, which has no element size"),
    })
}

/// Bytes one cache row occupies for `width` elements of `dtype`.
///
/// A scalar dtype is `width * elem_bytes`. A PACKED dtype has no per-element byte
/// size — its row is sized from the block layout.
///
/// The `KvU4` codec's block IS the model's `head_dim` (rotation-alignment) — pie
/// serves many head_dims, not just 256 — so its packed size is head_dim-driven,
/// NOT a baked 256-group: a row of `heads` whole heads packs to
/// `heads * (head_dim/2 + 2)` bytes (130 per head at head_dim 256, 66 at 128).
/// When `head_dim` is known (the allocation and migration paths have it), the
/// KvU4 row is sized from it. When it is not (the pre-facts demand estimate has
/// only the trace), the format's own `row_bytes` gives the head_dim-256 anchor
/// (`repr().row_bytes(256) == 130`), which is exact for the head_dim-256 flagship
/// — the only KvU4 geometry that ships today.
///
/// Any other packed dtype (block-quantized weight caches) has no head_dim and is
/// sized from its format's `row_bytes` as before. This is the ONE place packed
/// vs. scalar sizing forks — reserve, migration copy and demand accounting all
/// route through it so a packed plane is contiguous, whole-slot cells that a
/// byte-blit can move.
fn row_stride(name: &str, dtype: Dtype, width: u64, head_dim: Option<u32>) -> Result<u64> {
    if let Some(element) = poem_compiler::arena::elem_bytes(dtype) {
        return Ok(width * element);
    }
    // KvU4 sizes per head_dim-block when the head geometry is known.
    let kv_block = (dtype == Dtype::KvU4)
        .then(|| head_dim.filter(|&d| d > 0))
        .flatten();
    if let Some(block) = kv_block {
        let block = u64::from(block);
        if !width.is_multiple_of(block) {
            return Err(Fault::Unbound {
                what: format!(
                    "cache `{name}`, whose {width}-wide packed KV row is not a whole number of \
                     {block}-wide heads"
                ),
            });
        }
        let heads = width / block;
        return Ok(heads * (block / 2 + 2));
    }
    let k = u32::try_from(width).map_err(|_| Fault::Unbound {
        what: format!("cache `{name}`, whose {width}-wide row overflows a packed-plane width"),
    })?;
    dtype.row_bytes(k).ok_or_else(|| Fault::Unbound {
        what: format!(
            "cache `{name}`, stored as {dtype:?}, whose {width}-wide row is not a whole number of \
             the format's {}-element blocks",
            dtype.quantum().elems(k)
        ),
    })
}

fn elem_size(dtype: Dtype) -> u32 {
    poem_compiler::arena::elem_bytes(dtype).unwrap_or(1) as u32
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_pools(device: &Context, slots: u32, stride: u64) -> Pools {
        let bytes = stride * u64::from(slots) * u64::from(elem_size(STATE_DTYPE));
        Pools {
            planes: vec![Plane::Fixed(
                Buffer::zeroed(device, bytes).expect("a state slab"),
            )],
            shapes: vec![Shape::State {
                stride,
                dtype: STATE_DTYPE,
                plane: 0,
            }],
            paging: paging(slots),
            watermark: engine::frame::Demand::ZERO,
            disk: None,
            arena: None,
        }
    }

    fn paging(slots: u32) -> Paging {
        Paging {
            page_size: 16,
            pages_per_slot: 1,
            slots,
            pages: u64::from(slots),
            window: None,
        }
    }

    fn device() -> Option<Context> {
        if !crate::device::present() {
            eprintln!("no Metal device on this machine; skipping");
            return None;
        }
        Some(Context::bind().expect("the system device"))
    }

    /// A two-plane kv pool of four pages, fixed or elastic.
    fn kv_pools(device: &Context, backing: Backing) -> Pools {
        let paging = Paging {
            pages: 4,
            ..paging(1)
        };
        let cell = 3 * 4u64;
        let plane = paging.pages() * u64::from(paging.page_size) * cell;
        let shapes = vec![Shape::Kv {
            space: 0,
            head_dim: 3,
            kv_heads: 1,
            dtype: STATE_DTYPE,
            keys: 0,
            values: 1,
            values_width: 3,
        }];
        let mut planes = Vec::new();
        #[cfg_attr(not(target_vendor = "apple"), allow(unused_mut))]
        let mut arena = None;
        match backing {
            Backing::Fixed => {
                for _ in 0..2 {
                    planes.push(Plane::Fixed(
                        Buffer::zeroed(device, plane).expect("a kv slab"),
                    ));
                }
            }
            Backing::Elastic => {
                #[cfg(target_vendor = "apple")]
                {
                    let sparse = device
                        .sparse()
                        .expect("the mapping side opens")
                        .expect("a Metal 4 device");
                    let budget = elastic::Arena::new(2 * elastic::pages_up(plane));
                    for _ in 0..2 {
                        let store =
                            elastic::create(&sparse, &budget, plane).expect("a sparse plane");
                        planes.push(Plane::Elastic {
                            view: store.view(),
                            store,
                        });
                    }
                    arena = Some(budget);
                }
                #[cfg(not(target_vendor = "apple"))]
                unreachable!("no sparse buffers off Apple");
            }
        }
        Pools {
            planes,
            shapes,
            paging,
            watermark: engine::frame::Demand::ZERO,
            disk: None,
            arena,
        }
    }

    #[test]
    fn store_every_case() {
        a_copied_slot_reads_back_as_its_source();
        a_slot_past_the_pool_is_a_ceiling();
        an_attention_only_plan_answers_ok();
        kv_pages_come_back_from_disk_where_they_are_restored(Backing::Fixed);
        the_wanted_bytes_follow_the_demand_in_growth_units();
        #[cfg(target_vendor = "apple")]
        {
            kv_pages_come_back_from_disk_where_they_are_restored(Backing::Elastic);
            an_elastic_pool_maps_only_what_a_frame_commits();
        }
    }

    fn the_wanted_bytes_follow_the_demand_in_growth_units() {
        let Some(device) = device() else { return };
        let pools = kv_pools(&device, Backing::Fixed);
        let plane = pools.planes[0].bytes();
        assert_eq!(pools.wanted(engine::frame::Demand::ZERO), vec![0, 0]);
        let one = pools.wanted(engine::frame::Demand {
            kv_pages: 1,
            state_slots: 0,
            workspace: 0,
        });
        assert_eq!(
            one,
            vec![plane, plane],
            "one page of a plane smaller than the growth unit maps the whole plane"
        );
        let too_many = pools.wanted(engine::frame::Demand {
            kv_pages: 400,
            state_slots: 0,
            workspace: 0,
        });
        assert_eq!(too_many, vec![plane, plane], "capped at the plane");
    }

    #[cfg(target_vendor = "apple")]
    fn an_elastic_pool_maps_only_what_a_frame_commits() {
        let Some(device) = device() else { return };
        if !device.supports_elastic() {
            eprintln!("not a Metal 4 device; skipping");
            return;
        }
        let mut pools = kv_pools(&device, Backing::Elastic);
        assert_eq!(pools.committed_bytes(), 0, "creating the pool maps nothing");
        assert!(pools.is_elastic());
        assert_eq!(pools.elastic_page_bytes(), elastic::PAGE);
        let plane = pools.planes[0].bytes();
        let address = match &pools.planes[0] {
            Plane::Elastic { store, .. } => store.gpu_address(),
            Plane::Fixed(_) => unreachable!(),
        };

        engine::frame::Supply::commit(
            &mut pools,
            engine::frame::Demand {
                kv_pages: 2,
                state_slots: 0,
                workspace: 0,
            },
        )
        .expect("two pages fit");
        assert!(plane < elastic::TILE, "the test planes fit inside one tile");
        assert_eq!(
            pools.committed_bytes(),
            2 * elastic::TILE,
            "a plane smaller than one tile is mapped whole, and what is mapped is a \
             whole tile"
        );
        let mut back = vec![0xffu8; plane as usize];
        pools.planes[0].read(0, &mut back).expect("the plane reads");
        assert!(
            back.iter().all(|&b| b == 0),
            "freshly mapped memory is zeroed before a frame can read it"
        );
        match &pools.planes[0] {
            Plane::Elastic { store, .. } => assert_eq!(store.gpu_address(), address),
            Plane::Fixed(_) => unreachable!(),
        }

        // A blit through the sparse buffer lands where the host alias reads.
        let src: Vec<u8> = (0..plane).map(|at| (at % 251) as u8).collect();
        pools.planes[0].write(0, &src).expect("the plane writes");
        let mut frame = device.frame().expect("a frame");
        pools
            .copy_kv(
                &mut frame,
                &[Move {
                    src_page: 0,
                    src_token: 0,
                    dst_page: 1,
                    dst_token: 0,
                    tokens: 16,
                }],
            )
            .expect("a page copy");
        frame.commit().expect("the blit lands");
        let page = (plane / 4) as usize;
        pools.planes[0].read(0, &mut back).expect("the plane reads");
        assert_eq!(
            &back[page..2 * page],
            &src[..page],
            "the blit wrote page 1 from page 0"
        );

        // Releasing to a lower watermark hands the memory back.
        engine::frame::Supply::trim(&mut pools, engine::frame::Demand::ZERO);
        pools.release().expect("released");
        assert_eq!(pools.committed_bytes(), 0);
        assert_eq!(pools.high_water_bytes(), 2 * elastic::TILE);
        assert_eq!(
            pools.arena.as_ref().map(|arena| arena.budget().committed),
            Some(0)
        );
    }

    fn kv_pages_come_back_from_disk_where_they_are_restored(backing: Backing) {
        let Some(device) = device() else { return };
        if backing == Backing::Elastic && !device.supports_elastic() {
            eprintln!("not a Metal 4 device; skipping");
            return;
        }
        let mut pools = kv_pools(&device, backing);
        let plane = pools.planes[0].bytes();
        let bytes: Vec<u8> = (0..2 * plane).map(|at| (at % 251) as u8).collect();
        engine::frame::Supply::commit(
            &mut pools,
            engine::frame::Demand {
                kv_pages: 4,
                state_slots: 0,
                workspace: 0,
            },
        )
        .expect("the whole pool fits");
        pools.planes[0]
            .write(0, &bytes[..plane as usize])
            .expect("the keys written");
        pools.planes[1]
            .write(0, &bytes[plane as usize..])
            .expect("the values written");
        let dir =
            std::env::temp_dir().join(format!("pie-metal-kv-{}-{backing:?}", std::process::id()));
        assert_eq!(pools.seat_disk(&dir, 1 << 20).expect("a slot file"), 256);
        pools.spill_kv(true, &[1, 3], &[7, 8]).expect("suspended");
        pools.planes[0].zero(0, plane).expect("the keys cleared");
        pools.planes[1].zero(0, plane).expect("the values cleared");
        pools.spill_kv(false, &[2, 0], &[7, 8]).expect("restored");
        let page = plane / 4;
        let mut back = vec![0u8; bytes.len()];
        pools.planes[0]
            .read(0, &mut back[..plane as usize])
            .expect("the keys read");
        pools.planes[1]
            .read(0, &mut back[plane as usize..])
            .expect("the values read");
        for base in [0, plane] {
            let at = |page_id: u64| {
                (base + page_id * page) as usize..(base + (page_id + 1) * page) as usize
            };
            assert_eq!(back[at(2)], bytes[at(1)]);
            assert_eq!(back[at(0)], bytes[at(3)]);
            assert!(back[at(1)].iter().all(|&b| b == 0));
        }
        drop(pools);
        std::fs::remove_dir_all(dir).expect("the slot file removed");
    }

    fn a_copied_slot_reads_back_as_its_source() {
        let Some(device) = device() else { return };
        const STRIDE: u64 = 8;
        let mut pools = state_pools(&device, 4, STRIDE);
        let bytes = STRIDE * u64::from(elem_size(STATE_DTYPE));
        let src: Vec<u8> = (0..bytes)
            .map(|at| (at as u8).wrapping_mul(7).wrapping_add(1))
            .collect();
        pools.planes[0].write(bytes, &src).expect("seat 1 written");
        let bystander: Vec<u8> = vec![0xAB; bytes as usize];
        pools.planes[0]
            .write(2 * bytes, &bystander)
            .expect("seat 2 written");

        let mut frame = device.frame().expect("a frame");
        pools
            .copy_state(&mut frame, &[(1, 3), (0, 0)])
            .expect("a whole-slot copy");
        frame.commit().expect("the blit lands");

        assert_eq!(pools.read_slot(3).expect("seat 3"), src);
        assert_eq!(pools.read_slot(1).expect("seat 1"), src);
        assert_eq!(pools.read_slot(2).expect("seat 2"), bystander);
        assert_eq!(
            pools.read_slot(0).expect("seat 0"),
            vec![0u8; bytes as usize]
        );
        assert_eq!(pools.watermark().state_slots, 4);
    }

    fn a_slot_past_the_pool_is_a_ceiling() {
        let Some(device) = device() else { return };
        let mut pools = state_pools(&device, 2, 4);
        for moves in [&[(0u32, 2u32)][..], &[(5, 1)][..]] {
            let mut frame = device.frame().expect("a frame");
            let refused = pools
                .copy_state(&mut frame, moves)
                .expect_err("past the pool");
            assert!(
                matches!(
                    refused,
                    Fault::Ceiling {
                        what: "recurrent slots",
                        have: 2,
                        ..
                    }
                ),
                "{refused:?}"
            );
        }
    }

    fn an_attention_only_plan_answers_ok() {
        let Some(device) = device() else { return };
        let mut pools = Pools {
            planes: Vec::new(),
            shapes: Vec::new(),
            paging: paging(2),
            watermark: engine::frame::Demand::ZERO,
            disk: None,
            arena: None,
        };
        let mut frame = device.frame().expect("a frame");
        pools
            .copy_state(&mut frame, &[(0, 1), (7, 9)])
            .expect("nothing to move, nothing to refuse");
        assert_eq!(pools.watermark().state_slots, 0);
    }
}
