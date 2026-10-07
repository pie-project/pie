//! The kv and recurrent-state pools: one device buffer per plane, donated to
//! every program that writes it and replaced by what that program returns.
//!
//! Each kv plane carries one page past the pool (`Paging::pages()`), and each
//! state slab one slot past the slots: the *sink* page and slot a fire's
//! padding lanes read and write, so padding never touches a live row.
//!
//! A kv row declared with a window (when the context outruns it) lives in the
//! *windowed* pool instead: `Paging::window_pages()` pages, page 0 the null
//! page every page behind a window reads (zero, never written), and its own
//! sink page past them. Its space reads through the lanes' window tables
//! (`kv::window_table`), as long as their full tables, so positions and
//! masks read the same and the sliding-window rule masks the null pages.

pub mod kv;

use engine::transfer::KvCopy;
use kernels_xla::hlo::Elem;
use kernels_xla::{KvPool, RecurrentPool, Tensor};
use poem_ir::{CacheRow, Dtype, Trace};

/// Source and destination rows of a pool copy.
type Moves = (Vec<i64>, Vec<i64>);

use crate::device::Device;
use crate::error::{Fault, Result};
use crate::pjrt::Buffer;
use crate::store::kv::{Facts, Paging};
use kernels_xla::Emit;

use crate::trace::{Handles, Root, Source, Tracer};

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
        /// The values plane is the keys plane (a latent cache).
        shared: bool,
        /// The row lives in the windowed pool.
        windowed: bool,
    },
    State {
        stride: u64,
        dtype: Dtype,
    },
}

/// A fire's page geometry for one kv space, as handles.
#[derive(Debug, Clone, Copy)]
pub struct SpaceSeat {
    pub page_indptr: Tensor,
    pub page_indices: Tensor,
    pub max_pages: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum CachePool {
    Kv(KvPool),
    Recurrent(RecurrentPool),
}

#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>);

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
            moves.push(Move {
                src_page: cell.src_page_id,
                src_token: cell.src_token_offset,
                dst_page: cell.dst_page_id,
                dst_token: cell.dst_token_offset,
                tokens: 1,
            });
        }
        Ok(moves)
    }
}

pub struct Pools {
    /// Per cache row: `[keys, values]` for kv (values absent when shared),
    /// `[state]` for recurrent state. `None` only while a program holds it.
    planes: Vec<Vec<Option<Buffer>>>,
    /// The handle over each plane, minted at load.
    roots: Vec<Vec<Tensor>>,
    shapes: Vec<Shape>,
    /// The compressed-pool state planes (`[kv, score]`, one cell per kv
    /// cell) of each space a `PoolGather` folds into; their buffers sit in
    /// `planes` past the cache rows.
    compressor: Vec<(u32, [Tensor; 2])>,
    /// The recurrent-verb buffer plane of each rs layout plane.
    rs_buffers: Vec<Tensor>,
    paging: Paging,
    watermark: engine::frame::Demand,
    bytes: u64,
}

impl std::fmt::Debug for Pools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pools")
            .field("rows", &self.shapes.len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// Whether a kv row declared with `window` lives in the windowed pool.
fn windowed_row(paging: Paging, window: Option<u32>) -> bool {
    window.is_some() && paging.window.is_some()
}

/// The pages a kv plane holds, its sink page included.
fn plane_pages(paging: Paging, windowed: bool) -> u64 {
    if windowed {
        paging.window_pages() + 1
    } else {
        paging.pages() + 1
    }
}

/// The widest window a windowed kv row is declared with (engine-cuda
/// `store::window_of`). Every read of such a row must look through a window
/// no wider, since the pages behind it read the null page.
pub fn window_of(trace: &Trace) -> Result<Option<u32>> {
    for node in &trace.nodes {
        let Some(read) = kv::reads(&node.op) else {
            continue;
        };
        let Some(poem_ir::Def::Cache(row)) =
            trace.values.get(read.cache.0 as usize).map(|v| &v.def)
        else {
            continue;
        };
        if let Some(CacheRow::Kv {
            name,
            window: Some(held),
            ..
        }) = trace.caches.get(*row as usize)
            && read.window.is_none_or(|read| read > *held)
        {
            return Err(Fault::Unbound {
                what: format!(
                    "cache `{name}`, which holds a window of {held} tokens while `{}` reads \
                     it through {:?}",
                    poem_ir::Operands::name(&node.op),
                    read.window
                ),
            });
        }
    }
    Ok(trace
        .caches
        .iter()
        .filter_map(|row| match row {
            CacheRow::Kv { window, .. } => *window,
            CacheRow::State { .. } => None,
        })
        .max())
}

/// Bytes the pools of `trace` take at `paging`, sink page and slot included.
pub fn pool_demand(trace: &Trace, paging: Paging) -> Result<u64> {
    let mut total = 0u64;
    for row in &trace.caches {
        total += match row {
            CacheRow::Kv {
                name,
                planes,
                dtype,
                window,
                ..
            } => {
                let planes = split(name, planes)?;
                let cells = plane_pages(paging, windowed_row(paging, *window))
                    * u64::from(paging.page_size);
                let values = if planes.shared || planes.values == 0 {
                    0
                } else {
                    planes.values
                };
                cells * (planes.keys + values) * elem_bytes(name, *dtype)?
            }
            CacheRow::State { name, slab, dtype } => {
                let stride: u64 = slab.iter().product();
                stride * u64::from(paging.slots + 1) * elem_bytes(name, state_dtype(*dtype))?
            }
        };
    }
    Ok(total)
}

impl Pools {
    pub fn reserve(
        device: &Device,
        handles: &Handles,
        trace: &Trace,
        paging: Paging,
        facts: &Facts,
        rs: Option<&crate::rs::Layout>,
    ) -> Result<Pools> {
        let mut planes = Vec::with_capacity(trace.caches.len());
        let mut roots = Vec::with_capacity(trace.caches.len());
        let mut shapes = Vec::with_capacity(trace.caches.len());
        let mut compressor = Vec::new();
        let mut bytes = 0u64;
        let cells =
            u32::try_from((paging.pages() + 1) * u64::from(paging.page_size)).map_err(|_| {
                Fault::Ceiling {
                    what: "kv cells in one plane",
                    need: (paging.pages() + 1) * u64::from(paging.page_size),
                    have: u64::from(u32::MAX),
                }
            })?;
        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes: declared,
                    dtype,
                    space,
                    window,
                    ..
                } => {
                    let split = split(name, declared)?;
                    let windowed = windowed_row(paging, *window);
                    let cells = narrow(plane_pages(paging, windowed) * u64::from(paging.page_size));
                    let restated = facts
                        .rows
                        .get(index)
                        .copied()
                        .flatten()
                        .filter(|seat| seat.kv_heads != 0);
                    if let Some(seat) = restated {
                        let heads = u64::from(seat.kv_heads) * u64::from(seat.head_dim);
                        if heads != split.keys {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, whose row is {} wide while its \
                                     consumers state {} heads of {}",
                                    split.keys, seat.kv_heads, seat.head_dim
                                ),
                            });
                        }
                    }
                    let head_dim = restated.map_or(split.keys, |seat| u64::from(seat.head_dim));
                    let kv_heads = restated.map_or(1, |seat| u64::from(seat.kv_heads));
                    let shared = split.shared || split.values == 0;
                    let mut row_planes =
                        vec![device.zeros_or_dry(*dtype, cells, narrow(split.keys))?];
                    let keys = handles.root(Root {
                        source: Source::Pool {
                            row: index as u32,
                            plane: 0,
                        },
                        dtype: *dtype,
                        rows: cells,
                        width: narrow(split.keys),
                    });
                    let mut row_roots = vec![keys];
                    bytes += u64::from(cells) * split.keys * elem_bytes(name, *dtype)?;
                    if !shared {
                        row_planes.push(device.zeros_or_dry(
                            *dtype,
                            cells,
                            narrow(split.values),
                        )?);
                        row_roots.push(handles.root(Root {
                            source: Source::Pool {
                                row: index as u32,
                                plane: 1,
                            },
                            dtype: *dtype,
                            rows: cells,
                            width: narrow(split.values),
                        }));
                        bytes += u64::from(cells) * split.values * elem_bytes(name, *dtype)?;
                    }
                    planes.push(row_planes);
                    roots.push(row_roots);
                    shapes.push(Shape::Kv {
                        space: *space,
                        head_dim: narrow(head_dim),
                        kv_heads: narrow(kv_heads),
                        shared,
                        windowed,
                    });
                }
                CacheRow::State { name, slab, dtype } => {
                    let stride: u64 = slab.iter().product();
                    let dtype = state_dtype(*dtype);
                    let slots = paging.slots + 1;
                    planes.push(vec![device.zeros_or_dry(dtype, slots, narrow(stride))?]);
                    roots.push(vec![handles.root(Root {
                        source: Source::Pool {
                            row: index as u32,
                            plane: 0,
                        },
                        dtype,
                        rows: slots,
                        width: narrow(stride),
                    })]);
                    bytes += stride * u64::from(slots) * elem_bytes(name, dtype)?;
                    shapes.push(Shape::State { stride, dtype });
                }
            }
        }
        for (space, width) in compressor_spaces(trace) {
            let row = planes.len() as u32;
            let mut pair = Vec::with_capacity(2);
            let mut bufs = Vec::with_capacity(2);
            for plane in 0..2u8 {
                bufs.push(device.zeros_or_dry(Dtype::Bf16, cells, narrow(width))?);
                pair.push(handles.root(Root {
                    source: Source::Pool { row, plane },
                    dtype: Dtype::Bf16,
                    rows: cells,
                    width: narrow(width),
                }));
                bytes += u64::from(cells) * width * 2;
            }
            planes.push(bufs);
            roots.push(pair.clone());
            compressor.push((space, [pair[0], pair[1]]));
        }
        let mut rs_buffers = Vec::new();
        if let Some(layout) = rs {
            let rows = paging.slots.max(1) * paging.page_size.max(1);
            for plane in &layout.planes {
                let row = planes.len() as u32;
                planes.push(vec![device.zeros_or_dry(plane.dtype, rows, plane.width)?]);
                let t = handles.root(Root {
                    source: Source::Pool { row, plane: 0 },
                    dtype: plane.dtype,
                    rows,
                    width: plane.width,
                });
                roots.push(vec![t]);
                rs_buffers.push(t);
                bytes += u64::from(rows) * plane.row_bytes;
            }
        }
        Ok(Pools {
            planes,
            roots,
            shapes,
            compressor,
            rs_buffers,
            paging,
            watermark: engine::frame::Demand::ZERO,
            bytes,
        })
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// The page a padding lane's rows read and write.
    #[must_use]
    pub fn sink_page(&self) -> u32 {
        u32::try_from(self.paging.pages()).unwrap_or(u32::MAX)
    }

    /// The windowed pool's sink page: one past its pages.
    #[must_use]
    pub fn window_sink_page(&self) -> u32 {
        u32::try_from(self.paging.window_pages()).unwrap_or(u32::MAX)
    }

    /// Whether kv space `space` lives in the windowed pool.
    #[must_use]
    pub fn windowed_space(&self, space: u32) -> bool {
        self.shapes.iter().any(
            |shape| matches!(shape, Shape::Kv { space: at, windowed: true, .. } if *at == space),
        )
    }

    #[must_use]
    pub fn has_windowed(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::Kv { windowed: true, .. }))
    }

    /// The slot a padding lane's recurrent rows read and write.
    #[must_use]
    pub fn sink_slot(&self) -> u32 {
        self.paging.slots
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn watermark(&self) -> engine::frame::Demand {
        self.watermark
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::State { .. }))
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|shape| match shape {
                Shape::State { stride, dtype } => {
                    stride * poem_compiler::arena::elem_bytes(*dtype).unwrap_or(4)
                }
                Shape::Kv { .. } => 0,
            })
            .sum()
    }

    /// Takes plane `plane` of cache row `row` out for a program to donate.
    pub fn take(&mut self, row: u32, plane: u8) -> Result<Buffer> {
        self.planes
            .get_mut(row as usize)
            .and_then(|p| p.get_mut(plane as usize))
            .and_then(Option::take)
            .ok_or_else(|| Fault::Unbound {
                what: format!("pool plane {row}.{plane}, which a program still holds"),
            })
    }

    #[must_use]
    pub fn get(&self, row: u32, plane: u8) -> Option<&Buffer> {
        self.planes.get(row as usize)?.get(plane as usize)?.as_ref()
    }

    /// Puts back what a program returned for plane `plane` of row `row`.
    pub fn put(&mut self, row: u32, plane: u8, buffer: Buffer) {
        if let Some(slot) = self
            .planes
            .get_mut(row as usize)
            .and_then(|p| p.get_mut(plane as usize))
        {
            *slot = Some(buffer);
        }
    }

    /// The pools as kernel handles, with this fire's page geometry.
    pub fn table(&self, seats: &[SpaceSeat], slot_of_row: Tensor) -> Result<CacheTable> {
        let mut rows = Vec::with_capacity(self.shapes.len());
        for (roots, shape) in self.roots.iter().zip(&self.shapes) {
            rows.push(match *shape {
                Shape::Kv {
                    space,
                    head_dim,
                    kv_heads,
                    shared,
                    ..
                } => {
                    let seat = seats.get(space as usize).ok_or_else(|| Fault::Unbound {
                        what: format!("cache space {space}, for which this fire wrote no geometry"),
                    })?;
                    CachePool::Kv(KvPool {
                        keys: roots[0],
                        values: if shared { roots[0] } else { roots[1] },
                        page_indices: seat.page_indices,
                        page_indptr: seat.page_indptr,
                        page_size: i32::try_from(self.paging.page_size).unwrap_or(i32::MAX),
                        max_pages: seat.max_pages,
                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                    })
                }
                Shape::State { .. } => CachePool::Recurrent(RecurrentPool {
                    state: roots[0],
                    slots: slot_of_row,
                    conv_state: roots[0],
                    new_conv_state: roots[0],
                }),
            });
        }
        Ok(CacheTable(rows))
    }

    /// The recurrent-verb buffer planes, one per rs layout plane.
    #[must_use]
    pub fn rs_buffers(&self) -> Vec<Tensor> {
        self.rs_buffers.clone()
    }

    /// Every space's compressor planes, for a walk.
    #[must_use]
    pub fn compressors(&self) -> Vec<(u32, [Tensor; 2])> {
        self.compressor.clone()
    }

    /// The compressor state planes `[kv, score]` of kv space `space`.
    #[must_use]
    pub fn compressor(&self, space: u32) -> Option<[Tensor; 2]> {
        self.compressor
            .iter()
            .find(|(held, _)| *held == space)
            .map(|(_, pair)| *pair)
    }

    /// The state rows' handles, for maintenance programs.
    fn state_rows(&self) -> impl Iterator<Item = (u32, Tensor, u64)> + '_ {
        self.roots
            .iter()
            .zip(&self.shapes)
            .enumerate()
            .filter_map(|(row, (roots, shape))| match shape {
                Shape::State { stride, .. } => Some((row as u32, roots[0], *stride)),
                Shape::Kv { .. } => None,
            })
    }

    /// The kv planes' handles in the full (`windowed` false) or the
    /// windowed pool, for maintenance programs.
    fn kv_planes(&self, windowed: bool) -> impl Iterator<Item = Tensor> + '_ {
        self.roots
            .iter()
            .zip(&self.shapes)
            .filter(move |(_, shape)| {
                matches!(shape, Shape::Kv { windowed: held, .. } if *held == windowed)
            })
            .flat_map(|(roots, _)| roots.iter().copied())
    }

    fn check_slot(&self, slot: u32) -> Result<()> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        Ok(())
    }

    /// Zeros slot `slot` of every state slab.
    pub fn clear(&mut self, device: &Device, handles: &Handles, slot: u32) -> Result<()> {
        self.check_slot(slot)?;
        if !self.has_state() {
            return Ok(());
        }
        let tracer = Tracer::new(handles);
        let rows: Vec<Tensor> = self.state_rows().map(|r| r.1).collect();
        for &t in &rows {
            tracer.emit(&mut |cx| {
                let whole = cx.read(t)?;
                let elem = cx.elem(whole);
                let zero = cx.const_i(elem, 0, &[1, i64::from(t.width)]);
                let at = cx.const_i(Elem::I32, i64::from(slot), &[]);
                let col = cx.const_i(Elem::I32, 0, &[]);
                let next = cx.dynamic_update_slice(whole, zero, &[at, col])?;
                cx.write(t, next)
            })?;
        }
        crate::exec::maintain(device, tracer, self)
    }

    /// Copies slot `src` of every state slab over slot `dst`.
    pub fn copy_slot(
        &mut self,
        device: &Device,
        handles: &Handles,
        src: u32,
        dst: u32,
    ) -> Result<()> {
        self.check_slot(src)?;
        self.check_slot(dst)?;
        let tracer = Tracer::new(handles);
        let rows: Vec<Tensor> = self.state_rows().map(|r| r.1).collect();
        for &t in &rows {
            tracer.emit(&mut |cx| {
                let whole = cx.read(t)?;
                let row = cx.slice_axis(whole, 0, i64::from(src), i64::from(src) + 1)?;
                let at = cx.const_i(Elem::I32, i64::from(dst), &[]);
                let col = cx.const_i(Elem::I32, 0, &[]);
                let next = cx.dynamic_update_slice(whole, row, &[at, col])?;
                cx.write(t, next)
            })?;
        }
        crate::exec::maintain(device, tracer, self)
    }

    /// Applies kv `moves` to every full kv plane and `windowed` (whole
    /// pages, in the windowed pool's own ids) to every windowed one: per
    /// plane, one gather of the source cells, one scatter onto the
    /// destination cells.
    pub fn copy_kv(
        &mut self,
        device: &Device,
        handles: &Handles,
        moves: &[Move],
        windowed: &[Move],
    ) -> Result<()> {
        let page = self.paging.page_size;
        let cells = |moves: &[Move], pages: u64| -> Result<(Vec<i64>, Vec<i64>)> {
            let mut src: Vec<i64> = Vec::new();
            let mut dst: Vec<i64> = Vec::new();
            for m in moves {
                for (what, at) in [("source", m.src_page), ("destination", m.dst_page)] {
                    if u64::from(at) >= pages {
                        return Err(Fault::Ceiling {
                            what: if what == "source" {
                                "kv copy source pages"
                            } else {
                                "kv copy destination pages"
                            },
                            need: u64::from(at) + 1,
                            have: pages,
                        });
                    }
                }
                for t in 0..m.tokens {
                    src.push(i64::from(m.src_page) * i64::from(page) + i64::from(m.src_token + t));
                    dst.push(i64::from(m.dst_page) * i64::from(page) + i64::from(m.dst_token + t));
                }
            }
            Ok((src, dst))
        };
        let full = cells(moves, self.paging.pages())?;
        let window = if self.has_windowed() {
            cells(windowed, self.paging.window_pages())?
        } else {
            (Vec::new(), Vec::new())
        };
        if full.0.is_empty() && window.0.is_empty() {
            return Ok(());
        }
        let tracer = Tracer::new(handles);
        let mut planes: Vec<(Tensor, &Moves)> = Vec::new();
        if !full.0.is_empty() {
            planes.extend(self.kv_planes(false).map(|t| (t, &full)));
        }
        if !window.0.is_empty() {
            planes.extend(self.kv_planes(true).map(|t| (t, &window)));
        }
        for &(t, (src, dst)) in &planes {
            tracer.emit(&mut |cx| {
                let whole = cx.read(t)?;
                let n = src.len() as i64;
                let s = cx.const_ints(Elem::I32, src, &[n])?;
                let d = cx.const_ints(Elem::I32, dst, &[n])?;
                let rows = cx.take_rows(whole, s)?;
                let next = cx.put_rows(whole, d, rows, kernels_xla::hlo::Combine::Set)?;
                cx.write(t, next)
            })?;
        }
        crate::exec::maintain(device, tracer, self)
    }

    /// The bytes of slot `slot` across every state slab, in row order.
    pub fn read_slot(&self, slot: u32) -> Result<Vec<u8>> {
        self.check_slot(slot)?;
        let mut out = Vec::new();
        for (row, _, stride) in self.state_rows() {
            let Some(buffer) = self.get(row, 0) else {
                continue;
            };
            let all = buffer.download()?;
            let per = all.len() / (self.paging.slots as usize + 1);
            let _ = stride;
            out.extend_from_slice(&all[slot as usize * per..(slot as usize + 1) * per]);
        }
        Ok(out)
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
        if u64::from(demand.kv_pages) > self.paging.pages() {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(demand.kv_pages),
                have: self.paging.pages(),
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

/// Each kv space a `PoolGather` folds into, and the width of its
/// compressor state (`coff · head_dim`, coff 2 at ratio 4).
fn compressor_spaces(trace: &Trace) -> Vec<(u32, u64)> {
    let mut spaces: Vec<(u32, u64)> = Vec::new();
    for node in &trace.nodes {
        let poem_ir::Operation::Attention(poem_ir::Attention::PoolGather {
            pages,
            head_dim,
            ratio,
            ..
        }) = &node.op
        else {
            continue;
        };
        let Some(poem_ir::Def::Cache(space)) = trace.values.get(pages.0 as usize).map(|v| &v.def)
        else {
            continue;
        };
        let width = if *ratio == 4 { 2 } else { 1 } * u64::from(*head_dim);
        if width == 0 {
            continue;
        }
        match spaces.iter_mut().find(|(held, _)| *held == *space) {
            Some((_, held)) => *held = (*held).max(width),
            None => spaces.push((*space, width)),
        }
    }
    spaces
}

#[derive(Debug, Clone, Copy)]
struct Split {
    keys: u64,
    values: u64,
    shared: bool,
}

fn split(name: &str, planes: &[u64]) -> Result<Split> {
    match planes {
        [shared] => Ok(Split {
            keys: *shared,
            values: *shared,
            shared: true,
        }),
        [keys, values] => Ok(Split {
            keys: *keys,
            values: *values,
            shared: false,
        }),
        other => Err(Fault::Unbound {
            what: format!(
                "cache `{name}`, which declares {} plane(s) — this shell cuts kv pages into \
                 one shared plane or into a key half and a value half",
                other.len()
            ),
        }),
    }
}

fn elem_bytes(name: &str, dtype: Dtype) -> Result<u64> {
    poem_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Unbound {
        what: format!("cache `{name}` in {dtype:?}, which has no element size"),
    })
}

fn narrow(n: u64) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}
