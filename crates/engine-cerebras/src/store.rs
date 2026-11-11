//! The kv and recurrent state: a chunk pool on the host ([`pool`]) and the
//! planes one fire reads and writes ([`stage`]).
//!
//! Every slot a working set holds lives in the host pool, named by the slot
//! refs the runtime hands out (CONTRACT-BC 4, 18-21). A fire's program
//! reads typed planes, one host buffer a plane, donated to the program and
//! replaced by what it returns: before each fire the slots its lanes name
//! are copied into those planes under fire-local ids, and after it what the
//! fire wrote is copied back.
//!
//! Each kv plane carries one page past the staged ones, and each state slab
//! one record past them: the *sink* page and record a fire's padding lanes
//! read and write, so padding never touches a staged row.
//!
//! A kv row declared with a window (when the context outruns it) lives in
//! the *window list* instead: its planes' page 0 is the null page every
//! page behind a window reads (zero, never written), the staged pages
//! follow, and its own sink page past them. Its rows read through the
//! lanes' window tables, as long as their full tables, so positions and
//! masks read the same and the sliding-window rule masks the null pages.

pub mod kv;
pub mod pool;
pub mod stage;
pub mod tier;

use engine::transfer::KvRun;
use kernels_cerebras::{KvPool, RecurrentPool, Tensor};
use poem_ir::{CacheRow, Dtype, Trace};

pub use stage::{Capacity, Stage};

use crate::device::Buffer;
use crate::device::Device;
use crate::error::{Fault, Result};
use crate::store::kv::{Facts, Paging};

use crate::trace::{Handles, Root, Source};

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

/// A fire's page geometry for one page table, as handles.
#[derive(Debug, Clone, Copy)]
pub struct TableSeat {
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

pub struct Pools {
    /// Per cache row: `[keys, values]` for kv (values absent when shared),
    /// `[state]` for recurrent state. `None` only while a program holds it.
    planes: Vec<Vec<Option<Buffer>>>,
    /// The handle over each plane, minted at load.
    roots: Vec<Vec<Tensor>>,
    shapes: Vec<Shape>,
    /// The compressed-pool state planes (`[kv, score]`, one cell per kv
    /// cell) of each cache row a `PoolGather` folds into; their buffers sit in
    /// `planes` past the cache rows.
    compressor: Vec<(u32, [Tensor; 2])>,
    /// The recurrent-verb buffer plane of each rs layout plane.
    rs_buffers: Vec<Tensor>,
    /// The pools as the budget sized them (the legacy pool facts' counts).
    paging: Paging,
    /// What one fire stages: the planes' rows.
    capacity: Capacity,
    pool: pool::HostPool,
}

impl std::fmt::Debug for Pools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pools")
            .field("rows", &self.shapes.len())
            .field("capacity", &self.capacity)
            .field("pool", &self.pool.layout())
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

/// Bytes the pools of `trace` take at `paging` in their declared dtypes,
/// a sink page and slot included, compressor and recurrent-verb planes too:
/// the measure a kv budget sizes `paging` by. The host pool
/// ([`Pools::reserve`]) then holds every slot of that paging.
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
            CacheRow::State {
                name, slab, dtype, ..
            } => {
                let stride: u64 = slab.iter().product();
                stride * u64::from(paging.slots + 1) * elem_bytes(name, state_dtype(*dtype))?
            }
        };
    }
    let cells = plane_pages(paging, false) * u64::from(paging.page_size);
    for (_, width) in compressor_rows(trace) {
        // Its `[kv, score]` pair, both bf16.
        total += 2 * cells * width * 2;
    }
    if let Some(layout) = crate::rs::Layout::read(trace)? {
        total += rs_rows(paging) * layout.token_bytes;
    }
    Ok(total)
}

/// One page's bytes in the pools the degenerate facts state: a kv page
/// across the full pool's planes (compressor planes too), an rs buffer page
/// across the recurrent-verb planes.
pub fn page_bytes(trace: &Trace, paging: Paging) -> Result<engine::chunk::DegenerateBytes> {
    let page = u64::from(paging.page_size);
    let mut bytes = engine::chunk::DegenerateBytes::default();
    for row in &trace.caches {
        let CacheRow::Kv {
            name,
            planes,
            dtype,
            window,
            ..
        } = row
        else {
            continue;
        };
        let planes = split(name, planes)?;
        let values = if planes.shared || planes.values == 0 {
            0
        } else {
            planes.values
        };
        let row = page * (planes.keys + values) * elem_bytes(name, *dtype)?;
        if !windowed_row(paging, *window) {
            bytes.kv_page += row;
        }
    }
    for (_, width) in compressor_rows(trace) {
        bytes.kv_page += 2 * page * width * 2;
    }
    if let Some(layout) = crate::rs::Layout::read(trace)? {
        bytes.buffer_page = page * layout.token_bytes;
    }
    Ok(bytes)
}

/// The rows of a recurrent-verb buffer plane: a page of rows per slot.
fn rs_rows(paging: Paging) -> u64 {
    u64::from(paging.slots.max(1)) * u64::from(paging.page_size.max(1))
}

impl Pools {
    /// The staging planes of a fire of at most `lanes` lanes and the host
    /// pool behind them, which holds what `paging`'s pools held (its kv
    /// pages, windowed pages, state slots and rs pages) beside a
    /// copy-on-write record for each lane a fire runs.
    pub fn reserve(
        device: &Device,
        handles: &Handles,
        trace: &Trace,
        paging: Paging,
        lanes: u32,
        facts: &Facts,
        rs: Option<&crate::rs::Layout>,
    ) -> Result<Pools> {
        Pools::reserve_at(
            device,
            handles,
            (trace, paging, lanes),
            facts,
            rs,
            pool::TARGET_CHUNK_BYTES,
        )
    }

    /// [`Pools::reserve`] with chunks of up to `target` bytes.
    pub(crate) fn reserve_at(
        device: &Device,
        handles: &Handles,
        (trace, paging, lanes): (&Trace, Paging, u32),
        facts: &Facts,
        rs: Option<&crate::rs::Layout>,
        target: u64,
    ) -> Result<Pools> {
        let capacity = capacity_of(trace, paging, lanes, rs.is_some());
        let page = paging.page_size;
        let mut planes = Vec::with_capacity(trace.caches.len());
        let mut roots = Vec::with_capacity(trace.caches.len());
        let mut shapes = Vec::with_capacity(trace.caches.len());
        let mut compressor = Vec::new();
        let mut drafts = pool::Drafts::default();
        let cells = |pages: u32| -> Result<u32> {
            let need = (u64::from(pages) + 1) * u64::from(page);
            u32::try_from(need).map_err(|_| Fault::Ceiling {
                what: "kv cells in one plane",
                need,
                have: u64::from(u32::MAX),
            })
        };
        let list_draft = |slots: u64| pool::Draft {
            tokens: page,
            slots,
            ..pool::Draft::default()
        };
        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes: declared,
                    dtype,
                    window,
                    ..
                } => {
                    let split = split(name, declared)?;
                    let windowed = windowed_row(paging, *window);
                    let rows = if windowed {
                        cells(capacity.window_pages)?
                    } else {
                        cells(capacity.kv_pages)?
                    };
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
                    let draft = if windowed {
                        &mut drafts
                            .window
                            .get_or_insert_with(|| {
                                (
                                    list_draft(paging.window_pages().saturating_sub(1)),
                                    paging.window.map_or(0, |w| w.tokens),
                                )
                            })
                            .0
                    } else {
                        drafts
                            .list
                            .get_or_insert_with(|| list_draft(paging.pages()))
                    };
                    draft.rows.push(index as u32);
                    let widths = if shared {
                        vec![split.keys]
                    } else {
                        vec![split.keys, split.values]
                    };
                    let mut row_planes = Vec::with_capacity(widths.len());
                    let mut row_roots = Vec::with_capacity(widths.len());
                    for (plane, width) in widths.into_iter().enumerate() {
                        let (plane, width) = (plane as u8, narrow(width));
                        row_planes.push(device.zeros_or_dry(*dtype, rows, width)?);
                        row_roots.push(handles.root(Root {
                            source: Source::Pool {
                                row: index as u32,
                                plane,
                            },
                            dtype: *dtype,
                            rows,
                            width,
                        }));
                        draft.sections.push(pool::Section {
                            row: index as u32,
                            plane,
                            width,
                        });
                    }
                    planes.push(row_planes);
                    roots.push(row_roots);
                    shapes.push(Shape::Kv {
                        head_dim: narrow(head_dim),
                        kv_heads: narrow(kv_heads),
                        shared,
                        windowed,
                    });
                }
                CacheRow::State { slab, dtype, .. } => {
                    let stride: u64 = slab.iter().product();
                    let dtype = state_dtype(*dtype);
                    let slots = capacity.records + 1;
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
                    shapes.push(Shape::State { stride, dtype });
                    let draft = drafts.fixed.get_or_insert_with(|| pool::Draft {
                        tokens: 1,
                        slots: u64::from(paging.slots) + u64::from(capacity.records),
                        ..pool::Draft::default()
                    });
                    draft.rows.push(index as u32);
                    draft.sections.push(pool::Section {
                        row: index as u32,
                        plane: 0,
                        width: narrow(stride),
                    });
                }
            }
        }
        let kv_cells = cells(capacity.kv_pages)?;
        for (cache_row, width) in compressor_rows(trace) {
            let row = planes.len() as u32;
            let mut pair = Vec::with_capacity(2);
            let mut bufs = Vec::with_capacity(2);
            let draft = drafts
                .list
                .get_or_insert_with(|| list_draft(paging.pages()));
            for plane in 0..2u8 {
                bufs.push(device.zeros_or_dry(Dtype::Bf16, kv_cells, narrow(width))?);
                pair.push(handles.root(Root {
                    source: Source::Pool { row, plane },
                    dtype: Dtype::Bf16,
                    rows: kv_cells,
                    width: narrow(width),
                }));
                draft.sections.push(pool::Section {
                    row,
                    plane,
                    width: narrow(width),
                });
            }
            planes.push(bufs);
            roots.push(pair.clone());
            compressor.push((cache_row, [pair[0], pair[1]]));
        }
        let mut rs_buffers = Vec::new();
        if let Some(layout) = rs {
            let rows = narrow(u64::from(capacity.rs_pages) * u64::from(page));
            let mut draft = pool::Draft {
                tokens: page,
                // A page a state slot, and two a lane a fire runs.
                slots: u64::from(paging.slots.max(1)) + u64::from(capacity.rs_pages),
                ..pool::Draft::default()
            };
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
                if !draft.rows.contains(&plane.cache) {
                    draft.rows.push(plane.cache);
                }
                draft.sections.push(pool::Section {
                    row,
                    plane: 0,
                    width: plane.width,
                });
            }
            drafts.rs = Some(draft);
        }
        let layout = pool::Layout::plan(&drafts, target)?;
        Ok(Pools {
            planes,
            roots,
            shapes,
            compressor,
            rs_buffers,
            paging,
            capacity,
            pool: pool::HostPool::new(layout, device.is_dry()),
        })
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// What one fire stages at most.
    #[must_use]
    pub fn capacity(&self) -> Capacity {
        self.capacity
    }

    #[must_use]
    pub fn pool(&self) -> &pool::HostPool {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut pool::HostPool {
        &mut self.pool
    }

    /// The page a padding lane's rows read and write: past the staged ones.
    #[must_use]
    pub fn sink_page(&self) -> u32 {
        self.capacity.kv_pages
    }

    /// The windowed planes' sink page: past the staged ones.
    #[must_use]
    pub fn window_sink_page(&self) -> u32 {
        self.capacity.window_pages
    }

    /// The page tables a fire stages, by whether each pages the windowed
    /// pool: the full rows' first (the All axis's), then the windowed
    /// rows'. A model keeping only windowed rows reads its All axis off the
    /// windowed table, then the only one.
    #[must_use]
    pub fn tables(&self) -> Vec<bool> {
        let held = |windowed: bool| {
            self.shapes
                .iter()
                .any(|shape| matches!(shape, Shape::Kv { windowed: at, .. } if *at == windowed))
        };
        [false, true].into_iter().filter(|&w| held(w)).collect()
    }

    /// The page table of [`Pools::tables`] the rows in the windowed pool
    /// (`windowed`) or in the full one read.
    #[must_use]
    pub fn table_of(&self, windowed: bool) -> usize {
        let full = self.shapes.iter().any(|shape| {
            matches!(
                shape,
                Shape::Kv {
                    windowed: false,
                    ..
                }
            )
        });
        usize::from(windowed && full)
    }

    #[must_use]
    pub fn has_windowed(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::Kv { windowed: true, .. }))
    }

    /// The record a padding lane's recurrent rows read and write.
    #[must_use]
    pub fn sink_slot(&self) -> u32 {
        self.capacity.records
    }

    /// Bytes the host pool and the staging planes take.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.pool.layout().bytes() + self.staging_bytes()
    }

    /// Bytes the staging planes take, one word an element.
    #[must_use]
    pub fn staging_bytes(&self) -> u64 {
        self.roots
            .iter()
            .flatten()
            .map(|t| u64::from(t.rows) * u64::from(t.width) * 4)
            .sum()
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
    pub fn table(&self, seats: &[TableSeat], slot_of_row: Tensor) -> Result<CacheTable> {
        let mut rows = Vec::with_capacity(self.shapes.len());
        for (roots, shape) in self.roots.iter().zip(&self.shapes) {
            rows.push(match *shape {
                Shape::Kv {
                    windowed,
                    head_dim,
                    kv_heads,
                    shared,
                    ..
                } => {
                    let table = self.table_of(windowed);
                    let seat = seats.get(table).ok_or_else(|| Fault::Unbound {
                        what: format!("page table {table}, which this fire staged none of"),
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

    /// Every cache row's compressor planes, for a walk.
    #[must_use]
    pub fn compressors(&self) -> Vec<(u32, [Tensor; 2])> {
        self.compressor.clone()
    }

    /// The compressor state planes `[kv, score]` of cache row `row`.
    #[must_use]
    pub fn compressor(&self, row: u32) -> Option<[Tensor; 2]> {
        self.compressor
            .iter()
            .find(|(held, _)| *held == row)
            .map(|(_, pair)| *pair)
    }

    /// The rs buffer pages the legacy pool facts count: one a state slot.
    #[must_use]
    pub fn buffer_pages(&self) -> u32 {
        if self.rs_buffers.is_empty() {
            0
        } else {
            self.paging.slots.max(1)
        }
    }

    /// `n` slot refs of shape `id` from the first chunk past the null one,
    /// for fires whose contents nothing reads (the warm ladder's); none
    /// where the pool holds no such shape.
    #[must_use]
    pub fn warm_refs(&self, id: engine::chunk::ShapeId, n: u32) -> Vec<u32> {
        let Ok(shape) = self.pool.shape(id) else {
            return Vec::new();
        };
        let per = shape.per_chunk.max(1);
        let slots = per.saturating_mul(self.pool.layout().total_chunks.saturating_sub(1));
        (0..n).map(|i| per + i % slots.max(1)).collect()
    }

    /// A fire's staging, empty, at this load's capacity.
    #[must_use]
    pub fn begin(&self) -> Stage {
        Stage::new(self.capacity)
    }

    /// The shapes this pool holds, in id order.
    fn held(&self) -> Vec<engine::chunk::ShapeId> {
        self.pool.layout().shapes.iter().map(|s| s.id).collect()
    }

    /// Copies every slot `stage` names out of the host pool into its local
    /// rows of the staging planes (scratch rows and the null page zeroed).
    /// Nothing on a dry device.
    pub fn stage(&mut self, stage: &Stage) -> Result<()> {
        if self.pool.is_dry() {
            return Ok(());
        }
        for id in self.held() {
            let shape = self.pool.shape(id)?.clone();
            let sources = stage::sources(&self.pool, stage, id)?;
            for (section, first, words) in shape.sections() {
                let (first, words) = (first as usize, words as usize);
                edit(&mut self.planes, section.row, section.plane, |plane, _| {
                    for (local, source) in &sources {
                        let at = *local as usize * words;
                        let to = &mut plane[at..at + words];
                        match source {
                            Some(slot) => to.copy_from_slice(&slot[first..first + words]),
                            None => to.fill(0),
                        }
                    }
                })?;
            }
        }
        Ok(())
    }

    /// Copies what the fire left in the staged rows back into the host
    /// pool's slots: every staged slot but scratch rows, the null page and
    /// a record a lane only read.
    pub fn unstage(&mut self, stage: &Stage) -> Result<()> {
        if self.pool.is_dry() {
            return Ok(());
        }
        for id in self.held() {
            let shape = self.pool.shape(id)?.clone();
            let kept = stage::kept(stage, id);
            if kept.is_empty() {
                continue;
            }
            // Each kept unit's words: a list slot, or a whole record.
            let unit = usize::try_from(if id == pool::FIXED {
                shape.record_words
            } else {
                shape.slot_words
            })
            .unwrap_or(0);
            let mut units: Vec<Vec<u32>> = vec![vec![0; unit]; kept.len()];
            for (section, first, words) in shape.sections() {
                let (first, words) = (first as usize, words as usize);
                let plane = self
                    .get(section.row, section.plane)
                    .ok_or_else(|| Fault::Unbound {
                        what: format!(
                            "pool plane {}.{}, which a program still holds",
                            section.row, section.plane
                        ),
                    })?
                    .words();
                for ((local, _), out) in kept.iter().zip(&mut units) {
                    let at = *local as usize * words;
                    out[first..first + words].copy_from_slice(&plane[at..at + words]);
                }
            }
            let piece = usize::try_from(shape.slot_words).unwrap_or(0);
            for ((_, slots), words) in kept.iter().zip(units) {
                for (at, &slot) in slots.iter().enumerate() {
                    if let Some(to) = self.pool.slot_mut(id, slot)? {
                        let from = (at * piece).min(words.len());
                        let take = piece.min(words.len() - from);
                        to[..take].copy_from_slice(&words[from..from + take]);
                    }
                }
            }
        }
        Ok(())
    }

    /// Applies kv `moves` (token runs between slot refs of the kv list) and
    /// `windowed` (whole slots of the window list) in the host pool. Every
    /// source is read before anything is written: a slot may be both.
    pub fn copy_kv(&mut self, moves: &[KvRun], windowed: &[(u32, u32)]) -> Result<()> {
        if !moves.is_empty() {
            let shape = self.pool.shape(pool::LIST)?.clone();
            let mut sources = Vec::with_capacity(moves.len());
            for m in moves {
                if m.src_token.saturating_add(m.tokens) > shape.tokens
                    || m.dst_token.saturating_add(m.tokens) > shape.tokens
                {
                    return Err(Fault::Ceiling {
                        what: "rows of a kv page a copy reaches",
                        need: u64::from(m.src_token.max(m.dst_token)) + u64::from(m.tokens),
                        have: u64::from(shape.tokens),
                    });
                }
                self.pool.span(pool::LIST, m.dst_page)?;
                sources.push(self.pool.slot(pool::LIST, m.src_page)?.map(<[u32]>::to_vec));
            }
            for (m, source) in moves.iter().zip(sources) {
                let (Some(from), Some(to)) = (source, self.pool.slot_mut(pool::LIST, m.dst_page)?)
                else {
                    continue;
                };
                for (section, first, _) in shape.sections() {
                    let w = section.width as usize;
                    let s = first as usize + m.src_token as usize * w;
                    let d = first as usize + m.dst_token as usize * w;
                    let n = m.tokens as usize * w;
                    to[d..d + n].copy_from_slice(&from[s..s + n]);
                }
            }
        }
        if !windowed.is_empty() {
            self.pool.copy_slots(pool::WINDOW, windowed)?;
        }
        Ok(())
    }
}

/// What a fire of at most `lanes` lanes stages: a kv page for every page
/// its lanes may name (no more than the pool holds), a windowed page for
/// each beside the null page, a state record a lane (no more than the
/// slots the pool seats) and two rs pages a record.
#[must_use]
pub fn capacity_of(trace: &Trace, paging: Paging, lanes: u32, rs: bool) -> Capacity {
    let lane_pages = u64::from(lanes.max(1)) * u64::from(paging.pages_per_slot);
    let kv_pages = narrow(paging.pages().min(lane_pages).max(1));
    let states = trace
        .caches
        .iter()
        .any(|row| matches!(row, CacheRow::State { .. }));
    let records = if states {
        paging.slots.min(lanes).max(1)
    } else {
        0
    };
    Capacity {
        kv_pages,
        window_pages: if paging.window.is_some() {
            narrow(paging.window_pages().min(lane_pages).max(1)) + 1
        } else {
            1
        },
        records,
        rs_pages: if rs { 2 * records.max(1) } else { 0 },
    }
}

/// Edits plane `plane` of row `row` on the host: `f` gets its words and
/// row width. Nothing where the plane holds no buffer (a dry device).
fn edit(
    planes: &mut [Vec<Option<Buffer>>],
    row: u32,
    plane: u8,
    f: impl FnOnce(&mut [u32], usize),
) -> Result<()> {
    let slot = planes
        .get_mut(row as usize)
        .and_then(|p| p.get_mut(plane as usize))
        .ok_or_else(|| Fault::Unbound {
            what: format!("pool plane {row}.{plane}"),
        })?;
    let Some(buffer) = slot.take() else {
        return Ok(());
    };
    let (rows, width) = buffer.shape();
    let mut words = buffer.words().to_vec();
    f(&mut words, width as usize);
    *slot = Buffer::new(buffer.dtype(), rows, width, words);
    Ok(())
}

/// Each cache row a `PoolGather` folds into, and the width of its
/// compressor state (`coff · head_dim`, coff 2 at ratio 4).
fn compressor_rows(trace: &Trace) -> Vec<(u32, u64)> {
    let mut rows: Vec<(u32, u64)> = Vec::new();
    for node in &trace.nodes {
        let poem_ir::Operation::Attn(poem_ir::Attn::PoolGather {
            pages,
            head_dim,
            ratio,
            ..
        }) = &node.op
        else {
            continue;
        };
        let Some(poem_ir::Def::Cache(row)) = trace.values.get(pages.0 as usize).map(|v| &v.def)
        else {
            continue;
        };
        let width = if *ratio == 4 { 2 } else { 1 } * u64::from(*head_dim);
        if width == 0 {
            continue;
        }
        match rows.iter_mut().find(|(held, _)| *held == *row) {
            Some((_, held)) => *held = (*held).max(width),
            None => rows.push((*row, width)),
        }
    }
    rows
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
