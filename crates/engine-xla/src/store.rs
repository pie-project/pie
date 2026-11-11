//! The state pool: ONE device buffer of `chunks` chunks, fixed at load, that
//! every kind of held state shares — kv pages, window pages, recurrent
//! records, rs buffer pages — donated to every program that writes it and
//! replaced by what that program returns.
//!
//! The buffer is u32 `[chunks · C / 512, 128]` (`[chunks, rows, 128]`, its
//! major axes merged). The runtime hands out slots of the shapes
//! [`layout::Layout`] states (`ChunkFacts`), and every plane is read and
//! written through a view of the whole buffer at its own element and width,
//! a page or slot id resolved to the plane's block by its affine map
//! (`kernels_xla::tensor::PageMap`). A fire's tables carry grain indices
//! (`Layout::grain_of`), resolved here from the runtime's logical slot refs
//! through the engine's [`ChunkTable`].
//!
//! Chunk 0 is the null chunk a window list's trimmed entries read (zeros,
//! never written); chunk 1 is the sink a fire's padding lanes read and write.
//! The pool never grows or shrinks (XLA buffers are program parameters fixed
//! before anything traces); the table still moves chunks when the runtime
//! asks ([`Pools::evacuate`]), which it does not for a fixed pool.

pub mod kv;
pub mod layout;
pub mod streamed;
pub mod tier;

use crate::handle::{KvPool, RecurrentPool, Tensor};
use engine::chunk::{
    ChunkFacts, ChunkTable, Evacuation, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId,
    SlotKind, SlotRole, SlotShape,
};
use engine::transfer::KvRun;
use kernels_xla::hlo::Elem;
use poem_ir::{CacheRow, Dtype, Trace};

use crate::device::Device;
use crate::emit::Emit;
use crate::error::{Fault, Result};
use crate::pjrt::Buffer;
use crate::store::kv::{Facts, Paging};
use crate::store::layout::{Kind, Layout, Plane, ROW_BYTES, What};

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

const STATE_DTYPE: Dtype = Dtype::Bf16;

fn state_dtype(declared: Dtype) -> Dtype {
    match declared {
        Dtype::I32 => Dtype::I32,
        Dtype::Bf16 => Dtype::Bf16,
        _ => STATE_DTYPE,
    }
}

/// The null chunk; the sink is the pool's top chunk ([`Pools::sink_chunk`]):
/// every kernel that treats a plane view's last page or block as the sink
/// (the prefill page writer drops rows bound there) then finds it in a
/// chunk the runtime never hands out.
const NULL_CHUNK: u32 = 0;
const RESERVED_CHUNKS: u32 = 2;

/// Lone weight slots one program lands at most.
const LONE_SLOTS: usize = 64;

/// A plane's affine map: page or slot id `p` is the plane's block
/// `p · stride + offset` of its view.
pub type Map = kernels_xla::tensor::PageMap;

fn map_of(layout: &Layout, shape: &layout::Shape, plane: &Plane) -> Map {
    let (stride, offset) = layout.map(shape, plane);
    Map { stride, offset }
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

/// Each cache row's pool, and for a state row the record (fixed shape and
/// piece, [`Pools::records`]) its slot ids name.
#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>, pub Vec<Option<usize>>);

/// Slot refs of a lane the shell fires for itself ([`Pools::scratch`]).
#[derive(Debug, Clone, Default)]
pub struct Scratch {
    pub pages: Vec<u32>,
    pub window: Vec<u32>,
    pub fixed: Vec<(ShapeId, Vec<[u32; 2]>)>,
}

/// Where one cache row lives.
#[derive(Debug, Clone, Copy)]
enum Seat {
    Kv {
        windowed: bool,
        keys: (Tensor, Map),
        values: (Tensor, Map),
        head_dim: u32,
        kv_heads: u32,
    },
    State {
        record: usize,
        view: Tensor,
        map: Map,
    },
}

pub struct Pools {
    /// The pool; `None` while a program holds it (or on a dry device).
    buffer: Option<Buffer>,
    root: Tensor,
    layout: Layout,
    chunks: u32,
    table: ChunkTable,
    seats: Vec<Seat>,
    /// The compressor state planes `[kv, score]` of each cache row a
    /// `PoolGather` folds into.
    compressor: Vec<(u32, [Tensor; 2], [Map; 2])>,
    /// The rs list's plane of each rs layout plane.
    rs_buffers: Vec<(Tensor, Map)>,
    /// Every fixed shape's pieces, in order: the records' slot id tables.
    records: Vec<(ShapeId, u32)>,
    /// Every plane's view of the pool, and its map.
    views: Vec<(What, Tensor, Map)>,
    paging: Paging,
    bytes: u64,
    /// The chunks the load seated the streamed weights in, just below the
    /// sink (`crate::store::streamed`); 0 where none streams.
    weight_chunks: u32,
    /// Each streamed weight plane's view of the pool: `(width, view)`,
    /// bf16 rows of `width`.
    weight_views: Vec<(u32, Tensor)>,
}

impl std::fmt::Debug for Pools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pools")
            .field("chunks", &self.chunks)
            .field("chunk_bytes", &self.layout.chunk_bytes)
            .field("shapes", &self.layout.shapes.len())
            .finish()
    }
}

/// Whether a kv row declared with `window` lives in the window list.
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

/// Bytes the pools of `trace` take at `paging`, sink page and slot included:
/// what [`Pools::reserve`] lands there, compressor and recurrent-verb
/// planes too.
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
                cells * (lanes(planes.keys) + lanes(values)) * elem_bytes(name, *dtype)?
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
        total += 2 * cells * lanes(width) * 2;
    }
    if let Some(layout) = crate::rs::Layout::read(trace)? {
        total += rs_rows(paging) * layout.token_bytes;
    }
    Ok(total)
}

/// A row-major plane's row as the TPU lays it: whole 128-lane tiles.
fn lanes(width: u64) -> u64 {
    if width == 0 {
        0
    } else {
        width.div_ceil(128) * 128
    }
}

/// The rows of a recurrent-verb buffer plane: a page of rows per slot.
fn rs_rows(paging: Paging) -> u64 {
    u64::from(paging.slots.max(1)) * u64::from(paging.page_size.max(1))
}

/// The planes every cache row lays, as the layout reads them.
fn layout_rows<'a>(
    trace: &'a Trace,
    paging: Paging,
    facts: &Facts,
    rs: Option<&crate::rs::Layout>,
) -> Result<(layout::Rows<'a>, Vec<(u32, u32)>)> {
    let mut kv = vec![None; trace.caches.len()];
    let mut state = vec![None; trace.caches.len()];
    let mut heads = vec![(0u32, 0u32); trace.caches.len()];
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
                let restated = facts
                    .rows
                    .get(index)
                    .copied()
                    .flatten()
                    .filter(|seat| seat.kv_heads != 0);
                if let Some(seat) = restated {
                    let width = u64::from(seat.kv_heads) * u64::from(seat.head_dim);
                    if width != split.keys {
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
                heads[index] = (narrow(head_dim), narrow(kv_heads));
                let shared = split.shared || split.values == 0;
                kv[index] = Some((
                    narrow(split.keys),
                    (!shared).then(|| narrow(split.values)),
                    *dtype,
                    windowed_row(paging, *window),
                ));
            }
            CacheRow::State { slab, dtype, .. } => {
                let (per, width) = state_block(slab);
                state[index] = Some((narrow(per), narrow(width), state_dtype(*dtype)));
            }
        }
    }
    let compressor = compressor_rows(trace)
        .into_iter()
        .map(|(row, width)| (row, narrow(width)))
        .collect();
    let rs = rs.map_or_else(Vec::new, |layout| {
        layout
            .planes
            .iter()
            .map(|plane| (plane.width, plane.dtype))
            .collect()
    });
    Ok((
        layout::Rows {
            trace: Some(trace),
            page: paging.page_size,
            window: paging.window.map(|w| w.tokens),
            kv,
            state,
            compressor,
            rs,
        },
        heads,
    ))
}

/// The layout `trace` lays at `paging`, its chunk near `target` bytes.
pub fn layout_of(
    trace: &Trace,
    paging: Paging,
    rs: Option<&crate::rs::Layout>,
    target: u64,
) -> Result<Layout> {
    let facts = kv::probe(trace)?;
    let (rows, _) = layout_rows(trace, paging, &facts, rs)?;
    Layout::of(&rows, target)
}

impl Pools {
    /// The pool `bytes` buys (whole chunks, at least the reserved ones and
    /// one of each shape), every plane's view minted over it.
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        device: &Device,
        handles: &Handles,
        trace: &Trace,
        paging: Paging,
        facts: &Facts,
        rs: Option<&crate::rs::Layout>,
        bytes: u64,
        target: u64,
        streamed: Option<&mut streamed::Streamed>,
        extra_chunks: u32,
    ) -> Result<Pools> {
        let (rows, heads) = layout_rows(trace, paging, facts, rs)?;
        let layout = match &streamed {
            Some(s) if !s.is_empty() => Layout::of_with(&rows, target, s.unit, s.least)?,
            _ => Layout::of(&rows, target)?,
        };
        let least = RESERVED_CHUNKS + narrow(layout.shapes.len() as u64);
        let chunks = narrow(bytes / layout.chunk_bytes.max(1)).max(least);
        // The streamed weights seated whole, in chunks of their own just
        // below the sink: the state chunks the pool would have had stay
        // below them.
        let (chunks, weight_chunks) = match streamed {
            Some(s) if !s.is_empty() => {
                let weight_chunks = s.chunks(layout.chunk_bytes);
                let chunks = chunks + weight_chunks;
                s.seat(layout.chunk_bytes, chunks - 1 - weight_chunks)?;
                (chunks, weight_chunks)
            }
            _ => (chunks + extra_chunks, 0),
        };
        layout.check_reach(chunks)?;
        let pool_rows = u64::from(chunks) * layout.chunk_bytes / ROW_BYTES;
        let pool_rows = u32::try_from(pool_rows).map_err(|_| Fault::Ceiling {
            what: "rows of the pool buffer",
            need: pool_rows,
            have: u64::from(u32::MAX),
        })?;
        let buffer = device.zeros_or_dry(Dtype::U32, pool_rows, 128)?;
        let root = handles.root(Root {
            source: Source::Pool { row: 0, plane: 0 },
            dtype: Dtype::U32,
            rows: pool_rows,
            width: 128,
        });
        // A handle a plane: one pool backs them all, so a lowering lands
        // one plane an emit (`crate::lower::lower_planes`), and the tracer
        // refuses a plane landed off a pool another plane's write has since
        // replaced.
        let mut views: Vec<(What, Tensor, Map)> = Vec::new();
        for shape in &layout.shapes {
            for plane in &shape.planes {
                let t =
                    handles
                        .view(root, plane.dtype, plane.width)
                        .map_err(|e| Fault::Unbound {
                            what: format!("a view of pool plane {:?}: {e}", plane.what),
                        })?;
                views.push((plane.what, t, map_of(&layout, shape, plane)));
            }
        }
        let view = |what: What| -> Result<(Tensor, Map)> {
            views
                .iter()
                .find(|(held, ..)| *held == what)
                .map(|&(_, t, map)| (t, map))
                .ok_or_else(|| Fault::Unbound {
                    what: format!("pool plane {what:?}, which the layout lays nowhere"),
                })
        };
        let records: Vec<(ShapeId, u32)> = layout
            .fixed()
            .flat_map(|shape| (0..shape.pieces()).map(move |piece| (shape.id, piece)))
            .collect();
        let mut seats = Vec::with_capacity(trace.caches.len());
        for (index, row) in trace.caches.iter().enumerate() {
            let at = narrow(index as u64);
            seats.push(match row {
                CacheRow::Kv { .. } => {
                    let Some((_, values, _, windowed)) = rows.kv[index] else {
                        return Err(Fault::Unbound {
                            what: format!("cache row {index}, which lays no kv plane"),
                        });
                    };
                    let keys = view(What::Keys { row: at })?;
                    let values = match values {
                        Some(_) => view(What::Values { row: at })?,
                        None => keys,
                    };
                    Seat::Kv {
                        windowed,
                        keys,
                        values,
                        head_dim: heads[index].0,
                        kv_heads: heads[index].1,
                    }
                }
                CacheRow::State { .. } => {
                    let (shape, plane) =
                        layout
                            .find(What::State { row: at })
                            .ok_or_else(|| Fault::Unbound {
                                what: format!("state row {index}, which the layout lays nowhere"),
                            })?;
                    let record = records
                        .iter()
                        .position(|&r| r == (shape.id, plane.piece))
                        .ok_or_else(|| Fault::Unbound {
                            what: format!("state row {index}'s record piece"),
                        })?;
                    let (view, map) = view(What::State { row: at })?;
                    Seat::State { record, view, map }
                }
            });
        }
        let mut compressor = Vec::new();
        for &(row, _) in &rows.compressor {
            let (kv, kv_map) = view(What::Compressor { row, which: 0 })?;
            let (score, score_map) = view(What::Compressor { row, which: 1 })?;
            compressor.push((row, [kv, score], [kv_map, score_map]));
        }
        let rs_buffers = (0..rows.rs.len())
            .map(|plane| {
                view(What::Rs {
                    plane: plane as u32,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let total = u64::from(chunks) * layout.chunk_bytes;
        tracing::info!(
            chunks,
            chunk_mib = layout.chunk_bytes as f64 / f64::from(1u32 << 20),
            shapes = ?layout
                .shapes
                .iter()
                .map(|s| (s.id.0, s.kind, s.slot_bytes, layout.slots_per_chunk(s)))
                .collect::<Vec<_>>(),
            "xla state pool laid"
        );
        Ok(Pools {
            buffer,
            root,
            table: ChunkTable::identity(chunks),
            layout,
            chunks,
            seats,
            compressor,
            rs_buffers,
            records,
            views,
            paging,
            bytes: total,
            weight_chunks,
            weight_views: Vec::new(),
        })
    }

    /// Mints a bf16 view of the pool `width` wide for each width the
    /// streamed weights' planes read (before the handles seal).
    pub fn view_weights(&mut self, handles: &Handles, streamed: &streamed::Streamed) -> Result<()> {
        for shape in &streamed.shapes {
            for plane in &shape.planes {
                if self.weight_views.iter().any(|(w, _)| *w == plane.view) {
                    continue;
                }
                let t = handles
                    .view(self.root, Dtype::Bf16, plane.view)
                    .map_err(|e| Fault::Unbound {
                        what: format!("a {}-wide weight view of the pool: {e}", plane.view),
                    })?;
                self.weight_views.push((plane.view, t));
            }
        }
        Ok(())
    }

    /// The bf16 view of the pool `width` wide.
    #[must_use]
    pub fn weight_view(&self, width: u32) -> Option<Tensor> {
        self.weight_views
            .iter()
            .find(|(w, _)| *w == width)
            .map(|&(_, t)| t)
    }

    /// The first chunk the streamed weights were seated in (the sink where
    /// none streams).
    #[must_use]
    pub fn weight_start(&self) -> u32 {
        self.sink_chunk() - self.weight_chunks
    }

    /// Byte offset in the pool of weight slot ref `slot` of `shape`.
    pub fn weight_offset(&self, shape: &streamed::Shape, slot: u32) -> Result<u64> {
        let spc = shape.per_chunk.max(1);
        let logical = slot / spc;
        let chunk = self.table.physical(logical).ok_or(Fault::Ceiling {
            what: "chunks a weight slot ref names",
            need: u64::from(logical) + 1,
            have: u64::from(self.chunks),
        })?;
        if chunk == NULL_CHUNK || chunk >= self.sink_chunk() {
            return Err(Fault::Unbound {
                what: format!(
                    "weight slot ref {slot} of {}, which lies in reserved chunk {chunk}",
                    shape.name
                ),
            });
        }
        Ok(u64::from(chunk) * self.layout.chunk_bytes + u64::from(slot % spc) * shape.slot_bytes)
    }

    /// Lands whole slots of a weight shape: `(slot ref, bytes)` each its
    /// slot's bytes.
    pub fn write_weights(
        &mut self,
        device: &Device,
        handles: &Handles,
        shape: &streamed::Shape,
        slots: &[(u32, Vec<u8>)],
    ) -> Result<()> {
        let rows = shape.slot_bytes / ROW_BYTES;
        let mut placed: Vec<(i64, &[u8])> = slots
            .iter()
            .map(|(slot, bytes)| {
                self.weight_offset(shape, *slot).map(|o| {
                    (
                        i64::try_from(o / ROW_BYTES).unwrap_or(i64::MAX),
                        bytes.as_slice(),
                    )
                })
            })
            .collect::<Result<_>>()?;
        placed.sort_by_key(|&(at, _)| at);
        // Slots that lie back to back land as one block (the load seats a
        // shape's atoms in a row), up to a few hundred MiB an upload; lone
        // slots land a few dozen to a program (one update each), so the
        // programs are few and small.
        let most = i64::try_from(((256u64 << 20) / shape.slot_bytes.max(1)).max(1)).unwrap_or(1);
        let step = i64::try_from(rows).unwrap_or(i64::MAX);
        let mut runs: Vec<(i64, Vec<&[u8]>)> = Vec::new();
        for (at, bytes) in placed {
            match runs.last_mut() {
                Some((start, run))
                    if *start + step * run.len() as i64 == at && (run.len() as i64) < most =>
                {
                    run.push(bytes);
                }
                _ => runs.push((at, vec![bytes])),
            }
        }
        let (lone, long): (Vec<_>, Vec<_>) = runs.into_iter().partition(|(_, run)| run.len() == 1);
        for (start, run) in long {
            let bytes: Vec<u8> = run.concat();
            self.write_rows(
                device,
                handles,
                rows * run.len() as u64,
                vec![start],
                &bytes,
            )?;
        }
        for batch in lone.chunks(LONE_SLOTS) {
            let at: Vec<i64> = batch.iter().map(|(at, _)| *at).collect();
            let bytes: Vec<u8> = batch
                .iter()
                .flat_map(|(_, run)| run[0].iter().copied())
                .collect();
            self.write_rows(device, handles, rows, at, &bytes)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[must_use]
    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    /// The pool as chunks: one segment whole from the load, chunk 0 null,
    /// chunk 1 the padding lanes' sink, every shape drawing on the rest.
    #[must_use]
    pub fn facts(&self) -> ChunkFacts {
        let shapes = self
            .layout
            .shapes
            .iter()
            .map(|shape| SlotShape {
                id: shape.id,
                kind: match shape.kind {
                    Kind::Kv | Kind::Rs => SlotKind::List {
                        tokens: shape.tokens,
                        rate: 1,
                        window: None,
                    },
                    Kind::Window { window } => SlotKind::List {
                        tokens: shape.tokens,
                        rate: 1,
                        window: Some(window),
                    },
                    Kind::Fixed { pieces } => SlotKind::Fixed { pieces },
                },
                slot_bytes: shape.slot_bytes,
                slots_per_chunk: self.layout.slots_per_chunk(shape),
                sets_per_working_set: 1,
                rows: shape.rows.clone(),
                pinned: None,
                role: if shape.kind == Kind::Rs {
                    SlotRole::RsBuffer
                } else {
                    SlotRole::State
                },
            })
            .collect();
        ChunkFacts {
            chunk_bytes: self.layout.chunk_bytes,
            total_chunks: self.chunks,
            segment_chunks: self.chunks,
            segments: SegmentFacts {
                committed: 1,
                max: 1,
                growth: Growth::Fixed,
            },
            reserved: vec![
                ReservedRun {
                    what: Reserved::Null,
                    first: NULL_CHUNK,
                    chunks: 1,
                },
                ReservedRun {
                    what: Reserved::Sink,
                    first: self.sink_chunk(),
                    chunks: 1,
                },
            ],
            shapes,
            act_chunks: Vec::new(),
            host_chunks: 0,
            disk_chunks: 0,
            disk_persistent: false,
            disk_kept: false,
            weights: Vec::new(),
        }
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn shape(&self, id: ShapeId) -> Result<&layout::Shape> {
        self.layout.shape(id).ok_or_else(|| Fault::Unbound {
            what: format!("slot shape {id}, which this load does not state"),
        })
    }

    /// The grain index slot ref `slot` of shape `id` names: its logical
    /// chunk placed through the table. Ref 0 of a window list is the null
    /// chunk; every other ref must name a chunk the runtime may hand out.
    pub fn grain(&self, id: ShapeId, slot: u32) -> Result<u32> {
        let shape = self.shape(id)?;
        let spc = self.layout.slots_per_chunk(shape).max(1);
        let logical = slot / spc;
        let index = slot % spc;
        let null_ok = matches!(shape.kind, Kind::Window { .. }) && slot == 0;
        let chunk = self.table.physical(logical).ok_or(Fault::Ceiling {
            what: "chunks a slot ref names",
            need: u64::from(logical) + 1,
            have: u64::from(self.chunks),
        })?;
        if !null_ok && (chunk == NULL_CHUNK || chunk >= self.sink_chunk()) {
            return Err(Fault::Unbound {
                what: format!(
                    "slot ref {slot} of shape {id}, which lies in reserved chunk {chunk}"
                ),
            });
        }
        u32::try_from(self.layout.grain_of(shape, chunk, index)).map_err(|_| Fault::Ceiling {
            what: "grain indices a table carries",
            need: self.layout.grain_of(shape, chunk, index),
            have: u64::from(u32::MAX),
        })
    }

    /// Each of `slots` (refs of shape `id`) as its grain index.
    pub fn grains(&self, id: ShapeId, slots: &[u32]) -> Result<Vec<u32>> {
        slots.iter().map(|&slot| self.grain(id, slot)).collect()
    }

    /// The sink chunk: the pool's top one.
    #[must_use]
    pub fn sink_chunk(&self) -> u32 {
        self.chunks.saturating_sub(1)
    }

    /// The grain index of shape `id`'s slot in the sink chunk.
    #[must_use]
    pub fn sink(&self, id: ShapeId) -> u32 {
        self.layout.shape(id).map_or(0, |shape| {
            narrow(self.layout.grain_of(shape, self.sink_chunk(), 0))
        })
    }

    /// Slot refs for a lane the shell fires for itself (the warm ladder, a
    /// dry walk): `pages` of the kv list and of the window list, and a
    /// record, laid from the first chunk past the reserved ones on and
    /// wrapping over the pool. What such fires write is never read.
    #[must_use]
    pub fn scratch(&self, lane: u32, pages: usize) -> Scratch {
        let room = u64::from(
            self.chunks
                .saturating_sub(RESERVED_CHUNKS + self.weight_chunks)
                .max(1),
        );
        let refs = |shape: &layout::Shape, n: usize, from: u64| -> Vec<u32> {
            let spc = u64::from(self.layout.slots_per_chunk(shape).max(1));
            (0..n as u64)
                .map(|i| {
                    let at = (from + i) % (room * spc);
                    narrow((u64::from(NULL_CHUNK) + 1 + at / spc) * spc + at % spc)
                })
                .collect()
        };
        let from = u64::from(lane) * pages as u64;
        Scratch {
            pages: self
                .layout
                .kv()
                .map_or_else(Vec::new, |s| refs(s, pages, from)),
            window: self
                .layout
                .window()
                .map_or_else(Vec::new, |s| refs(s, pages, from)),
            fixed: self
                .layout
                .fixed()
                .map(|s| {
                    let pieces = s.pieces() as usize;
                    let r = refs(s, pieces, u64::from(lane) * pieces as u64);
                    (s.id, r.into_iter().map(|r| [r, r]).collect())
                })
                .collect(),
        }
    }

    /// Every fixed shape and its pieces a record.
    pub fn fixed_shapes(&self) -> impl Iterator<Item = (ShapeId, u32)> + '_ {
        self.layout.fixed().map(|s| (s.id, s.pieces()))
    }

    /// The kv list's shape, the window list's, the rs list's.
    #[must_use]
    pub fn kv_shape(&self) -> Option<ShapeId> {
        self.layout.kv().map(|s| s.id)
    }

    #[must_use]
    pub fn window_shape(&self) -> Option<ShapeId> {
        self.layout.window().map(|s| s.id)
    }

    #[must_use]
    pub fn rs_shape(&self) -> Option<ShapeId> {
        self.layout.rs().map(|s| s.id)
    }

    /// Every fixed shape's pieces, in order: the fire stages one slot id
    /// table each.
    #[must_use]
    pub fn records(&self) -> &[(ShapeId, u32)] {
        &self.records
    }

    /// The page tables a fire stages, by whether each pages the window
    /// list: the kv list's first (the All axis's), then the window list's.
    /// A model keeping only windowed rows reads its All axis off the window
    /// list's table, then the only one.
    #[must_use]
    pub fn tables(&self) -> Vec<bool> {
        [
            self.layout.kv().is_some().then_some(false),
            self.layout.window().is_some().then_some(true),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// The page table of [`Pools::tables`] the rows in the window list
    /// (`windowed`) or in the kv list read.
    #[must_use]
    pub fn table_of(&self, windowed: bool) -> usize {
        usize::from(windowed && self.layout.kv().is_some())
    }

    #[must_use]
    pub fn has_windowed(&self) -> bool {
        self.layout.window().is_some()
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        !self.records.is_empty()
    }

    /// A working set's record: every fixed shape's pieces.
    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.layout
            .fixed()
            .map(|s| s.slot_bytes * u64::from(s.pieces()))
            .sum()
    }

    /// Takes the pool out for a program to donate.
    pub fn take(&mut self, row: u32, plane: u8) -> Result<Buffer> {
        if (row, plane) != (0, 0) {
            return Err(Fault::Unbound {
                what: format!("pool plane {row}.{plane}; the pool is one buffer"),
            });
        }
        self.buffer.take().ok_or_else(|| Fault::Unbound {
            what: "the pool, which a program still holds".to_string(),
        })
    }

    #[must_use]
    pub fn get(&self, row: u32, plane: u8) -> Option<&Buffer> {
        ((row, plane) == (0, 0))
            .then_some(self.buffer.as_ref())
            .flatten()
    }

    /// Puts back what a program returned for the pool.
    pub fn put(&mut self, row: u32, plane: u8, buffer: Buffer) {
        if (row, plane) == (0, 0) {
            self.buffer = Some(buffer);
        }
    }

    /// The pools as kernel handles, with this fire's page geometry and each
    /// record piece's slot id table (`slots`, by [`Pools::records`]).
    pub fn table(&self, seats: &[TableSeat], slots: &[Tensor]) -> Result<CacheTable> {
        let mut rows = Vec::with_capacity(self.seats.len());
        let mut records = Vec::with_capacity(self.seats.len());
        for seat in &self.seats {
            match *seat {
                Seat::Kv {
                    windowed,
                    keys,
                    values,
                    head_dim,
                    kv_heads,
                } => {
                    let table = self.table_of(windowed);
                    let at = seats.get(table).ok_or_else(|| Fault::Unbound {
                        what: format!("page table {table}, which this fire staged none of"),
                    })?;
                    rows.push(CachePool::Kv(KvPool {
                        keys: keys.0,
                        values: values.0,
                        page_indices: at.page_indices,
                        page_indptr: at.page_indptr,
                        page_size: i32::try_from(self.paging.page_size).unwrap_or(i32::MAX),
                        max_pages: at.max_pages,
                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                        keys_map: keys.1,
                        values_map: values.1,
                    }));
                    records.push(None);
                }
                Seat::State { record, view, map } => {
                    let slots = *slots.get(record).ok_or_else(|| Fault::Unbound {
                        what: format!("record table {record}, which this fire staged none of"),
                    })?;
                    rows.push(CachePool::Recurrent(RecurrentPool {
                        state: view,
                        slots,
                        conv_state: view,
                        new_conv_state: view,
                        map,
                    }));
                    records.push(Some(record));
                }
            }
        }
        Ok(CacheTable(rows, records))
    }

    /// The rs list's planes, one per rs layout plane, each with its map.
    #[must_use]
    pub fn rs_buffers(&self) -> Vec<(Tensor, Map)> {
        self.rs_buffers.clone()
    }

    /// Every cache row's compressor planes, for a walk.
    #[must_use]
    pub fn compressors(&self) -> Vec<(u32, [Tensor; 2], [Map; 2])> {
        self.compressor.clone()
    }

    /// The pool rows grain index `q` of shape `id` starts at.
    fn rows_at(&self, id: ShapeId, q: u32) -> Result<i64> {
        let shape = self.shape(id)?;
        Ok(i64::try_from(u64::from(q) * self.layout.grain(shape) / ROW_BYTES).unwrap_or(i64::MAX))
    }

    /// Zeros each of `slots` (`(shape, slot ref)`) whole: the records a
    /// fresh state starts from.
    pub fn zero(
        &mut self,
        device: &Device,
        handles: &Handles,
        slots: &[(ShapeId, u32)],
    ) -> Result<()> {
        let mut by: std::collections::BTreeMap<u64, Vec<i64>> = Default::default();
        for &(id, slot) in slots {
            let q = self.grain(id, slot)?;
            let rows = self.shape(id)?.slot_bytes / ROW_BYTES;
            by.entry(rows).or_default().push(self.rows_at(id, q)?);
        }
        for (rows, mut at) in by {
            at.sort_unstable();
            at.dedup();
            pad_pow2(&mut at, 1);
            let (inputs, list) = indices(handles, &at);
            let tracer = Tracer::new(handles).with_pack(inputs.pack_len());
            let root = self.root;
            let n = at.len() as i64;
            tracer.emit(&mut |cx| {
                let mut whole = cx.read(root)?;
                let zero = cx.const_i(Elem::U32, 0, &[rows as i64, 128]);
                let col = cx.const_i(Elem::I32, 0, &[]);
                for m in 0..n {
                    let row = index_at(cx, list, m, 1)?;
                    whole = cx.dynamic_update_slice(whole, zero, &[row, col])?;
                }
                cx.write(root, whole)
            })?;
            crate::exec::maintain(device, tracer, self, &inputs)?;
        }
        Ok(())
    }

    /// Copies whole slots `(shape, src ref, dst ref)` in order (a later one
    /// may read what an earlier one wrote).
    pub fn copy(
        &mut self,
        device: &Device,
        handles: &Handles,
        copies: &[(ShapeId, u32, u32)],
    ) -> Result<()> {
        let mut spans: Vec<(u64, i64, i64)> = Vec::with_capacity(copies.len());
        for &(id, src, dst) in copies {
            if src == dst {
                continue;
            }
            let rows = self.shape(id)?.slot_bytes / ROW_BYTES;
            let from = self.rows_at(id, self.grain(id, src)?)?;
            let to = self.rows_at(id, self.grain(id, dst)?)?;
            spans.push((rows, from, to));
        }
        self.copy_rows(device, handles, &spans)
    }

    /// Copies `rows` pool rows from each span's first to its second, a
    /// program a run of equal sizes, in order.
    fn copy_rows(
        &mut self,
        device: &Device,
        handles: &Handles,
        spans: &[(u64, i64, i64)],
    ) -> Result<()> {
        let mut at = 0;
        while at < spans.len() {
            let rows = spans[at].0;
            let end = (at..spans.len())
                .find(|&i| spans[i].0 != rows)
                .unwrap_or(spans.len());
            let mut flat: Vec<i64> = spans[at..end]
                .iter()
                .flat_map(|&(_, from, to)| [from, to])
                .collect();
            pad_pow2(&mut flat, 2);
            let (inputs, list) = indices(handles, &flat);
            let tracer = Tracer::new(handles).with_pack(inputs.pack_len());
            let root = self.root;
            let n = flat.len() as i64 / 2;
            tracer.emit(&mut |cx| {
                let mut whole = cx.read(root)?;
                let col = cx.const_i(Elem::I32, 0, &[]);
                for m in 0..n {
                    let from = index_at(cx, list, 2 * m, 1)?;
                    let to = index_at(cx, list, 2 * m + 1, 1)?;
                    let block = cx.dynamic_slice(whole, &[from, col], &[rows as i64, 128])?;
                    whole = cx.dynamic_update_slice(whole, block, &[to, col])?;
                }
                cx.write(root, whole)
            })?;
            crate::exec::maintain(device, tracer, self, &inputs)?;
            at = end;
        }
        Ok(())
    }

    /// Applies an evacuation: each chunk's bytes moved, then the table.
    pub fn evacuate(
        &mut self,
        device: &Device,
        handles: &Handles,
        evacuation: &Evacuation,
    ) -> Result<()> {
        let rows = self.layout.chunk_bytes / ROW_BYTES;
        let spans: Vec<(u64, i64, i64)> = evacuation
            .moves
            .iter()
            .map(|&(from, to)| {
                (
                    rows,
                    i64::from(from) * rows as i64,
                    i64::from(to) * rows as i64,
                )
            })
            .collect();
        self.copy_rows(device, handles, &spans)?;
        self.table.apply(evacuation).map_err(|e| Fault::Unbound {
            what: format!("an evacuation the chunk table refuses: {e}"),
        })
    }

    /// Applies kv `moves` (slot refs of the kv list): a whole page by its
    /// slot, part of one plane by plane, cell by cell.
    pub fn copy_kv(&mut self, device: &Device, handles: &Handles, moves: &[KvRun]) -> Result<()> {
        let Some(id) = self.kv_shape() else {
            return Ok(());
        };
        let page = self.paging.page_size;
        let mut whole: Vec<(ShapeId, u32, u32)> = Vec::new();
        let mut cells: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
        for m in moves {
            if m.tokens == page && m.src_token == 0 && m.dst_token == 0 {
                whole.push((id, m.src_page, m.dst_page));
            } else {
                cells.push((
                    self.grain(id, m.src_page)?,
                    m.src_token,
                    self.grain(id, m.dst_page)?,
                    m.dst_token,
                    m.tokens,
                ));
            }
        }
        self.copy(device, handles, &whole)?;
        if cells.is_empty() {
            return Ok(());
        }
        let shape = self.shape(id)?.clone();
        for plane in &shape.planes {
            let (_, view, map) = *self
                .views
                .iter()
                .find(|(what, ..)| *what == plane.what)
                .ok_or_else(|| Fault::Unbound {
                    what: format!("pool plane {:?}, which has no view", plane.what),
                })?;
            let mut src: Vec<i64> = Vec::new();
            let mut dst: Vec<i64> = Vec::new();
            for &(sq, st, dq, dt, n) in &cells {
                for t in 0..n {
                    src.push(
                        (i64::from(sq) * map.stride + map.offset) * i64::from(page)
                            + i64::from(st + t),
                    );
                    dst.push(
                        (i64::from(dq) * map.stride + map.offset) * i64::from(page)
                            + i64::from(dt + t),
                    );
                }
            }
            let n = src.len().next_power_of_two();
            for list in [&mut src, &mut dst] {
                let last = list[list.len() - 1];
                list.resize(n, last);
            }
            let mut flat = src.clone();
            flat.extend_from_slice(&dst);
            let (inputs, at) = indices(handles, &flat);
            let tracer = Tracer::new(handles).with_pack(inputs.pack_len());
            let len = flat.len() as i64;
            let n = n as i64;
            tracer.emit(&mut |cx| {
                let whole = cx.read(view)?;
                let all = cx.read(at)?;
                let all = cx.reshape(all, &[len])?;
                let s = cx.slice_axis(all, 0, 0, n)?;
                let d = cx.slice_axis(all, 0, n, 2 * n)?;
                let rows = cx.take_rows(whole, s)?;
                let next = cx.put_rows(whole, d, rows, kernels_xla::hlo::Combine::Set)?;
                cx.write(view, next)
            })?;
            crate::exec::maintain(device, tracer, self, &inputs)?;
        }
        Ok(())
    }

    /// The bytes of each of `slots` (`(shape, slot ref)`, one shape), in
    /// order: one program gathers their pool rows into a result, which is
    /// downloaded. Ordered after every fire submitted before it.
    pub fn read_slots(
        &mut self,
        device: &Device,
        handles: &Handles,
        id: ShapeId,
        slots: &[u32],
    ) -> Result<Vec<u8>> {
        if slots.is_empty() || device.is_dry() {
            return Ok(Vec::new());
        }
        let rows = self.shape(id)?.slot_bytes / ROW_BYTES;
        let mut at: Vec<i64> = slots
            .iter()
            .map(|&slot| self.rows_at(id, self.grain(id, slot)?))
            .collect::<Result<_>>()?;
        let n = at.len();
        pad_pow2(&mut at, 1);
        let (inputs, list) = indices(handles, &at);
        let tracer = Tracer::new(handles).with_pack(inputs.pack_len());
        let root = self.root;
        let mut out = None;
        tracer.emit(&mut |cx| {
            let whole = cx.read(root)?;
            let col = cx.const_i(Elem::I32, 0, &[]);
            let mut blocks = Vec::with_capacity(at.len());
            for m in 0..at.len() as i64 {
                let row = index_at(cx, list, m, 1)?;
                blocks.push(cx.dynamic_slice(whole, &[row, col], &[rows as i64, 128])?);
            }
            out = Some(if blocks.len() == 1 {
                blocks[0]
            } else {
                cx.concat(&blocks, 0)?
            });
            Ok(())
        })?;
        let out = out.expect("the gather emitted");
        let got = crate::exec::run(device, tracer, &[out], None, self, &inputs, true)?;
        let bytes = got
            .first()
            .ok_or_else(|| Fault::Unbound {
                what: "the gathered slots, which the program did not return".to_string(),
            })?
            .download()?;
        let len = (n as u64 * rows * ROW_BYTES) as usize;
        Ok(bytes[..len.min(bytes.len())].to_vec())
    }

    /// Lands `bytes` (each of `slots` whole, in order, one shape) in the
    /// pool: one upload, a `dynamic_update_slice` a slot.
    pub fn write_slots(
        &mut self,
        device: &Device,
        handles: &Handles,
        id: ShapeId,
        slots: &[u32],
        bytes: &[u8],
    ) -> Result<()> {
        if slots.is_empty() || device.is_dry() {
            return Ok(());
        }
        let rows = self.shape(id)?.slot_bytes / ROW_BYTES;
        if bytes.len() as u64 != slots.len() as u64 * rows * ROW_BYTES {
            return Err(Fault::Unbound {
                what: format!(
                    "{} bytes for {} slots of {} bytes",
                    bytes.len(),
                    slots.len(),
                    rows * ROW_BYTES
                ),
            });
        }
        let at: Vec<i64> = slots
            .iter()
            .map(|&slot| self.rows_at(id, self.grain(id, slot)?))
            .collect::<Result<_>>()?;
        self.write_rows(device, handles, rows, at, bytes)
    }

    /// Lands `bytes`, blocks of `rows` pool rows each, at pool rows `at`:
    /// one upload, a `dynamic_update_slice` a block.
    fn write_rows(
        &mut self,
        device: &Device,
        handles: &Handles,
        rows: u64,
        mut at: Vec<i64>,
        bytes: &[u8],
    ) -> Result<()> {
        if at.is_empty() || device.is_dry() {
            return Ok(());
        }
        // Pads repeat the last slot with its own bytes.
        let n = at.len();
        pad_pow2(&mut at, 1);
        let mut data = bytes.to_vec();
        let last = bytes[bytes.len() - (rows * ROW_BYTES) as usize..].to_vec();
        for _ in n..at.len() {
            data.extend_from_slice(&last);
        }
        let mut inputs = crate::inputs::Inputs::new();
        let landed = inputs.raw(
            handles,
            Dtype::U32,
            narrow(at.len() as u64 * rows),
            128,
            data,
        );
        let values: Vec<i32> = at
            .iter()
            .map(|&v| i32::try_from(v).unwrap_or(i32::MAX))
            .collect();
        let list = inputs.i32s(handles, &values, 1);
        let tracer = Tracer::new(handles).with_pack(inputs.pack_len());
        let root = self.root;
        let count = at.len() as i64;
        tracer.emit(&mut |cx| {
            let mut whole = cx.read(root)?;
            let all = cx.read(landed)?;
            let col = cx.const_i(Elem::I32, 0, &[]);
            for m in 0..count {
                let row = index_at(cx, list, m, 1)?;
                let block = cx.slice_axis(all, 0, m * rows as i64, (m + 1) * rows as i64)?;
                whole = cx.dynamic_update_slice(whole, block, &[row, col])?;
            }
            cx.write(root, whole)
        })?;
        crate::exec::maintain(device, tracer, self, &inputs)
    }

    /// The bytes of each of `slots` (`(shape, slot ref)`), in order.
    pub fn read(&self, slots: &[(ShapeId, u32)]) -> Result<Vec<u8>> {
        let Some(buffer) = self.buffer.as_ref() else {
            return Ok(Vec::new());
        };
        let all = buffer.download()?;
        let mut out = Vec::new();
        for &(id, slot) in slots {
            let shape = self.shape(id)?;
            let q = self.grain(id, slot)?;
            let at = (u64::from(q) * self.layout.grain(shape)) as usize;
            out.extend_from_slice(&all[at..at + shape.slot_bytes as usize]);
        }
        Ok(out)
    }
}

/// Pads `list` (pairs when `per` is 2) to a power of two by repeating its
/// last entry, which lands again what it landed.
fn pad_pow2(list: &mut Vec<i64>, per: usize) {
    let n = (list.len() / per).next_power_of_two();
    let last: Vec<i64> = list[list.len() - per..].to_vec();
    while list.len() < n * per {
        list.extend_from_slice(&last);
    }
}

/// `values` as the one i32 input of a maintenance program, so its text names
/// how many slots, pages or cells it moves and not which: one program serves
/// every slot.
fn indices(handles: &Handles, values: &[i64]) -> (crate::inputs::Inputs, Tensor) {
    let mut inputs = crate::inputs::Inputs::new();
    let values: Vec<i32> = values
        .iter()
        .map(|&v| i32::try_from(v).unwrap_or(i32::MAX))
        .collect();
    let at = inputs.i32s(handles, &values, 1);
    (inputs, at)
}

/// Entry `i` of an `indices` input, times `scale`, as an i32 scalar.
fn index_at(
    cx: &mut crate::emit::Cx<'_>,
    at: Tensor,
    i: i64,
    scale: i64,
) -> std::result::Result<kernels_xla::hlo::Val, kernels_xla::Error> {
    let all = cx.read(at)?;
    let one = cx.slice_axis(all, 0, i, i + 1)?;
    let one = cx.reshape(one, &[])?;
    if scale == 1 {
        return Ok(one);
    }
    let by = cx.const_i(Elem::I32, scale, &[]);
    Ok(cx.mul(one, by)?)
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

fn state_block(slab: &[u64]) -> (u64, u64) {
    let stride: u64 = slab.iter().product();
    match *slab {
        [h, d1, d2] if (h * d1).is_multiple_of(8) && d2.is_multiple_of(128) => (h * d1, d2),
        [_, _] if stride.is_multiple_of(8 * 128) => (8, stride / 8),
        _ => (1, stride),
    }
}

fn narrow(n: u64) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}
