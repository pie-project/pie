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
pub mod stage;
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

/// The null chunk; the sink is the pool's top chunk ([`Pools::sink_chunk`]),
/// one past the range the load states (`ChunkFacts::total_chunks`): every
/// kernel that treats a plane view's last page or block as the sink (the
/// prefill page writer drops rows bound there) then finds it in a chunk the
/// runtime never names, and the activation span the runtime keeps at the
/// top of the stated range lies just below it.
const NULL_CHUNK: u32 = 0;
const RESERVED_CHUNKS: u32 = 2;


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
    /// The rs list's slots, as many as `pages`.
    pub rs: Vec<u32>,
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
    /// The windowed layers whose rings the record holds (CONTRACT-BC 21),
    /// as the load states them ([`Pools::set_windowed`]).
    windowed: Vec<engine::chunk::WindowedLayer>,
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
    /// Each streamed bank's miss copy in pinned host memory: `(param, pool
    /// rows an expert's plane takes, root, buffer)`.
    host_banks: Vec<(u32, u32, Tensor, Buffer)>,
    /// The staging slab: its first pool row and its rows (none on a shell
    /// the load did not size).
    slab: (u32, u32),
    /// The windowed state's own-row pages: their first pool row and count
    /// ([`Pools::own_grain`]).
    own: (u32, u32),
    /// `(rung tokens, chunks)`: what each rung's staged inputs take of the
    /// slab ([`ChunkFacts::act_chunks`]).
    act_chunks: Vec<(u32, u32)>,
    /// The staging ring's pinned host slots (none on a dry device).
    ring: Option<stage::Ring>,
    /// The bulk ring: pinned host slots of a chunk each, the bytes a
    /// maintenance program lands (none on a dry device).
    bulk: Option<stage::Ring>,
    /// An `Adapter` slot's bytes (whole pool rows): the largest plane of one
    /// adapter the model's registered banks hold; 0 where it has none.
    adapter_slot: u64,
    /// Each readout output's ring of planes laid at load
    /// ([`Pools::lay_readouts`]): a fire writes its readout rows into the
    /// next one in place (`trace::Source::Readout`).
    readouts: Vec<ReadoutRing>,
}

/// One readout output's planes: `[rows, width]` of `dtype` each, as many
/// as fires can be outstanding (`Runahead::staging_depth`), so a fire never
/// comes round to a plane a reader still holds.
#[derive(Debug)]
pub struct ReadoutRing {
    pub root: Tensor,
    pub dtype: Dtype,
    pub rows: u32,
    pub width: u32,
    slots: Vec<Option<std::sync::Arc<Buffer>>>,
    at: usize,
}

/// The bytes one adapter's largest plane takes in whole tiles of pool rows
/// (a maintenance program slices a slot by whole tiles: gpt-oss-20b's
/// 2880-wide rank-16 plane, 180 rows, refused "sublane slicing size not
/// multiple of chunk sublane size"): a registered bank `[slots, ..]` holds
/// one plane of `..` an adapter. 0 where the model registers none.
fn adapter_slot_bytes(trace: &Trace) -> u64 {
    trace
        .params
        .iter()
        .filter(|p| p.source == poem_ir::ParamSource::Registered)
        .map(|p| {
            let cells: u64 = p.shape.get(1..).unwrap_or(&[]).iter().product();
            let bytes = cells * poem_compiler::arena::elem_bytes(p.dtype).unwrap_or(0);
            bytes.div_ceil(layout::TILE_BYTES) * layout::TILE_BYTES
        })
        .max()
        .unwrap_or(0)
}

/// The bulk ring's slots: one a write fills while the write before it,
/// still in flight, reads the other.
const BULK_SLOTS: u32 = 2;

/// What a load stages a fire's inputs in: the slab's rows (the largest
/// staged set any fire of the load can take), the ring's slots (frames in
/// flight + 1), and each rung's own set `(rung tokens, rows)`, ascending.
#[derive(Debug, Clone, Default)]
pub struct Staging {
    pub rows: u32,
    pub slots: u32,
    pub rungs: Vec<(u32, u32)>,
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
                // a slot as the layout lays it: whole pool rows
                let elem = elem_bytes(name, state_dtype(*dtype))?;
                let (per, width) = state_block(slab, elem);
                per * width * u64::from(paging.slots + 1) * elem
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
            CacheRow::State {
                name, slab, dtype, ..
            } => {
                let elem = elem_bytes(name, state_dtype(*dtype))?;
                let (per, width) = state_block(slab, elem);
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
            rings: true,
        },
        heads,
    ))
}

/// The row order a kv plane's pages lie in (`kernels_xla::hlo::PoolOrder`):
/// part `j` (512 bytes) of a page's token `t` at row `j · T + t` of the
/// page's block, so a page's kv heads each lie as one run of whole tiles.
/// A token of one pool row or less lies as the rows lie (`None`: the order
/// is the identity there). A kv row past one pool row that is not whole
/// pool rows has no place in it: the load refuses it (no XLA kv kernel
/// takes such rows: the catalog's are e4m3 latent rows of 576 bytes).
fn kv_order(plane: &layout::Plane, pool_rows: u64) -> Result<Option<kernels_xla::hlo::PoolOrder>> {
    if !matches!(plane.what, What::Keys { .. } | What::Values { .. }) {
        return Ok(None);
    }
    let tokens = u64::from(plane.per);
    let row_bytes = plane.run / tokens.max(1);
    if row_bytes <= ROW_BYTES {
        return Ok(None);
    }
    if !row_bytes.is_multiple_of(ROW_BYTES)
        || plane.run != tokens * row_bytes
        || !pool_rows.is_multiple_of(tokens * row_bytes / ROW_BYTES)
    {
        return Err(Fault::Unbound {
            what: format!(
                "kv plane {:?}: {row_bytes}-byte rows ({} of {:?}), which the pool's \
                 row-pair order lays only as whole {ROW_BYTES}-byte rows",
                plane.what, plane.width, plane.dtype
            ),
        });
    }
    Ok(Some(kernels_xla::hlo::PoolOrder {
        tokens: tokens as i64,
        rpt: (row_bytes / ROW_BYTES) as i64,
    }))
}

/// Bytes the pool lays past the pages and state slots `pool_demand` counts:
/// its null and sink chunks.
pub fn reserved_bytes(
    trace: &Trace,
    paging: Paging,
    rs: Option<&crate::rs::Layout>,
    target: u64,
) -> Result<u64> {
    Ok(u64::from(RESERVED_CHUNKS) * layout_of(trace, paging, rs, target)?.chunk_bytes)
}

/// The staging slab's bytes in the pool at `paging`: `rows` pool rows in
/// whole chunks.
pub fn slab_bytes(
    trace: &Trace,
    paging: Paging,
    rs: Option<&crate::rs::Layout>,
    target: u64,
    rows: u32,
) -> Result<u64> {
    let chunk = layout_of(trace, paging, rs, target)?.chunk_bytes.max(1);
    Ok((u64::from(rows) * ROW_BYTES).div_ceil(chunk) * chunk)
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
        staging: &Staging,
    ) -> Result<Pools> {
        let (rows, heads) = layout_rows(trace, paging, facts, rs)?;
        let layout = match &streamed {
            Some(s) if !s.is_empty() => Layout::of_with(&rows, target, s.unit, s.least)?,
            _ => Layout::of(&rows, target)?,
        };
        // `bytes` buys pages and state slots; the null and sink chunks lie
        // past them ([`reserved_bytes`]).
        let least = RESERVED_CHUNKS + narrow(layout.shapes.len() as u64);
        let chunks = (RESERVED_CHUNKS + narrow(bytes / layout.chunk_bytes.max(1))).max(least);
        // The staging slab: the top of the stated range, just below the
        // sink, where the runtime keeps a fire's activation span. Counted
        // once, not once a frame in flight: fires reach the device in order
        // through the donated pool and each copies its own ring slot into
        // the slab at its start, so one slab is every fire's; the frames in
        // flight hold their copies in the host ring.
        let chunk_rows = layout.chunk_bytes / ROW_BYTES;
        // The own-row pages of the windowed state (CONTRACT-BC 19): a fire's
        // rows past its window buffer the ring cannot take whole while the
        // fire reads them, each laid as a window-buffer slot, after the
        // staged inputs at a page's grain.
        let (own_pages, page_rows) = match (layout.ring(), layout.window(), paging.window) {
            (Some(_), Some(window), Some(w)) => {
                if layout.chunk_bytes % window.slot_bytes != 0 {
                    return Err(Fault::Unbound {
                        what: format!(
                            "a windowed load whose chunk of {} bytes is no whole number of its \
                             {}-byte window pages: its ring pages would lie off the window \
                             buffer's grain",
                            layout.chunk_bytes, window.slot_bytes
                        ),
                    });
                }
                // a lane's own rows `r` span at most `ceil(r / T) + 1`
                // pages, `r / T + 2`: the fire's rows over `T`, two a lane
                (
                    (w.fire_pages - 1) + 2 * w.lanes.max(1),
                    window.slot_bytes / ROW_BYTES,
                )
            }
            _ => (0, 1),
        };
        let inputs_rows = u64::from(staging.rows).next_multiple_of(page_rows.max(1));
        let own_rows = u64::from(own_pages) * page_rows;
        let slab_chunks = narrow((inputs_rows + own_rows).div_ceil(chunk_rows.max(1)));
        let chunks = chunks + slab_chunks;
        // The streamed weights seated whole, in chunks of their own just
        // below the slab: the state chunks the pool would have had stay
        // below them.
        let (chunks, weight_chunks) = match streamed {
            Some(s) if !s.is_empty() => {
                let weight_chunks = s.chunks(layout.chunk_bytes);
                let chunks = chunks + weight_chunks;
                s.seat(layout.chunk_bytes, chunks - 1 - slab_chunks - weight_chunks)?;
                (chunks, weight_chunks)
            }
            _ => (chunks + extra_chunks, 0),
        };
        let slab_row = narrow(u64::from(chunks - 1 - slab_chunks) * chunk_rows);
        let act_chunks: Vec<(u32, u32)> = staging
            .rungs
            .iter()
            .map(|&(tokens, rows)| {
                let rows = if own_pages > 0 {
                    inputs_rows + own_rows
                } else {
                    u64::from(rows)
                };
                (tokens, narrow(rows.div_ceil(chunk_rows.max(1))))
            })
            .collect();
        let ring = if device.is_dry() || staging.rows == 0 {
            None
        } else {
            Some(stage::Ring::lay(device, staging.slots, staging.rows)?)
        };
        let bulk = if device.is_dry() {
            None
        } else {
            Some(stage::Ring::lay(device, BULK_SLOTS, narrow(chunk_rows))?)
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
                let order = kv_order(plane, u64::from(pool_rows))?;
                let t = handles
                    .view_ordered(root, plane.dtype, plane.width, order)
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
            windowed: Vec::new(),
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
            host_banks: Vec::new(),
            slab: (slab_row, staging.rows),
            own: (narrow(u64::from(slab_row) + inputs_rows), own_pages),
            act_chunks,
            ring,
            bulk,
            adapter_slot: adapter_slot_bytes(trace),
            readouts: Vec::new(),
        })
    }

    /// Lands each streamed bank's miss copy in pinned host memory
    /// (`Streamed::host_banks`), a root each (before the handles seal).
    pub fn land_host_banks(
        &mut self,
        device: &Device,
        handles: &Handles,
        streamed: &streamed::Streamed,
    ) -> Result<()> {
        for (param, rows, bytes) in streamed.host_banks()? {
            let all =
                u32::try_from(bytes.len() as u64 / ROW_BYTES).map_err(|_| Fault::Ceiling {
                    what: "rows of a host bank",
                    need: bytes.len() as u64 / ROW_BYTES,
                    have: u64::from(u32::MAX),
                })?;
            let buffer = device.upload_host(Dtype::U32, all, 128, &bytes)?;
            let p = u32::try_from(param).unwrap_or(u32::MAX);
            let tensor = handles.root(Root {
                source: Source::HostBank { param: p },
                dtype: Dtype::U32,
                rows: all,
                width: 128,
            });
            self.host_banks.push((p, rows, tensor, buffer));
        }
        Ok(())
    }

    /// The miss copy of streamed bank `param`: its buffer.
    #[must_use]
    pub fn host_bank(&self, param: u32) -> Option<&Buffer> {
        self.host_banks.iter().find(|b| b.0 == param).map(|b| &b.3)
    }

    /// The miss copy of streamed bank `param`: its root and the pool rows an
    /// expert's plane takes.
    #[must_use]
    pub fn host_bank_root(&self, param: u32) -> Option<(Tensor, u32)> {
        self.host_banks
            .iter()
            .find(|b| b.0 == param)
            .map(|b| (b.2, b.1))
    }

    /// The pool's whole root (u32 `[rows, 128]`).
    #[must_use]
    pub fn root(&self) -> Tensor {
        self.root
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
        // shape's atoms in a row), a chunk's worth a run (the bulk ring's
        // slot); lone slots land a chunk's worth a program run.
        let most = i64::try_from(self.units(rows)).unwrap_or(1);
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
        for batch in lone.chunks(self.units(rows)) {
            let at: Vec<i64> = batch.iter().map(|(at, _)| *at).collect();
            let bytes: Vec<u8> = batch
                .iter()
                .flat_map(|(_, run)| run[0].iter().copied())
                .collect();
            self.write_rows(device, handles, rows, at, &bytes)?;
        }
        Ok(())
    }

    /// Fills pool slots of streamed bank `param` from its pinned host copy,
    /// in the program: each `(expert, pool row)` lands that expert's plane
    /// at the row. One program a bank shape, as many experts a run as the
    /// bank holds (the rest copy nothing), so the load's seating warms what
    /// a residency later runs.
    pub fn fill_from_bank(
        &mut self,
        device: &Device,
        handles: &Handles,
        param: u32,
        fills: &[(u32, i64)],
    ) -> Result<()> {
        if fills.is_empty() || device.is_dry() {
            return Ok(());
        }
        let (host, rows) = self.host_bank_root(param).ok_or_else(|| Fault::Unbound {
            what: format!("the host copy of streamed bank {param}"),
        })?;
        let all = self
            .host_banks
            .iter()
            .find(|b| b.0 == param)
            .map_or(0, |b| {
                b.3.dims()
                    .ok()
                    .and_then(|d| d.first().copied())
                    .unwrap_or(0)
            });
        let width = usize::try_from(all / i64::from(rows.max(1)))
            .unwrap_or(0)
            .max(1);
        for batch in fills.chunks(width) {
            let mut from: Vec<i32> = batch
                .iter()
                .map(|&(e, _)| narrow(u64::from(e)) as i32)
                .collect();
            let mut to: Vec<i32> = batch
                .iter()
                .map(|&(_, at)| i32::try_from(at).unwrap_or(i32::MAX))
                .collect();
            from.resize(width, 0);
            to.resize(width, 0);
            let mut inputs = crate::inputs::Inputs::new();
            let from = inputs.i32s(handles, &from, 1);
            let to = inputs.i32s(handles, &to, 1);
            let count = inputs.i32s(handles, &[batch.len() as i32], 1);
            let tracer = Tracer::new(handles).with_stage(self.stage_at(inputs.rows()));
            let root = self.root;
            let w = width as i64;
            tracer.emit(&mut |cx| {
                let pool = cx.read(root)?;
                let bank = cx.read(host)?;
                let from = cx.read(from)?;
                let from = cx.reshape(from, &[w])?;
                let to = cx.read(to)?;
                let to = cx.reshape(to, &[w])?;
                let count = cx.read(count)?;
                let count = cx.reshape(count, &[])?;
                let pool = cx.copy_from_host(pool, bank, from, to, count, i64::from(rows))?;
                cx.write(root, pool)
            })?;
            crate::exec::maintain(device, tracer, self, &inputs)?;
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
                    // the window buffer (CONTRACT-BC 19, 21)
                    Kind::Window { .. } => SlotKind::List {
                        tokens: shape.tokens,
                        rate: 1,
                        window: None,
                    },
                    Kind::Fixed { pieces } | Kind::Ring { pieces } => SlotKind::Fixed { pieces },
                },
                slot_bytes: shape.slot_bytes,
                slots_per_chunk: self.layout.slots_per_chunk(shape),
                sets_per_working_set: 1,
                rows: shape.rows.clone(),
                pinned: None,
                role: match shape.kind {
                    Kind::Rs => SlotRole::RsBuffer,
                    Kind::Window { .. } => SlotRole::WindowBuffer,
                    _ => SlotRole::State,
                },
            })
            .chain(std::iter::once(SlotShape {
                id: layout::CHANNEL,
                kind: SlotKind::Fixed { pieces: 1 },
                slot_bytes: crate::guest::CHANNEL_SLOT,
                slots_per_chunk: narrow(self.layout.chunk_bytes / crate::guest::CHANNEL_SLOT),
                sets_per_working_set: 1,
                rows: Vec::new(),
                pinned: None,
                role: SlotRole::Channel,
            }))
            .chain((self.adapter_slot > 0).then(|| SlotShape {
                id: layout::ADAPTER,
                kind: SlotKind::Fixed { pieces: 1 },
                slot_bytes: self.adapter_slot,
                slots_per_chunk: narrow(self.layout.chunk_bytes / self.adapter_slot),
                sets_per_working_set: 1,
                rows: Vec::new(),
                pinned: None,
                role: SlotRole::Adapter,
            }))
            .collect();
        ChunkFacts {
            span_backed_on_demand: false,
            act_span_chunks: None,
            chunk_bytes: self.layout.chunk_bytes,
            // The sink lies past the stated range: the runtime never names it.
            total_chunks: self.sink_chunk(),
            segment_chunks: self.sink_chunk(),
            segments: SegmentFacts {
                committed: 1,
                max: 1,
                growth: Growth::Fixed,
            },
            reserved: vec![ReservedRun {
                what: Reserved::Null,
                first: NULL_CHUNK,
                chunks: 1,
            }],
            shapes,
            act_chunks: self.act_chunks.clone(),
            // Each rung's replay as prepare checks it (`crate::rs::rung_replay`).
            act_replay: self
                .act_chunks
                .iter()
                .map(|&(tokens, _)| crate::rs::rung_replay(tokens))
                .collect(),
            rs_compact: false,
            rs_scratch_per_fire: false,
            eager_rungs: Vec::new(),
            held_bytes: 0,
            outside_bytes: 0,
            host_chunks: 0,
            disk_chunks: 0,
            disk_persistent: false,
            disk_kept: false,
            weights: Vec::new(),
            // the record's bytes where it holds the rings
            record_bytes: if self.windowed.is_empty() {
                0
            } else {
                self.state_slot_bytes()
            },
            windowed: self.windowed.clone(),
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

    /// The pool's rows (u32 `[rows, 128]`).
    #[must_use]
    pub fn rows(&self) -> i64 {
        i64::try_from(u64::from(self.chunks) * self.layout.chunk_bytes / ROW_BYTES)
            .unwrap_or(i64::MAX)
    }

    /// The sink chunk's first row: where a padding lane's guest cells go.
    #[must_use]
    pub fn sink_row(&self) -> u32 {
        narrow(u64::from(self.sink_chunk()) * self.layout.chunk_bytes / ROW_BYTES)
    }

    /// The pool buffer itself, for programs that take it beside the
    /// tracer's (guest stages carrying cells).
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    pub fn buffer_mut(&mut self) -> &mut Option<Buffer> {
        &mut self.buffer
    }

    /// The pool row `Channel` slot ref `slot` starts at
    /// ([`layout::CHANNEL`], 1 KiB slots).
    #[must_use]
    pub fn channel_row(&self, slot: u32) -> Option<u32> {
        self.carved_row(crate::guest::CHANNEL_SLOT, slot)
    }

    /// The pool row `Adapter` slot ref `slot` starts at
    /// ([`layout::ADAPTER`]).
    #[must_use]
    pub fn adapter_row(&self, slot: u32) -> Option<u32> {
        (self.adapter_slot > 0)
            .then(|| self.carved_row(self.adapter_slot, slot))
            .flatten()
    }

    /// An `Adapter` slot's bytes (0: the model registers no adapter bank).
    #[must_use]
    pub fn adapter_slot(&self) -> u64 {
        self.adapter_slot
    }

    /// The first pool row of slot ref `slot` of a shape carved `bytes` a
    /// slot out of whole chunks (`Channel`, `Adapter`).
    fn carved_row(&self, bytes: u64, slot: u32) -> Option<u32> {
        let per = u32::try_from(self.layout.chunk_bytes / bytes.max(1)).ok()?;
        let chunk = self.table.physical(slot / per.max(1))?;
        if chunk == NULL_CHUNK || chunk >= self.sink_chunk() {
            return None;
        }
        let at = u64::from(chunk) * self.layout.chunk_bytes + u64::from(slot % per.max(1)) * bytes;
        u32::try_from(at / ROW_BYTES).ok()
    }

    /// Where a fire whose staged image takes `used` rows stages it. On a
    /// shell the load did not size (the stage-only pass's), the slab is the
    /// image's own rows.
    #[must_use]
    pub fn stage_at(&self, used: u32) -> crate::trace::StageAt {
        let (slab_row, rows) = self.slab;
        crate::trace::StageAt {
            slab_rows: if rows == 0 { used } else { rows },
            used_rows: used,
            slab_row,
        }
    }

    /// The slab's first row and rows, and the ring's slots.
    #[must_use]
    pub fn staging(&self) -> (u32, u32, u32) {
        let slots = self.ring.as_ref().map_or(0, |r| r.shape().0);
        (self.slab.0, self.slab.1, slots)
    }

    /// The units of `rows` pool rows one maintenance program run moves: a
    /// chunk's worth, what a bulk ring slot holds and a program's index
    /// list.
    #[must_use]
    pub fn units(&self, rows: u64) -> usize {
        usize::try_from(self.layout.chunk_bytes / ROW_BYTES / rows.max(1))
            .unwrap_or(1)
            .max(1)
    }

    /// The rows a bulk ring slot holds: a chunk's.
    #[must_use]
    pub fn bulk_rows(&self) -> u32 {
        narrow(self.layout.chunk_bytes / ROW_BYTES)
    }

    /// The bulk ring's slots and bytes.
    #[must_use]
    pub fn bulk_ring(&self) -> (u32, u64) {
        self.bulk.as_ref().map_or((0, 0), |r| {
            let (slots, rows) = r.shape();
            (slots, u64::from(slots) * u64::from(rows) * ROW_BYTES)
        })
    }

    /// The bulk ring, where the load laid one.
    pub fn bulk(&mut self) -> Option<&mut stage::Ring> {
        self.bulk.as_mut()
    }

    /// The bulk ring's slot `at`.
    #[must_use]
    pub fn bulk_buffer(&self, at: usize) -> Option<&Buffer> {
        self.bulk.as_ref().map(|r| r.buffer(at))
    }

    /// The staging ring's slot `at`.
    #[must_use]
    pub fn ring_buffer(&self, at: usize) -> Option<&Buffer> {
        self.ring.as_ref().map(|r| r.buffer(at))
    }

    /// The staging ring, where the load laid one.
    pub fn ring(&mut self) -> Option<&mut stage::Ring> {
        self.ring.as_mut()
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
            rs: self
                .layout
                .rs()
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
    /// (`windowed`) or in the kv list read. A load with no window list (its
    /// window holds the whole context, so it keeps none) reads windowed rows
    /// from the kv list's table.
    #[must_use]
    pub fn table_of(&self, windowed: bool) -> usize {
        usize::from(windowed && self.layout.kv().is_some() && self.layout.window().is_some())
    }

    #[must_use]
    pub fn has_windowed(&self) -> bool {
        self.layout.window().is_some()
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        // recurrent state, not a windowed load's rings: what the fire's
        // padding, the ladder and snapshots weigh
        self.layout
            .shapes
            .iter()
            .any(|s| matches!(s.kind, Kind::Fixed { .. }))
    }

    /// States the windowed layers of `recipe` (`poem_compiler::chunk::
    /// plan_windowed`'s), each its ring's bytes in the record: its rows of
    /// the ring pages as this store lays them, the padding the last's.
    pub fn set_windowed(&mut self, recipe: &poem_compiler::chunk::WindowedRecipe) {
        let Some(ring) = self.layout.ring() else {
            self.windowed = Vec::new();
            return;
        };
        // each layer's rows of the ring pages, the slot's padding the last's:
        // the record's ring bytes as this store lays them
        let ring_bytes = u64::from(ring.pieces()) * ring.slot_bytes;
        let rows = u64::from(ring.pieces()) * u64::from(self.paging.page_size);
        let mut windowed: Vec<engine::chunk::WindowedLayer> = recipe
            .layers
            .iter()
            .filter_map(|l| {
                Some(engine::chunk::WindowedLayer {
                    layer: l.layer?,
                    window: l.window,
                    bytes: rows * l.per_token,
                })
            })
            .collect();
        let laid: u64 = windowed.iter().map(|l| l.bytes).sum();
        if let Some(last) = windowed.last_mut() {
            last.bytes += ring_bytes.saturating_sub(laid);
        }
        self.windowed = windowed;
    }

    /// The windowed state's own-row pages, as grain indices of the window
    /// buffer's grain (the ids a composed window table names): none where
    /// the load holds no rings.
    #[must_use]
    pub fn own_grains(&self) -> Vec<u32> {
        let Some(window) = self.layout.window() else {
            return Vec::new();
        };
        let grain = self.layout.grain(window).max(1);
        let (first, pages) = self.own;
        (0..u64::from(pages))
            .map(|k| narrow((u64::from(first) * ROW_BYTES + k * window.slot_bytes) / grain))
            .collect()
    }

    /// The pool row a page of the window buffer's grain starts at.
    #[must_use]
    pub fn page_row(&self, grain: u32) -> i64 {
        let g = self
            .layout
            .window()
            .map_or(ROW_BYTES, |w| self.layout.grain(w));
        i64::try_from(u64::from(grain) * g / ROW_BYTES).unwrap_or(i64::MAX)
    }

    /// The pool rows a window-buffer page spans.
    #[must_use]
    pub fn page_rows(&self) -> u64 {
        self.layout.window().map_or(0, |w| w.slot_bytes / ROW_BYTES)
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

    /// Lays each readout output's ring: `depth` planes of `(root, dtype,
    /// rows, width)` each, zeros.
    pub fn lay_readouts(
        &mut self,
        device: &Device,
        rings: &[(Tensor, Dtype, u32, u32)],
        depth: usize,
    ) -> Result<()> {
        for &(root, dtype, rows, width) in rings {
            let mut slots = Vec::with_capacity(depth);
            for _ in 0..depth.max(1) {
                slots.push(device.zeros_or_dry(dtype, rows, width)?.map(std::sync::Arc::new));
            }
            self.readouts.push(ReadoutRing {
                root,
                dtype,
                rows,
                width,
                slots,
                at: 0,
            });
        }
        Ok(())
    }

    #[must_use]
    pub fn readouts(&self) -> &[ReadoutRing] {
        &self.readouts
    }

    /// The planes each readout ring holds.
    #[must_use]
    pub fn readout_depth(&self) -> usize {
        self.readouts.first().map_or(0, |ring| ring.slots.len())
    }

    /// The bytes every readout ring takes on the device.
    #[must_use]
    pub fn readout_bytes(&self) -> u64 {
        self.readouts
            .iter()
            .map(|r| r.slots.len() as u64 * u64::from(r.rows) * u64::from(r.width) * 4)
            .sum()
    }

    /// The next plane of readout ring `index`, for a fire to write into and
    /// return. Refused when a reader still holds it: a fire's readouts are
    /// read before the ring comes round (the fires that can be outstanding,
    /// `Runahead::staging_depth`), never past it.
    pub fn readout_take(&mut self, device: &Device, index: u16) -> Result<Buffer> {
        let _ = device;
        let ring = self
            .readouts
            .get_mut(usize::from(index))
            .ok_or_else(|| Fault::Unbound {
                what: format!("readout ring {index}, which this load did not lay"),
            })?;
        let depth = ring.slots.len().max(1);
        let next = (ring.at + 1) % depth;
        let plane = ring.slots[next].take().ok_or_else(|| Fault::Unbound {
            what: format!("readout ring {index}'s plane {next}, which a fire still holds"),
        })?;
        match std::sync::Arc::try_unwrap(plane) {
            Ok(buffer) => {
                ring.at = next;
                Ok(buffer)
            }
            Err(held) => {
                ring.slots[next] = Some(held);
                Err(Fault::Program {
                    at: "store::readout_take",
                    why: format!(
                        "readout ring {index}'s plane {next}, which a fire {depth} fires back is \
                         still read from: a fire's readouts are read before {depth} more fires \
                         run"
                    ),
                })
            }
        }
    }

    /// Refused when readout ring `index`'s next plane is still read: asked
    /// before a fire takes anything, so a refused fire leaves every buffer
    /// where it was.
    pub fn readout_ready(&self, index: u16) -> Result<()> {
        let ring = self
            .readouts
            .get(usize::from(index))
            .ok_or_else(|| Fault::Unbound {
                what: format!("readout ring {index}, which this load did not lay"),
            })?;
        let depth = ring.slots.len().max(1);
        let next = (ring.at + 1) % depth;
        match &ring.slots[next] {
            Some(plane) if std::sync::Arc::strong_count(plane) == 1 => Ok(()),
            Some(_) => Err(Fault::Program {
                at: "store::readout_ready",
                why: format!(
                    "readout ring {index}'s plane {next}, which a fire {depth} fires back is still \
                     read from: a fire's readouts are read before {depth} more fires run"
                ),
            }),
            None => Err(Fault::Unbound {
                what: format!("readout ring {index}'s plane {next}, which a fire still holds"),
            }),
        }
    }

    /// Puts back what a fire returned for readout ring `index`.
    pub fn readout_put(&mut self, index: u16, buffer: Buffer) {
        if let Some(ring) = self.readouts.get_mut(usize::from(index)) {
            ring.slots[ring.at] = Some(std::sync::Arc::new(buffer));
        }
    }

    /// The plane readout ring `index`'s last fire wrote.
    #[must_use]
    pub fn readout_current(&self, index: u16) -> Option<std::sync::Arc<Buffer>> {
        let ring = self.readouts.get(usize::from(index))?;
        ring.slots[ring.at].clone()
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

    /// Whether the load buffers recurrent rows (an rs list's buffer planes).
    #[must_use]
    pub fn buffers_rs(&self) -> bool {
        !self.rs_buffers.is_empty()
    }

    /// Every cache row's compressor planes, for a walk.
    #[must_use]
    pub fn compressors(&self) -> Vec<(u32, [Tensor; 2], [Map; 2])> {
        self.compressor.clone()
    }

    /// The pool rows grain index `q` of shape `id` starts at.
    /// Pool rows a slot of shape `id` takes: a state shape's, or a
    /// `Channel` slot's ([`layout::CHANNEL`]).
    fn slot_rows(&self, id: ShapeId) -> Result<u64> {
        if id == layout::CHANNEL {
            return Ok(crate::guest::CHANNEL_SLOT / ROW_BYTES);
        }
        if id == layout::ADAPTER && self.adapter_slot > 0 {
            return Ok(self.adapter_slot / ROW_BYTES);
        }
        Ok(self.shape(id)?.slot_bytes / ROW_BYTES)
    }

    /// The pool row slot `slot` of shape `id` starts at.
    fn slot_row(&self, id: ShapeId, slot: u32) -> Result<i64> {
        if id == layout::CHANNEL {
            return self.channel_row(slot).map(i64::from).ok_or_else(|| Fault::Unbound {
                what: format!("Channel slot {slot}, which lies past the pool"),
            });
        }
        if id == layout::ADAPTER {
            return self.adapter_row(slot).map(i64::from).ok_or_else(|| Fault::Unbound {
                what: format!("Adapter slot {slot}, which lies past the pool"),
            });
        }
        self.rows_at(id, self.grain(id, slot)?)
    }

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
        for (rows, mut all) in by {
            all.sort_unstable();
            all.dedup();
            self.zero_rows(device, handles, rows, &all)?;
        }
        Ok(())
    }

    /// Zeros blocks of `rows` pool rows from each of `all`, a chunk's worth
    /// a program run ([`Pools::units`]).
    fn zero_rows(&mut self, device: &Device, handles: &Handles, rows: u64, all: &[i64]) -> Result<()> {
        let cap = self.units(rows);
        for chunk in all.chunks(cap) {
            let (inputs, list, count) = listed(handles, chunk, cap);
            let tracer = Tracer::new(handles).with_stage(self.stage_at(inputs.rows()));
            let root = self.root;
            tracer.emit(&mut |cx| {
                let whole = cx.read(root)?;
                let list = cx.read(list)?;
                let count = cx.read(count)?;
                let count = cx.reshape(count, &[])?;
                let zero = cx.const_i(Elem::U32, 0, &[rows as i64, 128]);
                let whole = each(cx, whole, count, |f, i, pool| {
                    let row = entry(f, list, i, 0)?;
                    let col = f.const_i(Elem::I32, 0, &[]);
                    f.dynamic_update_slice(pool, zero, &[row, col])
                })?;
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
            let rows = self.slot_rows(id)?;
            let from = self.slot_row(id, src)?;
            let to = self.slot_row(id, dst)?;
            spans.push((rows, from, to));
        }
        self.copy_rows(device, handles, &spans)
    }

    /// Whole-page copies `(to, from)` of pages of the window buffer's grain
    /// (the windowed state's fills and folds, `serve::ring`): the slot copy
    /// at a window-buffer slot's size, which the load warms.
    pub fn copy_pages(
        &mut self,
        device: &Device,
        handles: &Handles,
        pages: &[(u32, u32)],
    ) -> Result<()> {
        let rows = self.page_rows();
        let spans: Vec<(u64, i64, i64)> = pages
            .iter()
            .filter(|(to, from)| to != from)
            .map(|&(to, from)| (rows, self.page_row(from), self.page_row(to)))
            .collect();
        if spans.is_empty() {
            return Ok(());
        }
        self.copy_rows(device, handles, &spans)
    }

    /// Runs, at load, every maintenance program a fire, a snapshot or a
    /// park lands later: the slot copy, write and zero, one program a slot
    /// size whatever the count, at each slot size the pool states (its state shapes, the
    /// `Channel` slots, a whole chunk for an evacuation). Each lands the
    /// null chunk's rows (zeros) onto themselves.
    pub fn warm_copies(&mut self, device: &Device, handles: &Handles) -> Result<()> {
        let mut sizes: std::collections::BTreeSet<u64> = self
            .layout
            .shapes
            .iter()
            .map(|shape| shape.slot_bytes / ROW_BYTES)
            .collect();
        sizes.insert(crate::guest::CHANNEL_SLOT / ROW_BYTES);
        if self.adapter_slot > 0 {
            sizes.insert(self.adapter_slot / ROW_BYTES);
        }
        sizes.insert(self.layout.chunk_bytes / ROW_BYTES);
        for rows in sizes {
            self.copy_rows(device, handles, &[(rows, 0, 0)])?;
            let zeros = vec![0u8; (rows * ROW_BYTES) as usize];
            self.write_rows(device, handles, rows, vec![0], &zeros)?;
            // A tier read's two programs: one block, and a batch.
            self.read_rows(device, handles, rows, &[0])?;
            self.read_rows(device, handles, rows, &vec![0; read_batch(rows)])?;
        }
        // The zero a fresh record starts from, at each fixed shape's size,
        // landed on the sink.
        let sink = i64::from(self.sink_row());
        let fixed: std::collections::BTreeSet<u64> = self
            .fixed_shapes()
            .filter_map(|(id, _)| self.shape(id).ok().map(|s| s.slot_bytes / ROW_BYTES))
            .collect();
        for rows in fixed {
            self.zero_rows(device, handles, rows, &[sink])?;
        }
        Ok(())
    }

    /// Copies `rows` pool rows from each span's first to its second, a
    /// program a run of equal sizes (a chunk's worth a run), in order.
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
            let cap = self.units(rows);
            for chunk in spans[at..end].chunks(cap) {
                let flat: Vec<i64> = chunk
                    .iter()
                    .flat_map(|&(_, from, to)| [from, to])
                    .collect();
                let (inputs, list, count) = listed(handles, &flat, 2 * cap);
                let tracer = Tracer::new(handles).with_stage(self.stage_at(inputs.rows()));
                let root = self.root;
                tracer.emit(&mut |cx| {
                    let whole = cx.read(root)?;
                    let list = cx.read(list)?;
                    let count = cx.read(count)?;
                    let count = cx.reshape(count, &[])?;
                    let two = cx.const_i(Elem::I32, 2, &[]);
                    let pairs = cx.div(count, two)?;
                    let whole = each(cx, whole, pairs, |f, i, pool| {
                        let from = entry(f, list, i, 2)?;
                        let two = f.const_i(Elem::I32, 2, &[]);
                        let one = f.const_i(Elem::I32, 1, &[]);
                        let at = f.mul(i, two)?;
                        let at = f.add(at, one)?;
                        let to = pick(f, list, at)?;
                        let col = f.const_i(Elem::I32, 0, &[]);
                        let block = f.dynamic_slice(pool, &[from, col], &[rows as i64, 128])?;
                        f.dynamic_update_slice(pool, block, &[to, col])
                    })?;
                    cx.write(root, whole)
                })?;
                crate::exec::maintain(device, tracer, self, &inputs)?;
            }
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
            let tracer = Tracer::new(handles).with_stage(self.stage_at(inputs.rows()));
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
        _handles: &Handles,
        id: ShapeId,
        slots: &[u32],
    ) -> Result<Vec<u8>> {
        if slots.is_empty() || device.is_dry() {
            return Ok(Vec::new());
        }
        let rows = self.slot_rows(id)?;
        let at: Vec<i64> = slots
            .iter()
            .map(|&slot| self.slot_row(id, slot))
            .collect::<Result<_>>()?;
        self.read_rows(device, _handles, rows, &at)
    }

    /// The bytes of blocks of `rows` pool rows each at pool rows `at`, in
    /// order: a batch of blocks gathered on the device into one buffer and
    /// copied to the host whole, [`READ_BATCH_BYTES`] a batch; a tail of
    /// fewer than a quarter batch a block at a time. Two programs a block
    /// size, whatever the count (warmed at load, [`Pools::warm_copies`]). A
    /// copy a block straight off the pool (`PJRT_Buffer_CopyRawToHost`)
    /// waits ~7 ms whatever its size (250 MB/s for a kv page on a v6e); a
    /// whole buffer copies at the link's speed.
    fn read_rows(
        &mut self,
        device: &Device,
        handles: &Handles,
        rows: u64,
        at: &[i64],
    ) -> Result<Vec<u8>> {
        let per = (rows * ROW_BYTES) as usize;
        let batch = read_batch(rows);
        let mut out = Vec::with_capacity(at.len() * per);
        let whole = at.len() / batch * batch;
        let tail = &at[whole..];
        let mut runs: Vec<(&[i64], usize)> =
            at[..whole].chunks(batch).map(|c| (c, batch)).collect();
        if tail.len() * 4 >= batch {
            runs.push((tail, batch));
        } else {
            runs.extend(tail.chunks(1).map(|c| (c, 1)));
        }
        for (chunk, cap) in runs {
            let mut inputs = crate::inputs::Inputs::new();
            let (list, _) = list_into(&mut inputs, handles, chunk, cap);
            let tracer = Tracer::new(handles).with_stage(self.stage_at(inputs.rows()));
            let root = self.root;
            let mut block = None;
            tracer.emit(&mut |cx| {
                let whole = cx.read(root)?;
                let list = cx.read(list)?;
                let f = cx.func();
                let got = f.gather(
                    whole,
                    list,
                    &kernels_xla::hlo::GatherDims {
                        offset_dims: vec![1, 2],
                        start_index_map: vec![0],
                        index_vector_dim: 1,
                        ..kernels_xla::hlo::GatherDims::default()
                    },
                    &[rows as i64, 128],
                )?;
                block = Some(f.reshape(got, &[cap as i64 * rows as i64, 128])?);
                cx.write(root, whole)
            })?;
            let block = block.ok_or_else(|| Fault::Unbound {
                what: "the gathered blocks, which the read program did not trace".to_string(),
            })?;
            let outs = crate::exec::run(device, tracer, &[block], None, self, &inputs, false)?;
            // A dry device traces the program and runs nothing.
            if device.is_dry() {
                continue;
            }
            let buffer = outs.into_iter().next().ok_or_else(|| Fault::Unbound {
                what: "the gathered blocks, which the read program did not return".to_string(),
            })?;
            let bytes = buffer.download()?;
            out.extend_from_slice(&bytes[..chunk.len() * per]);
        }
        Ok(out)
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
        let rows = self.slot_rows(id)?;
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
            .map(|&slot| self.slot_row(id, slot))
            .collect::<Result<_>>()?;
        self.write_rows(device, handles, rows, at, bytes)
    }

    /// Lands `bytes`, blocks of `rows` pool rows each, at pool rows `at`:
    /// a chunk's worth a run, each written into the bulk ring's next slot
    /// (pinned host memory, laid at load) and landed from it by the program
    /// a block at a time. Nothing is uploaded and nothing allocated.
    fn write_rows(
        &mut self,
        device: &Device,
        handles: &Handles,
        rows: u64,
        at: Vec<i64>,
        bytes: &[u8],
    ) -> Result<()> {
        if at.is_empty() || device.is_dry() {
            return Ok(());
        }
        let cap = self.units(rows);
        let per = (rows * ROW_BYTES) as usize;
        let ring_rows = self.bulk_rows();
        for (k, chunk) in at.chunks(cap).enumerate() {
            let data = bytes[k * cap * per..(k * cap + chunk.len()) * per].to_vec();
            let mut inputs = crate::inputs::Inputs::new();
            let landed = inputs.bulk(handles, ring_rows, data);
            let (list, count) = list_into(&mut inputs, handles, chunk, cap);
            let tracer = Tracer::new(handles).with_stage(self.stage_at(inputs.rows()));
            let root = self.root;
            tracer.emit(&mut |cx| {
                let whole = cx.read(root)?;
                let all = cx.read(landed)?;
                let list = cx.read(list)?;
                let count = cx.read(count)?;
                let count = cx.reshape(count, &[])?;
                let whole = each(cx, whole, count, |f, i, pool| {
                    let row = entry(f, list, i, 0)?;
                    let col = f.const_i(Elem::I32, 0, &[]);
                    let by = f.const_i(Elem::I32, rows as i64, &[]);
                    let from = f.mul(i, by)?;
                    let block = f.dynamic_slice(all, &[from, col], &[rows as i64, 128])?;
                    let block = f.to_device(block);
                    f.dynamic_update_slice(pool, block, &[row, col])
                })?;
                cx.write(root, whole)
            })?;
            crate::exec::maintain(device, tracer, self, &inputs)?;
        }
        Ok(())
    }

    /// Lands `planes`, each `(Adapter slot ref, bytes)`, in their slots:
    /// one adapter's planes, written once at the bind that lands it.
    pub fn write_adapter(
        &mut self,
        device: &Device,
        handles: &Handles,
        planes: &[(u32, &[u8])],
    ) -> Result<()> {
        let rows = self.adapter_slot / ROW_BYTES;
        for &(slot, bytes) in planes {
            let at = self.slot_row(layout::ADAPTER, slot)?;
            let mut whole = bytes.to_vec();
            whole.resize(self.adapter_slot as usize, 0);
            self.write_rows(device, handles, rows, vec![at], &whole)?;
        }
        Ok(())
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

/// The bytes a tier read gathers on the device and copies out at once.
const READ_BATCH_BYTES: u64 = 64 << 20;

/// The most blocks a tier read's batch lists: its row list is a staged
/// input, which a small load's slab holds a few hundred rows of.
const READ_BATCH_BLOCKS: usize = 256;

/// The blocks of `rows` pool rows a tier read's batch holds.
fn read_batch(rows: u64) -> usize {
    usize::try_from(READ_BATCH_BYTES / (rows * ROW_BYTES).max(1))
        .unwrap_or(1)
        .clamp(1, READ_BATCH_BLOCKS)
}

/// `values`, padded with zeros to `cap` entries, and how many there are:
/// the inputs of a maintenance program, so its text names how many entries
/// it can hold and not how many it moves, nor which: one program a size
/// serves every count.
fn listed(handles: &Handles, values: &[i64], cap: usize) -> (crate::inputs::Inputs, Tensor, Tensor) {
    let mut inputs = crate::inputs::Inputs::new();
    let (list, count) = list_into(&mut inputs, handles, values, cap);
    (inputs, list, count)
}

/// [`listed`] into `inputs`.
fn list_into(
    inputs: &mut crate::inputs::Inputs,
    handles: &Handles,
    values: &[i64],
    cap: usize,
) -> (Tensor, Tensor) {
    let mut padded: Vec<i32> = values
        .iter()
        .map(|&v| i32::try_from(v).unwrap_or(i32::MAX))
        .collect();
    padded.resize(cap.max(values.len()), 0);
    let list = inputs.i32s(handles, &padded, 1);
    let count = inputs.i32s(handles, &[values.len() as i32], 1);
    (list, count)
}

/// `body(i, pool)` for `i` in `0..count` (an i32 scalar), the pool carried:
/// one loop, whatever the count.
fn each(
    cx: &mut crate::emit::Cx<'_>,
    whole: kernels_xla::hlo::Val,
    count: kernels_xla::hlo::Val,
    body: impl FnOnce(&mut kernels_xla::hlo::Func, kernels_xla::hlo::Val, kernels_xla::hlo::Val) -> kernels_xla::hlo::Built<kernels_xla::hlo::Val>,
) -> std::result::Result<kernels_xla::hlo::Val, kernels_xla::Error> {
    let zero = cx.const_i(Elem::I32, 0, &[]);
    let out = cx.while_loop(
        &[zero, whole],
        |f, a| f.compare(kernels_xla::hlo::Cmp::Lt, a[0], count),
        |f, a| {
            let one = f.const_i(Elem::I32, 1, &[]);
            let next = f.add(a[0], one)?;
            let pool = body(f, a[0], a[1])?;
            Ok(vec![next, pool])
        },
    )?;
    Ok(out[1])
}

/// Entry `i · stride` of the i32 list `list` (`stride` 0: entry `i`).
fn entry(
    f: &mut kernels_xla::hlo::Func,
    list: kernels_xla::hlo::Val,
    i: kernels_xla::hlo::Val,
    stride: i64,
) -> kernels_xla::hlo::Built<kernels_xla::hlo::Val> {
    let at = if stride == 0 {
        i
    } else {
        let by = f.const_i(Elem::I32, stride, &[]);
        f.mul(i, by)?
    };
    pick(f, list, at)
}

/// Entry `at` (an i32 scalar) of the i32 list `list`.
fn pick(
    f: &mut kernels_xla::hlo::Func,
    list: kernels_xla::hlo::Val,
    at: kernels_xla::hlo::Val,
) -> kernels_xla::hlo::Built<kernels_xla::hlo::Val> {
    let flat = f.reshape(list, &[f.ty(list).elements()])?;
    let one = f.dynamic_slice(flat, &[at], &[1])?;
    f.reshape(one, &[])
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

/// A state slot's rows and width: blocks of whole tiles where the slab's
/// shape gives them, else one row; and a slot whose bytes fill no whole
/// pool rows (a small state: a narrow conv's frame cache) laid one row
/// padded to whole pool rows, the kernels reading its live stride at the
/// row's front (`kernels_xla::lanes::slot_rows`).
fn state_block(slab: &[u64], elem: u64) -> (u64, u64) {
    let stride: u64 = slab.iter().product();
    let (per, width) = match *slab {
        [h, d1, d2] if (h * d1).is_multiple_of(8) && d2.is_multiple_of(128) => (h * d1, d2),
        [_, _] if stride.is_multiple_of(8 * 128) => (8, stride / 8),
        _ => (1, stride),
    };
    let row = layout::ROW_BYTES / elem.max(1);
    if (per * width * elem).is_multiple_of(layout::ROW_BYTES) {
        (per, width)
    } else {
        (1, stride.div_ceil(row) * row)
    }
}

fn narrow(n: u64) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// XLA's ring hook (`poem_compiler::chunk::RingLayout`): the rings are the
/// record's ring pages ([`layout::Kind::Ring`]), each laid exactly as a
/// window-buffer slot, so a kernel reads a ring page by a page id of the
/// window buffer's grain; the widest window in pages and one page more,
/// placed in the record at a page's bytes. Each layer's share is its rows
/// of those pages, the slot's padding the last layer's.
pub struct Rings;

impl poem_compiler::chunk::RingLayout for Rings {
    fn ring_region(
        &self,
        layers: &[poem_compiler::chunk::WindowedRow],
        buffer: &poem_compiler::chunk::WindowBufferSlot<'_>,
    ) -> poem_compiler::chunk::RingRegion {
        let tokens = buffer.tokens.max(1);
        let widest = layers.iter().map(|l| l.window).max().unwrap_or(0);
        let pages = u64::from(widest.div_ceil(tokens) + 1);
        let bytes = pages * buffer.bytes;
        let mut layer_bytes: Vec<u64> = layers
            .iter()
            .map(|l| pages * u64::from(tokens) * l.per_token)
            .collect();
        let shared: u64 = layer_bytes.iter().sum();
        if let Some(last) = layer_bytes.last_mut() {
            *last += bytes.saturating_sub(shared);
        }
        poem_compiler::chunk::RingRegion {
            bytes,
            atom: buffer.bytes,
            placement: poem_compiler::chunk::RingPlacement::Aligned(buffer.bytes),
            layer_bytes,
        }
    }
}
