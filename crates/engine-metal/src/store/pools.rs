//! The load's held state in one chunk pool ([`super::pool::Pool`]): every
//! cache row placed in the slot shapes of [`super::layout`], the slot refs the
//! runtime hands out translated to the physical slots kernels address, and
//! the copies, spills and evacuations that move slots.

use engine::chunk::{ChunkFacts, Evacuation, ShapeId, SlotCopy};
use engine::transfer::{KvRun, MemoryDomain};
use kernels_metal::{KvPool, RecurrentPool, StateMap, Tensor};
use poem_compiler::chunk::{ChunkRecipe, RowId, SlotLayout};
use poem_ir::{CacheRow, Dtype, Trace};

use super::layout::{FIXED, LIST, Layout, MAX_WINDOWS, RS, Slots, WINDOWS};
use super::pool::Pool;
use super::{
    Seats, Spill, TableSeat, compressed_rows, elem_bytes, elem_size, narrow, row_stride, split,
    state_dtype,
};
use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable, PoolSlabs};
use crate::store::kv::{Facts, Paging};

/// One cache row's place in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Row {
    /// A kv row: its key run and value run in a list slot (one run when the
    /// trace shares one plane), `page_size` tokens of `*_cell` bytes each.
    Kv {
        /// The list its slots lie in, so the page table it reads.
        paging: TablePaging,
        head_dim: u32,
        kv_heads: u32,
        dtype: Dtype,
        keys: u64,
        values: u64,
        values_width: u64,
        keys_cell: u64,
        values_cell: u64,
    },
    /// A state row: `stride` elements, `pieces` pieces of `piece_atoms`
    /// atoms each at `offset` in the record's slots of `band`, found through
    /// segment table `map`.
    State {
        stride: u64,
        dtype: Dtype,
        offset: u64,
        band: usize,
        pieces: u32,
        piece_atoms: u32,
        map: usize,
    },
    /// The compressor state a pool gather keeps beside kv row `row`, in
    /// that row's slots (the recipe's `RowId::Pool` run): a projection plane
    /// at `kv` and a score plane at `score`, `width` elements a token.
    Compressed {
        row: u32,
        width: u64,
        paging: TablePaging,
        kv: u64,
        score: u64,
    },
}

/// One kv row's geometry as this shell stores it.
struct KvCells {
    head_dim: u32,
    kv_heads: u32,
    dtype: Dtype,
    keys_cell: u64,
    values_cell: u64,
    values_width: u64,
    own_values: bool,
}

/// Row `index` of `trace`, a kv row, as this shell stores it; `facts`
/// restates its heads.
#[allow(clippy::too_many_arguments)]
fn kv_cells(
    index: usize,
    name: &str,
    planes: &[u64],
    dtype: Dtype,
    trace_head_dim: u32,
    facts: &Facts,
) -> Result<KvCells> {
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
                    "cache `{name}`, whose row is {width} wide while its consumers state {} \
                     heads of {}",
                    seat.kv_heads, seat.head_dim
                ),
            });
        }
    }
    // A restated seat, else the row's own head width, else the whole row a
    // head.
    let row_head_dim = (trace_head_dim != 0).then_some(trace_head_dim);
    let head_dim = restated
        .map(|seat| u64::from(seat.head_dim))
        .or(row_head_dim.map(u64::from))
        .unwrap_or(width);
    let kv_heads = restated
        .map(|seat| u64::from(seat.kv_heads))
        .or(row_head_dim.map(|d| width / u64::from(d)))
        .unwrap_or(1);
    let (head_dim, kv_heads) = if kv_heads == 0 {
        (width, 1)
    } else {
        (head_dim, kv_heads)
    };
    let block = restated.map(|seat| seat.head_dim).or(row_head_dim);
    Ok(KvCells {
        head_dim: u32::try_from(head_dim).unwrap_or(u32::MAX),
        kv_heads: u32::try_from(kv_heads).unwrap_or(u32::MAX),
        dtype,
        keys_cell: row_stride(name, dtype, width, block)?,
        values_cell: if planes.values == 0 {
            0
        } else {
            row_stride(name, dtype, planes.values, block)?
        },
        values_width: planes.values,
        own_values: !planes.shared && planes.values != 0,
    })
}

/// The rs list's slot as this shell reads it: `tokens` rows of each rs
/// plane, each of the shell's planes at `plane_at` in the slot (its
/// `tokens` rows one after another).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsRows {
    pub tokens: u32,
    pub bytes: u64,
    pub plane_at: Vec<u64>,
}

/// The list a kv row's slots lie in, so the page table it reads: the kv
/// list or window list `k` (CONTRACT-BC 19, 21).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TablePaging {
    List,
    Window(u8),
}

/// One window list as the recipe lays it: its slot bytes, the rows its
/// window retains and the tokens (`T_w`) a slot holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowRows {
    pub bytes: u64,
    pub rows: u32,
    pub tokens: u32,
}

/// How a load's rows lay into the recipe's slots, before any pool is
/// reserved.
#[derive(Debug, Clone)]
pub struct Rows {
    rows: Vec<Row>,
    /// The recipe's chunk, its list slot and its tokens, its fixed slot and
    /// the slots a record takes.
    pub chunk_bytes: u64,
    pub list_bytes: u64,
    pub tokens: u32,
    pub fixed_bytes: u64,
    pub pieces: u32,
    /// The window lists (CONTRACT-BC 19, 21), in the recipe's order; none
    /// where nothing is windowed.
    pub windows: Vec<WindowRows>,
    /// The page tables a fire stages, by the list each pages: the kv
    /// list's first (the All axis's), then the window lists'.
    pub tables: Vec<TablePaging>,
    /// Each distinct set of record slots a state row's pieces lie in.
    pub bands: Vec<Vec<u32>>,
    pub rs: Option<RsRows>,
    /// The trace's cache rows each shape lays out, in order.
    pub kv_rows: Vec<u32>,
    pub state_rows: Vec<u32>,
}

/// Why the recipe and this shell disagree, as a load refusal.
fn disagree(what: String) -> Fault {
    Fault::Unbound {
        what: format!("the on-device Metal chunk recipe, which {what}"),
    }
}

impl Rows {
    /// The state rows: one segment map each in the tables run.
    #[must_use]
    pub fn state_rows(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| matches!(row, Row::State { .. }))
            .count()
    }
}

impl Rows {
    /// The bytes one token takes across every kv row, and a working set's
    /// state rows, unaligned: an estimate before any recipe.
    pub fn estimate(trace: &Trace, page_size: u32) -> Result<(u64, u64, u64)> {
        let facts = Facts::default();
        let mut token = 0u64;
        let mut state = 0u64;
        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    head_dim,
                    ..
                } => {
                    let kv = kv_cells(index, name, planes, *dtype, *head_dim, &facts)?;
                    token += kv.keys_cell + if kv.own_values { kv.values_cell } else { 0 };
                }
                CacheRow::State {
                    name, slab, dtype, ..
                } => {
                    let dtype = state_dtype(*dtype);
                    state += slab.iter().product::<u64>() * elem_bytes(name, dtype)?;
                }
            }
        }
        let side = compressed_rows(trace)
            .iter()
            .map(|(_, width)| 2 * u64::from(page_size) * width * u64::from(elem_size(Dtype::Bf16)))
            .sum();
        Ok((token, state, side))
    }

    /// Lays `trace`'s rows where `recipe` (the compiler's on-device Metal
    /// chunk recipe, the one layout source) puts them; `facts` restates kv
    /// rows' heads, `rs` is the shell's rs buffer. Refuses where the recipe
    /// lays a row otherwise than this shell stores it.
    pub fn plan(
        trace: &Trace,
        recipe: &ChunkRecipe,
        facts: &Facts,
        rs: Option<&crate::rs::Layout>,
    ) -> Result<Rows> {
        let mut windows: Vec<WindowRows> = Vec::new();
        let mut window_runs: Vec<&[poem_compiler::chunk::TokenRun]> = Vec::new();
        for layout in recipe.windows() {
            if let SlotLayout::List {
                tokens,
                bytes,
                rows,
                window: Some(window),
            } = layout
            {
                windows.push(WindowRows {
                    bytes: *bytes,
                    rows: *window,
                    tokens: *tokens,
                });
                window_runs.push(rows.as_slice());
            }
        }
        if windows.len() > MAX_WINDOWS {
            return Err(disagree(format!(
                "keeps {} window lists ({}), where a load pages at most {MAX_WINDOWS}",
                windows.len(),
                recipe.shapes()
            )));
        }
        // The window list a row's run lies in, and its run.
        let windowed = |id: RowId| {
            window_runs
                .iter()
                .enumerate()
                .find_map(|(k, runs)| runs.iter().find(|run| run.row == id).map(|run| (k, run)))
        };
        let (tokens, list_bytes, runs) = match recipe.list() {
            Some(SlotLayout::List {
                tokens,
                bytes,
                rows,
                ..
            }) => (*tokens, *bytes, rows.as_slice()),
            _ => (0, 0, &[][..]),
        };
        let (fixed_bytes, pieces, fixed_runs) = match recipe.fixed() {
            Some(SlotLayout::Fixed {
                bytes,
                pieces_per_ws,
                rows,
                ..
            }) => (*bytes, *pieces_per_ws, rows.as_slice()),
            _ => (0, 0, &[][..]),
        };
        // Without a list the widest window slot is the kv page.
        let tokens = if list_bytes == 0 {
            windows.iter().map(|w| w.tokens).max().unwrap_or(0)
        } else {
            tokens
        };
        if let Some(window) = windows.iter().find(|w| w.tokens == 0 || w.tokens > tokens) {
            return Err(disagree(format!(
                "pages windowed rows {} tokens a window slot beside {tokens}-token kv pages, \
                 where a window list's slot holds no more than a kv page",
                window.tokens
            )));
        }
        let mut tables: Vec<TablePaging> = Vec::new();
        let mut rows = Vec::with_capacity(trace.caches.len());
        let mut bands: Vec<Vec<u32>> = Vec::new();
        let mut maps = 0usize;
        let mut kv_rows = Vec::new();
        let mut state_rows = Vec::new();
        for (index, row) in trace.caches.iter().enumerate() {
            let id = RowId::Cache(u32::try_from(index).unwrap_or(u32::MAX));
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    head_dim,
                    ..
                } => {
                    let kv = kv_cells(index, name, planes, *dtype, *head_dim, facts)?;
                    let per_token = kv.keys_cell + if kv.own_values { kv.values_cell } else { 0 };
                    // Where the row lies: a kv list slot or a window list
                    // slot, and the tokens a run holds.
                    let (paging, offset, stated, run_tokens) =
                        if let Some(run) = runs.iter().find(|run| run.row == id) {
                            (TablePaging::List, run.offset, run.per_token, tokens)
                        } else if let Some((k, run)) = windowed(id) {
                            (
                                TablePaging::Window(u8::try_from(k).unwrap_or(u8::MAX)),
                                run.offset,
                                run.per_token,
                                windows[k].tokens,
                            )
                        } else {
                            return Err(disagree(format!("lays kv row `{name}` in no slot")));
                        };
                    if per_token != stated {
                        return Err(disagree(format!(
                            "lays kv row `{name}` at {stated} bytes a token, where this shell \
                             stores {per_token}"
                        )));
                    }
                    if !tables.contains(&paging) {
                        tables.push(paging);
                    }
                    let keys = offset;
                    let values = if kv.own_values {
                        keys + u64::from(run_tokens) * kv.keys_cell
                    } else {
                        keys
                    };
                    kv_rows.push(u32::try_from(index).unwrap_or(u32::MAX));
                    rows.push(Row::Kv {
                        paging,
                        head_dim: kv.head_dim,
                        kv_heads: kv.kv_heads,
                        dtype: kv.dtype,
                        keys,
                        values,
                        values_width: kv.values_width,
                        keys_cell: kv.keys_cell,
                        values_cell: kv.values_cell,
                    });
                }
                CacheRow::State {
                    name, slab, dtype, ..
                } => {
                    let stride: u64 = slab.iter().product();
                    let dtype = state_dtype(*dtype);
                    let bytes = stride * elem_bytes(name, dtype)?;
                    let run = fixed_runs.iter().find(|run| run.row == id).ok_or_else(|| {
                        disagree(format!("lays state row `{name}` in no fixed slot"))
                    })?;
                    let atom = recipe.atoms.get(&id).ok_or_else(|| {
                        disagree(format!("states no atoms of state row `{name}`"))
                    })?;
                    if atom.bytes * atom.extent != bytes {
                        return Err(disagree(format!(
                            "sizes state row `{name}` at {} bytes, where this shell stores {bytes} \
                             ({dtype:?})",
                            atom.bytes * atom.extent
                        )));
                    }
                    if run.offset % u64::from(elem_size(dtype)) != 0 {
                        return Err(disagree(format!(
                            "lays state row `{name}` at byte {}, inside an element",
                            run.offset
                        )));
                    }
                    let band = match bands.iter().position(|band| *band == run.slots) {
                        Some(band) => band,
                        None => {
                            bands.push(run.slots.clone());
                            bands.len() - 1
                        }
                    };
                    state_rows.push(u32::try_from(index).unwrap_or(u32::MAX));
                    rows.push(Row::State {
                        stride,
                        dtype,
                        offset: run.offset,
                        band,
                        pieces: run.pieces,
                        piece_atoms: if run.pieces <= 1 {
                            u32::MAX
                        } else {
                            u32::try_from(run.atoms_per_piece).unwrap_or(u32::MAX)
                        },
                        map: maps,
                    });
                    maps += 1;
                }
            }
        }
        let rs = match rs {
            None => None,
            Some(layout) => Some(rs_rows(recipe, layout)?),
        };
        for (row, width) in compressed_rows(trace) {
            let id = RowId::Pool(row);
            let (paging, run, run_tokens) = if let Some(run) = runs.iter().find(|run| run.row == id)
            {
                (TablePaging::List, run, tokens)
            } else if let Some((k, run)) = windowed(id) {
                (
                    TablePaging::Window(u8::try_from(k).unwrap_or(u8::MAX)),
                    run,
                    windows[k].tokens,
                )
            } else {
                return Err(disagree(format!(
                    "keeps no compressor state beside kv row {row}, which a pool gather keys by"
                )));
            };
            let plane = width * u64::from(elem_size(Dtype::Bf16));
            if run.per_token != 2 * plane {
                return Err(disagree(format!(
                    "keeps kv row {row}'s compressor state at {} bytes a token, where this \
                     shell keeps two {plane}-byte planes",
                    run.per_token
                )));
            }
            rows.push(Row::Compressed {
                row,
                width,
                paging,
                kv: run.offset,
                score: run.offset + u64::from(run_tokens) * plane,
            });
        }
        Ok(Rows {
            rows,
            chunk_bytes: recipe.chunk_bytes,
            list_bytes,
            tokens,
            fixed_bytes,
            pieces,
            windows,
            // The kv list's table first (the All axis reads it), then the
            // window lists' in order.
            tables: {
                tables.sort_by_key(|paging| match paging {
                    TablePaging::List => 0,
                    TablePaging::Window(k) => 1 + u32::from(*k),
                });
                tables
            },
            bands,
            rs,
            kv_rows,
            state_rows,
        })
    }
}

/// The rs planes' places in an rs-list slot, read off the recipe's
/// `RsLayout`: each rs row's run, its planes one after another, each plane
/// `tokens` rows; the shell's planes matched by the values they buffer.
fn rs_rows(recipe: &ChunkRecipe, layout: &crate::rs::Layout) -> Result<RsRows> {
    let rs = &recipe.rs;
    if rs.is_empty() {
        return Err(disagree(
            "states no rs list, where this shell buffers rs rows".to_string(),
        ));
    }
    let tokens = u64::from(rs.tokens);
    let mut plane_at: Vec<Option<u64>> = vec![None; layout.planes.len()];
    for (id, row) in &rs.rows {
        let run = rs
            .run(*id)
            .ok_or_else(|| disagree(format!("lays no run for rs row {id:?}")))?;
        for (at, plane) in row.planes.iter().enumerate() {
            let Some(&mine) = layout.in_of.get(&plane.value.0) else {
                return Err(disagree(format!(
                    "buffers value {} in rs row {id:?}, which this shell does not buffer",
                    plane.value.0
                )));
            };
            if layout.planes[mine].row_bytes != plane.bytes {
                return Err(disagree(format!(
                    "buffers value {} at {} bytes a token, where this shell stores {}",
                    plane.value.0, plane.bytes, layout.planes[mine].row_bytes
                )));
            }
            let offset = run.offset + row.plane_at(at, tokens);
            match plane_at[mine] {
                Some(held) if held != offset => {
                    return Err(disagree(format!(
                        "lays this shell's rs plane of value {} at two places",
                        plane.value.0
                    )));
                }
                _ => plane_at[mine] = Some(offset),
            }
        }
    }
    let plane_at = plane_at
        .into_iter()
        .enumerate()
        .map(|(at, offset)| {
            offset.ok_or_else(|| {
                let plane = &layout.planes[at];
                disagree(format!(
                    "keeps no rs row for the {:?} plane of cache row {} this shell buffers",
                    plane.role, plane.cache
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RsRows {
        tokens: rs.tokens,
        bytes: rs.bytes,
        plane_at,
    })
}

/// File fills with no host address: `(to, bytes)` for the pool to write.
type Unhosted = (u64, Vec<u8>);

#[derive(Debug)]
pub struct Pools {
    paging: Paging,
    rows: Vec<Row>,
    pool: Pool,
    layout: Layout,
    /// Every state row's segment table, one after another, in the tables
    /// run.
    maps: Option<crate::inputs::Region>,
    /// Each buffer's address less the first's, one ulong a buffer, where
    /// the range spans more than one: what kernels reach a page in another
    /// buffer by (`kv_page.h`'s `pie_fc_part_table`). A buffer of its own,
    /// outside the pool: fixed for the load, so nothing written into the
    /// pool can move a page.
    deltas: Option<Buffer>,
    /// Where the next of the engine's own tables lies in the tables run
    /// (`Reserved::Tables`), and where the run ends: range byte offsets.
    table_run: (u64, u64),
    shape_rows: [Vec<u32>; 2],
    /// Each distinct set of record slots a state row's pieces lie in.
    bands: Vec<Vec<u32>>,
    rs: Option<RsRows>,
    /// The page tables a fire stages, by the list each pages.
    tables: Vec<TablePaging>,
    /// The disk tier (CONTRACT-BC 22): one slot file of disk chunks, each a
    /// pool chunk on the direct I/O boundary, any shape's slots in any
    /// chunk, laid out as on the device.
    disk: Option<Spill>,
    /// The shared pool's host tier (CONTRACT-BC 22), where one is stated.
    tier: Option<HostTier>,
    /// The weight cache's row-streamed tables, a shape each (`super::rows`).
    weight_caches: Vec<super::streamed::Cache>,
    /// An evacuation moved the pool's table and restaging the weight
    /// caches' tables after it failed: the evacuation stands (the runtime
    /// counts it done), and every later fire is refused by this fault
    /// rather than read weights through tables that name moved chunks.
    weights_stale: Option<String>,
    /// The runtime holds the weight set for the load's life.
    weights_held: bool,
}

/// The host tier: chunks of the pool's size in host memory apart from the
/// pool, a slot of any shape at the place its host ref names, each chunk
/// allocated as a copy first touches it. On unified memory it is the same
/// DRAM, so a load seats one only where its budget is stated: what it buys
/// is room in the pool's heap, not in memory.
#[derive(Debug)]
struct HostTier {
    chunk_bytes: u64,
    chunks: Vec<Option<Box<[u8]>>>,
}

impl HostTier {
    /// Host ref `slot` of `shape`: its chunk and the byte range in it.
    fn place(&self, shape: &Slots, slot: u32) -> Result<(usize, std::ops::Range<usize>)> {
        let per_chunk = shape.per_chunk.max(1);
        let chunk = (slot / per_chunk) as usize;
        if chunk >= self.chunks.len() {
            return Err(Fault::Ceiling {
                what: "chunks of the host tier",
                need: chunk as u64 + 1,
                have: self.chunks.len() as u64,
            });
        }
        let at = usize::try_from(u64::from(slot % per_chunk) * shape.slot_bytes).unwrap_or(0);
        let len = usize::try_from(shape.slot_bytes).unwrap_or(0);
        Ok((chunk, at..at + len))
    }

    /// The bytes of host ref `slot` of `shape`, its chunk allocated first.
    fn slot(&mut self, shape: &Slots, slot: u32) -> Result<&mut [u8]> {
        let (chunk, range) = self.place(shape, slot)?;
        let bytes = usize::try_from(self.chunk_bytes).unwrap_or(usize::MAX);
        let held = self.chunks[chunk].get_or_insert_with(|| vec![0u8; bytes].into_boxed_slice());
        Ok(&mut held[range])
    }

    /// Copies host ref `slot` of `shape` into `into`: zeros where no copy
    /// has touched its chunk yet (nothing is allocated for a read).
    fn read(&self, shape: &Slots, slot: u32, into: &mut [u8]) -> Result<()> {
        let (chunk, range) = self.place(shape, slot)?;
        match &self.chunks[chunk] {
            Some(held) => into.copy_from_slice(&held[range]),
            None => into.fill(0),
        }
        Ok(())
    }
}

/// The first byte of physical slot `slot` of `shape`.
fn slot_at(layout: &Layout, shape: &Slots, slot: u32) -> u64 {
    u64::from(slot / shape.per_chunk) * layout.chunk_bytes
        + u64::from(slot % shape.per_chunk) * shape.slot_bytes
}

impl Pools {
    /// A pool of `chunks` chunks of `chunk_bytes` holding no state rows:
    /// only the guest shapes (`INSTANCE`, `CHANNEL`), every segment
    /// committed. For tests of what guests place in slots.
    #[cfg(all(test, target_vendor = "apple"))]
    pub(crate) fn guests_only(device: &Context, chunk_bytes: u64, chunks: u32) -> Result<Pools> {
        Self::guests_over(device, chunk_bytes, chunks, chunks)
    }

    /// [`Pools::guests_only`] in segments of `segment` chunks, every
    /// segment committed: under [`super::pool::cap_parts`], a buffer a
    /// segment.
    #[cfg(all(test, target_vendor = "apple"))]
    pub(crate) fn guests_over(
        device: &Context,
        chunk_bytes: u64,
        chunks: u32,
        segment: u32,
    ) -> Result<Pools> {
        let layout = Layout::new(chunk_bytes, chunks, segment, 16, [0, 0], 1, 0, 0)?;
        let rows = Rows {
            rows: Vec::new(),
            chunk_bytes,
            list_bytes: 0,
            tokens: 16,
            fixed_bytes: 0,
            pieces: 1,
            windows: Vec::new(),
            tables: Vec::new(),
            bands: Vec::new(),
            rs: None,
            kv_rows: Vec::new(),
            state_rows: Vec::new(),
        };
        let paging = super::kv::Paging::of(16, 256, 1, 1)?;
        let mut pools = Pools::reserve(device, rows, layout, paging)?;
        pools.commit_segments(chunks.div_ceil(segment))?;
        Ok(pools)
    }

    /// A pool of `chunks` chunks of `chunk_bytes`, in segments of `segment`
    /// chunks, holding a kv list of `list_bytes` slots and nothing else,
    /// every segment committed: under [`super::pool::cap_parts`], a buffer a
    /// segment. For tests of how kernels reach pages past the first buffer.
    #[cfg(all(test, target_vendor = "apple"))]
    pub(crate) fn listed_over(
        device: &Context,
        chunk_bytes: u64,
        chunks: u32,
        segment: u32,
        list_bytes: u64,
        tables: u32,
    ) -> Result<Pools> {
        let mut layout = Layout::new(chunk_bytes, chunks, segment, 16, [list_bytes, 0], 1, 0, 0)?;
        layout.tables = tables;
        let rows = Rows {
            rows: Vec::new(),
            chunk_bytes,
            list_bytes,
            tokens: 16,
            fixed_bytes: 0,
            pieces: 1,
            windows: Vec::new(),
            tables: Vec::new(),
            bands: Vec::new(),
            rs: None,
            kv_rows: Vec::new(),
            state_rows: Vec::new(),
        };
        let paging = super::kv::Paging::of(16, 256, 1, 1)?;
        let mut pools = Pools::reserve(device, rows, layout, paging)?;
        pools.commit_segments(chunks.div_ceil(segment))?;
        Ok(pools)
    }

    /// The pool `layout` sizes over `rows`, kv pages of `paging`'s size.
    pub fn reserve(device: &Context, rows: Rows, layout: Layout, paging: Paging) -> Result<Pools> {
        let mut pool = Pool::reserve(
            device,
            layout.chunk_bytes,
            layout.total_chunks,
            layout.segment_chunks,
            layout.act_span(),
            super::pool::part_cap(device),
        )?;
        // A page id names its slot under its buffer index, by its slot in
        // the buffer or by its chunk and index (`PageIds`): the paged shapes
        // (kv list, window lists, rs list) decide which, refused by name
        // where the bits do not hold them.
        let paged: Vec<(u32, u64)> = layout
            .list
            .iter()
            .chain(
                (0..MAX_WINDOWS)
                    .filter_map(|k| layout.window(k))
                    .map(|w| &w.slots),
            )
            .chain(layout.rs.iter())
            .map(|slots| (slots.per_chunk, slots.slot_bytes))
            .collect();
        pool.split_slots(&paged)?;
        let list_slots = layout.list.map_or(0, |list| {
            u64::from(list.per_chunk) * u64::from(layout.total_chunks)
        });
        let mut pools = Pools {
            paging: Paging {
                pages: list_slots.max(1),
                ..paging
            },
            maps: None,
            deltas: None,
            shape_rows: [rows.kv_rows.clone(), rows.state_rows.clone()],
            // A load without state rows still numbers each lane's seat: one
            // band of one id.
            bands: if rows.bands.is_empty() {
                vec![vec![0]]
            } else {
                rows.bands
            },
            rs: rows.rs,
            tables: rows.tables,
            rows: rows.rows,
            pool,
            layout,
            disk: None,
            tier: None,
            weight_caches: Vec::new(),
            weights_stale: None,
            weights_held: false,
            table_run: (
                layout.chunk_bytes,
                layout.chunk_bytes * u64::from(1 + layout.tables),
            ),
        };
        // The tables run is backed from the load on: the engine writes its
        // tables there before any fire.
        if layout.tables > 0 {
            let segments = (1 + layout.tables).div_ceil(layout.segment_chunks.max(1));
            pools.pool.commit_segments(segments)?;
        }
        pools.write_maps(device)?;
        pools.write_deltas(device)?;
        Ok(pools)
    }

    /// Writes the buffers' delta table where the range spans more than one
    /// and pages a shape (kv list, window lists, rs list) kernels reach
    /// through it; a state row reaches the buffers through its own map. The
    /// table is fixed for the load and every buffer-1 page is reached
    /// through it, so it lies in a buffer of its own, outside the pool: no
    /// write into the pool's chunks, stray or not, can move a page.
    fn write_deltas(&mut self, device: &Context) -> Result<()> {
        let paged = self.layout.list.is_some()
            || self.layout.rs.is_some()
            || (0..MAX_WINDOWS).any(|k| self.layout.window(k).is_some());
        if self.pool.parts() < 2 || !paged {
            return Ok(());
        }
        let words: Vec<u8> = self
            .pool
            .deltas()
            .iter()
            .flat_map(|delta| delta.to_le_bytes())
            .collect();
        let mut table = Buffer::zeroed(device, words.len() as u64)?;
        table.write(0, &words)?;
        self.deltas = Some(table);
        Ok(())
    }

    /// The device address of the buffers' delta table, where the range
    /// spans more than one buffer.
    #[must_use]
    pub fn delta_table(&self) -> Option<u64> {
        self.deltas.as_ref().and_then(|table| table.address_at(0))
    }

    /// Writes each state row's segment table: the bank is the whole pool,
    /// a chunk `chunk_bytes` apart, a slot the fixed slot apart, the row at
    /// its offset in each of its pieces' slots; a segment a buffer.
    fn write_maps(&mut self, device: &Context) -> Result<()> {
        let Some(fixed) = self.layout.fixed else {
            return Ok(());
        };
        let mut tables = Vec::new();
        for row in &self.rows {
            let Row::State {
                dtype,
                offset,
                pieces,
                piece_atoms,
                ..
            } = *row
            else {
                continue;
            };
            let element = u64::from(elem_size(dtype));
            // A segment a buffer: each base is the buffer's first element
            // from the first buffer's, where the bank is bound (wrapping, so
            // a buffer below the first is a negative step).
            let parts = self.pool.parts();
            let map = StateMap {
                slot_stride: fixed.slot_bytes / element,
                chunk_slots: fixed.per_chunk,
                chunk_stride: self.layout.chunk_bytes / element,
                seg_chunks: if parts == 1 {
                    u32::MAX
                } else {
                    self.pool.part_chunks()
                },
                piece_atoms,
                pieces: pieces.max(1),
                bases: self
                    .pool
                    .deltas()
                    .iter()
                    .map(|&delta| {
                        (((delta as i64) / element as i64) as u64).wrapping_add(offset / element)
                    })
                    .collect(),
            };
            tables.extend(map.bytes());
        }
        if tables.is_empty() {
            return Ok(());
        }
        let mut region = self.run_table(device, tables.len() as u64)?;
        region.write(0, &tables)?;
        self.maps = Some(region);
        Ok(())
    }

    /// The tables run's bytes the state maps of `state_rows` state rows take
    /// where the range spans at most `parts` buffers: what a load counts into
    /// the run before it lays it. (The buffers' delta table lies in a buffer
    /// of its own, [`Pools::delta_table`].)
    #[must_use]
    pub fn map_bytes(state_rows: usize, parts: usize) -> u64 {
        let maps = match state_rows {
            0 => 0,
            rows => crate::arena::room(rows as u64 * super::state_map_bytes(parts)),
        };
        maps
    }

    /// `bytes` of the tables run (`Reserved::Tables`), zeroed and written
    /// through its host runs: one of the engine's own load-time tables. The
    /// run was sized for every one at the load; one past it is refused.
    pub fn run_table(&mut self, device: &Context, bytes: u64) -> Result<crate::inputs::Region> {
        let (at, end) = self.table_run;
        let room = crate::arena::room(bytes.max(1));
        if at + room > end {
            return Err(Fault::Ceiling {
                what: "bytes of the engine's tables run",
                need: at + room - self.layout.chunk_bytes,
                have: end - self.layout.chunk_bytes,
            });
        }
        let (part, local) = self.pool.locate(at);
        if self.pool.locate(at + room - 1).0 != part {
            return Err(Fault::Residency(format!(
                "a {bytes}-byte table of the engine's tables run would straddle two of the \
                 pool's buffers"
            )));
        }
        let mut region = crate::inputs::Region::unplaced(bytes, Buffer::zeroed(device, 0)?);
        region.place(self.pool.part_view(part).clone(), local);
        region.rehost(self.pool.host_runs(at, room)?)?;
        self.table_run.0 = at + room;
        Ok(region)
    }

    /// The tables run's bytes, laid or not: model bytes in the pool.
    #[must_use]
    pub fn tables_run_bytes(&self) -> u64 {
        u64::from(self.layout.tables) * self.layout.chunk_bytes
    }

    /// Each band's width: the ids a lane's or a row's table holds for the
    /// state rows whose pieces lie in that band's record slots.
    #[must_use]
    pub fn bands(&self) -> Vec<u32> {
        self.bands
            .iter()
            .map(|band| u32::try_from(band.len()).unwrap_or(u32::MAX))
            .collect()
    }

    /// `record`'s physical piece ids as each band reads them, band after
    /// band.
    #[must_use]
    pub fn band_ids(&self, record: &[u32]) -> Vec<i32> {
        self.bands
            .iter()
            .flatten()
            .map(|&slot| record.get(slot as usize).map_or(-1, |&id| id as i32))
            .collect()
    }

    /// The rs buffer's place in a record, where the load buffers one.
    #[must_use]
    pub fn rs_rows(&self) -> Option<&RsRows> {
        self.rs.as_ref()
    }

    /// A lane's rs-list pages as kernels address them, in token order from
    /// the verb's base: `refs` (rs-list slot refs) as page ids
    /// ([`super::pool::PageIds`]), or, seat-addressed, seat `seat`'s run.
    pub fn rs_pages(&self, refs: &[u32], seat: Option<u32>) -> Result<Vec<u32>> {
        let rs = *self.shape(RS)?;
        refs.iter()
            .enumerate()
            .map(|(at, &r)| {
                let slot = match seat {
                    Some(seat) => {
                        let per = self.paging.pages_per_slot;
                        rs.seat_ref(seat * per + u32::try_from(at).unwrap_or(u32::MAX))?
                    }
                    None => r,
                };
                let physical = self.physical(RS, slot)?;
                self.pool.admit(physical / rs.per_chunk.max(1))?;
                Ok(self.pool.slot_page(physical, rs.per_chunk))
            })
            .collect()
    }

    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The pool's buffer, for the rs buffer pages that live in it.
    #[must_use]
    pub fn view(&self) -> &Buffer {
        self.pool.view()
    }

    /// The pool as the runtime counts it.
    #[must_use]
    pub fn chunk_facts(&self) -> ChunkFacts {
        let mut facts = self.layout.facts(
            self.pool.growth(),
            self.pool.committed_segments(),
            &self.shape_rows,
        );
        facts.host_chunks = self.tier.as_ref().map_or(0, |tier| {
            u32::try_from(tier.chunks.len()).unwrap_or(u32::MAX)
        });
        facts.weights = self
            .weight_caches
            .iter()
            .map(super::streamed::Cache::facts)
            .collect();
        facts.disk_chunks = self.disk_chunks();
        facts
    }

    /// Seats a host tier of `budget` bytes, in whole chunks (CONTRACT-BC
    /// 22).
    pub fn seat_tier(&mut self, budget: u64) {
        let chunk_bytes = self.layout.chunk_bytes;
        let chunks = budget.checked_div(chunk_bytes).unwrap_or(0);
        self.tier = (chunks > 0).then(|| HostTier {
            chunk_bytes,
            chunks: (0..chunks).map(|_| None).collect(),
        });
    }

    /// Copies whole slots of `id` between the pool (slot refs) and the host
    /// tier (host refs), `(src, dst)` the way `to_host` says. The caller has
    /// drained every frame.
    fn park(&mut self, id: ShapeId, to_host: bool, pairs: &[(u32, u32)]) -> Result<()> {
        let shape = *self.shape(id)?;
        for &(src, dst) in pairs {
            let (device, host) = if to_host { (src, dst) } else { (dst, src) };
            let at = self.slot_offset(id, device)?;
            self.pool
                .admit(u32::try_from(at / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
            let Some(tier) = self.tier.as_mut() else {
                return Err(Fault::Ceiling {
                    what: "chunks of the host tier",
                    need: u64::from(host / shape.per_chunk.max(1)) + 1,
                    have: 0,
                });
            };
            let slot = tier.slot(&shape, host)?;
            if to_host {
                self.pool.read(at, slot)?;
            } else {
                self.pool.write(at, slot)?;
            }
        }
        Ok(())
    }

    /// See [`engine::Engine::read_tier_slots`]: the bytes of host tier
    /// slots, shape by shape, each laid out as `copy_slots` moves it (the
    /// device slot's bytes). Unified memory: the host tier is plain host
    /// memory, so this is one memcpy a slot, no staging. The caller has
    /// drained every frame.
    pub fn read_tier_slots(&self, slots: &[(ShapeId, Vec<u32>)]) -> Result<Vec<u8>> {
        let tier = self.tier.as_ref().ok_or(Fault::Ceiling {
            what: "chunks of the host tier",
            need: 1,
            have: 0,
        })?;
        let mut total = 0u64;
        for (id, refs) in slots {
            total += self.shape(*id)?.slot_bytes * refs.len() as u64;
        }
        let mut out = vec![0u8; usize::try_from(total).unwrap_or(usize::MAX)];
        let mut at = 0usize;
        for (id, refs) in slots {
            let shape = *self.shape(*id)?;
            let len = usize::try_from(shape.slot_bytes).unwrap_or(usize::MAX);
            for &slot in refs {
                tier.read(&shape, slot, &mut out[at..at + len])?;
                at += len;
            }
        }
        Ok(out)
    }

    /// See [`engine::Engine::write_tier_slots`]: `bytes`, laid out as
    /// [`Self::read_tier_slots`] returns them, into host tier slots; refused
    /// whole when the byte count differs or a ref lies past the tier. The
    /// caller has drained every frame.
    pub fn write_tier_slots(&mut self, slots: &[(ShapeId, Vec<u32>)], bytes: &[u8]) -> Result<()> {
        let mut need = 0u64;
        for (id, refs) in slots {
            let shape = *self.shape(*id)?;
            need += shape.slot_bytes * refs.len() as u64;
            if let Some(tier) = self.tier.as_ref() {
                for &slot in refs {
                    tier.place(&shape, slot)?;
                }
            }
        }
        if bytes.len() as u64 != need {
            return Err(Fault::Ceiling {
                what: "bytes of host tier slots",
                need,
                have: bytes.len() as u64,
            });
        }
        let mut at = 0usize;
        for (id, refs) in slots {
            let shape = *self.shape(*id)?;
            let len = usize::try_from(shape.slot_bytes).unwrap_or(usize::MAX);
            for &slot in refs {
                let tier = self.tier.as_mut().ok_or(Fault::Ceiling {
                    what: "chunks of the host tier",
                    need: 1,
                    have: 0,
                })?;
                tier.slot(&shape, slot)?
                    .copy_from_slice(&bytes[at..at + len]);
                at += len;
            }
        }
        Ok(())
    }

    fn shape(&self, id: ShapeId) -> Result<&Slots> {
        self.layout.shape(id).ok_or_else(|| Fault::Unbound {
            what: format!("slot shape {id}, which this pool does not hold"),
        })
    }

    /// The physical chunk and index slot ref `slot` of `shape` sits at.
    fn place(&self, shape: &Slots, slot: u32) -> Result<(u32, u32)> {
        let (chunk, index) = shape.split(slot);
        if chunk == 0 || chunk >= self.layout.total_chunks {
            return Err(Fault::Ceiling {
                what: "chunks a slot ref may name (past the null chunk)",
                need: u64::from(chunk) + 1,
                have: u64::from(self.layout.total_chunks),
            });
        }
        let physical = self.pool.physical(chunk)?;
        // Chunks 1..1+tables are the engine's own (`Reserved::Tables`): a
        // kv page, a copy's end or a held table that named one would land
        // on the delta table, the state maps or the bank tables.
        let tables = 1..1 + self.layout.tables;
        if tables.contains(&chunk) || tables.contains(&physical) {
            return Err(Fault::Residency(format!(
                "slot ref {slot} of shape {} lies in chunk {chunk} (physical {physical}), \
                 inside the engine's tables run (chunks {}..{}), which no slot names",
                shape.id, tables.start, tables.end
            )));
        }
        self.pool.clear_of_carve(chunk, physical)?;
        Ok((physical, index))
    }

    /// The physical slot kernels address for slot ref `slot` of `shape`.
    fn physical(&self, id: ShapeId, slot: u32) -> Result<u32> {
        let shape = *self.shape(id)?;
        let (chunk, index) = self.place(&shape, slot)?;
        Ok(chunk * shape.per_chunk + index)
    }

    /// Slot ref `slot` of shape `id` as a guest's table or ring holds it
    /// (`program::held`): the buffer it lies in, its byte offset there, the host address of
    /// its first byte (through the heap under it), and its bytes.
    pub fn slot_place(
        &self,
        id: ShapeId,
        slot: u32,
    ) -> Result<(Buffer, u64, std::ptr::NonNull<u8>, u64)> {
        let (view, at, runs, slot_bytes) = self.slot_runs(id, slot)?;
        // One host address: the slot lies in one heap (a chunk may cross a
        // heap's end; a slot a table or ring holds does not).
        let [(_, _, host)] = runs[..] else {
            return Err(Fault::Device {
                call: "slot host address",
                why: format!(
                    "slot {slot} of shape {id} crosses a placement heap's end, so no one \
                     host address spans it; write it through its host runs"
                ),
            });
        };
        Ok((view, at, host, slot_bytes))
    }

    /// Bytes `[at, at + bytes)` from the start of slot ref `slot` of shape
    /// `id` (inside its chunk): the buffer they lie in, their byte offset
    /// there, and their host runs `(from, len, address)`, one a placement
    /// heap they cross (a chunk, and a slot in it, may cross a heap's end;
    /// the view holds them in one span either way).
    pub fn span_runs(
        &self,
        id: ShapeId,
        slot: u32,
        at: u64,
        bytes: u64,
    ) -> Result<(Buffer, u64, Vec<(u64, u64, std::ptr::NonNull<u8>)>)> {
        let shape = *self.shape(id)?;
        let (chunk, index) = self.place(&shape, slot)?;
        let global = u64::from(chunk) * self.layout.chunk_bytes
            + u64::from(index) * shape.slot_bytes
            + at;
        self.pool.admit(chunk)?;
        let runs = self.pool.host_runs(global, bytes)?;
        let (view, offset) = self.pool.view_at(global);
        Ok((view.clone(), offset, runs))
    }

    /// Slot ref `slot` of shape `id`: the buffer it lies in, its byte offset
    /// there, its bytes' host runs `(from, len, address)` (one a heap they
    /// cross) and its bytes.
    pub fn slot_runs(
        &self,
        id: ShapeId,
        slot: u32,
    ) -> Result<(Buffer, u64, Vec<(u64, u64, std::ptr::NonNull<u8>)>, u64)> {
        let shape = *self.shape(id)?;
        let (chunk, index) = self.place(&shape, slot)?;
        let global =
            u64::from(chunk) * self.layout.chunk_bytes + u64::from(index) * shape.slot_bytes;
        self.pool.admit(chunk)?;
        let runs = self.pool.host_runs(global, shape.slot_bytes)?;
        let (view, at) = self.pool.view_at(global);
        Ok((view.clone(), at, runs, shape.slot_bytes))
    }

    /// The contiguous pieces a guest's state lies in (`engine::ScratchPlace`):
    /// a run of `Instance` slots in one chunk, or runs of whole chunks, each
    /// from its first logical chunk. Each piece as the buffer it lies in,
    /// its offset there, its offset in the pool (what host runs are asked
    /// by) and its bytes; none for `None`.
    pub fn guest_pieces(&self, place: &engine::ScratchPlace) -> Result<Vec<GuestPiece>> {
        match place {
            engine::ScratchPlace::None => Ok(Vec::new()),
            engine::ScratchPlace::Run { first, slots } => {
                let shape = *self.shape(super::layout::INSTANCE)?;
                let (chunk, index) = self.place(&shape, *first)?;
                let last = index + slots.saturating_sub(1);
                if *slots == 0 || last >= shape.per_chunk {
                    return Err(Fault::Residency(format!(
                        "a guest's run of {slots} `Instance` slot(s) from ref {first} passes \
                         its chunk of {} slots",
                        shape.per_chunk
                    )));
                }
                self.pool.admit(chunk)?;
                let global = u64::from(chunk) * self.layout.chunk_bytes
                    + u64::from(index) * shape.slot_bytes;
                let (view, local) = self.pool.view_at(global);
                Ok(vec![GuestPiece {
                    view: view.clone(),
                    local,
                    global,
                    bytes: u64::from(*slots) * shape.slot_bytes,
                }])
            }
            engine::ScratchPlace::Chunks { run, pieces } => pieces
                .iter()
                .map(|&logical| {
                    let physical = self.pool.physical(logical)?;
                    // A piece is addressed from its first chunk and read on
                    // through the next physical chunks (the runtime moves a
                    // run whole, its chunks' logical ids not kept in order):
                    // those chunks must lie in its first chunk's segment, and
                    // each in memory the pool holds (`admit`), or its later
                    // bytes are another owner's or none.
                    let seg = self.pool.segment_chunks().max(1);
                    let last = physical + run.saturating_sub(1);
                    if physical / seg != last / seg {
                        return Err(Fault::Residency(format!(
                            "a guest piece of {run} chunk(s) from logical chunk {logical} lies \
                             on physical chunks {physical}..={last}, across the end of its \
                             segment of {seg} chunks: its bytes past the segment would be \
                             another owner's"
                        )));
                    }
                    for at in 0..*run {
                        self.pool.admit(physical + at)?;
                    }
                    let global = u64::from(physical) * self.layout.chunk_bytes;
                    let (view, local) = self.pool.view_at(global);
                    Ok(GuestPiece {
                        view: view.clone(),
                        local,
                        global,
                        bytes: u64::from(*run) * self.layout.chunk_bytes,
                    })
                })
                .collect(),
        }
    }

    /// The bytes of one `Instance` slot and the slots a chunk holds: what
    /// the runtime places a guest's state by.
    pub fn instance_slots(&self) -> Result<(u64, u32)> {
        let shape = *self.shape(super::layout::INSTANCE)?;
        Ok((shape.slot_bytes, shape.per_chunk))
    }

    /// A chunk's bytes.
    #[must_use]
    pub fn chunk_bytes(&self) -> u64 {
        self.layout.chunk_bytes
    }

    /// The slots of shape `id` one chunk holds.
    pub fn slots_per_chunk(&self, id: ShapeId) -> Result<u32> {
        Ok(self.shape(id)?.per_chunk)
    }

    /// The bytes of one slot of shape `id`.
    pub fn slot_bytes(&self, id: ShapeId) -> Result<u64> {
        Ok(self.shape(id)?.slot_bytes)
    }

    /// The byte offset of slot ref `slot` of shape `id`.
    fn slot_offset(&self, id: ShapeId, slot: u32) -> Result<u64> {
        let shape = *self.shape(id)?;
        let (chunk, index) = self.place(&shape, slot)?;
        Ok(u64::from(chunk) * self.layout.chunk_bytes + u64::from(index) * shape.slot_bytes)
    }

    /// A lane's kv pages as kernels address them: `refs` (slot refs of the
    /// list shape) as physical pages, or, seat-addressed, seat `seat`'s
    /// `pages` pages.
    pub fn kv_pages(&self, refs: &[u32], seat: Option<(u32, u32)>) -> Result<Vec<u32>> {
        // A load whose every kv row is windowed pages nothing in a list.
        if self.layout.list.is_none() {
            return Ok(Vec::new());
        }
        if let Some((slot, pages)) = seat
            && refs.is_empty()
        {
            let list = *self.shape(LIST)?;
            let first = u64::from(slot) * u64::from(self.paging.pages_per_slot);
            return (0..pages)
                .map(|page| {
                    let id = u32::try_from(first + u64::from(page)).unwrap_or(u32::MAX);
                    Ok(self
                        .pool
                        .slot_page(self.physical(LIST, list.seat_ref(id)?)?, list.per_chunk))
                })
                .collect();
        }
        let list = *self.shape(LIST)?;
        refs.iter()
            .map(|&r| Ok(self.pool.slot_page(self.physical(LIST, r)?, list.per_chunk)))
            .collect()
    }

    /// The page tables a fire stages, by the list each pages: the kv
    /// list's first, the All axis's.
    #[must_use]
    pub fn tables(&self) -> &[TablePaging] {
        &self.tables
    }

    /// The page table the rows paged in `paging`'s list read.
    #[must_use]
    pub fn table_of(&self, paging: TablePaging) -> usize {
        self.tables
            .iter()
            .position(|&held| held == paging)
            .unwrap_or(0)
    }

    /// Window list `k`'s paging: `T_w` tokens a page (CONTRACT-BC 21), as
    /// many pages a seat as its context fills.
    #[must_use]
    pub fn window_paging(&self, k: usize) -> Paging {
        let tokens = self.layout.window(k).map_or(1, |w| w.tokens.max(1));
        Paging {
            page_size: tokens,
            pages_per_slot: self.paging.context().div_ceil(tokens).max(1),
            ..self.paging
        }
    }

    /// The most pages a seat takes in any window list.
    #[must_use]
    pub fn window_pages_per_slot(&self) -> u32 {
        (0..MAX_WINDOWS)
            .filter(|&k| self.layout.window(k).is_some())
            .map(|k| self.window_paging(k).pages_per_slot)
            .max()
            .unwrap_or(0)
    }

    /// A window list `k` page's bytes: what its page id steps by in a
    /// chunk ([`super::pool::PageIds`]).
    fn window_stride(&self, k: usize) -> u64 {
        self.layout
            .window(k)
            .map_or(0, |window| window.slots.slot_bytes)
    }

    /// A lane's pages of window list `k` as kernels address them, one a
    /// window slot (`T_w` tokens) in token order: `refs` (window slot refs,
    /// 0 a slot trimmed behind the window: the null chunk) as page ids
    /// ([`super::pool::PageIds`]), or, seat-addressed, seat `seat`'s run,
    /// `pages` of them.
    pub fn window_pages(
        &self,
        k: usize,
        refs: &[u32],
        seat: Option<(u32, u32)>,
    ) -> Result<Vec<u32>> {
        let id = WINDOWS[k.min(MAX_WINDOWS - 1)];
        let window = *self.shape(id)?;
        let unit = |slot: u32| -> Result<u32> {
            if slot == 0 {
                return Ok(0);
            }
            let physical = self.physical(id, slot)?;
            self.pool.admit(physical / window.per_chunk.max(1))?;
            Ok(self.pool.slot_page(physical, window.per_chunk))
        };
        if let Some((slot, pages)) = seat
            && refs.is_empty()
        {
            let first = u64::from(slot) * u64::from(self.window_paging(k).pages_per_slot);
            return (0..pages)
                .map(|page| {
                    let id = u32::try_from(first + u64::from(page)).unwrap_or(u32::MAX);
                    unit(window.seat_ref(id)?)
                })
                .collect();
        }
        refs.iter().map(|&r| unit(r)).collect()
    }

    /// The page explicit write descriptor page `page` names, as kernels
    /// address it (its buffer in the high bits, as `kv_pages` states it).
    pub fn kv_page(&self, page: u32) -> Result<u32> {
        let list = *self.shape(LIST)?;
        Ok(self
            .pool
            .slot_page(self.physical(LIST, page)?, list.per_chunk))
    }

    /// A lane's record as kernels address it, one physical id a piece:
    /// `refs` (slot refs of the fixed shape, a piece each), or a lone ref of
    /// a one-piece record, or seat `seat`'s record.
    pub fn record(&self, refs: Option<&[u32]>, seat: u32) -> Result<Vec<u32>> {
        if !self.has_state() {
            return Ok(vec![seat]);
        }
        let pieces = self.layout.pieces;
        match refs {
            Some(refs) => {
                if refs.len() != pieces as usize {
                    return Err(Fault::Unbound {
                        what: format!(
                            "a record named by {} slot refs, where this pool's records are {pieces} \
                             pieces",
                            refs.len()
                        ),
                    });
                }
                refs.iter().map(|&r| self.physical(FIXED, r)).collect()
            }
            None => {
                let fixed = *self.shape(FIXED)?;
                (0..pieces)
                    .map(|p| self.physical(FIXED, fixed.seat_ref(seat * pieces + p)?))
                    .collect()
            }
        }
    }

    /// The physical fixed slot a one-piece record's slot ref (or, seated,
    /// its seat) names.
    pub fn state_slot(&self, id: u32, seated: bool) -> Result<u32> {
        if !self.has_state() {
            return Ok(id);
        }
        if seated {
            return Ok(self.record(None, id)?[0]);
        }
        if self.layout.pieces != 1 {
            return Err(Fault::Unbound {
                what: format!(
                    "a record named by one slot ref, where this pool's records are {} pieces",
                    self.layout.pieces
                ),
            });
        }
        self.physical(FIXED, id)
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.rows
            .iter()
            .map(|row| match row {
                Row::State { stride, dtype, .. } => stride * u64::from(elem_size(*dtype)),
                Row::Kv { .. } | Row::Compressed { .. } => 0,
            })
            .sum()
    }

    /// Slot ref `slot` of shape `id`, read whole on the host. The caller
    /// has drained every frame that writes it.
    pub fn read_slot_bytes(&self, id: ShapeId, slot: u32) -> Result<Vec<u8>> {
        let bytes = self.shape(id)?.slot_bytes;
        let at = self.slot_offset(id, slot)?;
        self.pool
            .admit(u32::try_from(at / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
        let mut out = vec![0u8; usize::try_from(bytes).unwrap_or(0)];
        self.pool.read(at, &mut out)?;
        Ok(out)
    }

    /// Writes `bytes` (one slot's) into slot ref `slot` of shape `id` on
    /// the host. The caller has drained every frame that reads it.
    pub fn write_slot_bytes(&mut self, id: ShapeId, slot: u32, bytes: &[u8]) -> Result<()> {
        let want = self.shape(id)?.slot_bytes;
        if bytes.len() as u64 != want {
            return Err(Fault::Ceiling {
                what: "bytes of one slot",
                need: bytes.len() as u64,
                have: want,
            });
        }
        let at = self.slot_offset(id, slot)?;
        self.pool
            .admit(u32::try_from(at / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
        self.pool.write(at, bytes)
    }

    /// The state rows of the record whose physical pieces are `record`,
    /// one after another, each row's pieces in order.
    /// The planes [`Pools::read_record`] lays end to end: each state row's
    /// trace cache index, bytes and dtype, in that order.
    #[must_use]
    pub fn state_planes(&self) -> Vec<(u32, u64, Dtype)> {
        self.rows
            .iter()
            .filter_map(|row| match *row {
                Row::State { stride, dtype, .. } => {
                    Some((stride * u64::from(elem_size(dtype)), dtype))
                }
                _ => None,
            })
            .zip(&self.shape_rows[1])
            .map(|((bytes, dtype), &cache)| (cache, bytes, dtype))
            .collect()
    }

    pub fn read_record(&self, record: &[u32]) -> Result<Vec<u8>> {
        let fixed = *self.shape(FIXED)?;
        let mut out = Vec::new();
        for row in &self.rows {
            let Row::State {
                stride,
                dtype,
                offset,
                band,
                pieces,
                piece_atoms,
                ..
            } = *row
            else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            let piece = if pieces <= 1 {
                bytes
            } else {
                bytes.div_ceil(u64::from(pieces))
            };
            let _ = piece_atoms;
            let mut left = bytes;
            for &slot in &self.bands[band] {
                let id = *record.get(slot as usize).ok_or(Fault::Ceiling {
                    what: "pieces of a record",
                    need: u64::from(slot) + 1,
                    have: record.len() as u64,
                })?;
                let take = piece.min(left);
                let mut span = vec![0u8; usize::try_from(take).unwrap_or(0)];
                self.pool
                    .read(self.physical_slot_at(&fixed, id)? + offset, &mut span)?;
                out.extend_from_slice(&span);
                left -= take;
            }
        }
        Ok(out)
    }

    /// The first byte of physical slot `slot` of `shape`, checked committed.
    fn physical_slot_at(&self, shape: &Slots, slot: u32) -> Result<u64> {
        self.pool.admit(slot / shape.per_chunk)?;
        Ok(slot_at(&self.layout, shape, slot))
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        self.rows.iter().any(|row| matches!(row, Row::State { .. }))
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// The pool's whole range: what it takes with every segment committed.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.pool.bytes()
    }

    /// The tables run's bytes its tables were laid in.
    #[must_use]
    pub fn table_bytes(&self) -> u64 {
        self.table_run.0 - self.layout.chunk_bytes
    }

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.pool.committed_bytes()
    }

    /// The device bytes behind the pool's committed segments.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        self.pool.held_bytes()
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.pool.high_water_bytes()
    }

    /// Where the activation span starts in the pool's buffer, and its bytes.
    #[must_use]
    pub fn span(&self) -> (u64, u64) {
        let offset = self.pool.span_offset();
        (offset, self.pool.bytes() - offset)
    }

    /// The buffers the range spans.
    #[must_use]
    pub fn parts(&self) -> usize {
        self.pool.parts()
    }

    /// The physical kv list slot a kernel's page id names (ids carry their
    /// buffer, [`Pool::slot_page`]).
    #[must_use]
    pub fn list_slot(&self, id: u32) -> u32 {
        let per = self.layout.list.map_or(1, |list| list.per_chunk);
        self.pool.page_slot(id, per)
    }

    /// Every buffer of the range past the first, which kernels reach
    /// through an address in the first.
    #[must_use]
    pub fn part_slabs(&self) -> Vec<crate::device::alloc::Slab> {
        (1..self.pool.parts())
            .map(|at| self.pool.part_view(at).slab().clone())
            .chain(self.deltas.as_ref().map(|table| table.slab().clone()))
            .collect()
    }

    /// Each buffer's address less the first's ([`Pool::deltas`]).
    #[must_use]
    pub fn deltas(&self) -> Vec<u64> {
        self.pool.deltas()
    }

    /// How a page id names its slot ([`super::pool::PageIds`]).
    #[must_use]
    pub fn page_ids(&self) -> super::pool::PageIds {
        self.pool.page_ids()
    }

    /// Each of the range's buffers, the first first.
    #[must_use]
    pub fn part_views(&self) -> Vec<Buffer> {
        (0..self.pool.parts())
            .map(|at| self.pool.part_view(at).clone())
            .collect()
    }

    /// The buffer the activation span lies in (the last), where in it the
    /// span starts, and its bytes.
    #[must_use]
    pub fn span_view(&self) -> (&Buffer, u64, u64) {
        let (offset, bytes) = self.span();
        let (view, at) = self.pool.view_at(offset);
        (view, at, bytes)
    }

    /// The device bytes the span's backed segments hold.
    #[must_use]
    pub fn span_held_bytes(&self) -> u64 {
        self.pool.top_held_bytes()
    }

    /// The span's bytes, backed or not, which the ledger holds reserved.
    #[must_use]
    pub fn span_reserved_bytes(&self) -> u64 {
        self.pool.top_bytes()
    }

    /// `len` bytes at `offset` of the pool's buffer as host runs
    /// ([`Pool::host_runs`]).
    pub fn host_runs(
        &self,
        offset: u64,
        len: u64,
    ) -> Result<Vec<(u64, u64, std::ptr::NonNull<u8>)>> {
        self.pool.host_runs(offset, len)
    }

    /// `into.len()` bytes at `offset` of the pool's buffer, read on the host.
    pub fn read_span(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.pool.read(offset, into)
    }

    /// `from` written at `offset` of the pool's buffer, on the host; for a
    /// test that lays bytes it then reads back.
    #[cfg(test)]
    pub fn write_span(&mut self, offset: u64, from: &[u8]) -> Result<()> {
        self.pool.write(offset, from)
    }

    /// Backs the range's top `bytes` for a fire's carve ([`Pool::cover`]).
    pub fn cover(&mut self, bytes: u64) -> Result<()> {
        self.pool.cover(bytes)
    }

    /// The span's segments, from the one it starts in to the range's end.
    #[must_use]
    pub fn span_segments(&self) -> u32 {
        self.pool.span_segments()
    }

    /// The bytes from the range's end the rung's carve backs.
    #[must_use]
    pub fn covered(&self) -> u64 {
        self.pool.covered()
    }

    /// Whether the range's top `bytes` are backed for a fire's carve.
    #[must_use]
    pub fn covers(&self, bytes: u64) -> bool {
        self.pool.covers(bytes)
    }

    /// Gives back the carve past the range's top `bytes`.
    ///
    /// # Safety
    ///
    /// As [`Pool::narrow`].
    pub unsafe fn narrow(&mut self, bytes: u64) -> Result<()> {
        // SAFETY: the caller's contract.
        unsafe { self.pool.narrow(bytes) }
    }

    /// Backs the span's top `segments` for state ([`Pool::commit_span`]).
    pub fn commit_span(&mut self, segments: u32) -> Result<()> {
        self.pool.commit_span(segments)
    }

    /// Gives back the span's segments past the top `segments`.
    ///
    /// # Safety
    ///
    /// As [`Pool::release_span`].
    pub unsafe fn release_span(&mut self, segments: u32) -> Result<()> {
        // SAFETY: the caller's contract.
        unsafe { self.pool.release_span(segments) }
    }

    /// Zeroes `len` bytes at `offset` of the pool's buffer on the host,
    /// where no frame reads them.
    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        self.pool.zero(offset, len)
    }

    /// The cache table one fire binds; `q8` reads every kv pool through the
    /// 8-bit attention kernels.
    pub fn table(&self, handles: &Handles, seats: &Seats, q8: bool) -> Result<CacheTable> {
        let view = self.pool.view();
        let cells = u32::try_from(self.paging.pages() * u64::from(self.paging.page_size))
            .unwrap_or(u32::MAX);
        let list_bytes = self.layout.list.map_or(0, |list| list.slot_bytes);
        let mut pools = Vec::with_capacity(self.rows.len());
        let mut compressed = Vec::new();
        let mut state_row = 0u64;
        let mut lane_slots = Vec::with_capacity(self.rows.len());
        let mut band_of = Vec::with_capacity(self.rows.len());
        let mut tables = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            pools.push(match *row {
                Row::Compressed {
                    row,
                    width,
                    paging,
                    kv,
                    score,
                } => {
                    // In the slots of the row it keys by: that row's page
                    // ids and page stride. Each plane is bound at its offset
                    // into a page (`plane_at`), so kernels read it from 0.
                    let plane = |offset: u64| -> Result<Tensor> {
                        Ok(Tensor::new(
                            handles.bind(view, offset, view.bytes() - offset)?,
                            cells,
                            u32::try_from(width).unwrap_or(u32::MAX),
                            Dtype::Bf16,
                        ))
                    };
                    let stride = match paging {
                        TablePaging::List => list_bytes,
                        TablePaging::Window(k) => self.window_stride(usize::from(k)),
                    };
                    compressed.push((
                        row,
                        PoolSlabs {
                            kv: plane(kv)?,
                            score: plane(score)?,
                            page_stride: [stride, stride],
                        },
                    ));
                    continue;
                }
                Row::Kv {
                    paging,
                    head_dim,
                    kv_heads,
                    dtype,
                    keys,
                    values,
                    values_width,
                    ..
                } => {
                    let table = self.table_of(paging);
                    let seat: &TableSeat =
                        seats.page_tables.get(table).ok_or_else(|| Fault::Unbound {
                            what: format!("page table {table}, which this fire staged none of"),
                        })?;
                    // Each plane bound at its offset into a page, so the
                    // paged kernels read it from 0 (`kernels/prim/kv_page.h`).
                    let plane = |offset: u64, width: u64| -> Result<Tensor> {
                        Ok(Tensor::new(
                            handles.bind(view, offset, view.bytes() - offset)?,
                            cells,
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        ))
                    };
                    lane_slots.push(None);
                    band_of.push(None);
                    tables.push(Some(table));
                    let (stride, page_size) = match paging {
                        TablePaging::List => (list_bytes, self.paging.page_size),
                        TablePaging::Window(k) => (
                            self.window_stride(usize::from(k)),
                            self.window_paging(usize::from(k)).page_size,
                        ),
                    };
                    CachePool::Kv(KvPool {
                        keys: plane(keys, u64::from(kv_heads) * u64::from(head_dim))?,
                        values: plane(values, values_width)?,
                        page_indices: seat.page_indices,
                        page_indptr: seat.page_indptr,
                        page_size: narrow(u64::from(page_size)),
                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                        q8,
                        page_stride: [stride, stride],
                    })
                }
                Row::State {
                    stride,
                    dtype,
                    band,
                    ..
                } => {
                    let fixed = self.shape(FIXED)?;
                    let slots = u32::try_from(
                        u64::from(fixed.per_chunk) * u64::from(self.layout.total_chunks),
                    )
                    .unwrap_or(u32::MAX);
                    let bank = Tensor::new(
                        handles.bind(view, 0, view.bytes())?,
                        slots,
                        u32::try_from(stride).unwrap_or(u32::MAX),
                        dtype,
                    );
                    let maps = self.maps.as_ref().ok_or_else(|| Fault::Unbound {
                        what: "the state rows' segment tables".to_string(),
                    })?;
                    let segments = Tensor::new(
                        handles.bind(
                            maps.view(),
                            maps.at(state_row * super::state_map_bytes(self.pool.parts())),
                            super::state_map_bytes(self.pool.parts()),
                        )?,
                        1,
                        (StateMap::HEAD + self.pool.parts()) as u32,
                        Dtype::U64,
                    );
                    state_row += 1;
                    band_of.push(Some(band));
                    tables.push(None);
                    lane_slots.push(Some(
                        seats
                            .lane_bands
                            .get(band)
                            .copied()
                            .unwrap_or(seats.slot_ids),
                    ));
                    CachePool::Recurrent(RecurrentPool {
                        state: bank,
                        slots: seats
                            .row_bands
                            .get(band)
                            .copied()
                            .unwrap_or(seats.slot_of_row),
                        conv_state: bank,
                        new_conv_state: bank,
                        segments,
                    })
                }
            });
        }
        Ok(CacheTable {
            pools,
            compressed,
            lane_slots,
            band_of,
            tables,
        })
    }

    /// Zeroes the record whose physical pieces are `record`: a fresh lane's
    /// state.
    /// A fresh record zeroed on the device in `frame` ([`Self::clear`]'s, in
    /// queue order): the fire that first reads it zeroes it first, and no
    /// frame in flight is drained for it.
    pub fn clear_into(&self, frame: &mut crate::device::ctx::Frame, record: &[u32]) -> Result<()> {
        if !self.has_state() {
            return Ok(());
        }
        let fixed = *self.shape(FIXED)?;
        for &piece in record {
            let at = self.physical_slot_at(&fixed, piece)?;
            self.pool.zero_in(frame, at, fixed.slot_bytes)?;
        }
        Ok(())
    }

    pub fn clear(&mut self, record: &[u32]) -> Result<()> {
        if !self.has_state() {
            return Ok(());
        }
        let fixed = *self.shape(FIXED)?;
        for &piece in record {
            let at = self.physical_slot_at(&fixed, piece)?;
            self.pool.zero(at, fixed.slot_bytes)?;
        }
        Ok(())
    }

    /// Every token run's place in a list slot: its offset and the bytes a
    /// token takes there (a kv row's planes, a compressor's state planes).
    fn kv_runs(&self) -> Vec<(u64, u64)> {
        let mut runs = Vec::new();
        for row in &self.rows {
            match *row {
                Row::Kv {
                    paging: TablePaging::List,
                    keys,
                    values,
                    keys_cell,
                    values_cell,
                    ..
                } => {
                    runs.push((keys, keys_cell));
                    if values != keys {
                        runs.push((values, values_cell));
                    }
                }
                Row::Compressed {
                    paging: TablePaging::List,
                    width,
                    kv,
                    score,
                    ..
                } => {
                    let cell = width * u64::from(elem_size(Dtype::Bf16));
                    runs.push((kv, cell));
                    runs.push((score, cell));
                }
                _ => {}
            }
        }
        runs
    }

    /// Copies token runs between kv pages (slot refs of the list shape).
    pub fn copy_kv(&mut self, frame: &mut Frame, moves: &[KvRun]) -> Result<()> {
        let page_size = u64::from(self.paging.page_size);
        let mut placed = Vec::with_capacity(moves.len());
        for span in moves {
            for (_, token) in [
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
            }
            let src = self.physical(LIST, span.src_page)?;
            let dst = self.physical(LIST, span.dst_page)?;
            placed.push((src, dst, *span));
        }
        let list = *self.shape(LIST)?;
        let runs = self.kv_runs();
        for (src, dst, span) in placed {
            if span.tokens == 0 {
                continue;
            }
            let src_at = self.physical_slot_at(&list, src)?;
            let dst_at = self.physical_slot_at(&list, dst)?;
            for &(offset, cell) in &runs {
                let bytes = u64::from(span.tokens) * cell;
                let from = src_at + offset + u64::from(span.src_token) * cell;
                let to = dst_at + offset + u64::from(span.dst_token) * cell;
                if from != to {
                    self.pool.copy(frame, from, to, bytes)?;
                }
            }
        }
        Ok(())
    }

    /// A kv page's bytes, as the slot file holds one: a list slot.
    pub fn page_bytes(&self) -> Result<u64> {
        Ok(self.layout.list.map_or(0, |list| list.slot_bytes))
    }

    /// Copies whole fixed slots (slot refs) `(src, dst)`.
    pub fn copy_state(&mut self, frame: &mut Frame, moves: &[(u32, u32)]) -> Result<()> {
        if moves.is_empty() || !self.has_state() {
            return Ok(());
        }
        self.copy_whole(frame, FIXED, moves)
    }

    /// Copies whole slots of shape `id` (slot refs) `(src, dst)` on the
    /// device: a fire's stated tail copies.
    pub fn copy_shape(&self, frame: &mut Frame, id: ShapeId, moves: &[(u32, u32)]) -> Result<()> {
        self.copy_whole(frame, id, moves)
    }

    /// Copies whole slots of `id` (slot refs) `(src, dst)` on the device.
    fn copy_whole(&self, frame: &mut Frame, id: ShapeId, moves: &[(u32, u32)]) -> Result<()> {
        let shape = *self.shape(id)?;
        for &(src, dst) in moves {
            let from = self.slot_offset(id, src)?;
            let to = self.slot_offset(id, dst)?;
            self.pool
                .admit(u32::try_from(from / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
            self.pool
                .admit(u32::try_from(to / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
            if from != to {
                self.pool.copy(frame, from, to, shape.slot_bytes)?;
            }
        }
        Ok(())
    }

    /// Slots of one shape copied as [`engine::Engine::copy_slots`] states
    /// them: on the device (whole slots, and token runs of the list), or
    /// between the device and the slot files.
    pub fn copy_slots(&mut self, frame: &mut Frame, copy: &SlotCopy) -> Result<()> {
        let device = MemoryDomain::MetalShared;
        match (copy.src, copy.dst) {
            (src, dst) if src == device && dst == device => {
                self.copy_whole(frame, copy.shape, &copy.slots)?;
                if !copy.runs.is_empty() {
                    if copy.shape != LIST {
                        return Err(Fault::Unbound {
                            what: format!(
                                "token runs of slot shape {}, which holds no tokens",
                                copy.shape
                            ),
                        });
                    }
                    let runs: Vec<KvRun> = copy
                        .runs
                        .iter()
                        .map(|run| KvRun {
                            src_page: run.src,
                            src_token: run.src_token,
                            dst_page: run.dst,
                            dst_token: run.dst_token,
                            tokens: run.tokens,
                        })
                        .collect();
                    self.copy_kv(frame, &runs)?;
                }
                Ok(())
            }
            (src, dst) if !copy.runs.is_empty() => Err(Fault::Unbound {
                what: format!(
                    "token runs from {src:?} to {dst:?}: between tiers a slot moves whole"
                ),
            }),
            (src, MemoryDomain::HostPinned) if src == device => {
                self.park(copy.shape, true, &copy.slots)
            }
            (MemoryDomain::HostPinned, dst) if dst == device => {
                self.park(copy.shape, false, &copy.slots)
            }
            (src, MemoryDomain::LocalDisk) if src == device => {
                let (ids, slots): (Vec<u32>, Vec<u32>) = copy.slots.iter().copied().unzip();
                self.spill(copy.shape, true, &ids, &slots)
            }
            (MemoryDomain::LocalDisk, dst) if dst == device => {
                let (slots, ids): (Vec<u32>, Vec<u32>) = copy.slots.iter().copied().unzip();
                self.spill(copy.shape, false, &ids, &slots)
            }
            (MemoryDomain::HostPinned, MemoryDomain::LocalDisk) => {
                let (hosts, slots): (Vec<u32>, Vec<u32>) = copy.slots.iter().copied().unzip();
                self.spill_host(copy.shape, true, &hosts, &slots)
            }
            (MemoryDomain::LocalDisk, MemoryDomain::HostPinned) => {
                let (slots, hosts): (Vec<u32>, Vec<u32>) = copy.slots.iter().copied().unzip();
                self.spill_host(copy.shape, false, &hosts, &slots)
            }
            (src, dst) => Err(Fault::Unbound {
                what: format!(
                    "a slot copy from {src:?} to {dst:?}: this pool moves within itself, to \
                     its host tier and to its slot files"
                ),
            }),
        }
    }

    /// The bytes one disk chunk takes: a pool chunk, on the slot file's
    /// direct I/O boundary (CONTRACT-BC 22).
    #[must_use]
    pub fn disk_chunk_bytes(&self) -> u64 {
        self.layout.chunk_bytes.next_multiple_of(engine::disk::ALIGN as u64)
    }

    /// The disk tier's chunks.
    #[must_use]
    pub fn disk_chunks(&self) -> u32 {
        Spill::slots(self.disk.as_ref())
    }

    /// The byte of the slot file disk ref `slot` of shape `id` starts at:
    /// slot `index` of disk chunk `chunk`, as on the device.
    fn disk_place(&self, id: ShapeId, slot: u32) -> Result<u64> {
        let shape = *self.shape(id)?;
        let per_chunk = shape.per_chunk.max(1);
        let (chunk, index) = (slot / per_chunk, slot % per_chunk);
        let have = self.disk_chunks();
        if chunk >= have {
            return Err(Fault::Ceiling {
                what: "chunks of the disk tier",
                need: u64::from(chunk) + 1,
                have: u64::from(have),
            });
        }
        Ok(u64::from(chunk) * self.disk_chunk_bytes() + u64::from(index) * shape.slot_bytes)
    }

    /// Copies whole slots of `id` (slot refs `ids`) and disk refs `slots`
    /// the way `to_disk` says. The caller has drained every frame.
    fn spill(&mut self, id: ShapeId, to_disk: bool, ids: &[u32], slots: &[u32]) -> Result<()> {
        let bytes = self.shape(id)?.slot_bytes;
        let places = slots
            .iter()
            .map(|&slot| self.disk_place(id, slot))
            .collect::<Result<Vec<_>>>()?;
        let offsets = ids
            .iter()
            .map(|&slot| self.slot_offset(id, slot))
            .collect::<Result<Vec<_>>>()?;
        for &at in &offsets {
            self.pool
                .admit(u32::try_from(at / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
        }
        let Some(spill) = self.disk.as_mut() else {
            return Ok(());
        };
        let pool = &mut self.pool;
        for (&at, &disk_at) in offsets.iter().zip(&places) {
            spill.copy(disk_at, bytes, to_disk, |slot, out| {
                if out {
                    pool.read(at, slot)
                } else {
                    pool.write(at, slot)
                }
            })?;
        }
        Ok(())
    }

    /// Copies whole slots of `id` between the host tier (host refs
    /// `hosts`) and disk refs `slots` the way `to_disk` says: the host
    /// tier's idle state going on down to disk, and back.
    fn spill_host(&mut self, id: ShapeId, to_disk: bool, hosts: &[u32], slots: &[u32]) -> Result<()> {
        let shape = *self.shape(id)?;
        let places = slots
            .iter()
            .map(|&slot| self.disk_place(id, slot))
            .collect::<Result<Vec<_>>>()?;
        let Some(spill) = self.disk.as_mut() else {
            return Ok(());
        };
        let Some(tier) = self.tier.as_mut() else {
            return Err(Fault::Ceiling {
                what: "chunks of the host tier",
                need: 1,
                have: 0,
            });
        };
        for (&host, &disk_at) in hosts.iter().zip(&places) {
            spill.copy(disk_at, shape.slot_bytes, to_disk, |slot, out| {
                if out {
                    tier.read(&shape, host, slot)
                } else {
                    tier.slot(&shape, host)?.copy_from_slice(slot);
                    Ok(())
                }
            })?;
        }
        Ok(())
    }

    /// Opens the slot file under `dir` within `budget` bytes: disk chunks of
    /// [`Self::disk_chunk_bytes`], any shape's slots in any chunk, laid out
    /// as on the device. The disk chunks it holds.
    pub fn seat_disk(&mut self, dir: &std::path::Path, budget: u64) -> Result<u32> {
        let chunk = self.disk_chunk_bytes();
        let widest = self.layout.shapes().map(|shape| shape.slot_bytes).max().unwrap_or(0);
        if chunk > 0 && widest > 0 {
            self.disk = Spill::open(dir, "r0.state", chunk, super::cover_room(widest), budget)?;
        }
        Ok(self.disk_chunks())
    }

    /// Checks the chunk kv list page id `id` names backed.
    pub fn admit_list_page(&self, id: u32) -> Result<()> {
        let list = *self.shape(LIST)?;
        self.pool
            .admit(self.pool.page_slot(id, list.per_chunk) / list.per_chunk.max(1))
    }

    /// Checks the chunk physical fixed slot `slot` lies in backed, where
    /// the pool holds state.
    pub fn admit_state_slot(&self, slot: u32) -> Result<()> {
        match self.layout.fixed {
            Some(fixed) if self.has_state() => self.pool.admit(slot / fixed.per_chunk.max(1)),
            _ => Ok(()),
        }
    }

    /// Backs the pool's first `segments` segments.
    pub fn commit_segments(&mut self, segments: u32) -> Result<()> {
        self.pool.commit_segments(segments)
    }

    /// Backs the segments the seat runs lie in, and those a fire's `kv`
    /// list slots and `state` fixed slots reach past them: what a caller
    /// firing directly asks, with no runtime committing ahead of it.
    pub fn commit_seats(&mut self, kv: u32, state: u32) -> Result<()> {
        let reach = |slots: u32, shape: Option<crate::store::layout::Slots>| {
            shape.map_or(0, |shape| slots.div_ceil(shape.per_chunk.max(1)))
        };
        let chunks = self
            .layout
            .free_chunk()
            .max(reach(kv, self.layout.list))
            .max(reach(state, self.layout.fixed));
        let segments = chunks.div_ceil(self.layout.segment_chunks.max(1));
        // The bottom grows to the segment the span starts in; state past it
        // lies in that segment, which the span's commit lends to state.
        let bottom = self.pool.segments() - self.pool.span_segments();
        self.pool.commit_segments(segments.min(bottom))?;
        if segments > bottom {
            self.pool.commit_span(self.pool.span_segments())?;
        }
        Ok(())
    }

    /// The bottom run of segments backed for state.
    #[must_use]
    pub fn committed_segments(&self) -> u32 {
        self.pool.committed_segments()
    }

    /// Gives back every segment at and past `segments`.
    ///
    /// # Safety
    ///
    /// No frame reads them: the caller drained every frame, and the
    /// runtime evacuated what they held.
    pub unsafe fn release_segments(&mut self, segments: u32) -> Result<()> {
        // SAFETY: the caller's contract above.
        unsafe { self.pool.release_segments(segments) }
    }

    /// Moves each evacuated chunk onto its new physical chunk in `frame`
    /// and takes the moves into the chunk table; the weight cache's address
    /// tables follow (no frame is in flight: the caller drained them).
    /// A refusal from the pool's own evacuate moved nothing (its table is
    /// swapped only once every copy is encoded): the runtime takes the
    /// evacuation back. Past it the table has moved, so the evacuation is
    /// done whatever follows: a weight cache that fails to restage is a
    /// fault of its own ([`Self::weights_ready`]), never a refusal, which
    /// would have the runtime roll back a move this engine made.
    pub fn evacuate(&mut self, frame: &mut Frame, evacuation: &Evacuation) -> Result<()> {
        self.pool.evacuate(frame, evacuation)?;
        if let Err(fault) = self.restage_weights() {
            tracing::error!(
                %fault,
                "metal: an evacuation moved the pool's chunks and the weight caches' tables could \
                 not follow; every later fire is refused"
            );
            self.weights_stale = Some(fault.to_string());
        }
        Ok(())
    }

    /// Refused once an evacuation left the weight caches' tables naming
    /// chunks it moved ([`Self::evacuate`]): read before every fire.
    pub fn weights_ready(&self) -> Result<()> {
        match &self.weights_stale {
            None => Ok(()),
            Some(why) => Err(Fault::Residency(format!(
                "an evacuation moved the pool's chunks and restaging the weight caches' tables \
                 after it failed ({why}): their tables name moved chunks, so no fire reads them"
            ))),
        }
    }

    /// Takes the weight cache's row-streamed tables (`super::rows`).
    /// Seats the weight caches, each one's tables laid in the tables run
    /// and its handles pointed there.
    pub fn seat_weights(
        &mut self,
        device: &Context,
        handles: &Handles,
        mut caches: Vec<super::streamed::Cache>,
    ) -> Result<()> {
        for cache in &mut caches {
            cache.lay(handles, |bytes| self.run_table(device, bytes))?;
        }
        self.weight_caches = caches;
        Ok(())
    }

    #[must_use]
    pub fn weight_caches(&self) -> &[super::streamed::Cache] {
        &self.weight_caches
    }

    /// The byte offset in the pool of weight slot ref `slot` of a shape of
    /// `per_chunk` slots of `slot_bytes`: refused on the null chunk, past
    /// the pool, or on a chunk with no memory behind it.
    fn weight_slot(&self, per_chunk: u32, slot_bytes: u64, slot: u32) -> Result<u64> {
        let (chunk, index) = (slot / per_chunk.max(1), slot % per_chunk.max(1));
        if chunk <= self.layout.tables || chunk >= self.layout.total_chunks {
            return Err(Fault::Ceiling {
                what: "chunks a weight slot ref may name (past the null chunk and the tables run)",
                need: u64::from(chunk) + 1,
                have: u64::from(self.layout.total_chunks),
            });
        }
        let physical = self.pool.physical(chunk)?;
        self.pool.admit(physical)?;
        Ok(u64::from(physical) * self.layout.chunk_bytes + u64::from(index) * slot_bytes)
    }

    /// See `Engine::weight_residency`: blocks newly named a slot are copied
    /// into it in `frame`. The caller has drained every frame, commits
    /// `frame`, then calls [`Pools::restage_weights`].
    pub fn weight_residency(
        &mut self,
        frame: &mut Frame,
        residency: &engine::chunk::WeightResidency,
    ) -> Result<()> {
        if self.weights_held && !residency.held {
            return Err(Fault::Unbound {
                what: "a weight residency that is not held after one that was: a held set is \
                       the load's for life"
                    .to_string(),
            });
        }
        if let Some(stray) = residency.resident.iter().find(|resident| {
            !self
                .weight_caches
                .iter()
                .any(|cache| cache.shape.id == resident.atom.shape)
        }) {
            return Err(Fault::Unbound {
                what: format!(
                    "weight shape {}, which this load does not state",
                    stray.atom.shape
                ),
            });
        }
        // An atom the load seated that the runtime names another slot for
        // is filled there again: said, since a runtime that adopted the
        // load's seating moves none.
        if self
            .weight_caches
            .iter()
            .any(super::streamed::Cache::preseated)
        {
            let moved = residency
                .resident
                .iter()
                .filter(|resident| {
                    self.weight_caches
                        .iter()
                        .find(|cache| cache.shape.id == resident.atom.shape)
                        .and_then(|cache| cache.slots().get(resident.atom.atom as usize))
                        .is_some_and(|held| *held != Some(resident.slot))
                })
                .count();
            if moved > 0 {
                tracing::warn!(
                    moved,
                    named = residency.resident.len(),
                    "the first weight residency names atoms a slot other than the one the load \
                     seated them in (`WeightShape::seated`); each is filled again"
                );
            }
            for cache in &mut self.weight_caches {
                cache.set_preseated(false);
            }
        }
        self.weights_held = residency.held;
        let mut caches = std::mem::take(&mut self.weight_caches);
        let mut fills = Vec::new();
        let taken = (|| -> Result<()> {
            for cache in &mut caches {
                let shape = cache.shape.clone();
                let mut named: Vec<Option<u32>> = vec![None; shape.count as usize];
                for resident in residency
                    .resident
                    .iter()
                    .filter(|resident| resident.atom.shape == shape.id)
                {
                    let Some(seat) = named.get_mut(resident.atom.atom as usize) else {
                        return Err(Fault::Ceiling {
                            what: "atoms of a weight shape",
                            need: u64::from(resident.atom.atom) + 1,
                            have: u64::from(shape.count),
                        });
                    };
                    *seat = Some(resident.slot);
                }
                let offset = |slot: u32| self.weight_slot(shape.per_chunk, shape.slot_bytes, slot);
                let into = |at: u64| self.pool.view_at(at);
                fills.extend(cache.take(frame, named, &into, &offset)?);
            }
            Ok(())
        })();
        self.weight_caches = caches;
        taken?;
        // Slots filled from the serving file, on the host: no frame reads
        // them (the caller drained every frame).
        self.fill_from_files(&fills)
    }

    /// Writes `fills` into the pool on the host. The file is read a window
    /// of up to 64 MiB at a time (a run's entries follow one another in the
    /// file, and so do a plane's runs); each entry is written through its
    /// chunk's host address where the chunk has one, else afterwards
    /// through the pool. The planes are spread over threads: their entries
    /// are small (a row block of one table) and land in disjoint bytes.
    /// No frame in flight reads the pool.
    fn fill_from_files(&mut self, fills: &[super::streamed::FileFills]) -> Result<()> {
        const RUN: u64 = 64 << 20;
        if fills.is_empty() {
            return Ok(());
        }
        let chunk_bytes = self.layout.chunk_bytes;
        let chunk_len = usize::try_from(chunk_bytes).unwrap_or(0);
        // Every touched chunk's host address (as an integer, to cross
        // threads), or none where the chunk spans two of the sparse pool's
        // heaps.
        let mut bases: Vec<Option<Option<usize>>> = vec![None; self.layout.total_chunks as usize];
        for group in fills {
            for run in &group.runs {
                for to in [run.to, run.to + (run.count - 1) * run.stride] {
                    let physical = usize::try_from(to / chunk_bytes).unwrap_or(usize::MAX);
                    if let Some(seen) = bases.get_mut(physical)
                        && seen.is_none()
                    {
                        *seen = Some(
                            self.pool
                                .chunk_host(u32::try_from(physical).unwrap_or(u32::MAX))
                                .ok()
                                .map(|base| base.as_ptr() as usize),
                        );
                    }
                }
            }
        }
        let bases = &bases;
        // One group's entries; those with no host address come back as
        // `(to, bytes)` for the pool to write.
        let fill =
            |group: &super::streamed::FileFills, bytes: &mut Vec<u8>| -> Result<Vec<Unhosted>> {
                let mut left = Vec::new();
                let (mut window, mut held) = (0u64, 0u64);
                for (at, run) in group.runs.iter().enumerate() {
                    let len = usize::try_from(run.len).unwrap_or(usize::MAX);
                    for entry in 0..run.count {
                        let from = run.from + entry * run.len;
                        let to = run.to + entry * run.stride;
                        if from < window || from + run.len > window + held {
                            // The window: from here through the runs that
                            // follow on in the file, at most `RUN` (or this
                            // run's rest, if longer).
                            let mut end = run.from + run.bytes();
                            for next in &group.runs[at + 1..] {
                                if next.from != end || end + next.bytes() - from > RUN {
                                    break;
                                }
                                end += next.bytes();
                            }
                            group.read(from, end - from, bytes)?;
                            (window, held) = (from, end - from);
                        }
                        let start = usize::try_from(from - window).unwrap_or(usize::MAX);
                        let piece = &bytes[start..start + len];
                        let within = usize::try_from(to % chunk_bytes).unwrap_or(usize::MAX);
                        let physical = usize::try_from(to / chunk_bytes).unwrap_or(usize::MAX);
                        match bases.get(physical).copied().flatten().flatten() {
                            Some(base) if within + len <= chunk_len => {
                                // SAFETY: `within + len` bytes inside the chunk
                                // the address names, committed (`chunk_host`
                                // admitted it), which no frame in flight reads;
                                // each entry's bytes are its own (no other
                                // entry of any group writes them).
                                #[allow(unsafe_code)]
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        piece.as_ptr(),
                                        (base as *mut u8).add(within),
                                        len,
                                    );
                                }
                            }
                            _ => left.push((to, piece.to_vec())),
                        }
                    }
                }
                Ok(left)
            };
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .clamp(1, 8)
            .min(fills.len());
        let per = fills.len().div_ceil(threads);
        let left: Vec<Result<Vec<Unhosted>>> = std::thread::scope(|scope| {
            let workers: Vec<_> = fills
                .chunks(per)
                .map(|part| {
                    scope.spawn(move || -> Result<Vec<Unhosted>> {
                        let mut left = Vec::new();
                        let mut bytes = Vec::new();
                        for group in part {
                            left.extend(fill(group, &mut bytes)?);
                        }
                        Ok(left)
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().expect("a fill thread does not panic"))
                .collect()
        });
        for part in left {
            for (to, piece) in part? {
                self.pool.write(to, &piece)?;
            }
        }
        Ok(())
    }

    /// The residency the load seats its weights by before the runtime names
    /// any slot: every atom of every shape, each shape in chunks of its
    /// own, from the first segment above the seats' runs (which a
    /// seat-addressed fire writes) where they fit there, else from chunk 1
    /// (a seat-addressed fire over them is then refused, `weights_in_seats`;
    /// the runtime names slot refs). `None` where the pool cannot hold them
    /// all.
    #[must_use]
    pub fn preseat_plan(&self) -> Option<engine::chunk::WeightResidency> {
        let segment = self.layout.segment_chunks.max(1);
        let above = self.layout.free_chunk().div_ceil(segment) * segment;
        [above, 1 + self.layout.tables]
            .into_iter()
            .find_map(|first| self.preseat_from(first))
    }

    /// Every atom of every shape from chunk `first` on, each shape in chunks
    /// of its own; `None` past the pool.
    fn preseat_from(&self, mut first: u32) -> Option<engine::chunk::WeightResidency> {
        let mut resident = Vec::new();
        for cache in &self.weight_caches {
            let per_chunk = cache.shape.per_chunk.max(1);
            resident.extend(
                (0..cache.shape.count).map(|atom| engine::chunk::ResidentWeight {
                    atom: engine::chunk::WeightAtom {
                        shape: cache.shape.id,
                        atom,
                    },
                    slot: first * per_chunk + atom,
                }),
            );
            first = first.checked_add(cache.shape.count.div_ceil(per_chunk))?;
        }
        (!resident.is_empty() && first <= self.layout.total_chunks).then_some(
            engine::chunk::WeightResidency {
                staging: Vec::new(),
                resident,
                held: false,
            },
        )
    }

    /// Seats the load's weights by `plan` ([`Pools::preseat_plan`]) in
    /// `frame`: the segments backed, each slot filled, every cache marked
    /// as seated by the load. The caller commits `frame`, then restages.
    pub fn preseat_weights(
        &mut self,
        frame: &mut Frame,
        plan: &engine::chunk::WeightResidency,
    ) -> Result<()> {
        let last = plan
            .resident
            .iter()
            .map(|resident| {
                let per_chunk = self
                    .weight_caches
                    .iter()
                    .find(|cache| cache.shape.id == resident.atom.shape)
                    .map_or(1, |cache| cache.shape.per_chunk.max(1));
                resident.slot / per_chunk
            })
            .max()
            .unwrap_or(0);
        // Backed through the chunks the load seats the weights in (the
        // runtime backs what else it seats beside them, `commit_segments`).
        let segment = self.layout.segment_chunks.max(1);
        let segments = (last + 1).div_ceil(segment).min(self.pool.segments());
        self.pool.commit_segments(segments)?;
        self.weight_residency(frame, plan)?;
        for cache in &mut self.weight_caches {
            cache.set_preseated(true);
        }
        Ok(())
    }

    /// The weight caches, to mark their own copies' residency.
    pub fn weight_caches_mut(&mut self) -> &mut [super::streamed::Cache] {
        &mut self.weight_caches
    }

    /// Rewrites every weight address table for where its blocks lie now.
    /// No frame is in flight.
    pub fn restage_weights(&mut self) -> Result<()> {
        let mut caches = std::mem::take(&mut self.weight_caches);
        let staged = caches.iter_mut().try_for_each(|cache| {
            let (per_chunk, slot_bytes) = (cache.shape.per_chunk, cache.shape.slot_bytes);
            cache.restage(&|slot| {
                Ok(self
                    .pool
                    .address(self.weight_slot(per_chunk, slot_bytes, slot)?))
            })
        });
        self.weight_caches = caches;
        staged
    }

    /// Whether the runtime holds the weight set for the load's life and
    /// names every atom of every shape: the tables' own copies may go.
    #[must_use]
    pub fn weights_whole(&self) -> bool {
        self.weights_held
            && !self.weight_caches.is_empty()
            && self.weight_caches.iter().all(super::streamed::Cache::whole)
    }

    /// Gives back every table's own copy (see [`super::streamed::Cache::release`]).
    pub fn release_weight_hosts(&mut self) -> Vec<(usize, u32, crate::device::alloc::Slab)> {
        if !self.weights_whole() {
            return Vec::new();
        }
        self.weight_caches
            .iter_mut()
            .flat_map(super::streamed::Cache::release)
            .collect()
    }

    /// See `Engine::weight_uses`. No frame is in flight.
    pub fn weight_uses(&mut self) -> Result<engine::chunk::WeightUses> {
        let mut uses = Vec::new();
        for cache in &mut self.weight_caches {
            cache.uses(&mut uses)?;
        }
        Ok(engine::chunk::WeightUses {
            uses,
            held: self
                .weight_caches
                .iter()
                .map(super::streamed::Cache::held)
                .collect(),
        })
    }

    /// Whether a weight slot is named in a chunk the seat runs take, where a
    /// seat-addressed lane (a caller firing directly) writes kv and state
    /// the runtime does not see.
    #[must_use]
    pub fn weights_in_seats(&self) -> bool {
        let seats = 1 + self.layout.tables..self.layout.free_chunk();
        self.weight_caches.iter().any(|cache| {
            cache
                .slots()
                .iter()
                .flatten()
                .any(|slot| seats.contains(&(slot / cache.shape.per_chunk.max(1))))
        })
    }
}


#[cfg(all(test, target_vendor = "apple"))]
mod delta_table_tests {
    use super::Pools;
    use crate::device::Context;

    const CHUNK: u64 = 16 << 20;

    /// The buffers' delta table, through which every kernel reaches a kv
    /// page past the first buffer, lies in a buffer of its own: no pool
    /// buffer holds its address, it holds the buffers' deltas, and every
    /// byte of the pool overwritten leaves it as it was. Laid in the pool's
    /// tables run it was overwritten on the M5 (bf16 rows where the deltas
    /// belong), and every buffer-1 kv write landed elsewhere.
    #[test]
    fn the_delta_table_lies_outside_the_pool_and_survives_any_write_into_it() {
        let device = Context::bind().expect("a Metal 4 device");
        super::super::pool::cap_parts(CHUNK);
        let pools = Pools::listed_over(&device, CHUNK, 4, 1, CHUNK / 4, 0);
        super::super::pool::cap_parts(0);
        let mut pools = pools.expect("a pool");
        assert_eq!(pools.parts(), 4, "a buffer a chunk");
        let table = pools.delta_table().expect("a kv list over four buffers has a delta table");
        let want = pools.deltas();
        assert!(want.iter().skip(1).any(|&delta| delta != 0), "the buffers lie apart: {want:x?}");
        let held = |pools: &Pools| -> Vec<u64> {
            let mut bytes = vec![0u8; want.len() * 8];
            pools
                .deltas
                .as_ref()
                .expect("the table")
                .read(0, &mut bytes)
                .expect("the table reads");
            bytes
                .chunks_exact(8)
                .map(|word| u64::from_le_bytes(word.try_into().expect("8 bytes")))
                .collect()
        };
        assert_eq!(held(&pools), want, "the table holds the buffers' deltas");
        for at in 0..pools.parts() {
            let view = pools.pool.part_view(at);
            let base = crate::device::alloc::slab_address(view.slab());
            assert!(
                !(base..base + view.bytes()).contains(&table),
                "buffer {at} of the pool holds the delta table"
            );
        }
        // Every byte of the pool, overwritten with bf16 rows.
        let rows = vec![0x87u8, 0x3e, 0x3b, 0xbe].repeat((CHUNK / 4) as usize);
        for chunk in 0..4u64 {
            pools.pool.write(chunk * CHUNK, &rows).expect("a chunk overwritten");
        }
        assert_eq!(held(&pools), want, "no write into the pool moves a page");
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod tables_run_tests {
    use super::Pools;
    use crate::device::Context;
    use crate::error::Fault;

    const CHUNK: u64 = 16 << 20;

    /// Chunks 1..1+tables are the engine's: a kv page, a copy's end or a
    /// held table naming one is refused where its slot is placed, by name,
    /// and the first slot past the run is placed as any other.
    #[test]
    fn a_slot_in_the_tables_run_is_refused_where_it_is_placed() {
        let device = Context::bind().expect("a Metal 4 device");
        let pools = Pools::listed_over(&device, CHUNK, 8, 8, CHUNK, 2).expect("a pool");
        for chunk in 1..3u32 {
            let refused = pools.kv_pages(&[chunk], None).err();
            assert!(
                matches!(&refused, Some(Fault::Residency(why)) if why.contains("tables run")),
                "chunk {chunk}: {refused:?}"
            );
        }
        let past = pools.kv_pages(&[3], None).expect("the first slot past the tables run");
        assert_eq!(past.len(), 1);
    }
}

/// One contiguous piece of a guest's state in the pool ([`Pools::guest_pieces`]).
#[derive(Debug, Clone)]
pub struct GuestPiece {
    /// The buffer it lies in, and its offset there.
    pub view: Buffer,
    pub local: u64,
    /// Its offset in the pool: what its host runs are asked by.
    pub global: u64,
    pub bytes: u64,
}

#[cfg(all(test, target_vendor = "apple"))]
mod copy_landing_tests {
    use super::Pools;
    use crate::device::Context;
    use crate::store::layout::CHANNEL;
    use engine::MemoryDomain;
    use engine::chunk::SlotCopy;

    const CHUNK: u64 = 16 << 20;
    const RUN: u32 = 8;

    /// A copy on the device is told landed from its command buffer's
    /// completion (`Shell::copy_slots_landing`), once it has read its
    /// source slots: what the runtime gives those slots to next (written by
    /// anything, from the host and ordered behind nothing) cannot reach the
    /// copied bytes. Each round the source slots are overwritten the moment
    /// the copy is told landed, and the destination holds what they held.
    #[test]
    fn a_device_copy_is_told_landed_once_it_has_read_its_slots() {
        let device = Context::bind().expect("a Metal 4 device");
        let mut pools = Pools::guests_only(&device, CHUNK, 24).expect("a pool");
        let per = pools.slots_per_chunk(CHANNEL).expect("a channel shape");
        let bytes = usize::try_from(pools.slot_bytes(CHANNEL).expect("a slot")).expect("a usize");
        let src: Vec<u32> = (2 * per..2 * per + RUN).collect();
        let dst: Vec<u32> = (3 * per..3 * per + RUN).collect();
        let at = |pools: &Pools, slot: u32| pools.slot_offset(CHANNEL, slot).expect("a placed slot");
        for round in 0..8u8 {
            let pattern: Vec<u8> = (0..bytes * RUN as usize)
                .map(|byte| (byte * 13 + 7) as u8 ^ (byte >> 9) as u8 ^ round)
                .collect();
            for (k, &slot) in src.iter().enumerate() {
                let offset = at(&pools, slot);
                pools
                    .pool
                    .write(offset, &pattern[k * bytes..(k + 1) * bytes])
                    .expect("the source slot is laid");
            }
            let copy = SlotCopy {
                shape: CHANNEL,
                src: MemoryDomain::MetalShared,
                dst: MemoryDomain::MetalShared,
                slots: src.iter().copied().zip(dst.iter().copied()).collect(),
                runs: Vec::new(),
            };
            let mut frame = device.frame().expect("a frame");
            pools.copy_slots(&mut frame, &copy).expect("the copy encodes");
            let (told, landed) = std::sync::mpsc::channel();
            let pending = frame
                .commit_async(Some(Box::new(move |_| {
                    let _ = told.send(());
                })))
                .expect("the copy commits");
            landed
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("the copy is told landed");
            // The slots go to their next owner, which writes them at once.
            for &slot in &src {
                let offset = at(&pools, slot);
                pools
                    .pool
                    .write(offset, &vec![0x5a ^ round; bytes])
                    .expect("the next owner writes the slot");
            }
            pending.wait().expect("the copy retired");
            let mut held = vec![0u8; bytes * RUN as usize];
            for (k, &slot) in dst.iter().enumerate() {
                pools
                    .pool
                    .read(at(&pools, slot), &mut held[k * bytes..(k + 1) * bytes])
                    .expect("the destination reads");
            }
            assert!(held == pattern, "round {round}: the destination holds what the source held");
        }
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod run_tests {
    use super::Pools;
    use crate::device::Context;
    use engine::chunk::Evacuation;

    /// A guest piece of several chunks is read on from its first chunk's
    /// physical chunk through the next ones: a run moved whole reads
    /// whatever its logical ids (the runtime keeps them as they were), and a
    /// run whose physical chunks would cross the end of a segment is
    /// refused by name, naming its chunks, rather than read as another
    /// owner's bytes.
    #[test]
    fn a_guest_run_across_a_segment_end_is_refused_naming_its_chunks() {
        let Ok(context) = Context::bind() else {
            eprintln!("no device: skipping");
            return;
        };
        // 24 chunks in segments of 8: a run from chunk 7 crosses one's end.
        let mut pools = Pools::guests_over(&context, 16 << 20, 24, 8).expect("a pool");
        let place = engine::ScratchPlace::Chunks { run: 2, pieces: vec![4] };
        assert_eq!(pools.guest_pieces(&place).expect("an adjacent run reads").len(), 1);
        // Its second chunk's logical id moved elsewhere: the piece still reads
        // the physical chunks after its first, adjacent and in one segment.
        let mut frame = context.frame().expect("a frame");
        pools
            .evacuate(&mut frame, &Evacuation { moves: vec![(5, 10)] })
            .expect("the evacuation");
        frame.commit().expect("the copy lands");
        assert_eq!(
            pools.guest_pieces(&place).expect("a physically adjacent run reads").len(),
            1
        );
        let seg = pools.pool.segment_chunks();
        assert_eq!(seg, 8, "the pool is laid in segments of 8");
        let across = engine::ScratchPlace::Chunks { run: 2, pieces: vec![seg - 1] };
        let refused = format!(
            "{}",
            pools.guest_pieces(&across).expect_err("a run across a segment's end is refused")
        );
        assert!(
            refused.contains(&format!("logical chunk {}", seg - 1))
                && refused.contains(&format!("{}..={}", seg - 1, seg)),
            "the refusal names the chunks: {refused}"
        );
    }
}
