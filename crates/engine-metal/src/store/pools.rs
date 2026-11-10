//! The load's held state in one chunk pool ([`super::pool::Pool`]): every
//! cache row placed in the slot shapes of [`super::layout`], the slot refs the
//! runtime hands out translated to the physical slots kernels address, and
//! the copies, spills and evacuations that move slots.

use engine::chunk::{ChunkFacts, Evacuation, ShapeId, SlotCopy};
use engine::transfer::{KvRun, MemoryDomain};
use kernels_metal::{KvPool, RecurrentPool, StateMap, Tensor};
use poem_compiler::chunk::{ChunkRecipe, RowId, SlotLayout};
use poem_ir::{CacheRow, Dtype, Trace};

use super::layout::{FIXED, LIST, Layout, MAX_WINDOWS, RS, Slots, WINDOWS, gcd};
use super::pool::Pool;
use super::{
    STATE_MAP_BYTES, Seats, Spill, TableSeat, compressed_rows, elem_bytes, elem_size, narrow,
    row_stride, split, state_dtype,
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

#[derive(Debug)]
pub struct Pools {
    paging: Paging,
    rows: Vec<Row>,
    pool: Pool,
    layout: Layout,
    /// Every state row's segment table, one after another.
    maps: Option<Buffer>,
    shape_rows: [Vec<u32>; 2],
    /// Each distinct set of record slots a state row's pieces lie in.
    bands: Vec<Vec<u32>>,
    rs: Option<RsRows>,
    /// The page tables a fire stages, by the list each pages.
    tables: Vec<TablePaging>,
    disk: Option<Spill>,
    parked: Option<Spill>,
    /// The shared pool's host tier (CONTRACT-BC 22), where one is stated.
    tier: Option<HostTier>,
    /// The weight cache's row-streamed tables, a shape each (`super::rows`).
    weight_caches: Vec<super::streamed::Cache>,
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
    /// The bytes of host ref `slot` of `shape`, its chunk allocated first.
    fn slot(&mut self, shape: &Slots, slot: u32) -> Result<&mut [u8]> {
        let per_chunk = shape.per_chunk.max(1);
        let chunk = (slot / per_chunk) as usize;
        let have = self.chunks.len();
        let held = self.chunks.get_mut(chunk).ok_or(Fault::Ceiling {
            what: "chunks of the host tier",
            need: chunk as u64 + 1,
            have: have as u64,
        })?;
        let bytes = usize::try_from(self.chunk_bytes).unwrap_or(usize::MAX);
        let held = held.get_or_insert_with(|| vec![0u8; bytes].into_boxed_slice());
        let at = usize::try_from(u64::from(slot % per_chunk) * shape.slot_bytes).unwrap_or(0);
        let len = usize::try_from(shape.slot_bytes).unwrap_or(0);
        Ok(&mut held[at..at + len])
    }
}

/// The first byte of physical slot `slot` of `shape`.
fn slot_at(layout: &Layout, shape: &Slots, slot: u32) -> u64 {
    u64::from(slot / shape.per_chunk) * layout.chunk_bytes
        + u64::from(slot % shape.per_chunk) * shape.slot_bytes
}

impl Pools {
    /// The pool `layout` sizes over `rows`, kv pages of `paging`'s size.
    pub fn reserve(device: &Context, rows: Rows, layout: Layout, paging: Paging) -> Result<Pools> {
        let pool = Pool::reserve(
            device,
            layout.chunk_bytes,
            layout.total_chunks,
            layout.segment_chunks,
        )?;
        let list_slots = layout.list.map_or(0, |list| {
            u64::from(list.per_chunk) * u64::from(layout.total_chunks)
        });
        let mut pools = Pools {
            paging: Paging {
                pages: list_slots.max(1),
                ..paging
            },
            maps: None,
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
            parked: None,
            tier: None,
            weight_caches: Vec::new(),
            weights_held: false,
        };
        pools.write_maps(device)?;
        Ok(pools)
    }

    /// Writes each state row's segment table: the bank is the whole pool,
    /// a chunk `chunk_bytes` apart, a slot the fixed slot apart, the row at
    /// its offset in each of its pieces' slots; one segment.
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
            let map = StateMap {
                slot_stride: fixed.slot_bytes / element,
                chunk_slots: fixed.per_chunk,
                chunk_stride: self.layout.chunk_bytes / element,
                seg_chunks: u32::MAX,
                piece_atoms,
                pieces: pieces.max(1),
                bases: vec![offset / element],
            };
            tables.extend(map.bytes());
        }
        if tables.is_empty() {
            return Ok(());
        }
        let mut buffer = Buffer::zeroed(device, tables.len() as u64)?;
        buffer.write(0, &tables)?;
        self.maps = Some(buffer);
        Ok(())
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
    /// the verb's base: `refs` (rs-list slot refs) as units of
    /// [`Layout::rs_stride`], or, seat-addressed, seat `seat`'s run.
    pub fn rs_pages(&self, refs: &[u32], seat: Option<u32>) -> Result<Vec<u32>> {
        let rs = *self.shape(RS)?;
        let stride = self.layout.rs_stride();
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
                let offset = self.slot_offset(RS, slot)?;
                self.pool
                    .admit(u32::try_from(offset / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
                Ok(u32::try_from(offset / stride).unwrap_or(u32::MAX))
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
            .map(|cache| cache.shape.facts())
            .collect();
        facts.disk_chunks = self.layout.list.map_or(0, |list| {
            Spill::slots(self.disk.as_ref())
                .checked_div(list.per_chunk)
                .unwrap_or(0)
        });
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
        Ok((self.pool.physical(chunk)?, index))
    }

    /// The physical slot kernels address for slot ref `slot` of `shape`.
    fn physical(&self, id: ShapeId, slot: u32) -> Result<u32> {
        let shape = *self.shape(id)?;
        let (chunk, index) = self.place(&shape, slot)?;
        Ok(chunk * shape.per_chunk + index)
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
                    self.physical(LIST, list.seat_ref(id)?)
                })
                .collect();
        }
        refs.iter().map(|&r| self.physical(LIST, r)).collect()
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

    /// What window list `k`'s page id counts: `gcd(chunk, window slot)`
    /// bytes.
    fn window_stride(&self, k: usize) -> u64 {
        self.layout.window(k).map_or(256, |window| {
            gcd(self.layout.chunk_bytes, window.slots.slot_bytes)
        })
    }

    /// A lane's pages of window list `k` as kernels address them, one a
    /// window slot (`T_w` tokens) in token order: `refs` (window slot refs,
    /// 0 a slot trimmed behind the window: the null chunk) as units of
    /// [`Pools::window_stride`], or, seat-addressed, seat `seat`'s run,
    /// `pages` of them.
    pub fn window_pages(
        &self,
        k: usize,
        refs: &[u32],
        seat: Option<(u32, u32)>,
    ) -> Result<Vec<u32>> {
        let id = WINDOWS[k.min(MAX_WINDOWS - 1)];
        let window = *self.shape(id)?;
        let stride = self.window_stride(k);
        let unit = |slot: u32| -> Result<u32> {
            if slot == 0 {
                return Ok(0);
            }
            let at = self.slot_offset(id, slot)?;
            self.pool
                .admit(u32::try_from(at / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
            Ok(u32::try_from(at / stride).unwrap_or(u32::MAX))
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

    /// The physical page explicit write descriptor page `page` names.
    pub fn kv_page(&self, page: u32) -> Result<u32> {
        self.physical(LIST, page)
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

    /// The state rows of the record whose physical pieces are `record`,
    /// one after another, each row's pieces in order.
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

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.pool.committed_bytes()
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.pool.high_water_bytes()
    }

    #[must_use]
    pub fn is_elastic(&self) -> bool {
        self.pool.is_sparse()
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
                    // ids and page stride.
                    let plane = Tensor::new(
                        handles.bind(view, 0, view.bytes())?,
                        cells,
                        u32::try_from(width).unwrap_or(u32::MAX),
                        Dtype::Bf16,
                    );
                    let stride = match paging {
                        TablePaging::List => list_bytes,
                        TablePaging::Window(k) => self.window_stride(usize::from(k)),
                    };
                    compressed.push((
                        row,
                        PoolSlabs {
                            kv: plane,
                            score: plane,
                            page_stride: [stride, stride],
                            plane_offset: [kv, score],
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
                    let plane = |width: u64| -> Result<Tensor> {
                        Ok(Tensor::new(
                            handles.bind(view, 0, view.bytes())?,
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
                        keys: plane(u64::from(kv_heads) * u64::from(head_dim))?,
                        values: plane(values_width)?,
                        page_indices: seat.page_indices,
                        page_indptr: seat.page_indptr,
                        page_size: narrow(u64::from(page_size)),
                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                        q8,
                        page_stride: [stride, stride],
                        plane_offset: [keys, values],
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
                        handles.bind(maps, state_row * STATE_MAP_BYTES, STATE_MAP_BYTES)?,
                        1,
                        (StateMap::HEAD + 1) as u32,
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
        let view = self.pool.view();
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
                    frame.copy_span(view, from, view, to, bytes)?;
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
        let view = self.pool.view();
        for &(src, dst) in moves {
            let from = self.slot_offset(id, src)?;
            let to = self.slot_offset(id, dst)?;
            self.pool
                .admit(u32::try_from(from / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
            self.pool
                .admit(u32::try_from(to / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
            if from != to {
                frame.copy_span(view, from, view, to, shape.slot_bytes)?;
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
            (src, dst) => Err(Fault::Unbound {
                what: format!(
                    "a slot copy from {src:?} to {dst:?}: this pool moves within itself, to \
                     its host tier and to its slot files"
                ),
            }),
        }
    }

    /// Copies whole slots of `id` (slot refs `ids`) and the slot file's
    /// `slots` the way `to_disk` says. The caller has drained every frame.
    fn spill(&mut self, id: ShapeId, to_disk: bool, ids: &[u32], slots: &[u32]) -> Result<()> {
        let (what, bytes) = match id {
            LIST => ("disk kv pages", self.shape(LIST)?.slot_bytes),
            FIXED => ("parked rs rows", self.shape(FIXED)?.slot_bytes),
            other => {
                return Err(Fault::Unbound {
                    what: format!("slot shape {other}, which this pool does not hold"),
                });
            }
        };
        let offsets = ids
            .iter()
            .map(|&slot| self.slot_offset(id, slot))
            .collect::<Result<Vec<_>>>()?;
        for &at in &offsets {
            self.pool
                .admit(u32::try_from(at / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
        }
        let file = match id {
            LIST => &mut self.disk,
            _ => &mut self.parked,
        };
        Spill::room(file.as_ref(), what, slots)?;
        let Some(spill) = file.as_mut() else {
            return Ok(());
        };
        for (&at, &slot) in offsets.iter().zip(slots) {
            let staged = &mut spill.stage.as_mut_slice()[..usize::try_from(bytes).unwrap_or(0)];
            if to_disk {
                self.pool.read(at, staged)?;
                spill
                    .file
                    .write(slot, spill.stage.as_mut_slice())
                    .map_err(super::slot_file)?;
            } else {
                spill
                    .file
                    .read(slot, spill.stage.as_mut_slice())
                    .map_err(super::slot_file)?;
                let staged = &spill.stage.as_mut_slice()[..usize::try_from(bytes).unwrap_or(0)];
                self.pool.write(at, staged)?;
            }
        }
        Ok(())
    }

    /// Opens the slot files under `dir` within `budget` bytes: parked
    /// record pieces first, as many as the seats' records, then kv pages;
    /// the kv pages they hold.
    pub fn seat_disk(&mut self, dir: &std::path::Path, budget: u64) -> Result<u32> {
        let stride = |bytes: u64| bytes.next_multiple_of(engine::disk::ALIGN as u64);
        let state = self.layout.fixed.map_or(0, |f| f.slot_bytes);
        let rows = if state == 0 {
            0
        } else {
            (u64::from(self.paging.slots) * u64::from(self.layout.pieces))
                .min(budget.checked_div(stride(state)).unwrap_or(0))
        };
        if state > 0 {
            self.parked = Spill::open(dir, "r0.rs", state, rows * stride(state))?;
        }
        let kv = self.page_bytes()?;
        if kv > 0 {
            self.disk = Spill::open(dir, "r0.kv", kv, budget - rows * stride(state))?;
        }
        Ok(Spill::slots(self.disk.as_ref()))
    }

    #[must_use]
    pub fn parked_slots(&self) -> u32 {
        Spill::slots(self.parked.as_ref())
    }

    /// Copies whole kv pages (slot refs) between the pool and the slot
    /// file, `pages[i]` with `slots[i]`.
    pub fn spill_kv(&mut self, to_disk: bool, pages: &[u32], slots: &[u32]) -> Result<()> {
        self.spill(LIST, to_disk, pages, slots)
    }

    /// Copies whole record pieces (slot refs) between the pool and the
    /// slot file.
    pub fn park_state(&mut self, out: bool, device: &[u32], rows: &[u32]) -> Result<()> {
        self.spill(FIXED, out, device, rows)
    }

    /// Checks what a fire touches is committed: `demand`'s kv pages and
    /// state slots are one past the highest physical page and slot it
    /// names. An unmapped tile read by a kernel faults the whole queue.
    pub fn admit(&self, demand: engine::frame::Demand) -> Result<()> {
        if demand.workspace > 0 {
            return Err(Fault::Ceiling {
                what: "pool workspace bytes",
                need: demand.workspace,
                have: 0,
            });
        }
        if demand.kv_pages > 0
            && let Some(list) = self.layout.list
        {
            self.pool.admit((demand.kv_pages - 1) / list.per_chunk)?;
        }
        if demand.state_slots > 0
            && self.has_state()
            && let Some(fixed) = self.layout.fixed
        {
            self.pool
                .admit((demand.state_slots - 1) / fixed.per_chunk)?;
        }
        Ok(())
    }

    /// Backs the pool's first `segments` segments.
    pub fn commit_segments(&mut self, segments: u32) -> Result<()> {
        self.pool.commit_segments(segments)
    }

    /// Backs the segments the seat runs lie in: what a caller firing
    /// directly asks.
    pub fn commit_seats(&mut self) -> Result<()> {
        let chunks = 1 + self.layout.seat_chunks();
        let segments = chunks
            .div_ceil(self.layout.segment_chunks.max(1))
            .min(self.pool.segments());
        self.pool.commit_segments(segments)
    }

    /// Backs every segment up to the highest physical page and slot
    /// `demand` names (one past each, as [`Pools::admit`] reads them).
    pub fn commit_for(&mut self, demand: engine::frame::Demand) -> Result<()> {
        let mut chunk = 0u32;
        if demand.kv_pages > 0
            && let Some(list) = self.layout.list
        {
            chunk = chunk.max((demand.kv_pages - 1) / list.per_chunk);
        }
        if demand.state_slots > 0
            && self.has_state()
            && let Some(fixed) = self.layout.fixed
        {
            chunk = chunk.max((demand.state_slots - 1) / fixed.per_chunk);
        }
        let segments = (chunk + 1)
            .div_ceil(self.layout.segment_chunks.max(1))
            .min(self.pool.segments());
        self.pool.commit_segments(segments)
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
    pub fn evacuate(&mut self, frame: &mut Frame, evacuation: &Evacuation) -> Result<()> {
        self.pool.evacuate(frame, evacuation)?;
        self.restage_weights()
    }

    /// Takes the weight cache's row-streamed tables (`super::rows`).
    pub fn seat_weights(&mut self, caches: Vec<super::streamed::Cache>) {
        self.weight_caches = caches;
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
        if chunk == 0 || chunk >= self.layout.total_chunks {
            return Err(Fault::Ceiling {
                what: "chunks a weight slot ref may name (past the null chunk)",
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
        self.weights_held = residency.held;
        let mut caches = std::mem::take(&mut self.weight_caches);
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
                cache.take(frame, named, self.pool.view(), &offset)?;
            }
            Ok(())
        })();
        self.weight_caches = caches;
        taken
    }

    /// Rewrites every weight address table for where its blocks lie now.
    /// No frame is in flight.
    pub fn restage_weights(&mut self) -> Result<()> {
        let base = crate::device::alloc::slab_address(self.pool.view().slab());
        let mut caches = std::mem::take(&mut self.weight_caches);
        let staged = caches.iter_mut().try_for_each(|cache| {
            let (per_chunk, slot_bytes) = (cache.shape.per_chunk, cache.shape.slot_bytes);
            cache.restage(base, &|slot| self.weight_slot(per_chunk, slot_bytes, slot))
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
        let seats = 1..1 + self.layout.seat_chunks();
        self.weight_caches.iter().any(|cache| {
            cache
                .slots()
                .iter()
                .flatten()
                .any(|slot| seats.contains(&(slot / cache.shape.per_chunk.max(1))))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_starts_at_its_chunk_then_its_index() {
        let layout = Layout::new(1 << 20, 8, 4, 16, [64 << 10, 256 << 10], 1, 2, 4).unwrap();
        let list = layout.list.unwrap();
        assert_eq!(slot_at(&layout, &list, 16), 1 << 20);
        assert_eq!(slot_at(&layout, &list, 17), (1 << 20) + (64 << 10));
        let fixed = layout.fixed.unwrap();
        assert_eq!(slot_at(&layout, &fixed, 5), (1 << 20) + (256 << 10));
    }
}
