//! The load's held state in one chunk pool ([`super::pool::Pool`]): every
//! cache row placed in the slot shapes of [`super::layout`], the slot refs the
//! runtime hands out translated to the physical slots kernels address, and
//! the copies, spills and evacuations that move slots.

use engine::chunk::{ChunkFacts, Evacuation, ShapeId, SlotCopy};
use engine::transfer::{KvRun, MemoryDomain};
use kernels_metal::{KvPool, RecurrentPool, StateMap, Tensor};
use poem_compiler::chunk::{ChunkRecipe, RowId, SlotLayout};
use poem_ir::{CacheRow, Dtype, Trace};

use super::layout::{FIXED, LIST, Layout, RING, Slots, gcd};
use super::pool::Pool;
use super::{
    STATE_MAP_BYTES, Seats, SpaceSeat, Spill, compressed_spaces, elem_bytes, elem_size, narrow,
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
        space: u32,
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
    /// The compressor state a pool gather keeps beside kv space `space`:
    /// side planes `kv` and `score`, a cell a token of every physical list
    /// slot.
    Compressed {
        space: u32,
        width: u64,
        kv: usize,
        score: usize,
    },
}

/// One kv row's geometry as this shell stores it.
struct KvCells {
    space: u32,
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
    space: u32,
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
        space,
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

/// Where the rs buffer lies in a record (the recipe's `RsLayout`): every
/// rs row cut alike into `pieces` pieces of `tokens_per_piece` tokens,
/// piece `p` in record slot `slots[p]`, and each of the shell's rs planes'
/// offset in a piece (its `tokens_per_piece` rows one after another).
/// Buffered row `s` lies in piece `(s % 16) / tokens_per_piece`, row
/// `(s % 16) % tokens_per_piece` of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsRows {
    pub pieces: u32,
    pub tokens_per_piece: u32,
    pub slots: Vec<u32>,
    pub plane_at: Vec<u64>,
}

/// How a kv space's rows are paged: the list, the ring, or a small ring
/// folded into the record (`tokens` tokens in record slot `slot`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpacePaging {
    List,
    Ring,
    Folded { tokens: u32, slot: u32 },
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
    /// The ring slot's bytes, the slots a working set's ring takes, and the
    /// window it is read through; zero where nothing is windowed.
    pub ring_bytes: u64,
    pub ring_cap: u32,
    pub ring_window: u32,
    /// How each kv space is paged, by space.
    pub spaces: Vec<SpacePaging>,
    /// What one physical list slot's tokens take in the side planes.
    pub side_bytes: u64,
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
                    space,
                    head_dim,
                    ..
                } => {
                    let kv = kv_cells(index, name, planes, *dtype, *space, *head_dim, &facts)?;
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
        let side = compressed_spaces(trace)
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
        let rings: Vec<&SlotLayout> = recipe.rings().collect();
        if rings.len() > 1 {
            return Err(disagree(format!(
                "keeps {} ring shapes ({}), where a load pages one",
                rings.len(),
                recipe.shapes()
            )));
        }
        let (ring_tokens, ring_bytes, ring_cap, ring_window, ring_runs) = match rings.first() {
            Some(SlotLayout::Ring {
                tokens,
                bytes,
                cap_slots,
                cap,
                rows,
            }) => (*tokens, *bytes, *cap_slots, cap.window, rows.as_slice()),
            _ => (0, 0, 0, 0, &[][..]),
        };
        let folded: &[(RowId, u32)] = match recipe.fixed() {
            Some(SlotLayout::Fixed { folded, .. }) => folded,
            _ => &[],
        };
        let (tokens, list_bytes, runs) = match recipe.list() {
            Some(SlotLayout::List {
                tokens,
                bytes,
                rows,
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
        // Without a list the ring's slot is the kv page.
        let tokens = if list_bytes == 0 { ring_tokens } else { tokens };
        if ring_bytes > 0 && ring_tokens != tokens {
            return Err(disagree(format!(
                "pages windowed rows {ring_tokens} tokens a ring slot beside {tokens}-token kv \
                 pages, where a ring slot is one kv page"
            )));
        }
        let mut spaces: Vec<Option<SpacePaging>> = Vec::new();
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
                    space,
                    head_dim,
                    ..
                } => {
                    let kv = kv_cells(index, name, planes, *dtype, *space, *head_dim, facts)?;
                    let per_token = kv.keys_cell + if kv.own_values { kv.values_cell } else { 0 };
                    // Where the row lies: a list slot, a ring slot, or the
                    // record (a folded ring), and the tokens a run holds.
                    let (paging, offset, stated, run_tokens) = if let Some(run) =
                        runs.iter().find(|run| run.row == id)
                    {
                        (SpacePaging::List, run.offset, run.per_token, tokens)
                    } else if let Some(run) = ring_runs.iter().find(|run| run.row == id) {
                        (SpacePaging::Ring, run.offset, run.per_token, ring_tokens)
                    } else if let Some(&(_, held)) = folded.iter().find(|(row, _)| *row == id) {
                        let run = fixed_runs.iter().find(|run| run.row == id).ok_or_else(|| {
                            disagree(format!("folds kv row `{name}` into no record slot"))
                        })?;
                        if run.pieces != 1 {
                            return Err(disagree(format!(
                                "cuts folded ring row `{name}` into {} pieces, where this \
                                     shell reads a folded ring whole",
                                run.pieces
                            )));
                        }
                        let bytes = recipe.atoms.get(&id).map_or(0, |a| a.bytes);
                        (
                            SpacePaging::Folded {
                                tokens: held,
                                slot: run.slots[0],
                            },
                            run.offset,
                            bytes,
                            held,
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
                    let at = *space as usize;
                    if spaces.len() <= at {
                        spaces.resize(at + 1, None);
                    }
                    match spaces[at] {
                        None => spaces[at] = Some(paging),
                        Some(held) if held == paging => {}
                        Some(held) => {
                            return Err(disagree(format!(
                                "pages kv space {space}'s rows apart ({held:?} and {paging:?}), \
                                 where one table pages a space"
                            )));
                        }
                    }
                    let keys = offset;
                    let values = if kv.own_values {
                        keys + u64::from(run_tokens) * kv.keys_cell
                    } else {
                        keys
                    };
                    kv_rows.push(u32::try_from(index).unwrap_or(u32::MAX));
                    rows.push(Row::Kv {
                        space: kv.space,
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
            Some(layout) => Some(rs_rows(recipe, fixed_runs, layout)?),
        };
        let mut side = 0usize;
        let mut side_bytes = 0u64;
        for (space, width) in compressed_spaces(trace) {
            side_bytes += 2 * u64::from(tokens) * width * u64::from(elem_size(Dtype::Bf16));
            rows.push(Row::Compressed {
                space,
                width,
                kv: side,
                score: side + 1,
            });
            side += 2;
        }
        Ok(Rows {
            rows,
            chunk_bytes: recipe.chunk_bytes,
            list_bytes,
            tokens,
            fixed_bytes,
            pieces,
            ring_bytes,
            ring_cap,
            ring_window,
            spaces: spaces
                .into_iter()
                .map(|paging| paging.unwrap_or(SpacePaging::List))
                .collect(),
            side_bytes,
            bands,
            rs,
            kv_rows,
            state_rows,
        })
    }
}

/// The rs planes' places in a record piece, read off the recipe's
/// `RsLayout`: each rs row's planes one after another past its offset, each
/// plane `tokens_per_piece` rows; the shell's planes matched by the values
/// they buffer.
fn rs_rows(
    recipe: &ChunkRecipe,
    runs: &[poem_compiler::chunk::FixedRun],
    layout: &crate::rs::Layout,
) -> Result<RsRows> {
    let rs = &recipe.rs;
    let rows16 = poem_compiler::chunk::UNFOLDED_ROWS;
    let tokens = u32::try_from(rs.tokens_per_piece).unwrap_or(u32::MAX);
    // The rs buffer becomes a list shape of its own (CONTRACT-BC 18); until
    // then this shell reads only whole rs rows inside the record.
    if rs.pieces != 1 || tokens != rows16 {
        return Err(disagree(format!(
            "cuts the rs rows into {} pieces of {} tokens, where this shell reads them whole \
             until the rs buffer is a list shape of its own",
            rs.pieces, rs.tokens_per_piece
        )));
    }
    let mut plane_at: Vec<Option<u64>> = vec![None; layout.planes.len()];
    for (id, row) in &rs.rows {
        let run = runs
            .iter()
            .find(|run| run.row == *id)
            .ok_or_else(|| disagree(format!("lays no record slot for rs row {id:?}")))?;
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
            let offset = run.offset + row.plane_at(at, rs.tokens_per_piece);
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
        pieces: rs.pieces,
        tokens_per_piece: tokens,
        slots: rs.slots.clone(),
        plane_at,
    })
}

#[derive(Debug)]
pub struct Pools {
    paging: Paging,
    rows: Vec<Row>,
    pool: Pool,
    layout: Layout,
    /// The compressor side planes, a cell a token of every physical list
    /// slot.
    side: Vec<Buffer>,
    /// Every state row's segment table, one after another.
    maps: Option<Buffer>,
    shape_rows: [Vec<u32>; 2],
    /// Each distinct set of record slots a state row's pieces lie in.
    bands: Vec<Vec<u32>>,
    rs: Option<RsRows>,
    /// How each kv space is paged.
    spaces: Vec<SpacePaging>,
    disk: Option<Spill>,
    parked: Option<Spill>,
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
        let cells = list_slots * u64::from(paging.page_size);
        let mut side = Vec::new();
        for row in &rows.rows {
            if let Row::Compressed { width, .. } = *row {
                let bytes = cells * width * u64::from(elem_size(Dtype::Bf16));
                side.push(Buffer::zeroed(device, bytes.max(4))?);
                side.push(Buffer::zeroed(device, bytes.max(4))?);
            }
        }
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
            spaces: rows.spaces,
            rows: rows.rows,
            pool,
            layout,
            side,
            disk: None,
            parked: None,
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

    /// The unit (`Layout::buffer_stride`) the rs rows of the record whose
    /// physical pieces are `record` start at.
    pub fn rs_unit(&self, record: &[u32]) -> Result<u32> {
        let rs = self.rs.as_ref().ok_or_else(|| Fault::Unbound {
            what: "the rs buffer rows of a load that buffers none".to_string(),
        })?;
        let fixed = *self.shape(FIXED)?;
        let piece = *record.get(rs.slots[0] as usize).ok_or(Fault::Ceiling {
            what: "pieces of a record",
            need: u64::from(rs.slots[0]) + 1,
            have: record.len() as u64,
        })?;
        let at = slot_at(&self.layout, &fixed, piece);
        Ok(u32::try_from(at / self.layout.buffer_stride()).unwrap_or(u32::MAX))
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
        self.layout.facts(
            self.pool.growth(),
            self.pool.committed_segments(),
            &self.shape_rows,
        )
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

    /// How kv space `space` is paged.
    #[must_use]
    pub fn space_paging(&self, space: usize) -> SpacePaging {
        self.spaces.get(space).copied().unwrap_or(SpacePaging::List)
    }

    /// What a ring page id counts: `gcd(chunk, ring slot)` bytes.
    fn ring_stride(&self) -> u64 {
        self.layout
            .ring
            .map_or(256, |ring| gcd(self.layout.chunk_bytes, ring.slot_bytes))
    }

    /// A lane's ring pages as kernels address them, one a kv page in token
    /// order: `refs` (ring slot refs, 0 the null chunk) as units of
    /// [`Pools::ring_stride`], or, seat-addressed, seat `seat`'s ring slot
    /// `page % cap` for each of `pages` pages.
    pub fn ring_pages(&self, refs: &[u32], seat: Option<(u32, u32)>) -> Result<Vec<u32>> {
        let ring = *self.shape(RING)?;
        let stride = self.ring_stride();
        let unit = |slot: u32| -> Result<u32> {
            if slot == 0 {
                return Ok(0);
            }
            let at = self.slot_offset(RING, slot)?;
            self.pool
                .admit(u32::try_from(at / self.layout.chunk_bytes).unwrap_or(u32::MAX))?;
            Ok(u32::try_from(at / stride).unwrap_or(u32::MAX))
        };
        if let Some((slot, pages)) = seat
            && refs.is_empty()
        {
            let cap = self.layout.ring_cap.max(1);
            return (0..pages)
                .map(|page| unit(ring.seat_ref(slot * cap + page % cap)?))
                .collect();
        }
        refs.iter().map(|&r| unit(r)).collect()
    }

    /// The unit (`Layout::buffer_stride`) a folded ring in record slot
    /// `slot` starts at, for the record whose physical pieces are `record`.
    pub fn folded_unit(&self, record: &[u32], slot: u32) -> Result<u32> {
        let fixed = *self.shape(FIXED)?;
        let piece = *record.get(slot as usize).ok_or(Fault::Ceiling {
            what: "pieces of a record",
            need: u64::from(slot) + 1,
            have: record.len() as u64,
        })?;
        let at = slot_at(&self.layout, &fixed, piece);
        Ok(u32::try_from(at / self.layout.buffer_stride()).unwrap_or(u32::MAX))
    }

    /// Copies whole ring slots (slot refs) `(src, dst)`: a forked ring.
    pub fn copy_ring(&mut self, frame: &mut Frame, moves: &[(u32, u32)]) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }
        self.copy_whole(frame, RING, moves)
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

    /// The pool's whole range and the side planes: what it takes with every
    /// segment committed.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.pool.bytes() + self.side.iter().map(Buffer::bytes).sum::<u64>()
    }

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.pool.committed_bytes() + self.side.iter().map(Buffer::bytes).sum::<u64>()
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.pool.high_water_bytes() + self.side.iter().map(Buffer::bytes).sum::<u64>()
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
        for row in &self.rows {
            pools.push(match *row {
                Row::Compressed {
                    space,
                    width,
                    kv,
                    score,
                } => {
                    let plane = |at: usize| -> Result<Tensor> {
                        let buffer = &self.side[at];
                        Ok(Tensor::new(
                            handles.bind(buffer, 0, buffer.bytes())?,
                            cells,
                            u32::try_from(width).unwrap_or(u32::MAX),
                            Dtype::Bf16,
                        ))
                    };
                    compressed.push((
                        space,
                        PoolSlabs {
                            state_kv: plane(kv)?,
                            state_score: plane(score)?,
                        },
                    ));
                    continue;
                }
                Row::Kv {
                    space,
                    head_dim,
                    kv_heads,
                    dtype,
                    keys,
                    values,
                    values_width,
                    ..
                } => {
                    let seat: &SpaceSeat =
                        seats
                            .spaces
                            .get(space as usize)
                            .ok_or_else(|| Fault::Unbound {
                                what: format!(
                                    "cache space {space}, for which this fire wrote no geometry"
                                ),
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
                    let (stride, page_size) = match self.space_paging(space as usize) {
                        SpacePaging::List => (list_bytes, self.paging.page_size),
                        SpacePaging::Ring => (self.ring_stride(), self.paging.page_size),
                        SpacePaging::Folded { tokens, .. } => (self.layout.buffer_stride(), tokens),
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

    /// Every kv run's place in a list slot: its offset and the bytes a token
    /// takes there, side planes apart.
    fn kv_runs(&self) -> Vec<(u64, u64)> {
        let mut runs = Vec::new();
        for row in &self.rows {
            if let Row::Kv {
                keys,
                values,
                keys_cell,
                values_cell,
                ..
            } = *row
            {
                runs.push((keys, keys_cell));
                if values != keys {
                    runs.push((values, values_cell));
                }
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
        let sides: Vec<u64> = self
            .rows
            .iter()
            .filter_map(|row| match *row {
                Row::Compressed { width, .. } => Some(width * u64::from(elem_size(Dtype::Bf16))),
                _ => None,
            })
            .collect();
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
            for (at, &cell) in sides.iter().enumerate() {
                for plane in [&self.side[2 * at], &self.side[2 * at + 1]] {
                    let cell_at = |page: u32, token: u32| {
                        (u64::from(page) * page_size + u64::from(token)) * cell
                    };
                    let (from, to) = (cell_at(src, span.src_token), cell_at(dst, span.dst_token));
                    if from != to {
                        frame.copy_span(plane, from, plane, to, u64::from(span.tokens) * cell)?;
                    }
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
                    "a slot copy from {src:?} to {dst:?}: this pool is host memory, and moves \
                     only within itself and to its slot files"
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
    /// and takes the moves into the chunk table.
    pub fn evacuate(&mut self, frame: &mut Frame, evacuation: &Evacuation) -> Result<()> {
        // The compressor side planes are indexed by physical page: each
        // moved chunk's pages move with it.
        let mut sides = Vec::new();
        if let Some(list) = self.layout.list
            && !self.side.is_empty()
        {
            let cells = u64::from(list.per_chunk) * u64::from(self.paging.page_size);
            for &(logical, physical) in &evacuation.moves {
                sides.push((self.pool.physical(logical)?, physical, cells));
            }
        }
        self.pool.evacuate(frame, evacuation)?;
        for (from, to, cells) in sides {
            for row in &self.rows {
                let Row::Compressed {
                    width, kv: plane, ..
                } = *row
                else {
                    continue;
                };
                let cell = width * u64::from(elem_size(Dtype::Bf16));
                for buffer in [&self.side[plane], &self.side[plane + 1]] {
                    frame.copy_span(
                        buffer,
                        u64::from(from) * cells * cell,
                        buffer,
                        u64::from(to) * cells * cell,
                        cells * cell,
                    )?;
                }
            }
        }
        Ok(())
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
