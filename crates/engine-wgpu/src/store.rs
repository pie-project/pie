//! The load's held state in one chunk pool ([`pool::Pool`]): every cache
//! row placed in the slot shapes of [`layout`], the slot refs the runtime
//! hands out translated to the physical slots kernels address, and the
//! copies, spills and evacuations that move slots.
pub mod experts;
pub mod kv;
pub mod layout;
pub mod pool;
pub mod rows;

use engine::chunk::{ChunkFacts, Evacuation, ShapeId, SlotCopy};
use engine::transfer::{KvRun, MemoryDomain};
use kernels_slang::{KvPool, Segments, Tensor};
use poem_compiler::chunk::{ChunkRecipe, RowId, SlotLayout};
use poem_ir::{CacheRow, Dtype, Trace};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles, Memory};
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable};
use crate::store::kv::{Facts, Paging};
use crate::store::layout::{FIXED, LIST, Layout, RS, Slots, WINDOWS, gcd, window_shape};
use crate::store::pool::Pool;

/// A layer's recurrent cache as kernels bind it: `state` carries the row's
/// geometry, `slots` each row's record unit, and the row lies in the pool's
/// segments at word `origin + unit * slot_stride`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecurrentPool {
    pub state: Tensor,

    pub slots: Tensor,

    pub pool: Segments,

    pub origin: u32,

    pub slot_stride: u32,

    /// The units a `slots` entry holds: a record's pieces.
    pub pieces: u32,

    /// The entry's slot this row's piece 0 lies in.
    pub piece_first: u32,

    /// The atoms one piece of this row holds (`u32::MAX`: whole).
    pub piece_atoms: u32,
}

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

/// The list a kv row's pages lie in: the kv list, or window list `k`
/// (CONTRACT-BC 21).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Paged {
    List,
    Window(usize),
}

impl Paged {
    /// The window list this is, if one.
    #[must_use]
    pub fn window(self) -> Option<usize> {
        match self {
            Paged::List => None,
            Paged::Window(k) => Some(k),
        }
    }

    /// The slot shape of this list.
    #[must_use]
    pub fn shape(self) -> ShapeId {
        match self {
            Paged::List => LIST,
            Paged::Window(k) => window_shape(k),
        }
    }
}

/// A window list as a load lays it: its slot's bytes, the rows its window
/// retains, the tokens (`T_w`) a slot holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowRows {
    pub bytes: u64,
    pub window: u32,
    pub tokens: u32,
}

/// One cache row's place in a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Row {
    /// A kv row: its key run and value run in a slot of its list (one run
    /// when the trace shares one plane), the list's tokens of `*_cell`
    /// bytes each.
    Kv {
        paged: Paged,
        head_dim: u32,
        kv_heads: u32,
        dtype: Dtype,
        keys: u64,
        values: u64,
        values_width: u64,
        keys_cell: u64,
        values_cell: u64,
    },
    /// A state row: `stride` elements at `offset` in the record's slots
    /// `first ..`: whole in slot `first`, or cut `piece_atoms` atoms
    /// (`piece_bytes` bytes) a piece, piece `p` in slot `first + p`.
    State {
        stride: u64,
        dtype: Dtype,
        offset: u64,
        first: u32,
        piece_atoms: u32,
        piece_bytes: u64,
    },
}

/// The rs list's slot as this shell reads it: `tokens` rows of each rs
/// plane, each of the shell's planes at `plane_at` in the slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsRows {
    pub tokens: u32,
    pub bytes: u64,
    pub plane_at: Vec<u64>,
}

/// How a load's rows lay into the recipe's slots, before any pool.
#[derive(Debug, Clone)]
pub struct Rows {
    rows: Vec<Row>,
    pub chunk_bytes: u64,
    pub list_bytes: u64,
    /// The list slot's tokens: the kv page size.
    pub tokens: u32,
    pub fixed_bytes: u64,
    pub pieces: u32,
    /// The window lists, narrowest window first.
    pub windows: Vec<WindowRows>,
    /// The page tables a fire stages, by the list each pages: the kv
    /// list's first (the All axis reads it), then each window list's.
    pub tables: Vec<Paged>,
    pub rs: Option<RsRows>,
    pub kv_rows: Vec<u32>,
    /// The kv rows each window list lays out, in order.
    pub window_rows: Vec<Vec<u32>>,
    pub state_rows: Vec<u32>,
}

fn disagree(what: String) -> Fault {
    Fault::Unbound {
        what: format!("the wgpu chunk recipe, which {what}"),
    }
}

impl Rows {
    /// Lays `trace`'s rows where `recipe` (the compiler's wgpu chunk
    /// recipe, the one layout source) puts them; `facts` restates kv rows'
    /// heads, `rs` is the shell's rs buffer. Refuses where the recipe lays a
    /// row otherwise than this shell stores it.
    pub fn plan(
        trace: &Trace,
        recipe: &ChunkRecipe,
        facts: &Facts,
        rs: Option<&crate::rs::Layout>,
    ) -> Result<Rows> {
        let mut windows = Vec::new();
        let mut window_runs = Vec::new();
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
                    window: *window,
                    tokens: *tokens,
                });
                window_runs.push(rows.as_slice());
            }
        }
        if windows.len() > WINDOWS {
            return Err(disagree(format!(
                "keeps {} window lists ({}), where a load holds at most {WINDOWS}",
                windows.len(),
                recipe.shapes()
            )));
        }
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
        // Without a kv list the narrowest window list's slot is the kv page.
        let tokens = if list_bytes == 0 {
            windows.first().map_or(0, |w| w.tokens)
        } else {
            tokens
        };
        if let Some(window) = windows
            .iter()
            .find(|w| w.tokens == 0 || w.tokens > tokens || tokens % w.tokens != 0)
        {
            return Err(disagree(format!(
                "pages windowed rows {} tokens a window slot beside {tokens}-token kv pages, \
                 where a window list's slot divides a kv page",
                window.tokens
            )));
        }
        let mut tables: Vec<Paged> = Vec::new();
        let mut rows = Vec::with_capacity(trace.caches.len());
        let mut kv_rows = Vec::new();
        let mut window_rows = vec![Vec::new(); windows.len()];
        let mut state_rows = Vec::new();
        for (index, row) in trace.caches.iter().enumerate() {
            let id = RowId::Cache(u32::try_from(index).unwrap_or(u32::MAX));
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
                    if let Some(seat) = restated
                        && u64::from(seat.kv_heads) * u64::from(seat.head_dim) != width
                    {
                        return Err(Fault::Unbound {
                            what: format!(
                                "cache `{name}`, whose row is {width} wide while its \
                                 consumers state {} heads of {}",
                                seat.kv_heads, seat.head_dim
                            ),
                        });
                    }
                    let head_dim = restated.map_or(width, |seat| u64::from(seat.head_dim));
                    let kv_heads = restated.map_or(1, |seat| u64::from(seat.kv_heads));
                    let element = elem_bytes(name, *dtype)?;
                    let own_values = !planes.shared && planes.values != 0;
                    let keys_cell = width * element;
                    let values_cell = planes.values * element;
                    let per_token = keys_cell + if own_values { values_cell } else { 0 };
                    // Where the row lies: a kv list slot or a window list's,
                    // and the tokens a run there holds.
                    let (paged, run, run_tokens) =
                        if let Some(run) = runs.iter().find(|run| run.row == id) {
                            (Paged::List, run, tokens)
                        } else if let Some((k, run)) =
                            window_runs.iter().enumerate().find_map(|(k, runs)| {
                                runs.iter().find(|run| run.row == id).map(|run| (k, run))
                            })
                        {
                            (Paged::Window(k), run, windows[k].tokens)
                        } else {
                            return Err(disagree(format!("lays kv row `{name}` in no slot")));
                        };
                    if run.per_token != per_token {
                        return Err(disagree(format!(
                            "lays kv row `{name}` at {} bytes a token, where this shell \
                             stores {per_token}",
                            run.per_token
                        )));
                    }
                    let keys = run.offset;
                    let values = if own_values {
                        keys + u64::from(run_tokens) * keys_cell
                    } else {
                        keys
                    };
                    if keys % 4 != 0 || values % 4 != 0 {
                        return Err(disagree(format!(
                            "lays kv row `{name}` at byte {keys}, inside a word"
                        )));
                    }
                    if !tables.contains(&paged) {
                        tables.push(paged);
                    }
                    let at = u32::try_from(index).unwrap_or(u32::MAX);
                    match paged {
                        Paged::List => kv_rows.push(at),
                        Paged::Window(k) => window_rows[k].push(at),
                    }
                    rows.push(Row::Kv {
                        paged,
                        head_dim: u32::try_from(head_dim).unwrap_or(u32::MAX),
                        kv_heads: u32::try_from(kv_heads).unwrap_or(u32::MAX),
                        dtype: *dtype,
                        keys,
                        values,
                        values_width: planes.values,
                        keys_cell,
                        values_cell: if planes.values == 0 {
                            keys_cell
                        } else {
                            values_cell
                        },
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
                            "sizes state row `{name}` at {} bytes, where this shell stores \
                             {bytes} ({dtype:?})",
                            atom.bytes * atom.extent
                        )));
                    }
                    if run.offset % 4 != 0 {
                        return Err(disagree(format!(
                            "lays state row `{name}` at byte {}, inside a word",
                            run.offset
                        )));
                    }
                    state_rows.push(u32::try_from(index).unwrap_or(u32::MAX));
                    let first = run.slots.first().copied().unwrap_or(0);
                    if run
                        .slots
                        .iter()
                        .zip(first..)
                        .any(|(&slot, want)| slot != want)
                    {
                        return Err(disagree(format!(
                            "lays the pieces of state row `{name}` in record slots {:?}, \
                             where this shell reads them in consecutive slots",
                            run.slots
                        )));
                    }
                    let cut = run.pieces > 1;
                    rows.push(Row::State {
                        stride,
                        dtype,
                        offset: run.offset,
                        first,
                        piece_atoms: if cut {
                            u32::try_from(run.atoms_per_piece).unwrap_or(u32::MAX)
                        } else {
                            u32::MAX
                        },
                        piece_bytes: if cut {
                            run.atoms_per_piece * atom.bytes
                        } else {
                            bytes
                        },
                    });
                }
            }
        }
        let rs = match rs {
            None => None,
            Some(layout) => Some(rs_rows(recipe, layout)?),
        };
        Ok(Rows {
            rows,
            chunk_bytes: recipe.chunk_bytes,
            list_bytes,
            tokens,
            fixed_bytes,
            pieces: pieces.max(1),
            windows,
            // The kv list's table first: the All axis reads it (or the
            // narrowest window list's, where no row is in the kv list).
            tables: {
                tables.sort_by_key(|paged| match paged {
                    Paged::List => 0,
                    Paged::Window(k) => 1 + *k,
                });
                tables
            },
            rs,
            kv_rows,
            window_rows,
            state_rows,
        })
    }
}

/// The rs planes' places in an rs-list slot, read off the recipe's
/// `RsLayout`: the shell's planes matched by the values they buffer.
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
    /// The cache rows each shape lays out: the kv list's, the record's,
    /// then each window list's.
    shape_rows: Vec<Vec<u32>>,
    /// The page tables a fire stages, by the list each pages.
    tables: Vec<Paged>,
    rs: Option<RsRows>,
    /// Host chunks laid out like the device's, one id space beside them:
    /// host memory, the device side read and written through the queue.
    host: Vec<u8>,
    host_chunks: u32,
    disk: Option<Spill>,
    watermark: engine::frame::Demand,
}

impl Pools {
    /// The pool `layout` sizes over `rows`, kv pages of `paging`'s size,
    /// and `host_chunks` host chunks beside it; nothing committed yet.
    pub fn reserve(
        device: &Context,
        rows: Rows,
        layout: Layout,
        paging: Paging,
        host_chunks: u32,
    ) -> Result<Pools> {
        let pool = Pool::reserve(
            layout.chunk_bytes,
            layout.total_chunks,
            layout.segment_chunks,
        )?;
        let list_slots = layout.list.map_or(0, |list| {
            u64::from(list.per_chunk) * u64::from(layout.total_chunks)
        });
        let _ = device;
        let host =
            vec![0u8; usize::try_from(u64::from(host_chunks) * layout.chunk_bytes).unwrap_or(0)];
        Ok(Pools {
            paging: Paging {
                pages: list_slots.max(1),
                ..paging
            },
            shape_rows: [rows.kv_rows.clone(), rows.state_rows.clone()]
                .into_iter()
                .chain(rows.window_rows.iter().cloned())
                .collect(),
            tables: rows.tables.clone(),
            rs: rows.rs,
            rows: rows.rows,
            pool,
            layout,
            host,
            host_chunks,
            disk: None,
            watermark: engine::frame::Demand::ZERO,
        })
    }

    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The pool as the runtime counts it.
    #[must_use]
    pub fn chunk_facts(&self) -> ChunkFacts {
        self.layout.facts(
            self.pool.growth(),
            self.pool.committed_segments(),
            &self.shape_rows,
            [self.host_chunks, self.disk_pages()],
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

    /// The virtual byte slot ref `slot` of shape `id` starts at, its chunk
    /// committed.
    fn slot_offset(&self, id: ShapeId, slot: u32) -> Result<u64> {
        let shape = *self.shape(id)?;
        let (chunk, index) = self.place(&shape, slot)?;
        self.pool.admit(chunk)?;
        Ok(u64::from(chunk) * self.layout.chunk_bytes + u64::from(index) * shape.slot_bytes)
    }

    /// The virtual byte weight slot ref `slot` (of a shape `per_chunk`
    /// slots of `slot_bytes` a chunk) starts at, its chunk committed.
    pub fn weight_slot(&self, per_chunk: u32, slot_bytes: u64, slot: u32) -> Result<u64> {
        let per = per_chunk.max(1);
        let (chunk, index) = (slot / per, slot % per);
        if chunk == 0 || chunk >= self.layout.total_chunks {
            return Err(Fault::Ceiling {
                what: "chunks a weight slot ref may name (past the null chunk)",
                need: u64::from(chunk) + 1,
                have: u64::from(self.layout.total_chunks),
            });
        }
        let chunk = self.pool.physical(chunk)?;
        self.pool.admit(chunk)?;
        Ok(u64::from(chunk) * self.layout.chunk_bytes + u64::from(index) * slot_bytes)
    }

    /// The segment buffer virtual byte `at` lies in, and its offset there.
    pub fn locate(&self, at: u64) -> Result<(&Buffer, u64)> {
        self.pool.locate(at)
    }

    /// What one id of shape `id` counts as kernels address it: the slot
    /// where a chunk holds whole slots, else `gcd(chunk, slot)` bytes.
    fn unit(&self, id: ShapeId) -> Result<u64> {
        Ok(gcd(self.layout.chunk_bytes, self.shape(id)?.slot_bytes))
    }

    /// Slot ref `slot` of shape `id` in units of [`Pools::unit`].
    fn units(&self, id: ShapeId, slot: u32) -> Result<u32> {
        let at = self.slot_offset(id, slot)?;
        u32::try_from(at / self.unit(id)?).map_err(|_| Fault::Ceiling {
            what: "slot units a kernel indexes",
            need: at,
            have: u64::from(u32::MAX),
        })
    }

    /// A lane's kv pages as kernels address them: `refs` (slot refs of the
    /// list shape) as physical pages, or, seat-addressed, seat `seat`'s
    /// `pages` pages.
    pub fn kv_pages(&self, refs: &[u32], seat: Option<(u32, u32)>) -> Result<Vec<u32>> {
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
                    self.units(LIST, list.seat_ref(id)?)
                })
                .collect();
        }
        refs.iter().map(|&r| self.units(LIST, r)).collect()
    }

    /// The physical page explicit write descriptor page `page` names.
    pub fn kv_page(&self, page: u32) -> Result<u32> {
        self.units(LIST, page)
    }

    /// The page tables a fire stages, by the list each pages: the kv
    /// list's first, the All axis's.
    #[must_use]
    pub fn tables(&self) -> &[Paged] {
        &self.tables
    }

    /// The page table the rows paged in `paged`'s list read.
    #[must_use]
    pub fn table_of(&self, paged: Paged) -> usize {
        self.tables
            .iter()
            .position(|&held| held == paged)
            .unwrap_or(0)
    }

    /// The page table each window list's write descriptors read, by its
    /// window; a window the recipe kept whole reads the kv list's (table 0).
    #[must_use]
    pub fn window_tables(&self) -> Vec<(u32, usize)> {
        (0..WINDOWS)
            .filter_map(|k| {
                let list = self.layout.window(k)?;
                Some((list.window, self.table_of(Paged::Window(k))))
            })
            .collect()
    }

    /// The paging of `paged`'s list: the kv page, or window list `k`'s `T_w`
    /// tokens a page (CONTRACT-BC 21), as many pages a seat as its context
    /// fills.
    #[must_use]
    pub fn paging_of(&self, paged: Paged) -> Paging {
        match paged.window().and_then(|k| self.layout.window(k)) {
            Some(list) => {
                let tokens = list.tokens.max(1);
                Paging {
                    page_size: tokens,
                    pages_per_slot: self.paging.context().div_ceil(tokens).max(1),
                    ..self.paging
                }
            }
            None => self.paging,
        }
    }

    /// A lane's pages of `paged`'s list as kernels address them, in token
    /// order: `refs` (its slot refs; in a window list 0 is a slot trimmed
    /// behind the window, the null chunk) as units of [`Pools::unit`], or,
    /// seat-addressed, seat `seat`'s run, `pages` of them.
    pub fn list_pages(
        &self,
        paged: Paged,
        refs: &[u32],
        seat: Option<(u32, u32)>,
    ) -> Result<Vec<u32>> {
        let Paged::Window(k) = paged else {
            return self.kv_pages(refs, seat);
        };
        let id = window_shape(k);
        let window = *self.shape(id)?;
        let unit = |slot: u32| -> Result<u32> {
            if slot == 0 {
                return Ok(0);
            }
            self.units(id, slot)
        };
        if let Some((slot, pages)) = seat
            && refs.is_empty()
        {
            let first = u64::from(slot) * u64::from(self.paging_of(paged).pages_per_slot);
            return (0..pages)
                .map(|page| {
                    let at = u32::try_from(first + u64::from(page)).unwrap_or(u32::MAX);
                    unit(window.seat_ref(at)?)
                })
                .collect();
        }
        refs.iter().map(|&r| unit(r)).collect()
    }

    /// The slots a record takes.
    #[must_use]
    pub fn pieces(&self) -> u32 {
        self.layout.pieces.max(1)
    }

    /// A lane's record as kernels address it, one unit a piece: from `refs`
    /// (slot refs of the fixed shape, a piece each) or seat `seat`'s record.
    pub fn record(&self, refs: Option<&[u32]>, seat: u32) -> Result<Vec<u32>> {
        let pieces = self.pieces();
        if !self.has_state() {
            return Ok(vec![seat; pieces as usize]);
        }
        match refs {
            Some(refs) if refs.len() == pieces as usize => {
                refs.iter().map(|&r| self.units(FIXED, r)).collect()
            }
            Some(refs) => Err(Fault::Unbound {
                what: format!(
                    "a record named by {} slot refs, where this pool's records are {pieces} \
                     pieces",
                    refs.len()
                ),
            }),
            None => {
                let fixed = *self.shape(FIXED)?;
                (0..pieces)
                    .map(|p| self.units(FIXED, fixed.seat_ref(seat * pieces + p)?))
                    .collect()
            }
        }
    }

    /// The rs list's place in a slot, where the load buffers one.
    #[must_use]
    pub fn rs_rows(&self) -> Option<&RsRows> {
        self.rs.as_ref()
    }

    /// A lane's rs-list pages as kernels address them, in token order from
    /// the verb's base: `refs` (rs-list slot refs) as units of
    /// [`Pools::rs_stride`], or, seat-addressed, seat `seat`'s run.
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
                self.units(RS, slot)
            })
            .collect()
    }

    /// The bytes an rs-list page id counts.
    pub fn rs_stride(&self) -> u64 {
        self.unit(RS).unwrap_or(256)
    }

    /// The pool as kernels bind it.
    pub fn segments(&self, handles: &Handles) -> Result<Segments> {
        self.pool.bind(handles)
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.rows
            .iter()
            .map(|row| match row {
                Row::State { stride, dtype, .. } => stride * u64::from(elem_size(*dtype)),
                Row::Kv { .. } => 0,
            })
            .sum()
    }

    /// The state rows of the record at `units` (a piece each), one after
    /// another, each row's pieces in order. The caller has drained every
    /// frame that writes it.
    pub fn read_record(&self, units: &[u32]) -> Result<Vec<u8>> {
        let unit = self.unit(FIXED)?;
        let mut out = Vec::new();
        for row in &self.rows {
            let Row::State {
                stride,
                dtype,
                offset,
                first,
                piece_bytes,
                ..
            } = *row
            else {
                continue;
            };
            let mut left = stride * u64::from(elem_size(dtype));
            let mut piece = first as usize;
            while left > 0 {
                let id = *units.get(piece).ok_or(Fault::Ceiling {
                    what: "pieces of a record",
                    need: piece as u64 + 1,
                    have: units.len() as u64,
                })?;
                let take = piece_bytes.min(left);
                let mut span = vec![0u8; usize::try_from(take).unwrap_or(0)];
                self.pool.read(u64::from(id) * unit + offset, &mut span)?;
                out.extend_from_slice(&span);
                left -= take;
                piece += 1;
            }
        }
        Ok(out)
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        self.rows.iter().any(|row| matches!(row, Row::State { .. }))
    }

    #[must_use]
    pub fn watermark(&self) -> engine::frame::Demand {
        self.watermark
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// The pool's whole range.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.pool.bytes()
    }

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.pool.committed_bytes()
    }

    /// The cache table one fire binds.
    pub fn table(&self, handles: &Handles, seats: &Seats) -> Result<CacheTable> {
        let segments = self.pool.bind(handles)?;
        let cells = u32::try_from(self.paging.pages() * u64::from(self.paging.page_size))
            .unwrap_or(u32::MAX);
        let fixed_unit = self.unit(FIXED).unwrap_or(4);
        let mut rows = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            rows.push(match *row {
                Row::Kv {
                    paged,
                    head_dim,
                    kv_heads,
                    dtype,
                    keys,
                    values,
                    values_width,
                    ..
                } => {
                    let table = self.table_of(paged);
                    let seat = seats.page_tables.get(table).ok_or_else(|| Fault::Unbound {
                        what: format!("page table {table}, which this fire staged none of"),
                    })?;
                    // A page id counts the list's slot where a chunk holds
                    // whole ones, else `gcd(chunk, slot)` bytes.
                    let words = self.unit(paged.shape())? / 4;
                    let plane = |width: u64| {
                        Tensor::new(
                            kernels_slang::ABSENT,
                            cells,
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        )
                    };
                    let stride = u32::try_from(words).unwrap_or(u32::MAX);
                    CachePool::Kv(KvPool {
                        keys: plane(u64::from(kv_heads) * u64::from(head_dim)),
                        values: plane(values_width),
                        page_indices: seat.page_indices,
                        page_indptr: seat.page_indptr,
                        page_size: narrow(u64::from(self.paging_of(paged).page_size)),
                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                        pool: segments,
                        page_stride: [stride, stride],
                        origin: [
                            u32::try_from(keys / 4).unwrap_or(u32::MAX),
                            u32::try_from(values / 4).unwrap_or(u32::MAX),
                        ],
                    })
                }
                Row::State {
                    stride,
                    dtype,
                    offset,
                    first,
                    piece_atoms,
                    ..
                } => CachePool::Recurrent(RecurrentPool {
                    state: Tensor::new(
                        kernels_slang::ABSENT,
                        self.paging.slots,
                        u32::try_from(stride).unwrap_or(u32::MAX),
                        dtype,
                    ),
                    slots: seats.slot_of_row,
                    pool: segments,
                    origin: u32::try_from(offset / 4).unwrap_or(u32::MAX),
                    slot_stride: u32::try_from(fixed_unit / 4).unwrap_or(u32::MAX),
                    pieces: self.pieces(),
                    piece_first: first,
                    piece_atoms,
                }),
            });
        }
        Ok(CacheTable(rows))
    }

    /// Zeroes the record at `units` (a piece each): a fresh lane's state.
    /// The caller has drained every frame.
    pub fn clear(&mut self, units: &[u32]) -> Result<()> {
        if !self.has_state() {
            return Ok(());
        }
        let bytes = self.shape(FIXED)?.slot_bytes;
        let unit = self.unit(FIXED)?;
        for &id in units {
            self.pool.zero(u64::from(id) * unit, bytes)?;
        }
        Ok(())
    }

    /// Every kv list run's place in a list slot: its offset and the bytes a
    /// token takes there.
    fn kv_runs(&self) -> Vec<(u64, u64)> {
        let mut runs = Vec::new();
        for row in &self.rows {
            if let Row::Kv {
                paged: Paged::List,
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
    pub fn copy_kv(&self, frame: &mut Frame, moves: &[KvRun]) -> Result<()> {
        let page_size = u64::from(self.paging.page_size);
        let runs = self.kv_runs();
        for span in moves {
            for token in [span.src_token, span.dst_token] {
                let end = u64::from(token) + u64::from(span.tokens);
                if end > page_size {
                    return Err(Fault::Ceiling {
                        what: "token slots in one kv page",
                        need: end,
                        have: page_size,
                    });
                }
            }
            if span.tokens == 0 {
                continue;
            }
            let src = self.slot_offset(LIST, span.src_page)?;
            let dst = self.slot_offset(LIST, span.dst_page)?;
            for &(offset, cell) in &runs {
                let from = src + offset + u64::from(span.src_token) * cell;
                let to = dst + offset + u64::from(span.dst_token) * cell;
                if from != to {
                    self.pool
                        .copy(frame, from, to, u64::from(span.tokens) * cell)?;
                }
            }
        }
        Ok(())
    }

    /// A kv page's bytes, as the slot file holds one: a list slot.
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        self.layout.list.map_or(0, |list| list.slot_bytes)
    }

    /// Copies whole slots of `id` (slot refs) `(src, dst)` on the device.
    pub fn copy_shape(&self, frame: &mut Frame, id: ShapeId, moves: &[(u32, u32)]) -> Result<()> {
        let bytes = self.shape(id)?.slot_bytes;
        for &(src, dst) in moves {
            let from = self.slot_offset(id, src)?;
            let to = self.slot_offset(id, dst)?;
            if from != to {
                self.pool.copy(frame, from, to, bytes)?;
            }
        }
        Ok(())
    }

    /// The host byte host slot ref `slot` of shape `id` starts at: the host
    /// chunks are laid out like the device's.
    fn host_offset(&self, id: ShapeId, slot: u32) -> Result<u64> {
        let shape = *self.shape(id)?;
        let (chunk, index) = shape.split(slot);
        if chunk >= self.host_chunks {
            return Err(Fault::Ceiling {
                what: "host chunks of the state pool",
                need: u64::from(chunk) + 1,
                have: u64::from(self.host_chunks),
            });
        }
        Ok(u64::from(chunk) * self.layout.chunk_bytes + u64::from(index) * shape.slot_bytes)
    }

    /// Slots of one shape copied as [`engine::Engine::copy_slots`] states
    /// them: on the device (whole slots, and token runs of the list),
    /// between the device and the host chunks, or between the device and
    /// the slot file. A host or disk copy waits out its frame.
    pub fn copy_slots(
        &mut self,
        device: &Context,
        frame: &mut Frame,
        copy: &SlotCopy,
    ) -> Result<()> {
        let here = |domain: MemoryDomain| matches!(domain, MemoryDomain::WgpuDevice(_));
        match (copy.src, copy.dst) {
            (src, dst) if here(src) && here(dst) => {
                self.copy_shape(frame, copy.shape, &copy.slots)?;
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
            (src, MemoryDomain::HostPinned) if here(src) => {
                let bytes = usize::try_from(self.shape(copy.shape)?.slot_bytes).unwrap_or(0);
                for &(slot, at) in &copy.slots {
                    let from = self.slot_offset(copy.shape, slot)?;
                    let to = usize::try_from(self.host_offset(copy.shape, at)?).unwrap_or(0);
                    let (buffer, local) = self.pool.locate(from)?;
                    buffer.read(local, &mut self.host[to..to + bytes])?;
                }
                Ok(())
            }
            (MemoryDomain::HostPinned, dst) if here(dst) => {
                let bytes = usize::try_from(self.shape(copy.shape)?.slot_bytes).unwrap_or(0);
                for &(at, slot) in &copy.slots {
                    let to = self.slot_offset(copy.shape, slot)?;
                    let from = usize::try_from(self.host_offset(copy.shape, at)?).unwrap_or(0);
                    self.pool.write(to, &self.host[from..from + bytes])?;
                }
                Ok(())
            }
            (src, MemoryDomain::LocalDisk) if here(src) => {
                let (ids, slots): (Vec<u32>, Vec<u32>) = copy.slots.iter().copied().unzip();
                self.spill(device, copy.shape, true, &ids, &slots)
            }
            (MemoryDomain::LocalDisk, dst) if here(dst) => {
                let (slots, ids): (Vec<u32>, Vec<u32>) = copy.slots.iter().copied().unzip();
                self.spill(device, copy.shape, false, &ids, &slots)
            }
            (src, dst) => Err(Fault::Unbound {
                what: format!(
                    "a slot copy from {src:?} to {dst:?}: this pool moves slots within the \
                     device, to its host chunks and to its slot file"
                ),
            }),
        }
    }

    /// Where each of `slots` (host tier slot refs, shape by shape) lies in
    /// the host chunks, and its bytes, in order.
    fn tier_spans(&self, slots: &[(ShapeId, Vec<u32>)]) -> Result<Vec<(u64, u64)>> {
        let mut spans = Vec::new();
        for (shape, refs) in slots {
            let bytes = self.shape(*shape)?.slot_bytes;
            for &slot in refs {
                spans.push((self.host_offset(*shape, slot)?, bytes));
            }
        }
        Ok(spans)
    }

    /// See [`engine::Engine::read_tier_slots`]: the bytes of host tier
    /// slots, shape by shape, each laid out as `copy_slots` moves it. The
    /// caller has drained every frame.
    pub fn read_tier_slots(&self, slots: &[(ShapeId, Vec<u32>)]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for (offset, len) in self.tier_spans(slots)? {
            let from = usize::try_from(offset).unwrap_or(usize::MAX);
            let len = usize::try_from(len).unwrap_or(usize::MAX);
            let span = self.host.get(from..from.saturating_add(len)).ok_or(Fault::Ceiling {
                what: "bytes of the host chunks",
                need: offset + len as u64,
                have: self.host.len() as u64,
            })?;
            out.extend_from_slice(span);
        }
        Ok(out)
    }

    /// See [`engine::Engine::write_tier_slots`]: `bytes`, laid out as
    /// [`Self::read_tier_slots`] returns them, into host tier slots; refused
    /// whole when the byte count differs. The caller has drained every
    /// frame.
    pub fn write_tier_slots(&mut self, slots: &[(ShapeId, Vec<u32>)], bytes: &[u8]) -> Result<()> {
        let spans = self.tier_spans(slots)?;
        let need: u64 = spans.iter().map(|&(_, len)| len).sum();
        if bytes.len() as u64 != need {
            return Err(Fault::Ceiling {
                what: "bytes of host tier slots",
                need,
                have: bytes.len() as u64,
            });
        }
        let have = self.host.len() as u64;
        if let Some(&(offset, len)) = spans.iter().find(|&&(offset, len)| offset + len > have) {
            return Err(Fault::Ceiling {
                what: "bytes of the host chunks",
                need: offset + len,
                have,
            });
        }
        let mut at = 0usize;
        for (offset, len) in spans {
            let to = usize::try_from(offset).unwrap_or(usize::MAX);
            let len = usize::try_from(len).unwrap_or(usize::MAX);
            self.host[to..to + len].copy_from_slice(&bytes[at..at + len]);
            at += len;
        }
        Ok(())
    }

    /// Copies whole kv list slots (slot refs `ids`) and the slot file's
    /// `slots` the way `to_disk` says. The caller has drained every frame.
    fn spill(
        &mut self,
        device: &Context,
        id: ShapeId,
        to_disk: bool,
        ids: &[u32],
        slots: &[u32],
    ) -> Result<()> {
        if id != LIST {
            return Err(Fault::Unbound {
                what: format!("slot shape {id} to the slot file, which holds kv list slots"),
            });
        }
        Spill::room(self.disk.as_ref(), "disk kv pages", slots)?;
        let bytes = self.page_bytes();
        let places = ids
            .iter()
            .map(|&slot| self.slot_offset(LIST, slot))
            .collect::<Result<Vec<_>>>()?;
        let Some(disk) = self.disk.as_ref() else {
            return Ok(());
        };
        places.into_iter().zip(slots).try_for_each(|(at, &slot)| {
            let (buffer, local) = self.pool.locate(at)?;
            disk.copy(device, &[(buffer, local, bytes)], to_disk, &[0], &[slot])
        })
    }

    /// Opens the kv slot file under `dir` within `budget` bytes; the kv
    /// pages it holds.
    pub fn seat_disk(
        &mut self,
        device: &Context,
        dir: &std::path::Path,
        budget: u64,
    ) -> Result<u32> {
        let page = self.page_bytes();
        if page > 0 {
            let _ = device;
            self.disk = Spill::open(dir, "r0.kv", page, budget)?;
        }
        Ok(self.disk_pages())
    }

    pub fn unseat_disk(&mut self) {
        self.disk = None;
    }

    #[must_use]
    pub fn disk_pages(&self) -> u32 {
        Spill::slots(self.disk.as_ref())
    }

    /// Checks what a fire touches is committed: `demand`'s kv pages and
    /// state slots are one past the highest physical page and record unit
    /// it names; lifts the watermark over it.
    pub fn admit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        if demand.workspace > 0 {
            return Err(Fault::Ceiling {
                what: "pool workspace bytes",
                need: demand.workspace,
                have: 0,
            });
        }
        let chunk = self.layout.chunk_bytes;
        if demand.kv_pages > 0
            && let Some(list) = self.layout.list
        {
            let at = u64::from(demand.kv_pages - 1) * list.slot_bytes;
            self.pool
                .admit(u32::try_from(at / chunk).unwrap_or(u32::MAX))?;
        }
        if demand.state_slots > 0 && self.has_state() {
            let at = u64::from(demand.state_slots - 1) * self.unit(FIXED)?;
            self.pool
                .admit(u32::try_from(at / chunk).unwrap_or(u32::MAX))?;
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

    /// Backs the pool's first `segments` segments.
    pub fn commit_segments(&mut self, device: &Context, segments: u32) -> Result<()> {
        self.pool.commit_segments(device, segments)
    }

    /// The segments the seat runs lie in: what a caller firing directly
    /// backs.
    #[must_use]
    pub fn seat_segments(&self) -> u32 {
        let chunks = 1 + self.layout.seat_chunks();
        chunks
            .div_ceil(self.layout.segment_chunks.max(1))
            .min(self.pool.segments())
    }

    /// Gives back every segment at and past `segments`. The caller drained
    /// every frame, and the runtime evacuated what they held.
    pub fn release_segments(&mut self, segments: u32) {
        self.pool.release_segments(segments);
    }

    /// Moves each evacuated chunk onto its new physical chunk in `frame`
    /// and takes the moves into the chunk table.
    /// Returns each move as `(from, to)` physical list pages: the first
    /// page of each chunk moved and the pages a chunk holds.
    pub fn evacuate(
        &mut self,
        frame: &mut Frame,
        evacuation: &Evacuation,
    ) -> Result<(Vec<(u64, u64)>, u64)> {
        let moved = self.pool.evacuate(frame, evacuation)?;
        let per = u64::from(self.layout.list.map_or(0, |list| list.per_chunk));
        Ok((
            moved
                .into_iter()
                .map(|(from, to)| (u64::from(from) * per, u64::from(to) * per))
                .collect(),
            per,
        ))
    }
}

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
