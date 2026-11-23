//! The load's held state in one chunk pool ([`pool::Pool`]): every cache
//! row placed in the slot shapes of [`layout`], the slot refs the runtime
//! hands out translated to the physical slots kernels address, and the
//! copies, spills and evacuations that move slots.
pub mod experts;
pub mod frozen;
pub mod kv;
pub mod layout;
pub mod pool;
pub mod rows;

use engine::chunk::{ChunkFacts, Evacuation, ShapeId, SlotCopy};
use engine::transfer::{KvRun, MemoryDomain};
use kernels_slang::{KvPool, Segments, Tensor};
use poem_compiler::chunk::{ChunkRecipe, RowId, SlotLayout};
use poem_ir::{CacheRow, Dtype, Trace};

use crate::backend::{Backend, Buffer as _, Context as _, Frame as _, Memory, Pending as _};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use crate::run::{CachePool, CacheTable};
use crate::store::kv::{Facts, Paging};
use crate::store::layout::{FIXED, LIST, Layout, RS, Slots, WINDOW, gcd};
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

    /// `4 + pieces` ints a lane where the fire reads windowed rows: its
    /// folded index, buffer base, first row of the fire, folded index after
    /// it, and its record's units (`kernels_slang::Ring::lanes`).
    pub ring_lanes: Tensor,
}

/// The list a kv row's pages lie in: the kv list, or the window buffer of
/// a windowed row (CONTRACT-BC 19, 21), whose keys below a lane's folded
/// index lie in its ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Paged {
    List,
    Buffer,
}

impl Paged {
    /// The slot shape of this list.
    #[must_use]
    pub fn shape(self) -> ShapeId {
        match self {
            Paged::List => LIST,
            Paged::Buffer => WINDOW,
        }
    }
}

/// The window buffer as a load lays it: its slot's bytes and the tokens
/// (`T`, the kv page's) a slot holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferRows {
    pub bytes: u64,
    pub tokens: u32,
}

/// A windowed kv row's ring in a lane's record ([`crate::chunks::Rings`]):
/// `rows` (W) rows of its keys at byte `keys` of record piece `piece`, as
/// many of its values at byte `values` of piece `values_piece`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingAt {
    pub rows: u32,
    pub piece: u32,
    pub keys: u64,
    pub values_piece: u32,
    pub values: u64,
}

/// Where a windowed layer's ring lies in the recipe's ring run: its bytes
/// from `at` of the region, cut into pieces between `atom`s as `run` cuts
/// them.
#[derive(Debug, Clone)]
struct RingRun {
    rows: u32,
    at: u64,
    atom: u64,
    run: poem_compiler::chunk::FixedRun,
}

impl RingRun {
    /// The record piece and byte there the region's bytes `[from, from +
    /// len)` lie at, refused where a piece cut divides them.
    fn place(&self, from: u64, len: u64, what: &str) -> Result<(u32, u64)> {
        let per = self.run.atoms_per_piece.max(1);
        let from = self.at + from;
        let (first, last) = (from / self.atom, (from + len.max(1) - 1) / self.atom);
        if first / per != last / per {
            return Err(disagree(format!("cuts {what} across record pieces")));
        }
        let piece = *self
            .run
            .slots
            .get(usize::try_from(first / per).unwrap_or(usize::MAX))
            .ok_or_else(|| disagree(format!("lays {what} past the record's pieces")))?;
        let byte = self.run.offset + (first % per) * self.atom + from % self.atom;
        if byte % 4 != 0 {
            return Err(disagree(format!("lays {what} at byte {byte}, inside a word")));
        }
        Ok((piece, byte))
    }
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
        /// A windowed row's ring (its pages then the window buffer's).
        ring: Option<RingAt>,
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

/// A compressor's state beside kv row `row` (the row a pool gather keys
/// by its pages), in that row's slots: the recipe's `RowId::Pool` run, a
/// projection plane at `kv` and a score plane at `score` of the slot,
/// `width` bf16 a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Compressed {
    row: u32,
    paged: Paged,
    width: u64,
    kv: u64,
    score: u64,
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
    compressed: Vec<Compressed>,
    pub chunk_bytes: u64,
    pub list_bytes: u64,
    /// The list slot's tokens: the kv page size.
    pub tokens: u32,
    pub fixed_bytes: u64,
    pub pieces: u32,
    /// The window buffer, where the load has windowed rows.
    pub buffer: Option<BufferRows>,
    /// The page tables a fire stages, by the list each pages: the kv
    /// list's first (the All axis reads it), then the window buffer's.
    pub tables: Vec<Paged>,
    pub rs: Option<RsRows>,
    pub kv_rows: Vec<u32>,
    /// The kv rows the window buffer lays out, in order.
    pub buffer_rows: Vec<u32>,
    pub state_rows: Vec<u32>,
    /// The windowed layers' rings as the facts state them.
    pub windowed: Vec<engine::chunk::WindowedLayer>,
}

fn disagree(what: String) -> Fault {
    Fault::Unbound {
        what: format!("the GPU chunk recipe, which {what}"),
    }
}

impl Rows {
    /// Lays `trace`'s rows where `recipe` (the compiler's GPU chunk
    /// recipe, the one layout source) puts them; `facts` restates kv rows'
    /// heads, `rs` is the shell's rs buffer. Refuses where the recipe lays a
    /// row otherwise than this shell stores it.
    pub fn plan(
        trace: &Trace,
        recipe: &ChunkRecipe,
        facts: &Facts,
        rs: Option<&crate::rs::Layout>,
    ) -> Result<Rows> {
        if recipe.windows().next().is_some() {
            return Err(disagree(format!(
                "keeps window lists ({}), where this shell serves windowed rows as ring + \
                 buffer",
                recipe.shapes()
            )));
        }
        let (buffer, buffer_runs) = match recipe.window_buffer() {
            Some(SlotLayout::List {
                tokens,
                bytes,
                rows,
                ..
            }) => (
                Some(BufferRows {
                    bytes: *bytes,
                    tokens: *tokens,
                }),
                rows.as_slice(),
            ),
            _ => (None, &[][..]),
        };
        let rings = rings_of(recipe)?;
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
        // Without a kv list the window buffer's slot is the kv page.
        let tokens = if list_bytes == 0 {
            buffer.map_or(0, |b| b.tokens)
        } else {
            tokens
        };
        if let Some(held) = buffer.filter(|b| b.tokens != tokens) {
            return Err(disagree(format!(
                "buffers windowed rows {} tokens a slot beside {tokens}-token kv pages, \
                 where this shell's ring kernels read both as one page",
                held.tokens
            )));
        }
        let mut tables: Vec<Paged> = Vec::new();
        let mut rows = Vec::with_capacity(trace.caches.len());
        let mut kv_rows = Vec::new();
        let mut window_rows = Vec::new();
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
                    // Where the row lies: a kv list slot or the window
                    // buffer's, and the tokens a run there holds.
                    let (paged, run, run_tokens) =
                        if let Some(run) = runs.iter().find(|run| run.row == id) {
                            (Paged::List, run, tokens)
                        } else if let Some(run) = buffer_runs.iter().find(|run| run.row == id) {
                            (Paged::Buffer, run, tokens)
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
                    let ring = match paged {
                        Paged::List => {
                            kv_rows.push(at);
                            None
                        }
                        Paged::Buffer => {
                            window_rows.push(at);
                            let Some((_, at)) = rings.iter().find(|(row, _)| *row == id) else {
                                return Err(disagree(format!(
                                    "buffers windowed kv row `{name}` and gives it no ring"
                                )));
                            };
                            // Keys, then values: a shared plane reads its keys twice.
                            let half = u64::from(at.rows) * keys_cell;
                            let (piece, keys) =
                                at.place(0, half, &format!("kv row `{name}`'s ring keys"))?;
                            let (values_piece, values) = if own_values {
                                at.place(
                                    half,
                                    u64::from(at.rows) * values_cell,
                                    &format!("kv row `{name}`'s ring values"),
                                )?
                            } else {
                                (piece, keys)
                            };
                            Some(RingAt {
                                rows: at.rows,
                                piece,
                                keys,
                                values_piece,
                                values,
                            })
                        }
                    };
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
                        ring,
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
        let mut compressed = Vec::new();
        for (row, width) in poem_compiler::chunk::census::pooled(trace) {
            let id = RowId::Pool(row);
            let (paged, run, run_tokens) = if let Some(run) = runs.iter().find(|run| run.row == id)
            {
                (Paged::List, run, tokens)
            } else {
                return Err(disagree(format!(
                    "keeps no compressor state beside kv row {row}, which a pool gather \
                         keys by"
                )));
            };
            let plane = width * Dtype::Bf16.bytes_ceil();
            if run.per_token != 2 * plane || run.offset % 4 != 0 {
                return Err(disagree(format!(
                    "keeps kv row {row}'s compressor state at byte {} and {} bytes a token, \
                     where this shell keeps two word-aligned {plane}-byte planes",
                    run.offset, run.per_token
                )));
            }
            compressed.push(Compressed {
                row,
                paged,
                width,
                kv: run.offset,
                score: run.offset + u64::from(run_tokens) * plane,
            });
        }
        let rs = match rs {
            None => None,
            Some(layout) => Some(rs_rows(recipe, layout)?),
        };
        Ok(Rows {
            rows,
            compressed,
            chunk_bytes: recipe.chunk_bytes,
            list_bytes,
            tokens,
            fixed_bytes,
            pieces: pieces.max(1),
            buffer,
            // The kv list's table first: the All axis reads it (or the
            // window buffer's, where no row is in the kv list).
            tables: {
                tables.sort_by_key(|paged| match paged {
                    Paged::List => 0,
                    Paged::Buffer => 1,
                });
                tables
            },
            rs,
            kv_rows,
            buffer_rows: window_rows,
            state_rows,
            windowed: windowed_of(recipe),
        })
    }
}

/// The windowed layers' rings as the facts state them: each layer's share
/// of the ring region ([`crate::chunks::Rings`]).
fn windowed_of(recipe: &ChunkRecipe) -> Vec<engine::chunk::WindowedLayer> {
    let Some(windowed) = &recipe.windowed else {
        return Vec::new();
    };
    windowed
        .layers
        .iter()
        .zip(&windowed.region.layer_bytes)
        .map(|(layer, &bytes)| engine::chunk::WindowedLayer {
            layer: layer.layer.unwrap_or(u32::MAX),
            window: layer.window,
            bytes,
            ..engine::chunk::WindowedLayer::default()
        })
        .collect()
}

/// Each windowed kv row's ring: the plain per-layer rings
/// ([`crate::chunks::Rings`]) where the recipe's ring run lays the region.
fn rings_of(recipe: &ChunkRecipe) -> Result<Vec<(RowId, RingRun)>> {
    let Some(windowed) = &recipe.windowed else {
        return Ok(Vec::new());
    };
    let plain = poem_compiler::chunk::plain_rings(&windowed.layers, crate::chunks::RING_ALIGN);
    if plain.region != windowed.region {
        return Err(disagree(
            "lays a ring region other than this shell's plain per-layer rings".to_string(),
        ));
    }
    if windowed.ring_shape.is_some() {
        return Err(disagree(
            "lays the rings at chunk starts, where this shell reads them inside the record"
                .to_string(),
        ));
    }
    let run = windowed
        .ring_run(recipe)
        .ok_or_else(|| disagree("states windowed layers and no ring run".to_string()))?;
    let mut out = Vec::new();
    for (at, layer) in windowed.layers.iter().enumerate() {
        let [row] = layer.rows.as_slice() else {
            return Err(disagree(format!(
                "rings layer {:?} of {} kv rows, where this shell reads a ring a row",
                layer.layer,
                layer.rows.len()
            )));
        };
        out.push((
            *row,
            RingRun {
                rows: layer.window,
                at: plain.offsets[at],
                atom: windowed.region.atom.max(1),
                run: run.clone(),
            },
        ));
    }
    Ok(out)
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
pub struct Pools<B: Backend> {
    paging: Paging,
    rows: Vec<Row>,
    compressed: Vec<Compressed>,
    pool: Pool<B>,
    layout: Layout,
    /// The cache rows each shape lays out: the kv list's, the record's,
    /// then the window buffer's.
    shape_rows: Vec<Vec<u32>>,
    /// The windowed layers' rings, as the facts state them.
    windowed: Vec<engine::chunk::WindowedLayer>,
    /// The page tables a fire stages, by the list each pages.
    tables: Vec<Paged>,
    rs: Option<RsRows>,
    host: HostChunks<B>,
    host_chunks: u32,
    disk: Option<Spill<B>>,
    watermark: engine::frame::Demand,
    /// `(rung tokens, chunks)` of each activation rung, ascending: the top
    /// chunks a fire of the rung takes ([`crate::arena`]).
    act_chunks: Vec<(u32, u32)>,
    /// Where each record's own folds left it ([`frozen`]), carried by
    /// every whole-slot copy of the fixed shape this pool makes.
    frozen: frozen::Frozen,
}

/// A record's unit as a slot table's entry. Kernels read the entries as i32,
/// a negative one as no record, so a unit past `i32::MAX` is refused: staged,
/// it would read as a lane with no record.
pub fn slot_entry(unit: u32) -> Result<i32> {
    i32::try_from(unit).map_err(|_| Fault::Ceiling {
        what: "record units (a slot table's entry is an i32)",
        need: u64::from(unit),
        have: i32::MAX as u64,
    })
}

/// A record's units as a slot table's entries ([`slot_entry`]).
pub fn slot_entries(units: &[u32]) -> Result<Vec<i32>> {
    units.iter().map(|&unit| slot_entry(unit)).collect()
}

impl<B: Backend> Pools<B> {
    /// The bytes of one chunk of the pool.
    #[must_use]
    pub fn chunk_bytes(&self) -> u64 {
        self.pool.chunk_bytes()
    }

    /// The buffers of the pool's backed segments, in order.
    #[must_use]
    pub fn segment_buffers(&self) -> Vec<&B::Buffer> {
        self.pool.segment_buffers()
    }

    /// The pool `layout` sizes over `rows`, kv pages of `paging`'s size,
    /// and `host_chunks` host chunks beside it, its top chunks the
    /// activation span of `act_chunks`' rungs; nothing committed yet.
    pub fn reserve(
        device: &B::Context,
        rows: Rows,
        layout: Layout,
        paging: Paging,
        host_chunks: u32,
        act_chunks: Vec<(u32, u32)>,
    ) -> Result<Pools<B>> {
        let pool = Pool::reserve(
            layout.chunk_bytes,
            layout.total_chunks,
            layout.segment_chunks,
            [
                act_chunks.first().map_or(0, |&(_, chunks)| chunks),
                act_chunks.last().map_or(0, |&(_, chunks)| chunks),
            ],
        )?;
        let list_slots = layout.list.map_or(0, |list| {
            u64::from(list.per_chunk) * u64::from(layout.total_chunks)
        });
        let host = if B::MAPS_HOST {
            HostChunks::Mapped(
                (host_chunks > 0)
                    .then(|| {
                        B::Buffer::host(
                            device,
                            u64::from(host_chunks) * layout.chunk_bytes,
                            crate::books::owners::HOST_POOL,
                        )
                    })
                    .transpose()?,
            )
        } else {
            HostChunks::Heap(vec![
                0u8;
                usize::try_from(
                    u64::from(host_chunks) * layout.chunk_bytes
                )
                .unwrap_or(0)
            ])
        };
        Ok(Pools {
            paging: Paging {
                pages: list_slots.max(1),
                ..paging
            },
            shape_rows: vec![
                rows.kv_rows.clone(),
                rows.state_rows.clone(),
                rows.buffer_rows.clone(),
            ],
            windowed: rows.windowed.clone(),
            tables: rows.tables.clone(),
            rs: rows.rs,
            rows: rows.rows,
            compressed: rows.compressed,
            pool,
            layout,
            host,
            host_chunks,
            disk: None,
            watermark: engine::frame::Demand::ZERO,
            act_chunks,
            frozen: frozen::Frozen::default(),
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
            [self.host_chunks, self.disk_chunks()],
            &self.act_chunks,
        )
    }

    /// The activation span's buffer (the pool's last segment) and the byte
    /// in it the span ends at; `None` until the load backs it.
    #[must_use]
    pub fn span_place(&self) -> Option<(&B::Buffer, u64)> {
        self.pool.span_place()
    }

    /// The activation span's chunks: the pool's top ones, state's only where
    /// the runtime lowers the rung.
    #[must_use]
    pub fn span_chunks(&self) -> u32 {
        self.pool.span()
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
    pub fn locate(&self, at: u64) -> Result<(&B::Buffer, u64)> {
        self.pool.locate(at)
    }

    /// Logical chunk `chunk`'s first byte as a guest's state lies there
    /// (`guest::state`): the segment buffer its physical chunk sits in now
    /// and its offset there, its segment committed. A run of chunks from it
    /// lies at the next physical chunks of the same segment.
    pub fn chunk_place(&self, chunk: u32) -> Result<(&B::Buffer, u64)> {
        if chunk == 0 || chunk >= self.layout.total_chunks {
            return Err(Fault::Ceiling {
                what: "chunks a guest's state may name (past the null chunk)",
                need: u64::from(chunk) + 1,
                have: u64::from(self.layout.total_chunks),
            });
        }
        let physical = self.pool.physical(chunk)?;
        self.pool.admit(physical)?;
        self.pool
            .locate(u64::from(physical) * self.layout.chunk_bytes)
    }

    /// The bytes one slot of shape `id` holds.
    pub fn slot_bytes(&self, id: ShapeId) -> Result<u64> {
        Ok(self.shape(id)?.slot_bytes)
    }

    /// The slots of shape `id` one chunk holds.
    pub fn slots_per_chunk(&self, id: ShapeId) -> Result<u32> {
        Ok(self.shape(id)?.per_chunk)
    }

    /// Slot ref `slot` of shape `id` as a guest's table or cell lies
    /// (`guest::held`): the segment buffer its chunk sits in now and its
    /// first byte there.
    pub fn slot_place(&self, id: ShapeId, slot: u32) -> Result<(&B::Buffer, u64)> {
        self.pool.locate(self.slot_offset(id, slot)?)
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

    /// What a lane of a bind-only fire names by ref: `pages` slot refs of
    /// the list shape and a record (a slot ref of the fixed shape a piece),
    /// each in the first chunk past the null chunk. The fire reads and writes
    /// none of them; the pool takes them as any lane's (none of a shape the
    /// pool holds not).
    pub fn walk_refs(&self, pages: u32) -> Result<(Vec<u32>, Vec<u32>)> {
        let first = |id: ShapeId, count: u32| -> Result<Vec<u32>> {
            let per_chunk = self.shape(id)?.per_chunk.max(1);
            Ok((0..count).map(|at| per_chunk + at % per_chunk).collect())
        };
        let pages = if self.layout.list.is_some() {
            first(LIST, pages)?
        } else {
            Vec::new()
        };
        let record = if self.has_state() {
            first(FIXED, self.pieces())?
        } else {
            vec![0; self.pieces() as usize]
        };
        Ok((pages, record))
    }

    /// `pages` rs-list slot refs any lane may take, as
    /// [`Pools::walk_refs`] names the kv list's: a bind-only fire reads and
    /// writes none of them.
    pub fn walk_rs_refs(&self, pages: u32) -> Result<Vec<u32>> {
        let per_chunk = self.shape(RS)?.per_chunk.max(1);
        Ok((0..pages).map(|at| per_chunk + at % per_chunk).collect())
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

    /// The page table each windowed row's write descriptors read, by its
    /// window: the window buffer's.
    #[must_use]
    pub fn window_tables(&self) -> Vec<(u32, usize)> {
        let mut out: Vec<(u32, usize)> = Vec::new();
        for row in &self.rows {
            if let Row::Kv {
                ring: Some(ring), ..
            } = row
                && !out.iter().any(|&(w, _)| w == ring.rows)
            {
                out.push((ring.rows, self.table_of(Paged::Buffer)));
            }
        }
        out
    }

    /// The distinct windows the windowed rows read through.
    #[must_use]
    pub fn windows(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self.window_tables().into_iter().map(|(w, _)| w).collect();
        out.sort_unstable();
        out
    }

    /// The windowed layers' rings, as the facts state them.
    #[must_use]
    pub fn windowed(&self) -> &[engine::chunk::WindowedLayer] {
        &self.windowed
    }

    /// The widest ring a windowed row holds and the narrowest window, where
    /// the load has windowed rows: every ring holds at least its window.
    #[must_use]
    pub fn rings(&self) -> Option<(u32, u32)> {
        self.rows
            .iter()
            .filter_map(|row| match row {
                Row::Kv {
                    ring: Some(ring), ..
                } => Some(ring.rows),
                _ => None,
            })
            .fold(None, |held: Option<(u32, u32)>, w| {
                Some(held.map_or((w, w), |(a, b)| (a.max(w), b.min(w))))
            })
    }

    /// The paging of `paged`'s list: the kv page (the window buffer's slot
    /// holds as many rows), as many pages a seat as its context fills.
    #[must_use]
    pub fn paging_of(&self, _paged: Paged) -> Paging {
        self.paging
    }

    /// A lane's pages of `paged`'s list as kernels address them, in token
    /// order: `refs` (its slot refs) as units of [`Pools::unit`], or,
    /// seat-addressed, seat `seat`'s run from its page `first`, `pages` of
    /// them.
    pub fn list_pages(
        &self,
        paged: Paged,
        refs: &[u32],
        seat: Option<(u32, u32, u32)>,
    ) -> Result<Vec<u32>> {
        let Paged::Buffer = paged else {
            return self.kv_pages(refs, seat.map(|(slot, _, pages)| (slot, pages)));
        };
        let buffer = *self.shape(WINDOW)?;
        if let Some((slot, first, pages)) = seat
            && refs.is_empty()
        {
            let base = u64::from(slot) * u64::from(self.paging.pages_per_slot) + u64::from(first);
            return (0..pages)
                .map(|page| {
                    let at = u32::try_from(base + u64::from(page)).unwrap_or(u32::MAX);
                    self.units(WINDOW, buffer.seat_ref(at)?)
                })
                .collect();
        }
        refs.iter().map(|&r| self.units(WINDOW, r)).collect()
    }

    /// The window buffer's slot bytes and its unit (what one of its page ids
    /// counts), where the load has one.
    #[must_use]
    pub fn buffer_slot(&self) -> Option<(u64, u64)> {
        let shape = self.layout.shape(WINDOW)?;
        Some((shape.slot_bytes, self.unit(WINDOW).ok()?))
    }

    /// The fire's own window-buffer-shaped slots, `count` from byte `at`
    /// of the span's buffer (the first on the next unit), as kernel units.
    pub fn fire_units(&self, at: u64, count: u32) -> Result<Vec<u32>> {
        let (_, unit) = self.buffer_slot().ok_or_else(|| Fault::Unbound {
            what: "the window buffer, which this load holds none of".to_string(),
        })?;
        self.units_at((self.pool.span_base() + at).next_multiple_of(unit), count)
    }

    /// Window-buffer-shaped slots laid at virtual byte `at` of the pool (the
    /// fire's own, in its activation span), `count` of them, as kernel
    /// units; `at` a multiple of the unit.
    pub fn units_at(&self, at: u64, count: u32) -> Result<Vec<u32>> {
        let (slot, unit) = self.buffer_slot().ok_or_else(|| Fault::Unbound {
            what: "the window buffer, which this load holds none of".to_string(),
        })?;
        if !at.is_multiple_of(unit) || !slot.is_multiple_of(unit) {
            return Err(Fault::Unbound {
                what: format!(
                    "fire slots at byte {at}, off the window buffer's {unit}-byte units"
                ),
            });
        }
        (0..u64::from(count))
            .map(|k| {
                let byte = at + k * slot;
                u32::try_from(byte / unit).map_err(|_| Fault::Ceiling {
                    what: "slot units a kernel indexes",
                    need: byte,
                    have: u64::from(u32::MAX),
                })
            })
            .collect()
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
    pub fn segments(&self, handles: &Handles<B>) -> Result<Segments> {
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

    /// Whether a lane holds a record: recurrent state, or the windowed
    /// layers' rings.
    #[must_use]
    pub fn has_state(&self) -> bool {
        self.rows.iter().any(|row| {
            matches!(
                row,
                Row::State { .. } | Row::Kv {
                    ring: Some(_),
                    ..
                }
            )
        })
    }

    /// Whether the load holds recurrent state rows.
    #[must_use]
    pub fn has_recurrent(&self) -> bool {
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

    /// Each ring's geometry as the fold reads it (`attn/ring.slang`'s
    /// layer entries), its rows in order: its keys' and values' origins,
    /// its slot unit and row words, the pieces and the words its ring's keys
    /// and values lie at, and its ring rows, every word a u32 of the pool.
    /// [`Self::table`] binds the same rows in the same order.
    pub fn ring_layers(&self) -> Result<Vec<i32>> {
        let mut out = Vec::new();
        for row in &self.rows {
            let Row::Kv {
                paged,
                head_dim,
                kv_heads,
                keys,
                values,
                ring: Some(ring),
                ..
            } = *row
            else {
                continue;
            };
            let word = |v: u64| {
                i32::try_from(v).map_err(|_| Fault::Unbound {
                    what: format!("ring word {v}, which passes the fold's int"),
                })
            };
            out.extend([
                word(keys / 4)?,
                word(values / 4)?,
                word(self.unit(paged.shape())? / 4)?,
                word(u64::from(kv_heads) * u64::from(head_dim) / 2)?,
                word(u64::from(ring.piece))?,
                word(u64::from(ring.values_piece))?,
                word(ring.keys / 4)?,
                word(ring.values / 4)?,
                word(u64::from(ring.rows))?,
            ]);
        }
        Ok(out)
    }

    /// The cache table one fire binds.
    pub fn table(&self, handles: &Handles<B>, seats: &Seats) -> Result<CacheTable> {
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
                    ring,
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
                        ring: ring.map(|ring| kernels_slang::Ring {
                            lanes: seats.ring_lanes,
                            rows: ring.rows,
                            piece: ring.piece,
                            values_piece: ring.values_piece,
                            pieces: self.pieces(),
                            keys: u32::try_from(ring.keys / 4).unwrap_or(u32::MAX),
                            values: u32::try_from(ring.values / 4).unwrap_or(u32::MAX),
                            slot_stride: u32::try_from(fixed_unit / 4).unwrap_or(u32::MAX),
                        }),
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
        let states = self
            .compressed
            .iter()
            .map(|state| {
                let stride = u32::try_from(self.unit(state.paged.shape())? / 4).unwrap_or(u32::MAX);
                Ok((
                    state.row,
                    kernels_slang::attn::pool::PoolState {
                        pool: segments,
                        origin: [
                            u32::try_from(state.kv / 4).unwrap_or(u32::MAX),
                            u32::try_from(state.score / 4).unwrap_or(u32::MAX),
                        ],
                        page_stride: stride,
                        pitch: u32::try_from(state.width).unwrap_or(u32::MAX),
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(CacheTable(rows, states))
    }

    /// Zeroes the record at `units` (a piece each) in `frame`: a fresh
    /// lane's state, written in queue order, after every frame submitted
    /// before and before every one submitted after.
    pub fn clear(&self, frame: &mut B::Frame, units: &[u32]) -> Result<()> {
        if !self.has_state() {
            return Ok(());
        }
        let bytes = self.shape(FIXED)?.slot_bytes;
        let unit = self.unit(FIXED)?;
        for &id in units {
            self.pool.zero(frame, u64::from(id) * unit, bytes)?;
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
    pub fn copy_kv(&self, frame: &mut B::Frame, moves: &[KvRun]) -> Result<()> {
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

    /// The slot refs of the fixed shape a lane's record stands in, a piece
    /// each ([`Self::record`]); `None` where the pool holds no state.
    pub fn record_refs(&self, refs: Option<&[u32]>, seat: u32) -> Result<Option<Vec<u32>>> {
        if !self.has_state() {
            return Ok(None);
        }
        match refs {
            Some(refs) => Ok(Some(refs.to_vec())),
            None => {
                let pieces = self.pieces();
                let fixed = *self.shape(FIXED)?;
                (0..pieces)
                    .map(|p| fixed.seat_ref(seat * pieces + p))
                    .collect::<Result<Vec<u32>>>()
                    .map(Some)
            }
        }
    }

    /// The slot the last fold of the record in `pieces` left it at, where
    /// the record keeps its own position ([`frozen`]).
    #[must_use]
    pub fn frozen_at(&self, pieces: &[u32]) -> Option<u32> {
        self.frozen.held(pieces)
    }

    /// A fire left the record in `pieces` at `at`; `None` hands its
    /// position back to the verb that states it.
    pub fn leave_frozen(&mut self, pieces: &[u32], at: Option<u32>) {
        self.frozen.leave(pieces, at);
    }

    /// A kv page's bytes, as the slot file holds one: a list slot.
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        self.layout.list.map_or(0, |list| list.slot_bytes)
    }

    /// Copies whole slots of `id` (slot refs) `(src, dst)` on the device;
    /// a record's own frozen position goes with it ([`frozen`]).
    pub fn copy_shape(
        &mut self,
        frame: &mut B::Frame,
        id: ShapeId,
        moves: &[(u32, u32)],
    ) -> Result<()> {
        let bytes = self.shape(id)?.slot_bytes;
        for &(src, dst) in moves {
            let from = self.slot_offset(id, src)?;
            let to = self.slot_offset(id, dst)?;
            if from != to {
                self.pool.copy(frame, from, to, bytes)?;
            }
        }
        if id == FIXED {
            self.frozen
                .copied(frozen::Tier::Device, frozen::Tier::Device, moves);
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
    /// between the device and the host chunks, or between the device (or
    /// the host chunks) and the slot file. A host or disk copy waits out its
    /// frame. A record's
    /// own frozen position goes where its slot does ([`frozen`]).
    pub fn copy_slots(
        &mut self,
        device: &B::Context,
        frame: &mut B::Frame,
        copy: &SlotCopy,
    ) -> Result<()> {
        self.copy_slot_bytes(device, frame, copy)?;
        let (src, dst) = (B::is_device(copy.src), B::is_device(copy.dst));
        // Within the device `copy_shape` carried it.
        if copy.shape == FIXED && !(src && dst) {
            self.frozen.copied(
                frozen::Tier::of(copy.src, src),
                frozen::Tier::of(copy.dst, dst),
                &copy.slots,
            );
        }
        Ok(())
    }

    fn copy_slot_bytes(
        &mut self,
        device: &B::Context,
        frame: &mut B::Frame,
        copy: &SlotCopy,
    ) -> Result<()> {
        let here = |domain: MemoryDomain| B::is_device(domain);
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
                let bytes = self.shape(copy.shape)?.slot_bytes;
                let host = match &self.host {
                    HostChunks::Mapped(host) => host.as_ref().ok_or(Fault::Ceiling {
                        what: "host chunks of the state pool",
                        need: 1,
                        have: 0,
                    })?,
                    HostChunks::Heap(_) => {
                        let bytes = usize::try_from(bytes).unwrap_or(0);
                        for &(slot, at) in &copy.slots {
                            let from = self.slot_offset(copy.shape, slot)?;
                            let to =
                                usize::try_from(self.host_offset(copy.shape, at)?).unwrap_or(0);
                            let (buffer, local) = self.pool.locate(from)?;
                            let HostChunks::Heap(host) = &mut self.host else {
                                unreachable!("matched above")
                            };
                            buffer.read(local, &mut host[to..to + bytes])?;
                        }
                        return Ok(());
                    }
                };
                for &(slot, at) in &copy.slots {
                    let (buffer, local) = self.pool.locate(self.slot_offset(copy.shape, slot)?)?;
                    frame.copy(
                        buffer,
                        local,
                        host,
                        self.host_offset(copy.shape, at)?,
                        bytes,
                    )?;
                }
                Ok(())
            }
            (MemoryDomain::HostPinned, dst) if here(dst) => {
                let bytes = self.shape(copy.shape)?.slot_bytes;
                let host = match &self.host {
                    HostChunks::Mapped(host) => host.as_ref().ok_or(Fault::Ceiling {
                        what: "host chunks of the state pool",
                        need: 1,
                        have: 0,
                    })?,
                    HostChunks::Heap(_) => {
                        let bytes = usize::try_from(bytes).unwrap_or(0);
                        for &(at, slot) in &copy.slots {
                            let to = self.slot_offset(copy.shape, slot)?;
                            let from =
                                usize::try_from(self.host_offset(copy.shape, at)?).unwrap_or(0);
                            let HostChunks::Heap(host) = &self.host else {
                                unreachable!("matched above")
                            };
                            self.pool.write(to, &host[from..from + bytes])?;
                        }
                        return Ok(());
                    }
                };
                for &(at, slot) in &copy.slots {
                    let (buffer, local) = self.pool.locate(self.slot_offset(copy.shape, slot)?)?;
                    frame.copy(
                        host,
                        self.host_offset(copy.shape, at)?,
                        buffer,
                        local,
                        bytes,
                    )?;
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
            (MemoryDomain::HostPinned, MemoryDomain::LocalDisk) => {
                let (ids, slots): (Vec<u32>, Vec<u32>) = copy.slots.iter().copied().unzip();
                self.spill_host(copy.shape, true, &ids, &slots)
            }
            (MemoryDomain::LocalDisk, MemoryDomain::HostPinned) => {
                let (slots, ids): (Vec<u32>, Vec<u32>) = copy.slots.iter().copied().unzip();
                self.spill_host(copy.shape, false, &ids, &slots)
            }
            (src, dst) => Err(Fault::Unbound {
                what: format!(
                    "a slot copy from {src:?} to {dst:?}: this pool moves slots within the \
                     device, to its host chunks, and from either to its disk chunks"
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
        let spans = self.tier_spans(slots)?;
        let host = match &self.host {
            HostChunks::Mapped(host) => host.as_ref().ok_or(Fault::Ceiling {
                what: "host chunks of the state pool",
                need: 1,
                have: 0,
            })?,
            HostChunks::Heap(host) => {
                let mut out = Vec::new();
                for (offset, len) in spans {
                    let from = usize::try_from(offset).unwrap_or(usize::MAX);
                    let len = usize::try_from(len).unwrap_or(usize::MAX);
                    let span = host
                        .get(from..from.saturating_add(len))
                        .ok_or(Fault::Ceiling {
                            what: "bytes of the host chunks",
                            need: offset + len as u64,
                            have: host.len() as u64,
                        })?;
                    out.extend_from_slice(span);
                }
                return Ok(out);
            }
        };
        let total: u64 = spans.iter().map(|&(_, len)| len).sum();
        let mut out = vec![0u8; usize::try_from(total).unwrap_or(usize::MAX)];
        let mut at = 0usize;
        for (offset, len) in spans {
            let len = usize::try_from(len).unwrap_or(usize::MAX);
            host.read(offset, &mut out[at..at + len])?;
            at += len;
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
        // Bytes from outside the pool carry no position of a record's own:
        // what they hold stands where its verb states.
        for (shape, refs) in slots {
            if *shape == FIXED {
                self.frozen.written(frozen::Tier::Host, refs);
            }
        }
        let host = match &mut self.host {
            HostChunks::Mapped(host) => host.as_mut().ok_or(Fault::Ceiling {
                what: "host chunks of the state pool",
                need: 1,
                have: 0,
            })?,
            HostChunks::Heap(host) => {
                let have = host.len() as u64;
                if let Some(&(offset, len)) =
                    spans.iter().find(|&&(offset, len)| offset + len > have)
                {
                    return Err(Fault::Ceiling {
                        what: "bytes of the host chunks",
                        need: offset + len,
                        have,
                    });
                }
                let mut at = 0usize;
                for (offset, len) in spans {
                    let len = usize::try_from(len).unwrap_or(usize::MAX);
                    let to = usize::try_from(offset).unwrap_or(usize::MAX);
                    host[to..to + len].copy_from_slice(&bytes[at..at + len]);
                    at += len;
                }
                return Ok(());
            }
        };
        let mut at = 0usize;
        for (offset, len) in spans {
            let len = usize::try_from(len).unwrap_or(usize::MAX);
            host.write(offset, &bytes[at..at + len])?;
            at += len;
        }
        Ok(())
    }

    /// The bytes one disk chunk takes: a pool chunk, on a direct I/O
    /// boundary. A disk chunk is laid out as the device's (and the host
    /// tier's), any shape's slots in any chunk.
    #[must_use]
    pub fn disk_chunk_bytes(&self) -> u64 {
        self.layout
            .chunk_bytes
            .next_multiple_of(engine::disk::ALIGN as u64)
    }

    /// How the slot file divides the disk tier: each shape's id, its slots
    /// a disk chunk and the bytes of each, packed as on the device.
    #[must_use]
    pub fn disk_division(&self) -> Vec<(ShapeId, u32, u64)> {
        self.layout
            .shapes()
            .map(|shape| (shape.id, shape.per_chunk, shape.slot_bytes))
            .collect()
    }

    /// The byte of the slot file disk ref `slot` of shape `id` starts at:
    /// slot `index` of disk chunk `chunk`, as on the device.
    fn disk_place(&self, id: ShapeId, slot: u32) -> Result<u64> {
        let shape = *self.shape(id)?;
        let (chunk, index) = shape.split(slot);
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

    /// Copies whole slots of shape `id` between the device (slot refs
    /// `ids`) and the slot file (disk refs `slots`) the way `to_disk` says.
    /// The caller has drained every frame.
    fn spill(
        &mut self,
        device: &B::Context,
        id: ShapeId,
        to_disk: bool,
        ids: &[u32],
        slots: &[u32],
    ) -> Result<()> {
        let bytes = self.shape(id)?.slot_bytes;
        let places = slots
            .iter()
            .map(|&slot| self.disk_place(id, slot))
            .collect::<Result<Vec<_>>>()?;
        let offsets = ids
            .iter()
            .map(|&slot| self.slot_offset(id, slot))
            .collect::<Result<Vec<_>>>()?;
        let Some(mut disk) = self.disk.take() else {
            return Ok(());
        };
        let copied = offsets
            .into_iter()
            .map(|at| self.pool.locate(at))
            .collect::<Result<Vec<_>>>()
            .and_then(|rows| disk.copy(device, &rows, &places, bytes, to_disk));
        self.disk = Some(disk);
        copied
    }

    /// Copies whole slots of shape `id` between the host chunks (host slot
    /// refs `ids`) and the slot file (disk refs `slots`) the way `to_disk`
    /// says: the host tier's idle state going on down to disk, and back.
    /// The bytes pass through the host alone. The caller has drained every
    /// frame.
    fn spill_host(&mut self, id: ShapeId, to_disk: bool, ids: &[u32], slots: &[u32]) -> Result<()> {
        let bytes = self.shape(id)?.slot_bytes;
        let places = slots
            .iter()
            .map(|&slot| self.disk_place(id, slot))
            .collect::<Result<Vec<_>>>()?;
        let hosts = ids
            .iter()
            .map(|&slot| self.host_offset(id, slot))
            .collect::<Result<Vec<_>>>()?;
        let Some(file) = self.disk.as_ref().map(Spill::file) else {
            return Ok(());
        };
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        let mut stage = engine::disk::Aligned::zeroed(cover_room(bytes));
        let mut edge = engine::disk::Aligned::zeroed(engine::disk::ALIGN);
        for (at, disk_at) in hosts.into_iter().zip(places) {
            let (start, span, lead) = cover(disk_at, bytes);
            let staged = &mut stage.as_mut_slice()[..span];
            let span = |host_len: usize| {
                let from = usize::try_from(at).unwrap_or(usize::MAX);
                from.checked_add(len)
                    .filter(|&end| end <= host_len)
                    .map(|end| from..end)
                    .ok_or(Fault::Ceiling {
                        what: "bytes of the host chunks",
                        need: at + bytes,
                        have: host_len as u64,
                    })
            };
            if to_disk {
                let slot = &mut staged[lead..lead + len];
                match &self.host {
                    HostChunks::Mapped(host) => host
                        .as_ref()
                        .ok_or(Fault::Ceiling {
                            what: "host chunks of the state pool",
                            need: 1,
                            have: 0,
                        })?
                        .read(at, slot)?,
                    HostChunks::Heap(host) => {
                        slot.copy_from_slice(&host[span(host.len())?]);
                    }
                }
                keep_edges(file, start, staged, lead, len, &mut edge)?;
                file.write_span(start, staged).map_err(slot_file)?;
            } else {
                file.read_span(start, staged).map_err(slot_file)?;
                let slot = &staged[lead..lead + len];
                match &mut self.host {
                    HostChunks::Mapped(host) => host
                        .as_mut()
                        .ok_or(Fault::Ceiling {
                            what: "host chunks of the state pool",
                            need: 1,
                            have: 0,
                        })?
                        .write(at, slot)?,
                    HostChunks::Heap(host) => {
                        let span = span(host.len())?;
                        host[span].copy_from_slice(slot);
                    }
                }
            }
        }
        Ok(())
    }

    /// Opens the slot file under `dir` within `budget` bytes: chunks of
    /// [`Self::disk_chunk_bytes`], any shape's slots in any chunk, laid out
    /// as on the device. The disk chunks it holds.
    pub fn seat_disk(
        &mut self,
        device: &B::Context,
        dir: &std::path::Path,
        budget: u64,
    ) -> Result<u32> {
        let chunk = self.disk_chunk_bytes();
        let widest = self.layout.shapes().map(|shape| shape.slot_bytes).max();
        if let (true, Some(widest)) = (chunk > 0, widest) {
            self.disk = Spill::open(device, dir, "r0.state", chunk, cover_room(widest) as u64, budget)?;
        }
        Ok(self.disk_chunks())
    }

    /// Whether the host chunks are device-mapped memory a frame copies
    /// to and from (else a heap the host copies through, in the call).
    #[must_use]
    pub fn host_mapped(&self) -> bool {
        matches!(self.host, HostChunks::Mapped(_))
    }

    /// Waits out a slot file read's copy onto the device still in flight.
    pub fn settle_spill(&mut self) -> Result<()> {
        match &mut self.disk {
            Some(Spill::Mapped { landing, .. }) => {
                landing.take().map_or(Ok(()), |landing| landing.wait())
            }
            _ => Ok(()),
        }
    }

    pub fn unseat_disk(&mut self) {
        self.disk = None;
    }

    /// The disk tier's chunks.
    #[must_use]
    pub fn disk_chunks(&self) -> u32 {
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
    pub fn commit_segments(&mut self, device: &B::Context, segments: u32) -> Result<()> {
        self.pool.commit_segments(device, segments)
    }

    /// The segments a load backs: the null chunk's and the span's where
    /// the pool resizes, every one where it is laid whole
    /// ([`Backend::RESIZES_POOL`]).
    #[must_use]
    pub fn load_segments(&self) -> u32 {
        if B::RESIZES_POOL {
            1
        } else {
            self.pool.segments()
        }
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
    pub fn evacuate(&mut self, frame: &mut B::Frame, evacuation: &Evacuation) -> Result<()> {
        self.pool.evacuate(frame, evacuation)?;
        Ok(())
    }
}

/// Host chunks laid out like the device's, one id space beside them: a
/// mapped host buffer frames copy through where the API maps host memory
/// ([`Backend::MAPS_HOST`]), else process memory the device side is read
/// and written through the queue.
#[derive(Debug)]
enum HostChunks<B: Backend> {
    Mapped(Option<B::Buffer>),
    Heap(Vec<u8>),
}

/// The bytes a batched spill stages at once.
const STAGE_BYTES: usize = 32 << 20;

/// The aligned span of the slot file the bytes `[at, at + len)` lie in:
/// its first byte, its length, and the byte `at` falls on within it.
fn cover(at: u64, len: u64) -> (u64, usize, usize) {
    let align = engine::disk::ALIGN as u64;
    let start = at / align * align;
    let end = (at + len).next_multiple_of(align);
    (
        start,
        usize::try_from(end - start).unwrap_or(usize::MAX),
        usize::try_from(at - start).unwrap_or(usize::MAX),
    )
}

/// The bytes a cover of a slot of `slot_bytes` takes at most.
fn cover_room(slot_bytes: u64) -> usize {
    let align = engine::disk::ALIGN as u64;
    usize::try_from(slot_bytes.next_multiple_of(align) + align).unwrap_or(usize::MAX)
}

/// Fills `span` (the cover from `start`) outside the slot's `[lead, lead +
/// len)` from the file, so writing the whole span leaves the slots beside
/// it as they are; `edge` is one aligned block of scratch. Run right before
/// the write, after any earlier write of the same blocks.
fn keep_edges(
    file: &engine::disk::SlotFile,
    start: u64,
    span: &mut [u8],
    lead: usize,
    len: usize,
    edge: &mut engine::disk::Aligned,
) -> Result<()> {
    let align = engine::disk::ALIGN;
    let block = edge.as_mut_slice();
    if lead > 0 {
        file.read_span(start, block).map_err(slot_file)?;
        span[..lead].copy_from_slice(&block[..lead]);
    }
    let end = lead + len;
    if end < span.len() {
        let last = span.len() - align;
        file.read_span(start + last as u64, block)
            .map_err(slot_file)?;
        span[end..].copy_from_slice(&block[end - last..]);
    }
    Ok(())
}

/// The slot file and how slots reach it: through one mapped stage that
/// stands for the file's life where the API maps host memory, else through
/// stages that live for one copy, so an idle tier holds no memory.
#[derive(Debug)]
enum Spill<B: Backend> {
    Mapped {
        file: engine::disk::SlotFile,
        stage: B::Buffer,
        /// The mapping need not start on the boundary direct I/O asks for.
        at: usize,
        /// The bytes the stage holds past `at`: the widest slot's cover.
        room: usize,
        /// A copy out of the stage may still be in flight.
        landing: Option<B::Pending>,
    },
    Batched {
        file: engine::disk::SlotFile,
    },
}

impl<B: Backend> Spill<B> {
    /// The file of `budget` bytes in chunks of `chunk_bytes`, a slot's
    /// cover `stride` bytes at most.
    fn open(
        device: &B::Context,
        dir: &std::path::Path,
        name: &str,
        chunk_bytes: u64,
        stride: u64,
        budget: u64,
    ) -> Result<Option<Spill<B>>> {
        let Some(file) =
            engine::disk::SlotFile::open(dir, name, chunk_bytes, budget).map_err(slot_file)?
        else {
            return Ok(None);
        };
        if !B::MAPS_HOST {
            return Ok(Some(Spill::Batched { file }));
        }
        let room = usize::try_from(stride).unwrap_or(usize::MAX);
        let stage = B::Buffer::host(
            device,
            stride + engine::disk::ALIGN as u64,
            crate::books::owners::SPILL,
        )?;
        let mapped = stage.mapped_ptr().ok_or(Fault::Deviceless)?;
        Ok(Some(Spill::Mapped {
            file,
            stage,
            at: mapped.next_multiple_of(engine::disk::ALIGN) - mapped,
            room,
            landing: None,
        }))
    }

    fn file(&self) -> &engine::disk::SlotFile {
        match self {
            Spill::Mapped { file, .. } | Spill::Batched { file } => file,
        }
    }

    fn slots(spill: Option<&Spill<B>>) -> u32 {
        spill.map_or(0, |spill| spill.file().slots())
    }

    /// Slots of `bytes` each at `rows` (a buffer and the byte in it) and at
    /// bytes `places` of the file: a write returns once on disk; a read
    /// once its copy onto the device is submitted (mapped) or has landed
    /// (batched).
    #[allow(unsafe_code)]
    fn copy(
        &mut self,
        device: &B::Context,
        rows: &[(&B::Buffer, u64)],
        places: &[u64],
        bytes: u64,
        to_disk: bool,
    ) -> Result<()> {
        let Spill::Mapped {
            file,
            stage,
            at,
            room,
            landing,
        } = self
        else {
            return self.copy_batched(device, rows, places, bytes, to_disk);
        };
        let base = stage.mapped_ptr().ok_or(Fault::Deviceless)? + *at;
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        let mut edge = engine::disk::Aligned::zeroed(engine::disk::ALIGN);
        for (&(buffer, local), &disk_at) in rows.iter().zip(places) {
            let (start, span, lead) = cover(disk_at, bytes);
            if span > *room {
                return Err(Fault::Ceiling {
                    what: "bytes of the slot file's stage",
                    need: span as u64,
                    have: *room as u64,
                });
            }
            if let Some(landing) = landing.take() {
                landing.wait()?;
            }
            // SAFETY: the stage stays mapped while `self` lives, holds
            // `room` bytes past `at`, and no copy in flight touches it now
            // (waited above, or committed below).
            let staged = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, span) };
            if !to_disk {
                file.read_span(start, staged).map_err(slot_file)?;
            }
            let mut frame = device.frame()?;
            let offset = (*at + lead) as u64;
            if to_disk {
                frame.copy(buffer, local, stage, offset, bytes)?;
                frame.commit()?;
                keep_edges(file, start, staged, lead, len, &mut edge)?;
                file.write_span(start, staged).map_err(slot_file)?;
            } else {
                frame.copy(stage, offset, buffer, local, bytes)?;
                *landing = Some(frame.commit_async(None)?);
            }
        }
        Ok(())
    }

    /// Returns once every row is on disk or on the device. The stages live
    /// for the call, so an idle tier holds no memory. No drain: every frame
    /// was submitted before the lane took this copy, and the queue runs
    /// submissions in order, so a row leaving reads what those frames wrote
    /// and a row arriving is there for the frame after.
    fn copy_batched(
        &self,
        device: &B::Context,
        rows: &[(&B::Buffer, u64)],
        places: &[u64],
        bytes: u64,
        to_disk: bool,
    ) -> Result<()> {
        let file = self.file();
        if rows.is_empty() {
            return Ok(());
        }
        // Each row's cover in a stride of its own, from an `ALIGN` boundary.
        let stride = cover_room(bytes);
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        let batch = (STAGE_BYTES / stride).max(1).min(rows.len());
        let kind = if to_disk {
            Memory::Readback
        } else {
            Memory::Staging
        };
        let mut stage = B::Buffer::with(
            device,
            (batch * stride) as u64,
            kind,
            crate::books::owners::SPILL,
        )?;
        let mut host = engine::disk::Aligned::zeroed(batch * stride);
        let mut edge = engine::disk::Aligned::zeroed(engine::disk::ALIGN);
        for (rows, places) in rows.chunks(batch).zip(places.chunks(batch)) {
            let staged = &mut host.as_mut_slice()[..rows.len() * stride];
            if !to_disk {
                for (&at, row) in places.iter().zip(staged.chunks_exact_mut(stride)) {
                    let (start, span, _) = cover(at, bytes);
                    file.read_span(start, &mut row[..span])
                        .map_err(slot_file)?;
                }
                stage.write(0, staged)?;
            }
            let mut frame = device.frame()?;
            for (row, (&(slab, on_device), &at)) in rows.iter().zip(places).enumerate() {
                let offset = (row * stride + cover(at, bytes).2) as u64;
                if to_disk {
                    frame.copy(slab, on_device, &stage, offset, bytes)?;
                } else {
                    frame.copy(&stage, offset, slab, on_device, bytes)?;
                }
            }
            frame.commit()?;
            if to_disk {
                stage.read(0, staged)?;
                // One row after another, each keeping what the rows before
                // it wrote beside it.
                for (&at, row) in places.iter().zip(staged.chunks_exact_mut(stride)) {
                    let (start, span, lead) = cover(at, bytes);
                    let row = &mut row[..span];
                    keep_edges(file, start, row, lead, len, &mut edge)?;
                    file.write_span(start, row).map_err(slot_file)?;
                }
            }
        }
        Ok(())
    }
}

/// A slot-file write failing for want of room on its file system is the
/// tier below full ([`Fault::TierFull`]): nothing on the device changed.
fn slot_file(error: std::io::Error) -> Fault {
    if error.kind() == std::io::ErrorKind::StorageFull {
        return Fault::TierFull(error.to_string());
    }
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

#[cfg(test)]
mod tests {
    use super::{slot_entries, slot_entry};
    use crate::error::Fault;

    /// A unit a slot table's i32 cannot state is refused by name, where a
    /// cast would stage a negative entry: no record.
    #[test]
    fn a_record_unit_past_i32_is_refused() {
        assert_eq!(slot_entry(0).ok(), Some(0));
        assert_eq!(slot_entry(i32::MAX as u32).ok(), Some(i32::MAX));
        for unit in [1 << 31, u32::MAX] {
            let refused = slot_entry(unit).expect_err("past i32");
            assert!(
                matches!(
                    refused,
                    Fault::Ceiling { need, have, .. }
                        if need == u64::from(unit) && have == i32::MAX as u64
                ),
                "{refused}"
            );
        }
        assert_eq!(slot_entries(&[3, 7]).ok(), Some(vec![3, 7]));
        assert!(slot_entries(&[3, 1 << 31]).is_err());
    }
}
