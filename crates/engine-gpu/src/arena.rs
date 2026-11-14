//! A fire's activation: the arena the compiler carves by liveness and the
//! kernel scratch rooms ([`crate::scratch`]), laid together at the top of
//! the state pool (REDESIGN 2.3, 2.4). The span is the pool's top chunks,
//! in its last segment's buffer and never across two; chunk ids are the
//! pool's.
//!
//! A rung (fires of at most so many tokens, and as many lanes) takes the
//! carve laid for its width, anchored at the span's end: the place each of
//! its values and rooms takes is fixed, so a frame recorded at a rung binds
//! the same bytes whatever the pool's rung is, and the span under the
//! current rung's carve is state's to take.

use kernels_slang::Tensor;
use poem_compiler::{ArenaMap, Budgets, CompiledModel, FireRows};
use poem_exec::store::arena::rect;
use poem_ir::{Trace, ValueId};

use crate::backend::{Backend, Tactics};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use crate::run::SlotTable;
use crate::scratch::{FireScratch, Rooms};
use crate::store::kv::Paging;

/// What a rung's carve and its rooms are each aligned to: no storage
/// binding a device asks more of (Vulkan's `minStorageBufferOffsetAlignment`
/// is at most 256).
const ALIGN: u64 = 256;

/// One rung: fires of at most `tokens` tokens and as many lanes.
#[derive(Debug)]
pub struct Rung {
    pub tokens: u32,
    map: ArenaMap,
    rooms: Rooms,
    /// The rows each value's placement holds in this carve.
    capacity: Vec<u32>,
    /// Where the rooms start, past the carve.
    rooms_at: u64,
    /// Where the rs scratch starts, past the rooms, and its bytes: what a
    /// fire of the rung's tokens and lanes replays its recurrent rows in,
    /// its own rows and, a lane, the [`engine::fire::UNFOLDED_ROWS`] it may
    /// carry in (0 for a model with no recurrent buffer).
    rs_at: u64,
    rs_bytes: u64,
    /// The carve, the rooms and the rs scratch.
    bytes: u64,
}

impl Rung {
    #[allow(clippy::too_many_arguments)]
    fn lay<T: Tactics + ?Sized>(
        trace: &Trace,
        compiled: &CompiledModel,
        budgets: &Budgets,
        tokens: u32,
        paging: Paging,
        tuning: &T,
        routers: &[u32],
        rs: Option<&crate::rs::Layout>,
    ) -> Result<Rung> {
        let map = if tokens >= budgets.tokens.max_tokens {
            compiled.arena.clone()
        } else {
            poem_compiler::arena_rung(trace, budgets, compiled, tokens).map_err(|why| {
                Fault::Program {
                    at: "arena::Rung::lay",
                    why: format!("the carve for fires of {tokens} tokens: {why}"),
                }
            })?
        };
        let rooms = Rooms::lay(
            trace, compiled, &map, budgets, tokens, paging, tuning, routers,
        );
        let rooms_at = map.bytes.next_multiple_of(ALIGN);
        let rs_at = (rooms_at + rooms.bytes()).next_multiple_of(ALIGN);
        let rs_bytes = rs.map_or(0, |layout| {
            let lanes = tokens.min(budgets.tokens.max_lanes).max(1);
            let rows = tokens
                .min(budgets.tokens.max_tokens)
                .saturating_add(lanes.saturating_mul(engine::fire::UNFOLDED_ROWS));
            crate::rs::Seat::scratch_bytes(layout, rows, lanes)
        });
        let bytes = (rs_at + rs_bytes).next_multiple_of(ALIGN);
        Ok(Rung {
            tokens,
            capacity: capacities(&map),
            map,
            rooms,
            rooms_at,
            rs_at,
            rs_bytes,
            bytes,
        })
    }

    /// The carve, the rooms and the rs scratch, aligned.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether a fire of `rows` fits this rung's carve.
    fn holds(&self, rows: FireRows) -> bool {
        let tokens = u64::from(self.tokens);
        rows.tokens <= tokens && rows.lanes <= tokens && rows.readouts <= tokens
    }
}

/// The activation span: every rung's carve, and the buffer and place the
/// pool gave it.
#[derive(Debug)]
pub struct Span<B: Backend> {
    /// Ascending; the last is the top fire's (the compiler's own carve).
    rungs: Vec<Rung>,
    /// For each rung, the carve its fires take ([`Span::widen`]).
    widest: Vec<usize>,
    routers: Vec<u32>,
    /// The rungs the pool states, `(tokens, chunks)`, and its chunk's
    /// bytes ([`Span::widen`]).
    stated: Vec<(u32, u32)>,
    chunk_bytes: u64,
    /// The pool's last segment and the byte in it the span ends at.
    place: Option<(B::Buffer, u64)>,
}

impl<B: Backend> Span<B> {
    /// The carve, rooms and rs scratch (`rs` the recurrent buffer's
    /// layout, if any) of every rung of `tokens` (ascending, the last
    /// `budgets`' own ceiling); nothing placed yet.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        trace: &Trace,
        compiled: &CompiledModel,
        budgets: &Budgets,
        tokens: &[u32],
        paging: Paging,
        tuning: &B::Tuning,
        rs: Option<&crate::rs::Layout>,
    ) -> Result<Span<B>> {
        let routers = crate::scratch::routers(trace);
        let rungs = tokens
            .iter()
            .map(|&tokens| {
                Rung::lay(
                    trace, compiled, budgets, tokens, paging, tuning, &routers, rs,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Span {
            widest: (0..rungs.len()).collect(),
            rungs,
            routers,
            stated: Vec::new(),
            chunk_bytes: 0,
            place: None,
        })
    }

    #[must_use]
    pub fn rungs(&self) -> &[Rung] {
        &self.rungs
    }

    /// The bytes the widest rung takes: the span's.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.rungs.iter().map(Rung::bytes).max().unwrap_or(0)
    }

    /// `(rung tokens, chunks)` of every rung, chunks of `chunk_bytes`,
    /// ascending in both (`ChunkFacts::act_chunks`).
    #[must_use]
    pub fn act_chunks(&self, chunk_bytes: u64) -> Vec<(u32, u32)> {
        let mut most = 0u32;
        self.rungs
            .iter()
            .map(|rung| {
                let chunks =
                    u32::try_from(rung.bytes.div_ceil(chunk_bytes.max(1))).unwrap_or(u32::MAX);
                most = most.max(chunks);
                (rung.tokens, most)
            })
            .collect()
    }

    /// Lays the span in `buffer`, ending at its byte `end`.
    pub fn place(&mut self, buffer: B::Buffer, end: u64) -> Result<()> {
        let end = end - end % ALIGN;
        if end < self.bytes() {
            return Err(Fault::Ceiling {
                what: "bytes of the activation span in the pool's last segment",
                need: self.bytes(),
                have: end,
            });
        }
        self.place = Some((buffer, end));
        Ok(())
    }

    fn placed(&self) -> Result<(&B::Buffer, u64)> {
        self.place
            .as_ref()
            .map(|(buffer, end)| (buffer, *end))
            .ok_or_else(|| Fault::Unbound {
                what: "the activation span, which the pool has not placed".to_string(),
            })
    }

    /// The buffer the span lies in.
    pub fn buffer(&self) -> Result<&B::Buffer> {
        Ok(self.placed()?.0)
    }

    /// The carve a fire of `rows` takes: the widest whose bytes lie in the
    /// chunks stated for the narrowest rung that holds it
    /// ([`Span::widen`]).
    #[must_use]
    pub fn rung_of(&self, rows: FireRows) -> usize {
        let narrowest = self
            .rungs
            .iter()
            .position(|rung| rung.holds(rows))
            .unwrap_or(self.rungs.len().saturating_sub(1));
        self.widest.get(narrowest).copied().unwrap_or(narrowest)
    }

    /// Takes the rungs the pool states, `(tokens, chunks)` of `chunk_bytes`
    /// each class once: a fire then carves the widest carve that lies in
    /// its rung's stated chunks, so where rungs share their chunks (all of
    /// them, where the span lies past the kernels' state index) it lays
    /// out as at the top fire and lends nothing it would not.
    pub fn widen(&mut self, stated: &[(u32, u32)], chunk_bytes: u64) {
        self.stated = stated.to_vec();
        self.chunk_bytes = chunk_bytes;
        let room = |tokens: u32| {
            stated
                .iter()
                .find(|&&(rung, _)| rung >= tokens)
                .or(stated.last())
                .map_or(0, |&(_, chunks)| u64::from(chunks) * chunk_bytes)
        };
        self.widest = (0..self.rungs.len())
            .map(|at| {
                let room = room(self.rungs[at].tokens);
                (at..self.rungs.len())
                    .rev()
                    .find(|&wide| self.rungs[wide].bytes <= room)
                    .unwrap_or(at)
            })
            .collect();
    }

    /// Where the guest programs' room of a fire of `tokens` tokens and
    /// lanes ends in the span's buffer: under the chunks stated for the
    /// narrowest rung holding it, which its carve lies in. The runtime
    /// funds the programs' fire terms right under those chunks
    /// (`FireFacts`, folded into each rung's activation), so their value
    /// scratch lies there, a lane under another.
    pub fn guest_top(&self, tokens: u32) -> Result<(&B::Buffer, u64)> {
        let (buffer, end) = self.placed()?;
        let chunks = self
            .stated
            .iter()
            .find(|&&(rung, _)| rung >= tokens)
            .or(self.stated.last())
            .map_or(0, |&(_, chunks)| u64::from(chunks));
        let top = end.saturating_sub(chunks * self.chunk_bytes);
        Ok((buffer, top - top % ALIGN))
    }

    /// Where rung `rung`'s carve starts in the span's buffer: the span's
    /// end less its bytes.
    pub fn base(&self, rung: usize) -> Result<u64> {
        let (_, end) = self.placed()?;
        Ok(end - self.rungs[rung].bytes)
    }

    /// A fire's slots at rung `rung`: its carve at its place.
    pub fn slots(&self, handles: &Handles<B>, rung: usize, rows: FireRows) -> Result<SlotTable> {
        let (buffer, _) = self.placed()?;
        carve(
            handles,
            buffer,
            self.base(rung)?,
            &self.rungs[rung].map,
            rows,
        )
    }

    /// Rung `rung`'s rs scratch: the span's buffer, where the region
    /// starts in it and its bytes.
    pub fn rs(&self, rung: usize) -> Result<(&B::Buffer, u64, u64)> {
        let (buffer, _) = self.placed()?;
        let at = &self.rungs[rung];
        Ok((buffer, self.base(rung)? + at.rs_at, at.rs_bytes))
    }

    /// A fire's rooms at rung `rung`.
    pub fn scratch(&self, rung: usize) -> Result<FireScratch<'_, B>> {
        let (buffer, _) = self.placed()?;
        let at = &self.rungs[rung];
        Ok(FireScratch::new(
            &at.rooms,
            buffer,
            self.base(rung)? + at.rooms_at,
            &at.capacity,
            &self.routers,
        ))
    }
}

pub fn carve<B: Backend>(
    handles: &Handles<B>,
    store: &B::Buffer,
    base: u64,
    map: &ArenaMap,
    fire: FireRows,
) -> Result<SlotTable> {
    let mut rows: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);

        rows.push(match rect(map, value, fire) {
            Some(rect) => Some(Tensor::new(
                handles.bind(store, base + rect.offset, rect.bytes)?,
                rect.rows,
                rect.width,
                rect.dtype,
            )),
            None => None,
        });
    }
    Ok(SlotTable(rows))
}

#[must_use]
pub fn capacities(map: &ArenaMap) -> Vec<u32> {
    (0..map.placements.len())
        .map(|at| {
            let root = map.root(ValueId(at as u32));
            let Some(poem_compiler::Placement::Arena {
                bytes,
                width,
                dtype,
                ..
            }) = map.placements.get(root.0 as usize)
            else {
                return 0;
            };
            let row = width.saturating_mul(poem_compiler::arena::elem_bytes(*dtype).unwrap_or(0));
            if row == 0 {
                return 0;
            }
            u32::try_from(bytes / row).unwrap_or(u32::MAX)
        })
        .collect()
}
