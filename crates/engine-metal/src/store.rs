//! The model-state bytes: paged kv pages and recurrent slabs, one allocation
//! per cache row, alive for the model's whole load. Owns reservation and the
//! [`KvPool`]/[`RecurrentPool`] rows a cache id resolves to (page/cell
//! arithmetic is [`kv`]'s); also implements [`engine::frame::Supply`].
//!
//! [`Handles`]: crate::device::Handles
//! [`Context`]: crate::device::Context

pub mod accounting;
pub mod kv;

use engine::transfer::KvCopy;
use kernels_metal::{KvPool, RecurrentPool, Tensor};
use model_ir::{CacheRow, Dtype, Trace};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable};
use crate::store::kv::{Facts, Paging};

/// The neutral store's refusals, in this shell's vocabulary.
impl From<model_exec::store::Fault> for Fault {
    fn from(fault: model_exec::store::Fault) -> Fault {
        match fault {
            model_exec::store::Fault::Ceiling { what, need, have } => {
                Fault::Ceiling { what, need, have }
            }
            model_exec::store::Fault::Unbound { what } => Fault::Unbound { what },
            model_exec::store::Fault::Straddled {
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

/// The element the ssm entries hold their recurrent state at. `CacheRow::State`
/// carries no dtype, so each shell states its own; the Metal shaders read
/// `device float*`, unlike the CUDA plane's bf16.
const STATE_DTYPE: Dtype = Dtype::F32;

/// The element one state row lands at on this plane: f32 for any float
/// element (the shaders' own width), but an integer state (e.g. qwen4's
/// n-gram window, holding token ids) is honored as declared.
fn state_dtype(declared: Dtype) -> Dtype {
    match declared {
        Dtype::I32 => Dtype::I32,
        _ => STATE_DTYPE,
    }
}

/// How one cache row is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// A paged kv space: which geometry space it belongs to, and the row the
    /// pages are cut at.
    Kv {
        space: u32,
        head_dim: u32,
        kv_heads: u32,
        dtype: Dtype,
        /// One plane's length in bytes, and therefore the length each plane's
        /// handle is minted at.
        plane_bytes: u64,
        /// Bytes from the front of the allocation to the value pages:
        /// [`plane_bytes`](Shape::Kv::plane_bytes) for the two-plane form,
        /// zero for the one-plane shared form (key and value reader address
        /// the same cells).
        values_at: u64,
    },
    /// A recurrent slab: elements per slot.
    State { stride: u64, dtype: Dtype },
}

/// The per-fire handles a pool row borrows: the geometry vectors this fire
/// wrote, and the graph-padding mask beside them. Rebuilt each fire from
/// long-lived storage and short-lived geometry.
///
/// `page_indptr`/`page_indices` bind into `kernels_metal::KvPool` directly;
/// `last_page_lens`/`row_valid` arrive as `RuntimeInput::Geometry` via
/// [`CacheGeometry`](crate::run::CacheGeometry) and are kept here anyway for
/// a single fill-once, read-twice struct.
#[derive(Debug, Clone, Copy)]
pub struct SpaceSeat {
    /// `i32`, `[lanes + 1]`: this space's page-list bounds.
    pub page_indptr: Tensor,
    /// `i32`: the flat page-id list.
    pub page_indices: Tensor,
    /// `i32`, `[lanes]`: each lane's last-page fill.
    pub last_page_lens: Tensor,
    /// `u8`, `[rows]`: the padding mask the writers read.
    pub row_valid: Tensor,
}

/// What a fire lends the pools.
#[derive(Debug, Clone)]
pub struct Seats {
    /// This fire's lanes.
    pub lanes: u32,
    /// This fire's token rows.
    pub rows: u32,
    /// How many pages its geometry named.
    pub pages: u32,
    /// One seat per kv geometry space.
    pub spaces: Vec<SpaceSeat>,
    /// `i32`, `[lanes]`: which recurrent slot each lane owns. Kept for the
    /// readers that think in lanes; the ssm scans do not.
    pub slot_ids: Tensor,
    /// `i32`, one per token ROW: which recurrent slot that row's lane owns.
    /// The ssm shaders index this by token row (unlike the CUDA sibling,
    /// which indexes by lane); the two coincide for a fire of one lane.
    pub slot_of_row: Tensor,
}

/// One span of kv cells moved inside this load's own pools; the only shape
/// [`Pools::copy_kv`] takes. A whole-page copy and a single-token move are
/// both a run of `tokens` cells starting at `(page, token)`; [`Move::plan`]
/// flattens `KvCopy`'s two spellings into this one shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    /// The page read.
    pub src_page: u32,
    /// The first token slot read in it.
    pub src_token: u32,
    /// The page written.
    pub dst_page: u32,
    /// The first token slot written in it.
    pub dst_token: u32,
    /// How many consecutive token slots move. `page_size` is a whole page.
    pub tokens: u32,
}

impl Move {
    /// The contract's `copy_kv` argument, flattened into runs. Consecutive
    /// per-token moves are coalesced into one blit per plane where the pages
    /// match and the offsets are contiguous.
    ///
    /// # Errors
    ///
    /// Page lists that are not parallel, an offset past the page, or a run
    /// whose two ends overlap.
    pub fn plan(copy: &KvCopy, page_size: u32) -> std::result::Result<Vec<Move>, String> {
        if copy.src_page_ids.len() != copy.dst_page_ids.len() {
            return Err(format!(
                "src_page_ids has {} entries and dst_page_ids {}",
                copy.src_page_ids.len(),
                copy.dst_page_ids.len()
            ));
        }
        let mut moves: Vec<Move> =
            Vec::with_capacity(copy.src_page_ids.len() + copy.moves.len());
        // Whole-page half: every token slot moves, both sides at offset zero.
        for (src, dst) in copy.src_page_ids.iter().zip(&copy.dst_page_ids) {
            moves.push(Move {
                src_page: *src,
                src_token: 0,
                dst_page: *dst,
                dst_token: 0,
                tokens: page_size,
            });
        }
        // Token-granular half, coalesced into runs.
        for (at, cell) in copy.moves.iter().enumerate() {
            if cell.src_token_offset >= page_size || cell.dst_token_offset >= page_size {
                return Err(format!(
                    "kv move {at} names token offsets {}/{} in pages of {page_size} tokens",
                    cell.src_token_offset, cell.dst_token_offset
                ));
            }
            // A cell naming one place twice is dropped, not refused.
            if cell.src_page_id == cell.dst_page_id
                && cell.src_token_offset == cell.dst_token_offset
            {
                continue;
            }
            let run = moves.last_mut().filter(|run| {
                run.src_page == cell.src_page_id
                    && run.dst_page == cell.dst_page_id
                    && run.src_token + run.tokens == cell.src_token_offset
                    && run.dst_token + run.tokens == cell.dst_token_offset
                    && run.src_token + run.tokens < page_size
            });
            match run {
                Some(run) => run.tokens += 1,
                None => moves.push(Move {
                    src_page: cell.src_page_id,
                    src_token: cell.src_token_offset,
                    dst_page: cell.dst_page_id,
                    dst_token: cell.dst_token_offset,
                    tokens: 1,
                }),
            }
        }
        for run in &moves {
            if run.src_page != run.dst_page {
                continue;
            }
            let (lo, hi) = (
                u32::min(run.src_token, run.dst_token),
                u32::max(run.src_token, run.dst_token),
            );
            if hi - lo < run.tokens {
                return Err(format!(
                    "a kv move of {} tokens reads page {} from token {} and writes the same \
                     page at token {} — the two ends overlap, and a blit whose regions \
                     overlap is undefined rather than a shift",
                    run.tokens, run.src_page, run.src_token, run.dst_token
                ));
            }
        }
        Ok(moves)
    }
}

/// Every cache space's bytes, one allocation per row.
#[derive(Debug)]
pub struct Pools {
    slabs: Vec<Buffer>,
    shapes: Vec<Shape>,
    paging: Paging,
    /// The highest demand any admitted frame has stated, per arena. Not a
    /// physical commitment (the reservation is fixed at load); read by
    /// `pool_high_water_bytes` to see whether a load was carved too large.
    watermark: engine::frame::Demand,
}

impl Pools {
    /// Reserve the pools one plan needs at one deployment's budget.
    ///
    /// `facts` is indexed by cache row, not by geometry space: a page id
    /// says which page, never how wide the row it addresses is.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a cache row this shell cannot size — a kv row
    /// no attention op reads, or one whose declared row is not `k|v` by a
    /// head-multiple — [`Fault::Ceiling`] for a pool past `maxBufferLength`,
    /// [`Fault::Device`] when the device declined a length, and
    /// [`Fault::Deviceless`] for a non-Apple build.
    pub fn reserve(
        device: &Context,
        trace: &Trace,
        paging: Paging,
        facts: &Facts,
    ) -> Result<Pools> {
        let mut slabs = Vec::with_capacity(trace.caches.len());
        let mut shapes = Vec::with_capacity(trace.caches.len());

        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    space,
                } => {
                    let (planes, width) = split(name, planes)?;
                    // A restatement (kv_heads != 0) exists only where a paged
                    // launch made one; its absence is not an error, since some
                    // consumers take their widths from their own operands.
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
                                    "cache `{name}`, whose row is {width} wide while its \
                                     consumers state {} heads of {}",
                                    seat.kv_heads, seat.head_dim
                                ),
                            });
                        }
                    }
                    // One head of the whole plane where no head width was stated.
                    let head_dim = restated.map_or(width, |seat| u64::from(seat.head_dim));
                    let kv_heads = restated.map_or(1, |seat| u64::from(seat.kv_heads));
                    let element = elem_bytes(name, *dtype)?;
                    let plane = paging.pages() * u64::from(paging.page_size) * width * element;
                    slabs.push(Buffer::zeroed(device, plane * planes)?);
                    shapes.push(Shape::Kv {
                        space: *space,
                        head_dim: u32::try_from(head_dim).unwrap_or(u32::MAX),
                        kv_heads: u32::try_from(kv_heads).unwrap_or(u32::MAX),
                        dtype: *dtype,
                        plane_bytes: plane,
                        values_at: if planes == 2 { plane } else { 0 },
                    });
                }
                CacheRow::State { name, slab, dtype } => {
                    let stride: u64 = slab.iter().product();
                    let dtype = state_dtype(*dtype);
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, dtype)?;
                    slabs.push(Buffer::zeroed(device, bytes)?);
                    shapes.push(Shape::State { stride, dtype });
                }
            }
            debug_assert_eq!(slabs.len(), index + 1, "one allocation per cache row");
        }
        Ok(Pools {
            slabs,
            shapes,
            paging,
            watermark: engine::frame::Demand::ZERO,
        })
    }

    /// Bytes one recurrent slot occupies across every state row — what the
    /// contract publishes as `PoolFacts::state_slot_bytes`, and what tells the
    /// runtime this model folds a recurrent state at all.
    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|shape| match shape {
                Shape::State { stride, dtype } => stride * u64::from(elem_size(*dtype)),
                Shape::Kv { .. } => 0,
            })
            .sum()
    }

    /// One slot's recurrent banks, read back — every state row's span for
    /// `slot`, in cache-row order. A gate's instrument, not a fire-path verb.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot past the pool.
    pub fn read_slot(&self, slot: u32) -> Result<Vec<u8>> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        let mut out = Vec::new();
        for (slab, shape) in self.slabs.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            let mut span = vec![0u8; usize::try_from(bytes).unwrap_or(0)];
            slab.read(u64::from(slot) * bytes, &mut span)?;
            out.extend_from_slice(&span);
        }
        Ok(out)
    }

    /// Does any cache row carry recurrent state? Lets the fire path skip the
    /// drain-before-clear ordering cost for attention-only plans, which have
    /// nothing to clear.
    #[must_use]
    pub fn has_state(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::State { .. }))
    }

    /// The highest demand admission has committed. See the field.
    #[must_use]
    pub fn watermark(&self) -> engine::frame::Demand {
        self.watermark
    }

    /// How the pages are handed out.
    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// Every byte these pools hold.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slabs.iter().map(Buffer::bytes).sum()
    }

    /// The cache table one fire resolves its cache ids through. The pools'
    /// bytes are load-lived; the views into them are rebuilt each fire.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a kv row whose space this fire seated no
    /// geometry for, [`Fault::Ceiling`] for a plane that leaves its own
    /// reservation or a full handle table.
    pub fn table(&self, handles: &Handles, seats: &Seats) -> Result<CacheTable> {
        let mut rows = Vec::with_capacity(self.shapes.len());
        for (slab, shape) in self.slabs.iter().zip(&self.shapes) {
            rows.push(match *shape {
                Shape::Kv {
                    space,
                    head_dim,
                    kv_heads,
                    dtype,
                    plane_bytes,
                    values_at,
                } => {
                    let seat =
                        seats
                            .spaces
                            .get(space as usize)
                            .ok_or_else(|| Fault::Unbound {
                                what: format!(
                                    "cache space {space}, for which this fire wrote no \
                                     geometry"
                                ),
                            })?;
                    let cells = self.paging.pages() * u64::from(self.paging.page_size);
                    let plane = |at: u64| -> Result<Tensor> {
                        Ok(Tensor::new(
                            handles.bind(slab, at, plane_bytes)?,
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            kv_heads * head_dim,
                            dtype,
                        ))
                    };
                    CachePool::Kv(KvPool {
                        keys: plane(0)?,
                        values: plane(values_at)?,
                        page_indices: seat.page_indices,
                        page_indptr: seat.page_indptr,
                        page_size: narrow(u64::from(self.paging.page_size)),
                        // NHD layout: one token row is `kv_heads * head_dim`
                        // elements, one head plane is `head_dim`.
                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                    })
                }
                Shape::State { stride, dtype } => {
                    let bytes =
                        stride * u64::from(self.paging.slots) * u64::from(elem_size(dtype));
                    // One handle, read three times: `CacheRow::State` is one
                    // slab, and `new_conv_state` aliases it because the
                    // rolling update is in place.
                    let bank = Tensor::new(
                        handles.bind(slab, 0, bytes)?,
                        self.paging.slots,
                        u32::try_from(stride).unwrap_or(u32::MAX),
                        dtype,
                    );
                    CachePool::Recurrent(RecurrentPool {
                        state: bank,
                        slots: seats.slot_of_row,
                        conv_state: bank,
                        new_conv_state: bank,
                    })
                }
            });
        }
        Ok(CacheTable(rows))
    }

    /// Clear one slot's recurrent state. Needed because a recurrent slot is
    /// its history: opening a sequence in a slot another one used must zero
    /// what that one left (unlike a kv page, overwritten before it is read).
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot past the pool or a span past a slab,
    /// [`Fault::Deviceless`] for a non-Apple build.
    pub fn clear(&mut self, slot: u32) -> Result<()> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        for (slab, shape) in self.slabs.iter_mut().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            slab.zero_span(u64::from(slot) * bytes, bytes)?;
        }
        Ok(())
    }

    /// Copy kv cells between pages of these pools, into `frame`'s command
    /// buffer. The device half of a prefix-tree fork: a shared page run is
    /// grafted onto fresh ids. Loops over every plane of every row, since a
    /// page id names all of them.
    ///
    /// Encoded as a blit rather than a host `Buffer::write`: a host store
    /// isn't ordered against a command buffer already queued, so encoding is
    /// what lets the copy inherit queue order without a drain.
    ///
    /// Two passes: the first checks every span fits a page and commits the
    /// frame's demand; the second encodes. A refused plan leaves an empty,
    /// uncommitted command buffer.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a run past a page's tokens or a page past the
    /// pool, [`Fault::Device`] when the command buffer would not open a blit
    /// pass, [`Fault::Deviceless`] off Apple. An overlapping move is refused
    /// earlier, by [`Move::plan`].
    pub fn copy_kv(&mut self, frame: &mut Frame, moves: &[Move]) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }
        let page_size = u64::from(self.paging.page_size);
        let mut highest = 0u32;
        for span in moves {
            for (page, token) in [
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
                highest = highest.max(page.saturating_add(1));
            }
        }
        // Both ends have to be admitted; a fork names its destination page
        // before any frame's demand has covered it.
        engine::frame::Supply::commit(
            self,
            engine::frame::Demand {
                kv_pages: highest,
                state_slots: 0,
                workspace: 0,
            },
        )?;
        for (slab, shape) in self.slabs.iter().zip(&self.shapes) {
            let Shape::Kv {
                head_dim,
                kv_heads,
                dtype,
                values_at,
                ..
            } = *shape
            else {
                continue;
            };
            let cell =
                u64::from(kv_heads) * u64::from(head_dim) * u64::from(elem_size(dtype));
            // A shared row has one plane base: copying it twice would be a
            // self-overlapping blit (device fault).
            let bases = [0, values_at];
            let bases = if values_at == 0 { &bases[..1] } else { &bases[..] };
            for &plane in bases {
                for span in moves {
                    if span.tokens == 0 {
                        continue;
                    }
                    let bytes = u64::from(span.tokens) * cell;
                    let at = |page: u32, token: u32| {
                        plane + (u64::from(page) * page_size + u64::from(token)) * cell
                    };
                    let (src, dst) = (
                        at(span.src_page, span.src_token),
                        at(span.dst_page, span.dst_token),
                    );
                    if src == dst {
                        continue;
                    }
                    // The blit itself has no bounds check; a copy past a
                    // reservation is a device fault.
                    slab.span(src, bytes)?;
                    slab.span(dst, bytes)?;
                    frame.copy(slab.slab(), src, slab.slab(), dst, bytes)?;
                }
            }
        }
        Ok(())
    }
}

/// The engine's half of memory, on a plane whose reservation is fixed.
/// `commit` is a ceiling check raised before any command buffer opens, so a
/// refused frame leaves nothing to undo.
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
        let pages = self.paging.pages();
        if u64::from(demand.kv_pages) > pages {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(demand.kv_pages),
                have: pages,
            });
        }
        // This plane grants no per-fire workspace, so any nonzero demand is refused.
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

    /// Nothing is unmapped (a fixed reservation has no tail to give back);
    /// the hint only stops the watermark from being a stale ratchet.
    fn trim(&mut self, hint: engine::frame::Demand) {
        self.watermark = engine::frame::Demand {
            kv_pages: self.watermark.kv_pages.min(hint.kv_pages),
            state_slots: self.watermark.state_slots.min(hint.state_slots),
            workspace: self.watermark.workspace.min(hint.workspace),
        };
    }
}

/// The kv pool's resident bytes, off the trace's cache rows and paging
/// alone, read before a byte is reserved. Matches [`Pools::reserve`]'s
/// arithmetic without needing a device or `Facts`.
///
/// # Errors
///
/// [`Fault::Unbound`] for a cache row whose planes this shell cannot cut or
/// whose element has no byte size — the same two refusals `reserve` raises.
pub fn pool_demand(trace: &Trace, paging: Paging) -> Result<u64> {
    let mut bytes: u64 = 0;
    for row in &trace.caches {
        match row {
            CacheRow::Kv {
                name,
                planes,
                dtype,
                ..
            } => {
                let (count, width) = split(name, planes)?;
                let element = elem_bytes(name, *dtype)?;
                let plane = paging.pages() * u64::from(paging.page_size) * width * element;
                bytes = bytes.saturating_add(plane.saturating_mul(count));
            }
            CacheRow::State { name, slab, dtype } => {
                let stride: u64 = slab.iter().product();
                let dtype = state_dtype(*dtype);
                bytes = bytes.saturating_add(
                    stride
                        .saturating_mul(u64::from(paging.slots))
                        .saturating_mul(elem_bytes(name, dtype)?),
                );
            }
        }
    }
    Ok(bytes)
}

/// A kv row's `(planes, width)`: how many planes one entry is written as, and
/// how wide one of them is. Two forms served: `[w, w]` (key/value halves)
/// and `[w]` (MLA's shared latent, read as both k and v). `[k, v]` at
/// different widths is refused.
fn split(name: &str, planes: &[u64]) -> Result<(u64, u64)> {
    match planes {
        [shared] => Ok((1, *shared)),
        [keys, values] if keys == values => Ok((2, *keys)),
        [keys, values] => Err(Fault::Unbound {
            what: format!(
                "cache `{name}`, whose planes are {keys} and {values} wide — this \
                 shell cuts kv pages into equal key and value halves"
            ),
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
    model_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Unbound {
        what: format!("cache `{name}`, stored as {dtype:?}, which has no element size"),
    })
}

fn elem_size(dtype: Dtype) -> u32 {
    model_compiler::arena::elem_bytes(dtype).unwrap_or(1) as u32
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}
