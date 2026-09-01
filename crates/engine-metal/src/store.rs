//! The model-state bytes: paged kv pages and recurrent slabs, one allocation
//! per cache row, alive for the model's whole load.
//!
//! **THE GEOMETRY IS NOT HERE.** Where a lane's pages are and which cell a
//! token lands in is [`kv`]'s arithmetic, backend-neutral and host-tested;
//! this module owns only what that arithmetic cannot: the reservation, and
//! the [`KvPool`]/[`RecurrentPool`] rows the dispatch arms resolve a cache id
//! to. The split is design §6's `model_exec::store` / shell `store/` line, drawn
//! here ahead of the module that will hold the first half — and [`kv`] itself
//! is the same file on both shells, which is the cheapest possible proof that
//! the line is in the right place.
//!
//! # What the Metal plane changes, and it is one thing
//!
//! **A POOL ROW NAMES A HANDLE, NOT AN ADDRESS.** The CUDA sibling builds its
//! pool views as `slab.ptr() + at`; a `kernels_metal::Tensor` carries a `u32`
//! row into [`Handles`], because a compute encoder binds a BUFFER and an
//! OFFSET and there is no pointer to add to. So [`Pools::table`] takes the
//! handle table and MINTS — one row per plane it names — where its twin only
//! added, and [`Pools::reserve`] takes the [`Context`] because a Metal
//! reservation is a call on a device rather than a read of thread-local
//! state. Everything else in this file is the sibling's argument unchanged.
//!
//! A second, quieter difference: `zero_span` on this platform is a `memset`
//! through a shared mapping, not a `cudaMemsetAsync` on a stream. That makes
//! [`Pools::clear`] synchronous and streamless, and it makes the ORDER — the
//! clear before the command buffer that reads the slot is committed — a
//! property of the call site rather than of a stream token.
//!
//! **AND UNDER RUN-AHEAD THAT ORDER IS NOT FREE.** A `memset` is not ordered
//! against a command buffer already on the queue, so a fire path with steps in
//! flight has to DRAIN before it clears a slot one of them may be reading.
//! [`Pools::has_state`] is what keeps that from being a tax on every prefill:
//! only a plan with a `CacheRow::State` has anything here to zero, and every
//! attention-only artifact this shell serves answers no.
//!
//! # And the other half of design §8: admission
//!
//! This module implements [`engine::frame::Supply`]. The reservation is fixed
//! — carved at the deployment's ceiling at load, never grown — so `commit` is
//! a ceiling check at the RIGHT INSTANT rather than a physical mapping, and
//! `trim` moves a watermark rather than unmapping a page. Both readings
//! satisfy the contract, and the difference is visible as
//! `PoolFacts::elastic_page_bytes`, which this plane answers zero.
//!
//! # Sizes come off the plan, not off a config
//!
//! `CacheRow::Kv { planes, dtype }` names the planes one entry is written as,
//! at their own widths, and its element, and
//! `CacheRow::State { slab }` states a per-lane bank — so a pool's bytes are
//! the plan's declaration times the deployment's budget, and there is no
//! second place where a head count could disagree with the model text. The
//! one fact the IR genuinely does not carry is the recurrent element: the ssm
//! entries instantiate their state at bf16, so the slabs are bf16 and this
//! file is where that is written down.
//!
//! # k and v share a row and not a buffer
//!
//! `row[0]` is the k|v plane count and the rest is the row proper. The append
//! kernel writes two independent page arrays, so one allocation per cache row
//! is cut in half: keys in the front, values behind them — **unless the row
//! declares ONE plane**, which is the MLA family's shared latent and is cut
//! nowhere: the value reader addresses the key cells because there is one
//! entry per token and both readers want it ([`split`]). Cutting it here
//! rather than allocating twice keeps a layer's kv contiguous, which is what
//! the page addressing assumes — and on this plane it buys one thing more,
//! since two handles into one buffer cost two table rows where two buffers
//! would cost two reservations against `maxBufferLength`.
//!
//! # The layout is spelled by the strides, because there is nothing else
//!
//! The CUDA sibling's `KvPool` carries a `layout` enumerator beside its
//! strides and this plane's does not — `kernels_metal::KvPool` is storage
//! plus `page_size`/`seq_stride`/`head_stride` and stops there (design §7:
//! all geometry arrives as declared inputs). So NHD — `[page][token][head]
//! [dim]` — is stated ONLY by the stride pair this module writes:
//! `seq_stride` is one whole token row (`kv_heads * head_dim`) and
//! `head_stride` is one head plane (`head_dim`). Under HND the two would swap
//! roles. There is no enumerator here to disagree with them, which removes
//! the CUDA sibling's cross-check and also the thing it was checking; what
//! replaces it is that the pair is written once, in one expression, from the
//! same `SpaceFacts` the pool was sized at.
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
///
/// **THE CONDITION IS SHARED AND THE SENTENCE IS NOT.** `model_exec::store` owns
/// the arithmetic that decides a lane overran its block or a value's width is
/// symbolic; each shell owns how that reads to somebody holding a stack trace
/// ("the load reserved" here, "the shell reserved" on the CUDA plane). This
/// is the one place the two meet, and it is a variant-for-variant map because
/// both shells already carried these three under these names.
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

/// The element the ssm entries hold their recurrent state at.
///
/// Stated, not declared: `CacheRow::State` carries a slab shape and no dtype
/// (a standing open item of the IR), so each shell writes down what its own
/// entries read.
///
/// **AND THE TWO SHELLS DISAGREE, WHICH IS WHY THIS CONSTANT IS NOT A COPY.**
/// `kernels/attn/ssm.cuh` instantiates its state at `__nv_bfloat16`; every
/// state-taking Metal shader declares it `device float*` —
/// `ssm_gated_delta.metal`'s `rstate`, `ssm_causal_conv1d.metal`'s
/// `conv_state`/`new_conv_state`, `ssm_kda.metal`'s accumulator — and
/// `kernels_metal::RecurrentPool`'s own field docs say `f32` twice. The two
/// planes made different accuracy calls about the same bank and both are
/// entitled to.
///
/// **MEASURED, BECAUSE THE WRONG ANSWER HERE IS NOT A CRASH.** This file
/// first carried the CUDA constant, and the pool came out at HALF the bytes
/// the shaders address: every slot's bank overlapped the next one's, and
/// `clear` zeroed half of a slot's span. A slot's FIRST sequence was correct
/// (the whole reservation was zeroed at load) and its second was fluent
/// garbage built out of the first one's history, with the damage varying by
/// which other slot had fired in between — the exact shape of palo build log
/// 19's leak, arriving through a different door. `serve_smoke`'s
/// determinism and launch-isolation gates are what caught it.
const STATE_DTYPE: Dtype = Dtype::F32;

/// The element one state row lands at ON THIS PLANE. The shaders instantiate
/// their recurrent scans at f32 whatever float element the row declares —
/// the CUDA plane narrows the same declarations to bf16, and each shell's
/// kernels read what its own store allocates — but an INTEGER state is
/// semantic (qwen4's n-gram window holds token ids) and is honored as
/// declared.
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
        /// Bytes from the front of the allocation to the VALUE pages.
        ///
        /// [`plane_bytes`](Shape::Kv::plane_bytes) for the two-plane form —
        /// keys in front, values behind them — and **ZERO for the one-plane
        /// shared form**, where the value reader and the key reader address
        /// the same cells because there is only one entry per token to
        /// address. The two numbers were one field while two planes was the
        /// only form, and a shared row is exactly where they part.
        values_at: u64,
    },
    /// A recurrent slab: elements per slot.
    State { stride: u64, dtype: Dtype },
}

/// The per-fire handles a pool row borrows: the geometry vectors this fire
/// wrote, and the graph-padding mask beside them.
///
/// A `KvPool` is storage plus the tables that address it, and the tables are
/// fire data — so the [`CacheTable`] is rebuilt each fire out of long-lived
/// storage and short-lived geometry. Rebuilding 42 rows of `Copy` structs is
/// arithmetic, not allocation.
///
/// **TWO OF THESE FOUR REACH THE SHADERS ELSEWHERE ON THIS PLANE.**
/// `page_indptr` and `page_indices` are `kernels_metal::KvPool`'s own fields
/// and are bound from here. `last_page_lens` and `row_valid` are not: design
/// §7 moved every other geometry vector onto the ops that read it, so they
/// arrive as `RuntimeInput::Geometry` through
/// [`CacheGeometry`](crate::run::CacheGeometry) and are resolved by
/// `Run::tensor`. They are kept in this seat anyway, because a seat is what
/// the fire path HAS after `kv::geometry` and staging it — one struct the
/// shell fills once and reads twice is a simpler thing than two, and the day
/// an entry wants them back in the pool there is no new plumbing to invent.
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
    ///
    /// **THIS IS THE ONE THE SCANS READ**, and it is not the CUDA sibling's
    /// vector. Every metal ssm shader indexes `slots[r]` at a TOKEN ROW —
    /// the decode arms at the grid row, the chunked arms at the request's
    /// first row (`slots[indptr[r]]`) — where the CUDA entries index a lane.
    /// The two coincide for a fire of one lane, which is why the difference
    /// is invisible to every solo gate.
    pub slot_of_row: Tensor,
}

/// **One span of kv cells moved inside this load's own pools** — the shape
/// [`Pools::copy_kv`] takes, and the only one it takes.
///
/// **A WHOLE PAGE AND A SINGLE TOKEN ARE THE SAME MOVE.** The contract states
/// them apart — `KvCopy::src_page_ids`/`dst_page_ids` are the whole-page half
/// and `KvCopy::moves` the token-granular one — because a caller spells a
/// prefix graft and a fork's partial tail differently. What the page
/// arithmetic underneath sees is one thing either way: a run of `tokens` cells
/// starting at `(page, token)` on each side. So the two spellings are
/// flattened once, by [`Move::plan`], and the blit loop has one shape to walk.
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
    /// **The contract's `copy_kv` argument, flattened into runs.**
    ///
    /// Host arithmetic over two integers and two lists, which is why it lives
    /// beside the page arithmetic rather than in [`crate::api`] where its
    /// caller is: what a fork's moves come to is the same question with or
    /// without a device, and a gate can ask it without one.
    ///
    /// # Consecutive cells are COALESCED
    ///
    /// A fork copying a partial page's live tokens states one `KvMove` per
    /// TOKEN, and the run they form is one blit per plane rather than one per
    /// token per plane. The merge is exact — identical pages on both sides,
    /// both offsets continuing the previous run by one, and the run stopping
    /// at the page's own edge — and it is order-preserving, so a caller whose
    /// moves do not form runs gets the same bytes at more copies and never
    /// different bytes.
    ///
    /// # OVERLAPPING ENDS ARE THE CALLER'S ERROR AND ARE NAMED HERE
    ///
    /// Both ends of a run live in the same reservation, so a run that reads
    /// and writes overlapping cells of one page is a blit whose two regions
    /// overlap — undefined, and silently so. A caller that means "shift a
    /// page's tokens" states a staging page and two moves.
    ///
    /// # Errors
    ///
    /// The sentence a malformed plan is refused with, which
    /// [`Metal::copy_kv`](crate::Metal) hands back as `Error::Invalid`: page
    /// lists that are not parallel, an offset past the page, or a run whose
    /// two ends overlap.
    pub fn plan(copy: &KvCopy, page_size: u32) -> std::result::Result<Vec<Move>, String> {
        // The contract's own clause (`KvCopy::validate`), restated where the
        // arithmetic is. A belt and not a second policy: the verb has already
        // asked it, and a `zip` that silently dropped the longer list would
        // graft a fork's pages half over.
        if copy.src_page_ids.len() != copy.dst_page_ids.len() {
            return Err(format!(
                "src_page_ids has {} entries and dst_page_ids {}",
                copy.src_page_ids.len(),
                copy.dst_page_ids.len()
            ));
        }
        let mut moves: Vec<Move> =
            Vec::with_capacity(copy.src_page_ids.len() + copy.moves.len());
        // The whole-page half: every token slot of the page, both sides at
        // offset zero. A page's LIVE length is the runtime's bookkeeping and
        // not a number this verb is handed, so the whole page moves.
        for (src, dst) in copy.src_page_ids.iter().zip(&copy.dst_page_ids) {
            moves.push(Move {
                src_page: *src,
                src_token: 0,
                dst_page: *dst,
                dst_token: 0,
                tokens: page_size,
            });
        }
        // The token-granular half, coalesced into runs.
        for (at, cell) in copy.moves.iter().enumerate() {
            if cell.src_token_offset >= page_size || cell.dst_token_offset >= page_size {
                return Err(format!(
                    "kv move {at} names token offsets {}/{} in pages of {page_size} tokens",
                    cell.src_token_offset, cell.dst_token_offset
                ));
            }
            // A cell that names one place twice is not a move. Dropped rather
            // than refused: a caller listing a fork's whole tail states the
            // shared cells too, and it is asking for nothing.
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
    /// **The highest demand any admitted frame has stated**, per arena.
    ///
    /// Not a physical commitment — this plane reserves at the load's ceiling
    /// and never grows (see the [`Supply`](engine::frame::Supply) impl) — so
    /// this is the OBSERVABLE that says admission is doing arithmetic rather
    /// than nodding: `pool_high_water_bytes` is read off it, and a load that
    /// never rose above a fraction of its reservation can be seen to have
    /// been carved too large.
    watermark: engine::frame::Demand,
}

impl Pools {
    /// Reserve the pools one plan needs at one deployment's budget.
    ///
    /// `device` is the reservation's — `Buffer::zeroed` is
    /// `newBufferWithLength:options:` on THIS device, where the CUDA twin's
    /// `cudaMalloc` reads the thread's current device out of ambient state.
    ///
    /// `facts` is indexed by CACHE ROW, not by geometry space: a page id says
    /// which page, never how wide the row it addresses is, and gemma's
    /// sliding and global layers share one page-id space at two widths
    /// ([`SpaceFacts`](crate::store::kv::SpaceFacts)).
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
                    // **A RESTATEMENT EXISTS ONLY WHERE A PAGED LAUNCH MADE
                    // ONE**, and where it exists it must be the declaration.
                    // The prefill arms state a head count; the decode and
                    // masked arms state a head width alone (`kv_heads` 0); and
                    // the latent, index and pool launches do not feed the row
                    // pass at all (`model_exec::store::kv::SpaceFacts`). So a
                    // row with no restatement is not an error and never was —
                    // it is a row whose consumers take their widths from their
                    // own operands — and `Pools::reserve` refusing one by name
                    // was this shell asking a question the IR had already
                    // answered "absent". The CUDA sibling has always read it
                    // this way (`engine_cuda::store::Pools::reserve`).
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
                    // One head of the whole plane where no paged launch stated
                    // a head width: the latent, index and pool kernels take
                    // their widths from their op operands and never consult
                    // these strides.
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
                        // One plane means the value reader addresses the key
                        // cells; two means the values begin one plane in.
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

    /// **Does any cache row carry recurrent state?**
    ///
    /// Asked by the fire path for one reason and it is an ordering one: a
    /// sequence beginning in a slot another sequence used has to have that
    /// slot's banks zeroed, [`Pools::clear`] does it with a host `memset`
    /// through the shared mapping, and a `memset` is not ordered against a
    /// command buffer the device is still reading. A plan with no `State`
    /// row has nothing to clear and therefore nothing to order, which is
    /// every attention-only artifact this shell serves — so the run-ahead
    /// never pays for the exception it does not have.
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

    /// The cache table one fire resolves its cache ids through.
    ///
    /// `handles` is the fire's minting table. Every plane named here is one
    /// row in it, dropped with the rest of the fire's rows at
    /// `Handles::rewind` — the pools' BYTES are load-lived, but the views of
    /// them are not, because a view is a table row and the table is rebuilt
    /// each fire out of geometry that is also this fire's.
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
                    // Where a plane starts and how long it is are two numbers
                    // now: a shared row's value plane starts at zero and is
                    // still one whole plane long.
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
                        // NHD: one token row is `kv_heads * head_dim`
                        // elements and one head plane is `head_dim`. On this
                        // plane the pair is the ONLY statement of the layout
                        // — there is no enumerator beside it — so it is
                        // written from the facts the pool was sized at and
                        // nowhere else.
                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                    })
                }
                Shape::State { stride, dtype } => {
                    let bytes =
                        stride * u64::from(self.paging.slots) * u64::from(elem_size(dtype));
                    // ONE HANDLE, READ THREE TIMES. A `CacheRow::State` is
                    // one slab and the ops that read it name it once — the
                    // gated-delta scan through the state bank, the causal
                    // convolution through the conv bank — so pointing every
                    // seat at this row's bytes is what makes `conv.L` and
                    // `delta.L` two independent spaces rather than two halves
                    // of one. `new_conv_state` joins them because the rolling
                    // update is in place: this plane has no second slab to
                    // write into, exactly as the CUDA sibling's single
                    // `conv_slab` had none.
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

    /// Clear one slot's recurrent state.
    ///
    /// A RECURRENT SLOT IS ITS HISTORY, so opening a sequence in a slot
    /// another sequence used means zeroing what that one left — unlike a kv
    /// page, which is overwritten before it is read because `kv_len` says
    /// nothing lives past the append. There is no cheaper truth available:
    /// the scan reads the whole bank on its first step.
    ///
    /// Called from the shell's `open` by a caller whose page table is the
    /// shell's, and from the fire path for one whose page table is its own —
    /// there, a lane stating `held == 0` IS the sequence beginning. One fill
    /// per sequence either way; on this platform that fill is a `memset`
    /// through the shared mapping rather than a `cudaMemsetAsync`, so it is
    /// done when it returns and the ordering it needs is only that it happen
    /// before the fire's command buffer is committed.
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

    /// **Copy kv cells between pages of these pools, into `frame`'s command
    /// buffer** (alto survey §9's gap list; `engine_cuda::store::Pools::copy_kv`
    /// is the sibling).
    ///
    /// The device half of a prefix-tree fork: a page run two sequences share is
    /// grafted onto fresh ids, and the partial page at the boundary has its
    /// live tokens copied out so the fork can append past them without writing
    /// into the parent's cells.
    ///
    /// # Every plane of every row, because a page id names all of them
    ///
    /// A "page" is not one allocation. Page `p` exists once per PLANE of every
    /// `CacheRow::Kv` this plan declares — eighteen layers times a key half and
    /// a value half, for a dense text model — and a mover that copied a subset
    /// would leave a fork attending to the parent's keys at some layers and its
    /// own at others, which reads as fluent garbage rather than as an error. So
    /// the loop is over `rows × planes` and the caller names the page once,
    /// exactly as [`Pools::clear`] takes a slot once for every recurrent row
    /// underneath it.
    ///
    /// **THE TWO PLANES ARE ONE RESERVATION HERE**, which is this plane's own
    /// difference: the CUDA sibling walks an arena per plane, and this module
    /// cuts one allocation into a key half at `0` and a value half at
    /// `values_at` (the module header's "k and v share a row and not a
    /// buffer"). So the two plane bases are the loop and `values_at` is the
    /// second of them — and both halves are one width, because [`split`]
    /// admits no other form.
    ///
    /// # A BLIT AND NOT A `memcpy`, on a plane where a `memcpy` would work
    ///
    /// The pools are `StorageModeShared`, so the host could write these bytes
    /// with `Buffer::write` and no encoder at all. What it could not do is
    /// ORDER them: a host store is not ordered against a command buffer already
    /// on the queue, so a graft written that way would land in cells a step
    /// still in flight is reading, and the only fix would be a drain per fork
    /// (which is what [`Pools::clear`] pays, and why it is only paid by plans
    /// that have recurrent banks at all). Encoded into a command buffer the
    /// copies inherit the queue's own order instead: behind every step already
    /// committed, ahead of every step committed after — which is exactly the
    /// ordering a caller that forks and then fires against both halves is
    /// asking for, and it is article 2's no-sync-on-a-fire-path for free.
    ///
    /// # Refusals before bytes
    ///
    /// Two passes, and the split is article 4's: the first asks every span
    /// whether it fits a page and commits the frame's own demand for the pages
    /// named ([`Supply::commit`](engine::frame::Supply) is the admission gate,
    /// and a fork names a destination page before any fire has admitted one).
    /// The second encodes. So a refused plan leaves an empty command buffer,
    /// which is dropped without commit and puts nothing on the device.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a run past a page's tokens or a page past the
    /// pool, [`Fault::Device`] when the command buffer would not open a blit
    /// pass, [`Fault::Deviceless`] off Apple. A move whose two ends OVERLAP is
    /// refused earlier, by [`Move::plan`], where the contract's `Invalid` can
    /// be spoken: it is the caller's statement that is wrong and not this
    /// pool's arithmetic.
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
        // **BOTH ENDS HAVE TO BE ADMITTED, AND THIS IS THE GATE.** A fork is a
        // control-plane verb: it names the destination page before any frame's
        // demand has covered it, so the pages it addresses are stated here,
        // through the same `Supply` the fire path states its union through. On
        // this plane that is a ceiling check and a watermark union rather than
        // a mapping (the impl's own note), which is why a page past the pool
        // comes back as `Impossible` rather than as a blit into nothing.
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
            // **THE DISTINCT PLANE BASES, AND A SHARED ROW HAS ONE.** Copying
            // a shared latent twice would be the same bytes to the same
            // address — a self-overlapping blit, which is a device fault with
            // no sentence in it and not a redundant copy.
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
                    // The one bounds check the blit itself has none of: a
                    // `copyFromBuffer:` past the end of a reservation is a
                    // device fault with no sentence in it, and `Buffer::span`
                    // names both numbers.
                    slab.span(src, bytes)?;
                    slab.span(dst, bytes)?;
                    frame.copy(slab.slab(), src, slab.slab(), dst, bytes)?;
                }
            }
        }
        Ok(())
    }
}

/// **The engine's half of memory, on a plane whose reservation is fixed**
/// (alto design §8; article 8: the runtime owns policy, the engine owns
/// supply).
///
/// [`Pools::reserve`] carves every cache row at the deployment's ceiling at
/// LOAD, so there is no physical growth for admission to drive and
/// `PoolFacts::elastic_page_bytes` answers zero. The contract says both
/// readings satisfy it — *"a shell that has not been converted still carves
/// fixed pools at load, and for it `commit` is the ceiling check it already
/// had and `trim` has nothing to give back"* — and this is that shell.
///
/// **THE POINT OF ASKING ANYWAY IS THE INSTANT, NOT THE BYTES** (article 4).
/// The two refusals below are the ones `kv::geometry_with` already raises, to
/// the variant and to the `what` string; what admission buys is that they are
/// raised BEFORE any command buffer is opened, on a value the frame's steps
/// have taken the union of, so an `Impossible` frame leaves nothing behind to
/// undo. Past this call the fire is success-only.
///
/// The day this plane grows an `MTLHeap` under the pools, `commit` is where
/// the growth goes and nothing above it changes — which is the whole reason
/// the question is asked here rather than in the page arithmetic.
impl engine::frame::Supply for Pools {
    type Error = Fault;

    fn commit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        // Only the slots this shell PAGES are its supply. A lane that brought
        // its own page table brought its own addressing with it (article 8:
        // engine page ids are the runtime's, the shell's paging is sizing) —
        // but a page id is still a page id, and the demand this is handed is
        // the highest one the frame will address plus one, whoever minted it.
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
        // **THE THIRD ARENA IS EMPTY AND IS STILL ASKED ABOUT.** This plane
        // grants no per-fire workspace — `kernels-metal`'s plan builders are
        // pure carriers with no schedule and no split-kv partials to hold
        // (`inputs.rs`'s opening note) — so every `Prepared` here states zero
        // and this arm never fires. It is written rather than ignored because
        // a `Demand` arriving with bytes in it would mean a builder had grown
        // one, and the honest answer to that is a named ceiling rather than
        // silence.
        if demand.workspace > 0 {
            return Err(Fault::Ceiling {
                what: "pool workspace bytes",
                need: demand.workspace,
                have: 0,
            });
        }
        // Past the refusals, and only past them: nothing above this line
        // wrote anything, which is article 4's zero side effects.
        self.watermark = self.watermark.union(demand);
        Ok(())
    }

    /// **Nothing is unmapped, and the hint is still recorded.**
    ///
    /// A fixed reservation has no tail to give back — the bytes were taken
    /// from the device at load and are held until the load ends — so this
    /// method cannot do the thing its name promises on the elastic plane.
    /// What it can do is stop the watermark from being a ratchet: a caller
    /// that states a residency below what has been committed is telling this
    /// pool that the high water it is reporting is stale, and reporting a
    /// stale one would make a load look tighter than it is.
    ///
    /// **THE HINT IS A RESIDENCY STATEMENT AND ITS TRUTH IS THE CALLER'S**
    /// (the trait's own note). Nothing here invents a watermark of its own,
    /// and nothing here refuses one either: the cost of being wrong on this
    /// plane is one wrong number in a footprint line, not a page somebody's
    /// prefix was living in.
    fn trim(&mut self, hint: engine::frame::Demand) {
        self.watermark = engine::frame::Demand {
            kv_pages: self.watermark.kv_pages.min(hint.kv_pages),
            state_slots: self.watermark.state_slots.min(hint.state_slots),
            workspace: self.watermark.workspace.min(hint.workspace),
        };
    }
}

/// **The kv pool's resident bytes**, off the trace's cache rows and the paging
/// alone — the `minimum` term of [`accounting::Accounting`], read BEFORE a byte
/// is reserved so the wired ceiling can be checked ahead of the land.
///
/// Byte for byte [`Pools::reserve`]'s arithmetic, without the device and
/// without the compiled class table: a kv row's width is the plane `split`
/// reads off the declared planes, not the seat's head count, so this needs
/// neither the bound device nor `Facts`. What `reserve` does on top — the
/// head-multiple check — is validation, not sizing, and refuses there instead.
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
/// how wide one of them is.
///
/// **TWO FORMS, AND THE SECOND IS NOT A DEGENERATE FIRST.** M22 turned the
/// row's leading plane count into `CacheRow::Kv { planes }` — the per-plane
/// widths themselves — and the IR spells three readings of it (see
/// `model_ir::CacheRow`). This shell serves two:
///
/// * `[w, w]`, a key half and a value half at one width, cut out of one
///   allocation (the module header's "k and v share a row and not a buffer").
/// * `[w]`, **ONE PLANE READ AS BOTH k AND v**. This is not a two-plane row
///   with the value half missing; it is the MLA family's latent, where there
///   IS no second tensor — `attention.kv_append_shared` writes one entry per
///   token and the absorbed reader contracts against those same cells twice,
///   once as a key and once as a value. Allocating a second plane for it would
///   double a 43-layer model's kv pool to hold a copy nothing writes and
///   nothing reads.
///
/// `[k, v]` at DIFFERENT widths — the split latent page, `[kv_lora_rank,
/// rope_dim]` — is the third reading and is still refused by name: the page
/// addressing below cuts one allocation at one width, and two widths is a
/// different cut and not a different number.
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
