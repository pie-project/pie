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
//! is cut in half: keys in the front, values behind them. Cutting it here
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

pub mod kv;

use kernels_metal::{KvPool, RecurrentPool, Tensor};
use model_ir::{CacheRow, Dtype, Trace};

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
        /// Bytes from the front of the allocation to the value pages — which
        /// is also one whole plane's length, and therefore the length each
        /// plane's handle is minted at.
        values_at: u64,
    },
    /// A recurrent slab: elements per slot.
    State { stride: u64 },
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
                    let seat = facts.row(index, name)?;
                    let (planes, width) = split(name, planes)?;
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
                    let element = elem_bytes(name, *dtype)?;
                    let plane = paging.pages() * u64::from(paging.page_size) * width * element;
                    slabs.push(Buffer::zeroed(device, plane * planes)?);
                    shapes.push(Shape::Kv {
                        space: *space,
                        head_dim: seat.head_dim,
                        kv_heads: seat.kv_heads,
                        dtype: *dtype,
                        values_at: plane,
                    });
                }
                CacheRow::State { name, slab } => {
                    let stride: u64 = slab.iter().product();
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, STATE_DTYPE)?;
                    slabs.push(Buffer::zeroed(device, bytes)?);
                    shapes.push(Shape::State { stride });
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
                    // `values_at` is where the value pages start AND how long
                    // one plane is, so it is both arguments of the mint.
                    let plane = |at: u64| -> Result<Tensor> {
                        Ok(Tensor::new(
                            handles.bind(slab, at, values_at)?,
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
                Shape::State { stride } => {
                    let bytes =
                        stride * u64::from(self.paging.slots) * u64::from(elem_size(STATE_DTYPE));
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
                        STATE_DTYPE,
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
        let element = u64::from(elem_size(STATE_DTYPE));
        for (slab, shape) in self.slabs.iter_mut().zip(&self.shapes) {
            let Shape::State { stride } = *shape else {
                continue;
            };
            let bytes = stride * element;
            slab.zero_span(u64::from(slot) * bytes, bytes)?;
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

/// A kv row's `(planes, width)`: the leading dim is the k|v plane count and
/// the rest is one plane's row.
fn split(name: &str, planes: &[u64]) -> Result<(u64, u64)> {
    // M22 turned the row's leading plane count into `CacheRow::Kv { planes }`
    // — the per-plane widths themselves. This shell cuts a kv page into an
    // equal key half and value half, so it serves exactly the two-plane form
    // at one width and refuses the rest by name, as the old dims reading did.
    match planes {
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
                 pages into a key half and a value half, and knows no other form",
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
