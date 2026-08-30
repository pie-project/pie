//! The model-state bytes: paged kv pages and recurrent slabs, one ELASTIC
//! ARENA per cache-row plane, alive for the model's whole load.
//!
//! **THE GEOMETRY IS NOT HERE.** Where a lane's pages are and which cell a
//! token lands in is [`kv`]'s arithmetic, backend-neutral and host-tested;
//! this module owns only what that arithmetic cannot: `cudaMalloc`, and the
//! [`KvPool`]/[`RecurrentPool`] rows the dispatch arms resolve a cache id to.
//! The split is design §6's `model_exec::store` / shell `store/` line, drawn here
//! ahead of the module that will hold the first half.
//!
//! # Sizes come off the plan, not off a config
//!
//! `CacheRow::Kv { planes, dtype }` states the planes one token's entry is
//! written as, each by its own per-token width, and `CacheRow::State { slab }`
//! states a per-lane bank — so a pool's bytes are the plan's declaration times
//! the deployment's budget, and there is no second place where a head count
//! could disagree with the model text. The one fact the IR genuinely does not
//! carry is the recurrent element: the ssm entries instantiate their state at
//! `state_bf16`, so the slabs are bf16 and this file is where that is written
//! down.
//!
//! # A row's planes are one arena each, and which is k and which is v
//!
//! One arena per declared plane, each a contiguous virtual range holding
//! `pages · page_size` cells of that plane's own width. Page `p` of plane `i`
//! is at `p · page_size · width_i · element` from arena `i`'s base — which is
//! all the page addressing ever assumed, and the layout that makes a
//! page-count commit a PREFIX of every plane at once (wave C: an arena grows
//! and trims at its tail, so the bytes a watermark makes hot have to be at
//! its front). The planes used to be cut out of one allocation per row; the
//! per-plane split changes where the bases are and nothing about the
//! arithmetic between them.
//!
//! Which plane an entry reaches for is the plane COUNT, because the pool hands
//! out exactly two names:
//!
//! ```text
//! [w]        keys = values = plane 0      one plane, addressed as both
//! [w0, w1]   keys = plane 0, values = plane 1, and the widths may differ
//! ```
//!
//! `[w, w]` is the ordinary key|value pair. `[w]` is a row whose single plane
//! every reader walks through `pool.keys`: `attention.kv_append_shared` writes
//! the one rectangle to `keys` and to `values` alike
//! (`kernels-cuda/src/attn.rs`, `kv_append_shared`), and an indexer's keys and
//! a pooled cache's entries are written and read through `pool.keys` alone
//! (`attn/index.rs`, `attn/pool.rs`, both `kv_append`) — so pointing the two
//! handles at the same bytes is what makes one declared plane serve both
//! names. Two planes of DIFFERENT widths is the latent page: the mla kernels
//! take `pool.keys` as the compressed pages at `kv_lora_rank` and
//! `pool.values` as the rope pages at `rope_dim` (`attn/mla.rs`, `Layer::of`
//! and `kv_append`). Three planes or more is a refusal — this shell binds a
//! key plane and a value plane, and knows no third.

pub mod kv;
pub mod rs;

use kernels_cuda::{KvPool, RecurrentPool, Tensor};
use model_ir::{CacheRow, Dtype, Trace};

use crate::device::elastic::{self, Arena, Commit, PhysicalPool};
use crate::error::{Fault, Result};
use crate::settle::Airborne;
use crate::run::{CachePool, CacheTable};
use crate::store::kv::{Facts, Paging};

/// The neutral store's refusals, in this shell's vocabulary.
///
/// **THE CONDITION IS SHARED AND THE SENTENCE IS NOT.** `model_exec::store` owns
/// the arithmetic that decides a lane overran its block or a value's width is
/// symbolic; each shell owns how that reads to somebody holding a stack trace
/// ("the shell reserved" here, "the load reserved" on the Metal plane). This
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

/// The element the ssm entries instantiate their recurrent state at.
///
/// Stated, not declared: `CacheRow::State` carries a slab shape and no dtype,
/// and `kernels/attn/ssm.cuh` fixes the type at `__nv_bfloat16` in every
/// state-taking instantiation. A shell that guessed wider would allocate
/// twice the bytes and read every scan's history at half stride.
const STATE_DTYPE: Dtype = Dtype::Bf16;

/// The page layout enumerator this shell writes and the entries read: NHD,
/// `[page][token][head][dim]`.
///
/// One layout rather than a choice, because the strides that spell it are
/// cross-checked: `kv::head_split` reads `head_stride` as the head width
/// under NHD and `seq_stride` under HND, so a shell that set the pair for one
/// layout and the enumerator for the other would have every append refused —
/// or, worse, accepted at the wrong head count.
const NHD: i32 = 0;

/// **THE UNIFIED ACCOUNTING SENTENCE** — *weight tiers + elastic pool + safety
/// floor = the card* (alto streaming §3 item 5, `next.md` B2).
///
/// The two accountings this shell keeps have always summed correctly and
/// have never been WRITTEN DOWN, which is a different thing. The weight store
/// is a `cudaMalloc` in `Weights::resident`; [`Pools::reserve`] runs after it
/// and opens a [`PhysicalPool`] against whatever the card then reports. So
/// they unify BY ORDER — the pool gets what the weights left — and because
/// nothing states the sum, nothing can refuse AHEAD of it: a deployment whose
/// weights leave no room for its declared context discovers that as an
/// unrelated `Exhausted` on some later fire, or as a `cudaMalloc` that fails
/// inside a load, rather than as a sentence at boot naming the six numbers.
///
/// This is that sentence, as arithmetic:
///
/// ```text
/// card         what the device has, total
/// ceiling      card x utilization        the operator's whole allowance
/// weights      the T0 weight tier        what `Plan::device_demand` will hold
/// floor        min(128 MiB, card/10)     the driver's landing room
/// pool         ceiling - weights - floor what the elastic supply may hold
/// minimum      one slot at the declared context, every cache row
/// ```
///
/// and the claim is `pool >= minimum`.
///
/// # Why the minimum is one slot and not the whole reservation
///
/// The elastic pool is deliberately over-RESERVED: address space costs
/// nothing, every arena reserves at its own ceiling, and how far past each
/// base is readable is what admission decides per frame (design §8). So "the
/// pool cannot hold its reservation" is the normal case and refusing on it
/// would refuse every load. What is NOT normal is a pool that cannot hold ONE
/// SEQUENCE at the context the deployment declared: under that line
/// `[model] max_context` is a number no request can reach, every long request
/// dies `Exhausted`, and the deployment is misconfigured in a way no fire can
/// fix. That is the floor worth naming, and it is computed from the same
/// declaration [`Pools::reserve`] sizes the arenas from — `pages_per_slot`
/// pages of every kv plane at its own width, plus one slot of every recurrent
/// slab.
///
/// # `Impossible`, and why the card's OTHER tenants are not in this sum
///
/// Everything above is the CONFIG against the CARD: nothing another
/// deployment frees changes `card x utilization - weights - floor`, so the
/// refusal is [`Fault::Residency`] and reaches the contract as
/// `Error::Impossible`. What another process holds is physics, it is charged
/// against the pool by [`elastic::budget_bytes`] at open, and it refuses as
/// `Fault::OutOfMemory` -> `Error::Exhausted` with both numbers. The pair
/// `admit_tiers` draws — *statute is `Impossible` here, physics is `Exhausted`
/// there, and no path answers both* — is drawn here the same way and on
/// purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    /// What the device has, total.
    pub card: u64,
    /// `card x utilization` — the operator's whole allowance for pie.
    pub ceiling: u64,
    /// The T0 weight tier's bytes.
    pub weights: u64,
    /// `min(128 MiB, card/10)`, held back for the driver.
    pub floor: u64,
    /// `ceiling - weights - floor`: what is left for the elastic supply.
    pub pool: u64,
    /// One slot at the declared context, across every cache row.
    pub minimum: u64,
}

impl Accounting {
    /// Write the sentence down, from the card and the three demands.
    ///
    /// Pure arithmetic on four numbers, so the gate that spells the refusal
    /// needs no device.
    #[must_use]
    pub fn of(card: u64, utilization: f64, weights: u64, minimum: u64) -> Accounting {
        let fraction = if utilization.is_finite() {
            utilization.clamp(0.0, 1.0)
        } else {
            1.0
        };
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a byte count of a GPU card is far inside f64's exact integer range, \
                      and the product is floored back into u64 deliberately"
        )]
        let ceiling = (card as f64 * fraction) as u64;
        let floor = elastic::safety_floor_bytes(card);
        Accounting {
            card,
            ceiling,
            weights,
            floor,
            pool: ceiling.saturating_sub(weights).saturating_sub(floor),
            minimum,
        }
    }

    /// **Does the card hold this deployment?** One refusal, naming every term.
    ///
    /// # Errors
    ///
    /// [`Fault::Residency`] — `Error::Impossible` — when the elastic pool's
    /// share is under one slot at the declared context.
    pub fn admit(&self) -> Result<()> {
        if self.pool >= self.minimum {
            return Ok(());
        }
        Err(Fault::Residency(format!(
            "the card does not hold this deployment: {card} bytes on the device, of which \
             `[engine] gpu_mem_utilization` allows pie {ceiling}; this load's weight tier \
             takes {weights} and the driver's safety floor holds back {floor}, leaving the \
             elastic pool {pool} bytes — and one sequence at the declared context needs \
             {minimum} across this model's cache rows. weight tier + elastic pool + safety \
             floor must fit inside the fraction of the card, and here they do not. Lower \
             `[model] max_context` or `[model] slots`, raise `[engine] \
             gpu_mem_utilization`, or state a `[model] device_weight_budget` that streams \
             the weight tier down.",
            card = self.card,
            ceiling = self.ceiling,
            weights = self.weights,
            floor = self.floor,
            pool = self.pool,
            minimum = self.minimum,
        )))
    }
}

/// **The bytes ONE SLOT of every cache row occupies** — the elastic supply's
/// declared minimum, and [`Accounting`]'s last term.
///
/// The same declaration [`Pools::reserve`] sizes the arenas from, read one
/// slot wide instead of `slots` wide: `pages_per_slot` pages of every kv plane
/// at that plane's own width, plus one slot of every recurrent slab. A row
/// whose element has no byte size refuses here exactly as it refuses there.
///
/// # Errors
///
/// [`Fault::Unbound`] for a cache row whose element has no size.
pub fn one_slot_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    let mut bytes: u64 = 0;
    for row in &trace.caches {
        match row {
            CacheRow::Kv {
                name,
                planes,
                dtype,
                ..
            } => {
                let element = elem_bytes(name, *dtype)?;
                let cells = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
                for width in planes {
                    bytes = bytes.saturating_add(cells * width * element);
                }
            }
            CacheRow::State { name, slab } => {
                let stride: u64 = slab.iter().product();
                bytes = bytes.saturating_add(stride * elem_bytes(name, STATE_DTYPE)?);
            }
        }
    }
    Ok(bytes)
}

/// **Ask the card, then ask the sentence** — the one call `Shell::load` makes
/// before a byte of this load is allocated.
///
/// `weights` is [`Plan::device_demand`](crate::experts::Plan::device_demand),
/// with one reading applied: `Plan::default()` is FULL residency and stores a
/// derived zero, so a zero here means the whole table rather than no weights,
/// and the whole table is what `weights::plane_bytes` says it is.
///
/// # Errors
///
/// [`Fault::Runtimeless`] with no runtime selected, [`Fault::Device`] for the
/// memory query, [`Fault::Unbound`] for a cache row whose element has no size,
/// and [`Fault::Residency`] for the deployment the card does not hold.
pub fn admit_the_card(
    utilization: f64,
    weights: u64,
    trace: &Trace,
    paging: Paging,
) -> Result<Accounting> {
    let full: u64 = crate::weights::plane_bytes(trace)?
        .iter()
        .map(|plane| plane.next_multiple_of(crate::weights::ALIGN))
        .sum();
    let weights = match weights {
        0 => full,
        stated => stated.min(full),
    };
    let accounting = Accounting::of(
        card_bytes()?,
        utilization,
        weights,
        one_slot_bytes(trace, paging)?,
    );
    accounting.admit()?;
    Ok(accounting)
}

/// What this device has, total — the one number [`Accounting`] cannot derive.
///
/// # Errors
///
/// [`Fault::Runtimeless`] with no runtime selected, [`Fault::Device`] for the
/// query.
fn card_bytes() -> Result<u64> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        crate::device::ctx::check("cudaMemGetInfo", asked)?;
        Ok(total as u64)
    }
    #[cfg(not(feature = "_cuda"))]
    {
        Err(Fault::Runtimeless)
    }
}

/// How one cache row is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// A paged kv space: which geometry space it belongs to, and how the row's
    /// declared planes are handed out as the pool's key plane and value plane.
    Kv {
        space: u32,
        dtype: Dtype,
        /// Elements one token writes into the key plane — plane 0, whatever
        /// the row declared.
        keys_width: u64,
        /// Elements one token writes into the value plane: plane 1's width
        /// when the row declares two, and plane 0's again when it declares one
        /// (the same plane under both names).
        values_width: u64,
        /// Which of this row's plane arenas the value handle names. Zero for
        /// a one-plane row, which is how the two handles come to name the same
        /// bytes; one where the row declared a key plane and a value plane.
        values_plane: usize,
        /// One head plane under NHD, for the paged kernels that read a head
        /// count back out of the stride pair (`kv::head_split`). The token
        /// pitch beside it IS `keys_width` and is not written down twice.
        head_stride: u64,
    },
    /// A recurrent slab: elements per slot.
    State { stride: u64 },
}

/// The per-fire handles a pool row borrows: the geometry vectors this fire
/// wrote, and the graph-padding mask beside them.
///
/// A `KvPool` is a storage pointer plus the tables that address it, and the
/// tables are fire data — so the [`CacheTable`] is rebuilt each fire out of
/// long-lived storage and short-lived geometry. Rebuilding 42 rows of `Copy`
/// structs is arithmetic, not allocation.
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
    /// `i32`, `[lanes]`: which recurrent slot each lane owns.
    pub slot_ids: Tensor,
    /// **Does any lane of this fire fold at all?** `false` is the pure
    /// buffered scatter (design §6): every recurrent launch computes its rows
    /// and leaves the banks exactly where they were.
    pub write_state: bool,
    /// `u8`, `[lanes]`: the per-REQUEST fold predicate, or
    /// [`Tensor::ABSENT`] for a fire that predicates nothing.
    ///
    /// **PER LANE, NOT PER TOKEN ROW**, because that is what the kernels
    /// index: `attn/ssm.cuh`'s `row_persists(mask, r)` takes `r = blockIdx`
    /// over REQUESTS in every chunked arm. `channel::mask_from_commit` is
    /// handed the identity CSR for exactly this reason — its scatter
    /// degenerates to one byte per lane, which is the shape the scans read.
    pub write_state_mask: Tensor,
    /// `i32`, `[lanes]`: where each request's accepted prefix ends, or
    /// [`Tensor::ABSENT`] for a fire that truncates nothing.
    pub commit_len: Tensor,
    /// `i32`, `[lanes]`: where each request's TAIL segment begins, or
    /// [`Tensor::ABSENT`] for a fire no row splits (alto design §6's 2R
    /// interior split, wave F3b).
    ///
    /// **THE SAME VECTOR AS [`Seats::commit_len`], READ FROM THE OTHER END.**
    /// A row's fold boundary is one number: the head launch stops at it and
    /// the tail launch starts at it, so a fire that splits binds one device
    /// vector twice rather than staging two that could disagree. The two are
    /// never bound on the SAME launch — [`Run::recurrent`] hands the head the
    /// length and clears the origin, [`Run::recurrent_tail`] does the
    /// opposite.
    ///
    /// [`Run::recurrent`]: crate::run::Run
    /// [`Run::recurrent_tail`]: crate::run::Run
    pub begin_at: Tensor,
}

impl Seats {
    /// **The same seats, carrying this fire's recurrent verbs** (alto design
    /// §6).
    ///
    /// Additive and the shell chooses, exactly like [`Run::quantized`]: a fire
    /// never handed one folds unconditionally and truncates nothing, which is
    /// every fire in this tree that does not name `RsVerb::Buffer` or
    /// `RsVerb::FoldBuffered` — and is byte for byte the launch this shell has
    /// always made.
    ///
    /// [`Run::quantized`]: crate::run::Run::quantized
    #[must_use]
    pub fn rs(mut self, write_state: bool, mask: Tensor, commit_len: Tensor) -> Seats {
        self.write_state = write_state;
        self.write_state_mask = mask;
        self.commit_len = commit_len;
        self
    }

    /// **The same seats, told that some row's fold boundary is interior**
    /// (alto design §6's 2R split, wave F3b).
    ///
    /// Separate from [`Seats::rs`] and additive for the same reason it is:
    /// the origin is bound by the ONE fire shape that needs it — a row that
    /// folds a prefix of the tokens it is writing — and every other fire
    /// hands the launches the null pointer they have always taken. `boundary`
    /// is `commit_len`'s own vector: one number per lane, read as an end by
    /// the head launch and as a beginning by the tail.
    #[must_use]
    pub fn splitting(mut self, boundary: Tensor) -> Seats {
        self.begin_at = boundary;
        self
    }
}

/// **One span of kv cells moved inside this device's pools** — the shape
/// [`Pools::copy_kv`] takes, and the only one it takes.
///
/// **A WHOLE PAGE AND A SINGLE TOKEN ARE THE SAME MOVE.** The contract states
/// them apart — [`KvCopy::src_page_ids`]/`dst_page_ids` are the whole-page
/// half and [`KvCopy::moves`] the token-granular one — because a caller
/// spells a page swap and a fork's partial tail differently. What the page
/// arithmetic underneath sees is one thing either way: a run of `tokens`
/// cells starting at `(page, token)` on each side. So the shell flattens the
/// two spellings here rather than growing two loops that would drift.
///
/// [`KvCopy::src_page_ids`]: engine::transfer::KvCopy
/// [`KvCopy::moves`]: engine::transfer::KvCopy
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

/// **Every cache space's bytes, one ELASTIC ARENA per row plane** (alto
/// design §8, wave C).
///
/// What changed from the reservation model: a row's bytes are a virtual range
/// reserved at the budget's ceiling with nothing behind it, and physical
/// pages arrive under its FRONT as admission commits them. The addresses a
/// captured graph recorded do not move (article 7) — only how far past each
/// base is readable.
///
/// **ONE ARENA PER PLANE, NOT PER ROW.** An arena grows and trims at its
/// tail, so the bytes a watermark makes hot have to be at its front. A kv
/// row's key plane and value plane are two independent page-indexed tables
/// (`kv_append` walks each through its own `Tensor`), and laying them end to
/// end in one range would put value page 0 halfway down it — a prefix commit
/// would then leave every value page unmapped. The module header's "one
/// allocation per cache row" was about keeping a layer's kv in one place; a
/// plane is still one contiguous range and the pages inside it are still
/// contiguous, which is all the page arithmetic ever assumed.
#[derive(Debug)]
pub struct Pools {
    /// The budgeted supply every arena below draws physical pages from.
    pool: PhysicalPool,
    /// One entry per cache row, each holding that row's plane arenas: the
    /// declared planes for a kv row, one arena for a state row.
    rows: Vec<Vec<Arena>>,
    shapes: Vec<Shape>,
    paging: Paging,
    /// **Is the device idle?** The F2b run-ahead counter, cloned at load.
    /// [`Supply::trim`](engine::frame::Supply::trim) unmaps nothing while a
    /// step it did not settle may still be reading the tail it would take.
    airborne: Option<Airborne>,
    /// The kv-page watermark the last admitted frame committed to, kept so
    /// that [`Pools::committed_pages`] can be answered without asking every
    /// arena its byte count.
    committed_kv_pages: u32,
    /// The state-slot watermark, likewise.
    committed_state_slots: u32,
}

impl Pools {
    /// Reserve the pools one plan needs at one deployment's budget.
    ///
    /// **A KV ROW IS SIZED BY ITS DECLARATION**, plane by plane at each
    /// plane's own width, and `facts` is the RESTATEMENT checked against it
    /// where a paged launch made one. A row no paged launch reads — a latent
    /// page, an indexer's keys, a pooled cache's entries — is allocated as
    /// declared and checked against nothing, because nothing else states it.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a kv row this shell cannot cut into a key plane
    /// and a value plane — one that declares no planes, or three and more —
    /// for one whose paged readers restate a width its declaration does not
    /// spell, and for an element with no size; [`Fault::Device`] for the
    /// allocations.
    ///
    /// `facts` is indexed by CACHE ROW, not by geometry space: a page id says
    /// which page, never how wide the row it addresses is, and gemma's
    /// sliding and global layers share one page-id space at two widths
    /// ([`SpaceFacts`](crate::store::kv::SpaceFacts)).
    pub fn reserve(
        device: i32,
        utilization: f64,
        trace: &Trace,
        paging: Paging,
        facts: &Facts,
    ) -> Result<Pools> {
        // The budget is read from the card ONCE, here: the operator's fraction
        // of the whole card, less everything already on it (this load's weight
        // store first of all) and less a safety floor. Every arena below
        // reserves address space at its own ceiling out of that one supply, and
        // nothing is mapped until a frame's admission asks for it.
        //
        // **`utilization` IS `[engine] gpu_mem_utilization` AND THIS IS THE
        // ROUTE IT TAKES** (alto `next.md` B1): the worker's config declares
        // it, the boot document's `[engine]` table carries it, `crate::boot`
        // reads it into [`Knobs`](crate::Knobs), `Shell::load` hands it here,
        // and `PhysicalPool::open` is the one arithmetic that reads it. A
        // fraction of `1.0` is what this line did before the route existed.
        let pool = PhysicalPool::open(device, utilization)?;
        let mut rows: Vec<Vec<Arena>> = Vec::with_capacity(trace.caches.len());
        let mut shapes = Vec::with_capacity(trace.caches.len());

        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    space,
                } => {
                    let element = elem_bytes(name, *dtype)?;
                    let cells = paging.pages() * u64::from(paging.page_size);
                    let (keys_width, values_width, values_plane) = match planes.as_slice() {
                        // One plane, addressed as both k and v: the two
                        // handles name the same bytes, which is what
                        // `kv_append_shared` writes and what an index or pool
                        // reader's `keys`-only walk needs.
                        [plane] => (*plane, *plane, 0),
                        [keys, values] => (*keys, *values, 1),
                        [] => {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares no planes at all — one \
                                     token's entry is written as at least one plane"
                                ),
                            });
                        }
                        many => {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares {} planes — this shell \
                                     binds a key plane and a value plane, and knows no third",
                                    many.len()
                                ),
                            });
                        }
                    };
                    let restated = facts.row(index);
                    // A restatement only exists where a PAGED launch read the
                    // row and named a head count: the prefill arms state one,
                    // the decode and masked arms state a head width alone
                    // (`kv_heads` 0), and the latent, index and pool launches
                    // do not feed the row pass at all. Where it exists it must
                    // be the declaration.
                    if let Some(seat) = restated.filter(|seat| seat.kv_heads != 0) {
                        let heads = u64::from(seat.kv_heads) * u64::from(seat.head_dim);
                        if heads != keys_width || heads != values_width {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares the planes {planes:?} while \
                                     the paged launches that read it restate {} heads of {} — a \
                                     {heads}-wide row",
                                    seat.kv_heads, seat.head_dim
                                ),
                            });
                        }
                    }
                    // One arena per declared plane, each reserved at THIS
                    // plane's own ceiling: page `p` of plane `i` is at
                    // `p * page_size * width_i * element` from that plane's
                    // base, so a page-count commit is a prefix of every plane
                    // at once.
                    let mut planes_of_row = Vec::with_capacity(planes.len());
                    for width in planes {
                        planes_of_row.push(Arena::reserve(
                            &pool,
                            cells * width * element,
                            "bytes of a kv plane",
                        )?);
                    }
                    rows.push(planes_of_row);
                    shapes.push(Shape::Kv {
                        space: *space,
                        dtype: *dtype,
                        keys_width,
                        values_width,
                        values_plane,
                        // One head of the whole plane where no paged launch
                        // stated a head width: the latent, index and pool
                        // kernels take their widths from their op operands and
                        // never consult the strides.
                        head_stride: restated.map_or(keys_width, |seat| u64::from(seat.head_dim)),
                    });
                }
                CacheRow::State { name, slab } => {
                    let stride: u64 = slab.iter().product();
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, STATE_DTYPE)?;
                    // Slot `s` at `s * stride * element`: the slot watermark
                    // is a prefix here too.
                    rows.push(vec![Arena::reserve(
                        &pool,
                        bytes,
                        "bytes of a recurrent slab",
                    )?]);
                    shapes.push(Shape::State { stride });
                }
            }
            debug_assert_eq!(rows.len(), index + 1, "one arena set per cache row");
        }
        Ok(Pools {
            pool,
            rows,
            shapes,
            paging,
            airborne: None,
            committed_kv_pages: 0,
            committed_state_slots: 0,
        })
    }

    /// **Watch the run-ahead counter** (F2b), so `trim` can tell an idle
    /// device from a busy one.
    ///
    /// Additive and set once at load, the same seam
    /// [`GraphCache::watch`](crate::record::GraphCache) took and for the same
    /// reason: `Pools` is built before the counters exist.
    pub fn watch(&mut self, airborne: Airborne) {
        self.airborne = Some(airborne);
    }

    /// How the pages are handed out.
    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    /// **Every byte these pools may ever hold** — the ceiling the address
    /// space was reserved at, which is what `LoadFacts::pool_bytes` has
    /// always meant.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.rows
            .iter()
            .flatten()
            .map(elastic::Arena::max_bytes)
            .sum()
    }

    /// **Bytes actually under a mapping right now** (article 8: the engine
    /// owns this number, and it is the one the runtime used to re-derive from
    /// a free-list scan).
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.rows
            .iter()
            .flatten()
            .map(elastic::Arena::committed_bytes)
            .sum()
    }

    /// **The most that has ever been mapped**, arena by arena, summed. The
    /// high water a trim is measured against.
    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.rows
            .iter()
            .flatten()
            .map(elastic::Arena::high_water_bytes)
            .sum()
    }

    /// Bytes one logical page of the elastic supply holds — what
    /// `PoolFacts::elastic_page_bytes` publishes, and no longer zero.
    #[must_use]
    pub fn elastic_page_bytes(&self) -> u64 {
        self.pool.page_bytes()
    }

    /// The most logical pages this load may ever map.
    #[must_use]
    pub fn elastic_budget_pages(&self) -> u64 {
        self.pool.hard_pages()
    }

    /// The kv-page and state-slot watermarks the last admitted frame
    /// committed to.
    #[must_use]
    pub fn committed_watermarks(&self) -> (u32, u32) {
        (self.committed_kv_pages, self.committed_state_slots)
    }

    /// **Every arena's base address**, in row-then-plane order — the numbers
    /// article 7 says never move.
    ///
    /// Here so that a gate can hold them across a grow, a trim and a second
    /// grow and check the claim rather than trust it.
    #[must_use]
    pub fn bases(&self) -> Vec<u64> {
        self.rows
            .iter()
            .flatten()
            .map(elastic::Arena::base)
            .collect()
    }

    /// The cache table one fire resolves its cache ids through.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a kv row whose space this fire seated no
    /// geometry for.
    pub fn table(&self, seats: &Seats) -> Result<CacheTable> {
        let mut table = Vec::with_capacity(self.shapes.len());
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            table.push(match *shape {
                Shape::Kv {
                    space,
                    dtype,
                    keys_width,
                    values_width,
                    values_plane,
                    head_stride,
                } => {
                    let seat = seats
                        .spaces
                        .get(space as usize)
                        .ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "cache space {space}, for which this fire wrote no \
                                     geometry"
                            ),
                        })?;
                    let cells = self.paging.pages() * u64::from(self.paging.page_size);
                    // THE BASE, NOT THE COMMITTED EDGE. A pool handle names
                    // the whole plane at its ceiling because that is what the
                    // recorded graph will read forever (article 7); how much
                    // of it is backed is admission's business and is checked
                    // there, once per frame, rather than per handle.
                    let plane = |at: usize, width: u64| {
                        Tensor::new(
                            planes.get(at).map_or(0, elastic::Arena::base),
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        )
                    };
                    CachePool::Kv { space, pool: KvPool {
                        keys: plane(0, keys_width),
                        // A one-plane row seats `values_plane == 0` and the
                        // key plane's own width, so both handles are the one
                        // plane.
                        values: plane(values_plane, values_width),
                        // The shadow, scale and envelope planes belong to the
                        // quantized schemes; a native pool binds none and the
                        // entries never reach for them (`kv::native_bf16`).
                        bf16_keys: Tensor::new(0, 0, 0, dtype),
                        bf16_values: Tensor::new(0, 0, 0, dtype),
                        key_scales: Tensor::new(0, 0, 0, Dtype::U8),
                        value_scales: Tensor::new(0, 0, 0, Dtype::U8),
                        page_indices: seat.page_indices,
                        page_indptr: seat.page_indptr,
                        last_page_lens: seat.last_page_lens,
                        row_valid: seat.row_valid,
                        env_min: Tensor::new(0, 0, 0, dtype),
                        env_max: Tensor::new(0, 0, 0, dtype),
                        has_envelopes: false,
                        page_size: narrow(u64::from(self.paging.page_size)),
                        // NHD: one token's step through the key plane is that
                        // plane's whole width, and one head plane is a share
                        // of it. The pair is what `kv::head_split` reads the
                        // head width back out of and what `index::pool_pitch`
                        // reads the whole-row pitch out of.
                        seq_stride: wide(keys_width),
                        head_stride: wide(head_stride),
                        layout: NHD,
                        scheme_byte: 0,
                        block_size: 0,
                        max_pages_per_request: narrow(u64::from(self.paging.pages_per_slot)),
                        pages_in_batch: narrow(u64::from(seats.pages)),
                    },
                    }
                }
                Shape::State { stride } => CachePool::Recurrent(RecurrentPool {
                    write_state: seats.write_state,
                    write_state_mask: seats.write_state_mask,
                    commit_len: seats.commit_len,
                    begin_at: seats.begin_at,
                    // **THE FOLD'S OWN ROUNDING, ON EVERY PATH** (wave F3b).
                    // `false` is the plain fold-per-token forward's policy,
                    // and binding it everywhere is what makes a replay equal
                    // the fold it replaces to the byte. The seat exists so
                    // that the length beside it can stop meaning two things.
                    fused_decay: false,
                    // One row serves both seats. A `CacheRow::State` is one
                    // slab and the ops that read it name it once — the
                    // gated-delta scan through `slab`, the causal convolution
                    // through `conv_slab` — so pointing both at this row's
                    // bytes is what makes `conv.L` and `delta.L` two
                    // independent spaces rather than two halves of one.
                    slab: Tensor::new(
                        planes.first().map_or(0, elastic::Arena::base),
                        self.paging.slots,
                        narrow(stride) as u32,
                        STATE_DTYPE,
                    ),
                    slot_ids: seats.slot_ids,
                    slot_stride_elems: stride as i64,
                    conv_slab: Tensor::new(
                        planes.first().map_or(0, elastic::Arena::base),
                        self.paging.slots,
                        narrow(stride) as u32,
                        STATE_DTYPE,
                    ),
                    conv_stride: stride as i64,
                }),
            });
        }
        Ok(CacheTable(table))
    }

    /// Clear one slot's recurrent state.
    ///
    /// A RECURRENT SLOT IS ITS HISTORY, so opening a sequence in a slot
    /// another sequence used means zeroing what that one left — unlike a kv
    /// page, which is overwritten before it is read because `kv_len` says
    /// nothing lives past the append. There is no cheaper truth available:
    /// the scan reads the whole bank on its first step.
    ///
    /// Called from [`Shell::open`](crate::serve::Shell::open) by a caller
    /// whose page table is the shell's, and from the fire path for one whose
    /// page table is its own — there, a lane stating `held == 0` IS the
    /// sequence beginning. One `cudaMemset` per sequence either way.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for the
    /// fill.
    pub fn clear(&mut self, slot: u32) -> Result<()> {
        self.zero_slot(None, slot)
    }

    /// **[`Pools::clear`], on the fire's stream** — the fire path's spelling.
    ///
    /// The clear that begins a sequence is stream work: it is decided in
    /// `prepare` (which slots arrive with `have == 0`) and issued in
    /// `enqueue`, in front of the launches that read the bank. Doing it with a
    /// synchronous `cudaMemset` drained everything airborne, which is a host
    /// wait between two waves — article 2's forbidden transition, arriving on
    /// the first fire of every sequence. `Shell::open` keeps the synchronous
    /// spelling above, because it is control plane and orders against nothing.
    ///
    /// # Errors
    ///
    /// As [`Pools::clear`].
    pub fn clear_on(&mut self, stream: *mut core::ffi::c_void, slot: u32) -> Result<()> {
        self.zero_slot(Some(stream), slot)
    }

    /// **Copy one recurrent slot's banks onto another, on `stream`** (alto
    /// survey §9's gap list; dev `RecurrentStateCache::copy_slot_d2d`).
    ///
    /// The device half of a copy-on-write fork: a slot IS its history, so a
    /// fork that shares one and then folds into it would advance both
    /// sequences at once. Every `CacheRow::State` row of the plan is copied —
    /// a family's conv bank and its delta bank are two rows and both are the
    /// state — and a copy onto itself is nothing.
    ///
    /// **WHOLE SLOTS ONLY**, which is dev's rule and is not a simplification:
    /// a recurrent bank is a folded summary of a prefix, not an array of
    /// per-token entries, so "the first `n` tokens of a slot" names nothing
    /// that exists. A caller asking for a partial move is refused by
    /// [`Cuda::copy_state`](crate::Cuda) before it reaches here.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for the
    /// copy.
    pub fn copy_slot(
        &mut self,
        stream: *mut core::ffi::c_void,
        src: u32,
        dst: u32,
    ) -> Result<()> {
        for slot in [src, dst] {
            if slot >= self.paging.slots {
                return Err(Fault::Ceiling {
                    what: "recurrent slots",
                    need: u64::from(slot) + 1,
                    have: u64::from(self.paging.slots),
                });
            }
        }
        if src == dst {
            return Ok(());
        }
        // BOTH SLOTS HAVE TO BE BACKED. A fork is a control-plane verb and
        // may name a slot no frame has admitted yet, so the copy commits the
        // watermark it needs before it reads or writes a byte.
        self.ensure_state(src.max(dst) + 1)?;
        let element = u64::from(elem_size(STATE_DTYPE));
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * element;
            crate::device::copy_d2d(
                stream,
                arena.span(u64::from(dst) * bytes, bytes)?,
                arena.span(u64::from(src) * bytes, bytes)?,
                usize::try_from(bytes).unwrap_or(0),
            )?;
        }
        Ok(())
    }

    /// **Copy kv cells between pages of these pools, on `stream`** (alto
    /// survey §9's gap list; dev's `copy_kv_cells_kernel` and
    /// `KvSwapPool::copy_d2d_async`, context.cpp:2584-2712).
    ///
    /// The device half of a prefix-tree fork: a page run that two sequences
    /// share is grafted onto fresh ids, and the partial page at the boundary
    /// has its live tokens copied out so the fork can append past them without
    /// writing into the parent's cells.
    ///
    /// # Every plane of every row, because a page id names all of them
    ///
    /// A "page" is not one allocation. Page `p` exists once per PLANE of every
    /// `CacheRow::Kv` this plan declares — eighteen layers times a key plane
    /// and a value plane, for a dense text model — and a mover that copied a
    /// subset would leave a fork attending to the parent's keys at some layers
    /// and its own at others, which reads as fluent garbage rather than as an
    /// error. So the loop is over `rows × planes` and the caller names the
    /// page once, exactly as [`Pools::copy_slot`] takes a slot once for the 36
    /// recurrent rows underneath it.
    ///
    /// **AND THERE ARE NO ENVELOPE PLANES TO CARRY.** dev's pages could ride a
    /// per-page key envelope (`env_min`/`env_max`, the Quest criticality
    /// scores); this shell binds `has_envelopes: false` and two null tensors on
    /// every pool it hands out ([`Pools::table`]), because the quantized and
    /// enveloped schemes are not among the ones it seats. A shell that grows
    /// them grows an arena per envelope plane, and this loop — which walks
    /// whatever arenas a row owns — carries them with no change.
    ///
    /// # On the stream, and not synchronized
    ///
    /// `cudaMemcpyAsync` device-to-device on the fire stream (article 2: no
    /// synchronous memcpy, no stream sync on a path a fire follows). That is
    /// also what makes it CORRECT rather than merely fast: the copies queue
    /// behind every step already airborne — which may still be reading the
    /// source pages — and in front of every fire submitted after this verb
    /// returns, which is exactly the ordering a caller that forks and then
    /// fires against both halves is asking for. A host sync here would buy
    /// nothing and cost a drained device.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a page past the pool or a run past a page's
    /// tokens, and [`Fault::Device`] for the copies. A move whose two ends
    /// OVERLAP is refused a layer up, by [`Cuda::copy_kv`](crate::Cuda), where
    /// the contract's `Invalid` can be spoken: it is the caller's statement
    /// that is wrong, not this pool's arithmetic.
    pub fn copy_kv(
        &mut self,
        stream: *mut core::ffi::c_void,
        moves: &[Move],
    ) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }
        let page_size = u64::from(self.paging.page_size);
        let mut highest = 0u64;
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
                highest = highest.max(u64::from(page) + 1);
            }
        }
        // BOTH ENDS HAVE TO BE BACKED. A fork is a control-plane verb and
        // names the destination page before any frame has admitted it — the
        // same reason `copy_slot` commits its slot watermark before it reads a
        // byte.
        self.ensure_kv(u32::try_from(highest).unwrap_or(u32::MAX))?;
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::Kv {
                dtype,
                keys_width,
                values_width,
                ..
            } = *shape
            else {
                continue;
            };
            let element = u64::from(elem_size(dtype));
            for (at, arena) in planes.iter().enumerate() {
                // The same rule `watermark_bytes` reads: plane 0 is the key
                // plane at its own width, plane 1 the value plane at its own.
                // A one-plane row has one arena and both handles name it, so
                // copying it once IS copying k and v.
                let width = if at == 0 { keys_width } else { values_width };
                let cell = width * element;
                for span in moves {
                    if span.tokens == 0 {
                        continue;
                    }
                    let bytes = u64::from(span.tokens) * cell;
                    let src = (u64::from(span.src_page) * page_size
                        + u64::from(span.src_token))
                        * cell;
                    let dst = (u64::from(span.dst_page) * page_size
                        + u64::from(span.dst_token))
                        * cell;
                    if src == dst {
                        continue;
                    }
                    crate::device::copy_d2d(
                        stream,
                        arena.span(dst, bytes)?,
                        arena.span(src, bytes)?,
                        usize::try_from(bytes).unwrap_or(0),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// **One slot's recurrent banks, read back**, every `CacheRow::State` row
    /// end to end in plan order.
    ///
    /// Not on any fire path — it is a synchronous D2H — and it exists because
    /// "these two slots hold the same state" is the only observation that can
    /// settle whether a buffered fold folded what a plain one would have
    /// (design §6's own equivalence).
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a slot past the pool, [`Fault::Device`] for the
    /// read.
    pub fn state_bytes(&mut self, slot: u32) -> Result<Vec<u8>> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        self.ensure_state(slot + 1)?;
        let element = u64::from(elem_size(STATE_DTYPE));
        let mut out = Vec::new();
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * element;
            let at = out.len();
            out.resize(at + usize::try_from(bytes).unwrap_or(0), 0);
            crate::device::copy_d2h(
                arena.span(u64::from(slot) * bytes, bytes)?,
                &mut out[at..],
            )?;
        }
        Ok(out)
    }

    fn zero_slot(&mut self, stream: Option<*mut core::ffi::c_void>, slot: u32) -> Result<()> {
        if slot >= self.paging.slots {
            return Err(Fault::Ceiling {
                what: "recurrent slots",
                need: u64::from(slot) + 1,
                have: u64::from(self.paging.slots),
            });
        }
        // The bank has to EXIST before it can be zeroed. `Shell::open` runs
        // on the control plane and reaches a slot no frame has admitted, so
        // the commit belongs here rather than in the caller.
        self.ensure_state(slot + 1)?;
        let element = u64::from(elem_size(STATE_DTYPE));
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * element;
            let at = arena.span(u64::from(slot) * bytes, bytes)?;
            let len = usize::try_from(bytes).unwrap_or(0);
            match stream {
                Some(stream) => crate::device::zero_span_on(stream, at, len)?,
                None => crate::device::zero_span(at, len)?,
            }
        }
        Ok(())
    }

    /// **Commit the kv arenas up to a page watermark**, for the
    /// control-plane verbs that reach a page no frame admitted.
    ///
    /// The ceiling is the pool's own, spelled exactly as
    /// [`Supply::commit`](engine::frame::Supply::commit) spells it, so that a
    /// page id a fire would refuse is a page id this verb refuses first and in
    /// the same words.
    fn ensure_kv(&mut self, pages: u32) -> Result<()> {
        let capacity = self.paging.pages();
        if u64::from(pages) > capacity {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(pages),
                have: capacity,
            });
        }
        match self.commit_to(pages, 0)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// **Commit the recurrent arenas up to a slot watermark**, for the
    /// control-plane verbs that reach a slot no frame admitted.
    fn ensure_state(&mut self, slots: u32) -> Result<()> {
        match self.commit_to(0, slots)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// **The atomic multi-arena commit, in this pool's vocabulary** (dev's
    /// `frame_targets` + `commit_cuda_arena_targets_atomically`,
    /// context.cpp:2129-2175 and elastic.cpp:535-627).
    ///
    /// Every arena is asked for the PREFIX its watermark names — pages for a
    /// kv plane, slots for a recurrent slab — and the whole set moves or none
    /// of it does. Neither watermark may go DOWN here: `commit` only ever
    /// grows, and the one thing that lowers a watermark is
    /// [`Pools::release_to`], which is the trim.
    fn commit_to(&mut self, kv_pages: u32, state_slots: u32) -> Result<Commit> {
        let kv_pages = kv_pages.max(self.committed_kv_pages);
        let state_slots = state_slots.max(self.committed_state_slots);
        let page_size = self.paging.page_size;
        let Pools {
            pool,
            rows,
            shapes,
            ..
        } = self;
        let mut targets = Vec::new();
        for (planes, shape) in rows.iter_mut().zip(shapes.iter()) {
            for (at, arena) in planes.iter_mut().enumerate() {
                let bytes = watermark_bytes(shape, at, kv_pages, state_slots, page_size);
                targets.push(elastic::Target { arena, bytes });
            }
        }
        let outcome = elastic::commit_atomically(pool, &mut targets)?;
        if outcome == Commit::Committed {
            self.committed_kv_pages = kv_pages;
            self.committed_state_slots = state_slots;
        }
        Ok(outcome)
    }

    /// **Unmap every arena's tail down to the watermarks `hint` names.**
    ///
    /// The inverse of [`Pools::commit_to`], and the ONLY thing that lowers a
    /// watermark. Best-effort by construction: an arena releases whole map
    /// units, so a target inside the last unit gives nothing back, and one
    /// handle is kept mapped-out so the next grow costs a `cuMemMap` alone.
    fn release_to(&mut self, kv_pages: u32, state_slots: u32) {
        let page_size = self.paging.page_size;
        let Pools {
            pool,
            rows,
            shapes,
            ..
        } = self;
        for (planes, shape) in rows.iter_mut().zip(shapes.iter()) {
            for (at, arena) in planes.iter_mut().enumerate() {
                let bytes = watermark_bytes(shape, at, kv_pages, state_slots, page_size);
                arena.release_tail(pool, bytes);
            }
        }
        self.committed_kv_pages = kv_pages;
        self.committed_state_slots = state_slots;
    }
}

/// How many bytes of one arena a pair of watermarks makes hot.
///
/// A kv plane is `pages · page_size` cells of its own width; a recurrent slab
/// is `slots` banks of its own stride. Both are prefixes, which is the whole
/// reason a plane gets an arena of its own.
fn watermark_bytes(
    shape: &Shape,
    plane: usize,
    kv_pages: u32,
    state_slots: u32,
    page_size: u32,
) -> u64 {
    match *shape {
        Shape::Kv {
            dtype,
            keys_width,
            values_width,
            ..
        } => {
            let width = if plane == 0 { keys_width } else { values_width };
            u64::from(kv_pages) * u64::from(page_size) * width * u64::from(elem_size(dtype))
        }
        Shape::State { stride } => {
            u64::from(state_slots) * stride * u64::from(elem_size(STATE_DTYPE))
        }
    }
}

/// One refused commit, as the fault this shell already speaks.
///
/// **THE TWO REFUSALS KEEP THE TWO SENTENCES THEY ALREADY HAD**
/// (`api::fault`): `Fault::OutOfMemory` crosses as
/// `Error::Exhausted { resource, wanted, available }` — the frame is worth
/// re-submitting behind something that frees pages — and `Fault::Ceiling`
/// crosses as `Error::Impossible`. Nothing new had to be added to the
/// taxonomy for wave C, which is the sign the F1 seam was cut in the right
/// place.
fn refuse(pool: &PhysicalPool, outcome: Commit) -> Fault {
    let page = pool.page_bytes();
    match outcome {
        Commit::Committed => Fault::program(
            "store::commit",
            "a committed outcome reached the refusal path".to_string(),
        ),
        Commit::Exhausted { required, budget } => Fault::OutOfMemory {
            need: required.saturating_mul(page),
            have: budget.saturating_mul(page),
        },
        Commit::Impossible { required, ceiling } => Fault::Ceiling {
            what: "bytes of elastic device memory",
            need: required.saturating_mul(page),
            have: ceiling.saturating_mul(page),
        },
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

/// A plane width, as the `i64` a pool's strides are spelled in.
fn wide(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// **The engine's half of memory, elastic** (alto design §8; article 8: the
/// runtime owns policy, the engine owns supply).
///
/// [`Pools`] reserves address space at the budget's ceiling and maps physical
/// pages under the FRONT of it as frames are admitted, so what a load holds
/// is what it has demanded rather than what it might one day demand. The
/// admission gate is one atomic commit across every arena
/// ([`Pools::commit_to`]): article 4's zero side effects are the pool's own
/// reserve-then-map order, and past it the stream work is success-only.
///
/// **THE TWO CEILING REFUSALS ARE THE ONES `kv::geometry_with` ALREADY
/// WROTE**, down to the variant and the `what` string, because a frame that
/// fails admission here must fail it identically to the fire that would have
/// reached the page arithmetic a dozen lines later. F1 moved the question
/// earlier without changing its answer; wave C makes the answer cost real
/// bytes without changing the question.
impl engine::frame::Supply for Pools {
    type Error = Fault;

    fn commit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        // Only the slots this shell PAGES are its supply. A lane that brought
        // its own page table brought its own addressing with it (article 8:
        // engine page ids are the runtime's, and the shell's paging is
        // sizing), so its slot number is not an index into anything here —
        // which is exactly why the demand this is handed counts only the
        // shell-owned lanes.
        if demand.state_slots > self.paging.slots {
            return Err(Fault::Ceiling {
                what: "kv slots",
                need: u64::from(demand.state_slots),
                have: u64::from(self.paging.slots),
            });
        }
        let capacity =
            u64::from(self.paging.slots).saturating_mul(u64::from(self.paging.pages_per_slot));
        if u64::from(demand.kv_pages) > capacity {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(demand.kv_pages),
                have: capacity,
            });
        }
        match self.commit_to(demand.kv_pages, demand.state_slots)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// **Give the tails back, when the device is idle and the hint is below
    /// what is mapped.**
    ///
    /// Two conditions, and both are load-bearing.
    ///
    /// *Idle*, because an unmap is immediate and a step already on the stream
    /// is still reading. `settled == launched` is the whole test and the F2b
    /// run-ahead counter answers it in one atomic load
    /// ([`Airborne::count`](crate::settle::Airborne::count)); a shell with no
    /// counter watching it — one built by a test — is treated as idle.
    ///
    /// *Below what is mapped*, because growth is admission's business.
    /// `trim` never maps a page; a hint above the watermark is a no-op, not a
    /// second, non-atomic commit path (article 8 forbids the second
    /// allocator).
    ///
    /// **THE HINT IS A RESIDENCY STATEMENT AND ITS TRUTH IS THE CALLER'S.**
    /// A kv page holds bytes for as long as somebody's prefix is cached in
    /// it, and which pages those are is POLICY — the runtime's trie, its CoW
    /// and its eviction choice (article 8). The engine knows only what the
    /// last frame addressed, which is not the same set. So this method
    /// unmaps exactly what it is told to and invents nothing: a caller that
    /// hands it a frame's demand instead of a residency watermark will lose
    /// cached bytes, and that is the caller's error rather than a safety
    /// margin this method is entitled to add.
    ///
    /// # And a third condition: the drop has to be worth the unmap
    ///
    /// `cuMemUnmap` is not free and it is not asynchronous. A c=64 profile
    /// counted **186 of them mid-serving at 545 µs each** — a tenth of a
    /// second of host wall time, spent because a residency watermark that
    /// breathes by a page either side of its mean crosses the "below what is
    /// mapped" line on most frames, and every crossing pays a full unmap and
    /// then a `cuMemMap` on the frame after it.
    ///
    /// So the trim is HYSTERETIC: a drop smaller than
    /// [`TRIM_HYSTERESIS_SHIFT`] of what is mapped is not acted on, and the
    /// watermark stays where it is until the hint has genuinely moved away
    /// from it. Nothing is lost that the design cared about — the pages stay
    /// mapped, which is what they were a moment ago, and the next frame that
    /// wants them finds them.
    ///
    /// **THE BAND IS LIFTED UNDER PRESSURE**, because the whole reason to
    /// give a page back is that somebody else needs it: once the physical
    /// pool is within [`TRIM_HYSTERESIS_SHIFT`] of its budget, every byte the
    /// hint releases is released exactly as asked. A trim that had to happen
    /// still happens; only the ones nobody was waiting for are deferred.
    fn trim(&mut self, hint: engine::frame::Demand) {
        let idle = self.airborne.as_ref().is_none_or(|counts| counts.count() == 0);
        if !idle {
            return;
        }
        if u64::from(hint.kv_pages) >= u64::from(self.committed_kv_pages)
            && hint.state_slots >= self.committed_state_slots
        {
            return;
        }
        if !self.trim_is_worth_the_unmap(hint) {
            return;
        }
        self.release_to(
            hint.kv_pages.min(self.committed_kv_pages),
            hint.state_slots.min(self.committed_state_slots),
        );
    }
}

/// **How far below the watermark a hint must fall before an unmap is paid
/// for**, as a right shift: `1 << 3` is an eighth.
///
/// Doubles as the pressure line — a pool within an eighth of its budget trims
/// on any hint at all, because then the pages are wanted.
const TRIM_HYSTERESIS_SHIFT: u32 = 3;

impl Pools {
    /// Is this drop large enough — or the pool tight enough — to be worth a
    /// `cuMemUnmap`? See [`Supply::trim`](engine::frame::Supply::trim)'s note
    /// on the band.
    fn trim_is_worth_the_unmap(&self, hint: engine::frame::Demand) -> bool {
        let budget = self.pool.budget_pages();
        let free = budget.saturating_sub(self.pool.committed_pages());
        if free <= budget >> TRIM_HYSTERESIS_SHIFT {
            // Under pressure: the band is off and the hint is obeyed.
            return true;
        }
        let kv_drop = u64::from(self.committed_kv_pages)
            .saturating_sub(u64::from(hint.kv_pages.min(self.committed_kv_pages)));
        let state_drop = u64::from(
            self.committed_state_slots
                .saturating_sub(hint.state_slots.min(self.committed_state_slots)),
        );
        kv_drop > u64::from(self.committed_kv_pages) >> TRIM_HYSTERESIS_SHIFT
            || state_drop > u64::from(self.committed_state_slots) >> TRIM_HYSTERESIS_SHIFT
    }
}
