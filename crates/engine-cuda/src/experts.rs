//! **The routed-expert tier**: a device slab smaller than the bank, a pinned
//! host copy of the whole of it, and a device-resident indirection table
//! between them (alto design §7, wave D2).
//!
//! ```text
//! T2 mapped   what NEITHER budget holds: the warm artifact, mmap'd, faulted
//!             in from NVMe by the GPU touch itself (HMM)         no budget — a file
//! T1 pinned   every expert of every routed bank, budgeted        `host_weight_budget`
//! T0 device   `resident` slots of each bank, budgeted            `device_weight_budget`
//! table       one fixed-address `expert_id -> base address` row per bank
//! ```
//!
//! # Residency is a performance promotion, never a correctness condition
//!
//! Routing is computed ON DEVICE, so no host decision can precede a fire and
//! decide which experts it will need. Design §7's answer, and this module's
//! whole shape: the kernels read weights THROUGH the table, and an entry for
//! an expert that is not on the device points at its PINNED bytes over UVA.
//! A miss costs PCIe bandwidth for the reads that miss and nothing else — no
//! callback, no host round trip, no synchronize (article 2). The host then
//! promotes what the fires actually used, between fires, behind fixed
//! addresses (article 7), and a fire that overtakes the promotion is not
//! wrong; it is slow, for one fire.
//!
//! That is why there is no "streaming mode" here and no config key for one.
//! The only nouns are the two budgets, and full residency is the DEGENERATE
//! case of them — [`Plan::of`] with an uncapped budget answers an empty plan,
//! [`Tier`] is never opened, the weight table's rows stay
//! [`WeightRow::Dense`](crate::run::WeightRow::Dense), and the MoE select
//! kernels are handed [`ExpertTable::RESIDENT`], which is two null pointers
//! and the arithmetic they always did. That is dev's `place_all()`
//! (`driver/cuda/src/loader/group_stream_cache.hpp`), and it means a
//! fully-resident load pays nothing at all for this file existing.
//!
//! # The uniformity proof, restated for a plane
//!
//! Dev's `GroupStreamCache` needed two facts before it could page a group
//! into fixed slots: every instance's plan has the same `persistent_bytes`,
//! and every instance's buffers land at the same offsets inside its arena.
//! Here the unit is one PARAM PLANE rather than a group of plans, and the two
//! facts are shapes the model text already declared — but they are ASSERTED
//! off the plan ([`Plan::of`]) rather than assumed, because the day a bank
//! arrives that does not have them, the slot arithmetic below is silently
//! wrong rather than loudly refused:
//!
//! * a bank's leading axis IS its expert count, and its remaining axes are one
//!   expert's rectangle — so every expert of one bank occupies exactly
//!   `bytes / experts` bytes, at intra-slot offset zero, and a bank whose
//!   bytes do not divide is refused by name;
//! * every routed bank of one plan states the SAME expert count, so one
//!   residency decision covers the plan rather than one per layer.
//!
//! # The DENSE planes spill too, and by the OTHER demand shape (D2b)
//!
//! Embeddings, attention projections, norms, the head. Nothing on the device
//! chooses which of them a fire reads, so a table over them would be a table
//! every entry of which is hit every fire — design §7 gives them the static
//! demand shape instead, and its ordering principle is the compiler's
//! **prefetch schedule** ([`model_compiler::prefetch`]).
//!
//! The schedule is fire-invariant: the same planes in the same order every
//! step, so the compiler can say which plane a fire reaches FIRST and which it
//! reaches LAST. This module reads it from one end — a budget gives planes up
//! in REVERSE schedule order, so the embedding (read at node zero of every
//! fire) is the last thing to leave the device and a late projection is the
//! first. A plane that leaves becomes a group of ONE plane and walks the same
//! three tiers a packed bank walks; nothing below this line has a dense arm,
//! because a dense plane IS a group whose expert count is zero.
//!
//! **WHAT IS BUILT HERE IS THE TIER, NOT THE PUMP.** A spilled dense plane is
//! read where it lies, over UVA out of pinned memory or over HMM out of the
//! mapping, at the same PCIe floor streaming §2 states for the shape
//! (`spill_bytes / min(NVMe, PCIe)` per step). The SLOTTED pump — a fixed set
//! of device slots whose contents rotate ahead of the read, addresses never —
//! is `prefetch::Slotting`, which is emitted and proved and has no consumer:
//! rotating within a fire needs a seam at each region boundary, and the walk's
//! only such seam is a `Sink` this lane does not own. Streaming §3 item 4
//! carries the exact shape of what is left.
//!
//! **A REGISTERED PLANE NEVER SPILLS.** An adapter bank is written by
//! `register_adapter` at a store offset reserved at load (design §8's
//! pointer-stability rule), so a bank on another tier would be a registration
//! writing into a plane nobody reads. Those are the floor under any budget,
//! and a budget below THEM is still [`Fault::Residency`] by name.
//!
//! # A SPLIT-PLANE BANK IS ONE SEAT, AND THE SEAT IS THE WHOLE GROUP
//!
//! A quantized routed bank is not one plane: it is e2m1 CODES beside e8m0
//! SCALES, two `Trace::params` rows under one `Def::Weight` id, which
//! `weights.rs` now seats as [`WeightRow::Planes`](crate::run::WeightRow)
//! (alto streaming §3 item 6). Both planes are indexed by the SAME number —
//! the routing vector's expert id — and the mxfp4 select kernel does that
//! indexing ITSELF:
//!
//! ```text
//! codes  + e * n * (k / 2)        `moe_matmul_select_mxfp4`, quant.cuh
//! scales + e * n * (k / 32)
//! ```
//!
//! It reads no indirection table. That one fact fixes the granularity of
//! everything below: the per-expert table this module publishes is a thing
//! the DENSE select kernel dereferences (`moe_matmul_select_gemv_body` loads
//! `expert_table[expert]`), and there is no such load on the packed path to
//! point anywhere. So a split-plane bank cannot be seated expert by expert
//! without a kernel that asks.
//!
//! **The unit of residency is therefore the GROUP** — the code plane and
//! every companion plane the load plan pairs with it — and a group is held
//! WHOLE on one tier: every plane of it in the device store, or every plane
//! of it in the pinned tier, addressed by the kernel over UVA, or every plane
//! of it in the mapped artifact. Which is not a weaker promise than the dense
//! tier's, it is a different one, and it is the one that makes a torn pair
//! unconstructible. Seating the codes alone would be the failure the metal
//! sibling names in its own header — `scales += e * ...` reading another
//! expert's factors, a model that computes and is wrong — and this shell does
//! not have a shape that can express it.
//!
//! # THE LADDER: A GROUP MOVES NOW, AND THE PAIR IS ONE WORD (wave B7)
//!
//! W-5 made the torn pair unconstructible the cheap way — *a group never
//! changes tier after the load* — and paid for it with a residency decided
//! once, off a plan, before a token was served. What HMM did underneath was
//! the only promotion on this path, and streaming §3 item 3 said so.
//!
//! The kernel change it named is here, and it is one load. The two plane
//! bases were kernel PARAMETERS, and a captured graph holds its parameters
//! forever (article 7), so the group's tier was frozen with them. Now the
//! launch carries a CELL instead — one 16-byte, 16-byte-aligned word of data
//! at a fixed address, holding `(codes, scales)`, read by
//! `moe_matmul_select_mxfp4` with a single `ld.global.v2.u64`:
//!
//! ```text
//! before   codes  + e * n * (k / 2)          bases are launch parameters
//!          scales + e * n * (k / 32)
//! after    (codes, scales) = *cell           ONE extra load, per GROUP,
//!          codes  + e * n * (k / 2)          per LAUNCH: one address the
//!          scales + e * n * (k / 32)         whole grid shares
//! ```
//!
//! **One extra load per group, argued.** It is not per route and not per row:
//! every thread of the grid loads the same sixteen bytes, so it is one L1
//! broadcast against a kernel that then reads hundreds of kilobytes of codes
//! per warp. And a fully-resident load does not pay even that — the cell
//! pointer is null, the arm is the arithmetic it always did, and that is the
//! same degeneration [`ExpertTable::RESIDENT`] gives the dense path.
//!
//! **THE PAIR IS ONE WORD, WHICH IS WHY THE TORN PAIR IS STILL
//! UNCONSTRUCTIBLE — now across MOTION rather than by forbidding it.** Both
//! plane addresses live in the same sixteen bytes, filled together in the
//! pinned shadow before any copy is issued, so no state of the cell names one
//! group's codes beside another's exponents. And the move itself lands the
//! bytes before it flips the pointer:
//!
//! ```text
//! gap n     the berth's occupant's cell -> the artifact   16 bytes
//!           [the next fire waits HERE]
//!           the candidate's planes -> the berth           ~265 MiB, unwaited
//! gap n+k   the candidate's cell -> the berth             16 bytes
//!           [the next fire waits HERE]
//! ```
//!
//! Between the two gaps the berth is referenced by nobody: the occupant's
//! cell already names the file and the candidate's does not name the berth
//! yet. **Nothing a fire waits for is bigger than sixteen bytes** — the copy
//! that costs something is enqueued on the notify stream AFTER the event the
//! next fire waits on, and whether it has landed is asked at a later gap
//! rather than waited for (article 2, the same sentence one tier up). A gap
//! that would start a second copy is skipped rather than queued behind the
//! first, because a stream is ordered and a promotion that would have to wait
//! is not one.
//!
//! **And the vote is a strict-improvement rule, which has a consequence worth
//! writing down.** A swap is taken only when the candidate is strictly hotter
//! than the occupant it displaces, so `Σ hits × rung` strictly falls and the
//! sequence terminates. It also means a model whose routed banks are ALL read
//! every fire — which is every MoE in today's catalog — has a uniform vote
//! and a steady state at the plan's own assignment. That is the right answer
//! and not a missing feature: with equal demand any assignment costs the
//! same. What the ladder is for is the demand the plan cannot see — a bank a
//! session never routes to, holding a rung a hot mapped bank wants.
//! [`Tier::decide_group`] carries the argument in full.
//!
//! What a capped budget buys on the packed path is therefore per-BANK and not
//! per-expert: gpt-oss-20b is 48 groups of ~265 MiB, and a budget under its
//! ~13.8 GiB holds the ones that fit and reads the rest over PCIe. Slow, not
//! wrong — the same sentence, one tier up.
//!
//! # T2: what NEITHER budget holds (streaming §2, wave W-1)
//!
//! A group the device budget cannot seat goes to the pinned tier. A group the
//! HOST budget cannot seat either has one place left, and streaming §0 already
//! named it: the **warm-boot artifact**, which is a snapshot of the device
//! store with every dequant, cast and repack already applied — so serving out
//! of it requires no conversion, which is the whole precondition. It is
//! `mmap`'d `PROT_READ, MAP_PRIVATE` and NOT populated, and the group's weight
//! row points straight into the mapping.
//!
//! **AND THERE ARE TWO SUCH FILES SINCE §K-T4**, which [`Spill`] is the name
//! of. The whole-table artifact above is where a deployment starts, and only a
//! machine that could hold the model resident can ever have written one. The
//! SERVING artifact is the other: a capped boot with a weight cache directory
//! transcribes every plane of the trace into it, so from then on this
//! deployment's three tiers come out of one file — at any budget, since §M.3 —
//! and the whole-table snapshot can be deleted. Nothing below this line
//! branches on which — a plane is a window on a mapping either way.
//!
//! **THE GPU DEREFERENCES THE MAPPING ITSELF.** No `cudaHostRegister`, no
//! copy, no pinned staging: on a device that reports `pageableMemoryAccess`
//! (CUDA 12.2+ HMM) an unregistered host pointer is a valid device pointer,
//! and a touch of a cold page faults it in from NVMe while the SM stalls and
//! the HOST does not. Registering the mapping instead would page-LOCK it,
//! which is T1 wearing T2's name and would defeat the one property the tier is
//! for. A device that does not report the attribute is refused by name rather
//! than served through a fallback that silently costs host RAM.
//!
//! The constitutional sentence is the same one the UVA miss already owns:
//! **slow, not wrong** (article 2). A T2 group's reads are counted
//! ([`observed`]) and nothing is load-bearing on the count.
//!
//! **AND IT IS ALSO THE LADDER'S FLOOR.** The artifact is a snapshot of the
//! device store, so it carries every plane of every group — including the
//! ones that have never been off the device — and that is what a demotion
//! points a cell back at. So the ladder's precondition is exactly streaming
//! §0's: a load with no artifact has the assignment it booted with, which is
//! a shorter ladder rather than a wrong one, and nothing refuses.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

use model_ir::{Def, Linear, Operation, ParamSource, Trace, ValueId};

use crate::device::graph::Event;
use crate::device::{Buffer, Pinned, copy_any};
use crate::error::{Fault, Result};

/// One table entry: a device address, eight bytes.
const ENTRY: u64 = 8;

/// One usage counter: a `u32`, four bytes — the width of the `atomicAdd` the
/// select kernel does.
const COUNTER: u64 = 4;

/// **How many experts of one bank may change residency between two fires.**
///
/// A statute, not a constitution (design §1): the promotion rides the notify
/// stream and the next fire's enqueue waits for it, so the number is a bound
/// on how much copying one inter-fire gap may hold. Two per bank per gap
/// converges on a stable working set within a few dozen fires at
/// top-k 8 and costs, for the largest bank in the catalog, ~4 MiB of PCIe per
/// gap.
const MOVES: usize = 2;

/// **How many packed GROUPS may change tier between two fires** (wave B7).
///
/// One, and the arithmetic is why: an expert of a dense bank is ~2 MiB and a
/// gap may hold two of them; a gpt-oss group is ~265 MiB — the code plane and
/// its exponents, whole — which is ~10 ms of PCIe. That copy is never waited
/// for (see [`Tier::promote`]: it is enqueued AFTER the event the next fire
/// waits on), but it does occupy the notify stream, and a second one queued
/// behind it would put a fire's own staging behind ~20 ms of it.
///
/// So: one group in flight, and a gap that would start a second is SKIPPED —
/// the same doctrine [`MOVES`] rides, at the size the unit actually is.
const GROUP_MOVES: usize = 1;

/// One base cell: up to three device addresses and a pad, thirty-two bytes,
/// sixteen-byte aligned — the `(codes, scales[, biases])` bases the packed
/// select reads as one aggregate. See `MoeGroupBases` in `linear/quant.cuh`.
/// It was two addresses in sixteen bytes until the affine banks arrived,
/// whose element is `code * scale + bias` and whose zero points are a plane
/// of their own.
const CELL: u64 = 32;

/// How many planes a cell can name. Two for mxfp4 (codes beside exponents),
/// three for an affine bank (codes, scales, zero points); a fourth plane is
/// refused by name rather than seated with a plane unaddressed.
const CELL_PLANES: usize = 3;

/// **A quantized bank's OTHER device planes**, by `Trace::params` index — the
/// pairing [`weights::pairings`](crate::weights) reads off the load plan, in
/// the one shape this module needs it: *"which other params move when this
/// one's bank moves"*.
///
/// Empty for a plan with no packed bank, one entry per bank for an mxfp4 one
/// (its e8m0 scales). Keyed by the CODE plane's index, which is the param a
/// routed matmul's `bank` port names.
///
/// **THE PAIRING IS THE LOAD PLAN'S AND NEVER A NAME'S.** A `.scales` suffix
/// read off a param would pair a scale tensor with the wrong bank the first
/// time two banks of one layer sorted between each other; the loader recorded
/// the pair at the point of declaring it (`QuantAttachment`), so that is where
/// it is asked. [`weights::attachments`](crate::weights::attachments) is the
/// door that answers it before a byte is landed.
pub type Attachments = BTreeMap<usize, Vec<usize>>;

/// **The operator's two ceilings, as the planner is asked them** (alto
/// streaming §4).
///
/// `None` is uncapped on either axis and the pair of `None`s is the degenerate
/// plan. There is deliberately NO third field: T2 has no budget, because it is
/// not a reservation but a file — whether it exists is the SHELL's question
/// and it is asked at [`Tier::open`], not here. The planner's job is to say
/// how much would have to go there, and
/// [`Residency::admit_tiers`](engine::load::Residency::admit_tiers) is what
/// decides whether that is a load or a refusal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Budgets {
    /// T0, the device store.
    pub device: Option<u64>,
    /// T1, the pinned host tier.
    pub host: Option<u64>,
}

impl Budgets {
    /// Both uncapped — the degenerate plan, and what every load in this
    /// workspace that states no budget asks for.
    #[must_use]
    pub const fn uncapped() -> Budgets {
        Budgets {
            device: None,
            host: None,
        }
    }

    /// A device ceiling and an uncapped host tier — the two-tier shape wave
    /// W-5 planned, kept as a door because it is what most callers mean.
    #[must_use]
    pub const fn device(bytes: u64) -> Budgets {
        Budgets {
            device: Some(bytes),
            host: None,
        }
    }
}

/// **WHERE A SPILLED GROUP'S BYTES ARE READ FROM** — the two files that can
/// answer, and the reason this is an enum rather than a path (§K.6-T4).
///
/// T2 is a mapping, and what is mapped is an artifact this shell wrote on an
/// earlier boot. There are two of those, they are written by different loads
/// for different reasons, and both carry the same bytes for a spilled plane:
///
/// ```text
/// Resident  the whole-table artifact an UNCAPPED boot wrote — a snapshot of
///           a device store that held every plane, indexed by param. The only
///           source there was before T-4, and the one that bootstraps a
///           deployment whose budgets have never held it.
/// Tiers     this deployment's OWN serving artifact — written by a capped
///           boot that had one of these and copied every plane forward.
///           Budget-FREE since §M.3: it carries one image per plane of the
///           trace, so it answers for whatever THIS load spills and for
///           whatever the next budget pair spills too.
/// ```
///
/// The pair matters because of what it lets an operator do: once one capped
/// boot has written the serving artifact, that file IS the load — every plane
/// of the trace, cut onto three rungs by whichever budgets the boot has — so
/// the hundred-gigabyte resident artifact that seeded it can be deleted and
/// every later boot of that deployment is warm.
///
/// **NOTHING IS COPIED HERE EITHER.** Both arms hand out windows on a
/// mapping; which file a window came from is the only difference, and no
/// kernel can tell.
#[derive(Debug)]
pub enum Spill {
    /// **THE `.zt` ITSELF** (§M-4d) — the one file, which since §M-4b IS the
    /// serving artifact and needs no second file beside it.
    ///
    /// The names are the trace's param order, because that is what `plane`
    /// takes an id in and what a serving artifact addresses an object BY. The
    /// two indices this variant replaces are keyed by that ordinal directly;
    /// this one carries the translation instead of asking a reader to
    /// re-derive it per lookup.
    Serving(crate::checkpoint_serving::Serving),
}

impl Spill {
    /// Which file these planes come out of.
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        let Spill::Serving(serving) = self;
        serving.path()
    }

    /// **One plane's bytes, borrowed from the mapping**, or `None` for a plane
    /// this file does not carry.
    ///
    /// `id` is the param's ordinal in the plan's own order, and the plane axis
    /// is always zero: a split-plane bank is already two `Trace::params` rows
    /// here, so each is its own index entry. The `plane` axis of either index
    /// is for a shell that puts both planes under one id, which this one does
    /// not.
    ///
    /// # And a serving artifact answers for EVERY plane, on any rung
    ///
    /// Which used to take three lookups and now takes one. Format 2 held three
    /// budget-shaped sections and a plane was in whichever one the plan's rung
    /// put it in, so a lookup that asked only the mapped section would have
    /// answered for T2 and nothing else — which is enough for [`Tier::open`]'s
    /// seating but NOT for the LADDER, whose swaps send a berth's occupant
    /// back to the file and therefore need a backing for every routed group.
    ///
    /// §M's file has one image per plane and no rungs in it at all, so the
    /// question "where is this plane" has one answer for every budget, and the
    /// ladder is as long as the resident artifact ever made it.
    #[must_use]
    pub fn plane(&self, id: u32) -> Option<&[u8]> {
        let Spill::Serving(serving) = self;
        serving.plane(id)
    }

    /// **What an operator does about a file that cannot answer for a plane.**
    ///
    /// One remedy now, because there is one file. It said "the two arms have
    /// different remedies" while a spill could come out of a resident snapshot
    /// or a tier artifact, and §M-4d retired both: a spilled load reads the
    /// model's own `.zt`, so a hole is never a stale second file and never a
    /// key collision.
    ///
    /// What it can still be is narrower and the sentence says it: the stamp
    /// was checked at open (§M-4c), so the file is not a foreign recipe — it
    /// is a trace that has gained a plane since the artifact was written.
    fn remedy(&self) -> &'static str {
        "The artifact is the model's own `.zt`, which since §M-4 IS the serving file: \
         its objects are this trace's plane names, so a hole in it is a name this build \
         declares and that file does not hold. The stamp was checked at open, so this \
         is not a foreign recipe — it is a trace that has gained a plane since the \
         artifact was written. Run `pie model import --force` on the source it names."
    }
}

/// **Which tier holds a streamed group's planes.**
///
/// A group is held WHOLE on exactly one of them — the header says why — so
/// this is a property of the group and never of a plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Held {
    /// T0: the device store. **Never a [`Plan::groups`] entry** — a group the
    /// store holds is a group the plan did not have to place — and that is
    /// what [`Plan::seated`] answers instead. It IS a tier the LADDER can put
    /// a group back on (wave B7), which is why the word exists here at all.
    Device,
    /// T1: page-locked host memory the kernel reads over UVA. The tier owns
    /// the bytes and the landing sink writes them.
    Pinned,
    /// T2: the warm artifact, mapped. The tier owns nothing — the bytes were
    /// on disk before this load started and no one copies them.
    Mapped,
}

impl Held {
    /// **How far down the ladder this tier is**: 0 device, 1 pinned, 2 mapped.
    ///
    /// The ONE ordering the promotion reads, and it is a cost order and not a
    /// preference: HBM, then PCIe over UVA, then a page fault to NVMe. A move
    /// is legal only from a higher rung to a lower one.
    #[must_use]
    pub const fn rung(self) -> u8 {
        match self {
            Held::Device => 0,
            Held::Pinned => 1,
            Held::Mapped => 2,
        }
    }
}

// ── the T2 register (design §14: counted, never load-bearing) ───────────────

/// What the mapped tier has been asked for, process-wide.
///
/// **A REGISTERED EXCEPTION AND NOT A CONTROL INPUT** (design §14, the same
/// discipline `weight_cache::observed` follows). Nothing in this file reads
/// these numbers back: no plan branches on them, no promotion consults them,
/// no refusal is decided by them. They exist so that an operator asking *"is
/// this deployment actually serving out of the SSD, and how much of it?"* has
/// an answer that is not a guess — and so that a T2 arm silently doing nothing
/// is visible as a zero rather than invisible as a success.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Observed {
    /// Planes seated on T2 — one per plane of every mapped group, over every
    /// load this process has opened.
    pub seated: u64,
    /// Bytes those planes hold. What is being read off the disk rather than
    /// out of memory.
    pub bytes: u64,
    /// **Planes the artifact could not answer for**: absent from its index, or
    /// present at a length this plan does not agree with. Every one of them is
    /// also a refusal by name — the counter is the process-wide tally, the
    /// refusal is the load's own answer.
    pub absent: u64,
    /// Loads that planned a spill and opened a source for it.
    pub loads: u64,
}

#[derive(Clone, Copy)]
enum Stat {
    Seated = 0,
    Bytes = 1,
    Absent = 2,
    Loads = 3,
}

static T2: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn bump(stat: Stat) {
    add(stat, 1);
}

fn add(stat: Stat, by: u64) {
    T2[stat as usize].fetch_add(by, Ordering::Relaxed);
}

/// **What the mapped tier has done, process-wide.** See [`Observed`].
#[must_use]
pub fn observed() -> Observed {
    let at = |stat: Stat| T2[stat as usize].load(Ordering::Relaxed);
    Observed {
        seated: at(Stat::Seated),
        bytes: at(Stat::Bytes),
        absent: at(Stat::Absent),
        loads: at(Stat::Loads),
    }
}

/// **Can this device dereference an ordinary host mapping?**
///
/// `cudaDevAttrPageableMemoryAccess` — CUDA 12.2+ HMM. `true` says a GPU touch
/// of an unregistered, un-pinned host pointer is legal and that a cold page
/// faults in underneath it, which is the entire mechanism T2 stands on.
///
/// Read from the CURRENT device, because that is the one the store was
/// allocated on and the one the kernels will run on. `false` without a
/// runtime, which is the honest answer for a build that cannot ask.
#[must_use]
pub fn pageable_access() -> bool {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        let mut ordinal = 0i32;
        // SAFETY: `ordinal` is a live out-parameter.
        if unsafe { rt::cudaGetDevice(&raw mut ordinal) } != rt::cudaError::cudaSuccess {
            return false;
        }
        let mut value = 0i32;
        // SAFETY: `value` is a live out-parameter and `ordinal` came from the
        // runtime one line above.
        let status = unsafe {
            rt::cudaDeviceGetAttribute(
                &raw mut value,
                rt::cudaDeviceAttr::cudaDevAttrPageableMemoryAccess,
                ordinal,
            )
        };
        status == rt::cudaError::cudaSuccess && value != 0
    }
    #[cfg(not(feature = "_cuda"))]
    {
        false
    }
}

/// **What a plan wants and what a budget allows**, decided off the trace and
/// the load plan's pairings — before the device is bound and before a byte is
/// allocated.
///
/// The empty plan is full residency and is what an uncapped
/// [`Residency`](engine::load::Residency) produces. Everything
/// downstream reads [`Plan::streams`] and does nothing when it is false.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    banks: Vec<BankPlan>,
    /// The split-plane groups the device store does NOT hold — every plane of
    /// each, seated together in the pinned tier and read there over UVA.
    /// Empty for a plan whose packed banks all fit, and for a plan with none.
    groups: Vec<GroupPlan>,
    /// **The ROUTED packed groups the device store DOES hold** (wave B7).
    ///
    /// They are not in `groups` and must not be: `groups` is the list of
    /// placements the budget was forced to make, and every reader of it —
    /// `weights::places`, the landing sink, the two standing gates — reads it
    /// as *"these planes are not in the store"*. A T0 group is in the store
    /// like any other plane.
    ///
    /// It is carried separately because the LADDER needs it. A device seat is
    /// the fastest rung and a group on it is what a hotter group displaces, so
    /// the tier has to know which store regions are group-shaped and where
    /// they are. Dense planes the budget spilled are deliberately absent: they
    /// are `routed: false`, they are bound as one handle with no cell for a
    /// promotion to write, and moving them is D2b's pump and not this ladder.
    seated: Vec<GroupPlan>,
    /// `param index -> how many experts of it live on the device`. The one
    /// map `weights::places` consults to reserve a bank at less than its
    /// declared size.
    resident_of: BTreeMap<usize, u32>,
    /// Every param of every group held on T1, flattened.
    pinned_of: BTreeSet<usize>,
    /// Every param of every group held on T2 — the mapped artifact. Disjoint
    /// from `pinned_of` by construction: a group has one tier.
    mapped_of: BTreeSet<usize>,
    device_bytes: u64,
    host_bytes: u64,
    spill_bytes: u64,
}

/// One routed bank, as the plan sees it.
#[derive(Debug, Clone)]
pub struct BankPlan {
    /// Index into `Trace::params`.
    pub param: usize,
    /// The param's own name, which is the plan's and the contract's.
    pub name: String,
    /// The bank's leading axis.
    pub experts: u32,
    /// How many of them the device slab seats.
    pub resident: u32,
    /// One expert's bytes — the slot stride, uniform across the bank.
    pub stride: u64,
}

/// **One split-plane quantized bank, as the plan sees it** — the group that
/// moves together or not at all.
///
/// It carries no `resident` count and no `stride`, and the module header says
/// why: the packed select kernel computes each plane's expert base itself and
/// dereferences no table, so the only two residencies a group has are "in the
/// store" and "in the pinned tier". A `GroupPlan` exists only for the second.
/// **One plane of a packed group** — what it declares and what a tier gives
/// it.
///
/// The two numbers differ for the same reason `weights::Place`'s do: what the
/// checkpoint publishes is the plane, and what a store or a tier seats is that
/// rounded up to the handle alignment. The artifact's index states the FORMER
/// (it is a snapshot of published bytes) and the pinned tier seats the LATTER,
/// so both are carried rather than one being re-derived at each reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupPlane {
    /// Index into `Trace::params`.
    pub param: usize,
    /// The plane's own bytes, as the plan declares and the checkpoint
    /// publishes.
    pub bytes: u64,
    /// Those bytes at [`weights::ALIGN`](crate::weights::ALIGN) — what a tier
    /// seats, and what the store would have given it.
    pub reserved: u64,
}

#[derive(Debug, Clone)]
pub struct GroupPlan {
    /// Index into `Trace::params` of the CODE plane — the param a routed
    /// matmul's `bank` port names, and the group's own name.
    pub param: usize,
    /// That param's name, which is the plan's and the contract's.
    pub name: String,
    /// Every plane of the group, code plane first and then its companions,
    /// ascending by param index.
    pub planes: Vec<GroupPlane>,
    /// The bank's leading axis — reported, never divided by. See the type's
    /// own doc. **Zero for a spilled DENSE plane**, which is a group of one
    /// and has no expert axis at all.
    pub experts: u32,
    /// Is this a routed expert bank, or a dense plane the budget gave up?
    ///
    /// The tiers treat them identically — that is the point of making a dense
    /// plane a group — and only the residency REPORT tells them apart, because
    /// an operator reading "48 banks mapped" wants to know whether that
    /// included the embedding.
    pub routed: bool,
    /// Every plane's bytes, summed: what the device store gives this group
    /// back by not holding it, and what the tier below owes it instead.
    pub bytes: u64,
    /// **Which tier holds it.** Decided by the two budgets in the order they
    /// are stated: the device first, then the pinned host, then the mapping.
    pub held: Held,
}

/// **THE PRIORITY RANKING, BEFORE ANY BUDGET CUTS IT** (§M.3).
///
/// [`Plan::of`] used to be one function: it worked out which planes a budget
/// is allowed to move, ranked them, and then walked that ranking against the
/// two budgets. **The first two steps mention no budget at all** — they are a
/// pure function of the trace and the load plan's pairings — and separating
/// them is what makes the serving artifact budget-polymorphic: the file states
/// the RANKING, a boot states the CUT, and one file then serves any budget
/// pair on this setup.
///
/// ```text
///   floor      params no budget can move: ParamSource::Registered
///   sequence   the cut sequence, HOTTEST FIRST
///   banks      the dense routed banks: a device slab prefix AND a whole
///              pinned copy, at every budget
/// ```
///
/// Two readers, one statement. [`Plan::cut`] is the first; [`Ranking::images`]
/// is the second and lays the artifact out. A ranking written twice would be
/// two chances for a file to describe an order no boot walks.
#[derive(Debug, Clone)]
pub struct Ranking {
    /// What a FULLY RESIDENT load of this trace demands of the device store.
    full: u64,
    /// **The params no budget can move**, ascending.
    ///
    /// A registered plane is an adapter bank: `register_adapter` writes it at
    /// a store offset reserved at load, so a bank on another tier is a
    /// registration landing in bytes nobody reads. They are the floor under
    /// any budget this trace can serve, and they are the reason the artifact
    /// omits them — at write time they hold what `Buffer::zeroed` left.
    floor: Vec<usize>,
    /// What the floor reserves in the store, together.
    floor_bytes: u64,
    /// **THE CUT SEQUENCE, HOTTEST FIRST** — every dense plane a budget may
    /// spill, in `prefetch::Schedule` order, then every packed routed group in
    /// ascending param order.
    ///
    /// `prefetch::Schedule` is a pure function of the trace — the same planes
    /// in the same order every fire — so read from the front it is what a pump
    /// would copy first, and read from the back it is what a ceiling
    /// surrenders first. Every `held` in here is the resting value; the cut
    /// overwrites it.
    sequence: Vec<GroupPlan>,
    /// **The DENSE routed banks**, ascending by param.
    ///
    /// They are not in `sequence` and must not be: a bank has TWO readers at
    /// EVERY budget — the device slab takes its first `resident` experts and
    /// the pinned tier holds all of them, authoritatively — so it sits on
    /// neither side of a cut. That is also why the artifact puts them after
    /// the sequence: keeping them out of it leaves both cut points inside one
    /// contiguous run.
    banks: Vec<BankPlan>,
}

impl Ranking {
    /// **The ranking this trace declares**, with the uniformity proof checked.
    ///
    /// `planes` is the load plan's pairing — which other params move when a
    /// packed bank moves — read by [`weights::prospect`] before a byte is
    /// landed. An empty map is a trace with no packed bank, and a packed bank
    /// missing from it is refused by name rather than seated half.
    ///
    /// # Errors
    ///
    /// [`Fault::Param`] for a param whose dtype has no byte size, whose bank
    /// shape breaks the uniformity proof, or whose packed bank the load plan
    /// pairs no scales plane with.
    ///
    /// [`weights::prospect`]: crate::weights::prospect
    pub fn of(trace: &Trace, planes: &Attachments) -> Result<Ranking> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let full = bytes.iter().map(|b| b.next_multiple_of(crate::weights::ALIGN)).sum();
        // **THE "NOTHING IS ROUTED" REFUSAL STOOD HERE, AND D2b RETIRED IT.**
        // It said that a plan with no routed-expert bank had no tier to hold
        // less of, because only the DYNAMIC demand shape streamed and the
        // static one — dense overflow — was not built. It is built: a dense
        // plane the budget cannot seat becomes a group of one and takes the
        // same three tiers (streaming §2).
        let (banks, packed) = routed(trace, planes, &bytes)?;
        let streamable: BTreeSet<usize> = banks
            .iter()
            .map(|bank| bank.param)
            .chain(packed.iter().flat_map(|group| group.planes.iter().map(|plane| plane.param)))
            .collect();
        let mut floor: Vec<usize> = Vec::new();
        let mut floor_bytes = 0u64;
        let mut spillable: Vec<GroupPlan> = Vec::new();
        for (at, param) in trace.params.iter().enumerate() {
            if streamable.contains(&at) {
                continue;
            }
            let reserved = bytes[at].next_multiple_of(crate::weights::ALIGN);
            if param.source == ParamSource::Registered {
                floor.push(at);
                floor_bytes += reserved;
                continue;
            }
            spillable.push(GroupPlan {
                param: at,
                name: param.name.clone(),
                planes: vec![GroupPlane {
                    param: at,
                    bytes: bytes[at],
                    reserved,
                }],
                experts: 0,
                bytes: reserved,
                held: Held::Pinned,
                routed: false,
            });
        }
        // **THE ORDER A BUDGET GIVES PLANES UP IN IS THE COMPILER'S**, which
        // is what makes it budget-free and therefore a property of the file:
        // `prefetch::Schedule` is a pure function of the trace, so two boots
        // of one deployment rank the same planes the same way and an artifact
        // written by either is walked identically by the other.
        let schedule = model_compiler::prefetch::Schedule::of(trace);
        let rank: BTreeMap<usize, usize> = schedule
            .order()
            .into_iter()
            .enumerate()
            .map(|(at, param)| (param, at))
            .collect();
        spillable.sort_by_key(|group| {
            (rank.get(&group.param).copied().unwrap_or(usize::MAX), group.param)
        });
        // **THE DENSE PLANES COME BEFORE THE PACKED BANKS.** A dense plane is
        // read by every token of every fire; a routed expert may not be read
        // at all. So the unconditional demand ranks above the conditional one,
        // which is the only ordering that does not spend the scarcest tier on
        // the least certain reader.
        Ok(Ranking {
            full,
            floor,
            floor_bytes,
            sequence: spillable.into_iter().chain(packed).collect(),
            banks,
        })
    }

    /// What a FULLY RESIDENT load of this trace demands of the device store.
    #[must_use]
    pub fn full(&self) -> u64 {
        self.full
    }

    /// The params no budget can move — see the field.
    #[must_use]
    pub fn floor(&self) -> &[usize] {
        &self.floor
    }

    /// The cut sequence, hottest first — see the field.
    #[must_use]
    pub fn sequence(&self) -> &[GroupPlan] {
        &self.sequence
    }

    /// The dense routed banks — see the field.
    #[must_use]
    pub fn banks(&self) -> &[BankPlan] {
        &self.banks
    }

    /// **WHERE EVERY IMAGE OF THE SERVING ARTIFACT GOES**, one quad per image:
    /// `(param, offset, bytes, reserved)`, in payload order (§M.3).
    ///
    /// [`Plan::host_layout`]'s analogue one level up, and it exists for the
    /// same reason: the artifact's key mixes these triples, its writer
    /// transcribes them into the index, and its reader cuts them — so the walk
    /// that fills the payload and the index that describes it must be ONE
    /// statement.
    ///
    /// ```text
    ///   1. the cut sequence, in its own order, each group's planes in theirs
    ///   2. the dense routed banks, WHOLE — every expert, not the resident
    ///      ones — in ascending param order
    /// ```
    ///
    /// **The registered planes are absent**, and that is the whole of what the
    /// artifact's `FLAG_ADAPTERS_ZEROED` states: they hold what
    /// `Buffer::zeroed` left at the moment the file is written, so a restore
    /// leaves the store's zeros where they are rather than reading a hundred
    /// megabytes of them back.
    ///
    /// `bytes` is what the checkpoint publishes into the image and `reserved`
    /// is what the payload gives it — the same pair `weights::Place` carries.
    /// **A BANK'S SPAN IS PADDED HERE AND IN [`Plan::host_layout`] ALIKE**, so
    /// that every image on every rung starts on
    /// [`weights::ALIGN`](crate::weights::ALIGN): that alignment is the reason
    /// one image can serve all three rungs, and a span that only happened to
    /// be aligned would make it an accident of the shapes.
    #[must_use]
    pub fn images(&self) -> Vec<(u64, u64, u64, u64)> {
        let mut out = Vec::new();
        let mut at = 0u64;
        for group in &self.sequence {
            for plane in &group.planes {
                out.push((plane.param as u64, at, plane.bytes, plane.reserved));
                at += plane.reserved;
            }
        }
        for bank in &self.banks {
            let span = u64::from(bank.experts) * bank.stride;
            let reserved = span.next_multiple_of(crate::weights::ALIGN);
            out.push((bank.param as u64, at, span, reserved));
            at += reserved;
        }
        out
    }
}

impl Plan {
    /// **The residency plan for `trace` under `budgets`.**
    ///
    /// [`Ranking::of`] and then [`Plan::cut`], which is what this function has
    /// always been: the ranking is the part that mentions no budget, and the
    /// cut is the part that mentions nothing else.
    ///
    /// `None` — uncapped — answers the empty plan: land everything, open no
    /// tier, hand the kernels no table.
    ///
    /// # Errors
    ///
    /// [`Ranking::of`]'s, plus [`Fault::Residency`] for a budget that cannot
    /// hold the registered planes plus one slot of every dense bank.
    pub fn of(trace: &Trace, planes: &Attachments, budgets: Budgets) -> Result<Plan> {
        Plan::cut(&Ranking::of(trace, planes)?, budgets)
    }

    /// **THE CUT: what these budgets make of that ranking** (§M.3).
    ///
    /// A stated budget is met by holding fewer EXPERTS of a dense bank and
    /// fewer whole GROUPS of a packed one, and never by holding fewer dense
    /// planes than the ranking's floor.
    ///
    /// **`budgets.host` IS WHAT OPENS THE THIRD TIER.** A group the device
    /// budget cannot seat goes to T1; one the HOST budget cannot seat either
    /// goes to T2 — the mapped artifact — and the plan reports those bytes as
    /// [`Plan::spill_demand`]. This function never asks whether a T2 source
    /// exists: that is a file on a disk, and the statute that decides whether
    /// a spilled load is servable is
    /// [`Residency::admit_tiers`](engine::load::Residency::admit_tiers).
    ///
    /// **THE WALK CONTINUES PAST A GROUP THAT DOES NOT FIT.** A tier is
    /// offered each group in ranking order and a group too large for what is
    /// left falls through to the next tier while the walk goes on, because a
    /// later, smaller group that does fit is a group that tier may as well
    /// hold. So the two cut points are RANK THRESHOLDS with fitted exceptions
    /// rather than strict prefixes — which costs the artifact nothing, since
    /// every image is addressed by its own index entry and no rung is required
    /// to be one contiguous run of the file.
    ///
    /// # Errors
    ///
    /// [`Fault::Residency`] for a budget under the registered planes plus one
    /// expert slot of every dense bank, naming both numbers.
    pub fn cut(ranking: &Ranking, budgets: Budgets) -> Result<Plan> {
        let full = ranking.full;
        let Some(budget) = budgets.device else {
            return Ok(Plan {
                device_bytes: full,
                ..Plan::default()
            });
        };
        if budget >= full {
            // The budget covers the whole table: the degenerate case, and it
            // is answered as the degenerate case rather than as a streamed
            // load that happens to keep everything. `place_all()`.
            return Ok(Plan {
                device_bytes: full,
                ..Plan::default()
            });
        }

        let found = ranking.banks.clone();
        let experts = found.first().map_or(0, |bank| bank.experts);
        let dense = ranking.floor_bytes;
        // One slot of every DENSE bank. A packed group contributes nothing
        // here: its floor is zero, because the whole of it may live on the
        // pinned tier and still be read.
        let slots: u64 = found
            .iter()
            .map(|bank| bank.stride.next_multiple_of(crate::weights::ALIGN))
            .sum();
        let floor = dense + slots;
        if budget < floor {
            return Err(Fault::Residency(format!(
                "`device_weight_budget` is {budget} bytes; this plan's REGISTERED \
                 planes demand {dense} resident and its {} routed banks need one \
                 expert slot each on top, which is {floor} before anything else is \
                 seated. A registered plane is an adapter bank, written at a store \
                 offset reserved at load, so it cannot be moved to another tier — \
                 every OTHER dense plane in this plan can, and the budget already \
                 gave them up (alto streaming §2, wave D2b). Raise it to at least \
                 {floor}, or state `None`.",
                found.len(),
            )));
        }

        // ── THE RANKING, SEATED WHOLE OR NOT AT ALL, ACROSS THREE TIERS.
        //
        //    Each group is offered the device while what is left of the device
        //    budget holds one whole, then the pinned host while what is left
        //    of the host budget does, then the mapping, which has no budget to
        //    run out of. There is no "seat half of it": the module header
        //    states the reason, and it is the whole of why this loop is over
        //    GROUPS.
        //
        //    **THE DENSE BANKS' PINNED COPY IS RESERVED FIRST, AT ITS WORST
        //    CASE.** A streamed dense bank pins EVERY expert, and whether it
        //    streams is decided below this loop — so the host budget is
        //    charged for it here as though it does. The conservatism can only
        //    send a group to T2 that T1 could have held, never the reverse.
        let mut left = budget - floor;
        let dense_pin: u64 = found
            .iter()
            .map(|bank| {
                (u64::from(bank.experts) * bank.stride).next_multiple_of(crate::weights::ALIGN)
            })
            .sum();
        let mut host_left = budgets.host.map(|host| host.saturating_sub(dense_pin));
        let mut groups: Vec<GroupPlan> = Vec::new();
        let mut seated_groups: Vec<GroupPlan> = Vec::new();
        let mut pinned_of: BTreeSet<usize> = BTreeSet::new();
        let mut mapped_of: BTreeSet<usize> = BTreeSet::new();
        let mut seated = 0u64;
        let mut spill_bytes = 0u64;
        for mut group in ranking.sequence.iter().cloned() {
            if group.bytes <= left {
                left -= group.bytes;
                seated += group.bytes;
                // **A ROUTED GROUP THE STORE HOLDS IS STILL A SEAT** (B7).
                // It is not a placement — nothing below reserves anything for
                // it, because the store already did — but it is the fastest
                // rung on the ladder, and the tier cannot displace what it
                // cannot name. A spilled DENSE plane is not recorded: it has
                // no cell and no ladder.
                if group.routed {
                    group.held = Held::Device;
                    seated_groups.push(group);
                }
                continue;
            }
            let fits_host = host_left.is_none_or(|host| group.bytes <= host);
            if fits_host {
                host_left = host_left.map(|host| host - group.bytes);
                group.held = Held::Pinned;
                pinned_of.extend(group.planes.iter().map(|plane| plane.param));
            } else {
                group.held = Held::Mapped;
                spill_bytes += group.bytes;
                mapped_of.extend(group.planes.iter().map(|plane| plane.param));
            }
            groups.push(group);
        }

        // How many experts every DENSE bank seats, one number for the plan,
        // out of what the groups left behind. Monotone in `n`, and `experts`
        // is at most a few hundred, so it is walked rather than searched — and
        // walked DOWN from the whole bank so that a budget one byte under full
        // residency still lands on the largest count it can hold.
        let slack = slots + left;
        let mut resident = 0u32;
        for n in (1..=experts).rev() {
            let want: u64 = found
                .iter()
                .map(|bank| (u64::from(n) * bank.stride).next_multiple_of(crate::weights::ALIGN))
                .sum();
            if want <= slack {
                resident = n;
                break;
            }
        }
        debug_assert!(
            found.is_empty() || resident >= 1,
            "the floor check above proved one slot fits"
        );

        let device_bytes = dense
            + seated
            + found
                .iter()
                .map(|bank| {
                    (u64::from(resident) * bank.stride).next_multiple_of(crate::weights::ALIGN)
                })
                .sum::<u64>();
        // **A DENSE BANK THE BUDGET HOLDS WHOLE IS NOT A STREAMED BANK.** It
        // can happen the moment a plan has both shapes in it — the packed
        // groups gave back enough for every expert of every dense bank — and
        // seating it as a tier anyway would pin a second copy of the bank and
        // publish a table every entry of which names the slab. So the plan
        // says what is true: those banks are held whole, like any other plane.
        let banks: Vec<BankPlan> = match resident < experts {
            true => found
                .into_iter()
                .map(|bank| BankPlan { resident, ..bank })
                .collect(),
            false => Vec::new(),
        };
        let resident_of = banks.iter().map(|bank| (bank.param, resident)).collect();
        // **THE HOST TIER HOLDS EVERY EXPERT, NOT ONLY THE MISSING ONES.**
        // Pinned is the AUTHORITATIVE copy and the device slab is a cache over
        // it, which is what makes a demotion free: expert weights are
        // read-only, so evicting one is a table entry pointing back at bytes
        // that were never stale. The alternative — pinning only the
        // non-resident experts — would make every promotion a demotion's
        // write-back and would put the checkpoint back on the fire path's
        // horizon.
        //
        // A streamed GROUP is the degenerate case of the same sentence: the
        // pinned copy is the only copy, and the kernel reads it where it is.
        //
        // A MAPPED group owes the pinned tier nothing at all: its bytes were
        // on disk before this load began and stay there.
        //
        // **EVERY SPAN IS ALIGNED, BANKS INCLUDED** (§M.3): `host_walk` rounds
        // each bank's span up to `weights::ALIGN` and this does the same
        // arithmetic in the same order, so `host_demand` and `host_image` are
        // one number and every T1 image starts where the store and the mapping
        // would have started it.
        let host_bytes: u64 = banks
            .iter()
            .map(|bank| {
                (u64::from(bank.experts) * bank.stride).next_multiple_of(crate::weights::ALIGN)
            })
            .sum::<u64>()
            + groups
                .iter()
                .filter(|group| group.held == Held::Pinned)
                .map(|group| group.bytes)
                .sum::<u64>();
        Ok(Plan {
            banks,
            groups,
            seated: seated_groups,
            resident_of,
            pinned_of,
            mapped_of,
            device_bytes,
            host_bytes,
            spill_bytes,
        })
    }

    /// Does this load stream anything — a dense bank's experts, or a packed
    /// bank's whole group?
    #[must_use]
    pub fn streams(&self) -> bool {
        !self.banks.is_empty() || !self.groups.is_empty()
    }

    /// The DENSE banks it streams expert by expert, in param order.
    #[must_use]
    pub fn banks(&self) -> &[BankPlan] {
        &self.banks
    }

    /// The PACKED banks it holds whole on the pinned tier, in param order.
    #[must_use]
    pub fn groups(&self) -> &[GroupPlan] {
        &self.groups
    }

    /// **The ROUTED packed banks the DEVICE STORE holds**, in param order —
    /// every one of them [`Held::Device`], and none of them in
    /// [`Plan::groups`]. See the field's own doc for why the two lists are
    /// two lists.
    #[must_use]
    pub fn seated(&self) -> &[GroupPlan] {
        &self.seated
    }

    /// **Does the PINNED tier hold this param whole?** — what
    /// `weights::resident` consults to point its handle at T1 bytes.
    ///
    /// True for every plane of every group held [`Held::Pinned`] and false for
    /// everything else, including a streamed dense bank (whose slab IS in the
    /// store) and a group held on T2.
    #[must_use]
    pub fn pinned(&self, param: usize) -> bool {
        self.pinned_of.contains(&param)
    }

    /// **Does the MAPPED artifact hold this param whole?** — T2, and what
    /// points a handle into the mapping.
    #[must_use]
    pub fn mapped(&self, param: usize) -> bool {
        self.mapped_of.contains(&param)
    }

    /// **Does anything other than the device store hold this param whole?**
    ///
    /// The one bit `weights::places` consults to reserve NOTHING for a plane
    /// in the store, and the landing sink to send its bytes somewhere else.
    /// `pinned || mapped`, stated once so the two readers cannot drift.
    #[must_use]
    pub fn streamed_whole(&self, param: usize) -> bool {
        self.pinned(param) || self.mapped(param)
    }

    /// **What this plan demands of tier T2**, in bytes: the packed groups
    /// neither budget holds, to be read from the mapped artifact.
    ///
    /// Zero for every load that fits its budgets. Non-zero is what
    /// [`Residency::admit_tiers`](engine::load::Residency::admit_tiers) has to
    /// be told about, because a spilled load with no source is the one thing
    /// this shell cannot serve.
    #[must_use]
    pub fn spill_demand(&self) -> u64 {
        self.spill_bytes
    }

    /// How many experts of `param` the device slab seats, or `None` for a
    /// param that is held whole — which is every param of a full-residency
    /// load and every dense plane of a streamed one.
    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        self.resident_of.get(&param).copied()
    }

    /// **What this plan demands of tier T0**, in bytes — what
    /// [`Residency::admit`](engine::load::Residency::admit) is
    /// asked with, and what the device store will actually occupy.
    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.device_bytes
    }

    /// **What this plan demands of tier T1**, in bytes. Zero for a
    /// fully-resident load: it holds no pinned copy of anything, exactly as
    /// dev's `place_all()` allocates no host tier.
    #[must_use]
    pub fn host_demand(&self) -> u64 {
        self.host_bytes
    }

    /// **WHERE EVERY BYTE OF THE PINNED TIER GOES**, one quad per span:
    /// `(param, host_at, bytes, reserved)`, in the order [`Tier::open`] seats
    /// them — every dense bank, then every plane of every group held
    /// [`Held::Pinned`].
    ///
    /// **ONE STATEMENT, THREE READERS** (§K.2, §M.3). It was the offset walk
    /// inlined in [`Tier::open`], and it is extracted because two more callers
    /// need the same numbers and must not re-derive them: the boot's CUT reads
    /// these quads to route each image out of the serving artifact into the
    /// pinned tier, and the background refill reads them again to lay the
    /// page-locked copy out the same way. A walk written three times is three
    /// chances for a tier to be seated at offsets nobody else believes.
    ///
    /// `bytes` is what the checkpoint publishes into the span and `reserved`
    /// is what the tier gives it — the same pair `weights::Place` carries, and
    /// they differ by up to [`weights::ALIGN`](crate::weights::ALIGN) - 1 bytes
    /// of padding no landing writes. A dense bank's two numbers are equal: its
    /// span is `experts * stride` exactly, and the alignment it needs is the
    /// one the seats before it already left it on.
    #[must_use]
    pub fn host_layout(&self) -> Vec<(u64, u64, u64, u64)> {
        self.host_walk().0
    }

    /// **How many bytes the pinned tier allocates** — where [`host_layout`]'s
    /// walk ends.
    ///
    /// Not always the last span's end: a plan whose groups are ALL on the
    /// mapping still rounds up to [`weights::ALIGN`](crate::weights::ALIGN)
    /// after its banks, because the walk aligns before it looks at what the
    /// groups are held on. So the total is the walk's own answer rather than
    /// arithmetic on its last entry, and [`Tier::open`] allocates exactly it.
    ///
    /// [`host_layout`]: Plan::host_layout
    #[must_use]
    pub fn host_image(&self) -> u64 {
        self.host_walk().1
    }

    /// The walk itself: the spans and where they end. Private, because a
    /// caller that wants one of the two is asking a question with a name.
    fn host_walk(&self) -> (Vec<(u64, u64, u64, u64)>, u64) {
        let mut out = Vec::with_capacity(self.banks.len() + self.groups.len());
        let mut at = 0u64;
        for bank in &self.banks {
            // **THE WHOLE BANK, NOT THE MISSING EXPERTS.** Pinned is the
            // authoritative copy and the slab is a cache over it — `Plan::cut`
            // argues it — so the span is every expert of the bank and the
            // landing writes all of it.
            //
            // **AND IT IS PADDED TO `weights::ALIGN` LIKE EVERY OTHER IMAGE**
            // (§M.3). It used to be `experts * stride` exactly, with ONE
            // rounding after all the banks to line the groups up behind them —
            // which left a bank's span aligned only when its shapes happened
            // to make it so. The serving artifact carries one image per span
            // and cuts it onto whichever rung the budget picks, and the whole
            // reason one image can serve all three rungs is that every rung
            // gives it the same alignment. So the padding is per bank, here,
            // and `Ranking::images` does the same arithmetic in the same
            // order.
            let span = u64::from(bank.experts) * bank.stride;
            let reserved = span.next_multiple_of(crate::weights::ALIGN);
            out.push((bank.param as u64, at, span, reserved));
            at += reserved;
        }
        for group in &self.groups {
            if group.held != Held::Pinned {
                continue;
            }
            for plane in &group.planes {
                out.push((plane.param as u64, at, plane.bytes, plane.reserved));
                at += plane.reserved;
            }
        }
        (out, at)
    }

    /// **WHERE EVERY BYTE OF THE MAPPED SECTION GOES**, one quad per plane:
    /// `(param, offset, bytes, reserved)`, in the order [`Tier::open`] seats
    /// them — every plane of every group held [`Held::Mapped`], in plan order
    /// (§K.6-T4).
    ///
    /// [`Plan::host_layout`]'s twin one tier down, and it exists for the same
    /// reason: the tier artifact's writer transcribes these quads into the
    /// MAPPED section's index run, and a section whose index disagrees with
    /// the walk that filled it is the one failure a digest cannot catch. The
    /// offsets are relative to the section's own base, which is where
    /// [`tier::Artifact::plane`] resolves them from.
    ///
    /// **AND THE ALIGNMENT IS THE STORE'S**, not the section's: a kernel that
    /// reinterprets an mxfp4 code plane as 32-bit words reads an aligned
    /// address out of the mapping exactly as it does out of the store, and
    /// [`weights::ALIGN`](crate::weights::ALIGN) is what makes that true on
    /// all three tiers. The section as a whole starts on
    /// [`tier::TIER_ALIGN`](crate::weight_cache::tier::TIER_ALIGN), which the
    /// format arranges and this walk knows nothing about.
    ///
    /// [`tier::Artifact::plane`]: crate::weight_cache::tier::Artifact::plane
    #[must_use]
    pub fn mapped_layout(&self) -> Vec<(u64, u64, u64, u64)> {
        self.mapped_walk().0
    }

    /// **How many bytes this cut leaves on the mapping** — where
    /// [`mapped_layout`]'s walk ends.
    ///
    /// Zero for every plan that spills nothing, which is the plan every load
    /// before wave W-1 had. Larger than [`Plan::spill_demand`] by the padding
    /// between the planes: the demand is what the groups PUBLISH and this is
    /// what their images RESERVE, the same pair the other two rungs carry.
    ///
    /// [`mapped_layout`]: Plan::mapped_layout
    #[must_use]
    pub fn mapped_image(&self) -> u64 {
        self.mapped_walk().1
    }

    /// The walk itself, for [`Plan::host_walk`]'s reason.
    ///
    /// [`Plan::host_walk`]: Plan::host_walk
    fn mapped_walk(&self) -> (Vec<(u64, u64, u64, u64)>, u64) {
        let mut out = Vec::new();
        let mut at = 0u64;
        for group in &self.groups {
            if group.held != Held::Mapped {
                continue;
            }
            for plane in &group.planes {
                out.push((plane.param as u64, at, plane.bytes, plane.reserved));
                at += plane.reserved;
            }
        }
        (out, at)
    }
}

/// **The routed banks a trace declares, with the uniformity proof checked** —
/// the dense ones, seated expert by expert, and the packed ones, seated whole.
///
/// A bank is a param some `Linear::Moe*Select*` op reads at its `bank` port —
/// stated by the OP and not by a naming convention, the same rule
/// `weights::banks` follows for the adapter axis. The scan is over
/// `Trace::nodes` because that is where the reading is; `Trace::params` says
/// only what exists.
///
/// **WHICH OF THE TWO A BANK IS, IS ALSO THE OP'S TO SAY.**
/// `MoeMatmulSelect` reads one dense handle and its kernel dereferences the
/// indirection table; the two quantized twins read a split-plane bank whose
/// kernel does not. So the op name is what sorts a param into a `BankPlan` or
/// a `GroupPlan`, and neither a dtype nor a suffix is consulted.
fn routed(
    trace: &Trace,
    planes: &Attachments,
    bytes: &[u64],
) -> Result<(Vec<BankPlan>, Vec<GroupPlan>)> {
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut banks: Vec<BankPlan> = Vec::new();
    let mut groups: Vec<GroupPlan> = Vec::new();
    let mut arity: Option<u32> = None;
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        // THE THREE OPS, NAMED. `MoeMatmulSelect` reads one dense handle; the
        // two quantized twins read a split-plane bank, and they are named here
        // so that the pairing is looked up for exactly the params that need
        // one.
        let (bank, dense) = match op {
            Linear::MoeMatmulSelect { bank, .. } => (*bank, true),
            Linear::MoeMatmulSelectBias { bank, .. } | Linear::MoeMatmulSelectQuant { bank, .. } => {
                (*bank, false)
            }
            _ => continue,
        };
        let at = weight_of(trace, bank)?;
        if !seen.insert(at) {
            continue;
        }
        let param = &trace.params[at];
        // ── THE UNIFORMITY PROOF, OFF THE PLAN (dev's, restated). Both kinds
        //    are asked, because both state an expert count and one arity
        //    covers the plan either way.
        let experts = u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(0);
        if experts == 0 || param.shape.len() < 2 {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is read as a routed expert bank and does not declare \
                      `[experts, ...]`; a slot stride cannot be divided out of it",
            });
        }
        match arity {
            None => arity = Some(experts),
            Some(first) if first != experts => {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a routed expert bank whose expert count differs from an \
                          earlier bank of the same plan; one residency decision covers \
                          the plan, and two arities would make it two decisions",
                });
            }
            Some(_) => {}
        }
        if dense {
            let plane = bytes[at];
            if plane == 0 || plane % u64::from(experts) != 0 {
                return Err(Fault::Param {
                    name: param.name.clone(),
                    why: "is a routed expert bank whose bytes do not divide by its expert \
                          count — the experts of one bank are not equal, and the slot \
                          arithmetic the tier does would be wrong rather than refused",
                });
            }
            banks.push(BankPlan {
                param: at,
                name: param.name.clone(),
                experts,
                resident: experts,
                stride: plane / u64::from(experts),
            });
            continue;
        }
        // ── A PACKED BANK IS A GROUP, AND THE GROUP IS THE LOAD PLAN'S.
        //    An attachment this plan does not carry is the one thing that
        //    cannot be worked around: seating the codes and leaving the
        //    exponents to be found by name is how a bank reads another bank's
        //    factors, which computes and is wrong. Refused by name.
        let Some(companions) = planes.get(&at) else {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is read at a QUANTIZED routed matmul's `bank` port and the load \
                      plan pairs no scales plane with it. A split-plane bank is codes \
                      AND factors, both indexed by the same expert id, and this shell \
                      seats them as one group or not at all — a contract states the \
                      pair with `TensorContract::scaling`",
            });
        };
        let mut all: Vec<usize> = std::iter::once(at).chain(companions.iter().copied()).collect();
        all.sort_unstable();
        all.dedup();
        let planes: Vec<GroupPlane> = all
            .iter()
            .map(|at| GroupPlane {
                param: *at,
                bytes: bytes[*at],
                reserved: bytes[*at].next_multiple_of(crate::weights::ALIGN),
            })
            .collect();
        let total = planes.iter().map(|plane| plane.reserved).sum();
        if total == 0 {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is a split-plane quantized expert bank whose planes reserve no \
                      bytes at all; a group with nothing in it is a bank the store \
                      would seat at address zero",
            });
        }
        groups.push(GroupPlan {
            param: at,
            name: param.name.clone(),
            planes,
            experts,
            bytes: total,
            // The scan does not decide residency; `Plan::of`'s walk does, and
            // it overwrites this. Pinned is the resting value because it is
            // the tier that needs no file to exist.
            held: Held::Pinned,
            routed: true,
        });
    }
    banks.sort_by_key(|bank| bank.param);
    groups.sort_by_key(|group| group.param);
    Ok((banks, groups))
}

/// The `Trace::params` row a value id names, or a refusal.
fn weight_of(trace: &Trace, id: ValueId) -> Result<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Ok(*w as usize),
        _ => Err(Fault::Param {
            name: format!("value {}", id.0),
            why: "is read at a routed matmul's `bank` port and is not a weight; a bank \
                  is a `Def::Weight` row and nothing else resolves there",
        }),
    }
}

/// One streamed bank, seated: where its bytes are on both tiers, and which
/// expert is in which device slot right now.
#[derive(Debug)]
struct Seat {
    param: usize,
    name: String,
    experts: u32,
    resident: u32,
    stride: u64,
    /// Byte offset of expert 0 inside the PAGE-LOCKED image — where the
    /// landing sink writes and where the install will re-form `serving_at`.
    host_at: u64,
    /// **THE ADDRESS EXPERT 0 IS READ FROM RIGHT NOW** (§L.1, at §M's
    /// layout): `host.device() + host_at` for an eager seat, and the bank
    /// image's own address inside the serving artifact while a deferred seat's
    /// window is open.
    ///
    /// It is a per-span base and not one number for the whole tier, and §M is
    /// why. The artifact holds one image per plane in a budget-free order, so
    /// the images a given budget puts on T1 are a SUBSET of that order and not
    /// a contiguous run of it — there is no single base address the pinned
    /// offsets could be added to. What §L.2's install promised is unchanged:
    /// every one of these moves inside one `drained`/`ready` bracket, followed
    /// by one `publish_all` and one `publish_cells`, so a cell caught
    /// mid-flight still names one of two addresses of the same verified bytes.
    serving_at: u64,
    /// Index of expert 0's entry, in entries.
    entry_at: usize,
    /// Index of expert 0's counter, in counters.
    counter_at: usize,
    /// Device address of slot 0 of the slab.
    slab: u64,
    /// `expert -> slot`, or `None` for an expert that lives in pinned memory.
    slot_of: Vec<Option<u32>>,
    /// `slot -> expert`.
    in_slot: Vec<u32>,
}

/// **One plane of a packed group, seated on the pinned tier**: where its bytes
/// are, and nothing else.
///
/// There is no slot map and no counter row here, and the module header says
/// why: the packed select kernel indexes each plane by the routing vector's
/// own expert id and reads no table, so a group has no per-expert state to
/// keep. What it has is an address, and the address does not move for the
/// life of the load.
#[derive(Debug)]
struct Whole {
    param: usize,
    /// Byte offset of this plane inside the PAGE-LOCKED image.
    host_at: u64,
    /// **THE ADDRESS THIS PLANE IS READ FROM RIGHT NOW** — see
    /// [`Seat::serving_at`].
    serving_at: u64,
}

/// **One plane of a packed group, seated on T2** — an address inside the
/// mapped artifact, and nothing else.
///
/// The address is a HOST pointer that the GPU dereferences directly: on a
/// device reporting `pageableMemoryAccess` an unregistered mapping is a valid
/// device pointer, and the first touch of each page is the NVMe read. Nothing
/// was copied to produce it and nothing is page-locked by it.
#[derive(Debug)]
struct Mapped {
    param: usize,
    /// The plane's address inside the mapping — absolute, because a mapping
    /// has no base a weight row could be rebased against.
    at: u64,
    /// What the artifact says the plane holds.
    bytes: u64,
}

/// **ONE PACKED GROUP, AND WHICH RUNG OF THE LADDER IT IS ON RIGHT NOW**
/// (alto streaming §3 item 3, wave B7).
///
/// [`Whole`] and [`Mapped`] are the LOAD's answer — where the plan put a
/// group's planes and where the weight table pointed at them. This is the
/// LIVE answer, and the difference between the two is the whole of what this
/// wave added: a group's tier used to be fixed at load, because the two plane
/// bases were kernel parameters and a captured graph holds its parameters
/// forever (article 7). Now the launch reads a CELL instead, the cell is data,
/// and this is what the host knows the cell says.
#[derive(Debug)]
struct Group {
    /// Index into `Trace::params` of the CODE plane — the group's own name.
    param: usize,
    name: String,
    experts: u32,
    /// Every plane, in the plan's order: code plane first, then companions.
    planes: Vec<GroupPlane>,
    /// Index of this group's cell, in `cells` — also its counter's index.
    cell_at: usize,
    /// Where each plane is RIGHT NOW, in `planes` order. What the cell says.
    at: Vec<u64>,
    /// Which tier those addresses are on.
    held: Held,
    /// **Where each plane is IN THE ARTIFACT** — the rung a demotion falls
    /// back to, and the only one that is always there because it is a file.
    ///
    /// Empty when this load opened no artifact, or when the artifact does not
    /// carry every plane of this group. A group with no backing can be
    /// promoted (its bytes are read, never written) but never DEMOTED, so it
    /// is never the occupant a swap displaces. That is the honest statement of
    /// the precondition: **the ladder needs somewhere to put what it moves,
    /// and streaming §0 already named the file.**
    backing: Vec<u64>,
    /// Which berth it occupies, or `None` for a group read where it lies.
    berth: Option<usize>,
    /// **When it last changed rung**, on [`Tier::tick`]'s monotone clock; `0`
    /// for a group still where the plan seated it.
    ///
    /// The vote never reads it — a strict-improvement rule needs no tiebreak —
    /// but [`Tier::promote_now`] does, and for a reason worth stating: the
    /// door has the vote's clause struck out, so with a uniform demand every
    /// berth looks equally good and it would displace the group it just
    /// promoted. Least-recently-settled is the tiebreak that makes a sequence
    /// of forced rungs walk ACROSS the berths rather than in and out of one.
    settled: u64,
}

/// **One physical region a group's planes can be copied INTO** — a device
/// store region a T0 group was seated in, or a span of the pinned tier a T1
/// group was seated in.
///
/// T2 is deliberately NOT a berth. The mapping is a read-only view of a file:
/// nothing is copied into it, and a group demoted there simply points back at
/// its own bytes, which were never anywhere else. So the ladder's two upper
/// rungs are berths, its bottom rung is the file, and a swap is always
/// *"the occupant goes back to the file, the candidate comes up"*.
#[derive(Debug)]
struct Berth {
    /// Which rung. [`Held::Device`] or [`Held::Pinned`], never [`Held::Mapped`].
    tier: Held,
    /// One address per plane, in plane order.
    at: Vec<u64>,
    /// **The reserved bytes of each plane** — the SHAPE a group must match to
    /// take this berth.
    ///
    /// Exactly, plane for plane. A berth is sized for the group the plan put
    /// in it, and a group with different plane bytes taking it would either
    /// run off the end or leave a hole the next plane's reader walks into.
    /// gpt-oss-20b's 48 groups are two shapes — a gate/up bank and a down bank
    /// — so this partitions the berths into two interchangeable classes and
    /// refuses everything across them, which is what it should do.
    shape: Vec<u64>,
    /// Which group is in it, by index into `Tier::groups`.
    holds: Option<usize>,
}

/// **The one swap in flight** — a promotion between its two halves.
///
/// A group's bytes are hundreds of megabytes, so the copy that moves them is
/// the one thing on this path that cannot be finished inside a gap. It is
/// therefore SPLIT, and the split is what keeps article 2:
///
/// ```text
/// gap n    notify: [wait drained] the occupant's cell -> the file
///                  [record ready] ─────────── the next fire waits HERE
///                  the candidate's planes -> the berth   (~265 MiB)
///                  [record landed]
/// gap n+k  notify: [wait drained] the candidate's cell -> the berth
///                  [record ready] ─────────── the next fire waits HERE
/// ```
///
/// The next fire waits on `ready`, which is recorded BEFORE the bulk copy is
/// enqueued — so what a fire ever waits for is two sixteen-byte writes, and
/// the copy that actually costs something is behind it on the stream and
/// gates nothing. Between the two gaps the berth is referenced by NOBODY: the
/// occupant's cell already names the file, and the candidate's still names
/// wherever it was.
#[derive(Debug, Clone, Copy)]
struct Swap {
    berth: usize,
    group: usize,
}

/// Which half of a [`Swap`] this gap takes.
#[derive(Debug, Clone, Copy)]
enum Step {
    /// The berth's occupant goes back to the file, and the bulk copy follows
    /// on the notify stream behind the event the next fire waits on.
    Open(Swap),
    /// The bytes have landed; the candidate's cell arrives in the berth.
    Close(Swap),
}

/// **The two device addresses one packed GROUP hands its select kernel.** The
/// shell's spelling of `kernels_cuda::linear::moe::GroupSeat`, kept here for
/// the reason [`Handles`] is: `run.rs` carries a weight row without naming a
/// kernel type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GroupHandles {
    /// The group's 16-byte `(codes, scales)` base cell.
    pub cell: u64,
    /// The group's `u32` usage counter.
    pub hits: u64,
}

/// **The two device addresses one bank hands its select kernel.** The shell's
/// spelling of `kernels_cuda::linear::moe::ExpertTable`, kept here so that
/// `run.rs` does not have to name a kernel type to carry a weight row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handles {
    /// The bank's `expert_id -> base address` table.
    pub table: u64,
    /// The bank's per-expert usage counters.
    pub counts: u64,
}

/// **What one bank's residency looks like from outside** — the accessor a gate
/// reads, and the only observable a promotion has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankResidency {
    /// The param's name.
    pub name: String,
    /// Its expert count.
    pub experts: u32,
    /// How many of them the slab seats.
    pub slots: u32,
    /// Which expert is in which slot, ascending by slot.
    pub in_slot: Vec<u32>,
    /// Every expert's cumulative usage count, as the last settled readback
    /// carried it out.
    pub hits: Vec<u32>,
    /// **Which tier holds it**, for a packed group seated whole — `None` for a
    /// dense bank, whose experts are split between the slab and the pinned
    /// tier and whose answer is `in_slot` rather than one word.
    pub held: Option<Held>,
}

/// **WHERE THE PINNED TIER'S BYTES COME FROM**, decided by the caller of
/// [`Tier::open`] and by nothing else.
///
/// One argument with three answers, because the allocation it decides is the
/// only one in this file measured in TENS OF GIGABYTES and each answer costs
/// something different. It was a `bool` for two of them (§K.4, phase T-3) and
/// the third is what §L adds; the enum is here rather than a second parameter
/// so that the three remain exclusive by construction.
#[derive(Debug)]
pub enum Fill {
    /// **The cold load.** [`Pinned::mapped`]: zeroed here, then written span
    /// by span by the landing sink as the checkpoint finalizes.
    ///
    /// The zeros are not decoration. The sink writes each plane's PUBLISHED
    /// bytes and this tier seats each plane at `reserved` — those bytes
    /// rounded up to [`weights::ALIGN`](crate::weights::ALIGN) — so the
    /// padding at the end of every span is written by nothing. Zeroed, they
    /// are a deterministic image the artifact can carry; unzeroed, they are
    /// whatever the allocator handed out, which the writer would hash and
    /// publish.
    Cold,
    /// **The warm restore.** [`Pinned::mapped_uninit`]: not zeroed, because
    /// the caller holds a serving artifact carrying every image this tier
    /// seats, and `Plan::host_layout` tiles the allocation — so all of it is
    /// read in before anything reads a byte out (§K.4).
    ///
    /// Zeroing tens of gigabytes of page-locked memory costs tens of seconds
    /// of memory bandwidth on every boot, and it is pure loss on the one boot
    /// whose promise is that it does not touch the bytes twice. A caller that
    /// takes this and then cannot keep the promise owes this tier
    /// [`Tier::zero_host`] before the cold load runs.
    ///
    /// # A TRAP FOR THE §M-4d PORT, WRITTEN DOWN BEFORE ANYONE WALKS INTO IT
    ///
    /// This arm is about to be re-pointed at the model's own `.zt`, and the
    /// two sources do not cover the same bytes.
    ///
    /// A tier image is a `reserved` span — the plane's bytes ROUNDED UP to
    /// [`weights::ALIGN`](crate::weights::ALIGN) — so a restore from one
    /// writes the padding at the end of every plane, and `Plan::host_layout`
    /// tiles the whole allocation with those padded spans. That is what makes
    /// `mapped_uninit` safe here: every byte is written before any is read.
    ///
    /// `checkpoint::file::serve::Fill` writes **exactly the part's published
    /// length** and deliberately not the padding — the block table tiles the
    /// DECODED size (departure #1), so the padding is not hashed and is not
    /// read back. A fill built naively out of `serve::Fill` therefore leaves
    /// every inter-plane gap of a NOT-ZEROED allocation holding whatever the
    /// allocator last had there.
    ///
    /// Nothing would notice for a long time. No kernel reads a plane past its
    /// `bytes`, so the model serves correctly; the gaps surface only when
    /// something HASHES the image — a later write-back, a verify, a deferred
    /// seat's install comparing what it built against what it served.
    ///
    /// So the port owes one of three, and must say which: write the padding
    /// explicitly from [`serving::padded_spans`], zero the allocation first
    /// and give up this arm's whole reason to exist, or narrow the promise so
    /// the image is never hashed. The first is the cheap one — the padding is
    /// a spec MUST of `0x00` in the file and a `memset` per plane here.
    Restored,
    /// **THE DEFERRED SEAT** (§L.1, phase L-1): no page-locked image at all,
    /// yet. The tier serves each T1 image out of the artifact where it lies,
    /// and a background thread builds the page-locked copy behind it.
    ///
    /// The artifact travels INTO the tier because the mapping has to outlive
    /// the load: every T1 address this tier hands out is a pointer inside it
    /// until the install, and a `Tier` that served out of a mapping somebody
    /// else dropped is the one failure mode worth designing against.
    ///
    /// **The same bytes, at their own addresses.** §M's file holds one image
    /// per plane in a budget-free order, so the images this budget puts on T1
    /// are a subset of that order and not a run of it — each span carries its
    /// own serving address (see [`Seat::serving_at`]) and the install re-forms
    /// all of them at once rather than swapping one base.
    ///
    /// The caller owes two things: every image this seat will serve must be
    /// VERIFIED before a kernel is pointed at it (§L.3 — there is no
    /// first-touch hook, so the alternative to verifying up front is
    /// unretractable tokens), and the device must report
    /// `pageableMemoryAccess`, which is [`pageable_access`] and is the same
    /// mechanism T2 already stands on.
    Deferred(crate::weight_cache::tier::Artifact),
}

/// **THE BACKGROUND PAGE-LOCK-AND-FILL, WHILE IT IS OUTSTANDING** (§L.4).
///
/// One thread and one channel. The thread binds the device, maps the whole
/// pinned image, reads every T1 image of the artifact into its own span —
/// verifying each block against the table as the bytes land, because a
/// background copy does not get to trust the boot's verification of a file it
/// re-opened by path — page-locks the result and sends it. A thread that fails
/// sends NOTHING: it has already counted what it found, said it out loud with
/// the remedy, and LEFT THE FILE WHERE IT IS (§M.4); a seat whose refill
/// failed keeps serving the verified bytes it has been serving all along.
///
/// **NOTHING OF THE TIER CROSSES.** The closure carries a `PathBuf`, a
/// [`Head`](crate::weight_cache::tier::Head), a length and a device ordinal;
/// the allocation it makes is unreachable from this side until it comes back
/// over the channel, which is a stronger discipline than the one `Pinned`'s
/// own `Send` argument asks for.
#[derive(Debug)]
struct Refill {
    /// The thread, until [`Refill`] is dropped or the image is taken.
    filling: Option<std::thread::JoinHandle<()>>,
    /// The one message it will ever send.
    filled: std::sync::mpsc::Receiver<Pinned>,
    /// When the window opened. See [`Observed::window_ms`](
    /// crate::weight_cache::tier::Observed::window_ms).
    began: std::time::Instant,
}

/// What a [`Refill`] answers when it is asked between two fires.
enum Filled {
    /// Still page-locking, reading or hashing. Serving continues on T2.
    Waiting,
    /// The image is here, verified and page-locked. Install it.
    Ready(Pinned),
    /// The thread is finished and sent nothing, which is the only way it says
    /// it failed. It has counted and named whatever it found; serving
    /// continues, and this tier never promotes.
    Refused,
}

impl Refill {
    /// Ask, without waiting. See [`Filled`].
    fn poll(&mut self) -> Filled {
        use std::sync::mpsc::TryRecvError;

        match self.filled.try_recv() {
            Ok(host) => Filled::Ready(host),
            Err(TryRecvError::Empty) => Filled::Waiting,
            Err(TryRecvError::Disconnected) => Filled::Refused,
        }
    }

    /// **Wait for it.** The gate's half of [`Refill::poll`]: joins the thread
    /// and answers what it sent, which is `None` for a fill that failed.
    fn settle(mut self) -> Option<Pinned> {
        if let Some(filling) = self.filling.take() {
            let _ = filling.join();
        }
        self.filled.try_recv().ok()
    }

    /// How long this window has been open, in milliseconds.
    fn window_ms(&self) -> u64 {
        u64::try_from(self.began.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

impl Drop for Refill {
    /// **THE THREAD IS JOINED, NOT DETACHED** (§L, hazard H6).
    ///
    /// A detached refill outlives the tier that armed it and then page-locks
    /// tens of gigabytes against a process that has no use for them and no
    /// handle to free them with — `cudaHostAlloc` on a dead deployment, held
    /// until the process exits. A wait at teardown costs the seconds the read
    /// had left; a leaked sixty-four-gigabyte page-lock costs the machine.
    /// Between the two the wait is not close.
    ///
    /// It is usually free: the ordinary path takes the image out through
    /// [`Refill::poll`] or [`Refill::settle`], and by then the thread has
    /// already returned.
    fn drop(&mut self) {
        if let Some(filling) = self.filling.take() {
            let _ = filling.join();
        }
    }
}

/// **The tier**: pinned bytes, a device slab, a table between them, and the
/// counters that decide what moves.
#[derive(Debug)]
pub struct Tier {
    plan: Plan,
    /// T1: every expert of every streamed bank.
    ///
    /// **ZERO-LENGTH WHILE A DEFERRED SEAT IS OPEN** ([`Fill::Deferred`]) —
    /// the bytes are in `image` until the install swaps this for the
    /// page-locked copy the refill built.
    host: Pinned,
    /// The background fill, while the seat is deferred and the window open.
    /// `None` for every eager boot, and from the install on. See [`Refill`].
    refill: Option<Refill>,
    /// **THE MAPPING A DEFERRED SEAT SERVES T1 OUT OF**, held for the tier's
    /// whole life — see [`Fill::Deferred`].
    ///
    /// **AND NOT DROPPED AT THE INSTALL**, which would be the tidy thing and
    /// is the unsound one: unmapping is immediate and a kernel enqueued
    /// before the install is not. The install's two events order the STREAMS
    /// against each other; nothing on the host knows when the last fire that
    /// read a mapped plane retired, so the address range stays reserved. The
    /// pages behind it are clean file pages the kernel reclaims under
    /// pressure on its own.
    image: Option<crate::weight_cache::tier::Artifact>,
    /// The device-resident indirection tables, one row of `experts` entries
    /// per bank, concatenated. **The only mutable address surface** (article
    /// 7), and its own address never moves.
    table: Buffer,
    /// A pinned mirror of `table`, and the SOURCE of every entry update: the
    /// host writes a word here and the notify stream copies it across, so the
    /// bytes an in-flight copy reads belong to this tier for the load's whole
    /// life rather than to a temporary.
    shadow: Pinned,
    /// The per-expert usage counters the select kernels `atomicAdd` into.
    counts: Buffer,
    /// A pinned mirror of `counts`, filled by an asynchronous D2H the settle
    /// side enqueues. Read on the host between fires, WITHOUT a wait: a torn
    /// read is a slightly stale hint, and a hint is all a promotion is.
    mirror: Pinned,
    seats: Vec<Seat>,
    /// Every plane of every packed group the store does not hold, at its own
    /// offset inside [`Tier::host`]. Read by the weight table AT LOAD; the
    /// live answer after that is [`Group::at`], because a group moves now.
    wholes: Vec<Whole>,
    /// **THE LADDER'S ONLY MUTABLE ADDRESS SURFACE**, and its address never
    /// moves (article 7): one 16-byte `(codes, scales)` cell per packed group,
    /// which is what the mxfp4 select loads its two plane bases out of.
    cells: Buffer,
    /// A pinned mirror of `cells` and the SOURCE of every cell write, for
    /// [`Tier::shadow`]'s reason: the bytes an in-flight copy reads belong to
    /// this tier for the load's life rather than to a temporary. **The pair is
    /// filled here before the copy is issued, which is what makes a torn pair
    /// unconstructible** — no cell state names one group's codes beside
    /// another's exponents, because the sixteen bytes are written as one.
    cell_shadow: Pinned,
    /// One `u32` per packed group, `atomicAdd`ed by the select once per routed
    /// row per fire.
    group_counts: Buffer,
    /// A pinned mirror of `group_counts`, filled by the same asynchronous D2H
    /// [`Tier::mirror`] rides, and read the same way: without a wait, because
    /// a stale hint is all a promotion ever needed.
    group_mirror: Pinned,
    /// Every ROUTED packed group of this load — the ones the store holds and
    /// the ones it does not — in plan order.
    groups: Vec<Group>,
    /// Every region a group can be copied into. See [`Berth`].
    berths: Vec<Berth>,
    /// The promotion between its two halves, or `None`. See [`Swap`].
    swap: Option<Swap>,
    /// Records on the notify stream when a swap's bulk copy is past. **Nothing
    /// waits on it on a stream** — the host asks [`Event::done`] at the next
    /// gap, which is what makes the copy free of the fire path.
    landed: Event,
    /// `(groups promoted, groups demoted, gaps a swap in flight held back)`.
    ladder: (u64, u64, u64),
    /// **Can anything be displaced at all?** True when at least one group's
    /// planes were resolved against an artifact. Decided once, at
    /// [`Tier::open`], because it is a property of the load and not of a gap.
    ladder_open: bool,
    /// A monotone clock over rung changes; see [`Group::settled`].
    tick: u64,
    /// **T2**: the mapped warm artifact, held open for the load's whole life
    /// because the weight rows point into it. `None` for every plan that
    /// spills nothing. Either file can answer — see [`Spill`].
    source: Option<Spill>,
    /// Every plane of every group T2 holds, resolved against `source` at open
    /// time — the addresses stay valid because the mapping does.
    mapped: Vec<Mapped>,
    /// Records on the compute stream: everything already enqueued there is
    /// past. What an eviction waits for before it overwrites a slot.
    drained: Event,
    /// Records on the notify stream at the end of a promotion round: the
    /// copies and the entry writes are done. What the next fire's enqueue
    /// waits for, and what the NEXT round asks before it reuses the shadow.
    ready: Event,
    /// Has `ready` ever been recorded?
    moving: bool,
    promotions: u64,
    demotions: u64,
    skipped: u64,
}

impl Tier {
    /// Open the tier `plan` describes — the pinned bytes, the tables, the
    /// counters, and the mapped artifact behind any group the two budgets
    /// spilled. Called BEFORE the checkpoint lands, because the landing writes
    /// a streamed bank's planes straight into [`Tier::host`].
    ///
    /// `source` is the T2 artifact, already opened and verified by the caller
    /// — EITHER of the two files a [`Spill`] can be, and this function does
    /// not care which: it asks the enum for a plane and seats the window it
    /// gets back. It MUST be `Some` for a plan with a non-zero
    /// [`Plan::spill_demand`] — the caller is
    /// [`Residency::admit_tiers`](engine::load::Residency::admit_tiers)'s
    /// other half and has already refused the load if it is not — and a
    /// mismatch here is [`Fault::Residency`] by name rather than a weight row
    /// pointing at nothing.
    ///
    /// # The memset, the mapping, and who is allowed to skip which
    ///
    /// `fill` says where this tier's T1 bytes come from, and it is the only
    /// thing that decides the one allocation here worth arguing about. The
    /// three answers and what each of them owes are on [`Fill`].
    ///
    /// # Errors
    ///
    /// [`Fault::OutOfMemory`] or [`Fault::Device`] for the allocations,
    /// [`Fault::Runtimeless`] without a runtime, [`Fault::Residency`] for a
    /// spilled plan with no source, a device that cannot dereference a
    /// mapping, a plane the artifact does not carry, or a [`Fill::Deferred`]
    /// artifact that carries no image at this plan's span for a plane it
    /// pins.
    pub fn open(plan: Plan, source: Option<Spill>, fill: Fill) -> Result<Tier> {
        // ── WHERE EVERY BYTE OF THIS TIER GOES, AND IT IS THE PLAN THAT
        //    SAYS SO. `Plan::host_layout` is the walk this function used to
        //    inline: every dense bank's whole span, then — after one
        //    alignment — every plane of every group held on T1. It is
        //    extracted because the tier ARTIFACT's key mixes the same quads
        //    and its writer transcribes them (§K.2), and a tier seated at
        //    offsets the file disagrees with is the one failure a digest
        //    cannot catch.
        //
        //    The two loops below consume it in the order it was produced, so
        //    the zips are index-for-index; `image()` hands the same run back
        //    out to the writer.
        let layout = plan.host_layout();
        let host_at = plan.host_image();
        let mut seats = Vec::with_capacity(plan.banks.len());
        let (mut entry_at, mut counter_at) = (0usize, 0usize);
        for (bank, span) in plan.banks().iter().zip(&layout) {
            debug_assert_eq!(span.0, bank.param as u64, "the layout walks the banks first");
            seats.push(Seat {
                param: bank.param,
                name: bank.name.clone(),
                experts: bank.experts,
                resident: bank.resident,
                stride: bank.stride,
                host_at: span.1,
                // Filled below, once the fill has said which image this tier
                // is serving out of.
                serving_at: 0,
                entry_at,
                counter_at,
                slab: 0,
                slot_of: vec![None; bank.experts as usize],
                in_slot: Vec::new(),
            });
            entry_at += bank.experts as usize;
            counter_at += bank.experts as usize;
        }
        // ── THE PACKED GROUPS, AFTER THE SEATS. Each plane is given the same
        //    alignment the device store would have given it, so a kernel that
        //    reinterprets the codes as 32-bit words reads an aligned address
        //    on this tier exactly as it does on the other.
        let planes = plan
            .groups()
            .iter()
            .filter(|group| group.held == Held::Pinned)
            .flat_map(|group| &group.planes);
        let mut wholes: Vec<Whole> = planes
            .zip(layout.get(plan.banks().len()..).unwrap_or(&[]))
            .map(|(plane, span)| {
                debug_assert_eq!(span.0, plane.param as u64, "then every pinned plane");
                Whole {
                    param: plane.param,
                    host_at: span.1,
                    serving_at: 0,
                }
            })
            .collect();
        // ── T2. The groups neither budget held, resolved against the
        //    mapping. Nothing is copied and nothing is read: `plane` hands
        //    back a window on the mapping, and its address is what the weight
        //    row will carry.
        let spilled = plan.groups().iter().any(|group| group.held == Held::Mapped);
        if spilled {
            // **THE DEVICE HAS TO BE ABLE TO DEREFERENCE A MAPPING.** Without
            // `pageableMemoryAccess` an unregistered host pointer is not a
            // device pointer, and the alternative — registering the mapping —
            // page-locks it, which is T1 under another name and would spend
            // the host RAM the budget just said it did not have. Refused by
            // name rather than served through a fallback that lies about its
            // tier.
            if !pageable_access() {
                return Err(Fault::Residency(format!(
                    "this load plans {} bytes onto the mapped tier and this device does \
                     not report `pageableMemoryAccess` (CUDA 12.2+ HMM), so a GPU touch \
                     of a mapped page cannot fault it in. The T2 arm needs it: \
                     registering the mapping instead would page-lock every byte of it, \
                     which is the pinned tier under another name and is exactly what \
                     `host_weight_budget` said this machine does not have. Raise a \
                     budget, or run on a device that reports the attribute.",
                    plan.spill_demand(),
                )));
            }
        }
        let mut mapped = Vec::new();
        for group in plan.groups() {
            if group.held != Held::Mapped {
                continue;
            }
            let Some(artifact) = source.as_ref() else {
                return Err(Fault::Residency(format!(
                    "`{}` is planned onto the mapped tier and this load opened no \
                     artifact to map it out of; `Residency::admit_tiers` refuses that \
                     before the store is reserved, so reaching here is a shell that \
                     planned a spill it never sourced",
                    group.name,
                )));
            };
            for seat in &group.planes {
                let id = u32::try_from(seat.param).unwrap_or(u32::MAX);
                // ONE GROUP PER PARAM ON THIS PLANE. A split-plane bank is
                // already two `Trace::params` rows here, so each is its own
                // index entry at plane zero — the `plane` axis of the index is
                // for a shell that puts both planes under one id, which this
                // one does not.
                let Some(bytes) = artifact.plane(id) else {
                    bump(Stat::Absent);
                    return Err(Fault::Residency(format!(
                        "`{}` is planned onto the mapped tier and the artifact at {} \
                         carries no plane {id}. {}",
                        group.name,
                        artifact.path().display(),
                        artifact.remedy(),
                    )));
                };
                if bytes.len() as u64 != seat.bytes {
                    bump(Stat::Absent);
                    return Err(Fault::Residency(format!(
                        "`{}`'s plane {id} is {} bytes in the artifact and {} in this \
                         plan; the two were written from different traces",
                        group.name,
                        bytes.len(),
                        seat.bytes,
                    )));
                }
                mapped.push(Mapped {
                    param: seat.param,
                    at: bytes.as_ptr() as u64,
                    bytes: bytes.len() as u64,
                });
                bump(Stat::Seated);
                add(Stat::Bytes, bytes.len() as u64);
            }
        }
        if !mapped.is_empty() {
            bump(Stat::Loads);
        }
        // ── THE LADDER'S ROSTER (wave B7). Every ROUTED packed group of this
        //    load, whichever rung the plan put it on: the ones the store holds
        //    (`Plan::seated`) and the ones it does not (`Plan::groups`,
        //    routed). A spilled DENSE plane is not one of them — it has no
        //    cell for a promotion to write and no select kernel that would
        //    read one — and the roster's order is param order, so two boots of
        //    one deployment number the cells the same.
        //
        //    **THE BACKING IS RESOLVED FOR ALL OF THEM, NOT JUST FOR T2.**
        //    The swap's first half sends the berth's occupant back to the
        //    file, and the occupant may be a group that has never been off the
        //    device; so what makes a group displaceable is that the artifact
        //    carries it, and that is asked here, once, of every group. A group
        //    the artifact cannot answer for keeps an empty backing and is
        //    simply never displaced — no refusal, because the LOAD is not
        //    wrong, only the ladder is shorter.
        let mut roster: Vec<&GroupPlan> = plan
            .seated()
            .iter()
            .chain(plan.groups().iter().filter(|group| group.routed))
            .collect();
        roster.sort_by_key(|group| group.param);
        let mut groups = Vec::with_capacity(roster.len());
        for (cell_at, group) in roster.into_iter().enumerate() {
            if group.planes.len() < 2 || group.planes.len() > CELL_PLANES {
                return Err(Fault::Residency(format!(
                    "`{}` is a routed packed bank of {} planes and a base cell seats \
                     two or three — codes beside factors, and an affine bank's zero \
                     points beside those, written as one word so that no state of the \
                     cell can name one group's codes and another's factors. Refused \
                     rather than seated with a plane unaddressed.",
                    group.name,
                    group.planes.len(),
                )));
            }
            let backing = match source.as_ref() {
                None => Vec::new(),
                Some(artifact) => group
                    .planes
                    .iter()
                    .map(|plane| {
                        let id = u32::try_from(plane.param).unwrap_or(u32::MAX);
                        artifact
                            .plane(id)
                            .filter(|bytes| bytes.len() as u64 == plane.bytes)
                            .map(|bytes| bytes.as_ptr() as u64)
                    })
                    .collect::<Option<Vec<u64>>>()
                    .unwrap_or_default(),
            };
            groups.push(Group {
                param: group.param,
                name: group.name.clone(),
                experts: group.experts,
                planes: group.planes.clone(),
                cell_at,
                // `land` fills both: it is the first moment the store's own
                // addresses exist, and a cell published before them would name
                // where the plan MEANT a plane to be.
                at: Vec::new(),
                held: group.held,
                backing,
                berth: None,
                settled: 0,
            });
        }
        let entries = entry_at as u64 * ENTRY;
        let counters = counter_at as u64 * COUNTER;
        let cells = groups.len() as u64 * CELL;
        let group_counters = groups.len() as u64 * COUNTER;
        // ── **THE ONE ALLOCATION THE MEMSET IS WORTH ARGUING ABOUT** — see
        //    [`Fill`]. Every other pinned span here is a table, a mirror or a
        //    shadow of kilobytes, and each of them is READ before it is
        //    written; T1 is the only one measured in gigabytes, the only one a
        //    restore covers whole, and the only one a boot can decline to make
        //    at all.
        let want = usize::try_from(host_at).unwrap_or(usize::MAX);
        let (host, image) = match fill {
            Fill::Cold => (Pinned::mapped(want)?, None),
            Fill::Restored => (Pinned::mapped_uninit(want)?, None),
            // **NOTHING IS PAGE-LOCKED HERE AND NOTHING IS COPIED.** The
            // tier serves each T1 image out of the artifact where it lies, and
            // the check that it may is one entry at a time below.
            Fill::Deferred(artifact) => (Pinned::mapped(0)?, Some(artifact)),
        };
        // ── **AND THE ADDRESS EACH T1 SPAN IS READ FROM IS DECIDED HERE,
        //    ONCE** (§L.1, at §M's layout).
        //
        //    Format 2 answered this with ONE number: the host section was the
        //    pinned allocation's whole byte image at the same offsets, so a
        //    deferred seat swapped a base and every T1 address moved with it.
        //    §M's artifact is budget-free — one image per plane, in a ranking
        //    order the budget then cuts — so the images this budget puts on T1
        //    are a subset of that order rather than a contiguous run of it, and
        //    there is no base to swap. Each span gets its own address instead,
        //    and the install moves all of them inside the same bracket, which
        //    is what §L.2's atomicity argument actually rested on.
        //
        //    The check that used to be one length comparison is now one per
        //    image, and it is the stronger claim: a file that does not carry
        //    this plane, or carries it at another span, is a file this seat
        //    must not serve from, and it is named as that plane rather than as
        //    a section of the wrong size.
        let mut serving = Vec::with_capacity(layout.len());
        for (param, host_at, _, reserved) in layout.iter().copied() {
            serving.push(match &image {
                None => host.device().saturating_add(host_at),
                Some(artifact) => {
                    let id = u32::try_from(param).unwrap_or(u32::MAX);
                    let found = artifact
                        .resolve(id)
                        .filter(|group| group.reserved == reserved)
                        .and_then(|group| artifact.span(group.offset, group.reserved));
                    let Some(bytes) = found else {
                        return Err(Fault::Residency(format!(
                            "the serving artifact at {} carries no {reserved}-byte image \
                             for param {param}, and a deferred seat reads that plane's T1 \
                             bytes out of the file where they lie; the file names another \
                             deployment",
                            artifact.path().display(),
                        )));
                    };
                    bytes.as_ptr() as u64
                }
            });
        }
        for (seat, at) in seats.iter_mut().zip(&serving) {
            seat.serving_at = *at;
        }
        for (whole, at) in wholes
            .iter_mut()
            .zip(serving.get(plan.banks().len()..).unwrap_or(&[]))
        {
            whole.serving_at = *at;
        }
        Ok(Tier {
            plan,
            host,
            refill: None,
            image,
            table: Buffer::zeroed(usize::try_from(entries).unwrap_or(usize::MAX))?,
            shadow: Pinned::mapped(usize::try_from(entries).unwrap_or(usize::MAX))?,
            counts: Buffer::zeroed(usize::try_from(counters).unwrap_or(usize::MAX))?,
            mirror: Pinned::mapped(usize::try_from(counters).unwrap_or(usize::MAX))?,
            seats,
            wholes,
            cells: Buffer::zeroed(usize::try_from(cells).unwrap_or(usize::MAX))?,
            cell_shadow: Pinned::mapped(usize::try_from(cells).unwrap_or(usize::MAX))?,
            group_counts: Buffer::zeroed(usize::try_from(group_counters).unwrap_or(usize::MAX))?,
            group_mirror: Pinned::mapped(usize::try_from(group_counters).unwrap_or(usize::MAX))?,
            ladder_open: groups.iter().any(|group| !group.backing.is_empty()),
            groups,
            berths: Vec::new(),
            swap: None,
            landed: Event::new()?,
            ladder: (0, 0, 0),
            tick: 0,
            source,
            mapped,
            drained: Event::new()?,
            ready: Event::new()?,
            moving: false,
            promotions: 0,
            demotions: 0,
            skipped: 0,
        })
    }

    /// Where a streamed bank's plane goes as it lands: its byte offset inside
    /// the pinned tier, or `None` for a param the device store holds.
    #[must_use]
    pub fn host_offset(&self, param: usize) -> Option<u64> {
        self.seats
            .iter()
            .find(|seat| seat.param == param)
            .map(|seat| seat.host_at)
            .or_else(|| {
                self.wholes
                    .iter()
                    .find(|whole| whole.param == param)
                    .map(|whole| whole.host_at)
            })
    }

    /// **The device-visible address of a plane this tier holds WHOLE**, or
    /// `None` for a param the device store holds.
    ///
    /// What a packed group's weight row points at: pinned memory, mapped, so
    /// the address is one the kernel dereferences over UVA at PCIe bandwidth
    /// and without a host round trip. The same "slow, not wrong" the dense
    /// tier's misses ride, one tier up and for the whole bank at once.
    ///
    /// **AND IT IS [`Whole::serving_at`] AND NOT THE PAGE-LOCKED ALLOCATION**
    /// (§L.1): while a deferred seat's window is open that address is inside
    /// the serving artifact's mapping, and the kernel cannot tell the two
    /// apart — one is UVA over page-locked memory, the other is HMM over a
    /// file, and both are a host pointer a packed select dereferences.
    #[must_use]
    pub fn pinned_at(&self, param: usize) -> Option<u64> {
        self.wholes
            .iter()
            .find(|whole| whole.param == param)
            .map(|whole| whole.serving_at)
    }

    /// **The address of a plane this tier serves out of the MAPPED artifact**
    /// (T2), or `None` for a param it does not.
    ///
    /// A host address, handed to a kernel unchanged. That is the whole trick
    /// and it is the device's, not this module's: with `pageableMemoryAccess`
    /// the GPU dereferences an ordinary pointer and the driver faults the page
    /// in from NVMe underneath it. The SM stalls; the host does not; article 2
    /// holds one tier further down.
    #[must_use]
    pub fn mapped_at(&self, param: usize) -> Option<u64> {
        self.mapped
            .iter()
            .find(|plane| plane.param == param)
            .map(|plane| plane.at)
    }

    /// **Where a plane of this load actually is** — the store's own address
    /// for everything it holds, the pinned tier's for a T1 group, the
    /// mapping's for a T2 one.
    ///
    /// `None` means the store holds it, which is what every caller reads as
    /// "use the offset you already have".
    #[must_use]
    pub fn offloaded_at(&self, param: usize) -> Option<u64> {
        self.pinned_at(param).or_else(|| self.mapped_at(param))
    }

    /// **How many bytes this tier serves out of the mapping** — the T2
    /// footprint, summed from what the artifact actually answered for rather
    /// than from what the plan asked for.
    ///
    /// The two agree or `Tier::open` refused; reporting the resolved number is
    /// what makes that a statement about the file and not about the plan.
    #[must_use]
    pub fn spilled_bytes(&self) -> u64 {
        self.mapped.iter().map(|plane| plane.bytes).sum()
    }

    /// **The artifact this tier serves T2 out of**, or `None` for a load that
    /// spilled nothing. What a gate reads to say which file the bytes came
    /// from — and, after T-4, WHICH of the two it was.
    #[must_use]
    pub fn source(&self) -> Option<&Spill> {
        self.source.as_ref()
    }

    /// **One T2 plane's bytes, borrowed from the mapping this tier seated it
    /// out of** — `None` for a param it does not serve from T2.
    ///
    /// The door the tier ARTIFACT's writer reads its third section through
    /// (§K.6-T4). It is stated here rather than left to the caller to take
    /// from [`Tier::mapped_at`] because the address and the length were
    /// resolved TOGETHER at [`Tier::open`], against an index that agreed with
    /// this plan; handing back the pair as a slice is what keeps the writer
    /// from re-deriving either.
    #[must_use]
    pub fn mapped_plane(&self, param: usize) -> Option<&[u8]> {
        let plane = self.mapped.iter().find(|plane| plane.param == param)?;
        let at = usize::try_from(plane.at).ok()? as *const u8;
        let len = usize::try_from(plane.bytes).ok()?;
        // SAFETY: the pair came out of `Spill::plane` — a window on a mapping
        // this tier owns in `source` for its whole life — and neither number
        // has been touched since. The borrow is the tier's own, so nothing can
        // outlive the mapping.
        Some(unsafe { std::slice::from_raw_parts(at, len) })
    }

    /// The pinned tier's host address, for the landing sink to store through.
    ///
    /// **ZERO-LENGTH WHILE [`Tier::deferring`]**, and deliberately: the
    /// landing sink is the COLD load's writer, a deferred seat is a warm boot
    /// that skipped the cold load, and a caller that falls back to it owes
    /// this tier [`Tier::undefer`] first. `Pinned::write` refuses a span past
    /// the allocation, so the failure is a sentence rather than a stray store.
    #[must_use]
    pub fn host(&self) -> &Pinned {
        &self.host
    }

    /// **IS THIS TIER STILL SERVING T1 OUT OF THE ARTIFACT'S MAPPING?**
    ///
    /// True between [`Fill::Deferred`] and the install — which is the window
    /// §L.5 describes, and the one interval in which this tier holds no
    /// page-locked bytes at all. False for every eager boot, and false again
    /// the moment the refill's image lands.
    #[must_use]
    pub fn deferring(&self) -> bool {
        self.image.is_some() && self.host.bytes() == 0
    }

    /// **THE ARTIFACT A DEFERRED SEAT IS SEATED OVER**, or `None` for a tier
    /// that page-locked its own image.
    ///
    /// The door the boot verification goes through, and it hands back THIS
    /// tier's mapping rather than the restore's own — which are two `mmap`s of
    /// one path, taken a few microseconds apart, and are the same inode every
    /// time but one. That one is what the door is for: what §L.3 promises is
    /// that *every byte served was checksummed*, and the only way to promise
    /// it about the bytes SERVED is to hash the mapping the addresses point
    /// into. A file replaced between the two opens then meets the boot
    /// verification instead of slipping past it.
    #[must_use]
    pub fn deferred_image(&self) -> Option<&crate::weight_cache::tier::Artifact> {
        self.image.as_ref().filter(|_| self.host.bytes() == 0)
    }

    /// **THE HOST-SIDE ADDRESS OF ONE T1 PLANE RIGHT NOW** — the T1 doors'
    /// other face, for the one reader that is the CPU.
    ///
    /// The device address and the host address of a page-locked allocation are
    /// the same number under UVA, so every kernel-facing door answers
    /// [`Tier::pinned_at`] and is done. The rotating dense pump is not
    /// kernel-facing: it stages a plane through `cudaMemcpyAsync` from a HOST
    /// pointer, and while a deferred seat's window is open that pointer is
    /// inside the artifact's mapping. Reading a tenant out of the file rather
    /// than out of page-locked memory is slow, not wrong — the same sentence
    /// T2 has always been — and the pointer stays valid past the install
    /// because the mapping is held for the tier's life.
    ///
    /// **A PLANE AT A TIME, AND NOT A BASE** (§M.3): the images a budget puts
    /// on T1 are a subset of the artifact's ranking order rather than a run of
    /// it, so there is no base an offset could be added to. `None` for a param
    /// this tier does not hold on T1.
    #[must_use]
    pub fn serving_host_of(&self, param: usize) -> Option<*const u8> {
        self.seats
            .iter()
            .find(|seat| seat.param == param)
            .map(|seat| seat.serving_at)
            .or_else(|| {
                self.wholes
                    .iter()
                    .find(|whole| whole.param == param)
                    .map(|whole| whole.serving_at)
            })
            .map(|at| at as *const u8)
    }

    /// **ARM THE BACKGROUND FILL** — the deferred seat's other half (§L.4).
    ///
    /// Called once, by the restore that verified this seat's T1 images in
    /// place, after the staging lanes it shares a device with have been opened
    /// and closed. `filling` is the thread; `filled` carries the page-locked image
    /// it built, ONCE, and carries nothing at all when the fill failed — see
    /// [`Refill`] for why silence is the failure signal and what the thread
    /// owes before it goes quiet.
    ///
    /// The window closes at the first [`Tier::promote`] past the arrival, or
    /// at [`Tier::settle_refill`]. Nothing else in this file waits for it.
    pub fn arm_refill(
        &mut self,
        filling: std::thread::JoinHandle<()>,
        filled: std::sync::mpsc::Receiver<Pinned>,
    ) {
        self.refill = Some(Refill {
            filling: Some(filling),
            filled,
            began: std::time::Instant::now(),
        });
    }

    /// **PUT THE PAGE-LOCKED IMAGE BACK** — the fall-through half of
    /// [`Fill::Deferred`], and [`Tier::zero_host`]'s analogue for it.
    ///
    /// A deferred boot whose T1 images turned out not to hash to what the
    /// file's block table states has no verified bytes to serve and no
    /// allocation to land the cold load into. This is the recovery: the allocation the deferred
    /// arm declined to make, made now and ZEROED — because what runs next is
    /// the cold load, which writes only the spans the plan names, and the
    /// padding between them has to be the deterministic image the artifact
    /// this boot writes on its way out will carry.
    ///
    /// Called BEFORE [`Tier::land`], so there is no cell, no berth and no
    /// entry to reseat: this is a base address changing while nothing has read
    /// it. A tier that is not deferring is left exactly as it is.
    ///
    /// # Errors
    ///
    /// [`Fault::OutOfMemory`] or [`Fault::Device`] for the allocation the
    /// deferred boot avoided — which is the failure this whole arm exists to
    /// make rarer, met on the one path where it cannot be.
    pub fn undefer(&mut self) -> Result<()> {
        if !self.deferring() {
            return Ok(());
        }
        // The refill is dropped — and therefore joined — before the
        // allocation, so the two page-locks are never outstanding at once.
        self.refill = None;
        let want = usize::try_from(self.plan.host_image()).unwrap_or(usize::MAX);
        self.host = Pinned::mapped(want)?;
        self.reseat();
        self.image = None;
        Ok(())
    }

    /// **CLOSE THE WINDOW NOW** — the gate's door onto the deferred seat, and
    /// nothing in the shell calls it.
    ///
    /// [`Tier::promote_now`]'s precedent, for the same reason it exists: the
    /// install rides the inter-fire gap, so a shell that is idle never takes
    /// it and a gate cannot observe a promotion by waiting for one. This asks
    /// the same question with the waiting struck out — it JOINS the fill
    /// thread instead of polling the channel — and strikes out nothing else:
    /// the same two events in the same order, the same reseat, the same two
    /// publishes.
    ///
    /// Answers whether the image was installed. `false` for a tier that was
    /// never deferred, one whose window is already closed, and one whose fill
    /// failed — that last is the §L.3 corollary and not an error: the seat
    /// keeps serving the verified bytes it has been serving all along.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for an event or a copy the runtime refused.
    pub fn settle_refill(&mut self, compute: *mut c_void, notify: *mut c_void) -> Result<bool> {
        let Some(refill) = self.refill.take() else {
            return Ok(false);
        };
        let window = refill.window_ms();
        let Some(host) = refill.settle() else {
            return Ok(false);
        };
        self.install(host, window, compute, notify)?;
        Ok(true)
    }

    /// **THE INSTALL: ONE BASE ADDRESS, PUBLISHED ALL AT ONCE** (§L.2).
    ///
    /// Inside the same bracket every promotion runs in — `drained` recorded on
    /// compute and waited on notify, the writes, `ready` recorded on notify
    /// and waited on compute — because it is the same hazard: a table entry
    /// and a cell may not be rewritten under a fire that is already reading
    /// them.
    ///
    /// ```text
    ///   drained  everything airborne on compute is past
    ///   swap     host <- the page-locked image, every span's `serving_at`
    ///            <- its offset inside it
    ///   reseat   every T1 group's `at` re-formed, and its berth opened
    ///   publish  publish_all (entries x 8 B) + publish_cells (groups x 16 B)
    ///   ready    the next fire's launches wait here
    /// ```
    ///
    /// **AND THE ONE LEG THIS BRACKET HAS THAT A PROMOTION DOES NOT**: both
    /// rungs are byte-identical verified content — the mapping was hashed at
    /// the boot, the page-locked copy was hashed as it was read — so the two
    /// addresses name the same bytes. A cell caught mid-flight names one or
    /// the other and there is no third answer, which is why the whole tier
    /// moves in one round instead of one group per gap. The post-install
    /// tier is the eager boot's, address for address.
    fn install(
        &mut self,
        host: Pinned,
        window_ms: u64,
        compute: *mut c_void,
        notify: *mut c_void,
    ) -> Result<()> {
        self.drained.record(compute)?;
        self.drained.wait(notify)?;
        self.host = host;
        self.reseat();
        self.refill = None;
        // ── EVERY T1 GROUP, RE-FORMED AND GIVEN ITS BERTH. A group the
        //    ladder moved during the window is on another rung now and is not
        //    one of these: it holds a berth of its own, its cell names that
        //    berth, and neither is a function of this base.
        for at in 0..self.groups.len() {
            if self.groups[at].held != Held::Pinned || self.groups[at].berth.is_some() {
                continue;
            }
            let mut where_at = Vec::with_capacity(self.groups[at].planes.len());
            for plane in &self.groups[at].planes {
                let Some(whole) = self.wholes.iter().find(|whole| whole.param == plane.param)
                else {
                    return Err(Fault::Residency(format!(
                        "`{}` is held on the pinned tier and plane {} has no offset in \
                         its image; the seating and the install were built from \
                         different walks",
                        self.groups[at].name, plane.param,
                    )));
                };
                where_at.push(whole.serving_at);
            }
            self.groups[at].at = where_at.clone();
            let berth = self.berths.len();
            self.berths.push(Berth {
                tier: Held::Pinned,
                at: where_at,
                shape: self.groups[at].planes.iter().map(|plane| plane.reserved).collect(),
                holds: Some(at),
            });
            self.groups[at].berth = Some(berth);
        }
        self.publish_all(notify)?;
        self.publish_cells(notify)?;
        self.ready.record(notify)?;
        self.ready.wait(compute)?;
        self.moving = true;
        crate::weight_cache::tier::count_promoted(window_ms);
        Ok(())
    }

    /// **EVERY T1 SPAN, RE-FORMED AGAINST THE PAGE-LOCKED ALLOCATION.**
    ///
    /// What used to be `self.serving = self.host.device()` — one assignment,
    /// because format 2's host section was the allocation's whole image at the
    /// same offsets. §M's artifact holds one image per plane in a budget-free
    /// order, so a deferred seat's addresses are scattered through the file
    /// and only the ALLOCATION is contiguous. This is the walk back: each
    /// span's `host_at` is its offset in the page-locked image and always was,
    /// so re-forming from it is the same arithmetic, done per span.
    ///
    /// Called by the two doors that make the allocation real — the install and
    /// [`Tier::undefer`] — and by nothing else.
    fn reseat(&mut self) {
        let base = self.host.device();
        for seat in &mut self.seats {
            seat.serving_at = base.saturating_add(seat.host_at);
        }
        for whole in &mut self.wholes {
            whole.serving_at = base.saturating_add(whole.host_at);
        }
    }

    /// **PUT THE PINNED IMAGE BACK TO ZEROS** — the fall-through half of
    /// [`Tier::open`]'s `refilling`.
    ///
    /// A warm boot that took the uninitialized allocation and then could not
    /// fill it — an image whose bytes do not hash to what the file's block
    /// table states, or a disk that refused a read halfway through one — is
    /// about to
    /// run the cold load into this tier, and the cold load writes only the
    /// spans the plan names. This is the promise being kept late: after it,
    /// the padding is the same deterministic zeros [`Pinned::mapped`] would
    /// have left, and the artifact that boot writes on its way out carries no
    /// byte nobody wrote.
    ///
    /// Costs the memset that was skipped, on the path where the skip did not
    /// pay off. Nothing else in this tier is touched: the tables, the shadows
    /// and the mirrors were never uninitialized.
    pub fn zero_host(&self) {
        self.host.zero();
    }

    /// **THE POPULATED SPANS OF THE PINNED IMAGE**:
    /// `(param, host_at, bytes, reserved)` for every dense bank and every
    /// plane of every group this tier holds on T1 (§K.6-T2's `Tier::image`).
    ///
    /// They tile `0..Pinned::bytes` exactly — every bank's padded span, then
    /// every pinned plane's — which is what lets the restore promise that an
    /// uninitialized allocation is written whole. Each `host_at` is an offset
    /// in the TIER and not in the file: since §M.3 the artifact is ordered by
    /// the ranking and this walk by the cut, so the restore reads image by
    /// image rather than section to section.
    ///
    /// It is [`Plan::host_layout`] handed back: [`Tier::open`] SEATED this
    /// tier out of that walk, so asking the tier and asking the plan are the
    /// same question, and this door is the one whose name says which of the
    /// two the writer meant. Everything between the spans — the alignment gap
    /// after the banks, and each plane's padding up to
    /// [`weights::ALIGN`](crate::weights::ALIGN) — is in the image and in no
    /// entry, which is exactly what the reserved/published pair already says
    /// of the device store.
    #[must_use]
    pub fn image(&self) -> Vec<(u64, u64, u64, u64)> {
        self.plan.host_layout()
    }

    /// **Seat the slabs and publish the first table** — after the checkpoint
    /// has landed, before the first fire.
    ///
    /// `slab_of` gives each streamed bank's device address inside the weight
    /// store, in the plan's bank order. Slots `0..resident` take experts
    /// `0..resident`, which is an arbitrary and honest start: nothing has
    /// fired, so nothing has an opinion about which experts are hot, and the
    /// promotion loop's first few gaps are what turn this into a working set.
    ///
    /// `store_at` gives the store's own address for every plane the store
    /// holds, as `(param, address)`. It is what turns a T0 packed group into a
    /// BERTH — the ladder's fastest rung, and the one the tier cannot displace
    /// a group out of unless it knows where the region is. The loader is the
    /// only caller that can answer it, because the offsets are its.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the copies, [`Fault::Runtimeless`] without a
    /// runtime.
    pub fn land(
        &mut self,
        slab_of: &[u64],
        store_at: &[(usize, u64)],
        stream: *mut c_void,
    ) -> Result<()> {
        debug_assert_eq!(slab_of.len(), self.seats.len());
        self.seat_groups(store_at)?;
        for (seat, slab) in self.seats.iter_mut().zip(slab_of) {
            seat.slab = *slab;
            seat.in_slot = (0..seat.resident).collect();
            for expert in 0..seat.resident {
                seat.slot_of[expert as usize] = Some(expert);
            }
        }
        // The bytes: every resident slot filled from the pinned copy that the
        // landing sink just wrote. One copy per slot rather than one per bank,
        // because a slot's stride and an expert's stride are the same number
        // only while the slots are the first `resident` experts — which is
        // true right now and false after the first promotion.
        for seat in &self.seats {
            for (slot, expert) in seat.in_slot.iter().enumerate() {
                copy_any(
                    stream,
                    seat.slab + slot as u64 * seat.stride,
                    seat.serving_at + u64::from(*expert) * seat.stride,
                    usize::try_from(seat.stride).unwrap_or(usize::MAX),
                )?;
            }
        }
        self.publish_all(stream)?;
        self.publish_cells(stream)
    }

    /// **Seat every packed group and open its berth** — the first moment the
    /// store's addresses exist, which is why it is here and not in
    /// [`Tier::open`].
    ///
    /// A group's live addresses are the plan's: the store's for a T0 group,
    /// the pinned tier's for a T1 one, the mapping's for a T2 one. The first
    /// two are BERTHS — regions a later, hotter group may be copied into —
    /// and the third is not, because a read-only mapping of a file is not a
    /// place anything is copied to.
    fn seat_groups(&mut self, store_at: &[(usize, u64)]) -> Result<()> {
        for at in 0..self.groups.len() {
            let held = self.groups[at].held;
            let mut where_at = Vec::with_capacity(self.groups[at].planes.len());
            for plane in &self.groups[at].planes {
                let found = match held {
                    Held::Device => store_at
                        .iter()
                        .find(|(param, _)| *param == plane.param)
                        .map(|(_, at)| *at),
                    Held::Pinned => self
                        .wholes
                        .iter()
                        .find(|whole| whole.param == plane.param)
                        .map(|whole| whole.serving_at),
                    Held::Mapped => self
                        .mapped
                        .iter()
                        .find(|mapped| mapped.param == plane.param)
                        .map(|mapped| mapped.at),
                };
                let Some(found) = found else {
                    return Err(Fault::Residency(format!(
                        "`{}` is planned {held:?} and plane {} has no address on that tier; the \
                         plan and the seating were built from different walks",
                        self.groups[at].name, plane.param,
                    )));
                };
                where_at.push(found);
            }
            self.groups[at].at = where_at.clone();
            // ── **AND A DEFERRED SEAT OPENS NO T1 BERTH** (§L.1). A berth is
            //    a region a hotter group may be COPIED INTO, and while the
            //    window is open the addresses a T1 group holds are inside a
            //    read-only mapping of a file: `close_berth` would land its
            //    bulk copy in the artifact. The rung is still `Held::Pinned`
            //    and still true — the plan seated it there and the install
            //    will put it there — so `residency()` reports what the plan
            //    says and nothing about the ladder is relabelled. What the
            //    window costs is that these regions cannot be displaced INTO
            //    for a minute; the install pushes every one of them.
            let deferring = held == Held::Pinned && self.refill.is_some();
            if held != Held::Mapped && !deferring {
                let berth = self.berths.len();
                self.berths.push(Berth {
                    tier: held,
                    at: where_at,
                    shape: self.groups[at].planes.iter().map(|plane| plane.reserved).collect(),
                    holds: Some(at),
                });
                self.groups[at].berth = Some(berth);
            }
        }
        Ok(())
    }

    /// Write every group's cell from its live addresses, and copy the lot
    /// across. Called once at [`Tier::land`]; after that only ONE cell at a
    /// time is ever written, by a swap.
    fn publish_cells(&mut self, stream: *mut c_void) -> Result<()> {
        if self.cell_shadow.bytes() == 0 {
            return Ok(());
        }
        for at in 0..self.groups.len() {
            self.write_cell(at);
        }
        copy_any(
            stream,
            self.cells.ptr(),
            self.cell_shadow.device(),
            self.cell_shadow.bytes(),
        )
    }

    /// Fill one group's sixteen shadow bytes from its live addresses.
    ///
    /// **BOTH PLANES OR NEITHER.** The pair is one word and it is written as
    /// one, which is the property that makes a torn pair unconstructible on
    /// this path — there is no shadow state that names one group's codes and
    /// another group's exponents, so there is no cell state that can either.
    fn write_cell(&self, at: usize) {
        let group = &self.groups[at];
        let mut word = [0u8; CELL as usize];
        for (plane, address) in group.at.iter().enumerate().take(CELL_PLANES) {
            let byte = plane * 8;
            word[byte..byte + 8].copy_from_slice(&address.to_ne_bytes());
        }
        self.cell_shadow
            .write(group.cell_at * CELL as usize, &word);
    }

    /// Copy one group's cell across, on `stream`.
    fn copy_cell(&self, at: usize, stream: *mut c_void) -> Result<()> {
        let cell = self.groups[at].cell_at as u64 * CELL;
        copy_any(
            stream,
            self.cells.ptr() + cell,
            self.cell_shadow.device() + cell,
            CELL as usize,
        )
    }

    /// Write every entry of every bank's table, from the seats' current
    /// residency.
    fn publish_all(&mut self, stream: *mut c_void) -> Result<()> {
        // A plan whose only streamed banks are packed groups has no table at
        // all: nothing dereferences one, so nothing was allocated, and a copy
        // of zero bytes between two null addresses is a call the runtime does
        // not need to be asked to make.
        if self.shadow.bytes() == 0 {
            return Ok(());
        }
        for seat in &self.seats {
            for expert in 0..seat.experts {
                let entry = seat.entry_at + expert as usize;
                let value = address_of(seat, expert);
                self.shadow
                    .write(entry * ENTRY as usize, &value.to_ne_bytes());
            }
        }
        copy_any(
            stream,
            self.table.ptr(),
            self.shadow.device(),
            self.shadow.bytes(),
        )
    }

    /// **The two device addresses a PACKED group's select kernel reads**, or
    /// `None` for a param that is not one of this tier's routed packed banks.
    ///
    /// The cell is where the group's two plane bases live; the counter is
    /// where its routing is noted. Both are fixed addresses for the load's
    /// life — the CONTENTS move, which is the whole of the ladder — so a
    /// captured graph holds them across every promotion (article 7).
    #[must_use]
    pub fn group_handles(&self, param: usize) -> Option<GroupHandles> {
        self.groups
            .iter()
            .find(|group| group.param == param)
            .map(|group| GroupHandles {
                cell: self.cells.ptr() + group.cell_at as u64 * CELL,
                hits: self.group_counts.ptr() + group.cell_at as u64 * COUNTER,
            })
    }

    /// The two device addresses `param`'s select kernel reads, or `None` for a
    /// param this tier does not hold — which is every param of a plan whose
    /// banks are resident.
    #[must_use]
    pub fn handles(&self, param: usize) -> Option<Handles> {
        self.seats.iter().find(|seat| seat.param == param).map(|seat| Handles {
            table: self.table.ptr() + seat.entry_at as u64 * ENTRY,
            counts: self.counts.ptr() + seat.counter_at as u64 * COUNTER,
        })
    }

    /// **Carry the fire's usage counts out**, asynchronously, on `stream`.
    ///
    /// Called from `settle`, behind the event that says this step's work is
    /// done, on the NOTIFY stream — so it neither gates the transition
    /// between waves (article 2) nor blocks the host. Nothing waits for it:
    /// the host reads whatever has landed at the next gap.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the copy.
    pub fn drain(&self, stream: *mut c_void) -> Result<()> {
        if self.counts.bytes() > 0 {
            copy_any(
                stream,
                self.mirror.device(),
                self.counts.ptr(),
                self.counts.bytes(),
            )?;
        }
        // The GROUP counters ride the same asynchronous D2H and are read the
        // same way (wave B7): one `u32` per packed group, `atomicAdd`ed once
        // per routed row per fire by the block that owns that route's first
        // row tile. Nothing waits for them either.
        if self.group_counts.bytes() > 0 {
            copy_any(
                stream,
                self.group_mirror.device(),
                self.group_counts.ptr(),
                self.group_counts.bytes(),
            )?;
        }
        Ok(())
    }

    /// **The promotion**, between two fires (alto design §7; article 3
    /// applied to weights).
    ///
    /// Reads the counters the last settled fires carried out, moves at most
    /// [`MOVES`] experts of each bank onto the device, and points the table
    /// at where they now are. Answers how many experts moved.
    ///
    /// **NOTHING HERE BLOCKS.** The order is three enqueues and no wait:
    ///
    /// ```text
    /// compute ──record(drained)──────────────────────────┐
    /// notify   ──wait(drained)─ entry ─ bytes ─ entry ─ record(ready)
    /// compute ──wait(ready)── the next fire's launches
    /// ```
    ///
    /// The first edge is what makes an eviction safe: a slot is overwritten
    /// only after everything already enqueued on the compute stream — every
    /// airborne fire — is past it. The last is what makes the promotion
    /// visible: the next fire's launches are behind the entry writes, so no
    /// fire ever reads a table entry that names bytes on their way in.
    ///
    /// A round whose predecessor has not finished moving is SKIPPED, not
    /// waited for. That is the whole doctrine in one line: residency is a
    /// promotion, so a promotion that would have to wait simply does not
    /// happen this gap.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for an event or a copy the runtime refused.
    pub fn promote(&mut self, compute: *mut c_void, notify: *mut c_void) -> Result<u32> {
        if self.seats.is_empty() && self.groups.is_empty() {
            return Ok(0);
        }
        // The shadow is one allocation shared by every entry write, so a round
        // may not rewrite a word an earlier round's copy may still be reading.
        if self.moving && !self.ready.done()? {
            self.skipped += 1;
            return Ok(0);
        }
        // ── **A GROUP'S BULK COPY HOLDS THE WHOLE ROUND BACK** (wave B7), and
        //    that is the doctrine and not an oversight. The copy is hundreds
        //    of megabytes on the notify stream; anything a later round
        //    enqueued would land BEHIND it, and `ready` — which the next fire
        //    waits on — would then be recorded behind it too. So the copy that
        //    was deliberately kept off the fire path would arrive back on it
        //    by way of the stream order. A round that would have to wait
        //    simply does not happen.
        if self.swap.is_some() && !self.landed.done()? {
            self.ladder.2 += 1;
            self.skipped += 1;
            return Ok(0);
        }
        // ── **AND THE DEFERRED SEAT'S ONE INSTALL, IF ITS IMAGE HAS ARRIVED**
        //    (§L.2). Behind both guards above, because it writes the shadow
        //    and every cell; and behind `swap.is_none()`, because a bulk copy
        //    in flight is reading a berth address this round would re-form.
        //    The question is a `try_recv` on a channel nobody else touches —
        //    the gap's own budget is microseconds and this costs an atomic.
        //
        //    It takes the whole gap: the install IS the round, and the moves
        //    the counters wanted wait for the next one. There is one of these
        //    per load.
        if self.swap.is_none() && self.refill.is_some() {
            let (filled, window) = match self.refill.as_mut() {
                Some(refill) => (refill.poll(), refill.window_ms()),
                None => (Filled::Waiting, 0),
            };
            match filled {
                Filled::Waiting => {}
                Filled::Ready(host) => {
                    self.install(host, window, compute, notify)?;
                    return Ok(0);
                }
                // The thread said what it found and deleted what it found it
                // in; nothing is installed, ever, and the seat goes on serving
                // the bytes it verified at the boot (§L.3's corollary).
                Filled::Refused => self.refill = None,
            }
        }
        let hits = self.mirror.read(0, self.mirror.bytes());
        let moves = self.decide(&hits);
        let step = self.step();
        if moves.is_empty() && step.is_none() {
            return Ok(0);
        }

        self.drained.record(compute)?;
        self.drained.wait(notify)?;
        for (at, slot, out, into) in &moves {
            let seat = &mut self.seats[*at];
            // THE EVICTED EXPERT'S ENTRY GOES BACK FIRST, and it is ordered
            // before the bytes by the stream it rides: from this copy on, a
            // kernel that routes to `out` reads pinned memory, which is where
            // its bytes have been all along.
            if let Some(out) = out {
                seat.slot_of[*out as usize] = None;
                let entry = seat.entry_at + *out as usize;
                let value = pinned_address_of(seat, *out);
                self.shadow.write(entry * ENTRY as usize, &value.to_ne_bytes());
                copy_any(
                    notify,
                    self.table.ptr() + entry as u64 * ENTRY,
                    self.shadow.device() + entry as u64 * ENTRY,
                    ENTRY as usize,
                )?;
                self.demotions += 1;
            }
            let seat = &self.seats[*at];
            let dst = seat.slab + u64::from(*slot) * seat.stride;
            copy_any(
                notify,
                dst,
                seat.serving_at + u64::from(*into) * seat.stride,
                usize::try_from(seat.stride).unwrap_or(usize::MAX),
            )?;
            let entry = seat.entry_at + *into as usize;
            self.shadow.write(entry * ENTRY as usize, &dst.to_ne_bytes());
            copy_any(
                notify,
                self.table.ptr() + entry as u64 * ENTRY,
                self.shadow.device() + entry as u64 * ENTRY,
                ENTRY as usize,
            )?;
            let seat = &mut self.seats[*at];
            seat.slot_of[*into as usize] = Some(*slot);
            seat.in_slot[*slot as usize] = *into;
            self.promotions += 1;
        }
        // ── THE LADDER'S SIXTEEN-BYTE HALF. Either the occupant's cell
        //    leaving the berth, or the candidate's cell arriving in it — one
        //    or the other, never both in one gap, because between them stands
        //    a copy that has to land.
        let bulk = match step {
            Some(Step::Open(swap)) => {
                self.open_berth(swap, notify)?;
                Some(swap)
            }
            Some(Step::Close(swap)) => {
                self.close_berth(swap, notify)?;
                None
            }
            None => None,
        };
        // **EVERYTHING A FIRE EVER WAITS FOR IS ABOVE THIS LINE.** Table
        // entries, slot bytes and cells: all small, all already enqueued.
        self.ready.record(notify)?;
        self.ready.wait(compute)?;
        // ── AND THE BULK COPY IS BELOW IT. Enqueued on the notify stream
        //    after `ready` was recorded, so no fire is ordered against it: it
        //    runs while the fires run, the berth it writes is referenced by
        //    nobody (the occupant's cell left it one instruction ago and the
        //    candidate's does not name it yet), and `landed` is asked at a
        //    later gap rather than waited on here.
        if let Some(swap) = bulk {
            let into = self.berths[swap.berth].at.clone();
            let group = &self.groups[swap.group];
            let from: Vec<(u64, u64)> = group
                .at
                .iter()
                .zip(&group.planes)
                .map(|(at, plane)| (*at, plane.bytes))
                .collect();
            for (dst, (src, bytes)) in into.into_iter().zip(from) {
                copy_any(notify, dst, src, usize::try_from(bytes).unwrap_or(usize::MAX))?;
            }
            self.landed.record(notify)?;
            self.swap = Some(swap);
        }
        self.moving = true;
        Ok(u32::try_from(moves.len()).unwrap_or(u32::MAX))
    }

    /// **Half one: the berth's occupant goes back to the file.**
    ///
    /// The cell write is what does it, and it is ordered before the bulk copy
    /// by the stream both ride: from this sixteen bytes on, a kernel that
    /// reads the group reads the artifact, which is where its bytes have been
    /// all along (the artifact is a snapshot of the store, so a T0 group's
    /// bytes are in it too, unchanged — expert weights are read-only, which is
    /// what makes a demotion free rather than a write-back).
    ///
    /// Answers the swap to copy, so the caller can enqueue it after `ready`.
    fn open_berth(&mut self, swap: Swap, notify: *mut c_void) -> Result<()> {
        if let Some(out) = self.berths[swap.berth].holds {
            self.groups[out].at = self.groups[out].backing.clone();
            self.groups[out].held = Held::Mapped;
            self.groups[out].berth = None;
            self.tick += 1;
            self.groups[out].settled = self.tick;
            self.write_cell(out);
            self.copy_cell(out, notify)?;
            self.ladder.1 += 1;
        }
        self.berths[swap.berth].holds = None;
        Ok(())
    }

    /// **Half two: the candidate's cell arrives in the berth.**
    ///
    /// Only reached once [`Tier::landed`] says the bytes are there, so the
    /// address this publishes names a copy that is complete. Land the copy,
    /// THEN flip the pointer — the whole order of the thing, in two gaps.
    fn close_berth(&mut self, swap: Swap, notify: *mut c_void) -> Result<()> {
        let berth = &self.berths[swap.berth];
        let (tier, at) = (berth.tier, berth.at.clone());
        // The berth the candidate is LEAVING, if it was in one, is now free:
        // that is how the ladder walks more than one rung, one gap at a time.
        if let Some(was) = self.groups[swap.group].berth {
            self.berths[was].holds = None;
        }
        self.groups[swap.group].at = at;
        self.groups[swap.group].held = tier;
        self.groups[swap.group].berth = Some(swap.berth);
        self.berths[swap.berth].holds = Some(swap.group);
        self.tick += 1;
        self.groups[swap.group].settled = self.tick;
        self.write_cell(swap.group);
        self.copy_cell(swap.group, notify)?;
        self.swap = None;
        self.ladder.0 += 1;
        Ok(())
    }

    /// **What the ladder does this gap** — see [`Swap`] for the two halves.
    ///
    /// `Close` first and unconditionally: a swap whose bytes have landed is
    /// finished before another is started, because the berth's occupant is
    /// already back on the file and every gap that passes is a gap the
    /// candidate spends slower than it has to.
    fn step(&self) -> Option<Step> {
        if let Some(swap) = self.swap {
            return Some(Step::Close(swap));
        }
        // **A LOAD WITH NO ARTIFACT HAS THE ASSIGNMENT IT BOOTED WITH**, and
        // this is where that costs nothing rather than costing a vote every
        // gap: a swap's first half points the displaced group's cell back at
        // the file, so with no file nothing is displaceable and the question
        // is not worth asking. One bool, decided once at `open`.
        if GROUP_MOVES == 0 || self.berths.is_empty() || !self.ladder_open {
            return None;
        }
        let hits = self.group_mirror.read(0, self.group_mirror.bytes());
        self.decide_group(&hits).map(Step::Open)
    }

    /// **Which group should change rungs, given the counters** — the vote,
    /// and the ONE rule it follows.
    ///
    /// ```text
    /// a swap of candidate G into berth B, displacing occupant H, is taken
    /// iff   rung(B) < rung(G)          B is a faster tier than G is on
    /// and   shape(B) == shape(G)       the berth was sized for G's planes
    /// and   hits(G) > hits(H)          and G is strictly the hotter of the two
    /// ```
    ///
    /// **THE THIRD LINE IS A STRICT-IMPROVEMENT RULE AND IT IS WHY NOTHING
    /// CHURNS.** Read the deployment's cost as `Σ hits(g) × rung(tier(g))`:
    /// swapping G and H changes it by `(hits(G) − hits(H)) × (rung(G) −
    /// rung(B))`, both factors positive, so every move this function returns
    /// strictly lowers a bounded sum and the sequence terminates. It also
    /// says, exactly, what the ladder does NOT do — and that sentence is worth
    /// writing down because it is the honest reading of today's catalog:
    ///
    /// **A model whose routed banks are all read every fire has a UNIFORM
    /// vote, and a uniform vote is a steady state at the plan's own
    /// assignment.** Every one of gpt-oss-20b's 48 groups is read by every
    /// step, so after the counters warm they are equal, `hits(G) > hits(H)` is
    /// false everywhere, and the ladder correctly does nothing. That is not a
    /// missing feature: with equal demand any assignment costs the same, and a
    /// swap would be ~265 MiB of PCIe bought for zero. What the ladder is for
    /// is the demand the PLAN cannot see — a bank a session never routes to (a
    /// tower's, an aux head's, an arm of a conditional), which sits on the
    /// device holding a rung a hot mapped bank wants. Then the difference is
    /// large, the rule fires, and it stops as soon as the order is right.
    ///
    /// Ties break toward the biggest improvement, then toward the lowest berth
    /// and the lowest group, so that two boots of one deployment with the same
    /// counters make the same move.
    fn decide_group(&self, hits: &[u8]) -> Option<Swap> {
        let count = |at: usize| -> u32 {
            let byte = at * COUNTER as usize;
            hits.get(byte..byte + COUNTER as usize)
                .and_then(|word| word.try_into().ok())
                .map_or(0, u32::from_ne_bytes)
        };
        vote(&self.berths, &self.groups, count)
    }

    /// Which experts should change places, given the counters.
    ///
    /// `(seat, slot, evicted, promoted)`. Least-used resident out, most-used
    /// non-resident in, at most [`MOVES`] per bank, and only where the
    /// incoming expert has strictly more hits than the outgoing one — so a
    /// steady state is a gap with no moves rather than a gap that churns.
    fn decide(&self, hits: &[u8]) -> Vec<(usize, u32, Option<u32>, u32)> {
        let count = |at: usize| -> u32 {
            let byte = at * COUNTER as usize;
            hits.get(byte..byte + COUNTER as usize)
                .and_then(|w| w.try_into().ok())
                .map_or(0, u32::from_ne_bytes)
        };
        let mut out = Vec::new();
        for (at, seat) in self.seats.iter().enumerate() {
            let mut cold: Vec<(u32, u32)> = seat
                .in_slot
                .iter()
                .enumerate()
                .map(|(slot, expert)| (count(seat.counter_at + *expert as usize), slot as u32))
                .collect();
            cold.sort_unstable();
            let mut hot: Vec<(u32, u32)> = (0..seat.experts)
                .filter(|expert| seat.slot_of[*expert as usize].is_none())
                .map(|expert| (count(seat.counter_at + expert as usize), expert))
                .collect();
            hot.sort_unstable_by(|a, b| b.cmp(a));
            for ((cold_hits, slot), (hot_hits, expert)) in cold.iter().zip(&hot).take(MOVES) {
                if hot_hits <= cold_hits {
                    break;
                }
                out.push((at, *slot, Some(seat.in_slot[*slot as usize]), *expert));
            }
        }
        out
    }

    /// **What is resident right now, and what the fires asked for** — the
    /// accessor a gate reads.
    #[must_use]
    pub fn residency(&self) -> Vec<BankResidency> {
        let hits = self.mirror.read(0, self.mirror.bytes());
        let dense = self.seats.iter().map(|seat| BankResidency {
            name: seat.name.clone(),
            experts: seat.experts,
            slots: seat.resident,
            held: None,
            in_slot: seat.in_slot.clone(),
            hits: (0..seat.experts)
                .map(|expert| {
                    let byte = (seat.counter_at + expert as usize) * COUNTER as usize;
                    hits.get(byte..byte + COUNTER as usize)
                        .and_then(|w| w.try_into().ok())
                        .map_or(0, u32::from_ne_bytes)
                })
                .collect(),
        });
        // **A PACKED GROUP REPORTS ZERO SLOTS, WHICH IS THE TRUTH.** Its
        // planes are on ONE tier whole; the device slab seats none of its
        // experts, because no kernel on this path reads a per-expert table. A
        // gate that asks what is resident is told exactly that.
        //
        // **AND `held` IS THE LIVE ANSWER, NOT THE PLAN'S** (wave B7). Since a
        // group moves, the plan's word for where it is goes stale the first
        // time the ladder takes a rung, and the accessor a gate reads is the
        // one place that must not be stale. `hits` is the group's own counter
        // — one number, because a group is what can move — carried out by the
        // same asynchronous readback the dense counts ride.
        let counts = self.group_mirror.read(0, self.group_mirror.bytes());
        let packed = self.groups.iter().map(|group| {
            let byte = group.cell_at * COUNTER as usize;
            BankResidency {
                name: group.name.clone(),
                experts: group.experts,
                slots: 0,
                in_slot: Vec::new(),
                hits: vec![
                    counts
                        .get(byte..byte + COUNTER as usize)
                        .and_then(|word| word.try_into().ok())
                        .map_or(0, u32::from_ne_bytes),
                ],
                held: Some(group.held),
            }
        });
        // The spilled DENSE planes are groups too and are reported as such,
        // still off the plan: they have no cell, they never move, and an
        // operator reading "48 banks mapped" wants to know whether the
        // embedding was one of them.
        let planes = self
            .plan
            .groups()
            .iter()
            .filter(|group| !group.routed)
            .map(|group| BankResidency {
                name: group.name.clone(),
                experts: group.experts,
                slots: 0,
                in_slot: Vec::new(),
                hits: Vec::new(),
                held: Some(group.held),
            });
        dense.chain(packed).chain(planes).collect()
    }

    /// `(groups promoted, groups demoted, gaps a swap in flight held back)`,
    /// since load — the ladder's own motion, beside [`Tier::motion`]'s
    /// per-expert one.
    ///
    /// A promotion and a demotion come in pairs while every berth is full,
    /// which is the ordinary case; a demotion without a promotion behind it is
    /// a swap whose bytes have not landed yet.
    #[must_use]
    pub fn ladder(&self) -> (u64, u64, u64) {
        self.ladder
    }

    /// **Take one rung for `name` NOW, whatever the counters say** — the gate's
    /// door onto the ladder, and nothing in the shell calls it.
    ///
    /// [`Tier::decide_group`]'s rule is a strict-improvement rule, and its doc
    /// proves what follows from that: a deployment whose routed banks are all
    /// read every fire has a uniform vote and therefore a steady state at the
    /// plan's own assignment. That is the right behaviour and it is also why a
    /// GATE cannot observe a rung being taken by waiting for one.
    ///
    /// So this asks the same question with the vote's clause struck out, and
    /// strikes out nothing else: the same berth match, the same two halves in
    /// the same order, the same cell-written-last discipline. It is
    /// SYNCHRONOUS — it settles the bulk copy on the host rather than asking
    /// `landed` at a later gap — which is exactly the property the production
    /// path does not have, and the reason this is a door and not a policy.
    ///
    /// Answers `(from, to)` for the group that moved, or `None` when no berth
    /// of the right shape stands on a faster rung.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for an event or a copy the runtime refused.
    pub fn promote_now(
        &mut self,
        name: &str,
        compute: *mut c_void,
        notify: *mut c_void,
    ) -> Result<Option<(Held, Held)>> {
        let Some(group) = self.groups.iter().position(|group| group.name == name) else {
            return Ok(None);
        };
        let shape: Vec<u64> = self.groups[group]
            .planes
            .iter()
            .map(|plane| plane.reserved)
            .collect();
        let was = self.groups[group].held;
        let hits = self.group_mirror.read(0, self.group_mirror.bytes());
        let count = |at: usize| -> u32 {
            let byte = at * COUNTER as usize;
            hits.get(byte..byte + COUNTER as usize)
                .and_then(|word| word.try_into().ok())
                .map_or(0, u32::from_ne_bytes)
        };
        // An EMPTY berth first, then the coldest occupant, then the one that
        // has sat on its rung longest — see [`Group::settled`] for why the
        // third clause is here and why the vote does not need it.
        let Some(berth) = self
            .berths
            .iter()
            .enumerate()
            .filter(|(at, berth)| {
                berth.tier.rung() < was.rung()
                    && berth.shape == shape
                    && self.groups[group].berth != Some(*at)
                    && berth
                        .holds
                        .is_none_or(|out| !self.groups[out].backing.is_empty())
            })
            .min_by_key(|(_, berth)| match berth.holds {
                None => (0u8, 0u32, 0u64),
                Some(out) => (1, count(self.groups[out].cell_at), self.groups[out].settled),
            })
            .map(|(at, _)| at)
        else {
            return Ok(None);
        };
        let swap = Swap { berth, group };
        self.drained.record(compute)?;
        self.drained.wait(notify)?;
        self.open_berth(swap, notify)?;
        let into = self.berths[berth].at.clone();
        let from: Vec<(u64, u64)> = self.groups[group]
            .at
            .iter()
            .zip(&self.groups[group].planes)
            .map(|(at, plane)| (*at, plane.bytes))
            .collect();
        for (dst, (src, bytes)) in into.into_iter().zip(from) {
            copy_any(notify, dst, src, usize::try_from(bytes).unwrap_or(usize::MAX))?;
        }
        self.landed.record(notify)?;
        self.landed.settle()?;
        self.close_berth(swap, notify)?;
        self.ready.record(notify)?;
        self.ready.settle()?;
        self.moving = false;
        Ok(Some((was, self.groups[group].held)))
    }

    /// `(experts promoted, experts demoted, gaps skipped because the previous
    /// round was still moving)`, since load.
    #[must_use]
    pub fn motion(&self) -> (u64, u64, u64) {
        (self.promotions, self.demotions, self.skipped)
    }

    /// The plan this tier serves.
    #[must_use]
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Every byte the tier holds off the device store: the pinned tier, the
    /// tables and their shadow, the counters and their mirror.
    #[must_use]
    pub fn bytes(&self) -> (u64, u64) {
        let device = self.table.bytes() as u64
            + self.counts.bytes() as u64
            + self.cells.bytes() as u64
            + self.group_counts.bytes() as u64;
        let host = self.host.bytes() as u64
            + self.shadow.bytes() as u64
            + self.mirror.bytes() as u64
            + self.cell_shadow.bytes() as u64
            + self.group_mirror.bytes() as u64;
        (device, host)
    }
}

/// **The vote, as a pure function of the seating and the counters** — the
/// body of [`Tier::decide_group`], lifted out so that the rule can be
/// exercised without a device (the tier itself is four device allocations).
///
/// The rule and its termination argument are on [`Tier::decide_group`].
fn vote(berths: &[Berth], groups: &[Group], count: impl Fn(usize) -> u32) -> Option<Swap> {
    let mut best: Option<(u32, Swap)> = None;
    for (at, berth) in berths.iter().enumerate() {
        // An occupant with no backing cannot be displaced: there would be
        // nowhere to point its cell at. A load that opened no artifact has no
        // group with a backing, which is exactly the statement that its
        // assignment is the one it boots with.
        let out = match berth.holds {
            Some(out) if groups[out].backing.is_empty() => continue,
            held => held,
        };
        let floor = out.map_or(0, |out| count(groups[out].cell_at));
        for (group, candidate) in groups.iter().enumerate() {
            if candidate.berth == Some(at) || candidate.held.rung() <= berth.tier.rung() {
                continue;
            }
            // Plane for plane, and WITHOUT building a vector to say so: this
            // runs berths x groups times in every inter-fire gap, and a gap is
            // the one place in this file that is measured in microseconds.
            if candidate.planes.len() != berth.shape.len()
                || candidate
                    .planes
                    .iter()
                    .zip(&berth.shape)
                    .any(|(plane, want)| plane.reserved != *want)
            {
                continue;
            }
            let hot = count(candidate.cell_at);
            if hot <= floor {
                continue;
            }
            let gain = hot - floor;
            if best.is_none_or(|(had, _)| gain > had) {
                best = Some((gain, Swap { berth: at, group }));
            }
        }
    }
    best.map(|(_, swap)| swap)
}

/// Where expert `expert` of `seat` lives right now.
fn address_of(seat: &Seat, expert: u32) -> u64 {
    match seat.slot_of[expert as usize] {
        Some(slot) => seat.slab + u64::from(slot) * seat.stride,
        None => pinned_address_of(seat, expert),
    }
}

/// Where expert `expert` of `seat` lives on T1 — the address a kernel
/// dereferences when the expert is not on the device.
///
/// [`Seat::serving_at`] and not the pinned allocation's base: a deferred
/// seat's `serving_at` is the bank's image inside the artifact's mapping and
/// an eager one's is `host.device() + host_at`, and the expert arithmetic on
/// top of it is the same either way (§L.1, §M.3).
fn pinned_address_of(seat: &Seat, expert: u32) -> u64 {
    seat.serving_at + u64::from(expert) * seat.stride
}

#[cfg(test)]
mod tests {
    use model_dsl::Platform;

    use super::*;

    fn a3b() -> Trace {
        let trace = models::trace_of("qwen35-a3b-bf16-kv-bf16").expect("the catalog ships the SKU");
        trace(Platform::Cuda)
    }

    /// The catalog's one split-plane MoE: 32 experts, 24 layers, mxfp4 banks.
    fn gpt_oss() -> Trace {
        let trace =
            models::trace_of("gptoss-20b-bf16-mxfp4-kv-bf16").expect("the catalog ships the SKU");
        trace(Platform::Cuda)
    }

    /// **The pairing a load plan records, built here off the DSL's own
    /// name-minting** — so that a plan test needs no checkpoint on disk.
    ///
    /// The SHELL never does this (`weights::pairings` reads the loader's
    /// `QuantAttachment`, and this module's `Attachments` doc says why a
    /// suffix is the wrong source). A test may, because it is asserting about
    /// the arithmetic a pairing feeds and not about where the pairing came
    /// from, and `model_dsl::scales_name` is the one function that mints the
    /// name in the first place.
    fn scales_of(trace: &Trace) -> Attachments {
        let at: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        trace
            .params
            .iter()
            .enumerate()
            .filter(|(_, param)| param.dtype == model_ir::Dtype::Mxfp4)
            .map(|(codes, param)| {
                let scales = model_dsl::scales_name(&param.name);
                let scales = *at
                    .get(scales.as_str())
                    .unwrap_or_else(|| panic!("`{}` declares no scales plane", param.name));
                (codes, vec![scales])
            })
            .collect()
    }

    #[test]
    fn a_packed_bank_is_sized_where_it_used_to_have_no_element_size() {
        // The refusal this wave removed: `elem_bytes(Mxfp4)` is honestly
        // `None`, and reading that as "unsizable" refused gpt-oss at its first
        // bank param. The plane's declared rectangle IS its byte count.
        let trace = gpt_oss();
        let bytes = crate::weights::plane_bytes(&trace).expect("a packed SKU sizes");
        assert_eq!(bytes.len(), trace.params.len());
        for (plane, param) in bytes.iter().zip(&trace.params) {
            assert!(*plane > 0, "`{}` sizes to nothing", param.name);
        }
        let plan = Plan::of(&trace, &scales_of(&trace), Budgets::uncapped()).expect("an uncapped packed MoE plans");
        assert!(!plan.streams(), "an uncapped load streams nothing");
        // ~13.8 GiB of weights is what the SKU is; the assertion is the order
        // of magnitude, because the exact number is the model text's to move.
        let demand = plan.device_demand();
        assert!(
            (12 << 30..15 << 30).contains(&demand),
            "gpt-oss-20b's table is {demand} bytes, which is not the ~13.8 GiB it declares"
        );
    }

    #[test]
    fn a_capped_packed_load_streams_whole_groups_and_never_half_of_one() {
        let trace = gpt_oss();
        let planes = scales_of(&trace);
        let full = Plan::of(&trace, &planes, Budgets::uncapped())
            .expect("uncapped plans")
            .device_demand();
        let budget = full * 2 / 3;
        let plan = Plan::of(&trace, &planes, Budgets::device(budget)).expect("two thirds of the table serves");

        assert!(plan.streams(), "two thirds of the table cannot be held whole");
        assert!(
            plan.banks().is_empty(),
            "gpt-oss declares no DENSE routed bank; nothing is seated expert by expert"
        );
        assert!(!plan.groups().is_empty(), "and something had to give");
        assert!(
            plan.device_demand() <= budget,
            "the plan fits the budget it was given: {} > {budget}",
            plan.device_demand()
        );

        // **THE COHERENCE CLAIM, AS AN ASSERTION.** Every plane of a streamed
        // group is pinned; no group has one plane on each tier. A torn pair is
        // worse than a miss, and this is the shape that makes it impossible.
        let mut pinned = 0usize;
        for group in plan.groups() {
            assert!(
                group.planes.len() >= 2,
                "`{}` is a split-plane bank and the group holds {} plane(s)",
                group.name,
                group.planes.len()
            );
            for plane in &group.planes {
                assert!(
                    plan.pinned(plane.param),
                    "`{}` streams and its plane {} does not",
                    group.name,
                    plane.param
                );
                pinned += 1;
            }
            assert_eq!(
                group.bytes,
                group.planes.iter().map(|plane| plane.reserved).sum::<u64>(),
                "`{}`'s bytes are its planes' bytes",
                group.name
            );
        }
        // And nothing outside a group is pinned.
        let counted = (0..trace.params.len()).filter(|at| plan.pinned(*at)).count();
        assert_eq!(counted, pinned, "a plane is pinned only as part of a group");

        // T1 owes exactly what T0 gave back, rounded as the tier seats it.
        assert_eq!(
            plan.host_demand(),
            plan.groups().iter().map(|group| group.bytes).sum::<u64>(),
            "the pinned tier holds the streamed groups and nothing else"
        );
        // A tighter budget streams at least as much as a looser one.
        let tighter = Plan::of(&trace, &planes, Budgets::device(full / 3)).expect("a third of the table serves");
        assert!(
            tighter.host_demand() >= plan.host_demand(),
            "a tighter budget pinned {} and a looser one pinned {}",
            tighter.host_demand(),
            plan.host_demand()
        );
    }

    #[test]
    fn both_budgets_under_the_banks_plan_the_rest_onto_the_mapped_tier() {
        let trace = gpt_oss();
        let planes = scales_of(&trace);
        let full = Plan::of(&trace, &planes, Budgets::uncapped())
            .expect("uncapped plans")
            .device_demand();
        // W-1's shape: BOTH ceilings under the banks, so what neither holds
        // has one place left.
        let plan = Plan::of(
            &trace,
            &planes,
            Budgets {
                device: Some(4 << 30),
                host: Some(2 << 30),
            },
        )
        .expect("a plan past both budgets is planned, not refused");

        assert!(plan.streams());
        assert!(
            plan.spill_demand() > 0,
            "two budgets under {full} bytes of table have to spill something"
        );
        assert!(plan.device_demand() <= 4 << 30, "{}", plan.device_demand());
        assert!(plan.host_demand() <= 2 << 30, "{}", plan.host_demand());

        // **EVERY GROUP IS ON EXACTLY ONE TIER**, and the two sets do not
        // overlap. This is the coherence claim of wave W-5, restated for three
        // tiers: a group whose codes are pinned and whose exponents are mapped
        // is the torn pair, and the plan cannot express it.
        let (mut pinned, mut mapped) = (0u64, 0u64);
        for group in plan.groups() {
            for plane in &group.planes {
                assert_ne!(
                    plan.pinned(plane.param),
                    plan.mapped(plane.param),
                    "`{}`'s plane {} is on both tiers or neither",
                    group.name,
                    plane.param
                );
                assert_eq!(
                    plan.pinned(plane.param),
                    group.held == Held::Pinned,
                    "`{}` is held {:?} and its plane {} disagrees",
                    group.name,
                    group.held,
                    plane.param
                );
            }
            match group.held {
                Held::Pinned => pinned += group.bytes,
                Held::Mapped => mapped += group.bytes,
                // `Plan::groups` is the placements a budget was forced to
                // make; a group the STORE holds is never one of them and is
                // in `Plan::seated` instead (wave B7).
                Held::Device => panic!("`{}` is in `groups` and held on the device", group.name),
            }
        }
        assert!(pinned > 0 && mapped > 0, "this budget pair uses both tiers");
        assert_eq!(plan.spill_demand(), mapped);

        // **NOTHING IS LOST BETWEEN THE TIERS.** Every byte of the uncapped
        // table is on exactly one of the three, up to the per-plane alignment
        // the store and the tier each round to.
        let three = plan.device_demand() + pinned + mapped;
        assert!(
            three.abs_diff(full) < crate::weights::ALIGN * trace.params.len() as u64,
            "the three tiers hold {three} and the table is {full}"
        );

        // A tighter device budget spills at least as much.
        let tighter = Plan::of(
            &trace,
            &planes,
            Budgets {
                device: Some(4 << 30),
                host: Some(1 << 30),
            },
        )
        .expect("a tighter host budget plans");
        assert!(
            tighter.spill_demand() >= plan.spill_demand(),
            "a 1 GiB host tier spilled {} and a 2 GiB one spilled {}",
            tighter.spill_demand(),
            plan.spill_demand()
        );
    }

    #[test]
    fn an_uncapped_host_tier_spills_nothing_however_tight_the_device_is() {
        // The W-5 shape, restated as a claim about the third tier: with no
        // host ceiling there is nothing T2 is FOR, and a plan that opened it
        // anyway would be asking for a file it does not need.
        let trace = gpt_oss();
        let planes = scales_of(&trace);
        let full = Plan::of(&trace, &planes, Budgets::uncapped())
            .expect("uncapped plans")
            .device_demand();
        // 2 and 3 and not 4: gpt-oss's dense planes are 3.36 GiB of its 12.8,
        // so a quarter of the table is under the dense floor and is the OTHER
        // refusal — the one `a_budget_under_the_dense_planes_is_refused_by_name`
        // already stands on.
        for divisor in [2, 3] {
            let plan = Plan::of(&trace, &planes, Budgets::device(full / divisor))
                .expect("a device-only cap plans");
            assert_eq!(
                plan.spill_demand(),
                0,
                "an uncapped host tier spilled at 1/{divisor} of the table"
            );
            assert!(plan.host_demand() > 0, "and it pinned what the device let go");
        }
    }

    #[test]
    fn a_packed_bank_the_plan_pairs_no_scales_with_is_refused_by_name() {
        let trace = gpt_oss();
        let full = Plan::of(&trace, &scales_of(&trace), Budgets::uncapped())
            .expect("uncapped plans")
            .device_demand();
        // The same trace, with the loader's pairing withheld: the codes could
        // be seated alone and the factors left behind, which is the one
        // outcome this module refuses rather than approximates.
        let why = Plan::of(&trace, &Attachments::new(), Budgets::device(full / 2))
            .expect_err("a packed bank with no pairing is not a bank this shell seats");
        let said = why.to_string();
        assert!(
            said.contains("pairs no scales plane"),
            "the refusal names what is missing: {said}"
        );
        assert!(
            said.contains("one group or not at all"),
            "and says what the group is for: {said}"
        );
    }

    #[test]
    fn an_uncapped_budget_is_the_degenerate_plan() {
        let plan = Plan::of(&a3b(), &Attachments::new(), Budgets::uncapped()).expect("a bf16 MoE plans");
        assert!(!plan.streams(), "an uncapped load streams nothing");
        assert_eq!(plan.host_demand(), 0, "and holds no pinned copy of anything");
        assert!(plan.device_demand() > 0);
    }

    #[test]
    fn a_budget_over_the_whole_table_is_the_same_degenerate_plan() {
        let trace = a3b();
        let full = Plan::of(&trace, &Attachments::new(), Budgets::uncapped()).expect("uncapped plans").device_demand();
        let plan = Plan::of(&trace, &Attachments::new(), Budgets::device(full)).expect("a budget at the demand plans");
        assert!(!plan.streams(), "a budget at full residency is `place_all`");
        assert_eq!(plan.device_demand(), full);
    }

    #[test]
    fn a_budget_under_the_table_seats_fewer_experts_and_pins_them_all() {
        let trace = a3b();
        let full = Plan::of(&trace, &Attachments::new(), Budgets::uncapped()).expect("uncapped plans").device_demand();
        let plan = Plan::of(&trace, &Attachments::new(), Budgets::device(full / 2)).expect("half the table streams");
        assert!(plan.streams(), "half the table cannot be held whole");
        assert!(
            plan.device_demand() <= full / 2,
            "the plan fits the budget it was given: {} > {}",
            plan.device_demand(),
            full / 2
        );
        let banks = plan.banks();
        assert!(!banks.is_empty(), "a3b declares routed banks");
        let arity = banks[0].experts;
        for bank in banks {
            assert_eq!(bank.experts, arity, "one arity for the plan");
            assert!(bank.resident >= 1 && bank.resident < bank.experts);
            assert_eq!(
                bank.resident, banks[0].resident,
                "one residency decision covers the plan"
            );
            assert_eq!(
                u64::from(bank.experts) * bank.stride % u64::from(bank.experts),
                0,
                "the slot stride divides the bank"
            );
        }
        // T1 holds every expert of every bank, resident or not.
        let pinned: u64 = banks
            .iter()
            .map(|bank| u64::from(bank.experts) * bank.stride)
            .sum();
        assert_eq!(plan.host_demand(), pinned);
    }

    #[test]
    fn a_budget_under_the_planes_that_cannot_move_is_refused_by_name() {
        // **THE FLOOR MOVED, AND THE REFUSAL MOVED WITH IT** (wave D2b). It
        // used to be the DENSE planes, because none of them could leave the
        // device; they can now, so what is left under any budget is the
        // planes that genuinely cannot — a REGISTERED adapter bank, whose
        // store offset `register_adapter` writes at — plus one expert slot of
        // every routed bank.
        let trace = a3b();
        let why = Plan::of(&trace, &Attachments::new(), Budgets::device(1 << 20))
            .expect_err("a megabyte holds nothing");
        let said = why.to_string();
        assert!(said.contains("REGISTERED"), "the refusal names the floor: {said}");
        assert!(
            said.contains("cannot be moved to another tier"),
            "and says why those planes and not the others: {said}"
        );
        assert!(
            said.contains("every OTHER dense plane in this plan can"),
            "and that the rest already spilled: {said}"
        );
    }

    #[test]
    fn a_dense_plan_under_a_budget_spills_its_latest_planes_and_keeps_its_first() {
        // **D2b: THE STATIC DEMAND SHAPE SPILLS TOO.** This test used to
        // assert that a dense plan under a budget was REFUSED — "nothing in it
        // is a routed-expert bank, so there is no tier to hold less of". There
        // is now: a dense plane the budget cannot seat becomes a group of one
        // and takes the pinned tier, read over UVA where it lies.
        let trace = models::trace_of("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships it");
        let trace = trace(Platform::Cuda);
        let full = Plan::of(&trace, &Attachments::new(), Budgets::uncapped())
            .expect("uncapped plans")
            .device_demand();
        let plan = Plan::of(&trace, &Attachments::new(), Budgets::device(full / 2))
            .expect("a dense plan under a budget spills rather than refusing");

        assert!(plan.streams(), "half the table cannot be held whole");
        assert!(plan.banks().is_empty(), "and none of it is a routed bank");
        assert!(!plan.groups().is_empty(), "what spilled is dense planes");
        assert!(
            plan.device_demand() <= full / 2,
            "the plan fits its budget: {} > {}",
            plan.device_demand(),
            full / 2
        );
        for group in plan.groups() {
            assert!(!group.routed, "`{}` is dense and says so", group.name);
            assert_eq!(group.experts, 0, "a dense plane has no expert axis");
            assert_eq!(group.planes.len(), 1, "and it is a group of one");
        }

        // **THE ORDER IS THE COMPILER'S SCHEDULE, READ FROM THE BACK.** The
        // plane a fire reaches FIRST is the plane a budget surrenders LAST, so
        // the embedding — read at node zero of every fire — stays resident and
        // the tail of the plan is what leaves.
        let schedule = model_compiler::prefetch::Schedule::of(&trace);
        let rank: BTreeMap<usize, usize> = schedule
            .order()
            .into_iter()
            .enumerate()
            .map(|(at, param)| (param, at))
            .collect();
        let spilled_first = plan
            .groups()
            .iter()
            .filter_map(|group| rank.get(&group.param).copied())
            .min()
            .expect("something spilled");
        let resident_last = (0..trace.params.len())
            .filter(|at| !plan.pinned(*at) && !plan.mapped(*at))
            .filter_map(|at| rank.get(&at).copied())
            .max()
            .expect("something stayed");
        assert!(
            spilled_first < resident_last || plan.groups().len() == trace.params.len(),
            "a plane read at rank {spilled_first} spilled while one at {resident_last} stayed, \
             which is not the schedule's order"
        );
        // The plane read FIRST is resident, whatever the budget did.
        let first = schedule.order()[0];
        assert!(
            !plan.pinned(first) && !plan.mapped(first),
            "the plane a fire reads first (`{}`) left the device",
            trace.params[first].name
        );
    }

    // ── THE LADDER (wave B7) ────────────────────────────────────────────────

    #[test]
    fn a_capped_packed_plan_names_every_group_exactly_once() {
        // `Plan::groups` is the placements a budget was FORCED to make and
        // `Plan::seated` is the routed groups the store kept — two lists,
        // disjoint, and between them every packed bank of the trace. The
        // ladder reads both; the standing gates read only the first, which is
        // why the second is a second list and not a variant in the first.
        let trace = gpt_oss();
        let planes = scales_of(&trace);
        let full = Plan::of(&trace, &planes, Budgets::uncapped())
            .expect("uncapped plans")
            .device_demand();
        let plan = Plan::of(&trace, &planes, Budgets::device(full * 7 / 10))
            .expect("seven tenths of the table serves");

        assert!(!plan.seated().is_empty(), "seven tenths of it stays");
        assert!(!plan.groups().is_empty(), "and three tenths does not");
        for group in plan.seated() {
            assert_eq!(group.held, Held::Device, "`{}` is in the store", group.name);
            assert!(group.routed, "a spilled dense plane is never a seat");
            for plane in &group.planes {
                assert!(
                    !plan.streamed_whole(plane.param),
                    "`{}` is seated in the store and the plan says it is elsewhere",
                    group.name
                );
            }
        }
        let mut named: BTreeSet<usize> = BTreeSet::new();
        for group in plan.seated().iter().chain(plan.groups()) {
            assert!(named.insert(group.param), "`{}` is in both lists", group.name);
        }
        assert_eq!(
            named.len(),
            planes.len() + plan.groups().iter().filter(|group| !group.routed).count(),
            "every packed bank of the trace is named once, and so is every spilled plane"
        );
    }

    /// A berth of `shape`, on `tier`, holding `holds`.
    fn berth(tier: Held, shape: &[u64], holds: Option<usize>) -> Berth {
        Berth {
            tier,
            at: shape.iter().enumerate().map(|(at, _)| 0x1000 + at as u64).collect(),
            shape: shape.to_vec(),
            holds,
        }
    }

    /// A group of `shape` on `held`, in berth `berth`, with a backing unless
    /// `backed` says otherwise.
    fn group(cell_at: usize, held: Held, shape: &[u64], berth: Option<usize>, backed: bool) -> Group {
        Group {
            param: cell_at,
            name: format!("bank.{cell_at}"),
            experts: 32,
            planes: shape
                .iter()
                .enumerate()
                .map(|(at, bytes)| GroupPlane {
                    param: cell_at * 8 + at,
                    bytes: *bytes,
                    reserved: *bytes,
                })
                .collect(),
            cell_at,
            at: vec![0; shape.len()],
            held,
            backing: if backed { vec![0xf000; shape.len()] } else { Vec::new() },
            berth,
            settled: 0,
        }
    }

    #[test]
    fn a_uniform_vote_is_a_steady_state_and_a_cold_bank_is_displaced() {
        // **THE ONE RULE, BOTH WAYS ROUND.** Two berths, one on each of the
        // ladder's two upper rungs, and three same-shaped groups: one in each
        // berth and one on the file.
        let shape = [1024u64, 64];
        let berths = vec![
            berth(Held::Device, &shape, Some(0)),
            berth(Held::Pinned, &shape, Some(1)),
        ];
        let groups = vec![
            group(0, Held::Device, &shape, Some(0), true),
            group(1, Held::Pinned, &shape, Some(1), true),
            group(2, Held::Mapped, &shape, None, true),
        ];

        // (a) UNIFORM DEMAND MOVES NOTHING. Every bank read every fire is
        //     every catalog MoE today, and the right answer is to stay put:
        //     with equal hits any assignment costs the same and a swap is
        //     ~265 MiB of PCIe bought for zero.
        assert!(
            vote(&berths, &groups, |_| 100).is_none(),
            "an equal vote moved a group, which is a churn and not a promotion"
        );

        // (b) A COLD OCCUPANT IS DISPLACED BY A HOT CANDIDATE, and by the
        //     BIGGEST improvement: the mapped group is hotter than both, and
        //     the device berth is the rung that buys the most.
        let hits = |at: usize| [1u32, 50, 900][at];
        let swap = vote(&berths, &groups, hits).expect("a cold device seat gives way");
        assert_eq!(swap.group, 2, "the hot mapped bank is what comes up");
        assert_eq!(swap.berth, 0, "into the fastest rung it can reach");

        // (c) AND IT TERMINATES. Applying the move by hand, the same counters
        //     answer the next question with the berth the candidate LEFT — not
        //     with the move undone.
        let mut moved = groups;
        moved[0].held = Held::Mapped;
        moved[0].berth = None;
        moved[2].held = Held::Device;
        moved[2].berth = Some(0);
        let mut berths = berths;
        berths[0].holds = Some(2);
        let next = vote(&berths, &moved, hits);
        assert!(
            next.is_none_or(|swap| swap.group != 0),
            "the group just demoted came straight back: {next:?}"
        );
    }

    #[test]
    fn a_berth_takes_only_a_group_it_was_sized_for() {
        // gpt-oss's 48 groups are two shapes — a gate/up bank and a down bank
        // — and a berth is sized for the group the plan put in it. A group of
        // the other shape taking it would run off the end of the region or
        // leave a hole the next plane's reader walks into, so the shapes are
        // compared plane for plane and everything across them is declined.
        let small = [512u64, 32];
        let large = [1024u64, 64];
        let berths = vec![berth(Held::Device, &small, Some(0))];
        let groups = vec![
            group(0, Held::Device, &small, Some(0), true),
            group(1, Held::Mapped, &large, None, true),
        ];
        assert!(
            vote(&berths, &groups, |at| [1u32, 900][at]).is_none(),
            "a bank of another shape took a berth sized for this one"
        );
    }

    #[test]
    fn an_occupant_with_no_backing_is_never_displaced() {
        // The swap's first half points the occupant's cell back at the file.
        // A load that opened no artifact has no file to point at — so it has
        // the assignment it booted with, and that is a shorter ladder rather
        // than a wrong one.
        let shape = [1024u64, 64];
        let berths = vec![berth(Held::Device, &shape, Some(0))];
        let groups = vec![
            group(0, Held::Device, &shape, Some(0), false),
            group(1, Held::Mapped, &shape, None, true),
        ];
        assert!(
            vote(&berths, &groups, |at| [1u32, 900][at]).is_none(),
            "a group with nowhere to be demoted to was demoted"
        );
    }

    #[test]
    fn an_empty_berth_takes_the_hottest_bank_that_has_fired() {
        // A berth whose occupant was promoted out of it is empty, and the next
        // gap fills it — that is how the ladder walks more than one rung. The
        // floor is zero, so what it declines is a bank that has NEVER fired:
        // spending PCIe on a group no route has reached is the one promotion
        // that cannot pay for itself.
        let shape = [1024u64, 64];
        let berths = vec![berth(Held::Pinned, &shape, None)];
        let groups = vec![
            group(0, Held::Mapped, &shape, None, true),
            group(1, Held::Mapped, &shape, None, true),
        ];
        let swap = vote(&berths, &groups, |at| [7u32, 90][at]).expect("an empty berth fills");
        assert_eq!(swap.group, 1, "with the hotter of the two");
        assert!(
            vote(&berths, &groups, |_| 0).is_none(),
            "a bank no route has reached was promoted on the strength of nothing"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    //  O1: THE ORDER THE WRITER LAYS OUT IS THE ORDER THE BOOT WALKS
    // ════════════════════════════════════════════════════════════════════


    /// The pairings the gate ranks with — [`model_dsl::scales_name`] and
    /// `biases_name`, the two functions that mint a companion's name, asked
    /// of a trace whose params those same calls declared.
    ///
    /// Host-only because `model_dsl` is a DEV dependency of this crate on
    /// purpose: a shell must not know a model family. The production caller
    /// that needs this order is `runtime::engine::load::sequence`, which is
    /// above that line and can ask.
    fn pairings_of(trace: &Trace) -> Attachments {
        let at: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        let mut out = Attachments::new();
        for (codes, param) in trace.params.iter().enumerate() {
            let mut companions = Vec::new();
            if let Some(&scales) = at.get(model_dsl::scales_name(&param.name).as_str()) {
                companions.push(scales);
            }
            if let Some(&biases) = at.get(model_dsl::biases_name(&param.name).as_str()) {
                companions.push(biases);
            }
            if !companions.is_empty() {
                out.insert(codes, companions);
            }
        }
        out
    }

    /// **The number [`Plan::cut`] refuses under**, recomputed from the public
    /// ranking: every registered plane's reserved span, plus one expert slot
    /// of every dense routed bank.
    ///
    /// `Ranking::floor_bytes` is private and stays private — the cut is its
    /// only reader in the shell. The gate needs the number to STAND on the
    /// floor deliberately (the pathological pair below), and recomputing it
    /// from `floor()` and `banks()` is the honest way to get it: if the two
    /// ever disagree, the `Plan::cut` call at that budget refuses and the gate
    /// says so by name.
    fn floor_of(ranking: &Ranking, bytes: &[u64]) -> u64 {
        let registered: u64 = ranking
            .floor()
            .iter()
            .map(|param| bytes[*param].next_multiple_of(crate::weights::ALIGN))
            .sum();
        let slots: u64 = ranking
            .banks()
            .iter()
            .map(|bank| bank.stride.next_multiple_of(crate::weights::ALIGN))
            .sum();
        registered + slots
    }

    /// **THE ORDER AND THE SPANS THE WRITER LAYS OUT ARE THE ORDER AND THE
    /// SPANS EVERY BOOT WALKS** — over every SKU the catalog ships, at five
    /// budget pairs each, host-only (§M.3, gate O1).
    ///
    /// # Why this claim and not another
    ///
    /// §M.3 made the serving artifact budget-FREE: the file states the
    /// RANKING, a boot states the CUT, and one artifact then serves any budget
    /// pair on this setup. Everything that ruling buys rests on one measured
    /// premise — *a plane's BYTES are identical on all three rungs and only
    /// the WALK differs* — and that premise is stated twice in this file, in
    /// two functions that share no code:
    ///
    /// ```text
    ///   Ranking::images    param, at, bytes, reserved   the WRITER's walk
    ///   Plan::host_layout  param, at, bytes, reserved   the T1 walk
    ///   Plan::mapped_layout                             the T2 walk
    ///        + the store, which is every ranked plane neither of those holds
    /// ```
    ///
    /// **A plane written at the wrong span is the quietest failure this wave
    /// can produce.** Nothing catches it. The digests agree, because the
    /// writer hashes what the writer wrote. The index tiles, because
    /// [`payload_fault`](crate::weight_cache::tier) checks the file against
    /// ITSELF and not against a plan. The artifact opens, verifies, and every
    /// lookup answers. The model then computes on bytes from the middle of
    /// another plane and produces garbage logits with no error anywhere — no
    /// refusal, no counter, no log line. The only way to catch it is to ask
    /// the two walks whether they describe the same planes, which is what is
    /// below.
    ///
    /// **AND UNTIL THIS TEST THAT QUESTION WAS ASKED IN EXACTLY ONE PLACE:**
    /// claim 4 of `a_prepare_writes_the_tiers_a_bare_boot_refuses` — on a
    /// card, against one SKU, behind `#[ignore]`, in a file the M-4 wave
    /// deletes. So the wave that turns the ranking into the container's own
    /// layout was about to remove the only standing check on its own central
    /// premise. This gate is that check, moved to where it can run on every
    /// commit: no device, no checkpoint, no artifact on a disk — a trace, a
    /// pairing, and arithmetic.
    ///
    /// # What is asserted, and what is deliberately not
    ///
    /// ```text
    ///  1. tiled     the images concatenate from zero: `at` is the running sum
    ///               of `reserved`, every offset is `weights::ALIGN`-aligned,
    ///               and `reserved` is `bytes` rounded up and nothing else
    ///  2. whole     the ranked set is the trace minus the registered floor —
    ///               no plane ranked and unplaced, none placed and unranked
    ///  3. agreed    per plane, `bytes` and `reserved` are the SAME numbers in
    ///               the ranking and in whichever rung's walk seats it
    ///  4. partition every ranked plane is on exactly one rung, and each rung's
    ///               walk tiles from zero the way the ranking's does
    ///  5. monotone  raising `device_weight_budget` never raises what the two
    ///               tiers below the store owe, and never raises T2's share of
    ///               it — AS TOTALS. See the footnote: two narrower readings of
    ///               (5) are FALSE by design and both counterexamples are named
    /// ```
    ///
    /// **A DENSE ROUTED BANK IS THE ONE PLANE ON TWO RUNGS AT ONCE, AND THAT
    /// IS NOT A HOLE IN (4).** [`Plan::cut`] argues it: the device slab takes
    /// the bank's first `resident` experts and the pinned tier holds ALL of
    /// them authoritatively, so a bank sits on neither side of a cut. It is
    /// asserted as its own claim rather than excused from the partition.
    ///
    /// # The two readings of (5) that are FALSE, measured, and right
    ///
    /// The cut is a FIRST FIT down a hotness order — [`Plan::cut`]'s "the walk
    /// continues past a group that does not fit", which is what makes the two
    /// cut points rank THRESHOLDS with fitted exceptions rather than strict
    /// prefixes. First fit is not monotone in its capacity, and the same
    /// sentence therefore has two false readings, one per rung:
    ///
    /// ```text
    ///   per plane   `dsv4-base-bf16-kv-bf16`, host uncapped. At a
    ///               361_515_888-byte device ceiling the 505 MiB `embed` does
    ///               not fit and 85 later planes do; at 653_825_760 `embed`
    ///               fits, and 26 planes including `layer.0.gate_up` move
    ///               from the store to T1
    ///   T1 alone    `gemma4-e4b-eagle-bf16-kv-bf16`, `host_weight_budget`
    ///               3_804_259_968. Device 3_845_547_648: T1 owes
    ///               3_803_447_296 and T2 owes 7_568_359_424. Device
    ///               5_740_796_352: T1 owes 3_804_259_840 — 812_544 bytes
    ///               MORE — and T2 owes 5_671_983_872, 1.9 GiB less
    /// ```
    ///
    /// **Both are the ranking doing its job.** `embed` is read at node zero of
    /// every fire and the planes it displaced are not; the group that grew
    /// T1 is one a freed T1 could finally hold whole, and it displaced smaller
    /// ones onto a tier that is slower for them and faster for it. What is
    /// monotone is what the design actually promises — the store never holds
    /// less, T2 never holds more, and the two tiers below the store never owe
    /// more together — and asserting the narrower forms would be asserting
    /// promises this planner deliberately does not make.
    ///
    /// **AND NEITHER COSTS THE ARTIFACT ANYTHING**, which is the reason they
    /// belong in this test's footnote rather than in a defect report: the
    /// RANKING does not move under either. The file states an order that
    /// mentions no budget, and every one of these plans walks it.
    #[test]
    fn the_ranking_lays_out_what_every_cut_walks() {
        let mut walked = 0usize;
        for (sku, _, trace_fn, _) in models::catalog() {
            let trace = trace_fn(Platform::Cuda);
            let planes = pairings_of(&trace);
            // **EVERY CATALOG ROW RANKS, AND A ROW THAT DOES NOT IS THIS
            // GATE'S FAILURE AND NOT ITS EXCEPTION.** This read `continue`
            // while `weights::plane_bytes` had no arm for the two-bit MLX
            // affine dtypes, and a sibling test named the two SKUs that fell
            // through so the skip could not grow quietly. The arm landed and
            // the two joined the walk, so the skip has nothing left to
            // excuse: a row that cannot be sized here refuses at `Ranking::of`
            // at EVERY budget, uncapped included, before a device is touched,
            // which is a load that never happens rather than a plan that
            // comes out wrong — loud, and worth failing on.
            let ranking = Ranking::of(&trace, &planes).unwrap_or_else(|why| {
                panic!(
                    "{sku}: this shell cannot rank a catalog row, so no budget \
                     serves it: {why:?}"
                )
            });
            walked += 1;
            let bytes = crate::weights::plane_bytes(&trace)
                .unwrap_or_else(|why| panic!("{sku}: a ranked trace sizes: {why:?}"));

            // ── (1) THE RANKING TILES ITS OWN PAYLOAD FROM ZERO.
            //
            //    This is `tier::payload_fault`'s invariant, stated on the
            //    RANKING rather than on a file — which is where it is actually
            //    decided. The file can only ever be as tiled as the walk that
            //    filled it, and a writer handed an untileable ranking would
            //    produce a file that passes `payload_fault` and describes the
            //    wrong bytes.
            let images = ranking.images();
            let mut at = 0u64;
            let mut ranked: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
            for (which, &(param, offset, span, reserved)) in images.iter().enumerate() {
                let param = usize::try_from(param).expect("a param ordinal");
                let name = &trace.params[param].name;
                assert_eq!(
                    offset, at,
                    "{sku}: image {which} `{name}` starts at byte {offset} and {at} was \
                     next — the ranking's images are a concatenation, and a boot that \
                     cuts them addresses this one {} bytes from where it lies",
                    offset.abs_diff(at),
                );
                assert_eq!(
                    offset % crate::weights::ALIGN,
                    0,
                    "{sku}: image {which} `{name}` starts at {offset}, which is not a \
                     multiple of {}: the one property that lets ONE image serve all \
                     three rungs is that every rung gives it the store's alignment",
                    crate::weights::ALIGN,
                );
                assert_eq!(
                    reserved,
                    span.next_multiple_of(crate::weights::ALIGN),
                    "{sku}: `{name}` publishes {span} bytes into a {reserved}-byte \
                     span; the padding between images is the ALIGN rounding and \
                     nothing else, so {} bytes here are unaccounted for",
                    reserved.abs_diff(span.next_multiple_of(crate::weights::ALIGN)),
                );
                assert_eq!(
                    span, bytes[param],
                    "{sku}: `{name}` is {span} bytes in the ranking and {} bytes in \
                     the trace it was ranked from",
                    bytes[param],
                );
                at += reserved;
                assert!(
                    ranked.insert(param, (span, reserved)).is_none(),
                    "{sku}: `{name}` is ranked twice — two images name one plane, and \
                     a boot would seat whichever it read last"
                );
            }
            let registered: u64 = ranking
                .floor()
                .iter()
                .map(|param| bytes[*param].next_multiple_of(crate::weights::ALIGN))
                .sum();
            assert_eq!(
                at + registered,
                ranking.full(),
                "{sku}: the payload tiles {at} bytes and the registered planes reserve \
                 {registered}; together they are the whole table, which is what every \
                 budget in this file is compared against"
            );

            // ── (2) THE RANKED SET IS THE TRACE MINUS THE REGISTERED FLOOR.
            let floor: BTreeSet<usize> = ranking.floor().iter().copied().collect();
            for (param, decl) in trace.params.iter().enumerate() {
                match floor.contains(&param) {
                    true => {
                        assert_eq!(
                            decl.source,
                            ParamSource::Registered,
                            "{sku}: `{}` is on the floor and is not a registered plane",
                            decl.name
                        );
                        assert!(
                            !ranked.contains_key(&param),
                            "{sku}: `{}` is an adapter bank the artifact deliberately \
                             omits (FLAG_ADAPTERS_ZEROED) and the ranking carries an \
                             image for it",
                            decl.name
                        );
                    }
                    false => assert!(
                        ranked.contains_key(&param),
                        "{sku}: `{}` is a plane of this trace that no image of the \
                         ranking carries — a boot would find nothing to seat it from",
                        decl.name
                    ),
                }
            }

            // ── THE FIVE BUDGET PAIRS, AND WHY THESE FIVE.
            //
            //    Each is a SHAPE of plan rather than a number: derived from
            //    this SKU's own ranking so that one list covers a 400 MiB
            //    draft model and a 568 GiB flash MoE.
            let full = ranking.full();
            let bottom = floor_of(&ranking, &bytes);
            // What T1 owes when it is the ONLY place the spill can go — the
            // number the third pair halves to force a plan onto T2.
            let spilling = Budgets::device(bottom.max(full / 2));
            let t1_only = Plan::cut(&ranking, spilling)
                .unwrap_or_else(|why| panic!("{sku}: half the table serves: {why:?}"));
            let cuts = [
                // The degenerate plan. Nothing is placed, nothing streams, and
                // the ranking still states every image — which is the whole of
                // why an uncapped boot can WRITE the file a capped one reads.
                ("uncapped", Budgets::uncapped()),
                // A stated ceiling that happens to be enough. `Plan::cut` takes
                // the same early return, and it must: a budget one byte over
                // full residency is full residency, not a streamed load that
                // keeps everything.
                ("the whole table, stated", Budgets::device(full)),
                // Two thirds of the table with NO host ceiling: the two-tier
                // shape W-5 planned. Something must give, and everything that
                // gives goes to T1 — so this is the pair that exercises
                // `host_layout` at its longest and leaves `mapped_layout`
                // empty.
                ("two thirds, host uncapped", Budgets::device(bottom.max(full * 2 / 3))),
                // Half the table and a host tier under what that spills: the
                // only pair that puts planes on ALL THREE rungs at once, which
                // is the pair the partition claim is actually about.
                (
                    "half the table, T1 under the spill",
                    Budgets {
                        device: spilling.device,
                        host: Some(t1_only.host_demand() / 2),
                    },
                ),
                // The pathological pair: the exact floor `Plan::cut` refuses
                // under, and no host tier at all. Every plane a budget is
                // allowed to move is on T2, which is the longest
                // `mapped_layout` this SKU has and the shape a deployment gets
                // on a box that cannot hold the model twice.
                (
                    "the floor exactly, no host tier",
                    Budgets {
                        device: Some(bottom),
                        host: Some(0),
                    },
                ),
            ];

            for (why, budgets) in cuts {
                let plan = Plan::cut(&ranking, budgets)
                    .unwrap_or_else(|fault| panic!("{sku} / {why}: {fault:?}"));

                // **THE TWO DOORS ARE ONE STATEMENT.** `Plan::of` is
                // `Ranking::of` and then `Plan::cut`, and a boot reaches the
                // layouts through whichever it happens to hold.
                let door = Plan::of(&trace, &planes, budgets)
                    .unwrap_or_else(|fault| panic!("{sku} / {why}: {fault:?}"));
                assert_eq!(
                    door.host_layout(),
                    plan.host_layout(),
                    "{sku} / {why}: `Plan::of` and `Plan::cut` lay T1 out differently"
                );
                assert_eq!(
                    door.mapped_layout(),
                    plan.mapped_layout(),
                    "{sku} / {why}: `Plan::of` and `Plan::cut` lay T2 out differently"
                );

                // ── (3) AND (4), ONE RUNG AT A TIME.
                let banks: BTreeSet<usize> = plan.banks().iter().map(|b| b.param).collect();
                let on_t1 = walk(sku, why, "T1", &trace, &ranked, &plan.host_layout());
                assert_eq!(
                    on_t1.1,
                    plan.host_image(),
                    "{sku} / {why}: the T1 walk ends at {} and the tier allocates {}",
                    on_t1.1,
                    plan.host_image(),
                );
                let on_t2 = walk(sku, why, "T2", &trace, &ranked, &plan.mapped_layout());
                assert_eq!(
                    on_t2.1,
                    plan.mapped_image(),
                    "{sku} / {why}: the T2 walk ends at {} and the section states {}",
                    on_t2.1,
                    plan.mapped_image(),
                );

                let (on_t1, on_t2) = (on_t1.0, on_t2.0);
                let mut counted = (0usize, 0usize, 0usize);
                for &param in ranked.keys() {
                    let name = &trace.params[param].name;
                    assert!(
                        !(plan.pinned(param) && plan.mapped(param)),
                        "{sku} / {why}: `{name}` is on T1 and T2 at once; a group is \
                         held whole on ONE rung and a torn pair is the failure that \
                         computes and is wrong"
                    );
                    match (plan.pinned(param), plan.mapped(param)) {
                        (true, _) => {
                            assert!(
                                on_t1.contains(&param),
                                "{sku} / {why}: `{name}` is pinned and the T1 walk \
                                 does not seat it — the landing would write it nowhere"
                            );
                            counted.1 += 1;
                        }
                        (_, true) => {
                            assert!(
                                on_t2.contains(&param),
                                "{sku} / {why}: `{name}` is mapped and the T2 walk \
                                 does not carry it — the handle would point past the \
                                 section"
                            );
                            counted.2 += 1;
                        }
                        // Neither: the device store holds it whole, which is
                        // the rung with no walk of its own because
                        // `weights::places` lays it out off `streamed_whole`.
                        _ => {
                            assert!(
                                !on_t2.contains(&param),
                                "{sku} / {why}: `{name}` is in the store and the T2 \
                                 walk carries it too"
                            );
                            assert!(
                                on_t1.contains(&param) == banks.contains(&param),
                                "{sku} / {why}: `{name}` is in the store and appears \
                                 in the T1 walk without being a dense routed bank — \
                                 only a bank is on two rungs at once"
                            );
                            counted.0 += 1;
                        }
                    }
                }
                assert_eq!(
                    counted.0 + counted.1 + counted.2,
                    ranked.len(),
                    "{sku} / {why}: the three rungs hold {counted:?} of {} ranked \
                     planes",
                    ranked.len(),
                );
                assert_eq!(
                    on_t2.len(),
                    counted.2,
                    "{sku} / {why}: the T2 walk carries {} planes and {} are mapped",
                    on_t2.len(),
                    counted.2,
                );
                assert_eq!(
                    on_t1.len(),
                    counted.1 + banks.len(),
                    "{sku} / {why}: the T1 walk seats {} spans and the plan pins {} \
                     planes plus {} whole banks",
                    on_t1.len(),
                    counted.1,
                    banks.len(),
                );

                // **THE ONE PLANE ON TWO RUNGS, ASSERTED RATHER THAN
                // EXCUSED.** A dense routed bank's slab is a cache over its
                // pinned copy, so it is in the store AND in the T1 walk, and
                // the store holds a PREFIX of it.
                for bank in plan.banks() {
                    let name = &bank.name;
                    assert!(
                        on_t1.contains(&bank.param),
                        "{sku} / {why}: `{name}` is streamed expert by expert and the \
                         pinned tier holds no copy for the table to point at"
                    );
                    assert!(
                        !plan.streamed_whole(bank.param),
                        "{sku} / {why}: `{name}`'s slab is in the store and the plan \
                         says the store does not hold it"
                    );
                    let seats = plan.resident(bank.param).unwrap_or_else(|| {
                        panic!("{sku} / {why}: `{name}` is streamed and seats no experts")
                    });
                    assert!(
                        (1..bank.experts).contains(&seats),
                        "{sku} / {why}: `{name}` declares {} experts and seats {seats} \
                         — a streamed bank seats at least one and never all of them",
                        bank.experts,
                    );
                    assert_eq!(seats, bank.resident, "{sku} / {why}: `{name}` two counts");
                }

                // The budget is a ceiling and the plan is under it.
                if let Some(ceiling) = budgets.device {
                    assert!(
                        plan.device_demand() <= ceiling,
                        "{sku} / {why}: the plan demands {} of a {ceiling}-byte store",
                        plan.device_demand(),
                    );
                }
            }

            // ── THE SHAPE EACH PAIR WAS CHOSEN FOR.
            let degenerate = Plan::cut(&ranking, Budgets::uncapped()).expect("uncapped cuts");
            assert!(
                !degenerate.streams(),
                "{sku}: an uncapped budget is the degenerate plan and streams nothing"
            );
            assert_eq!(
                degenerate.device_demand(),
                full,
                "{sku}: and it demands the whole table of the store"
            );
            assert!(
                Plan::cut(&ranking, Budgets::device(full))
                    .expect("a stated ceiling over the table cuts")
                    .host_layout()
                    .is_empty(),
                "{sku}: a ceiling at full residency IS full residency"
            );
            let wide = Plan::cut(&ranking, Budgets::device(bottom.max(full * 2 / 3)))
                .expect("two thirds cuts");
            assert!(
                wide.mapped_layout().is_empty(),
                "{sku}: an uncapped host tier spills nothing to the mapping, however \
                 tight the device is"
            );
            assert!(
                !wide.host_layout().is_empty(),
                "{sku}: two thirds of the table cannot be held whole"
            );
            let narrow = Plan::cut(
                &ranking,
                Budgets {
                    device: Some(bottom),
                    host: Some(0),
                },
            )
            .expect("the floor cuts");
            for group in ranking.sequence() {
                for plane in &group.planes {
                    assert!(
                        narrow.mapped(plane.param),
                        "{sku}: at the floor with no host tier, `{}` has one place \
                         left and the plan does not put it there",
                        group.name
                    );
                }
            }
            assert!(
                Plan::cut(&ranking, Budgets::device(bottom.saturating_sub(1))).is_err()
                    || bottom == 0,
                "{sku}: one byte under the floor is a budget that cannot hold the \
                 planes no budget can move, and it is refused by name"
            );

            // ── (5) THE CUT POINTS ARE MONOTONE IN THE BUDGETS.
            //
            //    Twice, because the two-tier shape and the three-tier one make
            //    different promises and only one of them is about T1 alone.
            //    The footnote on this test names what is deliberately NOT
            //    asserted here, and the SKU and budgets that disprove it.
            for host in [None, Some(t1_only.host_demand() / 2)] {
                let mut before: Option<(u64, u64, u64, u64)> = None;
                for step in 0..=8u64 {
                    let device = Some(bottom + (full - bottom) * step / 8);
                    let plan = Plan::cut(&ranking, Budgets { device, host })
                        .unwrap_or_else(|fault| panic!("{sku}: step {step}: {fault:?}"));
                    let ceiling = device.unwrap_or_default();
                    let below = plan.host_demand() + plan.spill_demand();
                    let now = (ceiling, plan.device_demand(), plan.spill_demand(), below);
                    let Some(was) = before else {
                        before = Some(now);
                        continue;
                    };
                    assert!(
                        now.1 >= was.1,
                        "{sku} / host {host:?}: the store took {} at a {}-byte ceiling \
                         and {} at a {}-byte one — a larger ceiling holds at least as \
                         much, or the cut is not reading the same ranking",
                        was.1,
                        was.0,
                        now.1,
                        now.0,
                    );
                    assert!(
                        now.2 <= was.2,
                        "{sku} / host {host:?}: T2 owed {} at a {}-byte device ceiling \
                         and {} at a {}-byte one — a plane never moves DOWN to the \
                         mapping because the device got bigger",
                        was.2,
                        was.0,
                        now.2,
                        now.0,
                    );
                    assert!(
                        now.3 <= was.3,
                        "{sku} / host {host:?}: the two tiers below the store owed {} \
                         at a {}-byte device ceiling and {} at a {}-byte one",
                        was.3,
                        was.0,
                        now.3,
                        now.0,
                    );
                    // **AND T1 ALONE ONLY WHEN IT IS UNCAPPED.** With a stated
                    // host ceiling the first fit runs on T1 too, and the
                    // footnote names the SKU where it grows.
                    if host.is_none() {
                        assert!(
                            plan.host_demand() <= was.3,
                            "{sku}: an uncapped T1 owed {} at a {}-byte device ceiling \
                             and {} at a {}-byte one — with nowhere below it to spill \
                             to, T1 holds exactly what the store gave up",
                            was.3,
                            was.0,
                            plan.host_demand(),
                            now.0,
                        );
                        assert_eq!(
                            plan.spill_demand(),
                            0,
                            "{sku}: an uncapped host tier spills nothing to T2"
                        );
                    }
                    before = Some(now);
                }
            }
        }
        assert!(
            walked == models::catalog().len(),
            "the gate walked {walked} of the catalog's {} rows",
            models::catalog().len()
        );
    }

    /// One rung's walk, checked against the ranking: it tiles from zero, every
    /// offset is aligned, and every span is the plane's OWN span — the same
    /// two numbers the artifact's image states.
    ///
    /// Returns the params it seated and where the walk ended, so the caller
    /// can ask the partition question with them.
    fn walk(
        sku: &str,
        why: &str,
        rung: &str,
        trace: &Trace,
        ranked: &BTreeMap<usize, (u64, u64)>,
        layout: &[(u64, u64, u64, u64)],
    ) -> (BTreeSet<usize>, u64) {
        let mut at = 0u64;
        let mut seen = BTreeSet::new();
        for (which, &(param, offset, span, reserved)) in layout.iter().enumerate() {
            let param = usize::try_from(param).expect("a param ordinal");
            let name = &trace.params[param].name;
            assert_eq!(
                offset, at,
                "{sku} / {why}: {rung} span {which} `{name}` starts at {offset} and \
                 {at} was next; the rung is a concatenation like the payload is"
            );
            assert_eq!(
                offset % crate::weights::ALIGN,
                0,
                "{sku} / {why}: {rung} span {which} `{name}` starts at {offset}, which \
                 the store would not have given it"
            );
            let &(want_span, want_reserved) = ranked.get(&param).unwrap_or_else(|| {
                panic!(
                    "{sku} / {why}: {rung} seats `{name}` and the ranking never ranked \
                     it — the boot would have no image to cut it out of"
                )
            });
            assert_eq!(
                (span, reserved),
                (want_span, want_reserved),
                "{sku} / {why}: `{name}` is {span}/{reserved} (bytes/reserved) on \
                 {rung} and {want_span}/{want_reserved} in the artifact. A plane's \
                 bytes are the SAME on all three rungs and only the WALK differs, so \
                 one of those two numbers cuts this plane at a length nothing \
                 published it at"
            );
            assert!(
                seen.insert(param),
                "{sku} / {why}: {rung} seats `{name}` twice, at {} and at {offset}",
                at,
            );
            at += reserved;
        }
        (seen, at)
    }

    /// **THE GEOMETRY IS THE SAME ON BOTH SHELLS AND THE BYTES ARE NOT**,
    /// which is why the artifact needs a STAMP and not a length check (M-4c).
    ///
    /// The peer wave that landed under this gate made qwen_3's projections ask
    /// which shell is serving: a mlxu4 row declares `U4g64tiled` when the
    /// trace is taken at [`Platform::Cuda`] and plain `U4g64` at
    /// `Platform::Metal`. A repack is size-PRESERVING — the band padding rides
    /// in the shape and `plane_bytes` puts both dtypes in one arm — so the
    /// ranking, every image span, and the whole payload are byte-for-byte
    /// identical between the two, while the nibbles inside each plane are in a
    /// different order.
    ///
    /// **That is the exact shape a length check cannot catch.** Two artifacts
    /// with the same key geometry, the same digestible payload, the same index
    /// — and one of them serves garbage on the other's card. Nothing in this
    /// file can tell them apart, and nothing in this file should try: the
    /// recipe stamp is the answer, and this test is the reason it has to be.
    #[test]
    fn a_tiled_row_and_a_row_major_one_rank_to_the_same_spans() {
        let trace = models::trace_of("qwen36-27b-mlxu4-kv-bf16").expect("the catalog ships it");
        let cuda = trace(Platform::Cuda);
        let metal = trace(Platform::Metal);
        let tiled = |trace: &Trace| {
            trace
                .params
                .iter()
                .filter(|param| param.dtype == model_ir::Dtype::U4g64tiled)
                .count()
        };
        assert!(
            tiled(&cuda) > 0 && tiled(&metal) == 0,
            "the SKU's projections read the serving shell: {} tiled planes on CUDA \
             and {} on Metal",
            tiled(&cuda),
            tiled(&metal),
        );
        let images = |trace: &Trace| {
            Ranking::of(trace, &pairings_of(trace))
                .expect("a mlxu4 row ranks")
                .images()
        };
        assert_eq!(
            images(&cuda),
            images(&metal),
            "a repack is size-preserving, so the two shells rank the same planes at \
             the same spans — and an artifact written by one has the geometry the \
             other would have written. Only the recipe stamp separates them"
        );
    }
}

