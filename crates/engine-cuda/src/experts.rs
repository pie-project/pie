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
//! without a kernel that asks, and kernels are not this wave's.
//!
//! **The unit of residency is therefore the GROUP** — the code plane and
//! every companion plane the load plan pairs with it — and a group is held
//! WHOLE on one tier: every plane of it in the device store, or every plane
//! of it in the pinned tier, addressed by the kernel over UVA. Which is not a
//! weaker promise than the dense tier's, it is a different one, and it is the
//! one that makes a torn pair unconstructible: a group never changes tier
//! after the load, so there is no instant at which the codes have moved and
//! the exponents have not. Seating the codes alone would be the failure the
//! metal sibling names in its own header — `scales += e * ...` reading
//! another expert's factors, a model that computes and is wrong — and this
//! shell does not have a shape that can express it.
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

/// **Which tier holds a streamed group's planes.**
///
/// A group is held WHOLE on exactly one of them — the header says why — so
/// this is a property of the group and never of a plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Held {
    /// T1: page-locked host memory the kernel reads over UVA. The tier owns
    /// the bytes and the landing sink writes them.
    Pinned,
    /// T2: the warm artifact, mapped. The tier owns nothing — the bytes were
    /// on disk before this load started and no one copies them.
    Mapped,
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

impl Plan {
    /// **The residency plan for `trace` under `device_budget`.**
    ///
    /// `None` — uncapped — answers the empty plan: land everything, open no
    /// tier, hand the kernels no table. A stated budget is met by holding
    /// fewer EXPERTS of a dense bank and fewer whole GROUPS of a packed one,
    /// and never by holding fewer dense planes.
    ///
    /// `planes` is the load plan's pairing — which other params move when a
    /// packed bank moves — read by [`weights::prospect`] before a byte is
    /// landed. An empty map is a plan with no packed bank, and a packed bank
    /// missing from it is refused by name rather than seated half.
    ///
    /// **`budgets.host` IS WHAT OPENS THE THIRD TIER.** A packed group the
    /// device budget cannot seat goes to T1; one the HOST budget cannot seat
    /// either goes to T2 — the mapped artifact — and the plan reports those
    /// bytes as [`Plan::spill_demand`]. This function never asks whether a T2
    /// source exists: that is a file on a disk, and the statute that decides
    /// whether a spilled load is servable is
    /// [`Residency::admit_tiers`](engine::load::Residency::admit_tiers).
    ///
    /// # Errors
    ///
    /// [`Fault::Param`] for a param whose dtype has no byte size, whose bank
    /// shape breaks the uniformity proof, or whose packed bank the load plan
    /// pairs no scales plane with; [`Fault::Residency`] for a budget that
    /// cannot hold the dense planes plus one slot of every dense bank, naming
    /// both numbers.
    ///
    /// [`weights::prospect`]: crate::weights::prospect
    pub fn of(trace: &Trace, planes: &Attachments, budgets: Budgets) -> Result<Plan> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let full = bytes.iter().map(|b| b.next_multiple_of(crate::weights::ALIGN)).sum();
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

        // **THE "NOTHING IS ROUTED" REFUSAL STOOD HERE, AND D2b RETIRED IT.**
        // It said that a plan with no routed-expert bank had no tier to hold
        // less of, because only the DYNAMIC demand shape streamed and the
        // static one — dense overflow — was not built. It is built: a dense
        // plane the budget cannot seat becomes a group of one and takes the
        // same three tiers (streaming §2). A plan with no bank at all is now
        // planned like any other, and what refuses is the floor below, which
        // is about the planes that genuinely cannot move.
        let (found, packed) = routed(trace, planes, &bytes)?;

        // ── THE FLOOR, WHICH IS NOW THE UNSPILLABLE PLANES AND NOT THE DENSE
        //    ONES (alto streaming §2's static demand shape, wave D2b).
        //
        //    A dense plane CAN leave the device now: it becomes a group of one
        //    and walks the same three tiers a packed bank walks, given up in
        //    reverse PREFETCH-SCHEDULE order so the plane a fire reads first is
        //    the plane a budget surrenders last. What cannot leave is a
        //    REGISTERED plane — `register_adapter` writes it at a store offset
        //    reserved at load, so a bank on another tier is a registration
        //    landing in bytes nobody reads. Those, plus one expert slot of
        //    every dense bank, are the floor under any budget this plan can
        //    serve; a budget below it is `Residency` -> `Impossible` rather
        //    than `OutOfMemory` -> `Exhausted`, because nothing the deployment
        //    frees changes the answer.
        let streamable: BTreeSet<usize> = found
            .iter()
            .map(|bank| bank.param)
            .chain(packed.iter().flat_map(|group| group.planes.iter().map(|plane| plane.param)))
            .collect();
        let experts = found.first().map_or(0, |bank| bank.experts);
        let mut pinned_floor = 0u64;
        let mut spillable: Vec<GroupPlan> = Vec::new();
        for (at, param) in trace.params.iter().enumerate() {
            if streamable.contains(&at) {
                continue;
            }
            let reserved = bytes[at].next_multiple_of(crate::weights::ALIGN);
            if param.source == ParamSource::Registered {
                pinned_floor += reserved;
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
        // **THE ORDER A BUDGET GIVES PLANES UP IN IS THE COMPILER'S.**
        // `prefetch::Schedule` is a pure function of the trace — the same
        // planes in the same order every fire — so this ordering is a
        // compile-time constant and two boots of one deployment spill the same
        // set. Read from the front, it is what a pump would copy first; read
        // from the back, it is what a ceiling surrenders first.
        let schedule = model_compiler::prefetch::Schedule::of(trace);
        //    The walk below offers T0 front to back, so the planes that fall
        //    off its end are the schedule's tail — which is the spill order,
        //    arrived at by construction rather than by a second sort.
        let rank: BTreeMap<usize, usize> = schedule
            .order()
            .into_iter()
            .enumerate()
            .map(|(at, param)| (param, at))
            .collect();
        spillable.sort_by_key(|group| (rank.get(&group.param).copied().unwrap_or(usize::MAX), group.param));
        let dense = pinned_floor;
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

        // ── THE PACKED GROUPS, SEATED WHOLE OR NOT AT ALL, ACROSS THREE
        //    TIERS.
        //
        //    Walked in PLAN ORDER and offered each tier in turn: the device
        //    while what is left of the device budget holds one whole, then the
        //    pinned host while what is left of the host budget does, then the
        //    mapping, which has no budget to run out of. A group that does not
        //    fit a tier falls through and the walk CONTINUES rather than
        //    stopping, because a later, smaller group that does fit is a group
        //    that tier may as well hold — and because the answer has to be a
        //    function of the trace and the two budgets alone, so that two
        //    boots of one deployment plan the same table.
        //
        //    There is no "seat half of it": the module header states the
        //    reason, and it is the whole of why this loop is over GROUPS.
        //
        //    **THE DENSE BANKS' PINNED COPY IS RESERVED FIRST, AT ITS WORST
        //    CASE.** A streamed dense bank pins EVERY expert, and whether it
        //    streams is decided below this loop — so the host budget is
        //    charged for it here as though it does. The conservatism can only
        //    send a group to T2 that T1 could have held, never the reverse,
        //    and it applies to no catalog SKU today: no model text declares a
        //    dense routed bank and a packed one in one plan.
        let mut left = budget - floor;
        let dense_pin: u64 = found
            .iter()
            .map(|bank| u64::from(bank.experts) * bank.stride)
            .sum();
        let mut host_left = budgets.host.map(|host| host.saturating_sub(dense_pin));
        let mut groups: Vec<GroupPlan> = Vec::new();
        let mut pinned_of: BTreeSet<usize> = BTreeSet::new();
        let mut mapped_of: BTreeSet<usize> = BTreeSet::new();
        let mut seated = 0u64;
        let mut spill_bytes = 0u64;
        //    **THE DENSE PLANES ARE OFFERED T0 BEFORE THE PACKED BANKS.**
        //    A dense plane is read by every token of every fire; a routed
        //    expert may not be read at all. So the unconditional demand takes
        //    the device first and the conditional one takes what is left,
        //    which is the only ordering that does not spend the scarcest tier
        //    on the least certain reader.
        for mut group in spillable.into_iter().chain(packed) {
            if group.bytes <= left {
                left -= group.bytes;
                seated += group.bytes;
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
        let mut host_bytes: u64 = banks
            .iter()
            .map(|bank| u64::from(bank.experts) * bank.stride)
            .sum();
        let pinned: u64 = groups
            .iter()
            .filter(|group| group.held == Held::Pinned)
            .map(|group| group.bytes)
            .sum();
        if pinned > 0 {
            // The groups' own offsets are aligned, so one rounding here is the
            // whole of what the seats above owe them. `Tier::open` does the
            // same arithmetic in the same order, and the two numbers agree.
            host_bytes = host_bytes.next_multiple_of(crate::weights::ALIGN);
            host_bytes += pinned;
        }
        Ok(Plan {
            banks,
            groups,
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
    /// Byte offset of expert 0 inside the pinned tier.
    host_at: u64,
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
    /// Byte offset of this plane inside the pinned tier.
    host_at: u64,
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

/// **The tier**: pinned bytes, a device slab, a table between them, and the
/// counters that decide what moves.
#[derive(Debug)]
pub struct Tier {
    plan: Plan,
    /// T1: every expert of every streamed bank.
    host: Pinned,
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
    /// offset inside [`Tier::host`]. Read by the weight table, never by a
    /// promotion: a group does not move.
    wholes: Vec<Whole>,
    /// **T2**: the mapped warm artifact, held open for the load's whole life
    /// because the weight rows point into it. `None` for every plan that
    /// spills nothing.
    source: Option<crate::weight_cache::Artifact>,
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
    /// `source` is the T2 artifact, already opened and verified by the caller.
    /// It MUST be `Some` for a plan with a non-zero
    /// [`Plan::spill_demand`] — the caller is
    /// [`Residency::admit_tiers`](engine::load::Residency::admit_tiers)'s
    /// other half and has already refused the load if it is not — and a
    /// mismatch here is [`Fault::Residency`] by name rather than a weight row
    /// pointing at nothing.
    ///
    /// # Errors
    ///
    /// [`Fault::OutOfMemory`] or [`Fault::Device`] for the allocations,
    /// [`Fault::Runtimeless`] without a runtime, [`Fault::Residency`] for a
    /// spilled plan with no source, a device that cannot dereference a
    /// mapping, or a plane the artifact does not carry.
    pub fn open(plan: Plan, source: Option<crate::weight_cache::Artifact>) -> Result<Tier> {
        let mut seats = Vec::with_capacity(plan.banks.len());
        let (mut host_at, mut entry_at, mut counter_at) = (0u64, 0usize, 0usize);
        for bank in plan.banks() {
            seats.push(Seat {
                param: bank.param,
                name: bank.name.clone(),
                experts: bank.experts,
                resident: bank.resident,
                stride: bank.stride,
                host_at,
                entry_at,
                counter_at,
                slab: 0,
                slot_of: vec![None; bank.experts as usize],
                in_slot: Vec::new(),
            });
            host_at += u64::from(bank.experts) * bank.stride;
            entry_at += bank.experts as usize;
            counter_at += bank.experts as usize;
        }
        // ── THE PACKED GROUPS, AFTER THE SEATS. Each plane is given the same
        //    alignment the device store would have given it, so a kernel that
        //    reinterprets the codes as 32-bit words reads an aligned address
        //    on this tier exactly as it does on the other. `Plan::of` rounds
        //    once, here, in the same order, so `host_demand` and this
        //    allocation are the same number.
        let mut wholes = Vec::new();
        if !plan.groups.is_empty() {
            host_at = host_at.next_multiple_of(crate::weights::ALIGN);
        }
        for group in plan.groups() {
            if group.held != Held::Pinned {
                continue;
            }
            for plane in &group.planes {
                wholes.push(Whole {
                    param: plane.param,
                    host_at,
                });
                host_at += plane.reserved;
            }
        }
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
                let Some(bytes) = artifact.plane(id, 0) else {
                    bump(Stat::Absent);
                    return Err(Fault::Residency(format!(
                        "`{}` is planned onto the mapped tier and the artifact at {} \
                         carries no plane {id}. The artifact is the snapshot a FULLY \
                         RESIDENT load of this deployment wrote, and its index is keyed \
                         by the plan's own param order — a hole in it means the file \
                         and this plan were written from different traces. Delete it \
                         and boot once uncapped.",
                        group.name,
                        artifact.path().display(),
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
        let entries = entry_at as u64 * ENTRY;
        let counters = counter_at as u64 * COUNTER;
        Ok(Tier {
            plan,
            host: Pinned::mapped(usize::try_from(host_at).unwrap_or(usize::MAX))?,
            table: Buffer::zeroed(usize::try_from(entries).unwrap_or(usize::MAX))?,
            shadow: Pinned::mapped(usize::try_from(entries).unwrap_or(usize::MAX))?,
            counts: Buffer::zeroed(usize::try_from(counters).unwrap_or(usize::MAX))?,
            mirror: Pinned::mapped(usize::try_from(counters).unwrap_or(usize::MAX))?,
            seats,
            wholes,
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
    #[must_use]
    pub fn pinned_at(&self, param: usize) -> Option<u64> {
        self.wholes
            .iter()
            .find(|whole| whole.param == param)
            .map(|whole| self.host.device() + whole.host_at)
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
    /// from.
    #[must_use]
    pub fn source(&self) -> Option<&crate::weight_cache::Artifact> {
        self.source.as_ref()
    }

    /// The pinned tier's host address, for the landing sink to store through.
    #[must_use]
    pub fn host(&self) -> &Pinned {
        &self.host
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
    /// # Errors
    ///
    /// [`Fault::Device`] for the copies, [`Fault::Runtimeless`] without a
    /// runtime.
    pub fn land(&mut self, slab_of: &[u64], stream: *mut c_void) -> Result<()> {
        debug_assert_eq!(slab_of.len(), self.seats.len());
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
                    self.host.device() + seat.host_at + u64::from(*expert) * seat.stride,
                    usize::try_from(seat.stride).unwrap_or(usize::MAX),
                )?;
            }
        }
        self.publish_all(stream)
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
                let value = address_of(seat, expert, &self.host);
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
        if self.counts.bytes() == 0 {
            return Ok(());
        }
        copy_any(
            stream,
            self.mirror.device(),
            self.counts.ptr(),
            self.counts.bytes(),
        )
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
        if self.seats.is_empty() {
            return Ok(0);
        }
        // The shadow is one allocation shared by every entry write, so a round
        // may not rewrite a word an earlier round's copy may still be reading.
        if self.moving && !self.ready.done()? {
            self.skipped += 1;
            return Ok(0);
        }
        let hits = self.mirror.read(0, self.mirror.bytes());
        let moves = self.decide(&hits);
        if moves.is_empty() {
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
                let value = pinned_address_of(seat, *out, &self.host);
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
                self.host.device() + seat.host_at + u64::from(*into) * seat.stride,
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
        self.ready.record(notify)?;
        self.ready.wait(compute)?;
        self.moving = true;
        Ok(u32::try_from(moves.len()).unwrap_or(u32::MAX))
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
        // planes are on the pinned tier whole; the device slab seats none of
        // its experts and counts none of its routing, because no kernel on
        // this path reads a table or an `atomicAdd`s a counter. A gate that
        // asks what is resident is told exactly that.
        let packed = self.plan.groups().iter().map(|group| BankResidency {
            name: group.name.clone(),
            experts: group.experts,
            slots: 0,
            in_slot: Vec::new(),
            hits: Vec::new(),
            held: Some(group.held),
        });
        dense.chain(packed).collect()
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
        let device = self.table.bytes() as u64 + self.counts.bytes() as u64;
        let host = self.host.bytes() as u64 + self.shadow.bytes() as u64 + self.mirror.bytes() as u64;
        (device, host)
    }
}

/// Where expert `expert` of `seat` lives right now.
fn address_of(seat: &Seat, expert: u32, host: &Pinned) -> u64 {
    match seat.slot_of[expert as usize] {
        Some(slot) => seat.slab + u64::from(slot) * seat.stride,
        None => pinned_address_of(seat, expert, host),
    }
}

/// Where expert `expert` of `seat` lives in the pinned tier — the address a
/// kernel dereferences over UVA when the expert is not on the device.
fn pinned_address_of(seat: &Seat, expert: u32, host: &Pinned) -> u64 {
    host.device() + seat.host_at + u64::from(expert) * seat.stride
}

#[cfg(test)]
mod tests {
    use model_dsl::Platform;

    use super::*;

    fn a3b() -> Trace {
        let trace = model::trace_of("qwen35-a3b-bf16-kv-bf16").expect("the catalog ships the SKU");
        trace(Platform::Cuda)
    }

    /// The catalog's one split-plane MoE: 32 experts, 24 layers, mxfp4 banks.
    fn gpt_oss() -> Trace {
        let trace =
            model::trace_of("gptoss-20b-bf16-mxfp4-kv-bf16").expect("the catalog ships the SKU");
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
        let trace = model::trace_of("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships it");
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
            "a plane read at rank {spilled_first} spilled while one at {resident_last} stayed,              which is not the schedule's order"
        );
        // The plane read FIRST is resident, whatever the budget did.
        let first = schedule.order()[0];
        assert!(
            !plan.pinned(first) && !plan.mapped(first),
            "the plane a fire reads first (`{}`) left the device",
            trace.params[first].name
        );
    }
}
