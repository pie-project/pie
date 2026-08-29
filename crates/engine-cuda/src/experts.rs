//! **The routed-expert tier**: a device slab smaller than the bank, a pinned
//! host copy of the whole of it, and a device-resident indirection table
//! between them (alto design §7, wave D2).
//!
//! ```text
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
//! # What does NOT stream in this wave
//!
//! The DENSE planes — embeddings, attention projections, norms, the head. A
//! `device_weight_budget` that cannot hold them is [`Fault::Residency`],
//! because their demand is not routed: nothing on the device chooses which of
//! them a fire reads, so a table over them would be a table every entry of
//! which is hit every fire. Design §7 gives them the OTHER demand shape — a
//! compiler-emitted prefetch schedule — and that is D2b.
//!
//! A split-plane quantized bank does not stream either, and for a plainer
//! reason: this shell does not yet SEAT one (`weights.rs` builds
//! `WeightRow::Dense` for every row), so there is no such bank to stream. The
//! scan below names only the ops whose bank resolves as one dense handle, and
//! a quantized MoE arriving on a capped budget refuses rather than being
//! quietly held whole.

use std::collections::BTreeMap;
use std::ffi::c_void;

use model_ir::{Def, Linear, Operation, Trace, ValueId};

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

/// **What a plan wants and what a budget allows**, decided off the trace
/// alone — before the device is bound and before a byte is allocated.
///
/// The empty plan is full residency and is what an uncapped
/// [`Residency`](engine::engine_api::load::Residency) produces. Everything
/// downstream reads [`Plan::streams`] and does nothing when it is false.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    banks: Vec<BankPlan>,
    /// `param index -> how many experts of it live on the device`. The one
    /// map `weights::places` consults to reserve a bank at less than its
    /// declared size.
    resident_of: BTreeMap<usize, u32>,
    device_bytes: u64,
    host_bytes: u64,
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

impl Plan {
    /// **The residency plan for `trace` under `device_budget`.**
    ///
    /// `None` — uncapped — answers the empty plan: land everything, open no
    /// tier, hand the kernels no table. A stated budget is met by holding
    /// fewer EXPERTS and never by holding fewer dense planes.
    ///
    /// # Errors
    ///
    /// [`Fault::Param`] for a param whose dtype has no element size or whose
    /// bank shape breaks the uniformity proof; [`Fault::Residency`] for a
    /// budget that cannot hold the dense planes plus one slot of every bank,
    /// naming both numbers.
    pub fn of(trace: &Trace, device_budget: Option<u64>) -> Result<Plan> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let full = bytes.iter().map(|b| b.next_multiple_of(crate::weights::ALIGN)).sum();
        let Some(budget) = device_budget else {
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

        let found = banks_of(trace)?;
        if found.is_empty() {
            return Err(Fault::Residency(format!(
                "`device_weight_budget` is {budget} bytes and this plan's weight table \
                 demands {full}. Nothing in it is a routed-expert bank, so there is no \
                 tier to hold less of: alto design §7 streams the DYNAMIC demand shape \
                 (routed experts, whose residency is a promotion) and the static one \
                 (dense overflow, a compiler-emitted prefetch schedule) is not built. \
                 Raise the budget, or state `None` for uncapped."
            )));
        }

        // ── THE DENSE FLOOR. Every plane that is not a routed bank is held
        //    whole, so its reserved bytes are the floor under any budget this
        //    plan can serve. A budget below it is refused with both numbers,
        //    and it is `Residency` -> `Impossible` rather than `OutOfMemory`
        //    -> `Exhausted`: nothing the deployment frees changes the answer.
        let experts = found[0].experts;
        let dense: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !found.iter().any(|bank| bank.param == *at))
            .map(|(_, plane)| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        let floor = dense
            + found
                .iter()
                .map(|bank| bank.stride.next_multiple_of(crate::weights::ALIGN))
                .sum::<u64>();
        if budget < floor {
            return Err(Fault::Residency(format!(
                "`device_weight_budget` is {budget} bytes; this plan's DENSE planes \
                 demand {dense} resident and its {} routed banks need one expert slot \
                 each on top, which is {floor} before a second expert is seated. Dense \
                 planes do not stream in this build (alto design §7: their demand is \
                 static and its prefetch schedule is D2b), so the budget cannot be met \
                 by holding less. Raise it to at least {floor}, or state `None`.",
                found.len(),
            )));
        }

        // How many experts every bank seats, one number for the plan. Monotone
        // in `n`, and `experts` is at most a few hundred, so it is walked
        // rather than searched — and walked DOWN from the whole bank so that a
        // budget one byte under full residency still lands on the largest
        // count it can hold.
        let slack = budget - dense;
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
        debug_assert!(resident >= 1, "the floor check above proved one slot fits");

        let banks: Vec<BankPlan> = found
            .into_iter()
            .map(|bank| BankPlan { resident, ..bank })
            .collect();
        let resident_of = banks.iter().map(|bank| (bank.param, resident)).collect();
        let device_bytes = dense
            + banks
                .iter()
                .map(|bank| (u64::from(resident) * bank.stride).next_multiple_of(crate::weights::ALIGN))
                .sum::<u64>();
        // **THE HOST TIER HOLDS EVERY EXPERT, NOT ONLY THE MISSING ONES.**
        // Pinned is the AUTHORITATIVE copy and the device slab is a cache over
        // it, which is what makes a demotion free: expert weights are
        // read-only, so evicting one is a table entry pointing back at bytes
        // that were never stale. The alternative — pinning only the
        // non-resident experts — would make every promotion a demotion's
        // write-back and would put the checkpoint back on the fire path's
        // horizon.
        let host_bytes = banks
            .iter()
            .map(|bank| u64::from(bank.experts) * bank.stride)
            .sum();
        Ok(Plan {
            banks,
            resident_of,
            device_bytes,
            host_bytes,
        })
    }

    /// Does this load stream any bank?
    #[must_use]
    pub fn streams(&self) -> bool {
        !self.banks.is_empty()
    }

    /// The banks it streams, in param order.
    #[must_use]
    pub fn banks(&self) -> &[BankPlan] {
        &self.banks
    }

    /// How many experts of `param` the device slab seats, or `None` for a
    /// param that is held whole — which is every param of a full-residency
    /// load and every dense plane of a streamed one.
    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        self.resident_of.get(&param).copied()
    }

    /// **What this plan demands of tier T0**, in bytes — what
    /// [`Residency::admit`](engine::engine_api::load::Residency::admit) is
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

/// **The routed banks a trace declares, with the uniformity proof checked.**
///
/// A bank is a param some `Linear::Moe*Select*` op reads at its `bank` port —
/// stated by the OP and not by a naming convention, the same rule
/// `weights::banks` follows for the adapter axis. The scan is over
/// `Trace::nodes` because that is where the reading is; `Trace::params` says
/// only what exists.
fn banks_of(trace: &Trace) -> Result<Vec<BankPlan>> {
    let bytes = crate::weights::plane_bytes(trace)?;
    let mut seen: BTreeMap<usize, ()> = BTreeMap::new();
    let mut out: Vec<BankPlan> = Vec::new();
    let mut arity: Option<u32> = None;
    for node in &trace.nodes {
        let Operation::Linear(op) = &node.op else {
            continue;
        };
        // THE THREE OPS, NAMED. `MoeMatmulSelect` reads one dense handle;
        // the two quantized twins read a split-plane bank this shell does
        // not seat, and they are named here so that a capped budget over one
        // refuses by name instead of falling through as "no banks found".
        let (bank, dense) = match op {
            Linear::MoeMatmulSelect { bank, .. } => (*bank, true),
            Linear::MoeMatmulSelectBias { bank, .. } | Linear::MoeMatmulSelectQuant { bank, .. } => {
                (*bank, false)
            }
            _ => continue,
        };
        let at = weight_of(trace, bank)?;
        if seen.insert(at, ()).is_some() {
            continue;
        }
        let param = &trace.params[at];
        if !dense {
            return Err(Fault::Residency(format!(
                "`{}` is a split-plane quantized expert bank, and this shell binds \
                 every weight row as one dense handle — there is no such bank here to \
                 stream. A quantized MoE under a stated `device_weight_budget` is \
                 refused rather than held whole: the day the weight table seats \
                 `WeightRow::Planes`, this is the line that already asks about it.",
                param.name
            )));
        }
        // ── THE UNIFORMITY PROOF, OFF THE PLAN (dev's, restated).
        let experts = u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(0);
        if experts == 0 || param.shape.len() < 2 {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is read as a routed expert bank and does not declare \
                      `[experts, ...]`; a slot stride cannot be divided out of it",
            });
        }
        let plane = bytes[at];
        if plane == 0 || plane % u64::from(experts) != 0 {
            return Err(Fault::Param {
                name: param.name.clone(),
                why: "is a routed expert bank whose bytes do not divide by its expert \
                      count — the experts of one bank are not equal, and the slot \
                      arithmetic the tier does would be wrong rather than refused",
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
        out.push(BankPlan {
            param: at,
            name: param.name.clone(),
            experts,
            resident: experts,
            stride: plane / u64::from(experts),
        });
    }
    out.sort_by_key(|bank| bank.param);
    Ok(out)
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
    /// counters. Called BEFORE the checkpoint lands, because the landing
    /// writes a streamed bank's planes straight into [`Tier::host`].
    ///
    /// # Errors
    ///
    /// [`Fault::OutOfMemory`] or [`Fault::Device`] for the allocations,
    /// [`Fault::Runtimeless`] without a runtime.
    pub fn open(plan: Plan) -> Result<Tier> {
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
        self.seats
            .iter()
            .map(|seat| BankResidency {
                name: seat.name.clone(),
                experts: seat.experts,
                slots: seat.resident,
                in_slot: seat.in_slot.clone(),
                hits: (0..seat.experts)
                    .map(|expert| {
                        let byte = (seat.counter_at + expert as usize) * COUNTER as usize;
                        hits.get(byte..byte + COUNTER as usize)
                            .and_then(|w| w.try_into().ok())
                            .map_or(0, u32::from_ne_bytes)
                    })
                    .collect(),
            })
            .collect()
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

    #[test]
    fn an_uncapped_budget_is_the_degenerate_plan() {
        let plan = Plan::of(&a3b(), None).expect("a bf16 MoE plans");
        assert!(!plan.streams(), "an uncapped load streams nothing");
        assert_eq!(plan.host_demand(), 0, "and holds no pinned copy of anything");
        assert!(plan.device_demand() > 0);
    }

    #[test]
    fn a_budget_over_the_whole_table_is_the_same_degenerate_plan() {
        let trace = a3b();
        let full = Plan::of(&trace, None).expect("uncapped plans").device_demand();
        let plan = Plan::of(&trace, Some(full)).expect("a budget at the demand plans");
        assert!(!plan.streams(), "a budget at full residency is `place_all`");
        assert_eq!(plan.device_demand(), full);
    }

    #[test]
    fn a_budget_under_the_table_seats_fewer_experts_and_pins_them_all() {
        let trace = a3b();
        let full = Plan::of(&trace, None).expect("uncapped plans").device_demand();
        let plan = Plan::of(&trace, Some(full / 2)).expect("half the table streams");
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
    fn a_budget_under_the_dense_planes_is_refused_by_name() {
        let trace = a3b();
        let why = Plan::of(&trace, Some(1 << 20)).expect_err("a megabyte holds nothing");
        let said = why.to_string();
        assert!(said.contains("DENSE"), "the refusal names the tier: {said}");
        assert!(said.contains("do not stream"), "and says why: {said}");
    }

    #[test]
    fn a_dense_plan_under_a_budget_is_refused_rather_than_streamed() {
        let trace = model::trace_of("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships it");
        let trace = trace(Platform::Cuda);
        let full = Plan::of(&trace, None).expect("uncapped plans").device_demand();
        let why = Plan::of(&trace, Some(full / 2)).expect_err("a dense plan cannot stream");
        assert!(
            why.to_string().contains("routed-expert bank"),
            "the refusal says what is missing: {why}"
        );
    }
}
