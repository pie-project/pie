//! **THE ROTATING DENSE PUMP** — alto streaming §3 item 4's last piece (D2b),
//! the seam the design named and left unbuilt.
//!
//! # What was already there, and what this is
//!
//! `model_compiler::prefetch::Schedule` is the fire-invariant order a plan
//! reads its params in, and `Schedule::slotting` is the PROVED round-robin
//! assignment of a spilled set onto a fixed number of device slots: plane `i`
//! and plane `i + S` share slot `i % S`, so **contents rotate and addresses
//! never** — a weight row built against slot `k` is correct for every fire of
//! this load. `experts::Tier` already holds the spilled bytes in page-locked
//! host memory (T1), and `device::graph::Event` is the fork/join.
//!
//! What was missing was the seam: rotating a slot's contents WITHIN a fire, at
//! region boundaries. The walk has exactly one such seam — `Sink::region_begin`
//! — and this module is what a shell's cursor calls there.
//!
//! # The choreography, as streaming §3 item 4 wrote it
//!
//! ```text
//! open       one copy stream; per slot k an Event ready[k] and free[k]
//! region r   compute: record(free[k]) for every slot whose tenant's last
//!                     read was region r-1
//!            copy:    for every tenant coming due within LOOKAHEAD regions:
//!                       wait(free[k]); copy the plane in; record(ready[k])
//!            compute: wait(ready[k]) for every tenant first read at r
//! ```
//!
//! **THE FIRE PATH NEVER WAITS BY DESIGN.** Nothing here synchronizes, nothing
//! here reads a device clock, and no host call blocks: the whole mechanism is
//! event edges between two streams, enqueued in the walk's own order. A copy
//! that has not landed when its region opens is a stall the COMPUTE STREAM
//! takes, counted at [`Rotor::observed`] and never load-bearing — article 2's
//! "slow, not wrong", one tier over.
//!
//! # Why an event's wait is enqueued AFTER its record, always
//!
//! `cudaStreamWaitEvent` captures the event's contents AT THE HOST CALL, not
//! at execution. So enqueueing `wait(free[k])` before the host has recorded
//! `free[k]` for the slot's current tenant would capture the PREVIOUS record —
//! a wait that lets the copy overwrite bytes a kernel is still reading, and a
//! race no synchronization downstream can see. That is why [`Rotor::at`]
//! carries an occupancy table rather than a plain issue cursor: a slot whose
//! tenant has not been RELEASED ON THE HOST is a slot the issue cursor stops
//! at, whatever the device would have done.
//!
//! # Capture legality: DECLINED BY NAME, and the reason is the census
//!
//! The rotation runs on the EAGER path only, and a load that rotates declines
//! graph recording the way a buffered fire does (`serve.rs`, design §6's own
//! sentence). Two reasons, and the first is enough:
//!
//! 1. **A replay does not walk.** A captured graph is launched without the
//!    host loop that owns the issue cursor, so a captured rotation would have
//!    to bake all of a fire's copies as nodes — which is possible, the
//!    schedule being fire-invariant — and then the ring's backpressure becomes
//!    intra-graph edges fixed at record time rather than a cursor. That is a
//!    different design, not this one.
//! 2. **The copies and their events are not in the walk's own order.** A copy
//!    node and the event node behind it are forked onto a second stream, so
//!    they stand in no position of the parent chain the capture frontier
//!    walks — and a captured region whose nodes cannot be placed is a region
//!    a replay cannot stand for.
//!
//! So: a typed decline at the router beats a wrong graph at the replay, and
//! the EAGER walk serves these loads today — counted, not silent
//! (`record::BodyTally::eager_rotating`, and the boot line `Shell::load`
//! prints when a load arms a rotor under a recording mode).
//!
//! # What is NOT rotated, and why that is not a failure
//!
//! Two classes stay where they lie and are read over UVA exactly as they were
//! before this module existed:
//!
//! * **A plane bigger than [`SLOT_CAP`].** A slot is one rectangle for its
//!   whole life and is sized for its biggest tenant, so one 300 MB head in the
//!   rotation would cost a 300 MB slot — device bytes the budget that spilled
//!   it just said were absent. A capped rotation buys the overlap for the
//!   bytes that ARE the mass (a tower's projections) and leaves the tail on
//!   the tier that already serves it correctly.
//! * **A group held [`Held::Mapped`](crate::experts::Held::Mapped)** — T2, the
//!   artifact. Its pages are not page-locked, so an async H2D out of them is
//!   not async; the pump for T2 is the T2→T1 promotion streaming §3 item 3
//!   names as unbuilt, and this module does not pretend to be it.
//!
//! # The accounting this module does NOT do
//!
//! The slot arena is device weight bytes that `experts::Plan` did not plan.
//! It is reported ([`Rotation::arena`]) and it is bounded by statute, and it
//! is NOT subtracted from `device_weight_budget` — because the unified
//! accounting sentence (streaming §3 item 5, next.md B1/B2) is not
//! implemented, and adding a fourth uncoordinated accounting would make that
//! wave harder rather than easier. Named here so the next wave finds it.

use core::cell::Cell;
use core::ffi::c_void;

use model_compiler::CompiledModel;
use model_compiler::prefetch::Schedule;

use crate::device::alloc::Buffer;
use crate::device::graph::Event;
use crate::error::{Fault, Result};

/// **The deepest ring worth building**, in slots.
///
/// A statute with a measurement behind it and not a knob (article 9). On the
/// W-2 rig — 165 spilled planes over 137 regions, 910 MB a step — the spilled
/// step time against ring depth is
///
/// ```text
///   no pump   37.73 ms/step        (the planes read where they lie, over UVA)
///    9 slots  37.21 ms/step        147 MB of arena
///   16 slots  35.77 ms/step        237 MB
///   32 slots  36.24 ms/step        403 MB
/// ```
///
/// against a computed PCIe floor of 36.42 ms. The curve is FLAT past the knee
/// because the tier is bandwidth-bound: once the copy stream never idles, more
/// slots buy nothing and cost device memory. Thirty-two is past the knee on
/// every shape measured, and [`ARENA_CAP`] is what actually chooses.
pub const DEPTH_MAX: u32 = 32;

/// **How far ahead of a region's own boundary a copy is issued**, in regions.
///
/// The device throttles itself on `free[k]`, so this only decides how much
/// host work happens at which boundary. Two regions is one layer of runway on
/// a plan whose regions are layer-granular and a no-op on one whose regions
/// are finer.
pub const LOOKAHEAD: u32 = 2;

/// **The largest plane a slot will hold.** See the header: a slot is sized for
/// its biggest tenant, so the cap is what keeps the arena from re-seating the
/// very plane the budget refused. 32 MiB holds every projection of every
/// catalog row and declines the embedding and the head.
pub const SLOT_CAP: u64 = 32 << 20;

/// **THE WHOLE SLOT ARENA'S CEILING**, and the statute that actually picks the
/// depth.
///
/// The arena is device weight bytes `experts::Plan` did not plan (see the
/// header), so the honest bound is on the TOTAL and not on a slot count: a
/// ring deep enough to matter on a plan of 4 MB planes is a ring that would
/// re-seat a third of the table on a plan of 30 MB ones. [`Rotation::plan`]
/// takes the DEEPEST ring that fits under this, which on the W-2 rig is the
/// 16 slots at the measured knee.
///
/// 256 MiB, which is half a percent of the card this was measured on and is
/// the same order as the [`crate::staged_h2d`] pool's pinned footprint: the
/// pump is a buffer, and a buffer that costs a fifth of what it moves has
/// stopped being one.
pub const ARENA_CAP: u64 = 256 << 20;

/// **One plane's tenancy**: which slot holds it, and the region span over
/// which the slot is spoken for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tenant {
    /// Index into `Trace::params`.
    pub param: usize,
    /// The slot this plane always lands in — a compile-time constant.
    pub slot: u32,
    /// The plane's bytes.
    pub bytes: u64,
    /// The first region that reads it: where the compute stream waits.
    pub first: u32,
    /// The last region that reads it, inclusive: the slot is free from
    /// `last + 1` onward.
    pub last: u32,
}

/// **A PROVED ROTATION**: the tenants in schedule order, the slots they share,
/// and the per-region program a cursor runs.
///
/// Pure host arithmetic — no device call is made to build one — so the whole
/// of the decision is testable without a card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rotation {
    tenants: Vec<Tenant>,
    slots: u32,
    slot_bytes: Vec<u64>,
    declined: Vec<usize>,
    /// Region -> the tenants first read there. The compute stream waits their
    /// `ready` here.
    acquire: Vec<Vec<u32>>,
    /// Region -> the tenants whose last read was the PREVIOUS region. The
    /// compute stream records their `free` here.
    release: Vec<Vec<u32>>,
    /// Region -> the tenants whose copies are issued here.
    issue: Vec<Vec<u32>>,
}

/// **Why a candidate set does not rotate.** Every arm is a decline and none is
/// an error: the planes stay on the tier that already serves them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decline {
    /// Nothing was spilled that a slot may hold.
    Nothing,
    /// The rotation would need more slots than it has planes, which is
    /// residency by another name — and residency is what the budget refused.
    /// Carries the count the region-granular proof asked for.
    Residency { want: u32, planes: u32 },
    /// `Schedule::slotting` refused the count at NODE granularity, which is a
    /// plan whose live ranges genuinely overlap. Carries the compiler's own
    /// sentence.
    Overlap(String),
    /// The SHALLOWEST legal ring already costs more device memory than
    /// [`ARENA_CAP`] allows — a plan whose planes are too big for a pump to be
    /// a buffer rather than a second residency.
    Arena { need: u64, cap: u64 },
}

impl core::fmt::Display for Decline {
    fn fmt(&self, out: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Decline::Nothing => write!(
                out,
                "no spilled plane is both page-locked and under the slot cap"
            ),
            Decline::Residency { want, planes } => write!(
                out,
                "{planes} planes would need {want} slots to rotate, which is \
                 {planes} planes seated — and seating them is what the device \
                 budget refused"
            ),
            Decline::Overlap(why) => write!(out, "{why}"),
            Decline::Arena { need, cap } => write!(
                out,
                "the shallowest legal ring for this plan costs {need} bytes of \
                 slots against a {cap}-byte ceiling — a pump that big is a \
                 second residency, not a buffer"
            ),
        }
    }
}

impl Rotation {
    /// **Plan a rotation for `candidates`**, or say why there is none.
    ///
    /// `candidates` is `(param, bytes)` for every plane the budget spilled onto
    /// a page-locked tier, in any order. The schedule decides the ORDER, the
    /// region projection decides the SHALLOWEST legal ring, `arena_cap` decides
    /// the DEEPEST affordable one, and `Schedule::slotting` decides — and
    /// PROVES — the assignment.
    ///
    /// # The second proof, and why the compiler's is not enough
    ///
    /// `Schedule::slotting` proves the assignment at NODE granularity:
    /// `last_read(i) <= first_read(i + S)`, half-open, so a plane may be
    /// overwritten the moment the node after its last read begins. A pump
    /// works at REGION granularity and enqueues a whole region's launches at
    /// once, so it needs the strictly stronger
    /// `last_region(i) < first_region(i + S)` — otherwise the compute stream
    /// would be asked to wait on a copy whose slot only frees inside the
    /// region already being enqueued, which is a deadlock and not a stall.
    /// That is arithmetic over `Schedule::against`, and it is done here
    /// because the granularity is the CONSUMER's and not the compiler's.
    ///
    /// # Errors
    ///
    /// Never. A set that cannot rotate answers [`Decline`], because not
    /// rotating is a correct load and refusing one would be a shell declining
    /// to serve a model it can serve.
    #[must_use]
    pub fn plan(
        schedule: &Schedule,
        compiled: &CompiledModel,
        candidates: &[(usize, u64)],
        cap: u64,
        arena_cap: u64,
    ) -> core::result::Result<Rotation, Decline> {
        let regions = compiled.regions.len().max(1);
        // The region span of every param the schedule knows, keyed by param.
        let spans = schedule.against(compiled);
        let mut region_of: std::collections::BTreeMap<usize, core::ops::Range<u32>> =
            std::collections::BTreeMap::new();
        for (row, span) in schedule.reads().iter().zip(&spans) {
            if !row.unread() {
                region_of.insert(row.param, span.clone());
            }
        }

        // ── WHO IS ELIGIBLE. A plane over the cap, or one no region reads,
        //    stays where it lies; the declines are carried so a report can say
        //    what did not move rather than leaving it invisible.
        let mut kept: Vec<(usize, u64)> = Vec::new();
        let mut declined: Vec<usize> = Vec::new();
        for (param, bytes) in candidates {
            if *bytes > cap || *bytes == 0 || !region_of.contains_key(param) {
                declined.push(*param);
            } else {
                kept.push((*param, *bytes));
            }
        }
        if kept.is_empty() {
            return Err(Decline::Nothing);
        }
        // Schedule order, which is the order `Slotting` deals in.
        kept.sort_by_key(|(param, _)| {
            schedule
                .read_of(*param)
                .map_or((u32::MAX, *param), |row| (row.span.start, row.param))
        });
        let planes = u32::try_from(kept.len()).unwrap_or(u32::MAX);

        // ── THE REGION-GRANULAR DEPTH. For tenant `i`, every tenant whose
        //    first region is at or before `i`'s LAST region must be dealt into
        //    a different slot, so the count has to exceed how many of them
        //    there are past `i`.
        let mut least = 1u32;
        for (at, (param, _)) in kept.iter().enumerate() {
            let last = region_of[param].end.saturating_sub(1);
            let ahead = kept
                .iter()
                .skip(at)
                .take_while(|(other, _)| region_of[other].start <= last)
                .count();
            least = least.max(u32::try_from(ahead).unwrap_or(u32::MAX));
        }
        if least >= planes {
            return Err(Decline::Residency {
                want: least,
                planes,
            });
        }

        // ── AND NOW THE DEPTH, WHICH IS AN ARENA QUESTION AND NOT A COUNT.
        //    A slot is one rectangle for its whole life, sized for its biggest
        //    tenant, so what a ring costs depends on the plan's shape and not
        //    on the number of slots alone. Take the DEEPEST ring that fits
        //    under the ceiling — deeper decouples the copy stream further from
        //    compute, and the curve flattens at the bandwidth bound, so
        //    "deepest affordable" is both the right answer and a bounded one.
        let arena_of = |slots: u32| -> u64 {
            let mut seats = vec![0u64; slots as usize];
            for (at, (_, bytes)) in kept.iter().enumerate() {
                let seat = &mut seats[at % slots as usize];
                *seat = (*seat).max(*bytes);
            }
            seats.iter().sum()
        };
        let ceiling = DEPTH_MAX.min(planes.saturating_sub(1)).max(least);
        let Some(slots) = (least..=ceiling)
            .rev()
            .find(|slots| arena_of(*slots) <= arena_cap)
        else {
            return Err(Decline::Arena {
                need: arena_of(least),
                cap: arena_cap,
            });
        };

        // ── AND THE COMPILER'S OWN PROOF, ON TOP. `least` is necessary at
        //    region granularity and `slotting` is what says the node spans
        //    agree; a count that satisfies the first and not the second is a
        //    plan whose live ranges overlap for a reason the projection hid.
        let params: Vec<usize> = kept.iter().map(|(param, _)| *param).collect();
        let slotting = schedule
            .slotting(&params, slots)
            .map_err(|why| Decline::Overlap(why.to_string()))?;

        let tenants: Vec<Tenant> = kept
            .iter()
            .map(|(param, bytes)| {
                let span = &region_of[param];
                Tenant {
                    param: *param,
                    slot: slotting.slot_of(*param).unwrap_or(0),
                    bytes: *bytes,
                    first: span.start,
                    last: span.end.saturating_sub(1),
                }
            })
            .collect();

        let mut slot_bytes = vec![0u64; slots as usize];
        for tenant in &tenants {
            let seat = &mut slot_bytes[tenant.slot as usize];
            *seat = (*seat).max(tenant.bytes);
        }

        // ── THE PER-REGION PROGRAM, laid out once so the fire path reads a
        //    vector instead of scanning a list.
        let mut acquire = vec![Vec::new(); regions];
        let mut release = vec![Vec::new(); regions];
        let mut issue = vec![Vec::new(); regions];
        for (at, tenant) in tenants.iter().enumerate() {
            let at = u32::try_from(at).unwrap_or(u32::MAX);
            acquire[(tenant.first as usize).min(regions - 1)].push(at);
            // A tenant whose last read is the plan's last region is released
            // by the NEXT fire's region zero, which is where the reset lives.
            let free_at = tenant.last as usize + 1;
            if free_at < regions {
                release[free_at].push(at);
            }
            issue[(tenant.first.saturating_sub(LOOKAHEAD) as usize).min(regions - 1)].push(at);
        }
        // The issue cursor is monotone over tenants, so each region's issue
        // list must be too — and it is, `tenants` being in schedule order and
        // the shift being uniform.
        for list in &mut issue {
            list.sort_unstable();
        }

        Ok(Rotation {
            tenants,
            slots,
            slot_bytes,
            declined,
            acquire,
            release,
            issue,
        })
    }

    /// The tenants, in schedule order.
    #[must_use]
    pub fn tenants(&self) -> &[Tenant] {
        &self.tenants
    }

    /// How many slots the rotation uses.
    #[must_use]
    pub const fn slots(&self) -> u32 {
        self.slots
    }

    /// **What the slot arena costs on the device** — the sum of the slots,
    /// each sized for its biggest tenant.
    #[must_use]
    pub fn arena(&self) -> u64 {
        self.slot_bytes.iter().sum()
    }

    /// The bytes that rotate — what this pump moves every step.
    #[must_use]
    pub fn rotating(&self) -> u64 {
        self.tenants.iter().map(|tenant| tenant.bytes).sum()
    }

    /// The spilled planes this rotation left where they lie, by param.
    #[must_use]
    pub fn declined(&self) -> &[usize] {
        &self.declined
    }
}

/// **What the pump has done** — a §14 register, counted and never read back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Observed {
    /// Fires this rotor pumped.
    pub fires: u64,
    /// Copies issued.
    pub copies: u64,
    /// Bytes those copies moved.
    pub bytes: u64,
    /// **The counted exception**: a region that opened while its slot's copy
    /// was still outstanding on the host cursor — the pump falling behind its
    /// own schedule. Never a fault, never a control input.
    pub late: u64,
}

/// **THE PUMP ITSELF**: the slots, the events, the copy stream, and the
/// cursor a walk advances at each region boundary.
///
/// **HELD BY THE LOAD AND BORROWED BY A FIRE.** The slots are pointer-stable
/// for the life of the load (article 7) — that is the whole reason a weight
/// row may name one — and every per-fire number is a [`Cell`], because a
/// `Sink` method takes `&mut self` on the CURSOR and the cursor holds this by
/// shared reference beside the `Run` that reads the same weights.
pub struct Rotor {
    rotation: Rotation,
    /// One device rectangle per slot, sized for its biggest tenant.
    slots: Vec<Buffer>,
    /// Where each tenant's bytes come from: a page-locked host address in the
    /// pinned tier. Parallel to [`Rotation::tenants`].
    source: Vec<*const u8>,
    /// Recorded on the COPY stream after a plane has landed; waited on the
    /// compute stream at the region that first reads it.
    ready: Vec<Event>,
    /// Recorded on the COMPUTE stream after a slot's tenant is finished with;
    /// waited on the copy stream before the next tenant overwrites it.
    free: Vec<Event>,
    /// The copy stream. Opened here and destroyed in `Drop`; nothing else runs
    /// on it and nothing captured ever does.
    copy: *mut c_void,
    /// Which tenant each slot currently holds, and whether the host has
    /// already recorded its `free`. See the header on why this is host state
    /// and not a device question.
    occupant: Vec<Cell<Option<u32>>>,
    released: Vec<Cell<bool>>,
    /// The monotone issue cursor, reset at every region zero.
    next: Cell<u32>,
    fires: Cell<u64>,
    copies: Cell<u64>,
    bytes: Cell<u64>,
    late: Cell<u64>,
}

// SAFETY: a `Rotor` is device handles plus host `Cell`s that only the fire
// path touches, and the fire path is serialized by the shell's own lock — the
// same discipline `experts::Tier` and every arena in this crate run on. The
// raw pointers are a stream handle and page-locked source addresses the tier
// owns for the life of the load.
unsafe impl Send for Rotor {}

impl core::fmt::Debug for Rotor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Rotor")
            .field("slots", &self.rotation.slots)
            .field("tenants", &self.rotation.tenants.len())
            .field("arena", &self.rotation.arena())
            .field("observed", &self.observed())
            .finish()
    }
}

impl Rotor {
    /// **Open the pump**: allocate the slots, mint the events, open the copy
    /// stream.
    ///
    /// `source` is each tenant's page-locked host address, in
    /// [`Rotation::tenants`] order — the caller's, because only the loader
    /// knows which tier a plane landed in.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`]
    /// for a slot, an event or a stream the runtime refused, and
    /// [`Fault::Residency`] for a source list that does not describe the
    /// rotation it was planned for.
    pub fn open(rotation: Rotation, source: Vec<*const u8>) -> Result<Rotor> {
        if source.len() != rotation.tenants.len() {
            return Err(Fault::Residency(format!(
                "the rotation has {} tenants and the loader answered {} source \
                 addresses",
                rotation.tenants.len(),
                source.len(),
            )));
        }
        let mut slots = Vec::with_capacity(rotation.slots as usize);
        for bytes in &rotation.slot_bytes {
            slots.push(Buffer::zeroed(usize::try_from(*bytes).unwrap_or(usize::MAX))?);
        }
        let mut ready = Vec::with_capacity(slots.len());
        let mut free = Vec::with_capacity(slots.len());
        for _ in 0..slots.len() {
            ready.push(Event::new()?);
            free.push(Event::new()?);
        }
        let occupant = (0..slots.len()).map(|_| Cell::new(None)).collect();
        let released = (0..slots.len()).map(|_| Cell::new(true)).collect();
        Ok(Rotor {
            copy: copy_stream()?,
            slots,
            source,
            ready,
            free,
            occupant,
            released,
            next: Cell::new(0),
            fires: Cell::new(0),
            copies: Cell::new(0),
            bytes: Cell::new(0),
            late: Cell::new(0),
            rotation,
        })
    }

    /// The rotation this pump runs.
    #[must_use]
    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// **The device address a weight row for `param` must name** — the slot's
    /// base, fixed for the life of the load.
    #[must_use]
    pub fn seat(&self, param: usize) -> Option<u64> {
        let tenant = self
            .rotation
            .tenants
            .iter()
            .find(|tenant| tenant.param == param)?;
        Some(self.slots[tenant.slot as usize].ptr())
    }

    /// What this pump has done. See [`Observed`].
    #[must_use]
    pub fn observed(&self) -> Observed {
        Observed {
            fires: self.fires.get(),
            copies: self.copies.get(),
            bytes: self.bytes.get(),
            late: self.late.get(),
        }
    }

    /// **THE SEAM.** One region boundary, in the order streaming §3 item 4
    /// wrote it: release, issue, acquire.
    ///
    /// `compute` is the stream the region's launches are about to go on.
    /// Region zero is also the fire boundary: the cursor resets and every slot
    /// the last fire left occupied is released against the compute stream,
    /// which by then holds all of that fire's reads.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for a record, a wait or a copy the runtime refused.
    /// Nothing here is a policy refusal: a rotation that cannot pump is one
    /// that was never planned.
    pub fn at(&self, region: u32, compute: *mut c_void) -> Result<()> {
        let at = region as usize;
        if region == 0 {
            self.begin(compute)?;
        }
        // ── RELEASE. The slot's tenant was last read in region `at - 1`, and
        //    every launch of that region is already enqueued on `compute`, so
        //    recording here is recording after the last read of it.
        for tenant in self.rotation.release.get(at).into_iter().flatten() {
            let slot = self.rotation.tenants[*tenant as usize].slot as usize;
            self.free[slot].record(compute)?;
            self.released[slot].set(true);
        }
        // ── ISSUE. Everything due within the lookahead, in schedule order,
        //    stopping at the first slot whose tenant the host has not yet
        //    released — see the header on why that stop is not optional.
        let due = self
            .rotation
            .issue
            .get(at)
            .and_then(|list| list.last().copied())
            .map_or(self.next.get(), |last| last + 1);
        while self.next.get() < due {
            let which = self.next.get();
            if !self.issue(which)? {
                break;
            }
            self.next.set(which + 1);
        }
        // ── ACQUIRE. The compute stream waits for the plane it is about to
        //    read. A tenant the issue cursor has not reached is the counted
        //    exception: the wait would capture a stale `ready`, so the pump
        //    issues it now — late, correct, and recorded as late.
        for tenant in self.rotation.acquire.get(at).into_iter().flatten() {
            let which = *tenant;
            if self.next.get() <= which {
                self.late.set(self.late.get() + 1);
                while self.next.get() <= which {
                    let issuing = self.next.get();
                    if !self.issue(issuing)? {
                        // The slot's previous tenant is not released, which
                        // the region proof says cannot happen for a tenant due
                        // now. Say so rather than enqueueing a wait on a stale
                        // event.
                        return Err(Fault::Residency(format!(
                            "the rotation's slot {} still holds tenant {:?} when \
                             tenant {issuing} is due at region {region}",
                            self.rotation.tenants[issuing as usize].slot,
                            self.occupant[self.rotation.tenants[issuing as usize].slot as usize]
                                .get(),
                        )));
                    }
                    self.next.set(issuing + 1);
                }
            }
            let slot = self.rotation.tenants[which as usize].slot as usize;
            self.ready[slot].wait(compute)?;
        }
        Ok(())
    }

    /// The fire boundary: release whatever the last fire left held, and put
    /// the issue cursor back at the first tenant.
    fn begin(&self, compute: *mut c_void) -> Result<()> {
        for slot in 0..self.slots.len() {
            if self.occupant[slot].get().is_some() && !self.released[slot].get() {
                self.free[slot].record(compute)?;
                self.released[slot].set(true);
            }
        }
        self.next.set(0);
        self.fires.set(self.fires.get() + 1);
        Ok(())
    }

    /// One tenant's copy, or `false` for a slot whose occupant the host has
    /// not released yet — which is the ring's whole backpressure.
    fn issue(&self, which: u32) -> Result<bool> {
        let tenant = self.rotation.tenants[which as usize];
        let slot = tenant.slot as usize;
        if self.occupant[slot].get().is_some() {
            if !self.released[slot].get() {
                return Ok(false);
            }
            // The record this captures is the one the RELEASE above made, and
            // that is the whole ordering argument: host record, then host
            // wait, then the device may do as it likes.
            self.free[slot].wait(self.copy)?;
        }
        copy_in(
            self.copy,
            self.slots[slot].ptr(),
            self.source[which as usize],
            tenant.bytes,
        )?;
        self.ready[slot].record(self.copy)?;
        self.occupant[slot].set(Some(which));
        self.released[slot].set(false);
        self.copies.set(self.copies.get() + 1);
        self.bytes.set(self.bytes.get() + tenant.bytes);
        Ok(true)
    }
}

impl Drop for Rotor {
    fn drop(&mut self) {
        #[cfg(feature = "_cuda")]
        if !self.copy.is_null() {
            // SAFETY: the handle is this module's own `cudaStreamCreate`,
            // destroyed exactly once. Destroying a stream is asynchronous with
            // respect to the work on it and the runtime holds the slots until
            // it drains; the buffers below are freed after it either way.
            unsafe {
                let _ = cudarc::runtime::sys::cudaStreamSynchronize(self.copy.cast());
                let _ = cudarc::runtime::sys::cudaStreamDestroy(self.copy.cast());
            }
        }
    }
}

/// The pump's own stream. Non-blocking, so a copy never serializes against the
/// legacy default stream the way a plain `cudaStreamCreate` would.
fn copy_stream() -> Result<*mut c_void> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        let mut stream: rt::cudaStream_t = core::ptr::null_mut();
        // SAFETY: a live local out-parameter; the stream is this rotor's and
        // is destroyed exactly once in `Drop`.
        unsafe {
            crate::device::ctx::check(
                "cudaStreamCreateWithFlags (the rotor's copy stream)",
                rt::cudaStreamCreateWithFlags(&raw mut stream, 1 /* cudaStreamNonBlocking */),
            )?;
        }
        Ok(stream.cast())
    }
    #[cfg(not(feature = "_cuda"))]
    {
        Err(Fault::Runtimeless)
    }
}

/// One plane, page-locked host memory to a device slot, asynchronous.
fn copy_in(stream: *mut c_void, dst: u64, src: *const u8, bytes: u64) -> Result<()> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: `dst` is a slot this rotor allocated at least `bytes` long,
        // `src` is page-locked host memory the tier holds for the life of the
        // load, and `stream` is this rotor's own.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpyAsync (the rotor's plane)",
                rt::cudaMemcpyAsync(
                    dst as *mut c_void,
                    src.cast(),
                    usize::try_from(bytes).unwrap_or(usize::MAX),
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream.cast(),
                ),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, dst, src, bytes);
        Err(Fault::Runtimeless)
    }
}

#[cfg(test)]
mod tests {
    use model_compiler::{Budget, Budgets, DeviceProfile, compile_axes};
    use model_dsl::Platform;

    use super::*;
    use crate::experts::{Budgets as Tiers, Plan};

    const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

    /// The W-2 rig's own model and its own budget: two fifths of the table,
    /// which is what `a_spilled_dense_model_says_what_it_said` loads.
    fn rig() -> (model_ir::Trace, model_compiler::CompiledModel, Plan) {
        let trace = models::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
        let compiled = compile_axes(
            &trace,
            &Budgets {
                tokens: Budget::new(4, 256),
                patches: None,
            },
            &DeviceProfile::default(),
        )
        .expect("the plan bakes");
        let whole = Plan::of(&trace, &Default::default(), Tiers::uncapped())
            .expect("a dense plan plans")
            .device_demand();
        let plan = Plan::of(&trace, &Default::default(), Tiers::device(whole * 2 / 5))
            .expect("a dense plan under a budget spills rather than refusing");
        (trace, compiled, plan)
    }

    fn candidates(plan: &Plan) -> Vec<(usize, u64)> {
        plan.groups()
            .iter()
            .filter(|group| {
                !group.routed
                    && group.held == crate::experts::Held::Pinned
                    && group.planes.len() == 1
            })
            .map(|group| (group.param, group.bytes))
            .collect()
    }

    fn rotation() -> (model_ir::Trace, Rotation) {
        let (trace, compiled, plan) = rig();
        let schedule = Schedule::of(&trace);
        let rotation = Rotation::plan(&schedule, &compiled, &candidates(&plan), SLOT_CAP, ARENA_CAP)
            .expect("W-2's spilled set rotates");
        (trace, rotation)
    }

    /// **THE PROPERTY THE WHOLE MECHANISM RESTS ON**, asserted rather than
    /// assumed: a slot's tenant is finished being read STRICTLY BEFORE the
    /// region its successor is first read in.
    ///
    /// Not the same statement `Schedule::slotting` proves. The compiler's
    /// proof is half-open in NODES — a plane may be overwritten the instant
    /// the node after its last read begins — and a pump enqueues a whole
    /// region's launches at one boundary, so an assignment legal in nodes and
    /// illegal in regions would ask the compute stream to wait on a copy whose
    /// slot only frees inside the region already being enqueued. That is a
    /// deadlock, not a stall, and this is the line that says it cannot happen.
    #[test]
    fn a_slot_is_free_a_whole_region_before_its_next_tenant_arrives() {
        let (_, rotation) = rotation();
        let slots = rotation.slots() as usize;
        assert!(rotation.tenants().len() > slots, "there is rotation to do");
        for pair in rotation.tenants().windows(slots + 1) {
            let (evicted, by) = (&pair[0], &pair[slots]);
            assert_eq!(evicted.slot, by.slot, "round-robin puts them in one slot");
            assert!(
                evicted.last < by.first,
                "`{evicted:?}` is still live when `{by:?}` arrives",
            );
        }
    }

    /// **THE ADDRESSES ARE A COMPILE-TIME CONSTANT** (streaming §2). Two plans
    /// of one deployment place the same plane in the same slot, which is what
    /// makes a weight row built against a slot correct forever — and what
    /// would make a captured graph correct too, the day one is wanted.
    #[test]
    fn the_same_plan_places_the_same_plane_in_the_same_slot() {
        let (_, one) = rotation();
        let (_, two) = rotation();
        assert_eq!(one, two);
    }

    /// **A SLOT IS SIZED FOR ITS BIGGEST TENANT AND THE ARENA IS THE SUM.**
    /// The number a residency accounting will want, stated where it is
    /// computed — and bounded on both axes, which is the two caps' whole job:
    /// an unbounded arena would re-seat the plane the budget refused.
    #[test]
    fn the_arena_is_bounded_by_its_own_ceiling() {
        let (_, rotation) = rotation();
        assert!(rotation.arena() <= ARENA_CAP, "the arena statute is a ceiling");
        assert!(rotation.slots() <= DEPTH_MAX);
        assert!(
            rotation.arena() < rotation.rotating(),
            "an arena that held the rotating bytes would be residency, not a pump",
        );
        for tenant in rotation.tenants() {
            assert!(tenant.bytes <= SLOT_CAP);
        }
    }

    /// **THE CAP DECLINES RATHER THAN REFUSES.** A plane too big for a slot is
    /// left on the tier already serving it — the load is unchanged for it, and
    /// the pump is unchanged for everything else.
    #[test]
    fn a_plane_over_the_cap_stays_where_it_lies() {
        let (trace, compiled, plan) = rig();
        let schedule = Schedule::of(&trace);
        let all = candidates(&plan);
        let tiny = all.iter().map(|(_, bytes)| *bytes).min().expect("spilled");
        let rotation = Rotation::plan(&schedule, &compiled, &all, tiny, ARENA_CAP)
            .expect("the smallest planes still rotate");
        assert!(
            !rotation.declined().is_empty(),
            "a cap at the smallest plane declines every other one",
        );
        for tenant in rotation.tenants() {
            assert!(tenant.bytes <= tiny);
        }
        assert_eq!(
            rotation.tenants().len() + rotation.declined().len(),
            all.len(),
            "every candidate either rotates or is named as declining to",
        );
    }

    /// **THE ARENA CEILING IS WHAT PICKS THE DEPTH**, and picking it is
    /// monotone: a tighter ceiling never buys a deeper ring. Below the
    /// shallowest LEGAL ring it stops buying anything at all, which is the
    /// test below.
    #[test]
    fn a_tighter_arena_buys_a_shallower_ring() {
        let (trace, compiled, plan) = rig();
        let schedule = Schedule::of(&trace);
        let all = candidates(&plan);
        let mut last = u32::MAX;
        for ceiling in [ARENA_CAP, ARENA_CAP * 3 / 4, ARENA_CAP * 5 / 8] {
            let rotation = Rotation::plan(&schedule, &compiled, &all, SLOT_CAP, ceiling)
                .expect("W-2's set rotates under every ceiling that holds its floor");
            assert!(rotation.arena() <= ceiling, "{ceiling}: {rotation:?}");
            assert!(
                rotation.slots() <= last,
                "a tighter ceiling deepened the ring"
            );
            last = rotation.slots();
        }
    }

    /// **AN ARENA THAT CANNOT HOLD THE SHALLOWEST LEGAL RING DECLINES**, which
    /// is the sentence that keeps a pump from becoming a second residency.
    #[test]
    fn a_ceiling_under_the_shallowest_legal_ring_declines_by_name() {
        let (trace, compiled, plan) = rig();
        let schedule = Schedule::of(&trace);
        let why = Rotation::plan(&schedule, &compiled, &candidates(&plan), SLOT_CAP, 1)
            .expect_err("one byte of arena holds no ring at all");
        assert!(matches!(why, Decline::Arena { cap: 1, .. }), "{why}");
    }

    /// **A CAP THAT ADMITS NOTHING IS A DECLINE AND NOT A FAULT.**
    #[test]
    fn a_set_with_nothing_under_the_cap_declines_by_name() {
        let (trace, compiled, plan) = rig();
        let schedule = Schedule::of(&trace);
        assert_eq!(
            Rotation::plan(&schedule, &compiled, &candidates(&plan), 0, ARENA_CAP),
            Err(Decline::Nothing),
        );
    }

    /// **A RUNTIMELESS BUILD OPENS NO PUMP** — this crate's standing property,
    /// and what lets the planning tests above ride a plain workspace check.
    #[test]
    #[cfg(not(feature = "_cuda"))]
    fn a_runtimeless_build_opens_no_rotor() {
        let (_, rotation) = rotation();
        let source = vec![core::ptr::null(); rotation.tenants().len()];
        assert!(matches!(
            Rotor::open(rotation, source),
            Err(Fault::Runtimeless)
        ));
    }
}
