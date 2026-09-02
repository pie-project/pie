//! Rotating dense pump: cycles spilled dense weight planes through a small,
//! fixed ring of device slots at region boundaries within a fire — contents
//! rotate, but a slot's device address never changes. Per region `r`:
//! release `free[k]` for slots whose tenant's last read was `r-1`, issue
//! copies for tenants due within [`LOOKAHEAD`] regions (waiting `free[k]`
//! first), then have compute wait `ready[k]` for tenants first read at `r`.
//! A copy that hasn't landed stalls the compute stream, counted at
//! [`Rotor::observed`]; nothing on the fire path blocks otherwise.
//!
//! `cudaStreamWaitEvent` captures an event's contents at the host call, not
//! at execution, so a wait must never be enqueued before its matching
//! record — [`Rotor::at`] tracks host-side occupancy rather than a plain
//! issue cursor for this reason. Rotation runs on the eager path only; a
//! load that rotates declines graph recording, since a replay doesn't walk
//! the host issue cursor.
//!
//! Not rotated: a plane bigger than [`SLOT_CAP`] (stays on its tier, read
//! over UVA), and a group held as `Held::Mapped` (its pages aren't
//! page-locked, so an async H2D isn't actually async).

use core::cell::Cell;
use core::ffi::c_void;

use model_compiler::CompiledModel;
use model_compiler::prefetch::Schedule;

use crate::device::alloc::Buffer;
use crate::device::graph::Event;
use crate::error::{Fault, Result};

/// Deepest ring worth building, in slots. Measured on a 165-plane, 137-region
/// rig: gains flatten past ~16 slots since the tier is bandwidth-bound, so
/// deeper rings just cost device memory. [`ARENA_CAP`] is what actually picks
/// the depth in practice.
pub const DEPTH_MAX: u32 = 32;

/// How far ahead of a region's own boundary a copy is issued, in regions.
/// The device throttles itself on `free[k]`, so this only affects how much
/// host work happens at which boundary.
pub const LOOKAHEAD: u32 = 2;

/// Largest plane a slot will hold; a slot is sized for its biggest tenant,
/// so this bounds the arena. 32 MiB holds every projection in the catalog
/// and declines the embedding and the head.
pub const SLOT_CAP: u64 = 32 << 20;

/// Ceiling on the whole slot arena — the statute that actually picks the
/// ring depth, since a slot's cost depends on the plan's plane sizes and not
/// slot count alone. [`Rotation::plan`] takes the deepest ring that fits
/// under this. 256 MiB, the same order as the [`crate::staged_h2d`] pool.
pub const ARENA_CAP: u64 = 256 << 20;

/// One plane's tenancy: which slot holds it, and the region span over
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

/// A proved rotation: the tenants in schedule order, the slots they share,
/// and the per-region program a cursor runs.
///
/// Pure host arithmetic — no device call is made to build one — so it is
/// testable without a card.
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

/// Why a candidate set does not rotate. Every arm is a decline, not an
/// error: the planes stay on the tier that already serves them.
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
    /// Plans a rotation for `candidates` (`(param, bytes)` for every plane
    /// spilled onto a page-locked tier, any order), or says why there is
    /// none. Needs a proof stronger than `Schedule::slotting`'s node-granular
    /// one: a pump enqueues a whole region's launches at once, so it
    /// requires `last_region(i) < first_region(i + S)`, else the compute
    /// stream could wait on a copy whose slot frees inside the region
    /// already being enqueued (a deadlock, not a stall).
    ///
    /// # Errors
    ///
    /// Never. A set that cannot rotate answers [`Decline`] instead, since not
    /// rotating is still a correct load.
    #[must_use]
    pub fn plan(
        schedule: &Schedule,
        compiled: &CompiledModel,
        candidates: &[(usize, u64)],
        cap: u64,
        arena_cap: u64,
    ) -> core::result::Result<Rotation, Decline> {
        let regions = compiled.regions.len().max(1);
        // Region span of every param the schedule knows, keyed by param.
        let spans = schedule.against(compiled);
        let mut region_of: std::collections::BTreeMap<usize, core::ops::Range<u32>> =
            std::collections::BTreeMap::new();
        for (row, span) in schedule.reads().iter().zip(&spans) {
            if !row.unread() {
                region_of.insert(row.param, span.clone());
            }
        }

        // A plane over the cap, or one no region reads, stays where it
        // lies; declines are carried so a report can name what didn't move.
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
        // Schedule order, the order `Slotting` deals in.
        kept.sort_by_key(|(param, _)| {
            schedule
                .read_of(*param)
                .map_or((u32::MAX, *param), |row| (row.span.start, row.param))
        });
        let planes = u32::try_from(kept.len()).unwrap_or(u32::MAX);

        // Region-granular depth: every tenant overlapping `i`'s last region
        // must land in a different slot.
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

        // Depth is an arena question, not a slot count: take the deepest
        // ring that fits under the ceiling.
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

        // `least` is necessary at region granularity; `slotting` confirms
        // the node spans agree too.
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

        // Per-region program, laid out once so the fire path reads a
        // vector instead of scanning a list.
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
        // Issue cursor is monotone over tenants, so each region's issue
        // list must be sorted too.
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

    /// What the slot arena costs on the device: the sum of the slots, each
    /// sized for its biggest tenant.
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

/// What the pump has done — counted, never read back on the fire path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Observed {
    /// Fires this rotor pumped.
    pub fires: u64,
    /// Copies issued.
    pub copies: u64,
    /// Bytes those copies moved.
    pub bytes: u64,
    /// Counted exception: a region opened while its slot's copy was still
    /// outstanding — the pump falling behind its own schedule. Never a fault.
    pub late: u64,
}

/// The pump itself: slots, events, copy stream, and the cursor a walk
/// advances at each region boundary.
///
/// Held by the load, borrowed by a fire: slots are pointer-stable for the
/// load's life, and every per-fire number is a [`Cell`] since a `Sink`
/// method takes `&mut self` on the cursor while this is held by shared
/// reference alongside the `Run` reading the same weights.
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
    /// Which tenant each slot holds, and whether the host has recorded its
    /// `free` yet — host state, since `cudaStreamWaitEvent` captures at the
    /// host call, not at execution.
    occupant: Vec<Cell<Option<u32>>>,
    released: Vec<Cell<bool>>,
    /// The monotone issue cursor, reset at every region zero.
    next: Cell<u32>,
    fires: Cell<u64>,
    copies: Cell<u64>,
    bytes: Cell<u64>,
    late: Cell<u64>,
}

// SAFETY: the fire path is serialized by the shell's own lock, so the host
// `Cell`s are never touched concurrently; the raw pointers are a stream
// handle and page-locked source addresses the tier owns for the load's life.
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
    /// Opens the pump: allocates the slots, mints the events, opens the copy
    /// stream. `source` is each tenant's page-locked host address, in
    /// [`Rotation::tenants`] order (the caller's, since only the loader
    /// knows which tier a plane landed in).
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

    /// The device address a weight row for `param` must name — the slot's
    /// base, fixed for the load's life.
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

    /// One region boundary, in order: release, issue, acquire. `compute` is
    /// the stream the region's launches are about to go on. Region zero is
    /// also the fire boundary: the cursor resets and every slot the last
    /// fire left occupied is released.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for a record, a wait or a copy the runtime refused.
    pub fn at(&self, region: u32, compute: *mut c_void) -> Result<()> {
        let at = region as usize;
        if region == 0 {
            self.begin(compute)?;
        }
        // Release: tenant's last read was region `at - 1`, already
        // enqueued on `compute`.
        for tenant in self.rotation.release.get(at).into_iter().flatten() {
            let slot = self.rotation.tenants[*tenant as usize].slot as usize;
            self.free[slot].record(compute)?;
            self.released[slot].set(true);
        }
        // Issue everything due within the lookahead, stopping at the first
        // slot whose tenant isn't released yet.
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
        // Acquire: wait for the plane about to be read. A tenant not yet
        // issued is the counted exception — issue it now, late but correct.
        for tenant in self.rotation.acquire.get(at).into_iter().flatten() {
            let which = *tenant;
            if self.next.get() <= which {
                self.late.set(self.late.get() + 1);
                while self.next.get() <= which {
                    let issuing = self.next.get();
                    if !self.issue(issuing)? {
                        // Previous tenant not released — the region proof
                        // says this can't happen; report rather than wait
                        // on a stale event.
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
            // Captures the release record above: host record, then host
            // wait, then device order is free.
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
        #[cfg(feature = "cuda")]
        if !self.copy.is_null() {
            // SAFETY: handle is this module's own `cudaStreamCreate`,
            // destroyed exactly once. Stream destroy is async w.r.t. its
            // work; buffers below are freed after it either way.
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
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        let mut stream: rt::cudaStream_t = core::ptr::null_mut();
        // SAFETY: live local out-parameter; stream is this rotor's,
        // destroyed once in `Drop`.
        unsafe {
            crate::device::ctx::check(
                "cudaStreamCreateWithFlags (the rotor's copy stream)",
                rt::cudaStreamCreateWithFlags(&raw mut stream, 1 /* cudaStreamNonBlocking */),
            )?;
        }
        Ok(stream.cast())
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Fault::Runtimeless)
    }
}

/// One plane, page-locked host memory to a device slot, asynchronous.
fn copy_in(stream: *mut c_void, dst: u64, src: *const u8, bytes: u64) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: `dst` is a slot this rotor allocated at least `bytes`
        // long; `src` is page-locked host memory the tier holds for the
        // load's life; `stream` is this rotor's own.
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
    #[cfg(not(feature = "cuda"))]
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

    // Reference rig: model and budget (2/5 of the table) shared by the tests below.
    fn rig() -> (model_ir::Trace, model_compiler::CompiledModel, Plan) {
        let trace = (models::sku(SKU).expect("the catalog ships the SKU").trace)(Platform::Cuda);
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

    // An arena too small for the shallowest legal ring declines by name.
    #[test]
    fn a_ceiling_under_the_shallowest_legal_ring_declines_by_name() {
        let (trace, compiled, plan) = rig();
        let schedule = Schedule::of(&trace);
        let why = Rotation::plan(&schedule, &compiled, &candidates(&plan), SLOT_CAP, 1)
            .expect_err("one byte of arena holds no ring at all");
        assert!(matches!(why, Decline::Arena { cap: 1, .. }), "{why}");
    }

    #[cfg(not(feature = "cuda"))]
    fn rotation() -> (model_ir::Trace, Rotation) {
        let (trace, compiled, plan) = rig();
        let schedule = Schedule::of(&trace);
        let rotation = Rotation::plan(&schedule, &compiled, &candidates(&plan), SLOT_CAP, ARENA_CAP)
            .expect("the spilled set rotates");
        (trace, rotation)
    }

    // A runtimeless build opens no pump.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn a_runtimeless_build_opens_no_rotor() {
        let (_, rotation) = rotation();
        let source = vec![core::ptr::null(); rotation.tenants().len()];
        assert!(matches!(
            Rotor::open(rotation, source),
            Err(Fault::Runtimeless)
        ));
    }
}
