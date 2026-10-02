use core::cell::Cell;
use core::ffi::c_void;

use model_compiler::CompiledModel;
use model_compiler::prefetch::Schedule;

use crate::device::alloc::Buffer;
use crate::device::graph::Event;
use crate::error::{Fault, Result};

pub const DEPTH_MAX: u32 = 32;

pub const LOOKAHEAD: u32 = 2;

pub const SLOT_CAP: u64 = 32 << 20;

pub const ARENA_CAP: u64 = 256 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tenant {
    pub param: usize,
    pub slot: u32,
    pub bytes: u64,
    pub first: u32,
    pub last: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rotation {
    tenants: Vec<Tenant>,
    slots: u32,
    slot_bytes: Vec<u64>,
    declined: Vec<usize>,
    acquire: Vec<Vec<u32>>,
    release: Vec<Vec<u32>>,
    issue: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decline {
    Nothing,
    Residency { want: u32, planes: u32 },
    Overlap(String),
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
    #[must_use]
    pub fn plan(
        schedule: &Schedule,
        compiled: &CompiledModel,
        candidates: &[(usize, u64)],
        cap: u64,
        arena_cap: u64,
    ) -> core::result::Result<Rotation, Decline> {
        let regions = compiled.regions.len().max(1);
        let spans = schedule.against(compiled);
        let mut region_of: std::collections::BTreeMap<usize, core::ops::Range<u32>> =
            std::collections::BTreeMap::new();
        for (row, span) in schedule.reads().iter().zip(&spans) {
            if !row.unread() {
                region_of.insert(row.param, span.clone());
            }
        }

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
        kept.sort_by_key(|(param, _)| {
            schedule
                .read_of(*param)
                .map_or((u32::MAX, *param), |row| (row.span.start, row.param))
        });
        let planes = u32::try_from(kept.len()).unwrap_or(u32::MAX);

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

        let mut acquire = vec![Vec::new(); regions];
        let mut release = vec![Vec::new(); regions];
        let mut issue = vec![Vec::new(); regions];
        for (at, tenant) in tenants.iter().enumerate() {
            let at = u32::try_from(at).unwrap_or(u32::MAX);
            acquire[(tenant.first as usize).min(regions - 1)].push(at);
            let free_at = tenant.last as usize + 1;
            if free_at < regions {
                release[free_at].push(at);
            }
            issue[(tenant.first.saturating_sub(LOOKAHEAD) as usize).min(regions - 1)].push(at);
        }
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

    #[must_use]
    pub fn tenants(&self) -> &[Tenant] {
        &self.tenants
    }

    #[must_use]
    pub const fn slots(&self) -> u32 {
        self.slots
    }

    #[must_use]
    pub fn arena(&self) -> u64 {
        self.slot_bytes.iter().sum()
    }

    #[must_use]
    pub fn rotating(&self) -> u64 {
        self.tenants.iter().map(|tenant| tenant.bytes).sum()
    }

    #[must_use]
    pub fn declined(&self) -> &[usize] {
        &self.declined
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Observed {
    pub fires: u64,
    pub copies: u64,
    pub bytes: u64,
    pub late: u64,
}

pub struct Rotor {
    rotation: Rotation,
    slots: Vec<Buffer>,
    source: Vec<*const u8>,
    ready: Vec<Event>,
    free: Vec<Event>,
    copy: *mut c_void,
    occupant: Vec<Cell<Option<u32>>>,
    released: Vec<Cell<bool>>,
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
            slots.push(Buffer::zeroed(
                usize::try_from(*bytes).unwrap_or(usize::MAX),
            )?);
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

    #[must_use]
    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    #[must_use]
    pub fn seat(&self, param: usize) -> Option<u64> {
        let tenant = self
            .rotation
            .tenants
            .iter()
            .find(|tenant| tenant.param == param)?;
        Some(self.slots[tenant.slot as usize].ptr())
    }

    #[must_use]
    pub fn observed(&self) -> Observed {
        Observed {
            fires: self.fires.get(),
            copies: self.copies.get(),
            bytes: self.bytes.get(),
            late: self.late.get(),
        }
    }

    pub fn at(&self, region: u32, compute: *mut c_void) -> Result<()> {
        let at = region as usize;
        if region == 0 {
            self.begin(compute)?;
        }
        for tenant in self.rotation.release.get(at).into_iter().flatten() {
            let slot = self.rotation.tenants[*tenant as usize].slot as usize;
            self.free[slot].record(compute)?;
            self.released[slot].set(true);
        }
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
        for tenant in self.rotation.acquire.get(at).into_iter().flatten() {
            let which = *tenant;
            if self.next.get() <= which {
                self.late.set(self.late.get() + 1);
                while self.next.get() <= which {
                    let issuing = self.next.get();
                    if !self.issue(issuing)? {
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

    fn issue(&self, which: u32) -> Result<bool> {
        let tenant = self.rotation.tenants[which as usize];
        let slot = tenant.slot as usize;
        if self.occupant[slot].get().is_some() {
            if !self.released[slot].get() {
                return Ok(false);
            }
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

    fn rig() -> (model_ir::Trace, model_compiler::CompiledModel, Plan) {
        let trace = (models::sku(SKU).expect("the catalog ships the SKU").trace)(Platform::Cuda);
        let compiled = compile_axes(
            &trace,
            &Budgets {
                tokens: Budget::new(4, 256),
                patches: None,
                voxels: None,
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
        let rotation = Rotation::plan(
            &schedule,
            &compiled,
            &candidates(&plan),
            SLOT_CAP,
            ARENA_CAP,
        )
        .expect("the spilled set rotates");
        (trace, rotation)
    }

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
