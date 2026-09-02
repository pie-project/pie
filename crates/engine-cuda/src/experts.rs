//! The routed-expert tier: T0 is a budgeted device slab, T1 a budgeted
//! pinned host copy of every expert, T2 the mapped warm-boot artifact for
//! whatever neither budget holds. A device-resident table maps
//! `expert_id -> base address`.
//!
//! Routing runs on device, so residency is a performance promotion, not a
//! correctness condition. A quantized bank moves as one group via a
//! 16-byte `(codes, scales)` cell, so a torn pair is never observable.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

use model_ir::{Def, Linear, Operation, ParamSource, Trace, ValueId};

use crate::device::graph::Event;
use crate::device::{Buffer, Pinned, copy_any};
use crate::error::{Fault, Result};

/// One table entry: a device address, eight bytes.
const ENTRY: u64 = 8;

/// One usage counter: a `u32`, four bytes — the select kernel's `atomicAdd` width.
const COUNTER: u64 = 4;

/// How many experts of one bank may change residency between two fires, bounding copying per inter-fire gap on the notify stream.
const MOVES: usize = 2;

/// How many packed groups may change tier between two fires. One (a whole group, e.g. ~265 MiB) is far larger than a dense bank's expert.
const GROUP_MOVES: usize = 1;

/// One base cell: up to three device addresses and a pad, 32 bytes, 16-byte aligned, read by the packed select as one aggregate.
const CELL: u64 = 32;

/// How many planes a cell can name: two for mxfp4, three for an affine bank; a fourth is refused rather than seated with a plane unaddressed.
const CELL_PLANES: usize = 3;

/// A quantized bank's other device planes, keyed by the code plane's `Trace::params` index. Pairing comes from the load plan, not from name matching.
pub type Attachments = BTreeMap<usize, Vec<usize>>;

/// The operator's two ceilings. `None` is uncapped on that axis; T2 has no budget, only a file whose existence is decided at [`Tier::open`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Budgets {
    /// T0, the device store.
    pub device: Option<u64>,
    /// T1, the pinned host tier.
    pub host: Option<u64>,
}

impl Budgets {
    /// Both uncapped — the degenerate plan.
    #[must_use]
    pub const fn uncapped() -> Budgets {
        Budgets {
            device: None,
            host: None,
        }
    }

    /// A device ceiling with an uncapped host tier.
    #[must_use]
    pub const fn device(bytes: u64) -> Budgets {
        Budgets {
            device: Some(bytes),
            host: None,
        }
    }
}

/// Where a spilled group's bytes are read from: the mapped serving artifact. Both arms below hand out windows on the mapping — nothing is copied here.
#[derive(Debug)]
pub enum Spill {
    /// The `.zt` serving artifact, keyed by the trace's param order.
    Serving(crate::checkpoint_serving::Serving),
}

impl Spill {
    /// Which file these planes come out of.
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        let Spill::Serving(serving) = self;
        serving.path()
    }

    /// One plane's bytes, or `None` if this file doesn't carry it (`id` is the plan's ordinal).
    #[must_use]
    pub fn plane(&self, id: u32) -> Option<&[u8]> {
        let Spill::Serving(serving) = self;
        serving.plane(id)
    }

    /// Message for a plane the artifact cannot answer for.
    fn remedy(&self) -> &'static str {
        "The artifact is the model's own `.zt`, which IS the serving file: \
         its objects are this trace's plane names, so a hole in it is a name this build \
         declares and that file does not hold. The stamp was checked at open, so this \
         is not a foreign recipe — it is a trace that has gained a plane since the \
         artifact was written. Run `pie model import --force` on the source it names."
    }
}

/// Which tier holds a streamed group's planes — held whole on exactly one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Held {
    /// T0: the device store. Never a [`Plan::groups`] entry — see [`Plan::seated`].
    Device,
    /// T1: page-locked host memory, read over UVA.
    Pinned,
    /// T2: the mapped warm artifact — bytes already on disk, never copied.
    Mapped,
}

impl Held {
    /// How far down the ladder this tier is: 0 device, 1 pinned, 2 mapped. A cost order — a move is legal only from a higher rung to a lower one.
    #[must_use]
    pub const fn rung(self) -> u8 {
        match self {
            Held::Device => 0,
            Held::Pinned => 1,
            Held::Mapped => 2,
        }
    }
}

// ── the T2 register (counted, never load-bearing) ──────────────────────────

/// What the mapped tier has been asked for, process-wide. A read-only exception, not a control input — nothing in this file branches on it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Observed {
    /// Planes seated on T2, over every load this process has opened.
    pub seated: u64,
    /// Bytes those planes hold — read off disk rather than memory.
    pub bytes: u64,
    /// Planes the artifact could not answer for (absent, or wrong length).
    pub absent: u64,
    /// Loads that planned a spill and opened a source for it.
    pub loads: u64,
    /// Loads that took a deferred seat, served from the artifact while the page-locked copy built.
    pub deferred: u64,
    /// How many deferred windows closed (the page-locked copy landed).
    pub promoted: u64,
    /// Length of the last deferred window, in milliseconds; stored, not accumulated.
    pub window_ms: u64,
}

#[derive(Clone, Copy)]
enum Stat {
    Seated = 0,
    Bytes = 1,
    Absent = 2,
    Loads = 3,
    Deferred = 4,
    Promoted = 5,
    /// Stored, not added to — see [`Observed::window_ms`].
    WindowMs = 6,
}

static T2: [AtomicU64; 7] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
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

/// What the mapped tier has done, process-wide. See [`Observed`].
#[must_use]
pub fn observed() -> Observed {
    let at = |stat: Stat| T2[stat as usize].load(Ordering::Relaxed);
    Observed {
        seated: at(Stat::Seated),
        bytes: at(Stat::Bytes),
        absent: at(Stat::Absent),
        loads: at(Stat::Loads),
        deferred: at(Stat::Deferred),
        promoted: at(Stat::Promoted),
        window_ms: at(Stat::WindowMs),
    }
}

/// A load took a deferred seat. Counted where the seat is armed.
pub fn count_deferred() {
    bump(Stat::Deferred);
}

/// A deferred window closed, and how long it was.
pub fn count_promoted(window_ms: u64) {
    bump(Stat::Promoted);
    T2[Stat::WindowMs as usize].store(window_ms, Ordering::Relaxed);
}

/// Can this device dereference an ordinary host mapping (HMM, `cudaDevAttrPageableMemoryAccess`)? This is the mechanism T2 stands on. Read from the current device; `false` without a runtime.
#[must_use]
pub fn pageable_access() -> bool {
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        let mut ordinal = 0i32;
        // SAFETY: `ordinal` is a live out-parameter.
        if unsafe { rt::cudaGetDevice(&raw mut ordinal) } != rt::cudaError::cudaSuccess {
            return false;
        }
        let mut value = 0i32;
        // SAFETY: `value` is a live out-parameter and `ordinal` came from the runtime one line above.
        let status = unsafe {
            rt::cudaDeviceGetAttribute(
                &raw mut value,
                rt::cudaDeviceAttr::cudaDevAttrPageableMemoryAccess,
                ordinal,
            )
        };
        status == rt::cudaError::cudaSuccess && value != 0
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// What a plan wants and what a budget allows, decided off the trace and the load plan's pairings, before the device is bound. The empty plan is full residency, produced by an uncapped `Residency`.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    banks: Vec<BankPlan>,
    /// Split-plane groups the device store does not hold — seated together in the pinned tier and read there over UVA.
    groups: Vec<GroupPlan>,
    /// Routed packed groups the device store does hold — kept separate from `groups`, which readers take to mean "not in the store".
    seated: Vec<GroupPlan>,
    /// `param index -> how many experts of it live on the device`.
    resident_of: BTreeMap<usize, u32>,
    /// Every param of every group held on T1, flattened.
    pinned_of: BTreeSet<usize>,
    /// Every param of every group held on T2; disjoint from `pinned_of`.
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
    pub name: String,
    /// The bank's leading axis.
    pub experts: u32,
    /// How many of them the device slab seats.
    pub resident: u32,
    /// One expert's bytes — uniform stride across the bank.
    pub stride: u64,
}

/// One plane of a packed group. `bytes` is what the checkpoint publishes; `reserved` rounds that up to the handle alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupPlane {
    /// Index into `Trace::params`.
    pub param: usize,
    /// The plane's own bytes, as published by the checkpoint.
    pub bytes: u64,
    /// `bytes` rounded up to [`weights::ALIGN`](crate::weights::ALIGN).
    pub reserved: u64,
}

#[derive(Debug, Clone)]
pub struct GroupPlan {
    /// Index into `Trace::params` of the code plane — the param a routed matmul's `bank` port names.
    pub param: usize,
    pub name: String,
    /// Every plane of the group, code plane first, ascending by param index.
    pub planes: Vec<GroupPlane>,
    /// The bank's leading axis, reported, never divided by; zero for a spilled dense plane.
    pub experts: u32,
    /// Routed expert bank vs. dense plane the budget gave up.
    pub routed: bool,
    /// Every plane's bytes, summed.
    pub bytes: u64,
    /// Which tier holds it, decided device-then-pinned-then-mapping.
    pub held: Held,
}

/// The priority ranking, before any budget cuts it: which planes a budget may move, and in what order. A pure function of the trace and the load plan's pairings, so one serving artifact is budget-polymorphic.
#[derive(Debug, Clone)]
pub struct Ranking {
    /// What a fully resident load of this trace demands of the device store.
    full: u64,
    /// Params no budget can move, ascending: adapter banks written by `register_adapter`. Omitted from the artifact.
    floor: Vec<usize>,
    /// What the floor reserves in the store, together.
    floor_bytes: u64,
    /// The cut sequence, hottest first: every dense plane a budget may spill, then every packed routed group, in ascending param order.
    sequence: Vec<GroupPlan>,
    /// The dense routed banks, ascending by param. Not in `sequence`: a bank has two readers at every budget, so it sits on neither side.
    banks: Vec<BankPlan>,
}

impl Ranking {
    /// The ranking this trace declares, with the uniformity proof checked.
    /// # Errors: [`Fault::Param`] for a bad dtype, a broken bank shape, or a packed bank with no paired scales plane.
    pub fn of(trace: &Trace, planes: &Attachments) -> Result<Ranking> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let full = bytes.iter().map(|b| b.next_multiple_of(crate::weights::ALIGN)).sum();
        // A dense plane the budget cannot seat becomes a group of one and takes the same three tiers.
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
        // The give-up order is the compiler's: `prefetch::Schedule` is a pure function of the trace, so any two boots rank planes the same.
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
        // Dense planes rank before packed banks: read by every fire, vs. maybe never for a routed expert.
        Ok(Ranking {
            full,
            floor,
            floor_bytes,
            sequence: spillable.into_iter().chain(packed).collect(),
            banks,
        })
    }

    /// What a fully resident load of this trace demands of the device store.
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

    /// Where every image of the serving artifact goes: one `(param, offset, bytes, reserved)` quad per image, in payload order. Registered planes are absent; every image starts aligned.
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
    /// The residency plan for `trace` under `budgets`: [`Ranking::of`] then [`Plan::cut`]; `None` (uncapped) returns the empty plan.
    /// # Errors: [`Ranking::of`]'s, plus [`Fault::Residency`] for a budget under the registered planes plus one slot of every dense bank.
    pub fn of(trace: &Trace, planes: &Attachments, budgets: Budgets) -> Result<Plan> {
        Plan::cut(&Ranking::of(trace, planes)?, budgets)
    }

    /// The cut this ranking makes under `budgets`: fewer experts of a dense bank, fewer whole groups of a packed one, never below the ranking's floor; offers each group to the device budget, then host, then the mapping, continuing past one that doesn't fit.
    /// # Errors: [`Fault::Residency`] for a budget under the registered planes plus one expert slot of every dense bank.
    pub fn cut(ranking: &Ranking, budgets: Budgets) -> Result<Plan> {
        let full = ranking.full;
        let Some(budget) = budgets.device else {
            return Ok(Plan {
                device_bytes: full,
                ..Plan::default()
            });
        };
        if budget >= full {
            // Budget covers the whole table: answer as the degenerate case (mirrors `place_all()`) rather than a streamed load.
            return Ok(Plan {
                device_bytes: full,
                ..Plan::default()
            });
        }

        let found = ranking.banks.clone();
        let experts = found.first().map_or(0, |bank| bank.experts);
        let dense = ranking.floor_bytes;
        // One slot of every dense bank; packed groups contribute zero (may live entirely on the pinned tier).
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
                 gave them up. Raise it to at least \
                 {floor}, or state `None`.",
                found.len(),
            )));
        }

        // Each group seats whole or not: device budget, then host, then mapping. Dense banks' pinned copy is reserved first, at worst case.
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
                // A routed group the store already holds is still a seat, not a placement — nothing below reserves for it.
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

        // Experts every dense bank seats, out of what the groups left behind. Monotone in `n` (at most a few hundred), so walked down from full rather than searched.
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
        // A dense bank the budget holds whole is not a streamed bank — recording it as one would pin a second copy and publish a table naming the slab.
        let banks: Vec<BankPlan> = match resident < experts {
            true => found
                .into_iter()
                .map(|bank| BankPlan { resident, ..bank })
                .collect(),
            false => Vec::new(),
        };
        let resident_of = banks.iter().map(|bank| (bank.param, resident)).collect();
        // The host tier holds every expert, not only missing ones: pinned is the authoritative copy; every span aligns like `host_walk`'s.
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

    /// Does this load stream anything — a dense bank's experts, or a packed bank's whole group?
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

    /// The routed packed banks the device store holds, in param order — [`Held::Device`], disjoint from [`Plan::groups`].
    #[must_use]
    pub fn seated(&self) -> &[GroupPlan] {
        &self.seated
    }

    /// Does the pinned tier (T1) hold this param whole? False for a streamed dense bank (its slab is in the store) and for a group held on T2.
    #[must_use]
    pub fn pinned(&self, param: usize) -> bool {
        self.pinned_of.contains(&param)
    }

    /// Does the mapped artifact (T2) hold this param whole?
    #[must_use]
    pub fn mapped(&self, param: usize) -> bool {
        self.mapped_of.contains(&param)
    }

    /// Does anything other than the device store hold this param whole? `pinned || mapped`, stated once so the two callers cannot drift.
    #[must_use]
    pub fn streamed_whole(&self, param: usize) -> bool {
        self.pinned(param) || self.mapped(param)
    }

    /// What this plan demands of tier T2, in bytes: packed groups neither budget holds. Zero unless the load spills.
    #[must_use]
    pub fn spill_demand(&self) -> u64 {
        self.spill_bytes
    }

    /// How many experts of `param` the device slab seats, or `None` for a param held whole — every param of a full-residency load.
    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        self.resident_of.get(&param).copied()
    }

    /// What this plan demands of tier T0 (device), in bytes.
    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.device_bytes
    }

    /// What this plan demands of tier T1 (pinned), in bytes. Zero for a fully-resident load.
    #[must_use]
    pub fn host_demand(&self) -> u64 {
        self.host_bytes
    }

    /// Where every byte of the pinned tier (T1) goes: one `(param, host_at, bytes, reserved)` quad per span — every dense bank, then every plane of every group held [`Held::Pinned`].
    #[must_use]
    pub fn host_layout(&self) -> Vec<(u64, u64, u64, u64)> {
        self.host_walk().0
    }

    /// How many bytes the pinned tier allocates — where [`Plan::host_layout`]'s walk ends, rounded up to [`weights::ALIGN`](crate::weights::ALIGN).
    #[must_use]
    pub fn host_image(&self) -> u64 {
        self.host_walk().1
    }

    /// The spans and where they end; private since a caller wants one or the other.
    fn host_walk(&self) -> (Vec<(u64, u64, u64, u64)>, u64) {
        let mut out = Vec::with_capacity(self.banks.len() + self.groups.len());
        let mut at = 0u64;
        for bank in &self.banks {
            // Whole bank, not just missing experts: pinned is the authoritative copy (see `Plan::cut`).
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

    /// Where every byte of the mapped section (T2) goes: one `(param, offset, bytes, reserved)` quad per plane, relative to the section's own base — aligned to [`weights::ALIGN`](crate::weights::ALIGN).
    #[must_use]
    pub fn mapped_layout(&self) -> Vec<(u64, u64, u64, u64)> {
        self.mapped_walk().0
    }

    /// How many bytes this cut leaves on the mapping — where [`Plan::mapped_layout`]'s walk ends. Larger than [`Plan::spill_demand`] by the padding between planes.
    #[must_use]
    pub fn mapped_image(&self) -> u64 {
        self.mapped_walk().1
    }

    /// The walk itself, for [`Plan::host_walk`]'s reason.
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

/// The routed banks a trace declares, with the uniformity proof checked — dense ones seated expert by expert, packed ones seated whole. The OP decides dense vs. packed, never a dtype or name suffix.
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
        // `MoeMatmulSelect` reads one dense handle; the two quantized twins read a split-plane bank.
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
        // Both dense and split-plane banks state an expert count in shape[0].
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
        // A packed bank without a paired scales plane cannot be seated: the codes and exponents must be indexed together or a bank would read another bank's factors.
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
            // Resting value: `Plan::of`'s walk decides residency and overwrites this.
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

/// One streamed bank, seated: where its bytes are on both tiers, and which expert is in which device slot right now.
#[derive(Debug)]
struct Seat {
    param: usize,
    name: String,
    experts: u32,
    resident: u32,
    stride: u64,
    /// Byte offset of expert 0 inside the page-locked image.
    host_at: u64,
    /// The address expert 0 is read from right now — pinned memory for an eager seat, or the artifact's while deferred.
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

/// One plane of a packed group, seated on the pinned tier: where its bytes are, and nothing else — no slot map, since packed select indexes by expert id.
#[derive(Debug)]
struct Whole {
    param: usize,
    /// Byte offset of this plane inside the page-locked image.
    host_at: u64,
    /// The address this plane is read from right now — see [`Seat::serving_at`].
    serving_at: u64,
}

/// One plane of a packed group, seated on T2 — an address inside the mapped artifact, dereferenced directly by the GPU via HMM.
#[derive(Debug)]
struct Mapped {
    param: usize,
    /// The plane's address inside the mapping — absolute, since a mapping has no base to rebase against.
    at: u64,
    /// What the artifact says the plane holds.
    bytes: u64,
}

/// One packed group, and which rung of the ladder it is on right now — the live answer, since a group's tier can move after the plan is built.
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
    /// Where each plane is in the artifact — the rung a demotion falls back to. Empty when unbacked; an unbacked group is never displaced.
    backing: Vec<u64>,
    /// Which berth it occupies, or `None` for a group read where it lies.
    berth: Option<usize>,
    /// When it last changed rung, on [`Tier::tick`]'s monotone clock; `0` for a group still where the plan seated it. Tiebreak for `promote_now`.
    settled: u64,
}

/// One physical region a group's planes can be copied into — a device store region or a pinned-tier span. T2 is never a berth: a group demoted there just points back at its own file bytes.
#[derive(Debug)]
struct Berth {
    /// Which rung. [`Held::Device`] or [`Held::Pinned`], never [`Held::Mapped`].
    tier: Held,
    /// One address per plane, in plane order.
    at: Vec<u64>,
    /// The reserved bytes of each plane, plane for plane — the shape a group must match to take this berth.
    shape: Vec<u64>,
    /// Which group is in it, by index into `Tier::groups`.
    holds: Option<usize>,
}

/// The one swap in flight — a promotion between its two halves. Split across two gaps because the bulk copy cannot finish in one: `ready` records after the first, so a fire only ever waits on cell writes.
#[derive(Debug, Clone, Copy)]
struct Swap {
    berth: usize,
    group: usize,
}

/// Which half of a [`Swap`] this gap takes.
#[derive(Debug, Clone, Copy)]
enum Step {
    /// The berth's occupant goes back to the file; the bulk copy follows on the notify stream.
    Open(Swap),
    /// The bytes have landed; the candidate's cell arrives in the berth.
    Close(Swap),
}

/// The two device addresses one packed group hands its select kernel. The shell's spelling of `kernels_cuda::linear::moe::GroupSeat`, kept here so `run.rs` carries a weight row without naming a kernel type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GroupHandles {
    /// The group's 16-byte `(codes, scales)` base cell.
    pub cell: u64,
    /// The group's `u32` usage counter.
    pub hits: u64,
}

/// The two device addresses one bank hands its select kernel. The shell's spelling of `kernels_cuda::linear::moe::ExpertTable`, kept here so `run.rs` doesn't have to name a kernel type to carry a weight row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handles {
    /// The bank's `expert_id -> base address` table.
    pub table: u64,
    /// The bank's per-expert usage counters.
    pub counts: u64,
}

/// What one bank's residency looks like from outside — the accessor a gate reads, and the only observable a promotion has.
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
    /// Every expert's cumulative usage count, as the last settled readback carried it out.
    pub hits: Vec<u32>,
    /// Which tier holds it, for a packed group seated whole — `None` for a dense bank, whose answer is `in_slot` instead.
    pub held: Option<Held>,
}

/// Where the pinned tier's bytes come from, decided by the caller of [`Tier::open`].
#[derive(Debug)]
pub enum Fill {
    /// The cold load. [`Pinned::mapped`]: zeroed here, then written span by span by the landing sink; zeroing covers each span's tail padding, which the sink does not write.
    Cold,
    /// The warm restore. [`Pinned::mapped_uninit`]: not zeroed, because the artifact tiles the whole allocation before any byte is read out. A caller that cannot promise that owes [`Tier::zero_host`] first.
    Restored,
    /// The deferred seat: no page-locked image yet. Serves each T1 image out of the artifact while a background thread builds the page-locked copy. Needs `pageableMemoryAccess` — see [`pageable_access`].
    Deferred(crate::checkpoint_serving::Serving),
}

/// The background page-lock-and-fill, while it is outstanding. One thread and one channel: it binds the device, re-verifies and page-locks every T1 image, and sends the result — or nothing, if it failed.
#[derive(Debug)]
struct Refill {
    /// The thread, until [`Refill`] is dropped or the image is taken.
    filling: Option<std::thread::JoinHandle<()>>,
    /// The one message it will ever send.
    filled: std::sync::mpsc::Receiver<Pinned>,
    /// When the window opened. See [`Observed::window_ms`].
    began: std::time::Instant,
}

/// What a [`Refill`] answers when it is asked between two fires.
enum Filled {
    /// Still page-locking, reading or hashing. Serving continues on T2.
    Waiting,
    /// The image is here, verified and page-locked. Install it.
    Ready(Pinned),
    /// The thread finished and sent nothing — the only way it says it failed; serving continues on what it verified.
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

    /// Wait for it — the gate's half of [`Refill::poll`]; joins the thread.
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
    /// The thread is joined, not detached — a detached refill would outlive the tier and page-lock tens of gigabytes with no handle to free them.
    fn drop(&mut self) {
        if let Some(filling) = self.filling.take() {
            let _ = filling.join();
        }
    }
}

/// **The tier**: pinned bytes, a device slab, a table between them, and the counters that decide what moves.
#[derive(Debug)]
pub struct Tier {
    plan: Plan,
    /// T1: every expert of every streamed bank. Zero-length while a deferred seat is open — see [`Fill::Deferred`].
    host: Pinned,
    /// The background fill, while the seat is deferred and the window is open.
    refill: Option<Refill>,
    /// The mapping a deferred seat serves T1 out of, held for the tier's whole life. Not dropped at install: an in-flight kernel may still read a mapped plane, and its pages are reclaimable file pages.
    image: Option<crate::checkpoint_serving::Serving>,
    /// The device-resident indirection tables, one row of `experts` entries per bank, concatenated.
    table: Buffer,
    /// A pinned mirror of `table`: writes land here first, then the notify stream copies across, so an in-flight copy never reads a temporary.
    shadow: Pinned,
    /// The per-expert usage counters the select kernels `atomicAdd` into.
    counts: Buffer,
    /// A pinned mirror of `counts`, filled by an async D2H the settle side enqueues; read on the host between fires without a wait.
    mirror: Pinned,
    seats: Vec<Seat>,
    /// Every plane of every packed group the store does not hold, at its own offset inside [`Tier::host`].
    wholes: Vec<Whole>,
    /// The ladder's only mutable address surface: one 16-byte `(codes, scales)` cell per packed group.
    cells: Buffer,
    /// A pinned mirror of `cells`, filled before the copy is issued — makes a torn pair unconstructible, since both planes write as one word.
    cell_shadow: Pinned,
    /// One `u32` per packed group, `atomicAdd`ed by the select once per routed row per fire.
    group_counts: Buffer,
    /// A pinned mirror of `group_counts`, read the same way as [`Tier::mirror`]: without a wait.
    group_mirror: Pinned,
    /// Every routed packed group of this load — held and not — in plan order.
    groups: Vec<Group>,
    /// Every region a group can be copied into. See [`Berth`].
    berths: Vec<Berth>,
    /// The promotion between its two halves, or `None`. See [`Swap`].
    swap: Option<Swap>,
    /// Records on the notify stream when a swap's bulk copy is past — asked at the next gap, not waited on.
    landed: Event,
    /// `(groups promoted, groups demoted, gaps a swap in flight held back)`.
    ladder: (u64, u64, u64),
    /// Can anything be displaced at all? True when at least one group's planes resolved against an artifact; decided once, at [`Tier::open`].
    ladder_open: bool,
    /// A monotone clock over rung changes; see [`Group::settled`].
    tick: u64,
    /// T2: the mapped warm artifact, held open for the load's whole life because the weight rows point into it.
    source: Option<Spill>,
    /// Every plane of every group T2 holds, resolved against `source` at open time — the addresses stay valid because the mapping does.
    mapped: Vec<Mapped>,
    /// Records on the compute stream: everything already enqueued there is past. What an eviction waits for before it overwrites a slot.
    drained: Event,
    /// Records on the notify stream at the end of a promotion round — what the next fire's enqueue waits for.
    ready: Event,
    /// Has `ready` ever been recorded?
    moving: bool,
    promotions: u64,
    demotions: u64,
    skipped: u64,
}

impl Tier {
    /// Opens the tier `plan` describes: pinned bytes, tables, counters, and the mapped artifact behind any spilled group; must be called before the checkpoint lands. `source` must be `Some` when [`Plan::spill_demand`] is non-zero.
    /// # Errors: [`Fault::OutOfMemory`]/[`Fault::Device`] for allocations, [`Fault::Runtimeless`] without a runtime, [`Fault::Residency`] for a spilled plan with no source or missing plane.
    pub fn open(plan: Plan, source: Option<Spill>, fill: Fill) -> Result<Tier> {
        // `Plan::host_layout` walks every dense bank's span, then every T1 plane's.
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
                // Filled below, once `fill` is resolved.
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
        // Each pinned plane gets the store's own alignment, so a kernel reading codes as 32-bit words sees an aligned address on either tier.
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
        // T2: groups neither budget held, resolved against the mapping; nothing is copied here.
        let spilled = plan.groups().iter().any(|group| group.held == Held::Mapped);
        if spilled {
            // Without `pageableMemoryAccess` the device cannot dereference an unregistered host pointer; registering would page-lock it anyway.
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
                // One group per param on this plane: a split-plane bank is already two `Trace::params` rows.
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
        // The ladder's roster: every routed packed group, seated or not, in param order, so two boots number cells the same.
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
                // `land` fills this at the first moment the store's own addresses exist.
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
        // T1 is the one pinned span measured in gigabytes and the only one a boot can decline to allocate up front — see [`Fill`].
        let want = usize::try_from(host_at).unwrap_or(usize::MAX);
        let (host, image) = match fill {
            Fill::Cold => (Pinned::mapped(want)?, None),
            Fill::Restored => (Pinned::mapped_uninit(want)?, None),
            // Nothing page-locked, nothing copied: the tier serves each T1 image out of the artifact where it lies.
            Fill::Deferred(artifact) => (Pinned::mapped(0)?, Some(artifact)),
        };
        // The images this budget puts on T1 are a subset of the artifact's ranking order, not a contiguous run, so each span gets its own address.
        let mut serving = Vec::with_capacity(layout.len());
        for (param, host_at, _, reserved) in layout.iter().copied() {
            serving.push(match &image {
                None => host.device().saturating_add(host_at),
                Some(artifact) => {
                    let id = u32::try_from(param).unwrap_or(u32::MAX);
                    // The reserved extent, not the published one — a seat hands out a pointer treated as `reserved` bytes wide.
                    let Some(bytes) = artifact.plane_padded(id, reserved) else {
                        let plane = artifact
                            .name(id)
                            .map_or_else(|| format!("param {param}"), |name| format!("`{name}`"));
                        return Err(Fault::Residency(format!(
                            "the serving artifact at {} does not carry {plane} out to \
                             {reserved} bytes, and a deferred seat reads that plane's T1 \
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

    /// Where a streamed bank's plane goes as it lands: its byte offset inside the pinned tier, or `None` for a param the device store holds.
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

    /// The device-visible address of a plane this tier holds whole, or `None` if the device store holds it — see [`Seat::serving_at`].
    #[must_use]
    pub fn pinned_at(&self, param: usize) -> Option<u64> {
        self.wholes
            .iter()
            .find(|whole| whole.param == param)
            .map(|whole| whole.serving_at)
    }

    /// The address of a plane this tier serves out of the mapped artifact (T2), or `None` if it does not — dereferenced directly by the GPU via HMM.
    #[must_use]
    pub fn mapped_at(&self, param: usize) -> Option<u64> {
        self.mapped
            .iter()
            .find(|plane| plane.param == param)
            .map(|plane| plane.at)
    }

    /// Where a plane of this load actually is: the pinned tier's address for a T1 group, the mapping's for T2. `None` means the device store holds it.
    #[must_use]
    pub fn offloaded_at(&self, param: usize) -> Option<u64> {
        self.pinned_at(param).or_else(|| self.mapped_at(param))
    }

    /// The T2 footprint: bytes this tier serves out of the mapping, summed from what the artifact actually answered for.
    #[must_use]
    pub fn spilled_bytes(&self) -> u64 {
        self.mapped.iter().map(|plane| plane.bytes).sum()
    }

    /// The artifact this tier serves T2 out of, or `None` for a load that spilled nothing.
    #[must_use]
    pub fn source(&self) -> Option<&Spill> {
        self.source.as_ref()
    }

    /// One T2 plane's bytes, borrowed from the mapping — `None` if this tier does not serve `param` from T2.
    #[must_use]
    pub fn mapped_plane(&self, param: usize) -> Option<&[u8]> {
        let plane = self.mapped.iter().find(|plane| plane.param == param)?;
        let at = usize::try_from(plane.at).ok()? as *const u8;
        let len = usize::try_from(plane.bytes).ok()?;
        // SAFETY: the pair came from `Spill::plane`, a window on a mapping this tier owns in `source` for its whole life, unchanged since.
        Some(unsafe { std::slice::from_raw_parts(at, len) })
    }

    /// The pinned tier's host address, for the landing sink to store through. Zero-length while [`Tier::deferring`].
    #[must_use]
    pub fn host(&self) -> &Pinned {
        &self.host
    }

    /// Is this tier still serving T1 out of the artifact's mapping? True only between [`Fill::Deferred`] and the install.
    #[must_use]
    pub fn deferring(&self) -> bool {
        self.image.is_some() && self.host.bytes() == 0
    }

    /// The artifact a deferred seat is seated over, or `None` for a tier that page-locked its own image.
    #[must_use]
    pub fn deferred_image(&self) -> Option<&crate::checkpoint_serving::Serving> {
        self.image.as_ref().filter(|_| self.host.bytes() == 0)
    }

    /// The host-side address of one T1 plane right now, for the CPU-side reader rather than a kernel. `None` for a param not held on T1.
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

    /// Arms the background fill: the deferred seat's other half, called once by the restore after verifying this seat's T1 images.
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

    /// Recovery for a deferred boot whose T1 images failed to verify: makes (and zeroes) the allocation the deferred arm skipped. Must be called before [`Tier::land`]; a no-op if the tier is not deferring.
    /// # Errors: [`Fault::OutOfMemory`] or [`Fault::Device`] for the allocation.
    pub fn undefer(&mut self) -> Result<()> {
        if !self.deferring() {
            return Ok(());
        }
        // The refill is dropped — and therefore joined — before the allocation, so the two page-locks are never outstanding at once.
        self.refill = None;
        let want = usize::try_from(self.plan.host_image()).unwrap_or(usize::MAX);
        self.host = Pinned::mapped(want)?;
        self.reseat();
        self.image = None;
        Ok(())
    }

    /// Closes the deferred-fill window now by joining the fill thread. Returns whether the image was installed; `false` if never deferred, already closed, or the fill failed (not an error).
    /// # Errors: [`Fault::Device`] for an event or a copy the runtime refused.
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

    /// Installs the page-locked image as one base address, inside the same drained/reseat/publish/ready bracket a promotion runs in.
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
        // Logged once per load, at the instant the tier stops deferring.
        eprintln!(
            "engine-cuda: the deferred tier is INSTALLED after {window_ms} ms — every T1 \
             read is a page-locked read from here"
        );
        // A group the ladder moved during the window is on another rung now and is skipped: it already has its own berth.
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
        count_promoted(window_ms);
        Ok(())
    }

    /// Re-forms every T1 span's `serving_at` from `host_at` against the page-locked allocation.
    fn reseat(&mut self) {
        let base = self.host.device();
        for seat in &mut self.seats {
            seat.serving_at = base.saturating_add(seat.host_at);
        }
        for whole in &mut self.wholes {
            whole.serving_at = base.saturating_add(whole.host_at);
        }
    }

    /// Zeros the pinned image: recovery for a warm boot whose uninitialized allocation failed to fill, before the cold load writes into it.
    pub fn zero_host(&self) {
        self.host.zero();
    }

    /// The populated spans of the pinned image: `(param, host_at, bytes, reserved)` for every dense bank and every T1 plane, tiling `0..Pinned::bytes` exactly. Just [`Plan::host_layout`] handed back.
    #[must_use]
    pub fn image(&self) -> Vec<(u64, u64, u64, u64)> {
        self.plan.host_layout()
    }

    /// Seat the slabs and publish the first table, after the checkpoint has landed, before the first fire. Slots `0..resident` take experts `0..resident`.
    /// # Errors: [`Fault::Device`] for the copies, [`Fault::Runtimeless`] without a runtime.
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
        // One copy per slot, not per bank: slot stride equals expert stride only while slots are still the first `resident` experts.
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

    /// Seat every packed group and open its berth, once the store's addresses exist.
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
            // A deferred seat opens no T1 berth: its addresses still point into the mapped file until the install lands.
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

    /// Write every group's cell from its live addresses, and copy the lot across.
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

    /// Fill one group's sixteen shadow bytes from its live addresses. Both planes write as one word, so a torn pair is unconstructible.
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

    /// Write every entry of every bank's table, from the seats' current residency.
    fn publish_all(&mut self, stream: *mut c_void) -> Result<()> {
        // A plan with only packed (routed) banks has no table at all: skip the copy rather than move zero bytes between null addresses.
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

    /// The two device addresses a packed group's select kernel reads, or `None` if `param` names no routed packed bank.
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

    /// The two device addresses `param`'s select kernel reads, or `None` for a param this tier does not hold.
    #[must_use]
    pub fn handles(&self, param: usize) -> Option<Handles> {
        self.seats.iter().find(|seat| seat.param == param).map(|seat| Handles {
            table: self.table.ptr() + seat.entry_at as u64 * ENTRY,
            counts: self.counts.ptr() + seat.counter_at as u64 * COUNTER,
        })
    }

    /// Carry the fire's usage counts out, asynchronously, on `stream`. Called from `settle`; nothing waits for it.
    /// # Errors: [`Fault::Device`] for the copy.
    pub fn drain(&self, stream: *mut c_void) -> Result<()> {
        if self.counts.bytes() > 0 {
            copy_any(
                stream,
                self.mirror.device(),
                self.counts.ptr(),
                self.counts.bytes(),
            )?;
        }
        // Group counters ride the same async D2H: one `u32` per packed group, atomicAdd'ed once per routed row per fire.
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

    /// The promotion, between two fires: moves at most [`MOVES`] experts of each bank onto the device and repoints the table. Returns how many moved. Never blocks — a skipped round is not waited for.
    /// # Errors: [`Fault::Device`] for an event or a copy the runtime refused.
    pub fn promote(&mut self, compute: *mut c_void, notify: *mut c_void) -> Result<u32> {
        // A tier with nothing to promote can still have a refill to install (a dense model's tier has pinned planes and no seats at all).
        if self.seats.is_empty() && self.groups.is_empty() && self.refill.is_none() {
            return Ok(0);
        }
        // The shadow is one allocation shared by every entry write, so a round may not rewrite a word an earlier round's copy may still be reading.
        if self.moving && !self.ready.done()? {
            self.skipped += 1;
            return Ok(0);
        }
        // A group's bulk copy on the notify stream holds the whole round back: `ready` rides the same stream, so a later round would queue behind it and block the next fire.
        if self.swap.is_some() && !self.landed.done()? {
            self.ladder.2 += 1;
            self.skipped += 1;
            return Ok(0);
        }
        // The deferred seat's install, if its image has arrived: takes the whole gap and happens once per load, behind both guards above.
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
                // Nothing is installed; the seat keeps serving what it verified at boot.
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
            // The evicted expert's entry is written back first: after this, a kernel routing to `out` reads pinned memory.
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
        // The ladder's sixteen-byte half: either the occupant's cell leaves the berth or the candidate's arrives, never both in one gap.
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
        // Everything a fire ever waits for is above this line: table entries, slot bytes and cells — all small, all already enqueued.
        self.ready.record(notify)?;
        self.ready.wait(compute)?;
        // The bulk copy is below it: enqueued after `ready`, so no fire is ordered against it. `landed` is checked at a later gap, not waited on here.
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

    /// Half one: the berth's occupant goes back to the file. The cell write is ordered before the bulk copy on the same stream.
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

    /// Half two: the candidate's cell arrives in the berth, once its bytes have landed. Copy first, then flip the pointer.
    fn close_berth(&mut self, swap: Swap, notify: *mut c_void) -> Result<()> {
        let berth = &self.berths[swap.berth];
        let (tier, at) = (berth.tier, berth.at.clone());
        // The berth the candidate is leaving, if any, is now free — how the ladder walks more than one rung, one gap at a time.
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

    /// What the ladder does this gap; see [`Swap`] for the two halves. `Close` runs first and unconditionally.
    fn step(&self) -> Option<Step> {
        if let Some(swap) = self.swap {
            return Some(Step::Close(swap));
        }
        // A load with no artifact keeps the assignment it booted with: nothing is displaceable, so the vote is skipped every gap.
        if GROUP_MOVES == 0 || self.berths.is_empty() || !self.ladder_open {
            return None;
        }
        let hits = self.group_mirror.read(0, self.group_mirror.bytes());
        self.decide_group(&hits).map(Step::Open)
    }

    /// Which group should change rungs, given the counters.
    ///
    /// ```text
    /// swap G into berth B, displacing H, iff
    ///   rung(B) < rung(G) and shape(B) == shape(G) and hits(G) > hits(H)
    /// ```
    ///
    /// Strict improvement bounds the vote (lowers `Σ hits × rung` every
    /// move); ties break toward the biggest gain, then lowest berth/group.
    fn decide_group(&self, hits: &[u8]) -> Option<Swap> {
        let count = |at: usize| -> u32 {
            let byte = at * COUNTER as usize;
            hits.get(byte..byte + COUNTER as usize)
                .and_then(|word| word.try_into().ok())
                .map_or(0, u32::from_ne_bytes)
        };
        vote(&self.berths, &self.groups, count)
    }

    /// Which experts should change places: least-used resident out, most-used non-resident in, at most [`MOVES`] per bank.
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

    /// **What is resident right now, and what the fires asked for** — the accessor a gate reads.
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
        // A packed group reports zero slots (its planes live whole on one tier) and `held` is the live tier, not the plan's stale one, since a group's tier can move after the plan was built.
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
        // Spilled dense planes are reported too, off the plan: they have no cell and never move, but an operator still wants to see them.
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

    /// `(groups promoted, groups demoted, gaps a swap in flight held back)`, since load.
    #[must_use]
    pub fn ladder(&self) -> (u64, u64, u64) {
        self.ladder
    }

    /// Take one rung for `name` now, ignoring the strict-improvement clause — a gate's manual door onto the ladder, synchronous. Returns `(from, to)`, or `None` if no berth of the right shape is on a faster rung.
    /// # Errors: [`Fault::Device`] for an event or a copy the runtime refused.
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
        // An empty berth first, then the coldest occupant, then the one that has sat on its rung longest — see [`Group::settled`].
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

    /// `(experts promoted, experts demoted, gaps skipped because the previous round was still moving)`, since load.
    #[must_use]
    pub fn motion(&self) -> (u64, u64, u64) {
        (self.promotions, self.demotions, self.skipped)
    }

    /// The plan this tier serves.
    #[must_use]
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Every byte the tier holds off the device store: the pinned tier, the tables and their shadow, the counters and their mirror.
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

/// The vote, as a pure function of the seating and the counters — the body of [`Tier::decide_group`], lifted out to be exercised without a device.
fn vote(berths: &[Berth], groups: &[Group], count: impl Fn(usize) -> u32) -> Option<Swap> {
    let mut best: Option<(u32, Swap)> = None;
    for (at, berth) in berths.iter().enumerate() {
        // An occupant with no backing cannot be displaced — nowhere to point its cell at.
        let out = match berth.holds {
            Some(out) if groups[out].backing.is_empty() => continue,
            held => held,
        };
        let floor = out.map_or(0, |out| count(groups[out].cell_at));
        for (group, candidate) in groups.iter().enumerate() {
            if candidate.berth == Some(at) || candidate.held.rung() <= berth.tier.rung() {
                continue;
            }
            // Plane for plane, without allocating: this runs berths x groups times per inter-fire gap.
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

/// Where expert `expert` of `seat` lives on T1 — the address a kernel dereferences when it is not on the device.
fn pinned_address_of(seat: &Seat, expert: u32) -> u64 {
    seat.serving_at + u64::from(expert) * seat.stride
}

#[cfg(test)]
mod tests {
    use model_dsl::Platform;

    use super::*;

    fn a3b() -> Trace {
        let trace = models::sku("qwen35-a3b-bf16-kv-bf16").expect("the catalog ships the SKU").trace;
        trace(Platform::Cuda)
    }

    /// The catalog's one split-plane MoE: 32 experts, 24 layers, mxfp4 banks.
    fn gpt_oss() -> Trace {
        let trace =
            models::sku("gptoss-20b-bf16-mxfp4-kv-bf16").expect("the catalog ships the SKU").trace;
        trace(Platform::Cuda)
    }

    /// A load plan's pairing, built off the DSL's own name-minting so a plan test needs no checkpoint on disk.
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
    fn a_packed_bank_the_plan_pairs_no_scales_with_is_refused_by_name() {
        let trace = gpt_oss();
        let full = Plan::of(&trace, &scales_of(&trace), Budgets::uncapped())
            .expect("uncapped plans")
            .device_demand();
        // The same trace, with the loader's pairing withheld: the codes could be seated alone and factors left behind — refused, not approximated.
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
    fn a_budget_under_the_planes_that_cannot_move_is_refused_by_name() {
        // THE FLOOR under any budget is the planes that cannot leave the device: a REGISTERED adapter bank, plus one expert slot per bank.
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

    // ── THE LADDER ──────────────────────────────────────────────────────────

    // O1: the order the writer lays out is the order the boot walks.

}

