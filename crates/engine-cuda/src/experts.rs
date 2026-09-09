use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

use model_ir::{Def, Linear, Operation, ParamSource, Trace, ValueId};

use crate::device::graph::Event;
use crate::device::{Buffer, Pinned, copy_any};
use crate::error::{Fault, Result};

const ENTRY: u64 = 8;

const COUNTER: u64 = 4;

const MOVES: usize = 2;

const GROUP_MOVES: usize = 1;

const CELL: u64 = 32;

const CELL_PLANES: usize = 3;

pub type Attachments = BTreeMap<usize, Vec<usize>>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Budgets {
    pub device: Option<u64>,
    pub host: Option<u64>,
}

impl Budgets {
    #[must_use]
    pub const fn uncapped() -> Budgets {
        Budgets {
            device: None,
            host: None,
        }
    }

    #[must_use]
    pub const fn device(bytes: u64) -> Budgets {
        Budgets {
            device: Some(bytes),
            host: None,
        }
    }
}

#[derive(Debug)]
pub enum Spill {
    Serving(crate::checkpoint_serving::Serving),
}

impl Spill {
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        let Spill::Serving(serving) = self;
        serving.path()
    }

    #[must_use]
    pub fn plane(&self, id: u32) -> Option<&[u8]> {
        let Spill::Serving(serving) = self;
        serving.plane(id)
    }

    fn remedy(&self) -> &'static str {
        "The artifact is the model's own `.zt`, which IS the serving file: \
         its objects are this trace's plane names, so a hole in it is a name this build \
         declares and that file does not hold. The stamp was checked at open, so this \
         is not a foreign recipe — it is a trace that has gained a plane since the \
         artifact was written. Run `pie model import --force` on the source it names."
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Held {
    Device,
    Pinned,
    Mapped,
}

impl Held {
    #[must_use]
    pub const fn rung(self) -> u8 {
        match self {
            Held::Device => 0,
            Held::Pinned => 1,
            Held::Mapped => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Observed {
    pub seated: u64,
    pub bytes: u64,
    pub absent: u64,
    pub loads: u64,
    pub deferred: u64,
    pub promoted: u64,
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

pub fn count_deferred() {
    bump(Stat::Deferred);
}

pub fn count_promoted(window_ms: u64) {
    bump(Stat::Promoted);
    T2[Stat::WindowMs as usize].store(window_ms, Ordering::Relaxed);
}

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

#[derive(Debug, Clone, Default)]
pub struct Plan {
    banks: Vec<BankPlan>,
    groups: Vec<GroupPlan>,
    seated: Vec<GroupPlan>,
    resident_of: BTreeMap<usize, u32>,
    pinned_of: BTreeSet<usize>,
    mapped_of: BTreeSet<usize>,
    device_bytes: u64,
    host_bytes: u64,
    spill_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct BankPlan {
    pub param: usize,
    pub name: String,
    pub experts: u32,
    pub resident: u32,
    pub stride: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupPlane {
    pub param: usize,
    pub bytes: u64,
    pub reserved: u64,
}

#[derive(Debug, Clone)]
pub struct GroupPlan {
    pub param: usize,
    pub name: String,
    pub planes: Vec<GroupPlane>,
    pub experts: u32,
    pub routed: bool,
    pub bytes: u64,
    pub held: Held,
}

#[derive(Debug, Clone)]
pub struct Ranking {
    full: u64,
    floor: Vec<usize>,
    floor_bytes: u64,
    sequence: Vec<GroupPlan>,
    banks: Vec<BankPlan>,
}

impl Ranking {
    pub fn of(trace: &Trace, planes: &Attachments) -> Result<Ranking> {
        let bytes = crate::weights::plane_bytes(trace)?;
        let full = bytes.iter().map(|b| b.next_multiple_of(crate::weights::ALIGN)).sum();
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
        Ok(Ranking {
            full,
            floor,
            floor_bytes,
            sequence: spillable.into_iter().chain(packed).collect(),
            banks,
        })
    }

    #[must_use]
    pub fn full(&self) -> u64 {
        self.full
    }

    #[must_use]
    pub fn floor(&self) -> &[usize] {
        &self.floor
    }

    #[must_use]
    pub fn sequence(&self) -> &[GroupPlan] {
        &self.sequence
    }

    #[must_use]
    pub fn banks(&self) -> &[BankPlan] {
        &self.banks
    }

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
    pub fn of(trace: &Trace, planes: &Attachments, budgets: Budgets) -> Result<Plan> {
        Plan::cut(&Ranking::of(trace, planes)?, budgets)
    }

    pub fn cut(ranking: &Ranking, budgets: Budgets) -> Result<Plan> {
        let full = ranking.full;
        let Some(budget) = budgets.device else {
            return Ok(Plan {
                device_bytes: full,
                ..Plan::default()
            });
        };
        if budget >= full {
            return Ok(Plan {
                device_bytes: full,
                ..Plan::default()
            });
        }

        let found = ranking.banks.clone();
        let experts = found.first().map_or(0, |bank| bank.experts);
        let dense = ranking.floor_bytes;
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
        let banks: Vec<BankPlan> = match resident < experts {
            true => found
                .into_iter()
                .map(|bank| BankPlan { resident, ..bank })
                .collect(),
            false => Vec::new(),
        };
        let resident_of = banks.iter().map(|bank| (bank.param, resident)).collect();
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

    #[must_use]
    pub fn streams(&self) -> bool {
        !self.banks.is_empty() || !self.groups.is_empty()
    }

    #[must_use]
    pub fn banks(&self) -> &[BankPlan] {
        &self.banks
    }

    #[must_use]
    pub fn groups(&self) -> &[GroupPlan] {
        &self.groups
    }

    #[must_use]
    pub fn seated(&self) -> &[GroupPlan] {
        &self.seated
    }

    #[must_use]
    pub fn pinned(&self, param: usize) -> bool {
        self.pinned_of.contains(&param)
    }

    #[must_use]
    pub fn mapped(&self, param: usize) -> bool {
        self.mapped_of.contains(&param)
    }

    #[must_use]
    pub fn streamed_whole(&self, param: usize) -> bool {
        self.pinned(param) || self.mapped(param)
    }

    #[must_use]
    pub fn spill_demand(&self) -> u64 {
        self.spill_bytes
    }

    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        self.resident_of.get(&param).copied()
    }

    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.device_bytes
    }

    #[must_use]
    pub fn host_demand(&self) -> u64 {
        self.host_bytes
    }

    #[must_use]
    pub fn host_layout(&self) -> Vec<(u64, u64, u64, u64)> {
        self.host_walk().0
    }

    #[must_use]
    pub fn host_image(&self) -> u64 {
        self.host_walk().1
    }

    fn host_walk(&self) -> (Vec<(u64, u64, u64, u64)>, u64) {
        let mut out = Vec::with_capacity(self.banks.len() + self.groups.len());
        let mut at = 0u64;
        for bank in &self.banks {
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

    #[must_use]
    pub fn mapped_layout(&self) -> Vec<(u64, u64, u64, u64)> {
        self.mapped_walk().0
    }

    #[must_use]
    pub fn mapped_image(&self) -> u64 {
        self.mapped_walk().1
    }

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
            held: Held::Pinned,
            routed: true,
        });
    }
    banks.sort_by_key(|bank| bank.param);
    groups.sort_by_key(|group| group.param);
    Ok((banks, groups))
}

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

#[derive(Debug)]
struct Seat {
    param: usize,
    name: String,
    experts: u32,
    resident: u32,
    stride: u64,
    host_at: u64,
    serving_at: u64,
    entry_at: usize,
    counter_at: usize,
    slab: u64,
    slot_of: Vec<Option<u32>>,
    in_slot: Vec<u32>,
}

#[derive(Debug)]
struct Whole {
    param: usize,
    host_at: u64,
    serving_at: u64,
}

#[derive(Debug)]
struct Mapped {
    param: usize,
    at: u64,
    bytes: u64,
}

#[derive(Debug)]
struct Group {
    param: usize,
    name: String,
    experts: u32,
    planes: Vec<GroupPlane>,
    cell_at: usize,
    at: Vec<u64>,
    held: Held,
    backing: Vec<u64>,
    berth: Option<usize>,
    settled: u64,
}

#[derive(Debug)]
struct Berth {
    tier: Held,
    at: Vec<u64>,
    shape: Vec<u64>,
    holds: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct Swap {
    berth: usize,
    group: usize,
}

#[derive(Debug, Clone, Copy)]
enum Step {
    Open(Swap),
    Close(Swap),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GroupHandles {
    pub cell: u64,
    pub hits: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handles {
    pub table: u64,
    pub counts: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankResidency {
    pub name: String,
    pub experts: u32,
    pub slots: u32,
    pub in_slot: Vec<u32>,
    pub hits: Vec<u32>,
    pub held: Option<Held>,
}

#[derive(Debug)]
pub enum Fill {
    Cold,
    Restored,
    Deferred(crate::checkpoint_serving::Serving),
}

#[derive(Debug)]
struct Refill {
    filling: Option<std::thread::JoinHandle<()>>,
    filled: std::sync::mpsc::Receiver<Pinned>,
    began: std::time::Instant,
}

enum Filled {
    Waiting,
    Ready(Pinned),
    Refused,
}

impl Refill {
    fn poll(&mut self) -> Filled {
        use std::sync::mpsc::TryRecvError;

        match self.filled.try_recv() {
            Ok(host) => Filled::Ready(host),
            Err(TryRecvError::Empty) => Filled::Waiting,
            Err(TryRecvError::Disconnected) => Filled::Refused,
        }
    }

    fn settle(mut self) -> Option<Pinned> {
        if let Some(filling) = self.filling.take() {
            let _ = filling.join();
        }
        self.filled.try_recv().ok()
    }

    fn window_ms(&self) -> u64 {
        u64::try_from(self.began.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

impl Drop for Refill {
    fn drop(&mut self) {
        if let Some(filling) = self.filling.take() {
            let _ = filling.join();
        }
    }
}

#[derive(Debug)]
pub struct Tier {
    plan: Plan,
    host: Pinned,
    refill: Option<Refill>,
    image: Option<crate::checkpoint_serving::Serving>,
    table: Buffer,
    shadow: Pinned,
    counts: Buffer,
    mirror: Pinned,
    seats: Vec<Seat>,
    wholes: Vec<Whole>,
    cells: Buffer,
    cell_shadow: Pinned,
    group_counts: Buffer,
    group_mirror: Pinned,
    groups: Vec<Group>,
    berths: Vec<Berth>,
    swap: Option<Swap>,
    landed: Event,
    ladder: (u64, u64, u64),
    ladder_open: bool,
    tick: u64,
    source: Option<Spill>,
    mapped: Vec<Mapped>,
    drained: Event,
    ready: Event,
    moving: bool,
    promotions: u64,
    demotions: u64,
    skipped: u64,
}

impl Tier {
    pub fn open(plan: Plan, source: Option<Spill>, fill: Fill) -> Result<Tier> {
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
        let spilled = plan.groups().iter().any(|group| group.held == Held::Mapped);
        if spilled {
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
            let bytes: u64 = mapped.iter().map(|plane| plane.bytes).sum();
            let groups = plan.groups().iter().filter(|group| group.held == Held::Mapped).count();
            eprintln!(
                "engine-cuda: the MAPPED tier holds {groups} group(s), {} plane(s), {bytes} \
                 byte(s) read where they lie in {} — neither budget held them; a GPU touch \
                 faults the page in over HMM",
                mapped.len(),
                source.as_ref().map_or_else(|| "<no artifact>".to_string(), |artifact| artifact.path().display().to_string()),
            );
        }
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
        let want = usize::try_from(host_at).unwrap_or(usize::MAX);
        let (host, image) = match fill {
            Fill::Cold => (Pinned::mapped(want)?, None),
            Fill::Restored => (Pinned::mapped_uninit(want)?, None),
            Fill::Deferred(artifact) => (Pinned::mapped(0)?, Some(artifact)),
        };
        let mut serving = Vec::with_capacity(layout.len());
        for (param, host_at, _, reserved) in layout.iter().copied() {
            serving.push(match &image {
                None => host.device().saturating_add(host_at),
                Some(artifact) => {
                    let id = u32::try_from(param).unwrap_or(u32::MAX);
                    let Some(bytes) = artifact.plane_reserved(id, reserved) else {
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

    #[must_use]
    pub fn pinned_at(&self, param: usize) -> Option<u64> {
        self.wholes
            .iter()
            .find(|whole| whole.param == param)
            .map(|whole| whole.serving_at)
    }

    #[must_use]
    pub fn mapped_at(&self, param: usize) -> Option<u64> {
        self.mapped
            .iter()
            .find(|plane| plane.param == param)
            .map(|plane| plane.at)
    }

    #[must_use]
    pub fn offloaded_at(&self, param: usize) -> Option<u64> {
        self.pinned_at(param).or_else(|| self.mapped_at(param))
    }

    #[must_use]
    pub fn spilled_bytes(&self) -> u64 {
        self.mapped.iter().map(|plane| plane.bytes).sum()
    }

    #[must_use]
    pub fn source(&self) -> Option<&Spill> {
        self.source.as_ref()
    }

    #[must_use]
    pub fn mapped_plane(&self, param: usize) -> Option<&[u8]> {
        let plane = self.mapped.iter().find(|plane| plane.param == param)?;
        let at = usize::try_from(plane.at).ok()? as *const u8;
        let len = usize::try_from(plane.bytes).ok()?;
        // SAFETY: the pair came from `Spill::plane`, a window on a mapping this tier owns in `source` for its whole life, unchanged since.
        Some(unsafe { std::slice::from_raw_parts(at, len) })
    }

    #[must_use]
    pub fn host(&self) -> &Pinned {
        &self.host
    }

    #[must_use]
    pub fn deferring(&self) -> bool {
        self.image.is_some() && self.host.bytes() == 0
    }

    #[must_use]
    pub fn deferred_image(&self) -> Option<&crate::checkpoint_serving::Serving> {
        self.image.as_ref().filter(|_| self.host.bytes() == 0)
    }

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

    pub fn undefer(&mut self) -> Result<()> {
        if !self.deferring() {
            return Ok(());
        }
        self.refill = None;
        let want = usize::try_from(self.plan.host_image()).unwrap_or(usize::MAX);
        self.host = Pinned::mapped(want)?;
        self.reseat();
        self.image = None;
        Ok(())
    }

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
        eprintln!(
            "engine-cuda: the deferred tier is INSTALLED after {window_ms} ms — every T1 \
             read is a page-locked read from here"
        );
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

    fn reseat(&mut self) {
        let base = self.host.device();
        for seat in &mut self.seats {
            seat.serving_at = base.saturating_add(seat.host_at);
        }
        for whole in &mut self.wholes {
            whole.serving_at = base.saturating_add(whole.host_at);
        }
    }

    pub fn zero_host(&self) {
        self.host.zero();
    }

    #[must_use]
    pub fn image(&self) -> Vec<(u64, u64, u64, u64)> {
        self.plan.host_layout()
    }

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

    fn copy_cell(&self, at: usize, stream: *mut c_void) -> Result<()> {
        let cell = self.groups[at].cell_at as u64 * CELL;
        copy_any(
            stream,
            self.cells.ptr() + cell,
            self.cell_shadow.device() + cell,
            CELL as usize,
        )
    }

    fn publish_all(&mut self, stream: *mut c_void) -> Result<()> {
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

    #[must_use]
    pub fn handles(&self, param: usize) -> Option<Handles> {
        self.seats.iter().find(|seat| seat.param == param).map(|seat| Handles {
            table: self.table.ptr() + seat.entry_at as u64 * ENTRY,
            counts: self.counts.ptr() + seat.counter_at as u64 * COUNTER,
        })
    }

    pub fn drain(&self, stream: *mut c_void) -> Result<()> {
        if self.counts.bytes() > 0 {
            copy_any(
                stream,
                self.mirror.device(),
                self.counts.ptr(),
                self.counts.bytes(),
            )?;
        }
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

    pub fn promote(&mut self, compute: *mut c_void, notify: *mut c_void) -> Result<u32> {
        if self.seats.is_empty() && self.groups.is_empty() && self.refill.is_none() {
            return Ok(0);
        }
        if self.moving && !self.ready.done()? {
            self.skipped += 1;
            return Ok(0);
        }
        if self.swap.is_some() && !self.landed.done()? {
            self.ladder.2 += 1;
            self.skipped += 1;
            return Ok(0);
        }
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
        self.ready.record(notify)?;
        self.ready.wait(compute)?;
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

    fn close_berth(&mut self, swap: Swap, notify: *mut c_void) -> Result<()> {
        let berth = &self.berths[swap.berth];
        let (tier, at) = (berth.tier, berth.at.clone());
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

    fn step(&self) -> Option<Step> {
        if let Some(swap) = self.swap {
            return Some(Step::Close(swap));
        }
        if GROUP_MOVES == 0 || self.berths.is_empty() || !self.ladder_open {
            return None;
        }
        let hits = self.group_mirror.read(0, self.group_mirror.bytes());
        self.decide_group(&hits).map(Step::Open)
    }

    fn decide_group(&self, hits: &[u8]) -> Option<Swap> {
        let count = |at: usize| -> u32 {
            let byte = at * COUNTER as usize;
            hits.get(byte..byte + COUNTER as usize)
                .and_then(|word| word.try_into().ok())
                .map_or(0, u32::from_ne_bytes)
        };
        vote(&self.berths, &self.groups, count)
    }

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

    #[must_use]
    pub fn ladder(&self) -> (u64, u64, u64) {
        self.ladder
    }

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

    #[must_use]
    pub fn motion(&self) -> (u64, u64, u64) {
        (self.promotions, self.demotions, self.skipped)
    }

    #[must_use]
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

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

fn vote(berths: &[Berth], groups: &[Group], count: impl Fn(usize) -> u32) -> Option<Swap> {
    let mut best: Option<(u32, Swap)> = None;
    for (at, berth) in berths.iter().enumerate() {
        let out = match berth.holds {
            Some(out) if groups[out].backing.is_empty() => continue,
            held => held,
        };
        let floor = out.map_or(0, |out| count(groups[out].cell_at));
        for (group, candidate) in groups.iter().enumerate() {
            if candidate.berth == Some(at) || candidate.held.rung() <= berth.tier.rung() {
                continue;
            }
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

fn address_of(seat: &Seat, expert: u32) -> u64 {
    match seat.slot_of[expert as usize] {
        Some(slot) => seat.slab + u64::from(slot) * seat.stride,
        None => pinned_address_of(seat, expert),
    }
}

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

    fn gpt_oss() -> Trace {
        let trace =
            models::sku("gptoss-20b-bf16-mxfp4-kv-bf16").expect("the catalog ships the SKU").trace;
        trace(Platform::Cuda)
    }

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
    fn experts_every_case() {
        a_packed_bank_the_plan_pairs_no_scales_with_is_refused_by_name();
        a_budget_under_the_planes_that_cannot_move_is_refused_by_name();
    }

    fn a_packed_bank_the_plan_pairs_no_scales_with_is_refused_by_name() {
        let trace = gpt_oss();
        let full = Plan::of(&trace, &scales_of(&trace), Budgets::uncapped())
            .expect("uncapped plans")
            .device_demand();
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

    fn a_budget_under_the_planes_that_cannot_move_is_refused_by_name() {
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

}
