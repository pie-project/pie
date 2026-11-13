//! Tier pressure: how short a tier of state memory is, as the worker policy
//! reads it.
//!
//! Three roles share memory. The governor decides how much a tier holds;
//! PLEX decides what stays in it; the pool decides how it moves. So the
//! runtime never drops state on its own. A waiter short of room calls
//! [`raise`] with what its tier is short of. The worker policy hears that as
//! one `tier-pressure` edge per band crossing, and reads the bytes as attrs
//! on the worker (`{device,host}-{total,state,live,wanted,act,short}-bytes`
//! and `-band`), beside the device's two ledger terms
//! (`{device,host}-{model,outside}-bytes`, [`Ledger`]). It answers with the
//! verbs it already has: `discard` a working set, or install retention
//! keys. Attrs are not verbs: the model cache's bytes say what live demand
//! may take from the weights before it asks for state.
//!
//! When no installed policy subscribes to `tier-pressure`, the default
//! responder answers for it ([`choose`]). It uses the same `discard` verb a
//! policy would, lowest retention key first, so a parked waiter always has
//! someone to answer it. This is policy-layer code: it decides what goes, as
//! a policy would. It is not part of the stores, which only carry out
//! discards.
//!
//! The one thing the runtime may still reclaim without asking is idle cache
//! whose PLEX retention key says it may be dropped ([`may_drop`]). Everything
//! else goes through `tier-pressure`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use super::Event;
use crate::rt::Instant;
use crate::store::registry as store_registry;

/// A tier of state memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    Device,
    Host,
}

impl Tier {
    pub const ALL: [Tier; 2] = [Tier::Device, Tier::Host];

    /// Its attrs' prefix.
    pub const fn name(self) -> &'static str {
        match self {
            Tier::Device => "device",
            Tier::Host => "host",
        }
    }

    pub(crate) const fn index(self) -> usize {
        match self {
            Tier::Device => 0,
            Tier::Host => 1,
        }
    }
}

/// How short a tier is, coarsely. A policy hears one edge each time a tier
/// moves from one band to another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Band {
    /// No waiter is short.
    #[default]
    Clear,
    /// A waiter is short of room.
    Short,
    /// The waiters are short of a quarter of the tier or more.
    Critical,
}

impl Band {
    /// Its attr value (`{tier}-band`).
    pub const fn name(self) -> &'static str {
        match self {
            Band::Clear => "clear",
            Band::Short => "short",
            Band::Critical => "critical",
        }
    }

    /// The band `short` bytes of `total` fall in, coming from `from`.
    ///
    /// Critical is entered at a quarter of the tier and left only below an
    /// eighth, so a shortfall that hovers near the line crosses it once.
    /// A tier of unknown size (`total` 0) is never critical. Clear needs no
    /// margin: it is exact, every waiter served.
    pub fn of(from: Band, short: u64, total: u64) -> Band {
        if short == 0 {
            return Band::Clear;
        }
        if total == 0 {
            return Band::Short;
        }
        let enters = u128::from(short) * 4 >= u128::from(total);
        let stays = u128::from(short) * 8 >= u128::from(total);
        match from {
            Band::Critical if stays => Band::Critical,
            _ if enters => Band::Critical,
            _ => Band::Short,
        }
    }
}

/// What a tier is short of now, and its band.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reading {
    pub band: Band,
    /// What every waiter on the tier is short of, in bytes.
    pub short: u64,
}

/// A tier's bytes, as the worker policy reads them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bytes {
    /// What the tier holds for state and activations together.
    pub total: u64,
    /// Held by state now: working sets, snapshots and cache.
    pub state: u64,
    /// Of `state`, what no idle cache merely keeps.
    pub live: u64,
    /// What would be held if nothing were short: `live`, the cache, and the
    /// waiters' shortfall.
    pub wanted: u64,
    /// Held for activations inside `total`.
    pub act: u64,
}

/// One engine's device bytes outside its state pools, as the ledger
/// (REDESIGN 2.1, 2.8) puts them to the policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ledger {
    /// The model cache: weight atoms resident above the floor, what live
    /// demand may take before any state gives way.
    pub model: u64,
    /// What the driver holds beside the engine's span (graph bodies,
    /// modules, library workspaces), as the engine states it
    /// ([`::engine::live::Live::outside`]).
    pub outside: u64,
}

/// Per engine, its [`Ledger`] as its governor last read it.
static LEDGERS: RwLock<BTreeMap<usize, Ledger>> = RwLock::new(BTreeMap::new());

/// `engine`'s ledger terms now: what its governor reads each tick.
pub fn set_ledger(engine: usize, ledger: Ledger) {
    LEDGERS.write().unwrap().insert(engine, ledger);
}

/// Every engine's ledger terms on `tier`, summed; none off the device.
#[must_use]
pub fn ledger(tier: Tier) -> Ledger {
    if tier != Tier::Device {
        return Ledger::default();
    }
    LEDGERS
        .read()
        .unwrap()
        .values()
        .fold(Ledger::default(), |sum, one| Ledger {
            model: sum.model.saturating_add(one.model),
            outside: sum.outside.saturating_add(one.outside),
        })
}

/// What reads a tier's bytes. The stores' own reading ([`stores`]) stands
/// until a pool installs a better one ([`install_meter`]).
pub trait Meter: Send + Sync {
    /// The tier's bytes, or `None` where they cannot be told.
    fn bytes(&self, tier: Tier) -> Option<Bytes>;
}

/// What one engine's pages and slots take, in bytes; 0 where unknown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scale {
    pub kv_page_bytes: u64,
    pub state_slot_bytes: u64,
}

struct Gauge {
    /// Short bytes by who raised them (an engine's planner).
    short: BTreeMap<usize, u64>,
    band: Band,
    reported: Option<Instant>,
}

impl Gauge {
    const fn new() -> Self {
        Self {
            short: BTreeMap::new(),
            band: Band::Clear,
            reported: None,
        }
    }

    fn short(&self) -> u64 {
        self.short.values().fold(0, |sum, &s| sum.saturating_add(s))
    }
}

/// A shortfall that stands unchanged is reported again this often, so the
/// default responder looks again when its last discard freed too little.
pub const AGAIN: Duration = Duration::from_millis(100);

static GAUGES: Mutex<[Gauge; 2]> = Mutex::new([Gauge::new(), Gauge::new()]);
static METER: RwLock<Option<Arc<dyn Meter>>> = RwLock::new(None);
static SCALES: RwLock<Vec<Scale>> = RwLock::new(Vec::new());

/// Says that waiters on `tier` from `from` (an engine's planner) are short
/// of `short_bytes` in all, now: not an increment. Zero once they are all
/// served. The worker policy hears of it when the tier's band changes; the
/// default responder, whenever the shortfall changes or stands for a while.
pub fn raise(tier: Tier, from: usize, short_bytes: u64) {
    let total = meter_bytes(tier).map_or(0, |bytes| bytes.total);
    let reading = {
        let mut gauges = GAUGES.lock().unwrap();
        let gauge = &mut gauges[tier.index()];
        let before = gauge.short();
        if short_bytes == 0 {
            gauge.short.remove(&from);
        } else {
            gauge.short.insert(from, short_bytes);
        }
        let short = gauge.short();
        let band = Band::of(gauge.band, short, total);
        let now = Instant::now();
        let stale = short > 0 && gauge.reported.is_none_or(|at| now - at >= AGAIN);
        if band == gauge.band && short == before && !stale {
            return;
        }
        gauge.band = band;
        gauge.reported = Some(now);
        Reading { band, short }
    };
    tracing::debug!(
        tier = tier.name(),
        band = reading.band.name(),
        short = reading.short,
        "tier pressure"
    );
    super::report(Event::TierPressure(tier, reading));
}

/// The waiters from `from` on `tier` are all served.
pub fn relieve(tier: Tier, from: usize) {
    raise(tier, from, 0);
}

/// Where `tier` stands now.
pub fn reading(tier: Tier) -> Reading {
    let gauges = GAUGES.lock().unwrap();
    let gauge = &gauges[tier.index()];
    Reading {
        band: gauge.band,
        short: gauge.short(),
    }
}

/// Whether the runtime may drop idle cache under `key` without asking the
/// policy. Only a key at or below the retention floor says so; above it,
/// room is asked for with `tier-pressure`.
pub const fn may_drop(key: i64) -> bool {
    !super::retention::above_floor(key)
}

/// Replaces the meter the worker policy reads bytes from.
pub fn install_meter(meter: Option<Arc<dyn Meter>>) {
    *METER.write().unwrap() = meter;
}

/// What `engine`'s pages and slots take, for the stores' meter.
pub fn scale(engine: usize, scale: Scale) {
    let mut scales = SCALES.write().unwrap();
    if scales.len() <= engine {
        scales.resize(engine + 1, Scale::default());
    }
    scales[engine] = scale;
}

/// What `engine`'s pages and slots take, as [`scale`] last said.
pub fn scale_of(engine: usize) -> Scale {
    SCALES
        .read()
        .unwrap()
        .get(engine)
        .copied()
        .unwrap_or_default()
}

/// The scale every engine shares, if known: they run one model.
pub fn common_scale() -> Option<Scale> {
    let scales = SCALES.read().unwrap();
    let first = *scales.first()?;
    scales.iter().all(|s| *s == first).then_some(first)
}

/// The installed meter's reading, or the stores'.
pub fn meter_bytes(tier: Tier) -> Option<Bytes> {
    let meter = METER.read().unwrap().clone();
    match meter {
        Some(meter) => meter.bytes(tier),
        None => stores(tier),
    }
}

/// The stores' own reading: kv pages and recurrent slots at their scale,
/// summed over engines. `None` where a page's size is unknown. Activations
/// sit outside these pools today, so `act` is 0 and `total` is the pools'.
pub fn stores(tier: Tier) -> Option<Bytes> {
    let scales = SCALES.read().unwrap().clone();
    let all = store_registry::all_for_model(0);
    if all.is_empty() {
        return None;
    }
    let mut sum = Bytes::default();
    for (engine, stores) in all.iter().enumerate() {
        let scale = scales.get(engine).copied().unwrap_or_default();
        if scale.kv_page_bytes == 0 {
            return None;
        }
        let page = |pages: u64| pages.saturating_mul(scale.kv_page_bytes);
        let slot = |slots: u32| u64::from(slots).saturating_mul(scale.state_slot_bytes);
        let bytes = match tier {
            Tier::Device => {
                let (capacity, out, wanted, kept) =
                    store_registry::with_kv_lock(&stores.kv, "policy-pressure", |kv| {
                        (
                            kv.capacity_pages(),
                            kv.pages_out(),
                            kv.pages_wanted(),
                            kv.reclaimable_cache_pages(),
                        )
                    });
                let rs = stores.rs.lock().unwrap();
                let slots = rs.capacity().slots;
                let held = slots.saturating_sub(rs.available().slots);
                let (asked, in_use) = rs.demand();
                drop(rs);
                let state = page(u64::from(out)).saturating_add(slot(held));
                Bytes {
                    total: page(u64::from(capacity)).saturating_add(slot(slots)),
                    state,
                    live: page(u64::from(out.saturating_sub(kept)))
                        .saturating_add(slot(in_use.min(held))),
                    wanted: page(u64::from(wanted.max(out))).saturating_add(slot(asked.max(held))),
                    act: 0,
                }
            }
            Tier::Host => {
                let (capacity, free, cached) =
                    store_registry::with_kv_lock(&stores.kv, "policy-pressure", |kv| {
                        (
                            kv.host_swap_capacity(),
                            kv.host_swap_available(),
                            kv.host_cache_pages(),
                        )
                    });
                let rs = stores.rs.lock().unwrap();
                let (slots, free_slots) = (rs.host_capacity().slots, rs.host_available().slots);
                drop(rs);
                let state = page(u64::from(capacity).saturating_sub(free as u64))
                    .saturating_add(slot(slots.saturating_sub(free_slots)));
                // Idle cache parked on the host is cache, not live.
                Bytes {
                    total: page(u64::from(capacity)).saturating_add(slot(slots)),
                    state,
                    live: state.saturating_sub(page(u64::from(cached))),
                    wanted: state,
                    act: 0,
                }
            }
        };
        sum = Bytes {
            total: sum.total.saturating_add(bytes.total),
            state: sum.state.saturating_add(bytes.state),
            live: sum.live.saturating_add(bytes.live),
            wanted: sum.wanted.saturating_add(bytes.wanted),
            act: sum.act.saturating_add(bytes.act),
        };
    }
    Some(sum)
}

/// The retention key that pins a working set: the default responder never
/// discards it, however short a tier is. Only a policy that answers
/// `tier-pressure` itself may let it go.
pub const PINNED: i64 = i64::MAX;

/// A working set the default responder may discard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub id: String,
    /// Its retention key: the highest any of its tokens has.
    pub key: i64,
    pub last_use_ms: u64,
    /// What discarding it frees on the pressed tier; `None` where unknown.
    pub bytes: Option<u64>,
}

/// The default responder: which working sets to discard so that `short`
/// bytes come free. Lowest retention key first, so whatever a key `<= 0`
/// lets go goes before anything a policy keyed higher; then least recently
/// used. One that frees nothing on the tier, or one [`PINNED`], is passed
/// over. It stops once
/// what it chose covers `short`, or after one whose size is unknown: the
/// waiter's next [`raise`] says whether that was enough.
pub fn choose(mut candidates: Vec<Candidate>, short: u64) -> Vec<String> {
    candidates.retain(|c| c.bytes != Some(0) && c.key != PINNED);
    candidates.sort_by(|a, b| (a.key, a.last_use_ms, &a.id).cmp(&(b.key, b.last_use_ms, &b.id)));
    let mut chosen = Vec::new();
    let mut freed: u64 = 0;
    for candidate in candidates {
        if freed >= short {
            break;
        }
        chosen.push(candidate.id);
        match candidate.bytes {
            Some(bytes) => freed = freed.saturating_add(bytes),
            None => break,
        }
    }
    chosen
}
