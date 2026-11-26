//! A device's books: every reservation a backend makes on its device,
//! entered in one [`Ledger`] under its owner, beside the backend's own count
//! of the bytes the device gave it.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use engine::memory::{Audit, Class, Ledger};

/// Who holds a reservation: its class in the device's ledger, and a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Owner {
    pub class: Class,
    pub name: &'static str,
}

impl Owner {
    #[must_use]
    pub const fn model(name: &'static str) -> Owner {
        Owner {
            class: Class::Model,
            name,
        }
    }

    #[must_use]
    pub const fn state(name: &'static str) -> Owner {
        Owner {
            class: Class::State,
            name,
        }
    }

    #[must_use]
    pub const fn fire(name: &'static str) -> Owner {
        Owner {
            class: Class::Fire,
            name,
        }
    }
}

/// The owners the shared engine reserves under.
pub mod owners {
    use super::Owner;

    pub const WEIGHTS: Owner = Owner::model("weights");
    /// Planes made of the weights at load: decoded, relabelled.
    pub const DERIVED: Owner = Owner::model("derived weights");
    pub const NGRAM_HASHES: Owner = Owner::model("ngram hashes");
    /// The weight cache's tables, hit counters and upload buffers.
    pub const WEIGHT_CACHE: Owner = Owner::model("weight cache");
    /// The host tier's handle plane of streamed expert banks.
    pub const EXPERT_TIER: Owner = Owner::model("expert tier");
    pub const ADAPTERS: Owner = Owner::model("adapters");
    /// What a backend binds where a kernel names no buffer.
    pub const DUMMY: Owner = Owner::model("dummy");

    pub const POOL: Owner = Owner::state("pool");
    pub const HOST_POOL: Owner = Owner::state("host pool");
    pub const SPILL: Owner = Owner::state("spill staging");
    pub const CHANNELS: Owner = Owner::state("channels");
    pub const GUEST_STATE: Owner = Owner::state("guest state");

    /// The activation span at the top of the pool: entered as the fire's,
    /// out of the pool segment it lies in.
    pub const SPAN: Owner = Owner::fire("activation span");

    /// A kernel test's operands, outside any load.
    pub const TEST: Owner = Owner::fire("test operands");
    pub const INPUTS: Owner = Owner::fire("inputs");
    pub const READOUT: Owner = Owner::fire("readout");
    /// The recurrent planes and work plane ([`crate::rs::Reservation`]).
    pub const RS_WORK: Owner = Owner::fire("rs planes");
    pub const SNAPSHOTS: Owner = Owner::fire("snapshot table");
    pub const EXPERT_SEATS: Owner = Owner::fire("expert seats");
    pub const STAGING: Owner = Owner::fire("staging");
    pub const UNIFORMS: Owner = Owner::fire("uniforms");
    pub const FRAME_SCRATCH: Owner = Owner::fire("frame scratch");
    pub const TIMESTAMPS: Owner = Owner::fire("timestamps");
    pub const GUEST: Owner = Owner::fire("guest");
    pub const COLLECTIVE: Owner = Owner::fire("collective");
    pub const TUNING: Owner = Owner::fire("tuning");

    /// The fire-class owners named outside the span on a device with memory
    /// of its own (ONE-POOL step 2): host-visible memory (inputs, snapshot
    /// table, staging), query memory (timestamps), and the rigs the load
    /// times before any pool exists (tuning). A kernel test's operands
    /// stand outside any load.
    pub const EXCEPTIONS: [Owner; 6] = [INPUTS, SNAPSHOTS, STAGING, TIMESTAMPS, TUNING, TEST];
}

/// The device bytes every device's books in this process hold: a driver
/// reports its usage a process, not a device handle.
static PROCESS: AtomicU64 = AtomicU64::new(0);

/// What every reservation of this process holds on its devices.
#[must_use]
pub fn process_reserved() -> u64 {
    PROCESS.load(Ordering::Relaxed)
}

/// One device's books.
#[derive(Debug)]
pub struct Books {
    ledger: Mutex<Ledger>,
    /// The bytes the device gave this process, as the backend counts them
    /// where it allocates.
    device: AtomicU64,
    /// The host bytes the backend reserved through the device's driver off
    /// every device-local heap (pinned, host-visible memory: a guest
    /// stage's staging), which no device ledger holds: counted here so they
    /// are refused past the host's room rather than taken unseen.
    host: AtomicU64,
    /// The reservations taken, device and host, ever: read at a load's end
    /// and after, those since are what serving reserved, which the north
    /// star holds at none.
    takes: AtomicU64,
    /// Whether these bytes are the process's: a device's are, and a naming
    /// load's ([`Books::apart`]) reserve nothing.
    process: bool,
    /// The fire-class reservations held beside the span and booked under
    /// it (ONE-POOL step 2 (B)), each its owner's and the bytes it holds at
    /// most: the pool is sized without them, whether or not they are made
    /// yet ([`Books::unheld_booked`]).
    booked: Mutex<Vec<(Owner, u64)>>,
    /// The owners whose reservations are regions of the span
    /// ([`Books::region`]).
    regions: Mutex<Vec<Owner>>,
}

impl Default for Books {
    fn default() -> Books {
        Books::new()
    }
}

impl Books {
    /// Books of a device whose capacity is not yet measured.
    #[must_use]
    pub fn new() -> Books {
        Books {
            ledger: Mutex::new(Ledger::new(1, u64::MAX)),
            device: AtomicU64::new(0),
            host: AtomicU64::new(0),
            takes: AtomicU64::new(0),
            process: true,
            booked: Mutex::new(Vec::new()),
            regions: Mutex::new(Vec::new()),
        }
    }

    /// Books apart from the process's tally: what a load would reserve,
    /// entered while nothing is.
    #[must_use]
    pub fn apart() -> Books {
        Books {
            process: false,
            ..Books::new()
        }
    }

    fn with<R>(&self, f: impl FnOnce(&mut Ledger) -> R) -> R {
        f(&mut self.ledger.lock().expect("ledger lock poisoned"))
    }

    /// The device gave `bytes` to `owner`.
    pub fn take(&self, owner: Owner, bytes: u64) {
        self.takes.fetch_add(1, Ordering::Relaxed);
        self.device.fetch_add(bytes, Ordering::Relaxed);
        if self.process {
            PROCESS.fetch_add(bytes, Ordering::Relaxed);
        }
        self.with(|ledger| {
            let held = ledger.of(owner.class, owner.name);
            ledger.enter(owner.class, owner.name, held.saturating_add(bytes));
        });
    }

    /// `owner` gave `bytes` back to the device.
    pub fn give(&self, owner: Owner, bytes: u64) {
        self.device.fetch_sub(bytes, Ordering::Relaxed);
        if self.process {
            PROCESS.fetch_sub(bytes, Ordering::Relaxed);
        }
        self.with(|ledger| {
            let held = ledger.of(owner.class, owner.name);
            ledger.enter(owner.class, owner.name, held.saturating_sub(bytes));
        });
    }

    /// The driver gave `bytes` of host memory off the device's heaps.
    pub fn host_take(&self, bytes: u64) {
        self.takes.fetch_add(1, Ordering::Relaxed);
        self.host.fetch_add(bytes, Ordering::Relaxed);
    }

    /// The reservations taken so far, device and host ([`Books::take`],
    /// [`Books::host_take`]), none given back uncounted.
    #[must_use]
    pub fn takes(&self) -> u64 {
        self.takes.load(Ordering::Relaxed)
    }

    /// Host memory of `bytes` off the device's heaps went back.
    pub fn host_give(&self, bytes: u64) {
        self.host.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// The host bytes reserved off the device's heaps ([`Books::host_take`]).
    #[must_use]
    pub fn host(&self) -> u64 {
        self.host.load(Ordering::Relaxed)
    }

    /// `bytes` of a reservation entered under `from` are `to`'s: the
    /// device's count stands. A reservation relabelled in part gives back
    /// under its own owner, so the moved bytes stay under `to` once it is
    /// gone; only the activation span is, and it stands as long as its
    /// device.
    pub fn relabel(&self, from: Owner, to: Owner, bytes: u64) {
        self.with(|ledger| {
            let held = ledger.of(from.class, from.name);
            let moved = bytes.min(held);
            ledger.enter(from.class, from.name, held - moved);
            let there = ledger.of(to.class, to.name);
            ledger.enter(to.class, to.name, there.saturating_add(moved));
        });
    }

    /// `owner` holds at most `bytes` beside the span, booked under it: the
    /// pool is sized without them. A booking stated again replaces the one
    /// before.
    pub fn book(&self, owner: Owner, bytes: u64) {
        let mut booked = self.booked.lock().expect("booked lock poisoned");
        booked.retain(|&(held, _)| held != owner);
        booked.push((owner, bytes));
    }

    /// The bytes booked beside the span that their owners do not hold yet:
    /// what a pool sized now leaves out of the room the device has left.
    #[must_use]
    pub fn unheld_booked(&self) -> u64 {
        let booked = self.booked.lock().expect("booked lock poisoned").clone();
        self.unheld(&booked)
    }

    /// The bytes of `booked` (owners and the bytes each holds at most) that
    /// their owners do not hold yet.
    #[must_use]
    pub fn unheld(&self, booked: &[(Owner, u64)]) -> u64 {
        self.with(|ledger| {
            booked
                .iter()
                .map(|&(owner, bytes)| bytes.saturating_sub(ledger.of(owner.class, owner.name)))
                .sum()
        })
    }

    /// `bytes` of the activation span are `owner`'s region of it: entered
    /// under `owner`, out of the span's.
    pub fn region(&self, owner: Owner, bytes: u64) {
        self.relabel(owners::SPAN, owner, bytes);
        let mut regions = self.regions.lock().expect("regions lock poisoned");
        if !regions.contains(&owner) {
            regions.push(owner);
        }
    }

    /// The fire class as the pool holds it, in MiB: what lies in the span
    /// (its carve and each region), what is booked beside it (held, and
    /// booked at most), the exceptions named outside it, and any fire
    /// entry none of those names (`unbooked`), which step 2 leaves none of.
    #[must_use]
    pub fn fire_line(&self) -> String {
        let booked = self.booked.lock().expect("booked lock poisoned").clone();
        let regions = self.regions.lock().expect("regions lock poisoned").clone();
        let mib = |bytes: u64| format!("{:.1}", bytes as f64 / f64::from(1u32 << 20));
        let ledger = self.ledger();
        let fire: Vec<(&'static str, u64)> = ledger
            .entries()
            .filter(|&(class, _, _)| class == Class::Fire)
            .map(|(_, name, bytes)| (name, bytes))
            .collect();
        let named = |owner: &Owner| owner.class == Class::Fire;
        let list = |picked: &[(&'static str, u64)]| -> String {
            let total: u64 = picked.iter().map(|&(_, bytes)| bytes).sum();
            let each: Vec<String> = picked
                .iter()
                .map(|&(name, bytes)| format!("{name} {}", mib(bytes)))
                .collect();
            format!("{} MiB ({})", mib(total), each.join(", "))
        };
        let of = |name: &str| fire.iter().find(|&&(n, _)| n == name).map_or(0, |&(_, b)| b);
        let span: Vec<(&'static str, u64)> = std::iter::once(owners::SPAN)
            .chain(regions.iter().copied().filter(named))
            .map(|owner| (owner.name, of(owner.name)))
            .collect();
        let beside: Vec<String> = booked
            .iter()
            .filter(|(owner, _)| named(owner))
            .map(|&(owner, bytes)| {
                format!("{} {} of {}", owner.name, mib(of(owner.name)), mib(bytes))
            })
            .collect();
        let excepted: Vec<(&'static str, u64)> = fire
            .iter()
            .copied()
            .filter(|&(name, _)| owners::EXCEPTIONS.iter().any(|owner| owner.name == name))
            .collect();
        let unbooked: Vec<(&'static str, u64)> = fire
            .iter()
            .copied()
            .filter(|&(name, _)| {
                name != owners::SPAN.name
                    && !regions.iter().any(|owner| owner.name == name)
                    && !booked.iter().any(|(owner, _)| owner.name == name)
                    && !owners::EXCEPTIONS.iter().any(|owner| owner.name == name)
            })
            .collect();
        format!(
            "in the span {}; booked beside it {} MiB ({}); exceptions {}; unbooked {}",
            list(&span),
            mib(booked
                .iter()
                .filter(|(owner, _)| named(owner))
                .map(|(owner, _)| of(owner.name))
                .sum()),
            beside.join(", "),
            list(&excepted),
            list(&unbooked)
        )
    }

    /// What the driver holds for this process beside every reservation,
    /// as measured: an entry of its own, not a reservation.
    pub fn outside(&self, name: &'static str, bytes: u64) {
        self.with(|ledger| ledger.enter(Class::Outside, name, bytes));
    }

    /// The bytes the device gave this process, by the backend's count.
    #[must_use]
    pub fn device(&self) -> u64 {
        self.device.load(Ordering::Relaxed)
    }

    /// The bytes entered under every class but the driver's.
    #[must_use]
    pub fn reserved(&self) -> u64 {
        self.with(|ledger| {
            [Class::Model, Class::State, Class::Fire]
                .iter()
                .map(|class| ledger.bytes(*class))
                .sum()
        })
    }

    /// The ledger counted in chunks of `chunk_bytes` against `capacity`.
    pub fn settle(&self, chunk_bytes: u64, capacity: u64) {
        self.with(|ledger| {
            let mut settled = Ledger::new(chunk_bytes, capacity);
            for (class, owner, bytes) in ledger.entries() {
                settled.enter(class, owner, bytes);
            }
            *ledger = settled;
        });
    }

    /// The ledger as it stands.
    #[must_use]
    pub fn ledger(&self) -> Ledger {
        self.with(|ledger| ledger.clone())
    }

    /// The reservations entered against the backend's count of them.
    #[must_use]
    pub fn audit(&self) -> Audit {
        let reserved = self.reserved();
        Audit {
            entered: reserved,
            device: self.device(),
        }
    }
}
