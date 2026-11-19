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
    pub const RS_SCRATCH: Owner = Owner::fire("rs scratch");
    /// The lagged fold's carried rows' outputs, laid at load.
    pub const RS_CARRIED: Owner = Owner::fire("rs carried rows");
    pub const SNAPSHOTS: Owner = Owner::fire("snapshot table");
    pub const EXPERT_SEATS: Owner = Owner::fire("expert seats");
    pub const STAGING: Owner = Owner::fire("staging");
    pub const UNIFORMS: Owner = Owner::fire("uniforms");
    pub const FRAME_SCRATCH: Owner = Owner::fire("frame scratch");
    pub const TIMESTAMPS: Owner = Owner::fire("timestamps");
    pub const GUEST: Owner = Owner::fire("guest");
    pub const COLLECTIVE: Owner = Owner::fire("collective");
    pub const TUNING: Owner = Owner::fire("tuning");
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
    /// Whether these bytes are the process's: a device's are, and a naming
    /// load's ([`Books::apart`]) reserve nothing.
    process: bool,
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
            process: true,
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
        self.host.fetch_add(bytes, Ordering::Relaxed);
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

#[cfg(test)]
mod tests {
    use super::{Books, owners};

    /// Host memory a backend reserved off the device's heaps (a guest
    /// stage's staging) is counted apart from the device's: it moves the
    /// host count and leaves the device's books, and their audit, alone.
    #[test]
    fn host_memory_off_the_device_is_counted_apart() {
        let books = Books::apart();
        books.take(owners::GUEST, 100);
        books.host_take(4096);
        books.host_take(64);
        assert_eq!((books.device(), books.host()), (100, 4160));
        assert_eq!(books.audit().entered, 100);
        books.host_give(4096);
        assert_eq!(books.host(), 64);
    }
}
