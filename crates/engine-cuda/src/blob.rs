//! **THE SHARED ADAPTER STORE: FILES ARE THE TRUTH, THE BANK IS A CACHE**
//! (alto adapter §3.3, promoted to wave 1 by §6.1).
//!
//! # Why this module exists at all, and why it is not a channel
//!
//! The 0.3 surface seeds an adapter through a channel, and the audit
//! (adapter.md §6.1) ruled that a smell rather than a transport: a 12 MiB
//! cell is legal, but `CHAN_READ` materialises the whole cell into per-lane
//! scratch and `pull_validate` re-drags the mirror over mapped-pinned PCIe
//! **every fire**, so the weights would be re-paid per token forever. The
//! ruling: an adapter channel is a NAMING device, the bytes come off a file,
//! and the landing happens ONCE at instance bind. Nothing in this module is
//! reachable from the fire path — `Shell::fire` never calls into it, which is
//! the A-5 gate's whole claim.
//!
//! # The three pieces, and why they are three
//!
//! * [`Vfs`] — the mount. A read-only shared directory, stated by the
//!   deployment and never discovered from the environment (design article 9).
//!   Its whole job is turning a guest-spelled name into a path inside the
//!   mount, and refusing everything else BY NAME.
//! * [`Blobs`] — the host byte cache, refcounted and **single-flight**: two
//!   binds racing on one file perform one read, the second waiting on the
//!   first rather than starting a second. The bytes live as long as some
//!   handle does and no longer ([`Blobs`] holds `Weak`s), because once a blob
//!   has been landed into a slot the slot is the residency and the host copy
//!   is dead weight.
//! * [`Slots`] — the device residency, keyed by BLOB IDENTITY. This is where
//!   "N instances of one adapter occupy ONE slot" is decided; it is a pure
//!   host table, which is why every claim about sharing, eviction and
//!   exhaustion is checkable with no GPU in the machine.
//!
//! # Identity, and why it is a stamp and not only a fingerprint
//!
//! §3.3 states identity as "path + content fingerprint, snapshotted at load".
//! A fingerprint alone cannot be the KEY, because computing it means reading
//! the file, and a store that read the file to decide whether it needed to
//! read the file is not a cache. So the key is a [`Stamp`] — the adapter's
//! resolved directory plus `(file, len, mtime)` for its manifest and every
//! plane it names —
//! and the fingerprint is computed on the load that the stamp missed and
//! recorded on the occupant ([`Slots::resident`]) as the snapshot §3.3 asks
//! for. A rewritten file is a new stamp, therefore a new key, therefore a new
//! slot: no in-flight fire ever observes an adapter changing, which is the
//! property the sentence was protecting.
//!
//! # The mount's shape
//!
//! One directory per adapter, `adapter.toml` inside it:
//!
//! ```toml
//! rank = 8
//!
//! [[plane]]
//! role = "lora_a"
//! file = "lora_a.bin"
//! layout = "rank_major"   # [layers, rank, hidden]
//!
//! [[plane]]
//! role = "lora_b"
//! file = "lora_b.bin"
//! layout = "out_major"    # [layers, hidden, rank] — HF's native orientation
//! ```
//!
//! **THE FILE DECLARES ITS ORIENTATION BECAUSE BYTES CANNOT.** §6.3's statute
//! is that `B` ships out-major and a rank-major `B` is REFUSED rather than
//! repacked — and orientation is not observable in a byte string, so it has to
//! be said. It is said in the file rather than at the API because §3.3's
//! hot-add is a file drop: nobody is standing there to state it.
//!
//! `role` is the bank's name with its `layer.{l}.` prefix cut, which is how
//! §6.3's "the resolver slices, per layer" is spelled: a `[layers, …]` source
//! is `L` contiguous slices and the `L` banks that carry one role take one
//! each, in layer order. [`crate::weights`] itself still pairs nothing and
//! matches no suffix — that module's promise is intact, and the convention
//! lives here, in the resolver, which is where §6.3 put it.
//!
//! # What is refused, by name
//!
//! * a name outside the mount, or a mount that was never stated — [`Fault::Blob`]
//! * an adapter directory with no manifest, or a manifest naming a file that
//!   is not there — [`Fault::Blob`], the "unknown blob path" of §5
//! * a plane shorter (or longer) than the banks that carry its role seat —
//!   [`Fault::Blob`] with BOTH numbers
//! * a rank-major source into an out-major bank — [`Fault::Blob`], naming the
//!   repack kernel this shell does not ship
//! * every slot pinned by a live bind — [`Fault::AdapterSlots`], refused at
//!   the keying moment rather than by evicting something in flight (§5)

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
use std::time::UNIX_EPOCH;

use crate::error::{Fault, Result};
use crate::weights::BankSeat;
use crate::AdapterPlane;

/// The one file an adapter directory must carry.
pub const MANIFEST: &str = "adapter.toml";

// ── the mount ────────────────────────────────────────────────────────────

/// The read-only shared directory adapters are files in.
///
/// `None` for a root is the feature OFF — a deployment that mounted nothing,
/// whose every `open` is a refusal that says so. That is the same posture
/// [`Boot::weight_cache_dir`](crate::Boot::weight_cache_dir) takes, and for
/// the same reason: an absent directory is an answer, not a default to guess.
#[derive(Debug, Clone, Default)]
pub struct Vfs {
    root: Option<PathBuf>,
}

impl Vfs {
    /// Mount `root`, or nothing.
    #[must_use]
    pub fn new(root: Option<PathBuf>) -> Vfs {
        Vfs { root }
    }

    /// Where this mount is, if it is anywhere.
    #[must_use]
    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    /// Turn a guest-spelled adapter name into a directory inside the mount.
    ///
    /// A leading `/` is cut — the guest spells `/shared/alice-v2` against its
    /// own preopen and this shell holds the mount, so the two meet on the
    /// tail. Every component after that must be a plain name: no `..`, no
    /// second root, no prefix. A traversal is not sanitized into something
    /// else, it is refused.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for an unmounted shell, an empty name, a component
    /// that is not a plain one, or a directory that is not there.
    pub fn resolve(&self, name: &str) -> Result<PathBuf> {
        let root = self.root.as_ref().ok_or_else(|| Fault::Blob {
            path: name.to_string(),
            why: "cannot be opened: this shell has no shared adapter directory mounted, \
                  so there is no namespace for the name to be in"
                .to_string(),
        })?;
        let trimmed = name.trim().trim_start_matches('/');
        if trimmed.is_empty() {
            return Err(Fault::Blob {
                path: name.to_string(),
                why: "is not a name; an adapter is a directory in the mount".to_string(),
            });
        }
        let relative = Path::new(trimmed);
        for part in relative.components() {
            if !matches!(part, Component::Normal(_)) {
                return Err(Fault::Blob {
                    path: name.to_string(),
                    why: "leaves the mount; a shared adapter is named by plain path \
                          components under the mount root and never by `..` or a second root"
                        .to_string(),
                });
            }
        }
        let at = root.join(relative);
        if !at.is_dir() {
            return Err(Fault::Blob {
                path: name.to_string(),
                why: format!(
                    "is not a directory in the mount at {}; a shared adapter is a \
                     directory holding `{MANIFEST}` and the planes it names",
                    root.display()
                ),
            });
        }
        Ok(at)
    }
}

// ── the manifest ─────────────────────────────────────────────────────────

/// Which way a source plane is laid out (§6.3's statute, stated by the file).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `[layers, rank, hidden]` — the rank is the leading axis of a slot.
    RankMajor,
    /// `[layers, hidden, rank]` — HF's native `B`, the rank a stride inside
    /// every row.
    OutMajor,
}

impl Layout {
    fn spelled(self) -> &'static str {
        match self {
            Layout::RankMajor => "rank-major [rank, hidden]",
            Layout::OutMajor => "out-major [hidden, rank]",
        }
    }

    /// **WHICH ORIENTATION A BANK'S OWN RECTANGLE CARRIES.**
    ///
    /// The plan declares a bank as `[adapters, rank, hidden]` or
    /// `[adapters, hidden, rank]` and marks neither, so the shell reads the
    /// only thing that distinguishes them: the rank axis is the SHORT one. A
    /// LoRA whose rank meets its hidden width has no waist and is not the
    /// thing this class exists for, so the tie is degenerate; it is taken at
    /// the file's word rather than refused, because either reading of a
    /// square bank names the same number of bytes per row.
    fn of_bank(seat: &BankSeat) -> Layout {
        match seat.rows <= seat.cols {
            true => Layout::RankMajor,
            false => Layout::OutMajor,
        }
    }
}

/// One plane the manifest names: which bank role it fills, from what file,
/// laid out which way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneSpec {
    /// The bank's name with its `layer.{l}.` prefix cut — `lora_a`.
    pub role: String,
    /// The file inside the adapter's directory.
    pub file: String,
    /// Which way its bytes run.
    pub layout: Layout,
}

/// What an adapter directory declares about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The rank the adapter was trained at. May be under the bank's — the
    /// resolver pads, per orientation — and never over it.
    pub rank: u64,
    /// Its planes, in the order the file names them.
    pub planes: Vec<PlaneSpec>,
}

impl Manifest {
    /// Read `dir/adapter.toml`.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a directory with no manifest, a manifest that is
    /// not TOML, or one that omits a key. Every one of them names the adapter
    /// and says what was missing: a hot-added directory is written by an
    /// operator with no compiler in the loop, so the message IS the diagnostic.
    pub fn read(dir: &Path, name: &str) -> Result<Manifest> {
        let at = dir.join(MANIFEST);
        let refuse = |why: String| Fault::Blob {
            path: name.to_string(),
            why,
        };
        let text = std::fs::read_to_string(&at)
            .map_err(|error| refuse(format!("has no readable `{MANIFEST}`: {error}")))?;
        let doc: toml::Table = text
            .parse()
            .map_err(|error| refuse(format!("has a `{MANIFEST}` that is not TOML: {error}")))?;
        let rank = doc
            .get("rank")
            .and_then(toml::Value::as_integer)
            .and_then(|rank| u64::try_from(rank).ok())
            .filter(|rank| *rank > 0)
            .ok_or_else(|| {
                refuse(format!(
                    "declares no positive `rank` in its `{MANIFEST}`; the rank is what \
                     says how much of a bank's slot the source fills"
                ))
            })?;
        let planes = doc
            .get("plane")
            .and_then(toml::Value::as_array)
            .ok_or_else(|| {
                refuse(format!(
                    "names no `[[plane]]` in its `{MANIFEST}`; an adapter with no planes \
                     lands nothing"
                ))
            })?;
        let planes = planes
            .iter()
            .map(|plane| {
                let plane = plane.as_table().ok_or_else(|| {
                    refuse("has a `[[plane]]` that is not a table".to_string())
                })?;
                let field = |key: &str| {
                    plane
                        .get(key)
                        .and_then(toml::Value::as_str)
                        .map(str::to_string)
                        .ok_or_else(|| refuse(format!("has a `[[plane]]` with no `{key}`")))
                };
                let role = field("role")?;
                let file = field("file")?;
                let layout = match field("layout")?.as_str() {
                    "rank_major" => Layout::RankMajor,
                    "out_major" => Layout::OutMajor,
                    other => {
                        return Err(refuse(format!(
                            "declares plane `{role}` as layout `{other}`; the two layouts \
                             a bank can be filled from are `rank_major` ([rank, hidden]) \
                             and `out_major` ([hidden, rank])"
                        )));
                    }
                };
                Ok(PlaneSpec { role, file, layout })
            })
            .collect::<Result<Vec<PlaneSpec>>>()?;
        Ok(Manifest { rank, planes })
    }
}

/// The bank name's role — everything after a `layer.{l}.` prefix.
///
/// `layer.7.lora_a` is the seventh layer's `lora_a`; a bank named without a
/// numbered component is its own role at layer zero. This is the ONLY name
/// convention in the adapter axis and it lives here rather than in
/// [`crate::weights`] on purpose (§6.3: the resolver slices).
#[must_use]
pub fn role_of(bank: &str) -> &str {
    match bank.rsplit_once('.') {
        Some((head, tail)) if head.rsplit('.').next().is_some_and(numbered) => tail,
        _ => bank,
    }
}

/// Which layer a bank name puts itself at, or zero for an unnumbered one.
#[must_use]
pub fn layer_of(bank: &str) -> u64 {
    bank.rsplit_once('.')
        .and_then(|(head, _)| head.rsplit('.').next())
        .and_then(|part| part.parse::<u64>().ok())
        .unwrap_or(0)
}

fn numbered(part: &str) -> bool {
    !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())
}

// ── the host byte cache ──────────────────────────────────────────────────

/// One file's bytes, host-side, with the fingerprint §3.3 snapshots.
#[derive(Debug)]
pub struct Blob {
    /// Where it was read from.
    pub at: PathBuf,
    /// The bytes.
    pub bytes: Vec<u8>,
    /// FNV-1a over all of them — the content half of §3.3's identity,
    /// recorded on the slot that the bytes land in.
    pub fingerprint: u64,
}

enum Cell {
    /// Somebody is reading this file right now; everyone else waits.
    Loading,
    /// Somebody holds it — or held it, and the handle is gone.
    Held(Weak<Blob>),
}

/// The host byte cache: refcounted handles, one read per file per generation.
///
/// **SINGLE-FLIGHT IS THE POINT** (§3.3). Two instances binding the same
/// adapter at the same instant must perform ONE read; the second waits on the
/// first rather than doubling the disk traffic and the peak host footprint.
/// That is what [`Blobs::loads`] counts and what a gate asserts on.
///
/// **AND THE BYTES DIE WITH THE LAST HANDLE.** The map holds `Weak`s, so a
/// blob that has been landed into its slot and released costs nothing host-
/// side. The residency it left behind is the SLOT ([`Slots`]), which is what
/// §3.3 means by "device residency is a cache over the files".
#[derive(Debug, Default)]
pub struct Blobs {
    held: Mutex<HashMap<PathBuf, Cell>>,
    ready: Condvar,
    loads: AtomicU64,
}

impl std::fmt::Debug for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cell::Loading => f.write_str("Loading"),
            Cell::Held(_) => f.write_str("Held"),
        }
    }
}

impl Blobs {
    /// How many times this store has actually read a file.
    ///
    /// The single-flight observable: `n` concurrent opens of one path move it
    /// by one.
    #[must_use]
    pub fn loads(&self) -> u64 {
        self.loads.load(Ordering::Relaxed)
    }

    /// A handle on `at`'s bytes, reading them if nobody has.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a file that will not read, named by `path`.
    pub fn open(&self, at: &Path, path: &str) -> Result<Arc<Blob>> {
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            // The borrow of the map ends with this `match`, which is why the
            // upgrade's result is lifted out rather than acted on inside it.
            let seen = match held.get(at) {
                Some(Cell::Held(weak)) => Some(weak.upgrade()),
                Some(Cell::Loading) => None,
                None => Some(None),
            };
            match seen {
                Some(Some(blob)) => return Ok(blob),
                Some(None) => {
                    held.insert(at.to_path_buf(), Cell::Loading);
                    break;
                }
                None => {
                    held = self
                        .ready
                        .wait(held)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
        drop(held);
        self.loads.fetch_add(1, Ordering::Relaxed);
        let read = std::fs::read(at);
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        let out = match read {
            Ok(bytes) => {
                let blob = Arc::new(Blob {
                    at: at.to_path_buf(),
                    fingerprint: fingerprint(&bytes),
                    bytes,
                });
                held.insert(at.to_path_buf(), Cell::Held(Arc::downgrade(&blob)));
                Ok(blob)
            }
            Err(error) => {
                // The claim is dropped rather than left behind: a waiter that
                // woke onto a `Loading` nobody is in reads a stale promise.
                held.remove(at);
                Err(Fault::Blob {
                    path: path.to_string(),
                    why: format!("names `{}`, which will not read: {error}", at.display()),
                })
            }
        };
        drop(held);
        self.ready.notify_all();
        out
    }
}

/// FNV-1a, 64 bit — the content half of §3.3's identity.
///
/// Not a cryptographic digest and not asked to be one: what it settles is
/// "did the bytes behind this stamp change", against an accident and not an
/// adversary, and the mount is operator-owned.
#[must_use]
pub fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

// ── identity ─────────────────────────────────────────────────────────────

/// **THE KEY A SLOT IS HELD UNDER**, snapshotted at bind (§3.3).
///
/// The adapter's RESOLVED DIRECTORY plus `(file, len, mtime)` for its
/// manifest and every plane it names. A rewritten plane is a different stamp
/// and therefore a different slot, which is how "no in-flight fire ever
/// observes an adapter changing" is obtained without a single lock on the
/// fire path.
///
/// **THE RESOLVED PATH AND NOT THE SPELLING**, because §3.3's identity is the
/// FILE's: `/alice-v2` and `alice-v2` are one adapter, and two instances that
/// spelled it differently must share one slot or the sharing claim is about
/// strings instead of about bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Stamp {
    /// Where it resolved to under the mount.
    pub at: String,
    /// `(file, bytes, mtime nanoseconds)`, manifest first, then the planes in
    /// the order the manifest names them.
    pub files: Vec<(String, u64, u128)>,
}

/// What a bind names, and therefore what its slot is keyed by.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    /// A file in the mount. N instances naming it share ONE slot.
    Shared(Stamp),
    /// An instance's own bytes. Never shared — content-hash dedup across
    /// byte-seeded instances is §3.3's explicit "later optimization, not v1".
    Own(u64),
}

fn stat(at: &Path, file: &str, name: &str) -> Result<(String, u64, u128)> {
    let meta = std::fs::metadata(at).map_err(|error| Fault::Blob {
        path: name.to_string(),
        why: format!("names `{file}`, which is not in the directory: {error}"),
    })?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|at| at.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_nanos());
    Ok((file.to_string(), meta.len(), mtime))
}

// ── the residency ────────────────────────────────────────────────────────

/// Who is in a slot, how many binds hold it, and when it was last wanted.
#[derive(Debug, Clone)]
pub struct Occupant {
    /// What it is holding.
    pub key: Key,
    /// How many live binds name it. Zero is RECLAIMABLE and not reclaimed:
    /// §3.3's "eviction is LRU under pressure, not eager", so intermittent
    /// traffic on one adapter does not re-pay its H2D each time.
    pub refs: u32,
    /// The tick of the last acquire or release — the LRU order.
    pub used: u64,
    /// The content fingerprint of what landed, or zero for a byte-seeded
    /// slot. §3.3's snapshot, kept where a gate can read it.
    pub fingerprint: u64,
}

/// The bank's slots, as a host table.
///
/// **THIS IS WHERE SHARING IS DECIDED, AND IT IS PURE.** No device call, no
/// bytes: a key in, a slot number out, and the answer to "must the caller
/// land anything". Which is why A-3 — N instances of one blob occupy one slot
/// — is assertable with no GPU in the machine.
#[derive(Debug)]
pub struct Slots {
    seats: u32,
    occupants: Vec<Option<Occupant>>,
    clock: u64,
}

/// What [`Slots::acquire`] answers: the slot, and whether the caller owes it
/// a landing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grant {
    /// Which slot. This is §6.4's "the slot id is the engine's bind-time
    /// answer".
    pub slot: u32,
    /// `true` when this bind is the one that must write the bytes; `false`
    /// when the slot already holds this exact identity.
    pub fresh: bool,
}

impl Slots {
    /// A table of `seats` slots, all empty.
    #[must_use]
    pub fn new(seats: u32) -> Slots {
        Slots {
            seats,
            occupants: vec![None; seats as usize],
            clock: 0,
        }
    }

    /// How many slots the banks seat.
    #[must_use]
    pub fn seats(&self) -> u32 {
        self.seats
    }

    /// Every occupied slot, for a gate to read.
    #[must_use]
    pub fn resident(&self) -> Vec<(u32, &Occupant)> {
        self.occupants
            .iter()
            .enumerate()
            .filter_map(|(at, held)| held.as_ref().map(|held| (at as u32, held)))
            .collect()
    }

    /// How many live binds hold `slot`.
    #[must_use]
    pub fn refs(&self, slot: u32) -> u32 {
        self.occupants
            .get(slot as usize)
            .and_then(|held| held.as_ref().map(|held| held.refs))
            .unwrap_or(0)
    }

    /// Take a reference on the slot `key` belongs in, seating it if it is not
    /// seated.
    ///
    /// The order is: the same identity (share it), then an empty slot, then
    /// the least-recently-used slot NO BIND HOLDS. There is no fourth arm —
    /// evicting a slot a live fire routes to is the one thing this table must
    /// never do, so it refuses instead (§5).
    ///
    /// # Errors
    ///
    /// [`Fault::AdapterSlots`] when every slot is pinned; the same fault with
    /// zero seats for a load whose model text declares no bank at all.
    pub fn acquire(&mut self, key: Key) -> Result<Grant> {
        self.clock += 1;
        let clock = self.clock;
        if let Some(at) = self
            .occupants
            .iter()
            .position(|held| held.as_ref().is_some_and(|held| held.key == key))
        {
            let held = self.occupants[at]
                .as_mut()
                .expect("the slot the search just matched on");
            held.refs += 1;
            held.used = clock;
            return Ok(Grant {
                slot: at as u32,
                fresh: false,
            });
        }
        let free = self.occupants.iter().position(Option::is_none);
        let at = match free {
            Some(at) => at,
            None => self
                .occupants
                .iter()
                .enumerate()
                .filter(|(_, held)| held.as_ref().is_some_and(|held| held.refs == 0))
                .min_by_key(|(_, held)| held.as_ref().map_or(0, |held| held.used))
                .map(|(at, _)| at)
                .ok_or(Fault::AdapterSlots { seats: self.seats })?,
        };
        self.occupants[at] = Some(Occupant {
            key,
            refs: 1,
            used: clock,
            fingerprint: 0,
        });
        Ok(Grant {
            slot: at as u32,
            fresh: true,
        })
    }

    /// Record what landed in `slot`.
    pub fn stamp(&mut self, slot: u32, fingerprint: u64) {
        if let Some(Some(held)) = self.occupants.get_mut(slot as usize) {
            held.fingerprint = fingerprint;
        }
    }

    /// Drop one reference. At zero the slot becomes reclaimable and KEEPS its
    /// contents, so the next bind of the same identity is a hit.
    pub fn release(&mut self, slot: u32) {
        self.clock += 1;
        let clock = self.clock;
        if let Some(Some(held)) = self.occupants.get_mut(slot as usize) {
            held.refs = held.refs.saturating_sub(1);
            held.used = clock;
        }
    }

    /// Give a slot back UNSEATED — a landing that refused holds nothing.
    ///
    /// Distinct from [`Slots::release`] on purpose: a released slot keeps its
    /// contents and its key because they are true, and a slot whose landing
    /// failed has neither.
    pub fn abandon(&mut self, slot: u32) {
        if let Some(held) = self.occupants.get_mut(slot as usize) {
            *held = None;
        }
    }
}

// ── the store ────────────────────────────────────────────────────────────

/// What a bind names.
#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    /// A directory in the mount. Keyed by its stamp, so every instance that
    /// names it lands on one slot and pays one H2D.
    Shared {
        /// The adapter's name, as the guest spells it.
        name: &'a str,
    },
    /// An instance's own full-capacity planes — the private-adapter path of
    /// §3.3, and the shape [`crate::weights::Weights::register_adapter`]
    /// already took (§6.2).
    Own {
        /// Which instance. Its slot is its own and is never shared.
        instance: u64,
        /// The planes, exactly as the existing verb takes them.
        planes: &'a [AdapterPlane<'a>],
    },
}

/// What a bind answers (§6.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    /// The slot every lane of this instance routes to — what the runtime
    /// stamps onto `Lane::adapter`.
    pub slot: u32,
    /// Did this bind name a file in the mount?
    pub shared: bool,
    /// Did THIS bind pay the landing, or did it join one already resident?
    ///
    /// The A-3 observable: the second instance of one blob answers `false`
    /// with the first one's slot.
    pub landed: bool,
}

/// The mount, the host cache and the residency, together — the whole store.
#[derive(Debug)]
pub struct Adapters {
    vfs: Vfs,
    blobs: Blobs,
    slots: Slots,
}

impl Adapters {
    /// A store over `seats` slots, mounted nowhere.
    #[must_use]
    pub fn new(seats: u32) -> Adapters {
        Adapters {
            vfs: Vfs::default(),
            blobs: Blobs::default(),
            slots: Slots::new(seats),
        }
    }

    /// State where the shared adapters live.
    ///
    /// **A VERB AND NOT A `getenv`** (design article 9). It is a verb rather
    /// than a load field because §3.3's hot-add is a file drop: the mount is
    /// a deployment fact that outlives any one load, and a directory that
    /// grows an adapter while the box serves needs no restart and no second
    /// word here.
    pub fn mount(&mut self, root: Option<PathBuf>) {
        self.vfs = Vfs::new(root);
    }

    /// The mount, for a caller that wants to say where it is.
    #[must_use]
    pub fn vfs(&self) -> &Vfs {
        &self.vfs
    }

    /// The host byte cache — [`Blobs::loads`] is the single-flight observable.
    #[must_use]
    pub fn blobs(&self) -> &Blobs {
        &self.blobs
    }

    /// The residency table.
    #[must_use]
    pub fn slots(&self) -> &Slots {
        &self.slots
    }

    /// **BIND ONE INSTANCE TO ONE ADAPTER, LANDING ITS BYTES AT MOST ONCE.**
    ///
    /// The whole of §6.1's ruling in one function: the identity is computed
    /// from the files, the slot is keyed by it, and `land` — which is
    /// `register_adapter`, the verb §6.2 says already fits — runs only for
    /// the bind that seated the slot. N instances of one blob call this N
    /// times and the device sees one copy.
    ///
    /// `land` is a parameter rather than a field because the residency this
    /// table decides is a pure host question and the write is a device one:
    /// with the two apart, every claim about sharing, eviction and exhaustion
    /// is checkable on a machine with no GPU in it.
    ///
    /// **A REFUSED LANDING HOLDS NO SLOT.** Anything that goes wrong after
    /// the slot is seated abandons it, so no key is ever left pointing at
    /// bytes that did not arrive.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for every way the mount, the manifest or the shapes
    /// can disagree; [`Fault::AdapterSlots`] when every slot is pinned;
    /// whatever `land` answers.
    pub fn bind<L>(&mut self, source: Source<'_>, seats: &[BankSeat], land: L) -> Result<Binding>
    where
        L: FnOnce(u32, &[AdapterPlane<'_>]) -> Result<()>,
    {
        let shared = matches!(source, Source::Shared { .. });
        // **THE UNKNOWN PATH REFUSES BEFORE A SLOT IS TOUCHED** (§5): the key
        // cannot be formed without the files, so a name that is not there
        // never reaches the residency table at all.
        let key = self.key(&source)?;
        let grant = self.slots.acquire(key)?;
        if !grant.fresh {
            return Ok(Binding {
                slot: grant.slot,
                shared,
                landed: false,
            });
        }
        let landed = match source {
            Source::Own { planes, .. } => land(grant.slot, planes).map(|()| 0),
            Source::Shared { name } => self.land_shared(name, seats, grant.slot, land),
        };
        match landed {
            Ok(fingerprint) => {
                self.slots.stamp(grant.slot, fingerprint);
                Ok(Binding {
                    slot: grant.slot,
                    shared,
                    landed: true,
                })
            }
            Err(fault) => {
                self.slots.abandon(grant.slot);
                Err(fault)
            }
        }
    }

    /// Give a bind back. The slot keeps what it holds (§3.3: eviction is
    /// under pressure, not eager), so re-binding the same adapter later is a
    /// hit rather than a second H2D.
    pub fn release(&mut self, binding: Binding) {
        self.slots.release(binding.slot);
    }

    /// The key `source` belongs under — the moment the files are stat'ed.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a name outside the mount, a directory with no
    /// manifest, or a manifest naming a file that is not there.
    pub fn key(&self, source: &Source<'_>) -> Result<Key> {
        match source {
            Source::Own { instance, .. } => Ok(Key::Own(*instance)),
            Source::Shared { name } => {
                let dir = self.vfs.resolve(name)?;
                let manifest = Manifest::read(&dir, name)?;
                let mut files = Vec::with_capacity(manifest.planes.len() + 1);
                files.push(stat(&dir.join(MANIFEST), MANIFEST, name)?);
                for plane in &manifest.planes {
                    files.push(stat(&dir.join(&plane.file), &plane.file, name)?);
                }
                Ok(Key::Shared(Stamp {
                    at: dir.display().to_string(),
                    files,
                }))
            }
        }
    }

    /// Slice a shared adapter's files into one plane per bank and land them.
    /// Answers the fingerprint of the last plane read, which is what the
    /// occupant records.
    fn land_shared<L>(
        &self,
        name: &str,
        seats: &[BankSeat],
        slot: u32,
        land: L,
    ) -> Result<u64>
    where
        L: FnOnce(u32, &[AdapterPlane<'_>]) -> Result<()>,
    {
        let built = self.planes(name, seats)?;
        let fingerprint = built.1;
        let planes: Vec<AdapterPlane<'_>> = built
            .0
            .iter()
            .map(|(bank, bytes)| AdapterPlane {
                bank: bank.as_str(),
                bytes,
            })
            .collect();
        land(slot, &planes)?;
        Ok(fingerprint)
    }

    /// **THE RESOLVER** (§6.3): one shared adapter's files, sliced per layer
    /// and padded per orientation into one full-capacity plane per bank.
    ///
    /// Public because it is the whole host-side arithmetic of the landing —
    /// the slicing, the statute check and both refusals — and a gate that can
    /// call it needs no device to judge any of them.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a role this load declares no bank for, banks of one
    /// role that do not agree on a slot, a source whose orientation the bank's
    /// is not, a rank past the bank's, or a file whose length is not exactly
    /// `layers x rank x hidden` elements.
    pub fn planes(
        &self,
        name: &str,
        seats: &[BankSeat],
    ) -> Result<(Vec<(String, Vec<u8>)>, u64)> {
        let dir = self.vfs.resolve(name)?;
        let manifest = Manifest::read(&dir, name)?;
        let refuse = |why: String| Fault::Blob {
            path: name.to_string(),
            why,
        };
        let mut out = Vec::new();
        let mut fingerprint = 0u64;
        for spec in &manifest.planes {
            let mut banks: Vec<&BankSeat> = seats
                .iter()
                .filter(|seat| role_of(&seat.name) == spec.role)
                .collect();
            banks.sort_by_key(|seat| layer_of(&seat.name));
            let seat = *banks.first().ok_or_else(|| {
                refuse(format!(
                    "declares a plane for role `{}` and this load declares no bank by \
                     that name; its banks are {:?}",
                    spec.role,
                    seats.iter().map(|seat| &seat.name).collect::<Vec<_>>()
                ))
            })?;
            if let Some(odd) = banks
                .iter()
                .find(|bank| bank.slot != seat.slot || bank.rows != seat.rows)
            {
                return Err(refuse(format!(
                    "fills role `{}` across {} banks and they are not one shape: `{}` \
                     seats {} bytes and `{}` seats {}; a `[layers, ...]` source is L \
                     contiguous slices of ONE rectangle",
                    spec.role,
                    banks.len(),
                    seat.name,
                    seat.slot,
                    odd.name,
                    odd.slot
                )));
            }
            // ── §6.3's STATUTE. The bank's own rectangle says which way it
            //    runs and the file says which way it was written; a source
            //    that disagrees would need a transpose this shell does not
            //    ship, so it is refused rather than silently mis-strided.
            let bank_layout = Layout::of_bank(seat);
            if bank_layout != spec.layout {
                return Err(refuse(format!(
                    "declares plane `{}` {} and bank `{}` seats [{}, {}], which is {}; \
                     landing one as the other is a transpose, and a repack kernel is \
                     exactly what the out-major statute exists to avoid — so it is \
                     refused rather than repacked",
                    spec.role,
                    spec.layout.spelled(),
                    seat.name,
                    seat.rows,
                    seat.cols,
                    bank_layout.spelled()
                )));
            }
            let bank_rank = seat.rows.min(seat.cols);
            let hidden = seat.rows.max(seat.cols);
            if manifest.rank > bank_rank {
                return Err(refuse(format!(
                    "is rank {} and bank `{}` seats rank {}; the bank's capacity is a \
                     shape the model text declared, so the fix is a bank that seats it \
                     and not a retry",
                    manifest.rank, seat.name, bank_rank
                )));
            }
            let blob = self.blobs.open(&dir.join(&spec.file), name)?;
            // **FOLDED, NOT XORED.** Two planes of one adapter can carry the
            // same bytes — a zero `A` beside a zero `B` is the identity
            // adapter every gate starts from — and an exclusive-or of two
            // equal fingerprints is zero, which would record "no content" for
            // the one case a reader most wants named. The mix is FNV's own
            // step, which is what produced the halves.
            fingerprint = (fingerprint ^ blob.fingerprint).wrapping_mul(0x0000_0100_0000_01b3);
            let stride = manifest
                .rank
                .saturating_mul(hidden)
                .saturating_mul(seat.elem);
            let want = stride.saturating_mul(banks.len() as u64);
            if blob.bytes.len() as u64 != want {
                return Err(refuse(format!(
                    "carries {} bytes in `{}` and the {} banks of role `{}` want {} — \
                     {} layers x rank {} x {} at {} bytes an element",
                    blob.bytes.len(),
                    spec.file,
                    banks.len(),
                    spec.role,
                    want,
                    banks.len(),
                    manifest.rank,
                    hidden,
                    seat.elem
                )));
            }
            let stride = usize::try_from(stride).unwrap_or(usize::MAX);
            let slot = usize::try_from(seat.slot).unwrap_or(usize::MAX);
            for (layer, bank) in banks.iter().enumerate() {
                let source = &blob.bytes[layer * stride..(layer + 1) * stride];
                // **ZERO-PADDED HERE, AND PER ORIENTATION** — which is why the
                // resolver has to know the layout and why `AdapterPlane`'s
                // doc says the caller pads: `A`'s unused ranks are trailing
                // ROWS and `B`'s are a stride inside every row. A zero row of
                // `A` contributes zero to the waist and a zero column of `B`
                // contributes zero to the sum, so the padding is exact.
                let mut plane = vec![0u8; slot];
                match spec.layout {
                    Layout::RankMajor => plane[..source.len()].copy_from_slice(source),
                    Layout::OutMajor => {
                        let from = usize::try_from(manifest.rank.saturating_mul(seat.elem))
                            .unwrap_or(usize::MAX);
                        let to = usize::try_from(bank_rank.saturating_mul(seat.elem))
                            .unwrap_or(usize::MAX);
                        for row in 0..usize::try_from(hidden).unwrap_or(0) {
                            plane[row * to..row * to + from]
                                .copy_from_slice(&source[row * from..(row + 1) * from]);
                        }
                    }
                }
                out.push((bank.name.clone(), plane));
            }
        }
        Ok((out, fingerprint))
    }
}
