//! The shared adapter store: [`Vfs`] resolves adapter names to
//! directories, [`Blobs`] caches host bytes, [`Slots`] tracks device
//! residency by content identity — N instances of one adapter share one slot.

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
/// `None` for a root means the feature is off: every `open` refuses.
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
    /// A leading `/` is cut; every component must be plain (no `..`, no
    /// root, no prefix) — traversal is refused, not sanitized.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for an unmounted shell, an empty name, a bad
    /// component, or a missing directory.
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

/// Which way a source plane is laid out, as stated by the file.
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

    /// Which orientation a bank's rectangle carries: the rank axis is the
    /// short one. A square bank is degenerate, taken as rank-major.
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
    /// Which correction site's banks it fills; `None` selects the banks
    /// declaring no site. An unknown spelling is refused at [`Manifest::read`].
    pub site: Option<Site>,
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
    /// not TOML, or one that omits a key.
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
                // Absent site keeps the banks that declare none; an unknown
                // spelling is refused rather than silently ignored.
                let site = match plane.get("site") {
                    None => None,
                    Some(value) => {
                        let word = value.as_str().ok_or_else(|| {
                            refuse(format!(
                                "declares plane `{role}`'s `site` as something that is \
                                 not a string; a site is one word of {}",
                                Site::vocabulary()
                            ))
                        })?;
                        Some(Site::parse(word).ok_or_else(|| {
                            refuse(format!(
                                "declares plane `{role}` at site `{word}`, and the \
                                 correction sites a bank can be named at are {}; a site \
                                 nobody can name is refused rather than landed at \
                                 whatever site the model text happens to correct",
                                Site::vocabulary()
                            ))
                        })?)
                    }
                };
                Ok(PlaneSpec {
                    role,
                    file,
                    layout,
                    site,
                })
            })
            .collect::<Result<Vec<PlaneSpec>>>()?;
        Ok(Manifest { rank, planes })
    }
}

// ── the name convention ──────────────────────────────────────────────────

/// Which projection a bank corrects — six llama-like sites, mirrored by
/// the guest surface's own `Site::bit()`. Spelling is duplicated across
/// three crates since they can't depend on each other in a circle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The mixer's output projection — what an untagged bank means.
    O,
    /// The fused gate/up projection of the feed-forward sublayer.
    GateUp,
    /// Its down projection.
    Down,
}

impl Site {
    /// The vocabulary, in bit order.
    pub const ALL: [Site; 6] = [
        Site::Q,
        Site::K,
        Site::V,
        Site::O,
        Site::GateUp,
        Site::Down,
    ];

    /// How a bank name and a manifest spell it.
    #[must_use]
    pub const fn spelled(self) -> &'static str {
        match self {
            Site::Q => "q",
            Site::K => "k",
            Site::V => "v",
            Site::O => "o",
            Site::GateUp => "gate_up",
            Site::Down => "down",
        }
    }

    /// The guest surface's own bit for it, riding the `lora` sink's placement
    /// constant.
    #[must_use]
    pub const fn bit(self) -> u32 {
        match self {
            Site::Q => 1 << 0,
            Site::K => 1 << 1,
            Site::V => 1 << 2,
            Site::O => 1 << 3,
            Site::GateUp => 1 << 4,
            Site::Down => 1 << 5,
        }
    }

    /// A spelling, or `None` for a word outside the vocabulary. An unknown
    /// middle segment is not a site, so the whole name becomes the role.
    #[must_use]
    pub fn parse(word: &str) -> Option<Site> {
        Site::ALL.into_iter().find(|site| site.spelled() == word)
    }

    /// The vocabulary as a message names it.
    #[must_use]
    pub fn vocabulary() -> String {
        Site::ALL
            .iter()
            .map(|site| format!("`{}`", site.spelled()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// How a message spells "at this site", including the absent one.
    #[must_use]
    pub fn stated(site: Option<Site>) -> String {
        match site {
            Some(site) => format!("at site `{}`", site.spelled()),
            None => "at no stated site".to_string(),
        }
    }
}

/// The bank name's role — everything after an optional `layer.{l}.`
/// prefix and an optional site segment. A name with no numbered
/// component is its own role at layer zero.
#[must_use]
pub fn role_of(bank: &str) -> &str {
    parsed(bank).map_or(bank, |(_, _, role)| role)
}

/// Which layer a bank name puts itself at, or zero for an unnumbered one.
#[must_use]
pub fn layer_of(bank: &str) -> u64 {
    parsed(bank).map_or(0, |(layer, _, _)| layer)
}

/// Which site a bank declares it corrects. `None` means the text's own
/// default site, not "no site" — a load whose banks all answer `None`
/// behaves as it always did.
#[must_use]
pub fn site_of(bank: &str) -> Option<Site> {
    parsed(bank).and_then(|(_, site, _)| site)
}

/// `layer.{l}[.{site}].{role}`, read once — the whole grammar in one place.
fn parsed(bank: &str) -> Option<(u64, Option<Site>, &str)> {
    let (head, role) = bank.rsplit_once('.')?;
    let last = head.rsplit('.').next()?;
    // `layer.{l}.{role}`, no site.
    if let Some(layer) = numbered(last) {
        return Some((layer, None, role));
    }
    // `layer.{l}.{site}.{role}`.
    let site = Site::parse(last)?;
    let (rest, _) = head.rsplit_once('.')?;
    let layer = numbered(rest.rsplit('.').next()?)?;
    Some((layer, Some(site), role))
}

fn numbered(part: &str) -> Option<u64> {
    match !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()) {
        true => part.parse().ok(),
        false => None,
    }
}

// ── the host byte cache ──────────────────────────────────────────────────

/// One file's bytes, host-side, with its content fingerprint.
#[derive(Debug)]
pub struct Blob {
    /// Where it was read from.
    pub at: PathBuf,
    /// The bytes.
    pub bytes: Vec<u8>,
    /// FNV-1a over all of them, recorded on the slot the bytes land in.
    pub fingerprint: u64,
}

enum Cell {
    /// Somebody is reading this file right now; everyone else waits.
    Loading,
    /// Somebody holds it — or held it, and the handle is gone.
    Held(Weak<Blob>),
}

/// The host byte cache: refcounted handles, one read per file per
/// generation. Single-flight: concurrent opens of one path wait on the
/// first read. `Weak` handles let bytes die with the last one; residency
/// after that lives in [`Slots`].
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
    /// How many times this store has actually read a file. `n` concurrent
    /// opens of one path move it by one.
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
                // Drop the claim so a waiter doesn't wake onto a stale promise.
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

/// FNV-1a, 64 bit. Not cryptographic; only defends against accidental
/// change, not an adversary — the mount is operator-owned.
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

/// The key a slot is held under, snapshotted at bind: the adapter's
/// resolved directory plus `(file, len, mtime)` for its manifest and
/// every plane. A rewritten file is a different stamp and a different slot.
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
    /// An instance's own bytes. Never shared.
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
    /// How many live binds name it. Zero is reclaimable but not reclaimed
    /// eagerly (LRU under pressure only).
    pub refs: u32,
    /// The tick of the last acquire or release — the LRU order.
    pub used: u64,
    /// The content fingerprint of what landed, or zero for a byte-seeded slot.
    pub fingerprint: u64,
}

/// The bank's slots, as a host table. Pure: a key in, a slot number and
/// whether the caller must land anything out. No device call, no bytes.
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
    /// Which slot.
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

    /// Take a reference on the slot `key` belongs in, seating it if not
    /// seated. Order: same identity, then an empty slot, then
    /// least-recently-used unheld. Never evicts a slot a live fire routes to.
    ///
    /// # Errors
    ///
    /// [`Fault::AdapterSlots`] when every slot is pinned (or zero seats
    /// for a load with no bank at all).
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

    /// Give a slot back unseated — a landing that failed holds nothing.
    /// Distinct from [`Slots::release`], which keeps the slot's contents.
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
    /// An instance's own full-capacity planes.
    Own {
        /// Which instance. Its slot is its own and is never shared.
        instance: u64,
        /// The planes, exactly as the existing verb takes them.
        planes: &'a [AdapterPlane<'a>],
    },
}

/// What a bind answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    /// The slot every lane of this instance routes to.
    pub slot: u32,
    /// Did this bind name a file in the mount?
    pub shared: bool,
    /// Did this bind pay the landing, or join one already resident?
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

    /// State where the shared adapters live. A verb rather than a load field:
    /// the mount is a deployment fact that outlives any one load.
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

    /// Bind one instance to one adapter, landing its bytes at most once:
    /// the identity is computed from the files, the slot keyed by it, and
    /// `land` runs only for the bind that seats the slot. A failed landing
    /// abandons its slot.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for the mount, manifest, or shape disagreeing;
    /// [`Fault::AdapterSlots`] when every slot is pinned; whatever `land` answers.
    pub fn bind<L>(&mut self, source: Source<'_>, seats: &[BankSeat], land: L) -> Result<Binding>
    where
        L: FnOnce(u32, &[AdapterPlane<'_>]) -> Result<()>,
    {
        let shared = matches!(source, Source::Shared { .. });
        // An unknown name never reaches the residency table: the key can't
        // be formed without the files.
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

    /// Give a bind back. The slot keeps what it holds, so re-binding the
    /// same adapter later is a hit rather than a second H2D.
    pub fn release(&mut self, binding: Binding) {
        self.slots.release(binding.slot);
    }

    /// The key `source` belongs under — the moment the files are stat'ed.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a name outside the mount, a missing manifest,
    /// or a manifest naming a missing file.
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

    /// One shared adapter's files, sliced per layer and padded per
    /// orientation into one full-capacity plane per bank. Public so a
    /// gate can call it with no device.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a role with no declared bank, banks of one
    /// role disagreeing on shape, a mismatched orientation, a rank past
    /// the bank's, or a wrong file length.
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
            // Role and site together pick the banks; no site takes the
            // banks that declare none.
            let mut banks: Vec<&BankSeat> = seats
                .iter()
                .filter(|seat| role_of(&seat.name) == spec.role && site_of(&seat.name) == spec.site)
                .collect();
            banks.sort_by_key(|seat| layer_of(&seat.name));
            let seat = *banks.first().ok_or_else(|| {
                refuse(format!(
                    "declares a plane for role `{}` {} and this load declares no bank by \
                     that name; its banks are {:?}",
                    spec.role,
                    Site::stated(spec.site),
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
            // A layout mismatch would need a transpose this shell doesn't
            // ship, so it's refused rather than silently mis-strided.
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
            // Folded (not xored): two equal planes would xor to zero and
            // read as "no content".
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
                // Zero-padded per orientation: A's unused ranks are trailing
                // rows, B's are a stride inside every row.
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
