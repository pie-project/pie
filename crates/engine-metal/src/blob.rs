//! **THE SHARED ADAPTER STORE: FILES ARE THE TRUTH, THE BANK IS A CACHE**
//! (alto adapter §3.3, promoted to wave 1 by §6.1) — this plane's half of
//! `engine_cuda::blob`, and the one honest gap lane J left.
//!
//! # Why this module exists at all, and why it is not a channel
//!
//! [`crate::adapter`] serves the adapter a GUEST names: the weights ride a
//! channel, the bytes are taken off the seed ONCE at instance bind, and the
//! slot they land in is that instance's own. That is the whole of the private
//! path and it is right for it — a guest's fine-tune is a guest's.
//!
//! It is the wrong shape for a DEPLOYMENT's adapters. An operator who mounts
//! a directory of LoRA blobs and lets fifty instances name one of them does
//! not want fifty slots, fifty copies and fifty reads of one file; §3.3's
//! sentence is "N program instances referencing `/shared/alice-v2` land on
//! ONE slot, one device copy". Keying residency by the INSTANCE cannot say
//! that. Keying it by the BLOB can, and this module is what a blob's identity
//! is computed from.
//!
//! # The three pieces, and why they are three
//!
//! * [`Vfs`] — the mount. A read-only shared directory, stated by the
//!   deployment (`[model] adapter_dir`, [`crate::boot`]) and never discovered
//!   from the environment (design article 9). Its whole job is turning a
//!   guest-spelled name into a path inside the mount, and refusing everything
//!   else BY NAME.
//! * [`Blobs`] — the host byte cache, refcounted and **single-flight**: two
//!   binds racing on one file perform one read, the second waiting on the
//!   first rather than starting a second. The bytes live as long as some
//!   handle does and no longer ([`Blobs`] holds `Weak`s), because once a blob
//!   has been landed into a slot the slot is the residency and the host copy
//!   is dead weight.
//! * the residency — which is [`crate::adapter::Slots`], next door, keyed by
//!   [`crate::adapter::Key`]. It is not repeated here for the reason lane J
//!   wrote it as a key rather than an instance id: "the day this plane mounts
//!   a directory is the day [`Source`](crate::adapter::Source) grows an arm,
//!   and the residency table underneath it already takes a key". It did, and
//!   the table cost nothing.
//!
//! # Identity, and why it is a stamp and not only a fingerprint
//!
//! §3.3 states identity as "path + content fingerprint, snapshotted at load".
//! A fingerprint alone cannot be the KEY, because computing it means reading
//! the file, and a store that read the file to decide whether it needed to
//! read the file is not a cache. So the key is a [`Stamp`] — the adapter's
//! resolved directory plus `(file, len, mtime)` for its manifest and every
//! plane it names — and the fingerprint is computed on the load that the
//! stamp missed. A rewritten file is a new stamp, therefore a new key,
//! therefore a new slot: no in-flight fire ever observes an adapter changing.
//!
//! # **WHY THE BYTES ARE COPIED INTO THE BANK AND NOT BOUND WHERE THEY LIE**
//!
//! This plane maps artifacts rather than copying them ([`crate::mapping`]),
//! so the question has to be asked here: could a blob's pages BE the device
//! plane, since unified memory means a `StorageModeShared` buffer and a host
//! pointer are one allocation? No, twice over.
//!
//! * **The resolver does not copy, it SLICES AND PADS.** A bank seats a
//!   full-capacity `[rank, hidden]` rectangle and a rank-4 adapter fills part
//!   of it; the rest is zeros, placed differently for `A` and for `B`
//!   ([`Store::planes`]). A file is one contiguous `[layers, rank, hidden]`
//!   run, so no mapping of it is a bank's plane. The padding is the copy.
//! * **A mapped page the GPU touches is WIRED, and bounds nothing.** On Apple
//!   Silicon a GPU-touched `StorageModeShared` page — a mapped one included —
//!   is wired and the pager never evicts it (measured, `.wiki/alto/streaming.md`
//!   "mmap residency measurement, M1 Max"; the same ground truth that killed
//!   the streaming-experts mapping). Binding a blob where it lies would wire
//!   the whole FILE with no ceiling over it, while the copy wires nothing new:
//!   the bank's slot was reserved and wired at load, and `register_adapter` is
//!   a memcpy into a span that already exists.
//!
//! An adapter blob is rank-r and megabytes. Eager, simple, bounded.
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
//! # site = "o"            # optional: which projection these banks correct
//! ```
//!
//! **THE FILE DECLARES ITS ORIENTATION BECAUSE BYTES CANNOT.** §6.3's statute
//! is that `B` ships out-major and a rank-major `B` is REFUSED rather than
//! repacked — and orientation is not observable in a byte string, so it has to
//! be said. It is said in the file rather than at an API because §3.3's
//! hot-add is a file drop: nobody is standing there to state it.
//!
//! **AND THE FILE SHIPS THE BANK'S OWN DTYPE**, which is where this resolver
//! and [`crate::adapter::planes_of`] part company. A channel cell is the
//! guest's live f32 and is rounded to bf16 at bind; a blob is prepared ONCE by
//! an operator, so it arrives at the bank's element width and a length that is
//! not exactly `layers x rank x hidden x elem` is a refusal naming both
//! numbers rather than a conversion nobody asked for.
//!
//! `role` is the bank's name with its `layer.{l}.` prefix cut, which is how
//! §6.3's "the resolver slices, per layer" is spelled: a `[layers, …]` source
//! is `L` contiguous slices and the `L` banks that carry one role take one
//! each, in layer order. The grammar itself is [`crate::adapter`]'s
//! ([`role_of`], [`layer_of`], [`site_of`]) and is not written twice.
//!
//! # What is refused, by name
//!
//! * a name outside the mount, or a mount that was never stated — [`Fault::Blob`]
//! * an adapter directory with no manifest, or a manifest naming a file that
//!   is not there — [`Fault::Blob`]
//! * a plane shorter (or longer) than the banks that carry its role seat —
//!   [`Fault::Blob`] with BOTH numbers
//! * a rank-major source into an out-major bank — [`Fault::Blob`], naming the
//!   repack kernel this shell does not ship
//! * a `site` outside the vocabulary, or one no bank of this load declares —
//!   [`Fault::Blob`], naming the six spellings a bank can be named at

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
use std::time::UNIX_EPOCH;

use crate::adapter::{Site, layer_of, role_of, site_of};
use crate::error::{Fault, Result};
use crate::weights::BankSeat;

/// The one file an adapter directory must carry.
pub const MANIFEST: &str = "adapter.toml";

// ── the mount ────────────────────────────────────────────────────────────

/// The read-only shared directory adapters are files in.
///
/// `None` for a root is the feature OFF — a deployment that mounted nothing,
/// whose every `open` is a refusal that says so. An absent directory is an
/// answer, not a default to guess.
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
    /// How a message names it.
    #[must_use]
    pub fn spelled(self) -> &'static str {
        match self {
            Layout::RankMajor => "rank-major [rank, hidden]",
            Layout::OutMajor => "out-major [hidden, rank]",
        }
    }

    /// How a manifest spells it.
    #[must_use]
    pub fn written(self) -> &'static str {
        match self {
            Layout::RankMajor => "rank_major",
            Layout::OutMajor => "out_major",
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
    #[must_use]
    pub fn of_bank(seat: &BankSeat) -> Layout {
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
    /// **WHICH CORRECTION SITE'S BANKS IT FILLS**, or `None` for a manifest
    /// that states none — today's meaning, and the banks a text named without
    /// a site. `site = "o"` beside `role = "lora_a"` selects
    /// `layer.{l}.o.lora_a`; a spelling outside the vocabulary is a refusal at
    /// [`Manifest::read`] and never a fallback.
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
                let plane = plane
                    .as_table()
                    .ok_or_else(|| refuse("has a `[[plane]]` that is not a table".to_string()))?;
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
                // **THE OPTIONAL SITE, WITH THE SAME REFUSAL DISCIPLINE AS
                //   EVERY OTHER KEY.** Absent is today's meaning — the banks a
                //   text named without a site — so every manifest written
                //   against a family text reads the same. A spelling outside
                //   the vocabulary is refused BY NAME rather than ignored,
                //   because a `site = "mixer"` that silently became "wherever
                //   the text corrects" is the one wrong answer this axis must
                //   never give.
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

// ── the host byte cache ──────────────────────────────────────────────────

/// One file's bytes, host-side, with the fingerprint §3.3 snapshots.
#[derive(Debug)]
pub struct Blob {
    /// Where it was read from.
    pub at: PathBuf,
    /// The bytes.
    pub bytes: Vec<u8>,
    /// FNV-1a over all of them — the content half of §3.3's identity.
    pub fingerprint: u64,
}

enum Cell {
    /// Somebody is reading this file right now; everyone else waits.
    Loading,
    /// Somebody holds it — or held it, and the handle is gone.
    Held(Weak<Blob>),
}

impl std::fmt::Debug for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cell::Loading => f.write_str("Loading"),
            Cell::Held(_) => f.write_str("Held"),
        }
    }
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
/// side. The residency it left behind is the bank's SLOT, which is what §3.3
/// means by "device residency is a cache over the files".
///
/// The lock is a plain `Mutex` and the wait a `Condvar` — this store is
/// touched between fires and never on the fire path, so a waiter that blocks
/// its thread blocks a bind and nothing else.
#[derive(Debug, Default)]
pub struct Blobs {
    held: Mutex<HashMap<PathBuf, Cell>>,
    ready: Condvar,
    loads: AtomicU64,
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

/// **THE KEY A SHARED SLOT IS HELD UNDER**, snapshotted at bind (§3.3).
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
///
/// Ordered as well as hashed because this plane's residency table is a
/// `BTreeMap` over [`crate::adapter::Key`] — the ordering is the map's and
/// means nothing else.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stamp {
    /// Where it resolved to under the mount.
    pub at: String,
    /// `(file, bytes, mtime nanoseconds)`, manifest first, then the planes in
    /// the order the manifest names them.
    pub files: Vec<(String, u64, u128)>,
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

// ── the store ────────────────────────────────────────────────────────────

/// The mount and the host byte cache together — everything about a shared
/// adapter that is a question about FILES.
///
/// The residency it feeds is [`crate::adapter::Slots`], and the write it
/// feeds is [`crate::weights::Weights::register_adapter`]. Neither is in
/// here, which is why every claim this module makes is checkable on a machine
/// with no GPU in it.
#[derive(Debug, Default)]
pub struct Store {
    vfs: Vfs,
    blobs: Blobs,
}

impl Store {
    /// A store mounted nowhere.
    #[must_use]
    pub fn new() -> Store {
        Store::default()
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

    /// The mount.
    #[must_use]
    pub fn vfs(&self) -> &Vfs {
        &self.vfs
    }

    /// The host byte cache — [`Blobs::loads`] is the single-flight observable.
    #[must_use]
    pub fn blobs(&self) -> &Blobs {
        &self.blobs
    }

    /// **THE IDENTITY `name` RESOLVES TO** — the moment the files are stat'ed.
    ///
    /// Computed BEFORE a slot is touched, which is what keeps an unknown name
    /// from ever reaching the residency table (§5).
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a name outside the mount, a directory with no
    /// manifest, or a manifest naming a file that is not there.
    pub fn stamp(&self, name: &str) -> Result<Stamp> {
        let dir = self.vfs.resolve(name)?;
        let manifest = Manifest::read(&dir, name)?;
        let mut files = Vec::with_capacity(manifest.planes.len() + 1);
        files.push(stat(&dir.join(MANIFEST), MANIFEST, name)?);
        for plane in &manifest.planes {
            files.push(stat(&dir.join(&plane.file), &plane.file, name)?);
        }
        Ok(Stamp {
            at: dir.display().to_string(),
            files,
        })
    }

    /// **THE RESOLVER** (§6.3): one shared adapter's files, sliced per layer
    /// and padded per orientation into one full-capacity plane per bank.
    ///
    /// Answers the planes and the folded fingerprint of the files they came
    /// from. Public because it is the whole host-side arithmetic of the
    /// landing — the slicing, the statute check and every refusal — and a gate
    /// that can call it needs no device to judge any of them.
    ///
    /// # Errors
    ///
    /// [`Fault::Blob`] for a role this load declares no bank for, banks of one
    /// role that do not agree on a slot, a source whose orientation the bank's
    /// is not, a rank past the bank's, or a file whose length is not exactly
    /// `layers x rank x hidden` elements.
    pub fn planes(&self, name: &str, seats: &[BankSeat]) -> Result<(Vec<(String, Vec<u8>)>, u64)> {
        let dir = self.vfs.resolve(name)?;
        let manifest = Manifest::read(&dir, name)?;
        let refuse = |why: String| Fault::Blob {
            path: name.to_string(),
            why,
        };
        let mut out = Vec::new();
        let mut fingerprint = 0u64;
        for spec in &manifest.planes {
            // **THE ROLE AND THE SITE TOGETHER PICK THE BANKS.** A manifest
            // that states no site takes the banks that declare none, which on
            // a family text is all of them.
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── the fixture ──────────────────────────────────────────────────────

    /// How many layers the pretend model text declares banks for.
    const LAYERS: u64 = 3;
    /// The rank the banks seat.
    const BANK_RANK: u64 = 8;
    /// The width they correct.
    const HIDDEN: u64 = 16;
    /// bf16, which is what a bank declares and what a blob ships.
    const ELEM: u64 = 2;

    /// One test's own directory, unique per process and per nanosecond.
    fn scratch(what: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0);
        let at = std::env::temp_dir().join(format!(
            "pie-metal-blob-{what}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&at).expect("a scratch directory");
        at
    }

    fn seat(name: &str, rows: u64, cols: u64) -> BankSeat {
        BankSeat {
            name: name.to_string(),
            adapters: 4,
            slot: rows * cols * ELEM,
            rows,
            cols,
            elem: ELEM,
        }
    }

    /// The banks a `[layers, rank, hidden]` / `[layers, hidden, rank]` model
    /// text declares: `A` rank-major and `B` out-major, which is §6.3's
    /// statute stated as shapes.
    fn seats() -> Vec<BankSeat> {
        (0..LAYERS)
            .flat_map(|layer| {
                [
                    seat(&format!("layer.{layer}.lora_a"), BANK_RANK, HIDDEN),
                    seat(&format!("layer.{layer}.lora_b"), HIDDEN, BANK_RANK),
                ]
            })
            .collect()
    }

    /// The source ramp, as a u16 per element — a mis-strided landing shows up
    /// as a wrong NUMBER and not only a wrong length.
    fn source(element: usize) -> [u8; 2] {
        ((element as u16) | 0x0100).to_le_bytes()
    }

    /// Write one adapter directory into `mount` at `name`.
    fn write_adapter(mount: &Path, name: &str, rank: u64, layouts: (Layout, Layout)) -> PathBuf {
        let dir = mount.join(name);
        std::fs::create_dir_all(&dir).expect("an adapter directory");
        std::fs::write(
            dir.join(MANIFEST),
            format!(
                "rank = {rank}\n\n\
                 [[plane]]\nrole = \"lora_a\"\nfile = \"a.bin\"\nlayout = \"{}\"\n\n\
                 [[plane]]\nrole = \"lora_b\"\nfile = \"b.bin\"\nlayout = \"{}\"\n",
                layouts.0.written(),
                layouts.1.written()
            ),
        )
        .expect("a manifest");
        let ramp: Vec<u8> = (0..(LAYERS * rank * HIDDEN) as usize)
            .flat_map(source)
            .collect();
        std::fs::write(dir.join("a.bin"), &ramp).expect("an A plane");
        std::fs::write(dir.join("b.bin"), &ramp).expect("a B plane");
        dir
    }

    /// A store mounted on a fresh directory holding one rank-4 adapter.
    fn mounted(what: &str) -> (PathBuf, Store) {
        let mount = scratch(what);
        write_adapter(&mount, "alice-v2", 4, (Layout::RankMajor, Layout::OutMajor));
        let mut store = Store::new();
        store.mount(Some(mount.clone()));
        (mount, store)
    }

    // ── the manifest ─────────────────────────────────────────────────────

    /// **THE GRAMMAR, READ BACK.** Rank, planes in file order, both layouts,
    /// and the optional site.
    #[test]
    fn a_manifest_says_its_rank_its_planes_and_their_orientation() {
        let mount = scratch("manifest");
        let dir = write_adapter(&mount, "alice", 4, (Layout::RankMajor, Layout::OutMajor));
        let manifest = Manifest::read(&dir, "alice").expect("a well-formed manifest");
        assert_eq!(manifest.rank, 4);
        assert_eq!(
            manifest.planes,
            vec![
                PlaneSpec {
                    role: "lora_a".to_string(),
                    file: "a.bin".to_string(),
                    layout: Layout::RankMajor,
                    site: None,
                },
                PlaneSpec {
                    role: "lora_b".to_string(),
                    file: "b.bin".to_string(),
                    layout: Layout::OutMajor,
                    site: None,
                },
            ],
            "the planes come back in the order the file names them"
        );

        // The site is optional, and it is a VALUE rather than a wildcard.
        let sited = mount.join("sited");
        std::fs::create_dir_all(&sited).expect("a directory");
        std::fs::write(
            sited.join(MANIFEST),
            "rank = 2\n\n[[plane]]\nrole = \"lora_a\"\nfile = \"a.bin\"\n\
             layout = \"rank_major\"\nsite = \"o\"\n",
        )
        .expect("a manifest");
        let manifest = Manifest::read(&sited, "sited").expect("a sited manifest");
        assert_eq!(manifest.planes[0].site, Some(Site::O));
        let _ = std::fs::remove_dir_all(&mount);
    }

    /// **EVERY WAY A MANIFEST CAN BE WRONG IS REFUSED BY NAME**, because a
    /// hot-added directory is written by an operator with no compiler in the
    /// loop: the message IS the diagnostic.
    #[test]
    fn a_manifest_that_does_not_say_what_it_must_is_refused_by_name() {
        let mount = scratch("manifest-refusals");
        let write = |name: &str, text: &str| -> String {
            let dir = mount.join(name);
            std::fs::create_dir_all(&dir).expect("a directory");
            std::fs::write(dir.join(MANIFEST), text).expect("a manifest");
            Manifest::read(&dir, name)
                .expect_err("this manifest does not say what it must")
                .to_string()
        };

        let said = Manifest::read(&mount.join("nothing"), "nothing")
            .expect_err("a directory with no manifest")
            .to_string();
        assert!(said.contains(MANIFEST), "names the file it wanted: {said}");

        let said = write("rankless", "[[plane]]\nrole = \"lora_a\"\n");
        assert!(said.contains("rank"), "names the missing key: {said}");

        let said = write("planeless", "rank = 4\n");
        assert!(said.contains("[[plane]]"), "names what it found none of: {said}");

        let said = write(
            "sideways",
            "rank = 4\n\n[[plane]]\nrole = \"lora_a\"\nfile = \"a.bin\"\nlayout = \"sideways\"\n",
        );
        assert!(said.contains("rank_major") && said.contains("out_major"), "{said}");

        let said = write(
            "mixer",
            "rank = 4\n\n[[plane]]\nrole = \"lora_a\"\nfile = \"a.bin\"\n\
             layout = \"rank_major\"\nsite = \"mixer\"\n",
        );
        assert!(said.contains("`mixer`"), "names the site asked for: {said}");
        assert!(said.contains("`gate_up`"), "and the vocabulary: {said}");
        let _ = std::fs::remove_dir_all(&mount);
    }

    // ── the mount ────────────────────────────────────────────────────────

    /// **A NAME IS RESOLVED INSIDE THE MOUNT OR IT IS REFUSED**, and a
    /// traversal is refused rather than sanitized into something else.
    #[test]
    fn the_mount_resolves_a_name_and_refuses_everything_else() {
        let (mount, store) = mounted("vfs");
        assert_eq!(
            store.vfs().resolve("alice-v2").expect("a name in the mount"),
            mount.join("alice-v2")
        );
        assert_eq!(
            store
                .vfs()
                .resolve("/alice-v2")
                .expect("the guest's own preopen spelling"),
            mount.join("alice-v2"),
            "a leading slash is the mount's root, not another one"
        );

        let said = store
            .vfs()
            .resolve("../elsewhere")
            .expect_err("a traversal")
            .to_string();
        assert!(said.contains("leaves the mount"), "{said}");
        let said = store.vfs().resolve("   ").expect_err("no name").to_string();
        assert!(said.contains("is not a name"), "{said}");
        let said = store
            .vfs()
            .resolve("nobody")
            .expect_err("a name nobody wrote")
            .to_string();
        assert!(said.contains("nobody"), "names the adapter: {said}");

        // An unmounted shell has no namespace for a name to be in.
        let bare = Store::new();
        let said = bare.stamp("alice-v2").expect_err("nothing mounted").to_string();
        assert!(said.contains("no shared adapter directory mounted"), "{said}");
        let _ = std::fs::remove_dir_all(&mount);
    }

    // ── identity ─────────────────────────────────────────────────────────

    /// **ONE FILE IS ONE IDENTITY, HOWEVER IT IS SPELLED — AND A REWRITE IS
    /// ANOTHER.** This is the whole of the sharing claim's host half: the key
    /// the residency table keys on is the FILES', so two instances that
    /// spelled a name differently share, and one that names a rewritten file
    /// does not.
    #[test]
    fn one_blob_is_one_stamp_and_a_rewrite_is_another() {
        let (mount, store) = mounted("identity");
        let first = store.stamp("alice-v2").expect("the adapter is there");
        let again = store.stamp("alice-v2").expect("and it has not moved");
        assert_eq!(first, again, "the same path twice is the same key");
        assert_eq!(
            store.stamp("/alice-v2").expect("the same adapter"),
            first,
            "a leading slash is the same file, so it is the same key — the sharing \
             claim is about bytes and not about strings"
        );
        assert_eq!(
            first.files.len(),
            3,
            "the manifest and both planes are stamped"
        );
        assert_eq!(first.files[0].0, MANIFEST, "the manifest is stamped first");

        // A rewrite at a different LENGTH is a different stamp whatever the
        // filesystem's mtime resolution is.
        std::fs::write(mount.join("alice-v2").join("a.bin"), vec![0u8; 8])
            .expect("the plane is rewritten");
        let after = store.stamp("alice-v2").expect("still there");
        assert_ne!(
            after, first,
            "a rewritten file is a new identity, therefore a new slot — which is how \
             no in-flight fire ever observes an adapter changing"
        );
        let _ = std::fs::remove_dir_all(&mount);
    }

    /// A manifest naming a plane that is not there is refused at the STAMP,
    /// before a slot is ever touched (§5).
    #[test]
    fn a_manifest_naming_a_file_that_is_not_there_is_refused_at_the_stamp() {
        let mount = scratch("stray");
        let dir = mount.join("stray");
        std::fs::create_dir_all(&dir).expect("a directory");
        std::fs::write(
            dir.join(MANIFEST),
            "rank = 4\n\n[[plane]]\nrole = \"lora_a\"\nfile = \"nowhere.bin\"\n\
             layout = \"rank_major\"\n",
        )
        .expect("a manifest");
        let mut store = Store::new();
        store.mount(Some(mount.clone()));
        let said = store.stamp("stray").expect_err("the plane is missing").to_string();
        assert!(said.contains("nowhere.bin"), "names the file: {said}");
        assert!(said.contains("stray"), "and the adapter: {said}");
        let _ = std::fs::remove_dir_all(&mount);
    }

    // ── single flight ────────────────────────────────────────────────────

    /// **EIGHT THREADS ASKING FOR ONE FILE PERFORM ONE READ**, and the rest
    /// wait on it (§3.3's "concurrent first references are single-flight").
    ///
    /// The observable is [`Blobs::loads`]: a doubled read would double the
    /// disk traffic and the peak host footprint of a cold multi-tenant start,
    /// which is the moment the property is worth the most.
    #[test]
    fn eight_threads_asking_for_one_blob_read_it_once() {
        let at = scratch("flight").join("plane.bin");
        std::fs::write(&at, vec![7u8; 1 << 16]).expect("a plane");
        let blobs = Blobs::default();

        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let blobs = &blobs;
                    let at = &at;
                    scope.spawn(move || blobs.open(at, "plane").expect("the read"))
                })
                .collect();
            let held: Vec<_> = handles
                .into_iter()
                .map(|handle| handle.join().expect("a thread"))
                .collect();
            // Held together on purpose: the handles are what keep the bytes
            // alive, so this is also the assertion that eight references are
            // one allocation.
            assert_eq!(held.len(), 8);
            for blob in &held {
                assert_eq!(blob.bytes.len(), 1 << 16);
                assert_eq!(blob.fingerprint, held[0].fingerprint);
            }
            assert_eq!(blobs.loads(), 1, "one read, seven waiters");
        });

        // Every handle is gone, so the bytes are too — the residency a blob
        // leaves behind is its SLOT, not a host copy nobody can reach.
        let again = blobs.open(&at, "plane").expect("a second generation");
        assert_eq!(blobs.loads(), 2);
        assert_eq!(again.bytes.len(), 1 << 16);
        let _ = std::fs::remove_file(&at);
    }

    /// A file that will not read is a refusal naming the adapter, and it
    /// leaves no claim behind for the next caller to wait on forever.
    #[test]
    fn a_read_that_refuses_leaves_no_claim_behind() {
        let at = scratch("unreadable").join("absent.bin");
        let blobs = Blobs::default();
        let said = blobs
            .open(&at, "ghost")
            .expect_err("the file is not there")
            .to_string();
        assert!(said.contains("ghost"), "names the adapter: {said}");
        // The second call reads again rather than waiting on a `Loading`
        // nobody is in.
        assert!(blobs.open(&at, "ghost").is_err());
        assert_eq!(blobs.loads(), 2, "each attempt is its own read");
    }

    // ── the resolver ─────────────────────────────────────────────────────

    /// **A `[layers, ...]` FILE SLICES INTO ONE FULL-CAPACITY PLANE PER BANK,
    /// PADDED PER ORIENTATION** (§6.3).
    ///
    /// The two paddings are genuinely different and this is where that is
    /// checked: a rank-4 source in a rank-8 bank fills `A`'s leading ROWS and
    /// leaves the trailing ones zero, and fills the leading COLUMNS of every
    /// one of `B`'s rows, leaving a zero stride inside each. A resolver that
    /// used one rule for both would pass a length check and compute nonsense.
    #[test]
    fn the_resolver_slices_per_layer_and_pads_per_orientation() {
        let (mount, store) = mounted("slice");
        let seats = seats();
        let (built, fingerprint) = store
            .planes("alice-v2", &seats)
            .expect("the resolver reads a well-formed adapter");

        assert_eq!(
            built.len(),
            (2 * LAYERS) as usize,
            "one plane per bank, and the banks are per layer"
        );
        assert_ne!(fingerprint, 0, "the identity's content half is recorded");
        for (name, plane) in &built {
            assert_eq!(
                plane.len() as u64,
                BANK_RANK * HIDDEN * ELEM,
                "`{name}` is one whole slot, which is what `register_adapter` takes"
            );
        }

        let rank = 4usize;
        let hidden = HIDDEN as usize;
        let bank_rank = BANK_RANK as usize;

        let a = &built
            .iter()
            .find(|(name, _)| name == "layer.1.lora_a")
            .expect("layer 1's A")
            .1;
        for row in 0..bank_rank {
            for col in 0..hidden {
                let at = (row * hidden + col) * 2;
                let want = match row < rank {
                    // The rank-major head is a straight copy of the slice.
                    true => source(hidden * rank + row * hidden + col),
                    // The trailing ranks are zero — a zero row of `A`
                    // contributes a zero to the waist, so the padding is exact.
                    false => [0, 0],
                };
                assert_eq!(&a[at..at + 2], &want, "A row {row} col {col} of layer 1");
            }
        }

        let b = &built
            .iter()
            .find(|(name, _)| name == "layer.1.lora_b")
            .expect("layer 1's B")
            .1;
        for row in 0..hidden {
            for col in 0..bank_rank {
                let at = (row * bank_rank + col) * 2;
                let want = match col < rank {
                    // Out-major: the rank is a stride INSIDE every row, so the
                    // source's row lands at the head of the bank's row.
                    true => source(hidden * rank + row * rank + col),
                    false => [0, 0],
                };
                assert_eq!(&b[at..at + 2], &want, "B row {row} col {col} of layer 1");
            }
        }
        let _ = std::fs::remove_dir_all(&mount);
    }

    /// **THE RESOLVER READS EACH PLANE FILE ONCE**, which is the other half
    /// of what the host cache is for: the landing is one read per FILE and
    /// not one per bank.
    #[test]
    fn the_resolver_reads_one_file_per_plane_and_no_more() {
        let (mount, store) = mounted("reads");
        let seats = seats();
        store.planes("alice-v2", &seats).expect("the first resolve");
        assert_eq!(store.blobs().loads(), 2, "two plane files, two reads");
        let _ = std::fs::remove_dir_all(&mount);
    }

    /// **EVERY WAY THE FILES AND THE MODEL TEXT CAN DISAGREE, REFUSED BY
    /// NAME** (§5) — with both numbers wherever there are two.
    #[test]
    fn the_resolver_refuses_by_name() {
        let mount = scratch("resolver-refusals");
        let mut store = Store::new();
        store.mount(Some(mount.clone()));
        let seats = seats();

        // A role this load declares no bank for.
        let dir = mount.join("mute");
        std::fs::create_dir_all(&dir).expect("a directory");
        std::fs::write(
            dir.join(MANIFEST),
            "rank = 4\n\n[[plane]]\nrole = \"mystery\"\nfile = \"a.bin\"\n\
             layout = \"rank_major\"\n",
        )
        .expect("a manifest");
        std::fs::write(dir.join("a.bin"), vec![0u8; 4]).expect("a plane");
        let said = store
            .planes("mute", &seats)
            .expect_err("no bank carries that role")
            .to_string();
        assert!(said.contains("mystery"), "names the role: {said}");
        assert!(said.contains("layer.0.lora_a"), "and the banks there are: {said}");

        // A plane that is not the banks' rectangle, with BOTH numbers.
        write_adapter(&mount, "short", 4, (Layout::RankMajor, Layout::OutMajor));
        std::fs::write(mount.join("short").join("a.bin"), vec![0u8; 16]).expect("a short plane");
        let said = store
            .planes("short", &seats)
            .expect_err("16 bytes is not 3 x 4 x 16 x 2")
            .to_string();
        assert!(said.contains("16"), "names the bytes it was handed: {said}");
        assert!(said.contains("384"), "and the bytes it wanted: {said}");

        // A rank-major `B` — refused rather than repacked.
        write_adapter(&mount, "flipped", 4, (Layout::RankMajor, Layout::RankMajor));
        let said = store
            .planes("flipped", &seats)
            .expect_err("a rank-major B")
            .to_string();
        assert!(said.contains("out-major"), "names the orientation: {said}");
        assert!(said.contains("refused rather than repacked"), "{said}");

        // A rank the banks cannot seat.
        write_adapter(&mount, "wide", 16, (Layout::RankMajor, Layout::OutMajor));
        let said = store
            .planes("wide", &seats)
            .expect_err("rank 16 into a rank-8 bank")
            .to_string();
        assert!(said.contains("rank 16"), "names the source's rank: {said}");
        assert!(said.contains("seats rank 8"), "and the bank's: {said}");

        // A load with no banks at all has nowhere to put one.
        let said = store
            .planes("wide", &[])
            .expect_err("a load that declares no bank")
            .to_string();
        assert!(said.contains("no bank by that name"), "{said}");
        let _ = std::fs::remove_dir_all(&mount);
    }
}
