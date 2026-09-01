//! **The artifact, opened and mapped** — alto streaming §0's promotion, read
//! side.
//!
//! # What this is for
//!
//! The warm-boot artifact is a snapshot of the DEVICE STORE ITSELF: every
//! dequant, cast and repack already applied, offsets identical to store
//! offsets, plane-addressable. Streaming's stated precondition is that the
//! stored weights need no load-time conversion, and that format already exists
//! — it is this file. So SSD streaming is not a format design; it is the
//! promotion of this file from boot accelerator to serving-time T2 source, and
//! this module is the door: open it, map it, resolve a plane group to a span,
//! and hand back bytes nobody copied.
//!
//! The old rule *"a streamed load skips the warm artifact"* inverts here:
//! **the artifact becomes the source.**
//!
//! # Where it stops
//!
//! At the resolver. The third pointer class of streaming §2 — an indirection
//! entry pointing at an mmap'd page that a GPU touch faults in over HMM — is
//! the NEXT wave, and it is built on top of this: `resolve` is what turns a
//! plane group into the address that class points at. Nothing here maps
//! anything to a device.
//!
//! # Why a refusal and not a miss
//!
//! [`super::restore`] treats a stale artifact as a miss, because there the cost
//! of being wrong is one re-materialization and the shell simply recomputes.
//! **This door cannot do that.** A caller that opened an artifact to SERVE
//! from it has no recipe to fall back to, so every disagreement is a
//! [`Refused`] with a name and both numbers in it. That difference is the whole
//! reason the version lives in the header separately from the magic.
//!
//! # The mapping
//!
//! `mmap(PROT_READ, MAP_PRIVATE)` over the whole file, and nothing more:
//! private so a serving process cannot write the cache it reads, read-only so
//! the pages are shareable between processes serving the same deployment, and
//! **not populated**, because laziness is the point — a page that is never
//! touched is a page that never left the SSD.

use std::fmt;
use std::fs;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use super::{ENTRY, FORMAT, Group, HEADER, Head, STRIPES, fold, index_fault, stripe_digests};

/// **Why an artifact could not be served from.**
///
/// Every variant carries what it disagreed about, because the caller of this
/// door has nothing else to go on: it asked to serve out of a file and the
/// answer is that it cannot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// The path is not a file this process can read. The filesystem's words.
    Unreadable {
        /// What the filesystem said.
        why: String,
    },
    /// The bytes are readable and are not an artifact — the magic disagrees,
    /// or there are not even enough bytes to hold a header.
    NotAnArtifact,
    /// **The version refusal.** An artifact from a build whose format is not
    /// this one. Regenerate it — [`super::restore`] does exactly that on the
    /// boot path.
    StaleFormat {
        /// What the file says its format is.
        states: u32,
        /// What this build reads.
        reads: u32,
    },
    /// The file is shorter than its own header claims.
    Truncated {
        /// The last byte the header accounts for.
        states: u64,
        /// How many bytes the file actually holds.
        holds: u64,
    },
    /// The plane-group index does not describe the blob behind it: a digest
    /// that does not match, or an entry that points outside the table.
    IndexCorrupt {
        /// Which disagreement.
        why: String,
    },
    /// The file is fine and the kernel would not map it.
    Unmappable {
        /// `mmap`'s own errno, as a sentence.
        why: String,
    },
    /// **The name and the header disagree about which deployment this is.**
    ///
    /// A file under one key's name holding another key's bytes is the case
    /// that would restore the wrong weights with every digest agreeing, so it
    /// is refused with both numbers rather than believed on either. Raised by
    /// [`tier::Artifact::open`](super::tier::Artifact::open), whose filenames
    /// carry the key; the resident door is handed its expected key by the
    /// caller and answers a MISS instead (see [`super::restore`]).
    WrongKey {
        /// The key the header states.
        states: u64,
        /// The key the filename names.
        names: u64,
    },
}

impl fmt::Display for Refused {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Unreadable { why } => write!(out, "cannot be read: {why}"),
            Refused::NotAnArtifact => write!(out, "is not a weight artifact"),
            Refused::StaleFormat { states, reads } => write!(
                out,
                "states format {states} and this build reads {reads}; regenerate it"
            ),
            Refused::Truncated { states, holds } => write!(
                out,
                "accounts for {states} bytes and holds {holds}"
            ),
            Refused::IndexCorrupt { why } => write!(out, "has an index that {why}"),
            Refused::Unmappable { why } => write!(out, "cannot be mapped: {why}"),
            Refused::WrongKey { states, names } => write!(
                out,
                "is named for key {names:016x} and states key {states:016x}"
            ),
        }
    }
}

impl std::error::Error for Refused {}

/// **Read an artifact's header and index without mapping it.**
///
/// What the restore path needs before it decides which arm to take, and what
/// [`Artifact::open`] does first. The index's digest is checked here, so a
/// caller that gets a `Vec<Group>` back has entries the file vouches for.
///
/// # Errors
///
/// Every [`Refused`] except [`Refused::Unmappable`], which only the mapping
/// can produce.
pub fn read_head(path: &Path) -> Result<(Head, Vec<Group>), Refused> {
    use std::io::Read;

    let mut file = fs::File::open(path).map_err(|why| Refused::Unreadable {
        why: format!("{why}"),
    })?;
    let holds = file
        .metadata()
        .map_err(|why| Refused::Unreadable {
            why: format!("{why}"),
        })?
        .len();

    let mut bytes = [0u8; HEADER];
    // A short read is not an artifact rather than a truncated one: a file with
    // no header has not made a claim this reader could find it short of.
    if file.read_exact(&mut bytes).is_err() {
        return Err(Refused::NotAnArtifact);
    }
    let Some(head) = Head::decode(&bytes) else {
        return Err(Refused::NotAnArtifact);
    };
    // **THE VERSION, BEFORE ANYTHING ELSE IS BELIEVED.** Every field after the
    // format word means whatever the format says it means, so a stale one is
    // refused here rather than parsed under this build's rules.
    if head.format != FORMAT {
        return Err(Refused::StaleFormat {
            states: head.format,
            reads: FORMAT,
        });
    }

    let index_bytes = u64::from(head.groups) * ENTRY as u64;
    let index_end = head.index_at.saturating_add(index_bytes);
    let blob_end = head.blob_at.saturating_add(head.total);
    if head.blob_at < index_end {
        return Err(Refused::IndexCorrupt {
            why: format!(
                "runs to byte {index_end} and the blob it precedes starts at {}",
                head.blob_at
            ),
        });
    }
    if index_end > holds || blob_end > holds {
        return Err(Refused::Truncated {
            states: blob_end.max(index_end),
            holds,
        });
    }

    let mut groups = Vec::with_capacity(head.groups as usize);
    if head.groups > 0 {
        let mut raw = vec![0u8; usize::try_from(index_bytes).unwrap_or(0)];
        read_at(&file, head.index_at, &mut raw).map_err(|why| Refused::Unreadable { why })?;
        for entry in raw.chunks_exact(ENTRY) {
            let Some(group) = Group::decode(entry) else {
                return Err(Refused::IndexCorrupt {
                    why: "does not decode".to_string(),
                });
            };
            groups.push(group);
        }
    }
    if super::index_digest(&groups) != head.index_digest {
        return Err(Refused::IndexCorrupt {
            why: "does not match its digest".to_string(),
        });
    }
    if let Some(why) = index_fault(&groups, head.total) {
        return Err(Refused::IndexCorrupt { why });
    }
    Ok((head, groups))
}

/// **An artifact, open and mapped** — the serving-time weight source.
///
/// Holds the file's whole mapping for as long as it lives, and answers three
/// questions: what this file is ([`Artifact::head`]), which plane groups it
/// holds ([`Artifact::groups`]), and where one of them lives
/// ([`Artifact::resolve`], [`Artifact::plane`]). **Nothing is copied.** A
/// `&[u8]` handed out here points into the mapping, and touching a byte of it
/// is what reads the SSD.
#[derive(Debug)]
pub struct Artifact {
    path: PathBuf,
    head: Head,
    groups: Vec<Group>,
    map: Map,
}

impl Artifact {
    /// **Open the artifact at `path` and map it.**
    ///
    /// # Errors
    ///
    /// Any [`Refused`]. In particular [`Refused::StaleFormat`] for an artifact
    /// from another build, which is a refusal here and a miss on the boot
    /// path — see this module's header.
    pub fn open(path: &Path) -> Result<Artifact, Refused> {
        let (head, groups) = read_head(path)?;
        let file = fs::File::open(path).map_err(|why| Refused::Unreadable {
            why: format!("{why}"),
        })?;
        let holds = file
            .metadata()
            .map_err(|why| Refused::Unreadable {
                why: format!("{why}"),
            })?
            .len();
        let map = Map::open(&file, usize::try_from(holds).unwrap_or(0))
            .map_err(|why| Refused::Unmappable { why })?;
        Ok(Artifact {
            path: path.to_path_buf(),
            head,
            groups,
            map,
        })
    }

    /// Where this artifact came from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Everything the file says about itself.
    #[must_use]
    pub fn head(&self) -> Head {
        self.head
    }

    /// Which deployment's weights these are — [`super::Identity::key`].
    #[must_use]
    pub fn key(&self) -> u64 {
        self.head.key
    }

    /// How many bytes the device store this file snapshots holds.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.head.total
    }

    /// **The plane-group index**, in the order it was written, which is the
    /// plan's own param order.
    #[must_use]
    pub fn groups(&self) -> &[Group] {
        &self.groups
    }

    /// **Where a plane group lives** — the resolver, and the reason this
    /// module exists.
    ///
    /// `None` for a group this artifact does not carry, which includes every
    /// group of an artifact written before the loader passed its layout across
    /// (an index of zero entries is legal — see [`super::store`]).
    #[must_use]
    pub fn resolve(&self, id: u32, plane: u32) -> Option<Group> {
        self.groups
            .iter()
            .find(|group| group.id == id && group.plane == plane)
            .copied()
    }

    /// **A plane group's bytes, borrowed from the mapping.**
    ///
    /// `bytes`, not `reserved`: the reserved span is what the DEVICE gives the
    /// plane, and the bytes past the published length are padding no reader
    /// should be handed. `None` for a group this artifact does not carry.
    ///
    /// **NOTHING IS COPIED AND NOTHING IS READ YET.** The slice is a window on
    /// the mapping; the first touch of each page is the NVMe read.
    #[must_use]
    pub fn plane(&self, id: u32, plane: u32) -> Option<&[u8]> {
        let group = self.resolve(id, plane)?;
        let at = usize::try_from(self.head.blob_at.checked_add(group.offset)?).ok()?;
        let len = usize::try_from(group.bytes).ok()?;
        self.map.bytes().get(at..at.checked_add(len)?)
    }

    /// **The whole blob** — every plane, in store order, as one slice.
    ///
    /// What the restore path pumps to the device, and what [`Artifact::verify`]
    /// hashes.
    #[must_use]
    pub fn blob(&self) -> &[u8] {
        let at = usize::try_from(self.head.blob_at).unwrap_or(usize::MAX);
        let len = usize::try_from(self.head.total).unwrap_or(0);
        self.map
            .bytes()
            .get(at..at.saturating_add(len))
            .unwrap_or(&[])
    }

    /// **Hash the blob and compare it to what the header states.**
    ///
    /// The always-verified discipline, available on this side too: dev's rule
    /// is that *a silently-corrupt weight artifact produces garbage tokens
    /// with no error*, and a serving-time source is no less exposed to that
    /// than a boot-time one. It reads the whole file, so a caller that means to
    /// serve lazily out of the mapping pays for it deliberately.
    ///
    /// # Errors
    ///
    /// [`Refused::IndexCorrupt`] naming both digests.
    pub fn verify(&self) -> Result<(), Refused> {
        // The parallel arithmetic, against a header written through the
        // streaming one. Their agreement is what this call is really asserting
        // on every artifact it is pointed at.
        let found = stripe_digests(self.blob());
        for (which, (found, stated)) in found.iter().zip(self.head.stripes.iter()).enumerate() {
            if found != stated {
                return Err(Refused::IndexCorrupt {
                    why: format!(
                        "states digest {stated:016x} for stripe {which} of {STRIPES}, \
                         whose bytes hash to {found:016x}"
                    ),
                });
            }
        }
        let folded = fold(&found);
        if folded != self.head.digest {
            return Err(Refused::IndexCorrupt {
                why: format!(
                    "states digest {:016x} for stripes that fold to {folded:016x}",
                    self.head.digest
                ),
            });
        }
        Ok(())
    }
}

/// Read `into.len()` bytes from `at`, without moving the file's cursor.
fn read_at(file: &fs::File, at: u64, into: &mut [u8]) -> Result<(), String> {
    use std::os::unix::fs::FileExt;

    file.read_exact_at(into, at).map_err(|why| format!("{why}"))
}

/// One read-only private mapping of a whole file.
///
/// `pub(super)` so that [`tier`](super::tier) maps its payload through the
/// same twelve lines rather than through a second `mmap` call somebody has to
/// keep in step with this one.
///
/// `libc` rather than a crate: this is `mmap`, `munmap` and a length, and the
/// manifest already carries `libc` for the one other question the standard
/// library cannot answer (`statvfs`, for the declined write).
#[derive(Debug)]
pub(super) struct Map {
    base: *mut core::ffi::c_void,
    len: usize,
}

// SAFETY: a `Map` is an address and a length over a PROT_READ, MAP_PRIVATE
// mapping. Nothing can write it through this handle, so sharing it across
// threads hands out immutable bytes; the mapping outlives every borrow because
// `bytes` borrows from `&self`.
unsafe impl Send for Map {}
// SAFETY: as `Send`.
unsafe impl Sync for Map {}

impl Map {
    /// Map the whole file.
    pub(super) fn open(file: &fs::File, len: usize) -> Result<Map, String> {
        if len == 0 {
            return Ok(Map {
                base: core::ptr::null_mut(),
                len: 0,
            });
        }
        // SAFETY: a live descriptor, a length the file's own metadata gave,
        // and a kernel-chosen address. The mapping is this structure's and is
        // unmapped exactly once in `Drop`.
        let base = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(format!("{}", std::io::Error::last_os_error()));
        }
        Ok(Map { base, len })
    }

    /// The mapped bytes.
    pub(super) fn bytes(&self) -> &[u8] {
        if self.base.is_null() || self.len == 0 {
            return &[];
        }
        // SAFETY: the mapping is `self.len` readable bytes for as long as
        // `self` lives, and the borrow is tied to `&self`.
        unsafe { core::slice::from_raw_parts(self.base.cast::<u8>(), self.len) }
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        if !self.base.is_null() && self.len > 0 {
            // SAFETY: the address and length are this structure's own
            // mapping, unmapped exactly once.
            unsafe {
                libc::munmap(self.base, self.len);
            }
        }
    }
}
