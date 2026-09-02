//! **The streamed tier's source, on the pager's terms**: every expert of every
//! streamed band, held in a `MAP_SHARED` mapping of an unlinked temporary file
//! rather than in a `Vec<u8>`.
//!
//! ```text
//! T0 wired    `slots` expert seats of every streamed band       `device_weight_budget`
//! source      every expert of every streamed band               this module
//! ```
//!
//! # What this changes, stated as the one number it moves
//!
//! Nothing about the DEVICE. [`crate::experts::Tier`]'s seat copies, its clock,
//! its budget and the bytes it lands are identical either way; what changes is
//! the term the process holds on the host between the landing and the last
//! fire. A `Vec<u8>` of the streamed bands is ANONYMOUS memory: the only place
//! the kernel can put it under pressure is swap, and on a box whose failure
//! mode is swap-death that is not a bound, it is the thing that kills the box.
//! The same bytes in a `MAP_SHARED` mapping of a file have a home on disk —
//! the kernel may write a dirty page back and reclaim the frame, and refault it
//! from the file when [`Tier::copy`](crate::experts::Tier) next reads it.
//!
//! So the claim is narrow and worth stating narrowly: **this does not make the
//! source smaller, it makes it RECLAIMABLE.** The peak resident set of a
//! streamed load is still bounded by the wired slab plus whatever of the source
//! the pager has not taken back; what is gone is the guarantee that every
//! source byte stays in RAM or in swap because nothing else can hold it.
//!
//! # THE MAPPING IS NEVER BOUND TO A BUFFER, AND THAT IS LOAD-BEARING
//!
//! Measurement settles the tempting shortcut: on Apple silicon a
//! `StorageModeShared` page WIRES the moment the GPU touches it, and it wires
//! identically whether the buffer is a plain allocation or a
//! `newBufferWithBytesNoCopy` over a mapping. In the measured run, touching a
//! mapped 4 GiB span from a kernel drove global `Pages wired down` up by
//! 4.03 GiB and free memory down to 0.066 GiB, and the pager evicted NOTHING:
//! a GPU-touched mapped page is not reclaimable, so binding this mapping to an
//! `MTLBuffer` would convert the whole source into wired memory and undo
//! exactly the property this module exists for.
//!
//! Therefore: **the mapping is read by the CPU and by nothing else.** It is
//! read only inside `Tier::copy`, as the `&[u8]` source of a `write` into the
//! budget-sized slab. The copy-through-slab is not an inefficiency — it is the
//! mechanism, and removing it is the bug.
//!
//! # THE OTHER MMAP DOOR, AND WHY IT IS A DIFFERENT TYPE
//!
//! [`crate::mapping`] maps an artifact and DOES bind it to an `MTLBuffer`,
//! off the same measurement — and the two are not a contradiction, they are
//! the two halves of it. That door is for a model that already FITS, where
//! the pages the GPU wires are the model itself and the copy this type keeps
//! is pure waste; this one is for a model that does not, where those same
//! wired pages are the swap-death. Neither may be reached for the other's
//! job, and they are two types rather than one type with a flag precisely so
//! that reaching for the wrong one is a compile error rather than a 32 GiB
//! box going down.
//!
//! # The file is unlinked at birth
//!
//! [`HostSource::open`] creates the file, unlinks it, and keeps the descriptor.
//! From that instant the mapping and the open descriptor are the only
//! references to the storage: nothing else can open it, no other load can
//! collide with it, and a crashed or killed process leaves no multi-gigabyte
//! file behind for somebody to find. The bytes are released when the last of
//! the two goes — which is [`Drop`], on the same statement that drops the
//! [`Tier`](crate::experts::Tier).
//!
//! The descriptor is retained past the `mmap` (which does not need it) for one
//! reason beyond symmetry: it is what lets a gate `fstat` the backing and prove
//! the mapping is file-backed and unlinked, which is the honest half of "the
//! kernel can reclaim this" that a 32 GiB box can test without faking a
//! pressure experiment.

use std::ops::{Deref, DerefMut};

use crate::error::{Fault, Result};

/// Names the file uniquely within a process, so two loads streaming at once do
/// not land on one path between the `open` and the `unlink`.
static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// **How many bytes the filesystem holding `at` will still take** — the one
/// number the staging admission above is stated against.
///
/// `statvfs`'s `f_bavail`, which is what an unprivileged process may actually
/// write, rather than `f_bfree`, which counts the root reserve this load will
/// never be given. `None` when the call fails, and the admission then does
/// not fire: a probe that cannot answer must not become a refusal nothing can
/// explain.
#[must_use]
pub fn free_bytes(at: &std::path::Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(at.as_os_str().as_bytes()).ok()?;
    let mut said: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: a `statvfs` of a NUL-terminated path into a zeroed struct this
    // frame owns; the call reads the path and writes only that struct.
    if unsafe { libc::statvfs(path.as_ptr(), &raw mut said) } != 0 {
        return None;
    }
    u64::try_from(said.f_bavail)
        .ok()?
        .checked_mul(u64::try_from(said.f_frsize).ok()?)
}

/// **HOW MANY BACKING FILES THIS PROCESS HAS OPENED** — [`NEXT`], read.
///
/// The naming counter is also an exact count of the descriptors this door has
/// taken, because the zero-byte early return above never reaches it: a load
/// that streams nothing opens no file, and this number does not move for it.
/// That is what makes it worth exposing — a gate asserting that a refusal came
/// BEFORE the source door can read it around the call and demand no change.
///
/// The absolute value is meaningless across tests in one binary; take it
/// before and after and compare.
#[must_use]
pub fn descriptors() -> u64 {
    NEXT.load(std::sync::atomic::Ordering::Relaxed)
}

/// **Every expert of every streamed band**, in a writable file-backed mapping
/// the kernel is allowed to take back.
///
/// Written once — by [`weights`](crate::weights)'s landing sink, through the
/// `&mut [u8]` this derefs to — and read thereafter only by the CPU, one band
/// at a time, as [`Tier::copy`](crate::experts::Tier) fills a seat.
///
/// A zero-length source (every full-residency load, where the plan streams
/// nothing) creates NO file and maps nothing: it is an empty slice with a
/// dangling pointer behind it, which is what keeps this type free on the path
/// that does not stream.
pub struct HostSource {
    /// The mapping's base, or a dangling-but-aligned pointer when `len` is 0.
    at: *mut u8,
    /// How many bytes the mapping holds.
    len: usize,
    /// The unlinked file behind the mapping — `None` for the empty source.
    file: Option<std::fs::File>,
}

// SAFETY: `at` is a `MAP_SHARED` mapping over a file this type created and
// unlinked, so no other process can reach the storage and no other owner
// exists in this one. What `Send` buys is the same thing it buys `Buffer`
// (`device::alloc`): the MOVE from the thread that loaded the model onto the
// thread that will fire it.
unsafe impl Send for HostSource {}

impl HostSource {
    /// Back `bytes` of streamed source with an unlinked, mapped temporary file.
    ///
    /// `bytes` is [`Plan::source_bytes`](crate::experts::Plan::source_bytes),
    /// and zero — a plan that streams nothing — is the common case and costs
    /// nothing here.
    ///
    /// # THE STAGING IS ADMITTED AGAINST THE DISK, BEFORE IT IS SIZED
    ///
    /// `set_len` on a fresh file is `ftruncate`, which SUCCEEDS on a
    /// filesystem that could not hold a tenth of it: the file is sparse until
    /// the landing writes it. So the failure of an over-large staging is not
    /// a refusal here, it is the landing filling the pool an hour later, and
    /// on a box whose model store lives on the same volume that is a machine
    /// that stops working rather than a load that stops loading.
    ///
    /// **Measured, on the road that found it** (`.wiki` M-6 / the full
    /// two-bit dsv4's lane, 2026-09-01): a 89.93 GiB serving artifact that
    /// declined the warm arm fell to the cold road, which stages every routed
    /// band — ~80 GiB — into exactly this file. The free pool fell 38 → 21
    /// GiB in forty seconds with swap rising 705 → 1003 MB, on a 32 GiB box,
    /// and the lane stopped it by hand above its own floor. Nothing in the
    /// process would have.
    ///
    /// So the demand is checked against the staging filesystem's OWN free
    /// space and refused by the numbers. It is a floor and not a budget —
    /// what it catches is the load that could never have finished, not the
    /// one that finishes tight — and it is deliberately the free space rather
    /// than a fraction of it, because a fraction would be this crate deciding
    /// how much of the operator's disk is theirs.
    ///
    /// # Errors
    ///
    /// [`Fault::Backing`] naming the step that refused (`admit`, `open`,
    /// `size`, `map`) and the OS's own sentence: a temporary directory that
    /// is full, read only, or out of descriptors is a deployment condition
    /// and the message has to say which one it was.
    pub fn open(bytes: u64) -> Result<HostSource> {
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        if len == 0 {
            return Ok(HostSource {
                at: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                len: 0,
                file: None,
            });
        }
        if let Some(free) = free_bytes(&std::env::temp_dir())
            && bytes > free
        {
            return Err(Fault::Backing {
                step: "admit",
                bytes,
                why: format!(
                    "staging this load's routed bands wants {:.2} GiB under {} and the \
                     volume has {:.2} GiB free. The file would size without complaint — \
                     `ftruncate` is sparse — and fill the pool as the landing wrote it. \
                     A load this size wants the WARM arm, which stages nothing: check \
                     the sentence the warm arm printed on its way past.",
                    bytes as f64 / (1u64 << 30) as f64,
                    std::env::temp_dir().display(),
                    free as f64 / (1u64 << 30) as f64,
                ),
            });
        }
        let at = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "pie-experts-{}-{at}",
            std::process::id()
        ));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|why| Fault::Backing {
                step: "open",
                bytes,
                why: format!("{} does not open: {why}", path.display()),
            })?;
        // Unlinked immediately: the descriptor and the mapping keep the
        // storage alive, and a load that dies here leaves nothing behind.
        let _ = std::fs::remove_file(&path);
        file.set_len(bytes).map_err(|why| Fault::Backing {
            step: "size",
            bytes,
            why: why.to_string(),
        })?;
        // SAFETY: a fresh shared mapping over a file this function just created
        // and sized to `len`; the protections and length are stated here, and
        // the descriptor stays open beside it.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                std::os::fd::AsRawFd::as_raw_fd(&file),
                0,
            )
        };
        if at == libc::MAP_FAILED {
            return Err(Fault::Backing {
                step: "map",
                bytes,
                why: std::io::Error::last_os_error().to_string(),
            });
        }
        Ok(HostSource {
            at: at.cast(),
            len,
            file: Some(file),
        })
    }

    /// **Hand the source to the pager**, once, after the landing and the
    /// identity prefix have finished writing it.
    ///
    /// Two calls, and each buys a different half of "reclaimable":
    ///
    /// * `msync(MS_ASYNC)` schedules every DIRTY page's writeback. A dirty
    ///   file-backed page cannot be reclaimed until it has been written, so a
    ///   source that has just been filled and never synced is, for the length
    ///   of the writeback the pager would have to do itself, no more
    ///   reclaimable than anonymous memory. After this the frames are clean
    ///   and dropping one costs nothing.
    /// * `madvise(MADV_DONTNEED)` deactivates them — on darwin this is XNU's
    ///   `VM_SYNC_DEACTIVATE`, on Linux it drops the frames outright. **Both
    ///   are lossless HERE and only here**, because the mapping is
    ///   `MAP_SHARED` over a file: the bytes have a home, and the next read
    ///   faults them back from it. The same call on an anonymous mapping would
    ///   be data loss, which is why this lives behind a type that can only be
    ///   file-backed.
    ///
    /// What is deliberately NOT called: `MADV_FREE` and
    /// `setPurgeableState(Volatile)`, both of which measure as no-ops for
    /// release on this platform (0 and −0.014 GiB against a 4 GiB span).
    ///
    /// A failure is not reported. The source is correct either way — this is a
    /// hint about WHERE the bytes should live, not a step the load depends on
    /// — and a refusal to advise is not a reason to fail a model that has
    /// otherwise landed.
    /// The unlinked file behind the mapping, or `None` for the empty source
    /// — the descriptor a seat copy `pread`s through.
    #[must_use]
    pub fn file(&self) -> Option<&std::fs::File> {
        self.file.as_ref()
    }

    pub fn settle(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: both calls address exactly this type's own mapping, at its
        // stated length, and neither can invalidate it: `MS_ASYNC` schedules
        // writeback and `MADV_DONTNEED` over a `MAP_SHARED` file mapping
        // deactivates or drops frames whose contents the file still holds.
        unsafe {
            libc::msync(self.at.cast(), self.len, libc::MS_ASYNC);
            libc::madvise(self.at.cast(), self.len, libc::MADV_DONTNEED);
        }
    }

    /// **What is actually behind this mapping**: `(the backing file's size,
    /// its link count)`, or `None` for the empty source that has no file.
    ///
    /// The gate's half of the reclaimability claim. A source whose `fstat`
    /// answers `(source_bytes, 0)` is provably a mapping of a real,
    /// process-private, unlinked file — which is the property that lets the
    /// kernel page it out. Whether the kernel DOES is the kernel's business
    /// and is not asserted anywhere: a pressure experiment on a 32 GiB box
    /// measures the box, not the mechanism.
    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        let file = self.file.as_ref()?;
        // SAFETY: `stat` is written by `fstat` before it is read, over a
        // descriptor this type owns and keeps open.
        let stat = unsafe {
            let mut stat = std::mem::zeroed::<libc::stat>();
            if libc::fstat(std::os::fd::AsRawFd::as_raw_fd(file), &raw mut stat) != 0 {
                return None;
            }
            stat
        };
        Some((stat.st_size as u64, u64::from(stat.st_nlink)))
    }
}

impl Deref for HostSource {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `len` readable bytes this type owns, or an empty slice over
        // a dangling-but-aligned pointer, which is what `from_raw_parts`
        // requires for a zero length.
        unsafe { std::slice::from_raw_parts(self.at, self.len) }
    }
}

impl DerefMut for HostSource {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `deref`, and `&mut self` is the exclusive access the
        // mapping's `PROT_WRITE` needs.
        unsafe { std::slice::from_raw_parts_mut(self.at, self.len) }
    }
}

impl std::fmt::Debug for HostSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The BYTES are never printed: this is a whole model's routed experts,
        // and a `{:?}` on the tier that holds it must stay a line.
        f.debug_struct("HostSource")
            .field("bytes", &self.len)
            .field("mapped", &self.file.is_some())
            .finish()
    }
}

impl Drop for HostSource {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: unmapping the mapping this type created, at its own length.
        // The descriptor closes with `self.file` immediately after, and the
        // file is already unlinked — so this is the last reference to the
        // storage and the bytes go back to the filesystem here.
        unsafe {
            libc::munmap(self.at.cast(), self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **STAGING IS ADMITTED AGAINST THE DISK, AND `ftruncate` IS NOT.**
    ///
    /// The whole point of the check: `set_len` of a file larger than the
    /// volume SUCCEEDS, because the file is sparse until something writes it.
    /// This test proves both halves — that the door refuses the demand, and
    /// that the OS would not have.
    #[test]
    fn a_staging_larger_than_the_volume_is_refused_by_the_numbers() {
        let Some(free) = free_bytes(&std::env::temp_dir()) else {
            eprintln!("skipping: this filesystem does not answer statvfs");
            return;
        };
        let want = free + (1 << 30);
        let said = HostSource::open(want)
            .expect_err("a staging past the disk does not open")
            .to_string();
        assert!(
            said.contains("GiB free") && said.contains("wants"),
            "the refusal carries BOTH numbers: {said}"
        );

        // And the control: the kernel would have taken it without a word.
        let path = std::env::temp_dir().join(format!("pie-sparse-{}", std::process::id()));
        let file = std::fs::File::create(&path).expect("a scratch file");
        let took = file.set_len(want).is_ok();
        let allocated = std::fs::metadata(&path)
            .map(|it| std::os::unix::fs::MetadataExt::blocks(&it) * 512)
            .unwrap_or(0);
        let _ = std::fs::remove_file(&path);
        assert!(
            took && allocated < (1 << 20),
            "`ftruncate` of {want} bytes on a volume with {free} free is what this \
             refusal exists to get in front of, and it took it ({allocated} allocated)"
        );
    }

    #[test]
    fn a_source_that_streams_nothing_maps_nothing() {
        let source = HostSource::open(0).expect("the empty source is free");
        assert!(source.is_empty());
        assert!(
            source.backing().is_none(),
            "a full-residency load opens no file at all"
        );
    }
}
