//! **The artifact, mapped once and bound without a copy** — the warm-read
//! plan's M-1, and the one primitive under it.
//!
//! A load that fits in memory spends its whole boot doing one thing: reading
//! every weight byte off disk into a `Vec`, and then copying that `Vec` into
//! a `StorageModeShared` `MTLBuffer` whose pages the GPU is about to wire
//! anyway. On unified memory the second half of that is pure waste — the
//! device buffer IS host memory, and the bytes were already in host memory
//! when the pager faulted them out of the file. This module removes the
//! copy: it maps the artifact and hands Metal the mapping's own pages
//! through `newBufferWithBytesNoCopy:length:options:deallocator:`, so the
//! bytes are faulted from the file exactly once, into the frames the kernel
//! will read.
//!
//! # THIS PRIMITIVE WIRES EVERY PAGE IT TOUCHES, AND THAT IS THE DEAL
//!
//! The measurement in `.wiki/alto/streaming.md` ("mmap residency
//! measurement, M1 Max", 2026-08-31) is the ground truth and it is blunt: on
//! Apple silicon a `StorageModeShared` page WIRES the instant the GPU
//! touches it, and it wires identically whether the buffer is a plain
//! allocation or a `newBufferWithBytesNoCopy` over a mapping. Touching a
//! mapped 4 GiB span from a kernel drove global `Pages wired down` up by
//! **+4.03 GiB** and free memory down to **0.066 GiB**, and the pager
//! evicted NOTHING. A GPU-touched mapped page is not reclaimable.
//!
//! So this type bounds no memory whatsoever, and two rules fall out of that:
//!
//! * **It is for a model that already fits.** Wired-equals-the-model is not
//!   a cost this primitive adds — a full-residency load's eager buffer wires
//!   the same bytes, measured (the control row: +4.027 GiB). What the
//!   mapping removes is the DUPLICATE: the transient `Vec` and the `memcpy`,
//!   and with them the peak of double the model's size during boot. That is
//!   the whole claim, and it is a claim about the peak and the boot clock,
//!   never about the floor.
//! * **It must NEVER carry the streamed or oversized path.** Binding an
//!   over-budget artifact through here converts the entire touched span into
//!   wired memory that nothing will take back, which is precisely the
//!   swap-death this crate's streaming tier exists to avoid.
//!   [`crate::host_source`] is that tier's source and its module header
//!   states the same measurement from the other side: it maps and is read by
//!   the CPU ONLY, and binding its mapping to a buffer would undo the one
//!   property it exists for. The two mmap doors in this crate are deliberate
//!   opposites — this one binds because the model fits, that one refuses to
//!   bind because it does not — and neither may be reached for the other's
//!   job.
//!
//! # Alignment: the length is over-stated, not the file over-sized
//!
//! `newBufferWithBytesNoCopy` requires a page-aligned pointer AND a
//! page-aligned length. The pointer is free — `mmap` returns page-aligned
//! bases — but an artifact's byte count is whatever the writer wrote, and
//! rounding it is a real choice with two roads:
//!
//! * `ftruncate` the artifact up to a page multiple, so the length on disk
//!   is already aligned. Rejected: it mutates the operator's file, needs
//!   write permission on a read-only serving artifact, and makes the
//!   checkpoint's own byte count a lie.
//! * **State a longer length to Metal than the file holds.** Taken. `mmap`
//!   of an `L`-byte file reserves `round_up(L, page)` bytes of address
//!   space, and POSIX guarantees the partial page at the end is zero-filled
//!   and readable — the tail is real, mapped, faultable memory that simply
//!   is not backed by file bytes. So the `MTLBuffer` is minted over
//!   [`Mapping::span`] (page-rounded) while [`Mapping::len`] keeps the true
//!   payload length, and the [`Buffer`](crate::device::Buffer) minted from
//!   it reports the TRUE length: every `span` bounds check, every handle
//!   minted over it, and every kernel that reads through it are held to the
//!   file's own size. Metal is told about a tail no caller can address.
//!
//! # The mapping outlives the buffer BY CONSTRUCTION
//!
//! `newBufferWithBytesNoCopy` with a nil deallocator does not own the
//! pages: the moment the mapping is unmapped, the `MTLBuffer` points at
//! nothing and the next kernel that reads it faults on the device. The
//! guard is ownership rather than discipline —
//! [`Buffer::mapped`](crate::device::Buffer::mapped) takes an
//! `Arc<Mapping>` and HOLDS it, beside the retained buffer and declared
//! after it, so the `MTLBuffer` is released before the `munmap` on the same
//! drop and a clone of the `Buffer` extends both. There is no door that
//! binds a bare pointer, so there is no way to spell a buffer whose mapping
//! has gone: bind-after-drop is not refused at run time, it is unsayable.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{Fault, Result};

/// The host's page size, which is what both of
/// `newBufferWithBytesNoCopy`'s alignment rules are stated in.
///
/// Probed rather than assumed: 16 KiB on Apple silicon, 4 KiB on an Intel
/// Mac and under Rosetta, and a constant here would be wrong on two of the
/// three.
#[must_use]
pub fn page() -> usize {
    // SAFETY: `sysconf` of a defined name, reading no memory.
    let said = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if said > 0 {
        said as usize
    } else {
        4096
    }
}

/// **One artifact file, mapped read-only, whole.**
///
/// Held behind an [`Arc`] by every [`Buffer`](crate::device::Buffer) minted
/// over it, which is what keeps the pages alive for as long as any device
/// buffer names them. Read the module header before reaching for one: this
/// is the fits-in-memory door, and it wires what the GPU touches.
pub struct Mapping {
    /// The mapping's base — page-aligned, because `mmap` says so.
    at: std::ptr::NonNull<u8>,
    /// How many bytes of address space the mapping actually spans:
    /// `round_up(len, page())`, and the length Metal is told.
    span: usize,
    /// The artifact's true byte count — what every caller above is bounded
    /// to, and what the file holds.
    len: u64,
    /// Kept open past the `mmap` that does not need it, so a gate can
    /// `fstat` the backing and so `{:?}` can name what this is a mapping OF.
    file: std::fs::File,
    /// The path as the operator named it, for refusals and for `Debug`.
    path: PathBuf,
}

// SAFETY: the mapping is `MAP_PRIVATE` and `PROT_READ` — there is no way to
// write through this type and no interior mutability in it, so every
// reference reads bytes nothing in the process can change. That makes it
// `Sync` as well as `Send`, which is what an `Arc<Mapping>` shared by
// several `Buffer` clones needs.
unsafe impl Send for Mapping {}
// SAFETY: as `Send` — read-only pages, no interior mutability.
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Map `path` read-only, whole, and keep its descriptor.
    ///
    /// # Errors
    ///
    /// [`Fault::Mapped`] naming the step that refused (`open`, `stat`,
    /// `size`, `map`) and the OS's own sentence — an artifact that has moved,
    /// that this process may not read, or an address space with no room for
    /// it are all deployment conditions and the message has to say which.
    pub fn of(path: impl AsRef<Path>) -> Result<Arc<Mapping>> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|why| Fault::Mapped {
            step: "open",
            what: path.display().to_string(),
            why: why.to_string(),
        })?;
        Mapping::of_file(file, path.to_path_buf())
    }

    /// Map an already-open artifact, whole — the door the checkpoint reader
    /// (M-2) reaches for, which has the descriptor before it has a decision.
    ///
    /// `named` is what refusals and `Debug` will call this; it is never
    /// opened, so a caller holding a descriptor to an unlinked or otherwise
    /// unnameable file may say whatever identifies it.
    ///
    /// # Errors
    ///
    /// [`Fault::Mapped`] naming `stat`, `size` or `map`.
    pub fn of_file(file: std::fs::File, named: PathBuf) -> Result<Arc<Mapping>> {
        let what = || named.display().to_string();
        let len = file
            .metadata()
            .map_err(|why| Fault::Mapped {
                step: "stat",
                what: what(),
                why: why.to_string(),
            })?
            .len();
        // An empty artifact is refused rather than answered with an empty
        // mapping: `mmap` will not take a zero length, and a zero-byte
        // checkpoint is a broken one, not a legal one. (The EMPTY
        // reservation that a plan may legally hold is `Buffer::zeroed(_, 0)`
        // and lives on the eager path; nothing maps it.)
        if len == 0 {
            return Err(Fault::Mapped {
                step: "size",
                what: what(),
                why: "the artifact holds no bytes".into(),
            });
        }
        let page = page();
        let span = usize::try_from(len)
            .ok()
            .and_then(|len| len.checked_next_multiple_of(page))
            .ok_or_else(|| Fault::Mapped {
                step: "size",
                what: what(),
                why: format!("{len} bytes does not fit this process's address space"),
            })?;
        // SAFETY: a fresh private read-only mapping of a file this function
        // holds open, at a span that is the file's own length rounded up to
        // the page — the tail past `len` is the partial page POSIX
        // guarantees is zero-filled and readable.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                span,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                std::os::fd::AsRawFd::as_raw_fd(&file),
                0,
            )
        };
        if at == libc::MAP_FAILED {
            return Err(Fault::Mapped {
                step: "map",
                what: what(),
                why: std::io::Error::last_os_error().to_string(),
            });
        }
        let at = std::ptr::NonNull::new(at.cast::<u8>()).ok_or_else(|| Fault::Mapped {
            step: "map",
            what: what(),
            why: "the kernel answered a null mapping".into(),
        })?;
        Ok(Arc::new(Mapping {
            at,
            span,
            len,
            file,
            path: named,
        }))
    }

    /// The artifact's TRUE byte count — the number every caller above the
    /// bind is bounded to.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the artifact holds no bytes, which [`Mapping::of`] refuses to
    /// make, so this is always `false` and exists for the lint that pairs it
    /// with [`Mapping::len`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many bytes of address space the mapping spans:
    /// `round_up(len, page())`, the page-aligned length Metal is told and
    /// the one no caller above may address. See the module header.
    #[must_use]
    pub fn span(&self) -> usize {
        self.span
    }

    /// The mapping's page-aligned base, for the bind and for nothing else.
    pub(crate) fn base(&self) -> std::ptr::NonNull<u8> {
        self.at
    }

    /// What was mapped, as the caller named it.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// **What is actually behind this mapping**: the backing file's size, as
    /// `fstat` reports it now.
    ///
    /// A gate's half of "this buffer is the file": a mapped artifact whose
    /// `backing()` equals its [`len`](Mapping::len) is provably a window
    /// onto real storage rather than onto a copy of it.
    #[must_use]
    pub fn backing(&self) -> Option<u64> {
        self.file.metadata().ok().map(|it| it.len())
    }
}

impl std::ops::Deref for Mapping {
    type Target = [u8];

    /// The artifact's bytes at their TRUE length — never the page-rounded
    /// span, whose tail is zero-fill that belongs to no file.
    fn deref(&self) -> &[u8] {
        // SAFETY: `len` readable bytes inside a mapping of `span >= len`
        // bytes this type owns for its whole life, and which nothing may
        // write through.
        unsafe {
            std::slice::from_raw_parts(
                self.at.as_ptr(),
                usize::try_from(self.len).expect("a length inside a live mapping"),
            )
        }
    }
}

impl std::fmt::Debug for Mapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The BYTES are never printed: this is a whole model.
        f.debug_struct("Mapping")
            .field("path", &self.path)
            .field("bytes", &self.len)
            .field("span", &self.span)
            .finish()
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: unmapping this type's own mapping at its own span. Every
        // `MTLBuffer` minted over it holds an `Arc` to this type, so the
        // last of them was released before this ran — which is the whole
        // reason the bind takes an `Arc` and not a pointer.
        unsafe {
            libc::munmap(self.at.as_ptr().cast(), self.span);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A temporary file of `bytes` bytes of a known pattern, unlinked by the
    /// caller. Lives in `TMPDIR`, as the crate's hard rule says.
    fn scratch(name: &str, bytes: usize) -> PathBuf {
        let path = std::env::temp_dir().join(format!("pie-map-{}-{name}", std::process::id()));
        let mut file = std::fs::File::create(&path).expect("a scratch artifact");
        let pattern: Vec<u8> = (0..bytes).map(|at| (at % 251) as u8).collect();
        file.write_all(&pattern).expect("the pattern lands");
        path
    }

    #[test]
    fn a_mapping_states_the_files_length_and_spans_the_page() {
        let path = scratch("odd", 5000);
        let map = Mapping::of(&path).expect("the artifact maps");
        let _ = std::fs::remove_file(&path);

        assert_eq!(map.len(), 5000, "the length is the file's, to the byte");
        assert_eq!(
            map.span(),
            5000usize.next_multiple_of(page()),
            "the span is that length rounded up to the page, which is what Metal is told"
        );
        assert!(map.span() > map.len() as usize, "and this file needs the tail");
        assert_eq!(map.backing(), Some(5000), "the mapping is the file");
        assert!(
            map.iter().enumerate().all(|(at, byte)| *byte == (at % 251) as u8),
            "every byte of the payload reads back through the deref"
        );
        assert_eq!(
            (**map).len(),
            5000,
            "the deref is the TRUE length and never the span"
        );
    }

    #[test]
    fn an_exactly_paged_artifact_needs_no_tail() {
        let bytes = page() * 3;
        let path = scratch("even", bytes);
        let map = Mapping::of(&path).expect("the artifact maps");
        let _ = std::fs::remove_file(&path);
        assert_eq!(map.span(), bytes, "an aligned length is already the span");
        assert_eq!(map.len() as usize, bytes);
    }

    #[test]
    fn an_empty_artifact_is_refused_by_name() {
        let path = scratch("empty", 0);
        let fault = Mapping::of(&path).expect_err("a zero-byte artifact does not map");
        let _ = std::fs::remove_file(&path);
        let said = fault.to_string();
        assert!(said.contains("holds no bytes"), "the refusal says why: {said}");
    }

    #[test]
    fn an_artifact_that_is_not_there_is_refused_by_name() {
        let fault = Mapping::of("/nonesuch/pie/artifact.zt").expect_err("nothing to map");
        let said = fault.to_string();
        assert!(
            said.contains("artifact.zt") && said.contains("open"),
            "the refusal names the file and the step: {said}"
        );
    }
}
