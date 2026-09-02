//! The artifact, mapped once and bound without a copy: `newBufferWithBytesNoCopy`
//! hands Metal the mapping's own pages, faulted from the file exactly once.
//!
//! A reservation wires whole on first GPU touch, so this bounds no memory
//! and must never carry the streamed path; [`cut`] windows past
//! `maxBufferLength` around the planes a load binds, and each buffer's
//! `Arc<Mapping>` outlives it while named.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{Fault, Result};

/// The host's page size. Probed rather than assumed: 16 KiB on Apple
/// silicon, 4 KiB on an Intel Mac and under Rosetta.
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

/// One artifact file, mapped read-only, whole. Held behind an [`Arc`] by
/// every [`Buffer`](crate::device::Buffer) minted over it.
pub struct Mapping {
    at: std::ptr::NonNull<u8>,
    /// Address space spanned: `round_up(len, page())`, the length Metal is told.
    span: usize,
    /// The artifact's true byte count, what every caller above is bounded to.
    len: u64,
    file: std::fs::File,
    path: PathBuf,
}

// SAFETY: `MAP_PRIVATE` and `PROT_READ`, with no interior mutability — every
// reference reads bytes nothing in the process can change, which is `Sync`
// as well as `Send`.
unsafe impl Send for Mapping {}
// SAFETY: as `Send` — read-only pages, no interior mutability.
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Map `path` read-only, whole, and keep its descriptor. Errors with
    /// [`Fault::Mapped`] naming the step that refused.
    pub fn of(path: impl AsRef<Path>) -> Result<Arc<Mapping>> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|why| Fault::Mapped {
            step: "open",
            what: path.display().to_string(),
            why: why.to_string(),
        })?;
        Mapping::of_file(file, path.to_path_buf())
    }

    /// Map an already-open artifact, whole. `named` is what refusals and
    /// `Debug` will call this; it is never opened, so a caller holding a
    /// descriptor to an unlinked file may say whatever identifies it.
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
        // `mmap` will not take a zero length, and a zero-byte checkpoint is
        // broken, not legal.
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
        // SAFETY: a fresh mapping of a file held open, at a page-rounded
        // span POSIX zero-fills past `len`.
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

    /// The file this is a mapping of — a seat copy `pread`s through it.
    #[must_use]
    pub fn file(&self) -> &std::fs::File {
        &self.file
    }

    /// The artifact's true byte count, which every caller above is bounded to.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Always `false` ([`Mapping::of`] refuses an empty artifact); exists
    /// for the lint pairing it with [`Mapping::len`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Address space the mapping spans, page-aligned — no caller above may address it.
    #[must_use]
    pub fn span(&self) -> usize {
        self.span
    }

    /// The mapping's page-aligned base, for the bind and for nothing else.
    /// Read by the Apple-only `newBufferWithBytesNoCopy` arm, which is why a
    /// non-Apple dead-code sweep sees no caller.
    pub(crate) fn base(&self) -> std::ptr::NonNull<u8> {
        self.at
    }

    /// What was mapped, as the caller named it.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The backing file's size — equal to [`len`](Mapping::len) when this
    /// provably windows real storage.
    #[must_use]
    pub fn backing(&self) -> Option<u64> {
        self.file.metadata().ok().map(|it| it.len())
    }

    /// How many names the backing file has — at least one, unlike a staged
    /// (unlinked) copy, whose link count is `0`.
    #[must_use]
    pub fn links(&self) -> Option<u64> {
        use std::os::unix::fs::MetadataExt;
        self.file.metadata().ok().map(|it| it.nlink())
    }
}

/// One page-aligned window of a [`Mapping`], as one `MTLBuffer` will see it
/// — the unit [`cut`] answers in and
/// [`Buffer::window`](crate::device::Buffer::window) mints. Must be passed
/// back beside the same `Arc<Mapping>` it was cut from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut {
    /// Where this window starts inside the mapping — page-aligned.
    base: usize,
    /// How many bytes Metal is told, a page multiple never above `ceiling`.
    span: usize,
    /// Bytes of the artifact this window addresses — `span`, except the
    /// last chunk's zero-fill tail.
    bytes: u64,
}

impl Cut {
    /// The whole mapping as one window, for an artifact under the ceiling.
    #[must_use]
    pub fn whole(map: &Mapping) -> Cut {
        Cut {
            base: 0,
            span: map.span(),
            bytes: map.len(),
        }
    }

    /// Where the window starts inside the mapping, page-aligned.
    #[must_use]
    pub fn base(&self) -> usize {
        self.base
    }

    /// The page-aligned length Metal is told.
    #[must_use]
    pub fn span(&self) -> usize {
        self.span
    }

    /// The TRUE artifact bytes this window addresses.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether a blob at `offset` of `length` bytes lies wholly inside this window.
    #[must_use]
    pub fn holds(&self, offset: u64, length: u64) -> bool {
        let base = self.base as u64;
        offset >= base && offset.saturating_add(length) <= base.saturating_add(self.bytes)
    }

    /// Where a blob at `offset` sits within this window, for `Handles::bind`.
    #[must_use]
    pub fn view(&self, offset: u64) -> Option<u64> {
        offset.checked_sub(self.base as u64)
    }
}

/// How far apart two bound planes may be and still share one window — the
/// writer's own padding (`pie model import` aligns at 2 MiB), never a band.
const SLACK: u64 = 8 << 20;

/// The device's `maxBufferLength`, clamped down by `PIE_METAL_WINDOW_CEILING`
/// if that env var is a valid, positive, smaller byte count.
#[must_use]
pub fn ceiling(device: u64) -> u64 {
    std::env::var("PIE_METAL_WINDOW_CEILING")
        .ok()
        .and_then(|it| it.parse::<u64>().ok())
        .filter(|it| *it > 0)
        .map_or(device, |it| it.min(device))
}

/// Cut a mapping into windows, none larger than `ceiling`, that cover the
/// planes a load will bind — `bound` as `(name, offset, length)` — rather
/// than the whole file, since a window wires whole.
///
/// # Errors
///
/// [`Fault::Mapped`] at step `cut`, naming the plane that leaves the
/// artifact or alone exceeds the ceiling, or (nameless) for a sub-page
/// `ceiling` or an oversized mapping with no bound plane at all.
pub fn cut(map: &Mapping, ceiling: u64, bound: &[(&str, u64, u64)]) -> Result<Vec<Cut>> {
    let page = page();
    let what = || map.path().display().to_string();
    let refuse = |why: String| Fault::Mapped {
        step: "cut",
        what: what(),
        why,
    };
    // Floored once, here, so no window's page-multiple length can round
    // back above the device's own number.
    let ceiling = usize::try_from(ceiling).unwrap_or(usize::MAX) / page * page;
    if ceiling == 0 {
        return Err(refuse(format!(
            "the device reserves fewer than one {page}-byte page in a buffer"
        )));
    }
    let span = map.span();
    let len = map.len();
    // The whole file fits one reservation; the plane list goes unread.
    if span <= ceiling {
        return Ok(vec![Cut::whole(map)]);
    }
    if bound.is_empty() {
        return Err(refuse(format!(
            "it holds {len} bytes against a {ceiling}-byte reservation and this load \
             binds no plane of it, so there is nothing to cut a window around"
        )));
    }

    let mut sorted: Vec<(&str, u64, u64)> = bound.to_vec();
    sorted.sort_by_key(|(_, offset, length)| (*offset, *length));
    if std::env::var_os("PIE_METAL_PREFAULT").is_some_and(|v| v != "0") {
        prefault(map, &sorted);
    }

    let chunk = |base: usize, end: usize| Cut {
        base,
        span: end - base,
        bytes: (len - base as u64).min((end - base) as u64),
    };
    let mut cuts: Vec<Cut> = Vec::new();
    // The window being built: page-aligned base, and page above its last plane.
    let mut open: Option<(usize, usize)> = None;
    for (name, offset, length) in sorted {
        let end = offset
            .checked_add(length)
            .filter(|end| *end <= len)
            .ok_or_else(|| {
                refuse(format!(
                    "`{name}` claims [{offset}, {}) of an artifact that holds {len} bytes",
                    offset.saturating_add(length),
                ))
            })?;
        let lo = (offset as usize) / page * page;
        let hi = (end as usize).next_multiple_of(page).min(span);
        if hi - lo > ceiling {
            return Err(refuse(format!(
                "`{name}` spans [{offset}, {end}) — {length} bytes — and a reservation on \
                 this device holds {ceiling}, so it can be a view of no buffer this \
                 device mints"
            )));
        }
        match open {
            // Extend when only the writer's padding lies between, and it still fits.
            Some((base, cover))
                if lo.saturating_sub(cover) as u64 <= SLACK && hi - base <= ceiling =>
            {
                open = Some((base, cover.max(hi)));
            }
            Some((base, cover)) => {
                cuts.push(chunk(base, cover));
                open = Some((lo, hi));
            }
            None => open = Some((lo, hi)),
        }
    }
    if let Some((base, cover)) = open {
        cuts.push(chunk(base, cover));
    }
    Ok(cuts)
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
        // SAFETY: unmapping this type's own mapping; every `MTLBuffer`
        // minted over it holds an `Arc` to this, so the last was released first.
        unsafe {
            libc::munmap(self.at.as_ptr().cast(), self.span);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A temporary file of `bytes` bytes of a known pattern, unlinked by
    /// the caller.
    fn scratch(name: &str, bytes: usize) -> PathBuf {
        let path = std::env::temp_dir().join(format!("pie-map-{}-{name}", std::process::id()));
        let mut file = std::fs::File::create(&path).expect("a scratch artifact");
        let pattern: Vec<u8> = (0..bytes).map(|at| (at % 251) as u8).collect();
        file.write_all(&pattern).expect("the pattern lands");
        path
    }

    #[test]
    fn an_empty_artifact_is_refused_by_name() {
        let path = scratch("empty", 0);
        let fault = Mapping::of(&path).expect_err("a zero-byte artifact does not map");
        let _ = std::fs::remove_file(&path);
        let said = fault.to_string();
        assert!(said.contains("holds no bytes"), "the refusal says why: {said}");
    }

    /// A sparse scratch artifact of `bytes` bytes — `ftruncate` and no
    /// write, so it costs no disk despite exceeding `maxBufferLength`.
    fn sparse(name: &str, bytes: u64) -> PathBuf {
        let path = std::env::temp_dir().join(format!("pie-cut-{}-{name}", std::process::id()));
        let file = std::fs::File::create(&path).expect("a scratch artifact");
        file.set_len(bytes).expect("the artifact sizes");
        path
    }

    fn borrow(of: &[(String, u64, u64)]) -> Vec<(&str, u64, u64)> {
        of.iter()
            .map(|(name, offset, length)| (name.as_str(), *offset, *length))
            .collect()
    }

    /// A plane no buffer can hold is refused by name, since a view cannot
    /// span two reservations.
    #[test]
    fn a_blob_larger_than_one_buffer_is_refused_by_its_own_name() {
        let page = page();
        let path = sparse("huge-plane", 64 * page as u64);
        let map = Mapping::of(&path).expect("the sparse artifact maps");
        let _ = std::fs::remove_file(&path);

        let held = vec![
            ("small".to_string(), 0u64, 4 * page as u64),
            ("the-whale".to_string(), 8 * page as u64, 10 * page as u64),
        ];
        let fault = cut(&map, 8 * page as u64, &borrow(&held))
            .expect_err("a plane past the ceiling is not a view of anything");
        let said = fault.to_string();
        assert!(
            said.contains("the-whale"),
            "the refusal names the plane that cannot be served: {said}"
        );
        assert!(
            said.contains("view of no buffer"),
            "and says what it is that cannot be done: {said}"
        );
    }

    /// An oversized artifact this load binds no plane of is refused.
    #[test]
    fn an_oversized_artifact_with_no_bound_plane_is_refused() {
        let page = page();
        let path = sparse("nothing-bound", 64 * page as u64);
        let map = Mapping::of(&path).expect("the sparse artifact maps");
        let _ = std::fs::remove_file(&path);
        let fault = cut(&map, 8 * page as u64, &[]).expect_err("no plane, no window");
        assert!(
            fault.to_string().contains("binds no plane"),
            "the refusal says what is missing: {fault}"
        );
    }

    /// A ceiling below one page is refused before a manifest is walked.
    #[test]
    fn a_ceiling_under_one_page_is_refused_before_the_manifest() {
        let path = scratch("tiny-ceiling", 5000);
        let map = Mapping::of(&path).expect("the artifact maps");
        let _ = std::fs::remove_file(&path);
        let fault = cut(&map, 8, &[]).expect_err("a sub-page ceiling binds nothing");
        assert!(
            fault.to_string().contains("fewer than one"),
            "the refusal names the device's own number: {fault}"
        );
    }

    /// A blob whose extent leaves the artifact is refused by name.
    #[test]
    fn a_blob_that_leaves_the_artifact_is_refused_by_name() {
        let page = page();
        let path = sparse("overrun", 64 * page as u64);
        let map = Mapping::of(&path).expect("the sparse artifact maps");
        let _ = std::fs::remove_file(&path);
        let held = vec![("past-the-end".to_string(), 60 * page as u64, 16 * page as u64)];
        let fault = cut(&map, 8 * page as u64, &borrow(&held))
            .expect_err("a blob past the end does not cut");
        let said = fault.to_string();
        assert!(said.contains("past-the-end"), "the refusal names it: {said}");
    }

}

/// **PREFAULT THE RESIDENT PLANES** (`PIE_METAL_PREFAULT=1`): the artifact is
/// mapped and bound without a copy, so the device's first touch of a plane
/// page-faults it in — a random walk over the whole resident tier that the
/// first fire after boot waits on. Asking the kernel for the pages up front,
/// sequentially and on several threads, is the same bytes at the disk's
/// sequential rate instead.
fn prefault(map: &Mapping, planes: &[(&str, u64, u64)]) {
    let page = page() as u64;
    let base = map.base().as_ptr() as usize as u64;
    let len = map.len();
    let started = std::time::Instant::now();
    let mut ranges: Vec<(u64, u64)> = planes
        .iter()
        .map(|&(_, offset, length)| {
            let lo = offset / page * page;
            let hi = (offset + length).min(len).div_ceil(page) * page;
            (lo, hi.min(map.span() as u64))
        })
        .filter(|(lo, hi)| hi > lo)
        .collect();
    ranges.sort_unstable();
    let total: u64 = ranges.iter().map(|(lo, hi)| hi - lo).sum();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    let shares: Vec<Vec<(u64, u64)>> = (0..threads)
        .map(|t| ranges.iter().copied().skip(t).step_by(threads).collect())
        .collect();
    std::thread::scope(|scope| {
        for share in &shares {
            scope.spawn(move || {
                let mut sink = 0u64;
                for &(lo, hi) in share {
                    // SAFETY: `[lo, hi)` lies inside the live PROT_READ mapping.
                    unsafe {
                        libc::madvise(
                            (base + lo) as *mut libc::c_void,
                            (hi - lo) as usize,
                            libc::MADV_WILLNEED,
                        );
                        let mut at = lo;
                        while at < hi {
                            sink = sink.wrapping_add(u64::from(std::ptr::read_volatile(
                                (base + at) as *const u8,
                            )));
                            at += page;
                        }
                    }
                }
                std::hint::black_box(sink);
            });
        }
    });
    if std::env::var_os("PIE_TIER_TRACE").is_some() {
        eprintln!(
            "load: prefaulted {:.2} GiB of resident planes in {:.2} s",
            total as f64 / (1u64 << 30) as f64,
            started.elapsed().as_secs_f64()
        );
    }
}
