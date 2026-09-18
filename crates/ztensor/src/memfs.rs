//! An in-memory mount table keyed by path.
//!
//! Callers hand a `.zt` around by path, and on a platform with no filesystem
//! (a browser tab) the bytes arrive some other way. Mounting them here under
//! the path the callers already use lets `Store::open` and friends find them
//! without a new byte-source type being threaded through every signature.
//! Lookup is an exact path match; nothing is canonicalised.
//!
//! A mount is one or more chunks laid end to end (`Chunks`), or a lazy mount
//! whose bytes live somewhere else entirely and are fetched on demand. A
//! wasm32 allocation is capped below 2 GiB while an artifact may be larger,
//! so the bytes arrive in pieces; readers address the mount by absolute
//! offset through `read`, `with_slice` and `len`, and never see where one
//! piece ends and the next begins — nor, for a lazy mount, that the bytes
//! were not here until asked for.
//!
//! A lazy mount (`mount_lazy`) holds only a `Fetch` callback and a small
//! read-through cache: reads are served in aligned windows (8 MiB by
//! default), the last few windows are kept so that adjacent small reads and
//! the tail of one span running into the head of the next hit the cache,
//! and the windows holding the file's tail — the footer and manifest every
//! open re-reads — are pinned. A span wider than a window is fetched
//! straight into the caller's buffer, so the mount never holds more than its
//! cache.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

/// Where a lazy mount's bytes come from: fill `into` with the bytes at
/// `offset`. The window is always inside the mount's declared length. The
/// error text is kept as the mount's last error (`error`) for the reader
/// that finds its read refused.
pub trait Fetch: Send + Sync {
    fn fetch(&self, offset: u64, into: &mut [u8]) -> Result<(), String>;
}

impl<F> Fetch for F
where
    F: Fn(u64, &mut [u8]) -> Result<(), String> + Send + Sync,
{
    fn fetch(&self, offset: u64, into: &mut [u8]) -> Result<(), String> {
        self(offset, into)
    }
}

/// The window a lazy mount fetches by default: wide enough that one
/// request serves a run of small tensors, small enough that the cache stays
/// a few tens of MiB.
pub const LAZY_WINDOW: u64 = 8 << 20;

/// How many windows a lazy mount keeps beyond the pinned tail by default.
/// A loader's reads hop about within a few windows of each other (the plan's
/// order is not the file's), so a handful of kept windows turns most of
/// those hops into hits.
pub const LAZY_KEEP: usize = 8;

/// How many windows at the end of the mount stay cached: the footer and
/// manifest live there, and every open of the store reads them again.
pub const LAZY_PIN_TAIL: u64 = 2;

/// How many fetches a lazy mount made, and how many bytes they carried.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LazyStats {
    pub requests: u64,
    pub bytes: u64,
}

struct LazyCache {
    /// Cached windows by start offset, oldest first; a hit moves the window
    /// to the back. Pinned windows (the tail) are not evicted.
    windows: Vec<(u64, Vec<u8>)>,
    /// The last fetch that failed, kept for the error a reader reports.
    error: Option<String>,
}

/// A mount whose bytes are fetched on demand through `Fetch`.
pub struct Lazy {
    len: u64,
    window: u64,
    keep: usize,
    fetch: Box<dyn Fetch>,
    cache: Mutex<LazyCache>,
    requests: AtomicU64,
    fetched: AtomicU64,
}

impl std::fmt::Debug for Lazy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lazy")
            .field("len", &self.len)
            .field("window", &self.window)
            .field("stats", &self.stats())
            .finish()
    }
}

impl Lazy {
    fn new(len: u64, window: u64, keep: usize, fetch: Box<dyn Fetch>) -> Self {
        assert!(window > 0, "a lazy mount needs a window");
        Lazy {
            len,
            window,
            keep,
            fetch,
            cache: Mutex::new(LazyCache {
                windows: Vec::new(),
                error: None,
            }),
            requests: AtomicU64::new(0),
            fetched: AtomicU64::new(0),
        }
    }

    fn stats(&self) -> LazyStats {
        LazyStats {
            requests: self.requests.load(Ordering::Relaxed),
            bytes: self.fetched.load(Ordering::Relaxed),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LazyCache> {
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Where the pinned tail begins.
    fn pin_from(&self) -> u64 {
        let tail = self.window.saturating_mul(LAZY_PIN_TAIL);
        (self.len.div_ceil(self.window).saturating_mul(self.window)).saturating_sub(tail)
    }

    fn fetch(&self, offset: u64, into: &mut [u8]) -> Result<(), String> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.fetched.fetch_add(into.len() as u64, Ordering::Relaxed);
        self.fetch
            .fetch(offset, into)
            .inspect_err(|why| self.lock().error = Some(why.clone()))
    }

    /// Fetch the whole window starting at `start`.
    fn fill(&self, start: u64) -> Result<Vec<u8>, String> {
        let end = start.saturating_add(self.window).min(self.len);
        let mut window = vec![0u8; (end - start) as usize];
        self.fetch(start, &mut window)?;
        Ok(window)
    }

    /// Fill `into` from `offset`. Windows already cached are copied out;
    /// a window only partly covered by the read is fetched whole into the
    /// cache (the next read is likely its neighbour); a run of windows the
    /// read covers entirely is fetched in one request straight into `into`.
    fn read_at(&self, offset: u64, into: &mut [u8]) -> Result<(), String> {
        let end = offset + into.len() as u64;
        let mut cursor = offset;
        // A run of whole, uncached windows waiting to be fetched together.
        let mut run: Option<u64> = None;
        while cursor < end {
            let start = cursor - cursor % self.window;
            let stop = start.saturating_add(self.window).min(self.len);
            let take = stop.min(end);
            let whole = cursor == start && take == stop;
            let cached = {
                let mut cache = self.lock();
                match cache.windows.iter().position(|(at, _)| *at == start) {
                    Some(index) => {
                        let (at, window) = cache.windows.remove(index);
                        let from = (cursor - start) as usize;
                        let n = (take - cursor) as usize;
                        let dst = (cursor - offset) as usize;
                        into[dst..dst + n].copy_from_slice(&window[from..from + n]);
                        cache.windows.push((at, window));
                        true
                    }
                    None => false,
                }
            };
            if cached || !whole {
                if let Some(from) = run.take() {
                    let dst = (from - offset) as usize;
                    let n = (cursor - from) as usize;
                    self.fetch(from, &mut into[dst..dst + n])?;
                }
            }
            if !cached {
                if whole {
                    run.get_or_insert(cursor);
                } else {
                    let window = self.fill(start)?;
                    let from = (cursor - start) as usize;
                    let n = (take - cursor) as usize;
                    let dst = (cursor - offset) as usize;
                    into[dst..dst + n].copy_from_slice(&window[from..from + n]);
                    self.keep(start, window);
                }
            }
            cursor = take;
        }
        if let Some(from) = run.take() {
            let dst = (from - offset) as usize;
            let n = (end - from) as usize;
            self.fetch(from, &mut into[dst..dst + n])?;
        }
        Ok(())
    }

    /// Put a window in the cache, dropping the oldest unpinned one past the
    /// limit.
    fn keep(&self, start: u64, window: Vec<u8>) {
        let pin_from = self.pin_from();
        let mut cache = self.lock();
        cache.windows.retain(|(at, _)| *at != start);
        cache.windows.push((start, window));
        let mut unpinned = cache
            .windows
            .iter()
            .filter(|(at, _)| *at < pin_from)
            .count();
        while unpinned > self.keep {
            let Some(oldest) = cache.windows.iter().position(|(at, _)| *at < pin_from) else {
                break;
            };
            cache.windows.remove(oldest);
            unpinned -= 1;
        }
    }

    fn cached_bytes(&self) -> u64 {
        self.lock()
            .windows
            .iter()
            .map(|(_, window)| window.len() as u64)
            .sum()
    }
}

/// The bytes behind a mount: chunks laid end to end, addressed as one
/// sequence, or a lazy mount fetched on demand. Cloning shares the chunks
/// (and, for a lazy mount, its cache).
#[derive(Clone, Debug, Default)]
pub struct Chunks {
    parts: Vec<Arc<[u8]>>,
    /// `ends[i]` is the absolute offset just past `parts[i]`.
    ends: Vec<u64>,
    lazy: Option<Arc<Lazy>>,
}

impl Chunks {
    pub fn new(parts: Vec<Arc<[u8]>>) -> Self {
        let mut ends = Vec::with_capacity(parts.len());
        let mut total = 0u64;
        for part in &parts {
            total += part.len() as u64;
            ends.push(total);
        }
        Self {
            parts,
            ends,
            lazy: None,
        }
    }

    /// A mount of `len` bytes served through `fetch` in windows of `window`
    /// bytes, keeping [`LAZY_KEEP`] of them (see the module notes for what
    /// is cached).
    pub fn lazy(len: u64, window: u64, fetch: Box<dyn Fetch>) -> Self {
        Self::lazy_keeping(len, window, LAZY_KEEP, fetch)
    }

    /// `lazy` with the number of kept windows stated.
    pub fn lazy_keeping(len: u64, window: u64, keep: usize, fetch: Box<dyn Fetch>) -> Self {
        Self {
            parts: Vec::new(),
            ends: Vec::new(),
            lazy: Some(Arc::new(Lazy::new(len, window, keep, fetch))),
        }
    }

    pub fn len(&self) -> u64 {
        match &self.lazy {
            Some(lazy) => lazy.len,
            None => self.ends.last().copied().unwrap_or(0),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True when the bytes are fetched on demand rather than held here.
    pub fn is_lazy(&self) -> bool {
        self.lazy.is_some()
    }

    /// A lazy mount's fetch count and bytes; None for a mount held in memory.
    pub fn lazy_stats(&self) -> Option<LazyStats> {
        self.lazy.as_ref().map(|lazy| lazy.stats())
    }

    /// How many bytes a lazy mount's cache holds right now (0 otherwise).
    pub fn cached_bytes(&self) -> u64 {
        self.lazy.as_ref().map_or(0, |lazy| lazy.cached_bytes())
    }

    /// The last fetch failure of a lazy mount, if any.
    pub fn error(&self) -> Option<String> {
        self.lazy
            .as_ref()
            .and_then(|lazy| lazy.lock().error.clone())
    }

    /// The chunks held in memory; a lazy mount holds none.
    pub fn parts(&self) -> &[Arc<[u8]>] {
        &self.parts
    }

    pub fn into_parts(self) -> Vec<Arc<[u8]>> {
        self.parts
    }

    /// The whole mount as one slice, when it is one chunk (or nothing).
    pub fn contiguous(&self) -> Option<&[u8]> {
        if self.lazy.is_some() {
            return None;
        }
        match self.parts.as_slice() {
            [] => Some(&[]),
            [one] => Some(one),
            _ => None,
        }
    }

    /// The one allocation behind a single-chunk mount.
    pub fn single(&self) -> Option<Arc<[u8]>> {
        if self.lazy.is_some() {
            return None;
        }
        match self.parts.as_slice() {
            [] => Some(Arc::from(&[][..])),
            [one] => Some(one.clone()),
            _ => None,
        }
    }

    /// The chunk holding `offset` and the offset within it; None past the end.
    fn locate(&self, offset: u64) -> Option<(usize, usize)> {
        if offset >= self.len() {
            return None;
        }
        // The first chunk ending past `offset`; an empty chunk in the way
        // ends where its predecessor did and is skipped.
        let index = self.ends.partition_point(|&end| end <= offset);
        let start = if index == 0 { 0 } else { self.ends[index - 1] };
        Some((index, (offset - start) as usize))
    }

    fn in_range(&self, offset: u64, len: u64) -> bool {
        offset.checked_add(len).is_some_and(|end| end <= self.len())
    }

    /// Fill `into` from `offset`; None when the window falls outside the
    /// mount, or a lazy mount's fetch failed (see `error`).
    pub fn read_at(&self, offset: u64, into: &mut [u8]) -> Option<()> {
        if !self.in_range(offset, into.len() as u64) {
            return None;
        }
        if into.is_empty() {
            return Some(());
        }
        if let Some(lazy) = &self.lazy {
            return lazy.read_at(offset, into).ok();
        }
        let (mut index, mut at) = self.locate(offset)?;
        let mut done = 0;
        while done < into.len() {
            let part = &self.parts[index];
            let n = (part.len() - at).min(into.len() - done);
            into[done..done + n].copy_from_slice(&part[at..at + n]);
            done += n;
            at = 0;
            index += 1;
        }
        Some(())
    }

    /// A copy of the `offset..offset+len` window; None when it falls outside
    /// the mount.
    pub fn read(&self, offset: u64, len: u64) -> Option<Vec<u8>> {
        if !self.in_range(offset, len) {
            return None;
        }
        let mut out = vec![0u8; usize::try_from(len).ok()?];
        self.read_at(offset, &mut out)?;
        Some(out)
    }

    /// The chunk holding the whole `offset..offset+len` window, as the
    /// window's bounds within it; None when the window straddles a seam,
    /// falls outside the mount, or the mount is lazy (an empty window in
    /// range always has a view).
    fn inside_one(&self, offset: u64, len: u64) -> Option<(usize, std::ops::Range<usize>)> {
        if !self.in_range(offset, len) {
            return None;
        }
        if len == 0 {
            return Some((0, 0..0));
        }
        if self.lazy.is_some() {
            return None;
        }
        let (index, at) = self.locate(offset)?;
        let end = usize::try_from(len)
            .ok()
            .and_then(|len| at.checked_add(len))
            .filter(|&end| end <= self.parts[index].len())?;
        Some((index, at..end))
    }

    /// True when the window lies inside one chunk (or is empty and in
    /// range), so `slice` borrows it.
    pub fn is_contiguous(&self, offset: u64, len: u64) -> bool {
        self.inside_one(offset, len).is_some()
    }

    /// The `offset..offset+len` window: borrowed when it lies inside one
    /// chunk, a copy when it straddles a boundary or the mount is lazy;
    /// None when it falls outside the mount.
    pub fn slice(&self, offset: u64, len: u64) -> Option<Cow<'_, [u8]>> {
        match self.inside_one(offset, len) {
            Some((_, range)) if range.is_empty() => Some(Cow::Borrowed(&[])),
            Some((index, range)) => Some(Cow::Borrowed(&self.parts[index][range])),
            None if self.in_range(offset, len) => self.read(offset, len).map(Cow::Owned),
            None => None,
        }
    }

    /// Run `f` over the `offset..offset+len` window, borrowed when it lies
    /// inside one chunk and copied when it straddles; None when it falls
    /// outside the mount.
    pub fn with_slice<R>(&self, offset: u64, len: u64, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        self.slice(offset, len).map(|window| f(&window))
    }
}

impl From<Arc<[u8]>> for Chunks {
    fn from(bytes: Arc<[u8]>) -> Self {
        Chunks::new(vec![bytes])
    }
}

impl From<Vec<Arc<[u8]>>> for Chunks {
    fn from(parts: Vec<Arc<[u8]>>) -> Self {
        Chunks::new(parts)
    }
}

static MOUNTS: LazyLock<Mutex<HashMap<PathBuf, Chunks>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn table() -> std::sync::MutexGuard<'static, HashMap<PathBuf, Chunks>> {
    MOUNTS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Serve `bytes` for every open of `path`; a later mount of the same path
/// replaces the earlier one.
pub fn mount(path: impl AsRef<Path>, bytes: Arc<[u8]>) {
    mount_chunks(path, vec![bytes]);
}

/// Serve `parts`, laid end to end, for every open of `path`.
pub fn mount_chunks(path: impl AsRef<Path>, parts: Vec<Arc<[u8]>>) {
    table().insert(path.as_ref().to_path_buf(), Chunks::new(parts));
}

/// Serve `len` bytes for every open of `path`, fetched through `fetch` as
/// they are read, in windows of [`LAZY_WINDOW`] bytes.
pub fn mount_lazy(path: impl AsRef<Path>, len: u64, fetch: Box<dyn Fetch>) {
    mount_lazy_windowed(path, len, LAZY_WINDOW, fetch);
}

/// `mount_lazy` with the fetch window stated.
pub fn mount_lazy_windowed(path: impl AsRef<Path>, len: u64, window: u64, fetch: Box<dyn Fetch>) {
    mount_lazy_with(path, len, window, LAZY_KEEP, fetch);
}

/// `mount_lazy` with the fetch window and the number of kept windows stated.
pub fn mount_lazy_with(
    path: impl AsRef<Path>,
    len: u64,
    window: u64,
    keep: usize,
    fetch: Box<dyn Fetch>,
) {
    table().insert(
        path.as_ref().to_path_buf(),
        Chunks::lazy_keeping(len, window, keep, fetch),
    );
}

/// Forget a mount; opens of `path` fall back to the filesystem. Stores that
/// already opened the mount keep their bytes.
pub fn unmount(path: impl AsRef<Path>) -> Option<Chunks> {
    table().remove(path.as_ref())
}

/// The one allocation behind `path`, for a mount made of a single chunk;
/// None for a chunked or lazy mount, which is read through
/// `read`/`with_slice`.
pub fn get(path: impl AsRef<Path>) -> Option<Arc<[u8]>> {
    table().get(path.as_ref()).and_then(Chunks::single)
}

/// The chunks behind `path`.
pub fn chunks(path: impl AsRef<Path>) -> Option<Chunks> {
    table().get(path.as_ref()).cloned()
}

pub fn len(path: impl AsRef<Path>) -> Option<u64> {
    table().get(path.as_ref()).map(Chunks::len)
}

/// Fill `into` from `offset` of the mount at `path`; None when nothing is
/// mounted there, the window falls outside it, or a lazy fetch failed
/// (`error` then says why).
pub fn read(path: impl AsRef<Path>, offset: u64, into: &mut [u8]) -> Option<()> {
    chunks(path)?.read_at(offset, into)
}

/// Run `f` over a window of the mount at `path` (see `Chunks::with_slice`);
/// None when nothing is mounted there or the window falls outside it.
pub fn with_slice<R>(
    path: impl AsRef<Path>,
    offset: u64,
    len: u64,
    f: impl FnOnce(&[u8]) -> R,
) -> Option<R> {
    chunks(path)?.with_slice(offset, len, f)
}

pub fn is_mounted(path: impl AsRef<Path>) -> bool {
    table().contains_key(path.as_ref())
}

/// A lazy mount's fetch count and bytes so far; None when `path` is not a
/// lazy mount.
pub fn lazy_stats(path: impl AsRef<Path>) -> Option<LazyStats> {
    table().get(path.as_ref()).and_then(Chunks::lazy_stats)
}

/// The last fetch failure of the lazy mount at `path`, for the reader whose
/// read came back empty.
pub fn error(path: impl AsRef<Path>) -> Option<String> {
    table().get(path.as_ref()).and_then(Chunks::error)
}

/// The text a reader appends when a mounted read fails: the fetch failure
/// of a lazy mount, or nothing.
pub fn read_failure(path: impl AsRef<Path>) -> String {
    match error(path) {
        Some(why) => format!(" (the last fetch failed: {why})"),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn memfs_every_case() {
        a_single_chunk_mount();
        a_chunked_mount_reads_as_one_sequence();
        a_lazy_mount_fetches_windows_on_demand();
        a_lazy_mount_keeps_its_tail_and_reports_failures();
    }

    fn a_single_chunk_mount() {
        let path = Path::new("/no/such/dir/memfs-probe.zt");
        assert!(!is_mounted(path));
        assert!(get(path).is_none());
        assert!(len(path).is_none());

        let bytes: Arc<[u8]> = Arc::from(&b"hello"[..]);
        mount(path, bytes.clone());
        assert!(is_mounted(path));
        assert!(Arc::ptr_eq(&get(path).unwrap(), &bytes));
        assert_eq!(len(path), Some(5));
        assert!(
            !is_mounted("no/such/dir/memfs-probe.zt"),
            "an exact match: the relative spelling is another name"
        );

        let replaced: Arc<[u8]> = Arc::from(&b"world"[..]);
        mount(path, replaced.clone());
        assert!(Arc::ptr_eq(&get(path).unwrap(), &replaced));
        assert_eq!(
            with_slice(path, 1, 3, |w| w.to_vec()),
            Some(b"orl".to_vec())
        );
        assert_eq!(with_slice(path, 3, 3, |w| w.to_vec()), None);
        assert!(lazy_stats(path).is_none());
        assert!(error(path).is_none());
        assert_eq!(read_failure(path), "");

        let taken = unmount(path).unwrap();
        assert!(Arc::ptr_eq(&taken.single().unwrap(), &replaced));
        assert!(!taken.is_lazy());
        assert!(!is_mounted(path));
        assert!(unmount(path).is_none());
    }

    fn a_chunked_mount_reads_as_one_sequence() {
        let path = Path::new("/no/such/dir/memfs-chunked-probe.zt");
        let whole: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        let parts: Vec<Arc<[u8]>> = vec![
            Arc::from(&whole[..300]),
            Arc::from(&whole[300..300]),
            Arc::from(&whole[300..301]),
            Arc::from(&whole[301..700]),
            Arc::from(&whole[700..]),
        ];
        mount_chunks(path, parts.clone());
        assert!(is_mounted(path));
        assert!(get(path).is_none(), "several chunks are not one allocation");
        assert_eq!(len(path), Some(1000));

        let chunks = chunks(path).unwrap();
        assert_eq!(chunks.parts().len(), 5);
        assert!(chunks.contiguous().is_none());
        assert!(matches!(chunks.slice(0, 300).unwrap(), Cow::Borrowed(_)));
        assert!(matches!(chunks.slice(299, 1).unwrap(), Cow::Borrowed(_)));
        assert!(matches!(chunks.slice(300, 1).unwrap(), Cow::Borrowed(_)));
        assert!(matches!(chunks.slice(299, 2).unwrap(), Cow::Owned(_)));
        assert!(matches!(chunks.slice(0, 0).unwrap(), Cow::Borrowed(_)));
        assert!(matches!(chunks.slice(1000, 0).unwrap(), Cow::Borrowed(_)));
        assert!(chunks.slice(1000, 1).is_none());
        assert!(chunks.is_contiguous(299, 1));
        assert!(chunks.is_contiguous(1000, 0));
        assert!(!chunks.is_contiguous(299, 2));
        assert!(!chunks.is_contiguous(1000, 1));
        assert!(chunks.slice(u64::MAX, 1).is_none());
        assert!(chunks.read(999, 2).is_none());

        for (offset, n) in [
            (0u64, 1000u64),
            (0, 1),
            (299, 3),
            (300, 401),
            (298, 500),
            (699, 2),
            (999, 1),
            (1000, 0),
        ] {
            let expect = &whole[offset as usize..(offset + n) as usize];
            assert_eq!(
                &*chunks.slice(offset, n).unwrap(),
                expect,
                "slice {offset}+{n}"
            );
            assert_eq!(chunks.read(offset, n).unwrap(), expect, "read {offset}+{n}");
            let mut into = vec![0u8; n as usize];
            read(path, offset, &mut into).unwrap();
            assert_eq!(into, expect, "memfs::read {offset}+{n}");
            assert_eq!(
                with_slice(path, offset, n, |w| w.to_vec()).unwrap(),
                expect,
                "memfs::with_slice {offset}+{n}"
            );
        }
        let mut too_long = vec![0u8; 2];
        assert!(read(path, 999, &mut too_long).is_none());

        let taken = unmount(path).unwrap();
        assert_eq!(taken.into_parts().len(), 5);
        assert!(read(path, 0, &mut too_long).is_none());

        let empty = Chunks::new(Vec::new());
        assert!(empty.is_empty());
        assert_eq!(empty.contiguous(), Some(&[][..]));
        assert_eq!(empty.single().unwrap().len(), 0);
        assert!(matches!(empty.slice(0, 0).unwrap(), Cow::Borrowed(_)));
        assert!(empty.slice(0, 1).is_none());
        assert_eq!(Chunks::from(parts).len(), 1000);
        assert_eq!(Chunks::from(Arc::from(&whole[..])).parts().len(), 1);
    }

    /// A fetch over `whole` that records every request it serves.
    fn recording(whole: Arc<Vec<u8>>, log: Arc<Mutex<Vec<(u64, u64)>>>) -> Box<dyn Fetch> {
        Box::new(move |offset: u64, into: &mut [u8]| {
            log.lock().unwrap().push((offset, into.len() as u64));
            let at = offset as usize;
            into.copy_from_slice(&whole[at..at + into.len()]);
            Ok(())
        })
    }

    fn a_lazy_mount_fetches_windows_on_demand() {
        const WINDOW: u64 = 64;
        let path = Path::new("/no/such/dir/memfs-lazy-probe.zt");
        let whole: Arc<Vec<u8>> = Arc::new((0..=255u8).cycle().take(1000).collect());
        let log = Arc::new(Mutex::new(Vec::new()));
        mount_lazy_with(path, 1000, WINDOW, 2, recording(whole.clone(), log.clone()));
        assert!(is_mounted(path));
        assert!(get(path).is_none(), "nothing is held");
        assert_eq!(len(path), Some(1000));
        assert_eq!(lazy_stats(path), Some(LazyStats::default()));

        let chunks = chunks(path).unwrap();
        assert!(chunks.is_lazy());
        assert!(chunks.parts().is_empty());
        assert!(chunks.contiguous().is_none());
        assert!(chunks.single().is_none());
        assert!(!chunks.is_contiguous(0, 1));
        assert!(chunks.is_contiguous(0, 0), "an empty window has a view");
        assert!(chunks.is_contiguous(1000, 0));
        assert!(!chunks.is_contiguous(1000, 1));
        assert!(matches!(chunks.slice(0, 0).unwrap(), Cow::Borrowed(_)));
        assert!(chunks.slice(1000, 1).is_none());
        assert!(chunks.read(999, 2).is_none());
        assert_eq!(chunks.lazy_stats(), Some(LazyStats::default()));

        // A small read fetches its whole window and keeps it.
        assert_eq!(chunks.read(10, 5).unwrap(), &whole[10..15]);
        assert_eq!(log.lock().unwrap().as_slice(), &[(0, 64)]);
        assert_eq!(chunks.cached_bytes(), 64);
        // Its neighbours hit the cache.
        assert_eq!(chunks.read(0, 64).unwrap(), &whole[0..64]);
        assert_eq!(chunks.read(63, 1).unwrap(), &whole[63..64]);
        assert_eq!(log.lock().unwrap().len(), 1);
        // One crossing a window boundary fetches the other window too.
        assert_eq!(chunks.read(60, 8).unwrap(), &whole[60..68]);
        assert_eq!(log.lock().unwrap().as_slice(), &[(0, 64), (64, 64)]);
        assert_eq!(
            chunks.lazy_stats(),
            Some(LazyStats {
                requests: 2,
                bytes: 128
            })
        );

        // A wide read: the head window (64) is already cached, the whole
        // windows inside come in one request straight through, and the
        // partial tail window is fetched whole and kept.
        log.lock().unwrap().clear();
        assert_eq!(chunks.read(100, 300).unwrap(), &whole[100..400]);
        assert_eq!(
            log.lock().unwrap().as_slice(),
            &[(128, 256), (384, 64)],
            "one run of whole windows, then the tail window"
        );
        // The tail window stays for the next span; the first window was
        // evicted (two unpinned windows are kept: 64 and 384).
        log.lock().unwrap().clear();
        assert_eq!(chunks.read(400, 40).unwrap(), &whole[400..440]);
        assert!(log.lock().unwrap().is_empty(), "the tail window was kept");
        assert_eq!(chunks.read(0, 8).unwrap(), &whole[0..8]);
        assert_eq!(log.lock().unwrap().as_slice(), &[(0, 64)]);

        // Whole windows already cached inside a wide read (0 and 384) are
        // copied, and the run of fetched windows is split around them.
        log.lock().unwrap().clear();
        assert_eq!(chunks.read(0, 512).unwrap(), &whole[0..512]);
        let served = log.lock().unwrap().clone();
        assert_eq!(served, vec![(64, 320), (448, 64)]);

        // The last window, shorter than the rest, reads back whole; read
        // whole it goes straight through, read in part it is kept.
        log.lock().unwrap().clear();
        assert_eq!(chunks.read(960, 40).unwrap(), &whole[960..]);
        assert_eq!(log.lock().unwrap().as_slice(), &[(960, 40)]);
        assert_eq!(chunks.read(990, 10).unwrap(), &whole[990..]);
        assert_eq!(log.lock().unwrap().as_slice(), &[(960, 40), (960, 40)]);
        assert_eq!(chunks.read(990, 10).unwrap(), &whole[990..]);
        assert_eq!(log.lock().unwrap().len(), 2);
        assert_eq!(chunks.read(896, 104).unwrap(), &whole[896..]);
        assert_eq!(
            log.lock().unwrap().as_slice(),
            &[(960, 40), (960, 40), (896, 64)]
        );

        // Every offset and length reads the same bytes as the source.
        for (offset, n) in [
            (0u64, 1000u64),
            (0, 1),
            (63, 3),
            (64, 64),
            (65, 63),
            (1, 999),
            (127, 130),
            (999, 1),
            (1000, 0),
        ] {
            let expect = &whole[offset as usize..(offset + n) as usize];
            assert_eq!(
                &*chunks.slice(offset, n).unwrap(),
                expect,
                "slice {offset}+{n}"
            );
            if n > 0 {
                assert!(matches!(chunks.slice(offset, n).unwrap(), Cow::Owned(_)));
            }
            assert_eq!(chunks.read(offset, n).unwrap(), expect, "read {offset}+{n}");
            let mut into = vec![0u8; n as usize];
            read(path, offset, &mut into).unwrap();
            assert_eq!(into, expect, "memfs::read {offset}+{n}");
            assert_eq!(
                with_slice(path, offset, n, |w| w.to_vec()).unwrap(),
                expect,
                "memfs::with_slice {offset}+{n}"
            );
        }
        // The cache never grows past the kept windows plus the pinned tail.
        assert!(
            chunks.cached_bytes() <= 2 * WINDOW + 2 * WINDOW,
            "{} bytes cached",
            chunks.cached_bytes()
        );
        assert!(error(path).is_none());

        let taken = unmount(path).unwrap();
        assert!(taken.into_parts().is_empty());
        assert!(!is_mounted(path));
    }

    fn a_lazy_mount_keeps_its_tail_and_reports_failures() {
        const WINDOW: u64 = 100;
        let path = Path::new("/no/such/dir/memfs-lazy-tail-probe.zt");
        let whole: Arc<Vec<u8>> = Arc::new((0..=255u8).cycle().take(1000).collect());
        let log = Arc::new(Mutex::new(Vec::new()));
        mount_lazy_with(path, 1000, WINDOW, 2, recording(whole.clone(), log.clone()));
        let chunks = chunks(path).unwrap();

        // The footer and the manifest before it are read at every open.
        assert_eq!(chunks.read(992, 8).unwrap(), &whole[992..]);
        assert_eq!(chunks.read(850, 100).unwrap(), &whole[850..950]);
        assert_eq!(log.lock().unwrap().len(), 2, "two tail windows");
        // Reads elsewhere churn the unpinned windows...
        for at in (0..800).step_by(100) {
            assert_eq!(
                chunks.read(at + 1, 2).unwrap(),
                &whole[at as usize + 1..at as usize + 3]
            );
        }
        assert_eq!(log.lock().unwrap().len(), 10);
        // ...and the tail still hits.
        assert_eq!(chunks.read(992, 8).unwrap(), &whole[992..]);
        assert_eq!(chunks.read(850, 100).unwrap(), &whole[850..950]);
        assert_eq!(
            log.lock().unwrap().len(),
            10,
            "the tail windows were pinned"
        );
        assert_eq!(chunks.cached_bytes(), 400, "two kept plus two pinned");

        // A fetch that fails refuses the read and is reported.
        let failing = Path::new("/no/such/dir/memfs-lazy-failing-probe.zt");
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        mount_lazy(
            failing,
            1 << 30,
            Box::new(move |offset: u64, into: &mut [u8]| {
                seen.fetch_add(1, Ordering::SeqCst);
                if offset >= LAZY_WINDOW {
                    return Err(format!("no bytes at {offset}"));
                }
                into.fill(7);
                Ok(())
            }),
        );
        let mut into = [0u8; 3];
        assert_eq!(read(failing, 5, &mut into), Some(()));
        assert_eq!(into, [7, 7, 7]);
        assert_eq!(read(failing, LAZY_WINDOW + 1, &mut into), None);
        assert_eq!(error(failing).as_deref(), Some("no bytes at 8388608"));
        assert_eq!(
            read_failure(failing),
            " (the last fetch failed: no bytes at 8388608)"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let stats = lazy_stats(failing).unwrap();
        assert_eq!(stats.requests, 2);
        assert_eq!(stats.bytes, 2 * LAZY_WINDOW);
        assert!(with_slice(failing, LAZY_WINDOW, 1, |_| ()).is_none());
        unmount(failing);
        unmount(path);
    }
}
