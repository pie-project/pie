use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

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

pub const LAZY_WINDOW: u64 = 8 << 20;

pub const LAZY_KEEP: usize = 8;

pub const LAZY_PIN_TAIL: u64 = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LazyStats {
    pub requests: u64,
    pub bytes: u64,
}

struct LazyCache {
    windows: Vec<(u64, Vec<u8>)>,
    error: Option<String>,
}

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

    fn fill(&self, start: u64) -> Result<Vec<u8>, String> {
        let end = start.saturating_add(self.window).min(self.len);
        let mut window = vec![0u8; (end - start) as usize];
        self.fetch(start, &mut window)?;
        Ok(window)
    }

    fn read_at(&self, offset: u64, into: &mut [u8]) -> Result<(), String> {
        let end = offset + into.len() as u64;
        let mut cursor = offset;
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

#[derive(Clone, Debug, Default)]
pub struct Chunks {
    parts: Vec<Arc<[u8]>>,
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

    pub fn lazy(len: u64, window: u64, fetch: Box<dyn Fetch>) -> Self {
        Self::lazy_keeping(len, window, LAZY_KEEP, fetch)
    }

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

    pub fn is_lazy(&self) -> bool {
        self.lazy.is_some()
    }

    pub fn lazy_stats(&self) -> Option<LazyStats> {
        self.lazy.as_ref().map(|lazy| lazy.stats())
    }

    pub fn cached_bytes(&self) -> u64 {
        self.lazy.as_ref().map_or(0, |lazy| lazy.cached_bytes())
    }

    pub fn error(&self) -> Option<String> {
        self.lazy
            .as_ref()
            .and_then(|lazy| lazy.lock().error.clone())
    }

    pub fn parts(&self) -> &[Arc<[u8]>] {
        &self.parts
    }

    pub fn into_parts(self) -> Vec<Arc<[u8]>> {
        self.parts
    }

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

    fn locate(&self, offset: u64) -> Option<(usize, usize)> {
        if offset >= self.len() {
            return None;
        }
        let index = self.ends.partition_point(|&end| end <= offset);
        let start = if index == 0 { 0 } else { self.ends[index - 1] };
        Some((index, (offset - start) as usize))
    }

    fn in_range(&self, offset: u64, len: u64) -> bool {
        offset.checked_add(len).is_some_and(|end| end <= self.len())
    }

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

    pub fn read(&self, offset: u64, len: u64) -> Option<Vec<u8>> {
        if !self.in_range(offset, len) {
            return None;
        }
        let mut out = vec![0u8; usize::try_from(len).ok()?];
        self.read_at(offset, &mut out)?;
        Some(out)
    }

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

    pub fn is_contiguous(&self, offset: u64, len: u64) -> bool {
        self.inside_one(offset, len).is_some()
    }

    pub fn slice(&self, offset: u64, len: u64) -> Option<Cow<'_, [u8]>> {
        match self.inside_one(offset, len) {
            Some((_, range)) if range.is_empty() => Some(Cow::Borrowed(&[])),
            Some((index, range)) => Some(Cow::Borrowed(&self.parts[index][range])),
            None if self.in_range(offset, len) => self.read(offset, len).map(Cow::Owned),
            None => None,
        }
    }

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

pub fn mount(path: impl AsRef<Path>, bytes: Arc<[u8]>) {
    mount_chunks(path, vec![bytes]);
}

pub fn mount_chunks(path: impl AsRef<Path>, parts: Vec<Arc<[u8]>>) {
    table().insert(path.as_ref().to_path_buf(), Chunks::new(parts));
}

pub fn mount_lazy(path: impl AsRef<Path>, len: u64, fetch: Box<dyn Fetch>) {
    mount_lazy_windowed(path, len, LAZY_WINDOW, fetch);
}

pub fn mount_lazy_windowed(path: impl AsRef<Path>, len: u64, window: u64, fetch: Box<dyn Fetch>) {
    mount_lazy_with(path, len, window, LAZY_KEEP, fetch);
}

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

pub fn unmount(path: impl AsRef<Path>) -> Option<Chunks> {
    table().remove(path.as_ref())
}

pub fn get(path: impl AsRef<Path>) -> Option<Arc<[u8]>> {
    table().get(path.as_ref()).and_then(Chunks::single)
}

pub fn chunks(path: impl AsRef<Path>) -> Option<Chunks> {
    table().get(path.as_ref()).cloned()
}

pub fn len(path: impl AsRef<Path>) -> Option<u64> {
    table().get(path.as_ref()).map(Chunks::len)
}

pub fn read(path: impl AsRef<Path>, offset: u64, into: &mut [u8]) -> Option<()> {
    chunks(path)?.read_at(offset, into)
}

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

pub fn lazy_stats(path: impl AsRef<Path>) -> Option<LazyStats> {
    table().get(path.as_ref()).and_then(Chunks::lazy_stats)
}

pub fn error(path: impl AsRef<Path>) -> Option<String> {
    table().get(path.as_ref()).and_then(Chunks::error)
}

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

        assert_eq!(chunks.read(10, 5).unwrap(), &whole[10..15]);
        assert_eq!(log.lock().unwrap().as_slice(), &[(0, 64)]);
        assert_eq!(chunks.cached_bytes(), 64);
        assert_eq!(chunks.read(0, 64).unwrap(), &whole[0..64]);
        assert_eq!(chunks.read(63, 1).unwrap(), &whole[63..64]);
        assert_eq!(log.lock().unwrap().len(), 1);
        assert_eq!(chunks.read(60, 8).unwrap(), &whole[60..68]);
        assert_eq!(log.lock().unwrap().as_slice(), &[(0, 64), (64, 64)]);
        assert_eq!(
            chunks.lazy_stats(),
            Some(LazyStats {
                requests: 2,
                bytes: 128
            })
        );

        log.lock().unwrap().clear();
        assert_eq!(chunks.read(100, 300).unwrap(), &whole[100..400]);
        assert_eq!(
            log.lock().unwrap().as_slice(),
            &[(128, 256), (384, 64)],
            "one run of whole windows, then the tail window"
        );
        log.lock().unwrap().clear();
        assert_eq!(chunks.read(400, 40).unwrap(), &whole[400..440]);
        assert!(log.lock().unwrap().is_empty(), "the tail window was kept");
        assert_eq!(chunks.read(0, 8).unwrap(), &whole[0..8]);
        assert_eq!(log.lock().unwrap().as_slice(), &[(0, 64)]);

        log.lock().unwrap().clear();
        assert_eq!(chunks.read(0, 512).unwrap(), &whole[0..512]);
        let served = log.lock().unwrap().clone();
        assert_eq!(served, vec![(64, 320), (448, 64)]);

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

        assert_eq!(chunks.read(992, 8).unwrap(), &whole[992..]);
        assert_eq!(chunks.read(850, 100).unwrap(), &whole[850..950]);
        assert_eq!(log.lock().unwrap().len(), 2, "two tail windows");
        for at in (0..800).step_by(100) {
            assert_eq!(
                chunks.read(at + 1, 2).unwrap(),
                &whole[at as usize + 1..at as usize + 3]
            );
        }
        assert_eq!(log.lock().unwrap().len(), 10);
        assert_eq!(chunks.read(992, 8).unwrap(), &whole[992..]);
        assert_eq!(chunks.read(850, 100).unwrap(), &whole[850..950]);
        assert_eq!(
            log.lock().unwrap().len(),
            10,
            "the tail windows were pinned"
        );
        assert_eq!(chunks.cached_bytes(), 400, "two kept plus two pinned");

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
