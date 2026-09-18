use std::borrow::Cow;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::memfs::{self, Chunks};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoreId(pub u32);

impl std::fmt::Display for StoreId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

pub trait Decode: Send + Sync {
    fn decode(&self, key: u64, decoded_len: u64) -> Result<Vec<u8>>;
}

/// Where a store's bytes live. A file is read through the handle or, once
/// mapped, through the map; a memory mount is its own map from the start,
/// in one chunk or several laid end to end.
enum Backing {
    File { file: File, map: Option<Mmap> },
    Memory(Chunks),
}

pub struct Store {
    path: PathBuf,
    backing: Backing,
    len: u64,
    format: &'static str,
    occupied: Vec<(u64, u64)>,
    decoder: Option<Box<dyn Decode>>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("path", &self.path)
            .field("format", &self.format)
            .field("len", &self.len)
            .field("mapped", &self.is_mapped())
            .field("memory", &self.is_memory())
            .finish()
    }
}

impl Store {
    pub fn map(path: impl AsRef<Path>, format: &'static str) -> Result<Self> {
        let mut store = Self::index(path, format)?;
        if let Backing::File { file, map } = &mut store.backing {
            // SAFETY: read-only shared map; the contents are treated as untrusted
            // bytes and never assumed stable beyond the validation snapshot.
            *map = Some(unsafe { Mmap::map(&*file)? });
        }
        Ok(store)
    }

    /// A mount under `path` wins over a file at `path`, so the path a caller
    /// already holds keeps working on a platform that has no file there.
    pub fn index(path: impl AsRef<Path>, format: &'static str) -> Result<Self> {
        let path = path.as_ref();
        if let Some(chunks) = memfs::chunks(path) {
            return Ok(Self::from_memory(path, chunks, format));
        }
        let path = path.to_path_buf();
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            path,
            backing: Backing::File { file, map: None },
            len,
            format,
            occupied: Vec::new(),
            decoder: None,
        })
    }

    /// A store over bytes already in memory — one `Arc<[u8]>`, or several
    /// chunks laid end to end; `path` is only its name.
    pub fn from_memory(
        path: impl AsRef<Path>,
        bytes: impl Into<Chunks>,
        format: &'static str,
    ) -> Self {
        let chunks = bytes.into();
        Self {
            path: path.as_ref().to_path_buf(),
            len: chunks.len(),
            backing: Backing::Memory(chunks),
            format,
            occupied: Vec::new(),
            decoder: None,
        }
    }

    pub fn with_occupied(mut self, mut ranges: Vec<(u64, u64)>) -> Self {
        ranges.sort_unstable();
        ranges.dedup();
        self.occupied = ranges;
        self
    }

    pub fn with_decoder(mut self, decoder: Box<dyn Decode>) -> Self {
        self.decoder = Some(decoder);
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn format(&self) -> &'static str {
        self.format
    }

    /// True when every byte is addressable without a read: the file is
    /// mapped, or the store is a memory mount.
    pub fn is_mapped(&self) -> bool {
        match &self.backing {
            Backing::File { map, .. } => map.is_some(),
            Backing::Memory(_) => true,
        }
    }

    /// True when the bytes are an in-memory mount rather than a file; page
    /// advice and eviction mean nothing for such a store.
    pub fn is_memory(&self) -> bool {
        matches!(self.backing, Backing::Memory(_))
    }

    /// True when the window has a borrowed view: the file is mapped, or the
    /// window lies inside one chunk of a memory mount.
    pub fn is_contiguous(&self, offset: u64, len: u64) -> bool {
        match &self.backing {
            Backing::File { map, .. } => map.is_some(),
            Backing::Memory(chunks) => chunks.is_contiguous(offset, len),
        }
    }

    /// The whole store as one borrowed slice: a mapped file, or a memory
    /// mount that is a single chunk. A chunked mount has no such view; its
    /// bytes come through `slice` and `read`.
    pub fn bytes(&self) -> Option<&[u8]> {
        match &self.backing {
            Backing::File { map, .. } => map.as_deref(),
            Backing::Memory(chunks) => chunks.contiguous(),
        }
    }

    fn bounded(&self, offset: u64, len: u64) -> Result<(usize, usize)> {
        let end = offset
            .checked_add(len)
            .filter(|&e| e <= self.len)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "range {offset}+{len} is outside {} ({} bytes)",
                    self.path.display(),
                    self.len
                ))
            })?;
        Ok((offset as usize, end as usize))
    }

    /// The window without a read, when the store is mapped: borrowed from
    /// the map or the chunk it lies in, copied when it straddles two chunks
    /// of a memory mount. None for an unmapped file.
    pub fn slice(&self, offset: u64, len: u64) -> Result<Option<Cow<'_, [u8]>>> {
        let (start, end) = self.bounded(offset, len)?;
        Ok(match &self.backing {
            Backing::File { map, .. } => map.as_deref().map(|m| Cow::Borrowed(&m[start..end])),
            Backing::Memory(chunks) => Some(chunks.slice(offset, len).ok_or_else(|| {
                Error::Unsupported(format!(
                    "range {offset}+{len} is outside the {} mounted bytes of {}{}",
                    chunks.len(),
                    self.path.display(),
                    fetch_failure(chunks)
                ))
            })?),
        })
    }

    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let (start, end) = self.bounded(offset, len)?;
        match &self.backing {
            Backing::File { map: Some(map), .. } => Ok(map[start..end].to_vec()),
            Backing::File { file, map: None } => {
                let mut buf = vec![0u8; end - start];
                read_exact_at(file, &mut buf, offset)?;
                Ok(buf)
            }
            Backing::Memory(chunks) => chunks.read(offset, len).ok_or_else(|| {
                Error::Unsupported(format!(
                    "range {offset}+{len} is outside the {} mounted bytes of {}{}",
                    chunks.len(),
                    self.path.display(),
                    fetch_failure(chunks)
                ))
            }),
        }
    }

    pub(crate) fn decoder(&self) -> Option<&dyn Decode> {
        self.decoder.as_deref()
    }

    pub fn page_exclusive(&self, offset: u64, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        if self.occupied.is_empty() || self.is_memory() {
            return false;
        }
        let page = page_size();
        let (env_start, env_end) = page_envelope(offset, len, page);
        let Ok(i) = self.occupied.binary_search(&(offset, len)) else {
            return false;
        };
        let prev_clear = i == 0 || {
            let (o, l) = self.occupied[i - 1];
            o + l <= env_start
        };
        let next_clear = i + 1 >= self.occupied.len() || self.occupied[i + 1].0 >= env_end;
        prev_clear && next_clear
    }

    #[cfg(unix)]
    fn mmap(&self) -> Option<&Mmap> {
        match &self.backing {
            Backing::File { map, .. } => map.as_ref(),
            Backing::Memory(_) => None,
        }
    }

    pub fn prefetch(&self, offset: u64, len: u64) -> Result<()> {
        let (start, end) = self.bounded(offset, len)?;
        #[cfg(unix)]
        if let Some(map) = self.mmap() {
            if end > start {
                map.advise_range(memmap2::Advice::WillNeed, start, end - start)?;
            }
        }
        let _ = (start, end);
        Ok(())
    }

    // On a target without the unix arm the refusal is the tail expression,
    // which clippy would like written without `return`; the `return` is what
    // keeps the two arms readable side by side.
    #[allow(clippy::needless_return)]
    pub fn evict(&self, offset: u64, len: u64) -> Result<()> {
        let (_, _) = self.bounded(offset, len)?;
        if self.is_memory() {
            return Ok(());
        }
        #[cfg(not(unix))]
        {
            let _ = (offset, len);
            return Err(Error::Unsupported(
                "dropping page cache is a unix facility".into(),
            ));
        }
        #[cfg(unix)]
        {
            let Some(map) = self.mmap() else {
                return Ok(());
            };
            if len == 0 {
                return Ok(());
            }
            let (start, end) = page_envelope(offset, len, page_size());
            let end = end.min(map.len() as u64);
            // SAFETY: the map is a read-only shared file mapping, so DontNeed
            // only drops clean page-cache pages; later accesses re-fault from
            // the file. It cannot discard writes because none exist.
            unsafe {
                map.unchecked_advise_range(
                    memmap2::UncheckedAdvice::DontNeed,
                    start as usize,
                    (end - start) as usize,
                )?;
            }
            Ok(())
        }
    }
}

/// What a refused read of a lazy mount appends: the fetch that failed.
fn fetch_failure(chunks: &Chunks) -> String {
    match chunks.error() {
        Some(why) => format!(" (the last fetch failed: {why})"),
        None => String::new(),
    }
}

pub(crate) fn page_envelope(offset: u64, length: u64, page: u64) -> (u64, u64) {
    (
        offset & !(page - 1),
        (offset + length).div_ceil(page).saturating_mul(page),
    )
}

pub fn page_size() -> u64 {
    #[cfg(unix)]
    {
        // SAFETY: sysconf is always safe to call.
        let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if n > 0 {
            return n as u64;
        }
    }
    4096
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(not(unix))]
    {
        read_exact_at_portable(file, buf, offset)
    }
}

#[cfg_attr(unix, allow(dead_code))]
fn read_exact_at_portable(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut handle = file.try_clone()?;
    handle.seek(SeekFrom::Start(offset))?;
    handle.read_exact(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn store_with(ranges: &[(u64, u64)]) -> Store {
        let path = std::env::temp_dir().join("ztensor-store-exclusivity-probe");
        std::fs::write(&path, [0u8; 1]).unwrap();
        Store::index(&path, "zt")
            .unwrap()
            .with_occupied(ranges.to_vec())
    }

    #[test]
    fn store_every_case() {
        the_portable_read_path_reads_the_same_bytes();
        page_exclusivity();
        a_mounted_path_opens_as_memory();
        a_chunked_mount_reads_across_its_seams();
        a_lazy_mount_reads_by_fetching();
    }

    fn a_lazy_mount_reads_by_fetching() {
        let path = Path::new("/no/such/dir/ztensor-store-lazy-probe.zt");
        let content: Arc<Vec<u8>> = Arc::new((0..=255u8).cycle().take(4096).collect());
        let source = Arc::clone(&content);
        memfs::mount_lazy_windowed(
            path,
            4096,
            1000,
            Box::new(move |offset: u64, into: &mut [u8]| {
                let at = offset as usize;
                into.copy_from_slice(&source[at..at + into.len()]);
                Ok(())
            }),
        );

        let store = Store::map(path, "zt").unwrap();
        assert!(store.is_memory());
        assert!(store.is_mapped(), "a lazy mount answers every address");
        assert!(store.bytes().is_none(), "nothing is held to borrow");
        assert_eq!(store.len(), 4096);
        for (offset, len) in [(0u64, 4096u64), (999, 2), (1000, 1000), (0, 0), (4096, 0)] {
            let expect = &content[offset as usize..(offset + len) as usize];
            assert_eq!(&*store.slice(offset, len).unwrap().unwrap(), expect);
            assert_eq!(store.read(offset, len).unwrap(), expect);
        }
        assert!(matches!(
            store.slice(1000, 1000).unwrap().unwrap(),
            Cow::Owned(_)
        ));
        assert!(matches!(
            store.slice(1000, 0).unwrap().unwrap(),
            Cow::Borrowed(_)
        ));
        assert!(!store.is_contiguous(1000, 1000));
        assert!(store.is_contiguous(1000, 0));
        assert!(store.read(4090, 32).is_err());
        assert!(store.slice(4090, 32).is_err());
        assert!(store.prefetch(0, 4096).is_ok());
        assert!(store.evict(0, 4096).is_ok());
        assert!(!store.with_occupied(vec![(0, 8)]).page_exclusive(0, 8));
        let stats = memfs::lazy_stats(path).unwrap();
        assert!(stats.requests > 0 && stats.bytes >= 4096, "{stats:?}");
        memfs::unmount(path);
    }

    fn a_chunked_mount_reads_across_its_seams() {
        let path = Path::new("/no/such/dir/ztensor-store-chunked-probe.zt");
        let content: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let parts: Vec<Arc<[u8]>> = content.chunks(1000).map(Arc::from).collect();
        memfs::mount_chunks(path, parts.clone());

        let store = Store::map(path, "zt").unwrap();
        assert!(store.is_memory());
        assert!(store.is_mapped(), "memory is its own map, chunked or not");
        assert!(store.bytes().is_none(), "no single slice spans five chunks");
        assert_eq!(store.len(), 4096);
        for (offset, len) in [(0u64, 4096u64), (999, 2), (1000, 1000), (0, 0), (4096, 0)] {
            let expect = &content[offset as usize..(offset + len) as usize];
            assert_eq!(&*store.slice(offset, len).unwrap().unwrap(), expect);
            assert_eq!(store.read(offset, len).unwrap(), expect);
        }
        assert!(matches!(
            store.slice(999, 2).unwrap().unwrap(),
            Cow::Owned(_)
        ));
        assert!(matches!(
            store.slice(1000, 1000).unwrap().unwrap(),
            Cow::Borrowed(_)
        ));
        assert!(store.read(4090, 32).is_err());
        assert!(store.slice(4090, 32).is_err());
        assert!(store.prefetch(0, 4096).is_ok());
        assert!(store.evict(0, 4096).is_ok());

        let direct = Store::from_memory(path, parts, "zt");
        assert_eq!(direct.read(0, 4096).unwrap(), content);
        memfs::unmount(path);
    }

    fn a_mounted_path_opens_as_memory() {
        let path = Path::new("/no/such/dir/ztensor-store-memory-probe.zt");
        let content: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        memfs::mount(path, Arc::from(content.as_slice()));

        let indexed = Store::index(path, "zt").unwrap();
        assert!(indexed.is_memory());
        assert!(indexed.is_mapped(), "memory is its own map");
        assert_eq!(indexed.len(), 4096);
        assert_eq!(indexed.path(), path);
        assert_eq!(
            &*indexed.slice(1000, 100).unwrap().unwrap(),
            &content[1000..1100]
        );
        assert_eq!(indexed.read(4080, 16).unwrap(), &content[4080..]);
        assert!(indexed.read(4090, 32).is_err());
        assert!(indexed.prefetch(0, 4096).is_ok());
        assert!(indexed.evict(0, 4096).is_ok());
        assert!(!indexed.with_occupied(vec![(0, 8)]).page_exclusive(0, 8));

        let mapped = Store::map(path, "zt").unwrap();
        assert_eq!(mapped.bytes().unwrap(), content.as_slice());

        memfs::unmount(path);
        assert!(
            Store::index(path, "zt").is_err(),
            "no file stands behind the name"
        );
        assert_eq!(
            mapped.bytes().unwrap(),
            content.as_slice(),
            "an open store keeps its bytes"
        );
    }

    fn the_portable_read_path_reads_the_same_bytes() {
        let path = std::env::temp_dir().join("ztensor-portable-read-probe");
        let content: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        std::fs::write(&path, &content).unwrap();
        let file = File::open(&path).unwrap();

        for (offset, len) in [(0u64, 16usize), (1, 3), (1000, 100), (4080, 16)] {
            let mut portable = vec![0u8; len];
            read_exact_at_portable(&file, &mut portable, offset).unwrap();
            let mut platform = vec![0u8; len];
            read_exact_at(&file, &mut platform, offset).unwrap();
            let expect = &content[offset as usize..offset as usize + len];
            assert_eq!(portable, expect, "portable read at {offset}+{len}");
            assert_eq!(platform, expect, "platform read at {offset}+{len}");
        }

        let mut buf = [0u8; 32];
        assert!(read_exact_at_portable(&file, &mut buf, 4090).is_err());
        let _ = std::fs::remove_file(&path);
    }

    fn page_exclusivity() {
        let s = store_with(&[(0, 8), (4096, 8), (8192, 100), (12288, 340), (12628, 40)]);
        let page = page_size();
        if page <= 4096 {
            assert!(s.page_exclusive(4096, 8));
            assert!(s.page_exclusive(8192, 100));
        }
        let canonical = store_with(&[(0, 8), (65536, 8), (131072, 100), (196608, 380)]);
        assert!(canonical.page_exclusive(65536, 8));
        assert!(canonical.page_exclusive(131072, 100));
        let packed = store_with(&[(4096, 100), (4200, 50)]);
        assert!(!packed.page_exclusive(4096, 100));
        assert!(!packed.page_exclusive(4200, 50));
        assert!(s.page_exclusive(4096, 0));
        assert!(!s.page_exclusive(20480, 8));
        assert!(!store_with(&[]).page_exclusive(65536, 8));
    }
}
