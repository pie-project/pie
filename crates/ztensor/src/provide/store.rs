use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};

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

pub struct Store {
    path: PathBuf,
    file: File,
    len: u64,
    format: &'static str,
    map: Option<Mmap>,
    occupied: Vec<(u64, u64)>,
    decoder: Option<Box<dyn Decode>>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("path", &self.path)
            .field("format", &self.format)
            .field("len", &self.len)
            .field("mapped", &self.map.is_some())
            .finish()
    }
}

impl Store {
    pub fn map(path: impl AsRef<Path>, format: &'static str) -> Result<Self> {
        let mut store = Self::index(path, format)?;
        // SAFETY: read-only shared map; the contents are treated as untrusted
        // bytes and never assumed stable beyond the validation snapshot.
        store.map = Some(unsafe { Mmap::map(&store.file)? });
        Ok(store)
    }

    pub fn index(path: impl AsRef<Path>, format: &'static str) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            path,
            file,
            len,
            format,
            map: None,
            occupied: Vec::new(),
            decoder: None,
        })
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

    pub fn is_mapped(&self) -> bool {
        self.map.is_some()
    }

    pub fn bytes(&self) -> Option<&[u8]> {
        self.map.as_deref()
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

    pub fn slice(&self, offset: u64, len: u64) -> Result<Option<&[u8]>> {
        let (start, end) = self.bounded(offset, len)?;
        Ok(self.map.as_ref().map(|m| &m[start..end]))
    }

    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let (start, end) = self.bounded(offset, len)?;
        if let Some(map) = &self.map {
            return Ok(map[start..end].to_vec());
        }
        let mut buf = vec![0u8; end - start];
        read_exact_at(&self.file, &mut buf, offset)?;
        Ok(buf)
    }

    pub(crate) fn decoder(&self) -> Option<&dyn Decode> {
        self.decoder.as_deref()
    }

    pub fn page_exclusive(&self, offset: u64, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        if self.occupied.is_empty() {
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

    pub fn prefetch(&self, offset: u64, len: u64) -> Result<()> {
        let (start, end) = self.bounded(offset, len)?;
        #[cfg(unix)]
        if let Some(map) = &self.map {
            if end > start {
                map.advise_range(memmap2::Advice::WillNeed, start, end - start)?;
            }
        }
        let _ = (start, end);
        Ok(())
    }

    pub fn evict(&self, offset: u64, len: u64) -> Result<()> {
        let (_, _) = self.bounded(offset, len)?;
        #[cfg(not(unix))]
        {
            let _ = (offset, len);
            return Err(Error::Unsupported(
                "dropping page cache is a unix facility".into(),
            ));
        }
        #[cfg(unix)]
        {
            let Some(map) = &self.map else {
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
