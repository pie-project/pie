//! Sealed pages in host memory, by key, least recently used first to go;
//! and, given a directory, on disk too, so a pool that restarts comes back
//! with what it held. A page on disk is one file named by its key: its
//! layout's length and layout, then its bytes, written whole before it is
//! named.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use transport::tcp::{Answer, Key, Pages, Source, Stream, Taken};

struct Page {
    bytes: Vec<u8>,
    meta: Arc<Vec<u8>>,
    used: u64,
}

#[derive(Default)]
struct Inner {
    pages: HashMap<Key, Page>,
    /// `used` → key, oldest first.
    order: BTreeMap<u64, Key>,
    clock: u64,
    bytes: usize,
}

impl Inner {
    fn touch(&mut self, key: &Key) {
        self.clock += 1;
        let clock = self.clock;
        if let Some(page) = self.pages.get_mut(key) {
            self.order.remove(&page.used);
            page.used = clock;
            self.order.insert(clock, *key);
        }
    }

    fn remove(&mut self, key: &Key) {
        if let Some(page) = self.pages.remove(key) {
            self.order.remove(&page.used);
            self.bytes -= page.bytes.len();
        }
    }
}

/// Pages kept up to `capacity` bytes in memory, and every page sent on
/// disk where there is a directory.
#[derive(Clone)]
pub struct Pool {
    inner: Arc<Mutex<Inner>>,
    capacity: usize,
    disk: Option<Arc<Disk>>,
}

struct Disk {
    dir: std::path::PathBuf,
    keys: Mutex<std::collections::HashSet<Key>>,
}

fn hex(key: &Key) -> String {
    key.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(name: &str) -> Option<Key> {
    if name.len() != 64 {
        return None;
    }
    let mut key = [0u8; 32];
    for (i, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(name.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(key)
}

impl Disk {
    fn open(dir: std::path::PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        let mut keys = std::collections::HashSet::new();
        for entry in std::fs::read_dir(&dir)? {
            let name = entry?.file_name();
            if let Some(key) = name.to_str().and_then(unhex) {
                keys.insert(key);
            }
        }
        Ok(Self {
            dir,
            keys: Mutex::new(keys),
        })
    }

    fn write(&self, key: &Key, meta: &[u8], bytes: &[u8]) -> std::io::Result<()> {
        use std::io::Write;
        let name = hex(key);
        let tmp = self.dir.join(format!(".{name}.tmp"));
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&(meta.len() as u32).to_le_bytes())?;
        file.write_all(meta)?;
        file.write_all(bytes)?;
        file.sync_data()?;
        std::fs::rename(&tmp, self.dir.join(name))?;
        self.keys.lock().unwrap().insert(*key);
        Ok(())
    }

    fn read(&self, key: &Key) -> Option<(Vec<u8>, Vec<u8>)> {
        let all = std::fs::read(self.dir.join(hex(key))).ok()?;
        let len = u32::from_le_bytes(all.get(..4)?.try_into().ok()?) as usize;
        let meta = all.get(4..4 + len)?.to_vec();
        Some((meta, all[4 + len..].to_vec()))
    }
}

impl Pool {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::default(),
            capacity,
            disk: None,
        }
    }

    /// A pool that also keeps every page it is sent under `dir`, and holds
    /// what the directory held when it opens.
    pub fn with_dir(capacity: usize, dir: std::path::PathBuf) -> std::io::Result<Self> {
        let disk = Disk::open(dir)?;
        tracing::info!(
            pages = disk.keys.lock().unwrap().len(),
            "pool opened its directory"
        );
        Ok(Self {
            inner: Arc::default(),
            capacity,
            disk: Some(Arc::new(disk)),
        })
    }

    /// Keeps `pages`, one per key, letting the least recently used go to
    /// make room; refuses what is larger than the pool.
    pub fn put(&self, keys: &[Key], pages: Pages) -> Result<(), String> {
        let page_bytes = pages.page_bytes();
        if keys.len() != pages.count || page_bytes * pages.count != pages.data.len() {
            return Err("the pages do not match their keys".into());
        }
        // With a directory every page is kept on disk and memory caches
        // what fits; without one, more than memory holds is refused.
        if self.disk.is_none() && page_bytes * keys.len() > self.capacity {
            return Err(format!(
                "{} bytes are more than this pool holds",
                page_bytes * keys.len()
            ));
        }
        if let Some(disk) = &self.disk {
            for (key, bytes) in keys.iter().zip(pages.data.chunks(page_bytes.max(1))) {
                disk.write(key, &pages.meta, bytes)
                    .map_err(|e| format!("writing a page to the pool's directory: {e}"))?;
            }
        }
        let meta = Arc::new(pages.meta);
        let mut inner = self.inner.lock().unwrap();
        for (key, bytes) in keys.iter().zip(pages.data.chunks(page_bytes.max(1))) {
            inner.remove(key);
            if bytes.len() > self.capacity {
                continue;
            }
            while inner.bytes + bytes.len() > self.capacity {
                let Some((_, oldest)) = inner.order.pop_first() else {
                    break;
                };
                if let Some(page) = inner.pages.remove(&oldest) {
                    inner.bytes -= page.bytes.len();
                }
            }
            inner.clock += 1;
            let used = inner.clock;
            inner.bytes += bytes.len();
            inner.order.insert(used, *key);
            inner.pages.insert(
                *key,
                Page {
                    bytes: bytes.to_vec(),
                    meta: meta.clone(),
                    used,
                },
            );
        }
        Ok(())
    }

    /// Pages `keys[from..]`, if all of them are here and laid out alike.
    pub fn get(&self, keys: &[Key], from: usize) -> Result<Pages, String> {
        let mut inner = self.inner.lock().unwrap();
        let want = keys.get(from..).ok_or("past the run")?;
        let mut data = Vec::new();
        let mut meta: Option<Arc<Vec<u8>>> = None;
        for (at, key) in want.iter().enumerate() {
            let (page_meta, bytes): (Arc<Vec<u8>>, std::borrow::Cow<'_, [u8]>) =
                match inner.pages.get(key) {
                    Some(page) => (
                        page.meta.clone(),
                        std::borrow::Cow::Owned(page.bytes.clone()),
                    ),
                    None => {
                        let (meta, bytes) = self
                            .disk
                            .as_ref()
                            .and_then(|disk| disk.read(key))
                            .ok_or_else(|| format!("page {} of the run is not here", from + at))?;
                        (Arc::new(meta), std::borrow::Cow::Owned(bytes))
                    }
                };
            match &meta {
                Some(meta) if **meta != *page_meta => {
                    return Err("the run's pages are laid out differently".into());
                }
                Some(_) => {}
                None => meta = Some(page_meta),
            }
            data.extend_from_slice(&bytes);
        }
        for key in want {
            inner.touch(key);
        }
        Ok(Pages {
            data,
            count: want.len(),
            meta: meta.map(|meta| (*meta).clone()).unwrap_or_default(),
        })
    }

    /// Every key held, in memory or on disk: what the pool reports to the
    /// directory.
    pub fn keys(&self) -> Vec<Key> {
        let mut keys: std::collections::HashSet<Key> =
            self.inner.lock().unwrap().pages.keys().copied().collect();
        if let Some(disk) = &self.disk {
            keys.extend(disk.keys.lock().unwrap().iter().copied());
        }
        keys.into_iter().collect()
    }

    pub fn bytes(&self) -> usize {
        self.inner.lock().unwrap().bytes
    }
}

/// Reads and writes may touch the disk: off the async threads.
impl Source for Pool {
    fn read(&self, keys: Vec<Key>, from: u32) -> Answer {
        let pool = self.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || pool.get(&keys, from as usize))
                .await
                .map_err(|e| e.to_string())?
                .map(Stream::of)
        })
    }

    fn write(&self, keys: Vec<Key>, pages: Pages) -> Taken {
        let pool = self.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || pool.put(&keys, pages))
                .await
                .map_err(|e| e.to_string())?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(n: u8, count: usize) -> Pages {
        Pages {
            data: vec![n; 8 * count],
            count,
            meta: b"layout".to_vec(),
        }
    }

    #[test]
    fn a_run_comes_back_whole_and_the_oldest_go_first() {
        let pool = Pool::new(32);
        let keys = [[1u8; 32], [2u8; 32], [3u8; 32]];
        pool.put(&keys, pages(9, 3)).unwrap();
        let got = pool.get(&keys, 1).unwrap();
        assert_eq!(
            (got.count, got.data.len(), got.meta.as_slice()),
            (2, 16, &b"layout"[..])
        );
        // Reading pages 1.. made page 0 the oldest: a fifth page's room is
        // made from it.
        pool.put(&[[4u8; 32], [5u8; 32]], pages(7, 2)).unwrap();
        assert_eq!(pool.bytes(), 32);
        assert!(pool.get(&keys, 0).is_err(), "page 0 went");
        assert!(pool.get(&keys, 1).is_ok(), "and only it: the rest fit");
        assert!(
            pool.put(&keys, pages(1, 5)).is_err(),
            "more than the pool holds"
        );
    }

    #[test]
    fn a_pool_with_a_directory_comes_back_with_what_it_held() {
        let dir = std::env::temp_dir().join(format!("pie-pool-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let keys = [[1u8; 32], [2u8; 32]];
        {
            let pool = Pool::with_dir(8, dir.clone()).unwrap();
            pool.put(&keys, pages(3, 2)).unwrap();
            // Memory holds one page; the other is on disk alone.
            assert_eq!(pool.bytes(), 8);
            assert_eq!(pool.get(&keys, 0).unwrap().data, vec![3u8; 16]);
        }
        let reopened = Pool::with_dir(8, dir.clone()).unwrap();
        let mut held = reopened.keys();
        held.sort();
        assert_eq!(held, keys.to_vec());
        let got = reopened.get(&keys, 0).unwrap();
        assert_eq!((got.count, got.meta.as_slice()), (2, &b"layout"[..]));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
