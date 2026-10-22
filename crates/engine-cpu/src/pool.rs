//! Sealed pages in host memory, by key, the lowest retention key and then
//! the least recently used first to go; and, given a [`Store`], kept there
//! too, so a pool that restarts comes back with what it held. A page in a
//! store is one blob: `PIE2`, its layout's length and layout, its tokens'
//! hashes' count and hashes, then its bytes. (A blob without the mark holds
//! the layout and the bytes alone, as pools wrote before.)
//!
//! The runs a pool is asked to keep whole from token 0 are its groups, as
//! a worker's kept runs are its own. They are kept beside the pages, in the
//! store under [`RUNS`]: each run's length and keys, one after another.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::store::Store;

use transport::tcp::{Answer, Key, Pages, Source, Stream, Taken};

struct Page {
    bytes: Vec<u8>,
    meta: Arc<Vec<u8>>,
    /// The page's tokens' hashes; empty when it came without them.
    slots: Vec<Key>,
    /// The highest retention key of the runs it was sent in; 0 for a page
    /// sent in none.
    retention: i64,
    used: u64,
}

/// The key the runs are kept under in a store: no page's, which are
/// hashes.
pub const RUNS: Key = *b"pie-pool: the runs it keeps.....";

/// What a stored page begins with since pages carry their tokens' hashes.
const MARK: &[u8; 4] = b"PIE2";

#[derive(Default)]
struct Inner {
    pages: HashMap<Key, Page>,
    /// (`retention`, `used`) → key, the first to go first.
    order: BTreeMap<(i64, u64), Key>,
    /// The runs asked to be kept whole, by their last key.
    runs: HashMap<Key, Vec<Key>>,
    clock: u64,
    bytes: usize,
}

impl Inner {
    fn touch(&mut self, key: &Key) {
        self.clock += 1;
        let clock = self.clock;
        if let Some(page) = self.pages.get_mut(key) {
            self.order.remove(&(page.retention, page.used));
            page.used = clock;
            self.order.insert((page.retention, clock), *key);
        }
    }

    fn remove(&mut self, key: &Key) {
        if let Some(page) = self.pages.remove(key) {
            self.order.remove(&(page.retention, page.used));
            self.bytes -= page.bytes.len();
        }
    }
}

fn encode_runs<'a>(runs: impl Iterator<Item = &'a Vec<Key>>) -> Vec<u8> {
    let mut blob = Vec::new();
    for run in runs {
        blob.extend_from_slice(&(run.len() as u32).to_le_bytes());
        for key in run {
            blob.extend_from_slice(key);
        }
    }
    blob
}

fn decode_runs(mut all: &[u8]) -> Option<Vec<Vec<Key>>> {
    let mut runs = Vec::new();
    while !all.is_empty() {
        let len = u32::from_le_bytes(all.get(..4)?.try_into().ok()?) as usize;
        let keys = all.get(4..4 + 32 * len)?;
        runs.push(keys.chunks(32).map(|key| key.try_into().unwrap()).collect());
        all = &all[4 + 32 * len..];
    }
    Some(runs)
}

/// Pages kept up to `capacity` bytes in memory, and every page sent in the
/// store where there is one.
#[derive(Clone)]
pub struct Pool {
    inner: Arc<Mutex<Inner>>,
    capacity: usize,
    store: Option<Arc<dyn Store>>,
}

fn encode(meta: &[u8], slots: &[Key], bytes: &[u8]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(12 + meta.len() + 32 * slots.len() + bytes.len());
    blob.extend_from_slice(MARK);
    blob.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    blob.extend_from_slice(meta);
    blob.extend_from_slice(&(slots.len() as u32).to_le_bytes());
    for slot in slots {
        blob.extend_from_slice(slot);
    }
    blob.extend_from_slice(bytes);
    blob
}

/// The page's layout, tokens' hashes and bytes.
fn decode(all: &[u8]) -> Option<(Vec<u8>, Vec<Key>, Vec<u8>)> {
    let u32_at = |at: usize| -> Option<usize> {
        Some(u32::from_le_bytes(all.get(at..at + 4)?.try_into().ok()?) as usize)
    };
    let marked = all.get(..4) == Some(MARK.as_slice());
    let mut at = if marked { 4 } else { 0 };
    let len = u32_at(at)?;
    let meta = all.get(at + 4..at + 4 + len)?.to_vec();
    at += 4 + len;
    let mut slots = Vec::new();
    if marked {
        let count = u32_at(at)?;
        at += 4;
        for _ in 0..count {
            slots.push(all.get(at..at + 32)?.try_into().ok()?);
            at += 32;
        }
    }
    Some((meta, slots, all.get(at..)?.to_vec()))
}

impl Pool {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::default(),
            capacity,
            store: None,
        }
    }

    /// A pool that also keeps every page it is sent in `store`, and holds
    /// what the store held, its runs among it.
    pub fn with_store(capacity: usize, store: Arc<dyn Store>) -> Self {
        let runs = match store.get(&[RUNS]).remove(0) {
            Some(blob) => decode_runs(&blob).unwrap_or_else(|| {
                tracing::warn!("the pool's kept runs do not read; it keeps none");
                Vec::new()
            }),
            None => Vec::new(),
        };
        let inner = Inner {
            runs: runs
                .into_iter()
                .filter_map(|run| Some((*run.last()?, run)))
                .collect(),
            ..Default::default()
        };
        let pool = Self {
            inner: Arc::new(Mutex::new(inner)),
            capacity,
            store: Some(store),
        };
        tracing::info!(
            pages = pool.keys().len(),
            runs = pool.runs().len(),
            "pool opened its store"
        );
        pool
    }

    /// Keeps `pages`, those of `keys[from..]`, as [`Self::put`] at
    /// `retention`, and the run `keys` (from token 0) whole: one of
    /// [`Self::runs`] while every page of it is here.
    pub fn put_run(
        &self,
        keys: &[Key],
        from: usize,
        pages: Pages,
        retention: i64,
    ) -> Result<(), String> {
        let sent = keys.get(from..).ok_or("the pages start past their run")?;
        self.put_at(sent, pages, retention)?;
        let Some(last) = keys.last() else {
            return Ok(());
        };
        let held: HashSet<Key> = self.keys().into_iter().collect();
        let blob = {
            let mut inner = self.inner.lock().unwrap();
            inner.runs.insert(*last, keys.to_vec());
            inner
                .runs
                .retain(|_, run| run.iter().all(|key| held.contains(key)));
            encode_runs(inner.runs.values())
        };
        if let Some(store) = &self.store {
            store
                .put(&[(RUNS, blob)])
                .map_err(|e| format!("keeping the pool's runs in its store: {e}"))?;
        }
        Ok(())
    }

    /// Keeps `pages`, one per key, letting the lowest retention key and
    /// then the least recently used go to make room; refuses what is larger
    /// than the pool.
    pub fn put(&self, keys: &[Key], pages: Pages) -> Result<(), String> {
        self.put_at(keys, pages, 0)
    }

    /// [`Self::put`], each page at `retention` or the higher key it has;
    /// without a store, a page no lower one makes room for is refused.
    fn put_at(&self, keys: &[Key], pages: Pages, retention: i64) -> Result<(), String> {
        let page_bytes = pages.page_bytes();
        if keys.len() != pages.count || page_bytes * pages.count != pages.data.len() {
            return Err("the pages do not match their keys".into());
        }
        let per_page = pages.slots.len().checked_div(pages.count).unwrap_or(0);
        if per_page * pages.count != pages.slots.len() {
            return Err("the tokens' hashes do not split into the pages".into());
        }
        let slots_of = |at: usize| &pages.slots[at * per_page..(at + 1) * per_page];
        // With a store every page is kept there and memory caches what
        // fits; without one, more than memory holds is refused.
        if self.store.is_none() && page_bytes * keys.len() > self.capacity {
            return Err(format!(
                "{} bytes are more than this pool holds",
                page_bytes * keys.len()
            ));
        }
        if let Some(store) = &self.store {
            let blobs: Vec<(Key, Vec<u8>)> = keys
                .iter()
                .zip(pages.data.chunks(page_bytes.max(1)))
                .enumerate()
                .map(|(at, (key, bytes))| (*key, encode(&pages.meta, slots_of(at), bytes)))
                .collect();
            store
                .put(&blobs)
                .map_err(|e| format!("keeping pages in the pool's store: {e}"))?;
        }
        let meta = Arc::new(pages.meta);
        let sent: HashSet<Key> = keys.iter().copied().collect();
        let mut inner = self.inner.lock().unwrap();
        for (at, (key, bytes)) in keys
            .iter()
            .zip(pages.data.chunks(page_bytes.max(1)))
            .enumerate()
        {
            let retention = inner
                .pages
                .get(key)
                .map_or(retention, |page| page.retention.max(retention));
            inner.remove(key);
            if bytes.len() > self.capacity {
                continue;
            }
            // Room is made from pages kept at no higher key, and not from
            // the pages this put keeps.
            while inner.bytes + bytes.len() > self.capacity {
                match inner.order.first_key_value() {
                    Some((&(first, _), &going)) if first <= retention && !sent.contains(&going) => {
                        inner.remove(&going)
                    }
                    _ => break,
                }
            }
            if inner.bytes + bytes.len() > self.capacity {
                if self.store.is_none() {
                    return Err(format!(
                        "this pool is full of pages kept above key {retention}"
                    ));
                }
                continue;
            }
            inner.clock += 1;
            let used = inner.clock;
            inner.bytes += bytes.len();
            inner.order.insert((retention, used), *key);
            inner.pages.insert(
                *key,
                Page {
                    bytes: bytes.to_vec(),
                    meta: meta.clone(),
                    slots: slots_of(at).to_vec(),
                    retention,
                    used,
                },
            );
        }
        Ok(())
    }

    /// Pages `keys[from..]`, if all of them are here and laid out alike.
    pub fn get(&self, keys: &[Key], from: usize) -> Result<Pages, String> {
        let want = keys.get(from..).ok_or("past the run")?;
        // What memory lacks is read from the store in one batch, outside
        // the lock: a store may be across a network.
        let missing: Vec<Key> = {
            let inner = self.inner.lock().unwrap();
            want.iter()
                .filter(|key| !inner.pages.contains_key(*key))
                .copied()
                .collect()
        };
        let mut stored: HashMap<Key, Vec<u8>> = match (&self.store, missing.is_empty()) {
            (Some(store), false) => missing
                .iter()
                .zip(store.get(&missing))
                .filter_map(|(key, blob)| Some((*key, blob?)))
                .collect(),
            _ => HashMap::new(),
        };
        let mut inner = self.inner.lock().unwrap();
        let mut data = Vec::new();
        let mut meta: Option<Arc<Vec<u8>>> = None;
        // A run's hashes go out only if every page of it has them.
        let mut slots: Option<Vec<Key>> = Some(Vec::new());
        for (at, key) in want.iter().enumerate() {
            let (page_meta, page_slots, bytes): (Arc<Vec<u8>>, Vec<Key>, Vec<u8>) = match inner
                .pages
                .get(key)
            {
                Some(page) => (page.meta.clone(), page.slots.clone(), page.bytes.clone()),
                None => {
                    let (meta, slots, bytes) = stored
                        .remove(key)
                        .as_deref()
                        .and_then(decode)
                        .ok_or_else(|| format!("page {} of the run is not here", from + at))?;
                    (Arc::new(meta), slots, bytes)
                }
            };
            match (&mut slots, page_slots.is_empty()) {
                (Some(all), false) => all.extend(page_slots),
                _ => slots = None,
            }
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
            slots: slots.unwrap_or_default(),
        })
    }

    /// Every key held, in memory or in the store: what the pool reports to
    /// the directory.
    pub fn keys(&self) -> Vec<Key> {
        let mut keys: HashSet<Key> = self.inner.lock().unwrap().pages.keys().copied().collect();
        if let Some(store) = &self.store {
            keys.extend(store.keys());
        }
        keys.remove(&RUNS);
        keys.into_iter().collect()
    }

    /// The runs asked to be kept whole ([`Self::put_run`]) whose every page
    /// is held: the groups the pool reports to the directory.
    pub fn runs(&self) -> Vec<Vec<Key>> {
        let held: HashSet<Key> = self.keys().into_iter().collect();
        self.inner
            .lock()
            .unwrap()
            .runs
            .values()
            .filter(|run| run.iter().all(|key| held.contains(key)))
            .cloned()
            .collect()
    }

    pub fn bytes(&self) -> usize {
        self.inner.lock().unwrap().bytes
    }
}

/// Reads and writes may touch the store: off the async threads.
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

    /// A worker's own push, at key 0: a `replicate` for the run raises it.
    fn write(&self, keys: Vec<Key>, from: u32, pages: Pages) -> Taken {
        let pool = self.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || pool.put_run(&keys, from as usize, pages, 0))
                .await
                .map_err(|e| e.to_string())?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Dir;

    /// `count` pages of 8 bytes of `n`, two tokens a page, page `p`'s hashed
    /// `n + p`.
    fn pages(n: u8, count: usize) -> Pages {
        Pages {
            data: vec![n; 8 * count],
            count,
            meta: b"layout".to_vec(),
            slots: (0..count).flat_map(|p| [[n + p as u8; 32]; 2]).collect(),
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
        assert_eq!(got.slots, [[10u8; 32], [10u8; 32], [11u8; 32], [11u8; 32]]);
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
            let pool = Pool::with_store(8, Arc::new(Dir::open(dir.clone()).unwrap()));
            pool.put(&keys, pages(3, 2)).unwrap();
            // Memory holds one page; the other is in the store alone.
            assert_eq!(pool.bytes(), 8);
            assert_eq!(pool.get(&keys, 0).unwrap().data, vec![3u8; 16]);
        }
        let reopened = Pool::with_store(8, Arc::new(Dir::open(dir.clone()).unwrap()));
        let mut held = reopened.keys();
        held.sort();
        assert_eq!(held, keys.to_vec());
        let got = reopened.get(&keys, 0).unwrap();
        assert_eq!((got.count, got.meta.as_slice()), (2, &b"layout"[..]));
        assert_eq!(got.slots, [[3u8; 32], [3u8; 32], [4u8; 32], [4u8; 32]]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_full_pool_lets_the_lowest_key_go_first_and_never_for_a_lower_one() {
        let pool = Pool::new(32);
        let (high, low) = ([[1u8; 32], [2u8; 32]], [[3u8; 32], [4u8; 32]]);
        pool.put_run(&high, 0, pages(1, 2), 5).unwrap();
        pool.put_run(&low, 0, pages(3, 2), 1).unwrap();
        // Reading the high run makes it the most recent; still the low run
        // is what goes for a third run at key 3.
        pool.get(&high, 0).unwrap();
        pool.get(&low, 0).unwrap();
        let mid = [[5u8; 32], [6u8; 32]];
        pool.put_run(&mid, 0, pages(5, 2), 3).unwrap();
        let mut runs: Vec<Key> = pool.runs().iter().map(|run| run[1]).collect();
        runs.sort();
        assert_eq!(runs, [high[1], mid[1]], "the run at key 1 went");
        assert!(
            pool.put_run(&low, 0, pages(3, 2), 2).is_err(),
            "a run at key 2 takes no room from runs at 3 and 5"
        );
        // A page sent in no run is at key 0, and takes no room from them.
        assert!(pool.put(&[[7u8; 32]], pages(7, 1)).is_err());
        let mut runs: Vec<Key> = pool.runs().iter().map(|run| run[1]).collect();
        runs.sort();
        assert_eq!(runs, [high[1], mid[1]]);
    }

    #[test]
    fn a_pool_reports_the_runs_it_holds_whole_and_its_store_keeps_them() {
        let dir = std::env::temp_dir().join(format!("pie-pool-runs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let run = [[1u8; 32], [2u8; 32]];
        let prefix = [[1u8; 32]];
        {
            let pool = Pool::with_store(8, Arc::new(Dir::open(dir.clone()).unwrap()));
            pool.put_run(&run, 0, pages(3, 2), 1).unwrap();
            pool.put_run(&prefix, 0, pages(3, 1), 1).unwrap();
            // Pages sent alone make no run.
            pool.put(&[[9u8; 32]], pages(9, 1)).unwrap();
            let mut runs = pool.runs();
            runs.sort();
            assert_eq!(runs, [prefix.to_vec(), run.to_vec()]);
        }
        let reopened = Pool::with_store(8, Arc::new(Dir::open(dir.clone()).unwrap()));
        let mut runs = reopened.runs();
        runs.sort();
        assert_eq!(runs, [prefix.to_vec(), run.to_vec()]);
        let mut held = reopened.keys();
        held.sort();
        assert_eq!(held, [run[0], run[1], [9u8; 32]], "the runs are no page");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_run_a_worker_pushes_is_kept_as_a_run_at_key_0_and_a_replicate_raises_it() {
        let pool = Pool::new(16);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        tokio::spawn(transport::tcp::serve(listener, Arc::new(pool.clone())));
        let (prefix, run) = ([[1u8; 32]], [[1u8; 32], [2u8; 32]]);
        transport::tcp::put(&addr, &prefix, 0, &pages(1, 1))
            .await
            .unwrap();
        // The run is pushed from the page the pool was not sent.
        transport::tcp::put(&addr, &run, 1, &pages(2, 1))
            .await
            .unwrap();
        let mut runs = pool.runs();
        runs.sort();
        assert_eq!(runs, [prefix.to_vec(), run.to_vec()]);
        assert_eq!(
            pool.get(&run, 0).unwrap().data,
            [[1u8; 8], [2u8; 8]].concat()
        );
        // At key 0, a run at key 1 takes its room.
        let other = [[3u8; 32], [4u8; 32]];
        pool.put_run(&other, 0, pages(3, 2), 1).unwrap();
        assert_eq!(pool.runs(), [other.to_vec()]);
        // A replicate at key 3 raises the run, and a push again at key 0
        // does not lower it: a run at key 2 finds no room.
        pool.put_run(&run, 0, pages(1, 2), 3).unwrap();
        transport::tcp::put(&addr, &run, 0, &pages(1, 2))
            .await
            .unwrap();
        assert!(pool.put_run(&other, 0, pages(3, 2), 2).is_err());
        assert_eq!(pool.runs(), [run.to_vec()]);
    }

    #[test]
    fn a_run_with_a_page_sent_without_its_hashes_goes_out_without_any() {
        let pool = Pool::new(64);
        let keys = [[1u8; 32], [2u8; 32]];
        pool.put(&keys[..1], pages(5, 1)).unwrap();
        let bare = Pages {
            slots: Vec::new(),
            ..pages(6, 1)
        };
        pool.put(&keys[1..], bare).unwrap();
        let got = pool.get(&keys, 0).unwrap();
        assert_eq!(got.count, 2);
        assert!(got.slots.is_empty(), "half a run's hashes name nothing");
    }
}
