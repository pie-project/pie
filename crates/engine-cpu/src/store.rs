//! Where a pool keeps every page it is sent, beyond what its memory holds: a
//! directory of its own ([`Dir`]), or a store a NIXL storage plugin fronts
//! (`Objects`, with the `nixl` feature: S3 and what speaks it). A page is an
//! opaque blob by key; a store that opens again holds what it held.

use std::collections::HashSet;
use std::sync::Mutex;

use transport::tcp::Key;

pub trait Store: Send + Sync {
    /// Keeps each blob under its key; on error, any of them may be kept.
    fn put(&self, pages: &[(Key, Vec<u8>)]) -> Result<(), String>;
    /// The blobs under `keys`, in order; `None` for one not held.
    fn get(&self, keys: &[Key]) -> Vec<Option<Vec<u8>>>;
    /// Every key held.
    fn keys(&self) -> Vec<Key>;
}

pub(crate) fn hex(key: &Key) -> String {
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

/// A directory of files, one per page, named by its key: each written
/// whole before it is named, so a crash leaves no torn page.
pub struct Dir {
    dir: std::path::PathBuf,
    keys: Mutex<HashSet<Key>>,
}

impl Dir {
    pub fn open(dir: std::path::PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        let mut keys = HashSet::new();
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

    fn write(&self, key: &Key, blob: &[u8]) -> std::io::Result<()> {
        use std::io::Write;
        let name = hex(key);
        let tmp = self.dir.join(format!(".{name}.tmp"));
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(blob)?;
        file.sync_data()?;
        std::fs::rename(&tmp, self.dir.join(name))?;
        self.keys.lock().unwrap().insert(*key);
        Ok(())
    }
}

impl Store for Dir {
    fn put(&self, pages: &[(Key, Vec<u8>)]) -> Result<(), String> {
        for (key, blob) in pages {
            self.write(key, blob)
                .map_err(|e| format!("writing a page to {}: {e}", self.dir.display()))?;
        }
        Ok(())
    }

    fn get(&self, keys: &[Key]) -> Vec<Option<Vec<u8>>> {
        keys.iter()
            .map(|key| {
                let path = self.dir.join(hex(key));
                match std::fs::read(&path) {
                    Ok(blob) => Some(blob),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => {
                        tracing::warn!(path = %path.display(), %error, "a kept page not read");
                        None
                    }
                }
            })
            .collect()
    }

    fn keys(&self) -> Vec<Key> {
        self.keys.lock().unwrap().iter().copied().collect()
    }
}

#[cfg(feature = "nixl")]
pub use objects::Objects;

#[cfg(feature = "nixl")]
mod objects {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use transport::tcp::Key;

    use super::{Store, hex};

    /// A log entry: a page's key and its blob's length.
    const ENTRY: usize = 32 + 8;

    /// Pages as objects under a prefix of a store a NIXL storage plugin
    /// fronts. Such a store lists nothing and gives no object's length, so
    /// the pool keeps a log beside the pages: each put appends a segment
    /// naming its pages and their lengths, written after them. The log runs
    /// in epochs; `head` names the current one. Opening replays it, writes
    /// all it learned as the next epoch's first segment, and only then
    /// moves `head` there: a crash in between replays the same epoch again.
    ///
    /// Objects are names: `{prefix}/page/{key}`, `{prefix}/log/{epoch}/{n}`,
    /// `{prefix}/head`. Nothing is deleted; the store's own lifecycle rules
    /// age out what they will, and a page gone from it is a miss.
    pub struct Objects {
        objects: transport::Objects,
        prefix: String,
        log: Mutex<Log>,
    }

    struct Log {
        epoch: u64,
        /// The next segment's number in the epoch.
        next: u64,
        lens: HashMap<Key, usize>,
    }

    impl Objects {
        pub fn open(objects: transport::Objects, prefix: String) -> Result<Self, String> {
            let fail = |e: transport::TransportError| e.to_string();
            let head = format!("{prefix}/head");
            let epoch = match objects.exists(&[&head]).map_err(fail)?[0] {
                true => {
                    let bytes = objects.get(&[(&head, 8)]).map_err(fail)?.remove(0);
                    u64::from_le_bytes(bytes.try_into().expect("eight bytes asked for"))
                }
                false => 0,
            };
            let segment = |epoch: u64, n: u64| format!("{prefix}/log/{epoch}/{n}");
            let mut lens = HashMap::new();
            for n in 0.. {
                let name = segment(epoch, n);
                if !objects.exists(&[&name]).map_err(fail)?[0] {
                    break;
                }
                let count = objects.get(&[(&name, 4)]).map_err(fail)?.remove(0);
                let count = u32::from_le_bytes(count.try_into().expect("four bytes")) as usize;
                let all = objects
                    .get(&[(&name, 4 + count * ENTRY)])
                    .map_err(fail)?
                    .remove(0);
                for entry in all[4..].as_chunks::<ENTRY>().0 {
                    let key: Key = entry[..32].try_into().expect("a key's bytes");
                    let len = u64::from_le_bytes(entry[32..].try_into().expect("eight bytes"));
                    lens.insert(key, len as usize);
                }
            }
            let next = epoch + 1;
            let all: Vec<(Key, usize)> = lens.iter().map(|(k, l)| (*k, *l)).collect();
            objects
                .put(&[(&segment(next, 0), &encode(&all))])
                .map_err(fail)?;
            objects.put(&[(&head, &next.to_le_bytes())]).map_err(fail)?;
            tracing::info!(%prefix, pages = lens.len(), epoch = next, "pool opened its object store");
            Ok(Self {
                objects,
                prefix,
                log: Mutex::new(Log {
                    epoch: next,
                    next: 1,
                    lens,
                }),
            })
        }

        fn page(&self, key: &Key) -> String {
            format!("{}/page/{}", self.prefix, hex(key))
        }
    }

    fn encode(entries: &[(Key, usize)]) -> Vec<u8> {
        let mut segment = (entries.len() as u32).to_le_bytes().to_vec();
        for (key, len) in entries {
            segment.extend_from_slice(key);
            segment.extend_from_slice(&(*len as u64).to_le_bytes());
        }
        segment
    }

    impl Store for Objects {
        fn put(&self, pages: &[(Key, Vec<u8>)]) -> Result<(), String> {
            // Held through both writes, so segments are numbered without gaps.
            let mut log = self.log.lock().unwrap();
            let names: Vec<String> = pages.iter().map(|(key, _)| self.page(key)).collect();
            let objects: Vec<(&str, &[u8])> = names
                .iter()
                .zip(pages)
                .map(|(name, (_, blob))| (name.as_str(), blob.as_slice()))
                .collect();
            self.objects.put(&objects).map_err(|e| e.to_string())?;
            let entries: Vec<(Key, usize)> =
                pages.iter().map(|(key, blob)| (*key, blob.len())).collect();
            let segment = format!("{}/log/{}/{}", self.prefix, log.epoch, log.next);
            self.objects
                .put(&[(&segment, &encode(&entries))])
                .map_err(|e| e.to_string())?;
            log.next += 1;
            log.lens.extend(entries);
            Ok(())
        }

        fn get(&self, keys: &[Key]) -> Vec<Option<Vec<u8>>> {
            let lens: Vec<Option<usize>> = {
                let log = self.log.lock().unwrap();
                keys.iter().map(|key| log.lens.get(key).copied()).collect()
            };
            let names: Vec<String> = keys.iter().map(|key| self.page(key)).collect();
            let held: Vec<(&str, usize)> = names
                .iter()
                .zip(&lens)
                .filter_map(|(name, len)| Some((name.as_str(), (*len)?)))
                .collect();
            // One batch; if any of it is gone from the store, all of it is
            // a miss, and what is gone is no longer told as kept.
            let mut got = match self.objects.get(&held) {
                Ok(blobs) => blobs.into_iter(),
                Err(error) => {
                    tracing::warn!(%error, "pages not read from the object store");
                    self.forget_missing(keys, &names);
                    return vec![None; keys.len()];
                }
            };
            lens.iter()
                .map(|len| len.and_then(|_| got.next()))
                .collect()
        }

        fn keys(&self) -> Vec<Key> {
            self.log.lock().unwrap().lens.keys().copied().collect()
        }
    }

    impl Objects {
        /// Drops from the log's lengths the keys among `keys` whose objects
        /// (`names`, one each) the store no longer has.
        fn forget_missing(&self, keys: &[Key], names: &[String]) {
            let asked: Vec<&str> = names.iter().map(String::as_str).collect();
            match self.objects.exists(&asked) {
                Ok(present) => {
                    let mut log = self.log.lock().unwrap();
                    for (key, present) in keys.iter().zip(present) {
                        if !present {
                            log.lens.remove(key);
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "the object store would not say what it holds");
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use std::sync::Arc;

        use transport::tcp::Pages;

        use super::*;
        use crate::Pool;

        /// The S3 endpoint `PIE_TEST_S3` names (a URL; any credentials, the
        /// bucket `pie`), as an agent's store.
        fn objects(agent: &str) -> Option<transport::Objects> {
            let endpoint = std::env::var("PIE_TEST_S3").ok()?;
            let params = [
                ("bucket", "pie"),
                ("endpoint_override", endpoint.as_str()),
                ("scheme", "http"),
                ("region", "us-east-1"),
                ("access_key", "test"),
                ("secret_key", "test"),
            ]
            .map(|(k, v)| (k.to_string(), v.to_string()));
            Some(transport::Objects::open(agent, "OBJ", &params).unwrap())
        }

        fn pages(n: u8, count: usize) -> Pages {
            Pages {
                data: (0..8 * count).map(|i| n + i as u8).collect(),
                count,
                meta: b"layout".to_vec(),
                slots: (0..count).map(|p| [n + p as u8; 32]).collect(),
            }
        }

        #[test]
        fn a_pool_on_objects_comes_back_with_what_it_held_across_opens() {
            let Some(store) = objects("pool-a") else {
                eprintln!("no PIE_TEST_S3: skipping");
                return;
            };
            let prefix = format!("pool-test/{}", std::process::id());
            let (first, second) = ([[1u8; 32], [2u8; 32]], [[3u8; 32]]);
            {
                let pool =
                    Pool::with_store(0, Arc::new(Objects::open(store, prefix.clone()).unwrap()));
                pool.put(&first, pages(10, 2)).unwrap();
                pool.put(&second, pages(20, 1)).unwrap();
            }
            // Twice: the second open replays what the first wrote as its
            // epoch's first segment.
            for agent in ["pool-b", "pool-c"] {
                let store = Objects::open(objects(agent).unwrap(), prefix.clone()).unwrap();
                let pool = Pool::with_store(0, Arc::new(store));
                let mut held = pool.keys();
                held.sort();
                assert_eq!(held, [first[0], first[1], second[0]]);
                let got = pool.get(&first, 0).unwrap();
                assert_eq!(got.data, pages(10, 2).data);
                assert_eq!(got.slots, pages(10, 2).slots);
                assert_eq!(pool.get(&second, 0).unwrap().data, pages(20, 1).data);
            }
        }
    }
}
