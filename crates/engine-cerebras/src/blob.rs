use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
use std::time::UNIX_EPOCH;

use crate::adapter::{Site, layer_of, role_of, site_of};
use crate::error::{Fault, Result};
use crate::weights::BankSeat;

pub const MANIFEST: &str = "adapter.toml";

#[derive(Debug, Clone, Default)]
pub struct Vfs {
    root: Option<PathBuf>,
}

impl Vfs {
    #[must_use]
    pub fn new(root: Option<PathBuf>) -> Vfs {
        Vfs { root }
    }

    #[must_use]
    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    pub fn resolve(&self, name: &str) -> Result<PathBuf> {
        let root = self.root.as_ref().ok_or_else(|| Fault::Blob {
            path: name.to_string(),
            why: "cannot be opened: this shell has no shared adapter directory mounted, \
                  so there is no namespace for the name to be in"
                .to_string(),
        })?;
        let trimmed = name.trim().trim_start_matches('/');
        if trimmed.is_empty() {
            return Err(Fault::Blob {
                path: name.to_string(),
                why: "is not a name; an adapter is a directory in the mount".to_string(),
            });
        }
        let relative = Path::new(trimmed);
        for part in relative.components() {
            if !matches!(part, Component::Normal(_)) {
                return Err(Fault::Blob {
                    path: name.to_string(),
                    why: "leaves the mount; a shared adapter is named by plain path \
                          components under the mount root and never by `..` or a second root"
                        .to_string(),
                });
            }
        }
        let at = root.join(relative);
        if !at.is_dir() {
            return Err(Fault::Blob {
                path: name.to_string(),
                why: format!(
                    "is not a directory in the mount at {}; a shared adapter is a \
                     directory holding `{MANIFEST}` and the planes it names",
                    root.display()
                ),
            });
        }
        Ok(at)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RankMajor,

    OutMajor,
}

impl Layout {
    #[must_use]
    pub fn spelled(self) -> &'static str {
        match self {
            Layout::RankMajor => "rank-major [rank, hidden]",
            Layout::OutMajor => "out-major [hidden, rank]",
        }
    }

    #[must_use]
    pub fn written(self) -> &'static str {
        match self {
            Layout::RankMajor => "rank_major",
            Layout::OutMajor => "out_major",
        }
    }

    #[must_use]
    pub fn of_bank(seat: &BankSeat) -> Layout {
        match seat.rows <= seat.cols {
            true => Layout::RankMajor,
            false => Layout::OutMajor,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneSpec {
    pub role: String,

    pub file: String,

    pub layout: Layout,

    pub site: Option<Site>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub rank: u64,

    pub planes: Vec<PlaneSpec>,
}

impl Manifest {
    pub fn read(dir: &Path, name: &str) -> Result<Manifest> {
        let at = dir.join(MANIFEST);
        let refuse = |why: String| Fault::Blob {
            path: name.to_string(),
            why,
        };
        let text = std::fs::read_to_string(&at)
            .map_err(|error| refuse(format!("has no readable `{MANIFEST}`: {error}")))?;
        let doc: toml::Table = text
            .parse()
            .map_err(|error| refuse(format!("has a `{MANIFEST}` that is not TOML: {error}")))?;
        let rank = doc
            .get("rank")
            .and_then(toml::Value::as_integer)
            .and_then(|rank| u64::try_from(rank).ok())
            .filter(|rank| *rank > 0)
            .ok_or_else(|| {
                refuse(format!(
                    "declares no positive `rank` in its `{MANIFEST}`; the rank is what \
                     says how much of a bank's slot the source fills"
                ))
            })?;
        let planes = doc
            .get("plane")
            .and_then(toml::Value::as_array)
            .ok_or_else(|| {
                refuse(format!(
                    "names no `[[plane]]` in its `{MANIFEST}`; an adapter with no planes \
                     lands nothing"
                ))
            })?;
        let planes = planes
            .iter()
            .map(|plane| {
                let plane = plane
                    .as_table()
                    .ok_or_else(|| refuse("has a `[[plane]]` that is not a table".to_string()))?;
                let field = |key: &str| {
                    plane
                        .get(key)
                        .and_then(toml::Value::as_str)
                        .map(str::to_string)
                        .ok_or_else(|| refuse(format!("has a `[[plane]]` with no `{key}`")))
                };
                let role = field("role")?;
                let file = field("file")?;
                let layout = match field("layout")?.as_str() {
                    "rank_major" => Layout::RankMajor,
                    "out_major" => Layout::OutMajor,
                    other => {
                        return Err(refuse(format!(
                            "declares plane `{role}` as layout `{other}`; the two layouts \
                             a bank can be filled from are `rank_major` ([rank, hidden]) \
                             and `out_major` ([hidden, rank])"
                        )));
                    }
                };
                let site = match plane.get("site") {
                    None => None,
                    Some(value) => {
                        let word = value.as_str().ok_or_else(|| {
                            refuse(format!(
                                "declares plane `{role}`'s `site` as something that is \
                                 not a string; a site is one word of {}",
                                Site::vocabulary()
                            ))
                        })?;
                        Some(Site::parse(word).ok_or_else(|| {
                            refuse(format!(
                                "declares plane `{role}` at site `{word}`, and the \
                                 correction sites a bank can be named at are {}; a site \
                                 nobody can name is refused rather than landed at \
                                 whatever site the model text happens to correct",
                                Site::vocabulary()
                            ))
                        })?)
                    }
                };
                Ok(PlaneSpec {
                    role,
                    file,
                    layout,
                    site,
                })
            })
            .collect::<Result<Vec<PlaneSpec>>>()?;
        Ok(Manifest { rank, planes })
    }
}

#[derive(Debug)]
pub struct Blob {
    pub at: PathBuf,

    pub bytes: Vec<u8>,

    pub fingerprint: u64,
}

enum Cell {
    Loading,

    Held(Weak<Blob>),
}

impl std::fmt::Debug for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cell::Loading => f.write_str("Loading"),
            Cell::Held(_) => f.write_str("Held"),
        }
    }
}

#[derive(Debug, Default)]
pub struct Blobs {
    held: Mutex<HashMap<PathBuf, Cell>>,
    ready: Condvar,
    loads: AtomicU64,
}

impl Blobs {
    #[must_use]
    pub fn loads(&self) -> u64 {
        self.loads.load(Ordering::Relaxed)
    }

    pub fn open(&self, at: &Path, path: &str) -> Result<Arc<Blob>> {
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            let seen = match held.get(at) {
                Some(Cell::Held(weak)) => Some(weak.upgrade()),
                Some(Cell::Loading) => None,
                None => Some(None),
            };
            match seen {
                Some(Some(blob)) => return Ok(blob),
                Some(None) => {
                    held.insert(at.to_path_buf(), Cell::Loading);
                    break;
                }
                None => {
                    held = self
                        .ready
                        .wait(held)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
        drop(held);
        self.loads.fetch_add(1, Ordering::Relaxed);
        let read = std::fs::read(at);
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        let out = match read {
            Ok(bytes) => {
                let blob = Arc::new(Blob {
                    at: at.to_path_buf(),
                    fingerprint: fingerprint(&bytes),
                    bytes,
                });
                held.insert(at.to_path_buf(), Cell::Held(Arc::downgrade(&blob)));
                Ok(blob)
            }
            Err(error) => {
                held.remove(at);
                Err(Fault::Blob {
                    path: path.to_string(),
                    why: format!("names `{}`, which will not read: {error}", at.display()),
                })
            }
        };
        drop(held);
        self.ready.notify_all();
        out
    }
}

#[must_use]
pub fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stamp {
    pub at: String,

    pub files: Vec<(String, u64, u128)>,
}

fn stat(at: &Path, file: &str, name: &str) -> Result<(String, u64, u128)> {
    let meta = std::fs::metadata(at).map_err(|error| Fault::Blob {
        path: name.to_string(),
        why: format!("names `{file}`, which is not in the directory: {error}"),
    })?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|at| at.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_nanos());
    Ok((file.to_string(), meta.len(), mtime))
}

#[derive(Debug, Default)]
pub struct Store {
    vfs: Vfs,
    blobs: Blobs,
}

pub type Planes = (Vec<(String, Vec<u8>)>, u64);

impl Store {
    #[must_use]
    pub fn new() -> Store {
        Store::default()
    }

    pub fn mount(&mut self, root: Option<PathBuf>) {
        self.vfs = Vfs::new(root);
    }

    #[must_use]
    pub fn vfs(&self) -> &Vfs {
        &self.vfs
    }

    #[must_use]
    pub fn blobs(&self) -> &Blobs {
        &self.blobs
    }

    pub fn stamp(&self, name: &str) -> Result<Stamp> {
        let dir = self.vfs.resolve(name)?;
        let manifest = Manifest::read(&dir, name)?;
        let mut files = Vec::with_capacity(manifest.planes.len() + 1);
        files.push(stat(&dir.join(MANIFEST), MANIFEST, name)?);
        for plane in &manifest.planes {
            files.push(stat(&dir.join(&plane.file), &plane.file, name)?);
        }
        Ok(Stamp {
            at: dir.display().to_string(),
            files,
        })
    }

    pub fn planes(&self, name: &str, seats: &[BankSeat]) -> Result<Planes> {
        let dir = self.vfs.resolve(name)?;
        let manifest = Manifest::read(&dir, name)?;
        let refuse = |why: String| Fault::Blob {
            path: name.to_string(),
            why,
        };
        let mut out = Vec::new();
        let mut fingerprint = 0u64;
        for spec in &manifest.planes {
            let mut banks: Vec<&BankSeat> = seats
                .iter()
                .filter(|seat| role_of(&seat.name) == spec.role && site_of(&seat.name) == spec.site)
                .collect();
            banks.sort_by_key(|seat| layer_of(&seat.name));
            let seat = *banks.first().ok_or_else(|| {
                refuse(format!(
                    "declares a plane for role `{}` {} and this load declares no bank by \
                     that name; its banks are {:?}",
                    spec.role,
                    Site::stated(spec.site),
                    seats.iter().map(|seat| &seat.name).collect::<Vec<_>>()
                ))
            })?;
            if let Some(odd) = banks
                .iter()
                .find(|bank| bank.slot != seat.slot || bank.rows != seat.rows)
            {
                return Err(refuse(format!(
                    "fills role `{}` across {} banks and they are not one shape: `{}` \
                     seats {} bytes and `{}` seats {}; a `[layers, ...]` source is L \
                     contiguous slices of ONE rectangle",
                    spec.role,
                    banks.len(),
                    seat.name,
                    seat.slot,
                    odd.name,
                    odd.slot
                )));
            }

            let bank_layout = Layout::of_bank(seat);
            if bank_layout != spec.layout {
                return Err(refuse(format!(
                    "declares plane `{}` {} and bank `{}` seats [{}, {}], which is {}; \
                     landing one as the other is a transpose, and a repack kernel is \
                     exactly what the out-major statute exists to avoid — so it is \
                     refused rather than repacked",
                    spec.role,
                    spec.layout.spelled(),
                    seat.name,
                    seat.rows,
                    seat.cols,
                    bank_layout.spelled()
                )));
            }
            let bank_rank = seat.rows.min(seat.cols);
            let hidden = seat.rows.max(seat.cols);
            if manifest.rank > bank_rank {
                return Err(refuse(format!(
                    "is rank {} and bank `{}` seats rank {}; the bank's capacity is a \
                     shape the model text declared, so the fix is a bank that seats it \
                     and not a retry",
                    manifest.rank, seat.name, bank_rank
                )));
            }
            let blob = self.blobs.open(&dir.join(&spec.file), name)?;

            fingerprint = (fingerprint ^ blob.fingerprint).wrapping_mul(0x0000_0100_0000_01b3);
            let stride = manifest
                .rank
                .saturating_mul(hidden)
                .saturating_mul(seat.elem);
            let want = stride.saturating_mul(banks.len() as u64);
            if blob.bytes.len() as u64 != want {
                return Err(refuse(format!(
                    "carries {} bytes in `{}` and the {} banks of role `{}` want {} — \
                     {} layers x rank {} x {} at {} bytes an element",
                    blob.bytes.len(),
                    spec.file,
                    banks.len(),
                    spec.role,
                    want,
                    banks.len(),
                    manifest.rank,
                    hidden,
                    seat.elem
                )));
            }
            let stride = usize::try_from(stride).unwrap_or(usize::MAX);
            let slot = usize::try_from(seat.slot).unwrap_or(usize::MAX);
            for (layer, bank) in banks.iter().enumerate() {
                let source = &blob.bytes[layer * stride..(layer + 1) * stride];

                let mut plane = vec![0u8; slot];
                match spec.layout {
                    Layout::RankMajor => plane[..source.len()].copy_from_slice(source),
                    Layout::OutMajor => {
                        let from = usize::try_from(manifest.rank.saturating_mul(seat.elem))
                            .unwrap_or(usize::MAX);
                        let to = usize::try_from(bank_rank.saturating_mul(seat.elem))
                            .unwrap_or(usize::MAX);
                        for row in 0..usize::try_from(hidden).unwrap_or(0) {
                            plane[row * to..row * to + from]
                                .copy_from_slice(&source[row * from..(row + 1) * from]);
                        }
                    }
                }
                out.push((bank.name.clone(), plane));
            }
        }
        Ok((out, fingerprint))
    }
}
