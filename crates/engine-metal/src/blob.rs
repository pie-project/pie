use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
use std::time::UNIX_EPOCH;

use crate::adapter::{Site, layer_of, role_of, site_of};
use crate::error::{Fault, Result};
use crate::weights::BankSeat;
type Planes = (Vec<(String, Vec<u8>)>, u64);

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

#[cfg(test)]
mod tests {
    use super::*;

    const LAYERS: u64 = 3;
    const BANK_RANK: u64 = 8;
    const HIDDEN: u64 = 16;
    const ELEM: u64 = 2;

    fn scratch(what: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0);
        let at = std::env::temp_dir().join(format!(
            "pie-metal-blob-{what}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&at).expect("a scratch directory");
        at
    }

    fn seat(name: &str, rows: u64, cols: u64) -> BankSeat {
        BankSeat {
            name: name.to_string(),
            adapters: 4,
            slot: rows * cols * ELEM,
            rows,
            cols,
            elem: ELEM,
        }
    }

    fn seats() -> Vec<BankSeat> {
        (0..LAYERS)
            .flat_map(|layer| {
                [
                    seat(&format!("layer.{layer}.lora_a"), BANK_RANK, HIDDEN),
                    seat(&format!("layer.{layer}.lora_b"), HIDDEN, BANK_RANK),
                ]
            })
            .collect()
    }

    fn source(element: usize) -> [u8; 2] {
        ((element as u16) | 0x0100).to_le_bytes()
    }

    fn write_adapter(mount: &Path, name: &str, rank: u64, layouts: (Layout, Layout)) -> PathBuf {
        let dir = mount.join(name);
        std::fs::create_dir_all(&dir).expect("an adapter directory");
        std::fs::write(
            dir.join(MANIFEST),
            format!(
                "rank = {rank}\n\n\
                 [[plane]]\nrole = \"lora_a\"\nfile = \"a.bin\"\nlayout = \"{}\"\n\n\
                 [[plane]]\nrole = \"lora_b\"\nfile = \"b.bin\"\nlayout = \"{}\"\n",
                layouts.0.written(),
                layouts.1.written()
            ),
        )
        .expect("a manifest");
        let ramp: Vec<u8> = (0..(LAYERS * rank * HIDDEN) as usize)
            .flat_map(source)
            .collect();
        std::fs::write(dir.join("a.bin"), &ramp).expect("an A plane");
        std::fs::write(dir.join("b.bin"), &ramp).expect("a B plane");
        dir
    }

    fn mounted(what: &str) -> (PathBuf, Store) {
        let mount = scratch(what);
        write_adapter(&mount, "alice-v2", 4, (Layout::RankMajor, Layout::OutMajor));
        let mut store = Store::new();
        store.mount(Some(mount.clone()));
        (mount, store)
    }

    #[test]
    fn blob_every_case() {
        a_manifest_says_its_rank_its_planes_and_their_orientation();
        the_mount_resolves_a_name_and_refuses_everything_else();
        one_blob_is_one_stamp_and_a_rewrite_is_another();
        eight_threads_asking_for_one_blob_read_it_once();
        a_read_that_refuses_leaves_no_claim_behind();
        the_resolver_slices_per_layer_and_pads_per_orientation();
        the_resolver_refuses_by_name();
    }

    fn a_manifest_says_its_rank_its_planes_and_their_orientation() {
        let mount = scratch("manifest");
        let dir = write_adapter(&mount, "alice", 4, (Layout::RankMajor, Layout::OutMajor));
        let manifest = Manifest::read(&dir, "alice").expect("a well-formed manifest");
        assert_eq!(manifest.rank, 4);
        assert_eq!(
            manifest.planes,
            vec![
                PlaneSpec {
                    role: "lora_a".to_string(),
                    file: "a.bin".to_string(),
                    layout: Layout::RankMajor,
                    site: None,
                },
                PlaneSpec {
                    role: "lora_b".to_string(),
                    file: "b.bin".to_string(),
                    layout: Layout::OutMajor,
                    site: None,
                },
            ],
            "the planes come back in the order the file names them"
        );

        let sited = mount.join("sited");
        std::fs::create_dir_all(&sited).expect("a directory");
        std::fs::write(
            sited.join(MANIFEST),
            "rank = 2\n\n[[plane]]\nrole = \"lora_a\"\nfile = \"a.bin\"\n\
             layout = \"rank_major\"\nsite = \"o\"\n",
        )
        .expect("a manifest");
        let manifest = Manifest::read(&sited, "sited").expect("a sited manifest");
        assert_eq!(manifest.planes[0].site, Some(Site::O));
        let _ = std::fs::remove_dir_all(&mount);
    }

    fn the_mount_resolves_a_name_and_refuses_everything_else() {
        let (mount, store) = mounted("vfs");
        assert_eq!(
            store
                .vfs()
                .resolve("alice-v2")
                .expect("a name in the mount"),
            mount.join("alice-v2")
        );
        assert_eq!(
            store
                .vfs()
                .resolve("/alice-v2")
                .expect("the guest's own preopen spelling"),
            mount.join("alice-v2"),
            "a leading slash is the mount's root, not another one"
        );

        let said = store
            .vfs()
            .resolve("../elsewhere")
            .expect_err("a traversal")
            .to_string();
        assert!(said.contains("leaves the mount"), "{said}");
        let said = store.vfs().resolve("   ").expect_err("no name").to_string();
        assert!(said.contains("is not a name"), "{said}");
        let said = store
            .vfs()
            .resolve("nobody")
            .expect_err("a name nobody wrote")
            .to_string();
        assert!(said.contains("nobody"), "names the adapter: {said}");

        let bare = Store::new();
        let said = bare
            .stamp("alice-v2")
            .expect_err("nothing mounted")
            .to_string();
        assert!(
            said.contains("no shared adapter directory mounted"),
            "{said}"
        );
        let _ = std::fs::remove_dir_all(&mount);
    }

    fn one_blob_is_one_stamp_and_a_rewrite_is_another() {
        let (mount, store) = mounted("identity");
        let first = store.stamp("alice-v2").expect("the adapter is there");
        let again = store.stamp("alice-v2").expect("and it has not moved");
        assert_eq!(first, again, "the same path twice is the same key");
        assert_eq!(
            store.stamp("/alice-v2").expect("the same adapter"),
            first,
            "a leading slash is the same file, so it is the same key — the sharing \
             claim is about bytes and not about strings"
        );
        assert_eq!(
            first.files.len(),
            3,
            "the manifest and both planes are stamped"
        );
        assert_eq!(first.files[0].0, MANIFEST, "the manifest is stamped first");

        std::fs::write(mount.join("alice-v2").join("a.bin"), vec![0u8; 8])
            .expect("the plane is rewritten");
        let after = store.stamp("alice-v2").expect("still there");
        assert_ne!(
            after, first,
            "a rewritten file is a new identity, therefore a new slot — which is how \
             no in-flight fire ever observes an adapter changing"
        );
        let _ = std::fs::remove_dir_all(&mount);
    }

    fn eight_threads_asking_for_one_blob_read_it_once() {
        let at = scratch("flight").join("plane.bin");
        std::fs::write(&at, vec![7u8; 1 << 16]).expect("a plane");
        let blobs = Blobs::default();

        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let blobs = &blobs;
                    let at = &at;
                    scope.spawn(move || blobs.open(at, "plane").expect("the read"))
                })
                .collect();
            let held: Vec<_> = handles
                .into_iter()
                .map(|handle| handle.join().expect("a thread"))
                .collect();
            assert_eq!(held.len(), 8);
            for blob in &held {
                assert_eq!(blob.bytes.len(), 1 << 16);
                assert_eq!(blob.fingerprint, held[0].fingerprint);
            }
            assert_eq!(blobs.loads(), 1, "one read, seven waiters");
        });

        let again = blobs.open(&at, "plane").expect("a second generation");
        assert_eq!(blobs.loads(), 2);
        assert_eq!(again.bytes.len(), 1 << 16);
        let _ = std::fs::remove_file(&at);
    }

    fn a_read_that_refuses_leaves_no_claim_behind() {
        let at = scratch("unreadable").join("absent.bin");
        let blobs = Blobs::default();
        let said = blobs
            .open(&at, "ghost")
            .expect_err("the file is not there")
            .to_string();
        assert!(said.contains("ghost"), "names the adapter: {said}");
        assert!(blobs.open(&at, "ghost").is_err());
        assert_eq!(blobs.loads(), 2, "each attempt is its own read");
    }

    fn the_resolver_slices_per_layer_and_pads_per_orientation() {
        let (mount, store) = mounted("slice");
        let seats = seats();
        let (built, fingerprint) = store
            .planes("alice-v2", &seats)
            .expect("the resolver reads a well-formed adapter");

        assert_eq!(
            built.len(),
            (2 * LAYERS) as usize,
            "one plane per bank, and the banks are per layer"
        );
        assert_ne!(fingerprint, 0, "the identity's content half is recorded");
        for (name, plane) in &built {
            assert_eq!(
                plane.len() as u64,
                BANK_RANK * HIDDEN * ELEM,
                "`{name}` is one whole slot, which is what `register_adapter` takes"
            );
        }

        let rank = 4usize;
        let hidden = HIDDEN as usize;
        let bank_rank = BANK_RANK as usize;

        let a = &built
            .iter()
            .find(|(name, _)| name == "layer.1.lora_a")
            .expect("layer 1's A")
            .1;
        for row in 0..bank_rank {
            for col in 0..hidden {
                let at = (row * hidden + col) * 2;
                let want = match row < rank {
                    true => source(hidden * rank + row * hidden + col),
                    false => [0, 0],
                };
                assert_eq!(&a[at..at + 2], &want, "A row {row} col {col} of layer 1");
            }
        }

        let b = &built
            .iter()
            .find(|(name, _)| name == "layer.1.lora_b")
            .expect("layer 1's B")
            .1;
        for row in 0..hidden {
            for col in 0..bank_rank {
                let at = (row * bank_rank + col) * 2;
                let want = match col < rank {
                    true => source(hidden * rank + row * rank + col),
                    false => [0, 0],
                };
                assert_eq!(&b[at..at + 2], &want, "B row {row} col {col} of layer 1");
            }
        }
        let _ = std::fs::remove_dir_all(&mount);
    }

    fn the_resolver_refuses_by_name() {
        let mount = scratch("resolver-refusals");
        let mut store = Store::new();
        store.mount(Some(mount.clone()));
        let seats = seats();

        let dir = mount.join("mute");
        std::fs::create_dir_all(&dir).expect("a directory");
        std::fs::write(
            dir.join(MANIFEST),
            "rank = 4\n\n[[plane]]\nrole = \"mystery\"\nfile = \"a.bin\"\n\
             layout = \"rank_major\"\n",
        )
        .expect("a manifest");
        std::fs::write(dir.join("a.bin"), vec![0u8; 4]).expect("a plane");
        let said = store
            .planes("mute", &seats)
            .expect_err("no bank carries that role")
            .to_string();
        assert!(said.contains("mystery"), "names the role: {said}");
        assert!(
            said.contains("layer.0.lora_a"),
            "and the banks there are: {said}"
        );

        write_adapter(&mount, "short", 4, (Layout::RankMajor, Layout::OutMajor));
        std::fs::write(mount.join("short").join("a.bin"), vec![0u8; 16]).expect("a short plane");
        let said = store
            .planes("short", &seats)
            .expect_err("16 bytes is not 3 x 4 x 16 x 2")
            .to_string();
        assert!(said.contains("16"), "names the bytes it was handed: {said}");
        assert!(said.contains("384"), "and the bytes it wanted: {said}");

        write_adapter(&mount, "flipped", 4, (Layout::RankMajor, Layout::RankMajor));
        let said = store
            .planes("flipped", &seats)
            .expect_err("a rank-major B")
            .to_string();
        assert!(said.contains("out-major"), "names the orientation: {said}");
        assert!(said.contains("refused rather than repacked"), "{said}");

        write_adapter(&mount, "wide", 16, (Layout::RankMajor, Layout::OutMajor));
        let said = store
            .planes("wide", &seats)
            .expect_err("rank 16 into a rank-8 bank")
            .to_string();
        assert!(said.contains("rank 16"), "names the source's rank: {said}");
        assert!(said.contains("seats rank 8"), "and the bank's: {said}");

        let said = store
            .planes("wide", &[])
            .expect_err("a load that declares no bank")
            .to_string();
        assert!(said.contains("no bank by that name"), "{said}");
        let _ = std::fs::remove_dir_all(&mount);
    }
}
