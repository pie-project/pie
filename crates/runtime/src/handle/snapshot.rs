//! Snapshots: a working set kept on disk under a name, past its session and
//! its server.
//!
//! A snapshot is a directory, named for its tenant and its name, that appears
//! whole by a rename: its manifest, its recurrent state, and a file for each
//! kv page. Pages live once under `pages/`, by key or by content, and are hard
//! linked in, so turns of one conversation share what they share, and the disk
//! gives its space back with the last link.

use std::collections::HashSet;
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use super::HandleError;
use super::held::{Held, Parts};
use crate::inferlet::ProcessId;
use crate::inferlet::host::working_set::Cursor;
use crate::scheduler::{self, TierKind, TierMove};
use crate::store::kv::hash::Hash256;
use crate::store::kv::page_table::{ExportedPage, ExportedSet, HostKvSlotId, KvPageBacking};
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::registry as store_registry;
use crate::store::rs::working_set::RsWorkingSet;
use crate::store::rs::{RsExport, RsGeometry, RsSlotId, WindowState};

const VERSION: u32 = 1;
/// Names are file names here, written out in hex.
pub(crate) const MAX_NAME_BYTES: usize = 100;
/// Pages read or written at a time: each passes through host slots of the
/// tier live moves use.
const CHUNK_PAGES: usize = 64;
const SYNCS: usize = 8;
const MANIFEST: &str = "manifest.json";
const TOKENS: &str = "token-hashes.bin";
const STATE: &str = "state.bin";
const KV: &str = "kv";
const SHARED: &str = "pages";

static STORE: OnceLock<(PathBuf, u64)> = OnceLock::new();
/// One save, load or delete at a time: a page's link count is what says
/// whether it is still read.
static BUSY: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub(crate) fn configure(dir: PathBuf, budget: u64) {
    if let Err(error) = sweep(&dir) {
        tracing::warn!(%error, "snapshots: leftovers of a crash stay on disk");
    }
    let _ = STORE.set((dir, budget));
}

/// What a crash left: saves that never landed, deletes that never finished,
/// and pages nothing links now.
fn sweep(root: &Path) -> std::io::Result<()> {
    let tenants = match std::fs::read_dir(root) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        other => other?,
    };
    for tenant in tenants {
        let tenant = tenant?.path();
        for entry in std::fs::read_dir(&tenant)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with("tmp-") {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
        let shared = match std::fs::read_dir(tenant.join(SHARED)) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            other => other?,
        };
        for domain in shared {
            for page in std::fs::read_dir(domain?.path())? {
                let page = page?;
                if page.metadata()?.nlink() == 1 {
                    std::fs::remove_file(page.path())?;
                }
            }
        }
    }
    Ok(())
}

fn store() -> Result<&'static (PathBuf, u64), HandleError> {
    STORE
        .get()
        .ok_or_else(|| HandleError::invalid("this server keeps no snapshots"))
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    name: String,
    /// What says two runs' pages are one another's: the model and its layout.
    domain: String,
    page_tokens: u32,
    tokens: u64,
    bytes: u64,
    cursor: SavedCursor,
    page_len: u64,
    folded: u64,
    prompt_pages: u64,
    logged: Vec<u32>,
    /// Each page's key, where sealed.
    pages: Vec<Option<String>>,
    /// Each page's file under `pages/`, which holds what `kv/` links.
    files: Vec<String>,
    state: Option<SavedState>,
}

/// What writing a snapshot's pages came to.
struct Spent {
    files: Vec<String>,
    bytes: u64,
}

#[derive(Serialize, Deserialize)]
struct SavedCursor {
    folded: u32,
    tail: Option<u32>,
    reserved: u32,
    spans: Vec<(u32, u32, u32)>,
}

#[derive(Serialize, Deserialize)]
struct SavedState {
    state_size: u64,
    buffer_page_tokens: u32,
    folded: bool,
    buffer: Vec<bool>,
    base: u32,
    frozen: u32,
    bytes: u64,
    /// The windows' position, for a model whose state is one window list:
    /// `buffer` is then its table, and `window_tokens` the window.
    #[serde(default)]
    window: Option<WindowState>,
    #[serde(default)]
    window_tokens: u32,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.is_ascii() || !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len() / 2)
        .map(|at| u8::from_str_radix(&text[at * 2..at * 2 + 2], 16).ok())
        .collect()
}

fn unhash(text: &str) -> Option<Hash256> {
    unhex(text)?.try_into().ok()
}

fn tenant_dir(tenant: &str) -> Result<PathBuf, HandleError> {
    Ok(store()?.0.join(format!("t{}", hex(tenant.as_bytes()))))
}

fn dir_of(tenant: &str, name: &str) -> Result<PathBuf, HandleError> {
    if name.is_empty() || name.len() > MAX_NAME_BYTES {
        return Err(HandleError::invalid(format!(
            "a snapshot's name must be 1 to {MAX_NAME_BYTES} bytes"
        )));
    }
    Ok(tenant_dir(tenant)?.join(format!("s{}", hex(name.as_bytes()))))
}

fn no_snapshot(name: &str) -> HandleError {
    HandleError::new(
        client_api::HandleErrorCode::Unknown,
        format!("no snapshot named `{name}`"),
    )
}

fn failed(what: &str, error: impl std::fmt::Display) -> HandleError {
    HandleError::invalid(format!("{what}: {error}"))
}

/// Runs what touches the disk off the session's task.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> std::io::Result<T> + Send + 'static,
) -> Result<T, HandleError> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|error| failed("the disk task", error))?
        .map_err(|error| failed("the snapshot store", error))
}

/// The bytes on disk under `root`, a linked file once.
fn used(root: &Path) -> std::io::Result<u64> {
    let (mut seen, mut bytes, mut dirs) = (HashSet::new(), 0, vec![root.to_path_buf()]);
    while let Some(dir) = dirs.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                dirs.push(entry.path());
            } else if seen.insert(meta.ino()) {
                bytes += meta.len();
            }
        }
    }
    Ok(bytes)
}

fn alloc_slots(stores: &store_registry::Stores, n: usize) -> Result<Vec<u32>, HandleError> {
    store_registry::with_kv_lock(&stores.kv, "snapshot-host", |kv| kv.alloc_host_slots(n))
        .map(|slots| slots.into_iter().map(|slot| slot.0).collect())
        .ok_or_else(|| HandleError::quota(format!("the host has no room for {n} kv pages")))
}

fn free_slots(stores: &store_registry::Stores, slots: Vec<u32>) {
    store_registry::with_kv_lock(&stores.kv, "snapshot-host", |kv| {
        kv.free_host_slots(slots.into_iter().map(HostKvSlotId).collect())
    });
}

/// Reads `page`'s run of backings out, a kind at a time, in their order.
async fn read_pages(
    engine: usize,
    stores: &store_registry::Stores,
    backings: &[KvPageBacking],
) -> Result<Vec<u8>, HandleError> {
    let scratch = alloc_slots(stores, backings.len())?;
    let read = async {
        let mut out: Vec<Vec<u8>> = vec![Vec::new(); backings.len()];
        let swapped: Vec<(usize, u32)> = backings
            .iter()
            .enumerate()
            .filter_map(|(at, backing)| match backing {
                KvPageBacking::Swapped(slot) => Some((at, slot.0)),
                _ => None,
            })
            .collect();
        let mut pieces: Vec<(Vec<usize>, Vec<u8>)> = Vec::new();
        if !swapped.is_empty() {
            let slots = swapped.iter().map(|&(_, slot)| slot).collect();
            let bytes = scheduler::read_host_kv(engine, slots).await;
            pieces.push((
                swapped.iter().map(|&(at, _)| at).collect(),
                bytes.map_err(|error| failed("reading host pages", error))?,
            ));
        }
        for tier in [true, false] {
            let run: Vec<(usize, u32)> = backings
                .iter()
                .enumerate()
                .filter_map(|(at, backing)| match (backing, tier) {
                    (KvPageBacking::Resident(id), true) => Some((at, id.0)),
                    (KvPageBacking::OnDisk(slot), false) => Some((at, slot.0)),
                    _ => None,
                })
                .collect();
            if run.is_empty() {
                continue;
            }
            let domain = if tier {
                scheduler::device_domain(engine)
            } else {
                ::engine::MemoryDomain::LocalDisk
            };
            let pages = run.iter().map(|&(_, page)| page).collect();
            let slots = scratch[..run.len()].to_vec();
            let bytes = scheduler::read_tier_kv(engine, domain, pages, slots).await;
            pieces.push((
                run.iter().map(|&(at, _)| at).collect(),
                bytes.map_err(|error| failed("reading pages", error))?,
            ));
        }
        for (indexes, bytes) in pieces {
            let each = bytes.len() / indexes.len().max(1);
            for (chunk, at) in bytes.chunks(each.max(1)).zip(indexes) {
                out[at] = chunk.to_vec();
            }
        }
        Ok::<_, HandleError>(out.concat())
    }
    .await;
    free_slots(stores, scratch);
    read
}

/// A set's recurrent state's bytes, and what the manifest says of it.
async fn read_state(
    engine: usize,
    rs: &RsWorkingSet,
    set: &RsExport,
) -> Result<(SavedState, Vec<u8>), HandleError> {
    let stores = store_registry::get(rs.model, rs.engine);
    let held: Vec<RsSlotId> = set.folded.iter().copied().collect();
    if set.windows.len() > 1 {
        return Err(HandleError::invalid(
            "a state of several window lists is not saved to the host",
        ));
    }
    // A window list's table rides as the buffer.
    let table = set.windows.first().unwrap_or(&set.buffer);
    let pages: Vec<RsSlotId> = table.iter().flatten().copied().collect();
    let (ids, scratch) = {
        let mut store = stores.rs.lock().unwrap();
        let on_device = |ids: &[RsSlotId], store: &crate::store::rs::RsStore| {
            ids.iter().filter(|&&id| !store.on_host(id)).count()
        };
        let (slots, buffer) = (on_device(&held, &store), on_device(&pages, &store));
        let scratch = store.take_host(slots, buffer).ok_or_else(|| {
            HandleError::quota("the host has no room to read this recurrent state out")
        })?;
        (
            held.iter()
                .chain(&pages)
                .map(|&id| (id, store.on_host(id), store.engine_id(id)))
                .collect::<Vec<_>>(),
            scratch
                .iter()
                .map(|&id| (id, store.engine_id(id)))
                .collect::<Vec<_>>(),
        )
    };
    let read = async {
        let (mut free, mut row_slots, mut row_pages) = (scratch.iter(), Vec::new(), Vec::new());
        let (mut slot_moves, mut page_moves) = ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()));
        for (at, &(_, hosted, id)) in ids.iter().enumerate() {
            let is_slot = at < held.len();
            let row = if hosted {
                id
            } else {
                let &(_, row) = free.next().expect("one row for each off the host");
                let (device, slots) = if is_slot {
                    (&mut slot_moves.0, &mut slot_moves.1)
                } else {
                    (&mut page_moves.0, &mut page_moves.1)
                };
                device.push(id);
                slots.push(row);
                row
            };
            if is_slot {
                row_slots.push(row);
            } else {
                row_pages.push(row);
            }
        }
        let pages_kind = if set.window.is_some() {
            TierKind::Window
        } else {
            TierKind::RsPages
        };
        for (kind, (device, slots)) in [(TierKind::Rs, slot_moves), (pages_kind, page_moves)] {
            if device.is_empty() {
                continue;
            }
            let copy = TierMove {
                kind,
                tier: ::engine::MemoryDomain::HostPinned,
                out: true,
                device,
                slots,
                source: None,
            };
            let completion = scheduler::copy_tier_tracked(engine, &copy)
                .map_err(|error| failed("copying state to the host", error))?;
            completion
                .wait()
                .await
                .map_err(|error| failed("copying state to the host", error))?;
        }
        scheduler::read_host_state(engine, row_slots, row_pages)
            .await
            .map_err(|error| failed("reading recurrent state", error))
    }
    .await;
    stores
        .rs
        .lock()
        .unwrap()
        .give_host(scratch.into_iter().map(|(id, _)| id).collect());
    let bytes = read?;
    Ok((
        SavedState {
            state_size: rs.geom.state_size,
            buffer_page_tokens: rs.geom.buffer_page_tokens,
            folded: set.folded.is_some(),
            buffer: table.iter().map(Option::is_some).collect(),
            base: set.base,
            frozen: set.frozen,
            bytes: bytes.len() as u64,
            window: set.window,
            window_tokens: crate::model::model().rs_caps().window_tokens,
        },
        bytes,
    ))
}

fn token_bytes(pages: &[ExportedPage]) -> Vec<u8> {
    let mut out = Vec::new();
    for page in pages {
        out.extend_from_slice(&(page.token_hashes.len() as u32).to_le_bytes());
        for hash in &page.token_hashes {
            match hash {
                Some(hash) => {
                    out.push(1);
                    out.extend_from_slice(hash);
                }
                None => out.push(0),
            }
        }
    }
    out
}

fn token_hashes(mut bytes: &[u8], pages: usize) -> Option<Vec<Vec<Option<Hash256>>>> {
    let mut out = Vec::with_capacity(pages);
    for _ in 0..pages {
        let (count, rest) = bytes.split_first_chunk::<4>()?;
        bytes = rest;
        let mut page = Vec::with_capacity(u32::from_le_bytes(*count) as usize);
        for _ in 0..u32::from_le_bytes(*count) {
            let (&flag, rest) = bytes.split_first()?;
            bytes = rest;
            if flag == 0 {
                page.push(None);
            } else {
                let (hash, rest) = bytes.split_first_chunk::<32>()?;
                bytes = rest;
                page.push(Some(*hash));
            }
        }
        out.push(page);
    }
    Some(out)
}

/// Keeps `held` under `name`. It stays as it was.
pub(super) async fn save(held: &Held, tenant: &str, name: &str) -> Result<Value, HandleError> {
    let dir = dir_of(tenant, name)?;
    let (root, budget) = store()?.clone();
    let tenant_root = tenant_dir(tenant)?;
    let _busy = BUSY.lock().await;
    if dir.exists() {
        return Err(HandleError::new(
            client_api::HandleErrorCode::InUse,
            format!("a snapshot named `{name}` exists"),
        ));
    }
    let Parts { kv, rs, cursor } = held.parts();
    let stores = store_registry::get(kv.model, kv.engine);
    let (pin, set) = held.pinned().await.map_err(HandleError::moving)?;
    let wrote = async {
        let rs_set = match rs {
            Some(rs) => {
                let set = stores.rs.lock().unwrap().export(rs.id);
                let set = set.map_err(|error| failed("reading recurrent state", error))?;
                Some((rs, set))
            }
            None => None,
        };
        let state = match &rs_set {
            Some((rs, set)) => Some(read_state(kv.engine, rs, set).await?),
            None => None,
        };
        let tmp = tenant_root.join(format!("tmp-{}", Uuid::new_v4()));
        let shared = tenant_root.join(SHARED).join(hex(&store_domain(&stores)));
        let (t, s) = (tmp.clone(), shared.clone());
        blocking(move || {
            std::fs::create_dir_all(t.join(KV))?;
            std::fs::create_dir_all(&s)
        })
        .await?;
        let mut created = Vec::new();
        // Written beside the pages, never shared: it counts in full.
        let state_len = state.as_ref().map_or(0, |(_, bytes)| bytes.len() as u64);
        let spent = write_pages(
            kv.engine,
            &stores,
            &set,
            &tmp,
            &shared,
            budget.saturating_sub(state_len),
            &root,
            &mut created,
        )
        .await;
        let result = match spent {
            Ok((files, bytes)) => {
                let spent = Spent { files, bytes };
                finish(
                    kv,
                    &stores,
                    cursor,
                    name,
                    set,
                    state,
                    spent,
                    tmp.clone(),
                    dir,
                )
                .await
            }
            Err(error) => Err(error),
        };
        if result.is_err() {
            // The pages this save put down are linked from nothing once its
            // directory is gone: left there, they would count against the
            // budget until a restart sweeps them.
            let _ = blocking(move || {
                std::fs::remove_dir_all(&tmp)?;
                unlink_unread(&shared, &created)
            })
            .await;
        }
        result
    }
    .await;
    store_registry::with_kv_lock(&stores.kv, "snapshot-pin", |kv| kv.release_reader(pin));
    wrote
}

fn store_domain(stores: &store_registry::Stores) -> Hash256 {
    store_registry::with_kv_lock(&stores.kv, "snapshot-domain", |kv| kv.domain())
}

/// Removes each of `names` under `shared` that no snapshot links, with what a
/// write of it cut short left.
fn unlink_unread(shared: &Path, names: &[String]) -> std::io::Result<()> {
    for name in names {
        for path in [shared.join(name), shared.join(format!("part-{name}"))] {
            match std::fs::metadata(&path) {
                Ok(meta) if meta.nlink() == 1 => std::fs::remove_file(&path)?,
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}

/// Each page into `tmp`, linked from `shared` where a page of the same bytes
/// is there already: the file names, and the bytes written. A sealed page's
/// key says so without reading it; one a decode wrote has none, so its bytes
/// are read and hashed. What it puts down under `shared` is named in
/// `created` as it goes, so a save that fails can take it back. A run that
/// would pass `budget`, what the pages may take, is refused before it is
/// written.
#[allow(clippy::too_many_arguments, reason = "one snapshot's parts")]
async fn write_pages(
    engine: usize,
    stores: &store_registry::Stores,
    set: &ExportedSet,
    tmp: &Path,
    shared: &Path,
    budget: u64,
    root: &Path,
    created: &mut Vec<String>,
) -> Result<(Vec<String>, u64), HandleError> {
    let mut files = vec![String::new(); set.pages.len()];
    let mut wanted = Vec::new();
    for (at, page) in set.pages.iter().enumerate() {
        match &page.key {
            Some(key) if shared.join(hex(key)).exists() => files[at] = hex(key),
            _ => wanted.push(at),
        }
    }
    let have = {
        let root = root.to_path_buf();
        blocking(move || used(&root)).await?
    };
    let refuse = |total: u64| {
        HandleError::quota(format!(
            "snapshots would pass the budget by {} bytes; delete one, or raise \
             `[runtime] snapshot_budget`",
            total - budget
        ))
    };
    if have > budget {
        return Err(refuse(have));
    }
    let mut written = 0;
    for run in wanted.chunks(CHUNK_PAGES) {
        let backings: Vec<_> = run.iter().map(|&at| set.pages[at].backing).collect();
        let bytes = read_pages(engine, stores, &backings).await?;
        let each = bytes.len() / run.len();
        let named: Vec<(usize, String, Vec<u8>)> = run
            .iter()
            .zip(bytes.chunks(each))
            .map(|(&at, page)| {
                let name = match &set.pages[at].key {
                    Some(key) => hex(key),
                    None => format!("c{}", hex(blake3::hash(page).as_bytes())),
                };
                (at, name, page.to_vec())
            })
            .collect();
        for (at, name, _) in &named {
            files[*at] = name.clone();
        }
        let shared = shared.to_path_buf();
        let fresh: Vec<(usize, String, Vec<u8>)> = {
            let shared = shared.clone();
            blocking(move || {
                let mut seen = HashSet::new();
                Ok(named
                    .into_iter()
                    .filter(|(_, name, _)| seen.insert(name.clone()) && !shared.join(name).exists())
                    .collect())
            })
            .await?
        };
        let adding: u64 = fresh.iter().map(|(_, _, bytes)| bytes.len() as u64).sum();
        if have + written + adding > budget {
            return Err(refuse(have + written + adding));
        }
        created.extend(fresh.iter().map(|(_, name, _)| name.clone()));
        written += blocking(move || {
            let fresh: Vec<_> = fresh.iter().collect();
            // A sync is the cost of a save: they run side by side.
            let each = fresh.len().div_ceil(SYNCS).max(1);
            std::thread::scope(|scope| {
                let threads: Vec<_> = fresh
                    .chunks(each)
                    .map(|pages| scope.spawn(|| write_shared(&shared, pages)))
                    .collect();
                threads
                    .into_iter()
                    .try_for_each(|thread| thread.join().expect("a page's write panicked"))
            })?;
            Ok(fresh
                .iter()
                .map(|(_, _, bytes)| bytes.len() as u64)
                .sum::<u64>())
        })
        .await?;
    }
    let (tmp, shared, names) = (tmp.join(KV), shared.to_path_buf(), files.clone());
    blocking(move || {
        for (at, name) in names.iter().enumerate() {
            std::fs::hard_link(shared.join(name), tmp.join(at.to_string()))?;
        }
        Ok(())
    })
    .await?;
    Ok((files, written))
}

fn write_shared(shared: &Path, pages: &[&(usize, String, Vec<u8>)]) -> std::io::Result<()> {
    for (_, name, bytes) in pages {
        let part = shared.join(format!("part-{name}"));
        let mut file = std::fs::File::create(&part)?;
        file.write_all(bytes)?;
        file.sync_data()?;
        std::fs::rename(part, shared.join(name))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, reason = "one snapshot's parts")]
async fn finish(
    kv: &KvWorkingSet,
    stores: &store_registry::Stores,
    cursor: &Cursor,
    name: &str,
    set: ExportedSet,
    state: Option<(SavedState, Vec<u8>)>,
    spent: Spent,
    tmp: PathBuf,
    dir: PathBuf,
) -> Result<Value, HandleError> {
    let (state, state_bytes) = match state {
        Some((saved, bytes)) => (Some(saved), bytes),
        None => (None, Vec::new()),
    };
    let tokens = u64::from(cursor.tail.unwrap_or(cursor.folded));
    let bytes = spent.bytes + state_bytes.len() as u64;
    let manifest = Manifest {
        version: VERSION,
        name: name.to_string(),
        domain: hex(&store_domain(stores)),
        page_tokens: kv.page_size,
        tokens,
        bytes,
        cursor: SavedCursor {
            folded: cursor.folded,
            tail: cursor.tail,
            reserved: cursor.reserved,
            spans: cursor.spans.clone(),
        },
        page_len: set.page_len,
        folded: set.folded,
        prompt_pages: set.prompt_pages,
        logged: set.logged.clone(),
        pages: set
            .pages
            .iter()
            .map(|page| page.key.as_ref().map(|key| hex(key)))
            .collect(),
        files: spent.files,
        state,
    };
    let text = serde_json::to_vec(&manifest).map_err(|error| failed("the manifest", error))?;
    let hashes = token_bytes(&set.pages);
    blocking(move || {
        std::fs::write(tmp.join(STATE), state_bytes)?;
        std::fs::write(tmp.join(TOKENS), hashes)?;
        std::fs::write(tmp.join(MANIFEST), text)?;
        sync_disk(&tmp)?;
        std::fs::rename(&tmp, &dir)
    })
    .await?;
    Ok(json!({ "ok": true, "tokens": tokens, "bytes": bytes }))
}

/// What is written before the rename stands past a crash of the host, not
/// only of this process.
fn sync_disk(dir: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        std::fs::File::open(entry.path())?.sync_all()?;
    }
    std::fs::File::open(dir)?.sync_all()
}

/// A working set as `name` stands, off the device: the next process to bind
/// it brings it back as it would an evicted one.
pub(super) async fn load(holder: ProcessId, tenant: &str, name: &str) -> Result<Held, HandleError> {
    let dir = dir_of(tenant, name)?;
    let _busy = BUSY.lock().await;
    let engine = 0;
    let stores = store_registry::get(0, engine);
    if !dir.exists() {
        return Err(no_snapshot(name));
    }
    let d = dir.clone();
    let manifest: Manifest = blocking(move || {
        serde_json::from_slice(&std::fs::read(d.join(MANIFEST))?).map_err(std::io::Error::other)
    })
    .await?;
    if manifest.version != VERSION {
        return Err(HandleError::invalid(format!(
            "snapshot `{name}` is of version {}",
            manifest.version
        )));
    }
    if manifest.domain != hex(&store_domain(&stores)) || manifest.page_tokens != stores.kv_page_size
    {
        return Err(HandleError::invalid(format!(
            "snapshot `{name}` was kept by another model or layout"
        )));
    }
    let count = manifest.pages.len();
    let slots = alloc_slots(&stores, count)?;
    let filled = async {
        for run in (0..count).collect::<Vec<_>>().chunks(CHUNK_PAGES) {
            let files: Vec<PathBuf> = run
                .iter()
                .map(|at| dir.join(KV).join(at.to_string()))
                .collect();
            let bytes = blocking(move || {
                let mut all = Vec::new();
                for file in files {
                    all.extend(std::fs::read(file)?);
                }
                Ok(all)
            })
            .await?;
            let ids = run.iter().map(|&at| slots[at]).collect();
            scheduler::write_host_kv(engine, ids, bytes)
                .await
                .map_err(|error| failed("writing pages to the host", error))?;
        }
        Ok::<_, HandleError>(())
    }
    .await;
    if let Err(error) = filled {
        free_slots(&stores, slots);
        return Err(error);
    }
    let (d, n) = (dir.clone(), count);
    let (hashes, state_bytes) = match blocking(move || {
        let hashes = std::fs::read(d.join(TOKENS))?;
        let hashes = token_hashes(&hashes, n)
            .ok_or_else(|| std::io::Error::other("its token hashes are cut short"))?;
        Ok((hashes, std::fs::read(d.join(STATE))?))
    })
    .await
    {
        Ok(read) => read,
        Err(error) => {
            free_slots(&stores, slots);
            return Err(error);
        }
    };
    let rs = match &manifest.state {
        Some(saved) => match load_state(engine, saved, &state_bytes).await {
            Ok(rs) => Some(rs),
            Err(error) => {
                free_slots(&stores, slots);
                return Err(error);
            }
        },
        None => None,
    };
    let Manifest {
        pages,
        cursor,
        page_len,
        folded,
        prompt_pages,
        logged,
        ..
    } = manifest;
    let exported = ExportedSet {
        pages: pages
            .into_iter()
            .zip(hashes)
            .map(|(key, token_hashes)| ExportedPage {
                backing: KvPageBacking::Swapped(HostKvSlotId(0)),
                key: key.as_deref().and_then(unhash),
                token_hashes,
            })
            .collect(),
        page_len,
        folded,
        prompt_pages,
        logged,
    };
    let host_slots = slots.into_iter().map(HostKvSlotId).collect();
    let id = store_registry::with_kv_lock(&stores.kv, "snapshot-load", |kv| {
        kv.import(exported, host_slots)
    })
    .map_err(|error| failed("taking the pages in", error))?;
    let kv = KvWorkingSet::new(0, engine, id, stores.kv_page_size);
    let cursor = Cursor {
        folded: cursor.folded,
        folded_exact: true,
        unsettled: 0,
        tail: cursor.tail,
        reserved: cursor.reserved,
        spans: cursor.spans,
    };
    Ok(Held::park(holder, kv, rs, cursor))
}

async fn load_state(
    engine: usize,
    saved: &SavedState,
    bytes: &[u8],
) -> Result<RsWorkingSet, HandleError> {
    let stores = store_registry::get(0, engine);
    let geom = RsGeometry {
        state_size: saved.state_size,
        buffer_page_tokens: saved.buffer_page_tokens,
    };
    let caps = crate::model::model().rs_caps();
    if !caps.has_state()
        || caps.state_size != geom.state_size
        || caps.buffer_page_size != geom.buffer_page_tokens
        || caps.window_tokens != saved.window_tokens
        || (caps.window_tokens > 0) != saved.window.is_some()
    {
        return Err(HandleError::invalid(
            "its recurrent state is not this model's",
        ));
    }
    let slots = usize::from(saved.folded);
    let pages = saved.buffer.iter().filter(|&&page| page).count();
    let (rows, ids) = {
        let mut store = stores.rs.lock().unwrap();
        let rows = store
            .take_host(slots, pages)
            .ok_or_else(|| HandleError::quota("the host has no room for this recurrent state"))?;
        let ids: Vec<u32> = rows.iter().map(|&id| store.engine_id(id)).collect();
        (rows, ids)
    };
    let (row_slots, row_pages) = ids.split_at(slots);
    let written = scheduler::write_host_state(
        engine,
        row_slots.to_vec(),
        row_pages.to_vec(),
        bytes.to_vec(),
    )
    .await;
    let mut store = stores.rs.lock().unwrap();
    if let Err(error) = written {
        store.give_host(rows);
        return Err(failed("writing recurrent state to the host", error));
    }
    let mut buffered = rows[slots..].iter().copied();
    let table: Vec<Option<RsSlotId>> = saved
        .buffer
        .iter()
        .map(|&page| page.then(|| buffered.next().expect("counted above")))
        .collect();
    let windowed = saved.window.is_some();
    let set = RsExport {
        folded: saved.folded.then(|| rows[0]),
        buffer: if windowed { Vec::new() } else { table.clone() },
        base: saved.base,
        frozen: saved.frozen,
        windows: if windowed { vec![table] } else { Vec::new() },
        window: saved.window,
    };
    let id = store.import(geom, set);
    Ok(RsWorkingSet::new(0, engine, id, geom))
}

/// Gives back a name; what its pages share with others stays.
pub(super) async fn delete(tenant: &str, name: &str) -> Result<Value, HandleError> {
    let dir = dir_of(tenant, name)?;
    let root = tenant_dir(tenant)?;
    let _busy = BUSY.lock().await;
    if !dir.exists() {
        return Err(no_snapshot(name));
    }
    let domain = hex(&store_domain(&store_registry::get(0, 0)));
    blocking(move || {
        let gone = root.join(format!("tmp-{}", Uuid::new_v4()));
        std::fs::rename(&dir, &gone)?;
        let manifest: Option<Manifest> = std::fs::read(gone.join(MANIFEST))
            .ok()
            .and_then(|text| serde_json::from_slice(&text).ok());
        std::fs::remove_dir_all(&gone)?;
        let domain = manifest.as_ref().map_or(domain, |m| m.domain.clone());
        for file in manifest.iter().flat_map(|m| m.files.iter()) {
            let page = root.join(SHARED).join(&domain).join(file);
            if std::fs::metadata(&page).is_ok_and(|meta| meta.nlink() == 1) {
                std::fs::remove_file(page)?;
            }
        }
        Ok(())
    })
    .await?;
    Ok(json!({ "ok": true }))
}

pub(super) async fn list(tenant: &str, prefix: &str) -> Result<Value, HandleError> {
    let root = tenant_dir(tenant)?;
    let prefix = prefix.to_string();
    let snapshots = blocking(move || {
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&root) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            other => other?,
        };
        for entry in entries {
            let entry = entry?;
            let file = entry.file_name();
            let Some(name) = file
                .to_str()
                .and_then(|name| name.strip_prefix('s'))
                .and_then(unhex)
                .and_then(|name| String::from_utf8(name).ok())
            else {
                continue;
            };
            if !name.starts_with(&prefix) {
                continue;
            }
            let Ok(text) = std::fs::read(entry.path().join(MANIFEST)) else {
                continue;
            };
            let Ok(manifest) = serde_json::from_slice::<Manifest>(&text) else {
                continue;
            };
            out.push(json!({ "name": name, "tokens": manifest.tokens, "bytes": manifest.bytes }));
        }
        out.sort_by_key(|snapshot| snapshot["name"].to_string());
        Ok(out)
    })
    .await?;
    Ok(json!({ "snapshots": snapshots }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crash_leaves_nothing_a_restart_keeps() {
        let root = std::env::temp_dir().join(format!("pie-snapshots-{}", Uuid::new_v4()));
        let tenant = root.join("t00");
        let pages = tenant.join(SHARED).join("domain");
        std::fs::create_dir_all(tenant.join("tmp-1").join(KV)).unwrap();
        std::fs::create_dir_all(tenant.join("s00").join(KV)).unwrap();
        std::fs::create_dir_all(&pages).unwrap();
        std::fs::write(pages.join("linked"), b"a").unwrap();
        std::fs::write(pages.join("orphan"), b"b").unwrap();
        std::fs::write(pages.join("part-x"), b"c").unwrap();
        std::fs::hard_link(pages.join("linked"), tenant.join("s00").join(KV).join("0")).unwrap();

        sweep(&root).unwrap();

        assert!(!tenant.join("tmp-1").exists());
        assert!(pages.join("linked").exists(), "a snapshot reads it");
        assert!(!pages.join("orphan").exists());
        assert!(!pages.join("part-x").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_failed_save_takes_back_only_what_nothing_links() {
        let root = std::env::temp_dir().join(format!("pie-snapshots-{}", Uuid::new_v4()));
        let shared = root.join(SHARED).join("domain");
        let kept = root.join("s00").join(KV);
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::create_dir_all(&kept).unwrap();
        std::fs::write(shared.join("mine"), b"a").unwrap();
        std::fs::write(shared.join("part-cut"), b"b").unwrap();
        std::fs::write(shared.join("theirs"), b"c").unwrap();
        std::fs::hard_link(shared.join("theirs"), kept.join("0")).unwrap();
        std::fs::write(shared.join("unnamed"), b"d").unwrap();

        let names = ["mine", "cut", "theirs", "never"].map(String::from);
        unlink_unread(&shared, &names).unwrap();

        assert!(!shared.join("mine").exists());
        assert!(!shared.join("part-cut").exists(), "a write cut short");
        assert!(shared.join("theirs").exists(), "a snapshot links it");
        assert!(shared.join("unnamed").exists(), "not this save's");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn token_hashes_read_back_as_they_were_written() {
        let page = |token_hashes| ExportedPage {
            backing: KvPageBacking::Swapped(HostKvSlotId(0)),
            key: None,
            token_hashes,
        };
        let pages = [page(vec![Some([7; 32]), None]), page(Vec::new())];
        let bytes = token_bytes(&pages);
        let back = token_hashes(&bytes, 2).unwrap();
        assert_eq!(back[0], [Some([7; 32]), None]);
        assert!(back[1].is_empty());
        assert!(token_hashes(&bytes[..bytes.len() - 5], 2).is_none());
    }
}
