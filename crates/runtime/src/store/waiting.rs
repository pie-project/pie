//! The cached prefix of a process waiting for admission: its prompt's tokens
//! (`prompt-tokens`, or a completion's `raw-prompt` tokenized as the compat
//! inferlet will) looked up in the kv stores when it arrives, and again when
//! an entry that can change what it finds joins or leaves a store's cache,
//! until it starts. It reads the stores and holds no page: what to keep for
//! it is the policy's choice. The policy hears it as
//! [`Event::Waiting`](crate::policy::Event::Waiting), which stands as the
//! process's `cached-tokens` while it is pending.
//!
//! `cached-tokens` means two things, by the process's state. Pending: what
//! this module finds, the leading tokens of its prompt some engine's store
//! holds now, on any tier. Running and after: the tokens its fires mapped in
//! from the cache. Only a prompt the runtime has at arrival is looked up, so
//! a chat request (its tokens come from the inferlet's template, rendered once
//! admitted) reads none while it waits: a policy that ranks waiting work by
//! `cached-tokens` (lpm) passes chat over, and one that means to rank chat
//! reads `cached-tokens-estimate`.
//!
//! A cache change wakes only the prompts it can move. Each waiting prompt is
//! indexed, in each store, by the page key it needs next and by the last key
//! of the run it holds: an entry joining wakes the prompts whose next key is
//! on its path, and one leaving those whose last held key is (a page of a
//! run cannot go while a later one stays, the keys being chained).

use crate::policy::share::ShareAs;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use client_api::props::{self, Props};

use crate::inferlet::process::ProcessId;
use crate::store::kv::{KvStore, hash};
use crate::store::registry::{self, Stores};

use hash::Hash256;

/// The model whose stores a waiting prompt is looked up in.
const MODEL: usize = 0;

/// A store, by its kv lock's address: the index keeps one per engine.
type StoreId = usize;

fn store_id(stores: &Stores) -> StoreId {
    Arc::as_ptr(&stores.kv) as usize
}

/// A waiting prompt in one store: its page keys, and how many of the leading
/// ones the store holds.
struct Look {
    keys: Arc<[Hash256]>,
    run: usize,
}

impl Look {
    fn next(&self) -> Option<Hash256> {
        self.keys.get(self.run).copied()
    }

    fn last(&self) -> Option<Hash256> {
        self.run.checked_sub(1).map(|at| self.keys[at])
    }
}

/// One waiting process: its look in each store (none until its keys are
/// hashed), and the tokens last reported.
struct Watch {
    looks: HashMap<StoreId, Look>,
    page_size: HashMap<StoreId, u32>,
    cached: Option<u64>,
}

impl Watch {
    /// The tokens of the leading pages some store holds: the most any one
    /// holds.
    fn tokens(&self) -> u64 {
        self.looks
            .iter()
            .map(|(id, look)| look.run as u64 * u64::from(self.page_size[id]))
            .max()
            .unwrap_or(0)
    }
}

/// Pids by a page key, in each store.
type Index = HashMap<StoreId, HashMap<Hash256, HashSet<ProcessId>>>;

#[derive(Default)]
struct Waiting {
    watches: HashMap<ProcessId, Watch>,
    /// By the key each needs next.
    next: Index,
    /// By the last key of the run each holds.
    last: Index,
}

fn add(index: &mut Index, store: StoreId, key: Option<Hash256>, pid: ProcessId) {
    if let Some(key) = key {
        index
            .entry(store)
            .or_default()
            .entry(key)
            .or_default()
            .insert(pid);
    }
}

fn remove(index: &mut Index, store: StoreId, key: Option<Hash256>, pid: ProcessId) {
    let Some(key) = key else { return };
    let Some(keys) = index.get_mut(&store) else {
        return;
    };
    if let Some(pids) = keys.get_mut(&key) {
        pids.remove(&pid);
        if pids.is_empty() {
            keys.remove(&key);
        }
    }
}

impl Waiting {
    fn index(&mut self, pid: ProcessId, store: StoreId, look: &Look) {
        add(&mut self.next, store, look.next(), pid);
        add(&mut self.last, store, look.last(), pid);
    }

    fn unindex(&mut self, pid: ProcessId, store: StoreId, look: &Look) {
        remove(&mut self.next, store, look.next(), pid);
        remove(&mut self.last, store, look.last(), pid);
    }

    /// The pids a store's change can move: those whose next key joined, or
    /// whose last held key left.
    fn woken(
        &self,
        store: StoreId,
        joined: &HashSet<Hash256>,
        left: &HashSet<Hash256>,
    ) -> HashSet<ProcessId> {
        let by = |index: &Index, keys: &HashSet<Hash256>| -> Vec<ProcessId> {
            let Some(index) = index.get(&store) else {
                return Vec::new();
            };
            keys.iter()
                .filter_map(|key| index.get(key))
                .flatten()
                .copied()
                .collect()
        };
        by(&self.next, joined)
            .into_iter()
            .chain(by(&self.last, left))
            .collect()
    }

    /// Sets each pid's run in `store`, and the changed counts to report.
    fn settle(
        &mut self,
        store: StoreId,
        runs: Vec<(ProcessId, usize)>,
        reports: &mut Vec<(ProcessId, u64)>,
    ) {
        for (pid, run) in runs {
            let Some(look) = self
                .watches
                .get_mut(&pid)
                .and_then(|watch| watch.looks.remove(&store))
            else {
                continue;
            };
            self.unindex(pid, store, &look);
            let look = Look { run, ..look };
            self.index(pid, store, &look);
            let watch = self.watches.get_mut(&pid).expect("looked up above");
            watch.looks.insert(store, look);
            let tokens = watch.tokens();
            if watch.cached != Some(tokens) {
                watch.cached = Some(tokens);
                reports.push((pid, tokens));
            }
        }
    }

    fn forget(&mut self, pid: ProcessId) {
        if let Some(watch) = self.watches.remove(&pid) {
            for (store, look) in &watch.looks {
                self.unindex(pid, *store, look);
            }
        }
        WATCHED.store(!self.watches.is_empty(), Ordering::Relaxed);
    }
}

static WAITING: LazyLock<Mutex<Waiting>> = LazyLock::new(Default::default);
/// Some prompt waits: the stores keep the keys their changes touch.
static WATCHED: AtomicBool = AtomicBool::new(false);
/// The refresh task runs.
static REFRESHING: AtomicBool = AtomicBool::new(false);

/// Refreshes run (one for each store whose cache changed under a waiting
/// prompt).
static REFRESHES: AtomicU64 = AtomicU64::new(0);
/// Prompts looked up in a store, at arrival or in a refresh.
static LOOKUPS: AtomicU64 = AtomicU64::new(0);
/// Counts reported to the policy (each a change).
static REPORTS: AtomicU64 = AtomicU64::new(0);
/// Time spent looking up, in all.
static LOOKUP_NS: AtomicU64 = AtomicU64::new(0);
/// Time a store's lock was held for lookups: in all, and the longest once.
static LOCK_NS: AtomicU64 = AtomicU64::new(0);
static LOCK_NS_MAX: AtomicU64 = AtomicU64::new(0);

/// Forgets the boot's waiting prompts and counts.
pub(crate) fn reset() {
    *WAITING.lock().unwrap() = Waiting::default();
    WATCHED.store(false, Ordering::Release);
    REFRESHING.store(false, Ordering::Release);
    for count in [&REFRESHES, &LOOKUPS, &REPORTS, &LOOKUP_NS, &LOCK_NS, &LOCK_NS_MAX] {
        count.store(0, Ordering::Relaxed);
    }
}

/// The counters above, as `/metrics` lines.
pub fn metrics() -> String {
    let load = |c: &AtomicU64| c.load(Ordering::Relaxed);
    let secs = |c: &AtomicU64| load(c) as f64 / 1e9;
    format!(
        "# HELP pie_waiting_refreshes_total Changes of a store's cache that looked waiting prompts up again.\n\
         # TYPE pie_waiting_refreshes_total counter\n\
         pie_waiting_refreshes_total {}\n\
         # HELP pie_waiting_lookups_total Waiting prompts looked up in a kv store.\n\
         # TYPE pie_waiting_lookups_total counter\n\
         pie_waiting_lookups_total {}\n\
         # HELP pie_waiting_reports_total Changed cached-token counts of waiting processes told to the policy.\n\
         # TYPE pie_waiting_reports_total counter\n\
         pie_waiting_reports_total {}\n\
         # HELP pie_waiting_lookup_seconds_total Time spent looking waiting prompts up.\n\
         # TYPE pie_waiting_lookup_seconds_total counter\n\
         pie_waiting_lookup_seconds_total {}\n\
         # HELP pie_waiting_lock_seconds_total Time a kv store's lock was held for those lookups.\n\
         # TYPE pie_waiting_lock_seconds_total counter\n\
         pie_waiting_lock_seconds_total {}\n\
         # HELP pie_waiting_lock_seconds_max The longest one batch of lookups held a kv store's lock.\n\
         # TYPE pie_waiting_lock_seconds_max gauge\n\
         pie_waiting_lock_seconds_max {}\n",
        load(&REFRESHES),
        load(&LOOKUPS),
        load(&REPORTS),
        secs(&LOOKUP_NS),
        secs(&LOCK_NS),
        secs(&LOCK_NS_MAX),
    )
}

fn waiting() -> std::sync::MutexGuard<'static, Waiting> {
    WAITING.lock().expect("waiting prefixes lock poisoned")
}

/// Some prompt waits for admission: a store keeps the keys of what joins or
/// leaves its cache ([`KvStore::take_changes`]).
pub(crate) fn watched() -> bool {
    WATCHED.load(Ordering::Relaxed)
}

/// The prompt's tokens as the process will run them, where its props say.
pub(crate) fn prompt_tokens(props: &Props) -> Option<Vec<u32>> {
    if let Some(tokens) = props::prompt_tokens(props) {
        return Some(tokens);
    }
    let raw = props.get(props::RAW_PROMPT)?.as_str()?;
    let model = crate::model::model();
    Some(props::raw_prompt_tokens(
        model.instruct().prefix(),
        model.tokenize(raw),
    ))
}

/// The store's page keys for `tokens` as `ask` would key them: whole pages
/// from row 0, hashed outside its lock.
fn page_keys(stores: &Stores, ask: &ShareAs, tokens: &[u32]) -> Arc<[Hash256]> {
    let model = registry::with_kv_lock(&stores.kv, "waiting-prefix", |kv| kv.domain());
    let domain = crate::policy::share::domain_now(&model, ask);
    let slots = hash::chain_slots(&domain, None, 0, tokens);
    KvStore::fresh_page_keys(&slots, stores.kv_page_size).into()
}

/// How many leading pages of each of `keys` the store holds, on any tier,
/// under one hold of its lock.
fn runs(stores: &Stores, keys: &[(ProcessId, Arc<[Hash256]>)]) -> Vec<(ProcessId, usize)> {
    if keys.is_empty() {
        return Vec::new();
    }
    let looked = Instant::now();
    let runs = registry::with_kv_lock(&stores.kv, "waiting-prefix", |kv| {
        let held = Instant::now();
        let runs = keys
            .iter()
            .map(|(pid, keys)| (*pid, kv.cached_run_any_tier(keys).len()))
            .collect();
        let ns = held.elapsed().as_nanos() as u64;
        LOCK_NS.fetch_add(ns, Ordering::Relaxed);
        LOCK_NS_MAX.fetch_max(ns, Ordering::Relaxed);
        runs
    });
    LOOKUPS.fetch_add(keys.len() as u64, Ordering::Relaxed);
    LOOKUP_NS.fetch_add(looked.elapsed().as_nanos() as u64, Ordering::Relaxed);
    runs
}

fn report(reports: Vec<(ProcessId, u64)>) {
    for (pid, tokens) in reports {
        REPORTS.fetch_add(1, Ordering::Relaxed);
        crate::policy::report(crate::policy::Event::Waiting(pid, tokens));
    }
}

/// `pid`'s prompt, hashed for each of `stores`, is looked up and indexed.
/// The waiting lock is held from the lookup to the index, so no change a
/// refresh takes from a store falls between them.
fn watch(stores: &[Stores], pid: ProcessId, keys: Vec<(StoreId, Arc<[Hash256]>)>) {
    let mut reports = Vec::new();
    let mut waiting = waiting();
    let Some(watch) = waiting.watches.get_mut(&pid) else {
        return;
    };
    watch.page_size = stores
        .iter()
        .map(|stores| (store_id(stores), stores.kv_page_size))
        .collect();
    for (stores, (store, keys)) in stores.iter().zip(keys) {
        let run = runs(stores, &[(pid, Arc::clone(&keys))])[0].1;
        let look = Look { keys, run: 0 };
        waiting.index(pid, store, &look);
        waiting
            .watches
            .get_mut(&pid)
            .expect("checked above")
            .looks
            .insert(store, look);
        waiting.settle(store, vec![(pid, run)], &mut reports);
    }
    drop(waiting);
    report(reports);
}

/// `pid` arrived with `props`: if they give its prompt, it is looked up now
/// and kept up to date while it waits, in what `ask` would find (its
/// tenant's scope: it has asked nothing yet, and sees no other tenant's).
pub(crate) fn arrived(pid: ProcessId, props: &Props, ask: ShareAs) {
    if !props.contains_key(props::PROMPT_TOKENS) && !props.contains_key(props::RAW_PROMPT) {
        return;
    }
    if !crate::rt::has_runtime() {
        return;
    }
    // Registered now, so a start that comes before the hashing ends it.
    {
        let mut waiting = waiting();
        waiting.watches.insert(
            pid,
            Watch {
                looks: HashMap::new(),
                page_size: HashMap::new(),
                cached: None,
            },
        );
        WATCHED.store(true, Ordering::Relaxed);
    }
    let props = props.clone();
    crate::rt::spawn_blocking(move || {
        let stores = registry::all_for_model(MODEL);
        let tokens = match prompt_tokens(&props) {
            Some(tokens) if !stores.is_empty() => tokens,
            _ => {
                waiting().forget(pid);
                return;
            }
        };
        let keys = stores
            .iter()
            .map(|stores| (store_id(stores), page_keys(stores, &ask, &tokens)))
            .collect();
        watch(&stores, pid, keys);
        if !REFRESHING.swap(true, Ordering::AcqRel) {
            crate::rt::spawn(refresh());
        }
    });
}

/// `pid` started, or ended before it did: its own fires say what they find
/// cached from now on.
pub(crate) fn left(pid: ProcessId) {
    waiting().forget(pid);
}

/// Takes each store's changes and looks up again the prompts they can move;
/// how many lookups that made.
fn refresh_stores(stores: &[Stores]) -> usize {
    let mut looked = 0;
    let mut reports = Vec::new();
    for stores in stores {
        let store = store_id(stores);
        let (joined, left) =
            registry::with_kv_lock(&stores.kv, "waiting-prefix", |kv| kv.take_changes());
        if joined.is_empty() && left.is_empty() {
            continue;
        }
        let mut waiting = waiting();
        let keys: Vec<(ProcessId, Arc<[Hash256]>)> = waiting
            .woken(store, &joined, &left)
            .into_iter()
            .filter_map(|pid| {
                let look = waiting.watches.get(&pid)?.looks.get(&store)?;
                Some((pid, Arc::clone(&look.keys)))
            })
            .collect();
        if keys.is_empty() {
            continue;
        }
        REFRESHES.fetch_add(1, Ordering::Relaxed);
        looked += keys.len();
        let runs = runs(stores, &keys);
        waiting.settle(store, runs, &mut reports);
    }
    report(reports);
    looked
}

/// Each time a store's cache changes, the prompts the change can move are
/// looked up again. Ends when no store is registered; the next arrival that
/// finds one starts it again.
async fn refresh() {
    let mut changed: HashMap<StoreId, Arc<tokio::sync::Notify>> = HashMap::new();
    loop {
        let stores = registry::all_for_model(MODEL);
        if stores.is_empty() {
            REFRESHING.store(false, Ordering::Release);
            if registry::all_for_model(MODEL).is_empty() || REFRESHING.swap(true, Ordering::AcqRel)
            {
                return;
            }
            continue;
        }
        let notifies: Vec<Arc<tokio::sync::Notify>> = stores
            .iter()
            .map(|stores| {
                Arc::clone(changed.entry(store_id(stores)).or_insert_with(|| {
                    registry::with_kv_lock(&stores.kv, "waiting-prefix", |kv| kv.changed())
                }))
            })
            .collect();
        let mut told: Vec<_> = notifies
            .iter()
            .map(|notify| Box::pin(notify.notified()))
            .collect();
        for notified in &mut told {
            notified.as_mut().enable();
        }
        crate::rt::spawn_blocking(move || refresh_stores(&stores))
            .await
            .ok();
        futures::future::select_all(told).await;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `pid` waits on `tokens`, looked up in `stores` (what [`arrived`]
    /// does once it has hashed them).
    pub(crate) fn waits(stores: &[Stores], pid: ProcessId, ask: &ShareAs, tokens: &[u32]) {
        waiting().watches.insert(
            pid,
            Watch {
                looks: HashMap::new(),
                page_size: HashMap::new(),
                cached: None,
            },
        );
        WATCHED.store(true, Ordering::Relaxed);
        let keys = stores
            .iter()
            .map(|stores| (store_id(stores), page_keys(stores, ask, tokens)))
            .collect();
        watch(stores, pid, keys);
    }

    /// What `stores`' changes since the last call look up again: how many
    /// lookups, and what is now found cached for `pid`, while it waits.
    pub(crate) fn found(stores: &[Stores], pid: ProcessId) -> (usize, Option<u64>) {
        let looked = refresh_stores(stores);
        (looked, waiting().watches.get(&pid).and_then(|w| w.cached))
    }
}
