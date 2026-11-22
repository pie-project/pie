//! Prefills of one prefix at once are coalesced: a pass whose first pages
//! another pass is computing right now waits for them to seal and maps them
//! in, instead of computing them a second time.
//!
//! Programs that fork (a search's siblings, a batch of agents on one system
//! prompt) launch passes over a shared prefix together. The cache holds what
//! has sealed, and nothing has yet, so each would prefill all of it. Here a
//! pass claims the pages it is about to compute, by key, for as long as the
//! pass runs ([`ClaimGuard`], released when its last piece settles, after
//! what it landed and froze is sealed, or when it ends early). A later pass whose uncached pages begin with pages another
//! has claimed waits on events only: pages joining the cache (the kv store's
//! `cached`), and claims released ([`RELEASED`]). Once those pages are all
//! cached it maps them in; once the claims on them are released with some
//! still uncached (the claimant ended without sealing them), it computes them
//! itself, on that event, not after a deadline.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::store::kv::hash::Hash256;
use crate::store::kv::page_table::WorkingSetId;
use crate::store::registry::{Stores, with_kv_lock};

#[derive(Clone, Copy)]
struct Claim<By = WorkingSetId> {
    by: By,
}

static CLAIMS: LazyLock<Mutex<HashMap<Hash256, Claim>>> = LazyLock::new(Default::default);

/// Told whenever a pass's claims are released.
static RELEASED: tokio::sync::Notify = tokio::sync::Notify::const_new();

/// How many of `mine`, from the first, others have claimed.
fn claimed_run<By: PartialEq>(
    claims: &HashMap<Hash256, Claim<By>>,
    mine: &[Hash256],
    by: By,
) -> usize {
    mine.iter()
        .take_while(|key| claims.get(*key).is_some_and(|claim| claim.by != by))
        .count()
}

/// `None` while others have claimed the first of `rest`; else claims those
/// of `rest` no one has, and answers them.
fn claim<By: PartialEq + Copy>(
    claims: &mut HashMap<Hash256, Claim<By>>,
    rest: &[Hash256],
    by: By,
) -> Option<Vec<Hash256>> {
    if claimed_run(claims, rest, by) > 0 {
        return None;
    }
    Some(
        rest.iter()
            .copied()
            .filter(|key| {
                let free = !claims.contains_key(key);
                if free {
                    claims.insert(*key, Claim { by });
                }
                free
            })
            .collect(),
    )
}

/// Pages from token 0 cached in any tier, at least the `held` ones.
fn cached(stores: &Stores, keys: &[Hash256], held_pages: usize) -> usize {
    with_kv_lock(&stores.kv, "coalesce", |kv| {
        kv.cached_run_any_tier(keys).len()
    })
    .max(held_pages)
}

/// The claims a pass holds while it runs: released (and its waiters told)
/// when dropped, once its last piece settles or it ends early.
#[must_use = "the claims stand while the pass runs"]
pub(crate) struct ClaimGuard {
    by: WorkingSetId,
    keys: Vec<Hash256>,
}

impl Drop for ClaimGuard {
    fn drop(&mut self) {
        {
            let mut claims = CLAIMS.lock().unwrap();
            for key in &self.keys {
                if claims.get(key).is_some_and(|claim| claim.by == self.by) {
                    claims.remove(key);
                }
            }
        }
        RELEASED.notify_waiters();
    }
}

/// Before a keyed pass over `slots` from row `held` of working set `ws`:
/// waits out the pages another pass is computing that this one would
/// compute first, then claims the ones it will compute, for as long as the
/// guard it answers stands. `waits` is told once, as it first waits, and
/// what it answers is held until the wait ends: its process submits
/// nothing until those pages seal, so no frame may wait on its lanes
/// meanwhile (the frame would hold the claimant's fire whose pages it
/// waits on).
pub(crate) async fn before_pass<Told>(
    stores: &Stores,
    ws: WorkingSetId,
    held: u64,
    slots: &[Hash256],
    waits: impl FnOnce() -> Told,
) -> Option<ClaimGuard> {
    let mut waits = Some(waits);
    let mut _told: Option<Told> = None;
    let page_size = stores.kv_page_size;
    let page = u64::from(page_size.max(1));
    let keys = if held == 0 {
        Some(crate::store::kv::KvStore::fresh_page_keys(slots, page_size))
    } else {
        with_kv_lock(&stores.kv, "coalesce", |kv| {
            kv.page_keys(ws, held, slots, page_size).ok().flatten()
        })
    }?;
    let held_pages = (held / page) as usize;
    let cached_event = with_kv_lock(&stores.kv, "coalesce", |kv| kv.cached());
    let started = crate::rt::Instant::now();
    let mut waited = false;
    let mine = loop {
        // Armed before the look: a page cached or a claim released between
        // the look and the wait is not lost.
        let joined = cached_event.notified();
        let released = RELEASED.notified();
        let mut joined = std::pin::pin!(joined);
        let mut released = std::pin::pin!(released);
        joined.as_mut().enable();
        released.as_mut().enable();
        let have = cached(stores, &keys, held_pages);
        if have >= keys.len() {
            return None;
        }
        // Looked at and claimed under one lock: passes that arrive together
        // see each other's claims, and one computes what the others wait on.
        if let Some(mine) = claim(&mut CLAIMS.lock().unwrap(), &keys[have..], ws) {
            break mine;
        }
        waited = true;
        if let Some(waits) = waits.take() {
            _told = Some(waits());
        }
        tokio::select! {
            () = &mut joined => {}
            () = &mut released => {}
        }
    };
    if waited {
        tracing::debug!(
            target: "pie::coalesce",
            waited_ms = started.elapsed().as_millis() as u64,
            "a pass waited on pages another pass was computing"
        );
    }
    (!mine.is_empty()).then(|| ClaimGuard { by: ws, keys: mine })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u8) -> Hash256 {
        [n; 32]
    }

    #[test]
    fn a_pass_waits_on_the_run_others_claimed_from_its_first_page() {
        let (a, b) = (1u32, 2u32);
        let claims: HashMap<Hash256, Claim<u32>> = [
            (key(1), Claim { by: a }),
            (key(2), Claim { by: a }),
            (key(4), Claim { by: a }),
        ]
        .into();
        // b's first two pages are a's to compute; its third is no one's,
        // so the run stops there even though a claimed the fourth.
        assert_eq!(
            claimed_run(&claims, &[key(1), key(2), key(3), key(4)], b),
            2
        );
        // A pass does not wait on its own claims.
        assert_eq!(claimed_run(&claims, &[key(1)], a), 0);
        // Nothing claimed first: nothing to wait on.
        assert_eq!(claimed_run(&claims, &[key(3), key(4)], b), 0);
    }

    /// A claim stands until its guard drops, and the drop tells waiters.
    #[tokio::test(flavor = "current_thread")]
    async fn a_released_claim_wakes_its_waiters() {
        let model = crate::store::registry::register_model(32, &[1], &[1]);
        let ws = crate::store::registry::get(model, 0)
            .kv
            .lock()
            .create_working_set();
        let keys = vec![[0xC5; 32], [0xC6; 32]];
        {
            let mut claims = CLAIMS.lock().unwrap();
            for key in &keys {
                claims.insert(*key, Claim { by: ws });
            }
        }
        let guard = ClaimGuard {
            by: ws,
            keys: keys.clone(),
        };
        let waiting = tokio::spawn(async {
            RELEASED.notified().await;
        });
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(!waiting.is_finished());
        drop(guard);
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(waiting.is_finished(), "the release told its waiter");
        let claims = CLAIMS.lock().unwrap();
        assert!(keys.iter().all(|key| !claims.contains_key(key)));
    }
}
