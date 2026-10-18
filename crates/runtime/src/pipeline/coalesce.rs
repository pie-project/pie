//! Prefills of one prefix at once are coalesced: a pass whose first pages
//! another pass is computing right now waits for them to seal and maps them
//! in, instead of computing them a second time.
//!
//! Programs that fork (a search's siblings, a batch of agents on one system
//! prompt) launch passes over a shared prefix together. The cache holds what
//! has sealed, and nothing has yet, so each would prefill all of it. Here a
//! pass claims the pages it is about to compute, by key, until a deadline; a
//! later pass whose uncached pages begin with pages another has claimed waits
//! until the last of those is cached, or the deadline passes, and then runs
//! from what is there. A claimant that fails or stalls costs the waiter that
//! wait and no more: it computes the pages itself.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::store::kv::hash::Hash256;
use crate::store::kv::page_table::WorkingSetId;
use crate::store::registry::{Stores, with_kv_lock};

/// How often a waiter looks for the pages it waits on.
const POLL: Duration = Duration::from_millis(5);
/// A claim's floor and its ceiling: what a claimant may take to seal them.
const LEAST: Duration = Duration::from_secs(1);
const MOST: Duration = Duration::from_secs(10);
/// Tokens a second a claim is expected to prefill at, slow on purpose:
/// passes under load share the device.
const TOKENS_PER_SECOND: u64 = 2000;

#[derive(Clone, Copy)]
struct Claim<By = WorkingSetId> {
    by: By,
    until: Instant,
}

static CLAIMS: LazyLock<Mutex<HashMap<Hash256, Claim>>> = LazyLock::new(Default::default);

/// How many of `mine`, from the first, others have claimed and not let
/// lapse by `now`, and when the last of those claims lapses.
fn claimed_run<By: PartialEq>(
    claims: &HashMap<Hash256, Claim<By>>,
    mine: &[Hash256],
    by: By,
    now: Instant,
) -> (usize, Option<Instant>) {
    let mut until = None;
    let run = mine
        .iter()
        .take_while(|key| match claims.get(*key) {
            Some(claim) if claim.by != by && claim.until > now => {
                until = until.max(Some(claim.until));
                true
            }
            _ => false,
        })
        .count();
    (run, until)
}

fn deadline(pages: usize, page_size: u32, now: Instant) -> Instant {
    let tokens = pages as u64 * u64::from(page_size);
    now + (LEAST + Duration::from_millis(tokens * 1000 / TOKENS_PER_SECOND)).min(MOST)
}

/// Pages from token 0 cached in any tier, at least the `held` ones.
fn cached(stores: &Stores, keys: &[Hash256], held_pages: usize) -> usize {
    with_kv_lock(&stores.kv, "coalesce", |kv| {
        kv.cached_run_any_tier(keys).len()
    })
    .max(held_pages)
}

/// Before a keyed pass over `slots` from row `held` of working set `ws`:
/// waits out the pages another pass is computing that this one would
/// compute first, then claims the ones it will compute.
pub(crate) async fn before_pass(stores: &Stores, ws: WorkingSetId, held: u64, slots: &[Hash256]) {
    let page_size = stores.kv_page_size;
    let page = u64::from(page_size.max(1));
    let Some(keys) = with_kv_lock(&stores.kv, "coalesce", |kv| {
        kv.page_keys(ws, held, slots, page_size).ok().flatten()
    }) else {
        return;
    };
    let held_pages = (held / page) as usize;
    let mut have = cached(stores, &keys, held_pages);
    if have >= keys.len() {
        return;
    }
    let now = Instant::now();
    let (run, until) = {
        let mut claims = CLAIMS.lock().unwrap();
        claims.retain(|_, claim| claim.until > now);
        claimed_run(&claims, &keys[have..], ws, now)
    };
    if let Some(until) = until {
        let want = have + run;
        let started = Instant::now();
        while Instant::now() < until {
            have = cached(stores, &keys[..want], held_pages);
            if have >= want {
                break;
            }
            tokio::time::sleep(POLL).await;
        }
        have = cached(stores, &keys, held_pages);
        tracing::debug!(
            target: "pie::coalesce",
            waited_ms = started.elapsed().as_millis() as u64,
            pages = run,
            found = have >= want,
            "a pass waited on pages another pass was computing"
        );
    }
    if have < keys.len() {
        let now = Instant::now();
        let claim = Claim {
            by: ws,
            until: deadline(keys.len() - have, page_size, now),
        };
        let mut claims = CLAIMS.lock().unwrap();
        for key in &keys[have..] {
            claims.entry(*key).or_insert(claim);
        }
    }
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
        let now = Instant::now();
        let later = now + Duration::from_secs(2);
        let claims: HashMap<Hash256, Claim<u32>> = [
            (
                key(1),
                Claim {
                    by: a,
                    until: later,
                },
            ),
            (
                key(2),
                Claim {
                    by: a,
                    until: now + Duration::from_secs(3),
                },
            ),
            (
                key(4),
                Claim {
                    by: a,
                    until: later,
                },
            ),
        ]
        .into();
        // b's first two pages are a's to compute; its third is no one's,
        // so the run stops there even though a claimed the fourth.
        let (run, until) = claimed_run(&claims, &[key(1), key(2), key(3), key(4)], b, now);
        assert_eq!((run, until), (2, Some(now + Duration::from_secs(3))));
        // A pass does not wait on its own claims, nor on lapsed ones.
        assert_eq!(claimed_run(&claims, &[key(1)], a, now), (0, None));
        assert_eq!(claimed_run(&claims, &[key(1)], b, later), (0, None));
        // Nothing claimed first: nothing to wait on.
        assert_eq!(claimed_run(&claims, &[key(3), key(4)], b, now), (0, None));
    }

    #[test]
    fn a_claim_lasts_for_what_it_prefills_within_bounds() {
        let now = Instant::now();
        assert_eq!(deadline(0, 16, now), now + LEAST);
        // 250 pages of 16 tokens at 2000 a second: two seconds past the floor.
        assert_eq!(deadline(250, 16, now), now + LEAST + Duration::from_secs(2));
        assert_eq!(deadline(100_000, 16, now), now + MOST);
    }
}
