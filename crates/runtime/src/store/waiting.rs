//! The cached prefix of a process waiting for admission: its prompt's tokens
//! (`prompt-tokens`, or a completion's `raw-prompt` tokenized as the compat
//! inferlet will) looked up in the kv stores when it arrives, and again each
//! time an entry joins or leaves a store's cache, until it starts. It reads
//! the stores and holds no page: what to keep for it is the policy's choice.
//! The policy hears it as [`Event::Waiting`](crate::policy::Event::Waiting).

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Once};

use client_api::props::{self, Props};

use crate::inferlet::process::ProcessId;
use crate::store::kv::{KvStore, hash};
use crate::store::registry::{self, Stores};

/// The model whose stores a waiting prompt is looked up in.
const MODEL: usize = 0;

/// One waiting process: its page keys in each engine's store (`None` until
/// they are hashed), and the tokens last reported.
struct Watch {
    keys: Option<Vec<Vec<hash::Hash256>>>,
    cached: Option<u64>,
}

static WAITING: LazyLock<Mutex<HashMap<ProcessId, Watch>>> = LazyLock::new(Default::default);
static REFRESH: Once = Once::new();

fn waiting() -> std::sync::MutexGuard<'static, HashMap<ProcessId, Watch>> {
    WAITING.lock().expect("waiting prefixes lock poisoned")
}

/// The prompt's tokens as the process will run them, where its props say.
pub(crate) fn prompt_tokens(props: &Props) -> Option<Vec<u32>> {
    if let Some(tokens) = props::prompt_tokens(props) {
        return Some(tokens);
    }
    let raw = props.get(props::RAW_PROMPT)?.as_str()?;
    let model = crate::model::model();
    let body = model.tokenize(raw);
    // As the compat inferlet builds a raw prompt: the model's prefix first,
    // unless the text already spells it.
    let prefix = model.instruct().prefix();
    Some(if body.starts_with(&prefix) {
        body
    } else {
        [prefix, body].concat()
    })
}

/// Each engine's page keys for `tokens`: whole pages from row 0, hashed
/// outside the store's lock.
fn page_keys(stores: &[Stores], tokens: &[u32]) -> Vec<Vec<hash::Hash256>> {
    stores
        .iter()
        .map(|stores| {
            let domain = registry::with_kv_lock(&stores.kv, "waiting-prefix", |kv| kv.domain());
            let slots = hash::chain_slots(&domain, None, 0, tokens);
            KvStore::fresh_page_keys(&slots, stores.kv_page_size)
        })
        .collect()
}

/// The tokens of its leading pages some engine's store holds, on any tier:
/// the most any one holds.
fn cached(stores: &[Stores], keys: &[Vec<hash::Hash256>]) -> u64 {
    stores
        .iter()
        .zip(keys)
        .map(|(stores, keys)| {
            let pages = registry::with_kv_lock(&stores.kv, "waiting-prefix", |kv| {
                kv.cached_run_any_tier(keys).len()
            });
            pages as u64 * u64::from(stores.kv_page_size)
        })
        .max()
        .unwrap_or(0)
}

/// `pid` arrived with `props`: if they give its prompt, it is looked up now
/// and kept up to date while it waits.
pub(crate) fn arrived(pid: ProcessId, props: &Props) {
    if !props.contains_key(props::PROMPT_TOKENS) && !props.contains_key(props::RAW_PROMPT) {
        return;
    }
    if !crate::rt::has_runtime() {
        return;
    }
    // Registered now, so a start that comes before the hashing ends it.
    waiting().insert(
        pid,
        Watch {
            keys: None,
            cached: None,
        },
    );
    REFRESH.call_once(|| {
        crate::rt::spawn(refresh());
    });
    let props = props.clone();
    crate::rt::spawn(async move {
        let hashed = crate::rt::spawn_blocking(move || {
            let tokens = prompt_tokens(&props)?;
            let stores = registry::all_for_model(MODEL);
            Some(page_keys(&stores, &tokens))
        })
        .await
        .ok()
        .flatten();
        let mut waiting = waiting();
        match hashed {
            Some(keys) => {
                if let Some(watch) = waiting.get_mut(&pid) {
                    watch.keys = Some(keys);
                }
            }
            None => {
                waiting.remove(&pid);
                return;
            }
        }
        drop(waiting);
        update(&registry::all_for_model(MODEL), &[pid]);
    });
}

/// `pid` started, or ended before it did: its own fires say what they find
/// cached from now on.
pub(crate) fn left(pid: ProcessId) {
    waiting().remove(&pid);
}

/// Looks `pids` up again in `stores`, telling the policy each count that
/// changed.
fn update(stores: &[Stores], pids: &[ProcessId]) {
    let keys: Vec<(ProcessId, Arc<Vec<Vec<hash::Hash256>>>)> = {
        let waiting = waiting();
        pids.iter()
            .filter_map(|pid| {
                let keys = waiting.get(pid)?.keys.clone()?;
                Some((*pid, Arc::new(keys)))
            })
            .collect()
    };
    for (pid, keys) in keys {
        let tokens = cached(stores, &keys);
        let changed = waiting().get_mut(&pid).is_some_and(|watch| {
            let changed = watch.cached != Some(tokens);
            watch.cached = Some(tokens);
            changed
        });
        if changed {
            crate::policy::report(crate::policy::Event::Waiting(pid, tokens));
        }
    }
}

/// Each time an entry joins or leaves a store's cache, every waiting prompt
/// is looked up again.
async fn refresh() {
    loop {
        let stores = registry::all_for_model(MODEL);
        let changed: Vec<Arc<tokio::sync::Notify>> = stores
            .iter()
            .map(|stores| registry::with_kv_lock(&stores.kv, "waiting-prefix", |kv| kv.changed()))
            .collect();
        let mut told: Vec<_> = changed
            .iter()
            .map(|notify| Box::pin(notify.notified()))
            .collect();
        for notified in &mut told {
            notified.as_mut().enable();
        }
        let pids: Vec<ProcessId> = waiting().keys().copied().collect();
        if !pids.is_empty() {
            crate::rt::spawn_blocking(move || update(&stores, &pids))
                .await
                .ok();
        }
        if told.is_empty() {
            return;
        }
        futures::future::select_all(told).await;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn prompt_tokens_are_the_ones_stated() {
        let mut props = Props::new();
        props.insert(
            props::PROMPT_TOKENS.to_string(),
            serde_json::json!([5, 6, 7]),
        );
        assert_eq!(prompt_tokens(&props), Some(vec![5, 6, 7]));
        props.insert(props::PROMPT_TOKENS.to_string(), serde_json::json!(["x"]));
        assert_eq!(props::prompt_tokens(&props), None);
    }

    /// `pid` waits on `tokens`, looked up in `stores` (what [`arrived`]
    /// registers once it has hashed them).
    pub(crate) fn waits(stores: &[Stores], pid: ProcessId, tokens: &[u32]) {
        waiting().insert(
            pid,
            Watch {
                keys: Some(page_keys(stores, tokens)),
                cached: None,
            },
        );
        update(stores, &[pid]);
    }

    /// What was last found cached for `pid`, while it waits.
    pub(crate) fn found(stores: &[Stores], pid: ProcessId) -> Option<u64> {
        update(stores, &[pid]);
        waiting().get(&pid)?.cached
    }
}
