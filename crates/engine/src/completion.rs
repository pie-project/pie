//! When a copy the runtime asked for has landed.
//!
//! [`crate::Engine::copy_kv`], [`crate::Engine::copy_state`] and
//! [`crate::Engine::copy_slots`] answer a [`CopyCompletion`]: it resolves
//! once the copy's bytes are in place on its destination and the device
//! reads nothing more of its source. The engine holds the other end, a
//! [`CopyLanding`], and resolves it from wherever it learns the copy landed
//! (a stream event, a command buffer's handler, a transfer future).
//!
//! The runtime hands a slot a copy names to a new owner only once that
//! copy's completion has resolved, so an engine must not resolve it early:
//! a slot reused under a copy still in flight is silently overwritten.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::error::{Error, Result};

type LandedHook = Box<dyn FnOnce(&std::result::Result<(), String>) + Send>;

#[derive(Default)]
struct State {
    /// `None` while the copy is in flight; its outcome once it landed.
    outcome: Option<std::result::Result<(), String>>,
    wakers: Vec<Waker>,
    hooks: Vec<LandedHook>,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
}

impl Shared {
    fn resolve(&self, outcome: std::result::Result<(), String>) {
        let (wakers, hooks) = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.outcome.is_some() {
                return;
            }
            state.outcome = Some(outcome.clone());
            (
                std::mem::take(&mut state.wakers),
                std::mem::take(&mut state.hooks),
            )
        };
        for hook in hooks {
            hook(&outcome);
        }
        for waker in wakers {
            waker.wake();
        }
    }

    fn outcome(&self) -> Option<std::result::Result<(), String>> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .outcome
            .clone()
    }
}

/// The runtime's end of a copy: resolves once the copy has landed. Clones
/// wait on the same copy. Dropping it changes nothing on the device and
/// resolves nothing: the copy lands as it would have.
#[derive(Clone)]
pub struct CopyCompletion {
    shared: Arc<Shared>,
}

/// The engine's end of a copy: [`CopyLanding::landed`] once its bytes are
/// in place, [`CopyLanding::failed`] once it can no longer touch either
/// side but did not complete. Dropped unresolved, it resolves as failed:
/// an engine drops it only where the copy can no longer touch its slots
/// (the device is gone, or the engine waited the copy out).
pub struct CopyLanding {
    shared: Arc<Shared>,
}

/// A copy in flight: the engine's end and the runtime's. The engine keeps
/// the [`CopyLanding`] and resolves it when the copy lands; it answers the
/// [`CopyCompletion`].
#[must_use]
pub fn copy_in_flight() -> (CopyLanding, CopyCompletion) {
    let shared = Arc::new(Shared::default());
    (
        CopyLanding {
            shared: Arc::clone(&shared),
        },
        CopyCompletion { shared },
    )
}

/// One copy made of `parts` (a group's ranks, each copying its band):
/// lands once every part has, failed if any part failed. With no parts it
/// is an engine's own placeholder, so it asks for at least one.
///
/// # Panics
///
/// Where `parts` is empty.
#[must_use]
pub fn all(parts: Vec<CopyCompletion>) -> CopyCompletion {
    assert!(!parts.is_empty(), "a joined copy has at least one part");
    if parts.len() == 1 {
        return parts.into_iter().next().unwrap_or_else(|| unreachable!());
    }
    let (landing, completion) = copy_in_flight();
    struct Join {
        landing: Option<CopyLanding>,
        left: usize,
        failed: Option<String>,
    }
    let join = Arc::new(Mutex::new(Join {
        landing: Some(landing),
        left: parts.len(),
        failed: None,
    }));
    for part in parts {
        let join = Arc::clone(&join);
        part.on_landed(move |outcome| {
            let done = {
                let mut join = join.lock().unwrap_or_else(|e| e.into_inner());
                if let Err(why) = outcome {
                    join.failed.get_or_insert_with(|| why.clone());
                }
                join.left -= 1;
                if join.left == 0 {
                    join.landing.take().map(|landing| (landing, join.failed.take()))
                } else {
                    None
                }
            };
            match done {
                Some((landing, None)) => landing.landed(),
                Some((landing, Some(why))) => landing.failed(why),
                None => {}
            }
        });
    }
    completion
}

/// STAGING ONLY. A completion already resolved, for an engine that has not
/// yet wired its copies' real landing: it answers this after its current
/// synchronization, which may resolve before the device has finished the
/// copy (CUDA / ROCm's swap stream, Metal's grafted command buffer). The
/// layer must not land while anything calls this: each engine replaces it
/// with a [`CopyLanding`] it resolves when the copy has landed. Nothing
/// else may construct a resolved completion for a copy.
#[must_use]
pub fn copy_landed_placeholder() -> CopyCompletion {
    let (landing, completion) = copy_in_flight();
    landing.landed();
    completion
}

impl CopyLanding {
    /// The copy's bytes are in place on its destination, and the device
    /// reads nothing more of its source.
    pub fn landed(self) {
        self.shared.resolve(Ok(()));
    }

    /// The copy failed, and nothing of it is still in flight.
    pub fn failed(self, why: impl std::fmt::Display) {
        self.shared.resolve(Err(why.to_string()));
    }
}

impl Drop for CopyLanding {
    fn drop(&mut self) {
        self.shared
            .resolve(Err("the engine dropped a copy before it landed".to_string()));
    }
}

impl std::fmt::Debug for CopyCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CopyCompletion")
            .field("outcome", &self.shared.outcome())
            .finish()
    }
}

impl CopyCompletion {
    /// `None` while the copy is in flight; its outcome once it landed.
    #[must_use]
    pub fn check(&self) -> Option<Result<()>> {
        self.shared
            .outcome()
            .map(|outcome| outcome.map_err(Error::Device))
    }

    /// Whether the copy has resolved, landed or failed: either way nothing
    /// of it is in flight.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.shared.outcome().is_some()
    }

    /// Runs `hook` with the copy's outcome once it resolves: now, on the
    /// caller's thread, if it has; else on the thread that resolves it,
    /// which may be the engine's completion thread, so the hook must be
    /// short and take no lock the engine's resolving path could hold.
    pub fn on_landed(
        &self,
        hook: impl FnOnce(&std::result::Result<(), String>) + Send + 'static,
    ) {
        let outcome = {
            let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            match &state.outcome {
                Some(outcome) => outcome.clone(),
                None => {
                    state.hooks.push(Box::new(hook));
                    return;
                }
            }
        };
        hook(&outcome);
    }
}

impl Future for CopyCompletion {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        match &state.outcome {
            Some(outcome) => Poll::Ready(outcome.clone().map_err(Error::Device)),
            None => {
                if !state.wakers.iter().any(|w| w.will_wake(cx.waker())) {
                    state.wakers.push(cx.waker().clone());
                }
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_copy_resolves_only_when_its_engine_lands_it() {
        let (landing, completion) = copy_in_flight();
        let seen = Arc::new(Mutex::new(None));
        let hook_seen = Arc::clone(&seen);
        completion.on_landed(move |outcome| *hook_seen.lock().unwrap() = Some(outcome.is_ok()));
        assert!(completion.check().is_none());
        drop(completion.clone());
        assert!(seen.lock().unwrap().is_none());
        landing.landed();
        assert!(matches!(completion.check(), Some(Ok(()))));
        assert_eq!(*seen.lock().unwrap(), Some(true));
    }

    #[test]
    fn a_joined_copy_lands_with_its_last_part() {
        let (first, a) = copy_in_flight();
        let (second, b) = copy_in_flight();
        let joined = all(vec![a, b]);
        first.landed();
        assert!(joined.check().is_none());
        second.failed("rank 1 faulted");
        assert!(matches!(joined.check(), Some(Err(_))));
    }

    #[test]
    fn a_dropped_landing_fails_its_copy() {
        let (landing, completion) = copy_in_flight();
        drop(landing);
        assert!(matches!(completion.check(), Some(Err(_))));
    }
}
