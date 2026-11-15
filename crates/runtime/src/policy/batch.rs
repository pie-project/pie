//! The most tokens one batch of passes carries, as the worker plane's policy
//! installs it (`install-batch`). None leaves it to each engine: what one of
//! its fires carries.

use std::sync::atomic::{AtomicU64, Ordering};

/// A policy's `u32`, or [`Bound::UNSTATED`], past every one of them.
struct Bound(AtomicU64);

impl Bound {
    const UNSTATED: u64 = u64::MAX;

    const fn unstated() -> Self {
        Self(AtomicU64::new(Self::UNSTATED))
    }

    /// Whether it changed.
    fn install(&self, tokens: Option<u32>) -> bool {
        let next = tokens.map_or(Self::UNSTATED, u64::from);
        self.0.swap(next, Ordering::Relaxed) != next
    }

    fn tokens(&self) -> Option<usize> {
        match self.0.load(Ordering::Relaxed) {
            Self::UNSTATED => None,
            tokens => Some(tokens as usize),
        }
    }
}

static CURRENT: Bound = Bound::unstated();

pub(crate) fn install(tokens: Option<u32>) {
    if !CURRENT.install(tokens) {
        return;
    }
    match tokens {
        Some(tokens) => tracing::info!(tokens, "the worker policy bounds a batch of passes"),
        None => tracing::info!("no policy bounds a batch of passes: it is one fire's again"),
    }
}

/// What the policy stated, if one did.
pub fn tokens() -> Option<usize> {
    CURRENT.tokens()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bound_stands_until_it_is_withdrawn() {
        let bound = Bound::unstated();
        assert_eq!(bound.tokens(), None, "no policy stated one");
        assert!(bound.install(Some(8192)));
        assert_eq!(bound.tokens(), Some(8192));
        assert!(!bound.install(Some(8192)), "the same again changes nothing");
        assert!(bound.install(Some(u32::MAX)));
        assert_eq!(bound.tokens(), Some(u32::MAX as usize), "every u32 is one");
        assert!(bound.install(Some(0)));
        assert_eq!(bound.tokens(), Some(0));
        assert!(bound.install(None));
        assert_eq!(bound.tokens(), None, "withdrawn");
        assert!(!bound.install(None));
    }
}
