use super::ResidencyPlanner;
use super::inner::Inner;

impl ResidencyPlanner {
    pub(super) fn lock_inner(&self) -> parking_lot::MutexGuard<'_, Inner> {
        self.inner.lock()
    }
}
