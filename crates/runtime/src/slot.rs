//! What a boot sets once and [`crate::bootstrap::reset`] clears, so the next
//! boot in the same process sets its own: the per-boot `OnceLock`s.

use std::sync::RwLock;

pub struct Slot<T>(RwLock<Option<T>>);

impl<T: Clone> Slot<T> {
    pub const fn new() -> Self {
        Self(RwLock::new(None))
    }

    /// Sets the value, or hands it back where this boot set one already.
    pub fn set(&self, value: T) -> Result<(), T> {
        let mut slot = self.0.write().unwrap();
        if slot.is_some() {
            return Err(value);
        }
        *slot = Some(value);
        Ok(())
    }

    pub fn get(&self) -> Option<T> {
        self.0.read().unwrap().clone()
    }

    pub fn get_or_init(&self, init: impl FnOnce() -> T) -> T {
        if let Some(value) = self.get() {
            return value;
        }
        self.0.write().unwrap().get_or_insert_with(init).clone()
    }

    pub fn clear(&self) {
        self.0.write().unwrap().take();
    }
}
