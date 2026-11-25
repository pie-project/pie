//! The per-kernel profile a backend keeps while profiling: each kernel's
//! device nanoseconds and fire count, by name. Each backend holds its own
//! table in a static.

use std::collections::BTreeMap;
use std::sync::Mutex;

pub struct KernelProfile(Mutex<BTreeMap<String, (u64, u64)>>);

impl KernelProfile {
    #[must_use]
    pub const fn new() -> KernelProfile {
        KernelProfile(Mutex::new(BTreeMap::new()))
    }

    /// Each kernel's (name, nanoseconds, fires), the costliest first.
    #[must_use]
    pub fn rows(&self) -> Vec<(String, u64, u64)> {
        let mut rows: Vec<(String, u64, u64)> = self
            .0
            .lock()
            .expect("kernel profile lock poisoned")
            .iter()
            .map(|(name, &(ns, n))| (name.clone(), ns, n))
            .collect();
        rows.sort_by_key(|row| std::cmp::Reverse((row.1, row.2)));
        rows
    }

    /// Credits `ns` device nanoseconds to `name`.
    pub fn time(&self, name: &str, ns: u64) {
        let mut table = self.0.lock().expect("kernel profile lock poisoned");
        table.entry(name.to_string()).or_insert((0, 0)).0 += ns;
    }

    /// Counts a fire of `name`.
    pub fn fire(&self, name: &str) {
        let mut table = self.0.lock().expect("kernel profile lock poisoned");
        table.entry(name.to_string()).or_insert((0, 0)).1 += 1;
    }

    pub fn clear(&self) {
        self.0.lock().expect("kernel profile lock poisoned").clear();
    }
}

impl Default for KernelProfile {
    fn default() -> KernelProfile {
        KernelProfile::new()
    }
}
