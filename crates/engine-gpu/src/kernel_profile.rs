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

/// Diagnostic: where `PIE_VK_FAMILY_LOG` names a file, the arm walk's
/// compositions and the family each landed in, a line each, appended.
pub fn family_log(line: &str) {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    let Some(path) = PATH
        .get_or_init(|| std::env::var_os("PIE_VK_FAMILY_LOG").map(std::path::PathBuf::from))
        .as_deref()
    else {
        return;
    };
    use std::io::Write as _;
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

/// The arm walk's host time by phase (prepare, enqueue, settle, read back,
/// the fill walk, the backend's fill, its plan), in ns: a diagnostic the
/// load logs at ready under `PIE_VK_KERNEL_PROFILE`.
pub static WALK_PHASES: [std::sync::atomic::AtomicU64; 10] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 10];

/// Whether the walk's phases are timed: under `PIE_VK_KERNEL_PROFILE`.
pub fn phases_timed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_VK_KERNEL_PROFILE").is_some())
}

/// Credits `since`'s elapsed time to phase `phase`, where phases are timed.
pub fn phase_if(phase: usize, since: std::time::Instant) {
    if phases_timed() {
        self::phase(phase, since);
    }
}

/// Credits `since`'s elapsed time to phase `phase`.
pub fn phase(phase: usize, since: std::time::Instant) {
    WALK_PHASES[phase].fetch_add(
        u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
}
