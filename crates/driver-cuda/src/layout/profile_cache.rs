//! On-disk cache of measured planner shapes.
//!
//! The calibrator times candidate shapes and the planner later reuses the
//! winner for the same [`ProfileKey`]. Corrupt or incompatible files degrade
//! to [`Lookup::Miss`] or [`Lookup::Unusable`]; C++ JSON quirks are preserved.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Map, Value};

use super::json::dump_pretty;
use super::profile_key::{ProfileKey, ProfileShape, SCHEMA_VERSION, StoredField};

/// Allowed fractional drift from the measured planner budget.
pub const BUDGET_TOLERANCE: f64 = 0.05;

/// One measured point on the shape ladder, kept beside the selection so an
/// entry can be audited without re-running the calibration.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShapeSample {
    /// Token capacity swept at this point.
    pub max_forward_tokens: i32,
    /// Request capacity swept at this point.
    pub max_forward_requests: i32,
    /// Tokens per request in the synthetic batch.
    pub tokens_per_request: i32,
    /// Mean measured step time, in ms.
    pub step_ms: f64,
    /// Step-time standard deviation, in ms.
    pub step_ms_stddev: f64,
    /// Derived throughput, in tokens/s.
    pub tokens_per_s: f64,
}

/// Result of reading a cached shape.
#[derive(Debug, Clone, PartialEq)]
pub enum Lookup {
    /// An entry for this key, readable and at the expected schema version.
    Hit(ProfileShape),
    /// No entry for this key, including the absent-file first-boot case.
    Miss,
    /// A cache exists but cannot be trusted; proceed as for [`Lookup::Miss`].
    Unusable(String),
}

impl Lookup {
    /// The shape, if this is a hit.
    #[must_use]
    pub fn shape(&self) -> Option<&ProfileShape> {
        match self {
            Self::Hit(s) => Some(s),
            _ => None,
        }
    }

    /// What to tell the operator, if anything.
    #[must_use]
    pub fn complaint(&self) -> Option<&str> {
        match self {
            Self::Unusable(m) => Some(m.as_str()),
            _ => None,
        }
    }
}

impl ProfileShape {
    /// Whether the recorded budget is close enough to `budget`.
    ///
    /// A zero recorded budget means "not recorded" and always applies.
    #[must_use]
    pub fn applies_at(&self, budget: u64) -> bool {
        if self.budget_bytes == 0 {
            return true;
        }
        self.drift_from(budget) <= BUDGET_TOLERANCE
    }

    /// Fractional distance from the recorded budget, divided by the measured budget.
    #[must_use]
    pub fn drift_from(&self, budget: u64) -> f64 {
        if self.budget_bytes == 0 {
            return 0.0;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "byte counts below 2^53 are exact, and the ratio is what matters above it"
        )]
        let (now, then) = (budget as f64, self.budget_bytes as f64);
        (now - then).abs() / then
    }
}

/// Derive the cache path from the configured directory, XDG, or `$HOME`.
///
/// `env` is injected so tests need not mutate process-global environment.
pub fn cache_path(configured_dir: &str, env: impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    if !configured_dir.is_empty() {
        return Some(Path::new(configured_dir).join("cuda_memory_profiles.json"));
    }
    // An empty string counts as unset, matching the C++'s `xdg[0] != '\0'`.
    if let Some(xdg) = env("XDG_CACHE_HOME").filter(|s| !s.is_empty()) {
        return Some(
            Path::new(&xdg)
                .join("pie")
                .join("cuda_memory_profiles.json"),
        );
    }
    if let Some(home) = env("HOME").filter(|s| !s.is_empty()) {
        return Some(
            Path::new(&home)
                .join(".cache")
                .join("pie")
                .join("cuda_memory_profiles.json"),
        );
    }
    None
}

/// Planner budget for this boot, published for the calibrator.
///
/// Atomic because the planner and calibrator may be reached from different threads.
static PLANNER_BUDGET_BYTES: AtomicU64 = AtomicU64::new(0);

/// Publish the budget the planner settled on.
pub fn set_planner_budget_bytes(budget: u64) {
    PLANNER_BUDGET_BYTES.store(budget, Ordering::Relaxed);
}

/// Read back what [`set_planner_budget_bytes`] published; `0` until then.
#[must_use]
pub fn planner_budget_bytes() -> u64 {
    PLANNER_BUDGET_BYTES.load(Ordering::Relaxed)
}

/// Failure modes of [`ProfileCache::store`].
#[derive(Debug)]
pub enum StoreError {
    /// Neither a configured cache directory nor `$XDG_CACHE_HOME` nor `$HOME`.
    NoCacheDir,
    /// A filesystem operation failed; `what` names the step.
    Io {
        /// The step that failed.
        what: &'static str,
        /// The path it was operating on.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCacheDir => f.write_str(
                "no cache directory: neither the engine's [cache] dir, \
                 $XDG_CACHE_HOME, nor $HOME is set",
            ),
            Self::Io { what, path, source } => {
                write!(f, "cannot {what} {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::NoCacheDir => None,
        }
    }
}

/// Exclusive advisory lock held on a sibling `.lock` file.
struct CacheLock {
    /// Dropping the file releases the lock.
    _file: std::fs::File,
}

impl CacheLock {
    /// Open the sibling lock file and take the lock.
    fn acquire(cache_path: &Path) -> Result<Self, StoreError> {
        let mut name = cache_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        name.push_str(".lock");
        let lock_path = cache_path.parent().unwrap_or(Path::new(".")).join(name);

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|source| StoreError::Io {
                what: "open",
                path: lock_path.clone(),
                source,
            })?;
        file.lock().map_err(|source| StoreError::Io {
            what: "lock",
            path: lock_path,
            source,
        })?;
        Ok(Self { _file: file })
    }
}

/// The planner profile cache at a particular path.
#[derive(Debug, Clone)]
pub struct ProfileCache {
    path: PathBuf,
}

impl ProfileCache {
    /// A cache at an explicit path.
    #[must_use]
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The cache at the location [`cache_path`] derives from the real environment.
    ///
    /// # Errors
    /// [`StoreError::NoCacheDir`] when no cache directory can be derived.
    pub fn discover(configured_dir: &str) -> Result<Self, StoreError> {
        cache_path(configured_dir, |k| std::env::var(k).ok())
            .map(Self::at)
            .ok_or(StoreError::NoCacheDir)
    }

    /// Where this cache reads and writes.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read the whole document; malformed JSON becomes an empty map plus a complaint.
    fn read_document(&self) -> (Map<String, Value>, Option<String>) {
        let empty = Map::new();
        if self.path.as_os_str().is_empty() || !self.path.exists() {
            return (empty, None);
        }
        let text = match std::fs::read_to_string(&self.path) {
            Ok(t) => t,
            // Open failures are treated like a missing cache.
            Err(_) => return (empty, None),
        };
        match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => (map, None),
            // Non-object JSON is ignored.
            Ok(_) => (empty, None),
            Err(e) => (empty, Some(e.to_string())),
        }
    }

    /// The measured shape for `key`, or why it cannot be used.
    #[must_use]
    pub fn lookup(&self, key: &ProfileKey) -> Lookup {
        let (root, complaint) = self.read_document();
        if let Some(c) = complaint {
            return Lookup::Unusable(c);
        }
        let Some(Value::Array(entries)) = root.get("entries") else {
            return Lookup::Miss;
        };

        // Version is checked before emptiness to preserve C++ evaluation order.
        let version = match root.get("version") {
            None => Some(0),
            // Boolean versions read as 0 or 1, then fail the schema check.
            Some(Value::Bool(b)) => Some(i64::from(*b)),
            Some(Value::Number(n)) => n.as_i64().or_else(|| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "mirrors nlohmann's get<int>() on a float, which truncates"
                )]
                n.as_f64().map(|f| f as i64)
            }),
            Some(_) => None,
        };
        let Some(version) = version else {
            return Lookup::Unusable(
                "\"version\" is present but is not a number; delete the file and re-calibrate"
                    .to_owned(),
            );
        };
        // Wrong-version entries may have changed field meanings; refuse them.
        if version != i64::from(SCHEMA_VERSION) && !entries.is_empty() {
            return Lookup::Unusable(format!(
                "schema version {version}, expected {SCHEMA_VERSION}; \
                 delete the file and re-calibrate"
            ));
        }

        for entry in entries {
            let Some(entry) = entry.as_object() else {
                continue;
            };
            let (Some(stored_key), Some(stored_plan)) = (entry.get("key"), entry.get("plan"))
            else {
                continue;
            };
            let (Some(stored_key), Some(stored_plan)) =
                (stored_key.as_object(), stored_plan.as_object())
            else {
                continue;
            };
            if !key.matches(|name| stored_field(stored_key, name)) {
                continue;
            }
            return match read_shape(stored_plan) {
                Ok(shape) => Lookup::Hit(shape),
                Err(e) => Lookup::Unusable(e),
            };
        }
        Lookup::Miss
    }

    /// Replace or append the entry for `key`, preserving other entries.
    ///
    /// The sibling lock covers read-merge-rename; `rename` keeps readers from
    /// observing a partial file. `now_unix_secs` is injected for tests.
    ///
    /// # Errors
    /// [`StoreError`] when directory creation, locking, writing, or renaming fails.
    pub fn store(
        &self,
        key: &ProfileKey,
        shape: &ProfileShape,
        samples: &[ShapeSample],
        now_unix_secs: i64,
    ) -> Result<(), StoreError> {
        if self.path.as_os_str().is_empty() {
            return Err(StoreError::NoCacheDir);
        }
        let parent = self.path.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent).map_err(|source| StoreError::Io {
            what: "create",
            path: parent.to_path_buf(),
            source,
        })?;

        // The lock prevents concurrent writers from losing each other's entries.
        let _lock = CacheLock::acquire(&self.path)?;

        // Unparseable caches are discarded on write; the data is re-derivable.
        let (mut root, _) = self.read_document();
        root.insert("version".to_owned(), Value::from(SCHEMA_VERSION));
        if !root.get("entries").is_some_and(Value::is_array) {
            root.insert("entries".to_owned(), Value::Array(Vec::new()));
        }

        let entry = build_entry(key, shape, samples, now_unix_secs);

        let Some(Value::Array(entries)) = root.get_mut("entries") else {
            unreachable!("entries was just ensured to be an array");
        };
        let existing = entries.iter_mut().find(|e| {
            e.as_object()
                .and_then(|o| o.get("key"))
                .and_then(Value::as_object)
                .is_some_and(|k| key.matches(|name| stored_field(k, name)))
        });
        match existing {
            Some(slot) => *slot = entry,
            None => entries.push(entry),
        }

        let text = dump_pretty(&Value::Object(root), 2);

        // Include a timestamp so same-pid containers sharing $HOME do not collide.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut tmp_name = self
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        tmp_name.push_str(&format!(".tmp.{}.{stamp}", std::process::id()));
        let tmp = parent.join(tmp_name);

        let write_result = std::fs::write(&tmp, text.as_bytes());
        if let Err(source) = write_result {
            let _ = std::fs::remove_file(&tmp);
            return Err(StoreError::Io {
                what: "write",
                path: tmp,
                source,
            });
        }
        if let Err(source) = std::fs::rename(&tmp, &self.path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(StoreError::Io {
                what: "rename onto",
                path: self.path.clone(),
                source,
            });
        }
        Ok(())
    }
}

/// Classify a stored JSON value for [`ProfileKey::matches`].
fn stored_field(map: &Map<String, Value>, name: &str) -> StoredField {
    match map.get(name) {
        None => StoredField::Missing,
        Some(Value::Null) => StoredField::Null,
        Some(Value::Bool(b)) => StoredField::Bool(*b),
        Some(Value::String(s)) => StoredField::Str(s.clone()),
        Some(Value::Number(n)) => n.as_i64().map_or_else(
            || StoredField::Float(n.as_f64().unwrap_or(f64::NAN)),
            StoredField::Int,
        ),
        Some(Value::Array(_) | Value::Object(_)) => StoredField::Missing,
    }
}

/// Read the `plan` object with the same lenient numeric conversions as C++.
fn read_shape(plan: &Map<String, Value>) -> Result<ProfileShape, String> {
    let string = |name: &str| -> Result<String, String> {
        match plan.get(name) {
            None => Ok(String::new()),
            Some(Value::String(s)) => Ok(s.clone()),
            Some(v) => Err(type_complaint(name, "a string", v)),
        }
    };
    let int = |name: &str| -> Result<i32, String> {
        match plan.get(name) {
            None => Ok(0),
            // Booleans are accepted for signed int fields only.
            Some(Value::Bool(b)) => Ok(i32::from(*b)),
            Some(Value::Number(n)) => {
                // C++ truncates floats and narrows by `static_cast`.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "deliberately mirrors nlohmann's narrowing static_cast"
                )]
                let wide = n
                    .as_i64()
                    .unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "deliberately mirrors nlohmann's narrowing static_cast"
                )]
                Ok(wide as i32)
            }
            Some(v) => Err(type_complaint(name, "a number", v)),
        }
    };
    let unsigned = |name: &str| -> Result<u64, String> {
        match plan.get(name) {
            None => Ok(0),
            Some(Value::Number(n)) => {
                if let Some(u) = n.as_u64() {
                    Ok(u)
                } else if let Some(i) = n.as_i64() {
                    // Negative budgets wrap as in C++; drift checking rejects them.
                    #[expect(
                        clippy::cast_sign_loss,
                        reason = "deliberately mirrors the C++'s wrap to SIZE_MAX"
                    )]
                    Ok(i as u64)
                } else {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "deliberately mirrors nlohmann's narrowing static_cast"
                    )]
                    Ok(n.as_f64().unwrap_or(0.0) as u64)
                }
            }
            Some(v) => Err(type_complaint(name, "a number", v)),
        }
    };

    Ok(ProfileShape {
        policy_profile: string("policy_profile")?,
        kv_page_size: int("kv_page_size")?,
        max_forward_tokens: int("max_forward_tokens")?,
        max_forward_requests: int("max_forward_requests")?,
        budget_bytes: unsigned("budget_bytes")?,
    })
}

fn type_complaint(name: &str, want: &str, got: &Value) -> String {
    format!(
        "\"{name}\" must be {want} but is {}; delete the file and re-calibrate",
        kind(got)
    )
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Build the entry object, omitting zero and empty plan fields.
fn build_entry(
    key: &ProfileKey,
    shape: &ProfileShape,
    samples: &[ShapeSample],
    now_unix_secs: i64,
) -> Value {
    let mut plan = Map::new();
    if !shape.policy_profile.is_empty() {
        plan.insert(
            "policy_profile".to_owned(),
            Value::from(shape.policy_profile.clone()),
        );
    }
    if shape.kv_page_size > 0 {
        plan.insert("kv_page_size".to_owned(), Value::from(shape.kv_page_size));
    }
    if shape.max_forward_tokens > 0 {
        plan.insert(
            "max_forward_tokens".to_owned(),
            Value::from(shape.max_forward_tokens),
        );
    }
    if shape.max_forward_requests > 0 {
        plan.insert(
            "max_forward_requests".to_owned(),
            Value::from(shape.max_forward_requests),
        );
    }
    if shape.budget_bytes > 0 {
        plan.insert("budget_bytes".to_owned(), Value::from(shape.budget_bytes));
    }

    let measured: Vec<Value> = samples
        .iter()
        .map(|s| {
            let mut m = Map::new();
            m.insert(
                "max_forward_tokens".to_owned(),
                Value::from(s.max_forward_tokens),
            );
            m.insert(
                "max_forward_requests".to_owned(),
                Value::from(s.max_forward_requests),
            );
            m.insert(
                "tokens_per_request".to_owned(),
                Value::from(s.tokens_per_request),
            );
            m.insert("step_ms".to_owned(), Value::from(s.step_ms));
            m.insert("step_ms_stddev".to_owned(), Value::from(s.step_ms_stddev));
            m.insert("tokens_per_s".to_owned(), Value::from(s.tokens_per_s));
            Value::Object(m)
        })
        .collect();

    // Reparse `ProfileKey::to_json` so reader and writer share one key format.
    let key_value: Value =
        serde_json::from_str(&key.to_json()).unwrap_or_else(|_| Value::Object(Map::new()));

    let mut entry = Map::new();
    entry.insert("key".to_owned(), key_value);
    entry.insert("plan".to_owned(), Value::Object(plan));
    entry.insert("measured".to_owned(), Value::Array(measured));
    entry.insert("measured_at".to_owned(), Value::from(now_unix_secs));
    Value::Object(entry)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> ProfileKey {
        ProfileKey {
            gpu_name: "NVIDIA L40S".into(),
            compute_major: 8,
            compute_minor: 9,
            sm_count: 142,
            kv_cache_dtype: "bf16".into(),
            tp_size: 1,
            model_type: "llama".into(),
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
        }
    }

    fn shape() -> ProfileShape {
        ProfileShape {
            policy_profile: "throughput".into(),
            kv_page_size: 16,
            max_forward_tokens: 8192,
            max_forward_requests: 256,
            budget_bytes: 40 * 1024 * 1024 * 1024,
        }
    }

    #[test]
    fn the_path_prefers_the_configured_dir_then_xdg_then_home() {
        let env = |k: &str| match k {
            "XDG_CACHE_HOME" => Some("/xdg".to_owned()),
            "HOME" => Some("/home/u".to_owned()),
            _ => None,
        };
        assert_eq!(
            cache_path("/cfg", env).unwrap(),
            Path::new("/cfg/cuda_memory_profiles.json")
        );
        assert_eq!(
            cache_path("", env).unwrap(),
            Path::new("/xdg/pie/cuda_memory_profiles.json")
        );
        assert_eq!(
            cache_path("", |k| (k == "HOME").then(|| "/home/u".to_owned())).unwrap(),
            Path::new("/home/u/.cache/pie/cuda_memory_profiles.json")
        );
        assert!(cache_path("", |_| None).is_none());
    }

    #[test]
    fn an_empty_env_var_counts_as_unset() {
        // Empty XDG_CACHE_HOME is treated as unset.
        let env = |k: &str| match k {
            "XDG_CACHE_HOME" => Some(String::new()),
            "HOME" => Some("/home/u".to_owned()),
            _ => None,
        };
        assert_eq!(
            cache_path("", env).unwrap(),
            Path::new("/home/u/.cache/pie/cuda_memory_profiles.json")
        );
    }

    #[test]
    fn a_missing_cache_is_a_miss_and_not_a_complaint() {
        let c = ProfileCache::at("/nonexistent/dir/cuda_memory_profiles.json");
        assert_eq!(c.lookup(&key()), Lookup::Miss);
    }

    #[test]
    fn zero_budget_means_unrecorded_and_always_applies() {
        let s = ProfileShape {
            budget_bytes: 0,
            ..shape()
        };
        assert!(s.applies_at(1));
        assert!(s.applies_at(u64::MAX));
    }

    #[test]
    fn drift_is_measured_against_the_recorded_budget() {
        // The window is asymmetric because the denominator is the measured budget.
        let s = ProfileShape {
            budget_bytes: 1000,
            ..shape()
        };
        assert!(s.applies_at(1050));
        assert!(s.applies_at(950));
        assert!(!s.applies_at(1051));
        assert!(!s.applies_at(949));
        assert!((s.drift_from(1100) - 0.1).abs() < 1e-12);
    }
}
