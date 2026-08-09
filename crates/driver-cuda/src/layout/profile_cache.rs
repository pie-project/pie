//! The on-disk cache of **measured** planner shapes.
//!
//! `plan_cuda_memory` scores its candidate lattice analytically, and a score
//! is a model of how a shape will perform. Where the model disagreed with
//! reality the C++ tree accumulated per-(model, GPU) overrides with
//! hand-measured constants baked in. This cache is the mechanism that replaces
//! those constants with the driver's own measurement: the calibrator times the
//! real forward step across the shape ladder and stores the winner here, and
//! the planner reads it back and selects by evidence instead of by score.
//!
//! Reader and writer share [`ProfileKey`] deliberately. The key has twelve
//! fields, and if the two sides built it independently a single disagreement
//! would make every lookup miss silently -- the cache would look empty rather
//! than broken.
//!
//! # Divergences from the C++, and why
//!
//! `planner_profile_cache_lookup` documents itself as *"Never throws: a
//! corrupt cache degrades to 'no measurement'"*. It does not achieve that.
//! `nlohmann::json::value(key, default)` throws `type_error.302` when the key
//! is present with an incompatible type, and the lookup calls it outside any
//! `try`. Verified against nlohmann directly:
//!
//! | stored | C++ | here |
//! |---|---|---|
//! | `"version": 2` | 2 | same |
//! | `"version": 2.0` | 2 -- a float version *passes* | same |
//! | `"version": "2"` | **throws** | [`Lookup::Unusable`] |
//! | `"version": null` | **throws** | [`Lookup::Unusable`] |
//! | `"policy_profile": 7` | **throws** | [`Lookup::Unusable`] |
//! | `"budget_bytes": -1` | `18446744073709551615` | same |
//! | `"budget_bytes": 1.5` | `1` | same |
//! | `"budget_bytes": null` | **throws** | [`Lookup::Unusable`] |
//!
//! Where the C++ is well-defined this matches it, quirks included -- the
//! silent wrap of a negative budget to `u64::MAX` is reproduced rather than
//! fixed, because the two implementations have to agree about a file they
//! share. Where the C++ throws, this returns the degradation the C++ header
//! promises, since an exception escaping into `plan_cuda_memory` is the one
//! behaviour nothing downstream is written to survive.
//!
//! Note also that the key fields are compared **type-strictly** (a
//! `sm_count` of `132.0` does not match `132`) while the plan fields are
//! read **leniently** (a `kv_page_size` of `16.0` reads as `16`). That
//! inconsistency is in the C++ and is preserved.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Map, Value};

use super::json::dump_pretty;
use super::profile_key::{ProfileKey, ProfileShape, SCHEMA_VERSION, StoredField};

/// How far the planner budget may drift from the one a cached shape was
/// measured under before the shape stops applying.
///
/// Loose enough to absorb ordinary boot-to-boot variation in what the device
/// reports free; tight enough that a requantized checkpoint or a stray process
/// holding gigabytes is a miss rather than a silently stale answer.
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
    /// Mean measured step time.
    pub step_ms: f64,
    /// Standard deviation across repeats.
    pub step_ms_stddev: f64,
    /// Derived throughput.
    pub tokens_per_s: f64,
}

/// What a lookup found.
///
/// Three outcomes, because the C++ has three and expresses them through a
/// return value and an out-parameter that callers can and do read
/// independently. Collapsing "no entry" and "cache is broken" into one
/// `Option` would lose the distinction that decides whether anything gets
/// logged.
#[derive(Debug, Clone, PartialEq)]
pub enum Lookup {
    /// An entry for this key, readable and at the expected schema version.
    Hit(ProfileShape),
    /// No entry for this key. Includes "no cache file", which is the ordinary
    /// first-boot case and is not worth a word to the user.
    Miss,
    /// A cache exists but cannot be trusted: unparseable, or written at a
    /// schema version whose fields may not mean what they say. The string is
    /// for the operator, and the caller should proceed as if it were a
    /// [`Lookup::Miss`].
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
    /// Does a shape measured under its recorded budget still describe a
    /// machine whose budget is now `budget`?
    ///
    /// A shape is only an answer to the budget it was measured under. The key
    /// pins the device and the model, and neither notices that this boot has
    /// materially more or less memory to give -- another process holding VRAM,
    /// or a checkpoint requantized offline. Unchecked this fails in the quiet
    /// direction: with a *larger* budget the measured shape is still feasible,
    /// so it is selected and the extra memory is simply never used.
    ///
    /// A recorded budget of zero means "not recorded", and always applies.
    #[must_use]
    pub fn applies_at(&self, budget: u64) -> bool {
        if self.budget_bytes == 0 {
            return true;
        }
        self.drift_from(budget) <= BUDGET_TOLERANCE
    }

    /// The fractional distance from the recorded budget, as the C++ computes
    /// it: divided by the **measured** budget, not by the current one.
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

/// Where the cache lives.
///
/// The same derivation the module and tuning caches use: the configured cache
/// directory when the engine sent one, else XDG, else `$HOME/.cache`. `None`
/// when none of those is set, which is a real configuration on a locked-down
/// host and is why the C++ returns an empty path rather than guessing.
///
/// `env` is a parameter rather than a call to [`std::env::var`] so this is
/// testable without mutating the process environment -- which is unsound to do
/// from a test harness that runs threads in parallel.
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

/// The planner budget this boot computed, published so the calibrator can
/// record what its sweep ran inside.
///
/// A process-wide global for the same reason the C++ makes it a file-static:
/// `plan_cuda_memory` derives the budget and the calibrator stores the result,
/// and there is no object both of them hold. Written once, before anything
/// reads it.
///
/// The C++ is a bare `std::size_t`, which is a data race the moment two
/// threads touch it. An atomic costs nothing here -- the value is read once
/// per calibration -- and removes the question.
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
    /// A filesystem operation failed. `what` names the step, so the message
    /// says which of create/open/write/rename went wrong rather than leaving
    /// the errno to be interpreted.
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

/// An exclusive advisory lock, held on a sibling `.lock` file.
///
/// On the sibling rather than on the cache itself because [`ProfileCache::store`]
/// replaces the cache by `rename`, and a lock held on the replaced inode would
/// stop excluding anything the moment the rename landed.
struct CacheLock {
    /// Dropping the file releases the lock, which is why there is no
    /// `Drop` impl below any more.
    _file: std::fs::File,
}

impl CacheLock {
    /// Open the sibling and take the lock.
    ///
    /// NO `unsafe`, and that is the point rather than a detail. This was
    /// `libc::open` + `libc::flock` + a `Drop` calling `libc::close`,
    /// with three SAFETY comments carrying the argument that the
    /// descriptor is opened once, locked once and closed once. All three
    /// are now the type's: `File` owns the descriptor, `File::lock`
    /// takes the same advisory lock, and dropping the file releases and
    /// closes it.
    ///
    /// It is the only `unsafe` `layout/` had, and removing it is what
    /// lets this half of the crate carry `#![forbid(unsafe_code)]` —
    /// §8 row 11's thesis, held by the compiler on the half where it is
    /// reachable today.
    fn acquire(cache_path: &Path) -> Result<Self, StoreError> {
        let mut name = cache_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        name.push_str(".lock");
        let lock_path = cache_path.parent().unwrap_or(Path::new(".")).join(name);

        // The error carries the OS's own, not `last_os_error()` — which
        // is what the `libc` version had to read, and what a `File`
        // returns directly.
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

    /// The cache at the location [`cache_path`] derives, reading the real
    /// environment.
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

    /// Read the whole document, tolerating every failure mode.
    ///
    /// A cache that cannot be parsed must not take the process down: it is an
    /// optimisation record, and the planner's analytic score is a complete
    /// fallback. Returns the empty object plus a complaint when the file
    /// exists but is not usable JSON; the empty object with no complaint when
    /// it is simply absent.
    fn read_document(&self) -> (Map<String, Value>, Option<String>) {
        let empty = Map::new();
        if self.path.as_os_str().is_empty() || !self.path.exists() {
            return (empty, None);
        }
        let text = match std::fs::read_to_string(&self.path) {
            Ok(t) => t,
            // The C++ returns an empty document with no error when the stream
            // fails to open, and only reports parse failures.
            Err(_) => return (empty, None),
        };
        match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => (map, None),
            // Valid JSON that is not an object: the C++ discards it silently.
            Ok(_) => (empty, None),
            Err(e) => (empty, Some(e.to_string())),
        }
    }

    /// The measured shape for `key`.
    ///
    /// Never fails: see the module docs for the three outcomes and for where
    /// this deliberately differs from the C++.
    #[must_use]
    pub fn lookup(&self, key: &ProfileKey) -> Lookup {
        let (root, complaint) = self.read_document();
        if let Some(c) = complaint {
            return Lookup::Unusable(c);
        }
        let Some(Value::Array(entries)) = root.get("entries") else {
            return Lookup::Miss;
        };

        // Read the version before the emptiness check, matching the C++'s
        // evaluation order -- so a badly typed version is reported even when
        // there is nothing it could have applied to.
        let version = match root.get("version") {
            None => Some(0),
            // A boolean version reads as 0 or 1, and then fails the schema
            // check with that number in the message. Odd, but it is what the
            // C++ does, and the outcome -- refusal -- is the same.
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
        // A document with entries but the wrong version was written by another
        // build (or by hand). Its fields may not mean what they say, so refuse
        // it loudly rather than matching the subset that happens to line up --
        // a wrong `max_forward_tokens` is worse than none.
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

    /// Replace (or append) the entry for `key`, preserving every other entry.
    ///
    /// Serialised and atomic. An exclusive `flock` on a sibling `.lock` file
    /// covers the whole read -> merge -> rename, so two processes calibrating
    /// at once cannot drop each other's entries; the rename itself means a
    /// concurrent reader observes either the old document or the new one and
    /// never a partial write.
    ///
    /// `now_unix_secs` is a parameter rather than a clock read so the output
    /// is reproducible under test. The C++ takes it from
    /// `system_clock::now()`.
    ///
    /// # Errors
    /// [`StoreError`] when a directory cannot be created, the lock cannot be
    /// taken, or the write or rename fails. A cache that exists but cannot be
    /// parsed is **not** an error: like the C++, it is discarded and replaced.
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

        // `rename` makes the replacement atomic for readers but does nothing
        // for read-modify-write against another writer: two processes
        // calibrating at once would both read the pre-existing document and
        // the second rename would drop the first one's entry. One process per
        // GPU on a multi-GPU host sharing $HOME is exactly the case the
        // "preserve every other entry" contract exists for.
        let _lock = CacheLock::acquire(&self.path)?;

        // Deliberately ignoring the complaint: the C++ passes `nullptr` here,
        // so an unparseable cache is discarded and replaced rather than
        // blocking the write. The entries it held are lost, which is the
        // right trade for a file whose only content is re-derivable.
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

        // The temp name carries a clock component as well as the pid, because
        // two containers sharing a $HOME mount can present the same pid.
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

/// Read the `plan` object, reproducing `nlohmann::value()`'s exact tolerances.
///
/// Those tolerances are not uniform, and the asymmetry is not guessable -- it
/// was measured. `value(key, 0)` deduces `int`, which is *not* nlohmann's
/// `number_integer_t` (that is `int64_t`), so it takes a generic arithmetic
/// path that accepts booleans and truncates floats. `value<std::size_t>(key,
/// 0)` names `number_unsigned_t` exactly and takes a stricter path that
/// throws on a boolean while still accepting a float or a negative.
///
/// | stored | `int` field | `size_t` field | string field |
/// |---|---|---|---|
/// | `16` | 16 | 16 | throws |
/// | `-5` | -5 | wraps to `u64::MAX - 4` | throws |
/// | `16.9` | 16 | 16 | throws |
/// | `true` | **1** | **throws** | throws |
/// | `null` | throws | throws | throws |
/// | `"16"` | throws | throws | verbatim |
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
            // A boolean reads as 0 or 1 here but not in `unsigned` below.
            Some(Value::Bool(b)) => Ok(i32::from(*b)),
            Some(Value::Number(n)) => {
                // `get<int>()` truncates a float toward zero and narrows an
                // out-of-range value by `static_cast`, which wraps.
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
                    // A negative budget wraps to a colossal one. Preserved
                    // rather than fixed: it is what the C++ reads, and the
                    // drift check then rejects the entry, so the quirk
                    // happens to fail safe.
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

/// Build the entry object. Zero and empty fields are **omitted**, so a cache
/// written by a calibrator that only swept `max_forward_tokens` does not pin
/// the fields it never measured.
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

    // The key is serialised through `ProfileKey::to_json` and reparsed rather
    // than built field by field here, so there is exactly one definition of
    // what a stored key looks like and the writer cannot drift from the
    // reader.
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
        // `getenv` returning "" is not the same as returning null, and the
        // C++ checks `xdg[0] != '\0'` explicitly. Without this an exported
        // but empty XDG_CACHE_HOME would put the cache at `/pie/...`.
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
        // First boot is the common case and must be silent.
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
        // Asymmetric on purpose: the C++ divides by the measured budget, so
        // the window is not symmetric in absolute bytes around it.
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
