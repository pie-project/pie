//! Config keys that are gone, and the refusal each one earns.
//!
//! Every config struct denies unknown fields, so a key no struct declares
//! fails to parse. Alone, that answers a retired key with "unknown field",
//! which reads as a typo. [`explain`] maps that error, for a key this table
//! lists, to what happened to it:
//!
//! ```text
//! `max_model_len` was renamed to `max_context` in `[engine]`
//! `max_forward_requests` is no longer a setting: a fire's lanes are the
//! largest rung the device holds, derived at load
//! ```
//!
//! Retiring a key is deleting its field and adding one row to [`RETIRED`];
//! a renamed key is refused like a removed one, never read under its old
//! name. The same table answers `pie config set` of a retired key.

use super::EngineKind;

/// What became of a retired key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fate {
    /// Its value is read under another name, in the table named.
    Renamed {
        to: &'static str,
        table: &'static str,
    },
    /// Nothing reads it; the text says what replaces it.
    Gone(&'static str),
}

/// One retired key.
#[derive(Debug, Clone, Copy)]
pub struct Retired {
    /// The key as a file spelled it.
    pub key: &'static str,
    /// The one engine it is retired on, where another engine still reads a
    /// key of that name; `None` for every engine and table.
    pub engine: Option<EngineKind>,
    pub fate: Fate,
}

const fn renamed(key: &'static str, table: &'static str, to: &'static str) -> Retired {
    Retired {
        key,
        engine: None,
        fate: Fate::Renamed { to, table },
    }
}

const fn gone(key: &'static str, why: &'static str) -> Retired {
    Retired {
        key,
        engine: None,
        fate: Fate::Gone(why),
    }
}

/// Every retired key. The app settings (`embedded::Settings`) take the
/// current keys under the same names, so this one table serves both.
pub const RETIRED: &[Retired] = &[
    // `[engine]`, each engine's options.
    renamed("max_model_len", "engine", "max_context"),
    gone(
        "max_forward_tokens",
        "a fire's rows are the largest rung the device holds, derived at load; a request \
         past it is refused",
    ),
    gone(
        "max_forward_requests",
        "a fire's lanes are the largest rung the device holds, derived at load, and \
         `[runtime] max_concurrent_processes` caps the inferlets admitted at once",
    ),
    gone(
        "max_state_slots",
        "state shares the device with kv pages and takes what demand asks for",
    ),
    gone("kv_page_size", "a kv page is the compiler's chunk plan"),
    gone(
        "device_memory",
        "the load measures the device; where the adapter says nothing, state \
         `[engine.memory] budget_device`",
    ),
    Retired {
        key: "pipeline_cache",
        engine: Some(EngineKind::Wgpu),
        fate: Fate::Gone(
            "the wgpu engine serves the browser and keeps no pipeline cache; a native \
             desktop runs the vulkan engine, whose `pipeline_cache` stays",
        ),
    },
    // `[model]`, now `[engine.memory]`.
    renamed("host_kv_budget", "engine.memory", "budget_host"),
    renamed("disk_kv_budget", "engine.memory", "budget_disk"),
    renamed("disk_kv_dir", "engine.memory", "disk_path"),
    renamed("disk_kv_persistent", "engine.memory", "disk_persistent"),
    gone(
        "device_kv_budget",
        "the kv pages and state slots take what the device holds past the weights and \
         the fires; to cap what the whole load takes, state `[engine.memory] budget_device`",
    ),
    gone(
        "device_memory_share",
        "the load measures the device and takes all of it; to share the device with \
         another process, state `[engine.memory] budget_device`",
    ),
    gone(
        "memory_tolerance",
        "state `[engine.memory] governor_backoff` (the share of memory kept each congested \
         stretch: dedicated 1.0, aggressive 0.9, balanced 0.8, polite 0.7) and \
         `governor_release_idle = true` to give back memory demand stopped asking for",
    ),
    gone(
        "weight_streaming",
        "the weight cache has no mode: every streamable weight stays resident while the \
         device has room, and live demand takes chunks from it before any kv or state \
         gives way",
    ),
    gone(
        "max_patches",
        "the patch rungs are derived from the device like the token rungs; a request past the \
         largest is refused",
    ),
    gone(
        "max_images",
        "the patch rungs are derived from the device like the token rungs; a request past the \
         largest is refused",
    ),
    gone(
        "max_voxels",
        "the voxel rungs are derived from the device like the token rungs; a request past the \
         largest is refused",
    ),
    gone(
        "max_clips",
        "the voxel rungs are derived from the device like the token rungs; a request past the \
         largest is refused",
    ),
    // `[model]`, retired before main.
    gone(
        "device_weight_budget",
        "dense weights always live in device memory; only routed experts and token-id \
         tables leave it, through the weight cache. A model whose dense weights do not \
         fit the device needs a quantized deployment or tensor parallelism",
    ),
    gone(
        "host_weight_budget",
        "dense weights always live in device memory; only routed experts and token-id \
         tables leave it, through the weight cache, which sizes its host copy itself",
    ),
    gone(
        "host_weight_deferred",
        "dense weights always live in device memory, and the weight cache fills its own \
         host copy",
    ),
];

/// The refusal for `key` (dotted, as `pie config set` takes it, or bare)
/// on `engine`, if it names a retired key. `None` for `engine` matches only
/// keys retired everywhere.
#[must_use]
pub fn refusal_for(key: &str, engine: Option<EngineKind>) -> Option<String> {
    let leaf = key.rsplit('.').next().unwrap_or(key);
    RETIRED
        .iter()
        .find(|r| r.key == leaf && (r.engine.is_none() || r.engine == engine))
        .map(|r| refusal(r, true))
}

fn refusal(retired: &Retired, in_file: bool) -> String {
    let key = retired.key;
    match retired.fate {
        Fate::Renamed { to, table } if in_file => {
            format!("`{key}` was renamed to `{to}` in `[{table}]`")
        }
        Fate::Renamed { to, .. } => format!("`{key}` was renamed to `{to}`"),
        Fate::Gone(why) => format!("`{key}` is no longer a setting: {why}"),
    }
}

/// The retired key a deserializer's "unknown field" error names, if any.
fn named_by(error: &str) -> Option<&str> {
    let rest = error.split("unknown field `").nth(1)?;
    rest.split('`').next()
}

/// The refusal for a deserializer's error, where it is serde's "unknown
/// field" for a retired key; `None` for any other error.
#[must_use]
pub fn explain(error: &str, engine: Option<EngineKind>) -> Option<String> {
    refusal_for(named_by(error)?, engine)
}

/// [`explain`] for an app's settings, which are one flat table: a renamed
/// key is named without the config file's table.
#[must_use]
pub fn explain_settings(error: &str) -> Option<String> {
    let key = named_by(error)?;
    RETIRED
        .iter()
        .find(|r| r.key == key && r.engine.is_none())
        .map(|r| refusal(r, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retired_every_case() {
        each_key_is_listed_once_per_engine();
        a_renamed_key_names_its_new_name_and_table();
        only_an_unknown_field_error_is_explained();
    }

    fn each_key_is_listed_once_per_engine() {
        for (i, a) in RETIRED.iter().enumerate() {
            for b in &RETIRED[i + 1..] {
                assert!(
                    a.key != b.key || a.engine != b.engine,
                    "{} is listed twice",
                    a.key
                );
            }
        }
    }

    fn a_renamed_key_names_its_new_name_and_table() {
        assert_eq!(
            refusal_for("engine.max_model_len", None).as_deref(),
            Some("`max_model_len` was renamed to `max_context` in `[engine]`")
        );
        assert_eq!(
            refusal_for("host_kv_budget", None).as_deref(),
            Some("`host_kv_budget` was renamed to `budget_host` in `[engine.memory]`")
        );
    }

    fn only_an_unknown_field_error_is_explained() {
        assert!(
            explain("unknown field `max_forward_requests`, expected one of", None)
                .unwrap()
                .starts_with("`max_forward_requests` is no longer a setting: ")
        );
        assert_eq!(explain("unknown field `max_contxt`, expected", None), None);
        assert_eq!(explain("invalid type: string \"x\"", None), None);
        assert_eq!(
            explain("unknown field `pipeline_cache`", Some(EngineKind::Vulkan)),
            None
        );
        assert!(explain("unknown field `pipeline_cache`", Some(EngineKind::Wgpu)).is_some());
    }
}
