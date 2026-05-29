//! Resolve which Python interpreter to invoke for a subprocess driver.
//!
//! Precedence (highest → lowest):
//!
//!   1. `[model.driver.options].venv` or `…python` — per-model override
//!      written into the user's serve TOML.
//!   2. `$PIE_PYTHON` — process-env override (CI, scripts).
//!   3. `$VIRTUAL_ENV/bin/python` — the venv the user has activated.
//!   4. `~/.pie/drivers.toml` `[driver.<type>].venv|python` — persisted
//!      per-driver default set via `pie driver <type> set venv …`.
//!   5. `~/.pie/drivers.toml` `[python].venv|python` — persisted shared
//!      default for any subprocess driver that didn't override.
//!   6. `which python3` from `$PATH`.
//!   7. Hard error pointing at `pie driver <type> install`.
//!
//! `pie driver <type> show` prints the resolved path *and* the source
//! step from this chain so users can debug a wrong choice without
//! re-deriving the precedence.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde::Deserialize;

use crate::subprocess_driver::SubprocessFlavor;

/// `~/.pie/drivers.toml`. Distinct from `default_config_path()`
/// (`~/.pie/config.toml`, the *server* config) so the two concerns
/// — "which model do I serve" vs. "which interpreter do I use" —
/// stay decoupled.
pub fn drivers_config_path() -> PathBuf {
    crate::paths::pie_home().join("drivers.toml")
}

/// Top-level `~/.pie/drivers.toml` schema.
///
/// ```toml
/// # Shared default for every subprocess driver.
/// [python]
/// venv = "/home/me/envs/all"
///
/// # Per-driver override; takes precedence over the shared default.
/// [driver.vllm]
/// venv = "/home/me/envs/vllm-cu128"
///
/// [driver.sglang]
/// python = "/home/me/envs/sgl/bin/python"
///
/// [driver.tensorrt_llm]
/// venv = "/home/me/envs/trtllm"
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DriversConfig {
    #[serde(default)]
    pub python: PythonBlock,
    /// Keyed by `[driver.<type>]` — the same string as the
    /// `[[model]].driver.type` discriminator.
    #[serde(default, rename = "driver")]
    pub drivers: BTreeMap<String, PythonBlock>,
}

/// `[python]` / `[driver.<type>]` block. Either `venv` (resolved as
/// `<venv>/bin/python`) or `python` (direct path) — both set is an
/// error.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PythonBlock {
    pub venv: Option<String>,
    pub python: Option<String>,
}

impl DriversConfig {
    /// Read `~/.pie/drivers.toml`, returning an empty config if the file
    /// doesn't exist (the common case — users start without one and add
    /// entries via `pie driver <type> set venv …`).
    pub fn load() -> Result<Self> {
        let path = drivers_config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path).map_err(|e| anyhow!("read {path:?}: {e}"))?;
        toml::from_str(&text).map_err(|e| anyhow!("parse {path:?}: {e}"))
    }
}

/// One link in the resolution chain — both the resolved interpreter
/// path and the human-readable source it came from.
#[derive(Debug, Clone)]
pub struct ResolvedPython {
    pub path: PathBuf,
    /// Free-form description of which precedence step matched, e.g.
    /// `"$PIE_PYTHON"`, `"~/.pie/drivers.toml [driver.vllm].venv"`.
    /// Surfaced verbatim by `pie driver <type> show`.
    pub source: String,
}

/// Resolve the interpreter for one subprocess driver invocation.
///
/// `model_options` is the user's `[model.driver.options]` table —
/// `venv` / `python` are read out of it (and stripped by the caller
/// via [`strip_python_keys`] before the table is forwarded to the
/// launcher). Pass an empty table for CLI subcommands that aren't
/// tied to a particular `[[model]]` (e.g. `pie driver vllm show`).
///
/// `global` is the parsed `~/.pie/drivers.toml`; pass `None` when the
/// caller has already decided to bypass it (`pie driver <type> exec`'s
/// `--ignore-config` would).
pub fn resolve_python(
    flavor: SubprocessFlavor,
    model_options: &toml::Table,
    global: Option<&DriversConfig>,
) -> Result<ResolvedPython> {
    // 1. [model.driver.options].venv | python
    if let Some(p) = python_from_options(model_options, "model.driver.options")? {
        return Ok(p);
    }

    // 2. $PIE_PYTHON
    if let Ok(env) = std::env::var("PIE_PYTHON") {
        if !env.is_empty() {
            return Ok(ResolvedPython {
                path: PathBuf::from(env),
                source: "$PIE_PYTHON".into(),
            });
        }
    }

    // 3. $VIRTUAL_ENV/bin/python — only honoured if the binary actually
    // exists; otherwise we'd silently surface a stale env var.
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        if !venv.is_empty() {
            let candidate = PathBuf::from(&venv).join("bin").join("python");
            if candidate.is_file() {
                return Ok(ResolvedPython {
                    path: candidate,
                    source: format!("$VIRTUAL_ENV ({venv})"),
                });
            }
        }
    }

    // 4. ~/.pie/drivers.toml [driver.<type>]
    if let Some(g) = global {
        if let Some(per_driver) = g.drivers.get(flavor.as_str()) {
            if let Some(p) = python_from_block(
                per_driver,
                &format!("~/.pie/drivers.toml [driver.{}]", flavor.as_str()),
            )? {
                return Ok(p);
            }
        }
        // 5. ~/.pie/drivers.toml [python]
        if let Some(p) = python_from_block(&g.python, "~/.pie/drivers.toml [python]")? {
            return Ok(p);
        }
    }

    // 6. which python3
    if let Some(path) = which_in_path("python3") {
        return Ok(ResolvedPython {
            path,
            source: "$PATH (which python3)".into(),
        });
    }

    // 7. Hard error.
    bail!(
        "no Python interpreter found for driver `{name}`. \n\
         Configure one with one of:\n  \
         * `pie driver {name} set venv <path>`  (persists into \
         ~/.pie/drivers.toml)\n  \
         * set `$PIE_PYTHON=/path/to/python`\n  \
         * activate a venv (`$VIRTUAL_ENV` is honoured)\n  \
         * add `venv = \"<path>\"` under [model.driver.options]\n\
         If you don't have a venv yet, `pie driver {name} install` \
         prints the install recipe.",
        name = flavor.as_str(),
    );
}

/// Strip `venv` / `python` keys from the user's `[driver.options]`
/// table before passing it to the launcher TOML. The launcher's typed
/// `<DriverName>DriverConfig` dataclass would error on these unknown
/// keys; they're standalone-side concerns.
pub fn strip_python_keys(options: &mut toml::Table) {
    options.remove("venv");
    options.remove("python");
}

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

/// Read `venv` / `python` out of an options-shaped TOML table, with
/// "both set" rejected.
fn python_from_options(options: &toml::Table, source: &str) -> Result<Option<ResolvedPython>> {
    let venv = options
        .get("venv")
        .and_then(|v| v.as_str())
        .map(String::from);
    let python = options
        .get("python")
        .and_then(|v| v.as_str())
        .map(String::from);
    let block = PythonBlock { venv, python };
    python_from_block(&block, source)
}

/// Read `venv` / `python` out of a typed [`PythonBlock`].
fn python_from_block(block: &PythonBlock, source: &str) -> Result<Option<ResolvedPython>> {
    match (block.venv.as_deref(), block.python.as_deref()) {
        (Some(_), Some(_)) => Err(anyhow!(
            "{source}: both `venv` and `python` are set — pick one. `venv` \
             resolves as <venv>/bin/python; `python` is a direct interpreter path."
        )),
        (Some(venv), None) => Ok(Some(ResolvedPython {
            path: PathBuf::from(venv).join("bin").join("python"),
            source: format!("{source}.venv ({venv})"),
        })),
        (None, Some(python)) => Ok(Some(ResolvedPython {
            path: PathBuf::from(python),
            source: format!("{source}.python"),
        })),
        (None, None) => Ok(None),
    }
}

/// Minimal `which` — walk `$PATH` for an executable file. Avoids
/// pulling in the `which` crate for one call site.
fn which_in_path(name: &str) -> Option<PathBuf> {
    let path_var: OsString = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    meta.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_options(toml_options: &str) -> toml::Table {
        toml::from_str::<toml::Table>(toml_options).unwrap()
    }

    /// `cargo test` runs tests in parallel by default. The env-mutating
    /// tests below would race each other (one test's `set_var` getting
    /// `remove_var`'d by the next test's setup); serializing through
    /// this lock keeps each test's env state consistent during its
    /// `resolve_python` call.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_python_envs() {
        // SAFETY: caller holds `ENV_LOCK`, so no other test thread is
        // reading/writing these vars concurrently.
        unsafe {
            std::env::remove_var("PIE_PYTHON");
            std::env::remove_var("VIRTUAL_ENV");
        }
    }

    #[test]
    fn options_venv_wins() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_python_envs();
        let m = fixture_options("venv = \"/tmp/foo\"");
        let r = resolve_python(SubprocessFlavor::Vllm, &m, None).unwrap();
        assert_eq!(r.path, PathBuf::from("/tmp/foo/bin/python"));
        assert!(
            r.source.contains("model.driver.options"),
            "got: {}",
            r.source
        );
    }

    #[test]
    fn options_python_wins() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_python_envs();
        let m = fixture_options("python = \"/usr/bin/python3.12\"");
        let r = resolve_python(SubprocessFlavor::Vllm, &m, None).unwrap();
        assert_eq!(r.path, PathBuf::from("/usr/bin/python3.12"));
    }

    #[test]
    fn both_set_errors() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_python_envs();
        let m = fixture_options("venv = \"/a\"\npython = \"/b\"");
        let err = resolve_python(SubprocessFlavor::Vllm, &m, None).unwrap_err();
        assert!(
            err.to_string().contains("both `venv` and `python`"),
            "got: {err}"
        );
    }

    #[test]
    fn pie_python_env_wins_over_global() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_python_envs();
        unsafe { std::env::set_var("PIE_PYTHON", "/from/env") };
        let m = fixture_options("");
        let mut g = DriversConfig::default();
        g.python.venv = Some("/from/global".into());
        let r = resolve_python(SubprocessFlavor::Vllm, &m, Some(&g)).unwrap();
        assert_eq!(r.path, PathBuf::from("/from/env"));
        assert_eq!(r.source, "$PIE_PYTHON");
        unsafe { std::env::remove_var("PIE_PYTHON") };
    }

    #[test]
    fn per_driver_global_wins_over_top_level() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_python_envs();
        let m = fixture_options("");
        let mut g = DriversConfig::default();
        g.python.venv = Some("/shared".into());
        g.drivers.insert(
            "vllm".into(),
            PythonBlock {
                venv: Some("/per-driver".into()),
                python: None,
            },
        );
        let r = resolve_python(SubprocessFlavor::Vllm, &m, Some(&g)).unwrap();
        assert_eq!(r.path, PathBuf::from("/per-driver/bin/python"));
        assert!(r.source.contains("[driver.vllm]"), "got: {}", r.source);
    }

    #[test]
    fn top_level_global_used_when_per_driver_absent() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_python_envs();
        let m = fixture_options("");
        let mut g = DriversConfig::default();
        g.python.python = Some("/shared/python".into());
        let r = resolve_python(SubprocessFlavor::Sglang, &m, Some(&g)).unwrap();
        assert_eq!(r.path, PathBuf::from("/shared/python"));
        assert!(r.source.contains("[python]"), "got: {}", r.source);
    }

    #[test]
    fn unconfigured_falls_through_to_path() {
        // `python3` is virtually always on a dev box. If the test box
        // is exotic enough not to have it, accept the explicit error
        // — both paths through the chain validate the precedence
        // logic equally.
        let _g = ENV_LOCK.lock().unwrap();
        clear_python_envs();
        let m = fixture_options("");
        match resolve_python(SubprocessFlavor::Vllm, &m, None) {
            Ok(r) => {
                assert!(r.source.contains("$PATH"), "got: {}", r.source);
            }
            Err(e) => {
                assert!(
                    e.to_string().contains("no Python interpreter found"),
                    "got: {e}"
                );
            }
        }
    }

    #[test]
    fn strip_python_keys_drops_both() {
        let mut t: toml::Table =
            toml::from_str("venv = \"/tmp\"\npython = \"/x\"\nfoo = 1\n").unwrap();
        strip_python_keys(&mut t);
        assert!(t.get("venv").is_none());
        assert!(t.get("python").is_none());
        assert!(t.get("foo").is_some());
    }

    #[test]
    fn drivers_config_missing_file_is_default() {
        let _g = ENV_LOCK.lock().unwrap();
        // Use a tempdir as $PIE_HOME so we can verify `load` returns
        // Default::default() when the file doesn't exist.
        let tmp = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("PIE_HOME", tmp.path()) };
        let cfg = DriversConfig::load().unwrap();
        assert!(cfg.python.venv.is_none());
        assert!(cfg.drivers.is_empty());
        unsafe { std::env::remove_var("PIE_HOME") };
    }

    #[test]
    fn drivers_config_round_trips() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("drivers.toml");
        std::fs::write(
            &path,
            r#"
[python]
venv = "/shared"

[driver.vllm]
venv = "/vllm"

[driver.sglang]
python = "/sgl/bin/python"
"#,
        )
        .unwrap();

        unsafe { std::env::set_var("PIE_HOME", tmp.path()) };
        let cfg = DriversConfig::load().unwrap();
        assert_eq!(cfg.python.venv.as_deref(), Some("/shared"));
        assert_eq!(
            cfg.drivers.get("vllm").unwrap().venv.as_deref(),
            Some("/vllm")
        );
        assert_eq!(
            cfg.drivers.get("sglang").unwrap().python.as_deref(),
            Some("/sgl/bin/python"),
        );
        unsafe { std::env::remove_var("PIE_HOME") };

        // Suppress unused-binding warning on `_ = path`.
        let _ = path;
    }
}
