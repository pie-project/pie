//! What [`crate::init`] *would* resolve, computed without booting anything —
//! the model behind the `pie-env` command. Every value is produced by the
//! same code the daemons run, so the tool cannot report a path or port the
//! daemon would disagree with. Nothing here spawns a task, initialises
//! tracing, or holds a socket.

use std::fmt;
use std::net::SocketAddr;

use crate::{BootSpec, GlobalArgs, config, paths};

/// A resolved value plus where it came from, so a surprising value is
/// immediately traceable to the flag/env/default that produced it.
#[derive(Debug, Clone)]
pub struct Setting {
    /// The value the daemon would use.
    pub value: String,
    /// Provenance, e.g. `"--config flag"`, `"$PIE_HOME"`, `"role default"`.
    pub origin: &'static str,
}

impl Setting {
    fn new(value: impl Into<String>, origin: &'static str) -> Self {
        Self {
            value: value.into(),
            origin,
        }
    }
}

/// State of the config file at the resolved path.
#[derive(Debug, Clone)]
pub enum ConfigStatus {
    /// Readable. `bytes` is its size; `0` means the role lib gets an empty
    /// string and applies its own defaults.
    Present { bytes: u64 },
    /// Absent at the `$PIE_HOME` default location. Not an error — the role lib
    /// falls back to its defaults — but it is the usual cause of "my settings
    /// were ignored".
    MissingDefault,
    /// Explicitly requested via `--config`/`$PIE_CONFIG` but absent. Boot fails.
    MissingRequired,
    /// Present but unreadable (permissions, a directory, a broken symlink).
    Unreadable(String),
}

/// State of the `/metrics` listen address.
#[derive(Debug, Clone)]
pub enum MetricsStatus {
    /// No default for this role and no `--metrics-addr`: no endpoint.
    Disabled,
    /// Parses and binds — the daemon would come up.
    Bindable(SocketAddr),
    /// Not a `host:port` socket address. Boot fails.
    Unparsable { raw: String, error: String },
    /// Parses but cannot be bound, usually because something already owns the
    /// port (often another pie daemon). Boot fails.
    Unavailable { addr: SocketAddr, error: String },
}

/// The full picture for one role.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// Role name (`"worker"`, `"gateway"`, `"controller"`, `"pie"`).
    pub role: &'static str,
    /// `$PIE_HOME`, the root every other path hangs off.
    pub home: Setting,
    /// The config file this role would read.
    pub config_path: Setting,
    /// Whether that file is actually usable.
    pub config_status: ConfigStatus,
    /// The `tracing` filter that would apply.
    pub log_level: Setting,
    /// The `/metrics` address, if any.
    pub metrics: Option<Setting>,
    /// Whether that address would work.
    pub metrics_status: MetricsStatus,
}

impl Resolved {
    /// Reasons this role would fail to boot, in report order. Empty ⇒ the
    /// preconditions `init` checks are satisfied.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        match &self.config_status {
            ConfigStatus::MissingRequired => out.push(format!(
                "config file {} does not exist (requested via {})",
                self.config_path.value, self.config_path.origin
            )),
            ConfigStatus::Unreadable(e) => {
                out.push(format!("config file {}: {e}", self.config_path.value))
            }
            ConfigStatus::Present { .. } | ConfigStatus::MissingDefault => {}
        }
        match &self.metrics_status {
            MetricsStatus::Unparsable { raw, error } => out.push(format!(
                "metrics address {raw:?} is not host:port ({error})"
            )),
            MetricsStatus::Unavailable { addr, error } => {
                out.push(format!("cannot bind /metrics on {addr}: {error}"))
            }
            MetricsStatus::Disabled | MetricsStatus::Bindable(_) => {}
        }
        out
    }
}

/// Resolve everything [`crate::init`] would, for `spec` under `global`.
pub fn resolve(spec: &BootSpec, global: &GlobalArgs) -> Resolved {
    let home = match std::env::var("PIE_HOME") {
        Ok(v) if !v.trim().is_empty() => Setting::new(v, "$PIE_HOME"),
        // Mirrors `paths::pie_home`'s fallback chain; render what it produced
        // rather than guessing, so the two can't drift.
        _ => Setting::new(paths::pie_home().display().to_string(), "~/.pie default"),
    };

    let (config_path, config_origin) = config::resolve_path(spec, global);
    let config_status = match std::fs::metadata(&config_path) {
        Ok(m) if m.is_dir() => ConfigStatus::Unreadable("is a directory".into()),
        Ok(m) => ConfigStatus::Present { bytes: m.len() },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if config_origin.is_explicit() {
                ConfigStatus::MissingRequired
            } else {
                ConfigStatus::MissingDefault
            }
        }
        Err(e) => ConfigStatus::Unreadable(e.to_string()),
    };

    // `RUST_LOG` wins over `--log-level` (see `observe::init_tracing`), which is
    // the single most common reason a `--log-level` flag "does nothing".
    let log_level = match std::env::var("RUST_LOG") {
        Ok(v) if !v.trim().is_empty() => Setting::new(v, "$RUST_LOG (overrides --log-level)"),
        _ => Setting::new(&global.log_level, "--log-level"),
    };

    let metrics = match (global.metrics_addr.as_deref(), spec.default_metrics_addr) {
        (Some(a), _) => Some(Setting::new(a, "--metrics-addr")),
        (None, Some(a)) => Some(Setting::new(a, "role default")),
        (None, None) => None,
    };
    let metrics_status = match &metrics {
        None => MetricsStatus::Disabled,
        Some(s) => match s.value.parse::<SocketAddr>() {
            Err(e) => MetricsStatus::Unparsable {
                raw: s.value.clone(),
                error: e.to_string(),
            },
            // Probe exactly the way `observe::spawn_metrics` binds, then drop
            // the listener immediately — this must not disturb a running host.
            Ok(addr) => match std::net::TcpListener::bind(addr) {
                Ok(_) => MetricsStatus::Bindable(addr),
                Err(e) => MetricsStatus::Unavailable {
                    addr,
                    error: e.to_string(),
                },
            },
        },
    };

    Resolved {
        role: spec.name,
        home,
        config_path: Setting::new(config_path.display().to_string(), config_origin.describe()),
        config_status,
        log_level,
        metrics,
        metrics_status,
    }
}

impl fmt::Display for Resolved {
    /// The human report. Origin is always shown because "why is it this value"
    /// is the question this whole crate exists to answer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn row(f: &mut fmt::Formatter<'_>, label: &str, s: &Setting, note: &str) -> fmt::Result {
            let note = if note.is_empty() {
                String::new()
            } else {
                format!("  {note}")
            };
            writeln!(f, "  {label:<10} {}  [{}]{note}", s.value, s.origin)
        }

        writeln!(f, "{}", self.role)?;
        row(f, "PIE_HOME", &self.home, "")?;
        row(
            f,
            "config",
            &self.config_path,
            &match &self.config_status {
                ConfigStatus::Present { bytes: 0 } => "empty file; role defaults apply".into(),
                ConfigStatus::Present { bytes } => format!("{bytes} bytes"),
                ConfigStatus::MissingDefault => "not present; role defaults apply".into(),
                ConfigStatus::MissingRequired => "MISSING".into(),
                ConfigStatus::Unreadable(e) => format!("UNREADABLE: {e}"),
            },
        )?;
        row(f, "log level", &self.log_level, "")?;
        match &self.metrics {
            Some(s) => row(
                f,
                "metrics",
                s,
                match &self.metrics_status {
                    MetricsStatus::Bindable(_) => "port free",
                    MetricsStatus::Unavailable { .. } => "PORT IN USE",
                    MetricsStatus::Unparsable { .. } => "INVALID",
                    MetricsStatus::Disabled => "",
                },
            )?,
            None => row(f, "metrics", &Setting::new("(none)", "role default"), "")?,
        }

        let problems = self.problems();
        if problems.is_empty() {
            writeln!(f, "  ok: would boot")
        } else {
            problems
                .iter()
                .try_for_each(|p| writeln!(f, "  ERROR: {p}"))
        }
    }
}
