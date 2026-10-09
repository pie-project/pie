use std::fmt;
use std::net::SocketAddr;

use super::{BootSpec, GlobalArgs, config};
use crate::paths;

#[derive(Debug, Clone)]
pub struct Setting {
    pub value: String,
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

#[derive(Debug, Clone)]
pub enum ConfigStatus {
    Present { bytes: u64 },
    MissingDefault,
    MissingRequired,
    Unreadable(String),
}

#[derive(Debug, Clone)]
pub enum MetricsStatus {
    Disabled,
    Bindable(SocketAddr),
    Unparsable { raw: String, error: String },
    Unavailable { addr: SocketAddr, error: String },
}

#[derive(Debug, Clone)]
pub struct Resolved {
    pub role: &'static str,
    pub home: Setting,
    pub config_path: Setting,
    pub config_status: ConfigStatus,
    pub log_level: Setting,
    pub metrics: Option<Setting>,
    pub metrics_status: MetricsStatus,
}

impl Resolved {
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

pub fn resolve(spec: &BootSpec, global: &GlobalArgs) -> Resolved {
    let home = match std::env::var("PIE_HOME") {
        Ok(v) if !v.trim().is_empty() => Setting::new(v, "$PIE_HOME"),
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
