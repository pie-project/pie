//! Shared process skeleton for the pie bins: CLI flags, config sourcing,
//! observability, and lifecycle (signal/panic handling, boot banner, drain).
//! `bootstrap` depends on no role library; the `pie-env` binary reports what
//! the skeleton would resolve without booting a daemon.

mod config;
pub use config::{Origin, cli_config_path};
mod lifecycle;
mod observe;
pub mod paths;
pub mod report;

use std::future::Future;
use std::net::SocketAddr;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};

/// The cross-cutting CLI flags every bin shares. A bin flattens this into its own
/// `clap::Parser` (`#[command(flatten)] global: GlobalArgs`) and passes the
/// `global` field to [`init`]; role-specific flags live on the bin's own struct.
#[derive(clap::Args, Clone, Debug)]
pub struct GlobalArgs {
    /// Config file path (else `$PIE_CONFIG`, else `$PIE_HOME/<default>`).
    #[arg(short = 'c', long, value_name = "PATH")]
    pub config: Option<String>,
    /// Log level used when `RUST_LOG` is unset (error|warn|info|debug|trace).
    #[arg(long, value_name = "LEVEL", default_value = "info")]
    pub log_level: String,
    /// Serve Prometheus `/metrics` on this address (host:port); overrides the
    /// bin's default.
    #[arg(long, value_name = "ADDR")]
    pub metrics_addr: Option<String>,
}

/// Per-bin identity, handed to [`init`]. Built with the chaining setters; only
/// `name` is required.
pub struct BootSpec {
    /// Process/component name (`"worker"`, `"gateway"`, `"controller"`, `"pie"`)
    /// — used in the banner and the `pie_build_info` metric label.
    pub name: &'static str,
    /// Version string for the banner + `pie_build_info` (pass
    /// `env!("CARGO_PKG_VERSION")`).
    pub version: &'static str,
    /// Config filename looked up under `$PIE_HOME` when neither `--config` nor
    /// `$PIE_CONFIG` is set.
    pub default_config_filename: &'static str,
    /// Default `/metrics` listen address (overridable with `--metrics-addr`).
    /// `None` ⇒ no endpoint unless `--metrics-addr` is passed. Daemons set it;
    /// one-shot CLIs leave it `None`.
    pub default_metrics_addr: Option<&'static str>,
}

impl BootSpec {
    /// A spec for `name` with empty/`None` defaults (`config.toml`, no metrics).
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            version: "0.0.0",
            default_config_filename: "config.toml",
            default_metrics_addr: None,
        }
    }

    /// Set the version string (typically `env!("CARGO_PKG_VERSION")`).
    pub fn version(mut self, version: &'static str) -> Self {
        self.version = version;
        self
    }

    /// Set the `$PIE_HOME`-relative default config filename.
    pub fn default_config_filename(mut self, filename: &'static str) -> Self {
        self.default_config_filename = filename;
        self
    }

    /// Set the default `/metrics` listen address.
    pub fn default_metrics_addr(mut self, addr: &'static str) -> Self {
        self.default_metrics_addr = Some(addr);
        self
    }

    // Per-role conveniences: static identity only, no role-lib dependency.
    // Each daemon gets its own config filename since `deny_unknown_fields`
    // rules out sharing one top-level file.

    /// `bin/worker` daemon identity (`worker.toml`, metrics `127.0.0.1:9100`).
    pub fn worker() -> Self {
        Self::new("worker")
            .default_config_filename("worker.toml")
            .default_metrics_addr("127.0.0.1:9100")
    }

    /// `bin/gateway` daemon identity (`gateway.toml`, metrics `127.0.0.1:9101`).
    pub fn gateway() -> Self {
        Self::new("gateway")
            .default_config_filename("gateway.toml")
            .default_metrics_addr("127.0.0.1:9101")
    }

    /// `bin/controller` daemon identity (`controller.toml`, metrics `127.0.0.1:9102`).
    pub fn controller() -> Self {
        Self::new("controller")
            .default_config_filename("controller.toml")
            .default_metrics_addr("127.0.0.1:9102")
    }

    /// `bin/pie` standalone identity (`config.toml`, no metrics by default — the
    /// multi-call CLI opts in with `--metrics-addr` when it serves).
    pub fn pie() -> Self {
        Self::new("pie").default_config_filename("config.toml")
    }

    /// Every role name a spec exists for, in report order.
    pub const ROLES: [&'static str; 4] = ["worker", "gateway", "controller", "pie"];

    /// The spec for a role name from [`BootSpec::ROLES`]. Lets `bootstrap` (and
    /// anything else driven by a `--role` string) reach the exact same
    /// per-role defaults the daemons compile in, instead of restating them.
    pub fn for_role(role: &str) -> Option<Self> {
        match role {
            "worker" => Some(Self::worker()),
            "gateway" => Some(Self::gateway()),
            "controller" => Some(Self::controller()),
            "pie" => Some(Self::pie()),
            _ => None,
        }
    }
}

/// The initialised process context: the sourced config string and the component
/// name. Returned by [`init`]; consumed by [`Ctx::run_until_signal`].
pub struct Ctx {
    config: String,
    name: &'static str,
}

impl Ctx {
    /// The sourced config string — feed this to the role lib's `Config::parse`.
    pub fn config_str(&self) -> &str {
        &self.config
    }

    /// Block (async) until SIGINT/SIGTERM (Ctrl-C off Unix), then await the
    /// caller's `shutdown` future (typically
    /// `async move { handle.shutdown().await }`) and return an [`ExitCode`].
    ///
    /// The shutdown seam is a future, not a trait. `.await` this from the
    /// bin's `#[tokio::main]` body — the skeleton owns no runtime.
    pub async fn run_until_signal(self, shutdown: impl Future<Output = ()>) -> ExitCode {
        lifecycle::wait_for_signal().await;
        tracing::info!("{}: shutdown signal received, draining", self.name);
        shutdown.await;
        tracing::info!("{}: stopped cleanly", self.name);
        ExitCode::SUCCESS
    }
}

/// The lightweight setup both entry flavors share — tracing + the panic hook —
/// so they can't drift between the daemon and CLI paths.
fn init_observability(log_level: &str) {
    observe::init_tracing(log_level);
    lifecycle::install_panic_hook();
    install_crypto_provider();
}

/// Installs the process-wide TLS backend rustls needs (reqwest is built on
/// `rustls-no-provider`, so this must run before any HTTPS client is built).
/// Idempotent: an already-installed provider is not an error.
pub fn install_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Init for one-shot ops subcommands (`pie model list`, `doctor`, ...):
/// tracing and the panic hook only, no banner/config/metrics. Runtime-free.
/// Daemon/serving paths use the full [`init`].
pub fn init_cli(global: &GlobalArgs) -> Result<()> {
    init_observability(&global.log_level);
    Ok(())
}

/// Daemon boot init: sources the config string, inits tracing, installs the
/// panic hook, starts `/metrics`, and prints the banner. Call once near the
/// top of a bin's `#[tokio::main]` body; must run inside a tokio runtime.
pub fn init(spec: BootSpec, global: GlobalArgs) -> Result<Ctx> {
    // Same observability setup as the CLI flavor (single-sourced, can't drift).
    init_observability(&global.log_level);

    let config = config::source(&spec, &global)?;

    let metrics_addr: Option<SocketAddr> =
        match global.metrics_addr.as_deref().or(spec.default_metrics_addr) {
            Some(s) => Some(
                s.parse()
                    .with_context(|| format!("parsing metrics address {s:?}"))?,
            ),
            None => None,
        };

    if let Some(addr) = metrics_addr {
        observe::spawn_metrics(addr, Instant::now(), spec.name, spec.version)?;
    }

    lifecycle::banner(spec.name, spec.version, metrics_addr);

    Ok(Ctx {
        config,
        name: spec.name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Stand-in for a bin's CLI: the shared globals flattened in, plus a
    /// role-specific flag (mirrors `bin/controller`'s `--listen`).
    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        global: GlobalArgs,
        #[arg(long)]
        listen: Option<String>,
    }

    #[test]
    fn per_role_identities() {
        assert_eq!(BootSpec::worker().name, "worker");
        assert_eq!(BootSpec::worker().default_config_filename, "worker.toml");
        assert!(BootSpec::worker().default_metrics_addr.is_some());
        assert_eq!(BootSpec::gateway().name, "gateway");
        assert_eq!(BootSpec::controller().name, "controller");
        // `pie` is a CLI, not a daemon → no default metrics endpoint.
        assert_eq!(BootSpec::pie().name, "pie");
        assert!(BootSpec::pie().default_metrics_addr.is_none());
        // Builder overrides apply.
        assert_eq!(BootSpec::gateway().version("1.2.3").version, "1.2.3");
    }

    #[test]
    fn global_args_flatten_with_role_flag() {
        let cli = TestCli::try_parse_from(["bin", "--listen", "1.2.3.4:5"]).unwrap();
        assert_eq!(cli.global.log_level, "info"); // default
        assert_eq!(cli.global.config, None);
        assert_eq!(cli.listen.as_deref(), Some("1.2.3.4:5")); // role-specific flag

        let cli = TestCli::try_parse_from([
            "bin",
            "-c",
            "/tmp/x.toml",
            "--log-level",
            "debug",
            "--metrics-addr",
            "0.0.0.0:9",
        ])
        .unwrap();
        assert_eq!(cli.global.config.as_deref(), Some("/tmp/x.toml"));
        assert_eq!(cli.global.log_level, "debug");
        assert_eq!(cli.global.metrics_addr.as_deref(), Some("0.0.0.0:9"));
    }

    #[test]
    fn config_source_reads_explicit_and_errors_on_missing() {
        let spec = BootSpec::worker();
        let path = std::env::temp_dir().join(format!("pie-env-{}.toml", std::process::id()));
        std::fs::write(&path, "key = 1\n").unwrap();
        let present = GlobalArgs {
            config: Some(path.to_string_lossy().into_owned()),
            log_level: "info".into(),
            metrics_addr: None,
        };
        assert_eq!(config::source(&spec, &present).unwrap(), "key = 1\n");
        std::fs::remove_file(&path).ok();

        // An explicitly requested but missing file is an error (vs a missing
        // default, which yields an empty string for role defaults).
        let missing = GlobalArgs {
            config: Some("/nonexistent/pie-env-missing.toml".into()),
            log_level: "info".into(),
            metrics_addr: None,
        };
        assert!(config::source(&spec, &missing).is_err());
    }
}
