//! `pie-bootstrap` — the shared process skeleton for the pie bins.
//!
//! Every bin (`bin/{worker,gateway,controller,pie}`) is a thin shell that
//! composes `bootstrap` with one or more role libraries. `bootstrap` owns the
//! cross-cutting, domain-agnostic concerns so each role lib stays a pure library:
//!
//! - the shared CLI flags as [`GlobalArgs`] (a `clap::Args` the bin flattens into
//!   its own `Parser`, keeping one `--help` and typed role flags);
//! - path/dir resolution ([`paths`]);
//! - config *sourcing* — locate + read the config file into a **String**
//!   ([`Ctx::config_str`]); the role lib's `Config::parse(&str)` owns all domain
//!   parsing (bootstrap never sees a typed `Config`);
//! - observability — `tracing` init + a minimal Prometheus-text `/metrics`
//!   endpoint (ruling R2);
//! - lifecycle — signal/panic handling, the boot banner, and the unified
//!   wait-for-signal-then-drain loop ([`Ctx::run_until_signal`]).
//!
//! Dependency rule (Seam 2): `bootstrap` depends on **no role library**, and no
//! role library depends on `bootstrap`. It is **runtime-agnostic** — the *bin*
//! owns the tokio runtime (`#[tokio::main]`); bootstrap only `spawn`s onto the
//! ambient one and its async surface is `.await`ed by the bin. The shutdown seam
//! is a future, not a trait (ruling R1), so role `Handle`s never depend on
//! bootstrap.
//!
//! ## The bin shape (Seam 3) — identical across all four but the middle lines
//!
//! ```ignore
//! use clap::Parser;
//!
//! #[derive(Parser)]
//! #[command(version)]
//! struct Cli {
//!     #[command(flatten)]
//!     global: bootstrap::GlobalArgs,
//!     // role-specific flags, read directly off `cli` (typed):
//!     #[arg(long)]
//!     listen: Option<String>,
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<std::process::ExitCode> {
//!     let cli = Cli::parse();
//!     let ctx = bootstrap::init(
//!         bootstrap::BootSpec::controller()
//!             .version(env!("CARGO_PKG_VERSION")),
//!         cli.global,
//!     )?;
//!     let cfg = controller::Config::parse(ctx.config_str())?;
//!     let handle = controller::run(cfg).await?;            // async role run, awaited
//!     Ok(ctx.run_until_signal(async move { handle.shutdown().await }).await)
//! }
//! ```
//!
//! `init` must be called inside the runtime (from the `#[tokio::main]` body) — it
//! `spawn`s the `/metrics` task. `--version`/`--help` are handled by the bin's
//! own `clap::Parser`. The `?`→`ExitCode` plumbing is just
//! `-> anyhow::Result<ExitCode>` (both `Termination`), so no wrapper is needed.

mod config;
mod lifecycle;
mod observe;
pub mod paths;

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

    // ── Per-role conveniences ──────────────────────────────────────────────
    //
    // Just static identity (name + a per-role default config filename + a
    // default metrics port) — NO role-lib dependency. Each daemon gets its own
    // config filename because their `Config`s use `deny_unknown_fields` and so
    // can't share one top-level file (the multi-section standalone file is
    // `bin/pie`'s concern). Bins still chain `.version(env!("CARGO_PKG_VERSION"))`
    // (the bin's own version), and may override any field.

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
    /// The shutdown seam is a future, not a trait (R1). `.await` this from the
    /// bin's `#[tokio::main]` body — bootstrap owns no runtime.
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
}

/// **CLI invocation** init for one-shot ops subcommands (`pie model list`,
/// `doctor`, `config show`, …): tracing + the panic hook only — **no** banner,
/// config sourcing, or `/metrics` (that daemon ceremony is odd UX on a one-shot
/// op, and an op shouldn't error just because no config file exists). Pair with
/// [`paths`] for any path lookups. Daemon/serving paths use the full [`init`].
///
/// Runtime-free, so it's callable from a sync `main` too. Returns `Result<()>`
/// for symmetry with [`init`] and headroom for future fallible CLI setup
/// (today it cannot fail).
pub fn init_cli(global: &GlobalArgs) -> Result<()> {
    init_observability(&global.log_level);
    Ok(())
}

/// **Daemon boot** init: source the config string, init tracing, install the
/// panic hook, start `/metrics`, and print the banner. Call once near the top of
/// a bin's `#[tokio::main]` body, passing the bin's flattened [`GlobalArgs`].
///
/// Must run inside a tokio runtime (it `spawn`s the `/metrics` task). Returns an
/// error only for genuine startup failures (bad config path, unparsable /
/// unbindable `--metrics-addr`).
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
        let path = std::env::temp_dir().join(format!("pie-bootstrap-{}.toml", std::process::id()));
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
            config: Some("/nonexistent/pie-bootstrap-missing.toml".into()),
            log_level: "info".into(),
            metrics_addr: None,
        };
        assert!(config::source(&spec, &missing).is_err());
    }
}
