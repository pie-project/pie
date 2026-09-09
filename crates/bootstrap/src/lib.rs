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

#[derive(clap::Args, Clone, Debug)]
pub struct GlobalArgs {
    #[arg(short = 'c', long, value_name = "PATH")]
    pub config: Option<String>,
    #[arg(long, value_name = "LEVEL", default_value = "info")]
    pub log_level: String,
    #[arg(long, value_name = "ADDR")]
    pub metrics_addr: Option<String>,
}

pub struct BootSpec {
    pub name: &'static str,
    pub version: &'static str,
    pub default_config_filename: &'static str,
    pub default_metrics_addr: Option<&'static str>,
}

impl BootSpec {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            version: "0.0.0",
            default_config_filename: "config.toml",
            default_metrics_addr: None,
        }
    }

    pub fn version(mut self, version: &'static str) -> Self {
        self.version = version;
        self
    }

    pub fn default_config_filename(mut self, filename: &'static str) -> Self {
        self.default_config_filename = filename;
        self
    }

    pub fn default_metrics_addr(mut self, addr: &'static str) -> Self {
        self.default_metrics_addr = Some(addr);
        self
    }

    pub fn worker() -> Self {
        Self::new("worker")
            .default_config_filename("worker.toml")
            .default_metrics_addr("127.0.0.1:9100")
    }

    pub fn gateway() -> Self {
        Self::new("gateway")
            .default_config_filename("gateway.toml")
            .default_metrics_addr("127.0.0.1:9101")
    }

    pub fn controller() -> Self {
        Self::new("controller")
            .default_config_filename("controller.toml")
            .default_metrics_addr("127.0.0.1:9102")
    }

    pub fn pie() -> Self {
        Self::new("pie").default_config_filename("config.toml")
    }

    pub const ROLES: [&'static str; 4] = ["worker", "gateway", "controller", "pie"];

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

pub struct Ctx {
    config: String,
    name: &'static str,
}

impl Ctx {
    pub fn config_str(&self) -> &str {
        &self.config
    }

    pub async fn run_until_signal(self, shutdown: impl Future<Output = ()>) -> ExitCode {
        lifecycle::wait_for_signal().await;
        tracing::info!("{}: shutdown signal received, draining", self.name);
        shutdown.await;
        tracing::info!("{}: stopped cleanly", self.name);
        ExitCode::SUCCESS
    }
}

fn init_observability(log_level: &str) {
    observe::init_tracing(log_level);
    lifecycle::install_panic_hook();
    install_crypto_provider();
}

pub fn install_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

pub fn init_cli(global: &GlobalArgs) -> Result<()> {
    init_observability(&global.log_level);
    Ok(())
}

pub fn init(spec: BootSpec, global: GlobalArgs) -> Result<Ctx> {
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

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        global: GlobalArgs,
        #[arg(long)]
        listen: Option<String>,
    }

    #[test]
    fn lib_every_case() {
        per_role_identities();
        global_args_flatten_with_role_flag();
        config_source_reads_explicit_and_errors_on_missing();
    }

    fn per_role_identities() {
        assert_eq!(BootSpec::worker().name, "worker");
        assert_eq!(BootSpec::worker().default_config_filename, "worker.toml");
        assert!(BootSpec::worker().default_metrics_addr.is_some());
        assert_eq!(BootSpec::gateway().name, "gateway");
        assert_eq!(BootSpec::controller().name, "controller");
        assert_eq!(BootSpec::pie().name, "pie");
        assert!(BootSpec::pie().default_metrics_addr.is_none());
        assert_eq!(BootSpec::gateway().version("1.2.3").version, "1.2.3");
    }

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

        let missing = GlobalArgs {
            config: Some("/nonexistent/pie-env-missing.toml".into()),
            log_level: "info".into(),
            metrics_addr: None,
        };
        assert!(config::source(&spec, &missing).is_err());
    }
}
