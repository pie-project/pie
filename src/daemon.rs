//! What a long-running pie process does at start and stop: the role it boots
//! as, its config file, tracing and `/metrics`, signals and panics.

use std::future::Future;
use std::net::SocketAddr;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing_subscriber::EnvFilter;

use crate::args::{GlobalArgs, Origin, config_path_or};

/// Who a process boots as: its name, version, default config file and
/// `/metrics` address.
pub struct BootSpec {
    pub name: &'static str,
    pub version: &'static str,
    pub default_config_filename: &'static str,
    pub default_metrics_addr: Option<&'static str>,
}

impl BootSpec {
    fn new(name: &'static str) -> Self {
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

    fn default_config_filename(mut self, filename: &'static str) -> Self {
        self.default_config_filename = filename;
        self
    }

    fn default_metrics_addr(mut self, addr: &'static str) -> Self {
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
        wait_for_signal().await;
        tracing::info!("{}: shutdown signal received, draining", self.name);
        shutdown.await;
        tracing::info!("{}: stopped cleanly", self.name);
        ExitCode::SUCCESS
    }
}

fn init_observability(log_level: &str) {
    init_tracing(log_level);
    install_panic_hook();
    install_crypto_provider();
}

fn install_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// What a one-shot command (`pie config`, `pie doctor`, ...) sets up: tracing,
/// the panic hook and rustls.
pub fn init_command(global: &GlobalArgs) {
    init_observability(&global.log_level);
    runtime::catalog::install(&crate::paths::models_dir());
}

pub fn init(spec: BootSpec, global: GlobalArgs) -> Result<Ctx> {
    init_observability(&global.log_level);
    runtime::catalog::install(&crate::paths::models_dir());

    let config = read_config(&spec, &global)?;

    let metrics_addr: Option<SocketAddr> =
        match global.metrics_addr.as_deref().or(spec.default_metrics_addr) {
            Some(s) => Some(
                s.parse()
                    .with_context(|| format!("parsing metrics address {s:?}"))?,
            ),
            None => None,
        };

    if let Some(addr) = metrics_addr {
        spawn_metrics(addr, Instant::now(), spec.name, spec.version)?;
    }

    banner(spec.name, spec.version, metrics_addr);

    Ok(Ctx {
        config,
        name: spec.name,
    })
}

/// The role's config file as text; an absent default file is no config.
fn read_config(spec: &BootSpec, global: &GlobalArgs) -> Result<String> {
    let (path, origin) = config_path_or(global, spec.default_config_filename);
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !origin.is_explicit() => {
            tracing::debug!(path = %path.display(), "no config file; using role defaults");
            Ok(String::new())
        }
        Err(e) => Err(e).with_context(|| match origin {
            Origin::Flag => format!("reading --config {}", path.display()),
            Origin::Env => format!("reading $PIE_CONFIG {}", path.display()),
            Origin::Default => format!("reading {}", path.display()),
        }),
    }
}

fn init_tracing(log_level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn spawn_metrics(
    addr: SocketAddr,
    start: Instant,
    component: &'static str,
    version: &'static str,
) -> Result<()> {
    let std_listener =
        std::net::TcpListener::bind(addr).with_context(|| format!("bind /metrics on {addr}"))?;
    std_listener
        .set_nonblocking(true)
        .context("set /metrics listener non-blocking")?;
    let listener = TcpListener::from_std(std_listener).context("adopt /metrics listener")?;
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((sock, _)) => {
                    tokio::spawn(handle_scrape(sock, start, component, version));
                }
                Err(e) => tracing::warn!("/metrics accept error: {e}"),
            }
        }
    });
    Ok(())
}

async fn handle_scrape(mut sock: TcpStream, start: Instant, component: &str, version: &str) {
    let mut buf = [0u8; 1024];
    let n = sock.read(&mut buf).await.unwrap_or(0);
    let req = String::from_utf8_lossy(&buf[..n]);
    let resp = if req.starts_with("GET /metrics") {
        let body = render(start, component, version);
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/plain; version=0.0.4\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    } else {
        "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_string()
    };
    let _ = sock.write_all(resp.as_bytes()).await;
}

fn render(start: Instant, component: &str, version: &str) -> String {
    let uptime = start.elapsed().as_secs_f64();
    format!(
        "# HELP pie_build_info Build/identity info (always 1).\n\
         # TYPE pie_build_info gauge\n\
         pie_build_info{{component=\"{component}\",version=\"{version}\"}} 1\n\
         # HELP pie_uptime_seconds Seconds since process start.\n\
         # TYPE pie_uptime_seconds gauge\n\
         pie_uptime_seconds {uptime}\n"
    )
}

fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        tracing::error!("panic: {info}");
        default(info);
    }));
}

fn banner(name: &str, version: &str, metrics: Option<SocketAddr>) {
    eprintln!("pie-{name} {version}");
    if let Some(addr) = metrics {
        tracing::info!("/metrics serving on http://{addr}/metrics");
    }
}

async fn wait_for_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigint = match signal(SignalKind::interrupt()) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("install SIGINT handler: {e}");
                return;
            }
        };
        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("install SIGTERM handler: {e}");
                return;
            }
        };
        tokio::select! {
            _ = sigint.recv() => tracing::info!("received SIGINT"),
            _ = sigterm.recv() => tracing::info!("received SIGTERM"),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("received Ctrl-C");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_role_names_its_config_and_metrics() {
        assert_eq!(BootSpec::worker().name, "worker");
        assert_eq!(BootSpec::worker().default_config_filename, "worker.toml");
        assert!(BootSpec::worker().default_metrics_addr.is_some());
        assert_eq!(BootSpec::gateway().name, "gateway");
        assert_eq!(BootSpec::controller().name, "controller");
        assert_eq!(BootSpec::pie().name, "pie");
        assert!(BootSpec::pie().default_metrics_addr.is_none());
        assert_eq!(BootSpec::gateway().version("1.2.3").version, "1.2.3");
    }

    #[test]
    fn a_named_config_is_read_and_a_missing_one_refused() {
        let spec = BootSpec::worker();
        let path = std::env::temp_dir().join(format!("pie-daemon-{}.toml", std::process::id()));
        std::fs::write(&path, "key = 1\n").unwrap();
        let present = GlobalArgs {
            config: Some(path.to_string_lossy().into_owned()),
            log_level: "info".into(),
            metrics_addr: None,
        };
        assert_eq!(read_config(&spec, &present).unwrap(), "key = 1\n");
        std::fs::remove_file(&path).ok();

        let missing = GlobalArgs {
            config: Some("/nonexistent/pie-daemon-missing.toml".into()),
            log_level: "info".into(),
            metrics_addr: None,
        };
        assert!(read_config(&spec, &missing).is_err());
    }
}
