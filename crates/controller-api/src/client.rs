//! The link a worker or a gateway keeps to the controller: dialed again when
//! it goes, so a controller restarted on its address is found again. Its
//! watches start over against the new one, and the next heartbeat hears it
//! does not know the caller, which registers again under its id.

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use tarpc::client::RpcError;
use tarpc::context::Context;
use tarpc::serde_transport::tcp;
#[cfg(unix)]
use tarpc::serde_transport::unix;
use tarpc::tokio_serde::formats::Bincode;
use tokio::sync::watch;

use crate::ControlClient;

/// Four beats inside the controller's default heartbeat timeout (8 s).
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

/// The transport deadline on a long-poll watch: far past the controller's
/// hang time (20 s), so only a lost controller trips it.
const WATCH_DEADLINE: Duration = Duration::from_secs(300);

/// The pause after a failed watch, so a down controller is not spun on.
const WATCH_RETRY_BACKOFF: Duration = Duration::from_secs(1);

/// A link to the controller that is dialed again when it goes.
#[derive(Clone)]
pub struct ControllerLink {
    addr: Arc<str>,
    /// The client, and which dial it came from.
    client: Arc<Mutex<(ControlClient, u64)>>,
    dialing: Arc<tokio::sync::Mutex<()>>,
}

impl ControllerLink {
    pub async fn dial(addr: &str) -> Result<Self> {
        let client = dial(addr).await?;
        Ok(Self {
            addr: addr.into(),
            client: Arc::new(Mutex::new((client, 0))),
            dialing: Arc::default(),
        })
    }

    /// Which dial the link is on: a new one has heard nothing of the
    /// caller's earlier reports.
    pub fn generation(&self) -> u64 {
        self.client().1
    }

    fn client(&self) -> (ControlClient, u64) {
        self.client.lock().unwrap().clone()
    }

    /// After `error` on the link from dial `seen`: if the link is gone,
    /// dials again, unless another caller already has.
    async fn heal(&self, error: &RpcError, seen: u64) {
        if !matches!(error, RpcError::Shutdown) {
            return;
        }
        let _dialing = self.dialing.lock().await;
        if self.client().1 != seen {
            return;
        }
        match dial(&self.addr).await {
            Ok(client) => {
                *self.client.lock().unwrap() = (client, seen + 1);
                tracing::info!(controller = %self.addr, "dialed the controller again");
            }
            Err(e) => tracing::debug!(error = %format!("{e:#}"), "the controller is not back yet"),
        }
    }

    /// One call, `what` naming it in its error; a link found gone is dialed
    /// again for the next.
    pub async fn call<T, Fut>(
        &self,
        what: &'static str,
        call: impl FnOnce(ControlClient, Context) -> Fut,
    ) -> Result<T>
    where
        Fut: Future<Output = Result<T, RpcError>>,
    {
        let (client, seen) = self.client();
        match call(client, tarpc::context::current()).await {
            Ok(value) => Ok(value),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).with_context(|| format!("{what} rpc"))
            }
        }
    }

    /// A long-poll watch, `what` naming it: `poll` answers with the view
    /// past epoch `since`, and the receiver holds the newest. A new
    /// controller counts its epochs afresh, so the watch starts over on it.
    pub fn watch<V, Fut>(
        &self,
        what: &'static str,
        initial: V,
        epoch: fn(&V) -> u64,
        poll: impl Fn(ControlClient, Context, u64) -> Fut + Send + 'static,
    ) -> watch::Receiver<V>
    where
        V: Send + Sync + 'static,
        Fut: Future<Output = Result<V, RpcError>> + Send,
    {
        let (tx, rx) = watch::channel(initial);
        let link = self.clone();
        tokio::spawn(async move {
            let mut since = 0u64;
            loop {
                let (client, seen) = link.client();
                let mut ctx = tarpc::context::current();
                ctx.deadline = Instant::now() + WATCH_DEADLINE;
                match poll(client, ctx, since).await {
                    Ok(view) => {
                        since = epoch(&view);
                        tracing::debug!(watch = what, epoch = since, "controller view updated");
                        if tx.send(view).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(watch = what, error = %e, "controller watch failed; retrying");
                        link.heal(&e, seen).await;
                        if link.client().1 != seen {
                            since = 0;
                        }
                        tokio::time::sleep(WATCH_RETRY_BACKOFF).await;
                    }
                }
            }
        });
        rx
    }
}

/// A client on the controller at `addr` (`tcp://host:port`, a bare
/// `host:port`, or `unix://path`).
async fn dial(addr: &str) -> Result<ControlClient> {
    let cfg = tarpc::client::Config::default();
    if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        #[cfg(unix)]
        {
            let conn = unix::connect(path, Bincode::default)
                .await
                .with_context(|| format!("dialing controller at {addr}"))?;
            Ok(ControlClient::new(cfg, conn).spawn())
        }
        #[cfg(not(unix))]
        {
            let _ = (path, cfg);
            anyhow::bail!(
                "{addr}: a `unix://` control address is distributed serving, which needs a unix-domain socket; this build is single-node and speaks `tcp://`"
            )
        }
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let conn = tcp::connect(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        let _ = conn.get_ref().set_nodelay(true);
        Ok(ControlClient::new(cfg, conn).spawn())
    }
}
