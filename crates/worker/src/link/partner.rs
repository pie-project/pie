//! Decode-worker lifecycle for controller-assigned executor partners.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Result, anyhow};
use controller_api::{NeighborPeer, Role};
use ids::WorkerId;

use crate::executor::ModelIdentity;

use crate::executor;

// `palo B-remote`: every field below is read by the dial path, which refuses
// at its first line until the envelope exists (see `PartnerLinkManager::dial`
// and `crate::executor`'s header). They are kept because they are what the
// handshake states, and deleting them would make the next wave rediscover
// which numbers a partner link needs.
#[allow(
    dead_code,
    reason = "read by the dial handshake, which is palo B-remote"
)]
pub(crate) struct PartnerBootstrap {
    pub full_identity: ModelIdentity,
    pub encode_identity: ModelIdentity,
    pub kv_layout: engine_api::KvLayout,
    #[cfg_attr(not(feature = "nixl"), allow(dead_code))]
    pub home_kv_handle: engine_api::KvHandle,
    pub transfer: crate::config::OffloadTransfer,
    pub model_idx: usize,
    pub page_size: u32,
    pub request_timeout_secs: u64,
    pub max_outstanding: u32,
}

#[cfg(feature = "nixl")]
struct ClientNixl {
    _engine: std::sync::Arc<transport::NixlEngine>,
    metadata: Vec<u8>,
}

struct PartnerLink {
    peer: NeighborPeer,
    engine_id: Option<usize>,
    disconnect: Option<::runtime::engine::RemoteDisconnectHandle>,
    role: ::runtime::offload::PartnerRole,
    partner: std::sync::Arc<::runtime::offload::Partner>,
}

pub(crate) struct PartnerLinkManager {
    /// This worker's own id, which the handshake's client nonce is.
    /// See the note on [`PartnerBootstrap`].
    #[allow(dead_code, reason = "read by the dial handshake, which is palo B-remote")]
    worker_id: WorkerId,
    config: PartnerBootstrap,
    links: HashMap<WorkerId, PartnerLink>,
    #[cfg(feature = "nixl")]
    nixl: Option<ClientNixl>,
}

impl PartnerLinkManager {
    pub(crate) fn new(worker_id: WorkerId, config: PartnerBootstrap) -> Result<Self> {
        #[cfg(feature = "nixl")]
        let nixl = build_client_nixl(worker_id, &config)?;
        #[cfg(not(feature = "nixl"))]
        anyhow::ensure!(
            config.transfer != crate::config::OffloadTransfer::Nixl,
            "offload.transfer=nixl requires feature \"nixl\""
        );
        Ok(Self {
            worker_id,
            config,
            links: HashMap::new(),
            #[cfg(feature = "nixl")]
            nixl,
        })
    }

    pub(crate) async fn reconcile(&mut self, peers: &[NeighborPeer]) {
        let desired = peers
            .iter()
            .filter(|peer| matches!(peer.role, Role::Prefill | Role::Encode))
            .map(|peer| (peer.id, peer.clone()))
            .collect::<HashMap<_, _>>();

        let existing = self.links.keys().copied().collect::<Vec<_>>();
        for worker_id in existing {
            let keep = match (self.links.get(&worker_id), desired.get(&worker_id)) {
                (Some(link), Some(peer))
                    if link.peer.addr == peer.addr && link.peer.role == peer.role =>
                {
                    self.probe(link).await
                }
                _ => false,
            };
            if !keep {
                self.teardown(worker_id, "partner departed or failed health probe")
                    .await;
            }
        }

        for peer in desired.into_values() {
            if self.links.contains_key(&peer.id) {
                continue;
            }
            match self.dial(peer.clone()).await {
                Ok(link) => {
                    tracing::info!(
                        partner = %peer.id,
                        role = %peer.role,
                        engine_id = ?link.engine_id,
                        "executor partner connected"
                    );
                    self.links.insert(peer.id, link);
                }
                Err(error) => {
                    tracing::warn!(
                        partner = %peer.id,
                        role = %peer.role,
                        %error,
                        "executor partner connection failed; local fallback remains active"
                    );
                }
            }
        }
    }

    pub(crate) async fn shutdown(&mut self) {
        let workers = self.links.keys().copied().collect::<Vec<_>>();
        for worker_id in workers {
            self.teardown(worker_id, "partner manager shutdown").await;
        }
    }

    async fn probe(&self, link: &PartnerLink) -> bool {
        if link
            .disconnect
            .as_ref()
            .is_some_and(|disconnect| !disconnect.is_connected())
        {
            return false;
        }
        // palo B-remote: the liveness probe was one `LoadedModel` round trip.
        // The envelope must carry a cheap "are you still serving the model
        // you said you were" question, because a peer that answered a fire
        // wrongly is worse than one that answered nothing.
        let healthy = false;
        if healthy
            && link
                .disconnect
                .as_ref()
                .is_none_or(|disconnect| disconnect.is_connected())
        {
            link.partner.mark_available();
            return true;
        }
        false
    }

    /// Dial a controller-assigned executor partner.
    ///
    /// # `palo B-remote`
    ///
    /// What stood here was the whole client handshake: a tarpc connect, a
    /// `HelloRequest` carrying this worker's `REMOTE_WIRE_VERSION`, model
    /// identity, `KvLayout` and (under NIXL) its own registered KV handle; a
    /// `HelloResponse` whose scratch grant was range-checked against the
    /// peer's advertised pool; then `register_remote_store`,
    /// `register_engine_backend` with a `RemoteEngine` over the client, and
    /// `spawn_engine`. Every noun in it lived in `engine_api::remote`.
    ///
    /// It refuses at the top rather than part way through, because a
    /// half-dialled partner is a registered `EngineId` with no transport
    /// behind it, and the offload planner would then select it.
    ///
    /// The envelope's own requirements are listed in `crate::executor`'s
    /// header. What THIS half additionally needs: the grant's page range must
    /// be validated against the peer's pool before any page id is minted
    /// against it (`grant.end_page() <= capabilities.total_pages`, and a
    /// ceiling on `grant.num_pages`), because those ids go straight into a
    /// `KvCopy` this worker builds.
    ///
    /// # Errors
    ///
    /// Always, until the envelope exists.
    async fn dial(&self, peer: NeighborPeer) -> Result<PartnerLink> {
        let role = match peer.role {
            Role::Prefill => ::runtime::offload::PartnerRole::Prefill,
            Role::Encode => ::runtime::offload::PartnerRole::Encode,
            Role::Decode => anyhow::bail!("decode peer is not an executor partner"),
        };
        let identity = match role {
            ::runtime::offload::PartnerRole::Prefill => self.config.full_identity.clone(),
            ::runtime::offload::PartnerRole::Encode => self.config.encode_identity.clone(),
        };
        let _ = (identity, &self.config.kv_layout, self.config.transfer);
        executor::connect_with_local_ip(&peer.addr).await?;
        Err(anyhow!(
            "executor partner {} at {} cannot be dialled: palo B-remote",
            peer.id,
            peer.addr
        ))
    }

    async fn teardown(&mut self, worker_id: WorkerId, reason: &str) {
        let Some(link) = self.links.remove(&worker_id) else {
            return;
        };
        if let Some(disconnect) = &link.disconnect {
            disconnect.disconnect(reason.to_string());
        }
        ::runtime::offload::remove_partner(worker_id.0, link.role);
        let model_idx = self.config.model_idx;
        let cleanup = tokio::spawn(async move {
            link.partner.wait_drained().await;
            if let Err(error) =
                tokio::task::spawn_blocking(move || finish_cleanup(worker_id, link, model_idx))
                    .await
            {
                tracing::warn!(partner = %worker_id, %error, "remote partner cleanup task failed");
            }
        });
        if let Err(error) = cleanup.await {
            tracing::warn!(partner = %worker_id, %error, "remote partner cleanup join failed");
        }
    }
}

fn finish_cleanup(worker_id: WorkerId, link: PartnerLink, model_idx: usize) {
    let Some(engine_id) = link.engine_id else {
        return;
    };
    ::runtime::offload::close_engine_surrogates(engine_id);
    if let Err(error) = ::runtime::scheduler::stop_engine(engine_id) {
        tracing::warn!(
            partner = %worker_id,
            engine_id,
            %error,
            "stopping remote scheduler"
        );
    }
    if let Err(error) = ::runtime::offload::unregister_remote_store(model_idx, engine_id) {
        tracing::warn!(
            partner = %worker_id,
            engine_id,
            %error,
            "unregistering remote store"
        );
    }
    if let Err(error) = ::runtime::engine::unregister_engine(engine_id) {
        tracing::warn!(
            partner = %worker_id,
            engine_id,
            %error,
            "unregistering remote engine"
        );
    }
}

#[cfg(feature = "nixl")]
fn build_client_nixl(worker_id: WorkerId, config: &PartnerBootstrap) -> Result<Option<ClientNixl>> {
    use transport::Engine;

    if config.transfer == crate::config::OffloadTransfer::Inline {
        return Ok(None);
    }
    let result = (|| {
        let engine = std::sync::Arc::new(transport::NixlEngine::new(&format!(
            "pie-decode-{}-{}",
            worker_id.0,
            std::process::id()
        ))?);
        let _registered = engine.register(
            transport::WorkerId(worker_id.0),
            config.home_kv_handle.clone(),
        )?;
        let metadata = engine.local_metadata()?;
        Ok::<_, transport::TransportError>(ClientNixl {
            _engine: engine,
            metadata,
        })
    })();
    match (config.transfer, result) {
        (_, Ok(nixl)) => Ok(Some(nixl)),
        (crate::config::OffloadTransfer::Nixl, Err(error)) => {
            Err(anyhow!("initializing decode NIXL: {error}"))
        }
        (crate::config::OffloadTransfer::Auto, Err(error)) => {
            tracing::warn!(%error, "NIXL unavailable; decode worker using inline KV transfer");
            Ok(None)
        }
        (crate::config::OffloadTransfer::Inline, _) => unreachable!(),
    }
}

impl Drop for PartnerLinkManager {
    fn drop(&mut self) {
        let model_idx = self.config.model_idx;
        for (worker_id, link) in self.links.drain() {
            if let Some(disconnect) = &link.disconnect {
                disconnect.disconnect("partner manager dropped");
            }
            ::runtime::offload::remove_partner(worker_id.0, link.role);
            let _ = std::thread::Builder::new()
                .name(format!("pie-partner-cleanup-{}", worker_id.0))
                .spawn(move || {
                    while link.partner.outstanding() != 0 {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    finish_cleanup(worker_id, link, model_idx);
                });
        }
    }
}
