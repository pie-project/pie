//! Decode-worker lifecycle for controller-assigned executor partners.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use pie_controller_rpc::{NeighborPeer, Role};
use pie_driver_abi::{
    ExecutorRequest, ExecutorResponse, HelloRequest, ModelIdentity, REMOTE_WIRE_VERSION,
};
use pie_ids::WorkerId;

use crate::executor;

const PARTNER_RPC_DEADLINE: Duration = Duration::from_secs(10);

pub(crate) struct PartnerBootstrap {
    pub full_identity: ModelIdentity,
    pub encode_identity: ModelIdentity,
    pub kv_layout: pie_driver_abi::KvLayout,
    #[cfg_attr(not(feature = "nixl"), allow(dead_code))]
    pub home_kv_handle: pie_driver_abi::KvHandle,
    pub transfer: crate::config::OffloadTransfer,
    pub model_idx: usize,
    pub page_size: u32,
    pub request_timeout_secs: u64,
    pub max_outstanding: u32,
}

#[cfg(feature = "nixl")]
struct ClientNixl {
    _engine: std::sync::Arc<pie_transport::NixlEngine>,
    metadata: Vec<u8>,
}

struct PartnerLink {
    peer: NeighborPeer,
    driver_id: Option<usize>,
    client: pie_driver_abi::ExecutorRpcClient,
    disconnect: Option<pie_engine::driver::RemoteDisconnectHandle>,
    role: pie_engine::offload::PartnerRole,
    partner: std::sync::Arc<pie_engine::offload::Partner>,
}

pub(crate) struct PartnerLinkManager {
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
                        driver_id = ?link.driver_id,
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
        let mut context = tarpc::context::current();
        context.deadline = Instant::now() + PARTNER_RPC_DEADLINE;
        let healthy = matches!(
            link.client
                .execute(context, ExecutorRequest::LoadedModel)
                .await,
            Ok(Ok(ExecutorResponse::LoadedModel(true)))
        );
        if healthy {
            link.partner.mark_available();
        }
        healthy
    }

    async fn dial(&self, peer: NeighborPeer) -> Result<PartnerLink> {
        let role = match peer.role {
            Role::Prefill => pie_engine::offload::PartnerRole::Prefill,
            Role::Encode => pie_engine::offload::PartnerRole::Encode,
            Role::Decode => anyhow::bail!("decode peer is not an executor partner"),
        };
        let identity = match role {
            pie_engine::offload::PartnerRole::Prefill => self.config.full_identity.clone(),
            pie_engine::offload::PartnerRole::Encode => self.config.encode_identity.clone(),
        };
        let client = executor::connect(&peer.addr).await?;
        let peer_conn = {
            #[cfg(feature = "nixl")]
            {
                self.nixl
                    .as_ref()
                    .map(|nixl| pie_driver_abi::RemotePeerConn {
                        kind: pie_driver_abi::RemoteTransferKind::Nixl,
                        handle: Some(self.config.home_kv_handle.clone()),
                        metadata: nixl.metadata.clone(),
                    })
            }
            #[cfg(not(feature = "nixl"))]
            {
                None
            }
        };
        let mut context = tarpc::context::current();
        context.deadline = Instant::now() + PARTNER_RPC_DEADLINE;
        let response = client
            .execute(
                context,
                ExecutorRequest::Hello(HelloRequest {
                    wire_version: REMOTE_WIRE_VERSION,
                    client_nonce: self.worker_id.0,
                    model: identity.clone(),
                    kv_layout: self.config.kv_layout.clone(),
                    peer_conn,
                }),
            )
            .await
            .context("executor Hello transport")?
            .map_err(|error| anyhow!("executor Hello rejected: {error}"))?;
        let ExecutorResponse::Hello(hello) = response else {
            return Err(anyhow!("executor returned unexpected Hello response"));
        };
        ensure!(hello.wire_version == REMOTE_WIRE_VERSION, "wire mismatch");
        ensure!(hello.model == identity, "model identity mismatch");
        if role == pie_engine::offload::PartnerRole::Encode {
            ensure!(
                hello.capabilities.supports_media_encode,
                "encode partner does not advertise media encoding"
            );
            ensure!(
                hello.grant.num_pages == 0 && hello.peer_conn.handle.is_none(),
                "encode partner unexpectedly exposed a KV grant"
            );
            let partner = pie_engine::offload::register_partner(
                peer.id.0,
                self.worker_id.0,
                None::<usize>,
                role,
                self.config.max_outstanding,
                pie_driver_abi::RemoteTransferKind::Inline,
                Some(client.clone()),
            );
            return Ok(PartnerLink {
                peer,
                driver_id: None,
                client,
                disconnect: None,
                role,
                partner,
            });
        }
        ensure!(
            hello.kv_layout.compatible_with(&self.config.kv_layout),
            "KV layout mismatch"
        );
        let grant_end = hello
            .grant
            .end_page()
            .context("executor scratch grant overflows page id space")?;
        ensure!(
            hello.grant.num_pages > 0,
            "executor returned an empty scratch grant"
        );
        ensure!(
            grant_end <= hello.capabilities.total_pages,
            "executor scratch grant exceeds its advertised pool"
        );
        ensure!(
            hello.grant.num_pages <= 1_048_576,
            "executor scratch grant is unreasonably large"
        );
        if self.config.transfer == crate::config::OffloadTransfer::Nixl {
            ensure!(
                hello.peer_conn.kind == pie_driver_abi::RemoteTransferKind::Nixl,
                "executor did not accept required NIXL transfer"
            );
        }

        let remote = pie_engine::driver::RemoteDriver::new(
            client.clone(),
            tokio::runtime::Handle::current(),
            hello.capabilities.clone(),
            hello.grant,
        );
        let disconnect = remote.disconnect_handle();
        let driver_id = pie_engine::driver::register_driver_backend(
            pie_engine::driver::DriverSpec {
                num_kv_pages: hello.grant.num_pages as usize,
                limits: pie_engine::driver::SchedulerLimits {
                    max_forward_requests: hello.capabilities.max_forward_requests as usize,
                    max_forward_tokens: hello.capabilities.max_forward_tokens as usize,
                    max_page_refs: hello.capabilities.max_page_refs as usize,
                },
            },
            pie_engine::driver::DriverBackend::Remote(remote),
        );

        if let Err(error) = pie_engine::offload::register_remote_store(
            self.config.model_idx,
            driver_id,
            self.config.page_size,
            hello.grant.base_page,
            hello.grant.num_pages as usize,
        ) {
            disconnect.disconnect("remote store registration failed");
            let _ = pie_engine::driver::unregister_driver(driver_id);
            return Err(error).context("registering remote scratch store");
        }

        if let Err(error) = pie_engine::scheduler::spawn_driver(
            driver_id,
            self.config.page_size,
            self.config.request_timeout_secs,
        ) {
            disconnect.disconnect("remote scheduler registration failed");
            let _ = pie_engine::driver::unregister_driver(driver_id);
            return Err(error).context("spawning remote scheduler");
        }
        let partner = pie_engine::offload::register_partner(
            peer.id.0,
            self.worker_id.0,
            driver_id,
            role,
            self.config.max_outstanding,
            hello.peer_conn.kind,
            Some(client.clone()),
        );
        Ok(PartnerLink {
            peer,
            driver_id: Some(driver_id),
            client,
            disconnect: Some(disconnect),
            role,
            partner,
        })
    }

    async fn teardown(&mut self, worker_id: WorkerId, reason: &str) {
        let Some(link) = self.links.remove(&worker_id) else {
            return;
        };
        if let Some(disconnect) = &link.disconnect {
            disconnect.disconnect(reason.to_string());
        }
        pie_engine::offload::remove_partner(worker_id.0, link.role);
        link.partner.wait_drained().await;
        finish_cleanup(worker_id, link, self.config.model_idx);
    }
}

fn finish_cleanup(worker_id: WorkerId, link: PartnerLink, model_idx: usize) {
    let Some(driver_id) = link.driver_id else {
        return;
    };
    pie_engine::offload::close_driver_surrogates(driver_id);
    if let Err(error) = pie_engine::scheduler::stop_driver(driver_id) {
        tracing::warn!(
            partner = %worker_id,
            driver_id,
            %error,
            "stopping remote scheduler"
        );
    }
    if let Err(error) = pie_engine::driver::unregister_driver(driver_id) {
        tracing::warn!(
            partner = %worker_id,
            driver_id,
            %error,
            "unregistering remote driver"
        );
    }
    if let Err(error) = pie_engine::offload::unregister_remote_store(model_idx, driver_id) {
        tracing::warn!(
            partner = %worker_id,
            driver_id,
            %error,
            "unregistering remote store"
        );
    }
}

#[cfg(feature = "nixl")]
fn build_client_nixl(worker_id: WorkerId, config: &PartnerBootstrap) -> Result<Option<ClientNixl>> {
    use pie_transport::Engine;

    if config.transfer == crate::config::OffloadTransfer::Inline {
        return Ok(None);
    }
    let result = (|| {
        let engine = std::sync::Arc::new(pie_transport::NixlEngine::new(&format!(
            "pie-decode-{}-{}",
            worker_id.0,
            std::process::id()
        ))?);
        let _registered = engine.register(
            pie_transport::WorkerId(worker_id.0),
            config.home_kv_handle.clone(),
        )?;
        let metadata = engine.local_metadata()?;
        Ok::<_, pie_transport::TransportError>(ClientNixl {
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
            pie_engine::offload::remove_partner(worker_id.0, link.role);
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
