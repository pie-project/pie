//! M3 inversion smoke — the Phase-3 close-out regression guard.
//!
//! Proves the post-inversion topology end-to-end, in-proc, against the real
//! assembled `pie-gateway` crate:
//!
//! 1. **The worker dials INTO the gateway.** The gateway is the listening server
//!    (`worker_listen`, 1:N fan-in); a stub worker is the *client* that dials in
//!    (`connect_gateway_link`, `register` first) — never the pre-0.5.0
//!    gateway-dials-worker edge. A silent revert to that edge fails this test.
//! 2. **A token round-trips** `session → dispatch → push_tokens → back`. A turn
//!    driven through `Sessions::create` reaches the dialed-in worker over the
//!    reverse `WorkerControl` channel; the worker streams chunks back over
//!    `GatewayInbound::push_tokens`, and they surface on the session's `TokenRx`
//!    terminated by a clean `Tokens::Eos`.
//!
//! Pure in-proc: a stub `GatewayControl` injects a seeded `RoutingTable` (no live
//! controller), and the turn is driven on `gw.state.sessions` directly (no axum
//! ingress), so the test isolates the inversion + the dispatch/token pipe.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use futures::StreamExt;
use tokio::sync::watch;

use pie_client_api::{ClientMessage, ServerMessage};
use pie_controller_rpc::{
    Ack, GatewayInfo, Health, Role, RoutableWorker, RoutingTable, WorkerStatus,
};
use pie_gateway::session::{Affinity, Identity, TurnInput};
use pie_gateway::{Gateway, GatewayConfig, GatewayControl, bind};
use pie_ids::{GatewayId, NodeId, ReqId, TenantId, WorkerId};
use pie_worker_rpc::{Accepted, Control, Priority, Request, Tokens};
use pie_worker_rpc::{GatewayInboundClient, WorkerControl, connect_gateway_link, dispatch_codec};
use tarpc::serde_transport::tcp;
use tarpc::server::{BaseChannel, Channel};

/// How many `Tokens::Chunk`s the stub streams before the clean `Eos`.
const TOKENS_PER_TURN: usize = 3;

/// Max frame on the worker link — mirrors `worker/src/serve/gateway_link.rs` so
/// the stub dials in with the same transport setup the real worker uses.
const LINK_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

// ───────────────────────── stub control plane ─────────────────────────

/// A [`GatewayControl`] backend that yields a fixed, seeded [`RoutingTable`] —
/// the injection seam `bind` exposes, so the smoke selects a worker with no live
/// controller. Holds the watch sender (in an `Arc`, to stay `Clone`) to keep the
/// channel open for the gateway's lifetime.
#[derive(Clone)]
struct StubControl {
    routing: Arc<watch::Sender<RoutingTable>>,
}

impl StubControl {
    fn seeded(table: RoutingTable) -> Self {
        let (tx, _rx) = watch::channel(table);
        Self {
            routing: Arc::new(tx),
        }
    }
}

impl GatewayControl for StubControl {
    async fn register_gateway(&self, _info: GatewayInfo) -> Result<GatewayId> {
        Ok(GatewayId(1))
    }

    async fn heartbeat(&self, _id: NodeId) -> Result<Ack> {
        Ok(Ack::Ok)
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        self.routing.subscribe()
    }
}

// ───────────────────────── stub worker (dials in) ─────────────────────────

/// A minimal worker that serves [`WorkerControl`]: on `dispatch` it accepts and
/// streams `TOKENS_PER_TURN` canned chunks back over `push_tokens`, then a clean
/// `Eos`. No runtime — just the data-plane mechanics, so the smoke exercises the
/// real inversion + token pipe without an engine.
#[derive(Clone)]
struct StubWorker {
    worker_id: WorkerId,
    /// Push side back to the gateway that dispatched (the reverse direction).
    gateway: GatewayInboundClient,
}

impl WorkerControl for StubWorker {
    async fn dispatch(self, _: tarpc::context::Context, req: Request) -> Accepted {
        let gateway = self.gateway.clone();
        let req_id = req.req_id;
        tokio::spawn(async move {
            for i in 0..TOKENS_PER_TURN {
                let msg = ServerMessage::Response {
                    corr_id: 1,
                    ok: true,
                    result: format!("tok{i}"),
                };
                match gateway
                    .push_tokens(tarpc::context::current(), req_id, Tokens::Chunk(msg))
                    .await
                {
                    Ok(Control::Continue) => {}
                    // Gateway aborted (consumer gone) or the link died — stop.
                    _ => return,
                }
            }
            let _ = gateway
                .push_tokens(tarpc::context::current(), req_id, Tokens::Eos)
                .await;
        });
        Accepted::Ok {
            worker: self.worker_id,
        }
    }

    async fn cancel(self, _: tarpc::context::Context, _req_id: ReqId) {}
    async fn set_priority(self, _: tarpc::context::Context, _req_id: ReqId, _p: Priority) {}
    async fn drain(self, _: tarpc::context::Context) {}
}

/// Dial INTO the gateway's worker-facing listener (the M3 inversion), split the
/// connection, `register(worker_id)` FIRST, then serve `WorkerControl`. Returns
/// the serve task (kept alive so the connection stays open for the round-trip).
async fn spawn_stub_worker(
    worker_addr: SocketAddr,
    worker_id: WorkerId,
) -> Result<tokio::task::JoinHandle<()>> {
    let mut conn = tcp::connect(worker_addr, dispatch_codec);
    conn.config_mut().max_frame_length(LINK_MAX_FRAME_BYTES);
    let transport = conn.await?;
    let (server_half, gateway) = connect_gateway_link(transport);

    // Register FIRST — keys this worker into the gateway's connected set before
    // any dispatch can target it (the register-first invariant).
    gateway
        .register(tarpc::context::current(), worker_id)
        .await?;

    let server = StubWorker { worker_id, gateway };
    let task = tokio::spawn(
        BaseChannel::with_defaults(server_half)
            .execute(server.serve())
            .for_each_concurrent(None, |req| async move {
                tokio::spawn(req);
            }),
    );
    Ok(task)
}

// ───────────────────────── helpers ─────────────────────────

fn loopback() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

fn ident() -> Identity {
    Identity {
        tenant: TenantId("smoke".into()),
        user: "u".into(),
        client_ip: None,
        request_id: None,
    }
}

fn turn_input() -> TurnInput {
    TurnInput {
        message: ClientMessage::Ping { corr_id: 1 },
        blobs: Vec::new(),
        priority: Priority::Normal,
    }
}

/// A seeded routing table with one healthy worker holding full headroom (so
/// admission admits and selection picks it).
fn seeded_table(worker_id: WorkerId) -> RoutingTable {
    RoutingTable {
        epoch: 1,
        workers: vec![RoutableWorker {
            id: worker_id,
            addr: "127.0.0.1:0".to_string(),
            role: Role::Decode,
            model: "stub-model".to_string(),
            health: Health::Healthy,
            coarse_load: WorkerStatus {
                kv_pressure_bucket: 0,
                inflight: 0,
            },
        }],
    }
}

/// Wait until the dialed-in worker is reflected in the gateway's connected set
/// (the registry bump on `register`), so routing selects it.
async fn wait_for_connected(
    connected: &watch::Receiver<Arc<HashSet<WorkerId>>>,
    worker_id: WorkerId,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if connected.borrow().contains(&worker_id) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("worker {worker_id} never appeared in the gateway connected set");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// ───────────────────────── the smoke ─────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn worker_dials_in_and_token_round_trips() -> Result<()> {
    let worker_id = WorkerId(7);

    // Inject a seeded routing table (no live controller) and bind the assembled
    // gateway on ephemeral ports. `bind` binds BOTH sockets before returning and
    // starts the worker accept loop, so the dial-in below cannot race the bind.
    let control = StubControl::seeded(seeded_table(worker_id));
    let config = GatewayConfig {
        listen: loopback(),
        worker_listen: loopback(),
        controller: String::new(),
    };
    let gw: Gateway = bind(config, control).await?;

    // M3 INVERSION: the worker is the dialer; the gateway is the listener.
    let _worker = spawn_stub_worker(gw.worker_addr, worker_id).await?;
    wait_for_connected(&gw.state.workers.connected_watch(), worker_id).await?;

    // Drive a turn through the real one-path: create → admit → dispatch (over the
    // reverse channel to the dialed-in worker) → its push_tokens stream back.
    let (_handle, mut rx) = gw
        .state
        .sessions
        .create(ident(), turn_input(), Affinity::Ephemeral)
        .await?;

    // The token round-trip: every pushed chunk arrives in order, then a clean Eos.
    let mut results = Vec::new();
    let mut saw_eos = false;
    while let Some(tok) = rx.recv().await {
        match tok {
            Tokens::Chunk(ServerMessage::Response { result, .. }) => results.push(result),
            Tokens::Chunk(_) => {}
            Tokens::Eos => {
                saw_eos = true;
                break;
            }
        }
    }

    assert!(
        saw_eos,
        "turn must end with a clean Eos (the worker→gateway push stream completed)"
    );
    assert_eq!(
        results,
        vec!["tok0", "tok1", "tok2"],
        "all chunks the dialed-in worker pushed must round-trip to the session, in order"
    );
    // Clean end ⇒ the pipe closes after Eos (a bare None *without* a preceding
    // Eos would be an abort — the discriminator delta/ingress branch on).
    assert!(
        rx.recv().await.is_none(),
        "the per-turn pipe closes after Eos"
    );

    Ok(())
}
