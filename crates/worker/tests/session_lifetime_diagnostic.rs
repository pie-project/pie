use std::collections::BTreeSet;
use std::fmt::Debug;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use controller_api::{
    Ack, GatewayId, GatewayInfo, Health, NodeId, Role, RoutableWorker, RoutingTable, WorkerStatus,
};
use gateway::GatewayControl;
use ids::{SessionId, WorkerId};
use tokio::sync::watch;
use worker::session_lifetime_diagnostic::{SessionObserver, SessionOwnerKeys, connect_gateway};

const FRESH_SESSIONS: usize = 8;
const WARM_TURNS: usize = 8;
const STEP_LIMIT: Duration = Duration::from_secs(8);

#[derive(Clone)]
struct ControlFixture {
    routing: watch::Receiver<RoutingTable>,
}

impl GatewayControl for ControlFixture {
    async fn register_gateway(&self, _: GatewayInfo) -> Result<GatewayId> {
        Ok(GatewayId(1))
    }

    async fn heartbeat(&self, _: NodeId) -> Result<Ack> {
        Ok(Ack::Ok)
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        self.routing.clone()
    }
}

fn routing(worker: WorkerId) -> RoutingTable {
    RoutingTable {
        epoch: 1,
        workers: vec![RoutableWorker {
            id: worker,
            addr: "diagnostic-only".into(),
            role: Role::Decode,
            model: "model-free-ping".into(),
            health: Health::Healthy,
            coarse_load: WorkerStatus {
                kv_pressure_bucket: 0,
                inflight: 0,
            },
        }],
    }
}

async fn wait_for<T: Debug>(
    label: &str,
    mut sample: impl AsyncFnMut() -> T,
    accept: impl Fn(&T) -> bool,
) -> Result<T> {
    let last = Arc::new(StdMutex::new(None));
    let observed = last.clone();
    match tokio::time::timeout(STEP_LIMIT, async {
        loop {
            let value = sample().await;
            *observed.lock().expect("last observation lock") = Some(format!("{value:#?}"));
            if accept(&value) {
                return value;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    {
        Ok(value) => Ok(value),
        Err(_) => {
            println!(
                "phase=timeout label={label:?} last={:?}",
                last.lock().expect("last observation lock")
            );
            anyhow::bail!("timeout waiting for {label}")
        }
    }
}

async fn step<T>(label: &str, future: impl Future<Output = Result<T>>) -> Result<T> {
    tokio::time::timeout(STEP_LIMIT, future)
        .await
        .with_context(|| format!("timeout during {label}"))?
        .with_context(|| label.to_string())
}

async fn census(observer: &SessionObserver) -> Result<SessionOwnerKeys> {
    observer
        .session_owner_keys()
        .await
        .context("worker session registry disappeared before shared-link teardown")
}

async fn bounded_census(observer: &SessionObserver, label: &str) -> Result<SessionOwnerKeys> {
    step(label, census(observer)).await
}

fn client_ids(keys: &SessionOwnerKeys) -> BTreeSet<u32> {
    keys.sessions.iter().map(|(_, client)| *client).collect()
}

fn print_phase(name: &str, gateway_live: usize, keys: &SessionOwnerKeys) {
    println!(
        "phase={name} gateway_live={gateway_live} worker_sessions={} active={} runtime_outboxes={} runtime_services={} runtime_service_tasks={} owner_keys={keys:#?}",
        keys.sessions.len(),
        keys.active.len(),
        keys.runtime.outboxes.len(),
        keys.runtime.services.len(),
        keys.runtime.service_tasks.len(),
    );
}

fn assert_owner_alignment(keys: &SessionOwnerKeys) -> Result<()> {
    let mapped = client_ids(keys);
    let outboxes = keys
        .runtime
        .outboxes
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let services = keys
        .runtime
        .services
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let service_tasks = keys
        .runtime
        .service_tasks
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    ensure!(
        mapped == outboxes,
        "worker mappings and SESSION_OUTBOX differ: {keys:?}"
    );
    ensure!(
        mapped == services,
        "worker mappings and CLIENT_SERVICES senders differ: {keys:?}"
    );
    ensure!(
        mapped == service_tasks,
        "worker mappings and CLIENT_SERVICES tasks differ: {keys:?}"
    );
    Ok(())
}

fn owners_align(keys: &SessionOwnerKeys) -> bool {
    let mapped = client_ids(keys);
    mapped == keys.runtime.outboxes.iter().copied().collect()
        && mapped == keys.runtime.services.iter().copied().collect()
        && mapped == keys.runtime.service_tasks.iter().copied().collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_ping_client_close_and_shared_link_lifetimes() -> Result<()> {
    runtime::server::init_session_lifetime_diagnostic(1024 * 1024);
    let baseline_runtime = runtime::server::session_owner_keys();
    ensure!(
        baseline_runtime.outboxes.is_empty(),
        "isolated process has runtime outboxes at baseline: {baseline_runtime:?}"
    );
    ensure!(
        baseline_runtime.services.is_empty(),
        "isolated process has runtime services at baseline: {baseline_runtime:?}"
    );
    ensure!(
        baseline_runtime.service_tasks.is_empty(),
        "isolated process has runtime service tasks at baseline: {baseline_runtime:?}"
    );

    let worker_id = WorkerId(41);
    let (_routing_tx, routing_rx) = watch::channel(routing(worker_id));
    let gateway = step(
        "gateway bind",
        gateway::bind(
            gateway::Config {
                listen: SocketAddr::from(([127, 0, 0, 1], 0)),
                worker_listen: SocketAddr::from(([127, 0, 0, 1], 0)),
                controller: "unused-by-control-fixture".into(),
            },
            ControlFixture {
                routing: routing_rx,
            },
        ),
    )
    .await?;
    let gateway_state = gateway.state.clone();
    let gateway_addr = gateway.listen_addr;
    let worker_addr = gateway.worker_addr;
    let gateway = gateway.into_handle();
    let link = match step(
        "worker gateway-link connect",
        connect_gateway(&worker_addr.to_string(), worker_id),
    )
    .await
    {
        Ok(link) => link,
        Err(error) => {
            let gateway_cleanup = step("gateway shutdown after link setup failure", async {
                gateway.shutdown().await;
                Ok(())
            })
            .await;
            println!(
                "status primary=1 cleanup=0 gateway_cleanup={}",
                if gateway_cleanup.is_ok() { 0 } else { 1 }
            );
            gateway_cleanup?;
            return Err(error);
        }
    };
    let observer = link.session_observer();
    let setup_result: Result<SessionOwnerKeys> = async {
        wait_for(
            "gateway worker registration",
            || async { gateway_state.workers.is_connected(worker_id) },
            |connected| *connected,
        )
        .await?;
        let baseline = bounded_census(&observer, "baseline census").await?;
        print_phase("before", gateway_state.sessions.live(), &baseline);
        ensure!(
            baseline.runtime == baseline_runtime,
            "worker/runtime baseline changed: {baseline:?}"
        );
        assert_owner_alignment(&baseline)?;
        Ok(baseline)
    }
    .await;

    let primary_result: Result<()> = match setup_result {
        Ok(baseline) => {
            async {
                let url = format!("ws://{gateway_addr}/v1/ws");
                let mut fresh_sessions = BTreeSet::new();
                let mut fresh_clients = BTreeSet::new();
                for index in 0..FRESH_SESSIONS {
                    let before_client = bounded_census(&observer, "before-client census").await?;
                    let client = step(
                        "fresh client connect",
                        client::client::Client::connect_with_identity(&url, "diagnostic/fresh"),
                    )
                    .await?;
                    let cancellation_probe = index + 1 == FRESH_SESSIONS;
                    if cancellation_probe {
                        ensure!(
                            observer.hold_next_terminal(),
                            "terminal hold was already armed"
                        );
                    }
                    step("fresh Client::ping", client.ping()).await?;
                    let during = wait_for(
                        "fresh Ping ownership",
                        || census(&observer),
                        |keys| {
                            keys.as_ref().is_ok_and(|keys| {
                                (keys.active.is_empty() || cancellation_probe)
                                    && keys.sessions.len() == before_client.sessions.len() + 1
                            })
                        },
                    )
                    .await??;
                    print_phase(
                        &format!("fresh-{}-pong", index + 1),
                        gateway_state.sessions.live(),
                        &during,
                    );
                    assert_owner_alignment(&during)?;
                    let new_mapping = during
                        .sessions
                        .iter()
                        .filter(|pair| !before_client.sessions.contains(pair))
                        .copied()
                        .collect::<Vec<_>>();
                    ensure!(
                        new_mapping.len() == 1,
                        "fresh client did not add exactly one mapping: {during:?}"
                    );
                    fresh_sessions.insert(new_mapping[0].0);
                    fresh_clients.insert(new_mapping[0].1);

                    if cancellation_probe {
                        let held_req = step("held terminal checkpoint", async {
                            observer
                                .held_terminal()
                                .await
                                .context("session registry lost while terminal held")
                        })
                        .await?;
                        let held = bounded_census(&observer, "held-turn census").await?;
                        print_phase(
                            "fresh-8-held-before-client-close",
                            gateway_state.sessions.live(),
                            &held,
                        );
                        ensure!(
                            held.active.contains(&(held_req, new_mapping[0].0)),
                            "held turn missing from real active registry: {held:?}"
                        );
                    }

                    step("fresh Client::close", client.close()).await?;
                    wait_for(
                        "gateway SessionHandle drop after Client::close completion",
                        || async { gateway_state.sessions.live() },
                        |live| *live == 0,
                    )
                    .await?;
                    let after_close = wait_for(
                        "stable ownership after client close",
                        || census(&observer),
                        |keys| {
                            keys.as_ref().is_ok_and(|keys| {
                                keys.active.is_empty()
                                    && owners_align(keys)
                                    && keys.sessions == before_client.sessions
                            })
                        },
                    )
                    .await??;
                    print_phase(
                        &format!("fresh-{}-client-close", index + 1),
                        gateway_state.sessions.live(),
                        &after_close,
                    );
                    assert_owner_alignment(&after_close)?;
                }
                ensure!(
                    fresh_sessions.len() == FRESH_SESSIONS,
                    "fresh clients reused SessionIds"
                );
                ensure!(
                    fresh_clients.len() == FRESH_SESSIONS,
                    "fresh sessions reused ClientIds"
                );

                let after_fresh = bounded_census(&observer, "after-fresh census").await?;
                print_phase(
                    "after-fresh-client-closes",
                    gateway_state.sessions.live(),
                    &after_fresh,
                );
                ensure!(
                    after_fresh == baseline,
                    "fresh clients retained ownership after close: {after_fresh:?}"
                );
                println!("classification=fresh-clean");

                let warm_client = step(
                    "warm client connect",
                    client::client::Client::connect_with_identity(&url, "diagnostic/warm"),
                )
                .await?;
                let warm_baseline = bounded_census(&observer, "warm baseline census").await?;
                let mut warm_mapping = None;
                for turn in 0..WARM_TURNS {
                    step("warm Client::ping", warm_client.ping()).await?;
                    let during = wait_for(
                        "warm Ping ownership",
                        || census(&observer),
                        |keys| {
                            keys.as_ref().is_ok_and(|keys| {
                                keys.active.is_empty()
                                    && keys.sessions.len() == warm_baseline.sessions.len() + 1
                            })
                        },
                    )
                    .await??;
                    print_phase(
                        &format!("warm-pong-{}", turn + 1),
                        gateway_state.sessions.live(),
                        &during,
                    );
                    assert_owner_alignment(&during)?;
                    let added = during
                        .sessions
                        .iter()
                        .filter(|pair| !warm_baseline.sessions.contains(pair))
                        .copied()
                        .collect::<Vec<_>>();
                    ensure!(
                        added.len() == 1,
                        "warm client has other than one mapping: {during:?}"
                    );
                    ensure!(
                        warm_mapping.is_none() || warm_mapping == Some(added[0]),
                        "warm turns changed mapping: {during:?}"
                    );
                    warm_mapping = Some(added[0]);
                }

                step("warm Client::close", warm_client.close()).await?;
                wait_for(
                    "warm gateway SessionHandle drop",
                    || async { gateway_state.sessions.live() },
                    |live| *live == 0,
                )
                .await?;
                let after_warm_close = wait_for(
                    "stable warm ownership after client close",
                    || census(&observer),
                    |keys| {
                        keys.as_ref().is_ok_and(|keys| {
                            keys.active.is_empty()
                                && owners_align(keys)
                                && keys.sessions == warm_baseline.sessions
                        })
                    },
                )
                .await??;
                print_phase(
                    "warm-client-close",
                    gateway_state.sessions.live(),
                    &after_warm_close,
                );
                assert_owner_alignment(&after_warm_close)?;
                ensure!(
                    !after_warm_close
                        .sessions
                        .contains(&warm_mapping.expect("eight pings establish a mapping")),
                    "warm client retained ownership after close: {after_warm_close:?}"
                );
                println!("classification=warm-clean");
                Ok(())
            }
            .await
        }
        Err(error) => Err(error),
    };

    drop(link);
    let cleanup_result: Result<()> = async {
        let after_link = wait_for(
            "shared gateway-worker link teardown",
            || async { observer.session_owner_keys().await },
            |keys| keys.is_none(),
        )
        .await?;
        println!("phase=shared-link-teardown worker_registry={after_link:?}");
        let final_runtime = wait_for(
            "runtime owner baseline after link teardown",
            || async { runtime::server::session_owner_keys() },
            |keys| *keys == baseline_runtime,
        )
        .await?;
        println!("phase=after-cleanup runtime_owner_keys={final_runtime:#?}");
        Ok(())
    }
    .await;
    let gateway_cleanup = step("gateway shutdown", async {
        gateway.shutdown().await;
        Ok(())
    })
    .await;
    println!(
        "status primary={} cleanup={} gateway_cleanup={}",
        if primary_result.is_ok() { 0 } else { 1 },
        if cleanup_result.is_ok() { 0 } else { 1 },
        if gateway_cleanup.is_ok() { 0 } else { 1 },
    );
    primary_result?;
    cleanup_result?;
    gateway_cleanup
}
