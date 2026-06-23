//! Shared two-way connection glue for the gateway↔worker data plane.
//!
//! Both [`GatewayInbound`](crate::GatewayInbound) and
//! [`WorkerControl`](crate::WorkerControl) ride a SINGLE worker-initiated
//! connection: the worker dials in, the heavy token traffic flows on the plain
//! client→server direction (`push_tokens`), and the latency-sensitive commands
//! (`dispatch`/`cancel`/…) go reverse on the same socket. This module owns the
//! mux that splits one transport into two tarpc service channels.
//!
//! It lives here — defined ONCE — for the same reason this refactor deleted the
//! twin `WorkerSessionApi` defs: the [`TwoWayMessage`] enum's bincode variant
//! tags must be byte-identical on both ends, and the `(Req1, Resp1, Req2,
//! Resp2)` ordering threaded into [`spawn_twoway`] is the fragile bit — a
//! transposed pair is a silent wire break that still compiles. The two typed
//! constructors [`accept_gateway_link`] / [`connect_gateway_link`] fix that
//! ordering once, named, so each end is a single call:
//!
//! - gateway (`worker.rs`): `let (server_half, wc) = accept_gateway_link(t);`
//!   then `BaseChannel::with_defaults(server_half).execute(GwServer.serve())…`
//!   and hold `wc: WorkerControlClient` for `dispatch`/`cancel`.
//! - worker (dial-in): `let (server_half, gi) = connect_gateway_link(t);` then
//!   serve `WorkerControl` over `server_half` and hold `gi:
//!   GatewayInboundClient`, whose FIRST call must be `register(worker_id)`.
//!
//! NOTE: serve CONCRETELY per end (no generic serve helper) — tarpc 0.35's
//! `Channel::execute` requires `Req: RequestName`, which only the macro-generated
//! `*Request` types satisfy. The serde transport infers `TwoWayMessage` from the
//! wrapper bounds (no codec annotation); set the frame cap the usual way via
//! `config_mut().max_frame_length(N)` — small now that blobs ride HTTP.
//!
//! Proven end-to-end on tarpc 0.35 (both directions on one connection,
//! register-first) before landing.

use std::io;

use futures::{
    Sink, SinkExt, Stream, StreamExt, TryStreamExt,
    stream::{AbortHandle, Abortable},
};
use serde::{Deserialize, Serialize};
use tarpc::transport::channel::{ChannelError, UnboundedChannel};

use crate::{
    GatewayInboundClient, GatewayInboundRequest, GatewayInboundResponse, WorkerControlClient,
    WorkerControlRequest, WorkerControlResponse,
};

/// The muxed wire frame: every message on the one connection is either a request
/// for the locally-served service or a response to the locally-issued client.
/// Defined ONCE so its bincode variant tags (`Request` = 0, `Response` = 1) are
/// byte-identical on both ends.
#[derive(Serialize, Deserialize)]
pub enum TwoWayMessage<Req, Resp> {
    Request(tarpc::ClientMessage<Req>),
    Response(tarpc::Response<Resp>),
}

/// Error union for the two mux pump tasks: a closed in-proc channel half
/// ([`ChannelError`]) or a transport I/O error.
#[derive(thiserror::Error, Debug)]
pub enum ChannelOrIoError {
    #[error(transparent)]
    Channel(#[from] ChannelError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Split one bidirectional `transport` carrying [`TwoWayMessage`] into two tarpc
/// channels: a server-half (serve the local service, `Req1`/`Resp1`) and a
/// client-half (drive the remote service, `Req2`/`Resp2`). Spawns two pump tasks
/// (inbound demux + outbound merge); either failing aborts the other.
///
/// Prefer the typed [`accept_gateway_link`] / [`connect_gateway_link`] over
/// calling this directly — they pin the `(Req1, Resp1, Req2, Resp2)` ordering.
#[allow(clippy::type_complexity)]
pub fn spawn_twoway<Req1, Resp1, Req2, Resp2, T>(
    transport: T,
) -> (
    UnboundedChannel<tarpc::ClientMessage<Req1>, tarpc::Response<Resp1>>,
    UnboundedChannel<tarpc::Response<Resp2>, tarpc::ClientMessage<Req2>>,
)
where
    T: Stream<Item = Result<TwoWayMessage<Req1, Resp2>, io::Error>>,
    T: Sink<TwoWayMessage<Req2, Resp1>, Error = io::Error>,
    T: Unpin + Send + 'static,
    Req1: Send + 'static,
    Resp1: Send + 'static,
    Req2: Send + 'static,
    Resp2: Send + 'static,
{
    let (server, server_ret) = tarpc::transport::channel::unbounded();
    let (client, client_ret) = tarpc::transport::channel::unbounded();
    let (mut server_sink, server_stream) = server.split();
    let (mut client_sink, client_stream) = client.split();
    let (transport_sink, mut transport_stream) = transport.split();
    let (abort_handle, abort_registration) = AbortHandle::new_pair();

    tokio::spawn(async move {
        let res: Result<(), ChannelOrIoError> = async move {
            while let Some(msg) = transport_stream.next().await {
                match msg? {
                    TwoWayMessage::Request(req) => server_sink.send(req).await?,
                    TwoWayMessage::Response(rsp) => client_sink.send(rsp).await?,
                }
            }
            Ok(())
        }
        .await;
        if let Err(e) = res {
            tracing::warn!("inbound mux error: {e}");
        }
        abort_handle.abort();
    });

    let outbound = Abortable::new(
        futures::stream::select(
            server_stream.map_ok(TwoWayMessage::Response),
            client_stream.map_ok(TwoWayMessage::Request),
        )
        .map_err(ChannelOrIoError::Channel),
        abort_registration,
    );
    tokio::spawn(async move {
        let _ = outbound
            .forward(transport_sink.sink_map_err(ChannelOrIoError::Io))
            .await;
    });

    (server_ret, client_ret)
}

/// Gateway accept side (foxtrot's `worker.rs`): serve [`GatewayInbound`], call
/// [`WorkerControl`]. Returns the server-half to feed
/// `BaseChannel::with_defaults(..).execute(GwServer.serve())` and a ready
/// [`WorkerControlClient`] to dispatch turns to this worker.
///
/// [`GatewayInbound`]: crate::GatewayInbound
/// [`WorkerControl`]: crate::WorkerControl
#[allow(clippy::type_complexity)]
pub fn accept_gateway_link<T>(
    transport: T,
) -> (
    UnboundedChannel<
        tarpc::ClientMessage<GatewayInboundRequest>,
        tarpc::Response<GatewayInboundResponse>,
    >,
    WorkerControlClient,
)
where
    T: Stream<
        Item = Result<TwoWayMessage<GatewayInboundRequest, WorkerControlResponse>, io::Error>,
    >,
    T: Sink<TwoWayMessage<WorkerControlRequest, GatewayInboundResponse>, Error = io::Error>,
    T: Unpin + Send + 'static,
{
    let (server, client) = spawn_twoway::<
        GatewayInboundRequest,
        GatewayInboundResponse,
        WorkerControlRequest,
        WorkerControlResponse,
        _,
    >(transport);
    (
        server,
        WorkerControlClient::new(tarpc::client::Config::default(), client).spawn(),
    )
}

/// Worker dial side (golf's serve path): serve [`WorkerControl`], call
/// [`GatewayInbound`]. Returns the server-half to serve `WorkerControl` and a
/// ready [`GatewayInboundClient`] — its FIRST call must be `register(worker_id)`
/// (the register-first invariant that keys this worker into the gateway's
/// connected set).
///
/// [`GatewayInbound`]: crate::GatewayInbound
/// [`WorkerControl`]: crate::WorkerControl
#[allow(clippy::type_complexity)]
pub fn connect_gateway_link<T>(
    transport: T,
) -> (
    UnboundedChannel<
        tarpc::ClientMessage<WorkerControlRequest>,
        tarpc::Response<WorkerControlResponse>,
    >,
    GatewayInboundClient,
)
where
    T: Stream<
        Item = Result<TwoWayMessage<WorkerControlRequest, GatewayInboundResponse>, io::Error>,
    >,
    T: Sink<TwoWayMessage<GatewayInboundRequest, WorkerControlResponse>, Error = io::Error>,
    T: Unpin + Send + 'static,
{
    let (server, client) = spawn_twoway::<
        WorkerControlRequest,
        WorkerControlResponse,
        GatewayInboundRequest,
        GatewayInboundResponse,
        _,
    >(transport);
    (
        server,
        GatewayInboundClient::new(tarpc::client::Config::default(), client).spawn(),
    )
}
