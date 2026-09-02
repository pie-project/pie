//! Mux glue for the gateway<->worker data plane: splits one worker-initiated
//! connection into two tarpc service channels (`GatewayInbound` served one
//! way, `WorkerControl` the other). Use [`accept_gateway_link`] /
//! [`connect_gateway_link`] rather than [`spawn_twoway`] directly.

use std::io;

use futures::{
    Sink, SinkExt, Stream, StreamExt, TryStreamExt,
    stream::{AbortHandle, Abortable},
};
use serde::{Deserialize, Serialize};
use tarpc::transport::channel::{ChannelError, UnboundedChannel};
use tokio_serde::formats::MessagePack;

use crate::{
    GatewayInboundClient, GatewayInboundRequest, GatewayInboundResponse, WorkerControlClient,
    WorkerControlRequest, WorkerControlResponse,
};

/// The muxed wire frame: either a request for the locally-served service or a
/// response to the locally-issued client. Variant tags must stay byte-identical
/// on both ends.
#[derive(Serialize, Deserialize)]
pub enum TwoWayMessage<Req, Resp> {
    Request(tarpc::ClientMessage<Req>),
    Response(tarpc::Response<Resp>),
}

/// Shared gateway<->worker data-plane codec, passed by both ends as their
/// `codec_fn` so it can't diverge. MessagePack because the internally-tagged
/// `ClientMessage`/`ServerMessage` enums need a self-describing codec (bincode
/// can't decode `#[serde(tag)]`), and it matches the client WS wire format.
pub fn dispatch_codec<Item, SinkItem>() -> MessagePack<Item, SinkItem> {
    MessagePack::default()
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
/// Prefer [`accept_gateway_link`] / [`connect_gateway_link`] over calling this
/// directly.
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
        // Sinks are dropped (not moved into the loop) so closing them stays a
        // deliberate, graceful signal to the local halves.
        let res: Result<(), ChannelOrIoError> = async {
            while let Some(msg) = transport_stream.next().await {
                match msg? {
                    TwoWayMessage::Request(req) => server_sink.send(req).await?,
                    TwoWayMessage::Response(rsp) => client_sink.send(rsp).await?,
                }
            }
            Ok(())
        }
        .await;
        // Dropping these ends the local halves' inbound streams.
        drop(server_sink);
        drop(client_sink);
        match res {
            // Clean EOF (peer hung up): let the outbound pump end on its own
            // rather than aborting, so in-flight responses still flush.
            Ok(()) => {}
            // Local end closed first (shutdown/link drop). No abort needed:
            // the dropped sinks above already end the outbound streams.
            Err(ChannelOrIoError::Channel(e)) => {
                tracing::debug!("mux half closed, tearing the link down: {e}");
            }
            // Transport itself failed; stop the outbound pump rather than
            // write into a dead socket.
            Err(ChannelOrIoError::Io(e)) => {
                tracing::warn!("inbound mux error: {e}");
                abort_handle.abort();
            }
        }
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

/// Gateway accept side: serve [`GatewayInbound`], call [`WorkerControl`].
/// Returns the server-half to feed
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

/// Worker dial side: serve [`WorkerControl`], call [`GatewayInbound`]. Returns
/// the server-half to serve `WorkerControl` and a ready [`GatewayInboundClient`]
/// — its first call must be `register(worker_id)`.
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
