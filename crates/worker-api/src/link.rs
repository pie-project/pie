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

#[derive(Serialize, Deserialize)]
pub enum TwoWayMessage<Req, Resp> {
    Request(tarpc::ClientMessage<Req>),
    Response(tarpc::Response<Resp>),
}

pub fn dispatch_codec<Item, SinkItem>() -> MessagePack<Item, SinkItem> {
    MessagePack::default()
}

#[derive(thiserror::Error, Debug)]
pub enum ChannelOrIoError {
    #[error(transparent)]
    Channel(#[from] ChannelError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

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
        drop(server_sink);
        drop(client_sink);
        match res {
            Ok(()) => tracing::debug!("the link's peer closed it"),
            Err(ChannelOrIoError::Channel(e)) => {
                tracing::debug!("mux half closed, tearing the link down: {e}");
            }
            Err(ChannelOrIoError::Io(e)) => tracing::warn!("inbound mux error: {e}"),
        }
        // However the inbound side ends, the outbound goes too: it holds the
        // channels' other halves, and until they drop neither the server nor
        // the client here sees the link gone (a client kept anywhere, as a
        // registry keeps one, would keep a dead link looking alive).
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
