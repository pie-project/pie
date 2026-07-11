//! pie:core/messaging - Process-to-process messaging (push/pull and pub/sub).
//!
//! `pull` is a native `async func` (await the next message host-side);
//! `subscribe` returns a native `stream<string>` backed by a host
//! `StreamProducer` over the broadcast channel (drop the reader = unsubscribe).
//! Both need store access, so per the wasmtime-46 component-model-async model
//! they live on the `HostWithStore` trait taking an `Accessor`.

use crate::api::pie;
use crate::inferlet::ProcessCtx;
use crate::messaging;
use anyhow::Result;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use tokio::sync::mpsc;
use wasmtime::StoreContextMut;
use wasmtime::component::{
    Accessor, Destination, HasSelf, StreamProducer, StreamReader, StreamResult,
};

/// Host-side producer for a `subscribe` stream: pumps the broadcast channel
/// into the guest-readable stream. Dropping it (when the guest drops the
/// stream reader) unsubscribes from the topic.
struct BroadcastStream {
    receiver: mpsc::Receiver<String>,
    topic: String,
    sub_id: usize,
}

impl Drop for BroadcastStream {
    fn drop(&mut self) {
        let _ = messaging::unsubscribe(self.topic.clone(), self.sub_id);
    }
}

impl<D> StreamProducer<D> for BroadcastStream {
    type Item = String;
    type Buffer = Option<String>;

    fn poll_produce<'a>(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        _store: StoreContextMut<'a, D>,
        mut dst: Destination<'a, Self::Item, Self::Buffer>,
        _finish: bool,
    ) -> Poll<wasmtime::Result<StreamResult>> {
        let this = self.get_mut();
        match this.receiver.poll_recv(cx) {
            Poll::Ready(Some(msg)) => {
                dst.set_buffer(Some(msg));
                Poll::Ready(Ok(StreamResult::Completed))
            }
            Poll::Ready(None) => Poll::Ready(Ok(StreamResult::Dropped)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl pie::inferlet::messaging::Host for ProcessCtx {
    async fn push(&mut self, topic: String, message: String) -> Result<()> {
        let topic = format!("{}:{}", self.get_username(), topic);
        messaging::push(topic, message)?;
        Ok(())
    }

    async fn broadcast(&mut self, topic: String, message: String) -> Result<()> {
        let topic = format!("{}:{}", self.get_username(), topic);
        messaging::publish(topic, message)?;
        Ok(())
    }
}

impl pie::inferlet::messaging::HostWithStore<ProcessCtx> for HasSelf<ProcessCtx> {
    async fn pull(accessor: &Accessor<ProcessCtx, Self>, topic: String) -> Result<String> {
        let topic =
            accessor.with(|mut access| format!("{}:{}", access.get().get_username(), topic));
        messaging::pull(topic).await
    }

    async fn subscribe(
        accessor: &Accessor<ProcessCtx, Self>,
        topic: String,
    ) -> Result<StreamReader<String>> {
        let topic =
            accessor.with(|mut access| format!("{}:{}", access.get().get_username(), topic));
        let (tx, rx) = mpsc::channel(64);
        let sub_id = messaging::subscribe(topic.clone(), tx).await?;
        let producer = BroadcastStream {
            receiver: rx,
            topic,
            sub_id,
        };
        Ok(accessor.with(|mut access| StreamReader::new(&mut access, producer))?)
    }
}
