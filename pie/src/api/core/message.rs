use crate::api::core::{Blob, BlobResult};
use crate::api::inferlet;
use crate::instance::InstanceState;
use crate::messaging::{PubSubCommand, PushPullCommand};
use crate::server;
use crate::service::ServiceCommand;
use async_trait::async_trait;
use std::mem;
use tokio::sync::{mpsc, oneshot};
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

#[derive(Debug)]
pub struct Subscription {
    id: usize,
    topic: String,
    receiver: mpsc::Receiver<String>,
    result: Option<String>,
    done: bool,
}

#[derive(Debug)]
pub struct ReceiveResult {
    receiver: oneshot::Receiver<String>,
    result: Option<String>,
    done: bool,
}

#[async_trait]
impl Pollable for ReceiveResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        let res = (&mut self.receiver).await.unwrap();
        self.result = Some(res);
        self.done = true;
    }
}

#[async_trait]
impl Pollable for Subscription {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Some(result) = self.receiver.recv().await {
            self.result = Some(result);
            self.done = true;
        } else {
            self.done = true;
        }
    }
}

impl inferlet::core::message::Host for InstanceState {
    async fn send(&mut self, message: String) -> anyhow::Result<()> {
        server::InstanceEvent::SendMsgToClient {
            inst_id: self.id(),
            message,
        }
        .dispatch();
        Ok(())
    }

    async fn receive(&mut self) -> anyhow::Result<Resource<ReceiveResult>> {
        let (tx, rx) = oneshot::channel();
        PushPullCommand::Pull {
            topic: self.id().to_string(),
            message: tx,
        }
        .dispatch();
        let res = ReceiveResult {
            receiver: rx,
            result: None,
            done: false,
        };
        Ok(self.ctx().table.push(res)?)
    }

    async fn send_blob(&mut self, blob: Resource<Blob>) -> anyhow::Result<()> {
        let data = mem::take(&mut self.ctx().table.get_mut(&blob)?.data);

        server::InstanceEvent::SendBlobToClient {
            inst_id: self.id(),
            data,
        }
        .dispatch();
        Ok(())
    }

    async fn receive_blob(&mut self) -> anyhow::Result<Resource<BlobResult>> {
        let (tx, rx) = oneshot::channel();
        PushPullCommand::PullBlob {
            topic: self.id().to_string(),
            message: tx,
        }
        .dispatch();
        let res = BlobResult {
            receiver: rx,
            result: None,
            done: false,
        };
        Ok(self.ctx().table.push(res)?)
    }

    async fn broadcast(&mut self, topic: String, message: String) -> anyhow::Result<()> {
        PubSubCommand::Publish { topic, message }.dispatch();
        Ok(())
    }

    async fn subscribe(&mut self, topic: String) -> anyhow::Result<Resource<Subscription>> {
        let (tx, rx) = mpsc::channel(64);
        let (sub_tx, sub_rx) = oneshot::channel();
        PubSubCommand::Subscribe {
            topic: topic.clone(),
            sender: tx,
            sub_id: sub_tx,
        }
        .dispatch();
        let sub_id = sub_rx.await?;
        let sub = Subscription {
            id: sub_id,
            topic,
            receiver: rx,
            result: None,
            done: false,
        };
        Ok(self.ctx().table.push(sub)?)
    }
}

impl inferlet::core::message::HostReceiveResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<ReceiveResult>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<ReceiveResult>) -> anyhow::Result<Option<String>> {
        Ok(self.ctx().table.get_mut(&this)?.result.clone())
    }

    async fn drop(&mut self, this: Resource<ReceiveResult>) -> anyhow::Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl inferlet::core::message::HostSubscription for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<Subscription>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<Subscription>) -> anyhow::Result<Option<String>> {
        Ok(mem::take(&mut self.ctx().table.get_mut(&this)?.result))
    }

    async fn unsubscribe(&mut self, this: Resource<Subscription>) -> anyhow::Result<()> {
        let sub = self.ctx().table.get_mut(&this)?;
        sub.done = true;
        let topic = sub.topic.clone();
        let sub_id = sub.id;
        PubSubCommand::Unsubscribe { topic, sub_id }.dispatch();
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Subscription>) -> anyhow::Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
