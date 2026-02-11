//! pie:core/messaging - Messaging interface for send/receive and pub/sub

use crate::api::pie;
use crate::api::types::FutureString;
use crate::linker::InstanceState;
use crate::messaging;
use crate::server;
use anyhow::Result;
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

impl pie::core::messaging::Host for InstanceState {
    async fn send(&mut self, message: String) -> Result<()> {
        let inst_id = self.id();
        if let Some(client_id) = server::get_client_id(inst_id) {
            server::send_event(client_id, inst_id, "message", message).ok();
        }
        Ok(())
    }

    async fn receive(&mut self) -> Result<Resource<FutureString>> {
        let (tx, rx) = oneshot::channel();
        let topic = self.id().to_string();
        tokio::spawn(async move {
            if let Ok(msg) = messaging::pull(topic).await {
                let _ = tx.send(msg);
            }
        });
        let future_string = FutureString::new(rx);
        Ok(self.ctx().table.push(future_string)?)
    }

    async fn broadcast(&mut self, topic: String, message: String) -> Result<()> {
        messaging::publish(topic, message)?;
        Ok(())
    }

    async fn subscribe(&mut self, topic: String) -> Result<Resource<Subscription>> {
        let (tx, rx) = mpsc::channel(64);
        let sub_id = messaging::subscribe(topic.clone(), tx).await?;
        let sub = Subscription {
            id: sub_id,
            topic,
            receiver: rx,
            result: None,
            done: false,
        };
        Ok(self.ctx().table.push(sub)?)
    }

    async fn transfer_file(&mut self, data: Vec<u8>) -> Result<()> {
        let topic = format!("blob:{}", self.id());
        messaging::push_blob(topic, data.into())?;
        Ok(())
    }

    async fn receive_file(&mut self) -> Result<Resource<crate::api::types::FutureBlob>> {
        let (tx, rx) = oneshot::channel::<Vec<u8>>();
        let topic = format!("blob:{}", self.id());
        tokio::spawn(async move {
            if let Ok(data) = messaging::pull_blob(topic).await {
                let _ = tx.send(data.to_vec());
            }
        });
        let future_blob = crate::api::types::FutureBlob::new(rx);
        Ok(self.ctx().table.push(future_blob)?)
    }
}

impl pie::core::messaging::HostSubscription for InstanceState {
    async fn pollable(&mut self, this: Resource<Subscription>) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(&mut self, this: Resource<Subscription>) -> Result<Option<String>> {
        Ok(mem::take(&mut self.ctx().table.get_mut(&this)?.result))
    }

    async fn unsubscribe(&mut self, this: Resource<Subscription>) -> Result<()> {
        let sub = self.ctx().table.get_mut(&this)?;
        sub.done = true;
        let topic = sub.topic.clone();
        let sub_id = sub.id;
        messaging::unsubscribe(topic, sub_id)?;
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Subscription>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
