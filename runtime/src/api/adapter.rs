//! pie:core/adapter - Adapter resource for LoRA/fine-tuning weights

use crate::adapter::{AdapterId, LockId, Message};
use crate::api::pie;
use crate::api::types::FutureBool;
use crate::instance::InstanceState;
use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Adapter resource - represents a LoRA adapter managed by the AdapterActor.
#[derive(Debug)]
pub struct Adapter {
    /// The adapter ID assigned by the AdapterActor
    pub adapter_id: AdapterId,
    /// The model service index (for routing to the correct AdapterActor)
    pub model_idx: usize,
    /// Currently held lock ID (if any)
    pub lock_id: Option<LockId>,
}

impl pie::core::adapter::Host for InstanceState {}

impl pie::core::adapter::HostAdapter for InstanceState {
    async fn create(&mut self, name: String) -> Result<Result<Resource<Adapter>, String>> {
        // TODO: Get model_idx from instance context or infer from current model
        let model_idx = 0usize; // Placeholder - should be obtained from instance state

        let (tx, rx) = oneshot::channel();
        Message::Create { name, response: tx }.send(model_idx)?;

        match rx.await? {
            Ok(adapter_id) => {
                let adapter = Adapter {
                    adapter_id,
                    model_idx,
                    lock_id: None,
                };
                Ok(Ok(self.ctx().table.push(adapter)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn destroy(&mut self, this: Resource<Adapter>) -> Result<Result<(), String>> {
        let adapter = self.ctx().table.get(&this)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;

        let (tx, rx) = oneshot::channel();
        Message::Destroy {
            id: adapter_id,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(()) => {
                self.ctx().table.delete(this)?;
                Ok(Ok(()))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn get(&mut self, name: String) -> Result<Option<Resource<Adapter>>> {
        // TODO: Get model_idx from instance context or infer from current model
        let model_idx = 0usize; // Placeholder - should be obtained from instance state

        let (tx, rx) = oneshot::channel();
        Message::Get { name, response: tx }.send(model_idx)?;

        match rx.await? {
            Some(adapter_id) => {
                let adapter = Adapter {
                    adapter_id,
                    model_idx,
                    lock_id: None,
                };
                Ok(Some(self.ctx().table.push(adapter)?))
            }
            None => Ok(None),
        }
    }

    async fn clone(&mut self, this: Resource<Adapter>, name: String) -> Result<Resource<Adapter>> {
        let adapter = self.ctx().table.get(&this)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;

        let (tx, rx) = oneshot::channel();
        Message::Clone {
            id: adapter_id,
            new_name: name,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Some(new_adapter_id) => {
                let new_adapter = Adapter {
                    adapter_id: new_adapter_id,
                    model_idx,
                    lock_id: None,
                };
                Ok(self.ctx().table.push(new_adapter)?)
            }
            None => anyhow::bail!("Failed to clone adapter"),
        }
    }

    async fn drop(&mut self, this: Resource<Adapter>) -> Result<()> {
        // Note: This doesn't destroy the adapter in the actor, just the local resource
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn lock(&mut self, this: Resource<Adapter>) -> Result<Resource<FutureBool>> {
        let adapter = self.ctx().table.get(&this)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;

        let (tx, rx) = oneshot::channel();
        Message::Lock {
            id: adapter_id,
            response: tx,
        }
        .send(model_idx)?;

        // Create a channel to transform the LockId response into a bool
        let (bool_tx, bool_rx) = oneshot::channel();

        tokio::spawn(async move {
            if let Ok(lock_id) = rx.await {
                // Lock succeeded if lock_id != 0
                let _ = bool_tx.send(lock_id != 0);
            }
        });

        let future_bool = FutureBool::new(bool_rx);
        Ok(self.ctx().table.push(future_bool)?)
    }

    async fn unlock(&mut self, this: Resource<Adapter>) -> Result<()> {
        let adapter = self.ctx().table.get_mut(&this)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;
        let lock_id = adapter.lock_id.take().unwrap_or(0);

        Message::Unlock {
            id: adapter_id,
            lock_id,
        }
        .send(model_idx)?;

        Ok(())
    }

    async fn load(
        &mut self,
        this: Resource<Adapter>,
        path: String,
    ) -> Result<Result<(), String>> {
        let adapter = self.ctx().table.get(&this)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;

        let (tx, rx) = oneshot::channel();
        Message::Load {
            id: adapter_id,
            path,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn save(
        &mut self,
        this: Resource<Adapter>,
        path: String,
    ) -> Result<Result<(), String>> {
        let adapter = self.ctx().table.get(&this)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;

        let (tx, rx) = oneshot::channel();
        Message::Save {
            id: adapter_id,
            path,
            response: tx,
        }
        .send(model_idx)?;

        match rx.await? {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }
}
