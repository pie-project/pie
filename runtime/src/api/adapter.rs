//! pie:core/adapter - Adapter resource for LoRA/fine-tuning weights

use crate::adapter::{self, AdapterId};
use crate::api::pie;
use crate::api::model::Model;
use crate::linker::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Adapter resource - represents a LoRA adapter managed by the AdapterService.
#[derive(Debug)]
pub struct Adapter {
    /// The adapter ID assigned by the AdapterService
    pub adapter_id: AdapterId,
    /// The model service index (for routing to the correct AdapterService)
    pub model_idx: usize,
}

impl pie::core::adapter::Host for InstanceState {}

impl pie::core::adapter::HostAdapter for InstanceState {
    async fn create(&mut self, model: Resource<Model>, name: String) -> Result<Result<Resource<Adapter>, String>> {
        let model = self.ctx().table.get(&model)?;
        let model_idx = model.model_id;

        match adapter::create(model_idx, name).await {
            Ok(adapter_id) => {
                let adapter = Adapter { adapter_id, model_idx };
                Ok(Ok(self.ctx().table.push(adapter)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn destroy(&mut self, this: Resource<Adapter>) -> Result<()> {
        let adapter = self.ctx().table.get(&this)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;

        let _ = adapter::destroy(model_idx, adapter_id).await;
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn open(&mut self, model: Resource<Model>, name: String) -> Result<Option<Resource<Adapter>>> {
        let model = self.ctx().table.get(&model)?;
        let model_idx = model.model_id;

        match adapter::open(model_idx, name).await {
            Some(adapter_id) => {
                let adapter = Adapter { adapter_id, model_idx };
                Ok(Some(self.ctx().table.push(adapter)?))
            }
            None => Ok(None),
        }
    }

    async fn fork(&mut self, this: Resource<Adapter>, name: String) -> Result<Resource<Adapter>> {
        let adapter = self.ctx().table.get(&this)?;
        let adapter_id = adapter.adapter_id;
        let model_idx = adapter.model_idx;

        match adapter::fork(model_idx, adapter_id, name).await {
            Some(new_adapter_id) => {
                let new_adapter = Adapter { adapter_id: new_adapter_id, model_idx };
                Ok(self.ctx().table.push(new_adapter)?)
            }
            None => anyhow::bail!("Failed to fork adapter"),
        }
    }

    async fn drop(&mut self, this: Resource<Adapter>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn load(
        &mut self,
        this: Resource<Adapter>,
        path: String,
    ) -> Result<Result<(), String>> {
        let adapter_res = self.ctx().table.get(&this)?;
        let adapter_id = adapter_res.adapter_id;
        let model_idx = adapter_res.model_idx;

        match adapter::load(model_idx, adapter_id, path).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn save(
        &mut self,
        this: Resource<Adapter>,
        path: String,
    ) -> Result<Result<(), String>> {
        let adapter_res = self.ctx().table.get(&this)?;
        let adapter_id = adapter_res.adapter_id;
        let model_idx = adapter_res.model_idx;

        match adapter::save(model_idx, adapter_id, path).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }
}
