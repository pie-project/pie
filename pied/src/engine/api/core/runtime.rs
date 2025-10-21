use crate::engine::api::core::{DebugQueryResult, Model};
use crate::engine::api::inferlet;
use crate::engine::instance::InstanceState;
use crate::engine::model;
use std::sync::Arc;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl inferlet::core::runtime::Host for InstanceState {
    async fn get_version(&mut self) -> anyhow::Result<String> {
        let (tx, rx) = oneshot::channel();
        crate::engine::runtime::Command::GetVersion { event: tx }.dispatch()?;
        rx.await.map_err(Into::into)
    }

    async fn get_instance_id(&mut self) -> anyhow::Result<String> {
        Ok(self.id().to_string())
    }

    async fn get_arguments(&mut self) -> anyhow::Result<Vec<String>> {
        Ok(self.arguments().to_vec())
    }

    async fn set_return(&mut self, value: String) -> anyhow::Result<()> {
        self.return_value = Some(value);
        Ok(())
    }

    async fn get_model(&mut self, name: String) -> anyhow::Result<Option<Resource<Model>>> {
        if let Some(service_id) = model::model_service_id(&name) {
            let (tx, rx) = oneshot::channel();
            model::Command::GetInfo { response: tx }.dispatch(service_id)?;
            let info = rx.await?;
            let model = Model {
                service_id,
                info: Arc::new(info),
            };
            let res = self.ctx().table.push(model)?;
            return Ok(Some(res));
        }
        Ok(None)
    }

    async fn get_all_models(&mut self) -> anyhow::Result<Vec<String>> {
        Ok(model::registered_models())
    }

    async fn get_all_models_with_traits(
        &mut self,
        _traits: Vec<String>,
    ) -> anyhow::Result<Vec<String>> {
        // Placeholder: Implement trait filtering logic
        Ok(model::registered_models())
    }

    async fn debug_query(&mut self, query: String) -> anyhow::Result<Resource<DebugQueryResult>> {
        let (tx, rx) = oneshot::channel();

        let res = DebugQueryResult {
            receiver: rx,
            result: None,
            done: false,
        };

        crate::engine::runtime::Command::DebugQuery { query, event: tx }.dispatch()?;
        Ok(self.ctx().table.push(res)?)
    }
}
