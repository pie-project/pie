//! pie:core/runtime - Runtime information and control functions

use crate::api::pie;
use crate::api::types::FutureString;
use crate::instance::InstanceState;
use crate::model;
use crate::service::ServiceCommand;
use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::core::runtime::Host for InstanceState {
    async fn version(&mut self) -> Result<String> {
        let (tx, rx) = oneshot::channel();
        crate::runtime::Command::GetVersion { event: tx }.dispatch();
        rx.await.map_err(Into::into)
    }

    async fn instance_id(&mut self) -> Result<String> {
        Ok(self.id().to_string())
    }

    async fn username(&mut self) -> Result<String> {
        // TODO: Implement username access from InstanceState
        Ok("unknown".to_string())
    }

    async fn models(&mut self) -> Result<Vec<String>> {
        Ok(model::registered_models())
    }

    async fn arguments(&mut self) -> Result<Vec<String>> {
        let args = InstanceState::arguments(self);
        Ok(args.to_vec())
    }

    async fn set_return(&mut self, value: String) -> Result<()> {
        self.return_value = Some(value);
        Ok(())
    }

    async fn spawn(
        &mut self,
        package_name: String,
        args: Vec<String>,
    ) -> Result<Resource<FutureString>> {
        let (tx, rx) = oneshot::channel();
        
        // Dispatch spawn command to runtime
        crate::runtime::Command::Spawn {
            package_name,
            args,
            result: tx,
        }
        .dispatch();

        let future_string = FutureString::new(rx);
        Ok(self.ctx().table.push(future_string)?)
    }
}
