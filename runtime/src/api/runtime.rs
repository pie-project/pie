//! pie:core/runtime - Runtime information and control functions

use crate::api::pie;
use crate::api::types::FutureString;
use crate::instance::InstanceState;
use crate::model;
use crate::runtime;

use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::core::runtime::Host for InstanceState {
    async fn version(&mut self) -> Result<String> {
        runtime::get_version().await
    }

    async fn instance_id(&mut self) -> Result<String> {
        Ok(self.id().to_string())
    }

    async fn username(&mut self) -> Result<String> {
        Ok(self.get_username())
    }

    async fn models(&mut self) -> Result<Vec<String>> {
        Ok(model::models())
    }

    async fn spawn(
        &mut self,
        package_name: String,
        args: Vec<String>,
    ) -> Result<Result<Resource<FutureString>, String>> {
        let rx = runtime::spawn_child_rx(package_name, args)?;
        let future_string = FutureString::new(rx);
        Ok(Ok(self.ctx().table.push(future_string)?))
    }
}
