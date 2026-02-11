//! pie:core/runtime - Runtime information and control functions

use crate::api::pie;
use crate::api::types::FutureString;
use crate::linker::InstanceState;
use crate::model;
use crate::process;
use crate::program::ProgramName;

use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::core::runtime::Host for InstanceState {
    async fn version(&mut self) -> Result<String> {
        Ok(env!("CARGO_PKG_VERSION").to_string())
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
        let (tx, rx) = tokio::sync::oneshot::channel();
        match process::spawn(
            self.get_username(),
            ProgramName::parse(&package_name),
            args,
            None,
            Some(self.id()),
            false,
            Some(tx),
        ) {
            Ok(_process_id) => {
                let future_string = FutureString::new(rx);
                Ok(Ok(self.ctx().table.push(future_string)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }
}
