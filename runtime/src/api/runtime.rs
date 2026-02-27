//! pie:core/runtime - Runtime information and control functions

use crate::api::pie;
use crate::linker::InstanceState;
use crate::model;

use anyhow::Result;

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
}
