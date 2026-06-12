//! pie:core/runtime - Runtime information and control functions

use crate::api::pie;
use crate::instance::InstanceState;
use crate::metadata_store;
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

    async fn max_output_tokens(&mut self) -> Result<u32> {
        Ok(model::min_output_token_ceiling())
    }

    async fn metadata_put(
        &mut self,
        namespace: String,
        key: String,
        value: Vec<u8>,
    ) -> Result<Result<(), String>> {
        let owner = self.metadata_owner();
        Ok(metadata_store::put(&owner, &namespace, &key, value).map_err(|e| e.to_string()))
    }

    async fn metadata_get(
        &mut self,
        namespace: String,
        key: String,
    ) -> Result<Result<Option<Vec<u8>>, String>> {
        let owner = self.metadata_owner();
        Ok(metadata_store::get(&owner, &namespace, &key).map_err(|e| e.to_string()))
    }

    async fn metadata_delete(
        &mut self,
        namespace: String,
        key: String,
    ) -> Result<Result<bool, String>> {
        let owner = self.metadata_owner();
        Ok(metadata_store::delete(&owner, &namespace, &key).map_err(|e| e.to_string()))
    }
}
