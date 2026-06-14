//! pie:core/blob-store - small daemon-lifetime named byte blobs.

use crate::api::pie;
use crate::instance::InstanceState;
use anyhow::Result;

impl pie::core::blob_store::Host for InstanceState {
    async fn save_blob(
        &mut self,
        name: String,
        bytes: Vec<u8>,
        ttl_ms: u64,
    ) -> Result<Result<(), String>> {
        let username = self.get_username();
        match crate::blob_store::save_blob(&username, &name, bytes, ttl_ms) {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn open_blob(&mut self, name: String) -> Result<Result<Option<Vec<u8>>, String>> {
        let username = self.get_username();
        match crate::blob_store::open_blob(&username, &name) {
            Ok(bytes) => Ok(Ok(bytes)),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn delete_blob(&mut self, name: String) -> Result<Result<(), String>> {
        let username = self.get_username();
        match crate::blob_store::delete_blob(&username, &name) {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(e.to_string())),
        }
    }
}
