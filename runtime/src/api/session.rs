//! pie:core/session - User ↔ Process remote communication

use crate::api::pie;
use crate::api::types::FutureString;
use crate::linker::InstanceState;
use crate::messaging;
use crate::process;
use crate::server;
use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl pie::core::session::Host for InstanceState {
    async fn send(&mut self, message: String) -> Result<()> {
        let inst_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(inst_id).await {
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

    async fn send_file(&mut self, data: Vec<u8>) -> Result<()> {
        let process_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(process_id).await {
            server::send_file(client_id, process_id, data.into())?;
        }
        Ok(())
    }

    async fn receive_file(&mut self) -> Result<Resource<crate::api::types::FutureBlob>> {
        let (tx, rx) = oneshot::channel::<Vec<u8>>();
        let process_id = self.id();
        let client_id = process::get_client_id(process_id).await.ok().flatten();
        tokio::spawn(async move {
            if let Some(cid) = client_id {
                if let Ok(data) = server::receive_file(cid, process_id).await {
                    let _ = tx.send(data.to_vec());
                }
            }
        });
        let future_blob = crate::api::types::FutureBlob::new(rx);
        Ok(self.ctx().table.push(future_blob)?)
    }
}
