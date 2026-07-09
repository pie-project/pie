//! pie:core/session - User <-> Process remote communication.
//!
//! `receive`/`receive-file` are now native `async func`: the host awaits the
//! next message/file directly (no fabricated future resource). Per the
//! wasmtime-46 component-model-async model, async-func imports are generated on
//! the `HostWithStore` trait taking an `Accessor` rather than `&mut self`.

use crate::api::pie;
use crate::instance::InstanceState;
use crate::messaging;
use crate::process::{self, ProcessEvent};
use crate::server;
use anyhow::Result;
use wasmtime::component::{Accessor, HasSelf};

impl pie::inferlet::session::Host for InstanceState {
    async fn send(&mut self, message: String) -> Result<()> {
        let inst_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(inst_id).await {
            server::send_event(client_id, inst_id, &ProcessEvent::Message(message)).ok();
        }
        Ok(())
    }

    async fn send_file(&mut self, data: Vec<u8>) -> Result<()> {
        let process_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(process_id).await {
            server::send_file(client_id, process_id, data.into())?;
        }
        Ok(())
    }
}

impl pie::inferlet::session::HostWithStore<InstanceState> for HasSelf<InstanceState> {
    async fn receive(accessor: &Accessor<InstanceState, Self>) -> Result<Option<String>> {
        let topic = accessor.with(|mut access| access.get().id().to_string());
        match messaging::pull(topic).await {
            Ok(msg) => Ok(Some(msg)),
            Err(_) => Ok(None),
        }
    }

    async fn receive_file(accessor: &Accessor<InstanceState, Self>) -> Result<Option<Vec<u8>>> {
        let process_id = accessor.with(|mut access| access.get().id());
        let client_id = process::get_client_id(process_id).await.ok().flatten();
        match client_id {
            Some(cid) => match server::receive_file(cid, process_id).await {
                Ok(data) => Ok(Some(data.to_vec())),
                Err(_) => Ok(None),
            },
            None => Ok(None),
        }
    }
}
