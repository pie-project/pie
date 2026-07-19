//! pie:core/session - User <-> Process remote communication.
//!
//! `receive`/`receive-file` are now native `async func`: the host awaits the
//! next message/file directly (no fabricated future resource). Per the
//! wasmtime-46 component-model-async model, async-func imports are generated on
//! the `HostWithStore` trait taking an `Accessor` rather than `&mut self`.

use crate::inferlet::host::pie;
use crate::inferlet::process;
use crate::inferlet::{ProcessCtx, ProcessEvent};
use crate::server;
use anyhow::Result;
use wasmtime::component::{Accessor, HasSelf};

impl pie::inferlet::session::Host for ProcessCtx {
    async fn send(&mut self, message: String) -> Result<()> {
        crate::inferlet::process::preemption::honor(self).await?;
        let process_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(process_id).await {
            if let Err(err) =
                server::send_event(client_id, process_id, &ProcessEvent::Message(message))
            {
                tracing::warn!(
                    client_id,
                    process_id = %process_id,
                    error = %err,
                    "session.send delivery failed"
                );
            }
        }
        Ok(())
    }

    async fn send_file(&mut self, data: Vec<u8>) -> Result<()> {
        crate::inferlet::process::preemption::honor(self).await?;
        let process_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(process_id).await {
            server::send_file(client_id, process_id, data.into())?;
        }
        Ok(())
    }
}

impl pie::inferlet::session::HostWithStore<ProcessCtx> for HasSelf<ProcessCtx> {
    async fn receive(accessor: &Accessor<ProcessCtx, Self>) -> Result<Option<String>> {
        let (process_id, residency) =
            accessor.with(|mut access| (access.get().id(), access.get().residency_handle()));
        crate::inferlet::process::preemption::receive_message(process_id, residency).await
    }

    async fn receive_file(accessor: &Accessor<ProcessCtx, Self>) -> Result<Option<Vec<u8>>> {
        let (process_id, residency) =
            accessor.with(|mut access| (access.get().id(), access.get().residency_handle()));
        crate::inferlet::process::preemption::receive_file(process_id, residency).await
    }
}
