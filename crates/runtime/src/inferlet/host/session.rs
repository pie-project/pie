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
use anyhow::{Context, Result};
use wasmtime::component::{Accessor, HasSelf};

impl pie::inferlet::session::Host for ProcessCtx {
    async fn send(&mut self, message: String) -> Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let process_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(process_id).await
            && let Err(err) =
                server::send_event(client_id, process_id, &ProcessEvent::Message(message))
        {
            tracing::warn!(
                client_id,
                process_id = %process_id,
                error = %err,
                "session.send delivery failed"
            );
        }
        Ok(())
    }

    async fn send_file(&mut self, data: Vec<u8>) -> Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let process_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(process_id).await {
            server::send_file(client_id, process_id, data.into())?;
        }
        Ok(())
    }

    /// `receive` for a guest that cannot lower an `async func` (see
    /// `channel.take-blocking`): the guest's task is suspended in the call.
    async fn receive_blocking(&mut self) -> Result<Option<String>> {
        let process_id = self.id();
        crate::server::inbox::receive(process_id.to_string())
            .await
            .with_context(|| format!("session.receive-blocking failed for process {process_id}"))
            .map(Some)
    }

    /// `receive-file` for the same guests.
    async fn receive_file_blocking(&mut self) -> Result<Option<Vec<u8>>> {
        let process_id = self.id();
        let Some(client_id) = process::get_client_id(process_id).await.ok().flatten() else {
            return Ok(None);
        };
        match crate::server::receive_file(client_id, process_id).await {
            Ok(data) => Ok(Some(data.to_vec())),
            Err(error) => {
                tracing::warn!(
                    client_id,
                    process_id = %process_id,
                    %error,
                    "session.receive_file_blocking delivery failed"
                );
                Ok(None)
            }
        }
    }
}

impl pie::inferlet::session::HostWithStore<ProcessCtx> for HasSelf<ProcessCtx> {
    async fn receive(accessor: &Accessor<ProcessCtx, Self>) -> Result<Option<String>> {
        let process_id = accessor.with(|mut access| access.get().id());
        // A plain await: an idle receive holds no pooled state, so the
        // planner can evict around it freely — no standing safe point
        // needed. The residency gate re-engages on the next pooled call.
        crate::server::inbox::receive(process_id.to_string())
            .await
            .with_context(|| format!("session.receive failed for process {process_id}"))
            .map(Some)
    }

    async fn receive_file(accessor: &Accessor<ProcessCtx, Self>) -> Result<Option<Vec<u8>>> {
        let process_id = accessor.with(|mut access| access.get().id());
        let Some(client_id) = process::get_client_id(process_id).await.ok().flatten() else {
            return Ok(None);
        };
        match crate::server::receive_file(client_id, process_id).await {
            Ok(data) => Ok(Some(data.to_vec())),
            Err(error) => {
                tracing::warn!(
                    client_id,
                    process_id = %process_id,
                    %error,
                    "session.receive_file delivery failed"
                );
                Ok(None)
            }
        }
    }
}
