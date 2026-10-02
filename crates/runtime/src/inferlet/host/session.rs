use crate::inferlet::host::frames::{Frames, Pcm, audio_extension, image_extension};
use crate::inferlet::host::pie;
use crate::inferlet::host::pie::inferlet::frames::{AudioFormat, ImageFormat};
use crate::inferlet::process;
use crate::inferlet::{ProcessCtx, ProcessEvent};
use crate::server;
use anyhow::{Context, Result};
use wasmtime::component::{Accessor, HasSelf, Resource};
use wasmtime_wasi::WasiView;

/// Runs `wait` with the process marked idle: a process waiting on its client
/// is the one whose KV the planner may move off the device first.
async fn idle_while<T>(pid: crate::inferlet::ProcessId, wait: impl Future<Output = T>) -> T {
    let planner = crate::planner::planner();
    if let Some(planner) = planner {
        planner.set_idle(pid, true);
    }
    let out = wait.await;
    if let Some(planner) = planner {
        planner.set_idle(pid, false);
    }
    out
}

fn suggested_name(name: &str, extension: &str) -> String {
    let base = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('.');
    let base = if base.is_empty() { "output" } else { base };
    if std::path::Path::new(base)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(extension))
    {
        base.to_string()
    } else {
        format!("{base}.{extension}")
    }
}

fn bare_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or("").trim();
    let base = base.trim_matches('.');
    if base.is_empty() {
        "output".to_string()
    } else {
        base.to_string()
    }
}

async fn stream_out(ctx: &mut ProcessCtx, bytes: Vec<u8>, name: String) -> Result<()> {
    let process_id = ctx.id();
    if let Ok(Some(client_id)) = process::get_client_id(process_id).await {
        server::send_file(client_id, process_id, bytes.into(), Some(name))?;
    }
    Ok(())
}

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
            server::send_file(client_id, process_id, data.into(), None)?;
        }
        Ok(())
    }

    async fn send_file_as(&mut self, data: Vec<u8>, name: String) -> Result<()> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let process_id = self.id();
        if let Ok(Some(client_id)) = process::get_client_id(process_id).await {
            server::send_file(client_id, process_id, data.into(), Some(bare_name(&name)))?;
        }
        Ok(())
    }

    async fn send_frames(
        &mut self,
        f: Resource<Frames>,
        format: ImageFormat,
        name: String,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let encoded = match self.ctx().table.get(&f)?.encode(format) {
            Ok(bytes) => bytes,
            Err(e) => return Ok(Err(e)),
        };
        let name = suggested_name(&name, image_extension(format));
        stream_out(self, encoded, name).await?;
        Ok(Ok(()))
    }

    async fn send_pcm(
        &mut self,
        p: Resource<Pcm>,
        format: AudioFormat,
        name: String,
    ) -> Result<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        let encoded = self.ctx().table.get(&p)?.encode(format);
        let name = suggested_name(&name, audio_extension(format));
        stream_out(self, encoded, name).await?;
        Ok(Ok(()))
    }

    async fn receive_blocking(&mut self) -> Result<Option<String>> {
        let process_id = self.id();
        let message = idle_while(
            process_id,
            crate::server::inbox::receive(process_id.to_string()),
        )
        .await;
        message
            .with_context(|| format!("session.receive-blocking failed for process {process_id}"))
            .map(Some)
    }

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
        let message = idle_while(
            process_id,
            crate::server::inbox::receive(process_id.to_string()),
        )
        .await;
        message
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
