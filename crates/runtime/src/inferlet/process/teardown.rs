use std::sync::{Arc, Mutex};

pub(crate) struct TeardownFireContext {
    process_id: uuid::Uuid,
    resources: wasmtime::component::ResourceTable,
    bind_permit: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl crate::pipeline::fire::FireContext for TeardownFireContext {
    fn resources(&mut self) -> &mut wasmtime::component::ResourceTable {
        &mut self.resources
    }

    fn process_id(&self) -> uuid::Uuid {
        self.process_id
    }
}

fn remove_scratch(dir: Option<&std::path::Path>) {
    if let Some(dir) = dir {
        let _ = std::fs::remove_dir_all(dir);
    }
}

pub(crate) fn defer_resource_teardown(
    process_id: uuid::Uuid,
    resources: wasmtime::component::ResourceTable,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    terminate_fences: Option<Vec<crate::scheduler::worker::TerminateFence>>,
    bind_permit: Option<tokio::sync::OwnedSemaphorePermit>,
    scratch_dir: Option<std::path::PathBuf>,
) {
    let capped_execution = terminate_fences.is_some();
    let snapshot = residency.lock().unwrap().teardown_snapshot();
    let mut context = TeardownFireContext {
        process_id,
        resources,
        bind_permit,
    };
    if !capped_execution
        && snapshot.departed_pipeline_ids.is_empty()
        && snapshot
            .pipelines
            .iter()
            .all(|fires| fires.lock().unwrap().is_empty())
    {
        crate::inferlet::process::release_bind_permit(context.bind_permit.take());
        drop(context);
        remove_scratch(scratch_dir.as_deref());
        return;
    }
    if !crate::rt::has_runtime() {
        tracing::error!(
            pid = %process_id,
            "process teardown found pending fires without a Tokio runtime; preserving the \
             ResourceTable to avoid recycling pages under native work"
        );
        crate::inferlet::process::release_bind_permit(context.bind_permit.take());
        std::mem::forget(context);
        remove_scratch(scratch_dir.as_deref());
        return;
    };
    let task = async move {
        if capped_execution {
            crate::scheduler::worker::await_terminate_fences(terminate_fences.unwrap_or_default())
                .await;
        } else {
            for pipeline_id in snapshot.departed_pipeline_ids {
                crate::scheduler::worker::notify_pipeline_close(pipeline_id).await;
            }
        }
        for fires in snapshot.pipelines {
            let _ = crate::pipeline::fire::finalize_all(&mut context, &fires, true).await;
        }
        let channel_close_batches =
            crate::pipeline::channel::detach_channel_close_notifications(&mut context.resources);
        crate::inferlet::process::release_bind_permit(context.bind_permit.take());
        drop(context);
        remove_scratch(scratch_dir.as_deref());
        for (engine_id, ids) in channel_close_batches {
            if let Err(error) = crate::scheduler::close_channels(engine_id, ids) {
                tracing::warn!(pid = %process_id, engine_id, %error,
                    "process teardown failed to post its batched channel close");
            }
        }
        crate::scheduler::worker::notify_process_quiesced(process_id);
    };
    crate::rt::spawn(task);
}
