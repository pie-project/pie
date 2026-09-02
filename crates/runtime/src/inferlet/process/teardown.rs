//! Deferred process-resource teardown: finalizes a departing process's
//! pending pipeline operations off the guest task, removes its scratch
//! directory and batches its channel closes. The capped execution slot is NOT
//! released here — `ProcessCtx::drop` frees it synchronously, ahead of this
//! task, and hands down the terminate fences it posted (see
//! `post_process_terminate_fenced`).

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

/// Remove a process's scratch directory. `None` means the sandbox denied the
/// filesystem, so no directory was ever created — skipping it keeps a
/// cohort-boundary teardown herd from issuing 512 pointless `openat`s.
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
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        tracing::error!(
            pid = %process_id,
            "process teardown found pending fires without a Tokio runtime; preserving the \
             ResourceTable to avoid recycling pages under native work"
        );
        // Only pooled resources are preserved here, never admission capacity.
        crate::inferlet::process::release_bind_permit(context.bind_permit.take());
        std::mem::forget(context);
        remove_scratch(scratch_dir.as_deref());
        // `ProcessCtx::drop` already released the execution permit; the terminate
        // tombstone stays since late items for this pid remain possible.
        return;
    };
    let task = async move {
        if capped_execution {
            // Reference fence: after this resolves, every engine's scheduler has
            // purged the pid's queued work, so the finalize loop and resource
            // drop below run with no scheduler-side reference to recycled pages.
            crate::scheduler::worker::await_terminate_fences(terminate_fences.unwrap_or_default())
                .await;
        } else {
            for pipeline_id in snapshot.departed_pipeline_ids {
                crate::scheduler::worker::notify_pipeline_close(pipeline_id).await;
            }
        }
        for fires in snapshot.pipelines {
            // Teardown policy: log and keep draining — the table drops next.
            let _ = crate::pipeline::fire::finalize_all(&mut context, &fires, true).await;
        }
        // Take over the guest channels' close notifications before the table
        // drops, so a departing process posts one batched close per engine
        // instead of one mailbox item per channel.
        let channel_close_batches =
            crate::pipeline::channel::detach_channel_close_notifications(&mut context.resources);
        crate::inferlet::process::release_bind_permit(context.bind_permit.take());
        drop(context);
        remove_scratch(scratch_dir.as_deref());
        for (engine_id, ids) in channel_close_batches {
            if let Err(error) = crate::scheduler::close_channels(engine_id, ids) {
                // Same failure mode as the per-endpoint closer (the
                // scheduler is gone at shutdown); engine shutdown closes
                // any channels this batch could not reach.
                tracing::warn!(pid = %process_id, engine_id, %error,
                    "process teardown failed to post its batched channel close");
            }
        }
        // The process's last event on every mailbox: the tombstone that
        // deduped its Terminate leaves can now retire.
        crate::scheduler::worker::notify_process_quiesced(process_id);
    };
    runtime.spawn(task);
}
