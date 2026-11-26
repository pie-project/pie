use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Process contexts made and not yet torn down: a context counts from
/// [`Live::new`] until its teardown has closed what it held (its channels,
/// its instances) and said it quiesced, on the fast path or at the end of
/// the deferred task.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static QUIESCED: tokio::sync::Notify = tokio::sync::Notify::const_new();

/// One process context's place in [`LIVE`], given back where its teardown
/// ends.
pub(crate) struct Live(());

impl Live {
    pub(crate) fn new() -> Live {
        LIVE.fetch_add(1, Ordering::AcqRel);
        Live(())
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        if LIVE.fetch_sub(1, Ordering::AcqRel) == 1 {
            QUIESCED.notify_waiters();
        }
    }
}

/// Once every process context made has been torn down: what each held is
/// closed on the schedulers. No timer: a shutdown waits on it after ending
/// every process, before it stops the schedulers.
pub(crate) async fn all_torn_down() {
    loop {
        let quiesced = QUIESCED.notified();
        tokio::pin!(quiesced);
        quiesced.as_mut().enable();
        if LIVE.load(Ordering::Acquire) == 0 {
            return;
        }
        quiesced.await;
    }
}

pub(crate) struct TeardownFireContext {
    process_id: uuid::Uuid,
    resources: wasmtime::component::ResourceTable,
    bind_permit: Option<super::ordered::OrderedPermit>,
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

/// After the last fire, so what the guest gave is at rest when it moves.
fn settle_handles(
    handles: Option<crate::handle::Bag>,
    context: &mut TeardownFireContext,
    residency: &Mutex<crate::inferlet::process::ProcessResidency>,
) {
    if let Some(handles) = handles {
        handles.settle(&mut context.resources, residency);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn defer_resource_teardown(
    process_id: uuid::Uuid,
    resources: wasmtime::component::ResourceTable,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    terminate_fences: Option<Vec<crate::scheduler::worker::TerminateFence>>,
    bind_permit: Option<super::ordered::OrderedPermit>,
    scratch_dir: Option<std::path::PathBuf>,
    handles: Option<crate::handle::Bag>,
    live: Option<Live>,
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
        settle_handles(handles, &mut context, &residency);
        drop(context);
        remove_scratch(scratch_dir.as_deref());
        drop(live);
        return;
    }
    if !crate::rt::has_runtime() {
        tracing::error!(
            pid = %process_id,
            "process teardown found pending fires without a Tokio runtime; preserving the \
             ResourceTable to avoid recycling pages under native work"
        );
        crate::inferlet::process::release_bind_permit(context.bind_permit.take());
        if let Some(handles) = handles {
            handles.abandon();
        }
        std::mem::forget(context);
        remove_scratch(scratch_dir.as_deref());
        drop(live);
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
        settle_handles(handles, &mut context, &residency);
        drop(context);
        remove_scratch(scratch_dir.as_deref());
        for (engine_id, ids) in channel_close_batches {
            if let Err(error) = crate::scheduler::close_channels(engine_id, ids) {
                tracing::warn!(pid = %process_id, engine_id, %error,
                    "process teardown failed to post its batched channel close");
            }
        }
        crate::scheduler::worker::notify_process_quiesced(process_id);
        drop(live);
    };
    crate::rt::spawn(task);
}
