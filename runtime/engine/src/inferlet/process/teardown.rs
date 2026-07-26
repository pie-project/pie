//! Deferred process-resource teardown: finalizes a departing process's
//! pending pipeline operations off the guest task, releases its capped
//! execution slot in event order, and batches its channel closes.

use std::sync::{Arc, Mutex};

pub(crate) struct TeardownFireContext {
    process_id: uuid::Uuid,
    resources: wasmtime::component::ResourceTable,
    // Dropped after `resources` (field declaration order), so strict
    // admission advances only after pooled resources are released.
    _execution_permit: Option<tokio::sync::OwnedSemaphorePermit>,
    _bind_permit: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl crate::pipeline::fire::FireContext for TeardownFireContext {
    fn resources(&mut self) -> &mut wasmtime::component::ResourceTable {
        &mut self.resources
    }

    fn process_id(&self) -> uuid::Uuid {
        self.process_id
    }
}

pub(crate) fn defer_resource_teardown(
    process_id: uuid::Uuid,
    resources: wasmtime::component::ResourceTable,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    execution_permit: Option<tokio::sync::OwnedSemaphorePermit>,
    bind_permit: Option<tokio::sync::OwnedSemaphorePermit>,
) {
    let capped_execution = execution_permit.is_some();
    let snapshot = residency.lock().unwrap().teardown_snapshot();
    let mut context = TeardownFireContext {
        process_id,
        resources,
        _execution_permit: execution_permit,
        _bind_permit: bind_permit,
    };
    if !capped_execution
        && snapshot.departed_pipeline_ids.is_empty()
        && snapshot
            .pipelines
            .iter()
            .all(|fires| fires.lock().unwrap().is_empty())
    {
        drop(context);
        return;
    }
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        tracing::error!(
            pid = %process_id,
            "process teardown found pending fires without a Tokio runtime; preserving the \
             ResourceTable to avoid recycling pages under native work"
        );
        std::mem::forget(context);
        // The leaked permit shrank the execution pool by one; tell the
        // policy the departure resolves as a FORFEIT so its seal neither
        // waits forever for a release that cannot come nor credits a free
        // slot the semaphore no longer has. (Sync send — no runtime
        // needed.) The terminate tombstone stays: with the pending fires
        // forgotten, late items for this pid remain possible.
        if capped_execution {
            crate::scheduler::worker::notify_execution_slot_forfeited(process_id);
        }
        return;
    };
    runtime.spawn(async move {
        let timing = crate::scheduler::fire_timing_enabled();
        let task_started_us = if timing {
            crate::scheduler::fire_timing_now_us()
        } else {
            0
        };
        if capped_execution {
            // Reference fence, awaited on purpose: after this resolves,
            // every driver's scheduler has purged the pid's queued work and
            // cancelled its protected in-flight control, so the finalize
            // loop below and the resource drop at the end run with no
            // scheduler-side reference to the pages they recycle. (The
            // fence also serializes each retirement behind a scheduler
            // pass; that pacing is a side effect, not the contract.)
            crate::scheduler::worker::notify_process_terminate(process_id).await;
        } else {
            for pipeline_id in snapshot.departed_pipeline_ids {
                crate::scheduler::worker::notify_pipeline_close(pipeline_id).await;
            }
        }
        let terminate_acked_us = if timing {
            crate::scheduler::fire_timing_now_us()
        } else {
            0
        };
        for fires in snapshot.pipelines {
            // Teardown policy: log and keep draining — the table drops next.
            let _ = crate::pipeline::fire::finalize_all(&mut context, &fires, true).await;
        }
        if timing {
            let finalized_us = crate::scheduler::fire_timing_now_us();
            crate::scheduler::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "runtime",
                "event": "process_teardown",
                "process_id": process_id,
                "task_started_us": task_started_us,
                "terminate_ack_us": terminate_acked_us - task_started_us,
                "finalize_us": finalized_us - terminate_acked_us,
                "released_us": finalized_us,
            }));
        }
        // Take over the guest channels' close notifications before the
        // table drops, so a departing process posts one batched close per
        // driver instead of one mailbox item per channel (a teardown herd
        // otherwise inflates the epoch a worker pass has to drain). The
        // batch is posted only after `drop(context)` below: the drop's pass
        // teardowns post the instance closes first, and per-producer FIFO
        // then keeps the driver's instance-before-channel close order.
        let channel_close_batches =
            crate::pipeline::channel::detach_channel_close_notifications(&mut context.resources);
        if capped_execution {
            // Announce the slot BEFORE the permit drops (with `context`
            // below): the successor can only acquire after the drop, so its
            // slot-consumed event always enters the scheduler mailbox after
            // this release — the policy's slot balance never sees a consume
            // for a release it hasn't counted. (The terminate notification
            // above already removed this process's own lane: leave first,
            // release second, successor's admission and fire after.)
            crate::scheduler::worker::notify_execution_slot_released(process_id);
        }
        drop(context);
        for (driver_id, ids) in channel_close_batches {
            if let Err(error) = crate::scheduler::close_channels(driver_id, ids) {
                // Same failure mode as the per-endpoint closer (the
                // scheduler is gone at shutdown); driver shutdown closes
                // any channels this batch could not reach.
                tracing::warn!(pid = %process_id, driver_id, %error,
                    "process teardown failed to post its batched channel close");
            }
        }
        // Strictly the process's last event on every mailbox (this task is
        // its final producer, and the drop + batched close above already
        // posted its close controls): the tombstone that deduped its
        // Terminate leaves can now retire.
        crate::scheduler::worker::notify_process_quiesced(process_id);
    });
}
