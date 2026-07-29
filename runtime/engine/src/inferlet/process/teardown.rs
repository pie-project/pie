//! Deferred process-resource teardown: finalizes a departing process's
//! pending pipeline operations off the guest task and batches its channel
//! closes. The capped execution slot is NOT released here — `ProcessCtx::drop`
//! frees it synchronously, ahead of this task, and hands down the terminate
//! fences it posted (see `post_process_terminate_fenced`).

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

pub(crate) fn defer_resource_teardown(
    process_id: uuid::Uuid,
    resources: wasmtime::component::ResourceTable,
    residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    terminate_fences: Option<Vec<crate::scheduler::worker::TerminateFence>>,
    bind_permit: Option<tokio::sync::OwnedSemaphorePermit>,
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
        return;
    }
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        tracing::error!(
            pid = %process_id,
            "process teardown found pending fires without a Tokio runtime; preserving the \
             ResourceTable to avoid recycling pages under native work"
        );
        // The seat and the bind permit both leave before the leak: only
        // POOLED RESOURCES are preserved here, never admission capacity.
        crate::inferlet::process::release_bind_permit(context.bind_permit.take());
        std::mem::forget(context);
        // `ProcessCtx::drop` already released the execution permit and
        // broadcast the release, so the semaphore's capacity is intact and
        // the policy's departure is resolved. (Before the permit moved out
        // of this context the leak took the seat with it, which is what the
        // forfeit broadcast existed to announce.) The terminate tombstone
        // stays: with the pending fires forgotten, late items for this pid
        // remain possible.
        return;
    };
    let timing = crate::scheduler::fire_timing_enabled();
    let task = async move {
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
            // scheduler-side reference to the pages they recycle. The leave
            // itself was POSTED by `ProcessCtx::drop` (ahead of the slot
            // release, preserving leave-then-release); only the await is
            // deferred here, so a retirement no longer holds its successor's
            // seat hostage to this task's spawn latency.
            crate::scheduler::worker::await_terminate_fences(terminate_fences.unwrap_or_default())
                .await;
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
        let finalized_us = if timing {
            crate::scheduler::fire_timing_now_us()
        } else {
            0
        };
        // Take over the guest channels' close notifications before the
        // table drops, so a departing process posts one batched close per
        // driver instead of one mailbox item per channel (a teardown herd
        // otherwise inflates the epoch a worker pass has to drain). The
        // batch is posted only after `drop(context)` below: the drop's pass
        // teardowns post the instance closes first, and per-producer FIFO
        // then keeps the driver's instance-before-channel close order.
        let channel_close_batches =
            crate::pipeline::channel::detach_channel_close_notifications(&mut context.resources);
        let detached_us = if timing {
            crate::scheduler::fire_timing_now_us()
        } else {
            0
        };
        let channels = channel_close_batches
            .iter()
            .map(|(_, ids)| ids.len())
            .sum::<usize>();
        crate::inferlet::process::release_bind_permit(context.bind_permit.take());
        drop(context);
        let dropped_us = if timing {
            crate::scheduler::fire_timing_now_us()
        } else {
            0
        };
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
        if timing {
            let quiesced_us = crate::scheduler::fire_timing_now_us();
            crate::scheduler::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "runtime",
                "event": "process_teardown",
                "process_id": process_id,
                "task_started_us": task_started_us,
                "terminate_ack_us": terminate_acked_us - task_started_us,
                "finalize_us": finalized_us - terminate_acked_us,
                "detach_us": detached_us - finalized_us,
                // The wasmtime `ResourceTable` drop: previously invisible,
                // because the record was written before it.
                "table_drop_us": dropped_us - detached_us,
                "close_post_us": quiesced_us - dropped_us,
                "channels": channels,
                "released_us": quiesced_us,
            }));
        }
    };
    if timing {
        runtime.spawn(crate::scheduler::CpuMetered::new(
            crate::scheduler::CpuClass::Teardown,
            task,
        ));
    } else {
        runtime.spawn(task);
    }
}
