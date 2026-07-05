//! Fired-batch response marshaling.
//!
//! Takes a driver `ForwardResponse` (the awaited forward output) and fans it
//! out to each request's oneshot completion, reconstructing per-request tensor
//! slices, threading chunked continuations back to the submit channel, and
//! deferring request-husk dealloc off the dispatch hot path.

use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use anyhow::Result;

use crate::arena::PhysicalPageId;
use crate::driver::DriverId;

use super::scheduler::{Completion, PendingRequest, now_micros};
use super::stats::{BatchExecutionTiming, SchedulerStats, SYSTEM_SPEC_DRAFT_POS_BUCKETS};
use super::{ForwardOutput, request};


/// Dispatch a fired batch's response to the per-request oneshots and
/// accumulate spec-decode draft counters. `fire_result` is the awaited
/// forward response (the GPU wait already happened off-thread). The
/// `deferred_drop` punt still routes request-husk dealloc to the blocking
/// pool so it does not compete with response dispatch.
pub(crate) fn dispatch_fired_batch(
    fire_result: Result<pie_driver_abi::ForwardResponse>,
    requests: Vec<PendingRequest>,
    driver_id: DriverId,
    page_size: u32,
    rt_handle: &tokio::runtime::Handle,
    submit_tx: Option<crossbeam::channel::Sender<PendingRequest>>,
    stats: &SchedulerStats,
) -> BatchExecutionTiming {
    // Detect if ANY request carries system spec drafts. The
    // common case (256-conc decode) has none, so we skip the
    // per-request Vec build + position-histogram loop.
    let any_spec = requests.iter().any(|req| !req.request.spec_token_ids.is_empty());
    let system_spec_proposed_per_req: Vec<usize> = if any_spec {
        requests
            .iter()
            .map(|req| req.request.spec_token_ids.len())
            .collect()
    } else {
        Vec::new()
    };
    let system_spec_draft_tokens_proposed =
        system_spec_proposed_per_req.iter().sum::<usize>() as u64;
    let mut system_spec_draft_tokens_accepted = 0u64;
    let mut system_spec_draft_tokens_proposed_per_pos =
        [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];
    let mut system_spec_draft_tokens_accepted_per_pos =
        [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];
    if any_spec {
        for proposed in &system_spec_proposed_per_req {
            for pos in 0..(*proposed).min(SYSTEM_SPEC_DRAFT_POS_BUCKETS) {
                system_spec_draft_tokens_proposed_per_pos[pos] += 1;
            }
        }
    }

    // Response dispatch: per-request oneshot fires and queueing the
    // deferred_drop Vec. The GPU wait already happened off-thread (the
    // caller awaited `FireHandle::wait` before handing us `fire_result`).
    //
    // Per-completion-type counts are accumulated into these locals
    // inside the match arms and fetch_add'd once after the loop, so
    // we don't pay a per-request atomic op on the hot path.
    let response_dispatch_start = Instant::now();
    let mut direct_count: u64 = 0;
    let mut chunk_count: u64 = 0;
    match fire_result {
        Ok(batch_resp) => {
            let wp = batch_resp.probe_wire_parse_us as u64;
            let pl = batch_resp.probe_plan_us as u64;
            let hd = batch_resp.probe_h2d_us as u64;
            let kl = batch_resp.probe_kernel_launch_us as u64;
            let sy = batch_resp.probe_sync_us as u64;
            let rb = batch_resp.probe_response_build_us as u64;
            let di = batch_resp.probe_device_idle_us as u64;
            if wp | pl | hd | kl | sy | rb | di != 0 {
                stats.driver_cuda.wire_parse_us.fetch_add(wp, Relaxed);
                stats.driver_cuda.plan_us.fetch_add(pl, Relaxed);
                stats.driver_cuda.h2d_us.fetch_add(hd, Relaxed);
                stats.driver_cuda.kernel_launch_us.fetch_add(kl, Relaxed);
                stats.driver_cuda.sync_us.fetch_add(sy, Relaxed);
                stats.driver_cuda.response_build_us.fetch_add(rb, Relaxed);
                stats.driver_cuda.device_idle_us.fetch_add(di, Relaxed);
                // Feed the ACCURATE device-idle into the driver bubble
                // histogram (the true G3 p50). We're inside the non-zero
                // guard, so the CUDA driver IS profiling — `di == 0` here
                // legitimately means "device was busy, no bubble" (recorded
                // as bucket 0, parallel to the host proxy's 0s). When the
                // driver doesn't profile, this block is skipped entirely, so
                // the driver histogram stays empty and readers fall back to
                // the host-proxy histogram (`InferenceStats::bubble_p50`).
                stats.record_bubble_us_driver(di);
            }
            let n_results = batch_resp.num_requests as usize;
            if requests.len() == 1 && requests[0].prebuilt {
                // === G2 prebuilt-passthrough (beam) response ===
                // B driver-lanes but ONE program (program 0) + ONE
                // PendingRequest. Hand the WHOLE rich response verbatim to the
                // single completion — ptir_host reads `resp.ptir_output_at(0)`
                // (the [B] out/out_par/out_scr program tensors). Bypasses the
                // per-row `num_requests == requests.len()` split below
                // (num_requests = B ≠ the 1 PendingRequest).
                let req = requests
                    .into_iter()
                    .next()
                    .expect("prebuilt fire has exactly one request");
                // `..` drops `request` + `physical_page_ids` inline (one
                // request — negligible; the KV txn is finalized by ptir_host).
                let PendingRequest { completion, .. } = req;
                match completion {
                    Completion::Direct(tx) => {
                        direct_count += 1;
                        tx.send(Ok(ForwardOutput::Response(batch_resp))).ok();
                    }
                    Completion::Chunk { .. } => {
                        // Unreachable: a prebuilt beam fire is always a single
                        // Direct completion, never a chunked continuation.
                        tracing::error!(
                            "prebuilt beam fire had an unexpected Chunk completion"
                        );
                    }
                }
                stats.fire.last_dispatch_end_micros.store(now_micros(), Relaxed);
            } else if n_results != requests.len() {
                let msg = format!(
                    "batch response count mismatch from driver {driver_id}: \
                     expected {}, got {n_results}",
                    requests.len()
                );
                tracing::error!(
                    driver = driver_id,
                    expected = requests.len(),
                    got = n_results,
                    "Batch response count mismatch",
                );
                for req in requests {
                    req.send_result::<ForwardOutput>(
                        Err(anyhow::anyhow!(msg.clone())),
                        None,
                        page_size,
                    );
                }
            } else {
                let has_chunked = requests
                    .iter()
                    .any(|req| matches!(req.completion, Completion::Chunk { .. }));
                let token_payload_only = !has_chunked
                    && batch_resp.dists_ids.is_empty()
                    && batch_resp.dists_probs.is_empty()
                    && batch_resp.logits_bytes.is_empty()
                    && batch_resp.logprobs_values.is_empty()
                    && batch_resp.entropies.is_empty()
                    && batch_resp.spec_tokens.is_empty()
                    // Key on DECLARED program-token output SLOTS, not the flat
                    // `program_tokens` emptiness: a program that declares a
                    // `[k]`-Token output but produces an EMPTY accept-prefix
                    // (all `-1` → truncated) has empty flat `program_tokens` AND
                    // empty dense `tokens`, so the old `program_tokens.is_empty()`
                    // mis-routed it to the dense-token shortcut → its declared
                    // output was lost ("no output tensor"; the §6.1 mtpverify /
                    // grammar_inferlet_constrains_output root). `program_tokens_
                    // req_indptr.last()` = Σ declared slots; 0 ⇒ no program-token
                    // output ⇒ dense-token-only. Non-zero ⇒ take the rich
                    // Response path (which reconstructs the empty `[k]` tensor).
                    && batch_resp
                        .program_tokens_req_indptr
                        .last()
                        .map_or(true, |&n| n == 0)
                    && batch_resp.tokens_indptr.len() >= requests.len() + 1;

                // Send oneshot replies first, defer drop of the
                // request husks. Each PendingRequest's drop is
                // ~3-4 µs (22-Vec ForwardRequest), and doing it
                // inline adds avoidable tail latency to response
                // dispatch for large batches.
                let mut deferred_drop: Vec<(
                    pie_driver_abi::ForwardRequest,
                    Vec<PhysicalPageId>,
                )> = Vec::with_capacity(n_results);
                // #27 cut #1 eager-D2H fast-path (a2-mode): a request that set
                // up the `sampling_output_*` dst table had its sampled token
                // copied DIRECTLY to the pinned output Tensor (D2H), so the
                // driver response carries NO token (`tokens[]` empty). Resolve
                // each oneshot with success WITHOUT extracting `tokens[..]` —
                // the inferlet's `output()` reads the filled pinned buffer
                // (gated on `forward_result.is_some()`); an `Err`/drop here
                // would hit the abort path (txn drop + "no output tensor").
                // Keyed per-request on `sampling_output_*`, which
                // `populate_output_fastpath` sets iff it also stashed the
                // pinned outputs (1:1 with the host pinned-read gate, so no
                // skew). One-ahead MVP batches are all-or-nothing fast-path.
                let all_fast_path = !requests.is_empty()
                    && requests
                        .iter()
                        .all(|req| !req.request.sampling_output_dst_ptrs.is_empty());
                if all_fast_path {
                    for req in requests.into_iter() {
                        // Empty `Tokens` is `Some` → the host gate passes and
                        // reads the pinned buffer; the payload is ignored.
                        let output = ForwardOutput::Tokens(Vec::new());
                        let PendingRequest {
                            request,
                            completion,
                            physical_page_ids,
                            program_identity_hashes,
                            pipeline_id,
                            ..
                        } = req;
                        match completion {
                            Completion::Direct(tx) => {
                                direct_count += 1;
                                tx.send(Ok(output)).ok();
                                deferred_drop.push((request, physical_page_ids));
                            }
                            Completion::Chunk { .. } => {
                                chunk_count += 1;
                                let req = PendingRequest {
                                    request,
                                    completion,
                                    physical_page_ids,
                                    last_page_len: 0,
                                    program_identity_hashes,
                                    pipeline_id,
                                    submitted_at_us: 0,
                                    prebuilt: false,
                                };
                                req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                            }
                        }
                    }
                } else if token_payload_only {
                    for (r, req) in requests.into_iter().enumerate() {
                        let lo = batch_resp.tokens_indptr[r] as usize;
                        let hi = batch_resp.tokens_indptr[r + 1] as usize;
                        if system_spec_proposed_per_req
                            .get(r)
                            .copied()
                            .unwrap_or_default()
                            > 0
                        {
                            let accepted = hi.saturating_sub(lo).saturating_sub(1);
                            system_spec_draft_tokens_accepted += accepted as u64;
                            for pos in 0..accepted.min(SYSTEM_SPEC_DRAFT_POS_BUCKETS) {
                                system_spec_draft_tokens_accepted_per_pos[pos] += 1;
                            }
                        }
                        let output = if hi == lo + 1 {
                            ForwardOutput::Token(batch_resp.tokens[lo])
                        } else {
                            ForwardOutput::Tokens(batch_resp.tokens[lo..hi].to_vec())
                        };
                        let PendingRequest {
                            request,
                            completion,
                            physical_page_ids,
                            program_identity_hashes,
                            pipeline_id,
                            ..
                        } = req;
                        match completion {
                            Completion::Direct(tx) => {
                                direct_count += 1;
                                tx.send(Ok(output)).ok();
                                deferred_drop.push((request, physical_page_ids));
                            }
                            Completion::Chunk { .. } => {
                                chunk_count += 1;
                                let req = PendingRequest {
                                    request,
                                    completion,
                                    physical_page_ids,
                                    last_page_len: 0,
                                    program_identity_hashes,
                                    pipeline_id,
                                    submitted_at_us: 0,
                                    prebuilt: false,
                                };
                                req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                            }
                        }
                    }
                } else {
                    for (r, req) in requests.into_iter().enumerate() {
                        let per_req = request::extract_per_request(&batch_resp, r);
                        if system_spec_proposed_per_req
                            .get(r)
                            .copied()
                            .unwrap_or_default()
                            > 0
                        {
                            let accepted = per_req.tokens.len().saturating_sub(1);
                            system_spec_draft_tokens_accepted += accepted as u64;
                            for pos in 0..accepted.min(SYSTEM_SPEC_DRAFT_POS_BUCKETS) {
                                system_spec_draft_tokens_accepted_per_pos[pos] += 1;
                            }
                        }
                        let output = ForwardOutput::Response(per_req);
                        let PendingRequest {
                            request,
                            completion,
                            physical_page_ids,
                            program_identity_hashes,
                            pipeline_id,
                            ..
                        } = req;
                        match completion {
                            Completion::Direct(tx) => {
                                direct_count += 1;
                                tx.send(Ok(output)).ok();
                                deferred_drop.push((request, physical_page_ids));
                            }
                            Completion::Chunk { .. } => {
                                chunk_count += 1;
                                let req = PendingRequest {
                                    request,
                                    completion,
                                    physical_page_ids,
                                    last_page_len: 0,
                                    program_identity_hashes,
                                    pipeline_id,
                                    submitted_at_us: 0,
                                    prebuilt: false,
                                };
                                req.send_result(Ok(output), submit_tx.as_ref(), page_size);
                            }
                        }
                    }
                }
                stats.fire.last_dispatch_end_micros.store(now_micros(), Relaxed);
                if !deferred_drop.is_empty() {
                    // Dedicated blocking pool so this dealloc task
                    // does not compete with response dispatch. Use the captured
                    // `rt_handle` because we're now on the scheduler
                    // OS thread, not a tokio task — `tokio::task::
                    // spawn_blocking` would panic without an ambient
                    // runtime context.
                    rt_handle.spawn_blocking(move || drop(deferred_drop));
                }
            }
        }
        Err(e) => {
            tracing::error!("fire_batch failed for driver {}: {:?}", driver_id, e);
            for req in requests {
                req.send_result::<ForwardOutput>(
                    Err(anyhow::anyhow!(
                        "fire_batch failed for driver {driver_id}: {e:#}"
                    )),
                    None,
                    page_size,
                );
            }
        }
    }
    crate::probe_fire_record!(
        stats.fire.execute.response_dispatch.total_us,
        response_dispatch_start.elapsed()
    );
    // Task-B (carrier ⋈ contention): every response sent above is now
    // drainable by its owner's gate loop — wake lanes waiting to finalize
    // their own retired fires (releases their pins/grace refs so
    // `classify_for_suspend` can yield). No-op outside preempt mode.
    crate::inference::contention::notify_fire_retired();
    // Per-completion-type counts. Counters, not durations — three
    // atomic ops per fire regardless of batch size, so always-on
    // (no feature gate).
    if direct_count > 0 {
        stats
            .fire
            .execute
            .response_dispatch
            .direct_count
            .fetch_add(direct_count, Relaxed);
    }
    if chunk_count > 0 {
        stats
            .fire
            .execute
            .response_dispatch
            .chunk_count
            .fetch_add(chunk_count, Relaxed);
    }
    BatchExecutionTiming {
        system_spec_draft_tokens_proposed,
        system_spec_draft_tokens_accepted,
        system_spec_draft_tokens_proposed_per_pos,
        system_spec_draft_tokens_accepted_per_pos,
    }
}
