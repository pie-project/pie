//! One fire: prepare → run-ahead submit → finalize/poison — the non-glue
//! run-ahead engine (the WIT host glue lives one layer up, in the
//! `inferlet::host` forward/pipeline modules).
//!
//! **Run-ahead** (overview §3): `pipeline.submit` NEVER blocks. It prepares the
//! fire (seeds, host puts, KV/RS projection), hands the request to the
//! scheduler, and enqueues a [`PendingFire`] (the payload-free completion + the
//! open KV/RS txns) on the pass — the classic `execute()`/`output()` split
//! (`PendingForward`, Option A) applied to this engine. Step t+1 is prepared
//! against t's OPTIMISTIC post-state (the `committed_tokens` cursor advances at
//! submit), so a decode loop keeps the scheduler fed. `channel.take`/`read`
//! are the await points: they finalize in-flight fires FIFO until the cell
//! fills. A failed fire **poisons** the pass's host-reader channels (the only
//! error path once submit is non-blocking) and fails the pass for further
//! submits.
//!
//! **Layering.** The orchestration functions below (`submit_pass`,
//! `finalize_fire`, `drain_settled`, `wire_channels_to_pipeline`,
//! `fire_device_geometry`, `pipeline_close`/`pipeline_drop`, `copy_into_inner`)
//! need to get/get_mut/delete/push `Resource<Channel>`/`Resource<ForwardPass>`/
//! `Resource<Pipeline>` handles in the WASM component resource table, but they
//! are plain functions generic over [`FireContext`] — a narrow trait this
//! module defines (naming only the external `wasmtime::component::
//! ResourceTable` leaf type, never `ProcessCtx`/`inferlet`). `inferlet::host`
//! (L4) implements `FireContext` for `ProcessCtx` and its `Host*` impls call
//! these functions with `self`. This keeps `pipeline/` strictly below
//! `inferlet/` in the layering while the fire engine still owns every bit of
//! the algorithm — see [`FireContext`]'s doc for the design rationale.

pub mod context;
pub mod geometry;
pub mod kv;
pub mod lease;
pub mod rs;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use wasmtime::component::Resource;

pub use context::FireContext;

use crate::pipeline::Pipeline;
use crate::pipeline::channel::{BoundCells, Channel, ChannelError};
use crate::pipeline::instance::ForwardPass;
use crate::store::kv::working_set::{KvFireLease, KvWorkingSet};
use crate::store::rs::working_set::RsWorkingSet;
use lease::DevGeo;
use pie_ptir::container::{HostRole, PortSource, TraceContainer};
use pie_ptir::op::Op;
use pie_ptir::registry::Port;

/// A pass's in-flight fires, submit order. Plain mutex: never held across an
/// await (the op is popped out, then awaited).
pub type PendingFires = Arc<Mutex<VecDeque<PendingOp>>>;
pub type PipelineFailure = Arc<Mutex<Option<String>>>;

/// A pipeline FIFO entry: a forward FIRE or a KV cell MOVE (Design-B
/// compaction). Both hold an ordered slot on the same stream — the B3
/// happens-before invariant; `take`/`read` drain them in submit order.
pub enum PendingOp {
    Fire(PendingFire),
    Move(PendingMove),
}

impl PendingOp {
    /// Non-blocking probe: whether the op's completion has settled.
    fn is_settled(&self) -> bool {
        match self {
            PendingOp::Fire(fire) => fire.completion.is_settled(),
            PendingOp::Move(mv) => mv.completion.is_settled(),
        }
    }

    /// An owned, payload-free await on this op's completion (cloned so the
    /// pipeline queue lock is not held across the await). The outcome is
    /// ignored; the FIFO drain reads the real result.
    fn completion_signal(&self) -> OpSignal {
        match self {
            PendingOp::Fire(fire) => OpSignal::Fire(fire.completion.clone()),
            PendingOp::Move(mv) => OpSignal::Move(mv.completion.clone()),
        }
    }
}

/// See [`PendingOp::completion_signal`].
enum OpSignal {
    Fire(crate::driver::WorkItemCompletion),
    Move(crate::driver::SubmissionCompletion),
}

impl std::future::Future for OpSignal {
    type Output = ();

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        match self.get_mut() {
            OpSignal::Fire(completion) => std::pin::Pin::new(completion).poll(cx).map(|_| ()),
            OpSignal::Move(completion) => std::pin::Pin::new(completion).poll(cx).map(|_| ()),
        }
    }
}

/// One in-flight KV cell MOVE (`pipeline.copy-into`). It holds an ordered slot
/// in the pipeline FIFO and finalizes by awaiting its payload-free completion.
pub struct PendingMove {
    completion: crate::driver::SubmissionCompletion,
    failure: PipelineFailure,
}

/// Test-only inert FIFO entry: enough to make a pass's shared fires queue
/// non-empty for `ForwardPass::can_close_native_on_drop`/`Drop` tests
/// (`pipeline::instance`'s `native_cleanup` test module), without needing a
/// live completion. `PendingMove`/`PendingFire`'s fields are private
/// to this module, so cross-module tests construct a stub through here
/// rather than reaching in directly.
#[cfg(test)]
pub(crate) fn test_pending_move_stub() -> PendingOp {
    PendingOp::Move(PendingMove {
        completion: crate::driver::SubmissionCompletion::ready(),
        failure: Arc::new(Mutex::new(None)),
    })
}

/// The open KV/arena transaction(s) one in-flight fire holds until it resolves.
/// Two shapes: the ordinary single-seq / MTP projection ([`kv`]), or a
/// device-geometry fire whose KV the driver resolves+writes itself (B2's
/// explicit-KV path) — the runtime only pins the [`lease::PageLease`]-granted
/// physical pages for the fire, released at finalize (per-fire arena txn; the
/// plan's "pin float bounded by run-ahead depth × B, riding the per-fire arena
/// txns").
enum FireKv {
    Deferred(Arc<Mutex<Option<kv::KvTxn>>>),
    /// A device-geometry fire's prepared write over the lease-granted slots
    /// (B2's explicit-KV path): same commit/abort protocol, no host
    /// projection.
    DeviceGeom {
        kvtxn: kv::KvTxn,
    },
}

/// One in-flight fire: the work item completion plus everything needed to
/// finalize when it resolves — the open KV/RS txns (pins/CoW held until
/// commit/abort) and the bound cells whose mirror epochs become visible.
pub struct PendingFire {
    completion: crate::driver::WorkItemCompletion,
    kv: FireKv,
    rstxn: Option<rs::RsTxn>,
    ws_guard: KvFireLease,
    ws_rep: u32,
    rs_rep: Option<u32>,
    /// The owning pass, to fail it on a fire error (rep — the guest may have
    /// dropped the handle; failure marking is then moot).
    fwd_rep: u32,
    instance_id: u64,
    cells: BoundCells,
    failure: PipelineFailure,
}

/// Poison every host-reader cell of a pass with the failed fire's error —
/// under run-ahead this IS the error channel (`take`/`read` surface it).
fn poison_readers(cells: &BoundCells, reason: &str) {
    for cell in cells {
        let mut c = cell.lock().unwrap();
        if c.role == Some(HostRole::Reader) {
            c.poison(reason);
            // A waiter parked on the reader wait slot must observe the poison.
            if let Some(endpoint) = c.endpoint() {
                pie_waker::WakerTable::global().wake(endpoint.registered().reader_wait_id);
            }
        }
    }
}

fn channel_accesses(container: &TraceContainer) -> Vec<(bool, bool)> {
    let mut accesses = vec![(false, false); container.channels.len()];
    for stage in &container.stages {
        for op in &stage.ops {
            match *op {
                Op::ChanTake(channel) => accesses[channel as usize].0 = true,
                Op::ChanPut { chan, .. } => accesses[chan as usize].1 = true,
                _ => {}
            }
        }
    }
    for binding in &container.ports {
        let PortSource::Channel(channel) = &binding.source else {
            continue;
        };
        if matches!(
            binding.port,
            Port::EmbedTokens | Port::Positions | Port::WSlot | Port::WOff
        ) {
            accesses[*channel as usize].0 = true;
        }
    }
    accesses
}

fn reserve_channel_tickets(cells: &BoundCells, accesses: &[(bool, bool)]) -> (Vec<u64>, Vec<u64>) {
    cells
        .iter()
        .zip(accesses)
        .map(|(cell, &(consume, publish))| {
            cell.lock().unwrap().reserve_device_ticket(consume, publish)
        })
        .unzip()
}

fn rollback_channel_tickets(
    cells: &BoundCells,
    accesses: &[(bool, bool)],
    heads: &[u64],
    tails: &[u64],
) {
    for (((cell, _), &head), &tail) in cells.iter().zip(accesses).zip(heads).zip(tails).rev() {
        cell.lock().unwrap().rollback_device_ticket(head, tail);
    }
}

/// Park until the channel can make progress: its endpoint's reader word
/// advances (the driver's completion callback wakes the reader wait slot
/// directly), or the oldest in-flight pipeline op settles so the caller can
/// drain it. Errors surface poison/closure or a definitively empty channel
/// (no endpoint and nothing in flight: nothing can ever fill the cell).
pub(crate) async fn await_channel_progress(
    cell: &Arc<Mutex<crate::pipeline::channel::ChannelCell>>,
    fires: Option<&PendingFires>,
) -> Result<(), String> {
    let wait = cell.lock().unwrap().reader_wait_state();
    let oldest = fires.and_then(|f| f.lock().unwrap().front().map(|op| op.completion_signal()));
    match (wait, oldest) {
        (Some((endpoint, observed_tail)), Some(signal)) => {
            // Race the direct channel wake against the oldest op so a fire
            // that resolves without producing here still unblocks the loop.
            // Poison/closure re-classify on the caller's next take/read.
            tokio::select! {
                _ = endpoint.wait_for_reader_change(observed_tail) => {}
                _ = signal => {}
            }
            Ok(())
        }
        (Some((endpoint, observed_tail)), None) => endpoint
            .wait_for_reader_change(observed_tail)
            .await
            .map_err(|error| error.to_string()),
        (None, Some(signal)) => {
            signal.await;
            Ok(())
        }
        (None, None) => Err(ChannelError::Empty.to_string()),
    }
}

type Anyhow<T> = anyhow::Result<T>;

/// The body behind `forward-pass.submit(on)`.
pub async fn submit_pass<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
) -> Anyhow<Result<(), String>> {
    {
        // Device-geometry pass (Track B): the [B,P] geometry is
        // device-produced (the program traces the wire form in-graph) and
        // the driver resolves it pre-forward, so this pass leases physical
        // pages + fires solo/prebuilt via `map_geometry_relaxed` — but it
        // RUNS AHEAD like any pass (the FIFO carries it; NOT synchronous like
        // the deleted host-replay beam branch).
        if ctx.resources().get(&fwd)?.devgeo.is_some() {
            return fire_device_geometry(ctx, this, fwd).await;
        }
        // W3.1: the PIPELINE owns the in-flight FIFO. Point each of this
        // pass's channels at this pipeline's queue so their `take`/`read`
        // await the right FIFO — enforcing the same-pipeline constraint
        // (§3.4): every pass binding a channel must submit on one pipeline.
        let pipe_fires = ctx.resources().get(&this)?.fires.clone();
        let pipeline_failure = ctx.resources().get(&this)?.failure.clone();
        // Non-blocking settlement drain (plan §6): resolved fires' KV/RS
        // txns finalize here so arena pins stay bounded by run-ahead depth
        // even when the guest never takes.
        drain_settled(ctx, Some(&pipe_fires)).await?;
        if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
            return Ok(Err(format!("pipeline: pipeline failed: {reason}")));
        }
        if let Err(error) = wire_channels_to_pipeline(ctx, &fwd, &pipe_fires)? {
            return Ok(Err(error));
        }
        let (
            geometry,
            cells,
            ws_rep,
            rs_rep,
            committed_tokens,
            fwd_rep,
            instance_id,
            completion,
            canonical_evidence,
            dense_mask,
            accesses,
        ) = {
            let p = ctx.resources().get_mut(&fwd)?;
            if let Some(e) = &p.failed {
                return Ok(Err(format!(
                    "pipeline: forward-pass failed by an earlier fire: {e}"
                )));
            }
            let geometry = p
                .instance
                .fire_geometry(crate::pipeline::program::model_profile().page_size)
                .ok();
            let canonical_evidence = kv::canonical_fire_evidence(
                p.canonical_kv,
                &p.instance.program.bound.container,
                &p.cells,
                &p.instance.seeds,
                p.fired_once,
            );
            let accesses = channel_accesses(&p.instance.program.bound.container);
            (
                geometry,
                p.cells.clone(),
                p.kv_ws,
                p.rs_ws,
                p.committed_tokens,
                fwd.rep(),
                p.bound_instance.instance_id,
                p.bound_instance.reserve_completion(),
                canonical_evidence,
                p.dense_mask,
                accesses,
            )
        };
        let mut req = crate::driver::LaunchPlan::default();
        if let Some(g) = &geometry {
            g.apply_to(&mut req);
        }
        // A dense device mask (AttnMask channel) marks the fire
        // mask-carrying: the scheduler batches it SOLO (the composed
        // multi-program batch does not merge dense device masks — v1
        // scope).
        req.has_user_mask = dense_mask;
        let (ticket_heads, ticket_tails) = reserve_channel_tickets(&cells, &accesses);
        req.channel_expected_head = ticket_heads.clone();
        req.channel_expected_tail = ticket_tails.clone();
        // Prepare the guest-owned KV working set for this fire via
        // `pipeline::fire::kv` over the typed KvStore (reserve growth +
        // fresh / in-place / CoW classification + geometry projection).
        // The held `KvTxn` rides the PendingFire across the async fire;
        // finalized (commit → mapping publishes / abort → pending slots
        // release) when a take/read/drop drains it.
        let new_tokens: Vec<u32> = req.token_ids.clone();
        // Canonical gate, fire-time half: hash under the host-verified
        // token values only when they cover exactly this fire's embed
        // AND the pass attends the FULL context. Anything else (device-
        // carried decode tokens, partial/unknown kv-len) hashes opaque.
        let hash_tokens: Option<Vec<u32>> = canonical_evidence.and_then(|(toks, kv_len)| {
            (toks.len() == new_tokens.len()
                && kv_len == Some(committed_tokens + new_tokens.len() as u32))
            .then_some(toks)
        });
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let ws = ctx.resources().get(&ws_res)?.clone();
        let stores = crate::store::registry::get(ws.model, ws.driver as usize);
        let pid = ctx.process_id();
        if let Err(owner) = ws.claim_pipeline_scope(pid.as_u128()) {
            return Ok(Err(format!(
                "pipeline: KV working set is already scoped to pipeline {owner:032x}"
            )));
        }
        let ws_guard = match ws.fire_lease() {
            Ok(lease) => lease,
            Err(error) => return Ok(Err(format!("pipeline: KV working set: {error}"))),
        };

        // A pass's optimistic cursor covers its own pending fires. Another
        // pass on the same pipeline may also have prepared into this working
        // set, so floor it by the pipeline-scoped committed+pending view.
        // Other pipelines and the prefix CAS continue to see committed state.
        //
        // ANOTHER pass may have committed into the same working set
        // before this pass's first fire (a prefill pass feeding a decode
        // pass is the norm). Floor the cursor by the store's published
        // extent so prepare appends after the real committed content
        // instead of re-writing (CoW-forking) published slots. The
        // cursor stays authoritative when it runs AHEAD (this pass's
        // in-flight fires).
        let committed_tokens = {
            let kv = stores.kv.lock().unwrap();
            committed_tokens.max(kv.visible_token_len(ws.id, ws.page_size).unwrap_or(0) as u32)
        };

        // ── Prefix-cache graft (kv_refact.md, the matching consumer) ──
        // First canonical fire of a FRESH working set: probe the CAS for
        // the longest cached full-page prefix of this fire's embed, adopt
        // it (structurally shared pages — writes CoW like any shared
        // path), and TRIM the launch to the uncached suffix. A pure
        // execution-strategy swap, invisible to the guest: the total
        // attended context (KvLen) is unchanged, the kv projection covers
        // the full context from the store, sampled rows keep their
        // identity (shifted to the computed suffix), and the chain state
        // continues from the boundary hash — so the launch is
        // indistinguishable from a continuation fire over `kts` committed
        // tokens. Any miss or ineligible shape falls through to the full
        // computation.
        let (committed_tokens, new_tokens, hash_tokens) = {
            let mut committed = committed_tokens;
            let mut toks_new = new_tokens;
            let mut toks_hash = hash_tokens;
            let n = toks_new.len();
            // Eligible: nothing committed yet, host-verified tokens for
            // the whole embed, and the exact single-lane canonical shape.
            let eligible = committed == 0
                && n > 1
                && req.qo_indptr == [0, n as u32]
                && req.position_ids.len() == n
                && req.sampling_indices.iter().all(|&s| (s as usize) < n);
            if eligible {
                if let Some(toks) = toks_hash.as_deref() {
                    // A sampled row must stay in the computed suffix: cap
                    // the probe so no adopted page covers one.
                    let min_sample = req
                        .sampling_indices
                        .iter()
                        .copied()
                        .min()
                        .map(|s| s as usize)
                        .unwrap_or(usize::MAX);
                    let cap = n.min(min_sample.saturating_add(1));
                    let adopted = {
                        let mut kv = stores.kv.lock().unwrap();
                        kv::match_prefix(&mut kv, ws.id, &toks[..cap], ws.page_size)
                    };
                    match adopted {
                        Ok(Some(pages)) if pages > 0 => {
                            let kts = pages as usize * ws.page_size as usize;
                            debug_assert!(kts < n && kts <= min_sample);
                            req.token_ids.drain(..kts);
                            req.position_ids.drain(..kts);
                            req.qo_indptr = vec![0, (n - kts) as u32];
                            for s in &mut req.sampling_indices {
                                *s -= kts as u32;
                            }
                            toks_new.drain(..kts);
                            toks_hash = toks_hash.map(|t| t[kts..].to_vec());
                            committed = kts as u32;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::debug!("pipeline prefix-cache probe miss: {e}");
                        }
                    }
                }
            }
            (committed, toks_new, toks_hash)
        };

        let next_committed = committed_tokens + new_tokens.len() as u32;
        let kvtxn_slot = Arc::new(Mutex::new(None));

        // The recurrent-state slot for hybrid / linear-attention models
        // (GDN, Mamba2): fresh RESET slab on the first fire, CoW-continue
        // after.
        let rstxn = if let Some(rs_rep) = rs_rep {
            let rs_res: Resource<RsWorkingSet> = Resource::new_borrow(rs_rep);
            let rs = ctx.resources().get(&rs_res)?.clone();
            let prepared = {
                let mut rs_store = stores.rs.lock().unwrap();
                rs::prepare(&mut rs_store, rs.id)
            };
            match prepared {
                Ok((rs_slot_ids, rs_slot_flags, (rs_copy_src, rs_copy_dst), txn)) => {
                    req.rs_slot_ids = rs_slot_ids;
                    req.rs_slot_flags = rs_slot_flags;
                    if !rs_copy_src.is_empty() {
                        // The copy rides the pipeline FIFO (D4): an
                        // asynchronous failure poisons the pipeline
                        // failure domain instead of vanishing in a log.
                        match crate::scheduler::copy_d2d(0, &rs_copy_src, &rs_copy_dst) {
                            Ok(move_completion) => {
                                pipe_fires.lock().unwrap().push_back(PendingOp::Move(
                                    PendingMove {
                                        completion: move_completion,
                                        failure: pipeline_failure.clone(),
                                    },
                                ));
                            }
                            Err(e) => {
                                tracing::warn!("pipeline rs CoW d2d copy failed: {e:#}");
                            }
                        }
                    }
                    Some(txn)
                }
                Err(e) => {
                    return Ok(Err(format!("pipeline: rs prepare: {e}")));
                }
            }
        } else {
            None
        };

        let model = ws.model;
        let driver = ws.driver as usize;
        let ws_id = ws.id;
        let page_size = ws.page_size;
        let deferred_slot = kvtxn_slot.clone();
        let mut deferred_new_tokens = Some(new_tokens);
        let mut deferred_hash_tokens = Some(hash_tokens);
        let acquire_started = Arc::new(AtomicBool::new(false));
        let acquire_result = Arc::new(Mutex::new(None::<Result<(), String>>));
        let runtime = tokio::runtime::Handle::current();
        let preparation: crate::scheduler::LaunchPreparation = Box::new(move |request| {
            let new_tokens = deferred_new_tokens
                .as_ref()
                .expect("dispatch preparation payload remains live");
            let hash_tokens = deferred_hash_tokens
                .as_ref()
                .expect("dispatch preparation hash payload remains live");
            let stores = crate::store::registry::get(model, driver);
            let prepared = {
                let mut kv_store = stores.kv.lock().unwrap();
                kv::prepare(
                    &mut kv_store,
                    ws_id,
                    committed_tokens,
                    new_tokens,
                    page_size,
                    hash_tokens.as_deref(),
                )
            };
            match prepared {
                Ok((projection, (copy_src, copy_dst), translation, txn)) => {
                    request.kv_translation = translation;
                    *deferred_slot.lock().unwrap() = Some(txn);
                    deferred_new_tokens.take();
                    deferred_hash_tokens.take();
                    Ok(crate::scheduler::PreparedLaunch {
                        page_refs: (0..projection.physical_page_ids.len() as u32).collect(),
                        last_page_len: projection.last_page_len,
                        copy_src,
                        copy_dst,
                    })
                }
                Err(kv::KvError::OutOfPages { requested, .. }) => {
                    let mut kv_store = stores.kv.lock().unwrap();
                    let epoch = kv_store.current_epoch();
                    let reclaimed = kv_store.drop_unused_cache_leases(epoch);
                    kv_store.retire_idle();
                    drop(kv_store);
                    if reclaimed == 0
                        && let Some(orchestrator) = crate::store::reclaim::contention()
                    {
                        if let Some(result) = acquire_result.lock().unwrap().take() {
                            acquire_started.store(false, Ordering::Release);
                            if let Err(error) = result {
                                return Err(crate::scheduler::LaunchPreparationError::Failed(
                                    format!("pipeline: KV contention: {error}"),
                                ));
                            }
                        } else if !acquire_started.swap(true, Ordering::AcqRel) {
                            let result = acquire_result.clone();
                            runtime.spawn(async move {
                                let outcome = orchestrator
                                    .acquire(pid, requested as u32)
                                    .await
                                    .map_err(|error| error.to_string());
                                *result.lock().unwrap() = Some(outcome);
                            });
                        }
                    }
                    Err(crate::scheduler::LaunchPreparationError::Retry(
                        "KV pages unavailable at dispatch".to_string(),
                    ))
                }
                Err(error) => Err(crate::scheduler::LaunchPreparationError::Failed(format!(
                    "pipeline: kv dispatch preparation: {error}"
                ))),
            }
        });

        // Fire through the scheduler → dispatch-time KV preparation → driver.
        let submit_error = crate::scheduler::submit_async_deferred(
            req,
            0,
            instance_id,
            Some(pid),
            completion.clone(),
            preparation,
        )
        .err()
        .map(|error| format!("{error:#}"));
        if let Some(error) = submit_error {
            rollback_channel_tickets(&cells, &accesses, &ticket_heads, &ticket_tails);
            let reason = format!("pipeline: submit failed: {error}");
            if let Some(rstxn) = rstxn {
                let mut rs_store = stores.rs.lock().unwrap();
                let _ = rs::finalize(&mut rs_store, rstxn, false);
            }
            ctx.resources().get_mut(&fwd)?.failed = Some(reason.clone());
            return Ok(Err(reason));
        }

        // Optimistic cursor advance: the NEXT submit prepares against this
        // fire's post-state (the run-ahead overlap). A failed fire fails
        // the pass instead of rewinding.
        {
            let p = ctx.resources().get_mut(&fwd)?;
            p.committed_tokens = next_committed;
            p.fired_once = true; // seeds are consumed; the seed rule is off
        }

        pipe_fires
            .lock()
            .unwrap()
            .push_back(PendingOp::Fire(PendingFire {
                completion,
                kv: FireKv::Deferred(kvtxn_slot),
                rstxn,
                ws_guard,
                ws_rep,
                rs_rep,
                fwd_rep,
                instance_id,
                cells,
                failure: pipeline_failure,
            }));
        Ok(Ok(()))
    }
}

/// Compaction (Design-B lazy KV GC): move `n` token KV cells within `ws`,
/// all layers, from (`src_page_ids[i]`, `src_tok_idx[i]`) -> (`dst_page_ids[i]`,
/// `dst_tok_idx[i]`). Enqueues a `PendingOp::Move` on the SAME pipeline FIFO /
/// stream as forward fires (submitted solo/prebuilt), so ordering is automatic
/// (happens-after prior fires' KV writes, happens-before later fires' reads) —
/// no drain barrier (K6 dissolved). The guest's page ids are the PHYSICAL KV
/// block ids it also binds to the `Pages`/`WSlot` ports (Design B works in
/// physical page ids directly), so they pass straight through to the move — the
/// wire / kernel operate on exactly the physical pages the fires read/write.
/// The guest computed the post-move layout itself, so no result is taken.
pub async fn copy_into_inner<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    ws: Resource<KvWorkingSet>,
    dst_page_ids: Vec<u32>,
    dst_tok_idx: Vec<u32>,
    src_page_ids: Vec<u32>,
    src_tok_idx: Vec<u32>,
) -> Anyhow<Result<(), String>> {
    let n = dst_page_ids.len();
    if dst_tok_idx.len() != n || src_page_ids.len() != n || src_tok_idx.len() != n {
        return Ok(Err(format!(
            "pipeline copy_into: the four (dst_page,dst_tok,src_page,src_tok) lists \
                 must be equal length (got {}, {}, {}, {})",
            dst_page_ids.len(),
            dst_tok_idx.len(),
            src_page_ids.len(),
            src_tok_idx.len()
        )));
    }
    if n == 0 {
        return Ok(Ok(()));
    }

    // The WIT contract passes WorkingSet-RELATIVE page indexes (guests
    // never hold physical ids); translate through the flattened table so
    // the move lands on exactly the physical pages the fires read/write.
    // Translated at enqueue against the committed mapping: same-WS
    // in-flight fires that could remap these pages (a CoW rebase) are the
    // guest's ordering hazard, like any same-WS run-ahead write overlap.
    let (kv_move_dst_pages, kv_move_src_pages): (Vec<u32>, Vec<u32>) = {
        let ws = ctx.resources().get(&ws)?.clone();
        let stores = crate::store::registry::get(ws.model, ws.driver as usize);
        let pid = ctx.process_id();
        if let Err(owner) = ws.claim_pipeline_scope(pid.as_u128()) {
            return Ok(Err(format!(
                "pipeline: KV working set is already scoped to pipeline {owner:032x}"
            )));
        }
        let mut kv_store = stores.kv.lock().unwrap();
        let (_, flat) = kv_store
            .flat_table(ws.id)
            .map_err(|e| anyhow::anyhow!("copy_into flat table: {e}"))?;
        let translate = |ids: &[u32]| -> Result<Vec<u32>, String> {
            ids.iter()
                .map(|&i| {
                    flat.get(i as usize).map(|p| p.0).ok_or_else(|| {
                        format!("copy_into: page index {i} beyond the mapped extent")
                    })
                })
                .collect()
        };
        match (translate(&dst_page_ids), translate(&src_page_ids)) {
            (Ok(d), Ok(s)) => (d, s),
            (Err(e), _) | (_, Err(e)) => return Ok(Err(e)),
        }
    };

    // Point this pipeline's FIFO at the move so a later `take`/`read` on any
    // channel fed by this pipeline drains it in submit order.
    let pipe_fires = ctx.resources().get(&this)?.fires.clone();
    let pipeline_failure = ctx.resources().get(&this)?.failure.clone();
    if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
        return Ok(Err(format!("pipeline: pipeline failed: {reason}")));
    }

    let cells = kv_move_dst_pages
        .into_iter()
        .zip(dst_tok_idx.into_iter())
        .zip(kv_move_src_pages.into_iter().zip(src_tok_idx.into_iter()))
        .map(
            |((dst_page_id, dst_token_offset), (src_page_id, src_token_offset))| {
                pie_driver_abi::PieKvMoveCell {
                    dst_page_id,
                    dst_token_offset,
                    src_page_id,
                    src_token_offset,
                }
            },
        )
        .collect::<Vec<_>>();
    let completion = match crate::scheduler::copy_kv_cells(0, cells) {
        Ok(completion) => completion,
        Err(e) => return Ok(Err(format!("pipeline copy_into: submit failed: {e:#}"))),
    };
    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Move(PendingMove {
            completion,
            failure: pipeline_failure,
        }));
    Ok(Ok(()))
}

pub async fn pipeline_close<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    // Signal no further submissions: drain the pipeline's in-flight FIFO so
    // its fires' prepared KV/RS writes (snapshot pins) finalize before the
    // pipeline goes away (the pin-safety drain follows the FIFO — W3.1).
    let fires = ctx.resources().get(&this).ok().map(|p| p.fires.clone());
    if let Some(fires) = fires {
        loop {
            let fire = fires.lock().unwrap().pop_front();
            match fire {
                Some(f) => {
                    let _ = finalize_op(ctx, f).await;
                }
                None => break,
            }
        }
    }
    Ok(())
}

pub async fn pipeline_drop<C: FireContext>(ctx: &mut C, this: Resource<Pipeline>) -> Anyhow<()> {
    // Drain the pipeline's in-flight FIFO before releasing it: each fire
    // holds a prepared KV/RS write (snapshot pins, pending slots) and the
    // GPU may still be writing — await + finalize, never abandon
    // mid-flight (W3.1: the pin-safety drain lives here, not on the pass).
    let fires = ctx.resources().get(&this).ok().map(|p| p.fires.clone());
    if let Some(fires) = fires {
        loop {
            let fire = fires.lock().unwrap().pop_front();
            match fire {
                Some(f) => {
                    let _ = finalize_op(ctx, f).await;
                }
                None => break,
            }
        }
    }
    ctx.resources().delete(this)?;
    Ok(())
}

/// The body behind `kv-working-set.copy-into(on, ...)` (called from
/// `inferlet::host::kv_working_set`): an ordered KV cell move on the pipeline
/// FIFO.
pub async fn working_set_copy_into<C: FireContext>(
    ctx: &mut C,
    ws: Resource<KvWorkingSet>,
    on: Resource<Pipeline>,
    dst_page_ids: Vec<u32>,
    dst_tok_idx: Vec<u32>,
    src_page_ids: Vec<u32>,
    src_tok_idx: Vec<u32>,
) -> Anyhow<Result<(), String>> {
    copy_into_inner(
        ctx,
        on,
        ws,
        dst_page_ids,
        dst_tok_idx,
        src_page_ids,
        src_tok_idx,
    )
    .await
}

/// Drain one pipeline FIFO entry in submit order: a forward fire finalizes
/// its KV/RS txns and exposes mirror epochs; a KV cell MOVE awaits its
/// payload-free completion. Move failures are logged because no channel is
/// associated with the operation.
/// Pop and finalize pipeline ops whose completions have already settled,
/// without blocking (plan §6): submit and take/read entry call this so
/// KV/RS transaction pins stay bounded by run-ahead depth while value
/// waiting rides the channel wait slots. Returns whether anything drained.
pub async fn drain_settled<C: FireContext>(
    ctx: &mut C,
    fires: Option<&PendingFires>,
) -> Anyhow<bool> {
    let Some(fires) = fires else {
        return Ok(false);
    };
    let mut drained = false;
    loop {
        let op = {
            let mut queue = fires.lock().unwrap();
            if queue.front().is_some_and(PendingOp::is_settled) {
                queue.pop_front()
            } else {
                None
            }
        };
        match op {
            Some(op) => {
                finalize_op(ctx, op).await?;
                drained = true;
            }
            None => return Ok(drained),
        }
    }
}

pub async fn finalize_op<C: FireContext>(ctx: &mut C, op: PendingOp) -> Anyhow<()> {
    match op {
        PendingOp::Fire(fire) => finalize_fire(ctx, fire).await,
        PendingOp::Move(mv) => {
            if let Err(e) = mv.completion.await {
                let reason = format!("pipeline kv-move (copy_into) failed: {e:#}");
                let mut failure = mv.failure.lock().unwrap();
                if failure.is_none() {
                    *failure = Some(reason.clone());
                }
                tracing::warn!("{reason}");
            }
            Ok(())
        }
    }
}

/// Resolve one in-flight fire: await the payload-free callback, finalize the
/// KV/RS txns, and expose the release-published mirror tails. Values remain
/// in driver-owned channel memory until `channel.take` or `channel.read`.
async fn finalize_fire<C: FireContext>(ctx: &mut C, fire: PendingFire) -> Anyhow<()> {
    let PendingFire {
        completion,
        kv,
        rstxn,
        ws_guard,
        ws_rep,
        rs_rep,
        fwd_rep,
        instance_id,
        cells,
        failure,
    } = fire;
    let prior_failure = failure.lock().unwrap().clone();
    let result = completion.await;
    let success = result.is_ok() && prior_failure.is_none();

    {
        // Single-model, single-driver v1: fires are keyed to stores (0,0).
        let _ = ws_rep;
        let stores = crate::store::registry::get(0, 0);
        match kv {
            FireKv::DeviceGeom { kvtxn } => {
                let mut kv_store = stores.kv.lock().unwrap();
                let _ = kv::finalize(&mut kv_store, kvtxn, success);
            }
            FireKv::Deferred(slot) => {
                if let Some(kvtxn) = slot.lock().unwrap().take() {
                    let mut kv_store = stores.kv.lock().unwrap();
                    let _ = kv::finalize(&mut kv_store, kvtxn, success);
                }
            }
        }
        if let Some(rstxn) = rstxn {
            let _ = rs_rep;
            let mut rs_store = stores.rs.lock().unwrap();
            let _ = rs::finalize(&mut rs_store, rstxn, success);
        }
    } // store locks released before the contention drain re-locks pools

    // The fire's sequence retired: recycled slots (aborts, CoW'd tails,
    // collected suffixes) are allocatable now — wake parked allocators
    // and drain-gated lanes.
    if let Some(orch) = crate::store::reclaim::contention() {
        orch.on_blocks_freed();
        orch.on_fire_retired();
    }

    if let Some(reason) = prior_failure {
        poison_readers(&cells, &reason);
        fail_pass(ctx, fwd_rep, &reason);
        return Ok(());
    }

    // Values are already visible through the release-published tail words
    // (plan §4.5) — resolving the fire only classifies success and settles
    // the transactions above.
    let failure_reason = match result {
        Ok(()) => {
            reclaim_device_geometry_grants(ctx, fwd_rep, instance_id);
            None
        }
        Err(e) => {
            let reason = format!("pipeline: forward failed: {e:#}");
            poison_readers(&cells, &reason);
            fail_pass(ctx, fwd_rep, &reason);
            Some(reason)
        }
    };
    if let Some(reason) = failure_reason {
        let mut domain = failure.lock().unwrap();
        if domain.is_none() {
            *domain = Some(reason);
        }
    }
    drop(ws_guard);
    Ok(())
}

/// Mark a pass failed (first failure wins). The guest may have dropped
/// the pass handle already — then there is nothing to mark.
fn fail_pass<C: FireContext>(ctx: &mut C, fwd_rep: u32, reason: &str) {
    let res: Resource<ForwardPass> = Resource::new_borrow(fwd_rep);
    if let Ok(p) = ctx.resources().get_mut(&res) {
        if p.failed.is_none() {
            p.failed = Some(reason.to_string());
        }
    }
}

/// Device-geometry fire (Track B): the pass's [B,P] geometry is
/// DEVICE-produced (the program traces `page_indptr = CumSum(np)` + packed
/// live pages in-graph) and the driver resolves it pre-forward, so the host
/// neither replays the epilogue arithmetic nor projects per-lane KV. The
/// runtime leases `B` fresh physical pages, delivers them to the program as a
/// host-put on the `fresh` channel, marks the fire solo/prebuilt via
/// `map_geometry_relaxed` (wire fields empty), and fires it RUN-AHEAD onto the
/// pipeline FIFO (unlike the deleted synchronous host-replay beam branch).
/// The per-fire arena/write txns ride the `PendingFire`; `finalize_fire`
/// commits/aborts them and reclaims continuing heirs' unused grants (w_cont).
///
/// BRING-UP (4090, shadow-verify): the exact fresh-page materialization
/// (`cow_write_slot`), the fire-0 seed source, and the physical-page ids fed
/// on `fresh` are validated against the beam goldens on device; the geometry
/// contract itself is host-verified (`ptir-dsl` `beam_designb_goldens`).
async fn fire_device_geometry<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
) -> Anyhow<Result<(), String>> {
    // Wire each of this pass's channels at this pipeline's FIFO (§3.4: all
    // passes binding a channel must submit on ONE pipeline — the entire
    // ordering/FIFO correctness argument).
    let pipe_fires = ctx.resources().get(&this)?.fires.clone();
    let pipeline_failure = ctx.resources().get(&this)?.failure.clone();
    // Non-blocking settlement drain (plan §6), as in the ordinary submit.
    drain_settled(ctx, Some(&pipe_fires)).await?;
    if let Some(reason) = pipeline_failure.lock().unwrap().clone() {
        return Ok(Err(format!("pipeline: pipeline failed: {reason}")));
    }
    if let Err(e) = wire_channels_to_pipeline(ctx, &fwd, &pipe_fires)? {
        return Ok(Err(e));
    }

    let ws_rep = ctx.resources().get(&fwd)?.kv_ws;
    let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
    let ws = ctx.resources().get(&ws_res)?.clone();
    let page_size = ws.page_size;
    let stores = crate::store::registry::get(ws.model, ws.driver as usize);
    let pid = ctx.process_id();
    if let Err(owner) = ws.claim_pipeline_scope(pid.as_u128()) {
        return Ok(Err(format!(
            "pipeline: KV working set is already scoped to pipeline {owner:032x}"
        )));
    }
    let ws_guard = match ws.fire_lease() {
        Ok(lease) => lease,
        Err(error) => return Ok(Err(format!("pipeline: KV working set: {error}"))),
    };

    // Grant B pages, allocating their physical backing under one prepared
    // write (no borrow held across submit).
    let (
        completion,
        instance_id,
        cells,
        fwd_rep,
        kvtxn,
        kv_translation,
        wire_pages,
        copy_src,
        copy_dst,
        dense_mask,
        accesses,
    ) = {
        // Fail-fast.
        {
            let p = ctx.resources().get_mut(&fwd)?;
            if let Some(e) = &p.failed {
                return Ok(Err(format!(
                    "pipeline: forward-pass failed by an earlier fire: {e}"
                )));
            }
        }

        // Take the DevGeo out so the lease grant can use the store
        // (distinct table resources can't be borrowed mutably at once).
        let mut devgeo: DevGeo = ctx
            .resources()
            .get_mut(&fwd)?
            .devgeo
            .take()
            .expect("fire_device_geometry on a non-device-geometry pass");

        // Grant B slots: lease free-list first, then fresh logical
        // reserves. Purely logical — this can never exhaust the pool, so
        // it runs ONCE; only the physical prepare below retries under
        // contention.
        let grant_slots = {
            let mut kv_store = stores.kv.lock().unwrap();
            devgeo.lease.grant(|| {
                kv_store
                    .reserve(ws.id, 1)
                    .map(|r| r.start as u32)
                    .unwrap_or(0)
            })
        };
        let (pages, (copy_src, copy_dst), kv_translation, kvtxn) = loop {
            let prepared = {
                let mut kv_store = stores.kv.lock().unwrap();
                // One prepared write covers the grants plus any unwritten
                // gap up to the mapped end (left by an aborted earlier
                // fire), so fresh publication stays contiguous. The gap
                // is recomputed per attempt: a parked retry may resume
                // after another of this pass's fires committed.
                let mapped = kv_store
                    .visible_flat_table(ws.id)
                    .map_or(0, |table| table.len() as u64);
                let mut write_indexes: Vec<u64> = grant_slots.iter().map(|&s| s as u64).collect();
                if let Some(max_fresh) =
                    write_indexes.iter().copied().filter(|&i| i >= mapped).max()
                {
                    write_indexes.extend(mapped..max_fresh);
                }
                write_indexes.sort_unstable();
                write_indexes.dedup();
                kv::prepare_explicit(&mut kv_store, ws.id, &write_indexes)
            }; // kv lock dropped before any contention await below
            match prepared {
                Ok(v) => break v,
                Err(e @ kv::KvError::OutOfPages { requested, .. }) => {
                    // Same contention-ladder seam as the host-geometry
                    // path (see submit_pass), including the Error-mode
                    // inline rung 1.
                    let Some(orch) = crate::store::reclaim::contention() else {
                        let freed = {
                            let mut kv_store = stores.kv.lock().unwrap();
                            let epoch = kv_store.current_epoch();
                            let freed = kv_store.drop_unused_cache_leases(epoch);
                            kv_store.retire_idle();
                            freed
                        };
                        if freed > 0 {
                            continue;
                        }
                        ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                        return Ok(Err(format!("pipeline: device-geometry grant: {e}")));
                    };
                    if let Err(hard) = orch.acquire(pid, requested as u32).await {
                        ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                        return Ok(Err(format!("pipeline: device-geometry grant: {hard}")));
                    }
                }
                Err(e) => {
                    ctx.resources().get_mut(&fwd)?.devgeo = Some(devgeo);
                    return Ok(Err(format!("pipeline: device-geometry grant: {e}")));
                }
            }
        };
        // Wire page refs (scheduler capacity accounting): this fire's
        // physical write targets.
        let wire_pages: Vec<u32> = pages.iter().map(|&(_, dst)| dst).collect();
        // Deliver the fresh grant to the program as a direct put on its
        // `fresh` channel — a shared-ring write the driver pulls before
        // the pass (plan §4.2/§4.3). The grants are WorkingSet-RELATIVE
        // indexes — the program's in-graph geometry stays logical
        // end-to-end and the driver translates its resolved
        // `Pages`/`WSlot` values through `kv_translation` (which the
        // prepared write above backs physically). This also matches the
        // bind-time seed (`reserve(b)`), which was already logical.
        let fresh_dense = devgeo.fresh_dense;
        let dense_mask = devgeo.has_mask;
        let fresh_error = {
            let p = ctx.resources().get_mut(&fwd)?;
            let bytes: Vec<u8> = grant_slots.iter().flat_map(|s| s.to_le_bytes()).collect();
            let error = match p.cells.get(fresh_dense) {
                Some(cell) => cell.lock().unwrap().put(bytes).err(),
                None => Some(ChannelError::Empty),
            };
            p.devgeo = Some(devgeo);
            error
        };
        if let Some(error) = fresh_error {
            let mut kv_store = stores.kv.lock().unwrap();
            let _ = kv::finalize(&mut kv_store, kvtxn, false);
            return Ok(Err(format!(
                "pipeline: device-geometry fresh grant put: {error}"
            )));
        }

        let p = ctx.resources().get_mut(&fwd)?;
        let completion = p.bound_instance.reserve_completion();
        let accesses = channel_accesses(&p.instance.program.bound.container);
        (
            completion,
            p.bound_instance.instance_id,
            p.cells.clone(),
            fwd.rep(),
            kvtxn,
            kv_translation,
            wire_pages,
            copy_src,
            copy_dst,
            dense_mask,
            accesses,
        )
    };

    let mut req = crate::driver::LaunchPlan::default();
    req.kv_translation = kv_translation;
    // A dense device mask (AttnMask channel) marks the fire mask-carrying;
    // the scheduler batches it SOLO (the composed multi-program batch
    // does not merge dense device masks — v1 scope).
    req.has_user_mask = dense_mask;
    let (ticket_heads, ticket_tails) = reserve_channel_tickets(&cells, &accesses);
    req.channel_expected_head = ticket_heads.clone();
    req.channel_expected_tail = ticket_tails.clone();
    let last_page_len = wire_pages.last().map(|_| page_size).unwrap_or(0);
    let submit_error = crate::scheduler::submit_prebuilt_async_with_kv_copy(
        req,
        0,
        instance_id,
        wire_pages,
        last_page_len,
        completion.clone(),
        copy_src,
        copy_dst,
    )
    .err()
    .map(|error| format!("{error:#}"));
    if let Some(error) = submit_error {
        rollback_channel_tickets(&cells, &accesses, &ticket_heads, &ticket_tails);
        let reason = format!("pipeline: device-geometry submit failed: {error}");
        {
            let mut kv_store = stores.kv.lock().unwrap();
            let _ = kv::finalize(&mut kv_store, kvtxn, false);
        }
        ctx.resources().get_mut(&fwd)?.failed = Some(reason.clone());
        return Ok(Err(reason));
    }
    ctx.resources().get_mut(&fwd)?.fired_once = true;

    pipe_fires
        .lock()
        .unwrap()
        .push_back(PendingOp::Fire(PendingFire {
            completion,
            kv: FireKv::DeviceGeom { kvtxn },
            rstxn: None,
            ws_guard,
            ws_rep,
            rs_rep: None,
            fwd_rep,
            instance_id,
            cells,
            failure: pipeline_failure,
        }));
    Ok(Ok(()))
}

/// Point each of a pass's channels at `pipe_fires` (the feeding pipeline's
/// FIFO), enforcing the same-pipeline invariant (§3.4). Returns `Ok(Err(..))`
/// if a channel is already bound to a DIFFERENT pipeline.
fn wire_channels_to_pipeline<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
    pipe_fires: &PendingFires,
) -> Anyhow<Result<(), String>> {
    if let Some(existing) = &ctx.resources().get(fwd)?.fires {
        if !Arc::ptr_eq(existing, pipe_fires) {
            return Ok(Err(
                "pipeline: a pass cannot submit across different pipelines".into(),
            ));
        }
    }
    let reps = ctx.resources().get(fwd)?.channel_reps.clone();
    for rep in reps {
        let cres: Resource<Channel> = Resource::new_borrow(rep);
        if let Ok(ch) = ctx.resources().get_mut(&cres) {
            match &ch.fires {
                Some(existing) if !Arc::ptr_eq(existing, pipe_fires) => {
                    return Ok(Err("pipeline: a channel is shared across pipelines \
                         (all passes binding a channel must submit on the same \
                         pipeline)"
                        .into()));
                }
                _ => ch.fires = Some(pipe_fires.clone()),
            }
        }
    }
    ctx.resources().get_mut(fwd)?.fires = Some(pipe_fires.clone());
    Ok(Ok(()))
}

/// Device-geometry per-fire page reclaim: read the harvested `w_cont`
/// (`[B]` bool: heir(true)/fork(false)) from its bound mirror, reclaim the
/// continuing heirs' UNUSED fresh page grants into the lease free-list, and
/// free those ws slots. No-op for a non-device-geometry pass.
fn reclaim_device_geometry_grants<C: FireContext>(ctx: &mut C, fwd_rep: u32, instance_id: u64) {
    let res: Resource<ForwardPass> = Resource::new_borrow(fwd_rep);
    let (ws_rep, reclaimed) = {
        let Ok(p) = ctx.resources().get_mut(&res) else {
            return;
        };
        let Some(devgeo) = p.devgeo.as_mut() else {
            return;
        };
        let Some(cell) = p.cells.get(devgeo.w_cont_dense) else {
            return;
        };
        let w_cont = cell
            .lock()
            .unwrap()
            .latest_reader_value(instance_id)
            .ok()
            .flatten()
            .unwrap_or_default();
        let w_cont: Vec<bool> = w_cont.iter().map(|&byte| byte != 0).collect();
        let reclaimed = devgeo.lease.reclaim_after_fire(&w_cont);
        (p.kv_ws, reclaimed)
    };
    // Reclaimed grants return to the lease free-list (done above) and are
    // re-granted to later fires; the store mapping keeps their committed
    // pages until the working set discards or drops them (a discard here
    // would shift live indexes under the pass — see the pass-drop note).
    let _ = (ws_rep, reclaimed);
}
