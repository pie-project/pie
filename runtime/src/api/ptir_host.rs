//! Host wiring for the `ptir` WIT interface (thrust-3 P2b).
//!
//! Behaviour is gated behind the `ptir` cargo feature (manager decision (a): the
//! interface is present in the world unconditionally — additive, no-op for
//! legacy guests — while the impl is inert when the feature is off). When on,
//! `register-program` binds + caches via [`ptir_registry`](super::ptir_registry)
//! and `pipeline.instantiate` validates seeds via
//! [`ptir_instance`](super::ptir_instance); channel host I/O + `submit` land with
//! the channel store (P3).

use std::sync::{Arc, Mutex};

use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

use crate::api::pie;
use crate::inference::forward_prepare;
use crate::instance::InstanceState;
use crate::working_set::kv::KvWorkingSet;

use super::ptir_channel_store::ChannelStore;
use super::ptir_kv;

/// A host endpoint on an instance's host-facing channel (overview §1). Holds a
/// shared handle to the owning pipeline's [`ChannelStore`] plus its channel
/// index, so `put`/`take`/`read` route to the one per-instance store.
pub struct Channel {
    pub store: Arc<Mutex<ChannelStore>>,
    pub index: u32,
}

/// A run-ahead pipeline over one instance of a registered program.
pub struct Pipeline {
    #[allow(dead_code)]
    pub instance: super::ptir_instance::PtirInstance,
    /// The per-instance host channel store (P3): Writer puts staged here are
    /// D1-coalesced into each fire's carrier; Reader cells the fire produces are
    /// marshaled back here for the guest to `take`/`read`.
    pub store: Arc<Mutex<ChannelStore>>,
    /// The instance's persistent KV working set (the model forward writes the
    /// embedded token's K/V here + self-attends over it). Allocated at
    /// instantiation, projected each fire; the KV survives across fires like the
    /// channel arena (persistent-instance model). Without allocated pages the
    /// attention kernels illegal-read (the §6.2 case-b crash).
    pub kv_ws: Resource<KvWorkingSet>,
    /// The persistent ws's committed token length — the growing cursor threaded
    /// into [`ptir_kv::ptir_kv_prepare`]. Starts at 0 (fresh); advances by each
    /// fire's `new_tokens` on commit (single-token decode, MTP K drafts, …).
    /// Unused on the beam path (`fire_beam` owns its own [B,P] geometry).
    pub committed_tokens: u32,
    /// First-fire byte-ship tracking: the container bytes ride the first fire of
    /// this hash; steady-state fires carry the hash only (driver hash-cache).
    #[allow(dead_code)]
    pub shipped: bool,
    /// §6.2 beam host-replay state (Design X). `Some` iff this is a beam program
    /// whose [B,P] geometry is device-produced (`fire_geometry` can't resolve it)
    /// — then each `submit` fires the replayed multi-lane batch instead of the
    /// single-page projection.
    pub beam: Option<BeamRun>,
}

/// Per-instance beam replay state carried across fires (Design X).
pub struct BeamRun {
    /// The host-tracked freeze/heir geometry state.
    pub state: super::ptir_beam::BeamState,
    /// The [B,P] geometry for the NEXT fire (seeded for fire 0; re-derived by
    /// `BeamState::step` after each fire from the harvested `out_par`).
    pub geom: super::ptir_beam::BeamGeometry,
    /// `[B]` tokens to embed next fire (seeded prompt; then the harvested `out`).
    pub toks: Vec<u32>,
    /// `[B]` decode positions (advance by 1 per fire).
    pub pos: Vec<u32>,
}

/// Detect a §6.2-style beam program: its geometry ports (`Pages`/`KvLen`) bind
/// DEVICE-produced channels `fire_geometry` can't resolve. Returns `(B, P)` read
/// from the `pages` channel's `[B, P]` shape (channel 0 by the beam convention).
#[cfg(feature = "ptir")]
fn detect_beam(instance: &super::ptir_instance::PtirInstance) -> Option<(usize, usize)> {
    use pie_sampling_ir::ptir::container::PortSource;
    use pie_sampling_ir::ptir::registry::Port;
    let container = &instance.program.bound.container;
    // STRUCTURAL beam signal: the WSlot/WOff write descriptors are beam-specific
    // — a plain decode's `attn_working_set` binds only KvLen. This is robust to
    // seeding (unlike `fire_geometry`'s MissingChannelValue, which host-known
    // seeds for pages/klen/w_slot defeat → the beam would take the trivial path).
    let has_write_desc = container
        .ports
        .iter()
        .any(|p| matches!(p.port, Port::WSlot | Port::WOff));
    if !has_write_desc {
        return None;
    }
    // B, P from the [B, P] channel bound to the `Pages` port (P > 1 for a beam).
    let pages_ch = container.ports.iter().find_map(|p| match (&p.port, &p.source) {
        (Port::Pages, PortSource::Channel(c)) => Some(*c as usize),
        _ => None,
    })?;
    let dims = container.channels.get(pages_ch)?.shape.dims();
    if dims.len() == 2 && dims[1] > 1 {
        Some((dims[0] as usize, dims[1] as usize))
    } else {
        None
    }
}

type Anyhow<T> = anyhow::Result<T>;

/// The "feature off" guest error.
#[cfg(not(feature = "ptir"))]
fn disabled<T>() -> Anyhow<Result<T, String>> {
    Ok(Err("ptir: the runtime `ptir` feature is disabled".into()))
}

impl pie::core::ptir::Host for InstanceState {
    async fn register_program(&mut self, container_bytes: Vec<u8>) -> Anyhow<Result<u64, String>> {
        #[cfg(not(feature = "ptir"))]
        {
            let _ = container_bytes;
            return disabled();
        }
        #[cfg(feature = "ptir")]
        {
            match super::ptir_registry::register(container_bytes, &model_profile()) {
                Ok(prog) => Ok(Ok(prog.hash)),
                Err(e) => Ok(Err(e.to_string())),
            }
        }
    }
}

impl pie::core::ptir::HostPipeline for InstanceState {
    async fn instantiate(
        &mut self,
        program: u64,
        seeds: Vec<pie::core::ptir::ChannelSeed>,
    ) -> Anyhow<Result<Resource<Pipeline>, String>> {
        #[cfg(not(feature = "ptir"))]
        {
            let _ = (program, seeds);
            return disabled();
        }
        #[cfg(feature = "ptir")]
        {
            let seeds = seeds
                .into_iter()
                .map(|s| super::ptir_instance::ChannelSeed { channel: s.channel, data: s.value })
                .collect();
            match super::ptir_instance::instantiate(program, seeds) {
                Ok(instance) => {
                    let store = ChannelStore::new(&instance.program.bound.container);
                    // Allocate the instance's persistent KV working set. A normal
                    // program starts EMPTY — `ptir_kv::ptir_kv_prepare` grows it per
                    // fire (the growing-KV decode/MTP lifecycle). A §6.2 beam eagerly
                    // reserves B*P slots (the runtime owns the [B,P] layout: beam
                    // `l`'s pages occupy [l*P, l*P+P)) + carries the replay state.
                    let page_size = crate::page_size::tokens_per_page(0);
                    let beam_bp = detect_beam(&instance);
                    let mut ws = KvWorkingSet::new(page_size, 0);
                    let beam_run = match beam_bp {
                        Some((b, p)) => {
                            if let Err(e) = ws.alloc((b * p) as u32) {
                                return Ok(Err(format!("ptir: beam kv working-set alloc: {e}")));
                            }
                            let slot0: Vec<u32> = (0..b).map(|l| (l * p) as u32).collect();
                            let state =
                                super::ptir_beam::BeamState::seeded(b, p, page_size, &slot0);
                            let geom = state.geometry();
                            Some(BeamRun {
                                state,
                                geom,
                                // Placeholder prompt token per beam (the toks seed
                                // in echo's beam_trace); the real prompt is refined
                                // during the 4090 bring-up.
                                toks: vec![1u32; b],
                                pos: vec![0u32; b],
                            })
                        }
                        None => {
                            // Non-beam: the ws starts EMPTY — `ptir_kv::ptir_kv_prepare`
                            // grows it (one-or-more slots) per fire and threads the
                            // `committed_tokens` cursor. (The beam branch above keeps
                            // its eager B*P alloc — `fire_beam` owns that layout.)
                            None
                        }
                    };
                    let kv_ws = self.ctx().table.push(ws)?;
                    let res = self.ctx().table.push(Pipeline {
                        instance,
                        store: Arc::new(Mutex::new(store)),
                        kv_ws,
                        committed_tokens: 0,
                        shipped: false,
                        beam: beam_run,
                    })?;
                    Ok(Ok(res))
                }
                Err(e) => Ok(Err(e.to_string())),
            }
        }
    }

    async fn channel(
        &mut self,
        this: Resource<Pipeline>,
        index: u32,
    ) -> Anyhow<Result<Resource<Channel>, String>> {
        #[cfg(not(feature = "ptir"))]
        {
            let _ = (this, index);
            return disabled();
        }
        #[cfg(feature = "ptir")]
        {
            let store = self.ctx().table.get(&this)?.store.clone();
            if !store.lock().unwrap().contains(index) {
                return Ok(Err(format!("ptir: channel {index} is not a host-facing channel")));
            }
            let res = self.ctx().table.push(Channel { store, index })?;
            Ok(Ok(res))
        }
    }

    async fn submit(&mut self, this: Resource<Pipeline>) -> Anyhow<Result<(), String>> {
        #[cfg(not(feature = "ptir"))]
        {
            let _ = this;
            return disabled();
        }
        #[cfg(feature = "ptir")]
        {
            // §6.2 beam: the [B,P] geometry is device-produced (`fire_geometry`
            // can't resolve it) → the host-replay fire path (Design X) instead of
            // the single-page projection below.
            if self.ctx().table.get_mut(&this)?.beam.is_some() {
                return self.fire_beam(this).await;
            }
            // Build the PTIR carrier for this fire (thrust-3 P2c host emit): ship
            // the container bytes on the first fire of this instance, the hash +
            // instance id only thereafter (driver compile-cache + persistent
            // arena); attach the per-instance seeds (first fire) and the
            // D1-coalesced host-puts drained from the channel store (P3).
            let (submission, geometry, store, ws_rep, committed_tokens) = {
                let p = self.ctx().table.get_mut(&this)?;
                let ship = !p.shipped;
                p.shipped = true;
                let host_puts = p.store.lock().unwrap().drain_host_puts();
                let submission = p.instance.submission(ship, host_puts);
                // Host-known geometry prefill (token/positions/qo/readout for
                // seeded ports, e.g. a §3 single-seq decode). `None` ⇒ a port
                // binds a device-derived / ws / run-ahead channel the host can't
                // resolve — the driver fills the descriptor ports itself.
                let geometry = p.instance.fire_geometry(model_profile().page_size).ok();
                // Clone the store handle + grab the ws rep + the growing cursor out
                // of the table BEFORE awaiting the fire (no table borrow across await).
                (submission, geometry, p.store.clone(), p.kv_ws.rep(), p.committed_tokens)
            };
            let mut req = pie_driver_abi::ForwardRequest::default();
            req.push_ptir_program(&submission);
            if let Some(g) = &geometry {
                g.apply_to(&mut req);
            }

            // Project the persistent KV working set for this fire via `ptir_kv`
            // (alpha's ws-alloc + arena-txn + project_kv lifecycle): grow the ws,
            // resolve_read + pin prior context (`committed_tokens`), cow_write_slot
            // the new-token slots, project the real driver page geometry. Held
            // `PtirKvTxn` crosses the async fire; finalized (commit → KV persists /
            // abort → revert) after. Generalizes the single-token fresh prefill to
            // prior-context + multi-token growth (decode loops, MTP K drafts).
            let new_tokens: Vec<u32> = req.token_ids.clone();
            let page_size = crate::page_size::tokens_per_page(0);
            let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
            let arena_arc = crate::arena::get(0, 0);
            let prepared = {
                let mut arena = arena_arc.lock().unwrap();
                let ws = self.ctx().table.get_mut(&ws_res)?;
                ptir_kv::ptir_kv_prepare(ws, committed_tokens, &new_tokens, &mut arena, page_size)
            };
            let (proj, move_plans, kvtxn) = match prepared {
                Ok(v) => v,
                Err(e) => return Ok(Err(format!("ptir: kv prepare: {e}"))),
            };
            let next_committed = kvtxn.committed_tokens_after;

            // D2D-copy every CoW'd write target before the fire (empty for the
            // single-context pipeline; non-empty only under a forked/shared page).
            for mp in &move_plans {
                if let Err(e) = crate::driver::copy_d2d(0, &mp.from, &mp.to) {
                    tracing::warn!("ptir forward CoW d2d copy failed: {e:#}");
                }
            }

            // Fire through the scheduler → charlie's PTIR executor hook: the
            // forward writes the token K/V into the projected page + produces
            // `ws.logits`, then the hook decodes/instantiates (keyed by
            // `submission.instance`)/fires/harvests → `resp.ptir_output_*`.
            let rx = match crate::inference::submit_async(
                req,
                0,
                proj.physical_page_ids,
                proj.last_page_len,
                Vec::new(),
                None,
            ) {
                Ok(rx) => rx,
                Err(e) => {
                    let mut arena = arena_arc.lock().unwrap();
                    let ws = self.ctx().table.get_mut(&ws_res)?;
                    let _ = ptir_kv::ptir_kv_finalize(ws, &mut arena, kvtxn, false);
                    return Ok(Err(format!("ptir: submit failed: {e:#}")));
                }
            };
            let result = rx.await;

            // Finalize the KV txns (commit → KV persists for the next fire / abort
            // → revert this fire's writes), then advance the growing cursor.
            let success = matches!(result, Ok(Ok(_)));
            {
                let mut arena = arena_arc.lock().unwrap();
                let ws = self.ctx().table.get_mut(&ws_res)?;
                let _ = ptir_kv::ptir_kv_finalize(ws, &mut arena, kvtxn, success);
            }
            if success {
                self.ctx().table.get_mut(&this)?.committed_tokens = next_committed;
            }

            match result {
                Ok(Ok(out)) => {
                    // A PTIR fire returns the rich response carrying the harvested
                    // Reader-channel cells; marshal program 0's outputs back into
                    // the host store so the guest's `channel.take`/`read` see them.
                    let is_resp = matches!(out, crate::inference::ForwardOutput::Response(_));
                    if let crate::inference::ForwardOutput::Response(resp) = out {
                        let produced: Vec<(u32, Vec<u8>)> = resp
                            .ptir_output_at(0)
                            .unwrap_or_default()
                            .into_iter()
                            .map(|c| (c.channel, c.bytes))
                            .collect();
                        if std::env::var_os("PIE_PTIR_TRACE").is_some() {
                            eprintln!(
                                "[ptir-marshal] is_response=true ptir_output_count={} produced={}",
                                resp.ptir_output_indptr.len().saturating_sub(1),
                                produced.len()
                            );
                        }
                        if let Err(e) = store.lock().unwrap().marshal_response(&produced) {
                            return Ok(Err(format!("ptir: output marshal failed: {e}")));
                        }
                    } else if std::env::var_os("PIE_PTIR_TRACE").is_some() {
                        eprintln!("[ptir-marshal] NOT a Response (is_resp={is_resp}) — outputs dropped");
                    }
                    Ok(Ok(()))
                }
                Ok(Err(e)) => Ok(Err(format!("ptir: forward failed: {e:#}"))),
                Err(e) => Ok(Err(format!("ptir: forward channel closed: {e}"))),
            }
        }
    }

    async fn close(&mut self, _this: Resource<Pipeline>) -> Anyhow<()> {
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Pipeline>) -> Anyhow<()> {
        // Release the instance's KV working set alongside the pipeline. Decref
        // every page it holds via `destroy` (mirrors HostKvWorkingSet::drop)
        // BEFORE dropping the resource — else the arena pages leak (the growing
        // ws now holds ≥1 page per committed token, so the leak is unbounded).
        let ws_rep = self.ctx().table.get(&this).ok().map(|p| p.kv_ws.rep());
        self.ctx().table.delete(this)?;
        if let Some(rep) = ws_rep {
            let ws_res: Resource<KvWorkingSet> = Resource::new_own(rep);
            // Read the device first so the immutable borrow ends before `get_mut`.
            let dev = self.ctx().table.get(&ws_res).ok().map(|ws| ws.device());
            if let Some((m, d)) = dev {
                let arena_arc = crate::arena::get(m, d);
                let cas_arc = crate::working_set::kv_cas::get(m, d);
                let mut arena = arena_arc.lock().unwrap();
                let mut cas = cas_arc.lock().unwrap();
                if let Ok(ws) = self.ctx().table.get_mut(&ws_res) {
                    ws.destroy(&mut arena, &mut cas);
                }
            }
            let _ = self.ctx().table.delete(ws_res);
        }
        Ok(())
    }
}

/// Parse a beam program-output channel's bytes as `[n]` little-endian u32 (tokens
/// are non-negative i32 → the same bit pattern).
#[cfg(feature = "ptir")]
fn beam_channel_u32(
    produced: &[pie_driver_abi::PtirChannelValue],
    channel: u32,
    n: usize,
) -> Option<Vec<u32>> {
    let cell = produced.iter().find(|c| c.channel == channel)?;
    if cell.bytes.len() < n * 4 {
        return None;
    }
    Some(
        cell.bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

#[cfg(feature = "ptir")]
impl InstanceState {
    /// §6.2 beam host-replay fire (Design X). Builds the [B,P] decode batch from
    /// the replayed geometry (`BeamState`), resolves each beam's slots→physical in
    /// Rust (read pages via `resolve_read`; the write page via
    /// `write_slot_shared_inplace` for a HEIR continuing a shared tail, else
    /// `cow_write_slot` for a FORK), folds the B lanes via the existing batch
    /// assembly, fires the pre-assembled batch (bravo's `submit_prebuilt_async`),
    /// marshals the [B] program outputs back to the store, and replays the epilogue
    /// (`BeamState::step` on the harvested `out_par`) for the next fire.
    ///
    /// BRING-UP (4090, with charlie): the fire-0 / prompt seeding + the fresh-slot
    /// lifecycle (currently runtime-issued via `alloc_slots`) are refined against
    /// the value-verify vs the 3 beam goldens; the freeze/heir REPLAY itself is
    /// golden-verified host-side (`ptir_beam::tests`).
    async fn fire_beam(&mut self, this: Resource<Pipeline>) -> Anyhow<Result<(), String>> {
        use crate::inference::request;

        let page_size = crate::page_size::tokens_per_page(0);

        // 1) Snapshot the beam geometry + PTIR carrier (no borrow across await).
        let (geom, toks, pos, submission, store, ws_rep, b, p) = {
            let pl = self.ctx().table.get_mut(&this)?;
            let beam = pl.beam.as_ref().expect("fire_beam on a non-beam pipeline");
            let geom = beam.geom.clone();
            let toks = beam.toks.clone();
            let pos = beam.pos.clone();
            let (b, p) = (beam.state.b, beam.state.p);
            let ship = !pl.shipped;
            pl.shipped = true;
            let host_puts = pl.store.lock().unwrap().drain_host_puts();
            let submission = pl.instance.submission(ship, host_puts);
            (geom, toks, pos, submission, pl.store.clone(), pl.kv_ws.rep(), b, p)
        };

        // kvm → per-beam BRLE masks (1 query/beam).
        let (masks, _mask_indptr) = geom.masks(p, page_size);

        // 2) Resolve each beam's slots→physical + fold the B-lane batch under one
        //    KV write txn. HEIR writes go in-place (shared page preserved, alpha's
        //    `write_slot_shared_inplace`); FORK writes CoW-alloc a fresh page.
        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
        let arena_arc = crate::arena::get(0, 0);
        let wtx = self.ctx().table.get_mut(&ws_res)?.begin_write_txn();

        let built = {
            let mut arena = arena_arc.lock().unwrap();
            let mut txn = arena.txn_begin();
            let ws = self.ctx().table.get_mut(&ws_res)?;

            let mut batch = request::new_batched_forward_request_with_capacity(b);
            let mut union_phys = Vec::new();
            let mut err: Option<String> = None;

            'lanes: for lane in 0..b {
                let np_b = geom.np[lane] as usize;
                // Read pages = the beam's live pages before the tail; the tail
                // (index np_b-1 = w_slot) is the write page.
                let mut read_pages = Vec::with_capacity(np_b.saturating_sub(1));
                for pp in 0..np_b.saturating_sub(1) {
                    let slot = geom.pages[lane * p + pp];
                    match ws.resolve_read(slot, 1).map(|o| o[0]) {
                        Ok(obj) => {
                            if let Err(e) = arena.txn_pin(&mut txn, obj) {
                                err = Some(format!("beam pin read {lane}/{pp}: {e}"));
                                break 'lanes;
                            }
                            match arena.blocks(obj) {
                                Ok(bl) => read_pages.push(bl[0]),
                                Err(e) => {
                                    err = Some(format!("beam blocks read {lane}/{pp}: {e}"));
                                    break 'lanes;
                                }
                            }
                        }
                        Err(e) => {
                            err = Some(format!("beam resolve_read slot {slot}: {e}"));
                            break 'lanes;
                        }
                    }
                }
                // Write page: heir (shared in-place) vs fork (CoW fresh).
                let w_slot = geom.w_slot[lane];
                let write_obj = if geom.w_cont[lane] {
                    match ws.write_slot_shared_inplace(wtx, w_slot) {
                        Ok(o) => o,
                        Err(e) => {
                            err = Some(format!("beam heir write slot {w_slot}: {e}"));
                            break 'lanes;
                        }
                    }
                } else {
                    match ws.cow_write_slot(wtx, w_slot, &mut txn, &mut arena) {
                        Ok((o, _)) => o,
                        Err(e) => {
                            err = Some(format!("beam fork write slot {w_slot}: {e}"));
                            break 'lanes;
                        }
                    }
                };
                if let Err(e) = arena.txn_pin(&mut txn, write_obj) {
                    err = Some(format!("beam pin write {lane}: {e}"));
                    break 'lanes;
                }
                let write_page = match arena.blocks(write_obj) {
                    Ok(bl) => bl[0],
                    Err(e) => {
                        err = Some(format!("beam blocks write {lane}: {e}"));
                        break 'lanes;
                    }
                };

                let write = forward_prepare::KvWrite {
                    slot_index: (np_b - 1) as u32,
                    page: write_page,
                    valid_len: geom.w_off[lane] + 1,
                };
                let ctx_valid = np_b.saturating_sub(1) as u32 * page_size;
                let proj = match forward_prepare::project_kv(&read_pages, ctx_valid, &[write], page_size) {
                    Ok(pr) => pr,
                    Err(e) => {
                        err = Some(format!("beam project_kv {lane}: {e:?}"));
                        break 'lanes;
                    }
                };
                union_phys.extend_from_slice(&proj.physical_page_ids);

                // Per-lane decode request (1 token, custom kvm mask) folded into
                // the batch by the EXISTING assembly (physical kv_page_indices,
                // kv_last_page_lens, qo_indptr=[0..=B]).
                let req_l = request::new_per_request(
                    0,
                    vec![toks[lane]],
                    vec![pos[lane]],
                    vec![masks[lane].clone()],
                    true,
                    None,
                    vec![0],
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    false,
                    None,
                    None,
                );
                request::append_request_with_options(
                    &mut batch,
                    &req_l,
                    &proj.physical_page_ids,
                    proj.last_page_len,
                    page_size,
                    false,
                );
            }

            match err {
                None => Ok((txn, batch, union_phys)),
                Some(e) => {
                    arena.txn_abort(txn);
                    Err(e)
                }
            }
        };

        let (txn, mut batch, union_phys) = match built {
            Ok(v) => v,
            Err(e) => {
                self.ctx().table.get_mut(&ws_res)?.abort_writes(wtx);
                return Ok(Err(e));
            }
        };

        // 3) Attach the PTIR carrier + fire the pre-assembled B-lane batch.
        batch.push_ptir_program(&submission);
        let rx = match crate::inference::submit_prebuilt_async(batch, 0, union_phys, 0, Vec::new()) {
            Ok(rx) => rx,
            Err(e) => {
                arena_arc.lock().unwrap().txn_abort(txn);
                self.ctx().table.get_mut(&ws_res)?.abort_writes(wtx);
                return Ok(Err(format!("beam submit_prebuilt: {e:#}")));
            }
        };
        let result = rx.await;

        // 4) Finalize the KV txn.
        {
            let mut arena = arena_arc.lock().unwrap();
            let ws = self.ctx().table.get_mut(&ws_res)?;
            if matches!(result, Ok(Ok(_))) {
                let _ = arena.txn_commit(txn);
                ws.commit_writes(wtx);
            } else {
                arena.txn_abort(txn);
                ws.abort_writes(wtx);
            }
        }

        // 5) Marshal the [B] program outputs back to the store + replay the
        //    epilogue (step on out_par) for the next fire.
        match result {
            Ok(Ok(crate::inference::ForwardOutput::Response(resp))) => {
                let produced = resp.ptir_output_at(0).unwrap_or_default();
                let cells: Vec<(u32, Vec<u8>)> =
                    produced.iter().map(|c| (c.channel, c.bytes.clone())).collect();
                if let Err(e) = store.lock().unwrap().marshal_response(&cells) {
                    return Ok(Err(format!("beam output marshal failed: {e}")));
                }
                // out_par (ch14) = parent [B]; out (ch13) = survivor tokens [B].
                let parent = beam_channel_u32(&produced, 14, b);
                let out = beam_channel_u32(&produced, 13, b);
                // Runtime-issued fresh slots for the next step's forks.
                let fresh = match self.ctx().table.get_mut(&ws_res)?.alloc_slots(b as u32) {
                    Ok(f) => f,
                    Err(e) => return Ok(Err(format!("beam fresh alloc: {e}"))),
                };
                let pl = self.ctx().table.get_mut(&this)?;
                if let (Some(beam), Some(parent), Some(out)) = (pl.beam.as_mut(), parent, out) {
                    beam.geom = beam.state.step(&parent, &fresh);
                    beam.toks = out;
                    beam.pos.iter_mut().for_each(|x| *x += 1);
                }
                Ok(Ok(()))
            }
            Ok(Ok(_)) => Ok(Ok(())),
            Ok(Err(e)) => Ok(Err(format!("beam forward failed: {e:#}"))),
            Err(e) => Ok(Err(format!("beam forward channel closed: {e}"))),
        }
    }
}


impl pie::core::ptir::HostChannel for InstanceState {
    async fn put(
        &mut self,
        this: Resource<Channel>,
        value: Vec<u8>,
    ) -> Anyhow<Result<(), String>> {
        #[cfg(not(feature = "ptir"))]
        {
            let _ = (this, value);
            return disabled();
        }
        #[cfg(feature = "ptir")]
        {
            let ch = self.ctx().table.get(&this)?;
            let (store, index) = (ch.store.clone(), ch.index);
            Ok(store.lock().unwrap().put(index, value).map_err(|e| e.to_string()))
        }
    }

    async fn take(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        #[cfg(not(feature = "ptir"))]
        {
            let _ = this;
            return disabled();
        }
        #[cfg(feature = "ptir")]
        {
            let ch = self.ctx().table.get(&this)?;
            let (store, index) = (ch.store.clone(), ch.index);
            Ok(store.lock().unwrap().take(index).map_err(|e| e.to_string()))
        }
    }

    async fn read(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        #[cfg(not(feature = "ptir"))]
        {
            let _ = this;
            return disabled();
        }
        #[cfg(feature = "ptir")]
        {
            let ch = self.ctx().table.get(&this)?;
            let (store, index) = (ch.store.clone(), ch.index);
            Ok(store.lock().unwrap().read(index).map_err(|e| e.to_string()))
        }
    }

    async fn drop(&mut self, this: Resource<Channel>) -> Anyhow<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

/// Build the bind-time [`ModelProfile`] from the loaded model (P2b: vocab +
/// page-size + layer caps; model-gated intrinsics + second-party kernels default
/// conservative until the model surfaces them).
#[cfg(feature = "ptir")]
fn model_profile() -> pie_sampling_ir::ptir::registry::ModelProfile {
    let m = crate::model::model();
    pie_sampling_ir::ptir::registry::ModelProfile {
        vocab: m.vocab_size(),
        page_size: crate::page_size::tokens_per_page(0) as u32,
        num_layers: 1,
        activation: pie_sampling_ir::types::DType::F32,
        has_mtp_logits: false,
        has_mtp_drafts: false,
        has_value_head: false,
        kernels: Vec::new(),
    }
}
