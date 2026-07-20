//! WIT host glue for `pie:inferlet/forward` — thin `Host`/`HostChannel`/
//! `HostForwardPass` impls over the pipeline-owned [`Channel`]/[`ForwardPass`]
//! resource types (`crate::pipeline::channel`/`crate::pipeline::instance`).
//!
//! `forward-pass.new` binds + caches via [`crate::pipeline::program`] (the old
//! `register-program`, now an invisible compile/bind cache) and stamps the
//! container's roles onto the guest-constructed channels. The run-ahead fire
//! engine itself (`submit`'s body and FIFO finalization) lives in
//! [`crate::pipeline::fire`]; this file's
//! `HostForwardPass::submit` is a one-line delegation through the
//! [`crate::pipeline::fire::FireContext`] trait (implemented for `ProcessCtx`
//! below). `HostForwardPass::new` stays here (not `pipeline::fire`) because
//! it is fundamentally about validating and pushing the WIT resources
//! (`Resource<Channel>` handles, the KV/RS working sets) into the component
//! table — the WIT bind step, not the fire engine.

use std::sync::{Arc, Mutex};

use wasmtime::component::{Accessor, HasSelf, Resource};
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
pub use crate::pipeline::channel::Channel;
use crate::pipeline::channel::{BoundCells, ChannelCell, ChannelError};
use crate::pipeline::fire::lease::DevGeo;
pub use crate::pipeline::instance::ForwardPass;
use crate::pipeline::instance::Instance;
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::rs::working_set::RsWorkingSet;

use pie_ptir::container::HostRole;

use super::pie;

type Anyhow<T> = anyhow::Result<T>;

fn page_span(
    span: pie::inferlet::working_set::PageSpan,
) -> Result<crate::pipeline::instance::KvPageSpan, String> {
    let start = u64::from(span.start);
    let end = span.end.map(u64::from);
    if end.is_some_and(|end| start > end) {
        return Err(format!(
            "attention page-span start {start} exceeds end {}",
            end.unwrap()
        ));
    }
    Ok(crate::pipeline::instance::KvPageSpan { start, end })
}

#[derive(Clone, Copy)]
enum ChannelReadMode {
    Take,
    Read,
}

enum ChannelPoll {
    Ready(Result<Vec<u8>, String>),
    Finalize(crate::pipeline::fire::PendingOp),
    Pending {
        cell: Arc<Mutex<ChannelCell>>,
        fires: Option<crate::pipeline::fire::PendingFires>,
        process_id: uuid::Uuid,
        residency: Arc<Mutex<crate::inferlet::process::ProcessResidency>>,
    },
}

fn poll_channel(
    ctx: &mut ProcessCtx,
    this: &Resource<Channel>,
    mode: ChannelReadMode,
    pop_settled: bool,
) -> Anyhow<ChannelPoll> {
    let (cell, fires) = {
        let channel = ctx.ctx().table.get(this)?;
        (channel.cell.clone(), channel.fires.clone())
    };
    let value = {
        let mut cell = cell.lock().unwrap();
        match mode {
            ChannelReadMode::Take => cell.take(),
            ChannelReadMode::Read => cell.read(),
        }
    };
    match value {
        Ok(value) => return Ok(ChannelPoll::Ready(Ok(value))),
        Err(ChannelError::Empty) => {}
        Err(error) => return Ok(ChannelPoll::Ready(Err(error.to_string()))),
    }

    // Transfer only an already-settled FIFO entry while the Accessor lends us
    // the store. The caller holds the async finalizer gate through completion.
    if pop_settled && let Some(op) = crate::pipeline::fire::pop_settled(fires.as_ref()) {
        return Ok(ChannelPoll::Finalize(op));
    }

    Ok(ChannelPoll::Pending {
        cell,
        fires,
        process_id: ctx.id(),
        residency: ctx.residency_handle(),
    })
}

async fn materialize_channel(
    accessor: &Accessor<ProcessCtx, HasSelf<ProcessCtx>>,
    this: Resource<Channel>,
    mode: ChannelReadMode,
) -> Anyhow<Result<Vec<u8>, String>> {
    loop {
        let state = accessor.with(|mut access| poll_channel(access.get(), &this, mode, false))?;
        let state = match state {
            ChannelPoll::Pending {
                fires: Some(fires), ..
            } => {
                let _finalize_guard = fires.finalize_guard().await;
                let state =
                    accessor.with(|mut access| poll_channel(access.get(), &this, mode, true))?;
                match state {
                    ChannelPoll::Finalize(op) => {
                        let finalized = crate::pipeline::fire::finalize_op_await(op).await?;
                        accessor.with(|mut access| {
                            crate::pipeline::fire::complete_finalize(access.get(), finalized);
                        });
                        continue;
                    }
                    state => state,
                }
            }
            state => state,
        };

        match state {
            ChannelPoll::Ready(value) => return Ok(value),
            ChannelPoll::Finalize(_) => unreachable!("finalizer gate required before FIFO pop"),
            ChannelPoll::Pending {
                cell,
                fires,
                process_id,
                residency,
            } => {
                if let Err(error) =
                    crate::inferlet::process::preemption::await_channel_progress_idle(
                        process_id,
                        residency,
                        &cell,
                        fires.as_ref(),
                    )
                    .await
                {
                    return Ok(Err(error));
                }
            }
        }
    }
}

impl pie::inferlet::forward::Host for ProcessCtx {}

impl pie::inferlet::forward::HostChannel for ProcessCtx {
    async fn new(
        &mut self,
        shape: Vec<u32>,
        dtype: pie::inferlet::types::Dtype,
        capacity: u32,
    ) -> Anyhow<Resource<Channel>> {
        crate::inferlet::process::preemption::honor(self).await?;
        // Pure host bookkeeping — never fails at construction (the WIT
        // constructor cannot carry a result; a channel/decl mismatch instead
        // errors at forward-pass.new / submit).
        use pie::inferlet::types::Dtype;
        let dtype = match dtype {
            Dtype::F32 => pie_ptir::types::DType::F32,
            Dtype::I32 => pie_ptir::types::DType::I32,
            Dtype::U32 => pie_ptir::types::DType::U32,
            Dtype::Bool => pie_ptir::types::DType::Bool,
        };
        let cell = Arc::new(Mutex::new(ChannelCell::new(shape, dtype, capacity)));
        Ok(self.ctx().table.push(Channel { cell, fires: None })?)
    }

    async fn put(&mut self, this: Resource<Channel>, value: Vec<u8>) -> Anyhow<Result<(), String>> {
        crate::inferlet::process::preemption::contention_gate(self).await?;
        let cell = self.ctx().table.get(&this)?.cell.clone();
        loop {
            let result = cell.lock().unwrap().put_ref(&value);
            match result {
                Ok(()) => return Ok(Ok(())),
                Err(ChannelError::Full) => {}
                Err(error) => return Ok(Err(error.to_string())),
            }
            let wait = cell.lock().unwrap().writer_wait_state();
            let Some((endpoint, observed_head)) = wait else {
                return Ok(Err(ChannelError::Full.to_string()));
            };
            if let Err(error) = crate::inferlet::process::preemption::await_writer_progress(
                self,
                &endpoint,
                observed_head,
            )
            .await
            {
                return Ok(Err(error.to_string()));
            }
        }
    }

    async fn set(&mut self, this: Resource<Channel>, value: Vec<u8>) -> Anyhow<Result<(), String>> {
        crate::inferlet::process::preemption::contention_gate(self).await?;
        let cell = self.ctx().table.get(&this)?.cell.clone();
        let result = cell
            .lock()
            .unwrap()
            .set(value)
            .map_err(|error| error.to_string());
        Ok(result)
    }

    async fn drop(&mut self, this: Resource<Channel>) -> Anyhow<()> {
        // A pass that bound this channel holds its own Arc — dropping the
        // guest handle never dangles an in-flight fire. Native channel storage
        // is reference-counted by bound instances and releases on instance close.
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::inferlet::forward::HostChannelWithStore<ProcessCtx> for HasSelf<ProcessCtx> {
    /// The direct-wake await point (plan §4.5): while the cell is empty,
    /// non-blockingly drain already-settled pipeline ops (their KV/RS txns
    /// finalize here, bounding pin float), then park on the channel's reader
    /// wait slot — the driver's completion callback wakes it right after
    /// publishing the mirror tail. Store access is scoped to synchronous polls;
    /// the component task never holds an [`Accessor`] borrow across an await.
    async fn take(
        accessor: &Accessor<ProcessCtx, Self>,
        this: Resource<Channel>,
    ) -> Anyhow<Result<Vec<u8>, String>> {
        materialize_channel(accessor, this, ChannelReadMode::Take).await
    }

    /// Non-consuming peek; same await discipline as `take`.
    async fn read(
        accessor: &Accessor<ProcessCtx, Self>,
        this: Resource<Channel>,
    ) -> Anyhow<Result<Vec<u8>, String>> {
        materialize_channel(accessor, this, ChannelReadMode::Read).await
    }
}

impl pie::inferlet::forward::HostForwardPass for ProcessCtx {
    async fn new(
        &mut self,
        container_bytes: Vec<u8>,
        channels: Vec<Resource<Channel>>,
        kv_working_set: Resource<KvWorkingSet>,
        readable_pages: pie::inferlet::working_set::PageSpan,
        writable_pages: pie::inferlet::working_set::PageSpan,
        rs_working_sets: Vec<Resource<RsWorkingSet>>,
    ) -> Anyhow<Result<Resource<ForwardPass>, String>> {
        let bind_timing = crate::scheduler::fire_timing_enabled()
            .then(|| (self.id(), crate::scheduler::fire_timing_now_us()));
        let mut bind_stages = [0u64; 5];
        {
            // Identity dedup + bind against the model profile (the old
            // register-program, now invisible): hash-deduped compile/bind
            // cache; a malformed trace fails HERE with the validator's
            // message (the P2 exit).
            let prog = match crate::pipeline::program::register(
                container_bytes,
                &crate::pipeline::program::model_profile(),
            ) {
                Ok(p) => p,
                Err(e) => return Ok(Err(e.to_string())),
            };
            if bind_timing.is_some() {
                bind_stages[0] = crate::scheduler::fire_timing_now_us();
            }
            // Strict admission: the hash-deduped program compile above may
            // prewarm, but everything from here on creates per-instance
            // driver state (channel registration, instance bind) or claims
            // pooled KV (device-geometry seed pages) — execution first.
            crate::inferlet::process::ensure_execution_admitted(self).await;

            // Validate every handle against its dense declaration BEFORE
            // stamping any of them, so a failed `new` binds nothing.
            let decls = prog.bound.container.channels.clone();
            let extern_bindings = decls
                .iter()
                .enumerate()
                .map(|(dense, _)| {
                    prog.bound
                        .container
                        .externs
                        .iter()
                        .find(|binding| binding.chan == dense as u32)
                        .map(|binding| {
                            (
                                prog.bound.container.names[binding.name as usize].clone(),
                                binding.dir,
                            )
                        })
                })
                .collect::<Vec<_>>();
            if channels.len() != decls.len() {
                return Ok(Err(format!(
                    "pipeline: {} channel handles supplied for {} declared channels",
                    channels.len(),
                    decls.len()
                )));
            }
            let mut cells: BoundCells = Vec::with_capacity(channels.len());
            for (i, ch) in channels.iter().enumerate() {
                let cell = self.ctx().table.get(ch)?.cell.clone();
                if cells.iter().any(|prev| Arc::ptr_eq(prev, &cell)) {
                    return Ok(Err(format!(
                        "pipeline: channel {i} appears twice in the handle list"
                    )));
                }
                {
                    let c = cell.lock().unwrap();
                    // W3.2: a channel MAY bind to several passes (multi-pass
                    // channels). The old one-pass-per-channel gate is lifted; the
                    // driver's global channel registry (W0.1) resolves one shared
                    // device cell, and the pipeline enforces same-pipeline
                    // ordering (§3.4). Decl equality across the sharing passes is
                    // still validated (`matches_decl`) — a conflict is an error.
                    let extern_binding = extern_bindings[i]
                        .as_ref()
                        .map(|(name, dir)| (name.as_str(), *dir));
                    if let Err(e) = c.validate_attachment(&decls[i], extern_binding) {
                        return Ok(Err(format!("pipeline: channel {i}: {e}")));
                    }
                    // Pre-bind staged puts must fit the declared role: a
                    // Writer drains them per fire, a seeded non-Writer holds
                    // exactly its one seed, anything else never drains.
                    let staged = c.staged_len();
                    let staged_ok = match decls[i].host_role {
                        HostRole::Writer => true,
                        _ if decls[i].seeded => staged <= 1,
                        _ => staged == 0,
                    };
                    if !staged_ok {
                        return Ok(Err(format!(
                            "pipeline: channel {i}: {staged} staged put(s) don't fit its declared \
                             {:?}{} role",
                            decls[i].host_role,
                            if decls[i].seeded { " seeded" } else { "" }
                        )));
                    }
                }
                cells.push(cell);
            }

            let readable = match page_span(readable_pages) {
                Ok(span) => span,
                Err(error) => return Ok(Err(error)),
            };
            let writable = match page_span(writable_pages) {
                Ok(span) => span,
                Err(error) => return Ok(Err(error)),
            };
            let ws_rep = kv_working_set.rep();
            let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
            let bound_ws = self.ctx().table.get(&ws_res)?.clone();
            let stores = crate::store::registry::get(bound_ws.model, bound_ws.driver as usize);
            let page_len =
                match crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                    kv.page_len(bound_ws.id)
                }) {
                    Ok(page_len) => page_len,
                    Err(error) => {
                        return Ok(Err(format!("pipeline: KV page extent: {error}")));
                    }
                };
            if let Err(error) = readable.resolve(page_len) {
                return Ok(Err(error));
            }
            if let Err(error) = writable.resolve(page_len) {
                return Ok(Err(error));
            }
            // Classify once: DERIVABILITY decides the geometry class, not
            // op-pattern arity. Host-derivable submission geometry (the taint
            // fixpoint finds no device-decided port) is Host class on every
            // driver — the engine folds the geometry prologue per fire. Only
            // a device-dependent envelope on a driver with the device
            // geometry ports classifies DecodeEnvelope (run-ahead without
            // host round-trips); the same shape on a capability-less driver
            // falls back to Host and executes serialized, blocking loudly on
            // the first value the host truly cannot know.
            let device_port_mask =
                crate::driver::get_spec(bound_ws.driver as usize)?.device_geometry_port_mask;

            // Device-geometry pass (Track B): the program traces its COMPLETE
            // explicit geometry in-graph (loop-carried pages/write
            // descriptors); the runtime only leases physical pages. A real
            // wire class, capability-gated like every device-resolved family
            // — never a driver-side trace sniff (RV-6).
            let devgeo_capable = device_port_mask & pie_driver_abi::PIE_DEVICE_GEOMETRY_PORTS
                == pie_driver_abi::PIE_DEVICE_GEOMETRY_PORTS;
            let devgeo = match crate::pipeline::fire::lease::detect_device_geometry(
                &prog.bound.container,
            ) {
                Some(_) if !devgeo_capable => {
                    tracing::info!(
                        "device-geometry program on a driver without device geometry ports \
                         (mask {device_port_mask:#x}): falling back to host-evaluated \
                         serialized execution"
                    );
                    None
                }
                Some((b, fresh_dense, w_cont_dense)) => {
                    if readable.start != 0
                        || readable.end.is_some()
                        || writable.start != 0
                        || writable.end.is_some()
                    {
                        return Ok(Err(
                                "pipeline: device-geometry passes require full open readable and writable page spans"
                                    .to_string(),
                            ));
                    }
                    // Seed the physical-page lease with `B` fire-0 pages (one
                    // live page per lane) drawn from the guest's working set.
                    let reserved =
                        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                            kv.reserve(bound_ws.id, b as u64)
                        });
                    let seed_pages: Vec<u32> = match reserved {
                        Ok(range) => (range.start as u32..range.end as u32).collect(),
                        Err(e) => {
                            return Ok(Err(format!("pipeline: device-geometry seed alloc: {e}")));
                        }
                    };
                    let mut lease = crate::pipeline::fire::lease::PageLease::new(b);
                    lease.seed(seed_pages);
                    let has_mask = prog.bound.container.ports.iter().any(|p| {
                        matches!(p.port, pie_ptir::registry::Port::AttnMask)
                            && matches!(p.source, pie_ptir::container::PortSource::Channel(_))
                    });
                    Some(DevGeo {
                        lease,
                        b,
                        fresh_dense,
                        w_cont_dense,
                        has_mask,
                    })
                }
                None => None,
            };

            let taint = pie_ptir::pareval::geometry_taint(&prog.bound);
            let decode_envelope = if devgeo.is_some() || taint.host_derivable() {
                None
            } else {
                match crate::pipeline::fire::geometry::classify_decode_envelope(
                    &prog.bound.container,
                ) {
                    Ok(Some(envelope)) => {
                        let required =
                            crate::pipeline::fire::geometry::envelope_required_ports(&envelope);
                        if device_port_mask & required == required {
                            Some(envelope)
                        } else {
                            tracing::info!(
                                "decode envelope on a driver without device geometry ports \
                                 (mask {device_port_mask:#x}, needs {required:#x}): falling \
                                 back to host-evaluated serialized execution"
                            );
                            None
                        }
                    }
                    Ok(None) => None,
                    Err(reason) => {
                        tracing::warn!(
                            "device-dependent geometry is not a decode envelope ({reason}); \
                             falling back to host-evaluated execution — fires block loudly \
                             on values the host cannot derive"
                        );
                        None
                    }
                }
            };
            if decode_envelope.is_some() && (readable.start != 0 || readable.end.is_some()) {
                return Ok(Err(
                    "pipeline: device-resolved passes require a full open readable page span"
                        .to_string(),
                ));
            }
            let geometry_class = if devgeo.is_some() {
                pie_driver_abi::GeometryClass::DeviceGeometry
            } else if decode_envelope.is_some() {
                pie_driver_abi::GeometryClass::DecodeEnvelope
            } else {
                pie_driver_abi::GeometryClass::Host
            };
            let rs_reps = rs_working_sets.iter().map(Resource::rep).collect();

            if bind_timing.is_some() {
                bind_stages[1] = crate::scheduler::fire_timing_now_us();
            }

            let instance_id = crate::pipeline::instance::next_instance_id();
            for (dense, cell) in cells.iter().enumerate() {
                let extern_binding = extern_bindings[dense]
                    .as_ref()
                    .map(|(name, dir)| (name.as_str(), *dir));
                if let Err(error) =
                    cell.lock()
                        .unwrap()
                        .attach(instance_id, &decls[dense], extern_binding)
                {
                    for attached in &cells {
                        attached.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("pipeline: channel {dense} attach: {error}")));
                }
            }
            let mut missing_dense = Vec::new();
            let mut registration_plans = Vec::new();
            for (dense, cell) in cells.iter().enumerate() {
                if cell.lock().unwrap().endpoint().is_some() {
                    continue;
                }
                let extern_binding = extern_bindings[dense].as_ref();
                missing_dense.push(dense);
                registration_plans.push(crate::driver::ChannelRegistrationPlan {
                    driver_id: 0,
                    channel_id: cell.lock().unwrap().global_id,
                    shape: decls[dense].shape.dims().to_vec(),
                    dtype: decls[dense].dtype.tag(),
                    host_role: decls[dense].host_role as u8,
                    seeded: decls[dense].seeded,
                    extern_dir: extern_binding
                        .map(|(_, dir)| match dir {
                            pie_ptir::container::ExternDir::Import => {
                                pie_driver_abi::PIE_CHANNEL_EXTERN_IMPORT
                            }
                            pie_ptir::container::ExternDir::Export => {
                                pie_driver_abi::PIE_CHANNEL_EXTERN_EXPORT
                            }
                        })
                        .unwrap_or(pie_driver_abi::PIE_CHANNEL_EXTERN_NONE),
                    capacity: decls[dense].capacity,
                    reader_wait_id: 0,
                    writer_wait_id: 0,
                    extern_name: extern_binding
                        .map(|(name, _)| name.as_bytes().to_vec())
                        .unwrap_or_default(),
                });
            }
            // All validation passed — capture ids, register the program
            // (hash-cached, synchronous), and stage the seeds BEFORE the one
            // combined register+bind control: the bind consumes only
            // pre-known ids and host-staged seed bytes, so registration and
            // bind have a pure ORDERING dependency — two separate controls
            // doubled the turnover convoy (V6 iteration 25).
            let channel_ids: Vec<u64> = cells.iter().map(|c| c.lock().unwrap().global_id).collect();
            let channel_reps: Vec<u32> = channels.iter().map(|c| c.rep()).collect();
            let program_registration = crate::driver::ProgramRegistration {
                program_hash: prog.hash,
                canonical_bytes: prog.bytes.clone(),
                sidecar_bytes: prog.sidecar.clone(),
            };
            if bind_timing.is_some() {
                bind_stages[2] = crate::scheduler::fire_timing_now_us();
                bind_stages[3] = bind_stages[2];
            }
            let mut instance_seeds = Vec::new();
            let mut seed_values = Vec::new();
            for (dense, cell) in cells.iter().enumerate() {
                let cell = cell.lock().unwrap();
                if !cell.seeded {
                    continue;
                }
                let bytes = match cell.peek_seed() {
                    Ok(bytes) => bytes,
                    Err(e) => return Ok(Err(format!("pipeline: channel {dense} seed: {e}"))),
                };
                instance_seeds.push(crate::pipeline::instance::ChannelSeed {
                    channel: dense as u32,
                    data: bytes.clone(),
                });
                seed_values.push(crate::driver::ChannelValue {
                    channel: cell.global_id,
                    bytes,
                });
            }
            let instance = Instance {
                program: prog,
                instance_id,
                seeds: instance_seeds,
            };
            let process_id = self.id();
            let (registered, bound_instance, scheduler) =
                match crate::scheduler::register_channels_bind_classified(
                    0,
                    Some(process_id),
                    registration_plans,
                    program_registration,
                    instance.instance_id,
                    channel_ids.clone(),
                    seed_values,
                    geometry_class,
                )
                .await
                {
                    Ok(pair) => pair,
                    Err(error) => {
                        for attached in &cells {
                            attached.lock().unwrap().detach(instance_id);
                        }
                        return Ok(Err(format!("pipeline: register+bind: {error:#}")));
                    }
                };
            if registered.len() != missing_dense.len() {
                let _ = scheduler
                    .close_instance(bound_instance.instance_id, bound_instance.pacing_wait_id);
                for attached in &cells {
                    attached.lock().unwrap().detach(instance_id);
                }
                return Ok(Err(
                    "pipeline: channel registration count mismatch".to_string()
                ));
            }
            for (dense, endpoint) in missing_dense.into_iter().zip(registered) {
                if let Err(error) = cells[dense].lock().unwrap().attach_endpoint(endpoint) {
                    let _ = scheduler
                        .close_instance(bound_instance.instance_id, bound_instance.pacing_wait_id);
                    for attached in &cells {
                        attached.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("pipeline: channel {dense} endpoint: {error}")));
                }
            }
            if bind_timing.is_some() {
                bind_stages[4] = crate::scheduler::fire_timing_now_us();
            }
            for cell in &cells {
                let mut cell = cell.lock().unwrap();
                if cell.seeded {
                    cell.commit_seed();
                }
                // A seeded Writer held its staging back until the seed
                // settled into the instance descriptor — flush it now so
                // direct ring puts take over (plan §4.2).
                if cell.role == Some(HostRole::Writer)
                    && let Err(error) = cell.flush_writer_staging()
                {
                    drop(cell);
                    let _ = scheduler
                        .close_instance(bound_instance.instance_id, bound_instance.pacing_wait_id);
                    for cell in &cells {
                        cell.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("pipeline: writer staging flush: {error}")));
                }
            }
            let dense_mask = instance.program.bound.container.ports.iter().any(|p| {
                matches!(p.port, pie_ptir::registry::Port::AttnMask)
                    && matches!(p.source, pie_ptir::container::PortSource::Channel(_))
            });
            let host_shadow = crate::pipeline::fire::shadow::HostShadow::new(
                &instance.program.bound,
                &instance.seeds,
            );
            let res = self.ctx().table.push(ForwardPass {
                instance,
                bound_instance,
                scheduler,
                cells,
                channel_ids,
                channel_reps,
                fires: None,
                kv_ws: ws_rep,
                kv_declaration: crate::pipeline::instance::KvDeclaration {
                    ws_rep,
                    readable,
                    writable,
                },
                rs_ws: rs_reps,
                kv_declaration_realized: false,
                failed: None,
                devgeo,
                decode_envelope,
                dense_mask,
                host_shadow,
                closed: false,
            })?;
            if let Some((process_id, started_us)) = bind_timing {
                let finished_us = crate::scheduler::fire_timing_now_us();
                crate::scheduler::fire_timing_write(&serde_json::json!({
                    "schema": 1,
                    "source": "runtime",
                    "event": "forward_pass_bound",
                    "process_id": process_id,
                    "instance_id": instance_id,
                    "pass_resource": res.rep(),
                    "bind_started_us": started_us,
                    "bind_finished_us": finished_us,
                    "bind_us": finished_us.saturating_sub(started_us),
                    "program_compile_us": bind_stages[0].saturating_sub(started_us),
                    "bind_validate_us": bind_stages[1].saturating_sub(bind_stages[0]),
                    "channel_register_us": bind_stages[2].saturating_sub(bind_stages[1]),
                    "program_register_us": bind_stages[3].saturating_sub(bind_stages[2]),
                    "driver_bind_us": bind_stages[4].saturating_sub(bind_stages[3]),
                    "bind_finalize_us": finished_us.saturating_sub(bind_stages[4]),
                }));
            }
            Ok(Ok(res))
        }
    }

    async fn set_attn_working_set(
        &mut self,
        this: Resource<ForwardPass>,
        kv_working_set: Resource<KvWorkingSet>,
        readable_pages: pie::inferlet::working_set::PageSpan,
        writable_pages: pie::inferlet::working_set::PageSpan,
    ) -> Anyhow<Result<(), String>> {
        let readable = match page_span(readable_pages) {
            Ok(span) => span,
            Err(error) => return Ok(Err(error)),
        };
        let writable = match page_span(writable_pages) {
            Ok(span) => span,
            Err(error) => return Ok(Err(error)),
        };
        let new_ws = self.ctx().table.get(&kv_working_set)?.clone();
        let old_rep = self.ctx().table.get(&this)?.kv_ws;
        let old_resource: Resource<KvWorkingSet> = Resource::new_borrow(old_rep);
        let old_ws = self.ctx().table.get(&old_resource)?;
        if new_ws.model != old_ws.model || new_ws.driver != old_ws.driver {
            return Ok(Err(format!(
                "pipeline: replacement KV working set belongs to model/driver ({}, {}), \
                 expected ({}, {})",
                new_ws.model, new_ws.driver, old_ws.model, old_ws.driver
            )));
        }
        let stores = crate::store::registry::get(new_ws.model, new_ws.driver as usize);
        let page_len = match crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
            kv.page_len(new_ws.id)
        }) {
            Ok(value) => value,
            Err(error) => return Ok(Err(format!("pipeline: replacement KV state: {error}"))),
        };
        if let Err(error) = readable.resolve(page_len) {
            return Ok(Err(error));
        }
        if let Err(error) = writable.resolve(page_len) {
            return Ok(Err(error));
        }
        Ok(self
            .ctx()
            .table
            .get_mut(&this)?
            .replace_kv_declaration(kv_working_set.rep(), readable, writable)
            .map_err(|error| format!("pipeline: {error}")))
    }

    async fn set_rs_working_sets(
        &mut self,
        this: Resource<ForwardPass>,
        rs_working_sets: Vec<Resource<RsWorkingSet>>,
    ) -> Anyhow<Result<(), String>> {
        let has_recurrent_state = pie_model::model().rs_caps().state_size > 0;
        let (kv_rep, qo_indptr) = {
            let pass = self.ctx().table.get(&this)?;
            let pending = pass
                .fires
                .as_ref()
                .map(|fifo| fifo.lock().unwrap().len())
                .unwrap_or(0);
            if pending != 0 {
                return Ok(Err(format!(
                    "pipeline: cannot replace rs-working-sets while {pending} operation(s) \
                     remain in the pass FIFO"
                )));
            }
            let qo_indptr = if let Some(devgeo) = pass.devgeo.as_ref() {
                vec![0; devgeo.b + 1]
            } else if has_recurrent_state {
                match pass
                    .instance
                    .fire_geometry(crate::pipeline::program::model_profile().page_size)
                {
                    Ok(geometry) => geometry.qo_indptr,
                    Err(error) => {
                        return Ok(Err(format!(
                            "pipeline: cannot resolve request rows for rs-working-set rebind: \
                             {error:?}"
                        )));
                    }
                }
            } else {
                Vec::new()
            };
            (pass.kv_ws, qo_indptr)
        };

        if let Err(error) = crate::pipeline::fire::rs::validate_count(
            rs_working_sets.len(),
            &qo_indptr,
            has_recurrent_state,
        ) {
            return Ok(Err(format!("pipeline: recurrent-state binding: {error}")));
        }

        let kv_resource: Resource<KvWorkingSet> = Resource::new_borrow(kv_rep);
        let kv = self.ctx().table.get(&kv_resource)?.clone();
        let mut reps = Vec::with_capacity(rs_working_sets.len());
        let mut ids = Vec::with_capacity(rs_working_sets.len());
        for (row, resource) in rs_working_sets.iter().enumerate() {
            let rs = self.ctx().table.get(resource)?;
            if rs.model != kv.model || rs.driver != kv.driver {
                return Ok(Err(format!(
                    "pipeline: rs-working-set at request row {row} belongs to model/driver \
                     ({}, {}), expected ({}, {})",
                    rs.model, rs.driver, kv.model, kv.driver
                )));
            }
            if ids.contains(&rs.id) {
                return Ok(Err(format!(
                    "pipeline: rs-working-set at request row {row} aliases an earlier row"
                )));
            }
            ids.push(rs.id);
            reps.push(resource.rep());
        }

        Ok(self
            .ctx()
            .table
            .get_mut(&this)?
            .replace_rs_working_sets(reps)
            .map_err(|error| format!("pipeline: {error}")))
    }

    async fn drop(&mut self, this: Resource<ForwardPass>) -> Anyhow<()> {
        // A pass can be dropped before its pipeline. Drain the shared FIFO first
        // so every callback, mirror publication, and KV/RS transaction completes
        // before raw mirror pointers are detached or pages become reusable.
        let fires = self.ctx().table.get(&this)?.fires.clone();
        if let Some(fires) = fires {
            let _finalize_guard = fires.finalize_guard().await;
            loop {
                let op = fires.lock().unwrap().pop_front();
                match op {
                    Some(op) => crate::pipeline::fire::finalize_op(self, op).await?,
                    None => break,
                }
            }
        }

        // Native teardown (close the driver instance, detach every bound
        // cell, reclaim device-geometry grants) is idempotent and shared
        // with the `Drop` fallback (`ForwardPass::close_native`) — the local
        // `pass` binding below drops at the end of this function and would
        // otherwise repeat the work, but `close_native` no-ops the second
        // time.
        let mut pass = self.ctx().table.delete(this)?;
        pass.close_native();
        Ok(())
    }

    /// Run-ahead submit on `on`: pure-attention prepares immediately; RS-bound
    /// passes first finalize prior FIFO operations. See
    /// `crate::pipeline::fire`'s module docs; errors after this call surface
    /// via channel poison + `take`.
    async fn submit(
        &mut self,
        this: Resource<ForwardPass>,
        on: Resource<crate::pipeline::Pipeline>,
    ) -> Anyhow<Result<(), String>> {
        crate::inferlet::process::ensure_execution_admitted(self).await;
        crate::inferlet::process::preemption::contention_gate(self).await?;
        crate::pipeline::fire::submit_pass(self, on, this).await
    }
}
