//! WIT host glue for `pie:inferlet/forward` — thin `Host`/`HostChannel`/
//! `HostForwardPass` impls over the pipeline-owned [`Channel`]/[`ForwardPass`]
//! resource types (`crate::pipeline::channel`/`crate::pipeline::instance`).
//!
//! `forward-pass.new` binds + caches via [`crate::pipeline::program`] (the old
//! `register-program`, now an invisible compile/bind cache) and stamps the
//! container's roles onto the guest-constructed channels. The run-ahead fire
//! engine itself (`submit`'s body, FIFO finalization, the optimistic
//! `committed_tokens` cursor) lives in [`crate::pipeline::fire`]; this file's
//! `HostForwardPass::submit` is a one-line delegation through the
//! [`crate::pipeline::fire::FireContext`] trait (implemented for `ProcessCtx`
//! below). `HostForwardPass::new` stays here (not `pipeline::fire`) because
//! it is fundamentally about validating and pushing the WIT resources
//! (`Resource<Channel>` handles, the KV/RS working sets) into the component
//! table — the WIT bind step, not the fire engine.

use std::sync::{Arc, Mutex};

use wasmtime::component::Resource;
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

    /// The direct-wake await point (plan §4.5): while the cell is empty,
    /// non-blockingly drain already-settled pipeline ops (their KV/RS txns
    /// finalize here, bounding pin float), then park on the channel's reader
    /// wait slot — the driver's completion callback wakes it right after
    /// publishing the mirror tail. The park races the oldest in-flight op so
    /// a fire that resolves without producing on this channel still unblocks
    /// the loop; with no endpoint and nothing in flight, nothing can ever
    /// fill the cell and `Empty` is returned instead of parking.
    async fn take(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        loop {
            crate::inferlet::process::preemption::honor(self).await?;
            let (cell, fires) = {
                let ch = self.ctx().table.get(&this)?;
                (ch.cell.clone(), ch.fires.clone())
            };
            match cell.lock().unwrap().take() {
                Ok(v) => return Ok(Ok(v)),
                Err(ChannelError::Empty) => {}
                Err(e) => return Ok(Err(e.to_string())),
            }
            if crate::pipeline::fire::drain_settled(self, fires.as_ref()).await? {
                continue;
            }
            if let Err(error) = crate::inferlet::process::preemption::await_channel_progress(
                self,
                &cell,
                fires.as_ref(),
            )
            .await
            {
                return Ok(Err(error));
            }
        }
    }

    /// Non-consuming peek; same await discipline as `take`.
    async fn read(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        loop {
            crate::inferlet::process::preemption::honor(self).await?;
            let (cell, fires) = {
                let ch = self.ctx().table.get(&this)?;
                (ch.cell.clone(), ch.fires.clone())
            };
            match cell.lock().unwrap().read() {
                Ok(v) => return Ok(Ok(v)),
                Err(ChannelError::Empty) => {}
                Err(e) => return Ok(Err(e.to_string())),
            }
            if crate::pipeline::fire::drain_settled(self, fires.as_ref()).await? {
                continue;
            }
            if let Err(error) = crate::inferlet::process::preemption::await_channel_progress(
                self,
                &cell,
                fires.as_ref(),
            )
            .await
            {
                return Ok(Err(error));
            }
        }
    }

    async fn drop(&mut self, this: Resource<Channel>) -> Anyhow<()> {
        // A pass that bound this channel holds its own Arc — dropping the
        // guest handle never dangles an in-flight fire. Native channel storage
        // is reference-counted by bound instances and releases on instance close.
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::inferlet::forward::HostForwardPass for ProcessCtx {
    async fn new(
        &mut self,
        container_bytes: Vec<u8>,
        channels: Vec<Resource<Channel>>,
        kv_working_sets: Vec<Resource<KvWorkingSet>>,
        rs_working_sets: Vec<Resource<RsWorkingSet>>,
    ) -> Anyhow<Result<Resource<ForwardPass>, String>> {
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

            // v1 single-model contract: exactly one guest-owned KV working set
            // (the classic forward-pass borrow convention — the guest keeps it
            // alive for the pass's lifetime); at most one RS working set
            // (hybrid / linear-attention models).
            if kv_working_sets.len() != 1 {
                return Ok(Err(format!(
                    "pipeline: expected exactly one kv-working-set, got {}",
                    kv_working_sets.len()
                )));
            }
            if rs_working_sets.len() > 1 {
                return Ok(Err(format!(
                    "pipeline: expected at most one rs-working-set, got {}",
                    rs_working_sets.len()
                )));
            }
            let ws_rep = kv_working_sets[0].rep();
            let rs_rep = rs_working_sets.first().map(|r| r.rep());

            // Device-geometry pass (Track B): seed the physical-page lease with
            // `B` fire-0 pages (one live page per lane) drawn from the guest's
            // working set. The [B,P] geometry is device-produced (the program
            // traces `page_indptr = CumSum(np)` + packed pages in-graph); the
            // runtime no longer replays the epilogue arithmetic nor eagerly
            // reserves B*P slots. A normal program's ws starts as the guest left
            // it — `pipeline::fire::kv::prepare` grows it per fire.
            let devgeo =
                match crate::pipeline::fire::lease::detect_device_geometry(&prog.bound.container) {
                    Some((b, fresh_dense, w_cont_dense)) => {
                        let ws_res: Resource<KvWorkingSet> = Resource::new_borrow(ws_rep);
                        let ws = self.ctx().table.get(&ws_res)?.clone();
                        let stores = crate::store::registry::get(ws.model, ws.driver as usize);
                        let reserved = stores.kv.lock().unwrap().reserve(ws.id, b as u64);
                        let seed_pages: Vec<u32> = match reserved {
                            Ok(range) => (range.start as u32..range.end as u32).collect(),
                            Err(e) => {
                                return Ok(Err(format!(
                                    "pipeline: device-geometry seed alloc: {e}"
                                )));
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
            for (dense, cell) in cells.iter().enumerate() {
                let existing = cell.lock().unwrap().endpoint();
                let endpoint = match existing {
                    Some(endpoint) => endpoint,
                    None => {
                        let extern_binding = extern_bindings[dense].as_ref();
                        let plan = crate::driver::ChannelRegistrationPlan {
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
                        };
                        match crate::scheduler::register_channel(0, plan) {
                            Ok(endpoint) => endpoint,
                            Err(error) => {
                                for attached in &cells {
                                    attached.lock().unwrap().detach(instance_id);
                                }
                                return Ok(Err(format!(
                                    "pipeline: register channel {dense}: {error:#}"
                                )));
                            }
                        }
                    }
                };
                if let Err(error) = cell.lock().unwrap().attach_endpoint(endpoint) {
                    for attached in &cells {
                        attached.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("pipeline: channel {dense} endpoint: {error}")));
                }
            }

            // All validation passed — stamp the container's roles onto the
            // cells (the bind point) and mint the instance identity (the
            // driver's persistent channel-arena key). The await FIFO is owned by
            // the PIPELINE now (W3.1), wired to the channels at submit.
            // Capture the dense-index → global-channel-id map now that the cells
            // are validated (multi-pass channels: a global id is stable across
            // every pass a channel binds into).
            let channel_ids: Vec<u64> = cells.iter().map(|c| c.lock().unwrap().global_id).collect();
            // Capture the bound channel resource reps so `submit` can point each
            // channel's await queue at the feeding pipeline (W3.1).
            let channel_reps: Vec<u32> = channels.iter().map(|c| c.rep()).collect();
            let program_id = match crate::scheduler::register_program(
                0,
                crate::driver::ProgramRegistration {
                    program_hash: prog.hash,
                    canonical_bytes: prog.bytes.clone(),
                    sidecar_bytes: prog.sidecar.clone(),
                },
            ) {
                Ok(id) => id,
                Err(e) => return Ok(Err(format!("pipeline: register program: {e:#}"))),
            };
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
            let bound_instance = match crate::scheduler::bind_instance(
                0,
                program_id,
                instance.instance_id,
                channel_ids.clone(),
                seed_values,
            ) {
                Ok(bound) => bound,
                Err(e) => {
                    for cell in &cells {
                        cell.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("pipeline: bind instance: {e:#}")));
                }
            };
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
                    let _ = crate::scheduler::close_instance(&bound_instance);
                    for cell in &cells {
                        cell.lock().unwrap().detach(instance_id);
                    }
                    return Ok(Err(format!("pipeline: writer staging flush: {error}")));
                }
            }
            let canonical_kv =
                crate::pipeline::fire::kv::canonical_kv_shape(&instance.program.bound.container);
            let dense_mask = instance.program.bound.container.ports.iter().any(|p| {
                matches!(p.port, pie_ptir::registry::Port::AttnMask)
                    && matches!(p.source, pie_ptir::container::PortSource::Channel(_))
            });
            let res = self.ctx().table.push(ForwardPass {
                instance,
                bound_instance,
                cells,
                channel_ids,
                channel_reps,
                fires: None,
                kv_ws: ws_rep,
                rs_ws: rs_rep,
                committed_tokens: 0,
                failed: None,
                devgeo,
                canonical_kv,
                dense_mask,
                fired_once: false,
                closed: false,
            })?;
            Ok(Ok(res))
        }
    }

    async fn drop(&mut self, this: Resource<ForwardPass>) -> Anyhow<()> {
        // A pass can be dropped before its pipeline. Drain the shared FIFO first
        // so every callback, mirror publication, and KV/RS transaction completes
        // before raw mirror pointers are detached or pages become reusable.
        let fires = self.ctx().table.get(&this)?.fires.clone();
        if let Some(fires) = fires {
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

    /// Run-ahead submit on `on`: prepare + fire + enqueue, NO await. See
    /// `crate::pipeline::fire`'s module docs; errors after this call surface
    /// via channel poison + `take`.
    async fn submit(
        &mut self,
        this: Resource<ForwardPass>,
        on: Resource<crate::pipeline::Pipeline>,
    ) -> Anyhow<Result<(), String>> {
        crate::inferlet::process::preemption::contention_gate(self).await?;
        crate::pipeline::fire::submit_pass(self, on, this).await
    }
}
