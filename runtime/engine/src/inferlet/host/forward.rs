//! WIT host glue for `pie:inferlet/forward` — thin `Host`/`HostChannel`/
//! `HostForwardPass` impls over the pipeline-owned [`Channel`]/[`ForwardPass`]
//! resource types (`crate::pipeline::channel`/`crate::pipeline::instance`).
//!
//! `forward-pass.program` binds + caches via [`crate::pipeline::program`] (the old
//! `register-program`, now an invisible compile/bind cache) and stamps the
//! container's roles onto the guest-constructed channels. The run-ahead fire
//! engine itself (`submit`'s body and FIFO finalization) lives in
//! [`crate::pipeline::fire`]; this file's
//! `HostForwardPass::submit` is a one-line delegation through the
//! [`crate::pipeline::fire::FireContext`] trait (implemented for `ProcessCtx`
//! below). `HostForwardPass::program` stays here (not `pipeline::fire`) because
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
use crate::pipeline::instance::{
    AttentionBinding, BoundForwardPass, EmbedBinding, PassKind, RsGeometryBinding,
};
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::rs::working_set::RsWorkingSet;

use pie_ir::container::{HostRole, PortSource, TraceContainer};
use pie_ir::registry::Port;
use pie_ir::types::DType;

use super::pie;

type Anyhow<T> = anyhow::Result<T>;

/// Which forward interface this model requires. Must stay in lockstep with
/// `model.pass-kind()` (`host/model.rs`) -- that call is how a guest picks its
/// interface, and `core_gate` is how the host proves the pick was right.
fn model_pass_kind() -> PassKind {
    let model = crate::model::model();
    match (model.kv_page_size() > 0, model.rs_caps().state_size > 0) {
        (_, false) => PassKind::Attention,
        (true, true) => PassKind::Hybrid,
        (false, true) => PassKind::Recurrent,
    }
}

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

fn validate_descriptor_bindings(
    container: &TraceContainer,
    channel_reps: &[u32],
    expected: &[(Port, Option<u32>)],
) -> Result<(), String> {
    validate_bindings(container, channel_reps, expected, false)
}

/// The RS geometry ports, which are OPTIONAL: a pass always binds
/// `rs-geometry`, but a program only traces the buffer addressing when it
/// actually addresses the buffer. Absent-but-attached is therefore legal here
/// and an error for the KV family. Present-but-mismatched is an error for both.
fn validate_optional_descriptor_bindings(
    container: &TraceContainer,
    channel_reps: &[u32],
    expected: &[(Port, Option<u32>)],
) -> Result<(), String> {
    validate_bindings(container, channel_reps, expected, true)
}

fn validate_bindings(
    container: &TraceContainer,
    channel_reps: &[u32],
    expected: &[(Port, Option<u32>)],
    optional: bool,
) -> Result<(), String> {
    for &(port, expected_rep) in expected {
        let binding = container.ports.iter().find(|binding| binding.port == port);
        if optional && binding.is_none() {
            continue;
        }
        match (binding, expected_rep) {
            (None, None) => {}
            (None, Some(_)) => {
                return Err(format!(
                    "pipeline: attached {} channel is absent from the traced program",
                    port.name()
                ));
            }
            (Some(_), None) => {
                return Err(format!(
                    "pipeline: traced program binds {} without a resource attachment",
                    port.name()
                ));
            }
            (Some(binding), Some(expected_rep)) => {
                let PortSource::Channel(dense) = &binding.source else {
                    return Err(format!(
                        "pipeline: descriptor port {} must be channel-bound",
                        port.name()
                    ));
                };
                let actual_rep = channel_reps.get(*dense as usize).ok_or_else(|| {
                    format!(
                        "pipeline: descriptor port {} references missing channel {}",
                        port.name(),
                        dense
                    )
                })?;
                if *actual_rep != expected_rep {
                    return Err(format!(
                        "pipeline: descriptor port {} uses channel resource {}, \
                             attached resource is {}",
                        port.name(),
                        actual_rep,
                        expected_rep
                    ));
                }
            }
        }
    }
    Ok(())
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
    },
}

fn poll_channel(
    ctx: &mut ProcessCtx,
    this: &Resource<Channel>,
    mode: ChannelReadMode,
    pop_settled: bool,
    settle_ready_take: bool,
) -> Anyhow<ChannelPoll> {
    let (cell, fires) = {
        let channel = ctx.ctx().table.get(this)?;
        (channel.cell.clone(), channel.fires.clone())
    };
    if settle_ready_take
        && matches!(mode, ChannelReadMode::Take)
        && fires
            .as_ref()
            .is_some_and(|fires| !fires.lock().unwrap().is_empty())
    {
        let ready = cell.lock().unwrap().read();
        match ready {
            Ok(_) if pop_settled => {
                if let Some(op) = fires
                    .as_ref()
                    .and_then(|fires| fires.lock().unwrap().pop_front())
                {
                    return Ok(ChannelPoll::Finalize(op));
                }
            }
            Ok(_) => {
                return Ok(ChannelPoll::Pending { cell, fires });
            }
            Err(ChannelError::Empty) => {}
            Err(error) => return Ok(ChannelPoll::Ready(Err(error.to_string()))),
        }
    }

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

    Ok(ChannelPoll::Pending { cell, fires })
}

async fn materialize_channel(
    accessor: &Accessor<ProcessCtx, HasSelf<ProcessCtx>>,
    this: Resource<Channel>,
    mode: ChannelReadMode,
) -> Anyhow<Result<Vec<u8>, String>> {
    let mut settle_ready_take = true;
    loop {
        let state = accessor
            .with(|mut access| poll_channel(access.get(), &this, mode, false, settle_ready_take))?;
        let state = match state {
            ChannelPoll::Pending {
                fires: Some(fires), ..
            } => {
                let _finalize_guard = fires.finalize_guard().await;
                let state = accessor.with(|mut access| {
                    poll_channel(access.get(), &this, mode, true, settle_ready_take)
                })?;
                match state {
                    ChannelPoll::Finalize(op) => {
                        let finalized = crate::pipeline::fire::finalize_op_await(op).await?;
                        accessor.with(|mut access| {
                            crate::pipeline::fire::complete_finalize(access.get(), finalized);
                        });
                        settle_ready_take = false;
                        continue;
                    }
                    state => state,
                }
            }
            state => state,
        };

        match state {
            ChannelPoll::Ready(value) => {
                return Ok(value);
            }
            ChannelPoll::Finalize(_) => unreachable!("finalizer gate required before FIFO pop"),
            ChannelPoll::Pending { cell, fires, .. } => {
                settle_ready_take = true;
                // A plain await: an idle channel wait holds no pooled
                // state, so the planner can evict around it freely.
                if let Err(error) =
                    crate::pipeline::fire::await_channel_progress(&cell, fires.as_ref()).await
                {
                    return Ok(Err(error));
                }
            }
        }
    }
}

/// The `channel` interface declares no free functions — only the `channel`
/// resource — but `add_to_linker` still requires the interface-level `Host`
/// bound, so this impl is empty by construction.
impl pie::inferlet::channel::Host for ProcessCtx {}

impl ProcessCtx {
    /// Frame submission (`submit(on, slots)`): exactly `model.frame-size()`
    /// ordered slots; slot i executes in wave i; `none` is a no-op. Delegates
    /// to [`crate::pipeline::fire::submit_frame`].
    ///
    /// Duplicated in WIT across all three forward interfaces, implemented once
    /// here. A frame is monomorphic in its slot type, which is exactly what
    /// keeps a hybrid's recurrent-only fire and its normal fire in the same
    /// frame while keeping an attention-only pass out of it entirely.
    async fn core_submit(
        &mut self,
        on: Resource<crate::pipeline::Pipeline>,
        slots: Vec<Option<Resource<ForwardPass>>>,
    ) -> Anyhow<Result<(), String>> {
        let slot_reps: Vec<Option<u32>> = slots
            .iter()
            .map(|slot| slot.as_ref().map(Resource::rep))
            .collect();
        for rep in slot_reps.iter().flatten() {
            let fwd: Resource<ForwardPass> = Resource::new_borrow(*rep);
            let _ = self.ctx().table.get(&fwd)?;
        }
        crate::inferlet::process::ensure_execution_admitted(self).await;
        crate::inferlet::process::gate::residency_gate(self).await?;
        crate::pipeline::fire::submit_frame(self, on, slot_reps).await
    }
}

impl pie::inferlet::channel::HostChannel for ProcessCtx {
    async fn new(
        &mut self,
        shape: Vec<u32>,
        dtype: pie::inferlet::types::Dtype,
        capacity: u32,
    ) -> Anyhow<Resource<Channel>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        // Pure host bookkeeping — never fails at construction (the WIT
        // constructor cannot carry a result; a channel/decl mismatch instead
        // errors at forward-pass.new / submit).
        use pie::inferlet::types::Dtype;
        let dtype = match dtype {
            Dtype::F32 => pie_ir::types::DType::F32,
            Dtype::I32 => pie_ir::types::DType::I32,
            Dtype::U32 => pie_ir::types::DType::U32,
            Dtype::Bool => pie_ir::types::DType::Bool,
        };
        let cell = Arc::new(Mutex::new(ChannelCell::new(shape, dtype, capacity)));
        Ok(self.ctx().table.push(Channel { cell, fires: None })?)
    }

    async fn put(&mut self, this: Resource<Channel>, value: Vec<u8>) -> Anyhow<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
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
            if let Err(error) = endpoint.wait_for_writer_change(observed_head).await {
                return Ok(Err(error.to_string()));
            }
        }
    }

    async fn set(&mut self, this: Resource<Channel>, value: Vec<u8>) -> Anyhow<Result<(), String>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
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

impl pie::inferlet::channel::HostChannelWithStore<ProcessCtx> for HasSelf<ProcessCtx> {
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

/// The single host implementation behind all three WIT forward interfaces.
///
/// WIT duplicates `embed` / `readout` / `program` / `submit` across `forward`,
/// `forward-recurrent`, and `forward-hybrid` so that cross-kind states are
/// UNREPRESENTABLE in the guest surface (forward_refactor.md D3). That is a
/// statement about the interface, not about the implementation: all three map
/// their `forward-pass` resource to this one Rust type, so the three trait
/// impls below are thin delegations to these methods and the binding logic
/// exists exactly once.
impl ProcessCtx {
    async fn core_new(&mut self, kind: PassKind) -> Anyhow<Resource<ForwardPass>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        Ok(self.ctx().table.push(ForwardPass::new(kind))?)
    }

    /// The interface-selection gate. `constructor()` is infallible in WIT, so
    /// the check lands on the first state-binding call instead -- which is
    /// still before anything is attached, and is the call whose SIGNATURE is
    /// the thing that differs.
    fn core_gate(&mut self, this: &Resource<ForwardPass>) -> Anyhow<Result<(), String>> {
        let kind = self.ctx().table.get(this)?.kind;
        let actual = model_pass_kind();
        if kind != actual {
            return Ok(Err(format!(
                "this model's forward pass is `{}`, but the pass was built through the `{}` \
                 interface; use `{}` instead. Attention-only state algorithms are not valid on \
                 a folded recurrent state.",
                actual.name(),
                kind.interface(),
                actual.interface(),
            )));
        }
        Ok(Ok(()))
    }

    async fn core_embed(
        &mut self,
        this: Resource<ForwardPass>,
        tokens: Resource<Channel>,
        indptr: Resource<Channel>,
    ) -> Anyhow<Result<(), String>> {
        let _ = self.ctx().table.get(&tokens)?;
        let _ = self.ctx().table.get(&indptr)?;
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.embed.is_some() {
            return Ok(Err(
                "forward pass embed binding is already attached".to_string()
            ));
        }
        pass.bindings.embed = Some(EmbedBinding {
            tokens: tokens.rep(),
            indptr: indptr.rep(),
        });
        Ok(Ok(()))
    }

    #[allow(clippy::too_many_arguments)]
    async fn core_attention(
        &mut self,
        this: Resource<ForwardPass>,
        kv_working_set: Resource<KvWorkingSet>,
        readable_pages: pie::inferlet::working_set::PageSpan,
        writable_pages: pie::inferlet::working_set::PageSpan,
        kv_len: Resource<Channel>,
        pages: Resource<Channel>,
        page_indptr: Resource<Channel>,
        w_slot: Resource<Channel>,
        w_off: Resource<Channel>,
        positions: Resource<Channel>,
        mask: Option<Resource<Channel>>,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this)? {
            return Ok(Err(error));
        }
        let readable = match page_span(readable_pages) {
            Ok(span) => span,
            Err(error) => return Ok(Err(error)),
        };
        let writable = match page_span(writable_pages) {
            Ok(span) => span,
            Err(error) => return Ok(Err(error)),
        };
        let _ = self.ctx().table.get(&kv_working_set)?;
        for channel in [&kv_len, &pages, &page_indptr, &w_slot, &w_off, &positions] {
            let _ = self.ctx().table.get(channel)?;
        }
        if let Some(mask) = mask.as_ref() {
            let _ = self.ctx().table.get(mask)?;
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.attention.is_some() {
            return Ok(Err(
                "forward pass attention binding is already attached".to_string()
            ));
        }
        pass.bindings.attention = Some(AttentionBinding {
            kv_ws: kv_working_set.rep(),
            readable,
            writable,
            kv_len: kv_len.rep(),
            pages: pages.rep(),
            page_indptr: page_indptr.rep(),
            w_slot: w_slot.rep(),
            w_off: w_off.rep(),
            positions: positions.rep(),
            mask: mask.map(|resource| resource.rep()),
        });
        Ok(Ok(()))
    }

    async fn core_readout(
        &mut self,
        this: Resource<ForwardPass>,
        indices: Resource<Channel>,
    ) -> Anyhow<Result<(), String>> {
        let _ = self.ctx().table.get(&indices)?;
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.readout.is_some() {
            return Ok(Err(
                "forward pass readout binding is already attached".to_string()
            ));
        }
        pass.bindings.readout = Some(indices.rep());
        Ok(Ok(()))
    }

    async fn core_set_max_layers(
        &mut self,
        this: Resource<ForwardPass>,
        max_layers: u32,
    ) -> Anyhow<Result<(), String>> {
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if max_layers == 0 {
            return Ok(Err("max-layers must be at least 1".to_string()));
        }
        pass.bindings.max_layers = Some(max_layers);
        Ok(Ok(()))
    }

    async fn core_program(
        &mut self,
        this: Resource<ForwardPass>,
        container_bytes: Vec<u8>,
        channels: Vec<Resource<Channel>>,
    ) -> Anyhow<Result<(), String>> {
        let (embed, attention, readout, rs_working_sets, rs_fold_len, rs_fold_len_rep, pass_max_layers) = {
            let pass = self.ctx().table.get(&this)?;
            if pass.is_bound() {
                return Ok(Err("forward pass program is already attached".to_string()));
            }
            let Some(embed) = pass.bindings.embed else {
                return Ok(Err(
                    "forward pass embed binding must be attached before program".to_string(),
                ));
            };
            let Some(attention) = pass.bindings.attention else {
                return Ok(Err(
                    "forward pass attention binding must be attached before program".to_string(),
                ));
            };
            (
                embed,
                attention,
                pass.bindings.readout,
                pass.bindings
                    .rs_ws
                    .iter()
                    .copied()
                    .map(Resource::new_borrow)
                    .collect::<Vec<Resource<RsWorkingSet>>>(),
                pass.bindings.rs_fold_len.clone(),
                pass.bindings.rs_geom.map(|geom| geom.fold_len),
                pass.bindings.max_layers,
            )
        };
        let kv_working_set: Resource<KvWorkingSet> = Resource::new_borrow(attention.kv_ws);
        let readable_pages = attention.readable;
        let writable_pages = attention.writable;
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
            // Strict admission: the hash-deduped program compile above may
            // prewarm, but everything from here on creates per-instance
            // driver state (channel registration, instance bind) or claims
            // pooled KV (device-geometry seed pages) — execution first.
            crate::inferlet::process::ensure_bind_admitted(self).await;

            // Validate every handle against its dense declaration BEFORE
            // stamping any of them, so a failed `program` attachment binds nothing.
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
            let channel_reps = channels.iter().map(Resource::rep).collect::<Vec<_>>();
            let expected = [
                (Port::EmbedTokens, Some(embed.tokens)),
                (Port::EmbedIndptr, Some(embed.indptr)),
                (Port::KvLen, Some(attention.kv_len)),
                (Port::Pages, Some(attention.pages)),
                (Port::PageIndptr, Some(attention.page_indptr)),
                (Port::WSlot, Some(attention.w_slot)),
                (Port::WOff, Some(attention.w_off)),
                (Port::Positions, Some(attention.positions)),
                (Port::AttnMask, attention.mask),
                (Port::Readout, readout),
            ];
            if let Err(error) =
                validate_descriptor_bindings(&prog.bound.container, &channel_reps, &expected)
            {
                return Ok(Err(error));
            }
            // Optional: a pass that folds everything unconditionally has
            // nothing to compute and claims no port, so the traced program
            // legitimately lacks this binding.
            if let Some(fold_len) = rs_fold_len_rep {
                if let Err(error) = validate_optional_descriptor_bindings(
                    &prog.bound.container,
                    &channel_reps,
                    &[(Port::RsFoldLen, Some(fold_len))],
                ) {
                    return Ok(Err(error));
                }
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

            let readable = readable_pages;
            let writable = writable_pages;
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
            // RV-26 extension: a Track B pass whose AttnMask binds a CHANNEL
            // ships the mask as a descriptor the driver must resolve
            // per-step; a driver that does not claim the mask port cannot
            // (CUDA today), and admitting the pass poisons at frame prepare
            // ("descriptor channel not ready"). Such a pass takes the same
            // loud Host fallback as a geometry-incapable driver.
            let needs_mask_port = prog.bound.container.ports.iter().any(|binding| {
                matches!(binding.port, pie_ir::registry::Port::AttnMask)
                    && matches!(
                        binding.source,
                        pie_ir::container::PortSource::Channel(_)
                    )
            });
            let devgeo_capable = device_port_mask & pie_driver_abi::PIE_DEVICE_GEOMETRY_PORTS
                == pie_driver_abi::PIE_DEVICE_GEOMETRY_PORTS
                && (!needs_mask_port
                    || device_port_mask & pie_driver_abi::PIE_DEVICE_PORT_ATTN_MASK != 0);
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
                        matches!(p.port, pie_ir::registry::Port::AttnMask)
                            && matches!(p.source, pie_ir::container::PortSource::Channel(_))
                    });
                    Some(DevGeo {
                        lease,
                        b,
                        fresh_dense,
                        w_cont_dense,
                        has_mask,
                        pooled: false,
                    })
                }
                None => None,
            };

            let taint = prog.geometry_taint();
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
            // A decode loop that carries a DENSE device `AttnMask` has no home
            // in the envelope class (the compose paths hold no per-lane mask
            // state) and none in the Host class (a device-sampled token is not
            // host-derivable). It DOES have one in the pool-owned device
            // geometry class: the descriptor resolver reads every port —
            // including the mask — straight from the channel cells, and such
            // fires are scheduled solo. Take that route rather than letting the
            // fire die on the first value the host cannot know.
            let devgeo = match devgeo {
                Some(devgeo) => Some(devgeo),
                None if decode_envelope.is_none()
                    && devgeo_capable
                    && !taint.host_derivable()
                    && readable.start == 0
                    && readable.end.is_none() =>
                {
                    crate::pipeline::fire::lease::detect_pooled_device_geometry(
                        &prog.bound.container,
                    )
                    .map(|lanes| {
                        tracing::info!(
                            "masked device-carried decode ({lanes} lane(s)) executes as a \
                             pool-owned device-geometry pass"
                        );
                        crate::pipeline::fire::lease::DevGeo::pooled(lanes, true)
                    })
                }
                None => None,
            };
            let geometry_class = if devgeo.is_some() {
                pie_driver_abi::GeometryClass::DeviceGeometry
            } else if decode_envelope.is_some() {
                pie_driver_abi::GeometryClass::DecodeEnvelope
            } else {
                pie_driver_abi::GeometryClass::Host
            };
            let rs_reps = rs_working_sets.iter().map(Resource::rep).collect();


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
                            pie_ir::container::ExternDir::Import => {
                                pie_driver_abi::PIE_CHANNEL_EXTERN_IMPORT
                            }
                            pie_ir::container::ExternDir::Export => {
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
                launch: prog.launch().clone(),
                reference_ptir: prog.bytes.clone(),
                ..Default::default()
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
                matches!(p.port, pie_ir::registry::Port::AttnMask)
                    && matches!(p.source, pie_ir::container::PortSource::Channel(_))
            });
            let host_shadow = crate::pipeline::fire::shadow::HostShadow::new(
                &instance.program.bound,
                instance.program.shadow_plan(),
                &instance.seeds,
            );
            let bound = BoundForwardPass {
                instance,
                bound_instance,
                scheduler,
                cells,
                channel_reps,
                fires: None,
                kv_ws: ws_rep,
                kv_declaration: crate::pipeline::instance::KvDeclaration { readable, writable },
                max_layers: pass_max_layers,
                rs_ws: rs_reps,
                rs_fold_len,
                kv_declaration_realized: false,
                failed: None,
                devgeo,
                decode_envelope,
                dense_mask,
                host_shadow,
                closed: false,
            };
            if let Err(error) = self.ctx().table.get_mut(&this)?.attach_bound(bound) {
                return Ok(Err(format!("pipeline: {error}")));
            }
            Ok(Ok(()))
        }
    }

    async fn core_set_rs_working_sets(
        &mut self,
        this: Resource<ForwardPass>,
        rs_working_sets: Vec<Resource<RsWorkingSet>>,
        geometry: RsGeometryBinding,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this)? {
            return Ok(Err(error));
        }
        if rs_working_sets.is_empty() {
            return Ok(Err(
                "forward pass recurrent-state binding needs one working set per request"
                    .to_string(),
            ));
        }
        let fold_len = match self.read_fold_len(&geometry, rs_working_sets.len())? {
            Ok(lens) => lens,
            Err(error) => return Ok(Err(error)),
        };
        if !self.ctx().table.get(&this)?.is_bound() {
            for resource in &rs_working_sets {
                let _ = self.ctx().table.get(resource)?;
            }
            let pass = self.ctx().table.get_mut(&this)?;
            pass.bindings.rs_ws = rs_working_sets.iter().map(Resource::rep).collect();
            pass.bindings.rs_geom = Some(geometry);
            pass.bindings.rs_fold_len = fold_len;
            return Ok(Ok(()));
        }
        let has_recurrent_state = crate::model::model().rs_caps().state_size > 0;
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

        let replacement = reps.clone();
        let result = self
            .ctx()
            .table
            .get_mut(&this)?
            .replace_rs_working_sets(reps)
            .map_err(|error| format!("pipeline: {error}"));
        if result.is_ok() {
            let pass = self.ctx().table.get_mut(&this)?;
            pass.bindings.rs_ws = replacement;
            pass.bindings.rs_geom = Some(geometry);
            pass.bindings.rs_fold_len = fold_len.clone();
            if let Ok(bound) = pass.bound_mut() {
                bound.rs_fold_len = fold_len;
            }
        }
        Ok(result)
    }

    /// The host-known value of `rs-geometry.fold-len`, one entry per bound
    /// working set.
    ///
    /// `Ok(None)` when the channel carries no host-known seed. That is not an
    /// error: it means a stage computes the fold length ON DEVICE, and the
    /// value reaches the driver through the `rs_fold_len` descriptor port
    /// instead of through this host. The host then keeps only an UPPER BOUND
    /// on the folded boundary — see `store::rs::Occupancy` — which is
    /// the whole point of making the field a channel
    /// (`.wiki/designs/linear-state-programming-model.md` §4.2): a speculative
    /// decode's accepted count never round-trips through the host between the
    /// fire that computes it and the fire that folds it.
    fn read_fold_len(
        &mut self,
        geometry: &RsGeometryBinding,
        rows: usize,
    ) -> Anyhow<Result<Option<Vec<u32>>, String>> {
        let resource: Resource<Channel> = Resource::new_borrow(geometry.fold_len);
        let cell = self.ctx().table.get(&resource)?.cell.clone();
        let cell = cell.lock().unwrap();
        if !matches!(cell.dtype, DType::U32 | DType::I32) {
            return Ok(Err(format!(
                "forward pass: rs-geometry.fold-len must be a u32/i32 channel, got {:?}",
                cell.dtype
            )));
        }
        let Ok(bytes) = cell.peek_seed() else {
            return Ok(Ok(None));
        };
        if bytes.len() % 4 != 0 {
            return Ok(Err(format!(
                "forward pass: rs-geometry.fold-len holds {} byte(s), which is not a whole number \
                 of 32-bit values",
                bytes.len()
            )));
        }
        let mut lens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        if lens.len() == 1 && rows > 1 {
            lens = vec![lens[0]; rows];
        }
        if lens.len() != rows {
            return Ok(Err(format!(
                "forward pass: rs-geometry.fold-len supplies {} value(s) for {rows} bound \
                 recurrent-state working set(s)",
                lens.len()
            )));
        }
        Ok(Ok(Some(lens)))
    }

    async fn core_drop(&mut self, this: Resource<ForwardPass>) -> Anyhow<()> {
        // A pass can be dropped before its pipeline. Drain the shared FIFO first
        // so every callback, mirror publication, and KV/RS transaction completes
        // before raw mirror pointers are detached or pages become reusable.
        let fires = self
            .ctx()
            .table
            .get(&this)?
            .bound()
            .ok()
            .and_then(|pass| pass.fires.clone());
        if let Some(fires) = fires {
            crate::pipeline::fire::finalize_all(self, &fires, false).await?;
        }

        // Native teardown (close the driver instance, detach every bound
        // cell, reclaim device-geometry grants) is idempotent and shared
        // with the `Drop` fallback (`ForwardPass::close_native`) — the local
        // `pass` binding below drops at the end of this function and would
        // otherwise repeat the work, but `close_native` no-ops the second
        // time.
        let mut pass = self.ctx().table.delete(this)?;
        if let Ok(bound) = pass.bound_mut() {
            bound.close_native();
        }
        Ok(())
    }
}

/// Shared body of the three `HostForwardPass` impls. Everything except the
/// state-binding call and the fold-mode methods is IDENTICAL by construction --
/// the WIT duplication exists to make cross-kind states unrepresentable in the
/// guest, not to fork the host.
macro_rules! forward_pass_common {
    ($iface:ident, $kind:expr) => {
        async fn new(&mut self) -> Anyhow<Resource<ForwardPass>> {
            self.core_new($kind).await
        }

        async fn embed(
            &mut self,
            this: Resource<ForwardPass>,
            tokens: Resource<Channel>,
            indptr: Resource<Channel>,
        ) -> Anyhow<Result<(), String>> {
            self.core_embed(this, tokens, indptr).await
        }

        async fn readout(
            &mut self,
            this: Resource<ForwardPass>,
            indices: Resource<Channel>,
        ) -> Anyhow<Result<(), String>> {
            self.core_readout(this, indices).await
        }

        async fn set_max_layers(
            &mut self,
            this: Resource<ForwardPass>,
            max_layers: u32,
        ) -> Anyhow<Result<(), String>> {
            self.core_set_max_layers(this, max_layers).await
        }

        async fn program(
            &mut self,
            this: Resource<ForwardPass>,
            container_bytes: Vec<u8>,
            channels: Vec<Resource<Channel>>,
        ) -> Anyhow<Result<(), String>> {
            self.core_program(this, container_bytes, channels).await
        }

        async fn drop(&mut self, this: Resource<ForwardPass>) -> Anyhow<()> {
            self.core_drop(this).await
        }
    };
}

/// Converts an interface-local `rs-geometry` record into the host binding.
/// Written as a macro rather than a trait because each interface generates its
/// OWN nominally distinct record type (D8): they are structurally identical
/// today, and adding a field to one must not move the other's ABI.
macro_rules! rs_geometry_binding {
    ($self:ident, $geom:expr) => {{
        let geom = $geom;
        let buffer = match page_span(geom.buffer) {
            Ok(span) => span,
            Err(error) => return Ok(Err(error)),
        };
        let _ = $self.ctx().table.get(&geom.fold_len)?;
        RsGeometryBinding {
            fold_len: geom.fold_len.rep(),
            buffer,
        }
    }};
}

// ---------------------------------------------------------------------------
// pie:inferlet/forward — attention only.
// ---------------------------------------------------------------------------

impl pie::inferlet::forward::Host for ProcessCtx {
    async fn submit(
        &mut self,
        on: Resource<crate::pipeline::Pipeline>,
        slots: Vec<Option<Resource<ForwardPass>>>,
    ) -> Anyhow<Result<(), String>> {
        self.core_submit(on, slots).await
    }

    async fn park(&mut self, on: Resource<crate::pipeline::Pipeline>) -> Anyhow<()> {
        crate::pipeline::fire::park_frame(self, on)
    }
}

impl pie::inferlet::forward::HostForwardPass for ProcessCtx {
    forward_pass_common!(forward, PassKind::Attention);

    async fn attention(
        &mut self,
        this: Resource<ForwardPass>,
        kv: Resource<crate::store::kv::working_set::KvWorkingSet>,
        geom: pie::inferlet::forward::KvGeometry,
    ) -> Anyhow<Result<(), String>> {
        self.core_attention(
            this,
            kv,
            geom.readable_pages,
            geom.writable_pages,
            geom.kv_len,
            geom.pages,
            geom.page_indptr,
            geom.w_slot,
            geom.w_off,
            geom.positions,
            geom.mask,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// pie:inferlet/forward-recurrent — folded recurrent state only.
// ---------------------------------------------------------------------------

impl pie::inferlet::forward_recurrent::Host for ProcessCtx {
    async fn submit(
        &mut self,
        on: Resource<crate::pipeline::Pipeline>,
        slots: Vec<Option<Resource<ForwardPass>>>,
    ) -> Anyhow<Result<(), String>> {
        self.core_submit(on, slots).await
    }

    async fn park(&mut self, on: Resource<crate::pipeline::Pipeline>) -> Anyhow<()> {
        crate::pipeline::fire::park_frame(self, on)
    }
}

impl pie::inferlet::forward_recurrent::HostForwardPass for ProcessCtx {
    forward_pass_common!(forward_recurrent, PassKind::Recurrent);

    async fn attention(
        &mut self,
        this: Resource<ForwardPass>,
        rs: Vec<Resource<RsWorkingSet>>,
        geom: pie::inferlet::forward_recurrent::RsGeometry,
    ) -> Anyhow<Result<(), String>> {
        let geometry = rs_geometry_binding!(self, geom);
        self.core_set_rs_working_sets(this, rs, geometry).await
    }
}

// ---------------------------------------------------------------------------
// pie:inferlet/forward-hybrid — attention and recurrent layers in one forward.
// ---------------------------------------------------------------------------

impl pie::inferlet::forward_hybrid::Host for ProcessCtx {
    async fn submit(
        &mut self,
        on: Resource<crate::pipeline::Pipeline>,
        slots: Vec<Option<Resource<ForwardPass>>>,
    ) -> Anyhow<Result<(), String>> {
        self.core_submit(on, slots).await
    }

    async fn park(&mut self, on: Resource<crate::pipeline::Pipeline>) -> Anyhow<()> {
        crate::pipeline::fire::park_frame(self, on)
    }
}

impl pie::inferlet::forward_hybrid::HostForwardPass for ProcessCtx {
    forward_pass_common!(forward_hybrid, PassKind::Hybrid);

    async fn attention(
        &mut self,
        this: Resource<ForwardPass>,
        kv: Option<pie::inferlet::forward_hybrid::KvBinding>,
        rs: Vec<Resource<RsWorkingSet>>,
        rs_geom: pie::inferlet::forward_hybrid::RsGeometry,
    ) -> Anyhow<Result<(), String>> {
        let Some(kv) = kv else {
            // The WIT admits `none` so that a recurrent-only commit fire is
            // expressible without inventing dummy attention geometry.
            // The bound-pass representation still requires a KV working set
            // (`BoundForwardPass.kv_ws`), so wiring that arm end to end is
            // follow-up work; today a commit fire binds KV like any
            // other and this is the honest error rather than a silent
            // half-bound pass.
            return Ok(Err(
                "forward pass: a hybrid pass with no attention binding is not supported yet; \
                 bind the KV working set even for a recurrent-only fire"
                    .to_string(),
            ));
        };
        let geometry = rs_geometry_binding!(self, rs_geom);
        let geom = kv.geometry;
        if let Err(error) = self
            .core_attention(
                Resource::new_borrow(this.rep()),
                kv.working_set,
                geom.readable_pages,
                geom.writable_pages,
                geom.kv_len,
                geom.pages,
                geom.page_indptr,
                geom.w_slot,
                geom.w_off,
                geom.positions,
                geom.mask,
            )
            .await?
        {
            return Ok(Err(error));
        }
        self.core_set_rs_working_sets(this, rs, geometry).await
    }
}

#[cfg(test)]
mod descriptor_binding_tests {
    use super::*;
    use pie_ir::container::PortBinding;
    use pie_ir::types::{DType, Shape};

    fn container(ports: Vec<PortBinding>) -> TraceContainer {
        TraceContainer {
            names: Vec::new(),
            externs: Vec::new(),
            channels: Vec::new(),
            ports,
            stages: Vec::new(),
        }
    }

    #[test]
    fn optional_attention_mask_is_omitted_or_channel_bound() {
        let no_mask = container(Vec::new());
        validate_descriptor_bindings(&no_mask, &[], &[(Port::AttnMask, None)]).unwrap();

        let mask = container(vec![PortBinding {
            port: Port::AttnMask,
            source: PortSource::Channel(0),
        }]);
        validate_descriptor_bindings(&mask, &[41], &[(Port::AttnMask, Some(41))]).unwrap();
        assert!(
            validate_descriptor_bindings(&mask, &[41], &[(Port::AttnMask, None)])
                .unwrap_err()
                .contains("without a resource attachment")
        );
    }

    #[test]
    fn descriptor_attachment_rejects_constants_and_wrong_resources() {
        let constant = container(vec![PortBinding {
            port: Port::EmbedIndptr,
            source: PortSource::Const {
                dtype: DType::U32,
                shape: Shape::vector(2),
                data: [0u32, 1].into_iter().flat_map(u32::to_le_bytes).collect(),
            },
        }]);
        assert!(
            validate_descriptor_bindings(&constant, &[], &[(Port::EmbedIndptr, Some(7))])
                .unwrap_err()
                .contains("must be channel-bound")
        );

        let channel = container(vec![PortBinding {
            port: Port::EmbedIndptr,
            source: PortSource::Channel(0),
        }]);
        assert!(
            validate_descriptor_bindings(&channel, &[8], &[(Port::EmbedIndptr, Some(7))])
                .unwrap_err()
                .contains("attached resource is 7")
        );
    }
}
