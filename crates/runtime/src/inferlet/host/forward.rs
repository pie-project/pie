//! WIT host glue for `pie:inferlet/forward`: `Host`/`HostChannel`/
//! `HostForwardPass` impls over the pipeline-owned `Channel`/`ForwardPass`.

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

use eta_ir::container::{HostRole, PortSource, TraceContainer};
use eta_ir::registry::{GeometryClass, Port, PortMask};
use eta_ir::types::Dtype;

use super::pie;

type Anyhow<T> = anyhow::Result<T>;

/// Which forward interface this model requires; must match `model.pass-kind()`
/// (`host/model.rs`).
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

/// The first field by which `next` departs from `existing`, named as the guest
/// sees it in `kv-geometry` (`kv-working-set` for the working set itself), or
/// `None` when a rebind re-states the attention binding exactly. Reps are
/// compared, not values: a bound program's ports are tied to these channels.
fn attention_rebind_diff(
    existing: &AttentionBinding,
    next: &AttentionBinding,
) -> Option<&'static str> {
    if existing.kv_ws != next.kv_ws {
        return Some("kv-working-set");
    }
    if existing.readable != next.readable {
        return Some("readable-pages");
    }
    if existing.writable != next.writable {
        return Some("writable-pages");
    }
    if existing.kv_len != next.kv_len {
        return Some("kv-len");
    }
    if existing.pages != next.pages {
        return Some("pages");
    }
    if existing.page_indptr != next.page_indptr {
        return Some("page-indptr");
    }
    if existing.w_slot != next.w_slot {
        return Some("w-slot");
    }
    if existing.w_off != next.w_off {
        return Some("w-off");
    }
    if existing.positions != next.positions {
        return Some("positions");
    }
    if existing.mask.is_some() != next.mask.is_some() {
        return Some("mask (present on one binding, absent on the other)");
    }
    if existing.mask != next.mask {
        return Some("mask");
    }
    None
}

fn validate_descriptor_bindings(
    container: &TraceContainer,
    channel_reps: &[u32],
    expected: &[(Port, Option<u32>)],
) -> Result<(), String> {
    validate_bindings(container, channel_reps, expected, false)
}

/// RS geometry ports are optional: absent-but-attached is legal here (unlike
/// the KV family); present-but-mismatched is an error for both.
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

    // Only pops an already-settled FIFO entry; caller holds the finalizer gate.
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
                // Idle channel wait holds no pooled state; planner may evict around it.
                if let Err(error) =
                    crate::pipeline::fire::await_channel_progress(&cell, fires.as_ref()).await
                {
                    return Ok(Err(error));
                }
            }
        }
    }
}

/// `take-blocking` / `read-blocking`: the same polling loop as
/// [`materialize_channel`], driven from a plain `async fn(&mut self)` host
/// import rather than an `Accessor`. Holding the store across the awaits is
/// what "blocking" means here: the guest's task is suspended inside the
/// call, nothing else in the instance runs, and every await below is on
/// engine-side progress (fire settlement, the reader wait slot) that never
/// needs the store to advance.
async fn materialize_channel_blocking(
    ctx: &mut ProcessCtx,
    this: Resource<Channel>,
    mode: ChannelReadMode,
) -> Anyhow<Result<Vec<u8>, String>> {
    let mut settle_ready_take = true;
    loop {
        let state = poll_channel(ctx, &this, mode, false, settle_ready_take)?;
        let state = match state {
            ChannelPoll::Pending {
                fires: Some(fires), ..
            } => {
                let _finalize_guard = fires.finalize_guard().await;
                let state = poll_channel(ctx, &this, mode, true, settle_ready_take)?;
                match state {
                    ChannelPoll::Finalize(op) => {
                        let finalized = crate::pipeline::fire::finalize_op_await(op).await?;
                        crate::pipeline::fire::complete_finalize(ctx, finalized);
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
                if let Err(error) =
                    crate::pipeline::fire::await_channel_progress(&cell, fires.as_ref()).await
                {
                    return Ok(Err(error));
                }
            }
        }
    }
}

// `add_to_linker` requires the interface-level `Host` bound even though
// `channel` declares no free functions, so this impl is empty by construction.
impl pie::inferlet::channel::Host for ProcessCtx {}

impl ProcessCtx {
    /// `submit(on, slots)`: exactly `model.frame-size()` ordered slots; slot i
    /// executes in wave i; `none` is a no-op. Shared by all three forward
    /// interfaces (WIT duplicates the signature, not the implementation).
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
        // Construction never fails; a channel/decl mismatch errors later at
        // forward-pass.new / submit instead (the WIT constructor has no Result).
        use pie::inferlet::types::Dtype;
        let dtype = match dtype {
            Dtype::F32 => eta_ir::types::Dtype::F32,
            Dtype::I32 => eta_ir::types::Dtype::I32,
            Dtype::U32 => eta_ir::types::Dtype::U32,
            Dtype::Bool => eta_ir::types::Dtype::Bool,
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

    /// The sync-lowerable `take`; see the WIT door for who needs it.
    async fn take_blocking(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        materialize_channel_blocking(self, this, ChannelReadMode::Take).await
    }

    /// The sync-lowerable `read`.
    async fn read_blocking(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        materialize_channel_blocking(self, this, ChannelReadMode::Read).await
    }

    async fn drop(&mut self, this: Resource<Channel>) -> Anyhow<()> {
        // A bound pass holds its own Arc, so dropping the guest handle never
        // dangles an in-flight fire; storage releases when the instance closes.
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::inferlet::channel::HostChannelWithStore<ProcessCtx> for HasSelf<ProcessCtx> {
    /// While empty: drains already-settled pipeline ops, then parks on the
    /// channel's reader wait slot. Store access stays scoped to synchronous
    /// polls; never holds an `Accessor` borrow across an await.
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

/// Single host implementation behind all three WIT forward interfaces; WIT
/// duplicates the interface so cross-kind states are unrepresentable in the
/// guest, but all three map to this one Rust type.
impl ProcessCtx {
    async fn core_new(&mut self, kind: PassKind) -> Anyhow<Resource<ForwardPass>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        Ok(self.ctx().table.push(ForwardPass::new(kind))?)
    }

    /// Interface-selection gate, checked on the first state-binding call since
    /// `constructor()` is infallible in WIT.
    fn core_gate(&mut self, this: &Resource<ForwardPass>) -> Anyhow<Result<(), String>> {
        let kind = self.ctx().table.get(this)?.kind;
        let actual = model_pass_kind();
        // A HYBRID PASS WITH NO RECURRENT STATE IS AN ATTENTION PASS. The
        // hybrid interface already makes one half of its state optional
        // (`kv: none` for a recurrent-only fire); this is the other half. A
        // program that only reads logits binds `rs = []` and runs on every
        // KV-carrying model, and its fold policy is a value it states rather
        // than a type it picks. The gate stays for the attention interface on
        // a folding model: that interface carries the KV-editing verbs
        // (`discard`, `fork`, `slice`) which are wrong on a fold, and the
        // hybrid interface has none of them. `validate_count` refuses a
        // non-empty `rs` on an attention model, so the leniency ends where
        // the state does.
        if kind == PassKind::Hybrid && actual == PassKind::Attention {
            return Ok(Ok(()));
        }
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
        let binding = AttentionBinding {
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
        };
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            // A rebind. The hybrid `attention` verb states both halves of the
            // state in one call, and a guest that only wants new recurrent
            // working sets (a beam fork) re-states the KV half with it; the
            // SDK already treats a post-attach `attention` as a rebind and
            // claims no ports. The compiled program's ports are tied to these
            // channels, so the KV half may be re-stated but never changed:
            // an identical statement is a no-op, a differing one is refused
            // by the field that differs.
            let Some(existing) = pass.bindings.attention else {
                return Ok(Err("forward pass program is already attached".to_string()));
            };
            return Ok(match attention_rebind_diff(&existing, &binding) {
                None => Ok(()),
                Some(field) => Err(format!(
                    "forward pass attention binding cannot change after the program is \
                     attached (`{field}` differs); rebind with the same KV geometry, or build \
                     a new pass"
                )),
            });
        }
        if pass.bindings.attention.is_some() {
            return Ok(Err(
                "forward pass attention binding is already attached".to_string()
            ));
        }
        pass.bindings.attention = Some(binding);
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

    async fn core_set_drafting_block(
        &mut self,
        this: Resource<ForwardPass>,
        on: bool,
    ) -> Anyhow<Result<(), String>> {
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        pass.bindings.block_draft = on;
        Ok(Ok(()))
    }

    /// Spans are cloned by handle (`Arc`), so a decoded image submitted to two
    /// passes decodes once. Their position in the sequence is not recorded
    /// here — it's scanned out of the submitted tokens at submit time
    /// (`pipeline::media::scan`).
    async fn core_media(
        &mut self,
        this: Resource<ForwardPass>,
        spans: Vec<pie::inferlet::forward::MediaSpan>,
    ) -> Anyhow<Result<(), String>> {
        use pie::inferlet::forward::MediaSpan;
        let mut attached = Vec::with_capacity(spans.len());
        for span in &spans {
            let encoded = match span {
                MediaSpan::Image(image) => {
                    let handle: Resource<crate::inferlet::host::media::Image> =
                        Resource::new_borrow(image.rep());
                    std::sync::Arc::clone(&self.ctx().table.get(&handle)?.span)
                }
                MediaSpan::Audio(audio) => {
                    let handle: Resource<crate::inferlet::host::media::Audio> =
                        Resource::new_borrow(audio.rep());
                    std::sync::Arc::clone(&self.ctx().table.get(&handle)?.span)
                }
            };
            attached.push(encoded);
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if !pass.bindings.media.is_empty() {
            return Ok(Err(
                "forward pass media spans are already attached".to_string()
            ));
        }
        pass.bindings.media = attached;
        Ok(Ok(()))
    }

    async fn core_program(
        &mut self,
        this: Resource<ForwardPass>,
        container_bytes: Vec<u8>,
        channels: Vec<Resource<Channel>>,
    ) -> Anyhow<Result<(), String>> {
        let (
            embed,
            attention,
            readout,
            rs_working_sets,
            rs_fold_len,
            rs_fold_len_rep,
            pass_max_layers,
            pass_block_draft,
        ) = {
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
                pass.bindings.block_draft,
            )
        };
        let kv_working_set: Resource<KvWorkingSet> = Resource::new_borrow(attention.kv_ws);
        let readable_pages = attention.readable;
        let writable_pages = attention.writable;
        {
            // Hash-deduped compile/bind cache; a malformed trace fails here.
            let prog = match crate::pipeline::program::register(
                container_bytes,
                &crate::pipeline::program::model_profile(),
            ) {
                Ok(p) => p,
                Err(e) => return Ok(Err(e.to_string())),
            };
            // Everything from here on creates per-instance engine state or
            // claims pooled KV, so admission is required first.
            crate::inferlet::process::ensure_bind_admitted(self).await;

            // Validate every handle before stamping any, so a failed
            // attachment binds nothing.
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
            // A pass that folds unconditionally claims no port, so the
            // program may legitimately lack this binding.
            if let Some(fold_len) = rs_fold_len_rep
                && let Err(error) = validate_optional_descriptor_bindings(
                    &prog.bound.container,
                    &channel_reps,
                    &[(Port::RsFoldLen, Some(fold_len))],
                )
            {
                return Ok(Err(error));
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
                    // A channel may bind to several passes; decl equality
                    // across sharing passes is still validated as a conflict.
                    let extern_binding = extern_bindings[i]
                        .as_ref()
                        .map(|(name, dir)| (name.as_str(), *dir));
                    if let Err(e) = c.validate_attachment(&decls[i], extern_binding) {
                        // The handle-list index alone cannot be chased: it says
                        // WHERE in this pass the channel sits, never WHICH
                        // channel it is. `global_id` is the object's identity,
                        // so two passes naming the same id is visible from the
                        // message instead of needing a debugger.
                        return Ok(Err(format!(
                            "pipeline: channel {i} (id {}): {e}",
                            c.global_id
                        )));
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
            let stores = crate::store::registry::get(bound_ws.model, bound_ws.engine);
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
            // Derivability decides the geometry class, not op-pattern arity:
            // host-derivable geometry is Host class on every engine; a
            // device-dependent envelope classifies DecodeEnvelope only when
            // the engine has the needed device geometry ports, else it falls
            // back to Host and blocks loudly on the first undecidable value.
            let device_port_mask =
                crate::engine::get_spec(bound_ws.engine)?.device_geometry_port_mask;

            // Device-geometry pass: the program traces its full explicit
            // geometry in-graph; the runtime only leases physical pages. If
            // AttnMask binds a channel, the engine must be able to resolve it
            // per-step (CUDA today cannot); otherwise it falls back to Host.
            let needs_mask_port = prog.bound.container.ports.iter().any(|binding| {
                matches!(binding.port, eta_ir::registry::Port::AttnMask)
                    && matches!(binding.source, eta_ir::container::PortSource::Channel(_))
            });
            let devgeo_capable = device_port_mask.covers(PortMask::DEVICE_GEOMETRY)
                && (!needs_mask_port || device_port_mask.covers(PortMask::of(&[Port::AttnMask])));
            let devgeo = match crate::pipeline::fire::lease::detect_device_geometry(
                &prog.bound.container,
            ) {
                Some(_) if !devgeo_capable => {
                    tracing::info!(
                        "device-geometry program on an engine without device geometry ports \
                         (mask {device_port_mask:?}): falling back to host-evaluated \
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
                    // Seed the lease with `B` fire-0 pages, one per lane.
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
                        matches!(p.port, eta_ir::registry::Port::AttnMask)
                            && matches!(p.source, eta_ir::container::PortSource::Channel(_))
                    });
                    Some(DevGeo {
                        lease,
                        b,
                        fresh_dense,
                        w_cont_dense,
                        has_mask,
                        pooled: false,
                        qo_indptr: None,
                    })
                }
                None => None,
            };

            let taint = prog.geometry_taint();
            // A device-carried decode that re-publishes EVERY descriptor port
            // — tokens, positions, pages, page bounds, kv length, write
            // targets — states its whole geometry in-graph, so the engine
            // resolves it there and the host only leases the pool. Asked
            // before the envelope class: an envelope still folds every port
            // but the token on the host, and a loop whose accepted count is
            // device-decided (a speculative window) has nothing for the host
            // to fold.
            let devgeo = match devgeo {
                Some(devgeo) => Some(devgeo),
                None if devgeo_capable
                    && !taint.host_derivable()
                    && readable.start == 0
                    && readable.end.is_none() =>
                {
                    crate::pipeline::fire::lease::detect_pooled_device_geometry(
                        &prog.bound.container,
                    )
                    .map(|qo_indptr| {
                        tracing::info!(
                            "device-carried decode ({} lane(s), {} row(s)) re-publishes every \
                             descriptor port: executes as a pool-owned device-geometry pass",
                            qo_indptr.len().saturating_sub(1),
                            qo_indptr.last().copied().unwrap_or(0)
                        );
                        crate::pipeline::fire::lease::DevGeo::pooled(qo_indptr, needs_mask_port)
                    })
                }
                None => None,
            };
            let decode_envelope = if devgeo.is_some() || taint.host_derivable() {
                None
            } else {
                let mut why = String::new();
                match crate::pipeline::fire::geometry::classify_decode_envelope_why(
                    &prog.bound.container,
                    &mut why,
                ) {
                    Ok(Some(envelope)) => {
                        let required =
                            crate::pipeline::fire::geometry::envelope_required_ports(&envelope);
                        if device_port_mask.covers(required) {
                            Some(envelope)
                        } else {
                            tracing::info!(
                                "decode envelope on an engine without device geometry ports \
                                 (mask {device_port_mask:?}, needs {required:?}): falling \
                                 back to host-evaluated serialized execution"
                            );
                            None
                        }
                    }
                    Ok(None) => {
                        tracing::info!(
                            "not a decode envelope: {why}; falling back to \
                             host-evaluated execution"
                        );
                        None
                    }
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
                GeometryClass::DeviceGeometry
            } else if decode_envelope.is_some() {
                GeometryClass::DecodeEnvelope
            } else {
                GeometryClass::Host
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
                registration_plans.push(::engine::ChannelRegistration {
                    id: cell.lock().unwrap().global_id,
                    shape: decls[dense].shape.dims().to_vec(),
                    dtype: decls[dense].dtype,
                    host_role: decls[dense].host_role,
                    seeded: decls[dense].seeded,
                    extern_dir: extern_binding.map(|(_, dir)| *dir),
                    capacity: decls[dense].capacity,
                    extern_name: extern_binding
                        .map(|(name, _)| name.as_bytes().to_vec())
                        .unwrap_or_default(),
                });
            }
            // Capture ids and stage seeds before the combined register+bind:
            // bind consumes only pre-known ids and host-staged seed bytes.
            let channel_ids: Vec<u64> = cells.iter().map(|c| c.lock().unwrap().global_id).collect();
            let channel_reps: Vec<u32> = channels.iter().map(|c| c.rep()).collect();
            let program_registration = ::engine::ProgramRegistration {
                program_hash: prog.hash,
                launch: prog.launch().clone(),
                reference_ptir: prog.bytes.clone(),
                ..Default::default()
            };
            let pricing_rows = prog.pricing.rows;
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
                // Native cell is one byte per bool; the engine wire ABI is
                // bit-packed, sized (numel + 7) / 8. Every other dtype is four
                // bytes either way and needs no repacking.
                let wire = if cell.dtype == eta_ir::types::Dtype::Bool {
                    let mut packed = vec![0u8; bytes.len().div_ceil(8)];
                    crate::pipeline::channel::pack_bool_into(&bytes, &mut packed);
                    packed
                } else {
                    bytes
                };
                seed_values.push(crate::engine::ChannelValue {
                    channel: cell.global_id,
                    bytes: wire,
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
                    // sampled_rows is how many readout rows the program reads,
                    // taken from the pricing already computed at registration
                    // rather than recomputed. Every other role stays at one row.
                    ::engine::BindExtents {
                        sampled_rows: pricing_rows.max(1),
                        ..::engine::BindExtents::default()
                    },
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
                // A seeded Writer held staging until the seed settled; flush
                // now so direct ring puts take over.
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
                block_draft: pass_block_draft,
                rs_ws: rs_reps,
                rs_fold_len,
                kv_declaration_realized: false,
                failed: None,
                devgeo,
                decode_envelope,
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
            // The attention case of a hybrid pass (see `core_gate`): nothing to
            // bind, and `RsGeometry` is a policy over a state this model does
            // not fold. A folding model still needs one set per request row.
            if model_pass_kind() == PassKind::Attention {
                return Ok(Ok(()));
            }
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
                match pass.instance.fire_geometry() {
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
            if rs.model != kv.model || rs.engine != kv.engine {
                return Ok(Err(format!(
                    "pipeline: rs-working-set at request row {row} belongs to model/engine \
                     ({}, {}), expected ({}, {})",
                    rs.model, rs.engine, kv.model, kv.engine
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

    /// Host-known value of `rs-geometry.fold-len`, one entry per bound working
    /// set. `Ok(None)` means the fold length is computed on device instead
    /// (reaches the engine via the `rs_fold_len` descriptor port); the host
    /// then keeps only an upper bound (see `store::rs::Occupancy`).
    fn read_fold_len(
        &mut self,
        geometry: &RsGeometryBinding,
        rows: usize,
    ) -> Anyhow<Result<Option<Vec<u32>>, String>> {
        let resource: Resource<Channel> = Resource::new_borrow(geometry.fold_len);
        let cell = self.ctx().table.get(&resource)?.cell.clone();
        let cell = cell.lock().unwrap();
        if !matches!(cell.dtype, Dtype::U32 | Dtype::I32) {
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
        // Drain the shared FIFO first so every callback and KV/RS transaction
        // completes before mirror pointers are detached or pages reused.
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

        // close_native is idempotent, shared with the Drop fallback.
        let mut pass = self.ctx().table.delete(this)?;
        if let Ok(bound) = pass.bound_mut() {
            bound.close_native();
        }
        Ok(())
    }
}

/// Shared body of the three `HostForwardPass` impls; only the state-binding
/// call differs per interface.
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

        async fn set_drafting_block(
            &mut self,
            this: Resource<ForwardPass>,
            on: bool,
        ) -> Anyhow<Result<(), String>> {
            self.core_set_drafting_block(this, on).await
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
/// A macro, not a trait, because each interface generates its own nominally
/// distinct record type.
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

    /// `media` rides the attention and hybrid interfaces (a hybrid tower
    /// family exists: qwen3.8-flash-next); recurrent-only gets it when one
    /// of those grows a tower.
    async fn media(
        &mut self,
        this: Resource<ForwardPass>,
        spans: Vec<pie::inferlet::forward::MediaSpan>,
    ) -> Anyhow<Result<(), String>> {
        self.core_media(this, spans).await
    }

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

    /// The attention interface's `media`, same host half: the span type is
    /// `forward`'s (`use forward.{media-span}`), so nothing is translated.
    async fn media(
        &mut self,
        this: Resource<ForwardPass>,
        spans: Vec<pie::inferlet::forward_hybrid::MediaSpan>,
    ) -> Anyhow<Result<(), String>> {
        self.core_media(this, spans).await
    }

    async fn attention(
        &mut self,
        this: Resource<ForwardPass>,
        kv: Option<pie::inferlet::forward_hybrid::KvBinding>,
        rs: Vec<Resource<RsWorkingSet>>,
        rs_geom: pie::inferlet::forward_hybrid::RsGeometry,
    ) -> Anyhow<Result<(), String>> {
        let Some(kv) = kv else {
            // `none` lets a recurrent-only commit fire be expressed without
            // dummy attention geometry, but BoundForwardPass still requires a
            // KV working set, so this path errors rather than half-binding.
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
mod tests {
    use super::attention_rebind_diff;
    use crate::pipeline::instance::{AttentionBinding, KvPageSpan};

    fn binding() -> AttentionBinding {
        AttentionBinding {
            kv_ws: 1,
            readable: KvPageSpan { start: 0, end: None },
            writable: KvPageSpan { start: 0, end: None },
            kv_len: 2,
            pages: 3,
            page_indptr: 4,
            w_slot: 5,
            w_off: 6,
            positions: 7,
            mask: Some(8),
        }
    }

    #[test]
    fn identical_rebind_is_a_no_op() {
        assert_eq!(attention_rebind_diff(&binding(), &binding()), None);
    }

    #[test]
    fn differing_rebind_names_the_field() {
        let mut next = binding();
        next.kv_ws = 9;
        assert_eq!(
            attention_rebind_diff(&binding(), &next),
            Some("kv-working-set")
        );
        let mut next = binding();
        next.writable = KvPageSpan { start: 0, end: Some(4) };
        assert_eq!(
            attention_rebind_diff(&binding(), &next),
            Some("writable-pages")
        );
        let mut next = binding();
        next.positions = 9;
        assert_eq!(attention_rebind_diff(&binding(), &next), Some("positions"));
        let mut next = binding();
        next.mask = None;
        assert_eq!(
            attention_rebind_diff(&binding(), &next),
            Some("mask (present on one binding, absent on the other)")
        );
        let mut next = binding();
        next.mask = Some(9);
        assert_eq!(attention_rebind_diff(&binding(), &next), Some("mask"));
    }
}
