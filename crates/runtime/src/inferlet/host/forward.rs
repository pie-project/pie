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
    AttentionBinding, BoundForwardPass, CanvasMode, EmbedBinding, FloatLane, LaneFacts, PassKind,
    PortBinding, RsGeometryBinding, lane_stream_of,
};
use crate::store::kv::working_set::KvWorkingSet;
use crate::store::rs::working_set::RsWorkingSet;

use eta_ir::container::{HostRole, PortSource, TraceContainer};
use eta_ir::registry::{GeometryClass, Port, PortMask};
use eta_ir::types::Dtype;

use super::pie;

type Anyhow<T> = anyhow::Result<T>;

fn model_pass_kind() -> PassKind {
    let model = crate::model::model();
    if model.diffusion().is_some() {
        return PassKind::Diffusion;
    }
    match (model.kv_page_size() > 0, model.rs_caps().state_size > 0) {
        (_, false) => PassKind::Attention,
        (true, true) => PassKind::Hybrid,
        (false, true) => PassKind::Recurrent,
    }
}

fn reading_of(pass: &ForwardPass) -> Result<Option<&'static models::ReadingFact>, String> {
    let model = crate::model::model();
    if let Some(index) = pass.bindings.reading {
        return Ok(model.readings().get(usize::from(index)));
    }
    if let Some(only) = model.sole_reading() {
        return Ok(Some(only));
    }
    if model.readings().is_empty() {
        return Ok(None);
    }
    Err(format!(
        "this model declares {} readings ({}) and the pass named none; call `reading(name)` \
         before `program`",
        model.readings().len(),
        reading_names(model.readings())
    ))
}

fn reading_names(readings: &[models::ReadingFact]) -> String {
    readings
        .iter()
        .map(|reading| format!("`{}`", reading.name))
        .collect::<Vec<_>>()
        .join(", ")
}

fn reading_lists_stream(reading: &models::ReadingFact, stream: models::Stream) -> bool {
    if reading.streams.is_empty() {
        return stream == models::Stream::Text;
    }
    reading.streams.contains(&stream)
}

fn stream_names(reading: &models::ReadingFact) -> String {
    if reading.streams.is_empty() {
        return "`text`".to_string();
    }
    reading
        .streams
        .iter()
        .map(|stream| format!("`{}`", stream.name()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn engine_port_kind(kind: models::PortKind) -> ::engine::fire::PortKind {
    use ::engine::fire::PortKind;
    match kind {
        models::PortKind::Latents => PortKind::Latents,
        models::PortKind::LaneVector => PortKind::LaneVector,
        models::PortKind::Context => PortKind::Context,
        models::PortKind::AxisPositions => PortKind::AxisPositions,
        models::PortKind::Voxels => PortKind::Voxels,
    }
}

pub(crate) fn validate_port_channel(
    port: &models::PortFact,
    shape: &[u32],
    dtype: Dtype,
) -> Result<Option<u32>, String> {
    if dtype != Dtype::F32 {
        return Err(format!(
            "port `{}` reads an f32 channel; this one is {dtype:?}",
            port.name
        ));
    }
    match port.kind {
        models::PortKind::LaneVector => match shape {
            [width] | [1, width] if *width == port.width => Ok(None),
            _ => Err(format!(
                "port `{}` is a lane vector of width {}: its channel must be `[{}]` or `[1, {}]` \
                 f32; this one is {shape:?}",
                port.name, port.width, port.width, port.width
            )),
        },
        models::PortKind::Latents | models::PortKind::Context | models::PortKind::AxisPositions => {
            match shape {
                [rows, width] if *width == port.width && *rows > 0 => Ok(Some(*rows)),
                _ => Err(format!(
                    "port `{}` reads `[rows, {}]` f32; this channel is {shape:?}",
                    port.name, port.width
                )),
            }
        }
        models::PortKind::Voxels => match shape {
            [h, w, width] if *width == port.width && *h > 0 && *w > 0 => Ok(Some(h * w)),
            [t, h, w, width] if *width == port.width && *t > 0 && *h > 0 && *w > 0 => {
                Ok(Some(t * h * w))
            }
            _ => Err(format!(
                "port `{}` reads one clip `[h, w, {}]` (or `[t, h, w, {}]`) f32 on the voxel \
                 axis; this channel is {shape:?}",
                port.name, port.width, port.width
            )),
        },
    }
}

pub(crate) fn port_clip(port: &models::PortFact, shape: &[u32]) -> Option<[u32; 3]> {
    if port.kind != models::PortKind::Voxels {
        return None;
    }
    match *shape {
        [h, w, width] if width == port.width && h > 0 && w > 0 => Some([1, h, w]),
        [t, h, w, width] if width == port.width && t > 0 && h > 0 && w > 0 => Some([t, h, w]),
        _ => None,
    }
}

pub(crate) fn port_rows(ports: &[PortBinding]) -> Result<Option<u32>, String> {
    let mut rows: Option<(u32, &str)> = None;
    let mut context_rows: Option<u32> = None;
    for port in ports {
        let Some(these) = port.rows else { continue };
        if port.kind == ::engine::fire::PortKind::Voxels {
            continue;
        }
        if port.kind == ::engine::fire::PortKind::Context {
            context_rows = context_rows.or(Some(these));
            continue;
        }
        match rows {
            None => rows = Some((these, &port.name)),
            Some((agreed, name)) if agreed != these => {
                return Err(format!(
                    "port `{}` binds {these} rows but port `{name}` binds {agreed}; a pass's \
                     latents and positions ports carry the same rows",
                    port.name
                ));
            }
            Some(_) => {}
        }
    }
    Ok(rows.map(|(rows, _)| rows).or(context_rows))
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
pub(crate) enum ChannelReadMode {
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
                if let Err(error) =
                    crate::pipeline::fire::await_channel_progress(&cell, fires.as_ref()).await
                {
                    return Ok(Err(error));
                }
            }
        }
    }
}

pub(crate) async fn materialize_channel_blocking(
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

impl pie::inferlet::channel::Host for ProcessCtx {}

impl ProcessCtx {
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

    async fn take_blocking(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        materialize_channel_blocking(self, this, ChannelReadMode::Take).await
    }

    async fn read_blocking(&mut self, this: Resource<Channel>) -> Anyhow<Result<Vec<u8>, String>> {
        materialize_channel_blocking(self, this, ChannelReadMode::Read).await
    }

    async fn drop(&mut self, this: Resource<Channel>) -> Anyhow<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::inferlet::channel::HostChannelWithStore<ProcessCtx> for HasSelf<ProcessCtx> {
    async fn take(
        accessor: &Accessor<ProcessCtx, Self>,
        this: Resource<Channel>,
    ) -> Anyhow<Result<Vec<u8>, String>> {
        materialize_channel(accessor, this, ChannelReadMode::Take).await
    }

    async fn read(
        accessor: &Accessor<ProcessCtx, Self>,
        this: Resource<Channel>,
    ) -> Anyhow<Result<Vec<u8>, String>> {
        materialize_channel(accessor, this, ChannelReadMode::Read).await
    }
}

impl ProcessCtx {
    async fn core_new(&mut self, kind: PassKind) -> Anyhow<Resource<ForwardPass>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        Ok(self.ctx().table.push(ForwardPass::new(kind))?)
    }

    fn core_gate(
        &mut self,
        this: &Resource<ForwardPass>,
        named: Option<&'static models::ReadingFact>,
    ) -> Anyhow<Result<(), String>> {
        let kind = self.ctx().table.get(this)?.kind;
        let actual = model_pass_kind();
        if kind == PassKind::Attention {
            let reading = match named {
                Some(reading) => Ok(Some(reading)),
                None => reading_of(self.ctx().table.get(this)?),
            };
            if let Ok(Some(reading)) = reading
                && !reading.has_kv
            {
                return Ok(Ok(()));
            }
        }
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
        if let Ok(Some(reading)) = reading_of(pass)
            && !reading.takes_tokens
        {
            return Ok(Err(format!(
                "reading `{}` embeds no tokens; `embed` is refused on its pass (its rows are \
                 the latents port's)",
                reading.name
            )));
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
        if let Err(error) = self.core_gate(&this, None)? {
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
        if let Ok(Some(reading)) = reading_of(pass)
            && !reading.has_kv
        {
            return Ok(Err(format!(
                "reading `{}` declares no KV space; `attention` is refused on its pass (nothing \
                 there is a sequence)",
                reading.name
            )));
        }
        pass.bindings.attention = Some(binding);
        Ok(Ok(()))
    }

    async fn core_reading(
        &mut self,
        this: Resource<ForwardPass>,
        name: String,
    ) -> Anyhow<Result<(), String>> {
        let model = crate::model::model();
        let Some(reading) = model.readings().iter().find(|reading| reading.name == name) else {
            return Ok(Err(if model.readings().is_empty() {
                format!(
                    "this model declares no readings (one implicit reading); `reading(\"{name}\")` \
                     has nothing to name"
                )
            } else {
                format!(
                    "this model declares no reading `{name}`; it declares {}",
                    reading_names(model.readings())
                )
            }));
        };
        if let Err(error) = self.core_gate(&this, Some(reading))? {
            return Ok(Err(error));
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.reading.is_some() {
            return Ok(Err(
                "forward pass reading is already set; a pass keeps one reading for its life"
                    .to_string(),
            ));
        }
        if !reading.takes_tokens && pass.bindings.embed.is_some() {
            return Ok(Err(format!(
                "reading `{}` embeds no tokens, but this pass already bound `embed`",
                reading.name
            )));
        }
        if !reading.has_kv && pass.bindings.attention.is_some() {
            return Ok(Err(format!(
                "reading `{}` declares no KV space, but this pass already bound `attention`",
                reading.name
            )));
        }
        if let Some(port) = pass
            .bindings
            .ports
            .iter()
            .find(|port| reading.port(&port.name).is_none())
        {
            return Ok(Err(format!(
                "reading `{}` declares no port `{}`, but this pass already bound one",
                reading.name, port.name
            )));
        }
        if let Some(stream) = pass.bindings.stream
            && !reading_lists_stream(reading, stream)
        {
            return Ok(Err(format!(
                "reading `{}` lists no `{}` stream, but this pass already stated it",
                reading.name,
                stream.name()
            )));
        }
        pass.bindings.reading = Some(reading.index);
        Ok(Ok(()))
    }

    async fn core_input(
        &mut self,
        this: Resource<ForwardPass>,
        port: String,
        channel: Resource<Channel>,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        let (shape, dtype, global_id) = {
            let resource = self.ctx().table.get(&channel)?;
            let cell = resource.cell.lock().unwrap();
            (cell.shape.clone(), cell.dtype, cell.global_id)
        };
        let pass = self.ctx().table.get(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        let model = crate::model::model();
        let candidates: Vec<&models::ReadingFact> = match reading_of(pass) {
            Ok(Some(reading)) => vec![reading],
            Ok(None) => Vec::new(),
            Err(_) => model.readings().iter().collect(),
        };
        if candidates.is_empty() {
            return Ok(Err(format!(
                "this model declares no float ports; `input(\"{port}\")` has nothing to bind"
            )));
        }
        let Some((reading, (index, fact))) = candidates
            .iter()
            .find_map(|reading| reading.port(&port).map(|found| (*reading, found)))
        else {
            return Ok(Err(format!(
                "no reading of this model declares a port `{port}`; {}",
                candidates
                    .iter()
                    .map(|reading| format!(
                        "`{}` declares {}",
                        reading.name,
                        if reading.ports.is_empty() {
                            "none".to_string()
                        } else {
                            reading
                                .ports
                                .iter()
                                .map(|p| format!("`{}`", p.name))
                                .collect::<Vec<_>>()
                                .join(", ")
                        }
                    ))
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        };
        let rows = match validate_port_channel(fact, &shape, dtype) {
            Ok(rows) => rows,
            Err(error) => return Ok(Err(error)),
        };
        if let Some(rows) = rows
            && fact.kind == models::PortKind::Latents
            && let Some(generative) = model.generative()
            && rows > generative.max_rows
        {
            return Ok(Err(format!(
                "port `{port}` binds {rows} latent rows; this model carries at most {} \
                 (`model.max-latent-rows()`)",
                generative.max_rows
            )));
        }
        let binding = PortBinding {
            name: port.clone(),
            kind: engine_port_kind(fact.kind),
            port: index,
            channel_rep: channel.rep(),
            channel_id: global_id,
            rows,
            clip: port_clip(fact, &shape),
        };
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.bindings.ports.iter().any(|bound| bound.name == port) {
            return Ok(Err(format!(
                "port `{port}` is already bound on this pass (reading `{}`)",
                reading.name
            )));
        }
        let mut ports = pass.bindings.ports.clone();
        ports.push(binding.clone());
        if let Err(error) = port_rows(&ports) {
            return Ok(Err(error));
        }
        pass.bindings.ports.push(binding);
        Ok(Ok(()))
    }

    async fn core_stream(
        &mut self,
        this: Resource<ForwardPass>,
        stream: pie::inferlet::model::LaneStream,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        let stream = super::model::catalog_stream(stream);
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.stream.is_some() {
            return Ok(Err(
                "forward pass stream is already set; a pass is one lane of one stream".to_string(),
            ));
        }
        if let Ok(Some(reading)) = reading_of(pass)
            && !reading_lists_stream(reading, stream)
        {
            return Ok(Err(format!(
                "reading `{}` lists no `{}` stream; it lists {}",
                reading.name,
                stream.name(),
                stream_names(reading)
            )));
        }
        pass.bindings.stream = Some(stream);
        Ok(Ok(()))
    }

    async fn core_group(
        &mut self,
        this: Resource<ForwardPass>,
        id: u32,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.group.is_some() {
            return Ok(Err("forward pass group is already set".to_string()));
        }
        pass.bindings.group = Some(id);
        Ok(Ok(()))
    }

    async fn core_peer(
        &mut self,
        this: Resource<ForwardPass>,
        group: u32,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.peer.is_some() {
            return Ok(Err("forward pass peer is already set".to_string()));
        }
        let Some(own) = pass.bindings.group else {
            return Ok(Err(
                "a peer is a lane of ANOTHER attention group, and this pass names \
                 no group of its own: call `group` first"
                    .to_string(),
            ));
        };
        if own == group {
            return Ok(Err(format!(
                "this pass names attention group {group} as its own peer group; \
                 guidance combines two INDEPENDENT denoisings, and a lane guided \
                 by itself is not guided"
            )));
        }
        pass.bindings.peer = Some(group);
        Ok(Ok(()))
    }

    async fn core_canvas(
        &mut self,
        this: Resource<ForwardPass>,
        mode: CanvasMode,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.canvas.is_some() {
            return Ok(Err(
                "forward pass canvas mode is already set; a pass keeps one reading for its life"
                    .to_string(),
            ));
        }
        pass.bindings.canvas = Some(mode);
        Ok(Ok(()))
    }

    async fn core_self_conditioning(
        &mut self,
        this: Resource<ForwardPass>,
        rows: Vec<u32>,
        weights: Vec<f32>,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        let Some(shape) = crate::model::model().diffusion() else {
            return Ok(Err(
                "self-conditioning is a diffusion model's input; this model states no canvas"
                    .to_string(),
            ));
        };
        let vocab = crate::model::model().vocab_size();
        let cells = shape.canvas as usize * shape.self_cond_taps as usize;
        if rows.len() != cells || weights.len() != cells {
            return Ok(Err(format!(
                "self-conditioning takes {cells} ids and {cells} weights ({} canvas rows x {} \
                 taps, row major); {} ids and {} weights were staged",
                shape.canvas,
                shape.self_cond_taps,
                rows.len(),
                weights.len()
            )));
        }
        if let Some(bad) = rows.iter().find(|&&id| id >= vocab) {
            return Ok(Err(format!(
                "self-conditioning tap id {bad} is outside the model's {vocab}-wide table"
            )));
        }
        if weights.iter().any(|w| !w.is_finite()) {
            return Ok(Err("self-conditioning weights must be finite".to_string()));
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.bindings.canvas != Some(CanvasMode::Denoise) {
            return Ok(Err(
                "self-conditioning is a denoise pass's input; set `canvas(denoise)` first, and \
                 never stage it on an encode pass"
                    .to_string(),
            ));
        }
        if pass.bindings.self_cond.is_some() {
            return Ok(Err(
                "a self-conditioning payload is already staged for this pass's next submit; \
                 staging another would lose it"
                    .to_string(),
            ));
        }
        pass.bindings.self_cond = Some(crate::pipeline::instance::SelfCondPayload {
            taps: shape.self_cond_taps,
            rows,
            weights,
            channels: None,
        });
        Ok(Ok(()))
    }

    async fn core_self_conditioning_from(
        &mut self,
        this: Resource<ForwardPass>,
        rows: Resource<Channel>,
        weights: Resource<Channel>,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        let Some(shape) = crate::model::model().diffusion() else {
            return Ok(Err(
                "self-conditioning is a diffusion model's input; this model states no canvas"
                    .to_string(),
            ));
        };
        let want = vec![shape.canvas, shape.self_cond_taps];
        let mut ids = [0u64; 2];
        for (slot, (channel, dtype, what)) in [
            (&rows, eta_ir::types::Dtype::U32, "ids"),
            (&weights, eta_ir::types::Dtype::F32, "weights"),
        ]
        .into_iter()
        .enumerate()
        {
            let resource = self.ctx().table.get(channel)?;
            let cell = resource.cell.lock().unwrap();
            if cell.shape != want || cell.dtype != dtype {
                return Ok(Err(format!(
                    "self-conditioning {what} channel must be `[{}, {}]` {dtype:?}; this one is {:?} {:?}",
                    shape.canvas, shape.self_cond_taps, cell.shape, cell.dtype
                )));
            }
            ids[slot] = cell.global_id;
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.bindings.canvas != Some(CanvasMode::Denoise) {
            return Ok(Err(
                "self-conditioning is a denoise pass's input; set `canvas(denoise)` first"
                    .to_string(),
            ));
        }
        if pass.bindings.self_cond.is_some() {
            return Ok(Err(
                "a self-conditioning payload or binding is already on this pass".to_string(),
            ));
        }
        pass.bindings.self_cond = Some(crate::pipeline::instance::SelfCondPayload {
            taps: shape.self_cond_taps,
            rows: Vec::new(),
            weights: Vec::new(),
            channels: Some((ids[0], ids[1])),
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
            lane_facts,
            port_bindings,
            float_rows,
        ) = {
            let port_cells: Vec<(String, Vec<u32>, Dtype)> = {
                let pass = self.ctx().table.get(&this)?;
                pass.bindings
                    .ports
                    .iter()
                    .map(|port| (port.name.clone(), port.channel_rep))
                    .collect::<Vec<_>>()
            }
            .into_iter()
            .map(|(name, rep)| {
                let resource: Resource<Channel> = Resource::new_borrow(rep);
                let cell = self.ctx().table.get(&resource)?.cell.clone();
                let cell = cell.lock().unwrap();
                Ok::<_, anyhow::Error>((name, cell.shape.clone(), cell.dtype))
            })
            .collect::<Result<_, _>>()?;
            let embed_ids: Option<usize> = {
                let rep = self
                    .ctx()
                    .table
                    .get(&this)?
                    .bindings
                    .embed
                    .map(|e| e.tokens);
                match rep {
                    Some(rep) => {
                        let resource: Resource<Channel> = Resource::new_borrow(rep);
                        let cell = self.ctx().table.get(&resource)?.cell.clone();
                        let ids = cell.lock().unwrap().numel();
                        Some(ids)
                    }
                    None => None,
                }
            };
            let pass = self.ctx().table.get(&this)?;
            if pass.is_bound() {
                return Ok(Err("forward pass program is already attached".to_string()));
            }
            let reading = match reading_of(pass) {
                Ok(reading) => reading,
                Err(error) => return Ok(Err(error)),
            };
            let wants_tokens = reading.is_none_or(|reading| reading.takes_tokens);
            let wants_kv = reading.is_none_or(|reading| reading.has_kv);
            let embed = match (pass.bindings.embed, wants_tokens) {
                (Some(embed), true) => Some(embed),
                (None, true) => {
                    return Ok(Err(
                        "forward pass embed binding must be attached before program".to_string(),
                    ));
                }
                (Some(_), false) => {
                    return Ok(Err(format!(
                        "reading `{}` embeds no tokens, but this pass bound `embed`",
                        reading.map_or("", |reading| reading.name)
                    )));
                }
                (None, false) => None,
            };
            let attention = match (pass.bindings.attention, wants_kv) {
                (Some(attention), true) => Some(attention),
                (None, true) => {
                    return Ok(Err(
                        "forward pass attention binding must be attached before program"
                            .to_string(),
                    ));
                }
                (Some(_), false) => {
                    return Ok(Err(format!(
                        "reading `{}` declares no KV space, but this pass bound `attention`",
                        reading.map_or("", |reading| reading.name)
                    )));
                }
                (None, false) => None,
            };
            let pass_stream = pass.bindings.stream.unwrap_or_default();
            let carried = |port: &models::PortFact| {
                port.streams.is_empty() || port.streams.contains(&pass_stream)
            };
            if let Some(reading) = reading {
                if let Some((_, port)) = reading.ports_indexed().find(|(_, port)| {
                    carried(port)
                        && !pass
                            .bindings
                            .ports
                            .iter()
                            .any(|bound| bound.name == port.name)
                }) {
                    return Ok(Err(format!(
                        "reading `{}` declares port `{}` for this pass's stream and this pass \
                         bound no channel to it; call `input(\"{}\", channel)` before `program`",
                        reading.name, port.name, port.name
                    )));
                }
                if let Some(bound) = pass.bindings.ports.iter().find(|bound| {
                    reading
                        .port(&bound.name)
                        .is_none_or(|(_, port)| !carried(port))
                }) {
                    return Ok(Err(format!(
                        "reading `{}` declares no port `{}` for this pass's stream, but this \
                         pass bound one",
                        reading.name, bound.name
                    )));
                }
            } else if let Some(bound) = pass.bindings.ports.first() {
                return Ok(Err(format!(
                    "this model declares no float ports, but this pass bound `{}`",
                    bound.name
                )));
            }
            let stream = pass.bindings.stream.unwrap_or_default();
            if let Some(reading) = reading
                && !reading_lists_stream(reading, stream)
            {
                return Ok(Err(format!(
                    "reading `{}` lists no `{}` stream; it lists {}",
                    reading.name,
                    stream.name(),
                    stream_names(reading)
                )));
            }
            if wants_kv && !wants_tokens {
                return Ok(Err(format!(
                    "reading `{}` embeds no tokens but declares a KV space; a sequence's rows \
                     are its tokens, and this reading states none",
                    reading.map_or("", |reading| reading.name)
                )));
            }
            if pass.kind == PassKind::Diffusion && pass.bindings.canvas.is_none() {
                return Ok(Err(
                    "forward pass canvas mode must be set before program on a diffusion pass"
                        .to_string(),
                ));
            }
            let mut port_bindings = pass.bindings.ports.clone();
            if let Some(reading) = reading {
                for binding in &mut port_bindings {
                    let Some((index, fact)) = reading.port(&binding.name) else {
                        continue;
                    };
                    let Some((_, shape, dtype)) =
                        port_cells.iter().find(|(name, _, _)| *name == binding.name)
                    else {
                        continue;
                    };
                    match validate_port_channel(fact, shape, *dtype) {
                        Ok(rows) => binding.rows = rows,
                        Err(error) => {
                            return Ok(Err(format!("reading `{}`: {error}", reading.name)));
                        }
                    }
                    binding.clip = port_clip(fact, shape);
                    if fact.kind == models::PortKind::Latents
                        && let Some(rows) = binding.rows
                        && let Some(generative) = crate::model::model().generative()
                        && rows > generative.max_rows
                    {
                        return Ok(Err(format!(
                            "port `{}` binds {rows} latent rows; this model carries at most {} \
                             (`model.max-latent-rows()`)",
                            binding.name, generative.max_rows
                        )));
                    }
                    binding.kind = engine_port_kind(fact.kind);
                    binding.port = index;
                }
                if let Err(error) = port_rows(&port_bindings) {
                    return Ok(Err(error));
                }
            }
            let clips: Vec<[u32; 3]> = port_bindings
                .iter()
                .filter(|binding| binding.kind == ::engine::fire::PortKind::Voxels)
                .filter_map(|binding| binding.clip)
                .collect();
            let float_rows = if wants_tokens && wants_kv {
                None
            } else if wants_tokens {
                let Some(ids) = embed_ids else {
                    return Ok(Err(
                        "a cacheless encoder pass binds `embed`; this one bound none".to_string(),
                    ));
                };
                match u32::try_from(ids).ok().filter(|rows| *rows > 0) {
                    Some(rows) => Some(rows),
                    None => {
                        return Ok(Err(format!(
                            "reading `{}` embeds tokens and the channel bound to `embed` holds \
                             {ids} of them",
                            reading.map_or("", |reading| reading.name)
                        )));
                    }
                }
            } else {
                match port_rows(&port_bindings) {
                    Ok(Some(rows)) => Some(rows),
                    Ok(None) if !clips.is_empty() => Some(1),
                    Ok(None) => {
                        return Ok(Err(format!(
                            "reading `{}` embeds no tokens and this pass bound no `[rows, ·]` \
                             port; nothing states its lane's row count",
                            reading.map_or("", |reading| reading.name)
                        )));
                    }
                    Err(error) => return Ok(Err(error)),
                }
            };
            let lane_facts = LaneFacts {
                reading: reading.map_or(0, |reading| reading.index),
                stream: lane_stream_of(stream),
                group: pass.bindings.group,
                peer: pass.bindings.peer,
                ports: port_bindings.iter().map(PortBinding::feed).collect(),
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
                lane_facts,
                port_bindings,
                float_rows,
            )
        };
        {
            let prog = match crate::pipeline::program::register(
                container_bytes,
                &crate::pipeline::program::model_profile(),
            ) {
                Ok(p) => p,
                Err(e) => return Ok(Err(e.to_string())),
            };
            crate::inferlet::process::ensure_bind_admitted(self).await;

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
                (Port::EmbedTokens, embed.map(|embed| embed.tokens)),
                (Port::EmbedIndptr, embed.map(|embed| embed.indptr)),
                (Port::KvLen, attention.map(|attention| attention.kv_len)),
                (Port::Pages, attention.map(|attention| attention.pages)),
                (
                    Port::PageIndptr,
                    attention.map(|attention| attention.page_indptr),
                ),
                (Port::WSlot, attention.map(|attention| attention.w_slot)),
                (Port::WOff, attention.map(|attention| attention.w_off)),
                (
                    Port::Positions,
                    attention.map(|attention| attention.positions),
                ),
                (
                    Port::AttnMask,
                    attention.and_then(|attention| attention.mask),
                ),
                (Port::Readout, readout),
            ];
            if let Err(error) =
                validate_descriptor_bindings(&prog.bound.container, &channel_reps, &expected)
            {
                return Ok(Err(error));
            }
            if let Some(port) = port_bindings
                .iter()
                .find(|port| !channel_reps.contains(&port.channel_rep))
            {
                return Ok(Err(format!(
                    "pipeline: port `{}`'s channel is not bound into this pass's program; \
                     read it in a stage (the SDK's `input` does) so the program declares it",
                    port.name
                )));
            }
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
                    let extern_binding = extern_bindings[i]
                        .as_ref()
                        .map(|(name, dir)| (name.as_str(), *dir));
                    if let Err(e) = c.validate_attachment(&decls[i], extern_binding) {
                        return Ok(Err(format!(
                            "pipeline: channel {i} (id {}): {e}",
                            c.global_id
                        )));
                    }
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

            let (ws_rep, readable, writable, devgeo, decode_envelope, geometry_class) =
                if let Some(attention) = attention {
                    let readable = attention.readable;
                    let writable = attention.writable;
                    let ws_rep = attention.kv_ws;
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
                    let device_port_mask =
                        crate::engine::get_spec(bound_ws.engine)?.device_geometry_port_mask;

                    let needs_mask_port = prog.bound.container.ports.iter().any(|binding| {
                        matches!(binding.port, eta_ir::registry::Port::AttnMask)
                            && matches!(binding.source, eta_ir::container::PortSource::Channel(_))
                    });
                    let devgeo_capable = device_port_mask.covers(PortMask::DEVICE_GEOMETRY)
                        && (!needs_mask_port
                            || device_port_mask.covers(PortMask::of(&[Port::AttnMask])));
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
                            let reserved = crate::store::registry::with_kv_lock(
                                &stores.kv,
                                "host-other",
                                |kv| kv.reserve(bound_ws.id, b as u64),
                            );
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
                                    crate::pipeline::fire::geometry::envelope_required_ports(
                                        &envelope,
                                    );
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
                    if decode_envelope.is_some() && (readable.start != 0 || readable.end.is_some())
                    {
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
                    (
                        ws_rep,
                        readable,
                        writable,
                        devgeo,
                        decode_envelope,
                        geometry_class,
                    )
                } else {
                    let open = crate::pipeline::instance::KvPageSpan {
                        start: 0,
                        end: None,
                    };
                    (
                        self.mint_scratch_working_set()?,
                        open,
                        open,
                        None,
                        None,
                        GeometryClass::Host,
                    )
                };
            let rs_reps: Vec<u32> = rs_working_sets.iter().map(Resource::rep).collect();
            if float_rows.is_some() && !rs_reps.is_empty() {
                self.release_scratch_working_set(ws_rep)?;
                return Ok(Err("a float lane binds no recurrent state".to_string()));
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
                lane: lane_facts,
                float: float_rows.map(|rows| FloatLane {
                    rows,
                    clips: port_bindings
                        .iter()
                        .filter(|binding| binding.kind == ::engine::fire::PortKind::Voxels)
                        .filter_map(|binding| binding.clip)
                        .collect(),
                    embed: embed.is_some(),
                }),
                closed: false,
            };
            if let Err(error) = self.ctx().table.get_mut(&this)?.attach_bound(bound) {
                return Ok(Err(format!("pipeline: {error}")));
            }
            Ok(Ok(()))
        }
    }

    fn mint_scratch_working_set(&mut self) -> Anyhow<u32> {
        let stores = crate::store::registry::get(0, 0);
        let prepared = crate::store::kv::PreparedWorkingSet::new();
        let id = crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", move |kv| {
            kv.install_working_set(prepared)
        });
        let ws = KvWorkingSet::new(0, 0, id, stores.kv_page_size);
        self.register_kv_working_set(&ws);
        Ok(self.ctx().table.push(ws)?.rep())
    }

    fn release_scratch_working_set(&mut self, rep: u32) -> Anyhow<()> {
        let resource: Resource<KvWorkingSet> = Resource::new_own(rep);
        let ws = self.ctx().table.delete(resource)?;
        self.unregister_kv_working_set(ws.model, ws.engine, ws.id);
        ws.release();
        Ok(())
    }

    async fn core_set_rs_working_sets(
        &mut self,
        this: Resource<ForwardPass>,
        rs_working_sets: Vec<Resource<RsWorkingSet>>,
        geometry: RsGeometryBinding,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        if rs_working_sets.is_empty() {
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
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

        let mut pass = self.ctx().table.delete(this)?;
        let scratch = pass
            .bound()
            .ok()
            .filter(|bound| bound.float.is_some())
            .map(|bound| bound.kv_ws);
        if let Ok(bound) = pass.bound_mut() {
            bound.close_native();
        }
        drop(pass);
        if let Some(rep) = scratch {
            self.release_scratch_working_set(rep)?;
        }
        Ok(())
    }
}

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

macro_rules! forward_pass_readings {
    () => {
        async fn reading(
            &mut self,
            this: Resource<ForwardPass>,
            name: String,
        ) -> Anyhow<Result<(), String>> {
            self.core_reading(this, name).await
        }

        async fn input(
            &mut self,
            this: Resource<ForwardPass>,
            port: String,
            ch: Resource<Channel>,
        ) -> Anyhow<Result<(), String>> {
            self.core_input(this, port, ch).await
        }

        async fn stream(
            &mut self,
            this: Resource<ForwardPass>,
            s: pie::inferlet::model::LaneStream,
        ) -> Anyhow<Result<(), String>> {
            self.core_stream(this, s).await
        }

        async fn group(
            &mut self,
            this: Resource<ForwardPass>,
            id: u32,
        ) -> Anyhow<Result<(), String>> {
            self.core_group(this, id).await
        }

        async fn peer(
            &mut self,
            this: Resource<ForwardPass>,
            group: u32,
        ) -> Anyhow<Result<(), String>> {
            self.core_peer(this, group).await
        }
    };
}

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
    forward_pass_readings!();

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

impl pie::inferlet::forward_diffusion::Host for ProcessCtx {
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

impl pie::inferlet::forward_diffusion::HostForwardPass for ProcessCtx {
    forward_pass_common!(forward_diffusion, PassKind::Diffusion);
    forward_pass_readings!();

    async fn media(
        &mut self,
        this: Resource<ForwardPass>,
        spans: Vec<pie::inferlet::forward_diffusion::MediaSpan>,
    ) -> Anyhow<Result<(), String>> {
        self.core_media(this, spans).await
    }

    async fn attention(
        &mut self,
        this: Resource<ForwardPass>,
        kv: Resource<crate::store::kv::working_set::KvWorkingSet>,
        geom: pie::inferlet::forward_diffusion::KvGeometry,
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

    async fn self_conditioning(
        &mut self,
        this: Resource<ForwardPass>,
        rows: Vec<u32>,
        weights: Vec<f32>,
    ) -> Anyhow<Result<(), String>> {
        self.core_self_conditioning(this, rows, weights).await
    }

    async fn self_conditioning_from(
        &mut self,
        this: Resource<ForwardPass>,
        rows: Resource<Channel>,
        weights: Resource<Channel>,
    ) -> Anyhow<Result<(), String>> {
        self.core_self_conditioning_from(this, rows, weights).await
    }

    async fn canvas(
        &mut self,
        this: Resource<ForwardPass>,
        mode: pie::inferlet::forward_diffusion::Mode,
    ) -> Anyhow<Result<(), String>> {
        use pie::inferlet::forward_diffusion::Mode;
        let mode = match mode {
            Mode::Encode => CanvasMode::Encode,
            Mode::Denoise => CanvasMode::Denoise,
        };
        self.core_canvas(this, mode).await
    }
}

#[cfg(test)]
mod tests {
    use super::{attention_rebind_diff, port_rows, validate_port_channel};
    use crate::pipeline::instance::{AttentionBinding, KvPageSpan, PortBinding};
    use eta_ir::types::Dtype;

    fn port(name: &'static str, kind: models::PortKind, width: u32) -> models::PortFact {
        models::PortFact {
            name,
            kind,
            width,
            streams: Vec::new(),
            at: None,
            rows: None,
        }
    }

    fn bound(name: &str, kind: ::engine::fire::PortKind, rows: Option<u32>) -> PortBinding {
        PortBinding {
            name: name.to_string(),
            kind,
            port: 0,
            channel_rep: 1,
            channel_id: 1,
            rows,
            clip: None,
        }
    }

    #[test]
    fn forward_every_case() {
        a_port_channel_is_validated_against_its_fact();
        a_pass_s_row_ports_agree_on_their_rows();
        identical_rebind_is_a_no_op();
        differing_rebind_names_the_field();
    }

    fn a_port_channel_is_validated_against_its_fact() {
        let latents = port("latents", models::PortKind::Latents, 64);
        assert_eq!(
            validate_port_channel(&latents, &[256, 64], Dtype::F32),
            Ok(Some(256))
        );
        assert!(
            validate_port_channel(&latents, &[256, 32], Dtype::F32)
                .unwrap_err()
                .contains("`latents`")
        );
        assert!(validate_port_channel(&latents, &[256 * 64], Dtype::F32).is_err());
        assert!(validate_port_channel(&latents, &[0, 64], Dtype::F32).is_err());
        assert!(
            validate_port_channel(&latents, &[256, 64], Dtype::I32)
                .unwrap_err()
                .contains("f32")
        );

        let timestep = port("timestep", models::PortKind::LaneVector, 1);
        assert_eq!(validate_port_channel(&timestep, &[1], Dtype::F32), Ok(None));
        assert_eq!(
            validate_port_channel(&timestep, &[1, 1], Dtype::F32),
            Ok(None)
        );
        assert!(validate_port_channel(&timestep, &[2], Dtype::F32).is_err());

        let positions = port("positions", models::PortKind::AxisPositions, 3);
        assert_eq!(
            validate_port_channel(&positions, &[256, 3], Dtype::F32),
            Ok(Some(256))
        );
    }

    fn a_pass_s_row_ports_agree_on_their_rows() {
        use ::engine::fire::PortKind;
        let agree = [
            bound("latents", PortKind::Latents, Some(256)),
            bound("positions", PortKind::AxisPositions, Some(256)),
            bound("timestep", PortKind::LaneVector, None),
            bound("context", PortKind::Context, Some(77)),
        ];
        assert_eq!(port_rows(&agree), Ok(Some(256)));
        let disagree = [
            bound("latents", PortKind::Latents, Some(256)),
            bound("positions", PortKind::AxisPositions, Some(128)),
        ];
        assert!(port_rows(&disagree).unwrap_err().contains("`positions`"));
        assert_eq!(
            port_rows(&[bound("timestep", PortKind::LaneVector, None)]),
            Ok(None)
        );
    }

    fn binding() -> AttentionBinding {
        AttentionBinding {
            kv_ws: 1,
            readable: KvPageSpan {
                start: 0,
                end: None,
            },
            writable: KvPageSpan {
                start: 0,
                end: None,
            },
            kv_len: 2,
            pages: 3,
            page_indptr: 4,
            w_slot: 5,
            w_off: 6,
            positions: 7,
            mask: Some(8),
        }
    }

    fn identical_rebind_is_a_no_op() {
        assert_eq!(attention_rebind_diff(&binding(), &binding()), None);
    }

    fn differing_rebind_names_the_field() {
        let mut next = binding();
        next.kv_ws = 9;
        assert_eq!(
            attention_rebind_diff(&binding(), &next),
            Some("kv-working-set")
        );
        let mut next = binding();
        next.writable = KvPageSpan {
            start: 0,
            end: Some(4),
        };
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
