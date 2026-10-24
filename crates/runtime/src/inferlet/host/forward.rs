use std::sync::{Arc, Mutex};

use wasmtime::component::{Accessor, HasSelf, Resource};
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
pub use crate::pipeline::channel::Channel;
use crate::pipeline::channel::{BoundCells, ChannelCell, ChannelError};
pub use crate::pipeline::instance::ForwardPass;
use crate::pipeline::instance::Instance;
use crate::pipeline::instance::{
    BoundForwardPass, CanvasMode, EmbedBinding, FloatLane, LaneFacts, PassKind, PortBinding,
    StateBinding, lane_stream_of,
};
use crate::store::kv::working_set::KvWorkingSet;

use super::working_set::WorkingSet;

use eta_ir::container::{HostRole, PortSource, TraceContainer};
use eta_ir::registry::{GeometryClass, Port, PortMask};
use eta_ir::types::Dtype;

use super::pie;

type Anyhow<T> = anyhow::Result<T>;

fn reading_of(
    pass: &ForwardPass,
) -> Result<Option<&'static poem::generative::ReadingFact>, String> {
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

fn reading_names(readings: &[poem::generative::ReadingFact]) -> String {
    readings
        .iter()
        .map(|reading| format!("`{}`", reading.name))
        .collect::<Vec<_>>()
        .join(", ")
}

fn reading_lists_stream(reading: &poem::generative::ReadingFact, stream: poem::Stream) -> bool {
    if reading.streams.is_empty() {
        return stream == poem::Stream::Text;
    }
    reading.streams.contains(&stream)
}

fn stream_names(reading: &poem::generative::ReadingFact) -> String {
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

fn engine_port_kind(kind: poem::generative::PortKind) -> ::engine::fire::PortKind {
    use ::engine::fire::PortKind;
    match kind {
        poem::generative::PortKind::Latents => PortKind::Latents,
        poem::generative::PortKind::LaneVector => PortKind::LaneVector,
        poem::generative::PortKind::Context => PortKind::Context,
        poem::generative::PortKind::AxisPositions => PortKind::AxisPositions,
        poem::generative::PortKind::Voxels => PortKind::Voxels,
    }
}

pub(crate) fn validate_port_channel(
    port: &poem::generative::PortFact,
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
        poem::generative::PortKind::LaneVector => match shape {
            [width] | [1, width] if *width == port.width => Ok(None),
            _ => Err(format!(
                "port `{}` is a lane vector of width {}: its channel must be `[{}]` or `[1, {}]` \
                 f32; this one is {shape:?}",
                port.name, port.width, port.width, port.width
            )),
        },
        poem::generative::PortKind::Latents
        | poem::generative::PortKind::Context
        | poem::generative::PortKind::AxisPositions => match shape {
            [rows, width] if *width == port.width && *rows > 0 => Ok(Some(*rows)),
            _ => Err(format!(
                "port `{}` reads `[rows, {}]` f32; this channel is {shape:?}",
                port.name, port.width
            )),
        },
        poem::generative::PortKind::Voxels => match shape {
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

pub(crate) fn port_clip(port: &poem::generative::PortFact, shape: &[u32]) -> Option<[u32; 3]> {
    if port.kind != poem::generative::PortKind::Voxels {
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

/// After attach a pass may bind other working sets, never other channels.
fn state_rebind_diff(existing: &StateBinding, next: &StateBinding) -> Option<&'static str> {
    if existing.lanes.len() != next.lanes.len() {
        return Some("lanes (a different count)");
    }
    if existing.positions != next.positions {
        return Some("positions");
    }
    if existing.slots != next.slots {
        return Some("slots");
    }
    if existing.kv_len != next.kv_len {
        return Some("kv-len");
    }
    if existing.mask != next.mask {
        return Some("mask");
    }
    if existing.fold != next.fold {
        return Some("fold");
    }
    None
}

fn first_repeat(reps: &[u32]) -> Option<(usize, usize)> {
    reps.iter().enumerate().find_map(|(again, rep)| {
        reps[..again]
            .iter()
            .position(|earlier| earlier == rep)
            .map(|first| (first, again))
    })
}

fn validate_descriptor_bindings(
    container: &TraceContainer,
    channel_reps: &[u32],
    expected: &[(Port, Option<u32>)],
) -> Result<(), String> {
    for &(port, expected_rep) in expected {
        let binding = container.ports.iter().find(|binding| binding.port == port);
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
    AwaitCompletion(crate::pipeline::fire::PendingFires),
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
        let (ready, unread) = {
            let mut cell = cell.lock().unwrap();
            let ready = cell.read();
            (ready, cell.unread_len())
        };
        match ready {
            Ok(_) => {
                if pop_settled && let Some(op) = crate::pipeline::fire::pop_settled(fires.as_ref())
                {
                    return Ok(ChannelPoll::Finalize(op));
                }
                let queue = fires
                    .as_ref()
                    .expect("the ready take has a pending fire queue");
                // A visible channel may precede fire completion. Wait on that
                // completion, not the already-ready channel, before taking it,
                // unless the front cell cannot be a pending fire's: then its
                // fire is finalized already, by an earlier take or by a
                // submit's drain, and holding it for the next completion would
                // leave the lane that many tokens behind for good.
                if unread <= queue.writers_of(&cell) {
                    return Ok(ChannelPoll::AwaitCompletion(Arc::clone(queue)));
                }
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
            }
            | ChannelPoll::AwaitCompletion(fires) => {
                let _finalize_guard = fires.finalize_guard().await;
                let state = accessor.with(|mut access| {
                    poll_channel(access.get(), &this, mode, true, settle_ready_take)
                })?;
                match state {
                    ChannelPoll::AwaitCompletion(fires) => {
                        fires.await_front_completion().await;
                        continue;
                    }
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
            ChannelPoll::AwaitCompletion(_) => unreachable!("completion wait holds finalizer gate"),
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
            }
            | ChannelPoll::AwaitCompletion(fires) => {
                let _finalize_guard = fires.finalize_guard().await;
                let state = poll_channel(ctx, &this, mode, true, settle_ready_take)?;
                match state {
                    ChannelPoll::AwaitCompletion(fires) => {
                        fires.await_front_completion().await;
                        continue;
                    }
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
            ChannelPoll::AwaitCompletion(_) => unreachable!("completion wait holds finalizer gate"),
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
        named: Option<&'static poem::generative::ReadingFact>,
    ) -> Anyhow<Result<(), String>> {
        let kind = self.ctx().table.get(this)?.kind;
        let actual = crate::model::model().pass_kind();
        if kind == PassKind::Text {
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
        if kind != actual {
            return Ok(Err(format!(
                "this model's forward pass is `{}`, but the pass was built through the `{}` \
                 interface; use `{}` instead",
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

    async fn core_state(
        &mut self,
        this: Resource<ForwardPass>,
        lanes: Vec<Resource<WorkingSet>>,
        geom: pie::inferlet::forward::StateGeometry,
    ) -> Anyhow<Result<(), String>> {
        let pie::inferlet::forward::StateGeometry {
            positions,
            slots,
            kv_len,
            mask,
            fold,
        } = geom;
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        if lanes.is_empty() {
            return Ok(Err(
                "forward pass state binding names no working set; give one per lane".to_string(),
            ));
        }
        for lane in &lanes {
            let _ = self.ctx().table.get(lane)?;
        }
        for channel in [&positions, &slots, &kv_len, &mask, &fold]
            .into_iter()
            .flatten()
        {
            let _ = self.ctx().table.get(channel)?;
        }
        let reps: Vec<u32> = lanes.iter().map(Resource::rep).collect();
        if crate::model::model().rs_caps().window_tokens > 0
            && let Some((first, again)) = first_repeat(&reps)
        {
            return Ok(Err(format!(
                "lanes {first} and {again} name the same working set; on a model whose \
                 state is a window each lane needs a working set of its own"
            )));
        }
        let binding = StateBinding {
            lanes: reps,
            positions: positions.map(|resource| resource.rep()),
            slots: slots.map(|resource| resource.rep()),
            kv_len: kv_len.map(|resource| resource.rep()),
            mask: mask.map(|resource| resource.rep()),
            fold: fold.map(|resource| resource.rep()),
        };
        let pass = self.ctx().table.get(&this)?;
        if pass.is_bound() {
            let Some(existing) = pass.bindings.state.clone() else {
                return Ok(Err("forward pass program is already attached".to_string()));
            };
            if let Some(field) = state_rebind_diff(&existing, &binding) {
                return Ok(Err(format!(
                    "forward pass state binding cannot change its geometry after the program \
                     is attached (`{field}` differs); bind the same channels, or build a new \
                     pass"
                )));
            }
            let (kv_lanes, rs_ws) = self.lane_parts(&binding.lanes)?;
            let pass = self.ctx().table.get_mut(&this)?;
            let bound = match pass.bound_mut() {
                Ok(bound) => bound,
                Err(error) => return Ok(Err(error)),
            };
            if let Err(error) = bound.replace_lanes(binding.lanes.clone(), kv_lanes, rs_ws) {
                return Ok(Err(format!("pipeline: {error}")));
            }
            pass.bindings.state = Some(binding);
            return Ok(Ok(()));
        }
        if pass.bindings.state.is_some() {
            return Ok(Err(
                "forward pass state binding is already attached".to_string()
            ));
        }
        if let Ok(Some(reading)) = reading_of(pass)
            && !reading.has_kv
        {
            return Ok(Err(format!(
                "reading `{}` declares no KV space; `state` is refused on its pass (nothing \
                 there is a sequence)",
                reading.name
            )));
        }
        self.ctx().table.get_mut(&this)?.bindings.state = Some(binding);
        Ok(Ok(()))
    }

    fn lane_parts(&mut self, lanes: &[u32]) -> Anyhow<(Vec<u32>, Vec<u32>)> {
        let mut kv = Vec::with_capacity(lanes.len());
        let mut rs = Vec::new();
        for &rep in lanes {
            let res: Resource<WorkingSet> = Resource::new_borrow(rep);
            let ws = self.ctx().table.get(&res)?;
            kv.push(ws.kv);
            if let Some(state) = ws.rs {
                rs.push(state);
            }
        }
        Ok((kv, rs))
    }

    async fn core_checkpoints(
        &mut self,
        this: Resource<ForwardPass>,
        mut slots: Vec<u32>,
    ) -> Anyhow<Result<(), String>> {
        slots.sort_unstable();
        slots.dedup();
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        pass.bindings.checkpoints = slots;
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
        if !reading.has_kv && pass.bindings.state.is_some() {
            return Ok(Err(format!(
                "reading `{}` declares no KV space, but this pass already bound `state`",
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
        let candidates: Vec<&poem::generative::ReadingFact> = match reading_of(pass) {
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
            && fact.kind == poem::generative::PortKind::Latents
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

    async fn core_attention_classes(
        &mut self,
        this: Resource<ForwardPass>,
        classes: Vec<i32>,
        table: Vec<u8>,
        count: u32,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = self.core_gate(&this, None)? {
            return Ok(Err(error));
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.attn_classes.is_some() {
            return Ok(Err(
                "forward pass attention classes are already set".to_string()
            ));
        }
        let max = ::engine::fire::ATTN_CLASSES_MAX;
        if count == 0 || count > max {
            return Ok(Err(format!(
                "attention classes: `count` is {count}; the table holds 1..={max} classes"
            )));
        }
        if table.len() as u64 != u64::from(count) * u64::from(count) {
            return Ok(Err(format!(
                "attention classes: the table is {} bytes for {count} classes; it is \
                 count x count, row-major, q class by kv class",
                table.len()
            )));
        }
        if let Some(bad) = classes.iter().find(|c| **c >= 0 && **c as u32 >= count) {
            return Ok(Err(format!(
                "attention classes: a row names class {bad} and the table holds {count}; \
                 a class is in 0..count, or -1 for an unconstrained row"
            )));
        }
        pass.bindings.attn_classes = Some(::engine::fire::AttnClasses {
            classes,
            table,
            count,
        });
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
        let mut ids = [::engine::ChannelId::default(); 2];
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
        let (embed, state, readout, pass_block_draft, lane_facts, port_bindings, float_rows) = {
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
                        reading.map_or("", |reading| reading.name.as_str())
                    )));
                }
                (None, false) => None,
            };
            let state = match (pass.bindings.state.clone(), wants_kv) {
                (Some(state), true) => Some(state),
                (None, true) => {
                    return Ok(Err(
                        "forward pass state binding must be attached before program".to_string(),
                    ));
                }
                (Some(_), false) => {
                    return Ok(Err(format!(
                        "reading `{}` declares no KV space, but this pass bound `state`",
                        reading.map_or("", |reading| reading.name.as_str())
                    )));
                }
                (None, false) => None,
            };
            let pass_stream = pass.bindings.stream.unwrap_or_default();
            let carried = |port: &poem::generative::PortFact| {
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
                    reading.map_or("", |reading| reading.name.as_str())
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
                    if fact.kind == poem::generative::PortKind::Latents
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
                            reading.map_or("", |reading| reading.name.as_str())
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
                            reading.map_or("", |reading| reading.name.as_str())
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
                attn_classes: pass.bindings.attn_classes.clone(),
            };
            (
                embed,
                state,
                pass.bindings.readout,
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
            let geometry = |pick: fn(&StateBinding) -> Option<u32>| state.as_ref().and_then(pick);
            let expected = [
                (Port::EmbedTokens, embed.map(|embed| embed.tokens)),
                (Port::EmbedIndptr, embed.map(|embed| embed.indptr)),
                (Port::Positions, geometry(|state| state.positions)),
                (Port::Slots, geometry(|state| state.slots)),
                (Port::KvLen, geometry(|state| state.kv_len)),
                (Port::AttnMask, geometry(|state| state.mask)),
                (Port::Fold, geometry(|state| state.fold)),
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
                     read it in a stage (the inferlet library's `input` does) so the program declares it",
                    port.name
                )));
            }
            let writes = |dense: usize| prog.channel_accesses[dense] != (false, false);
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
                    if let Err(e) = c.validate_attachment(&decls[i], writes(i), extern_binding) {
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

            let lanes: Vec<u32> = state
                .as_ref()
                .map(|state| state.lanes.clone())
                .unwrap_or_default();
            let (kv_lanes, rs_reps) = if state.is_some() {
                self.lane_parts(&lanes)?
            } else {
                (vec![self.mint_scratch_working_set()?], Vec::new())
            };
            let (devgeo, decode_envelope, geometry_class) = if state.is_some() {
                let engine = super::working_set::kv_of(self, kv_lanes[0])?.engine;
                let device_port_mask = crate::engine::get_spec(engine)?.device_geometry_port_mask;
                let needs_mask_port = prog.bound.container.ports.iter().any(|binding| {
                    matches!(binding.port, eta_ir::registry::Port::AttnMask)
                        && matches!(binding.source, eta_ir::container::PortSource::Channel(_))
                });
                let devgeo_capable = device_port_mask.covers(PortMask::DEVICE_GEOMETRY)
                    && (!needs_mask_port
                        || device_port_mask.covers(PortMask::of(&[Port::AttnMask])));
                let taint = prog.geometry_taint();
                let devgeo = if !taint.host_derivable() {
                    match crate::pipeline::fire::geometry::detect_pooled_device_geometry(
                        &prog.bound.container,
                    ) {
                        Some(_) if !devgeo_capable => {
                            tracing::info!(
                                "device-geometry program on an engine without device geometry \
                                 ports (mask {device_port_mask:?}): falling back to \
                                 host-evaluated serialized execution"
                            );
                            None
                        }
                        // A program the decode envelope covers runs ahead: its
                        // pages and write slots are the host's, so the next fire
                        // is staged while this one runs. Device geometry reads
                        // every descriptor port back from the device before it
                        // stages, so it is only the pass for programs the
                        // envelope declines.
                        Some(_)
                            if envelope_runs_ahead(
                                &prog.bound.container,
                                taint,
                                device_port_mask,
                            ) =>
                        {
                            None
                        }
                        Some(qo_indptr) => {
                            tracing::info!(
                                "device-carried decode ({} lane(s), {} row(s)) re-publishes \
                                 where its rows land: executes as a device-geometry pass",
                                qo_indptr.len().saturating_sub(1),
                                qo_indptr.last().copied().unwrap_or(0)
                            );
                            Some(qo_indptr)
                        }
                        None => None,
                    }
                } else {
                    None
                };
                let decode_envelope = if devgeo.is_some() || taint.host_derivable() {
                    None
                } else {
                    let mut why = String::new();
                    match crate::pipeline::fire::geometry::classify_decode_envelope_why(
                        &prog.bound.container,
                        taint,
                        &mut why,
                    ) {
                        Ok(Some(required)) => {
                            if device_port_mask.covers(required) {
                                Some(required)
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
                                "not a decode envelope: {why}; falling back to host-evaluated \
                                 execution"
                            );
                            None
                        }
                        Err(reason) => {
                            tracing::warn!(
                                "device-dependent geometry is not a decode envelope \
                                 ({reason}); falling back to host-evaluated execution — fires \
                                 block loudly on values the host cannot derive"
                            );
                            None
                        }
                    }
                };
                let geometry_class = if devgeo.is_some() {
                    GeometryClass::DeviceGeometry
                } else if decode_envelope.is_some() {
                    GeometryClass::DecodeEnvelope
                } else {
                    GeometryClass::Host
                };
                (devgeo, decode_envelope, geometry_class)
            } else {
                (None, None, GeometryClass::Host)
            };

            let instance_id = crate::pipeline::instance::next_instance_id();
            for (dense, cell) in cells.iter().enumerate() {
                let extern_binding = extern_bindings[dense]
                    .as_ref()
                    .map(|(name, dir)| (name.as_str(), *dir));
                if let Err(error) = cell.lock().unwrap().attach(
                    instance_id,
                    &decls[dense],
                    writes(dense),
                    extern_binding,
                ) {
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
            let channel_ids: Vec<::engine::ChannelId> =
                cells.iter().map(|c| c.lock().unwrap().global_id).collect();
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
                lanes,
                kv_lanes,
                block_draft: pass_block_draft,
                rs_ws: rs_reps,
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
        let id = crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            kv.create_working_set()
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
            .map(|bound| bound.kv_lanes[0]);
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
    ($kind:expr) => {
        async fn new(&mut self) -> Anyhow<Resource<ForwardPass>> {
            self.core_new($kind).await
        }

        async fn state(
            &mut self,
            this: Resource<ForwardPass>,
            lanes: Vec<Resource<WorkingSet>>,
            geom: pie::inferlet::forward::StateGeometry,
        ) -> Anyhow<Result<(), String>> {
            self.core_state(this, lanes, geom).await
        }

        async fn media(
            &mut self,
            this: Resource<ForwardPass>,
            spans: Vec<pie::inferlet::forward::MediaSpan>,
        ) -> Anyhow<Result<(), String>> {
            self.core_media(this, spans).await
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

        async fn attention_classes(
            &mut self,
            this: Resource<ForwardPass>,
            classes: Vec<i32>,
            table: Vec<u8>,
            count: u32,
        ) -> Anyhow<Result<(), String>> {
            self.core_attention_classes(this, classes, table, count)
                .await
        }
    };
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
    forward_pass_common!(PassKind::Text);

    async fn checkpoints(
        &mut self,
        this: Resource<ForwardPass>,
        slots: Vec<u32>,
    ) -> Anyhow<Result<(), String>> {
        self.core_checkpoints(this, slots).await
    }

    async fn set_drafting_block(
        &mut self,
        this: Resource<ForwardPass>,
        on: bool,
    ) -> Anyhow<Result<(), String>> {
        self.core_set_drafting_block(this, on).await
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
}

impl pie::inferlet::forward_diffusion::HostForwardPass for ProcessCtx {
    forward_pass_common!(PassKind::Diffusion);

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

/// Whether the decode envelope classifies `container` and the engine reads
/// every port that envelope requires from the device.
fn envelope_runs_ahead(
    container: &eta_ir::container::TraceContainer,
    taint: &eta_compiler::eval::pareval::GeometryTaint,
    device_port_mask: PortMask,
) -> bool {
    let mut why = String::new();
    match crate::pipeline::fire::geometry::classify_decode_envelope_why(container, taint, &mut why)
    {
        Ok(Some(required)) => device_port_mask.covers(required),
        Ok(None) | Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{port_rows, state_rebind_diff, validate_port_channel};
    use crate::pipeline::instance::{PortBinding, StateBinding};
    use eta_ir::types::Dtype;

    fn port(
        name: &'static str,
        kind: poem::generative::PortKind,
        width: u32,
    ) -> poem::generative::PortFact {
        poem::generative::PortFact {
            name: name.to_string(),
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
            channel_id: ::engine::ChannelId(1),
            rows,
            clip: None,
        }
    }

    #[test]
    fn forward_every_case() {
        a_port_channel_is_validated_against_its_fact();
        a_pass_s_row_ports_agree_on_their_rows();
        a_rebind_changes_working_sets_not_geometry();
    }

    fn a_port_channel_is_validated_against_its_fact() {
        let latents = port("latents", poem::generative::PortKind::Latents, 64);
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

        let timestep = port("timestep", poem::generative::PortKind::LaneVector, 1);
        assert_eq!(validate_port_channel(&timestep, &[1], Dtype::F32), Ok(None));
        assert_eq!(
            validate_port_channel(&timestep, &[1, 1], Dtype::F32),
            Ok(None)
        );
        assert!(validate_port_channel(&timestep, &[2], Dtype::F32).is_err());

        let positions = port("positions", poem::generative::PortKind::AxisPositions, 3);
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

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_ready_take_keeps_pending_kv_for_teardown() -> anyhow::Result<()> {
        use super::*;
        use crate::pipeline::fire::lifecycle_tests::{
            pending_fire_fixture, resolve_test_completion,
        };
        use std::future::Future;
        use std::sync::atomic::{AtomicU64, Ordering};

        for failed in [false, true] {
            // The channel is visible before the native completion, which is
            // the ready-Take path that used to remove an unresolved fire.
            let mirror = Box::new([7i32, 0]);
            let words = Box::new([
                AtomicU64::new(0),
                AtomicU64::new(1),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ]);
            let mut cell = ChannelCell::new(vec![1], Dtype::I32, 1);
            cell.role = Some(HostRole::Reader);
            cell.attach_reader_mirror(
                0,
                mirror.as_ptr() as u64,
                words.as_ptr() as u64,
                4,
                1,
                0,
                0,
                1,
                2,
                3,
            )
            .map_err(anyhow::Error::msg)?;
            let cell = Arc::new(Mutex::new(cell));
            assert_eq!(
                cell.lock().expect("test channel").read(),
                Ok(7i32.to_ne_bytes().to_vec())
            );
            let (fires, completion, ws, stores, failure) = pending_fire_fixture(vec![cell.clone()]);
            let mut context = ProcessCtx::new(
                uuid::Uuid::new_v4(),
                "test".into(),
                crate::inferlet::process::OutputMode::Discard,
                &crate::inferlet::sandbox::InstancePolicy::deny_all(),
                None,
            )
            .await?;
            context.ctx().table.push(ws)?;
            let channel = context.ctx().table.push(Channel {
                cell: cell.clone(),
                fires: Some(fires.clone()),
            })?;
            {
                let mut take = Box::pin(materialize_channel_blocking(
                    &mut context,
                    channel,
                    ChannelReadMode::Take,
                ));
                let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                assert!(take.as_mut().poll(&mut task).is_pending());
            }
            assert_eq!(
                fires.lock().expect("test FIFO").len(),
                1,
                "cancelled ready Take must not remove an unresolved fire"
            );
            assert_eq!(
                words[0].load(Ordering::Acquire),
                0,
                "the visible channel must not be consumed before finalization"
            );
            *context.ctx().table = wasmtime::component::ResourceTable::new();
            assert_eq!(stores.kv.lock().available_pages(), 0);
            resolve_test_completion(&completion, failed);
            crate::pipeline::fire::finalize_all(&mut context, &fires, true).await?;
            assert!(fires.lock().expect("test FIFO").is_empty());
            assert_eq!(
                stores.kv.lock().available_pages(),
                1,
                "the original submitted page must be reclaimed after teardown"
            );
            assert_eq!(failure.lock().expect("test failure").is_some(), failed);
            let value = cell.lock().expect("test channel").read();
            if failed {
                assert!(
                    matches!(value, Err(ChannelError::Poisoned(_))),
                    "a failed callback must still poison its reader"
                );
            } else {
                assert_eq!(value, Ok(7i32.to_ne_bytes().to_vec()));
            }
        }
        Ok(())
    }

    fn a_rebind_changes_working_sets_not_geometry() {
        let bound = StateBinding {
            lanes: vec![1, 2],
            positions: Some(3),
            slots: None,
            kv_len: Some(4),
            mask: Some(5),
            fold: Some(6),
        };
        let forks = StateBinding {
            lanes: vec![7, 8],
            ..bound.clone()
        };
        assert_eq!(state_rebind_diff(&bound, &forks), None);
        let moved = StateBinding {
            positions: Some(9),
            ..forks
        };
        assert_eq!(state_rebind_diff(&bound, &moved), Some("positions"));
    }
}
