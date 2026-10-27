use std::sync::{Arc, Mutex};

use wasmtime::component::{Accessor, HasSelf, Resource};
use wasmtime_wasi::WasiView;

use crate::inferlet::ProcessCtx;
pub use crate::pipeline::channel::Channel;
use crate::pipeline::channel::{BoundCells, ChannelCell, ChannelError};
pub use crate::pipeline::instance::ForwardPass;
use crate::pipeline::instance::Instance;
use crate::pipeline::instance::{
    BoundForwardPass, EmbedBinding, FloatLane, FloatPass, InputBinding, LaneFacts, StateBinding,
};

use super::working_set::WorkingSet;

use eta_ir::container::{HostRole, PortSource, TraceContainer};
use eta_ir::registry::{GeometryClass, Port, PortMask};
use eta_ir::types::Dtype;

use super::pie;

type Anyhow<T> = anyhow::Result<T>;

/// The rows or clip a channel bound to `input` carries, once its shape and dtype match it.
pub(crate) fn validate_input_channel(
    input: &poem_ir::InputDecl,
    shape: &[u32],
    dtype: Dtype,
) -> Result<(Option<u32>, Option<[u32; 3]>), String> {
    let name = &input.name;
    let want = match input.dtype {
        poem_ir::Dtype::I32 => Dtype::I32,
        _ => Dtype::F32,
    };
    if dtype != want {
        return Err(format!(
            "input `{name}` reads a {want:?} channel; this one is {dtype:?}"
        ));
    }
    let width = input.width;
    match (input.axis, shape) {
        (poem_ir::InputAxis::Lane, [w]) if *w == width => Ok((None, None)),
        (poem_ir::InputAxis::Lane, _) => Err(format!(
            "input `{name}` is one `[{width}]` row a lane; this channel is {shape:?}"
        )),
        (poem_ir::InputAxis::Row, [rows, w]) if *w == width && *rows > 0 => Ok((Some(*rows), None)),
        (poem_ir::InputAxis::Row, _) => Err(format!(
            "input `{name}` reads `[rows, {width}]`; this channel is {shape:?}"
        )),
        (poem_ir::InputAxis::Voxel, [h, w, c]) if *c == width && *h > 0 && *w > 0 => {
            Ok((None, Some([1, *h, *w])))
        }
        (poem_ir::InputAxis::Voxel, [t, h, w, c]) if *c == width && *t > 0 && *h > 0 && *w > 0 => {
            Ok((None, Some([*t, *h, *w])))
        }
        (poem_ir::InputAxis::Voxel, _) => Err(format!(
            "input `{name}` reads one clip `[h, w, {width}]` (or `[t, h, w, {width}]`) on the \
             voxel axis; this channel is {shape:?}"
        )),
    }
}

pub(crate) fn input_rows(inputs: &[InputBinding]) -> Result<Option<u32>, String> {
    let mut rows: Option<(u32, &str)> = None;
    for input in inputs {
        let Some(these) = input.rows else { continue };
        match rows {
            None => rows = Some((these, &input.name)),
            Some((agreed, name)) if agreed != these => {
                return Err(format!(
                    "input `{}` binds {these} rows but input `{name}` binds {agreed}; a lane's \
                     row inputs carry the same rows",
                    input.name
                ));
            }
            Some(_) => {}
        }
    }
    Ok(rows.map(|(rows, _)| rows))
}

fn input_names(model: &crate::model::Model) -> String {
    if model.inputs().is_empty() {
        return "it reads none".to_string();
    }
    let mut names: Vec<String> = Vec::new();
    for input in model.inputs() {
        let name = format!("`{}`", input.name);
        if !names.contains(&name) {
            names.push(name);
        }
    }
    format!("it reads {}", names.join(", "))
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
    async fn core_new(&mut self) -> Anyhow<Resource<ForwardPass>> {
        crate::inferlet::process::gate::residency_gate(self).await?;
        Ok(self.ctx().table.push(ForwardPass::default())?)
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
            fold,
        } = geom;
        if lanes.is_empty() {
            return Ok(Err(
                "forward pass state binding names no working set; give one per lane".to_string(),
            ));
        }
        for lane in &lanes {
            let _ = self.ctx().table.get(lane)?;
        }
        for channel in [&positions, &slots, &kv_len, &fold].into_iter().flatten() {
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

    async fn core_switch(
        &mut self,
        this: Resource<ForwardPass>,
        lane: u32,
        name: String,
        value: String,
    ) -> Anyhow<Result<(), String>> {
        if let Err(error) = crate::model::model().facts().check(&name, &value) {
            return Ok(Err(error));
        }
        let lane = match lane_index(lane) {
            Ok(lane) => lane,
            Err(error) => return Ok(Err(error)),
        };
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        let switches = &mut pass.bindings.lane_mut(lane).switches;
        if switches.iter().any(|(set, _)| *set == name) {
            return Ok(Err(format!(
                "lane {lane}'s switch `{name}` is already set; a lane keeps its switches for \
                 its pass's life"
            )));
        }
        switches.push((name, value));
        Ok(Ok(()))
    }

    async fn core_input(
        &mut self,
        this: Resource<ForwardPass>,
        lane: u32,
        name: String,
        channel: Resource<Channel>,
    ) -> Anyhow<Result<(), String>> {
        let lane = match lane_index(lane) {
            Ok(lane) => lane,
            Err(error) => return Ok(Err(error)),
        };
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
        let switches = pass
            .bindings
            .lanes
            .get(lane)
            .map(|bound| bound.switches.clone())
            .unwrap_or_default();
        let Some((id, input)) = model.input(&name, &switches) else {
            return Ok(Err(format!(
                "this model reads no input `{name}`; {}",
                input_names(model)
            )));
        };
        let (rows, clip) = match validate_input_channel(input, &shape, dtype) {
            Ok(carried) => carried,
            Err(error) => return Ok(Err(error)),
        };
        let binding = InputBinding {
            name: name.clone(),
            id,
            channel_rep: channel.rep(),
            channel_id: global_id,
            rows,
            clip,
        };
        let pass = self.ctx().table.get_mut(&this)?;
        let inputs = &mut pass.bindings.lane_mut(lane).inputs;
        if inputs.iter().any(|bound| bound.id == id) {
            return Ok(Err(format!(
                "input `{name}` is already bound on lane {lane}"
            )));
        }
        let mut next = inputs.clone();
        next.push(binding.clone());
        if let Err(error) = input_rows(&next) {
            return Ok(Err(format!("lane {lane}: {error}")));
        }
        inputs.push(binding);
        Ok(Ok(()))
    }

    async fn core_group(
        &mut self,
        this: Resource<ForwardPass>,
        lane: u32,
        id: u32,
    ) -> Anyhow<Result<(), String>> {
        let lane = match lane_index(lane) {
            Ok(lane) => lane,
            Err(error) => return Ok(Err(error)),
        };
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        let group = &mut pass.bindings.lane_mut(lane).group;
        if group.is_some() {
            return Ok(Err(format!("lane {lane}'s group is already set")));
        }
        *group = Some(id);
        Ok(Ok(()))
    }

    async fn core_mask(
        &mut self,
        this: Resource<ForwardPass>,
        rows: Resource<Channel>,
    ) -> Anyhow<Result<(), String>> {
        let _ = self.ctx().table.get(&rows)?;
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        if pass.bindings.mask.is_some() {
            return Ok(Err("forward pass mask is already attached".to_string()));
        }
        pass.bindings.mask = Some(rows.rep());
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

    async fn core_media(
        &mut self,
        this: Resource<ForwardPass>,
        lane: u32,
        spans: Vec<pie::inferlet::forward::MediaSpan>,
    ) -> Anyhow<Result<(), String>> {
        use pie::inferlet::forward::MediaSpan;
        let lane = match lane_index(lane) {
            Ok(lane) => lane,
            Err(error) => return Ok(Err(error)),
        };
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
        // A span's tokens ride one fire.
        let max_rows = crate::engine::get_spec(0)?.limits.max_forward_tokens;
        if let Some(span) = attached.iter().find(|s| s.token_count as usize > max_rows) {
            return Ok(Err(format!(
                "lane {lane}'s media span is {} tokens and one fire carries at most {max_rows}; \
                 a span rides one fire, so encode it smaller",
                span.token_count
            )));
        }
        let pass = self.ctx().table.get_mut(&this)?;
        if pass.is_bound() {
            return Ok(Err("forward pass program is already attached".to_string()));
        }
        let media = &mut pass.bindings.lane_mut(lane).media;
        if !media.is_empty() {
            return Ok(Err(format!(
                "lane {lane}'s media spans are already attached"
            )));
        }
        *media = attached;
        Ok(Ok(()))
    }

    async fn core_program(
        &mut self,
        this: Resource<ForwardPass>,
        container_bytes: Vec<u8>,
        channels: Vec<Resource<Channel>>,
    ) -> Anyhow<Result<(), String>> {
        let (embed, state, readout, mask, facts, input_bindings, float, readouts) = {
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
            let model = crate::model::model();
            let count = match &pass.bindings.state {
                Some(state) => state.lanes.len(),
                None => pass.bindings.lanes.len().max(1),
            };
            if pass.bindings.lanes.len() > count {
                return Ok(Err(format!(
                    "lane {} binds a switch, an input, a group or media, and the pass's \
                     `state` binds {count} working set(s): one a lane",
                    pass.bindings.lanes.len() - 1
                )));
            }
            let mut lanes = pass.bindings.lanes.clone();
            lanes.resize_with(count, Default::default);
            let mut decls = Vec::with_capacity(count);
            for (at, lane) in lanes.iter().enumerate() {
                match model.lane(&lane.switches) {
                    Ok(decl) => decls.push(decl),
                    Err(error) => return Ok(Err(format!("lane {at}: {error}"))),
                }
            }
            let (reads_kv, embeds) = (decls[0].reads_kv, decls[0].embeds);
            if let Some(at) = decls
                .iter()
                .position(|decl| decl.reads_kv != reads_kv || decl.embeds != embeds)
            {
                return Ok(Err(format!(
                    "lane {at} {} a kv cache and {} tokens, and lane 0 {} and {}; a pass's \
                     lanes all read kv or none, and all embed or none",
                    if decls[at].reads_kv {
                        "reads"
                    } else {
                        "reads no"
                    },
                    if decls[at].embeds {
                        "embeds"
                    } else {
                        "embeds no"
                    },
                    if reads_kv { "reads one" } else { "reads none" },
                    if embeds { "embeds" } else { "embeds none" },
                )));
            }
            let embed = match (pass.bindings.embed, embeds) {
                (Some(embed), true) => Some(embed),
                (None, true) => {
                    return Ok(Err(
                        "forward pass embed binding must be attached before program".to_string(),
                    ));
                }
                (Some(_), false) => {
                    return Ok(Err(
                        "this pass's lanes embed no tokens, but the pass bound `embed`".to_string(),
                    ));
                }
                (None, false) => None,
            };
            // A float lane reading a recurrent state binds `state` and states no geometry.
            let reads_state = !reads_kv && decls[0].reads_state;
            if let Some(at) = decls
                .iter()
                .position(|decl| !reads_kv && decl.reads_state != reads_state)
            {
                return Ok(Err(format!(
                    "lane {at} {} a recurrent state and lane 0 {}; a pass's lanes all read one \
                     or none",
                    if reads_state { "reads no" } else { "reads" },
                    if reads_state {
                        "reads one"
                    } else {
                        "reads none"
                    },
                )));
            }
            let state = match (pass.bindings.state.clone(), reads_kv || reads_state) {
                (Some(state), true) => Some(state),
                (None, true) => {
                    return Ok(Err(
                        "forward pass state binding must be attached before program".to_string(),
                    ));
                }
                (Some(_), false) => {
                    return Ok(Err(
                        "this pass's lanes read no kv cache and no recurrent state, but the pass \
                         bound `state`"
                            .to_string(),
                    ));
                }
                (None, false) => None,
            };
            if reads_state
                && let Some(state) = &state
                && [state.positions, state.slots, state.kv_len, state.fold]
                    .iter()
                    .any(Option::is_some)
            {
                return Ok(Err(
                    "this pass's lanes read no kv cache: each fire folds one slot of their \
                     recurrent state, so their `state` binds no geometry"
                        .to_string(),
                ));
            }
            if reads_kv && !embeds {
                return Ok(Err(
                    "this pass's lanes read a kv cache and embed no tokens; a sequence's rows \
                     are its tokens, and these lanes state none"
                        .to_string(),
                ));
            }
            for (at, (lane, decl)) in lanes.iter().zip(&decls).enumerate() {
                let reads = &decl.inputs;
                if let Some(&missing) = reads
                    .iter()
                    .find(|&&id| !lane.inputs.iter().any(|bound| bound.id == id))
                {
                    let name = &model.inputs()[missing as usize].name;
                    return Ok(Err(format!(
                        "lane {at} reads input `{name}` and the pass bound no channel to it; \
                         call `input({at}, \"{name}\", channel)` before `program`"
                    )));
                }
                if let Some(bound) = lane.inputs.iter().find(|bound| !reads.contains(&bound.id)) {
                    return Ok(Err(format!(
                        "lane {at} reads no input `{}`, but the pass bound one",
                        bound.name
                    )));
                }
            }
            let max_rows = crate::engine::get_spec(0)?.limits.max_forward_tokens;
            let float = if reads_kv {
                None
            } else if embeds {
                let Some(ids) = embed_ids else {
                    return Ok(Err(
                        "a cacheless encoder pass binds `embed`; this one bound none".to_string(),
                    ));
                };
                if ids == 0 || ids > max_rows {
                    return Ok(Err(format!(
                        "this pass's lanes embed tokens and the channel bound to `embed` holds \
                         {ids} of them; one fire carries 1..={max_rows}"
                    )));
                }
                Some(FloatPass {
                    lanes: vec![
                        FloatLane {
                            rows: 0,
                            clips: Vec::new(),
                        };
                        count
                    ],
                    embed: true,
                })
            } else {
                let mut floats = Vec::with_capacity(count);
                for (at, lane) in lanes.iter().enumerate() {
                    let clips: Vec<[u32; 3]> = lane
                        .inputs
                        .iter()
                        .filter_map(|binding| binding.clip)
                        .collect();
                    // One row per patch of each clip, padded to its input's `rows`.
                    let tokens = |[pt, ph, pw]: [u32; 3]| {
                        clips
                            .iter()
                            .map(|[t, h, w]| (t / pt.max(1)) * (h / ph.max(1)) * (w / pw.max(1)))
                            .fold(0u32, u32::saturating_add)
                    };
                    let stated = lane
                        .inputs
                        .iter()
                        .filter(|binding| binding.clip.is_some())
                        .find_map(|binding| model.inputs()[binding.id as usize].rows);
                    let padded = |tokens: u32| match stated {
                        None => Some(tokens),
                        Some(poem_ir::RowCount::Exact(n)) => (tokens <= n).then_some(n),
                        Some(poem_ir::RowCount::Multiple(n)) => tokens.checked_next_multiple_of(n),
                    };
                    let rows = match input_rows(&lane.inputs) {
                        Ok(Some(rows)) => rows,
                        Ok(None) if !clips.is_empty() => {
                            match decls[at].patch.map(tokens).map(padded) {
                                None => 1,
                                Some(Some(rows)) => rows,
                                Some(None) => {
                                    return Ok(Err(format!(
                                        "lane {at}'s clips hold more patches than its voxel \
                                         input's {stated:?} rows"
                                    )));
                                }
                            }
                        }
                        Ok(None) => {
                            return Ok(Err(format!(
                                "lane {at} embeds no tokens and binds no row input; nothing \
                                 states its row count"
                            )));
                        }
                        Err(error) => return Ok(Err(format!("lane {at}: {error}"))),
                    };
                    floats.push(FloatLane { rows, clips });
                }
                let total: u64 = floats.iter().map(|lane| u64::from(lane.rows)).sum();
                if total > max_rows as u64 {
                    return Ok(Err(format!(
                        "this pass's lanes bind {total} rows; one fire carries at most \
                         {max_rows} (`model.max-latent-rows()`)"
                    )));
                }
                Some(FloatPass {
                    lanes: floats,
                    embed: false,
                })
            };
            let facts: Vec<LaneFacts> = lanes
                .iter()
                .zip(&decls)
                .map(|(lane, decl)| LaneFacts {
                    switches: decl.switches.clone(),
                    group: lane.group,
                    inputs: lane.inputs.iter().map(InputBinding::feed).collect(),
                })
                .collect();
            let readouts: Vec<Vec<eta_ir::registry::OutputDecl>> = decls
                .iter()
                .map(|decl| decl.outputs.iter().map(eta_output).collect())
                .collect();
            let input_bindings: Vec<InputBinding> =
                lanes.iter().flat_map(|lane| lane.inputs.clone()).collect();
            (
                embed,
                state,
                pass.bindings.readout,
                pass.bindings.mask,
                facts,
                input_bindings,
                float,
                readouts,
            )
        };
        {
            let prog = match crate::pipeline::program::register(container_bytes, readouts) {
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
                (Port::AttnMask, mask),
                (Port::Fold, geometry(|state| state.fold)),
                (Port::Readout, readout),
            ];
            if let Err(error) =
                validate_descriptor_bindings(&prog.bound.container, &channel_reps, &expected)
            {
                return Ok(Err(error));
            }
            let listed: Vec<u32> = prog
                .bound
                .container
                .inputs
                .iter()
                .filter_map(|&dense| channel_reps.get(dense as usize).copied())
                .collect();
            if let Some(input) = input_bindings
                .iter()
                .find(|input| !listed.contains(&input.channel_rep))
            {
                return Ok(Err(format!(
                    "pipeline: input `{}`'s channel is not listed as an input of this pass's program",
                    input.name
                )));
            }
            if let Some(&dense) = prog.bound.container.inputs.iter().find(|&&dense| {
                !input_bindings
                    .iter()
                    .any(|input| channel_reps.get(dense as usize) == Some(&input.channel_rep))
            }) {
                return Ok(Err(format!(
                    "pipeline: the program lists channel {dense} as an input that no `input` binds"
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
            let (kv_lanes, rs_reps) = self.lane_parts(&lanes)?;
            // A seatless float pass binds on its process's engine.
            let engine = match kv_lanes.first() {
                Some(&rep) => super::working_set::kv_of(self, rep)?.engine,
                None => crate::planner::engine_of(self.id()),
            };
            let (devgeo, decode_envelope, geometry_class) = if float.is_none() {
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
                    engine,
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
                    // The ports the program computes on the device: the
                    // engine's to resolve, and no others.
                    decode_envelope.unwrap_or(PortMask::NONE),
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
                rs_ws: rs_reps,
                failed: None,
                devgeo,
                decode_envelope,
                host_shadow,
                facts,
                float,
                closed: false,
            };
            if let Err(error) = self.ctx().table.get_mut(&this)?.attach_bound(bound) {
                return Ok(Err(format!("pipeline: {error}")));
            }
            Ok(Ok(()))
        }
    }

    async fn core_drop(&mut self, this: Resource<ForwardPass>) -> Anyhow<()> {
        let rep = this.rep();
        let Some(bound) = self.ctx().table.delete(this)?.take_bound() else {
            return Ok(());
        };
        if let Some(mut bound) = crate::pipeline::fire::retire_pass(rep, bound) {
            bound.close_native();
        }
        Ok(())
    }
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
    async fn new(&mut self) -> Anyhow<Resource<ForwardPass>> {
        self.core_new().await
    }

    async fn state(
        &mut self,
        this: Resource<ForwardPass>,
        lanes: Vec<Resource<WorkingSet>>,
        geom: pie::inferlet::forward::StateGeometry,
    ) -> Anyhow<Result<(), String>> {
        self.core_state(this, lanes, geom).await
    }

    async fn checkpoints(
        &mut self,
        this: Resource<ForwardPass>,
        slots: Vec<u32>,
    ) -> Anyhow<Result<(), String>> {
        self.core_checkpoints(this, slots).await
    }

    async fn media(
        &mut self,
        this: Resource<ForwardPass>,
        lane: u32,
        spans: Vec<pie::inferlet::forward::MediaSpan>,
    ) -> Anyhow<Result<(), String>> {
        self.core_media(this, lane, spans).await
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

    async fn switch(
        &mut self,
        this: Resource<ForwardPass>,
        lane: u32,
        name: String,
        value: String,
    ) -> Anyhow<Result<(), String>> {
        self.core_switch(this, lane, name, value).await
    }

    async fn input(
        &mut self,
        this: Resource<ForwardPass>,
        lane: u32,
        name: String,
        ch: Resource<Channel>,
    ) -> Anyhow<Result<(), String>> {
        self.core_input(this, lane, name, ch).await
    }

    async fn group(
        &mut self,
        this: Resource<ForwardPass>,
        lane: u32,
        id: u32,
    ) -> Anyhow<Result<(), String>> {
        self.core_group(this, lane, id).await
    }

    async fn mask(
        &mut self,
        this: Resource<ForwardPass>,
        rows: Resource<Channel>,
    ) -> Anyhow<Result<(), String>> {
        self.core_mask(this, rows).await
    }
}

fn lane_index(lane: u32) -> Result<usize, String> {
    let max = crate::engine::get_spec(0)
        .map_err(|error| error.to_string())?
        .limits
        .max_forward_requests;
    match usize::try_from(lane) {
        Ok(lane) if lane < max => Ok(lane),
        _ => Err(format!(
            "lane {lane} is past the {max} lanes one fire carries"
        )),
    }
}

/// An output as a program reads it: an i32 output as i32, any float one f32.
fn eta_output(output: &poem_ir::OutputDecl) -> eta_ir::registry::OutputDecl {
    use eta_ir::registry::Axis;
    eta_ir::registry::OutputDecl {
        name: output.name.clone(),
        axis: match output.axis {
            poem_ir::InputAxis::Lane => Axis::Lane,
            poem_ir::InputAxis::Row => Axis::Row,
            poem_ir::InputAxis::Voxel => Axis::Voxel,
        },
        width: output.width,
        dtype: match output.dtype {
            poem_ir::Dtype::I32 => eta_ir::Dtype::I32,
            _ => eta_ir::Dtype::F32,
        },
        layers: output.layers,
        readout: output.readout,
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
    use super::{input_rows, state_rebind_diff, validate_input_channel};
    use crate::pipeline::instance::{InputBinding, StateBinding};
    use eta_ir::types::Dtype;

    fn input(
        name: &'static str,
        axis: poem_ir::InputAxis,
        width: u32,
        dtype: poem_ir::Dtype,
    ) -> poem_ir::InputDecl {
        poem_ir::InputDecl {
            name: name.to_string(),
            axis,
            width,
            dtype,
            rows: None,
            mask: 0,
            readers: Vec::new(),
        }
    }

    fn bound(name: &str, rows: Option<u32>) -> InputBinding {
        InputBinding {
            name: name.to_string(),
            id: 0,
            channel_rep: 1,
            channel_id: ::engine::ChannelId(1),
            rows,
            clip: None,
        }
    }

    #[test]
    fn forward_every_case() {
        an_input_channel_is_validated_against_its_declaration();
        a_lane_s_row_inputs_agree_on_their_rows();
        a_rebind_changes_working_sets_not_geometry();
    }

    fn an_input_channel_is_validated_against_its_declaration() {
        use poem_ir::InputAxis::{Lane, Row, Voxel};
        let latents = input("latents", Row, 64, poem_ir::Dtype::Bf16);
        assert_eq!(
            validate_input_channel(&latents, &[256, 64], Dtype::F32),
            Ok((Some(256), None))
        );
        assert!(
            validate_input_channel(&latents, &[256, 32], Dtype::F32)
                .unwrap_err()
                .contains("`latents`")
        );
        assert!(validate_input_channel(&latents, &[256 * 64], Dtype::F32).is_err());
        assert!(validate_input_channel(&latents, &[0, 64], Dtype::F32).is_err());
        assert!(
            validate_input_channel(&latents, &[256, 64], Dtype::I32)
                .unwrap_err()
                .contains("F32")
        );

        let timestep = input("timestep", Lane, 1, poem_ir::Dtype::F32);
        assert_eq!(
            validate_input_channel(&timestep, &[1], Dtype::F32),
            Ok((None, None))
        );
        assert!(validate_input_channel(&timestep, &[1, 1], Dtype::F32).is_err());
        assert!(validate_input_channel(&timestep, &[2], Dtype::F32).is_err());

        let ids = input("self_cond_ids", Row, 4, poem_ir::Dtype::I32);
        assert_eq!(
            validate_input_channel(&ids, &[16, 4], Dtype::I32),
            Ok((Some(16), None))
        );
        assert!(validate_input_channel(&ids, &[16, 4], Dtype::F32).is_err());

        let latent = input("latent", Voxel, 16, poem_ir::Dtype::Bf16);
        assert_eq!(
            validate_input_channel(&latent, &[3, 4, 16], Dtype::F32),
            Ok((None, Some([1, 3, 4])))
        );
        assert_eq!(
            validate_input_channel(&latent, &[2, 3, 4, 16], Dtype::F32),
            Ok((None, Some([2, 3, 4])))
        );
    }

    fn a_lane_s_row_inputs_agree_on_their_rows() {
        let agree = [
            bound("latents", Some(256)),
            bound("positions", Some(256)),
            bound("timestep", None),
        ];
        assert_eq!(input_rows(&agree), Ok(Some(256)));
        let disagree = [bound("latents", Some(256)), bound("positions", Some(128))];
        assert!(input_rows(&disagree).unwrap_err().contains("`positions`"));
        assert_eq!(input_rows(&[bound("timestep", None)]), Ok(None));
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
