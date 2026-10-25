use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::program::RegisteredProgram;

static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSeed {
    pub channel: u32,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct Instance {
    pub program: Arc<RegisteredProgram>,
    pub instance_id: u64,
    pub seeds: Vec<ChannelSeed>,
}

impl Instance {
    pub fn channel_values(&self) -> Vec<Option<Vec<u8>>> {
        let mut v = vec![None; self.program.bound.container.channels.len()];
        for s in &self.seeds {
            v[s.channel as usize] = Some(s.data.clone());
        }
        v
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Text,
    Diffusion,
}

impl PassKind {
    pub fn name(self) -> &'static str {
        match self {
            PassKind::Text => "text",
            PassKind::Diffusion => "diffusion",
        }
    }

    pub fn interface(self) -> &'static str {
        match self {
            PassKind::Text => "pie:inferlet/forward",
            PassKind::Diffusion => "pie:inferlet/forward-diffusion",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanvasMode {
    Encode,
    Denoise,
}

#[derive(Default)]
pub struct ForwardBindings {
    pub embed: Option<EmbedBinding>,
    pub state: Option<StateBinding>,
    pub readout: Option<u32>,
    pub block_draft: bool,
    pub checkpoints: Vec<u32>,
    pub media: Vec<std::sync::Arc<media::front::EncodedSpan>>,
    pub canvas: Option<CanvasMode>,
    pub reading: Option<u8>,
    pub stream: Option<poem::Stream>,
    pub group: Option<u32>,
    pub peer: Option<u32>,
    pub inputs: Vec<InputBinding>,
    pub attn_classes: Option<::engine::fire::AttnClasses>,
}

/// A channel bound to the model's input `name` (id `id`): the rows it
/// carries when a row input, the clip when a voxel one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputBinding {
    pub name: String,
    pub id: u32,
    pub channel_rep: u32,
    pub channel_id: ::engine::ChannelId,
    pub rows: Option<u32>,
    pub clip: Option<[u32; 3]>,
}

impl InputBinding {
    #[must_use]
    pub fn feed(&self) -> ::engine::fire::InputFeed {
        ::engine::fire::InputFeed {
            id: self.id,
            channel: self.channel_id,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaneFacts {
    pub reading: u8,
    pub stream: ::engine::fire::LaneStream,
    pub group: Option<u32>,
    pub peer: Option<u32>,
    pub inputs: Vec<::engine::fire::InputFeed>,
    pub attn_classes: Option<::engine::fire::AttnClasses>,
}

impl LaneFacts {
    pub fn stamp(&self, req: &mut crate::engine::FireRequest) {
        for lane in &mut req.lanes {
            lane.reading = self.reading;
            lane.stream = self.stream;
            lane.group = self.group;
            lane.peer = self.peer;
            lane.inputs = self.inputs.clone();
            lane.attn_classes = self.attn_classes.clone();
        }
    }
}

#[must_use]
pub fn cohort_of(
    table: &mut wasmtime::component::ResourceTable,
    group: Option<u32>,
    peer: Option<u32>,
) -> Option<u32> {
    let group = group?;
    let members = table
        .iter_mut()
        .filter_map(|entry| entry.downcast_ref::<ForwardPass>())
        .filter(|pass| {
            pass.bindings.group == Some(group) || (peer.is_some() && pass.bindings.group == peer)
        })
        .count();
    Some(u32::try_from(members).unwrap_or(u32::MAX).max(1))
}

#[must_use]
pub fn cohort_key(group: Option<u32>, peer: Option<u32>) -> Option<u32> {
    match (group, peer) {
        (Some(group), Some(peer)) => Some(group.min(peer)),
        (group, _) => group,
    }
}

#[must_use]
pub fn lane_stream_of(stream: poem::Stream) -> ::engine::fire::LaneStream {
    use ::engine::fire::LaneStream;
    match stream {
        poem::Stream::Text => LaneStream::Text,
        poem::Stream::Image => LaneStream::Image,
        poem::Stream::Video => LaneStream::Video,
        poem::Stream::Audio => LaneStream::Audio,
        poem::Stream::Context => LaneStream::Context,
        poem::Stream::Reference => LaneStream::Reference,
    }
}

#[must_use]
pub fn stream_of_lane(stream: ::engine::fire::LaneStream) -> poem::Stream {
    use ::engine::fire::LaneStream;
    match stream {
        LaneStream::Text => poem::Stream::Text,
        LaneStream::Image => poem::Stream::Image,
        LaneStream::Video => poem::Stream::Video,
        LaneStream::Audio => poem::Stream::Audio,
        LaneStream::Context => poem::Stream::Context,
        LaneStream::Reference => poem::Stream::Reference,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatLane {
    pub rows: u32,
    pub clips: Vec<[u32; 3]>,
    pub embed: bool,
}

#[derive(Clone, Copy)]
pub struct EmbedBinding {
    pub tokens: u32,
    pub indptr: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateBinding {
    pub lanes: Vec<u32>,
    pub positions: Option<u32>,
    pub slots: Option<u32>,
    pub kv_len: Option<u32>,
    pub mask: Option<u32>,
    pub fold: Option<u32>,
}

pub struct ForwardPass {
    pub kind: PassKind,
    pub bindings: ForwardBindings,
    bound: Option<Box<BoundForwardPass>>,
}

impl ForwardPass {
    pub fn new(kind: PassKind) -> Self {
        Self {
            kind,
            bindings: ForwardBindings::default(),
            bound: None,
        }
    }

    pub fn is_bound(&self) -> bool {
        self.bound.is_some()
    }

    pub fn attach_bound(&mut self, bound: BoundForwardPass) -> Result<(), String> {
        if self.bound.is_some() {
            return Err("forward pass program is already attached".to_string());
        }
        self.bound = Some(Box::new(bound));
        Ok(())
    }

    pub fn bound(&self) -> Result<&BoundForwardPass, String> {
        self.bound
            .as_deref()
            .ok_or_else(|| "forward pass program is not attached".to_string())
    }

    pub fn bound_mut(&mut self) -> Result<&mut BoundForwardPass, String> {
        self.bound
            .as_deref_mut()
            .ok_or_else(|| "forward pass program is not attached".to_string())
    }
}

impl std::ops::Deref for ForwardPass {
    type Target = BoundForwardPass;

    fn deref(&self) -> &Self::Target {
        self.bound
            .as_deref()
            .expect("forward-pass runtime use requires an attached program")
    }
}

impl std::ops::DerefMut for ForwardPass {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.bound
            .as_deref_mut()
            .expect("forward-pass runtime use requires an attached program")
    }
}

pub struct BoundForwardPass {
    pub instance: Instance,
    pub bound_instance: crate::engine::BoundInstance,
    pub scheduler: crate::scheduler::worker::SchedulerHandle,
    pub cells: crate::pipeline::channel::BoundCells,
    pub channel_reps: Vec<u32>,
    pub fires: Option<crate::pipeline::fire::PendingFires>,
    pub lanes: Vec<u32>,
    pub kv_lanes: Vec<u32>,
    pub block_draft: bool,
    pub rs_ws: Vec<u32>,
    pub failed: Option<String>,
    pub devgeo: Option<Vec<u32>>,
    pub decode_envelope: Option<eta_ir::registry::PortMask>,
    pub host_shadow: crate::pipeline::fire::shadow::HostShadow,
    pub lane: LaneFacts,
    pub float: Option<FloatLane>,
    pub(crate) closed: bool,
}

impl BoundForwardPass {
    pub fn replace_lanes(
        &mut self,
        lanes: Vec<u32>,
        kv_lanes: Vec<u32>,
        rs_ws: Vec<u32>,
    ) -> Result<(), String> {
        let pending = self
            .fires
            .as_ref()
            .map(|fifo| fifo.lock().unwrap().len())
            .unwrap_or(0);
        if pending != 0 {
            return Err(format!(
                "cannot bind other working sets while {pending} operation(s) remain in the pass FIFO"
            ));
        }
        self.lanes = lanes;
        self.kv_lanes = kv_lanes;
        self.rs_ws = rs_ws;
        Ok(())
    }

    pub fn close_native(&mut self) {
        if std::mem::replace(&mut self.closed, true) {
            return;
        }
        if let Err(error) = self.scheduler.close_instance(
            self.bound_instance.instance_id,
            self.bound_instance.pacing_wait_id,
        ) {
            tracing::warn!(
                instance_id = self.bound_instance.instance_id.0,
                %error,
                "forward-pass native cleanup: close_instance failed"
            );
        }
        for cell in &self.cells {
            cell.lock().unwrap().detach(self.instance.instance_id);
        }
    }

    pub(crate) fn can_close_native_on_drop(&self) -> bool {
        match &self.fires {
            None => true,
            Some(fifo) => fifo.lock().unwrap().is_empty(),
        }
    }
}

impl Drop for BoundForwardPass {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        if self.can_close_native_on_drop() {
            self.close_native();
        } else {
            tracing::error!(
                instance_id = self.bound_instance.instance_id.0,
                pending_fires = self
                    .fires
                    .as_ref()
                    .map(|fifo| fifo.lock().unwrap().len())
                    .unwrap_or(0),
                "forward pass dropped with fires in flight; its engine instance and \
                 channel attachments leak until exit rather than race a live completion"
            );
        }
    }
}
