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

    pub fn fire_geometry(
        &self,
    ) -> Result<
        crate::pipeline::fire::geometry::ReqGeometry,
        crate::pipeline::fire::geometry::GeometryError,
    > {
        crate::pipeline::fire::geometry::map_geometry(
            &self.program.bound.container,
            &self.channel_values(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPageSpan {
    pub start: u64,
    pub end: Option<u64>,
}

impl KvPageSpan {
    pub fn resolve(self, page_len: u64) -> Result<std::ops::Range<u64>, String> {
        let end = self.end.unwrap_or(page_len);
        if self.start > end || end > page_len {
            return Err(format!(
                "KV page declaration {}..{} exceeds lease extent {page_len}",
                self.start, end
            ));
        }
        Ok(self.start..end)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvDeclaration {
    pub readable: KvPageSpan,
    pub writable: KvPageSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Attention,
    Recurrent,
    Hybrid,
    Diffusion,
}

impl PassKind {
    pub fn name(self) -> &'static str {
        match self {
            PassKind::Attention => "attention",
            PassKind::Recurrent => "recurrent",
            PassKind::Hybrid => "hybrid",
            PassKind::Diffusion => "diffusion",
        }
    }

    pub fn interface(self) -> &'static str {
        match self {
            PassKind::Attention => "pie:inferlet/forward",
            PassKind::Recurrent => "pie:inferlet/forward-recurrent",
            PassKind::Hybrid => "pie:inferlet/forward-hybrid",
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
    pub attention: Option<AttentionBinding>,
    pub readout: Option<u32>,
    pub rs_ws: Vec<u32>,
    pub rs_geom: Option<RsGeometryBinding>,
    pub rs_fold_len: Option<Vec<u32>>,
    pub max_layers: Option<u32>,
    pub block_draft: bool,
    pub media: Vec<std::sync::Arc<models::media::EncodedSpan>>,
    pub canvas: Option<CanvasMode>,
    pub self_cond: Option<SelfCondPayload>,
    pub reading: Option<u8>,
    pub stream: Option<models::Stream>,
    pub group: Option<u32>,
    pub peer: Option<u32>,
    pub ports: Vec<PortBinding>,
    pub attn_classes: Option<::engine::fire::AttnClasses>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortBinding {
    pub name: String,
    pub kind: ::engine::fire::PortKind,
    pub port: u8,
    pub channel_rep: u32,
    pub channel_id: u64,
    pub rows: Option<u32>,
    pub clip: Option<[u32; 3]>,
}

impl PortBinding {
    #[must_use]
    pub fn feed(&self) -> ::engine::fire::PortFeed {
        ::engine::fire::PortFeed {
            kind: self.kind,
            port: self.port,
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
    pub ports: Vec<::engine::fire::PortFeed>,
    pub attn_classes: Option<::engine::fire::AttnClasses>,
}

impl LaneFacts {
    pub fn stamp(&self, req: &mut crate::engine::FireRequest) {
        for lane in &mut req.lanes {
            lane.reading = self.reading;
            lane.stream = self.stream;
            lane.group = self.group;
            lane.peer = self.peer;
            lane.ports = self.ports.clone();
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
pub fn lane_stream_of(stream: models::Stream) -> ::engine::fire::LaneStream {
    use ::engine::fire::LaneStream;
    match stream {
        models::Stream::Text => LaneStream::Text,
        models::Stream::Image => LaneStream::Image,
        models::Stream::Video => LaneStream::Video,
        models::Stream::Audio => LaneStream::Audio,
        models::Stream::Context => LaneStream::Context,
        models::Stream::Reference => LaneStream::Reference,
    }
}

#[must_use]
pub fn stream_of_lane(stream: ::engine::fire::LaneStream) -> models::Stream {
    use ::engine::fire::LaneStream;
    match stream {
        LaneStream::Text => models::Stream::Text,
        LaneStream::Image => models::Stream::Image,
        LaneStream::Video => models::Stream::Video,
        LaneStream::Audio => models::Stream::Audio,
        LaneStream::Context => models::Stream::Context,
        LaneStream::Reference => models::Stream::Reference,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatLane {
    pub rows: u32,
    pub clips: Vec<[u32; 3]>,
    pub embed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelfCondPayload {
    pub taps: u32,
    pub rows: Vec<u32>,
    pub weights: Vec<f32>,
    pub channels: Option<(u64, u64)>,
}

#[derive(Clone, Copy, Debug)]
pub struct RsGeometryBinding {
    pub fold_len: u32,
    #[allow(
        dead_code,
        reason = "written at bind time from the guest's `rs-geometry` record \
                  (`host/forward.rs`, where `page_span` still rejects a malformed \
                  span) and read by nothing. That is not a missing check. The WIT \
                  doc justifies the field by saying a fire needing an ungranted \
                  page must fail rather than have the runtime quietly find one -- \
                  and the runtime has no way to find one: `RsStore::alloc_buffer` \
                  is reachable ONLY from the guest's own `alloc-buffer` call, so \
                  nothing grows a buffer mid-fire. What could actually overrun is \
                  a fold, and `validate_fold` refuses that against both physical \
                  capacity and live occupancy on every `prepare_*` path. So the \
                  grant is the SEVENTH buffer-addressing channel, not a ceiling: \
                  the six above it were deleted for exactly this reason -- the \
                  runtime derives them from the store it is already authoritative \
                  for. Removing it is a WIT change and wants the deliberation the \
                  other six got, so it is described here rather than done here"
    )]
    pub buffer: KvPageSpan,
}

#[derive(Clone, Copy)]
pub struct EmbedBinding {
    pub tokens: u32,
    pub indptr: u32,
}

#[derive(Clone, Copy)]
pub struct AttentionBinding {
    pub kv_ws: u32,
    pub readable: KvPageSpan,
    pub writable: KvPageSpan,
    pub kv_len: u32,
    pub pages: u32,
    pub page_indptr: u32,
    pub w_slot: u32,
    pub w_off: u32,
    pub positions: u32,
    pub mask: Option<u32>,
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
    pub kv_ws: u32,
    pub kv_declaration: KvDeclaration,
    pub max_layers: Option<u32>,
    pub block_draft: bool,
    pub rs_ws: Vec<u32>,
    pub rs_fold_len: Option<Vec<u32>>,
    pub kv_declaration_realized: bool,
    pub failed: Option<String>,
    pub devgeo: Option<crate::pipeline::fire::lease::DevGeo>,
    pub decode_envelope: Option<crate::pipeline::fire::geometry::DecodeEnvelope>,
    pub host_shadow: crate::pipeline::fire::shadow::HostShadow,
    pub lane: LaneFacts,
    pub float: Option<FloatLane>,
    pub(crate) closed: bool,
}

impl BoundForwardPass {
    pub fn replace_rs_working_sets(&mut self, reps: Vec<u32>) -> Result<(), String> {
        let pending = self
            .fires
            .as_ref()
            .map(|fifo| fifo.lock().unwrap().len())
            .unwrap_or(0);
        if pending != 0 {
            return Err(format!(
                "cannot replace rs-working-sets while {pending} operation(s) remain in the pass FIFO"
            ));
        }
        self.rs_ws = reps;
        Ok(())
    }

    pub fn close_native(&mut self) {
        if std::mem::replace(&mut self.closed, true) {
            return;
        }
        crate::offload::close_home_instance(self.bound_instance.instance_id);
        if let Err(error) = self.scheduler.close_instance(
            self.bound_instance.instance_id,
            self.bound_instance.pacing_wait_id,
        ) {
            tracing::warn!(
                instance_id = self.bound_instance.instance_id,
                %error,
                "forward-pass native cleanup: close_instance failed"
            );
        }
        for cell in &self.cells {
            cell.lock().unwrap().detach(self.bound_instance.instance_id);
        }
        if let Some(devgeo) = self.devgeo.as_mut() {
            let _ = devgeo.lease.reclaim_all();
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
                instance_id = self.bound_instance.instance_id,
                pending_fires = self
                    .fires
                    .as_ref()
                    .map(|fifo| fifo.lock().unwrap().len())
                    .unwrap_or(0),
                "forward-pass dropped with its fires FIFO non-empty, bypassing \
                 HostForwardPass::drop's async drain; skipping native teardown \
                 (close_instance / channel detach / device-geometry reclaim) to avoid \
                 racing a live engine completion into a use-after-free or premature \
                 page reuse — this leaks the engine instance and its channel \
                 attachments until process exit"
            );
        }
    }
}
