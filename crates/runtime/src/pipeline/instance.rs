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

#[derive(Default)]
pub struct ForwardBindings {
    pub embed: Option<EmbedBinding>,
    pub state: Option<StateBinding>,
    pub readout: Option<u32>,
    pub mask: Option<u32>,
    pub checkpoints: Vec<u32>,
    /// What each lane binds, by lane: as many as the highest lane a verb
    /// named, plus one.
    pub lanes: Vec<LaneBinding>,
}

impl ForwardBindings {
    /// Lane `lane`'s bindings, the lanes before it bound to nothing.
    pub fn lane_mut(&mut self, lane: usize) -> &mut LaneBinding {
        if self.lanes.len() <= lane {
            self.lanes.resize_with(lane + 1, LaneBinding::default);
        }
        &mut self.lanes[lane]
    }
}

/// What one lane of a pass binds.
#[derive(Clone, Default)]
pub struct LaneBinding {
    /// The model's switches the lane set, each checked against its
    /// declaration.
    pub switches: Vec<(String, String)>,
    pub group: Option<u32>,
    pub inputs: Vec<InputBinding>,
    pub media: Vec<std::sync::Arc<media::front::EncodedSpan>>,
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

/// What one lane of a bound pass carries to its fires.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaneFacts {
    /// The model's switches the lane set; its word is built from them and
    /// the runtime switches.
    pub switches: Vec<(String, String)>,
    pub group: Option<u32>,
    pub inputs: Vec<::engine::fire::InputFeed>,
}

impl LaneFacts {
    /// Stamps each lane of `req` with its own facts, `lanes[i]` on lane `i`.
    pub fn stamp(lanes: &[LaneFacts], req: &mut crate::engine::FireRequest) {
        for (lane, facts) in req.lanes.iter_mut().zip(lanes) {
            lane.group = facts.group;
            lane.inputs = facts.inputs.clone();
        }
    }
}

/// A lane of a pass whose lanes read no kv: the rows it carries and the
/// clips its voxel inputs bind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatLane {
    pub rows: u32,
    pub clips: Vec<[u32; 3]>,
}

/// A pass whose lanes read no kv: each lane's rows, and whether its rows are
/// embedded tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatPass {
    pub lanes: Vec<FloatLane>,
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
    pub fold: Option<u32>,
}

#[derive(Default)]
pub struct ForwardPass {
    pub bindings: ForwardBindings,
    bound: Option<Box<BoundForwardPass>>,
}

impl ForwardPass {
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
    pub rs_ws: Vec<u32>,
    pub failed: Option<String>,
    pub devgeo: Option<Vec<u32>>,
    pub decode_envelope: Option<eta_ir::registry::PortMask>,
    pub host_shadow: crate::pipeline::fire::shadow::HostShadow,
    /// Each lane's facts, in lane order.
    pub facts: Vec<LaneFacts>,
    pub float: Option<FloatPass>,
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
