use crate::ports::GeometryClass;
use serde::{Deserialize, Serialize};

use crate::adapter::AdapterRegistration;
use crate::caps::DeviceFacts;
use crate::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use crate::chunk::{Evacuation, SlotCopy, WeightResidency, WeightUses};
use crate::error::{Error, Result};
use crate::fire::{FrameId, FrameSubmission, FrameTicket, MediaEncode, Step};
use crate::load::{LoadRequest, Loaded};
use crate::program::{
    BindExtents, BoundInstance, FireFacts, GuestState, InstanceBinding, InstanceId, ProgramFacts, ProgramId,
    ProgramRegistration, ReadoutFacts, Rebind, RingFacts,
};
use crate::transfer::{KvCopy, KvHandle, StateCopy};

/// What a program's compile made, off the engine's thread, for the engine
/// that prepared it ([`Engine::finish_program`]).
pub struct Prepared(pub Box<dyn std::any::Any + Send>);

/// A program's compile ([`Engine::prepare_program`]): run once, on any
/// thread.
pub type ProgramCompile = Box<dyn FnOnce() -> Prepared + Send>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepDone {
    pub frame: FrameId,
    pub step: u32,
}

/// How a step landed: what its lanes wrote, and the lanes that faulted.
/// Every lane not in `faulted` committed, its outputs and its `wrote`
/// standing as on a step with no fault at all.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StepOutcome {
    /// What the step's lanes wrote to the KV cache, where they report it.
    pub wrote: Vec<KvWrites>,
    /// The lanes that faulted, each once. The runtime fails each one's
    /// request by its reason, and poisons its instance; no other lane is
    /// touched.
    pub faulted: Vec<LaneFault>,
    /// Each beam group's common slot as the step's head step left it
    /// (`BEAM-PAGE-LISTS.md` §11.3): the slot below which every live beam's
    /// rows lie on the same pages, by the group's instance. Monotone; the
    /// runtime folds the group's set to the largest it has seen.
    pub beam_common: Vec<(InstanceId, u32)>,
}

/// A lane of a step that faulted: lane `lane` of the step's
/// [`crate::fire::Step::lanes`], the instance whose pass faulted where one
/// did, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneFault {
    pub lane: u32,
    pub instance: Option<InstanceId>,
    pub reason: String,
}

impl StepOutcome {
    /// Every lane committed, and none reports what it wrote.
    #[must_use]
    pub fn committed() -> StepOutcome {
        StepOutcome::default()
    }

    /// Every lane committed, and these are what they wrote.
    #[must_use]
    pub fn wrote(wrote: Vec<KvWrites>) -> StepOutcome {
        StepOutcome {
            wrote,
            faulted: Vec::new(),
            beam_common: Vec::new(),
        }
    }

    /// The step was lost on the device after its submit returned (a
    /// command buffer refused, the device gone): every one of its `lanes`
    /// faulted, for `reason`.
    #[must_use]
    pub fn lost(lanes: u32, reason: impl Into<String>) -> StepOutcome {
        StepOutcome {
            wrote: Vec::new(),
            faulted: every_lane(lanes, reason.into()),
            beam_common: Vec::new(),
        }
    }

    /// Lanes `lanes` of `instance`'s pass faulted for `reason`; a lane
    /// already faulted keeps its first reason.
    pub fn fault_lanes(
        &mut self,
        lanes: std::ops::Range<u32>,
        instance: Option<InstanceId>,
        reason: &str,
    ) {
        for lane in lanes {
            if self.faulted.iter().all(|fault| fault.lane != lane) {
                self.faulted.push(LaneFault {
                    lane,
                    instance,
                    reason: reason.to_string(),
                });
            }
        }
    }
}

fn every_lane(lanes: u32, reason: String) -> Vec<LaneFault> {
    (0..lanes)
        .map(|lane| LaneFault {
            lane,
            instance: None,
            reason: reason.clone(),
        })
        .collect()
}

/// What one lane wrote to the KV cache in a step, row by row: the token it
/// embedded at its position, into the slot at `offsets[i]` of pool page
/// `pages[i]`. Rows are empty where the engine cannot place them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvWrites {
    pub tag: u64,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub pages: Vec<u32>,
    pub offsets: Vec<u32>,
    /// The slots this fire's fold froze for the lane: a count, this fire's
    /// alone (not a running total, not a position), after the engine's
    /// clamp to the fire's end; `Some(0)` where it froze none. The runtime
    /// adds it to the set's frozen length when the fire settles, for its
    /// kv and its recurrent state alike. An engine reports it for every
    /// lane that asks a report on a fire whose fold length is the device's
    /// ([`crate::fire::FoldLen::Device`]). `None` there freezes nothing,
    /// and the runtime fails the set by name when the fire settles: the
    /// engine's record went on by a count the runtime never takes, so the
    /// set is not forked, checkpointed, exported or moved from then on, and
    /// every later fire of it carries all the lane wrote since.
    pub fold: Option<u32>,
    /// One past the last slot the lane's rows landed at, in the set's own
    /// slot numbering: a position. `None` where the engine does not tell
    /// it; the set is then taken to have written as far as it reserved
    /// ([`crate::fire::rung_rows`], device geometry). `end` and `fold` are
    /// one report: an engine that reports `end` for a lane whose fire
    /// folds on the device reports `fold` too.
    pub end: Option<u32>,
}

pub type CompletionSink = std::sync::Arc<dyn Fn(StepDone, StepOutcome) + Send + Sync>;

/// How a copy the engine answered as still landing
/// ([`Engine::copy_slots_landing`]) ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Landing {
    /// The copy read every source slot and wrote every destination.
    Landed,
    /// The copy did not complete (its device event faulted, or the tier
    /// could not move its bytes): the destination is not to be trusted; the
    /// sources were not given to any other owner by the engine and hold what
    /// they held before the copy. The reason names the fault.
    Failed(String),
}

/// Where an engine tells how a copy it answered as still landing
/// ([`Engine::copy_slots_landing`]) ended: called once with the copy's
/// token and its [`Landing`], from any thread.
pub type LandingSink = std::sync::Arc<dyn Fn(u64, Landing) + Send + Sync>;

/// A group of a tapped lane's pages, read out: each page's spans of group
/// `group` (of [`Engine::tap_groups`]), page after page; or why not. A tap
/// that fails says so once, for its first group not told.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tapped {
    pub tag: u64,
    pub group: u32,
    pub bytes: std::result::Result<Vec<u8>, String>,
}

pub type TapSink = std::sync::Arc<dyn Fn(Tapped) + Send + Sync>;

/// The groups a tap reads a page's planes out in, each its spans `(offset,
/// bytes)` of a host kv slot.
pub type TapGroups = Vec<Vec<(u64, u64)>>;

/// What the host can still give this process now, in bytes, read afresh on
/// each call: for a device that shares the host's memory, what it has
/// available less what the platform defends past that and the floor;
/// negative by what the host is short of that line, which the process gives
/// back. An error names the reading that could not be taken.
pub type HostRoom = std::sync::Arc<dyn Fn() -> std::result::Result<i64, String> + Send + Sync>;

pub trait Engine: Send {
    fn kind(&self) -> &'static str;

    fn device_facts(&self) -> Option<&DeviceFacts> {
        None
    }

    /// What the runtime's host cover depends on in this engine (`runtime`'s
    /// `measure_host_cover`): its device and driver, and the kernels and
    /// guest code a sampler's fire runs on it, as one fingerprint. A boot
    /// whose engines state the same fingerprints, for the same deployment,
    /// sampler program and frame shape, reads the cover an earlier boot
    /// measured instead of measuring it again. `None`, the default: the
    /// cover is measured at every load.
    fn cover_fingerprint(&self) -> Option<String> {
        None
    }

    /// The reader of what the host can still give this process, where the
    /// device shares the host's memory and no budget of the device's says
    /// what it has: the runtime's governor reads it on its tick and holds
    /// the pool within what the pool holds plus it. `None`, the default:
    /// the device has memory of its own, sized at load.
    fn host_room(&self) -> Option<HostRoom> {
        None
    }

    fn export_kv_handle(&self) -> Option<KvHandle> {
        None
    }

    /// What it counts while it serves, read by the runtime from load on
    /// ([`crate::live`]): the fires served other than by a captured body.
    /// Nothing of arming: which rungs are armed and which eager is a load
    /// fact ([`crate::chunk::ChunkFacts::eager_rungs`]).
    fn live(&self) -> crate::live::Live {
        crate::live::Live::armed()
    }

    fn bind_thread(&mut self) -> Result<()> {
        Ok(())
    }

    fn load(&mut self, request: LoadRequest) -> Result<Loaded>;

    fn register_program(&mut self, registration: &ProgramRegistration) -> Result<ProgramId> {
        let _ = registration;
        Err(self.unsupported("register_program"))
    }

    /// Builds what registering each of `registrations` would build (a
    /// backend's compiled guest code), without registering any: the
    /// runtime's prewarm of the programs its standard samplers register,
    /// before ready, so a registration of one of them builds nothing
    /// (`crate::live::Tally::BuiltAtRegister`). The default builds nothing.
    fn prewarm_programs(&mut self, registrations: &[ProgramRegistration]) -> Result<()> {
        let _ = registrations;
        Ok(())
    }

    /// A registration in two steps, so the thread that fires does not
    /// wait out a program's compile: this, on the engine's thread and
    /// quick, hands back the program's compile, which the runtime runs on a
    /// worker thread while the engine fires on; [`Engine::finish_program`],
    /// on the engine's thread again, takes what the compile made and
    /// answers the program's id. The program binds no instance until then.
    fn prepare_program(&mut self, registration: &ProgramRegistration) -> Result<ProgramCompile>;

    /// The second half of [`Engine::prepare_program`]: the program's id,
    /// from what its compile made; a compile that refused refuses here, by
    /// the program's name.
    fn finish_program(&mut self, prepared: Prepared) -> Result<ProgramId>;

    /// Releases a registered program no instance binds: its id answers
    /// nothing after. The runtime closes one it could not learn the facts
    /// of, so no registration that fails leaves a program behind.
    fn close_program(&mut self, program: ProgramId) -> Result<()>;

    /// What an instance of a registered program holds in the pool's
    /// `Instance` and `Channel` slots, asked before the runtime takes them
    /// and binds. The default states none: the engine keeps its tables and
    /// rings itself.
    fn program_facts(&self, program: ProgramId) -> Result<ProgramFacts> {
        let _ = program;
        Ok(ProgramFacts::default())
    }

    /// The adapter an instance bound by `binding` lands, by its content: a
    /// digest of its planes (every bank, in bank-name order). Asked before
    /// the runtime takes the binding's slots: bindings with one key share
    /// one set of `Adapter` slots ([`InstanceBinding::adapter_slots`]),
    /// landed once, freed at the last close. `None`: it lands none.
    fn adapter_key(&self, binding: &InstanceBinding) -> Result<Option<u128>> {
        let _ = binding;
        Ok(None)
    }

    /// What a fire holding lanes of `program` bound at `extents` needs
    /// beside the model's activation, asked before the first bind of those
    /// extents is sent. The facts are the pair's; what a rung backs is a
    /// fire's ([`crate::GuestTerm`]): the worst fire of the rung's lanes
    /// over the pairs bound, each paying for the members the fire gives it
    /// and nothing where it gives none. The default states none.
    fn fire_facts(&self, program: ProgramId, extents: &BindExtents) -> Result<FireFacts> {
        let _ = (program, extents);
        Ok(FireFacts::default())
    }

    /// What `program` bound at `extents` holds as state ([`GuestState`]):
    /// each instance's scratch and the pair's, taken in the `Instance` role
    /// at the bind ([`crate::ScratchPlace`]), held while bound, and told to
    /// the engine in the binding ([`InstanceBinding::scratch`],
    /// [`InstanceBinding::pair`]) and in every [`Rebind::Instance`]. Asked
    /// at every bind, on the lane, before it is sent. The default states
    /// none: an engine that lays its guests' scratch in the fire's span
    /// ([`Self::fire_facts`]) states it there until it moves it here.
    /// `class` is the geometry class the instance is bound at
    /// ([`InstanceBinding::geometry`]): the engine states its scratch arms
    /// per class ([`GuestState::scratch_arms`]).
    fn guest_state(
        &self,
        program: ProgramId,
        extents: &BindExtents,
        class: GeometryClass,
    ) -> Result<GuestState> {
        let _ = (program, extents, class);
        Ok(GuestState::default())
    }

    /// What the rows a fire reads out take ([`ReadoutFacts`]): the bytes a
    /// row beside the model's activation, or none where they land inside
    /// the span already, and the rows a fixed store holds, if any. Asked
    /// once, after load. A sibling of [`Engine::fire_facts`], not a field
    /// of it: every fire reads out, programs or none, so the facts are the
    /// engine's, not a (program, extents) pair's. The default states the
    /// rows inside the span and no store: no term, no bound, nothing
    /// changes.
    fn readout_facts(&self) -> Result<ReadoutFacts> {
        Ok(ReadoutFacts::default())
    }

    /// The standing guest bytes the engine would hold once `binding` is
    /// bound: the span bytes it keeps for the live instances of registered
    /// programs beside every fire (Metal: the batch tables it lays once a
    /// set, every set together), computed without changing any state. The
    /// total changes only at a bind and a close; a set shared by instances
    /// counts once, so a bind that joins a set already laid adds nothing.
    /// Asked before every bind is sent: the runtime backs the total below
    /// the carve, beside the fire term, before the engine lays it. The
    /// total a bind leaves is [`BoundInstance::standing_guest_bytes`], the
    /// one a close leaves what [`Engine::close_instance`] returns. The
    /// default keeps none.
    fn standing_after(&self, binding: &InstanceBinding) -> Result<u64> {
        let _ = binding;
        Ok(0)
    }

    /// Whether [`Engine::register_channel`] takes `registration`: the
    /// engine then holds the channel's ring (in its `Channel` slots, or
    /// itself, as a host mirror), and an instance bound on the channel
    /// reads that ring. `false` where it answers `Unsupported`: the runtime
    /// keeps the channel alone, and each instance holds its own ring in
    /// slots of its own ([`InstanceBinding::ring_slots`]). A fact of the
    /// registration, asked before anything of a bind is taken, so a bind
    /// that registers a channel takes its instance's rings by it. The
    /// default holds none, as the default `register_channel` takes none.
    fn holds_channel(&self, registration: &ChannelRegistration) -> bool {
        let _ = registration;
        false
    }

    /// The `Channel` slots a registered channel's ring takes, asked before
    /// [`Engine::register_channel`]. `None`: the engine keeps the ring
    /// itself.
    fn ring_facts(&self, registration: &ChannelRegistration) -> Result<Option<RingFacts>> {
        let _ = registration;
        Ok(None)
    }

    /// An instance's or a channel's pool slots moved: the runtime parked
    /// them on another tier and copied them back into the fresh refs
    /// `rebind` names, each shape apart. Called before the next fire that
    /// reads them, which must address the new slots. A park itself is said
    /// too ([`Rebind::Parked`]): the instance's old slots are another
    /// owner's from then on. A channel closed with
    /// an instance still bound on it is still registered
    /// ([`Engine::close_channel`]): its rebind is served like an open one's.
    /// The default does nothing: an engine that states no `Instance` or
    /// `Channel` shape is never asked.
    fn rebind_slots(&mut self, rebind: Rebind) -> Result<()> {
        let _ = rebind;
        Ok(())
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> Result<RegisteredChannel> {
        let _ = registration;
        Err(self.unsupported("register_channel"))
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> Result<BoundInstance> {
        let _ = binding;
        Err(self.unsupported("bind_instance"))
    }

    /// Closes instance `id`: the engine's standing guest bytes after the
    /// close ([`Engine::standing_after`]), less any set no live instance
    /// shares any more. A closed channel this was the last instance bound
    /// on is released with it ([`Engine::close_channel`]): once this
    /// returns, the runtime gives the channel's `Channel` slots back.
    fn close_instance(&mut self, id: InstanceId) -> Result<u64> {
        let _ = id;
        Err(self.unsupported("close_instance"))
    }

    /// Closes channel `id`. With no instance bound on it, the channel and
    /// its endpoint are released now. With an instance still bound on it,
    /// the channel is only closing: its registration, endpoint and ring
    /// stay, every instance bound on it keeps reading and writing it, a
    /// [`Engine::rebind_slots`] of its ring (the runtime parked and
    /// restored it with those instances) is served as an open channel's,
    /// and an evacuation that moves its slots rebases it; no new instance
    /// binds on it. The last such instance's [`Engine::close_instance`]
    /// releases it. The runtime keeps the channel's `Channel` slots owned
    /// until then, and gives them back only after that close returns, so
    /// the engine never addresses slots the runtime handed elsewhere.
    fn close_channel(&mut self, id: ChannelId) -> Result<()> {
        let _ = id;
        Err(self.unsupported("close_channel"))
    }

    fn publish_channel(&mut self, instance: InstanceId, channel: u32, cell: &[u8]) -> Result<bool> {
        let _ = (instance, channel, cell);
        Err(self.unsupported("publish_channel"))
    }

    fn take_channel(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>> {
        let _ = (instance, channel);
        Err(self.unsupported("take_channel"))
    }

    /// Beam `beam`'s page list of instance `instance`'s beam group, as its
    /// last fire left it (`BEAM-PAGE-LISTS.md` §6.6): the pool page of each
    /// column, the group's list width of them (columns past the last fire's
    /// unspecified). Read once a search ends, after its fires have landed;
    /// an engine waits for them.
    fn beam_list(&mut self, instance: InstanceId, beam: u32) -> Result<Vec<u32>> {
        let _ = (instance, beam);
        Err(self.unsupported("beam_list"))
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> Result<()> {
        let _ = registration;
        Err(self.unsupported("register_adapter"))
    }

    /// Submits a frame's steps. A submit that returns `Err` refused the
    /// frame before any of its lanes ran, and fails the pipeline of every
    /// lane in it. A fault an engine can place on lanes (a guest pass
    /// faulting, a guard, a row out of bounds) is not an `Err`: the step
    /// settles with those lanes in [`StepOutcome::faulted`], each failing
    /// its own pipeline, and every other lane commits. The sets a failed
    /// pipeline owns cannot be claimed by another. Nothing replays from
    /// what such a step left. A position an engine keeps by slot
    /// still goes back where a refused step found it: the slot outlives
    /// the set, and its next tenant must not meet it.
    ///
    /// A refusal a step's lanes earn for their own sake (a lane's word,
    /// mask, positions, adapter or payload this load does not serve as
    /// stated) names those lanes, never the frame: refused at the frame's
    /// first step it is [`Error::Lanes`], nothing of the frame having run;
    /// refused at a later step, the frame stops there and answers a ticket
    /// of the steps before it with the lanes in
    /// [`FrameTicket::refused`]. The runtime fails those lanes' fires alone
    /// and submits what is left again.
    fn submit(&mut self, frame: &FrameSubmission) -> Result<FrameTicket>;

    /// Backs the pool's first `segments` segments
    /// ([`crate::chunk::ChunkFacts::segments`]), ahead of the runtime
    /// handing out any of their chunks; never on a fire's path. A refusal
    /// leaves what was committed standing.
    fn commit_segments(&mut self, segments: u32) -> Result<()> {
        let _ = segments;
        Err(self.unsupported("commit_segments"))
    }

    /// Gives back every segment at and past `segments`. The runtime has
    /// evacuated their chunks and waited out every frame that named them;
    /// an [`Self::evacuate`] copy that still reads them, the engine waits
    /// out here.
    fn release_segments(&mut self, segments: u32) -> Result<()> {
        let _ = segments;
        Err(self.unsupported("release_segments"))
    }

    /// Backs the top `segments` segments of the span under the largest
    /// rung's activation, counted from the top of the pool, ahead of the
    /// runtime opening any chunk of theirs for state; never on a fire's
    /// path. Monotone as [`Self::commit_segments`]' prefix, from the other
    /// end. Only where [`crate::chunk::ChunkFacts::span_backed_on_demand`];
    /// the engine backs its own activation carve itself.
    fn commit_span(&mut self, segments: u32) -> Result<()> {
        let _ = segments;
        Ok(())
    }

    /// Keeps the top `segments` span segments backed and gives back the
    /// span segments below them, which hold no state chunk and which no
    /// rung lends any more ([`Self::commit_span`]'s counting). Whatever of
    /// them the engine's own activation carve uses stays its own to back.
    fn release_span(&mut self, segments: u32) -> Result<()> {
        let _ = segments;
        Ok(())
    }

    /// The activation rung moves from `from`, the rung the engine was last
    /// told (the smallest, from the load), to `to` tokens; only where
    /// [`crate::chunk::ChunkFacts::span_backed_on_demand`], through the
    /// lane, never on a fire's path, and only where the two differ. A raise
    /// (`to > from`) comes before any fire of the wider rung, and before
    /// the raise's [`Self::evacuate`] where it moves state: the engine
    /// backs its carve for `to` here, and a refusal (`Error::Exhausted`)
    /// takes the raise back, no fire passing `from`. A lower (`to < from`)
    /// is a notice once every fire wider than `to` has retired: the engine
    /// may give back the carve above `to`; a failed notice is logged, not
    /// fatal.
    ///
    /// A guest term that grows at a standing rung is no rung change and
    /// sends none: the runtime sets the room aside before it sends the bind
    /// that grows it, and the engine covers it inside that bind, both
    /// counting the same [`crate::GuestTerm`]. An engine that cannot cover
    /// it answers the bind with what it lacks (`Error::Impossible`), and
    /// the runtime gives the room back when the bind is through.
    fn set_rung(&mut self, from: u32, to: u32) -> Result<()> {
        let _ = (from, to);
        Ok(())
    }

    /// Copies each moved chunk's bytes onto its new physical chunk and
    /// updates the engine's [`crate::chunk::ChunkTable`], stream-ordered
    /// before the next fire; the slot refs the runtime holds stand. The
    /// copy has a completion of its own, which [`Self::release_segments`]
    /// waits on before the evacuated segment goes: the runtime waits on no
    /// later fire for it, so an idle engine's segment goes too.
    fn evacuate(&mut self, evacuation: &Evacuation) -> Result<()> {
        let _ = evacuation;
        Err(self.unsupported("evacuate"))
    }

    /// Copies slots of one shape between tiers or within one; replaces
    /// [`Engine::copy_kv`] and [`Engine::copy_state`] once every backend
    /// serves it.
    ///
    /// Tiers (CONTRACT-BC 22): an engine that states
    /// [`crate::chunk::ChunkFacts::host_chunks`] moves whole slots of every
    /// state shape between its device domain and
    /// [`crate::transfer::MemoryDomain::HostPinned`], a host ref `r` of
    /// shape `s` at [`crate::chunk::SlotShape::tier_offset`] of its host
    /// chunks, laid out as the device slot (one span a slot); one that
    /// states [`crate::chunk::ChunkFacts::disk_chunks`] moves whole slots
    /// of every state shape between the device (or the host tier) and
    /// [`crate::transfer::MemoryDomain::LocalDisk`], a disk ref laid out
    /// in its disk chunks as a host ref is in its host chunks. Token runs are device to device only. Each copy
    /// is ordered after every fire submitted before it and before every one
    /// after (a stream-ordered copy, or a drain), and copies in submission
    /// order: the runtime names a destination again as soon as the call
    /// returns. A source slot's memory may go to any other slot, of any
    /// shape, which anything the engine does may write (a bind's seeds, a
    /// channel's cells, a guest's stage, not a fire alone), so the runtime
    /// gives a source slot back only once the copy has read it: at once
    /// after this call, or when [`Engine::copy_slots_landing`] tells it.
    fn copy_slots(&mut self, copy: &SlotCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_slots"))
    }

    /// [`Engine::copy_slots`], answering whether its source slots may go
    /// back now (`true`): the copy has read them, or every write that could
    /// reach them after they go back is ordered behind the copy. `false`:
    /// the copy may still read them, and the engine calls the sink
    /// [`Engine::on_landed`] gave it with `token` once it has read every
    /// source slot. The sink is called with [`Landing::Failed`] where the
    /// copy faulted; its sources then still hold their bytes (an engine never
    /// hands them on before telling). Every engine
    /// states it: an answer of `true` for a copy still reading its sources
    /// hands their memory to its next owner under the copy.
    fn copy_slots_landing(&mut self, copy: &SlotCopy, token: u64) -> Result<bool>;

    /// Where [`Engine::copy_slots_landing`] tells a copy landed.
    fn on_landed(&mut self, sink: LandingSink) {
        let _ = sink;
    }

    /// Returns once every copy submitted so far has read its sources and
    /// written its destinations: what a move that gives back slots a copy
    /// was still writing (an aborted restore's destinations) waits for, so
    /// no late write reaches them once they are another's. Every engine
    /// states it.
    fn settle_copies(&mut self) -> Result<()>;

    /// Which atoms of the streamable weights
    /// ([`crate::chunk::ChunkFacts::weights`]) are resident from the next
    /// fire on, and in which pool slots; the whole set each time. The
    /// runtime calls it once a governor tick at most, ordered before the
    /// next fire; the engine fills slots newly named (from its host copy)
    /// before a fire reads them. A use of an atom not named is the
    /// engine's: it streams through a staging slot and is not kept.
    ///
    /// Answers the bytes the engine holds back of the chunks it stated
    /// ([`crate::chunk::ChunkFacts::held_bytes`]) once it has taken the
    /// set. An engine that armed nothing at load arms at the residency the
    /// load starts with and settles its fit there, so it holds less from
    /// then on than its load stated; the runtime lifts the pool's hold-back
    /// by that answer before the load returns. Nothing moves what is held
    /// after that: every later answer repeats it.
    fn weight_residency(&mut self, residency: &WeightResidency) -> Result<u64> {
        let _ = residency;
        Err(self.unsupported("weight_residency"))
    }

    /// How often fires read each streamable atom, resident or not: the
    /// record the runtime orders the weight cache by. An engine may answer
    /// a call behind (the reads between the two calls before, its use
    /// counters read without waiting for the device), every read counted
    /// once.
    fn weight_uses(&mut self) -> Result<WeightUses> {
        Err(self.unsupported("weight_uses"))
    }

    fn settles_asynchronously(&self) -> bool {
        false
    }

    /// Where a step's outcome goes. An engine that settles asynchronously
    /// answers every step here. One that settles inside
    /// [`Engine::submit`] tells here, before `submit` returns, each step
    /// whose lanes wrote or faulted; a step it does not tell committed.
    fn on_complete(&mut self, sink: CompletionSink) {
        let _ = sink;
    }

    /// The groups a tapped page's planes are read out in, each as spans
    /// `(offset, bytes)` of a host kv slot, in the order they land; `None`:
    /// this engine taps nothing.
    fn tap_groups(&self) -> Option<TapGroups> {
        None
    }

    fn on_tapped(&mut self, sink: TapSink) {
        let _ = sink;
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> Result<()> {
        let _ = ticket;
        Ok(())
    }

    /// Publishes what the guests' programs of the steps already landed put
    /// on their channels, without waiting on a step still in flight. An
    /// engine that publishes as each step lands needs nothing here.
    fn publish_landed(&mut self) -> Result<()> {
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_kv"))
    }

    fn copy_state(&mut self, copy: &StateCopy) -> Result<()> {
        let _ = copy;
        Err(self.unsupported("copy_state"))
    }

    fn encode(&mut self, plan: &mut MediaEncode) -> Result<()> {
        let _ = plan;
        Err(self.unsupported("encode"))
    }

    /// How [`Engine::read_host_kv`] lays out pages, when another engine of
    /// another layout should be able to take them; `None` when this one
    /// cannot say.
    fn host_kv_layout(&self) -> Option<state::PageLayout> {
        None
    }

    /// The bytes of host kv slots `slots`, one slot after another, laid out
    /// as this engine lays a slot out, once every copy into them has landed.
    /// What one worker's slots hold, another worker of the same layout takes
    /// with [`Engine::write_host_kv`].
    fn read_host_kv(&mut self, slots: &[u32]) -> Result<Vec<u8>> {
        let _ = slots;
        Err(self.unsupported("read_host_kv"))
    }

    /// Writes `bytes`, laid out as [`Engine::read_host_kv`] returns them, into
    /// host kv slots `slots`.
    fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> Result<()> {
        let _ = (slots, bytes);
        Err(self.unsupported("write_host_kv"))
    }

    /// The bytes of host rs rows: `slots`' state, then `pages`' buffers, one
    /// after another, once every copy into them has landed. For a model whose
    /// state is a window, `pages` are host windowed pages and `slots` empty.
    fn read_host_state(&mut self, slots: &[u32], pages: &[u32]) -> Result<Vec<u8>> {
        let _ = (slots, pages);
        Err(self.unsupported("read_host_state"))
    }

    /// Writes `bytes`, laid out as [`Engine::read_host_state`] returns them.
    fn write_host_state(&mut self, slots: &[u32], pages: &[u32], bytes: &[u8]) -> Result<()> {
        let _ = (slots, pages, bytes);
        Err(self.unsupported("write_host_state"))
    }

    /// The bytes of host tier slots (CONTRACT-BC 22): for each `(shape,
    /// refs)` in order, each ref's slot of that shape at
    /// [`crate::chunk::SlotShape::tier_offset`] of the host chunks, laid out
    /// as the device slot (what [`Engine::copy_slots`] moves whole), one
    /// slot after another; once every copy into them has landed (the
    /// runtime waits on that copy's completion first). What a shared pool's
    /// snapshot reads of a set's state: its record's pieces and its rs list
    /// or window slots, parked on the host.
    fn read_tier_slots(&mut self, slots: &[(crate::chunk::ShapeId, Vec<u32>)]) -> Result<Vec<u8>> {
        let _ = slots;
        Err(self.unsupported("read_tier_slots"))
    }

    /// Writes `bytes`, laid out as [`Engine::read_tier_slots`] returns
    /// them, into host tier slots `slots`; a later [`Engine::copy_slots`]
    /// from the host tier reads them.
    fn write_tier_slots(
        &mut self,
        slots: &[(crate::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> Result<()> {
        let _ = (slots, bytes);
        Err(self.unsupported("write_tier_slots"))
    }

    fn disconnect(&self, message: &str) {
        let _ = message;
    }

    fn unsupported(&self, verb: &'static str) -> Error {
        Error::unsupported(self.kind(), verb)
    }
}

const _: () = {
    #[allow(dead_code)]
    fn object_safe(engine: &dyn Engine) -> &'static str {
        engine.kind()
    }
};
