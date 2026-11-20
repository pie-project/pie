//! The guest program plane as the shared engine drives it: registering
//! programs and channels, binding instances, moving cells, and what a fire
//! reads of an instance. Each backend's plane runs its device halves its own
//! way.

use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::program::{
    BoundInstance, InstanceBinding, InstanceId, ProgramFacts, ProgramId, ProgramRegistration,
    Rebind, RingFacts,
};
use eta_compiler::codegen::launch::LaunchPackage;

use crate::serve::{InputCell, Serve};
use crate::store::Pools;

/// Where a plane places what an instance or channel holds on the device:
/// the device, the state pool whose `Instance` and `Channel` slots the
/// runtime took for it, and the alignment of a table in its slot.
pub struct Place<'a, B: Serve> {
    pub device: &'a B::Context,
    pub pools: &'a Pools<B>,
    pub align: u64,
    /// The buffers a guest pass reads a fire's outputs from: each arm's
    /// readout and pixel seats, as the load made them.
    pub seats: Vec<&'a B::Buffer>,
    /// The most lanes a fire carries, and the planes it reads out (readout
    /// and pixel): a pass reads at most one row of each plane per lane.
    pub max_lanes: u32,
    pub planes: u32,
    /// The backend's state of the load, holding what its shelf reserved
    /// for the plane before the pool was sized.
    pub shelf: &'a B::Shelf,
}

#[derive(Debug)]
pub enum Refusal {
    Closed { what: &'static str, id: u64 },

    Program(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Closed { what, id } => write!(f, "{what} {id} is closed"),
            Refusal::Program(why) => f.write_str(why),
        }
    }
}

impl From<String> for Refusal {
    fn from(why: String) -> Refusal {
        Refusal::Program(why)
    }
}

/// The ring ends `package`'s stages move, as (its dense channel, whether the
/// end is the tail), each once.
#[must_use]
pub fn moved_ends(package: &LaunchPackage) -> std::collections::BTreeSet<(u32, bool)> {
    let mut ends = std::collections::BTreeSet::new();
    for stage in &package.stages {
        ends.extend(stage.takes.iter().map(|&channel| (channel, false)));
        ends.extend(stage.puts.iter().map(|put| (put.channel, true)));
    }
    ends
}

/// A program as its registration's compile adopted it, on any thread: its
/// plan and its stages' dispatches ([`crate::guest::Compiled`]).
pub struct Adopted {
    pub plan: eta_exec::ExecPlan,

    pub form: crate::guest::Compiled,
}

/// What a registration's compile makes for [`Programs::finish`]: the
/// program adopted, or why its registration is refused.
pub type Prepared = Result<Adopted, Refusal>;

/// A registration's compile, on any thread ([`engine::Engine::prepare_program`]):
/// `registration`'s launch package adopted and its stages lowered to the
/// calls the backend built at load; refused by the program's hash where it
/// does not interpret or a stage has no device form (a guest pass runs on
/// the device whole or not at all, and these engines run no other).
pub fn adopt(registration: &ProgramRegistration) -> Prepared {
    let hash = registration.program_hash;
    let plan = eta_exec::adopt_launch_package(registration.launch.clone()).map_err(|error| {
        Refusal::Program(format!("program 0x{hash:016x} does not adopt: {error}"))
    })?;
    if !plan.executable {
        return Err(Refusal::Program(format!(
            "program 0x{hash:016x} calls a semantic boundary this backend does not implement: {}",
            plan.reject_reason.as_deref().unwrap_or("no reason given")
        )));
    }
    let form = crate::guest::Compiled::of(&plan.package).map_err(|why| {
        Refusal::Program(format!("program 0x{hash:016x} has no device form: {why}"))
    })?;
    Ok(Adopted { plan, form })
}

/// A backend's program plane.
pub trait Programs<B: Serve>: Default {
    /// A registration's last step, on the engine's thread: the adopted
    /// program's id assigned and its form held; a compile that refused
    /// refuses here.
    fn finish(&mut self, prepared: Prepared) -> Result<ProgramId, Refusal>;

    /// Releases `id`, which no instance binds (`Engine::close_program`).
    fn close_program(&mut self, id: ProgramId) -> Result<(), Refusal>;

    /// A registration whole, on the calling thread.
    fn register(&mut self, registration: &ProgramRegistration) -> Result<ProgramId, Refusal> {
        self.finish(adopt(registration))
    }

    fn package(&self, program: ProgramId) -> Option<&LaunchPackage>;

    /// Builds what every instance's device half shares, at the load: the
    /// pipeline of every call a guest stage can dispatch
    /// ([`crate::guest::step_calls`]), and the readout widen's pipeline and
    /// its config. Binding an instance, or registering a program, while
    /// serving then builds no pipeline.
    fn load(&mut self, place: &Place<'_, B>) -> Result<(), Refusal>;

    /// The pipelines built after the load, at a bind: none, where
    /// [`Programs::load`] built every one a guest stage can dispatch.
    fn built_serving(&self) -> u64 {
        0
    }

    /// Takes guest step call `call`'s pipeline out, so the next bind that
    /// dispatches it builds it while serving: the control of
    /// [`Programs::built_serving`].
    fn forget_step(&mut self, call: u32) {
        let _ = call;
    }

    /// What an instance of `program` holds in the pool's `Instance` and
    /// `Channel` slots (`Engine::program_facts`).
    fn facts(&self, program: ProgramId, place: &Place<'_, B>) -> Result<ProgramFacts, Refusal>;

    /// The `Channel` slots a registered channel takes, if any
    /// (`Engine::ring_facts`).
    fn ring_facts(
        &self,
        registration: &ChannelRegistration,
        place: &Place<'_, B>,
    ) -> Result<Option<RingFacts>, Refusal>;

    /// Registers a channel; what it keeps on the device in the slots its
    /// registration names, where a model is loaded.
    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
        place: Option<&Place<'_, B>>,
    ) -> Result<RegisteredChannel, Refusal>;

    fn close_channel(&mut self, id: ChannelId) -> Result<(), Refusal>;

    /// Binds an instance; its device half in the slots its binding names,
    /// where a model is loaded, its pair's state at the binding's `pair`
    /// place as `layout` lays it ([`crate::guest::state`]).
    fn bind(
        &mut self,
        binding: &InstanceBinding,
        place: Option<&Place<'_, B>>,
        layout: &crate::guest::state::GuestLayout,
    ) -> Result<BoundInstance, Refusal>;

    /// The score room of `instance`'s pair, where its program reads
    /// captured scores: a fire carrying its lanes may lay its capturing
    /// lanes' rows there ([`crate::guest::state`]).
    fn scores(&self, instance: InstanceId) -> Option<crate::serve::DeviceCell<B>> {
        let _ = instance;
        None
    }

    /// Before an evacuation moves chunks: every guest pass the plane has
    /// yet to put on the queue is on it, so none binds a chunk's old place
    /// after the move.
    fn quiesce(&mut self) -> Result<(), Refusal> {
        Ok(())
    }

    /// Points every table and cell the plane holds in the pool at the
    /// chunk its slot ref sits in now, after an evacuation moved chunks.
    fn rebind(&mut self, pools: &Pools<B>) -> Result<(), Refusal>;

    /// Moves an instance's or a channel's tables and cells to the slot
    /// refs `rebind` names, whose bytes the runtime copied there
    /// (`Engine::rebind_slots`).
    fn rebind_slots(&mut self, pools: &Pools<B>, rebind: &Rebind) -> Result<(), Refusal>;

    fn close_instance(&mut self, id: InstanceId) -> Result<(), Refusal>;

    /// The ring ends `instance`'s stages move, as (channel, whether the end
    /// is the tail), each once; none for an instance the plane holds no
    /// longer.
    fn moved_ends(&self, instance: InstanceId) -> Vec<(ChannelId, bool)>;

    /// The oldest cell guest input channel `id` of `instance` holds.
    fn input_cell(&self, instance: InstanceId, id: ChannelId) -> Result<InputCell<B>, Refusal>;

    /// The rows a lane of `instance` folds, where the instance states them.
    fn envelope_fold(&self, instance: InstanceId) -> Result<Option<u32>, Refusal>;

    fn publish(&mut self, instance: InstanceId, channel: u32, cell: &[u8])
    -> Result<bool, Refusal>;

    fn take(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>, Refusal>;

    /// The outputs `instance` reads, each with the lane of its attachment.
    fn reads(&self, instance: InstanceId) -> Vec<(String, u32)>;

    /// Fires `instance`'s epilogue over the output seats bound to it,
    /// `seats[k]` its lane `k`'s, and waits for it. A seat no row the
    /// program reads lies in is refused by name before anything runs.
    fn fire_seated(
        &mut self,
        device: &B::Context,
        instance: InstanceId,
        seats: Vec<Vec<crate::guest::run::Readout<'_, B>>>,
    ) -> Result<(), Refusal>;
}

/// The refusal of a seat no row of the program reads.
#[must_use]
pub fn unread(instance: InstanceId, output: &str, lane: usize) -> Refusal {
    Refusal::Program(format!(
        "instance {instance}: output `{output}` is bound at lane {lane}, which this program \
         reads no row of"
    ))
}

/// The rows `embed_indptr` places each of an instance's lanes at: its const
/// value, or the cell its ring holds at the fire; empty when it binds none.
#[must_use]
pub fn embed_indptr(plan: &eta_exec::ExecPlan, inst: &eta_exec::InterpInstance) -> Vec<u32> {
    use eta_ir::registry::Port;
    let words = |value: &eta_exec::Value| match value {
        eta_exec::Value::U32(v) => v.clone(),
        eta_exec::Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
        _ => Vec::new(),
    };
    match plan
        .package
        .ports
        .iter()
        .find(|binding| binding.port == Port::EmbedIndptr)
    {
        Some(binding) if binding.is_const => plan
            .const_ports
            .iter()
            .find(|c| c.port == Port::EmbedIndptr)
            .map(|c| words(&c.value))
            .unwrap_or_default(),
        Some(binding) => inst
            .channels
            .get(binding.channel as usize)
            .filter(|ring| !ring.is_empty())
            .map(|ring| words(&ring.front()))
            .unwrap_or_default(),
        None => Vec::new(),
    }
}

/// The lanes an instance's descriptor ports describe: one per
/// `embed_indptr` bound pair, and one where it binds none.
#[must_use]
pub fn lane_count(plan: &eta_exec::ExecPlan, inst: &eta_exec::InterpInstance) -> usize {
    embed_indptr(plan, inst).len().saturating_sub(1).max(1)
}

/// Lane `lane_at`'s rows of a port cell of `cells` rows a lane of `rows`
/// rows reads: all of it for a lone lane, else where `embed_indptr` places it.
pub fn lane_span(
    indptr: &[u32],
    instance: InstanceId,
    what: &str,
    cells: usize,
    rows: usize,
    lane_at: usize,
) -> Result<std::ops::Range<usize>, Refusal> {
    if lane_at == 0 && cells == rows {
        return Ok(0..rows);
    }
    let (Some(&start), Some(&end)) = (indptr.get(lane_at), indptr.get(lane_at + 1)) else {
        return Err(Refusal::Program(format!(
            "instance {instance}: `{what}` carries {cells} cell(s) for lane {lane_at} of \
             {rows} row(s), and the program's `embed_indptr` ({indptr:?}) does not place it"
        )));
    };
    let (start, end) = (start as usize, end as usize);
    if end < start || end > cells || end - start != rows {
        return Err(Refusal::Program(format!(
            "instance {instance}: `{what}` places lane {lane_at} at {start}..{end} of \
             {cells} cell(s), and the lane has {rows} row(s)"
        )));
    }
    Ok(start..end)
}

/// Lane `lane_at`'s attention mask out of the `attn_mask` cell `value` of an
/// instance whose mask channel is `[.., keys]`: a row of runs for each of the
/// lane's `rows` query rows, one extent where it has one.
pub fn lane_mask(
    value: &eta_exec::Value,
    keys: usize,
    indptr: &[u32],
    instance: InstanceId,
    rows: usize,
    lane_at: usize,
) -> Result<engine::fire::Masking, Refusal> {
    let eta_exec::Value::Bool(cells) = value else {
        return Err(Refusal::Program(format!(
            "instance {instance}: `attn_mask` holds {:?}, not bools",
            value.dtype()
        )));
    };
    if keys == 0 || cells.len() % keys != 0 {
        return Err(Refusal::Program(format!(
            "instance {instance}: `attn_mask` holds {} cell(s) over {keys} key(s)",
            cells.len()
        )));
    }
    let span = lane_span(
        indptr,
        instance,
        "attn_mask",
        cells.len() / keys,
        rows,
        lane_at,
    )?;
    let masks: Vec<engine::fire::Mask> = cells[span.start * keys..span.end * keys]
        .chunks_exact(keys)
        .map(|row| engine::fire::Mask::new(mask_runs(row), keys as u64))
        .collect();
    Ok(match masks.as_slice() {
        [only] => engine::fire::Masking::Extent(only.clone()),
        _ => engine::fire::Masking::Rows(masks),
    })
}

/// A row of bool cells as alternating run lengths, the first of cells unset.
#[must_use]
pub fn mask_runs(row: &[bool]) -> Vec<u32> {
    let mut runs = Vec::new();
    let mut current = false;
    let mut count = 0u32;
    if row.first().is_some_and(|&cell| cell) {
        runs.push(0);
        current = true;
    }
    for &value in row {
        if value == current {
            count += 1;
        } else {
            runs.push(count);
            current = value;
            count = 1;
        }
    }
    if !row.is_empty() {
        runs.push(count);
    }
    runs
}

/// The elements a cell of `shape` holds (a scalar holds one).
#[must_use]
pub fn numel_of(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product::<usize>().max(1)
}

/// A wire cell of `dtype` and `shape` as a host value; `None` where its
/// bytes do not fit the cell or the dtype has no host form.
#[must_use]
pub fn decode(
    bytes: &[u8],
    dtype: eta_ir::container::ChanDType,
    shape: &[u32],
) -> Option<eta_exec::Value> {
    use eta_ir::types::Dtype;
    let dtype = eta_exec::concrete_dtype(dtype);
    let numel = numel_of(shape);
    if bytes.len() != eta_exec::wire_cell_bytes(dtype, numel) {
        return None;
    }
    let words = || {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2], c[3]])
    };
    Some(match dtype {
        Dtype::Bool => eta_exec::Value::Bool(
            (0..numel)
                .map(|j| (bytes[j / 8] >> (j % 8)) & 1 != 0)
                .collect(),
        ),
        Dtype::I32 => eta_exec::Value::I32(words().map(i32::from_le_bytes).collect()),
        Dtype::U32 => eta_exec::Value::U32(words().map(u32::from_le_bytes).collect()),
        Dtype::F32 => eta_exec::Value::F32(words().map(f32::from_le_bytes).collect()),
        _ => return None,
    })
}
