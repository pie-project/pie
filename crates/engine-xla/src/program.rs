//! The eta program plane, from engine-wgpu's plane without its wgpu device half.
//! A program runs its passes on the device (`Plane::fire_guests`),
//! instances of one program batched into one executable per stage; one with
//! a stage that does not lower to StableHLO (`crate::guest`), or that reads
//! an output no fire of the load returns on the device, is refused at its
//! registration.
#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Where an attachment runs: its plan, its content hash, where it reads its
/// outputs.
type Placed = (Arc<ExecPlan>, [u8; 32], Vec<crate::guest::run::Bound>);

use engine::channel::{ChannelId, ChannelRegistration, HostMirror, RegisteredChannel};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use eta_exec::{ChannelState, ExecPlan, HostOp, InterpInstance, PassInputs, StepOutcome, Value};
use eta_ir::registry::{GeometryClass, Port, Stage};

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

struct Instance {
    program: ProgramId,
    plan: Arc<ExecPlan>,
    inst: InterpInstance,
    geometry: GeometryClass,

    channels: Vec<ChannelId>,
    /// The `Channel` slots of the rings the instance holds itself, as its
    /// binding (or the last rebind) names them.
    own_rings: Vec<engine::program::RingSlots>,

    poisoned: Arc<AtomicBool>,

    /// Each carried channel's cell: the pool row its `Channel` slots start
    /// at (empty where the program carries none), and whether the cells
    /// there are the instance's current ones (else its rings' fronts are).
    cells: Vec<(u32, u32)>,
    on_device: bool,
    /// Each fed channel's ring (`embed_tokens`): its first pool row, its
    /// capacity and a cell's rows. Sequence `q`'s cell lies at `first +
    /// (q mod capacity) · rows`, where a pass puts it and the next fire
    /// reads it.
    fed: Vec<(u32, u32, u32, u32)>,
}

impl Instance {
    /// Each pooled channel's row this pass: a carried cell's own, a fed
    /// ring's at its tail (where the pass's put lands).
    fn rows(&self) -> Vec<(u32, u32)> {
        let mut out = self.cells.clone();
        for &(channel, first, capacity, rows) in &self.fed {
            let tail = self.inst.channels[channel as usize].tail();
            let at = u32::try_from(tail % u64::from(capacity.max(1))).unwrap_or(0);
            out.push((channel, first + at * rows));
        }
        out
    }
}

impl Instance {
    fn host(&mut self) -> &InterpInstance {
        if self.poisoned.load(Ordering::Acquire) {
            self.inst.poisoned = true;
        }
        &self.inst
    }
}

#[derive(Default)]
pub(crate) struct Tally {
    device: AtomicU64,
}

/// A step's completion: `sink(at, ..)` runs once every hold on it is
/// released (the submitter's own, and one per flight or device event the
/// step waits on), with the first fault any of them reported.
pub(crate) struct Latch {
    at: engine::StepDone,
    /// The step's lanes: every one faulted where the step faults.
    lanes: u32,
    sink: Option<engine::CompletionSink>,
    holds: std::sync::atomic::AtomicUsize,
    /// A fault of the whole step (its fire, a device event): every lane.
    fault: std::sync::Mutex<Option<String>>,
    /// Each attached instance's lanes of the step, where its pass's fault
    /// lands (`Latch::fault_instances`).
    spans: std::sync::Mutex<Vec<(InstanceId, std::ops::Range<u32>)>>,
    /// The lanes whose instance's pass faulted; the rest commit.
    lane_faults: std::sync::Mutex<engine::StepOutcome>,
    /// What the step's reported lanes wrote, told when it lands.
    writes: std::sync::Mutex<Vec<engine::KvWrites>>,
    /// The step's outcome once every hold is released: `Some(fault)`.
    landed: std::sync::Mutex<Option<Option<String>>>,
    done: std::sync::Condvar,
}

impl Latch {
    /// A latch holding once, for the submitter.
    pub(crate) fn new(
        at: engine::StepDone,
        lanes: u32,
        sink: Option<engine::CompletionSink>,
    ) -> Arc<Latch> {
        Arc::new(Latch {
            at,
            lanes,
            sink,
            holds: std::sync::atomic::AtomicUsize::new(1),
            fault: std::sync::Mutex::new(None),
            spans: std::sync::Mutex::new(Vec::new()),
            lane_faults: std::sync::Mutex::new(engine::StepOutcome::committed()),
            writes: std::sync::Mutex::new(Vec::new()),
            landed: std::sync::Mutex::new(None),
            done: std::sync::Condvar::new(),
        })
    }

    /// Waits until every hold is released; the first fault any reported.
    pub(crate) fn wait(&self) -> Option<String> {
        let mut landed = self.landed.lock().expect("landed lock poisoned");
        while landed.is_none() {
            landed = self.done.wait(landed).expect("landed lock poisoned");
        }
        landed.clone().flatten()
    }

    pub(crate) fn hold(&self) {
        self.holds.fetch_add(1, Ordering::AcqRel);
    }

    /// The step's attached instances and the lanes each one's pass is for.
    pub(crate) fn seat_spans(&self, spans: Vec<(InstanceId, std::ops::Range<u32>)>) {
        *self.spans.lock().expect("spans lock poisoned") = spans;
    }

    /// `instances`' passes faulted for `why`: their lanes fault, every
    /// other lane of the step commits. An instance the step does not seat
    /// faults the whole step.
    pub(crate) fn fault_instances(&self, instances: &[InstanceId], why: &str) {
        let spans = self.spans.lock().expect("spans lock poisoned");
        let mut faults = self.lane_faults.lock().expect("lane faults lock poisoned");
        for instance in instances {
            match spans.iter().find(|(id, _)| id == instance) {
                Some((_, lanes)) => faults.fault_lanes(lanes.clone(), Some(*instance), why),
                None => {
                    self.fault
                        .lock()
                        .expect("fault lock poisoned")
                        .get_or_insert_with(|| {
                            format!("instance {instance}, which the step does not seat: {why}")
                        });
                }
            }
        }
    }

    /// The step's reported lanes wrote `writes`: told with its landing.
    pub(crate) fn report(&self, writes: Vec<engine::KvWrites>) {
        self.writes
            .lock()
            .expect("writes lock poisoned")
            .extend(writes);
    }

    pub(crate) fn release(&self, outcome: Result<(), String>) {
        if let Err(why) = outcome {
            self.fault
                .lock()
                .expect("fault lock poisoned")
                .get_or_insert(why);
        }
        if self.holds.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        tracing::debug!(
            frame = self.at.frame.0,
            step = self.at.step,
            "xla step landed"
        );
        let fault = self.fault.lock().expect("fault lock poisoned").take();
        // The sink hears the step (its fold reports with it) before a
        // waiting settle wakes: an engine that settles inside submit tells
        // the sink before submit returns.
        if let Some(sink) = &self.sink {
            sink(
                self.at,
                match fault.clone() {
                    None => {
                        let mut outcome = engine::StepOutcome::wrote(std::mem::take(
                            &mut *self.writes.lock().expect("writes lock poisoned"),
                        ));
                        outcome.faulted = std::mem::take(
                            &mut self
                                .lane_faults
                                .lock()
                                .expect("lane faults lock poisoned")
                                .faulted,
                        );
                        outcome
                    }
                    Some(why) => engine::StepOutcome::lost(self.lanes, why),
                },
            );
        }
        *self.landed.lock().expect("landed lock poisoned") = Some(fault);
        self.done.notify_all();
    }
}

/// A guest pass whose last stage's output is still coming back: the puts
/// that read it, and the poison flags of the instances it commits for.
struct Flight {
    out: Arc<crate::pjrt::Buffer>,
    out_words: usize,
    puts: Vec<crate::guest::run::DeferredPut>,
    poison: Vec<Arc<AtomicBool>>,
}

/// A landed flight's outcome on its step: a fault lands on the lanes of
/// the instances it committed for.
fn landed(latch: &Latch, instances: &[InstanceId], outcome: Result<(), String>) {
    if let Err(why) = outcome {
        latch.fault_instances(instances, &why);
    }
    latch.release(Ok(()));
}

/// Guest passes in flight. A flight lands on the plugin's thread when its
/// download completes: every deferred cell is written and only then its
/// ring's tail stored, under the book's lock, which every host reader of a
/// ring that may hold an in-flight cell takes first.
#[derive(Default)]
pub(crate) struct Flights {
    book: std::sync::Mutex<Book>,
    landed: std::sync::Condvar,
}

/// The open flights by id, and the device buffers landed flights let go
/// of: those are dropped on the engine's thread, never on the plugin's.
#[derive(Default)]
struct Book {
    next: u64,
    open: BTreeMap<u64, Flight>,
    dead: Vec<Arc<crate::pjrt::Buffer>>,
}

/// A raw download destination handed to the plugin's thread.
struct Landing(*mut [u8]);
// SAFETY: the plugin writes the bytes and then runs the callback that
// reclaims them; nothing else touches them in between.
unsafe impl Send for Landing {}

/// Where an in-flight cell sits: the pass's output buffer, the flat word
/// its first element is at, and its slot.
pub struct InFlight {
    pub out: Arc<crate::pjrt::Buffer>,
    pub word: usize,
    pub slot: crate::guest::lower::Slot,
}

impl Flights {
    fn lock(&self) -> std::sync::MutexGuard<'_, Book> {
        self.book.lock().expect("book lock poisoned")
    }

    /// Starts a deferred pass's download; it lands itself when the bytes
    /// are back and releases its hold on `latch`.
    fn launch(
        self: &Arc<Self>,
        deferred: crate::guest::run::Deferred,
        poison: Vec<Arc<AtomicBool>>,
        instances: Vec<InstanceId>,
        latch: Option<&Arc<Latch>>,
    ) {
        let crate::guest::run::Deferred {
            out,
            out_words,
            batch,
            puts,
            ..
        } = deferred;
        let (id, dead) = {
            let mut book = self.lock();
            book.next += 1;
            let id = book.next;
            book.open.insert(
                id,
                Flight {
                    out: Arc::clone(&out),
                    out_words,
                    puts,
                    poison,
                },
            );
            (id, std::mem::take(&mut book.dead))
        };
        drop(dead);
        if let Some(latch) = latch {
            latch.hold();
        }
        let bytes = out.host_bytes().unwrap_or(batch * out_words * 4);
        let dst: *mut [u8] = Box::into_raw(vec![0u8; bytes].into_boxed_slice());
        // SAFETY: `dst` is freed only by the callback below, after the
        // event says the copy landed (or here, if it never started).
        let event = unsafe { out.download_into(&mut *dst) };
        let event = match event {
            Ok(event) => event,
            Err(why) => {
                // SAFETY: `dst` came from `Box::into_raw` above, and the
                // copy failed to start, so the plugin never took it.
                drop(unsafe { Box::from_raw(dst) });
                let outcome = self.land(id, Err(why.to_string()), None);
                if let Some(latch) = latch {
                    landed(latch, &instances, outcome);
                }
                return;
            }
        };
        let flights = Arc::clone(self);
        let hold = latch.cloned();
        let theirs = instances.clone();
        let landing = Landing(dst);
        let keep = Arc::clone(&out);
        let registered = event.on_ready(move |done| {
            let landing = landing;
            // SAFETY: the copy has landed (or failed); the bytes are ours.
            let raw = unsafe { Box::from_raw(landing.0) };
            let outcome = flights.land(
                id,
                done.map(|()| raw.into_vec()).map_err(|e| e.to_string()),
                Some(keep),
            );
            if let Some(latch) = hold {
                landed(&latch, &theirs, outcome);
            }
        });
        if let Err(why) = registered {
            // The callback never runs: its bytes are leaked rather than
            // freed under a copy that may still be writing them.
            let outcome = self.land(id, Err(why.to_string()), None);
            if let Some(latch) = latch {
                landed(latch, &instances, outcome);
            }
        }
    }

    /// Writes flight `id`'s cells into their rings, then their tails.
    fn land(
        &self,
        id: u64,
        raw: Result<Vec<u8>, String>,
        keep: Option<Arc<crate::pjrt::Buffer>>,
    ) -> Result<(), String> {
        let mut book = self.lock();
        book.dead.extend(keep);
        let Some(flight) = book.open.remove(&id) else {
            return Ok(());
        };
        let outcome = raw.and_then(|raw| {
            let all = crate::guest::run::words_of(&raw);
            for put in &flight.puts {
                let value = crate::guest::run::cell_of(&all, flight.out_words, put.row, &put.slot)?;
                put.ring.encode_sequence(put.seq, &value);
            }
            Ok(())
        });
        match &outcome {
            Ok(()) => {
                for put in &flight.puts {
                    if put.ring.tail() <= put.seq {
                        put.ring.store_tail(put.seq + 1);
                    }
                }
            }
            Err(why) => {
                tracing::warn!(%why, "a guest pass's cells did not come back from the device");
                for flag in &flight.poison {
                    flag.store(true, Ordering::Release);
                }
            }
        }
        book.dead.push(Arc::clone(&flight.out));
        let Flight { puts, poison, .. } = flight;
        drop((puts, poison));
        drop(book);
        self.landed.notify_all();
        outcome.map_err(|why| format!("a guest pass faulted on the device: {why}"))
    }

    fn on_ring(book: &Book, ring: &Arc<ChannelState>) -> bool {
        book.open
            .values()
            .any(|flight| flight.puts.iter().any(|put| Arc::ptr_eq(&put.ring, ring)))
    }

    /// Blocks until no flight puts into `ring`.
    pub(crate) fn wait_ring(&self, ring: &Arc<ChannelState>) {
        let mut book = self.lock();
        while Self::on_ring(&book, ring) {
            book = self.landed.wait(book).expect("landed lock poisoned");
        }
    }

    /// Blocks until every flight has landed.
    pub(crate) fn wait_all(&self) {
        let mut book = self.lock();
        while !book.open.is_empty() {
            book = self.landed.wait(book).expect("landed lock poisoned");
        }
    }

    /// Whether any flight is open.
    pub(crate) fn airborne(&self) -> bool {
        !self.lock().open.is_empty()
    }

    /// The in-flight cell `ring` will hold at `seq`, if one is coming.
    fn cell_at(&self, ring: &Arc<ChannelState>, seq: u64) -> Option<InFlight> {
        let book = self.lock();
        book.open.values().find_map(|flight| {
            flight
                .puts
                .iter()
                .find(|put| Arc::ptr_eq(&put.ring, ring) && put.seq == seq)
                .map(|put| InFlight {
                    out: Arc::clone(&flight.out),
                    word: put.row * flight.out_words + put.slot.at,
                    slot: put.slot,
                })
        })
    }
}

impl Flights {
    /// Waits out every flight and lets go of the buffers they left, while
    /// the client that made them is still up.
    fn drain(&self) {
        self.wait_all();
        let dead = std::mem::take(&mut self.lock().dead);
        drop(dead);
    }
}

/// A lane's tokens off its `embed_tokens` channel: on the host, or still on
/// the device in the previous pass's output (`words` are the flat `u32`
/// indexes of each token in `out`).
pub enum Tokens {
    Host(Vec<u32>),
    /// Where in the pool the last pass put the tokens: each a pool row and
    /// a lane of it (its fed ring's slots), which the fire gathers. Row and
    /// lane apart, never one element index over the whole pool (an i32 of
    /// those reaches 8 GiB).
    Device { at: Vec<(u32, u32)> },
}

impl Tokens {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Tokens::Host(ids) => ids.len(),
            Tokens::Device { at } => at.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Default)]
pub struct Plane {
    programs: BTreeMap<ProgramId, Arc<ExecPlan>>,

    forms: crate::guest::Forms,

    channels: BTreeMap<ChannelId, Arc<ChannelState>>,
    /// Each registered channel's `Channel` slots, as its registration names
    /// them ([`Plane::ring_facts`]): where an instance that carries or feeds
    /// it on the device keeps its cells, the same rows for every instance
    /// bound to it.
    channel_slots: BTreeMap<ChannelId, Vec<u32>>,
    instances: BTreeMap<InstanceId, Instance>,
    next_program: u64,
    next_instance: u64,

    tally: Arc<Tally>,
    /// Where a pass the host interprets is counted ([`Plane::report_to`]).
    live: Option<engine::live::Live>,

    /// Compiled guest stages, by program content and batch shape.
    stages: crate::guest::run::Stages,

    /// Guest passes whose last stage is still on the device.
    flights: Arc<Flights>,
}

impl Plane {
    /// From now on every pass the host interprets is counted in `live`.
    pub fn report_to(&mut self, live: engine::live::Live) {
        self.live = Some(live);
    }

    /// The most lanes a fire seats, which a stage first seen compiles up to.
    pub fn seat(&mut self, lanes: u32) {
        self.stages.top = lanes;
    }

    /// The readout rows a fire's planes come in, which a stage first seen
    /// compiles for too.
    pub fn seat_readouts(&mut self, rows: Vec<u32>) {
        self.stages.readouts = rows;
    }

    /// The pool a guest stage carrying cells takes: its rows and its sink's
    /// first row (a padding lane's cells).
    pub fn seat_pool(&mut self, rows: i64, sink_row: u32) {
        self.stages.pool_rows = rows;
        self.stages.sink_row = sink_row;
    }

    /// Every plane a fire of the load can return (`Shell::guest_planes`):
    /// what a registration checks every output its program reads against
    /// and compiles each stage for.
    pub fn seat_planes(
        &mut self,
        planes: Vec<(String, u32, u32, u32, bool, poem_ir::Dtype, bool)>,
    ) {
        self.stages.planes = planes;
    }

    /// Guest stages a fire compiled: a shape its program's registration
    /// did not, which is a defect (a registration is its program's ready).
    #[must_use]
    pub fn fire_compiles(&self) -> u64 {
        self.stages.fire_compiles
    }

    /// Registers a program; where every stage lowers (its device form),
    /// compiles every stage for every shape a pass can take on `device`
    /// (`Stages::compile_all`), so no pass compiles: a stage that does not
    /// compile refuses the registration.
    pub fn register(
        &mut self,
        registration: &ProgramRegistration,
        device: Option<&crate::device::Device>,
    ) -> Result<ProgramId, Refusal> {
        let pending = self.prepare(registration)?;
        let staged = match device {
            Some(device) => pending.compile(device),
            None => Ok(Vec::new()),
        };
        self.finish(pending, staged)
    }

    /// A registration's first step, on the engine's thread: the program
    /// interprets and every stage lowers (else refused by name), and the
    /// stage shapes its passes take are named; nothing compiles yet. Its
    /// compile ([`Pending::compile`]) runs on any thread, and
    /// [`Plane::finish`] makes it the program. Until then no instance of
    /// it binds.
    pub fn prepare(&mut self, registration: &ProgramRegistration) -> Result<Pending, Refusal> {
        let plan = eta_exec::adopt_launch_package(registration.launch.clone())
            .map_err(|error| Refusal::Program(error.to_string()))?;
        if !plan.executable {
            return Err(Refusal::Program(format!(
                "program 0x{:016x} calls a semantic boundary this backend does not implement: {}",
                registration.program_hash,
                plan.reject_reason.as_deref().unwrap_or("no reason given")
            )));
        }
        // A guest runs on the device or is refused here, by name. Nothing
        // of the plane's is touched until the finish, which assigns the id:
        // a prepare whose compile never finishes leaves nothing behind.
        let form = crate::guest::form_of(&plan.package).map_err(|why| {
            Refusal::Program(format!(
                "a stage of program 0x{:016x} does not lower on XLA: {why}",
                registration.program_hash
            ))
        })?;
        let wanted = self
            .stages
            .wanted(&plan.package, form.key, &form.pooled())
            .map_err(|why| {
                Refusal::Program(format!(
                    "a stage of program 0x{:016x} does not compile: {why}",
                    registration.program_hash
                ))
            })?;
        Ok(Pending {
            hash: registration.program_hash,
            form,
            plan: Arc::new(plan),
            wanted,
            pool_rows: self.stages.pool_rows(),
        })
    }

    /// A registration's last step, on the engine's thread: the stages its
    /// compile made held, and the program registered; a compile that
    /// refused refuses it.
    pub fn finish(
        &mut self,
        pending: Pending,
        staged: Result<Vec<Arc<crate::guest::run::Staged>>, String>,
    ) -> Result<ProgramId, Refusal> {
        let Pending {
            hash,
            form,
            plan,
            wanted,
            ..
        } = pending;
        let staged = staged.map_err(|why| {
            Refusal::Program(format!(
                "a stage of program 0x{hash:016x} does not compile: {why}"
            ))
        })?;
        self.next_program += 1;
        let id = ProgramId(self.next_program);
        let key = form.key;
        self.forms.hold(id, Ok(form));
        if !staged.is_empty() {
            self.stages.install(key, wanted, staged);
        }
        tracing::debug!(program = id.0, "guest program registered");
        self.programs.insert(id, plan);
        Ok(id)
    }

    /// The device cache keys of the compiled stages the stage cache evicted
    /// since the last call ([`crate::guest::run::Stages::yield_room`]).
    pub fn take_evicted(&mut self) -> Vec<[u8; 32]> {
        self.stages.take_evicted()
    }

    /// Releases `id`, which no instance binds (`Engine::close_program`):
    /// its id answers nothing after.
    pub fn close_program(&mut self, id: ProgramId) -> Result<(), Refusal> {
        if self.instances.values().any(|instance| instance.program == id) {
            return Err(Refusal::Program(format!(
                "program {id} still has an instance bound"
            )));
        }
        self.programs.remove(&id).ok_or(Refusal::Closed {
            what: "program",
            id: id.0,
        })?;
        self.forms.forget(id);
        Ok(())
    }

    #[must_use]
    pub fn package(
        &self,
        program: ProgramId,
    ) -> Option<&eta_compiler::codegen::launch::LaunchPackage> {
        self.programs.get(&program).map(|plan| &plan.package)
    }

    /// An instance's carried rings moved (`Engine::rebind_slots`): the
    /// runtime copied their cells into the fresh `Channel` slots `rings`
    /// names, so the instance reads and writes them there from now on.
    pub fn rebind(
        &mut self,
        id: InstanceId,
        rings: &[engine::program::RingSlots],
        cell_row: &dyn Fn(u32) -> Option<u32>,
    ) -> Result<(), Refusal> {
        let held = self.instances.get(&id).ok_or(Refusal::Closed {
            what: "instance",
            id: id.0,
        })?;
        let (cells, fed) = self.rows_of(held.program, &held.channels, rings, cell_row)?;
        if let Some(held) = self.instances.get_mut(&id) {
            held.cells = cells;
            held.fed = fed;
            held.own_rings = rings.to_vec();
        }
        Ok(())
    }

    /// A registered channel's ring moved to fresh `Channel` slots (a park
    /// and its return): every instance bound to it reads and writes its
    /// cells there from now on.
    pub fn rebind_channel(
        &mut self,
        id: ChannelId,
        slots: &[u32],
        cell_row: &dyn Fn(u32) -> Option<u32>,
    ) -> Result<(), Refusal> {
        let Some(held) = self.channel_slots.get_mut(&id) else {
            return Err(Refusal::Closed {
                what: "channel",
                id: id.0,
            });
        };
        *held = slots.to_vec();
        let bound: Vec<InstanceId> = (self.instances.iter())
            .filter(|(_, seat)| seat.channels.contains(&id))
            .map(|(&at, _)| at)
            .collect();
        for at in bound {
            let seat = &self.instances[&at];
            let (cells, fed) =
                self.rows_of(seat.program, &seat.channels, &seat.own_rings, cell_row)?;
            let seat = self.instances.get_mut(&at).expect("listed above");
            seat.cells = cells;
            seat.fed = fed;
        }
        Ok(())
    }

    /// The `Channel` slots a registered channel's ring takes in the pool:
    /// one ring of whole-slot cells, for a channel an instance may keep on
    /// the device (`guest::carried_channels`, `guest::fed_channels`: no host
    /// role, no extern). `None` for one the host alone reads or writes: the
    /// plane keeps its ring itself.
    #[must_use]
    pub fn ring_facts(registration: &ChannelRegistration) -> Option<engine::program::RingFacts> {
        if registration.host_role != eta_ir::container::HostRole::None
            || registration.extern_dir.is_some()
        {
            return None;
        }
        let dtype = eta_exec::concrete_dtype(registration.dtype);
        let words = crate::guest::lower::words(dtype, numel_of(&registration.shape));
        let bytes = words as u64 * 4;
        let capacity = registration.capacity.max(1);
        let slots = bytes.div_ceil(crate::guest::CHANNEL_SLOT).max(1) * u64::from(capacity);
        Some(engine::program::RingFacts {
            channel: 0,
            cell_bytes: bytes,
            capacity,
            footprint: engine::program::Footprint {
                bytes: slots * crate::guest::CHANNEL_SLOT,
                slots: u32::try_from(slots).unwrap_or(u32::MAX),
                contiguous: true,
            },
        })
    }

    /// An instance of `program` bound to `channels`: each carried and fed
    /// channel's pool rows, from the rings it holds itself (`own`) or, for
    /// a channel the binding names a registered one for, that channel's.
    fn rows_of(
        &self,
        program: ProgramId,
        channels: &[ChannelId],
        own: &[engine::program::RingSlots],
        cell_row: &dyn Fn(u32) -> Option<u32>,
    ) -> Result<(Cells, Fed), Refusal> {
        let Some(form) = self.forms.get(program) else {
            return Ok((Vec::new(), Vec::new()));
        };
        let mut rings = own.to_vec();
        for ring in form.rings() {
            if rings.iter().any(|r| r.channel == ring.channel) {
                continue;
            }
            let Some(id) = channels.get(ring.channel as usize) else {
                continue;
            };
            let Some(slots) = self.channel_slots.get(id) else {
                continue;
            };
            if (slots.len() as u64) < u64::from(ring.footprint.slots) {
                return Err(Refusal::Program(format!(
                    "program {program} keeps channel {} in {} Channel slot(s) and registered \
                     channel {id} holds {}",
                    ring.channel,
                    ring.footprint.slots,
                    slots.len()
                )));
            }
            rings.push(engine::program::RingSlots {
                channel: ring.channel,
                slots: slots.clone(),
            });
        }
        let cells = cells_of(&form.carried_words, &rings, cell_row)?;
        let fed = fed_of(&form.fed, &rings, cell_row)?;
        Ok((cells, fed))
    }

    /// Every other instance whose cells on the device are those of a ring
    /// one of `instances` shares (a registered channel two programs are
    /// bound to) puts them back in the ring first: the ring's front is then
    /// the channel's current cell, wherever the next pass reads it.
    fn flush_sharers(
        &mut self,
        instances: &[InstanceId],
        pool: Option<&crate::pjrt::Buffer>,
    ) -> Result<(), Refusal> {
        let mut touched = std::collections::HashSet::new();
        for id in instances {
            if let Some(seat) = self.instances.get(id) {
                touched.extend(seat.inst.channels.iter().map(|r| Arc::as_ptr(r) as usize));
            }
        }
        for (id, held) in &mut self.instances {
            if instances.contains(id) || !held.on_device || held.cells.is_empty() {
                continue;
            }
            let shares = (held.cells.iter()).any(|&(c, _)| {
                touched.contains(&(Arc::as_ptr(&held.inst.channels[c as usize]) as usize))
            });
            if shares {
                cells_to_host(held, pool).map_err(|why| {
                    held.inst.poisoned = true;
                    Refusal::Program(format!("instance {id}'s cells back to its rings: {why}"))
                })?;
            }
        }
        Ok(())
    }

    /// Before instance `id` closes: its cells on the device of a registered
    /// channel (one that outlives it) back in the channel's ring.
    pub fn flush_for_close(
        &mut self,
        id: InstanceId,
        pool: Option<&crate::pjrt::Buffer>,
    ) -> Result<(), Refusal> {
        self.flights.wait_all();
        let registered: std::collections::HashSet<usize> =
            self.channels.values().map(|r| Arc::as_ptr(r) as usize).collect();
        let Some(held) = self.instances.get_mut(&id) else {
            return Ok(());
        };
        let keeps = (held.cells.iter()).any(|&(c, _)| {
            registered.contains(&(Arc::as_ptr(&held.inst.channels[c as usize]) as usize))
        });
        if keeps {
            cells_to_host(held, pool).map_err(Refusal::Program)?;
        }
        Ok(())
    }

    /// What a pass of `program` takes on the device beyond the load's own,
    /// from its stages as registration compiled them; none for a program
    /// the host interprets.
    #[must_use]
    pub fn fire_facts(&self, program: ProgramId, load_temp: u64) -> engine::program::FireFacts {
        self.forms
            .get(program)
            .map(|form| self.stages.fire_facts(form.key, &form.pooled(), load_temp))
            .unwrap_or_default()
    }

    /// Program `program`'s device form, where its every stage lowers.
    #[must_use]
    pub fn compiled_form(&self, program: ProgramId) -> Option<&Arc<crate::guest::Compiled>> {
        self.forms.get(program)
    }

        pub fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> Result<RegisteredChannel, Refusal> {
        if self.channels.contains_key(&registration.id) {
            return Err(Refusal::Program(format!(
                "channel {} is already registered",
                registration.id
            )));
        }
        let numel = numel_of(&registration.shape);
        let ring = Arc::new(ChannelState::host(
            eta_exec::concrete_dtype(registration.dtype),
            numel,
            registration.capacity.max(1) as usize,
        ));
        let mirror = (registration.host_role != eta_ir::container::HostRole::None
            && registration.capacity > 0)
            .then(|| HostMirror {
                mirror: ring.cells_ptr() as u64,
                words: ring.words_ptr() as u64,
                cell_bytes: u32::try_from(ring.cell_bytes()).unwrap_or(u32::MAX),
                capacity: registration.capacity,
            });
        self.channels.insert(registration.id, ring);
        if !registration.slots.is_empty() {
            self.channel_slots
                .insert(registration.id, registration.slots.clone());
        }
        Ok(RegisteredChannel {
            id: registration.id,

            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror,
        })
    }

    pub fn close_channel(&mut self, id: ChannelId) -> Result<(), Refusal> {
        self.channel_slots.remove(&id);
        self.channels
            .remove(&id)
            .map(|_| ())
            .ok_or(Refusal::Closed {
                what: "channel",
                id: id.0,
            })
    }

    /// Binds an instance; `cell_row` answers the pool row a `Channel` slot
    /// ref starts at, for the carried rings it keeps on the device.
    pub fn bind(
        &mut self,
        binding: &InstanceBinding,
        cell_row: &dyn Fn(u32) -> Option<u32>,
    ) -> Result<BoundInstance, Refusal> {
        let plan = self
            .programs
            .get(&binding.program)
            .cloned()
            .ok_or(Refusal::Closed {
                what: "program",
                id: binding.program.0,
            })?;
        let declared = plan.package.channels.len();
        if binding.channels.len() != declared {
            return Err(Refusal::Program(format!(
                "program {} declares {declared} channel(s); the bind names {}",
                binding.program,
                binding.channels.len()
            )));
        }
        let mut externs = BTreeMap::new();
        for (dense, id) in binding.channels.iter().enumerate() {
            if let Some(ring) = self.channels.get(id) {
                externs.insert(dense as u32, Arc::clone(ring));
            }
        }
        let mut seeds = BTreeMap::new();
        for seed in &binding.seeds {
            let decl = plan
                .package
                .channels
                .get(seed.channel as usize)
                .ok_or_else(|| {
                    Refusal::Program(format!(
                        "seed names channel {}, past the {declared} declared",
                        seed.channel
                    ))
                })?;
            let value = decode(&seed.bytes, decl.dtype, &decl.shape).ok_or_else(|| {
                Refusal::Program(format!(
                    "seed for channel {} is {} byte(s), not one cell of {:?}{:?}",
                    seed.channel,
                    seed.bytes.len(),
                    decl.dtype,
                    decl.shape
                ))
            })?;
            seeds.insert(seed.channel, value);
        }
        let inst = eta_exec::make_host_instance(&plan, &externs, &seeds);
        let (cells, fed) =
            self.rows_of(binding.program, &binding.channels, &binding.ring_slots, cell_row)?;

        self.next_instance += 1;
        let id = InstanceId(self.next_instance);
        let poisoned = Arc::new(AtomicBool::new(false));
        self.instances.insert(
            id,
            Instance {
                program: binding.program,
                plan,
                inst,
                geometry: binding.geometry,
                channels: binding.channels.clone(),
                own_rings: binding.ring_slots.clone(),
                poisoned,
                cells,
                on_device: false,
                fed,
            },
        );
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    #[must_use]
    pub fn device_form(&self, program: ProgramId) -> (bool, Option<String>) {
        (
            self.forms.get(program).is_some(),
            self.forms.refusal(program).map(ToString::to_string),
        )
    }

    pub fn close_instance(&mut self, id: InstanceId) -> Result<(), Refusal> {
        self.instances
            .remove(&id)
            .map(|_| ())
            .ok_or(Refusal::Closed {
                what: "instance",
                id: id.0,
            })
    }

    pub fn envelope_fold(&self, instance: InstanceId) -> Result<Option<u32>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let Some(binding) = seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::Fold && !binding.is_const)
        else {
            return Ok(None);
        };
        let ring = &seat.inst.channels[binding.channel as usize];
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `fold` ring is empty at the fire"
            )));
        }
        let len = match ring.front() {
            Value::U32(cells) => cells.first().copied(),
            Value::I32(cells) => cells.first().map(|&n| n.max(0) as u32),
            other => {
                return Err(Refusal::Program(format!(
                    "instance {instance}: `fold` holds {:?}, not a count",
                    other.dtype()
                )));
            }
        };
        len.map(Some).ok_or_else(|| {
            Refusal::Program(format!("instance {instance}: `fold` holds an empty cell"))
        })
    }

    pub fn envelope_tokens(
        &self,
        instance: InstanceId,
        rows: usize,
        lane_at: usize,
    ) -> Result<Option<Tokens>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        if !seat.geometry.ports().contains(Port::EmbedTokens) {
            return Ok(None);
        }
        let binding = seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::EmbedTokens && !binding.is_const)
            .ok_or_else(|| {
                Refusal::Program(format!(
                    "instance {instance} is bound in {:?} and its program binds no                      `embed_tokens` channel",
                    seat.geometry
                ))
            })?;
        let ring = &seat.inst.channels[binding.channel as usize];
        // The cell this fire reads is the last pass's, still on its way
        // to the host: the pass put it in the fed ring's slot too, where the
        // fire gathers it, with no trip to the host.
        if ring.is_empty()
            && let Some(cell) = self.flights.cell_at(ring, ring.head())
            && let crate::guest::lower::Feed::Value(
                eta_ir::types::Dtype::I32 | eta_ir::types::Dtype::U32,
                n,
            ) = cell.slot.feed
            && let Some(&(_, first, capacity, cell_rows)) =
                seat.fed.iter().find(|f| f.0 == binding.channel)
        {
            let span = self.lane_span(seat, instance, "embed_tokens", n, rows, lane_at)?;
            let at = u32::try_from(ring.head() % u64::from(capacity.max(1))).unwrap_or(0);
            let row = first + at * cell_rows;
            return Ok(Some(Tokens::Device {
                at: span.map(|i| (row + i as u32 / 128, i as u32 % 128)).collect(),
            }));
        }
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `embed_tokens` ring is empty at the fire"
            )));
        }
        let ids: Vec<u32> = match ring.front() {
            Value::U32(ids) => ids,
            Value::I32(ids) => ids.into_iter().map(|id| id as u32).collect(),
            other => {
                return Err(Refusal::Program(format!(
                    "instance {instance}: `embed_tokens` holds {:?}, not an index",
                    other.dtype()
                )));
            }
        };
        let span = self.lane_span(seat, instance, "embed_tokens", ids.len(), rows, lane_at)?;
        Ok(Some(Tokens::Host(ids[span].to_vec())))
    }

    pub fn lane_count(&self, instance: InstanceId) -> Result<usize, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        Ok(self.embed_indptr(seat).len().saturating_sub(1).max(1))
    }

    fn embed_indptr(&self, seat: &Instance) -> Vec<u32> {
        match seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::EmbedIndptr)
        {
            Some(binding) if binding.is_const => seat
                .plan
                .const_ports
                .iter()
                .find(|c| c.port == Port::EmbedIndptr)
                .map(|c| match &c.value {
                    Value::U32(v) => v.clone(),
                    Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
                    _ => Vec::new(),
                })
                .unwrap_or_default(),
            Some(binding) => {
                let ring = &seat.inst.channels[binding.channel as usize];
                self.settle_ring(ring);
                match ring.front() {
                    Value::U32(v) => v,
                    Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
                    _ => Vec::new(),
                }
            }
            None => Vec::new(),
        }
    }

    fn lane_span(
        &self,
        seat: &Instance,
        instance: InstanceId,
        what: &str,
        cells: usize,
        rows: usize,
        lane_at: usize,
    ) -> Result<std::ops::Range<usize>, Refusal> {
        if lane_at == 0 && cells == rows {
            return Ok(0..rows);
        }
        let indptr = self.embed_indptr(seat);
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

    pub fn publish(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> Result<bool, Refusal> {
        let flights = Arc::clone(&self.flights);
        let seat = self.instance_mut(instance)?;
        let decl = seat
            .plan
            .package
            .channels
            .get(channel as usize)
            .ok_or_else(|| {
                Refusal::Program(format!("instance {instance} carries no channel {channel}"))
            })?;
        let value = decode(cell, decl.dtype, &decl.shape).ok_or_else(|| {
            Refusal::Program(format!(
                "channel {channel}: a {}-byte cell into a ring of {:?}{:?}",
                cell.len(),
                decl.dtype,
                decl.shape
            ))
        })?;
        let plan = Arc::clone(&seat.plan);
        if let Some(ring) = seat.inst.channels.get(channel as usize) {
            flights.wait_ring(ring);
        }
        match eta_exec::host_put(seat.host(), &plan, channel, &value) {
            HostOp::Ok => Ok(true),
            HostOp::WouldBlock => Ok(false),
            HostOp::Poisoned => Err(Refusal::Program(format!("instance {instance} is poisoned"))),
            HostOp::WrongRole => Err(Refusal::Program(format!(
                "channel {channel} of instance {instance} is not host-written"
            ))),
            HostOp::TypeMismatch => Err(Refusal::Program(format!(
                "channel {channel} of instance {instance}: cell dtype or shape mismatch"
            ))),
        }
    }

    pub fn take(&mut self, instance: InstanceId, channel: u32) -> Result<Option<Vec<u8>>, Refusal> {
        let flights = Arc::clone(&self.flights);
        let seat = self.instance_mut(instance)?;
        if let Some(ring) = seat.inst.channels.get(channel as usize) {
            flights.wait_ring(ring);
        }
        if channel as usize >= seat.plan.package.channels.len() {
            return Err(Refusal::Program(format!(
                "instance {instance} carries no channel {channel}"
            )));
        }
        let plan = Arc::clone(&seat.plan);
        match eta_exec::host_take(seat.host(), &plan, channel) {
            (HostOp::Ok, Some(value)) => {
                let mut bytes = vec![0u8; eta_exec::wire_cell_bytes(value.dtype(), value.len())];
                eta_exec::encode_wire(&value, &mut bytes);
                Ok(Some(bytes))
            }
            (HostOp::Ok, None) | (HostOp::WouldBlock, _) => Ok(None),
            (HostOp::Poisoned, _) => {
                Err(Refusal::Program(format!("instance {instance} is poisoned")))
            }
            (HostOp::WrongRole, _) => Err(Refusal::Program(format!(
                "channel {channel} of instance {instance} is not host-read"
            ))),
            (HostOp::TypeMismatch, _) => Err(Refusal::Program(format!(
                "channel {channel} of instance {instance}: cell dtype mismatch"
            ))),
        }
    }

    pub fn envelope_mask(
        &self,
        instance: InstanceId,
        rows: usize,
        lane_at: usize,
    ) -> Result<Option<engine::fire::Masking>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        if !seat.geometry.ports().contains(Port::EmbedTokens) {
            return Ok(None);
        }
        let Some(binding) = seat
            .plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == Port::AttnMask && !binding.is_const)
        else {
            return Ok(None);
        };
        let ring = &seat.inst.channels[binding.channel as usize];
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `attn_mask` ring is empty at the fire"
            )));
        }
        let Value::Bool(cells) = ring.front() else {
            return Err(Refusal::Program(format!(
                "instance {instance}: `attn_mask` holds {:?}, not bools",
                ring.front().dtype()
            )));
        };
        let keys = seat
            .plan
            .package
            .channels
            .get(binding.channel as usize)
            .and_then(|decl| decl.shape.last().copied())
            .map_or(0, |keys| keys as usize);
        if keys == 0 || cells.len() % keys != 0 {
            return Err(Refusal::Program(format!(
                "instance {instance}: `attn_mask` holds {} cell(s) over {keys} key(s)",
                cells.len()
            )));
        }
        let span = self.lane_span(
            seat,
            instance,
            "attn_mask",
            cells.len() / keys,
            rows,
            lane_at,
        )?;
        let masks: Vec<engine::fire::Mask> = cells[span.start * keys..span.end * keys]
            .chunks_exact(keys)
            .map(|row| engine::fire::Mask::new(runs_of(row), keys as u64))
            .collect();
        Ok(Some(match masks.as_slice() {
            [only] => engine::fire::Masking::Extent(only.clone()),
            _ => engine::fire::Masking::Rows(masks),
        }))
    }

    /// Runs the guest passes of a fire's `attachments` (instance, first
    /// lane, lanes): on the device, reading `kept` where it lies, for every
    /// instance whose program lowered and whose outputs bind there —
    /// instances of one program as one batched executable per stage — and
    /// in the interpreter, over the rows downloaded from `kept`, for the
    /// rest. Returns, per attachment, whether it ran on the device.
    pub fn fire_guests(
        &mut self,
        device: &crate::device::Device,
        attachments: &[(InstanceId, u32, u32)],
        kept: Option<&crate::readout::Kept>,
        pool: &mut Option<crate::pjrt::Buffer>,
        at: Stage,
    ) -> Result<Vec<bool>, Refusal> {
        let (ran, faults) = self.fire_guests_with(device, attachments, kept, pool, None, at)?;
        match faults.into_iter().next() {
            Some((_, why)) => Err(Refusal::Program(why)),
            None => Ok(ran),
        }
    }

    /// `fire_guests`, with each device group's last stage left in flight
    /// when `latch` is given: its puts land when their download does (and
    /// release a hold on `latch` then), and the next fire's `embed_tokens`
    /// can read them where they are. Every pass still in flight lands
    /// before this one binds its roots. Answers which attachments ran on
    /// the device and each instance whose pass faulted, with why: its
    /// lanes fault, the others' passes stand. A pass that lands later
    /// places its fault on `latch`.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn fire_guests_with(
        &mut self,
        device: &crate::device::Device,
        attachments: &[(InstanceId, u32, u32)],
        kept: Option<&crate::readout::Kept>,
        pool: &mut Option<crate::pjrt::Buffer>,
        latch: Option<&Arc<Latch>>,
        point: Stage,
    ) -> Result<(Vec<bool>, Vec<(InstanceId, String)>), Refusal> {
        use crate::guest::run::{Member, OnDevice, bound, step_group, step_group_deferred};
        self.flights.wait_all();
        let members: Vec<InstanceId> = attachments.iter().map(|&(id, _, _)| id).collect();
        self.flush_sharers(&members, pool.as_ref())?;
        for &(instance, _, _) in attachments {
            if let Some(seat) = self.instances.get_mut(&instance)
                && seat.poisoned.load(Ordering::Acquire)
            {
                seat.inst.poisoned = true;
            }
        }
        // Where each attachment runs.
        let mut plans: Vec<Option<Placed>> = Vec::with_capacity(attachments.len());
        // Per attachment, why its outputs were refused a binding.
        let mut refused: Vec<Option<String>> = vec![None; attachments.len()];
        let mut carried_of: BTreeMap<[u8; 32], Vec<u32>> = BTreeMap::new();
        for (at, &(instance, lane, lanes)) in attachments.iter().enumerate() {
            let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
                what: "instance",
                id: instance.0,
            })?;
            let form = self.forms.get(seat.program).map(|c| c.key);
            if let Some(c) = self.forms.get(seat.program) {
                carried_of.insert(c.key, c.pooled());
            }
            let bound = match bound(&seat.plan, kept, lane, lanes) {
                Ok(bound) => bound,
                Err(why) => {
                    refused[at] = Some(format!("instance {instance} at lane {lane}: {why}"));
                    None
                }
            };
            plans.push(match (form, bound) {
                (Some(key), Some(bound)) => Some((Arc::clone(&seat.plan), key, bound)),
                _ => None,
            });
        }
        // Instances that share a ring step in attachment order, one by one.
        let mut rings = std::collections::HashSet::new();
        let mut shared = false;
        for &(instance, _, _) in attachments {
            if let Some(seat) = self.instances.get(&instance) {
                for ring in &seat.inst.channels {
                    shared |= !rings.insert(Arc::as_ptr(ring) as usize);
                }
            }
        }
        let mut faults: Vec<(InstanceId, String)> = Vec::new();
        let mut note = |outcome: StepOutcome, instance: InstanceId, program: ProgramId| {
            let why = match outcome {
                StepOutcome::Committed => return,
                StepOutcome::Blocked(channel) => {
                    format!("instance {instance} (program {program}) blocked on channel {channel}")
                }
                StepOutcome::Faulted(why) => {
                    format!("instance {instance} (program {program}) faulted: {why}")
                }
            };
            faults.push((instance, why));
        };
        let mut ran = vec![false; attachments.len()];
        if shared {
            // These step one by one on the host's rings: their cells come
            // back from the pool first.
            for &(instance, _, _) in attachments {
                if let Some(held) = self.instances.get_mut(&instance)
                    && let Err(why) = cells_to_host(held, pool.as_ref())
                {
                    held.inst.poisoned = true;
                    note(StepOutcome::Faulted(why), instance, held.program);
                }
            }
            for (at, &(instance, _, _)) in attachments.iter().enumerate() {
                let Some((plan, key, bound)) = plans[at].clone() else {
                    continue;
                };
                let Some(held) = self.instances.get_mut(&instance) else {
                    continue;
                };
                let mut runner = OnDevice::new(device, &mut self.stages, key, kept, bound);
                let outcome = eta_exec::step_with(
                    &mut held.inst,
                    &plan,
                    &PassInputs::none(),
                    &mut runner,
                    point,
                );
                ran[at] = true;
                note(outcome, instance, held.program);
            }
        } else {
            // Group by program content and the planes read, in order of
            // first appearance.
            type Planes = Vec<(String, u32, usize)>;
            type Group = ([u8; 32], Planes, Arc<ExecPlan>, Vec<usize>);
            let planes_of = |bound: &[crate::guest::run::Bound]| -> Planes {
                (bound.iter())
                    .map(|b| (b.name.clone(), b.lane, b.plane))
                    .collect()
            };
            let mut groups: Vec<Group> = Vec::new();
            for (at, planned) in plans.iter().enumerate() {
                let Some((plan, key, bound)) = planned else {
                    continue;
                };
                let planes = planes_of(bound);
                match (groups.iter_mut()).find(|(k, p, _, _)| k == key && *p == planes) {
                    Some((_, _, _, members)) => members.push(at),
                    None => groups.push((*key, planes, Arc::clone(plan), vec![at])),
                }
            }
            for (key, _, plan, members) in groups {
                let wanted: BTreeMap<InstanceId, usize> =
                    members.iter().map(|&at| (attachments[at].0, at)).collect();
                let carried = carried_of.get(&key).cloned().unwrap_or_default();
                // Cells the rings hold (a fresh bind, a pass on the host) land
                // in the members' slots before the pass reads them there.
                {
                    let fresh: Vec<(&InterpInstance, &[(u32, u32)])> = (self.instances.iter())
                        .filter(|(id, held)| {
                            wanted.contains_key(id) && !held.on_device && !held.cells.is_empty()
                        })
                        .map(|(_, held)| (&held.inst, held.cells.as_slice()))
                        .collect();
                    if !fresh.is_empty() {
                        crate::guest::run::write_cells(
                            device,
                            &plan,
                            pool,
                            self.stages.pool_rows,
                            self.stages.sink_row,
                            &fresh,
                        )
                        .map_err(Refusal::Program)?;
                    }
                    for (id, held) in &mut self.instances {
                        if wanted.contains_key(id) {
                            held.on_device = true;
                        }
                    }
                }
                let mut order: Vec<(usize, InstanceId, ProgramId)> = Vec::new();
                let mut group: Vec<Member<'_>> = Vec::new();
                for (&id, held) in &mut self.instances {
                    let Some(&at) = wanted.get(&id) else {
                        continue;
                    };
                    order.push((at, id, held.program));
                    let cells = held.rows();
                    group.push(Member {
                        inst: &mut held.inst,
                        bound: plans[at].as_ref().map(|p| p.2.clone()).unwrap_or_default(),
                        cells,
                    });
                }
                let (outcomes, deferred) = if latch.is_some() {
                    step_group_deferred(
                        device,
                        &mut self.stages,
                        key,
                        &plan,
                        kept,
                        &mut group,
                        &carried,
                        pool,
                        point,
                    )
                } else {
                    let outcomes = step_group(
                        device,
                        &mut self.stages,
                        key,
                        &plan,
                        kept,
                        &mut group,
                        &carried,
                        pool,
                        point,
                    );
                    (outcomes, None)
                };
                if let Some(deferred) = deferred {
                    let instances: Vec<InstanceId> = deferred
                        .members
                        .iter()
                        .filter_map(|&m| order.get(m))
                        .map(|(_, id, _)| *id)
                        .collect();
                    let poison: Vec<Arc<AtomicBool>> = (instances.iter())
                        .filter_map(|id| self.instances.get(id))
                        .map(|seat| Arc::clone(&seat.poisoned))
                        .collect();
                    self.flights.launch(deferred, poison, instances, latch);
                }
                for ((at, id, program), outcome) in order.into_iter().zip(outcomes) {
                    ran[at] = true;
                    note(outcome, id, program);
                }
            }
        }
        let on_device = ran.iter().filter(|&&r| r).count() as u64;
        if on_device > 0 && self.tally.device.fetch_add(on_device, Ordering::AcqRel) == 0 {
            tracing::info!(passes = on_device, "a guest pass ran on the device");
        }
        // Registration refused every program whose outputs a fire does not
        // return on the device (`Stages::compile_all`), and every program
        // with a stage that does not lower: an attachment that did not run
        // is a fault, not a pass for another path.
        for (at, &(instance, lane, _)) in attachments.iter().enumerate() {
            if let Some(why) = refused[at].take() {
                faults.push((instance, why));
            } else if !ran[at] {
                faults.push((
                    instance,
                    format!(
                        "instance {instance} at lane {lane}: its outputs are not on the device in \
                         this fire, which its registration checked they would be"
                    ),
                ));
            }
        }
        Ok((ran, faults))
    }

    #[must_use]
    pub fn prologue_runs(&self, instance: InstanceId) -> bool {
        self.instances
            .get(&instance)
            .is_some_and(|seat| seat.plan.prologue_runs())
    }

    /// Waits for the in-flight cell `ring`'s next read would find, when
    /// the ring holds nothing committed.
    fn settle_ring(&self, ring: &Arc<ChannelState>) {
        if ring.is_empty() {
            self.flights.wait_ring(ring);
        }
    }

    /// Blocks until every guest pass in flight has landed its cells.
    pub fn settle(&self) {
        self.flights.wait_all();
    }

    /// Whether a guest pass is still in flight.
    #[must_use]
    pub fn airborne(&self) -> bool {
        self.flights.airborne()
    }

    #[must_use]
    pub fn tally(&self) -> u64 {
        self.tally.device.load(Ordering::Acquire)
    }

    fn instance_mut(&mut self, id: InstanceId) -> Result<&mut Instance, Refusal> {
        self.instances.get_mut(&id).ok_or(Refusal::Closed {
            what: "instance",
            id: id.0,
        })
    }
}

fn numel_of(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product::<usize>().max(1)
}

fn decode(bytes: &[u8], dtype: eta_ir::container::ChanDType, shape: &[u32]) -> Option<Value> {
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
        Dtype::Bool => Value::Bool((0..numel).map(|j| (bytes[j / 8] >> (j % 8)) & 1 != 0).collect()),
        Dtype::I32 => Value::I32(words().map(i32::from_le_bytes).collect()),
        Dtype::U32 => Value::U32(words().map(u32::from_le_bytes).collect()),
        Dtype::F32 => Value::F32(words().map(f32::from_le_bytes).collect()),
        _ => return None,
    })
}

fn runs_of(row: &[bool]) -> Vec<u32> {
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

// The plane drops before the shell that owns the device client: a pass
// still in flight lands first, and its buffers go with the client up.
impl Drop for Plane {
    fn drop(&mut self) {
        self.flights.drain();
    }
}

impl Plane {
    /// The committed front cell of channel `channel` (a registered channel
    /// id) of instance `instance`: the cell its own `take` would read this
    /// fire. The ring keeps it.
    pub fn feed_cell(&self, instance: InstanceId, channel: ChannelId) -> Result<Value, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        let dense = seat
            .channels
            .iter()
            .position(|&bound| bound == channel)
            .ok_or_else(|| {
                Refusal::Program(format!(
                    "instance {instance} binds no channel {channel}, so no input reads a cell of it"
                ))
            })?;
        let ring = &seat.inst.channels[dense];
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "input feed channel {channel} of instance {instance} holds no committed \
                 cell; an input is fed from the cell the instance's own `take` would read this \
                 fire, so publish (or `put`) one before submitting"
            )));
        }
        Ok(ring.front())
    }
}

// Device-resolved geometry (engine-cuda `program/ports.rs::resolve`): the
// descriptor ports an instance bound in `DeviceGeometry` (or the decode
// envelope) publishes, read off the host rings' committed fronts.
impl Plane {
    /// The geometry class instance `instance` was bound in.
    #[must_use]
    pub fn geometry_of(&self, instance: InstanceId) -> Option<GeometryClass> {
        self.instances.get(&instance).map(|seat| seat.geometry)
    }

    /// The instance's descriptor ports as this fire reads them; `None` for
    /// an instance bound in the host class, whose geometry the runtime
    /// resolved.
    pub fn device_envelope(
        &self,
        instance: InstanceId,
    ) -> Result<Option<engine::ports::Envelope>, Refusal> {
        let seat = self.instances.get(&instance).ok_or(Refusal::Closed {
            what: "instance",
            id: instance.0,
        })?;
        if seat.geometry == GeometryClass::Host {
            return Ok(None);
        }
        let mut out = engine::ports::Envelope::default();
        for binding in &seat.plan.package.ports {
            if !engine::ports::resolves(seat.geometry, binding.port) {
                continue;
            }
            if binding.port == Port::AttnMask {
                if binding.is_const {
                    continue;
                }
                let value = self.port_front(seat, instance, binding.port, binding.channel)?;
                let Value::Bool(cells) = value else {
                    return Err(Refusal::Program(format!(
                        "port {}'s channel {} holds {:?}, and an attention mask is a \
                         rectangle of KEPT/DROPPED and not a rectangle of numbers",
                        binding.port.name(),
                        binding.channel,
                        value.dtype()
                    )));
                };
                out.mask = Some(cells);
                continue;
            }
            let slot = match binding.port {
                Port::EmbedIndptr => &mut out.qo_indptr,
                Port::EmbedTokens => &mut out.tokens,
                Port::Positions => &mut out.positions,
                Port::Slots => &mut out.slots,
                Port::KvLen => &mut out.kv_len,
                Port::Fold => &mut out.fold,
                _ => continue,
            };
            let value = if binding.is_const {
                seat.plan
                    .const_ports
                    .iter()
                    .find(|folded| folded.port == binding.port)
                    .map(|folded| folded.value.clone())
                    .ok_or_else(|| {
                        Refusal::Program(format!(
                            "port {} is declared const and no folded value was kept for it",
                            binding.port.name()
                        ))
                    })?
            } else {
                self.port_front(seat, instance, binding.port, binding.channel)?
            };
            *slot = Some(match value {
                Value::U32(words) => words,
                Value::I32(words) => words.into_iter().map(|word| word as u32).collect(),
                other => {
                    return Err(Refusal::Program(format!(
                        "port {}'s channel {} holds {:?}, and a geometry index is not an \
                         activation",
                        binding.port.name(),
                        binding.channel,
                        other.dtype()
                    )));
                }
            });
        }
        Ok(Some(out))
    }

    fn port_front(
        &self,
        seat: &Instance,
        instance: InstanceId,
        port: Port,
        channel: u32,
    ) -> Result<Value, Refusal> {
        let ring = seat.inst.channels.get(channel as usize).ok_or_else(|| {
            Refusal::Program(format!(
                "port {} names channel {channel}, which instance {instance} does not carry",
                port.name()
            ))
        })?;
        self.settle_ring(ring);
        if ring.is_empty() {
            return Err(Refusal::Program(format!(
                "instance {instance}: the `{}` ring is empty at the fire",
                port.name()
            )));
        }
        Ok(ring.front())
    }
}

/// Each carried channel's cell: its channel and the pool row it lies at.
type Cells = Vec<(u32, u32)>;
/// Each fed channel's ring: its channel, first pool row, capacity and a
/// cell's rows.
type Fed = Vec<(u32, u32, u32, u32)>;

/// Each carried channel's first pool row, from the `Channel` slots the
/// binding names for it (`ProgramFacts::rings`, one contiguous run each).
fn cells_of(
    carried: &[(u32, u32)],
    rings: &[engine::program::RingSlots],
    cell_row: &dyn Fn(u32) -> Option<u32>,
) -> Result<Vec<(u32, u32)>, Refusal> {
    carried
        .iter()
        .map(|&(channel, _)| {
            let first = rings
                .iter()
                .find(|r| r.channel == channel)
                .and_then(|r| r.slots.first().copied())
                .ok_or_else(|| {
                    Refusal::Program(format!(
                        "the program keeps channel {channel} in Channel slots and the binding \
                         names none for it"
                    ))
                })?;
            let row = cell_row(first).ok_or_else(|| {
                Refusal::Program(format!(
                    "Channel slot {first} of carried channel {channel} lies past the pool"
                ))
            })?;
            Ok((channel, row))
        })
        .collect()
}

/// An instance's carried cells back into its rings' fronts, where the pool
/// held them: the host runs its next pass.
fn cells_to_host(held: &mut Instance, pool: Option<&crate::pjrt::Buffer>) -> Result<(), String> {
    if !held.on_device || held.cells.is_empty() {
        return Ok(());
    }
    let pool = pool.ok_or("the pool, which another program holds")?;
    crate::guest::run::read_cells(&held.plan, pool, &held.cells, &held.inst)?;
    held.on_device = false;
    Ok(())
}

/// Each fed channel's ring: its first pool row (the binding's slots for it,
/// one contiguous run), its capacity and a cell's rows (whole slots).
fn fed_of(
    fed: &[(u32, u32, u32)],
    rings: &[engine::program::RingSlots],
    cell_row: &dyn Fn(u32) -> Option<u32>,
) -> Result<Vec<(u32, u32, u32, u32)>, Refusal> {
    let words: Vec<(u32, u32)> = fed.iter().map(|&(c, w, _)| (c, w)).collect();
    let firsts = cells_of(&words, rings, cell_row)?;
    Ok(fed
        .iter()
        .zip(firsts)
        .map(|(&(channel, w, capacity), (_, first))| {
            let slots = (u64::from(w) * 4).div_ceil(crate::guest::CHANNEL_SLOT);
            (channel, first, capacity, u32::try_from(slots * 2).unwrap_or(u32::MAX))
        })
        .collect())
}

#[cfg(test)]
mod latch_tests {
    use super::Latch;

    /// A fault a device event reports after the submitter let go is the
    /// step's, waited for (`PIE_XLA_SYNC`'s settle).
    #[test]
    fn a_fault_released_late_is_the_steps() {
        let at = engine::StepDone {
            frame: engine::fire::FrameId(1),
            step: 0,
        };
        let latch = Latch::new(at, 1, None);
        latch.hold();
        latch.release(Ok(()));
        let late = std::sync::Arc::clone(&latch);
        let event = std::thread::spawn(move || late.release(Err("the fire failed".into())));
        assert_eq!(latch.wait().as_deref(), Some("the fire failed"));
        event.join().expect("the event thread");
        let clean = Latch::new(at, 1, None);
        clean.release(Ok(()));
        assert_eq!(clean.wait(), None);
    }

    /// A settle waiting on the latch wakes only after the sink heard the
    /// step, its fold reports with it: what submit returns after under
    /// `PIE_XLA_SYNC` the sink already knows.
    #[test]
    fn the_sink_hears_the_step_before_a_settle_wakes() {
        use std::sync::{Arc, Mutex};
        let at = engine::StepDone {
            frame: engine::fire::FrameId(2),
            step: 0,
        };
        let heard: Arc<Mutex<Vec<engine::StepOutcome>>> = Arc::default();
        let into = Arc::clone(&heard);
        let sink: engine::CompletionSink = Arc::new(move |_, outcome| {
            // Slow on purpose: a settle woken first would pass the check
            // below before this lands.
            std::thread::sleep(std::time::Duration::from_millis(50));
            into.lock().expect("heard lock").push(outcome);
        });
        let latch = Latch::new(at, 1, Some(sink));
        latch.hold();
        latch.report(vec![engine::KvWrites::default()]);
        latch.release(Ok(()));
        let late = Arc::clone(&latch);
        let event = std::thread::spawn(move || late.release(Ok(())));
        assert_eq!(latch.wait(), None);
        let heard = heard.lock().expect("heard lock");
        assert_eq!(heard.len(), 1, "the sink heard the step before the settle woke");
        assert!(
            !heard[0].wrote.is_empty() && heard[0].faulted.is_empty(),
            "with its fold reports: {:?}",
            heard[0]
        );
        drop(heard);
        event.join().expect("the event thread");
    }
}

/// A program a registration prepared ([`Plane::prepare`]): what its
/// compile, on any thread, compiles, and what [`Plane::finish`] holds.
pub struct Pending {
    hash: u64,
    /// Its form: every stage lowers.
    form: Arc<crate::guest::Compiled>,
    plan: Arc<ExecPlan>,
    wanted: Vec<crate::guest::run::Want>,
    pool_rows: i64,
}

impl Pending {
    /// Every stage shape the program's passes take, compiled on `device`
    /// (on whatever thread calls this).
    pub fn compile(
        &self,
        device: &crate::device::Device,
    ) -> Result<Vec<Arc<crate::guest::run::Staged>>, String> {
        if self.wanted.is_empty() {
            return Ok(Vec::new());
        }
        crate::guest::run::compile_wanted(device, &self.plan.package, &self.wanted, self.pool_rows)
    }
}
