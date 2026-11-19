use engine::ports::Envelope;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_exec::{ExecPlan, Extents};
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::name_or_unknown;
use eta_ir::validate::Direction;

use crate::device::{Buffer, Context, Pipelines};
use crate::error::{Fault, Result};
use crate::store::Pools;
use crate::store::layout::INSTANCE;

use super::compile::Compiled;
use super::held::{self, TableLayout, Tables};
use super::launch::{ChannelShape, Cursor, Prepared, Rings};
use super::ports;
use super::shared::SharedRing;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fired {
    Committed,
    Blocked(u32),
    Declined,
    Faulted(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocked {
    pub channel: u32,
    pub needs: Option<Direction>,
    pub live: u64,
    pub capacity: u64,
}

impl std::fmt::Display for Blocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.needs {
            Some(Direction::NeedsFull) => write!(
                f,
                "channel {} is empty and its program takes from it (needs a cell, \
                 holds {} of {})",
                self.channel, self.live, self.capacity
            ),
            Some(Direction::NeedsEmpty) => write!(
                f,
                "channel {} is full and its program puts into it (needs room, holds \
                 {} of {})",
                self.channel, self.live, self.capacity
            ),
            None => write!(
                f,
                "channel {} holds {} of {}",
                self.channel, self.live, self.capacity
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Launched {
    Airborne,
    Refused(Fired),
}

/// Whether stage `index` of `compiled` launches kernels (and so holds
/// launch tables).
pub(crate) fn launches(compiled: &Compiled, index: usize) -> bool {
    compiled
        .stages
        .get(index)
        .is_some_and(|stage| !stage.regions.is_empty())
}

/// The pool slots the runtime took for an instance (REDESIGN step 4).
#[derive(Clone, Copy)]
pub struct Pooled<'a> {
    pub pools: &'a Pools,
    /// `Instance` slot refs for its bind-time tables.
    pub instance: &'a [u32],
    /// `Channel` slot refs for the rings it holds itself.
    pub rings: &'a [engine::program::RingSlots],
}

/// The alignment an instance's score rows start at: a binding's, as the
/// rows of a fire's guest region were.
pub const CAPTURE_ALIGN: u64 = 256;

/// The bytes from `held`'s start to its first [`CAPTURE_ALIGN`]ed GPU
/// address (none off the device).
fn capture_pad(held: &held::Held) -> u64 {
    held.address_at(0)
        .map_or(0, |first| first.next_multiple_of(CAPTURE_ALIGN) - first)
}

#[derive(Debug)]
pub struct Session {
    /// The `Instance` slot refs the tables lie in, in order.
    instance_slots: Vec<u32>,
    /// Whether they are one run (a table larger than a slot).
    tables_run: bool,
    rings: Rings,
    shapes: Vec<ChannelShape>,
    cursors: Vec<Cursor>,
    /// Each launching stage's tables, a set a step the load keeps in
    /// flight (its arms, at least two): a pass is staged in a set no pass
    /// still in flight reads. What the host writes for a fire (the status word reset to
    /// ready, the pending flags, the grouped lane table and its slots, the
    /// row tables and output addresses) lies in its set; the value
    /// scratch does not, the queue running a fire's kernels after the
    /// last's.
    sets: Vec<Vec<Option<Prepared>>>,
    /// The set the pass being staged (or fired alone) takes.
    staging: usize,
    /// Passes staged so far: the next one's set is this modulo the sets
    /// it may use.
    passes: u64,
    /// Whether it may have two passes in flight ([`Session::allow_run_ahead`]).
    run_ahead: bool,
    /// The score rows its capturing lanes write (a key row and its page
    /// row), where its program reads them out.
    capture: Option<held::Held>,
    /// The outputs bound, by name and lane.
    bound: Vec<(String, u32)>,
    /// Seats past lane 0 that no stage reads (no output there, and past the
    /// rows of the output's lane-0 read): refused at the next fire, by name,
    /// never dropped.
    unread: Vec<(String, u32)>,
    poisoned: bool,
    fires: u64,
    /// The passes in flight, oldest first: at most one, two where it runs
    /// ahead.
    airborne: std::collections::VecDeque<Airborne>,
    /// The verdict of a pass that did not commit while another was staged
    /// after it: that one was staged on cursors its commit would have left,
    /// so it does not commit either and answers this.
    stale: Option<Fired>,
}

/// The fewest sets of tables a session holds ([`Session::sets`]): one a
/// pass in flight and one being staged.
pub const MIN_SETS: usize = 2;

/// A pass in flight: the stages it launched, the set it reads, and where
/// its verdict is told once its harvest reads it (the next pass's
/// completion check reads it there: [`EpilogueCheck`]).
#[derive(Debug)]
struct Airborne {
    stages: Vec<usize>,
    set: usize,
    told: std::sync::Arc<std::sync::OnceLock<Option<String>>>,
}

/// Why a verdict is not a commit, or `None` for a commit.
fn not_committed(fired: &Fired) -> Option<String> {
    match fired {
        Fired::Committed => None,
        Fired::Blocked(channel) => Some(format!("blocked on channel {channel}")),
        Fired::Declined => Some("the program declined the pass".to_string()),
        Fired::Faulted(why) => Some(why.clone()),
    }
}

impl Session {
    /// The bind-time tables an instance of `compiled` takes, every
    /// launching stage's in order.
    #[must_use]
    /// The value scratch one lane of the program takes in a fire at
    /// `extents`: the widest launching stage's. Stages fire one after
    /// another in one serial pass order and none reads another's scratch
    /// (values cross stages through channels), so one stage's room serves
    /// every stage of the fire.
    pub fn scratch_for(compiled: &Compiled, extents: &Extents) -> Result<u64> {
        let mut widest = 0;
        for (index, plan) in compiled.plans.iter().enumerate() {
            if launches(compiled, index) {
                widest = widest.max(Prepared::scratch_for(plan, &compiled.stages[index].regions, extents)?);
            }
        }
        Ok(widest)
    }

    /// The stages this instance carries, launching or not.
    #[must_use]
    pub fn stages(&self) -> usize {
        self.sets[0].len()
    }

    /// The tables an instance lays, in order: every launching stage's, a
    /// set after a set (`sets` of them, at least [`MIN_SETS`]), then its
    /// score rows.
    pub fn table_sizes(
        compiled: &Compiled,
        channels: usize,
        rows: u32,
        capture: u64,
        sets: usize,
    ) -> Vec<u64> {
        let channels = u32::try_from(channels).unwrap_or(u32::MAX);
        let set: Vec<u64> = compiled
            .plans
            .iter()
            .enumerate()
            .filter(|(index, _)| launches(compiled, *index))
            .flat_map(|(_, plan)| Prepared::table_sizes(plan, channels, rows))
            .collect();
        (0..sets.max(MIN_SETS))
            .flat_map(|_| set.iter().copied())
            .chain((capture > 0).then_some(capture + CAPTURE_ALIGN))
            .collect()
    }

    /// Lets it keep two passes in flight: a decode envelope's pages and
    /// slots are the host's and its token rides the device, so its next
    /// pass is staged while this one runs, the token copied from the cell
    /// the pass before it puts ([`Self::token_cell`]). Not where a stage of
    /// it fires alone (a prologue reads committed cells on the host), nor
    /// where the host reads another port from a channel for the next pass
    /// (a mask, positions, slots): that cell is not written yet.
    pub fn allow_run_ahead(
        &mut self,
        compiled: &Compiled,
        plan: &ExecPlan,
        class: GeometryClass,
        device: eta_ir::registry::PortMask,
    ) {
        let fires_alone = (0..compiled.stages.len()).any(|index| plan.runs_at(index, Stage::Prologue));
        let other_ports = plan.package.ports.iter().any(|binding| {
            !binding.is_const
                && binding.port != eta_ir::registry::Port::EmbedTokens
                && engine::ports::engine_resolves(class, device, binding.port)
        });
        self.run_ahead = !fires_alone && !other_ports;
    }

    /// Where the next pass's `embed_tokens` cell lies: the cell at the
    /// head its staging cursors name (the one [`Self::envelope`] reads),
    /// one u32 a lane. `None` where the port is not read from a channel.
    pub fn token_cell(&self, plan: &ExecPlan) -> Result<Option<(Buffer, u64)>> {
        let Some(binding) = plan
            .package
            .ports
            .iter()
            .find(|binding| binding.port == eta_ir::registry::Port::EmbedTokens && !binding.is_const)
        else {
            return Ok(None);
        };
        let channel = binding.channel as usize;
        let cursors = self.staging_cursors(plan);
        let Some(cursor) = cursors.get(channel) else {
            return Ok(None);
        };
        self.rings.cell_at(channel, cursor.head).map(Some)
    }

    /// Whether it may keep two passes in flight. A ring it shares stays
    /// predictable: the fence lets it run ahead only while no sharer of its
    /// rings fires in the step or has a frame in flight (`fence_instances`).
    #[must_use]
    pub const fn runs_ahead(&self) -> bool {
        self.run_ahead
    }

    /// Passes of it that may fly while another is staged: a set each, less
    /// the one being staged.
    #[must_use]
    pub fn ahead_depth(&self) -> usize {
        self.sets.len() - 1
    }

    /// Passes in flight.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.airborne.len()
    }

    /// Where this instance's score rows lie now: the buffer and its byte
    /// offset there, at [`CAPTURE_ALIGN`] of the GPU address; `None` where
    /// it holds none.
    #[must_use]
    pub fn capture_row(&self) -> Option<(&Buffer, u64)> {
        let held = self.capture.as_ref()?;
        Some((held.buffer(), held.offset() + capture_pad(held)))
    }

    /// Reads `into.len()` bytes of this instance's score rows from byte `at`.
    pub fn read_capture(&self, at: u64, into: &mut [u8]) -> Result<()> {
        let held = self.capture.as_ref().ok_or_else(|| {
            Fault::program("program::session", "this instance holds no score rows")
        })?;
        held.read(capture_pad(held) + at, into)
    }

    /// The instance's bind-time tables and rings go in `pooled`'s slots.
    pub fn bind(
        compiled: &Compiled,
        plan: &ExecPlan,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        adopted: &[Option<std::sync::Arc<SharedRing>>],
        pooled: Pooled<'_>,
        rows: u32,
        capture: u64,
        set_count: usize,
    ) -> Result<Session> {
        stages_and_plans_agree(compiled)?;
        if plan.package.channels.is_empty() {
            return Err(Fault::program(
                "program::session",
                "an instance with no channels: there is nothing for a fire to read \
                 or publish",
            ));
        }

        let shapes: Vec<ChannelShape> =
            plan.package.channels.iter().map(ChannelShape::of).collect();
        let rings = Rings::allocate(&shapes, adopted, pooled.pools, pooled.rings)?;
        let sizes = Session::table_sizes(compiled, shapes.len(), rows, capture, set_count);
        let tables_run = TableLayout::of(&sizes, pooled.pools.slot_bytes(INSTANCE)?).run;
        let mut tables = Tables::pooled(pooled.pools, INSTANCE, pooled.instance, &sizes)?;
        let cursors = vec![Cursor { head: 0, tail: 0 }; shapes.len()];

        let mut sets = Vec::with_capacity(set_count.max(MIN_SETS));
        for _ in 0..set_count.max(MIN_SETS) {
            let mut prepared = Vec::with_capacity(compiled.plans.len());
            for (index, stage_plan) in compiled.plans.iter().enumerate() {
                prepared.push(if launches(compiled, index) {
                    Some(Prepared::build(
                        stage_plan,
                        &compiled.stages[index].regions,
                        &shapes,
                        extents,
                        rows,
                        &mut tables,
                    )?)
                } else {
                    None
                });
            }
            sets.push(prepared);
        }
        // The score rows its fires' capturing lanes write, after the
        // tables: state of the instance, parked and unparked with it.
        let capture = (capture > 0)
            .then(|| tables.take(capture + CAPTURE_ALIGN))
            .transpose()?;
        tables.finish()?;

        let mut session = Session {
            instance_slots: pooled.instance.to_vec(),
            tables_run,
            rings,
            shapes,
            cursors,
            sets,
            staging: 0,
            passes: 0,
            run_ahead: false,
            capture,
            bound: Vec::new(),
            unread: Vec::new(),
            poisoned: false,
            fires: 0,
            airborne: std::collections::VecDeque::new(),
            stale: None,
        };
        for (channel, wire) in seeds {
            if let Some(ring) = session.rings.shared(*channel as usize)
                && !ring.claim_seeding()
            {
                continue;
            }
            if !session.publish(*channel, wire)? {
                return Err(Fault::program(
                    "program::session",
                    format!(
                        "channel {channel}'s seed does not fit: its ring already holds \
                         a cell, so the seed would be the second value rather than the first"
                    ),
                ));
            }
        }
        Ok(session)
    }

    /// Points the pooled tables and rings at the chunks their slots sit in
    /// now (after an evacuation moved them).
    pub fn rebind(&mut self, pools: &Pools) -> Result<()> {
        self.rings.rebind(pools)?;
        if let Some(capture) = self.capture.as_mut() {
            capture.rebind(pools)?;
        }
        self.sets
            .iter_mut()
            .flatten()
            .flatten()
            .try_for_each(|prepared| prepared.rebind(pools))
    }

    /// Moves the pooled tables to `instance` (the refs, in order, that
    /// replace this instance's) and its own rings to `rings`; the runtime
    /// already copied their bytes there.
    pub fn retarget(
        &mut self,
        pools: &Pools,
        instance: &[u32],
        rings: &[engine::program::RingSlots],
    ) -> Result<()> {
        if instance.len() != self.instance_slots.len()
            || (self.tables_run && !held::consecutive(instance))
        {
            return Err(Fault::program(
                "program::session",
                format!(
                    "an instance in {} instance slot(s) rebound to {}",
                    self.instance_slots.len(),
                    instance.len()
                ),
            ));
        }
        let old = self.instance_slots.clone();
        let to = |slot: u32| -> Result<u32> {
            old.iter()
                .position(|&held| held == slot)
                .map(|at| instance[at])
                .ok_or_else(|| {
                    Fault::program(
                        "program::session",
                        format!("a table in slot {slot}, which this instance does not hold"),
                    )
                })
        };
        for prepared in self.sets.iter_mut().flatten().flatten() {
            prepared.retarget(pools, &to)?;
        }
        if let Some(capture) = self.capture.as_mut()
            && let Some(slot) = capture.slot()
        {
            capture.retarget(pools, to(slot)?)?;
        }
        self.rings.retarget(pools, rings)?;
        self.instance_slots = instance.to_vec();
        Ok(())
    }

    #[must_use]
    pub fn channels(&self) -> usize {
        self.shapes.len()
    }

    #[must_use]
    pub fn shape(&self, channel: u32) -> Option<ChannelShape> {
        self.shapes.get(channel as usize).copied()
    }

    #[must_use]
    pub fn cursor(&self, channel: u32) -> Option<Cursor> {
        let channel = channel as usize;
        if let Some(ring) = self.rings.shared(channel) {
            return Some(ring.cursor());
        }
        self.cursors.get(channel).copied()
    }

    #[must_use]
    pub fn cursors_now(&self) -> Vec<Cursor> {
        let mut cursors = self.cursors.clone();
        for (channel, cursor) in cursors.iter_mut().enumerate() {
            if let Some(ring) = self.rings.shared(channel) {
                *cursor = ring.cursor();
            }
        }
        cursors
    }

    /// The cell at channel `channel`'s head, as the engine's port reads see
    /// it (`ports::resolve`): for a test of the port's view.
    #[cfg(test)]
    pub fn head_cell(&self, channel: u32) -> Result<Vec<u8>> {
        let head = self.cursors_now()[channel as usize].head;
        self.rings.read_cell(channel as usize, head)
    }

    #[must_use]
    pub const fn fires(&self) -> u64 {
        self.fires
    }

    #[must_use]
    pub const fn poisoned(&self) -> bool {
        self.poisoned
    }

    #[must_use]
    pub fn depth(&self, channel: u32) -> u64 {
        self.cursor(channel)
            .map_or(0, |cursor| cursor.tail.saturating_sub(cursor.head))
    }

    pub fn publish(&mut self, channel: u32, wire: &[u8]) -> Result<bool> {
        let shape = self.shape_of(channel)?;
        if self.depth(channel) >= u64::from(shape.capacity) {
            return Ok(false);
        }
        let want = shape.cell_bytes();
        if wire.len() != want {
            return Err(Fault::program(
                "program::session",
                format!(
                    "a {} cell of {} lane(s) on channel {channel} is {want} bytes and {} \
                     were offered",
                    name_or_unknown(shape.dtype),
                    shape.numel,
                    wire.len()
                ),
            ));
        }
        let slot = channel as usize;
        let tail = self.cursor(channel).unwrap_or_default().tail;
        self.rings.write_cell(slot, tail, wire)?;
        match self.rings.shared(slot) {
            Some(ring) => ring.bump_tail(),
            None => self.cursors[slot].tail = tail + 1,
        }
        Ok(true)
    }

    pub fn take(&mut self, channel: u32) -> Result<Option<Vec<u8>>> {
        if self.depth(channel) == 0 {
            return Ok(None);
        }
        let slot = channel as usize;
        let head = self.cursor(channel).unwrap_or_default().head;
        let cell = self.rings.read_cell(slot, head)?;
        match self.rings.shared(slot) {
            Some(ring) => ring.bump_head(),
            None => self.cursors[slot].head = head + 1,
        }
        Ok(Some(cell))
    }

    pub fn feed_cell(&self, channel: u32) -> Result<Option<(Buffer, u64, u64, eta_ir::Dtype)>> {
        let slot = channel as usize;
        let shape = self.shape_of(channel)?;
        if self.depth(channel) == 0 {
            return Ok(None);
        }
        let head = self.cursor(channel).unwrap_or_default().head;
        let (buffer, at) = self.rings.cell_at(slot, head)?;
        Ok(Some((buffer, at, shape.cell_bytes() as u64, shape.dtype)))
    }

    pub fn peek(&self, channel: u32, sequence: u64) -> Result<Vec<u8>> {
        self.shape_of(channel)?;
        self.rings.read_cell(channel as usize, sequence)
    }

    pub fn envelope(
        &self,
        plan: &ExecPlan,
        class: GeometryClass,
        device: eta_ir::registry::PortMask,
    ) -> Result<Envelope> {
        ports::resolve(
            plan,
            class,
            device,
            &self.rings,
            &self.staging_cursors(plan),
            &self.shapes,
        )
    }

    /// The cursors the next pass is staged on: the committed ones, then
    /// each pass in flight's commit as its plan states it (it commits in
    /// order, before the next; one that does not takes the next with it).
    #[must_use]
    pub fn staging_cursors(&self, plan: &ExecPlan) -> Vec<Cursor> {
        let mut cursors = self.cursors_now();
        for _ in &self.airborne {
            if let Ok((next, _)) = self.commit_plan_from(&cursors, plan, Stage::Epilogue) {
                cursors = next;
            }
        }
        cursors
    }

    #[allow(clippy::too_many_arguments)]
    /// Refuses a fire while a seat no stage reads is bound: an output at a
    /// lane past 0 that the program reads nothing at, nor a row of its
    /// lane-0 read reaches.
    /// Whether a stage's lane-0 read of `name` covers the instance's lane
    /// `lane` ([`super::launch::Prepared::covers`]).
    #[must_use]
    pub fn covers(&self, name: &str, lane: u32) -> bool {
        self.sets[0].iter().flatten().any(|prepared| prepared.covers(name, lane))
    }

    /// Why a stage of this instance reads rows past what its lanes' seats
    /// hold, if one does ([`super::launch::Prepared::rows_unbounded`]).
    #[must_use]
    pub fn rows_unbounded(&self) -> Option<String> {
        self.sets[0].iter().flatten().find_map(super::launch::Prepared::rows_unbounded)
    }

    pub fn refuse_unread(&self) -> Result<()> {
        match self.unread.first() {
            None => Ok(()),
            Some((name, lane)) => Err(Fault::program(
                "program::session",
                format!(
                    "output `{name}` is seated at lane {lane}, which this program reads \
                     nothing at: no stage reads `{name}` there, and its lane-0 read covers \
                     fewer rows than {}",
                    lane + 1
                ),
            )),
        }
    }

    pub fn bind_output(
        &mut self,
        name: &str,
        lane: u32,
        base: &Buffer,
        offset: u64,
        dtype: eta_ir::Dtype,
        width: u32,
        stride: u32,
        rows_held: u32,
    ) -> Result<()> {
        let mut read = false;
        for (set, prepared) in self.sets.iter_mut().enumerate() {
            for prepared in prepared.iter_mut().flatten() {
                let here = prepared.bind_output(name, lane, base, offset, dtype, width, stride, rows_held)?;
                if set == 0 {
                    read |= here;
                }
            }
        }
        self.unread.retain(|(named, at)| !(named == name && *at == lane));
        if !read && lane > 0 {
            self.unread.push((name.to_string(), lane));
        }
        if !self
            .bound
            .iter()
            .any(|(named, at)| named == name && *at == lane)
        {
            self.bound.push((name.to_string(), lane));
        }
        Ok(())
    }

    /// Runs the stages at `at` to their commit: the prologue's before the forward, the rest after.
    pub fn fire(
        &mut self,
        device: &Context,
        pipelines: &Pipelines,
        compiled: &Compiled,
        plan: &ExecPlan,
        at: Stage,
        scratch: Option<super::launch::ScratchAt>,
    ) -> Result<Fired> {
        let _ = pipelines;
        let mut frame = device.frame()?;
        let launched = match self.encode_stages(&mut frame, compiled, plan, at, scratch)? {
            Ok(launched) => launched,
            Err(refused) => return Ok(refused),
        };
        frame.commit()?;
        self.verdict_of_launched(plan, &launched, at)
    }

    /// The stages at `at` encoded into `frame`, each region a pass after the
    /// one before (so they run in order, as one command buffer each did):
    /// the stages launched, or the gate's refusal (nothing encoded). Their
    /// verdict is read once `frame` has landed
    /// ([`Self::verdict_of_launched`]): a step's prologues share one buffer
    /// and one wait, not one a region an instance.
    pub fn encode_stages(
        &mut self,
        frame: &mut crate::device::ctx::Frame,
        compiled: &Compiled,
        plan: &ExecPlan,
        at: Stage,
        scratch: Option<super::launch::ScratchAt>,
    ) -> Result<std::result::Result<Vec<usize>, Fired>> {
        if !self.airborne.is_empty() {
            return Err(Fault::program(
                "program::session",
                "this instance has a pass airborne in someone else's command buffer: a \
                 standalone fire would reset the status word that pass has not been \
                 read from, and both fires would report the second's verdict",
            ));
        }
        self.refuse_unread()?;
        if let Some(refused) = self.gate(compiled, plan, at)? {
            return Ok(Err(refused));
        }
        self.staging = 0;

        let cursors = self.cursors_now();
        let mut launched = Vec::with_capacity(compiled.stages.len());
        for (index, stage) in compiled.stages.iter().enumerate() {
            if !plan.runs_at(index, at) {
                continue;
            }
            let Some(prepared) = self.sets[self.staging].get_mut(index).and_then(Option::as_mut) else {
                continue;
            };
            prepared.refresh(&self.rings, &cursors)?;
            // Its own scratch (its state's first frame in flight): a
            // standalone fire drains every frame before it.
            prepared.seat_scratch(scratch.clone().ok_or_else(|| {
                Fault::program("program::session", "an instance fires a stage and holds no scratch")
            })?)?;
            for (at, region) in stage.regions.iter().enumerate() {
                prepared.encode_region(frame, region, at == 0)?;
            }
            launched.push(index);
        }
        Ok(Ok(launched))
    }

    /// The verdict of stages [`Self::encode_stages`] launched, read once
    /// their frame has landed.
    pub fn verdict_of_launched(
        &mut self,
        plan: &ExecPlan,
        launched: &[usize],
        at: Stage,
    ) -> Result<Fired> {
        self.verdict(plan, launched, self.staging, at)
    }

    pub fn prepare_airborne(
        &mut self,
        compiled: &Compiled,
        plan: &ExecPlan,
    ) -> Result<Option<Fired>> {
        self.refuse_unread()?;
        let ahead = self.runs_ahead();
        let deepest = if ahead { self.sets.len() } else { 1 };
        if self.airborne.len() >= deepest {
            return Err(Fault::program(
                "program::session",
                "this instance already has as many passes airborne as it has sets of \
                 tables: another staging would refresh the cells and the status word \
                 of stages whose pass has not been read, so both fires would report \
                 the second's verdict and the first's commit would be lost",
            ));
        }
        if let Some(refused) = self.gate(compiled, plan, Stage::Epilogue)? {
            return Ok(Some(refused));
        }
        // The set the pass in flight before it does not read (with none in
        // flight, the first).
        self.staging = self.airborne.back().map_or(0, |pass| (pass.set + 1) % self.sets.len());
        let _ = ahead;
        let cursors = self.staging_cursors(plan);
        let mut launched = Vec::with_capacity(compiled.stages.len());
        for (index, stage) in compiled.stages.iter().enumerate() {
            if !plan.runs_at(index, Stage::Epilogue) {
                continue;
            }
            let Some(prepared) = self.sets[self.staging].get_mut(index).and_then(Option::as_mut) else {
                continue;
            };
            if stage.regions.is_empty() {
                continue;
            }
            prepared.refresh_cells(&self.rings, &cursors)?;
            launched.push(index);
        }
        self.passes += 1;
        self.airborne.push_back(Airborne {
            stages: launched,
            set: self.staging,
            told: std::sync::Arc::default(),
        });
        Ok(None)
    }

    /// Stage `stage`'s tables in the set being staged.
    #[must_use]
    pub fn prepared(&self, stage: usize) -> Option<&Prepared> {
        self.sets[self.staging].get(stage).and_then(Option::as_ref)
    }

    pub fn prepared_mut(&mut self, stage: usize) -> Option<&mut Prepared> {
        self.sets[self.staging].get_mut(stage).and_then(Option::as_mut)
    }

    /// What the completion handler of the frame carrying this instance's
    /// newest airborne pass reads to tell its verdict without the lane
    /// thread: each launched stage's status word in its set, and the commit
    /// its harvest will make from the cursors the passes before it leave.
    /// Where a pass is in flight before it, the check carries that one's
    /// too: if it does not commit, neither does this one. `None` with
    /// nothing airborne.
    #[must_use]
    pub fn epilogue_check(&self, plan: &ExecPlan) -> Option<EpilogueCheck> {
        let newest = self.airborne.back()?;
        let statuses = newest
            .stages
            .iter()
            .filter_map(|&index| {
                let prepared = self.sets[newest.set].get(index).and_then(Option::as_ref)?;
                Some((index, prepared.status_region()))
            })
            .collect();
        // The cursors before it: the committed ones, then each older pass's.
        let mut cursors = self.cursors_now();
        let older = self.airborne.len() - 1;
        let mut before = None;
        for pass in self.airborne.iter().take(older) {
            let statuses = pass
                .stages
                .iter()
                .filter_map(|&index| {
                    let prepared = self.sets[pass.set].get(index).and_then(Option::as_ref)?;
                    Some((index, prepared.status_region()))
                })
                .collect();
            let commit = self.commit_plan_from(&cursors, plan, Stage::Epilogue);
            if let Ok((next, _)) = &commit {
                cursors = next.clone();
            }
            before = Some(Box::new((
                pass.told.clone(),
                EpilogueCheck {
                    statuses,
                    commit: commit.map(|_| ()),
                    before: None,
                    told: pass.told.clone(),
                },
            )));
        }
        Some(EpilogueCheck {
            statuses,
            commit: self.commit_plan_from(&cursors, plan, Stage::Epilogue).map(|_| ()),
            before,
            told: newest.told.clone(),
        })
    }

    /// The oldest pass in flight, settled: read and committed, or, where
    /// the pass before it did not commit, answered that one's verdict
    /// without committing (it was staged on cursors that commit would have
    /// left). Its verdict is told where the next pass's check reads it.
    pub fn settle_launched(&mut self, plan: &ExecPlan) -> Result<Fired> {
        let Some(pass) = self.airborne.pop_front() else {
            return Ok(Fired::Committed);
        };
        let fired = match self.stale.clone() {
            Some(before) => before,
            None => self.verdict(plan, &pass.stages, pass.set, Stage::Epilogue)?,
        };
        self.stale = if self.airborne.is_empty() || fired == Fired::Committed {
            None
        } else {
            Some(fired.clone())
        };
        let _ = pass.told.set(not_committed(&fired));
        Ok(fired)
    }

    /// Lets the oldest pass in flight go unread (its frame failed): a pass
    /// staged after it was staged on its commit, so it answers this too.
    pub fn abandon_launched(&mut self) {
        let Some(pass) = self.airborne.pop_front() else {
            return;
        };
        let why = "the pass staged before it did not land".to_string();
        let _ = pass.told.set(Some(why.clone()));
        self.stale = (!self.airborne.is_empty()).then(|| Fired::Faulted(why));
    }

    fn gate(&mut self, compiled: &Compiled, plan: &ExecPlan, at: Stage) -> Result<Option<Fired>> {
        if self.poisoned {
            return Ok(Some(Fired::Faulted("instance is poisoned".to_string())));
        }
        stages_and_plans_agree(compiled)?;
        if at != Stage::Prologue {
            self.outputs_bound(plan)?;
        }
        if let Some(blocked) = self.blocked_at(plan, at) {
            return Ok(Some(Fired::Blocked(blocked)));
        }
        Ok(None)
    }

    /// The emitted kernel reads an output's slot regardless: refuse one left unbound.
    fn outputs_bound(&self, plan: &ExecPlan) -> Result<()> {
        for (name, lane) in plan.outputs() {
            if !self
                .bound
                .iter()
                .any(|(named, at)| named == name && *at == lane)
            {
                return Err(Fault::program(
                    "program::session",
                    format!(
                        "this program reads output `{name}` of its pass's lane {lane} and \
                         nothing has been bound to it: the pass has no such lane, or the \
                         lane's class writes no such output"
                    ),
                ));
            }
        }
        Ok(())
    }

    fn blocked_at(&self, plan: &ExecPlan, at: Stage) -> Option<u32> {
        let cursors = self.staging_cursors(plan);
        (0..self.shapes.len()).find_map(|channel| {
            let live = cursors
                .get(channel)
                .map_or(0, |cursor| cursor.tail.saturating_sub(cursor.head));
            let capacity = u64::from(self.shapes[channel].capacity);
            let ready = match plan.readiness(channel as u32, at) {
                Some(Direction::NeedsFull) => live != 0,
                Some(Direction::NeedsEmpty) => live < capacity,
                None => true,
            };
            (!ready).then_some(channel as u32)
        })
    }

    fn verdict(&mut self, plan: &ExecPlan, launched: &[usize], set: usize, at: Stage) -> Result<Fired> {
        let max_channel = u32::try_from(self.shapes.len().saturating_sub(1)).unwrap_or(u32::MAX);

        let mut verdict = Verdict::Committed;
        for &index in launched {
            let Some(prepared) = self.sets[set].get(index).and_then(Option::as_ref) else {
                continue;
            };
            verdict = verdict.worse(verdict_of(
                &plan.package,
                index,
                prepared.status()?,
                max_channel,
            ));
        }

        match verdict {
            Verdict::Faulted(why) => {
                self.poisoned = true;
                return Ok(Fired::Faulted(why));
            }
            Verdict::Declined => return Ok(Fired::Declined),
            Verdict::Committed => {}
        }

        match self.commit(plan, at) {
            Ok(()) => {
                self.fires += 1;
                Ok(Fired::Committed)
            }
            Err(why) => {
                self.poisoned = true;
                Ok(Fired::Faulted(why))
            }
        }
    }

    #[must_use]
    pub fn readiness(&self, plan: &ExecPlan) -> Option<Blocked> {
        let cursors = self.staging_cursors(plan);
        for channel in 0..self.shapes.len() {
            let needs = plan
                .package
                .channels
                .get(channel)
                .and_then(|declared| declared.readiness);
            let live = cursors
                .get(channel)
                .map_or(0, |cursor| cursor.tail.saturating_sub(cursor.head));
            let capacity = u64::from(self.shapes[channel].capacity);
            let ready = match needs {
                Some(Direction::NeedsFull) => live != 0,
                Some(Direction::NeedsEmpty) => live < capacity,
                None => true,
            };
            if !ready {
                return Some(Blocked {
                    channel: channel as u32,
                    needs,
                    live,
                    capacity,
                });
            }
        }
        None
    }

    fn commit(&mut self, plan: &ExecPlan, at: Stage) -> std::result::Result<(), String> {
        let (next, bumps) = self.commit_plan(plan, at)?;
        for (channel, took, put) in bumps {
            let Some(ring) = self.rings.shared(channel) else {
                continue;
            };
            if took {
                ring.bump_head();
            }
            if put {
                ring.bump_tail();
            }
        }
        self.cursors = next;
        Ok(())
    }

    /// What [`Self::commit`] at `at` would set the cursors to and which
    /// shared rings it would bump, or why it refuses; nothing moves.
    #[allow(clippy::type_complexity)]
    fn commit_plan(
        &self,
        plan: &ExecPlan,
        at: Stage,
    ) -> std::result::Result<(Vec<Cursor>, Vec<(usize, bool, bool)>), String> {
        self.commit_plan_from(&self.cursors_now(), plan, at)
    }

    /// [`Self::commit_plan`] from `now` rather than the committed cursors:
    /// a pass staged behind another commits from where that one leaves them.
    #[allow(clippy::type_complexity)]
    fn commit_plan_from(
        &self,
        now: &[Cursor],
        plan: &ExecPlan,
        at: Stage,
    ) -> std::result::Result<(Vec<Cursor>, Vec<(usize, bool, bool)>), String> {
        let now = now.to_vec();
        let mut next = now.clone();
        let mut bumps: Vec<(usize, bool, bool)> = Vec::new();
        for (channel, cursor) in now.iter().enumerate() {
            if cursor.tail < cursor.head {
                return Err(format!("channel {channel}: tail precedes head at commit"));
            }
            let mut used = cursor.tail - cursor.head;
            let capacity = u64::from(self.shapes[channel].capacity);
            let took = plan.takes_channel(channel as u32, at) && used != 0;
            if took {
                next[channel].head = cursor.head + 1;
                used -= 1;
            }
            let put = plan.puts_channel(channel as u32, at);
            if put {
                if used >= capacity {
                    return Err(format!(
                        "channel {channel}: put overflows capacity {capacity} at commit"
                    ));
                }
                next[channel].tail = cursor.tail + 1;
            }
            if (took || put) && self.rings.shared(channel).is_some() {
                bumps.push((channel, took, put));
            }
        }
        Ok((next, bumps))
    }

    fn shape_of(&self, channel: u32) -> Result<ChannelShape> {
        self.shapes.get(channel as usize).copied().ok_or_else(|| {
            Fault::program(
                "program::session",
                format!("channel {channel} is not one this instance carries"),
            )
        })
    }
}

#[derive(Clone, Debug)]
enum Verdict {
    Committed,
    Declined,
    Faulted(String),
}

impl Verdict {
    fn worse(self, other: Verdict) -> Verdict {
        match (self, other) {
            (fault @ Verdict::Faulted(_), _) => fault,
            (_, fault @ Verdict::Faulted(_)) => fault,
            (Verdict::Declined, _) | (_, Verdict::Declined) => Verdict::Declined,
            (Verdict::Committed, Verdict::Committed) => Verdict::Committed,
        }
    }
}

fn verdict_of(
    package: &LaunchPackage,
    stage: usize,
    status: eta_exec::Status,
    max_channel: u32,
) -> Verdict {
    match status.state() {
        Some(eta_exec::State::Running) => Verdict::Committed,
        Some(eta_exec::State::Committed) => Verdict::Committed,
        Some(eta_exec::State::Retry) => Verdict::Declined,
        Some(eta_exec::State::Fault | eta_exec::State::Unset) | None => Verdict::Faulted(format!(
            "stage {stage}: {}",
            eta_exec::report_status(package, status, true, max_channel)
        )),
    }
}

fn stages_and_plans_agree(compiled: &Compiled) -> Result<()> {
    stage_plans_are_parallel(
        &compiled
            .stages
            .iter()
            .map(|stage| (stage.signature_hash, !stage.regions.is_empty()))
            .collect::<Vec<_>>(),
        &compiled
            .plans
            .iter()
            .map(|plan| plan.signature_hash)
            .collect::<Vec<_>>(),
    )
}

fn stage_plans_are_parallel(stages: &[(u64, bool)], plans: &[u64]) -> Result<()> {
    if stages.len() != plans.len() {
        return Err(Fault::program(
            "program::session",
            format!(
                "this program has {} compiled stage(s) and {} plan(s): the fire pairs \
                 them by index to prepare each stage's own scratch, so it cannot tell \
                 which plan describes which stage",
                stages.len(),
                plans.len()
            ),
        ));
    }
    for (index, (&(signature, launches), &plan)) in stages.iter().zip(plans).enumerate() {
        if launches && signature != plan {
            return Err(Fault::program(
                "program::session",
                format!(
                    "stage {index} has regions to launch and the plan at that index \
                     describes a different stage: its regions would index scratch, \
                     descriptors and a channel table sized for someone else"
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use eta_compiler::codegen::launch::LaunchPackage;

    use super::{Verdict, verdict_of};

    fn package() -> LaunchPackage {
        LaunchPackage {
            fault_classes: eta_compiler::codegen::fault::classes(),
            ..LaunchPackage::default()
        }
    }

    #[test]
    fn session_every_case() {
        an_adapter_prologue_and_a_sampling_epilogue_are_one_fire();
        two_launching_stages_each_get_their_own_plan();
        a_stage_that_launches_nothing_is_not_compared();
        a_kernel_readiness_miss_is_a_decline_and_not_a_fault();
        a_fault_outranks_a_decline_in_either_order();
    }

    fn an_adapter_prologue_and_a_sampling_epilogue_are_one_fire() {
        super::stage_plans_are_parallel(&[(0xa11, false), (0xb22, true)], &[0xa11, 0xb22])
            .expect("the plans are parallel and the launching stage is its own");
    }

    fn two_launching_stages_each_get_their_own_plan() {
        super::stage_plans_are_parallel(&[(0xa11, true), (0xb22, true)], &[0xa11, 0xb22])
            .expect("both launch, both are prepared from their own plan");
    }

    fn a_stage_that_launches_nothing_is_not_compared() {
        super::stage_plans_are_parallel(&[(0xa11, false)], &[0xdead])
            .expect("nothing is prepared from it");
    }

    fn a_kernel_readiness_miss_is_a_decline_and_not_a_fault() {
        let status = eta_exec::Status {
            state: 2,
            fault: 0x480,
            reserved0: 0,
            reserved1: 0,
        };
        assert!(matches!(
            verdict_of(&package(), 0, status, 3),
            Verdict::Declined
        ));
    }

    fn a_fault_outranks_a_decline_in_either_order() {
        let fault = || Verdict::Faulted("stage 0: guard".to_string());
        assert!(matches!(
            fault().worse(Verdict::Declined),
            Verdict::Faulted(_)
        ));
        assert!(matches!(
            Verdict::Declined.worse(fault()),
            Verdict::Faulted(_)
        ));
        assert!(matches!(
            Verdict::Committed.worse(Verdict::Declined),
            Verdict::Declined
        ));
        assert!(matches!(
            Verdict::Committed.worse(Verdict::Committed),
            Verdict::Committed
        ));
    }
}

/// An airborne pass's verdict as [`Session::verdict`] will reach it at the
/// harvest, readable from any thread once its frame's buffer completed
/// ([`Session::epilogue_check`]).
pub struct EpilogueCheck {
    statuses: Vec<(usize, held::Held)>,
    commit: std::result::Result<(), String>,
    /// The pass in flight before it, where one is: its verdict as its
    /// harvest told it, else as its own check reads it (its set is not
    /// staged again before that harvest). If it does not commit, this one
    /// does not either.
    before: Option<Box<(std::sync::Arc<std::sync::OnceLock<Option<String>>>, EpilogueCheck)>>,
    /// Where its own verdict is told for the pass staged after it.
    told: std::sync::Arc<std::sync::OnceLock<Option<String>>>,
}

impl EpilogueCheck {
    /// `None` where the pass commits; why it doesn't otherwise. Its
    /// status words are read here, so only once the buffer completed.
    /// Tells its verdict where the pass staged after it reads it: from the
    /// completion handler, which Metal calls in the order a queue's buffers
    /// complete, so before that pass's handler. Its harvest tells the same.
    pub fn tell(&self, fault: Option<String>) {
        let _ = self.told.set(fault);
    }

    #[must_use]
    pub fn fault(&self) -> Option<String> {
        if let Some(before) = &self.before {
            let (told, check) = &**before;
            let earlier = match told.get() {
                Some(told) => told.clone(),
                None => check.fault(),
            };
            if let Some(why) = earlier {
                return Some(format!("the pass staged before it did not commit: {why}"));
            }
        }
        let mut verdict = Verdict::Committed;
        for (stage, status) in &self.statuses {
            let mut bytes = [0u8; eta_exec::STATUS_BYTES];
            let read = status
                .read(0, &mut bytes)
                .ok()
                .and_then(|()| eta_exec::Status::read(&bytes));
            let here = match read.and_then(|status| status.state()) {
                Some(eta_exec::State::Running | eta_exec::State::Committed) => Verdict::Committed,
                Some(eta_exec::State::Retry) => Verdict::Declined,
                other => Verdict::Faulted(format!(
                    "stage {stage}: its status word reads {other:?} (the harvest logs its report)"
                )),
            };
            verdict = verdict.worse(here);
        }
        match verdict {
            Verdict::Faulted(why) => Some(why),
            Verdict::Declined => Some("the program declined the pass".to_string()),
            Verdict::Committed => self.commit.as_ref().err().cloned(),
        }
    }
}
