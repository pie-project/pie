//! One bound instance: its rings, its per-stage buffers, and one fire.
//!
//! **THE HOST GATES AND THE HOST COMMITS; THE DEVICE COMPUTES.** A fire is
//! readiness, then every stage's regions, then one commit for the whole
//! program — and the first and third of those are arithmetic over ring
//! cursors that [`driver::program`](driver) already writes, once, for the host
//! interpreter. The lineage compiled that arithmetic into device control
//! kernels so a blocked fire never had to reach the host; this half runs it
//! host-side instead, and the reason is the one the whole step exists for:
//! with readiness and commit shared, the parity test is left diffing the part
//! that can actually differ — the emitted arithmetic — rather than diffing a
//! second implementation of a ring against the first.
//!
//! It is also honest about where PTIR sits. Design §9 puts guest computation
//! *at the fire's boundary*, outside the immutable graph, and a boundary is
//! exactly where a wait is already paid.
//!
//! **THERE IS NO STREAM ON THIS PLANE, AND NO `synchronize`.** That is the
//! largest single difference from the CUDA twin, and it is not a
//! simplification — it is what the platform is. Every buffer this plane
//! allocates is `StorageModeShared` on unified memory, so:
//!
//! * a host write IS the device write. The CUDA half stages a cell through
//!   `cuMemcpyHtoDAsync` on a stream and then has to order that copy against
//!   the launch that reads it; here [`Session::publish`] writes the bytes into
//!   mapped memory and there is nothing to order it against but the encode.
//! * a device write IS a host read. The CUDA half copies the commit word back
//!   before it can look at it; here [`Prepared::status`] loads sixteen bytes
//!   out of the same allocation the kernel wrote.
//!
//! **THE ONE ORDERING FACT THAT SURVIVES: A HOST WRITE MUST PRECEDE THE COMMIT
//! OF THE COMMAND BUFFER THAT READS IT**, and a host read must follow that
//! command buffer's completion. Shared storage removes the copy, not the
//! race. This file honours both by construction: everything it writes
//! (`publish`, `refresh`, `bind_intrinsic`) happens on the host before
//! [`Prepared::launch_region`] is called, and `launch_region` is where the wait
//! lives — it opens its own command buffer, encodes, commits and blocks until
//! completion before returning. So by the time [`Session::fire`] reads a
//! stage's status, or the next stage's `refresh` writes its lane table, the
//! previous stage's kernel has finished. There is no `context.synchronize()`
//! in this file because there is nothing left for it to do.
//!
//! **RUN-AHEAD CAME, AND THE CONTROL KERNELS DID NOT COME BACK** (`palo B3`,
//! Build log 15 named this the day to decide). The rewrite's two device
//! control kernels — a stage readiness and a commit bump — existed so a
//! blocked fire never had to reach the host, and the reason the C++ before it
//! needed them was a driver whose fires were genuinely asynchronous: the host
//! predicted where a cursor WOULD be (`expected_head`/`expected_tail` tickets)
//! because it could not look. This shell can look, for free. At the instant
//! the shell resolves a descriptor port every earlier region has already
//! completed and the cursor's real position is a load out of mapped memory. A
//! device readiness kernel would compute, on the GPU, an answer the host
//! already has — and it would compute it from a SECOND implementation of the
//! ring arithmetic, which is exactly the thing a parity test exists to avoid
//! diffing.
//!
//! What run-ahead needed was not a device gate. It was for the token to stop
//! travelling through the ENGINE's asynchronous host plane — out through
//! `take_channel`, through a guest `await`, back in through
//! `publish_channel` — and that is [`super::ports`], which reads the committed
//! cell in the shell. The two kernels stay unbuilt, and the condition that
//! would build them is unchanged and now sharper: a fire path that does NOT
//! wait on its command buffer before the next fire's ports are read. That is
//! not this shell.
//!
//! **ONE COMMIT PER FIRE, NOT PER STAGE.** A program's stages are separate
//! programs joined only by channels: nothing flows between them in scratch,
//! every stage resolves its cell addresses from the SAME cursors, and a stage
//! whose status is not `Committed` refuses the whole fire. Every stage still
//! launches when one refuses — the dummy run, so a refused fire costs what a
//! running one does — and then nothing moves.
//!
//! ```text
//! fire
//! ----
//! readiness   every channel the program declares a requirement for   host
//! refresh     each stage's lane table -> this fire's cell addresses  host
//! launch      each stage's regions, in stage order, one wait each    device
//! verdict     the worst `driver::Status` any stage left behind       device
//! commit      heads advance for takes, tails for puts                host
//! ```

use driver::tensor_ir::op::IntrinsicId;
use driver::tensor_ir::validate::Direction;
use driver::{ExecPlan, Extents};

use crate::device::{Buffer, Context, Pipelines};
use crate::error::{Fault, Result};

use super::compile::Compiled;
use super::launch::{ChannelShape, Cursor, Prepared, Rings};
use super::ports::{self, Envelope};

/// What one fire produced.
///
/// Deliberately the shape of [`driver::StepOutcome`], because the parity test
/// compares the two directly: a device fire that blocks must block on the same
/// channel the host interpreter blocks on, and neither half may quietly turn a
/// refusal into a commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fired {
    /// Every stage ran and the cursors advanced.
    Committed,
    /// Nothing launched: this channel did not meet the program's declared
    /// requirement. The counterpart of [`driver::StepOutcome::Blocked`].
    Blocked(u32),
    /// A stage's kernel declined the fire from inside — a readiness guard the
    /// emitted code observed for itself. The cursors are left where they
    /// were, so the next fire sees the same inputs and a caller may retry.
    Declined,
    /// The instance is unusable and stays so. The counterpart of
    /// [`driver::StepOutcome::Faulted`], and on this plane it is reachable
    /// from the DEVICE as well as from the commit: see [`Session::fire`].
    Faulted(String),
}

/// One bound instance's device state.
///
/// The cursors are `u64` sequence numbers that never wrap, exactly as the host
/// half's [`ChannelState`](driver::ChannelState) keeps them; a ring position
/// is the residue. Keeping the same spelling is what makes a slot-for-slot
/// diff of the two halves mean anything.
#[derive(Debug)]
pub struct Session {
    rings: Rings,
    shapes: Vec<ChannelShape>,
    cursors: Vec<Cursor>,
    /// One per stage plan, `None` for a stage with nothing to launch — the
    /// adapter prologue is exactly that, its one region being a sink the model
    /// fire reads out of the plan rather than a body anyone compiled.
    prepared: Vec<Option<Prepared>>,
    /// Which intrinsics have been pointed at a buffer, one bit per
    /// [`IntrinsicId`]. Tracked because the emitted kernel READS an unbound
    /// intrinsic's argument slot — an argument nobody bound is not a stated
    /// error on this plane, it is whatever the encoder last left there — so a
    /// program that reads the readout and was never handed one is a wrong
    /// answer or a GPU fault rather than a refusal. Refusing by name is the
    /// difference between a sentence and a command buffer that dies with
    /// `MTLCommandBufferErrorPageFault` and takes the queue with it.
    bound: u64,
    poisoned: bool,
    fires: u64,
}

impl Session {
    /// Allocate this instance's rings and every stage's fire-path buffers,
    /// then seed the channels the program declares seeds for.
    ///
    /// `seeds` are WIRE cells, one per `(channel, bytes)` pair — the same
    /// encoding [`driver::Registry::bind_instance`] takes, so an instance that
    /// already exists on the host half is adopted by handing over what its
    /// rings hold (see [`seeds_of`]) rather than by a second seeding rule.
    ///
    /// `extents` is what the program's symbolic value shapes resolve against.
    /// A guest program with no intrinsic resolves entirely from static dims
    /// and never reads it; one attached to a model fire is handed that fire's.
    ///
    /// `device` is taken where the CUDA twin takes nothing: an allocation on
    /// this plane is `[MTLDevice newBufferWithLength:options:]`, which needs
    /// the device object, where CUDA's is a context-global `cuMemAlloc`.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the compiled stages and the plans are not
    /// parallel, when a seed names a channel the instance does not carry or is
    /// not one cell wide, and whatever the allocations said.
    pub fn bind(
        device: &Context,
        compiled: &Compiled,
        plan: &ExecPlan,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
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
        let rings = Rings::allocate(device, &shapes)?;
        let cursors = vec![Cursor { head: 0, tail: 0 }; shapes.len()];

        let mut prepared = Vec::with_capacity(compiled.plans.len());
        for (index, stage_plan) in compiled.plans.iter().enumerate() {
            let launches = compiled
                .stages
                .get(index)
                .is_some_and(|stage| !stage.regions.is_empty());
            prepared.push(if launches {
                Some(Prepared::build(device, stage_plan, &shapes, extents)?)
            } else {
                None
            });
        }

        let mut session = Session {
            rings,
            shapes,
            cursors,
            prepared,
            bound: 0,
            poisoned: false,
            fires: 0,
        };
        for (channel, wire) in seeds {
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

    /// How many channels this instance carries.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.shapes.len()
    }

    /// Channel `channel`'s geometry.
    #[must_use]
    pub fn shape(&self, channel: u32) -> Option<ChannelShape> {
        self.shapes.get(channel as usize).copied()
    }

    /// Channel `channel`'s cursors.
    #[must_use]
    pub fn cursor(&self, channel: u32) -> Option<Cursor> {
        self.cursors.get(channel as usize).copied()
    }

    /// How many fires have committed on this instance.
    #[must_use]
    pub const fn fires(&self) -> u64 {
        self.fires
    }

    /// Whether this instance is unusable.
    #[must_use]
    pub const fn poisoned(&self) -> bool {
        self.poisoned
    }

    /// How many unconsumed cells channel `channel` holds.
    #[must_use]
    pub fn depth(&self, channel: u32) -> u64 {
        self.cursors
            .get(channel as usize)
            .map_or(0, |cursor| cursor.tail.saturating_sub(cursor.head))
    }

    /// Push one wire cell into channel `channel`, answering `false` when the
    /// ring has no room — back-pressure, not a drop.
    ///
    /// The host-side counterpart of [`driver::host_put`], and the only door a
    /// caller's bytes enter this plane through.
    ///
    /// **A WIRE CELL AND A METAL CELL ARE THE SAME BYTES, SO NOTHING IS
    /// CONVERTED HERE.** The CUDA twin calls `wire_to_native` on this line,
    /// because its emitted kernels read a `Bool` lane as one whole byte and
    /// the wire packs eight to a byte. This plane's runtime
    /// (`ptir_m1_runtime.metal`, the `0x90`/`0x91`/`0x92` tags) packs and
    /// unpacks bools on the DEVICE, so the ring holds wire bytes for every
    /// dtype and the boundary has nothing to do. That is a genuine ABI
    /// difference between the two shells — the cell layouts are not the same
    /// on the two devices — and not a corner this port cut. What remains is
    /// the width check the conversion used to carry, kept here so a short
    /// cell is a named refusal rather than a cell with real-looking garbage
    /// past its end.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown channel or a cell of the wrong width,
    /// and whatever the write said.
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
                    shape.dtype.name(),
                    shape.numel,
                    wire.len()
                ),
            ));
        }
        let tail = self.cursors[channel as usize].tail;
        self.rings.write_cell(channel as usize, tail, wire)?;
        self.cursors[channel as usize].tail = tail + 1;
        Ok(true)
    }

    /// Take channel `channel`'s committed cell as wire bytes, advancing its
    /// head; `None` when the ring is empty.
    ///
    /// The counterpart of [`driver::host_take`]. The bytes come back exactly
    /// as the ring holds them — see [`Session::publish`] for why there is no
    /// unpacking step on this plane.
    ///
    /// An unknown channel has depth zero, so it answers `None` rather than
    /// refusing; that is the CUDA twin's behaviour too, and it is the reason
    /// there is no channel lookup on this path.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for whatever the read said.
    pub fn take(&mut self, channel: u32) -> Result<Option<Vec<u8>>> {
        if self.depth(channel) == 0 {
            return Ok(None);
        }
        let head = self.cursors[channel as usize].head;
        let cell = self.rings.read_cell(channel as usize, head)?;
        self.cursors[channel as usize].head = head + 1;
        Ok(Some(cell))
    }

    /// Channel `channel`'s cell at ring position `sequence`, as wire bytes,
    /// touching no cursor.
    ///
    /// **FOR DIFFING, NOT FOR SERVING.** The parity test reads every slot of
    /// every ring after every fire, because comparing only what was drained
    /// would miss a program that wrote the right value into the wrong slot.
    ///
    /// The channel is looked up before the read even though nothing is
    /// converted: [`Rings::read_cell`] answers an unknown index with the ring
    /// vocabulary, and the caller asked with a channel number.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an unknown channel, and whatever the read said.
    pub fn peek(&self, channel: u32, sequence: u64) -> Result<Vec<u8>> {
        self.shape_of(channel)?;
        self.rings.read_cell(channel as usize, sequence)
    }

    /// What this instance's descriptor ports hold right now.
    ///
    /// **THE COMMITTED FRONT, WHICH IS THE CELL THE GUEST'S OWN PASS TAKES.**
    /// A port's value for THIS fire is the cell at `head` — the same address
    /// [`Prepared::refresh`] publishes to the emitted kernel as
    /// `committed_cell` — so the shell's read and the guest's take are one
    /// value. Nothing is consumed: the pass's own commit advances `head` for
    /// every port [`Port::consumes`](driver::tensor_ir::registry::Port::consumes)
    /// names, and draining here as well would spend two cells per fire.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a port naming a channel this instance does not
    /// carry or holding a non-integer cell, and whatever the read said.
    pub fn envelope(&self, plan: &ExecPlan) -> Result<Envelope> {
        ports::resolve(plan, &self.rings, &self.cursors, &self.shapes)
    }

    /// Point one intrinsic at a device buffer, for every stage of this
    /// instance.
    ///
    /// **EVERY STAGE, BECAUSE A PROGRAM IS ONE FIRE.** The side tables are
    /// per-stage buffers, but `logits` means the same buffer to a prologue and
    /// to an epilogue: binding one stage would leave the other reading an
    /// argument slot nobody filled.
    ///
    /// The binding SURVIVES a fire — [`Session::fire`] resets the pending
    /// flags, the scratch and the status word, and nothing else — so a caller
    /// that rebinds the same buffer every fire and one that binds it once
    /// behave the same.
    ///
    /// **THE ROW GEOMETRY IS NOT AN ARGUMENT HERE, AND THAT IS THE PLATFORM
    /// AGAIN.** The CUDA twin takes `(base, storage, width, row_stride,
    /// row_offset)` and writes five side tables, because a CUDA kernel is
    /// handed a raw `u64` device address and has to be told how to walk it.
    /// Metal binds an object: `base` is the allocation and `offset` is the
    /// byte the intrinsic starts at, which is what
    /// `setBuffer:offset:atIndex:` takes, and the row walk is a property of
    /// the buffer the encoder bound rather than of a number in a table. Any
    /// caller that used to pass a `row_offset` passes it as bytes in `offset`.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for an intrinsic past the side tables' pitch, and
    /// whatever the binds said.
    pub fn bind_intrinsic(
        &mut self,
        intrinsic: IntrinsicId,
        base: &Buffer,
        offset: u64,
    ) -> Result<()> {
        for prepared in self.prepared.iter_mut().flatten() {
            prepared.bind_intrinsic(intrinsic, base, offset)?;
        }
        self.bound |= 1u64 << (intrinsic as u32);
        Ok(())
    }

    /// Run every stage of `compiled` as ONE fire.
    ///
    /// **NO STREAM ARGUMENT, BECAUSE THERE IS NO STREAM.** The CUDA twin takes
    /// a context to pull a stream off and synchronizes it once per stage;
    /// this one takes the device and the pipeline cache because that is what
    /// [`Prepared::launch_region`] needs to open a command buffer, and the
    /// wait that used to be `context.synchronize()` happens inside it, once
    /// per region. The per-stage ordering the CUDA comment justified is
    /// therefore stronger here, not weaker: the next stage's regions may read
    /// a channel cell this one wrote through the shared rings, and that write
    /// is already complete before `launch_region` returns.
    ///
    /// **A STATUS, NOT A BOOLEAN.** The CUDA twin reads one `u32` its kernel
    /// started at 1 and CLEARS to refuse, so every refusal it can express is
    /// [`Fired::Declined`]. The M1 kernel writes a sixteen-byte
    /// [`driver::Status`] — a state, a fault class and a guard site — so this
    /// half can tell the two apart and does:
    ///
    /// ```text
    /// State::Committed   the stage's commit region ran        -> commit
    /// State::Retry       a readiness guard inside the kernel  -> Declined
    /// State::Fault       a guard refused, with a class        -> Faulted
    /// State::Unset       nothing wrote a status at all        -> Faulted
    /// State::Running     the kernel did not reach its end     -> Faulted
    /// ```
    ///
    /// The mapping is not this file's opinion: it is
    /// [`driver::StatusOutcome::of`], the same three-way verdict the host half
    /// reads, and the sentence a fault carries is
    /// [`driver::report_status`] — which composes `describe_fault`'s class and
    /// channel with the diagnosis and the guard site. (`describe_fault` alone
    /// answers a struct with no `Display`; reaching for it directly would mean
    /// re-writing the formatter `driver` already ships, and then a fault would
    /// read differently depending on which shell printed it.)
    ///
    /// A device fault poisons the instance, because that is what
    /// [`Fired::Faulted`] promises: a guard that refused this fire refuses the
    /// next one from the same inputs, and the CUDA half's inability to see the
    /// difference is the reason it had to call such a fire merely declined.
    ///
    /// The last stage to speak wins nothing: the verdict is folded over every
    /// stage, worst-first, and every stage still launches. A refused fire
    /// costs what a running one does.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the compiled stages and their plans are not
    /// parallel, and whatever the launches and reads said. A program that
    /// merely refuses is [`Fired::Declined`] or [`Fired::Faulted`], not an
    /// error, and a program whose inputs are not there is [`Fired::Blocked`].
    pub fn fire(
        &mut self,
        device: &Context,
        pipelines: &Pipelines,
        compiled: &Compiled,
        plan: &ExecPlan,
    ) -> Result<Fired> {
        if self.poisoned {
            return Ok(Fired::Faulted("instance is poisoned".to_string()));
        }
        stages_and_plans_agree(compiled)?;

        // AN UNBOUND INTRINSIC IS AN ARGUMENT SLOT THE KERNEL READS ANYWAY,
        // and a Metal buffer argument nobody set is not a stated error — the
        // command buffer either reads whatever the encoder last left at that
        // index or dies with a page fault and takes the queue with it. So the
        // one thing the host can check before the launch, it checks.
        if plan.needs_logits && self.bound & (1u64 << (IntrinsicId::Logits as u32)) == 0 {
            return Err(Fault::program(
                "program::session",
                "this program reads the `logits` intrinsic and no buffer has been \
                 bound to it; the emitted kernel reads the argument slot regardless \
                 of whether anything was ever encoded into it",
            ));
        }
        // THE DRAFT COLUMN GETS ITS OWN GUARD, BECAUSE IT IS ITS OWN BUFFER
        // (palo C3b). `needs_mtp_logits` was a flag nothing checked while
        // there was one rectangle to bind; now the shell binds `MtpLogits` at
        // the `mtp` export and a load whose plan declares none binds nothing,
        // so a program that reads drafts against a headless model would take
        // the same unset-argument read the line above exists to prevent.
        if plan.needs_mtp_logits && self.bound & (1u64 << (IntrinsicId::MtpLogits as u32)) == 0 {
            return Err(Fault::program(
                "program::session",
                "this program reads the `mtp_logits` intrinsic and no buffer has \
                 been bound to it; a model whose text declares no `mtp` export has \
                 no draft column for it to point at",
            ));
        }

        // ── Readiness. THE SAME GATE THE HOST INTERPRETER OPENS. ──
        //
        // Read off the program's declared per-channel requirement, in channel
        // order, and answering with the FIRST channel that fails — because
        // `driver::step` does exactly that and a caller retries on the name.
        if let Some(blocked) = self.blocked_channel(plan) {
            return Ok(Fired::Blocked(blocked));
        }

        // The highest channel index a per-channel fault class can encode, as
        // `driver::describe_fault` spells the bound: a class occupies
        // `base ..= base + max_channel`, so the number is the last valid
        // index and not the count.
        let max_channel = u32::try_from(self.shapes.len().saturating_sub(1)).unwrap_or(u32::MAX);

        let mut verdict = Verdict::Committed;
        for (index, stage) in compiled.stages.iter().enumerate() {
            let Some(prepared) = self.prepared.get_mut(index).and_then(Option::as_mut) else {
                // A stage with nothing to launch needs no buffers and no
                // launch. Not an empty case: the adapter prologue is exactly
                // this shape.
                continue;
            };
            prepared.refresh(&self.rings, &self.cursors)?;
            for region in stage.regions.iter() {
                prepared.launch_region(device, pipelines, &self.rings, region)?;
            }
            // No synchronize between stages: `launch_region` has already
            // waited on the command buffer it committed, so this stage's
            // writes to the shared rings are visible to the next stage's
            // `refresh` and to the read on the next line.
            verdict = verdict.worse(verdict_of(index, prepared.status()?, max_channel));
        }

        match verdict {
            Verdict::Faulted(why) => {
                self.poisoned = true;
                return Ok(Fired::Faulted(why));
            }
            Verdict::Declined => return Ok(Fired::Declined),
            Verdict::Committed => {}
        }

        match self.commit(plan) {
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

    /// The first channel whose declared requirement a fire right now would
    /// not meet, or `None` when this instance is ready to fire.
    ///
    /// **THE GATE, ASKED WITHOUT FIRING.** [`Session::fire`] opens it for
    /// itself, and that is enough for a program fired on its own. It is not
    /// enough at a model fire's boundary: an epilogue is fired AFTER the
    /// forward has run and written the lane's KV, so discovering there that
    /// its rings are not ready would leave a fire the caller cannot retry —
    /// the tokens are in the cache and the guest's pass never happened. So
    /// the attachment path asks first, over every attached instance, and
    /// refuses the whole fire before anything launches.
    ///
    /// Same arithmetic, one implementation: [`Session::fire`] calls this.
    #[must_use]
    pub fn blocked_channel(&self, plan: &ExecPlan) -> Option<u32> {
        for channel in 0..self.shapes.len() {
            let readiness = plan
                .package
                .channels
                .get(channel)
                .and_then(|declared| declared.readiness);
            let live = self.depth(channel as u32);
            let capacity = u64::from(self.shapes[channel].capacity);
            let ready = match readiness {
                Some(Direction::NeedsFull) => live != 0,
                Some(Direction::NeedsEmpty) => live < capacity,
                None => true,
            };
            if !ready {
                return Some(channel as u32);
            }
        }
        None
    }

    /// Advance the cursors of a fire that ran.
    ///
    /// **THIS IS `driver::program::step`'s COMMIT, OVER THIS SHELL'S OWN
    /// CURSORS**, and it is transcribed rather than called because the host's
    /// operates on `ChannelState` — a host ring with a host mutex — and this
    /// one operates on a device ring's sequence numbers. The arithmetic is the
    /// same three lines and the parity test is what holds it that way:
    ///
    /// * a take advances the head, but only when the ring held something;
    /// * a put advances the tail, and overflows the ring rather than wrapping
    ///   past the head, which is an instance that can never be trusted again;
    /// * the capacity check counts the take's credit, so a loop-carried
    ///   channel — taken and put in one fire, which is every decode loop —
    ///   commits at capacity 1.
    ///
    /// The cells themselves are already where they belong: the kernel wrote
    /// each put into the pending cell, which is the cell at `tail`, and on
    /// shared storage that write is already visible to this function.
    fn commit(&mut self, plan: &ExecPlan) -> std::result::Result<(), String> {
        let mut next = self.cursors.clone();
        for (channel, cursor) in self.cursors.iter().enumerate() {
            if cursor.tail < cursor.head {
                return Err(format!("channel {channel}: tail precedes head at commit"));
            }
            let mut used = cursor.tail - cursor.head;
            let capacity = u64::from(self.shapes[channel].capacity);
            if plan.takes_channel(channel as u32) && used != 0 {
                next[channel].head = cursor.head + 1;
                used -= 1;
            }
            if plan.puts_channel(channel as u32) {
                if used >= capacity {
                    return Err(format!(
                        "channel {channel}: put overflows capacity {capacity} at commit"
                    ));
                }
                next[channel].tail = cursor.tail + 1;
            }
        }
        self.cursors = next;
        Ok(())
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

/// What one stage's status says about the whole fire, before the stages are
/// folded together.
///
/// Not public: [`Fired`] is the vocabulary a caller reads, and it also carries
/// `Blocked`, which no stage can say — a block is decided on the host before
/// anything launches.
#[derive(Clone, Debug)]
enum Verdict {
    /// The stage's commit region ran.
    Committed,
    /// The stage refused from inside and the cursors must not move.
    Declined,
    /// The stage faulted, and this is `driver`'s own sentence for it.
    Faulted(String),
}

impl Verdict {
    /// The worse of two stage verdicts: a fault beats a decline beats a
    /// commit.
    ///
    /// **THE FOLD IS ORDERED, NOT `AND`ED.** The CUDA twin folds booleans with
    /// `&=`, which loses nothing because it has one kind of refusal. With two,
    /// the order matters: a fire where stage 0 faulted and stage 1 merely
    /// declined is a poisoned instance, and answering [`Fired::Declined`]
    /// because the last stage said so would invite the caller to retry
    /// something that cannot succeed.
    fn worse(self, other: Verdict) -> Verdict {
        match (self, other) {
            (fault @ Verdict::Faulted(_), _) => fault,
            (_, fault @ Verdict::Faulted(_)) => fault,
            (Verdict::Declined, _) | (_, Verdict::Declined) => Verdict::Declined,
            (Verdict::Committed, Verdict::Committed) => Verdict::Committed,
        }
    }
}

/// One stage's [`driver::Status`], read as a [`Verdict`].
///
/// `dispatched` is passed as `true` unconditionally, and that is a statement
/// rather than a shortcut: this function is only reached after every one of
/// the stage's regions came back from [`Prepared::launch_region`], which does
/// not return until the command buffer it committed has completed. The
/// "prepared but never encoded" diagnosis therefore cannot arise here — the
/// path that would produce it returns a [`Fault`] from `launch_region`
/// instead, and calling it a status would turn a shell error into a guest
/// refusal.
fn verdict_of(stage: usize, status: driver::Status, max_channel: u32) -> Verdict {
    match status.state() {
        // **STATE ONE IS THE COMMIT ON THIS PATH, AND THAT IS THE M2
        // KERNEL'S CONTRACT RATHER THAN A LOOSE READING.**
        // `driver::StatusOutcome::of` reads `Running` as "the kernel started
        // and did not reach its end", because the plane it was written for
        // ends every pass with a DEVICE commit kernel that raises the word
        // to `Committed`. Build log 15 and 18 ruled readiness and commit
        // host-side on both planes, and the M2 fused kernel the Metal
        // emitter writes has no commit step at all: `emit_fused_region`
        // opens with `if (gid != 0 || status->state != 1) return;`, runs its
        // ops, and re-reads `status->state != 1` after each — so the word
        // comes back at ONE exactly when every op ran and none refused, and
        // at THREE when `m1_fault_op` raised it. Reading one as a failure
        // faults every correct pass; that is the first thing this plane's
        // parity gate found.
        Some(driver::State::Running) => Verdict::Committed,
        Some(driver::State::Committed) => Verdict::Committed,
        Some(driver::State::Retry) => Verdict::Declined,
        // `Unset` is the word nothing wrote, which on this plane means the
        // status reservation was clobbered — `Session::fire` writes ONE into
        // it before every launch.
        Some(driver::State::Fault | driver::State::Unset) | None => Verdict::Faulted(format!(
            "stage {stage}: {}",
            driver::report_status(status, true, max_channel)
        )),
    }
}

/// The wire cells a host-half instance's rings hold, as [`Session::bind`]
/// takes them.
///
/// **"CURSORS INITIALISED FROM THE HOST-SIDE STATE, OR FRESH" IS ONE CALL.**
/// A fresh instance is `bind(.., &[], ..)`; an instance the host half already
/// carries — because the engine ran its prologue there, or because a previous
/// shell owned it — is `bind(.., &seeds_of(interp, plan), ..)`. There is no
/// second seeding rule, and no way for the two to disagree about what a cell
/// is: both sides speak wire bytes, and on this plane so does the ring.
///
/// Cells are returned oldest first, so republishing them reproduces the ring's
/// order. A channel whose ring is empty contributes nothing.
#[must_use]
pub fn seeds_of(interp: &driver::InterpInstance, plan: &ExecPlan) -> Vec<(u32, Vec<u8>)> {
    let mut seeds = Vec::new();
    for (channel, ring) in interp.channels.iter().enumerate() {
        let declared = match plan.package.channels.get(channel) {
            Some(declared) => declared,
            None => continue,
        };
        let dtype = driver::concrete_dtype(declared.dtype);
        let numel = declared
            .shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        for sequence in ring.head()..ring.tail() {
            let mut wire = vec![0u8; driver::wire_cell_bytes(dtype, numel)];
            driver::encode_wire(&ring.decode_sequence(sequence), &mut wire);
            seeds.push((channel as u32, wire));
        }
    }
    seeds
}

/// The pairing every stage's [`Prepared`] depends on.
///
/// A fire builds one `Prepared` per LAUNCHING stage from `compiled.plans[i]`,
/// so the two arrays have to be parallel and each launching stage has to be
/// the one its plan describes. A plan that is off by one sizes a stage's
/// scratch from someone else's value types, indexes the lane table with
/// someone else's bindings and strides the params by someone else's op count —
/// none of which faults on the device.
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

/// The decision [`stages_and_plans_agree`] makes, over the facts it reads.
///
/// Projected to signatures and "launches?" first, because a compiled region
/// owns a loaded pipeline state, and a precondition nobody can test without a
/// device is a precondition nobody tests.
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
        // Only a LAUNCHING stage is prepared, so only a launching stage's
        // signature has to match. A stage with no regions carries nothing to
        // compare.
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
    use super::{Verdict, verdict_of};

    /// The served shape, and the one a "single stage only" guard used to
    /// refuse outright: an adapter prologue that launches nothing beside a
    /// sampling epilogue that does. Two stages, two plans, one fire.
    #[test]
    fn an_adapter_prologue_and_a_sampling_epilogue_are_one_fire() {
        super::stage_plans_are_parallel(&[(0xa11, false), (0xb22, true)], &[0xa11, 0xb22])
            .expect("the plans are parallel and the launching stage is its own");
    }

    /// Two stages that both launch are served too — each gets its own
    /// `Prepared` and they share the commit.
    #[test]
    fn two_launching_stages_each_get_their_own_plan() {
        super::stage_plans_are_parallel(&[(0xa11, true), (0xb22, true)], &[0xa11, 0xb22])
            .expect("both launch, both are prepared from their own plan");
    }

    /// A plan array that is not parallel is refused before anything is
    /// prepared: the fire pairs by index and has nothing else to pair on.
    #[test]
    fn stages_without_a_plan_apiece_are_refused() {
        let refusal = super::stage_plans_are_parallel(&[(0xa11, true), (0xb22, true)], &[0xa11])
            .expect_err("two stages, one plan");
        let text = format!("{refusal}");
        assert!(text.contains("2 compiled stage"), "names how many: {text}");
        assert!(text.contains("1 plan"), "and how many of the other: {text}");
    }

    /// A launching stage paired with somebody else's plan is refused, which is
    /// the failure nothing on the device would fault on.
    #[test]
    fn a_launching_stage_paired_with_the_wrong_plan_is_refused() {
        let refusal =
            super::stage_plans_are_parallel(&[(0xa11, true), (0xb22, true)], &[0xa11, 0xdead])
                .expect_err("stage 1 is not what plan 1 describes");
        let text = format!("{refusal}");
        assert!(text.contains("stage 1"), "names which: {text}");
        assert!(
            text.contains("scratch") || text.contains("descriptors"),
            "and what would go wrong: {text}"
        );
    }

    /// A stage that launches nothing is never prepared, so its signature is
    /// never compared — demanding a match would refuse the adapter prologue
    /// over a plan the fire does not read.
    #[test]
    fn a_stage_that_launches_nothing_is_not_compared() {
        super::stage_plans_are_parallel(&[(0xa11, false)], &[0xdead])
            .expect("nothing is prepared from it");
    }

    /// The status the CUDA twin cannot see: a state-3 fault is `Faulted` and
    /// its sentence carries `driver`'s own class name, not a bare number. This
    /// is the whole reason the boolean commit slot became a sixteen-byte
    /// status, so it is the thing worth pinning.
    #[test]
    fn a_kernel_fault_is_faulted_and_names_its_class() {
        // `M1_NOT_FULL` is `0x400` and is per-channel, so `0x401` is that
        // class on channel 1.
        let status = driver::Status {
            state: 3,
            fault: 0x401,
            reserved0: 0,
            reserved1: 0,
        };
        let Verdict::Faulted(why) = verdict_of(2, status, 3) else {
            panic!("state 3 is a fault, not a decline");
        };
        assert!(why.contains("stage 2"), "names which stage: {why}");
        assert!(why.contains("M1_NOT_FULL"), "and the class: {why}");
        assert!(why.contains("channel 1"), "and the channel: {why}");
    }

    /// A readiness guard the kernel observed for itself is `Retry`, and this
    /// half must NOT poison an instance over it — it is exactly the CUDA
    /// twin's cleared commit slot and the cursors are meant to stay put so the
    /// caller can fire again.
    #[test]
    fn a_kernel_readiness_miss_is_a_decline_and_not_a_fault() {
        let status = driver::Status {
            state: 2,
            fault: 0x480,
            reserved0: 0,
            reserved1: 0,
        };
        assert!(matches!(verdict_of(0, status, 3), Verdict::Declined));
    }

    /// A stage whose status was never written did not run, and a fire that
    /// committed cursors off an unwritten status would publish whatever the
    /// host guessed. The CUDA twin's word starts at 1 and would read as a
    /// commit here; this is the case the richer status buys.
    #[test]
    fn a_status_nothing_wrote_is_a_fault_rather_than_a_commit() {
        let status = driver::Status::default();
        assert!(matches!(verdict_of(0, status, 0), Verdict::Faulted(_)));
    }

    /// The fold is ordered: a fault anywhere in the fire outranks a decline
    /// anywhere else, in either arrival order, so a caller is never told to
    /// retry a poisoned instance.
    #[test]
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
