//! Encoding a step, committing it, and waiting for it with a bound.
//!
//! A step is an allocator reset, one or more encoded command buffers, a
//! commit, and a wait on a shared event. The shape is the C++ shell's
//! `encode_one_command_buffer` and `await_event`, with the two things that
//! were comments there made into types here.
//!
//! # Three shapes, and what separates them
//!
//! [`Stepper::run`] is one command buffer. [`Stepper::run_parallel`] is N
//! buffers in ONE `commit:count:options:` under ONE event signal -- the
//! buffers race each other on purpose, so what it buys is `N - 1` submissions
//! and what it demands is that the chunks be mutually hazard-free. Metal
//! gives no ordering between command buffers in a batch and a barrier only
//! orders dispatches inside the encoder that issued it, so nothing here can
//! check that requirement; it is the caller's.
//!
//! [`Stepper::run_segments`] is the opposite trade and the reason both exist:
//! N buffers committed and waited for one at a time, with a host callback
//! between them. It is the only one of the three where the host can read what
//! the GPU just computed and change what it reads next, because the other two
//! commit everything before waiting for anything. That is right when the host
//! has nothing to add and impossible when it does -- for expert paging what
//! the host holds is which weights exist at all. It costs a submit and a
//! completion wait per segment, so splitting a step that does not need the
//! host is a straight loss.
//!
//! # The wait has a bound, and running out of it is terminal
//!
//! `waitUntilSignaledValue:` with no timeout is how the C++ shell spent
//! twenty-two minutes silent inside a bare retry loop. The bound here is
//! deliberately far past any real step -- the slowest measured on this
//! machine, a 192-token prefill through a 30B mixture, is about 200 ms -- so
//! reaching it does not mean "slow", it means the GPU is not coming back.
//!
//! What happens then is the part worth stating: nothing. A command buffer
//! that has not signalled may still be executing, so its allocator cannot be
//! reset and the heap it reads cannot be freed. There is no recovery, so
//! [`Stepper`] latches into a wedged state and refuses every later step
//! rather than papering over it with a retry.

use std::ptr::NonNull;
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder,
    MTL4UpdateSparseBufferMappingOperation, MTL4VisibilityOptions, MTLBuffer,
    MTLComputePipelineState, MTLDevice, MTLHeap, MTLSharedEvent, MTLSize,
    MTLSparseTextureMappingMode, MTLStages,
};

use super::context::{Context, describe};
use super::elastic::{self, Arena, Elastic, Mappings, Need, Pressure};
use super::feedback::Feedbacks;
use super::heap::Slot;
use super::tables::Tables;
use super::timestamp::{Granularity, Timestamps};
use super::timing::Timing;
use crate::error::{Error, Result};

/// How long one probe of the completion wait lasts.
///
/// Split into probes rather than one long timeout so that a step which is
/// merely slow can be COUNTED as slow the moment it passes the first probe,
/// while the total is what decides to give up.
const WAIT_PROBE: Duration = Duration::from_secs(5);

/// How many probes before the step is declared not coming back.
///
/// Sixty seconds total, two orders of magnitude past the slowest real step.
const WAIT_PROBES: u32 = 12;

/// What a dispatch's barrier makes visible to the next one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    /// Order execution only, without flushing caches.
    ///
    /// The default, and measured to be the correct one: the placement heap is
    /// L2-coherent within a single encoder on this UMA part, so the consumer
    /// of a producer's heap write sees it without an explicit flush. Both the
    /// device-flush and execution-only sweeps landed within noise of each
    /// other, so the cheaper one is not a gamble taken for speed.
    #[default]
    ExecutionOnly,
    /// Flush caches to the device coherence point.
    Device,
}

impl From<Visibility> for MTL4VisibilityOptions {
    fn from(v: Visibility) -> Self {
        match v {
            Visibility::ExecutionOnly => Self::None,
            Visibility::Device => Self::Device,
        }
    }
}

/// The buffer addresses one dispatch reads.
///
/// A table, not a list of `setBuffer:` calls: MTL4 binds by GPU address, and
/// an address outlives the encoder it was bound in. That is what lets a table
/// be built once, before any step, and reused by a byte-identical command
/// buffer every token -- which is the whole reason the encode cost of a step
/// is flat in the number of tokens.
pub struct ArgumentTable {
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    capacity: usize,
}

impl ArgumentTable {
    /// A table with room for `capacity` buffer bindings.
    pub fn new(context: &Context, capacity: usize) -> Result<Self> {
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(capacity);
        let table = context
            .device()
            .newArgumentTableWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Create {
                what: "MTL4ArgumentTable",
                message: describe(&e),
            })?;
        Ok(Self { table, capacity })
    }

    /// How many bindings this table holds.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bind `slot` at `index`.
    ///
    /// Out of range is an error rather than a silent no-op. Metal's own
    /// behaviour past `maxBufferBindCount` is not a diagnostic, and a binding
    /// that did not happen surfaces as a kernel reading address zero -- which
    /// on this driver means a kernel reading whatever the last step left.
    pub fn bind(&self, index: usize, slot: &Slot<'_>) -> Result<()> {
        self.bind_address(index, slot.gpu_address())
    }

    /// Bind a raw GPU address at `index`.
    pub fn bind_address(&self, index: usize, address: u64) -> Result<()> {
        if index >= self.capacity {
            return Err(Error::Create {
                what: "argument table binding",
                message: format!("index {index} past the table's {} bindings", self.capacity),
            });
        }
        // SAFETY: `address` is a GPU address obtained from a buffer that the
        // heap keeps alive, and `index` is within the bind count the table was
        // created with. Metal validates neither.
        unsafe { self.table.setAddress_atIndex(address, index) };
        Ok(())
    }
}

impl std::fmt::Debug for ArgumentTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArgumentTable")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

/// The per-dispatch surface, live only while a step is being encoded.
pub struct StepEncoder<'a> {
    encoder: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    /// The thread limit of the pipeline currently set, or 0 if none is.
    max_threads: usize,
}

impl StepEncoder<'_> {
    /// Set the pipeline the next dispatch runs.
    pub fn set_pipeline(&mut self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.encoder.setComputePipelineState(pipeline);
        self.max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    }

    /// Set the table the next dispatch reads its addresses from.
    pub fn set_argument_table(&mut self, table: &ArgumentTable) {
        self.encoder.setArgumentTable(Some(&table.table));
    }

    /// Set the table built for `ordinal`, or refuse.
    ///
    /// A miss is an error rather than a skipped call. Skipping it leaves the
    /// PREVIOUS dispatch's table bound, so the kernel runs to completion over
    /// another dispatch's buffers and the step reports success -- the same
    /// failure shape as a dispatch with no pipeline.
    pub fn set_argument_table_for(&mut self, tables: &Tables, ordinal: u32) -> Result<()> {
        self.set_argument_table(tables.expect(ordinal)?);
        Ok(())
    }

    /// Dispatch `threads`, in threadgroups of `threadgroup`.
    ///
    /// Refuses a threadgroup wider than the pipeline allows. Metal does not:
    /// the dispatch is simply not performed, its output keeps whatever it
    /// held, and the step reports success. That is how a model that answers
    /// nonsense passes every check -- which is exactly how it went unnoticed
    /// in the C++ shell, where this is a printf that fires once per pipeline.
    pub fn dispatch(&mut self, threads: [usize; 3], threadgroup: [usize; 3]) -> Result<()> {
        if self.max_threads == 0 {
            return Err(Error::Create {
                what: "dispatch",
                message: "no pipeline is set; the dispatch would run the previous kernel or none"
                    .to_string(),
            });
        }
        let per_group = threadgroup[0] * threadgroup[1] * threadgroup[2];
        if per_group == 0 {
            return Err(Error::Create {
                what: "dispatch",
                message: "a threadgroup of no threads runs nothing".to_string(),
            });
        }
        if per_group > self.max_threads {
            return Err(Error::Create {
                what: "dispatch",
                message: format!(
                    "{per_group} threads a threadgroup, and the pipeline allows {}; \
                     Metal would skip this dispatch and report success",
                    self.max_threads
                ),
            });
        }
        self.encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: threads[0],
                height: threads[1],
                depth: threads[2],
            },
            MTLSize {
                width: threadgroup[0],
                height: threadgroup[1],
                depth: threadgroup[2],
            },
        );
        Ok(())
    }

    /// Order the next dispatch after the ones already encoded.
    pub fn barrier(&mut self, visibility: Visibility) {
        self.encoder
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                visibility.into(),
            );
    }

    /// Write a GPU timestamp into `timestamps` at `index`.
    ///
    /// Takes `&self` because a mark changes nothing the encoder tracks: it
    /// neither sets state a later dispatch reads nor orders anything, so it
    /// can sit between two `&mut self` calls without being one.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] if `index` is not below `timestamps.count()`.
    /// Metal's own behaviour past the end of a counter heap is undefined and
    /// unreported, and the C++ shell cannot check this at all -- its heap is
    /// a `void*`, so the bound is not available at the call site. Here it
    /// travels with the heap.
    pub fn mark_timestamp(
        &self,
        timestamps: &Timestamps,
        index: u32,
        granularity: Granularity,
    ) -> Result<()> {
        if index >= timestamps.count() {
            return Err(Error::OutOfRange {
                what: "timestamp index",
                offset: u64::from(index),
                bytes: 1,
                len: u64::from(timestamps.count()),
            });
        }
        // SAFETY: the write is `unsafe` because Metal does not bounds-check
        // `index`. The check above is that bound, against the count the heap
        // was created with, and `timestamps` is borrowed for the call so the
        // heap cannot be released while Metal is encoding against it.
        unsafe {
            self.encoder.writeTimestampWithGranularity_intoHeap_atIndex(
                granularity.into(),
                timestamps.heap(),
                index as usize,
            );
        }
        Ok(())
    }
}

/// Runs steps against a context, one at a time.
///
/// Synchronous: [`Stepper::run`] does not return until the GPU has signalled.
/// The allocator pair is still alternated, because the parity is what a
/// pipelined version needs and a synchronous one that ignores it would have to
/// grow the state back when it stops being synchronous.
pub struct Stepper<'ctx> {
    context: &'ctx Context,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// The event value the last committed step signals.
    committed: u64,
    /// Set once a wait ran out. There is no way back; see the module docs.
    wedged: bool,
    /// What the GPU said about the steps that have finished.
    feedback: Feedbacks,
    /// The newest step whose fault has already been raised, so it is raised
    /// exactly once.
    surfaced: u64,
    /// The queue sparse remappings go on, built on first use.
    ///
    /// Lazy because most steppers never remap anything, and a second command
    /// queue is not free -- it is a scheduler context in the driver.
    mappings: Option<Mappings>,
}

impl<'ctx> Stepper<'ctx> {
    /// Build a stepper for `context`.
    pub fn new(context: &'ctx Context) -> Result<Self> {
        let event = context.device().newSharedEvent().ok_or(Error::Create {
            what: "MTLSharedEvent",
            message: String::new(),
        })?;
        Ok(Self {
            context,
            event,
            committed: 0,
            wedged: false,
            feedback: Feedbacks::new(),
            surfaced: 0,
            mappings: None,
        })
    }

    /// How many steps have been committed.
    #[must_use]
    pub const fn steps(&self) -> u64 {
        self.committed
    }

    /// Whether a wait ran out, after which nothing more will run.
    #[must_use]
    pub const fn is_wedged(&self) -> bool {
        self.wedged
    }

    /// What the GPU reported about the steps that have finished.
    ///
    /// Lags by about a step: see [`super::feedback`]. Clone it to observe
    /// from elsewhere.
    #[must_use]
    pub const fn feedback(&self) -> &Feedbacks {
        &self.feedback
    }

    /// Encode one step, commit it, and wait for it.
    ///
    /// `encode` is handed the live encoder. Its error is returned as-is and
    /// the command buffer is still closed -- an encoder abandoned mid-step
    /// leaves Metal holding an open command buffer against an allocator this
    /// type is about to reset.
    pub fn run<F>(&mut self, encode: F) -> Result<Timing>
    where
        F: FnOnce(&mut StepEncoder<'_>) -> Result<()>,
    {
        self.preflight()?;
        let encode_begin = Instant::now();
        let allocator = self.allocator();
        allocator.reset();
        let buffer = self.encode_one(allocator, encode)?;
        // Read before the commit, so the encode half is the encode half. A
        // single reading either side of the commit would put the submission
        // in whichever of the two the caller was not looking at.
        let committed = Instant::now();
        let value = self.commit(std::slice::from_ref(&buffer));
        self.await_value(value)?;
        Ok(self.timing(value, committed - encode_begin, committed.elapsed()))
    }

    /// Encode `count` command buffers and submit them in ONE commit.
    ///
    /// The buffers execute with NO ordering guarantee relative to one another
    /// -- that is the point, and it is also the requirement: `encode` must
    /// produce chunks that are mutually hazard-free, because nothing here
    /// will serialise them. A barrier only orders dispatches within the
    /// encoder that issued it.
    ///
    /// One event signal fences the whole batch, so the wait is the same wait
    /// [`run`](Self::run) does and the timeline advances by one. What is
    /// saved is `count - 1` submissions, which is why this exists at all: a
    /// pass that splits into parallel chunks otherwise pays a submit per
    /// chunk for work that could have gone in one.
    ///
    /// The allocator is reset once, before the first buffer. Several command
    /// buffers may be drawn from one allocator; what `reset` requires is that
    /// every buffer previously drawn from it has completed, and the
    /// alternating pair plus the synchronous wait is what gives that.
    ///
    /// # Errors
    ///
    /// The first encode error, with the buffers encoded so far dropped and
    /// nothing committed -- a partial batch is work the caller did not ask
    /// for and cannot undo. Otherwise as [`run`](Self::run).
    pub fn run_parallel<F>(&mut self, count: usize, mut encode: F) -> Result<Timing>
    where
        F: FnMut(usize, &mut StepEncoder<'_>) -> Result<()>,
    {
        self.preflight()?;
        if count == 0 {
            // A zeroed timing, not a refusal. Nothing was encoded and nothing
            // ran, and every field of that is honestly zero.
            return Ok(Timing::default());
        }
        let encode_begin = Instant::now();
        let allocator = self.allocator();
        allocator.reset();

        let mut buffers = Vec::with_capacity(count);
        for index in 0..count {
            buffers.push(self.encode_one(allocator, |step| encode(index, step))?);
        }
        let committed = Instant::now();
        let value = self.commit(&buffers);
        self.await_value(value)?;
        Ok(self.timing(value, committed - encode_begin, committed.elapsed()))
    }

    /// Encode and run `count` segments IN ORDER, with the host between them.
    ///
    /// `between(i)` runs after segment `i` has completed on the GPU and
    /// before segment `i + 1` is encoded, so it may read what the segment
    /// computed and change what the next one reads. It runs after EVERY
    /// segment, the last included, because a caller that pinned something per
    /// segment needs somewhere to give the last pin back.
    ///
    /// That is the whole reason this exists, and the one thing neither
    /// [`run`](Self::run) nor [`run_parallel`](Self::run_parallel) can do:
    /// both commit everything before waiting for anything, which is right
    /// when the host has nothing to add and impossible when it does. For
    /// expert paging the host holds something the GPU cannot compute for
    /// itself -- which weights exist at all.
    ///
    /// The cost is a submit and a completion wait per segment, so splitting a
    /// step that does NOT need the host is a straight loss.
    ///
    /// # Errors
    ///
    /// From the first segment that fails. A segment that never finished
    /// leaves the host holding results that were never computed, so `between`
    /// is not called for it and the remaining segments are not encoded.
    pub fn run_segments<F, B>(
        &mut self,
        count: usize,
        mut encode: F,
        mut between: B,
    ) -> Result<Timing>
    where
        F: FnMut(usize, &mut StepEncoder<'_>) -> Result<()>,
        B: FnMut(usize) -> Result<()>,
    {
        self.preflight()?;
        let mut total = Timing::default();
        for index in 0..count {
            let encode_begin = Instant::now();
            let allocator = self.allocator();
            // Legal every time round for the same reason as in `run`, and
            // resetting per segment rather than once is what keeps a long
            // model from growing the allocator by its segment count.
            allocator.reset();
            let buffer = self.encode_one(allocator, |step| encode(index, step))?;
            let committed = Instant::now();
            let value = self.commit(std::slice::from_ref(&buffer));
            self.await_value(value)?;
            // `between` is the caller's, not the step's. Timing it here would
            // charge the GPU for host work that happens to sit between two
            // submissions, which is the opposite of what this split is for.
            let segment = self.timing(value, committed - encode_begin, committed.elapsed());
            total.extend(segment);
            between(index)?;
        }
        Ok(total)
    }

    /// Assemble the timing for a submission that has just completed.
    ///
    /// The GPU's own number is read here and not waited for. The feedback
    /// block is dispatched asynchronously, and the C++ spins up to 200 times
    /// at 50us for it -- 10 ms of sleeping per step, on the token path, for a
    /// number the step does not need. `None` says it has not arrived; a
    /// caller that wants it can ask [`super::Feedbacks::await_step`] for it
    /// with [`Timing::step`].
    fn timing(&self, value: u64, encode: Duration, gpu_exec: Duration) -> Timing {
        Timing {
            encode,
            gpu_exec,
            // Only when it describes THIS submission. The slot holds the most
            // recent report, which before this one lands is the previous
            // step's -- attributing that to this step would be a number that
            // is real, plausible, and about something else.
            gpu: self
                .feedback
                .latest()
                .filter(|report| report.step == value)
                .map(|report| report.gpu_time()),
            step: value,
        }
    }

    /// Attach memory to `buffer` until at least `bytes` of it is mapped.
    ///
    /// On the stepper, not on the buffer, because a remap has to be ordered
    /// against steps: the mapping waits for the last committed step and the
    /// next step waits for the mapping. The type that owns the timeline is
    /// the only one that can express that, and putting the operation
    /// elsewhere would mean a second counter that has to agree with this one.
    ///
    /// Idempotent below the current size, so a caller may ask on every step
    /// rather than tracking what it last asked for.
    ///
    /// # Errors
    ///
    /// If `bytes` is past the buffer's length, if the arena has no room under
    /// `pressure` for a request of this [`Need`], or if a placement heap is
    /// refused. In every case the buffer is left exactly as it was found,
    /// minus any tiles that did map -- which are real and stay charged.
    pub fn ensure(
        &mut self,
        buffer: &mut Elastic,
        bytes: u64,
        pressure: Pressure,
        need: Need,
    ) -> Result<()> {
        let through = self.remap(|context, schedule| {
            elastic::grow(context, buffer, bytes, pressure, need, schedule)
        })?;
        // The growth itself is not waited for -- see `remap_shrink` for why
        // only the unmap half has to be. What the buffer is told is where the
        // mapping lands, so that its destructor can wait for it rather than
        // freeing the heaps out from under it.
        if let Some(through) = through {
            buffer.fence_at(&self.event, through);
        }
        Ok(())
    }

    /// Attach memory to several buffers, or to none of them.
    ///
    /// A step that needs three pools grown cannot use two of them. Without
    /// this, growing them one at a time can leave the first two mapped and
    /// the third refused, and the caller is holding memory it cannot use and
    /// did not ask to keep. The whole ask is checked against the budget
    /// first, and a failure part-way rolls the earlier ones back to where
    /// they were.
    ///
    /// Duplicated buffers collapse to their largest target rather than
    /// summing, because two asks for the same buffer are one requirement
    /// stated twice -- summing them would refuse a batch that fits.
    ///
    /// # Errors
    ///
    /// As [`ensure`](Self::ensure), and nothing is left grown.
    pub fn ensure_all(
        &mut self,
        targets: &mut [(&mut Elastic, u64)],
        pressure: Pressure,
        need: Need,
    ) -> Result<()> {
        // No two entries can name the same buffer: `&mut Elastic` cannot
        // alias and `Elastic` is not `Clone`. The C++ collapses duplicates
        // here because it takes a list of `void*` handles, where the same
        // buffer twice is a sentence you can say. Here it is not.

        // Priced before anything is mapped. Growing them one at a time and
        // discovering the last one does not fit means unwinding work that has
        // already touched the GPU.
        let mut total = 0u64;
        for (buffer, bytes) in targets.iter() {
            if *bytes > buffer.len() {
                return Err(Error::Create {
                    what: "elastic growth",
                    message: format!(
                        "asked for {bytes} bytes of a buffer that is {} long",
                        buffer.len()
                    ),
                });
            }
            total = total.saturating_add(
                bytes
                    .next_multiple_of(elastic::TILE)
                    .saturating_sub(buffer.committed()),
            );
        }
        let arena_room = self.arena_headroom(targets, pressure, need)?;
        if total > arena_room {
            return Err(Error::Create {
                what: "elastic growth",
                message: format!(
                    "{total} bytes across {} buffers exceeds the {arena_room} available \
                     under {pressure:?} pressure for a {need:?} request",
                    targets.len()
                ),
            });
        }

        let prior: Vec<u64> = targets.iter().map(|(b, _)| b.committed()).collect();
        for position in 0..targets.len() {
            let bytes = targets[position].1;
            let grown = {
                let (buffer, _) = &mut targets[position];
                self.remap(|context, schedule| {
                    elastic::grow(context, buffer, bytes, pressure, need, schedule)
                })
            };
            match grown {
                Ok(Some(through)) => targets[position].0.fence_at(&self.event, through),
                Ok(None) => {}
                Err(error) => {
                    // Only reachable when the allocator itself refuses -- the
                    // price above is exact, so a shortfall here means the
                    // machine could not produce a heap it had budget for. Put
                    // each earlier buffer back exactly where it was, in
                    // reverse, so heaps come off in the order they went on.
                    for earlier in (0..position).rev() {
                        let (buffer, _) = &mut targets[earlier];
                        let _ = self.remap_shrink(buffer, prior[earlier]);
                    }
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Detach memory from `buffer` down to `bytes`.
    ///
    /// The heaps this empties are not freed until the GPU has been observed
    /// past the unmap -- see [`Arena`]. Asking for more than is mapped does
    /// nothing.
    ///
    /// # Errors
    ///
    /// If the wait for the unmap runs out, which wedges the context for the
    /// same reason a step's wait running out does: a mapping that may or may
    /// not have happened is not a state anything can be encoded against.
    pub fn trim(&mut self, buffer: &mut Elastic, bytes: u64) -> Result<()> {
        self.remap_shrink(buffer, bytes)
    }

    /// Declare what `buffer` cannot exist without, so pressure cannot clamp
    /// below it. See [`Need`].
    pub fn declare_mandatory(&mut self, buffer: &mut Elastic, bytes: u64) {
        elastic::declare_mandatory(buffer, bytes);
    }

    /// Give back every heap whose unmap the GPU has passed.
    ///
    /// Called automatically by [`trim`](Self::trim); public so a caller that
    /// wants the memory back at a moment of its choosing can ask, rather than
    /// waiting for the next trim to notice.
    pub fn collect(&self, arena: &Arena) {
        arena.collect(self.event.signaledValue(), self.context.residency());
    }

    /// Headroom for a multi-buffer ask, which is just the arena's.
    /// Headroom for a batch, refusing a batch that spans more than one arena.
    ///
    /// A batch is priced once against one budget. If two of the buffers draw
    /// on different arenas, that one price is a number about neither of them:
    /// it would check the whole batch against the first arena and let the
    /// second overrun silently. Refusing is not a limitation of the caller --
    /// two arenas are two independent budgets and there is no such thing as
    /// an atomic growth across both.
    fn arena_headroom(
        &self,
        targets: &[(&mut Elastic, u64)],
        pressure: Pressure,
        need: Need,
    ) -> Result<u64> {
        let mut found: Option<Arena> = None;
        for (buffer, _) in targets {
            let Some(arena) = buffer.arena() else {
                continue;
            };
            match &found {
                Some(first) if !first.is(&arena) => {
                    return Err(Error::Create {
                        what: "elastic growth",
                        message: "a batch growth spans two arenas, which cannot be \
                                  priced or rolled back as one"
                            .to_owned(),
                    });
                }
                Some(_) => {}
                None => found = Some(arena),
            }
        }
        Ok(found.map_or(0, |arena| arena.headroom(pressure, need)))
    }

    /// Run `body` with a scheduler that puts each remap on the timeline.
    fn remap<T>(
        &mut self,
        body: impl FnOnce(&Context, &mut elastic::Map<'_>) -> Result<T>,
    ) -> Result<T> {
        self.preflight()?;
        if self.mappings.is_none() {
            self.mappings = Some(Mappings::new(self.context)?);
        }
        let queue = &self.mappings.as_ref().expect("just built").queue;
        let event = &*self.event;
        let committed = &mut self.committed;
        let step_queue = self.context.queue();

        let mut schedule = |buffer: &ProtocolObject<dyn MTLBuffer>,
                            heap: &ProtocolObject<dyn MTLHeap>,
                            first_tile: u64,
                            tiles: u64,
                            heap_tile: u64|
         -> u64 {
            issue(
                queue,
                step_queue,
                event,
                committed,
                buffer,
                Some(heap),
                MTLSparseTextureMappingMode::Map,
                first_tile,
                tiles,
                heap_tile,
            )
        };
        body(self.context, &mut schedule)
    }

    /// The shrink half, which needs no heap and must be waited for.
    fn remap_shrink(&mut self, buffer: &mut Elastic, bytes: u64) -> Result<()> {
        self.preflight()?;
        if self.mappings.is_none() {
            self.mappings = Some(Mappings::new(self.context)?);
        }
        let through = {
            let queue = &self.mappings.as_ref().expect("just built").queue;
            let event = &*self.event;
            let committed = &mut self.committed;
            let step_queue = self.context.queue();
            elastic::shrink(buffer, bytes, &mut |target, first_tile, tiles| {
                issue(
                    queue,
                    step_queue,
                    event,
                    committed,
                    target,
                    None,
                    MTLSparseTextureMappingMode::Unmap,
                    first_tile,
                    tiles,
                    0,
                )
            })
        };
        // Waited for, unlike a grow. Mapping more is safe to leave in flight
        // because nothing reads a tile it did not ask for; unmapping is not,
        // because the heap is about to be handed back and the caller is
        // entitled to believe the bytes are gone when this returns.
        if let Some(through) = through {
            self.await_value(through)?;
            buffer.fence_at(&self.event, through);
            if let Some(arena) = buffer.arena() {
                arena.collect(self.event.signaledValue(), self.context.residency());
            }
        }
        Ok(())
    }

    /// Refuse before encoding anything, for the two reasons a step cannot run.
    fn preflight(&mut self) -> Result<()> {
        if self.wedged {
            return Err(Error::Create {
                what: "step",
                message: "this context was abandoned after a completion wait ran out".to_string(),
            });
        }
        // A faulted command buffer still reaches the signal, so the previous
        // step's wait returned Ok. This is the first moment its fault can be
        // known -- raised here rather than swallowed, once.
        self.raise_pending_fault()
    }

    /// The allocator this submission draws from.
    ///
    /// Safe because this stepper is synchronous: the work drawn from this
    /// allocator was waited for two submissions ago. The parity is what makes
    /// that sentence still true when the wait moves off the commit path.
    fn allocator(&self) -> &'ctx ProtocolObject<dyn MTL4CommandAllocator> {
        self.context.allocator(self.committed as usize)
    }

    /// Encode one closed command buffer against `allocator`.
    fn encode_one<F>(
        &self,
        allocator: &ProtocolObject<dyn MTL4CommandAllocator>,
        encode: F,
    ) -> Result<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>
    where
        F: FnOnce(&mut StepEncoder<'_>) -> Result<()>,
    {
        let command_buffer = self
            .context
            .device()
            .newCommandBuffer()
            .ok_or(Error::Create {
                what: "MTL4CommandBuffer",
                message: String::new(),
            })?;
        command_buffer.beginCommandBufferWithAllocator(allocator);
        // The heap was made resident once; this is what tells THIS command
        // buffer to use that set. Without it every address the argument table
        // holds is a page the GPU has not been told to keep.
        command_buffer.useResidencySet(self.context.residency());

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(Error::Create {
                what: "MTL4ComputeCommandEncoder",
                message: String::new(),
            })?;

        let mut step = StepEncoder {
            encoder: &encoder,
            max_threads: 0,
        };
        let encoded = encode(&mut step);

        // Closed on both paths. See `run`: an abandoned encoder outlives the
        // allocator reset that comes next.
        encoder.endEncoding();
        command_buffer.endCommandBuffer();
        encoded?;
        Ok(command_buffer)
    }

    /// Commit `buffers` as one submission and signal the timeline once.
    ///
    /// Returns the value that submission will signal.
    fn commit(&mut self, buffers: &[Retained<ProtocolObject<dyn MTL4CommandBuffer>>]) -> u64 {
        let value = self.committed + 1;
        // The handler is built before the commit so it can tag itself with the
        // timeline point it describes; `_handler` keeps the block alive across
        // the call that copies it.
        let (_handler, options) = self.feedback.options(value);
        let mut pointers: Vec<NonNull<ProtocolObject<dyn MTL4CommandBuffer>>> =
            buffers.iter().map(|b| NonNull::from(&**b)).collect();
        // SAFETY: the pointer is to a live array of exactly `len` command
        // buffers, which outlive the call; `commit` reads the array and does
        // not retain it.
        unsafe {
            self.context.queue().commit_count_options(
                NonNull::from(pointers.as_mut_slice()).cast(),
                pointers.len(),
                &options,
            );
        }
        self.context
            .queue()
            .signalEvent_value(ProtocolObject::from_ref(&*self.event), value);
        self.committed = value;
        value
    }

    /// Raise a GPU fault reported since the last time one was raised.
    fn raise_pending_fault(&mut self) -> Result<()> {
        let Some(fault) = self.feedback.take_error_after(self.surfaced) else {
            return Ok(());
        };
        self.surfaced = fault.step;
        Err(Error::Create {
            what: "step",
            message: format!(
                "the GPU faulted on step {}: {}",
                fault.step,
                fault.error.unwrap_or_default()
            ),
        })
    }

    /// Wait for the event to reach `value`, or wedge.
    fn await_value(&mut self, value: u64) -> Result<()> {
        let probe_ms = u64::try_from(WAIT_PROBE.as_millis()).unwrap_or(u64::MAX);
        for _ in 0..WAIT_PROBES {
            if self.event.waitUntilSignaledValue_timeoutMS(value, probe_ms) {
                return Ok(());
            }
        }
        self.wedged = true;
        Err(Error::Create {
            what: "step",
            message: format!(
                "the GPU did not reach event {value} within {} ms; this context is abandoned \
                 because its command buffers may still be running",
                probe_ms * u64::from(WAIT_PROBES)
            ),
        })
    }
}

impl std::fmt::Debug for Stepper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stepper")
            .field("steps", &self.committed)
            .field("wedged", &self.wedged)
            .finish_non_exhaustive()
    }
}

/// Put one sparse remap on the shared timeline and say where it lands.
///
/// The bracket is the whole function. The mapping queue waits for the last
/// committed step, so no kernel that was already submitted can be reading a
/// tile while it moves; the step queue then waits for the remap, so nothing
/// submitted afterwards runs before it. The timeline advances by one, which
/// is why `committed` is taken by `&mut` -- a remap is a point on the same
/// counter as a step, and a second counter would be a second truth.
///
/// The wait on the mapping queue is skipped at value zero because a shared
/// event starts at zero and waiting for it would be satisfied immediately
/// anyway; issuing the wait costs a scheduling round-trip for nothing.
#[allow(clippy::too_many_arguments)]
fn issue(
    mapping_queue: &ProtocolObject<dyn MTL4CommandQueue>,
    step_queue: &ProtocolObject<dyn MTL4CommandQueue>,
    event: &ProtocolObject<dyn MTLSharedEvent>,
    committed: &mut u64,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    heap: Option<&ProtocolObject<dyn MTLHeap>>,
    mode: MTLSparseTextureMappingMode,
    first_tile: u64,
    tiles: u64,
    heap_tile: u64,
) -> u64 {
    let as_event = ProtocolObject::from_ref(event);
    if *committed != 0 {
        mapping_queue.waitForEvent_value(as_event, *committed);
    }

    let operation = MTL4UpdateSparseBufferMappingOperation {
        mode,
        bufferRange: NSRange {
            location: usize::try_from(first_tile).unwrap_or(usize::MAX),
            length: usize::try_from(tiles).unwrap_or(0),
        },
        heapOffset: usize::try_from(heap_tile).unwrap_or(0),
    };
    // SAFETY: the operation is a live `repr(C)` value for the duration of the
    // call, `count` is one and matches, and the buffer and heap outlive it --
    // the heap is borrowed from a `Chunk` the caller still owns.
    unsafe {
        mapping_queue.updateBufferMappings_heap_operations_count(
            buffer,
            heap,
            NonNull::from(&operation),
            1,
        );
    }

    *committed += 1;
    let value = *committed;
    mapping_queue.signalEvent_value(as_event, value);
    step_queue.waitForEvent_value(as_event, value);
    value
}
