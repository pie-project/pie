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
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder, MTL4VisibilityOptions,
    MTLComputePipelineState, MTLDevice, MTLSharedEvent, MTLSize, MTLStages,
};

use super::context::{Context, describe};
use super::feedback::Feedbacks;
use super::heap::Slot;
use super::tables::Tables;
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
    pub fn run<F>(&mut self, encode: F) -> Result<()>
    where
        F: FnOnce(&mut StepEncoder<'_>) -> Result<()>,
    {
        self.preflight()?;
        let allocator = self.allocator();
        allocator.reset();
        let buffer = self.encode_one(allocator, encode)?;
        let value = self.commit(std::slice::from_ref(&buffer));
        self.await_value(value)
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
    pub fn run_parallel<F>(&mut self, count: usize, mut encode: F) -> Result<()>
    where
        F: FnMut(usize, &mut StepEncoder<'_>) -> Result<()>,
    {
        self.preflight()?;
        if count == 0 {
            return Ok(());
        }
        let allocator = self.allocator();
        allocator.reset();

        let mut buffers = Vec::with_capacity(count);
        for index in 0..count {
            buffers.push(self.encode_one(allocator, |step| encode(index, step))?);
        }
        let value = self.commit(&buffers);
        self.await_value(value)
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
    pub fn run_segments<F, B>(&mut self, count: usize, mut encode: F, mut between: B) -> Result<()>
    where
        F: FnMut(usize, &mut StepEncoder<'_>) -> Result<()>,
        B: FnMut(usize) -> Result<()>,
    {
        self.preflight()?;
        for index in 0..count {
            let allocator = self.allocator();
            // Legal every time round for the same reason as in `run`, and
            // resetting per segment rather than once is what keeps a long
            // model from growing the allocator by its segment count.
            allocator.reset();
            let buffer = self.encode_one(allocator, |step| encode(index, step))?;
            let value = self.commit(std::slice::from_ref(&buffer));
            self.await_value(value)?;
            between(index)?;
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
