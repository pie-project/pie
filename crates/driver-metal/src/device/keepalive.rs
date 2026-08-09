//! A background thread that keeps the GPU clock domain busy, as an RAII handle.
//!
//! [`Keepalive`] spawns a thread on a SEPARATE [`MTL4CommandQueue`] that
//! commits a compute-spin dispatch back to back with a bounded in-flight
//! depth and no per-command-buffer host wait, so the GPU clock domain never
//! gates between the main loop's per-token drains.
//!
//! # This is an experiment, not a fix
//!
//! The C++ header this is ported from says so and the sentence is worth
//! repeating rather than paraphrasing: this is the EXPERIMENT that measures
//! whether the per-token gap is entirely DVFS downclock -- does `gpu_exec`
//! reach the hot floor when the clocks are never allowed to drop? -- and it
//! is NOT a shippable fix. A resident loop that keeps real work on the GPU is
//! the fix. Burning a slice of the GPU on a kernel whose output is discarded
//! costs power and steals occupancy from the model, so nothing on the serving
//! path should hold one of these; it exists so the number can be measured.
//!
//! # The two minimums
//!
//! `depth` is clamped up to [`MIN_DEPTH`] and `threadgroups` up to
//! [`MIN_THREADGROUPS`], and each minimum is load-bearing for a different
//! reason. A depth of 1 would make the thread wait for command buffer `n`
//! before submitting `n+1`, which fully drains the queue between dispatches
//! -- exactly the idle window the keepalive exists to remove, so a depth of 1
//! measures nothing. A depth of 0 is worse: it is not a depth at all. Two is
//! the smallest depth that always leaves one command buffer in flight while
//! the next is being built. `threadgroups` of 0 dispatches no threads, so the
//! GPU is handed a command buffer that does nothing and the clocks drop
//! anyway; one threadgroup is the smallest grid that is real work.
//!
//! # Where this differs from the C++, deliberately
//!
//! Four defects in `RawMetalContext::start_keepalive` are not reproduced.
//!
//! 1. **Half-initialised state after a failed start.** The C++ assigns
//!    `ka_queue` before compiling the spin pipeline and before creating the
//!    argument table, then returns early if either fails. `ka_queue` is now
//!    non-nil, so a second `start_keepalive` takes the `else` branch and
//!    writes `spin_iters` through `ka_iters.contents` -- and `ka_iters` is
//!    still nil, because the first call never got that far. A failed start
//!    arms a null-pointer write in the next one. [`Keepalive::start`] builds
//!    every object into locals and only assembles them into a value at the
//!    end, so a failure returns [`Err`] having left nothing behind at all;
//!    there is no partially-built keepalive for a second call to find,
//!    because there is no shared slot for one to live in.
//! 2. **Failure that the caller cannot see.** The C++ reports both failures
//!    by printing to stderr and returning `void`, so a caller cannot
//!    distinguish a running keepalive from one that never started -- and a
//!    downclock measurement taken against a keepalive that silently failed to
//!    start is a measurement of nothing, reported as a result. [`Keepalive::start`]
//!    returns [`Result`]. Nothing here prints: this crate denies
//!    `clippy::print_stdout` and `clippy::print_stderr`, and a driver that
//!    writes to a serving process's stderr is a driver that decided on its
//!    caller's behalf what is worth reporting.
//! 3. **A thread with no lifetime relationship to what it borrows.** The C++
//!    thread captures `&I`, a reference to the context's `Impl`, and reads
//!    `I.dev`, `I.ka_queue`, `I.ka_alloc`, `I.ka_pso` and `I.ka_event` on
//!    every iteration. Nothing forces `stop_keepalive` to be called before
//!    `~RawMetalContext`, and `Impl`'s destructor does not join -- so
//!    destroying a context with the keepalive running releases the objects
//!    the thread is still committing through. Here the thread owns the Metal
//!    objects it uses (see [`Keepalive::start`] on the `Send` question) and
//!    [`Keepalive`]'s [`Drop`] joins it, so the thread cannot outlive the
//!    handle and there is no stop call to forget. That matches what this
//!    crate already does for [`super::Mapped`], [`super::External`] and
//!    [`super::Transient`], each of which turned a "call this afterwards"
//!    into a destructor.
//! 4. **A residency set rebuilt on every start.** The C++ creates a fresh
//!    `MTLResidencySet` on each `start_keepalive` even though the two buffers
//!    it covers are created once and never replaced, so every start/stop
//!    cycle leaves another set behind holding them resident. Here the set is
//!    created once, beside the buffers it covers, and is owned by the same
//!    value -- a rebuild cannot be requested because starting is
//!    construction, and construction is what makes the buffers.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLBuffer,
    MTLComputePipelineState, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor,
    MTLResourceOptions, MTLSharedEvent, MTLSize,
};

use crate::device::archive::Archives;
use crate::device::context::{Context, describe};
use crate::error::{Error, Result};
use crate::program::compile::Compiler;

/// The smallest in-flight depth the keepalive will run at.
///
/// Two, because two is the smallest number that keeps one command buffer
/// executing while the next is encoded. See the module docs: at a depth of
/// one the queue drains between every dispatch, which reintroduces the idle
/// window this exists to close.
pub const MIN_DEPTH: u32 = 2;

/// The smallest grid width the keepalive will run at.
///
/// One threadgroup. Zero dispatches nothing, and a command buffer that
/// dispatches nothing keeps the GPU no busier than an empty queue does.
pub const MIN_THREADGROUPS: u32 = 1;

/// Threads per threadgroup in the spin dispatch.
///
/// 64 -- one Apple GPU SIMD width, so every thread in a group is one
/// execution unit's worth of work and the grid is `threadgroups * 64`. Not
/// tunable: the point of the grid is duty, and duty is what `threadgroups`
/// and `spin_iters` are for.
pub const THREADS_PER_THREADGROUP: u32 = 64;

/// How long a single event wait is allowed to block, in milliseconds.
///
/// The same 5000 ms the C++ uses. It is a liveness bound, not a deadline: a
/// GPU that has not retired a trivial spin dispatch in five seconds is wedged,
/// and the thread stops rather than committing more work into it.
const WAIT_TIMEOUT_MS: u64 = 5000;

/// The entry point in [`SPIN_SOURCE`].
const SPIN_FUNCTION: &str = "ka_spin";

/// The spin kernel, as it is in the C++.
///
/// An LCG accumulator looped `iters` times, and then a comparison against a
/// value the accumulator will not take. The `if` is the load-bearing part: an
/// accumulator nothing ever reads is dead code, and the optimiser deletes the
/// loop that produced it -- leaving a kernel that dispatches and returns, and
/// a measurement of an idle GPU. Feeding `acc` to an `atomic_fetch_add` under
/// a branch that is never taken keeps the loop alive at the cost of one
/// compare per thread.
const SPIN_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void ka_spin(device atomic_uint* sink   [[buffer(0)]],
                    constant uint&      iters  [[buffer(1)]],
                    uint                tid    [[thread_position_in_grid]]) {
    uint acc = tid * 2654435761u + 1u;
    for (uint i = 0; i < iters; ++i) acc = acc * 1664525u + 1013904223u;
    if (acc == 0xFFFFFFFFu)  // never true in practice; defeats dead-code elimination
        atomic_fetch_add_explicit(sink, acc, memory_order_relaxed);
}
";

/// Everything the keepalive thread owns, moved onto it at spawn.
///
/// A struct rather than nine captured locals so the `Send` justification can
/// be written once, about a value whose every field is a Metal object, rather
/// than be implied nine times by a closure.
struct Owned {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// The atomic sink the kernel is allowed to write and nothing reads. Held
    /// only so it outlives the dispatches whose argument table names its
    /// address.
    _sink: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// The spin count the kernel reads. Held for the same reason as `_sink`.
    _iters: Retained<ProtocolObject<dyn MTLBuffer>>,
}

// SAFETY: `Retained<ProtocolObject<dyn MTL*>>` is `!Send` because objc2
// cannot know whether a given Objective-C class has thread affinity -- a
// `UIView` does, and releasing one off the main thread is undefined. Metal's
// objects do not. Every field here is a Metal object whose reference counting
// is documented as thread-safe, none of them is autoreleased into a
// per-thread pool by the calls that create them (each is a `new*`, owned on
// return), and none has main-thread affinity.
//
// The narrower question is the objects Metal documents as NOT safe for
// CONCURRENT use: `MTL4CommandAllocator`, `MTL4ArgumentTable` and the
// residency set's mutators. This is a `Send` impl, not a `Sync` one, so it
// grants transfer and not sharing: the value is moved into
// `std::thread::spawn`'s closure and the spawning thread cannot name it
// afterwards, so exactly one thread touches these objects at a time and the
// happens-before edge of the spawn orders the creating thread's writes before
// the running thread's reads. The keepalive queue, allocator, event,
// residency set, buffers and argument table are created for this thread and
// reachable from nowhere else; the device and the pipeline are shared with
// the caller, and both are documented thread-safe for concurrent use.
unsafe impl Send for Owned {}

/// A background GPU keepalive, stopped and joined when dropped.
///
/// Construction starts the thread and [`Drop`] stops it. There is no `stop`
/// method, deliberately: see the module docs -- the C++'s `stop_keepalive` is
/// a call that can be forgotten, and the thread it fails to join outlives the
/// Metal objects it is committing through.
pub struct Keepalive {
    /// Cleared by [`Drop`] to ask the thread to finish its current iteration.
    running: Arc<AtomicBool>,
    /// How many command buffers the thread has committed so far.
    committed: Arc<AtomicU64>,
    /// `None` only between [`Drop`]'s take and the end of the join.
    thread: Option<JoinHandle<()>>,
    spin_iters: u32,
    threadgroups: u32,
    depth: u32,
}

impl Keepalive {
    /// Start the keepalive thread on its own queue.
    ///
    /// `spin_iters` is the inner loop count per thread, which is the GPU duty
    /// of one dispatch; `threadgroups` is the grid width, which is its
    /// occupancy; `depth` is the number of command buffers allowed in flight
    /// at once. `depth` is clamped up to [`MIN_DEPTH`] and `threadgroups` to
    /// [`MIN_THREADGROUPS`] -- see the module docs for why neither minimum is
    /// arbitrary. The clamped values are what [`Keepalive::depth`] and
    /// [`Keepalive::threadgroups`] report, so a caller that passed 0 can see
    /// what it actually got rather than assume its request survived.
    ///
    /// The spin kernel is compiled HERE, on the calling thread, before the
    /// thread is spawned. That is not incidental: concurrent Metal pipeline
    /// compilation corrupts the process heap (see [`crate::program::Compiler`]'s
    /// module docs for the evidence), which is why every compilation in this
    /// crate goes through one process-wide mutex. Compiling through
    /// [`Compiler::compile`] puts this one behind that mutex too, and doing
    /// it before the spawn means the background thread never compiles
    /// anything -- so the keepalive cannot be the second compiler in a race
    /// with a load that is compiling model kernels.
    ///
    /// # Errors
    ///
    /// [`Error::Compile`] if the spin kernel does not build, and
    /// [`Error::Create`] if the device declines the queue, the allocator, the
    /// event, either buffer, the argument table or the residency set. Every
    /// one of those leaves nothing behind: the objects built so far are
    /// locals and are released on the way out, so a failed start is
    /// indistinguishable from one that was never attempted. That is the whole
    /// of C++ bug 1 -- see the module docs.
    pub fn start(
        context: &Context,
        spin_iters: u32,
        threadgroups: u32,
        depth: u32,
    ) -> Result<Self> {
        let threadgroups = clamp_threadgroups(threadgroups);
        let depth = clamp_depth(depth);

        // First, because it is the only step that takes a process-wide lock
        // and the only one that can be slow. Its own cache location, and an
        // empty one: `compile` is the uncached path by design (one source has
        // no batch to be keyed as), so an [`Archives`] with no directory says
        // that plainly instead of relying on it.
        let compiler = Compiler::with_archives(context, Archives::new(None))?;
        let pipeline = compiler.compile(context, SPIN_SOURCE, SPIN_FUNCTION)?;

        let device = context.device().retain();

        // Its own queue. The whole experiment is that this work does not
        // serialise against the main loop's steps: sharing the context's
        // queue would put the spin dispatches in the same timeline as the
        // tokens, which is a way of making the tokens slower rather than of
        // keeping the clocks up between them.
        let queue = device.newMTL4CommandQueue().ok_or(Error::Create {
            what: "keepalive MTL4CommandQueue",
            message: String::new(),
        })?;
        // Its own allocator, for the same reason and one more: this thread
        // resets its allocator on every iteration, and resetting one the main
        // loop is still encoding into is a use-after-free Metal does not
        // diagnose.
        let allocator = device.newCommandAllocator().ok_or(Error::Create {
            what: "keepalive MTL4CommandAllocator",
            message: String::new(),
        })?;
        // Its own event, so the depth bound is counted on a timeline nothing
        // else signals.
        let event = device.newSharedEvent().ok_or(Error::Create {
            what: "keepalive MTLSharedEvent",
            message: String::new(),
        })?;

        let sink = new_word_buffer(&device, "keepalive sink buffer")?;
        let iters = new_word_buffer(&device, "keepalive iters buffer")?;
        // SAFETY: `iters` is a shared-storage buffer of exactly one 4-byte
        // word, `contents()` is its host mapping, and nothing has been
        // dispatched against it yet -- this is the only writer and there is
        // no GPU work in flight to race with.
        unsafe { iters.contents().cast::<u32>().write(spin_iters) };

        let table_descriptor = MTL4ArgumentTableDescriptor::new();
        table_descriptor.setMaxBufferBindCount(2);
        let table = device
            .newArgumentTableWithDescriptor_error(&table_descriptor)
            .map_err(|e| Error::Create {
                what: "keepalive MTL4ArgumentTable",
                message: describe(&e),
            })?;
        // SAFETY: both addresses come from buffers this value owns and
        // outlive every dispatch that reads the table, and both indices are
        // below the bind count the table was just created with.
        unsafe {
            table.setAddress_atIndex(sink.gpuAddress(), 0);
            table.setAddress_atIndex(iters.gpuAddress(), 1);
        }

        // Its own residency set, covering exactly these two buffers. Not the
        // context's: these buffers are the keepalive's and go away with it,
        // and adding them to the context's set would make the experiment
        // leave residency behind after it stops.
        let residency_descriptor = MTLResidencySetDescriptor::new();
        let residency = device
            .newResidencySetWithDescriptor_error(&residency_descriptor)
            .map_err(|e| Error::Create {
                what: "keepalive MTLResidencySet",
                message: describe(&e),
            })?;
        residency.addAllocation(ProtocolObject::from_ref(&*sink));
        residency.addAllocation(ProtocolObject::from_ref(&*iters));
        residency.commit();
        residency.requestResidency();

        let owned = Owned {
            device,
            queue,
            allocator,
            event,
            residency,
            table,
            pipeline,
            _sink: sink,
            _iters: iters,
        };

        let running = Arc::new(AtomicBool::new(true));
        let committed = Arc::new(AtomicU64::new(0));
        let thread = std::thread::Builder::new()
            .name("metal-keepalive".to_string())
            .spawn({
                let running = Arc::clone(&running);
                let committed = Arc::clone(&committed);
                move || {
                    // A drain that timed out leaves command buffers running
                    // against every object in `owned`. Releasing them here
                    // is not a leak and not a stall -- it is the
                    // use-after-free the join exists to prevent, arriving
                    // through the timeout that bounds the join. Leaking is
                    // the safe half: it costs this process two buffers and a
                    // queue, and the alternative costs the machine.
                    if !spin(&owned, threadgroups, depth, &running, &committed) {
                        let _: &'static mut Owned = Box::leak(Box::new(owned));
                    }
                }
            })
            .map_err(|e| Error::Create {
                what: "keepalive thread",
                message: e.to_string(),
            })?;

        Ok(Self {
            running,
            committed,
            thread: Some(thread),
            spin_iters,
            threadgroups,
            depth,
        })
    }

    /// How many command buffers the thread has committed.
    ///
    /// The only externally visible proof that the thread is doing its job. A
    /// live thread is not evidence of anything -- one parked forever on its
    /// first event wait looks exactly like one submitting flat out -- so the
    /// counter is what a test can assert has moved past [`Keepalive::depth`],
    /// which is the point at which the depth bound has been reached and
    /// released at least once.
    ///
    /// Relaxed: it is a monotonic counter read for its value, and no other
    /// memory is being published through it.
    #[must_use]
    pub fn committed(&self) -> u64 {
        self.committed.load(Ordering::Relaxed)
    }

    /// The per-thread spin count each dispatch runs.
    #[must_use]
    pub const fn spin_iters(&self) -> u32 {
        self.spin_iters
    }

    /// The grid width in threadgroups, after clamping to [`MIN_THREADGROUPS`].
    #[must_use]
    pub const fn threadgroups(&self) -> u32 {
        self.threadgroups
    }

    /// The in-flight command buffer bound, after clamping to [`MIN_DEPTH`].
    #[must_use]
    pub const fn depth(&self) -> u32 {
        self.depth
    }
}

impl Drop for Keepalive {
    /// Stop the thread and wait for it.
    ///
    /// Joining is not optional and not a tidiness measure. The thread owns
    /// Metal objects and is mid-flight on a queue: returning from here while
    /// it still runs would let this value's fields -- and, through the
    /// retained device and pipeline, objects the context handed out -- be
    /// released while the thread is still encoding against them and while the
    /// GPU is still executing command buffers built from its allocator.
    /// Detaching would trade a bounded wait for a use-after-free that surfaces
    /// as a driver crash somewhere else.
    ///
    /// The wait is bounded by construction: the thread checks the flag once
    /// per iteration and its only blocking call has a `WAIT_TIMEOUT_MS`
    /// timeout, so the worst case is one timeout for the in-flight bound plus
    /// one for the final drain.
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            // A panicked keepalive thread is not a reason to panic the
            // dropping thread -- and dropping during an unwind would abort.
            // The join itself is what matters here; the result is not.
            drop(thread.join());
        }
    }
}

impl std::fmt::Debug for Keepalive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keepalive")
            .field("spin_iters", &self.spin_iters)
            .field("threadgroups", &self.threadgroups)
            .field("depth", &self.depth)
            .field("committed", &self.committed())
            .finish_non_exhaustive()
    }
}

/// Raise `requested` to the smallest depth that keeps work in flight.
///
/// See the module docs: 1 drains the queue between dispatches and 0 is not a
/// depth. A named function rather than a `.max()` at the call site so the
/// clamp can be asserted about directly, without a GPU.
const fn clamp_depth(requested: u32) -> u32 {
    if requested < MIN_DEPTH {
        MIN_DEPTH
    } else {
        requested
    }
}

/// Raise `requested` to the smallest grid that dispatches anything.
const fn clamp_threadgroups(requested: u32) -> u32 {
    if requested < MIN_THREADGROUPS {
        MIN_THREADGROUPS
    } else {
        requested
    }
}

/// A 4-byte shared-storage buffer, or why the device refused one.
fn new_word_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    what: &'static str,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    device
        .newBufferWithLength_options(
            size_of::<u32>(),
            // Shared, so the host can write the spin count without a blit.
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or(Error::Create {
            what,
            message: String::new(),
        })
}

/// The thread body: commit spin dispatches until `running` is cleared.
///
/// The in-flight bound is a wait for `committed - depth + 1` rather than for
/// `committed`, which is what keeps `depth - 1` command buffers executing
/// while this one is encoded. Waiting for `committed` would drain the queue
/// every iteration and measure the idle window instead of removing it.
fn spin(
    owned: &Owned,
    threadgroups: u32,
    depth: u32,
    running: &AtomicBool,
    committed: &AtomicU64,
) -> bool {
    // Widened before multiplying, and saturating: a caller is free to ask for
    // a grid that does not fit, and a debug-build overflow panic on the
    // keepalive thread would take the process down over a tuning parameter.
    let width = (threadgroups as usize).saturating_mul(THREADS_PER_THREADGROUP as usize);
    let grid = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let group = MTLSize {
        width: THREADS_PER_THREADGROUP as usize,
        height: 1,
        depth: 1,
    };
    let inflight = u64::from(depth);
    let mut count: u64 = 0;

    while running.load(Ordering::Relaxed) {
        if count >= inflight
            && !owned
                .event
                .waitUntilSignaledValue_timeoutMS(count - inflight + 1, WAIT_TIMEOUT_MS)
        {
            // Five seconds without retiring a spin dispatch is a wedged GPU,
            // not a slow one. The C++ ignores the return and commits anyway,
            // which piles command buffers onto a queue that is not draining;
            // stopping leaves the drain below as the only thing still owed.
            break;
        }

        // Safe to reset because the wait above proves the command buffer
        // built from it `depth` iterations ago has retired, and no other
        // thread can name this allocator.
        owned.allocator.reset();

        let Some(command_buffer) = owned.device.newCommandBuffer() else {
            break;
        };
        command_buffer.beginCommandBufferWithAllocator(&owned.allocator);
        command_buffer.useResidencySet(&owned.residency);

        let Some(encoder) = command_buffer.computeCommandEncoder() else {
            // The buffer was begun, so it is closed before it is dropped.
            command_buffer.endCommandBuffer();
            break;
        };
        encoder.setComputePipelineState(&owned.pipeline);
        encoder.setArgumentTable(Some(&owned.table));
        encoder.dispatchThreads_threadsPerThreadgroup(grid, group);
        encoder.endEncoding();
        command_buffer.endCommandBuffer();

        let mut pointer = NonNull::from(&*command_buffer);
        // SAFETY: the pointer is to a live single-element array of command
        // buffers that outlives the call -- `command_buffer` is in scope --
        // and `commit:count:` reads the array without retaining it.
        unsafe {
            owned
                .queue
                .commit_count(NonNull::from(&mut pointer).cast(), 1);
        }

        count += 1;
        owned
            .queue
            .signalEvent_value(ProtocolObject::from_ref(&*owned.event), count);
        committed.store(count, Ordering::Relaxed);
    }

    // Drain what is still in flight before the objects go. The thread returns
    // into `Keepalive::drop`'s join, which releases everything here -- so
    // returning with command buffers still executing is the use-after-free
    // the join was supposed to prevent. Bounded, because a wedged GPU must
    // not turn a drop into a hang -- and the answer is reported rather than
    // discarded, because the bound is exactly the case where releasing is
    // the wrong thing to do. See the spawn site.
    owned
        .event
        .waitUntilSignaledValue_timeoutMS(count, WAIT_TIMEOUT_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spin_source_names_the_entry_point_that_is_compiled() {
        assert!(
            SPIN_SOURCE.contains(&format!("kernel void {SPIN_FUNCTION}")),
            "the source must export the function `start` asks for"
        );
    }

    #[test]
    fn the_spin_source_keeps_the_branch_that_defeats_dead_code_elimination() {
        assert!(
            SPIN_SOURCE.contains("atomic_fetch_add_explicit"),
            "without a use of `acc` the optimiser deletes the loop and the GPU idles"
        );
    }

    #[test]
    fn a_depth_below_two_would_drain_the_queue_so_two_is_the_floor() {
        for requested in [0, 1] {
            assert_eq!(clamp_depth(requested), MIN_DEPTH, "{requested}");
        }
        for requested in [2, 8, u32::MAX] {
            assert_eq!(clamp_depth(requested), requested, "above the floor is kept");
        }
    }

    #[test]
    fn a_grid_of_no_threadgroups_would_dispatch_nothing_so_one_is_the_floor() {
        assert_eq!(clamp_threadgroups(0), MIN_THREADGROUPS);
        for requested in [1, 3, 64] {
            assert_eq!(
                clamp_threadgroups(requested),
                requested,
                "above the floor is kept"
            );
        }
    }
}
