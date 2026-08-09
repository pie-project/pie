//! Executors: run a finished plan.
//!
//! Promoted out of `testkit` because one executor is a production path:
//! `pie model convert` materializes artifacts through [`host`], so the module
//! that does it can no longer live under a name that says "exists only to
//! check the loader". The *oracle* the executor is diffed against
//! ([`crate::testkit::reference`]) keeps that name and its feature gate; the
//! line between them is the line between running a plan and second-guessing
//! one.
//!
//! [`sink`] is where finalized tensors go. An executor that returns a map of
//! every tensor forces its caller's peak memory to the whole output; a sink
//! receives each tensor once, in schedule order, and the streaming entry
//! point frees buffers the moment the schedule is done with them.

pub mod arena;
pub mod chunked;
pub mod sink;
pub mod walk;

/// The CUDA arena, and the load-time transforms that run on the device.
///
/// Behind `feature = "cuda"`, which is off by default. The gate is this one
/// line: [`arena::ArenaBacking`] is already the seam between deciding a load
/// and performing it, so [`host`] does not branch on whether a GPU is present
/// and does not learn that one can be.
#[cfg(feature = "cuda")]
pub mod cuda;

use std::collections::HashMap;
use std::path::Path;

use crate::error::Error;
use crate::executor::arena::ArenaBacking;
use crate::executor::sink::{MemorySink, TensorSink};
use crate::plan::LoadPlan;

/// Everything an execution produced that no caller took.
///
/// [`Execution`] writes into whatever it was given. What it was NOT given, it
/// allocates and returns here: an arena when no [`Execution::arena`] was set,
/// and a tensor map when no [`Execution::sink`] was. A caller that supplied
/// both gets two empty collections, which is the right answer — it already
/// holds the results.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HostStorage {
    pub arena: Vec<u8>,
    pub tensors: HashMap<String, Vec<u8>>,
}

/// Where a plan's persistent buffers live.
///
/// This used to be a `stream: bool` beside an arena the flag said to ignore,
/// which meant the streaming caller passed `&mut &mut [][..]` — a zero-length
/// backing standing in for the absence of one. Two values for one fact, and
/// the false one had to be threaded through every arena operation in the
/// walker as a thing not to touch.
///
/// The three behaviours the flag selected — buffers are owned, buffers are
/// freed at their last use, there is no arena — are one condition: *the
/// plan's persistent buffers have nowhere to live*. It is a property of what
/// the caller handed over, so it is stated by what the caller handed over.
pub enum Residency<'a> {
    /// The plan's arena, wherever it is. Host memory, a mapped device
    /// allocation, or a discrete device reached through
    /// [`ArenaBacking`]'s copies.
    Arena(&'a mut dyn ArenaBacking),
    /// There is no arena. Every buffer is an owned allocation freed at its
    /// last use in the schedule, and finalized tensors reach the caller only
    /// through its sink.
    ///
    /// [`StorageInstr::BulkExtentWrite`](crate::plan::StorageInstr::BulkExtentWrite)
    /// is refused under this residency: it addresses the persistent arena by
    /// offset, and there is not one.
    Streaming,
}

/// Run a finished plan.
///
/// One entry point, where there were five: `execute_plan`,
/// `execute_plan_with_progress`, `execute_plan_into`, `execute_plan_into_arena`
/// and `execute_plan_into_backing`. Every one of them was the same execution
/// with a different subset of the same four decisions — where the arena is,
/// where finalized tensors go, whether anyone is watching — so each new
/// combination meant another function, and the two that took a `bool` took it
/// positionally: `run(plan, dir, arena, sink, progress, /*stream=*/ false)`.
///
/// The defaults are the ones a test wants and a driver overrides:
///
/// ```ignore
/// // Allocate the arena, keep every tensor, hand both back.
/// let storage = Execution::new(&plan, dir).run()?;
///
/// // A discrete device's arena, and the driver's own sink.
/// Execution::new(&plan, dir).arena(&mut backing).sink(&mut sink).run()?;
///
/// // `convert`: no arena at all, each tensor streamed out as it finalizes.
/// Execution::new(&plan, dir).streaming().sink(&mut sink).progress(&mut cb).run()?;
/// ```
pub struct Execution<'a> {
    plan: &'a LoadPlan,
    snapshot_dir: &'a Path,
    arena: Option<&'a mut dyn ArenaBacking>,
    streaming: bool,
    sink: Option<&'a mut dyn TensorSink>,
    progress: Option<&'a mut dyn FnMut(Progress<'_>)>,
}

impl<'a> Execution<'a> {
    /// Execute `plan` against the checkpoint it names.
    ///
    /// `snapshot_dir` is only a base for relative paths. The files themselves
    /// come from `plan.files`, which is the rule for every executor: one that
    /// rediscovered the checkpoint by scanning a directory could disagree with
    /// the plan about which file id means which file, and every offset in the
    /// plan is expressed against that table.
    ///
    /// By default the executor allocates the arena as a `Vec<u8>` and keeps
    /// every finalized tensor, returning both from [`Execution::run`].
    pub fn new(plan: &'a LoadPlan, snapshot_dir: &'a Path) -> Self {
        Self {
            plan,
            snapshot_dir,
            arena: None,
            streaming: false,
            sink: None,
            progress: None,
        }
    }

    /// Write the plan's persistent buffers into an arena the CALLER owns.
    ///
    /// The shape a resident device load wants. Without it the executor
    /// allocates the arena itself and hands it back, so a driver staging
    /// weights holds the whole model TWICE — once in that vector and once in
    /// the buffer it copies the vector into. On a machine where the model is a
    /// meaningful fraction of RAM that is the difference between loading and
    /// being killed.
    ///
    /// `arena` must be at least `plan.memory.persistent_bytes`; a shorter one
    /// is refused rather than truncated, and its contents are overwritten. A
    /// `&mut &mut [u8]` is an [`ArenaBacking`], so host memory needs no
    /// wrapper.
    #[must_use]
    pub fn arena(mut self, arena: &'a mut dyn ArenaBacking) -> Self {
        self.arena = Some(arena);
        self.streaming = false;
        self
    }

    /// Execute with no arena: see [`Residency::Streaming`].
    ///
    /// The memory shape `convert` wants. Peak memory is the largest working
    /// set of one tensor's chain rather than the size of the output, which is
    /// why it is worth refusing `BulkExtentWrite` to have.
    #[must_use]
    pub fn streaming(mut self) -> Self {
        self.streaming = true;
        self.arena = None;
        self
    }

    /// Send each finalized tensor to `sink` instead of collecting them.
    ///
    /// An executor that returns a map of every tensor forces its caller's peak
    /// memory to the whole output. A sink receives each tensor once, in
    /// schedule order.
    #[must_use]
    pub fn sink(mut self, sink: &'a mut dyn TensorSink) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Report a [`Progress`] after every retired instruction.
    ///
    /// For rendering, so it can do no harm: it sees each state once, after the
    /// work is done, and returns nothing.
    #[must_use]
    pub fn progress(mut self, progress: &'a mut dyn FnMut(Progress<'_>)) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Execute, and return whatever no caller claimed.
    ///
    /// # Errors
    ///
    /// The plan names a file that is not there, advertises a transform the
    /// host does not implement, carries a source extent that does not hold
    /// what the plan says it holds, or is handed an arena shorter than its
    /// persistent bytes.
    pub fn run(self) -> Result<HostStorage, Error> {
        let Self {
            plan,
            snapshot_dir,
            arena,
            streaming,
            sink,
            progress,
        } = self;
        let mut owned_sink = MemorySink::default();
        let mut unwatched = |_: Progress<'_>| {};
        let progress = progress.unwrap_or(&mut unwatched);
        // The arena the executor allocates when the caller supplied none.
        // Empty in every other case, and returned either way — which is how
        // `HostStorage` says "what nobody took" without a second return type.
        let mut owned_arena = match (&arena, streaming) {
            (None, false) => vec![
                0u8;
                usize::try_from(plan.memory.persistent_bytes).map_err(|_| {
                    Error::Contract("persistent arena does not fit host address space".into())
                })?
            ],
            _ => Vec::new(),
        };
        {
            let mut owned_backing: &mut [u8] = &mut owned_arena;
            let residency = match (arena, streaming) {
                (Some(arena), _) => Residency::Arena(arena),
                (None, true) => Residency::Streaming,
                (None, false) => Residency::Arena(&mut owned_backing),
            };
            match sink {
                Some(sink) => walk::run(plan, snapshot_dir, residency, sink, progress)?,
                None => walk::run(plan, snapshot_dir, residency, &mut owned_sink, progress)?,
            }
        }
        Ok(HostStorage {
            arena: owned_arena,
            tensors: owned_sink.tensors,
        })
    }
}

/// One retired instruction of an executing plan, for a caller rendering
/// progress.
///
/// The smooth axis is bytes, not instructions: one tensor can be half the
/// model, so an instruction count jerks where the checkpoint bytes consumed
/// so far advance evenly. The total is the plan's own statement
/// ([`crate::plan::MemoryPlan::checkpoint_read_bytes`]), known before the
/// first instruction runs — which is what makes a percentage possible at all.
pub struct Progress<'a> {
    /// Checkpoint bytes consumed so far.
    pub read_bytes: u64,
    /// What `read_bytes` counts toward.
    pub total_read_bytes: u64,
    /// The runtime tensor this instruction published, when it published one.
    pub finalized: Option<&'a str>,
}
