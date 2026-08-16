//! Executors: run a finished plan.
//!
//! This is the production path -- `pie model convert` materializes
//! artifacts through [`host`] -- distinct from the *oracle* it is diffed
//! against ([`crate::testkit::reference`]), which exists only to
//! second-guess a plan rather than run one.
//!
//! [`sink`] is where finalized tensors go: an executor that returns a map
//! of every tensor forces its caller's peak memory to the whole output,
//! while a sink receives each tensor once, in schedule order, freed as
//! soon as the schedule is done with it.

pub mod arena;
pub mod chunked;
pub mod iq_grid;
pub mod sink;
pub mod walk;

/// The CUDA arena, and the load-time transforms that run on the device,
/// behind `feature = "cuda"` (off by default): [`arena::ArenaBacking`] is
/// already the seam, so [`host`] never branches on whether a GPU exists.
#[cfg(feature = "cuda")]
pub mod cuda;

// `cuda` says *that* there is a device; `cuda-12`/`cuda-13` say WHICH
// libcudart this binary will meet, and `kernels-cuda` gates its whole
// runtime layer on that choice -- so `--features cuda` alone would reach
// a missing `kernels_cuda::api`. This is the diagnostic a reader gets
// instead of cudarc's build-script panic two crates down.
#[cfg(all(feature = "cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "model-loader's `cuda` feature needs a runtime version too: enable \
     `cuda-12` or `cuda-13`, matching the libcudart this binary will load. \
     `driver-cuda`'s own `cuda-12`/`cuda-13` already pass the choice on."
);

use std::collections::HashMap;
use std::path::Path;

use crate::error::Error;
use crate::executor::arena::ArenaBacking;
use crate::executor::sink::{MemorySink, TensorSink};
use crate::plan::LoadPlan;

/// Everything an execution produced that no caller took. [`Execution`]
/// writes into whatever it was given; what it was NOT given, it allocates
/// and returns here. A caller that supplied both gets two empty
/// collections, which is the right answer -- it already holds the results.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HostStorage {
    /// The persistent arena, when the caller supplied no backing for it.
    pub arena: Vec<u8>,
    /// Finalized tensors by name, when the caller supplied no sink.
    pub tensors: HashMap<String, Vec<u8>>,
}

/// Where a plan's persistent buffers live: one property (arena vs
/// streaming) rather than a `stream: bool` beside an arena the flag said
/// to ignore, which is the property the caller states by what it hands
/// over.
pub enum Residency<'a> {
    /// The plan's arena, wherever it is. Host memory, a mapped device
    /// allocation, or a discrete device reached through
    /// [`ArenaBacking`]'s copies.
    Arena(&'a mut dyn ArenaBacking),
    /// There is no arena. Every buffer is an owned allocation freed at
    /// its last use in the schedule, and finalized tensors reach the
    /// caller only through its sink.
    /// [`StorageInstr::BulkExtentWrite`](crate::plan::StorageInstr::BulkExtentWrite)
    /// is refused under this residency: it addresses the persistent arena by offset, and there is not one.
    Streaming,
}

/// Run a finished plan. One entry point for four independent decisions
/// -- where the arena is, where finalized tensors go, whether anyone is
/// watching, whether to stream -- so a caller states only the ones that
/// differ from the defaults a test wants:
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
    /// Execute `plan` against the checkpoint it names. `snapshot_dir` is
    /// only a base for relative paths -- the files themselves come from
    /// `plan.files`, since rediscovering the checkpoint by scanning a
    /// directory could disagree with the plan about which file id means
    /// which file.
    /// By default the executor allocates the arena as a `Vec<u8>` and
    /// keeps every finalized tensor, returning both from [`Execution::run`].
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
    /// `arena` must be at least `plan.memory.arena_bytes()` -- the persistent
    /// buffers AND the staging region behind them, which are one
    /// allocation; a shorter one
    /// is refused rather than truncated, and its contents are overwritten. A
    /// `&mut &mut [u8]` is an [`ArenaBacking`], so host memory needs no
    /// wrapper.
    #[must_use]
    pub fn arena(mut self, arena: &'a mut dyn ArenaBacking) -> Self {
        self.arena = Some(arena);
        self.streaming = false;
        self
    }

    /// Execute with no arena: see [`Residency::Streaming`]. The memory
    /// shape `convert` wants, where peak memory is the largest working set
    /// of one tensor's chain rather than the size of the output.
    #[must_use]
    pub fn streaming(mut self) -> Self {
        self.streaming = true;
        self.arena = None;
        self
    }

    /// Send each finalized tensor to `sink` instead of collecting them:
    /// an executor that returns a map of every tensor forces its caller's
    /// peak memory to the whole output.
    #[must_use]
    pub fn sink(mut self, sink: &'a mut dyn TensorSink) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Report a [`Progress`] after every retired instruction, for
    /// rendering: it sees each state once, after the work is done, and
    /// returns nothing, so it can do no harm.
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
    /// what the plan says it holds, or is handed an arena shorter than
    /// `plan.memory.arena_bytes()`.
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
            // `arena_bytes()`, not `persistent_bytes`: the staging region
            // lives in the same allocation and every offset in the plan is
            // measured from the same base, so a caller who allocates only the
            // persistent half hands `walk::run` an arena it refuses. This
            // self-allocating path IS such a caller, and got it wrong -- any
            // plan with a non-zero `scratch_bytes` failed here before reading
            // a single byte, with `arena is N bytes and the plan needs M`.
            (None, false) => vec![
                0u8;
                usize::try_from(plan.memory.arena_bytes()).map_err(|_| {
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
/// progress. The smooth axis is bytes, not instructions -- one tensor can
/// be half the model -- and the total is the plan's own
/// [`crate::plan::MemoryPlan::checkpoint_read_bytes`], known before the
/// first instruction runs.
pub struct Progress<'a> {
    /// Checkpoint bytes consumed so far.
    pub read_bytes: u64,
    /// What `read_bytes` counts toward.
    pub total_read_bytes: u64,
    /// The runtime tensor this instruction published, when it published one.
    pub finalized: Option<&'a str>,
}
