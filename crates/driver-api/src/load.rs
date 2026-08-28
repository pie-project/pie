//! `LoadRequest` in, `Loaded` out — the one door a model comes through.
//!
//! # The `Trace` crosses; `CompiledModel` never does (decision 18)
//!
//! The ENGINE traces the model. It links `model` anyway — `Classify::of` runs
//! per lane on the fire path to compute [`Lane::word`](crate::fire::Lane::word)
//! — so the supergraph is already in its address space, and handing it across
//! costs a `serde` round trip a remote driver was going to need regardless.
//!
//! What does NOT cross is `model_compiler::CompiledModel`: the region template, the
//! class table, the arena carve. Those are the SHELL's, produced by
//! `compile(trace, budgets, profile)` on the far side of this boundary, because
//! they are answers about a device the engine does not have. The consequence
//! is the one the design wants: a remote driver is a `serde`-able `Trace` on a
//! socket and nothing else, and the compiler never has to be portable.
//!
//! ```text
//!  engine                          |  shell
//!  ------                          |  -----
//!  model/ forward -> Trace  --------|--> compile(trace, budgets, profile) -> CompiledModel
//!  Classify::of(req) -> word       |    record one graph per bucket
//!                                  |    land the checkpoint
//!  FireSubmission{lanes} ----------|--> compose -> walk -> replay
//! ```
//!
//! # What died here
//!
//! `ModelLoadDesc` was `{ snapshot_dir: PathBuf, runtime_quant: String,
//! mxfp4_moe: Mxfp4MoeRequest, component: ModelComponent }` and `load_model`
//! took a `Vec` of them. Every field of it was a way of saying something the
//! plan now says outright:
//!
//! * `snapshot_dir` -> [`LoadRequest::checkpoint`], unchanged in substance.
//! * `runtime_quant: String` — a backend-parsed quantization *word*. The
//!   plan's params carry their own dtypes; a string the shell string-matched
//!   is not a second opinion worth keeping.
//! * `mxfp4_moe: Mxfp4MoeRequest` — four ways to run one op, chosen at load by
//!   name. Which kernel answers an op is the dispatch arm's decision (design
//!   §6), and the axis it was really selecting is a model-declared one.
//! * `component: ModelComponent{Full,Text,Encode}` — WHICH graph to load, by
//!   enum. It is now which `Trace` you hand over: the encoder is a traced plan
//!   like any other, and `Vec<ModelLoadDesc>` collapses to one request per
//!   plan.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::caps::Capabilities;

/// The ceilings a load is baked against.
///
/// The same four numbers `model_compiler::Budget` states, carried across the
/// boundary because the compile happens on the far side of it and a shell that
/// invented its own ceilings would bake a graph the engine cannot fill.
///
/// A duplicate spelling is exactly what decision 1 kills, so this is written
/// once here and converted by the shell in one place — the alternative,
/// `driver-api` depending on `model-compiler`, would put `CompiledModel` in the
/// dependency graph of `transport` and `controller-api`, which is the edge
/// decision 18 exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    /// The most requests one fire may carry. `Dim::Lanes` is this number.
    pub max_lanes: u32,
    /// The most token rows one fire may carry. `Dim::Tokens` is this number.
    pub max_tokens: u32,
    /// The shape lattice a fire's row count is rounded up to — one immutable
    /// graph per entry. Ascending, each entry at most `max_tokens`.
    pub buckets: Vec<u32>,
    /// How many adapter banks the device pool holds (design §8).
    pub max_adapters: u32,
    /// Tokens per KV page.
    pub page_size: u32,
    /// The most tokens one sequence may hold.
    pub max_context: u32,
    /// How many sequences the pools seat at once.
    pub slots: u32,
}

impl Default for Budgets {
    /// 256 lanes, 8192 rows, 16-token pages, 4096 tokens of context, 256
    /// slots: a deployment that runs, for a caller who has measured nothing.
    fn default() -> Budgets {
        Budgets {
            max_lanes: 256,
            max_tokens: 8192,
            buckets: Vec::new(),
            max_adapters: 0,
            page_size: 16,
            max_context: 4096,
            slots: 256,
        }
    }
}

/// Where a load's weights come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Checkpoint {
    /// A snapshot directory, or one container file.
    Path(PathBuf),
    /// No weights: bind the device and bake the plan, but land nothing. What a
    /// shape-only smoke test loads, and what makes a `CompiledModel` inspectable
    /// without a checkpoint on the machine.
    None,
}

/// Everything a load states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadRequest {
    /// The traced supergraph. The engine traces it (decision 18).
    pub trace: model_ir::Trace,
    /// Where the weights are.
    pub checkpoint: Checkpoint,
    /// The ceilings every fire is baked against.
    pub budgets: Budgets,
    /// Which device to bind, when the shell serves more than one.
    pub ordinal: i32,
}

/// What a load answers with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Loaded {
    /// The facts this load carries — the plan's own name and the residency it
    /// achieved.
    pub facts: LoadFacts,
    /// What it can do.
    pub caps: Capabilities,
}

/// What came of a load, as numbers a caller can log and act on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadFacts {
    /// The trace's own name, as the model text declared it.
    pub trace_name: String,
    /// Bytes the weight tables occupy on the device.
    pub weight_bytes: u64,
    /// Bytes the activation arena occupies.
    pub arena_bytes: u64,
    /// Bytes the pools occupy.
    pub pool_bytes: u64,
    /// Bytes the resident fire inputs occupy.
    pub input_bytes: u64,
}
