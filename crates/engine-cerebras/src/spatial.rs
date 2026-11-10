//! The spatial engine: the model forward runs as a resident wafer program on
//! a CS-3 (or the fabric simulator), driven from this process ([`wafer`]),
//! and everything else stays as the cerebras engine has it: guest (eta)
//! programs in the host interpreter, channels, the runtime's frames.
//!
//! The program is compiled outside the engine (kernels-cerebras
//! `wafer-compile`: the plan from the model's trace, the build, `link.sh`)
//! and carries its [`artifact`]: the host's map of the wafer, every copy onto
//! it, every PE's route words. The engine opens it and nothing else; no model
//! is named here.
//!
//! The program holds one sequence's KV cache. The engine keeps every lane's
//! cache as the tokens that made it (by slot, and by page for `copy_kv`):
//! before a lane's rows run the program's cache is brought to that lane,
//! kept as far as it agrees and the rest replayed, so rejected speculative
//! rows, forks, shared prefixes and several lanes a fire (run one after
//! another) all answer as a device cache would, slowly where lanes
//! alternate. Attention there is causal over a lane's own history only: a
//! lane whose mask says otherwise is refused, as is a program reading
//! attention scores.
//!
//! Selected by `[cerebras.spatial]` in the engine config (`program`: the
//! compiled program's output directory). The program runs on the
//! `[cerebras]` platform (`cmaddr`, else the simulator); its weights are the
//! prep's tiles (`PIE_CEREBRAS_WEIGHTS`).
//!
//! - [`artifact`]: the program on a [`device::Device`], placed and driven;
//! - [`wafer`]: that model on a thread of its own;
//! - `engine`: PIE's `Engine` over it ([`Spatial`]).

pub mod artifact;
mod config;
pub mod device;
mod engine;
pub mod wafer;

pub use config::SpatialBoot;
pub use engine::Spatial;
