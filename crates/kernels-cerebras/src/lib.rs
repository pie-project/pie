//! Cerebras kernels: the wafer program's compiler. It plans a model's trace onto the wafer, builds
//! the program (floor, links, kernels, weight bindings of a whole-wafer program, `programs/wafer`),
//! links it with the SDK's toolchain and cuts the checkpoint into the tiles it places, written as an
//! [`artifact::Artifact`] the engine loads and runs.
//!
//! The wafer program's compiler (`programs/wafer`: one sequence on the wafer, the layers in units
//! of weight PEs, the head and the token loop beside them): poem's spatial pass over the model's
//! trace ([`target`]) makes a [`plan`] (the model and how it is cut, no coordinates), which goes
//! through the [`family`]'s [`block`]s to the [`floor`], [`wiring`] and laid [`fabric`], every
//! kernel's [`abi`] ([`kernels`]), the build's outputs ([`emit`]) and the weight [`weights`]
//! bindings; `ARCHITECTURE.md` (the crate root) walks the layers.
//!
//! What the engine runs is the [`artifact::Artifact`]: the host's map of the program, every copy
//! onto the wafer and every PE's rcfg words. `wafer-compile build` writes it beside the program's
//! sources; the engine names no model.

pub mod abi;
pub mod artifact;
pub mod block;
pub mod build;
pub mod compose;
pub mod cost;
pub mod emit;
pub mod fabric;
pub mod family;
pub mod floor;
pub mod kernels;
pub mod layers;
pub mod npy;
pub mod plan;
pub mod program;
pub mod role;
pub mod sources;
pub mod target;
pub mod tiles;
pub mod toolchain;
pub mod weights;
pub mod wiring;
