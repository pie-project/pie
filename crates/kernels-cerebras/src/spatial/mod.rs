//! The role-split wafer program's build (`programs/split`: one sequence on the wafer, the
//! layers in units of weight PEs, the head and the token loop beside them): a [`plan`] (the
//! model and how it is cut, no coordinates) through the [`family`]'s [`block`]s to the
//! [`floor`], [`wiring`] and laid [`fabric`], every kernel's [`abi`] ([`kernels`]), the build's
//! outputs ([`emit`]) and the weight [`weights`] bindings; `ARCHITECTURE.md` walks the layers.
//! The engine (`engine-cerebras`, `spatial::split`) places and drives a built program.

pub mod abi;
pub mod block;
pub mod emit;
pub mod fabric;
pub mod family;
pub mod floor;
pub mod kernels;
pub mod npy;
pub mod plan;
pub mod program;
pub mod role;
pub mod weights;
pub mod wiring;
