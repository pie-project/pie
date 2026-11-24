//! The wafer program's compiler. Poem's spatial pass over the model's trace ([`target`]) makes a
//! [`plan`], which goes through the [`block`]s its layers need ([`compose`]) to the [`floor`],
//! [`wiring`] and laid [`fabric`], every kernel's [`abi`] ([`kernels`]), the build's outputs
//! ([`emit`]) and the [`weights`] bindings.

pub mod abi;
pub mod artifact;
pub mod block;
pub mod build;
pub mod check;
pub mod cold;
pub mod compose;
pub mod cost;
pub mod emit;
pub mod fabric;
pub mod floor;
pub mod floorplan;
pub mod kernels;
pub mod layers;
pub mod npy;
pub mod place;
pub mod plan;
pub mod program;
pub mod role;
pub mod search;
pub mod stream;
pub mod target;
pub mod tiles;
pub mod toolchain;
pub mod weights;
pub mod wiring;
