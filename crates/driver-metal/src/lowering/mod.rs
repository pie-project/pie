//! A fire's shape becomes a lowered program: symbols, grids and operands.
//!
//! `model_compiler::lower` states what to run (a flat list of launches); this
//! module turns that into which symbol, what grid, and which addresses to bind.
//!
//! Pure arithmetic and lookup, no device — `tests/{model_bind,model_dispatch,
//! polymorphism}.rs` prove it, which is why this lives above `gpu/`.

pub mod abi;
pub mod bind;
pub mod cached;
pub mod consts;
pub mod dispatch;
pub mod executor;
pub mod frame;
pub mod hold;
pub mod resolve;
pub mod routine;

pub use dispatch::{Dispatch, Geometry, Undispatchable, plan as plan_dispatches};
pub use executor::{BindRefusal, BoundArg, BoundLaunch, Frame, Resolver, Slice, bind, resolve_arg};
pub use frame::{Step, Unbridgeable, Unbridged, fire_class, lower_step, rows_of};
pub use resolve::{Names, Store};
