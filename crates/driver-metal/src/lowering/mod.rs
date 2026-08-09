//! A fire's shape becomes a lowered program: symbols, grids and operands.
//!
//! `model_compiler::lower` states what to run — a flat list of launches, each
//! naming a kernel symbol and carrying its operands. Nothing here chooses a
//! kernel. What it does is turn that list into the three things a bind needs:
//! which symbol, what grid, which addresses.
//!
//! **Portable, all of it.** Every module here is arithmetic and lookup, and
//! `tests/{model_bind,model_dispatch,polymorphism}.rs` prove them with no
//! device. That is why this sits above `gpu/` rather than in it, and it is
//! `.wiki/driver/real-metal-north-star.md` §7's own rule — *move the
//! arithmetic, not the crate* — applied to the binder rather than to
//! `kv::Shape`.
//!
//! * [`abi`] — the launch ABI: kernel ids, IO slots, scratch regions.
//! * [`consts`] — the geometry-derived constants a dispatch binds. Reads
//!   `batch::DecodeGeometry` and retires with it.
//! * [`dispatch`] — the walk: every launch of a fire becomes a symbol, a grid
//!   and a list of addresses.
//! * [`executor`] — binding one launch's operands.
//! * [`frame`] — the other end: a sealed frame's step becomes the `&[Row]`
//!   the lowering takes.
//! * [`grid`] — the launch arithmetic itself: what a kernel's
//!   `[[thread_position_in_grid]]` contract says its grid must be.
//! * [`launch`] — how a rectangle becomes a launch: the rule a row names, so
//!   the executor is a loop rather than a switch.
//! * [`resolve`] — the map from the names a text states to the tensors a
//!   checkpoint holds. A map, not a switch: it chooses nothing.

pub mod abi;
pub mod consts;
pub mod dispatch;
pub mod executor;
pub mod frame;
pub mod grid;
pub mod launch;
pub mod resolve;

pub use dispatch::{Dispatch, Geometry, Undispatchable, plan as plan_dispatches};
pub use executor::{BindRefusal, BoundArg, BoundLaunch, Frame, Resolver, Slice, bind, resolve_arg};
pub use frame::{Step, Unbridgeable, Unbridged, fire_class, lower_step, rows_of};
pub use launch::{Dims, Rule, Ungeometric, eval as eval_launch};
pub use resolve::{Names, Store};
