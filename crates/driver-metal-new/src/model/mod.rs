//! The model executor: running a lowered fire.
//!
//! `model_compiler::lower` states what to run — a flat list of launches, each
//! naming a kernel symbol and carrying its operands. Nothing here chooses a
//! kernel; see `DIRECTION.md` and `model-compiler/DSL-DESIGN.md`.
//!
//! * [`executor`] — binding a launch's operands. Host logic, no device.
//! * [`geometry`] — turning a rectangle into a thread grid: the rule a row
//!   names, so the executor is a loop rather than a switch.
//! * [`dispatch`] — the walk that uses both: every launch of a fire becomes a
//!   symbol, a grid and a list of addresses. Still host logic.
//! * [`frame`] — the other end: a sealed frame's step becomes the `&[Row]` the
//!   lowering takes. Host logic too, and the piece that was expected to have
//!   no predecessor.
//! * [`grid`] — the launch arithmetic itself, moved out of the retiring
//!   `batch/dispatch.rs` because a kernel's thread-position contract is
//!   backend knowledge that survives the DAG builder beside it.
//! * [`kv`] — the paged KV pool, sized by the fire's geometry rather than by
//!   a model. Apple-only.
//! * [`load`] — the call that was missing between `loader/` and
//!   `metal::stage_plan_weights`, producing what `resolve` reads. Apple-only.
//! * [`text`] — which text the loaded checkpoint is. A LOOKUP, not a choice:
//!   remove it and the same kernels fire.
//! * [`resolve`] — the map from the names a text states to the tensors a
//!   checkpoint holds. The one per-family piece, and a map rather than a
//!   switch: it chooses nothing, it translates a spelling.
//! * [`encode`] — the one half that needs a GPU: compile the symbols, bind the
//!   addresses, dispatch. Apple-only.
//! * [`run`] — the four calls in one place: allocate the arena the lowering
//!   asked for, plan, compile, encode. Apple-only.

pub mod dispatch;
#[cfg(target_vendor = "apple")]
pub mod encode;
pub mod executor;
pub mod frame;
pub mod geometry;
pub mod grid;
#[cfg(target_vendor = "apple")]
pub mod kv;
#[cfg(target_vendor = "apple")]
pub mod load;
pub mod resolve;
pub mod tables;
pub mod rope;
pub mod text;
#[cfg(target_vendor = "apple")]
pub mod run;

pub use dispatch::{Dispatch, Geometry, Undispatchable, plan as plan_dispatches};
pub use executor::{BindRefusal, BoundArg, BoundLaunch, Frame, Resolver, Slice, bind, resolve_arg};
pub use frame::{Step, Unbridged, Unbridgeable, fire_class, lower_step, rows_of};
pub use geometry::{Dims, Rule, Ungeometric, eval as eval_launch};
pub use resolve::{Names, Store};
