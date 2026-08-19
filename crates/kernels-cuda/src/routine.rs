//! The vocabulary a routine in this crate is written against.
//!
//! THE PLANE'S PRELUDE, WHICH IS WHY THIS MODULE EXISTS AT ALL. Every other
//! plane already had one: a `kernels-*/src/routine.rs` that names the shared
//! machinery once so a kernel file opens with a single `use crate::routine::{
//! .. }` and states nothing about where any of it lives. CUDA alone had none,
//! and its kernel files each re-derived the list -- `kernels::{Bind, Fire}`
//! beside `crate::jit::{Ctx, Launch, Routine}` beside `kernels::routine::{Env,
//! In, InOut, Out}` -- in ten-odd lines that no two files spelled the same way.
//!
//! That is a structural difference readable off any single file, which is the
//! one thing a routine is not allowed to have: a body should not say which
//! backend compiled it. It also cost more than symmetry. A file that forgot a
//! line did not fail at the forgotten name but at every use of it, so the same
//! omission read as twenty-four *"cannot find type `Fire`"*, twenty *"no method
//! named `arg`"*, and a drift of type errors underneath -- which is how this
//! crate came to hold a hundred and sixty-eight of them while the three shader
//! planes, each with the prelude, held none.
//!
//! Nothing here is new. Every name is a re-export of something this crate or
//! `kernels` already had; the module's whole content is that they are reachable
//! by one path, and the same path the other three planes use.

/// The fact keys a BODY asks the runtime with — `ctx.ask::<i32, keys::Rows>()`.
///
/// Not what a signature binds its scalars from any more: a scalar the
/// checkpoint fixes is a `Const` the statement carries, and a key names only
/// what a fire decides.
pub use kernels::keys;

/// The position wrappers and the launch statement. `Fire` is
/// [`kernels::routine::Fire`] and not a CUDA type: the four facts a launch
/// carries were always the same four, and this plane only used to pass them
/// positionally.
pub use kernels::routine::{Const, Fire, In, InOut, Out};

/// The tensor constructor, this plane's. See [`crate::jit::abi::Tensor`].
pub use crate::jit::abi::Tensor;

/// What a body asks the runtime for, once `Env` is out of the parameter list.
pub use kernels::routine::{Answers, Asks};

/// The words a routine refuses in, and the shapes it refuses about.
pub use kernels::{Bind, Refusal, Region, Stride, Ty};

/// What a routine body launches through, and the geometry it hands the launch.
///
/// A struct here where the shader planes have `dyn Encode`, because this plane
/// CAN name its device: the context holds the JIT cache and the cuBLAS handles.
/// That difference is the associated type's whole purpose and is stated once,
/// in [`kernels::routine::Backend`].
pub use crate::jit::{Ctx, Launch};

/// One row of this plane's table.
pub use crate::jit::Routine;

/// The value a binder hands one argument.
pub use crate::jit::ArgValue;

/// The ABI a `__global__` parameter is written in, and the element types whose
/// widths it is written over.
pub use crate::jit::Abi;
pub use crate::jit::abi::{Inst, bf16, f16};

/// The alignment a vectorised load needs, asked of an address.
pub use crate::jit::aligned16;
