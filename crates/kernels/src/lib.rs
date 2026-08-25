#![allow(clippy::missing_safety_doc)]

pub mod bound;
pub mod intern;
pub mod plane;
pub mod points;
pub mod raises;
pub mod shader;

pub use plane::Refusal;
pub use plane::{Addressed, Bind, BindMut, Fire, Geometry, Grid, NullArg};
pub use plane::{Cache, Const, ConstRun, In, InOut, Out};
pub use plane::{Elem, Region, Stride};
