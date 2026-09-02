//! `eta-dsl`: the ETA embedded DSL (Rust SDK). Authors trace Rust closures
//! once into a canonical [`TraceContainer`](eta_ir::container::TraceContainer)
//! via [`Builder`]; it knows nothing of WIT, which `inferlet` wraps around it.

extern crate alloc;

pub mod builder;
pub mod channel;
mod context;
pub mod dtype;
pub mod error;
pub mod intrinsics;
mod lint;
pub mod model;
pub mod value;

pub use builder::{Builder, PortInput, Traced};
pub use channel::{Channel, IntoPut, Put};
pub use error::{Endpoint, Span, TraceError, TraceErrors};
/// The eDSL op surface. Glob-re-exported rather than listed: an op is public
/// exactly when it is `pub` in [`value`], so adding one cannot be half-done.
pub use value::*;

/// The canonical ETA contract, re-exported for tests and downstream carriers.
pub use eta_ir as eta;
pub use eta_ir::registry::{Port, Stage};
pub use eta_ir::types::{Dtype, Shape, ValueType};

/// Glob-import surface for the DSL's op/value names.
pub mod prelude {
    pub use crate::builder::{Builder, PortInput};
    pub use crate::channel::Channel;
    pub use crate::dtype;
    pub use crate::intrinsics;
    pub use crate::value::*;
    pub use eta_ir::registry::{Port, Stage};
}
