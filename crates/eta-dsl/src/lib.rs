extern crate alloc;

pub mod builder;
pub mod channel;
mod context;
pub mod dtype;
pub mod error;
pub mod intrinsics;
mod lint;
pub mod model;
pub mod sampler;
pub mod value;

pub use builder::{Builder, PortInput, Traced};
pub use channel::{Channel, IntoPut, Put};
pub use error::{Endpoint, Span, TraceError, TraceErrors};
pub use value::*;

pub use eta_ir as eta;
pub use eta_ir::registry::{Port, Stage};
pub use eta_ir::types::{Dtype, Shape, ValueType};

pub mod prelude {
    pub use crate::builder::{Builder, PortInput};
    pub use crate::channel::Channel;
    pub use crate::dtype;
    pub use crate::intrinsics;
    pub use crate::value::*;
    pub use eta_ir::registry::{Port, Stage};
}
