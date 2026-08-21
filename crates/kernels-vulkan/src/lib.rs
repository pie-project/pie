mod capability;
pub use crate::capability::Capability;

pub mod module;
pub use crate::module::{MODULES, code, embedded};

pub mod runtime;

#[allow(unused_imports)]
use kernels::Axis;

pub mod routine;
pub mod views;

pub mod attn;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod ptir;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;

pub type Plane = crate::routine::Vulkan;

#[cfg(not(target_family = "wasm"))]
#[::linkme::distributed_slice]
pub static VULKAN_ROUTINES: [::kernels::routine::Routine<Plane>];

#[cfg(not(target_family = "wasm"))]
pub use VULKAN_ROUTINES as ROUTINES;

#[cfg(target_family = "wasm")]
#[doc(hidden)]
pub struct Registered(pub ::kernels::routine::Routine<Plane>);

#[cfg(target_family = "wasm")]
::inventory::collect!(Registered);

pub fn rows() -> impl Iterator<Item = &'static ::kernels::routine::Routine<Plane>> {
    #[cfg(not(target_family = "wasm"))]
    {
        ROUTINES.iter()
    }
    #[cfg(target_family = "wasm")]
    {
        ::inventory::iter::<Registered>.into_iter().map(|r| &r.0)
    }
}

#[must_use]
pub fn entrypoints() -> Vec<String> {
    module::CENSUS.iter().map(|n| (*n).to_owned()).collect()
}

#[must_use]
pub fn declared() -> Vec<kernels::routine::Declared> {
    rows().map(kernels::routine::Routine::declared).collect()
}

#[must_use]
pub fn routines() -> Vec<&'static routine::Routine> {
    rows().collect()
}

pub trait RoutineElem: kernels::Elem {}

impl<T: kernels::Elem> RoutineElem for T {}
