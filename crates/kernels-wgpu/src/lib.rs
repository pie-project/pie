pub use kernels::{Axis, Cap};

mod capability;
pub use crate::capability::Capability;

pub mod preproc;
pub use crate::preproc::{Directive, Malformed, Variant, expand, instantiations};

pub mod source;
pub use crate::source::{Missing, SOURCES, entrypoint_source, source};

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

pub type Plane = crate::routine::Wgpu;

#[cfg(not(target_family = "wasm"))]
#[::linkme::distributed_slice]
pub static WGPU_ROUTINES: [::kernels::routine::Routine<Plane>];

#[cfg(not(target_family = "wasm"))]
pub use WGPU_ROUTINES as ROUTINES;

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
    static CENSUS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    CENSUS
        .get_or_init(|| {
            let mut out: Vec<String> = source::declared()
                .into_iter()
                .map(|(_, variant)| variant.entrypoint)
                .collect();
            out.sort();

            out.dedup();
            out
        })
        .clone()
}

#[must_use]
pub fn routines() -> Vec<&'static routine::Routine> {
    rows().collect()
}

#[must_use]
pub fn declared() -> Vec<kernels::routine::Declared> {
    rows().map(kernels::routine::Routine::declared).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    Storage(u32),
    Uniform(u32),
    Packed,
}

pub const DOWNLEVEL_STORAGE_BUFFERS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformField {
    pub name: &'static str,
    pub offset: u32,
    pub size: u32,
    pub split: bool,
}

pub const DOWNLEVEL_UNIFORM_BYTES: u32 = 16 * 1024;

pub const fn is_buffer(ty: kernels::Ty) -> bool {
    !matches!(ty.binds(), kernels::Binds::Nothing)
}

pub trait RoutineElem: kernels::Elem {}

impl<T: kernels::Elem> RoutineElem for T {}
