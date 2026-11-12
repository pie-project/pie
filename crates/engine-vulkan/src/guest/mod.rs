//! The shared guest forms ([`engine_gpu::guest`]) bound to Vulkan: a
//! stage's WGSL lowers to SPIR-V.

pub use engine_gpu::guest::{ENTRY, Refused, step_source};

pub type Compiled = engine_gpu::guest::Compiled<crate::device::Api>;
pub type Lowered = engine_gpu::guest::Lowered<Vec<u32>>;
pub type Forms = engine_gpu::guest::Forms<crate::device::Api>;

impl engine_gpu::guest::Lower for crate::device::Api {
    type Code = Vec<u32>;

    fn lower(stage: usize, source: String) -> Result<Vec<u32>, Refused> {
        lower(stage, &source)
    }
}

#[cfg(feature = "vulkan")]
pub fn lower(stage: usize, source: &str) -> Result<Vec<u32>, Refused> {
    let refuse = |why: String| Refused::Lowering { stage, why };
    let module =
        naga::front::wgsl::parse_str(source).map_err(|e| refuse(e.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| refuse(e.emit_to_string(source)))?;
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .map_err(|e| refuse(e.to_string()))
}

#[cfg(not(feature = "vulkan"))]
pub fn lower(stage: usize, _source: &str) -> Result<Vec<u32>, Refused> {
    Err(Refused::Lowering {
        stage,
        why: "this build carries no Vulkan device".to_string(),
    })
}

#[cfg(feature = "vulkan")]
pub mod run;
#[cfg(feature = "vulkan")]
pub mod session;
#[cfg(feature = "vulkan")]
pub mod widen;
