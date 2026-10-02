use crate::config::EngineKind;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Flavor {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    Metal,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(feature = "wgpu")]
    Wgpu,
    #[cfg(feature = "xla")]
    Xla,
    #[cfg(feature = "cerebras")]
    Cerebras,
}

impl Flavor {
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Flavor::Cuda => "cuda",
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Flavor::Metal => "metal",
            #[cfg(feature = "vulkan")]
            Flavor::Vulkan => "vulkan",
            #[cfg(feature = "wgpu")]
            Flavor::Wgpu => "wgpu",
            #[cfg(feature = "xla")]
            Flavor::Xla => "xla",
            #[cfg(feature = "cerebras")]
            Flavor::Cerebras => "cerebras",
        }
    }

    pub fn from_kind(kind: EngineKind) -> Result<Self, String> {
        match kind {
            EngineKind::CudaNative => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Flavor::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(missing_feature_msg("cuda_native", "cuda"))
                }
            }
            EngineKind::Metal => {
                #[cfg(all(feature = "metal", target_vendor = "apple"))]
                {
                    Ok(Flavor::Metal)
                }
                #[cfg(all(feature = "metal", not(target_vendor = "apple")))]
                {
                    Err(non_apple_msg())
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(missing_feature_msg("metal", "metal"))
                }
            }
            EngineKind::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    Ok(Flavor::Vulkan)
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    Err(missing_feature_msg("vulkan", "vulkan"))
                }
            }
            EngineKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    Ok(Flavor::Wgpu)
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    Err(missing_feature_msg("wgpu", "wgpu"))
                }
            }
            EngineKind::Xla => {
                #[cfg(feature = "xla")]
                {
                    Ok(Flavor::Xla)
                }
                #[cfg(not(feature = "xla"))]
                {
                    Err(missing_feature_msg("xla", "xla"))
                }
            }
            EngineKind::Cerebras => {
                #[cfg(feature = "cerebras")]
                {
                    Ok(Flavor::Cerebras)
                }
                #[cfg(not(feature = "cerebras"))]
                {
                    Err(missing_feature_msg("cerebras", "cerebras"))
                }
            }
        }
    }
}

#[cfg(all(feature = "metal", not(target_vendor = "apple")))]
fn non_apple_msg() -> String {
    format!(
        "engine type \"metal\" needs an Apple target and this binary was \
         built for another. The `metal` feature IS on — what it selects \
         off Apple is the engine's portable half, which answers questions no \
         GPU changes and serves nothing. Compiled flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

#[cfg(not(all(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "wgpu",
    feature = "xla",
    feature = "cerebras"
)))]
fn missing_feature_msg(toml_type: &str, feature: &str) -> String {
    format!(
        "engine type {toml_type:?} is not built into this binary. \
         Rebuild `worker` with `--features {feature}` (or include \
         it alongside the other backend features). Compiled flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

#[allow(
    clippy::vec_init_then_push,
    reason = "the pushes are `#[cfg]`-gated, and an attribute cannot be \
              attached to an element inside `vec![]`"
)]
pub fn compiled_summary() -> String {
    #[cfg_attr(
        not(any(
            feature = "cuda",
            feature = "vulkan",
            feature = "wgpu",
            feature = "xla",
            feature = "cerebras",
            all(feature = "metal", target_vendor = "apple")
        )),
        allow(unused_mut, reason = "the pushes below are feature-gated")
    )]
    let mut out: Vec<&'static str> = Vec::new();
    #[cfg(feature = "cuda")]
    out.push("cuda");
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    out.push("metal");
    #[cfg(feature = "vulkan")]
    out.push("vulkan");
    #[cfg(feature = "wgpu")]
    out.push("wgpu");
    #[cfg(feature = "xla")]
    out.push("xla");
    #[cfg(feature = "cerebras")]
    out.push("cerebras");
    out.join(", ")
}

pub fn compiled_embedded() -> [(&'static str, bool); 6] {
    [
        ("cuda_native", cfg!(feature = "cuda")),
        (
            "metal",
            cfg!(all(feature = "metal", target_vendor = "apple")),
        ),
        ("vulkan", cfg!(feature = "vulkan")),
        ("wgpu", cfg!(feature = "wgpu")),
        ("xla", cfg!(feature = "xla")),
        ("cerebras", cfg!(feature = "cerebras")),
    ]
}

pub fn default_flavor() -> Option<Flavor> {
    match runtime::engine::load::this_box()? {
        #[cfg(feature = "cuda")]
        runtime::engine::load::Platform::Cuda => Some(Flavor::Cuda),
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        runtime::engine::load::Platform::Metal => Some(Flavor::Metal),
        #[cfg(feature = "vulkan")]
        runtime::engine::load::Platform::Vulkan => Some(Flavor::Vulkan),
        #[cfg(feature = "wgpu")]
        runtime::engine::load::Platform::Wgpu => Some(Flavor::Wgpu),
        #[cfg(feature = "xla")]
        runtime::engine::load::Platform::Xla => Some(Flavor::Xla),
        #[cfg(feature = "cerebras")]
        runtime::engine::load::Platform::Cerebras => Some(Flavor::Cerebras),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

use anyhow::{Result, anyhow};

pub fn resolve(kind: EngineKind, model_name: &str) -> Result<Flavor> {
    Flavor::from_kind(kind).map_err(|msg| anyhow!("model {model_name:?}: {msg}"))
}
