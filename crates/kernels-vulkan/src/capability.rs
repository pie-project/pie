//! Which optional Vulkan device features a compiled module is allowed to use.
//!
//! This module is deliberately free of dependencies -- it imports nothing, not
//! even `kernels`. `build.rs` pulls it in with `#[path]`. The tier vocabulary
//! is shared here because the build STAMPS the module names and the library
//! TELLS a driver what to look for; two copies would drift.

/// Which optional device features a module was compiled to use.
///
/// Unlike `kernels-cuda` (compiles per-device at runtime) or `driver-metal`
/// (refuses low-end devices), one SPIR-V tree ships to every vendor, so this
/// backend compiles each entrypoint once per tier and picks a pipeline at
/// creation time, as llama.cpp's Vulkan backend does.
/// **Every entrypoint has a [`Capability::Baseline`] module**; a tier is
/// additive, never a replacement (`tests/entrypoints.rs` asserts this), and
/// the tier is absent from the signature table so the compiler never learns
/// which device a plan targets.
// `Hash`: a driver's pipeline cache is keyed by entrypoint AND tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    /// Core Vulkan 1.3 plus the 8/16-bit storage extensions the whole tree
    /// already requires. Always present, and the fallback for everything.
    Baseline,
    /// `GL_EXT_shader_explicit_arithmetic_types_float16` — true `float16_t`
    /// arithmetic, vs. the baseline's fp16-storage/fp32-math rounding.
    Fp16,
    /// `GL_KHR_cooperative_matrix` — the subgroup matrix unit, which is what
    /// the Metal tree reaches through MMA and the CUDA tree through FlashInfer.
    Coopmat,
}

impl Capability {
    /// Every tier, best first. A driver takes the first one both the device
    /// and the module directory support; ordering lives here so it isn't
    /// decided twice and differently at each call site.
    pub const PREFERENCE: [Self; 3] = [Self::Coopmat, Self::Fp16, Self::Baseline];

    /// The tag a `// pie:instantiate` directive spells after `@`.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Fp16 => "fp16",
            Self::Coopmat => "coopmat",
        }
    }

    /// Parse a directive's `@` tag.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "baseline" => Some(Self::Baseline),
            "fp16" => Some(Self::Fp16),
            "coopmat" => Some(Self::Coopmat),
            _ => None,
        }
    }

    /// The file name a `native` build writes this tier's module under.
    /// Baseline is UNSUFFIXED so a driver that has never heard of tiers still
    /// reads the right file by entrypoint name alone.
    #[must_use]
    pub fn module(self, entrypoint: &str) -> String {
        match self {
            Self::Baseline => format!("{entrypoint}.spv"),
            other => format!("{entrypoint}.{}.spv", other.tag()),
        }
    }

    /// The device features a driver must find before it may load this tier.
    ///
    /// Checked against `vkGetPhysicalDeviceFeatures2`, not a shader's declared
    /// capability. `Coopmat` needs `shaderFloat16` because every matrix unit
    /// tested offers fp16 A/B operands only; using the op without it is
    /// undefined behaviour, not a slow path.
    /// `vulkanMemoryModel` is required because `GL_KHR_cooperative_matrix`
    /// pulls in `GL_KHR_memory_scope_semantics`, and a module may not declare
    /// `OpCapability VulkanMemoryModel` with the feature unset
    /// (`VUID-VkShaderModuleCreateInfo-pCode-08740`).
    /// `vulkanMemoryModelDeviceScope` follows because enabling the memory
    /// model affects every module the device loads, including `moe/route.
    /// slang`'s device-scoped `atomicAdd` (`VUID-RuntimeSpirv-
    /// vulkanMemoryModel-06265`). A driver must check EVERY name here.
    #[must_use]
    pub const fn requires(self) -> &'static [&'static str] {
        match self {
            Self::Baseline => &[],
            Self::Fp16 => &["shaderFloat16"],
            Self::Coopmat => &[
                "cooperativeMatrix",
                "shaderFloat16",
                "vulkanMemoryModel",
                "vulkanMemoryModelDeviceScope",
            ],
        }
    }
}
