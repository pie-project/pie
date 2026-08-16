//! Which optional Vulkan device features a compiled module is allowed to use.
//!
//! This module is deliberately free of dependencies -- it imports nothing, not
//! even `kernels`. `build.rs` pulls it in with `#[path]`, the same trick the
//! archive crate `kernels-cuda` (deleted at `85c6c674b`) used to read its own
//! tables, fourteen modules of them, and that only works while the file needs
//! nothing a build script cannot have. The tier vocabulary has to be shared
//! because the build STAMPS the module names and the library TELLS a driver
//! what to look for; two copies would drift.

/// Which optional device features a module was compiled to use.
///
/// # Why a Vulkan backend needs this and the other two do not
///
/// `kernels-cuda` hands NVRTC the architecture the device reported and
/// `driver-metal` refuses a device below `MTLGPUFamily::Metal4`. Both can do
/// that for reasons this backend does not have: CUDA's compiler runs IN the
/// process, so the code is built after the hardware is known, and Metal is one
/// vendor, so refusing costs a known set of Macs. Neither is true here. One
/// SPIR-V tree ships to AMD, Intel, NVIDIA, Qualcomm and lavapipe, and the
/// features that matter for speed — cooperative matrix above all — are
/// OPTIONAL in the Vulkan sense: a conformant driver may simply not have them.
///
/// So refusing (Metal's answer) would refuse most of the market, and building
/// per-device (CUDA's answer) is not available. What is left is the answer
/// llama.cpp's Vulkan backend reached: compile the same entrypoint more than
/// once, once per tier, and choose at pipeline-creation time. Its `_cm1` /
/// `_cm2` / `_fp32` module suffixes are exactly this enum.
///
/// # The invariant that makes it safe
///
/// **Every entrypoint has a [`Capability::Baseline`] module.** A tier is an
/// ADDITIONAL module for an entrypoint that already exists, never a new
/// entrypoint and never a replacement. `tests/entrypoints.rs` asserts this, and
/// it is the whole of the backward-compatibility guarantee: a device with no
/// optional features still resolves all 480 entrypoints, and a driver that
/// understands no tier at all is still correct, only slower.
///
/// This is also why the tier does not appear in the signature table. The table
/// is what `model-ir` reads, and the compiler must not learn which device
/// it is compiling for — a plan that named a tier would stop being portable
/// between two machines running the same build.
// `Hash` because a driver's pipeline cache is keyed by entrypoint AND tier:
// one entrypoint has up to three modules, compiled from different bodies with
// different extensions, and a cache keyed by the name alone would hand a
// caller the first one built whatever tier it asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    /// Core Vulkan 1.3 plus the 8/16-bit storage extensions the whole tree
    /// already requires. Always present, and the fallback for everything.
    Baseline,
    /// `GL_EXT_shader_explicit_arithmetic_types_float16` — true `float16_t`
    /// arithmetic, as opposed to the baseline's rounding through fp16 STORAGE
    /// with fp32 math.
    Fp16,
    /// `GL_KHR_cooperative_matrix` — the subgroup matrix unit, which is what
    /// the Metal tree reaches through MMA and the CUDA tree through FlashInfer.
    Coopmat,
}

impl Capability {
    /// Every tier, best first.
    ///
    /// A driver walks this and takes the first one the device supports and the
    /// module directory has. Ordering it here rather than at the call site
    /// keeps "which is better" from being decided twice and differently.
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
    ///
    /// Baseline is UNSUFFIXED so that a driver which has never heard of tiers
    /// reads the right file by knowing only the entrypoint name.
    #[must_use]
    pub fn module(self, entrypoint: &str) -> String {
        match self {
            Self::Baseline => format!("{entrypoint}.spv"),
            other => format!("{entrypoint}.{}.spv", other.tag()),
        }
    }

    /// The device features a driver must find before it may load this tier.
    ///
    /// Named as Vulkan names them, because the driver checks them against
    /// `vkGetPhysicalDeviceFeatures2` and not against a shader's declared
    /// capability
    /// spelling.
    ///
    /// `Coopmat` names `shaderFloat16` as well as `cooperativeMatrix`, and that
    /// is not belt-and-braces. A device advertises a LIST of `(M, N, K, types)`
    /// its matrix unit implements, and every list that has anything at all has
    /// fp16 A/B operands -- fp32 A/B is not offered by the hardware this was
    /// first run on, and using a `coopmat` outside the list is undefined
    /// behaviour rather than a slow path. So the tier's operands are
    /// `float16_t`, and a device without `shaderFloat16` cannot load it even if
    /// it has a matrix unit.
    ///
    /// It names `vulkanMemoryModel` for a third reason, and that one was found
    /// by a validation layer rather than reasoned out. `GL_KHR_cooperative_
    /// matrix` requires `GL_KHR_memory_scope_semantics`, so every module in the
    /// tier -- all 146 of them -- declares `OpCapability VulkanMemoryModel`,
    /// and a module may not declare a capability whose feature is not enabled
    /// (`VUID-VkShaderModuleCreateInfo-pCode-08740`). The driver this was
    /// written against creates the pipeline anyway, which is exactly what makes
    /// the omission dangerous: a shell that enabled the tier from this list
    /// would be relying on undefined behaviour that happens to work here.
    ///
    /// And `vulkanMemoryModelDeviceScope` follows from that one, which is the
    /// subtlest name on this list. Enabling the memory model changes the rules
    /// for EVERY module the device loads, not just the tier's: with it on, an
    /// instruction may use `Device` memory scope only if the device-scope
    /// feature is on too (`VUID-RuntimeSpirv-vulkanMemoryModel-06265`), and
    /// `moe/route.slang` -- a BASELINE module -- counts token histograms with a
    /// device-scoped `atomicAdd`. So a shell that turns on the coopmat tier
    /// without this name breaks a kernel that has nothing to do with matrices.
    ///
    /// A driver must therefore check EVERY name here, not the first one.
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
