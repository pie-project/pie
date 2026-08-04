//! What the caller decides and the checkpoint cannot answer.
//!
//! Every field here fails the schema test — "to check this statement you
//! would have to read GPU code, not the checkpoint" — which is exactly what
//! makes it a parameter. A family's `author` is one function over
//! `(ModelFacts, checkpoint, StorageTarget, Policy)`; CUDA, Metal and an
//! offline `pie model optimize` are three points in this type's space, not
//! three authors.
//!
//! The types mirror the driver's own enums (`model/contract.hpp`) during the
//! migration, wire values included, so a request marshals without a mapping
//! table on either side.
//!
//! Deliberately absent: the tensor-parallel partition and the device's
//! capabilities. Those live in the [`StorageTarget`](pie_loader::plan::StorageTarget)
//! an author receives beside this policy — the same value `compile` reads —
//! so the contract and the plan cannot be told two different worlds.

/// Which half of a multimodal checkpoint to load.
///
/// An authoring concept, not a loader one: the loader is told a contract, and
/// a contract that should not load the vision tower simply does not declare
/// it. The author is the party that decides what to declare.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Component {
    #[default]
    Full = 0,
    Text = 1,
    Encode = 2,
}

/// How the caller asked for MXFP4 expert weights to be lowered, before the
/// device has had its say.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Mxfp4MoeRequest {
    #[default]
    Auto = 0,
    RoutedDecode = 1,
    NativeGemm = 2,
    EagerBf16 = 3,
}

/// How MXFP4 expert weights are lowered, after resolution.
///
/// Split from [`Mxfp4MoeRequest`] because `Auto` is not a policy — it is the
/// absence of one, answered by what the device can run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Mxfp4MoePolicy {
    #[default]
    RoutedDecode = 0,
    NativeGemm = 1,
    EagerBf16 = 2,
}

/// Load-time requantization, already resolved against the device.
///
/// The driver's `resolve_runtime_quant` collapses a request the device cannot
/// serve (`fp8` without native FP8) to `None` before anything is authored;
/// this type carries only the outcome, so an author can never see a policy
/// the target will refuse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RuntimeQuant {
    #[default]
    None,
    Fp8,
    Int8,
    Mxfp4,
}

impl RuntimeQuant {
    /// The driver-side resolution rule, verbatim: an `fp8` request on a
    /// device without native FP8 is no request at all.
    pub fn resolve(request: &str, fp8_native: bool) -> Result<Self, String> {
        match request {
            "" => Ok(Self::None),
            "fp8" if !fp8_native => Ok(Self::None),
            "fp8" => Ok(Self::Fp8),
            "int8" => Ok(Self::Int8),
            "mxfp4" => Ok(Self::Mxfp4),
            other => Err(format!("unknown runtime quantization {other:?}")),
        }
    }
}

/// Whether attention/MLP projections are declared fused or one by one.
///
/// The distinction that used to be "which driver wrote this contract": CUDA
/// fuses q/k/v (and gate/up) into single GEMM operands, Metal binds what the
/// file holds. Now it is a field, and the two drivers are two values of it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projections {
    #[default]
    Fused,
    InPlace,
}

/// Which name vocabulary the declared tensors use.
///
/// `Hf` publishes checkpoint-shaped names; `Mlx` renames for MLX's binder,
/// which is what the Metal driver's forward passes read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Naming {
    #[default]
    Hf,
    Mlx,
}

/// Env-gated per-family switches, mirrored from the driver during the
/// migration.
///
/// Each of these is a `getenv` in the C++ driver today; the *reading* of the
/// environment stays the caller's business (the driver or the CLI fills this
/// struct), so an author never consults the environment itself and two calls
/// with equal inputs cannot author different contracts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyKnobs {
    /// `PIE_GLM5_MOE_FLASHINFER` (default on): publish each GLM-5.2 expert's
    /// halves as `[up | gate]`, the order flashinfer's CUTLASS grouped GEMM
    /// reads fc1 in.
    pub glm5_moe_gate_up_swapped: bool,
    /// `PIE_QWEN35_FUSED_GDN_PROJ` (default off): publish the GDN qkv
    /// projection as the first leg of a fused `in_proj_qkvz`.
    pub qwen35_fused_gdn_projection: bool,
    /// `PIE_QWEN35_MTP_INT8_LM_HEAD` (default off): give the speculative
    /// head an int8 view of `lm_head` beside the main path's bf16.
    pub qwen35_mtp_int8_lm_head: bool,
    /// `PIE_QWEN35_MOE_FLASHINFER` (default on): publish MoE fc1 halves in
    /// the `[up | gate]` order flashinfer's CUTLASS grouped GEMM reads.
    pub qwen35_moe_gate_up_swapped: bool,
    /// `PIE_QWEN35_FUSED_SHARED_SCALAR_GATE` (default off): fold the shared
    /// expert's scalar gate row into its fused gate/up slab.
    pub qwen35_fused_shared_scalar_gate: bool,
    /// `PIE_KIMI_K3_MOE_FLASHINFER` (default off): the same fc1 reorder for
    /// Kimi-K3's dequantized expert stacks.
    pub kimi_k3_moe_gate_up_swapped: bool,
    /// `PIE_KIMI_MOE_FLASHINFER` (default off): publish Kimi's stacked
    /// expert halves as `[up | gate]` for flashinfer's grouped GEMM.
    pub kimi_moe_gate_up_swapped: bool,
    /// `PIE_NEMOTRON_DISABLE_TP_MAMBA_SHARD` inverted (default on): split
    /// the Mamba mixers across ranks. A kill switch for bisecting a
    /// numerical regression; flipping it changes the contract and therefore
    /// re-plans.
    pub nemotron_tp_mamba_sharding: bool,
}

impl Default for FamilyKnobs {
    fn default() -> Self {
        Self {
            glm5_moe_gate_up_swapped: true,
            qwen35_fused_gdn_projection: false,
            qwen35_mtp_int8_lm_head: false,
            qwen35_moe_gate_up_swapped: true,
            qwen35_fused_shared_scalar_gate: false,
            kimi_k3_moe_gate_up_swapped: false,
            kimi_moe_gate_up_swapped: false,
            nemotron_tp_mamba_sharding: true,
        }
    }
}

/// Everything one authoring call was decided by.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Policy {
    pub projections: Projections,
    pub naming: Naming,
    pub runtime_quant: RuntimeQuant,
    /// The MXFP4 MoE *request*, unresolved: `Auto` is answered by the
    /// target's `native_mxfp4_moe` inside the builder, and a family with a
    /// reason to disagree overrides the answer there — so the resolution rule
    /// lives in exactly one place.
    pub moe_request: Mxfp4MoeRequest,
    pub component: Component,
    /// Stream routed experts from the artifact instead of materializing them
    /// resident — the driver's paging mode for MoE weights.
    pub stream_routed_experts: bool,
    /// Per-family switches; see [`FamilyKnobs`].
    pub knobs: FamilyKnobs,
}
