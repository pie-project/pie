//! What the caller decides and the checkpoint cannot answer.
//!
//! Every field here fails the schema test — "to check this statement you
//! would have to read GPU code, not the checkpoint" — which is exactly what
//! makes it a parameter. A family's `author` is one function over
//! `(ModelFacts, checkpoint, StorageTarget, Policy)`; CUDA, Metal and an
//! offline `pie model build` are three points in this type's space, not
//! three authors.
//!
//! The types mirror the driver's own enums (`model/contract.hpp`) during the
//! migration, wire values included, so a request marshals without a mapping
//! table on either side.
//!
//! Deliberately absent: the tensor-parallel partition and the device's
//! capabilities. Those live in the [`StorageTarget`](model_loader::plan::StorageTarget)
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
    /// MLX affine, 4 bits over groups of 64 — the format every Metal matvec
    /// reads, and the only one its loader can encode into.
    ///
    /// Last, because the wire values are positional: the C ABI sends this as
    /// an integer and inserting a variant in the middle renumbers the rest.
    Int4,
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
            "int4" => Ok(Self::Int4),
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

/// Env-gated per-family switches, mirrored from the driver.
///
/// Each of these is a `getenv` in the C++ driver; the *reading* of the
/// environment stays the caller's business (the driver or the CLI fills this
/// struct), so an author never consults the environment itself and two calls
/// with equal inputs cannot author different contracts.
///
/// **A knob may only live here if the forward path reads the same switch.**
/// Six of these were expert-layout choices — `[up | gate]` reorders for
/// flashinfer, the fused GDN projection, the folded shared scalar gate —
/// whose forward arms were deleted, leaving an environment variable that
/// still moved the *weights* while the *matmuls* had settled on a constant.
/// That failure is silent: not a load error, wrong output. Their values are
/// constants in each family's author now, written down beside the C++
/// constant they have to agree with. What is left is what both sides really
/// do read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyKnobs {
    /// `PIE_QWEN35_MTP_INT8_LM_HEAD` (default off): give the speculative
    /// head an int8 view of `lm_head` beside the main path's bf16. The
    /// forward reads the same variable (`qwen35_mtp_int8_lm_head_enabled`).
    pub qwen35_mtp_int8_lm_head: bool,
    /// `PIE_NEMOTRON_DISABLE_TP_MAMBA_SHARD` inverted (default on): split
    /// the Mamba mixers across ranks. A kill switch for bisecting a
    /// numerical regression; flipping it changes the contract and therefore
    /// re-plans. The forward reads the same variable
    /// (`nemotron_h_tp_mamba_sharding_enabled`), which is why it is a knob
    /// and not a constant.
    pub nemotron_tp_mamba_sharding: bool,
}

impl Default for FamilyKnobs {
    fn default() -> Self {
        Self {
            qwen35_mtp_int8_lm_head: false,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The words `pie model build --quant` accepts, and what each means.
    ///
    /// This crate holds no caller: [`RuntimeQuant::resolve`] is reached from
    /// `src/ops/model/build.rs` in the WORKSPACE ROOT, which a `-p model`
    /// run never executes. That is why the rule was written down here and
    /// then measured at 23% — the one file in the crate with no test at
    /// all, because a grep for callers under `crates/` finds none and the
    /// binary lives beside `crates/` rather than under it.
    #[test]
    fn every_quantization_the_cli_accepts_resolves_to_one_outcome() {
        for (request, fp8_native, expected) in [
            ("", false, RuntimeQuant::None),
            ("", true, RuntimeQuant::None),
            ("int8", false, RuntimeQuant::Int8),
            ("mxfp4", false, RuntimeQuant::Mxfp4),
            ("int4", false, RuntimeQuant::Int4),
        ] {
            assert_eq!(
                RuntimeQuant::resolve(request, fp8_native),
                Ok(expected),
                "--quant {request:?} with fp8_native={fp8_native}"
            );
        }
    }

    /// FP8 is the one request the device can veto.
    ///
    /// Every other word means the same thing on every device, which is why
    /// `fp8_native` is a parameter rather than a field: it changes exactly
    /// one answer. `pie model build` prints "dropped, as serve would drop
    /// it" by comparing this function's answer against the word it passed
    /// in, so a rule that stopped collapsing would make the CLI bake in a
    /// requantization it announced as dropped, and one that always
    /// collapsed would make it silently bake in none.
    #[test]
    fn fp8_without_a_device_that_runs_it_is_no_request_at_all() {
        assert_eq!(
            RuntimeQuant::resolve("fp8", false),
            Ok(RuntimeQuant::None),
            "an fp8 request on a device without native FP8 collapses"
        );
        assert_eq!(
            RuntimeQuant::resolve("fp8", true),
            Ok(RuntimeQuant::Fp8),
            "and stands on a device with it"
        );
    }

    /// An unknown word is refused, and the refusal says which word.
    ///
    /// The alternative — falling through to `None` — is the failure this
    /// crate is written against: a typo (`--quant fp-8`) that builds
    /// successfully, serves bf16, and reports nothing.
    #[test]
    fn a_quantization_this_build_does_not_know_is_refused_by_name() {
        let message = RuntimeQuant::resolve("fp-8", true).expect_err("`fp-8` is no quantization");
        assert!(
            message.contains("fp-8"),
            "the refusal names the word the operator typed: {message:?}"
        );
        assert!(
            RuntimeQuant::resolve("bf16", true).is_err(),
            "`bf16` is the ABSENCE of a requantization and is spelled by \
             omitting the flag; accepting it here would give the operator \
             two spellings whose only difference is which one is checked"
        );
    }

    /// The variants are sent as their positions, and the positions are these.
    ///
    /// [`RuntimeQuant`] carries no `#[repr]` and no written-out
    /// discriminants, so `as u32` reports DECLARATION ORDER. The `Int4` doc
    /// says it must stay last for exactly this reason and nothing enforced
    /// it: reordering two variants renumbers the wire, both sides still
    /// compile, and the far side reads `Mxfp4` where the caller wrote
    /// `Int8`.
    #[test]
    fn the_wire_positions_are_the_ones_the_other_side_reads() {
        for (quant, position) in [
            (RuntimeQuant::None, 0),
            (RuntimeQuant::Fp8, 1),
            (RuntimeQuant::Int8, 2),
            (RuntimeQuant::Mxfp4, 3),
            (RuntimeQuant::Int4, 4),
        ] {
            assert_eq!(quant as u32, position, "{quant:?} is sent as {position}");
        }
    }

    /// The three mirrored enums send the numbers the C header declares.
    ///
    /// These carry `#[repr(u32)]` and spelled-out discriminants because
    /// they mirror `model/contract.hpp`. The numbers are restated here
    /// rather than read off the variant, since a test that derives them
    /// from the type agrees with any renumbering.
    #[test]
    fn the_mirrored_enums_send_the_numbers_the_header_declares() {
        assert_eq!(Component::Full as u32, 0);
        assert_eq!(Component::Text as u32, 1);
        assert_eq!(Component::Encode as u32, 2);

        assert_eq!(Mxfp4MoeRequest::Auto as u32, 0);
        assert_eq!(Mxfp4MoeRequest::RoutedDecode as u32, 1);
        assert_eq!(Mxfp4MoeRequest::NativeGemm as u32, 2);
        assert_eq!(Mxfp4MoeRequest::EagerBf16 as u32, 3);

        assert_eq!(Mxfp4MoePolicy::RoutedDecode as u32, 0);
        assert_eq!(Mxfp4MoePolicy::NativeGemm as u32, 1);
        assert_eq!(Mxfp4MoePolicy::EagerBf16 as u32, 2);
    }

    /// Every default is the inert choice at its own position.
    ///
    /// `boot.rs` states all seven fields and so never reaches these, but a
    /// family's fixture builds `Policy::default()` and any call that grows
    /// a field later fills it from here. The whole point of the type is
    /// that a value nobody asked for asks for nothing.
    #[test]
    fn the_default_policy_asks_for_nothing() {
        let policy = Policy::default();
        assert_eq!(policy.runtime_quant, RuntimeQuant::None);
        assert_eq!(policy.moe_request, Mxfp4MoeRequest::Auto);
        assert_eq!(policy.component, Component::Full);
        assert!(!policy.stream_routed_experts);
        assert_eq!(policy.projections, Projections::Fused);
        assert_eq!(policy.naming, Naming::Hf);
    }

    /// The one knob that defaults ON, and why it has a hand-written impl.
    ///
    /// `nemotron_tp_mamba_sharding` mirrors an env var spelled as a DISABLE
    /// (`PIE_NEMOTRON_DISABLE_TP_MAMBA_SHARD`), so its absence means on. A
    /// `#[derive(Default)]` on this struct would quietly make it off and
    /// un-shard every Mamba mixer — a contract change that re-plans, with
    /// no error anywhere.
    #[test]
    fn the_inverted_knob_defaults_to_the_sharding_being_on() {
        let knobs = FamilyKnobs::default();
        assert!(
            knobs.nemotron_tp_mamba_sharding,
            "the env var is a kill switch, so its absence is `true`"
        );
        assert!(
            !knobs.qwen35_mtp_int8_lm_head,
            "this one is spelled as an enable, so its absence is `false`"
        );
    }
}
