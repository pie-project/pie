//! The element type, as one enum for the whole tree.
//!
//! There were two, and they said the same thing in two spellings: the IR's
//! `model_ir::Dtype`, which named what a traced value computes in and what a
//! weight plane or kv page stores, and the loader's `checkpoint::types::DType`,
//! which named what a checkpoint tensor holds on disk. Neither vocabulary was a
//! subset of the other — the IR knew the sub-byte quant codes (`Fp4`, `Mxfp4`)
//! and the loader knew the wide ints and `Bool` — so every edge between them was
//! a hand-written table that could be wrong in one direction only, and the third
//! size table (`engine::transfer::dtype_bits`) could disagree with both.
//!
//! This crate is the union of the two vocabularies and nothing else: the enum,
//! its width, and serde. It is a leaf — it deps `serde` and no crate in the
//! tree — so anything from the loader to a kernel entry function can name it
//! without naming a plane it has no business in.
//!
//! `no_std`, and `serde` is a feature rather than a fact. Both are for the
//! guests: `eta-ir` is what a wasm inferlet imports, it is `no_std` with serde
//! off by default, and a leaf that forced either on would force it on every
//! guest build in the tree. The feature is **on by default**, so every host
//! consumer sees the crate it always saw; `eta-ir` is the one that takes it
//! with `default-features = false` and forwards its own `serde` feature here.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(test)]
extern crate std;

/// Element type as data, not a generic: monomorphization's guarantee moved to
/// the trace-time validator plus a launch-site match. The one such enum in the
/// stack — it names storage representations as well as compute elements, so a
/// weight plane, a kv page, a checkpoint tensor and a traced value all say what
/// they hold in one spelling.
///
/// `Mxfp4` is a weight plane's 32-code block packed to 16 bytes; the companion
/// `.scales` plane beside it is `E8m0`, which is only ever that companion and
/// never something an author declares. `MlxU4` is the other packed weight
/// plane, and it carries two companions rather than one — `.scales` and
/// `.biases`, both `Bf16`. `Fp8E4m3`, `I8` and `Fp4` name kv-page
/// quant schemes. What a scheme's granularity is (per-tensor vs per-token-head)
/// and how wide an fp4 block runs are not facts about the element — they are
/// sibling fields of the cache row that chose the scheme.
///
/// The `#[serde(alias = ...)]` attributes below are the loader's old spellings.
/// A `LoadPlan` is `Serialize`, and plans recorded before the two enums merged
/// name their dtypes `"BF16"`, `"F8E4M3"`, `"F8E5M2"` and `"E8M0"`; the aliases
/// keep those readable. Nothing *writes* them any more — a plan serialized
/// today carries the spelling on the variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Dtype {
    // ── floats ──────────────────────────────────────────────────────────
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// bfloat16: binary32's exponent range at binary16's width.
    #[cfg_attr(feature = "serde", serde(alias = "BF16"))]
    Bf16,
    /// OCP FP8 with a 4-bit exponent and a 3-bit mantissa.
    #[cfg_attr(feature = "serde", serde(alias = "F8E4M3"))]
    Fp8E4m3,
    /// OCP FP8 with a 5-bit exponent and a 2-bit mantissa.
    #[cfg_attr(feature = "serde", serde(alias = "F8E5M2"))]
    Fp8E5m2,
    /// Bare 4-bit float codes, two per byte, scaled by whatever chose them.
    Fp4,
    /// OCP Microscaling FP4: a 32-code block of `Fp4` packed to 16 bytes,
    /// with an `E8m0` scale per block in the companion `.scales` plane.
    Mxfp4,
    /// MLX's affine 4-bit weight codes: one unsigned 4-bit code per element,
    /// eight to a `U32` word, with one `Bf16` scale and one `Bf16` zero point
    /// per 64-code group -- an element is `code * scale + bias`. The two
    /// companion planes are named `.scales` and `.biases`, which is MLX's own
    /// convention and the one a converted checkpoint is bound by.
    ///
    /// `Mxfp4`'s sibling and not its cousin. Both name a weight bank's codes
    /// plane, neither is anything an activation is stated in, and both are
    /// four bits wide; what differs is the group width and what a group's
    /// companion holds — an exponent byte there, a scale AND an offset here.
    /// The whole scheme is `checkpoint::types::QuantScheme::MlxAffineU4`,
    /// which is where the group width and the scale form are settled. This
    /// name is only what a weight declaration stamps to reach it.
    MlxU4,
    /// **MLX's affine 8-bit weight codes**, `MlxU4`'s own scheme at twice the
    /// width: one unsigned 8-bit code per element, four to a `U32` word, with
    /// one `Bf16` scale and one `Bf16` zero point per 64-code group. An
    /// element is `code * scale + bias`, exactly as at four bits, and the
    /// companion planes carry MLX's same `.scales` and `.biases` names.
    ///
    /// **IT EXISTS BECAUSE ONE CHECKPOINT MIXES THE TWO, AND MIXES THEM ON
    /// PURPOSE.** `mlx_lm` lets a family name planes that must not be
    /// quantized as hard as the rest, through `quant_predicate`, and
    /// `gpt_oss.py`'s says
    ///
    /// ```python
    /// if path.endswith("router"):
    ///     return {"group_size": 64, "bits": 8}
    /// ```
    ///
    /// so `mlx-community/gpt-oss-20b-MXFP4-Q4` publishes a stack whose
    /// attention is affine-U4 and whose twenty-four MoE ROUTER GATES are
    /// affine-U8. A router picks which experts a token visits; four bits of a
    /// `[32, 2880]` gate is a different set of experts, and the model that
    /// results is not the model. `qwen3_5.py` and `gemma4_text.py` carry the
    /// same predicate for their own gates, so this is a convention rather than
    /// one file's accident.
    ///
    /// **THE WIDTH IS A NUMBER THE PLANE CARRIES, NOT A NEW ARITHMETIC.**
    /// `QuantScheme::MlxAffineU4` names the SCHEME — affine codes, 64 to a
    /// group, bf16 scale and bf16 offset — and `QuantSpec::bits_per_element`
    /// has always been the field that says how wide a code is. Both widths
    /// therefore land on that one scheme, and the affine kernels already
    /// stamp both: `kernels_metal::linear::quant`'s `WIDTHS` is `[4, 8]`, and
    /// the `bits` its points are chosen by comes off
    /// `checkpoint::plan::Landing::affine_point_of`, which reads the plane's
    /// own spec. What this variant adds is the ability for a MODEL TEXT to say
    /// which width a weight is, which is the one thing `Dtype` is for.
    MlxU8,
    /// `MlxU4` grouped by THIRTY-TWO codes instead of sixty-four — the same
    /// affine scheme, the same `.scales`/`.biases` companions, half the
    /// group.
    ///
    /// **IT EXISTS BECAUSE A ROW CAN BE TOO NARROW TO GROUP BY 64.** MLX
    /// quantizes along the last axis and requires the group to divide it;
    /// qwen4's PLE n-gram table stores 160-wide rows, `160 % 64 != 0`, so
    /// `mlx_lm.convert` drops that one tensor family to `group_size: 32`
    /// (`Qwen3.8-Flash-Next-MLX`'s `config.json` lists all 128 shards under
    /// it) while everything beside it stays at 64. The width is four bits
    /// either way; `QuantSpec::group_size` has always been the field that
    /// says how many codes share a scale, and this variant is what lets a
    /// model TEXT say it — `MlxU8`'s own argument, one spec field over.
    MlxU4G32,
    /// OCP Microscaling's 8-bit exponent-only scale format: the stored byte
    /// `b` denotes `2^(b - 127)`. It carries no sign and no mantissa, so it
    /// only ever appears as the scale beside a block-scaled tensor -- which is
    /// why `QuantScheme` long knew it only as half of `Mxfp4E2M1E8M0`.
    /// DeepSeek-V4 pairs it with FP8-E4M3 weights instead, a combination that
    /// composite cannot name.
    #[cfg_attr(feature = "serde", serde(alias = "E8M0"))]
    E8m0,

    // ── ints ────────────────────────────────────────────────────────────
    /// Signed 64-bit.
    I64,
    /// Signed 32-bit.
    I32,
    /// Signed 16-bit.
    I16,
    /// Signed 8-bit.
    I8,
    /// Unsigned 64-bit.
    U64,
    /// Unsigned 32-bit.
    U32,
    /// Unsigned 16-bit.
    U16,
    /// Unsigned 8-bit — also the byte a packed quant plane is carried in.
    U8,

    // ── logical ─────────────────────────────────────────────────────────
    /// One byte per element, as every checkpoint format stores it.
    Bool,
}

impl Dtype {
    /// Bits one element of this dtype occupies.
    ///
    /// Bits and not bytes because the sub-byte formats are real: `Fp4` and
    /// `Mxfp4` pack two elements per byte, and a `size() -> usize` that rounded
    /// them up to one would over-report every quantized pool by 2×.
    #[must_use]
    pub const fn bits(self) -> u64 {
        match self {
            Self::I64 | Self::U64 => 64,
            Self::F32 | Self::I32 | Self::U32 => 32,
            Self::F16 | Self::Bf16 | Self::I16 | Self::U16 => 16,
            Self::Fp8E4m3
            | Self::Fp8E5m2
            | Self::E8m0
            | Self::I8
            | Self::U8
            | Self::MlxU8
            | Self::Bool => 8,
            Self::Fp4 | Self::Mxfp4 | Self::MlxU4 | Self::MlxU4G32 => 4,
        }
    }

    /// Bytes one element occupies, rounded up.
    ///
    /// Exact for every dtype a checkpoint stores one element per address in,
    /// and a deliberate over-report for the two sub-byte codes — an element of
    /// `Fp4` has no address of its own. Multiply [`bits`](Dtype::bits) and
    /// divide once when the count is known instead, wherever the answer is a
    /// span rather than a stride.
    #[must_use]
    pub const fn bytes_ceil(self) -> u64 {
        self.bits().div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::Dtype;
    #[cfg(feature = "serde")]
    use std::vec;
    #[cfg(feature = "serde")]
    use std::vec::Vec;

    /// The width table, against the two it was merged from.
    #[test]
    fn widths_are_what_both_tables_said() {
        // checkpoint's `DType::bytes`.
        assert_eq!(Dtype::I64.bytes_ceil(), 8);
        assert_eq!(Dtype::U64.bytes_ceil(), 8);
        assert_eq!(Dtype::F32.bytes_ceil(), 4);
        assert_eq!(Dtype::I32.bytes_ceil(), 4);
        assert_eq!(Dtype::U32.bytes_ceil(), 4);
        assert_eq!(Dtype::F16.bytes_ceil(), 2);
        assert_eq!(Dtype::Bf16.bytes_ceil(), 2);
        assert_eq!(Dtype::I16.bytes_ceil(), 2);
        assert_eq!(Dtype::U16.bytes_ceil(), 2);
        for narrow in [
            Dtype::Fp8E4m3,
            Dtype::Fp8E5m2,
            Dtype::E8m0,
            Dtype::I8,
            Dtype::U8,
            Dtype::Bool,
        ] {
            assert_eq!(narrow.bytes_ceil(), 1);
        }
        // `engine::transfer::dtype_bits` — the sub-byte half nobody else had.
        assert_eq!(Dtype::Fp4.bits(), 4);
        assert_eq!(Dtype::Mxfp4.bits(), 4);
        assert_eq!(Dtype::MlxU4.bits(), 4);
    }

    /// A plan recorded under the loader's old spellings still reads.
    #[cfg(feature = "serde")]
    #[test]
    fn the_old_loader_spellings_still_deserialize() {
        let old = r#"["BF16","F8E4M3","F8E5M2","E8M0"]"#;
        let read: Vec<Dtype> = serde_json::from_str(old).expect("old spellings read");
        assert_eq!(
            read,
            vec![Dtype::Bf16, Dtype::Fp8E4m3, Dtype::Fp8E5m2, Dtype::E8m0]
        );
    }

    /// What it writes is the variant's own spelling, not the alias.
    #[cfg(feature = "serde")]
    #[test]
    fn what_it_writes_is_the_variant_name() {
        let text = serde_json::to_string(&Dtype::Bf16).expect("write");
        assert_eq!(text, "\"Bf16\"");
    }
}
