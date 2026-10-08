//! Bonsai RHT sign ingest (M3b).
//!
//! The Bonsai rotation is a fixed Randomized Hadamard Transform `H·(S·x)`: `H`
//! is the normalized Sylvester block-1024 matrix pie regenerates at load with
//! the existing `elementwise.hadamard` op (it is NOT stored), and `S` is a ±1
//! sign diagonal that IS stored, in the GGUF metadata. This module ingests the
//! sign diagonals `S` and exposes them as registered params keyed by input
//! width, ready to feed the Hadamard op's `signs` argument in M3c. It does NOT
//! wire them into the forward pass — that is M3c.
//!
//! ## Where the signs live and how they are encoded
//!
//! Two GGUF metadata KV arrays hold every sign, mirroring the PrismML llama.cpp
//! fork (`src/llama-model.cpp`, `sign_mode == "explicit"`):
//!
//! * `prism.hadamard.sign_widths` — an `int32` array of the per-group widths, in
//!   the order the values are concatenated. For Ternary-Bonsai-2-27B this is
//!   `[5120, 6144, 17408]`.
//! * `prism.hadamard.sign_values` — an `int32` array of every sign, each `+1` or
//!   `-1` (one whole `int32` per sign, NOT bit-packed), the widths laid end to
//!   end. Its length equals the sum of the widths (`28672`).
//!
//! The fork splits `sign_values` into contiguous slices of each `sign_widths`
//! entry (in order), keying the resulting diagonal by its width, and later
//! selects a weight's diagonal purely by the weight's input width
//! (`weight->ne[0]`). [`decode_signs`] reproduces that split and validation
//! byte-for-byte; [`sign_param_name`] reproduces the fork's on-device tensor
//! name `prism.hadamard.signs.<W>`.
//!
//! ## Width → site mapping (confirmed against the fork graph, oracle §6)
//!
//! The diagonal is chosen by the rotated projection's input width:
//!
//! * [`WIDTH_HIDDEN`] (`5120`, `n_embd`) — every hidden/residual-stream input:
//!   `attn_qkv`, `attn_gate` (`z`), the o-proj input (`attn_output`), `ffn_gate`,
//!   `ffn_up`, the output head (`result_norm` → `output.weight`), the
//!   token-embedding inverse-after-lookup, and — on full-attention layers —
//!   `attn_q` / `attn_k`.
//! * [`WIDTH_SSM_OUT`] (`6144`, GDN `ssm.inner_size`) — the GDN `ssm_out` input.
//!   NOTE: with `gdn_v_grouped=true` the fork reorders the GDN output's v-heads
//!   (`{128,16,3,5} → {128,3,16,5}`) BEFORE the sign multiply; the sign vector
//!   itself is stored/consumed unpermuted (this module returns it unpermuted).
//! * [`WIDTH_FFN_DOWN`] (`17408`, `n_ff`) — the FFN intermediate (`ffn_down`
//!   input).
//!
//! All three widths are whole multiples of the block size 1024 (`5×`, `6×`,
//! `17×`), as the transform requires.

use std::collections::BTreeMap;
use std::fmt;

use poem::{Dtype, Weight};
use ztensor::format::cbor::Value;

/// GGUF metadata key: the Hadamard block size (the sign widths must divide it).
pub const BLOCK_SIZE_KEY: &str = "prism.hadamard.block_size";
/// GGUF metadata key: `"identity"` (no signs) or `"explicit"` (signs present).
pub const SIGN_MODE_KEY: &str = "prism.hadamard.sign_mode";
/// GGUF metadata key: the `int32` array of per-group sign widths, in order.
pub const SIGN_WIDTHS_KEY: &str = "prism.hadamard.sign_widths";
/// GGUF metadata key: the `int32` array of every `±1` sign, widths end to end.
pub const SIGN_VALUES_KEY: &str = "prism.hadamard.sign_values";
/// GGUF metadata key: whether the GDN v-heads are stored **tiled** ("v-grouped",
/// v-head `p` pairs k-head `p % k_heads`) rather than **block** (`p / rep`). When
/// true, the GGUF import reorders every v-head-indexed GDN tensor tiled→block so
/// pie's block-pairing GDN scan kernel pairs v→k correctly, leaving the shared
/// kernel untouched. Absent or false => the file is already block-stored.
pub const GDN_V_GROUPED_KEY: &str = "prism.hadamard.gdn_v_grouped";

/// Read the [`GDN_V_GROUPED_KEY`] flag from an opened GGUF source's top-level
/// metadata. Absent or false => the GDN v-heads are already in block order.
#[must_use]
pub fn gdn_v_grouped(src: &ztensor::Source) -> bool {
    gdn_v_grouped_in(src.attributes())
}

/// The [`gdn_v_grouped`] decision over a bare metadata map — absent map, absent
/// key, or a non-`true` value all read as `false`. Split out so it is testable
/// without opening a GGUF.
#[must_use]
pub fn gdn_v_grouped_in(attributes: Option<&Value>) -> bool {
    matches!(
        attributes.and_then(|attrs| attrs.get(GDN_V_GROUPED_KEY)),
        Some(Value::Bool(true))
    )
}

/// Input width of every hidden/residual-stream rotated site (`n_embd`).
pub const WIDTH_HIDDEN: u32 = 5120;
/// Input width of the GDN `ssm_out` rotated site (`ssm.inner_size`).
pub const WIDTH_SSM_OUT: u32 = 6144;
/// Input width of the FFN `ffn_down` rotated site (`n_ff`).
pub const WIDTH_FFN_DOWN: u32 = 17408;

/// The three Bonsai sign widths, in the order they are stored in the GGUF.
pub const BONSAI_SIGN_WIDTHS: [u32; 3] = [WIDTH_HIDDEN, WIDTH_SSM_OUT, WIDTH_FFN_DOWN];

/// A decoded `±1` sign diagonal `S` for one input width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignVector {
    /// The input width this diagonal rotates (a whole number of blocks).
    pub width: u32,
    /// One entry per column, each exactly `+1` or `-1`, in stored order.
    pub signs: Vec<i8>,
}

impl SignVector {
    /// The diagonal as `f32` `±1`, the element type the Metal Hadamard op binds
    /// its `signs` buffer as (matching the fork's on-device `GGML_TYPE_F32` sign
    /// tensor). `±1` is exact in `f32`, so this loses nothing.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        self.signs.iter().map(|&s| f32::from(s)).collect()
    }

    /// The `f32` `±1` diagonal as little-endian bytes, ready to write into a
    /// device buffer.
    #[must_use]
    pub fn to_f32_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.signs.len() * 4);
        for &s in &self.signs {
            out.extend_from_slice(&f32::from(s).to_le_bytes());
        }
        out
    }

    /// The `±1` diagonal as `bf16` little-endian bytes — the compute dtype the
    /// `Ptq1_0`-served Bonsai binds its sign banks in (`Shell::register_adapter`).
    /// `+1` is `0x3F80`, `-1` is `0xBF80`; both are exact in `bf16`. Each sign is
    /// the top 16 bits of its `f32` bit pattern (round-to-nearest is a no-op for a
    /// value whose low 16 mantissa bits are all zero, as `±1` are).
    #[must_use]
    pub fn to_bf16_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.signs.len() * 2);
        for &s in &self.signs {
            let hi = (f32::from(s).to_bits() >> 16) as u16;
            out.extend_from_slice(&hi.to_le_bytes());
        }
        out
    }
}

/// Why a GGUF's Hadamard sign metadata could not be ingested.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignError {
    /// The source carried no top-level metadata map at all.
    NoAttributes,
    /// A required key is absent.
    Missing(&'static str),
    /// A key is present but not the CBOR shape it must be (array / int / text).
    Malformed(&'static str),
    /// `sign_mode` is neither `"identity"` nor `"explicit"`.
    UnsupportedMode(String),
    /// A width is `<= 0`, does not divide the block size, or runs past the end
    /// of `sign_values`.
    BadWidth { width: i64 },
    /// A stored sign is not exactly `+1` or `-1`.
    NotPlusMinusOne { width: u32, at: usize, value: i64 },
    /// The widths do not consume `sign_values` exactly.
    LengthMismatch { consumed: usize, total: usize },
}

impl fmt::Display for SignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignError::NoAttributes => {
                write!(f, "prism.hadamard: the source carries no metadata map")
            }
            SignError::Missing(key) => write!(f, "prism.hadamard: missing `{key}`"),
            SignError::Malformed(key) => {
                write!(f, "prism.hadamard: `{key}` is not the expected type")
            }
            SignError::UnsupportedMode(mode) => {
                write!(
                    f,
                    "prism.hadamard.sign_mode `{mode}` is not identity|explicit"
                )
            }
            SignError::BadWidth { width } => {
                write!(f, "prism.hadamard: invalid sign width {width}")
            }
            SignError::NotPlusMinusOne { width, at, value } => write!(
                f,
                "prism.hadamard: sign {at} of width {width} is {value}, not +/-1",
            ),
            SignError::LengthMismatch { consumed, total } => write!(
                f,
                "prism.hadamard.sign_values length mismatch: widths consume {consumed} of {total}",
            ),
        }
    }
}

impl std::error::Error for SignError {}

/// Ingest the Bonsai RHT sign diagonals straight from an opened GGUF source,
/// keyed by input width. Convenience wrapper over [`decode_signs`] reading the
/// source's top-level metadata.
pub fn signs_from_gguf(src: &ztensor::Source) -> Result<BTreeMap<u32, SignVector>, SignError> {
    let attrs = src.attributes();
    // The Bonsai RHT is defined on the canonical 1024-wide Hadamard block. The
    // generic `decode_signs` accepts any positive block (it is reused for smaller
    // test geometries); here, at the real GGUF ingest, pin explicit-mode signs to
    // 1024 so a checkpoint declaring a block the serving kernels never use is
    // rejected rather than silently decoded.
    if let Some(attrs) = attrs {
        let explicit = attrs
            .get(SIGN_MODE_KEY)
            .and_then(|m| m.as_text())
            .is_none_or(|m| m == "explicit");
        let block = attrs.get(BLOCK_SIZE_KEY).and_then(|b| b.as_u64());
        if explicit && block.is_some_and(|b| b != 1024) {
            return Err(SignError::Malformed(BLOCK_SIZE_KEY));
        }
    }
    decode_signs(attrs)
}

/// Decode the sign diagonals from a GGUF's top-level metadata map, reproducing
/// the fork's `explicit`-mode split byte-for-byte.
///
/// * `identity` mode (or a table that would be empty) yields an empty map — no
///   diagonals, the transform is a plain Hadamard.
/// * `explicit` mode splits `sign_values` into one diagonal per `sign_widths`
///   entry (in order), keyed by width, validating each is a whole number of
///   blocks, every value is `±1`, and the widths consume `sign_values` exactly.
pub fn decode_signs(attributes: Option<&Value>) -> Result<BTreeMap<u32, SignVector>, SignError> {
    let attrs = attributes.ok_or(SignError::NoAttributes)?;

    // `identity` mode stores no signs and reads as a plain Hadamard; only
    // `explicit` carries a table. Absent key => treat as explicit (the fork
    // requires the key, but the widths/values below are the real gate).
    if let Some(mode) = attrs.get(SIGN_MODE_KEY) {
        let mode = mode.as_text().ok_or(SignError::Malformed(SIGN_MODE_KEY))?;
        match mode {
            "identity" => return Ok(BTreeMap::new()),
            "explicit" => {}
            other => return Err(SignError::UnsupportedMode(other.to_string())),
        }
    }

    let block_size = attrs
        .get(BLOCK_SIZE_KEY)
        .ok_or(SignError::Missing(BLOCK_SIZE_KEY))?
        .as_u64()
        .filter(|&b| b > 0)
        .ok_or(SignError::Malformed(BLOCK_SIZE_KEY))?;

    let widths = attrs
        .get(SIGN_WIDTHS_KEY)
        .ok_or(SignError::Missing(SIGN_WIDTHS_KEY))?
        .as_array()
        .ok_or(SignError::Malformed(SIGN_WIDTHS_KEY))?;
    let values = attrs
        .get(SIGN_VALUES_KEY)
        .ok_or(SignError::Missing(SIGN_VALUES_KEY))?
        .as_array()
        .ok_or(SignError::Malformed(SIGN_VALUES_KEY))?;

    let mut out: BTreeMap<u32, SignVector> = BTreeMap::new();
    let mut off = 0usize;
    for w in widths {
        let width = as_i64(w).ok_or(SignError::Malformed(SIGN_WIDTHS_KEY))?;
        // Mirror the fork: positive, a whole number of blocks, and in bounds.
        let fits = width > 0
            && u64::try_from(width).is_ok_and(|w| w % block_size == 0)
            && off
                .checked_add(usize::try_from(width).unwrap_or(usize::MAX))
                .is_some_and(|end| end <= values.len());
        if !fits {
            return Err(SignError::BadWidth { width });
        }
        let width_u = u32::try_from(width).map_err(|_| SignError::BadWidth { width })?;
        let span = width as usize;

        let mut signs = Vec::with_capacity(span);
        for (k, slot) in values[off..off + span].iter().enumerate() {
            let v = as_i64(slot).ok_or(SignError::Malformed(SIGN_VALUES_KEY))?;
            if v != 1 && v != -1 {
                return Err(SignError::NotPlusMinusOne {
                    width: width_u,
                    at: k,
                    value: v,
                });
            }
            signs.push(v as i8);
        }
        off += span;

        // Keyed by width, last wins on a repeat — the fork's `std::map` behaviour.
        out.insert(
            width_u,
            SignVector {
                width: width_u,
                signs,
            },
        );
    }

    if off != values.len() {
        return Err(SignError::LengthMismatch {
            consumed: off,
            total: values.len(),
        });
    }

    Ok(out)
}

/// The registered-param name of the sign diagonal for a given input width,
/// matching the fork's on-device tensor `prism.hadamard.signs.<W>`.
#[must_use]
pub fn sign_param_name(width: u32) -> String {
    format!("prism.hadamard.signs.{width}")
}

/// The registered-param declaration for one width's sign diagonal: a host-provided
/// `[1, width]` `f32` `±1` bank, the same `ParamSource::Registered` mechanism
/// adapters use. M3d binds the decoded [`SignVector`] bytes to it via
/// `Shell::register_adapter` and feeds it to the Hadamard op's `signs` argument.
///
/// The leading `1` is load-bearing: the engine's registered-bank bookkeeper
/// (`engine_metal::weights::banks`) reads `shape[0]` as the number of adapter
/// SEATS and the remaining axes as the per-seat rectangle. A whole width-wide
/// diagonal is one seat, so the shape is `[1, width]` (one seat, `width` wide),
/// never a bare `[width]` (which would register `width` seats of one element).
/// The Metal Hadamard kernel keys the sign count off `rows * width` = `width`
/// regardless, so the leading `1` is a binding detail, not a numeric one.
///
/// Declared `f32` to match the fork's on-device sign tensor. The Metal Hadamard
/// kernel requires `signs.dtype == activation.dtype`; when M3d rotates a non-`f32`
/// activation it re-declares the bank in that compute dtype (`±1` is exact in
/// both `f32` and `bf16`).
#[must_use]
pub fn sign_param(width: u32) -> Weight {
    Weight::sym(sign_param_name(width), [1, u64::from(width)], Dtype::F32).registered()
}

/// The registered-param declaration for one width's sign diagonal in a chosen
/// element type. The Metal Hadamard kernel requires `signs.dtype ==
/// activation.dtype`, so M3d declares the bank in the activation's compute dtype
/// (`bf16` for the `Ptq1_0`-served Bonsai). `±1` is exact in `bf16` and `f32`,
/// so the element type is a device-binding detail, not a numeric one. The `[1,
/// width]` shape is the one-seat registered-bank shape — see [`sign_param`].
#[must_use]
pub fn sign_param_in(width: u32, dtype: Dtype) -> Weight {
    Weight::sym(sign_param_name(width), [1, u64::from(width)], dtype).registered()
}

/// The registered-param declarations for every ingested diagonal, width order.
#[must_use]
pub fn sign_params(signs: &BTreeMap<u32, SignVector>) -> Vec<Weight> {
    signs.keys().copied().map(sign_param).collect()
}

/// Read a CBOR integer scalar as `i64`, spanning GGUF signed ints (`Nint`) and
/// unsigned (`Uint`). `Nint(n)` encodes `-1 - n`.
fn as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Uint(n) => i64::try_from(*n).ok(),
        Value::Nint(n) => i64::try_from(*n).ok().map(|n| -1 - n),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poem_ir::ParamSource;
    use ztensor::format::cbor;

    /// Build a `sign_values`-style CBOR int array (GGUF `int32` encoding: `Nint`
    /// for negatives, `Uint` for non-negatives).
    fn int_array(xs: &[i64]) -> Value {
        Value::Array(xs.iter().map(|&x| cbor::Value::from(x)).collect())
    }

    fn attrs(mode: &str, block: u64, widths: &[i64], values: &[i64]) -> Value {
        Value::Map(vec![
            (Value::Text(SIGN_MODE_KEY.into()), Value::Text(mode.into())),
            (Value::Text(BLOCK_SIZE_KEY.into()), Value::Uint(block)),
            (Value::Text(SIGN_WIDTHS_KEY.into()), int_array(widths)),
            (Value::Text(SIGN_VALUES_KEY.into()), int_array(values)),
        ])
    }

    #[test]
    fn explicit_signs_split_by_width_in_order() {
        // Two blocks of 2: width 2 = [+1,-1], width 4 = [-1,-1,+1,+1].
        let a = attrs("explicit", 2, &[2, 4], &[1, -1, -1, -1, 1, 1]);
        let got = decode_signs(Some(&a)).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[&2].signs, vec![1, -1]);
        assert_eq!(got[&4].signs, vec![-1, -1, 1, 1]);
        assert_eq!(got[&2].width, 2);
        // f32 view is exact ±1.
        assert_eq!(got[&4].to_f32(), vec![-1.0, -1.0, 1.0, 1.0]);
    }

    #[test]
    fn identity_mode_has_no_signs() {
        let a = attrs("identity", 2, &[2], &[1, -1]);
        assert!(decode_signs(Some(&a)).unwrap().is_empty());
    }

    #[test]
    fn a_non_pm1_value_is_rejected() {
        let a = attrs("explicit", 2, &[2], &[1, 2]);
        assert_eq!(
            decode_signs(Some(&a)),
            Err(SignError::NotPlusMinusOne {
                width: 2,
                at: 1,
                value: 2,
            })
        );
    }

    #[test]
    fn a_width_off_the_block_grid_is_rejected() {
        // block 4, width 2 does not divide it.
        let a = attrs("explicit", 4, &[2], &[1, -1]);
        assert_eq!(
            decode_signs(Some(&a)),
            Err(SignError::BadWidth { width: 2 })
        );
    }

    #[test]
    fn leftover_values_are_a_length_mismatch() {
        let a = attrs("explicit", 2, &[2], &[1, -1, 1, 1]);
        assert_eq!(
            decode_signs(Some(&a)),
            Err(SignError::LengthMismatch {
                consumed: 2,
                total: 4,
            })
        );
    }

    #[test]
    fn missing_attributes_and_keys_are_reported() {
        assert_eq!(decode_signs(None), Err(SignError::NoAttributes));
        let no_widths = Value::Map(vec![
            (
                Value::Text(SIGN_MODE_KEY.into()),
                Value::Text("explicit".into()),
            ),
            (Value::Text(BLOCK_SIZE_KEY.into()), Value::Uint(2)),
            (Value::Text(SIGN_VALUES_KEY.into()), int_array(&[1, -1])),
        ]);
        assert_eq!(
            decode_signs(Some(&no_widths)),
            Err(SignError::Missing(SIGN_WIDTHS_KEY))
        );
    }

    #[test]
    fn the_registered_param_is_a_host_bank_named_like_the_fork() {
        assert_eq!(sign_param_name(6144), "prism.hadamard.signs.6144");
        let w = sign_param(WIDTH_FFN_DOWN);
        assert_eq!(w.name, "prism.hadamard.signs.17408");
        // One registered SEAT, `width` wide — see `sign_param`.
        assert_eq!(w.shape, vec![1, u64::from(WIDTH_FFN_DOWN)]);
        assert_eq!(w.dtype, Dtype::F32);
        assert!(matches!(w.source, ParamSource::Registered));
    }
}
