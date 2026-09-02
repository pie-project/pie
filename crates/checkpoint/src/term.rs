//! The algebra ([`Fmt`]: group width, code element, gain, offset) a [`QuantScheme`] name stands for, from [`QuantSpec::term`]. `None` is always a refusal, never a default.

use dtype::Dtype;
use dtype::{Elem, Fmt, Group, Off, spells};

use crate::types::{QuantScheme, QuantSpec};

// ─────────────────────────────────────────────────────────────────────────
// The rows this crate spells that no kernel serves
// ─────────────────────────────────────────────────────────────────────────
// Named by their own spelling uppercased; a `const` assertion ties the two.

/// A leaf, once, for the promoted references below.
const F16: Fmt<'static> = Fmt::Elem(Elem::F16);
/// See [`F16`].
const F32: Fmt<'static> = Fmt::Elem(Elem::F32);

/// ggml's `q4_0`: excess-binary 4-bit codes in blocks of 32 with an f16
/// scale and no offset.
const G32_I4_F16_N: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::I(4),
    gain: &F16,
    offset: None,
};
const _: () = assert!(spells(&G32_I4_F16_N, "g32_i4_f16_n"));

/// ggml's `q4_1`: unsigned nibbles in blocks of 32 under an f16 scale and an
/// f16 offset that adds — `nibble * d + m`, the affine sibling of `q4_0`.
const G32_U4_F16_B_F16: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::U(4),
    gain: &F16,
    offset: Some(Off::Post(&F16)),
};
const _: () = assert!(spells(&G32_U4_F16_B_F16, "g32_u4_f16_b_f16"));

/// ggml's `q5_0`: `q4_0` with a fifth bit per element, excess-16 over five
/// bits. The plane the fifth bits live in is a container fact, not a term.
const G32_I5_F16_N: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::I(5),
    gain: &F16,
    offset: None,
};
const _: () = assert!(spells(&G32_I5_F16_N, "g32_i5_f16_n"));

/// ggml's `q5_1`: [`G32_U4_F16_B_F16`] at five bits, the fifth carried in the
/// same plane [`G32_I5_F16_N`] uses.
const G32_U5_F16_B_F16: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::U(5),
    gain: &F16,
    offset: Some(Off::Post(&F16)),
};
const _: () = assert!(spells(&G32_U5_F16_B_F16, "g32_u5_f16_b_f16"));

/// ggml's `q8_0`: excess-binary 8-bit codes in blocks of 32 with an f16
/// scale.
const G32_I8_F16_N: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::I(8),
    gain: &F16,
    offset: None,
};
const _: () = assert!(spells(&G32_I8_F16_N, "g32_i8_f16_n"));

/// FP8-E5M2 weights with one f32 scale per output row —
/// [`Dtype::E4m3row`]'s repr at the other FP8 width, unserved.
const GR_E5M2_F32_N: Fmt<'static> = Fmt::Q {
    g: Group::Row,
    elem: Elem::E { e: 5, m: 2 },
    gain: &F32,
    offset: None,
};
const _: () = assert!(spells(&GR_E5M2_F32_N, "gr_e5m2_f32_n"));

/// Symmetric 8-bit codes with one f32 scale per output row.
const GR_I8_F32_N: Fmt<'static> = Fmt::Q {
    g: Group::Row,
    elem: Elem::I(8),
    gain: &F32,
    offset: None,
};
const _: () = assert!(spells(&GR_I8_F32_N, "gr_i8_f32_n"));

// ─────────────────────────────────────────────────────────────────────────
// Shapes the parametric families take
// ─────────────────────────────────────────────────────────────────────────

/// The affine row whose gain and offset are both a bf16 leaf, offset adding:
/// `code * scale + bias`. MLX's shape at whichever width/group the plane declares.
fn post_affine_bf16(group: u32, elem: Elem) -> Fmt<'static> {
    const BF16: Fmt<'static> = Fmt::Elem(Elem::Bf16);
    Fmt::Q {
        g: Group::N(group),
        elem,
        gain: &BF16,
        offset: Some(Off::Post(&BF16)),
    }
}

/// The affine row whose zero point is an integer code of the weights' own
/// four-bit width, subtracted before the gain applies: `scale * (code - zero)`.
fn pre_affine_u4(group: u32) -> Fmt<'static> {
    const U4: Fmt<'static> = Fmt::Elem(Elem::U(4));
    Fmt::Q {
        g: Group::N(group),
        elem: Elem::U(4),
        gain: &F16,
        offset: Some(Off::Pre(&U4)),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The door
// ─────────────────────────────────────────────────────────────────────────

impl QuantSpec {
    /// The QNF term this encoding stands for, or `None` for a scheme with
    /// none yet. `bits`/`group_size` fill in the parametric families only;
    /// they're inert for GGUF blocks (own layout fixes width) and per-channel
    /// schemes (one f32 factor per output row).
    #[must_use]
    pub fn term(&self) -> Option<Fmt<'static>> {
        let bits = self.normalized_bits();
        let group = self.normalized_group_size();
        Some(match self.scheme {
            // A raw tensor's term is its dtype's own `repr`.
            QuantScheme::None => return None,

            // Plain f8/i8 elements with one f32 factor per output row. A
            // block-scaled FP8 checkpoint is a different row this can't name.
            QuantScheme::Fp8E4M3 => *Dtype::E4m3row.repr(),
            QuantScheme::Fp8E5M2 => GR_E5M2_F32_N,
            QuantScheme::Int8Symmetric => GR_I8_F32_N,

            // Whether the zero point subtracts in the code domain or the
            // offset adds in the value domain is undecided, so this refuses.
            QuantScheme::Int8Asymmetric => return None,

            // AWQ and GPTQ differ only in `packing.order`; the arithmetic is
            // one row here.
            QuantScheme::AwqInt4 | QuantScheme::GptqInt4 => {
                if bits != 4 {
                    // Named at four bits only in `zt.quant_group/1`.
                    return None;
                }
                pre_affine_u4(group)
            }

            // MLX's affine codes at width 2, 4, or 8. The offset adds:
            // `code * scale + bias`, a value-domain bias, not a zero point.
            QuantScheme::MlxAffineU4 => match bits {
                2 | 4 | 8 => post_affine_bf16(group, Elem::U(bits)),
                _ => return None,
            },

            // The zero point is the 8, so codes are excess-binary with no
            // offset. Gain isn't fixed by the scheme; f16 is what this family stores.
            QuantScheme::Int4B8 => {
                if bits != 4 {
                    return None;
                }
                Fmt::Q {
                    g: Group::N(group),
                    elem: Elem::I(4),
                    gain: &F16,
                    offset: None,
                }
            }

            // OCP Microscaling, two-plane form.
            QuantScheme::Mxfp4E2M1E8M0 => {
                const E8M0: Fmt<'static> = Fmt::Elem(Elem::E { e: 8, m: 0 });
                Fmt::Q {
                    g: Group::N(group),
                    elem: Elem::E { e: 2, m: 1 },
                    gain: &E8M0,
                    offset: None,
                }
            }

            // A block fixes its own widths. The K family and mxfp4 are
            // served (`Dtype` reprs); the scalar-group four are not.
            QuantScheme::GgufQ4_0 => G32_I4_F16_N,
            QuantScheme::GgufQ4_1 => G32_U4_F16_B_F16,
            QuantScheme::GgufQ5_0 => G32_I5_F16_N,
            QuantScheme::GgufQ5_1 => G32_U5_F16_B_F16,
            QuantScheme::GgufQ8_0 => G32_I8_F16_N,
            QuantScheme::GgufQ2K => *Dtype::U2g16k.repr(),
            QuantScheme::GgufQ3K => *Dtype::I3g16k.repr(),
            QuantScheme::GgufQ4K => *Dtype::U4g32k.repr(),
            QuantScheme::GgufQ5K => *Dtype::U5g32k.repr(),
            QuantScheme::GgufQ6K => *Dtype::I6g16k.repr(),

            // ggml's own MXFP4 block: OCP numbers with the scale byte interleaved.
            QuantScheme::GgufMxfp4 => *Dtype::Mxfp4.repr(),

            // IQ4_NL/IQ4_XS index llama.cpp's codebook.
            QuantScheme::GgufIq4Nl | QuantScheme::GgufIq4Xs => return None,

            // These schemes quantize a direction (a compiled-in lattice
            // point); QNF has no node for that.
            QuantScheme::GgufIq2Xxs
            | QuantScheme::GgufIq2Xs
            | QuantScheme::GgufIq2S
            | QuantScheme::GgufIq3Xxs
            | QuantScheme::GgufIq3S => return None,
        })
    }

    /// Whether this spec means OCP MXFP4 in its two-plane (not interleaved) form.
    #[must_use]
    pub fn is_mxfp4(&self) -> bool {
        !self.scheme.is_self_contained()
            && matches!(self.term(), Some(t) if t == *Dtype::Mxfp4.repr())
    }

    /// `(group, bits)` for integer codes in `N`-groups stored leaf-per-plane;
    /// `None` for raw, mxfp4, a per-row scale, or a self-contained GGUF block.
    #[must_use]
    pub fn affine_point(&self) -> Option<(u32, u32)> {
        if self.scheme.is_self_contained() {
            return None;
        }
        match self.term()? {
            Fmt::Q {
                g: Group::N(n),
                elem: Elem::U(b) | Elem::I(b),
                ..
            } => Some((n, u32::from(b))),
            _ => None,
        }
    }
}

/// Every scheme that keeps its scales inside its payload. Written out since
/// `QuantScheme` has no iterator; a row missing here is a refusal, not a wrong answer.
const BLOCKS: &[QuantScheme] = &[
    QuantScheme::GgufQ4_0,
    QuantScheme::GgufQ4_1,
    QuantScheme::GgufQ5_0,
    QuantScheme::GgufQ5_1,
    QuantScheme::GgufQ8_0,
    QuantScheme::GgufQ2K,
    QuantScheme::GgufQ3K,
    QuantScheme::GgufQ4K,
    QuantScheme::GgufQ5K,
    QuantScheme::GgufQ6K,
    QuantScheme::GgufMxfp4,
    QuantScheme::GgufIq4Nl,
    QuantScheme::GgufIq4Xs,
    QuantScheme::GgufIq2Xxs,
    QuantScheme::GgufIq2Xs,
    QuantScheme::GgufIq2S,
    QuantScheme::GgufIq3Xxs,
    QuantScheme::GgufIq3S,
];

/// `QuantSpec::term` read backwards, over the block family only — parametric
/// families are absent, since a term alone can't say which `(bits, group)`
/// produced it. `None` is a refusal, never a default.
#[must_use]
pub fn spec_of_term(f: &Fmt<'_>) -> Option<QuantSpec> {
    BLOCKS.iter().copied().find_map(|scheme| {
        let (elems, _) = scheme.block_layout()?;
        let spec = QuantSpec {
            scheme,
            logical_dtype: Dtype::Bf16,
            bits_per_element: scheme.default_bits(),
            group_size: u32::try_from(elems).ok()?,
            channel_axis: None,
        };
        (spec.term() == Some(*f)).then_some(spec)
    })
}
