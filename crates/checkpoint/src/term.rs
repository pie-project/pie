//! What a [`QuantSpec`] MEANS: the algebra behind this crate's quant names.
//!
//! [`QuantScheme`] is a NAME. `dtype`'s [`Fmt`] is the arithmetic the
//! name stands for — a group width, a code element, a gain, and an offset that
//! says whether it subtracts in the code domain or adds in the value domain.
//! [`QuantSpec::term`] is the whole of that mapping, total over the scheme
//! enum, and it lives on the spec because the spec holds both halves of the
//! answer: the scheme, and the numbers the parametric families take their
//! group and width from. (A plain [`DType`](crate::types::DType)'s term needs
//! no mapping at all — it is `d.repr()`, the dtype's own table.) The
//! structural questions the plan answers an engine —
//! [`QuantSpec::is_mxfp4`], [`QuantSpec::affine_point`] — are derived from
//! the term here too, so a name comparison never stands in for the algebra.
//!
//! **`None` IS A REFUSAL AND NEVER A DEFAULT.** It means this row has no QNF
//! term yet — an IQ lattice whose points are compiled into llama.cpp rather
//! than stored, an `IQ4` table that wants an [`Elem::Cb`] registry that does
//! not exist, a scheme whose arithmetic nothing in this tree fixes. A guessed
//! row would be the silent-wrong-number bug the quant module was built to
//! end, and it would be worse here than there: this is the door a loader
//! walks through to decide what a kernel may be handed.
//!
//! **THE TERM IS NOT YET A LANDING.** What this function answers is what the
//! bytes MEAN; whether the engine can serve that meaning is
//! [`Dtype::of_fmt`]'s question, one call later. A scheme whose term maps to
//! a [`Dtype`] variant lands as stored; one whose term does not — GPTQ's row,
//! `q4_1`, the per-row e5m2 — is an import fact and must repack toward a
//! variant or refuse.
//!
//! # Where the rows come from
//!
//! Not from a vendor's README. Every row below is read off something this
//! crate already states about the bytes:
//!
//! * `types.rs` — the scheme docs, [`QuantScheme::block_layout`]'s
//!   `(elements, bytes)` table, and the default widths.
//! * `executor/walk.rs` — the host block decoders, which are the arithmetic
//!   written out: `q4_0`'s `(nibble − 8) × scale`, `q4_k`'s
//!   `d × scaleᵢ × nibble − dmin × minᵢ`, `q3_k`'s inverted mask bit.
//! * `file/write.rs` — `zt.quant_group/1`'s parameters, which is the tree's
//!   parametric statement of an affine scheme: packing order, scale form and
//!   zero-point form. That AWQ and GPTQ differ only in `order` is why they
//!   converge here.
//! * `codec/mlx.rs` and `codec/int4.rs` — the two element formats whose
//!   decoders live beside the schemes that name them.
//!
//! # What a term deliberately does not say
//!
//! A term is the ALGEBRA. Everything that is a fact about the CONTAINER
//! stays where it already lives, and two schemes that share a row are still
//! two schemes to a reader addressing bytes:
//!
//! * packing order — AWQ packs its nibbles low-first and GPTQ high-first, and
//!   `file/zt.rs` recovers the scheme from that. One row, two byte layouts.
//! * planar against interleaved —
//!   [`GgufMxfp4`](QuantScheme::GgufMxfp4) interleaves its `e8m0` byte into
//!   each 17-byte block where [`Mxfp4E2M1E8M0`](QuantScheme::Mxfp4E2M1E8M0)
//!   ships the OCP two-plane form. Same numbers, spans that differ by 17/16.
//! * the bits a value is spread across — `q5_0`'s fifth bit lives in a 32-bit
//!   plane and `q3_k`'s third in a mask that reads inverted. The row says the
//!   value set is five bits and three bits wide; where those bits sit is the
//!   block's business.
//!
//! # The three offset rulings, applied
//!
//! `dtype`'s canonical rules decide every offset slot below, and they
//! are worth restating in this crate's own vocabulary because all three have a
//! scheme here that would otherwise be spelled two ways:
//!
//! * an INTEGER ZERO POINT of the codes' own width — `zero_point.form`
//!   `tensor`, packed `same_as_data` — is [`Off::Pre`], because it is
//!   subtracted before the gain applies: AWQ and GPTQ.
//! * a REAL BIAS OR MINIMUM is [`Off::Post`], sign-normalized to an
//!   addition. MLX stores `code × scale + bias` and is already an addition;
//!   the K-quant mins and `q2_k`'s are SUBTRACTED, and reach the same slot by
//!   negating their own f16 factor at repack, which is bit-exact.
//! * a SYMMETRIC RECENTERED CODE has no offset at all — the zero point folds
//!   into [`Elem::I`], which is excess-binary. `q4_0`'s nibble 8 is zero, and
//!   [`Int4B8`](QuantScheme::Int4B8)'s doc says the same thing in this crate's
//!   words: *the zero point is the 8*.
//!
//! A two's-complement plane — `q8_0`'s bytes, `q6_k`'s sub-block scales — is
//! the same information with its sign bit relabeled, which a repack does
//! bit-exactly, so it spells `i8` here rather than growing a second row.
//!
//! # Two places the tree says two things
//!
//! Both are about a FACTOR's element, both are recorded here rather than
//! quietly resolved, and both would be a one-token change if a ruling lands:
//!
//! * MLX's factors are bf16 — [`Dtype::U4g64`]'s repr says so and
//!   `mlx_lm.convert` writes so — while `file/write.rs` labels the same
//!   planes `f16_factors` in the zTensor profile it writes. That attribute
//!   has two values for three families, so it is reporting half-width rather
//!   than the exact element; the row below follows the three that agree.
//! * `Int4B8`'s factors have no attested element at all. The same profile
//!   writes `f32_factors` for it where its four-bit neighbours get
//!   `f16_factors`, and the one end-to-end test that decodes such a plane —
//!   `an_int4b8_source_is_dequantized_by_its_factors`, in `executor/walk.rs`
//!   — pairs the codes with a BF16 tensor. The row below spells f16, with its
//!   neighbours; nothing in the tree contradicts that and nothing confirms it.

use dtype::Dtype;
use dtype::{Elem, Fmt, Group, Off, spells};

use crate::types::{QuantScheme, QuantSpec};

// ─────────────────────────────────────────────────────────────────────────
// The rows this crate spells that no kernel serves
// ─────────────────────────────────────────────────────────────────────────
//
// Named by their own spelling uppercased, which is the repr algebra's rule for
// a registered row, and tied to that spelling by a `const` assertion so a
// drift in either half is a BUILD error. Rows a kernel DOES serve are not
// spelled here: they are `Dtype` variants, and their reprs are the constants.

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
/// f16 offset that ADDS — `nibble × d + m`, the affine sibling of `q4_0`.
const G32_U4_F16_B_F16: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::U(4),
    gain: &F16,
    offset: Some(Off::Post(&F16)),
};
const _: () = assert!(spells(&G32_U4_F16_B_F16, "g32_u4_f16_b_f16"));

/// ggml's `q5_0`: `q4_0` with a fifth bit per element, so the value set is
/// excess-16 over five bits — `(nibble | bit⁴) − 16`. The 32-bit plane the
/// fifth bits live in is a container fact and not a term.
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

/// The affine row whose gain and offset are both a bf16 leaf and whose
/// offset ADDS: `code × scale + bias`. MLX's shape at whichever width and
/// group the plane declares. At `(4, 64)`, `(8, 64)` and `(4, 32)` the
/// result IS a [`Dtype`] repr, and [`Dtype::of_fmt`] recognizes it.
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
/// four-bit width, subtracted before the gain applies: `scale × (code −
/// zero)`. What `zt.quant_group/1` writes as `zero_point.packing:
/// same_as_data`.
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
    /// The QNF term this encoding stands for, or `None` for a scheme that
    /// has none yet.
    ///
    /// The scheme is authoritative and the spec's numbers fill in the
    /// parametric families' group and width. They are read through
    /// [`QuantSpec::normalized_bits`] and
    /// [`QuantSpec::normalized_group_size`], so a zero field means "the
    /// scheme's default" here exactly as it does everywhere else in this
    /// crate.
    ///
    /// **THE SPEC'S NUMBERS ARE INERT FOR THE GGUF FAMILY.** A scheme with a
    /// [`block_layout`](QuantScheme::block_layout) carries its scales inside
    /// its payload, so its group width and code width are the block's and not
    /// the declaration's — `default_group_size`'s own doc calls those fields
    /// inert for exactly these schemes — and every GGUF row below is a
    /// constant.
    ///
    /// They are inert for a second family for a second reason: the
    /// per-channel schemes ([`Fp8E4M3`](QuantScheme::Fp8E4M3),
    /// [`Int8Symmetric`](QuantScheme::Int8Symmetric)) answer `1` from
    /// `default_group_size`, which is a placeholder and not a group of one
    /// element. What they store is one f32 factor per output row — the host
    /// encoder in `executor/walk.rs` writes `rows × 4` bytes and refuses any
    /// other scale shape — so they spell [`Group::Row`].
    #[must_use]
    pub fn term(&self) -> Option<Fmt<'static>> {
        let bits = self.normalized_bits();
        let group = self.normalized_group_size();
        Some(match self.scheme {
            // A raw tensor's term is its dtype's own `repr`: without a dtype
            // there is no element, and an element is the whole of a plain term.
            QuantScheme::None => return None,

            // ── the per-channel families ───────────────────────────────────
            //
            // Plain f8 or i8 elements whose factors are a separate tensor — what
            // `file/write.rs` writes as `dense` — one f32 per output row.
            //
            // A BLOCK-SCALED FP8 CHECKPOINT IS A DIFFERENT ROW AND THIS SPEC
            // CANNOT NAME IT. DeepSeek- and GLM-style files carry one scale per
            // `[B, B]` tile (`is_block_scaled` is that fact), and `B` belongs to
            // the consuming target rather than to the declaration —
            // `plan::StorageTarget::block_scale_rows` is where it is settled. At
            // the tile the tree targets, that row is [`Dtype::E4m3tile128`]; the
            // scheme alone says the per-row form the encoder writes.
            QuantScheme::Fp8E4M3 => *Dtype::E4m3row.repr(),
            QuantScheme::Fp8E5M2 => GR_E5M2_F32_N,
            QuantScheme::Int8Symmetric => GR_I8_F32_N,

            // No arithmetic in this tree fixes this one. Its zTensor profile
            // records a zero-point TENSOR and f32 factors, and nothing else: there
            // is no decoder, no encoder and no kernel that says whether the zero
            // subtracts in the code domain (`gr_i8_f32_z_i8`, the ONNX and PyTorch
            // convention) or the offset adds in the value domain
            // (`gr_i8_f32_b_f32`). Pre and Post are not the same arithmetic and no
            // normalization joins them, so this stays a refusal until something
            // decodes one.
            QuantScheme::Int8Asymmetric => return None,

            // ── the affine-group families ──────────────────────────────────
            //
            // ONE ROW FOR TWO PIPELINES, WHICH IS THE POINT. AWQ and GPTQ differ
            // in `packing.order` — `lsb_first` against `msb_first` — and in
            // nothing else: same bits, same f16 factors, same zero point packed at
            // the codes' own width. The packing order is a container facet that
            // `file/zt.rs` recovers the scheme from; the arithmetic is one row,
            // and a kernel table keyed on the row serves both.
            QuantScheme::AwqInt4 | QuantScheme::GptqInt4 => {
                if bits != 4 {
                    // `zt.quant_group/1` names these two at four bits and only
                    // there, so a wider declaration is a scheme this crate cannot
                    // read back rather than a row to invent.
                    return None;
                }
                pre_affine_u4(group)
            }

            // MLX's affine codes at whichever width the plane declares — four, or
            // eight for the MoE router gates a `quant_predicate` lifts. Both
            // widths are this one scheme, which is [`Dtype::U8g64`]'s whole
            // argument, and `codec/mlx.rs` decodes exactly those two.
            //
            // The offset ADDS: `mlx_affine_group_params` returns a scale that is
            // usually NEGATIVE and a bias that is the group's dominant endpoint,
            // and an element is `code × scale + bias`. It is a bias in the value
            // domain and not a zero point in the code domain, whatever the
            // companion plane is called.
            QuantScheme::MlxAffineU4 => match bits {
                4 | 8 => post_affine_bf16(group, Elem::U(bits)),
                _ => return None,
            },

            // The zero point IS the 8 — this scheme's own doc line — so the codes
            // are excess-binary and the term carries no offset at all.
            //
            // THE GAIN IS THE ONE LEAF THIS SCHEME DOES NOT FIX, and the module
            // doc's second disagreement is about which it is: the scales are a
            // separate tensor a contract pairs in with `Expr::Scale`, so the
            // element is whatever that tensor declares. f16 is what the family it
            // sits in stores, and it is the row here.
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

            // OCP Microscaling in its two-plane form. The block is 32 by the
            // specification and `zt.mx/1` echoes the declaration's `group_size`
            // into `block_size`, so the group is read from the spec; at the
            // specified 32 the term IS [`Dtype::Mxfp4`]'s repr, and
            // [`Dtype::of_fmt`] says so.
            QuantScheme::Mxfp4E2M1E8M0 => {
                const E8M0: Fmt<'static> = Fmt::Elem(Elem::E { e: 8, m: 0 });
                Fmt::Q {
                    g: Group::N(group),
                    elem: Elem::E { e: 2, m: 1 },
                    gain: &E8M0,
                    offset: None,
                }
            }

            // ── the GGUF blocks ────────────────────────────────────────────
            //
            // Constants, because a block fixes its own widths. Each row is
            // checked against `block_layout` by `tests/qnf_bridge.rs`: bits per
            // weight times the block's element count is the block's byte count
            // exactly, for all eleven. The K family and `mxfp4` are SERVED —
            // their terms are `Dtype` reprs — and the scalar-group four are not.
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

            // ggml's own MXFP4 block: the OCP numbers with the scale byte
            // interleaved into each block. Same term, different container, and
            // `is_self_contained` is where the difference is already written down.
            QuantScheme::GgufMxfp4 => *Dtype::Mxfp4.repr(),

            // ── the rows with no term ──────────────────────────────────────
            //
            // A codebook leaf, reserved. `IQ4_NL` and `IQ4_XS` index llama.cpp's
            // sixteen-entry `kvalues_iq4nl`, which is a table and not a value set:
            // `Elem::Cb` is where such a row will land, and the registry that
            // says which table is which does not exist. `Elem::Nf4` is not it —
            // that leaf is bitsandbytes' normal-float table, a different sixteen
            // values, and reading one as the other would decode every element to
            // a plausible wrong magnitude.
            QuantScheme::GgufIq4Nl | QuantScheme::GgufIq4Xs => return None,

            // A lattice, and the lattice is not in the file. These schemes
            // quantize a DIRECTION: a code addresses a table of eight- or
            // four-component points compiled into llama.cpp, so no group width and
            // no code element describes what a byte holds. QNF has no node for a
            // point, and inventing one from the scale layout would name the least
            // interesting half of the format.
            QuantScheme::GgufIq2Xxs
            | QuantScheme::GgufIq2Xs
            | QuantScheme::GgufIq2S
            | QuantScheme::GgufIq3Xxs
            | QuantScheme::GgufIq3S => return None,
        })
    }

    /// Whether this spec means OCP MXFP4 in its two-plane form — the term is
    /// [`Dtype::Mxfp4`]'s repr and the container is leaf-per-plane.
    ///
    /// The container half matters: [`GgufMxfp4`](QuantScheme::GgufMxfp4) is
    /// the same algebra with the scale byte interleaved into each 17-byte
    /// block, and a binder asking this question is asking about the two-plane
    /// bytes an mxfp4 kernel reads — `is_self_contained` is where that
    /// difference is already written down, so it is asked, not restated.
    #[must_use]
    pub fn is_mxfp4(&self) -> bool {
        !self.scheme.is_self_contained()
            && matches!(self.term(), Some(t) if t == *Dtype::Mxfp4.repr())
    }

    /// The affine point this spec reads at — `(group, bits)` when the term is
    /// integer codes in `N`-groups stored leaf-per-plane, which is the shape
    /// the affine kernels stamp — and `None` for everything else: raw, mxfp4
    /// (its own kernel at its own group), a per-row scale (no finite group),
    /// and the self-contained blocks (a GGUF block reads by its own decoder,
    /// never at an affine point, whatever numbers its term carries).
    ///
    /// STRUCTURAL, NOT NOMINAL: the answer is read off the term, so a new
    /// scheme whose term has this shape reports its point without joining a
    /// list here.
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
