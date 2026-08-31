//! The canon bridge: this crate's quant vocabulary, said in QNF.
//!
//! [`QuantScheme`] and [`DType`] are NAMES. `dtype::quant`'s [`Sig`] is the
//! arithmetic those names stand for — a group width, a code leaf, a gain, and
//! an offset that says whether it subtracts in the code domain or adds in the
//! value domain. Two functions here, both total over their enum and both
//! answering `Option<Sig>`, are the whole of the mapping:
//! [`sig_of_scheme`] for a quantized encoding and [`sig_of_dtype`] for a plain
//! one.
//!
//! **`None` IS A REFUSAL AND NEVER A DEFAULT.** It means this row has no QNF
//! signature yet — an IQ lattice whose points are compiled into llama.cpp
//! rather than stored, an `IQ4` table that wants a [`Leaf::Cb`] registry that
//! does not exist, a scheme whose arithmetic nothing in this tree fixes. A
//! guessed row would be the silent-wrong-number bug the quant module was built
//! to end, and it would be worse here than there: this is the door a loader
//! walks through to decide what a kernel may be handed.
//!
//! # Where the rows come from
//!
//! Not from a vendor's README. Every row below is read off something this
//! crate already states about the bytes:
//!
//! * `types.rs` — the scheme docs, [`QuantScheme::block_layout`]'s
//!   `(elements, bytes)` table, and the default widths.
//! * `executor/walk.rs` — the host block decoders, which are the arithmetic
//!   written out: `Q4_0`'s `(nibble − 8) × scale`, `Q4_K`'s
//!   `d × scaleᵢ × nibble − dmin × minᵢ`, `Q3_K`'s inverted mask bit.
//! * `file/write.rs` — `zt.quant_group/1`'s parameters, which is the tree's
//!   parametric statement of an affine scheme: packing order, scale form and
//!   zero-point form. That AWQ and GPTQ differ only in `order` is why they
//!   converge here.
//! * `codec/mlx.rs` and `codec/int4.rs` — the two element formats whose
//!   decoders live beside the schemes that name them.
//!
//! # What a `Sig` deliberately does not say
//!
//! A signature is the ALGEBRA. Everything that is a fact about the CONTAINER
//! stays where it already lives, and two schemes that share a row are still
//! two schemes to a reader addressing bytes:
//!
//! * packing order — AWQ packs its nibbles low-first and GPTQ high-first, and
//!   `file/zt.rs` recovers the scheme from that. One row, two byte layouts.
//! * planar against interleaved —
//!   [`GgufMxfp4`](QuantScheme::GgufMxfp4) interleaves its `e8m0` byte into
//!   each 17-byte block where [`Mxfp4E2M1E8M0`](QuantScheme::Mxfp4E2M1E8M0)
//!   ships the OCP two-plane form. Same numbers, spans that differ by 17/16.
//! * the bits a value is spread across — `Q5_0`'s fifth bit lives in a 32-bit
//!   plane and `Q3_K`'s third in a mask that reads inverted. The row says the
//!   value set is five bits and three bits wide; where those bits sit is the
//!   block's business.
//!
//! # The three offset rulings, applied
//!
//! `dtype::quant`'s canonical rules decide every offset slot below, and they
//! are worth restating in this crate's own vocabulary because all three have a
//! scheme here that would otherwise be spelled two ways:
//!
//! * an INTEGER ZERO POINT of the codes' own width — `zero_point.form`
//!   `tensor`, packed `same_as_data` — is [`OffSub::Pre`], because it is
//!   subtracted before the gain applies: AWQ and GPTQ.
//! * a REAL BIAS OR MINIMUM is [`OffSub::Post`], sign-normalized to an
//!   addition. MLX stores `code × scale + bias` and is already an addition;
//!   the K-quant mins and `Q2_K`'s are SUBTRACTED, and reach the same slot by
//!   negating their own f16 factor at repack, which is bit-exact.
//! * a SYMMETRIC RECENTERED CODE has no offset at all — the zero point folds
//!   into [`Leaf::I`], which is excess-binary. `Q4_0`'s nibble 8 is zero, and
//!   [`Int4B8`](QuantScheme::Int4B8)'s doc says the same thing in this crate's
//!   words: *the zero point is the 8*.
//!
//! A two's-complement plane — `Q8_0`'s bytes, `Q6_K`'s sub-block scales — is
//! the same information with its sign bit relabeled, which a repack does
//! bit-exactly, so it spells `i8` here rather than growing a second row.
//!
//! # Two places the tree says two things
//!
//! Both are about a FACTOR's element, both are recorded here rather than
//! quietly resolved, and both would be a one-token change if a ruling lands:
//!
//! * MLX's factors are bf16 — `DType::MlxU4`'s docs say so, `mlx_lm.convert`
//!   writes so, and `dtype::quant`'s registered row is `g64_u4_bf16_b_bf16` —
//!   while `file/write.rs` labels the same planes `f16_factors` in the
//!   zTensor profile it writes. That attribute has two values for three
//!   families, so it is reporting half-width rather than the exact element;
//!   the row below follows the three that agree.
//! * `Int4B8`'s factors have no attested element at all. The same profile
//!   writes `f32_factors` for it where its four-bit neighbours get
//!   `f16_factors`, and the one end-to-end test that decodes such a plane —
//!   `an_int4b8_source_is_dequantized_by_its_factors`, in `executor/walk.rs`
//!   — pairs the codes with a BF16 tensor. The row below spells f16, with its
//!   neighbours; nothing in the tree contradicts that and nothing confirms it.

use dtype::quant::{
    G32_U4_BF16_B_BF16, G64_U4_BF16_B_BF16, G64_U8_BF16_B_BF16, GR_E4M3_F32_N, Group, Leaf,
    MXFP4, OffSub, Q4_0, Q4_K, Q6_K, Q8_0, Sig, Sub, sig,
};

use crate::types::{DType, QuantScheme, QuantSpec};

// ─────────────────────────────────────────────────────────────────────────
// The rows this crate spells that the quant registry does not
// ─────────────────────────────────────────────────────────────────────────
//
// Named by their own spelling uppercased, which is `dtype::quant`'s rule for a
// registered row and the reason a typo below is a BUILD error: `sig` is a
// `const fn` and these are `const`.

/// ggml's `Q4_1`: unsigned nibbles in blocks of 32 under an f16 scale and an
/// f16 offset that ADDS — `nibble × d + m`, the affine sibling of `Q4_0`.
const G32_U4_F16_B_F16: Sig = sig("g32_u4_f16_b_f16");

/// ggml's `Q5_0`: `Q4_0` with a fifth bit per element, so the value set is
/// excess-16 over five bits — `(nibble | bit⁴) − 16`. The 32-bit plane the
/// fifth bits live in is a container fact and not a term.
const G32_I5_F16_N: Sig = sig("g32_i5_f16_n");

/// ggml's `Q5_1`: [`G32_U4_F16_B_F16`] at five bits, the fifth carried in the
/// same plane [`G32_I5_F16_N`] uses.
const G32_U5_F16_B_F16: Sig = sig("g32_u5_f16_b_f16");

/// ggml's `Q2_K`: 2-bit codes in sixteen sub-blocks of sixteen, whose 4-bit
/// scales and 4-bit mins are grouped sixteen to a super-block under one f16
/// each. `Q4_K`'s tree at half the code width and half the sub-block.
const G16_U2_G16_U4_F16_N_B_G16_U4_F16_N: Sig = sig("g16_u2_g16_u4_f16_n_b_g16_u4_f16_n");

/// ggml's `Q3_K`: excess-4 3-bit codes in sixteen sub-blocks of sixteen, whose
/// 6-bit scales — stored biased by 32, hence excess-binary — are grouped
/// sixteen to a super-block under one f16. Symmetric, so no offset anywhere.
const G16_I3_G16_I6_F16_N_N: Sig = sig("g16_i3_g16_i6_f16_n_n");

/// ggml's `Q5_K`: [`Q4_K`]'s factor tree over five-bit codes.
const G32_U5_G8_U6_F16_N_B_G8_U6_F16_N: Sig = sig("g32_u5_g8_u6_f16_n_b_g8_u6_f16_n");

/// FP8-E5M2 weights with one f32 scale per output row — [`dtype::quant`]'s
/// `GR_E4M3_F32_N` at the other FP8 width.
const GR_E5M2_F32_N: Sig = sig("gr_e5m2_f32_n");

/// Symmetric 8-bit codes with one f32 scale per output row.
const GR_I8_F32_N: Sig = sig("gr_i8_f32_n");

// ─────────────────────────────────────────────────────────────────────────
// Shapes the parametric families take
// ─────────────────────────────────────────────────────────────────────────

/// The affine row whose gain and offset are both stored in `factor` and whose
/// offset ADDS: `code × scale + bias`. MLX's shape, and `Q4_1`'s.
fn post_affine(group: u32, elem: Leaf, factor: Leaf) -> Sig {
    Sig::Q {
        g: Group::N(group),
        elem,
        gain: Sub::L(factor),
        offset: OffSub::Post(Sub::L(factor)),
    }
}

/// The affine row whose zero point is an integer code of the weights' own
/// width, subtracted before the gain applies: `scale × (code − zero)`. What
/// `zt.quant_group/1` writes as `zero_point.packing: same_as_data`.
fn pre_affine(group: u32, bits: u8, gain: Leaf) -> Sig {
    Sig::Q {
        g: Group::N(group),
        elem: Leaf::U(bits),
        gain: Sub::L(gain),
        offset: OffSub::Pre(Sub::L(Leaf::U(bits))),
    }
}

/// A symmetric row: codes under one leaf gain and no offset at all.
fn symmetric(group: Group, elem: Leaf, gain: Leaf) -> Sig {
    Sig::Q {
        g: group,
        elem,
        gain: Sub::L(gain),
        offset: OffSub::Nil,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The two doors
// ─────────────────────────────────────────────────────────────────────────

/// The QNF signature a quantized encoding stands for, or `None` for a scheme
/// that has none yet.
///
/// `scheme` is authoritative and `spec` is consulted for its numbers, which is
/// why both are arguments: a caller holding a [`QuantSpec`] passes
/// `spec.scheme` and gets the same answer, and a caller holding only a scheme
/// can ask what a default-shaped spec would say. The numbers are read through
/// [`QuantSpec::normalized_bits`] and
/// [`QuantSpec::normalized_group_size`], so a zero field means "the scheme's
/// default" here exactly as it does everywhere else in this crate.
///
/// **THE SPEC'S NUMBERS ARE INERT FOR THE GGUF FAMILY.** A scheme with a
/// [`block_layout`](QuantScheme::block_layout) carries its scales inside its
/// payload, so its group width and code width are the block's and not the
/// declaration's — `default_group_size`'s own doc calls those fields inert for
/// exactly these schemes — and every GGUF row below is a constant.
///
/// They are inert for a second family for a second reason: the per-channel
/// schemes ([`Fp8E4M3`](QuantScheme::Fp8E4M3),
/// [`Int8Symmetric`](QuantScheme::Int8Symmetric)) answer `1` from
/// `default_group_size`, which is a placeholder and not a group of one
/// element. What they store is one f32 factor per output row — the host
/// encoder in `executor/walk.rs` writes `rows × 4` bytes and refuses any other
/// scale shape — so they spell [`Group::Row`].
#[must_use]
pub fn sig_of_scheme(scheme: QuantScheme, spec: &QuantSpec) -> Option<Sig> {
    let bits = spec.normalized_bits();
    let group = spec.normalized_group_size();
    Some(match scheme {
        // A raw tensor has a signature, and it is `sig_of_dtype`'s to give:
        // without a dtype there is no leaf, and a leaf is the whole of a
        // `Plain` row.
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
        // the tile the tree targets that row is `dtype::quant`'s
        // `G128X128_E4M3_F32_N`; the scheme alone says the per-row form the
        // encoder writes.
        QuantScheme::Fp8E4M3 => GR_E4M3_F32_N,
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
            pre_affine(group, bits, Leaf::F16)
        }

        // MLX's affine codes at whichever width the plane declares — four, or
        // eight for the MoE router gates a `quant_predicate` lifts. Both
        // widths are this one scheme, which is `DType::MlxU8`'s whole
        // argument, and `codec/mlx.rs` decodes exactly those two.
        //
        // The offset ADDS: `mlx_affine_group_params` returns a scale that is
        // usually NEGATIVE and a bias that is the group's dominant endpoint,
        // and an element is `code × scale + bias`. It is a bias in the value
        // domain and not a zero point in the code domain, whatever the
        // companion plane is called.
        QuantScheme::MlxAffineU4 => match bits {
            4 | 8 => post_affine(group, Leaf::U(bits), Leaf::Bf16),
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
            symmetric(Group::N(group), Leaf::I(4), Leaf::F16)
        }

        // OCP Microscaling in its two-plane form. The block is 32 by the
        // specification and `zt.mx/1` echoes the declaration's `group_size`
        // into `block_size`, so the group is read from the spec and lands on
        // `dtype::quant`'s `MXFP4` for every checkpoint that follows OCP.
        QuantScheme::Mxfp4E2M1E8M0 => {
            symmetric(Group::N(group), Leaf::E { e: 2, m: 1 }, Leaf::E { e: 8, m: 0 })
        }

        // ── the GGUF blocks ────────────────────────────────────────────
        //
        // Constants, because a block fixes its own widths. Each row is
        // checked against `block_layout` by `tests/qnf_bridge.rs`: bits per
        // weight times the block's element count is the block's byte count
        // exactly, for all eleven.
        QuantScheme::GgufQ4_0 => Q4_0,
        QuantScheme::GgufQ4_1 => G32_U4_F16_B_F16,
        QuantScheme::GgufQ5_0 => G32_I5_F16_N,
        QuantScheme::GgufQ5_1 => G32_U5_F16_B_F16,
        QuantScheme::GgufQ8_0 => Q8_0,
        QuantScheme::GgufQ2K => G16_U2_G16_U4_F16_N_B_G16_U4_F16_N,
        QuantScheme::GgufQ3K => G16_I3_G16_I6_F16_N_N,
        QuantScheme::GgufQ4K => Q4_K,
        QuantScheme::GgufQ5K => G32_U5_G8_U6_F16_N_B_G8_U6_F16_N,
        QuantScheme::GgufQ6K => Q6_K,

        // ggml's own MXFP4 block: the OCP numbers with the scale byte
        // interleaved into each block. Same term, different container, and
        // `is_self_contained` is where the difference is already written down.
        QuantScheme::GgufMxfp4 => MXFP4,

        // ── the rows with no signature ─────────────────────────────────
        //
        // A codebook leaf, reserved. `IQ4_NL` and `IQ4_XS` index llama.cpp's
        // sixteen-entry `kvalues_iq4nl`, which is a table and not a value set:
        // `Leaf::Cb` is where such a row will land, and the registry that
        // says which table is which does not exist. `Leaf::Nf4` is not it —
        // that leaf is bitsandbytes' normal-float table, a different sixteen
        // values, and reading one as the other would decode every element to
        // a plausible wrong magnitude.
        QuantScheme::GgufIq4Nl | QuantScheme::GgufIq4Xs => return None,

        // A lattice, and the lattice is not in the file. These schemes
        // quantize a DIRECTION: a code addresses a table of eight- or
        // four-component points compiled into llama.cpp, so no group width and
        // no code leaf describes what a byte holds. QNF has no node for a
        // point, and inventing one from the scale layout would name the least
        // interesting half of the format.
        QuantScheme::GgufIq2Xxs
        | QuantScheme::GgufIq2Xs
        | QuantScheme::GgufIq2S
        | QuantScheme::GgufIq3Xxs
        | QuantScheme::GgufIq3S => return None,
    })
}

/// The QNF signature a [`DType`] stands for.
///
/// Total, and `Some` for every row: every variant of that enum is something a
/// plane stores, so every one has a term. Most are a [`Sig::Plain`] leaf — the
/// element IS the format — and the four that are not are the packed weight
/// codes, which name a whole scheme in one word and expand to it here.
///
/// **A KV SCHEME'S GRANULARITY IS THE CACHE ROW'S FACT, NOT THE ELEMENT'S.**
/// `Fp8E4m3`, `I8` and `Fp4` are also the names a kv page's quant scheme goes
/// by, and this function answers for the ELEMENT: a `Plain` leaf of what one
/// code is. Whether that scheme scales per tensor or per token-head, and how
/// wide an fp4 block runs, are sibling fields of the row that chose the
/// scheme, exactly as `dtype`'s own docs say. A row that means "fp8 codes
/// under a per-row f32" is a `Sig` that says so, and it is
/// [`sig_of_scheme`]'s to give.
///
/// **THE SIGNED WIDTHS SPELL `i{n}`, WHICH IS EXCESS-BINARY.** QNF has one
/// signed leaf and it is the recentered one, because a two's-complement source
/// is the same information with its sign bit relabeled and a repack does that
/// bit-exactly. So `I8` answers `i8`: the width and the signedness are what
/// the answer names, and the encoding of the sign is the container's. Nothing
/// repacks an index plane, which is what the wide ints are.
#[must_use]
pub fn sig_of_dtype(d: DType) -> Option<Sig> {
    Some(match d {
        // ── floats ────────────────────────────────────────────────────
        DType::F32 => Sig::Plain(Leaf::F32),
        DType::F16 => Sig::Plain(Leaf::F16),
        DType::Bf16 => Sig::Plain(Leaf::Bf16),
        DType::Fp8E4m3 => Sig::Plain(Leaf::E { e: 4, m: 3 }),
        DType::Fp8E5m2 => Sig::Plain(Leaf::E { e: 5, m: 2 }),
        // Bare 4-bit float codes, scaled by whatever chose them — so a leaf,
        // and OCP's `e2m1`, which is the element `storage_of` writes as
        // `f4_e2m1`.
        DType::Fp4 => Sig::Plain(Leaf::E { e: 2, m: 1 }),
        // Exponent-only, hence unsigned and eight bits rather than nine. It is
        // only ever the companion beside a block-scaled tensor, and a `Plain`
        // row is what that companion plane holds.
        DType::E8m0 => Sig::Plain(Leaf::E { e: 8, m: 0 }),

        // ── the packed weight codes ───────────────────────────────────
        //
        // Each of these names a SCHEME in one word — that is what the variant
        // is for — so each expands to the whole term rather than to a leaf.
        // The three MLX rows differ in two integers, which is the observation
        // `dtype::quant` opens with; here they are those two integers.
        DType::Mxfp4 => MXFP4,
        DType::MlxU4 => G64_U4_BF16_B_BF16,
        DType::MlxU8 => G64_U8_BF16_B_BF16,
        DType::MlxU4G32 => G32_U4_BF16_B_BF16,

        // ── ints ──────────────────────────────────────────────────────
        DType::I64 => Sig::Plain(Leaf::I(64)),
        DType::I32 => Sig::Plain(Leaf::I(32)),
        DType::I16 => Sig::Plain(Leaf::I(16)),
        DType::I8 => Sig::Plain(Leaf::I(8)),
        DType::U64 => Sig::Plain(Leaf::U(64)),
        DType::U32 => Sig::Plain(Leaf::U(32)),
        DType::U16 => Sig::Plain(Leaf::U(16)),
        DType::U8 => Sig::Plain(Leaf::U(8)),

        // ── logical ───────────────────────────────────────────────────
        //
        // One byte per element, as every checkpoint format stores it, so `u8`
        // is what the plane holds and every width answer over it is right.
        // QNF has no boolean leaf and none is missing: which byte values are
        // legal is the LOGICAL type, and `Encoding` is what carries that
        // beside the storage.
        DType::Bool => Sig::Plain(Leaf::U(8)),
    })
}
