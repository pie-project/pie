//! The canon bridge, checked against what this crate already knows.
//!
//! A mapping table is prose until something computes with it. Every row
//! [`checkpoint::qnf`] emits is a claim about bytes, and this crate holds the
//! independent statement of those bytes — `QuantScheme::block_layout`'s
//! `(elements, bytes)` pairs, measured off real GGUF files rather than read
//! off a struct — so the two can be multiplied against each other. That is
//! what the first test does, and it is the reason a wrong leaf or a
//! misremembered sub-block width cannot survive here: `Q4_K` and `Q5_K` differ
//! by one bit of code width and 32 bytes of block, and no other pair of
//! numbers satisfies both.
//!
//! The rest pin the three properties the bridge exists to have: that two
//! pipelines shipping identical bytes land on ONE row, that a scheme and the
//! dtype naming it land on the SAME row, and that every row it emits is
//! canonical and survives its own spelling.

use checkpoint::qnf::{sig_of_dtype, sig_of_scheme};
use checkpoint::types::{DType, QuantScheme, QuantSpec};
use dtype::quant::{Leaf, Sig};

/// A spec whose numbers are the scheme's own defaults, which is what a zero
/// field means everywhere else in this crate.
fn spec(scheme: QuantScheme) -> QuantSpec {
    QuantSpec {
        scheme,
        logical_dtype: DType::Bf16,
        bits_per_element: 0,
        group_size: 0,
        channel_axis: None,
    }
}

/// The same, at a stated width and group.
fn sized(scheme: QuantScheme, bits: u8, group: u32) -> QuantSpec {
    QuantSpec {
        bits_per_element: bits,
        group_size: group,
        ..spec(scheme)
    }
}

/// Every scheme that has a signature, with the spelling it takes under a
/// default-shaped spec.
///
/// The parametric rows are here at their DEFAULTS, which is why AWQ reads
/// `g32` and not the `g128` its checkpoints ship: `default_group_size` answers
/// 32 for it, and this table is what that default spells. The group is the
/// spec's everywhere it is a spec field at all.
const ROWS: &[(QuantScheme, &str)] = &[
    (QuantScheme::Fp8E4M3, "gr_e4m3_f32_n"),
    (QuantScheme::Fp8E5M2, "gr_e5m2_f32_n"),
    (QuantScheme::Int8Symmetric, "gr_i8_f32_n"),
    (QuantScheme::AwqInt4, "g32_u4_f16_z_u4"),
    (QuantScheme::GptqInt4, "g32_u4_f16_z_u4"),
    (QuantScheme::Mxfp4E2M1E8M0, "g32_e2m1_e8m0_n"),
    (QuantScheme::MlxAffineU4, "g64_u4_bf16_b_bf16"),
    (QuantScheme::Int4B8, "g32_i4_f16_n"),
    (QuantScheme::GgufQ4_0, "g32_i4_f16_n"),
    (QuantScheme::GgufQ4_1, "g32_u4_f16_b_f16"),
    (QuantScheme::GgufQ5_0, "g32_i5_f16_n"),
    (QuantScheme::GgufQ5_1, "g32_u5_f16_b_f16"),
    (QuantScheme::GgufQ8_0, "g32_i8_f16_n"),
    (QuantScheme::GgufQ2K, "g16_u2_g16_u4_f16_n_b_g16_u4_f16_n"),
    (QuantScheme::GgufQ3K, "g16_i3_g16_i6_f16_n_n"),
    (QuantScheme::GgufQ4K, "g32_u4_g8_u6_f16_n_b_g8_u6_f16_n"),
    (QuantScheme::GgufQ5K, "g32_u5_g8_u6_f16_n_b_g8_u6_f16_n"),
    (QuantScheme::GgufQ6K, "g16_i6_g16_i8_f16_n_n"),
    (QuantScheme::GgufMxfp4, "g32_e2m1_e8m0_n"),
];

/// Every scheme that has none, and the family it belongs to.
///
/// Three reasons, and none of them is "not implemented yet" in the sense that
/// invites a guess later: a raw tensor's row is the dtype's to give, a
/// codebook needs a table registry that does not exist, and a lattice has no
/// node in the algebra at all.
const NO_ROW: &[QuantScheme] = &[
    QuantScheme::None,
    QuantScheme::Int8Asymmetric,
    QuantScheme::GgufIq4Nl,
    QuantScheme::GgufIq4Xs,
    QuantScheme::GgufIq2Xxs,
    QuantScheme::GgufIq2Xs,
    QuantScheme::GgufIq2S,
    QuantScheme::GgufIq3Xxs,
    QuantScheme::GgufIq3S,
];

/// Every dtype, with the row it names.
const DTYPES: &[(DType, &str)] = &[
    (DType::F32, "f32"),
    (DType::F16, "f16"),
    (DType::Bf16, "bf16"),
    (DType::Fp8E4m3, "e4m3"),
    (DType::Fp8E5m2, "e5m2"),
    (DType::Fp4, "e2m1"),
    (DType::Mxfp4, "g32_e2m1_e8m0_n"),
    (DType::MlxU4, "g64_u4_bf16_b_bf16"),
    (DType::MlxU8, "g64_u8_bf16_b_bf16"),
    (DType::MlxU4G32, "g32_u4_bf16_b_bf16"),
    (DType::E8m0, "e8m0"),
    (DType::I64, "i64"),
    (DType::I32, "i32"),
    (DType::I16, "i16"),
    (DType::I8, "i8"),
    (DType::U64, "u64"),
    (DType::U32, "u32"),
    (DType::U16, "u16"),
    (DType::U8, "u8"),
    (DType::Bool, "u8"),
];

/// The signature, or a failure naming the scheme that had none.
fn row(scheme: QuantScheme) -> Sig {
    sig_of_scheme(scheme, &spec(scheme))
        .unwrap_or_else(|| panic!("{scheme:?} is listed as having a signature and answered None"))
}

#[test]
fn every_scheme_spells_the_row_the_table_says() {
    for (scheme, spelling) in ROWS {
        assert_eq!(
            row(*scheme).mangle().as_str(),
            *spelling,
            "{scheme:?} does not spell what the bridge's table says"
        );
    }
}

#[test]
fn the_schemes_with_no_signature_say_so_rather_than_guessing() {
    for scheme in NO_ROW {
        assert_eq!(
            sig_of_scheme(*scheme, &spec(*scheme)),
            None,
            "{scheme:?} answered a signature the bridge has no grounds for"
        );
    }
}

/// **THE CROSS-CHECK.** A GGUF block's byte count is stated twice over: once
/// by `QuantScheme::block_layout`, which this crate measured, and once by the
/// signature, which says bits per weight and knows nothing about ggml. The two
/// have to agree exactly, and for all eleven blocks they do — the scales, the
/// mins, the sub-block bytes and the super-block f16s all fall out of the term.
///
/// No skips and no tolerance. Every one of these containers is packed with
/// nothing wasted: `Q3_K`'s twelve bytes carry sixteen six-bit scales,
/// `Q4_K`'s twelve carry eight scales AND eight mins, and `Q6_K`'s 210 are
/// 128 + 64 + 16 + 2 with no padding anywhere. A block that ever does carry
/// padding would fail here, and the right answer then is to say so in the
/// scheme's own docs rather than to loosen this.
#[test]
fn a_gguf_block_weighs_what_its_signature_says_it_does() {
    let mut checked = 0;
    for (scheme, _) in ROWS {
        let Some((elems, bytes)) = scheme.block_layout() else {
            continue;
        };
        let sig = row(*scheme);
        let k = u32::try_from(elems).expect("a block's element count fits u32");
        let bpw = sig
            .bpw(k)
            .unwrap_or_else(|| panic!("{scheme:?} has a rate for every leaf"));
        assert_eq!(
            bpw * f64::from(k),
            (bytes * 8) as f64,
            "{scheme:?}: {sig} says {bpw} bits per weight over {elems} elements, \
             which is not the {bytes}-byte block this crate measured"
        );
        checked += 1;
    }
    assert_eq!(checked, 11, "the GGUF rows the bridge maps");
}

/// **AWQ AND GPTQ ARE ONE ROW.** They ship the identical numbers and differ in
/// the order nibbles sit inside a word, which `file/zt.rs` recovers the scheme
/// from and which no arithmetic ever sees. A dispatch table keyed on the
/// signature serves both with one kernel, and that convergence is the whole
/// argument for naming rows rather than pipelines.
#[test]
fn awq_and_gptq_converge_on_one_row() {
    let awq = sig_of_scheme(QuantScheme::AwqInt4, &sized(QuantScheme::AwqInt4, 4, 128));
    let gptq = sig_of_scheme(QuantScheme::GptqInt4, &sized(QuantScheme::GptqInt4, 4, 128));
    assert_eq!(awq, gptq, "two pipelines, identical bytes, one row");
    assert_eq!(
        awq.expect("the row exists").mangle().as_str(),
        "g128_u4_f16_z_u4",
        "the row GPTQ, AWQ and compressed-tensors all publish"
    );
}

/// One truth, two doors: a plane that reaches the loader as
/// `QuantScheme::MlxAffineU4` at four bits and 64, and a model text that says
/// `DType::MlxU4`, are the same format and must produce the same value. The
/// dtype is the declaration's shorthand for the scheme, which is what
/// `DType::MlxU4`'s own docs say; if the two doors ever disagreed the
/// shorthand would be a second format.
#[test]
fn the_mlx_row_has_one_truth_and_two_doors() {
    let scheme = QuantScheme::MlxAffineU4;
    assert_eq!(
        sig_of_scheme(scheme, &sized(scheme, 4, 64)),
        sig_of_dtype(DType::MlxU4),
        "MlxAffineU4 at four bits and 64 is what MlxU4 names"
    );
    assert_eq!(
        sig_of_scheme(scheme, &sized(scheme, 8, 64)),
        sig_of_dtype(DType::MlxU8),
        "the router gates' width is the same scheme, one spec field over"
    );
    assert_eq!(
        sig_of_scheme(scheme, &sized(scheme, 4, 32)),
        sig_of_dtype(DType::MlxU4G32),
        "a row too narrow for 64 is the same scheme, the other spec field over"
    );
}

/// The two widths MLX quantizes at are the two `codec/mlx.rs` decodes; a
/// third is a plane this crate cannot read, so it gets no row.
#[test]
fn an_mlx_width_the_decoder_does_not_know_gets_no_row() {
    let scheme = QuantScheme::MlxAffineU4;
    assert_eq!(sig_of_scheme(scheme, &sized(scheme, 6, 64)), None);
    assert_eq!(
        sig_of_scheme(QuantScheme::GptqInt4, &sized(QuantScheme::GptqInt4, 8, 128)),
        None,
        "zt.quant_group/1 names GPTQ at four bits and nowhere else"
    );
}

/// Every row the bridge can emit is canonical and survives its own spelling.
///
/// Canonical is the property that makes a row nameable at all — a `Sub::Nil`
/// in a factor slot has no spelling and `mangle` panics on one — and the round
/// trip is what makes the spelling the NAME rather than a rendering of it.
/// Both doors are walked, at every width and group the parametric families
/// take in this tree.
#[test]
fn every_row_the_bridge_emits_is_canonical_and_round_trips() {
    let mut rows: Vec<(String, Sig)> = Vec::new();
    for (scheme, _) in ROWS {
        rows.push((format!("{scheme:?}"), row(*scheme)));
    }
    for (scheme, bits, group) in [
        (QuantScheme::MlxAffineU4, 4, 32),
        (QuantScheme::MlxAffineU4, 8, 64),
        (QuantScheme::AwqInt4, 4, 128),
        (QuantScheme::GptqInt4, 4, 64),
        (QuantScheme::Int4B8, 4, 128),
        (QuantScheme::Mxfp4E2M1E8M0, 4, 32),
    ] {
        let sig = sig_of_scheme(scheme, &sized(scheme, bits, group))
            .unwrap_or_else(|| panic!("{scheme:?} at {bits} bits and {group} has a row"));
        rows.push((format!("{scheme:?} at {bits}/{group}"), sig));
    }
    for (d, _) in DTYPES {
        let sig = sig_of_dtype(*d).unwrap_or_else(|| panic!("{d:?} has a row"));
        rows.push((format!("{d:?}"), sig));
    }
    for (who, sig) in rows {
        assert!(sig.is_canonical(), "{who} emitted a non-canonical row");
        let spelling = sig.mangle();
        assert_eq!(
            Sig::parse(&spelling),
            Ok(sig),
            "{who} spells {spelling}, which does not parse back to itself"
        );
    }
}

#[test]
fn every_dtype_names_the_row_the_table_says() {
    for (d, spelling) in DTYPES {
        let sig = sig_of_dtype(*d).unwrap_or_else(|| panic!("{d:?} has a row"));
        assert_eq!(
            sig.mangle().as_str(),
            *spelling,
            "{d:?} does not spell what the bridge's table says"
        );
    }
}

/// The scalar rows are `Plain` of their own leaf and nothing more — no group,
/// no gain, no offset — which is what makes a raw tensor's signature the
/// element itself.
#[test]
fn a_scalar_dtype_is_plain_of_its_own_leaf() {
    assert_eq!(sig_of_dtype(DType::F32), Some(Sig::Plain(Leaf::F32)));
    assert_eq!(sig_of_dtype(DType::Bf16), Some(Sig::Plain(Leaf::Bf16)));
    assert_eq!(
        sig_of_dtype(DType::Fp8E4m3),
        Some(Sig::Plain(Leaf::E { e: 4, m: 3 })),
        "an fp8 element is a leaf; the per-row f32 scale beside it is the \
         SCHEME's fact and sig_of_scheme is what states it"
    );
    assert_eq!(
        sig_of_dtype(DType::I8),
        Some(Sig::Plain(Leaf::I(8))),
        "a kv page's i8 codes: the element, not the cache row's granularity"
    );
}

/// A blocked scheme ignores the spec's numbers, because its block already
/// answered them. `default_group_size`'s own docs call those fields inert
/// here; this is that word spent.
#[test]
fn a_gguf_row_does_not_move_when_the_spec_says_otherwise() {
    for (scheme, spelling) in ROWS {
        if scheme.block_layout().is_none() {
            continue;
        }
        let odd = sig_of_scheme(*scheme, &sized(*scheme, 3, 7))
            .unwrap_or_else(|| panic!("{scheme:?} has a row"));
        assert_eq!(
            odd.mangle().as_str(),
            *spelling,
            "{scheme:?} took its widths from a declaration rather than its block"
        );
    }
}
