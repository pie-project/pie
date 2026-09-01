//! `QuantSpec::term`, checked against what this crate already knows.
//!
//! A mapping table is prose until something computes with it. Every term
//! [`QuantSpec::term`] emits is a claim about bytes, and this crate holds the
//! independent statement of those bytes — `QuantScheme::block_layout`'s
//! `(elements, bytes)` pairs, measured off real GGUF files rather than read
//! off a struct — so the two can be multiplied against each other. That is
//! what the first test does, and it is the reason a wrong element or a
//! misremembered sub-block width cannot survive here: `q4_k` and `q5_k` differ
//! by one bit of code width and 32 bytes of block, and no other pair of
//! numbers satisfies both.
//!
//! The rest pin the three properties the bridge exists to have: that two
//! pipelines shipping identical bytes land on ONE row, that a scheme and the
//! dtype naming it land on the SAME row, and that a served row's term is
//! recognized by `Dtype::of_fmt` while an unserved one is refused.

use checkpoint::types::{DType, QuantScheme, QuantSpec};
use dtype::Dtype;
use dtype::{Elem, Fmt, spells};

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

/// Every scheme that has a term, with the spelling it takes under a
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

/// The term, or a failure naming the scheme that had none.
fn row(scheme: QuantScheme) -> Fmt<'static> {
    spec(scheme).term()
        .unwrap_or_else(|| panic!("{scheme:?} is listed as having a term and answered None"))
}

#[test]
fn every_scheme_spells_the_row_the_table_says() {
    for (scheme, spelling) in ROWS {
        assert_eq!(
            row(*scheme).mangle().as_str(),
            *spelling,
            "{scheme:?} does not spell what the bridge's table says"
        );
        assert!(
            spells(&row(*scheme), spelling),
            "{scheme:?}: Display and the const walker disagree"
        );
    }
}

#[test]
fn the_schemes_with_no_term_say_so_rather_than_guessing() {
    for scheme in NO_ROW {
        assert_eq!(
            spec(*scheme).term(),
            None,
            "{scheme:?} answered a term the bridge has no grounds for"
        );
    }
}

/// **THE CROSS-CHECK.** A GGUF block's byte count is stated twice over: once
/// by `QuantScheme::block_layout`, which this crate measured, and once by the
/// term, which says bits per weight and knows nothing about ggml. The two
/// have to agree exactly, and for all eleven blocks they do — the scales, the
/// mins, the sub-block bytes and the super-block f16s all fall out of the term.
///
/// No skips and no tolerance. Every one of these containers is packed with
/// nothing wasted: `q3_k`'s twelve bytes carry sixteen six-bit scales,
/// `q4_k`'s twelve carry eight scales AND eight mins, and `q6_k`'s 210 are
/// 128 + 64 + 16 + 2 with no padding anywhere. A block that ever does carry
/// padding would fail here, and the right answer then is to say so in the
/// scheme's own docs rather than to loosen this.
#[test]
fn a_gguf_block_weighs_what_its_term_says_it_does() {
    let mut checked = 0;
    for (scheme, _) in ROWS {
        let Some((elems, bytes)) = scheme.block_layout() else {
            continue;
        };
        let fmt = row(*scheme);
        let k = u32::try_from(elems).expect("a block's element count fits u32");
        let bpw = fmt
            .bpw(k)
            .unwrap_or_else(|| panic!("{scheme:?} has a rate for every element"));
        assert_eq!(
            bpw * f64::from(k),
            (bytes * 8) as f64,
            "{scheme:?}: {fmt} says {bpw} bits per weight over {elems} elements, \
             which is not the {bytes}-byte block this crate measured"
        );
        checked += 1;
    }
    assert_eq!(checked, 11, "the GGUF rows the bridge maps");
}

/// **AWQ AND GPTQ ARE ONE ROW.** They ship the identical numbers and differ in
/// the order nibbles sit inside a word, which `file/zt.rs` recovers the scheme
/// from and which no arithmetic ever sees. A dispatch table keyed on the
/// term serves both with one kernel, and that convergence is the whole
/// argument for naming rows rather than pipelines.
#[test]
fn awq_and_gptq_converge_on_one_row() {
    let awq = sized(QuantScheme::AwqInt4, 4, 128).term();
    let gptq = sized(QuantScheme::GptqInt4, 4, 128).term();
    assert_eq!(awq, gptq, "two pipelines, identical bytes, one row");
    assert_eq!(
        awq.expect("the row exists").mangle().as_str(),
        "g128_u4_f16_z_u4",
        "the row GPTQ, AWQ and compressed-tensors all publish"
    );
}

/// One truth, two doors: a plane that reaches the loader as
/// `QuantScheme::MlxAffineU4` at four bits and 64, and a model text that says
/// `Dtype::U4g64`, are the same format and must produce the same value. The
/// dtype is the declaration's shorthand for the scheme; if the two doors ever
/// disagreed the shorthand would be a second format.
#[test]
fn the_mlx_row_has_one_truth_and_two_doors() {
    let scheme = QuantScheme::MlxAffineU4;
    assert_eq!(
        sized(scheme, 4, 64).term().as_ref(),
        Some(Dtype::U4g64.repr()),
        "MlxAffineU4 at four bits and 64 is what U4g64 names"
    );
    assert_eq!(
        sized(scheme, 8, 64).term().as_ref(),
        Some(Dtype::U8g64.repr()),
        "the router gates' width is the same scheme, one spec field over"
    );
    assert_eq!(
        sized(scheme, 4, 32).term().as_ref(),
        Some(Dtype::U4g32.repr()),
        "a row too narrow for 64 is the same scheme, the other spec field over"
    );
    // THE THIRD WIDTH IS A DOOR TOO. The DQ stacks quantize their expert
    // banks to two bits at three groups, and each of the three is a `Dtype`
    // the engine lands. A width missing here is not a narrower answer: it is
    // `term()` answering `None`, which is `affine_point()` answering `None`,
    // which is `engine-metal` refusing the bank for carrying scale factors
    // with nothing to be factors of.
    for (group, d) in [(32, Dtype::U2g32), (64, Dtype::U2g64), (128, Dtype::U2g128)] {
        assert_eq!(
            sized(scheme, 2, group).term().as_ref(),
            Some(d.repr()),
            "MlxAffineU4 at two bits and {group} is what {d:?} names"
        );
        assert_eq!(sized(scheme, 2, group).affine_point(), Some((group, 2)));
    }
}

/// **SERVED IS `of_fmt` SAYING SO.** The bridge's terms fall in two piles:
/// those the engine lands (a `Dtype` variant exists — the k-quants, mxfp4,
/// the per-row e4m3, MLX's three) and those it merely reads (GPTQ's row,
/// `q4_1`, the per-row e5m2 — repack or refuse). The pile is not a new
/// table: it is `Dtype::of_fmt`, asked of every row the bridge emits.
#[test]
fn of_fmt_sorts_the_bridge_rows_into_served_and_import_only() {
    let served: &[(QuantScheme, Dtype)] = &[
        (QuantScheme::Fp8E4M3, Dtype::E4m3row),
        (QuantScheme::Mxfp4E2M1E8M0, Dtype::Mxfp4),
        (QuantScheme::MlxAffineU4, Dtype::U4g64),
        (QuantScheme::GgufQ2K, Dtype::U2g16k),
        (QuantScheme::GgufQ3K, Dtype::I3g16k),
        (QuantScheme::GgufQ4K, Dtype::U4g32k),
        (QuantScheme::GgufQ5K, Dtype::U5g32k),
        (QuantScheme::GgufQ6K, Dtype::I6g16k),
        (QuantScheme::GgufMxfp4, Dtype::Mxfp4),
    ];
    for (scheme, d) in served {
        assert_eq!(
            Dtype::of_fmt(&row(*scheme)),
            Some(*d),
            "{scheme:?} names a served format"
        );
    }
    let import_only = [
        QuantScheme::Fp8E5M2,
        QuantScheme::Int8Symmetric,
        QuantScheme::AwqInt4,
        QuantScheme::GptqInt4,
        QuantScheme::Int4B8,
        QuantScheme::GgufQ4_0,
        QuantScheme::GgufQ4_1,
        QuantScheme::GgufQ5_0,
        QuantScheme::GgufQ5_1,
        QuantScheme::GgufQ8_0,
    ];
    for scheme in import_only {
        assert_eq!(
            Dtype::of_fmt(&row(scheme)),
            None,
            "{scheme:?} is import-only: readable, not landable as stored"
        );
    }
}

/// The two widths MLX quantizes at are the two `codec/mlx.rs` decodes; a
/// third is a plane this crate cannot read, so it gets no row.
#[test]
fn an_mlx_width_the_decoder_does_not_know_gets_no_row() {
    let scheme = QuantScheme::MlxAffineU4;
    assert_eq!(sized(scheme, 6, 64).term(), None);
    assert_eq!(
        sized(QuantScheme::GptqInt4, 8, 128).term(),
        None,
        "zt.quant_group/1 names GPTQ at four bits and nowhere else"
    );
}

/// A dtype needs no bridge: its term is its own `repr`, and the scalar rows
/// are `Elem` of their own element and nothing more — no group, no gain, no
/// offset — which is what makes a raw tensor's term the element itself.
#[test]
fn a_scalar_dtype_is_its_own_element() {
    assert_eq!(DType::F32.repr(), &Fmt::Elem(Elem::F32));
    assert_eq!(DType::Bf16.repr(), &Fmt::Elem(Elem::Bf16));
    assert_eq!(
        DType::E4m3.repr(),
        &Fmt::Elem(Elem::E { e: 4, m: 3 }),
        "an fp8 element is an element; the per-row f32 scale beside it is the \
         SCHEME's fact and fmt_of_scheme is what states it"
    );
    assert_eq!(
        DType::I8.repr(),
        &Fmt::Elem(Elem::I(8)),
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
        let odd = sized(*scheme, 3, 7).term()
            .unwrap_or_else(|| panic!("{scheme:?} has a row"));
        assert_eq!(
            odd.mangle().as_str(),
            *spelling,
            "{scheme:?} took its widths from a declaration rather than its block"
        );
    }
}

/// The two structural answers the plan hands an engine, read off the term.
///
/// `is_mxfp4` is the two-plane form only: `GgufMxfp4` shares the algebra but
/// interleaves its scale byte, and a binder asking about two-plane bytes must
/// not be told yes. `affine_point` is the term's `(group, bits)` for the
/// leaf-per-plane integer-code families and `None` for everything a block
/// decoder or a dedicated kernel owns — whatever numbers the spec carries.
#[test]
fn the_structural_answers_come_off_the_term_not_the_name() {
    assert!(spec(QuantScheme::Mxfp4E2M1E8M0).is_mxfp4());
    assert!(
        !spec(QuantScheme::GgufMxfp4).is_mxfp4(),
        "the interleaved block is the same algebra in a different container"
    );
    assert!(!spec(QuantScheme::MlxAffineU4).is_mxfp4());

    assert_eq!(
        sized(QuantScheme::MlxAffineU4, 4, 64).affine_point(),
        Some((64, 4))
    );
    assert_eq!(
        sized(QuantScheme::MlxAffineU4, 8, 64).affine_point(),
        Some((64, 8))
    );
    assert_eq!(
        sized(QuantScheme::AwqInt4, 4, 128).affine_point(),
        Some((128, 4))
    );
    assert_eq!(
        sized(QuantScheme::Int4B8, 4, 128).affine_point(),
        Some((128, 4)),
        "excess-binary codes read at an affine point too"
    );
    // A default-shaped MLX spec reads at the scheme's default point.
    assert_eq!(spec(QuantScheme::MlxAffineU4).affine_point(), Some((64, 4)));
    // Everything a block decoder or a dedicated kernel owns: no point.
    assert_eq!(spec(QuantScheme::Mxfp4E2M1E8M0).affine_point(), None);
    assert_eq!(
        spec(QuantScheme::GgufQ4_0).affine_point(),
        None,
        "a self-contained block reads by its own decoder, whatever its term's numbers"
    );
    assert_eq!(spec(QuantScheme::GgufQ4K).affine_point(), None);
    assert_eq!(
        spec(QuantScheme::Fp8E4M3).affine_point(),
        None,
        "a per-row scale has no finite group"
    );
    assert_eq!(spec(QuantScheme::None).affine_point(), None);
}
