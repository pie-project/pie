use checkpoint::types::{DType, QuantScheme, QuantSpec};
use dtype::Dtype;
use dtype::{Elem, Fmt, spells};

fn spec(scheme: QuantScheme) -> QuantSpec {
    QuantSpec {
        scheme,
        logical_dtype: DType::Bf16,
        bits_per_element: 0,
        group_size: 0,
        channel_axis: None,
    }
}

fn sized(scheme: QuantScheme, bits: u8, group: u32) -> QuantSpec {
    QuantSpec {
        bits_per_element: bits,
        group_size: group,
        ..spec(scheme)
    }
}

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

fn row(scheme: QuantScheme) -> Fmt<'static> {
    spec(scheme)
        .term()
        .unwrap_or_else(|| panic!("{scheme:?} is listed as having a term and answered None"))
}

#[test]
fn quant_terms_every_case() {
    every_scheme_spells_the_row_the_table_says();
    the_schemes_with_no_term_say_so_rather_than_guessing();
    a_gguf_block_weighs_what_its_term_says_it_does();
    awq_and_gptq_converge_on_one_row();
    the_mlx_row_has_one_truth_and_two_doors();
    of_fmt_sorts_the_bridge_rows_into_served_and_import_only();
    an_mlx_width_the_decoder_does_not_know_gets_no_row();
    a_scalar_dtype_is_its_own_element();
    a_gguf_row_does_not_move_when_the_spec_says_otherwise();
    the_structural_answers_come_off_the_term_not_the_name();
}

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

fn the_schemes_with_no_term_say_so_rather_than_guessing() {
    for scheme in NO_ROW {
        assert_eq!(
            spec(*scheme).term(),
            None,
            "{scheme:?} answered a term the bridge has no grounds for"
        );
    }
}

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
    for (group, d) in [(32, Dtype::U2g32), (64, Dtype::U2g64), (128, Dtype::U2g128)] {
        assert_eq!(
            sized(scheme, 2, group).term().as_ref(),
            Some(d.repr()),
            "MlxAffineU4 at two bits and {group} is what {d:?} names"
        );
        assert_eq!(sized(scheme, 2, group).affine_point(), Some((group, 2)));
    }
}

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

fn an_mlx_width_the_decoder_does_not_know_gets_no_row() {
    let scheme = QuantScheme::MlxAffineU4;
    assert_eq!(sized(scheme, 6, 64).term(), None);
    assert_eq!(
        sized(QuantScheme::GptqInt4, 8, 128).term(),
        None,
        "zt.quant_group/1 names GPTQ at four bits and nowhere else"
    );
}

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

fn a_gguf_row_does_not_move_when_the_spec_says_otherwise() {
    for (scheme, spelling) in ROWS {
        if scheme.block_layout().is_none() {
            continue;
        }
        let odd = sized(*scheme, 3, 7)
            .term()
            .unwrap_or_else(|| panic!("{scheme:?} has a row"));
        assert_eq!(
            odd.mangle().as_str(),
            *spelling,
            "{scheme:?} took its widths from a declaration rather than its block"
        );
    }
}

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
    assert_eq!(spec(QuantScheme::MlxAffineU4).affine_point(), Some((64, 4)));
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
