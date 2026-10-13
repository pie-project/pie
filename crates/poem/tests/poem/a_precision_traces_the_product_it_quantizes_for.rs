use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, Platform, Precision, Value, Weight, ops, trace_hybrid,
};
use poem_ir::{ActQuant, Def, Linear, Operation, Trace};

const HIDDEN: u64 = 256;

const VOCAB: u32 = 512;

struct OneProduct(Precision);

impl ForwardHybrid for OneProduct {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) -> Value {
        let table = Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16);
        let x = ops::layout::embed(&inputs.tokens(), &table, VOCAB);
        let w = Weight::at("proj", [384, HIDDEN], self.0);
        ops::linear::matmul(&x, &w)
    }
}

fn traced(precision: Precision) -> Trace {
    trace_hybrid("one_product", &OneProduct(precision), Platform::Cuda)
}

fn product(trace: &Trace) -> &Linear {
    trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Linear(op @ (Linear::Matmul { .. } | Linear::MatmulActQuant { .. })) => {
                Some(op)
            }
            _ => None,
        })
        .expect("the trace holds its product")
}

fn param_of(trace: &Trace, id: poem_ir::ValueId) -> &poem_ir::Param {
    match trace.values[id.0 as usize].def {
        Def::Weight(at) => &trace.params[at as usize],
        ref other => panic!("value {} is {other:?}, not a weight", id.0),
    }
}

#[test]
fn a_precision_traces_the_product_it_quantizes_for_every_case() {
    a_plain_dtype_reads_its_activations_as_they_are();
    an_fp8_block_weight_lands_raw_codes_beside_its_block_exponents();
    the_words_name_each_precision();
}

fn a_plain_dtype_reads_its_activations_as_they_are() {
    let trace = traced(Precision::of(Dtype::Bf16));
    assert!(matches!(product(&trace), Linear::Matmul { .. }));
}

fn an_fp8_block_weight_lands_raw_codes_beside_its_block_exponents() {
    let w8a8 = Precision::ACTED
        .into_iter()
        .find(|p| p.dtype == Dtype::Fp8b128)
        .expect("fp8b128 is read at a W8A8 precision");
    let trace = traced(w8a8);
    let Linear::MatmulActQuant {
        w,
        quant,
        input_scales,
        scales: Some(scales),
        ..
    } = product(&trace)
    else {
        panic!("an fp8b128a8 product names its block exponents")
    };
    assert_eq!(*quant, ActQuant::E4m3 { gran: 128 });
    assert_eq!(
        *input_scales, None,
        "e4m3 activations quantize under no static global"
    );
    let (codes, exps) = (param_of(&trace, *w), param_of(&trace, *scales));
    assert_eq!(
        (codes.dtype, codes.shape.as_slice()),
        (Dtype::E4m3, &[384, HIDDEN][..])
    );
    assert_eq!(
        (exps.name.as_str(), exps.dtype, exps.shape.as_slice()),
        ("proj.scales", Dtype::E8m0, &[3, 2][..]),
        "one exponent a 128x128 block"
    );
}

fn the_words_name_each_precision() {
    let words: Vec<String> = Precision::ACTED
        .into_iter()
        .map(poem::star::precision_word)
        .collect();
    assert_eq!(words, ["nvfp4a4", "mxfp4a8", "fp8b128a8"]);
    for word in &words {
        assert_eq!(
            poem::star::precision_of(word)
                .map(poem::star::precision_word)
                .as_ref(),
            Some(word),
            "`{word}` reads back as itself"
        );
    }
}
