use poem::Platform;
use poem_ir::{Elemwise, Operation, RopeFreqs};

/// Every stated table a trace turns at.
fn tables(trace: &poem_ir::Trace) -> Vec<&RopeFreqs> {
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Elemwise(
                Elemwise::RopeFull { freqs, .. }
                | Elemwise::RopePartial { freqs, .. }
                | Elemwise::RopePartialQ { freqs, .. }
                | Elemwise::RopeMrope { freqs, .. },
            ) => freqs.as_ref(),
            _ => None,
        })
        .collect()
}

/// `-yarn4` on Qwen3: each layer's rotation turns at YaRN's table over the
/// native 32768 tokens (vLLM's `YaRNScalingRotaryEmbedding`: the fastest pair
/// kept, the slowest interpolated four times, cos and sin scaled by
/// 0.1 ln 4 + 1); the name round-trips; without the word nothing is stated.
#[test]
fn a_deployment_stretched_by_yarn_turns_at_the_scaled_frequencies() {
    let catalog = poem_compiler::catalog::repository();
    let stretched = catalog
        .parse("qwen3-0.6b-bf16-kv-bf16-yarn4")
        .expect("the yarn word parses");
    assert_eq!(stretched.deploy.yarn, Some(4));
    assert_eq!(stretched.name, "qwen3-0.6b-bf16-kv-bf16-yarn4");
    let trace = stretched.trace(Platform::Cuda);
    let turned = tables(&trace);
    assert_eq!(turned.len(), 28, "one stated table a layer");
    let table = turned[0];
    assert_eq!(table.inv_freq.len(), 64);
    assert!((table.mscale - (0.1 * 4f32.ln() + 1.0)).abs() < 1e-6);
    assert!((table.inv_freq[0] - 1.0).abs() < 1e-7);
    let slowest = 1_000_000f64.powf(-63.0 / 64.0) / 4.0;
    assert!((f64::from(table.inv_freq[63]) - slowest).abs() < 1e-6 * slowest);

    let native = catalog
        .parse("qwen3-0.6b-bf16-kv-bf16")
        .expect("the listed deployment parses");
    assert!(tables(&native.trace(Platform::Cuda)).is_empty());
}

/// The form of each rotation a trace turns at a stated table.
fn stated_forms(trace: &poem_ir::Trace) -> Vec<&'static str> {
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Elemwise(Elemwise::RopeFull { freqs: Some(_), .. }) => Some("full"),
            Operation::Elemwise(
                Elemwise::RopePartial { freqs: Some(_), .. }
                | Elemwise::RopePartialQ { freqs: Some(_), .. },
            ) => Some("partial"),
            Operation::Elemwise(Elemwise::RopeMrope { freqs: Some(_), .. }) => Some("mrope"),
            _ => None,
        })
        .collect()
}

/// Qwen3.5 and later stretch their partial rotation (and the M-RoPE a vision
/// deployment turns its text by) over the native 262144 tokens: YaRN's table
/// over the 32 pairs of a 64-wide rotary prefix at theta 1e7, whose slowest
/// pair (a wavelength of ~3.8e7 tokens) is interpolated the full four times.
/// Every stretched layer turns the same table, and the native name states
/// none.
#[test]
fn qwen_3_5_and_later_stretch_their_partial_rotation() {
    let catalog = poem_compiler::catalog::repository();
    let slowest = 10_000_000f64.powf(-62.0 / 64.0) / 4.0;
    for (name, form) in [
        ("qwen35-d0.8b-bf16-kv-bf16", "partial"),
        ("qwen35-d0.8b-vision-bf16-kv-bf16", "mrope"),
        ("qwen38-flash-next-mini-bf16-kv-bf16", "partial"),
    ] {
        let stretched = format!("{name}-yarn4");
        let deployment = catalog
            .parse(&stretched)
            .unwrap_or_else(|| panic!("{stretched} parses"));
        assert_eq!(deployment.name, stretched);
        let trace = deployment.trace(Platform::Cuda);
        let forms = stated_forms(&trace);
        assert!(!forms.is_empty(), "{stretched} turns a stated table");
        assert!(
            forms.iter().all(|turned| *turned == form),
            "{stretched} turns {forms:?}, not only {form}"
        );
        let turned = tables(&trace);
        let table = turned[0];
        assert!(turned.iter().all(|other| *other == table), "{stretched}");
        assert_eq!(table.inv_freq.len(), 32, "{stretched}");
        assert!(
            (table.mscale - (0.1 * 4f32.ln() + 1.0)).abs() < 1e-6,
            "{stretched}"
        );
        assert!((table.inv_freq[0] - 1.0).abs() < 1e-7, "{stretched}");
        assert!(
            (f64::from(table.inv_freq[31]) - slowest).abs() < 1e-6 * slowest,
            "{stretched}: slowest pair turns at {}",
            table.inv_freq[31]
        );

        let native = catalog
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"));
        assert!(tables(&native.trace(Platform::Cuda)).is_empty(), "{name}");
    }
}

/// A package whose forward reads no stretch refuses one rather than serving
/// its native rotation under a stretched name.
#[test]
fn a_package_that_turns_no_scaling_refuses_a_yarn_word() {
    let catalog = poem_compiler::catalog::repository();
    let unstretchable = catalog
        .parse("gemma4-e4b-bf16-kv-bf16-yarn4")
        .expect("the yarn word parses for any model");
    let why = unstretchable
        .try_trace(Platform::Cuda)
        .expect_err("gemma 4 turns no scaling");
    assert!(why.to_string().contains("YaRN"), "{why}");
}
