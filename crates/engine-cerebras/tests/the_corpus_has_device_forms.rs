//! The eta compiler's engine corpus (the programs real inferlets' guests
//! look like) against the device half: which lower, over how many PEs at
//! Qwen's vocabulary, how many words and segments a PE carries; which are
//! refused, and why. No device needed.

use engine_cerebras::guest::lower::{Batch, lower};
use eta_compiler::plan::compile_bound;
use eta_ir::registry::ModelProfile;

#[test]
fn the_corpus_has_device_forms() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eta-compiler/tests/engine-corpus"
    );
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("the corpus directory")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".ptir") && !n.starts_with("neg_"))
        .collect();
    names.sort();
    let vocab = 151_936;
    let mut admitted = 0;
    let _ = vocab;
    for name in &names {
        // The corpus files are the container's bytes as hex text.
        let text = std::fs::read_to_string(format!("{dir}/{name}")).expect("a corpus file");
        let hex: Vec<u8> = text.bytes().filter(u8::is_ascii_hexdigit).collect();
        let bytes: Vec<u8> = hex
            .chunks(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        let container = eta_ir::container::decode(&bytes).expect("a trace container");
        // A trace bakes its vocabulary into the logits' shape; bind with it.
        let vocab = container
            .stages
            .iter()
            .flat_map(|s| s.ops.iter())
            .find_map(|op| match op {
                eta_ir::op::Op::IntrinsicVal {
                    intr: eta_ir::op::IntrinsicId::Logits | eta_ir::op::IntrinsicId::MtpLogits,
                    shape,
                    ..
                } => shape.dims().get(1).copied(),
                _ => None,
            })
            .unwrap_or(vocab);
        let profile = ModelProfile {
            vocab,
            ..ModelProfile::dummy()
        };
        let bound = match eta_ir::validate::bind(container, profile) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{name:32} does not bind: {e:?}");
                continue;
            }
        };
        let stages = compile_bound(&bound);
        let launch = eta_compiler::codegen::launch::build(&bound, &stages);
        let plan = match eta_exec::adopt_launch_package(launch) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{name:32} does not adopt: {e:?}");
                continue;
            }
        };
        match engine_cerebras::guest::admits(&plan.package) {
            Ok(cols) => {
                admitted += 1;
                let mut words = 0;
                let mut segments = 0;
                for at in 0..plan.package.stages.len() {
                    let l = lower(
                        &plan.package,
                        at,
                        Batch {
                            lanes: 1,
                            cols,
                            logits: None,
                            mtp: None,
                        },
                    )
                    .expect("admitted stages lower");
                    words = words.max(l.words);
                    segments = segments.max(l.text.matches("fn seg").count());
                }
                eprintln!(
                    "{name:32} device: vocab {vocab:6}, {cols:3} PEs a lane, {} stage(s), at most {words} words, {segments} segments",
                    plan.package.stages.len()
                );
            }
            Err(why) => eprintln!("{name:32} host:   vocab {vocab:6}: {why}"),
        }
    }
    assert!(admitted > 0, "some corpus program has a device form");
}
