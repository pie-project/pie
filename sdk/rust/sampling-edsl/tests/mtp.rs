//! mtp-logits (de-hardwired speculation): the `Binding::MtpLogits` draft-logits
//! intrinsic + the manifest-only round-trip guard. The intrinsic-kind is an
//! attach-time **manifest** property (binding-free v4), NOT a bytecode byte — so
//! an mtp draft sampler and a plain logits sampler emit byte-identical bytecode
//! and differ only in `Built.bindings`. These tests pin that contract.

use sampling_edsl::ir::{self, Binding, Op};
use sampling_edsl::{Graph, OutputKind, program};

const VOCAB: u32 = 128;

#[test]
fn mtp_argmax_binds_mtp_logits_and_round_trips() {
    let built = program::mtp_argmax(VOCAB).expect("mtp_argmax builds");

    // The single input slot carries the de-hardwired draft-logits binding.
    assert_eq!(built.bindings, vec![Binding::MtpLogits]);
    // It is an intrinsic, so it declares no host inputs.
    assert!(built.host_inputs.is_empty(), "mtp-logits is an intrinsic, not a host input");
    // Program shape: [Input(0), ReduceArgmax(0)] -> Token.
    assert_eq!(built.program.ops, vec![Op::Input(0), Op::ReduceArgmax(0)]);
    assert_eq!(built.outputs, vec![OutputKind::Token]);

    // intrinsic(mtp-logits) encode -> decode == identity (the round-trip guard
    // the bytecode owners required: prove it, don't assume it).
    let bytes = ir::encode(&built.program);
    let decoded = ir::decode(&bytes).expect("decode");
    assert_eq!(decoded, built.program, "bytecode round-trip is lossless");
}

#[test]
fn mtp_logits_is_manifest_only_not_bytecode() {
    // The mtp-logits source-select is a MANIFEST property, not bytecode: an mtp
    // draft-argmax and a plain logits-argmax produce byte-identical bytecode;
    // only `Built.bindings` differs (MtpLogits vs Logits). This is what makes
    // mtp-logits additive/manifest-only (no version bump, loud-reject by a
    // distinct binding variant rather than a layout-changing kind byte).
    let mtp = program::mtp_argmax(VOCAB).expect("mtp builds");

    let g = Graph::new(VOCAB);
    let token = g.intrinsic_logits_dyn().argmax();
    g.output(&token, OutputKind::Token);
    let plain = g.build().expect("plain logits argmax builds");

    assert_eq!(
        ir::encode(&mtp.program),
        ir::encode(&plain.program),
        "binding is manifest-only -> bytecode identical"
    );
    assert_eq!(mtp.bindings, vec![Binding::MtpLogits]);
    assert_eq!(plain.bindings, vec![Binding::Logits]);
    assert_ne!(mtp.bindings, plain.bindings, "only the manifest binding distinguishes them");
}
