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

#[test]
fn mtp_logits_matrix_k_binds_and_validates_stage2() {
    // Stage 2 (PTIR-native MTP): the [K, vocab] draft-proposal matrix — one
    // row per draft position, argmax per row = the K fresh drafts (overview
    // §6.1's `drf = reduce_argmax(mtp_logits)`). K is trace-known.
    const K: u32 = 3;
    let g = Graph::new(VOCAB);
    let drafts = g.intrinsic_mtp_logits_matrix_dyn(K); // [K, vocab] f32
    let toks = drafts.argmax(); // per-row -> [K] i32
    g.output(&toks, OutputKind::Token);
    let built = g.build().expect("stage-2 matrix mtp builds");

    assert_eq!(built.bindings, vec![Binding::MtpLogits]);
    assert!(built.host_inputs.is_empty());
    // The decl carries the trace-known K (the driver reads it from here).
    let decl = &built.program.inputs[0];
    assert_eq!(decl.shape.dims(), &[K, VOCAB]);
    assert_eq!(decl.dtype, ir::DType::F32);
    // The shared attach contract accepts it (the same check the runtime
    // enforces at bind — one definition, no drift).
    assert!(ir::validate::intrinsic_decl_ok(&Binding::MtpLogits, decl, Some(VOCAB)));
    assert!(!ir::validate::intrinsic_decl_ok(&Binding::MtpLogits, decl, Some(VOCAB + 1)));

    // Round-trip + stable identity: same program -> same bytes -> same hash;
    // and it hashes IDENTICALLY to a plain-Logits matrix program (intrinsic
    // kind is not distinguished — the driver dedup contract).
    let bytes = ir::encode(&built.program);
    assert_eq!(ir::decode(&bytes).expect("decode"), built.program);
    let g2 = Graph::new(VOCAB);
    let t2 = g2.intrinsic_logits_matrix_dyn(K).argmax();
    g2.output(&t2, OutputKind::Token);
    let plain = g2.build().expect("plain matrix builds");
    assert_eq!(ir::encode(&built.program), ir::encode(&plain.program));
    assert_eq!(
        ir::program_identity_hash(&bytes, &built.bindings),
        ir::program_identity_hash(&ir::encode(&plain.program), &plain.bindings),
        "Logits and MtpLogits share one compiled identity (dedup contract)"
    );
}

#[test]
fn stage2_composed_target_plus_mtp_matrix_builds() {
    // The full Stage-2 §6.1 binding shape bravo's mtpverify moves to: ONE
    // program consuming BOTH intrinsics — target `logits [K+1, vocab]`
    // (verify rows) and `mtp_logits [K, vocab]` (next-step draft proposals).
    // Distinct manifest bindings, one bytecode; slot order = input order.
    const K: u32 = 3;
    let g = Graph::new(VOCAB);
    let target = g.intrinsic_logits_matrix_dyn(K + 1); // [K+1, vocab]
    let drafts = g.intrinsic_mtp_logits_matrix_dyn(K); // [K, vocab]
    let picked = target.argmax(); // [K+1]
    let drf = drafts.argmax(); // [K]
    g.output(&picked, OutputKind::Token);
    g.output(&drf, OutputKind::Token);
    let built = g.build().expect("composed stage-2 program builds");
    assert_eq!(built.bindings, vec![Binding::Logits, Binding::MtpLogits]);
    assert_eq!(built.program.inputs[0].shape.dims(), &[K + 1, VOCAB]);
    assert_eq!(built.program.inputs[1].shape.dims(), &[K, VOCAB]);
    for (b, d) in built.bindings.iter().zip(&built.program.inputs) {
        assert!(ir::validate::intrinsic_decl_ok(b, d, Some(VOCAB)));
    }
    let bytes = ir::encode(&built.program);
    assert_eq!(ir::decode(&bytes).expect("decode"), built.program);
}
