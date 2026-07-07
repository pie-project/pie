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

#[test]
fn mtp_native_verify_matches_the_golden_tail_contract() {
    // Byte-aligned to echo's `mtp_verify_tail` golden values (K=3, V=8):
    // picked=[3,5,2,4], embedded drafts=[3,5,6] -> hits at 0,1; miss at 2 ->
    // n_acc=2 -> commit = [3, 5, 2, -1] (prefix + BONUS picked[2] + sentinel);
    // fresh drafts = argmax(mtp) = [1, 4, 0].
    use pie_sampling_ir::eval::{eval, InputBindings, Value};
    const K: u32 = 3;
    const V8: u32 = 8;
    let (built, keys) = program::mtp_native_verify(V8, K).expect("builds");
    built.program.validate().expect("validates");
    let bytes = ir::encode(&built.program);
    assert_eq!(ir::decode(&bytes).expect("decode"), built.program);
    assert_eq!(
        built.bindings[..2],
        [Binding::Logits, Binding::MtpLogits],
        "intrinsics first, in declaration order"
    );
    let mut target = vec![0.0f32; ((K + 1) * V8) as usize];
    target[3] = 9.0; // row0 -> 3 (hit d1=3)
    target[8 + 5] = 9.0; // row1 -> 5 (hit d2=5)
    target[16 + 2] = 9.0; // row2 -> 2 (MISS vs d3=6) -> bonus = 2
    target[24 + 4] = 9.0; // row3 -> 4 (never reached)
    let mut mtp = vec![0.0f32; (K * V8) as usize];
    mtp[1] = 7.0;
    mtp[8 + 4] = 7.0;
    mtp[16] = 7.0;
    let out = eval(
        &built.program,
        &InputBindings::new(
            &[
                Value::F32(target),
                Value::F32(mtp),
                Value::I32(vec![3, 5, 6]),
                Value::I32(vec![0, 1, 2]),
                Value::F32(vec![0.0, 1.0, 2.0, 3.0]),
            ],
            0,
        ),
    )
    .expect("evals");
    assert_eq!(out[0], Value::I32(vec![3, 5, 2, -1]), "prefix + bonus + sentinel");
    assert_eq!(out[1], Value::I32(vec![1, 4, 0]), "fresh drafts for the next window");
    let _ = keys;
}

#[test]
fn mtp_specdecode_seed_is_the_bonus_token_device_resident() {
    // `mtp_specdecode` = `mtp_native_verify` + out[2] = the on-device SEED
    // (`picked[n_acc]` = the bonus = the host's `committed.last()`), so the
    // `[seed, drafts]` next-window is driver-composed with NO host round-trip
    // (ptir-mtp-specdecode-spec §8.1, echo path (a)).
    //
    // Same golden vectors as the tail contract (K=3, V=8): picked=[3,5,2,4],
    // drafts=[3,5,6] -> hits 0,1; miss 2 -> n_acc=2 -> commit=[3,5,2,-1],
    // fresh=[1,4,0], and SEED = picked[n_acc] = picked[2] = 2.
    use pie_sampling_ir::eval::{eval, InputBindings, Value};
    const K: u32 = 3;
    const V8: u32 = 8;
    let (built, _keys) = program::mtp_specdecode(V8, K).expect("builds");
    built.program.validate().expect("validates");
    // out[0], out[1] identical to mtp_native_verify; out[2] is the new seed.
    assert_eq!(built.outputs, vec![OutputKind::Token, OutputKind::Token, OutputKind::Token]);
    assert_eq!(built.bindings[..2], [Binding::Logits, Binding::MtpLogits]);

    let eval_case = |picked_rows: &[usize], drafts: &[i32], mtp_rows: &[usize]| {
        let mut target = vec![0.0f32; ((K + 1) * V8) as usize];
        for (r, &c) in picked_rows.iter().enumerate() {
            target[r * V8 as usize + c] = 9.0;
        }
        let mut mtp = vec![0.0f32; (K * V8) as usize];
        for (r, &c) in mtp_rows.iter().enumerate() {
            mtp[r * V8 as usize + c] = 7.0;
        }
        eval(
            &built.program,
            &InputBindings::new(
                &[
                    Value::F32(target),
                    Value::F32(mtp),
                    Value::I32(drafts.to_vec()),
                    Value::I32(vec![0, 1, 2]),
                    Value::F32(vec![0.0, 1.0, 2.0, 3.0]),
                ],
                0,
            ),
        )
        .expect("evals")
    };

    // Partial-accept: 2 hits, miss at 2 -> n_acc=2 -> seed = picked[2] = 2.
    let out = eval_case(&[3, 5, 2, 4], &[3, 5, 6], &[1, 4, 0]);
    assert_eq!(out[0], Value::I32(vec![3, 5, 2, -1]), "commit unchanged vs tail contract");
    assert_eq!(out[1], Value::I32(vec![1, 4, 0]), "drafts unchanged vs tail contract");
    assert_eq!(out[2], Value::I32(vec![2]), "seed = picked[n_acc=2] = the bonus");

    // FULL-accept edge: all 3 drafts hit -> n_acc=3 -> seed = picked[3] = 4
    // (the bonus rides lane k). This is the case a naive `argmax(1-cumprod)`
    // recovery would mis-fire on (all-zero -> lane 0); the one-hot argmax is
    // exact here.
    let out = eval_case(&[3, 5, 6, 4], &[3, 5, 6], &[1, 4, 0]);
    assert_eq!(out[0], Value::I32(vec![3, 5, 6, 4]), "full-accept: whole tail committed");
    assert_eq!(out[2], Value::I32(vec![4]), "full-accept: seed = picked[k] = bonus @ lane k");

    // FULL-reject edge: draft 0 misses -> n_acc=0 -> seed = picked[0] = 3
    // (a full-reject step still advances by 1 via the bonus).
    let out = eval_case(&[3, 5, 2, 4], &[9, 5, 6], &[1, 4, 0]);
    assert_eq!(out[0], Value::I32(vec![3, -1, -1, -1]), "full-reject: bonus only");
    assert_eq!(out[2], Value::I32(vec![3]), "full-reject: seed = picked[0] = bonus @ lane 0");
}

#[test]
fn mtp_specdecode_device_binds_mtpdrafts_and_validates() {
    // Device-resident MTP swap: drafts read from the retained buffer via the
    // MtpDrafts intrinsic (I32 [k]) instead of a host submit. Bindings =
    // [Logits, MtpLogits, MtpDrafts]; outputs = [commit, drafts, seed]; no host
    // `draft` key (only the 2 constant lane vectors).
    const K: u32 = 3;
    const V8: u32 = 8;
    let (built, keys) = program::mtp_specdecode_device(V8, K).expect("builds");
    built.program.validate().expect("validates");
    assert_eq!(built.outputs, vec![OutputKind::Token, OutputKind::Token, OutputKind::Token]);
    assert_eq!(
        built.bindings[..3],
        [Binding::Logits, Binding::MtpLogits, Binding::MtpDrafts],
        "intrinsics in declaration order (target, mtp-logits, mtp-drafts)"
    );
    // The MtpDrafts intrinsic decl is I32 [k] (the shape contract echo pinned).
    let drafts_decl = built
        .program
        .inputs
        .iter()
        .zip(&built.bindings)
        .find(|(_, b)| **b == Binding::MtpDrafts)
        .map(|(d, _)| d)
        .expect("mtp-drafts input present");
    assert_eq!(drafts_decl.shape.dims(), &[K]);
    assert_eq!(drafts_decl.dtype, ir::DType::I32);
    assert!(ir::validate::intrinsic_decl_ok(&Binding::MtpDrafts, drafts_decl, Some(V8)));
    let _ = keys;
}

#[test]
fn mtp_specdecode_bootstrap_builds() {
    const K: u32 = 4;
    const V: u32 = 16;
    let built = program::mtp_specdecode_bootstrap(V, K).expect("builds");
    built.program.validate().expect("validates");
    assert_eq!(built.outputs, vec![OutputKind::Token, OutputKind::Token, OutputKind::Token]);
    assert_eq!(built.bindings[..2], [Binding::Logits, Binding::MtpLogits], "no verify draft");
}
