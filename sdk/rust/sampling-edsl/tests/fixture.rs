//! #12 cross-language byte-parity fixture (`standard_program_bytecode.txt`).
//!
//! SINGLE SOURCE OF TRUTH for the standard-sampler bytecode, emitted by foxtrot's
//! `standard_programs` (all 6 kinds, exact form — k-invariant since #25, where
//! top-k `k` is a submit value-id, not a baked immediate). Read at TEST RUNTIME (a
//! fixture, not a
//! build dep) by BOTH sides so the SDK emit and the driver bake are provably
//! byte-identical:
//!   - Rust SDK leg (this test): live emit == committed fixture.
//!   - driver C++ leg (delta): `standard_sampler_program(k,V)` bytes == row, and
//!     `recognize(fnv1a64(row)) == kind`.
//!
//! Byte-diffable on divergence (golf's rationale): a row mismatch localizes the
//! diverging byte, not just "the hashes differ".
//!
//! Regenerate after an intentional emit change:
//!   STDPROG_FIXTURE_REGEN=1 cargo test -p sampling-edsl --test fixture

use pie_sampling_ir::program_hash;
use sampling_edsl::{CanonicalKind, standard_programs};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../interface/sampling-ir/tests/standard_program_bytecode.txt"
);

/// Vocab sizes: qwen3-0.6b (the HW-verify model — the byte-parity that matters on
/// hardware) + a tiny vocab for fast cross-checks.
const VOCABS: &[u32] = &[151936, 128];

fn kind_name(k: CanonicalKind) -> &'static str {
    match k {
        CanonicalKind::Argmax => "argmax",
        CanonicalKind::Temperature => "temperature",
        CanonicalKind::MinP => "min_p",
        CanonicalKind::TopK => "top_k",
        CanonicalKind::TopP => "top_p",
        CanonicalKind::TopKTopP => "top_k_top_p",
        CanonicalKind::Custom => "custom",
    }
}

fn row(kind: CanonicalKind, vocab: u32, bytecode: &[u8]) -> String {
    let hex: String = bytecode.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{},{},{:016x},{}\n",
        kind_name(kind),
        vocab,
        program_hash(bytecode),
        hex
    )
}

fn emit() -> String {
    let mut s = String::new();
    s.push_str("# #12 standard-sampler bytecode fixture — SINGLE SOURCE for the\n");
    s.push_str("# cross-language byte-parity guard. Emitted by foxtrot's SDK:\n");
    s.push_str("#   standard_programs(V) -> all 6 standard kinds (k-invariant since #25:\n");
    s.push_str("#                           top-k k is a submit value-id, not a baked immediate)\n");
    s.push_str("#\n");
    s.push_str("# Format: kind,vocab,fnv1a64,hex\n");
    s.push_str("#   fnv1a64 = pie_sampling_ir::program_hash(bytecode) == driver jit::fnv1a64.\n");
    s.push_str("# Recognizer use: every kind (incl. top_k / top_k_top_p) matches by EXACT hash;\n");
    s.push_str("# the driver reads k from the submit binding (no op-shape canonicalization).\n");
    s.push_str("# Regenerate: STDPROG_FIXTURE_REGEN=1 cargo test -p sampling-edsl --test fixture\n");
    s.push_str("kind,vocab,fnv1a64,hex\n");
    for &vocab in VOCABS {
        for (bytecode, kind) in standard_programs(vocab).unwrap() {
            s.push_str(&row(kind, vocab, &bytecode));
        }
    }
    s
}

#[test]
fn standard_program_bytecode_fixture_matches_live_emit() {
    let live = emit();
    if std::env::var_os("STDPROG_FIXTURE_REGEN").is_some() {
        std::fs::write(FIXTURE, &live).expect("write fixture");
        eprintln!("regenerated {FIXTURE}");
        return;
    }
    let committed = std::fs::read_to_string(FIXTURE).expect("read committed fixture");
    assert_eq!(
        live, committed,
        "live SDK emit drifted from the committed fixture — \
         regenerate with STDPROG_FIXTURE_REGEN=1 and review the byte diff"
    );
}
