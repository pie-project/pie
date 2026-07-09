//! Unit tests for the 4090-pass capability assertions (lane L7 / hotel).
//!
//! Verifies the `sampler_assert` logic against golf's locked JSON schema
//! (`b3e84e92`) WITHOUT the GPU, so the assertion module the stitched 4090
//! harness depends on is itself proven correct. The real on-GPU run feeds live
//! inferlet output through these same functions.

mod common;
use common::sampler_assert::{assert_grammar_conformant, assert_mirostat_converged};

// golf's exact mirostat schema.
const MIROSTAT_OK: &str = r#"{"sampler":"mirostat","count":16,"tau":3,
    "final_mu":12.0223,"mean_surprise":2.3727,"tail_mean_surprise":2.4744,
    "s_flowed":true,"tokens":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}"#;

const GRAMMAR_OK: &str = r#"{"sampler":"grammar","conformant":true,"count":4,
    "tokens":[10,11,12,13]}"#;

#[test]
fn mirostat_converged_passes() {
    // tail_mean_surprise 2.474 vs τ 3.0 → gap 0.526; tol 0.6 → pass.
    assert!(assert_mirostat_converged(MIROSTAT_OK, 0.6).is_ok());
}

#[test]
fn mirostat_tight_tol_fails() {
    let e = assert_mirostat_converged(MIROSTAT_OK, 0.1).unwrap_err();
    assert!(e.contains("did not converge"), "{e}");
}

#[test]
fn mirostat_s_not_flowed_fails() {
    let j = MIROSTAT_OK.replace("\"s_flowed\":true", "\"s_flowed\":false");
    let e = assert_mirostat_converged(&j, 0.6).unwrap_err();
    assert!(e.contains("s_flowed=false"), "{e}");
}

#[test]
fn mirostat_count_mismatch_fails() {
    let j = MIROSTAT_OK.replace("\"count\":16", "\"count\":99");
    assert!(assert_mirostat_converged(&j, 0.6).is_err());
}

#[test]
fn grammar_conformant_passes() {
    assert!(assert_grammar_conformant(GRAMMAR_OK, &[10, 11, 12, 13]).is_ok());
}

#[test]
fn grammar_out_of_alphabet_fails() {
    let j = GRAMMAR_OK.replace("[10,11,12,13]", "[10,11,99,13]");
    let e = assert_grammar_conformant(&j, &[10, 11, 12, 13]).unwrap_err();
    assert!(e.contains("not in allowed alphabet"), "{e}");
}

#[test]
fn grammar_immediate_repeat_fails() {
    let j = GRAMMAR_OK.replace("[10,11,12,13]", "[10,11,11,13]");
    let e = assert_grammar_conformant(&j, &[10, 11, 12, 13]).unwrap_err();
    assert!(e.contains("repeats previous"), "{e}");
}

#[test]
fn grammar_self_report_false_fails() {
    let j = GRAMMAR_OK.replace("\"conformant\":true", "\"conformant\":false");
    assert!(assert_grammar_conformant(&j, &[10, 11, 12, 13]).is_err());
}
