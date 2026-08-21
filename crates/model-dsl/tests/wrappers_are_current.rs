//! The generated CUDA wrappers are CURRENT, and they round-trip.
//!
//! `wrappers_are_current` re-runs the generator (`tests/generator/mod.rs`) over
//! `crates/kernels-cuda/src` and diffs the result against the committed
//! `src/cuda/generated.rs` — the `model-loader/tests/golden_plans.rs`
//! idiom. A stale file refuses;
//! `UPDATE_WRAPPERS=1 cargo test -p model-dsl --test wrappers_are_current`
//! rewrites it (and, like every golden rewrite, the SAME run still tests
//! the code compiled from the old file — run it once more to prove the
//! new one).
//!
//! The round-trip tests are B4-gen step 5 (design-no-ask §10): one
//! statement traced through the generated fn and through the hand-written
//! wrapper must record IDENTICAL ops — the only difference allowed is the
//! retired restatement at the CALL SITE (`intermediate` / `width`), which
//! the routine's `out(..)` rule now derives.

mod generator;

use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn wrappers_are_current() {
    let kernels = manifest_dir().join("../kernels-cuda/src");
    let want = generator::generate(&kernels);
    let at = manifest_dir().join("src/cuda/generated.rs");
    let have = std::fs::read_to_string(&at).unwrap_or_default();
    if want == have {
        return;
    }
    if std::env::var_os("UPDATE_WRAPPERS").is_some() {
        std::fs::write(&at, &want)
            .unwrap_or_else(|e| panic!("rewriting {}: {e}", at.display()));
        return;
    }
    // Point at the first diverging line rather than dumping two files.
    let line = want
        .lines()
        .zip(have.lines())
        .position(|(w, h)| w != h)
        .map_or_else(
            || want.lines().count().min(have.lines().count()) + 1,
            |i| i + 1,
        );
    panic!(
        "src/cuda/generated.rs is STALE against crates/kernels-cuda/src \
         (first divergence at line {line}). The wrappers are generated, \
         never edited: regenerate with \
         UPDATE_WRAPPERS=1 cargo test -p model-dsl --test wrappers_are_current \
         and review the diff."
    );
}

/// `mlp::chunked_swiglu`, hand vs generated. The hand wrapper takes
/// `intermediate` and states the result; the generated fn derives it from
/// the routine's `out(y = rows(packed) x half(packed))` rule. Same
/// statement either way.
#[test]
fn chunked_swiglu_round_trips() {
    let hand = model_dsl::trace_named("b4gen.cuda.decode", |t| {
        let packed = model_dsl::input(t, 512);
        let _y = model_dsl::cuda::swiglu(&packed, 256);
    });
    let generated = model_dsl::trace_named("b4gen.cuda.decode", |t| {
        let packed = model_dsl::input(t, 512);
        let _y = model_dsl::cuda::generated::chunked_swiglu(&packed, None, None);
    });
    assert_eq!(
        hand, generated,
        "the generated `chunked_swiglu` must record exactly what the hand \
         wrapper records, with the `intermediate` restatement retired"
    );
}

/// `mlp::relu2`, hand vs generated. The hand wrapper takes `width` and
/// states the result; the generated fn derives it from `out(y = like(x))`.
#[test]
fn relu2_round_trips() {
    let hand = model_dsl::trace_named("b4gen.cuda.decode", |t| {
        let x = model_dsl::input(t, 384);
        let _y = model_dsl::cuda::relu2(&x, 384);
    });
    let generated = model_dsl::trace_named("b4gen.cuda.decode", |t| {
        let x = model_dsl::input(t, 384);
        let _y = model_dsl::cuda::generated::relu2(&x, None, None);
    });
    assert_eq!(
        hand, generated,
        "the generated `relu2` must record exactly what the hand wrapper \
         records, with the `width` restatement retired"
    );
}
