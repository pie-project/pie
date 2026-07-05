//! Lane D (RS working set, Phase 3) — fold-granularity contract lock.
//!
//! A fast, lib-independent unit test of the `fold(n)` validation rule (design
//! note `workingset-rs-design` §3.1). The full Phase-3 gate (fresh state, lazy
//! fork, first-mutation CoW, fold success/error, no-rollback, W10 buffered
//! write, …) is realized as 17 real tests in `runtime/src/working_set/rs.rs`
//! against the integrated arena — see the note at the bottom of this file.
//!
//! Design + recon: Source note `workingset-rs-design`.

// =============================================================================
// Fold-granularity validation — pure logic, runnable now.
//
// Mirror of the rule the runtime `fold(n)` path will enforce BEFORE any driver
// submission. The production implementation must match this exactly; keeping a
// standalone copy lets us pin the semantics in Wave 0 without the host resource.
// =============================================================================

/// Why a `fold(n)` request was rejected. Maps 1:1 to the runtime errors the
/// real resource will return (as `result<_, error>` across the WIT boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FoldError {
    /// `n == 0`.
    ZeroTokens,
    /// `n` exceeds the buffered (un-folded) token count.
    ExceedsBuffered,
    /// `n` is not a whole multiple of the model fold granularity (`g > 1`).
    Granularity,
}

/// Validate an explicit `fold(n)` request. `n` = tokens to fold, `buffered` =
/// current buffered token count, `granularity` = model token fold granularity
/// (>= 1; 1 = token-causal recurrences such as GDN / Mamba2, i.e. any length).
fn validate_fold(n: u32, buffered: u32, granularity: u32) -> Result<(), FoldError> {
    debug_assert!(granularity >= 1, "granularity must be >= 1");
    if n == 0 {
        return Err(FoldError::ZeroTokens);
    }
    if n > buffered {
        return Err(FoldError::ExceedsBuffered);
    }
    if granularity > 1 && n % granularity != 0 {
        return Err(FoldError::Granularity);
    }
    Ok(())
}

#[test]
fn fold_rejects_zero_tokens() {
    assert_eq!(validate_fold(0, 8, 1), Err(FoldError::ZeroTokens));
    assert_eq!(validate_fold(0, 0, 4), Err(FoldError::ZeroTokens));
}

#[test]
fn fold_rejects_more_than_buffered() {
    assert_eq!(validate_fold(9, 8, 1), Err(FoldError::ExceedsBuffered));
    // Exact-buffered is allowed; one past it is not.
    assert_eq!(validate_fold(8, 8, 1), Ok(()));
    assert_eq!(validate_fold(8, 7, 1), Err(FoldError::ExceedsBuffered));
}

#[test]
fn fold_granularity_one_accepts_any_length() {
    // Token-causal models (Qwen3.5 GDN, Nemotron-H Mamba2) report g = 1.
    for n in 1..=16u32 {
        assert_eq!(validate_fold(n, 16, 1), Ok(()), "n={n} must fold at g=1");
    }
}

#[test]
fn fold_granularity_gt_one_requires_multiple() {
    // Forward-looking: block-recurrent models with g > 1.
    let g = 4;
    assert_eq!(validate_fold(4, 16, g), Ok(()));
    assert_eq!(validate_fold(8, 16, g), Ok(()));
    assert_eq!(validate_fold(5, 16, g), Err(FoldError::Granularity));
    assert_eq!(validate_fold(6, 16, g), Err(FoldError::Granularity));
    // Granularity is only checked after the bounds checks.
    assert_eq!(validate_fold(20, 16, g), Err(FoldError::ExceedsBuffered));
}

// =============================================================================
// `model.is-linear()` derivation — the linear/attention class gate.
//
// Mirror of the host rule (`runtime/src/api/model.rs`: `is_linear` =
// `rs_caps().state_size > 0`). The runtime keys the speculative-commit strategy
// on this — FOLD-COMMIT (fold only the accepted prefix into the recurrent state)
// for linear/recurrent models vs KV-slot discard for attention — so the SENSE
// must never flip. TRUE iff the model carries a folded recurrent state (a
// non-zero folded-state byte size), matching the CUDA executor's `use_slots`
// gate. Pins the class contract independent of the byte accounting.
// =============================================================================

/// The `model.is-linear()` predicate: TRUE iff the model has a folded recurrent
/// state. `state_size` = bytes of one folded RS object (`rs-state-size`).
fn is_linear(state_size: u64) -> bool {
    state_size > 0
}

#[test]
fn is_linear_true_iff_recurrent_state_present() {
    // Pure attention (RS caps 0/0/1 per bootstrap.rs) → NOT linear: commit via
    // KV-slot discard, never a fold.
    assert!(
        !is_linear(0),
        "state_size 0 (pure attention) must be non-linear"
    );
    // Any non-zero folded-state size (GDN / Mamba2 recurrent state) → linear:
    // commit via fold-commit (`rs-fold-buffered`).
    assert!(is_linear(4096), "a folded recurrent state must be linear");
    assert!(
        is_linear(1),
        "even a 1-byte folded state counts as recurrent"
    );
}

// =============================================================================
// Phase-3 gate integration coverage.
//
// The 7 Wave-0 `#[ignore]`d integration stubs that previously lived here are now
// fully realized — as 17 real gate tests in `runtime/src/working_set/rs.rs`
// (`#[cfg(test)] mod tests`), exercising the real `RsWorkingSet` core against
// bravo's arena: fresh state, alloc/free/reorder buffer, lazy fork (no copy),
// first-fold-after-fork CoW, read-only-fork-never-copies, no-rollback (pre-fork
// snapshot independence), fold success/error + granularity, fold abort, W10
// `resolve_buffer`/`cow_write_buffer` (reserved→materialized + shared CoW),
// release, and driver binding. Host-level e2e (the WIT `rs-working-set` resource
// + `inference.fold` through a wasm instance) is deferred to the Phase-6/7
// inferlet harness. The standalone validation tests above remain as a fast,
// lib-independent lock on the fold-granularity contract (mirrors §3.1).
// =============================================================================
