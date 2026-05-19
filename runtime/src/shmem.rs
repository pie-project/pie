//! POSIX shmem region naming for pie's per-DP IPC regions.
//!
//! Single source of truth shared by `pie::device` (runtime side, attach)
//! and `pie-server`'s `embedded_driver` (server side, create). The two
//! sides must agree on the region name byte-for-byte or the runtime
//! attaches to a region the server never created.
//!
//! Region layout: `<base>_g{group_id}`. Base defaults to `/pie_shmem`;
//! `$PIE_SHMEM_NAME` overrides so a launcher running multiple `pie serve`
//! processes on the same host (POSIX shmem is host-global) can give each
//! process disjoint regions.
//!
//! Validation is enforced at the first `region_name(...)` call — both
//! callers run inside `bootstrap_inner`, well before any user-visible
//! work, so a clear panic here is the right place to surface a bad env.

/// Maximum POSIX shmem name length on macOS (`PSHMNAMLEN`). Linux is
/// laxer (`NAME_MAX = 255`), but the override is supposed to work on
/// both, so we validate against the tighter limit.
const PSHMNAMLEN: usize = 31;

/// Suffix budget reserved for `_g{group_id}`. `_g99` is 4 bytes; we
/// leave 1 byte of slack against a future DP-100+ topology, so the base
/// is capped at `PSHMNAMLEN - 5 = 26`.
const SUFFIX_BUDGET: usize = 5;

/// Per-DP shmem region name, honoring `$PIE_SHMEM_NAME` as the base.
/// Panics with a clear diagnostic if the env value is malformed —
/// `shm_open` would otherwise fail with an opaque `ENAMETOOLONG` /
/// `EINVAL` / `NulError` deep inside driver boot.
///
/// Validation order:
///   1. base-only checks (`validate_base`): leading `/`, no NUL,
///      base length ≤ `PSHMNAMLEN - SUFFIX_BUDGET` (heuristic budget
///      assuming `group_id < 100`).
///   2. assembled-name check (`validate_assembled`): the final
///      `<base>_g{group_id}` length ≤ `PSHMNAMLEN`. Catches the
///      DP ≥ 100 case at a cap-length base where step (1) passes
///      but the formatted string still overruns.
pub fn region_name(group_id: usize) -> String {
    let base = std::env::var("PIE_SHMEM_NAME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "/pie_shmem".to_string());
    validate_base(&base);
    let name = format!("{base}_g{group_id}");
    validate_assembled(&name, &base, group_id);
    name
}

// `pub(crate)` so unit tests can probe NUL handling without going
// through `std::env::set_var`, which itself panics on NUL bytes (and
// would poison `ENV_LOCK` since the panic fires outside `catch_unwind`).
pub(crate) fn validate_base(base: &str) {
    assert!(
        base.starts_with('/'),
        "PIE_SHMEM_NAME must start with '/' (got {base:?}); POSIX shm_open \
         rejects non-absolute region names"
    );
    assert!(
        !base.contains('\0'),
        "PIE_SHMEM_NAME must not contain NUL bytes (got {base:?}); embedded \
         NULs trip CString::new with NulError and truncate at the C boundary"
    );
    let max_base = PSHMNAMLEN - SUFFIX_BUDGET;
    assert!(
        base.len() <= max_base,
        "PIE_SHMEM_NAME base too long ({} chars, max {}); the appended \
         `_g{{group_id}}` suffix would exceed macOS PSHMNAMLEN={}. \
         got {base:?}",
        base.len(),
        max_base,
        PSHMNAMLEN,
    );
}

/// Belt-and-suspenders length check on the assembled region name.
/// `validate_base` reserves `SUFFIX_BUDGET` bytes for `_g{0..99}`, but
/// `group_id: usize` is unbounded — a future DP-100+ topology with the
/// base at the cap would format to a 32+ byte name that `shm_open`
/// rejects with opaque `ENAMETOOLONG`. Catch it here with a named
/// diagnostic instead.
fn validate_assembled(name: &str, base: &str, group_id: usize) {
    assert!(
        name.len() <= PSHMNAMLEN,
        "assembled shmem region name {name:?} is {} chars, exceeds macOS \
         PSHMNAMLEN={}. base={base:?} group_id={group_id}. Shorten \
         $PIE_SHMEM_NAME or reduce DP fanout.",
        name.len(),
        PSHMNAMLEN,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // `std::env` is process-global; serialize the env-mutating tests so
    // they don't race when `cargo test` runs them in parallel.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce() -> R + std::panic::UnwindSafe, R>(
        value: Option<&str>,
        body: F,
    ) -> std::thread::Result<R> {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var("PIE_SHMEM_NAME").ok();
        // SAFETY: tests serialize via ENV_LOCK; no other thread reads PIE_SHMEM_NAME here.
        unsafe {
            match value {
                Some(v) => std::env::set_var("PIE_SHMEM_NAME", v),
                None => std::env::remove_var("PIE_SHMEM_NAME"),
            }
        }
        let out = std::panic::catch_unwind(body);
        // SAFETY: restore prior value under the same lock.
        unsafe {
            match prev {
                Some(p) => std::env::set_var("PIE_SHMEM_NAME", p),
                None => std::env::remove_var("PIE_SHMEM_NAME"),
            }
        }
        out
    }

    #[test]
    fn unset_returns_default_base() {
        let got = with_env(None, || region_name(0)).unwrap();
        assert_eq!(got, "/pie_shmem_g0");
    }

    #[test]
    fn empty_falls_back_to_default() {
        let got = with_env(Some(""), || region_name(3)).unwrap();
        assert_eq!(got, "/pie_shmem_g3");
    }

    #[test]
    fn whitespace_only_falls_back_to_default() {
        let got = with_env(Some("   "), || region_name(0)).unwrap();
        assert_eq!(got, "/pie_shmem_g0");
    }

    #[test]
    fn valid_override_is_appended_with_group_suffix() {
        let got = with_env(Some("/pie_t_12345_abcdef"), || region_name(2)).unwrap();
        assert_eq!(got, "/pie_t_12345_abcdef_g2");
    }

    #[test]
    fn missing_leading_slash_panics() {
        let err = with_env(Some("pie_shmem"), || region_name(0)).unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&'static str>().map(|s| (*s).to_string()))
            .unwrap_or_default();
        assert!(msg.contains("must start with '/'"), "got: {msg}");
    }

    #[test]
    fn over_length_panics() {
        // Base length = 27 chars (1 char above the 26-char cap).
        let too_long = format!("/{}", "x".repeat(26));
        assert_eq!(too_long.len(), 27);
        let err = with_env(Some(&too_long), || region_name(0)).unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&'static str>().map(|s| (*s).to_string()))
            .unwrap_or_default();
        assert!(msg.contains("too long"), "got: {msg}");
    }

    #[test]
    fn at_cap_succeeds() {
        // Base length = 26 chars (exactly at the cap).
        let at_cap = format!("/{}", "x".repeat(25));
        assert_eq!(at_cap.len(), 26);
        let got = with_env(Some(&at_cap), || region_name(99)).unwrap();
        assert_eq!(got.len(), 26 + 4);
        assert!(got.ends_with("_g99"));
    }

    #[test]
    fn embedded_nul_panics() {
        // Cannot route through `with_env` here: `std::env::set_var` itself
        // panics on NUL, and that panic fires outside `catch_unwind`
        // — it would poison ENV_LOCK and break every later test in the
        // same module. Probe `validate_base` directly instead.
        let err = std::panic::catch_unwind(|| validate_base("/pie\0evil")).unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&'static str>().map(|s| (*s).to_string()))
            .unwrap_or_default();
        assert!(msg.contains("NUL"), "got: {msg}");
    }

    #[test]
    fn dp_100_at_cap_panics_assembled_check() {
        // Base passes `validate_base` (26-char cap), but `_g100` (5 bytes)
        // makes the assembled name 31 chars: still ok in this case, so
        // step up to `group_id = 1000` where `_g1000` (6 bytes) pushes
        // the assembled length to 32 = above PSHMNAMLEN.
        let at_cap = format!("/{}", "x".repeat(25));
        assert_eq!(at_cap.len(), 26);
        let err = with_env(Some(&at_cap), || region_name(1000)).unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&'static str>().map(|s| (*s).to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("exceeds macOS PSHMNAMLEN"),
            "got: {msg}",
        );
    }

    #[test]
    fn dp_100_within_budget_succeeds() {
        // Base length 25 leaves 6 bytes for the suffix; `_g100` fits.
        let base = format!("/{}", "x".repeat(24));
        assert_eq!(base.len(), 25);
        let got = with_env(Some(&base), || region_name(100)).unwrap();
        assert_eq!(got.len(), 25 + 5);
        assert!(got.ends_with("_g100"));
    }
}
