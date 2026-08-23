//! `canon::ROLES` closes the claim vocabulary — this closes it the other way.
//!
//! `is_role` refuses a claim from OUTSIDE the list, which is the direction
//! a typo in a `#[routine(canon = ..)]` attribute travels. The direction
//! nothing walked is the converse: a name INSIDE the list that no plane
//! claims — bare or at any point — and no default delegates is a role in
//! name only, and it decays silently because a list entry has no compiler
//! and no caller.
//!
//! # Why this test lives in `model-ir`
//!
//! It needs to see the planes, and `model-ir` is the crate that already
//! does: `Backend` has exactly two arms, and `canon_symbol` reads
//! `kernels_cuda::sigs()` for one and `kernels_metal::declared()` for the
//! other. That is not a partial view of the workspace's four planes — the
//! three SHADER planes are proven claim-for-claim identical by
//! `kernels`'s `the_three_shader_planes_claim_identically`, so Metal
//! stands for all three here and the pair is the whole picture.

use kernels::canon::{DEFAULTS, ROLES};
use model_ir::kernels::{Backend, canon_symbol};

/// The role prefix of a claim — the part before the first `.`.
fn role_of(claim: &str) -> &str {
    claim.split('.').next().unwrap()
}

/// Every claim either plane makes, at whatever point.
fn all_claims() -> Vec<&'static str> {
    let cuda = kernels_cuda::sigs().iter().filter_map(|k| k.canon);
    let metal = kernels_metal::declared().into_iter().filter_map(|d| d.canon);
    cuda.chain(metal).collect()
}

#[test]
fn every_role_is_claimed_or_delegated() {
    let claims = all_claims();
    for role in ROLES {
        let claimed = claims.iter().any(|c| role_of(c) == *role);
        let delegated = DEFAULTS.iter().any(|(r, _)| r == role);
        assert!(
            claimed || delegated,
            "`{role}` is a role no plane claims at any point and no default \
             delegates; a plan stating it lowers to an unresolved backlog row \
             forever, so either a routine should claim it or it should leave \
             `ROLES`",
        );
    }
}

#[test]
fn every_default_lands_on_a_claim() {
    for (role, target) in DEFAULTS {
        assert!(
            ROLES.contains(role) && ROLES.contains(target),
            "the default `{role}` -> `{target}` names a non-role",
        );
        assert!(
            canon_symbol(Backend::Cuda, target).is_some(),
            "the default `{role}` -> `{target}` dangles on the CUDA plane: \
             nothing claims `{target}` bare",
        );
    }
}

/// `canon_symbol` answers by FIRST MATCH over a linear scan of the CUDA
/// signature table, so two routines claiming one point means the second is
/// unreachable and which one answers is table order. The shader planes have
/// this assertion in `kernels`'s `canon_claims_agree`; the CUDA plane, which
/// is the one production resolves against, had it nowhere.
#[test]
fn no_cuda_point_is_claimed_twice() {
    let mut seen = std::collections::BTreeMap::<&str, &str>::new();
    for k in kernels_cuda::sigs() {
        let Some(c) = k.canon else { continue };
        if let Some(prior) = seen.insert(c, k.symbol) {
            panic!(
                "claim `{c}` is made twice on the CUDA plane (by `{prior}` \
                 and `{}`); first match answers, so one is unreachable",
                k.symbol,
            );
        }
    }
    assert!(!seen.is_empty(), "the CUDA plane claims nothing at all");
}

/// Every CUDA claim names a role. The `Routine::canon` builder asserts this
/// at const-eval for routines declared through it, but `sigs()` is the table
/// as `model-ir` reads it and a hand-written row would bypass that.
#[test]
fn every_cuda_claim_names_a_role() {
    for k in kernels_cuda::sigs() {
        if let Some(c) = k.canon {
            assert!(
                kernels::canon::is_role(c),
                "`{}` claims `{c}`, which is not a role in canon::ROLES",
                k.symbol,
            );
        }
    }
}
