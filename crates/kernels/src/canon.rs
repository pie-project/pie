//! The canonical op registry — tier-1's closed role list.
//!
//! A ROLE is a backend-agnostic operation the model DSL may state without
//! naming a backend symbol: `dsl::rmsnorm(..)` resolves through this list to
//! whichever routine of the family's backend CLAIMS the role. A routine
//! claims one with `#[routine(canon = rmsnorm)]`, which lands on
//! [`crate::routine::Routine::canon`] — so tier membership is the routine's
//! own declaration, in the crate that is the single source of truth for the
//! operation, and never a table in the DSL.
//!
//! Tier-2 is the absence of a claim: a routine with `canon: None` is reached
//! only through its own symbol (`dsl::cuda::fire::<R>`), never through a
//! role.
//!
//! # Why strings and not an enum
//!
//! The claim crosses `#[routine]`, the row, the DSL's resolution and the
//! engine's site derivation. An enum would put the vocabulary in four
//! signatures; a `&'static str` checked against [`ROLES`] keeps it in one
//! list with one membership test, which is the same trade `GuardPred`'s wire
//! kinds make.
//!
//! # Axes ride the role
//!
//! `NormVariant` (plain vs Gemma's `(1 + w)`) is not two roles: it is one
//! role, and the VARIANT picks between the claiming routines the way dtype
//! already does. A claim may therefore be shared by several routines of one
//! backend; what must be unique per backend is the (role, axes) point, and
//! the test that holds it lives with the tables (B9).

/// Every role a trace may state without a backend symbol.
///
/// Closed: a name not in this list is refused where the claim is read
/// ([`is_role`]), so a typo in an attribute is a build-time refusal at the
/// registry rather than an unreachable routine.
pub const ROLES: &[&str] = &[
    // GEMM family. `matmul` may be answered by a driver op (cuBLAS) — a
    // claim does not promise a column, it promises an answer.
    "matmul",
    "matmul_select",
    "gemv",
    // Norms.
    "rmsnorm",
    "rmsnorm_per_head",
    "rmsnorm_gated",
    "add_bias",
    "residual_add",
    // Attention.
    "attention",
    "kv_append",
    "split_qkv",
    // Rope.
    "rope",
    "rope_partial",
    // MLP.
    "swiglu",
    // Embedding and head.
    "embed",
    "lm_head",
    // MoE.
    "topk",
    "weighted_sum",
    // Gating and splits.
    "sigmoid_gate_add",
    "sigmoid_gate_mul",
    "split_gdn",
    "split_q_gate",
    // Recurrent.
    "causal_conv1d",
    "gdn_prep",
    "gated_delta",
];

/// Whether `name` is a role this registry closes over.
#[must_use]
pub const fn is_role(name: &str) -> bool {
    let mut i = 0;
    while i < ROLES.len() {
        let (a, b) = (ROLES[i].as_bytes(), name.as_bytes());
        if a.len() == b.len() {
            let mut j = 0;
            let mut eq = true;
            while j < a.len() {
                if a[j] != b[j] {
                    eq = false;
                    break;
                }
                j += 1;
            }
            if eq {
                return true;
            }
        }
        i += 1;
    }
    false
}
