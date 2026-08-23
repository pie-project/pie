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
    "lm_head",
    // Norms.
    "rmsnorm",
    "rmsnorm_gated",
    "add_bias",
    "residual_add",
    "mul_scalar",
    "logit_softcap",
    // Attention. The point after the `.` names the fire shape
    // (`attention.decode`, `attention.prefill_lse`) — see `is_role`.
    "attention",
    "attention_landing",
    "kv_append",
    "split_qkv",
    "rope",
    // MLP activations; the bare spelling takes the packed `[gate | up]` row.
    "swiglu",
    "situ",
    // Embedding.
    "embed",
    // MoE.
    "topk",
    "weighted_sum",
    // Gating and splits.
    "sigmoid_gate_add",
    "sigmoid_gate_mul",
    "split_rows",
    "split_q_gate",
    // Recurrent.
    "causal_conv1d",
    "gdn_prep",
    "gated_delta",
    // MLA.
    "mla_latents",
    "mla_absorb",
    "split_q_b",
    // The DSA index path.
    "index_layernorm_rope",
    "index_rope",
    // Pooled attention (dsv4).
    "pool_boundary",
    "pool_gather",
    "lse_ln",
    // Hyper-connections (dsv4).
    "hc_expand",
    "hc_rmsnorm_f32",
    "hc_gates",
    "hc_fold",
    "hc_collapse",
    // Residual blending (kimi).
    "res_blend",
    // Collectives.
    "all_reduce",
];

/// Roles that are another role wearing a purpose: a plane with no claim of
/// its own answers with the target's. `lm_head` and `attention_landing` ARE
/// matmuls — the role exists so a plane MAY answer specially (a fused
/// decode gemv, a TP landing fused with its all-reduce) and so the driver
/// can find the site; where none does, the delegation is the claim.
pub const DEFAULTS: &[(&str, &str)] = &[("lm_head", "matmul"), ("attention_landing", "matmul")];



/// Whether `claim` names a role this registry closes over.
///
/// A claim may carry an AXIS POINT after the role — `"rmsnorm.gemma"`,
/// `"rope.partial"` — and the role is the part before the first `.`: the
/// axis names a variant of the role, and closing over every point would put
/// each backend's variant spellings in the floor's list.
#[must_use]
pub const fn is_role(claim: &str) -> bool {
    // The role prefix, by byte scan (const context).
    let bytes = claim.as_bytes();
    let mut role_len = bytes.len();
    let mut k = 0;
    while k < bytes.len() {
        if bytes[k] == b'.' {
            role_len = k;
            break;
        }
        k += 1;
    }
    let mut i = 0;
    while i < ROLES.len() {
        let (a, b) = (ROLES[i].as_bytes(), claim.as_bytes());
        if a.len() == role_len {
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
