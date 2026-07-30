//! Golden traced forms.
//!
//! The unit tests argue that individual ops are right; this one checks the
//! thing a driver actually receives: the whole traced form, byte for byte,
//! for the family configurations the executor runs. Its value is not that
//! any particular op is *right* — it is that a change to the traced form
//! cannot happen quietly. The executor's op→kernel mapping is a
//! deterministic walk of this structure, so pinning the structure pins the
//! emitted kernel sequence too: this is the CI-enforceable half of the
//! parity story, the half that needs no GPU
//! (`driver/cuda/src/model/llama_like/declared_forward.cpp` is the other).
//!
//! Regenerate after an intended change:
//!
//! ```text
//! UPDATE_GOLDEN=1 cargo test -p pie-forward --test golden_plans
//! ```
//!
//! The pattern is `loader/tests/golden_plans.rs`, including the rule that a
//! regenerated golden is a diff a human reads and approves.

use std::path::PathBuf;

use pie_forward::family::llama_like;
use pie_forward::LlamaLikeFacts;

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.json"))
}

fn check(name: &str, facts: &LlamaLikeFacts) {
    let plan = llama_like(facts);
    let fresh = serde_json::to_string_pretty(&plan).expect("serialize plan");
    let path = golden_path(name);
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir golden");
        std::fs::write(&path, &fresh).expect("write golden");
        return;
    }
    let stored = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "no golden at {}: {err}.\n\
             If this plan is new, regenerate with UPDATE_GOLDEN=1.",
            path.display()
        )
    });
    assert_eq!(
        stored, fresh,
        "traced form for `{name}` changed.\n\
         If the change is intended, regenerate with UPDATE_GOLDEN=1 and \
         review the diff — the executor emits kernels by walking exactly \
         this structure."
    );
}

/// The parity model: what `declared_forward.cpp` ran token-identical to the
/// hand-written pass (stage3 commit trail).
#[test]
fn qwen3_0_6b() {
    check("qwen3_0_6b", &LlamaLikeFacts::qwen3_0_6b());
}

/// The second declared configuration (stage 3 rung d): no qk-norm — the
/// RmsnormPerHead pair vanishes from every layer — an untied lm_head, and
/// the unfused QKV binding (the contract splits the checkpoint's fused
/// qkv_proj and the dense join cannot re-fuse the bands). The 96 → 128
/// head-dim pad, the 2047 sliding window and the null rope scaling are
/// backend cfg, invisible here by design.
#[test]
fn phi3_mini() {
    check("phi3_mini", &LlamaLikeFacts::phi3_mini());
}

/// The unfused-binding variant: three projection matmuls, no SplitQkv. Kept
/// golden so the binding-driven divergence stays a reviewed artifact rather
/// than an emergent one.
#[test]
fn qwen3_0_6b_unfused_qkv() {
    check(
        "qwen3_0_6b_unfused_qkv",
        &LlamaLikeFacts {
            fused_qkv: false,
            ..LlamaLikeFacts::qwen3_0_6b()
        },
    );
}
