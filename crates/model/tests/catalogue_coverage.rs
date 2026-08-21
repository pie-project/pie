//! Every catalogued SKU loads — the demand set closes by CHECK.
//!
//! `rustc` does not instantiate kernels through a forward (the trace
//! decouples them), so enumerating SKUs in a family's `CATALOG` closes
//! nothing by itself. THIS is the closure: trace every row at both fire
//! classes, and `TraceBuilder::finish`'s `check_plan` refuses a point
//! whose statements reach a routine row that does not exist — a missing
//! `#[routine]` point fails the build here, never a fire.

use model::gemma_4::forward::CATALOG;
use model::gemma_4::forward::facts::{Gemma4CudaFacts, Gemma4Facts};
use model_ir::trace::FireClass;

#[test]
fn every_gemma4_sku_traces_at_both_classes() {
    let facts = Gemma4Facts::gemma_4_e4b();
    let cuda = Gemma4CudaFacts::gemma_4_e4b_synthetic();
    assert!(!CATALOG.is_empty(), "gemma-4 catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            // `finish` asserts check_plan internally; reaching here is the pass.
            let plan = trace(&facts, &cuda, class, 1e-6);
            assert!(
                plan.family.starts_with("gemma4-"),
                "`{name}` traced under family `{}` — the SKU must join the \
                 family's first segment",
                plan.family,
            );
            assert!(
                !plan.ops.is_empty(),
                "`{name}` traced an empty text at {class:?}"
            );
        }
    }
}

#[test]
fn the_sku_name_is_the_axes() {
    // One row today; its name spells the instantiation, so a second SKU
    // cannot silently reuse the first's cache key.
    let names: Vec<&str> = CATALOG.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["gemma4-bf16-kv-bf16"]);
}

// ── The rollout: every family with a CUDA forward, same closure ─────────
//
// One section per family (S2c's rollout half). Each traces its CATALOG at
// both fire classes against the family's own synthetic facts fixture —
// the fixtures the golden tests read — and pins that the SKU joined the
// family string's first segment. The name lists are pinned together at
// the end, one place to read what this build ships.

/// The full-family check every section below runs: non-empty catalogue,
/// every row traced at both classes (tracing runs `check_plan` inside
/// `finish`, so reaching the assertions IS the pass), the SKU in the
/// first segment, a non-empty text.
fn check_traced(name: &str, class: FireClass, plan: &model_ir::trace::ForwardPlan) {
    assert_eq!(
        plan.family,
        format!("{name}.cuda.{}", class.suffix()),
        "`{name}` must join the family's first segment and keep the \
         backend in the second",
    );
    assert!(
        !plan.ops.is_empty(),
        "`{name}` traced an empty text at {class:?}"
    );
}

#[test]
fn every_gemma2_sku_traces_at_both_classes() {
    use model::gemma_2::forward::CATALOG;
    use model::gemma_2::forward::facts::Gemma2Facts;
    let facts = Gemma2Facts::gemma_2_9b();
    assert!(!CATALOG.is_empty(), "gemma-2 catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(name, class, &trace(&facts, class, 1e-6, 10_000.0));
        }
    }
}

#[test]
fn every_gemma3n_sku_traces_at_both_classes() {
    use model::gemma_3n::forward::CATALOG;
    use model::gemma_3n::forward::facts::Gemma3nFacts;
    let facts = Gemma3nFacts::gemma3n_synthetic();
    assert!(!CATALOG.is_empty(), "gemma-3n catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(name, class, &trace(&facts, class, 1e-6, 1_000_000.0, 10_000.0));
        }
    }
}

#[test]
fn every_glm5_sku_traces_at_both_classes() {
    use model::glm_5::forward::CATALOG;
    use model::glm_5::forward::facts::Glm5Facts;
    let facts = Glm5Facts::glm5_106b_a12b();
    assert!(!CATALOG.is_empty(), "glm-5 catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(name, class, &trace(&facts, class, 1e-6, 10_000.0));
        }
    }
}

#[test]
fn every_gpt_oss_sku_traces_at_both_classes() {
    use model::gpt_oss::forward::CATALOG;
    use model::gpt_oss::forward::facts::{GptOssCudaFacts, GptOssFacts};
    let facts = GptOssFacts::gpt_oss_20b();
    let cuda = GptOssCudaFacts::gpt_oss_20b_synthetic();
    assert!(!CATALOG.is_empty(), "gpt-oss catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(
                name,
                class,
                &trace(&facts, &cuda, class, 1e-6, 150_000.0, 128),
            );
        }
    }
}

#[test]
fn every_deepseek_v4_sku_traces_at_both_classes() {
    use model::deepseek_v4::forward::CATALOG;
    use model::deepseek_v4::forward::facts::Dsv4Facts;
    let facts = Dsv4Facts::dsv4_synthetic();
    assert!(!CATALOG.is_empty(), "deepseek-v4 catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(name, class, &trace(&facts, class, 1e-6, 10_000.0));
        }
    }
}

#[test]
fn every_kimi_k2_sku_traces_at_both_classes() {
    use model::kimi_k2::forward::CATALOG;
    use model::kimi_k2::forward::facts::{KimiCudaFacts, KimiFacts};
    let facts = KimiFacts::kimi_k2();
    let cuda = KimiCudaFacts::kimi_k2_synthetic();
    assert!(!CATALOG.is_empty(), "kimi-k2 catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(name, class, &trace(&facts, &cuda, class, 1e-6));
        }
    }
}

#[test]
fn every_kimi_k3_sku_traces_at_both_classes() {
    use model::kimi_k3::forward::CATALOG;
    use model::kimi_k3::forward::facts::KimiK3Facts;
    let facts = KimiK3Facts::kimi_k3_synthetic();
    assert!(!CATALOG.is_empty(), "kimi-k3 catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(name, class, &trace(&facts, class, 1e-6));
        }
    }
}

#[test]
fn every_nemotron_h_sku_traces_at_both_classes() {
    use model::nemotron_h::forward::CATALOG;
    use model::nemotron_h::forward::facts::NemotronHFacts;
    let facts = NemotronHFacts::nemotron_h_synthetic();
    assert!(!CATALOG.is_empty(), "nemotron-h catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(name, class, &trace(&facts, class, 1e-6, 10_000.0));
        }
    }
}

#[test]
fn every_qwen3_5_sku_traces_at_both_classes() {
    use model::qwen_3_5::forward::CATALOG;
    use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let cuda = Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
    assert!(!CATALOG.is_empty(), "qwen-3.5 catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(name, class, &trace(&facts, &cuda, class, 1e-6, 10_000.0));
        }
    }
}

#[test]
fn every_llama_like_sku_traces_at_both_classes() {
    use model::shared::llama_like::forward::CATALOG;
    use model::shared::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    let facts = LlamaLikeFacts::qwen3_0_6b();
    let cuda = LlamaLikeCudaFacts::qwen3_0_6b_l40s();
    assert!(!CATALOG.is_empty(), "llama_like catalogues nothing");
    for (name, trace) in CATALOG {
        for class in [FireClass::Decode, FireClass::Prefill] {
            check_traced(name, class, &trace(&facts, &cuda, class, 1e-6, 10_000.0));
        }
    }
}

#[test]
fn every_sku_name_spells_its_instantiation() {
    // The rollout's names, pinned in one place: each spells its family's
    // instantiated axes (the pinned A omitted, gemma-4's precedent), so a
    // second SKU anywhere cannot silently reuse the first's cache key.
    // llama_like's spells only the axes its rows share — the weight repr
    // is per-row DATA there, deliberately (see its `CATALOG` doc).
    fn one<'a, T>(c: &'a [(&'a str, T)]) -> Vec<&'a str> {
        c.iter().map(|(n, _)| *n).collect()
    }
    assert_eq!(one(model::gemma_2::forward::CATALOG), ["gemma_2-bf16-kv-bf16"]);
    assert_eq!(one(model::gemma_3n::forward::CATALOG), ["gemma3n-bf16-kv-bf16"]);
    assert_eq!(one(model::glm_5::forward::CATALOG), ["glm5-bf16-bf16-kv-bf16"]);
    assert_eq!(
        one(model::gpt_oss::forward::CATALOG),
        ["gpt_oss-bf16-mxfp4-kv-bf16"]
    );
    assert_eq!(
        one(model::deepseek_v4::forward::CATALOG),
        ["deepseek_v4-bf16-bf16-kv-bf16"]
    );
    assert_eq!(
        one(model::kimi_k2::forward::CATALOG),
        ["kimi-bf16-wna16-kv-bf16"]
    );
    assert_eq!(
        one(model::kimi_k3::forward::CATALOG),
        ["kimi_k3-bf16-mxfp4-kv-bf16"]
    );
    assert_eq!(
        one(model::nemotron_h::forward::CATALOG),
        ["nemotron_h-bf16-bf16-kv-bf16"]
    );
    assert_eq!(
        one(model::qwen_3_5::forward::CATALOG),
        ["qwen3_5_hybrid-bf16-bf16-kv-bf16"]
    );
    assert_eq!(
        one(model::shared::llama_like::forward::CATALOG),
        ["llama_like-kv-bf16"]
    );
}
