//! Every runtime name a catalogued text mints, this driver answers —
//! `every_plane_is_answered`, reborn on the operand channel.
//!
//! The old test walked the derived column's `Env` half; `ctx.ask` broke
//! that walk ("a call, not a declaration") and its own doc mourned the
//! loss. The no-ask migration made runtime needs OPERANDS, statically
//! enumerable again: `plan.runtime` lists every name a text mints, and
//! [`driver_cuda::bind::views::ANSWERED`] lists every name the driver's
//! resolver answers. A mint outside the union fails HERE, not at a fire.

use std::collections::BTreeSet;

#[test]
fn every_catalogued_runtime_name_is_answered_or_deliberately_unstaged() {
    let answered: BTreeSet<&str> = driver_cuda::bind::views::ANSWERED.iter().copied().collect();
    let unstaged: BTreeSet<&str> = driver_cuda::bind::views::UNSTAGED.iter().copied().collect();

    let mut seen = BTreeSet::new();
    let mut orphans = Vec::new();
    for (sku, plan) in all_catalogued_plans() {
        for b in &plan.runtime {
            seen.insert(b.name.clone());
            if !answered.contains(b.name.as_str()) && !unstaged.contains(b.name.as_str()) {
                orphans.push(format!("{sku}: `{}`", b.name));
            }
        }
    }
    assert!(
        orphans.is_empty(),
        "runtime names nothing answers (and nothing declares unstaged):\n  {}",
        orphans.join("\n  ")
    );
    assert!(
        seen.contains("kv_cache") && seen.contains("positions"),
        "the walk saw no attention text at all — the harness is broken, not the tree"
    );

    // AND THE CONVERSE, which is the half this gate was missing.
    //
    // A name in `ANSWERED` that no catalogued text mints is a claim nothing
    // exercises, and the claim rots the way every unexercised claim in this
    // tree rots: `"fa2.decode"` sat in that list with no arm behind it in
    // `raised()` at all, so the union check above would have waved through
    // the one text that finally minted it and let the fire do the refusing.
    //
    // There is no excuse list here on purpose. The invariant is exact today —
    // every one of the sixteen entries is minted by some catalogued SKU — and
    // an exception list is how it would stop being exact. An arm written
    // ahead of the text that needs it fails here, which is a conversation
    // worth having at the time rather than a year later.
    let unminted: Vec<&str> = driver_cuda::bind::views::ANSWERED
        .iter()
        .copied()
        .filter(|n| !seen.contains(*n))
        .collect();
    assert!(
        unminted.is_empty(),
        "ANSWERED claims names no catalogued text mints (dead claims, or an \
         arm that landed ahead of its text): {unminted:?}"
    );

    // The two lists are about opposite things — answered, and knowingly
    // refused — so a name in both makes the union check unfalsifiable.
    let both: Vec<&&str> = answered.intersection(&unstaged).collect();
    assert!(
        both.is_empty(),
        "a name cannot be both answered and deliberately unstaged: {both:?}"
    );
}

/// Every catalogued SKU's plans, both classes, by family fixture —
/// mirroring `model/tests/catalogue_coverage.rs`'s invocations, so the two
/// gates walk the same texts.
fn all_catalogued_plans() -> Vec<(String, model_ir::trace::ForwardPlan)> {
    use model_ir::trace::FireClass;
    const CLASSES: [FireClass; 2] = [FireClass::Decode, FireClass::Prefill];
    let mut out: Vec<(String, model_ir::trace::ForwardPlan)> = Vec::new();
    let mut push = |name: &str, plan: model_ir::trace::ForwardPlan| {
        out.push((name.to_string(), plan));
    };

    {
        use model::gemma_4::forward::{
            CATALOG,
            facts::{Gemma4CudaFacts, Gemma4Facts},
        };
        let (f, c) = (
            Gemma4Facts::gemma_4_e4b(),
            Gemma4CudaFacts::gemma_4_e4b_synthetic(),
        );
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, &c, cl, 1e-6));
            }
        }
    }
    {
        use model::gemma_2::forward::{CATALOG, facts::Gemma2Facts};
        let f = Gemma2Facts::gemma_2_9b();
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, cl, 1e-6, 10_000.0));
            }
        }
    }
    {
        use model::gemma_3n::forward::{CATALOG, facts::Gemma3nFacts};
        let f = Gemma3nFacts::gemma3n_synthetic();
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, cl, 1e-6, 1_000_000.0, 10_000.0));
            }
        }
    }
    {
        use model::glm_5::forward::{CATALOG, facts::Glm5Facts};
        let f = Glm5Facts::glm5_106b_a12b();
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, cl, 1e-6, 10_000.0));
            }
        }
    }
    {
        use model::gpt_oss::forward::{
            CATALOG,
            facts::{GptOssCudaFacts, GptOssFacts},
        };
        let (f, c) = (
            GptOssFacts::gpt_oss_20b(),
            GptOssCudaFacts::gpt_oss_20b_synthetic(),
        );
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, &c, cl, 1e-6, 150_000.0, 128));
            }
        }
    }
    {
        use model::deepseek_v4::forward::{CATALOG, facts::Dsv4Facts};
        let f = Dsv4Facts::dsv4_synthetic();
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, cl, 1e-6, 10_000.0));
            }
        }
    }
    {
        use model::kimi_k2::forward::{
            CATALOG,
            facts::{KimiCudaFacts, KimiFacts},
        };
        let (f, c) = (KimiFacts::kimi_k2(), KimiCudaFacts::kimi_k2_synthetic());
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, &c, cl, 1e-6));
            }
        }
    }
    {
        use model::kimi_k3::forward::{CATALOG, facts::KimiK3Facts};
        let f = KimiK3Facts::kimi_k3_synthetic();
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, cl, 1e-6));
            }
        }
    }
    {
        use model::nemotron_h::forward::{CATALOG, facts::NemotronHFacts};
        let f = NemotronHFacts::nemotron_h_synthetic();
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, cl, 1e-6, 10_000.0));
            }
        }
    }
    {
        use model::qwen_3_5::forward::{
            CATALOG,
            facts::{Qwen35CudaFacts, Qwen35HybridFacts},
        };
        let (f, c) = (
            Qwen35HybridFacts::qwen3_5_0_8b(),
            Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
        );
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, &c, cl, 1e-6, 10_000.0));
            }
        }
    }
    {
        use model::shared::llama_like::forward::{
            CATALOG,
            facts::{LlamaLikeCudaFacts, LlamaLikeFacts},
        };
        let (f, c) = (
            LlamaLikeFacts::qwen3_0_6b(),
            LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        );
        for (n, t) in CATALOG {
            for cl in CLASSES {
                push(n, t(&f, &c, cl, 1e-6, 10_000.0));
            }
        }
    }
    out
}
