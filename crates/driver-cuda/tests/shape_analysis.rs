//! **C1: how many distinct SHAPES are behind eleven families?**
//!
//! `cuda.md` §5.C1 asks this and says the answer sizes all of C: "if the
//! families collapse toward a smaller set of *shapes* + facts, the
//! descriptor→plan step gets much smaller." `llama_like` already serves
//! five deployments off facts alone, so the question is whether that is
//! the exception or the rule.
//!
//! # What is measured, and why it is the right proxy
//!
//! A family's SHAPE is the sequence of kernel symbols one layer launches.
//! Not the counts, not the widths, not the layer total — those are facts,
//! and facts are exactly what a descriptor already carries. Two families
//! with the same per-layer symbol sequence differ only in numbers, so one
//! described plan plus two fact rows would serve both; two families with
//! different sequences are genuinely different programs.
//!
//! Layer 1 is read rather than layer 0, because layer 0 carries the embed
//! and some families fold a prologue into it; and one layer rather than
//! the whole trace, because the epilogue is shared and would drown the
//! signal.
//!
//! # Why this is a test and not a note
//!
//! The number it prints is an input to a plan, and a plan that quotes a
//! number nobody can re-derive goes stale silently. Run it and it answers
//! for the tree as it is:
//!
//! ```sh
//! cargo test -p driver-cuda --features cuda-13 --test shape_analysis -- --nocapture
//! ```

#![cfg(feature = "_cuda")]

use std::collections::BTreeMap;

use model_compiler::lower::{Fire, Lowered, Row, lower};
use model_compiler::trace::FireClass;

fn rows(n: usize) -> Vec<Row> {
    vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ]
}

fn lower_plan(plan: &model_compiler::trace::ForwardPlan) -> Lowered {
    lower(
        plan,
        &rows(4),
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("lowers")
}

/// The op-kind sequence one layer STATES, as distinct from the kernels it
/// lowers to.
///
/// The two answer different questions and C1 needs both. A kernel shape
/// that differs may only mean the lowering picked a fusion this
/// deployment's facts allow — `qkv_decode_qk_norm_rope_write_kv_bf16`
/// where another deployment spells `split_qkv` + `rope` +
/// `write_kv_to_pages` is one statement lowering two ways, not two
/// declarations. An OP shape that differs means the declaration itself
/// said something else, and only that is work C has to describe.
fn op_shape(plan: &model_compiler::trace::ForwardPlan, l: &Lowered) -> Vec<String> {
    l.launches
        .iter()
        .filter(|x| x.layers.start == 1)
        .map(|x| {
            let d = format!("{:?}", plan.ops[x.op as usize].kind);
            d.split([' ', '(', '{']).next().unwrap_or("?").to_string()
        })
        .collect()
}

/// Every family this tree declares a CUDA decode for, by name.
fn families() -> Vec<(&'static str, Lowered)> {
    use model::shared::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    vec![
        (
            "llama_like",
            lower_plan(&model::shared::llama_like::forward::llama_like_cuda(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                FireClass::Decode,
            )),
        ),
        (
            "gemma_2",
            lower_plan(&model::gemma_2::forward::gemma2_cuda(
                &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
                FireClass::Decode,
            )),
        ),
        (
            "glm5",
            lower_plan(&model::glm_5::forward::glm5_cuda(
                &model::glm_5::forward::facts::Glm5Facts::glm5_106b_a12b(),
                FireClass::Decode,
            )),
        ),
        (
            "kimi_k2",
            lower_plan(&model::kimi_k2::forward::kimi_cuda(
                &model::kimi_k2::forward::facts::KimiFacts::kimi_k2(),
                &model::kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
                FireClass::Decode,
            )),
        ),
        (
            "kimi_k3",
            lower_plan(&model::kimi_k3::forward::kimi_k3_cuda(
                &model::kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(),
                FireClass::Decode,
            )),
        ),
        (
            "deepseek_v4",
            lower_plan(&model::deepseek_v4::forward::dsv4_cuda(
                &model::deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(),
                FireClass::Decode,
            )),
        ),
    ]
}

/// The per-layer kernel-symbol sequence — the family's shape.
fn shape_of(l: &Lowered) -> Vec<&str> {
    l.launches
        .iter()
        .filter(|x| x.layers.start == 1)
        .map(|x| l.kernels[x.kernel as usize].as_str())
        .collect()
}

#[test]
fn how_many_shapes_are_behind_the_families() {
    let fams = families();
    let mut by_shape: BTreeMap<Vec<&str>, Vec<&str>> = BTreeMap::new();
    for (name, l) in &fams {
        by_shape.entry(shape_of(l)).or_default().push(name);
    }

    println!("\n── C1: shapes behind the families ──────────────────────");
    for (shape, names) in &by_shape {
        println!("  {:?}  ({} ops/layer)", names, shape.len());
    }
    println!(
        "\n  {} families → {} distinct shapes\n",
        fams.len(),
        by_shape.len()
    );

    // The finding, pinned so that it cannot drift unnoticed: the families
    // do NOT collapse. Each declares its own per-layer program, and the
    // one axis that already collapses (llama_like's five deployments) does
    // so through FACTS on one shape rather than by sharing a shape with
    // another family.
    //
    // That is C's real size: descriptor→plan cannot be "describe the one
    // shape and vary the numbers". It has to describe a per-layer op
    // sequence as DATA — which is what §5.C2 says, and this is the
    // measurement that says C2 cannot be skipped.
    assert_eq!(
        by_shape.len(),
        fams.len(),
        "two families share a per-layer shape — if this ever fires, C got \
         smaller and the plan should say so"
    );
}

#[test]
fn facts_select_programs_not_only_numbers() {
    // THE C1 RESULT, and it is not the one the plan expected.
    //
    // §5.C1 reasoned: "`llama_like` already serves 5 deployments off
    // facts. If the families collapse toward a smaller set of shapes +
    // facts, the descriptor→plan step gets much smaller." The existence
    // proof it leaned on does not hold. Measured, llama_like's five
    // deployments produce FIVE distinct per-layer programs — 9, 11, 12, 14
    // and 16 ops — and they differ at the OP level, not merely in which
    // fused kernel the lowering picked.
    //
    // So a fact does not fill in a number. It selects a program. C cannot
    // be a fact schema over one described shape, in any family, and the
    // optimistic branch of C1 is closed.
    use model::shared::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};

    let cuda = LlamaLikeCudaFacts::qwen3_0_6b_l40s();
    let deployments: Vec<(&str, LlamaLikeFacts)> = vec![
        ("qwen2_5_1_5b", LlamaLikeFacts::qwen2_5_1_5b()),
        ("qwen3_0_6b", LlamaLikeFacts::qwen3_0_6b()),
        ("phi3_mini", LlamaLikeFacts::phi3_mini()),
        ("mistral_7b_v03", LlamaLikeFacts::mistral_7b_v03()),
        ("olmo2_1b", LlamaLikeFacts::olmo2_1b()),
    ];

    let mut by_kernel: BTreeMap<Vec<&str>, Vec<&str>> = BTreeMap::new();
    let mut by_op: BTreeMap<Vec<String>, Vec<&str>> = BTreeMap::new();
    let mut lowered = Vec::new();
    for (name, facts) in deployments {
        let plan =
            model::shared::llama_like::forward::llama_like_cuda(&facts, &cuda, FireClass::Decode);
        let l = lower_plan(&plan);
        lowered.push((name, plan, l));
    }
    for (name, plan, l) in &lowered {
        by_kernel.entry(shape_of(l)).or_default().push(name);
        by_op.entry(op_shape(plan, l)).or_default().push(name);
    }

    println!("\n── C1: shapes WITHIN llama_like ────────────────────────");
    println!("  by KERNEL (what the lowering emits):");
    for (shape, names) in &by_kernel {
        println!("    {:?}  ({} ops/layer)", names, shape.len());
    }
    println!("  by OP (what the declaration states):");
    for (shape, names) in &by_op {
        println!("    {:?}  ({} ops/layer)", names, shape.len());
    }
    println!(
        "\n  {} deployments → {} kernel shapes, {} op shapes\n",
        lowered.len(),
        by_kernel.len(),
        by_op.len()
    );

    // Pinned as the finding. If either of these ever drops, C got smaller
    // and the plan should be rewritten to say so — which is the only
    // reason to pin a number rather than print it.
    assert_eq!(
        by_kernel.len(),
        lowered.len(),
        "llama_like deployments no longer differ in kernel shape"
    );
    assert_eq!(
        by_op.len(),
        lowered.len(),
        "llama_like deployments no longer differ in OP shape — the \
         difference would then be lowering-level fusion, and C would be \
         much smaller than measured"
    );
}
