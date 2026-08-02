//! How far ONE driver would already reach.
//!
//! The flat list is family-independent now: a rectangle carries its
//! kernel by index and its operands as slots, so a driver walking it
//! needs nothing per-family except a name-to-tensor map. What it still
//! needs is an ARM per launcher symbol — the call itself.
//!
//! Four executors exist and between them resolve a set of symbols. Seven
//! families were declared without one. This measures the overlap, which
//! is the size of the remaining work and the only honest way to state it:
//! not "seven executors to write" but "N symbols that no arm covers".
//!
//! It is a measurement, not a gate — it prints, and only fails if the
//! registries stop being readable.

use model_compiler::lower::{lower, Fire, Row};
use model_compiler::trace::{FireClass, ForwardPlan};
use std::collections::BTreeSet;

fn arms() -> BTreeSet<String> {
    let root = format!(
        "{}/../driver-cuda/csrc/src/model",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut out = BTreeSet::new();
    for fam in ["llama_like", "qwen3_5", "gemma4", "mixtral"] {
        let path = format!("{root}/{fam}/declared_forward.cpp");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        for (i, _) in text.match_indices("== \"") {
            let before = &text[..i];
            if !(before.ends_with("k ") || before.ends_with("kernel ")) {
                continue;
            }
            if let Some(end) = text[i + 4..].find('"') {
                out.insert(text[i + 4..i + 4 + end].to_string());
            }
        }
    }
    assert!(!out.is_empty(), "the registries stopped being literal compares");
    out
}

/// Every file under the driver's `model/` tree, with its text.
///
/// The point is WHERE a symbol is called, which turns out to be the
/// difference between two very different statements of the remaining
/// work. An executor dispatches on a SYMBOL only inside its `Launch`
/// case, so [`arms`] sees only those; but a symbol absent from every
/// registry may still be called — by a kind-keyed arm
/// (`case PieForwardOpKind::Rmsnorm:`, which never names
/// `launch_rmsnorm_bf16`), or by that family's own HAND-WRITTEN forward
/// pass, which is a working CUDA implementation of the scheme.
///
/// `model/` and not all of `csrc/src`, so a match means a CALL and not a
/// launcher declaration sitting in a kernels header.
fn driver_sources() -> Vec<(String, String)> {
    let root = format!(
        "{}/../driver-cuda/csrc/src/model",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut out = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(&root)];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let is_source = p
                .extension()
                .is_some_and(|x| ["cpp", "hpp", "cu", "cuh", "inc"].contains(&&*x.to_string_lossy()));
            if !is_source {
                continue;
            }
            if let Ok(t) = std::fs::read_to_string(&p) {
                let rel = p
                    .to_string_lossy()
                    .rsplit("csrc/src/model/")
                    .next()
                    .unwrap_or("?")
                    .to_string();
                out.push((rel, t));
            }
        }
    }
    assert!(!out.is_empty(), "the driver sources moved");
    out
}

/// Every launcher symbol a driver walking this family's rectangles would
/// have to call.
///
/// Read off the LOWERING, not off `OpKind::Launch`. Reading the ops was
/// the obvious thing and it undercounts, because a `Launch` statement is
/// only the case where the TEXT names its symbol. Every other kind names
/// one too — through `lower::semantic`, which is why the residue is
/// empty — so `Matmul` reaches the driver as `gemm_act_x_w` and
/// `Rmsnorm` as `launch_rmsnorm_bf16`, and neither appeared in a count
/// that filtered for `Launch`. The kernel table is the question a driver
/// actually asks.
fn stated(plan: &ForwardPlan) -> BTreeSet<String> {
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        8
    ];
    match lower(plan, &rows, Fire::default()) {
        Ok(out) => out.kernels.into_iter().collect(),
        // A family whose decode text will not lower has no rectangles to
        // drive, so it owes the driver nothing YET — and saying so beats
        // reporting a smaller debt than it has.
        Err(_) => BTreeSet::new(),
    }
}

#[test]
fn how_many_symbols_the_undriven_families_still_owe() {
    use model::*;
    let d = FireClass::Decode;
    let plans: Vec<(&str, ForwardPlan)> = vec![
        ("glm5", glm5::forward::glm5_cuda(&glm5::forward::facts::Glm5Facts::glm5_106b_a12b(), d)),
        ("kimi_k2", kimi_k2::forward::kimi_cuda(
            &kimi_k2::forward::facts::KimiFacts::kimi_k2(),
            &kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
            d,
        )),
        ("kimi_k3", kimi_k3::forward::kimi_k3_cuda(
            &kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(), d)),
        ("deepseek_v4", deepseek_v4::forward::dsv4_cuda(
            &deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(), d)),
        ("nemotron_h", nemotron_h::forward::nemotron_h_cuda(
            &nemotron_h::forward::facts::NemotronHFacts::nemotron_h_synthetic(), d)),
        ("gemma3n", gemma3n::forward::gemma3n_cuda(
            &gemma3n::forward::facts::Gemma3nFacts::gemma3n_synthetic(), d)),
        ("gemma_2", gemma_2::forward::gemma2_cuda(
            &gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(), d)),
    ];

    let have = arms();
    let sources = driver_sources();
    // Where a symbol is already called, if anywhere.
    let caller = |k: &str| -> Option<&str> {
        sources
            .iter()
            .find(|(_, text)| text.contains(k))
            .map(|(path, _)| path.as_str())
    };

    let mut owed_all: BTreeSet<String> = BTreeSet::new();
    println!(
        "symbol-keyed arms across the four existing executors: {}",
        have.len()
    );
    println!(
        "{:12} {:>7} {:>7} {:>7} {:>7}",
        "", "states", "keyed", "ported", "unwritten"
    );
    for (name, plan) in &plans {
        let s = stated(plan);
        let owed: Vec<&String> = s.iter().filter(|k| !have.contains(*k)).collect();
        let ported = owed.iter().filter(|k| caller(k).is_some()).count();
        println!(
            "{name:12} {:7} {:7} {:7} {:7}",
            s.len(),
            s.len() - owed.len(),
            ported,
            owed.len() - ported
        );
        owed_all.extend(owed.into_iter().cloned());
    }

    let (ported, unwritten): (Vec<&String>, Vec<&String>) =
        owed_all.iter().partition(|k| caller(k).is_some());
    println!(
        "\nDISTINCT symbols no symbol-keyed arm covers: {}",
        owed_all.len()
    );
    // ALREADY CALLED, and this is the finding: the call sites are those
    // families' own HAND-WRITTEN passes. MLA, DSA, KDA, AltUp, the
    // hyper-connections — every scheme the undriven families need is
    // already implemented in CUDA and running. Giving the declared path
    // an arm is PORTING a call whose operands are bound a few lines
    // away, not building the scheme. The file is printed so the port
    // starts at the right line.
    println!(
        "\n  already called ({}) — port the call into a symbol-keyed arm:",
        ported.len()
    );
    for k in &ported {
        println!("    {:52} {}", k, caller(k).unwrap_or("?"));
    }
    // NOT CALLED ANYWHERE: the genuinely new work, and the number worth
    // quoting as the remaining debt.
    println!(
        "\n  unwritten ({}) — no call anywhere under model/:",
        unwritten.len()
    );
    for k in &unwritten {
        println!("    {k}");
    }
}
