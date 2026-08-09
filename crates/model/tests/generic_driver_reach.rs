//! How far ONE driver would already reach.
//!
//! The flat list is family-independent now: a rectangle carries its
//! kernel by index and its operands as slots, so a driver walking it
//! needs nothing per-family except a name-to-tensor map. What it still
//! needs is an ARM per launcher symbol — the call itself.
//!
//! ONE registry resolves a set of symbols. Seven families were declared
//! without any arm for theirs. This measures the overlap, which is the
//! size of the remaining work and the only honest way to state it: not
//! "seven executors to write" -- there is one executor -- but "N symbols
//! the registry does not resolve".
//!
//! It is a measurement, not a gate — it prints, and only fails if the
//! registries stop being readable.

use model_compiler::lower::{lower, Fire, Row};
use model_compiler::trace::{FireClass, ForwardPlan};
use std::collections::BTreeSet;

fn arms() -> BTreeSet<String> {
    // ONE registry now. There were four -- one per family executor --
    // and this function read all four; the merge is what made
    // `AttnFlashinferPrefill` naming two different kernels visible.
    //
    // Reading the header rather than the executors is also the more
    // honest measure: an arm serves a symbol only if the registry
    // resolves it, and the registry is where that is decided.
    let path = format!(
        "{}/../driver-cuda/csrc/src/model/declared/registry.hpp",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut out = BTreeSet::new();
    for (i, _) in text.match_indices("k == \"") {
        if let Some(end) = text[i + 6..].find('"') {
            out.insert(text[i + 6..i + 6 + end].to_string());
        }
    }
    assert!(
        !out.is_empty(),
        "the registry stopped being literal compares"
    );

    // AND the GENERATED dispatch, which needs no registry entry: it is
    // keyed by the symbol the statement carries, so the enum a
    // hand-written switch wanted never enters the path.
    //
    // A branch there is a STRONGER claim than a registry entry. An entry
    // says some executor may have an arm; a branch says the shared
    // switch fires this symbol for every family that states it, from a
    // row that named where each argument comes from. Counting only the
    // registry made a symbol whose row is fully stated read as unserved.
    let dispatch = format!(
        "{}/../driver-cuda/csrc/src/model/declared/generated_dispatch.inc",
        env!("CARGO_MANIFEST_DIR")
    );
    let emitted = std::fs::read_to_string(&dispatch)
        .unwrap_or_else(|e| panic!("cannot read {dispatch}: {e}"));
    let mut generated = 0usize;
    for (i, _) in emitted.match_indices("if (sym == \"") {
        if let Some(end) = emitted[i + 12..].find('"') {
            out.insert(emitted[i + 12..i + 12 + end].to_string());
            generated += 1;
        }
    }
    assert!(
        generated > 0,
        "the generated dispatch stopped emitting symbol compares"
    );
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
        "symbol-keyed arms — registry entries plus generated branches: {}",
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

/// How many arms are LEFT in the four family executors.
///
/// The number to steer by, and it is not the count of `case` labels: a
/// family file holds three switches over the same enum — the walk's
/// commit-advance filter, the pin table's A/B, and the executor proper —
/// and only the third is an arm. So a label counts here when the body
/// under it actually LAUNCHES: it names a `kernels::` entry point or one
/// of the shared `arm_*` helpers.
///
/// A measurement, like everything else in this file. It prints, and only
/// fails if the executors stop being readable — because the honest
/// version of "how much is left" is a number nobody can talk up, and one
/// hand-counted in a commit message is exactly that.
#[test]
fn how_many_arms_the_four_executors_still_hold() {
    let families = ["llama_like", "mixtral", "gemma4", "qwen3_5"];
    let mut total = 0usize;
    println!("{:12} {:>6} {:>7}", "", "arms", "labels");
    for f in families {
        let path = format!(
            "{}/../driver-cuda/csrc/src/model/{f}/declared_forward.cpp",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));

        let mut arms = 0usize;
        let mut labels = 0usize;
        let mut pending = 0usize;
        for line in text.lines() {
            let t = line.trim_start();
            if let Some(rest) = t.strip_prefix("case declared::Kernel::") {
                if rest.contains(':') {
                    pending += 1;
                    continue;
                }
            }
            if pending == 0 {
                continue;
            }
            // A LAUNCHING body closes the run of labels above it.
            if t.contains("kernels::") || t.contains("declared::arm_") {
                arms += 1;
                labels += pending;
                pending = 0;
            } else if t.starts_with("break;")
                || t.starts_with("return ")
                || t.starts_with("place(")
                || t.starts_with("pin(")
            {
                // A filter's label, or a pin table's. Not an arm.
                pending = 0;
            }
        }
        println!("{f:12} {arms:>6} {labels:>7}");
        total += arms;
    }
    println!("\nlaunching arms left across the four executors: {total}");
    assert!(
        total > 0,
        "no executor holds a launching arm — either D1 is finished or \
         this measurement stopped reading the executors"
    );
}
