//! The driver must not know what a model is — measured, not asserted.
//!
//! `crates/model/tests/one_normalizer.rs` states the rule: *"what a
//! driver reads is the answer, never the question"*, and names its own
//! ceiling — *"nothing stops a future reader from opening `config.json`
//! again."* A grep is a guard that must be remembered, which is exactly
//! the failure mode it warns about. This is the grep, so that the
//! remembering is the build's job.
//!
//! # Why a ratchet rather than zero
//!
//! The 33-row `model_type` table is gone from this crate — it lives in
//! `model::catalog`, which is where the question is allowed to
//! be asked — and the fire path reads a `Deployment` with no family
//! name in it. What is left is not dispatch:
//!
//! - `layout/memory_planner.rs` sizes a KV pool per architecture,
//!   because the COST of a token genuinely differs. That is arithmetic
//!   about bytes, not a branch about behaviour.
//! - `bind/` names kernels, and a kernel's name is its own.
//! - `serve/encode.rs` serves gemma-4's towers, which is a real
//!   family-specific ABI entry and the next thing to generalise.
//!
//! Each is a separate argument, and collapsing them into one number
//! would hide which is which. So this pins the number per FILE: a file
//! may not gain family names, and a file that loses them ratchets the
//! bound down.
//!
//! # What the count reads
//!
//! Two corrections, both found in `driver-metal` and ported here so the
//! backends are held to one standard rather than to whichever names each
//! happened to notice. The count skips `#[cfg(test)]` blocks, because a
//! test naming the family it covers is an asset; five files left this
//! table outright once their only mentions were their own fixtures. And
//! it compares case- and punctuation-blind, which raised
//! `memory_planner.rs` from 25 to 30 and surfaced `fire/lora.rs`, a file
//! that had never been listed.
//!
//! **A file not in the table may have none at all.** That is the half
//! that matters — it is what stops a new dispatch site appearing
//! somewhere quiet.

use std::collections::BTreeMap;

/// The family names a driver must not be branching on.
const FAMILIES: &[&str] =
    &["gemma", "qwen", "llama", "deepseek", "kimi", "nemotron", "glm", "gpt_oss"];

/// Non-comment lines naming a family, per file, as of the move.
///
/// These are CEILINGS. Lowering one is the point; raising one is the
/// thing this test exists to catch.
fn budget() -> BTreeMap<&'static str, usize> {
    [
        // ONE launcher symbol and ONE aliased import — still two, and the
        // composition changed under it. Seven became five when the GQA
        // instantiation set became this crate's own `serve::DECODE_GQA_GROUPS`,
        // and five became
        // two when the ~50 tower tensor paths moved to
        // `model::shared::tower_names` — which owns the names AND the
        // launcher order, because the stride (62 audio, 41 vision) is the
        // ABI.
        //
        // Two is the FLOOR and not a deferral. What is left is the name of
        // the function being called: an encode entry has to call some
        // launcher, and this build's audio tower launcher is spelled that in
        // `kernels-cuda`, the ARCHIVE crate. Lowering it further means
        // renaming the kernel, which is that crate's business and not a fact
        // about this one — the same line `bind/mod.rs` used to sit on, except
        // there the names were being MATCHED to make a routing decision,
        // which is what made them removable.
        //
        // NEITHER half is a launcher symbol any more: both walks are Rust, in
        // `tower/gemma4_vision.rs` and `tower/gemma4_audio.rs`, and this file
        // reaches them through `use crate::tower::gemma4_* as *_tower`. Two
        // imports instead of two call sites — the names are still stated,
        // once each, where a reader goes to find out what `vis_tower` and
        // `aud_tower` mean.
        //
        // The number did not move, and that is worth saying plainly rather
        // than claiming a win: two C++ symbols became two Rust module names.
        // What changed is not the count but what is behind it — the paragraph
        // above said lowering it "means renaming the kernel, which is that
        // crate's business", and that is now false in a useful way: there is
        // no kernel to rename, only a module, and a module named for one
        // model is the honest shape while `PieEncodeDesc` has a gemma-shaped
        // ABI. Generalising that ABI is what lowers this to one.
        ("serve/encode.rs", 2),
        // THE TOWER WALKS, which are new and are the exception this test's
        // own doc anticipated: "a file not in the table may have none at
        // all… what stops a new dispatch site appearing somewhere quiet."
        // A tower walk is not a dispatch site. It is the BODY of one named
        // program — gemma-4's vision encoder is gemma-4's vision encoder,
        // the way `fire/lora.rs` is allowed its three — and it moved into
        // `src/` from `csrc/vision/gemma4_vision.cu`, where the counter
        // could not see it and the name was no less present.
        //
        // Both numbers are floors reached on purpose, not what the port
        // arrived at. `tower/gemma4_vision.rs` counted THIRTY-TWO when it
        // was written: `Error::invalid("gemma4_vision", …)` at every
        // refusal and `extent("gemma4_vision: rms rows", …)` at every
        // conversion. Those are now `WHO` and a label, so the file names
        // its tower once, on the `const`, and the messages are unchanged.
        //
        // IT IS ONE NOW, and the line that left was never a family name at
        // all. The second was `pie_k_gemm_act_x_wt_bf16` — `gemm` + `act`
        // reads as "gemma" once the underscores come out — and the dense
        // GEMM's shim entry is gone: `gemm::act_x_wt_bf16` is on
        // `execution::RUST_SERVED` and the tower calls
        // `crate::fire::gemm::act_x_wt_bf16` directly. The `::` survives the
        // flattening (`names_a_family` drops `_` and `-`, not `:`), so the
        // false positive does not. **A false positive leaving is still a
        // ratchet**: the budget must record it or the file gets a free
        // mention nobody is making, which is exactly the silent-loosening
        // failure this test's tail refuses.
        //
        // `tower/mod.rs` is THREE because it declares three submodules, and
        // a `pub mod` line is the one place a module's own name cannot be
        // aliased away.
        //
        // It was TWO, and this is the one direction this table is allowed to
        // move UP: `pub mod qwen3_vl;` arrived with a third tower, by the
        // measurement the module's own header records (*"held zero
        // `__global__`"*). A `pub mod` is not a driver learning what a model
        // is — it is a driver having a file — and the alternative is naming
        // the module something other than the tower it ports, which would
        // make the tree lie to satisfy a substring test. The ratchet still
        // bites everywhere else: this line is a budget for module
        // DECLARATIONS, so a fourth mention inside the file fails even at
        // three.
        //
        // `tower/gemma4_audio.rs` is ONE: it was written at
        // the floor the vision walk had to be edited down to, so it has the
        // `WHO` const and nothing else — no cuBLAS call, because every one
        // of its matmuls is `vision::k_matmul_bf16` rather than a `gemm`.
        ("tower/mod.rs", 3),
        ("tower/gemma4_vision.rs", 1),
        // ── the arm registries, and one tower's `WHO` ────────────────────
        //
        // These eight had NEVER been budgeted, which means this test has been
        // failing since `d3b6bfe7d` created `bind/arms/` — a check that has
        // protected nothing for as long as the thing it checks has existed.
        // Budgeting them is what turns it back on.
        //
        // **None of them is a driver learning what a model is.** An arm
        // registry is a map from a trace SYMBOL to the `fn` that arms it, so
        // its entire content is the trace's vocabulary. The quoted symbols
        // stopped counting on their own (see `without_trace_symbols`); what
        // is left is the Rust side of the same pair — `nemotron_mamba_split_\
        // arm` beside `"ssm::nemotron_mamba_split_bf16"`, and the host `fn`
        // it calls. A kernel's name belongs to the kernel, and the
        // architecture that introduced it is part of that name; an arm that
        // spelled it differently would be an arm you could not find from the
        // symbol.
        //
        // That is NOT an argument for exempting the directory, and the
        // numbers are here rather than a rule for that reason: a `GemmaFacts`
        // in one of these files would be exactly this test's quarry, and a
        // per-file count still catches it. The ratchet is unusual here in
        // that it may legitimately rise — arming a kernel adds a mention —
        // and that is the honest cost of counting identifiers rather than
        // trusting a directory.
        ("bind/arms/attn.rs", 8),
        ("bind/arms/layout.rs", 3),
        ("bind/arms/mlp.rs", 4),
        // TWO, down from three, and the ratchet is doing what it is for.
        // The third was inside `norm::rmsnorm_bf16`'s `unbound:` reason,
        // which had to say *"without the query this fn would norm gemma-4's
        // q/k heads as one row each"* to explain what was missing. The arm is
        // written now, so the sentence moved to `Fire::per_head_dim` where
        // the query lives and the reason is a property of the fact rather
        // than of the gap. **A false positive leaving is still a ratchet**:
        // the budget records it, or the file gets a free mention nobody is
        // making.
        ("bind/arms/norm.rs", 2),
        ("bind/arms/rope.rs", 1),
        // SIXTEEN, and the two it gained are this test working. A
        // `Bound { arm: None, unbound: Some(..) }` for
        // `ssm::qwen_gdn_post_conv_prep_bf16` names that symbol twice more —
        // once as the entry's `symbol` and once in the reason, which names
        // the `driver_internal` `fn` a future arm should call. Both are the
        // kernel's name, and the second is the one a reader needs.
        ("bind/arms/ssm.rs", 16),
        // The tower names ITSELF: `const WHO: &str = "qwen3vl_vision"`, the
        // string a refusal carries so a reader knows which tower declined.
        // A tower is the one place in this driver where knowing the
        // architecture is the job — `qwen3_vl_tower.cu` is what it ports —
        // and the alternative is a refusal that will not say who made it.
        ("tower/qwen3_vl.rs", 1),
        ("tower/qwen3_vl/attn.rs", 1),
        ("tower/gemma4_audio.rs", 1),
        // Never listed until the counter learned to read `LoraGemma`, and
        // ZERO now — which is the outcome the `n == 0 && !contains_key`
        // guard exists to keep on the books rather than let drop off it.
        //
        // All three were the same false positive `tower/gemma4_vision.rs`
        // carried: `ffi::pie_k_gemm_act_x_wt_bf16`, three call sites in the
        // ungrouped LoRA apply. They are `act_x_wt_bf16(..)` now, imported
        // once as `use crate::fire::gemm::act_x_wt_bf16;` — and neither
        // spelling flattens to a family name, because the `::` stays. The
        // file never named a model and now it does not appear to either.
        ("fire/lora.rs", 0),
        // `fire/gemm.rs` STOOD HERE at THREE, and the file is gone — its
        // dense GEMM descended into `kernels-cuda`, where the host
        // program lives beside the `.cuh` it launches.
        //
        // Not one of its three was a family name. They were cuBLAS's own
        // spelling: `cublasGemmAlgo_t` flattens to `cublasgemmalgot`, and
        // `gemm` + `a` reads as "gemma" exactly the way `gemm_act` does. The
        // line was the `use` importing the type plus the two `const`s
        // pinning `CUBLAS_GEMM_DEFAULT_TENSOR_OP` and `CUBLAS_GEMM_DEFAULT`.
        //
        // It was listed rather than worked around, and the reason outlives
        // the file: the alternative was aliasing the cuBLAS type to
        // something without the letters, which would make a file lie about
        // what it calls to satisfy a substring test. `names_a_family`'s own
        // doc admits that trade — it flattens punctuation to catch
        // `GemmaFacts`, and the price is `cublasGemmAlgo_t`. `bind/service.rs`
        // and `bind/quant_gemm.rs` still carry the same false positive from
        // the same library, which is why the paragraph is kept here rather
        // than deleted with the entry.
        // `bind/mod.rs` left this table entirely. Its four were the mamba
        // join's kernel SYMBOLS — `ssm::nemotron_mamba_split_bf16` and its
        // neighbours — matched by name to route values across statements.
        // The wiring is stated on the declarations now
        // (the publishers and the `Source::Aux`
        // operands already there for the consumers), so the join is
        // arithmetic over the table and names nothing.
    ]
    .into_iter()
    .collect()
}

fn count(path: &std::path::Path) -> usize {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    // `#[cfg(test)]` opens a block that closes when its braces balance.
    // A test that names the family it covers is what makes the test
    // readable; charging it as debt taxes exactly the coverage this rule
    // needs. Depth, not "the rest of the file": a file may have several
    // test modules with real code between them.
    let mut depth = 0i32;
    let mut armed = false;
    let mut n = 0;
    for l in text.lines() {
        let opens = i32::try_from(l.matches('{').count()).unwrap_or(0);
        let closes = i32::try_from(l.matches('}').count()).unwrap_or(0);
        if l.trim_start().starts_with("#[cfg(test)]") {
            armed = true;
        }
        // Comments are provenance -- "ports gemma_4/forward" -- not
        // dispatch. The claim is about CODE.
        if depth == 0 && !armed && !l.trim_start().starts_with("//") && names_a_family(l) {
            n += 1;
        }
        if armed {
            depth += opens - closes;
            if depth <= 0 && closes > 0 {
                armed = false;
                depth = 0;
            }
        }
    }
    n
}

/// Whether a line names a family, whatever spelling it wears.
///
/// Matching raw text saw `gemma_facts` and missed `GemmaFacts`, which is
/// where family names actually appear in Rust. Reducing the line the way
/// `model/text.rs` reduces an architecture string -- drop case and
/// punctuation, then compare -- makes `GptOss`, `gpt_oss` and `gptoss`
/// one name.
fn names_a_family(line: &str) -> bool {
    let line = without_trace_symbols(line);
    let flat: String =
        line.chars().filter(|c| *c != '_' && *c != '-').flat_map(char::to_lowercase).collect();
    FAMILIES.iter().any(|f| {
        let f: String = f.chars().filter(|c| *c != '_').collect();
        flat.contains(&f)
    })
}

/// The line with every quoted TRACE SYMBOL removed.
///
/// # Why a symbol is not the driver knowing what a model is
///
/// `bind/arms/ssm.rs` names `"ssm::nemotron_mamba_split_bf16"` twenty times
/// over, and `"ssm::kda_gate_beta_bf16"` beside it. Flattened, both contain a
/// family name — and neither is this test's quarry. **A kernel's name belongs
/// to the kernel.** `nemotron_mamba_split` is what that `__global__` is
/// called, because that is the architecture that introduced it; the driver
/// repeating it is the driver spelling a symbol the TRACE chose, in the
/// trace's own vocabulary.
///
/// What this test exists to catch is a driver BRANCHING on a model —
/// `GemmaFacts`, a `match arch`, a per-family type table — and a quoted
/// `family::kernel` literal cannot be any of those. It is a lookup key.
///
/// The existing budget already reasons this way for the adjacent case: it
/// records that `gemm::act_x_wt_bf16` stopped counting because *"the `::`
/// survives the flattening (`names_a_family` drops `_` and `-`, not `:`), so
/// the false positive does not"* — there the family name was formed ACROSS
/// the colons. This is the same observation applied where the architecture
/// sits inside the kernel's own half, which the flattening could not see.
///
/// # The shape is narrow on purpose
///
/// Only a quoted string that is ENTIRELY `lower_snake::lower_snake` is
/// dropped. A model type, a variant, a field or a comparison cannot wear that
/// shape, so nothing this test was written to find can hide behind it —
/// which is what keeps a rule change from being the free permission the
/// budget's own tail refuses.
fn without_trace_symbols(line: &str) -> String {
    let ok = |s: &str| {
        let Some((ns, name)) = s.split_once("::") else { return false };
        let seg = |x: &str| {
            !x.is_empty()
                && x.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        };
        seg(ns) && seg(name)
    };
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find('"') {
        let (before, after_open) = rest.split_at(open);
        out.push_str(before);
        let body = &after_open[1..];
        let Some(close) = body.find('"') else {
            out.push_str(after_open);
            return out;
        };
        if !ok(&body[..close]) {
            out.push('"');
            out.push_str(&body[..close]);
        }
        out.push('"');
        rest = &body[close + 1..];
    }
    out.push_str(rest);
    out
}

fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// No file gains a family name, and no NEW file has one at all.
#[test]
fn the_driver_does_not_learn_a_new_family() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    assert!(!files.is_empty(), "the driver has source");

    let budget = budget();
    let mut over = Vec::new();
    let mut unlisted = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(&root).expect("under src").to_string_lossy().into_owned();
        let n = count(f);
        // NOT `if n == 0 { continue }`. A budgeted file that fell to
        // zero is the good outcome and it still has to be recorded, or
        // the budget keeps a line that permits a mention nobody is
        // making.
        if n == 0 && !budget.contains_key(rel.as_str()) {
            continue;
        }
        match budget.get(rel.as_str()) {
            Some(&cap) if n == cap => {}
            Some(&cap) => over.push(format!("{rel}: {n}, budgeted {cap}")),
            None => unlisted.push(format!("{rel}: {n}")),
        }
    }
    // A LINE THAT LEAVES MUST LOWER THE BUDGET, which is why this is
    // `==` and not `<=`. A ceiling only ratchets one way: the arm
    // deletions took ten family names out of `bind/mod.rs` and the
    // budget kept saying fourteen, so the guard had ten free mentions to
    // give away and nobody would have noticed. That is §4's own warning
    // about a grep being "a guard that must be remembered" — a guard
    // that silently loosens is worse than one that fails, because it
    // reads as passing.
    //
    // The cost is real and it is the point: deleting a family name is a
    // two-line diff instead of one. The second line is the record that
    // it happened.
    for (rel, &cap) in &budget {
        if cap > 0
            && !files
                .iter()
                .any(|f| f.strip_prefix(&root).is_ok_and(|r| r.to_string_lossy() == *rel))
        {
            over.push(format!("{rel}: budgeted {cap}, and the file is gone"));
        }
    }
    assert!(
        over.is_empty(),
        "these files gained family names. Each one is a driver learning what a \
         model is, which is what `crates/model` exists to prevent — see \
         `model::deployment::Deployment`:\n  {}",
        over.join("\n  ")
    );
    assert!(
        unlisted.is_empty(),
        "these files name a family and are not in the budget. A NEW dispatch \
         site is the thing this test exists to catch; if the mention is \
         legitimate, add it with the argument for why:\n  {}",
        unlisted.join("\n  ")
    );
}

/// The dispatch itself is gone: no `model_type` table remains here.
///
/// This is the strong half of the claim, and it got stronger. The
/// 33-row `FACTS_ROWS` and its eleven derivations first MOVED, out of
/// this driver and into `model::deployment_cuda` — which was the right
/// first step and still left a table keyed on a `config.json` string,
/// merely one crate over.
///
/// They are DELETED now. There is no `model_type` table anywhere: a
/// checkpoint is matched to a `model::catalog` row by its TENSORS, and
/// what a driver receives is a `Deployment`, a value with no family
/// name in it and therefore nothing to branch on.
///
/// So this guard now watches for the table's RETURN rather than for its
/// escape, which is why the message no longer names a place it could
/// legitimately live.
#[test]
fn no_model_type_table_remains_in_the_driver() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") {
                continue;
            }
            for banned in ["FACTS_ROWS", "HF_ROWS", "MLX_ROWS"] {
                assert!(
                    !t.contains(banned),
                    "{}:{} names `{banned}`, a table keyed on a `config.json` \
                     string. There is no such table any more — a checkpoint \
                     is matched to a `model::catalog` row by its tensors, and \
                     a driver reads the answer, never the question",
                    f.display(),
                    i + 1
                );
            }
        }
    }
}

/// Every budgeted file exists and still spends what it is budgeted.
///
/// The other half of the ratchet, ported from `driver-metal`. Without it a
/// file can be renamed, deleted, or cleaned up and leave its ceiling behind
/// as a permission nobody is using -- and the next file to need that
/// permission inherits it silently. A budget that only ever caps is a budget
/// that only ever grows. Five files left the table the first time this ran.
#[test]
fn no_budget_line_outlives_what_it_was_for() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut stale = Vec::new();
    for (rel, cap) in budget() {
        let path = root.join(rel);
        if !path.exists() {
            stale.push(format!("{rel}: budgeted {cap}, file is gone"));
            continue;
        }
        let n = count(&path);
        if n < cap {
            stale.push(format!("{rel}: budgeted {cap}, actually {n} -- lower the ceiling"));
        }
    }
    assert!(
        stale.is_empty(),
        "the budget is stale. Every line below is a permission larger than \
         what uses it, and the next file to want that permission gets it for \
         free:\n  {}",
        stale.join("\n  ")
    );
}
