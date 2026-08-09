//! Every scalar a kernel reads off the launch context comes from the
//! model, or says why it does not.
//!
//! `kernels-cuda` binds an operand with `Source::Ctx("name")`, which means
//! "the driver holds this, not the text". `fire::launch` builds that
//! context. When it fills one of those names with a bare literal, the
//! kernel gets a number nobody measured — and every one of them is a
//! scalar, so the model does not fault, it degrades.
//!
//! Four had gone that way before this test existed:
//!
//! - `moe_norm_topk: false` — DeepSeek-V3 and Kimi-K2 publish `false`,
//!   GLM-4.5 publishes `true`. Every mixture routed on the first answer.
//! - `moe_routed_scaling: 1.0` — DeepSeek and GLM publish 2.5, Kimi-K2
//!   2.0. A routed token arrived at two-fifths of its trained size.
//! - `glu_limit`/`glu_alpha: 0.0` — `Deployment::mlp_gate` had stated
//!   gpt-oss's `SiluClamped { limit: 7.0, alpha: 1.702 }` all along, and
//!   `alpha` scales the gate INSIDE the sigmoid, so zero collapses
//!   `silu(a*x)` to `x/2` on every routed expert.
//! - `yarn: [0.0; 4]` — gpt-oss states `factor: 32.0, beta_fast: 32.0,
//!   beta_slow: 1.0, original_max_position: 4096` as its generation's
//!   `ROPE_SCALING`, and `driver-metal` REFUSES a load rather than zero
//!   it: "zeroing YaRN's factor would serve the model with an unrescaled
//!   ladder rather than refuse it".
//!
//! None of those was a hard question. Each was a field the deployment
//! already carried, next to a dozen the launch read off it correctly.
//! That is what this test is for: not to judge the value, but to notice
//! when a name stops being connected to the thing that states it.
//!
//! It stopped noticing anything at all for a while. `Source::Ctx` moved
//! to `kernels-cuda-new`, and into a `src/table/` the scan's `read_dir`
//! did not descend into, so `ctx_names()` returned nothing and the test
//! agreed with every literal in the launch. Only the `> 15` floor said
//! so. A scan is a claim about a directory, and a claim about a
//! directory expires when the code moves.

use std::collections::BTreeSet;

/// Names that a CATALOG ROW varies, kept literal because that row has
/// no text to dispatch.
///
/// This is a DEBT, not an argument, and the two must not share a list:
/// [`CONSTANT_BY_ARGUMENT`] means "no checkpoint varies it", and an
/// entry that is merely unreachable makes a weaker claim which reads
/// like the stronger one. The distinction is the whole safety margin —
/// `moe_routed_scaling: 1.0` in the header above was also "safe" right
/// up until a routed text arrived.
///
/// Each entry names the generation that varies it and the fact whose
/// UNREAD status is the premise, so the excuse EXPIRES BY ITSELF: the
/// day someone writes that text, the fact leaves
/// `model/tests/facts_are_read.rs`'s `DECLARED_BUT_UNREAD` and
/// [`the_deferred_names_are_still_out_of_reach`] fails here.
const VARIED_BY_A_ROW_WITH_NO_TEXT: &[(&str, &str, &str, &str)] = &[(
    "rope_interleaved",
    "kimi_k2",
    "rope_yarn_original",
    "read only by `rope_yarn_original`, whose one dispatching caller is \
     gpt-oss, whose rope is the half-split HF writes as `rotate_half`. \
     kimi-k2 is the second caller and it DOES vary this, measured \
     against the modeling code `moonshotai/Kimi-K2-Instruct` publishes \
     rather than assumed from the shared kernel name: its `rotate_half` \
     is the ordinary half-split, but `apply_rotary_pos_emb` first does \
     `q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)` on \
     both `q_pe` and `k_pe` — a DE-INTERLEAVE — so with respect to the \
     tensor as stored the ladder is interleaved and gpt-oss's is not. \
     A literal here would rotate every K2 position against the wrong \
     pair. It cannot today: no MLA text reads `rope_yarn_original`.",
)];

/// The names a literal is the right answer for, and why.
///
/// A name here is a claim that no checkpoint in the catalog varies it.
/// Adding one is cheap and wrong is silent, so each carries the reason
/// it is not a measurement.
const CONSTANT_BY_ARGUMENT: &[(&str, &str)] = &[
    (
        "gate_second",
        "the ORDER of the two halves in a fused gate/up tensor. \
         `contract.rs` decides it per family when it stacks the experts \
         (`hf_moe_expert_stacks(b, gate_second = true)`), so by the time \
         a tensor reaches a kernel it is already in this driver's order. \
         The context copy is the dense path's, and it stacks the other \
         way round.",
    ),
    (
        "write_state",
        "whether the fire advances recurrent state, and every class that \
         still exists does. `GdnCtx::write_state`'s own doc says \
         \"the frozen-verify service classes pass false\" — and \
         `FireClass` says THE REPAIR CLASSES ARE GONE: `FrozenVerify`, \
         `CommitAdvance` and `StateOnly` were retired when the driver \
         accepted PIE_RS_FLAG_FOLD, because a speculative decode writes \
         to a buffer and folds only the accepted prefix, so nothing is \
         ever wrong and nothing needs freezing. Decode and Prefill are \
         what remain and both advance. A false here would need a class \
         to come back.",
    ),
    (
        "wna16_group_size",
        "the AWQ/GPTQ group width, and no row in this catalog is loaded \
         from a w4a16 checkpoint — `alt_quant` is how a quantized bank \
         gets identified and no family sets it. A row that did would \
         have to state the group size, and this line is where it would \
         be read.",
    ),
    (
        "situ_beta",
        "SITU's gate constant. No text in `crates/model` names a situ \
         kernel, so nothing dispatches one.",
    ),
    (
        "situ_linear_beta",
        "as `situ_beta` — the linear half of the same unreached kernel.",
    ),
    (
        "altup_streams",
        "gemma-3n's AltUp width. gemma-3n has no CUDA text: it is \
         projected for shape and refused for serving, so no `norm` \
         kernel of its is ever encoded.",
    ),
    (
        "altup_active",
        "as `altup_streams` — the active stream index of the same \
         unreached block.",
    ),
];

/// A literal, as far as this test is concerned.
fn is_literal(rhs: &str) -> bool {
    let r = rhs.trim().trim_end_matches(',').trim();
    matches!(r, "true" | "false" | "None" | "Vec::new()")
        || r.parse::<f64>().is_ok()
        || (r.starts_with('[') && r.trim_matches(['[', ']'].as_ref()).starts_with("0.0;"))
}

fn read(rel: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The `Source::Ctx` and `Source::Attn` names, less the `yarn[0]`-style
/// indexed ones, which name a slot of an array the launch fills in one
/// statement.
///
/// `Attn` joined after `logits_soft_cap` slipped past a scan that read
/// only `Ctx`. The two sources name two structs — `KernelCtx` and
/// `AttnCtx` — and both are BUILT BY THE SAME FUNCTION out of the same
/// model, so a literal in one is the same defect as a literal in the
/// other. Reading only half of a struct-literal audit is how the half
/// gets read twice.
/// `src` with every `//` tail removed.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every `.rs` under `dir`, at any depth.
fn rust_sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

fn ctx_names() -> BTreeSet<String> {
    // The rows moved: `kernels-cuda`'s flat per-domain modules are
    // `kernels-cuda-new/src/table/` now, and that crate authors them. Read
    // the tree rather than one directory's entries, so the next move empties
    // no scan silently — the count assert below is the backstop, and it is
    // only a backstop if the walk is the thing that would have to be wrong.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels-cuda-new/src");
    let mut files = Vec::new();
    rust_sources(&dir, &mut files);
    let mut out = BTreeSet::new();
    for p in files {
        // Comments off first. The walk reaches `families/`, which ARGUES
        // about sources in prose — `ssm.rs` says a `Source::Gdn("group_size")`
        // "would have made it a coincidence" — and a scan over raw text reads
        // the road not taken as a name the launch owes a field for.
        let source = strip_line_comments(&std::fs::read_to_string(&p).expect("read"));
        let s = source.as_str();
        for tag in [
            "Source::Ctx(\"",
            "Source::CtxNonZero(\"",
            "Source::CtxByLayer(\"",
            "Source::Attn(\"",
            "Source::AttnNonZero(\"",
            "Source::Gdn(\"",
            "Source::GdnSlab(\"",
        ] {
            let mut rest = s;
            while let Some(i) = rest.find(tag) {
                rest = &rest[i + tag.len()..];
                let Some(j) = rest.find('"') else { break };
                let name = &rest[..j];
                if !name.contains('[') {
                    out.insert(name.to_string());
                }
            }
        }
    }
    assert!(
        out.len() > 15,
        "found only {} Ctx names — the scan broke",
        out.len()
    );
    out
}

#[test]
fn every_context_scalar_is_read_off_the_model_or_argued_for() {
    let launch = read("src/fire/launch.rs");
    // Both lists excuse a literal; they differ in what they claim, not
    // in what they permit.
    let argued: BTreeSet<&str> = CONSTANT_BY_ARGUMENT
        .iter()
        .map(|(n, _)| *n)
        .chain(VARIED_BY_A_ROW_WITH_NO_TEXT.iter().map(|(n, ..)| *n))
        .collect();

    let mut unstated = BTreeSet::new();
    let mut unlocated = BTreeSet::new();
    for name in ctx_names() {
        // The launch's own construction: `name: <rhs>,` at any indent.
        // A name it cannot find is SKIPPED and recorded, not passed:
        // silently continuing is how `n_groups: 0` sat in `GdnCtx`
        // through two versions of this test, at twelve spaces rather
        // than eight.
        let Some(i) = indented(&launch, &name) else {
            unlocated.insert(name.clone());
            continue;
        };
        let needle = format!("{name}: ");
        let rhs = &launch[i + needle.len()..];
        let rhs = &rhs[..rhs.find('\n').unwrap_or(rhs.len())];
        if is_literal(rhs) && !argued.contains(name.as_str()) {
            unstated.insert(format!("{name} = {}", rhs.trim()));
        }
    }

    let strays: Vec<&String> = unlocated
        .iter()
        .filter(|n| !ACCESSOR_NOT_FIELD.contains(&n.as_str()))
        .collect();
    assert!(
        strays.is_empty(),
        "this test could not find where the launch fills these, so it \
         has been saying nothing about them:\n  {strays:?}\n\nEither they \
         are `Source::CtxByLayer`-style ACCESSORS rather than struct \
         fields — add them to `ACCESSOR_NOT_FIELD` — or the launch \
         spells them somewhere this scan does not look, which is \
         exactly the gap `n_groups` hid in.",
    );

    assert!(
        unstated.is_empty(),
        "a kernel reads these off the launch context and the launch \
         invents them:\n  {}\n\nEither read the field off `model.deployment` \
         — which is where `eps`, `vocab`, `moe_norm_topk` and the rest of \
         this struct come from — or add the name to \
         `CONSTANT_BY_ARGUMENT` with the reason no checkpoint varies it, \
         or, if one DOES and merely cannot dispatch yet, to \
         `VARIED_BY_A_ROW_WITH_NO_TEXT` with the row and the fact that \
         holds it back.",
        unstated.into_iter().collect::<Vec<_>>().join("\n  "),
    );
}

/// The argued-for list does not outlive the names it argues about.
///
/// A `Source::Ctx` that disappears takes its entry with it, or the entry
/// becomes an argument about nothing — which reads, to the next person,
/// as a live constant somebody thought about.
#[test]
fn nothing_is_argued_for_that_no_kernel_reads() {
    let names = ctx_names();
    let stale: Vec<&str> = CONSTANT_BY_ARGUMENT
        .iter()
        .map(|(n, _)| *n)
        .chain(VARIED_BY_A_ROW_WITH_NO_TEXT.iter().map(|(n, ..)| *n))
        .filter(|n| !names.contains(*n))
        .collect();
    assert!(
        stale.is_empty(),
        "no kernel reads these any more: {stale:?}"
    );
}

/// A deferred literal stops being excused the day its row gets a text.
///
/// The premise is not restated here, it is READ: `model`'s own
/// `facts_are_read` keeps `DECLARED_BUT_UNREAD`, the list of per-backend
/// facts a family declares and no text consumes. While `kimi_k2`'s
/// `rope_yarn_original` sits on it, nothing dispatches the kernel that
/// reads `rope_interleaved` with a K2 ladder. Writing that text means
/// deleting the entry there, which fails this test HERE — which is the
/// point, because the person writing an MLA rope pass is exactly the
/// person who must decide what `rope_interleaved` is.
///
/// The four excuses this crate's siblings deleted last week all named a
/// future reader their own layering ruled out, so they could never
/// expire. This one names a line someone has to delete to finish the
/// work it defers.
#[test]
fn the_deferred_names_are_still_out_of_reach() {
    let unread = read("../model/tests/facts_are_read.rs");
    let table = unread
        .find("DECLARED_BUT_UNREAD")
        .map(|i| &unread[i..])
        .expect("`model` still keeps a `DECLARED_BUT_UNREAD` list");

    for (name, family, fact, _) in VARIED_BY_A_ROW_WITH_NO_TEXT {
        let entry = table
            .find(&format!("(\"{family}\""))
            .map(|i| &table[i..])
            .and_then(|e| e.find("])").map(|j| &e[..j]));
        assert!(
            entry.is_some_and(|e| e.contains(&format!("\"{fact}\""))),
            "`{name}` is a literal in `fire::launch` because {family} \
             cannot dispatch the kernel that reads it, and the proof of \
             that was `{fact}` sitting in `DECLARED_BUT_UNREAD`. It is \
             not there now. Either a text reads it — in which case fill \
             `{name}` off the model, because that text's checkpoint \
             varies it — or the fact was deleted, in which case say so \
             here.",
        );
    }
}

/// Names that are METHODS on the context, not fields of a literal.
///
/// `Source::CtxByLayer("theta")` lowers to `ctx.theta(layer)` — a
/// per-layer accessor, so there is no assignment to inspect and no
/// literal to catch. They are listed rather than skipped so that a name
/// which stops being an accessor stops being excused.
const ACCESSOR_NOT_FIELD: &[&str] = &["theta", "altup_std_mult"];

/// The launch's assignment of `name` at any indent: `name: ` or the
/// shorthand `name,`.
///
/// Shorthand counts as stated: `sm_scale,` names a local the launch
/// computed, which is the opposite of inventing one.
fn indented(launch: &str, name: &str) -> Option<usize> {
    let mut rest = launch;
    let mut base = 0usize;
    loop {
        let i = rest.find(name)?;
        let pre = &rest[..i];
        let after = &rest[i + name.len()..];
        let at_start = pre
            .rsplit('\n')
            .next()
            .is_some_and(|l| l.chars().all(char::is_whitespace) && !l.is_empty());
        if at_start && (after.starts_with(": ") || after.starts_with(',')) {
            return Some(base + i);
        }
        base += i + name.len();
        rest = after;
    }
}
