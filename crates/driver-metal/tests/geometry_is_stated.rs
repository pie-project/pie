//! Every field of the geometry is filled by something, or says what.
//!
//! `DecodeGeometry` is what the whole driver reads a fire's shape off.
//! `geometry_from_deployment` fills the half a catalog row states; its
//! doc says the caller fills the rest — "the capacity fields
//! (`max_tokens`, `max_requests`, `max_slots`, the paging trio) because
//! those are the OPERATOR's numbers ... and `alt_quant`/`mxfp4_experts`
//! because those are solved from the staged tensors".
//!
//! That sentence was a plan, not a description. Nothing set those
//! fields and nothing read them, so a `Default` answered questions the
//! driver never asked — and while the doc read as a division of labour,
//! it was one end of a contract with no other end.
//!
//! Two fields had already gone the whole way: `mrope_section` and
//! `norm_topk_prob` were declared, defaulted, and read by nobody at all,
//! and the second one MATTERED — the routing denominator it named
//! reaches a Metal shader as a word of `RouterParams`, stated by the
//! row, and would have been silently answered here if anything had
//! thought to look.
//!
//! So this holds the struct to a rule with two halves: a field is
//! STATED by `geometry_from_deployment`, or it is named below with what
//! fills it and when — AND a field is READ by something in this crate,
//! or named as unread.
//!
//! The second half is not the first one twice. Five fields passed the
//! first and failed the second: `final_logit_softcap`,
//! `attention_k_eq_v`, `per_layer_emb_dim`, and the
//! `swiglu_limit`/`swiglu_alpha` pair were each filled correctly from a
//! `Deployment` and read by nobody, because the quantity they name
//! reaches the Metal text by a DIFFERENT road — `LlamaLikeMetalFacts`
//! carries `logit_softcap`, `v_from_k`, `per_layer_emb_dim` and
//! `Activation::SwiGlu { limit, alpha }`, and the text reads those. Two
//! readings of one quantity, with only one of them live: the same shape
//! as the deployment/trace seam and the `MetalRow` merge. They are
//! deleted, and this test is why the next one cannot arrive quietly.

use std::collections::BTreeSet;

/// Fields `geometry_from_deployment` deliberately leaves, and who fills
/// them.
///
/// An entry is a claim that something else answers this. A field with no
/// filler is not "the operator's" — it is a `Default` nobody chose.
const FILLED_ELSEWHERE: &[(&str, &str)] = &[
    (
        "alt_quant",
        "NOT WIRED. The checkpoint's SECOND affine point: mlx_lm quantizes \
         per tensor and spares the two that decide where a token goes, so a \
         routed checkpoint ships an 8-bit router gate inside a 4-bit stack. \
         Reading it at the stack's width produced cosine 0.84 logits rather \
         than an error. `has_alt_quant()` exists to be asked and nothing \
         asks it, because no load path solves the format from the staged \
         tensors yet. The field stays so the fix has a place to land, and \
         this entry stays so nobody reads the absence as a decision.",
    ),
    (
        "mxfp4_experts",
        "NOT WIRED, as `alt_quant`. Whether the expert bank stays in the \
         checkpoint's MXFP4 rather than being re-quantized at load, which \
         changes what is BOUND — MXFP4 has block exponents and no zero \
         point, so there is no `.biases` to bind. No gpt-oss row is served \
         on Metal yet (`manifest.rs` cannot identify MLX's split \
         publication), so nothing has had to answer.",
    ),
];

fn source() -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/batch/geometry.rs");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The braces-matched body of an item starting at `from`.
fn block(s: &str, from: usize) -> &str {
    let o = s[from..].find('{').expect("a block") + from;
    let (mut d, mut e) = (0usize, o);
    for (i, c) in s[o..].char_indices() {
        match c {
            '{' => d += 1,
            '}' => {
                d -= 1;
                if d == 0 {
                    e = o + i;
                    break;
                }
            }
            _ => {}
        }
    }
    &s[o..=e]
}

fn declared(s: &str) -> BTreeSet<String> {
    let b = block(s, s.find("pub struct DecodeGeometry").expect("the struct"));
    b.lines()
        .filter_map(|l| l.trim().strip_prefix("pub "))
        .filter_map(|l| l.split(':').next())
        .map(str::to_string)
        .collect()
}

fn stated(s: &str, names: &BTreeSet<String>) -> BTreeSet<String> {
    let b = block(
        s,
        s.find("fn geometry_from_deployment").expect("the builder"),
    );
    names
        .iter()
        .filter(|n| {
            b.lines().any(|l| {
                let t = l.trim();
                // `name: <expr>,` or the shorthand `name,`
                t.strip_prefix(n.as_str())
                    .is_some_and(|r| r.starts_with(':') || r == ",")
            })
        })
        .cloned()
        .collect()
}

#[test]
fn every_geometry_field_is_stated_or_accounted_for() {
    let s = source();
    let names = declared(&s);
    assert!(
        names.len() > 35,
        "found only {} fields — the scan broke",
        names.len()
    );

    let stated = stated(&s, &names);
    let accounted: BTreeSet<&str> = FILLED_ELSEWHERE.iter().map(|(n, _)| *n).collect();

    let orphans: Vec<&String> = names
        .iter()
        .filter(|n| !stated.contains(*n) && !accounted.contains(n.as_str()))
        .collect();

    assert!(
        orphans.is_empty(),
        "these fields fall through to `Default` and nothing accounts for \
         them:\n  {orphans:?}\n\nA `DecodeGeometry` field is what a kernel \
         reads a shape off. Either state it in `geometry_from_deployment` \
         from the `Deployment` that describes it, delete it if no kernel \
         reads it, or name it in `FILLED_ELSEWHERE` with what fills it.",
    );
}

/// The account does not outlive the field it accounts for.
#[test]
fn nothing_is_accounted_for_that_the_struct_no_longer_declares() {
    let names = declared(&source());
    let stale: Vec<&str> = FILLED_ELSEWHERE
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| !names.contains(*n))
        .collect();
    assert!(stale.is_empty(), "no such field any more: {stale:?}");
}

/// Fields no kernel and no lowering reads, and why they are still here.
///
/// A `DecodeGeometry` field exists to be read. One that nothing reads
/// is either a quantity that already travels another road — in which
/// case there are now two answers to one question and only one of them
/// is live — or a plan.
const DECLARED_BUT_UNREAD: &[(&str, &str)] = &[(
    "mxfp4_experts",
    "The same gap `FILLED_ELSEWHERE` names: nothing states it because \
     nothing solves the format from the staged tensors, and nothing \
     reads it because no gpt-oss row is served on Metal. Both ends \
     unwritten, which is why it is on both lists.",
)];

/// Every `.field` read anywhere under `src/`.
///
/// `batch/geometry.rs` counts. An accessor beside the declaration —
/// `has_alt_quant()` reading `alt_quant` — is the crate's interface to
/// the field, and a field reached only through one is reached.
fn read_names() -> BTreeSet<String> {
    fn walk(d: &std::path::Path, out: &mut String) {
        for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push_str(&std::fs::read_to_string(&p).unwrap_or_default());
            }
        }
    }
    let mut all = String::new();
    walk(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut all,
    );
    let mut seen = BTreeSet::new();
    for (i, _) in all.match_indices('.') {
        let r = &all[i + 1..];
        let n: String = r
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !n.is_empty() {
            seen.insert(n);
        }
    }
    seen
}

#[test]
fn every_geometry_field_is_read_by_something() {
    let names = declared(&source());
    let read = read_names();
    let excused: BTreeSet<&str> = DECLARED_BUT_UNREAD.iter().map(|(n, _)| *n).collect();

    let dead: Vec<&String> = names
        .iter()
        .filter(|n| !read.contains(*n) && !excused.contains(n.as_str()))
        .collect();

    assert!(
        dead.is_empty(),
        "these fields are declared and nothing in this crate reads \
         them:\n  {dead:?}\n\nCheck first whether the quantity already \
         reaches the text another way — `LlamaLikeMetalFacts` carries \
         most of what a fire's shape needs, and a second dead copy of a \
         live number is how the wrong one gets picked up later. Delete \
         it, or name it in `DECLARED_BUT_UNREAD` with the pass that will \
         read it.",
    );
}
