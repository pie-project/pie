//! Every builder in `model_dsl::kernels` IS the point its declaration
//! states — checked, not assumed.
//!
//! THE PAIR THIS IMITATES is `model-dsl-legacy`'s retired
//! `tests/generator/mod.rs` + `tests/wrappers_are_current.rs`: a generator
//! writes the surface it expects and a test diffs it against what is
//! committed. The one difference is the DIRECTION. There the file was
//! generated and the test refused a stale copy; here `src/kernels.rs` is
//! still hand-written — texts read it, and it carries prose the tables have
//! no column for — so this test generates the builder each `*_POINTS` row
//! implies and refuses a builder that has DRIFTED off its declaration. When
//! the generator lands for real, the expected string below becomes the
//! emitted file and this test becomes `..._are_current`; nothing else moves.
//!
//! WHY A STRING AND NOT REFLECTION: nothing at run time can look at a Rust
//! fn's parameter list. So the check generates the source it expects, parses
//! both sides with `syn`, strips attributes (the prose is the hand file's to
//! keep), and compares TOKEN STREAMS — whitespace, line breaks and rustfmt's
//! opinions cannot make this test fail or pass.
//!
//! THE MAPPING, which is `.wiki/baker.md`'s and is restated here because
//! this file is where it is now enforced:
//!
//! * the fn's name is the method's, its module is the family's, and the
//!   recorded string is the point's path, verbatim;
//! * the RECEIVER is the first slot in declaration order that rides the
//!   statement's operand column — the first `In` or `InOut`. (baker.md says
//!   "the `InOut` slot when one exists and the first `In` otherwise", which
//!   is the same rule everywhere except `norm.residual_add`, whose `InOut`
//!   is its SECOND slot; the plan's `inputs` column is in declaration order,
//!   so the first operand is the only receiver that keeps it that way.);
//! * every other slot is a parameter AT ITS DECLARED POSITION — `Out` slots
//!   are the only ones that are not, because they are the return;
//! * further `In`/`InOut` → `.value`, `Const` → `.weight`, `Cache` →
//!   `.cache`, bare `u32` → `.int`, bare `f32` → `.float`, bare `bool` →
//!   `.int(u32::from(..))`;
//! * `Out` + `InOut` count is the return: 0 → `.effect()` and no value, 1 →
//!   `.done() -> Value`, 2 → `.pair()`, 3 → `.triple()`;
//! * a point whose `Const` slots spell a weight repr is generic over it,
//!   `<W: Dtype>`. The declaration's own dtype column does not reach here:
//!   `Dtype` on this surface is the CHECKPOINT's repr axis (bf16 / mxfp4 /
//!   wna16) and the declaration's is the kernel's ELEMENT (`f32`, or the
//!   method's `T`). There is no `Dtype` for `f32`, so a `Const` pinned to
//!   `Self::Tensor<f32>` — `moe.topk_sqrt_softplus`, `hc.gates`,
//!   `hc.collapse`, `norm.rmsnorm_gated{,_by}` — still takes a
//!   `&Tensor<W>` here. That is a SURFACE GAP, not drift. The slots that
//!   have crossed already — the recurrent decay pair, `a_log` and
//!   `dt_bias` — take `&Tensor<F32>`, because the models declare them that
//!   way; `ssm.kda_step` and `ssm.kda_chunked` spell BOTH that way and are
//!   therefore the two points with a `Const` and no `<W: Dtype>` at all.
//!   `attention.sink` LEFT THAT LIST BY BEING FIXED rather than by being
//!   spelled: its sink slot rides the point's own element now, because
//!   that is what the checkpoints ship it at, so the `&Tensor<W>` the hand
//!   surface always wrote is the mapping's own answer and no longer a gap.
//! * a `Const` slot at `Dtype::Bank(..)` is a QUANTISED bank: it rides the
//!   point's repr axis rather than its element one, takes `&Tensor<W>` all
//!   the same (the DSL's `Dtype` IS the repr axis) and records with
//!   `.bank(..)`, which is as many weight columns as the repr stores planes.
//!   A point with `Const` slots of BOTH kinds spells two repr parameters —
//!   `W` for the kind its slot list shows first, `B` for the other — which
//!   is `moe.matmul_select_bias` and, today, only that.
//!
//! [`EXCEPTIONS`] carries every place the hand surface deliberately says
//! something the table cannot, one row per point with the reason on it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use kernels::points::{
    ATTENTION_POINTS, DIST_POINTS, Dtype, GATE_POINTS, GEMM_POINTS, HC_POINTS, INDEX_POINTS,
    LAYOUT_POINTS, MLA_POINTS, MLP_POINTS, MOE_POINTS, Mark, NORM_POINTS, POOL_POINTS, Point,
    Prim, ROPE_POINTS, SSM_POINTS,
};
use quote::ToTokens;

/// Every family's table, in the order `kernels/src/points.rs` declares them.
fn points() -> Vec<&'static Point> {
    NORM_POINTS
        .iter()
        .chain(MLP_POINTS)
        .chain(GEMM_POINTS)
        .chain(DIST_POINTS)
        .chain(ROPE_POINTS)
        .chain(MOE_POINTS)
        .chain(GATE_POINTS)
        .chain(LAYOUT_POINTS)
        .chain(SSM_POINTS)
        .chain(ATTENTION_POINTS)
        .chain(MLA_POINTS)
        .chain(INDEX_POINTS)
        .chain(POOL_POINTS)
        .chain(HC_POINTS)
        .collect()
}

// ── The exceptions ──────────────────────────────────────────────────────

/// One thing the hand surface says that the mapping alone does not.
///
/// EVERY ONE IS RECORD-IDENTICAL. An exception changes what a TEXT writes,
/// never what the plan carries: `.window(w)` is `.int(w.unwrap_or(0))`,
/// `.norm(n)` is `.weight(&n.weight).float(n.eps)`, a `&Windows` is the two
/// values the ragged pairing already holds as one. That is the bar for
/// adding a row here — if it moved a column, it would be drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Except {
    /// The point's first two `In` slots are the ragged pair (the rows, then
    /// the request boundaries) and the text holds them as ONE value. The fn
    /// takes `w: &Windows` and unpacks it, rather than asking a text for two
    /// halves of something it never split.
    Windows,

    /// `window: u32` is `Option<u32>` here and rides `.window(..)`, which IS
    /// `.int(w.unwrap_or(0))`. The text says "no window" and the plan
    /// carries the zero the driver reads; neither spelling moved when the
    /// attention family landed.
    WindowOpt,

    /// A `(Const weight, f32 eps)` pair is one `&Norm<W>` here, recorded
    /// with `.norm(..)` = `.weight(&n.weight).float(n.eps)`. A norm is one
    /// thing in every text that has one, and the two halves land in two
    /// different columns of the statement either way. The `eps` slot leaves
    /// the parameter list wherever it stands — `index.layernorm_rope`
    /// declares it two slots after its weight.
    NormBundle,

    /// `up_cap` is optional in the text and rides a `0.0` SENTINEL in the
    /// statement, because the declaration has no way to say "absent" about a
    /// bare scalar. baker-todo's "Option/0-sentinel params encoding
    /// decision" is this row; when presence bits land, it goes.
    UpCap,

    /// `layer` is the statement's own TAG (`Op::layer`), not a param —
    /// `.layer(l)` rather than `.int(l)`. That is where the driver has
    /// always read an attention landing's layer, and moving it into the
    /// params run would change every plan for nothing. `layout.select`
    /// declares a `layer` too and it IS a param there, which is why this is
    /// a per-point row and not a rule about the name.
    LayerTag,

    /// `norm.res_blend`'s `blocks` is `&[Value]`: it grows by one every
    /// layer that blends, where the declaration states the single
    /// concatenated rectangle its routine takes. THE ONE VARIADIC BUILDER on
    /// this surface, and the declaration names it an open ledger item.
    Blend,
}

/// The points that say something the mapping does not, and what.
///
/// `moe.matmul_select_bias` USED TO BE A ROW HERE, under a `TwoBanks`
/// exception: it spelled two repr parameters where the declaration rode both
/// its banks on one element axis `T`, and the note said the more truthful
/// side was the hand file's. The floor's `Bank<R: Repr>` closed that — the
/// point now quantifies over an element axis AND a repr axis, its bank slot
/// carries `Self::Bank<R>`, and the two parameters the builder spells are
/// what the two axes say. The mapping produces them from the KINDS of the
/// point's `Const` slots, which is the rule below and no longer an exception.
const EXCEPTIONS: &[(&str, &[Except])] = &[
    ("norm.res_blend", &[Except::Blend, Except::NormBundle]),
    ("mlp.situ", &[Except::UpCap]),
    ("gemm.attention_landing", &[Except::LayerTag]),
    ("attention.decode", &[Except::WindowOpt]),
    ("attention.prefill", &[Except::Windows, Except::WindowOpt]),
    ("attention.masked", &[Except::Windows, Except::WindowOpt]),
    ("attention.decode_lse", &[Except::WindowOpt]),
    ("attention.prefill_lse", &[Except::Windows, Except::WindowOpt]),
    ("ssm.causal_conv1d_chunked", &[Except::Windows]),
    ("ssm.gated_delta_chunked", &[Except::Windows]),
    ("ssm.kda_chunked", &[Except::Windows]),
    ("mla.latents", &[Except::NormBundle]),
    ("mla.latents_rope", &[Except::NormBundle]),
    ("mla.attention_prefill", &[Except::Windows]),
    ("mla.attention_prefill_selected", &[Except::Windows]),
    ("index.layernorm_rope", &[Except::NormBundle]),
    ("pool.boundary_prefill", &[Except::Windows]),
];

fn excepted(point: &str) -> &'static [Except] {
    EXCEPTIONS
        .iter()
        .find(|(n, _)| *n == point)
        .map_or(&[][..], |(_, e)| *e)
}

/// Whether the point's FIRST `Const` slot is a bank, which is what decides
/// whether `W` names the repr axis or the element one.
///
/// The `a_log`/`dt_bias` pair below is excluded for the reason it is excluded
/// there: those two spell `axes::F32` outright and claim neither name.
fn first_kind_is_bank(p: &Point) -> bool {
    p.slots
        .iter()
        .filter(|s| s.mark == Mark::Const)
        .filter(|s| !matches!((s.dtype, s.name), (Dtype::Fixed(Prim::F32), "a_log" | "dt_bias")))
        .map(|s| matches!(s.dtype, Dtype::Bank(_)))
        .next()
        .unwrap_or(false)
}

/// WHICH POOL a `Cache` slot names, which the point table cannot say.
///
/// `Mark::Cache` carries `Dtype::Opaque` for both pools — a pool row's
/// element was decided when the slab was allocated and no method quantifies
/// over it — so the table records THAT a slot is a cache row and not WHICH
/// of the floor's two associated types it rides. The recurrent slabs are
/// `Ssm`'s and the paged KV is everyone else's; the DSL spells them `&State`
/// and `&Pages`, and this fn is where the missing column is stood in for. A
/// `Mark::Cache(Pool)` on the floor would retire it.
fn pool_type(family: &str) -> &'static str {
    if family == "ssm" { "&State" } else { "&Pages" }
}

// ── The generator ───────────────────────────────────────────────────────

/// The builder `p` implies, as source.
fn expected(p: &Point) -> String {
    let (family, method) = p.name.split_once('.').expect("a point's name is `family.method`");
    let ex = excepted(p.name);
    let has = |e: Except| ex.contains(&e);

    // A REPR PARAMETER IS DECLARED WHEN A SLOT USES IT, which is not the
    // same as "the point has a `Const`". `ssm.kda_step` has two and both
    // land on `axes::F32`, so a `<W: Dtype>` on it would be a type
    // parameter no argument mentions — which does not compile, and is
    // therefore the one place this generator could write a builder the
    // hand file could not be. The `Const` loop below records what it
    // actually spelled.
    let mut spelled_w = false;
    let mut spelled_b = false;

    let mut params: Vec<String> = Vec::new();
    let mut chain: Vec<String> = Vec::new();
    let mut receiver: Option<String> = None;
    // `NormBundle` folds the `eps` that follows the weight into the bundle,
    // so the scalar is dropped from the parameter list wherever it stands.
    let mut eat_eps = false;

    for (i, s) in p.slots.iter().enumerate() {
        let name = s.name;
        if has(Except::Windows) && i == 0 {
            params.push("w: &Windows".into());
            receiver = Some("w.data".into());
            continue;
        }
        if has(Except::Windows) && i == 1 {
            chain.push(".value(&w.indptr)".into());
            continue;
        }
        match s.mark {
            Mark::In | Mark::InOut => {
                if has(Except::Blend) && name == "blocks" {
                    params.push("blocks: &[Value]".into());
                    continue;
                }
                params.push(format!("{name}: &Value"));
                if receiver.is_none() {
                    receiver = Some(name.to_string());
                } else {
                    chain.push(format!(".value({name})"));
                }
            }
            Mark::Const => {
                if has(Except::NormBundle) && name == "weight" {
                    // A bundle is a weight too: `&Norm<W>` spells the repr
                    // parameter just as `&Tensor<W>` does.
                    spelled_w = true;
                    params.push("norm: &Norm<W>".into());
                    chain.push(".norm(norm)".into());
                    eat_eps = true;
                    continue;
                }
                // A Const pinned to a fixed element takes the matching
                // AXIS type where one exists — `axes::F32` landed with the
                // a_log/gdn_norm truth. The generic `W` remains the
                // documented stand-in for the fixed-f32 slots whose model
                // structs still ride the W1 axis (the repr-vs-element gap).
                //
                // TWO NAMES, and the second is `ssm.kda_*`'s half of the
                // same pair. `ssm/kda.cuh`'s `kda_gate_beta` takes `A_log`
                // AND `dt_bias` as `const float*`, both points declare both
                // slots `Const<Self::Tensor<f32>>`, and `Kda` in
                // `model/src/kimi_k3/model.rs` declares both `Tensor<F32>`
                // — so `W` here would be the lie rather than the stand-in.
                // `ssm.gdn_prep`'s `dt_bias` never reaches this arm: qwen's
                // kernel reads it at the model's element, so that slot is
                // `Generic(0)` and falls through to `W` on its own.
                //
                // THE PARAMETER IS CHOSEN BY THE SLOT'S KIND, and a point
                // whose `Const` slots are of BOTH kinds spells two. A
                // `Self::Bank<R>` slot rides the point's REPR axis — the
                // storage form its bytes are in — and an element `Const`
                // rides the activation's; `moe.matmul_select_bias` has one of
                // each and gpt-oss instantiates them apart (an mxfp4 stack, a
                // bf16 bias). `W` goes to the first kind the slot list shows
                // and `B` to the second, which for that point is the bank
                // then the bias, and for every point with one kind is `W`
                // throughout.
                let banked = matches!(s.dtype, Dtype::Bank(_));
                let repr = match s.dtype {
                    Dtype::Fixed(Prim::F32) if matches!(name, "a_log" | "dt_bias") => "F32",
                    _ if banked == first_kind_is_bank(p) => {
                        spelled_w = true;
                        "W"
                    }
                    _ => {
                        spelled_b = true;
                        "B"
                    }
                };
                params.push(format!("{name}: &Tensor<{repr}>"));
                // `.bank(..)` and not `.weight(..)`: a bank at a quantised
                // repr records `Repr::PLANES` weight columns, and the
                // declaration's slot is what says which verb applies.
                chain.push(format!(
                    ".{}({name})",
                    if banked { "bank" } else { "weight" }
                ));
            }
            Mark::Cache => {
                params.push(format!("{name}: {}", pool_type(family)));
                chain.push(format!(".cache(&{name}.name)"));
            }
            Mark::Out => {}
            Mark::Scalar => {
                if eat_eps && name == "eps" {
                    eat_eps = false;
                    continue;
                }
                if has(Except::WindowOpt) && name == "window" {
                    params.push("window: Option<u32>".into());
                    chain.push(".window(window)".into());
                    continue;
                }
                if has(Except::UpCap) && name == "up_cap" {
                    params.push("up_cap: Option<f32>".into());
                    chain.push(".float(up_cap.unwrap_or(0.0))".into());
                    continue;
                }
                if has(Except::LayerTag) && name == "layer" {
                    params.push("layer: u32".into());
                    chain.push(".layer(layer)".into());
                    continue;
                }
                match s.dtype {
                    Dtype::Fixed(Prim::F32) => {
                        params.push(format!("{name}: f32"));
                        chain.push(format!(".float({name})"));
                    }
                    Dtype::Fixed(Prim::U32) => {
                        params.push(format!("{name}: u32"));
                        chain.push(format!(".int({name})"));
                    }
                    Dtype::Fixed(Prim::I32) => {
                        params.push(format!("{name}: i32"));
                        chain.push(format!(".int({name} as u32)"));
                    }
                    Dtype::Fixed(Prim::Bool) => {
                        params.push(format!("{name}: bool"));
                        chain.push(format!(".int(u32::from({name}))"));
                    }
                    other => panic!("`{}`: slot `{name}` is a scalar riding {other:?}", p.name),
                }
            }
        }
    }

    let results = p
        .slots
        .iter()
        .filter(|s| matches!(s.mark, Mark::Out | Mark::InOut))
        .count();
    let (answers, finish) = match results {
        0 => (String::new(), ".effect();"),
        1 => (" -> Value".into(), ".done()"),
        2 => (" -> (Value, Value)".into(), ".pair()"),
        3 => (" -> (Value, Value, Value)".into(), ".triple()"),
        n => panic!("`{}` states {n} results and the recorder tops out at three", p.name),
    };

    let recv = receiver.unwrap_or_else(|| {
        panic!("`{}` rides no operand column: nothing to record the statement on", p.name)
    });
    let generic = match (spelled_w, spelled_b) {
        (true, true) => "<W: Dtype, B: Dtype>",
        (true, false) => "<W: Dtype>",
        // A second kind with no first is not a shape this mapping can
        // produce: `W` goes to whichever kind the slot list shows FIRST, so
        // a point that spells `B` has already spelled `W`.
        (false, true) => panic!("`{}` spells `B` with no `W`", p.name),
        (false, false) => "",
    };
    let head = format!("pub fn {method}{generic}({}){answers}", params.join(", "));
    let stmt = format!("{recv}.stmt(\"{}\")", p.name);

    // The one variadic: the blocks run cannot be a link in a chain.
    if has(Except::Blend) {
        let rest = chain.join("");
        return format!(
            "{head} {{ let mut s = {stmt}; for block in blocks {{ s = s.value(block); }} \
             s{rest}{finish} }}"
        );
    }
    format!("{head} {{ {stmt}{}{finish} }}", chain.join(""))
}

// ── The hand surface, as parsed ─────────────────────────────────────────

/// `src/kernels.rs`, as `(module, fn name) -> the fn`. A top-level fn wears
/// the empty module.
fn surface() -> BTreeMap<(String, String), syn::ItemFn> {
    let at = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/kernels.rs");
    let text = std::fs::read_to_string(&at)
        .unwrap_or_else(|e| panic!("reading {}: {e}", at.display()));
    let file: syn::File = syn::parse_str(&text)
        .unwrap_or_else(|e| panic!("parsing {}: {e}", at.display()));
    let mut out = BTreeMap::new();
    for item in &file.items {
        match item {
            syn::Item::Fn(f) => {
                out.insert((String::new(), f.sig.ident.to_string()), f.clone());
            }
            syn::Item::Mod(m) => {
                let family = m.ident.to_string();
                for inner in m.content.iter().flat_map(|(_, items)| items) {
                    if let syn::Item::Fn(f) = inner {
                        out.insert((family.clone(), f.sig.ident.to_string()), f.clone());
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// The fn with its prose taken off: the tables have no column for a doc
/// comment and the hand file is right to carry them.
fn bare(f: &syn::ItemFn) -> String {
    let mut f = f.clone();
    f.attrs.clear();
    for arg in &mut f.sig.inputs {
        if let syn::FnArg::Typed(t) = arg {
            t.attrs.clear();
        }
    }
    // rustfmt writes a trailing comma when it breaks a parameter list over
    // lines, and that comma IS a token. It says nothing about the surface.
    while f.sig.inputs.trailing_punct() {
        f.sig.inputs.pop_punct();
    }
    f.to_token_stream().to_string()
}

fn normalized(source: &str) -> String {
    let f: syn::ItemFn = syn::parse_str(source)
        .unwrap_or_else(|e| panic!("the generator wrote something unparseable: {e}\n{source}"));
    bare(&f)
}

// ── The check ───────────────────────────────────────────────────────────

#[test]
fn builders_are_the_points() {
    let have = surface();
    let mut drift: Vec<String> = Vec::new();
    let mut claimed: BTreeSet<(String, String)> = BTreeSet::new();

    for p in points() {
        let (family, method) = p.name.split_once('.').expect("`family.method`");
        let key = (family.to_string(), method.to_string());
        claimed.insert(key.clone());
        let want = expected(p);
        let Some(f) = have.get(&key) else {
            drift.push(format!(
                "MISSING  {}\n         no `model_dsl::kernels::{family}::{method}`; the point \
                 implies\n           {want}",
                p.name
            ));
            continue;
        };
        let (want, got) = (normalized(&want), bare(f));
        if want != got {
            drift.push(format!(
                "DRIFTED  {}\n  states  {want}\n  writes  {got}",
                p.name
            ));
        }
    }

    // A builder no table names. `cuda::*` is tier-2 by construction (an
    // inherent method on the plane's `Ctx`, which no trait and therefore no
    // `#[points]` table can see) and `query_windows` is the ragged pairing's
    // constructor, not a statement.
    for (family, method) in have.keys() {
        if claimed.contains(&(family.clone(), method.clone())) || family == "cuda" {
            continue;
        }
        if family.is_empty() && method == "query_windows" {
            continue;
        }
        let path = if family.is_empty() {
            format!("model_dsl::kernels::{method}")
        } else {
            format!("model_dsl::kernels::{family}::{method}")
        };
        drift.push(format!("UNDECLARED  {path} records a statement no `*_POINTS` row states"));
    }

    assert!(
        drift.is_empty(),
        "{} builder(s) disagree with the declaration that states them.\n\
         The tables in `kernels/src/points.rs` are the source of truth: fix the\n\
         builder, or — if the hand form says something true the table cannot —\n\
         add a row to `EXCEPTIONS` in this file with the reason on it.\n\n{}",
        drift.len(),
        drift.join("\n\n")
    );
}

/// The tier-2 surface has NO declaration to check against, and that is the
/// shape of the gap rather than an oversight.
///
/// A tier-2 point is an inherent method on one plane's `Ctx` — no trait, so
/// `#[points]` never sees it and no `*_POINTS` row exists to generate from.
/// `model_dsl::kernels::cuda` is therefore hand-written and stays that way
/// until the floor grows a way to declare a one-plane surface. This test
/// records what is in there so the list cannot grow silently.
#[test]
fn tier_two_is_unchecked_and_small() {
    let have = surface();
    let tier2: Vec<&str> = have
        .keys()
        .filter(|(family, _)| family == "cuda")
        .map(|(_, method)| method.as_str())
        .collect();
    assert_eq!(
        tier2,
        vec!["qkv_fused_qknorm_rope_vnorm_write"],
        "the tier-2 builders are hand-written against no table; a new one is a \
         decision, not a mapping"
    );
}
