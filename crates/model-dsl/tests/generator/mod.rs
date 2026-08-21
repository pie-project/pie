//! The wrapper GENERATOR (design-no-ask §10, B4-gen): parse every traced
//! `#[routine]` under `crates/kernels-cuda/src` and emit
//! `src/cuda/generated.rs` — one named `pub fn` per routine, named by its
//! trace name, recording through `crate::fire::fire::<marker>` so the
//! symbol and every run come off the marker rather than off a respelled
//! string.
//!
//! Deliberately a TEST-SIDE module, not a build script. The emitted file is
//! CHECKED IN, so nothing needs to generate at build time;
//! `wrappers_are_current` regenerates and diffs (the
//! `model-loader/tests/golden_plans.rs` idiom), and `UPDATE_WRAPPERS=1`
//! rewrites. A build.rs writing into `src/` on every build would race
//! parallel builds and break read-only checkouts, and the
//! `rerun-if-changed` plumbing §10 sketched exists only for `OUT_DIR`
//! generation, which this is not.
//!
//! Determinism: files are walked in sorted relative-path order and
//! routines in source order, so one tree generates one byte sequence.
//!
//! Since B4-gen step 6 this module also generates the METAL half
//! (`src/metal/generated.rs` from `crates/kernels-metal/src`) — same
//! contract, two plane-specific readings documented on [`generate_metal`]:
//! the statement's SYMBOL is the plane's instantiated entrypoint rather
//! than `namespace::name`, and a trailing `Const<i32>` named `rows` is a
//! spliced fire extent rather than an argument.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The mark → argument mapping, one to one (§10: runtime streams and views
/// are ordinary arguments; a wrapper mints NOTHING in secret).
#[derive(Debug, Clone, PartialEq)]
enum Mark {
    /// `In<..>` (`Tensor` and `Struct` alike) → `&Val`;
    /// `Option<In<..>>` → `Option<&Val>`, omitted from the run when `None`.
    In { optional: bool },
    /// `InOut<..>` → `&Val` for the operand half; the result half behaves
    /// like an unruled `Out` and takes `{name}_out: (Shape, DType)`.
    InOut,
    /// `Out<..>` → nothing when the routine states an `out(..)` rule;
    /// `(Shape, DType)` when `Unstated`; `Option<(Shape, DType)>` when the
    /// mark itself is optional (then the fn returns `Option<Val>` there).
    Out { optional: bool },
    /// `Const<Tensor<..>>` → `&str` (the weight's name);
    /// `Option<Const<Tensor<..>>>` → `Option<&str>`, omitted when `None`.
    Weight { optional: bool },
    /// A scalar `Const` → the scalar itself, encoded into the params run.
    Scalar(Scalar),
    /// A bare pointer, `MaybeConst<..>` or `#[unbound]` parameter: the ARM
    /// supplies it; no statement and therefore no wrapper argument.
    Unmarked,
}

/// The scalar `Const` carriers and their params-run encodings.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Scalar {
    I32,
    U32,
    U8,
    F32,
    Bool,
    /// Truncating: the params run is `u32` words, so `usize` and `i64`
    /// enter as `as u32` — the same width every hand wrapper carried.
    Usize,
    I64,
}

impl Scalar {
    fn arg_ty(self) -> &'static str {
        match self {
            Scalar::I32 => "i32",
            Scalar::U32 => "u32",
            Scalar::U8 => "u8",
            Scalar::F32 => "f32",
            Scalar::Bool => "bool",
            Scalar::Usize => "usize",
            Scalar::I64 => "i64",
        }
    }

    /// The expression that puts `name` into the `u32` params run.
    /// Signed values use two's complement, exactly as the hand helpers'
    /// doc states; `usize`/`i64` truncate to the run's word.
    fn encode(self, name: &str) -> String {
        match self {
            Scalar::I32 => format!("{name} as u32"),
            Scalar::U32 => name.to_string(),
            Scalar::U8 => format!("u32::from({name})"),
            Scalar::F32 => format!("{name}.to_bits()"),
            Scalar::Bool => format!("u32::from({name})"),
            Scalar::Usize | Scalar::I64 => format!("{name} as u32"),
        }
    }
}

/// One parameter of the routine's signature, classified.
struct Param {
    name: String,
    mark: Mark,
}

/// One `out(target = rule)` from the attribute, names as written.
struct OutSpec {
    target: String,
    rule: RuleExpr,
}

enum RuleExpr {
    Like(String),
    Split(String, String),
    Shaped { rows_of: String, width: WidthExpr },
}

enum WidthExpr {
    Half(String),
    Of(String),
    Weight(String),
    Param(String),
}

/// What the attribute stated, as far as the generator reads it.
#[derive(Default)]
struct Spec {
    facts: Vec<String>,
    namespace: Option<String>,
    outs: Vec<OutSpec>,
}

/// One traced routine, resolved and ready to emit.
struct Routine {
    /// The trace name — the `fn`'s own, no dtype suffix.
    name: String,
    /// `kernels_cuda`-relative module segments, for the marker path.
    mod_segs: Vec<String>,
    /// The trace namespace (`attn`, `mlp`, ...) — the symbol's prefix.
    namespace: String,
    /// The routine's own doc lines, verbatim.
    docs: Vec<String>,
    params: Vec<Param>,
    /// One entry per `Out`/`InOut` slot, in slot order: `Some(rule text)`
    /// when the routine states a rule this generator can carry, `None` for
    /// `Unstated` (the caller supplies the `Shape`).
    out_rules: Vec<Option<String>>,
    /// Fallback notes appended to the doc block (a rule that had to be
    /// treated as `Unstated`, a truncating scalar, ...).
    notes: Vec<String>,
}

/// Parse the walked tree into its traced routines, in emission order.
fn collect(kernels_src: &Path) -> Vec<Routine> {
    let mut files = Vec::new();
    collect_rs(kernels_src, &mut files);
    files.sort();

    let mut routines: Vec<Routine> = Vec::new();
    for file in &files {
        let src = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("reading {}: {e}", file.display()));
        let parsed =
            syn::parse_file(&src).unwrap_or_else(|e| panic!("parsing {}: {e}", file.display()));
        let rel = file
            .strip_prefix(kernels_src)
            .expect("file is under the walked tree");
        let segs = mod_segs(rel);
        walk_items(&parsed.items, &segs, &mut routines, file);
    }
    routines
}

/// The traced routines' `(symbol, fn name)` pairs — what the coverage and
/// no-shadowing pins walk. The symbol is the statement's spelling
/// (`namespace::trace_name`); the fn name is the generated wrapper's.
pub fn traced(kernels_src: &Path) -> Vec<(String, String)> {
    collect(kernels_src)
        .into_iter()
        .map(|r| (format!("{}::{}", r.namespace, r.name), r.name))
        .collect()
}

/// Generate the whole `src/cuda/generated.rs` from the kernels tree.
pub fn generate(kernels_src: &Path) -> String {
    let routines = collect(kernels_src);

    // The generated module is FLAT, so trace names must be unique and must
    // be legal item names. Both hold today; a routine that breaks either
    // is a decision to surface, not to paper over.
    let mut seen = std::collections::BTreeMap::new();
    for r in &routines {
        if let Some(prior) = seen.insert(r.name.clone(), r.mod_segs.join("::")) {
            panic!(
                "two traced routines share the trace name `{}` \
                 (`{}` and `{}`); the flat generated module cannot hold both",
                r.name,
                prior,
                r.mod_segs.join("::"),
            );
        }
        assert!(
            !is_keyword(&r.name),
            "trace name `{}` is a Rust keyword; the generated fn cannot wear it",
            r.name
        );
    }

    let mut out = String::new();
    out.push_str(HEADER);
    for r in &routines {
        out.push('\n');
        emit(&mut out, r);
    }
    out
}

const HEADER: &str = "\
//! GENERATED — do not edit. One named `pub fn` per traced `#[routine]` in
//! `crates/kernels-cuda/src`, named by its trace name, in sorted-file then
//! source order (design-no-ask §10, B4-gen).
//!
//! The generator is `tests/generator/mod.rs`;
//! `cargo test -p model-dsl --test wrappers_are_current` refuses a stale
//! file and `UPDATE_WRAPPERS=1` rewrites it.
//!
//! Every mark is one argument — runtime streams and views included; a
//! wrapper here mints NOTHING in secret. A result whose routine states an
//! `out(..)` rule is derived at trace time through
//! [`model_ir::kernels::out_shape`]; an `Unstated` result stays a
//! `(Shape, DType)` argument. Trailing `layer` and `state` are the
//! statement's tags, uniformly. Recording goes through
//! [`crate::fire::fire`], so the symbol and the run arities come off the
//! routine's own marker.
#![cfg_attr(rustfmt, rustfmt::skip)]

// The prelude is fixed while the surface below is generated from another
// crate's tree, so any one regeneration may leave part of it unused.
#![allow(unused_imports)]

use kernels::{OutRule, OutWidth};
use model_ir::trace::{DType, Shape, StateRef, ValueId};

use super::ruled_out;
use crate::fire::{Call, fire};
use crate::{Trace, Val};
";

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("listing {}: {e}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("walking {}: {e}", dir.display()))
            .path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// `mlp.rs` → `[mlp]`; `attn/mod.rs` → `[attn]`; `attn/fa2/mod.rs` →
/// `[attn, fa2]`; `lib.rs` → `[]`.
fn mod_segs(rel: &Path) -> Vec<String> {
    let mut segs: Vec<String> = rel
        .iter()
        .map(|s| s.to_string_lossy().into_owned())
        .collect();
    let last = segs.pop().expect("a file path has a file name");
    if last != "lib.rs" && last != "mod.rs" {
        segs.push(last.trim_end_matches(".rs").to_string());
    }
    segs
}

fn walk_items(items: &[syn::Item], segs: &[String], out: &mut Vec<Routine>, file: &Path) {
    for item in items {
        match item {
            syn::Item::Mod(m) => {
                if let Some((_, inner)) = &m.content {
                    let mut deeper = segs.to_vec();
                    deeper.push(m.ident.to_string());
                    walk_items(inner, &deeper, out, file);
                }
            }
            syn::Item::Fn(f) => {
                if let Some(r) = routine_of(f, segs, file) {
                    out.push(r);
                }
            }
            _ => {}
        }
    }
}

fn routine_of(f: &syn::ItemFn, segs: &[String], file: &Path) -> Option<Routine> {
    let attr = f.attrs.iter().find(|a| a.path().is_ident("routine"))?;
    let spec = parse_spec(attr, file);
    // The skipped rows: `untraced`/`uncolumned` have no column for a trace
    // to fill; `internal`/`driver` are reached by the driver, not by a
    // statement a declaration writes.
    for skip in ["untraced", "uncolumned", "internal", "driver"] {
        if spec.facts.iter().any(|f| f == skip) {
            return None;
        }
    }
    let name = f.sig.ident.to_string();
    assert!(
        matches!(f.vis, syn::Visibility::Public(_)),
        "`{name}` ({}) is a traced routine that is not `pub`; \
         its marker cannot be named from model-dsl",
        file.display()
    );
    assert!(
        !f.attrs.iter().any(|a| a.path().is_ident("cfg")),
        "`{name}` ({}) is cfg-gated; its marker may not exist for this \
         crate's feature set",
        file.display()
    );

    let docs = doc_lines(&f.attrs);
    let mut params = Vec::new();
    for (i, arg) in f.sig.inputs.iter().enumerate() {
        if i == 0 {
            continue; // the `Ctx`, which is the backend's, never the statement's
        }
        let syn::FnArg::Typed(pt) = arg else {
            panic!("`{name}` ({}): a routine takes no `self`", file.display())
        };
        let syn::Pat::Ident(pi) = pt.pat.as_ref() else {
            panic!("`{name}` ({}): unnamed routine parameter", file.display())
        };
        let pname = pi.ident.to_string();
        let unbound = pt.attrs.iter().any(|a| a.path().is_ident("unbound"));
        let mark = if unbound {
            Mark::Unmarked
        } else {
            classify(&pt.ty, &name, &pname, file)
        };
        params.push(Param { name: pname, mark });
    }

    let namespace = spec.namespace.clone().unwrap_or_else(|| {
        segs.first()
            .unwrap_or_else(|| {
                panic!(
                    "`{name}` ({}) sits at the crate root with no \
                         `namespace = ..`; the trace prefix is unknowable",
                    file.display()
                )
            })
            .clone()
    });

    let mut notes = Vec::new();
    let out_rules = resolve_rules(&name, &params, &spec.outs, &mut notes, file);
    for p in &params {
        if let Mark::Scalar(s @ (Scalar::Usize | Scalar::I64)) = p.mark {
            notes.push(format!(
                "`{}` is a `{}` and the params run is `u32` words: the \
                 encoding truncates (`as u32`), as every hand wrapper's did.",
                p.name,
                s.arg_ty()
            ));
        }
    }

    Some(Routine {
        name,
        mod_segs: segs.to_vec(),
        namespace,
        docs,
        params,
        out_rules,
        notes,
    })
}

fn doc_lines(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut out = Vec::new();
    for a in attrs {
        if !a.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &a.meta
            && let syn::Expr::Lit(l) = &nv.value
            && let syn::Lit::Str(s) = &l.lit
        {
            for line in s.value().split('\n') {
                out.push(line.to_string());
            }
        }
    }
    out
}

/// Classify one parameter type into its [`Mark`], through one `Option`
/// layer — the same walk `#[routine]`'s own `slot_names` does.
fn classify(ty: &syn::Type, routine: &str, pname: &str, file: &Path) -> Mark {
    fn head(ty: &syn::Type) -> Option<(&syn::PathSegment, Option<&syn::Type>)> {
        let syn::Type::Path(p) = ty else { return None };
        let seg = p.path.segments.last()?;
        let inner = match &seg.arguments {
            syn::PathArguments::AngleBracketed(a) => a.args.iter().find_map(|g| match g {
                syn::GenericArgument::Type(t) => Some(t),
                _ => None,
            }),
            _ => None,
        };
        Some((seg, inner))
    }
    if matches!(ty, syn::Type::Ptr(_)) {
        return Mark::Unmarked; // the stated absence: the arm supplies it
    }
    let Some((seg, inner)) = head(ty) else {
        panic!(
            "`{routine}` ({}): parameter `{pname}` has a type shape this \
             generator does not read",
            file.display()
        )
    };
    let (optional, seg, inner) = if seg.ident == "Option" {
        let Some((s, i)) = inner.and_then(head) else {
            panic!(
                "`{routine}` ({}): `{pname}: Option<..>` wraps no mark",
                file.display()
            )
        };
        (true, s, i)
    } else {
        (false, seg, inner)
    };
    match seg.ident.to_string().as_str() {
        "In" => Mark::In { optional },
        "InOut" => {
            assert!(
                !optional,
                "`{routine}` ({}): `{pname}: Option<InOut<..>>` has no \
                 wrapper spelling",
                file.display()
            );
            Mark::InOut
        }
        "Out" => Mark::Out { optional },
        "Const" => {
            let is_tensor = inner.is_some_and(|t| {
                matches!(t, syn::Type::Path(p)
                    if p.path.segments.last().is_some_and(|s| s.ident == "Tensor"))
            });
            if is_tensor {
                return Mark::Weight { optional };
            }
            assert!(
                !optional,
                "`{routine}` ({}): `{pname}: Option<Const<scalar>>` has no \
                 wrapper spelling",
                file.display()
            );
            let scalar = inner.and_then(|t| match t {
                syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
                _ => None,
            });
            match scalar.as_deref() {
                Some("i32") => Mark::Scalar(Scalar::I32),
                Some("u32") => Mark::Scalar(Scalar::U32),
                Some("u8") => Mark::Scalar(Scalar::U8),
                Some("f32") => Mark::Scalar(Scalar::F32),
                Some("bool") => Mark::Scalar(Scalar::Bool),
                Some("usize") => Mark::Scalar(Scalar::Usize),
                Some("i64") => Mark::Scalar(Scalar::I64),
                other => panic!(
                    "`{routine}` ({}): `{pname}: Const<{}>` is not a carrier \
                     this generator spells",
                    file.display(),
                    other.unwrap_or("?")
                ),
            }
        }
        "MaybeConst" => Mark::Unmarked,
        other => panic!(
            "`{routine}` ({}): parameter `{pname}: {other}<..>` is not a mark",
            file.display()
        ),
    }
}

/// Resolve the attribute's `out(..)` specs against the classified
/// signature, in out-slot order — the same name-to-ordinal walk
/// `#[routine]` performs, so the emitted literals match the row.
fn resolve_rules(
    routine: &str,
    params: &[Param],
    specs: &[OutSpec],
    notes: &mut Vec<String>,
    file: &Path,
) -> Vec<Option<String>> {
    let mut inputs = Vec::new();
    let mut outs = Vec::new();
    let mut weights = Vec::new();
    let mut pruns = Vec::new();
    let mut any_optional_input = false;
    for p in params {
        match &p.mark {
            Mark::In { optional } => {
                any_optional_input |= optional;
                inputs.push(p.name.clone());
            }
            Mark::InOut => {
                inputs.push(p.name.clone());
                outs.push(p.name.clone());
            }
            Mark::Out { .. } => outs.push(p.name.clone()),
            Mark::Weight { .. } => weights.push(p.name.clone()),
            Mark::Scalar(_) => pruns.push(p.name.clone()),
            Mark::Unmarked => {}
        }
    }
    let find = |list: &[String], id: &str, run: &str| -> usize {
        list.iter().position(|n| n == id).unwrap_or_else(|| {
            panic!(
                "`{routine}` ({}): out rule names `{id}`, which is no {run} \
                 slot of the signature",
                file.display()
            )
        })
    };
    let mut rules: Vec<Option<String>> = vec![None; outs.len()];
    for spec in specs {
        let at = find(&outs, &spec.target, "result");
        if any_optional_input {
            // An absent optional operand shifts the statement's input run
            // under the rule's slot-indexed feet. No ruled routine has an
            // optional input today; when one does, the evaluation needs a
            // presence-aware gather, not a guess.
            notes.push(format!(
                "The routine states an `out({} = ..)` rule, but it also has \
                 optional operands, which this generator does not yet index \
                 under a rule; the result stays `Unstated` and the caller \
                 supplies the `Shape`.",
                spec.target
            ));
            continue;
        }
        let resolved = match &spec.rule {
            RuleExpr::Like(a) => {
                let of = find(&inputs, a, "input");
                Some(format!("OutRule::Like {{ of: {of} }}"))
            }
            RuleExpr::Split(a, p) => {
                let of = find(&inputs, a, "input");
                let dim = find(&pruns, p, "params-run");
                Some(format!("OutRule::Split {{ of: {of}, dim_param: {dim} }}"))
            }
            RuleExpr::Shaped { rows_of, width } => {
                let rof = find(&inputs, rows_of, "input");
                let w = match width {
                    WidthExpr::Half(a) => Some(format!(
                        "OutWidth::Half {{ of: {} }}",
                        find(&inputs, a, "input")
                    )),
                    WidthExpr::Of(a) => Some(format!(
                        "OutWidth::Of {{ of: {} }}",
                        find(&inputs, a, "input")
                    )),
                    WidthExpr::Weight(w) => {
                        // `out_shape` answers `None` for a weight width —
                        // the statement carries names, not handles. §10
                        // leaves resolving it to a later step; until then
                        // the honest reading is `Unstated`.
                        notes.push(format!(
                            "The routine's `out({} = rows(..) x weight({w}))` \
                             rule needs the weight handle's width, which the \
                             statement does not carry; the result stays \
                             `Unstated` and the caller supplies the `Shape`.",
                            spec.target
                        ));
                        None
                    }
                    WidthExpr::Param(a) => Some(format!(
                        "OutWidth::Param {{ of: {} }}",
                        find(&pruns, a, "params-run")
                    )),
                };
                w.map(|w| format!("OutRule::Shaped {{ rows_of: {rof}, width: {w} }}"))
            }
        };
        rules[at] = resolved;
    }
    rules
}

// ── attribute parsing ────────────────────────────────────────────────────

fn parse_spec(attr: &syn::Attribute, file: &Path) -> Spec {
    match &attr.meta {
        syn::Meta::Path(_) => Spec::default(),
        syn::Meta::List(list) => {
            syn::parse2::<Args>(list.tokens.clone())
                .unwrap_or_else(|e| panic!("parsing #[routine(..)] in {}: {e}", file.display()))
                .0
        }
        syn::Meta::NameValue(_) => {
            panic!(
                "#[routine = ..] in {} is not a form the macro takes",
                file.display()
            )
        }
    }
}

struct Args(Spec);

impl syn::parse::Parse for Args {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        // The facts the macro itself recognizes; everything else bare is a
        // generic instantiation the generator has no use for.
        const FACTS: [&str; 7] = [
            "whole",
            "depth_prefix_plan",
            "uncolumned",
            "untraced",
            "no_join",
            "internal",
            "driver",
        ];
        let mut spec = Spec::default();
        let items = syn::punctuated::Punctuated::<Item, syn::Token![,]>::parse_terminated(input)?;
        for item in items {
            match item {
                Item::Namespace(ns) => spec.namespace = Some(ns),
                Item::Out(o) => spec.outs.push(o),
                Item::Other(t) => {
                    if let syn::Type::Path(p) = &t
                        && p.qself.is_none()
                        && p.path.segments.len() == 1
                        && p.path.segments[0].arguments.is_none()
                    {
                        let word = p.path.segments[0].ident.to_string();
                        if FACTS.contains(&word.as_str()) {
                            spec.facts.push(word);
                        }
                    }
                }
                Item::Ignored => {}
            }
        }
        Ok(Self(spec))
    }
}

enum Item {
    Namespace(String),
    Out(OutSpec),
    /// A bare type or fact word.
    Other(syn::Type),
    /// `canon = ..` / `dtypes(..)`: parsed past, nothing for the generator.
    Ignored,
}

impl syn::parse::Parse for Item {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        if input.peek(syn::Ident) && input.peek2(syn::token::Paren) {
            let key: syn::Ident = input.parse()?;
            let inner;
            syn::parenthesized!(inner in input);
            if key == "out" {
                return Ok(Self::Out(inner.parse()?));
            }
            // `dtypes(..)`: an ident list; drain and drop.
            let _ = syn::punctuated::Punctuated::<syn::Ident, syn::Token![,]>::parse_terminated(
                &inner,
            )?;
            return Ok(Self::Ignored);
        }
        if input.peek(syn::Ident) && input.peek2(syn::Token![=]) {
            let key: syn::Ident = input.parse()?;
            let _: syn::Token![=] = input.parse()?;
            if key == "namespace" {
                let lit: syn::LitStr = input.parse()?;
                return Ok(Self::Namespace(lit.value()));
            }
            // `canon = role` (ident or string): drop.
            if input.peek(syn::LitStr) {
                let _: syn::LitStr = input.parse()?;
            } else {
                let _: syn::Ident = input.parse()?;
            }
            return Ok(Self::Ignored);
        }
        Ok(Self::Other(input.parse()?))
    }
}

impl syn::parse::Parse for OutSpec {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let target: syn::Ident = input.parse()?;
        let _: syn::Token![=] = input.parse()?;
        let head: syn::Ident = input.parse()?;
        let inner;
        syn::parenthesized!(inner in input);
        let rule = match head.to_string().as_str() {
            "like" => RuleExpr::Like(inner.parse::<syn::Ident>()?.to_string()),
            "split" => {
                let a: syn::Ident = inner.parse()?;
                let _: syn::Token![,] = inner.parse()?;
                let b: syn::Ident = inner.parse()?;
                RuleExpr::Split(a.to_string(), b.to_string())
            }
            "rows" => {
                let rows_of: syn::Ident = inner.parse()?;
                let sep: syn::Ident = input.parse()?;
                if sep != "x" {
                    return Err(syn::Error::new(
                        sep.span(),
                        "expected `x` between rows(..) and the width",
                    ));
                }
                let width = if input.peek(syn::Token![const]) {
                    let _: syn::Token![const] = input.parse()?;
                    let w;
                    syn::parenthesized!(w in input);
                    WidthExpr::Param(w.parse::<syn::Ident>()?.to_string())
                } else {
                    let wname: syn::Ident = input.parse()?;
                    let w;
                    syn::parenthesized!(w in input);
                    let id = w.parse::<syn::Ident>()?.to_string();
                    match wname.to_string().as_str() {
                        "half" => WidthExpr::Half(id),
                        "width" => WidthExpr::Of(id),
                        "weight" => WidthExpr::Weight(id),
                        other => {
                            return Err(syn::Error::new(
                                wname.span(),
                                format!("unknown width constructor `{other}`"),
                            ));
                        }
                    }
                };
                RuleExpr::Shaped {
                    rows_of: rows_of.to_string(),
                    width,
                }
            }
            other => {
                return Err(syn::Error::new(
                    head.span(),
                    format!("unknown out rule `{other}`"),
                ));
            }
        };
        Ok(Self {
            target: target.to_string(),
            rule,
        })
    }
}

// ── emission ─────────────────────────────────────────────────────────────

/// One out slot's emission plan.
struct OutSlot {
    /// The binding the made value gets, and the expect message's subject.
    name: String,
    /// `Some(param name)` when the slot takes a `(Shape, DType)` argument.
    shape_param: Option<String>,
    optional: bool,
    rule: Option<String>,
}

fn emit(out: &mut String, r: &Routine) {
    let symbol = format!("{}::{}", r.namespace, r.name);
    let marker = std::iter::once("kernels_cuda")
        .chain(r.mod_segs.iter().map(String::as_str))
        .chain(std::iter::once(r.name.as_str()))
        .collect::<Vec<_>>()
        .join("::");

    // The out slots, in slot order, with their argument spelling.
    let mut out_slots: Vec<OutSlot> = Vec::new();
    {
        let mut at = 0usize;
        for p in &r.params {
            match &p.mark {
                Mark::InOut => {
                    let rule = r.out_rules.get(at).cloned().flatten();
                    out_slots.push(OutSlot {
                        name: p.name.clone(),
                        shape_param: if rule.is_some() {
                            None
                        } else {
                            Some(format!("{}_out", p.name))
                        },
                        optional: false,
                        rule,
                    });
                    at += 1;
                }
                Mark::Out { optional } => {
                    let rule = r.out_rules.get(at).cloned().flatten();
                    assert!(
                        rule.is_none() || !*optional,
                        "`{symbol}`: an optional result with an out rule has \
                         no wrapper spelling"
                    );
                    out_slots.push(OutSlot {
                        name: p.name.clone(),
                        shape_param: if rule.is_some() {
                            None
                        } else {
                            Some(p.name.clone())
                        },
                        optional: *optional,
                        rule,
                    });
                    at += 1;
                }
                _ => {}
            }
        }
    }

    let required_inputs: Vec<&Param> = r
        .params
        .iter()
        .filter(|p| matches!(p.mark, Mark::In { optional: false } | Mark::InOut))
        .collect();
    let has_optional_input = r
        .params
        .iter()
        .any(|p| matches!(p.mark, Mark::In { optional: true }));
    let has_optional_weight = r
        .params
        .iter()
        .any(|p| matches!(p.mark, Mark::Weight { optional: true }));
    let needs_trace_param = required_inputs.is_empty();

    // ── docs ──
    for line in &r.docs {
        if line.is_empty() {
            out.push_str("///\n");
        } else {
            let _ = writeln!(out, "///{line}");
        }
    }
    if !r.docs.is_empty() {
        out.push_str("///\n");
    }
    for line in wrap(
        &format!(
            "Generated for `{symbol}` from the routine's own signature \
             (`{marker}`); the statement records through \
             [`crate::fire::fire`], one argument per mark."
        ),
        72,
    ) {
        let _ = writeln!(out, "/// {line}");
    }
    for note in &r.notes {
        out.push_str("///\n");
        for line in wrap(note, 68) {
            let _ = writeln!(out, "/// {line}");
        }
    }

    // ── signature ──
    if !out_slots.is_empty() {
        out.push_str("#[must_use]\n");
    }
    let _ = writeln!(out, "pub fn {}(", ident(&r.name));
    if needs_trace_param {
        out.push_str("    t: &Trace,\n");
    }
    for p in &r.params {
        let arg = match &p.mark {
            Mark::In { optional: false } | Mark::InOut => Some(format!("{}: &Val", ident(&p.name))),
            Mark::In { optional: true } => Some(format!("{}: Option<&Val>", ident(&p.name))),
            Mark::Out { .. } => None, // spelled from out_slots below, in place
            Mark::Weight { optional: false } => Some(format!("{}: &str", ident(&p.name))),
            Mark::Weight { optional: true } => Some(format!("{}: Option<&str>", ident(&p.name))),
            Mark::Scalar(s) => Some(format!("{}: {}", ident(&p.name), s.arg_ty())),
            Mark::Unmarked => None,
        };
        if let Some(a) = arg {
            let _ = writeln!(out, "    {a},");
        }
        // The result half, where it takes an argument, sits where the mark
        // sits so the generated signature reads like the routine's.
        let slot = match &p.mark {
            Mark::InOut | Mark::Out { .. } => out_slots.iter().find(|s| s.name == p.name),
            _ => None,
        };
        if let Some(slot) = slot
            && let Some(param) = &slot.shape_param
        {
            if slot.optional {
                let _ = writeln!(out, "    {}: Option<(Shape, DType)>,", ident(param));
            } else {
                let _ = writeln!(out, "    {}: (Shape, DType),", ident(param));
            }
        }
    }
    out.push_str("    layer: Option<u32>,\n");
    out.push_str("    state: Option<StateRef>,\n");
    let ret = match out_slots.len() {
        0 => String::new(),
        1 => format!(
            " -> {}",
            if out_slots[0].optional {
                "Option<Val>"
            } else {
                "Val"
            }
        ),
        _ => format!(
            " -> ({})",
            out_slots
                .iter()
                .map(|s| if s.optional { "Option<Val>" } else { "Val" })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let _ = writeln!(out, "){ret} {{");

    // ── body ──
    if needs_trace_param {
        out.push_str("    let t = t.clone();\n");
    } else {
        let _ = writeln!(
            out,
            "    let t = {}.t.clone();",
            ident(&required_inputs[0].name)
        );
    }

    // presence flags for optional outs, before their arguments move
    for s in &out_slots {
        if s.optional
            && let Some(param) = &s.shape_param
        {
            let _ = writeln!(out, "    let has_{} = {}.is_some();", s.name, ident(param));
        }
    }

    // the params run
    let scalars: Vec<&Param> = r
        .params
        .iter()
        .filter(|p| matches!(p.mark, Mark::Scalar(_)))
        .collect();
    if scalars.is_empty() {
        out.push_str("    let run_params: Vec<u32> = Vec::new();\n");
    } else {
        let items: Vec<String> = scalars
            .iter()
            .map(|p| {
                let Mark::Scalar(s) = p.mark else {
                    unreachable!()
                };
                s.encode(&ident(&p.name))
            })
            .collect();
        emit_vec(out, "run_params", &items, None);
    }

    // the operand run
    let input_marks: Vec<&Param> = r
        .params
        .iter()
        .filter(|p| matches!(p.mark, Mark::In { .. } | Mark::InOut))
        .collect();
    if input_marks.is_empty() {
        out.push_str("    let run_inputs: Vec<ValueId> = Vec::new();\n");
    } else if has_optional_input {
        out.push_str("    let mut run_inputs = Vec::new();\n");
        for p in &input_marks {
            if matches!(p.mark, Mark::In { optional: true }) {
                let _ = writeln!(
                    out,
                    "    if let Some(v) = {} {{\n        run_inputs.push(v.id);\n    }}",
                    ident(&p.name)
                );
            } else {
                let _ = writeln!(out, "    run_inputs.push({}.id);", ident(&p.name));
            }
        }
    } else {
        let items: Vec<String> = input_marks
            .iter()
            .map(|p| format!("{}.id", ident(&p.name)))
            .collect();
        emit_vec(out, "run_inputs", &items, None);
    }

    // the weight run
    let weight_marks: Vec<&Param> = r
        .params
        .iter()
        .filter(|p| matches!(p.mark, Mark::Weight { .. }))
        .collect();
    if weight_marks.is_empty() {
        out.push_str("    let run_weights: Vec<String> = Vec::new();\n");
    } else if has_optional_weight {
        out.push_str("    let mut run_weights = Vec::new();\n");
        for p in &weight_marks {
            if matches!(p.mark, Mark::Weight { optional: true }) {
                let _ = writeln!(
                    out,
                    "    if let Some(w) = {} {{\n        run_weights.push(w.to_string());\n    }}",
                    ident(&p.name)
                );
            } else {
                let _ = writeln!(out, "    run_weights.push({}.to_string());", ident(&p.name));
            }
        }
    } else {
        let items: Vec<String> = weight_marks
            .iter()
            .map(|p| format!("{}.to_string()", ident(&p.name)))
            .collect();
        emit_vec(out, "run_weights", &items, None);
    }

    // the result run
    let any_optional_out = out_slots.iter().any(|s| s.optional);
    if out_slots.is_empty() {
        out.push_str("    let run_outs: Vec<(Shape, DType)> = Vec::new();\n");
    } else if any_optional_out {
        out.push_str("    let mut run_outs = Vec::new();\n");
        for s in &out_slots {
            match (&s.rule, &s.shape_param, s.optional) {
                (Some(rule), _, _) => {
                    let _ = writeln!(
                        out,
                        "    run_outs.push(ruled_out(&t, \"{symbol}\", {rule}, &run_inputs, &run_params));"
                    );
                }
                (None, Some(param), false) => {
                    let _ = writeln!(out, "    run_outs.push({});", ident(param));
                }
                (None, Some(param), true) => {
                    let _ = writeln!(
                        out,
                        "    if let Some(o) = {} {{\n        run_outs.push(o);\n    }}",
                        ident(param)
                    );
                }
                (None, None, _) => unreachable!("an unruled slot has a param"),
            }
        }
    } else {
        let items: Vec<String> = out_slots
            .iter()
            .map(|s| match (&s.rule, &s.shape_param) {
                (Some(rule), _) => {
                    format!("ruled_out(&t, \"{symbol}\", {rule}, &run_inputs, &run_params)")
                }
                (None, Some(param)) => ident(param),
                (None, None) => unreachable!("an unruled slot has a param"),
            })
            .collect();
        emit_vec(out, "run_outs", &items, None);
    }

    // the fire
    let _ = writeln!(out, "    let made = fire::<{marker}>(&t, Call {{");
    out.push_str("        inputs: run_inputs,\n");
    out.push_str("        weights: run_weights,\n");
    out.push_str("        params: run_params,\n");
    out.push_str("        outs: run_outs,\n");
    out.push_str("        state,\n");
    out.push_str("        layer,\n");
    out.push_str("        extents: Vec::new(),\n");
    out.push_str("    });\n");

    // the return
    match out_slots.len() {
        0 => {
            let _ = writeln!(
                out,
                "    assert!(made.is_empty(), \"`{symbol}` states no result\");"
            );
        }
        1 if !out_slots[0].optional => {
            let _ = writeln!(
                out,
                "    made.into_iter().next().expect(\"`{symbol}` states `{}`\")",
                out_slots[0].name
            );
        }
        _ => {
            out.push_str("    let mut made = made.into_iter();\n");
            for s in &out_slots {
                if s.optional {
                    let _ = writeln!(
                        out,
                        "    let {b} = has_{n}.then(|| made.next().expect(\"`{symbol}` states `{n}`\"));",
                        b = ident(&s.name),
                        n = s.name
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "    let {b} = made.next().expect(\"`{symbol}` states `{n}`\");",
                        b = ident(&s.name),
                        n = s.name
                    );
                }
            }
            let names: Vec<String> = out_slots.iter().map(|s| ident(&s.name)).collect();
            if names.len() == 1 {
                let _ = writeln!(out, "    {}", names[0]);
            } else {
                let _ = writeln!(out, "    ({})", names.join(", "));
            }
        }
    }
    out.push_str("}\n");
}

/// `let {name} = vec![..];`, one line when it fits, one item per line when
/// it does not.
fn emit_vec(out: &mut String, name: &str, items: &[String], ty: Option<&str>) {
    let ann = ty.map(|t| format!(": {t}")).unwrap_or_default();
    let one = format!("    let {name}{ann} = vec![{}];", items.join(", "));
    if one.len() <= 78 {
        out.push_str(&one);
        out.push('\n');
    } else {
        let _ = writeln!(out, "    let {name}{ann} = vec![");
        for i in items {
            let _ = writeln!(out, "        {i},");
        }
        out.push_str("    ];\n");
    }
}

/// Raw-ident spelling where the name collides with a keyword.
fn ident(name: &str) -> String {
    if is_keyword(name) {
        format!("r#{name}")
    } else {
        name.to_string()
    }
}

fn is_keyword(name: &str) -> bool {
    const KW: &[&str] = &[
        "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn",
        "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
        "return", "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe",
        "use", "where", "while", "async", "await", "dyn", "abstract", "become", "box", "do",
        "final", "macro", "override", "priv", "typeof", "unsized", "virtual", "yield", "try",
        "gen",
    ];
    KW.contains(&name)
}

/// Greedy wrap for note lines.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

// ── the metal half (B4-gen step 6) ───────────────────────────────────────

/// A metal routine's SYMBOL, read off its body's `Fire::at(file, entry)`.
///
/// On the shader planes a statement names an instantiated ENTRYPOINT
/// (`rms_single_row_bfloat16`), which the drivers resolve back to the
/// routine by census stem — so the symbol is not `namespace::name` and
/// cannot come off the marker alone. Where the body names one literal the
/// wrapper states it verbatim; where the body composes the entrypoint from
/// an instantiation point (a head-dim table, an affine codec, a tile) the
/// choice is the CALLER's, and the wrapper takes the symbol as its first
/// argument — the same class of decision the hand-written choosers keep.
enum Symbol {
    Fixed(String),
    Pointed,
}

/// One traced metal routine: the shared classification plus the plane's
/// two extras — the symbol reading and the spliced-`rows` slots.
struct MetalRoutine {
    base: Routine,
    symbol: Symbol,
    /// Params-run indices spliced as the fire's row extent (the trailing
    /// `rows: Const<i32>` convention).
    spliced: Vec<usize>,
}

/// Routines whose trailing `rows` is NOT the first operand's row axis, so
/// the convention must not splice it: guessing would state the wrong
/// extent. `rows` stays a stated argument on these, with the reason in the
/// generated doc; the hand keeper remains the fireable form.
const ROWS_NOT_THE_FIRST_OPERANDS: &[(&str, &str)] = &[
    (
        "qmv_routed",
        "`rows` is the FIRE's token count, but the first operand is the \
         SORTED STACK (`MoeAlignedRoutes`), so the rows convention would \
         splice the wrong axis. It stays a stated argument here; the hand \
         `routed_qmv` keeper splices `Dim::Tokens` and remains the \
         fireable form.",
    ),
    (
        "qmv_routed_bias",
        "as `qmv_routed`: the fire's token count over a stack-rowed first \
         operand; the hand keeper splices `Dim::Tokens`.",
    ),
    (
        "mxfp4_qmv_routed_bias",
        "as `qmv_routed`: the fire's token count over a stack-rowed first \
         operand; the hand keeper splices `Dim::Tokens`.",
    ),
];

/// Read the body's `Fire::at(file, entry)` entries.
fn entry_of(f: &syn::ItemFn) -> Symbol {
    use syn::visit::Visit;
    struct V {
        entries: Vec<Option<String>>,
    }
    impl<'a> Visit<'a> for V {
        fn visit_expr_call(&mut self, c: &'a syn::ExprCall) {
            if let syn::Expr::Path(p) = c.func.as_ref() {
                let n = p.path.segments.len();
                if n >= 2
                    && p.path.segments[n - 2].ident == "Fire"
                    && p.path.segments[n - 1].ident == "at"
                {
                    let lit = c.args.iter().nth(1).and_then(|a| match a {
                        syn::Expr::Lit(l) => match &l.lit {
                            syn::Lit::Str(s) => Some(s.value()),
                            _ => None,
                        },
                        _ => None,
                    });
                    self.entries.push(lit);
                }
            }
            syn::visit::visit_expr_call(self, c);
        }
    }
    let mut v = V {
        entries: Vec::new(),
    };
    v.visit_item_fn(f);
    match v.entries.as_slice() {
        [Some(one)] => Symbol::Fixed(one.clone()),
        _ => Symbol::Pointed,
    }
}

/// Parse the metal tree into its traced routines, in emission order.
fn collect_metal(kernels_src: &Path) -> Vec<MetalRoutine> {
    let mut files = Vec::new();
    collect_rs(kernels_src, &mut files);
    files.sort();

    let mut routines: Vec<MetalRoutine> = Vec::new();
    for file in &files {
        let src = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("reading {}: {e}", file.display()));
        let parsed =
            syn::parse_file(&src).unwrap_or_else(|e| panic!("parsing {}: {e}", file.display()));
        let rel = file
            .strip_prefix(kernels_src)
            .expect("file is under the walked tree");
        let segs = mod_segs(rel);
        walk_items_metal(&parsed.items, &segs, &mut routines, file);
    }
    routines
}

fn walk_items_metal(
    items: &[syn::Item],
    segs: &[String],
    out: &mut Vec<MetalRoutine>,
    file: &Path,
) {
    for item in items {
        match item {
            syn::Item::Mod(m) => {
                if let Some((_, inner)) = &m.content {
                    let mut deeper = segs.to_vec();
                    deeper.push(m.ident.to_string());
                    walk_items_metal(inner, &deeper, out, file);
                }
            }
            syn::Item::Fn(f) => {
                if let Some(base) = routine_of(f, segs, file) {
                    out.push(finish_metal(base, f));
                }
            }
            _ => {}
        }
    }
}

/// Classify the plane's extras onto one routine: the symbol reading, the
/// spliced-`rows` slots, and their doc notes.
fn finish_metal(mut base: Routine, f: &syn::ItemFn) -> MetalRoutine {
    let symbol = entry_of(f);
    let mut spliced = Vec::new();
    if let Some((_, why)) = ROWS_NOT_THE_FIRST_OPERANDS
        .iter()
        .find(|(n, _)| *n == base.name)
    {
        base.notes.push((*why).to_string());
    } else {
        let mut at = 0usize;
        for p in &base.params {
            if let Mark::Scalar(s) = p.mark {
                if p.name == "rows" && s == Scalar::I32 {
                    spliced.push(at);
                }
                at += 1;
            }
        }
    }
    if let Symbol::Pointed = symbol {
        base.notes.push(
            "The routine's entrypoint is COMPOSED from an instantiation \
             point, so the SYMBOL is this wrapper's first argument; \
             `fire_at` refuses one the census does not resolve to this \
             routine."
                .to_string(),
        );
    }
    MetalRoutine {
        base,
        symbol,
        spliced,
    }
}

/// The traced metal routines' `(fixed symbol, fn name)` pairs — what the
/// coverage and no-shadowing pins walk. The symbol is `None` where the
/// entrypoint is composed (the wrapper takes it as an argument), `Some`
/// where the wrapper states the body's one literal.
pub fn traced_metal(kernels_src: &Path) -> Vec<(Option<String>, String)> {
    collect_metal(kernels_src)
        .into_iter()
        .map(|r| {
            let sym = match r.symbol {
                Symbol::Fixed(s) => Some(s),
                Symbol::Pointed => None,
            };
            (sym, r.base.name)
        })
        .collect()
}

/// Generate the whole `src/metal/generated.rs` from the kernels tree.
pub fn generate_metal(kernels_src: &Path) -> String {
    let routines = collect_metal(kernels_src);

    let mut seen = std::collections::BTreeMap::new();
    for r in &routines {
        if let Some(prior) = seen.insert(r.base.name.clone(), r.base.mod_segs.join("::")) {
            panic!(
                "two traced routines share the name `{}` (`{}` and `{}`); \
                 the flat generated module cannot hold both",
                r.base.name,
                prior,
                r.base.mod_segs.join("::"),
            );
        }
        assert!(
            !is_keyword(&r.base.name),
            "routine name `{}` is a Rust keyword; the generated fn cannot wear it",
            r.base.name
        );
    }

    let mut out = String::new();
    out.push_str(METAL_HEADER);
    for r in &routines {
        out.push('\n');
        emit_metal(&mut out, r);
    }
    out
}

const METAL_HEADER: &str = "\
//! GENERATED — do not edit. One named `pub fn` per traced `#[routine]` in
//! `crates/kernels-metal/src`, named by the routine's own name, in
//! sorted-file then source order (design-no-ask §10, B4-gen step 6 — the
//! metal half; vulkan and wgpu execute the same statements off their
//! pinned-equal tables, so this is the one generated surface for all
//! three shader planes).
//!
//! The generator is `tests/generator/mod.rs`;
//! `cargo test -p model-dsl --test wrappers_are_current` refuses a stale
//! file and `UPDATE_WRAPPERS=1` rewrites it.
//!
//! Every mark is one argument — runtime streams and views included; a
//! wrapper here mints NOTHING in secret. Two readings are this plane's
//! own. The SYMBOL is the instantiated entrypoint the drivers resolve by
//! census stem: where the routine's body names one literal the wrapper
//! states it verbatim, and where the body composes an instantiation
//! point the symbol is the caller's first argument, checked against the
//! routine by [`crate::fire::fire_at`]. And a trailing `Const<i32>`
//! named `rows` is the FIRE's row extent by this plane's convention: the
//! wrapper takes no argument for it, writes the zero placeholder, and
//! splices the first operand's row axis (`rows_of`) — exactly what every
//! hand statement recorded. A result whose routine states an `out(..)`
//! rule is derived at trace time through
//! [`model_ir::kernels::out_shape`]; an `Unstated` result stays a
//! `(Shape, DType)` argument. Trailing `layer` and `state` are the
//! statement's tags, uniformly.
#![cfg_attr(rustfmt, rustfmt::skip)]

// The prelude is fixed while the surface below is generated from another
// crate's tree, so any one regeneration may leave part of it unused.
#![allow(unused_imports)]

use kernels::{OutRule, OutWidth};
use model_ir::trace::{DType, Shape, StateRef, ValueId};

use super::{ruled_out, rows_of};
use crate::fire::{Call, fire_at};
use crate::{Trace, Val};
";

fn emit_metal(out: &mut String, r: &MetalRoutine) {
    let base = &r.base;
    let marker = std::iter::once("kernels_metal")
        .chain(base.mod_segs.iter().map(String::as_str))
        .chain(std::iter::once(base.name.as_str()))
        .collect::<Vec<_>>()
        .join("::");
    // What the panic messages and the doc line call this statement.
    let subject = match &r.symbol {
        Symbol::Fixed(s) => s.clone(),
        Symbol::Pointed => base.name.clone(),
    };
    // The symbol expression `fire_at` and `ruled_out` receive.
    let sym_expr = match &r.symbol {
        Symbol::Fixed(s) => format!("\"{s}\""),
        Symbol::Pointed => "symbol".to_string(),
    };

    // The out slots, in slot order, with their argument spelling.
    let mut out_slots: Vec<OutSlot> = Vec::new();
    {
        let mut at = 0usize;
        for p in &base.params {
            match &p.mark {
                Mark::InOut => {
                    let rule = base.out_rules.get(at).cloned().flatten();
                    out_slots.push(OutSlot {
                        name: p.name.clone(),
                        shape_param: if rule.is_some() {
                            None
                        } else {
                            Some(format!("{}_out", p.name))
                        },
                        optional: false,
                        rule,
                    });
                    at += 1;
                }
                Mark::Out { optional } => {
                    let rule = base.out_rules.get(at).cloned().flatten();
                    assert!(
                        rule.is_none() || !*optional,
                        "`{subject}`: an optional result with an out rule has \
                         no wrapper spelling"
                    );
                    out_slots.push(OutSlot {
                        name: p.name.clone(),
                        shape_param: if rule.is_some() {
                            None
                        } else {
                            Some(p.name.clone())
                        },
                        optional: *optional,
                        rule,
                    });
                    at += 1;
                }
                _ => {}
            }
        }
    }

    let required_inputs: Vec<&Param> = base
        .params
        .iter()
        .filter(|p| matches!(p.mark, Mark::In { optional: false } | Mark::InOut))
        .collect();
    let has_optional_input = base
        .params
        .iter()
        .any(|p| matches!(p.mark, Mark::In { optional: true }));
    let has_optional_weight = base
        .params
        .iter()
        .any(|p| matches!(p.mark, Mark::Weight { optional: true }));
    let needs_trace_param = required_inputs.is_empty();
    assert!(
        !(needs_trace_param && !r.spliced.is_empty()),
        "`{subject}`: a spliced `rows` needs a first operand to read the \
         row axis from"
    );

    // ── docs ──
    for line in &base.docs {
        if line.is_empty() {
            out.push_str("///\n");
        } else {
            let _ = writeln!(out, "///{line}");
        }
    }
    if !base.docs.is_empty() {
        out.push_str("///\n");
    }
    let generated_line = match &r.symbol {
        Symbol::Fixed(s) => format!(
            "Generated for `{s}` from the routine's own signature \
             (`{marker}`); the statement records through \
             [`crate::fire::fire_at`], one argument per mark."
        ),
        Symbol::Pointed => format!(
            "Generated for `{}`'s instantiations from the routine's own \
             signature (`{marker}`); the statement records through \
             [`crate::fire::fire_at`], one argument per mark.",
            base.name
        ),
    };
    for line in wrap(&generated_line, 72) {
        let _ = writeln!(out, "/// {line}");
    }
    for note in &base.notes {
        out.push_str("///\n");
        for line in wrap(note, 68) {
            let _ = writeln!(out, "/// {line}");
        }
    }

    // ── signature ──
    if !out_slots.is_empty() {
        out.push_str("#[must_use]\n");
    }
    let _ = writeln!(out, "pub fn {}(", ident(&base.name));
    if let Symbol::Pointed = r.symbol {
        out.push_str("    symbol: &str,\n");
    }
    if needs_trace_param {
        out.push_str("    t: &Trace,\n");
    }
    {
        let mut scalar_at = 0usize;
        for p in &base.params {
            let spliced_here =
                matches!(p.mark, Mark::Scalar(_)) && r.spliced.contains(&scalar_at);
            if matches!(p.mark, Mark::Scalar(_)) {
                scalar_at += 1;
            }
            let arg = match &p.mark {
                Mark::In { optional: false } | Mark::InOut => {
                    Some(format!("{}: &Val", ident(&p.name)))
                }
                Mark::In { optional: true } => Some(format!("{}: Option<&Val>", ident(&p.name))),
                Mark::Out { .. } => None, // spelled from out_slots below, in place
                Mark::Weight { optional: false } => Some(format!("{}: &str", ident(&p.name))),
                Mark::Weight { optional: true } => {
                    Some(format!("{}: Option<&str>", ident(&p.name)))
                }
                Mark::Scalar(s) if !spliced_here => {
                    Some(format!("{}: {}", ident(&p.name), s.arg_ty()))
                }
                Mark::Scalar(_) => None, // the spliced fire extent
                Mark::Unmarked => None,
            };
            if let Some(a) = arg {
                let _ = writeln!(out, "    {a},");
            }
            let slot = match &p.mark {
                Mark::InOut | Mark::Out { .. } => out_slots.iter().find(|s| s.name == p.name),
                _ => None,
            };
            if let Some(slot) = slot
                && let Some(param) = &slot.shape_param
            {
                if slot.optional {
                    let _ = writeln!(out, "    {}: Option<(Shape, DType)>,", ident(param));
                } else {
                    let _ = writeln!(out, "    {}: (Shape, DType),", ident(param));
                }
            }
        }
    }
    out.push_str("    layer: Option<u32>,\n");
    out.push_str("    state: Option<StateRef>,\n");
    let ret = match out_slots.len() {
        0 => String::new(),
        1 => format!(
            " -> {}",
            if out_slots[0].optional {
                "Option<Val>"
            } else {
                "Val"
            }
        ),
        _ => format!(
            " -> ({})",
            out_slots
                .iter()
                .map(|s| if s.optional { "Option<Val>" } else { "Val" })
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let _ = writeln!(out, "){ret} {{");

    // ── body ──
    if needs_trace_param {
        out.push_str("    let t = t.clone();\n");
    } else {
        let _ = writeln!(
            out,
            "    let t = {}.t.clone();",
            ident(&required_inputs[0].name)
        );
    }

    for s in &out_slots {
        if s.optional
            && let Some(param) = &s.shape_param
        {
            let _ = writeln!(out, "    let has_{} = {}.is_some();", s.name, ident(param));
        }
    }

    // the params run: signature order, a zero placeholder at each spliced
    // `rows` slot
    let scalars: Vec<&Param> = base
        .params
        .iter()
        .filter(|p| matches!(p.mark, Mark::Scalar(_)))
        .collect();
    if scalars.is_empty() {
        out.push_str("    let run_params: Vec<u32> = Vec::new();\n");
    } else {
        let items: Vec<String> = scalars
            .iter()
            .enumerate()
            .map(|(at, p)| {
                if r.spliced.contains(&at) {
                    "0".to_string()
                } else {
                    let Mark::Scalar(s) = p.mark else {
                        unreachable!()
                    };
                    s.encode(&ident(&p.name))
                }
            })
            .collect();
        emit_vec(out, "run_params", &items, None);
    }

    // the operand run
    let input_marks: Vec<&Param> = base
        .params
        .iter()
        .filter(|p| matches!(p.mark, Mark::In { .. } | Mark::InOut))
        .collect();
    if input_marks.is_empty() {
        out.push_str("    let run_inputs: Vec<ValueId> = Vec::new();\n");
    } else if has_optional_input {
        out.push_str("    let mut run_inputs = Vec::new();\n");
        for p in &input_marks {
            if matches!(p.mark, Mark::In { optional: true }) {
                let _ = writeln!(
                    out,
                    "    if let Some(v) = {} {{\n        run_inputs.push(v.id);\n    }}",
                    ident(&p.name)
                );
            } else {
                let _ = writeln!(out, "    run_inputs.push({}.id);", ident(&p.name));
            }
        }
    } else {
        let items: Vec<String> = input_marks
            .iter()
            .map(|p| format!("{}.id", ident(&p.name)))
            .collect();
        emit_vec(out, "run_inputs", &items, None);
    }

    // the weight run
    let weight_marks: Vec<&Param> = base
        .params
        .iter()
        .filter(|p| matches!(p.mark, Mark::Weight { .. }))
        .collect();
    if weight_marks.is_empty() {
        out.push_str("    let run_weights: Vec<String> = Vec::new();\n");
    } else if has_optional_weight {
        out.push_str("    let mut run_weights = Vec::new();\n");
        for p in &weight_marks {
            if matches!(p.mark, Mark::Weight { optional: true }) {
                let _ = writeln!(
                    out,
                    "    if let Some(w) = {} {{\n        run_weights.push(w.to_string());\n    }}",
                    ident(&p.name)
                );
            } else {
                let _ = writeln!(out, "    run_weights.push({}.to_string());", ident(&p.name));
            }
        }
    } else {
        let items: Vec<String> = weight_marks
            .iter()
            .map(|p| format!("{}.to_string()", ident(&p.name)))
            .collect();
        emit_vec(out, "run_weights", &items, None);
    }

    // the result run
    let any_optional_out = out_slots.iter().any(|s| s.optional);
    if out_slots.is_empty() {
        out.push_str("    let run_outs: Vec<(Shape, DType)> = Vec::new();\n");
    } else if any_optional_out {
        out.push_str("    let mut run_outs = Vec::new();\n");
        for s in &out_slots {
            match (&s.rule, &s.shape_param, s.optional) {
                (Some(rule), _, _) => {
                    let _ = writeln!(
                        out,
                        "    run_outs.push(ruled_out(&t, {sym_expr}, {rule}, &run_inputs, &run_params));"
                    );
                }
                (None, Some(param), false) => {
                    let _ = writeln!(out, "    run_outs.push({});", ident(param));
                }
                (None, Some(param), true) => {
                    let _ = writeln!(
                        out,
                        "    if let Some(o) = {} {{\n        run_outs.push(o);\n    }}",
                        ident(param)
                    );
                }
                (None, None, _) => unreachable!("an unruled slot has a param"),
            }
        }
    } else {
        let items: Vec<String> = out_slots
            .iter()
            .map(|s| match (&s.rule, &s.shape_param) {
                (Some(rule), _) => {
                    format!("ruled_out(&t, {sym_expr}, {rule}, &run_inputs, &run_params)")
                }
                (None, Some(param)) => ident(param),
                (None, None) => unreachable!("an unruled slot has a param"),
            })
            .collect();
        emit_vec(out, "run_outs", &items, None);
    }

    // the fire-extent run
    if r.spliced.is_empty() {
        out.push_str("    let run_extents: Vec<(u8, Shape)> = Vec::new();\n");
    } else {
        let first = ident(&required_inputs[0].name);
        let items: Vec<String> = r
            .spliced
            .iter()
            .map(|at| format!("({at}, Shape(vec![rows_of({first})]))"))
            .collect();
        emit_vec(out, "run_extents", &items, None);
    }

    // the fire
    let _ = writeln!(
        out,
        "    let made = fire_at::<{marker}>(&t, {sym_expr}, Call {{"
    );
    out.push_str("        inputs: run_inputs,\n");
    out.push_str("        weights: run_weights,\n");
    out.push_str("        params: run_params,\n");
    out.push_str("        outs: run_outs,\n");
    out.push_str("        state,\n");
    out.push_str("        layer,\n");
    out.push_str("        extents: run_extents,\n");
    out.push_str("    });\n");

    // the return
    match out_slots.len() {
        0 => {
            let _ = writeln!(
                out,
                "    assert!(made.is_empty(), \"`{subject}` states no result\");"
            );
        }
        1 if !out_slots[0].optional => {
            let _ = writeln!(
                out,
                "    made.into_iter().next().expect(\"`{subject}` states `{}`\")",
                out_slots[0].name
            );
        }
        _ => {
            out.push_str("    let mut made = made.into_iter();\n");
            for s in &out_slots {
                if s.optional {
                    let _ = writeln!(
                        out,
                        "    let {b} = has_{n}.then(|| made.next().expect(\"`{subject}` states `{n}`\"));",
                        b = ident(&s.name),
                        n = s.name
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "    let {b} = made.next().expect(\"`{subject}` states `{n}`\");",
                        b = ident(&s.name),
                        n = s.name
                    );
                }
            }
            let names: Vec<String> = out_slots.iter().map(|s| ident(&s.name)).collect();
            if names.len() == 1 {
                let _ = writeln!(out, "    {}", names[0]);
            } else {
                let _ = writeln!(out, "    ({})", names.join(", "));
            }
        }
    }
    out.push_str("}\n");
}
