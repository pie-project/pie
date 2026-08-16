//! Every template-id a routine body names is one NVRTC can lower.
//!
//! The refactor deleted the row lists, so nothing enumerates the crate's
//! instantiations any more and nothing checked them. A body names its
//! template-id as a string; a typo in one is an NVRTC error at the first fire
//! of that kernel on a GPU, which is the worst place to find it. Some 750 of
//! these strings were transcribed by hand during the port, and until this
//! fixture ran, not one of them had been through a compiler.
//!
//! The crate declares its roots two ways, so this reads them two ways:
//!
//! * **Written** — `Root::new("norm/rmsnorm.cuh")` with a `mod inst` beside
//!   it. Fifty-nine of these. They are read out of the SOURCE rather than kept
//!   in a list here, because a list here is one more thing to forget to
//!   update; a new constant is covered the moment it is written. The cost is a
//!   parser, and a parser that quietly stopped matching would turn this into a
//!   test that passes by finding nothing — so every root constructor is
//!   reconciled against what was recovered, and a declaration the parser
//!   cannot read fails rather than skips.
//!
//! * **Computed** — the two lattices, where a `const fn` or a macro stamps a
//!   root per member out of one file under a different `-D` set. Fifty-six
//!   FA2 points and five XQA ones, declared with `Root::variant` because their
//!   NAME is computed. Those come from the crate's own public statics, which
//!   is better than parsing: `DECODE`, `PREFILL` and `ROOTS` carry both the
//!   root and its arms, so the fixture asks the crate what it built rather
//!   than guessing.
//!
//! Needs `libnvrtc`, not a device: `nvrtcCompileProgram` targets an
//! architecture, it does not talk to one. Skips with a message when the
//! library will not load, so a box without CUDA still runs the suite.

#![cfg(feature = "_cuda")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use kernels_cuda::jit::{Root, Toolchain};
use kernels_cuda::jit::abi::Elem;
use kernels_cuda::jit::nvrtc;
use kernels_cuda::source::Header;
use kernels_cuda::quant;

// ===========================================================================
// One compile, however its root was declared
// ===========================================================================

/// One root and the template-ids to ask it for.
struct Job {
    /// Where it came from, for a message that can be acted on.
    site: String,
    /// The root's name, as a diagnostic will spell it.
    name: String,
    /// The device text.
    text: String,
    /// NVRTC options this root needs.
    options: Vec<String>,
    /// The header set its `#include`s resolve against.
    headers: &'static [Header],
    /// The template-ids handed to `nvrtcAddNameExpression`.
    wanted: Vec<String>,
}

impl Job {
    /// The same job, as the crate's compiler takes it.
    fn from_root(site: String, root: &Root, wanted: Vec<String>) -> Self {
        Self {
            site,
            name: root.name.to_owned(),
            text: root.text.to_owned(),
            options: root.options.iter().map(|&o| o.to_owned()).collect(),
            headers: root.header_set(),
            wanted,
        }
    }

    /// Ask NVRTC for every one of them.
    fn compile(&self, arch: &str) -> Result<(), String> {
        let options: Vec<&str> = self.options.iter().map(String::as_str).collect();
        let job = nvrtc::Job {
            name: Box::leak(self.name.clone().into_boxed_str()),
            source: self.text.clone(),
            arch,
            options: &options,
            headers: self.headers,
            // Not the root's own `.since`: this asks whether the toolchain
            // that IS here can lower the symbol, and a floor would answer a
            // different question by refusing before the compile.
            floor: Toolchain::ANY,
            wanted: &self.wanted,
            device_link: options.iter().any(|o| o.contains("relocatable-device-code")),
        };
        nvrtc::compile_text(&job).map(|_| ()).map_err(|why| why.to_string())
    }
}

// ===========================================================================
// The written roots: read out of `src/`
// ===========================================================================

/// A Rust string literal's value: continuations joined, escapes resolved.
///
/// `"a\` + newline + spaces + `b"` is `"ab"` — the backslash eats the newline
/// AND the indentation after it, which is how every long template-id in this
/// crate is written.
fn unescape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('\n') => {
                chars.next();
                while chars.next_if(|c| c.is_whitespace()).is_some() {}
            }
            Some('"' | '\\') => out.push(chars.next().expect("peeked")),
            _ => out.push(c),
        }
    }
    out
}

/// Every `"..."` literal in `text`, unescaped, in source order.
///
/// Comments are skipped rather than scanned. They are prose about kernels, and
/// prose about kernels quotes code: the doc comment over `xqa`'s `mod inst`
/// explains that `kernel_mha` is `extern "C"`, and a scanner that took every
/// pair of quotes handed NVRTC `C` as a name expression.
fn literals(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i = text[i..].find('\n').map_or(bytes.len(), |e| i + e + 1);
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i = text[i + 2..].find("*/").map_or(bytes.len(), |e| i + 2 + e + 2);
            }
            b'"' => {
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() && bytes[j] != b'"' {
                    j += if bytes[j] == b'\\' { 2 } else { 1 };
                }
                if j >= bytes.len() {
                    break;
                }
                out.push(unescape(&text[start..j]));
                i = j + 1;
            }
            _ => i += 1,
        }
    }
    out
}

/// The contents of the balanced group opening at the first `open` in `text`,
/// and the byte offset just past its close.
fn group(text: &str, open: char, close: char) -> Option<(&str, usize)> {
    let at = text.find(open)?;
    let mut depth = 0usize;
    for (i, c) in text[at..].char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some((&text[at + 1..at + i], at + i + 1));
            }
        }
    }
    None
}

/// Template-ids that sit in a `fn` naming no carried file, and how each is
/// reached instead.
///
/// The reconciliation in [`written`] turns any OTHER one into a failure, so a
/// template-id cannot go uncompiled and unmentioned.
const ORPHANS: &[(&str, &str)] = &[
    ("attn/xqa.rs", "the five members' arms; compiled through `xqa::ROOTS` by `computed`"),
    ("attn/fa2/mod.rs", "the lattice's arms; compiled through `fa2::{DECODE, PREFILL}`"),
    ("attn/fa2/dispatch.rs", "as `attn/fa2/mod.rs`"),
    ("attn/fa2/plan.rs", "as `attn/fa2/mod.rs`"),
    ("attn/fa2/params.rs", "as `attn/fa2/mod.rs`"),
    ("attn/fa2/geometry.rs", "as `attn/fa2/mod.rs`"),
];

/// The carried files that name no instantiation, and why each has none.
///
/// A root with nothing to compile is nearly always a mistake -- the launch that
/// fired it was deleted -- so one is a failure unless it is written down here.
///
/// The five CuTile files are one entry each rather than a `starts_with` on a
/// prefix, because they do not share one: `tile/alternatives` is a directory
/// of its own and the other four are inside three different families.
const NO_INSTANTIATIONS: &[(&str, &str)] = &[
    (
        "attn/pack_dense_mask",
        "neither packer has a host program: the file is carried so the text \
         stays reachable, and nothing in this crate instantiates out of it",
    ),
    (
        "tile/alternatives",
        "CuTile, and this crate cannot compile any of it -- `src/tile.rs` has \
         the six measured reasons. This one holds no kernel in any case: it is \
         `static_assert`s over the `*_tile_preferred` bounds",
    ),
    ("sample/argmax_tile", "CuTile: see `src/tile.rs`"),
    ("layout/gather_rows_tile", "CuTile: see `src/tile.rs`"),
    ("quant/dequant_wna16_tile", "CuTile: see `src/tile.rs`"),
    ("quant/wna16_gemv_tile", "CuTile: see `src/tile.rs`"),
    ("moe/expert_offsets", "the CUTLASS fused MoE it serves has no host program in this crate"),
    ("attn/attention_xqa_mha", "XQA's five members, compiled through `xqa::ROOTS` by `computed`"),
    ("attn/fa2", "the FA2 lattice, compiled through `fa2::{DECODE, PREFILL}` by `computed`"),
];

/// A literal that names a carried file rather than a template-id.
fn is_carried_name(s: &str) -> bool {
    s.ends_with(".cuh") && !s.contains(' ') && !s.starts_with("::")
}

/// A literal that is a template-id: an absolute C++ name.
fn is_template_id(s: &str) -> bool {
    s.starts_with("::")
}

/// Every absolute C++ name in this crate that is a TYPE and not an entry point.
///
/// `KvDType`, `StructuredMaskParams` and `::flashinfer::MLAParams<..>` are
/// aggregates, handed to `typecheck_tu` so a measured layout can be compared
/// against the header that declares it. They are spelled exactly like a
/// template-id and asking `nvrtcAddNameExpression` for one is asking for the
/// address of a struct, which does not resolve.
///
/// Asked of the CRATE rather than parsed, because the crate already states it:
/// a `Layout` carries the spelling, and `tests/typecheck_tu.rs` compiles these
/// under the same roots as layouts. Parsing for them would mean chasing every
/// form an `Abi::CPP` is declared in — a bare `const`, a `by_value!`, an `impl`
/// on a pointer — and a form missed would fail this fixture rather than skip.
fn abi_spellings(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (at, _) in text.match_indices("CPP: &'static str") {
        let rest = &text[at..];
        out.extend(literals(&rest[..rest.find(';').unwrap_or(rest.len())]));
    }
    out
}

/// The aggregates, as the crate itself states them.
///
/// [`abi_spellings`] reads the `Abi` impls out of the source and this asks the
/// crate; neither subsumes the other. A scalar `Abi` -- `KvDType` is a `u8`
/// with a C++ name -- has no `Layout` at all and only the parse sees it, and a
/// `Layout` reaches this fixture through a `by_value!` whose `CPP` the parse
/// would have to know the macro's shape to find.
fn aggregates() -> BTreeSet<String> {
    use kernels_cuda::attn::{self, xqa};

    [attn::params::LAYOUTS, attn::mla_params::LAYOUTS, xqa::LAYOUTS, quant::transcode::LAYOUTS]
        .into_iter()
        .flatten()
        .flat_map(|layout| {
            // The pointer spellings too: an `Abi` on `*const T` states
            // `"const ..T*"`, which is the same type wearing punctuation.
            [layout.cpp.to_owned(), format!("const {}*", layout.cpp), format!("{}*", layout.cpp)]
        })
        .collect()
}

/// Every `fn` body in `text`, as `(name, span)`.
///
/// Crude on purpose: a `fn` keyword followed by a balanced brace group. It does
/// not have to be a parser, because what it partitions is only the SCOPE a
/// template-id is attributed to, and the reconciliation below fails loudly if
/// a template-id ends up in no scope that names a file.
fn bodies(text: &str) -> Vec<(String, &str)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    for (at, _) in text.match_indices("fn ") {
        if at > 0 && (bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_') {
            continue;
        }
        let rest = &text[at..];
        let Some(open) = rest.find('{') else { continue };
        // A `fn` whose next brace is past the next `fn` is a declaration in a
        // trait or a pointer type, not a body.
        if rest[..open].contains("fn ") && rest[3..open].contains("fn ") {
            continue;
        }
        let Some((body, _)) = group(rest, '{', '}') else { continue };
        let name = rest[3..].split(|c: char| !(c.is_alphanumeric() || c == '_')).next().unwrap_or("");
        out.push((name.to_owned(), body));
    }
    out
}

/// Every `pub mod` in `text` that names exactly one carried file, as
/// `(span, file)`.
///
/// A module that declares a root and holds a `mod inst` beside it answers for
/// every template-id inside it, wherever in the module it sits: two of these
/// keep their instantiations in a `const [&str; N]` that a launcher indexes, so
/// no `fn` holds them and the module is the only scope there is.
fn anchored_modules(text: &str) -> Vec<((usize, usize), String)> {
    let mut out = Vec::new();
    for (start, _) in text.match_indices("pub mod ") {
        let Some((body, past)) = group(&text[start..], '{', '}') else { continue };
        let named: BTreeSet<String> =
            literals(body).into_iter().filter(|s| is_carried_name(s)).collect();
        if named.len() == 1 {
            out.push(((start, start + past), named.into_iter().next().expect("one")));
        }
    }
    out
}

/// `id` with its `{}` filled, once per element type it is instantiated at.
///
/// An id with no hole is itself; an id with a hole and no instantiation is
/// dropped, and the reconciliation below is what reports it.
fn fill(id: &str, elems: &[&'static str]) -> Vec<String> {
    if !id.contains("{}") {
        return vec![id.to_owned()];
    }
    elems.iter().map(|cpp| id.replace("{}", cpp)).collect()
}

/// Every element type a routine is instantiated at, by the `fn` it fills in.
///
/// # Why the enumeration is the fixture's input now
///
/// A routine body is generic over its element type, so the template-id it
/// hands NVRTC is a `format!` with a hole in it:
///
/// ```ignore
/// &format!("::pie::mlp::swiglu<{}>", T::CPP)
/// ```
///
/// There is no whole id in the source to read any more, and that is the point:
/// the element type is stated ONCE, where the instantiation is chosen. So this
/// reads the choice from where it is made — `routine!(swiglu_bf16 =
/// swiglu::<bf16>)` — and fills the hole with what the crate says that type is
/// spelled, `<bf16 as Elem>::CPP`. The two halves of an id can no longer
/// disagree, because neither is written twice.
fn instantiated_at(text: &str) -> BTreeMap<String, BTreeSet<&'static str>> {
    let mut out: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
    for (at, _) in text.match_indices("::<") {
        let head = &text[..at];
        let name: String = head
            .chars()
            .rev()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if name.is_empty() {
            continue;
        }
        let Some(end) = text[at + 3..].find('>') else { continue };
        let args = &text[at + 3..at + 3 + end];
        for arg in args.split(',') {
            if let Some(cpp) = ELEMENTS.iter().find(|(rust, _)| *rust == arg.trim()) {
                out.entry(name.clone()).or_default().insert(cpp.1);
            }
        }
    }
    out
}

/// The element types a routine may be instantiated at, as the crate spells
/// them. Asked of the crate rather than written out, so a new element type is
/// one `impl Elem` and not two statements.
const ELEMENTS: &[(&str, &str)] = &[
    ("bf16", <kernels_cuda::jit::abi::bf16 as Elem>::CPP),
    ("f16", <kernels_cuda::jit::abi::f16 as Elem>::CPP),
    ("fp8_e4m3", <kernels_cuda::jit::abi::fp8_e4m3 as Elem>::CPP),
];

/// Every carried file the crate compiles, with the template-ids it asks for.
///
/// # Why this reads launch sites and not declarations
///
/// It read `Root::new("name", include_str!(..), ..)` with the `mod inst` beside
/// it, because that is where the two strings lived. They live at the FIRE now:
/// `ctx.launch("mlp/swiglu.cuh", "::pie::mlp::relu2<..>", ..)`
/// names its file and its template-id in one call, and the `static ROOT` and
/// the named constant that used to stand between are gone.
///
/// That makes the pairing better than it was rather than worse. The old reading
/// was PROXIMITY — the `mod inst` that happened to follow a declaration — which
/// is why a body firing a root declared in another module had to be stitched
/// back on by a `loose` pass. Here the two strings are arguments to one call.
///
/// The scope is the enclosing `fn` rather than the call, because 63 of these
/// choose their template-id in a `let` before firing it:
///
/// ```ignore
/// let instantiation = if vec { "..true_type.." } else { "..false_type.." };
/// ctx.launch("layout/embed.cuh", instantiation, ..)
/// ```
///
/// A `fn` naming exactly one carried file is what makes that attributable, and
/// 222 of the 227 that launch anything name exactly one. The rest are handled
/// below.
fn written() -> Vec<Job> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut files = Vec::new();
    walk(&src, &mut files);

    // Every `fn` in the crate, with the carried files it names, the
    // template-ids it names, the `fn`s it calls, and where it was read.
    struct Scope {
        site: String,
        named: BTreeSet<String>,
        wanted: Vec<String>,
        calls: BTreeSet<String>,
        launches: Vec<(String, String)>,
        /// Every carried file the whole `.rs` names, for a body that reaches
        /// its root through a `static` instead of naming the file.
        whole: BTreeSet<String>,
    }
    let mut scopes: Vec<Scope> = Vec::new();
    let mut module_orphans: Vec<String> = Vec::new();
    let abi = aggregates();
    let mut by_file: BTreeMap<String, (String, Vec<String>)> = BTreeMap::new();

    for path in files {
        let text = std::fs::read_to_string(&path).expect("a readable source file");
        let show = path.strip_prefix(&src).unwrap_or(&path).to_string_lossy().into_owned();
        // A file whose template-ids are the lattices' contributes none of them
        // here: they are `concat!` fragments rather than whole ids, and the
        // points that assemble them are compiled by `computed` instead.
        let lattice = ORPHANS.iter().any(|(at, _)| show.starts_with(at));
        let named_types = abi_spellings(&text);
        let is_template_id =
            |s: &String| is_template_id(s) && !abi.contains(s) && !named_types.contains(s);
        // Which element types each `fn` in this file is instantiated at, read
        // from its `routine!(name = body::<T>)` lines.
        let elems_of = instantiated_at(&text);

        // The roots the crate compiles, taken from the four places a carried
        // file can be named: a launch, the two constructors, and `carried`
        // itself. NOT from every `.cuh` literal -- `graph.rs` asserts that a
        // walk reaches `pie_device.cuh`, which is a header and not a root, and
        // `vision.rs` builds `format!("{}.cuh", ..)`. Both are literals ending
        // in `.cuh` and neither is a program source.
        for form in ["ctx.launch(", "Root::new(", "Root::of(", "Root::variant(", "source::carried("] {
            for (at, _) in text.match_indices(form) {
                let Some((args, _)) = group(&text[at + form.len() - 1..], '(', ')') else {
                    continue;
                };
                if let Some(name) = literals(args).into_iter().find(|s| is_carried_name(s)) {
                    by_file.entry(name).or_insert_with(|| (show.clone(), Vec::new()));
                }
            }
        }

        // Modules that name one file answer for everything inside them,
        // which is what covers a `mod inst` whose ids sit in a `const` array
        // rather than in any `fn`.
        let anchored = anchored_modules(&text);
        for ((from, to), file) in anchored.iter().filter(|_| !lattice) {
            let inside = &text[*from..*to];
            let ids: Vec<String> =
                literals(inside).into_iter().filter(&is_template_id).collect();
            if let Some(entry) = by_file.get_mut(file) {
                entry.1.extend(ids);
            }
        }
        let covered = |at: usize| anchored.iter().any(|((f, t), _)| at >= *f && at < *t);

        // Template-ids outside every `fn` AND outside every anchored module.
        // A file naming exactly one carried file answers for them.
        // Only names that were SEEDED as roots: a `.cuh` literal may also be a
        // header a test asserts a walk reaches, and that is not a compile.
        let whole_named: BTreeSet<String> = literals(&text)
            .into_iter()
            .filter(|s| is_carried_name(s) && by_file.contains_key(s))
            .collect();
        let in_bodies: BTreeSet<String> =
            bodies(&text).iter().flat_map(|(_, b)| literals(b)).collect();
        let loose: Vec<String> = if lattice {
            Vec::new()
        } else {
            literals(&text)
                .into_iter()
                .filter(|s| is_template_id(s) && !in_bodies.contains(s))
                .collect()
        };
        if !loose.is_empty() {
            let whole = &whole_named;
            let already: BTreeSet<String> = anchored
                .iter()
                .flat_map(|((f, t), _)| literals(&text[*f..*t]))
                .filter(&is_template_id)
                .collect();
            for id in loose {
                if already.contains(&id) {
                    continue;
                }
                if let Some(entry) =
                    whole.iter().next().filter(|_| whole.len() == 1).and_then(|f| by_file.get_mut(f))
                {
                    entry.1.push(id);
                } else {
                    module_orphans.push(format!("{show}: {id}"));
                }
            }
        }

        for (fname, body) in bodies(&text) {
            let at = body.as_ptr() as usize - text.as_ptr() as usize;
            if covered(at) {
                continue;
            }
            let lits = literals(body);
            let named: BTreeSet<String> =
                lits.iter().filter(|s| is_carried_name(s)).cloned().collect();
            // A `{}` is the element type the body is generic over. It is
            // filled from what this `fn` is instantiated at, which is stated
            // in `ROUTINES` and nowhere else.
            let fn_elems: Vec<&'static str> =
                elems_of.get(&fname).map(|s| s.iter().copied().collect()).unwrap_or_default();
            let wanted: Vec<String> =
                lits.iter().filter(|s| is_template_id(s)).flat_map(|s| fill(s, &fn_elems)).collect();
            if named.is_empty() && wanted.is_empty() {
                continue;
            }
            // The launches that name both, for a body that names more than one
            // file and so cannot be attributed wholesale.
            let mut launches = Vec::new();
            for (at, _) in body.match_indices("ctx.launch(") {
                let Some((args, _)) = group(&body[at..], '(', ')') else { continue };
                let pair = literals(args);
                if let (Some(f), Some(i)) = (pair.first(), pair.get(1)) {
                    if is_carried_name(f) && is_template_id(i) {
                        for one in fill(i, &fn_elems) {
                            launches.push((f.clone(), one));
                        }
                    }
                }
            }
            let calls = body
                .match_indices('(')
                .filter_map(|(at, _)| {
                    let head = &body[..at];
                    let name: String = head
                        .chars()
                        .rev()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    (!name.is_empty()).then_some(name)
                })
                .collect();
            scopes.push(Scope {
                whole: if whole_named.len() == 1 { whole_named.clone() } else { BTreeSet::new() },
                site: format!("{show}:{fname}"),
                // A lattice file keeps only what a launch names OUTRIGHT.
                // `ctx.launch("attn/attention_xqa.cuh", "..build_xqa_metadata", ..)`
                // is a written root like any other; the arms beside it are
                // `concat!` fragments and belong to `computed`.
                named: if lattice { BTreeSet::new() } else { named },
                wanted: if lattice { Vec::new() } else { wanted },
                calls,
                launches,
            });
        }
    }

    // A `fn` that names exactly one carried file answers for every template-id
    // in it. 222 of the 227 that launch anything are this shape.
    let anchored: BTreeMap<String, String> = scopes
        .iter()
        .filter(|s| s.named.len() == 1)
        .map(|s| {
            let name = s.site.rsplit(':').next().unwrap_or("").to_owned();
            (name, s.named.iter().next().expect("one").clone())
        })
        .collect();

    let mut orphans: Vec<String> = module_orphans;
    for scope in &scopes {
        let fname = scope.site.rsplit(':').next().unwrap_or("").to_owned();
        // A launch naming both outright is direct evidence and is taken
        // whatever else the body does. The arms below attribute what is left;
        // the duplicates this can produce are folded out at the end.
        for (file, inst) in &scope.launches {
            if let Some(entry) = by_file.get_mut(file) {
                entry.1.push(inst.clone());
            }
        }
        match scope.named.len() {
            1 => {
                let file = scope.named.iter().next().expect("one");
                if let Some(entry) = by_file.get_mut(file) {
                    entry.1.extend(scope.wanted.iter().cloned());
                }
            }
            // Names no file. A template-id here is either handed to a helper
            // that launches it (`chunk_prefill(ctx, "..fla<..>", "..<..>", ..)`)
            // or returned to a caller that does (`warp_instantiation`). One
            // step of the call graph in each direction settles which, and a
            // template-id nothing can be attributed to is a failure below.
            0 if !scope.wanted.is_empty() => {
                let mut candidates: BTreeSet<&String> = scope
                    .calls
                    .iter()
                    .filter_map(|callee| anchored.get(callee))
                    .collect();
                if candidates.is_empty() {
                    candidates = scopes
                        .iter()
                        .filter(|other| other.calls.contains(&fname))
                        .filter_map(|other| other.named.iter().next())
                        .collect();
                }
                // Still nothing: a body that reaches its root by a `static`
                // rather than by naming the file -- `graph::warm` resolves
                // `&ROOT` to warm the two arming kernels. The FILE names one
                // carried file and only one, so that is the answer.
                if candidates.is_empty() {
                    candidates = scope.whole.iter().collect();
                }
                if let Some(entry) = candidates
                    .iter()
                    .next()
                    .filter(|_| candidates.len() == 1)
                    .and_then(|f| by_file.get_mut(*f))
                {
                    entry.1.extend(scope.wanted.iter().cloned());
                } else {
                    orphans
                        .extend(scope.wanted.iter().map(|w| format!("{}: {w}", scope.site)));
                }
            }
            0 => {}
            // More than one: pair by the launch that names both.
            _ => {
                for (file, inst) in &scope.launches {
                    if let Some(entry) = by_file.get_mut(file) {
                        entry.1.push(inst.clone());
                    }
                }
                let paired: BTreeSet<&String> = scope.launches.iter().map(|(_, i)| i).collect();
                orphans.extend(
                    scope
                        .wanted
                        .iter()
                        .filter(|w| !paired.contains(w))
                        .map(|w| format!("{}: {w}", scope.site)),
                );
            }
        }
    }

    // The scanner is this fixture's own subject. A template-id nothing can be
    // attributed to is one nothing would compile, and a scanner that silently
    // dropped them would turn this into a test that passes by finding nothing.
    orphans.retain(|o| !ORPHANS.iter().any(|(at, _)| o.starts_with(at)));
    orphans.sort();
    orphans.dedup();
    assert!(
        orphans.is_empty(),
        "{} template-id(s) could not be attributed to a carried file, so nothing \
         here would compile them: {orphans:#?}",
        orphans.len()
    );

    assert!(
        by_file.len() > 40,
        "only {} carried files were found named under `src/`, which is fewer \
         than this crate compiles -- the scan has stopped matching",
        by_file.len()
    );

    let mut bodiless: Vec<String> = by_file
        .iter()
        .filter(|(file, (_, wanted))| {
            wanted.is_empty()
                && !NO_INSTANTIATIONS.iter().any(|(n, _)| format!("{n}.cuh") == **file)
        })
        .map(|(file, (site, _))| format!("{file} ({site})"))
        .collect();
    bodiless.sort();
    assert!(
        bodiless.is_empty(),
        "{} carried file(s) are named by the crate and nothing is instantiated \
         out of them, so either the launch that did was deleted or the file is \
         named for some other reason: {bodiless:?}",
        bodiless.len()
    );

    by_file
        .into_iter()
        .filter(|(_, (_, wanted))| !wanted.is_empty())
        .map(|(file, (site, mut wanted))| {
            wanted.sort();
            wanted.dedup();
            let root = Root::of(String::leak(file.clone())).unwrap_or_else(|| {
                panic!("{site}: `{file}` is named as a root and nothing carries it")
            });
            Job {
                text: std::fs::read_to_string(kernels.join(&file))
                    .unwrap_or_else(|why| panic!("{site}: `{file}`: {why}")),
                headers: root.headers.set(),
                options: root.options.iter().map(|&o| o.to_owned()).collect(),
                site: format!("{site} -> {file}"),
                name: root.name.to_owned(),
                wanted,
            }
        })
        .collect()
}

/// Every `.rs` under `dir`, depth first and in name order.
///
/// Two exclusions, and both are the same mistake in different clothes.
/// `src/jit` DEFINES the root constructors, and their doc comments spell the
/// calls this fixture greps for. `src/source.rs` holds the carried set itself —
/// every `.cuh` under `kernels/` by name — and reading it as a list of roots
/// would seed this fixture with the whole tree, which reaches everything and
/// proves nothing.
fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    if dir.file_name().is_some_and(|n| n == "jit") {
        return;
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("a readable directory")
        .map(|e| e.expect("a readable entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs")
            && path.file_name().is_none_or(|n| n != "source.rs")
        {
            out.push(path);
        }
    }
}

// ===========================================================================
// The computed roots: asked of the crate
// ===========================================================================

/// Both lattices, each point with the arms it was built to answer.
fn computed() -> Vec<Job> {
    use kernels_cuda::attn::{fa2, xqa};

    let mut out = Vec::new();
    for point in &fa2::DECODE {
        let site = format!("fa2::DECODE hd{} g{}", point.head_dim, point.group_size);
        let wanted = point.arms.iter().map(|&a| a.to_owned()).collect();
        out.push(Job::from_root(site, &point.root, wanted));
    }
    for point in &fa2::PREFILL {
        let site = format!(
            "fa2::PREFILL hd{} q{} kv{}",
            point.head_dim, point.cta_tile_q, point.num_mma_kv
        );
        let wanted = point.arms.iter().map(|&a| a.to_owned()).collect();
        out.push(Job::from_root(site, &point.root, wanted));
    }
    for (nth, root) in xqa::ROOTS.iter().enumerate() {
        let site = format!("xqa::ROOTS[{nth}]");
        out.push(Job::from_root(site, root, vec![xqa::inst::MHA[nth].to_owned()]));
    }
    out
}

// ===========================================================================

#[test]
fn every_instantiation_a_body_names_compiles() {
    let Ok(have) = nvrtc::version() else {
        eprintln!("SKIPPED: libnvrtc will not load, so nothing here can be compiled");
        return;
    };
    let arch = kernels_cuda::jit::cache::arch().unwrap_or("compute_89");

    let (written, computed) = (written(), computed());
    let count = |jobs: &[Job]| jobs.iter().map(|j| j.wanted.len()).sum::<usize>();
    eprintln!(
        "nvrtc {have} targeting {arch}: {} written roots ({} instantiations), {} computed ({})",
        written.len(),
        count(&written),
        computed.len(),
        count(&computed)
    );

    // One thread per root. The lattices are FlashInfer, whose points take tens
    // of seconds each and would otherwise make this too slow to leave on.
    let jobs: Vec<Job> = written.into_iter().chain(computed).collect();
    let failed: BTreeMap<String, String> = std::thread::scope(|scope| {
        let running: Vec<_> =
            jobs.iter().map(|job| scope.spawn(move || (job, job.compile(arch)))).collect();
        running
            .into_iter()
            .filter_map(|handle| match handle.join().expect("a compile thread") {
                (_, Ok(())) => None,
                (job, Err(why)) => Some((format!("{} ({})", job.name, job.site), why)),
            })
            .collect()
    });

    assert!(
        failed.is_empty(),
        "{} of {} roots would not compile:\n\n{}",
        failed.len(),
        jobs.len(),
        failed
            .iter()
            .map(|(what, why)| format!("── {what} ──\n{why}\n"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
