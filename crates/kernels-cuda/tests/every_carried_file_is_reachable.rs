//! Every file the carried set holds is one an `#include` can arrive at.
//!
//! `src/source.rs` lists **every** file the crate carries, and that is the point of
//! it: a header set assembled by hand is a
//! set somebody has to remember to add to, and forgetting is not a compile
//! error anywhere — it is an NVRTC *"could not open source file"* at the first
//! fire of whatever needed it, on a machine with a GPU. The rule buys that,
//! and its converse is what it costs. **Carried is a superset of
//! compiled-against**, and until this fixture nothing measured the gap.
//!
//! The omission itself is caught next door, by
//! `source::every_device_include_resolves`: it is quantified over the carried
//! set's own quoted `#include`s, so a header added to the tree and included by
//! something already in the set fails at `cargo test` and without a GPU. This
//! fixture is the other half, and neither needs a build script to run.
//!
//! A file in the gap is not inert. `LIBRARY` is handed to every compile in the
//! crate, so a dead `.cuh` is copied into `nvrtcCreateProgram` for a `norm`
//! kernel that cannot name it; and `Root::key` folds `source::digest` over the
//! set a root is handed, so editing one invalidates every cached cubin that set
//! ever produced, on every machine.
//! It costs a reader more than either: a `.cuh` in this tree reads as text some
//! kernel compiles, because that is what all the others are.
//!
//! # The class, which is what this is for
//!
//! `source::reachable` was written for the other direction — *does every
//! include a root reaches resolve* — and its caller,
//! `source::every_include_reachable_from_a_unit_resolves`, died in the sweep
//! that deleted the unit world. `kernels/ptir/tier0.cuh`, 1,156 lines that no
//! Rust in the workspace names, walked in behind it and survived the cleanup,
//! because the tool that would have named it had no reader left. Both
//! directions are asserted below off one walk, so the next orphan fails at
//! `cargo test` rather than shipping.
//!
//! It is a fixture and not a `mod tests` in `src/source.rs`, where the check it
//! replaces lived and where `every_device_include_resolves` — its other half —
//! still is. The subject moved: that claim is quantified over the carried set
//! and needs nothing else, and this one is about the relation between two
//! trees, so it reads `src/` off the filesystem the way
//! `every_instantiation_compiles` does. A library module that walks its own
//! crate's directory is the wrong place for it.
//!
//! # Why the roots are read out of the source and not asked of the crate
//!
//! Asking would be sounder and it is not available. `FAMILIES`, a `Family`'s
//! `routines`, and `routine()` all answer with a `Routine`, and a `Routine` is
//! a name, a signature and a `body` — nothing on it names the `Root` its body
//! compiles, because the body reaches its root by PATH (`rmsnorm::ROOT`) at the
//! point of launch and no registry stands between them. The two lattices are
//! the exception: `fa2::{DECODE, PREFILL}` and `xqa::ROOTS` are public arrays
//! of `Root`, which is why `every_instantiation_compiles` reads those two at
//! run time. Every other root is a `pub static ROOT` inside a per-kernel
//! module, reachable only by naming each one here — the hand-list this crate's
//! principle exists to refuse, and the exact thing that lets an orphan hide.
//!
//! So: the source. But **not** `every_instantiation_compiles`'s parser, and
//! the difference is the question rather than the effort. That fixture
//! reconstructs a COMPILE — name, text, options, header set — so its
//! declaration forms defeat a naive reading and an unreadable `.options(CONST)`
//! has to fail rather than skip, on pain of compiling a root under options it
//! never uses. This one needs one fact: which carried files the Rust half
//! hands NVRTC as a program source. Every one of them arrives as the carried
//! name a root is declared with — `Root::new`, `Root::variant` and
//! `source::carried` — and that name is a string literal or it does not
//! compile, because `carried` resolves it during const evaluation. So the
//! parse is one token followed by one string literal and it has no form to
//! miss.
//!
//! # Why the two bracket styles are followed differently
//!
//! `source::reachable` follows quoted directives only, and in this tree that is
//! not an oversight but a spelling convention with a comment defending it —
//! `kernels/attn/attention_xqa_mha.cuh:245` sets out the whole of it. NVRTC
//! matches both styles literally against `includeNames[]`, measured, so the
//! choice is free as far as a compile goes and the tree spends it on saying
//! whether the file EXISTS: `#include "xqa/mha.cuh"` is carried and under
//! the walk, `#include <attn/xqa/mha_sm90.cuh>` is deliberately not carried and
//! stays angled so that no walk reports it missing. `<algorithm>`, `<mma.h>`
//! and `<crt/cuda_tile.h>` are the same case for the toolkit's own headers.
//! Refusing on an angled spelling would refuse all of those, which is why
//! `reachable` does not read them.
//!
//! `shim/` is the one place where the convention runs the other way, and it
//! has to: a shim exists to answer `#include <cuda_bf16.h>` ahead of the real
//! toolkit header, so it is carried AND angled, and all but two of them are
//! reached by no quoted directive anywhere — `cuda_fp16.h` and `cuda_fp8.h`
//! are the exceptions. Dropping the angled edge here would call the rest
//! dead; exempting the shim tree wholesale would leave
//! a genuinely dead shim uncovered. So the walk follows angled spellings the
//! set answers and ignores the ones it does not — which is the rule the
//! aliases in `src/source.rs` were built under, and it makes the edges walked
//! here the edges the set was assembled from.
//!
//! `quoted_includes` reads column zero only. Nothing here defends that,
//! because an indented quoted directive would make a live file look dead —
//! a failure naming a file that plainly has an includer, which is loud rather
//! than silent, and unlike the reverse it cannot hide anything.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use kernels_cuda::source::{ALL_HEADERS, Header, SHIM, reachable};

/// Where a carried file's name is relative to: the tree its walk started at.
///
/// `SHIM` is `shim/` and the other two groups are both `kernels/`, so
/// membership in one array decides it. No spelling names two different files
/// across the three groups, which
/// is what makes asking one array an answer rather than a first guess. This
/// decides a path in a message and nothing else, so being wrong would cost a
/// reader one `ls` — it is not load-bearing for the claim.
fn tree_of(name: &str) -> &'static str {
    if SHIM.iter().any(|h| h.name == name) { "shim" } else { "kernels" }
}

/// The path a reader can open, for a file carried under `names`.
///
/// A file is carried under its tree-relative path and under every relative
/// spelling some directive reaches it by — `attn/flashinfer/cp_async.cuh` is
/// also `../cp_async.cuh` and `cp_async.cuh`, because NVRTC matches the
/// literal string in the directive and has no notion of "relative to the
/// includer". Exactly one of those names is the file, and the filesystem says
/// which: `build.rs` registers an alias only for a spelling that names no
/// carried file, and it carries every file, so no alias can land on one.
fn path_of(names: &[&str]) -> String {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sorted = names.to_vec();
    sorted.sort_unstable();
    let real = sorted
        .iter()
        .find(|name| crate_root.join(tree_of(name)).join(name).is_file())
        .unwrap_or_else(|| {
            panic!(
                "none of {sorted:?} names a carried file, so the carried \
                 set and the tree it was walked from disagree"
            )
        });
    let also: Vec<&&str> = sorted.iter().filter(|n| n != &real).collect();
    let path = format!("{}/{real}", tree_of(real));
    if also.is_empty() { path } else { format!("{path} (also carried as {also:?})") }
}


/// Every carried file a root declaration under `src/` names, as
/// `(site, carried name)`.
///
/// The site is carried so a failure can be acted on without a grep, and it is
/// the path under `src/` rather than the file's own name, because a tree with
/// a `mod.rs` per family has more than one file called `mod.rs`.
///
/// # Why this reads declarations and not `include_str!`
///
/// It read `include_str!` until the carried set stopped being generated. While
/// `source.rs` reached its table through `include!(concat!(env!("OUT_DIR"),
/// …))` the only literal `include_str!`s under `src/` were the roots, so
/// scanning for them found the program sources and nothing else. Hardcoding
/// the table put an `include_str!` in this crate for EVERY carried file, which
/// makes that scan answer with the whole set — every file a root, every file
/// trivially reached, and an assertion below quantified over nothing. A
/// fixture that cannot fail is worse than no fixture, so the seed is now the
/// thing it always meant: a `Root`.
///
/// The three forms are the three ways a root names its file, and all of them
/// end in a literal because `source::carried` is a `const fn` over
/// `LIBRARY` — a computed name would not compile, so there is no fourth form
/// to miss. `source::carried(` and not `carried(`: a bare one also matches the
/// tail of `fn every_include_the_root_reaches_is_carried(`. The name needs no
/// resolution either: it is already the spelling the carried set is keyed by.
fn root_sites() -> Vec<(String, String)> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = crate_root.join("src");
    let mut files = Vec::new();
    walk(&src, &mut files);
    walk(&crate_root.join("tests"), &mut files);

    let mut out = Vec::new();
    for file in files {
        let under_src = file.strip_prefix(crate_root).expect("the walk started in the crate");
        if under_src.starts_with("src/jit")
            || under_src == Path::new("src/source.rs")
            || under_src == Path::new("tests/every_carried_file_is_reachable.rs")
            || under_src == Path::new("tests/every_instantiation_compiles.rs")
        {
            continue;
        }
        let text = std::fs::read_to_string(&file).expect("a readable source file");
        let show = under_src.to_string_lossy().into_owned();
        // A launch names its file too, and after the inlining that is where
        // most of them are. `ctx.launch("mlp/swiglu.cuh", "::pie..", ..)` was
        // the spelling; `Fire::at("mlp/swiglu.cuh", "::pie..")` is the one a
        // body uses now, and a scanner that knows only the first found
        // fifteen roots where the crate has a hundred and thirty.
        for form in [
            "Root::new(",
            "Root::variant(",
            "source::carried(",
            "ctx.launch(",
            "Fire::at(",
        ] {
            for (at, _) in text.match_indices(form) {
                let line = text[..at].bytes().filter(|&b| b == b'\n').count() + 1;
                let Some((args, _)) = group(&text[at + form.len() - 1..], '(', ')') else {
                    continue;
                };
                let _ = line;
                if let Some(literal) = literals(args).into_iter().find(|s| is_carried_name(s)) {
                    out.push((format!("{show}:{line}"), literal));
                }
            }
        }
    }
    out
}

/// A literal that names a carried file rather than a template-id or prose.
fn is_carried_name(s: &str) -> bool {
    s.ends_with(".cuh") && !s.contains(' ') && !s.contains('{') && !s.starts_with("::")
}

/// The string literals in `span`, in order.
fn literals(span: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = span.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let mut j = i + 1;
            let mut lit = String::new();
            while j < bytes.len() && bytes[j] != b'"' {
                if bytes[j] == b'\\' && j + 1 < bytes.len() {
                    if bytes[j + 1] == b'\n' {
                        j += 2;
                        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                            j += 1;
                        }
                        continue;
                    }
                    j += 2;
                    continue;
                }
                lit.push(bytes[j] as char);
                j += 1;
            }
            out.push(lit);
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// The span between `open` and its matching `close`, and how far past it ends.
fn group(span: &str, open: char, close: char) -> Option<(&str, usize)> {
    let start = span.find(open)?;
    let mut depth = 0usize;
    for (at, ch) in span[start..].char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some((&span[start + 1..start + at], start + at + 1));
            }
        }
    }
    None
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|why| panic!("{dir:?} reads: {why}"))
        .map(|e| e.expect("a readable entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The carried headers the Rust half compiles as program sources.
///
/// A root names a carried file or it does not compile — [`source::carried`] is
/// a `const fn` over `LIBRARY` — so the lookup here cannot fail for a root that
/// exists. It is still checked rather than assumed: this fixture reads the
/// name out of the SOURCE, and a name the set does not hold means the scan
/// matched something that is not a root declaration.
fn roots() -> Vec<(String, &'static Header)> {
    let mut out = Vec::new();
    for (site, name) in root_sites() {
        let header = ALL_HEADERS.iter().find(|h| h.name == name).unwrap_or_else(|| {
            panic!(
                "{site}: `{name}` is compiled as a program source and the carried \
                 set does not hold it, so nothing it includes of its own tree can \
                 resolve"
            )
        });
        out.push((site, header));
    }
    out
}


/// Every angled `#include <…>` in `source`, in order of appearance.
///
/// `source::quoted_includes` with the angled half instead of the quoted one,
/// leading whitespace and
/// all: upstream indents directives inside `#if` blocks, and the aliases in the
/// carried set were built with that tolerance, so a stricter reading here would
/// walk a smaller graph than the one the set was assembled from.
fn angled_includes(source: &str) -> Vec<&str> {
    source
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix('#')?.trim_start().strip_prefix("include")?;
            rest.trim_start().strip_prefix('<')?.split('>').next()
        })
        .collect()
}

/// Every carried name some root arrives at, the roots themselves included.
///
/// `reachable` already answers transitively for quoted directives, so its
/// result would be the whole answer if angled ones did not exist. They do, and
/// a header reached through one has quoted directives of its own, so this is a
/// fixpoint over both edge kinds rather than one pass per root.
fn reached(roots: &[(String, &'static Header)]) -> BTreeSet<&'static str> {
    let mut seen: BTreeSet<&'static str> = BTreeSet::new();
    let mut queue: Vec<&'static Header> = Vec::new();
    for (_, header) in roots {
        if seen.insert(header.name) {
            queue.push(header);
        }
    }

    while let Some(header) = queue.pop() {
        let quoted = reachable(header.name, header.text, ALL_HEADERS)
            .unwrap_or_else(|why| panic!("{why}"));
        let angled = angled_includes(header.text);
        for name in quoted.into_iter().chain(angled) {
            let Some(next) = ALL_HEADERS.iter().find(|h| h.name == name) else {
                continue;
            };
            if seen.insert(next.name) {
                queue.push(next);
            }
        }
    }
    seen
}

// ===========================================================================

/// Nothing in the device tree is carried that no compile can reach.
///
/// The claim is about FILES, so the entries are grouped by their text first: a
/// file carried under three spellings is reached if any one of them is, and
/// comparing names alone would report `attn/flashinfer/attention/mask.cuh` dead
/// for the sole reason that its includer spells it `mask.cuh`.
///
/// Reachability is asked against `ALL_HEADERS` for every root, not against the
/// root's own `Headers` choice. The question here is whether ANY compile in
/// this crate can arrive at a file; whether one particular root's includes
/// resolve under the set it actually gets is
/// `every_instantiation_compiles`'s, which compiles each root under its real
/// options and its real header set.
#[test]
fn every_carried_file_is_reachable_from_a_root() {
    let roots = roots();
    assert!(
        !roots.is_empty(),
        "no root declaration under `src/` names a carried file, so this walk \
         has no root to start from and proves nothing"
    );

    let distinct: BTreeSet<&'static str> = roots.iter().map(|(_, h)| h.name).collect();
    assert!(
        distinct.len() * 2 < ALL_HEADERS.len(),
        "the walk starts at {} of the {} carried names, which is not a set of \
         program sources but most of the tree. A seed that large reaches \
         everything and proves nothing.",
        distinct.len(),
        ALL_HEADERS.len()
    );

    let reached = reached(&roots);
    assert!(
        reached.len() > distinct.len(),
        "the walk started at {} roots and reached nothing beyond any of them. \
         A root that includes nothing is possible and a tree of them is not, \
         so this is the include scanner having stopped matching rather than a \
         graph with no edges.",
        distinct.len()
    );

    let mut files: BTreeMap<&'static str, Vec<&'static str>> = BTreeMap::new();
    for header in ALL_HEADERS {
        files.entry(header.text).or_default().push(header.name);
    }

    let mut dead: Vec<String> = files
        .values()
        .filter(|names| !names.iter().any(|name| reached.contains(name)))
        .map(|names| path_of(names))
        .collect();
    dead.sort();

    assert!(
        dead.is_empty(),
        "{} of the {} carried files are in the binary and \
         reachable from none of the {} program sources `src/` compiles. Each \
         is bytes in every process, a term in every cache key, and text a \
         reader will take for a kernel:\n  {}",
        dead.len(),
        files.len(),
        distinct.len(),
        dead.join("\n  ")
    );
}
