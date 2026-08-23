//! Every `.wgsl` this crate carries is reached by something.
//!
//! The twin of `kernels-vulkan/tests/every_source_is_reached.rs`, and it exists
//! for the same reason: `build.rs` walks the whole `kernels/` tree and carries
//! each file into the binary as a `SOURCES` entry, so a shader nobody includes
//! and nobody instantiates costs nothing at runtime and warns about nothing at
//! build time. It just sits there being cited.
//!
//! # What it found
//!
//! `moe/params.inc.wgsl` held `RouterParams`, `ExpertCombineParams` and
//! `MoeRouteParams` -- the three storage blocks the five MoE routing arms bound
//! on `@group(0)`. They do not bind them any more. `moe/route.wgsl`'s own
//! header says so in capitals: every one of those fields is a `Const<u32>` mark
//! in its routine's signature now, packed into the `@group(1)` uniform block by
//! `driver-wgpu::lowering::routine::bind`.
//!
//! Nothing has included the file since. Its header opened with "This tree is
//! the last one still reading them from a block", which was true when the Slang
//! and Metal copies were deleted and false by the time the WGSL arms followed
//! them -- a claim about the present tense, in a file with no reader left to
//! falsify it. It was even edited, for an unrelated citation, without anyone
//! noticing it was unreachable. That is the whole failure mode: an orphan does
//! not decay into a compile error, it decays into confident wrong prose.
//!
//! # What "reached" means here
//!
//! A source is reached if it declares a variant (`// pie:instantiate`, which is
//! how `build.rs` finds an entrypoint) or if another source `//#include`s it.
//! There is no third way in on this backend: `Fire::at` names a file that
//! stamps an entrypoint, and a `.inc.wgsl` stamps none, so an include file with
//! no includer is unreachable by construction.

/// The `//#include "…"` targets in one source, as tree-relative paths.
///
/// The directive is a COMMENT to WGSL -- the language has no preprocessor --
/// and `preproc.rs` expands it before the module ever reaches naga. So a
/// mistyped path is not a syntax error anywhere; it is a line that silently
/// stays a comment, which is the second test's subject.
fn includes(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("//#include")?;
            let rest = rest.trim_start().strip_prefix('"')?;
            let (path, _) = rest.split_once('"')?;
            Some(path.to_string())
        })
        .collect()
}

#[test]
fn no_source_is_carried_that_nothing_includes_and_nothing_instantiates() {
    let sources = kernels_wgpu::SOURCES;
    assert!(!sources.is_empty(), "the tree cannot be empty");

    let included: std::collections::BTreeSet<String> = sources
        .iter()
        .flat_map(|(_, text)| includes(text))
        .collect();

    let orphans: Vec<&str> = sources
        .iter()
        .filter(|(name, text)| !text.contains("pie:instantiate") && !included.contains(*name))
        .map(|&(name, _)| name)
        .collect();

    assert!(
        orphans.is_empty(),
        "carried but unreachable -- delete them, or include them from the \
         shader whose ABI they state: {orphans:?}"
    );
}

/// The other half of the same rule: an `//#include` naming a file the tree does
/// not carry.
///
/// `preproc.rs` does report this, as `Malformed::Unincluded`, but only for the
/// variants that reach the line and only once something asks for one. A source
/// whose only bad include sits behind a `//#if` for a tier this machine does
/// not have is never expanded and never complains. This says it plainly, for
/// every file, with no device involved.
#[test]
fn every_include_names_a_source_the_tree_carries() {
    let sources = kernels_wgpu::SOURCES;
    let carried: std::collections::BTreeSet<&str> =
        sources.iter().map(|&(name, _)| name).collect();

    let mut dangling = Vec::new();
    for (name, text) in sources {
        for target in includes(text) {
            if !carried.contains(target.as_str()) {
                dangling.push(format!("{name} includes {target}"));
            }
        }
    }
    assert!(dangling.is_empty(), "includes with no source: {dangling:?}");
}
