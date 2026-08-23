//! Every `.slang` this crate carries is reached by something.
//!
//! `build.rs` walks the whole `kernels/` tree and carries each file into the
//! binary as a `SOURCES` entry, so a shader nobody includes and nobody
//! instantiates costs nothing at runtime and warns about nothing at build
//! time. It just sits there being cited.
//!
//! That is not hypothetical. `norm/rms_params.slang` held the RMS family's
//! parameter blocks until every one of its readers moved its scalars into a
//! `[[vk::push_constant]]` range. Nothing included it afterwards, so its
//! `RmsRopeParams` went on stating "It is one block and not a block plus a
//! push range" about six modules that by then had a push range and no block —
//! a false claim, in a file that could not be reached to be falsified, with
//! three live shaders still citing the path in their own history.
//!
//! A source is reached if it declares a variant (`// pie:instantiate`, which
//! is how `build.rs` finds an entrypoint) or if another source `#include`s it.

/// The `#include "…"` targets in one source, as tree-relative paths.
fn includes(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("#include")?;
            let rest = rest.trim_start().strip_prefix('"')?;
            let (path, _) = rest.split_once('"')?;
            Some(path.to_string())
        })
        .collect()
}

#[test]
fn no_source_is_carried_that_nothing_includes_and_nothing_instantiates() {
    let sources = kernels_vulkan::runtime::sources();
    assert!(!sources.is_empty(), "the tree cannot be empty");

    let included: std::collections::BTreeSet<String> = sources
        .iter()
        .flat_map(|(_, text)| includes(text))
        .collect();

    let orphans: Vec<&str> = sources
        .iter()
        .filter(|(name, text)| {
            !text.contains("pie:instantiate") && !included.contains(*name)
        })
        .map(|&(name, _)| name)
        .collect();

    assert!(
        orphans.is_empty(),
        "carried but unreachable — delete them, or include them from the \
         shader whose ABI they state: {orphans:?}"
    );
}

/// The other half of the same rule: an `#include` naming a file the tree does
/// not carry would fail the Slang compile, but only for the variants that
/// reach it, and only on a machine with `slangc`. This says it plainly.
#[test]
fn every_include_names_a_source_the_tree_carries() {
    let sources = kernels_vulkan::runtime::sources();
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
