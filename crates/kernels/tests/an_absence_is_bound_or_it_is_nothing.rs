//! **A routine that resolves an absence must bind it.**
//!
//! [`Asks::absent`] exists for one shape: an argument list is positional, so a
//! slot a routine deliberately leaves empty still occupies a cell, and the
//! body says so where it says everything else about its arguments. It is
//! `resolve(Ty::Buf, Source::Lit(Lit::Null))` -- the launcher supplying its
//! own null, written down instead of implied.
//!
//! A call whose result is DISCARDED is therefore not an absence. It is a
//! resolve that reaches the plane's binder, mints whatever that plane mints
//! for a null, drops it, and leaves behind exactly one observable effect: the
//! `?`, which can refuse a dispatch over a slot nothing was ever going to
//! pass. An absence that is not bound cannot make a kernel more correct and
//! can make one fail.
//!
//! # How twenty-two of them got there, and why Metal had none
//!
//! Metal's `kv_write.metal` numbers its buffers 0,1,2,3,5,10,12,13,14,15 and
//! leaves 4, 6-9 and 11 as HOLES -- a shared ring ABI the kernel does not
//! read, whose indices the argument list still has to reach past. Metal binds
//! all seven, because on that plane they are real cells. The `slang` and
//! `wgsl` siblings of the same kernel were written without the holes: their
//! parameter lists are nine long and dense.
//!
//! Both ports were made from the Metal body, and both trimmed the fire list
//! to nine while leaving the seven `let _ring_n = ctx.absent()?;` lines above
//! it. `_biases`, `_bias`, `_per_expert_scale` and four `_sinks` arrived the
//! same way -- slots the OTHER instantiation of a template fills, named in
//! the body they were copied from. Fourteen in `kernels-vulkan`, eight in
//! `kernels-wgpu`, none in `kernels-metal`.
//!
//! That the count is zero on exactly one backend is the finding, not a
//! coincidence: Metal grew `dispatch_matches_the_shader`, which compares
//! every bind against the declaration it lands on, so a Metal body whose
//! locals and whose fire list disagree is caught by the comparison the other
//! two do not have. This file is the part of that comparison the other two
//! CAN have without a preprocessor -- it needs no shader at all, only the
//! observation that a value resolved and dropped was never an argument.
//!
//! Not one of the twenty-two carried a sentence saying why it was empty,
//! which is the tell. Every deliberate absence in `kernels-metal` does.
//!
//! # The count that settled it
//!
//! Counting after the deletion corrected the paragraph above, which is worth
//! leaving in rather than smoothing over. `kernels-vulkan` is now at ZERO
//! live `absent()` calls -- not "fourteen of twenty-five were residue" but
//! ALL FOURTEEN of its live calls were, because that backend's `slang`
//! shaders have no holes anywhere, so it never needed the vocabulary at all.
//! The thirty that remain are `kernels-wgpu`'s seventeen and
//! `kernels-metal`'s thirteen, and both of those planes really do have
//! declared slots with nothing to put in them.
//!
//! So the shape of the defect was not "a port kept some of what it copied".
//! It was "a port kept a construct its own shader language made unnecessary",
//! and the two backends that did it did it for different amounts.

use std::fs;
use std::path::{Path, PathBuf};

/// The three crates whose bodies state their own argument lists.
///
/// `kernels-cuda` is not here: it did not cross to this vocabulary and states
/// its arguments a different way, so the pattern below would find nothing and
/// say so by passing, which is worse than not looking.
const BACKENDS: &[&str] = &["kernels-metal", "kernels-vulkan", "kernels-wgpu"];

fn sources(crate_name: &str) -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crates dir")
        .join(crate_name)
        .join("src");
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn no_body_resolves_an_absence_it_does_not_bind() {
    let mut dropped: Vec<String> = Vec::new();
    let mut bound = 0_usize;

    for backend in BACKENDS {
        let files = sources(backend);
        assert!(
            !files.is_empty(),
            "{backend}/src has no .rs files -- this test would pass by \
             reading nothing. If a target directory was shared with a `git \
             worktree`, `CARGO_MANIFEST_DIR` is baked from the wrong tree; \
             `cargo clean -p kernels` is the fix."
        );
        for file in files {
            let text = fs::read_to_string(&file).expect("a source file");
            for (n, line) in text.lines().enumerate() {
                let code = line.split_once("//").map_or(line, |(before, _)| before);
                if !code.contains("absent()") {
                    continue;
                }
                let trimmed = code.trim();
                // A binding whose name starts with `_` is one Rust will not
                // warn about and nothing can read: the ONLY way to spend an
                // `absent()` is to pass it, so this is the whole pattern.
                if trimmed.starts_with("let _") {
                    let name = trimmed
                        .trim_start_matches("let ")
                        .split([' ', ':', '='])
                        .next()
                        .unwrap_or("?");
                    dropped.push(format!(
                        "  {}:{} binds `{name}`, which nothing can read",
                        file.display(),
                        n + 1
                    ));
                } else {
                    bound += 1;
                }
            }
        }
    }

    assert!(
        dropped.is_empty(),
        "{} resolved absence(s) are never bound. Each one reaches the \
         plane's binder for a null, drops it, and can refuse a dispatch over \
         a slot nothing passes. Either bind it in the fire list or delete \
         it:\n{}",
        dropped.len(),
        dropped.join("\n")
    );

    // A CENSUS, so that the check above cannot pass by finding nothing.
    //
    // Seventeen in `kernels-wgpu`, thirteen in `kernels-metal`, and none at
    // all in `kernels-vulkan`, whose `slang` shaders declare no holes. A
    // floor was the first spelling and it was worthless: measured after the
    // deletion the floor's number WAS the count, so it could only ever fail
    // on a legitimate removal. The number moving is not a fault; the number
    // moving without anyone noticing is, which is what an equality says and a
    // floor does not.
    assert_eq!(
        bound, 30,
        "the number of bound absences moved. That is allowed -- a real hole \
         gained or lost one -- but it is not allowed to move silently, \
         because a spelling change that made this scan blind would land here \
         as a fall to zero and nowhere else."
    );
}
