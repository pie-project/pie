//! Every kernel source the tree holds is a source the build compiles.
//!
//! # The failure this exists for
//!
//! A `.cu` holds **both** a host launcher and the `__global__` it launches.
//! Migrating a kernel to the JIT path moves the `__global__` into a `.cuh` as
//! a template and replaces the launcher with a `LaunchRule`, and only then is
//! the `.cu` safe to delete — so "delete the `.cu`" is the LAST step of a
//! migration and never a mechanical one.
//!
//! Two ways to get that wrong, and they fail very differently:
//!
//! * **Delete the file, keep the CMake entry.** cmake stops with "cannot find
//!   source file": loud, immediate, needs no test.
//! * **Keep the file, drop the CMake entry.** The `.cu` stops being compiled
//!   and nothing complains — the archive simply no longer holds those kernels,
//!   and the first symptom is a link error somewhere unrelated. **That** is
//!   what this file catches.
//!
//! And a third, worse than either: a `__global__` copied into a `.cuh` while
//! the `.cu` keeps its own. Both compile, the archive gets one and the JIT
//! gets the other, and they drift — with every test passing on whichever half
//! it happens to exercise.
//!
//! The `.cuh` files live in `kernels-cuda-new/csrc/src` now — the JIT crate
//! owns the text it compiles, and this crate keeps the `.cu` translation
//! units its CMake builds — so that third check reads both trees. A test that
//! stopped at this crate's boundary would scan no headers at all and pass by
//! finding nothing.
//!
//! A filesystem walk and a text scan, so it runs anywhere: no GPU, no toolkit,
//! and no build of the archive it describes.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

fn csrc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc")
}

/// Every file under `csrc/src` with `ext`, relative to `csrc`.
///
/// `third_party` is excluded: those are vendored CUTLASS/FlashInfer
/// instantiations with their own lists, and the island stays the island
/// (`new-horizon.md` §5).
fn sources_with(ext: &str) -> Vec<String> {
    let root = csrc();
    let mut out = Vec::new();
    walk_ext(&root.join("src"), &root, ext, &mut out);
    out
}

/// Every `.cuh` in the JIT crate's tree, as a path relative to `csrc`.
///
/// The device headers are not under `csrc/` any more: fifty-seven `.cuh`
/// files moved to `kernels-cuda-new/csrc/src`, which compiles them through
/// NVRTC and now owns the text it compiles, while this crate keeps the `.cu`
/// translation units and the `.hpp` host interfaces and reaches the headers
/// with an `-iquote` its CMake spells.
///
/// The walk follows them, because the check that matters here spans BOTH
/// trees: a `__global__` in a `.cu` and the same one in a `.cuh` is the
/// half-finished migration this file exists to catch, and after the move
/// those two files are in two crates. A walk that stopped at this crate's
/// boundary would find no `.cuh` at all and pass by finding nothing — the
/// quietest way for a test to stop testing.
fn device_headers() -> Vec<String> {
    let root = csrc();
    let mut out = Vec::new();
    walk_ext(
        &root.join("../../kernels-cuda-new/csrc/src"),
        &root,
        "cuh",
        &mut out,
    );
    assert!(
        out.len() > 40,
        "the walk found almost no device headers: {out:?}"
    );
    out
}

fn walk_ext(dir: &Path, root: &Path, ext: &str, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| Some(e.ok()?.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "third_party") {
                continue;
            }
            walk_ext(&path, root, ext, out);
        } else if path.extension().is_some_and(|e| e == ext) {
            out.push(
                path.strip_prefix(root)
                    .expect("under csrc")
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
}

/// Every `src/**.cu` in the archive, as a path relative to `csrc`.
///
/// # The floor is `>= 1`, and it was `> 40`
///
/// `> 40` was measured when 71 `.cu` files were on disk, and it is the shape
/// `a_split_file_uses_the_header_it_was_split_into` already corrected in its
/// own guard: *"A guard against a vacuous pass must not be satisfiable only
/// while the thing it guards is broken."* This population SHRINKS BY
/// DELETION as the migration finishes — 71, then 40, then 16, and 15 with
/// `attn/kv_paged.cu` gone — so a fixed floor asserts that the work is
/// incomplete and goes red on its own success. It had already: the tree
/// passed under 40 several passes before this one.
///
/// What the guard is actually for is one failure and it is binary. `walk_ext`
/// takes a path, and a path that stops resolving yields an empty vector and
/// makes every caller below pass by iterating nothing — the §21.2 failure
/// this file names twice. `>= 1` catches exactly that and nothing else,
/// which is all the walk can honestly claim.
fn cu_on_disk() -> Vec<String> {
    let out = sources_with("cu");
    assert!(
        !out.is_empty(),
        "the walk under `csrc/src` found no `.cu` at all, so every check \
         below iterates an empty set and passes without looking"
    );
    out
}

/// Every `src/**.cu` path the CMakeLists names, however it names it.
///
/// A substring scan rather than a parse: the file has conditionals, `set()`
/// indirections and several `add_library` calls, and what is asked is only
/// whether the path APPEARS. A path in a branch not taken still counts — the
/// question is "did someone forget it entirely", not "is it compiled in this
/// configuration", and answering the second would mean evaluating cmake.
fn cu_in_cmake() -> HashSet<String> {
    let text = std::fs::read_to_string(csrc().join("CMakeLists.txt")).expect("read CMakeLists");
    text.split_whitespace()
        .map(|t| t.trim_matches(|c: char| c == '"' || c == ')' || c == '(').to_string())
        .filter(|t| t.starts_with("src/") && t.ends_with(".cu"))
        .chain(interpolated(&text))
        .collect()
}

/// Paths the CMakeLists builds rather than spells.
///
/// `attention_flashinfer_hd${_pie_hd}.cu` is one path per head dim, chosen by
/// a loop over `PIE_ATTN_HEAD_DIM` — a literal scan sees a name with a `${}`
/// in it and no file. Rather than evaluate cmake, the prefix before the first
/// `${` is taken and any file that starts with it counts as named.
///
/// This is the one place the scan is approximate, and it is approximate in
/// the safe direction: it can only ever call a file NAMED, never missing.
fn interpolated(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split_whitespace()
        .map(|t| t.trim_matches(|c: char| c == '"' || c == ')' || c == '('))
        .filter(|t| t.starts_with("src/") && t.contains("${") && t.ends_with(".cu"))
        .flat_map(|pattern| {
            let prefix = pattern.split("${").next().unwrap_or("").to_string();
            sources_with("cu")
                .into_iter()
                .filter(move |p| !prefix.is_empty() && p.starts_with(&prefix))
        })
}

/// A `.cu` on disk that the build does not compile is a kernel that silently
/// is not there.
#[test]
fn every_cu_on_disk_is_in_the_build() {
    let listed = cu_in_cmake();
    let missing: Vec<String> = cu_on_disk()
        .into_iter()
        .filter(|p| !listed.contains(p))
        .collect();
    assert!(
        missing.is_empty(),
        "these .cu files exist and nothing compiles them, so the kernels in \
         them are absent from the archive with no diagnostic:\n  {}",
        missing.join("\n  ")
    );
}

/// And the other direction, so the pair is total.
#[test]
fn every_cu_in_the_build_is_on_disk() {
    let on_disk: HashSet<String> = cu_on_disk().into_iter().collect();
    let dangling: Vec<String> = cu_in_cmake()
        .into_iter()
        // A stub is chosen by a `set()` on a feature; both spellings appear
        // in the text and only one has a file in any configuration.
        // A stub is chosen by a `set()` on a feature; both spellings appear
        // and only one has a file in any configuration. A `${}` is a pattern
        // rather than a path, and `interpolated` has already expanded it into
        // the files it names.
        .filter(|p| !on_disk.contains(p) && !p.contains("_stub") && !p.contains("${"))
        .collect();
    assert!(
        dangling.is_empty(),
        "the build names these and they are not there:\n  {}",
        dangling.join("\n  ")
    );
}

/// A `__global__` may be defined once.
///
/// Two definitions of one name is a half-finished migration: the archive gets
/// one and the JIT gets the other. Templates and plain kernels are compared by
/// NAME, which is what a `nvrtcAddNameExpression` and a `<<<>>>` would both
/// resolve.
///
/// Both trees, because the `.cuh` files live in `kernels-cuda-new/csrc/src`
/// now and the `.cu` files are still here. The move is exactly the operation
/// that could have created the copy this refuses, so the scan crosses the
/// crate boundary the headers crossed.
#[test]
fn no_global_is_defined_twice() {
    let root = csrc();
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut clashes: Vec<String> = Vec::new();
    for rel in cu_on_disk().into_iter().chain(device_headers()) {
        let Ok(text) = std::fs::read_to_string(root.join(&rel)) else {
            continue;
        };
        // QUALIFIED, because a bare name is not an identity. `k_matmul` is a
        // `kernels::ptir` template and an anonymous-namespace helper in
        // `model`, and those are two kernels that share a spelling — which is
        // exactly what a namespace is for. Comparing bare names would report
        // them and teach a reader to ignore this test.
        let mut ns = String::new();
        for line in text.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("namespace ")
                && let Some(named) = rest.split(&[' ', '{'][..]).next()
                && !named.is_empty()
            {
                ns = named.to_string();
            }
            let Some(after) = line.trim_start().strip_prefix("__global__ void ") else {
                continue;
            };
            let leaf: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if leaf.is_empty() {
                continue;
            }
            let name = format!("{ns}::{leaf}");
            if let Some((_, first)) = seen.iter().find(|(n, _)| *n == name) {
                clashes.push(format!("{name}: {first} and {rel}"));
            } else {
                seen.push((name, rel.clone()));
            }
        }
    }
    assert!(
        clashes.is_empty(),
        "a `__global__` is defined in two places, so a migration was left \
         half-done and the two copies can drift:\n  {}",
        clashes.join("\n  ")
    );
}

/// A `.cu` that has a `.cuh` USES it, rather than keeping its own copy.
///
/// # Why the name test above cannot see this
///
/// [`no_global_is_defined_twice`] compares qualified names, which is right for
/// what it is named for — one spelling that two texts both define. But **a
/// split RENAMES**: `write_kv_kernel` in the archive becomes `write_kv` in the
/// header, in a different namespace, because the `_kernel` suffix was there to
/// distinguish the kernel from its host launcher and the header has no host
/// launcher to distinguish it from. So the two copies never share a name, and
/// a name test is structurally unable to report them however carefully it is
/// written.
///
/// It happened. `attn/kv_paged.cu` kept all fourteen of its `__global__`s
/// while `attn/kv_paged.cuh` carried fourteen more with identical bodies —
/// `dequant_fp8_pages_active_kernel` and `dequant_fp8_pages_active` differ
/// only in `__nv_bfloat16` versus `device::bf16`. Every gate was green. The
/// header itself opens by claiming the arrangement this test now enforces:
/// *"`kv_paged.cu` includes this and keeps every `<<<>>>`, so the
/// ahead-of-time build and NVRTC compile ONE text."* It did not.
///
/// # The rule, which needs no names
///
/// If `x.cu` defines a `__global__` and `x.cuh` exists in the other tree, then
/// `x.cu` must include it. Nothing else is consistent: the header was created
/// FROM that file, so either the file uses it or there are two texts. A file
/// with no counterpart header is untouched and says nothing; a file that
/// includes its header may still define kernels of its own, which is the
/// non-template single-includer case §21.6 records and is not a copy.
///
/// The failure this refuses is the one the header names in its own words:
/// *"two copies that agree today are two kernels that drift, each right for
/// whichever half of the tree its tests exercise. `norm/altup_aux` shipped
/// exactly that for a release with every test green."*
#[test]
fn a_split_file_uses_the_header_it_was_split_into() {
    let root = csrc();
    let mut orphaned: Vec<String> = Vec::new();
    let mut paired = 0usize;
    for rel in cu_on_disk() {
        let Ok(text) = std::fs::read_to_string(root.join(&rel)) else { continue };
        let stem = rel.strip_suffix(".cu").unwrap_or(&rel);
        // `cu_on_disk` yields paths relative to `csrc`, so they LEAD WITH
        // `src/` while the header tree's own root is its `src`. Not stripping
        // it built a path that never exists, so every file took the
        // `continue` below and the test passed over an empty set — which is
        // what the first version of this test did, in the same hour as the
        // §21.2 and §21.7 notes about gates that pass without looking. The
        // assertion at the end is what makes that unrepeatable.
        let stem = stem.strip_prefix("src/").unwrap_or(stem);
        let header = root.join("../../kernels-cuda-new/csrc/src").join(format!("{stem}.cuh"));
        if !header.exists() {
            continue;
        }
        paired += 1;
        // THE `__global__` FILTER SITS HERE, BELOW THE WITNESS, AND THE ORDER
        // IS THE WHOLE POINT. The rule needs it -- a `.cu` that holds no
        // device text has nothing to duplicate, so it is out of scope and a
        // file that merely calls launchers must not be told to include a
        // header it has no use for. But `paired` is not the rule, it is the
        // proof that the rule LOOKED, and counting it above this line made it
        // a count of files that still hold device text AND have a header --
        // a population that empties precisely when the migration succeeds.
        // `attn/kv_paged.cu` was the only member. The moment its split was
        // completed correctly -- fourteen `__global__`s deleted, the header
        // included -- the count went to zero and the floor fired, so the test
        // demanded a transformation and then failed on it being done. The
        // witness has to be the thing that cannot vanish for a good reason:
        // that a `.cu`/`.cuh` counterpart pair was FOUND AT ALL, which is
        // exactly what the join produces and what the §21.2 note was about.
        if !text.contains("__global__ void ") {
            continue;
        }
        // The spelling the tree uses is the path under `src`, which is what
        // `stem` already is -- `attn/attn_res.cuh`. Asked as a substring
        // because a file may reach it through a relative spelling.
        if !text.contains(&format!("{stem}.cuh")) {
            let mine = text.matches("__global__ void ").count();
            let theirs = std::fs::read_to_string(&header)
                .map(|t| t.matches("__global__ void ").count())
                .unwrap_or(0);
            orphaned.push(format!(
                "{rel} defines {mine} `__global__` and does not include \
                 {stem}.cuh, which carries {theirs}"
            ));
        }
    }
    // THE SET WAS NOT EMPTY. Every check above is inside two `continue`s, so a
    // path that stopped resolving would empty the loop and leave the assertion
    // below trivially true — which is exactly what the first version of this
    // test did, silently, in the same hour as the §21.2 and §21.7 notes about
    // gates that pass without looking. `cu_on_disk` yields `src/…` and the
    // header tree's root is its own `src`, so the join produced a path that
    // never existed and every file took the `continue`.
    //
    // COUNTED OVER PAIRS, NOT OVER UNFINISHED WORK. `paired` is every `.cu`
    // with a counterpart `.cuh`, whether or not it still defines a kernel, so
    // it rises as files are split and never falls as they are finished. The
    // first version counted only files that still held device text, which made
    // the floor an assertion that the migration was INCOMPLETE: `kv_paged` was
    // the sole pair under that spelling, and completing it correctly took the
    // count to zero and turned this test red on its own success. A guard
    // against a vacuous pass must not be satisfiable only while the thing it
    // guards is broken.
    //
    // FORTY-EIGHT, and none of them holds a kernel. Measured after `kv_paged`
    // landed: 48 of the 71 `.cu` files on disk have a counterpart header and
    // every one of them includes it, and the ones that still define a
    // `__global__` — `attn/attention_flashinfer`, `attention_naive_paged`,
    // `attention_xqa`, `mla_paged`, `pack_dense_mask` and `qkv_fused` — have
    // no header at all, because they have not been split. So the orphan list
    // below is empty for the best reason available and this test is now a
    // REGRESSION guard: it fires when device text reappears in a `.cu` that
    // has a header, or when a new split lands without the include. The floor
    // is what keeps it from being empty for the worst reason instead, and 48
    // is a number that only grows.
    //
    // ZERO NOW, AND THE LIST OF SIX IS EMPTY FOR THE BEST REASON THERE IS.
    // Re-walked after `attn/attention_xqa.cu`'s `build_xqa_metadata_kernel`
    // left for `kernels-cuda-new/csrc/src/attn/attention_xqa.cuh`: NO `.cu`
    // under `csrc/src` contains `__global__ void ` any longer, so every file
    // takes the `continue` above and `orphaned` is empty because there is
    // nothing left to orphan. Four of the six the paragraph above names --
    // `attention_naive_paged`, `mla_paged`, `pack_dense_mask`, `qkv_fused` --
    // are still on disk and were split by other passes; `attention_flashinfer`
    // moved out to `driver-cuda/csrc/attn/`, which this walk does not see; and
    // `attention_xqa` is this pass's. The 71 and the 48 are both older than
    // the tree: 40 `.cu` remain and 25 of them are paired. That is why the
    // floor below is `>= 1` and not a number -- the population it guards
    // against a vacuous pass shrinks by deletion as well as growing by
    // splitting. This test is now purely a regression guard against device
    // text REAPPEARING in a file that has a header.
    //
    // That list was SEVEN and 17 kernels; `gemm/gemv` was the seventh and its
    // two `__global__` templates are the two that left, so it is six files and
    // 15 kernels now. `gemm/gemv.cu` is deleted outright rather than split —
    // the kernels are `kernels-cuda-new/csrc/src/gemm/gemv.cuh` and the host
    // launcher is `driver-cuda/src/fire/gemv.rs` — so it never becomes a
    // `.cu`/`.cuh` pair and never enters `paired`. The 71 and the 48 are the
    // earlier measurement and were NOT re-taken here; only the sentence naming
    // the deleted file was corrected, because a count this change did not
    // measure is not a count it should restate.
    assert!(
        paired >= 1,
        "no `.cu` file was paired with a header, so the pairing itself is \
         broken and the check below tested nothing"
    );
    assert!(
        orphaned.is_empty(),
        "a header was split out of these files and then not used, so the \
         archive and the JIT compile two texts that no name test can \
         compare:\n  {}",
        orphaned.join("\n  ")
    );
}

/// A split moves device text out; it does not change what the host launches.
///
/// # What this catches, and what it deliberately does not
///
/// The migration split fifty-five `.cu` files: every `__global__` moved into a
/// `.cuh` as a template, and the `.cu` kept its host launchers. That is a
/// mechanical-looking edit over thousands of lines, and the way it goes wrong
/// is not a compile error — it is a launcher that quietly stops being called,
/// or a `<<<>>>` that acquires a different grid on the way across.
///
/// So this counts the launches. Not what they launch, not with what geometry —
/// **how many `<<<>>>` each file executes**. A split that dropped a launcher
/// loses one; a split that duplicated a kernel body and launched both gains
/// one. Both are silent at build time and both are caught here.
///
/// It found exactly one, and the one was correct: `quant/dequant_fp8.cu` went
/// from three launches to four because `dequant_fp8_e4m3_to_bf16_per_group`
/// used to DELEGATE to the `blocked` launcher, passing `group_size` as both
/// block dimensions, and now calls its own template directly. Same
/// `dequant_fp8_e4m3_tile`, same arithmetic, one fewer indirection — checked
/// by reading both. Every other file preserved its count exactly.
///
/// # Why a floor and not a golden number
///
/// The count is compared against a table written here rather than against git,
/// because a test that reads the repository's history is a test that cannot run
/// on a checkout. The table is a floor: a file may gain a launcher when a
/// kernel that had none is given one, and that is a deliberate act someone
/// records here. Losing one never is.
#[test]
fn no_file_lost_a_launch_in_the_split() {
    // Comment lines are excluded: a header that explains what the `<<<>>>`
    // became mentions it, and every migrated file's header does.
    fn launches(text: &str) -> usize {
        text.lines()
            .filter(|line| {
                let t = line.trim_start();
                line.contains("<<<") && !t.starts_with("//") && !t.starts_with('*')
            })
            .count()
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src");
    let mut total = 0usize;
    let mut files = 0usize;
    for path in walk(&root) {
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("a source file reads");
        let n = launches(&text);
        total += n;
        files += 1;
        // A file that HAD launches must still have them. A file that never
        // had any is not evidence of anything: `attn/attention_flashinfer_*`
        // hand their work to FlashInfer, which does its own launching, and a
        // completed migration will eventually leave `.cu` files with none at
        // all. Only a DROP is a defect, and the total below is what sees it.
        let _ = &text;
    }

    // 217.
    //
    // This number was 401 for the whole of the migration, and 401 was the
    // same number the tree carried BEFORE the migration -- a coincidence
    // that read as reassurance for months. It was not one. It meant every
    // launcher a routed row stopped calling was still here, still compiled,
    // still linked, and called by nothing.
    //
    // `new-horizon.md` §41 built the transitive audit that says so and §43
    // acted on it. The list below is no longer a list of DELTAS, because a
    // delta from 401 is a diff against a tree nobody would recognise; it is
    // a list of what a drop from here would MEAN.
    //
    // Where the 184 that left went:
    //
    //   §42 the three multimodal towers moved OUT of this archive to
    //       `crates/driver-cuda/csrc/vision/`. Those launches did not stop
    //       happening; they stopped happening HERE, and this walk only sees
    //       `csrc/src`.
    //       Two launchers under `vision/` did NOT move, because they were
    //       deleted: `scatter_gemma4_vision` (1 launch) and
    //       `scatter_gemma4_audio` (0), neither reachable from any root in
    //       the audit and neither carrying a `pie_k_` shim entry, so no Rust
    //       could name them either. Separately the Gemma-4 vision tower's
    //       six `norm::rmsnorm_bf16` CALLS became two LAUNCHES before it
    //       left, which is why `norm/rmsnorm.cu` is now that launcher's only
    //       C++ consumer; the two launches are counted at the new path, not
    //       here.
    //   §43 deleted the unreachable surface in `comm/`, `sample/`,
    //       `layout/`, `mlp/`, `moe/`, `quant/`, `norm/` and `ssm/` -- 63
    //       host launchers, five whole `.cu`/`.hpp` pairs and one whole
    //       `__global__`, none of which any root could reach. A routed row
    //       fires its kernel through NVRTC out of a `.cuh`; the ahead-of-
    //       time launcher it left behind is what §10.10 calls a consumer
    //       set of size zero, and this count is where such a launcher was
    //       visible as a number.
    //   §45 deleted `gemm/gemv.cu`'s unreachable surface -- `gemv3_bf16`
    //       and the three `_tuned` sweep entry points, five launches over
    //       the `gemv3_bf16_kernel` that went with them. NO ROW LOST A
    //       LAUNCHER: `gemv_bf16` was the file's whole live surface and kept
    //       all four of its launches. The sweeps' harness,
    //       `driver/cuda/bench/gemv_bench.cu`, is in no source directory of
    //       this repository. `gemm/gemm.cpp` contributes nothing here and
    //       never did -- it holds 0 `<<<>>>` -- so the ~380 lines §45 moved
    //       out of it into Rust are invisible to this count, which is the
    //       point the section makes.
    //       THE FILE IS NOW GONE ENTIRELY, and those last four launches with
    //       it: the two `__global__` templates are NVRTC's
    //       (`kernels-cuda-new/csrc/src/gemm/gemv.cuh`, four rows in that
    //       crate's `families::gemm`) and `gemv_bf16` is Rust
    //       (`driver-cuda/src/fire/gemv.rs`). See the note at `EXPECTED`.
    //   §44 deleted 20 host launchers under `attn/`, taking that family's
    //       subtotal from 71 to 52 across 52 -> 46 files, in three kinds it
    //       keeps separate on purpose: JIT-routed residue (a symbol in
    //       `JIT_DISPATCHED` gets no shim entry, so its consumer set is
    //       empty BY CONSTRUCTION and not by search), closed cycles of dead
    //       callers, and two collected `Backlog` keepers. NO ROW LOST A
    //       LAUNCHER. One SPLIT is in that number and is exactly the failure
    //       described below: `attn_score_fold_heads`'s device text moved to
    //       `kernels-cuda-new/csrc/src/attn/attention_flashinfer.cuh` and its
    //       host launcher stayed, so the launch is still counted here while
    //       the driver fires the JIT unit instead. It stays until the row
    //       does.
    //   the concurrent `vision/` work, measured the same way and recorded in
    //       its own section.
    //
    // What a DROP still means, and why this test is still worth having: a
    // launcher that goes missing in a SPLIT -- device text moved to a
    // `.cuh`, host launcher deleted with it, row still pointing at the shim
    // entry -- compiles, links, and silently stops launching. That is the
    // failure this number was built to catch, and deleting an UNREACHABLE
    // launcher looks identical to it from here. The difference is not in
    // this file: it is in `scripts/csrc-reachability-audit.py`, which
    // answers "can any root reach it" and is the check that has to be run
    // before this number is moved down.
    //
    // So: re-derive this from the tree when you have run that audit and can
    // say which roots stopped reaching what. Never compute a delta -- three
    // agents were deleting from this tree at once the night it went from
    // 401 to 210, and a delta from any one of them would have been wrong.
    //
    // It read 210 and now reads 205, and the five are a RESTORATION, which is
    // the one direction this number was not built to explain. An upstream
    // merge landed on that night's uncommitted work and took most of it; the
    // vision towers came back from the object store, `ssm/gated_delta_net.cu`
    // and `rope/rope.hpp` came back from verified blobs, and
    // `attn/attention_xqa.cu` was restored to HEAD because the merge had left
    // it half-deleted -- its `__global__` gone and its launcher still calling
    // it, which is not a deletion but a file that does not compile. Those
    // five launches are its. They are dead by the reachability audit and a
    // later commit should take the file whole, with its CMake entry.
    //
    // The per-directory shape at 205, because a scalar cannot localise a
    // disagreement and a table can: attn 56, ssm 35, moe 27, layout 23,
    // quant 20, norm 18, rope 12, mlp 6, gemm 4, sample 4, comm 0, vision 0.
    //
    // 205 -> 201, and the four are `attn/`'s §44.4 Kind A, re-landed after
    // the same merge: `softcap`, `attn_sink` and `attn_res` went as whole
    // `.cu`/`.hpp` pairs. Their CMake entries had survived the merge while
    // the files had not, so `every_cu_on_disk_is_in_the_build` was already
    // red on exactly these three. All three rows are LIVE and have live
    // `crates/model/src` callers; all three are in `device::JIT_DISPATCHED`,
    // so `abi::emit_c_shim` skips them and the generated shim holds zero
    // entries for `logit_softcap_bf16`, `attention_sink_rescale_bf16` and
    // `attn_res_blend_bf16` -- the consumer set is empty by construction
    // rather than by search. The surviving text is
    // `kernels-cuda-new/csrc/src/attn/{softcap,attn_res,attn_sink}.cuh`,
    // which `families/attn.rs` `include_str!`s and NVRTC compiles.
    //
    // 201 -> 167, measured 2025-06 against the worktree and not against this
    // constant, and it is the REST of §43: the six families the merge took
    // and nobody had put back. Thirty-two of the thirty-four are that, and
    // they are re-landed here; the last two are a CONCURRENT §45 deletion in
    // `sample/` and are counted, not claimed.
    //
    //   layout  23 -> 15   `geometry.cu` (2), `gather_rows.cu` (4),
    //                      `graph_pad.cu` (1) and `split_gate_up.cu` (1)
    //                      went WHOLE, with their `.hpp`s and their four
    //                      CMake entries. `gather_bf16_rows`,
    //                      `transpose_bf16_nld_to_lnd` and the two
    //                      `layout::` geometry helpers are the routed kind;
    //                      `split_gate_up_bf16`, `embed_scaled_concat_bf16`,
    //                      `launch_derive_kv_len`, `launch_resolve_slot_to_block`
    //                      and `launch_graph_pad_rows` never had a row at
    //                      all, so they never had a shim entry to lose.
    //   moe     27 -> 18   `topk_sigmoid.cu` (1) whole; `dsv4_routing.cu`
    //                      (2 -> 1) lost `topk_sqrtsoftplus_bf16`;
    //                      `moe_dispatch.cu` (16 -> 9) lost four launchers
    //                      whose seven launches were forks over a block
    //                      width. `moe_aligned_block` and
    //                      `flashinfer_cutlass_moe_*` STAY -- Rust cites the
    //                      first as an authority and fires the second
    //                      unconditionally under a DIFFERENT name
    //                      (`dsl::cuda::moe_fused_cutlass`).
    //   quant   20 -> 15   `transcode.cu` (1) whole; `dequant_fp4.cu`
    //                      (3 -> 1) and `dequant_wna16.cu` (4 -> 2) each
    //                      lost their two routed MoE decode GEMVs. The
    //                      `dequant_*`, `bf16_to_fp16`, `cast_fp32_to_bf16`
    //                      and `scale_rows_bf16` launchers in those files
    //                      STAY: `gemm/gemm.cpp` calls the first group and
    //                      hand-written `ffi::pie_k_quant_*` fires in
    //                      `driver-cuda/src/fire/lora.rs` call the second.
    //                      (HALF OF THAT REASON IS VOID: `gemm/gemm.cpp` is
    //                      DELETED. The four `quant::dequant_*` rows it held
    //                      are in `device::JIT_DISPATCHED` and the arms that
    //                      called them are `bind::quant_gemm`. The `lora.rs`
    //                      half stands. Re-measure the first group before
    //                      relying on this line.)
    //   norm    18 -> 13   `altup.cu` (2) whole; `dsv4_hc.cu` (7 -> 4) lost
    //                      three. `rmsnorm.cu` is untouched -- all four of
    //                      its launchers have a C++ caller.
    //   mlp      6 -> 1    `gaussian_topk.cu` (1) whole; `swiglu.cu`
    //                      (5 -> 1) lost `chunked_swiglu_bf16`, whose four
    //                      launches were one launcher's fork over an
    //                      activation. `sigmoid_gate_inplace_bf16` stays --
    //                      it is still a shim root.
    //   sample   4 -> 2    NOT §43's and not this pass's. §43 recorded
    //                      `argmax_accumulate_bf16` and
    //                      `argmax_finalize_bf16` as KEPT, held by
    //                      `gemm/gemm.cpp`'s `lm_head_argmax_chunked`. A
    //                      concurrent §45 found that holder had no caller of
    //                      its own, deleted it, and the two followed. The
    //                      hold was real when §43 measured it; what changed
    //                      is the holder, not the reading.
    //   comm     0 -> 0    §43's `comm/` work was already at HEAD. The
    //                      `CustomAllReduce` LIFECYCLE stays and the audit
    //                      is wrong about it (§43.4a); what went was the one
    //                      `__global__` and the `_exact` method, and neither
    //                      was a `<<<>>>` in this directory to begin with.
    //
    // The per-directory shape at 166: attn 52, ssm 35, moe 18, layout 15,
    // quant 15, norm 12, rope 12, gemm 4, sample 2, mlp 1, comm 0,
    // vision 0 -- across 41 files, down from 65.
    //
    // `norm` is 13 → 12 and the one that left is worth a line, because it is
    // the shape the whole countdown takes. `norm/add_bias.cu` held a single
    // `<<<>>>` behind `add_bias_bf16`, whose last C++ caller was an
    // `#include` in `gemm.cpp` that no longer resolved to a call. Nothing
    // was ported to remove it: the device text had been in
    // `kernels-cuda-new/csrc/src/norm/add_bias.cuh` for weeks, and the row
    // needed only to be named in `device::JIT_DISPATCHED` for the shim to
    // stop emitting a forwarder. The file then had no consumer at all.
    //
    // That is the countdown's real shape -- a row stops calling C++ when
    // three facts that live in three different files are true at once
    // (a unit hosts it, every operand states a `Source`, no C++ TU calls its
    // launcher), and nothing in the tree joined them. Computed over the 126
    // rows still holding a shim entry: 55 had a unit, 25 of those sourced
    // every operand, 5 of those were still called from `gemm.cpp` or
    // `rmsnorm.cu`, and the remaining 18 could have been routed at any point
    // since their unit landed.
    //
    // Every one of the 32 was re-verified against the tree before it was
    // taken, not carried over from §43's table, because the tree moved
    // underneath that table: `JIT_DISPATCHED` went from 7 entries to 69,
    // which EMPTIES a consumer set that a search would still have found
    // full. The filter each had to pass is §43.9's: no shim entry, no C++
    // caller in any `.cu`/`.cuh`/`.cpp`/`.hpp`, no hand-written
    // `ffi::pie_k_*` fire under `crates/driver-cuda/src`, and its row -- if
    // it has one -- in `device::JIT_DISPATCHED`. NO ROW LOST A LAUNCHER and
    // no `__global__` was deleted: every one of the eight whole files was
    // already a pure host shell that `include`d its `.cuh` from
    // `kernels-cuda-new/csrc/src`, and all eight of those headers are still
    // there, which is what makes each of these re-addable.
    //
    // NOT taken, against the note above: `attn/attention_xqa.cu` must not go
    // whole. Its five launches read dead to the audit, and one of them is
    // `prepare_attention_xqa_decode_bf16`, which the audit cannot see is
    // held: the live row `attn::attention_xqa_decode_bf16_prepared`
    // (`kernels-cuda-new/src/table/attn.rs:208`) states
    // `needs = Prepare::FireWide`, that row is AOT -- the generated shim has
    // exactly one XQA entry and it is this one -- and this function is the
    // only text in the tree that writes the page table and sequence lengths
    // into the workspace at the offsets the prepared launcher reads back.
    // A `Prepare` is an obligation stated in the TABLE and discharged by the
    // driver, so no call edge exists for the audit to find. Deleting the
    // file would not remove a dead launcher; it would leave a live row
    // stating a prepare that no code implements, discovered at run time.
    // `new-horizon.md` §44.5 is the long form.
    //
    // THE CLAUSE THAT KEPT IT IS DISCHARGED, AND THE FILE STILL STAYS -- for
    // a different reason, which is why this paragraph is corrected rather
    // than deleted. `prepare_attention_xqa_decode_bf16` is
    // `driver-cuda/src/fire/xqa.rs::prepare_decode` now: same offsets, same
    // `page_bucket` rounding, same `<<<num_requests, 128>>>`, fired through
    // `KernelModule::fire` against `attn/attention_xqa.cuh`. So "the only
    // text in the tree that writes the page table" is no longer this file,
    // and the §44.5 argument moved with the implementation. What holds the
    // `.cu` now is `attention_xqa_decode_bf16_prepared` itself: it ends in
    // `launchMHAFlashInfer_xqa_gqa5_bf16_p32_h128`, an upstream FlashInfer
    // HOST function pulled into this translation unit by
    // `#include <xqa/mha.cu>` under a renamed symbol. There is no device text
    // of ours left in it, so §48's split is degenerate here -- it becomes
    // Rust entire or it does not move -- and §50.9 is where that is owed.
    // 166 -> 162, and NEITHER end of that is a delta: both were derived by
    // walking `csrc/src` and counting, because a delta against a tree two
    // other agents are editing is arithmetic on a moving object.
    //
    // The four are `attn/attention_flashinfer.cu`'s. That file was moved to
    // `crates/driver-cuda/csrc/attn/` by an earlier pass which updated neither
    // this constant nor the `add_library` entry naming it -- so this test and
    // `every_cu_in_the_build_is_on_disk` were both red on the same omission
    // before this change touched anything. `attn` is 52 -> 48; nothing else
    // moved.
    //
    // NOTHING ELSE MOVED IT, and that is worth stating because a lot of files
    // were examined for this pass. The six `#include <flashinfer/...>` files
    // -- `attn/attention_merge_states.cu`, `attn/attention_mla.cu`,
    // `attn/attention_flashinfer_hopper.cu`, `comm/custom_all_reduce.cu` and
    // the two stubs -- hold ZERO `<<<>>>` and zero `__global__` between them.
    // Every kernel they reach is a template instantiated inside an upstream
    // host function. They are host walks, so this census cannot see them at
    // all, and it will not move when they become Rust either.
    //
    // `ssm/flashinfer_mamba.{cu,hpp}` WAS deleted by this pass and also held
    // zero launches, for the same reason.
    //
    // Measured, 54 files:
    //
    //   attn 48   ssm 35   moe 18   layout 15   quant 15   rope 12
    //   norm 12   gemm  4   sample 2   mlp 1   comm 0   vision 0
    //
    // `comm` and `vision` are zero and are still directories: `vision`'s three
    // towers moved to `driver-cuda/csrc/vision/` in §42 and left their five
    // `.hpp` behind as the generated shim's contract, and `comm` holds one
    // `.hpp` and two host translation units with no device text at all.
    //
    // A SECOND -4 WAS IN FLIGHT WHEN THE PARAGRAPH ABOVE WAS WRITTEN, AND IT
    // HAS LANDED. `gemm/gemv.{cu,hpp}` are deleted. The note that stood here
    // stated the arithmetic and left the constant at 162 on purpose, so that
    // whoever landed second did the walk rather than the subtraction — this
    // is that pass, and it did:
    //
    //   * before, with this function's own rule, over `csrc/src` as it stood
    //     with `gemm/gemv.cu` still on disk: 162 across 54 files, gemm 4.
    //   * after: 158 across 53 files, gemm ABSENT — `gemm/` now holds only
    //     `gemm.cpp` and `gemm.hpp`, which have 0 `<<<>>>` between them and
    //     always did, so the directory stops appearing in a census of `.cu`.
    //
    // Both ends are measurements. That they differ by exactly the four
    // launches `gemv.cu` held is a CHECK on the walk, not the method that
    // produced the number.
    //
    // NO ROW LOST A LAUNCHER, in the sense this test exists to catch. The
    // four launches did not stop happening; they stopped being spelled in
    // C++. The two `__global__` templates are
    // `kernels-cuda-new/csrc/src/gemm/gemv.cuh`, where NVRTC compiles them,
    // and that crate's `families::gemm` holds one row per instantiation. The
    // host launcher `gemv_bf16` — the `cudaDevAttrComputeCapabilityMajor`
    // read, the `N <= 4096` split-K threshold, the `K % 8` and 16-byte
    // refusal, and `ceil(N / 4)` with its overflow refusal — is
    // `driver-cuda/src/fire/gemv.rs`, which builds a `Launch` by hand
    // (`block: [32, kWarps, 1]`; no `LaunchRule` states a 2-D block and none
    // was added) and fires it through `KernelModule::fire`. This census reads
    // `.cu` under `csrc/src` and can see none of that, which is what the
    // countdown looks like when it is working.
    //
    // The two calls that pass named — `read_src("gemm/gemv.cu")` in
    // `the_audited_launchers_read_no_environment_variable` and its two
    // `pinned` citations — are gone with the file, and each says at its site
    // where the thing it pinned now lives.
    // A THIRD PASS, and this one is `moe/` and `ssm/`'s. Two whole
    // `.cu`/`.hpp` pairs left `moe/`; the walk was redone from scratch
    // rather than subtracted, for the reason the paragraph above gives, and
    // for a second reason it did not have to face: five agents were editing
    // `csrc/src` at the same time, so a delta would have been a diff against
    // a tree that no longer existed by the time it was written.
    //
    // The walk, this function's own rule, per directory:
    //
    //   attn    48 across 25    layout   8 across  2    quant  11 across  4
    //   comm     0 across  1    moe     10 across  3    rope   12 across  1
    //   norm    12 across  3    sample   2 across  1    ssm    35 across  4
    //
    //                                          TOTAL  138 across 44
    //
    // `gemm/` is still absent and `mlp/`, `vision/` and `dist/` never had a
    // `.cu` with a `<<<>>>` here.
    //
    // What this pass removed from `moe/`, and neither is a drop:
    //
    //   * `moe/topk_softmax.{cu,hpp}` — 7 launches, ALL UNREACHABLE. Two of
    //     its three launchers are named by `device::JIT_DISPATCHED`
    //     (`topk_softmax_bf16`, `apply_per_expert_scale_bf16`), so
    //     `abi::emit_c_shim` emitted no entry for either, and the third
    //     (`topk_softmax_bf16_form`) has no `table::moe` row at all and so
    //     never had an entry to lose. The consumer sweep found the symbols
    //     in no `.cu`, `.cuh`, `.cpp` or `.hpp` outside the pair itself.
    //     Six of the seven launches were the WARP LADDER, which is a host
    //     decision and not a launcher: it is recorded on `families::moe`'s
    //     `topk_softmax` row and specified in `new-horizon.md` §52, and it
    //     has NOT been re-landed. That is a stated gap, not a silent one.
    //   * `moe/moe_grouped_gemm.{cu,hpp}` — 1 launch, MIGRATED.
    //     `execution::RUST_SERVED` names `moe::moe_grouped_gemm_bf16`, so
    //     the shim drops the entry and `emit_dispatch` calls
    //     `bind::service::moe_moe_grouped_gemm_bf16`, whose body is
    //     `driver-cuda/src/fire/moe.rs`. The `__global__` is unmoved in
    //     `kernels-cuda-new/csrc/src/moe/moe_grouped_gemm.cuh`, which is now
    //     also the `MOE_GROUPED_GEMM` NVRTC unit's root, and the launch
    //     `dim3(N / kNTile, max_blocks)` is a hand-built `Launch` with the
    //     deleted file's line numbers cited beside it. The support
    //     predicate — `M == kFrag && K <= 512 && N % kNTile == 0 &&
    //     K % kFrag == 0`, and the `down K=256 7.94 -> 5.91` /
    //     `gate_up K=2048 11.08 -> 11.98` measurement behind the 512 — is
    //     `fire::moe::supported`, which returns a `Decline` naming which
    //     conjunct refused. The launcher returned `void` and could not.
    //
    // `ssm/` is unchanged by this pass: all four of its files still hold
    // every launch they held, and `new-horizon.md` §52 says what remains and
    // why it was not attempted.
    // A FOURTH PASS: `norm/` and `rope/`. Three whole `.cu` files left, and
    // the walk below was redone from scratch for the reason every table above
    // gives, plus one this pass can date precisely — see the `quant` line.
    //
    //   * `rope/rope.cu` — 12 launches, ALL MIGRATED, and the directory now
    //     holds no `.cu` at all. Three of its twelve launchers are named by
    //     `device::JIT_DISPATCHED` (`rope_standard_table`,
    //     `qk_rmsnorm_rope_bf16`, `rope_partial_bf16`) and nine by
    //     `execution::RUST_SERVED`, so `abi::emit_c_shim` emits no entry for
    //     any of them. The host program is `driver-cuda/src/fire/rope.rs`;
    //     the `__global__`s are unmoved in
    //     `kernels-cuda-new/csrc/src/rope/rope.cuh`.
    //     `rope/rope.hpp` SURVIVES and says at its own top why: it is the
    //     subject of `driver-cuda/tests/launch_abi.rs`, the launch-ABI pilot,
    //     which reads it from disk by a hard-coded path. Nothing `#include`s
    //     it, so nothing can fail to link.
    //   * `norm/rmsnorm.cu` — 7 launches, ALL MIGRATED, and the one worth a
    //     line in this file because of what it FREED. Three fully-migrated
    //     rows were blocked on it and on nothing else: `norm::rmsnorm_bf16`
    //     (`rmsnorm.cu:42,59,63`), `norm::rmsnorm_strided_bf16` and
    //     `quant::bf16_to_fp16` (`:64`). Each had a unit, each sourced every
    //     operand, and each was still called by a C++ translation unit —
    //     §10.10, a launcher goes when its WHOLE consumer set has gone, and a
    //     file composing with its own siblings is C++ calling C++ that no
    //     Rust dispatch can intercept. `driver-cuda/src/fire/rmsnorm.rs`
    //     makes those calls Rust and all three are in
    //     `device::JIT_DISPATCHED` now. `quant::bf16_to_fp16`'s own launcher
    //     is in `quant/dequant_wna16.cu` and is that file's problem, not
    //     this one's.
    //   * `norm/dsv4_hc.cu` — 4 launches, ALL MIGRATED, all four launchers
    //     `execution::RUST_SERVED`, host program
    //     `driver-cuda/src/fire/dsv4_hc.rs`. Every one of the four AOT rows
    //     states operands no `Source` can bind, so `abi::emit_dispatch`
    //     skipped them whole and they were UNREACHABLE before this change and
    //     are reachable after it.
    //
    // `norm/residual_add.cu` STAYS, and it is the counter-example that makes
    // the other three legible: `gemm/gemm.cpp:1990` calls `residual_add_bf16`
    // from the INT8 `beta != 0` arm. One C++ caller is a full consumer set.
    // `norm/` therefore goes 12 -> 1 rather than 12 -> 0.
    //
    // THAT `STAYS` IS SUPERSEDED: the ninth pass, at `EXPECTED` below, moved
    // `gemm.cpp:1990`'s arm into `driver-cuda/src/bind/quant_gemm.rs` and
    // the call with it, so the full consumer set emptied and `norm/` went
    // 1 -> 0. The paragraph is kept because the READING is still right — one
    // C++ caller is a full consumer set — and because it is the clearest
    // record of what had to happen before the file could go.
    //
    // The walk, this function's own rule, per directory:
    //
    //   attn    48 across 25    layout   8 across  2    quant  10 across  4
    //   comm     0 across  1    moe     10 across  3    norm    1 across  1
    //   ssm     35 across  4
    //
    //                                          TOTAL  112 across 40
    //
    // THAT `attn 48` IS SUPERSEDED: the sixth pass, at `EXPECTED` below, took
    // `attn/attention_xqa.cu`'s one launch and with it the archive's last
    // `__global__`. Corrected there rather than here, because a table records
    // the walk that was taken when it was written and a later walk gets its
    // own.
    //
    // `gemm/`, `sample/` and `rope/` hold no `.cu` at all now, and `mlp/`,
    // `vision/` and `dist/` never had one with a `<<<>>>` here.
    //
    // 136 -> 112 IS 24 AND THIS PASS DELETED 23. The twenty-fourth is not
    // ours and the discrepancy is the point of walking rather than
    // subtracting: `quant/dequant_wna16.cu` lost one `<<<>>>` to a
    // CONCURRENT edit between this pass's first walk (which reproduced 136
    // across 43 exactly, so the constant was right when it started) and its
    // second. `quant` is 11 -> 10 for a reason that belongs to whoever made
    // it. Had this pass written `136 - 23 = 113` it would have left the
    // constant wrong by one and blamed the next reader's change for it.
    // A FIFTH PASS: `quant/`, `layout/` and `sample/`. One whole `.cu`/`.hpp`
    // pair left `sample/` (the directory with it) and three launchers left
    // files that stay. The walk below was redone from scratch, per this
    // function's own rule, and it is the reason the paragraph after it exists.
    //
    // The walk, per directory:
    //
    //   attn    48 across 25    layout   7 across  2    quant   9 across  4
    //   comm     0 across  1    moe     10 across  3    norm    1 across  1
    //   ssm     35 across  4
    //
    //                                          TOTAL  110 across 40
    //
    // `gemm/`, `rope/` and now `sample/` hold no `.cu` at all, and `mlp/`,
    // `vision/` and `dist/` never had one with a `<<<>>>` here.
    //
    // **112 DID NOT REPRODUCE, AND THE GAP IS NOT THIS PASS'S FIVE.** The
    // 110 above is a walk. The fourth pass's 112 is not reachable from it by
    // adding back what this pass removed — that lands on 115 across 41 — and
    // the 3 left over are in two pieces this pass MEASURED DIRECTLY rather
    // than inferred, each of which the table above 112 names by omission:
    //
    //   * `sample/argmax.cu` was ON DISK, holding 2 launches at `:123` and
    //     `:136`, when the fourth pass wrote *"`gemm/`, `sample/` and `rope/`
    //     hold no `.cu` at all now"*. It holds none now — this pass deleted
    //     the pair — but it did not then, and those 2 are 2 of the 3.
    //   * `quant` held 11 across 4 when this pass opened those files:
    //     `dequant_fp4.cu` 1, `dequant_fp8.cu` 3, `dequant_wna16.cu` 2,
    //     `quant_bf16_to_fp8.cu` 5. The fourth pass's table says 10. That is
    //     the third, and it is the same directory that pass had already
    //     flagged as moving under it.
    //
    // Neither is corrected by subtraction and neither is blamed: this pass
    // read the files. Whoever lands sixth should walk rather than diff, and
    // should expect to find something, because a census taken from a delta is
    // a census of a tree that no longer exists.
    //
    // What this pass removed, and none of the five is a drop:
    //
    //   * `sample/argmax.{cu,hpp}` — 2 launches, MIGRATED, and the directory
    //     went with the pair. `execution::WALKED` classifies
    //     `sample::lm_head_gemv_argmax_int8` and `execution::RUST_SERVED`
    //     names it, so `abi::emit_c_shim` drops the entry and
    //     `emit_dispatch` calls `bind::service::sample_lm_head_gemv_argmax_
    //     int8`, whose body is `driver-cuda/src/fire/lm_head_argmax.rs`. The
    //     two `__global__`s are unmoved in
    //     `kernels-cuda-new/csrc/src/sample/argmax.cuh`, now with a
    //     `DeviceKernel` row each in `families::sample`, and BOTH launches
    //     are hand-built `Launch`es with the deleted file's line numbers
    //     cited beside every constant. No `LaunchRule` variant was added:
    //     `grid.x` is `min(num_sms * 2, ceil(vocab / 8))`, read off
    //     `cudaDevAttrMultiProcessorCount`, and a rule whose extent comes
    //     from a device query would serve exactly one kernel. The growable
    //     `static device::u64*` pair scratch between the two launches is a
    //     Rust `static` now, and the cross-stream hazard it always had is
    //     written down at the port rather than reproduced silently.
    //   * `quant/quant_bf16_to_fp8.cu` — 1 launch of 5, ROUTED and
    //     unreachable. `quant::quantize_bf16_to_fp8_e4m3_per_channel` is in
    //     `device::JIT_DISPATCHED`, so the shim emitted no entry;
    //     `model-loader`'s one call (`executor/cuda.rs:622`) goes through
    //     `api::quant_quantize_bf16_to_fp8_e4m3_per_channel` and lands on
    //     `LaunchRule::Rms`, which is `<<<rows, BLOCK, ROW_REDUCE_SHMEM>>>`
    //     digit for digit; `gemm/gemm.cpp`'s include went with §45's
    //     continuation. The file STAYS: three of its remaining four
    //     launchers are named by hand-written `ffi::pie_k_quant_*` arms in
    //     `driver-cuda/src`, which is a full consumer set.
    //   * `quant/dequant_wna16.cu` — 1 launch of 2, ROUTED and unreachable,
    //     the same shape as the two §43.9 took from this file.
    //     `norm/rmsnorm.cu:64` includes this header for `bf16_to_fp16`
    //     ALONE, which is why the file stays and the launcher goes. Its host
    //     guards — `in_dim % 8 != 0 || in_dim % group_size != 0`, which no
    //     `LaunchRule` reproduces — are recorded at the deletion and on the
    //     row.
    //   * `layout/envelope.cu` — 1 launch of 6, DEAD.
    //     `launch_envelope_recompute_bf16` had no caller in any language and
    //     its only row was `table::driver_internal`'s, which no model text
    //     can reach; `new-horizon.md`'s dead sweep had already put it among
    //     the seven rows with nothing at all. The row went with it. The file
    //     stays on `attn/kv_paged.cu:144` and `:321`.
    //
    // `layout/embed.cu` is UNCHANGED and is a stated gap: its host code
    // chooses `embed<true>` or `embed<false>` on a 16-byte alignment test of
    // three pointers, it is live through `lower.rs:1462`, and it is a
    // `table::driver_internal` row — which `execution::RUST_SERVED` cannot
    // take, because `every_taken_over_row_is_stated` resolves through
    // `table::sig` and `driver_internal` is not in `TABLES`. See
    // `new-horizon.md` §54.
    // A SIXTH PASS, AND THE LAST `__global__` IN THE ARCHIVE WENT WITH IT.
    //
    //   * `attn/attention_xqa.cu` — 1 launch of 1, MIGRATED, and this walk is
    //     now a walk over a tree that holds no `__global__` at all. The kernel
    //     is `kernels-cuda-new/csrc/src/attn/attention_xqa.cuh`, compiled by
    //     NVRTC as the `attn/attention_xqa` unit; the host half is
    //     `driver-cuda/src/fire/xqa.rs::prepare_decode`, which carves the
    //     workspace and builds the `Launch` by hand because the grid is
    //     `<<<num_requests, 128>>>` and `LaunchRule::PerRequest` states
    //     `BLOCK = 256`. The launcher it came from,
    //     `prepare_attention_xqa_decode_bf16`, had no table row and therefore
    //     no `pie_k_` shim entry, no C++ caller and no `ffi::` arm; what kept
    //     it was `attn::attention_xqa_decode_bf16_prepared`'s
    //     `needs = Prepare::FireWide`, and the Rust discharges that now.
    //     The `.cu` STAYS and holds no `<<<>>>`: the rest of it ends in
    //     `launchMHAFlashInfer_xqa_gqa5_bf16_p32_h128`, upstream FlashInfer
    //     HOST code reached by `#include <xqa/mha.cu>`, which is §50.1's
    //     measurement and cannot be split further.
    //
    // The sixth pass took the census 110 -> 109 (`attn 48 -> 47`), and its
    // own note recorded that `layout 8 -> 7` and `quant 10 -> 9` in the same
    // table were a CONCURRENT pass's and not its own. That is why the table
    // below is re-walked and never subtracted.
    //
    // A SEVENTH PASS: `attn/`, and the launchers the routing left behind.
    //
    // Nine launches and two whole files, and ALL NINE were
    // ALREADY DEAD when this pass found them — a row had been named in
    // `device::JIT_DISPATCHED` by an earlier pass, `abi::emit_c_shim` had
    // stopped emitting its `pie_k_*` entry, and the launcher was left in the
    // archive. Three of the files still carried a banner saying "Neither has
    // a row". A routed row and a live launcher is the shape §10.10 is about:
    // the shim entry was the whole consumer set and it was already gone.
    //
    //   * `attn/head_dim_pad.cu` — 2 of 2, DEAD, file and `.hpp` DELETED.
    //     `attn::pad_head_dim_bf16` and `attn::strip_head_dim_bf16` are both
    //     in `JIT_DISPATCHED`; `LaunchRule::PerHead` states the
    //     `dim3(num_heads, num_tokens)` grid that was here.
    //   * `attn/kimi_mla.cu` — 1 of 1, DEAD, file and `.hpp` DELETED.
    //     `attn::kimi_split_q_b_bf16` joined `kimi_split_kv_a_norm_bf16` in
    //     `JIT_DISPATCHED`; the file's own header predicted this deletion
    //     and named the measurement that would land it.
    //   * `attn/attention_naive_paged.cu` — 2 of 2, DEAD. The FILE STAYED and
    //     held no `<<<>>>`, on the `attn/attention_xqa.cu` precedent: its
    //     `static_assert`s were the only place `KvCacheScheme`/`DType` were
    //     compared with the `device::KvScheme`/`KvDType` mirrors NVRTC
    //     reads, and deleting launchers does not make mirrors agree. The
    //     `.hpp` is deleted. `attention_naive_paged_bf16` had no row at all
    //     and its whole consumer set was the sibling overload that
    //     dequantised first, so the two went together — and with them the
    //     archive's last call to `dequant_kv_cache_layer_to_bf16_active`
    //     outside `driver-cuda/csrc/`.
    //     THE FILE IS NOW DELETED TOO and the sentence above is why it took a
    //     separate pass: the condition it set for itself was a REPLACEMENT for
    //     that comparison, not the observation that the launchers were dead.
    //     `crates/driver-cuda/tests/enum_mirrors.rs` is that replacement and
    //     it compares the pair that can actually renumber a page — Rust
    //     against the `.cuh` — where the `static_assert`s compared host C++,
    //     which under NVRTC no longer reaches a launch. Writing it found two
    //     `DType` enumerators, `MXFP4_PACKED` and `E8M0`, that the device
    //     mirror never had. `kernels-cuda/csrc/CMakeLists.txt` carries the
    //     evidence block, including the ONE assert the replacement does not
    //     reproduce (`MAX_HEAD_DIM == BLOCK * 8`) and the argument that its
    //     premise had already expired.
    //   * `attn/split_packed.cu` — 1 of 2, DEAD. `attn::split_qkv_bf16` is
    //     routed; `split_qkv_bf16_devwin` stays, because no device row
    //     states its instantiation and its shim entry is still reached.
    //   * `attn/qkv_fused.cu` — 1 of 5, DEAD.
    //     `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` is routed to
    //     `LaunchRule::RowsPackedHeads`. The other four stay: the warp form
    //     is rowed at ONE head-dim expansion (`_warp_d128`), so the `== 64`
    //     and `== 256` arms and the block-form fallthrough reach no row.
    //   * `attn/dsv4_compress.cu` — 2 of 6, DEAD.
    //     `attn::dsv4_compress_gather_paged_bf16` and
    //     `attn::dsv4_store_comp_entries_bf16` are routed. Their warp-rounded
    //     block width — `head_dim < 256 ? ceil_to_warp(head_dim) : 256` — is
    //     a measurement and is recorded at both deletions.
    //   * `attn/kv_paged.cu` — 0 of 18, and the count is UNCHANGED on
    //     purpose. `write_kv_explicit_bf16_devwin` was ported —
    //     `driver-cuda/src/fire/kv_paged.rs` is the whole program, geometry
    //     cited line by line — and then NOT routed, because the takeover is
    //     structurally blocked: `execution::tests::a_walk_is_only_a_walk`
    //     requires a walked symbol to have no unit, and
    //     `device::Specialisation::agrees` requires `WRITE_KV_EXPLICIT_DEVWIN`'s
    //     base to HAVE one. The symbol must both have and not have a device
    //     row. Backing the routing out and leaving the C++ in place is the
    //     honest answer; `.wiki/driver/new-horizon.md` §56.1 carries it, and
    //     §56 states what each of the file's eight launchers is blocked on.
    //
    // EVERY ONE OF THE NINE WAS CLOSED IN THE SAME EDIT THAT DELETED IT.
    // A deleted launcher whose row still states `operands` and is named in
    // neither `device::JIT_DISPATCHED` nor `execution::RUST_SERVED` gets a
    // `pie_k_*` shim entry forwarding to a definition that is not there --
    // `abi.rs:141-145` is the filter pair. All nine were ALREADY in
    // `JIT_DISPATCHED`, which is why their launchers were dead; the ninth,
    // `attention_naive_paged_bf16`, never had a row. `.wiki/driver/
    // new-horizon.md` §56.7 has the table and the tree-wide re-scan: 55 rows
    // stated-and-shimmed after both filters, 0 of them missing a launcher.
    //
    // THIS PASS'S OWN ARITHMETIC IS 109 -> 100: nine launches, all in `attn/`.
    //
    // THE CONSTANT BELOW IS 67, AND THE DIFFERENCE IS NOT THIS PASS'S. An
    // `ssm/` pass is running concurrently and deleted `causal_conv1d.cu`,
    // `gated_delta_net.cu` and `kda.cu` -- 33 launches -- between this
    // function's first walk and its last, without yet updating this number.
    // Both walks are recorded because the rule here is *re-derive, never
    // subtract*, and a reader who finds 67 and expects 100 needs to know
    // which pass moved which 33:
    //
    //   this pass, `attn/` only:      47 -> 38 across 23 files   (-9)
    //   the concurrent `ssm/` pass:   35 ->  2 across  1 file    (-33)
    //
    // The walk below, this function's own rule, per directory, taken fresh
    // and last:
    //
    //   attn    38 across 23    layout   7 across  2    quant   9 across  4
    //   comm     0 across  1    moe     10 across  3    norm    1 across  1
    //   ssm      2 across  1
    //
    //                                          TOTAL   67 across 35
    //
    // If the `ssm/` pass lands more, it re-walks and this number moves again;
    // that is the mechanism working, not a conflict.
    //
    // AN EIGHTH PASS: `ssm/`, and it is the 33 the note above predicted.
    // `.wiki/driver/new-horizon.md` §57 is the account in full.
    //
    // The number was already correct when this pass arrived — the `attn/`
    // pass walked the tree AFTER these deletions landed on disk and wrote
    // what it found. This pass re-walked independently and got the same
    // seven directory counts, so nothing here changes `EXPECTED`; what was
    // missing was the account of WHICH 33 and why, which is the part a
    // number cannot carry.
    //
    // Every kernel behind these launches was ALREADY compiled by NVRTC — the
    // archive holds no `__global__` — so nothing here is a kernel migration.
    // What moved is host code, into
    // `driver-cuda/src/fire/{causal_conv1d,gated_delta_net,kda,nemotron_h}.rs`,
    // and `execution::WALKED` states each launcher with the `.cu` line its
    // geometry came from.
    //
    //   * `ssm/causal_conv1d.cu` — 3 of 3, file and `.hpp` DELETED.
    //     `causal_conv1d_update_batched_bf16` was already routed;
    //     `causal_conv1d_prefill_batched_bf16` is `execution::RUST_SERVED`
    //     and carries a `Control::Switch` on `R >= 8`. That `if` was CHECKED
    //     against §30's precedent and is NOT the identical-arms shape: the
    //     two kernels open on different indices (`blockIdx.x * blockDim.x +
    //     threadIdx.x` at `causal_conv1d.cuh:310` against `blockIdx.x` at
    //     `:225`), so each is correct only under its own grid and there is no
    //     shape at which they agree.
    //   * `ssm/gated_delta_net.cu` — 17 of 17, file and `.hpp` DELETED, and
    //     EIGHT OF THE SEVENTEEN WERE UNREACHABLE. Three `constexpr bool`
    //     selectors in its anonymous namespace — `qwen_gdn_gqa_ilp2_enabled`,
    //     `qwen_gdn_k_last_state_enabled`, `qwen_gdn_fused_step_enabled` —
    //     were all `false`, so every `_fused` launch and every `KLast = true`
    //     instantiation sat behind a branch no build reached. None is ported
    //     and none gets a row: a row for an unreachable kernel is a contract
    //     with an empty consumer set. Five of the nine launchers were already
    //     routed; the four that remained are `RUST_SERVED`. The file's
    //     `gdn_raise_shmem_cap` moved to
    //     `runtime::module::raise_dynamic_smem_cap` — at the fire, for every
    //     kernel over the 48 KiB default, rather than for the one that
    //     needed it.
    //   * `ssm/kda.cu` — 4 of 4, file and `.hpp` DELETED. Two launchers were
    //     already routed; `kda_recurrent_step_batched` and
    //     `kda_prefill_batched` are `RUST_SERVED` with rows that state no
    //     `Source` on any operand, so `abi::emit_rust_dispatch` writes no arm
    //     for either and neither is reachable from a model trace. Listing
    //     them is still what dropped the shim entries, which is what let the
    //     file go.
    //   * `ssm/nemotron_h.cu` — 9 of 11, and the FILE STAYS for the two
    //     `build_nemotron_moe_ptrs_*` launchers, which fill arrays of device
    //     pointers into a driver-owned slab and are blocked on
    //     `Source::Scratch` — `new-horizon.md` §52.3, §56 and §57.3. TWO of
    //     the nine were dead before this pass: an `if constexpr (false)`
    //     decode-tile arm at `:143` and a launch after an unconditional
    //     `return` at `:182`, the only launch of `mamba_ssm_batched` anywhere
    //     in the tree. Neither is ported and neither gets a row.
    //
    //   this pass, `ssm/` only:       35 ->  2 across  1 file    (-33)
    //
    // A NINTH PASS: `moe/`, `quant/`, `norm/` and the tail of `ssm/` — every
    // launcher in the tree that no other pass owned. `.wiki/driver/
    // new-horizon.md` §60 is the account in full.
    //
    // THE CONSTANT DID NOT MOVE FOR THIS PASS EITHER, and for the same reason
    // the eighth's did not: the `attn/` pass walks concurrently and had
    // already written what it found after these deletions reached disk. This
    // pass re-walked with the rule below, independently and last, and got 47.
    // Re-derive, never subtract — a delta against a tree two passes are
    // editing is arithmetic about a state neither of them saw.
    //
    // The account, per file, and every kernel behind every one of these was
    // already NVRTC's:
    //
    //   * `quant/dequant_fp8.cu` (3), `quant/dequant_fp4.cu` (1) and
    //     `quant/dequant_wna16.cu` (1) — files and `.hpp`s DELETED. All five
    //     launchers were ALREADY in `device::JIT_DISPATCHED`, so the shim
    //     emitted no entry and the entry was the whole consumer set. The
    //     `norm/rmsnorm.cu:64` caller the migration notes recorded for
    //     `dequant_wna16` went with that file, one pass earlier.
    //   * `quant/quant_bf16_to_fp8.cu` (4) — file and `.hpp` DELETED, which
    //     empties `csrc/src/quant/`. THIS IS THE FILE THREE HAND-WRITTEN
    //     `ffi::pie_k_quant_*` ARMS HELD, and a hand arm is invisible to
    //     every check that reads generated text. All three were in
    //     `bind/quant_gemm.rs` and now call
    //     `driver-cuda/src/fire/quant_int8.rs`; the three
    //     `table/driver_internal.rs` rows that existed only to give them a
    //     shim entry are deleted with them. ONE OF THE FOUR LAUNCHERS WAS
    //     DEAD: `launch_dequant_int8_to_bf16_per_channel` had no caller in
    //     any language and is deleted rather than ported.
    //   * `norm/residual_add.cu` (1) — file and `.hpp` DELETED, which empties
    //     `csrc/src/norm/`. Its last C++ caller left with `gemm.cpp`'s
    //     quantized arms; `bind/quant_gemm.rs` fires the row through
    //     `bind::jit::fire` and `norm::residual_add_bf16` joined
    //     `device::JIT_DISPATCHED`.
    //   * `ssm/nemotron_h.cu` (2) — file and `.hpp` DELETED, which empties
    //     `csrc/src/ssm/`. The eighth pass left these two for
    //     `Source::Scratch`, and THE ROWS ARE STILL UNBOUND: §52.3's gap is
    //     exactly where it was. What moved is only that
    //     `execution::RUST_SERVED` names both, so `emit_c_shim` stops
    //     emitting an entry — an unbound row keeps its shim entry only while
    //     the shim is its only executor, and `fire/nemotron_h.rs` is one now.
    //   * `moe/moe_dispatch.cu` (9 -> 3 -> 0) — FILE AND `.hpp` NOW DELETED.
    //     This entry recorded a block and the block is gone, so both halves
    //     stay: the reasoning was right about every mechanism it named and
    //     wrong about the conclusion, which is worth more than a corrected
    //     line.
    //
    //     What it said: `moe::scatter_add_weighted_bf16`,
    //     `moe::add_moe_route_bias_bf16` and `moe::moe_bucket_exact` are each
    //     ALREADY unit-hosted (`LaunchRule::PerRow`, `Rms`, `RouterSort`)
    //     AND each has a deliberately unsourced `table::moe` row. A
    //     unit-hosted symbol cannot be `Walk`, `Service` or `Composed` —
    //     all three assert `unit_of` is `None` — so its only route out of the
    //     shim is `JIT_DISPATCHED`, which `emit_rust_dispatch` skips WHOLE
    //     for a row with an unsourced operand. Every clause of that is still
    //     true.
    //
    //     What it missed: the unsourcedness is not what blocks the walk, it
    //     is what EARNS it. `Control::Supplies` is a value the launch needs
    //     that no row can state, and a row left unsourced because a host must
    //     fill it is that value by definition — `moe_bucket_exact`'s
    //     `(3E + 1) * 4` shared slab, `add_moe_route_bias`'s `cols` and
    //     `out_stride`, `scatter_add_weighted`'s `num_routed`, which is not
    //     even an argument of the `__global__`. So all three took §60.6's
    //     symbol split: the device rows are `moe::scatter_add_weighted_dev_bf16`,
    //     `moe::moe_bucket_exact_dev` and `moe::add_moe_route_bias_dev_bf16`,
    //     the ABI symbols are `Execution::Walk` and `execution::RUST_SERVED`,
    //     and `driver-cuda/src/fire/moe_dispatch.rs` now holds all eight of
    //     the file's launchers. The `_dev` rows KEEP their `LaunchRule`s: a
    //     rule and a Rust launcher stating one rectangle is a check, not a
    //     duplication.
    //
    //     The header's `moe_aligned_block()` went with it — no C++ caller, and
    //     its kimi26-mini batch-128 measurement (16 -> 1.184 ms, 32 -> 0.811,
    //     64 -> 0.746, 128 -> 0.796) is on the Rust. Its `forced` override was
    //     DELETED rather than ported: the static lambda's first statement is
    //     `return 0`, so `if (forced != 0)` is a branch whose taken arm cannot
    //     be entered — §30's reading, reached without measuring anything.
    //   * `moe/dsv4_routing.cu` (1 -> 0) — FILE AND `.hpp` NOW DELETED, the
    //     same door and the same key. `moe::hash_route_lookup`'s device row is
    //     `moe::hash_route_lookup_dev` and keeps `LaunchRule::RowsFlat`, which
    //     it is still the only member of; the symbol is walked and
    //     `RUST_SERVED`; the launcher is
    //     `driver-cuda/src/fire/dsv4_routing.rs`. What no `Source` names here
    //     is `tid2eid`, a `[vocab, K]` table keyed by TOKEN ID, and its first
    //     extent `vocab_size` — the fire's rectangle does not carry the
    //     vocabulary. §60.3 called this blocked for the same reason as the
    //     three above.
    //
    //     **`csrc/src/moe/` is now at ZERO launches**, holding
    //     `flashinfer_moe.cu` alone, which contains no `<<<` at all — and a
    //     second reading of that zero finished the argument. The file was 817
    //     lines with 0 `__global__`, 0 `__device__`, 2 `std::mutex` and 1
    //     `std::unordered_map`: a workspace query, an arch probe, an
    //     autotuner, a tactic cache and a dispatch, none of which NVRTC could
    //     have compiled and none of which it was ever asked to. That host
    //     program is `driver-cuda/src/fire/flashinfer_moe.rs`; what is left
    //     at this path is a five-function `extern "C"` seam that instantiates
    //     CUTLASS templates. That seam is the LAST ahead-of-time CUDA
    //     compilation in this family and is a state, not a settlement — the
    //     principle has no exception now, so `pie_flashinfer_cutlass_moe`,
    //     its generated `_SM90_`/`_SM100_` lists and the CPM fetch all still
    //     have to go — and that is no longer blocked on an unmeasured
    //     question. A concrete sm90 ptr-array grouped GEMM compiles under
    //     NVRTC at compute_90a to 1,245,452 B of PTX with one `.entry`; the
    //     gap was three `std::` names, three specific cub includes instead
    //     of the umbrella, and two `griddepcontrol` asm lines. §13.6's price
    //     was FA2's and does not transfer. `Params` is what is still open.
    //     **The census does not move**: 0 `<<<` before, 0 after.
    //     `moe::flashinfer_cutlass_moe_bf16` is LIVE —
    //     `crates/model/src/qwen_3_5/forward/mod.rs:362` reaches it through
    //     `dsl::cuda::moe_fused_cutlass`, a different token from the symbol
    //     string, which is how a survey once reported it uncalled. It is
    //     `RUST_SERVED` now, which is what drops its shim entry.
    //   * `layout/embed.cu` (2) SURVIVES, blocked exactly where §60.4 left it:
    //     `every_taken_over_row_is_stated` resolves through `table::sig`,
    //     which scans `TABLES`, and `driver_internal` is deliberately not in
    //     `TABLES`.
    //
    // ── THE PASS AFTER THAT ONE (§61): the envelope wall, taken ───────────
    //
    // §60.5 called `layout/envelope.cu` blocked on *"a `layout/envelope`
    // NVRTC unit (there is no `layout` unit at all) and a `DeviceKernel` row
    // spelling `Tu = 0`"*. Both existed to be written and both are written.
    //
    //   * `layout/envelope.cu` (5 -> 0) — FILE AND `.hpp` DELETED, which
    //     leaves `csrc/src/layout/` holding `embed.cu` alone.
    //     `families::layout::ENVELOPE` is the unit, five
    //     `LaunchRule::Unstated` rows are its instantiations, and
    //     `driver-cuda/src/fire/envelope.rs` states every geometry beside the
    //     `<<<>>>` it came from. The `Tu = 0` spelling is `device::i32(0)`;
    //     the cost is that `abi::emit_device_typecheck` refuses a value-headed
    //     `elem`, which the module doc says out loud. The two
    //     `table::driver_internal` rows are DELETED — a `driver_internal` row
    //     can never be `RUST_SERVED`, because `table::sig` cannot see it, so
    //     deletion is its only close (the `copy_kv_cells_bf16` precedent) —
    //     and `bind::abi::seed_envelopes_empty`, the tree's one hand
    //     `ffi::pie_k_layout_*` arm, calls `fire::envelope::seed_empty` with
    //     its own signature unchanged.
    //   * `attn/kv_paged.cu` (16 -> 8) — FILE SURVIVES, and the reason is
    //     named below rather than left as a number. **THAT `SURVIVES` IS
    //     SUPERSEDED: the pass at `EXPECTED` below took the file to 0 and
    //     deleted it, with `attn/kv_paged.hpp`.** Three launchers moved:
    //     `write_kv_to_pages_bf16` (2, no `table` row of its own — a C++
    //     helper, so no shim entry to drop), `write_kv_to_pages` (4) and
    //     `write_kv_explicit_bf16` (2). The last two are LIVE `table::attn`
    //     rows with `dsl.rs` wrappers; both were classified
    //     `Execution::Walk` and then named in `execution::RUST_SERVED`, which
    //     drops the shim entry and routes the generated arm to
    //     `bind::service::attn_write_kv_{to_pages,explicit_bf16}`.
    //     `write_kv_explicit_bf16` needed §60.6's symbol split first: its
    //     DEVICE rows are `attn::write_kv_explicit_bf16_dev` and arms now, so
    //     `a_walk_is_only_a_walk`'s `unit_of(sym).is_none()` holds for the
    //     symbol a trace records.
    //
    // WHY THIS FILE CANNOT REACH ZERO, stated so the next pass does not
    // re-derive it: `dequant_kv_cache_layer_to_bf16_active` (4 of the
    // remaining 8) has a live C++ caller in
    // `crates/driver-cuda/csrc/attn/attention_flashinfer.cu`. C++ calling C++
    // is not interceptable by any of the three shim mechanisms, and that
    // directory is off limits to this migration.
    //
    // The walk below, this function's own rule, per directory, taken fresh
    // and last:
    //
    //   attn    28 across 23    layout   2 across  1
    //   comm     0 across  1    moe      4 across  3
    //
    //                                          TOTAL   34 across 28
    //
    //   this pass:  47 -> 34 across 28 files   (-13, -1 file)
    //
    // ── THE moe PASS DID NOT UPDATE THIS NUMBER, AND HERE IS WHY ─────────
    //
    // The pass that emptied `csrc/src/moe/` walked the tree fresh, by this
    // function's own rule, and got **14 across 19**:
    //
    //   attn    14 across 17    comm     0 across  1
    //   moe      0 across  1
    //
    // Its own contribution to that is exactly `moe 4 across 3 -> 0 across 1`,
    // −4 launches and −2 files. Everything else in the gap belongs to another
    // pass running concurrently in `csrc/src/attn/**` and `csrc/src/layout/`,
    // which are not this one's to count: `attn` is mid-flight at 28 -> 14 and
    // `layout` has gone to zero files, so any constant written here would be
    // a snapshot of someone else's half-finished edit and would be wrong
    // again by the time it landed.
    //
    // ── THE PASS THAT FINISHES `attn` HAS WALKED. THIS IS THAT WALK ─────
    //
    // The note above ended "the pass that finishes `attn` owns the
    // re-derivation". It is this one, and this is the re-derivation --
    // taken fresh by this function's own rule over the tree as it stands,
    // never as a delta from 34 or from anything else:
    //
    //   attn     4 across 15    comm     0 across  1
    //   moe      0 across  1    gemm/layout/rope/vision: no `.cu` at all
    //
    //                                          TOTAL    4 across 17
    //
    // Every one of the four is in ONE file, `attn/kv_paged.cu`, and every one
    // of the four belongs to ONE launcher,
    // `dequant_kv_cache_layer_to_bf16_active`, whose block the note above
    // already states and which has not moved: four live C++ callers in
    // `crates/driver-cuda/csrc/attn/attention_flashinfer.cu` (`:648`, `:675`,
    // `:1098`, `:1244`). C++ calling C++ is intercepted by none of the three
    // shim mechanisms -- `device::JIT_DISPATCHED` and `execution::RUST_SERVED`
    // both act on the GENERATED shim, and there is no generated shim between
    // one `.cu` and another -- and that directory is off limits.
    //
    // So FOUR IS THE FLOOR THIS MIGRATION CAN REACH, and reaching it is what
    // this pass did. What closed in it, per file:
    //
    //   * `attn/qkv_fused.cu` (4 -> file DELETED) — the last `attn`
    //     dispatch, and the one no `Specialisation` could state: `head_dim`
    //     picks the warp form at 64/128/256 and falls THROUGH to the block
    //     form otherwise, and the two forms have different `LaunchRule`s.
    //     `Execution::Walk` with `Control::Switch` + `RUST_SERVED`, and FOUR
    //     device rows written for the `_d64`/`_d256` warp expansions the unit
    //     never had.
    //   * `attn/kv_paged.cu` (6 -> 4) — `write_kv_explicit_bf16_devwin` (2)
    //     closed by §60.6's symbol split, which is what §58 was missing: the
    //     DEVICE rows are `..._devwin_dev` and `WRITE_KV_EXPLICIT_DEVWIN`'s
    //     `base` moved with them, so the ahead-of-time symbol became
    //     unit-free and walkable while the `Specialisation` still resolves.
    //     Its Rust had been written a pass earlier and was waiting for
    //     exactly this.
    //
    // Earlier files in the same pass -- `pack_dense_mask`, `mla_paged`,
    // `layout/embed`, `split_packed`, `page_compact`, `attention_naive`,
    // `dsa_indexer`, `dsv4_compress`, and the two page-view builders -- are
    // in the walk above as absences; `csrc/src/layout/` holds no `.cu` at
    // all now, and neither do `gemm`, `rope` or `vision`.
    //
    // ── `comm/` IS GONE. THE COUNT DOES NOT MOVE, AND THAT IS THE POINT ──
    //
    // Walked fresh over the tree as it stands, by this function's own rule,
    // never as a delta:
    //
    //   attn     4 across 15    moe      0 across  1
    //   comm/gemm/layout/norm/rope/sample/ssm/vision: no `.cu` at all
    //
    //                                          TOTAL    4 across 16
    //
    // `-0 launches, -1 file`. `comm/custom_all_reduce.cu` is DELETED, along
    // with `comm/custom_all_reduce.hpp` and `comm/custom_all_reduce_stub.cpp`,
    // and `csrc/src/comm/` is removed.
    //
    // It contributed **zero** to `total` when it arrived at this list and
    // zero when it left, and the reason is the finding: 664 lines with zero
    // `__global__` and zero `<<<>>>`. It was a HOST PROGRAM wearing a `.cu`
    // extension for linkage -- the fifth this migration has found, after
    // `moe/flashinfer_moe.cu` (817, 0), `vision/qwen3_vl_tower.cu` (522, 0),
    // `attn/attention_flashinfer.cu` (1,258, 0) and `gemm/gemm.cpp` (1,267,
    // 0, 95 cuBLAS calls). A launch census cannot see a file like that, which
    // is exactly why the FILE count is asserted beside the launch count and
    // why this note re-derives both.
    //
    // What crossed: the whole lifecycle -- peer-access enablement, the IPC
    // handle exchange, the `RankData` slab, the fusion plane's four
    // allocations and its Lamport initialisation, the NCCL crossover query
    // and the 240-point template dispatch -- to
    // `driver-cuda/src/fire/all_reduce.rs`. What did not: two calls into
    // CPM-fetched flashinfer headers `csrc/vendor/` does not carry, which are
    // refusals naming the exact template point. Both symbols are
    // `execution::RUST_SERVED`, so `emit_c_shim` drops both entries.
    //
    // FOUR REMAINS THE FLOOR for the reason stated above, unchanged: all four
    // are `dequant_kv_cache_layer_to_bf16_active` in `attn/kv_paged.cu`, with
    // four live C++ callers in `crates/driver-cuda/csrc/attn/
    // attention_flashinfer.cu`, a directory off limits to this migration.
    //
    // ── ZERO. `attn/kv_paged.cu` IS DELETED AND THE CENSUS IS CLOSED ─────
    //
    // Walked fresh over the tree as it stands, by this function's own rule,
    // never as a delta:
    //
    //   attn     0 across 14    moe      0 across  1
    //   comm/gemm/layout/norm/rope/sample/ssm/vision: no `.cu` at all
    //
    //                                          TOTAL    0 across 15
    //
    // `-4 launches, -1 file`, and it is the last `-4` there is to take. The
    // paragraph above named the exact thing holding the floor — four live C++
    // callers in `driver-cuda/csrc/attn/attention_flashinfer.cu` — and that
    // file is deleted, along with the whole of `driver-cuda/csrc/`. The four
    // call sites are `driver-cuda/src/bind/service.rs`' four FA2 entry points
    // calling `fire::kv_paged::dequant_kv_cache_layer_to_bf16_active`, which
    // fires the same four kernels through NVRTC out of the same
    // `kernels-cuda-new/csrc/src/attn/kv_paged.cuh` the archive compiled.
    //
    // `attn/kv_paged.hpp` went with it, and the symbol is
    // `execution::RUST_SERVED` — classified `Execution::Walk` with
    // `Control::Switch { on: "layer.scheme" }` first — so `emit_c_shim` emits
    // no `pie_k_` entry for a launcher that no longer exists. The ROW is
    // live and unchanged: `model-compiler/src/dsl.rs:7750` states it and
    // `emit_rust_dispatch` still writes its arm.
    //
    // WHAT ZERO DOES AND DOES NOT MEAN. It does not mean no kernel launches;
    // it means **no launch is issued from C++ in this tree**. Every one is
    // issued by `driver-cuda` through `kernels_cuda_new::runtime`, against a
    // `__global__` NVRTC compiled from a `.cuh`.
    //
    // AND IT IS ZERO OVER `.cu` FILES, WHICH IS NOT THE SAME AS ZERO OVER
    // WHAT NVCC COMPILES. `sources_with("cu")` opens `.cu` and nothing else,
    // so a launch in an INCLUDED header is outside this count by
    // construction, and there are two of them: this crate's
    // `attn/attention_mla.cu:17` includes
    // `kernels-cuda-new/csrc/src/attn/attention_mla_naive.cuh`, whose host
    // launchers fire `mla_naive_paged_kernel` at `:266` and
    // `mla_mma_paged_kernel` at `:725`. Both are reachable from nvcc through
    // that one include and through no other -- measured, that header has
    // exactly one includer in either `csrc` tree.
    //
    // They are NOT counted here and they are NOT this pass's: `attention_mla`
    // and its header belong to the pass that is porting the XQA and MLA
    // launchers, and a count that reached into another pass's files to make
    // its own number look rounder would be the same dishonesty as widening a
    // scan to hide a deleted root. The number this file states is the number
    // this file measures. The tree reaches nvcc-zero when that header's two
    // launchers land in Rust, and `attn/attention_mla.cu` -- the only
    // translation unit that can reach them -- is deleted with them.
    //
    // The `.cu` files that remain hold zero `<<<>>>` each, which is why the
    // total is the same whether they are deleted or not, and the set is
    // shrinking under that other pass as this is written, so it is
    // deliberately not enumerated here: a list of filenames in a comment is
    // a second census that can disagree with the first. `files` in the
    // message below is the live count and needs no help.
    //
    // A `.cu` with no launch is a translation unit awaiting a reason to
    // exist, not a hole in this count.
    //
    // THE ASSERTION BELOW CHANGES MEANING WITH THIS NUMBER, and that is
    // deliberate rather than a weakening. At 401 and at 4 it caught a DROP —
    // a split that moved device text out and took a host launcher with it,
    // which compiles, links and silently stops launching. At 0 there is
    // nothing left to drop, so what it catches is a RISE: a `<<<>>>`
    // reappearing in this archive at all. Whoever adds one is adding nvcc
    // back, and this line is where that argument has to be made.
    // ── TWO. THE COUNT RISES AND NO LAUNCH WAS ADDED ────────────────────
    //
    // Walked fresh over the tree as it stands, by this function's own rule,
    // never as a delta:
    //
    //   attn     2 across  8    moe      0 across  1
    //   comm/gemm/layout/norm/rope/sample/ssm/vision: no `.cu` at all
    //
    //                                          TOTAL    2 across  9
    //
    // `+2 launches, -6 files`, and the two halves of that have nothing to do
    // with each other.
    //
    // THE `-6` FIRST, because it is the ordinary half. `attn` went 14 -> 8:
    // `attention_flashinfer_hd{64,128,256,512}.cu` (13 lines each, a single
    // `template struct AttnHd<N>;` apiece, replaced by `families::fa2`'s 56
    // units), `attention_merge_states.cu` (47) and
    // `attention_flashinfer_hopper.cu` (392). Every one contributed ZERO to
    // this total on arrival and zero on departure, so the launch count could
    // not move and did not.
    //
    // THE `+2`, WHICH IS THE ENTRY WORTH READING. **No launch was written,
    // duplicated, or restored. Two that this census could not see became
    // visible to it.** The paragraph above this one predicted them by name:
    //
    // > a launch in an INCLUDED header is outside this count by construction,
    // > and there are two of them: this crate's `attn/attention_mla.cu:17`
    // > includes `kernels-cuda-new/csrc/src/attn/attention_mla_naive.cuh`,
    // > whose host launchers fire `mla_naive_paged_kernel` at `:266` and
    // > `mla_mma_paged_kernel` at `:725`.
    //
    // That header is now DEVICE TEXT ONLY — two `__global__`s, zero `<<<>>>`,
    // NVRTC-clean at `sm_89` under this tree's numerics contract — and its
    // four host functions (`mma_detail::smem_bytes`, `mla_mma_supported`,
    // `launch_mla_mma_paged_raw`, `launch_mla_naive_paged_raw`) are in
    // `attn/attention_mla.cu:48-200`, the only translation unit that ever
    // reached them. The launches did not move ACROSS the nvcc boundary; they
    // moved DOWN to the side of it they were always on.
    //
    // So the honest reading of the previous entry is that **`0` was under by
    // two**, and the reason is scope rather than arithmetic: `sources_with
    // ("cu")` was a correct classifier for the entire life of this tree —
    // a `.cuh` under `kernels-cuda-new/csrc` is device text carried into
    // NVRTC, and device text does not launch — and it stopped being one the
    // moment a `.cuh` grew a launcher. A measured premise that names its
    // evidence can still expire, and the evidence is what has to be
    // re-checked, not the conclusion. The repair was to move the launchers
    // into the file this census already opens, NOT to widen the scan:
    // widening was available and would have bought a bigger number over a
    // tree that still hid host code in a header.
    //
    // WHAT IT WOULD TAKE TO GET BACK TO ZERO, precisely, because "port the
    // launchers" is not it — they ARE ported, at
    // `driver-cuda/src/fire/mla_naive.rs`, with both grids cited and the
    // `std::call_once` opt-in answered by `runtime::module::
    // raise_dynamic_smem_cap`. What holds `attn/attention_mla.cu` on disk is
    // its OTHER arm: `dispatch_attention_mla_bf16` is one row with two, and a
    // row loses its shim entry whole or not at all. The FA2 arm calls
    // `flashinfer::mla::BatchMLAPagedAttention<MASK, 512, 64>` passing one
    // `MLAParams` BY VALUE, which needs `ArgValue::Bytes`, which
    // `Args::bind` says only `x::Abi` produces. That capability is in
    // flight in another crate and it clears XQA's `KVCacheList` at the same
    // time. Until it lands, deleting this file would leave `emit_c_shim`
    // writing a `pie_k_` entry with no definition — a link error, not a
    // clean removal, which is the exact trap `kv_paged.cu` set and passed.
    //
    // THE ASSERTION BELOW CHANGES MEANING AGAIN, and back. At 401 and at 4 it
    // caught a DROP. At 0 it caught a RISE — nvcc coming back. At 2 it is a
    // FLOOR with a named holder, the shape this file has used four times
    // before: the two are `attn/attention_mla.cu`'s and they leave when the
    // `x::Abi` impl lands and the file does. A third would be new nvcc and
    // has to be argued here.
    // ── THREE. A FILE LEAVES AND THE COUNT DOES NOT MOVE ────────────────
    //
    // Walked fresh over the tree as it stands, by this function's own rule,
    // never as a delta:
    //
    //   attn     2 across  7    moe      0 across  1
    //   comm/gemm/layout/norm/rope/sample/ssm/vision: no `.cu` at all
    //
    //                                          TOTAL    2 across  8
    //
    // `+0 launches, -1 file`. `attn/attention_naive_paged.cu` is DELETED, and
    // it is the only file this archive has lost that contributed nothing to
    // this count at either end AND had no host program to name on the way out
    // — it held no launcher, no launch and no `__global__`. Its content was
    // two `static_assert` blocks, and what replaces it is a CHECK:
    // `crates/driver-cuda/tests/enum_mirrors.rs`. The full argument, the one
    // assert the replacement does not reproduce, and both halves of its
    // consumer set are in `kernels-cuda/csrc/CMakeLists.txt`; the entry at
    // `attn/attention_naive_paged.cu` in the list above is amended in place.
    //
    // THE ONE SENTENCE ABOVE THAT HAS EXPIRED, corrected rather than edited
    // away, because what it got wrong is the shape of the remaining work and
    // not the conclusion. The `TWO` entry says the `ArgValue::Bytes`
    // capability *"is in flight in another crate and it clears XQA's
    // `KVCacheList` at the same time."* **It has landed** —
    // `kernels-cuda-new/src/x/xqa.rs` holds the `KvCacheList` mirror, a
    // `by_value!` whose size, alignment and five offsets were MEASURED out of
    // NVRTC's PTX rather than read off the declaration, and the `Abi` impl
    // that produces the variant. `EXPECTED` is still 2 and the holder is
    // still `attn/attention_mla.cu`, so the correction changes the reason and
    // not the number:
    //
    //   * XQA needed the floor AND the six-member enrolment on top of it —
    //     `unit!`, `contract!` and `bind!` for `families::attn::XQA_LATTICE`,
    //     which `x/xqa.rs`'s own header says *"belong here too and are yours
    //     to add"* and which are not written. Until they are,
    //     `attn::attention_xqa_decode_bf16_prepared` states `operands`, is in
    //     neither `JIT_DISPATCHED` nor `RUST_SERVED`, and keeps a `pie_k_*`
    //     entry that the six `attention_xqa*.cu` are the definition of.
    //   * MLA needs the same three plus a `by_value!` for `MLAParams`, which
    //     is the HARDER mirror: `KVCacheList<true>` is four pointers and a
    //     `uint32_t` with no nested aggregate, while `MLAParams` embeds two
    //     `uint_fastdiv`s — 24 bytes each where the declaration reads like
    //     one `u32`, per `x/xqa.rs`'s own table of transcription traps.
    //
    // So the blocker moved from a missing capability to an unwritten
    // registration, in a directory this pass does not own. Neither is a
    // reason to widen this scan or to move the number.
    const EXPECTED: usize = 2;
    assert_eq!(
        total, EXPECTED,
        "the tree executes {total} launches across {files} files, not {EXPECTED}. \
         A DROP means a split moved device text out and took a host launcher \
         with it -- which compiles and links and silently stops launching. A \
         RISE means a kernel body was duplicated, or a launcher deliberately \
         added; if deliberate, update this number and say why in the list above."
    );
}

/// Every `.cu` under `dir`, recursively.
fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The environment-variable audit (`new-horizon.md` §36)
// ---------------------------------------------------------------------------

/// A source line that is CODE — a `//` comment is not.
///
/// The distinction is the whole guard. A launcher that removed a selector
/// SHOULD name the one it used to have, at the decision that replaced it, or
/// the next reader re-invents it; every file audited in §36 does exactly
/// that. So the prose may say `getenv` and the code may not, and the only
/// thing that tells them apart is whether the line is a comment. `*` catches
/// the continuation lines of a block comment.
fn code_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines().enumerate().filter_map(|(i, line)| {
        let t = line.trim_start();
        if t.starts_with("//") || t.starts_with('*') || t.starts_with("/*") {
            None
        } else {
            Some((i + 1, line))
        }
    })
}

/// Every CODE line of `text` that reads the environment, as `line: text`.
fn env_reads(text: &str) -> Vec<String> {
    code_lines(text)
        .filter(|(_, line)| line.contains("getenv"))
        .map(|(n, line)| format!("  {n}: {}", line.trim()))
        .collect()
}

fn read_src(rel: &str) -> String {
    std::fs::read_to_string(csrc().join("src").join(rel))
        .unwrap_or_else(|e| panic!("{rel} reads: {e}"))
}

/// A quoted line is still there, and still where the doc says it is.
///
/// Copied in spirit from `kernels-cuda-new/tests/launch_rules.rs`, and for
/// the same reason: a citation that is only checked for CONTAINMENT survives
/// the code moving out from under it, and a stale citation agrees with
/// everything. Both failure modes are fired in
/// `the_environment_guard_fails_when_a_selector_comes_back`.
fn pinned(text: &str, file: &str, line: usize, want: &str, what: &str) {
    let lines: Vec<&str> = text.lines().collect();
    let found: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.trim() == want.trim())
        .map(|(i, _)| i + 1)
        .collect();
    assert!(
        !found.is_empty(),
        "{file} no longer contains the line {what} was replaced by:\n  {want}\n\
         Either the launcher was rewritten — in which case re-measure the arms \
         and move the citation — or the selector came back."
    );
    assert!(
        found.contains(&line),
        "{file}: {what}'s replacement is at line(s) {found:?}, not {line}. \
         The text is intact and has MOVED; update the citation, because a pin \
         that only checks containment stops noticing when the code drifts \
         away from the section that describes it."
    );
}

/// The audited launchers pick on the fire, on the device, or on a fact
/// someone states — and read no environment variable.
///
/// # What this is guarding against
///
/// A `getenv` that selects a kernel makes a replay irreproducible. Same
/// trace, same weights, same GPU, and a different `__global__` runs with
/// nothing in the plan recording which one. §30 closed the first of these
/// (`PIE_QWEN35_GDN_SMEM_STEP`, whose two arms turned out to be
/// byte-identical); §36 closed four more, and those were NOT all harmless:
///
/// * `gemm/gemv.cu` — `PIE_GEMV_B200_TUNING` reached three launchers, and two
///   of the three arm-pairs it chose between emit DIFFERENT BYTES once the
///   weight exponents spread (5 and 3 differing bytes over nine shapes,
///   measured on an L40S). `PIE_GEMV_SPLITK_MAX_ROWS` chose which of two
///   different `__global__`s ran at all. Both deleted; what is left is a
///   device attribute and a named constant. **The file itself is now deleted
///   too** — kernels to `kernels-cuda-new/csrc/src/gemm/gemv.cuh`, launcher
///   to `driver-cuda/src/fire/gemv.rs` — so this test no longer checks it and
///   the loop below says where the two replacements went.
/// * `attn/attention_flashinfer_hopper.cu` —
///   `PIE_CUDA_DISABLE_HOPPER_EXTENDED` answered a CAPABILITY per launch.
///   Folded to a constant that names its default, `true`, which is exactly
///   what an unset variable meant. **The file itself is now deleted too** —
///   392 lines of FA3/sm_90 prefill funnel, unreachable by call graph and not
///   merely by a flag — so this test no longer checks it, and the capability
///   is prose on `driver-cuda/src/fire/flashinfer_fa2_dispatch.rs`'s
///   `Decline::Sm90Unported` rather than a constant anywhere. The loop below
///   says so at the name's old position.
/// * `comm/custom_all_reduce.cu` — three `getenv` helpers that took the
///   variable's name as a parameter and that nothing called. **The file
///   itself is now deleted too** — 664 lines with zero `__global__` and zero
///   `<<<>>>`, a host program wearing a `.cu` extension, now
///   `driver-cuda/src/fire/all_reduce.rs` — so this test no longer checks it.
///   There was no finding to carry: three uncalled helpers reading nothing.
///
/// It says nothing about the rest of the tree. `ssm/nemotron_h.cu`,
/// `quant/dequant_fp4.cu` and `kernels_manifest.hpp` still read their own
/// variables; each is its own measurement and none is claimed here.
///
/// `gemm/gemm.cpp`, `tensor.cpp` and `tuning_cache.hpp` were in that sentence
/// and ALL THREE ARE DELETED. `gemm.cpp` was a 2,216-line host program with
/// zero `__global__` and zero `<<<>>>` — the sixth file this migration found
/// wearing a CUDA extension for linkage reasons — and it is
/// `driver-cuda/src/fire/gemm.rs` now. `tensor.cpp` and `tuning_cache.hpp`
/// (with `cache_root.hpp`) were its dependents and had no other includer.
/// Their variables are not unaudited, they are OUT OF SCOPE, the same verdict
/// `gemm/gemv.cu` got above: `PIE_GEMM_TUNE_LOG` and `PIE_GEMM_PATH_TRACE`
/// are `fire::gemm`'s `tune_log()` and `path_trace_take()`, both still
/// LOGGING-ONLY reads that pick no kernel, which is why they moved intact
/// rather than being deleted the way a selector would be.
///
/// `moe/flashinfer_moe.cu` was in that sentence and its three `getenv` sites
/// are GONE, by the route §36 keeps recommending and this is the first file
/// to take whole: the 817-line host program moved to
/// `driver-cuda/src/fire/flashinfer_moe.rs` and what is left here is an
/// `extern "C"` seam that instantiates CUTLASS templates and decides
/// nothing. The three sites were `env_truthy` (a truthiness parser that
/// reads no environment and that NOTHING CALLED — deleted, the third such
/// find after `comm/custom_all_reduce.cu`'s three) and `env_int` +
/// `fused_window_overridden` reading `PIE_MOE_FUSED_MAX_ROWS` and
/// `PIE_MOE_FUSED_MIN_ROWS`. Those two are the more interesting verdict:
/// they were policy that was OFF BY CONSTRUCTION, because the launcher
/// applied the row window only `if (fused_window_overridden())` — that is,
/// only when one of the variables was actually set — so on every run anyone
/// ever measured, BOTH comparisons were dead code. §36's tally of knobs whose
/// arms agree gains an entry that did not even have two arms. The defaults
/// survive as `fire::flashinfer_moe`'s `max_rows()` = 1024 and `min_rows()` =
/// 0 with `WINDOW: Option<RowWindow> = None`, so the unenforced state is
/// still the state and the numbers are still readable. §36's rule that a
/// deletion may not consume an open question is honoured there: the standing
/// question — the window was never swept against a measurement, so 1024 is
/// undefended — is carried verbatim in `WINDOW`'s doc comment.
///
/// `moe/topk_softmax.cu` was in that sentence and is DELETED. Its variable
/// was `PIE_TOPK_WARP`, which forced the block form for A/B measurement, and
/// it goes the way §30's `PIE_QWEN35_GDN_SMEM_STEP` went — deleted, not
/// moved, because a `getenv` may not pick a kernel. What the A/B MEASURED is
/// kept: 7.56 us/call, 4.9% of Qwen3.6-35B-A3B's step, on
/// `families::moe`'s `topk_softmax` row and in `new-horizon.md` §52.
#[test]
fn the_audited_launchers_read_no_environment_variable() {
    for rel in [
        // `gemm/gemv.cu` was the first name in this list and is DELETED: its
        // two `__global__` templates are `kernels-cuda-new/csrc/src/gemm/
        // gemv.cuh` (NVRTC compiles them) and its host launcher is
        // `driver-cuda/src/fire/gemv.rs`. Both §36 findings went with it and
        // NEITHER was dropped — `PIE_GEMV_SPLITK_MAX_ROWS` is
        // `fire::gemv::SPLIT_K_MAX_ROWS`, still 4096, and
        // `PIE_GEMV_B200_TUNING`'s replacement is
        // `fire::gemv::unroll_depth`, which asks
        // `device::Device::compute_capability` and nothing else.
        //
        // This test can no longer see either. It reads `csrc/src`, and the
        // guarantee now lives in a `.cuh` outside this crate and in Rust; the
        // Rust half carries the measurement (the 5-byte and 3-byte
        // wide-exponent disagreements) in its own doc comment so the next
        // reader meets it where the decision is made. Nothing here is a
        // weaker claim about the OTHER two files.
        // `attn/attention_flashinfer_hopper.cu` was the second name in this
        // list and is DELETED — 392 lines, the FA3/sm_90 prefill funnel — so
        // this test can no longer read it, and reading a deleted file is a
        // panic rather than a failure, which is why the name had to come out
        // rather than be left as documentation.
        //
        // ITS §36 FINDING IS NOT DROPPED, it is out of scope the way
        // `gemm/gemv.cu`'s two are, and it moved further than they did.
        // `PIE_CUDA_DISABLE_HOPPER_EXTENDED` became `constexpr bool
        // kHopperExtendedShapes = true` in that file, and now it is prose on
        // `driver-cuda/src/fire/flashinfer_fa2_dispatch.rs`'s
        // `Decline::Sm90Unported` — the refusal that stands where the file
        // stood. Everything the pin below used to hold is stated there:
        // that `true` is exactly `getenv(...) == nullptr` because nothing in
        // this repository ever set the variable; that a capability answered
        // per launch is answered too late to be refusable; that the
        // `set_hopper_extended_shapes(bool)` seam was written and WITHDRAWN
        // for having no caller; and the one real run the predicate was worth
        // (gemma-4-26B-A4B at 1k context, attention 4.19 ms -> 2.73 ms,
        // 122.5 -> 144.1 tok/s, output unchanged).
        //
        // The `.cpp` stub below is what is left of that pair on disk, and it
        // takes the loop's place rather than emptying it: a `for` over an
        // empty array is a test that passes by having nothing to check.
        "attn/attention_flashinfer_hopper_stub.cpp",
        // `comm/custom_all_reduce.cu` was the second name here and is
        // DELETED. Its three `getenv` helpers took the variable's name as a
        // parameter and NOTHING CALLED THEM, so there was no finding to
        // carry across -- unlike `gemv.cu`'s two, which both survive as Rust
        // constants. `driver-cuda/src/fire/all_reduce.rs` reads no
        // environment variable at all, and the tuning data the file did
        // carry (`kMaxBlocks = 36`, `threads = 512`, the 2 MiB fusion
        // alignment, the 256 barrier flags, the Lamport cap and the NCCL
        // crossover ladder) is stated there as named constants with their
        // derivations, not as knobs.
    ] {
        let text = read_src(rel);
        let offenders = env_reads(&text);
        assert!(
            offenders.is_empty(),
            "{rel} reads an environment variable in CODE:\n{}\n\
             A launcher that picks a kernel on `getenv` makes the same trace on \
             the same weights on the same GPU run different text, with nothing \
             recording which. If a configuration fact really must reach this \
             decision it arrives as a fact the driver holds at load, never as a \
             call inside a `.cu`.",
            offenders.join("\n")
        );
        assert!(
            !text.contains("#include <cstdlib>"),
            "{rel} included <cstdlib> for its `std::getenv`; the include coming \
             back is the selector coming back"
        );
    }

    // What replaced them, pinned like any other transcription. A tuned
    // constant that names its default, and a device attribute — which is a
    // different kind of input from an environment variable and is allowed to
    // stay: same machine, same answer, and any backend can ask it.
    //
    // TWO PINS WERE HERE AND ARE GONE WITH THEIR FILE: `gemm/gemv.cu:316`'s
    // `constexpr int kSplitKMaxRows = 4096;` and `:156`'s
    // `cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, ...)`.
    // They are not unpinned, they are OUT OF SCOPE: the first is
    // `driver-cuda/src/fire/gemv.rs`'s `SPLIT_K_MAX_ROWS`, the second is that
    // module's `unroll_depth()` calling `device::Device::compute_capability`,
    // and a test that walks `csrc/src` cannot pin a line in another crate's
    // Rust. Re-pinning them wants a test that lives beside them; this one
    // deliberately does not grow a second root to reach across.

    // THE HOPPER CAPABILITY PIN WAS HERE AND IS GONE WITH ITS FILE. It read
    //
    //     pinned(&hopper, "attn/attention_flashinfer_hopper.cu", 179,
    //            "constexpr bool kHopperExtendedShapes = true;", ...)
    //
    // plus a check that `if (!hopper_extended_shapes_enabled())` still asked
    // it, on the rule that a constant nothing reads binds nothing. Both are
    // unpinnable now for the same reason the two `gemv.cu` pins above are:
    // the subject is not in this crate. The constant did not become a Rust
    // constant either — it became an ARGUMENT, on
    // `fire::flashinfer_fa2_dispatch::Decline::Sm90Unported`, because the
    // capability it answered has no live caller to answer for. Re-pinning it
    // wants a test that lives beside that enum; this one deliberately does
    // not grow a second root to reach across.
    //
    // The stub `.cpp`'s `env_reads` check is NOT gone: it moved into the loop
    // above, where it is now the only live member. It was asserted twice
    // otherwise, and a duplicated assertion is a second census.
}

// `the_graph_stats_environment_read_only_prints` WAS HERE AND IS DELETED,
// WITH THE FILE IT READ.
//
// It read `attn/attention_flashinfer.cu` through `read_src`, which joins
// `csrc/src` — and that file left this crate for `driver-cuda/csrc/attn/` in
// an earlier pass without the test following it, so the `read_src` `expect`
// had been panicking on a missing path rather than checking anything. The
// file is now deleted outright, with the whole of `driver-cuda/csrc/`, so
// there is no path to repoint it at: it can never pass again.
//
// WHAT IT ASSERTED, so that nothing is lost by the deletion. Exactly one
// `getenv` in the file; it names `PIE_GRAPH_STATS`; its brace-matched block
// contains a `fprintf` and does NOT touch `enable_cuda_graph`. The rule
// behind it is the one worth keeping and it is general, not this file's: a
// diagnostic that only writes to stderr changes no kernel, no grid and no
// byte, so it is not a selector; the moment it touches the decision it
// reports on, it is one, and §30/§36 removed selectors read from the
// environment everywhere in this tree.
//
// WHERE THE DECISION WENT. `enable_cuda_graph`'s demotion is
// `driver-cuda/src/fire/flashinfer_fa2.rs::plan_prefill`, in Rust, and it
// reads no environment variable at all — the graph-mode retry is a plan
// failure re-planned eagerly, not a knob. So the property this test guarded
// is now enforced by there being nothing to enforce it against, which is
// the same way `comm/custom_all_reduce.cu`'s three `getenv` helpers left.
//
// The sibling above, `no_environment_variable_selects_a_kernel`, is
// untouched and still walks its own list; it never named this file.

/// Every arm of the two guards above, fired — each must FAIL.
///
/// A check that cannot fail is not a check. The comment/code split in
/// particular is one `starts_with` away from passing everything: every one of
/// these files NAMES its deleted variable in prose, so a guard that stopped
/// filtering comments would flag all of them, and a guard that filtered too
/// much would flag none. Both directions are fired here.
#[test]
fn the_environment_guard_fails_when_a_selector_comes_back() {
    // 1. The scanner sees code and not comments.
    let prose = "// std::getenv(\"PIE_GEMV_B200_TUNING\") used to be read here\n\
                 int depth = 4;\n";
    assert!(
        env_reads(prose).is_empty(),
        "a comment naming the deleted variable must not be an offence — every \
         audited file names its own"
    );
    let code = "// the knob is gone\n    const char* v = std::getenv(\"PIE_X\");\n";
    assert_eq!(
        env_reads(code).len(),
        1,
        "a real `getenv` on a code line must be caught, or the guard passes \
         everything"
    );

    // 2. `pinned` fails when the text is gone, and when it has MOVED.
    let src = "alpha\nbeta\ngamma\n";
    let rewritten = std::panic::catch_unwind(|| pinned(src, "synthetic.cu", 2, "delta", "`nothing`"));
    assert!(rewritten.is_err(), "a rewritten launcher must fail the pin");
    let moved = std::panic::catch_unwind(|| pinned(src, "synthetic.cu", 3, "beta", "`nothing`"));
    assert!(moved.is_err(), "a moved line must fail the pin");
    let intact = std::panic::catch_unwind(|| pinned(src, "synthetic.cu", 2, "beta", "`nothing`"));
    assert!(intact.is_ok(), "an intact citation must PASS, or the pin is a tautology");

    // 3. The telemetry guard is not a rubber stamp: a block that decides
    //    something must fail it. This is the mutant of the real code.
    let mutant = "\
    if (const char* const env = std::getenv(\"PIE_GRAPH_STATS\");
        env != nullptr) {
        enable_cuda_graph = false;
        std::fprintf(stderr, \"demoted\\n\");
    }
";
    let lines: Vec<&str> = mutant.lines().collect();
    let start = lines
        .iter()
        .position(|l| !l.trim_start().starts_with("//") && l.contains("getenv"))
        .expect("the mutant reads the environment");
    let mut depth = 0i32;
    let mut end = start;
    let mut opened = false;
    for (i, line) in lines.iter().enumerate().skip(start) {
        depth += line.matches('{').count() as i32;
        depth -= line.matches('}').count() as i32;
        if line.contains('{') {
            opened = true;
        }
        if opened && depth <= 0 {
            end = i;
            break;
        }
    }
    assert!(
        lines[start..=end].iter().any(|l| l.contains("enable_cuda_graph")),
        "the brace matcher must find the assignment inside the mutant's block, \
         or `the_graph_stats_environment_read_only_prints` is checking an empty \
         range and would pass a diagnostic that had become a selector"
    );
}

// ---------------------------------------------------------------------------
// The dangling-forwarder guard (`new-horizon.md` §57.8)
// ---------------------------------------------------------------------------

/// Every `pie_k_*` the shim emits forwards onto a launcher that EXISTS.
///
/// # The failure this exists for
///
/// `abi::emit_c_shim` writes, for each row it does not skip, a body reading
/// `static R (*const fwd)(...) = &::pie_cuda_driver::kernels::<symbol>;`.
/// That is a definition reference: if the launcher is gone, the generated
/// `shim.cpp` does not compile, and it does not fail at the row — it fails
/// with a wall of *"`x` is not a member of `pie_cuda_driver::kernels::y`"*
/// naming a generated file nobody wrote, in an order that says nothing about
/// which deletion caused which line. It is the one breakage in this migration
/// that stops the whole workspace rather than one crate.
///
/// Deleting a launcher is therefore never finished until the row it served is
/// closed, and there are exactly three ways to close one:
///
/// 1. **`device::JIT_DISPATCHED`** — NVRTC compiles the kernel and
///    `bind::jit::fire` launches it. Only legal when EVERY operand states a
///    `Source`; an unsourced operand makes `emit_rust_dispatch` skip the row
///    whole, so it gets no arm of either kind while the shim entry is dropped
///    anyway, and the fire fails at LINK time (§22.1).
/// 2. **`execution::RUST_SERVED`** — the host program is Rust now, in
///    `driver-cuda/src/{fire,bind}`. This is the door `gemm.cpp` needed and
///    the one `ssm/` used for eleven rows.
/// 3. **Delete the row, or take its `operands` off.** An unstated row gets no
///    entry at all; six rows are deliberately in that state.
///
/// # Why this is a test and not a script
///
/// A scan written from the *description* of the emitter drifts from the
/// emitter. §57.8's false alarm is the worked example: a check carrying the
/// pre-`gemm.cpp` rule — "stated, and not in `JIT_DISPATCHED`" — reported 36
/// dangling forwarders, which was precisely `RUST_SERVED`'s membership, every
/// one of them already closed by `abi.rs`'s second filter. Acting on any of
/// the three remedies above would have destroyed 36 working dispatch arms.
///
/// So this calls `emit_c_shim` itself and reads the text it produced. It
/// cannot describe the filter wrongly, because it does not describe it.
///
/// A text scan over a generated string: no GPU, no toolkit, no CMake, and no
/// build of the archive it is protecting.
#[test]
fn every_emitted_shim_entry_has_a_launcher() {
    // `table::driver_internal::DRIVER_KERNELS` was pushed here too, and is
    // gone with that module: §5 step 5 made its six launchers `fn`s with no
    // `contract!`, so no shim entry is emitted for them and none is wanted.
    let tables = kernels_cuda_new::table::TABLES.to_vec();
    // `build.rs::shim` builds this list the same way, and the include list is
    // deliberately empty: `includes` only chooses `#include` lines, never
    // which rows get a body, so an empty one asks the emitter exactly the
    // question this test has.
    let shim = kernels_cuda_new::abi::emit_c_shim(
        &tables,
        &[],
        &kernels_cuda_new::device::jit_dispatched(),
    )
    .expect("two rows may not claim one entry point");

    // `&::pie_cuda_driver::kernels::attn::foo;` -> `attn::foo`
    const MARK: &str = "&::pie_cuda_driver::kernels::";
    let wanted: Vec<String> = shim
        .match_indices(MARK)
        .map(|(i, _)| {
            shim[i + MARK.len()..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
                .collect()
        })
        .collect();
    assert!(
        wanted.len() > 40,
        "the shim forwards onto {} launchers, which is too few for the emitter \
         to have run — the marker `{MARK}` has probably changed shape in \
         `abi::cpp_path`, and this test would then pass by finding nothing.",
        wanted.len()
    );

    // Every C++ the archive and the two driver-side trees hold. A launcher
    // may be DECLARED in an `.hpp` and DEFINED in a `.cu`; either proves the
    // name resolves, and only the compiler can tell them apart.
    //
    // `third_party` and `vendor` are skipped for `sources_with`'s reason —
    // the island stays the island (§5) — and for a second one here: CMake
    // FETCHES FlashInfer and CUTLASS into those paths, so on a machine that
    // has built once they are the largest thing in the tree, and this test
    // must cost the same on both.
    let mut cpp = String::new();
    for root in [
        csrc().join("src"),
        csrc().join("../../driver-cuda/csrc"),
        csrc().join("../../kernels-cuda-new/csrc/src"),
    ] {
        for path in walk(&root) {
            if path
                .components()
                .any(|c| matches!(c.as_os_str().to_str(), Some("third_party") | Some("vendor")))
            {
                continue;
            }
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "cu" | "cpp" | "cc" | "hpp" | "cuh" | "h") {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    cpp.push_str(&text);
                    cpp.push('\n');
                }
            }
        }
    }

    let mut dangling: Vec<&str> = Vec::new();
    for symbol in &wanted {
        let bare = symbol.rsplit("::").next().unwrap_or(symbol);
        // The declaration this is looking for is `<name>(` — the launcher's
        // own, in its `.hpp`, or its definition in the `.cu`. A call site
        // spells the same thing, which is why a HIT is weak evidence and a
        // MISS is strong: nothing anywhere names it, so `&...::<name>` cannot
        // resolve.
        let found = cpp.match_indices(bare).any(|(i, _)| {
            let before_ok = i == 0
                || !cpp[..i]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_');
            let after = cpp[i + bare.len()..].trim_start();
            before_ok && after.starts_with('(')
        });
        if !found {
            dangling.push(symbol);
        }
    }

    assert!(
        dangling.is_empty(),
        "{} shim entries forward onto a launcher that no longer exists \
         anywhere in C++:\n{}\n\n\
         Each one is a row whose launcher was deleted and whose row was not \
         closed. Close it the way the doc above says — `JIT_DISPATCHED` if \
         every operand sources, `RUST_SERVED` if the host program is Rust \
         now, or take the row's `operands` off if neither is true yet — and \
         do it in the SAME edit as the deletion. Leaving it breaks the \
         generated `shim.cpp`, which is every crate's build and not just \
         this one's.",
        dangling.len(),
        dangling
            .iter()
            .map(|s| format!("  {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
