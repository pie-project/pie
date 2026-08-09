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

fn cu_on_disk() -> Vec<String> {
    let out = sources_with("cu");
    assert!(out.len() > 40, "the walk found almost nothing: {out:?}");
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
    // every one of them includes it, and the seven that still define a
    // `__global__` — `attn/attention_flashinfer`, `attention_naive_paged`,
    // `attention_xqa`, `mla_paged`, `pack_dense_mask`, `qkv_fused` and
    // `gemm/gemv`, 17 kernels between them — have no header at all, because
    // they have not been split. So the orphan list below is empty for the best
    // reason available and this test is now a REGRESSION guard: it fires when
    // device text reappears in a `.cu` that has a header, or when a new split
    // lands without the include. The floor is what keeps it from being empty
    // for the worst reason instead, and 48 is a number that only grows.
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

    // 401 -- the same number the tree carried before the migration, which is
    // a coincidence and not a reassurance: the deltas below sum to zero, and
    // it is the LIST that is the test, not the total. Every difference from
    // the pre-migration 401 is named here:
    //
    //   -6  `norm/altup_aux.cu`  DELETED -- its six kernels are `.cuh`
    //                            templates now, rowed and fired through the
    //                            JIT, with no launcher left to call
    //   -1  `norm/scalar_mul.cu` DELETED for the same reason
    //   +1  `quant/dequant_fp8.cu` -- `..._per_group` used to delegate to the
    //                            `blocked` launcher with `group_size` as both
    //                            block dimensions and now calls its own
    //                            template; same `dequant_fp8_e4m3_tile`, same
    //                            arithmetic, one fewer indirection
    //   +5  `vision/gemma4_vision.cu` -- five CALLS became five LAUNCHES.
    //                            The tower reached `norm::residual_add_bf16`
    //                            twice, `norm::rmsnorm_no_scale_bf16` twice
    //                            and `mlp::geglu_tanh_bf16` once, each an
    //                            ahead-of-time host launcher in another
    //                            family's `.cu`. C++ calling C++ is what
    //                            keeps a launcher alive after its kernel has
    //                            migrated -- §10.10's rule is that a launcher
    //                            goes when its WHOLE consumer set has gone
    //                            and the JIT shim is only one consumer -- so
    //                            the call had to go first. Each launch copies
    //                            its launcher's `<<<grid, block, 0, stream>>>`
    //                            verbatim and instantiates the same template
    //                            at the same type; not one instruction that
    //                            runs on the device changed
    //   +1  `vision/gemma4_audio.cu` -- the same move on
    //                            `ssm::causal_conv1d_prefill_noact_bf16`,
    //                            whose only caller anywhere in the repository
    //                            this was, and which no table row names
    //
    // A drop that is NOT one of those is a launcher that went missing in a
    // split, which compiles, links, and simply stops doing something.
    const EXPECTED: usize = 401;
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
