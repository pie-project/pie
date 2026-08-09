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
    //       LAUNCHER: `gemv_bf16` is the file's whole live surface and keeps
    //       all four of its launches. The sweeps' harness,
    //       `driver/cuda/bench/gemv_bench.cu`, is in no source directory of
    //       this repository. `gemm/gemm.cpp` contributes nothing here and
    //       never did -- it holds 0 `<<<>>>` -- so the ~380 lines §45 moved
    //       out of it into Rust are invisible to this count, which is the
    //       point the section makes.
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
    // `layout`, `mlp`, `moe`, `norm` and `quant` still hold deletions that
    // were measured, written down in `new-horizon.md` §43, and lost in the
    // merge before they could be committed.
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
    // `kernels-cuda-new/csrc/src/attn/{softcap,attn_sink,attn_res}.cuh`,
    // which `families/attn.rs` `include_str!`s and NVRTC compiles.
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
    const EXPECTED: usize = 201;
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
///   device attribute and a named constant.
/// * `attn/attention_flashinfer_hopper.cu` —
///   `PIE_CUDA_DISABLE_HOPPER_EXTENDED` answered a CAPABILITY per launch.
///   Folded to a constant that names its default, `true`, which is exactly
///   what an unset variable meant.
/// * `comm/custom_all_reduce.cu` — three `getenv` helpers that took the
///   variable's name as a parameter and that nothing called.
///
/// It says nothing about the rest of the tree. `ssm/nemotron_h.cu`,
/// `moe/flashinfer_moe.cu`, `moe/topk_softmax.cu`,
/// `quant/dequant_fp4.cu`, `gemm/gemm.cpp`, `kernels_manifest.hpp`,
/// `tensor.cpp` and `tuning_cache.hpp` still read their own variables; each
/// is its own measurement and none is claimed here.
#[test]
fn the_audited_launchers_read_no_environment_variable() {
    for rel in [
        "gemm/gemv.cu",
        "attn/attention_flashinfer_hopper.cu",
        "comm/custom_all_reduce.cu",
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
    let gemv = read_src("gemm/gemv.cu");
    pinned(
        &gemv,
        "gemm/gemv.cu",
        // 409 until §45 deleted the file's four unreachable launchers and the
        // `__global__` two of them shared. The constant did not move relative
        // to the code that reads it; 232 lines above it went away.
        316,
        "constexpr int kSplitKMaxRows = 4096;",
        "`the deleted PIE_GEMV_SPLITK_MAX_ROWS`",
    );
    pinned(
        &gemv,
        "gemm/gemv.cu",
        // 155 until §45; the note replacing the `<type_traits>` the deleted
        // sweep entry points needed is one line longer than the include was.
        156,
        "if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,",
        "`the deleted PIE_GEMV_B200_TUNING`",
    );

    // The capability that replaced the Hopper read: a constant that names its
    // default, pinned, and still consulted. A constant nothing reads is a
    // fact nothing is bound by, so the call site is checked too.
    let hopper = read_src("attn/attention_flashinfer_hopper.cu");
    pinned(
        &hopper,
        "attn/attention_flashinfer_hopper.cu",
        179,
        "constexpr bool kHopperExtendedShapes = true;",
        "`the deleted PIE_CUDA_DISABLE_HOPPER_EXTENDED`",
    );
    assert!(
        code_lines(&hopper)
            .any(|(_, l)| l.contains("if (!hopper_extended_shapes_enabled())")),
        "attn/attention_flashinfer_hopper.cu states the capability but no \
         longer asks it; a default nothing reads binds nothing"
    );
    assert!(
        env_reads(&read_src("attn/attention_flashinfer_hopper_stub.cpp")).is_empty(),
        "the hopper stub must not read the environment either"
    );
}

/// `PIE_GRAPH_STATS` prints, and decides nothing.
///
/// This one is NOT deleted, and the audit that deleted the other five says so
/// on purpose: a diagnostic that only writes to stderr changes no kernel, no
/// grid and no byte, so it is not a selector and there is nothing to make
/// reproducible. What would make it one is a line inside its block that
/// touched the decision it reports on — and the decision is right there,
/// `enable_cuda_graph = false`, six lines above.
///
/// So the guard is not "no `getenv` in this file". It is: exactly one, it
/// names `PIE_GRAPH_STATS`, and its block assigns nothing.
#[test]
fn the_graph_stats_environment_read_only_prints() {
    let rel = "attn/attention_flashinfer.cu";
    let text = read_src(rel);
    let reads = env_reads(&text);
    assert_eq!(
        reads.len(),
        1,
        "{rel} should read the environment exactly once — the `PIE_GRAPH_STATS` \
         diagnostic — and reads it {} times:\n{}",
        reads.len(),
        reads.join("\n")
    );
    assert!(
        reads[0].contains("PIE_GRAPH_STATS"),
        "{rel}'s one environment read is no longer the graph-stats diagnostic:\n{}",
        reads[0]
    );

    // Brace-match the guarded block and assert it only prints. `enable_cuda_graph`
    // is the demotion this block REPORTS; the moment it also performs it, a
    // diagnostic has become a selector.
    let lines: Vec<&str> = text.lines().collect();
    let start = lines
        .iter()
        .position(|l| !l.trim_start().starts_with("//") && l.contains("getenv"))
        .expect("the read was just found");
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
    assert!(opened && end > start, "the graph-stats block did not parse");
    let body = &lines[start..=end];
    for (offset, line) in body.iter().enumerate() {
        let t = line.trim_start();
        if t.starts_with("//") {
            continue;
        }
        assert!(
            !line.contains("enable_cuda_graph"),
            "{rel}:{} — the `PIE_GRAPH_STATS` block touches `enable_cuda_graph`:\n  {}\n\
             It is allowed to exist because it only writes to stderr. A \
             diagnostic that decides anything is a selector, and a selector \
             read from the environment is what §30 and §36 removed everywhere \
             else in this tree.",
            start + offset + 1,
            line.trim()
        );
    }
    assert!(
        body.iter().any(|l| l.contains("fprintf")),
        "{rel}'s graph-stats block no longer prints; if it stopped being a \
         diagnostic, it stopped being exempt"
    );
}

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
