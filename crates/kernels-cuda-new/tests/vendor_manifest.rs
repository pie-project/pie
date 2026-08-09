//! `MODIFICATIONS` is a claim about the vendored tree. This checks it.
//!
//! # The claim, and why prose was not enough
//!
//! `csrc/vendor/MODIFICATIONS` opens by asserting a property rather than
//! describing one:
//!
//! > *"Generated from the tree, and kept honest by the property it asserts:
//! > strip every `// PIE:` marker, the `#ifndef __CUDACC_RTC__` under it and
//! > that directive's matching `#endif`, and each file below is byte-for-byte
//! > FlashInfer v0.6.15. That was checked per file before this list was
//! > written."*
//!
//! **"Was checked" is a past-tense fact about a person, not a property of the
//! tree.** Nothing re-checks it. A guard widened by one line, a stray edit
//! inside a `#ifndef __CUDACC_RTC__` block, or a row left stale after a file
//! changed all leave a `MODIFICATIONS` that reads correct and is not — and
//! the whole value of vendoring upstream *byte-identically* is that a reader
//! can diff against upstream and a bump is a re-fetch rather than a merge.
//!
//! # What this test can check without a network, and what it cannot
//!
//! Byte-identity against upstream needs upstream, and a unit test must not
//! fetch. So the property is split, and this file takes the half that is a
//! function of the tree alone — the three columns:
//!
//! * `guards` — the count of `// PIE:` markers in the file;
//! * `lines` — the file's line count;
//! * `added` — how many lines the strip removes, which is what the guards
//!   cost over upstream.
//!
//! Every one of those is recomputable here, and a drift that changes the
//! vendored bytes changes at least one of them unless it is *exactly*
//! line-count-neutral outside every guard. That residue is what the
//! network-side check covers, and `.wiki/driver/new-horizon.md` §23.8 records
//! it being run: **28 of 28 vendored FlashInfer files were byte-identical to
//! v0.6.15 after stripping**, with two negative controls proving the checker
//! can fail.
//!
//! So: this test catches the drift a maintainer actually causes (edit a
//! guard, forget the row), and the doc carries the one-time proof of the part
//! that needs the internet.
//!
//! # Why the stripper lives in the test rather than the build
//!
//! Nothing in the crate needs it at run time. It exists to answer a question
//! about the repository, which is what a test is for.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn csrc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc")
}

/// One row of `MODIFICATIONS`: the three numbers it asserts about a file.
#[derive(Debug, PartialEq, Eq)]
struct Row {
    guards: usize,
    added: usize,
    lines: usize,
}

/// Parse the table. The format is fixed-ish columns under a `---` rule, and
/// the parse is deliberately strict: a row this cannot read is a row the test
/// would otherwise skip silently, which is how a manifest check goes vacuous.
fn manifest() -> BTreeMap<String, Row> {
    let text = std::fs::read_to_string(csrc().join("vendor/MODIFICATIONS"))
        .expect("csrc/vendor/MODIFICATIONS");
    let mut out = BTreeMap::new();
    let mut in_table = false;
    for line in text.lines() {
        if line.starts_with("---") {
            in_table = true;
            continue;
        }
        if !in_table || line.trim().is_empty() {
            continue;
        }
        // The table ends with its own totals row and then prose. Both are
        // stopped on here; the totals get their own assertion below, because
        // a summary line that drifts from the rows above it is exactly the
        // kind of thing a reader trusts without checking.
        if line.starts_with(char::is_numeric) || !line.starts_with(|c: char| c.is_ascii_graphic())
        {
            in_table = false;
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(file), Some(g), Some(a), Some(l)) =
            (it.next(), it.next(), it.next(), it.next())
        else {
            continue;
        };
        let (Ok(guards), Ok(added), Ok(lines)) =
            (g.parse::<usize>(), a.parse::<usize>(), l.parse::<usize>())
        else {
            panic!("MODIFICATIONS row is not three numbers: {line:?}");
        };
        out.insert(file.to_string(), Row { guards, added, lines });
    }
    assert!(
        out.len() > 20,
        "parsed only {} rows from MODIFICATIONS — the format moved and this \
         test would have passed vacuously",
        out.len()
    );
    out
}

/// The strip `MODIFICATIONS` describes: drop a `// PIE:` marker with its
/// continuation comments, and if a `#ifndef __CUDACC_RTC__` follows, drop that
/// directive and its matching `#endif` while KEEPING the body.
///
/// Keeping the body is the point. The guards exist to hide host-only code
/// from NVRTC, so the body is upstream's and must survive the strip; only the
/// three lines this tree added come out.
fn strip(text: &str) -> (String, usize) {
    let src: Vec<&str> = text.lines().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut removed = 0usize;
    let mut i = 0usize;
    while i < src.len() {
        let t = src[i].trim_start();
        if !t.starts_with("// PIE:") {
            out.push(src[i]);
            i += 1;
            continue;
        }
        // the marker and its continuation comment lines
        i += 1;
        removed += 1;
        while i < src.len() {
            let c = src[i].trim_start();
            if c.starts_with("//") && !c.starts_with("// PIE:") {
                i += 1;
                removed += 1;
            } else {
                break;
            }
        }
        // the guard under it, if there is one
        let is_open = |s: &str| {
            let s = s.trim_start();
            s.starts_with("#ifndef __CUDACC_RTC__")
                || s.starts_with("#if !defined(__CUDACC_RTC__)")
        };
        if i < src.len() && is_open(src[i]) {
            i += 1;
            removed += 1;
            let mut depth = 1usize;
            while i < src.len() && depth > 0 {
                let d = src[i].trim_start();
                if d.starts_with("#if") {
                    depth += 1;
                } else if d.starts_with("#endif") {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        removed += 1;
                        break;
                    }
                }
                out.push(src[i]);
                i += 1;
            }
        }
    }
    (out.join("\n"), removed)
}

/// Every vendored FlashInfer file, as `MODIFICATIONS` names it.
fn vendored() -> Vec<(String, PathBuf)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, PathBuf)>) {
        for e in std::fs::read_dir(dir).expect("read vendor dir").flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else if p.file_name().is_some_and(|n| n != "LICENSE") {
                let rel = p.strip_prefix(root).expect("under root");
                out.push((rel.to_string_lossy().replace('\\', "/"), p.clone()));
            }
        }
    }
    let root = csrc().join("vendor/flashinfer");
    let mut v = Vec::new();
    walk(&root, &root, &mut v);
    v.sort();
    v
}

/// The manifest describes the tree it is in.
///
/// Checks all three columns against the files, and both directions of the set
/// — a vendored file with no row, and a row with no file, are each a way for
/// the manifest to stop being a description.
#[test]
fn modifications_describes_the_vendored_tree() {
    let rows = manifest();
    let files = vendored();
    assert!(!files.is_empty(), "no vendored FlashInfer files found");

    let mut wrong = Vec::new();
    for (name, path) in &files {
        let text = std::fs::read_to_string(path).expect("read vendored file");
        let guards = text.lines().filter(|l| l.trim_start().starts_with("// PIE:")).count();
        let lines = text.lines().count();
        let (_, added) = strip(&text);

        let Some(row) = rows.get(name) else {
            wrong.push(format!("{name}: vendored but absent from MODIFICATIONS"));
            continue;
        };
        let got = Row { guards, added, lines };
        if got != *row {
            wrong.push(format!(
                "{name}: MODIFICATIONS says guards={} added={} lines={}, file has \
                 guards={} added={} lines={}",
                row.guards, row.added, row.lines, got.guards, got.added, got.lines
            ));
        }
    }
    for name in rows.keys() {
        if !files.iter().any(|(n, _)| n == name) {
            wrong.push(format!("{name}: in MODIFICATIONS but not vendored"));
        }
    }

    assert!(
        wrong.is_empty(),
        "MODIFICATIONS no longer describes csrc/vendor/flashinfer:\n  {}",
        wrong.join("\n  ")
    );
}

/// The table's own totals row agrees with the rows above it.
///
/// `28 files   33   206   18187` is a summary, and a summary is the line a
/// reader takes on trust. It is also the line that survives a per-file edit
/// unchanged, so it is the one most likely to go stale.
#[test]
fn the_totals_row_sums_the_table() {
    let text = std::fs::read_to_string(csrc().join("vendor/MODIFICATIONS")).expect("manifest");
    let totals = text
        .lines()
        .find(|l| l.split_whitespace().next().is_some_and(|w| w.parse::<usize>().is_ok()))
        .expect("MODIFICATIONS has a totals row");
    let n: Vec<usize> = totals
        .split_whitespace()
        .filter_map(|w| w.parse().ok())
        .collect();
    assert_eq!(n.len(), 4, "totals row is not four numbers: {totals:?}");

    let rows = manifest();
    let files = n[0];
    let (guards, added, lines) = rows.values().fold((0, 0, 0), |(g, a, l), r| {
        (g + r.guards, a + r.added, l + r.lines)
    });
    assert_eq!(
        (files, n[1], n[2], n[3]),
        (rows.len(), guards, added, lines),
        "the totals row disagrees with the rows it summarises"
    );
}

/// The marker total the manifest's own prose quotes.
///
/// Separate from the per-row check because it is the number a reader greps
/// for — `grep -rn "// PIE:" csrc/vendor/flashinfer/` — and a prose number
/// nothing checks is the thing this file exists about.
#[test]
fn the_quoted_marker_total_is_the_real_one() {
    let text = std::fs::read_to_string(csrc().join("vendor/MODIFICATIONS")).expect("manifest");
    let quoted: usize = text
        .split_whitespace()
        .zip(text.split_whitespace().skip(1))
        .find_map(|(a, b)| (b == "markers.").then(|| a.parse().ok())?)
        .expect("MODIFICATIONS quotes a marker total");

    let actual: usize = vendored()
        .iter()
        .map(|(_, p)| {
            std::fs::read_to_string(p)
                .expect("read")
                .lines()
                .filter(|l| l.trim_start().starts_with("// PIE:"))
                .count()
        })
        .sum();

    assert_eq!(
        quoted, actual,
        "MODIFICATIONS says {quoted} markers, the tree has {actual}"
    );
}

/// The stripper can fail.
///
/// Without this the two tests above pass on a stripper that returned its
/// input. §20.11's rule, one level down: a check that cannot fail has not
/// checked anything — which this session learned by publishing two
/// conclusions from a probe that matched `SUCCESS` inside its own label.
#[test]
fn the_stripper_is_not_a_no_op() {
    let guarded = "\
before
// PIE: guarded for NVRTC -- host-only include
// continuation of the marker's explanation
#ifndef __CUDACC_RTC__
#include <iostream>
#endif
after";
    let (out, removed) = strip(guarded);
    assert_eq!(out, "before\n#include <iostream>\nafter", "body must survive");
    assert_eq!(removed, 4, "marker + continuation + ifndef + endif");

    let (same, none) = strip("nothing\nto strip\n");
    assert_eq!(same, "nothing\nto strip\n".trim_end());
    assert_eq!(none, 0, "a file with no markers loses no lines");

    // a nested #if inside the guard must not end it early
    let nested = "\
// PIE: guarded
#ifndef __CUDACC_RTC__
#if FOO
int x;
#endif
#endif
tail";
    let (out, removed) = strip(nested);
    assert_eq!(out, "#if FOO\nint x;\n#endif\ntail");
    assert_eq!(removed, 3);
}

/// The counted claims `csrc/src/cooperative_groups.h` makes about the closure
/// it replaces.
///
/// That header justifies its own existence with a census — *"`cg::this_thread_block()`
/// at seven sites (four in `decode.cuh`, three in `prefill.cuh`),
/// `block.sync()` at forty-nine, and `cg::this_grid()` at two"* — and the
/// census is the argument: a shim that answers a hand of call sites is
/// defensible where one answering an open-ended API is not. Nothing checked
/// it, and a FlashInfer bump moves every one of these numbers.
///
/// The `.sync()` figure is deliberately spelled as decode + prefill here,
/// because that is what forty-nine counts and the tree holds fifty. The
/// fiftieth is in `attention/mla.cuh`, which is vendored and which nothing
/// includes (`new-horizon.md` §23.7) — so the prose is right about the
/// reachable closure and short by one about the directory. Asserting both
/// numbers is what keeps that distinction from being re-derived.
#[test]
fn the_cooperative_groups_census_still_holds() {
    let dir = csrc().join("vendor/flashinfer");
    let read = |rel: &str| std::fs::read_to_string(dir.join(rel)).expect(rel);
    let count = |hay: &str, needle: &str| hay.matches(needle).count();

    let decode = read("attention/decode.cuh");
    let prefill = read("attention/prefill.cuh");
    let mla = read("attention/mla.cuh");

    assert_eq!(count(&decode, "this_thread_block()"), 4, "decode.cuh");
    assert_eq!(count(&prefill, "this_thread_block()"), 3, "prefill.cuh");
    assert_eq!(
        count(&decode, ".sync()") + count(&prefill, ".sync()"),
        49,
        "the forty-nine the header names are decode + prefill"
    );
    assert_eq!(count(&mla, ".sync()"), 1, "and mla.cuh, unreachable, holds the fiftieth");

    let all: usize = vendored()
        .iter()
        .map(|(_, p)| count(&std::fs::read_to_string(p).expect("read"), "this_grid()"))
        .sum();
    assert_eq!(all, 2, "cg::this_grid() sites across the vendored closure");

    // The three doors. If a fourth file learns to include it, the header's
    // "exactly four doors" argument needs rewriting before the shim does.
    let files = vendored();
    let doors: Vec<&str> = files
        .iter()
        .filter(|(_, p)| {
            std::fs::read_to_string(p)
                .expect("read")
                .contains("include <cooperative_groups.h>")
        })
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(
        doors,
        ["attention/decode.cuh", "attention/mla.cuh", "attention/prefill.cuh"],
        "the files that open the cooperative_groups door"
    );
}

/// Transitive includers of a vendored file, by the two forms FlashInfer uses.
fn transitive_includers(target: &str) -> Vec<String> {
    let files = vendored();
    let text: BTreeMap<String, String> = files
        .iter()
        .map(|(n, p)| (n.clone(), std::fs::read_to_string(p).expect("read")))
        .collect();
    let direct = |of: &str| -> Vec<String> {
        let src = &text[of];
        let dir = std::path::Path::new(of).parent().unwrap_or(std::path::Path::new(""));
        let mut out = Vec::new();
        for line in src.lines() {
            let t = line.trim_start();
            if let Some(rest) = t.strip_prefix("#include \"") {
                if let Some(rel) = rest.split('"').next() {
                    let joined = dir.join(rel).to_string_lossy().into_owned();
                    let mut norm: Vec<&str> = Vec::new();
                    for c in joined.split('/') {
                        match c {
                            "." | "" => {}
                            ".." => { norm.pop(); }
                            other => norm.push(other),
                        }
                    }
                    let key = norm.join("/");
                    if text.contains_key(&key) { out.push(key); }
                }
            } else if let Some(rest) = t.strip_prefix("#include <flashinfer/") {
                if let Some(rel) = rest.split('>').next() {
                    if text.contains_key(rel) { out.push(rel.to_string()); }
                }
            }
        }
        out
    };
    let mut reach: Vec<String> = Vec::new();
    loop {
        let before = reach.len();
        for (name, _) in &files {
            if reach.iter().any(|r| r == name) { continue; }
            let d = direct(name);
            if d.iter().any(|x| x == target || reach.iter().any(|r| r == x)) {
                reach.push(name.clone());
            }
        }
        if reach.len() == before { break; }
    }
    reach.sort();
    reach
}

/// `csrc/src/cuda_fp16.h` justifies an eight-line alias block with a cost.
///
/// Its claim is a causal chain rather than a count: *"One missing alias, one
/// file that uses it, seven files that include that file. An alias costs
/// nothing and its absence cost a quarter of the closure."* Each link is
/// checkable, and the middle one is what a FlashInfer bump moves.
#[test]
fn the_fp16_alias_cost_still_holds() {
    let users: Vec<String> = vendored()
        .iter()
        .filter(|(_, p)| std::fs::read_to_string(p).expect("read").contains("nv_half"))
        .map(|(n, _)| n.clone())
        .collect();
    assert_eq!(users, ["page.cuh"], "the one file that uses `nv_half`");

    let reachers = transitive_includers("page.cuh");
    assert_eq!(
        reachers.len(),
        7,
        "seven files reach page.cuh, and the alias's absence cost exactly them: {reachers:?}"
    );
}

/// `csrc/src/cuda_fp8.h` refuses the e8m0 family on a zero-use census.
///
/// The refusal is the interesting half — the header declines to implement
/// `__nv_fp8_e8m0` because nothing reaches it, and names the sites in the
/// wider FlashInfer tree that do. If the closure ever grows one, the refusal
/// needs revisiting before a compile finds out.
#[test]
fn the_fp8_e8m0_refusal_is_still_unreached() {
    let names = ["__nv_fp8_e8m0", "__nv_cvt_float_to_e8m0", "__nv_fp8x2_e8m0"];
    let used: Vec<String> = vendored()
        .iter()
        .filter(|(_, p)| {
            let t = std::fs::read_to_string(p).expect("read");
            names.iter().any(|n| t.contains(n))
        })
        .map(|(n, _)| n.clone())
        .collect();
    assert!(
        used.is_empty(),
        "cuda_fp8.h declines the e8m0 family because the closure never reaches \
         it; these files now do: {used:?}"
    );
}

/// `pie_mma.cuh` is the one header whose correctness no test can check here,
/// so this checks that the thing which CAN check it still exists and still
/// says it can fail.
///
/// The shim is a per-lane register map: wrong, it compiles and returns
/// plausible numbers (`examples/mma_probe.rs` opens with why). Its parity
/// against `nvcuda::wmma` needs `nvcc` and a device, which `tests/` in this
/// crate deliberately does not have — so the guarantee lives in an example,
/// and an example is only a guarantee while something can invoke it and read
/// a verdict.
///
/// §23.11 found that probe exiting 0 on every path, pass or fail, for as long
/// as it had existed. §23.12 then found it was the ONLY example that was both
/// unfailable and the sole custodian of its claim. This keeps the second half
/// from silently coming back: if the exit path goes, the shim's only check
/// becomes a report again.
#[test]
fn the_mma_shim_still_has_something_that_can_fail() {
    let probe = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/mma_probe.rs");
    let src = std::fs::read_to_string(&probe).expect("examples/mma_probe.rs");

    assert!(
        src.contains("std::process::exit(1)"),
        "mma_probe must exit non-zero on failure, or nothing can be gated on \
         the only check `pie_mma.cuh` has"
    );
    assert!(
        src.contains("pub fn run() -> bool"),
        "mma_probe's run() must report a verdict rather than print one"
    );
    assert!(
        src.contains("fn sensitivity(") && src.contains("-> bool"),
        "the transposed-store control must report whether it was CAUGHT; an \
         unmeasured control is not a pass"
    );

    // And the shim it checks is still the one the units compile against.
    let shim = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src/pie_mma.cuh");
    assert!(shim.is_file(), "csrc/src/pie_mma.cuh");
    assert!(
        src.contains("pie_mma.cuh"),
        "mma_probe must name the header it is the check for"
    );
}

/// A tile kernel needs CUDA **13.3 or newer runtime headers**, and nothing in
/// the toolchain says so when they are older.
///
/// CUDA 13.3's tile frontend injects `-D__NV_TL_BUILTIN__=__tile_builtin__`,
/// and 13.3's `cuda_bf16.h` carries that marker on the struct:
///
/// ```text
///   13.0   struct                   __CUDA_ALIGN__(2) __nv_bfloat16 {...}
///   13.3   struct __NV_TL_BUILTIN__ __CUDA_ALIGN__(2) __nv_bfloat16 {...}
/// ```
///
/// Without it `__nv_bfloat16` is an ordinary two-byte aggregate, every tile
/// of it lowers as `tile<2xi8>`, and tile codegen dies a thousand lines deep
/// with `"Unexpected element type in tile!"` naming a type the user never
/// wrote. Adding the attribute by hand to a 13.0 header is the entire fix.
///
/// This is easy to hit because the toolchain arrives as independently
/// versioned pip wheels — `nvidia-cuda-nvcc` and `nvidia-cuda-nvrtc` can be
/// 13.3 while `nvidia-cuda-runtime`, which owns these headers, is 13.0 — and
/// no version check fires. It cost this tree a day and a retracted bug
/// report; `.wiki/driver/cutile-16bit-header-trap.cu` is the account.
///
/// So the detector is written down where a tile build can reach it. There is
/// no tile build in this crate yet, which is exactly why: the knowledge has
/// to outlive the session that bought it.
///
/// `cuda_tf32.h` is the cheap version of the same check — it ships only in
/// 13.3+, so its absence dates the headers in one `stat`.
#[test]
fn the_cutile_header_floor_is_written_down_where_a_build_can_find_it() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let kernel = root.join("csrc/src/moe/moe_grouped_gemm_tile.cuh");
    let src = std::fs::read_to_string(&kernel).expect("moe_grouped_gemm_tile.cuh");

    for needle in ["__NV_TL_BUILTIN__", "__tile_builtin__", "cuda_tf32.h", "CUDA_ROOT"] {
        assert!(
            src.contains(needle),
            "the tile kernel no longer records `{needle}`, so the 13.3 runtime \
             requirement is undocumented at the one place a tile build would \
             look. The four are load-bearing: the first three date the runtime \
             headers, and CUDA_ROOT is what tileiras silently needs -- without \
             it every input, including nvcc's own .tilebc, fails with a bare \
             `failed to compile Tile IR program`. See \
             .wiki/driver/cutile-16bit-header-trap.cu and new-horizon 23.18"
        );
    }

    assert!(
        src.contains("__nv_bfloat16"),
        "the tile kernel no longer names NVIDIA's bf16. If it went back to \
         carrying bf16 as `unsigned short`, note what that cost when it was \
         last done: 224 registers against 92 at 16x64x32, and 255 with spills \
         against 160 at kTileM=64, plus every performance conclusion drawn \
         while it was in place"
    );
}

/// The tile kernel must not include this tree's `cuda_bf16.h` adapter.
///
/// `cuda::tiles` constrains tile elements to the scalar types it knows, so
/// this tree's `device::bf16` is refused outright — `template constraint not
/// satisfied` — even carrying `__tile_builtin__`. A tile kernel must name
/// NVIDIA's `__nv_bfloat16`.
///
/// But `csrc/src/cuda_bf16.h` aliases that same name to `device::bf16`, so
/// FlashInfer stays byte-identical to upstream. The two cannot share a
/// translation unit: `cuda_tile.h` forward-declares `struct __nv_bfloat16;`
/// and a struct declaration cannot share a name with a type alias. Whichever
/// include directory comes first decides, and when it is the tree's the
/// build dies on a redefinition rather than on anything informative.
///
/// The kernel includes nothing that wants the adapter, which is what keeps
/// this tractable. This test is here so it stays that way.
#[test]
fn the_tile_kernel_stays_out_of_the_adapter_headers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("csrc/src/moe/moe_grouped_gemm_tile.cuh"))
        .expect("moe_grouped_gemm_tile.cuh");

    let includes: Vec<&str> = src
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("#include"))
        .collect();

    for bad in ["pie_mma.cuh", "\"cuda_bf16.h\"", "\"cuda_fp16.h\""] {
        assert!(
            !includes.iter().any(|l| l.contains(bad)),
            "the tile kernel now includes {bad}, which redeclares \
             __nv_bfloat16 or __half against cuda_tile.h's own forward \
             declarations. Includes are: {includes:?}"
        );
    }

    assert!(
        includes.iter().any(|l| l.contains("<cuda_bf16.h>")),
        "the tile kernel no longer includes NVIDIA's <cuda_bf16.h>. It needs \
         the real type, and it needs NVIDIA's include directory to precede \
         csrc/src so the angle-bracket include does not find the adapter. \
         Includes are: {includes:?}"
    );
}

/// The fused MoE tile kernel is a NEGATIVE result and must keep saying so.
///
/// `moe_fused_tile.cuh` writes fc1 + swiglu + fc2 as one `__tile_global__`
/// with the intermediate never stored — the thing that would close the
/// CUTLASS island's remaining advantage at the decode census. It is correct
/// and it is slower than not fusing: 1.778 ms against 0.933 for two unfused
/// tile GEMMs and 0.581 for the island.
///
/// The cause is shared memory. The tile compiler stages `partition_view`
/// loads through it, and the fused working set takes 92-99 KB of a 100 KB
/// budget — one block per SM, where the unfused grouped GEMM takes 16 KB.
///
/// A file like this is dangerous precisely because it looks like a kernel
/// someone should finish. The banner is the only thing stopping that, so the
/// banner is a gate.
#[test]
fn the_fused_tile_kernel_still_declares_itself_a_negative_result() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("csrc/src/moe/moe_fused_tile.cuh");
    let src = std::fs::read_to_string(&path).expect("csrc/src/moe/moe_fused_tile.cuh");

    assert!(
        src.contains("NEGATIVE result"),
        "moe_fused_tile.cuh no longer announces that it is a negative result. \
         It is slower than not fusing (1.778 ms against 0.933) and the next \
         reader needs to know that before they invest in it"
    );

    for needle in ["SHARED", "occupancy", "0.581"] {
        assert!(
            src.contains(needle),
            "moe_fused_tile.cuh no longer records `{needle}`, so the reason it \
             loses — shared-memory staging collapsing occupancy, against a \
             named island figure — is no longer on the file. See \
             .wiki/driver/new-horizon.md 23.17"
        );
    }
}

/// A `Toolchain` floor is necessary and NOT sufficient for the tile unit, and
/// `unit.rs` must keep saying so.
///
/// Every other unit fails safe: NVRTC rejects source it cannot compile,
/// loudly, in `tests/units.rs`. `moe/moe_grouped_gemm_tile` does not. Measured
/// with NVRTC 13.3.33 and a bf16 tile `mma`:
///
/// ```text
///   nvrtcCompileProgram   rc = 0
///   nvrtcGetCUBIN         .note.nv.tkinfo and NO .text
///   cuModuleLoadData      SUCCESS
///   cuModuleGetFunction   CUDA_ERROR_NOT_FOUND
/// ```
///
/// A tile kernel compiles to Tile IR, not SASS, and something downstream must
/// assemble it — a driver new enough to do it at load, or `tileiras` over
/// `nvrtcGetTileIR`'s output before the cubin is cached. So a floored tile
/// unit under a 13.3 NVRTC would compile clean, cache, load, and fail at the
/// FIRST LAUNCH.
///
/// That is the one shape this crate's gates cannot see, which is exactly why
/// it has to be written down where the person adding the demand will read it.
#[test]
fn the_tile_units_floor_is_still_marked_insufficient_on_its_own() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("src/unit.rs")).expect("src/unit.rs");

    for needle in ["tileiras", "CUDA_ROOT", "NOT_FOUND", "nvrtcGetTileIR"] {
        assert!(
            src.contains(needle),
            "src/unit.rs no longer records `{needle}`. A Toolchain floor alone \
             makes a tile unit compile clean and fail at the first launch, and \
             the DEMANDS table is where whoever adds that floor will look. See \
             .wiki/driver/new-horizon.md 23.18"
        );
    }
}

/// RMSNorm was rewritten in CuTile twice. The second one is FASTER, and both
/// files must keep saying so.
///
/// The first attempt measured 3.84 us against the tree's 2.93 and was written
/// off as "a code-size argument, not a speed one". It was written in a naive
/// dialect. In NVIDIA's own idiom -- `ct::iota` plus a `ct::load` gather
/// rather than a `partition_view` over a 1-D row, `latency=1` on each load,
/// the hidden size a template parameter, `assume_aligned<16>` -- it is
/// 1.51x faster at H=4096 and 1.59x at H=7168, exact at both, and ties the
/// hand-vectorised `rmsnorm_vec8` without needing its alignment check.
///
/// Both halves are gated. A negative result about an uncommitted thing leaves
/// no artifact; a REVERSED negative result leaves a worse one, because the
/// old conclusion is already quoted elsewhere.
#[test]
fn the_rmsnorm_cutile_result_is_still_recorded_and_still_says_faster() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let hand = std::fs::read_to_string(root.join("csrc/src/norm/rmsnorm.cuh"))
        .expect("csrc/src/norm/rmsnorm.cuh");
    let tile = std::fs::read_to_string(root.join("csrc/src/norm/rmsnorm_tile.cuh"))
        .expect("csrc/src/norm/rmsnorm_tile.cuh");

    assert!(
        hand.contains("FASTER"),
        "rmsnorm.cuh no longer records that the CuTile twin beats it. That was \
         a reversal of this file's own earlier claim, and a reversal that goes \
         missing leaves the withdrawn version standing"
    );
    for needle in ["iota", "latency=1", "assume_aligned"] {
        assert!(
            tile.contains(needle),
            "rmsnorm_tile.cuh no longer records `{needle}`. Those three are the \
             difference between the CuTile RMSNorm that lost and the one that \
             won -- see .wiki/driver/new-horizon.md 23.20"
        );
    }
    assert!(
        tile.contains("0.1103") || tile.contains("masked"),
        "rmsnorm_tile.cuh no longer records that the tail must be masked. \
         Unmasked it looks healthy at H=4096 and is wrong at H=7168, which is \
         the shape a careless bench does not pick"
    );
}

/// The elementwise result is a ROOFLINE result and must keep saying so.
///
/// `mlp/swiglu_tile.cuh` is bit-exact against `swiglu.cuh` and 1.53x faster
/// at 25 MB — and 4% SLOWER at 805 MB, where both sit at 77-80% of the
/// L40S's ~864 GB/s HBM peak. Quoting only the first number would make an
/// elementwise CuTile rewrite look like a free 1.5x, which it is not: no
/// programming model makes a kernel at the memory roofline faster.
///
/// This is the shape of claim that decays worst, because the favourable half
/// is the one that gets repeated.
#[test]
fn the_swiglu_tile_result_still_carries_both_halves() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("csrc/src/mlp/swiglu_tile.cuh"))
        .expect("csrc/src/mlp/swiglu_tile.cuh");

    assert!(
        src.contains("roofline"),
        "swiglu_tile.cuh no longer names the roofline, which is the half of \
         its result that bounds the claim: it is 1.53x faster cached and 4% \
         slower at 805 MB"
    );
    for needle in ["0.008", "1.218", "load_masked", "store_masked"] {
        assert!(
            src.contains(needle),
            "swiglu_tile.cuh no longer records `{needle}`. The two timings are \
             the two ends of the roofline finding; the masks are what make the \
             kernel general rather than correct-at-one-shape. See \
             .wiki/driver/new-horizon.md 23.21"
        );
    }
}

/// The router top-K result has three traps on it and all three must stay.
///
/// `moe/topk_softmax_tile.cuh` beats `topk_softmax_warp_x1` — a hand-tuned
/// warp-resident reduction, not a first draft — by 1.28x at decode, with
/// identical expert indices. The traps are what make that result reproducible
/// rather than lucky:
///
/// * the weights renormalise by the WINNERS' own sum, not all experts. Get it
///   wrong and the indices still match while the weights differ by 0.108;
/// * a local `int[TOPK]` array costs 6.7x — 20.38 us against 3.05 — because a
///   `__tile_global__` has no per-thread scratch;
/// * `ct::exp` on a scalar is FREE, which is the hypothesis that was almost
///   published as the cause of the 6.7x and was wrong.
///
/// A trap that produced a plausible wrong answer is worth more written down
/// than the result it guards.
#[test]
fn the_topk_tile_traps_are_still_on_the_file() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("csrc/src/moe/topk_softmax_tile.cuh"))
        .expect("csrc/src/moe/topk_softmax_tile.cuh");

    for (needle, why) in [
        ("0.108", "the weight-definition trap: dividing by the sum over ALL \
                   experts leaves the indices identical and the weights wrong"),
        ("20.38", "the local-array cost, which is 6.7x and survives \
                   #pragma unroll and a compile-time TOPK"),
        ("3.06", "the measured A/B showing scalar ct::exp is free -- the \
                  explanation that was almost published in place of the real one"),
        ("IDENTICAL", "the correctness bar: expert indices must match, because \
                       a different expert is a different model"),
    ] {
        assert!(
            src.contains(needle),
            "topk_softmax_tile.cuh no longer records `{needle}` -- {why}. See \
             .wiki/driver/new-horizon.md 23.22"
        );
    }
}

/// The tile kernels are ADDITIONS. Every one must carry a preference
/// predicate, and no incumbent may be described as replaced.
///
/// This was asked for explicitly and it is also the only defensible shape:
/// the alternatives need NVRTC 13.3, 13.3 runtime headers and `tileiras`,
/// and this crate loads NVRTC 13.0.88. An alternative that cannot be
/// selected on the machine in front of you is not an alternative, it is a
/// removal.
///
/// `csrc/src/tile_alternatives.cuh` pins each predicate to the rows of the
/// sweep that produced it with `static_assert`s, so a bound that gets
/// rounded fails a compile rather than quietly firing the slower kernel.
/// That is not hypothetical: `swiglu_tile_preferred` was first written as
/// `6 * n <= 100 MB`, which excluded the very point it was measured at.
#[test]
fn every_tile_kernel_is_an_alternative_with_a_predicate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src");

    for (file, pred) in [
        ("moe/moe_grouped_gemm_tile.cuh", "moe_grouped_gemm_tile_preferred"),
        ("norm/rmsnorm_tile.cuh", "rmsnorm_tile_preferred"),
        ("mlp/swiglu_tile.cuh", "swiglu_tile_preferred"),
        ("moe/topk_softmax_tile.cuh", "topk_softmax_tile_preferred"),
        ("norm/rmsnorm_rasr_tile.cuh", "rmsnorm_rasr_tile_preferred"),
    ] {
        let src = std::fs::read_to_string(root.join(file)).expect(file);
        assert!(
            src.contains(&format!("constexpr bool {pred}")),
            "{file} no longer defines `{pred}`. A tile kernel without a \
             preference predicate is a replacement by default, and these are \
             additions"
        );
        assert!(
            src.contains("ALTERNATIVE"),
            "{file} no longer says it is an alternative. The incumbent is the \
             fallback for every toolchain that cannot compile a tile kernel, \
             which today is every toolchain this crate loads"
        );
    }

    let alts = std::fs::read_to_string(root.join("tile_alternatives.cuh"))
        .expect("csrc/src/tile_alternatives.cuh");
    let asserts = alts.matches("static_assert").count();
    assert!(
        asserts >= 11,
        "tile_alternatives.cuh has {asserts} static_asserts; it had 11, one per \
         measured endpoint. A predicate that is no longer pinned to its sweep \
         is a comment with a type"
    );
    for measured in ["1.94 vs 2.93", "0.038 vs 0.057", "3.06 vs 3.90", "7.22 vs 6.08",
                     "2.41 vs 4.33"] {
        assert!(
            alts.contains(measured),
            "tile_alternatives.cuh no longer cites `{measured}`. The point of \
             that file is that each bound names the measurement it came from"
        );
    }
}

/// No tile kernel may spell a `<cstdint>` type. NVRTC does not have them.
///
/// `nvcc` force-includes `cuda_runtime.h` and so has `<cstdint>`
/// transitively; NVRTC does not, and every tile kernel here is destined for
/// NVRTC. `moe_fused_tile.cuh` said `ct::extents<uint32_t, ...>` — copied
/// from NVIDIA's own `matmul.cuh` — compiled clean under `nvcc`, and failed
/// through the JIT path with `identifier "uint32_t" is undefined`.
///
/// An AOT build cannot see this class of defect at all, which is why it is a
/// gate on the text rather than a note in a header. The builtin spellings
/// (`unsigned`, `int`, `long long`) always work.
#[test]
fn no_tile_kernel_spells_a_cstdint_type() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/src");

    let tile_kernels = [
        "moe/moe_grouped_gemm_tile.cuh",
        "moe/moe_fused_tile.cuh",
        "moe/topk_softmax_tile.cuh",
        "norm/rmsnorm_tile.cuh",
        "norm/rmsnorm_rasr_tile.cuh",
        "mlp/swiglu_tile.cuh",
    ];

    for file in tile_kernels {
        let src = std::fs::read_to_string(root.join(file)).expect(file);
        for (line_no, line) in src.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") || code.starts_with("///") {
                continue;
            }
            for ty in ["uint32_t", "uint64_t", "int32_t", "int64_t", "uint16_t", "size_t"] {
                assert!(
                    !line.contains(ty),
                    "{file}:{} spells `{ty}`, which NVRTC does not have -- nvcc \
                     only sees it because it force-includes cuda_runtime.h. Use \
                     a builtin. This exact defect was found by JIT-compiling \
                     these kernels rather than by any AOT build.",
                    line_no + 1
                );
            }
        }
    }
}
