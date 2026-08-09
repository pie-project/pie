//! What the JIT migration is blocked on, per file, for the whole tree.
//!
//! ```text
//! cargo run -p kernels-cuda --example migration_plan
//! ```
//!
//! # Why this exists
//!
//! "Convert everything to `.cuh`" is not a step, it is an outcome, and trying
//! to take it as a step is how a migration corrupts a tree. The unit that has
//! to move together is not the file — it is the file's **closure**:
//!
//! * every `.cu` that includes a shared device header moves when that header
//!   moves, because `__nv_bfloat16*` and `device::bf16*` are the same two
//!   bytes and C++ correctly refuses to confuse them;
//! * a host launcher may only be deleted when its whole CONSUMER set is gone,
//!   and the consumer set is the generated shim entry PLUS every `.cu` that
//!   calls it as a building block;
//! * some files must never move: the CUTLASS/FlashInfer island is host
//!   libraries and link-time dispatch, and JIT does nothing for either.
//!
//! Those three facts are properties of a GRAPH, and reading them off 160
//! files by eye is how a migration ends up half-done in the middle. This
//! walks the graph and prints the order.
//!
//! # What it does not know
//!
//! Whether a kernel's arithmetic survives the move. `__hfma2` is a hardware
//! half2 instruction, and emulating it changes rounding — a file that uses
//! one is flagged here as needing a decision, not as ready. The decision is a
//! human's, and it comes with parity evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn csrc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc")
}

/// A source file, and what the graph knows about it.
#[derive(Default, Debug)]
struct File {
    /// `__global__` names it defines, namespace-qualified.
    globals: Vec<String>,
    /// Host functions at namespace scope — the launchers.
    launchers: Vec<String>,
    /// Quoted includes.
    includes: Vec<String>,
    /// It reaches for CUTLASS/CuTe/FlashInfer, so it is the island.
    island: bool,
    /// It uses an instruction the prelude cannot restate.
    hardware: Vec<String>,
    /// It still names a CUDA scalar header's types.
    untouched: bool,
}

/// Instructions whose meaning is the hardware's, not arithmetic the prelude
/// can write out.
///
/// `__hfma2` and `__hsub2` are half2 SIMD: emulating them means unpacking to
/// fp32, which is a different rounding and a different instruction count. A
/// file using one is a body change with its own parity evidence
/// (`new-horizon.md` §8), not a substitution.
const HARDWARE: &[&str] = &[
    // half2 SIMD: emulating one means unpacking to fp32, which is a different
    // rounding and a different instruction count.
    "__hfma2",
    "__hsub2",
    "__hmul2",
    "__hadd2",
    "__hmax2",
    "__hmin2",
    // The tensor-core fragment types. `wmma::fragment<..., __nv_bfloat16, ...>`
    // is specialised on NVIDIA's type by the MMA headers; a structurally
    // identical `device::bf16` is an incomplete type, because what selects the
    // specialisation is the NAME.
    "wmma::fragment",
    // The FP8 conversions return `__half_raw`, a type from the header this
    // prelude exists to avoid. A kernel using one is converting through
    // NVIDIA's fp16 representation, not through ours.
    "__nv_cvt_fp8",
];

fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| Some(e.ok()?.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk(&path, root, out);
        } else if path.extension().is_some_and(|e| e == "cu" || e == "cuh" || e == "cpp") {
            out.push(
                path.strip_prefix(root)
                    .expect("under csrc")
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
}

fn read(rel: &str) -> File {
    let text = std::fs::read_to_string(csrc().join(rel)).unwrap_or_default();
    let mut file = File {
        island: text.contains("cutlass") || text.contains("cute/") || text.contains("flashinfer"),
        // CODE, not prose. `altup_aux.cuh`'s header comment explains at
        // length why it does not include `cuda_bf16.h`, and a substring scan
        // reads that as a dependency -- which would report every migrated
        // file as unmigrated and make the plan exactly backwards.
        untouched: text.lines().any(|l| {
            let t = l.trim_start();
            !t.starts_with("//")
                && (t.starts_with("#include <cuda_bf16.h>")
                    || t.starts_with("#include <cuda_fp16.h>")
                    || t.contains("__nv_bfloat16")
                    || t.contains("__half"))
        }),
        ..File::default()
    };
    for instruction in HARDWARE {
        if text
            .lines()
            .any(|l| !l.trim_start().starts_with("//") && l.contains(instruction))
        {
            file.hardware.push((*instruction).to_string());
        }
    }
    let mut ns = String::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("namespace ")
            && let Some(named) = rest.split([' ', '{']).next()
            && !named.is_empty()
        {
            ns = named.to_string();
        }
        if let Some(after) = trimmed.strip_prefix("__global__ void ") {
            let leaf: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !leaf.is_empty() {
                file.globals.push(format!("{ns}::{leaf}"));
            }
        }
        // A launcher is a host function at column zero: `void name(`.
        if line.starts_with("void ")
            && let Some(after) = line.strip_prefix("void ")
        {
            let leaf: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !leaf.is_empty() && after[leaf.len()..].starts_with('(') {
                file.launchers.push(leaf);
            }
        }
        if let Some(rest) = trimmed.strip_prefix("#include \"")
            && let Some(name) = rest.split('"').next()
        {
            file.includes.push(name.to_string());
        }
    }
    file
}

fn main() {
    let root = csrc();
    let mut rels = Vec::new();
    walk(&root.join("src"), &root, &mut rels);
    let files: BTreeMap<String, File> = rels.iter().map(|r| (r.clone(), read(r))).collect();

    // Who calls each launcher, other than the file that defines it.
    let mut callers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (rel, file) in &files {
        let text = std::fs::read_to_string(root.join(rel)).unwrap_or_default();
        for (other, other_file) in &files {
            if other == rel {
                continue;
            }
            for launcher in &other_file.launchers {
                // A call, not the declaration: `name(` preceded by `::` or a
                // space is close enough to name a consumer, and this is a
                // plan rather than a compiler.
                if text.contains(&format!("::{launcher}(")) || text.contains(&format!(" {launcher}("))
                {
                    callers.entry(launcher.clone()).or_default().insert(rel.clone());
                }
            }
            let _ = file;
        }
    }

    let (mut ready, mut blocked, mut island, mut decide, mut done) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (rel, file) in &files {
        if !rel.ends_with(".cu") {
            continue;
        }
        if file.island {
            island.push(rel.clone());
            continue;
        }
        if !file.hardware.is_empty() {
            decide.push(format!("{rel}  ({})", file.hardware.join(", ")));
            continue;
        }
        if file.globals.is_empty() && file.launchers.is_empty() {
            continue;
        }
        let blockers: Vec<String> = file
            .launchers
            .iter()
            .filter_map(|l| {
                let who = callers.get(l)?;
                (!who.is_empty()).then(|| format!("{l} <- {}", who.iter().cloned().collect::<Vec<_>>().join(", ")))
            })
            .collect();
        if file.untouched {
            blocked.push(format!("{rel}  [still on the CUDA scalar headers]"));
        } else if blockers.is_empty() {
            ready.push(format!("{rel}  ({} globals)", file.globals.len()));
        } else {
            blocked.push(format!("{rel}\n      {}", blockers.join("\n      ")));
        }
    }
    for (rel, file) in &files {
        if rel.ends_with(".cuh") && !file.globals.is_empty() && !file.island && !file.untouched {
            done.push(rel.clone());
        }
    }

    println!("# JIT migration plan\n");
    println!("## Ready: no launcher has another C++ caller, nothing hardware-bound");
    println!("These can be extracted to `.cuh` templates, given rows, switched");
    println!("in `JIT_DISPATCHED`, and deleted — in that order.\n");
    for r in &ready {
        println!("  {r}");
    }
    println!("\n## Blocked: a launcher still has C++ callers, or the file has not converted\n");
    for b in &blocked {
        println!("  {b}");
    }
    println!("\n## Needs a decision: hardware instructions the prelude cannot restate\n");
    for d in &decide {
        println!("  {d}");
    }
    println!("\n## Island: stays nvcc (new-horizon.md §5)\n");
    println!("  {} files", island.len());
    println!("\n## Already templates in a .cuh\n");
    for d in &done {
        println!("  {d}");
    }
    println!(
        "\n{} ready, {} blocked, {} need a decision, {} island",
        ready.len(),
        blocked.len(),
        decide.len(),
        island.len()
    );
}
