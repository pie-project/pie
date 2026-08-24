//! The Load contract's first real test: a shipped HF checkpoint through a
//! family's production table, joined against the plan that will consume it.
//!
//! ```text
//! baker_load <sku> [hf-cache-dir] [--base <flavor>] [--verbose]
//!                   [--digest <param>]...
//! ```
//!
//! Two tables meet here and nothing else does. The DEMAND is `plan.params`
//! -- every `Const<Tensor>` a model text stated, recorded at trace time
//! with the canonical name, shape, shard cut and repr. The SUPPLY is the
//! family's `IMPORTS` row for the same SKU, run over the checkpoint by
//! `model::produce`. A row of one that finds no row of the other is a bug
//! in whichever was written blind, and this binary is what stops that
//! being discovered on a GPU.
//!
//! THE DEVICE HALF IS DELIBERATELY NOT HERE. Uploading would mean `cudarc`
//! in `crates/model`'s graph -- the crate whose whole job is to hold model
//! texts and be linked by nothing -- to prove a `cudaMemcpy` that has no
//! decision in it: every produced tensor is already dense, row-major and
//! canonical, so the upload is one contiguous copy per row and the arena
//! layout is the loader's question rather than the import's. What this
//! prints instead is the upload's own arithmetic (bytes per row, total,
//! largest single allocation), which is what a driver would need to size
//! an arena, and the join that says the bytes are the right bytes.

use std::collections::BTreeMap;

use model::produce::{Dtype, HostTensor, produce};
use model::snapshot::Snapshot;
use model_dsl::Plane;

/// Where a SKU's checkpoint sits under `~/.cache/huggingface/hub`, for the
/// SKUs this harness has actually been run against. An explicit second
/// argument overrides it; a SKU that is in neither is an error rather than
/// a guess.
const CACHED: &[(&str, &str)] = &[
    ("qwen35-d0.8b-bf16-kv-bf16", "models--Qwen--Qwen3.5-0.8B"),
    ("qwen35-a3b-bf16-kv-bf16", "models--Qwen--Qwen3.5-35B-A3B"),
    ("gemma4-e4b-bf16-kv-bf16", "models--google--gemma-4-E4B-it"),
    (
        "gptoss-20b-bf16-mxfp4-kv-bf16",
        "models--openai--gpt-oss-20b",
    ),
];

fn main() {
    let mut args = std::env::args().skip(1);
    let sku = args.next().unwrap_or_else(|| {
        eprintln!("usage: baker_load <sku> [hf-cache-dir] [--base <flavor>] [--verbose]");
        std::process::exit(2);
    });
    let mut cache_dir = None;
    let mut base = "safetensors-bf16".to_string();
    let mut verbose = false;
    let mut digests: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--base" => base = args.next().expect("--base wants a flavor"),
            "--digest" => digests.push(args.next().expect("--digest wants a param name")),
            "--verbose" | "-v" => verbose = true,
            other => cache_dir = Some(other.to_string()),
        }
    }
    let cache_dir = cache_dir
        .or_else(|| {
            CACHED
                .iter()
                .find(|(s, _)| *s == sku)
                .map(|(_, d)| (*d).to_string())
        })
        .unwrap_or_else(|| {
            eprintln!("no cached checkpoint is filed for `{sku}`; name one as the second argument");
            std::process::exit(2);
        });

    let trace = model::trace_of(&sku).unwrap_or_else(|| {
        eprintln!("`{sku}` is not a catalog row");
        std::process::exit(2);
    });
    let import = model::import_of(&sku, &base).unwrap_or_else(|| {
        eprintln!(
            "`{sku}` names no `{base}` import; it ships {:?}",
            model::bases_for(&sku)
        );
        std::process::exit(2);
    });
    let snap = Snapshot::open(&cache_dir).unwrap_or_else(|| {
        eprintln!("no safetensors snapshot under ~/.cache/huggingface/hub/{cache_dir}");
        std::process::exit(2);
    });

    println!("sku `{sku}`, base `{base}`");
    println!(
        "  checkpoint {}\n    {} shard(s), {} tensors",
        snap.dir.display(),
        snap.shards(),
        snap.len()
    );

    let plan = trace(Plane::Cuda);
    println!(
        "  plan `{}`: {} params, {} caches, {} ops",
        plan.name,
        plan.params.len(),
        plan.caches.len(),
        plan.ops.len()
    );
    println!("  import: {} rows", import.rows.len());

    let t0 = std::time::Instant::now();
    let produced = match produce(&import, &|n| snap.read(n)) {
        Ok(p) => p,
        Err(e) => {
            println!("\nPRODUCTION REFUSED: {e}");
            std::process::exit(1);
        }
    };
    let bytes: usize = produced.iter().map(|(_, t)| t.bytes.len()).sum();
    println!(
        "  produced: {} tensors, {} in {:.1}s",
        produced.len(),
        human(bytes as u64),
        t0.elapsed().as_secs_f64()
    );

    let supply: BTreeMap<&str, &HostTensor> =
        produced.iter().map(|(n, t)| (n.as_str(), t)).collect();

    let mut satisfied = 0usize;
    let mut missing: Vec<&str> = Vec::new();
    let mut wrong_shape: Vec<(&str, &[u64], &[u64])> = Vec::new();
    let mut repr_notes: Vec<(&str, &str, Dtype)> = Vec::new();
    let mut demanded_bytes = 0u64;
    let mut largest = (0u64, "");

    for p in &plan.params {
        let Some(t) = supply.get(p.name.as_str()) else {
            missing.push(&p.name);
            continue;
        };
        if t.shape != p.shape {
            wrong_shape.push((&p.name, &p.shape, &t.shape));
            continue;
        }
        satisfied += 1;
        let n = t.bytes.len() as u64;
        demanded_bytes += n;
        if n > largest.0 {
            largest = (n, &p.name);
        }
        if !repr_agrees(&p.repr, t.dtype) {
            repr_notes.push((&p.name, &p.repr, t.dtype));
        }
    }

    let unclaimed: Vec<&str> = produced
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| !plan.params.iter().any(|p| p.name == *n))
        .collect();

    println!(
        "\njoin: {} demanded / {} satisfied / {} shape mismatches / {} missing",
        plan.params.len(),
        satisfied,
        wrong_shape.len(),
        missing.len()
    );
    for (n, want, got) in &wrong_shape {
        println!("  MISMATCH {n}: plan wants {want:?}, import produced {got:?}");
    }
    for n in &missing {
        println!("  MISSING  {n}: no import row produces it");
    }
    if !unclaimed.is_empty() {
        println!(
            "  {} produced row(s) no param demands: {}",
            unclaimed.len(),
            preview(&unclaimed)
        );
    }
    if repr_notes.is_empty() {
        println!("  repr: every satisfied param's storage matches its plan repr");
    } else {
        println!(
            "  repr: {} param(s) stored at a dtype the plan does not name",
            repr_notes.len()
        );
        let mut seen: BTreeMap<(&str, &str), Vec<&str>> = BTreeMap::new();
        for (n, repr, dt) in &repr_notes {
            seen.entry((repr, dt.name())).or_default().push(n);
        }
        for ((repr, dt), names) in &seen {
            println!(
                "    plan says {repr}, checkpoint holds {dt} -- {} row(s): {}",
                names.len(),
                preview(names)
            );
        }
    }

    let unread = snap.len() - snap.taken();
    if verbose && unread > 0 {
        let names: Vec<&str> = snap.untaken();
        println!(
            "  {} checkpoint tensor(s) no import row reads: {}",
            unread,
            preview(&names)
        );
    } else if unread > 0 {
        println!("  {unread} checkpoint tensor(s) no import row reads (-v to list)");
    }

    if !digests.is_empty() {
        println!("\nprobes (FNV-1a/64 over the produced bytes)");
        for name in &digests {
            match supply.get(name.as_str()) {
                None => println!("  {name}: no import row produces it"),
                Some(t) => println!(
                    "  {name}: {:?} {} fnv={:016x} head=[{}]",
                    t.shape,
                    t.dtype.name(),
                    fnv1a64(&t.bytes),
                    head(t)
                ),
            }
        }
    }

    // The upload's own arithmetic, which is the part of the device half
    // that has a decision in it. 256 is the alignment every cuda global
    // load in this tree assumes; a row that starts unaligned costs a
    // second transaction per warp.
    const ALIGN: u64 = 256;
    let mut arena = 0u64;
    for p in &plan.params {
        if let Some(t) = supply.get(p.name.as_str())
            && t.shape == p.shape
        {
            arena = arena.div_ceil(ALIGN) * ALIGN + t.bytes.len() as u64;
        }
    }
    println!("\nwhat a device upload would move");
    println!("  {} across {} rows", human(demanded_bytes), satisfied);
    println!(
        "  largest single row {} (`{}`)",
        human(largest.0),
        largest.1
    );
    println!(
        "  one arena at {ALIGN}-byte row alignment: {} ({} of padding)",
        human(arena),
        human(arena - demanded_bytes)
    );
    println!(
        "  every row is dense and row-major already, so the upload is one\n  \
         contiguous H2D copy per row -- no restride, no repack, no cast"
    );

    let ok = missing.is_empty() && wrong_shape.is_empty();
    println!(
        "\n{}",
        if ok {
            format!("JOIN 100% -- `{sku}` loads")
        } else {
            format!(
                "JOIN {:.1}% -- `{sku}` does NOT load",
                100.0 * satisfied as f64 / plan.params.len() as f64
            )
        }
    );
    if !ok {
        std::process::exit(1);
    }
}

/// The plan's `repr` column is the axis a model text declared its banks
/// at; a checkpoint stores what it stores. They agree for the dense case
/// and are simply different questions for a quantized one, which is why a
/// disagreement is a note and not a refusal.
fn repr_agrees(repr: &str, dtype: Dtype) -> bool {
    match repr {
        "bf16" => dtype == Dtype::Bf16,
        "f16" => dtype == Dtype::F16,
        "f32" => dtype == Dtype::F32,
        // A QUANTISED PLANE STILL HAS A STORAGE ANSWER, and leaving it in
        // the `_` arm was the check declining to run: every repr it did not
        // know agreed with every dtype, so `mxfp4` would have "matched" a
        // bank of bf16 and the line would still have read "every satisfied
        // param's storage matches its plan repr". Both mxfp4 planes are byte
        // runs — 4-bit codes packed two to a byte, E8M0 exponents one to a
        // byte — so `U8` is what the checkpoint must say, and now does.
        "mxfp4" | "e8m0" => dtype == Dtype::U8,
        _ => true,
    }
}

/// A digest a second implementation can be checked against without this
/// binary having to write a file. FNV-1a over the whole payload: order
/// sensitive, which is the property a `pack` needs proved.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The first four elements, so a fold's arithmetic is readable and not
/// only its hash.
fn head(t: &HostTensor) -> String {
    let n = 4.min(t.elems() as usize);
    let mut out = Vec::new();
    for i in 0..n {
        out.push(match t.dtype {
            Dtype::Bf16 => {
                let b = &t.bytes[i * 2..i * 2 + 2];
                format!(
                    "{:.6}",
                    model::produce::bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))
                )
            }
            Dtype::F32 => {
                let b = &t.bytes[i * 4..i * 4 + 4];
                format!("{:.6}", f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            _ => format!("0x{:02x}", t.bytes[i]),
        });
    }
    out.join(", ")
}

fn preview(names: &[&str]) -> String {
    let head: Vec<&str> = names.iter().take(4).copied().collect();
    if names.len() > head.len() {
        format!("{}, ... (+{})", head.join(", "), names.len() - head.len())
    } else {
        head.join(", ")
    }
}

fn human(n: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", U[i])
}
