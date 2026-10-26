//! Catalog coverage: for every single-device deployment (or those whose name holds
//! one of the arguments), trace the plan for `Platform::Cerebras`, load it on a
//! dry shell (no weights, no buffers) and trace each class of the plan at a
//! prefill and a decode shape (`engine_cerebras::serve::dry`). Prints one line
//! per deployment: `OK` (every fire emitted its programs), `REFUSED` (the first
//! refusal), or `PANIC`.
//!
//! ```text
//! cargo run --release -p engine-cerebras --example coverage -- [--compile] [--lean] [--dump DIR] [--shard I/N] [-v] [filter...]
//! ```
//!
//! `--lean` skips the classes that run a custom-mask or adapter arm.
//! `--compile` also compiles every emitted program with cslc;
//! `--dump DIR` writes each deployment's programs as
//! `DIR/<deployment>.<n>.csl.txt`; `--shard I/N` takes every Nth deployment
//! from the Ith (one process of N side by side); `-v` prints every fire.

use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};

use engine_cerebras::DeviceBoot;
use engine_cerebras::serve::{Boot, Shell};
use poem::Platform;
use poem_compiler::Budget;

const PREFILL: u32 = 8;

fn main() {
    let mut compile = false;
    let mut lean = false;
    let mut verbose = false;
    let mut dump: Option<PathBuf> = None;
    let mut shard = (0usize, 1usize);
    let mut filters: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--compile" => compile = true,
            "-v" => verbose = true,
            "--lean" => lean = true,
            "--dump" => dump = args.next().map(PathBuf::from),
            "--shard" => {
                shard = args
                    .next()
                    .and_then(|s| {
                        let (i, n) = s.split_once('/')?;
                        Some((i.parse().ok()?, n.parse().ok()?))
                    })
                    .filter(|(i, n)| i < n)
                    .expect("--shard I/N with I < N");
            }
            other => filters.push(other.to_string()),
        }
    }
    std::panic::set_hook(Box::new(|_| {}));
    let mut tally = [0usize; 3];
    let catalog = poem_compiler::catalog::deployments()
        .filter(|d| d.deploy.tp == 1)
        .map(|d| {
            (
                d.name.clone(),
                Box::new(|| d.trace(Platform::Cerebras)) as TraceFn,
            )
        });
    for (at, (name, trace)) in catalog.chain(dev_models()).enumerate() {
        if at % shard.1 != shard.0 {
            continue;
        }
        if !filters.is_empty() && !filters.iter().any(|f| name.contains(f.as_str())) {
            continue;
        }
        let started = std::time::Instant::now();
        let got = std::panic::catch_unwind(AssertUnwindSafe(|| {
            one(&name, trace(), compile, lean, verbose, dump.as_deref())
        }));
        let secs = started.elapsed().as_secs_f64();
        let (status, detail) = match got {
            Ok(Ok(detail)) => ("OK", detail),
            Ok(Err(why)) => ("REFUSED", why),
            Err(panic) => (
                "PANIC",
                panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_default(),
            ),
        };
        tally[match status {
            "OK" => 0,
            "REFUSED" => 1,
            _ => 2,
        }] += 1;
        println!(
            "{:<58} {:<8} {:>6.1}s  {}",
            name,
            status,
            secs,
            detail.replace('\n', " ")
        );
    }
    println!("ok {} refused {} panicked {}", tally[0], tally[1], tally[2]);
}

type TraceFn<'a> = Box<dyn Fn() -> poem::Trace + 'a>;

/// The bring-up fixtures beside the catalog: the tiny Qwen at bf16 weights,
/// and the pico Qwen that fits one PE.
fn dev_models() -> impl Iterator<Item = (String, TraceFn<'static>)> {
    ["qwen35-tiny", "qwen35-pico"].into_iter().map(|id| {
        let trace = move || {
            poem_compiler::catalog::repository().trace_of(
                id,
                poem::Dtype::Bf16,
                poem::Dtype::Bf16,
                Platform::Cerebras,
            )
        };
        (format!("{id}-bf16"), Box::new(trace) as TraceFn)
    })
}

fn one(
    name: &str,
    trace: poem::Trace,
    compile: bool,
    lean: bool,
    verbose: bool,
    dump: Option<&Path>,
) -> Result<String, String> {
    let weights: u64 = trace
        .params
        .iter()
        .map(|p| {
            let n: u64 = p.shape.iter().product();
            n * u64::from(dtype_bits(p.dtype)) / 8
        })
        .sum();
    let budgets = engine::load::Budgets {
        max_lanes: 2,
        max_tokens: 24,
        max_adapters: u32::from(
            trace
                .params
                .iter()
                .any(|p| p.source == poem_ir::ParamSource::Registered),
        ),
        page_size: 16,
        max_context: 32,
        slots: 4,
        ..engine::load::Budgets::default()
    };
    // The bytes of 4 pages.
    let kv_budget =
        engine_cerebras::store::kv::Paging::of(budgets.page_size, budgets.max_context, 4, 4)
            .map_err(engine_cerebras::Fault::from)
            .and_then(|paging| engine_cerebras::store::pool_demand(&trace, paging))
            .map_err(|fault| format!("pool: {fault}"))?;
    let patches = engine_cerebras::api::patch_ladder(&trace, &budgets);
    let voxels = engine_cerebras::dit::voxel_ladder(&trace, Some(4096), Some(4), budgets.max_lanes);
    let contract = checkpoint::contract::ModelContract {
        alignment: 1,
        tensors: Vec::new(),
        groups: Vec::new(),
    };
    let boot = DeviceBoot::default();
    let mut shell = Shell::load_dry(
        Boot {
            trace,
            contract: &contract,
            checkpoint: Path::new("/dev/null"),
            budget: Budget {
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                buckets: Vec::new(),
                max_adapters: budgets.max_adapters,
                readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
            },
            patches,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots: budgets.slots,
            kv_budget: Some(kv_budget),
            memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
            device: &boot,
        },
        voxels,
        compile,
    )
    .map_err(|fault| format!("load: {fault}"))?;
    let probes = shell.synthetic_fires(PREFILL, lean);
    if let Some(dir) = dump {
        let _ = std::fs::create_dir_all(dir);
        for (n, text) in shell.device().dry_texts().iter().enumerate() {
            let _ = std::fs::write(dir.join(format!("{}.{n}.csl.txt", name)), text);
        }
    }
    let texts = shell.device().dry_texts();
    let bytes: usize = texts.iter().map(String::len).sum();
    let classes = probes.iter().map(|p| p.class).max().map_or(0, |c| c + 1);
    if verbose {
        for p in &probes {
            eprintln!(
                "  class {} rows {:>3}: {:?}  {}",
                p.class, p.rows, p.outcome, p.request
            );
        }
    }
    if let Some(bad) = probes.iter().find(|p| p.outcome.is_err()) {
        let failed = probes.iter().filter(|p| p.outcome.is_err()).count();
        return Err(format!(
            "{failed}/{} fires; class {} rows {} ({}): {}",
            probes.len(),
            bad.class,
            bad.rows,
            bad.request,
            bad.outcome.as_ref().err().cloned().unwrap_or_default()
        ));
    }
    Ok(format!(
        "{} classes, {} fires, {} programs ({} KiB){}; weights {} MiB",
        classes,
        probes.len(),
        texts.len(),
        bytes >> 10,
        if compile { ", compiled" } else { "" },
        weights >> 20
    ))
}

fn dtype_bits(dtype: poem::Dtype) -> u32 {
    use poem::Dtype as D;
    match dtype {
        D::F32 | D::I32 | D::U32 => 32,
        D::I64 | D::U64 => 64,
        D::U8 | D::I8 | D::Bool | D::E4m3 | D::E5m2 | D::E8m0 | D::U8g64 => 8,
        D::Mxfp4 => 8,
        D::U4g64 | D::U4g32 | D::U4g64tiled | D::Nvfp4 | D::E2m1 => 4,
        D::U2g32 | D::U2g64 | D::U2g128 => 2,
        _ => 16,
    }
}
