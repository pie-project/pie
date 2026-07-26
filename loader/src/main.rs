//! The loader as a standalone tool.
//!
//! Everything a driver does at load time is reachable here without a GPU, a
//! runtime, or a driver: compile a snapshot, print the plan, check it, compare
//! it against a stored one, and execute it against the real checkpoint bytes.
//!
//! ```text
//! pie-loader dump   SNAPSHOT [options]        the compiled plan, as JSON
//! pie-loader verify SNAPSHOT [options]        compile, then check the result
//! pie-loader diff   SNAPSHOT GOLDEN [options] compile and compare to a dump
//! pie-loader replay SNAPSHOT [options]        execute the plan on the host
//! ```
//!
//! `replay` is the strongest statement the loader can make about itself
//! offline: it reads the checkpoint through the plan and reports what each
//! runtime tensor actually contains, so a plan that verifies but moves the
//! wrong bytes is still caught.
//!
//! Options, positionally, all optional:
//! `BACKEND` (`cuda`|`metal`|`host`), `RUNTIME_QUANT`, `MXFP4_POLICY`
//! (`routed`|`native`|`bf16`), `fused`|`unfused`, `TP_RANK/TP_SIZE`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use pie_loader::arch::RuntimeAbi;
use pie_loader::checkpoint::CheckpointMetadata;
use pie_loader::dump::dump_load_plan_json;
use pie_loader::error::CompileError;
use pie_loader::ffi::inproc::{parse_checkpoint_metadata, parse_model_config};
use pie_loader::load_plan::{
    CUDA_TILE_MAP_MASK, HOST_TILE_MAP_MASK, LoadPlan, METAL_TILE_MAP_MASK, StorageTarget,
};
use pie_loader::planner::compile_load_plan;
use pie_loader::types::{BackendKind, Mxfp4MoePolicy};
use pie_loader::verify::{ContractView, PlanView, verify};

const USAGE: &str = "\
usage: pie-loader <command> SNAPSHOT [BACKEND] [RUNTIME_QUANT] [MXFP4] [FUSION] [TP]

commands:
  dump   SNAPSHOT          compile and print the plan as JSON
  verify SNAPSHOT          compile, then check the plan against its contract
  diff   SNAPSHOT GOLDEN   compile and compare against a stored `dump` output
  replay SNAPSHOT          compile and execute the plan against the checkpoint

options (positional, all optional):
  BACKEND        cuda | metal | host             (default cuda)
  RUNTIME_QUANT  e.g. fp8, int8; '-' for none    (default none)
  MXFP4          routed | native | bf16          (default routed)
  FUSION         fused | unfused                 (default: fused on cuda)
  TP             RANK/SIZE                       (default 0/1)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Fail::Usage(message)) => {
            eprintln!("{message}\n\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Fail::Failed(message)) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

enum Fail {
    Usage(String),
    Failed(String),
}

impl From<CompileError> for Fail {
    fn from(error: CompileError) -> Self {
        Fail::Failed(error.to_string())
    }
}

fn run(args: &[String]) -> Result<(), Fail> {
    let Some(command) = args.first() else {
        return Err(Fail::Usage("no command".to_string()));
    };
    let Some(snapshot) = args.get(1).map(PathBuf::from) else {
        return Err(Fail::Usage(format!("{command} needs a snapshot directory")));
    };

    // `diff` takes one extra positional before the options.
    let (golden, rest) = if command == "diff" {
        let Some(golden) = args.get(2).map(PathBuf::from) else {
            return Err(Fail::Usage("diff needs a golden dump".to_string()));
        };
        (Some(golden), &args[3..])
    } else {
        (None, &args[2..])
    };

    let options = Options::parse(rest)?;
    match command.as_str() {
        "dump" => dump(&snapshot, &options),
        "verify" => run_verify(&snapshot, &options),
        "diff" => diff(&snapshot, golden.as_deref().unwrap(), &options),
        "replay" => replay(&snapshot, &options),
        other => Err(Fail::Usage(format!("unknown command '{other}'"))),
    }
}

struct Options {
    backend: BackendKind,
    runtime_quant: String,
    mxfp4_moe: Mxfp4MoePolicy,
    fused_transcode: bool,
    tp_rank: u32,
    tp_size: u32,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, Fail> {
        let mut args = args.iter().map(String::as_str);
        let backend = match args.next().unwrap_or("cuda") {
            "cuda" => BackendKind::Cuda,
            "metal" => BackendKind::Metal,
            "host" | "dummy" => BackendKind::Unknown,
            other => return Err(Fail::Usage(format!("unknown backend '{other}'"))),
        };
        let runtime_quant = match args.next().unwrap_or("") {
            "-" | "none" => String::new(),
            quant => quant.to_string(),
        };
        let mxfp4_moe = match args.next().unwrap_or("routed") {
            "routed" | "routed_decode" | "auto" => Mxfp4MoePolicy::RoutedDecode,
            "native" | "native_gemm" => Mxfp4MoePolicy::NativeGemm,
            "bf16" | "eager_bf16" => Mxfp4MoePolicy::EagerBf16,
            other => return Err(Fail::Usage(format!("unknown MXFP4 policy '{other}'"))),
        };
        // Not inferred from the backend: a CUDA driver running with
        // PIE_CUDA_DISABLE_FUSED_TRANSCODE=1 compiles a *different* plan, and a
        // tool that could not express that would fail to reproduce it (§2 P2).
        let fused_transcode = match args.next() {
            None => backend == BackendKind::Cuda,
            Some("fused") => true,
            Some("unfused") => false,
            Some(other) => {
                return Err(Fail::Usage(format!(
                    "unknown fusion setting '{other}'; expected 'fused' or 'unfused'"
                )));
            }
        };
        let (tp_rank, tp_size) = match args.next() {
            None => (0, 1),
            Some(spec) => {
                let (rank, size) = spec.split_once('/').ok_or_else(|| {
                    Fail::Usage(format!("TP must be written RANK/SIZE, got '{spec}'"))
                })?;
                let parse = |what: &str, text: &str| {
                    text.parse::<u32>()
                        .map_err(|_| Fail::Usage(format!("TP {what} '{text}' is not a number")))
                };
                (parse("rank", rank)?, parse("size", size)?)
            }
        };
        if tp_size == 0 || tp_rank >= tp_size {
            return Err(Fail::Usage(format!(
                "TP {tp_rank}/{tp_size} is not a rank of a world"
            )));
        }
        Ok(Self {
            backend,
            runtime_quant,
            mxfp4_moe,
            fused_transcode,
            tp_rank,
            tp_size,
        })
    }

    fn target(&self) -> StorageTarget {
        StorageTarget {
            backend: self.backend,
            tp_rank: self.tp_rank,
            tp_size: self.tp_size,
            max_tile_bytes: 64 << 20,
            preferred_alignment: 256,
            tile_map_mask: match self.backend {
                BackendKind::Cuda => CUDA_TILE_MAP_MASK,
                BackendKind::Metal => METAL_TILE_MAP_MASK,
                BackendKind::Unknown => HOST_TILE_MAP_MASK,
            },
            mxfp4_moe: self.mxfp4_moe,
            native_mxfp4_moe: self.mxfp4_moe == Mxfp4MoePolicy::NativeGemm,
            fused_transcode: self.fused_transcode,
        }
    }
}

/// Compile, reporting how long it took and how big the answer is.
///
/// A driver states the model facts in its request; a tool holding only a
/// directory reads them the way the driver's own config parser does.
fn compile(snapshot: &Path, options: &Options) -> Result<(LoadPlan, RuntimeAbi), Fail> {
    let model = parse_model_config(snapshot, options.runtime_quant.clone())?;
    let target = options.target();
    let metadata: CheckpointMetadata = parse_checkpoint_metadata(snapshot)?;
    let started = Instant::now();
    // Kept separate from `compile_snapshot` so `verify` can be handed the
    // contract as well as the plan: checking one against the other is only
    // worth anything when they arrive from different places.
    let abi = RuntimeAbi::default_for_target(&metadata, &model, &target)?;
    let plan = compile_load_plan(&metadata, &abi, target)?;
    eprintln!(
        "compiled {} source tensors into {} runtime tensors and {} instructions in {:?}",
        plan.sources.len(),
        plan.tensors.len(),
        plan.instrs.len(),
        started.elapsed()
    );
    Ok((plan, abi))
}

fn dump(snapshot: &Path, options: &Options) -> Result<(), Fail> {
    let (plan, _) = compile(snapshot, options)?;
    println!("{}", dump_load_plan_json(&plan)?);
    Ok(())
}

fn run_verify(snapshot: &Path, options: &Options) -> Result<(), Fail> {
    let (plan, abi) = compile(snapshot, options)?;
    match verify(
        &PlanView::from(&plan),
        Some(&ContractView::of(&abi, options.tp_size)),
    ) {
        Ok(certificate) => {
            println!("{certificate}");
            Ok(())
        }
        Err(violations) => {
            for violation in &violations {
                eprintln!("violation: {violation}");
            }
            Err(Fail::Failed(format!(
                "{} violation(s); the plan does not honour its contract",
                violations.len()
            )))
        }
    }
}

/// Compare a freshly compiled plan against a stored `dump`.
///
/// The comparison is on the dump text, line by line, because that is the form
/// a golden file is reviewed in: a reader who is shown the differing line can
/// tell whether the change was intended.
fn diff(snapshot: &Path, golden: &Path, options: &Options) -> Result<(), Fail> {
    let (plan, _) = compile(snapshot, options)?;
    let fresh = dump_load_plan_json(&plan)?;
    let stored = std::fs::read_to_string(golden)
        .map_err(|err| Fail::Failed(format!("cannot read {}: {err}", golden.display())))?;
    if fresh.trim() == stored.trim() {
        println!("identical to {}", golden.display());
        return Ok(());
    }
    let mut differences = 0;
    for (line, (a, b)) in stored.lines().zip(fresh.lines()).enumerate() {
        if a != b {
            differences += 1;
            if differences <= 20 {
                println!("line {}:\n  golden: {a}\n  fresh:  {b}", line + 1);
            }
        }
    }
    let (stored_lines, fresh_lines) = (stored.lines().count(), fresh.lines().count());
    if stored_lines != fresh_lines {
        println!("golden has {stored_lines} lines, fresh has {fresh_lines}");
    }
    Err(Fail::Failed(format!(
        "{differences} differing line(s) against {}",
        golden.display()
    )))
}

/// Compile the plan and then actually run it, on the CPU, against the real
/// checkpoint bytes.
///
/// This is the only offline check that can fail because the plan moved the
/// *wrong* bytes rather than an ill-formed number of them.
fn replay(snapshot: &Path, options: &Options) -> Result<(), Fail> {
    let (plan, _) = compile(snapshot, options)?;
    let started = Instant::now();
    let storage = pie_loader::host::execute_plan(&plan, snapshot)?;
    let bytes: usize = storage.tensors.values().map(|t| t.bytes.len()).sum();
    eprintln!(
        "replayed {} tensors ({bytes} bytes materialized, {} arena bytes) in {:?}",
        storage.tensors.len(),
        storage.arena.len(),
        started.elapsed()
    );
    // Names sorted so two runs of the tool are comparable with `diff`.
    let mut names: Vec<&String> = storage.tensors.keys().collect();
    names.sort();
    for name in names {
        let tensor = &storage.tensors[name];
        println!(
            "{name}\t{}\t{:016x}",
            tensor.bytes.len(),
            checksum(&tensor.bytes)
        );
    }
    Ok(())
}

/// FNV-1a over the materialized bytes. Not cryptographic — it exists so two
/// plans that claim to produce the same tensor can be compared cheaply.
fn checksum(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
