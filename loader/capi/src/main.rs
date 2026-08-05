//! The loader as a standalone tool.
//!
//! Everything a driver does at load time is reachable here without a GPU, a
//! runtime, or a driver: compile a snapshot, print the plan, check it, compare
//! it against a stored one, and execute it against the real checkpoint bytes.
//!
//! ```text
//! pie-loader dump    SNAPSHOT CONTRACT [options]        the compiled plan, as JSON
//! pie-loader verify  SNAPSHOT CONTRACT [options]        compile, then check the result
//! pie-loader diff    SNAPSHOT CONTRACT GOLDEN [options] compile and compare to a dump
//! pie-loader replay  SNAPSHOT CONTRACT [options]        execute the plan on the host
//! pie-loader convert SNAPSHOT CONTRACT OUT [options]    execute, write a checkpoint
//! ```
//!
//! `CONTRACT` is a JSON [`ModelContract`] — what the tool holds instead of a
//! model's name, because the compiler proper does not have families. In
//! production the contract is authored from a model request on the Rust side
//! (`pie_model::contract`, `plan/model-in-rust.md` §2) and never serialized;
//! this JSON form exists for exactly this tool and the golden tests, and
//! `loader/tests/golden/contracts/` holds the ones the tests use.
//!
//! `convert` is the one command that writes a checkpoint rather than reading
//! one, and what it writes is `.zt`. There used to be a second writer and an
//! `align` command beside it: a streamed weight is read by the device where it
//! lies, through a mapping, so it has to begin on a page of its own, and
//! published safetensors checkpoints put essentially no tensor on a page. That
//! rewrite is gone because its output was, in the end, a checkpoint whose
//! placement only pie relied on -- and `.zt` places every tensor on a page by
//! construction, with no filler tensors invented to express a gap the format
//! has no word for.
//!
//! `replay` is the strongest statement the loader can make about itself
//! offline: it reads the checkpoint through the plan and reports what each
//! runtime tensor actually contains, so a plan that verifies but moves the
//! wrong bytes is still caught.
//!
//! `convert` is `replay` with the output kept: it executes the plan on the
//! host — including runtime quantization, which is why it compiles against
//! [`CONVERT_TILE_MAP_MASK`] rather than the host verification mask — and
//! writes what came out as a new `.zt` checkpoint. The contract is the
//! conversion: whatever it declares (an MXFP4 payload and its scales, a cast,
//! a fold) is what lands on disk, under the names it declares. The expensive
//! encode then happens once, offline, and loading the converted checkpoint is
//! extent writes.
//!
//! Options, positionally, all optional:
//! `BACKEND` (`cuda`|`metal`|`host`), `RUNTIME_QUANT`, `MXFP4_POLICY`
//! (`routed`|`native`|`bf16`), `fused`|`unfused`, `TP_RANK/TP_SIZE`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use pie_loader::checkpoint::CheckpointMetadata;
use pie_loader::checkpoint::read::parse_checkpoint_metadata;
use pie_loader::checkpoint::write::{WriteTensor, write_zt};
use pie_loader::contract::ModelContract;
use pie_loader::dump::dump_load_plan_json;
use pie_loader::error::Error;
use pie_loader::plan::compile as compile_load_plan;
use pie_loader::plan::{
    CONVERT_TILE_MAP_MASK, CUDA_TILE_MAP_MASK, FUSION_FP8_TO_MXFP4, HOST_TILE_MAP_MASK, LoadPlan,
    METAL_TILE_MAP_MASK, StorageTarget,
};
use pie_loader::types::{BackendKind, DType};
use pie_loader::verify::ContractView;
use pie_loader_capi::view::verify_marshalled;

const USAGE: &str = "\
usage: pie-loader <command> SNAPSHOT CONTRACT [BACKEND] [FUSION] [TP] [TARGET]

commands:
  dump    SNAPSHOT CONTRACT          compile and print the plan as JSON
  verify  SNAPSHOT CONTRACT          compile, then check the plan against its contract
  diff    SNAPSHOT CONTRACT GOLDEN   compile and compare against a stored `dump` output
  replay  SNAPSHOT CONTRACT          compile and execute the plan against the checkpoint
  convert SNAPSHOT CONTRACT OUT      execute on the host, write OUT as a new `.zt`

CONTRACT is a JSON ModelContract; see loader/tests/golden/contracts/.

options (positional, all optional):
  BACKEND        cuda | metal | host             (default cuda)
  FUSION         fused | unfused                 (default: fused on cuda)
  TP             RANK/SIZE                       (default 0/1)
  TARGET         a JSON StorageTarget            (default: a CUDA-shaped stand-in)

Pass TARGET to reproduce one driver's plan exactly; the built-in default is a
copy of CUDA's numbers that nothing keeps up to date.
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

impl From<Error> for Fail {
    fn from(error: Error) -> Self {
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

    let Some(contract_path) = args.get(2).map(PathBuf::from) else {
        return Err(Fail::Usage(format!("{command} needs a contract file")));
    };

    // `diff` and `convert` each take one extra positional before the
    // options.
    let (extra, rest) = match command.as_str() {
        "diff" | "convert" => {
            let what = if command == "diff" {
                "a golden dump"
            } else {
                "an output directory"
            };
            let Some(extra) = args.get(3).map(PathBuf::from) else {
                return Err(Fail::Usage(format!("{command} needs {what}")));
            };
            (Some(extra), &args[4..])
        }
        _ => (None, &args[3..]),
    };

    let options = Options::parse(rest)?;
    let contract = read_contract_file(&contract_path)?;
    match command.as_str() {
        "dump" => dump(&snapshot, &contract, &options),
        "verify" => run_verify(&snapshot, &contract, &options),
        "diff" => diff(&snapshot, &contract, extra.as_deref().unwrap(), &options),
        "replay" => replay(&snapshot, &contract, &options),
        "convert" => convert(&snapshot, &contract, extra.as_deref().unwrap(), &options),
        other => Err(Fail::Usage(format!("unknown command '{other}'"))),
    }
}

/// Read the contract to compile.
///
/// The loader cannot produce one: authoring is the driver's job, and this tool
/// is standing in for a driver. Taking it as a file is what makes the tool able
/// to reproduce *any* driver's plan rather than only the families someone
/// remembered to teach it.
fn read_contract_file(path: &Path) -> Result<ModelContract, Fail> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| Fail::Failed(format!("cannot read {}: {err}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|err| Fail::Failed(format!("cannot parse {}: {err}", path.display())))
}

struct Options {
    backend: BackendKind,
    fused_transcode: bool,
    tp_rank: u32,
    tp_size: u32,
    /// A driver's own `StorageTarget`, read from JSON, or none for the built-in
    /// stand-in.
    target: Option<StorageTarget>,
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
        // Not inferred from the backend: a CUDA driver running with
        // PIE_CUDA_DISABLE_FUSED_TRANSCODE=1 compiles a *different* plan, and a
        // tool that could not express that would fail to reproduce it
        // (`architecture.md` §2 P2).
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
        // The driver's own target, when the caller has one. Same argument as
        // `read_contract_file`: a tool that reproduces a driver's plan has to
        // reproduce the driver's *target*, and the only way to be sure is to be
        // handed it rather than to keep a copy of it here.
        let target =
            match args.next() {
                None => None,
                Some(path) => {
                    let text = std::fs::read_to_string(path)
                        .map_err(|err| Fail::Failed(format!("cannot read target {path}: {err}")))?;
                    Some(serde_json::from_str(&text).map_err(|err| {
                        Fail::Failed(format!("cannot parse target {path}: {err}"))
                    })?)
                }
            };
        Ok(Self {
            backend,
            fused_transcode,
            tp_rank,
            tp_size,
            target,
        })
    }

    /// The target to compile against.
    ///
    /// Without a target file this is a *stand-in*, not a claim: the numbers are
    /// CUDA's as of writing, and nothing keeps them equal to
    /// `driver/cuda/src/loader/load_plan.hpp`. They live here rather than in
    /// the library because a device constant inside the compiler is the thing
    /// `architecture.md` §9 removed, and `tests/standalone.rs` enforces that.
    /// Pass a target file to reproduce a specific driver exactly.
    fn target(&self) -> StorageTarget {
        if let Some(target) = &self.target {
            return target.clone();
        }
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
            native_mxfp4_moe: true,
            fusion_mask: if self.fused_transcode {
                FUSION_FP8_TO_MXFP4
            } else {
                0
            },
            encode_scratch_dtype: DType::BF16,
            block_scale_rows: 128,
        }
    }

    /// The target `convert` compiles against: backend-neutral, the host's
    /// transforms plus `Encode`.
    ///
    /// A TARGET file still wins, for a caller reproducing something exact.
    /// BACKEND is deliberately not consulted — the output of a conversion is
    /// bytes for a checkpoint, not for a device, and a plan lowered for a
    /// device would carry transforms (`Repack`) whose output has no on-disk
    /// representation.
    fn convert_target(&self) -> StorageTarget {
        if let Some(target) = &self.target {
            return target.clone();
        }
        StorageTarget {
            backend: BackendKind::Unknown,
            tp_rank: self.tp_rank,
            tp_size: self.tp_size,
            max_tile_bytes: 64 << 20,
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        }
    }
}

/// Compile, reporting how long it took and how big the answer is.
///
/// A driver states the model facts in its request; a tool holding only a
/// directory reads them the way the driver's own config parser does.
fn compile(snapshot: &Path, contract: &ModelContract, options: &Options) -> Result<LoadPlan, Fail> {
    let target = options.target();
    let metadata: CheckpointMetadata = parse_checkpoint_metadata(snapshot)?;
    let started = Instant::now();
    let plan = compile_load_plan(&metadata, contract, target)?;
    eprintln!(
        "compiled {} source tensors into {} runtime tensors and {} instructions in {:?}",
        plan.sources.len(),
        plan.tensors.len(),
        plan.instrs.len(),
        started.elapsed()
    );
    Ok(plan)
}

fn dump(snapshot: &Path, contract: &ModelContract, options: &Options) -> Result<(), Fail> {
    let plan = compile(snapshot, contract, options)?;
    println!("{}", dump_load_plan_json(&plan)?);
    Ok(())
}

fn run_verify(snapshot: &Path, contract: &ModelContract, options: &Options) -> Result<(), Fail> {
    let plan = compile(snapshot, contract, options)?;
    match verify_marshalled(&plan, Some(&ContractView::of(contract))) {
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
fn diff(
    snapshot: &Path,
    contract: &ModelContract,
    golden: &Path,
    options: &Options,
) -> Result<(), Fail> {
    let plan = compile(snapshot, contract, options)?;
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
fn replay(snapshot: &Path, contract: &ModelContract, options: &Options) -> Result<(), Fail> {
    let plan = compile(snapshot, contract, options)?;
    let started = Instant::now();
    let storage = pie_loader::executor::host::execute_plan(&plan, snapshot)?;
    let bytes: usize = storage.tensors.values().map(Vec::len).sum();
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
            tensor.len(),
            pie_loader::cache_key::fnv1a(tensor)
        );
    }
    Ok(())
}

/// Execute the plan on the host and keep the output: write every public
/// runtime tensor, in plan order, as a new `.zt` checkpoint under `out`.
///
/// The contract *is* the conversion. It names the output tensors, and its
/// expressions say what each one is made of — a `Cast` into a quantized
/// encoding is how "quantize this offline" is spelled, and the scales that
/// encode publishes arrive as the extra declarations the plan already carries.
/// Loading the result afterwards is a different, cheaper contract against the
/// converted names.
///
/// Whole-model in memory: the host executor materializes the persistent arena
/// before anything is written, so converting a checkpoint takes roughly its
/// decoded size in RAM. Streaming the write per tensor is possible if that
/// ever binds, but it means teaching the executor to retire buffers early —
/// not worth it before someone hits the wall.
fn convert(
    snapshot: &Path,
    contract: &ModelContract,
    out: &Path,
    options: &Options,
) -> Result<(), Fail> {
    if out.exists() {
        return Err(Fail::Failed(format!(
            "{} already exists; convert writes a fresh checkpoint and is not in a \
             position to know what an existing one is",
            out.display()
        )));
    }
    if !contract.groups.is_empty() {
        return Err(Fail::Failed(
            "this contract declares groups, which convert does not run: a group \
             compiles to one template plus bindings, not to resident tensors. \
             Declare the members as plain tensors for conversion"
                .to_string(),
        ));
    }
    let metadata = parse_checkpoint_metadata(snapshot)?;
    let plan = compile_load_plan(&metadata, contract, options.convert_target())?;
    let started = Instant::now();
    let storage = pie_loader::executor::host::execute_plan(&plan, snapshot)?;
    eprintln!(
        "executed {} instructions ({} arena bytes) in {:?}",
        plan.instrs.len(),
        storage.arena.len(),
        started.elapsed()
    );

    let mut tensors = Vec::new();
    for decl in &plan.tensors {
        if !decl.visibility.is_public() {
            continue;
        }
        let bytes = storage.tensors.get(&decl.name).ok_or_else(|| {
            Fail::Failed(format!(
                "the plan declares '{}' but executing it produced no such tensor",
                decl.name
            ))
        })?;
        tensors.push(WriteTensor { decl, bytes });
    }
    if tensors.is_empty() {
        return Err(Fail::Failed(
            "the contract declares no public tensors, so there is nothing to write".to_string(),
        ));
    }

    // Provenance, not meaning: enough to tell two conversions apart and name
    // what produced this one. Digests rather than paths so the file does not
    // record one machine's directory layout.
    let mut provenance = std::collections::BTreeMap::new();
    provenance.insert(
        "pie_convert_compiler".to_string(),
        pie_loader::plan::compiler_version().to_string(),
    );
    let contract_json = serde_json::to_vec(contract)
        .map_err(|err| Fail::Failed(format!("cannot serialize the contract: {err}")))?;
    provenance.insert(
        "pie_convert_contract".to_string(),
        format!("{:016x}", pie_loader::cache_key::fnv1a(&contract_json)),
    );
    let source = plan
        .files
        .iter()
        .map(|file| {
            let name = Path::new(&file.path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&file.path);
            format!("{name}:{}", file.size_bytes)
        })
        .collect::<Vec<_>>()
        .join(",");
    provenance.insert(
        "pie_convert_source".to_string(),
        format!("{:016x}", pie_loader::cache_key::fnv1a(source.as_bytes())),
    );

    // `.zt` is the only output: it can name every scheme the plan language
    // admits, places tensors on pages without inventing filler tensors to
    // express a gap, and carries a digest.
    let path = out.join("model.zt");
    write_zt(&path, &provenance, &tensors)?;
    let bytes: u64 = tensors.iter().map(|tensor| tensor.bytes.len() as u64).sum();
    println!(
        "{}: {} tensors, {bytes} bytes",
        path.display(),
        tensors.len()
    );
    Ok(())
}
