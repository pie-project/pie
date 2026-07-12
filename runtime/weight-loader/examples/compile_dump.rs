use std::path::PathBuf;
use std::time::Instant;

use pie_weight_loader::dump::dump_storage_program_json;
use pie_weight_loader::inproc::{compile_snapshot, parse_model_config};
use pie_weight_loader::storage::StorageTarget;
use pie_weight_loader::types::{BackendKind, Mxfp4MoePolicy};

fn main() {
    let mut args = std::env::args().skip(1);
    let snapshot = args.next().map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("usage: compile_dump SNAPSHOT [cuda|metal|dummy] [runtime_quant] [mxfp4_policy]");
        std::process::exit(2);
    });
    let backend = match args.next().as_deref().unwrap_or("cuda") {
        "cuda" => BackendKind::Cuda,
        "metal" => BackendKind::Metal,
        "dummy" => BackendKind::Unknown,
        other => {
            eprintln!("unknown backend '{other}'");
            std::process::exit(2);
        }
    };
    let runtime_quant = args.next().unwrap_or_default();
    let mxfp4_moe = match args.next().as_deref().unwrap_or("routed") {
        "routed" | "routed_decode" | "auto" => Mxfp4MoePolicy::RoutedDecode,
        "native" | "native_gemm" => Mxfp4MoePolicy::NativeGemm,
        "bf16" | "eager_bf16" => Mxfp4MoePolicy::EagerBf16,
        other => {
            eprintln!("unknown MXFP4 policy '{other}'");
            std::process::exit(2);
        }
    };
    let model = parse_model_config(&snapshot, runtime_quant).unwrap_or_else(|error| {
        eprintln!("config parse failed: {error}");
        std::process::exit(1);
    });
    let target = StorageTarget {
        backend,
        tp_rank: 0,
        tp_size: 1,
        max_tile_bytes: 64 << 20,
        preferred_alignment: 256,
        mxfp4_moe,
        native_mxfp4_moe: mxfp4_moe == Mxfp4MoePolicy::NativeGemm,
    };
    let started = Instant::now();
    let program = compile_snapshot(&snapshot, &model, target).unwrap_or_else(|error| {
        eprintln!("compile failed: {error}");
        std::process::exit(1);
    });
    eprintln!(
        "compiled {} source tensors into {} runtime tensors and {} instructions in {:?}",
        program.sources.len(),
        program.tensors.len(),
        program.instrs.len(),
        started.elapsed()
    );
    println!("{}", dump_storage_program_json(&program).unwrap());
}
