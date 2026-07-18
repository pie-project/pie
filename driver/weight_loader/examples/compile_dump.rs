use std::time::Instant;

use pie_weight_loader::abi::{RuntimeAbi, RuntimeTensorContract, RuntimeTensorSource};
use pie_weight_loader::config::ModelConfig;
use pie_weight_loader::dump::dump_storage_program_json;
use pie_weight_loader::source::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_weight_loader::storage::StorageTarget;
use pie_weight_loader::storage_compiler::compile_storage_program;
use pie_weight_loader::types::{
    BackendKind, CheckpointFormat, DType, Encoding, FileId, Layout, Mxfp4MoePolicy, Sharding,
    TensorId,
};

fn main() {
    let tensors = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(32);
    let metadata = synthetic_metadata(tensors);
    let abi = synthetic_abi(tensors);
    let cfg = ModelConfig {
        model_type: "qwen3".to_string(),
        quant_method: String::new(),
        runtime_quant: String::new(),
        num_hidden_layers: tensors,
        num_experts: 0,
        num_experts_per_tok: 0,
        ..Default::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank: 0,
        tp_size: 1,
        max_tile_bytes: 64 << 20,
        preferred_alignment: 256,
        mxfp4_moe: Mxfp4MoePolicy::RoutedDecode,
        native_mxfp4_moe: false,
    };

    let start = Instant::now();
    let program = compile_storage_program(&metadata, &cfg, &abi, target).unwrap();
    let elapsed = start.elapsed();
    eprintln!(
        "compiled {} tensors into {} instructions in {:?}",
        tensors,
        program.instrs.len(),
        elapsed
    );
    println!("{}", dump_storage_program_json(&program).unwrap());
}

fn synthetic_metadata(tensors: u32) -> CheckpointMetadata {
    let mut raw_tensors = Vec::new();
    for id in 0..tensors {
        raw_tensors.push(RawTensor {
            id: TensorId(id),
            name: format!("model.layers.{id}.weight"),
            file_id: FileId(0),
            file_offset: u64::from(id) * 8192,
            span_bytes: 8192,
            shape: vec![64, 64],
            encoding: Encoding::Raw(DType::BF16),
            layout: Layout::dense(1),
        });
    }
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "synthetic.safetensors".to_string(),
            size_bytes: u64::from(tensors) * 8192,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: raw_tensors,
    }
}

fn synthetic_abi(tensors: u32) -> RuntimeAbi {
    RuntimeAbi {
        name: "pie-cuda-synthetic".to_string(),
        version: 1,
        tensors: (0..tensors)
            .map(|id| RuntimeTensorContract {
                output_name: format!("runtime.layers.{id}.weight"),
                source: RuntimeTensorSource::DirectTensor(TensorId(id)),
                metadata: Vec::new(),
                dtype: DType::BF16,
                encoding: Encoding::Raw(DType::BF16),
                shape: vec![64, 64],
                layout: Layout::dense(256),
                sharding: Sharding::replicated(),
                alignment: 256,
                shard_axis: None,
            })
            .collect(),
    }
}
