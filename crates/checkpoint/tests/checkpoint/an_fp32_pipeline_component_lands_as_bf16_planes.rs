use std::path::PathBuf;

use checkpoint::contract::{Expr, ModelContract, TensorContract};
use checkpoint::executor::Execution;
use checkpoint::executor::sink::MemorySink;
use checkpoint::file::diffusers;
use checkpoint::file::read::parse_metadata;
use checkpoint::plan::{StorageTarget, compile_streaming};
use checkpoint::types::{DType, Encoding};

fn hub() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub")
}

fn snapshot() -> Option<PathBuf> {
    let snapshots = hub().join("models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("model_index.json").is_file())
}

const READ: [(&str, &str, [i64; 2]); 2] = [
    ("x_embedder_bias", "dit.all_x_embedder.2-1.bias", [3840, 1]),
    (
        "final_proj",
        "dit.all_final_layer.2-1.linear.weight",
        [64, 3840],
    ),
];

#[test]
fn an_fp32_pipeline_component_lands_as_bf16_planes() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: no Tongyi-MAI/Z-Image-Turbo snapshot in the HuggingFace cache");
        return;
    };

    let source = diffusers::open(&root).unwrap();
    let metadata = parse_metadata(&root).unwrap();

    for (_, name, _) in READ {
        assert_eq!(
            source.tensor(name).unwrap().nbytes() % 4,
            0,
            "{name} is not a 4-byte-per-element tensor"
        );
        assert_eq!(
            metadata.tensor_by_name(name).unwrap().encoding,
            Encoding::Raw(DType::F32),
            "{name} should be stored fp32"
        );
    }

    let contract = ModelContract {
        alignment: 256,
        tensors: READ
            .iter()
            .map(|(plane, name, shape)| {
                let shape = if shape[1] == 1 {
                    vec![shape[0]]
                } else {
                    shape.to_vec()
                };
                TensorContract::new(
                    *plane,
                    Expr::src(*name).cast(Encoding::Raw(DType::Bf16)),
                    shape,
                    Encoding::Raw(DType::Bf16),
                )
            })
            .collect(),
        groups: Vec::new(),
    };

    let plan = compile_streaming(&metadata, &contract, StorageTarget::default())
        .expect("a bf16 declaration over an fp32 source compiles");
    let mut sink = MemorySink::default();
    Execution::new(&plan, &root)
        .streaming()
        .sink(&mut sink)
        .run()
        .expect("the cast runs on the host");

    for (plane, name, shape) in READ {
        let landed = sink
            .tensors
            .get(plane)
            .unwrap_or_else(|| panic!("{plane} was not published"));
        let elements = shape[0] as usize * shape[1] as usize;
        assert_eq!(landed.len(), elements * 2, "{plane} did not land bf16");
        assert_eq!(
            landed.len() * 2,
            source.tensor(name).unwrap().nbytes() as usize
        );
    }

    let (plane, name, shape) = READ[0];
    let stored = source.tensor(name).unwrap().bytes().unwrap().into_owned();
    let landed = &sink.tensors[plane];
    let mut checked = 0usize;
    for index in 0..shape[0] as usize {
        let from = f32::from_le_bytes(stored[index * 4..index * 4 + 4].try_into().unwrap());
        let to = half::bf16::from_bits(u16::from_le_bytes(
            landed[index * 2..index * 2 + 2].try_into().unwrap(),
        ))
        .to_f32();
        assert_eq!(
            to,
            half::bf16::from_f32(from).to_f32(),
            "{name}[{index}] rounded to something other than its bf16 neighbour"
        );
        checked += 1;
    }
    assert_eq!(checked, 3840);
}
