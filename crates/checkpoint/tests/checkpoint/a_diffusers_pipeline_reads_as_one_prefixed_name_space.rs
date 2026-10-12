use std::path::{Path, PathBuf};

use checkpoint::file::diffusers;
use checkpoint::file::read::parse_metadata;

fn safetensors(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
    let mut entries = Vec::new();
    let mut cursor = 0usize;
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let dims: Vec<String> = shape.iter().map(u64::to_string).collect();
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            dims.join(","),
            cursor,
            cursor + bytes.len()
        ));
        cursor += bytes.len();
        data.extend_from_slice(bytes);
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, out).unwrap();
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}

fn snapshot(dir: &Path) {
    write(
        &dir.join("model_index.json"),
        r#"{
            "_class_name": "APipeline",
            "_diffusers_version": "0.36.0",
            "boundary_ratio": null,
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "AnEncoder"],
            "tokenizer": ["transformers", "ATokenizer"],
            "transformer": ["diffusers", "ATransformer2DModel"],
            "transformer_2": [null, null],
            "vae": ["diffusers", "AutoencoderKL"]
        }"#,
    );

    safetensors(
        &dir.join("transformer/diffusion_pytorch_model-00001-of-00002.safetensors"),
        &[("norm.weight", "BF16", &[4], &bf16(&[1.0, 2.0, 3.0, 4.0]))],
    );
    safetensors(
        &dir.join("transformer/diffusion_pytorch_model-00002-of-00002.safetensors"),
        &[("blocks.0.attn.q.weight", "F32", &[2, 2], &[0u8; 16])],
    );
    safetensors(
        &dir.join("transformer/diffusion_pytorch_model-00003-of-00003.safetensors"),
        &[("stale.weight", "BF16", &[2], &bf16(&[9.0, 9.0]))],
    );
    write(
        &dir.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
        r#"{"metadata": {}, "weight_map": {
            "norm.weight": "diffusion_pytorch_model-00001-of-00002.safetensors",
            "blocks.0.attn.q.weight": "diffusion_pytorch_model-00002-of-00002.safetensors"
        }}"#,
    );
    write(
        &dir.join("transformer/config.json"),
        r#"{"num_layers": 30}"#,
    );

    safetensors(
        &dir.join("text_encoder/model.safetensors"),
        &[
            ("norm.weight", "BF16", &[4], &bf16(&[5.0, 6.0, 7.0, 8.0])),
            ("embed_tokens.weight", "BF16", &[2, 3], &bf16(&[0.0; 6])),
        ],
    );
    write(
        &dir.join("text_encoder/config.json"),
        r#"{"hidden_size": 2560}"#,
    );

    safetensors(
        &dir.join("vae/diffusion_pytorch_model.safetensors"),
        &[("norm.weight", "BF16", &[4], &bf16(&[9.0, 10.0, 11.0, 12.0]))],
    );
    write(&dir.join("vae/config.json"), r#"{"latent_channels": 16}"#);

    write(
        &dir.join("scheduler/scheduler_config.json"),
        r#"{"num_train_timesteps": 1000}"#,
    );
    write(
        &dir.join("tokenizer/tokenizer_config.json"),
        r#"{"model_max_length": 512}"#,
    );

    safetensors(
        &dir.join("a-pipeline-bundle.safetensors"),
        &[("bundle.only.weight", "BF16", &[2], &bf16(&[42.0, 42.0]))],
    );
    write(&dir.join("README.md"), "# a pipeline\n");
}

fn built() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    snapshot(&root);
    (dir, root)
}

#[test]
fn a_diffusers_pipeline_reads_as_one_prefixed_name_space() {
    let (_guard, root) = built();

    assert!(diffusers::is_pipeline(&root));

    let components = diffusers::components(&root).unwrap();
    let seen: Vec<(&str, &str, usize)> = components
        .iter()
        .map(|c| (c.folder.as_str(), c.prefix.as_str(), c.weights.len()))
        .collect();
    assert_eq!(
        seen,
        vec![
            ("text_encoder", "te.", 1),
            ("transformer", "dit.", 2),
            ("vae", "vae.", 1),
        ]
    );
    assert_eq!(components[1].library, "diffusers");
    assert_eq!(components[1].class, "ATransformer2DModel");
    assert!(components.iter().all(|c| c.config.is_some()));

    assert!(
        components[1]
            .weights
            .iter()
            .all(|path| !path.to_string_lossy().contains("00003")),
        "the index names the set, not the directory listing"
    );

    let source = diffusers::open(&root).unwrap();
    let mut names: Vec<&str> = source.names().collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            "dit.blocks.0.attn.q.weight",
            "dit.norm.weight",
            "te.embed_tokens.weight",
            "te.norm.weight",
            "vae.norm.weight",
        ]
    );

    assert_eq!(
        source.tensor("dit.blocks.0.attn.q.weight").unwrap().shape(),
        &[2, 2]
    );
    assert_eq!(
        source.tensor("te.embed_tokens.weight").unwrap().shape(),
        &[2, 3]
    );
    assert_eq!(
        source
            .tensor("vae.norm.weight")
            .unwrap()
            .bytes()
            .unwrap()
            .into_owned(),
        bf16(&[9.0, 10.0, 11.0, 12.0])
    );

    let metadata = parse_metadata(&root).unwrap();
    let mut described: Vec<&str> = metadata.tensors.iter().map(|t| t.name.as_str()).collect();
    described.sort_unstable();
    assert_eq!(described, names);
    assert_eq!(
        metadata.files.len(),
        4,
        "two transformer shards, one te, one vae"
    );

    let configs = diffusers::configs(&root).unwrap();
    let folders: Vec<&str> = configs.iter().map(|(folder, _)| folder.as_str()).collect();
    assert_eq!(
        folders,
        vec!["", "scheduler", "text_encoder", "transformer", "vae"]
    );
    let transformer = configs.iter().find(|(f, _)| f == "transformer").unwrap();
    assert_eq!(transformer.1, br#"{"num_layers": 30}"#);
    let pipeline = &configs[0].1;
    assert!(String::from_utf8_lossy(pipeline).contains("APipeline"));
}
