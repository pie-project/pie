use std::path::PathBuf;

use checkpoint::file::diffusers;
use checkpoint::file::read::parse_metadata;

fn hub() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub")
}

pub fn snapshot() -> Option<PathBuf> {
    let snapshots = hub().join("models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("model_index.json").is_file())
}

#[test]
fn the_z_image_snapshot_opens_as_one_prefixed_source() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: no Tongyi-MAI/Z-Image-Turbo snapshot in the HuggingFace cache");
        return;
    };

    let components = diffusers::components(&root).unwrap();
    let seen: Vec<(&str, &str, usize)> = components
        .iter()
        .map(|c| (c.folder.as_str(), c.prefix.as_str(), c.weights.len()))
        .collect();
    assert_eq!(
        seen,
        vec![
            ("text_encoder", "te.", 3),
            ("transformer", "dit.", 3),
            ("vae", "vae.", 1),
        ],
        "the pipeline's weight-bearing components; `scheduler` and `tokenizer` hold none"
    );

    let source = diffusers::open(&root).unwrap();

    let mut counts = [0usize; 3];
    for name in source.names() {
        match name {
            n if n.starts_with("dit.") => counts[0] += 1,
            n if n.starts_with("te.") => counts[1] += 1,
            n if n.starts_with("vae.") => counts[2] += 1,
            other => panic!("{other} is under no component prefix"),
        }
    }
    assert_eq!(counts[0], 521, "the transformer's index names 521 tensors");
    assert!(counts[1] > 0 && counts[2] > 0);

    for (name, shape) in [
        ("dit.all_x_embedder.2-1.weight", vec![3840u64, 64]),
        ("dit.all_x_embedder.2-1.bias", vec![3840]),
        ("dit.all_final_layer.2-1.linear.weight", vec![64, 3840]),
        ("te.model.embed_tokens.weight", vec![151936, 2560]),
        ("vae.decoder.conv_in.weight", vec![512, 16, 3, 3]),
    ] {
        let tensor = source
            .tensor(name)
            .unwrap_or_else(|why| panic!("{name}: {why}"));
        assert_eq!(tensor.shape().to_vec(), shape, "{name}");
    }

    assert!(source.get("all_x_embedder.2-1.weight").is_none());
    assert!(source.get("model.embed_tokens.weight").is_none());

    let metadata = parse_metadata(&root).unwrap();
    assert_eq!(metadata.tensors.len(), source.len());
    assert_eq!(
        metadata.files.len(),
        7,
        "3 dit shards + 3 te shards + 1 vae"
    );
    let embedder = metadata
        .tensor_by_name("dit.all_x_embedder.2-1.weight")
        .expect("the prefixed name is the one the metadata carries");
    assert_eq!(embedder.shape, vec![3840, 64]);
    assert_eq!(
        embedder.encoding,
        checkpoint::types::Encoding::Raw(checkpoint::types::DType::F32),
        "Z-Image ships its transformer in fp32"
    );
}
