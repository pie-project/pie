use std::path::{Path, PathBuf};

use checkpoint::file::diffusers;

fn safetensors(path: &Path, name: &str) {
    let bytes = half::bf16::from_f32(1.0).to_bits().to_le_bytes();
    let header = format!(
        "{{\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,{}]}}}}",
        bytes.len()
    );
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&bytes);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, out).unwrap();
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn snapshot(dir: &Path) {
    write(
        &dir.join("model_index.json"),
        r#"{
            "_class_name": "APipeline",
            "audio_vae": ["diffusers", "AnAudioVAE"],
            "image_encoder": ["diffusers", "AnImageEncoder"],
            "transformer": ["diffusers", "ATransformer"],
            "video_vae": ["diffusers", "AVideoVAE"]
        }"#,
    );

    safetensors(
        &dir.join("transformer/diffusion_pytorch_model.safetensors"),
        "beside.weight",
    );
    safetensors(
        &dir.join("transformer/original/model.safetensors"),
        "below.weight",
    );
    write(&dir.join("transformer/config.json"), r#"{"layers": 2}"#);

    write(&dir.join("video_vae/vae_cnn.py"), "# a module\n");
    write(&dir.join("video_vae/config.json"), r#"{"z": 32}"#);
    write(&dir.join("video_vae/source/config.json"), r#"{"z": 32}"#);
    safetensors(
        &dir.join("video_vae/source/model.safetensors"),
        "cnn.weight",
    );

    write(&dir.join("image_encoder/config.json"), r#"{"dim": 8}"#);
    safetensors(
        &dir.join("image_encoder/checkpoints/step-100/model.safetensors"),
        "deep.weight",
    );

    write(&dir.join("audio_vae/config.json"), r#"{"bins": 64}"#);
    safetensors(&dir.join("audio_vae/.cache/model.safetensors"), "cached");

    safetensors(&dir.join("a-pipeline-bundle.safetensors"), "bundle.weight");
}

fn built() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    snapshot(&root);
    (dir, root)
}

#[test]
fn a_component_that_keeps_its_weights_a_folder_down_is_still_read() {
    let (_guard, root) = built();

    let components = diffusers::components(&root).unwrap();
    let mut seen: Vec<(&str, &str, usize)> = components
        .iter()
        .map(|c| (c.folder.as_str(), c.prefix.as_str(), c.weights.len()))
        .collect();
    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![("transformer", "dit.", 1), ("video_vae", "vae.", 1)],
        "the transformer's own file and the video VAE's file a folder down; \
         two levels down and a hidden folder are not a component's weights, \
         and a component with neither is skipped"
    );

    let named = |folder: &str| {
        components
            .iter()
            .find(|c| c.folder == folder)
            .unwrap_or_else(|| panic!("{folder} is in the set"))
    };
    let vae = named("video_vae");
    assert_eq!(
        vae.weights,
        vec![root.join("video_vae/source/model.safetensors")]
    );
    assert_eq!(vae.dir, root.join("video_vae"));
    assert_eq!(vae.config, Some(root.join("video_vae/config.json")));

    assert_eq!(
        named("transformer").weights,
        vec![root.join("transformer/diffusion_pytorch_model.safetensors")]
    );

    let source = diffusers::open(&root).unwrap();
    let mut names: Vec<&str> = source.names().collect();
    names.sort_unstable();
    assert_eq!(names, vec!["dit.beside.weight", "vae.cnn.weight"]);
}
