use std::path::{Path, PathBuf};

use checkpoint::file::diffusers;

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
    let snapshots = hub().join("models--MiniMaxAI--MiniMax-H3/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("FL2VA/model_index.json").is_file())
}

fn roster(dir: &Path) -> Vec<(String, String, usize)> {
    let mut seen: Vec<(String, String, usize)> = diffusers::components(dir)
        .unwrap()
        .into_iter()
        .map(|c| (c.folder, c.prefix, c.weights.len()))
        .collect();
    seen.sort();
    seen
}

#[test]
fn the_minimax_h3_snapshot_finds_its_video_vae_a_folder_down() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: no MiniMaxAI/MiniMax-H3 snapshot in the HuggingFace cache");
        return;
    };

    for partition in ["FL2VA", "Ref2VA"] {
        let dir = root.join(partition);
        let seen = roster(&dir);
        let named: Vec<(&str, &str, usize)> = seen
            .iter()
            .map(|(f, p, n)| (f.as_str(), p.as_str(), *n))
            .collect();
        assert_eq!(
            named,
            vec![
                ("audio_vae", "avae.", 1),
                ("text_encoder", "te.", 14),
                ("transformer", "dit.", 13),
                ("video_vae", "vae.", 1),
            ],
            "{partition}: the video VAE is the component the flat discovery missed"
        );

        let components = diffusers::components(&dir).unwrap();
        let vae = components
            .iter()
            .find(|c| c.folder == "video_vae")
            .expect("the video VAE is in the set");
        assert_eq!(
            vae.weights,
            vec![dir.join("video_vae/source/model.safetensors")]
        );
        assert_eq!(vae.dir, dir.join("video_vae"));
        assert!(
            !dir.join("video_vae/model.safetensors").exists()
                && !dir
                    .join("video_vae/diffusion_pytorch_model.safetensors")
                    .exists(),
            "{partition}: the component folder itself holds no set — which is \
             why it was skipped before the descent"
        );
    }

    assert!(
        roster(&root).is_empty(),
        "the modular top-level index is not read by component discovery"
    );
    assert!(
        root.join("vae/diffusion_pytorch_model.safetensors.index.json")
            .is_file(),
        "and not because the modular pipeline holds no weights"
    );

    let source = diffusers::open(&root.join("FL2VA")).unwrap();
    let vae_names = source.names().filter(|n| n.starts_with("vae.")).count();
    assert!(
        vae_names > 0,
        "the video VAE contributes tensors to the merged name space"
    );
    eprintln!("FL2VA: {vae_names} tensors under `vae.`");
}
