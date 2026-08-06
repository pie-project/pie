//! Booting the standalone from a `.zt` artifact — the end-to-end gate for the
//! whole conversion path, on a machine with no GPU.
//!
//! `boot_smoke.rs` boots the same stack from a HuggingFace snapshot: a
//! directory of `tokenizer.json`, `config.json` and a safetensors file, each
//! found by name at boot. This one converts that snapshot into a single `.zt`
//! first and boots from *that*, which exercises every piece the artifact
//! introduced:
//!
//!   * `weights::resolve` taking a `.zt` path and reporting an artifact;
//!   * the compiled tokenizer read back out of `__meta__/tokenizer/*` and
//!     rebuilt without `tokenizer.json` existing anywhere;
//!   * `vocab_size` and `num_hidden_layers` read from the `pie.model/1`
//!     descriptor rather than probed out of a `config.json`;
//!   * the descriptor written beside the driver's startup TOML.
//!
//! The proof that this is not just "it booted" is negative: the artifact is
//! the *only* file in its directory. There is no `tokenizer.json` and no
//! `config.json` to fall back to, so a boot that reached for either would fail
//! rather than quietly pass.
//!
//! **One boot per test process**, same as `boot_smoke.rs` — the runtime owns
//! process-global singletons and the dummy driver grabs a fixed POSIX shmem —
//! which is why this is its own test binary rather than a case added there.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};
use pie_bin::derive::derive_standalone;
use pie_bin::run_standalone;

/// The same standalone TOML `boot_smoke` uses, pointed at an artifact.
///
/// `model` takes a path here rather than a store name so the test does not
/// depend on `PIE_HOME`; the store lookup is covered by `weights`' own tests.
fn standalone_toml(artifact: &str) -> String {
    format!(
        "[server]\n\
         port = 0\n\
         \n\
         [model]\n\
         name = \"artifact-smoke\"\n\
         model = \"{artifact}\"\n\
         \n\
         [driver]\n\
         type = \"dummy\"\n\
         device = [\"cpu\"]\n\
         vocab_size = 256\n\
         arch_name = \"qwen3\"\n"
    )
}

/// Builds the snapshot `boot_smoke` uses, then converts it to one `.zt`.
///
/// Returns the artifact path. Its directory holds nothing else — see the
/// module docs for why that matters.
fn fixture_artifact() -> String {
    static ARTIFACT: OnceLock<(tempfile::TempDir, PathBuf)> = OnceLock::new();
    let (_dir, artifact) = ARTIFACT.get_or_init(|| {
        let fixtures =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/smoke-model-ascii");

        // The snapshot to convert *from*, in its own directory.
        let staging = tempfile::tempdir().expect("create staging dir");
        for name in ["tokenizer.json", "config.json"] {
            std::fs::copy(fixtures.join(name), staging.path().join(name))
                .unwrap_or_else(|err| panic!("copy {name}: {err}"));
        }
        // The dummy load planner still wants one valid safetensors header even
        // though it never reads real weights.
        let header =
            r#"{"model.embed_tokens.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut checkpoint = (header.len() as u64).to_le_bytes().to_vec();
        checkpoint.extend_from_slice(header.as_bytes());
        checkpoint.extend_from_slice(&[1, 2, 3, 4]);
        std::fs::write(staging.path().join("model.safetensors"), checkpoint)
            .expect("write smoke checkpoint");

        // Convert into a directory of its own, so what boots is provably the
        // artifact and not the files it was made from.
        let store = tempfile::tempdir().expect("create artifact dir");
        let artifact = store.path().join("smoke.zt");
        pie_bin::ops::model::import::run(pie_bin::ops::model::import::ImportArgs {
            source: staging.path().to_string_lossy().into_owned(),
            out: Some(artifact.clone()),
            dry_run: false,
            force: true,
            delete_source: false,
            max_shard_size: None,
        })
        .expect("convert the smoke snapshot");

        assert_only_the_artifact(store.path(), &artifact);
        (store, artifact)
    });
    artifact.to_string_lossy().into_owned()
}

/// The artifact is alone in its directory: nothing to fall back to.
fn assert_only_the_artifact(dir: &Path, artifact: &Path) {
    let names: Vec<String> = std::fs::read_dir(dir)
        .expect("read artifact dir")
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec![artifact.file_name().unwrap().to_string_lossy().into_owned()],
        "the artifact must be the only file here, or this test proves nothing"
    );
}

async fn boot() -> Result<pie_bin::StandaloneHandle> {
    let (controller, gateway, worker) = derive_standalone(&standalone_toml(&fixture_artifact()))?;
    run_standalone(controller, gateway, worker).await
}

/// THE GATE — converts, boots once from the artifact alone, round-trips a ping
/// through the real client path, and shuts down.
///
/// A ping is enough for what this test is for. It is not re-checking the
/// runtime's request path (`boot_smoke` does that); it is checking that a model
/// whose tokenizer and config exist *only* inside a `.zt` gets as far as
/// serving requests at all — which means `register` built a tokenizer from the
/// compiled objects and took its vocab and layer count from the descriptor.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_standalone_serves_a_model_that_exists_only_as_an_artifact() -> Result<()> {
    let pie = boot().await?;

    assert_ne!(
        pie.listen_addr.port(),
        0,
        "client-facing edge must bind a real ephemeral port"
    );

    // A ping through ingress: the full client path, no tokenization of its own.
    let payload = serde_json::to_vec(&serde_json::json!({ "type": "ping", "corr_id": 1 }))?;
    let resp = tokio::time::timeout(
        Duration::from_secs(20),
        reqwest::Client::new()
            .post(format!("http://{}/v1/generate", pie.listen_addr))
            .header("x-pie-identity", "artifact/test")
            .header("content-type", "application/json")
            .body(payload)
            .send(),
    )
    .await
    .context("REST ping timed out")??;
    assert_eq!(resp.status(), 200, "ingress one-shot must accept the turn");

    let body = tokio::time::timeout(Duration::from_secs(20), resp.text())
        .await
        .context("REST ping body timed out")??;
    assert!(
        body.contains("[DONE]"),
        "the turn must stream back and reach the clean [DONE] sentinel; got: {body}"
    );

    pie.shutdown().await;
    Ok(())
}
