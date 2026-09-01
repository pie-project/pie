//! **THE SERVING ARTIFACT AS A SPILL SOURCE, ON A DEVICE** (§M-4d step 1).
//!
//! `experts::Spill::Serving` lets a boot read its planes out of the model's
//! own `.zt` instead of out of a tier file beside it, and it is engaged only
//! under two conditions AT ONCE:
//!
//! * the checkpoint is a SERVING artifact — a raw snapshot never reaches it,
//!   so `cuda_boot_smoke`, which serves one, exercises the FALLBACK and not
//!   this;
//! * the load SPILLS (`plan.spill_demand() > 0`) — a model that fits its
//!   budget whole never reaches it either, and a 0.8B on a 46 GiB card fits.
//!
//! Neither is the default, which is why this is its own gate: a flag on the
//! boot smoke that let either condition lapse would go green over the old
//! road and say nothing.
//!
//! # The boot IS the assertion
//!
//! A spilled load asks its source for every plane it did not keep resident. If
//! `Spill::Serving` addressed them wrongly — by position instead of by name,
//! which is exactly what the two arms it replaces do — the load refuses at the
//! first hole with `Spill::remedy`'s sentence. So a boot that COMPLETES under
//! a forced spill from an artifact is the proof; the ping after it is the
//! boot smoke's own check that the turn path is alive.
//!
//! # What it needs, and why it is `#[ignore]`d
//!
//! An imported artifact, 1.6 GiB for the 0.8B, which is not something to build
//! inside a test run:
//!
//! ```text
//! pie model import ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<rev> \
//!     --out /tmp/art --no-prepare
//! PIE_SERVING_ARTIFACT=/tmp/art/<the .zt it wrote> \
//!   cargo test -p pie-gpu-tests --features engine-cuda-13 \
//!   --test cuda_serving_spill -- --ignored --nocapture
//! ```
//!
//! The env var rather than a fixed path because an artifact's NAME states its
//! specialization (§M-4) — `<slug>.<sku>.<backend>-tp<n>.<precision>.zt` —
//! so there is no constant to hardcode. That is the naming working, not an
//! inconvenience.

mod common;

use anyhow::{Context, Result, bail};

/// A budget under the model's weight table, so the load must spill.
///
/// 600 MiB against a 1.6 GiB table: enough of the table is left over that a
/// source is genuinely read, and not so tight that `Residency::admit` refuses
/// the plan before any of this is reached.
const DEVICE_WEIGHT_BUDGET: &str = "600MiB";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a CUDA device and PIE_SERVING_ARTIFACT naming an imported .zt"]
async fn a_spilled_load_reads_its_planes_out_of_the_serving_artifact() -> Result<()> {
    common::init_trace();
    let Ok(artifact) = std::env::var("PIE_SERVING_ARTIFACT") else {
        bail!("PIE_SERVING_ARTIFACT is unset; this file's header says how to make one");
    };
    if !std::path::Path::new(&artifact).is_file() {
        bail!("PIE_SERVING_ARTIFACT names {artifact}, which is not a file");
    }

    // The budget rides `[model]`, where `worker::config::ModelConfig` reads it.
    let toml = common::serving_standalone_toml(&artifact).replace(
        "\n[engine]",
        &format!("device_weight_budget = \"{DEVICE_WEIGHT_BUDGET}\"\n\n[engine]"),
    );
    let pie = common::boot_from_toml(&toml)
        .await
        .context("a spilled load out of a serving artifact must boot")?;

    let payload = serde_json::to_vec(&serde_json::json!({ "type": "ping", "corr_id": 1 }))?;
    let resp = reqwest::Client::new()
        .post(format!("http://{}/v1/generate", pie.listen_addr))
        .header("x-pie-identity", "serving-spill/test")
        .header("content-type", "application/json")
        .body(payload)
        .send()
        .await?;
    assert_eq!(resp.status(), 200, "ingress must accept the turn");
    assert!(
        resp.text().await?.contains("[DONE]"),
        "the turn must stream back to [DONE]"
    );
    Ok(())
}
