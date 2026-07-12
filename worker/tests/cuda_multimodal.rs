//! Real-hardware multimodal validation — image (vision) splice path on
//! `cuda_native`.
//!
//! Proves the forward-pass `input-image` splice path end-to-end on real silicon:
//! the inferlet hands the host raw encoded image bytes, the host runs the bound
//! model's vision tower driver-side (`gemma4_vision_forward.cu`), scatters the
//! projected soft-token rows into the hidden state, commits them as ordinary KV
//! through the working-set forward txn, and a coherent text answer proves the
//! spliced visual span actually conditioned generation. Structurally supported
//! since the WIT `media` interface; this is the first end-to-end exercise on the
//! merged WASI-P3 single-model tree (the `pass.input_image(...) → execute()`
//! path the P3 forward-path port carries).
//!
//! Hermetic by construction: `image-qa-bench` takes the image as base64 in its
//! input (NO http fetch / loopback server / network-allow), and `wait_for_start`
//! defaults off so it runs straight through `spawn_input`. The local bench image
//! (`benches/assets/bench_image.png`) is the only asset needed.
//!
//! Needs a multimodal model — `gemma-4-E4B` is the ONLY driver-supported vision
//! model cached (gemma3n has no vision forward; qwen3_vl isn't cached). Snapshot
//! overridable via `PIE_CUDA_TEST_MM_SNAPSHOT`.
//!
//! ⚠️ VRAM-BLOCKED on a 24G GPU: gemma-4-E4B's 15G weights + the vision-encoder
//! activation workspace exceed a 24G 4090 even at the max-fit config (gpu_mem
//! 0.97 + fp8 KV → ~2545 MiB planner budget, still no viable forward layout).
//! The weight-halving lever (`runtime_quant=fp8`) is unimplemented for gemma4 in
//! the load planner, so it can't free the weight side. The splice CODE is
//! shipped + complete (this builds green vs the SDK + drives the real path);
//! hardware validation needs a >24G GPU (or a driver-supported smaller / fp8
//! vision checkpoint). Run there:
//!   cargo test -p pie-worker --features driver-cuda --test cuda_multimodal -- --ignored --nocapture

mod common;

use std::path::PathBuf;

/// Local `gemma-4-E4B` HF snapshot (vision + audio). Override with
/// `PIE_CUDA_TEST_MM_SNAPSHOT=/path/to/snapshot`.
const DEFAULT_MM_SNAPSHOT: &str = "/home/ingim/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/7aa32e6889efd6300124851b164f8b364314c3d8";

fn mm_snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_MM_SNAPSHOT").unwrap_or_else(|_| DEFAULT_MM_SNAPSHOT.to_string())
}

/// Max-fit single-model cuda config for the 15G gemma-4-E4B vision model. The
/// default 0.90 / bf16 / auto-KV layout leaves only ~857 MiB after weights +
/// encoders; this squeezes the two axes the cuda planner accepts:
///   * `gpu_mem_utilization = 0.97`   — +~1.7G headroom (→ ~2545 MiB budget).
///   * `kv_cache_dtype = "fp8"`       — halves the KV cache (negligible here —
///     KV is tiny at batch=1/short-seq; the wall is weights + encoder workspace).
/// Even so this does NOT fit a forward layout on a 24G 4090 (see module note):
/// the weight-halving lever (`runtime_quant=fp8`) is unimplemented for gemma4, so
/// the 15G weight side can't be freed. Runs on a >24G GPU.
fn mm_cuda_toml(snapshot_path: &str) -> String {
    let scratch = std::env::temp_dir().join("pie-cuda-mm-scratch");
    let _ = std::fs::create_dir_all(&scratch);
    format!(
        "[server]\n\
         host = \"127.0.0.1\"\n\
         port = 0\n\n\
         [runtime]\n\
         allow_fs = true\n\
         fs_scratch_dir = \"{scratch}\"\n\n\
         [auth]\n\
         enabled = false\n\n\
         [model]\n\
         name = \"default\"\n\
         hf_repo = \"{snapshot}\"\n\n\
         [model.driver]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\n\
         [model.driver.options]\n\
         gpu_mem_utilization = 0.97\n\
         memory_profile = \"latency\"\n\
         kv_cache_dtype = \"fp8\"\n",
        scratch = scratch.display(),
        snapshot = snapshot_path,
    )
}

/// Boot the embedded cuda engine with the multimodal model under the tight
/// [`mm_cuda_toml`] fit config.
async fn boot_mm_cuda(snapshot_path: &str) -> pie_worker::WorkerHandle {
    let cfg = pie_worker::Config::parse(&mm_cuda_toml(snapshot_path))
        .expect("parse mm cuda worker config");
    pie_worker::run(cfg)
        .await
        .expect("boot embedded cuda engine (gemma-4-E4B)")
}

/// Standard base64 (RFC 4648, padded) — matches `image-qa-bench`'s self-contained
/// `b64_decode` (accepts padded/unpadded, whitespace-tolerant).
fn b64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Pull an integer field out of a flat JSON object string without a serde dep.
fn extract_u64(json: &str, field: &str) -> Option<u64> {
    let key = format!("\"{field}\":");
    let start = json.find(&key)? + key.len();
    let digits: String = json[start..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Vision splice: encode a local image with the bound model's vision tower,
/// splice the soft-token KV, then answer about it with ordinary text generation.
#[test]
#[ignore = "real-hardware + VRAM-blocked on 24G: gemma-4-E4B (15G) + vision encoder exceeds a 24G 4090 (max-fit gpu_mem 0.97 + fp8 KV still yields no viable forward layout); needs a >24G GPU. Splice code is shipped + builds green. Run with --features driver-cuda + a local gemma-4-E4B snapshot on a >24G GPU."]
fn cuda_native_image_splice_conditions_generation() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // (1) Boot the embedded cuda engine with the multimodal model under the
        //     tight fit config (gemma-4-E4B is 15G — see `mm_cuda_toml`).
        let worker = boot_mm_cuda(&mm_snapshot()).await;
        eprintln!("[cuda_multimodal] gemma-4-E4B up on {}", worker.url());

        // (2) Local bench image → base64 (no network).
        let img_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../benches/assets/bench_image.png");
        let img_bytes = std::fs::read(&img_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", img_path.display()));
        let img_b64 = b64_encode(&img_bytes);
        eprintln!(
            "[cuda_multimodal] image {} bytes -> {} b64 chars",
            img_bytes.len(),
            img_b64.len()
        );

        // (3) Drive image-qa-bench in-proc. Greedy (temp 0) for determinism;
        //     return_text so we can eyeball coherence in --nocapture.
        let input = format!(
            r#"{{"image_b64":"{img_b64}","question":"What is in this image? Answer in one sentence.","system":"You are a helpful visual assistant.","max_tokens":32,"temperature":0.0,"return_text":true}}"#
        );
        let program = common::install_inferlet("image-qa-bench").await;
        let result = common::spawn_input(&program, &input).await;
        eprintln!("[cuda_multimodal] image RESULT = {result:?}");

        let out = result.expect("image-qa-bench errored on cuda (vision splice path)");

        // (4a) Generation ran: the spliced image conditioned a coherent forward
        //      → multi-token decode. (Surfaces any host/forward error instead of
        //      a silent "completed".)
        let n_out = extract_u64(&out, "num_output_tokens")
            .unwrap_or_else(|| panic!("no num_output_tokens in result: {out}"));
        assert!(n_out > 0, "vision splice forward decoded no tokens: {out}");

        // (4b) The image was ACTUALLY spliced: the prompt carries the vision
        //      soft-token rows (gemma SigLIP2 → hundreds of soft tokens), far
        //      above the ~tens of text tokens in the prompt. A silent text-only
        //      fallthrough (splice no-op) would leave this at the text count.
        let n_prompt = extract_u64(&out, "num_prompt_tokens")
            .unwrap_or_else(|| panic!("no num_prompt_tokens in result: {out}"));
        assert!(
            n_prompt > 100,
            "prompt missing image soft tokens (got {n_prompt}, expected >> text-only) \
             — the vision splice likely no-op'd: {out}"
        );

        eprintln!(
            "[cuda_multimodal] ✓ image splice end-to-end: {n_prompt} prompt tokens \
             (incl vision soft tokens) → {n_out} generated"
        );
        worker.shutdown().await;
    });
}
