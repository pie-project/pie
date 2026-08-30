//! Real-hardware multimodal validation -- image (vision) splice path on
//! `cuda_native`.
//!
//! What it MEANT to prove: the inferlet hands the host raw encoded image bytes,
//! the host runs the bound model's vision tower engine-side, scatters the
//! projected soft-token rows into the hidden state, commits them as ordinary KV
//! through the working-set forward txn, and a coherent text answer proves the
//! spliced visual span actually conditioned generation.
//!
//! It has not proved that in a long time, and this header used to say why in a
//! way that was wrong. Four of the five things standing between this file and a
//! run were config rot; the fifth is real and is not VRAM.
//!
//! # The VRAM claim was false, and it was the load-bearing one
//!
//! This file carried a `VRAM-BLOCKED on a 24G GPU` banner and an `#[ignore]`
//! reason repeating it: gemma-4-E4B's 15 G of weights plus the vision encoder
//! were said to exceed a 24 G 4090 "even at the max-fit config (gpu_mem 0.97 +
//! fp8 KV -> ~2545 MiB planner budget, still no viable forward layout)", so
//! hardware validation was said to need a bigger card.
//!
//! Measured on a 24 G 4090: it boots. `[cuda_multimodal] gemma-4-E4B up on
//! ws://127.0.0.1:...`, 43 s, and it boots at the DEFAULT `gpu_mem_utilization
//! = 0.90` with an ordinary bf16 KV cache -- the two-axis squeeze the config
//! function existed to apply is not needed either, and has been removed rather
//! than left as a cargo-culted tightening nobody can re-derive.
//!
//! The banner was almost certainly true when written. What made it outlive its
//! truth is that nothing could reach the boot: the config this file emitted had
//! stopped parsing, so every run died before allocating a byte, and a test that
//! cannot reach the thing it claims to be blocked on cannot notice when the
//! block lifts. A claim about hardware is worth exactly one measurement, and
//! this one had none.
//!
//! # The three layers of config rot, in the order they surfaced
//!
//! Each hid the next; this file hand-rolls its own worker TOML instead of going
//! through `common::cuda_toml_for`, so nothing that kept the shared harness
//! honest reached it.
//!
//!   1. The sandbox section is `[sandbox]`, and the engine's knobs are FLAT
//!      under `[engine]` -- there is no nested `[model.engine.options]` table.
//!   2. `invalid kv_cache_dtype "fp8"; expected one of: auto, bf16, bfloat16,
//!      fp8_e4m3, fp8_e5m2, ...` -- the dtype is spelled by its FORMAT, not by
//!      its width. Moot now that the squeeze is gone, but it was layer two.
//!   3. The bench image lives at the REPO root's `benches/assets`, not
//!      `crates/benches/assets`. `CARGO_MANIFEST_DIR` is `crates/worker`, so it
//!      is two hops up, not one. The one-hop form failed as "No such file or
//!      directory", which reads like a missing asset rather than a wrong path.
//!
//! # What actually blocks it now: there is no splice endpoint
//!
//! `image-qa-bench` is gone from `tests/inferlets`, along with every other
//! fixture that ever fed a picture to a model -- nothing under there mentions an
//! image today. That alone would just be a fixture to rewrite. It cannot be
//! rewritten, because the host-side door it would knock on is gone too.
//!
//! `crates/inferlet/wit/media.wit` is intact and `world.wit` still imports it,
//! so a guest can construct an `image` from encoded bytes and ask it for
//! `token-count`, `grid`, `prefix-tokens`. But NO interface consumes one:
//! `forward.wit`, `forward-hybrid.wit` and `forward-recurrent.wit` do not
//! contain the word "image" between them. The `pass.input_image(...) ->
//! execute()` path this header used to name did not survive the ETA forward
//! rewrite. The SDK's `media` module re-exports the raw bindings and nothing
//! else -- there is no `Context` helper -- and `MULTIMODAL.md`, which every one
//! of these comments pointed at, is no longer in the tree.
//!
//! The engine side is still there and still built: `kernels-cuda/kernels/vision`
//! holds `gemma4_vision.cuh` and `qwen3_vl_tower.cuh`. So the tower can run and
//! nothing can ask it to. That is the ticket this file now carries, and it is a
//! larger one than a missing fixture: restoring an image entry point on the
//! forward interface, not writing thirty lines of guest code.
//!
//! # Running it
//!
//! Needs a multimodal model. `gemma-4-E4B` is the only engine-supported vision
//! checkpoint cached here (gemma3n has no vision forward). Snapshot overridable
//! via `PIE_CUDA_TEST_MM_SNAPSHOT`. Use `--release`: a debug engine makes the
//! host the entire cost on a model this size.
//!
//!   cargo test --release -p worker --features engine-cuda-13 \
//!       --test cuda_multimodal -- --ignored --nocapture
//!
//! It will boot, which is the part that works, and then fail on the missing
//! fixture, which is the part that needs the WIT.

mod common;

use std::path::PathBuf;

/// Local `gemma-4-E4B` HF snapshot (vision + audio). Override with
/// `PIE_CUDA_TEST_MM_SNAPSHOT=/path/to/snapshot`.
const DEFAULT_MM_SNAPSHOT: &str = "/home/ingim/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/7aa32e6889efd6300124851b164f8b364314c3d8";

fn mm_snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_MM_SNAPSHOT").unwrap_or_else(|_| DEFAULT_MM_SNAPSHOT.to_string())
}

/// RETIRED -- `fn mm_cuda_toml` and `fn boot_mm_cuda` STOOD HERE.
///
/// They existed for one reason: a "max-fit" squeeze on the two axes the cuda
/// planner accepts, `gpu_mem_utilization = 0.97` and an fp8 KV cache, meant to
/// wedge gemma-4-E4B's 15 G onto a 24 G card. Measured, the model boots at the
/// shared harness's default 0.90 with a bf16 KV cache, in 43 s, so the squeeze
/// bought nothing and the duplicate config bought only somewhere for four
/// migrations to rot unobserved (see this file's header). Both are gone, and
/// this goes through `common::boot_cuda_model` like every other cuda gate.
const _RETIRED_MM_CONFIG: () = ();

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
#[ignore = "real-hardware, and CANNOT PASS on any GPU today: there is no image entry point on the forward interface -- forward.wit, forward-hybrid.wit and forward-recurrent.wit do not mention an image, and no `image-qa-bench` fixture survives to call one. media.wit and the engine-side vision towers are both intact, so the tower can run and nothing can ask it to. NOT VRAM: gemma-4-E4B boots on a 24 G 4090 in 43 s at the default 0.90 utilization -- measured, see this file's header, which the previous reason got wrong."]
fn cuda_native_image_splice_conditions_generation() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        // (1) Boot the embedded cuda engine with the multimodal model under the
        //     shared harness config -- it fits a 24 G card at the default
        //     0.90 utilization, which is the whole finding (see header).
        let worker = common::boot_cuda_model(&mm_snapshot()).await;
        eprintln!("[cuda_multimodal] gemma-4-E4B up on {}", worker.url());

        // (2) Local bench image -> base64 (no network). The asset lives at the
        //     REPO root's `benches/assets`, not `crates/benches/assets`, and
        //     `CARGO_MANIFEST_DIR` here is `crates/worker` -- so it takes two
        //     hops up, not one. The one-hop form read a directory that does not
        //     exist and failed as "No such file or directory", which reads like
        //     a missing asset rather than a wrong path.
        let img_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../benches/assets/bench_image.png");
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
