//! Page-trim benchmark.
//!
//! Measures the per-request cost of `BatchedForwardPassRequest::add_request`
//! across the patterns we care about:
//!
//! 1. Causal (no trim, fast path) — must stay at parity with pre-change.
//! 2. Attention sink + window (heavy trim) — both decode and prefill.
//! 3. Sliding window (trim leading pages).
//! 4. Multi-row disagree (early-exit case).
//!
//! For each scenario we report per-call wall time plus the resulting wire
//! size — pages emitted to `kv_page_indices` and bytes emitted to
//! `flattened_masks` — so we can verify the trim is doing real work.
//!
//! Usage:
//!   cargo bench --bench page_trim_bench

use std::time::Instant;

use pie::inference::brle::Brle;
use pie::inference::request::{BatchedForwardPassRequest, ForwardPassRequest};

const PAGE_SIZE: u32 = 16;

// =============================================================================
// Helpers
// =============================================================================

fn time_ns<F: FnMut()>(iters: u32, mut f: F) -> u64 {
    // Warm up
    for _ in 0..iters / 4 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_nanos() as u64 / iters as u64
}

fn fmt_time(ns: u64) -> String {
    if ns >= 1_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.2}µs", ns as f64 / 1_000.0)
    } else {
        format!("{ns}ns")
    }
}

fn make_request(masks: Vec<Brle>, num_input: u32, kv_before: u32) -> ForwardPassRequest {
    let positions: Vec<u32> = (kv_before..kv_before + num_input).collect();
    let tokens: Vec<u32> = vec![0; num_input as usize];
    let has_user_mask = !masks.is_empty();
    ForwardPassRequest {
        context_id: 0,
        tokens,
        positions,
        speculative_tokens: vec![],
        speculative_positions: vec![],
        output_speculative_tokens: false,
        masks,
        has_user_mask,
        logit_mask: None,
        sampling_indices: vec![],
        samplers: vec![],
        adapter_id: None,
        adapter_seed: None,
        arrival_time: None,
    }
}

fn last_page_len(total_kv: u32) -> u32 {
    let r = total_kv % PAGE_SIZE;
    if r == 0 { PAGE_SIZE } else { r }
}

/// Build a causal mask for a token at absolute position `pos`. Matches the
/// runtime's synthesized causal mask in `api/inference.rs`.
fn causal_row(pos: u32) -> Brle {
    Brle::all_true((pos + 1) as usize)
}

/// Sink+window mask covering [0, seq_len): sink leading trues, gap of falses,
/// then window trailing trues.
fn sink_window_row(seq_len: u32, sink: u32, window: u32) -> Brle {
    if seq_len <= sink + window {
        Brle::from_vec(vec![0, seq_len])
    } else {
        let gap = seq_len - sink - window;
        Brle::from_vec(vec![0, sink, gap, window])
    }
}

/// Sliding-window mask: gap of falses, then window of trues to the end.
fn window_row(seq_len: u32, window: u32) -> Brle {
    if seq_len <= window {
        Brle::from_vec(vec![0, seq_len])
    } else {
        let gap = seq_len - window;
        Brle::from_vec(vec![gap, window])
    }
}

/// Run a scenario: build a fresh `BatchedForwardPassRequest`, add the same
/// request `batch_size` times, time the loop, and return wire-size stats from
/// one iteration plus per-call wall time.
struct ScenarioResult {
    /// Average time per `add_request` call.
    per_call_ns: u64,
    /// Pages emitted to kv_page_indices for one request after trim.
    pages_after_trim: usize,
    /// Bytes emitted to flattened_masks for one request after trim.
    mask_bytes_after_trim: usize,
}

fn run_scenario(
    req: &ForwardPassRequest,
    pages: &[u32],
    last_page_len: u32,
    iters: u32,
) -> ScenarioResult {
    // Single-call wire stats (build once, inspect what got emitted).
    let mut probe = BatchedForwardPassRequest::new(0);
    probe.add_request(req, pages, last_page_len, PAGE_SIZE);
    let pages_after = probe.kv_page_indices.0.len();
    let mask_words = probe.flattened_masks.0.len();

    // Time the call: each iteration constructs a fresh batch and adds one
    // request. We pay an alloc per iter (for the fresh batch), but it's
    // identical across scenarios so the relative numbers are still meaningful.
    let per_call_ns = time_ns(iters, || {
        let mut batch = BatchedForwardPassRequest::new(0);
        batch.add_request(req, pages, last_page_len, PAGE_SIZE);
        std::hint::black_box(&batch);
    });

    ScenarioResult {
        per_call_ns,
        pages_after_trim: pages_after,
        mask_bytes_after_trim: mask_words * 4,
    }
}

fn print_row(scenario: &str, n_pages: u32, num_input: u32, r: ScenarioResult) {
    let pages_in = n_pages as usize;
    let saved_pct = if pages_in > 0 {
        100.0 * (1.0 - r.pages_after_trim as f64 / pages_in as f64)
    } else {
        0.0
    };
    println!(
        "  {:<28} {:>6} {:>6} {:>9} {:>6}/{:<6} {:>6.1}%  {:>6}B",
        scenario,
        n_pages,
        num_input,
        fmt_time(r.per_call_ns),
        r.pages_after_trim,
        pages_in,
        saved_pct,
        r.mask_bytes_after_trim,
    );
}

fn header() {
    println!(
        "  {:<28} {:>6} {:>6} {:>9} {:>13} {:>7}  {:>7}",
        "scenario", "pages", "input", "per_call", "pages_out/in", "saved", "mask"
    );
}

// =============================================================================
// Scenarios
// =============================================================================

fn bench_causal_decode(num_pages: u32, iters: u32) {
    // 1-token decode at the tail of the sequence. Causal mask = all-true up
    // to and including the new token's position. No trim possible.
    let total_kv = num_pages * PAGE_SIZE;
    let last = last_page_len(total_kv);
    let pages: Vec<u32> = (0..num_pages).collect();
    let pos = total_kv - 1;
    let req = make_request(vec![causal_row(pos)], 1, total_kv - 1);
    let r = run_scenario(&req, &pages, last, iters);
    print_row("causal_decode", num_pages, 1, r);
}

fn bench_causal_prefill(num_pages: u32, num_input: u32, iters: u32) {
    // N-token prefill landing at the tail. Each row gets its own causal mask
    // of length pos+1. No trim possible.
    let total_kv = num_pages * PAGE_SIZE;
    let last = last_page_len(total_kv);
    let pages: Vec<u32> = (0..num_pages).collect();
    let kv_before = total_kv - num_input;
    let masks: Vec<Brle> = (0..num_input).map(|i| causal_row(kv_before + i)).collect();
    let req = make_request(masks, num_input, kv_before);
    let r = run_scenario(&req, &pages, last, iters);
    print_row("causal_prefill", num_pages, num_input, r);
}

fn bench_sink_window_decode(num_pages: u32, iters: u32) {
    // 1-token decode with the attention-sink mask shape. Most middle pages
    // should drop.
    let total_kv = num_pages * PAGE_SIZE;
    let last = last_page_len(total_kv);
    let pages: Vec<u32> = (0..num_pages).collect();
    let mask = sink_window_row(total_kv, 4, 64);
    let req = make_request(vec![mask], 1, total_kv - 1);
    let r = run_scenario(&req, &pages, last, iters);
    print_row("sink_window_decode", num_pages, 1, r);
}

fn bench_sink_window_prefill(num_pages: u32, num_input: u32, iters: u32) {
    // Prefill with N rows, all sharing the same sink+window mask (matches
    // the inferlets/attention-sink pattern: clone one mask for every input).
    let total_kv = num_pages * PAGE_SIZE;
    let last = last_page_len(total_kv);
    let pages: Vec<u32> = (0..num_pages).collect();
    let mask = sink_window_row(total_kv, 4, 64);
    let masks: Vec<Brle> = (0..num_input).map(|_| mask.clone()).collect();
    let kv_before = total_kv - num_input;
    let req = make_request(masks, num_input, kv_before);
    let r = run_scenario(&req, &pages, last, iters);
    print_row("sink_window_prefill", num_pages, num_input, r);
}

fn bench_window_decode(num_pages: u32, iters: u32) {
    // Pure sliding window: leading pages drop.
    let total_kv = num_pages * PAGE_SIZE;
    let last = last_page_len(total_kv);
    let pages: Vec<u32> = (0..num_pages).collect();
    let mask = window_row(total_kv, 64);
    let req = make_request(vec![mask], 1, total_kv - 1);
    let r = run_scenario(&req, &pages, last, iters);
    print_row("window_decode", num_pages, 1, r);
}

fn bench_disagree_early_exit(num_pages: u32, num_input: u32, iters: u32) {
    // Multi-row request where row 0 marks early pages droppable, but row 1
    // has its false run elsewhere — eligibility should collapse on row 1
    // and short-circuit before scanning the remaining rows.
    let total_kv = num_pages * PAGE_SIZE;
    let last = last_page_len(total_kv);
    let pages: Vec<u32> = (0..num_pages).collect();
    // Row 0: false run [0, total_kv/2) then true to end.
    let half = total_kv / 2;
    let row0 = Brle::from_vec(vec![half, total_kv - half]);
    // Row 1: true [0, half), false [half, half+PAGE_SIZE), true rest.
    let mid_false = PAGE_SIZE;
    let row1 = Brle::from_vec(vec![0, half, mid_false, total_kv - half - mid_false]);
    // Fill remaining rows with row1 — row 1 already collapses eligibility.
    let mut masks: Vec<Brle> = Vec::with_capacity(num_input as usize);
    masks.push(row0);
    for _ in 1..num_input {
        masks.push(row1.clone());
    }
    let kv_before = total_kv - num_input;
    let req = make_request(masks, num_input, kv_before);
    let r = run_scenario(&req, &pages, last, iters);
    print_row("disagree_early_exit", num_pages, num_input, r);
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("\n=== add_request: per-call cost & trim effectiveness ===");
    println!("    page_size = {PAGE_SIZE}");
    header();

    let iters = 50_000;
    let page_counts = [16u32, 64, 256, 1024];

    println!("\n  -- causal (fast path, no trim) --");
    for &n in &page_counts {
        bench_causal_decode(n, iters);
    }
    bench_causal_prefill(64, 16, iters);
    bench_causal_prefill(256, 64, iters);
    bench_causal_prefill(1024, 128, iters);

    println!("\n  -- attention sink + window (heavy trim) --");
    for &n in &page_counts {
        bench_sink_window_decode(n, iters);
    }
    bench_sink_window_prefill(64, 16, iters);
    bench_sink_window_prefill(256, 64, iters);
    bench_sink_window_prefill(1024, 128, iters);

    println!("\n  -- sliding window (leading-page trim) --");
    for &n in &page_counts {
        bench_window_decode(n, iters);
    }

    println!("\n  -- multi-row disagree (early exit) --");
    bench_disagree_early_exit(256, 16, iters);
    bench_disagree_early_exit(1024, 64, iters);
    bench_disagree_early_exit(1024, 128, iters);

    println!();
}
