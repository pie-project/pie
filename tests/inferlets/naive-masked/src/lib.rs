//! MEASUREMENT INSTRUMENT — naive decode with an optional semantically-causal
//! custom AttnMask. NOT a policy inferlet; exists only to measure what a
//! plain-decode fire loses when it is forced onto the driver's custom-mask
//! path (`has_custom_mask` disables `use_decode_path` and
//! `fused_decode_qkv_post` fire-wide — Stage 2 verdict item A).
//!
//! The decode pass is chat-completion's decode shape (every descriptor port
//! channel-bound and re-put each fire — the device-geometry wire-form) with
//! naive-baseline's gumbel sampler. `mask_mode="none"` runs the identical
//! pass without the AttnMask binding, so the A/B differs ONLY in
//! the mask binding plus the mask-evolution epilogue ops (iota/le/put) —
//! the attention-path swap is the measured object.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
    /// Mask mode for the decode pass:
    ///   "none"       identical pass without the mask binding (the control)
    ///   "dense"      causal mask from generic ops (iota/le) — packs a dense
    ///                custom mask, forcing the custom-mask prefill path
    ///   "structured" causal mask via the CausalMask opcode sugar — the
    ///                driver recognizes it and takes the window-override
    ///                decode path (structured_window_left = -1)
    ///   "dense-prefill" causal host mask on the PREFILL chunks instead
    ///                (decode unmasked) — the chunk fires are
    ///                wire-geometry, so their BRLE rows are the
    ///                dense-mask-compose producer: concurrent decode
    ///                envelopes co-batch with them and the frame
    ///                assembles wire rows + causal fill host-side
    #[serde(default = "default_mask_mode")]
    mask_mode: String,
    /// STRUCTURAL: run only the first k layers (fire-level uniform
    /// truncation — composes with every mask mode).
    #[serde(default)]
    max_layers: Option<u32>,
    /// Step-logit probe: emit reduce_max(logits) per decode step into
    /// `lg` — the fingerprint the state-effect oracle diffs.
    #[serde(default)]
    logit_probe: bool,
}

fn default_prompt() -> String {
    "Write a short paragraph about naive sampling.".into()
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x7ce1
}

fn default_mask_mode() -> String {
    "dense".into()
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    mask_mode: String,
    text: String,
    count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    lg: Vec<f32>,
}

/// One sampling step: temperature, then a Gumbel-max draw over the full vocab.
fn step(logits: Tensor, temperature: f32, rng_state: impl AsTensor + Copy) -> Tensor {
    let scaled = if temperature == 1.0 {
        logits
    } else {
        div(&logits, temperature)
    };
    gumbel_max(scaled, rng_state)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let mask_mode = input.mask_mode.clone();
    let probe = input.logit_probe;
    let mut lg: Vec<f32> = Vec::new();
    if !matches!(
        mask_mode.as_str(),
        "none" | "dense" | "structured" | "dense-prefill" | "dense-prefill-hole"
            | "doc-isolation"
    ) {
        return Err(format!("unknown mask_mode: {mask_mode}"));
    }
    let masked = matches!(
        mask_mode.as_str(),
        "dense" | "structured" | "doc-isolation"
    );

    let structured = mask_mode == "structured";
    let masked_prefill = matches!(mask_mode.as_str(), "dense-prefill" | "dense-prefill-hole");
    // The hole: knock column 1 out of the causal envelope for rows p >= 2.
    // Non-causal by construction, so `is_pure_causal` cannot elide it — a
    // composed batch is forced through the wire-mask ASSEMBLY branch.
    let holed = mask_mode == "dense-prefill-hole";
    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "naive-masked",
            mask_mode,
            text: String::new(),
            count: 0,
            lg: Vec::new(),
        });
    }

    let mut prompt = wit_model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    // The first REAL mask policy through the spatial path: RAG document
    // isolation — the prompt's first half is a "retrieved document" the
    // decode queries must NOT attend to; the second half plus everything
    // generated stays visible.
    let doc_start: u32 = if mask_mode == "doc-isolation" { n / 2 } else { 0 };
    let pool_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    let pool_len = pool_pages * page_size;
    let slots = ws
        .reserve(pool_pages)
        .map_err(|e| format!("reserve KV: {e}"))?;
    let pool_ids = slots.ids().to_vec();

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked, C-wide): identical to naive-baseline, no mask. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let spans = prefill_chunks(n, None);
    let pipe = Pipeline::new();

    let mut g0 = 0i32;
    for &(base, end) in &spans {
        let len = end - base;

        let toks_p =
            Channel::from(prompt_i32[base as usize..end as usize].to_vec()).named("toks_p");
        let embed_indptr_p = Channel::from(vec![0u32, len]).named("embed_indptr_p");
        let positions_p = Channel::from((base..end).collect::<Vec<_>>()).named("positions_p");
        let pages_p = Channel::from(pool_ids.clone()).named("pages_p");
        let page_indptr_p =
            Channel::from(vec![0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot_p = Channel::from(
            (base..end)
                .map(|p| pool_ids[(p / page_size) as usize])
                .collect::<Vec<_>>(),
        )
        .named("w_slot_p");
        let w_off_p =
            Channel::from((base..end).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off_p");
        let kv_len_p = Channel::from(vec![end]).named("kv_len_p");
        let rng_p = Channel::from(vec![input.seed, 0]).named("rng_p");
        let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
        // dense-prefill: a semantically-causal host mask over the chunk's
        // query rows (row for position p allows j <= p). The chunk fire is
        // wire-geometry, so this lowers to wire BRLE rows — the lane shape
        // the dense-mask compose admits into shared batches.
        let mask_p = masked_prefill.then(|| {
            let rows: Vec<bool> = (base..end)
                .flat_map(|p| {
                    (0..pool_len).map(move |j| j <= p && !(holed && j == 1 && p >= 2))
                })
                .collect();
            Channel::from_shaped([len, pool_len], rows).named("mask_p")
        });

        let fwd_p = ForwardPass::new();
        if let Some(k) = input.max_layers {
            fwd_p.set_max_layers(k)?;
        }
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            &ws,
            ..,
            ..,
            &kv_len_p,
            &pages_p,
            &page_indptr_p,
            &w_slot_p,
            &w_off_p,
            &positions_p,
            mask_p.as_ref(),
        )?;
        fwd_p.epilogue(move || {
            let r = rng_p.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);
            let r_next = add(&r, iota(2));
            tok_out_p.put(&token);
            rng_p.put(&r_next);
        });

        fwd_p
            .submit(&pipe)
            .map_err(|e| format!("prefill submit @{base}: {e}"))?;

        g0 = tok_out_p
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("g0 take @{base}: {e}"))?[0];
    }
    generated.push(g0 as u32);

    // ── DECODE LOOP (1-wide, run-ahead) — chat-completion's decode shape. ──
    if generated.len() < max_tokens {
        let slot_n = pool_ids[(n / page_size) as usize];
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let pos = Channel::from(vec![n; 1]).named("pos");
        let fill = Channel::from(vec![n + 1; 1]).named("fill");
        let klen = Channel::from(vec![n + 1; 1]).named("klen");
        let w_slot = Channel::from(vec![slot_n; 1]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size; 1]).named("w_off");
        // Causal row for the fire-0 query at position n: attend all j <= n
        // (doc-isolation additionally blocks j < doc_start).
        let seed_mask: Vec<bool> =
            (0..pool_len).map(|j| j >= doc_start && j <= n).collect();
        let mask = Channel::from_shaped([1, pool_len], seed_mask).named("mask");
        let doc_row = Channel::from_shaped(
            [pool_len],
            (0..pool_len).map(|j| j >= doc_start).collect::<Vec<bool>>(),
        )
        .named("doc_row");
        let pages = Channel::from(pool_ids.clone()).named("pages");
        let page_indptr =
            Channel::from_shaped([2], vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
        let lg_out = Channel::new([1], dtype::f32)
            .capacity(8)
            .named("lg_out");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let rng = Channel::from(vec![input.seed ^ 0x5bd1, 0]).named("rng");
        let lane1 = Channel::from(vec![0u32, 1u32]).named("embed_indptr");

        let fwd = ForwardPass::new();
        if let Some(k) = input.max_layers {
            fwd.set_max_layers(k)?;
        }
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            &ws,
            ..,
            (n / page_size)..,
            &klen,
            &pages,
            &page_indptr,
            &w_slot,
            &w_off,
            &pos,
            if masked { Some(&mask) } else { None },
        )?;
        fwd.epilogue(move || {
            // TAKES + compute first, PUTS last (value-id discipline).
            let base = fill.take().tensor(); // [1] u32 — position this fire writes
            let pids = pool_ids_ch.take().tensor();
            let r = rng.take();

            let logits = intrinsics::logits();
            if probe {
                lg_out.put(&reduce_max(&logits));
            }
            let token = step(logits, temperature, &r);
            let r_next = add(&r, iota(2));

            let logical_slot = div(&base, page_size);
            let w_slot_v = gather(&pids, &logical_slot);
            let w_off_v = rem(&base, page_size);
            let klen_v = add(&base, 1u32);
            let next_free = add(&base, 1u32);
            let pages_v = reshape(&pids, [pool_pages]);
            // Page count tracks the new kv length, never the pool size.
            let page_count = div(add(&klen_v, page_size - 1), page_size);
            let pidx_v = mul(iota(2), broadcast(&page_count, [2]));

            tok_in.take();
            tok_in.put(&token);
            tok_out.put(&token);
            if masked {
                // Full causal mask for the next query at `base`: j <= base.
                let new_mask = if structured {
                    // CausalMask opcode — the driver's structured-mask
                    // recognizer lowers it to a runtime window override.
                    reshape(causal_mask(&base, pool_len), [1, pool_len])
                } else {
                    // Same semantics from generic ops — packs a dense
                    // custom mask (the custom-mask prefill path).
                    let col = iota(pool_len);
                    let base_b = broadcast(reshape(&base, [1]), [pool_len]);
                    reshape(le(&col, &base_b), [1, pool_len])
                };
                // doc-isolation: AND the static document-boundary row in
                // (a seeded channel read — the causal half evolves, the
                // document block is a constant of the request).
                let new_mask = if doc_start > 0 {
                    reshape(
                        and(&reshape(new_mask, [pool_len]), &doc_row.read()),
                        [1, pool_len],
                    )
                } else {
                    new_mask
                };
                mask.take();
                mask.put(&new_mask);
            }
            w_slot.take();
            w_slot.put(&w_slot_v);
            w_off.take();
            w_off.put(&w_off_v);
            klen.take();
            klen.put(&klen_v);
            pos.take();
            pos.put(&base);
            fill.put(&next_free);
            pages.take();
            pages.put(&pages_v);
            page_indptr.take();
            page_indptr.put(&pidx_v);
            rng.put(&r_next);
            pool_ids_ch.put(&pids);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget, async || {
            let t = tok_out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("tok_out.take @{}: {e}", generated.len()))?[0];
            if probe {
                let v = lg_out
                    .take()
                    .get::<f32>()
                    .await
                    .map_err(|e| format!("lg_out.take: {e}"))?[0];
                lg.push(v);
            }
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    Ok(Output {
        sampler: "naive-masked",
        mask_mode,
        text: wit_model::decode(&generated)?,
        count: generated.len(),
        lg,
    })
}
