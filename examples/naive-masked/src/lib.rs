//! A measurement instrument: naive decode forced onto the engine's custom-mask path.

use inferlet::eta::prelude::*;
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
    ///                engine recognizes it and takes the window-override
    ///                decode path (structured_window_left = -1)
    ///   "dense-prefill" causal host mask on the PREFILL instead (decode
    ///                unmasked), so concurrent decodes co-batch with its
    ///                dense rows
    ///   "doc-isolation" causal decode mask that also hides the prompt's
    ///                first half, a RAG document-isolation policy
    #[serde(default = "default_mask_mode")]
    mask_mode: String,
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
}

/// One sampling step: temperature, then a Gumbel-max draw over the full vocab.
fn step(logits: Tensor, temperature: f32, rng_state: &Tensor) -> Tensor {
    let scaled = if temperature == 1.0 {
        logits
    } else {
        &logits / temperature
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
    if !matches!(
        mask_mode.as_str(),
        "none" | "dense" | "structured" | "dense-prefill" | "doc-isolation"
    ) {
        return Err(format!("unknown mask_mode: {mask_mode}"));
    }
    let masked = matches!(mask_mode.as_str(), "dense" | "structured" | "doc-isolation");

    let structured = mask_mode == "structured";
    let masked_prefill = mask_mode == "dense-prefill";
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise.
    let prompt: Vec<i32> = inferlet::chat::prefix()
        .into_iter()
        .chain(wit_model::encode(&input.prompt))
        .map(|t| t as i32)
        .collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
    let n = prompt.len() as u32;
    // The first REAL mask policy through the spatial path: RAG document
    // isolation — the prompt's first half is a "retrieved document" the
    // decode queries must NOT attend to; the second half plus everything
    // generated stays visible.
    let doc_start: u32 = if mask_mode == "doc-isolation" {
        n / 2
    } else {
        0
    };
    let span = n + max_tokens as u32 + 1;

    if !head.is_empty() {
        let h = head.len() as u32;
        let fill = ForwardPass::new();
        fill.embed(&Channel::from(head), &Channel::from([0u32, h]))?;
        if masked_prefill {
            let rows: Vec<bool> = (0..h)
                .flat_map(|p| (0..span).map(move |j| j <= p))
                .collect();
            fill.mask(&Channel::from_shaped([h, span], rows))?;
        }
        fill.state(&[&ws], StateGeometry::folding())?;
        fill.submit(&pipe)
            .map_err(|e| format!("prefill submit: {e}"))?;
    }

    let tok_in = Channel::from(vec![last; 1]);
    let fill = Channel::from(vec![n; 1]);
    // Causal row for the first query, `last` at position n - 1
    // (doc-isolation additionally blocks j < doc_start).
    let seed_mask: Vec<bool> = (0..span).map(|j| j >= doc_start && j < n).collect();
    let mask = Channel::from_shaped([1, span], seed_mask);
    let doc_row = Channel::from_shaped(
        [span],
        (0..span).map(|j| j >= doc_start).collect::<Vec<bool>>(),
    );
    let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);
    let rng = Channel::from(vec![input.seed, 0]);
    let lane1 = Channel::from(vec![0u32, 1u32]);

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    if masked {
        fwd.mask(&mask)?;
    }
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        // TAKES + compute first, PUTS last (value-id discipline).
        let base = fill.take(); // [1] u32
        let r = rng.take();

        let token = step(intrinsics::output(0, "logits"), temperature, &r);
        let r_next = &r + iota(2);
        let next_free = &base + 1u32;

        tok_in.take();
        tok_in.put(&token);
        tok_out.put(&token);
        if masked {
            // Full causal mask for the next query at `base`: j <= base.
            let new_mask = if structured {
                // CausalMask opcode — the engine's structured-mask
                // recognizer lowers it to a runtime window override.
                reshape(causal_mask(&base, span), [1, span])
            } else {
                // Same semantics from generic ops — packs a dense
                // custom mask (the custom-mask prefill path).
                let col = iota(span);
                let base_b = broadcast(reshape(&base, [1]), [span]);
                reshape(le(&col, &base_b), [1, span])
            };
            // doc-isolation: AND the static document-boundary row in
            // (a seeded channel read — the causal half evolves, the
            // document block is a constant of the request).
            let new_mask = if doc_start > 0 {
                reshape(and(reshape(new_mask, [span]), doc_row.read()), [1, span])
            } else {
                new_mask
            };
            mask.take();
            mask.put(&new_mask);
        }
        fill.put(&next_free);
        rng.put(&r_next);
    });

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    run_ahead(&pipe, &fwd, max_tokens, async || {
        let t = tok_out
            .take_host::<i32>()
            .await
            .map_err(|e| format!("tok_out.take @{}: {e}", generated.len()))?;
        generated.push(t as u32);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    pipe.close();

    Ok(Output {
        sampler: "naive-masked",
        mask_mode,
        text: wit_model::decode(&generated)?,
        count: generated.len(),
    })
}
