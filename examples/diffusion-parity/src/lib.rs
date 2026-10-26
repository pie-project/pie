//! The pie half of the DiffusionGemma golden. Hands the model exactly what
//! `scripts/diffusiongemma_parity_ref.py ref` handed transformers — the
//! prompt ids, two fixed canvases, the temperature, and step 0's top-taps
//! as step 1's self-conditioning — and reads out, per row, what the
//! reference dumped: argmax, entropy of `softmax(logits / T)`, top-8.
//!
//! Three fires: an encode prefill (its last row read out), a denoise step
//! over `canvas0` with no signal, a denoise step over `canvas1` with the
//! taps. No sampling, no loop: the point is the numbers, row by row.

use inferlet::eta::diffusion::prelude::*;
use serde::{Deserialize, Serialize};

/// Tokens per block's canvas.
const CANVAS: u32 = 256;

#[derive(Deserialize)]
struct Input {
    /// `case.json`, as one string — for a case small enough to ride an
    /// argument.
    #[serde(default)]
    case: Option<String>,
    /// Or its file name under the sandbox's `/scratch` (the host's
    /// `sandbox.fs_scratch_dir`, with `allow_fs = true`).
    #[serde(default)]
    case_file: Option<String>,
    /// Or the same JSON cut into pieces `case_0 .. case_7`, concatenated
    /// here: the taps alone outgrow one command-line argument (128 KiB).
    #[serde(default)]
    case_0: Option<String>,
    #[serde(default)]
    case_1: Option<String>,
    #[serde(default)]
    case_2: Option<String>,
    #[serde(default)]
    case_3: Option<String>,
    #[serde(default)]
    case_4: Option<String>,
    #[serde(default)]
    case_5: Option<String>,
    #[serde(default)]
    case_6: Option<String>,
    #[serde(default)]
    case_7: Option<String>,
    /// Stop after the prefill rows (the trunk alone).
    #[serde(default)]
    only_prefill: bool,
}

#[derive(Deserialize)]
struct Case {
    prompt_ids: Vec<u32>,
    canvas0: Vec<i32>,
    canvas1: Vec<i32>,
    temperature: f32,
    taps: u32,
    taps_ids: Vec<u32>,
    taps_weights: Vec<f32>,
}

#[derive(Serialize, Default)]
struct Rows {
    argmax: Vec<i32>,
    entropy: Vec<f32>,
    top8_ids: Vec<u32>,
    top8_probs: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    prefill_argmax: i32,
    prefill_top8_ids: Vec<u32>,
    prefill_top8_probs: Vec<f32>,
    /// Every prompt row of the prefill, teacher-forced (T = 1).
    prefill_rows: Rows,
    step0: Rows,
    step1: Rows,
    /// `canvas0` read the ENCODE way (causal, no post-norm) — the trunk
    /// alone, against the reference encoder over `[prompt | canvas0]`.
    encode_canvas: Rows,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !model::switch("mode").is_some_and(|mode| mode.values.iter().any(|v| v == "denoise")) {
        return Err(
            "this program reads a block-diffusion model; the bound model is not one".into(),
        );
    }
    let pieces: String = [
        &input.case_0,
        &input.case_1,
        &input.case_2,
        &input.case_3,
        &input.case_4,
        &input.case_5,
        &input.case_6,
        &input.case_7,
    ]
    .into_iter()
    .flatten()
    .map(String::as_str)
    .collect();
    let text = match (&input.case, &input.case_file) {
        (Some(text), _) => text.clone(),
        (None, Some(name)) => std::fs::read_to_string(format!("/scratch/{name}"))
            .map_err(|why| format!("reading /scratch/{name}: {why}"))?,
        (None, None) if !pieces.is_empty() => pieces,
        (None, None) => {
            return Err("pass `case` (json), `case_0..7` (its pieces) or `case_file`".into());
        }
    };
    let case: Case =
        inferlet::serde_json::from_str(&text).map_err(|why| format!("case json: {why}"))?;
    let length = CANVAS;
    if case.canvas0.len() != length as usize || case.canvas1.len() != length as usize {
        return Err(format!("the case's canvases are not {length} long"));
    }
    let lane = model::lane_of(&[("mode", "denoise")])?;
    let taps = lane
        .inputs
        .iter()
        .find(|input| input.name == "self_cond_ids")
        .ok_or("the denoise lane reads no `self_cond_ids` input")?
        .width;
    if case.taps != taps {
        return Err(format!(
            "the case carries {} taps per row and this model takes {taps}",
            case.taps
        ));
    }
    let n = u32::try_from(case.prompt_ids.len()).map_err(|_| "prompt is too long")?;
    let temperature = case.temperature;

    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    let prompt_i32: Vec<i32> = case.prompt_ids.iter().map(|&t| t as i32).collect();
    let spans = readout_spans(
        n,
        model::lane_of(&[])?
            .outputs
            .iter()
            .find(|output| output.name == "logits")
            .map_or(1, |output| output.max_rows),
    );
    let mut prefill_rows = Rows::default();
    for &(base, end) in spans.iter() {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]);
        let embed_indptr = Channel::from([0u32, len]);
        let slots = Channel::from_iter(base..end);
        let readout = Channel::from_iter(0..len);
        let arg_out = Channel::new([len], dtype::i32);
        let ent_out = Channel::new([len], dtype::f32);
        let ids_out = Channel::new([len, 8], dtype::u32);
        let probs_out = Channel::new([len, 8], dtype::f32);

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
        fwd.state(
            &[&ws],
            StateGeometry {
                slots: Some(&slots),
                fold: Fold::All,
                ..StateGeometry::default()
            },
        )?;
        fwd.readout(&readout)?;
        fwd.epilogue(move || {
            let logits = intrinsics::output(0, "logits"); // [len, vocab]
            let probs = softmax(&logits);
            let (top_p, top_i) = top_k(&probs, 8);
            arg_out.put(reduce_argmax(&logits));
            ent_out.put(entropy(&probs));
            ids_out.put(top_i);
            probs_out.put(top_p);
        });
        fwd.submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        prefill_rows
            .argmax
            .extend(arg_out.take_host::<Vec<i32>>().await?);
        prefill_rows
            .entropy
            .extend(ent_out.take_host::<Vec<f32>>().await?);
        prefill_rows
            .top8_ids
            .extend(ids_out.take_host::<Vec<u32>>().await?);
        prefill_rows
            .top8_probs
            .extend(probs_out.take_host::<Vec<f32>>().await?);
    }
    let prefill_argmax = *prefill_rows.argmax.last().unwrap_or(&0);
    let prefill_top8_ids = prefill_rows
        .top8_ids
        .iter()
        .rev()
        .take(8)
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let prefill_top8_probs = prefill_rows
        .top8_probs
        .iter()
        .rev()
        .take(8)
        .rev()
        .copied()
        .collect::<Vec<_>>();

    // ── two denoise steps over the two stated canvases, then canvas0 the
    //    encode way (last: an encode pass commits its rows) ─────────────────
    let mut steps: Vec<Rows> = Vec::new();
    let canvases: Vec<&Vec<i32>> = if input.only_prefill {
        Vec::new()
    } else {
        vec![&case.canvas0, &case.canvas1, &case.canvas0]
    };
    for (at, canvas) in canvases.into_iter().enumerate() {
        let toks = Channel::from(canvas.as_slice());
        let embed_indptr = Channel::from([0u32, length]);
        let slots = Channel::from_iter(n..n + length);
        let readout = Channel::from_iter(0..length);
        let arg_out = Channel::new([length], dtype::i32);
        let ent_out = Channel::new([length], dtype::f32);
        let ids_out = Channel::new([length, 8], dtype::u32);
        let probs_out = Channel::new([length, 8], dtype::f32);

        let fwd = ForwardPass::new();
        let commit = at == 2;
        if !commit {
            fwd.switch(0, "mode", "denoise")?;
        }
        fwd.embed(&toks, &embed_indptr)?;
        // A committed canvas folds; one the next step rewrites does not.
        fwd.state(
            &[&ws],
            StateGeometry {
                slots: Some(&slots),
                fold: if commit { Fold::All } else { Fold::None },
                ..StateGeometry::default()
            },
        )?;
        fwd.readout(&readout)?;
        fwd.epilogue(move || {
            let logits = intrinsics::output(0, "logits"); // [length, vocab]
            let scaled = &logits / temperature;
            let probs = softmax(&scaled);
            let (top_p, top_i) = top_k(&probs, 8);
            arg_out.put(reduce_argmax(&scaled));
            ent_out.put(entropy(&probs));
            ids_out.put(top_i);
            probs_out.put(top_p);
        });
        // A denoise step reads the taps: zeros (no signal) on step 0, the
        // case's on step 1.
        if !commit {
            let cells = (length * taps) as usize;
            let (ids, weights) = if at == 1 {
                (
                    case.taps_ids.iter().map(|&id| id as i32).collect(),
                    case.taps_weights.clone(),
                )
            } else {
                (vec![0i32; cells], vec![0f32; cells])
            };
            let ids = Channel::from_shaped([length, taps], ids);
            let weights = Channel::from_shaped([length, taps], weights);
            fwd.input(0, "self_cond_ids", &ids)
                .context("bind self_cond_ids")?;
            fwd.input(0, "self_cond_weights", &weights)
                .context("bind self_cond_weights")?;
        }
        fwd.submit(&pipe)
            .with_context(|| format!("denoise submit {at}"))?;
        steps.push(Rows {
            argmax: arg_out.take_host::<Vec<i32>>().await?,
            entropy: ent_out.take_host::<Vec<f32>>().await?,
            top8_ids: ids_out.take_host::<Vec<u32>>().await?,
            top8_probs: probs_out.take_host::<Vec<f32>>().await?,
        });
    }
    pipe.close();

    let encode_canvas = steps.pop().unwrap_or_default();
    let step1 = steps.pop().unwrap_or_default();
    let step0 = steps.pop().unwrap_or_default();
    Ok(Output {
        prefill_argmax,
        prefill_top8_ids,
        prefill_top8_probs,
        prefill_rows,
        step0,
        step1,
        encode_canvas,
    })
}
