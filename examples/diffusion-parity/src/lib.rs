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
    /// Run only the first `max_layers` layers and take the head there —
    /// the logit lens, against the reference's per-layer hidden states.
    #[serde(default)]
    max_layers: Option<u32>,
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

/// The geometry of one span of `len` rows at `base`, over a working set of
/// `max_pages` pages; every channel seeded, nothing device-advanced.
struct Span {
    embed_indptr: Channel,
    positions: Channel,
    pages: Channel,
    page_indptr: Channel,
    w_slot: Channel,
    w_off: Channel,
    kv_len: Channel,
}

impl Span {
    fn new(base: u32, len: u32, max_pages: u32, page_size: u32, tag: &str) -> Span {
        let end = base + len;
        Span {
            embed_indptr: Channel::from([0u32, len]).named(&format!("embed_indptr_{tag}")),
            positions: Channel::from_iter(base..end).named(&format!("positions_{tag}")),
            pages: Channel::from_iter(0..max_pages).named(&format!("pages_{tag}")),
            page_indptr: Channel::from([0u32, end.div_ceil(page_size)])
                .named(&format!("page_indptr_{tag}")),
            w_slot: Channel::from_iter((base..end).map(|p| p / page_size))
                .named(&format!("w_slot_{tag}")),
            w_off: Channel::from_iter((base..end).map(|p| p % page_size))
                .named(&format!("w_off_{tag}")),
            kv_len: Channel::from([end]).named(&format!("kv_len_{tag}")),
        }
    }

    fn geometry(&self) -> KvGeometry<'_, std::ops::RangeFull, std::ops::RangeFull> {
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &self.kv_len,
            pages: &self.pages,
            page_indptr: &self.page_indptr,
            w_slot: &self.w_slot,
            w_off: &self.w_off,
            positions: &self.positions,
            mask: None,
        }
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Diffusion {
        return Err("this program reads a block-diffusion model; the bound model is not one".into());
    }
    let pieces: String = [
        &input.case_0, &input.case_1, &input.case_2, &input.case_3,
        &input.case_4, &input.case_5, &input.case_6, &input.case_7,
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
        (None, None) => return Err("pass `case` (json), `case_0..7` (its pieces) or `case_file`".into()),
    };
    let case: Case = inferlet::serde_json::from_str(&text)
        .map_err(|why| format!("case json: {why}"))?;
    let shape = model::canvas().ok_or("a diffusion model states its canvas")?;
    let length = shape.length;
    if case.canvas0.len() != length as usize || case.canvas1.len() != length as usize {
        return Err(format!("the case's canvases are not {length} long").into());
    }
    if case.taps != shape.self_cond_taps {
        return Err(format!(
            "the case carries {} taps per row and this model takes {}",
            case.taps, shape.self_cond_taps
        )
        .into());
    }
    let page_size = kv_page_size();
    let n = u32::try_from(case.prompt_ids.len()).map_err(|_| "prompt is too long")?;
    let max_pages = (n + length).div_ceil(page_size).max(1);
    let temperature = case.temperature;

    let ws = WorkingSet::new();
    ws.reserve(max_pages).context("reserve KV")?;
    let pipe = Pipeline::new();

    // ── prefill: encode passes, the last one read out ─────────────────────
    let prompt_i32: Vec<i32> = case.prompt_ids.iter().map(|&t| t as i32).collect();
    let chunks = prefill_chunks(n, None);
    let mut prefill_rows = Rows::default();
    for (at, &(base, end)) in chunks.iter().enumerate() {
        let len = end - base;
        let tag = format!("p{at}");
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named(&format!("toks_{tag}"));
        let span = Span::new(base, len, max_pages, page_size, &tag);
        let readout = Channel::from_iter(0..len).named(&format!("readout_{tag}"));
        let arg_out = Channel::new([len], dtype::i32).named(&format!("arg_{tag}"));
        let ent_out = Channel::new([len], dtype::f32).named(&format!("ent_{tag}"));
        let ids_out = Channel::new([len, 8], dtype::u32).named(&format!("ids_{tag}"));
        let probs_out = Channel::new([len, 8], dtype::f32).named(&format!("probs_{tag}"));

        let fwd = ForwardPass::new();
        fwd.canvas(Mode::Encode)?;
        if let Some(k) = input.max_layers {
            fwd.set_max_layers(k)?;
        }
        fwd.embed(&toks, &span.embed_indptr)?;
        fwd.attention(&ws, span.geometry())?;
        fwd.readout(&readout)?;
        fwd.epilogue(move || {
            let logits = intrinsics::logits(); // [len, vocab]
            let probs = softmax(&logits);
            let (top_p, top_i) = top_k(&probs, 8);
            arg_out.put(reduce_argmax(&logits));
            ent_out.put(entropy(&probs));
            ids_out.put(top_i);
            probs_out.put(top_p);
        });
        fwd.submit(&pipe).with_context(|| format!("prefill submit @{base}"))?;
        prefill_rows.argmax.extend(arg_out.take_host::<Vec<i32>>().await?);
        prefill_rows.entropy.extend(ent_out.take_host::<Vec<f32>>().await?);
        prefill_rows.top8_ids.extend(ids_out.take_host::<Vec<u32>>().await?);
        prefill_rows.top8_probs.extend(probs_out.take_host::<Vec<f32>>().await?);
    }
    let prefill_argmax = *prefill_rows.argmax.last().unwrap_or(&0);
    let prefill_top8_ids = prefill_rows.top8_ids.iter().rev().take(8).rev().copied().collect::<Vec<_>>();
    let prefill_top8_probs = prefill_rows.top8_probs.iter().rev().take(8).rev().copied().collect::<Vec<_>>();

    // ── two denoise steps over the two stated canvases, then canvas0 the
    //    encode way (last: an encode pass commits its rows) ─────────────────
    let mut steps: Vec<Rows> = Vec::new();
    let canvases: Vec<&Vec<i32>> = if input.only_prefill {
        Vec::new()
    } else {
        vec![&case.canvas0, &case.canvas1, &case.canvas0]
    };
    for (at, canvas) in canvases.into_iter().enumerate() {
        let tag = format!("d{at}");
        let toks = Channel::from(canvas.as_slice()).named(&format!("canvas_{tag}"));
        let span = Span::new(n, length, max_pages, page_size, &tag);
        let readout = Channel::from_iter(0..length).named(&format!("readout_{tag}"));
        let arg_out = Channel::new([length], dtype::i32).named(&format!("arg_{tag}"));
        let ent_out = Channel::new([length], dtype::f32).named(&format!("ent_{tag}"));
        let ids_out = Channel::new([length, 8], dtype::u32).named(&format!("ids_{tag}"));
        let probs_out = Channel::new([length, 8], dtype::f32).named(&format!("probs_{tag}"));

        let fwd = ForwardPass::new();
        fwd.canvas(if at == 2 { Mode::Encode } else { Mode::Denoise })?;
        fwd.embed(&toks, &span.embed_indptr)?;
        fwd.attention(&ws, span.geometry())?;
        fwd.readout(&readout)?;
        fwd.epilogue(move || {
            let logits = intrinsics::logits(); // [length, vocab]
            let scaled = &logits / temperature;
            let probs = softmax(&scaled);
            let (top_p, top_i) = top_k(&probs, 8);
            arg_out.put(reduce_argmax(&scaled));
            ent_out.put(entropy(&probs));
            ids_out.put(top_i);
            probs_out.put(top_p);
        });
        if at == 1 {
            fwd.self_conditioning(&case.taps_ids, &case.taps_weights)
                .context("stage self-conditioning")?;
        }
        fwd.submit(&pipe).with_context(|| format!("denoise submit {at}"))?;
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
