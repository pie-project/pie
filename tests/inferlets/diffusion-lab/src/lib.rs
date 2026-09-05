//! Sampler variants for DiffusionGemma, every one a guest program against
//! the same `forward-diffusion` host as `diffusion-baseline`: the host owns
//! the two attention readings and the KV; the guest owns the canvas, its
//! acceptance rule, its schedule, its self-conditioning taps, its stopping
//! rule, and what gets committed. Nothing here needed a host change.
//!
//! Knobs (all inputs; see `Pie.toml`):
//! - `variant`: the acceptance rule — `reference` (entropy-bound budget),
//!   `remask` (the `k_t` lowest-entropy rows, `k_t` on a linear count
//!   schedule — LLaDA's low-confidence remasking), `margin` (top-1 minus
//!   top-2 probability over a threshold).
//! - `sticky`: an accepted row keeps its token from then on (masked-diffusion
//!   decoding), device-carried as a frozen mask + ids; the reference
//!   re-decides every row every step (full renoise).
//! - `schedule`: `linear` or `cosine` temperature over the remaining steps.
//! - `stability`: how many steps the argmax canvas must hold before the
//!   confidence rule may stop the block.
//! - `taps`: self-conditioning taps staged per row (0 = none).
//! - `elide_commit`: the last block's causal encode is skipped — nothing
//!   reads its KV.
//! - `quality`: the commit encode reads every row out and reports the mean
//!   log-probability of the committed tokens under the causal reading.
//! - `best_of`: K canvases per block, the lowest final entropy committed.
//! - `clamp`: infilling — canvas positions held at given tokens.

use inferlet::eta::diffusion::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default = "default_variant")]
    variant: String,
    #[serde(default)]
    sticky: bool,
    #[serde(default = "default_schedule")]
    schedule: String,
    #[serde(default = "default_t_max")]
    t_max: f32,
    #[serde(default = "default_t_min")]
    t_min: f32,
    #[serde(default = "default_entropy_bound")]
    entropy_bound: f32,
    #[serde(default = "default_margin")]
    margin: f32,
    #[serde(default = "default_confidence")]
    confidence: f32,
    #[serde(default = "default_stability")]
    stability: u32,
    #[serde(default)]
    taps: Option<u32>,
    #[serde(default)]
    elide_commit: bool,
    #[serde(default)]
    quality: bool,
    #[serde(default = "default_best_of")]
    best_of: u32,
    #[serde(default)]
    clamp: Option<String>,
    #[serde(default = "default_seed")]
    seed: u32,
}

fn default_prompt() -> String {
    "Why is the sky blue?".into()
}
fn default_max_steps() -> u32 {
    48
}
fn default_variant() -> String {
    "reference".into()
}
fn default_schedule() -> String {
    "linear".into()
}
fn default_t_max() -> f32 {
    0.8
}
fn default_t_min() -> f32 {
    0.4
}
fn default_entropy_bound() -> f32 {
    0.1
}
fn default_margin() -> f32 {
    0.5
}
fn default_confidence() -> f32 {
    0.005
}
fn default_stability() -> u32 {
    1
}
fn default_best_of() -> u32 {
    1
}
fn default_seed() -> u32 {
    0x1234_5678
}

#[derive(Serialize)]
struct Output {
    text: String,
    tokens: Vec<u32>,
    /// Steps the committed canvas of each block took.
    steps: Vec<u32>,
    /// Mean per-position entropy at the committed canvas's last step.
    final_entropy: Vec<f32>,
    /// Final entropies of every candidate canvas per block (`best_of`).
    candidates: Vec<Vec<f32>>,
    /// Mean log-probability of each committed block's tokens under the
    /// causal reading (`quality`), empty otherwise.
    self_logprob: Vec<f32>,
    /// Denoise fires submitted, and encode fires.
    denoise_fires: u32,
    encode_fires: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    Reference,
    Remask,
    Margin,
}

/// Host-side uniform ids for a fresh canvas: xorshift32, seeded per block.
fn noise_canvas(seed: u32, length: u32, vocab: u32) -> Vec<i32> {
    let mut x = seed | 1;
    (0..length)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % vocab) as i32
        })
        .collect()
}

fn temperature(schedule: &str, remaining: u32, max_steps: u32, t_max: f32, t_min: f32) -> f32 {
    match schedule {
        "cosine" => {
            let x = remaining as f32 / max_steps.max(1) as f32;
            t_min + (t_max - t_min) * (1.0 - (std::f32::consts::PI * x).cos()) / 2.0
        }
        _ => linear_temperature(remaining, max_steps, t_max, t_min),
    }
}

/// What one denoised canvas came to.
struct Denoised {
    commit: Vec<i32>,
    steps: u32,
    entropy: f32,
    fires: u32,
}

struct Block<'a> {
    input: &'a Input,
    rule: Rule,
    ws: &'a WorkingSet,
    pipe: &'a Pipeline,
    length: u32,
    vocab: u32,
    taps_width: u32,
    taps_used: u32,
    page_size: u32,
    max_pages: u32,
    clamps: &'a [(u32, i32)],
}

impl Block<'_> {
    /// Denoise one canvas at positions `base..base+length`, seeded by `seed`.
    async fn denoise(&self, base: u32, seed: u32) -> Result<Denoised> {
        let Block {
            input,
            rule,
            ws,
            pipe,
            length,
            vocab,
            taps_width,
            taps_used,
            page_size,
            max_pages,
            clamps,
        } = *self;
        let end = base + length;
        let n = length as usize;
        let sticky = input.sticky;
        let bound = input.entropy_bound;
        let margin = input.margin;
        let confidence = input.confidence;
        let stability = input.stability as i32;

        let mut canvas = noise_canvas(seed, length, vocab);
        for &(p, t) in clamps {
            canvas[p as usize] = t;
        }
        let toks = Channel::from(canvas.clone()).named("canvas");
        let embed_indptr = Channel::from([0u32, length]).named("embed_indptr");
        let positions = Channel::from_iter(base..end).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off");
        let kv_len = Channel::from([end]).named("kv_len");
        let readout = Channel::from_iter(0..length).named("readout");
        // Control words, host-`set` before each submit.
        let temp = Channel::from([input.t_max]).named("temperature");
        let count = Channel::from([0i32]).named("count");
        let rng_state = Channel::from([seed, 0]).named("rng");
        // Device-carried state across steps.
        let history = Channel::from(vec![-1i32; n]).named("argmax_history");
        let stable_run = Channel::from([0i32]).named("stable_run");
        let frozen = Channel::from(vec![false; n]).named("frozen");
        let frozen_ids = Channel::from(vec![-1i32; n]).named("frozen_ids");
        // Outputs the host drains each step.
        let canvas_out = Channel::new([length], dtype::i32).named("canvas_out");
        let commit_out = Channel::new([length], dtype::i32).named("commit_out");
        let stop = Channel::new([1], dtype::bool).named("stop");
        let mean_out = Channel::new([1], dtype::f32).named("mean_entropy");
        let tap_ids_out = (taps_used > 0)
            .then(|| Channel::new([length, taps_used], dtype::u32).named("tap_ids"));
        let tap_weights_out = (taps_used > 0)
            .then(|| Channel::new([length, taps_used], dtype::f32).named("tap_weights"));

        let fwd = ForwardPass::new();
        fwd.canvas(Mode::Denoise)?;
        fwd.embed(&toks, &embed_indptr)?;
        fwd.attention(
            ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
        fwd.readout(&readout)?;
        {
            let (tap_ids_out, tap_weights_out) = (tap_ids_out.clone(), tap_weights_out.clone());
            fwd.epilogue(move || {
                positions.put(positions.take());
                w_slot.put(w_slot.take());
                w_off.put(w_off.take());

                let r = rng_state.take();
                let t = reshape(temp.read(), []);
                let logits = intrinsics::logits(); // [length, vocab]
                let scaled = &logits / &t;
                let probs = softmax(&scaled);
                let h = entropy(&probs); // [length]
                let sampled = gumbel_max(&scaled, &r);
                let argmax = reduce_argmax(&scaled);

                // ── the acceptance rule ──
                let accept = match rule {
                    Rule::Reference => entropy_bound_accept(&h, bound),
                    Rule::Remask => {
                        // Rank rows by entropy (ascending); accept the
                        // first `count` of them, `count` the host's schedule.
                        let (_, order) = sort_desc(-&h);
                        let zeros = cast(lt(iota(length), 0u32), dtype::i32);
                        let rank = scatter_set(&zeros, &order, &cast(iota(length), dtype::i32));
                        lt(&rank, &reshape(count.read(), []))
                    }
                    Rule::Margin => {
                        let (top2, _) = top_k(&probs, 2);
                        let col0 = cast(lt(iota(length), 0u32), dtype::u32);
                        let col1 = cast(ge(iota(length), 0u32), dtype::u32);
                        let gap = &gather_row(&top2, &col0) - &gather_row(&top2, &col1);
                        gt(&gap, margin)
                    }
                };

                let r_noise = &r + iota(2);
                let noise = cast(&rng(&r_noise, [length]) * (vocab as f32), dtype::i32);

                // ── sticky: an accepted row keeps its token from then on ──
                let (next, commit, all_frozen) = if sticky {
                    let was = frozen.take();
                    let ids = frozen_ids.take();
                    let newly = and(&accept, &not(&was));
                    let now = or(&was, &newly);
                    let kept = select(&was, &ids, &sampled);
                    let next = select(&now, &kept, &noise);
                    frozen_ids.put(&select(&now, &next, -1i32));
                    let count_frozen = reduce_sum(cast(&now, dtype::i32));
                    frozen.put(&now);
                    let commit = select(&now, &next, &argmax);
                    (next, commit, eq(&count_frozen, length as i32))
                } else {
                    (select(&accept, &sampled, &noise), argmax.clone(), lt(iota(1), 0u32))
                };

                // ── the stopping rule: stable for `stability` steps + confident ──
                let previous = history.take();
                history.put(&commit);
                let unchanged = reduce_sum(cast(eq(&commit, &previous), dtype::i32));
                let stable_now = reshape(eq(&unchanged, length as i32), [1]);
                let run = stable_run.take();
                let run_next = select(&stable_now, &(&run + 1i32), 0i32);
                stable_run.put(&run_next);
                let mean = &reduce_sum(&h) / (length as f32);
                let confident = reshape(lt(&mean, confidence), [1]);
                let done = or(&and(&ge(&run_next, stability), &confident), &all_frozen);

                if let (Some(ids_out), Some(weights_out)) = (&tap_ids_out, &tap_weights_out) {
                    let (tap_weights, tap_ids) = top_k(&probs, taps_used);
                    ids_out.put(&tap_ids);
                    weights_out.put(&tap_weights);
                }
                canvas_out.put(&next);
                commit_out.put(&commit);
                stop.put(&done);
                mean_out.put(reshape(&mean, [1]));
                rng_state.put(&(&r_noise + iota(2)));
            });
        }

        let mut commit: Vec<i32> = Vec::new();
        let mut steps = 0u32;
        let mut mean = f32::NAN;
        let max_steps = input.max_steps;
        for remaining in (1..=max_steps).rev() {
            let t = temperature(&input.schedule, remaining, max_steps, input.t_max, input.t_min);
            // Rows to accept this step under `remask`: a linear count schedule
            // reaching the whole canvas on the last step.
            let step_index = max_steps - remaining; // 0-based
            let k_t = ((length as u64 * (step_index as u64 + 1)).div_ceil(max_steps as u64)) as i32;
            if steps > 0 {
                temp.set([t]).context("set temperature")?;
                // Only the remask rule reads `count`; a channel the program
                // never reads has no host role, so `set` would be refused.
                if rule == Rule::Remask {
                    count.set([k_t]).context("set count")?;
                }
            }
            fwd.submit(pipe).with_context(|| format!("denoise submit @{base} step {steps}"))?;
            steps += 1;
            let mut next = canvas_out.take_host::<Vec<i32>>().await.context("canvas drain")?;
            commit = commit_out.take_host::<Vec<i32>>().await.context("commit drain")?;
            let done = stop.take_host::<bool>().await.context("stop drain")?;
            mean = mean_out.take_host::<f32>().await.context("entropy drain")?;
            let taps_data = match (&tap_ids_out, &tap_weights_out) {
                (Some(ids), Some(weights)) => Some((
                    ids.take_host::<Vec<u32>>().await.context("tap ids drain")?,
                    weights.take_host::<Vec<f32>>().await.context("tap weights drain")?,
                )),
                _ => None,
            };
            for &(p, tok) in clamps {
                next[p as usize] = tok;
                commit[p as usize] = tok;
            }
            if done || remaining == 1 {
                break;
            }
            toks.put(next);
            if let Some((ids, weights)) = taps_data {
                // The host wants the model's full width per row; pad the
                // narrower staging with zero-weight taps.
                if taps_used == taps_width {
                    fwd.self_conditioning(&ids, &weights).context("stage self-conditioning")?;
                } else {
                    let (w, u) = (taps_width as usize, taps_used as usize);
                    let mut full_ids = vec![0u32; n * w];
                    let mut full_weights = vec![0f32; n * w];
                    for row in 0..n {
                        full_ids[row * w..row * w + u].copy_from_slice(&ids[row * u..(row + 1) * u]);
                        full_weights[row * w..row * w + u].copy_from_slice(&weights[row * u..(row + 1) * u]);
                    }
                    fwd.self_conditioning(&full_ids, &full_weights)
                        .context("stage self-conditioning")?;
                }
            }
        }
        Ok(Denoised {
            commit,
            steps,
            entropy: mean,
            fires: steps,
        })
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Diffusion {
        return Err("this program drives a block-diffusion model; the bound model is not one".into());
    }
    let shape = model::canvas().ok_or("a diffusion model states its canvas")?;
    let length = shape.length;
    let taps_width = shape.self_cond_taps;
    let taps_used = input.taps.unwrap_or(taps_width).min(taps_width);
    if max_embed_length() < length as usize {
        return Err(format!(
            "the engine embeds at most {} tokens a pass and a canvas is {length}",
            max_embed_length()
        ));
    }
    if !(input.t_min > 0.0 && input.t_max >= input.t_min) {
        return Err("temperatures must satisfy 0 < t_min <= t_max".into());
    }
    let rule = match input.variant.as_str() {
        "reference" => Rule::Reference,
        "remask" => Rule::Remask,
        "margin" => Rule::Margin,
        other => return Err(format!("unknown variant {other:?}: reference | remask | margin")),
    };
    let clamps: Vec<(u32, i32)> = match &input.clamp {
        Some(text) => {
            let pairs: Vec<(u32, i32)> = serde_json::from_str(text).map_err(|e| format!("clamp: {e}"))?;
            if pairs.iter().any(|&(p, _)| p >= length) {
                return Err("clamp: a position is past the canvas".into());
            }
            pairs
        }
        None => Vec::new(),
    };
    let vocab = model::output_vocab_size();
    let page_size = kv_page_size();
    let stop_tokens = inferlet::chat::stop_tokens();
    let max_tokens = input.max_tokens.unwrap_or(length as usize);
    let best_of = input.best_of.max(1);
    let canvases = (max_tokens as u32).div_ceil(length).max(1);

    let mut prompt = inferlet::chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(inferlet::chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = u32::try_from(prompt.len()).map_err(|_| "prompt is too long")?;
    let max_pages = (n + canvases * length).div_ceil(page_size).max(1);

    let ws = WorkingSet::new();
    ws.reserve(max_pages).context("reserve KV")?;
    let pipe = Pipeline::new();
    let mut encode_fires = 0u32;
    let mut denoise_fires = 0u32;

    // ── 1. Prefill ────────────────────────────────────────────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    for &(base, end) in &prefill_chunks(n, None) {
        encode(&ws, &pipe, &prompt_i32[base as usize..end as usize], base, max_pages, page_size, false)
            .await
            .with_context(|| format!("prefill @{base}"))?;
        encode_fires += 1;
    }

    // ── 2. Blocks ─────────────────────────────────────────────────────────
    let mut generated: Vec<u32> = Vec::new();
    let mut steps_taken: Vec<u32> = Vec::new();
    let mut final_entropy: Vec<f32> = Vec::new();
    let mut candidates: Vec<Vec<f32>> = Vec::new();
    let mut self_logprob: Vec<f32> = Vec::new();

    for block in 0..canvases {
        let base = n + block * length;
        let block_clamps: &[(u32, i32)] = if block == 0 { &clamps } else { &[] };
        let runner = Block {
            input: &input,
            rule,
            ws: &ws,
            pipe: &pipe,
            length,
            vocab,
            taps_width,
            taps_used,
            page_size,
            max_pages,
            clamps: block_clamps,
        };
        // Best-of-K: K canvases on the same prefix (the denoise reading
        // writes no KV, so they share the pages untouched), the lowest final
        // entropy committed.
        let mut best: Option<Denoised> = None;
        let mut entropies = Vec::new();
        for k in 0..best_of {
            let seed = input.seed ^ block.wrapping_mul(0x9e37_79b9) ^ k.wrapping_mul(0x85eb_ca6b);
            let d = runner.denoise(base, seed).await?;
            denoise_fires += d.fires;
            entropies.push(d.entropy);
            let better = best.as_ref().is_none_or(|b| d.entropy < b.entropy);
            if better {
                best = Some(d);
            }
        }
        let chosen = best.ok_or("no canvas")?;
        candidates.push(entropies);
        steps_taken.push(chosen.steps);
        final_entropy.push(chosen.entropy);

        // Commit: the causal encode of the chosen canvas — unless this is
        // the last block and nothing will read its KV.
        let last_block = block + 1 == canvases;
        let mut finished = false;
        let mut kept = 0usize;
        for &id in &chosen.commit {
            let id = id as u32;
            if stop_tokens.contains(&id) {
                finished = true;
                break;
            }
            kept += 1;
            generated.push(id);
        }
        let done = finished || generated.len() >= max_tokens;
        if input.quality || !(input.elide_commit && (last_block || done)) {
            let q = encode(&ws, &pipe, &chosen.commit, base, max_pages, page_size, input.quality)
                .await
                .with_context(|| format!("commit @{base}"))?;
            encode_fires += 1;
            if let Some(q) = q {
                let _ = kept;
                self_logprob.push(q);
            }
        }
        if done {
            break;
        }
    }
    pipe.close();

    Ok(Output {
        text: model::decode(&generated)?,
        tokens: generated,
        steps: steps_taken,
        final_entropy,
        candidates,
        self_logprob,
        denoise_fires,
        encode_fires,
    })
}

/// One `encode` pass over `tokens` at positions `base..`: the causal
/// reading, whose K/V become the sequence. With `quality`, every row is read
/// out and the mean log-probability of `tokens[1..]` under the rows that
/// predict them comes back; otherwise only the last row is read (a pass
/// with nothing to say still says one word, so the fire has a drained
/// output).
async fn encode(
    ws: &WorkingSet,
    pipe: &Pipeline,
    tokens: &[i32],
    base: u32,
    max_pages: u32,
    page_size: u32,
    quality: bool,
) -> Result<Option<f32>> {
    let len = u32::try_from(tokens.len()).map_err(|_| "an encode span is too long")?;
    let end = base + len;
    let toks = Channel::from(tokens).named("toks_e");
    let embed_indptr = Channel::from([0u32, len]).named("embed_indptr_e");
    let positions = Channel::from_iter(base..end).named("positions_e");
    let pages = Channel::from_iter(0..max_pages).named("pages_e");
    let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_e");
    let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_e");
    let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_e");
    let kv_len = Channel::from([end]).named("kv_len_e");

    let fwd = ForwardPass::new();
    fwd.canvas(Mode::Encode)?;
    fwd.embed(&toks, &embed_indptr)?;
    fwd.attention(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &positions,
            mask: None,
        },
    )?;
    if quality && len > 1 {
        let readout = Channel::from_iter(0..len).named("readout_e");
        fwd.readout(&readout)?;
        // Row i predicts tokens[i + 1]; the last row predicts past the block
        // and gets weight 0.
        let mut next: Vec<u32> = tokens[1..].iter().map(|&t| t as u32).collect();
        next.push(0);
        let mut weight = vec![1.0f32; len as usize];
        weight[len as usize - 1] = 0.0;
        let next = Channel::from(next).named("next_e");
        let weight = Channel::from(weight).named("weight_e");
        let q_out = Channel::new([1], dtype::f32).named("quality_e");
        let denom = (len - 1) as f32;
        fwd.epilogue(move || {
            let lp = log_softmax(intrinsics::logits()); // [len, vocab]
            let picks = gather_row(&lp, &next.read()); // [len]
            let mean = &reduce_sum(&(&picks * &weight.read())) / denom;
            q_out.put(reshape(&mean, [1]));
        });
        fwd.submit(pipe)?;
        let q = q_out.take_host::<f32>().await?;
        Ok(Some(q))
    } else {
        let last = Channel::new([1], dtype::i32).named("last_e");
        fwd.epilogue(move || {
            last.put(reshape(cast(reduce_argmax(intrinsics::logits()), dtype::i32), [1]));
        });
        fwd.submit(pipe)?;
        last.take_host::<i32>().await?;
        Ok(quality.then_some(f32::NAN))
    }
}
