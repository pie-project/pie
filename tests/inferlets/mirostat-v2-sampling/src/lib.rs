//! Mirostat v2 adaptive-surprise sampling.
//!
//! Each step keeps tokens whose surprise is at most the current control value
//! `mu`, samples from that set with Gumbel-max, observes the selected token's
//! surprise, and applies `mu <- mu - learning_rate * (surprise - tau)`.
//!
//! Floor selection (keeps the keep-mask non-empty regardless of μ):
//!   - `"argmax"` (DEFAULT): OR the surprise mask with the max-prob token's own
//!     indicator (`ge(probs, broadcast(reduce_max(probs)))`) — proven-ops,
//!     never empty-keep.
//!   - `"rank"` with `k_min>0`: OR with a top-`k_min` rank mask
//!     (`pivot_threshold(logits, rank_le(k_min))`).
//!   - `k_min=0` (no floor override): the plain, degenerate control.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_tau")]
    tau: f32,
    #[serde(default = "default_learning_rate")]
    learning_rate: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_k_min")]
    k_min: u32,
    #[serde(default = "default_floor")]
    floor: String,
    #[serde(default)]
    mu0: Option<f32>,
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    tau: f32,
    final_mu: f32,
    mean_surprise: f32,
    tail_mean_surprise: f32,
}

fn default_prompt() -> String {
    "Write a short paragraph about adaptive sampling.".into()
}

fn default_tau() -> f32 {
    3.0
}

fn default_learning_rate() -> f32 {
    0.6
}

fn default_max_tokens() -> usize {
    64
}

fn default_k_min() -> u32 {
    8
}

fn default_floor() -> String {
    "argmax".into()
}

#[derive(Clone, Copy)]
enum Floor {
    Argmax,
    Rank(u32),
    Plain,
}

/// The mirostat keep-mask + Gumbel-max sample + observed surprise, over the
/// FULL (untruncated) `probs`/`logits` distribution. `mu` is reshaped to a
/// scalar here — `pivot_threshold`'s predicate threshold operand is validated
/// as EXACTLY scalar (or a per-row `[rows]` vector for rank-2 input), not a
/// `[1]` vector, so a `[1]`-channel-sourced `mu` must be squeezed down. `r` is
/// the taken `[2]` u32 rng state (`[key, ctr]`) — `gumbel`'s `state` operand is
/// likewise validated as EXACTLY `[2]` u32. Returns `(token, surprise)`.
fn mirostat_step(
    floor: Floor,
    logits: Tensor,
    vocab: u32,
    mu: impl AsTensor,
    r: impl AsTensor,
) -> (Tensor, Tensor) {
    let mu = reshape(mu, []); // [1] -> scalar
    let probs = softmax(&logits);
    let thr = exp(neg(&mu));
    let base_mask = pivot_threshold(&probs, prob_ge(thr));
    let mask = match floor {
        Floor::Argmax => {
            let argmax_ind = ge(&probs, broadcast(reduce_max(&probs), [vocab]));
            or(base_mask, argmax_ind)
        }
        Floor::Rank(k_min) if k_min > 0 => {
            let rank_mask = pivot_threshold(&logits, rank_le(k_min));
            or(base_mask, rank_mask)
        }
        _ => base_mask,
    };
    let neg_inf = broadcast(Tensor::constant(f32::NEG_INFINITY), [vocab]);
    let masked = select(&mask, &logits, &neg_inf);
    let token = gumbel_max(masked, r); // [1] i32
    let p_token = scalar_gather(&probs, cast(&token, DType::U32)); // [1] f32
    let surprise = neg(log(p_token)); // [1] f32
    (token, surprise)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.tau.is_finite() || input.tau <= 0.0 {
        return Err("tau must be finite and greater than 0".into());
    }
    if !input.learning_rate.is_finite() || input.learning_rate <= 0.0 {
        return Err("learning_rate must be finite and greater than 0".into());
    }
    if input.mu0.is_some_and(|mu| !mu.is_finite()) {
        return Err("mu0 must be finite".into());
    }

    let floor = match input.floor.as_str() {
        "argmax" => Floor::Argmax,
        "rank" if input.k_min > 0 => Floor::Rank(input.k_min),
        "plain" => Floor::Plain,
        "rank" => return Err("rank floor requires k_min > 0".into()),
        other => return Err(format!("unknown floor mode: {other}")),
    };
    let tau = input.tau;
    let lr = input.learning_rate;
    let max_tokens = input.max_tokens;

    let vocab = wit_model::output_vocab_size();
    let ws = WorkingSet::new();
    model::configure(vocab, ws.page_size(), 1);

    let mut mu = input.mu0.unwrap_or_else(|| (vocab as f32).ln() + 1.0);
    if max_tokens == 0 {
        return Ok(Output {
            sampler: "mirostat-v2",
            text: String::new(),
            count: 0,
            tau,
            final_mu: mu,
            mean_surprise: 0.0,
            tail_mean_surprise: 0.0,
        });
    }

    let mut prompt = wit_model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;

    let mut surprises: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL FIRE (N-wide) — first mirostat step over the prompt. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let klen_p = Channel::from(vec![n; 1]).named("klen_p");
    let mu_p = Channel::new([1], dtype::f32).named("mu_p");
    let rng_p = Channel::from(vec![0x9e37_u32, 0]).named("rng_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let s_out_p = Channel::new([1], dtype::f32).named("s_out_p");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, Tensor::constant(vec![0u32, n]));
    fwd_p.attn_working_set(&ws, &klen_p);
    fwd_p.epilogue(|| {
        let mu_v = mu_p.take().tensor();
        let r = rng_p.take(); // [2] u32 rng state (key, ctr)
        let logits = intrinsics::logits(); // [vocab] f32 (single read-out row)
        let (token, surprise) = mirostat_step(floor, logits, vocab, mu_v, &r);
        let r_next = add(&r, iota(2));
        tok_out_p.put(&token);
        s_out_p.put(&surprise);
        rng_p.put(&r_next);
    });

    mu_p.put(vec![mu]);
    let prefill = Pipeline::new();
    fwd_p
        .submit(&prefill)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = tok_out_p
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];
    let s0 = s_out_p
        .take()
        .get::<f32>()
        .await
        .map_err(|e| format!("s0 take: {e}"))?[0];
    prefill.close();

    generated.push(g0 as u32);
    surprises.push(s0);
    mu -= lr * (s0 - tau);

    // ── DECODE LOOP (1-wide) ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let pos = Channel::from(vec![n; 1]).named("pos");
        let klen = Channel::from(vec![n + 1; 1]).named("klen");
        let fill = Channel::from(vec![n + 1; 1]).named("fill");
        let mu_state = Channel::from(vec![mu]).named("mu_state");
        let rng = Channel::from(vec![0x51ed_u32, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
            .named("tok_out");
        let s_out = Channel::new([1], dtype::f32)
            .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
            .named("s_out");
        let lane1 = Tensor::constant(vec![0u32, 1u32]);

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, lane1);
        fwd.positions(&pos);
        fwd.attn_working_set(&ws, &klen);
        fwd.epilogue(|| {
            // Takes + compute first, PUTS last (value-id discipline).
            let base = fill.take().tensor(); // [1] u32
            let mu_v = mu_state.take().tensor(); // [1] f32, device loop-carried
            let r = rng.take(); // [2] u32 rng state
            let logits = intrinsics::logits(); // [vocab] f32
            let (token, surprise) = mirostat_step(floor, logits, vocab, &mu_v, &r);

            let klen_v = add(&base, 1u32);
            let next_free = add(&base, 1u32);
            let r_next = add(&r, iota(2));
            let mu_next = sub(&mu_v, mul(sub(&surprise, tau), lr));

            tok_in.put(&token);
            tok_out.put(&token);
            s_out.put(&surprise);
            pos.put(&base);
            klen.put(&klen_v);
            fill.put(&next_free);
            rng.put(&r_next);
            mu_state.put(&mu_next);
        });

        let decode = Pipeline::new();
        let budget = max_tokens - 1;
        let mut submitted = 0usize;
        let mut in_flight = 0usize;
        while in_flight < DEFAULT_RUNAHEAD_DEPTH && submitted < budget {
            fwd.submit(&decode)
                .map_err(|e| format!("decode submit @{}: {e}", submitted + 1))?;
            submitted += 1;
            in_flight += 1;
        }
        while in_flight > 0 {
            let t = tok_out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("tok_out.take @{}: {e}", generated.len()))?[0];
            let s = s_out
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("s_out.take @{}: {e}", generated.len()))?[0];
            in_flight -= 1;
            generated.push(t as u32);
            surprises.push(s);
            mu -= lr * (s - tau);
            if submitted < budget {
                fwd.submit(&decode)
                    .map_err(|e| format!("decode submit @{}: {e}", submitted + 1))?;
                submitted += 1;
                in_flight += 1;
            }
        }
        decode.close();
    }

    let mean_s = if surprises.is_empty() {
        0.0
    } else {
        surprises.iter().sum::<f32>() / surprises.len() as f32
    };
    let tail_mean_s = if surprises.len() >= 2 {
        let tail = &surprises[surprises.len() / 2..];
        tail.iter().sum::<f32>() / tail.len() as f32
    } else {
        mean_s
    };
    Ok(Output {
        sampler: "mirostat-v2",
        text: wit_model::decode(&generated)?,
        count: generated.len(),
        tau,
        final_mu: mu,
        mean_surprise: mean_s,
        tail_mean_surprise: tail_mean_s,
    })
}
