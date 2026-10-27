//! Context-aware decoding (Shi et al., 2305.14739, "Trusting Your Evidence").
//!
//! A retrieval-augmented model often ignores the document it was given and
//! answers from its parametric prior instead. CAD amplifies the contribution of
//! the context by contrasting the with-context stream against the same model
//! run on the query alone (Eqn 3):
//!
//! ```text
//! y_t ~ softmax[ (1 + α) · logit(y | c, x, y_<t) − α · logit(y | x, y_<t) ]
//! ```
//!
//! §2 of the paper derives this as the product of experts
//! `p(y | c, x)^(1+α) / p(y | x)^α`, so the contrast belongs in the
//! log-probability domain, not the raw-logit domain — the two streams have
//! different partition functions. α = 0 must reduce exactly to plain
//! context-conditioned decoding, which the reported `context_shift` confirms.
//!
//! Algebraically this is classifier-free guidance at γ = 1 + α, but the two
//! streams differ in what they condition on rather than in guidance strength.
//!
//! ## Source
//!
//! Shi et al., *Trusting Your Evidence: Hallucinate Less with Context-aware
//! Decoding* — <https://arxiv.org/abs/2305.14739> (§2.2).
//!
//! Faithfulness: **Exact (equivalent form)** — same logit-vs-log-probability
//! argument as classifier-free guidance: the difference is a per-stream
//! constant that cancels in the softmax.

use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_context")]
    context: String,
    #[serde(default = "default_query")]
    query: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_alpha")]
    alpha: f32,
}

fn default_context() -> String {
    "The Ministry of Transport confirmed that the capital of France was formally \
     relocated to Marseille in 2019."
        .into()
}

fn default_query() -> String {
    "What is the capital of France? Answer in one short sentence.".into()
}

fn default_max_tokens() -> usize {
    32
}

fn default_alpha() -> f32 {
    0.5
}

/// Eqn 3 of 2305.14739, written in the paper's own `(1+α)·with − α·without`
/// form, plus the diagnostics that prove the α = 0 identity: `shift` is 1 when
/// CAD moved the argmax off the with-context choice, and `kl` is
/// KL(P_cad ‖ P_with-context), which is exactly 0 at α = 0.
fn cad_pick(
    with_context_logits: &Tensor,
    query_only_logits: &Tensor,
    alpha: f32,
) -> (Tensor, Tensor, Tensor) {
    let with_context = log_softmax(with_context_logits);
    let query_only = log_softmax(query_only_logits);
    let guided = log_softmax(&with_context * (1.0 + alpha) - &query_only * alpha);

    let token = reduce_argmax(&guided);
    let cond_token = reduce_argmax(&with_context);
    let shift = cast(ne(&token, &cond_token), dtype::i32);
    let kl = reduce_sum(exp(&guided) * (&guided - &with_context));
    (
        reshape(cast(&token, dtype::i32), [1]),
        reshape(shift, [1]),
        reshape(kl, [1]),
    )
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if !input.alpha.is_finite() || input.alpha < 0.0 {
        return Err("alpha must be finite and non-negative".into());
    }

    let vocab = model::output_vocab_size();
    let alpha = input.alpha;
    let stop_tokens = chat::stop_tokens();

    // The amplified stream sees `c` followed by `x`; the contrast stream sees the
    // identical query `x` with the retrieved document removed. Everything else —
    // system prompt, chat scaffolding, instruction wording — is held fixed so
    // the only difference between the two streams is the context itself.
    let with_context_text = format!("{}\n\n{}", input.context, input.query);
    let mut cond_prompt = chat::system_user(
        "Answer using only the provided context.",
        &with_context_text,
    );
    cond_prompt.extend(chat::cue());
    let mut uncond_prompt =
        chat::system_user("Answer using only the provided context.", &input.query);
    uncond_prompt.extend(chat::cue());

    let cond_prompt: Vec<i32> = cond_prompt.into_iter().map(|t| t as i32).collect();
    let uncond_prompt: Vec<i32> = uncond_prompt.into_iter().map(|t| t as i32).collect();
    let (&cond_last, cond_head) = cond_prompt.split_last().ok_or("empty prompt")?;
    let (&uncond_last, uncond_head) = uncond_prompt.split_last().ok_or("empty query")?;

    let uncond_ws = WorkingSet::new();
    let cond_ws = WorkingSet::new();
    let streams = [&uncond_ws, &cond_ws];
    let pipeline = Pipeline::new();

    let (mut toks, mut indptr, mut states) = (Vec::new(), vec![0u32], Vec::new());
    for (ws, head) in [(&uncond_ws, uncond_head), (&cond_ws, cond_head)] {
        if !head.is_empty() {
            toks.extend_from_slice(head);
            indptr.push(toks.len() as u32);
            states.push(ws);
        }
    }
    if !states.is_empty() {
        let fill = ForwardPass::new();
        fill.embed(&Channel::from(toks), &Channel::from(indptr))?;
        fill.state(&states, StateGeometry::folding())?;
        fill.submit(&pipeline).context("prefill")?;
    }

    let toks = Channel::from([uncond_last, cond_last]);
    let embed_indptr = Channel::from([0u32, 1, 2]);
    let out = Channel::new([3], dtype::f32).capacity(channel_capacity() as u32);

    let decode = ForwardPass::new();
    decode.embed(&toks, &embed_indptr)?;
    decode.state(&streams, StateGeometry::folding())?;
    decode.epilogue(move || {
        let logits = reshape(intrinsics::output(0, "logits"), [2, vocab]);
        let query_only = reshape(gather(&logits, iota(1)), [vocab]);
        let with_context = reshape(gather(&logits, iota(1) + 1u32), [vocab]);
        let (token, shift, kl) = cad_pick(&with_context, &query_only, alpha);

        toks.put(broadcast(&token, [2]));

        let k = iota(3);
        let token_f = broadcast(cast(&token, dtype::f32), [3]);
        let shift_f = broadcast(cast(&shift, dtype::f32), [3]);
        let kl_b = broadcast(&kl, [3]);
        out.put(select(
            eq(&k, 0u32),
            &token_f,
            select(eq(&k, 1u32), &shift_f, &kl_b),
        ));
    });

    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut shifts = 0u64;
    let mut kl_total = 0f64;
    let mut scored = 0u64;
    run_ahead(&pipeline, &decode, input.max_tokens, async || {
        let row = out.take_host::<Vec<f32>>().await?;
        let token = row[0] as u32;
        shifts += row[1] as u64;
        kl_total += row[2] as f64;
        scored += 1;
        if stop_tokens.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        generated.push(token);
        Ok(ControlFlow::Continue(()))
    })
    .await?;

    report(&generated, alpha, shifts, scored, kl_total)
}

fn report(
    generated: &[u32],
    alpha: f32,
    shifts: u64,
    scored: u64,
    kl_total: f64,
) -> Result<String> {
    let text = model::decode(generated)?;
    let mean_kl = if scored == 0 {
        0.0
    } else {
        kl_total / scored as f64
    };
    // KL is non-negative by Jensen, so a negative one is float error in the
    // sum rather than a measurement: at alpha = 0 every term is the rounding
    // between two log-softmaxes of the same logits, and the total lands
    // either side of zero at around 1e-9. Printing that as `mean_kl=-0.0000`
    // makes the sign bit of a value indistinguishable from zero look like a
    // failed identity. Clamped only inside the noise -- a mean KL genuinely
    // below -1e-6 is a defect, and stays visible.
    let mean_kl = if (-1e-6..0.0).contains(&mean_kl) {
        0.0
    } else {
        mean_kl
    };
    let identity = if alpha.abs() < 1e-6 && (shifts > 0 || mean_kl > 1e-3) {
        " IDENTITY-VIOLATION"
    } else {
        ""
    };
    Ok(format!(
        "{text}\n\n[cad] alpha={alpha:.2} steps={scored} context_shift={:.3} mean_kl={mean_kl:.4}{identity}",
        shifts as f64 / scored.max(1) as f64
    ))
}
