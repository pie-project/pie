use inferlet::{Context, Result, sample::Logits};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max_tokens")] max_tokens: usize,
    #[serde(default = "default_delta")] delta: f32,
    #[serde(default = "default_gamma")] gamma: f32,
}
fn default_max_tokens() -> usize { 256 }
fn default_delta() -> f32 { 4.0 }
fn default_gamma() -> f32 { 0.5 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mut ctx = Context::new()?;
    let cue = inferlet::chat::cue();
    ctx.user(&input.prompt).append(&cue);
    ctx.flush().await?;

    let mut tokens = Vec::new();
    let mut last = *cue.last().unwrap();

    for _ in 0..input.max_tokens {
        let mut pass = ctx.forward();
        pass.input(&[last]);
        let p = pass.probe(0, Logits);
        let out = pass.execute().await?;

        let mut logits: Vec<f32> = out.logits(p).unwrap()
            .chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
        let vocab = logits.len() as u32;
        for t in green_list(last, vocab, input.gamma) { logits[t as usize] += input.delta; }
        last = argmax(&logits);
        tokens.push(last);
    }
    Ok(inferlet::model::decode(&tokens)?)
}

fn green_list(seed: u32, vocab: u32, gamma: f32) -> impl Iterator<Item = u32> {
    let mut s = seed.wrapping_mul(0x9E3779B1).wrapping_add(1);
    let n = (vocab as f32 * gamma) as u32;
    (0..n).map(move |_| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s % vocab
    })
}
fn argmax(xs: &[f32]) -> u32 {
    xs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}
