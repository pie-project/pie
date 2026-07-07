//! Demo: inferlet pipeline via `messaging::broadcast` / `subscribe`.
//!
//! Uniquely-Pie demo: a 3-stage writer's-room pipeline (idea → plot →
//! dialogue) where each stage publishes its result on a topic and the
//! next stage subscribes. No coordinator client process pulls and
//! pushes between stages — inferlets in the same engine talk to each
//! other directly through the broker. A naive client-driven flow has
//! to ship each intermediate result back to a coordinator and forward
//! it to the next agent, costing a round-trip per hop.
//!
//! For demo purposes the three stages run inside one inferlet so the
//! comparison is reproducible in one `pie run`. The same topic API can
//! connect separate inferlets when they are attached to one shared,
//! long-running engine.
//!
//! Two strategies, same pipeline:
//!
//! - **BASELINE** — every stage hand-off costs a simulated `rtt_ms`
//!   sleep, mimicking client RTT.
//! - **PUBSUB** — `messaging::broadcast` / `subscribe`; hand-offs are
//!   essentially free (in-process broker).
//!
//! `mode = plain | smart | both | idea | plot | dialogue` (default
//! `both`).

use std::io::{self, Write};
use std::time::{Duration, Instant};

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, messaging, model, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_mode")]
    mode: String,

    #[serde(default = "default_prompt")]
    prompt: String,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_rtt")]
    rtt_ms: u64,

    #[serde(default = "default_sys_idea")]
    sys_idea: String,

    #[serde(default = "default_sys_plot")]
    sys_plot: String,

    #[serde(default = "default_sys_dialogue")]
    sys_dialogue: String,

    #[serde(default)]
    delay: u64,
}

fn default_sys_idea() -> String {
    "You are a brisk story-idea generator. Reply with one short sentence \
     that establishes the central conflict."
        .into()
}
fn default_sys_plot() -> String {
    "You are a brisk plot writer. Read the concept, then write three short \
     beat lines: setup, confrontation, resolution."
        .into()
}
fn default_sys_dialogue() -> String {
    "You are a brisk dialogue writer. Read the concept and beats, then \
     write a single line of climax dialogue from the protagonist."
        .into()
}

fn default_mode() -> String {
    "both".into()
}
fn default_prompt() -> String {
    "A short story about daydreaming in a park.".into()
}
fn default_max_tokens() -> usize {
    80
}
fn default_rtt() -> u64 {
    100
}

// Topic names — same shape as `inferlets/agent-swarm/`.
const TOPIC_IDEA_TO_PLOT: &str = "demo-idea-to-plot";
const TOPIC_PLOT_TO_DIALOGUE: &str = "demo-plot-to-dialogue";

// ── ANSI helpers ───────────────────────────────────────────────────────
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mode = input.mode.to_lowercase();

    let model_name = inferlet::model::name();

    match mode.as_str() {
        "baseline" | "plain" => {
            run_pipeline(&model_name, &input, true).await?;
        }
        "pubsub" | "smart" => {
            run_pipeline(&model_name, &input, false).await?;
        }
        "idea" | "idea-generator" => {
            run_idea_role(&model_name, &input).await?;
        }
        "plot" | "plot-developer" => {
            run_plot_role(&model_name, &input).await?;
        }
        "dialogue" | "dialogue-writer" => {
            run_dialogue_role(&model_name, &input).await?;
        }
        "both" | "" => {
            let b = run_pipeline(&model_name, &input, true).await?;
            println!();
            let p = run_pipeline(&model_name, &input, false).await?;
            println!();
            comparison(&b, &p);
        }
        other => {
            return Err(format!(
                "unknown mode '{}': expected 'baseline', 'pubsub', 'idea', 'plot', 'dialogue', or 'both'",
                other
            ));
        }
    }
    Ok(String::new())
}

#[derive(Default, Clone)]
struct ModeResult {
    elapsed: Duration,
    rtt_overhead: Duration,
    hops: usize,
}

async fn run_pipeline(
    model_name: &str,
    input: &Input,
    simulate_rtt: bool,
) -> Result<ModeResult> {
    let (label, color, tagline) = if simulate_rtt {
        (
            "BASELINE",
            YELLOW,
            "client-orchestrated pipeline — every hop is a round-trip",
        )
    } else {
        (
            "PUBSUB",
            GREEN,
            "messaging::broadcast / subscribe — direct inferlet hops",
        )
    };
    print_header(label, color, tagline, model_name, &input.prompt);

    let start = Instant::now();
    let mut rtt_overhead = Duration::ZERO;
    let mut hops = 0usize;

    // ── stage 1: idea ──
    println!("  {}{}stage 1 ▸ idea generator{}", BOLD, color, RESET);
    let idea = run_stage(
        &input.sys_idea,
        &input.prompt,
        input.max_tokens,
        input.delay,
        color,
    )
    .await?;
    let idea = idea.trim().to_string();

    let plot_input = if simulate_rtt {
        let t = relay_via_client(input.rtt_ms, color, "idea → coordinator → plot worker").await;
        rtt_overhead += t;
        hops += 1;
        idea.clone()
    } else {
        // Pie path: subscribe first (broker captures from now on), then
        // broadcast, then await the message. In a real multi-inferlet
        // pipeline the subscriber would be a separate `pie run` that
        // already subscribed before the producer broadcasts.
        let mut sub = messaging::subscribe(TOPIC_IDEA_TO_PLOT);
        messaging::broadcast(TOPIC_IDEA_TO_PLOT, &idea);
        let received = sub.next().await.unwrap_or_default();
        println!(
            "  {}{}↪ broadcast on \"{}\" → subscriber received{}",
            DIM, color, TOPIC_IDEA_TO_PLOT, RESET
        );
        hops += 1;
        received
    };
    println!();

    // ── stage 2: plot ──
    println!("  {}{}stage 2 ▸ plot developer{}", BOLD, color, RESET);
    let plot = run_stage(
        &input.sys_plot,
        &format!("Concept: {}\nWrite the beats.", plot_input),
        input.max_tokens,
        input.delay,
        color,
    )
    .await?;
    let plot = plot.trim().to_string();

    let dialogue_input = if simulate_rtt {
        let t = relay_via_client(input.rtt_ms, color, "plot → coordinator → dialogue worker").await;
        rtt_overhead += t;
        hops += 1;
        format!("{}\n\nPlot:\n{}", plot_input, plot)
    } else {
        let combined = format!("{}\n\nPlot:\n{}", plot_input, plot);
        let mut sub = messaging::subscribe(TOPIC_PLOT_TO_DIALOGUE);
        messaging::broadcast(TOPIC_PLOT_TO_DIALOGUE, &combined);
        let received = sub.next().await.unwrap_or_default();
        println!(
            "  {}{}↪ broadcast on \"{}\" → subscriber received{}",
            DIM, color, TOPIC_PLOT_TO_DIALOGUE, RESET
        );
        hops += 1;
        received
    };
    println!();

    // ── stage 3: dialogue ──
    println!("  {}{}stage 3 ▸ dialogue writer{}", BOLD, color, RESET);
    let dialogue = run_stage(
        &input.sys_dialogue,
        &format!(
            "Story so far:\n{}\nWrite ONE line of climax dialogue.",
            dialogue_input
        ),
        input.max_tokens,
        input.delay,
        color,
    )
    .await?;
    let _ = dialogue;

    let elapsed = start.elapsed();
    print_footer(label, color, hops, rtt_overhead, elapsed);
    Ok(ModeResult {
        elapsed,
        rtt_overhead,
        hops,
    })
}

// ── Separate-inferlet roles ──────────────────────────────────────────
//
// Launch these roles only when the invocations attach to the same shared
// engine. Start subscribers first:
//   pie run demo-messaging -- --mode dialogue
//   pie run demo-messaging -- --mode plot
//   pie run demo-messaging -- --mode idea --prompt "A park story"
async fn run_idea_role(model_name: &str, input: &Input) -> Result<()> {
    print_header(
        "IDEA",
        GREEN,
        "producer: generate an idea and broadcast to plot worker",
        model_name,
        &input.prompt,
    );
    let idea = run_stage(
        &input.sys_idea,
        &input.prompt,
        input.max_tokens,
        input.delay,
        GREEN,
    )
    .await?;
    messaging::broadcast(TOPIC_IDEA_TO_PLOT, idea.trim());
    println!("  {}broadcast on \"{}\"{}", DIM, TOPIC_IDEA_TO_PLOT, RESET);
    Ok(())
}

async fn run_plot_role(model_name: &str, input: &Input) -> Result<()> {
    print_header(
        "PLOT",
        GREEN,
        "consumer/producer: subscribe to idea, then broadcast plot",
        model_name,
        &input.prompt,
    );
    println!("  {}waiting on \"{}\"{}", DIM, TOPIC_IDEA_TO_PLOT, RESET);
    let mut sub = messaging::subscribe(TOPIC_IDEA_TO_PLOT);
    let idea = sub.next().await.unwrap_or_default();
    println!("  {}received idea: {}{}", DIM, oneline(&idea), RESET);
    let plot = run_stage(
        &input.sys_plot,
        &format!("Concept: {}\nWrite the beats.", idea.trim()),
        input.max_tokens,
        input.delay,
        GREEN,
    )
    .await?;
    let combined = format!("{}\n\nPlot:\n{}", idea.trim(), plot.trim());
    messaging::broadcast(TOPIC_PLOT_TO_DIALOGUE, &combined);
    println!(
        "  {}broadcast on \"{}\"{}",
        DIM, TOPIC_PLOT_TO_DIALOGUE, RESET
    );
    Ok(())
}

async fn run_dialogue_role(model_name: &str, input: &Input) -> Result<()> {
    print_header(
        "DIALOGUE",
        GREEN,
        "consumer: subscribe to plot and write the final line",
        model_name,
        &input.prompt,
    );
    println!(
        "  {}waiting on \"{}\"{}",
        DIM, TOPIC_PLOT_TO_DIALOGUE, RESET
    );
    let mut sub = messaging::subscribe(TOPIC_PLOT_TO_DIALOGUE);
    let plot = sub.next().await.unwrap_or_default();
    println!("  {}received plot context{}", DIM, RESET);
    let _dialogue = run_stage(
        &input.sys_dialogue,
        &format!(
            "Story so far:\n{}\nWrite ONE line of climax dialogue.",
            plot
        ),
        input.max_tokens,
        input.delay,
        GREEN,
    )
    .await?;
    Ok(())
}

async fn relay_via_client(rtt_ms: u64, color: &str, what: &str) -> Duration {
    println!(
        "  {}{}… simulated client relay ({}): {} ms{}",
        DIM, color, what, rtt_ms, RESET
    );
    let t = Instant::now();
    inferlet::sleep(std::time::Duration::from_millis(rtt_ms)).await;
    t.elapsed()
}

// ── Run one pipeline stage ────────────────────────────────────────────
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

fn pass_carries(stop_empty: bool, max_tokens: usize, produced_token_index: usize) -> bool {
    !(stop_empty && max_tokens == produced_token_index)
}

/// Run-ahead (pipelined) chat-EOS decode with depth-1 EOS rollback.
async fn decode_pipelined(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    s: &sampler::LoweredSampler,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: &[u32],
) -> Result<Vec<u32>> {
    let pending = if prompt.is_empty() { vec![0u32] } else { prompt };
    let mut out: Vec<u32> = Vec::with_capacity(max_tokens);
    if max_tokens == 0 {
        return Ok(out);
    }
    let prime_carry = pass_carries(stop.is_empty(), max_tokens, 1);
    let mut producer = carrier::submit_pass(kv, seq_len, fresh, s, &pending, prime_carry)?;
    let mut generated = 0usize;
    loop {
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            let carry = pass_carries(stop.is_empty(), max_tokens, generated + 2);
            Some(carrier::submit_pass(kv, seq_len, fresh, s, &[0u32], carry)?)
        } else {
            None
        };
        let token = read_token(producer).await?;
        if stop.contains(&token) {
            if let Some(c) = consumer {
                carrier::discard_pass(c, seq_len).await;
            }
            break;
        }
        out.push(token);
        generated += 1;
        match consumer {
            Some(c) => producer = c,
            None => break,
        }
    }
    Ok(out)
}

/// Low-level ① rewrite (chat-EOS, pipelined): each pipeline stage's decode runs
/// on the run-ahead carrier (NO `Context`/`Generator`/`Sampler` facade; each stage
/// is a fresh single generation — the pub/sub hand-off between stages passes TEXT,
/// not context). Per-token streaming traded for the pipelined overlap; the final
/// `<think>`-stripped text is identical.
async fn run_stage(
    system: &str,
    user: &str,
    max_tokens: usize,
    _delay_ms: u64,
    color: &str,
) -> Result<String> {
    let vocab = model::output_vocab_size();
    let s = sampler::sampler_program(SamplerSpec::Argmax, vocab)?;
    let stop = chat::stop_tokens();

    let mut prompt = chat::system_user(system, &format!("{} /no_think", user.trim()));
    prompt.extend(chat::cue());

    print!("  {}>{}{}{} ", color, RESET, CYAN, RESET);
    let _ = io::stdout().flush();

    let kv = KvWorkingSet::new();
    let mut seq = 0u32;
    let mut fresh = true;
    let toks = decode_pipelined(&kv, &mut seq, &mut fresh, &s, prompt, max_tokens, &stop).await?;

    let mut decoder = chat::Decoder::new();
    let mut text = String::new();
    match decoder.feed(&toks)? {
        chat::Event::Delta(s) | chat::Event::Done(s) => text.push_str(&s),
        _ => {}
    }
    println!();
    Ok(strip_think_blocks(&text))
}

fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        match rest.find("<think>") {
            None => {
                out.push_str(rest);
                break;
            }
            Some(start) => {
                out.push_str(&rest[..start]);
                let after_open = &rest[start + "<think>".len()..];
                match after_open.find("</think>") {
                    Some(end) => rest = &after_open[end + "</think>".len()..],
                    None => break,
                }
            }
        }
    }
    out
}

struct ThinkStripper {
    in_think: bool,
    pending: String,
}

impl ThinkStripper {
    fn new() -> Self {
        Self {
            in_think: false,
            pending: String::new(),
        }
    }

    fn process(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(idx) = self.pending.find("</think>") {
                    self.pending = self.pending.split_off(idx + "</think>".len());
                    self.in_think = false;
                    continue;
                }
                let len = self.pending.len();
                if len > 7 {
                    self.pending = self.pending.split_off(len - 7);
                }
                break;
            } else {
                if let Some(idx) = self.pending.find("<think>") {
                    out.push_str(&self.pending[..idx]);
                    self.pending = self.pending.split_off(idx + "<think>".len());
                    self.in_think = true;
                    continue;
                }
                let len = self.pending.len();
                let safe = len.saturating_sub(6);
                if safe > 0 {
                    let head: String = self.pending.drain(..safe).collect();
                    out.push_str(&head);
                }
                break;
            }
        }
        out
    }
}

// ── TUI helpers ────────────────────────────────────────────────────────
fn print_header(mode: &str, color: &str, tagline: &str, model_name: &str, prompt: &str) {
    let bar = "═".repeat(64);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  MESSAGING DEMO   ▸   mode: {}{}  ({}){}",
        BOLD, color, mode, RESET, tagline, RESET
    );
    println!("  {}model {}{}", DIM, model_name, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!();
    println!("  {}Seed:{} {}", BOLD, RESET, oneline(prompt));
    println!();
}

fn print_footer(mode: &str, color: &str, hops: usize, rtt_overhead: Duration, elapsed: Duration) {
    let bar = "═".repeat(64);
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  {} pipeline hop(s)  •  RTT overhead {:?}  •  {:?}{}",
        BOLD, color, mode, hops, rtt_overhead, elapsed, RESET
    );
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn comparison(b: &ModeResult, p: &ModeResult) {
    let speedup = if p.elapsed.as_secs_f64() > 0.0 {
        b.elapsed.as_secs_f64() / p.elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "{}{}BASELINE {:?} ({} hops, {:?} RTT)   ▸   PUBSUB {:?} (0 RTT)   ▸   {:.2}× wall-time, {:?} of pure transport saved{}",
        BOLD, MAGENTA, b.elapsed, b.hops, b.rtt_overhead, p.elapsed, speedup, b.rtt_overhead, RESET
    );
}

fn oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
