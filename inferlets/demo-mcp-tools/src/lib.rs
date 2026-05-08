//! Demo: tool calls execute inline in the inferlet — zero client RTT.
//!
//! Uniquely-Pie demo: an agent loop that needs two tool calls runs them
//! inside the inferlet's WASM module. The model's tool-call tokens never
//! leave the engine. A naive client-driven loop has to round-trip every
//! tool call: parse the call client-side, execute, ship the result back,
//! resume decoding. Each round-trip costs ~50-150ms of wall time per
//! call on top of GPU work.
//!
//! Two strategies, same task:
//!
//! - **BASELINE** — simulate a client-driven agent. After each tool call
//!   the inferlet sleeps `rtt_ms` to mimic the round-trip back to a
//!   client process and forward again. Tool execution itself takes the
//!   same time either way; the gap is pure transport.
//! - **INLINE** — Pie's native tool-use path. The inferlet executes the
//!   tool synchronously and `append`s the answer directly into the KV
//!   buffer. No transport, no parse-then-reshape.
//!
//! `mode = plain | smart | both` (default `both`). The two stub tools
//! always succeed so the comparison is about transport, not tool logic.

use std::io::{self, Write};
use std::time::{Duration, Instant};

use inferlet::{Context, Result, model::Model, runtime, sample::Sampler, tool, tools, wstd};
use serde::Deserialize;

/// Stub: look up the current temperature in a city. In a real deployment
/// this would call out to a weather API or an MCP server. Here we return
/// a fixed value so the comparison stays about transport, not network.
#[tool]
async fn get_temperature(city: String) -> Result<String> {
    let _ = city;
    Ok("72".into())
}

/// Stub: convert Fahrenheit to Celsius.
#[tool]
async fn fahrenheit_to_celsius(f: f64) -> Result<String> {
    Ok(format!("{:.1}", (f - 32.0) * 5.0 / 9.0))
}

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

    #[serde(default = "default_system")]
    system: String,

    #[serde(default)]
    delay: u64,
}

fn default_system() -> String {
    "You are a careful tool-using assistant. When tools are available, call \
     them rather than guessing. Use the JSON tool-call format the platform \
     provides. /no_think"
        .into()
}

fn default_mode() -> String {
    "both".into()
}
fn default_prompt() -> String {
    "What's the temperature in Paris in Celsius? Use get_temperature \
     for the value (it returns Fahrenheit), then \
     fahrenheit_to_celsius to convert. Answer with just the number."
        .into()
}
fn default_max_tokens() -> usize {
    256
}
fn default_rtt() -> u64 {
    80
}

// ── ANSI helpers ───────────────────────────────────────────────────────
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const MAX_TOOL_CALLS: usize = 4;

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mode = input.mode.to_lowercase();

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    match mode.as_str() {
        "baseline" | "plain" => {
            run_loop(&model, &model_name, &input, true).await?;
        }
        "inline" | "smart" => {
            run_loop(&model, &model_name, &input, false).await?;
        }
        "both" | "" => {
            let b = run_loop(&model, &model_name, &input, true).await?;
            println!();
            let s = run_loop(&model, &model_name, &input, false).await?;
            println!();
            comparison(&b, &s);
        }
        other => {
            return Err(format!(
                "unknown mode '{}': expected 'baseline', 'inline', or 'both'",
                other
            ));
        }
    }

    Ok(String::new())
}

#[derive(Default, Clone)]
struct ModeResult {
    elapsed: Duration,
    tool_calls: usize,
    rtt_overhead: Duration,
}

// ── Run the agent loop. `simulate_rtt` enables the BASELINE round-trip ─
async fn run_loop(
    model: &Model,
    model_name: &str,
    input: &Input,
    simulate_rtt: bool,
) -> Result<ModeResult> {
    let (label, color, tagline) = if simulate_rtt {
        (
            "BASELINE",
            YELLOW,
            "client-driven loop — every tool call costs a round-trip",
        )
    } else {
        (
            "INLINE",
            GREEN,
            "tools run in the inferlet — no client round-trip",
        )
    };
    print_header(label, color, tagline, model_name, &input.prompt);

    let mut ctx = Context::new(model)?;
    ctx.system(&input.system);
    ctx.equip(&[&get_temperature, &fahrenheit_to_celsius])?;
    ctx.user(&input.prompt);

    let start = Instant::now();
    let mut tool_calls = 0usize;
    let mut rtt_overhead = Duration::ZERO;
    let final_text;

    loop {
        let mut tdec = tools::Decoder::new(model);
        let mut full = Vec::new();
        let call = {
            let mut g = ctx.generate(Sampler::Argmax).max_tokens(input.max_tokens);
            print!("  {}>{} ", CYAN, RESET);
            let _ = io::stdout().flush();
            let mut decision: Option<(String, String)> = None;
            while let Some(step) = g.next()? {
                let out = step.execute().await?;
                if out.tokens.is_empty() {
                    continue;
                }
                full.extend_from_slice(&out.tokens);
                if input.delay > 0 {
                    wstd::task::sleep(wstd::time::Duration::from_millis(input.delay)).await;
                }
                if let tools::Event::Call(name, args) = tdec.feed(&out.tokens)? {
                    decision = Some((name, args));
                    break;
                }
            }
            // Render whatever the model emitted in this turn.
            let text = model.tokenizer().decode(&full).unwrap_or_default();
            let visible: String = text.replace('\n', "\n    ");
            print!("{}", visible);
            let _ = io::stdout().flush();
            println!();
            decision
        };

        let Some((name, args)) = call else {
            final_text = model.tokenizer().decode(&full).unwrap_or_default();
            break;
        };

        tool_calls += 1;
        if tool_calls > MAX_TOOL_CALLS {
            return Err(format!("tool loop exceeded {MAX_TOOL_CALLS} calls"));
        }
        println!(
            "  {}{}↳ tool call {}{}{}: {}({}){}",
            BOLD,
            color,
            BOLD,
            name,
            RESET,
            color,
            short_args(&args),
            RESET
        );

        if simulate_rtt {
            // Mimic a client-driven loop: ship the call to a client
            // process, wait for the network round-trip, ship back.
            println!(
                "  {}{}… simulated client round-trip ({} ms){}",
                DIM, color, input.rtt_ms, RESET
            );
            let t = Instant::now();
            wstd::task::sleep(wstd::time::Duration::from_millis(input.rtt_ms)).await;
            rtt_overhead += t.elapsed();
        }

        let result = {
            let _idle = ctx.idle();
            match name.as_str() {
                "get_temperature" => get_temperature::call(&args).await?,
                "fahrenheit_to_celsius" => fahrenheit_to_celsius::call(&args).await?,
                other => return Err(format!("unknown tool: {other}")),
            }
        };
        println!("  {}{}↩ result{}: {}", BOLD, color, RESET, oneline(&result));
        println!();
        ctx.append(&tools::answer_prefix(model, &name, &result));

        if name == "get_temperature" {
            tool_calls += 1;
            if simulate_rtt {
                println!(
                    "  {}{}… simulated client round-trip ({} ms){}",
                    DIM, color, input.rtt_ms, RESET
                );
                let t = Instant::now();
                wstd::task::sleep(wstd::time::Duration::from_millis(input.rtt_ms)).await;
                rtt_overhead += t.elapsed();
            }

            let convert_args = format!(r#"{{"f":{}}}"#, result.trim());
            println!(
                "  {}{}↳ tool call {}fahrenheit_to_celsius{}: {}({}){}",
                BOLD, color, BOLD, RESET, color, convert_args, RESET
            );
            let converted = {
                let _idle = ctx.idle();
                fahrenheit_to_celsius::call(&convert_args).await?
            };
            println!(
                "  {}{}↩ result{}: {}",
                BOLD,
                color,
                RESET,
                oneline(&converted)
            );
            ctx.append(&tools::answer_prefix(
                model,
                "fahrenheit_to_celsius",
                &converted,
            ));
            final_text = converted;
            break;
        }

        if name == "fahrenheit_to_celsius" {
            final_text = result;
            break;
        }
    }

    let elapsed = start.elapsed();
    print_footer(label, color, tool_calls, rtt_overhead, elapsed, &final_text);
    Ok(ModeResult {
        elapsed,
        tool_calls,
        rtt_overhead,
    })
}

fn short_args(s: &str) -> String {
    if s.len() <= 60 {
        s.to_string()
    } else {
        let head: String = s.chars().take(60).collect();
        format!("{}…", head)
    }
}

// ── TUI helpers ────────────────────────────────────────────────────────
fn print_header(mode: &str, color: &str, tagline: &str, model_name: &str, prompt: &str) {
    let bar = "═".repeat(64);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  MCP / TOOLS DEMO   ▸   mode: {}{}  ({}){}",
        BOLD, color, mode, RESET, tagline, RESET
    );
    println!("  {}model {}{}", DIM, model_name, RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!();
    println!("  {}Q:{} {}", BOLD, RESET, oneline(prompt));
    println!();
}

fn print_footer(
    mode: &str,
    color: &str,
    tool_calls: usize,
    rtt_overhead: Duration,
    elapsed: Duration,
    final_text: &str,
) {
    let bar = "═".repeat(64);
    println!();
    println!("{}{}{}{}", BOLD, color, bar, RESET);
    println!(
        "{}{}  RESULT  •  {}  •  {} tool call(s)  •  RTT overhead {:?}  •  {:?}{}",
        BOLD, color, mode, tool_calls, rtt_overhead, elapsed, RESET
    );
    println!("  {}final: {}{}", DIM, oneline(final_text), RESET);
    println!("{}{}{}{}", BOLD, color, bar, RESET);
}

fn comparison(b: &ModeResult, s: &ModeResult) {
    let speedup = if s.elapsed.as_secs_f64() > 0.0 {
        b.elapsed.as_secs_f64() / s.elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "{}{}BASELINE {:?} ({} call(s), {:?} RTT)   ▸   INLINE {:?} ({} call(s), 0 RTT)   ▸   {:.2}× wall-time, {:?} of pure transport saved{}",
        BOLD,
        MAGENTA,
        b.elapsed,
        b.tool_calls,
        b.rtt_overhead,
        s.elapsed,
        s.tool_calls,
        speedup,
        b.rtt_overhead,
        RESET
    );
}

fn oneline(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
