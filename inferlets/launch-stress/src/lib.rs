//! Stress / corner-case driver for the v2 `runtime::launch` API.
//!
//! Single binary that runs as caller or callee depending on the top-level
//! `role` field. The Python test invokes the caller role with a scenario
//! name; the caller in turn launches the callee role of the same binary
//! with a directive.

use futures::future;
use inferlet::{Context, Result, chat, launch, messaging, model::Model, runtime, sample::Sampler};
use serde::Deserialize;
use serde_json::json;
use inferlet::wstd::future::FutureExt as _;
use inferlet::wstd::task;

const SELF_NAME: &str = "launch-stress@0.1.0";

#[derive(Deserialize)]
struct Input {
    role: String,
    #[serde(default)]
    scenario: String,
    #[serde(default)]
    callee: String,
    #[serde(default)]
    n: usize,
    #[serde(default)]
    directive: String,
    #[serde(default)]
    payload: String,
    #[serde(default)]
    err_msg: String,
    #[serde(default)]
    notify_topic: String,
    #[serde(default)]
    sleep_ms: u64,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    match input.role.as_str() {
        "callee" => run_callee(input).await,
        "caller" => run_caller(input).await,
        other => Err(format!("unknown role: {other}")),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Callee directives
// ────────────────────────────────────────────────────────────────────────────

async fn run_callee(input: Input) -> Result<String> {
    match input.directive.as_str() {
        "" | "echo" => Ok(input.payload),
        "err" => {
            let msg = if input.err_msg.is_empty() {
                "boom".to_string()
            } else {
                input.err_msg
            };
            Err(msg)
        }
        "username" => Ok(runtime::username()),
        "multiline" => Ok(format!("line-1\nline-2\nline-3\nbody:{}", input.payload)),
        "unicode" => Ok(format!("✓ {} → 漢字 🎉", input.payload)),
        "notify_then_echo" => {
            if !input.notify_topic.is_empty() {
                messaging::push(&input.notify_topic, &format!("callee saw: {}", input.payload));
            }
            Ok(input.payload)
        }
        "generate" => generate(&input.payload).await,
        "sleep_then_echo" => {
            if input.sleep_ms > 0 {
                task::sleep(inferlet::wstd::time::Duration::from_millis(input.sleep_ms)).await;
            }
            if !input.notify_topic.is_empty() {
                messaging::push(&input.notify_topic, &format!("callee done: {}", input.payload));
            }
            Ok(input.payload)
        }
        "recurse" => {
            let inner = json!({
                "role": "callee",
                "directive": "echo",
                "payload": format!("nested[{}]", input.payload),
            })
            .to_string();
            let inner_result = launch(SELF_NAME, &inner)?.await?;
            Ok(format!("outer({})", inner_result))
        }
        "descend" => {
            let (depth_str, tag) = input
                .payload
                .split_once('|')
                .ok_or_else(|| "descend payload must be 'N|tag'".to_string())?;
            let depth: usize = depth_str
                .parse()
                .map_err(|e: std::num::ParseIntError| e.to_string())?;
            if depth == 0 {
                Ok(format!("base[{}]", tag))
            } else {
                let inner = json!({
                    "role": "callee",
                    "directive": "descend",
                    "payload": format!("{}|{}", depth - 1, tag),
                })
                .to_string();
                let inner = launch(SELF_NAME, &inner)?.await?;
                Ok(format!("d{}({})", depth, inner))
            }
        }
        "fanout_inner" => {
            let (n_str, tag) = input
                .payload
                .split_once('|')
                .ok_or_else(|| "fanout_inner payload must be 'N|tag'".to_string())?;
            let n: usize = n_str
                .parse()
                .map_err(|e: std::num::ParseIntError| e.to_string())?;
            let mut futs = Vec::with_capacity(n);
            for i in 0..n {
                let p = callee_payload("echo", &format!("{}-gc{}", tag, i));
                let child = launch(SELF_NAME, &p)?;
                futs.push(async move { child.await });
            }
            let results = future::join_all(futs).await;
            let mut all = Vec::with_capacity(n);
            for (i, r) in results.into_iter().enumerate() {
                all.push(r.map_err(|e| format!("grandchild {} failed: {}", i, e))?);
            }
            Ok(format!("inner[{}]:{}", tag, all.join(",")))
        }
        other => Err(format!("unknown directive: {other}")),
    }
}

async fn generate(prompt: &str) -> Result<String> {
    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("no models available")?;
    let model = Model::load(&model_name)?;
    let mut ctx = Context::new(&model)?;
    ctx.system("Answer with one short sentence.")
        .user(&format!("{} /no_think", prompt))
        .cue();
    let mut decoder = chat::Decoder::new(&model);
    let mut text = String::new();
    let mut g = ctx
        .generate(Sampler::Argmax)
        .max_tokens(32)
        .stop(&chat::stop_tokens(&model));
    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        match decoder.feed(&out.tokens)? {
            chat::Event::Delta(s) => text.push_str(&s),
            chat::Event::Done(s) => {
                text = s;
                break;
            }
            _ => {}
        }
    }
    Ok(strip_think_blocks(&text).trim().to_string())
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

// ────────────────────────────────────────────────────────────────────────────
// Caller scenarios
// ────────────────────────────────────────────────────────────────────────────

async fn run_caller(input: Input) -> Result<String> {
    let callee = if input.callee.is_empty() {
        SELF_NAME.to_string()
    } else {
        input.callee.clone()
    };
    match input.scenario.as_str() {
        // ── v1-era scenarios ported to v2 ──
        "fanout_echo" => scenario_fanout_echo(&callee, input.n.max(1)).await,
        "sequential_echo" => scenario_sequential_echo(&callee, input.n.max(1)).await,
        "error_propagation" => scenario_error_propagation(&callee).await,
        "invalid_program_format" => scenario_invalid_program_format().await,
        "missing_program" => scenario_missing_program().await,
        "nested_chain" => scenario_nested_chain(&callee).await,
        "multiline" => scenario_multiline(&callee).await,
        "unicode" => scenario_unicode(&callee).await,
        "username_inherited" => scenario_username_inherited(&callee).await,
        "mixed_outcomes" => scenario_mixed_outcomes(&callee, input.n.max(1)).await,
        "infer_once" => scenario_infer_once(&callee).await,
        "fanout_giant" => scenario_fanout_giant(&callee, input.n.max(1)).await,
        "long_payload" => scenario_long_payload(&callee, input.n.max(1)).await,
        "deep_nesting" => scenario_deep_nesting(&callee, input.n.max(1)).await,
        "repeated_fanout" => scenario_repeated_fanout(&callee, input.n.max(1)).await,
        "fanout_infer" => scenario_fanout_infer(&callee, input.n.max(1)).await,
        "nested_fanout" => scenario_nested_fanout(&callee, input.n.max(1)).await,
        "concurrent_self_in_callee" => scenario_concurrent_self_in_callee(&callee).await,
        // ── v2-only scenarios ──
        "pid_is_uuid" => scenario_pid_is_uuid(&callee).await,
        "drop_detach" => scenario_drop_detach(&callee).await,
        "cancel_mid_flight" => scenario_cancel_mid_flight(&callee).await,
        "timeout_then_cancel" => scenario_timeout_then_cancel(&callee).await,
        "cancel_after_done" => scenario_cancel_after_done(&callee).await,
        "fanout_with_cancel" => scenario_fanout_with_cancel(&callee, input.n.max(1)).await,
        other => Err(format!("unknown scenario: {other}")),
    }
}

fn callee_payload(directive: &str, payload: &str) -> String {
    json!({
        "role": "callee",
        "directive": directive,
        "payload": payload,
    })
    .to_string()
}

fn callee_payload_with(values: serde_json::Value) -> String {
    values.to_string()
}

// ── Ported v1 scenarios ────────────────────────────────────────────────────

async fn scenario_fanout_echo(callee: &str, n: usize) -> Result<String> {
    println!("[scenario] fanout_echo n={}", n);
    let mut futs = Vec::with_capacity(n);
    for i in 0..n {
        let child = launch(callee, &callee_payload("echo", &format!("payload_{}", i)))?;
        futs.push(async move { (i, child.await) });
    }
    let results = future::join_all(futs).await;
    for (i, res) in &results {
        let expected = format!("payload_{}", i);
        match res {
            Ok(s) if *s == expected => {}
            Ok(s) => return Err(format!("mismatch at {}: expected {:?} got {:?}", i, expected, s)),
            Err(e) => return Err(format!("call {} failed: {}", i, e)),
        }
    }
    Ok(format!("all {} concurrent echoes returned correctly", n))
}

async fn scenario_sequential_echo(callee: &str, n: usize) -> Result<String> {
    println!("[scenario] sequential_echo n={}", n);
    for i in 0..n {
        let got = launch(callee, &callee_payload("echo", &format!("seq_{}", i)))?
            .await?;
        let expected = format!("seq_{}", i);
        if got != expected {
            return Err(format!("mismatch at {}: expected {:?} got {:?}", i, expected, got));
        }
    }
    Ok(format!("all {} sequential echoes returned correctly", n))
}

async fn scenario_error_propagation(callee: &str) -> Result<String> {
    println!("[scenario] error_propagation");
    let payload = callee_payload_with(json!({
        "role": "callee",
        "directive": "err",
        "err_msg": "deliberate failure from callee",
    }));
    match launch(callee, &payload)?.await {
        Ok(v) => Err(format!("expected error, got Ok({:?})", v)),
        Err(e) if e.contains("deliberate failure from callee") => {
            Ok(format!("got expected error: {}", e))
        }
        Err(e) => Err(format!("error message lost: {}", e)),
    }
}

async fn scenario_invalid_program_format() -> Result<String> {
    println!("[scenario] invalid_program_format");
    match launch("not-a-valid-name", &callee_payload("echo", "x")) {
        Ok(_) => Err("expected launch error".into()),
        Err(e) => Ok(format!("rejected with: {}", e)),
    }
}

async fn scenario_missing_program() -> Result<String> {
    println!("[scenario] missing_program");
    match launch("definitely-not-a-real-inferlet@9.9.9", &callee_payload("echo", "x")) {
        Ok(_) => Err("expected install error".into()),
        Err(e) => Ok(format!("rejected with: {}", e)),
    }
}

async fn scenario_nested_chain(callee: &str) -> Result<String> {
    println!("[scenario] nested_chain");
    let payload = callee_payload_with(json!({
        "role": "callee",
        "directive": "recurse",
        "payload": "hello",
    }));
    let got = launch(callee, &payload)?.await?;
    let expected = "outer(nested[hello])";
    if got != expected {
        return Err(format!("expected {:?} got {:?}", expected, got));
    }
    Ok(format!("got expected nested response: {}", got))
}

async fn scenario_multiline(callee: &str) -> Result<String> {
    println!("[scenario] multiline");
    let got = launch(callee, &callee_payload("multiline", "WITHBODY"))?.await?;
    let expected = "line-1\nline-2\nline-3\nbody:WITHBODY";
    if got != expected {
        return Err(format!("multiline mismatch: got {:?}", got));
    }
    Ok("multiline round-trip preserved newlines".to_string())
}

async fn scenario_unicode(callee: &str) -> Result<String> {
    println!("[scenario] unicode");
    let got = launch(callee, &callee_payload("unicode", "héllo"))?.await?;
    let expected = "✓ héllo → 漢字 🎉";
    if got != expected {
        return Err(format!("unicode mismatch: got {:?}", got));
    }
    Ok(format!("unicode round-trip: {}", got))
}

async fn scenario_username_inherited(callee: &str) -> Result<String> {
    println!("[scenario] username_inherited");
    let mine = runtime::username();
    let payload = callee_payload_with(json!({"role": "callee", "directive": "username"}));
    let got = launch(callee, &payload)?.await?;
    if got != mine {
        return Err(format!("username NOT inherited: caller={:?} callee={:?}", mine, got));
    }
    Ok(format!("username inherited correctly: {}", got))
}

async fn scenario_mixed_outcomes(callee: &str, n: usize) -> Result<String> {
    println!("[scenario] mixed_outcomes n={}", n);
    let mut futs = Vec::with_capacity(n);
    for i in 0..n {
        let payload = if i % 2 == 0 {
            callee_payload("echo", &format!("ok_{}", i))
        } else {
            callee_payload_with(json!({
                "role": "callee",
                "directive": "err",
                "err_msg": format!("planned_err_{}", i),
            }))
        };
        let child = launch(callee, &payload)?;
        futs.push(async move { (i, child.await) });
    }
    let results = future::join_all(futs).await;
    let mut ok_count = 0usize;
    let mut err_count = 0usize;
    for (i, res) in &results {
        if i % 2 == 0 {
            match res {
                Ok(s) if *s == format!("ok_{}", i) => ok_count += 1,
                Ok(s) => return Err(format!("even {} wrong body: {:?}", i, s)),
                Err(e) => return Err(format!("even {} unexpectedly failed: {}", i, e)),
            }
        } else {
            match res {
                Ok(v) => return Err(format!("odd {} unexpectedly succeeded: {:?}", i, v)),
                Err(e) if e.contains(&format!("planned_err_{}", i)) => err_count += 1,
                Err(e) => return Err(format!("odd {} wrong error: {}", i, e)),
            }
        }
    }
    Ok(format!("mixed_outcomes: {} ok, {} err (n={})", ok_count, err_count, n))
}

async fn scenario_infer_once(callee: &str) -> Result<String> {
    println!("[scenario] infer_once");
    let got = launch(callee, &callee_payload("generate", "What is 2+2?"))?.await?;
    if got.is_empty() {
        return Err("empty generation".into());
    }
    Ok(format!("generated: {}", got))
}

async fn scenario_fanout_giant(callee: &str, n: usize) -> Result<String> {
    println!("[scenario] fanout_giant n={}", n);
    let mut futs = Vec::with_capacity(n);
    for i in 0..n {
        let child = launch(callee, &callee_payload("echo", &format!("g_{}", i)))?;
        futs.push(async move { (i, child.await) });
    }
    let results = future::join_all(futs).await;
    let mut fails = 0usize;
    for (i, res) in &results {
        let expected = format!("g_{}", i);
        match res {
            Ok(s) if *s == expected => {}
            _ => fails += 1,
        }
    }
    if fails > 0 {
        return Err(format!("giant fanout: {} of {} failed", fails, n));
    }
    Ok(format!("giant fanout: all {} returned correctly", n))
}

async fn scenario_long_payload(callee: &str, len: usize) -> Result<String> {
    println!("[scenario] long_payload len={}", len);
    let mut big = String::with_capacity(len);
    let pat = b"abcdefghijklmnop";
    while big.len() < len {
        big.push(pat[big.len() % pat.len()] as char);
    }
    big.truncate(len);
    let got = launch(callee, &callee_payload("echo", &big))?.await?;
    if got.len() != big.len() {
        return Err(format!("length mismatch: sent {} got {}", big.len(), got.len()));
    }
    if got != big {
        let div = big.as_bytes().iter().zip(got.as_bytes())
            .position(|(a, b)| a != b).unwrap_or(0);
        return Err(format!("payload corrupted at byte {}", div));
    }
    Ok(format!("long payload round-trip OK at len={}", len))
}

async fn scenario_deep_nesting(callee: &str, depth: usize) -> Result<String> {
    println!("[scenario] deep_nesting depth={}", depth);
    let payload = callee_payload_with(json!({
        "role": "callee",
        "directive": "descend",
        "payload": format!("{}|tag", depth),
    }));
    let got = launch(callee, &payload)?.await?;
    let mut expected = "base[tag]".to_string();
    for d in 1..=depth {
        expected = format!("d{}({})", d, expected);
    }
    if got != expected {
        return Err(format!("deep_nesting mismatch:\n  expected {:?}\n  got      {:?}", expected, got));
    }
    Ok(format!("deep_nesting depth={} returned correctly", depth))
}

async fn scenario_repeated_fanout(callee: &str, rounds: usize) -> Result<String> {
    println!("[scenario] repeated_fanout rounds={}", rounds);
    let per_round = 20usize;
    for round in 0..rounds {
        let mut futs = Vec::with_capacity(per_round);
        for i in 0..per_round {
            let child = launch(callee, &callee_payload("echo", &format!("r{}_{}", round, i)))?;
            futs.push(async move { (i, child.await) });
        }
        let results = future::join_all(futs).await;
        for (i, res) in &results {
            let expected = format!("r{}_{}", round, i);
            match res {
                Ok(s) if *s == expected => {}
                Ok(s) => return Err(format!("round {} i={}: {:?} != {:?}", round, i, s, expected)),
                Err(e) => return Err(format!("round {} i={} failed: {}", round, i, e)),
            }
        }
    }
    Ok(format!("repeated_fanout: {} rounds × {} = {} calls all clean", rounds, per_round, rounds * per_round))
}

async fn scenario_fanout_infer(callee: &str, n: usize) -> Result<String> {
    println!("[scenario] fanout_infer n={}", n);
    let prompts = [
        "What is 2+2?",
        "Name one color.",
        "Say yes.",
        "Print the letter A.",
        "What is 5+5?",
        "Name one fruit.",
        "Say hello.",
        "Print the digit 7.",
    ];
    let mut futs = Vec::with_capacity(n);
    for i in 0..n {
        let p = prompts[i % prompts.len()];
        let child = launch(callee, &callee_payload("generate", p))?;
        let prompt = p.to_string();
        futs.push(async move { (i, prompt, child.await) });
    }
    let results = future::join_all(futs).await;
    let mut summary = Vec::with_capacity(n);
    for (i, p, res) in results {
        match res {
            Ok(s) if !s.is_empty() => summary.push(format!("[{}] {} → {}", i, p, s)),
            Ok(s) => return Err(format!("empty generation at i={}: {:?}", i, s)),
            Err(e) => return Err(format!("infer call {} failed: {}", i, e)),
        }
    }
    println!("fanout_infer detail:");
    for line in &summary {
        println!("  {}", line);
    }
    Ok(format!("fanout_infer n={} all generated", n))
}

async fn scenario_nested_fanout(callee: &str, breadth: usize) -> Result<String> {
    println!("[scenario] nested_fanout breadth={}", breadth);
    let mut futs = Vec::with_capacity(breadth);
    for i in 0..breadth {
        let payload = callee_payload_with(json!({
            "role": "callee",
            "directive": "fanout_inner",
            "payload": format!("{}|child{}", breadth, i),
        }));
        let child = launch(callee, &payload)?;
        futs.push(async move { (i, child.await) });
    }
    let results = future::join_all(futs).await;
    for (i, res) in &results {
        let s = res.as_ref().map_err(|e| format!("child {} failed: {}", i, e))?;
        let prefix = format!("inner[child{}]:", i);
        if !s.starts_with(&prefix) {
            return Err(format!("child {} wrong: {:?}", i, s));
        }
        for g in 0..breadth {
            let needle = format!("child{}-gc{}", i, g);
            if !s.contains(&needle) {
                return Err(format!("child {} missing gc{} in {:?}", i, g, s));
            }
        }
    }
    Ok(format!("nested_fanout: {}×{} = {} processes all correct", breadth, breadth, breadth * breadth))
}

async fn scenario_concurrent_self_in_callee(callee: &str) -> Result<String> {
    println!("[scenario] concurrent_self_in_callee");
    let n = 8usize;
    let inner_breadth = 4usize;
    let mut futs = Vec::with_capacity(n);
    for i in 0..n {
        let payload = callee_payload_with(json!({
            "role": "callee",
            "directive": "fanout_inner",
            "payload": format!("{}|p{}", inner_breadth, i),
        }));
        let child = launch(callee, &payload)?;
        futs.push(async move { (i, child.await) });
    }
    let results = future::join_all(futs).await;
    for (i, res) in &results {
        let s = res.as_ref().map_err(|e| format!("child {} failed: {}", i, e))?;
        for g in 0..inner_breadth {
            let needle = format!("p{}-gc{}", i, g);
            if !s.contains(&needle) {
                return Err(format!("child {} missing {:?} in {:?}", i, needle, s));
            }
        }
    }
    Ok(format!(
        "concurrent_self_in_callee: {}×{} = {} processes (parallel callees, each with parallel grandchildren)",
        n, inner_breadth, n * (1 + inner_breadth)
    ))
}

// ── v2-only scenarios ──────────────────────────────────────────────────────

async fn scenario_pid_is_uuid(callee: &str) -> Result<String> {
    println!("[scenario] pid_is_uuid");
    let child = launch(callee, &callee_payload("echo", "x"))?;
    let pid = child.pid();
    let _ = child.await?;
    // UUIDs are 36 chars with 4 dashes at fixed positions.
    if pid.len() != 36 || pid.matches('-').count() != 4 {
        return Err(format!("pid does not look like a UUID: {:?}", pid));
    }
    Ok(format!("pid was {}", pid))
}

async fn scenario_drop_detach(callee: &str) -> Result<String> {
    println!("[scenario] drop_detach");
    // Launch a child that pushes to a side-channel topic and then echoes.
    // Drop the handle immediately; the child should still complete and the
    // side-channel push proves it.
    let notify_topic = format!("drop_detach_notify_{}", runtime::instance_id());
    let payload = callee_payload_with(json!({
        "role": "callee",
        "directive": "notify_then_echo",
        "payload": "detached!",
        "notify_topic": notify_topic.clone(),
    }));
    let child = launch(callee, &payload)?;
    let pid = child.pid();
    drop(child); // ← detach
    let received = inferlet::messaging::pull(&notify_topic);
    use inferlet::FutureStringExt;
    let msg = received.wait_async().await.ok_or("no notify received")?;
    if msg != "callee saw: detached!" {
        return Err(format!("unexpected notify: {:?}", msg));
    }
    Ok(format!("drop_detach: child pid={} ran to completion after parent dropped handle", pid))
}

async fn scenario_cancel_mid_flight(callee: &str) -> Result<String> {
    println!("[scenario] cancel_mid_flight");
    // Launch a sleep-then-echo callee, cancel before it finishes. Expect
    // wait() to return Err("cancelled").
    let payload = callee_payload_with(json!({
        "role": "callee",
        "directive": "sleep_then_echo",
        "payload": "should_not_arrive",
        "sleep_ms": 2_000,
    }));
    let mut child = launch(callee, &payload)?;
    let pid = child.pid();
    // Give it a moment to start, then cancel.
    task::sleep(inferlet::wstd::time::Duration::from_millis(50)).await;
    child.cancel();
    match child.wait().await {
        Ok(v) => Err(format!("expected cancellation, got Ok({:?})", v)),
        Err(e) if e.contains("cancelled") => {
            Ok(format!("cancel_mid_flight: pid={} got expected err: {}", pid, e))
        }
        Err(e) => Err(format!("cancel produced wrong error: {}", e)),
    }
}

async fn scenario_timeout_then_cancel(callee: &str) -> Result<String> {
    println!("[scenario] timeout_then_cancel");
    // Launch a slow callee; wait with a short timeout, cancel on expiry.
    let payload = callee_payload_with(json!({
        "role": "callee",
        "directive": "sleep_then_echo",
        "payload": "too_slow",
        "sleep_ms": 5_000,
    }));
    let mut child = launch(callee, &payload)?;
    let pid = child.pid();
    let timed = child.wait().timeout(inferlet::wstd::time::Duration::from_millis(200)).await;
    match timed {
        Ok(Ok(_)) | Ok(Err(_)) => Err("expected timeout, child finished too fast".into()),
        Err(_elapsed) => {
            // Timeout fired — cancel and confirm the child reports cancelled.
            child.cancel();
            match child.wait().await {
                Ok(v) => Err(format!("post-cancel still got Ok({:?})", v)),
                Err(e) if e.contains("cancelled") => {
                    Ok(format!("timeout_then_cancel: pid={} cancelled with: {}", pid, e))
                }
                Err(e) => Err(format!("post-cancel wrong error: {}", e)),
            }
        }
    }
}

async fn scenario_cancel_after_done(callee: &str) -> Result<String> {
    println!("[scenario] cancel_after_done");
    // Wait until the child is fully done via .wait(), then cancel() — must
    // be a silent no-op, not a panic or stale-pid kill.
    let mut child = launch(callee, &callee_payload("echo", "already_done"))?;
    let pid = child.pid();
    let result = child.wait().await?;
    if result != "already_done" {
        return Err(format!("unexpected result: {:?}", result));
    }
    child.cancel(); // idempotent no-op
    Ok(format!("cancel_after_done: pid={} child finished, cancel() would be a no-op", pid))
}

async fn scenario_fanout_with_cancel(callee: &str, n: usize) -> Result<String> {
    println!("[scenario] fanout_with_cancel n={}", n);
    // Launch n slow children, cancel all of them, expect every wait() to
    // return cancelled (or Ok if the race favored Ok, which is rare with
    // a 2s sleep).
    let mut children = Vec::with_capacity(n);
    for i in 0..n {
        let payload = callee_payload_with(json!({
            "role": "callee",
            "directive": "sleep_then_echo",
            "payload": format!("never_{}", i),
            "sleep_ms": 2_000,
        }));
        children.push(launch(callee, &payload)?);
    }
    // Give them a beat to start, then cancel.
    task::sleep(inferlet::wstd::time::Duration::from_millis(50)).await;
    for c in &children {
        c.cancel();
    }
    let mut cancelled = 0usize;
    let mut sneaked_ok = 0usize;
    for (i, mut c) in children.into_iter().enumerate() {
        match c.wait().await {
            Ok(_) => sneaked_ok += 1,
            Err(e) if e.contains("cancelled") => cancelled += 1,
            Err(e) => return Err(format!("child {} unexpected error: {}", i, e)),
        }
    }
    Ok(format!("fanout_with_cancel n={}: {} cancelled, {} sneaked Ok", n, cancelled, sneaked_ok))
}
