//! The shim end to end on this Mac's GPU, through the same `extern "C"`
//! functions the app calls: a plain turn, a turn cancelled mid-reply, the
//! turn after it, a turn cancelled before it started, a thinking turn, and a
//! barge-in: a reply cancelled partway and the next turn of the same
//! conversation, which must pick up the prefix the cancelled turn published.
//!
//! The engine is process-global (it boots once and stays up), so all of it
//! is one test, in order. It needs a Metal model artifact and a built
//! voice-chat inferlet:
//!
//! ```text
//! PIE_SHIM_TEST_MODEL=/path/to/model.metal.zt \
//! PIE_SHIM_TEST_WASM=/path/to/voice_chat.wasm \
//!     cargo test --release -- --nocapture
//! ```
//!
//! Without them it says so and passes.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use pie_ios_shim::{
    PIE_CANCELLED, PIE_CHUNK_REASONING, PIE_CHUNK_REPLY, pie_ios_cancel, pie_ios_free,
    pie_ios_run_stream,
};
use serde_json::{Value, json};

/// How soon after `pie_ios_cancel` a turn in flight must return.
const CANCEL_LATENCY: Duration = Duration::from_secs(1);

/// How soon the turn after a cancelled one must stream its first chunk:
/// the cancelled reply must not still be holding the engine.
const NEXT_FIRST_CHUNK: Duration = Duration::from_secs(2);

/// How many reply chunks the barge-in step lets through before it cancels,
/// so the partial reply the next turn carries is a real one.
const BARGE_AFTER_CHUNKS: usize = 16;

#[test]
fn turns_stream_cancel_and_think() {
    let (Some(model), Some(wasm)) = (env("PIE_SHIM_TEST_MODEL"), env("PIE_SHIM_TEST_WASM")) else {
        println!(
            "skipped: set PIE_SHIM_TEST_MODEL to a Metal .zt artifact and \
             PIE_SHIM_TEST_WASM to voice_chat.wasm to run the shim on this GPU"
        );
        return;
    };
    // A home of its own, so the test neither reads nor litters a real one.
    let home = std::env::temp_dir().join(format!("pie-shim-test-{}", std::process::id()));
    std::fs::create_dir_all(&home).expect("create the test PIE_HOME");
    // SAFETY: set before the engine boots, and nothing else in this test
    // binary is running yet to read the environment concurrently.
    unsafe { std::env::set_var("PIE_HOME", &home) };
    let config_path = home.join("config.toml");
    std::fs::write(&config_path, config_toml(&model)).expect("write the engine config");
    let config = c_string(config_path.to_str().expect("UTF-8 temp path"));
    let wasm = c_string(&wasm);

    // (a) A plain turn returns its JSON with the reply in it. The first
    // turn also boots the engine and installs the inferlet.
    let plain = run(
        &config,
        &wasm,
        1,
        &input("a", "What is the capital of France?", 60, None),
        None,
    );
    plain.report("(a) plain turn, cold engine");
    let value = plain.json();
    assert!(
        !str_field(&value, "text").is_empty(),
        "plain turn returned no text: {}",
        plain.result
    );
    assert_eq!(
        str_field(&value, "text"),
        plain.streamed(PIE_CHUNK_REPLY).trim()
    );

    // (b) A long reply, cancelled from another thread as soon as it speaks.
    let (spoke, first_words) = mpsc::channel();
    let long = std::thread::spawn({
        let (config, wasm) = (config.clone(), wasm.clone());
        move || {
            let turn = run(
                &config,
                &wasm,
                2,
                &input(
                    "b",
                    "Write a long story about a lighthouse keeper.",
                    400,
                    None,
                ),
                Some((1, spoke)),
            );
            (turn, Instant::now())
        }
    });
    first_words
        .recv_timeout(Duration::from_secs(60))
        .expect("the long turn never streamed reply text");
    let cancelled_at = Instant::now();
    pie_ios_cancel(2);
    let (cancelled, returned_at) = long.join().expect("the long turn's thread panicked");
    let latency = returned_at - cancelled_at;
    cancelled.report("(b) long turn, cancelled after its first reply chunk");
    println!("    cancel -> return: {} ms", latency.as_millis());
    assert_eq!(cancelled.result, PIE_CANCELLED);
    assert!(
        latency < CANCEL_LATENCY,
        "cancelled turn took {latency:?} to return"
    );
    assert!(cancelled.first(PIE_CHUNK_REPLY).is_some());

    // (c) The next turn starts right away: nothing of the cancelled reply
    // is still generating in front of it.
    let next = run(
        &config,
        &wasm,
        3,
        &input("c", "Say hello in five words.", 40, None),
        None,
    );
    next.report("(c) the turn after the cancel");
    let first = next
        .first(PIE_CHUNK_REPLY)
        .expect("the turn after the cancel streamed no reply");
    assert!(
        first < NEXT_FIRST_CHUNK,
        "the turn after the cancel took {first:?} to its first chunk"
    );
    assert!(!str_field(&next.json(), "text").is_empty());

    // (d) A turn cancelled before it was called never launches.
    pie_ios_cancel(4);
    let early = run(
        &config,
        &wasm,
        4,
        &input("d", "This should never run.", 40, None),
        None,
    );
    early.report("(d) cancelled before it started");
    assert_eq!(early.result, PIE_CANCELLED);
    assert!(early.chunks.is_empty(), "a pre-cancelled turn streamed");
    assert!(
        early.elapsed < Duration::from_millis(50),
        "a pre-cancelled turn took {:?}",
        early.elapsed
    );

    // A cancel for a turn that already returned changes nothing, here or
    // for the turns after it.
    pie_ios_cancel(3);

    // (e) Thinking: reasoning streams as kind 1, all of it before the
    // first reply chunk, and comes back in the JSON.
    let thought = run(
        &config,
        &wasm,
        5,
        &input(
            "e",
            "A bat and a ball cost 1.10 dollars in total. The bat costs 1 dollar \
             more than the ball. How much does the ball cost?",
            400,
            Some(120),
        ),
        None,
    );
    thought.report("(e) thinking turn, thinking_budget 120");
    let value = thought.json();
    println!(
        "    thought_tokens {}, generated {}",
        value["thought_tokens"], value["generated"]
    );
    let last_reasoning = thought
        .chunks
        .iter()
        .rposition(|chunk| chunk.kind == PIE_CHUNK_REASONING)
        .expect("the thinking turn streamed no reasoning");
    let first_reply = thought
        .chunks
        .iter()
        .position(|chunk| chunk.kind == PIE_CHUNK_REPLY)
        .expect("the thinking turn streamed no reply");
    assert!(
        last_reasoning < first_reply,
        "reasoning chunk {last_reasoning} came after reply chunk {first_reply}"
    );
    assert!(!str_field(&value, "reasoning").is_empty());
    assert!(!str_field(&value, "text").is_empty());
    assert_eq!(
        str_field(&value, "reasoning"),
        thought.streamed(PIE_CHUNK_REASONING).trim()
    );

    // (f) A barge-in as the app does it: a greedy reply cancelled partway,
    // then the next turn of the same conversation, whose transcript carries
    // the partial reply. It must adopt the prefix the cancelled turn
    // published and answer exactly as the same transcript is answered in a
    // conversation whose first reply ran to completion.
    let question = "Describe the city of Paris in a few sentences.";
    let (spoke, partway) = mpsc::channel();
    let barged = std::thread::spawn({
        let (config, wasm) = (config.clone(), wasm.clone());
        move || {
            run(
                &config,
                &wasm,
                6,
                &greedy("f-barged", &[("user", question)], 200),
                Some((BARGE_AFTER_CHUNKS, spoke)),
            )
        }
    });
    partway
        .recv_timeout(Duration::from_secs(60))
        .expect("the greedy turn ended before it was cancelled");
    pie_ios_cancel(6);
    let barged = barged.join().expect("the greedy turn's thread panicked");
    barged.report("(f) greedy turn, cancelled partway");
    assert_eq!(barged.result, PIE_CANCELLED);
    let partial = barged.streamed(PIE_CHUNK_REPLY);
    let partial = partial.trim();

    let completed = run(
        &config,
        &wasm,
        7,
        &greedy("f-completed", &[("user", question)], 200),
        None,
    );
    completed.report("(f) the same turn, run to completion in another conversation");
    assert!(!str_field(&completed.json(), "text").is_empty());

    let next_turn = [
        ("user", question),
        ("assistant", partial),
        ("user", "Which country is it the capital of?"),
    ];
    let after_barge = run(&config, &wasm, 8, &greedy("f-barged", &next_turn, 60), None);
    after_barge.report("(f) the turn after the barge-in");
    let after_completion = run(
        &config,
        &wasm,
        9,
        &greedy("f-completed", &next_turn, 60),
        None,
    );
    after_completion.report("(f) the same turn after the completed reply");
    let (after_barge, after_completion) = (after_barge.json(), after_completion.json());
    assert_eq!(
        after_barge["resumed"], true,
        "the turn after the barge-in adopted nothing the cancelled turn published"
    );
    assert_eq!(after_barge["reused"], after_completion["reused"]);
    assert_eq!(after_barge["new_prefill"], after_completion["new_prefill"]);
    assert!(!str_field(&after_barge, "text").is_empty());
    assert_eq!(
        str_field(&after_barge, "text"),
        str_field(&after_completion, "text")
    );

    let _ = std::fs::remove_dir_all(&home);
}

/// Everything one turn streamed and returned.
struct Turn {
    result: String,
    chunks: Vec<Chunk>,
    elapsed: Duration,
}

struct Chunk {
    kind: i32,
    text: String,
    /// Since the call.
    at: Duration,
}

impl Turn {
    fn first(&self, kind: i32) -> Option<Duration> {
        self.chunks
            .iter()
            .find(|chunk| chunk.kind == kind)
            .map(|chunk| chunk.at)
    }

    fn streamed(&self, kind: i32) -> String {
        self.chunks
            .iter()
            .filter(|chunk| chunk.kind == kind)
            .map(|chunk| chunk.text.as_str())
            .collect()
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.result)
            .unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.result))
    }

    fn report(&self, what: &str) {
        let ms =
            |d: Option<Duration>| d.map_or("-".to_string(), |d| format!("{} ms", d.as_millis()));
        println!(
            "{what}: returned in {} ms; first reasoning {}, first reply {}; {} chunks",
            self.elapsed.as_millis(),
            ms(self.first(PIE_CHUNK_REASONING)),
            ms(self.first(PIE_CHUNK_REPLY)),
            self.chunks.len()
        );
        println!("    reply: {:?}", self.streamed(PIE_CHUNK_REPLY).trim());
        println!("    result: {}", self.result);
    }
}

/// What the callback writes into, through the call's `ctx`.
struct Sink {
    started: Instant,
    chunks: Vec<Chunk>,
    replies: usize,
    /// Told once, when this many reply chunks have arrived.
    spoke: Option<(usize, mpsc::Sender<()>)>,
}

unsafe extern "C" fn collect(kind: i32, chunk: *const c_char, ctx: *mut c_void) {
    // SAFETY: `ctx` is the `Sink` that `run` passed for this call, which
    // only this callback touches until the call returns, and `chunk` is a
    // NUL-terminated string valid for the callback.
    let (sink, text) = unsafe { (&mut *ctx.cast::<Sink>(), CStr::from_ptr(chunk)) };
    if kind == PIE_CHUNK_REPLY {
        sink.replies += 1;
        if sink
            .spoke
            .as_ref()
            .is_some_and(|&(after, _)| sink.replies >= after)
            && let Some((_, spoke)) = sink.spoke.take()
        {
            let _ = spoke.send(());
        }
    }
    sink.chunks.push(Chunk {
        kind,
        text: text.to_string_lossy().into_owned(),
        at: sink.started.elapsed(),
    });
}

fn run(
    config: &CStr,
    wasm: &CStr,
    turn_id: u64,
    input: &str,
    spoke: Option<(usize, mpsc::Sender<()>)>,
) -> Turn {
    let input = c_string(input);
    let mut sink = Sink {
        started: Instant::now(),
        chunks: Vec::new(),
        replies: 0,
        spoke,
    };
    // SAFETY: every string is a live NUL-terminated `CString`, the version
    // may be null, `collect` has the callback's signature, and `sink`
    // outlives the call.
    let raw = unsafe {
        pie_ios_run_stream(
            config.as_ptr(),
            wasm.as_ptr(),
            std::ptr::null(),
            input.as_ptr(),
            turn_id,
            collect,
            (&raw mut sink).cast(),
        )
    };
    let elapsed = sink.started.elapsed();
    // SAFETY: `raw` is the non-null string `pie_ios_run_stream` returns,
    // released exactly once, after it is copied.
    let result = unsafe { CStr::from_ptr(raw) }
        .to_string_lossy()
        .into_owned();
    unsafe { pie_ios_free(raw) };
    Turn {
        result,
        chunks: sink.chunks,
        elapsed,
    }
}

/// The voice-chat input the app sends, one user message in its own
/// session at the app's temperature; `thinking_budget` turns thinking on.
fn input(session: &str, text: &str, max_tokens: u32, thinking_budget: Option<u32>) -> String {
    let mut input = request(session, &[("user", text)], max_tokens, 0.7);
    if let Some(budget) = thinking_budget {
        input["think"] = true.into();
        input["thinking_budget"] = budget.into();
    }
    input.to_string()
}

/// A greedy turn over `(role, content)` messages, so one transcript gets
/// one reply whichever conversation it runs in.
fn greedy(session: &str, messages: &[(&str, &str)], max_tokens: u32) -> String {
    request(session, messages, max_tokens, 0.0).to_string()
}

fn request(session: &str, messages: &[(&str, &str)], max_tokens: u32, temperature: f64) -> Value {
    let messages: Vec<Value> = messages
        .iter()
        .map(|&(role, content)| json!({"role": role, "content": content}))
        .collect();
    json!({
        "messages": messages,
        "session": format!("shim-test-{session}"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "think": false,
    })
}

/// The phone's config (ios/voice-app/Sources/PieKit/PieRuntimeConfig.swift),
/// pointed at `artifact`.
fn config_toml(artifact: &str) -> String {
    format!(
        r#"[server]
host = "127.0.0.1"
port = 0

[model]
name = "default"
model = "{artifact}"

[engine]
type = "metal"
device = ["metal:0"]
activation_dtype = "bfloat16"
gpu_mem_utilization = 0.85
kv_page_size = 32
total_pages = 256
max_forward_tokens = 2048
max_forward_requests = 4
max_model_len = 8192
max_state_slots = 8

[runtime]
max_concurrent_processes = 4

[sandbox]
allow_fs = false
allow_network = false
max_instances = 4
max_memory = "128MiB"
warm_memory = "0B"
warm_slots = 1
"#
    )
}

fn env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| Path::new(value).is_file())
}

fn c_string(s: &str) -> CString {
    CString::new(s).expect("no NUL in test strings")
}

fn str_field<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key].as_str().unwrap_or_default()
}
