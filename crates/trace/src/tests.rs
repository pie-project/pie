use serde_json::Value;

use super::*;

const A: AgentKey = 0xA;
const B: AgentKey = 0xB;

fn part(agent: AgentKey, prefill: u32, decode: u32, pages: u32) -> Part {
    Part {
        agent,
        prefill,
        decode,
        cached: 0,
        pages,
        queued_since: None,
    }
}

fn meta() -> Meta {
    Meta {
        pie_version: Some("0.5.4".into()),
        commit: Some("905e7b531".into()),
        started_at: Some("2026-10-18T12:00:00Z".into()),
        model: Some("qwen3.5-0.8b".into()),
        scheme: Some("affine_u4_g64".into()),
        device: Some("metal".into()),
        hardware: Hardware {
            name: Some("Apple M5 Max".into()),
            memory_gb: Some(48.0),
            bandwidth_gbs: Some(614.0),
        },
        kv_page_bytes: Some(262_144),
        kv_pages_total: Some(1024),
        weights_bytes: Some(480_000_000),
    }
}

/// Two agents: A prefills alone while B waits, both decode together, A calls
/// a tool, B finishes; device spans for every step.
fn two_agents() -> Vec<Record> {
    let mut log = vec![
        Record::AgentStart {
            t: 0,
            agent: A,
            program: "agent-loop".into(),
        },
        Record::AgentStart {
            t: 100,
            agent: B,
            program: "agent-loop".into(),
        },
    ];
    // Step 1: A's 512-row prefill; B submitted at 200 but waits.
    log.push(Record::Step {
        id: 1,
        t: 150,
        parts: vec![Part {
            cached: 0,
            ..part(A, 512, 0, 16)
        }],
    });
    log.push(Record::Gpu {
        step: 1,
        start: 200,
        end: 8_200,
    });
    log.push(Record::StepRetired { id: 1, t: 8_300 });
    log.push(Record::Step {
        id: 2,
        t: 8_300,
        parts: vec![Part {
            queued_since: Some(200),
            ..part(B, 300, 0, 10)
        }],
    });
    log.push(Record::Gpu {
        step: 2,
        start: 8_350,
        end: 13_350,
    });
    log.push(Record::StepRetired { id: 2, t: 13_400 });
    // Steps 3..=12: both decode, 1 ms each, back to back.
    let mut t = 13_400;
    for id in 3..=12 {
        log.push(Record::Step {
            id,
            t,
            parts: vec![part(A, 0, 1, 17), part(B, 0, 2, 11)],
        });
        log.push(Record::Gpu {
            step: id,
            start: t + 20,
            end: t + 1_000,
        });
        log.push(Record::StepRetired { id, t: t + 1_050 });
        t += 1_050;
    }
    // A waits 50 ms on a tool and feeds it back.
    log.push(Record::Tool {
        t: t + 50_000,
        agent: A,
        name: "search".into(),
    });
    log.push(Record::Memory {
        t: 10,
        bytes: 6 << 30,
    });
    log.push(Record::Memory {
        t: 20_000,
        bytes: (6 << 30) + (1 << 28),
    });
    log.push(Record::Evict {
        t: t + 1,
        agent: B,
        pages: 4,
    });
    log.push(Record::AgentEnd { t: t + 2, agent: B });
    log.push(Record::AgentEnd {
        t: t + 60_000,
        agent: A,
    });
    log
}

fn events(json: &Value) -> &Vec<Value> {
    json["traceEvents"].as_array().unwrap()
}

fn spans(json: &Value, pid: u64, tid: u64) -> Vec<&Value> {
    events(json)
        .iter()
        .filter(|e| e["ph"] == "X" && e["pid"] == pid && e["tid"] == tid)
        .collect()
}

fn counters<'a>(json: &'a Value, name: &str) -> Vec<&'a Value> {
    events(json)
        .iter()
        .filter(|e| e["ph"] == "C" && e["name"] == name)
        .collect()
}

#[test]
fn the_file_has_the_header_and_meta_the_format_names() {
    let json = render(&meta(), &two_agents()).json;
    assert_eq!(json["format"], "pie-trace");
    assert_eq!(json["version"], 1);
    for key in [
        "pie_version",
        "commit",
        "started_at",
        "model",
        "scheme",
        "device",
        "hardware",
        "kv_page_bytes",
        "kv_pages_total",
        "weights_bytes",
    ] {
        assert!(json["meta"].get(key).is_some(), "meta lacks {key}");
    }
    for key in ["name", "memory_gb", "bandwidth_gbs"] {
        assert!(json["meta"]["hardware"].get(key).is_some());
    }
    let keys: Vec<&String> = json.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["format", "version", "meta", "traceEvents"]);
}

#[test]
fn every_event_has_the_shape_its_phase_needs() {
    let json = render(&meta(), &two_agents()).json;
    let mut last_ts = 0;
    for event in events(&json) {
        let ph = event["ph"].as_str().unwrap();
        assert!(["M", "X", "C", "i"].contains(&ph), "phase {ph}");
        assert!(event["name"].is_string());
        assert!(event["pid"].is_u64());
        match ph {
            "M" => assert!(event["args"]["name"].is_string()),
            "X" => {
                assert!(event["tid"].is_u64());
                assert!(event["dur"].is_u64());
                let ts = event["ts"].as_u64().unwrap();
                assert!(ts >= last_ts, "events are in time order");
                last_ts = ts;
            }
            "C" => {
                assert_eq!(event["pid"], 2);
                assert!(
                    event["args"]
                        .as_object()
                        .unwrap()
                        .values()
                        .all(Value::is_number)
                );
            }
            "i" => assert_eq!(event["s"], "t"),
            _ => unreachable!(),
        }
    }
}

#[test]
fn tracks_are_named() {
    let json = render(&meta(), &two_agents()).json;
    let names: Vec<(u64, Option<u64>, String)> = events(&json)
        .iter()
        .filter(|e| e["ph"] == "M")
        .map(|e| {
            (
                e["pid"].as_u64().unwrap(),
                e["tid"].as_u64(),
                e["args"]["name"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert!(names.contains(&(1, None, "Agents".into())));
    assert!(names.contains(&(2, None, "Engine".into())));
    assert!(names.contains(&(2, Some(1), "Scheduler".into())));
    assert!(names.contains(&(2, Some(2), "GPU".into())));
    assert!(names.contains(&(1, Some(1), "agent-1 (agent-loop)".into())));
    assert!(names.contains(&(1, Some(2), "agent-2 (agent-loop)".into())));
}

#[test]
fn agent_tracks_show_prefill_queued_decode_and_tool() {
    let json = render(&meta(), &two_agents()).json;
    let a = spans(&json, 1, 1);
    let kinds: Vec<&str> = a.iter().map(|e| e["name"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["prefill", "decode", "tool"]);
    assert_eq!(a[0]["args"]["tokens"], 512);
    assert_eq!(a[0]["args"]["cached_tokens"], 0);
    // Ten back-to-back decode steps are one span.
    assert_eq!(a[1]["args"]["tokens"], 10);
    assert_eq!(a[2]["args"]["tool"], "search");
    assert!(a[2]["dur"].as_u64().unwrap() >= 50_000);

    let b = spans(&json, 1, 2);
    let kinds: Vec<&str> = b.iter().map(|e| e["name"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["queued", "prefill", "decode"]);
    assert_eq!(b[0]["args"], serde_json::json!({}));
    assert_eq!(b[2]["args"]["tokens"], 20);

    for track in [a, b] {
        for pair in track.windows(2) {
            let end = pair[0]["ts"].as_u64().unwrap() + pair[0]["dur"].as_u64().unwrap();
            assert!(
                pair[1]["ts"].as_u64().unwrap() >= end,
                "an agent's spans never overlap"
            );
        }
    }
}

#[test]
fn steps_carry_their_rows_by_agent_and_device_time() {
    let json = render(&meta(), &two_agents()).json;
    let steps = spans(&json, 2, 1);
    assert_eq!(steps.len(), 12);
    let third = &steps[2]["args"];
    assert_eq!(third["step"], 3);
    assert_eq!(third["prefill_tokens"], 0);
    assert_eq!(third["decode_tokens"], 3);
    assert_eq!(
        third["agents"]["1"],
        serde_json::json!({"prefill": 0, "decode": 1})
    );
    assert_eq!(
        third["agents"]["2"],
        serde_json::json!({"prefill": 0, "decode": 2})
    );
    assert_eq!(third["gpu_us"], 980);
    let gpu = spans(&json, 2, 2);
    assert_eq!(gpu.len(), 12);
    assert_eq!(gpu[0]["name"], "gpu");
    assert_eq!(gpu[0]["args"]["step"], 1);
    assert_eq!(gpu[0]["dur"], 8_000);
    for pair in steps.windows(2) {
        let end = pair[0]["ts"].as_u64().unwrap() + pair[0]["dur"].as_u64().unwrap();
        assert!(
            pair[1]["ts"].as_u64().unwrap() >= end,
            "steps never overlap"
        );
    }
}

#[test]
fn per_agent_gpu_share_sums_to_gpu_busy() {
    let json = render(&meta(), &two_agents()).json;
    let busy = counters(&json, "gpu_busy");
    let share = counters(&json, "gpu_share");
    assert!(!busy.is_empty());
    assert_eq!(busy.len(), share.len());
    let mut any_busy = false;
    for (b, s) in busy.iter().zip(&share) {
        assert_eq!(b["ts"], s["ts"]);
        let pct = b["args"]["pct"].as_f64().unwrap();
        assert!((0.0..=100.0).contains(&pct));
        let sum: f64 = s["args"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_f64().unwrap())
            .sum();
        assert!((sum - pct).abs() < 0.01, "share {sum} vs busy {pct}");
        any_busy |= pct > 0.0;
    }
    assert!(any_busy);
    // The first window: A's prefill 200..8200, then B's from 8350.
    assert_eq!(busy[0]["args"]["pct"], 96.5);
    assert_eq!(share[0]["args"]["agent-1"], 80.0);
    assert_eq!(share[0]["args"]["agent-2"], 16.5);
    // The trace ends idle.
    assert_eq!(busy.last().unwrap()["args"]["pct"], 0.0);
    // Decode windows split 1:2 by rows.
    let both = share
        .iter()
        .find(|s| s["ts"] == 20_000)
        .expect("a window of shared decode");
    let a = both["args"]["agent-1"].as_f64().unwrap();
    let b = both["args"]["agent-2"].as_f64().unwrap();
    assert!((b - 2.0 * a).abs() < 0.01);
}

#[test]
fn kv_pages_track_each_agent_and_the_free_rest() {
    let json = render(&meta(), &two_agents()).json;
    let kv = counters(&json, "kv_pages");
    assert!(!kv.is_empty());
    for c in &kv {
        let args = c["args"].as_object().unwrap();
        let used: u64 = args
            .iter()
            .filter(|(k, _)| k.starts_with("agent-"))
            .map(|(_, v)| v.as_u64().unwrap())
            .sum();
        assert_eq!(args["free"].as_u64().unwrap(), 1024 - used);
    }
    assert_eq!(kv[0]["args"]["agent-1"], 16);
    // Eviction takes 4 of B's 11, then its end releases the rest.
    assert!(kv.iter().any(|c| c["args"]["agent-2"] == 7));
    let last = kv.last().unwrap();
    assert_eq!(last["args"]["agent-1"], 0);
    assert_eq!(last["args"]["agent-2"], 0);
    assert_eq!(last["args"]["free"], 1024);
}

#[test]
fn memory_starts_ends_and_evictions_are_emitted() {
    let json = render(&meta(), &two_agents()).json;
    let memory = counters(&json, "memory");
    assert_eq!(memory.len(), 2);
    assert_eq!(memory[0]["args"]["allocated_gb"], 6.0);
    assert_eq!(memory[1]["args"]["allocated_gb"], 6.25);
    let instants: Vec<(&str, u64)> = events(&json)
        .iter()
        .filter(|e| e["ph"] == "i")
        .map(|e| (e["name"].as_str().unwrap(), e["pid"].as_u64().unwrap()))
        .collect();
    assert_eq!(instants.iter().filter(|(n, _)| *n == "start").count(), 2);
    assert_eq!(instants.iter().filter(|(n, _)| *n == "end").count(), 2);
    let evict = events(&json).iter().find(|e| e["name"] == "evict").unwrap();
    assert_eq!(evict["pid"], 2);
    assert_eq!(evict["tid"], 1);
    assert_eq!(evict["args"], serde_json::json!({"agent": 2, "pages": 4}));
}

fn keys(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                out.push(k.clone());
                keys(v, out);
            }
        }
        Value::Array(items) => items.iter().for_each(|v| keys(v, out)),
        _ => {}
    }
}

#[test]
fn no_field_can_carry_text() {
    let json = render(&meta(), &two_agents()).json;
    let mut all = Vec::new();
    keys(&json, &mut all);
    for banned in [
        "text",
        "prompt",
        "output",
        "content",
        "arguments",
        "arguments_json",
        "completion",
        "message",
        "input",
    ] {
        assert!(!all.iter().any(|k| k == banned), "a `{banned}` field");
    }
    // Every string value is a name the format allows.
    fn strings<'a>(value: &'a Value, out: &mut Vec<&'a str>) {
        match value {
            Value::String(s) => out.push(s),
            Value::Object(map) => map.values().for_each(|v| strings(v, out)),
            Value::Array(items) => items.iter().for_each(|v| strings(v, out)),
            _ => {}
        }
    }
    let mut values = Vec::new();
    strings(&json["traceEvents"], &mut values);
    for value in values {
        assert!(
            value.len() <= 64,
            "a long string in the events is not a name: {value:?}"
        );
    }
}

#[test]
fn a_program_version_is_not_part_of_the_agent_label() {
    let log = vec![Record::AgentStart {
        t: 0,
        agent: A,
        program: "react".into(),
    }];
    let json = render(&Meta::default(), &log).json;
    assert!(
        events(&json)
            .iter()
            .any(|e| e["args"]["name"] == "agent-1 (react)")
    );
    assert!(json["meta"]["model"].is_null());
}

#[test]
fn the_recorder_is_inert_until_started_then_writes_the_file() {
    // One test owns the global recorder.
    assert!(!enabled());
    assert_eq!(step_posted(vec![part(A, 1, 0, 1)]), 0);
    agent_start(A, "x@1.0.0");
    assert!(finish().unwrap().is_none());

    let dir = std::env::temp_dir().join(format!("pie-trace-test-{}", std::process::id()));
    let path = dir.join("t.pie");
    start(&path, meta());
    assert!(enabled());
    agent_start(A, "agent-loop@0.1.0");
    let s1 = step_posted(vec![part(A, 40, 0, 2)]);
    let s2 = step_posted(vec![part(A, 0, 1, 2)]);
    assert_eq!((s1, s2), (1, 2));
    frame_submitted(7, &[s1, s2]);
    // Keyed out of order, then unkeyed to the oldest without a span.
    gpu_span(Some((7, 1)), 3_000, 3_500);
    gpu_span(None, 1_000, 2_000);
    gpu_span(None, 5_000, 6_000);
    // A frame's span can land before its submit returns.
    let s3 = step_posted(vec![part(A, 0, 1, 2)]);
    gpu_span(Some((8, 0)), 7_000, 7_400);
    frame_submitted(8, &[s3]);
    step_retired(&[s1, s2]);
    memory(1 << 30);
    memory(2 << 30);
    tool(A, "fake_search");
    agent_end(A);
    let written = finish().unwrap().unwrap();
    assert!(!enabled());
    assert_eq!(written.agents, 1);
    assert_eq!(written.steps, 3);

    let json: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let gpu = spans(&json, 2, 2);
    assert_eq!(gpu.len(), 3);
    assert_eq!(gpu[2]["args"]["step"], 3);
    assert_eq!(gpu[0]["args"]["step"], 1);
    assert_eq!(gpu[0]["ts"], 1_000);
    assert_eq!(gpu[1]["args"]["step"], 2);
    assert_eq!(gpu[1]["ts"], 3_000);
    assert_eq!(counters(&json, "memory").len(), 1, "one sample per window");
    assert!(
        events(&json)
            .iter()
            .any(|e| e["args"]["name"] == "agent-1 (agent-loop)")
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn schemes_from_deployment_names() {
    assert_eq!(
        scheme_of("qwen35-d0-8b-u4g64-kv-bf16").as_deref(),
        Some("affine_u4_g64")
    );
    assert_eq!(scheme_of("qwen35-d0-8b-bf16-kv-bf16").as_deref(), Some("bf16"));
    assert_eq!(scheme_of("plain"), None);
}

#[test]
fn rfc3339_and_bandwidth() {
    let at = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_792_324_800);
    assert_eq!(rfc3339_utc(at), "2026-10-18T12:00:00Z");
    assert_eq!(rfc3339_utc(SystemTime::UNIX_EPOCH), "1970-01-01T00:00:00Z");
    assert_eq!(apple_bandwidth_gbs("Apple M5 Max", Some(40)), Some(614.0));
    assert_eq!(apple_bandwidth_gbs("Apple M4 Max", Some(32)), Some(410.0));
    assert_eq!(apple_bandwidth_gbs("Apple M1", None), Some(68.25));
    assert_eq!(apple_bandwidth_gbs("NVIDIA H100", None), None);
}
