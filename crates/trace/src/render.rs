//! The log, as the `.pie` file `TRACE_FORMAT.md` specifies.

use std::collections::{BTreeMap, HashMap};

use serde_json::{Map, Value, json};

use crate::{AgentKey, Meta, Part, Record, WINDOW_US};

/// Two of an agent's decode (or prefill) steps closer than this, with
/// nothing of its own between them, are one span.
pub const MERGE_GAP_US: u64 = 2_000;

/// A wait for a step shorter than this is not drawn as `queued`: a
/// run-ahead request always waits a little for the step before it.
pub const QUEUED_MIN_US: u64 = 1_000;

const AGENTS_PID: u64 = 1;
const ENGINE_PID: u64 = 2;
const SCHEDULER_TID: u64 = 1;
const GPU_TID: u64 = 2;

const GIB: f64 = (1u64 << 30) as f64;

/// A rendered trace.
#[derive(Debug, Clone)]
pub struct Rendered {
    pub json: Value,
    pub agents: usize,
    pub steps: usize,
    pub events: usize,
}

struct Agent {
    tid: u64,
    program: Option<String>,
    start: Option<u64>,
    end: Option<u64>,
}

struct Step<'a> {
    id: u64,
    posted: u64,
    parts: &'a [Part],
    retired: Option<u64>,
    gpu: Vec<(u64, u64)>,
    start: u64,
    end: u64,
}

impl Step<'_> {
    fn gpu_us(&self) -> u64 {
        self.gpu.iter().map(|(s, e)| e - s).sum()
    }

    fn tokens(&self) -> u64 {
        self.parts.iter().map(|p| u64::from(p.tokens())).sum()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Prefill,
    Decode,
    Queued,
    Tool,
}

struct Span {
    kind: Kind,
    start: u64,
    end: u64,
    tokens: u64,
    cached: u64,
    tool: Option<String>,
}

fn agent_name(tid: u64) -> String {
    format!("agent-{tid}")
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn x_event(name: &str, pid: u64, tid: u64, start: u64, end: u64, args: Value) -> Value {
    json!({
        "ph": "X", "name": name, "pid": pid, "tid": tid,
        "ts": start, "dur": end.saturating_sub(start), "args": args,
    })
}

fn counter(name: &str, ts: u64, args: Map<String, Value>) -> Value {
    json!({ "ph": "C", "name": name, "pid": ENGINE_PID, "ts": ts, "args": args })
}

fn instant(name: &str, pid: u64, tid: u64, ts: u64, args: Option<Value>) -> Value {
    let mut event = json!({ "ph": "i", "s": "t", "name": name, "pid": pid, "tid": tid, "ts": ts });
    if let Some(args) = args {
        event["args"] = args;
    }
    event
}

fn meta_json(meta: &Meta) -> Value {
    json!({
        "pie_version": meta.pie_version,
        "commit": meta.commit,
        "started_at": meta.started_at,
        "model": meta.model,
        "scheme": meta.scheme,
        "device": meta.device,
        "hardware": {
            "name": meta.hardware.name,
            "memory_gb": meta.hardware.memory_gb,
            "bandwidth_gbs": meta.hardware.bandwidth_gbs,
        },
        "kv_page_bytes": meta.kv_page_bytes,
        "kv_pages_total": meta.kv_pages_total,
        "weights_bytes": meta.weights_bytes,
    })
}

/// Renders `log` with `meta` into the file's JSON object.
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "one pass per track, in the order the format lists them"
)]
pub fn render(meta: &Meta, log: &[Record]) -> Rendered {
    // ── agents, in the order they first appear ──────────────────────────
    let mut agents: HashMap<AgentKey, Agent> = HashMap::new();
    let mut order: Vec<AgentKey> = Vec::new();
    let mut agent = |key: AgentKey| -> u64 {
        let next = agents.len() as u64 + 1;
        agents
            .entry(key)
            .or_insert_with(|| {
                order.push(key);
                Agent {
                    tid: next,
                    program: None,
                    start: None,
                    end: None,
                }
            })
            .tid
    };
    for record in log {
        match record {
            Record::AgentStart { agent: key, .. }
            | Record::AgentEnd { agent: key, .. }
            | Record::Tool { agent: key, .. }
            | Record::Evict { agent: key, .. } => {
                agent(*key);
            }
            Record::Step { parts, .. } => {
                for part in parts {
                    agent(part.agent);
                }
            }
            _ => {}
        }
    }
    for record in log {
        match record {
            Record::AgentStart { t, agent, program } => {
                let a = agents.get_mut(agent).expect("seen above");
                a.start.get_or_insert(*t);
                a.program.get_or_insert_with(|| program.clone());
            }
            Record::AgentEnd { t, agent } => {
                agents.get_mut(agent).expect("seen above").end = Some(*t);
            }
            _ => {}
        }
    }
    let tid_of = |key: &AgentKey| agents[key].tid;

    // ── steps and their device spans ────────────────────────────────────
    let mut steps: BTreeMap<u64, Step<'_>> = BTreeMap::new();
    for record in log {
        if let Record::Step { id, t, parts } = record {
            steps.insert(
                *id,
                Step {
                    id: *id,
                    posted: *t,
                    parts,
                    retired: None,
                    gpu: Vec::new(),
                    start: 0,
                    end: 0,
                },
            );
        }
    }
    for record in log {
        if let Record::StepRetired { id, t } = record
            && let Some(step) = steps.get_mut(id)
        {
            step.retired.get_or_insert(*t);
        }
    }
    // One device runs one submission at a time: clip what the clocks'
    // jitter overlaps.
    let mut gpu: Vec<(u64, u64, u64)> = log
        .iter()
        .filter_map(|r| match r {
            Record::Gpu { step, start, end } if steps.contains_key(step) => {
                Some((*start, *end, *step))
            }
            _ => None,
        })
        .collect();
    gpu.sort_unstable();
    let mut gpu_end = 0u64;
    for (start, end, step) in &mut gpu {
        *start = (*start).max(gpu_end);
        *end = (*end).max(*start);
        gpu_end = *end;
        steps
            .get_mut(step)
            .expect("filtered")
            .gpu
            .push((*start, *end));
    }
    let mut prev_end = 0u64;
    for step in steps.values_mut() {
        let start = step.posted.max(prev_end);
        let done = step
            .gpu
            .iter()
            .map(|&(_, e)| e)
            .max()
            .or(step.retired)
            .unwrap_or(step.posted);
        step.start = start;
        step.end = done.max(start);
        prev_end = step.end;
    }

    let mut events: Vec<Value> = Vec::new();

    // ── agent spans ─────────────────────────────────────────────────────
    enum Item<'a> {
        Work(&'a Step<'a>, &'a Part),
        Tool(u64, &'a str),
    }
    let mut items: HashMap<AgentKey, Vec<(u64, Item<'_>)>> = HashMap::new();
    for step in steps.values() {
        for part in step.parts.iter().filter(|p| p.tokens() > 0) {
            items
                .entry(part.agent)
                .or_default()
                .push((step.start, Item::Work(step, part)));
        }
    }
    for record in log {
        if let Record::Tool { t, agent, name } = record {
            items
                .entry(*agent)
                .or_default()
                .push((*t, Item::Tool(*t, name)));
        }
    }
    for key in &order {
        let a = &agents[key];
        let mut list = items.remove(key).unwrap_or_default();
        list.sort_by_key(|(t, _)| *t);
        let mut spans: Vec<Span> = Vec::new();
        let mut cursor = a.start.unwrap_or(0);
        let mut mergeable = false;
        for (_, item) in list {
            match item {
                Item::Tool(t, name) => {
                    if t > cursor {
                        spans.push(Span {
                            kind: Kind::Tool,
                            start: cursor,
                            end: t,
                            tokens: 0,
                            cached: 0,
                            tool: Some(name.to_string()),
                        });
                        cursor = t;
                    }
                    mergeable = false;
                }
                Item::Work(step, part) => {
                    if let Some(since) = part.queued_since {
                        let q = since.max(cursor);
                        if step.start >= q + QUEUED_MIN_US {
                            spans.push(Span {
                                kind: Kind::Queued,
                                start: q,
                                end: step.start,
                                tokens: 0,
                                cached: 0,
                                tool: None,
                            });
                            cursor = step.start;
                            mergeable = false;
                        }
                    }
                    let kind = if part.prefill > 0 {
                        Kind::Prefill
                    } else {
                        Kind::Decode
                    };
                    let tokens = u64::from(if part.prefill > 0 {
                        part.prefill
                    } else {
                        part.decode
                    });
                    let start = step.start.max(cursor);
                    let end = step.end.max(start);
                    match spans.last_mut() {
                        Some(last)
                            if mergeable
                                && last.kind == kind
                                && start.saturating_sub(last.end) <= MERGE_GAP_US =>
                        {
                            last.end = end;
                            last.tokens += tokens;
                        }
                        _ => spans.push(Span {
                            kind,
                            start,
                            end,
                            tokens,
                            cached: u64::from(part.cached),
                            tool: None,
                        }),
                    }
                    mergeable = true;
                    cursor = end;
                }
            }
        }
        for span in spans {
            let (name, args) = match span.kind {
                Kind::Prefill => (
                    "prefill",
                    json!({ "tokens": span.tokens, "cached_tokens": span.cached }),
                ),
                Kind::Decode => ("decode", json!({ "tokens": span.tokens })),
                Kind::Queued => ("queued", json!({})),
                Kind::Tool => ("tool", json!({ "tool": span.tool })),
            };
            events.push(x_event(name, AGENTS_PID, a.tid, span.start, span.end, args));
        }
        if let Some(t) = a.start {
            events.push(instant("start", AGENTS_PID, a.tid, t, None));
        }
        if let Some(t) = a.end {
            events.push(instant("end", AGENTS_PID, a.tid, t, None));
        }
    }

    // ── scheduler steps and device submissions ──────────────────────────
    for step in steps.values() {
        let mut per_agent: BTreeMap<u64, (u64, u64)> = BTreeMap::new();
        for part in step.parts {
            let entry = per_agent.entry(tid_of(&part.agent)).or_default();
            entry.0 += u64::from(part.prefill);
            entry.1 += u64::from(part.decode);
        }
        let mut agents_arg = Map::new();
        for (tid, (prefill, decode)) in &per_agent {
            agents_arg.insert(
                tid.to_string(),
                json!({ "prefill": prefill, "decode": decode }),
            );
        }
        let prefill: u64 = per_agent.values().map(|v| v.0).sum();
        let decode: u64 = per_agent.values().map(|v| v.1).sum();
        events.push(x_event(
            "step",
            ENGINE_PID,
            SCHEDULER_TID,
            step.start,
            step.end,
            json!({
                "step": step.id,
                "prefill_tokens": prefill,
                "decode_tokens": decode,
                "agents": agents_arg,
                "gpu_us": step.gpu_us(),
            }),
        ));
        for &(start, end) in &step.gpu {
            events.push(x_event(
                "gpu",
                ENGINE_PID,
                GPU_TID,
                start,
                end,
                json!({ "step": step.id }),
            ));
        }
    }

    // ── kv_pages ────────────────────────────────────────────────────────
    enum PageChange<'a> {
        Step(&'a [Part]),
        Release(AgentKey),
        Evict(AgentKey, u32),
    }
    let mut changes: Vec<(u64, PageChange<'_>)> = Vec::new();
    for step in steps.values() {
        changes.push((step.start, PageChange::Step(step.parts)));
    }
    for record in log {
        match record {
            Record::AgentEnd { t, agent } => changes.push((*t, PageChange::Release(*agent))),
            Record::Evict { t, agent, pages } => {
                changes.push((*t, PageChange::Evict(*agent, *pages)));
            }
            _ => {}
        }
    }
    changes.sort_by_key(|(t, _)| *t);
    let mut held: BTreeMap<u64, u64> = BTreeMap::new();
    let total = meta.kv_pages_total;
    for (t, change) in changes {
        let before = held.clone();
        match change {
            PageChange::Step(parts) => {
                let mut pages: BTreeMap<u64, u64> = BTreeMap::new();
                for part in parts {
                    let entry = pages.entry(tid_of(&part.agent)).or_default();
                    *entry = (*entry).max(u64::from(part.pages));
                }
                for (tid, n) in pages {
                    held.insert(tid, n);
                }
            }
            PageChange::Release(key) => {
                held.insert(tid_of(&key), 0);
            }
            PageChange::Evict(key, pages) => {
                let entry = held.entry(tid_of(&key)).or_default();
                *entry = entry.saturating_sub(u64::from(pages));
            }
        }
        if held == before {
            continue;
        }
        let mut args = Map::new();
        for (tid, n) in &held {
            args.insert(agent_name(*tid), json!(n));
        }
        if let Some(total) = total {
            let used: u64 = held.values().sum();
            args.insert("free".into(), json!(total.saturating_sub(used)));
        }
        events.push(counter("kv_pages", t, args));
    }

    // ── gpu_busy and gpu_share over windows ─────────────────────────────
    let last = steps
        .values()
        .flat_map(|s| s.gpu.iter().map(|&(_, e)| e))
        .max();
    if let Some(last) = last {
        let windows = usize::try_from(last / WINDOW_US + 1).unwrap_or(usize::MAX);
        let tids: Vec<u64> = order.iter().map(&tid_of).collect();
        let mut busy = vec![0f64; windows];
        let mut share = vec![vec![0f64; tids.len()]; windows];
        for step in steps.values() {
            let tokens = step.tokens();
            if tokens == 0 {
                continue;
            }
            let mut weights = vec![0f64; tids.len()];
            for part in step.parts {
                let index = (tid_of(&part.agent) - 1) as usize;
                weights[index] += f64::from(part.tokens()) / tokens as f64;
            }
            for &(start, end) in &step.gpu {
                let mut w = (start / WINDOW_US) as usize;
                while w < windows && (w as u64 * WINDOW_US) < end {
                    let lo = (w as u64 * WINDOW_US).max(start);
                    let hi = ((w as u64 + 1) * WINDOW_US).min(end);
                    let overlap = hi.saturating_sub(lo) as f64;
                    busy[w] += overlap;
                    for (slot, weight) in share[w].iter_mut().zip(&weights) {
                        *slot += overlap * weight;
                    }
                    w += 1;
                }
            }
        }
        let pct = |us: f64| round3(us / WINDOW_US as f64 * 100.0);
        let mut last_emitted: Option<(f64, Vec<f64>)> = None;
        for w in 0..=windows {
            let (b, s) = if w < windows {
                (pct(busy[w]), share[w].iter().map(|&x| pct(x)).collect())
            } else {
                (0.0, vec![0.0; tids.len()])
            };
            if last_emitted
                .as_ref()
                .is_some_and(|(lb, ls)| *lb == b && *ls == s)
            {
                continue;
            }
            let ts = w as u64 * WINDOW_US;
            let mut busy_args = Map::new();
            busy_args.insert("pct".into(), json!(b));
            events.push(counter("gpu_busy", ts, busy_args));
            let mut share_args = Map::new();
            for (tid, value) in tids.iter().zip(&s) {
                share_args.insert(agent_name(*tid), json!(value));
            }
            events.push(counter("gpu_share", ts, share_args));
            last_emitted = Some((b, s));
        }
    }

    // ── memory ──────────────────────────────────────────────────────────
    let mut samples: Vec<(u64, u64)> = log
        .iter()
        .filter_map(|r| match r {
            Record::Memory { t, bytes } => Some((*t, *bytes)),
            _ => None,
        })
        .collect();
    samples.sort_unstable();
    let mut last_gb: Option<f64> = None;
    for (t, bytes) in samples {
        let gb = round3(bytes as f64 / GIB);
        if last_gb == Some(gb) {
            continue;
        }
        last_gb = Some(gb);
        let mut args = Map::new();
        args.insert("allocated_gb".into(), json!(gb));
        events.push(counter("memory", t, args));
    }

    // ── evictions ───────────────────────────────────────────────────────
    for record in log {
        if let Record::Evict { t, agent, pages } = record {
            events.push(instant(
                "evict",
                ENGINE_PID,
                SCHEDULER_TID,
                *t,
                Some(json!({ "agent": tid_of(agent), "pages": pages })),
            ));
        }
    }

    events.sort_by_key(|e| e["ts"].as_u64().unwrap_or(0));

    // ── track names first ───────────────────────────────────────────────
    let mut named: Vec<Value> = vec![
        json!({"ph": "M", "name": "process_name", "pid": AGENTS_PID, "args": {"name": "Agents"}}),
        json!({"ph": "M", "name": "process_name", "pid": ENGINE_PID, "args": {"name": "Engine"}}),
        json!({"ph": "M", "name": "thread_name", "pid": ENGINE_PID, "tid": SCHEDULER_TID, "args": {"name": "Scheduler"}}),
        json!({"ph": "M", "name": "thread_name", "pid": ENGINE_PID, "tid": GPU_TID, "args": {"name": "GPU"}}),
    ];
    for key in &order {
        let a = &agents[key];
        let label = match &a.program {
            Some(program) => format!("{} ({program})", agent_name(a.tid)),
            None => agent_name(a.tid),
        };
        named.push(json!({
            "ph": "M", "name": "thread_name", "pid": AGENTS_PID, "tid": a.tid,
            "args": {"name": label},
        }));
    }
    named.extend(events);
    let count = named.len();

    Rendered {
        json: json!({
            "format": "pie-trace",
            "version": 1,
            "meta": meta_json(meta),
            "traceEvents": named,
        }),
        agents: order.len(),
        steps: steps.len(),
        events: count,
    }
}
