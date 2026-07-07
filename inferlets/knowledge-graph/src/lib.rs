//! Demonstrates knowledge graph extraction and querying with petgraph.
//!
//! The LLM extracts entity-relation triples from a text passage; those are
//! parsed into a `petgraph` DiGraph; BFS traversal collects facts within a
//! depth bound; the facts are fed back to the model as context for a
//! follow-up question.
//!
//! Also demonstrates GPU/CPU overlap: the query context's system-prompt
//! prefill is submitted asynchronously while the graph is being built on
//! CPU, joined via `futures::join!`.

//! Low-level ① rewrite (chat-EOS, pipelined): the 2 chat decodes run on the raw
//! run-ahead carrier (`sampler::sampler_program(Argmax)` + `carrier::submit_pass`
//! / `discard_pass` depth-1 EOS rollback, `chat::` templating) — NO `Context`/
//! `Generator`/`Sampler` facade. The GPU/CPU overlap (query system-prefill ∥
//! graph build) is preserved raw: the system prefill fire is submitted in-flight,
//! the CPU graph work runs while it computes, then it is drained (no `futures::join!`).

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, prefill, Result};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};

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

/// Run-ahead (pipelined) decode with depth-1 EOS rollback (chat-EOS pattern).
/// Continues from the current `*seq_len` (so a prior prefill on `kv` is reused).
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

/// Detokenize generated tokens through the chat decoder (strip template markers).
fn decode_text(tokens: &[u32]) -> Result<String> {
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    match dec.feed(tokens)? {
        chat::Event::Delta(s) | chat::Event::Done(s) => text.push_str(&s),
        _ => {}
    }
    Ok(text)
}

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_depth")]
    depth: usize,
}

fn default_max_tokens() -> usize { 2048 }
fn default_depth() -> usize { 3 }

const PASSAGE: &str = "\
France is a country in Western Europe. Paris is the capital of France. \
The Eiffel Tower is a landmark located in Paris. France borders Germany to the east. \
Berlin is the capital of Germany. The Brandenburg Gate is a landmark in Berlin. \
Germany borders Poland to the east. Warsaw is the capital of Poland. \
The Palace of Culture and Science is a landmark in Warsaw. \
France is a member of the European Union. Germany is a member of the European Union. \
Poland is a member of the European Union. The European Union is headquartered in Brussels. \
Brussels is the capital of Belgium. Belgium borders France to the south.";

const EXTRACTION_SYSTEM_PROMPT: &str = "\
You are a knowledge extraction assistant. Given a text passage, extract factual \
relationships as triples.\n\n\
Output format: start with the line \"RELATIONS:\" followed by one triple per line \
in the exact format:\n\
subject | relation | object\n\n\
Rules:\n\
- Use consistent entity names (e.g. always \"France\", not \"france\" or \"the country of France\")\n\
- Each triple should capture a single factual relationship\n\
- Do not output anything after the last triple";

const QUERY_SYSTEM_PROMPT: &str = "\
You are a helpful assistant that answers questions using provided knowledge graph data. \
You will receive a list of facts extracted from a knowledge graph. Use only these facts \
to answer the question. Be concise.";

const QUESTION: &str = "What landmarks can you find in the capitals of EU member countries?";

struct Triple {
    subject: String,
    relation: String,
    object: String,
}

/// Returns the text after the last "RELATIONS:" marker, discarding any
/// leading thinking tokens.
fn extract_relations_section(text: &str) -> &str {
    text.rfind("RELATIONS:")
        .map(|pos| text[pos + "RELATIONS:".len()..].trim())
        .unwrap_or_else(|| text.trim())
}

fn parse_triples(text: &str) -> Vec<Triple> {
    let section = extract_relations_section(text);
    section
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
            if parts.len() == 3
                && !parts[0].is_empty()
                && !parts[1].is_empty()
                && !parts[2].is_empty()
            {
                Some(Triple {
                    subject: parts[0].to_string(),
                    relation: parts[1].to_string(),
                    object: parts[2].to_string(),
                })
            } else {
                None
            }
        })
        .collect()
}

fn get_or_insert_node(
    graph: &mut DiGraph<String, String>,
    node_map: &mut HashMap<String, NodeIndex>,
    name: &str,
) -> NodeIndex {
    *node_map
        .entry(name.to_string())
        .or_insert_with(|| graph.add_node(name.to_string()))
}

/// BFS retrieval: from `seed_entities`, collect all facts within `depth` hops.
/// At each level, every newly discovered neighbor's edges are collected next.
fn retrieve_facts(
    graph: &DiGraph<String, String>,
    node_map: &HashMap<String, NodeIndex>,
    seed_entities: &[&str],
    depth: usize,
) -> Vec<String> {
    let mut visited: HashSet<NodeIndex> = HashSet::new();
    let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
    let mut facts: Vec<String> = Vec::new();

    for &entity in seed_entities {
        if let Some(&idx) = node_map.get(entity) {
            if visited.insert(idx) {
                queue.push_back((idx, 0));
            }
        }
    }

    while let Some((node_idx, current_depth)) = queue.pop_front() {
        let entity = &graph[node_idx];

        for edge in graph.edges(node_idx) {
            let target_idx = edge.target();
            let target = &graph[target_idx];
            facts.push(format!("{} {} {}", entity, edge.weight(), target));
            if current_depth + 1 < depth && visited.insert(target_idx) {
                queue.push_back((target_idx, current_depth + 1));
            }
        }

        for edge in graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
            let source_idx = edge.source();
            let source = &graph[source_idx];
            facts.push(format!("{} {} {}", source, edge.weight(), entity));
            if current_depth + 1 < depth && visited.insert(source_idx) {
                queue.push_back((source_idx, current_depth + 1));
            }
        }
    }

    facts
}

/// Build a graph from extracted triples + BFS-retrieve the query facts (pure CPU).
fn build_and_query(extraction_output: &str, depth: usize) -> Vec<String> {
    // --- Stage 2: Parse triples and build the knowledge graph ---
    println!("\n--- Stage 2: Building knowledge graph ---");

    let triples = parse_triples(extraction_output);
    println!("Extracted {} triples:", triples.len());
    for t in &triples {
        println!("  {} | {} | {}", t.subject, t.relation, t.object);
    }

    let mut graph = DiGraph::<String, String>::new();
    let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

    for triple in &triples {
        let src = get_or_insert_node(&mut graph, &mut node_map, &triple.subject);
        let dst = get_or_insert_node(&mut graph, &mut node_map, &triple.object);
        graph.add_edge(src, dst, triple.relation.clone());
    }

    println!("Graph: {} nodes, {} edges", graph.node_count(), graph.edge_count());
    println!("Entities: {}", node_map.keys().cloned().collect::<Vec<_>>().join(", "));

    // --- Stage 3: Query the graph for relevant context ---
    println!("\n--- Stage 3: Querying graph (depth={}) for: \"{}\" ---", depth, QUESTION);

    let query_entities = ["European Union"];
    let mut all_facts = retrieve_facts(&graph, &node_map, &query_entities, depth);
    all_facts.sort();
    all_facts.dedup();

    println!("Retrieved {} relevant facts:", all_facts.len());
    for fact in &all_facts {
        println!("  - {}", fact);
    }
    all_facts
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_tokens = input.max_tokens;
    let depth = input.depth;
    let vocab = model::output_vocab_size();
    let s = sampler::sampler_program(SamplerSpec::Argmax, vocab)?; // greedy
    let stop = chat::stop_tokens();

    // --- Stage 1: Extract triples (chat-EOS decode on the run-ahead carrier) ---
    println!("--- Stage 1: Extracting knowledge triples ---");
    let mut ext_prompt = chat::system_user(
        EXTRACTION_SYSTEM_PROMPT,
        &format!("Extract all factual triples from this passage:\n\n{}", PASSAGE),
    );
    ext_prompt.extend(chat::cue());

    let ext_kv = KvWorkingSet::new();
    let mut e_seq = 0u32;
    let mut e_fresh = true;
    let ext_tokens =
        decode_pipelined(&ext_kv, &mut e_seq, &mut e_fresh, &s, ext_prompt, max_tokens, &stop).await?;
    let extraction_output = decode_text(&ext_tokens)?;
    println!("Extraction output: {}", extraction_output);

    // --- Stage 2/3: build the graph + retrieve facts (CPU), OVERLAPPED with the
    // query system-prompt prefill (GPU). Raw run-ahead overlap: submit the system
    // prefill IN-FLIGHT, run the CPU graph work while it computes, then drain it. ---
    let query_kv = KvWorkingSet::new();
    let mut q_seq = 0u32;
    let sys_tokens = chat::system(QUERY_SYSTEM_PROMPT);
    // Prefill the query system prompt IN-FLIGHT via the non-sampling keep-core
    // primitive (KV materialization, no sampler/await); the CPU graph build below
    // overlaps the GPU prefill, and the decode's prime is stream-ordered after it
    // (no explicit drain needed).
    prefill::tokens(&query_kv, &mut q_seq, &sys_tokens)?;
    let mut q_fresh = sys_tokens.is_empty(); // the prefill was the first fire (unless empty)

    let all_facts = build_and_query(&extraction_output, depth); // CPU ∥ the in-flight prefill

    // --- Stage 4: Answer using graph context (continues from the prefilled KV) ---
    println!("\n--- Stage 4: Generating answer ---");
    let mut q_prompt = chat::user(&format!(
        "Knowledge graph facts:\n{}\n\nQuestion: {}",
        all_facts.iter().map(|f| format!("- {}", f)).collect::<Vec<_>>().join("\n"),
        QUESTION
    ));
    q_prompt.extend(chat::cue());

    let ans_tokens =
        decode_pipelined(&query_kv, &mut q_seq, &mut q_fresh, &s, q_prompt, max_tokens, &stop).await?;
    let answer = decode_text(&ans_tokens)?;
    println!("Answer: {}", answer);

    Ok(String::new())
}
