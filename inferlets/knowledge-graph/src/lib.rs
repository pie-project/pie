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

use inferlet::{Context, Result, sample::Sampler, model::Model, runtime};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};

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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_tokens = input.max_tokens;
    let depth = input.depth;

    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    // --- Stage 1: Extract triples from the passage ---
    println!("--- Stage 1: Extracting knowledge triples ---");

    let mut extraction_ctx = Context::new(&model)?;
    extraction_ctx.system(EXTRACTION_SYSTEM_PROMPT);
    extraction_ctx.user(&format!(
        "Extract all factual triples from this passage:\n\n{}",
        PASSAGE
    ));
    extraction_ctx.cue();

    let extraction_output = extraction_ctx
        .generate(Sampler::Argmax) // greedy
        .max_tokens(max_tokens)
        .collect_text()
        .await?;

    println!("Extraction output: {}", extraction_output);

    // Stage 2 needs the query context with the system prompt prefilled.
    // We kick off that prefill (GPU) concurrently with graph construction
    // (CPU) below, via futures::join!.
    let mut query_ctx = Context::new(&model)?;
    query_ctx.system(QUERY_SYSTEM_PROMPT);

    let graph_work = async {
        // --- Stage 2: Parse triples and build the knowledge graph ---
        println!("\n--- Stage 2: Building knowledge graph ---");

        let triples = parse_triples(&extraction_output);
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

        println!(
            "Graph: {} nodes, {} edges",
            graph.node_count(),
            graph.edge_count()
        );
        println!(
            "Entities: {}",
            node_map.keys().cloned().collect::<Vec<_>>().join(", ")
        );

        // --- Stage 3: Query the graph for relevant context ---
        println!(
            "\n--- Stage 3: Querying graph (depth={}) for: \"{}\" ---",
            depth, QUESTION
        );

        let query_entities = ["European Union"];
        let mut all_facts = retrieve_facts(&graph, &node_map, &query_entities, depth);
        all_facts.sort();
        all_facts.dedup();

        println!("Retrieved {} relevant facts:", all_facts.len());
        for fact in &all_facts {
            println!("  - {}", fact);
        }

        all_facts
    };

    let (flush_res, all_facts) = futures::join!(query_ctx.flush(), graph_work);
    flush_res?;

    // --- Stage 4: Answer the question using graph context ---
    println!("\n--- Stage 4: Generating answer ---");

    query_ctx.user(&format!(
        "Knowledge graph facts:\n{}\n\nQuestion: {}",
        all_facts
            .iter()
            .map(|f| format!("- {}", f))
            .collect::<Vec<_>>()
            .join("\n"),
        QUESTION
    ));
    query_ctx.cue();

    let answer = query_ctx
        .generate(Sampler::Argmax)
        .max_tokens(max_tokens)
        .collect_text()
        .await?;

    println!("Answer: {}", answer);

    Ok(String::new())
}
