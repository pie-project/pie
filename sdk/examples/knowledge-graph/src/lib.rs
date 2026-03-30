//! Demonstrates knowledge graph extraction and querying with petgraph.
//!
//! This example has the LLM extract entity-relation triples from a text
//! passage, builds a directed graph using `petgraph`, then uses the graph
//! structure to provide context for answering a follow-up question.
//!
//! It also demonstrates overlapping GPU and CPU work: the query context's
//! system prompt is submitted for prefill while the graph is being built,
//! using `futures::join!` so the GPU forward pass runs concurrently with
//! the CPU-bound triple parsing and graph construction.

use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Result, Sampler};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet, VecDeque};

const HELP: &str = "\
Usage: knowledge-graph [OPTIONS]

Extracts a knowledge graph from text using an LLM and petgraph,
then answers a question using the graph as context.

Options:
  -t, --max-tokens <N>  Max tokens per generation step [default: 2048]
  -d, --depth <N>       Graph traversal depth for fact retrieval [default: 3]
  -h, --help            Prints help information";

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

/// Extracts the section after the "RELATIONS:" marker, discarding any leading
/// thinking tokens. Falls back to the full text if the marker is absent.
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

/// BFS retrieval: starting from `seed_entities`, collects all facts within
/// `depth` hops. At each depth level, every newly discovered neighbor's
/// edges are collected in the next round.
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
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let max_tokens: usize = args.value_from_str(["-t", "--max-tokens"]).unwrap_or(2048);
    let depth: usize = args.value_from_str(["-d", "--depth"]).unwrap_or(3);

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();
    let stop_condition =
        stop_condition::max_len(max_tokens).or(stop_condition::ends_with_any(eos_tokens));

    // --- Stage 1: Extract triples from the passage ---
    println!("--- Stage 1: Extracting knowledge triples ---");

    let mut extraction_ctx = model.create_context();
    extraction_ctx.fill_system(EXTRACTION_SYSTEM_PROMPT);
    extraction_ctx.fill_user(&format!(
        "Extract all factual triples from this passage:\n\n{}",
        PASSAGE
    ));

    let extraction_output = extraction_ctx
        .generate(Sampler::greedy(), stop_condition.clone())
        .await;

    println!("Extraction output: {}", extraction_output);

    // Start prefilling the query context's system prompt on the GPU while
    // we do CPU-bound graph construction below.
    let mut query_ctx = model.create_context();
    query_ctx.fill_system(QUERY_SYSTEM_PROMPT);

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

    let (_, all_facts) = futures::join!(query_ctx.flush(), graph_work);

    // --- Stage 4: Answer the question using graph context ---
    println!("\n--- Stage 4: Generating answer ---");

    query_ctx.fill_user(&format!(
        "Knowledge graph facts:\n{}\n\nQuestion: {}",
        all_facts
            .iter()
            .map(|f| format!("- {}", f))
            .collect::<Vec<_>>()
            .join("\n"),
        QUESTION
    ));

    let answer = query_ctx
        .generate(Sampler::greedy(), stop_condition.clone())
        .await;

    println!("Answer: {}", answer);

    Ok(())
}
