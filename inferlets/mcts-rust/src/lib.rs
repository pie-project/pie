//! MCTS-reasoning inferlet.
//!
//! A standalone example of **Monte Carlo Tree Search (MCTS) reasoning** for
//! LLM problem solving. Instead of generating a fixed breadth-first tree of
//! thoughts and pruning (that is `tree-of-thought`), MCTS *budgets* its model
//! calls: it repeatedly walks the most promising path, expands it, simulates a
//! candidate answer, scores it, and feeds that score back up the tree. Over
//! many iterations the visit counts concentrate on the branch that keeps
//! scoring well.
//!
//! The classic four-phase loop, once per iteration:
//!
//! 1. **Selection** — from the root, descend by Upper Confidence Bound (UCB)
//!    until reaching a node that is not fully expanded (or is terminal).
//! 2. **Expansion** — ask the model for `branch_factor` candidate next
//!    reasoning steps and attach them as children.
//! 3. **Simulation / rollout** — from one new child, generate a candidate
//!    final answer.
//! 4. **Evaluation + backpropagation** — score the candidate 0–100, normalize
//!    to `[0,1]`, and add it to every node on the path while bumping visits.
//!
//! After the budget is spent we follow the most-visited child at each level to
//! recover the best reasoning path, then synthesize a final answer from it.
//!
//! Prompts that advertise the `ACTION:` / `ARGS:` / `FINAL:` protocol use an
//! agent mode: expansion proposes complete next actions, evaluation scores those
//! actions, and the best action is returned directly to the outer tool loop.
//! Ordinary reasoning prompts retain the full rollout and synthesis phases.
//!
//! Why UCB? It trades **exploitation** (a node's average score so far) against
//! **exploration** (a bonus for nodes visited few times relative to their
//! parent), so the search neither fixates on an early lucky branch nor wastes
//! the whole budget sampling uniformly. Unvisited children get an infinite
//! score, so every child is tried at least once before any is revisited.
//!
//! MVP scope: each phase uses a fresh short [`Context`] (expand / rollout /
//! evaluate / synthesize). That keeps the control flow obviously correct at
//! the cost of recomputing shared prompt prefixes — see the README for the
//! `Context::fork()` optimization left as future work. This is a *correctness
//! / demonstration* inferlet, not a tuned reasoning benchmark.

use inferlet::{
    Context, Result,
    chat::{self, Decoder, Event},
    model::Model,
    runtime,
    sample::Sampler,
};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_iterations")]
    max_iterations: usize,
    #[serde(default = "default_max_depth")]
    max_depth: usize,
    #[serde(default = "default_branch_factor")]
    branch_factor: usize,
    #[serde(default = "default_rollout_tokens")]
    rollout_tokens: usize,
    #[serde(default = "default_final_tokens")]
    final_tokens: usize,
    #[serde(default = "default_exploration_constant")]
    exploration_constant: f32,
    #[serde(default = "default_show_trace")]
    show_trace: bool,
}

fn default_prompt() -> String {
    "A farmer has 17 sheep and all but 9 run away. How many are left?".to_string()
}
fn default_max_iterations() -> usize {
    16
}
fn default_max_depth() -> usize {
    4
}
fn default_branch_factor() -> usize {
    3
}
fn default_rollout_tokens() -> usize {
    128
}
fn default_final_tokens() -> usize {
    256
}
fn default_exploration_constant() -> f32 {
    1.414
}
fn default_show_trace() -> bool {
    true
}

// =============================================================================
// Search tree (arena-allocated)
// =============================================================================
//
// Nodes are stored in a flat `Vec` and referenced by index. An arena avoids
// the `Rc<RefCell<Node>>` dance you'd otherwise need for a tree with parent
// pointers, and makes the pure search logic trivially unit-testable.

/// One node in the search tree — a partial reasoning state.
#[derive(Debug, Clone)]
struct Node {
    /// Index of the parent node, or `None` for the root.
    parent: Option<usize>,
    /// Indices of attached children.
    children: Vec<usize>,
    /// Distance from the root (root = 0).
    depth: usize,
    /// The reasoning step (action) that produced this node. Empty at the root.
    action: String,
    /// How many simulations have passed through this node.
    visits: u32,
    /// Sum of normalized rollout values backed up through this node.
    value_sum: f32,
    /// Terminal nodes are never expanded (they sit at `max_depth`).
    terminal: bool,
}

impl Node {
    fn mean_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

/// The MCTS search tree. Holds the node arena and all *pure* search logic
/// (selection, UCB, expansion bookkeeping, backpropagation). Nothing here
/// touches the model, so the whole module is unit-testable on the host.
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    /// Create a tree containing only the root (depth 0, no action).
    fn new() -> Self {
        Tree {
            nodes: vec![Node {
                parent: None,
                children: Vec::new(),
                depth: 0,
                action: String::new(),
                visits: 0,
                value_sum: 0.0,
                terminal: false,
            }],
        }
    }

    fn root(&self) -> usize {
        0
    }

    /// Attach a child carrying `action` under `parent`. The child is marked
    /// terminal when it reaches `max_depth` (it can never be expanded further).
    fn add_child(&mut self, parent: usize, action: String, max_depth: usize) -> usize {
        let depth = self.nodes[parent].depth + 1;
        let id = self.nodes.len();
        self.nodes.push(Node {
            parent: Some(parent),
            children: Vec::new(),
            depth,
            action,
            visits: 0,
            value_sum: 0.0,
            terminal: depth >= max_depth,
        });
        self.nodes[parent].children.push(id);
        id
    }

    /// A node is fully expanded once it has `branch_factor` children.
    fn is_fully_expanded(&self, node: usize, branch_factor: usize) -> bool {
        self.nodes[node].children.len() >= branch_factor
    }

    /// The child of `node` with the highest UCB score, or `None` if it has no
    /// children. Unvisited children win automatically (UCB returns +∞).
    fn best_ucb_child(&self, node: usize, c: f32) -> Option<usize> {
        let parent_visits = self.nodes[node].visits;
        self.nodes[node].children.iter().copied().max_by(|&a, &b| {
            let sa = ucb(
                self.nodes[a].mean_value(),
                self.nodes[a].visits,
                parent_visits,
                c,
            );
            let sb = ucb(
                self.nodes[b].mean_value(),
                self.nodes[b].visits,
                parent_visits,
                c,
            );
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// **Selection**: descend from `node` by UCB while the current node is
    /// fully expanded, non-terminal, and has children. Stops at the first node
    /// that can still grow a child (or at a terminal/leaf node).
    fn select(&self, mut node: usize, c: f32, branch_factor: usize) -> usize {
        loop {
            if self.nodes[node].terminal {
                return node;
            }
            if !self.is_fully_expanded(node, branch_factor) {
                return node;
            }
            match self.best_ucb_child(node, c) {
                Some(child) => node = child,
                None => return node, // fully expanded but childless (branch_factor==0)
            }
        }
    }

    /// **Backpropagation**: add `value` to every node from `node` up to the
    /// root, incrementing each one's visit count.
    fn backpropagate(&mut self, node: usize, value: f32) {
        let mut cur = Some(node);
        while let Some(i) = cur {
            self.nodes[i].visits += 1;
            self.nodes[i].value_sum += value;
            cur = self.nodes[i].parent;
        }
    }

    /// The most-visited child of `node` — the *robust* choice used to read off
    /// the final path (visit count is a steadier signal than mean value).
    fn most_visited_child(&self, node: usize) -> Option<usize> {
        self.nodes[node]
            .children
            .iter()
            .copied()
            .max_by_key(|&c| self.nodes[c].visits)
    }

    /// Actions from the root down to `node`, in order (root's empty action is
    /// skipped).
    fn path_actions(&self, node: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = Some(node);
        while let Some(i) = cur {
            if !self.nodes[i].action.is_empty() {
                out.push(self.nodes[i].action.clone());
            }
            cur = self.nodes[i].parent;
        }
        out.reverse();
        out
    }

    /// Follow `most_visited_child` from the root to a leaf and return the
    /// actions along that best path.
    fn best_path(&self) -> Vec<String> {
        let mut node = self.root();
        while let Some(child) = self.most_visited_child(node) {
            node = child;
        }
        self.path_actions(node)
    }
}

// =============================================================================
// Pure helpers (unit-tested)
// =============================================================================

/// UCB1 score for one child.
///
/// ```text
/// ucb = mean_value + c * sqrt( ln(parent_visits + 1) / (child_visits + 1) )
/// ```
///
/// An unvisited child returns `+∞` so it is always tried before any sibling is
/// revisited — the standard "expand every action at least once" rule.
fn ucb(mean_value: f32, child_visits: u32, parent_visits: u32, c: f32) -> f32 {
    if child_visits == 0 {
        return f32::INFINITY;
    }
    let exploration = ((parent_visits as f32 + 1.0).ln() / (child_visits as f32 + 1.0)).sqrt();
    mean_value + c * exploration
}

/// Parse a model-produced score into a normalized value in `[0,1]`.
///
/// The evaluation prompt asks for a number 0–100, but models are inconsistent
/// ("85", "Score: 85/100", "0.85"). We grab the first numeric token and
/// normalize: anything `> 1.0` is treated as a 0–100 score and divided by 100;
/// a bare fraction `<= 1.0` is taken as already normalized. Unparseable output
/// falls back to a neutral `0.5` so a flaky evaluator never crashes the search.
fn parse_score(text: &str) -> f32 {
    match first_number(text) {
        Some(v) => {
            let normalized = if v > 1.0 { v / 100.0 } else { v };
            normalized.clamp(0.0, 1.0)
        }
        None => 0.5,
    }
}

/// Extract the first decimal number from arbitrary text, or `None`.
fn first_number(text: &str) -> Option<f32> {
    let mut buf = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() || (ch == '.' && !buf.contains('.')) {
            buf.push(ch);
        } else if !buf.is_empty() {
            break;
        }
    }
    // Reject a lone "." that no digit followed.
    if buf.is_empty() || buf == "." {
        None
    } else {
        buf.parse::<f32>().ok()
    }
}

// =============================================================================
// Model-backed phases
// =============================================================================
//
// Each helper runs one isolated generation in a fresh Context. Splitting the
// phases keeps the prompts small and the control flow readable; the trade-off
// is recomputing the shared problem prefix every call (see README).

fn path_block(path: &[String]) -> String {
    if path.is_empty() {
        "(no steps yet)".to_string()
    } else {
        path.iter()
            .enumerate()
            .map(|(i, s)| format!("{}. {}", i + 1, s.trim()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn is_agent_protocol(problem: &str) -> bool {
    problem.contains("ACTION:") && problem.contains("ARGS:") && problem.contains("FINAL:")
}

fn phase_token_budget(problem: &str, requested: usize) -> usize {
    if is_agent_protocol(problem) {
        requested.max(64)
    } else {
        requested
    }
}

async fn generate_text(
    mut ctx: Context,
    model: &Model,
    sampler: Sampler,
    max_tokens: usize,
) -> Result<(String, usize)> {
    let result: Result<(String, usize)> = async {
        let stop_tokens = chat::stop_tokens(model);
        let mut generator = ctx
            .generate(sampler)
            .max_tokens(max_tokens)
            .stop(&stop_tokens)
            .disable_system_speculation()
            .rebid_each_step(false);
        let mut decoder = Decoder::new(model);
        let mut text = String::new();
        let mut generated_tokens = 0usize;
        while let Some(step) = generator.next()? {
            let output = step.execute().await?;
            generated_tokens += output.tokens.len();
            match decoder.feed(&output.tokens)? {
                Event::Delta(delta) => text.push_str(&delta),
                Event::Done(complete) => {
                    text = complete;
                    break;
                }
                Event::Idle | Event::Interrupt(_) => {}
            }
        }
        Ok((text, generated_tokens))
    }
    .await;

    // Dropping a WIT context handle does not free its pages until process
    // cleanup. MCTS creates several phase contexts per iteration, so destroy
    // each one explicitly before starting the next phase.
    ctx.destroy();
    result
}

const PREFILL_CHUNK_TOKENS: usize = 240;

async fn flush_tokens_bounded(
    ctx: &mut Context,
    tokens: &[u32],
    phase: &str,
    turn: &str,
) -> Result<()> {
    for (chunk_index, chunk) in tokens.chunks(PREFILL_CHUNK_TOKENS).enumerate() {
        ctx.append(chunk);
        ctx.flush()
            .await
            .map_err(|e| format!("mcts {phase} {turn} prefill chunk {chunk_index}: {e}"))?;
    }
    Ok(())
}

/// Preserve the serialized chat prompt while keeping each forward request
/// below drivers that expose at most 256 probability rows.
async fn prefill_phase(ctx: &mut Context, system: &str, user: &str, phase: &str) -> Result<()> {
    let system_tokens = chat::system(ctx.model(), system);
    flush_tokens_bounded(ctx, &system_tokens, phase, "system").await?;

    let user_tokens = chat::user(ctx.model(), user);
    flush_tokens_bounded(ctx, &user_tokens, phase, "user").await?;
    Ok(())
}

/// **Expansion** — propose one diverse next reasoning step given the path so
/// far. We sample `branch_factor` separate short completions (rather than
/// parsing one numbered list) so each child is a clean, independent step.
async fn expand_step(
    model: &Model,
    problem: &str,
    path: &[String],
    rollout_tokens: usize,
) -> Result<(String, usize)> {
    let mut ctx = Context::new(model)?;
    let system = if is_agent_protocol(problem) {
        "Choose the single best next tool-use action for the agent transcript. Return exactly ACTION plus ARGS, or FINAL if the task is complete. Do not claim success before the required tool action has run."
    } else {
        "You extend a chain of reasoning. Given a problem and the steps so far, propose ONE concise next reasoning step. Output only that step."
    };
    let user = format!(
        "Problem:\n{problem}\n\nReasoning so far:\n{}\n\nNext reasoning step:",
        path_block(path)
    );
    prefill_phase(&mut ctx, system, &user, "expansion").await?;
    ctx.cue();
    let (text, generated_tokens) = generate_text(
        ctx,
        model,
        Sampler::TopP {
            temperature: 0.8,
            p: 0.95,
        },
        phase_token_budget(problem, rollout_tokens.min(64).max(16)),
    )
    .await
    .map_err(|e| format!("mcts expansion generation: {e}"))?;
    Ok((text.trim().to_string(), generated_tokens))
}

/// **Simulation / rollout** — from the current path, produce a candidate final
/// answer in one short generation.
async fn rollout(
    model: &Model,
    problem: &str,
    path: &[String],
    rollout_tokens: usize,
) -> Result<(String, usize)> {
    let mut ctx = Context::new(model)?;
    let system = if is_agent_protocol(problem) {
        "Propose the single best next command for this tool-using agent. Return exactly ACTION plus ARGS, or FINAL only when the transcript proves the task is complete."
    } else {
        "You finish a partial chain of reasoning. Continue from the steps given and produce a short, concrete candidate final answer."
    };
    let user = format!(
        "Problem:\n{problem}\n\nReasoning so far:\n{}\n\nCandidate:",
        path_block(path)
    );
    prefill_phase(&mut ctx, system, &user, "rollout").await?;
    ctx.cue();
    let (text, generated_tokens) = generate_text(
        ctx,
        model,
        Sampler::TopP {
            temperature: 0.7,
            p: 0.95,
        },
        phase_token_budget(problem, rollout_tokens),
    )
    .await
    .map_err(|e| format!("mcts rollout generation: {e}"))?;
    Ok((text.trim().to_string(), generated_tokens))
}

/// **Evaluation** — score a candidate answer 0–100. Returns the normalized
/// value in `[0,1]` (with the `0.5` fallback baked into [`parse_score`]).
async fn evaluate(model: &Model, problem: &str, candidate: &str) -> Result<(f32, usize)> {
    let mut ctx = Context::new(model)?;
    let system = if is_agent_protocol(problem) {
        "Score this proposed next agent command from 0 to 100. Reward valid, necessary tool use and penalize premature FINAL answers. Return ONLY the number."
    } else {
        "You are a strict grader. Score the candidate answer from 0 to 100 for correctness, completeness, and reasoning quality. Return ONLY the number."
    };
    let user = format!("Problem:\n{problem}\n\nCandidate:\n{candidate}\n\nScore (0-100):");
    prefill_phase(&mut ctx, system, &user, "evaluation").await?;
    ctx.cue();
    let (text, generated_tokens) = generate_text(ctx, model, Sampler::Argmax, 8)
        .await
        .map_err(|e| format!("mcts evaluation generation: {e}"))?;
    Ok((parse_score(&text), generated_tokens))
}

/// **Final synthesis** — write the answer the inferlet returns, conditioned on
/// the best reasoning path MCTS found.
async fn synthesize(
    model: &Model,
    problem: &str,
    best_path: &[String],
    final_tokens: usize,
) -> Result<(String, usize)> {
    let mut ctx = Context::new(model)?;
    let system = if is_agent_protocol(problem) {
        "Select the single best next command from the vetted path. Return exactly ACTION plus ARGS, or FINAL only if the transcript shows the task is complete. Output no commentary."
    } else {
        "You write the final answer to a problem, guided by a vetted chain of reasoning. Be clear and correct."
    };
    let user = format!(
        "Problem:\n{problem}\n\nBest reasoning path:\n{}\n\nResponse:",
        path_block(best_path)
    );
    prefill_phase(&mut ctx, system, &user, "synthesis").await?;
    ctx.cue();
    let (text, generated_tokens) = generate_text(ctx, model, Sampler::Argmax, final_tokens)
        .await
        .map_err(|e| format!("mcts synthesis generation: {e}"))?;
    Ok((text.trim().to_string(), generated_tokens))
}

// =============================================================================
// Entry point
// =============================================================================

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("No models available")?)?;

    // Clamp pathological inputs so the control flow always terminates sensibly.
    let max_iterations = input.max_iterations.max(1);
    let max_depth = input.max_depth.max(1);
    let branch_factor = input.branch_factor.max(1);
    let c = input.exploration_constant;
    let agent_mode = is_agent_protocol(&input.prompt);

    println!("--- mcts-rust ---");
    println!(
        "iterations={} max_depth={} branch_factor={} c={:.3}",
        max_iterations, max_depth, branch_factor, c
    );

    let mut tree = Tree::new();
    let mut best_score = 0.0f32;
    let mut best_candidate = String::new();
    let mut generated_tokens_total = 0usize;

    for iter in 0..max_iterations {
        // (a) Selection.
        let selected = tree.select(tree.root(), c, branch_factor);

        // (b) Expansion: grow one child unless the node is terminal or full.
        let can_expand = !tree.nodes[selected].terminal
            && tree.nodes[selected].depth < max_depth
            && !tree.is_fully_expanded(selected, branch_factor);

        let (sim_node, expanded_children) = if can_expand {
            let path = tree.path_actions(selected);
            let (action, generated_tokens) =
                expand_step(&model, &input.prompt, &path, input.rollout_tokens).await?;
            generated_tokens_total += generated_tokens;
            let child = tree.add_child(selected, action, max_depth);
            (child, tree.nodes[selected].children.len())
        } else {
            // Re-simulate the selected node (e.g. a terminal leaf or, with
            // branch_factor==1, an already-expanded path).
            (selected, tree.nodes[selected].children.len())
        };

        // (c) For ordinary reasoning, simulate a final answer from this node.
        // Agent benchmarks already expand complete next actions; rolling those
        // out and synthesizing again only replays the full transcript and can
        // exhaust the launch token wallet before the next agent step.
        let (candidate, generated_tokens) = if agent_mode {
            (tree.nodes[sim_node].action.clone(), 0)
        } else {
            let sim_path = tree.path_actions(sim_node);
            rollout(&model, &input.prompt, &sim_path, input.rollout_tokens).await?
        };
        generated_tokens_total += generated_tokens;

        // (d) Evaluation.
        let (value, generated_tokens) = evaluate(&model, &input.prompt, &candidate).await?;
        generated_tokens_total += generated_tokens;

        // (e) Backpropagation.
        tree.backpropagate(sim_node, value);

        if value >= best_score {
            best_score = value;
            best_candidate = candidate;
        }

        println!(
            "iteration={} selected_node={} expanded_children={} rollout_score={:.3} best_score={:.3}",
            iter, selected, expanded_children, value, best_score
        );
    }

    // Read off the best path (most-visited child at each level) and synthesize.
    let best_path = tree.best_path();
    let (final_answer, final_generated_tokens) = if agent_mode && !best_candidate.is_empty() {
        // Expansion candidates are already complete next actions in agent mode.
        (best_candidate.clone(), 0)
    } else if best_path.is_empty() {
        // No expansion happened (e.g. max_iterations far below branch_factor):
        // fall back to the best rollout candidate we saw.
        if best_candidate.is_empty() {
            rollout(&model, &input.prompt, &[], input.final_tokens).await?
        } else {
            (best_candidate.clone(), 0)
        }
    } else {
        synthesize(&model, &input.prompt, &best_path, input.final_tokens).await?
    };
    generated_tokens_total += final_generated_tokens;
    println!("generated_tokens_total={}", generated_tokens_total);

    if !input.show_trace {
        return Ok(final_answer);
    }

    let mut out = String::new();
    out.push_str("Final answer:\n");
    out.push_str(&final_answer);
    out.push_str("\n\nBest reasoning path:\n");
    if best_path.is_empty() {
        out.push_str("(search produced no expanded steps)\n");
    } else {
        for (i, step) in best_path.iter().enumerate() {
            out.push_str(&format!("{}. {}\n", i + 1, step.trim()));
        }
    }
    out.push_str(&format!(
        "\nMCTS summary:\niterations={}\nnodes={}\nbest_score={:.3}\n",
        max_iterations,
        tree.nodes.len(),
        best_score
    ));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_agent_protocol_and_reserves_action_budget() {
        let prompt = "ACTION: tool\nARGS: {}\nOR FINAL: answer";
        assert!(is_agent_protocol(prompt));
        assert_eq!(phase_token_budget(prompt, 32), 64);
        assert_eq!(phase_token_budget("ordinary reasoning", 32), 32);
    }
}
