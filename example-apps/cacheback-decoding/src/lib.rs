//! Demonstrates speculative decoding with a cache-based drafter (Cacheback Decoding).
//!
//! This example uses a `CacheDrafter` that records token patterns from the
//! previous context and speculates future tokens based on n-gram matching. The main
//! model then verifies the speculated tokens, accepting matches and rejecting
//! mismatches.

use inferlet::drafter::Drafter;
use inferlet::sampler::Sample;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Result};
use std::cmp::Ordering;
use std::time::Instant;

const HELP: &str = "\
Usage: cacheback-decoding [OPTIONS]

Demonstrates cacheback decoding with a cache-based drafter.

Options:
  -p, --prompt <STRING>    The prompt to send to the model
                           [default: Keep printing 'hello, world!' 100 times.]
  -n, --max-tokens <INT>   The maximum number of new tokens to generate [default: 256]
  -h, --help               Print help information";

/// A simple greedy sampler that always picks the token with highest probability.
struct GreedySampler;

impl Sample for GreedySampler {
    fn sample(&self, ids: &[u32], probs: &[f32]) -> u32 {
        if ids.is_empty() {
            return 0;
        }

        // Find the index of the maximum probability
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        ids[max_idx]
    }
}

/// A simple fixed-size LRU cache.
struct LruRow<T: Copy + PartialEq, const N_COLUMN: usize> {
    items: [Option<T>; N_COLUMN],
    len: usize,
}

impl<T: Copy + PartialEq, const N_COLUMN: usize> LruRow<T, N_COLUMN> {
    fn new() -> Self {
        Self {
            items: [None; N_COLUMN],
            len: 0,
        }
    }

    fn insert(&mut self, item: T) {
        // If the item is already in the cache, move it to the front.
        if let Some(pos) = self.items[..self.len].iter().position(|&x| x == Some(item)) {
            for i in (1..=pos).rev() {
                self.items[i] = self.items[i - 1];
            }
            self.items[0] = Some(item);
        // If the item is not in the cache and the cache is not full, add it to the front.
        } else if self.len < N_COLUMN {
            for i in (1..=self.len).rev() {
                self.items[i] = self.items[i - 1];
            }
            self.items[0] = Some(item);
            self.len += 1;
        // If the item is not in the cache and the cache is full, remove the last item
        // and add the new item to the front.
        } else {
            for i in (1..N_COLUMN).rev() {
                self.items[i] = self.items[i - 1];
            }
            self.items[0] = Some(item);
        }
    }
}

/// A cache table that maps previous tokens to next token sequences. This is a proof
/// of concept implementation that is not optimized for performance.
struct CacheDrafter<
    const N_PREV: usize,
    const N_NEXT: usize,
    const N_ROW: usize,
    const N_COLUMN: usize,
> {
    /// A vector of (key, values) pairs with LRU ordering (most recent at front).
    table: Vec<([u32; N_PREV], LruRow<[u32; N_NEXT], N_COLUMN>)>,

    /// A sliding window recording the last few tokens in the previous update.
    prev_window: Vec<u32>,
}

impl<const N_PREV: usize, const N_NEXT: usize, const N_ROW: usize, const N_COLUMN: usize>
    CacheDrafter<N_PREV, N_NEXT, N_ROW, N_COLUMN>
{
    fn new() -> Self {
        Self {
            table: Vec::new(),
            prev_window: vec![0; N_PREV + N_NEXT - 1],
        }
    }

    fn update_cache(&mut self, prev_tokens: [u32; N_PREV], next_tokens: [u32; N_NEXT]) {
        // If the key already exists, move it to the front (LRU update) and insert the next_tokens.
        if let Some(pos) = self.table.iter().position(|(k, _)| *k == prev_tokens) {
            let (key, mut cache) = self.table.remove(pos);
            cache.insert(next_tokens);
            self.table.insert(0, (key, cache));
        // If the key does not exist, create a new entry and insert the next_tokens.
        } else {
            let mut cache = LruRow::new();
            cache.insert(next_tokens);

            // Evict oldest entry if at capacity
            if self.table.len() >= N_ROW {
                self.table.pop();
            }

            // Insert at front
            self.table.insert(0, (prev_tokens, cache));
        }
    }

    fn get(&self, key: &[u32]) -> Option<&LruRow<[u32; N_NEXT], N_COLUMN>> {
        self.table
            .iter()
            .find(|(k, _)| k[..] == *key)
            .map(|(_, v)| v)
    }
}

impl<const N_PREV: usize, const N_NEXT: usize, const N_ROW: usize, const N_COLUMN: usize> Drafter
    for CacheDrafter<N_PREV, N_NEXT, N_ROW, N_COLUMN>
{
    fn update(&mut self, context: &[u32]) {
        // Combine previously generated tokens with the new context tokens.
        let full_window = self
            .prev_window
            .iter()
            .chain(context.iter())
            .cloned()
            .collect::<Vec<_>>();

        // Update the cache with a sliding window over the combined tokens.
        for window in full_window.windows(N_PREV + N_NEXT) {
            let prev_tokens: [u32; N_PREV] = window[..N_PREV].try_into().unwrap();
            let next_tokens: [u32; N_NEXT] = window[N_PREV..].try_into().unwrap();
            self.update_cache(prev_tokens, next_tokens);
        }

        // Retain the last N_PREV + N_NEXT - 1 tokens.
        self.prev_window
            .copy_from_slice(&full_window[full_window.len() - N_PREV - N_NEXT + 1..]);
    }

    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        // Relative position IDs of the draft tokens. Should start at 1.
        let positions = (1..=N_NEXT as u32).collect::<Vec<_>>();

        // Create a new Trie forest.
        let mut trie = TrieForest::new(1);

        // Update the Trie forest with the cached next tokens.
        if let Some(cache) = self.get(&self.prev_window[self.prev_window.len() - N_PREV..]) {
            for item in cache.items {
                if let Some(item) = item {
                    trie.insert(&item[..], &positions);
                }
            }
        }

        // Linearize the Trie forest into a DFS order.
        trie.linearize()
    }
}

struct TrieNode {
    children: Vec<TrieNode>,
    token: u32,
    position: u32,
}

impl TrieNode {
    fn new(position: u32, token: u32) -> Self {
        Self {
            children: Vec::new(),
            token,
            position,
        }
    }
}

/// See [`inferlet::drafter::Drafter`] for the expected shape of the Trie forest.
struct TrieForest {
    roots: Vec<TrieNode>,
    root_position: u32,
}

impl TrieForest {
    fn new(root_position: u32) -> Self {
        Self {
            roots: Vec::new(),
            root_position,
        }
    }

    // Grow a new path in the Trie forest given a sequence of tokens and positions.
    fn insert(&mut self, tokens: &[u32], positions: &[u32]) {
        if tokens.is_empty() || positions.is_empty() {
            return;
        }

        if positions[0] != self.root_position {
            return;
        }

        let mut candidate_nodes = &mut self.roots;

        for (&token, &position) in tokens.iter().zip(positions.iter()) {
            let candidate_node = candidate_nodes
                .iter()
                .find(|node| node.position == position && node.token == token);

            if candidate_node.is_none() {
                candidate_nodes.push(TrieNode::new(position, token));
            }

            let next_node = candidate_nodes
                .iter_mut()
                .find(|node| node.position == position && node.token == token)
                .unwrap();

            candidate_nodes = &mut next_node.children;
        }
    }

    // Linearize the Trie forest into a DFS order.
    fn linearize(&self) -> (Vec<u32>, Vec<u32>) {
        fn dfs(node: &TrieNode, tokens: &mut Vec<u32>, positions: &mut Vec<u32>) {
            tokens.push(node.token);
            positions.push(node.position);
            for child in node.children.iter() {
                dfs(child, tokens, positions);
            }
        }

        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        for node in self.roots.iter() {
            dfs(node, &mut tokens, &mut positions);
        }
        (tokens, positions)
    }
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Keep printing 'hello, world!' 100 times.".to_string());
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    let mut ctx = model.create_context();
    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    let mut sampler = GreedySampler;
    let mut stop_condition =
        stop_condition::max_len(max_num_outputs).or(stop_condition::ends_with_any(eos_tokens));

    // Use a CacheDrafter with 1 previous token, 2 next tokens, 256 rows, and 4 columns.
    let mut drafter = CacheDrafter::<1, 2, 256, 4>::new();

    let mut num_token_generated_per_step = Vec::new();

    println!("Starting generation with speculative decoding...");

    let output = ctx
        .generate_with_drafter(
            &mut drafter,
            &mut sampler,
            &mut stop_condition,
            Some(&mut num_token_generated_per_step),
        )
        .await;

    println!("Generation completed.");

    let output_token_ids = ctx.tokenizer.tokenize(&output);

    println!(
        "Output: {:?} (total elapsed: {:?})",
        output,
        start.elapsed()
    );

    // Compute per-token latency, avoiding division by zero.
    if !output_token_ids.is_empty() {
        println!(
            "Per token latency: {:?}, Mean accepted tokens per step: {:.4}",
            start.elapsed() / (output_token_ids.len() as u32),
            num_token_generated_per_step.iter().sum::<usize>() as f64
                / num_token_generated_per_step.len() as f64
        );
    }

    Ok(())
}
