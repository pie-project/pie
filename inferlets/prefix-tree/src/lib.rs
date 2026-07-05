//! Demonstrates prefix tree caching — **low-level ① rewrite (chat-EOS, FORK)**.
//!
//! A 1×2×2×2 = 8-leaf prompt tree: each level appends a user turn and forks, so
//! children share their common-prefix KV. Rewritten off the `Context`/`Sampler`
//! facade onto the keep-core: intermediate turns are materialized with
//! `prefill::tokens`, branches with `kv.fork()`, and each of the 8 leaves decodes
//! on the run-ahead carrier (`sampler_program(Argmax)` + `submit_pass`/
//! `discard_pass` depth-1 EOS rollback, `chat::` templating).

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, model, prefill, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_num_tokens")]
    num_tokens: usize,
}

fn default_num_tokens() -> usize { 128 }

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

fn decode_text(tokens: &[u32]) -> Result<String> {
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    match dec.feed(tokens)? {
        chat::Event::Delta(s) | chat::Event::Done(s) => text.push_str(&s),
        _ => {}
    }
    Ok(text)
}

/// Fork `parent` and materialize a user turn on the child (an intermediate tree
/// node), returning the child working set + its cursor.
fn fork_and_prefill(
    parent_kv: &KvWorkingSet,
    parent_seq: u32,
    user_turn: &str,
) -> Result<(KvWorkingSet, u32)> {
    let kv = parent_kv.fork().map_err(|e| format!("fork: {e}"))?;
    let mut seq = parent_seq;
    prefill::tokens(&kv, &mut seq, &chat::user(user_turn))?;
    Ok((kv, seq))
}

const SYSTEM: &str = "You are a helpful, friendly, and knowledgeable science tutor for students of all ages. \
Your goal is to explain complex biological concepts in a clear, accessible, and engaging \
manner, tailoring your language to the specified audience.";

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max = input.num_tokens;
    let start = std::time::Instant::now();
    let vocab = model::output_vocab_size();
    let s = sampler::sampler_program(SamplerSpec::Argmax, vocab)?;
    let stop = chat::stop_tokens();

    // Level 0 — root: prefill the system prompt.
    let root_kv = KvWorkingSet::new();
    let mut root_seq = 0u32;
    prefill::tokens(&root_kv, &mut root_seq, &chat::system(SYSTEM))?;

    // Level 1 — photosynthesis / cellular respiration.
    let (photo_kv, photo_seq) = fork_and_prefill(&root_kv, root_seq,
        "I'm curious about the fundamental process of photosynthesis. \
        Could you provide a detailed overview of how plants create their own food using sunlight, \
        water, and carbon dioxide?")?;
    let (resp_kv, resp_seq) = fork_and_prefill(&root_kv, root_seq,
        "Now, could you explain the equally important process of cellular respiration? \
        I'd like to understand how organisms, including plants and animals, break down glucose to \
        release the energy needed for life.")?;

    // Level 2 — audience / focus.
    let (eli5_kv, eli5_seq) = fork_and_prefill(&photo_kv, photo_seq,
        "That sounds complicated. Could you simplify it significantly for me? \
        Please explain the core idea in a way that a curious 5-year-old child could easily grasp \
        and remember. Use a simple analogy.")?;
    let (hs_kv, hs_seq) = fork_and_prefill(&photo_kv, photo_seq,
        "Thank you. Now, could you provide a more technical explanation suitable for a high school \
        biology student? I'm familiar with basic cell biology and chemistry, so please include \
        relevant terminology like chloroplasts, chlorophyll, and light-dependent reactions.")?;
    let (loc_kv, loc_seq) = fork_and_prefill(&resp_kv, resp_seq,
        "I'm interested in the specific location within the cell where this process occurs. \
        Can you describe the organelles involved and why their specific structures are uniquely \
        suited for this essential energy-releasing function?")?;
    let (prod_kv, prod_seq) = fork_and_prefill(&resp_kv, resp_seq,
        "Focusing on the outputs of this metabolic reaction, what are the primary products \
        that result from this process? Please list and briefly describe the significance of \
        each of these molecules for the cell.")?;

    // Level 3 — the 8 leaves (parent node + question). Decoded sequentially.
    let parents: [(&KvWorkingSet, u32); 4] =
        [(&eli5_kv, eli5_seq), (&hs_kv, hs_seq), (&loc_kv, loc_seq), (&prod_kv, prod_seq)];
    let leaves: [(usize, &str); 8] = [
        (0, "I love cooking! Can you explain the main idea of this process to me by comparing it to \
             a chef's recipe in a kitchen? What are the ingredients, and what is the final dish \
             that the plant creates?"),
        (0, "My favorite thing is playing outside in the sunshine. How does sunlight specifically \
             help a plant? If I covered a plant and blocked all the light, what would happen to it \
             over time, and why?"),
        (1, "For my exam, I need to know the specific chemical equation for this process. Can you \
             write it out with the proper reactants and products, and briefly explain what each \
             component represents?"),
        (1, "Beyond typical land plants, do other organisms like algae or certain bacteria also \
             perform this same process? How does their approach differ from what happens in a \
             typical green leaf?"),
        (2, "Please elaborate specifically on the role of the mitochondria. Describe its inner and \
             outer membranes and the matrix, and explain how this structure makes it the perfect \
             'powerhouse' of the cell during this process."),
        (2, "Is this metabolic pathway entirely identical in both plant and animal cells? Please \
             compare and contrast the process, highlighting any key similarities or differences in \
             where or how cellular respiration occurs in these two major kingdoms."),
        (3, "One of the key products is usable energy. Could you explain in detail the role of \
             adenosine triphosphate (ATP) as the main energy currency? How is it synthesized and \
             then used by the cell to power its activities?"),
        (3, "I understand that carbon dioxide is considered a waste product of this process. Can you \
             elaborate on what exactly happens to this CO2? How does the organism expel it, and what \
             is its ultimate fate in the larger ecosystem?"),
    ];

    println!("--- Starting generation for 8 prompts (max {} tokens each) ---", max);

    let mut results: Vec<Result<String>> = Vec::new();
    for (parent_idx, question) in leaves {
        let (parent_kv, parent_seq) = parents[parent_idx];
        let leaf_kv = parent_kv.fork().map_err(|e| format!("leaf fork: {e}"))?;
        let mut seq = parent_seq;
        let mut fresh = false; // inherited (prefilled) prefix, not a fresh generate
        let mut prompt = chat::user(question);
        prompt.extend(chat::cue());
        let result = decode_pipelined(&leaf_kv, &mut seq, &mut fresh, &s, prompt, max, &stop)
            .await
            .and_then(|t| decode_text(&t));
        results.push(result);
    }

    println!("\n--- All 8 generations completed in {:?} ---\n", start.elapsed());
    for (i, output_text) in results.iter().enumerate() {
        match output_text {
            Ok(text) => println!("Prompt #{}:\n{:?}\n", i + 1, text),
            Err(e) => println!("Prompt #{}: Error: {}\n", i + 1, e),
        }
    }

    Ok(String::new())
}
