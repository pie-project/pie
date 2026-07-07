//! Demonstrates a collaborative agent swarm for story writing.
//!
//! This example implements a single agent worker in a multi-agent pipeline
//! where each agent has a specific role (idea generator, plot developer,
//! character creator, or dialogue writer) and passes work to the next agent.

use inferlet::inference::ForwardPass;
use inferlet::sampler::{self, LoweredSampler, SamplerSpec};
use inferlet::working_set::KvWorkingSet;
use inferlet::{carrier, chat, messaging, model, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    role: String,
    #[serde(default = "default_group_id")]
    group_id: u32,
    #[serde(default = "default_tokens_per_step")]
    tokens_per_step: usize,
    #[serde(default = "default_prompt")]
    prompt: String,
}

fn default_group_id() -> u32 { 0 }
fn default_tokens_per_step() -> usize { 512 }
fn default_prompt() -> String { "A story about day dreaming in a park".to_string() }

struct AgentConfig {
    #[allow(dead_code)]
    name: &'static str,
    system_message: &'static str,
    task_instruction: &'static str,
    section_header: &'static str,
    prev_topic: Option<&'static str>,
    next_topic: Option<&'static str>,
}

fn get_agent_config(role: &str) -> Result<AgentConfig> {
    match role {
        "idea_generator" => Ok(AgentConfig {
            name: "Story Idea Generator",
            system_message: "You are an expert idea generator on a collaborative story-writing \
                             team. Your role is to create a compelling, one-sentence story \
                             concept.",
            task_instruction: "Based on the user's request, generate a single, captivating \
                               sentence that establishes the core conflict or mystery of a story.",
            section_header: "Concept",
            prev_topic: None,
            next_topic: Some("concept_to_plot"),
        }),
        "plot_developer" => Ok(AgentConfig {
            name: "Plot Developer",
            system_message: "You are a master storyteller on a collaborative writing team. Your \
                            role is to expand a story concept into a structured plot outline.",
            task_instruction: "Read the provided story **Concept**. Your task is to write a brief \
                               plot outline with three distinct acts (Act 1: Setup, Act 2: \
                               Confrontation, Act 3: Resolution).",
            section_header: "Plot Outline",
            prev_topic: Some("concept_to_plot"),
            next_topic: Some("plot_to_chars"),
        }),
        "character_creator" => Ok(AgentConfig {
            name: "Character Creator",
            system_message: "You are an expert character designer on a collaborative writing team. \
                             Your role is to create a memorable protagonist and antagonist.",
            task_instruction: "Read the **Concept** and **Plot Outline**. Your task is to create a \
                               one-sentence description for a compelling protagonist and a \
                               formidable antagonist that fit the story.",
            section_header: "Characters",
            prev_topic: Some("plot_to_chars"),
            next_topic: Some("chars_to_dialogue"),
        }),
        "dialogue_writer" => Ok(AgentConfig {
            name: "Dialogue Writer",
            system_message: "You are a skilled dialogue writer on a collaborative writing team. \
                             Your role is to write a key piece of dialogue.",
            task_instruction: "Read all the story elements. Your task is to write a single, \
                               impactful line of dialogue spoken by the protagonist during the \
                               story's climax.",
            section_header: "Climax Dialogue",
            prev_topic: Some("chars_to_dialogue"),
            next_topic: None,
        }),
        _ => Err(format!("Unknown role: {}", role)),
    }
}

/// Read the sampled token off a finalized pass's single-`Token` output tensor.
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

/// Chat-EOS depth-1 pipelined decode over the keep-core carrier, returning the
/// detokenized text. `prompt` is the full prompt tail (the first sampling pass).
/// Speculate the next forward eagerly, roll an over-shot pass back with
/// `carrier::discard_pass` on a stop. See `ptir-pipelined-eos-rollback-spec`.
async fn decode_chat(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    s: &LoweredSampler,
    prompt: &[u32],
    max_tokens: usize,
) -> Result<String> {
    let stop = chat::stop_tokens();
    let mut dec = chat::Decoder::new();
    let mut text = String::new();
    if max_tokens == 0 {
        return Ok(text);
    }
    let prompt = if prompt.is_empty() { &[0u32][..] } else { prompt };
    let mut producer = carrier::submit_pass(kv, seq_len, fresh, s, prompt, true)?;
    let mut generated = 0usize;
    loop {
        let speculate = generated + 1 < max_tokens;
        let consumer = if speculate {
            Some(carrier::submit_pass(kv, seq_len, fresh, s, &[0u32], true)?)
        } else {
            None
        };
        let token = read_token(producer).await?;
        generated += 1;
        let mut done = stop.contains(&token);
        if !done {
            match dec.feed(&[token])? {
                chat::Event::Delta(t) => text.push_str(&t),
                chat::Event::Done(t) => {
                    text = t;
                    done = true;
                }
                _ => {}
            }
        }
        if generated >= max_tokens {
            done = true;
        }
        if done {
            if let Some(c) = consumer {
                carrier::discard_pass(c, seq_len).await;
            }
            break;
        }
        producer = consumer.expect("consumer speculated when not terminal");
    }
    Ok(text)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let my_role = input.role;
    let group_id = input.group_id;
    let tokens_per_step = input.tokens_per_step;

    let config = get_agent_config(&my_role)?;

    let (user_prompt, accumulated_story) = if let Some(prev_topic) = config.prev_topic {
        // Subscribe to the previous agent's topic and wait for a message
        let mut subscription = messaging::subscribe(&format!("{}-{}", prev_topic, group_id));
        let accumulated = subscription.next().await
            .ok_or_else(|| "No message received from previous agent".to_string())?;
        let prompt = format!(
            "**Previous Story Elements:**\n---\n{}\n---\n\n**Your Specific Task:**\n{}",
            accumulated, config.task_instruction
        );
        (prompt, accumulated)
    } else {
        (input.prompt, String::new())
    };

    // Build the prompt on the raw WIT surface: a deferred system folds into the
    // first user turn via `system_user` (mirrors `Context::user`), then cue —
    // the whole prompt is the tail the first decode pass samples from.
    let kv = KvWorkingSet::new();
    let mut seq_len: u32 = 0;
    let mut fresh = true;

    let mut tail = chat::system_user(
        config.system_message,
        &format!(
            "{}\nPlease start with \"### {}\"",
            user_prompt, config.section_header
        ),
    );
    tail.extend(chat::cue());

    let s = sampler::sampler_program(SamplerSpec::Argmax, model::output_vocab_size())?;
    let contribution =
        decode_chat(&kv, &mut seq_len, &mut fresh, &s, &tail, tokens_per_step).await?;

    // Strip any EOS token text from the contribution
    let stop_tokens = inferlet::chat::stop_tokens();
    let stop_text: Vec<String> = stop_tokens
        .iter()
        .filter_map(|&t| inferlet::model::decode(&[t]).ok())
        .collect();
    let contribution: &str = stop_text
        .iter()
        .find_map(|eos| contribution.strip_suffix(eos.as_str()))
        .unwrap_or(&contribution);

    let new_accumulated_story = format!("{}\n{}", accumulated_story, contribution)
        .trim()
        .to_string();

    if let Some(next_topic) = config.next_topic {
        messaging::broadcast(
            &format!("{}-{}", next_topic, group_id),
            &new_accumulated_story,
        );
        println!("Broadcasted story to channel: {}-{}", next_topic, group_id);
    } else {
        println!("Final story:\n{}", new_accumulated_story);
    }

    Ok(String::new())
}
