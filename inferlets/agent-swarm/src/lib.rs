//! Demonstrates a collaborative agent swarm for story writing.
//!
//! This example implements a single agent worker in a multi-agent pipeline
//! where each agent has a specific role (idea generator, plot developer,
//! character creator, or dialogue writer) and passes work to the next agent.

use inferlet::{
    Context, model::Model, runtime,
    messaging, SubscriptionExt,
    Result,
    sample::Sampler,
};
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

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let my_role = input.role;
    let group_id = input.group_id;
    let tokens_per_step = input.tokens_per_step;

    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;
    let tokenizer = model.tokenizer();
    let config = get_agent_config(&my_role)?;

    let (user_prompt, accumulated_story) = if let Some(prev_topic) = config.prev_topic {
        // Subscribe to the previous agent's topic and wait for a message
        let subscription = messaging::subscribe(&format!("{}-{}", prev_topic, group_id));
        let accumulated = subscription.get_async().await
            .ok_or_else(|| "No message received from previous agent".to_string())?;
        let prompt = format!(
            "**Previous Story Elements:**\n---\n{}\n---\n\n**Your Specific Task:**\n{}",
            accumulated, config.task_instruction
        );
        (prompt, accumulated)
    } else {
        (input.prompt, String::new())
    };

    let mut ctx = Context::new(&model)?;
    ctx.system(config.system_message);
    ctx.user(&format!(
        "{}\nPlease start with \"### {}\"",
        user_prompt, config.section_header
    ));
    ctx.cue();

    let contribution = ctx
        .generate(Sampler::Argmax)
        .max_tokens(tokens_per_step)
        .collect_text()
        .await?;

    // Strip any EOS token text from the contribution
    let stop_tokens = inferlet::chat::stop_tokens(&model);
    let stop_text: Vec<String> = stop_tokens
        .iter()
        .filter_map(|&t| tokenizer.decode(&[t]).ok())
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
