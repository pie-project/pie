//! Demonstrates a collaborative agent swarm for story writing.
//!
//! This example implements a single agent worker in a multi-agent pipeline
//! where each agent has a specific role (idea generator, plot developer,
//! character creator, or dialogue writer) and passes work to the next agent.

use inferlet::stop_condition::{self, StopCondition};
use inferlet::{Args, Result, Sampler, Tokenizer, anyhow};

const HELP: &str = "\
Usage: agent-swarm <role> [OPTIONS]

A single agent worker for a collaborative story-writing pipeline.

Arguments:
  <role>    The role of the agent. Must be one of:
            - idea_generator
            - plot_developer
            - character_creator
            - dialogue_writer

Options:
  -g, --group-id <ID>         The pipeline/group ID for this agent instance [default: 0]
  -t, --tokens-per-step <N>   Max new tokens this agent should generate [default: 512]
  -p, --prompt <PROMPT>       The prompt to send to the agent
                              [default: \"A story about day dreaming in a park\"]
  -h, --help                  Prints this help message";

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
        _ => Err(anyhow!("Unknown role: {}", role)),
    }
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let my_role: String = args
        .free_from_str()
        .map_err(|_| anyhow!("Missing required <role> argument.\n\n{}", HELP))?;
    let group_id: u32 = args.value_from_str(["-g", "--group-id"]).unwrap_or(0);
    let tokens_per_step: usize = args
        .value_from_str(["-t", "--tokens-per-step"])
        .unwrap_or(512);

    let model = inferlet::get_auto_model();

    if !model.get_name().starts_with("llama-3") {
        return Err(anyhow!(
            "This example works with only non-thinking models. \
            Please use Llama 3 models."
        ));
    }

    let eos_tokens = model.eos_tokens();
    let config = get_agent_config(&my_role)?;

    let (user_prompt, accumulated_story) = if let Some(prev_topic) = config.prev_topic {
        let accumulated = inferlet::subscribe(&format!("{}-{}", prev_topic, group_id)).await;
        let prompt = format!(
            "**Previous Story Elements:**\n---\n{}\n---\n\n**Your Specific Task:**\n{}",
            accumulated, config.task_instruction
        );
        (prompt, accumulated)
    } else {
        let initial_prompt = args
            .value_from_str(["-p", "--prompt"])
            .unwrap_or("A story about day dreaming in a park".to_string());
        (initial_prompt, String::new())
    };

    let mut ctx = model.create_context();
    ctx.fill_system(config.system_message);
    ctx.fill_user(&format!(
        "{}\nPlease start with \"### {}\"",
        user_prompt, config.section_header
    ));

    let stop_condition = stop_condition::max_len(tokens_per_step)
        .or(stop_condition::ends_with_any(eos_tokens.clone()));
    let contribution = ctx.generate(Sampler::greedy(), stop_condition).await;

    // Detokenize EOS tokens to get their string forms
    let tokenizer = Tokenizer::new(&model);
    let eos_strings: Vec<String> = eos_tokens
        .iter()
        .map(|tokens| tokenizer.detokenize(tokens))
        .collect();

    // Strip the special ending suffix from the generated string if one exists
    let contribution: &str = eos_strings
        .iter()
        .find_map(|eos| contribution.strip_suffix(eos))
        .unwrap_or(&contribution);

    let new_accumulated_story = format!("{}\n{}", accumulated_story, contribution)
        .trim()
        .to_string();

    if let Some(next_topic) = config.next_topic {
        inferlet::broadcast(
            &format!("{}-{}", next_topic, group_id),
            &new_accumulated_story,
        );
        println!("Broadcasted story to channel: {}-{}", next_topic, group_id);
    } else {
        println!("Final story:\n{}", new_accumulated_story);
    }

    Ok(())
}
