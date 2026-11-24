use std::ops::ControlFlow;

use compat::generate::{Sampling, generate};
use compat::prompt::Prompt;
use inferlet::{Result, chat, session};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    turns: usize,
    tokens: usize,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let mut sampling = Sampling {
        speculative: false,
        temperature: 0.7,
        top_p: 0.8,
        top_k: 20,
        repetition_penalty: 1.05,
        ..Sampling::default()
    };
    sampling.set_seed(7);
    let mut tokens = chat::system_user("You are terse.", "Count from one.");
    tokens.extend(chat::cue());
    let prompt = Prompt {
        tokens,
        images: Vec::new(),
        ends: Vec::new(),
        system: None,
    };
    for turn in 0..input.turns {
        let mut sampled = 0;
        generate(
            &prompt,
            sampling,
            input.tokens,
            None,
            &mut |_| {},
            &mut |_, _| {
                sampled += 1;
                ControlFlow::Continue(())
            },
        )
        .await?;
        session::send(&format!("{turn} {sampled}"));
        session::receive().await;
    }
    Ok(String::new())
}
