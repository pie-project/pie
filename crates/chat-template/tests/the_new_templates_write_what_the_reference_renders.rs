use std::path::PathBuf;
use std::sync::Arc;

use chat_template::Instruct;
use tokenizer::Tokenizer;

const MESSAGE: &str = "What is 17 times 23?";

fn snapshot(repo: &str, revision: &str) -> Option<Arc<Tokenizer>> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let path = home
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(revision)
        .join("tokenizer.json");
    if !path.is_file() {
        eprintln!("skipping: no tokenizer at {}", path.display());
        return None;
    }
    Some(Arc::new(
        Tokenizer::from_file(&path).expect("the tokenizer loads"),
    ))
}

#[test]
fn the_new_templates_write_what_the_reference_renders_every_case() {
    inkling_writes_the_effort_line_the_user_turn_and_the_cue();
    atem_writes_the_user_turn_and_the_cue_after_the_opening();
    chatml_trims_the_turn_and_closes_an_empty_thinking_block();
    gemma_closes_an_empty_thought_channel_where_its_template_does();
}

/// Qwen3.5's template trims every turn and, with `enable_thinking: false`,
/// cues past an empty thinking block.
fn chatml_trims_the_turn_and_closes_an_empty_thinking_block() {
    let Some(tokenizer) = snapshot(
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
    ) else {
        return;
    };
    let template = chat_template::chatml::ChatMLInstruct::new(
        tokenizer,
        chat_template::chatml::ChatML {
            thinking: true,
            preserve_thinking: false,
            tools: true,
            generation_suffix: String::new(),
            stop_tokens: vec!["<|im_end|>".to_string()],
        },
    );
    let mut got = template.system_user("You are terse.", &format!("{MESSAGE}\n"));
    got.extend(template.cue_without_thinking());
    let want: Vec<u32> = vec![
        248045, 8678, 198, 2523, 513, 48834, 13, 248046, 198, 248045, 846, 198, 3710, 369, 220, 16,
        22, 2942, 220, 17, 18, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ];
    assert_eq!(got, want);
}

/// Gemma 4 26B-A4B's and 31B's templates close an empty thought channel
/// after the model header; E4B's writes the header alone.
fn gemma_closes_an_empty_thought_channel_where_its_template_does() {
    let Some(tokenizer) = snapshot(
        "google/gemma-4-E4B-it",
        "fee6332c1abaafb77f6f9624236c63aa2f1d0187",
    ) else {
        return;
    };
    let turns: Vec<u32> = vec![
        2, 105, 9731, 107, 3048, 659, 17514, 236761, 106, 107, 105, 2364, 107, 3689, 563, 236743,
        236770, 236832, 2782, 236743, 236778, 236800, 236881, 106, 107, 105, 4368, 107,
    ];
    for (thinking, closed) in [(false, vec![]), (true, vec![100, 45518, 107, 101])] {
        let template = chat_template::gemma::Gemma::new(tokenizer.clone(), thinking);
        let mut got = template.system_user("You are terse.", &format!("{MESSAGE}\n"));
        got.extend(template.cue());
        assert_eq!(got, [turns.clone(), closed].concat());
        assert_eq!(template.cue_without_thinking(), template.cue());
    }
}

fn inkling_writes_the_effort_line_the_user_turn_and_the_cue() {
    let Some(tokenizer) = snapshot("thinkingmachines/Inkling", "mini-l7-e8") else {
        return;
    };
    let template = chat_template::inkling::Inkling::new(tokenizer);
    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        200002, 200004, 133850, 6942, 3211, 25, 220, 15, 13, 24, 200010, 200000, 200004, 4827, 382,
        220, 1422, 4238, 220, 1860, 30, 200010, 200001,
    ];
    assert_eq!(got, want);
    assert_eq!(template.seal(), vec![200006, 199999]);
}

fn atem_writes_the_user_turn_and_the_cue_after_the_opening() {
    let Some(tokenizer) = snapshot("meta-models/Muse-Glimmer-30B", "mini-l8-ends") else {
        return;
    };
    let template = chat_template::atem::Atem::new(tokenizer);
    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        200000, 200022, 1556, 200023, 3668, 373, 220, 1087, 4332, 220, 1504, 43, 200008, 200022,
        140680,
    ];
    assert_eq!(got, want);
    assert_eq!(template.prefix(), vec![200000]);
    assert_eq!(template.seal(), vec![200008, 200001]);

    let mut got = template.system_user("You are terse.", MESSAGE);
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        200000, 200022, 15651, 200023, 4662, 583, 260, 6201, 1574, 34956, 300, 9762, 38, 2244,
        1574, 15, 14757, 73965, 38, 392, 2540, 706, 392, 1556, 4205, 200008, 200022, 1556, 200023,
        3668, 373, 220, 1087, 4332, 220, 1504, 43, 200008, 200022, 140680,
    ];
    assert_eq!(got, want);
    let mut got = template.system_user("You are terse.\n\nReasoning strength: low.", MESSAGE);
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        200000, 200022, 15651, 200023, 4662, 583, 260, 6201, 1574, 34956, 300, 9762, 38, 4463,
        1574, 15, 14757, 73965, 38, 392, 2540, 706, 392, 1556, 4205, 200008, 200022, 1556, 200023,
        3668, 373, 220, 1087, 4332, 220, 1504, 43, 200008, 200022, 140680,
    ];
    assert_eq!(got, want);
}
