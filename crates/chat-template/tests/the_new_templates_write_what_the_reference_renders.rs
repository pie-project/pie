use std::path::PathBuf;
use std::sync::Arc;

use chat_template::{Effort, Instruct, ReasoningEvent};
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
    qwen38_replays_every_earlier_answer_inside_its_thinking_block();
    qwen38_opens_a_thinking_request_with_its_reasoning_effort_instruction();
    qwen38_cues_a_thinking_answer_inside_an_open_think_block();
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
            reasoning_effort: false,
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

/// Qwen3.8's template opens a thinking request's system turn with its
/// reasoning-effort instruction (`xhigh` by default, `low` its own, `medium`
/// none), in a system turn of its own when the request has no system message.
/// The reference ids are `apply_chat_template(..., enable_thinking=True)` over
/// the same snapshot, up to the generation cue.
fn qwen38_opens_a_thinking_request_with_its_reasoning_effort_instruction() {
    let Some(tokenizer) = snapshot(
        "mlx-community/Qwen3.8-27B-4bit",
        "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
    ) else {
        return;
    };
    let config = |reasoning_effort| chat_template::chatml::ChatML {
        thinking: true,
        preserve_thinking: true,
        reasoning_effort,
        tools: true,
        generation_suffix: String::new(),
        stop_tokens: vec!["<|im_end|>".to_string()],
    };
    let qwen38 = chat_template::chatml::ChatMLInstruct::new(tokenizer.clone(), config(true));
    let render =
        |template: &chat_template::chatml::ChatMLInstruct, system: Option<&str>, effort| {
            let mut got = template.system_thinking(system, effort);
            got.extend(template.user(&format!("{MESSAGE}\n")));
            got
        };
    let xhigh: &[u32] = &[
        24342, 286, 4879, 369, 716, 310, 830, 11553, 13, 5044, 1683, 15060, 1472, 279, 3274, 11,
        9307, 1328, 30800, 11, 2814, 47675, 25605, 11, 321, 60445, 55404, 11, 27224, 11, 321,
        30246, 303, 279, 1534, 4087, 13,
    ];
    let low: &[u32] = &[
        24342, 286, 4879, 369, 716, 310, 3238, 13, 13262, 678, 7047, 9522, 321, 10419, 11, 6992,
        5774, 310, 279, 16198, 1973, 24366, 24150, 362, 13,
    ];
    let system: &[u32] = &[248045, 8678, 198];
    let terse: &[u32] = &[2523, 513, 48834, 13];
    let end: &[u32] = &[248046, 198];
    let user: &[u32] = &[
        248045, 846, 198, 3710, 369, 220, 16, 22, 2942, 220, 17, 18, 30, 248046, 198,
    ];
    let gap: &[u32] = &[271];
    assert_eq!(
        render(&qwen38, Some("You are terse."), Effort::Xhigh),
        [system, xhigh, gap, terse, end, user].concat()
    );
    assert_eq!(
        render(&qwen38, None, Effort::Xhigh),
        [system, xhigh, end, user].concat()
    );
    assert_eq!(
        render(&qwen38, Some("You are terse."), Effort::Low),
        [system, low, gap, terse, end, user].concat()
    );
    assert_eq!(
        render(&qwen38, Some("You are terse."), Effort::Medium),
        [system, terse, end, user].concat()
    );
    // A template that states no instruction writes the plain system turn.
    let plain = chat_template::chatml::ChatMLInstruct::new(tokenizer, config(false));
    assert_eq!(
        render(&plain, Some("You are terse."), Effort::Xhigh),
        [system, terse, end, user].concat()
    );
    assert_eq!(render(&plain, None, Effort::Xhigh), user.to_vec());
}

/// Qwen 3.5/3.6/3.8 templates end a thinking request's prompt inside an open
/// `<think>`; the reasoning decoder for that cue starts inside the block, and
/// the one for the no-thinking cue (which closes it) starts outside. The
/// reference ids are the whole `apply_chat_template(..., enable_thinking=True)`.
fn qwen38_cues_a_thinking_answer_inside_an_open_think_block() {
    let Some(tokenizer) = snapshot(
        "mlx-community/Qwen3.8-27B-4bit",
        "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
    ) else {
        return;
    };
    let qwen38 = chat_template::chatml::ChatMLInstruct::new(
        tokenizer.clone(),
        chat_template::chatml::ChatML {
            thinking: true,
            preserve_thinking: true,
            reasoning_effort: true,
            tools: true,
            generation_suffix: "<think>\n".to_string(),
            stop_tokens: vec!["<|im_end|>".to_string()],
        },
    );
    let mut got = qwen38.system_thinking(Some("You are terse."), Effort::Xhigh);
    got.extend(qwen38.user(&format!("{MESSAGE}\n")));
    got.extend(qwen38.cue());
    let want: Vec<u32> = vec![
        248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830, 11553, 13, 5044, 1683, 15060,
        1472, 279, 3274, 11, 9307, 1328, 30800, 11, 2814, 47675, 25605, 11, 321, 60445, 55404, 11,
        27224, 11, 321, 30246, 303, 279, 1534, 4087, 13, 271, 2523, 513, 48834, 13, 248046, 198,
        248045, 846, 198, 3710, 369, 220, 16, 22, 2942, 220, 17, 18, 30, 248046, 198, 248045,
        74455, 198, 248068, 198,
    ];
    assert_eq!(got, want);

    let reasoning = tokenizer.encode("seventeen times twenty-three");
    let close = 248069;
    let mut thinking = qwen38.reasoning_decoder_for_cue(&qwen38.cue());
    assert!(thinking.starts_inside());
    let mut events = thinking.feed(&reasoning);
    events.extend(thinking.feed(&[close]));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ReasoningEvent::Complete(text) if text.trim() == "seventeen times twenty-three")),
        "{events:?}"
    );
    let mut quiet = qwen38.reasoning_decoder_for_cue(&qwen38.cue_without_thinking());
    assert!(!quiet.starts_inside());
    assert!(
        quiet
            .feed(&reasoning)
            .iter()
            .all(|e| matches!(e, ReasoningEvent::Delta(text) if text.is_empty()))
    );
}

/// Qwen3.8's template preserves thinking: every earlier assistant turn is
/// written `<think>\n{reasoning}\n</think>\n\n{answer}`, with an empty block
/// when the turn had no reasoning. The reference ids are
/// `apply_chat_template(..., enable_thinking=False)` over the same snapshot.
fn qwen38_replays_every_earlier_answer_inside_its_thinking_block() {
    let Some(tokenizer) = snapshot(
        "mlx-community/Qwen3.8-27B-4bit",
        "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
    ) else {
        return;
    };
    let qwen38 = chat_template::chatml::ChatMLInstruct::new(
        tokenizer,
        chat_template::chatml::ChatML {
            thinking: true,
            preserve_thinking: true,
            reasoning_effort: true,
            tools: true,
            generation_suffix: String::new(),
            stop_tokens: vec!["<|im_end|>".to_string()],
        },
    );
    let render = |answer: &str| {
        let mut got = qwen38.system_user("You are terse.", &format!("{MESSAGE}\n"));
        got.extend(qwen38.assistant(answer));
        got.extend(qwen38.user("And 18 times 23?"));
        got.extend(qwen38.cue_without_thinking());
        got
    };
    let head: &[u32] = &[
        248045, 8678, 198, 2523, 513, 48834, 13, 248046, 198, 248045, 846, 198, 3710, 369, 220, 16,
        22, 2942, 220, 17, 18, 30, 248046, 198, 248045, 74455, 198, 248068,
    ];
    let tail: &[u32] = &[
        18, 24, 16, 248046, 198, 248045, 846, 198, 2939, 220, 16, 23, 2942, 220, 17, 18, 30,
        248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ];
    let empty: &[u32] = &[271, 248069, 271];
    let reasoned: &[u32] = &[
        198, 16, 22, 9, 17, 18, 369, 220, 18, 24, 16, 198, 248069, 271,
    ];
    assert_eq!(render("391"), [head, empty, tail].concat());
    assert_eq!(
        render("<think>\n17*23 is 391\n</think>\n\n391"),
        [head, reasoned, tail].concat()
    );
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
