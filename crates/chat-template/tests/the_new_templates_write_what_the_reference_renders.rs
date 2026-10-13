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
    minimax_opens_with_the_system_and_developer_turns();
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

/// MiniMax-M3's template opens every prompt with its system turn (identity
/// and adaptive thinking instructions) and a developer turn: the caller's
/// system prompt, or the default identity and the tools.
fn minimax_opens_with_the_system_and_developer_turns() {
    let Some(tokenizer) = snapshot(
        "MiniMaxAI/MiniMax-M3",
        "f0e1c1e04d40177e4673a22097036854f536e9c0",
    ) else {
        return;
    };
    let template = chat_template::minimax::MiniMax::new(tokenizer);
    let opening: Vec<u32> = vec![
        200034, 200019, 28463, 10, 11393, 2428, 4423, 355, 35353, 12973, 5145, 51, 44, 6415, 531,
        35353, 12973, 46, 31058, 70273, 58, 8031, 32, 1421, 54, 46, 106114, 296, 4364, 32, 1421,
        50, 44, 35353, 12973, 355, 258, 4746, 14409, 16001, 2428, 3245, 13323, 301, 47613, 275,
        148531, 300, 14409, 6853, 19960, 73, 634, 60, 85255, 96871, 10353, 1100, 2985, 581, 258,
        6995, 23362, 389, 6369, 390, 301, 3682, 3135, 531, 3135, 1865, 34573, 46, 3437, 6995, 355,
        17239, 44, 18220, 641, 29751, 296, 32, 200059, 200060, 20211, 1865, 641, 4108, 46, 3437,
        6995, 355, 22147, 44, 4236, 641, 4108, 6467, 1619, 275, 32, 200060, 24255, 46, 3437, 6995,
        355, 37760, 44, 10941, 375, 641, 1813, 3784, 301, 1817, 360, 275, 2516, 2906, 320, 14455,
        6995, 6972, 58, 37760, 46, 1781, 457, 21280, 301, 1817, 360, 4794, 5663, 23844, 44, 6775,
        31428, 29751, 44, 436, 994, 32342, 1695, 164445, 3168, 320, 1579, 85255, 96871, 10353, 62,
        200020, 10, 200019, 53556, 10, 2985, 457,
    ];

    let mut got = template.system_user("You are terse.", MESSAGE);
    got.extend(template.cue());
    let tail: Vec<u32> = vec![
        6000, 101, 46, 200020, 10, 200019, 3995, 10, 3376, 355, 32, 1462, 3539, 32, 2019, 63,
        200020, 10, 200019, 1361, 10,
    ];
    assert_eq!(got, [opening.clone(), tail].concat());

    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    let tail: Vec<u32> = vec![
        258, 12473, 23413, 46, 200020, 10, 200019, 3995, 10, 3376, 355, 32, 1462, 3539, 32, 2019,
        63, 200020, 10, 200019, 1361, 10,
    ];
    assert_eq!(got, [opening.clone(), tail].concat());

    let tool = r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}"#;
    let mut got = template.equip(&[tool.to_string()]);
    got.extend(template.user("Weather in Paris?"));
    got.extend(template.assistant(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 2, \"opts\": {\"units\": [\"c\", \"f\"], \"hourly\": true}}}\n</tool_call>",
    ));
    got.extend(template.answer("get_weather", "sunny"));
    got.extend(template.assistant("<mm:think>\nok\n</mm:think>\n\nIt is sunny."));
    got.extend(template.user("Thanks"));
    got.extend(template.cue());
    let tail: Vec<u32> = vec![
        258, 12473, 23413, 634, 35, 31455, 10, 2985, 992, 1728, 841, 436, 812, 6629, 301, 7202,
        418, 275, 3100, 9560, 320, 9806, 457, 275, 6629, 3136, 296, 19535, 19357, 6634, 3728, 60,
        32650, 1100, 60, 45382, 62, 16673, 1906, 2803, 494, 848, 95, 51872, 1164, 494, 16944, 2803,
        494, 66079, 300, 258, 4781, 38298, 494, 24344, 2803, 18396, 4467, 2803, 494, 7052, 1164,
        494, 41187, 2803, 18396, 44044, 2803, 18396, 4467, 2803, 494, 3386, 47488, 494, 29525,
        2803, 18396, 4467, 2803, 494, 37405, 34, 18194, 494, 18915, 2803, 16248, 44044, 8760,
        21384, 45382, 1100, 1579, 32650, 6983, 2455, 1728, 6629, 44, 18220, 679, 1088, 73651, 296,
        258, 3950, 32, 200058, 200052, 200058, 200053, 4733, 46, 44288, 4134, 10490, 44343, 8972,
        436, 34801, 457, 112939, 21166, 1097, 27190, 6387, 46, 18245, 3728, 200058, 200052, 10,
        200058, 60, 88638, 1925, 1139, 45382, 28145, 45, 49, 3361, 200058, 60, 2949, 45, 49, 62,
        3582, 45, 49, 200058, 1579, 2949, 45, 49, 62, 200058, 60, 2949, 45, 50, 62, 200058, 60,
        1896, 62, 200058, 60, 3512, 10383, 62, 1523, 10383, 200058, 1579, 3512, 10383, 62, 200058,
        60, 3512, 1785, 62, 1523, 1785, 200058, 1579, 3512, 1785, 62, 200058, 1579, 1896, 62,
        200058, 1579, 2949, 45, 50, 62, 200058, 1579, 88638, 1100, 200058, 60, 88638, 1925, 1139,
        45382, 28145, 45, 50, 3361, 200058, 60, 2949, 45, 49, 62, 3582, 45, 49, 200058, 1579, 2949,
        45, 49, 62, 200058, 1579, 88638, 1100, 200058, 200053, 200020, 10, 200019, 3995, 10, 66079,
        296, 8261, 63, 200020, 10, 200019, 1361, 10, 200060, 200058, 200052, 10, 200058, 60, 88638,
        1925, 1139, 848, 95, 51872, 3361, 200058, 60, 44044, 62, 44890, 200058, 1579, 44044, 62,
        200058, 60, 29525, 62, 50, 200058, 1579, 29525, 62, 200058, 60, 44410, 62, 200058, 60,
        62204, 62, 200058, 60, 1896, 114317, 200058, 1579, 1896, 62, 200058, 60, 1896, 134435,
        200058, 1579, 1896, 62, 200058, 1579, 62204, 62, 200058, 41298, 456, 368, 115724, 200058,
        1579, 95922, 368, 62, 200058, 1579, 44410, 62, 200058, 1579, 88638, 1100, 200058, 200053,
        200020, 10, 200019, 45382, 10, 60, 9679, 61639, 165476, 1579, 9679, 62, 200020, 10, 200019,
        1361, 10, 200059, 609, 200060, 1838, 355, 63623, 46, 200020, 10, 200019, 3995, 10, 14981,
        200020, 10, 200019, 1361, 10,
    ];
    assert_eq!(got, [opening, tail].concat());
    assert_eq!(template.seal(), vec![200020]);
    assert_eq!(
        template.cue_without_thinking(),
        vec![200019, 1361, 10, 200060]
    );
}
