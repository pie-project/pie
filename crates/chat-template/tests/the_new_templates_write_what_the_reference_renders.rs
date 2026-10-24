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
    qwen38_writes_its_tools_calls_and_results_as_its_template_does();
    gemma_closes_an_empty_thought_channel_where_its_template_does();
    minimax_opens_with_the_system_and_developer_turns();
    hy_writes_the_reasoning_mode_the_turns_the_calls_and_the_cue();
    dots_opens_on_the_default_system_turn_and_answers_a_tool_in_a_user_turn();
    mimo_writes_its_turns_untrimmed_and_its_tools_a_turn_ahead();
    llama_opens_on_its_dated_system_turn_and_writes_its_calls_as_json();
    glm47_writes_its_turns_back_to_back_and_its_calls_in_xml();
    glm45_writes_a_newline_after_each_marker();
    minimax2_opens_on_its_identity_and_cues_an_open_think_block();
    qwen3_coder_lists_its_tools_as_xml_and_writes_its_calls_as_functions();
}

/// Qwen3-Coder's template: plain ChatML turns, the tools a system turn of
/// their own on the coder's identity, calls as `<function=…>` blocks,
/// results in a user turn.
fn qwen3_coder_lists_its_tools_as_xml_and_writes_its_calls_as_functions() {
    let Some(tokenizer) = snapshot(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
    ) else {
        return;
    };
    let template = chat_template::qwen3_coder::Qwen3Coder::try_new(tokenizer)
        .expect("the tokenizer has the template's markers");
    let question: Vec<u32> = vec![
        151644, 872, 198, 3838, 374, 220, 16, 22, 3039, 220, 17, 18, 30, 151645, 198,
    ];
    let cue: Vec<u32> = vec![151644, 77091, 198];

    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    assert_eq!(got, [question.clone(), cue.clone()].concat());

    let mut got = template.system_user("You are terse.", MESSAGE);
    got.extend(template.cue());
    let terse: Vec<u32> = vec![151644, 8948, 198, 2610, 525, 50537, 13, 151645, 198];
    assert_eq!(got, [terse, question.clone(), cue.clone()].concat());

    let mut got = template.first_user(MESSAGE);
    got.extend(template.assistant("391"));
    got.extend(template.user("And 18 times 23?"));
    got.extend(template.cue());
    let want = [
        question,
        cue.clone(),
        vec![18, 24, 16, 151645, 198],
        vec![
            151644, 872, 198, 3036, 220, 16, 23, 3039, 220, 17, 18, 30, 151645, 198,
        ],
        cue,
    ]
    .concat();
    assert_eq!(got, want);

    let mut got = template.equip(&[GLM_TOOL.to_string()]);
    got.extend(template.user("Weather in Paris?"));
    got.extend(template.assistant(GLM_CALL));
    got.extend(template.answer("get_weather", "{\"temp\": 21}"));
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 264, 10950, 15235, 17847, 429, 646, 16282,
        448, 264, 6366, 311, 11625, 9079, 382, 2, 13852, 271, 2610, 614, 2615, 311, 279, 2701,
        5746, 1447, 27, 15918, 397, 27, 1688, 397, 27, 606, 29, 455, 69364, 522, 606, 397, 27,
        4684, 29, 28981, 522, 4684, 397, 27, 13786, 397, 27, 16181, 397, 27, 606, 29, 8926, 522,
        606, 397, 27, 1313, 29, 917, 522, 1313, 397, 522, 16181, 397, 522, 13786, 397, 522, 1688,
        397, 522, 15918, 1339, 2679, 498, 5157, 311, 1618, 264, 729, 26687, 9851, 304, 279, 2701,
        3561, 448, 5664, 20525, 1447, 151657, 198, 27, 1688, 28, 8687, 9174, 1269, 397, 27, 16181,
        28, 8687, 24899, 62, 16, 397, 957, 62, 16, 198, 522, 16181, 397, 27, 16181, 28, 8687,
        24899, 62, 17, 397, 1986, 374, 279, 897, 369, 279, 2086, 5733, 198, 9033, 646, 9390, 198,
        35673, 5128, 198, 522, 16181, 397, 522, 1688, 397, 151658, 271, 27, 98743, 397, 95359, 510,
        12, 5712, 6738, 27732, 1795, 279, 5189, 3561, 25, 458, 9179, 366, 1688, 28, 1112, 1472,
        1688, 29, 2504, 1969, 387, 24034, 2878, 220, 151657, 151658, 11874, 9492, 198, 12, 12659,
        5029, 27732, 387, 5189, 198, 12, 1446, 1231, 3410, 10101, 32711, 369, 697, 729, 1618, 304,
        5810, 4128, 56482, 279, 729, 1618, 11, 714, 4183, 1283, 198, 12, 1416, 1052, 374, 902, 729,
        1618, 2500, 11, 4226, 279, 3405, 1075, 4622, 448, 697, 1482, 6540, 323, 653, 537, 3291,
        279, 1196, 911, 729, 6738, 198, 522, 98743, 29, 151645, 198, 151644, 872, 198, 28981, 304,
        12095, 30, 151645, 198, 151644, 77091, 198, 151657, 198, 27, 1688, 28280, 69364, 397, 27,
        16181, 28, 8926, 397, 59604, 198, 522, 16181, 397, 27, 16181, 28, 13778, 397, 17, 198, 522,
        16181, 397, 522, 1688, 397, 151658, 151645, 198, 151644, 872, 198, 151665, 198, 4913, 3888,
        788, 220, 17, 16, 532, 151666, 198, 151645, 198, 151644, 77091, 198,
    ];
    assert_eq!(got, want);
    assert_eq!(template.seal(), vec![151645, 151643]);
}

/// MiniMax-M2.5's template: a system turn of the release's identity (or the
/// caller's prompt, or the identity and the tools), a cue inside an open
/// `<think>` block, calls as `<invoke>`s, results as `<response>`s.
fn minimax2_opens_on_its_identity_and_cues_an_open_think_block() {
    let Some(tokenizer) = snapshot(
        "MiniMaxAI/MiniMax-M2.5",
        "f710177d938eff80b684d42c5aa84b382612f21f",
    ) else {
        return;
    };
    let template = chat_template::minimax2::MiniMax2::try_new(
        tokenizer,
        "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.",
    )
    .expect("the tokenizer has the template's markers");
    let opening: Vec<u32> = vec![
        200034, 200019, 28463, 10, 2985, 457, 258, 12473, 23413, 46, 5324, 1925, 355, 35353, 12973,
        5145, 50, 46, 53, 306, 355, 6904, 531, 35353, 12973, 46, 200020, 10,
    ];
    let question: Vec<u32> = vec![
        200019, 3995, 10, 3376, 355, 32, 1462, 3539, 32, 2019, 63, 200020, 10,
    ];
    let cue: Vec<u32> = vec![200019, 1361, 10, 200050, 10];

    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    assert_eq!(
        got,
        [opening.clone(), question.clone(), cue.clone()].concat()
    );

    let mut got = template.system_user("You are terse.", MESSAGE);
    got.extend(template.cue());
    let terse: Vec<u32> = vec![
        200034, 200019, 28463, 10, 2985, 457, 6000, 101, 46, 200020, 10,
    ];
    assert_eq!(got, [terse, question.clone(), cue.clone()].concat());

    let mut got = template.first_user(MESSAGE);
    got.extend(template.assistant("<think>\n17*23\n</think>\n\n391"));
    got.extend(template.user("And 18 times 23?"));
    got.extend(template.cue());
    let want = [
        opening,
        question,
        vec![
            200019, 1361, 10, 42131, 200020, 10, 200019, 3995, 10, 3622, 32, 1081, 3539, 32, 2019,
            63, 200020, 10,
        ],
        cue,
    ]
    .concat();
    assert_eq!(got, want);

    let mut got = template.equip(&[GLM_TOOL.to_string()]);
    got.extend(template.user("Weather in Paris?"));
    got.extend(template.assistant(GLM_CALL));
    got.extend(template.answer("get_weather", "{\"temp\": 21}"));
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        200034, 200019, 28463, 10, 2985, 457, 258, 12473, 23413, 46, 5324, 1925, 355, 35353, 12973,
        5145, 50, 46, 53, 306, 355, 6904, 531, 35353, 12973, 634, 35, 31455, 10, 2985, 992, 1728,
        841, 436, 812, 6629, 301, 7202, 418, 275, 3100, 9560, 320, 9806, 457, 275, 6629, 3136, 296,
        19535, 19357, 6634, 3728, 60, 32650, 1100, 60, 45382, 62, 16673, 1906, 2803, 494, 848, 95,
        51872, 1164, 494, 16944, 2803, 494, 66079, 1164, 494, 24344, 2803, 18396, 4467, 2803, 494,
        7052, 1164, 494, 41187, 2803, 18396, 44044, 2803, 18396, 4467, 2803, 494, 3386, 150370,
        21384, 45382, 1100, 1579, 32650, 6983, 4625, 3527, 3994, 9885, 44, 1223, 27190, 6634, 301,
        48687, 6629, 306, 1916, 7729, 3728, 200052, 10, 60, 88638, 1925, 1139, 45382, 28145, 45,
        49, 6143, 60, 24662, 1925, 1139, 2949, 38158, 45, 49, 3361, 2949, 27306, 45, 49, 1579,
        24662, 1100, 60, 24662, 1925, 1139, 2949, 38158, 45, 50, 3361, 2949, 27306, 45, 50, 1579,
        24662, 1100, 4563, 1579, 88638, 1100, 200053, 200020, 10, 200019, 3995, 10, 66079, 296,
        8261, 63, 200020, 10, 200019, 1361, 367, 200052, 10, 60, 88638, 1925, 1139, 848, 95, 51872,
        6143, 60, 24662, 1925, 1139, 44044, 3361, 44890, 1579, 24662, 1100, 60, 24662, 1925, 1139,
        29525, 3361, 50, 1579, 24662, 1100, 1579, 88638, 1100, 200053, 200020, 10, 200019, 45382,
        10, 60, 9679, 62, 16673, 40886, 2803, 32, 2254, 30085, 9679, 62, 200020, 10, 200019, 1361,
        10, 200050, 10,
    ];
    assert_eq!(got, want);
    assert_eq!(template.seal(), vec![200020]);
}

const GLM_TOOL: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}"#;
const GLM_CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 2}}\n</tool_call>";

/// GLM-4.7's template: no effort line, turns back to back, a past answer
/// behind a bare `</think>`, calls in its XML, results in an observation.
fn glm47_writes_its_turns_back_to_back_and_its_calls_in_xml() {
    let Some(tokenizer) = snapshot(
        "zai-org/GLM-4.7",
        "602d01efcdd332c5238ca4bcede555defbe83eb7",
    ) else {
        return;
    };
    let template =
        chat_template::glm::Glm::try_with_style(tokenizer, chat_template::glm::Style::Glm47)
            .expect("the tokenizer has the template's markers");
    let question: Vec<u32> = vec![151336, 3838, 374, 220, 99419, 3039, 220, 99619, 30];

    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    assert_eq!(
        got,
        [vec![151331, 151333], question.clone(), vec![151337, 151350]].concat()
    );
    assert_eq!(template.cue_without_thinking(), vec![151337, 151351]);

    let mut got = template.system_user("You are terse.", MESSAGE);
    got.extend(template.cue());
    let want = [
        vec![151331, 151333, 151335, 2610, 525, 50205, 13],
        question.clone(),
        vec![151337, 151350],
    ]
    .concat();
    assert_eq!(got, want);

    let mut got = template.first_user(MESSAGE);
    got.extend(template.assistant("<think>17*23</think>391"));
    got.extend(template.user("And 18 times 23?"));
    got.extend(template.cue());
    let want = [
        vec![151331, 151333],
        question,
        vec![
            151337, 151351, 101294, 16, 151336, 3036, 220, 99243, 3039, 220, 99619, 30, 151337,
            151350,
        ],
    ]
    .concat();
    assert_eq!(got, want);

    let mut got = template.equip(&[GLM_TOOL.to_string()]);
    got.extend(template.user("Weather in Paris?"));
    got.extend(template.assistant(GLM_CALL));
    got.extend(template.answer("get_weather", "{\"temp\": 21}"));
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        151331, 151333, 151335, 198, 2, 13843, 271, 2610, 1231, 1618, 825, 476, 803, 5746, 311,
        7789, 448, 279, 1196, 3239, 382, 2610, 525, 3897, 448, 729, 32479, 2878, 366, 15906, 1472,
        15906, 29, 11868, 9488, 510, 27, 15906, 397, 4913, 1313, 788, 330, 1688, 497, 330, 1688,
        788, 5212, 606, 788, 330, 455, 68852, 497, 330, 4684, 788, 330, 28876, 497, 330, 13777,
        788, 5212, 1313, 788, 330, 1700, 497, 330, 13185, 788, 5212, 8923, 788, 5212, 1313, 788,
        330, 917, 30849, 3417, 532, 522, 15906, 1339, 2461, 1817, 729, 1618, 11, 2550, 279, 729,
        829, 323, 5977, 2878, 279, 2701, 11868, 3561, 510, 151352, 90, 1688, 11489, 92, 151356, 90,
        858, 16159, 12, 16, 92, 151357, 151358, 90, 858, 19048, 12, 16, 92, 151359, 151356, 90,
        858, 16159, 12, 17, 92, 151357, 151358, 90, 858, 19048, 12, 17, 92, 151359, 1112, 151353,
        151336, 28876, 304, 12089, 30, 151337, 151351, 151352, 455, 68852, 151356, 8923, 151357,
        151358, 59190, 151359, 151356, 13769, 151357, 151358, 17, 151359, 151353, 151338, 151354,
        4913, 3888, 788, 220, 99146, 92, 151355, 151337, 151350,
    ];
    assert_eq!(got, want);
}

/// GLM-4.5 / 4.6's template: a newline after each marker, the cue left for
/// the model to open its own thinking block, an empty block for a turn that
/// does not think.
fn glm45_writes_a_newline_after_each_marker() {
    let Some(tokenizer) = snapshot(
        "zai-org/GLM-4.5",
        "cbb2c7cfb52fa128a9660cb1a7a78e017899e115",
    ) else {
        return;
    };
    let template =
        chat_template::glm::Glm::try_with_style(tokenizer, chat_template::glm::Style::Glm45)
            .expect("the tokenizer has the template's markers");
    let question: Vec<u32> = vec![151336, 198, 3838, 374, 220, 99419, 3039, 220, 99619, 30];

    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    assert_eq!(
        got,
        [vec![151331, 151333], question.clone(), vec![151337]].concat()
    );
    assert_eq!(
        template.cue_without_thinking(),
        vec![151337, 198, 151350, 151351]
    );

    let mut got = template.system_user("You are terse.", MESSAGE);
    got.extend(template.cue());
    let want = [
        vec![151331, 151333, 151335, 198, 2610, 525, 50205, 13],
        question.clone(),
        vec![151337],
    ]
    .concat();
    assert_eq!(got, want);

    let mut got = template.first_user(MESSAGE);
    got.extend(template.assistant("391"));
    got.extend(template.user("And 18 times 23?"));
    got.extend(template.cue());
    let want = [
        vec![151331, 151333],
        question,
        vec![
            151337, 198, 151350, 151351, 198, 101294, 16, 151336, 198, 3036, 220, 99243, 3039, 220,
            99619, 30, 151337,
        ],
    ]
    .concat();
    assert_eq!(got, want);

    let mut got = template.equip(&[GLM_TOOL.to_string()]);
    got.extend(template.user("Weather in Paris?"));
    got.extend(template.assistant(GLM_CALL));
    got.extend(template.answer("get_weather", "{\"temp\": 21}"));
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        151331, 151333, 151335, 198, 2, 13843, 271, 2610, 1231, 1618, 825, 476, 803, 5746, 311,
        7789, 448, 279, 1196, 3239, 382, 2610, 525, 3897, 448, 729, 32479, 2878, 366, 15906, 1472,
        15906, 29, 11868, 9488, 510, 27, 15906, 397, 4913, 1313, 788, 330, 1688, 497, 330, 1688,
        788, 5212, 606, 788, 330, 455, 68852, 497, 330, 4684, 788, 330, 28876, 497, 330, 13777,
        788, 5212, 1313, 788, 330, 1700, 497, 330, 13185, 788, 5212, 8923, 788, 5212, 1313, 788,
        330, 917, 30849, 3417, 532, 522, 15906, 1339, 2461, 1817, 729, 1618, 11, 2550, 279, 729,
        829, 323, 5977, 2878, 279, 2701, 11868, 3561, 510, 151352, 90, 1688, 11489, 532, 151356,
        90, 858, 16159, 12, 16, 92, 151357, 198, 151358, 90, 858, 19048, 12, 16, 92, 151359, 198,
        151356, 90, 858, 16159, 12, 17, 92, 151357, 198, 151358, 90, 858, 19048, 12, 17, 92,
        151359, 198, 9335, 151353, 151336, 198, 28876, 304, 12089, 30, 151337, 198, 151350, 151351,
        198, 151352, 455, 68852, 198, 151356, 8923, 151357, 198, 151358, 59190, 151359, 198,
        151356, 13769, 151357, 198, 151358, 17, 151359, 198, 151353, 151338, 198, 151354, 198,
        4913, 3888, 788, 220, 99146, 532, 151355, 151337,
    ];
    assert_eq!(got, want);
}

/// Llama 3.1's template always opens on a system turn carrying its dates,
/// lists the tools in that turn under `Environment: ipython` (with
/// `tools_in_user_message: false`), writes a call as its JSON and hands the
/// result back quoted in an `ipython` turn.
fn llama_opens_on_its_dated_system_turn_and_writes_its_calls_as_json() {
    let Some(tokenizer) = snapshot(
        "unsloth/Llama-3.1-8B-Instruct",
        "4699cc75b550f9c6f3173fb80f4703b62d946aa5",
    ) else {
        return;
    };
    let template = chat_template::llama::Llama::try_new(tokenizer)
        .expect("the tokenizer has the template's markers");
    let dates: Vec<u32> = vec![
        38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263,
        220, 2366, 19, 271,
    ];
    let system = |body: &[u32]| {
        [
            vec![128000, 128006, 9125, 128007, 271],
            dates.clone(),
            body.to_vec(),
            vec![128009],
        ]
        .concat()
    };
    let question: Vec<u32> = vec![
        128006, 882, 128007, 271, 3923, 374, 220, 1114, 3115, 220, 1419, 30, 128009,
    ];
    let cue: Vec<u32> = vec![128006, 78191, 128007, 271];

    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    assert_eq!(got, [system(&[]), question.clone(), cue.clone()].concat());

    let mut got = template.system_user("You are terse.", MESSAGE);
    got.extend(template.cue());
    let terse = system(&[2675, 527, 51637, 13]);
    assert_eq!(got, [terse, question.clone(), cue.clone()].concat());

    let mut got = template.first_user(MESSAGE);
    got.extend(template.assistant("391"));
    got.extend(template.user("And 18 times 23?"));
    got.extend(template.cue());
    let want = [
        system(&[]),
        question,
        cue.clone(),
        vec![19631, 128009],
        vec![
            128006, 882, 128007, 271, 3112, 220, 972, 3115, 220, 1419, 30, 128009,
        ],
        cue.clone(),
    ]
    .concat();
    assert_eq!(got, want);

    let tool = r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}"#;
    let mut got = template.equip(&[tool.to_string()]);
    got.extend(template.user("Weather in Paris?"));
    got.extend(template.assistant(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
    ));
    got.extend(template.answer("get_weather", "{\"temp\": 21}"));
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        128000, 128006, 9125, 128007, 271, 13013, 25, 6125, 27993, 198, 38766, 1303, 33025, 2696,
        25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675,
        617, 2680, 311, 279, 2768, 5865, 13, 2057, 1650, 264, 734, 11, 4587, 6013, 449, 4823, 369,
        264, 734, 1650, 8542, 3595, 304, 279, 3645, 5324, 609, 794, 734, 836, 11, 330, 14105, 794,
        11240, 315, 5811, 836, 323, 1202, 907, 7966, 5519, 539, 1005, 7482, 382, 517, 262, 330,
        1337, 794, 330, 1723, 761, 262, 330, 1723, 794, 341, 286, 330, 609, 794, 330, 456, 70464,
        761, 286, 330, 4789, 794, 330, 30081, 761, 286, 330, 14105, 794, 341, 310, 330, 1337, 794,
        330, 1735, 761, 310, 330, 13495, 794, 341, 394, 330, 9103, 794, 341, 504, 330, 1337, 794,
        330, 928, 702, 394, 457, 310, 457, 286, 457, 262, 457, 633, 128009, 128006, 882, 128007,
        271, 30081, 304, 12366, 30, 128009, 128006, 78191, 128007, 271, 5018, 609, 794, 330, 456,
        70464, 498, 330, 14105, 794, 5324, 9103, 794, 330, 60704, 32075, 128009, 128006, 23799,
        4690, 128007, 271, 14461, 2153, 3975, 11955, 220, 1691, 10064, 128009, 128006, 78191,
        128007, 271,
    ];
    assert_eq!(got, want);
    assert_eq!(template.seal(), vec![128009, 128008, 128001]);
}

/// Dots3's template opens a conversation with no system turn on its default
/// one, writes no newline between turns, and hands a tool's result back in a
/// user turn. With `enable_thinking: false` it cues past an empty thinking
/// block (and ends the user's text with `<no_think>`, written here by hand).
fn dots_opens_on_the_default_system_turn_and_answers_a_tool_in_a_user_turn() {
    let Some(tokenizer) = snapshot("dots-studio/dots3-note-prev", "mini-l6-e16") else {
        return;
    };
    let template = chat_template::dots::Dots::try_new(tokenizer)
        .expect("the tokenizer has the template's markers");
    let question: Vec<u32> = vec![151665, 3838, 374, 220, 16, 22, 3039, 220, 17, 18, 30];
    let default_system: Vec<u32> = vec![151650, 2610, 525, 264, 10950, 17847, 13, 151651];
    let terse: Vec<u32> = vec![151650, 2610, 525, 50537, 13, 151651];

    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    assert_eq!(
        got,
        [
            default_system.clone(),
            question.clone(),
            vec![151666, 151667]
        ]
        .concat()
    );

    let mut got = template.system_user("You are terse.", &format!("{MESSAGE}<no_think>"));
    got.extend(template.cue_without_thinking());
    let want = [
        terse,
        question.clone(),
        vec![151723, 151666, 151667, 151721, 271, 151722, 271],
    ]
    .concat();
    assert_eq!(got, want);

    let mut got = template.first_user(MESSAGE);
    got.extend(template.assistant("Let me check."));
    got.extend(template.answer("multiply", "391"));
    got.extend(template.cue());
    let want = [
        default_system,
        question,
        vec![151666, 151667, 10061, 752, 1779, 13, 151668],
        vec![
            151665, 198, 151726, 198, 18, 24, 16, 198, 151727, 151666, 151667,
        ],
    ]
    .concat();
    assert_eq!(got, want);
    assert_eq!(template.seal(), vec![151643, 151668]);
}

/// Hy4's template opens with the reasoning mode, writes a past turn's
/// thinking block empty, lists the tools in the leading system turn, and
/// cues an opened thinking block, or a closed one under `no_think`.
fn hy_writes_the_reasoning_mode_the_turns_the_calls_and_the_cue() {
    let Some(tokenizer) = snapshot(
        "tencent/Hy4-preview",
        "705d81ee51566a186d645b74c974d642ef2828fe",
    ) else {
        return;
    };
    let template = chat_template::hy::Hy::try_new(tokenizer.clone(), true)
        .expect("the tokenizer has the template's markers");
    let quiet = chat_template::hy::Hy::try_new(tokenizer, false)
        .expect("the tokenizer has the template's markers");
    let high: Vec<u32> = vec![120039, 59785, 287, 62, 4138, 506, 25, 19369, 120025];
    let no_think: Vec<u32> = vec![
        120039, 59785, 287, 62, 4138, 506, 25, 3202, 32980, 1121, 120025,
    ];
    let user: Vec<u32> = vec![
        120000, 3717, 120001, 3462, 341, 220, 1162, 3697, 220, 1498, 30, 120025,
    ];
    let answered: Vec<u32> = vec![
        120000, 611, 10372, 120001, 120029, 120030, 29206, 120025, 120000, 3717, 120001, 4231, 220,
        1004, 3697, 220, 1498, 30, 120025,
    ];
    let cue: Vec<u32> = vec![120000, 611, 10372, 120001, 120029];
    let closed: Vec<u32> = [cue.clone(), vec![120030]].concat();
    let opening = |mode: &[u32]| [vec![120000, 13251, 120001], mode.to_vec()].concat();

    let mut got = template.first_user(MESSAGE);
    got.extend(template.cue());
    assert_eq!(got, [opening(&high), user.clone(), cue.clone()].concat());

    let mut got = template.system_user("You are terse.", MESSAGE);
    got.extend(template.cue());
    let system: Vec<u32> = vec![120000, 13251, 120001, 3681, 488, 257, 8940, 13];
    assert_eq!(
        got,
        [system, high.clone(), user.clone(), cue.clone()].concat()
    );

    let mut got = template.first_user(MESSAGE);
    got.extend(template.assistant("391"));
    got.extend(template.user("And 18 times 23?"));
    got.extend(template.cue());
    let want = [opening(&high), user.clone(), answered.clone(), cue.clone()].concat();
    assert_eq!(got, want);

    let mut got = quiet.first_user(MESSAGE);
    got.extend(quiet.assistant("<think:opensource>17*23 = 391.</think:opensource>391"));
    got.extend(quiet.user("And 18 times 23?"));
    got.extend(quiet.cue());
    let want = [opening(&no_think), user.clone(), answered, closed.clone()].concat();
    assert_eq!(got, want);
    assert_eq!(template.cue_without_thinking(), closed);

    let tools = [r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city, in °C.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}"#.to_string()];
    let mut got = template.equip(&tools);
    got.extend(template.user("Weather in Paris for 2 days?"));
    got.extend(template.assistant(
        "<think:opensource>Call the tool.</think:opensource><tool_call>\n\
         {\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\",\"days\":2}}\n</tool_call>",
    ));
    got.extend(template.answer("get_weather", "{\"temp\": 21}"));
    got.extend(template.cue());
    let want: Vec<u32> = vec![
        120000, 13251, 120001, 2, 21650, 268, 3681, 1296, 1971, 936, 474, 984, 4405, 302, 12824,
        430, 269, 2122, 7343, 364, 3681, 488, 4308, 430, 1477, 41207, 2742, 676, 17736, 1569,
        17736, 29, 14816, 17017, 926, 27, 17736, 332, 16249, 1859, 2486, 549, 998, 62, 49005, 1453,
        549, 18264, 2486, 549, 59185, 373, 259, 7370, 11, 297, 11184, 34, 50150, 549, 28125, 2486,
        19634, 5298, 2486, 549, 6045, 1453, 549, 37488, 2486, 19634, 38090, 2486, 19634, 5298,
        2486, 549, 3828, 44156, 549, 30405, 2486, 19634, 5298, 2486, 549, 28751, 1, 21115, 549,
        21062, 2486, 17044, 38090, 7081, 16472, 358, 17736, 4079, 2943, 1477, 1971, 8344, 11, 435,
        1595, 1419, 3329, 220, 120035, 198, 2943, 1630, 1477, 1971, 11, 435, 1595, 1315, 2341,
        1322, 926, 120033, 90, 6703, 18826, 92, 120040, 90, 1692, 26105, 12, 16, 92, 120041,
        120042, 90, 1692, 18177, 12, 16, 92, 120043, 120040, 90, 1692, 26105, 12, 17, 92, 120041,
        120042, 90, 1692, 18177, 12, 17, 92, 120043, 1605, 120034, 198, 4428, 269, 1754, 299, 1477,
        1971, 8344, 11, 435, 1595, 3329, 220, 120036, 120039, 59785, 287, 62, 4138, 506, 25, 19369,
        120025, 120000, 3717, 120001, 59185, 297, 15641, 373, 220, 17, 4341, 30, 120025, 120000,
        611, 10372, 120001, 120029, 10523, 269, 5509, 13, 120030, 120035, 120033, 998, 62, 49005,
        120040, 38090, 120041, 120042, 65431, 120043, 120040, 30405, 120041, 120042, 17, 120043,
        120034, 120036, 120025, 120000, 23441, 120001, 120037, 16249, 22093, 2486, 220, 1723, 92,
        120038, 120025, 120000, 611, 10372, 120001, 120029,
    ];
    assert_eq!(got, want);
    assert_eq!(template.seal(), vec![120025]);
}

/// MiMo-V2.6's template closes a turn on a bare `<|im_end|>`, writes it
/// untrimmed, states the tools in a system turn of their own, and cues past
/// an empty thinking block for `enable_thinking: false`.
fn mimo_writes_its_turns_untrimmed_and_its_tools_a_turn_ahead() {
    let Some(tokenizer) = snapshot("XiaomiMiMo/MiMo-V2.6-Flash-RL", "mini-l6-e8") else {
        return;
    };
    let template = chat_template::mimo::Mimo::try_new(tokenizer)
        .expect("the tokenizer has the template's markers");
    let turns: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 50537, 13, 151645, 151644, 872, 198, 3838, 374, 220, 16, 22,
        3039, 220, 17, 18, 5267, 151645,
    ];
    let cue: Vec<u32> = vec![151644, 77091, 198];
    let mut got = template.system_user("You are terse.", &format!("{MESSAGE}\n"));
    got.extend(template.cue());
    assert_eq!(got, [turns, cue.clone()].concat());
    assert_eq!(
        template.cue_without_thinking(),
        [cue, vec![151667, 151668]].concat()
    );

    let tool = r#"{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}"#;
    let tools: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 3897, 448, 279, 2701, 7375, 1447, 27, 15918, 397, 4913, 1313,
        788, 330, 1688, 497, 330, 1688, 788, 5212, 606, 788, 330, 455, 69364, 497, 330, 4684, 788,
        330, 28981, 497, 330, 13786, 788, 5212, 1313, 788, 330, 1700, 497, 330, 13193, 788, 5212,
        8926, 788, 5212, 1313, 788, 330, 917, 30975, 3417, 532, 522, 15918, 29, 151645,
    ];
    assert_eq!(template.equip(&[tool.to_string()]), tools);
    assert_eq!(
        template.answer("get_weather", "sunny"),
        vec![151644, 14172, 198, 82, 27297, 151645]
    );

    let mut got = template.assistant("391");
    got.extend(template.user("And 18?"));
    let want: Vec<u32> = vec![
        151644, 77091, 198, 151667, 151668, 18, 24, 16, 151645, 151644, 872, 198, 3036, 220, 16,
        23, 30, 151645,
    ];
    assert_eq!(got, want);
    assert_eq!(template.seal(), vec![151645, 151643]);
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
    let template = chat_template::chatml::ChatMLInstruct::try_new(
        tokenizer,
        chat_template::chatml::ChatML {
            thinking: true,
            preserve_thinking: false,
            reasoning_effort: false,
            tools: true,
            xml_tools: false,
            generation_suffix: String::new(),
            stop_tokens: vec!["<|im_end|>".to_string()],
        },
    )
    .expect("the tokenizer has the template's markers");
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
        xml_tools: false,
        generation_suffix: String::new(),
        stop_tokens: vec!["<|im_end|>".to_string()],
    };
    let qwen38 = chat_template::chatml::ChatMLInstruct::try_new(tokenizer.clone(), config(true))
        .expect("the tokenizer has the template's markers");
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
    let plain = chat_template::chatml::ChatMLInstruct::try_new(tokenizer, config(false))
        .expect("the tokenizer has the template's markers");
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
    let qwen38 = chat_template::chatml::ChatMLInstruct::try_new(
        tokenizer.clone(),
        chat_template::chatml::ChatML {
            thinking: true,
            preserve_thinking: true,
            reasoning_effort: true,
            tools: true,
            xml_tools: false,
            generation_suffix: "<think>\n".to_string(),
            stop_tokens: vec!["<|im_end|>".to_string()],
        },
    )
    .expect("the tokenizer has the template's markers");
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
    let qwen38 = chat_template::chatml::ChatMLInstruct::try_new(
        tokenizer,
        chat_template::chatml::ChatML {
            thinking: true,
            preserve_thinking: true,
            reasoning_effort: true,
            tools: true,
            xml_tools: false,
            generation_suffix: String::new(),
            stop_tokens: vec!["<|im_end|>".to_string()],
        },
    )
    .expect("the tokenizer has the template's markers");
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
        let template = chat_template::gemma::Gemma::try_new(tokenizer.clone(), thinking)
            .expect("the tokenizer has the template's markers");
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
    let template = chat_template::inkling::Inkling::try_new(tokenizer)
        .expect("the tokenizer has the template's markers");
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
    let template = chat_template::atem::Atem::try_new(tokenizer)
        .expect("the tokenizer has the template's markers");
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
    let template = chat_template::minimax::MiniMax::try_new(tokenizer)
        .expect("the tokenizer has the template's markers");
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

/// Qwen3.8's template writes the tools into its one system turn (behind the
/// reasoning-effort instruction, ahead of the system message, each tool as
/// `tojson` writes it), replays calls in its XML, puts consecutive tool
/// results in one user turn, and writes no turn for an empty system
/// message. The references are its `chat_template.jinja` rendered over the
/// same snapshot (`fixtures/qwen38_reference.json`), tokenized whole.
fn qwen38_writes_its_tools_calls_and_results_as_its_template_does() {
    let Some(tokenizer) = snapshot(
        "mlx-community/Qwen3.8-27B-4bit",
        "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
    ) else {
        return;
    };
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen38_reference.json")).unwrap();
    let expect = |case: &str| tokenizer.encode(reference[case].as_str().unwrap());
    let qwen38 = chat_template::chatml::ChatMLInstruct::try_new(
        tokenizer.clone(),
        chat_template::chatml::ChatML {
            thinking: true,
            preserve_thinking: true,
            reasoning_effort: true,
            tools: true,
            xml_tools: true,
            generation_suffix: "<think>\n".to_string(),
            stop_tokens: vec!["<|im_end|>".to_string()],
        },
    )
    .expect("the tokenizer has the template's markers");
    let tool = r#"{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}"#.to_string();
    let tools = std::slice::from_ref(&tool);

    let mut got = qwen38.open(Some("You are terse."), tools, Some(Effort::Xhigh));
    got.extend(qwen38.user("Weather in Paris and Rome?"));
    got.extend(qwen38.assistant_calls(
        "<think>Need both.</think>",
        &[
            ("get_weather".into(), r#"{"city":"Paris"}"#.into()),
            ("get_weather".into(), r#"{"city":"Rome","days":3}"#.into()),
        ],
        true,
    ));
    got.extend(qwen38.answers(&[
        ("get_weather".into(), "Sunny".into()),
        ("get_weather".into(), "Rain".into()),
    ]));
    got.extend(qwen38.cue());
    assert_eq!(got, expect("tools"));

    let mut got = qwen38.open(Some(""), &[], Some(Effort::Xhigh));
    got.extend(qwen38.user("Hi"));
    got.extend(qwen38.cue());
    assert_eq!(got, expect("empty_system"));

    let mut got = qwen38.open(None, tools, None);
    got.extend(qwen38.user("Hi"));
    got.extend(qwen38.cue_without_thinking());
    assert_eq!(got, expect("tools_quiet"));
}

/// A request that opens on a system message and the tools, calls a tool and
/// gets two results back, written as the compat prompt writes it.
fn opened_tool_loop(template: &dyn Instruct) -> Vec<u32> {
    let mut got = template.open(Some("You are terse."), &[GLM_TOOL.to_string()], None);
    got.extend(template.user("Weather in Paris?"));
    got.extend(template.assistant_calls(
        "",
        &[(
            "get_weather".to_string(),
            "{\"city\": \"Paris\"}".to_string(),
        )],
        true,
    ));
    got.extend(template.answers(&[
        ("get_weather".to_string(), "{\"temp\": 21}".to_string()),
        ("get_weather".to_string(), "{\"wind\": 3}".to_string()),
    ]));
    got.extend(template.cue());
    got
}

#[test]
fn every_new_template_opens_on_its_system_turn_and_its_tools_as_the_reference_does() {
    opens_llama_3_1_8b_instruct();
    opens_glm_4_7();
    opens_glm_4_5();
    opens_minimax_m2_5();
    opens_qwen3_coder_30b_a3b_instruct();
}

fn opens_llama_3_1_8b_instruct() {
    let Some(tokenizer) = snapshot(
        "unsloth/Llama-3.1-8B-Instruct",
        "4699cc75b550f9c6f3173fb80f4703b62d946aa5",
    ) else {
        return;
    };
    let template = chat_template::llama::Llama::try_new(tokenizer)
        .expect("the tokenizer has the template's markers");
    let want: Vec<u32> = vec![
        128000, 128006, 9125, 128007, 271, 13013, 25, 6125, 27993, 198, 38766, 1303, 33025, 2696,
        25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 271, 2675,
        617, 2680, 311, 279, 2768, 5865, 13, 2057, 1650, 264, 734, 11, 4587, 6013, 449, 4823, 369,
        264, 734, 1650, 8542, 3595, 304, 279, 3645, 5324, 609, 794, 734, 836, 11, 330, 14105, 794,
        11240, 315, 5811, 836, 323, 1202, 907, 7966, 5519, 539, 1005, 7482, 382, 517, 262, 330,
        1337, 794, 330, 1723, 761, 262, 330, 1723, 794, 341, 286, 330, 609, 794, 330, 456, 70464,
        761, 286, 330, 4789, 794, 330, 30081, 761, 286, 330, 14105, 794, 341, 310, 330, 1337, 794,
        330, 1735, 761, 310, 330, 13495, 794, 341, 394, 330, 9103, 794, 341, 504, 330, 1337, 794,
        330, 928, 702, 394, 457, 310, 457, 286, 457, 262, 457, 633, 2675, 527, 51637, 13, 128009,
        128006, 882, 128007, 271, 30081, 304, 12366, 30, 128009, 128006, 78191, 128007, 271, 5018,
        609, 794, 330, 456, 70464, 498, 330, 14105, 794, 5324, 9103, 794, 330, 60704, 32075,
        128009, 128006, 23799, 4690, 128007, 271, 14461, 2153, 3975, 11955, 220, 1691, 10064,
        128009, 128006, 23799, 4690, 128007, 271, 14461, 2153, 19703, 11955, 220, 18, 10064,
        128009, 128006, 78191, 128007, 271,
    ];
    assert_eq!(
        opened_tool_loop(&template),
        want,
        "unsloth/Llama-3.1-8B-Instruct"
    );
}
fn opens_glm_4_7() {
    let Some(tokenizer) = snapshot(
        "zai-org/GLM-4.7",
        "602d01efcdd332c5238ca4bcede555defbe83eb7",
    ) else {
        return;
    };
    let template =
        chat_template::glm::Glm::try_with_style(tokenizer, chat_template::glm::Style::Glm47)
            .expect("the tokenizer has the template's markers");
    let want: Vec<u32> = vec![
        151331, 151333, 151335, 198, 2, 13843, 271, 2610, 1231, 1618, 825, 476, 803, 5746, 311,
        7789, 448, 279, 1196, 3239, 382, 2610, 525, 3897, 448, 729, 32479, 2878, 366, 15906, 1472,
        15906, 29, 11868, 9488, 510, 27, 15906, 397, 4913, 1313, 788, 330, 1688, 497, 330, 1688,
        788, 5212, 606, 788, 330, 455, 68852, 497, 330, 4684, 788, 330, 28876, 497, 330, 13777,
        788, 5212, 1313, 788, 330, 1700, 497, 330, 13185, 788, 5212, 8923, 788, 5212, 1313, 788,
        330, 917, 30849, 3417, 532, 522, 15906, 1339, 2461, 1817, 729, 1618, 11, 2550, 279, 729,
        829, 323, 5977, 2878, 279, 2701, 11868, 3561, 510, 151352, 90, 1688, 11489, 92, 151356, 90,
        858, 16159, 12, 16, 92, 151357, 151358, 90, 858, 19048, 12, 16, 92, 151359, 151356, 90,
        858, 16159, 12, 17, 92, 151357, 151358, 90, 858, 19048, 12, 17, 92, 151359, 1112, 151353,
        151335, 2610, 525, 50205, 13, 151336, 28876, 304, 12089, 30, 151337, 151351, 151352, 455,
        68852, 151356, 8923, 151357, 151358, 59190, 151359, 151353, 151338, 151354, 4913, 3888,
        788, 220, 99146, 92, 151355, 151354, 4913, 19118, 788, 220, 18, 92, 151355, 151337, 151350,
    ];
    assert_eq!(opened_tool_loop(&template), want, "zai-org/GLM-4.7");
}
fn opens_glm_4_5() {
    let Some(tokenizer) = snapshot(
        "zai-org/GLM-4.5",
        "cbb2c7cfb52fa128a9660cb1a7a78e017899e115",
    ) else {
        return;
    };
    let template =
        chat_template::glm::Glm::try_with_style(tokenizer, chat_template::glm::Style::Glm45)
            .expect("the tokenizer has the template's markers");
    let want: Vec<u32> = vec![
        151331, 151333, 151335, 198, 2, 13843, 271, 2610, 1231, 1618, 825, 476, 803, 5746, 311,
        7789, 448, 279, 1196, 3239, 382, 2610, 525, 3897, 448, 729, 32479, 2878, 366, 15906, 1472,
        15906, 29, 11868, 9488, 510, 27, 15906, 397, 4913, 1313, 788, 330, 1688, 497, 330, 1688,
        788, 5212, 606, 788, 330, 455, 68852, 497, 330, 4684, 788, 330, 28876, 497, 330, 13777,
        788, 5212, 1313, 788, 330, 1700, 497, 330, 13185, 788, 5212, 8923, 788, 5212, 1313, 788,
        330, 917, 30849, 3417, 532, 522, 15906, 1339, 2461, 1817, 729, 1618, 11, 2550, 279, 729,
        829, 323, 5977, 2878, 279, 2701, 11868, 3561, 510, 151352, 90, 1688, 11489, 532, 151356,
        90, 858, 16159, 12, 16, 92, 151357, 198, 151358, 90, 858, 19048, 12, 16, 92, 151359, 198,
        151356, 90, 858, 16159, 12, 17, 92, 151357, 198, 151358, 90, 858, 19048, 12, 17, 92,
        151359, 198, 9335, 151353, 151335, 198, 2610, 525, 50205, 13, 151336, 198, 28876, 304,
        12089, 30, 151337, 198, 151350, 151351, 198, 151352, 455, 68852, 198, 151356, 8923, 151357,
        198, 151358, 59190, 151359, 198, 151353, 151338, 198, 151354, 198, 4913, 3888, 788, 220,
        99146, 532, 151355, 198, 151354, 198, 4913, 19118, 788, 220, 18, 532, 151355, 151337,
    ];
    assert_eq!(opened_tool_loop(&template), want, "zai-org/GLM-4.5");
}
fn opens_minimax_m2_5() {
    let Some(tokenizer) = snapshot(
        "MiniMaxAI/MiniMax-M2.5",
        "f710177d938eff80b684d42c5aa84b382612f21f",
    ) else {
        return;
    };
    let template = chat_template::minimax2::MiniMax2::try_new(
        tokenizer,
        "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.",
    )
    .expect("the tokenizer has the template's markers");
    let want: Vec<u32> = vec![
        200034, 200019, 28463, 10, 2985, 457, 6000, 101, 634, 35, 31455, 10, 2985, 992, 1728, 841,
        436, 812, 6629, 301, 7202, 418, 275, 3100, 9560, 320, 9806, 457, 275, 6629, 3136, 296,
        19535, 19357, 6634, 3728, 60, 32650, 1100, 60, 45382, 62, 16673, 1906, 2803, 494, 848, 95,
        51872, 1164, 494, 16944, 2803, 494, 66079, 1164, 494, 24344, 2803, 18396, 4467, 2803, 494,
        7052, 1164, 494, 41187, 2803, 18396, 44044, 2803, 18396, 4467, 2803, 494, 3386, 150370,
        21384, 45382, 1100, 1579, 32650, 6983, 4625, 3527, 3994, 9885, 44, 1223, 27190, 6634, 301,
        48687, 6629, 306, 1916, 7729, 3728, 200052, 10, 60, 88638, 1925, 1139, 45382, 28145, 45,
        49, 6143, 60, 24662, 1925, 1139, 2949, 38158, 45, 49, 3361, 2949, 27306, 45, 49, 1579,
        24662, 1100, 60, 24662, 1925, 1139, 2949, 38158, 45, 50, 3361, 2949, 27306, 45, 50, 1579,
        24662, 1100, 4563, 1579, 88638, 1100, 200053, 200020, 10, 200019, 3995, 10, 66079, 296,
        8261, 63, 200020, 10, 200019, 1361, 367, 200052, 10, 60, 88638, 1925, 1139, 848, 95, 51872,
        6143, 60, 24662, 1925, 1139, 44044, 3361, 44890, 1579, 24662, 1100, 1579, 88638, 1100,
        200053, 200020, 10, 200019, 45382, 10, 60, 9679, 62, 16673, 40886, 2803, 32, 2254, 30085,
        9679, 1100, 60, 9679, 62, 16673, 31533, 2803, 32, 51, 30085, 9679, 62, 200020, 10, 200019,
        1361, 10, 200050, 10,
    ];
    assert_eq!(opened_tool_loop(&template), want, "MiniMaxAI/MiniMax-M2.5");
}
fn opens_qwen3_coder_30b_a3b_instruct() {
    let Some(tokenizer) = snapshot(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
    ) else {
        return;
    };
    let template = chat_template::qwen3_coder::Qwen3Coder::try_new(tokenizer)
        .expect("the tokenizer has the template's markers");
    let want: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 50537, 382, 2, 13852, 271, 2610, 614, 2615, 311, 279, 2701,
        5746, 1447, 27, 15918, 397, 27, 1688, 397, 27, 606, 29, 455, 69364, 522, 606, 397, 27,
        4684, 29, 28981, 522, 4684, 397, 27, 13786, 397, 27, 16181, 397, 27, 606, 29, 8926, 522,
        606, 397, 27, 1313, 29, 917, 522, 1313, 397, 522, 16181, 397, 522, 13786, 397, 522, 1688,
        397, 522, 15918, 1339, 2679, 498, 5157, 311, 1618, 264, 729, 26687, 9851, 304, 279, 2701,
        3561, 448, 5664, 20525, 1447, 151657, 198, 27, 1688, 28, 8687, 9174, 1269, 397, 27, 16181,
        28, 8687, 24899, 62, 16, 397, 957, 62, 16, 198, 522, 16181, 397, 27, 16181, 28, 8687,
        24899, 62, 17, 397, 1986, 374, 279, 897, 369, 279, 2086, 5733, 198, 9033, 646, 9390, 198,
        35673, 5128, 198, 522, 16181, 397, 522, 1688, 397, 151658, 271, 27, 98743, 397, 95359, 510,
        12, 5712, 6738, 27732, 1795, 279, 5189, 3561, 25, 458, 9179, 366, 1688, 28, 1112, 1472,
        1688, 29, 2504, 1969, 387, 24034, 2878, 220, 151657, 151658, 11874, 9492, 198, 12, 12659,
        5029, 27732, 387, 5189, 198, 12, 1446, 1231, 3410, 10101, 32711, 369, 697, 729, 1618, 304,
        5810, 4128, 56482, 279, 729, 1618, 11, 714, 4183, 1283, 198, 12, 1416, 1052, 374, 902, 729,
        1618, 2500, 11, 4226, 279, 3405, 1075, 4622, 448, 697, 1482, 6540, 323, 653, 537, 3291,
        279, 1196, 911, 729, 6738, 198, 522, 98743, 29, 151645, 198, 151644, 872, 198, 28981, 304,
        12095, 30, 151645, 198, 151644, 77091, 198, 151657, 198, 27, 1688, 28280, 69364, 397, 27,
        16181, 28, 8926, 397, 59604, 198, 522, 16181, 397, 522, 1688, 397, 151658, 151645, 198,
        151644, 872, 198, 151665, 198, 4913, 3888, 788, 220, 17, 16, 532, 151666, 198, 151665, 198,
        4913, 19154, 788, 220, 18, 532, 151666, 198, 151645, 198, 151644, 77091, 198,
    ];
    assert_eq!(
        opened_tool_loop(&template),
        want,
        "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    );
}
