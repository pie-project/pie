//! Tool-use test inferlet — exercises the tool-use host APIs from real WASM.
//!
//! This is the only place the tool path is driven across every boundary it
//! crosses in production: SDK -> generated WIT bindings -> host
//! `pie::instruct::tool_use` -> the model's `Instruct` implementation. Runtime
//! unit tests can reach the last hop only.

use inferlet::{Context, Result, chat, model::Model, runtime, tools};

struct Weather;

impl tools::Tool for Weather {
    fn name(&self) -> &'static str {
        "get_weather"
    }
    fn description(&self) -> &'static str {
        "Look up weather"
    }
    fn schema(&self) -> &'static str {
        r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
    }
}

/// The same tool in the flat envelope `Context::equip` builds for the host, so
/// the raw and ergonomic paths can be compared token for token.
fn weather_envelope() -> String {
    r#"{"name":"get_weather","description":"Look up weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}"#
        .to_string()
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let models = runtime::models();
    let model = Model::load(&models[0])?;
    let tokenizer = model.tokenizer();
    let schemas = vec![weather_envelope()];
    let call_text = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";

    // (a) Declaration. Standalone equip must fail loudly for an architecture
    //     that nests declarations in the system turn, rather than returning no
    //     tokens while the tool grammar still compiles. The fused host call is
    //     the supported path and must carry declarations.
    let standalone = match tools::equip_prefix(&model, &schemas) {
        Ok(tokens) => format!("ok:{}", tokens.len()),
        Err(error) => format!("err:{error}"),
    };
    let fused = tools::system_equip_prefix(&model, "You are helpful.", &schemas)?;
    // Newlines are escaped so each reported field stays on one line;
    // the prompt itself contains turn-delimiting newlines.
    let fused_text = tokenizer.decode(&fused)?.replace('\n', "\\n");

    // `Context::equip` fuses a pending system message the same way, so the
    // ergonomic SDK path and the raw host call agree token for token.
    let mut ctx = Context::new(&model)?;
    ctx.system("You are helpful.");
    ctx.equip(&[&Weather])?;
    let ctx_matches = ctx.buffer() == fused.as_slice();

    // (b) The context remembers what was equipped, so `Context::tool_decoder`
    //     reports the declared call and refuses an undeclared one without the
    //     caller threading schemas through by hand.
    let declared_call = {
        let mut d = ctx.tool_decoder();
        let mut got = "none".to_string();
        for token in tokenizer.encode(call_text) {
            if let tools::Event::Call(name, arguments) = d.feed(&[token])? {
                got = format!("{name}|{arguments}");
            }
        }
        got
    };
    let undeclared_rejected = {
        let mut d = ctx.tool_decoder();
        let mut reported = false;
        for token in tokenizer.encode("<|tool_call>call:read_file{}<tool_call|>") {
            if let tools::Event::Call(..) = d.feed(&[token])? {
                reported = true;
            }
        }
        !reported
    };

    // (c) Response framing. `answer_prefix` frames a tool result for the next
    //     turn without opening a user turn.
    let answer = tokenizer
        .decode(&tools::answer_prefix(
            &model,
            "get_weather",
            "18C and clear",
        ))?
        .replace('\n', "\\n");

    // (d) The full loop, as a prompt. declaration -> call -> observation ->
    //     continuation, assembled the way an agent actually builds it: the
    //     model's call is appended, the tool result is framed by `answer`, and
    //     generation resumes with *no* second cue, because the model turn
    //     opened for the call is still open. Reporting the whole thing lets the
    //     host test assert the resulting turn structure exactly.
    let mut loop_tokens = fused.clone();
    loop_tokens.extend(tokenizer.encode("<|turn>user\nWeather in Paris?<turn|>\n"));
    loop_tokens.extend(chat::cue(&model));
    loop_tokens.extend(tokenizer.encode(call_text));
    loop_tokens.extend(tools::answer_prefix(&model, "get_weather", "18C and clear"));
    let loop_text = tokenizer.decode(&loop_tokens)?.replace('\n', "\\n");

    Ok(format!(
        "standalone={standalone}\n\
         fused={fused_text}\n\
         ctx_matches={ctx_matches}\n\
         declared_call={declared_call}\n\
         undeclared_rejected={undeclared_rejected}\n\
         answer={answer}\n\
         loop={loop_text}"
    ))
}
