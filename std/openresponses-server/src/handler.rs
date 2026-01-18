//! Request handler for POST /responses endpoint.

use crate::streaming::StreamEmitter;
use crate::types::*;
use wstd::http::body::BodyForthcoming;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};
use wstd::io::AsyncWrite;
use inferlet::stop_condition::StopCondition;

/// Generate a unique ID for responses and messages
fn generate_id(prefix: &str) -> String {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    format!("{}_{:016x}", prefix, count)
}

#[derive(Debug, Clone)]
pub enum ContextAction {
    User(String),
    Assistant(String),
    ToolCall(String, serde_json::Value),
    ToolResponse(String),
}

/// Handle the POST /responses endpoint
pub async fn handle_responses<B>(
    body_bytes: Vec<u8>,
    responder: Responder,
) -> Finished {
    // Parse the request body
    let request: CreateResponseBody = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return error_response(responder, 400, "invalid_request", &format!("Invalid JSON: {}", e)).await;
        }
    };

    // Extract messages from input
    let mut system_message = request.instructions.clone();

    // Inject tools into system message if present
    if !request.tools.is_empty() {
        let tools_json = serde_json::to_string_pretty(&request.tools).unwrap_or_default();
        let tool_prompt = format!("\n\n# Available Tools\n\nYou have access to the following tools. To use a tool, you MUST start your response with the following format:\nCall: tool_name(json_arguments)\n\nExample:\nCall: get_weather({{\"location\": \"London\"}})\n\nAvailable tools:\n{}\n", tools_json);
        system_message = Some(match system_message {
            Some(s) => format!("{}{}", s, tool_prompt),
            None => tool_prompt,
        });
    }

    let mut context_actions = Vec::new();

    for item in &request.input {
        match item {
            InputItem::Message(msg) => {
                let text = msg.content.as_text();
                match msg.role {
                    Role::System | Role::Developer => {
                        system_message = Some(text);
                    }
                    Role::User => {
                        context_actions.push(ContextAction::User(text));
                    }
                    Role::Assistant => {
                        context_actions.push(ContextAction::Assistant(text));
                    }
                }
            }
            InputItem::FunctionCall(fc) => {
                let args = serde_json::from_str(&fc.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(fc.arguments.clone()));
                context_actions.push(ContextAction::ToolCall(fc.name.clone(), args));
            }
            InputItem::FunctionCallOutput(fco) => {
                context_actions.push(ContextAction::ToolResponse(fco.output.clone()));
            }
            InputItem::ItemReference { .. } => {
                // Skip references for now
            }
        }
    }

    if request.stream {
        handle_streaming_response(
            responder,
            generate_id("resp"),
            generate_id("msg"),
            system_message,
            context_actions,
            request.temperature.unwrap_or(0.7),
            request.top_p.unwrap_or(1.0),
            request.max_output_tokens.unwrap_or(1024) as u32,
        ).await
    } else {
        handle_non_streaming_response(
            responder,
            generate_id("resp"),
            generate_id("msg"),
            system_message,
            context_actions,
            request.temperature.unwrap_or(0.7),
            request.top_p.unwrap_or(1.0),
            request.max_output_tokens.unwrap_or(1024) as u32,
        ).await
    }
}

/// Handle streaming response with SSE - TRUE incremental streaming with flush()
async fn handle_streaming_response(
    responder: Responder,
    response_id: String,
    message_id: String,
    system_message: Option<String>,
    context_actions: Vec<ContextAction>,
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
) -> Finished {
    use inferlet::stop_condition::{max_len, ends_with_any};
    use inferlet::Sampler;

    // Start SSE response with BodyForthcoming for true streaming
    let sse_response = Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(BodyForthcoming)
        .unwrap();

    let mut body = responder.start_response(sse_response);
    let mut emitter = StreamEmitter::new();

    // Helper to emit and flush an SSE event
    macro_rules! emit {
        ($event:expr) => {{
            if body.write_all($event.as_bytes()).await.is_err() {
                return Finished::finish(body, Ok(()), None);
            }
            // CRITICAL: flush to push data to client immediately
            if body.flush().await.is_err() {
                return Finished::finish(body, Ok(()), None);
            }
        }};
    }

    // Create initial response object
    let mut response = ResponseResource::new(response_id.clone());

    // Emit response.created
    emit!(emitter.response_created(&response));

    // Emit response.in_progress
    response.status = ResponseStatus::InProgress;
    emit!(emitter.response_in_progress(&response));

    // Variable to track if we have decided the output type yet
    let mut output_type_decided = false;
    let mut is_tool_call = false;
    let mut buffer = String::new();
    let mut tool_name = String::new();
    let mut tool_call_id = String::new();

    // Regex to detect "Call: tool_name(" pattern
    // We expect the model to output: Call: tool_name(arguments)
    use regex::Regex;
    let tool_pattern = Regex::new(r"^Call:\s*([a-zA-Z0-9_]+)\(").unwrap();

    // Set up model and generate token by token
    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();
    let tokenizer = model.get_tokenizer();

    // Fill context and calculate input tokens (approximate)
    let mut input_tokens = 0;
    if let Some(sys) = &system_message {
        ctx.fill_system(sys);
        input_tokens += (sys.len() / 4) as u32;
    }
    for action in &context_actions {
        match action {
            ContextAction::User(msg) => {
                ctx.fill_user(msg);
                input_tokens += (msg.len() / 4) as u32;
            }
            ContextAction::Assistant(msg) => {
                ctx.fill_assistant(msg);
                input_tokens += (msg.len() / 4) as u32;
            }
            ContextAction::ToolCall(name, args) => {
                ctx.fill_assistant_tool_call(name, args.clone());
                input_tokens += (name.len() / 4) as u32 + (args.to_string().len() / 4) as u32;
            }
            ContextAction::ToolResponse(msg) => {
                ctx.fill_tool(msg);
                input_tokens += (msg.len() / 4) as u32;
            }
        }
    }

    let sampler = Sampler::top_p(temperature, top_p);
    let stop_cond = max_len(max_tokens as usize).or(ends_with_any(model.eos_tokens()));

    let mut generated_token_ids = Vec::new();
    let mut full_text = String::new();

    // Token-by-token generation loop with TRUE streaming
    loop {
        let next_token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);

        // Decode just this token to get the delta text
        let delta_text = tokenizer.detokenize(&[next_token_id]);

        if !output_type_decided {
            buffer.push_str(&delta_text);
            
            // simple heuristic: if buffer length > 20 and no match, it's text
            // or if we have a newline, it's text (assuming Call: is on first line)
            
            // Check if buffer matches the tool pattern
            if let Some(caps) = tool_pattern.captures(&buffer) {
                // It is a tool call!
                output_type_decided = true;
                is_tool_call = true;
                tool_name = caps[1].to_string();
                
                // Extract arguments part so far (after the first '(')
                let full_match = caps.get(0).unwrap().as_str();
                let args_start = full_match.len(); // Regex includes '(' at the end
                let args_delta = &buffer[args_start..];

                // Create and emit FunctionCall item
                tool_call_id = generate_id("call");
                let output_item = OutputItem::FunctionCall(OutputFunctionCall {
                    id: message_id.clone(), // Item ID
                    call_id: tool_call_id.clone(),
                    name: tool_name.clone(),
                    arguments: String::new(), // will be accumulated
                    status: ItemStatus::InProgress,
                });
                emit!(emitter.output_item_added(0, &output_item));
                
                // Emit initial arguments delta
                if !args_delta.is_empty() {
                    emit!(emitter.function_call_arguments_delta(&message_id, &tool_call_id, 0, args_delta));
                    full_text.push_str(args_delta);
                }
            } else if buffer.len() > 30 || buffer.contains('\n') {
                // It is NOT a tool call (gave up waiting for pattern)
                output_type_decided = true;
                is_tool_call = false;

                // Create and emit Message item (normal text)
                let output_item = OutputItem::Message(OutputMessage {
                    id: message_id.clone(),
                    role: Role::Assistant,
                    status: ItemStatus::InProgress,
                    content: vec![],
                });
                emit!(emitter.output_item_added(0, &output_item));

                // Emit content part added
                let content_part = OutputContentPart::OutputText {
                    text: String::new(),
                    annotations: vec![],
                };
                emit!(emitter.content_part_added(&message_id, 0, 0, &content_part));

                // Emit buffered text as delta
                if !buffer.is_empty() {
                    emit!(emitter.output_text_delta(&message_id, 0, 0, &buffer));
                    full_text.push_str(&buffer);
                }
            }
        } else {
            // Streaming delta
            if is_tool_call {
                // Function call arguments
                // Strip closing parenthesis if it appears at the very end? 
                // For simplicity, we just stream everything. real parser would check for closure.
                emit!(emitter.function_call_arguments_delta(&message_id, &tool_call_id, 0, &delta_text));
                full_text.push_str(&delta_text);
            } else {
                // Normal text
                if !delta_text.is_empty() {
                    emit!(emitter.output_text_delta(&message_id, 0, 0, &delta_text));
                    full_text.push_str(&delta_text);
                }
            }
        }

        // Check stop condition
        if stop_cond.check(&generated_token_ids) {
            break;
        }
    }

    // Finalize output
    if is_tool_call {
        // Clean up arguments: remove trailing ')' if present
        let clean_args = full_text.trim_end().trim_end_matches(')').to_string();
        
        // Emit arguments.done
        emit!(emitter.function_call_arguments_done(&message_id, &tool_call_id, 0, &clean_args));

        // Final output item
        let final_output_item = OutputItem::FunctionCall(OutputFunctionCall {
            id: message_id.clone(),
            call_id: tool_call_id.clone(),
            name: tool_name.clone(),
            arguments: clean_args.clone(),
            status: ItemStatus::Completed,
        });
        emit!(emitter.output_item_done(0, &final_output_item));
        
        // Update response
        response.output = vec![final_output_item];
        
    } else {
        // Assume text if no type decided (e.g. empty output)
        if !output_type_decided {
             let output_item = OutputItem::Message(OutputMessage {
                id: message_id.clone(),
                role: Role::Assistant,
                status: ItemStatus::InProgress,
                content: vec![],
            });
            emit!(emitter.output_item_added(0, &output_item));
             let content_part = OutputContentPart::OutputText {
                text: String::new(),
                annotations: vec![],
            };
            emit!(emitter.content_part_added(&message_id, 0, 0, &content_part));
            // Buffer might have partial data
             if !buffer.is_empty() {
                emit!(emitter.output_text_delta(&message_id, 0, 0, &buffer));
                full_text.push_str(&buffer);
            }
        }
        
        // Emit response.output_text.done
        emit!(emitter.output_text_done(&message_id, 0, 0, &full_text));

        // Final content part
        let final_content_part = OutputContentPart::OutputText {
            text: full_text.clone(),
            annotations: vec![],
        };

        // Emit response.content_part.done
        emit!(emitter.content_part_done(&message_id, 0, 0, &final_content_part));

        // Final output item
        let final_output_item = OutputItem::Message(OutputMessage {
            id: message_id.clone(),
            role: Role::Assistant,
            status: ItemStatus::Completed,
            content: vec![final_content_part],
        });

        // Emit response.output_item.done
        emit!(emitter.output_item_done(0, &final_output_item));
        
        response.output = vec![final_output_item];
    }

    // Create usage stats
    let output_tokens = generated_token_ids.len() as u32;
    let usage = Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
    };

    // Update and emit response.completed
    response.status = ResponseStatus::Completed;
    response.usage = Some(usage);
    emit!(emitter.response_completed(&response));

    // Emit [DONE]
    emit!(StreamEmitter::done());

    Finished::finish(body, Ok(()), None)
}

/// Handle non-streaming response (return JSON directly)
async fn handle_non_streaming_response(
    responder: Responder,
    response_id: String,
    message_id: String,
    system_message: Option<String>,
    context_actions: Vec<ContextAction>,
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
) -> Finished {
    use inferlet::stop_condition::{max_len, ends_with_any};
    use inferlet::Sampler;

    // Set up model and generate
    let model = inferlet::get_auto_model();
    // let tokenizer = model.get_tokenizer(); // Removed
    let mut ctx = model.create_context();

    // Fill context and calculate input tokens (approximate)
    let mut input_tokens = 0;
    if let Some(sys) = &system_message {
        ctx.fill_system(sys);
        input_tokens += (sys.len() / 4) as u32;
    }
    for action in &context_actions {
        match action {
            ContextAction::User(msg) => {
                ctx.fill_user(msg);
                input_tokens += (msg.len() / 4) as u32;
            }
            ContextAction::Assistant(msg) => {
                ctx.fill_assistant(msg);
                input_tokens += (msg.len() / 4) as u32;
            }
            ContextAction::ToolCall(name, args) => {
                ctx.fill_assistant_tool_call(name, args.clone());
                input_tokens += (name.len() / 4) as u32 + (args.to_string().len() / 4) as u32;
            }
            ContextAction::ToolResponse(msg) => {
                ctx.fill_tool(msg);
                input_tokens += (msg.len() / 4) as u32;
            }
        }
    }

    // Generate
    let sampler = Sampler::top_p(temperature, top_p);
    let stop_cond = max_len(max_tokens as usize).or(ends_with_any(model.eos_tokens()));

    let generated = ctx.generate(sampler, stop_cond).await;
    
    // Estimate output tokens (approximate)
    let output_tokens = (generated.len() / 4) as u32;

    let usage = Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
    };

    // Build response
    let output_item = OutputItem::Message(OutputMessage {
        id: message_id,
        role: Role::Assistant,
        status: ItemStatus::Completed,
        content: vec![OutputContentPart::OutputText {
            text: generated,
            annotations: vec![],
        }],
    });

    let response = ResponseResource {
        id: response_id,
        response_type: "response".to_string(),
        status: ResponseStatus::Completed,
        output: vec![output_item],
        error: None,
        usage: Some(usage),
    };

    let json = serde_json::to_string(&response).unwrap_or_default();

    let http_response = Response::builder()
        .header("Content-Type", "application/json")
        .body(json.into_body())
        .unwrap();

    responder.respond(http_response).await
}

/// Return an error response
async fn error_response(
    responder: Responder,
    status_code: u16,
    error_type: &str,
    message: &str,
) -> Finished {
    let error = serde_json::json!({
        "error": {
            "type": error_type,
            "message": message,
        }
    });

    let response = Response::builder()
        .status(status_code)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();

    responder.respond(response).await
}
