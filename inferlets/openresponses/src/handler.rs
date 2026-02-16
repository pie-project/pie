//! Request handler for POST /responses endpoint.

use crate::streaming::StreamEmitter;
use crate::types::*;
use wstd::http::body::BodyForthcoming;
use wstd::http::server::{Finished, Responder};
use wstd::http::{IntoBody, Response};
use wstd::io::AsyncWrite;

use inferlet::context::Context;
use inferlet::inference::Sampler;
use inferlet::model::Model;
use inferlet::runtime;
use inferlet::{ContextExt, InstructExt};

/// Generate a unique ID for responses and messages
fn generate_id(prefix: &str) -> String {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    format!("{}_{:016x}", prefix, count)
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
            return error_response(responder, 400, "invalid_request", &format!("Invalid JSON: {}", e), None, None).await;
        }
    };

    // Extract messages from input
    let mut system_message = request.instructions.clone();
    let mut user_messages = Vec::new();

    for item in &request.input {
        match item {
            InputItem::Message(msg) => {
                let text = msg.content.as_text();
                match msg.role {
                    Role::System | Role::Developer => {
                        system_message = Some(text);
                    }
                    Role::User => {
                        user_messages.push(text);
                    }
                    Role::Assistant => {
                        // Could be used for multi-turn, skip for now
                    }
                }
            }
            InputItem::FunctionCall(_fc) => {
                // Function calls from previous turns - would be used for multi-turn
                // For now, skip these
            }
            InputItem::FunctionCallOutput(fco) => {
                // Function call output - include as user message for context
                user_messages.push(format!("Function result: {}", fco.output));
            }
            InputItem::ItemReference { .. } => {
                // Skip references for now
            }
        }
    }

    if user_messages.is_empty() {
        return error_response(responder, 400, "invalid_request", "No user message provided", None, None).await;
    }

    // Get sampling parameters
    let max_tokens = request.max_output_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.6);
    let top_p = request.top_p.unwrap_or(0.95);

    // Generate response
    if request.stream {
        handle_streaming_response(
            responder,
            system_message,
            user_messages,
            max_tokens,
            temperature,
            top_p,
        ).await
    } else {
        handle_non_streaming_response(
            responder,
            system_message,
            user_messages,
            max_tokens,
            temperature,
            top_p,
        ).await
    }
}

/// Handle streaming response with SSE - TRUE incremental streaming with flush()
async fn handle_streaming_response(
    responder: Responder,
    system_message: Option<String>,
    user_messages: Vec<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Finished {
    // Create IDs
    let response_id = generate_id("resp");
    let message_id = generate_id("msg");

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

    // Set up model
    let models = runtime::models();
    let model_name = models.first().map(|m| m.to_string()).unwrap_or_else(|| "unknown".to_string());

    // Create initial response object
    let mut response = ResponseResource::new(response_id.clone(), model_name.clone());

    // Emit response.created
    emit!(emitter.response_created(&response));

    // Emit response.in_progress
    response.status = ResponseStatus::InProgress;
    emit!(emitter.response_in_progress(&response));

    // Create output message item
    let output_item = OutputItem::Message(OutputMessage {
        id: message_id.clone(),
        role: Role::Assistant,
        status: ItemStatus::InProgress,
        content: vec![],
    });

    // Emit response.output_item.added
    emit!(emitter.output_item_added(0, &output_item));

    // Create content part
    let content_part = OutputContentPart::OutputText {
        text: String::new(),
        annotations: vec![],
    };

    // Emit response.content_part.added
    emit!(emitter.content_part_added(&message_id, 0, 0, &content_part));

    // Load model
    let model = match Model::load(&models[0]) {
        Ok(m) => m,
        Err(e) => {
            response.status = ResponseStatus::Failed;
            response.error = Some(ErrorPayload {
                error_type: "server_error".to_string(),
                code: Some("model_load_failed".to_string()),
                message: format!("Failed to load model: {}", e),
                param: None,
            });
            emit!(emitter.response_failed(&response));
            emit!(StreamEmitter::done());
            return Finished::finish(body, Ok(()), None);
        }
    };
    let tokenizer = model.tokenizer();

    let ctx = match Context::new(&model) {
        Ok(c) => c,
        Err(e) => {
            response.status = ResponseStatus::Failed;
            response.error = Some(ErrorPayload {
                error_type: "server_error".to_string(),
                code: Some("context_creation_failed".to_string()),
                message: format!("Failed to create context: {}", e),
                param: None,
            });
            emit!(emitter.response_failed(&response));
            emit!(StreamEmitter::done());
            return Finished::finish(body, Ok(()), None);
        }
    };

    // Fill context using InstructExt
    if let Some(sys) = &system_message {
        ctx.system(sys);
    }
    for msg in &user_messages {
        ctx.user(msg);
    }
    ctx.cue();

    // Flush the prompt
    if let Err(e) = ctx.flush().await {
        response.status = ResponseStatus::Failed;
        response.error = Some(ErrorPayload {
            error_type: "server_error".to_string(),
            code: Some("flush_failed".to_string()),
            message: format!("Flush failed: {}", e),
            param: None,
        });
        emit!(emitter.response_failed(&response));
        emit!(StreamEmitter::done());
        return Finished::finish(body, Ok(()), None);
    }

    let sampler = Sampler::TopP((temperature, top_p));

    let mut full_text = String::new();
    let mut tokens_generated: usize = 0;

    // Token-by-token generation loop with TRUE streaming
    let mut stream = ctx.generate(sampler).with_max_tokens(max_tokens);
    while let Ok(Some(tokens)) = stream.next().await {
        tokens_generated += tokens.len();
        let delta_text = tokenizer.decode(&tokens).unwrap_or_default();

        // Emit response.output_text.delta for this batch (with flush!)
        if !delta_text.is_empty() {
            emit!(emitter.output_text_delta(&message_id, 0, 0, &delta_text));
            full_text.push_str(&delta_text);
        }
    }

    // Determine if we hit max_tokens (incomplete) or finished naturally (completed)
    let hit_max = tokens_generated >= max_tokens;
    let item_status = if hit_max { ItemStatus::Incomplete } else { ItemStatus::Completed };
    let response_status = if hit_max { ResponseStatus::Incomplete } else { ResponseStatus::Completed };

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
        status: item_status,
        content: vec![final_content_part],
    });

    // Emit response.output_item.done
    emit!(emitter.output_item_done(0, &final_output_item));

    // Update and emit response.completed
    response.status = response_status;
    response.output = vec![final_output_item];
    if hit_max {
        emit!(emitter.response_completed(&response));
    } else {
        emit!(emitter.response_completed(&response));
    }

    // Emit [DONE]
    emit!(StreamEmitter::done());

    Finished::finish(body, Ok(()), None)
}

/// Handle non-streaming response (return JSON directly)
async fn handle_non_streaming_response(
    responder: Responder,
    system_message: Option<String>,
    user_messages: Vec<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Finished {
    // Create IDs
    let response_id = generate_id("resp");
    let message_id = generate_id("msg");

    // Set up model and generate
    let models = runtime::models();
    let model_name = models.first().map(|m| m.to_string()).unwrap_or_else(|| "unknown".to_string());
    let model = match Model::load(&models[0]) {
        Ok(m) => m,
        Err(e) => {
            return error_response(responder, 500, "server_error", &format!("Failed to load model: {}", e), None, None).await;
        }
    };
    let tokenizer = model.tokenizer();

    let ctx = match Context::new(&model) {
        Ok(c) => c,
        Err(e) => {
            return error_response(responder, 500, "server_error", &format!("Failed to create context: {}", e), None, None).await;
        }
    };

    // Fill context using InstructExt
    if let Some(sys) = &system_message {
        ctx.system(sys);
    }
    for msg in &user_messages {
        ctx.user(msg);
    }
    ctx.cue();

    // Flush the prompt
    if let Err(e) = ctx.flush().await {
        return error_response(responder, 500, "server_error", &format!("Flush failed: {}", e), None, None).await;
    }

    // Generate token by token to count tokens for incomplete detection
    let sampler = Sampler::TopP((temperature, top_p));
    let mut stream = ctx.generate(sampler).with_max_tokens(max_tokens);
    let mut full_text = String::new();
    let mut tokens_generated: usize = 0;

    while let Ok(Some(tokens)) = stream.next().await {
        tokens_generated += tokens.len();
        let text = tokenizer.decode(&tokens).unwrap_or_default();
        full_text.push_str(&text);
    }

    // Determine if we hit max_tokens
    let hit_max = tokens_generated >= max_tokens;
    let item_status = if hit_max { ItemStatus::Incomplete } else { ItemStatus::Completed };
    let response_status = if hit_max { ResponseStatus::Incomplete } else { ResponseStatus::Completed };

    // Build response
    let output_item = OutputItem::Message(OutputMessage {
        id: message_id,
        role: Role::Assistant,
        status: item_status,
        content: vec![OutputContentPart::OutputText {
            text: full_text,
            annotations: vec![],
        }],
    });

    let response = ResponseResource {
        id: response_id,
        response_type: "response".to_string(),
        status: response_status,
        model: model_name,
        output: vec![output_item],
        error: None,
        usage: None,
    };

    let json = serde_json::to_string(&response).unwrap_or_default();

    let http_response = Response::builder()
        .header("Content-Type", "application/json")
        .body(json.into_body())
        .unwrap();

    responder.respond(http_response).await
}

/// Return an error response per OpenResponses spec
async fn error_response(
    responder: Responder,
    status_code: u16,
    error_type: &str,
    message: &str,
    code: Option<&str>,
    param: Option<&str>,
) -> Finished {
    let error = serde_json::json!({
        "error": {
            "type": error_type,
            "message": message,
            "code": code,
            "param": param,
        }
    });

    let response = Response::builder()
        .status(status_code)
        .header("Content-Type", "application/json")
        .body(error.to_string().into_body())
        .unwrap();

    responder.respond(response).await
}
