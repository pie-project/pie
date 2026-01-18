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
        return error_response(responder, 400, "invalid_request", "No user message provided").await;
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
    use inferlet::stop_condition::{max_len, ends_with_any};
    use inferlet::Sampler;

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

    // Create initial response object
    let mut response = ResponseResource::new(response_id.clone());

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

    // Set up model and generate token by token
    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();
    let tokenizer = model.get_tokenizer();

    // Fill context
    if let Some(sys) = &system_message {
        ctx.fill_system(sys);
    }
    for msg in &user_messages {
        ctx.fill_user(msg);
    }

    let sampler = Sampler::top_p(temperature, top_p);
    let stop_cond = max_len(max_tokens).or(ends_with_any(model.eos_tokens()));

    let mut generated_token_ids = Vec::new();
    let mut full_text = String::new();

    // Token-by-token generation loop with TRUE streaming
    loop {
        let next_token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);

        // Decode just this token to get the delta text
        let delta_text = tokenizer.detokenize(&[next_token_id]);

        // Emit response.output_text.delta for this token (with flush!)
        if !delta_text.is_empty() {
            emit!(emitter.output_text_delta(&message_id, 0, 0, &delta_text));
            full_text.push_str(&delta_text);
        }

        // Check stop condition
        if stop_cond.check(&generated_token_ids) {
            break;
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

    // Update and emit response.completed
    response.status = ResponseStatus::Completed;
    response.output = vec![final_output_item];
    emit!(emitter.response_completed(&response));

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
    use inferlet::stop_condition::{max_len, ends_with_any};
    use inferlet::Sampler;

    // Create IDs
    let response_id = generate_id("resp");
    let message_id = generate_id("msg");

    // Set up model and generate
    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();

    // Fill context
    if let Some(sys) = &system_message {
        ctx.fill_system(sys);
    }
    for msg in &user_messages {
        ctx.fill_user(msg);
    }

    // Generate
    let sampler = Sampler::top_p(temperature, top_p);
    let stop_cond = max_len(max_tokens).or(ends_with_any(model.eos_tokens()));

    let generated = ctx.generate(sampler, stop_cond).await;

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
        usage: None,
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
