//! SSE streaming utilities for OpenResponses.

use crate::types::*;
use serde::Serialize;

/// SSE event emitter for building the streaming response
pub struct StreamEmitter {
    sequence: u32,
}

impl StreamEmitter {
    pub fn new() -> Self {
        Self { sequence: 0 }
    }

    fn next_seq(&mut self) -> u32 {
        let seq = self.sequence;
        self.sequence += 1;
        seq
    }

    /// Format an SSE event with the given event type and JSON data
    fn format_event<T: Serialize>(&mut self, event_type: &str, data: T) -> String {
        let event = StreamingEvent {
            event_type: event_type.to_string(),
            sequence_number: self.next_seq(),
            data,
        };
        let json = serde_json::to_string(&event).unwrap_or_default();
        format!("event: {}\ndata: {}\n\n", event_type, json)
    }

    /// Emit response.created
    pub fn response_created(&mut self, response: &ResponseResource) -> String {
        self.format_event("response.created", ResponseCreatedData {
            response: clone_response(response),
        })
    }

    /// Emit response.in_progress
    pub fn response_in_progress(&mut self, response: &ResponseResource) -> String {
        self.format_event("response.in_progress", ResponseInProgressData {
            response: clone_response(response),
        })
    }

    /// Emit response.output_item.added
    pub fn output_item_added(&mut self, output_index: u32, item: &OutputItem) -> String {
        self.format_event("response.output_item.added", OutputItemAddedData {
            output_index,
            item: item.clone(),
        })
    }

    /// Emit response.content_part.added
    pub fn content_part_added(
        &mut self,
        item_id: &str,
        output_index: u32,
        content_index: u32,
        part: &OutputContentPart,
    ) -> String {
        self.format_event("response.content_part.added", ContentPartAddedData {
            item_id: item_id.to_string(),
            output_index,
            content_index,
            part: part.clone(),
        })
    }

    /// Emit response.output_text.delta
    pub fn output_text_delta(
        &mut self,
        item_id: &str,
        output_index: u32,
        content_index: u32,
        delta: &str,
    ) -> String {
        self.format_event("response.output_text.delta", OutputTextDeltaData {
            item_id: item_id.to_string(),
            output_index,
            content_index,
            delta: delta.to_string(),
        })
    }

    /// Emit response.output_text.done
    pub fn output_text_done(
        &mut self,
        item_id: &str,
        output_index: u32,
        content_index: u32,
        text: &str,
    ) -> String {
        self.format_event("response.output_text.done", OutputTextDoneData {
            item_id: item_id.to_string(),
            output_index,
            content_index,
            text: text.to_string(),
        })
    }

    /// Emit response.content_part.done
    pub fn content_part_done(
        &mut self,
        item_id: &str,
        output_index: u32,
        content_index: u32,
        part: &OutputContentPart,
    ) -> String {
        self.format_event("response.content_part.done", ContentPartDoneData {
            item_id: item_id.to_string(),
            output_index,
            content_index,
            part: part.clone(),
        })
    }

    /// Emit response.output_item.done
    pub fn output_item_done(&mut self, output_index: u32, item: &OutputItem) -> String {
        self.format_event("response.output_item.done", OutputItemDoneData {
            output_index,
            item: item.clone(),
        })
    }

    /// Emit response.completed
    pub fn response_completed(&mut self, response: &ResponseResource) -> String {
        self.format_event("response.completed", ResponseCompletedData {
            response: clone_response(response),
        })
    }

    /// Final [DONE] marker
    pub fn done() -> String {
        "data: [DONE]\n\n".to_string()
    }
}

/// Helper to clone a ResponseResource for serialization
fn clone_response(r: &ResponseResource) -> ResponseResource {
    ResponseResource {
        id: r.id.clone(),
        response_type: r.response_type.clone(),
        status: r.status.clone(),
        output: r.output.clone(),
        error: None, // Don't include error in normal events
        usage: None, // Usage is typically only in final response
    }
}
