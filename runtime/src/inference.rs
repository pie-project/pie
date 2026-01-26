//! Inference Service - Forward pass management for model execution
//!
//! This module provides a model-specific actor for executing forward passes
//! with configurable samplers, input tokens, and attention masks.

use std::sync::LazyLock;
use tokio::sync::oneshot;

use crate::adapter::AdapterId;
use crate::context::{ContextId, LockId};
use crate::actor::{Handle, Actors, SendError};

/// Global registry for inference actors.
static ACTOR: LazyLock<Actors<Message>> = LazyLock::new(Actors::new);

/// Spawns a new inference actor.
pub(crate) fn spawn() -> usize {
    ACTOR.spawn::<InferenceActor>()
}

/// Sampler configuration for token generation.
#[derive(Debug, Clone)]
pub enum Sampler {
    Multinomial { temperature: f32 },
    TopK { temperature: f32, k: u32 },
    TopP { temperature: f32, p: f32 },
    MinP { temperature: f32, p: f32 },
    TopKTopP { temperature: f32, k: u32, p: f32 },
    Dist { temperature: f32, num_tokens: u32 },
}

/// Output from a forward pass.
#[derive(Debug, Clone)]
pub enum Output {
    None,
    Tokens(Vec<u32>),
    Distributions(Vec<(Vec<u32>, Vec<f32>)>),
}

/// Forward pass request containing all inference parameters.
#[derive(Debug, Clone)]
pub struct ForwardPass {
    pub context_id: ContextId,
    pub context_lock_id: LockId,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub speculative_tokens: Vec<u32>,
    pub speculative_positions: Vec<u32>,
    pub attention_mask: Vec<Vec<u32>>,
    pub sampling_mask: Vec<u32>,
    pub samplers: Vec<(Vec<u32>, Sampler)>,
    pub adapter_id: Option<AdapterId>,
}

/// Messages for the inference actor.
#[derive(Debug)]
pub enum Message {
    /// Gets the KV page size for this model.
    GetKvPageSize {
        response: oneshot::Sender<Option<u32>>,
    },
   
    /// Executes a forward pass.
    ForwardPass {
        forward_pass: ForwardPass,
        queue_id: u32,
        response: oneshot::Sender<Output>,
    },
}

impl Message {
    /// Sends this message to the inference actor for the given model.
    pub fn send(self, model_idx: usize) -> Result<(), SendError> {
        ACTOR.send(model_idx, self)
    }
}

/// The inference actor manages forward pass execution for a model.
#[derive(Debug)]
struct InferenceActor {
    kv_page_size: u32,
}

impl Handle for InferenceActor {
    type Message = Message;

    fn new() -> Self {
        InferenceActor {
            kv_page_size: 16,
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetKvPageSize { response } => {
                let _ = response.send(Some(self.kv_page_size));
            }
            Message::ForwardPass { forward_pass: _, queue_id: _, response } => {
                
                
                // 1. translate context -> kv pages using cas
                
                // 2. 
                
                
                
                
                // TODO: Actually execute the forward pass via the model backend




                let output = Output::None;
                let _ = response.send(output);
            }
        }
    }
}
