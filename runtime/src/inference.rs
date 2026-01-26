//! Inference Service - Forward pass management for model execution
//!
//! This module provides a model-specific actor for managing forward passes
//! with configurable samplers, input tokens, and attention masks.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;

use crate::actor::{Handle, Actors, SendError};

/// Unique identifier for a forward pass.
pub type ForwardPassId = u64;

/// Global registry for inference actors.
static ACTOR: LazyLock<Actors<Message>> = LazyLock::new(Actors::new);

/// Spawns a new inference actor.
pub(crate) fn spawn() -> usize {
    ACTOR.spawn::<InferenceActor>()
}

/// Sampler configuration for token generation.
#[derive(Debug, Clone)]
pub enum Sampler {
    Multinomial { temperature: f32, seed: u32 },
    TopK { temperature: f32, k: u32 },
    TopP { temperature: f32, p: f32 },
    MinP { temperature: f32, p: f32 },
    TopKTopP { temperature: f32, k: u32, p: f32 },
    Dist { temperature: f32, seed: u32 },
}

/// Output from a forward pass.
#[derive(Debug, Clone)]
pub enum Output {
    None,
    Tokens(Vec<u32>),
    Distributions(Vec<(Vec<u32>, Vec<f32>)>),
}

/// Messages for the inference actor.
#[derive(Debug)]
pub enum Message {
    /// Creates a new forward pass.
    CreateForwardPass {
        response: oneshot::Sender<ForwardPassId>,
    },
    /// Gets the KV page size for a forward pass.
    GetKvPageSize {
        fp_id: ForwardPassId,
        response: oneshot::Sender<Option<u32>>,
    },
    /// Sets the context for a forward pass.
    SetContext {
        fp_id: ForwardPassId,
        context_id: u64,
    },
    /// Sets the input tokens for a forward pass.
    SetInputTokens {
        fp_id: ForwardPassId,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    },
    /// Sets speculative input tokens for a forward pass.
    SetSpeculativeTokens {
        fp_id: ForwardPassId,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    },
    /// Sets the attention mask for a forward pass.
    SetAttentionMask {
        fp_id: ForwardPassId,
        mask: Vec<Vec<u32>>,
    },
    /// Sets the sampling mask for a forward pass.
    SetSamplingMask {
        fp_id: ForwardPassId,
        mask: Vec<u32>,
    },
    /// Sets the sampler for specific indices.
    SetSampler {
        fp_id: ForwardPassId,
        indices: Vec<u32>,
        sampler: Sampler,
    },
    /// Sets the adapter for a forward pass.
    SetAdapter {
        fp_id: ForwardPassId,
        adapter_id: u64,
    },
    /// Executes the forward pass.
    Execute {
        fp_id: ForwardPassId,
        queue_id: u64,
        response: oneshot::Sender<Output>,
    },
    /// Destroys a forward pass.
    Destroy {
        fp_id: ForwardPassId,
    },
}

impl Message {
    /// Sends this message to the inference actor for the given model.
    pub fn send(self, model_idx: usize) -> Result<(), SendError> {
        ACTOR.send(model_idx, self)
    }
}

/// Internal representation of a forward pass.
#[derive(Debug, Clone, Default)]
struct ForwardPass {
    context_id: Option<u64>,
    tokens: Vec<u32>,
    positions: Vec<u32>,
    speculative_tokens: Vec<u32>,
    speculative_positions: Vec<u32>,
    attention_mask: Vec<Vec<u32>>,
    sampling_mask: Vec<u32>,
    samplers: Vec<(Vec<u32>, Sampler)>,
    adapter_id: Option<u64>,
}

/// The inference actor manages forward passes for a model.
#[derive(Debug)]
struct InferenceActor {
    forward_passes: dashmap::DashMap<ForwardPassId, ForwardPass>,
    next_id: AtomicU64,
    kv_page_size: u32,
}

impl Handle for InferenceActor {
    type Message = Message;

    fn new() -> Self {
        InferenceActor {
            forward_passes: dashmap::DashMap::new(),
            next_id: AtomicU64::new(1),
            kv_page_size: 16,
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::CreateForwardPass { response } => {
                let id = self.next_id();
                self.forward_passes.insert(id, ForwardPass::default());
                let _ = response.send(id);
            }
            Message::GetKvPageSize { fp_id, response } => {
                let result = if self.forward_passes.contains_key(&fp_id) {
                    Some(self.kv_page_size)
                } else {
                    None
                };
                let _ = response.send(result);
            }
            Message::SetContext { fp_id, context_id } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.context_id = Some(context_id);
                }
            }
            Message::SetInputTokens { fp_id, tokens, positions } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.tokens = tokens;
                    fp.positions = positions;
                }
            }
            Message::SetSpeculativeTokens { fp_id, tokens, positions } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.speculative_tokens = tokens;
                    fp.speculative_positions = positions;
                }
            }
            Message::SetAttentionMask { fp_id, mask } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.attention_mask = mask;
                }
            }
            Message::SetSamplingMask { fp_id, mask } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.sampling_mask = mask;
                }
            }
            Message::SetSampler { fp_id, indices, sampler } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.samplers.push((indices, sampler));
                }
            }
            Message::SetAdapter { fp_id, adapter_id } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.adapter_id = Some(adapter_id);
                }
            }
            Message::Execute { fp_id, queue_id: _, response } => {
                // TODO: Actually execute the forward pass via the model backend
                let output = if self.forward_passes.contains_key(&fp_id) {
                    Output::None
                } else {
                    Output::None
                };
                let _ = response.send(output);
            }
            Message::Destroy { fp_id } => {
                self.forward_passes.remove(&fp_id);
            }
        }
    }
}

impl InferenceActor {
    fn next_id(&self) -> ForwardPassId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}
