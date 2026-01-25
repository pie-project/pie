use std::sync::LazyLock;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::task;

/// Unique identifier for a forward pass.
pub type ForwardPassId = u64;

/// Model-indexed dispatcher for inference services.
static INFERENCE_DISPATCHER: LazyLock<InferenceDispatcher> = LazyLock::new(|| InferenceDispatcher {
    services: boxcar::Vec::new(),
});

#[derive(Debug, Error)]
pub enum InferenceDispatchError {
    #[error("Invalid model index: {0}")]
    InvalidModelIndex(usize),
}

#[derive(Debug)]
struct InferenceDispatcher {
    services: boxcar::Vec<mpsc::UnboundedSender<Command>>,
}

/// Installs a new inference service for the given model ID.
/// This is only called internally by model_new::install_model.
pub(super) fn install_service(model_id: usize) {
    let svc = InferenceService::new();
    let (tx, mut rx) = mpsc::unbounded_channel();
    
    let idx = INFERENCE_DISPATCHER.services.push(tx);
    debug_assert_eq!(idx, model_id, "Inference service ID mismatch");

    task::spawn(async move {
        let mut svc = svc;
        while let Some(cmd) = rx.recv().await {
            svc.handle(cmd).await;
        }
    });
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

/// Defines the set of operations available for the inference service.
#[derive(Debug)]
pub enum Command {
    /// Creates a new forward pass for the given model.
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

impl Command {
    /// Dispatches this command to the inference service for the given model.
    pub fn dispatch(self, model_id: usize) -> Result<(), InferenceDispatchError> {
        let tx = INFERENCE_DISPATCHER
            .services
            .get(model_id)
            .ok_or(InferenceDispatchError::InvalidModelIndex(model_id))?;
        tx.send(self).unwrap();
        Ok(())
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

/// The inference service manages forward passes for a model.
#[derive(Debug)]
struct InferenceService {
    forward_passes: dashmap::DashMap<ForwardPassId, ForwardPass>,
    next_id: std::sync::atomic::AtomicU64,
    kv_page_size: u32,
}

impl InferenceService {
    /// Creates a new `InferenceService`.
    fn new() -> Self {
        InferenceService {
            forward_passes: dashmap::DashMap::new(),
            next_id: std::sync::atomic::AtomicU64::new(1),
            kv_page_size: 16, // Default page size
        }
    }

    /// Generates a new unique forward pass ID.
    fn next_id(&self) -> ForwardPassId {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    async fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::CreateForwardPass { response } => {
                let id = self.next_id();
                self.forward_passes.insert(id, ForwardPass::default());
                let _ = response.send(id);
            }
            Command::GetKvPageSize { fp_id, response } => {
                let result = if self.forward_passes.contains_key(&fp_id) {
                    Some(self.kv_page_size)
                } else {
                    None
                };
                let _ = response.send(result);
            }
            Command::SetContext { fp_id, context_id } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.context_id = Some(context_id);
                }
            }
            Command::SetInputTokens {
                fp_id,
                tokens,
                positions,
            } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.tokens = tokens;
                    fp.positions = positions;
                }
            }
            Command::SetSpeculativeTokens {
                fp_id,
                tokens,
                positions,
            } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.speculative_tokens = tokens;
                    fp.speculative_positions = positions;
                }
            }
            Command::SetAttentionMask { fp_id, mask } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.attention_mask = mask;
                }
            }
            Command::SetSamplingMask { fp_id, mask } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.sampling_mask = mask;
                }
            }
            Command::SetSampler {
                fp_id,
                indices,
                sampler,
            } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.samplers.push((indices, sampler));
                }
            }
            Command::SetAdapter { fp_id, adapter_id } => {
                if let Some(mut fp) = self.forward_passes.get_mut(&fp_id) {
                    fp.adapter_id = Some(adapter_id);
                }
            }
            Command::Execute {
                fp_id,
                queue_id: _,
                response,
            } => {
                // TODO: Actually execute the forward pass via the model backend
                // For now, just return empty output
                let output = if self.forward_passes.contains_key(&fp_id) {
                    Output::None
                } else {
                    Output::None
                };
                let _ = response.send(output);
            }
            Command::Destroy { fp_id } => {
                self.forward_passes.remove(&fp_id);
            }
        }
    }
}
