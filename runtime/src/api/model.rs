//! pie:core/model - Model and Tokenizer resources

use crate::api::pie;
use crate::instance::InstanceState;
use crate::model;
use anyhow::Result;
use std::sync::Arc;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Model resource - represents a reference to a registered model.
#[derive(Debug, Clone)]
pub struct Model {
    /// The model ID (for routing to the correct backend)
    pub model_id: usize,
    /// Cached model handle
    pub model: Arc<model::Model>,
}

/// Tokenizer resource - for tokenization operations.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// The model handle (contains tokenizer + stop tokens)
    pub model: Arc<model::Model>,
}

impl pie::core::model::Host for InstanceState {}

impl pie::core::model::HostModel for InstanceState {
    async fn load(&mut self, name: String) -> Result<Result<Resource<Model>, String>> {
        if let Some(model_id) = model::get_model_id(&name) {
            if let Some(m) = model::get_model(model_id) {
                let model = Model {
                    model_id,
                    model: m.clone(),
                };
                return Ok(Ok(self.ctx().table.push(model)?));
            }
        }
        Ok(Err(format!("Model '{}' not found", name)))
    }

    async fn tokenizer(&mut self, this: Resource<Model>) -> Result<Resource<Tokenizer>> {
        let model = self.ctx().table.get(&this)?;
        let tokenizer = Tokenizer {
            model: model.model.clone(),
        };
        Ok(self.ctx().table.push(tokenizer)?)
    }

    async fn default_system_speculation(&mut self, this: Resource<Model>) -> Result<bool> {
        let model = self.ctx().table.get(&this)?;
        // The effective "speculate by default?" decision the SDK reflects: the
        // model must support a system drafter AND the operator must have opted
        // in (`enable_system_speculation`, default off). The runtime owns this
        // decision; the SDK only requests system drafts when both hold. (Manual
        // drafts are a separate path, gated in api/inference.rs.)
        Ok(model.model.system_speculation_supported() && model.model.enable_system_speculation())
    }

    async fn output_vocab_size(&mut self, this: Resource<Model>) -> Result<u32> {
        let model = self.ctx().table.get(&this)?;
        Ok(model.model.vocab_size())
    }

    async fn drop(&mut self, this: Resource<Model>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::model::HostTokenizer for InstanceState {
    async fn encode(&mut self, this: Resource<Tokenizer>, text: String) -> Result<Vec<u32>> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.model.tokenize(&text))
    }

    async fn decode(
        &mut self,
        this: Resource<Tokenizer>,
        tokens: Vec<u32>,
    ) -> Result<Result<String, String>> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(Ok(tokenizer.model.detokenize(&tokens)))
    }

    async fn vocabs(&mut self, this: Resource<Tokenizer>) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.model.get_vocabs())
    }

    async fn split_regex(&mut self, this: Resource<Tokenizer>) -> Result<String> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.model.get_split_regex())
    }

    async fn special_tokens(
        &mut self,
        this: Resource<Tokenizer>,
    ) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.model.get_special_tokens())
    }

    async fn drop(&mut self, this: Resource<Tokenizer>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
