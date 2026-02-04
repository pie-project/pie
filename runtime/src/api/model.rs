//! pie:core/model - Model and Tokenizer resources

use crate::api::pie;
use crate::instance::InstanceState;
use crate::model::{self, ModelInfo};
use crate::model::tokenizer::BytePairEncoder;
use anyhow::Result;
use std::sync::Arc;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Model resource - represents a reference to a registered model.
#[derive(Debug, Clone)]
pub struct Model {
    /// The model ID (for routing to the correct backend)
    pub model_id: usize,
    /// Cached model info
    pub info: Arc<ModelInfo>,
    /// Cached tokenizer
    pub tokenizer: Arc<BytePairEncoder>,
}

/// Tokenizer resource - for tokenization operations.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Model info (for stop tokens)
    pub info: Arc<ModelInfo>,
    /// The tokenizer
    pub tokenizer: Arc<BytePairEncoder>,
}

impl pie::core::model::Host for InstanceState {}

impl pie::core::model::HostModel for InstanceState {
    async fn load(&mut self, name: String) -> Result<Result<Resource<Model>, String>> {
        if let Some(model_id) = model::model_id(&name) {
            // Get cached model directly - no message passing needed
            if let Some(m) = model::get_model(model_id) {
                let model = Model {
                    model_id,
                    info: m.info,
                    tokenizer: m.tokenizer,
                };
                return Ok(Ok(self.ctx().table.push(model)?));
            }
        }
        Ok(Err(format!("Model '{}' not found", name)))
    }

    async fn chat_template(&mut self, this: Resource<Model>) -> Result<String> {
        let model = self.ctx().table.get(&this)?;
        Ok(model.info.prompt_template.clone())
    }

    async fn tokenizer(&mut self, this: Resource<Model>) -> Result<Resource<Tokenizer>> {
        let model = self.ctx().table.get(&this)?;
        let tokenizer = Tokenizer {
            info: Arc::clone(&model.info),
            tokenizer: Arc::clone(&model.tokenizer),
        };
        Ok(self.ctx().table.push(tokenizer)?)
    }

    async fn drop(&mut self, this: Resource<Model>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::model::HostTokenizer for InstanceState {
    async fn encode(&mut self, this: Resource<Tokenizer>, text: String) -> Result<Vec<u32>> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.tokenizer.encode_with_special_tokens(&text))
    }

    async fn decode(
        &mut self,
        this: Resource<Tokenizer>,
        tokens: Vec<u32>,
    ) -> Result<Result<String, String>> {
        let tokenizer = self.ctx().table.get(&this)?;
        match tokenizer.tokenizer.decode(&tokens) {
            Ok(text) => Ok(Ok(text)),
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn vocabs(&mut self, this: Resource<Tokenizer>) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.tokenizer.get_vocabs())
    }

    async fn split_regex(&mut self, this: Resource<Tokenizer>) -> Result<String> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.tokenizer.get_split_regex())
    }

    async fn special_tokens(&mut self, this: Resource<Tokenizer>) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.tokenizer.get_special_tokens())
    }

    async fn stop_tokens(&mut self, this: Resource<Tokenizer>) -> Result<Vec<u32>> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.info.stop_token_ids.clone())
    }

    async fn drop(&mut self, this: Resource<Tokenizer>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

