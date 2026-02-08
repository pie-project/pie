//! pie:core/model - Model and Tokenizer resources

use crate::api::pie;
use crate::instance::InstanceState;
use crate::model;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Model resource - represents a reference to a registered model.
#[derive(Debug, Clone)]
pub struct Model {
    /// The model ID (for routing to the correct backend)
    pub model_id: usize,
    /// Cached model handle
    pub model: model::Model,
}

/// Tokenizer resource - for tokenization operations.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// The model handle (contains tokenizer + stop tokens)
    pub model: model::Model,
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

    async fn chat_template(&mut self, this: Resource<Model>) -> Result<String> {
        let model = self.ctx().table.get(&this)?;
        Ok(model.model.chat_template().to_string())
    }

    async fn tokenizer(&mut self, this: Resource<Model>) -> Result<Resource<Tokenizer>> {
        let model = self.ctx().table.get(&this)?;
        let tokenizer = Tokenizer {
            model: model.model.clone(),
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

    async fn special_tokens(&mut self, this: Resource<Tokenizer>) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.model.get_special_tokens())
    }

    async fn stop_tokens(&mut self, this: Resource<Tokenizer>) -> Result<Vec<u32>> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.model.stop_tokens().to_vec())
    }

    async fn drop(&mut self, this: Resource<Tokenizer>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
