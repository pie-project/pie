//! pie:core/model - Model and Tokenizer resources

use crate::api::pie;
use crate::instance::InstanceState;
use crate::model::{self, Message, ModelInfo};
use crate::model::tokenizer::BytePairEncoder;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// Model resource - represents a reference to a registered model.
#[derive(Debug, Clone)]
pub struct Model {
    /// The model service index (for routing to the correct ModelActor)
    pub service_id: usize,
    /// Cached model info
    pub info: Arc<ModelInfo>,
    /// Cached tokenizer (if available)
    pub tokenizer: Option<Arc<BytePairEncoder>>,
}

/// Tokenizer resource - for tokenization operations.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// The model service index 
    pub service_id: usize,
}

impl pie::core::model::Host for InstanceState {}

impl pie::core::model::HostModel for InstanceState {
    async fn new(&mut self, name: String) -> Result<Resource<Model>> {
        if let Some(service_id) = model::model_service_id(&name) {
            // Get model info from the actor
            let (tx, rx) = oneshot::channel();
            Message::GetInfo { response: tx }.send(service_id)?;
            let info = rx.await?;

            let model = Model {
                service_id,
                info: Arc::new(info),
                tokenizer: None,
            };
            return Ok(self.ctx().table.push(model)?);
        }
        anyhow::bail!("Model '{}' not found", name)
    }

    async fn chat_template(&mut self, this: Resource<Model>) -> Result<String> {
        let model = self.ctx().table.get(&this)?;
        Ok(model.info.prompt_template.clone())
    }

    async fn tokenizer(&mut self, this: Resource<Model>) -> Result<Resource<Tokenizer>> {
        let model = self.ctx().table.get(&this)?;
        let tokenizer = Tokenizer {
            service_id: model.service_id,
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
        let service_id = tokenizer.service_id;

        let (tx, rx) = oneshot::channel();
        Message::Tokenize { text, response: tx }.send(service_id)?;
        Ok(rx.await?)
    }

    async fn decode(
        &mut self,
        this: Resource<Tokenizer>,
        tokens: Vec<u32>,
    ) -> Result<Result<String, String>> {
        let tokenizer = self.ctx().table.get(&this)?;
        let service_id = tokenizer.service_id;

        let (tx, rx) = oneshot::channel();
        Message::Detokenize { tokens, response: tx }.send(service_id)?;
        let text = rx.await?;
        Ok(Ok(text))
    }

    async fn vocabs(&mut self, this: Resource<Tokenizer>) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        let service_id = tokenizer.service_id;

        let (tx, rx) = oneshot::channel();
        Message::GetVocabs { response: tx }.send(service_id)?;
        Ok(rx.await?)
    }

    async fn split_regex(&mut self, this: Resource<Tokenizer>) -> Result<String> {
        let tokenizer = self.ctx().table.get(&this)?;
        let service_id = tokenizer.service_id;

        let (tx, rx) = oneshot::channel();
        Message::GetSplitRegex { response: tx }.send(service_id)?;
        Ok(rx.await?)
    }

    async fn special_tokens(&mut self, this: Resource<Tokenizer>) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        let service_id = tokenizer.service_id;

        let (tx, rx) = oneshot::channel();
        Message::GetSpecialTokens { response: tx }.send(service_id)?;
        Ok(rx.await?)
    }

    async fn stop_tokens(&mut self, this: Resource<Tokenizer>) -> Result<Vec<u32>> {
        let tokenizer = self.ctx().table.get(&this)?;
        let service_id = tokenizer.service_id;

        let (tx, rx) = oneshot::channel();
        Message::GetStopTokens { response: tx }.send(service_id)?;
        Ok(rx.await?)
    }

    async fn drop(&mut self, this: Resource<Tokenizer>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
