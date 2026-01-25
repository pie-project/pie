//! pie:core/model - Model and Tokenizer resources

use crate::api::pie;
use crate::instance::InstanceState;
use crate::model::tokenizer::BytePairEncoder;
use crate::model::ModelInfo;
use anyhow::Result;
use std::sync::Arc;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Debug, Clone)]
pub struct Model {
    pub service_id: usize,
    pub info: Arc<ModelInfo>,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub inner: Arc<BytePairEncoder>,
}

impl pie::core::model::Host for InstanceState {}

impl pie::core::model::HostModel for InstanceState {
    async fn new(&mut self, name: String) -> Result<Resource<Model>> {
        if let Some(service_id) = crate::model::model_service_id(&name) {
            let (tx, rx) = tokio::sync::oneshot::channel();
            crate::model::Command::GetInfo { response: tx }.dispatch(service_id)?;
            let info = rx.await?;
            let model = Model {
                service_id,
                info: Arc::new(info),
            };
            return Ok(self.ctx().table.push(model)?);
        }
        anyhow::bail!("Model '{}' not found", name)
    }

    async fn prompt_template(&mut self, this: Resource<Model>) -> Result<String> {
        let model = self.ctx().table.get(&this)?;
        Ok(model.info.prompt_template.clone())
    }

    async fn stop_tokens(&mut self, this: Resource<Model>) -> Result<Vec<String>> {
        let model = self.ctx().table.get(&this)?;
        Ok(model.info.prompt_stop_tokens.clone())
    }

    async fn tokenizer(&mut self, this: Resource<Model>) -> Result<Resource<Tokenizer>> {
        let model = self.ctx().table.get(&this)?;
        let tokenizer = Tokenizer {
            inner: model.info.tokenizer.clone(),
        };
        Ok(self.ctx().table.push(tokenizer)?)
    }

    async fn drop(&mut self, this: Resource<Model>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::model::HostTokenizer for InstanceState {
    async fn tokenize(&mut self, this: Resource<Tokenizer>, text: String) -> Result<Vec<u32>> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.inner.encode_with_special_tokens(&text))
    }

    async fn detokenize(&mut self, this: Resource<Tokenizer>, tokens: Vec<u32>) -> Result<String> {
        let tokenizer = self.ctx().table.get(&this)?;
        tokenizer
            .inner
            .decode(&tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {:?}", e))
    }

    async fn vocabs(&mut self, this: Resource<Tokenizer>) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.inner.get_vocabs())
    }

    async fn split_regex(&mut self, this: Resource<Tokenizer>) -> Result<String> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.inner.get_split_regex())
    }

    async fn special_tokens(&mut self, this: Resource<Tokenizer>) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.inner.get_special_tokens())
    }

    async fn drop(&mut self, this: Resource<Tokenizer>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
