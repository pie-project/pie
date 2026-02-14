//! pie:core/model - Model and Tokenizer resources

use std::sync::Arc;
use crate::api::pie;
use crate::model::chat_templates::{self as ct};
use crate::linker::InstanceState;
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

    async fn chat_template(&mut self, this: Resource<Model>) -> Result<pie::core::model::ChatTemplate> {
        let model = self.ctx().table.get(&this)?;
        let ct = model.model.chat_template();

        let sys_handling = match ct.system_handling {
            ct::SystemHandling::Standalone => pie::core::model::SystemHandling::Standalone,
            ct::SystemHandling::MergeWithUser => pie::core::model::SystemHandling::MergeWithUser,
            ct::SystemHandling::BarePrepend => pie::core::model::SystemHandling::BarePrepend,
        };

        Ok(pie::core::model::ChatTemplate {
            start_token: ct.start_token.to_string(),
            stop_tokens: ct.stop_tokens.iter().map(|s| s.to_string()).collect(),
            role_prefixes: ct.role_prefixes.iter().map(|(r, p)| (r.to_string(), p.to_string())).collect(),
            role_suffixes: ct.role_suffixes.iter().map(|(r, s)| (r.to_string(), s.to_string())).collect(),
            system_handling: sys_handling,
            system_separator: ct.system_separator.to_string(),
            generation_header: ct.generation_header.to_string(),
            thinking_prefix: ct.thinking_prefix.to_string(),
            thinking_suffix: ct.thinking_suffix.to_string(),
            tool_call_template: ct.tool_call_template.to_string(),
            tool_calls_prefix: ct.tool_calls_prefix.to_string(),
            tool_calls_suffix: ct.tool_calls_suffix.to_string(),
            tool_response_role: ct.tool_response_role.to_string(),
            tool_response_prefix: ct.tool_response_prefix.to_string(),
            tool_response_suffix: ct.tool_response_suffix.to_string(),
        })
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


    async fn drop(&mut self, this: Resource<Tokenizer>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
