use crate::api::core::Model;
use crate::api::inferlet;
use crate::instance::InstanceState;
use crate::model::tokenizer::BytePairEncoder;
use anyhow::bail;
use std::sync::Arc;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub inner: Arc<BytePairEncoder>,
}

impl inferlet::core::tokenize::Host for InstanceState {
    async fn get_tokenizer(
        &mut self,
        model: Resource<Model>,
    ) -> anyhow::Result<Resource<Tokenizer>> {
        let inner = self.ctx().table.get(&model)?.info.tokenizer.clone();

        Ok(self.ctx().table.push(Tokenizer { inner })?)
    }
}

impl inferlet::core::tokenize::HostTokenizer for InstanceState {
    async fn tokenize(
        &mut self,
        this: Resource<Tokenizer>,
        text: String,
    ) -> anyhow::Result<Vec<u32>> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.inner.encode_with_special_tokens(&text))
    }

    async fn detokenize(
        &mut self,
        this: Resource<Tokenizer>,
        tokens: Vec<u32>,
    ) -> anyhow::Result<String> {
        let tokenizer = self.ctx().table.get(&this)?;
        let out = tokenizer.inner.decode(&tokens);

        if let Ok(out) = out {
            Ok(out)
        } else {
            println!("Failed to decode tokens: {:?}", out);
            bail!("Failed to decode tokens: {:?}", out);
        }
    }

    async fn get_vocabs(
        &mut self,
        this: Resource<Tokenizer>,
    ) -> anyhow::Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.inner.get_vocabs())
    }

    async fn drop(&mut self, this: Resource<Tokenizer>) -> anyhow::Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
