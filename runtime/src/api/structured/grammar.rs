//! pie:structured/grammar - Grammar compilation for structured output

use std::sync::Arc;
use crate::api::pie;
use crate::linker::InstanceState;
use crate::inference::structured::grammar::Grammar as InternalGrammar;
use crate::inference::structured::json_schema::{builtin_json_grammar, json_schema_to_grammar, JsonSchemaOptions};
use crate::inference::structured::regex::regex_to_grammar;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

/// A compiled grammar that describes valid output structure.
#[derive(Debug)]
pub struct Grammar {
    /// The original source string (for compiled grammar cache keying).
    pub source: String,
    /// The parsed grammar AST.
    pub inner: Arc<InternalGrammar>,
}

impl pie::structured::grammar::Host for InstanceState {}

impl pie::structured::grammar::HostGrammar for InstanceState {
    async fn from_json_schema(
        &mut self,
        schema: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
        match json_schema_to_grammar(&schema, &JsonSchemaOptions::default()) {
            Ok(g) => {
                let grammar = Grammar {
                    source: schema,
                    inner: Arc::new(g),
                };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn json(&mut self) -> Result<Resource<Grammar>> {
        let g = builtin_json_grammar()?;
        let grammar = Grammar {
            source: "__builtin_json__".into(),
            inner: Arc::new(g),
        };
        Ok(self.ctx().table.push(grammar)?)
    }

    async fn from_regex(
        &mut self,
        pattern: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
        match regex_to_grammar(&pattern) {
            Ok(g) => {
                let grammar = Grammar {
                    source: pattern,
                    inner: Arc::new(g),
                };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn from_ebnf(
        &mut self,
        ebnf: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
        match InternalGrammar::from_ebnf(&ebnf, "root") {
            Ok(g) => {
                let grammar = Grammar {
                    source: ebnf,
                    inner: Arc::new(g),
                };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn drop(&mut self, this: Resource<Grammar>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
