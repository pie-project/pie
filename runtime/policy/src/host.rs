use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pie_plex::Document;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StagedAction {
    pub id: u64,
    pub method: String,
    pub args: Document,
}

pub trait QueryHandler: Send + Sync + 'static {
    fn query(&self, method: &str, args: &Document) -> Result<Document, QueryError>;
}

#[derive(Debug, Default)]
pub struct RejectingQueryHandler;

impl QueryHandler for RejectingQueryHandler {
    fn query(&self, method: &str, _args: &Document) -> Result<Document, QueryError> {
        Err(QueryError::Unsupported(method.to_owned()))
    }
}

#[derive(Clone, Default)]
pub struct DictionaryQueryHandler {
    responses: Arc<Mutex<BTreeMap<String, Document>>>,
}

impl DictionaryQueryHandler {
    pub fn new(responses: BTreeMap<String, Document>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
        }
    }

    pub fn insert(&self, method: impl Into<String>, response: Document) {
        self.responses
            .lock()
            .unwrap()
            .insert(method.into(), response);
    }
}

impl QueryHandler for DictionaryQueryHandler {
    fn query(&self, method: &str, _args: &Document) -> Result<Document, QueryError> {
        self.responses
            .lock()
            .unwrap()
            .get(method)
            .cloned()
            .ok_or_else(|| QueryError::Unsupported(method.to_owned()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum QueryError {
    #[error("unsupported query method {0}")]
    Unsupported(String),
    #[error("query handler failed: {0}")]
    Handler(String),
}
