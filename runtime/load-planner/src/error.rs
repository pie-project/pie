use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("internal error: {0}")]
    Internal(String),
}
