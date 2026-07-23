//! External tokenizer format loaders.
//!
//! This module owns file-format detection and delegates format-specific
//! parsing to [`huggingface`] or [`tiktoken`]. Encoding and decoding remain in
//! the crate root.

use std::path::Path;

use anyhow::Result;

use crate::Tokenizer;

pub mod huggingface;
pub mod tiktoken;

/// Load a supported tokenizer artifact.
pub fn from_file(path: &Path) -> Result<Tokenizer> {
    if is_tiktoken_path(path) {
        tiktoken::from_file(path)
    } else {
        huggingface::from_file(path)
    }
}

fn is_tiktoken_path(path: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("tiktoken.model")
        || path.extension().and_then(|extension| extension.to_str()) == Some("tiktoken")
}
