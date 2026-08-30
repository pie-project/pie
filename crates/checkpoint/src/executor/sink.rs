//! Where finalized tensors go.
//!
//! The executor hands each tensor over exactly once, at its
//! [`Finalize`](crate::plan::StorageInstr::Finalize), in schedule order, so
//! peak memory is the caller's choice: collect them, spool them to disk, or
//! write them into an artifact.

use std::collections::HashMap;

use crate::error::Error;

/// Receives each finalized tensor exactly once, in schedule order.
///
/// `bytes` is borrowed for the call: a sink that keeps them copies them.
pub trait TensorSink {
    /// Take one finalized tensor.
    fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error>;
}

/// The collecting sink: every tensor, resident. What
/// [`Execution`](crate::executor::Execution) uses by default; callers with
/// large outputs want a streaming sink instead.
#[derive(Debug, Default)]
pub struct MemorySink {
    /// Every tensor published so far, by name.
    pub tensors: HashMap<String, Vec<u8>>,
}

impl TensorSink for MemorySink {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error> {
        self.tensors.insert(name.to_string(), bytes.to_vec());
        Ok(())
    }
}
