//! Where finalized tensors go.
//!
//! The executor used to return a `HashMap` of every tensor, which pinned its
//! caller's peak memory to the whole output — the reason `pie model convert`
//! held a model's worth of decoded bytes. A sink inverts the flow: the
//! executor hands each tensor over exactly once, at its
//! [`Finalize`](crate::plan::StorageInstr::Finalize), in schedule order, and
//! what happens to the bytes is the caller's business — collect them, spool
//! them to disk, write them into an artifact.

use std::collections::HashMap;

use crate::error::Error;

/// Receives each finalized tensor exactly once, in schedule order.
///
/// `bytes` is borrowed for the call: a sink that keeps them copies them,
/// which is the honest statement of who pays for residency.
pub trait TensorSink {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error>;
}

/// The collecting sink: every tensor, resident.
///
/// What [`execute_plan`](crate::executor::host::execute_plan) uses to keep
/// its shape — tests and the differential oracle want the whole output in
/// hand. Production callers with large outputs want a streaming sink
/// instead.
#[derive(Debug, Default)]
pub struct MemorySink {
    pub tensors: HashMap<String, Vec<u8>>,
}

impl TensorSink for MemorySink {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error> {
        self.tensors.insert(name.to_string(), bytes.to_vec());
        Ok(())
    }
}
