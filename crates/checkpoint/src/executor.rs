pub mod arena;
pub mod iq_grid;
pub mod sink;
pub mod walk;

use std::collections::HashMap;
use std::path::Path;

use crate::error::Error;
use crate::executor::arena::ArenaBacking;
use crate::executor::sink::{MemorySink, TensorSink};
use crate::plan::LoadPlan;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HostStorage {
    pub arena: Vec<u8>,
    pub tensors: HashMap<String, Vec<u8>>,
}

pub enum Residency<'a> {
    Arena(&'a mut dyn ArenaBacking),
    Streaming,
}

pub struct Execution<'a> {
    plan: &'a LoadPlan,
    snapshot_dir: &'a Path,
    arena: Option<&'a mut dyn ArenaBacking>,
    streaming: bool,
    sink: Option<&'a mut dyn TensorSink>,
    progress: Option<&'a mut dyn FnMut(Progress<'_>)>,
    consume: Option<&'a crate::consume::SourceLedger>,
}

impl<'a> Execution<'a> {
    pub fn new(plan: &'a LoadPlan, snapshot_dir: &'a Path) -> Self {
        Self {
            plan,
            snapshot_dir,
            arena: None,
            streaming: false,
            sink: None,
            progress: None,
            consume: None,
        }
    }

    #[must_use]
    pub fn consuming(mut self, ledger: &'a crate::consume::SourceLedger) -> Self {
        self.consume = Some(ledger);
        self
    }

    #[must_use]
    pub fn arena(mut self, arena: &'a mut dyn ArenaBacking) -> Self {
        self.arena = Some(arena);
        self.streaming = false;
        self
    }

    #[must_use]
    pub fn streaming(mut self) -> Self {
        self.streaming = true;
        self.arena = None;
        self
    }

    #[must_use]
    pub fn sink(mut self, sink: &'a mut dyn TensorSink) -> Self {
        self.sink = Some(sink);
        self
    }

    #[must_use]
    pub fn progress(mut self, progress: &'a mut dyn FnMut(Progress<'_>)) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn run(self) -> Result<HostStorage, Error> {
        let Self {
            plan,
            snapshot_dir,
            arena,
            streaming,
            sink,
            progress,
            consume,
        } = self;
        let mut owned_sink = MemorySink::default();
        let mut unwatched = |_: Progress<'_>| {};
        let progress = progress.unwrap_or(&mut unwatched);
        let mut owned_arena = match (&arena, streaming) {
            (None, false) => vec![
                0u8;
                usize::try_from(plan.memory.arena_bytes()).map_err(|_| {
                    Error::Contract("persistent arena does not fit host address space".into())
                })?
            ],
            _ => Vec::new(),
        };
        {
            let mut owned_backing: &mut [u8] = &mut owned_arena;
            let residency = match (arena, streaming) {
                (Some(arena), _) => Residency::Arena(arena),
                (None, true) => Residency::Streaming,
                (None, false) => Residency::Arena(&mut owned_backing),
            };
            match sink {
                Some(sink) => walk::run(plan, snapshot_dir, residency, sink, progress, consume)?,
                None => {
                    walk::run(
                        plan,
                        snapshot_dir,
                        residency,
                        &mut owned_sink,
                        progress,
                        consume,
                    )?;
                }
            }
        }
        Ok(HostStorage {
            arena: owned_arena,
            tensors: owned_sink.tensors,
        })
    }
}

pub struct Progress<'a> {
    pub read_bytes: u64,
    pub total_read_bytes: u64,
    pub finalized: Option<&'a str>,
}
