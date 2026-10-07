use poem_compiler::{Lowering, Region};

pub use poem_compiler::EventId;

pub trait Sink {
    fn region_begin(&mut self, region: &Region);

    fn region_end(&mut self, region: &Region);

    fn run(&mut self, run: u32, runs: u32);

    fn cond_begin(&mut self, lowering: &Lowering);

    fn cond_arm(&mut self, arm: u8);

    fn cond_end(&mut self);

    fn tail(&mut self, _in_tail: bool) {}

    fn fork(&mut self, event: EventId);

    fn join(&mut self, event: EventId);
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EagerSink;

impl Sink for EagerSink {
    fn region_begin(&mut self, _region: &Region) {}
    fn region_end(&mut self, _region: &Region) {}
    fn run(&mut self, _run: u32, _runs: u32) {}
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    fn fork(&mut self, _event: EventId) {}
    fn join(&mut self, _event: EventId) {}
}
