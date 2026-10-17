use std::cell::Cell;

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

#[derive(Debug, Default)]
pub struct At {
    pub region: Cell<u32>,
    pub run: Cell<u32>,
    pub tail: Cell<bool>,
}

impl At {
    #[must_use]
    pub fn new() -> At {
        At::default()
    }
}

#[derive(Debug)]
pub struct Cursor<'a> {
    at: u32,
    place: &'a At,
}

impl<'a> Cursor<'a> {
    #[must_use]
    pub fn new(place: &'a At) -> Cursor<'a> {
        place.region.set(0);
        place.run.set(0);
        Cursor { at: 0, place }
    }
}

impl Sink for Cursor<'_> {
    fn region_begin(&mut self, _region: &Region) {
        self.place.region.set(self.at);
        self.place.run.set(0);
        self.at += 1;
    }
    fn region_end(&mut self, _region: &Region) {}

    fn run(&mut self, run: u32, _runs: u32) {
        self.place.run.set(run);
        self.place.tail.set(false);
    }
    fn tail(&mut self, in_tail: bool) {
        self.place.tail.set(in_tail);
    }
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    fn fork(&mut self, _event: EventId) {}
    fn join(&mut self, _event: EventId) {}
}
