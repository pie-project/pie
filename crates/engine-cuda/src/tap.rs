//! Taps: a lane's pages read out to host slots while the fire that writes
//! them runs, a layer's planes at a time. A tapped fire is walked eagerly;
//! as the walk leaves a layer, everything on the stream before that point has
//! written the layer's rows, so an event there lets the tap stream copy the
//! layer's planes out while the next layer computes, and a host callback
//! behind the copy hands their bytes on.
//!
//! A layer's group is told only once the walk has moved past it, and a walk
//! that comes back to a layer it has left fails the tap: the group already
//! gone may have missed that layer's rows.
#![allow(unsafe_code)]

use std::cell::Cell;

use engine::{TapSink, Tapped};
use poem_ir::{Def, Operands, Trace};

use crate::device::graph::Event;
use crate::error::Result;

/// Each cache row's group: the last layer of the nodes that reference it,
/// since a row is whole once its last writer has run. `None` where the
/// trace's nodes are not in layer order (a tap then could not know when a
/// layer is done), or no node references a row.
#[must_use]
pub fn row_layers(trace: &Trace) -> Option<Vec<Option<u32>>> {
    let mut rows = vec![None; trace.caches.len()];
    let mut last = 0u32;
    let mut ids = Vec::new();
    for node in &trace.nodes {
        let Some(layer) = node.layer else {
            continue;
        };
        if layer < last {
            return None;
        }
        last = layer;
        ids.clear();
        node.op.inputs(&mut ids);
        node.op.outputs(&mut ids);
        for id in &ids {
            if let Some(Def::Cache(row)) = trace.values.get(id.0 as usize).map(|decl| &decl.def) {
                let row = &mut rows[*row as usize];
                *row = Some(row.map_or(layer, |was: u32| was.max(layer)));
            }
        }
    }
    Some(rows)
}

/// One group of a fire's taps: the copies for every tapped lane, and for
/// each lane (by tag) where its pages' planes of the group land, page after
/// page.
pub struct Group {
    pub index: u32,
    pub layer: u32,
    pub copies: Vec<(u64, u64, usize)>,
    pub lands: Vec<(u64, Vec<(u64, usize)>)>,
}

/// A fire's taps as its walk goes: the groups in layer order, the next not
/// yet issued, and the layer the walk is in.
pub struct Tapping {
    groups: Vec<Group>,
    next: Cell<usize>,
    at: Cell<Option<u32>>,
    torn: Cell<bool>,
    main: *mut core::ffi::c_void,
    tap: *mut core::ffi::c_void,
    sink: TapSink,
}

impl Tapping {
    #[must_use]
    pub fn new(
        groups: Vec<Group>,
        main: *mut core::ffi::c_void,
        tap: *mut core::ffi::c_void,
        sink: TapSink,
    ) -> Self {
        Self {
            groups,
            next: Cell::new(0),
            at: Cell::new(None),
            torn: Cell::new(false),
            main,
            tap,
            sink,
        }
    }

    /// The walk has enqueued a node of `layer`: the layers before it are
    /// done.
    pub fn saw(&self, layer: Option<u32>) {
        let Some(layer) = layer else {
            return;
        };
        if self.torn.get() {
            return;
        }
        match self.at.get() {
            Some(at) if layer < at => {
                self.fail("the walk came back to a layer it had left");
                return;
            }
            Some(at) if layer > at => self.issue_below(layer),
            _ => {}
        }
        self.at.set(Some(layer));
    }

    /// The walk is over: every group left is done.
    pub fn finish(&self) {
        if !self.torn.get() {
            self.issue_below(u32::MAX);
        }
    }

    /// The walk failed: what is not yet told never will be.
    pub fn fail(&self, why: &str) {
        if self.torn.replace(true) {
            return;
        }
        let group = self
            .groups
            .get(self.next.get())
            .map_or(u32::MAX, |group| group.index);
        let tags = self
            .groups
            .first()
            .map(|group| group.lands.iter().map(|(tag, _)| *tag));
        for tag in tags.into_iter().flatten() {
            (self.sink)(Tapped {
                tag,
                group,
                bytes: Err(why.to_string()),
            });
        }
    }

    fn issue_below(&self, layer: u32) {
        while let Some(group) = self.groups.get(self.next.get()) {
            if group.layer >= layer {
                return;
            }
            if let Err(error) = self.issue(group) {
                self.fail(&format!("reading a layer out: {error}"));
                return;
            }
            self.next.set(self.next.get() + 1);
        }
    }

    fn issue(&self, group: &Group) -> Result<()> {
        // Dropped once enqueued: the driver frees it when it is done.
        let done = Event::new()?;
        done.record(self.main)?;
        done.wait(self.tap)?;
        crate::device::copy_batch(self.tap, true, &group.copies)?;
        let (lands, sink, index) = (group.lands.clone(), self.sink.clone(), group.index);
        crate::device::ctx::host_fn_on(
            self.tap,
            Box::new(move || {
                for (tag, spans) in lands {
                    let mut bytes = Vec::with_capacity(spans.iter().map(|(_, len)| len).sum());
                    for (at, len) in spans {
                        // SAFETY: a span of the pinned host tier the copies
                        // before this callback on the tap stream filled; the
                        // tier lives as long as the load.
                        bytes.extend_from_slice(unsafe {
                            std::slice::from_raw_parts(at as *const u8, len)
                        });
                    }
                    sink(Tapped {
                        tag,
                        group: index,
                        bytes: Ok(bytes),
                    });
                }
            }),
        )
    }
}
