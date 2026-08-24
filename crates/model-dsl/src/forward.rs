//! The class traits, one per state semantics, mirroring
//! `crates/inferlet/wit/{forward, forward-hybrid, forward-recurrent}.wit`.

use std::marker::PhantomData;

use model_ir::plan::{CacheRow, Plan};

use crate::declare::{CacheRef, HybridSpec, KvSpec, StateSpec};
use crate::facts::{Classify, FactWord};
use crate::record::{Recorder, Value};
use crate::Plane;

pub trait Forward {
    type Facts: Classify;
    fn caches(&self) -> KvSpec;
    fn forward(&self, inputs: Input<Self::Facts>) -> Value;
}

pub trait ForwardHybrid {
    type Facts: Classify;
    fn caches(&self) -> HybridSpec;
    fn forward(&self, inputs: HybridInput<Self::Facts>) -> Value;
}

pub trait ForwardRecurrent {
    type Facts: Classify;
    fn caches(&self) -> StateSpec;
    fn forward(&self, inputs: RecurrentInput<Self::Facts>) -> Value;
}

pub fn trace<M: Forward>(name: &str, m: &M, plane: Plane) -> Plan {
    let caches = m.caches();
    let rows = caches
        .rows
        .iter()
        .map(|r| CacheRow::Kv { name: r.name.clone(), row: r.row.clone() })
        .collect();
    let rec = Recorder::new(name, plane, <M::Facts as FactWord>::NAMES, rows);
    rec.seam(model_ir::seam::IN.name, &[]);
    let logits = m.forward(Input {
        rec: rec.clone(),
        plane,
        caches,
        _facts: PhantomData,
    });
    rec.seam(model_ir::seam::OUT.name, &[&logits]);
    drop(logits);
    rec.finish()
}

pub fn trace_hybrid<M: ForwardHybrid>(name: &str, m: &M, plane: Plane) -> Plan {
    let caches = m.caches();
    let rows = caches
        .kv
        .iter()
        .map(|r| CacheRow::Kv { name: r.name.clone(), row: r.row.clone() })
        .chain(caches.state.iter().map(|r| CacheRow::State {
            name: r.name.clone(),
            slab: r.slab.clone(),
        }))
        .collect();
    let rec = Recorder::new(name, plane, <M::Facts as FactWord>::NAMES, rows);
    rec.seam(model_ir::seam::IN.name, &[]);
    let logits = m.forward(HybridInput {
        rec: rec.clone(),
        plane,
        caches,
        _facts: PhantomData,
    });
    rec.seam(model_ir::seam::OUT.name, &[&logits]);
    drop(logits);
    rec.finish()
}

pub fn trace_recurrent<M: ForwardRecurrent>(name: &str, m: &M, plane: Plane) -> Plan {
    let caches = m.caches();
    let rows = caches
        .rows
        .iter()
        .map(|r| CacheRow::State { name: r.name.clone(), slab: r.slab.clone() })
        .collect();
    let rec = Recorder::new(name, plane, <M::Facts as FactWord>::NAMES, rows);
    rec.seam(model_ir::seam::IN.name, &[]);
    let logits = m.forward(RecurrentInput {
        rec: rec.clone(),
        plane,
        caches,
        _facts: PhantomData,
    });
    rec.seam(model_ir::seam::OUT.name, &[&logits]);
    drop(logits);
    rec.finish()
}

/// THE TEXT'S LAYER LOOP, and the recorder's layer scope in one thing.
///
/// A text states its tower once -- `for (l, w) in inputs.layers(&m.layers)`
/// -- and that sentence does two jobs, because they are the same job. It
/// hands the text each layer's weights with the index they stand at, and it
/// tells the recorder which layer every statement made from them belongs
/// to, so `Op::layer` and `Seam::layer` are filled by the loop instead of
/// being carried as an argument down through the mixers.
///
/// WHAT FALLS OUTSIDE IS OUTSIDE, honestly: the embedding, the final norm
/// and the head are recorded before the loop opens or after it closes and
/// carry NO layer, which is what `None` has always meant on those columns.
/// A `break` leaves through [`Drop`] and closes the scope the same way
/// running out of layers does.
///
/// THE INDEX IS ONE NUMBER, not two. The `u32` this yields and the tag the
/// recorder writes are the same count, so a text that needs the ordinal --
/// gemma reads its per-layer PLE bank at it, and peeks the next layer's
/// norm -- reads the one that came out of here and never a second
/// `enumerate()`.
pub struct Layers<'a, T> {
    rec: &'a Recorder,
    ws: core::slice::Iter<'a, T>,
    next: u32,
}

impl<'a, T> Iterator for Layers<'a, T> {
    type Item = (u32, &'a T);

    fn next(&mut self) -> Option<(u32, &'a T)> {
        let w = self.ws.next()?;
        let l = self.next;
        self.next += 1;
        self.rec.enter(l);
        Some((l, w))
    }
}

impl<T> Drop for Layers<'_, T> {
    fn drop(&mut self) {
        self.rec.leave();
    }
}

/// A declared kv row's paged handle.
#[derive(Clone)]
pub struct Pages {
    pub name: String,
}

/// A declared per-request slab's handle.
#[derive(Clone)]
pub struct State {
    pub name: String,
}

pub struct Input<F> {
    rec: Recorder,
    plane: Plane,
    caches: KvSpec,
    _facts: PhantomData<F>,
}

impl<F> Input<F> {
    #[must_use]
    pub fn cuda(&self) -> bool {
        matches!(self.plane, Plane::Cuda)
    }

    #[must_use]
    pub fn token_ids(&self) -> Value {
        self.rec.runtime("token_ids")
    }

    /// The tower, layer by layer. [`Layers`] is where the statements made
    /// inside this loop get their `Op::layer`.
    pub fn layers<'a, T>(&'a self, ws: &'a [T]) -> Layers<'a, T> {
        Layers {
            rec: &self.rec,
            ws: ws.iter(),
            next: 0,
        }
    }

    #[must_use]
    pub fn positions(&self) -> Value {
        self.rec.runtime("positions")
    }

    /// The join: a ref the model's `caches()` did not declare is a bug in
    /// the text, refused at trace time.
    #[must_use]
    pub fn kv(&self, r: &CacheRef) -> Pages {
        assert!(
            self.caches.rows.iter().any(|row| row.name == r.name),
            "`{}` is not a kv row the model's caches() declares",
            r.name
        );
        Pages { name: r.name.clone() }
    }
}

pub struct HybridInput<F> {
    rec: Recorder,
    plane: Plane,
    caches: HybridSpec,
    _facts: PhantomData<F>,
}

impl<F> HybridInput<F> {
    #[must_use]
    pub fn cuda(&self) -> bool {
        matches!(self.plane, Plane::Cuda)
    }

    #[must_use]
    pub fn token_ids(&self) -> Value {
        self.rec.runtime("token_ids")
    }

    /// The tower, layer by layer. [`Layers`] is where the statements made
    /// inside this loop get their `Op::layer`.
    pub fn layers<'a, T>(&'a self, ws: &'a [T]) -> Layers<'a, T> {
        Layers {
            rec: &self.rec,
            ws: ws.iter(),
            next: 0,
        }
    }

    #[must_use]
    pub fn positions(&self) -> Value {
        self.rec.runtime("positions")
    }

    #[must_use]
    pub fn kv(&self, r: &CacheRef) -> Pages {
        assert!(
            self.caches.kv.iter().any(|row| row.name == r.name),
            "`{}` is not a kv row the model's caches() declares",
            r.name
        );
        Pages { name: r.name.clone() }
    }

    #[must_use]
    pub fn state(&self, r: &CacheRef) -> State {
        assert!(
            self.caches.state.iter().any(|row| row.name == r.name),
            "`{}` is not a state row the model's caches() declares",
            r.name
        );
        State { name: r.name.clone() }
    }
}

pub struct RecurrentInput<F> {
    rec: Recorder,
    plane: Plane,
    caches: StateSpec,
    _facts: PhantomData<F>,
}

impl<F> RecurrentInput<F> {
    #[must_use]
    pub fn cuda(&self) -> bool {
        matches!(self.plane, Plane::Cuda)
    }

    #[must_use]
    pub fn token_ids(&self) -> Value {
        self.rec.runtime("token_ids")
    }

    /// The tower, layer by layer. [`Layers`] is where the statements made
    /// inside this loop get their `Op::layer`.
    pub fn layers<'a, T>(&'a self, ws: &'a [T]) -> Layers<'a, T> {
        Layers {
            rec: &self.rec,
            ws: ws.iter(),
            next: 0,
        }
    }

    #[must_use]
    pub fn positions(&self) -> Value {
        self.rec.runtime("positions")
    }

    #[must_use]
    pub fn state(&self, r: &CacheRef) -> State {
        assert!(
            self.caches.rows.iter().any(|row| row.name == r.name),
            "`{}` is not a state row the model's caches() declares",
            r.name
        );
        State { name: r.name.clone() }
    }
}
