//! The model-facing entry: classify a request into facts, declare cache
//! spaces, trace the forward pass, and hand back a checked `Plan`.
//! Re-imagined from the old `forward.rs`: the inputs handle is typed now —
//! tokens and positions arrive as `[Tokens] i32` values, cache geometry as
//! declared `RuntimeInput::Geometry` inputs (§7) — and `qo_indptr` is gone
//! outright, because raggedness is ambient (§5): every fire-aligned value is
//! viewable through the fire's shared indptr, so there is nothing to attach.

use std::marker::PhantomData;

use model_ir::{CacheRow, Dim, Dtype, GeomKind, Plan, Plane, RuntimeInput, Ty, ValueId};

use crate::ops;
use crate::record::{Recorder, Value};
use crate::seam;

/// One request's shape facts, as the engine states them per fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    query_len: u32,
    custom_mask: bool,
}

impl Request {
    #[must_use]
    pub fn new(query_len: u32, custom_mask: bool) -> Request {
        Request {
            query_len,
            custom_mask,
        }
    }

    #[must_use]
    pub fn query_len(&self) -> u32 {
        self.query_len
    }

    #[must_use]
    pub fn has_custom_mask(&self) -> bool {
        self.custom_mask
    }
}

/// How a model sorts a request into its facts, and how it packs them into the
/// one `u64` the fire carries.
///
/// EACH FAMILY WRITES ITS OWN, BY HAND. There was a `facts!` macro here that
/// generated the struct, the bit constants, the predicate constructors and the
/// packing from a list of field names — six lines of model text expanded from
/// one, and the one thing a reader wanted to know (which bit is `masked`?) was
/// the one thing it did not say. A `Facts` struct is four visible lines per
/// fact; written out, the bit a predicate tests and the bit `word` sets are
/// the same literal on the page.
pub trait Classify: Sized {
    fn of(r: &Request) -> Self;
    fn word(&self) -> u64;
}

/// The body of a catalog row's [`ClassifyFn`](crate::ClassifyFn) column,
/// monomorphized on the family the row was written for.
///
/// THE MODEL EXPRESSION IS A THUNK AND IS NEVER CALLED. All this needs off it
/// is the TYPE — `M::Facts` — and a lane's word is computed on the fire path,
/// once per lane per fire, so building a `Model` to read an associated type
/// off it would put a weight-table walk under every decode token. The
/// `catalog!` macro hands `|| Model::a3b(..)` in, inference reads `M` from the
/// closure's return type, and the closure is dropped.
#[must_use]
pub fn word_of<M: ForwardHybrid>(_model: impl FnOnce() -> M, r: &Request) -> u64 {
    <M::Facts as Classify>::of(r).word()
}

/// A declared kv geometry space: the group of kv rows one page table serves,
/// and the id [`Input::geometry`] and the plan wrappers are keyed by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvSpace(pub u32);

/// The caches a model declares, in the order `Plan::caches` will carry them —
/// kv rows and recurrent state slabs collect straight into `CacheRow`. Every
/// kv row joins a [`KvSpace`], and the space's [`Dtype`] is its rows' element
/// layout: the model states its kv-cache dtype here, so no driver ever picks
/// one silently. The dtype is all a page's element layout says — quant
/// granularity and the fp4 block size are not dtype facts, and become sibling
/// fields on `CacheRow::Kv` when the shell is written. One spec serves
/// attention-only models too: they simply never call `state`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HybridSpec {
    pub rows: Vec<CacheRow>,
    dtypes: Vec<Dtype>,
}

impl HybridSpec {
    #[must_use]
    pub fn new() -> HybridSpec {
        HybridSpec::default()
    }

    /// Declare a geometry space: one paged group of kv rows, laid out
    /// identically per fire, storing `dtype` elements.
    pub fn kv_space(&mut self, dtype: Dtype) -> KvSpace {
        self.dtypes.push(dtype);
        KvSpace(self.dtypes.len() as u32 - 1)
    }

    /// One kv row of `space`, with its per-token row shape.
    pub fn kv(
        &mut self,
        space: KvSpace,
        name: impl Into<String>,
        row: impl IntoIterator<Item = u64>,
    ) {
        let dtype = *self
            .dtypes
            .get(space.0 as usize)
            .unwrap_or_else(|| panic!("kv space {} is not one this spec declared", space.0));
        self.rows.push(CacheRow::Kv {
            name: name.into(),
            row: row.into_iter().collect(),
            dtype,
            space: space.0,
        });
    }

    pub fn state(&mut self, name: impl Into<String>, slab: impl IntoIterator<Item = u64>) {
        self.rows.push(CacheRow::State {
            name: name.into(),
            slab: slab.into_iter().collect(),
        });
    }
}

/// A forward pass over paged kv and/or recurrent state caches.
pub trait ForwardHybrid {
    type Facts: Classify;
    fn caches(&self) -> HybridSpec;
    fn forward(&self, inputs: Input<Self::Facts>) -> Value;
}

/// Trace one plan for one plane: seam the boundary, run the model's forward,
/// and `finish` through the validator.
pub fn trace_hybrid<M: ForwardHybrid>(name: &str, m: &M, plane: Plane) -> Plan {
    let caches = m.caches();
    let rec = Recorder::new(name, plane, caches.rows.clone());
    rec.seam(seam::IN.name, &[]);
    let logits = m.forward(Input {
        rec: rec.clone(),
        plane,
        caches,
        _facts: PhantomData,
    });
    rec.seam(seam::OUT.name, &[&logits]);
    drop(logits);
    rec.finish()
}

/// Walks a model's per-layer weights, keeping the recorder's layer mark in
/// step so every node knows which layer said it.
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

/// What a forward pass may reach for: the typed runtime inputs, the declared
/// cache spaces, and the plane it is tracing for. `F` ties the model's fact
/// vocabulary to its trace and is otherwise phantom.
pub struct Input<F> {
    rec: Recorder,
    plane: Plane,
    caches: HybridSpec,
    _facts: PhantomData<F>,
}

impl<F> Input<F> {
    /// Which plane this trace is for — the sanctioned way for model source to
    /// emit a backend-conditional fused op (design §10).
    #[must_use]
    pub fn plane(&self) -> Plane {
        self.plane
    }

    #[must_use]
    pub fn cuda(&self) -> bool {
        matches!(self.plane, Plane::Cuda)
    }

    #[must_use]
    pub fn tokens(&self) -> Value {
        self.rec.input(
            RuntimeInput::Tokens,
            Ty::Tensor {
                shape: vec![Dim::Tokens],
                dtype: Dtype::I32,
            },
        )
    }

    #[must_use]
    pub fn positions(&self) -> Value {
        self.rec.input(
            RuntimeInput::Positions,
            Ty::Tensor {
                shape: vec![Dim::Tokens],
                dtype: Dtype::I32,
            },
        )
    }

    /// One geometry vector of a kv space, as a declared input — the plan ops
    /// it feeds are pure functions of visible inputs (§7). The dims state
    /// alignment, not an arena size: geometry buffers are driver-bound, and
    /// `Indices` in particular is lane-aligned ragged, viewed through the
    /// indptr beside it. The kind→shape table lives on [`ops::geometry`].
    #[must_use]
    pub fn geometry(&self, space: u32, kind: GeomKind) -> Value {
        ops::geometry(&self.rec, space, kind)
    }

    /// The decode plan over the model's paged-kv space.
    ///
    /// THE FOUR CONSTRUCTORS BELOW EXIST BECAUSE THE RECORDER IS NOT THE
    /// AUTHOR'S TO NAME. Every forward pass wanted `ops::attn::plan_decode`
    /// against its own trace, and the only handle on the recorder a model
    /// could reach was `positions.rec()` — so every model asked for the
    /// positions input it did not want, purely to borrow the recorder off it.
    /// The wrappers stay `pub` for a text that plans against a second space;
    /// these four are the first space, which is what every shipped model
    /// means.
    #[must_use]
    pub fn plan_decode(&self) -> Value {
        ops::attn::plan_decode(&self.rec, self.kv_space())
    }

    #[must_use]
    pub fn plan_prefill(&self) -> Value {
        ops::attn::plan_prefill(&self.rec, self.kv_space())
    }

    #[must_use]
    pub fn mla_plan(&self) -> Value {
        ops::attn::mla_plan(&self.rec, self.kv_space())
    }

    /// The custom attention mask over the model's paged-kv space.
    #[must_use]
    pub fn mask(&self) -> Value {
        ops::mask(&self.rec, self.kv_space())
    }

    /// The model's paged-kv geometry space: the FIRST space `caches()`
    /// declared. Every shipped model declares exactly one paged-kv group —
    /// per-layer pool and index spaces come after it and are reached by row
    /// name through [`space_of`](Input::space_of) — so first is the one.
    #[must_use]
    pub fn kv_space(&self) -> u32 {
        assert!(
            !self.caches.dtypes.is_empty(),
            "the model's caches() declares no kv space",
        );
        0
    }

    /// The geometry space a named kv row joined — how a per-layer pool or
    /// index cache reaches its own space.
    #[must_use]
    pub fn space_of(&self, name: &str) -> u32 {
        self.caches
            .rows
            .iter()
            .find_map(|row| match row {
                CacheRow::Kv { name: n, space, .. } if n == name => Some(*space),
                _ => None,
            })
            .unwrap_or_else(|| panic!("`{name}` is not a kv row the model's caches() declares"))
    }

    /// The storage value of a paged kv space — a pool pointer, nothing more.
    #[must_use]
    pub fn kv(&self, name: &str) -> ValueId {
        assert!(
            self.caches
                .rows
                .iter()
                .any(|row| matches!(row, CacheRow::Kv { name: n, .. } if n == name)),
            "`{name}` is not a kv row the model's caches() declares",
        );
        self.rec.cache(name)
    }

    /// The storage value of a recurrent state space.
    #[must_use]
    pub fn state(&self, name: &str) -> ValueId {
        assert!(
            self.caches
                .rows
                .iter()
                .any(|row| matches!(row, CacheRow::State { name: n, .. } if n == name)),
            "`{name}` is not a state row the model's caches() declares",
        );
        self.rec.cache(name)
    }

    pub fn walk_layers<'a, T>(&'a self, ws: &'a [T]) -> Layers<'a, T> {
        Layers {
            rec: &self.rec,
            ws: ws.iter(),
            next: 0,
        }
    }
}
