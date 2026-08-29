//! The model-facing entry: classify a request into facts, declare cache
//! spaces, trace the forward pass, and hand back a checked `Trace`.
//! Re-imagined from the old `forward.rs`: the inputs handle is typed now —
//! tokens and positions arrive as `[Tokens] i32` values, cache geometry as
//! declared `RuntimeInput::Geometry` inputs (§7) — and `qo_indptr` is gone
//! outright, because raggedness is ambient (§5): every fire-aligned value is
//! viewable through the fire's shared indptr, so there is nothing to attach.

use std::cell::Cell;
use std::marker::PhantomData;

use model_ir::{CacheRow, Guard, Dim, Dtype, GeomKind, Trace, Platform, RuntimeInput, Ty, ValueId};

use crate::record::{Recorder, Refine, SplitSpec, Value};
use crate::seam;

/// One request's shape facts, as the runtime states them per fire.
///
/// **FIVE FACTS ABOUT A REQUEST, NOT ABOUT A FIRE.** Everything here is
/// per-lane by construction (design §0's vocabulary note): how many rows this
/// request feeds, whether it brought a mask of its own, whether it routes to
/// an adapter, whether it wants the draft head run over its rows, and whether
/// it wants the attention scores kept. A model's `Classify::of` reads what it
/// declared bits for and ignores the rest.
///
/// **THE LIST GROWS ONE FIELD PER AXIS, AND THAT IS THE POINT** (design §8).
/// An axis is declared by a model text — a fact, a window, an arm — and the
/// only thing the axis needs from THIS side is a per-lane boolean the family's
/// `Classify::of` can read. `masked` came in with C1, `adapter` with C2, and
/// `drafts`/`captures_scores` with C3/C4. Nothing else here moves: a family
/// that declares no draft head never reads `drafts`, and its word is the word
/// it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    query_len: u32,
    custom_mask: bool,
    adapter: bool,
    drafts: bool,
    captures_scores: bool,
}

impl Request {
    #[must_use]
    pub fn new(query_len: u32, custom_mask: bool) -> Request {
        Request {
            query_len,
            custom_mask,
            adapter: false,
            drafts: false,
            captures_scores: false,
        }
    }

    /// The same request, routing to an adapter bank (design §8).
    ///
    /// A builder rather than a fourth positional argument: `Request::new` has
    /// forty call sites across the tests and the runtime, and every one of them
    /// that does NOT route would have had to write `false` — which is exactly
    /// the shape a reader stops checking. Whoever routes says so.
    #[must_use]
    pub fn adapted(mut self, adapter: bool) -> Request {
        self.adapter = adapter;
        self
    }

    /// The same request, with the model's draft head run over its rows
    /// (design §8's MTP axis, palo C3).
    ///
    /// A BUILDER FOR `adapted`'s REASON, and one more: `Request::new` has
    /// forty call sites that route nowhere and draft nothing, and every axis
    /// added as a positional argument makes every one of them state a `false`
    /// that a reader stops checking. Whoever drafts says so.
    #[must_use]
    pub fn drafting(mut self, drafts: bool) -> Request {
        self.drafts = drafts;
        self
    }

    /// The same request, with its attention's per-key mass kept (design §9's
    /// score-capture archetype, palo C4).
    #[must_use]
    pub fn capturing_scores(mut self, captures_scores: bool) -> Request {
        self.captures_scores = captures_scores;
        self
    }

    #[must_use]
    pub fn query_len(&self) -> u32 {
        self.query_len
    }

    #[must_use]
    pub fn has_custom_mask(&self) -> bool {
        self.custom_mask
    }

    #[must_use]
    pub fn has_adapter(&self) -> bool {
        self.adapter
    }

    #[must_use]
    pub fn drafts(&self) -> bool {
        self.drafts
    }

    #[must_use]
    pub fn captures_scores(&self) -> bool {
        self.captures_scores
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
/// and the id every `RuntimeInput::Geometry` a forward reaches for is keyed
/// by. A model text never writes the number — it names a kv row and
/// [`Input`] resolves the space the row joined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvSpace(pub u32);

/// The caches a model declares, in the order `Trace::caches` will carry them —
/// kv rows and recurrent state slabs collect straight into `CacheRow`. A kv
/// row is declared as a plane list — the pieces one token's entry is written
/// as, each by its per-token width in elements, so a k|v pair is `[w, w]`, a
/// plane shared as both k and v is `[w]`, and a latent page, whose two planes
/// are not the same width, is `[kv_lora_rank, rope_dim]`. Every kv row joins a
/// [`KvSpace`], and the space's [`Dtype`] is its rows' element layout: the
/// model states its kv-cache dtype here, so no engine ever picks one silently.
/// The dtype is all a page's element layout says — quant granularity and the
/// fp4 block size are not dtype facts, and become sibling fields on
/// `CacheRow::Kv` when the shell is written. One spec serves attention-only
/// models too: they simply never call `state`.
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

    /// One kv row of `space`: `planes` states the pieces one token's entry is
    /// written as, each by its per-token width in elements — `[w, w]` a k|v
    /// pair, `[w]` one plane shared as both k and v (the rows only an indexer
    /// or a pooled reader walks), `[kv_lora_rank, rope_dim]` a latent page,
    /// whose two planes are not the same width.
    pub fn kv(
        &mut self,
        space: KvSpace,
        name: impl Into<String>,
        planes: impl IntoIterator<Item = u64>,
    ) {
        let dtype = *self
            .dtypes
            .get(space.0 as usize)
            .unwrap_or_else(|| panic!("kv space {} is not one this spec declared", space.0));
        self.rows.push(CacheRow::Kv {
            name: name.into(),
            planes: planes.into_iter().collect(),
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

thread_local! {
    /// The platform of the trace this thread is inside, set by [`trace_hybrid`]
    /// around the model's `forward` and cleared after. A forward asks it
    /// through [`platform`]; nothing else writes it.
    static TRACING: Cell<Option<Platform>> = const { Cell::new(None) };
}

/// The platform this trace is being taken for — the sanctioned way for model
/// source to emit a backend-conditional fused op (design §10).
///
/// **AMBIENT BECAUSE IT IS A FACT ABOUT THE TRACE, NOT ABOUT THE INPUTS.** It
/// rode on `Input` until M20, where `Input` became splittable: an arm of the
/// inputs is a class of rows, and the platform is the same on every row of
/// every arm. Threading a whole-trace constant through a per-class handle made
/// every arm restate it. The trace scopes it instead — set for the length of
/// one `forward`, and a panic outside one.
#[must_use]
pub fn platform() -> Platform {
    TRACING
        .with(|tracing| tracing.get())
        .expect("no trace is being taken")
}

/// Trace one plan for one platform: seam the boundary, run the model's
/// forward, and `finish` through the validator. The platform is ambient for
/// the length of `forward` and nowhere else — a nested trace on one thread
/// would have two answers to [`platform`], so it is refused here.
pub fn trace_hybrid<M: ForwardHybrid>(name: &str, m: &M, platform: Platform) -> Trace {
    let caches = m.caches();
    let rec = Recorder::new(name, platform, caches.rows.clone());
    rec.seam(seam::IN.name, &[]);
    TRACING.with(|tracing| {
        assert!(
            tracing.get().is_none(),
            "a trace is already being taken on this thread",
        );
        tracing.set(Some(platform));
    });
    let logits = m.forward(Input {
        rec: rec.clone(),
        caches,
        over: Guard::Always,
        _facts: PhantomData,
    });
    TRACING.with(|tracing| tracing.set(None));
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

/// What a forward pass may reach for: the typed runtime inputs and the
/// declared cache spaces, read for one class of rows. `F` ties the model's
/// fact vocabulary to its trace and is otherwise phantom.
///
/// **AN INPUT HANDLE IS SPLITTABLE, EXACTLY AS A [`Value`] IS.** `over` is
/// `Guard::Always` for the whole fire; [`split`](Input::split) hands back arms
/// that carry a class in it, and every value read off an arm carries that
/// class too. That is what lets a schedule say which class it was carved for
/// — it is built off that class's arm — instead of leaving the answer to be
/// inferred from who reads it.
pub struct Input<F> {
    rec: Recorder,
    caches: HybridSpec,
    /// `Always` for the whole fire; a split arm carries its class here, and
    /// every value read off the arm carries it too.
    over: Guard,
    _facts: PhantomData<F>,
}

/// Written out rather than derived: `#[derive(Clone)]` would demand
/// `F: Clone`, and `F` is a phantom fact vocabulary that no family bothers to
/// make cloneable.
impl<F> Clone for Input<F> {
    fn clone(&self) -> Input<F> {
        Input {
            rec: self.rec.clone(),
            caches: self.caches.clone(),
            over: self.over.clone(),
            _facts: PhantomData,
        }
    }
}

impl<F> Refine for Input<F> {
    fn refined(&self, cond: Guard) -> Input<F> {
        Input {
            rec: self.rec.clone(),
            caches: self.caches.clone(),
            over: Guard::and(self.over.clone(), cond),
            _facts: PhantomData,
        }
    }
}

impl<F> Input<F> {
    /// The inputs of one class of rows each, cut by `spec` — the same
    /// algorithm and the same conds [`Value::split`] cuts by, so an `Input`
    /// arm and a `Value` arm taken with the same spec meet without complaint
    /// and arms of different classes do not (design §8).
    pub fn split<S: SplitSpec>(&self, spec: S) -> S::Arms<Input<F>> {
        spec.arms(self)
    }

    #[must_use]
    pub fn tokens(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::Tokens,
                Ty::Tensor {
                    shape: vec![Dim::Tokens],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    #[must_use]
    pub fn positions(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::Positions,
                Ty::Tensor {
                    shape: vec![Dim::Tokens],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    /// The model's paged-kv space's custom attention mask: packed `u8` mask
    /// bits, token-aligned, read by `attention.masked`. Both platforms carry
    /// the bits this way — metal's fire tables and the cuda plan's `Mask`
    /// pair; the per-request enabled bits and spans stay engine-derived for
    /// now.
    #[must_use]
    pub fn mask(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::Mask {
                    space: self.kv_space(),
                },
                Ty::Tensor {
                    shape: vec![Dim::Tokens],
                    dtype: Dtype::U8,
                },
            )
            .refined(self.over.clone())
    }

    /// The fire's per-row adapter ids (design §8): `i32`, one entry per token
    /// row, `-1` for a row whose lane registered no adapter.
    ///
    /// ONE VECTOR FOR EVERY CORRECTION SITE, and the dedup in
    /// [`Recorder::input`] is what makes that literally true — an adapter is a
    /// property of the request, so a plan with forty correction ops declares
    /// one input and forty readers of it.
    #[must_use]
    pub fn adapter_routes(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::AdapterRoutes,
                Ty::Tensor {
                    shape: vec![Dim::Tokens],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    /// Where each lane's run of page indices starts, `lanes + 1` long.
    #[must_use]
    pub fn kv_indptr(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::Indptr)
    }

    /// The page indices themselves, viewed through [`kv_indptr`](Input::kv_indptr).
    #[must_use]
    pub fn kv_indices(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::Indices)
    }

    /// How many slots of each lane's last page are filled.
    #[must_use]
    pub fn last_page_len(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::LastPageLen)
    }

    /// Each lane's cached key length.
    #[must_use]
    pub fn kv_len(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::KvLen)
    }

    /// The packed `u8` graph-padding mask: which token rows of the fire are
    /// real.
    #[must_use]
    pub fn row_valid(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::RowValid)
    }

    /// The token→lane map: which request each token row belongs to.
    #[must_use]
    pub fn request_of_token(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::RequestOfToken)
    }

    /// The page each token row appends to, in the space the kv row `row`
    /// joined — the same key [`kv`](Input::kv) takes, so a text names its
    /// cache once and gets the storage and the addressing off the one name.
    #[must_use]
    pub fn write_page(&self, row: &str) -> Value {
        self.geometry(self.space_of(row), GeomKind::WritePage)
    }

    /// The offset within that page.
    #[must_use]
    pub fn write_offset(&self, row: &str) -> Value {
        self.geometry(self.space_of(row), GeomKind::WriteOffset)
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

    /// One geometry vector of a kv space, as a declared runtime input (§7) —
    /// the plan ops it feeds are pure functions of visible inputs. The
    /// indptr-shaped vectors are `lanes + 1` long, the page-table vectors are
    /// per-lane, and the fire tables — the padding mask, the token→lane map,
    /// and the write addressing — are per-token. Everything is `i32` except
    /// `RowValid`, the packed `u8` graph-padding mask. The dims state
    /// alignment, not an arena size: geometry buffers are engine-bound, and
    /// `Indices` in particular is lane-aligned ragged, viewed through the
    /// indptr beside it.
    ///
    /// PRIVATE, AND KEYED BY THE ROW ABOVE IT. A model text used to write
    /// `inputs.geometry(inputs.kv_space(), GeomKind::WritePage)` — three
    /// names, two of them the DSL's own bookkeeping, to say "where this row
    /// appends". The accessors above are the surface; this is the one place
    /// the kind→shape table is stated, and the one place an arm's class lands
    /// on a geometry value, so `over` rides out on all of them and no
    /// accessor can forget it.
    fn geometry(&self, space: u32, kind: GeomKind) -> Value {
        let (rows, dtype) = match kind {
            GeomKind::Indptr => (Dim::LanesPlus(1), Dtype::I32),
            GeomKind::Indices | GeomKind::SeqLens | GeomKind::LastPageLen | GeomKind::KvLen => {
                (Dim::Lanes, Dtype::I32)
            }
            GeomKind::RowValid => (Dim::Tokens, Dtype::U8),
            GeomKind::RequestOfToken | GeomKind::WritePage | GeomKind::WriteOffset => {
                (Dim::Tokens, Dtype::I32)
            }
        };
        self.rec
            .input(
                RuntimeInput::Geometry { space, kind },
                Ty::Tensor {
                    shape: vec![rows],
                    dtype,
                },
            )
            .refined(self.over.clone())
    }

    /// The model's paged-kv geometry space: the FIRST space `caches()`
    /// declared. Every shipped model declares exactly one paged-kv group —
    /// per-layer pool and index spaces come after it and are reached by row
    /// name through [`space_of`](Input::space_of) — so first is the one.
    fn kv_space(&self) -> u32 {
        assert!(
            !self.caches.dtypes.is_empty(),
            "the model's caches() declares no kv space",
        );
        0
    }

    /// The geometry space a named kv row joined — how a per-layer pool or
    /// index cache reaches its own space.
    fn space_of(&self, name: &str) -> u32 {
        self.caches
            .rows
            .iter()
            .find_map(|row| match row {
                CacheRow::Kv { name: n, space, .. } if n == name => Some(*space),
                _ => None,
            })
            .unwrap_or_else(|| panic!("`{name}` is not a kv row the model's caches() declares"))
    }
}
