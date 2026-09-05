//! The model-facing entry: classify a request into facts, declare cache
//! spaces, trace the forward pass, and hand back a checked `Trace`.

use std::cell::Cell;
use std::marker::PhantomData;

use model_ir::{CacheRow, Guard, Dim, Dtype, GeomKind, Trace, Platform, RuntimeInput, Ty, ValueId};

use crate::record::{Recorder, Refine, SplitSpec, Value};
use crate::seam;

pub use model_ir::Request;

/// How a model sorts a request into its facts, and how it packs them into the
/// one `u64` the fire carries. Each family writes its own by hand.
pub trait Classify: Sized {
    fn of(r: &Request) -> Self;
    fn word(&self) -> u64;
}

/// The body of a catalog row's [`ClassifyFn`](crate::ClassifyFn) column.
/// `_model` is a thunk, never called — only its return type `M` is used.
#[must_use]
pub fn word_of<M: ForwardHybrid>(_model: impl FnOnce() -> M, r: &Request) -> u64 {
    <M::Facts as Classify>::of(r).word()
}

/// A declared kv geometry space: the group of kv rows one page table serves,
/// keyed by id. A model never writes the number — it names a kv row and
/// [`Input`] resolves the space the row joined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvSpace(pub u32);

/// The caches a model declares, in `Trace::caches` order. A kv row is
/// declared as a plane list — per-token width in elements, e.g. `[w, w]`
/// for a k|v pair, `[w]` shared as both, `[kv_lora_rank, rope_dim]` a latent
/// page. Every kv row joins a [`KvSpace`], whose [`Dtype`] is its layout.
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

    /// Declare a geometry space: one paged group of kv rows storing `dtype`
    /// elements.
    pub fn kv_space(&mut self, dtype: Dtype) -> KvSpace {
        self.dtypes.push(dtype);
        KvSpace(self.dtypes.len() as u32 - 1)
    }

    /// One kv row of `space`: `planes` states the per-token widths in
    /// elements — `[w, w]` a k|v pair, `[w]` shared, `[kv_lora_rank,
    /// rope_dim]` a latent page.
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

    /// One recurrent-state slab. `dtype` is the slab's own: e.g. bf16 for
    /// ssm history, or an integer dtype for a window that keeps token ids.
    pub fn state(
        &mut self,
        name: impl Into<String>,
        slab: impl IntoIterator<Item = u64>,
        dtype: Dtype,
    ) {
        self.rows.push(CacheRow::State {
            name: name.into(),
            slab: slab.into_iter().collect(),
            dtype,
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
    // Set by [`trace_hybrid`] around the model's `forward`, cleared after.
    static TRACING: Cell<Option<Platform>> = const { Cell::new(None) };
}

/// The platform this trace is being taken for, for a backend-conditional
/// fused op. Ambient for the length of one `forward`; panics outside one.
#[must_use]
pub fn platform() -> Platform {
    TRACING
        .with(|tracing| tracing.get())
        .expect("no trace is being taken")
}

/// Trace one plan for one platform: seam the boundary, run the model's
/// forward, and `finish` through the validator. A nested trace on one
/// thread is refused (it would have two answers to [`platform`]).
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
/// fact vocabulary to its trace and is otherwise phantom. Splittable like
/// [`Value`]: [`split`](Input::split) hands back arms carrying a class.
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
    /// algorithm and conds [`Value::split`] uses, so an `Input` arm and a
    /// `Value` arm taken with the same spec meet without complaint.
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

    /// The trunk's triple-wide position stream: `[Dim::Tokens, 3]` `i32`,
    /// one `(t, h, w)` per token row. A text lane's triple is `(p, p, p)`.
    #[must_use]
    pub fn mrope_positions(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::MropePositions,
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(3)],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    /// The model's paged-kv space's custom attention mask: packed `u8` mask
    /// bits, token-aligned, read by `attention.masked`.
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

    /// The fire's per-row adapter ids: `i32`, one entry per token row, `-1`
    /// for a row whose lane registered no adapter. Deduped by
    /// [`Recorder::input`], so many correction ops share one input.
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

    /// The fire's patch rows, pre-unfolded: `[Dim::Patches, width]`. `width`
    /// is `C * T * P^2` (channels x temporal/spatial patch extents).
    #[must_use]
    pub fn patches(&self, width: impl Into<u64>) -> Value {
        self.rec
            .input(
                RuntimeInput::Patches,
                Ty::Tensor {
                    shape: vec![Dim::Patches, Dim::Const(width.into())],
                    dtype: Dtype::Bf16,
                },
            )
            .refined(self.over.clone())
    }

    /// The patch axis's own indptr: `i32`, `images + 1` long, where image
    /// `i` owns patch rows `[segments[i], segments[i + 1])`.
    #[must_use]
    pub fn patch_segments(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::PatchSegments,
                Ty::Tensor {
                    shape: vec![Dim::ImagesPlus(1)],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    /// Where each tower row lands in the token rectangle: `i32`, one
    /// destination token row per patch row.
    #[must_use]
    pub fn patch_routes(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::PatchRoutes,
                Ty::Tensor {
                    shape: vec![Dim::Patches],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    /// The tower's own position stream: `[Dim::Patches, 3]` `i32`, one
    /// `(t, h, w)` per patch row (each patch's `(h, w)` in its own image's
    /// grid). The `t` column is currently always zero.
    #[must_use]
    pub fn patch_positions(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::PatchPositions,
                Ty::Tensor {
                    shape: vec![Dim::Patches, Dim::Const(3)],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    /// Which row of the learned position table each patch reads:
    /// `[Dim::Patches, taps]` `i32`. `taps`: `1` native grid, `4` bilinear,
    /// `16` bicubic — paired with
    /// [`patch_embed_weights`](Input::patch_embed_weights).
    #[must_use]
    pub fn patch_embed_rows(&self, taps: u32) -> Value {
        self.rec
            .input(
                RuntimeInput::PatchEmbedRows,
                Ty::Tensor {
                    shape: vec![Dim::Patches, Dim::Const(u64::from(taps))],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    /// How much of each tap: `[Dim::Patches, taps]` `f32`. A native-grid
    /// tower never calls this.
    #[must_use]
    pub fn patch_embed_weights(&self, taps: u32) -> Value {
        self.rec
            .input(
                RuntimeInput::PatchEmbedWeights,
                Ty::Tensor {
                    shape: vec![Dim::Patches, Dim::Const(u64::from(taps))],
                    dtype: Dtype::F32,
                },
            )
            .refined(self.over.clone())
    }

    /// The denoiser's self-conditioning taps: `[Dim::Tokens, taps]` `i32`
    /// token ids, one row of `taps` per token row, paired with
    /// [`self_cond_weights`](Input::self_cond_weights) in a weighted gather
    /// over the embedding table.
    #[must_use]
    pub fn self_cond_rows(&self, taps: u32) -> Value {
        self.rec
            .input(
                RuntimeInput::SelfCondRows,
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(u64::from(taps))],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    /// How much of each self-conditioning tap: `[Dim::Tokens, taps]` `f32`.
    #[must_use]
    pub fn self_cond_weights(&self, taps: u32) -> Value {
        self.rec
            .input(
                RuntimeInput::SelfCondWeights,
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(u64::from(taps))],
                    dtype: Dtype::F32,
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
    /// joined — the same key [`kv`](Input::kv) takes.
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

    /// One geometry vector of a kv space. Indptr is `lanes + 1` long,
    /// page-table vectors are per-lane, fire tables are per-token. All
    /// `i32` except `RowValid` (`u8`).
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

    /// The model's paged-kv geometry space: the first space `caches()`
    /// declared.
    fn kv_space(&self) -> u32 {
        assert!(
            !self.caches.dtypes.is_empty(),
            "the model's caches() declares no kv space",
        );
        0
    }

    /// The geometry space a named kv row joined.
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
