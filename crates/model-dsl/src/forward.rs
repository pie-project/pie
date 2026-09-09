use std::cell::Cell;
use std::marker::PhantomData;

use model_ir::{
    CacheRow, Dim, Dtype, GeomKind, Guard, Platform, RuntimeInput, Selection, Trace, Ty, ValueId,
};

use crate::record::{Recorder, Refine, SplitSpec, Value};
use crate::seam;

pub use model_ir::Request;

pub trait Classify: Sized {
    fn of(r: &Request) -> Self;
    fn word(&self) -> u64;
}

#[must_use]
pub fn word_of<M: ForwardHybrid>(_model: impl FnOnce() -> M, r: &Request) -> u64 {
    <M::Facts as Classify>::of(r).word()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvSpace(pub u32);

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

    pub fn kv_space(&mut self, dtype: Dtype) -> KvSpace {
        self.dtypes.push(dtype);
        KvSpace(self.dtypes.len() as u32 - 1)
    }

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

pub trait ForwardHybrid {
    type Facts: Classify;
    fn caches(&self) -> HybridSpec;
    fn forward(&self, inputs: Input<Self::Facts>) -> Value;
}

thread_local! {
    static TRACING: Cell<Option<Platform>> = const { Cell::new(None) };
}

#[must_use]
pub fn platform() -> Platform {
    TRACING
        .with(|tracing| tracing.get())
        .expect("no trace is being taken")
}

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
    struct Tracing;
    impl Drop for Tracing {
        fn drop(&mut self) {
            TRACING.with(|tracing| tracing.set(None));
        }
    }
    let tracing = Tracing;
    let logits = m.forward(Input {
        rec: rec.clone(),
        caches,
        over: Guard::Always,
        _facts: PhantomData,
    });
    drop(tracing);
    let float_readout = seam::FLOAT_READOUTS
        .iter()
        .any(|name| rec.seamed(name, &logits));
    if !float_readout {
        rec.seam(seam::OUT.name, &[&logits]);
    }
    drop(logits);
    rec.finish()
}

const TOKEN_SPACE: u32 = 0;

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

pub struct Input<F> {
    rec: Recorder,
    caches: HybridSpec,
    over: Guard,
    _facts: PhantomData<F>,
}

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
            over: Guard::narrow(self.over.clone(), cond),
            _facts: PhantomData,
        }
    }
}

impl<F> Input<F> {
    #[must_use]
    pub fn recorder(&self) -> &Recorder {
        &self.rec
    }

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

    #[must_use]
    pub fn readout_rows(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::ReadoutRows,
                Ty::Tensor {
                    shape: vec![Dim::Readouts],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

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

    #[must_use]
    pub fn kv_indptr(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::Indptr)
    }

    #[must_use]
    pub fn kv_indices(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::Indices)
    }

    #[must_use]
    pub fn last_page_len(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::LastPageLen)
    }

    #[must_use]
    pub fn kv_len(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::KvLen)
    }

    #[must_use]
    pub fn row_valid(&self) -> Value {
        self.geometry(self.kv_space(), GeomKind::RowValid)
    }

    #[must_use]
    pub fn request_of_token(&self) -> Value {
        self.geometry(TOKEN_SPACE, GeomKind::RequestOfToken)
    }

    #[must_use]
    pub fn group_of_lane(&self) -> Value {
        self.geometry(TOKEN_SPACE, GeomKind::GroupOfLane)
    }

    #[must_use]
    pub fn group_indptr(&self) -> Value {
        self.geometry(
            TOKEN_SPACE,
            GeomKind::GroupIndptr {
                select: self.selection(),
            },
        )
    }

    #[must_use]
    pub fn lane_indptr(&self) -> Value {
        self.geometry(
            TOKEN_SPACE,
            GeomKind::LaneIndptr {
                select: self.selection(),
            },
        )
    }

    #[must_use]
    pub fn reference_tags(&self) -> Value {
        self.geometry(
            TOKEN_SPACE,
            GeomKind::ReferenceTag {
                select: self.selection(),
            },
        )
    }

    #[must_use]
    pub fn row_permutation(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::RowPermutation {
                    select: self.selection(),
                },
                Ty::Tensor {
                    shape: vec![Dim::Tokens],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    #[must_use]
    pub fn latents(&self, port: u8, width: u32, dtype: Dtype) -> Value {
        assert!(
            matches!(dtype, Dtype::F32 | Dtype::Bf16),
            "a latent port is f32 or bf16, not {dtype:?}"
        );
        self.rec
            .input(
                RuntimeInput::Latents { port, width },
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(u64::from(width))],
                    dtype,
                },
            )
            .refined(self.over.clone())
    }

    #[must_use]
    pub fn lane_vector(&self, port: u8, width: u32) -> Value {
        self.rec
            .input(
                RuntimeInput::LaneVector { port, width },
                Ty::Tensor {
                    shape: vec![Dim::Lanes, Dim::Const(u64::from(width))],
                    dtype: Dtype::F32,
                },
            )
            .refined(self.over.clone())
    }

    #[must_use]
    pub fn context(&self, port: u8, width: u32) -> Value {
        self.rec
            .input(
                RuntimeInput::Context { port, width },
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(u64::from(width))],
                    dtype: Dtype::Bf16,
                },
            )
            .refined(self.over.clone())
    }

    #[must_use]
    pub fn axis_positions(&self, port: u8, axes: u8) -> Value {
        assert!(
            (1..=4).contains(&axes),
            "a rope has one to four axes, not {axes}"
        );
        self.rec
            .input(
                RuntimeInput::AxisPositions { port, axes },
                Ty::Tensor {
                    shape: vec![Dim::Tokens, Dim::Const(u64::from(axes))],
                    dtype: Dtype::F32,
                },
            )
            .refined(self.over.clone())
    }

    #[must_use]
    pub fn grid(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::Grid,
                Ty::Tensor {
                    shape: vec![Dim::Clips, Dim::Const(4)],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    #[must_use]
    pub fn voxels(&self, port: u8, channels: u32, dtype: Dtype) -> Value {
        assert!(
            matches!(dtype, Dtype::F32 | Dtype::Bf16),
            "a voxel port is f32 or bf16, not {dtype:?}"
        );
        self.rec
            .input(
                RuntimeInput::Voxels { port, channels },
                Ty::Tensor {
                    shape: vec![Dim::Voxels, Dim::Const(u64::from(channels))],
                    dtype,
                },
            )
            .refined(self.over.clone())
    }

    #[must_use]
    pub fn token_grid(&self, p: [u32; 3]) -> Value {
        assert!(p.iter().all(|&n| n > 0), "a patch of {p:?} is empty");
        self.rec
            .input(
                RuntimeInput::TokenGrid { p },
                Ty::Tensor {
                    shape: vec![Dim::Clips, Dim::Const(4)],
                    dtype: Dtype::I32,
                },
            )
            .refined(self.over.clone())
    }

    fn selection(&self) -> Selection {
        Selection::of(&self.over).unwrap_or_else(|| {
            panic!(
                "an input arm guarded by {:?} is not a conjunction of facts and names no \
                 row selection",
                self.over
            )
        })
    }

    #[must_use]
    pub fn write_page(&self, row: &str) -> Value {
        self.geometry(self.space_of(row), GeomKind::WritePage)
    }

    #[must_use]
    pub fn write_offset(&self, row: &str) -> Value {
        self.geometry(self.space_of(row), GeomKind::WriteOffset)
    }

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

    fn geometry(&self, space: u32, kind: GeomKind) -> Value {
        let (rows, dtype) = match kind {
            GeomKind::Indptr | GeomKind::GroupIndptr { .. } | GeomKind::LaneIndptr { .. } => {
                (Dim::LanesPlus(1), Dtype::I32)
            }
            GeomKind::Indices
            | GeomKind::SeqLens
            | GeomKind::LastPageLen
            | GeomKind::KvLen
            | GeomKind::GroupOfLane => (Dim::Lanes, Dtype::I32),
            GeomKind::RowValid => (Dim::Tokens, Dtype::U8),
            GeomKind::RequestOfToken
            | GeomKind::WritePage
            | GeomKind::WriteOffset
            | GeomKind::ReferenceTag { .. } => (Dim::Tokens, Dtype::I32),
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

    fn kv_space(&self) -> u32 {
        assert!(
            !self.caches.dtypes.is_empty(),
            "the model's caches() declares no kv space",
        );
        0
    }

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
