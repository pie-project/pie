use poem_ir::{
    CacheRow, Dim, Dtype, GeomKind, Guard, InputAxis, Platform, RuntimeInput, Selection, Shard,
    Trace, Ty, ValueId,
};

use crate::fact::Predicate;
use crate::record::{Recorder, Refine, Value};

pub use poem_ir::Request;

/// The rows a forward written in Rust keeps from one fire to the next: a
/// per-token (kv) row grows a row a token, a state row is a fixed slab a
/// sequence holds one of. Which list a kv row's slots lie in is the
/// compiler's to decide (`chunk::plan`), from the row's retention.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateRows {
    pub rows: Vec<CacheRow>,
}

impl StateRows {
    #[must_use]
    pub fn new() -> StateRows {
        StateRows::default()
    }

    pub fn kv(
        &mut self,
        name: impl Into<String>,
        dtype: Dtype,
        planes: impl IntoIterator<Item = u64>,
        head_dim: u32,
    ) -> DeclaredCache<'_> {
        self.rows.push(CacheRow::Kv {
            name: name.into(),
            planes: planes.into_iter().collect(),
            dtype,
            window: None,
            head_dim,
            live: None,
            shard: Shard::Replicated,
            rows: None,
        });
        DeclaredCache(self.rows.last_mut().expect("a row was just pushed"))
    }

    pub fn state(
        &mut self,
        name: impl Into<String>,
        slab: impl IntoIterator<Item = u64>,
        dtype: Dtype,
    ) -> DeclaredCache<'_> {
        self.rows.push(CacheRow::State {
            name: name.into(),
            slab: slab.into_iter().collect(),
            dtype,
            shard: Shard::Replicated,
        });
        DeclaredCache(self.rows.last_mut().expect("a row was just pushed"))
    }

    /// Whether any row grows a row a token.
    fn per_token(&self) -> bool {
        self.rows
            .iter()
            .any(|row| matches!(row, CacheRow::Kv { .. }))
    }
}

/// A cache row just declared, to state how tensor-parallel ranks split it.
pub struct DeclaredCache<'a>(&'a mut CacheRow);

impl DeclaredCache<'_> {
    /// The kv row is only ever read through a sliding window of `window`
    /// tokens, so an engine may keep just that tail of each sequence.
    pub fn window(&mut self, window: u32) -> &mut Self {
        let CacheRow::Kv { window: kept, .. } = &mut *self.0 else {
            panic!("only a kv row is read through a window");
        };
        *kept = Some(window);
        self
    }

    /// The kv row's heads are zero-padded past `key` and `value` values.
    pub fn live(&mut self, key: u32, value: u32) -> &mut Self {
        let CacheRow::Kv { live, .. } = &mut *self.0 else {
            panic!("only a kv row holds heads");
        };
        *live = Some([key, value]);
        self
    }

    /// The kv row's u8 planes hold `format`'s block-scaled rows.
    pub fn rows(&mut self, format: poem_ir::KvRows) -> &mut Self {
        let CacheRow::Kv { rows, .. } = &mut *self.0 else {
            panic!("only a kv row holds quantized rows");
        };
        *rows = Some(format);
        self
    }

    /// The kv row's planes hold heads, which the ranks split between them.
    pub fn heads(self) {
        let CacheRow::Kv { planes, shard, .. } = self.0 else {
            panic!("only a kv row holds heads");
        };
        *shard = Shard::Cut {
            axis: 0,
            segments: planes.clone(),
            heads: Vec::new(),
        };
    }

    /// The state row's slab is split between the ranks along `axis`.
    pub fn split(self, axis: u32) {
        let CacheRow::State { slab, shard, .. } = self.0 else {
            panic!("only a state row has a slab to split");
        };
        let extent = *slab
            .get(axis as usize)
            .unwrap_or_else(|| panic!("a slab of {slab:?} has no axis {axis}"));
        *shard = Shard::Cut {
            axis,
            segments: vec![extent],
            heads: Vec::new(),
        };
    }
}

pub trait ForwardHybrid {
    fn states(&self) -> StateRows;
    fn forward(&self, inputs: Input);
}

pub fn trace_hybrid<M: ForwardHybrid>(name: &str, m: &M, platform: Platform) -> Trace {
    let caches = m.states();
    let rec = Recorder::new(name, platform, caches.rows.clone());
    m.forward(Input {
        rec: rec.clone(),
        caches,
        over: Guard::Always,
    });
    rec.finish()
}

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

#[derive(Clone)]
pub struct Input {
    rec: Recorder,
    caches: StateRows,
    over: Guard,
}

impl Refine for Input {
    fn refined(&self, cond: Guard) -> Input {
        Input {
            rec: self.rec.clone(),
            caches: self.caches.clone(),
            over: Guard::narrow(self.over.clone(), cond),
        }
    }
}

impl Input {
    #[must_use]
    pub fn recorder(&self) -> &Recorder {
        &self.rec
    }

    /// The rows these are.
    pub(crate) fn rows(&self) -> &Guard {
        &self.over
    }

    /// Every lane's own mask, over every row: what a masked arm reads,
    /// whichever rows it runs on.
    pub(crate) fn whole_mask(&self) -> Value {
        Input {
            rec: self.rec.clone(),
            caches: self.caches.clone(),
            over: Guard::Always,
        }
        .mask()
    }

    /// The rows `predicate` holds for.
    #[must_use]
    pub fn on(&self, predicate: Predicate) -> Input {
        self.refined(self.rec.guard_of(&predicate))
    }

    /// The rows parted by `cases`, as [`Value::partition`] parts a value's.
    #[must_use]
    pub fn partition<const N: usize>(&self, cases: [Predicate; N]) -> ([Input; N], Input) {
        crate::record::partition(self, &self.rec, cases)
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
        self.per_token();
        self.rec
            .input(
                RuntimeInput::Mask,
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

    /// Each patch's `width` raw RGB bytes, `3 · patch²`, pixel-major.
    #[must_use]
    pub fn pixels(&self, width: impl Into<u64>) -> Value {
        self.rec
            .input(
                RuntimeInput::Patches,
                Ty::Tensor {
                    shape: vec![Dim::Patches, Dim::Const(width.into())],
                    dtype: Dtype::U8,
                },
            )
            .refined(self.over.clone())
    }

    /// Each image's `(t, rows, cols)` patch grid.
    #[must_use]
    pub fn image_grids(&self) -> Value {
        self.rec
            .input(
                RuntimeInput::ImageGrids,
                Ty::Tensor {
                    shape: vec![Dim::Images, Dim::Const(3)],
                    dtype: Dtype::I32,
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
    pub fn kv_indptr(&self) -> Value {
        self.read_geometry(GeomKind::Indptr)
    }

    #[must_use]
    pub fn kv_indices(&self) -> Value {
        self.read_geometry(GeomKind::Indices)
    }

    #[must_use]
    pub fn last_page_len(&self) -> Value {
        self.read_geometry(GeomKind::LastPageLen)
    }

    #[must_use]
    pub fn kv_len(&self) -> Value {
        self.read_geometry(GeomKind::KvLen)
    }

    #[must_use]
    pub fn row_valid(&self) -> Value {
        self.read_geometry(GeomKind::RowValid)
    }

    /// Whether the backend reads a router's hint of the experts the next
    /// layer will take (to stream their weights ahead). CUDA holds every
    /// expert resident and reads none, so a forward computes none for it.
    #[must_use]
    pub fn route_hints(&self) -> bool {
        self.rec.platform() != Platform::Cuda
    }

    /// Whether a decode attention takes readout rows for its queries, each
    /// attending as the token it reads out. Metal seats each readout's
    /// position and lane beside the tokens', and XLA gathers them from the
    /// tokens' at the readout rows; the other backends attend only token
    /// rows, so a forward keeps every row for them.
    #[must_use]
    pub fn readout_attention(&self) -> bool {
        matches!(self.rec.platform(), Platform::Metal | Platform::Xla)
    }

    #[must_use]
    pub fn request_of_token(&self) -> Value {
        self.geometry(GeomKind::RequestOfToken)
    }

    #[must_use]
    pub fn group_of_lane(&self) -> Value {
        self.geometry(GeomKind::GroupOfLane)
    }

    #[must_use]
    pub fn group_indptr(&self) -> Value {
        self.geometry(GeomKind::GroupIndptr {
            select: self.selection(),
        })
    }

    #[must_use]
    pub fn lane_indptr(&self) -> Value {
        self.geometry(GeomKind::LaneIndptr {
            select: self.selection(),
        })
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

    /// The guest input `name`, one input within any reading that reads it.
    #[must_use]
    pub fn input(
        &self,
        name: &str,
        dtype: Dtype,
        axis: InputAxis,
        width: u32,
        rows: Option<poem_ir::RowCount>,
    ) -> Value {
        assert!(
            matches!(dtype, Dtype::F32 | Dtype::Bf16 | Dtype::I32),
            "input `{name}` is f32, bf16 or i32, not {dtype:?}"
        );
        assert!(width > 0, "input `{name}` is zero wide");
        self.rec
            .guest(name, dtype, axis, width, rows)
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
        self.geometry(GeomKind::WritePage {
            window: self.window_of(row),
        })
    }

    #[must_use]
    pub fn write_offset(&self, row: &str) -> Value {
        self.geometry(GeomKind::WriteOffset {
            window: self.window_of(row),
        })
    }

    #[must_use]
    pub fn kv(&self, name: &str) -> ValueId {
        assert!(
            self.caches
                .rows
                .iter()
                .any(|row| matches!(row, CacheRow::Kv { name: n, .. } if n == name)),
            "`{name}` is not a kv row the model declares",
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
            "`{name}` is not a state row the model declares",
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

    /// A page geometry the attention plans read: the All axis's, which
    /// only a model keeping per-token rows has.
    fn read_geometry(&self, kind: GeomKind) -> Value {
        self.per_token();
        self.geometry(kind)
    }

    fn geometry(&self, kind: GeomKind) -> Value {
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
            | GeomKind::WritePage { .. }
            | GeomKind::WriteOffset { .. } => (Dim::Tokens, Dtype::I32),
        };
        self.rec
            .input(
                RuntimeInput::Geometry { kind },
                Ty::Tensor {
                    shape: vec![rows],
                    dtype,
                },
            )
            .refined(self.over.clone())
    }

    fn per_token(&self) {
        assert!(
            self.caches.per_token(),
            "the model declares no per-token state",
        );
    }

    /// The window kv row `name` is retained through.
    fn window_of(&self, name: &str) -> Option<u32> {
        self.caches
            .rows
            .iter()
            .find_map(|row| match row {
                CacheRow::Kv {
                    name: n, window, ..
                } if n == name => Some(*window),
                _ => None,
            })
            .unwrap_or_else(|| panic!("`{name}` is not a kv row the model declares"))
    }
}
