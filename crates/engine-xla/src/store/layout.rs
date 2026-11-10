//! How this engine lays every pool plane into chunks of the one pool: the
//! slot shapes the load states (`engine::chunk::ChunkFacts`), where each
//! plane's run sits in its shape's slot, and the chunk size.
//!
//! The pool is one u32 buffer, `[chunks · C / 512, 128]` (the tiled
//! `[chunks, rows, 128]` realization, its two major axes merged). A kernel
//! reads a plane through a view of the whole pool at the plane's own element
//! and width (`[pool bytes / row bytes, width]`) — a lazy view that is never
//! built: its reads and writes go by pool rows, the elements of a row in the
//! packed rows order a Mosaic kernel's bitcast sees (`kernels_xla::hlo`
//! `pool_view`), so every run covers whole 512-byte rows — and a plane's run — its
//! `per` rows of one slot: `T` tokens of a kv or rs plane, one working
//! set's block of a state row — is a block of that view wherever it starts
//! at a multiple of its own bytes. The layout guarantees that for every slot
//! the runtime may hand out:
//!
//! - every run of a shape starts at a multiple of its bytes inside the slot;
//! - the slot's bytes `S` and the chunk's `C` are multiples of every run's;
//!
//! so a slot ref resolves to its *grain* index `q = (chunk · C + index · S)
//! / G` (`G = gcd(C, S)`), and the plane's block index is `q · G / run +
//! offset / run`: one affine map a plane ([`Plane::map`]), the kernels'
//! `PageMap`.
//!
//! Shapes: the kv list (every kv row outside the window pool, compressor
//! planes with their row), one window list (the windowed kv rows, `T_w` the
//! kv page), one fixed shape for each distinct state block size (a record
//! is a slot of each, cut into pieces where it is larger than a chunk), and
//! the rs list (buffered recurrence inputs, `T_rs` the kv page).

use engine::chunk::ShapeId;
use poem_ir::{CacheRow, Dtype, Trace};

use crate::error::{Fault, Result};

/// One row of the pool buffer: 128 u32 lanes.
pub const ROW_BYTES: u64 = 512;
/// One `(8, 128)` u32 tile.
pub const TILE_BYTES: u64 = 8 * ROW_BYTES;
/// The chunk size aimed at where the compiler names none (the TPU profile's).
pub const TARGET_CHUNK: u64 = 16 << 20;
/// The most a chunk may be stretched to hold whole runs of every plane.
const MOST_CHUNK: u64 = 256 << 20;

/// What a plane of the pool holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum What {
    /// A kv row's key plane (its only plane where keys and values share).
    Keys {
        row: u32,
    },
    Values {
        row: u32,
    },
    /// The compressor state of the kv row a pool gather keys by its pages:
    /// `which` 0 the kv projection, 1 its score.
    Compressor {
        row: u32,
        which: u8,
    },
    /// A recurrent (or conv, or frame-cache) state row.
    State {
        row: u32,
    },
    /// A buffered recurrence input, the rs layout's plane `plane`.
    Rs {
        plane: u32,
    },
}

/// One plane's place in its shape's slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    pub what: What,
    pub dtype: Dtype,
    /// The view's width, in elements.
    pub width: u32,
    /// The view's rows a run holds: the tokens of a list slot, a state
    /// block's rows.
    pub per: u32,
    /// A run's bytes: `per · width · element`.
    pub run: u64,
    /// The record piece it lies in (fixed shapes); 0 else.
    pub piece: u32,
    /// Bytes into its slot.
    pub offset: u64,
}

impl Plane {
    /// Bytes of one view row.
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        self.run / u64::from(self.per.max(1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The kv list.
    Kv,
    /// The window list, its rows retaining `window` tokens.
    Window { window: u32 },
    /// A fixed shape: state blocks of one size, a record `pieces` slots.
    Fixed { pieces: u32 },
    /// The rs list.
    Rs,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub id: ShapeId,
    pub kind: Kind,
    /// Tokens a slot holds (lists); 0 for fixed shapes.
    pub tokens: u32,
    pub slot_bytes: u64,
    pub planes: Vec<Plane>,
    /// The trace's cache rows the slot lays out, in order.
    pub rows: Vec<u32>,
}

impl Shape {
    #[must_use]
    pub fn pieces(&self) -> u32 {
        match self.kind {
            Kind::Fixed { pieces } => pieces,
            _ => 1,
        }
    }

    #[must_use]
    pub fn plane(&self, what: What) -> Option<&Plane> {
        self.planes.iter().find(|p| p.what == what)
    }
}

/// Every shape and the chunk size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub chunk_bytes: u64,
    pub shapes: Vec<Shape>,
}

/// One kv-like row a list lays out.
struct ListRow {
    row: u32,
    planes: Vec<(What, Dtype, u32)>,
}

pub(crate) fn elem_bytes(name: &str, dtype: Dtype) -> Result<u64> {
    poem_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Unbound {
        what: format!("`{name}` in {dtype:?}, which has no element size"),
    })
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        a.max(b)
    } else {
        a / gcd(a, b) * b
    }
}

fn align_up(x: u64, to: u64) -> u64 {
    x.div_ceil(to.max(1)) * to.max(1)
}

fn narrow(n: u64) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// A kv row's `(keys width, values width (None where shared), dtype,
/// windowed)`.
pub type KvRow = (u32, Option<u32>, Dtype, bool);

/// What the layout is built from: the trace's rows as the store reads them.
pub struct Rows<'a> {
    pub trace: Option<&'a Trace>,
    /// The kv page: the tokens of a kv, window and rs list slot.
    pub page: u32,
    /// The window the window list retains, when windowed rows live apart.
    pub window: Option<u32>,
    /// Each kv row's planes.
    pub kv: Vec<Option<KvRow>>,
    /// Each state row's `(per, width, dtype)` block.
    pub state: Vec<Option<(u32, u32, Dtype)>>,
    /// Each compressor's `(cache row, width)`, bf16 `[kv, score]`.
    pub compressor: Vec<(u32, u32)>,
    /// The rs layout's planes `(width, dtype)`.
    pub rs: Vec<(u32, Dtype)>,
}

impl Layout {
    /// The layout of `rows`, its chunk near `target` bytes.
    pub fn of(rows: &Rows<'_>, target: u64) -> Result<Layout> {
        Layout::of_with(rows, target, 1, 0)
    }

    /// [`Layout::of`] with a chunk that is also a multiple of `unit` and no
    /// smaller than `least` bytes: what the streamed weights' slots need
    /// (`crate::store::streamed`). `(1, 0)` is [`Layout::of`].
    pub fn of_with(rows: &Rows<'_>, target: u64, unit: u64, least: u64) -> Result<Layout> {
        let page = rows.page.max(1);
        let mut kv_rows: Vec<ListRow> = Vec::new();
        let mut window_rows: Vec<ListRow> = Vec::new();
        for (at, kv) in rows.kv.iter().enumerate() {
            let Some((keys, values, dtype, windowed)) = *kv else {
                continue;
            };
            let row = at as u32;
            let mut planes = vec![(What::Keys { row }, dtype, keys)];
            if let Some(values) = values {
                planes.push((What::Values { row }, dtype, values));
            }
            for &(held, width) in &rows.compressor {
                if held == row {
                    for which in 0..2u8 {
                        planes.push((What::Compressor { row, which }, Dtype::Bf16, width));
                    }
                }
            }
            if windowed {
                window_rows.push(ListRow { row, planes });
            } else {
                kv_rows.push(ListRow { row, planes });
            }
        }

        let mut shapes: Vec<Shape> = Vec::new();
        let mut next = 0u32;
        let mut id = || {
            let at = ShapeId(next);
            next += 1;
            at
        };
        if !kv_rows.is_empty() {
            shapes.push(list(id(), Kind::Kv, page, &kv_rows, rows.trace)?);
        }
        if !window_rows.is_empty() {
            let window = rows.window.ok_or_else(|| Fault::Unbound {
                what: "windowed kv rows with no window to keep them by".to_string(),
            })?;
            shapes.push(list(
                id(),
                Kind::Window { window },
                page,
                &window_rows,
                rows.trace,
            )?);
        }
        // One fixed shape a distinct state block size, in row order of its
        // first row.
        let mut blocks: Vec<(u64, Vec<Plane>, Vec<u32>)> = Vec::new();
        for (at, state) in rows.state.iter().enumerate() {
            let Some((per, width, dtype)) = *state else {
                continue;
            };
            let name = cache_name(rows.trace, at);
            let run = u64::from(per) * u64::from(width) * elem_bytes(name, dtype)?;
            let plane = Plane {
                what: What::State { row: at as u32 },
                dtype,
                width,
                per,
                run,
                piece: 0,
                offset: 0,
            };
            match blocks.iter_mut().find(|(held, ..)| *held == run) {
                Some((_, planes, cache)) => {
                    planes.push(plane);
                    cache.push(at as u32);
                }
                None => blocks.push((run, vec![plane], vec![at as u32])),
            }
        }
        let mut rs_shape: Option<Shape> = None;
        if !rows.rs.is_empty() {
            let mut planes = Vec::new();
            for (at, &(width, dtype)) in rows.rs.iter().enumerate() {
                let run = u64::from(page) * u64::from(width) * elem_bytes("rs buffer", dtype)?;
                planes.push(Plane {
                    what: What::Rs { plane: at as u32 },
                    dtype,
                    width,
                    per: page,
                    run,
                    piece: 0,
                    offset: 0,
                });
            }
            rs_shape = Some(lay(ShapeId(0), Kind::Rs, page, planes, Vec::new()));
        }

        // The chunk: a multiple of every run, of a tile, and of the kv list
        // slot where that costs nothing past the target; never smaller than
        // a list slot or a state block.
        let mut grain = lcm(TILE_BYTES, unit.max(1));
        let mut least = least;
        for shape in shapes.iter().chain(rs_shape.as_ref()) {
            least = least.max(shape.slot_bytes);
            for plane in &shape.planes {
                grain = lcm(grain, plane.run);
            }
        }
        for (run, ..) in &blocks {
            grain = lcm(grain, *run);
            least = least.max(*run);
        }
        if grain > MOST_CHUNK {
            return Err(Fault::Unbound {
                what: format!(
                    "a pool whose plane runs share no chunk size under {} MiB (their least \
                     common multiple is {grain} bytes)",
                    MOST_CHUNK >> 20
                ),
            });
        }
        let kv_slot = shapes
            .iter()
            .find(|s| s.kind == Kind::Kv)
            .map_or(1, |s| s.slot_bytes);
        let with_kv = lcm(grain, kv_slot);
        let grain = if with_kv <= target.max(least) {
            with_kv
        } else {
            grain
        };
        let chunk_bytes = align_up(target.max(least).max(grain), grain);

        let mut next = shapes.len() as u32;
        for (run, planes, cache) in blocks {
            // As many blocks a piece as a chunk holds.
            let per_piece = (chunk_bytes / run).max(1) as usize;
            let pieces = planes.len().div_ceil(per_piece);
            let per_piece = planes.len().div_ceil(pieces);
            let planes: Vec<Plane> = planes
                .into_iter()
                .enumerate()
                .map(|(i, plane)| Plane {
                    piece: (i / per_piece) as u32,
                    offset: (i % per_piece) as u64 * run,
                    ..plane
                })
                .collect();
            shapes.push(Shape {
                id: ShapeId(next),
                kind: Kind::Fixed {
                    pieces: pieces as u32,
                },
                tokens: 0,
                slot_bytes: align_up(per_piece as u64 * run, lcm(run, ROW_BYTES)),
                planes,
                rows: cache,
            });
            next += 1;
        }
        if let Some(mut rs) = rs_shape {
            rs.id = ShapeId(next);
            shapes.push(rs);
        }
        let layout = Layout {
            chunk_bytes,
            shapes,
        };
        layout.check()?;
        Ok(layout)
    }

    /// The invariants the views' block addressing stands on.
    fn check(&self) -> Result<()> {
        let c = self.chunk_bytes;
        if !c.is_multiple_of(TILE_BYTES) {
            return Err(bad(format!("chunk of {c} bytes is no whole tiles")));
        }
        for shape in &self.shapes {
            let s = shape.slot_bytes;
            if s == 0 || s > c || !s.is_multiple_of(ROW_BYTES) {
                return Err(bad(format!(
                    "shape {} slot of {s} bytes in a {c}-byte chunk",
                    shape.id
                )));
            }
            for plane in &shape.planes {
                let r = plane.run;
                // Whole pool rows: a row holds one plane's elements in the
                // packed rows order (`kernels_xla::hlo` pool views).
                if r == 0
                    || !r.is_multiple_of(ROW_BYTES)
                    || !c.is_multiple_of(r)
                    || !s.is_multiple_of(r)
                    || !plane.offset.is_multiple_of(r)
                    || plane.offset + r > s
                {
                    return Err(bad(format!(
                        "shape {} plane {:?}: a {r}-byte run at {} of a {s}-byte slot in a \
                         {c}-byte chunk",
                        shape.id, plane.what, plane.offset
                    )));
                }
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn shape(&self, id: ShapeId) -> Option<&Shape> {
        self.shapes.iter().find(|s| s.id == id)
    }

    #[must_use]
    pub fn kv(&self) -> Option<&Shape> {
        self.shapes.iter().find(|s| s.kind == Kind::Kv)
    }

    #[must_use]
    pub fn window(&self) -> Option<&Shape> {
        self.shapes
            .iter()
            .find(|s| matches!(s.kind, Kind::Window { .. }))
    }

    #[must_use]
    pub fn rs(&self) -> Option<&Shape> {
        self.shapes.iter().find(|s| s.kind == Kind::Rs)
    }

    pub fn fixed(&self) -> impl Iterator<Item = &Shape> {
        self.shapes
            .iter()
            .filter(|s| matches!(s.kind, Kind::Fixed { .. }))
    }

    /// The shape and plane `what` lies in.
    #[must_use]
    pub fn find(&self, what: What) -> Option<(&Shape, &Plane)> {
        self.shapes
            .iter()
            .find_map(|s| s.plane(what).map(|p| (s, p)))
    }

    /// The slots of `shape` one chunk holds.
    #[must_use]
    pub fn slots_per_chunk(&self, shape: &Shape) -> u32 {
        narrow(self.chunk_bytes / shape.slot_bytes.max(1))
    }

    /// The grain a slot of `shape` is named by: `gcd(C, S)`.
    #[must_use]
    pub fn grain(&self, shape: &Shape) -> u64 {
        gcd(self.chunk_bytes, shape.slot_bytes)
    }

    /// Slot `index` of physical chunk `chunk`, as the grain index the fire's
    /// tables carry.
    #[must_use]
    pub fn grain_of(&self, shape: &Shape, chunk: u32, index: u32) -> u64 {
        (u64::from(chunk) * self.chunk_bytes + u64::from(index) * shape.slot_bytes)
            / self.grain(shape)
    }

    /// `plane`'s block index of grain index `q`: `q · stride + offset`.
    #[must_use]
    pub fn map(&self, shape: &Shape, plane: &Plane) -> (i64, i64) {
        let g = self.grain(shape);
        (
            i64::try_from(g / plane.run).unwrap_or(i64::MAX),
            i64::try_from(plane.offset / plane.run).unwrap_or(i64::MAX),
        )
    }

    /// Every view of a pool of `chunks` chunks addresses its rows with an
    /// i32: refused where one would not.
    pub fn check_reach(&self, chunks: u32) -> Result<()> {
        let bytes = u64::from(chunks) * self.chunk_bytes;
        for shape in &self.shapes {
            for plane in &shape.planes {
                let rows = bytes / plane.row_bytes().max(1);
                if rows > i32::MAX as u64 {
                    return Err(Fault::Ceiling {
                        what: "rows of a pool view an i32 addresses",
                        need: rows,
                        have: i32::MAX as u64,
                    });
                }
            }
        }
        Ok(())
    }
}

fn bad(why: String) -> Fault {
    Fault::Unbound {
        what: format!("a pool layout that breaks block addressing: {why}"),
    }
}

fn cache_name(trace: Option<&Trace>, row: usize) -> &str {
    match trace.and_then(|t| t.caches.get(row)) {
        Some(CacheRow::Kv { name, .. } | CacheRow::State { name, .. }) => name,
        None => "?",
    }
}

/// A list shape over `rows`, `tokens` a slot.
fn list(
    id: ShapeId,
    kind: Kind,
    tokens: u32,
    rows: &[ListRow],
    trace: Option<&Trace>,
) -> Result<Shape> {
    let mut planes = Vec::new();
    let mut cache = Vec::new();
    for row in rows {
        cache.push(row.row);
        for &(what, dtype, width) in &row.planes {
            let name = cache_name(trace, row.row as usize);
            let run = u64::from(tokens) * u64::from(width) * elem_bytes(name, dtype)?;
            planes.push(Plane {
                what,
                dtype,
                width,
                per: tokens,
                run,
                piece: 0,
                offset: 0,
            });
        }
    }
    Ok(lay(id, kind, tokens, planes, cache))
}

/// Lays `planes` one after another, each at a multiple of its own run,
/// the slot a multiple of every run and of a pool row.
fn lay(id: ShapeId, kind: Kind, tokens: u32, planes: Vec<Plane>, rows: Vec<u32>) -> Shape {
    let mut at = 0u64;
    let mut all = ROW_BYTES;
    let planes: Vec<Plane> = planes
        .into_iter()
        .map(|plane| {
            let offset = align_up(at, plane.run);
            at = offset + plane.run;
            all = lcm(all, plane.run);
            Plane { offset, ..plane }
        })
        .collect();
    Shape {
        id,
        kind,
        tokens,
        slot_bytes: align_up(at, all),
        planes,
        rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(shape: &Shape, at: usize) -> Plane {
        shape.planes[at]
    }

    #[test]
    fn every_run_lands_on_a_block_of_its_view_in_any_slot() {
        let rows = Rows {
            trace: None,
            page: 16,
            window: Some(64),
            kv: vec![
                Some((512, Some(512), Dtype::Bf16, false)),
                Some((576, None, Dtype::Bf16, false)),
                Some((256, Some(256), Dtype::Bf16, true)),
            ],
            state: vec![
                None,
                None,
                None,
                Some((2048, 128, Dtype::Bf16)),
                Some((8, 2304, Dtype::Bf16)),
                Some((2048, 128, Dtype::Bf16)),
            ],
            compressor: vec![(0, 256)],
            rs: vec![(6144, Dtype::Bf16), (32, Dtype::F32)],
        };
        let layout = Layout::of(&rows, TARGET_CHUNK).expect("the layout");
        let c = layout.chunk_bytes;
        assert!(c >= TARGET_CHUNK && c.is_multiple_of(TILE_BYTES), "{c}");
        let kinds: Vec<Kind> = layout.shapes.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Kind::Kv,
                Kind::Window { window: 64 },
                Kind::Fixed { pieces: 1 },
                Kind::Fixed { pieces: 1 },
                Kind::Rs
            ]
        );
        for shape in &layout.shapes {
            let spc = layout.slots_per_chunk(shape);
            assert!(spc >= 1);
            for plane in &shape.planes {
                let (stride, offset) = layout.map(shape, plane);
                for chunk in [1u32, 2, 7] {
                    for index in [0, spc - 1] {
                        let bytes = u64::from(chunk) * c
                            + u64::from(index) * shape.slot_bytes
                            + plane.offset;
                        assert_eq!(bytes % plane.run, 0, "{:?}", plane.what);
                        let q = layout.grain_of(shape, chunk, index) as i64;
                        assert_eq!(
                            (q * stride + offset) as u64,
                            bytes / plane.run,
                            "{:?}",
                            plane.what
                        );
                    }
                }
            }
        }
        let kv = layout.kv().expect("a kv list");
        assert_eq!(kv.tokens, 16);
        assert_eq!(c % kv.slot_bytes, 0, "the kv list fills its chunks");
        assert_eq!(plane(kv, 0).run, 16 * 512 * 2);
        assert_eq!(
            kv.planes.len(),
            5,
            "keys, values, the latent, two compressor planes"
        );
    }

    #[test]
    fn a_record_larger_than_a_chunk_is_cut_between_blocks() {
        let rows = Rows {
            trace: None,
            page: 16,
            window: None,
            kv: Vec::new(),
            state: vec![Some((2048, 1024, Dtype::Bf16)); 6],
            compressor: Vec::new(),
            rs: Vec::new(),
        };
        // Blocks of 4 MiB, a chunk of 8: two a piece.
        let layout = Layout::of(&rows, 8 << 20).expect("the layout");
        let fixed = layout.fixed().next().expect("a fixed shape");
        assert_eq!(fixed.kind, Kind::Fixed { pieces: 3 });
        assert_eq!(fixed.slot_bytes, 8 << 20);
        let pieces: Vec<(u32, u64)> = fixed.planes.iter().map(|p| (p.piece, p.offset)).collect();
        assert_eq!(
            pieces,
            vec![
                (0, 0),
                (0, 4 << 20),
                (1, 0),
                (1, 4 << 20),
                (2, 0),
                (2, 4 << 20)
            ]
        );
    }
}
