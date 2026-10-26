//! The anchored region: a contraction and the pointwise math around it.
//!
//! - **The anchor** ([`Contraction`]): `y = a · wᵀ` over `m` rows of `a`
//!   (`k` wide) and a weight of `n` rows (`k` wide), its layout (one bank,
//!   or `parts` banks of `n` rows packed one after another, a GLU's
//!   `[gate | up]`) and how its operands are stored ([`Quant`]). Each part
//!   lands one product a row and column, summed in f32 and read by the
//!   epilogue as it is, or rounded first to the dtype it declares where it
//!   declares one.
//! - **The prologue** (optional, a [`Pointwise`]): a region over the
//!   operand tile, a function of `a[m][k]` at each `(m, k)`: what feeds the
//!   contraction (a dequantized operand, an activation's scale).
//! - **The epilogue** (optional, a [`Pointwise`]): a region over the
//!   product tile, a function of the parts' products at each `(m, n)`: what
//!   lands (a bias, a soft cap, a residual, a GLU of a packed bank's two
//!   products). No epilogue is the plain store: the products as they are.
//!
//! Each region is a [`Region`] whose row is the tile's inner axis (`k` for
//! the prologue, `n` for the epilogue) and whose scalars are one a tile
//! row (`m`), so it is pointwise in its index space exactly when nothing
//! in it combines along the inner axis: no reduction (a sum over `n` in an
//! epilogue is not a function of one `(m, n)`). [`Anchored::verify`] holds
//! a region to that, so a skeleton that fills its hooks with one never has to
//! ask. Its parameters are, in order, the tile's own values (the operand;
//! each part's product) and then its [`Input`]s, each a tile, a vector a
//! column, a vector a row or one value.
//!
//! **A head's epilogue.** An epilogue may combine columns within a group of
//! them, its span ([`Pointwise::span`]): each head it reduces (a head's
//! norm, `ReduceHeads`) and each column's gathered mate (a rotary pair,
//! `Peer` at a column that is a function of the column alone), every group
//! aligned, dividing `n` and at most [`MAX_SPAN`] wide, so a skeleton whose
//! tile holds whole groups computes it a tile at a time. Its rotation's
//! angle is the row's position (an [`Input::Row`]) by a frequency each
//! column states (the region's column), the table of them the schedule's.
//! A sum over the whole row is one group of every column, legal where the
//! row is at most [`MAX_SPAN`] wide (a row one head wide); over a wider row,
//! a head wider than [`MAX_SPAN`], or a gather at a column its values pick,
//! it crosses tiles, and is refused.
//!
//! The interpreter ([`eval`]) is the contract a kernel is held to: the
//! contraction summed in f64 and rounded to f32 once (as the interpreter
//! sums a reduction), its declared rounding, then each
//! region by [`crate::interp::eval`] a tile row at a time. A kernel sums in
//! its own order, so it is held to it within a tolerance, never bit for bit
//! against another kernel; what it must land exactly is the rounding the
//! IR declares, and the same bits on the same input.

use std::fmt;

use dtype::Dtype;

use crate::interp::{self, Arg, InterpError};
use crate::region::{Op, Region, RegionError, Shape, ValueId};
use crate::skeleton::{round_bf16, round_e4m3};

/// The widest group of a product tile's columns an epilogue combines
/// within: a head, which a skeleton holds whole (a contraction skeleton's
/// tile, or a row skeleton's row), so a head's norm (its sum of squares over
/// the head) and its rotary turn (each column with its mate in the head)
/// are its own alone. The widest head a catalog model norms or turns is 512
/// columns (Gemma 4's global layers); a backend serves a span its skeletons
/// hold.
pub const MAX_SPAN: u32 = 512;

/// How the weight's rows lay out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// One bank of `n` rows: one product a `(m, n)`.
    Dense,
    /// `parts` banks of `n` rows each, one after another (a GLU's packed
    /// `[gate | up]`): part `p`'s column `c` is weight row `p * n + c`, and
    /// each `(m, n)` has one product a part.
    Packed(u32),
}

impl Layout {
    /// The products each `(m, n)` has.
    #[must_use]
    pub const fn parts(self) -> u32 {
        match self {
            Layout::Dense => 1,
            Layout::Packed(parts) => parts,
        }
    }
}

/// How the contraction's operands are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quant {
    /// Both at one dtype, read as they are.
    Plain(Dtype),
}

/// The anchor: `y[p][m][c] = Σₖ a[m][k] · w[p · n + c][k]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Contraction {
    /// The columns each part lands.
    pub n: u32,
    /// The contraction's length.
    pub k: u32,
    pub layout: Layout,
    pub quant: Quant,
    /// The dtype each product rounds to before an epilogue reads it, where
    /// an op between them rounds it (a cast the program states), or `None`:
    /// the epilogue reads the f32 accumulator, as a matmul and its bias are
    /// one sum, and only the epilogue's own values round. The plain store
    /// rounds the product to the dtype it lands.
    pub acc: Option<Dtype>,
}

/// A tile's index space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Space {
    /// The operand tile, `(m, k)`: the prologue's.
    Operand,
    /// The product tile, `(m, n)`: the epilogue's.
    Product,
}

/// What one of a region's inputs is, over its tile's `(m, i)`, `i` the
/// inner axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    /// One value an `(m, i)` (a residual).
    Tile,
    /// One value a column `i`, every row reading it (a bias).
    Column,
    /// One value a column of a head `size` columns wide, every head of every
    /// row reading it: column `i` reads entry `i % size` (a head's norm
    /// weight).
    Head(u32),
    /// One value a row `m` (a row's scale).
    Row,
    /// One value for the whole tile (a layer's scale).
    One,
    /// The operand row's norm scale the contraction computes as it streams
    /// the row ([`RowNorm`]): one value a row, no plane.
    Norm,
}

impl Input {
    /// The shape its parameter has in the region: a tile row's elements or
    /// one value a tile row.
    #[must_use]
    pub const fn shape(self) -> Shape {
        match self {
            Input::Tile | Input::Column | Input::Head(_) => Shape::Row,
            Input::Row | Input::One | Input::Norm => Shape::Scalar,
        }
    }
}

/// A region over a tile, pointwise in its index space: its parameters the
/// tile's own values (the operand, or each part's product), then
/// `inputs`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pointwise {
    pub region: Region,
    pub inputs: Vec<Input>,
}

/// The columns of the product tile one result lands: `width` of them from
/// column `from` (a split's piece: a packed qkv product's q, k or v), each
/// row of its plane `width` wide; or, where `head` is set, `head` of them
/// from every `stride` (a head-major slice: Qwen 3.5's q and gate, a head of
/// each interleaved), `width / head` heads, landed one after another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Window {
    pub from: u32,
    pub width: u32,
    /// The columns a head lands of every `stride` (0: the window is whole).
    pub head: u32,
    pub stride: u32,
}

impl Window {
    /// `width` columns from `from`.
    #[must_use]
    pub const fn of(from: u32, width: u32) -> Window {
        Window {
            from,
            width,
            head: 0,
            stride: 0,
        }
    }

    /// `width / head` heads of `head` columns, one every `stride` from `from`.
    #[must_use]
    pub const fn strided(from: u32, width: u32, head: u32, stride: u32) -> Window {
        Window {
            from,
            width,
            head,
            stride,
        }
    }

    /// The tile column the window's `j`th landed column is.
    #[must_use]
    pub const fn column(self, j: u32) -> u32 {
        if self.head == 0 {
            self.from + j
        } else {
            self.from + j / self.head * self.stride + j % self.head
        }
    }

    /// The landed column the tile's column `c` is, where the window holds it.
    #[must_use]
    pub const fn landed(self, c: u32) -> Option<u32> {
        if c < self.from {
            return None;
        }
        let at = c - self.from;
        if self.head == 0 {
            return if at < self.width { Some(at) } else { None };
        }
        let (h, j) = (at / self.stride, at % self.stride);
        if j < self.head && h < self.width / self.head {
            Some(h * self.head + j)
        } else {
            None
        }
    }

    /// One past the last tile column it holds.
    #[must_use]
    pub const fn end(self) -> u32 {
        if self.width == 0 {
            self.from
        } else {
            self.column(self.width - 1) + 1
        }
    }

    /// Whether it is whole, no head stride.
    #[must_use]
    pub const fn whole(self) -> bool {
        self.head == 0
    }
}

/// How a paged cache holds its elements: bf16, or e4m3 under a scale (each
/// element `e4m3(x / scale)`, saturating; vLLM's `fp8` KV is a unit scale).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CacheDtype {
    Bf16,
    E4m3 { scale: f32 },
}

impl CacheDtype {
    /// `x` as the cache stores it, read back: rounded to bf16 as the
    /// result's own plane would hold it, then to the cache's element.
    #[must_use]
    pub fn stored(self, x: f32) -> f32 {
        let x = round_bf16(x);
        match self {
            CacheDtype::Bf16 => x,
            CacheDtype::E4m3 { scale } => round_e4m3(x / scale),
        }
    }
}

/// A result landed in a paged cache's plane (a key's or a value's) rather
/// than a plane of its own: each row at the cache row its page and slot
/// name, `page · page_size + slot`, the row whose slot lies past its page
/// landing nothing. `page` and `slot` are the epilogue's inputs (each one
/// value a row, an i32), the slot mapping its fire states; how a cache lays
/// out its pages and which rows are live is the backend's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Paged {
    pub plane: CachePlane,
    pub dtype: CacheDtype,
    pub page: u32,
    pub slot: u32,
}

/// Which of a cache's planes a paged result lands in: its keys' or its
/// values' (a shared-kv latent's one plane is its keys').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CachePlane {
    Keys,
    Values,
}

/// How a norm's gain `g` folds into the bank the contraction reads: none
/// (a norm of no weight), `w ⊙ g`, or `w ⊙ (1 + g)` (Gemma's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gain {
    None,
    Scale,
    PlusOne,
}

/// The operand's rows RMS-normed before the contraction, rewritten as a
/// scale of each row's products (`rmsnorm(a, g) · wᵀ = r ⊙ (a · w'ᵀ)`):
/// the contraction reads each row as it is stored and computes its scale
/// `r[m] = rsqrt(Σₖ a[m][k]² / k + eps)` as it streams the row, which the
/// epilogue reads as its [`Input::Norm`]; and it reads the bank as the
/// declared transform [`fold_bank`] lands it, the gain folded in and each
/// weight rounded once more to the bank's dtype. That rounding is the
/// rewrite's own, where the norm's rounding of its normed row was the
/// program's: a fold is held to the unfolded program within a tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowNorm {
    pub eps: f32,
    pub gain: Gain,
}

/// A contraction with its prologue and epilogue.
#[derive(Clone, Debug, PartialEq)]
pub struct Anchored {
    pub contraction: Contraction,
    pub prologue: Option<Pointwise>,
    pub epilogue: Option<Pointwise>,
    /// The window of the tile each result lands, one a result, or none:
    /// every result lands every column.
    pub windows: Vec<Window>,
    /// The operand's rows normed, their scale the epilogue's
    /// [`Input::Norm`] and the gain folded into the bank.
    pub norm: Option<RowNorm>,
    /// Where each result lands in a paged cache ([`Paged`]), one a result,
    /// or none: every result its own plane.
    pub paged: Vec<Option<Paged>>,
}

/// Why an anchored region is not one.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnchorError {
    /// The contraction itself: what it lacks.
    Contraction(&'static str),
    /// A region that is not a function of one row.
    Region { space: Space, why: RegionError },
    /// A region whose row is not its tile's inner axis.
    Width {
        space: Space,
        width: u32,
        wants: u32,
    },
    /// A region of more or fewer parameters than its tile's values and its
    /// inputs.
    Params {
        space: Space,
        params: usize,
        wants: usize,
    },
    /// A parameter whose shape is not its value's (a product or a tile's
    /// input is a row, a row's input or one value a scalar).
    Param { space: Space, param: u32 },
    /// An op that combines along the tile's inner axis (a reduction across
    /// `n` in an epilogue, across `k` in a prologue): not pointwise.
    Across { space: Space, op: u32 },
    /// An epilogue whose gathers and head reductions stay within no group of
    /// columns at most [`MAX_SPAN`] wide that divides `n` (a head's norm or
    /// turn across more columns than one tile holds), or whose gather reads
    /// a column its values pick.
    Span { space: Space, op: u32 },
    /// A prologue that lands other than one operand, or an epilogue that
    /// lands nothing.
    Results { space: Space, results: usize },
    /// A result that is not an f32 element of the tile.
    Result { space: Space, result: u32 },
    /// Windows not one a result, or one past the tile's columns or of none.
    Window { result: u32 },
    /// A paged result whose page or slot is no input one value a row, a
    /// scale no finite positive one, or pagings not one a result: a result
    /// a cache cannot address by row.
    Paged { result: u32 },
    /// A row's norm scale read with no norm declared, or a norm declared and
    /// its scale read by no epilogue (the products landed unnormed), or a
    /// norm over a prologue's operand (the scale is of the row as stored).
    Norm,
}

impl fmt::Display for AnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnchorError::Contraction(why) => write!(f, "the contraction: {why}"),
            AnchorError::Region { space, why } => write!(f, "the {space} region: {why}"),
            AnchorError::Width {
                space,
                width,
                wants,
            } => write!(
                f,
                "the {space} region's row is {width} wide and its tile's inner axis {wants}"
            ),
            AnchorError::Params {
                space,
                params,
                wants,
            } => write!(
                f,
                "the {space} region takes {params} parameter(s), and its tile's values and \
                 inputs are {wants}"
            ),
            AnchorError::Param { space, param } => write!(
                f,
                "the {space} region's parameter {param} is not of its value's shape"
            ),
            AnchorError::Across { space, op } => write!(
                f,
                "the {space} region's op {op} combines along the tile's inner axis, and a \
                 region over a tile is pointwise in it"
            ),
            AnchorError::Span { space, op } => write!(
                f,
                "the {space} region's op {op} combines columns past one group of at most \
                 {MAX_SPAN} that divides the tile's, or gathers from a column its values pick"
            ),
            AnchorError::Results { space, results } => {
                write!(f, "the {space} region lands {results} result(s)")
            }
            AnchorError::Window { result } => write!(
                f,
                "result {result}'s window is not one of the tile's columns, or the windows are \
                 not one a result"
            ),
            AnchorError::Paged { result } => write!(
                f,
                "result {result} lands in a cache it cannot address by row: its page or slot is \
                 no input one value a row, or its scale no finite positive one"
            ),
            AnchorError::Norm => f.write_str(
                "a row's norm scale read with no norm declared, a declared norm whose scale \
                 no epilogue reads, or a norm beside a prologue",
            ),
            AnchorError::Result { space, result } => write!(
                f,
                "the {space} region's result {result} is not an f32 element of its tile"
            ),
        }
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Space::Operand => "prologue",
            Space::Product => "epilogue",
        })
    }
}

impl std::error::Error for AnchorError {}

impl Pointwise {
    /// Whether this is a region over `space`'s tile, `inner` wide, its
    /// first `own` parameters the tile's values: typed, of the parameters
    /// its tile and inputs give, combining nothing along `inner`, and
    /// landing f32 elements of the tile.
    fn verify(&self, space: Space, inner: u32, own: u32) -> Result<(), AnchorError> {
        let region = &self.region;
        let types = region
            .types()
            .map_err(|why| AnchorError::Region { space, why })?;
        if region.width != inner {
            return Err(AnchorError::Width {
                space,
                width: region.width,
                wants: inner,
            });
        }
        let wants = own as usize + self.inputs.len();
        if region.params.len() != wants {
            return Err(AnchorError::Params {
                space,
                params: region.params.len(),
                wants,
            });
        }
        let shapes = (0..own)
            .map(|_| Shape::Row)
            .chain(self.inputs.iter().map(|input| input.shape()));
        for ((param, &shape), wants) in (0u32..).zip(&region.params).zip(shapes) {
            if shape != wants {
                return Err(AnchorError::Param { space, param });
            }
        }
        for (param, input) in (own..).zip(&self.inputs) {
            if let Input::Head(size) = *input
                && (size == 0 || !inner.is_multiple_of(size))
            {
                return Err(AnchorError::Param { space, param });
            }
        }
        for (op, at) in region.ops.iter().zip(0u32..) {
            // Every op is classed: one a new kind of op adds is pointwise
            // only once it is named here.
            let pointwise = match op {
                Op::Const(_)
                | Op::Unary(..)
                | Op::Binary(..)
                | Op::Select(..)
                | Op::Cast(..)
                | Op::Broadcast(_)
                | Op::Fma(..)
                | Op::Column => true,
                // A head's sum and a gather within the row read other
                // columns than their own: an epilogue's, within its span
                // ([`Pointwise::span`]).
                Op::ReduceHeads(..) | Op::Peer(..) => space == Space::Product,
                // A sum across the whole row reads every column: an
                // epilogue's where the row is one span (a row one head
                // wide, [`MAX_SPAN`] at most).
                Op::Reduce(..) => space == Space::Product && inner <= MAX_SPAN,
            };
            if !pointwise {
                return Err(AnchorError::Across { space, op: at });
            }
        }
        if space == Space::Product {
            self.span_of(inner)?;
        }
        let results = region.results.len();
        if results == 0 || (space == Space::Operand && results != 1) {
            return Err(AnchorError::Results { space, results });
        }
        for (result, &value) in (0u32..).zip(&region.results) {
            if types[value as usize] != (Shape::Row, crate::region::Elem::F32) {
                return Err(AnchorError::Result { space, result });
            }
        }
        Ok(())
    }
}

impl Pointwise {
    /// The columns of each group its ops combine within, over a tile `n`
    /// wide: 1 where it is pointwise, else the narrowest group (dividing
    /// `n`, at most [`MAX_SPAN`]) that holds each head it reduces and each
    /// column's gathered mate. A skeleton serves it only where a tile holds
    /// whole groups. Its region is taken as verified.
    #[must_use]
    pub fn span(&self, n: u32) -> u32 {
        self.span_of(n).unwrap_or(n)
    }

    fn span_of(&self, n: u32) -> Result<u32, AnchorError> {
        let space = Space::Product;
        let region = &self.region;
        let mut heads = 1u32;
        let mut peers: Vec<(u32, ValueId)> = Vec::new();
        for (op, at) in region.ops.iter().zip(0u32..) {
            match *op {
                Op::ReduceHeads(_, _, size) => heads = lcm(heads, size),
                // the row whole: one group of every column
                Op::Reduce(..) => heads = lcm(heads, n),
                Op::Peer(_, mate) => peers.push((at, mate)),
                _ => {}
            }
        }
        if peers.is_empty() && heads == 1 {
            return Ok(1);
        }
        // Each gather's column, a function of the column alone (no
        // parameter in it), over the tile's every column.
        let mut mates: Vec<(u32, Vec<u32>)> = Vec::with_capacity(peers.len());
        for &(op, mate) in &peers {
            let columns = self
                .columns(mate, n)
                .ok_or(AnchorError::Span { space, op })?;
            mates.push((op, columns));
        }
        let first = peers
            .first()
            .map_or_else(|| region.ops.len() as u32, |(op, _)| *op);
        (1..=n.min(MAX_SPAN))
            .filter(|g| n.is_multiple_of(*g) && g.is_multiple_of(heads))
            .find(|&g| {
                mates.iter().all(|(_, columns)| {
                    (0u32..)
                        .zip(columns)
                        .all(|(c, &mate)| mate < n && mate / g == c / g)
                })
            })
            .ok_or_else(|| {
                let op = mates
                    .iter()
                    .find(|(_, columns)| columns.iter().any(|&mate| mate >= n))
                    .map_or(first, |(op, _)| *op);
                AnchorError::Span { space, op }
            })
    }

    /// Whether, over a tile `n` wide, every `w` columns from a multiple of
    /// `w` compute what the first `w` do: each value of the column alone
    /// that the ops over the parameters read repeats every `w` columns (a
    /// head's position in its head, its pair's frequency), and each
    /// gathered mate sits as far from its column in every slice. A slice of
    /// such a tile is then any slice's region, each a part of one launch's
    /// rows.
    #[must_use]
    pub fn repeats(&self, n: u32, w: u32) -> bool {
        let region = &self.region;
        if w == 0 || !n.is_multiple_of(w) {
            return false;
        }
        if w == n {
            return true;
        }
        let params = region.params.len() as u32;
        // whether each value is a function of the column alone
        let mut alone = vec![false; params as usize + region.ops.len()];
        for (at, op) in region.ops.iter().enumerate() {
            alone[params as usize + at] = match op {
                Op::Peer(..) | Op::Reduce(..) | Op::ReduceHeads(..) => false,
                op => op.args().iter().all(|&a| a >= params && alone[a as usize]),
            };
        }
        // what the ops over the parameters read of the column alone, and
        // each gather's mate
        let mut read: Vec<ValueId> = Vec::new();
        let mut mates: Vec<ValueId> = Vec::new();
        for (at, op) in region.ops.iter().enumerate() {
            if alone[params as usize + at] {
                continue;
            }
            // a gather's mate is its column's: as far from it in each slice
            let mate = match *op {
                Op::Peer(_, mate) => {
                    mates.push(mate);
                    Some(mate)
                }
                _ => None,
            };
            for a in op.args() {
                if Some(a) != mate && a >= params && alone[a as usize] && !read.contains(&a) {
                    read.push(a);
                }
            }
        }
        for &r in &region.results {
            if r >= params && alone[r as usize] && !read.contains(&r) {
                read.push(r);
            }
        }
        let periodic = |values: &[f32], shifted: bool| {
            let (n, w) = (n as usize, w as usize);
            (w..n).all(|c| {
                let (x, first) = (values[c], values[c % w]);
                if shifted {
                    x - c as f32 == first - (c % w) as f32
                } else {
                    x.to_bits() == first.to_bits()
                }
            })
        };
        let probe = |v: ValueId| -> Option<Vec<f32>> {
            let mut ops: Vec<Op> = region.ops[..(v - params + 1) as usize].to_vec();
            ops.push(Op::Cast(v, crate::region::Elem::F32));
            let probe = Region {
                width: n,
                params: region.params.clone(),
                ops,
                results: vec![params + (v - params + 1)],
                rounds: Vec::new(),
            };
            let args: Vec<Arg> = region
                .params
                .iter()
                .map(|shape| match shape {
                    Shape::Scalar => Arg::Scalar(0.0),
                    _ => Arg::Row(vec![0.0; n as usize]),
                })
                .collect();
            interp::eval(&probe, &args)
                .ok()
                .map(|mut l| l.swap_remove(0))
        };
        read.iter()
            .all(|&v| probe(v).is_some_and(|values| periodic(&values, false)))
            && mates.iter().all(|&m| {
                self.columns(m, n).is_some_and(|c| {
                    periodic(&c.iter().map(|x| *x as f32).collect::<Vec<_>>(), true)
                })
            })
    }

    /// Value `v`'s u32 at each of `n` columns, where it is a function of
    /// the column alone: its ops read no parameter. `None` where one does.
    fn columns(&self, v: ValueId, n: u32) -> Option<Vec<u32>> {
        let region = &self.region;
        let params = region.params.len() as u32;
        // The ops `v` reads, transitively.
        let mut needed = vec![false; params as usize + region.ops.len()];
        needed[v as usize] = true;
        for at in (0..region.ops.len() as u32).rev() {
            let id = params + at;
            if !needed[id as usize] {
                continue;
            }
            for arg in region.ops[at as usize].args() {
                if arg < params {
                    return None;
                }
                needed[arg as usize] = true;
            }
        }
        // The region's own ops up to `v`, landing it as an f32 (a column's
        // index is exact there): its parameters unread, each given zeros.
        let mut ops: Vec<Op> = region.ops[..(v - params + 1) as usize].to_vec();
        ops.push(Op::Cast(v, crate::region::Elem::F32));
        let probe = Region {
            width: n,
            params: region.params.clone(),
            ops,
            results: vec![params + (v - params + 1)],
            rounds: Vec::new(),
        };
        let args: Vec<Arg> = region
            .params
            .iter()
            .map(|shape| match shape {
                Shape::Scalar => Arg::Scalar(0.0),
                _ => Arg::Row(vec![0.0; n as usize]),
            })
            .collect();
        let landed = interp::eval(&probe, &args).ok()?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(landed[0].iter().map(|x| *x as u32).collect())
    }
}

const fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

const fn lcm(a: u32, b: u32) -> u32 {
    a / gcd(a, b) * b
}

impl Anchored {
    /// The plain contraction: no prologue, the products stored as they are.
    #[must_use]
    pub const fn plain(contraction: Contraction) -> Anchored {
        Anchored {
            contraction,
            prologue: None,
            epilogue: None,
            windows: Vec::new(),
            norm: None,
            paged: Vec::new(),
        }
    }

    /// Whether this is an anchored region: a contraction of some extent,
    /// each region pointwise over its tile ([`Space`]) and of the
    /// parameters its tile and inputs give, an epilogue combining columns
    /// only within its span ([`Pointwise::span`]).
    pub fn verify(&self) -> Result<(), AnchorError> {
        let c = &self.contraction;
        if c.n == 0 || c.k == 0 {
            return Err(AnchorError::Contraction("it contracts or lands nothing"));
        }
        if c.layout.parts() == 0 || matches!(c.layout, Layout::Packed(1)) {
            return Err(AnchorError::Contraction(
                "a packed bank of fewer than two parts",
            ));
        }
        if !matches!(c.acc, None | Some(Dtype::Bf16 | Dtype::F16)) {
            return Err(AnchorError::Contraction(
                "its products round to no narrower float",
            ));
        }
        let Quant::Plain(dtype) = c.quant;
        if !matches!(dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
            return Err(AnchorError::Contraction(
                "its operands are stored in no float",
            ));
        }
        if let Some(prologue) = &self.prologue {
            prologue.verify(Space::Operand, c.k, 1)?;
        }
        if let Some(epilogue) = &self.epilogue {
            epilogue.verify(Space::Product, c.n, c.layout.parts())?;
        }
        let reads_norm = |p: &Pointwise| p.inputs.contains(&Input::Norm);
        if self.prologue.as_ref().is_some_and(reads_norm)
            || self.norm.is_some()
                != self.epilogue.as_ref().is_some_and(reads_norm)
            || (self.norm.is_some() && self.prologue.is_some())
        {
            return Err(AnchorError::Norm);
        }
        if self.norm.is_some_and(|norm| !(norm.eps >= 0.0 && norm.eps.is_finite())) {
            return Err(AnchorError::Norm);
        }
        if !self.paged.is_empty() {
            let inputs = self.epilogue.as_ref().map_or(&[][..], |e| &e.inputs[..]);
            let row = |k: u32| inputs.get(k as usize) == Some(&Input::Row);
            if self.paged.len() != self.results() {
                let result = u32::try_from(self.paged.len().min(self.results())).unwrap_or(0);
                return Err(AnchorError::Paged { result });
            }
            for (result, paged) in (0u32..).zip(&self.paged) {
                let Some(paged) = paged else { continue };
                let scaled = match paged.dtype {
                    CacheDtype::Bf16 => true,
                    CacheDtype::E4m3 { scale } => scale.is_finite() && scale > 0.0,
                };
                if !row(paged.page) || !row(paged.slot) || !scaled {
                    return Err(AnchorError::Paged { result });
                }
            }
        }
        if !self.windows.is_empty() {
            if self.windows.len() != self.results() {
                let result = u32::try_from(self.windows.len().min(self.results())).unwrap_or(0);
                return Err(AnchorError::Window { result });
            }
            for (result, w) in (0u32..).zip(&self.windows) {
                let strided = w.head == 0
                    || (w.stride >= w.head && w.width.is_multiple_of(w.head));
                let inside = w.from.checked_add(w.width).is_some()
                    && w.from.checked_add(w.stride).is_some()
                    && w.end() <= c.n;
                if w.width == 0 || !strided || !inside {
                    return Err(AnchorError::Window { result });
                }
            }
        }
        Ok(())
    }

    /// The same function of rows normed by `norm` before the contraction, as
    /// one over the rows as they are stored ([`RowNorm`]): each part's
    /// product scaled by the row's norm scale (an [`Input::Norm`] appended
    /// to the epilogue's inputs) before the epilogue reads it, the gain
    /// folded into the bank. `None` where it has no epilogue (a plain
    /// store's products would land unnormed), a prologue, or a norm already.
    #[must_use]
    pub fn normed(&self, norm: RowNorm) -> Option<Anchored> {
        if self.norm.is_some() || self.prologue.is_some() {
            return None;
        }
        let epi = self.epilogue.as_ref()?;
        let parts = self.contraction.layout.parts();
        let region = &epi.region;
        let params = u32::try_from(region.params.len()).ok()?;
        // The scale, then each part's product times it; every later value
        // moved past the scale and the parts' products.
        let scale = params;
        let at = |v: ValueId| {
            if v < parts {
                params + 1 + v
            } else if v < params {
                v
            } else {
                v + 1 + parts
            }
        };
        let mut ops: Vec<Op> = (0..parts)
            .map(|p| Op::Binary(crate::region::Binary::Mul, p, scale))
            .collect();
        for op in &region.ops {
            ops.push(match *op {
                Op::Const(l) => Op::Const(l),
                Op::Column => Op::Column,
                Op::Unary(u, a) => Op::Unary(u, at(a)),
                Op::Binary(b, x, y) => Op::Binary(b, at(x), at(y)),
                Op::Select(c, x, y) => Op::Select(at(c), at(x), at(y)),
                Op::Cast(a, e) => Op::Cast(at(a), e),
                Op::Broadcast(a) => Op::Broadcast(at(a)),
                Op::Reduce(r, a) => Op::Reduce(r, at(a)),
                Op::ReduceHeads(r, a, size) => Op::ReduceHeads(r, at(a), size),
                Op::Fma(x, y, z) => Op::Fma(at(x), at(y), at(z)),
                Op::Peer(src, c) => Op::Peer(at(src), at(c)),
            });
        }
        let mut shapes = region.params.clone();
        shapes.push(Shape::Scalar);
        let mut inputs = epi.inputs.clone();
        inputs.push(Input::Norm);
        Some(Anchored {
            epilogue: Some(Pointwise {
                region: Region {
                    width: region.width,
                    params: shapes,
                    ops,
                    results: region.results.iter().map(|&r| at(r)).collect(),
                    rounds: region.rounds.iter().map(|&(v, d)| (at(v), d)).collect(),
                },
                inputs,
            }),
            norm: Some(norm),
            ..self.clone()
        })
    }

    /// The window result `k` lands: its own, or every column.
    #[must_use]
    pub fn window(&self, k: usize) -> Window {
        self.windows.get(k).copied().unwrap_or(Window::of(0, self.contraction.n))
    }

    /// The tiles it lands, each `m x n`: the epilogue's results, or each
    /// part's products where there is none.
    #[must_use]
    pub fn results(&self) -> usize {
        self.epilogue.as_ref().map_or_else(
            || self.contraction.layout.parts() as usize,
            |epilogue| epilogue.region.results.len(),
        )
    }

    /// Its key: a hash of what it is, not of any text a backend writes for
    /// it. Two anchored regions of one key are the same function.
    #[must_use]
    pub fn key(&self) -> u64 {
        crate::fnv1a64(format!("{self:?}").as_bytes())
    }
}

/// An input's values: a tile's `m x i` row-major, a column vector's `i`, a
/// row vector's `m`, or the one value.
#[derive(Clone, Debug, PartialEq)]
pub enum Given {
    Tile(Vec<f32>),
    Column(Vec<f32>),
    Row(Vec<f32>),
    One(f32),
}

/// Why the interpreter does not run an anchored region.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum EvalError {
    Anchor(AnchorError),
    /// The operand or the weight is not `m x k` and `parts · n x k`.
    Operands,
    /// An input's values are not of its kind or its tile's extent.
    Given {
        space: Space,
        input: u32,
    },
    Interp(InterpError),
    /// A rounding to a dtype the interpreter does not round to.
    Rounding(Dtype),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::Anchor(why) => write!(f, "{why}"),
            EvalError::Operands => f.write_str("the operands are not of the contraction's extent"),
            EvalError::Given { space, input } => {
                write!(f, "the {space} region's input {input} is not of its kind")
            }
            EvalError::Interp(why) => write!(f, "{why}"),
            EvalError::Rounding(dtype) => {
                write!(f, "the interpreter rounds no product to {dtype:?}")
            }
        }
    }
}

impl std::error::Error for EvalError {}

/// The region's inputs at tile row `row`, the tile `inner` wide.
fn row_args(
    space: Space,
    inputs: &[Input],
    given: &[Given],
    rows: usize,
    inner: usize,
    row: usize,
    norm: Option<f32>,
) -> Result<Vec<Arg>, EvalError> {
    let stated = inputs.iter().filter(|i| **i != Input::Norm).count();
    if given.len() != stated {
        let input = u32::try_from(given.len().min(stated)).unwrap_or(u32::MAX);
        return Err(EvalError::Given { space, input });
    }
    let mut given = given.iter();
    let mut args = Vec::with_capacity(inputs.len());
    for (input, kind) in (0u32..).zip(inputs) {
        let wrong = EvalError::Given { space, input };
        if *kind == Input::Norm {
            args.push(Arg::Scalar(norm.ok_or(wrong)?));
            continue;
        }
        let values = given.next().ok_or_else(|| wrong.clone())?;
        args.push(match (kind, values) {
            (Input::Tile, Given::Tile(v)) if v.len() == rows * inner => {
                Arg::Row(v[row * inner..(row + 1) * inner].to_vec())
            }
            (Input::Column, Given::Column(v)) if v.len() == inner => Arg::Row(v.clone()),
            (Input::Head(size), Given::Column(v)) if v.len() == *size as usize => {
                Arg::Row((0..inner).map(|c| v[c % v.len()]).collect())
            }
            (Input::Row, Given::Row(v)) if v.len() == rows => Arg::Scalar(v[row]),
            (Input::One, Given::One(x)) => Arg::Scalar(*x),
            _ => return Err(wrong),
        });
    }
    Ok(args)
}

/// The bank a contraction under `norm` reads (`parts · n x k`, row-major):
/// each weight `w[r][k]` times the gain's `g[k]` (or `1 + g[k]`), computed
/// in f32 and rounded once to bf16, the bank's dtype. The checkpoint's bank
/// is `w`, the norm's gain `g` (`k` of them, none where the norm has none).
#[must_use]
pub fn fold_bank(norm: &RowNorm, w: &[f32], g: &[f32]) -> Vec<f32> {
    let k = g.len();
    w.iter()
        .enumerate()
        .map(|(i, &x)| {
            let gain = match norm.gain {
                Gain::None => return x,
                Gain::Scale => g[i % k],
                Gain::PlusOne => 1.0 + g[i % k],
            };
            round_bf16(x * gain)
        })
        .collect()
}

/// Each row's norm scale under `norm`: its squares (f32) summed in f64 and
/// rounded to f32 once, as the interpreter sums a reduction, over `k`, the
/// `eps` added, its `rsqrt`.
#[must_use]
pub fn norm_scales(norm: &RowNorm, a: &[f32], k: usize) -> Vec<f32> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    a.chunks(k)
        .map(|row| {
            let sum = row.iter().fold(0f64, |s, x| s + f64::from(x * x)) as f32;
            1.0 / (sum / k as f32 + norm.eps).sqrt()
        })
        .collect()
}

/// `anchored` over `m` rows: `a` the operand (`m x k`), `w` the weight
/// (`parts · n x k`, as the contraction reads it: under a norm, the bank
/// [`fold_bank`] lands), each row-major, and each region's inputs (a head's
/// vector given as a column's, its `size` entries; a row's norm scale the
/// contraction's own, given none). Its results come back
/// each `m x n` (or `m` rows of its window), row-major, as the declared
/// roundings land them.
pub fn eval(
    anchored: &Anchored,
    m: usize,
    a: &[f32],
    w: &[f32],
    prologue: &[Given],
    epilogue: &[Given],
) -> Result<Vec<Vec<f32>>, EvalError> {
    anchored.verify().map_err(EvalError::Anchor)?;
    let c = &anchored.contraction;
    let (n, k, parts) = (c.n as usize, c.k as usize, c.layout.parts() as usize);
    if a.len() != m * k || w.len() != parts * n * k {
        return Err(EvalError::Operands);
    }
    // The operand the contraction reads, through the prologue.
    let mut operand = a.to_vec();
    if let Some(pro) = &anchored.prologue {
        for row in 0..m {
            let mut args = vec![Arg::Row(a[row * k..(row + 1) * k].to_vec())];
            args.extend(row_args(Space::Operand, &pro.inputs, prologue, m, k, row, None)?);
            let fed = interp::eval(&pro.region, &args).map_err(EvalError::Interp)?;
            operand[row * k..(row + 1) * k].copy_from_slice(&fed[0]);
        }
    }
    let round = |x: f32| match c.acc {
        None => Ok(x),
        Some(Dtype::Bf16) => Ok(round_bf16(x)),
        Some(other) => Err(EvalError::Rounding(other)),
    };
    // Each part's products, `m x n`, summed over `k` in f64.
    let mut products = vec![vec![0f32; m * n]; parts];
    for (p, plane) in products.iter_mut().enumerate() {
        for row in 0..m {
            let x = &operand[row * k..(row + 1) * k];
            for col in 0..n {
                let wr = &w[(p * n + col) * k..(p * n + col + 1) * k];
                let dot = x
                    .iter()
                    .zip(wr)
                    .fold(0f64, |acc, (a, b)| acc + f64::from(*a) * f64::from(*b));
                #[allow(clippy::cast_possible_truncation)]
                let dot = dot as f32;
                plane[row * n + col] = round(dot)?;
            }
        }
    }
    let Some(epi) = &anchored.epilogue else {
        return Ok(windowed(anchored, products, m, n));
    };
    let scales = anchored.norm.map(|norm| norm_scales(&norm, a, k));
    let mut out = vec![vec![0f32; m * n]; epi.region.results.len()];
    for row in 0..m {
        let mut args: Vec<Arg> = products
            .iter()
            .map(|plane| Arg::Row(plane[row * n..(row + 1) * n].to_vec()))
            .collect();
        let norm = scales.as_ref().map(|s| s[row]);
        args.extend(row_args(
            Space::Product,
            &epi.inputs,
            epilogue,
            m,
            n,
            row,
            norm,
        )?);
        let landed = interp::eval(&epi.region, &args).map_err(EvalError::Interp)?;
        for (plane, values) in out.iter_mut().zip(landed) {
            plane[row * n..(row + 1) * n].copy_from_slice(&values);
        }
    }
    Ok(cached(anchored, windowed(anchored, out, m, n)))
}

/// Each paged result's values as its cache stores them ([`CacheDtype::stored`]).
fn cached(anchored: &Anchored, mut tiles: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    for (tile, paged) in tiles.iter_mut().zip(&anchored.paged) {
        if let Some(paged) = paged {
            for x in tile.iter_mut() {
                *x = paged.dtype.stored(*x);
            }
        }
    }
    tiles
}

/// The cache row each of `m` rows of paged result `k` lands at,
/// `page · page_size + slot` (`None` where its slot lies past its page), as
/// the epilogue's given inputs name them; `None` where `k` is not paged.
#[must_use]
pub fn paged_rows(
    anchored: &Anchored,
    k: usize,
    epilogue: &[Given],
    page_size: u32,
) -> Option<Vec<Option<u64>>> {
    let paged = (*anchored.paged.get(k)?)?;
    let inputs = &anchored.epilogue.as_ref()?.inputs;
    // the given values, one an input but a row's norm scale (none)
    let given = |at: u32| {
        let skipped = inputs[..at as usize]
            .iter()
            .filter(|i| **i == Input::Norm)
            .count();
        match epilogue.get(at as usize - skipped)? {
            Given::Row(v) => Some(v),
            _ => None,
        }
    };
    let (pages, slots) = (given(paged.page)?, given(paged.slot)?);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(
        pages
            .iter()
            .zip(slots)
            .map(|(&page, &slot)| {
                let (page, slot) = (page as i64, slot as i64);
                (slot >= 0 && slot < i64::from(page_size) && page >= 0)
                    .then(|| (page * i64::from(page_size) + slot) as u64)
            })
            .collect(),
    )
}

/// Each result's window of its `m x n` tile.
fn windowed(anchored: &Anchored, tiles: Vec<Vec<f32>>, m: usize, n: usize) -> Vec<Vec<f32>> {
    if anchored.windows.is_empty() {
        return tiles;
    }
    tiles
        .into_iter()
        .zip(&anchored.windows)
        .map(|(tile, w)| {
            let tile = &tile;
            (0..m)
                .flat_map(|row| {
                    (0..w.width).map(move |j| tile[row * n + w.column(j) as usize])
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::{Binary, Elem, Literal, Reduce, Unary};

    fn contraction(n: u32, k: u32, layout: Layout) -> Contraction {
        Contraction {
            n,
            k,
            layout,
            quant: Quant::Plain(Dtype::Bf16),
            acc: Some(Dtype::Bf16),
        }
    }

    fn values(count: usize, seed: usize) -> Vec<f32> {
        (0..count)
            .map(|i| round_bf16(((i * 37 + seed * 11) % 23) as f32 / 8.0 - 1.4))
            .collect()
    }

    /// The host's own product, rounded to bf16 as the contraction declares.
    fn products(a: &[f32], w: &[f32], [m, n, k]: [usize; 3]) -> Vec<f32> {
        (0..m * n)
            .map(|i| {
                let (row, col) = (i / n, i % n);
                let dot = (0..k).fold(0f64, |s, j| {
                    s + f64::from(a[row * k + j]) * f64::from(w[col * k + j])
                });
                round_bf16(dot as f32)
            })
            .collect()
    }

    /// `softcap(y + bias) * s[m] + r`: a bias a column, a soft cap
    /// (`cap · tanh(x / cap)`, as `cap · (2 / (1 + e^(-2x/cap)) - 1)`), a
    /// row's scale and a residual tile, each rounded to bf16 as its op
    /// would store it.
    fn legal(n: u32) -> Pointwise {
        let cap = 30.0f32;
        Pointwise {
            region: Region {
                width: n,
                // y, bias, s, r
                params: vec![Shape::Row, Shape::Row, Shape::Scalar, Shape::Row],
                ops: vec![
                    Op::Binary(Binary::Add, 0, 1),       // 4: y + bias
                    Op::Const(Literal::F32(-2.0 / cap)), // 5
                    Op::Binary(Binary::Mul, 4, 5),       // 6
                    Op::Unary(Unary::Exp, 6),            // 7
                    Op::Const(Literal::F32(1.0)),        // 8
                    Op::Binary(Binary::Add, 8, 7),       // 9
                    Op::Const(Literal::F32(2.0)),        // 10
                    Op::Binary(Binary::Div, 10, 9),      // 11
                    Op::Binary(Binary::Sub, 11, 8),      // 12: tanh
                    Op::Const(Literal::F32(cap)),        // 13
                    Op::Binary(Binary::Mul, 13, 12),     // 14: softcap
                    Op::Binary(Binary::Mul, 14, 2),      // 15: · s[m]
                    Op::Binary(Binary::Add, 15, 3),      // 16: + r
                ],
                results: vec![16],
                rounds: vec![(4, Dtype::Bf16), (14, Dtype::Bf16), (16, Dtype::Bf16)],
            },
            inputs: vec![Input::Column, Input::Row, Input::Tile],
        }
    }

    #[test]
    fn a_legal_epilogue_is_accepted_and_lands_its_ops() {
        let [m, n, k] = [3usize, 16, 24];
        let anchored = Anchored {
            epilogue: Some(legal(n as u32)),
            ..Anchored::plain(contraction(n as u32, k as u32, Layout::Dense))
        };
        assert_eq!(anchored.verify(), Ok(()));
        let (a, w) = (values(m * k, 1), values(n * k, 2));
        let (bias, scale, residual) = (values(n, 3), values(m, 4), values(m * n, 5));
        let got = eval(
            &anchored,
            m,
            &a,
            &w,
            &[],
            &[
                Given::Column(bias.clone()),
                Given::Row(scale.clone()),
                Given::Tile(residual.clone()),
            ],
        )
        .expect("the anchored region runs");
        let y = products(&a, &w, [m, n, k]);
        let want: Vec<f32> = (0..m * n)
            .map(|i| {
                let x = round_bf16(y[i] + bias[i % n]);
                let tanh = 2.0 / (1.0 + (x * (-2.0 / 30.0)).exp()) - 1.0;
                let capped = round_bf16(30.0 * tanh);
                round_bf16(capped * scale[i / n] + residual[i])
            })
            .collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got[0]), bits(&want));
    }

    #[test]
    fn the_plain_store_lands_each_parts_products() {
        let [m, n, k] = [2usize, 8, 16];
        let anchored = Anchored::plain(contraction(n as u32, k as u32, Layout::Packed(2)));
        assert_eq!(anchored.verify(), Ok(()));
        assert_eq!(anchored.results(), 2);
        let (a, w) = (values(m * k, 6), values(2 * n * k, 7));
        let got = eval(&anchored, m, &a, &w, &[], &[]).expect("runs");
        assert_eq!(got[0], products(&a, &w[..n * k], [m, n, k]));
        assert_eq!(got[1], products(&a, &w[n * k..], [m, n, k]));
    }

    /// `gelu_tanh(gate) · up` over a packed bank's two products: pointwise
    /// in `(m, n)`, the two parts one value each there.
    #[test]
    fn a_glu_over_a_packed_bank_is_pointwise() {
        let [m, n, k] = [2usize, 8, 16];
        let region = Region {
            width: n as u32,
            params: vec![Shape::Row, Shape::Row],
            ops: vec![
                Op::Unary(Unary::Neg, 0),      // 2
                Op::Unary(Unary::Exp, 2),      // 3
                Op::Const(Literal::F32(1.0)),  // 4
                Op::Binary(Binary::Add, 4, 3), // 5
                Op::Binary(Binary::Div, 0, 5), // 6: silu(gate)
                Op::Binary(Binary::Mul, 6, 1), // 7
            ],
            results: vec![7],
            rounds: vec![(7, Dtype::Bf16)],
        };
        let anchored = Anchored {
            epilogue: Some(Pointwise {
                region,
                inputs: vec![],
            }),
            ..Anchored::plain(contraction(n as u32, k as u32, Layout::Packed(2)))
        };
        assert_eq!(anchored.verify(), Ok(()));
        let (a, w) = (values(m * k, 8), values(2 * n * k, 9));
        let got = eval(&anchored, m, &a, &w, &[], &[]).expect("runs");
        let (g, u) = (
            products(&a, &w[..n * k], [m, n, k]),
            products(&a, &w[n * k..], [m, n, k]),
        );
        let want: Vec<f32> = (0..m * n)
            .map(|i| round_bf16(g[i] / (1.0 + (-g[i]).exp()) * u[i]))
            .collect();
        assert_eq!(got[0], want);
    }

    #[test]
    fn an_epilogue_that_sums_across_n_is_rejected() {
        // y / Σₙ y: a softmax's denominator, a function of the whole row.
        let anchored = |n: u32| Anchored {
            epilogue: Some(Pointwise {
                region: Region {
                    width: n,
                    params: vec![Shape::Row],
                    ops: vec![
                        Op::Reduce(Reduce::Sum, 0),    // 1
                        Op::Binary(Binary::Div, 0, 1), // 2
                    ],
                    results: vec![2],
                    rounds: vec![],
                },
                inputs: vec![],
            }),
            ..Anchored::plain(contraction(n, 32, Layout::Dense))
        };
        // A row wider than a span is every tile's.
        let wide = anchored(1024);
        assert_eq!(
            wide.verify(),
            Err(AnchorError::Across {
                space: Space::Product,
                op: 0
            })
        );
        let why = eval(&wide, 1, &[0.0; 32], &[0.0; 1024 * 32], &[], &[]);
        assert!(matches!(
            why,
            Err(EvalError::Anchor(AnchorError::Across { .. }))
        ));
        // One a span wide (a row one head wide) is its own group.
        let narrow = anchored(16);
        assert_eq!(narrow.verify(), Ok(()));
        assert_eq!(narrow.epilogue.as_ref().unwrap().span(16), 16);
    }

    #[test]
    fn a_prologue_scales_the_operand_before_the_contraction() {
        let [m, n, k] = [2usize, 8, 16];
        // a · s[k] · t[m]: a column's scale and a row's.
        let prologue = Pointwise {
            region: Region {
                width: k as u32,
                params: vec![Shape::Row, Shape::Row, Shape::Scalar],
                ops: vec![
                    Op::Binary(Binary::Mul, 0, 1), // 3
                    Op::Binary(Binary::Mul, 3, 2), // 4
                ],
                results: vec![4],
                rounds: vec![(4, Dtype::Bf16)],
            },
            inputs: vec![Input::Column, Input::Row],
        };
        let anchored = Anchored {
            prologue: Some(prologue),
            ..Anchored::plain(contraction(n as u32, k as u32, Layout::Dense))
        };
        assert_eq!(anchored.verify(), Ok(()));
        let (a, w, s, t) = (
            values(m * k, 1),
            values(n * k, 2),
            values(k, 3),
            values(m, 4),
        );
        let got = eval(
            &anchored,
            m,
            &a,
            &w,
            &[Given::Column(s.clone()), Given::Row(t.clone())],
            &[],
        )
        .expect("runs");
        let fed: Vec<f32> = (0..m * k)
            .map(|i| round_bf16(a[i] * s[i % k] * t[i / k]))
            .collect();
        assert_eq!(got[0], products(&fed, &w, [m, n, k]));
    }

    #[test]
    fn a_region_off_its_tile_is_rejected_by_name() {
        let c = contraction(16, 32, Layout::Dense);
        let with = |pointwise: Pointwise| Anchored {
            epilogue: Some(pointwise),
            ..Anchored::plain(c)
        };
        let mut wide = legal(32);
        assert!(matches!(
            with(wide.clone()).verify(),
            Err(AnchorError::Width { wants: 16, .. })
        ));
        wide = legal(16);
        wide.inputs.pop();
        assert!(matches!(
            with(wide).verify(),
            Err(AnchorError::Params { wants: 3, .. })
        ));
        // A row's scale read as a column's vector.
        let mut misread = legal(16);
        misread.inputs[1] = Input::Column;
        assert_eq!(
            with(misread).verify(),
            Err(AnchorError::Param {
                space: Space::Product,
                param: 2
            })
        );
        // A result one a row lands no tile.
        let mut scalar = legal(16);
        scalar.region.results = vec![2];
        assert_eq!(
            with(scalar).verify(),
            Err(AnchorError::Result {
                space: Space::Product,
                result: 0
            })
        );
        let packed_of_one = Anchored::plain(contraction(16, 32, Layout::Packed(1)));
        assert!(matches!(
            packed_of_one.verify(),
            Err(AnchorError::Contraction(_))
        ));
    }

    /// Each head of a packed qkv product normed by its own mean square, its
    /// q and k heads turned at the row's position (NeoX halves) and its v
    /// heads kept: an epilogue whose span is one head.
    fn head_norm_rope(heads: u32, hd: u32, turned: u32) -> Pointwise {
        let mut b = crate::build::Ops::new(2);
        let normed = b.rms_normed_heads(0, hd, 1e-6);
        let turn = crate::build::Turn {
            head_dim: hd,
            pairs: crate::build::Pairs::Halves {
                rotary: hd,
                span: hd / 2,
            },
            theta: 10000.0,
            inverse: false,
            ramp: None,
        mscale: None,
        };
        let rotated = b.rotate(normed, 1, &turn);
        let col = b.push(Op::Column);
        let edge = b.u32(turned * hd);
        let qk = b.push(Op::Binary(Binary::Lt, col, edge));
        let out = b.push(Op::Select(qk, rotated, 0));
        Pointwise {
            region: Region {
                width: heads * hd,
                params: vec![Shape::Row, Shape::Scalar],
                ops: b.ops,
                results: vec![out],
                rounds: vec![(normed, Dtype::Bf16), (out, Dtype::Bf16)],
            },
            inputs: vec![Input::Row],
        }
    }

    #[test]
    fn a_heads_norm_and_turn_is_an_epilogue_of_one_heads_span() {
        let [m, heads, hd, k] = [3usize, 4, 64, 32];
        let n = heads * hd;
        let epi = head_norm_rope(heads as u32, hd as u32, 3);
        let anchored = Anchored {
            epilogue: Some(epi.clone()),
            ..Anchored::plain(contraction(n as u32, k as u32, Layout::Dense))
        };
        assert_eq!(anchored.verify(), Ok(()));
        assert_eq!(epi.span(n as u32), hd as u32);
        let (a, w, pos) = (values(m * k, 1), values(n * k, 2), vec![0.0, 5.0, 37.0]);
        let got = eval(&anchored, m, &a, &w, &[], &[Given::Row(pos.clone())]).expect("runs");
        // The region over each product row, as the row core runs it.
        let y = products(&a, &w, [m, n, k]);
        for row in 0..m {
            let want = interp::eval(
                &epi.region,
                &[
                    Arg::Row(y[row * n..(row + 1) * n].to_vec()),
                    Arg::Scalar(pos[row]),
                ],
            )
            .expect("the region runs");
            assert_eq!(got[0][row * n..(row + 1) * n], want[0][..]);
        }
        // The v heads land the products as they are.
        assert_eq!(got[0][3 * hd], y[3 * hd]);
        // A pointwise epilogue's span is one column.
        assert_eq!(legal(16).span(16), 1);
    }

    #[test]
    fn a_head_or_a_gather_past_one_tiles_span_is_rejected() {
        let c = |n: u32| contraction(n, 32, Layout::Dense);
        // A head 1024 wide: past MAX_SPAN.
        let wide = head_norm_rope(2, 1024, 2);
        let anchored = Anchored {
            epilogue: Some(wide),
            ..Anchored::plain(c(2048))
        };
        assert!(matches!(
            anchored.verify(),
            Err(AnchorError::Span {
                space: Space::Product,
                ..
            })
        ));
        // Each column's mate half the row away, over a row of 2048.
        let n = 2048;
        let mut b = crate::build::Ops::new(1);
        let col = b.push(Op::Column);
        let half = b.u32(n / 2);
        let up = b.push(Op::Binary(Binary::Add, col, half));
        let whole = b.u32(n);
        let mate = b.push(Op::Binary(Binary::Rem, up, whole));
        let out = b.push(Op::Peer(0, mate));
        let far = Pointwise {
            region: Region {
                width: n,
                params: vec![Shape::Row],
                ops: b.ops,
                results: vec![out],
                rounds: vec![],
            },
            inputs: vec![],
        };
        let anchored = Anchored {
            epilogue: Some(far.clone()),
            ..Anchored::plain(c(n))
        };
        assert_eq!(
            anchored.verify(),
            Err(AnchorError::Span {
                space: Space::Product,
                op: 5
            })
        );
        // Over a row of 256 the same gather is one tile's: one group.
        let mut near = far;
        near.region.width = 256;
        near.region.ops[1] = Op::Const(Literal::U32(128));
        near.region.ops[3] = Op::Const(Literal::U32(256));
        assert_eq!(near.span(256), 256);
        // A gather at a column a value picks crosses tiles where it likes.
        let picked = Pointwise {
            region: Region {
                width: 64,
                params: vec![Shape::Row, Shape::Row],
                ops: vec![
                    Op::Cast(1, Elem::U32), // 2
                    Op::Peer(0, 2),         // 3
                ],
                results: vec![3],
                rounds: vec![],
            },
            inputs: vec![Input::Tile],
        };
        let anchored = Anchored {
            epilogue: Some(picked),
            ..Anchored::plain(c(64))
        };
        assert_eq!(
            anchored.verify(),
            Err(AnchorError::Span {
                space: Space::Product,
                op: 1
            })
        );
        // A prologue gathers nothing: its row is the contraction's.
        let mut gathered = head_norm_rope(1, 32, 1);
        gathered.inputs = vec![Input::Row];
        let anchored = Anchored {
            prologue: Some(gathered),
            ..Anchored::plain(c(64))
        };
        assert!(matches!(
            anchored.verify(),
            Err(AnchorError::Across {
                space: Space::Operand,
                ..
            })
        ));
    }

    /// A packed qkv product: q's and k's heads normed by their own weights
    /// (one head wide, every head reading it) and turned, v as it is, each
    /// landing its own window.
    #[test]
    fn a_packed_qkv_lands_each_piece_in_its_window() {
        let [m, hd, k] = [2usize, 32, 16];
        let (q, kv) = (2 * hd, hd);
        let n = q + 2 * kv;
        let mut b = crate::build::Ops::new(4);
        let normed = b.rms_normed_heads(0, hd as u32, 1e-6);
        let col = b.push(Op::Column);
        let edge = b.u32(q as u32);
        let in_q = b.push(Op::Binary(Binary::Lt, col, edge));
        let w = b.push(Op::Select(in_q, 1, 2));
        let scaled = b.push(Op::Binary(Binary::Mul, normed, w));
        let turn = crate::build::Turn {
            head_dim: hd as u32,
            pairs: crate::build::Pairs::Halves {
                rotary: hd as u32,
                span: hd as u32 / 2,
            },
            theta: 10000.0,
            inverse: false,
            ramp: None,
        mscale: None,
        };
        let turned = b.rotate(scaled, 3, &turn);
        let anchored = Anchored {
            epilogue: Some(Pointwise {
                region: Region {
                    width: n as u32,
                    params: vec![Shape::Row, Shape::Row, Shape::Row, Shape::Scalar],
                    ops: b.ops,
                    results: vec![turned, turned, 0],
                    rounds: vec![(scaled, Dtype::Bf16)],
                },
                inputs: vec![Input::Head(hd as u32), Input::Head(hd as u32), Input::Row],
            }),
            windows: vec![
                Window::of(0, q as u32),
                Window::of(q as u32, kv as u32),
                Window::of((q + kv) as u32, kv as u32),
            ],
            ..Anchored::plain(Contraction {
                acc: None,
                ..contraction(n as u32, k as u32, Layout::Dense)
            })
        };
        assert_eq!(anchored.verify(), Ok(()));
        let (a, wt) = (values(m * k, 3), values(n * k, 4));
        let (qw, kw, pos) = (values(hd, 5), values(hd, 6), vec![3.0, 11.0]);
        let given = [
            Given::Column(qw.clone()),
            Given::Column(kw.clone()),
            Given::Row(pos.clone()),
        ];
        let got = eval(&anchored, m, &a, &wt, &[], &given).expect("runs");
        let whole = Anchored {
            windows: Vec::new(),
            ..anchored.clone()
        };
        let tiles = eval(&whole, m, &a, &wt, &[], &given).expect("runs");
        for (r, (from, width)) in [(0, q), (q, kv), (q + kv, kv)].into_iter().enumerate() {
            assert_eq!(got[r].len(), m * width);
            for row in 0..m {
                assert_eq!(
                    got[r][row * width..(row + 1) * width],
                    tiles[r][row * n + from..row * n + from + width]
                );
            }
        }
        // v lands its f32 products as they are; its plane rounds them.
        let mut plain = whole.clone();
        plain.epilogue = None;
        let y = eval(&plain, m, &a, &wt, &[], &[]).expect("runs");
        assert_eq!(got[2][..kv], y[0][q + kv..q + 2 * kv]);
        // A head's weight past the tile's heads, and windows off the tile.
        let mut off = anchored.clone();
        off.epilogue.as_mut().unwrap().inputs[0] = Input::Head(48);
        assert!(matches!(off.verify(), Err(AnchorError::Param { .. })));
        let mut off = anchored.clone();
        off.windows[2].width += 1;
        assert_eq!(off.verify(), Err(AnchorError::Window { result: 2 }));
        let mut off = anchored;
        off.windows.pop();
        assert!(matches!(off.verify(), Err(AnchorError::Window { .. })));
    }

    /// `rmsnorm(a, g) · wᵀ` as the rewrite states it: the bank the gain
    /// folded into, the products scaled by each row's norm in the epilogue.
    /// It lands what the unfolded program does (the normed row rounded to
    /// bf16, then the product) within each side's rounding of its terms: a
    /// bf16 place of their magnitudes' sum, and the result's own ulp.
    #[test]
    fn a_folded_norm_lands_the_unfolded_program_within_its_rounding() {
        let [m, n, k] = [3usize, 16, 64];
        for gain in [Gain::None, Gain::Scale, Gain::PlusOne] {
            let norm = RowNorm { eps: 1e-6, gain };
            let anchored = Anchored {
                norm: Some(norm),
                epilogue: Some(Pointwise {
                    region: Region {
                        width: n as u32,
                        params: vec![Shape::Row, Shape::Scalar],
                        ops: vec![Op::Binary(Binary::Mul, 0, 1)],
                        results: vec![2],
                        rounds: vec![(2, Dtype::Bf16)],
                    },
                    inputs: vec![Input::Norm],
                }),
                ..Anchored::plain(Contraction {
                    acc: None,
                    ..contraction(n as u32, k as u32, Layout::Dense)
                })
            };
            assert_eq!(anchored.verify(), Ok(()));
            let (a, w, g) = (values(m * k, 1), values(n * k, 2), values(k, 3));
            let bank = fold_bank(&norm, &w, &g);
            let got = eval(&anchored, m, &a, &bank, &[], &[]).expect("runs");
            let r = norm_scales(&norm, &a, k);
            let normed: Vec<f32> = (0..m * k)
                .map(|i| {
                    let s = match gain {
                        Gain::None => 1.0,
                        Gain::Scale => g[i % k],
                        Gain::PlusOne => 1.0 + g[i % k],
                    };
                    round_bf16(a[i] * r[i / k] * s)
                })
                .collect();
            let want = products(&normed, &w, [m, n, k]);
            for (i, (got, want)) in got[0].iter().zip(&want).enumerate() {
                // Each side rounds each term once (the normed row, or the
                // folded bank): off by a bf16 place of the sum of the terms'
                // magnitudes, then each result's own rounding.
                let (row, col) = (i / n, i % n);
                let magnitude: f32 = (0..k)
                    .map(|j| (normed[row * k + j] * w[col * k + j]).abs())
                    .sum();
                let ulp = 2f32.powi(want.abs().max(0.5).log2().floor() as i32 - 7);
                let bound = ulp + magnitude * 2f32.powi(-8);
                assert!((got - want).abs() <= bound, "{gain:?}: {got} vs {want}");
            }
        }
        // A scale read with no norm, a norm no epilogue reads, a norm beside
        // a prologue: each refused.
        let c = contraction(16, 64, Layout::Dense);
        let reads = Pointwise {
            region: Region {
                width: 16,
                params: vec![Shape::Row, Shape::Scalar],
                ops: vec![Op::Binary(Binary::Mul, 0, 1)],
                results: vec![2],
                rounds: vec![],
            },
            inputs: vec![Input::Norm],
        };
        let unread = Anchored {
            norm: Some(RowNorm {
                eps: 1e-6,
                gain: Gain::Scale,
            }),
            ..Anchored::plain(c)
        };
        assert_eq!(unread.verify(), Err(AnchorError::Norm));
        let undeclared = Anchored {
            epilogue: Some(reads),
            ..Anchored::plain(c)
        };
        assert_eq!(undeclared.verify(), Err(AnchorError::Norm));
    }

    /// A region rewritten over normed rows lands what the region does over
    /// the rows normed: each part's product scaled first.
    #[test]
    fn a_regions_normed_form_scales_each_parts_product() {
        let [m, n, k] = [2usize, 8, 32];
        let glu = Anchored {
            epilogue: Some(Pointwise {
                region: Region {
                    width: n as u32,
                    params: vec![Shape::Row, Shape::Row],
                    ops: vec![Op::Binary(Binary::Mul, 0, 1)],
                    results: vec![2],
                    rounds: vec![],
                },
                inputs: vec![],
            }),
            ..Anchored::plain(Contraction {
                acc: None,
                ..contraction(n as u32, k as u32, Layout::Packed(2))
            })
        };
        let norm = RowNorm {
            eps: 1e-5,
            gain: Gain::None,
        };
        let normed = glu.normed(norm).expect("a region of an epilogue");
        assert_eq!(normed.verify(), Ok(()));
        let (a, w) = (values(m * k, 1), values(2 * n * k, 2));
        let got = eval(&normed, m, &a, &w, &[], &[]).expect("runs");
        let r = norm_scales(&norm, &a, k);
        let scaled: Vec<f32> = (0..m * k).map(|i| a[i] * r[i / k]).collect();
        let want = eval(&glu, m, &scaled, &w, &[], &[]).expect("runs");
        for (got, want) in got[0].iter().zip(&want[0]) {
            assert!((got - want).abs() <= want.abs() * 1e-5 + 1e-6, "{got} vs {want}");
        }
        assert!(Anchored::plain(glu.contraction).normed(norm).is_none());
        assert!(normed.normed(norm).is_none());
    }

    /// e4m3's rounding as CUDA's `satfinite` conversion lands it: the
    /// nearest of 3 mantissa bits (ties to even), subnormals 2^-9 apart,
    /// ±448 at most.
    #[test]
    fn e4m3_rounds_to_the_nearest_and_saturates() {
        use crate::skeleton::round_e4m3;
        for (x, want) in [
            (0.0, 0.0),
            (1.0, 1.0),
            (1.0625, 1.0),   // a tie between 1 and 1.125: to even
            (1.1875, 1.25),  // a tie between 1.125 and 1.25: to even
            (1.2, 1.25),
            (-3.3, -3.25),
            (448.0, 448.0),
            (500.0, 448.0),
            (-1e9, -448.0),
            (0.001_953_125, 0.001_953_125), // 2^-9, the least subnormal
            (0.0009, 0.0),                  // below half of it
            (0.0015, 0.001_953_125),
            (0.017, 0.017_578_125),         // 9 · 2^-9
        ] {
            assert_eq!(round_e4m3(x), want, "{x}");
        }
        assert!(round_e4m3(f32::NAN).is_nan());
    }

    /// k and v of a packed qkv product landed in a cache's pages: each row at
    /// `page · page_size + slot`, stored as the cache's element; one whose
    /// slot lies past its page lands nothing. A paged result's page or slot
    /// must be an input one value a row.
    #[test]
    fn a_paged_result_lands_each_row_at_its_page_and_slot() {
        let [m, n, k] = [3usize, 16, 32];
        let region = Region {
            width: n as u32,
            params: vec![Shape::Row, Shape::Scalar, Shape::Scalar],
            ops: vec![],
            results: vec![0, 0],
            rounds: vec![],
        };
        let paged = Paged {
            plane: CachePlane::Values,
            dtype: CacheDtype::E4m3 { scale: 0.5 },
            page: 0,
            slot: 1,
        };
        let anchored = Anchored {
            epilogue: Some(Pointwise {
                region,
                inputs: vec![Input::Row, Input::Row],
            }),
            paged: vec![None, Some(paged)],
            ..Anchored::plain(Contraction {
                acc: None,
                ..contraction(n as u32, k as u32, Layout::Dense)
            })
        };
        assert_eq!(anchored.verify(), Ok(()));
        let (a, w) = (values(m * k, 1), values(n * k, 2));
        let given = [
            Given::Row(vec![2.0, 0.0, 5.0]),
            Given::Row(vec![3.0, 15.0, 16.0]),
        ];
        let got = eval(&anchored, m, &a, &w, &[], &given).expect("runs");
        for (plain, cached) in got[0].iter().zip(&got[1]) {
            assert_eq!(*cached, crate::skeleton::round_e4m3(round_bf16(*plain) / 0.5));
        }
        assert_eq!(
            paged_rows(&anchored, 1, &given, 16),
            Some(vec![Some(35), Some(15), None])
        );
        assert_eq!(paged_rows(&anchored, 0, &given, 16), None);
        // A slot read from a column's vector names no row's slot.
        let mut off = anchored.clone();
        off.epilogue.as_mut().unwrap().inputs[1] = Input::One;
        assert_eq!(off.verify(), Err(AnchorError::Paged { result: 1 }));
        let mut off = anchored;
        off.paged[1] = Some(Paged {
            dtype: CacheDtype::E4m3 { scale: 0.0 },
            ..paged
        });
        assert_eq!(off.verify(), Err(AnchorError::Paged { result: 1 }));
    }

    /// A head-major window: `head` columns of every `stride` (Qwen 3.5's q,
    /// a head interleaved with its gate's), its heads landed one after
    /// another.
    #[test]
    fn a_strided_window_lands_a_head_of_every_stride() {
        let [m, n, k] = [2usize, 16, 8];
        let region = Region {
            width: n as u32,
            params: vec![Shape::Row],
            ops: vec![],
            results: vec![0, 0],
            rounds: vec![],
        };
        let anchored = Anchored {
            epilogue: Some(Pointwise {
                region,
                inputs: vec![],
            }),
            windows: vec![Window::strided(0, 8, 4, 8), Window::strided(4, 8, 4, 8)],
            ..Anchored::plain(Contraction {
                acc: None,
                ..contraction(n as u32, k as u32, Layout::Dense)
            })
        };
        assert_eq!(anchored.verify(), Ok(()));
        let w = Window::strided(4, 8, 4, 8);
        assert_eq!((w.column(5), w.landed(13), w.landed(8), w.end()), (13, Some(5), None, 16));
        let (a, wt) = (values(m * k, 1), values(n * k, 2));
        let got = eval(&anchored, m, &a, &wt, &[], &[]).expect("runs");
        let y = eval(&Anchored::plain(anchored.contraction), m, &a, &wt, &[], &[]).expect("runs");
        for row in 0..m {
            for j in 0..8 {
                let (q, g) = (j / 4 * 8 + j % 4, j / 4 * 8 + 4 + j % 4);
                assert_eq!(got[0][row * 8 + j], y[0][row * n + q]);
                assert_eq!(got[1][row * 8 + j], y[0][row * n + g]);
            }
        }
        // A head past its stride, or a window past the tile's columns.
        let mut off = anchored.clone();
        off.windows[0] = Window::strided(0, 8, 8, 4);
        assert_eq!(off.verify(), Err(AnchorError::Window { result: 0 }));
        let mut off = anchored;
        off.windows[1] = Window::strided(12, 8, 4, 8);
        assert_eq!(off.verify(), Err(AnchorError::Window { result: 1 }));
    }

    /// A turn at frequencies stated outright (Llama 3's scaled rope): each
    /// pair's from the table, its cos and sin times the table's scale; and
    /// A turn of 64-wide heads computes in every head what it computes in
    /// the first: its slices of whole heads repeat, a half head's do not.
    #[test]
    fn a_heads_turn_repeats_every_head_and_no_narrower() {
        use crate::build::{Ops, Pairs, Turn};
        let mut b = Ops::new(2);
        let turn = Turn {
            head_dim: 64,
            pairs: Pairs::Halves {
                rotary: 64,
                span: 32,
            },
            theta: 10000.0,
            inverse: false,
            ramp: None,
            mscale: None,
        };
        let turned = b.rotate(0, 1, &turn);
        let turned = Pointwise {
            region: Region {
                width: 256,
                params: vec![Shape::Row, Shape::Scalar],
                ops: b.ops,
                results: vec![turned],
                rounds: vec![],
            },
            inputs: vec![Input::Row],
        };
        assert!(turned.repeats(256, 64));
        assert!(turned.repeats(256, 128));
        assert!(!turned.repeats(256, 32));
        // a column read as it is never repeats
        let column = Pointwise {
            region: Region {
                width: 256,
                params: vec![Shape::Row],
                ops: vec![
                    Op::Column,
                    Op::Cast(1, crate::region::Elem::F32),
                    Op::Binary(Binary::Add, 0, 2),
                ],
                results: vec![3],
                rounds: vec![],
            },
            inputs: vec![],
        };
        assert!(!column.repeats(256, 64));
        assert!(column.repeats(256, 256));
    }

    /// YaRN's every pair at its ramped frequency, times its attention factor.
    #[test]
    fn a_turn_reads_its_tables_frequencies_and_scale() {
        use crate::build::{Ops, Pairs, Turn};
        let hd = 8u32;
        let table = [1.0f32, 0.25, 0.0625, 0.015_625];
        let mut b = Ops::new(2);
        let turn = Turn {
            head_dim: hd,
            pairs: Pairs::Halves {
                rotary: hd,
                span: hd / 2,
            },
            theta: 10000.0,
            inverse: false,
            ramp: None,
            mscale: Some(0.5),
        };
        let out = b.rotate_with(0, 1, &turn, Some(&table));
        let region = Region {
            width: hd,
            params: vec![Shape::Row, Shape::Scalar],
            ops: b.ops,
            results: vec![out],
            rounds: vec![],
        };
        let x: Vec<f32> = (0..hd).map(|i| i as f32 * 0.25 - 1.0).collect();
        let pos = 5.0f32;
        let got = interp::eval(&region, &[Arg::Row(x.clone()), Arg::Scalar(pos)]).expect("runs");
        for i in 0..hd as usize {
            let d = i % 4;
            let angle = pos * table[d];
            let (c, s) = (angle.cos() * 0.5, angle.sin() * 0.5);
            let want = if i < 4 {
                x[i] * c - x[i + 4] * s
            } else {
                x[i] * c + x[i - 4] * s
            };
            assert!((got[0][i] - want).abs() <= 1e-6, "{i}: {} vs {want}", got[0][i]);
        }
    }

    #[test]
    fn an_anchored_regions_key_is_its_function() {
        let c = contraction(16, 32, Layout::Dense);
        let a = Anchored {
            epilogue: Some(legal(16)),
            ..Anchored::plain(c)
        };
        assert_eq!(a.key(), a.clone().key());
        assert_ne!(a.key(), Anchored::plain(c).key());
    }
}
