//! The contraction skeletons: pie's own decode GEMMs (`gemv`, the few-rows
//! kernel, `skinny`), each a hand-written parallel skeleton
//! (`linear/contract.cuh`) that hands its products, a few elements of one
//! row of the product tile at a time, to an epilogue body through the hooks
//! of `TileHook`.
//!
//! - **The epilogue** ([`Epilogue`]): an anchored region's (kernels-ir's
//!   [`Anchored`]) epilogue scheduled and printed by the row skeleton's printer
//!   ([`crate::synth::Body`]), whose hooks the tile hook provides, `f.ld(k,
//!   e)` reading the parts' products below the parts and the inputs above.
//!   No epilogue is the plain store: each product stored as it is. A
//!   pointwise region's body is the same at every width, so one epilogue
//!   serves every `n`; one that combines a head's columns (its span: a
//!   head's norm, a rotary pair), reads a column's index or lands its
//!   results in windows (a split's pieces) is of its own `n` alone, and
//!   only the skinny skeleton, whose tile holds whole heads, fires a span.
//!   An input is bf16, or i32 one element a row (a row's position).
//! - **The skeletons** ([`Skeleton`]): which shapes each covers, and its fire with
//!   an epilogue filled in.
//! - **Identity.** One generated unit a body, `linear/hooked/<hash>.cu`:
//!   the skeletons' header and the body's source; one kernel a skeleton, its
//!   shape class (warps, rows, splits) and the body's planes. The disk cache
//!   keys the unit's text, the headers and the instantiation, as it does a
//!   carried unit's.
//! - **Loading.** A kernel loads on its first fire, as the carried kernels
//!   these skeletons replace did: the plain store's body is fixed, and the
//!   tuner fires it before a capture. An anchored region a load forms
//!   loads with the model, every kernel a fire of it may take
//!   ([`Epilogue::load_every`]).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use dtype::Dtype;
pub use kernels_ir::anchor::{
    Anchored, Contraction, Gain, Input, Layout, Pointwise, Quant, RowNorm, Window,
};
use kernels_ir::{Binary, Literal, Op, Region, Shape};

use crate::error::Error;
use crate::jit::{Kernel, Root, refuse};
use crate::synth::Body;
use crate::tuning::Tuning;

const OP: &str = "linear.skeleton";

/// The struct every epilogue body defines, in the namespace its unit puts
/// it in.
const BODY: &str = "::pie::linear::hooked::Body";

/// The most inputs and results an epilogue's planes carry.
pub(crate) const MAX_INS: usize = 8;
pub(crate) const MAX_OUTS: usize = 4;

/// `pie::linear::EpiIn`, field for field: an input's element `(row, col)`
/// at `base[row * row_stride + (col % period) * col_stride]` (`period` 0:
/// the column as it is), bf16 (`dtype` 0) or i32 (1).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct In {
    base: u64,
    row_stride: i64,
    col_stride: i64,
    dtype: i64,
    period: i64,
}

impl In {
    const fn of(base: u64, row_stride: i64, col_stride: i64) -> In {
        In {
            base,
            row_stride,
            col_stride,
            dtype: 0,
            period: 0,
        }
    }

    /// A tile of the product's extent, its rows `ld` apart (a residual).
    #[must_use]
    pub const fn tile(base: u64, ld: u32) -> In {
        In::of(base, ld as i64, 1)
    }

    /// A vector one a column, every row reading it (a bias).
    #[must_use]
    pub const fn column(base: u64) -> In {
        In::of(base, 0, 1)
    }

    /// A vector of a head `size` columns wide, every head reading it (a
    /// head's norm weight).
    #[must_use]
    pub const fn head(base: u64, size: u32) -> In {
        In {
            period: size as i64,
            ..In::of(base, 0, 1)
        }
    }

    /// A vector one a row (a row's scale).
    #[must_use]
    pub const fn row(base: u64) -> In {
        In::of(base, 1, 0)
    }

    /// One value for the whole tile (a layer's scale).
    #[must_use]
    pub const fn one(base: u64) -> In {
        In::of(base, 0, 0)
    }

    /// The same input stored as i32 (a row's position).
    #[must_use]
    pub const fn i32(self) -> In {
        In { dtype: 1, ..self }
    }

    /// The operand row's norm scale: no plane, the contraction's own.
    #[must_use]
    pub const fn norm() -> In {
        In {
            dtype: 3,
            ..In::of(0, 0, 0)
        }
    }

    /// The norm scale read from `scales` (f32, one a row): the unfused
    /// leg's, landed by `norm_scales`.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) const fn at(self, scales: u64) -> In {
        In {
            base: scales,
            ..self
        }
    }

    /// Whether it is a row's norm scale.
    #[must_use]
    pub const fn is_norm(self) -> bool {
        self.dtype == 3
    }

    /// How its elements are stored.
    #[must_use]
    pub const fn dtype(self) -> Dtype {
        match self.dtype {
            1 => Dtype::I32,
            3 => Dtype::F32,
            _ => Dtype::Bf16,
        }
    }

    /// The same input read from tile row `row` on: a run of the tile's rows
    /// served apart.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) const fn from_row(self, row: usize) -> In {
        let size = if self.dtype == 1 { 4 } else { 2 };
        In {
            base: self.base + (row as i64 * self.row_stride * size) as u64,
            ..self
        }
    }

    /// Its first element's address.
    #[cfg(feature = "cuda")]
    pub(crate) const fn base(self) -> u64 {
        self.base
    }

    /// The input kind its strides read.
    const fn kind(self) -> Input {
        if self.dtype == 3 {
            return Input::Norm;
        }
        match (self.row_stride, self.col_stride) {
            (0, 0) => Input::One,
            (0, _) if self.period > 0 => Input::Head(self.period as u32),
            (0, _) => Input::Column,
            (_, 0) => Input::Row,
            _ => Input::Tile,
        }
    }
}

/// An anchored region's epilogue printed for the skeletons' hook: the body,
/// what its planes are, and whether each product rounds to bf16 before the
/// body reads it.
#[derive(Clone, Debug)]
pub struct Epilogue {
    /// The epilogue's region (the plain store's where it has none).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    region: Region,
    body: Body,
    root: Root,
    parts: u32,
    inputs: Vec<Input>,
    results: usize,
    round: bool,
    /// The plain store: no epilogue.
    plain: bool,
    /// The cuBLASLt epilogue that lands what this one does, where one does.
    lt: Option<LtEpilogue>,
    /// The window of the tile each result lands, or none: every column.
    windows: Vec<Window>,
    /// The columns of each group the body combines within (a head), 1
    /// where it is pointwise ([`Pointwise::span`]).
    span: u32,
    /// The tile's columns, where the body is of them alone (it combines
    /// columns, reads a column's index or lands windows): a fire of any
    /// other `n` is no fire of it.
    fixed: Option<u32>,
    /// How each input is stored: bf16, or i32 (a row's position); a row's
    /// norm scale f32, where the unfused leg reads it from a plane.
    stored: Vec<Dtype>,
    /// The operand's rows normed: each row's scale the epilogue's
    /// `Input::Norm`, the skeleton's own.
    norm: Option<RowNorm>,
}

/// What cuBLASLt's own epilogues land over one dense bank's products: a
/// bias a column, a ReLU, or a bias then a ReLU, each summed in f32 and
/// stored once to bf16. Its GELU is none of these: its GELU after a bias
/// lands up to 6 bf16 ulps from the tanh GELU at a prefill's rows (a
/// 14336-column bank over 5120), past the 3 an epilogue of one product is
/// held to, and erf's has no cuBLASLt form. An anchored region whose
/// epilogue is one of these over its products as they are summed (its
/// contraction declares no rounding of them) lands exactly what cuBLASLt
/// does, and may run as a cuBLASLt product with it: one more fused
/// candidate the GEMM tuner times.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LtEpilogue {
    Bias,
    Relu,
    ReluBias,
}

impl LtEpilogue {
    const ALL: [LtEpilogue; 3] = [LtEpilogue::Bias, LtEpilogue::Relu, LtEpilogue::ReluBias];

    /// Whether it reads a bias, its one input.
    #[must_use]
    pub const fn biased(self) -> bool {
        matches!(self, LtEpilogue::Bias | LtEpilogue::ReluBias)
    }

    /// What it lands of the product `y` (and the bias `b`), in f32.
    fn of(self, y: f32, b: f32) -> f32 {
        let x = if self.biased() { y + b } else { y };
        match self {
            LtEpilogue::Bias => x,
            LtEpilogue::Relu | LtEpilogue::ReluBias => x.max(0.0),
        }
    }

    /// The one of these `region` lands over one f32 product (and a bias
    /// where `inputs` is a vector a column), probed by the interpreter over
    /// products off the bf16 grid: each result the bits cuBLASLt's sum and
    /// one store land.
    fn matching(region: &Region, inputs: &[Input]) -> Option<LtEpilogue> {
        let biased = match inputs {
            [] => false,
            [Input::Column] => true,
            _ => return None,
        };
        if region.results.len() != 1 || region.params.len() != 1 + usize::from(biased) {
            return None;
        }
        const WIDTH: usize = 64;
        let probe = Region {
            width: WIDTH as u32,
            ..region.clone()
        };
        let bf16 = kernels_ir::skeleton::round_bf16;
        #[allow(clippy::cast_precision_loss)]
        let y: Vec<f32> = (0..WIDTH).map(|i| (i as f32 - 31.5) * 0.190_137).collect();
        #[allow(clippy::cast_precision_loss)]
        let b: Vec<f32> = (0..WIDTH)
            .map(|i| bf16(((i * 37) % 11) as f32 * 0.21 - 1.0))
            .collect();
        let mut args = vec![kernels_ir::interp::Arg::Row(y.clone())];
        if biased {
            args.push(kernels_ir::interp::Arg::Row(b.clone()));
        }
        let landed = kernels_ir::interp::eval(&probe, &args).ok()?;
        let landed = landed.first()?;
        LtEpilogue::ALL
            .into_iter()
            .filter(|lt| lt.biased() == biased)
            .find(|lt| (0..WIDTH).all(|i| landed[i].to_bits() == bf16(lt.of(y[i], b[i])).to_bits()))
    }
}

impl PartialEq for Epilogue {
    fn eq(&self, other: &Epilogue) -> bool {
        self.body == other.body
            && self.parts == other.parts
            && self.inputs == other.inputs
            && self.results == other.results
            && self.round == other.round
            && self.windows == other.windows
            && self.fixed == other.fixed
            && self.stored == other.stored
            && self.norm == other.norm
    }
}

impl Eq for Epilogue {}

impl Epilogue {
    /// `anchored`'s epilogue printed for the skeletons' hook, or why no skeleton
    /// takes it: a skeleton reads bf16 operands as they are stored (a prologue
    /// runs as a region of its own before the product), rounds a product to
    /// bf16 or not at all, and lands bf16 planes; its inputs bf16.
    pub fn of(anchored: &Anchored) -> Result<Epilogue, Error> {
        Epilogue::of_stored(anchored, &vec![Dtype::Bf16; planes_of(anchored)])
    }

    /// [`Epilogue::of`] with its inputs' planes stored as `stored` says (one
    /// an input but a row's norm scale, which has none): bf16, or i32 where
    /// an input is one element a row or one value (a position).
    pub fn of_stored(anchored: &Anchored, stored: &[Dtype]) -> Result<Epilogue, Error> {
        anchored
            .verify()
            .map_err(|why| refuse(OP, format!("not an anchored region: {why}")))?;
        let c = &anchored.contraction;
        if anchored.prologue.is_some() {
            return Err(refuse(
                OP,
                "a prologue: the skeletons read the operand as it is stored, and one runs as its \
                 own region before the product",
            ));
        }
        if c.quant != Quant::Plain(Dtype::Bf16) {
            return Err(refuse(
                OP,
                format!(
                    "the operands are stored {:?}, and the skeletons read bf16",
                    c.quant
                ),
            ));
        }
        let round = match c.acc {
            None => false,
            Some(Dtype::Bf16) => true,
            Some(other) => {
                return Err(refuse(
                    OP,
                    format!("the products round to {other:?}, and the skeletons round to bf16"),
                ));
            }
        };
        let parts = c.layout.parts();
        let (region, inputs) = match &anchored.epilogue {
            Some(Pointwise { region, inputs }) => (region.clone(), inputs.clone()),
            None => (store(c.n, parts), Vec::new()),
        };
        if inputs.len() > MAX_INS || region.results.len() > MAX_OUTS {
            return Err(refuse(
                OP,
                format!(
                    "an epilogue of {} input(s) and {} result(s), past the skeletons' {MAX_INS} and \
                     {MAX_OUTS}",
                    inputs.len(),
                    region.results.len()
                ),
            ));
        }
        if region
            .rounds
            .iter()
            .any(|(v, dtype)| region.results.contains(v) && *dtype != Dtype::Bf16)
        {
            return Err(refuse(OP, "a result stored other than bf16"));
        }
        if stored.len() != planes_of(anchored) {
            return Err(refuse(OP, "the inputs' dtypes are not one a plane"));
        }
        // A row's norm scale is f32 where a plane holds it (the leg's).
        let mut planes = stored.iter();
        let stored: Vec<Dtype> = inputs
            .iter()
            .map(|input| match input {
                Input::Norm => Dtype::F32,
                _ => *planes.next().unwrap_or(&Dtype::Bf16),
            })
            .collect();
        for (input, dtype) in inputs.iter().zip(&stored) {
            let held = match dtype {
                Dtype::Bf16 => *input != Input::Norm,
                Dtype::I32 => matches!(input, Input::Row | Input::One),
                Dtype::F32 => *input == Input::Norm,
                _ => false,
            };
            if !held {
                return Err(refuse(
                    OP,
                    format!(
                        "an input {input:?} stored {dtype:?}: the skeletons read bf16, and i32 one \
                         element a row or one value"
                    ),
                ));
            }
        }
        let body = Body::of(&region).map_err(|why| refuse(OP, why.to_string()))?;
        let plain = anchored.epilogue.is_none();
        let root = unit(&body)?;
        let windows = anchored.windows.clone();
        if windows.iter().any(|w| w.from % 8 != 0 || w.width % 8 != 0) {
            return Err(refuse(
                OP,
                "a result's window is no whole run of eight columns, and the skeletons store \
                 whole runs",
            ));
        }
        let span = anchored.epilogue.as_ref().map_or(1, |e| e.span(c.n));
        let columned = region.ops.iter().any(|op| matches!(op, Op::Column));
        let fixed = (span > 1 || columned || !windows.is_empty()).then_some(c.n);
        let lt = if parts == 1 && !round && !plain && fixed.is_none() && anchored.norm.is_none() {
            LtEpilogue::matching(&region, &inputs)
        } else {
            None
        };
        Ok(Epilogue {
            lt,
            results: region.results.len(),
            region,
            body,
            root,
            parts,
            inputs,
            round,
            plain,
            windows,
            span,
            fixed,
            stored,
            norm: anchored.norm,
        })
    }

    /// `anchored`'s epilogue ([`Epilogue::of`]), one for the life of the
    /// process: the same reference for every call with the same region.
    pub fn interned(anchored: &Anchored) -> Result<&'static Epilogue, Error> {
        Epilogue::interned_stored(anchored, &vec![Dtype::Bf16; planes_of(anchored)])
    }

    /// [`Epilogue::of_stored`], one for the life of the process.
    pub fn interned_stored(
        anchored: &Anchored,
        stored: &[Dtype],
    ) -> Result<&'static Epilogue, Error> {
        static EPILOGUES: OnceLock<Mutex<HashMap<u64, &'static Epilogue>>> = OnceLock::new();
        let key =
            anchored.key() ^ kernels_ir::fnv1a64(format!("{stored:?}").as_bytes()).rotate_left(29);
        let table = EPILOGUES.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(found) = table
            .lock()
            .expect("epilogue table lock poisoned")
            .get(&key)
        {
            return Ok(found);
        }
        let made: &'static Epilogue = Box::leak(Box::new(Epilogue::of_stored(anchored, stored)?));
        Ok(*table
            .lock()
            .expect("epilogue table lock poisoned")
            .entry(key)
            .or_insert(made))
    }

    /// The plain store over one dense bank: each product rounded to bf16, as
    /// the unfused product stores it.
    #[must_use]
    pub fn store() -> &'static Epilogue {
        static STORE: OnceLock<&'static Epilogue> = OnceLock::new();
        STORE.get_or_init(|| {
            Epilogue::interned(&Anchored::plain(rule(Layout::Dense)))
                .expect("the plain store is an epilogue the skeletons take")
        })
    }

    /// The products each `(m, n)` has, one a packed part.
    #[must_use]
    pub const fn parts(&self) -> u32 {
        self.parts
    }

    /// What its inputs are, in order.
    #[must_use]
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// The tiles it lands.
    #[must_use]
    pub const fn results(&self) -> usize {
        self.results
    }

    /// The cuBLASLt epilogue that lands what it does, where one does.
    #[must_use]
    pub const fn lt(&self) -> Option<LtEpilogue> {
        self.lt
    }

    /// Whether each product rounds to bf16 before the body reads it.
    #[must_use]
    pub const fn rounds(&self) -> bool {
        self.round
    }

    /// The bytes of one product as the unfused leg lands it: bf16 where
    /// it rounds, its f32 sum where it does not.
    #[must_use]
    pub const fn product_bytes(&self) -> usize {
        if self.round { 2 } else { 4 }
    }

    /// Whether it is the plain store, no epilogue at all.
    #[must_use]
    pub const fn plain(&self) -> bool {
        self.plain
    }

    /// The columns of each group its body combines within: 1 where it is
    /// pointwise.
    #[must_use]
    pub const fn span(&self) -> u32 {
        self.span
    }

    /// The window of the tile result `k` lands at `n` columns.
    #[must_use]
    pub fn window(&self, k: usize, n: u32) -> Window {
        self.windows
            .get(k)
            .copied()
            .unwrap_or(Window { from: 0, width: n })
    }

    /// Whether it serves a tile of `n` columns: any, or the one its body is
    /// of alone.
    #[must_use]
    pub fn fits(&self, n: u32) -> bool {
        self.fixed.is_none_or(|fixed| fixed == n)
    }

    /// The printed body's hash: its part of a kernel's key.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.body.hash
    }

    /// The norm its operand's rows take, where they are normed.
    #[must_use]
    pub const fn norm(&self) -> Option<RowNorm> {
        self.norm
    }

    /// The floats its body tables what a column alone computes in.
    #[must_use]
    pub const fn table(&self) -> u32 {
        self.body.table
    }

    /// The kernel of skeleton template `template` (`::pie::linear::<template>`)
    /// at its shape class `class` with this body filled in.
    pub(crate) fn kernel(&self, template: &str, class: &str) -> &'static Kernel {
        crate::jit::generated(
            self.root,
            &format!(
                "::pie::linear::{template}<{BODY}, {class}, {}, {}, {}, {}, {}>",
                self.parts,
                self.inputs.len(),
                self.results,
                self.round,
                self.norm.is_some()
            ),
        )
    }

    /// The `pie::linear::Epi<IN, OUT>` a fire passes: `ins` (one a region
    /// input, each of its kind) and `outs` (one a result, each `m` rows of
    /// its window of the `n`-column tile, as wide a row).
    pub(crate) fn planes(&self, ins: &[In], outs: &[u64], n: u32) -> Result<Vec<u64>, Error> {
        if ins.len() != self.inputs.len() || outs.len() != self.results {
            return Err(refuse(
                OP,
                format!(
                    "the epilogue takes {} input and {} result planes and the fire passes {} and \
                     {}",
                    self.inputs.len(),
                    self.results,
                    ins.len(),
                    outs.len()
                ),
            ));
        }
        for (k, ((input, kind), dtype)) in
            ins.iter().zip(&self.inputs).zip(&self.stored).enumerate()
        {
            let unplaned = *kind == Input::Norm;
            if (input.base == 0 && !unplaned) || input.kind() != *kind || input.dtype() != *dtype
            {
                return Err(refuse(
                    OP,
                    format!("input {k} is not a {kind:?} plane stored {dtype:?}"),
                ));
            }
        }
        if outs.contains(&0) {
            return Err(refuse(OP, "a result's plane is null"));
        }
        // The inputs and one unread past them, the results, then each
        // result's row and its window's columns.
        let mut words: Vec<u64> = ins
            .iter()
            .chain([&In::one(0)])
            .flat_map(|p| {
                [
                    p.base,
                    p.row_stride.cast_unsigned(),
                    p.col_stride.cast_unsigned(),
                    p.dtype.cast_unsigned(),
                    p.period.cast_unsigned(),
                ]
            })
            .collect();
        words.extend(outs);
        let windows: Vec<Window> = (0..outs.len()).map(|k| self.window(k, n)).collect();
        words.extend(windows.iter().map(|w| u64::from(w.width)));
        words.extend(windows.iter().map(|w| u64::from(w.from)));
        words.extend(windows.iter().map(|w| u64::from(w.from + w.width)));
        words.push(u64::from(self.norm.map_or(0, |norm| norm.eps.to_bits())));
        Ok(words)
    }

    /// Compiles every kernel a fire of this body at `n x k` may take (each
    /// skeleton at each count of rows it covers, and the unfused leg's row
    /// kernel) or reads it from the disk cache, and loads it on the current
    /// device: an anchored region a load forms is loaded with the model.
    pub fn load_every(&self, n: u32, k: u32, tuning: &Tuning) -> Result<(), Error> {
        let (ni, ki) = (count_of(n)?, count_of(k)?);
        let mut skeletons = Vec::new();
        for skeleton in Skeleton::ALL {
            let rows: Vec<u32> = match skeleton {
                Skeleton::Gemv => vec![1],
                Skeleton::Rows => (1..=ROWS_M).collect(),
                // one kernel at every count of rows
                Skeleton::Skinny => vec![1],
            };
            for m in rows {
                if skeleton.covers(self, count_of(m)?, ni, ki, tuning) {
                    skeletons.push((skeleton, m));
                }
            }
        }
        self.load(&skeletons, n, k, tuning)?;
        #[cfg(feature = "cuda")]
        if self.leg(n).is_some() && RowLeg::of(self, n).is_none() {
            return Err(refuse(
                OP,
                format!("the unfused leg's row kernel over {n} columns did not load"),
            ));
        }
        Ok(())
    }

    /// The row skeleton's pieces of a row for this body's unfused leg over
    /// `n` columns, or `None` where the row skeleton does not carry it: a row
    /// in no whole runs of eight, a tile its body is not of, or a region it
    /// refuses. A row wider than the skeleton holds is cut into slices of
    /// equal width (whole groups of its span), each its own launch over every
    /// row; an epilogue of windows is cut into them, each landing its
    /// result. A piece from column `from` reads every input as it does
    /// whole (a vector a column from the piece's column on) and computes
    /// what the whole row does there: each column it reads and gathers at
    /// shifted by `from` ([`shifted`]).
    fn leg(&self, n: u32) -> Option<LegShape> {
        use crate::synth::{In as Form, MAX_WIDTH, Out};
        if self.plain || n == 0 || !n.is_multiple_of(8) || !self.fits(n) {
            return None;
        }
        let bf16 = Dtype::Bf16;
        let product = if self.round { bf16 } else { Dtype::F32 };
        let ins: Vec<Form> = (0..self.parts)
            .map(|_| Form::Rows(product))
            .chain(
                self.inputs
                    .iter()
                    .zip(&self.stored)
                    .map(|(input, &dtype)| match input {
                        Input::Tile => Form::Rows(bf16),
                        Input::Column | Input::Head(_) => Form::Shared(bf16),
                        Input::Row | Input::Norm => Form::Each(dtype),
                        Input::One => Form::One(dtype),
                    }),
            )
            .collect();
        // Each window (the whole tile, or each result's), cut into slices of
        // equal width the row skeleton holds, whole groups of the span:
        // `(from, width, outs)`.
        let segments: Vec<Cut> = if self.windows.is_empty() {
            vec![(0, n, (0..self.results).map(|k| (k, 0, n)).collect())]
        } else {
            (0..self.results)
                .map(|k| {
                    let w = self.windows[k];
                    (w.from, w.width, vec![(k, 0, w.width)])
                })
                .collect()
        };
        let span = self.span.max(1);
        let mut pieces: Vec<Cut> = Vec::new();
        for (from, wide, outs) in segments {
            let slices = (wide.div_ceil(MAX_WIDTH)..=wide / 8).find(|c| {
                wide.is_multiple_of(*c)
                    && (wide / c).is_multiple_of(8)
                    && (wide / c).is_multiple_of(span)
            })?;
            let width = wide / slices;
            for s in 0..slices {
                let at = s * width;
                let landed = outs.iter().map(|&(k, col, ld)| (k, col + at, ld)).collect();
                pieces.push((from + at, width, landed));
            }
        }
        let mut shaped = Vec::with_capacity(pieces.len());
        for (from, width, outs) in pieces {
            let results: Vec<u32> = outs.iter().map(|(k, ..)| self.region.results[*k]).collect();
            let region = shifted(&self.region, from, width, results);
            let landed = vec![Out::Rows(bf16); outs.len()];
            crate::synth::supports(&region, &ins, &landed).ok()?;
            shaped.push(PieceShape {
                region,
                from,
                width,
                outs,
            });
        }
        Some(LegShape {
            ins,
            pieces: shaped,
        })
    }

    /// Compiles every kernel `skeletons` fires of this body at `n x k` (or
    /// reads it from the disk cache) and loads it on the current device: an
    /// epilogue a load forms is loaded with the model, before a fire.
    pub fn load(
        &self,
        skeletons: &[(Skeleton, u32)],
        n: u32,
        k: u32,
        tuning: &Tuning,
    ) -> Result<(), Error> {
        for &(skeleton, m) in skeletons {
            skeleton.kernel(self, m, n, k, tuning)?.load()?;
        }
        Ok(())
    }
}

/// Whether the contraction skeletons serve `anchored` at every count of rows
/// a fire may hand it, its inputs stored as `ins` and its results as
/// `outs`, or why not: an epilogue the skeletons take (bf16 planes in and out)
/// whose unfused leg the row skeleton carries, so that a shape no skeleton covers
/// lands all the same ([`crate::linear::gemm::anchored`]). What a compiler
/// asks before it forms one.
pub fn serves(anchored: &Anchored, ins: &[Dtype], outs: &[Dtype]) -> Result<(), Error> {
    if let Some(dtype) = outs.iter().find(|d| **d != Dtype::Bf16) {
        return Err(refuse(
            OP,
            format!("a result stored {dtype:?}, and the skeletons land bf16"),
        ));
    }
    let epi = Epilogue::interned_stored(anchored, ins)?;
    if ins.len() != planes_of(anchored) || outs.len() != epi.results {
        return Err(refuse(
            OP,
            "the planes given are not the epilogue's inputs and results",
        ));
    }
    let n = anchored.contraction.n;
    if epi.leg(n).is_none() {
        return Err(refuse(
            OP,
            format!(
                "the row skeleton carries no unfused leg of this epilogue over {n} columns, so a \
                 shape no skeleton covers would land nothing"
            ),
        ));
    }
    Ok(())
}

/// A run of the tile's columns, `(from, width, outs)`: each result `(k,
/// at, ld)` of `outs` landed from its plane's column `at`, its rows `ld`
/// apart.
type Cut = (u32, u32, Vec<(usize, u32, u32)>);

/// The planes `anchored`'s epilogue reads: one an input but a row's norm
/// scale, the contraction's own.
fn planes_of(anchored: &Anchored) -> usize {
    anchored.epilogue.as_ref().map_or(0, |e| {
        e.inputs.iter().filter(|i| **i != Input::Norm).count()
    })
}

/// An epilogue's unfused leg as the row skeleton takes it: its parameters'
/// forms and its pieces.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct LegShape {
    ins: Vec<crate::synth::In>,
    pieces: Vec<PieceShape>,
}

/// One piece of a leg: the region over the tile's columns `from..from +
/// width`, landing each result `k` of `outs` into its plane from that
/// plane's column `at`, its rows `ld` apart: `(k, at, ld)`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct PieceShape {
    region: Region,
    from: u32,
    width: u32,
    outs: Vec<(usize, u32, u32)>,
}

/// `region` over the `width` columns from `from` on, as a row of its own,
/// landing `results`: each column it reads (`Op::Column`) and each one it
/// gathers at (`Op::Peer`) shifted by `from`, so that it computes what the
/// whole row computes there.
fn shifted(region: &Region, from: u32, width: u32, results: Vec<u32>) -> Region {
    let reads = region
        .ops
        .iter()
        .any(|op| matches!(op, Op::Column | Op::Peer(..)));
    if from == 0 || !reads {
        let kept: Vec<u32> = results.clone();
        return Region {
            width,
            results: kept,
            ..region.clone()
        };
    }
    let params = u32::try_from(region.params.len()).unwrap_or(u32::MAX);
    let mut ops: Vec<Op> = Vec::with_capacity(region.ops.len() + 8);
    let mut of: Vec<u32> = (0..params).collect();
    let push = |ops: &mut Vec<Op>, op: Op| {
        ops.push(op);
        params + u32::try_from(ops.len() - 1).unwrap_or(u32::MAX)
    };
    for op in &region.ops {
        let m = |v: u32| of[v as usize];
        let id = match *op {
            Op::Column => {
                let c = push(&mut ops, Op::Column);
                let k = push(&mut ops, Op::Const(Literal::U32(from)));
                push(&mut ops, Op::Binary(Binary::Add, c, k))
            }
            Op::Peer(src, at) => {
                let k = push(&mut ops, Op::Const(Literal::U32(from)));
                let local = push(&mut ops, Op::Binary(Binary::Sub, m(at), k));
                push(&mut ops, Op::Peer(m(src), local))
            }
            Op::Const(l) => push(&mut ops, Op::Const(l)),
            Op::Unary(u, a) => push(&mut ops, Op::Unary(u, m(a))),
            Op::Binary(b, x, y) => push(&mut ops, Op::Binary(b, m(x), m(y))),
            Op::Select(c, x, y) => push(&mut ops, Op::Select(m(c), m(x), m(y))),
            Op::Cast(a, e) => push(&mut ops, Op::Cast(m(a), e)),
            Op::Broadcast(a) => push(&mut ops, Op::Broadcast(m(a))),
            Op::Reduce(r, a) => push(&mut ops, Op::Reduce(r, m(a))),
            Op::ReduceHeads(r, a, size) => push(&mut ops, Op::ReduceHeads(r, m(a), size)),
            Op::Fma(x, y, z) => push(&mut ops, Op::Fma(m(x), m(y), m(z))),
        };
        of.push(id);
    }
    Region {
        width,
        params: region.params.clone(),
        ops,
        results: results.into_iter().map(|r| of[r as usize]).collect(),
        rounds: region
            .rounds
            .iter()
            .map(|&(v, dtype)| (of[v as usize], dtype))
            .collect(),
    }
}

fn count_of(x: u32) -> Result<i32, Error> {
    i32::try_from(x).map_err(|_| refuse(OP, format!("an extent of {x} is past an i32")))
}

/// The width the plain store's region is stated at: a pointwise region's
/// body is the same at every width.
const RULE_WIDTH: u32 = 8;

/// The plain store's contraction: bf16 operands, each product rounded to
/// bf16 as the unfused product stores it.
const fn rule(layout: Layout) -> Contraction {
    Contraction {
        n: RULE_WIDTH,
        k: RULE_WIDTH,
        layout,
        quant: Quant::Plain(Dtype::Bf16),
        acc: Some(Dtype::Bf16),
    }
}

/// The plain store's region: each part's product, as it is.
fn store(width: u32, parts: u32) -> Region {
    Region {
        width,
        params: vec![Shape::Row; parts as usize],
        ops: Vec::new(),
        results: (0..parts).collect(),
        rounds: Vec::new(),
    }
}

/// The generated unit of `body`: one a body hash, for the life of the
/// process.
fn unit(body: &Body) -> Result<Root, Error> {
    static UNITS: OnceLock<Mutex<HashMap<u64, Root>>> = OnceLock::new();
    let text = format!(
        "// A contraction skeleton's epilogue: region body {:016x}.\n\
         #include \"linear/contract.cuh\"\n\n\
         namespace pie::linear::hooked {{\n\n{}\n}}\n",
        body.hash, body.source
    );
    let mut units = UNITS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("epilogue unit table lock poisoned");
    if let Some(root) = units.get(&body.hash) {
        return if root.text == text {
            Ok(*root)
        } else {
            Err(refuse(
                OP,
                format!(
                    "two bodies of different source share the hash {:016x}",
                    body.hash
                ),
            ))
        };
    }
    let file: &'static str =
        Box::leak(format!("linear/hooked/{:016x}.cu", body.hash).into_boxed_str());
    let root = Root::generated(file, Box::leak(text.into_boxed_str()));
    units.insert(body.hash, root);
    Ok(root)
}

/// A contraction skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Skeleton {
    /// One row, a warp (or a block split over K) a column.
    Gemv,
    /// 1 to 16 rows, a block's eight warps streaming weight rows; one dense
    /// bank.
    Rows,
    /// 1 to 64 rows on mma.sync, `kernel.gemm.skinny_rows` at most.
    Skinny,
}

impl Skeleton {
    pub const ALL: [Skeleton; 3] = [Skeleton::Gemv, Skeleton::Rows, Skeleton::Skinny];

    /// Whether the skeleton serves `m x n x k` with `epi` filled in under
    /// `tuning`.
    #[must_use]
    pub fn covers(self, epi: &Epilogue, m: i32, n: i32, k: i32, tuning: &Tuning) -> bool {
        if m <= 0 || n <= 0 || k <= 0 || k % 8 != 0 || !epi.fits(n.unsigned_abs()) {
            return false;
        }
        // A body that combines a head's columns needs the head's lanes side
        // by side: the skinny skeleton's tile alone holds whole heads.
        match self {
            Skeleton::Gemv => m == 1 && epi.span == 1,
            Skeleton::Rows => epi.parts == 1 && m <= ROWS_M as i32 && epi.span == 1,
            Skeleton::Skinny => crate::linear::skinny::covers_span(
                m,
                n,
                k,
                epi.parts,
                tuning.gemm_skinny_rows,
                epi.span,
            ),
        }
    }

    /// The kernel this skeleton fires at `m x n x k` with `epi` filled in.
    pub(crate) fn kernel(
        self,
        epi: &Epilogue,
        m: u32,
        n: u32,
        k: u32,
        tuning: &Tuning,
    ) -> Result<&'static Kernel, Error> {
        #[cfg(feature = "cuda")]
        {
            let (m, n, k) = (count_of(m)?, count_of(n)?, count_of(k)?);
            match self {
                Skeleton::Gemv => Ok(crate::linear::gemv::gemv_kernel(epi, n, tuning).0),
                Skeleton::Rows => Ok(crate::linear::gemv::rows_kernel(epi, m, n, k).0),
                Skeleton::Skinny => Ok(crate::linear::skinny::kernel(epi)),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (epi, m, n, k, tuning);
            Err(crate::jit::runtimeless(OP))
        }
    }

    /// `outs = epilogue(act . w^T)` over `m` rows on this skeleton: `w` the
    /// weight (`parts · n x k`), `act` the activation (`m x k`), `ins` the
    /// epilogue's inputs and `outs` its results, each `m x n`.
    #[allow(clippy::too_many_arguments)]
    pub fn fire(
        self,
        ctx: &crate::jit::Ctx,
        epi: &Epilogue,
        w: u64,
        act: u64,
        ins: &[In],
        outs: &[u64],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), Error> {
        #[cfg(feature = "cuda")]
        {
            match self {
                Skeleton::Gemv => {
                    if m != 1 {
                        return Err(refuse(OP, format!("the gemv serves one row, not {m}")));
                    }
                    crate::linear::gemv::gemv(ctx, epi, w, act, ins, outs, n, k)
                }
                Skeleton::Rows => crate::linear::gemv::gemm_rows(
                    ctx,
                    epi,
                    w,
                    act,
                    ins,
                    outs,
                    m,
                    n,
                    k,
                    ctx.tuning().pdl,
                ),
                Skeleton::Skinny => {
                    crate::linear::skinny::skinny(ctx, epi, w, act, ins, outs, m, n, k)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (ctx, epi, w, act, ins, outs, m, n, k);
            Err(crate::jit::runtimeless(OP))
        }
    }
}

/// An epilogue unfused: the products landed by the plain contraction in a
/// plane of their own, then the epilogue's region over them on the row
/// skeleton (`crate::synth`), a row of the product tile a row of the skeleton
/// (or, for a row wider than the skeleton holds or one of windows, a piece
/// of one a launch). The candidate a fused skeleton is measured against.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowLeg {
    pieces: &'static [Piece],
    parts: u32,
    /// The products' dtype: bf16, or f32 where the epilogue reads them as
    /// they are summed.
    product: Dtype,
}

/// One launch of a leg: the kernel over the tile's columns `from..from +
/// width`, landing each result `(k, at, ld)` of `outs` from its plane's
/// column `at`, its rows `ld` apart.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
struct Piece {
    kernel: crate::synth::Synthesized,
    from: u32,
    width: u32,
    outs: Vec<(usize, u32, u32)>,
}

#[cfg(feature = "cuda")]
impl RowLeg {
    /// The row skeleton's kernels for `epi` over `n` columns, loaded on the
    /// current device, or `None` where the row skeleton does not carry it
    /// ([`Epilogue::leg`]).
    pub(crate) fn of(epi: &Epilogue, n: u32) -> Option<RowLeg> {
        /// Each epilogue's leg by body, parts, the products' rounding, its
        /// windows and inputs' storage, and the tile's columns.
        type Legs = HashMap<(u64, u32), Option<RowLeg>>;
        static LEGS: OnceLock<Mutex<Legs>> = OnceLock::new();
        let shape = epi.leg(n)?;
        let mut legs = LEGS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("row leg table lock poisoned");
        let laid = kernels_ir::fnv1a64(format!("{:?}{:?}", epi.windows, epi.stored).as_bytes());
        *legs
            .entry((
                epi.body.hash ^ u64::from(epi.parts) ^ (u64::from(epi.round) << 63) ^ laid,
                n,
            ))
            .or_insert_with(|| {
                let mut pieces = Vec::with_capacity(shape.pieces.len());
                for piece in shape.pieces {
                    let kernel =
                        crate::synth::synth(&piece.region, &shape.ins, piece.outs.len()).ok()?;
                    kernel.load().ok()?;
                    pieces.push(Piece {
                        kernel,
                        from: piece.from,
                        width: piece.width,
                        outs: piece.outs,
                    });
                }
                Some(RowLeg {
                    pieces: Box::leak(pieces.into_boxed_slice()),
                    parts: epi.parts,
                    product: if epi.round { Dtype::Bf16 } else { Dtype::F32 },
                })
            })
    }

    /// The epilogue over `products` (`m x parts · n`, the plain product of
    /// the packed bank), `ins` its inputs and `outs` its results.
    pub(crate) fn fire(
        self,
        ctx: &crate::jit::Ctx,
        products: u64,
        ins: &[In],
        outs: &[u64],
        m: u32,
        n: u32,
    ) -> Result<(), Error> {
        use crate::synth::Plane;
        use crate::tensor::Tensor;
        let rows = |base: u64, stride: u32| Plane::rows(&Tensor::new(base, m, stride, Dtype::Bf16));
        let products_of =
            |base: u64, stride: u32| Plane::rows(&Tensor::new(base, m, stride, self.product));
        let size = self.product.bits() / 8;
        // The leg covers every row its product landed, as the product and
        // the skeletons do: a staged window's live count would leave rows past
        // it unwritten.
        let crate::jit::ArgValue::Ptr(stage) = ctx.stage() else {
            unreachable!("a stage window is an address");
        };
        ctx.disarm_stage();
        let mut fired = Ok(());
        for piece in self.pieces {
            // The piece's first column, in bytes of bf16.
            let at = u64::from(piece.from) * 2;
            let mut planes: Vec<Plane> = (0..self.parts)
                .map(|p| {
                    let from = u64::from(p * n + piece.from) * size;
                    products_of(products + from, self.parts * n)
                })
                .collect();
            for input in ins {
                let dtype = input.dtype();
                planes.push(match input.kind() {
                    Input::Tile => {
                        let ld = u32::try_from(input.row_stride).unwrap_or(0);
                        rows(input.base + at, ld)
                    }
                    Input::Column => {
                        Plane::shared(&Tensor::new(input.base + at, 1, piece.width, Dtype::Bf16))
                    }
                    Input::Head(head) => {
                        Plane::shared(&Tensor::new(input.base, 1, head, Dtype::Bf16))
                    }
                    Input::Row => Plane::each(&Tensor::new(input.base, m, 1, dtype)),
                    // each row's norm scale, the plane `norm_scales` landed
                    Input::Norm => Plane::each(&Tensor::new(input.base, m, 1, Dtype::F32)),
                    Input::One => Plane::shared(&Tensor::new(input.base, 1, piece.width, dtype)),
                });
            }
            let results: Vec<Plane> = piece
                .outs
                .iter()
                .map(|&(k, col, ld)| rows(outs[k] + u64::from(col) * 2, ld))
                .collect();
            fired = piece.kernel.fire(ctx, &planes, &results, m);
            if fired.is_err() {
                break;
            }
        }
        if stage != 0 {
            ctx.arm_stage(stage);
        }
        fired
    }
}

/// Each of `m` operand rows' norm scale under `norm` (`rsqrt(Σ a² / k +
/// eps)`, f32) into `out`: what a normed epilogue's unfused leg reads as its
/// `Input::Norm` (`In::norm().at(out)`), a skeleton computing its own.
#[cfg(feature = "cuda")]
pub(crate) fn norm_scales(
    ctx: &crate::jit::Ctx,
    norm: RowNorm,
    act: u64,
    out: u64,
    m: u32,
    k: u32,
) -> Result<(), Error> {
    use crate::jit::{ArgValue, Fire, Launch};
    const BLOCK: u32 = 256;
    if k == 0 || !k.is_multiple_of(8) {
        return Err(refuse(OP, format!("a row of {k} is no whole run of eight")));
    }
    ctx.fire(
        "linear.norm_scales",
        Fire::at(crate::jit::kernel(
            "linear/contract.cu",
            &format!("::pie::linear::norm_scales<{BLOCK}>"),
        ))
        .apply(Launch::per_row(m, BLOCK).programmatic(ctx.tuning().pdl)),
        &[
            ArgValue::Ptr(act),
            ArgValue::Ptr(out),
            ArgValue::I32(count_of(k)?),
            ArgValue::F32(norm.eps),
        ],
    )
}

/// The most rows the few-rows skeleton holds.
pub const ROWS_M: u32 = 16;

#[cfg(test)]
mod tests {
    use super::*;

    use kernels_ir::{Binary, Op};

    fn ruled(layout: Layout, region: Region, inputs: Vec<Input>) -> &'static Epilogue {
        Epilogue::interned(&Anchored {
            epilogue: Some(Pointwise { region, inputs }),
            ..Anchored::plain(rule(layout))
        })
        .expect("a hand rule's epilogue is one the skeletons take")
    }

    #[test]
    fn the_plain_store_is_an_epilogue_the_skeletons_take() {
        let store = Epilogue::store();
        assert_eq!(
            (store.parts(), store.inputs(), store.results()),
            (1, &[][..], 1)
        );
        assert!(std::ptr::eq(
            Epilogue::interned(&Anchored::plain(rule(Layout::Dense))).expect("the store"),
            store
        ));
        // The plain store copies each product through its store.
        assert!(
            store.body.source.contains("f.st(0, e, f.ld(0, e))"),
            "{}",
            store.body.source
        );
    }

    #[test]
    fn a_prologue_or_another_dtype_is_refused_by_name() {
        let mut anchored = Anchored::plain(rule(Layout::Dense));
        anchored.contraction.acc = Some(Dtype::F16);
        let why = Epilogue::of(&anchored).expect_err("refused").to_string();
        assert!(why.contains("round to F16"), "{why}");
        let mut anchored = Anchored::plain(rule(Layout::Dense));
        anchored.contraction.quant = Quant::Plain(Dtype::F32);
        let why = Epilogue::of(&anchored).expect_err("refused").to_string();
        assert!(why.contains("stored Plain(F32)"), "{why}");
    }

    #[test]
    fn planes_not_of_the_epilogues_kinds_are_refused() {
        let bias = ruled(
            Layout::Dense,
            Region {
                width: RULE_WIDTH,
                params: vec![Shape::Row, Shape::Row],
                ops: vec![Op::Binary(Binary::Add, 0, 1)],
                results: vec![2],
                rounds: vec![(2, Dtype::Bf16)],
            },
            vec![Input::Column],
        );
        assert!(bias.planes(&[In::row(64)], &[128], 8).is_err());
        assert!(bias.planes(&[], &[128], 8).is_err());
        let words = bias.planes(&[In::column(64)], &[128], 8).expect("a bias");
        // One input (its base, strides, dtype and period), one unread, one
        // result, its row's stride and its window's columns, then the norm's
        // eps (none).
        assert_eq!(words, vec![64, 0, 1, 0, 0, 0, 0, 0, 0, 0, 128, 8, 0, 8, 0]);
        // A row's norm scale is no plane: its input carries none.
        assert_eq!(In::norm().kind(), Input::Norm);
        // A head's weight and a row's i32 position, each of its kind.
        assert_eq!(In::head(64, 32).kind(), Input::Head(32));
        assert_eq!(In::row(64).i32().dtype(), Dtype::I32);
        assert!(bias.planes(&[In::column(64).i32()], &[128], 8).is_err());
    }
}
