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
//!   No epilogue is the plain store: each product stored as it is. A region
//!   over a tile is pointwise (kernels-ir refuses one that is not), so its
//!   body is the same at every width, and one epilogue serves every `n`.
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
pub use kernels_ir::anchor::{Anchored, Contraction, Input, Layout, Pointwise, Quant};
use kernels_ir::{Region, Shape};

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
/// at `base[row * row_stride + col * col_stride]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct In {
    base: u64,
    row_stride: i64,
    col_stride: i64,
}

impl In {
    /// A tile of the product's extent, its rows `ld` apart (a residual).
    #[must_use]
    pub const fn tile(base: u64, ld: u32) -> In {
        In {
            base,
            row_stride: ld as i64,
            col_stride: 1,
        }
    }

    /// A vector one a column, every row reading it (a bias).
    #[must_use]
    pub const fn column(base: u64) -> In {
        In {
            base,
            row_stride: 0,
            col_stride: 1,
        }
    }

    /// A vector one a row (a row's scale).
    #[must_use]
    pub const fn row(base: u64) -> In {
        In {
            base,
            row_stride: 1,
            col_stride: 0,
        }
    }

    /// One value for the whole tile (a layer's scale).
    #[must_use]
    pub const fn one(base: u64) -> In {
        In {
            base,
            row_stride: 0,
            col_stride: 0,
        }
    }

    /// The same input read from tile row `row` on: a run of the tile's rows
    /// served apart.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub(crate) const fn from_row(self, row: usize) -> In {
        In {
            base: self.base + (row as i64 * self.row_stride * 2) as u64,
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
        match (self.row_stride, self.col_stride) {
            (0, 0) => Input::One,
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
    }
}

impl Eq for Epilogue {}

impl Epilogue {
    /// `anchored`'s epilogue printed for the skeletons' hook, or why no skeleton
    /// takes it: a skeleton reads bf16 operands as they are stored (a prologue
    /// runs as a region of its own before the product), rounds a product to
    /// bf16 or not at all, and lands bf16 planes.
    pub fn of(anchored: &Anchored) -> Result<Epilogue, Error> {
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
        let body = Body::of(&region).map_err(|why| refuse(OP, why.to_string()))?;
        let plain = anchored.epilogue.is_none();
        let root = unit(&body)?;
        let lt = if parts == 1 && !round && !plain {
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
        })
    }

    /// `anchored`'s epilogue ([`Epilogue::of`]), one for the life of the
    /// process: the same reference for every call with the same region.
    pub fn interned(anchored: &Anchored) -> Result<&'static Epilogue, Error> {
        static EPILOGUES: OnceLock<Mutex<HashMap<u64, &'static Epilogue>>> = OnceLock::new();
        let key = anchored.key();
        let table = EPILOGUES.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(found) = table
            .lock()
            .expect("epilogue table lock poisoned")
            .get(&key)
        {
            return Ok(found);
        }
        let made: &'static Epilogue = Box::leak(Box::new(Epilogue::of(anchored)?));
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

    /// The printed body's hash: its part of a kernel's key.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.body.hash
    }

    /// The kernel of skeleton template `template` (`::pie::linear::<template>`)
    /// at its shape class `class` with this body filled in.
    pub(crate) fn kernel(&self, template: &str, class: &str) -> &'static Kernel {
        crate::jit::generated(
            self.root,
            &format!(
                "::pie::linear::{template}<{BODY}, {class}, {}, {}, {}, {}>",
                self.parts,
                self.inputs.len(),
                self.results,
                self.round
            ),
        )
    }

    /// The `pie::linear::Epi<IN, OUT>` a fire passes: `ins` (one a region
    /// input, each of its kind) and `outs` (one a result, each `m x n`, `ld`
    /// elements a row).
    pub(crate) fn planes(&self, ins: &[In], outs: &[u64], ld: u32) -> Result<Vec<u64>, Error> {
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
        for (k, (input, kind)) in ins.iter().zip(&self.inputs).enumerate() {
            if input.base == 0 || input.kind() != *kind {
                return Err(refuse(OP, format!("input {k} is not a {kind:?} plane")));
            }
        }
        if outs.contains(&0) {
            return Err(refuse(OP, "a result's plane is null"));
        }
        // The inputs and one unread past them, the results, then `ld`.
        let mut words: Vec<u64> = ins
            .iter()
            .chain([&In::one(0)])
            .flat_map(|p| {
                [
                    p.base,
                    p.row_stride.cast_unsigned(),
                    p.col_stride.cast_unsigned(),
                ]
            })
            .collect();
        words.extend(outs);
        words.push(u64::from(ld));
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

    /// The row skeleton's region, planes and slices of a row for this body's
    /// unfused leg over `n` columns, or `None` where the row skeleton does not
    /// carry it: a row in no whole runs of eight, or a region it refuses. A
    /// row wider than the skeleton holds is cut into slices of equal width,
    /// each its own launch over every row, so every input reads as it does
    /// whole (a vector a column from the slice's column on).
    fn leg(&self, n: u32) -> Option<LegShape> {
        use crate::synth::{In as Form, MAX_WIDTH, Out};
        if self.plain || n == 0 || !n.is_multiple_of(8) {
            return None;
        }
        let slices = (n.div_ceil(MAX_WIDTH)..=n / 8)
            .find(|c| n.is_multiple_of(*c) && (n / c).is_multiple_of(8))?;
        let width = n / slices;
        let bf16 = Dtype::Bf16;
        let product = if self.round { bf16 } else { Dtype::F32 };
        let ins: Vec<Form> = (0..self.parts)
            .map(|_| Form::Rows(product))
            .chain(self.inputs.iter().map(|input| match input {
                Input::Tile => Form::Rows(bf16),
                Input::Column => Form::Shared(bf16),
                Input::Row => Form::Each(bf16),
                Input::One => Form::One(bf16),
            }))
            .collect();
        let outs = vec![Out::Rows(bf16); self.results];
        let region = Region {
            width,
            ..self.region.clone()
        };
        crate::synth::supports(&region, &ins, &outs).ok()?;
        Some(LegShape {
            region,
            ins,
            outs: outs.len(),
            width,
            slices,
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
    if let Some(dtype) = ins.iter().chain(outs).find(|d| **d != Dtype::Bf16) {
        return Err(refuse(
            OP,
            format!("a plane stored {dtype:?}, and the skeletons read and land bf16"),
        ));
    }
    let epi = Epilogue::interned(anchored)?;
    if ins.len() != epi.inputs.len() || outs.len() != epi.results {
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

/// An epilogue's unfused leg as the row skeleton takes it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct LegShape {
    region: Region,
    ins: Vec<crate::synth::In>,
    outs: usize,
    width: u32,
    slices: u32,
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
        if m <= 0 || n <= 0 || k <= 0 || k % 8 != 0 {
            return false;
        }
        match self {
            Skeleton::Gemv => m == 1,
            Skeleton::Rows => epi.parts == 1 && m <= ROWS_M as i32,
            Skeleton::Skinny => {
                crate::linear::skinny::covers(m, n, k, epi.parts, tuning.gemm_skinny_rows)
            }
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
/// skeleton (`crate::synth`), a row of the product tile a row of the skeleton (or,
/// for a row wider than the skeleton holds, a slice of one a launch). The
/// candidate a fused skeleton is measured against.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowLeg {
    kernel: crate::synth::Synthesized,
    parts: u32,
    /// The products' dtype: bf16, or f32 where the epilogue reads them as
    /// they are summed.
    product: Dtype,
    width: u32,
    slices: u32,
}

#[cfg(feature = "cuda")]
impl RowLeg {
    /// The row skeleton's kernel for `epi` over `n` columns, loaded on the
    /// current device, or `None` where the row skeleton does not carry it
    /// ([`Epilogue::leg`]).
    pub(crate) fn of(epi: &Epilogue, n: u32) -> Option<RowLeg> {
        /// Each epilogue's leg by body, parts, the products' rounding and
        /// skeleton row width.
        type Legs = HashMap<(u64, u32), Option<RowLeg>>;
        static LEGS: OnceLock<Mutex<Legs>> = OnceLock::new();
        let shape = epi.leg(n)?;
        let mut legs = LEGS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("row leg table lock poisoned");
        *legs
            .entry((
                epi.body.hash ^ u64::from(epi.parts) ^ (u64::from(epi.round) << 63),
                shape.width,
            ))
            .or_insert_with(|| {
                let kernel = crate::synth::synth(&shape.region, &shape.ins, shape.outs).ok()?;
                kernel.load().ok()?;
                Some(RowLeg {
                    kernel,
                    parts: epi.parts,
                    product: if epi.round { Dtype::Bf16 } else { Dtype::F32 },
                    width: shape.width,
                    slices: shape.slices,
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
        for slice in 0..self.slices {
            // The slice's first column, in bytes.
            let at = u64::from(slice * self.width) * 2;
            let mut planes: Vec<Plane> = (0..self.parts)
                .map(|p| {
                    let from = u64::from(p * n + slice * self.width) * size;
                    products_of(products + from, self.parts * n)
                })
                .collect();
            for input in ins {
                planes.push(match input.kind() {
                    Input::Tile => {
                        let ld = u32::try_from(input.row_stride).unwrap_or(0);
                        rows(input.base + at, ld)
                    }
                    Input::Column => {
                        Plane::shared(&Tensor::new(input.base + at, 1, self.width, Dtype::Bf16))
                    }
                    Input::Row => Plane::each(&Tensor::new(input.base, m, 1, Dtype::Bf16)),
                    Input::One => {
                        Plane::shared(&Tensor::new(input.base, 1, self.width, Dtype::Bf16))
                    }
                });
            }
            let results: Vec<Plane> = outs.iter().map(|&out| rows(out + at, n)).collect();
            fired = self.kernel.fire(ctx, &planes, &results, m);
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
        // One input, one unread, one result, the row's stride.
        assert_eq!(words, vec![64, 0, 1, 0, 0, 0, 128, 8]);
    }
}
