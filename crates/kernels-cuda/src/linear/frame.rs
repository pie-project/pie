//! The contraction frames: pie's own decode GEMMs (`gemv`, the few-rows
//! kernel, `skinny`), each a hand-written parallel skeleton
//! (`linear/contract.cuh`) that hands its products, a few elements of one
//! row of the product tile at a time, to an epilogue body through the hooks
//! of `TileHook`.
//!
//! - **The epilogue** ([`Epilogue`]): an anchored region's (kernels-ir's
//!   [`Anchored`]) epilogue scheduled and printed by the row frame's printer
//!   ([`crate::synth::Body`]), whose hooks the tile hook provides, `f.ld(k,
//!   e)` reading the parts' products below the parts and the inputs above.
//!   No epilogue is the plain store: each product stored as it is. A region
//!   over a tile is pointwise (kernels-ir refuses one that is not), so its
//!   body is the same at every width, and one epilogue serves every `n`.
//! - **The frames** ([`Frame`]): which shapes each covers, and its fire with
//!   an epilogue filled in.
//! - **Identity.** One generated unit a body, `linear/hooked/<hash>.cu`:
//!   the frames' header and the body's source; one kernel a frame, its
//!   shape class (warps, rows, splits) and the body's planes. The disk cache
//!   keys the unit's text, the headers and the instantiation, as it does a
//!   carried unit's.
//! - **Loading.** A kernel loads on its first fire, as the carried kernels
//!   these frames replace did: the fused hand rules' bodies (the store, a
//!   bias, a soft cap, GEGLU) are fixed, and the tuner fires each before a
//!   capture. An epilogue a load forms loads with the model
//!   ([`Epilogue::load`]).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use dtype::Dtype;
pub use kernels_ir::anchor::{Anchored, Contraction, Input, Layout, Pointwise, Quant};
use kernels_ir::{Binary, Literal, Op, Region, Shape, Unary};

use crate::error::Error;
use crate::jit::{Kernel, Root, refuse};
use crate::synth::Body;
use crate::tuning::Tuning;

const OP: &str = "linear.frame";

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

/// An anchored region's epilogue printed for the frames' hook: the body,
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
    /// `anchored`'s epilogue printed for the frames' hook, or why no frame
    /// takes it: a frame reads bf16 operands as they are stored (a prologue
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
                "a prologue: the frames read the operand as it is stored, and one runs as its \
                 own region before the product",
            ));
        }
        if c.quant != Quant::Plain(Dtype::Bf16) {
            return Err(refuse(
                OP,
                format!(
                    "the operands are stored {:?}, and the frames read bf16",
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
                    format!("the products round to {other:?}, and the frames round to bf16"),
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
                    "an epilogue of {} input(s) and {} result(s), past the frames' {MAX_INS} and \
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
        Ok(Epilogue {
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
                .expect("the plain store is an epilogue the frames take")
        })
    }

    /// `y + bias[n]`, the product rounded first and the sum after: what
    /// `fused.matmul_bias` lands.
    #[must_use]
    pub fn bias() -> &'static Epilogue {
        static BIAS: OnceLock<&'static Epilogue> = OnceLock::new();
        BIAS.get_or_init(|| {
            let region = Region {
                width: RULE_WIDTH,
                params: vec![Shape::Row, Shape::Row],
                ops: vec![Op::Binary(Binary::Add, 0, 1)],
                results: vec![2],
                rounds: vec![(2, Dtype::Bf16)],
            };
            ruled(Layout::Dense, region, vec![Input::Column])
        })
    }

    /// `cap · tanh(y / cap)` over the rounded product: what
    /// `fused.matmul_softcap` lands. kernels-ir has no `tanh`, so it is
    /// `2 / (1 + e^(-2x)) - 1`, within an f32 rounding or two of `tanhf`
    /// but where `x` is so small that the bf16 product rounds it away.
    pub fn softcap(cap: f32) -> Result<&'static Epilogue, Error> {
        if !(cap.is_finite() && cap > 0.0) {
            return Err(refuse(
                OP,
                format!("a softcap of {cap} is not a positive number"),
            ));
        }
        let region = Region {
            width: RULE_WIDTH,
            params: vec![Shape::Row],
            ops: vec![
                Op::Const(Literal::F32(-2.0 / cap)), // 1
                Op::Binary(Binary::Mul, 0, 1),       // 2: -2y / cap
                Op::Unary(Unary::Exp, 2),            // 3
                Op::Const(Literal::F32(1.0)),        // 4
                Op::Binary(Binary::Add, 4, 3),       // 5
                Op::Const(Literal::F32(2.0)),        // 6
                Op::Binary(Binary::Div, 6, 5),       // 7
                Op::Binary(Binary::Sub, 7, 4),       // 8: tanh(y / cap)
                Op::Const(Literal::F32(cap)),        // 9
                Op::Binary(Binary::Mul, 9, 8),       // 10
            ],
            results: vec![10],
            rounds: vec![(10, Dtype::Bf16)],
        };
        Epilogue::interned(&Anchored {
            epilogue: Some(Pointwise {
                region,
                inputs: Vec::new(),
            }),
            ..Anchored::plain(rule(Layout::Dense))
        })
    }

    /// `gelu_tanh(gate) · up` over a packed `[gate | up]` bank's two rounded
    /// products, each `(m, n)` its gate and its up: what `fused.matmul_glu`
    /// (GEGLU) lands. `0.5 g (1 + tanh(z))` is `g / (1 + e^(-2z))`, `z =
    /// √(2/π) (g + 0.044715 g³)`.
    #[must_use]
    pub fn geglu_tanh() -> &'static Epilogue {
        static GEGLU: OnceLock<&'static Epilogue> = OnceLock::new();
        GEGLU.get_or_init(|| {
            const ALPHA: f32 = 0.797_884_6;
            const BETA: f32 = 0.044_715;
            let region = Region {
                width: RULE_WIDTH,
                params: vec![Shape::Row, Shape::Row],
                ops: vec![
                    Op::Binary(Binary::Mul, 0, 0),         // 2: g²
                    Op::Binary(Binary::Mul, 2, 0),         // 3: g³
                    Op::Const(Literal::F32(BETA)),         // 4
                    Op::Binary(Binary::Mul, 4, 3),         // 5
                    Op::Binary(Binary::Add, 0, 5),         // 6
                    Op::Const(Literal::F32(-2.0 * ALPHA)), // 7
                    Op::Binary(Binary::Mul, 7, 6),         // 8: -2z
                    Op::Unary(Unary::Exp, 8),              // 9
                    Op::Const(Literal::F32(1.0)),          // 10
                    Op::Binary(Binary::Add, 10, 9),        // 11
                    Op::Binary(Binary::Div, 0, 11),        // 12: gelu(g)
                    Op::Binary(Binary::Mul, 12, 1),        // 13
                ],
                results: vec![13],
                rounds: vec![(13, Dtype::Bf16)],
            };
            ruled(Layout::Packed(2), region, Vec::new())
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

    /// The kernel of frame template `template` (`::pie::linear::<template>`)
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

    /// Compiles every kernel `frames` fires of this body at `n x k` (or
    /// reads it from the disk cache) and loads it on the current device: an
    /// epilogue a load forms is loaded with the model, before a fire.
    pub fn load(
        &self,
        frames: &[(Frame, u32)],
        n: u32,
        k: u32,
        tuning: &Tuning,
    ) -> Result<(), Error> {
        for &(frame, m) in frames {
            frame.kernel(self, m, n, k, tuning)?.load()?;
        }
        Ok(())
    }
}

/// The width the hand rules' regions are stated at: a pointwise region's
/// body is the same at every width.
const RULE_WIDTH: u32 = 8;

/// A hand rule's contraction: bf16 operands, each product rounded to bf16
/// as the unfused product stores it.
const fn rule(layout: Layout) -> Contraction {
    Contraction {
        n: RULE_WIDTH,
        k: RULE_WIDTH,
        layout,
        quant: Quant::Plain(Dtype::Bf16),
        acc: Some(Dtype::Bf16),
    }
}

fn ruled(layout: Layout, region: Region, inputs: Vec<Input>) -> &'static Epilogue {
    Epilogue::interned(&Anchored {
        epilogue: Some(Pointwise { region, inputs }),
        ..Anchored::plain(rule(layout))
    })
    .expect("a hand rule's epilogue is one the frames take")
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
        "// A contraction frame's epilogue: region body {:016x}.\n\
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

/// A contraction frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Frame {
    /// One row, a warp (or a block split over K) a column.
    Gemv,
    /// 1 to 16 rows, a block's eight warps streaming weight rows; one dense
    /// bank.
    Rows,
    /// 1 to 64 rows on mma.sync, `kernel.gemm.skinny_rows` at most.
    Skinny,
}

impl Frame {
    pub const ALL: [Frame; 3] = [Frame::Gemv, Frame::Rows, Frame::Skinny];

    /// Whether the frame serves `m x n x k` with `epi` filled in under
    /// `tuning`.
    #[must_use]
    pub fn covers(self, epi: &Epilogue, m: i32, n: i32, k: i32, tuning: &Tuning) -> bool {
        if m <= 0 || n <= 0 || k <= 0 || k % 8 != 0 {
            return false;
        }
        match self {
            Frame::Gemv => m == 1,
            Frame::Rows => epi.parts == 1 && m <= ROWS_M as i32,
            Frame::Skinny => {
                crate::linear::skinny::covers(m, n, k, epi.parts, tuning.gemm_skinny_rows)
            }
        }
    }

    /// The kernel this frame fires at `m x n x k` with `epi` filled in.
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
            let (m, n, k) = (count(m)?, count(n)?, count(k)?);
            match self {
                Frame::Gemv => Ok(crate::linear::gemv::gemv_kernel(epi, n, tuning).0),
                Frame::Rows => Ok(crate::linear::gemv::rows_kernel(epi, m, n, k).0),
                Frame::Skinny => Ok(crate::linear::skinny::kernel(epi)),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (epi, m, n, k, tuning);
            Err(crate::jit::runtimeless(OP))
        }
    }

    /// `outs = epilogue(act . w^T)` over `m` rows on this frame: `w` the
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
                Frame::Gemv => {
                    if m != 1 {
                        return Err(refuse(OP, format!("the gemv serves one row, not {m}")));
                    }
                    crate::linear::gemv::gemv(ctx, epi, w, act, ins, outs, n, k)
                }
                Frame::Rows => crate::linear::gemv::gemm_rows(
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
                Frame::Skinny => {
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
/// frame (`crate::synth`), a row of the product tile a row of the frame (or,
/// for a row wider than the frame holds, a slice of one). The candidate a
/// fused frame is measured against.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowLeg {
    kernel: crate::synth::Synthesized,
    parts: u32,
    width: u32,
    slices: u32,
}

#[cfg(feature = "cuda")]
impl RowLeg {
    /// The row frame's kernel for `epi` over `n` columns, loaded on the
    /// current device, or `None` where the row frame does not carry it: a
    /// vector a row (the row frame reads one value for every row), a packed
    /// bank or a vector a column past one frame row, or a region it refuses.
    pub(crate) fn of(epi: &Epilogue, n: u32) -> Option<RowLeg> {
        use crate::synth::{In as Form, MAX_WIDTH, Out};
        if epi.plain || n == 0 || !n.is_multiple_of(8) || epi.inputs.contains(&Input::Row) {
            return None;
        }
        let slices = if n <= MAX_WIDTH {
            1
        } else if epi.parts == 1 && !epi.inputs.contains(&Input::Column) {
            (n.div_ceil(MAX_WIDTH)..=n / 8)
                .find(|c| n.is_multiple_of(*c) && (n / c).is_multiple_of(8))?
        } else {
            return None;
        };
        let width = n / slices;
        let bf16 = Dtype::Bf16;
        let ins: Vec<Form> = (0..epi.parts)
            .map(|_| Form::Rows(bf16))
            .chain(epi.inputs.iter().map(|input| match input {
                Input::Tile => Form::Rows(bf16),
                Input::Column => Form::Shared(bf16),
                Input::Row | Input::One => Form::One(bf16),
            }))
            .collect();
        let outs = vec![Out::Rows(bf16); epi.results];
        let region = Region {
            width,
            ..epi.region.clone()
        };
        /// Each epilogue's leg by body, parts and frame row width.
        type Legs = HashMap<(u64, u32), Option<RowLeg>>;
        static LEGS: OnceLock<Mutex<Legs>> = OnceLock::new();
        let mut legs = LEGS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("row leg table lock poisoned");
        *legs
            .entry((epi.body.hash ^ u64::from(epi.parts), width))
            .or_insert_with(|| {
                crate::synth::supports(&region, &ins, &outs).ok()?;
                let kernel = crate::synth::synth(&region, &ins, outs.len()).ok()?;
                kernel.load().ok()?;
                Some(RowLeg {
                    kernel,
                    parts: epi.parts,
                    width,
                    slices,
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
        let rows = m * self.slices;
        let plane =
            |base: u64, stride: u32| Plane::rows(&Tensor::new(base, rows, stride, Dtype::Bf16));
        // A slice's rows are `width` apart; a whole row's, the bank's.
        let stride = if self.slices > 1 {
            self.width
        } else {
            self.parts * n
        };
        let mut planes: Vec<Plane> = (0..self.parts)
            .map(|p| plane(products + u64::from(p * n) * 2, stride))
            .collect();
        for input in ins {
            planes.push(match input.kind() {
                Input::Tile => {
                    let ld = u32::try_from(input.row_stride).unwrap_or(0);
                    if self.slices > 1 && ld != n {
                        return Err(refuse(
                            OP,
                            "a tile input to a sliced row frame is not one contiguous plane",
                        ));
                    }
                    plane(input.base, if self.slices > 1 { self.width } else { ld })
                }
                Input::Column | Input::One | Input::Row => {
                    Plane::shared(&Tensor::new(input.base, 1, self.width, Dtype::Bf16))
                }
            });
        }
        let results: Vec<Plane> = outs
            .iter()
            .map(|&out| plane(out, if self.slices > 1 { self.width } else { n }))
            .collect();
        self.kernel.fire(ctx, &planes, &results, rows)
    }
}

/// The most rows the few-rows frame holds.
pub const ROWS_M: u32 = 16;

#[cfg(feature = "cuda")]
fn count(x: u32) -> Result<i32, Error> {
    i32::try_from(x).map_err(|_| refuse(OP, format!("an extent of {x} is past an i32")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hand_rules_bodies_are_epilogues_the_frames_take() {
        let store = Epilogue::store();
        assert_eq!(
            (store.parts(), store.inputs(), store.results()),
            (1, &[][..], 1)
        );
        assert_eq!(Epilogue::bias().inputs(), &[Input::Column]);
        assert_eq!(Epilogue::geglu_tanh().parts(), 2);
        let cap = Epilogue::softcap(30.0).expect("a positive cap");
        assert!(std::ptr::eq(cap, Epilogue::softcap(30.0).expect("again")));
        assert!(Epilogue::softcap(0.0).is_err());
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
        let bias = Epilogue::bias();
        assert!(bias.planes(&[In::row(64)], &[128], 8).is_err());
        assert!(bias.planes(&[], &[128], 8).is_err());
        let words = bias.planes(&[In::column(64)], &[128], 8).expect("a bias");
        // One input, one unread, one result, the row's stride.
        assert_eq!(words, vec![64, 0, 1, 0, 0, 0, 128, 8]);
    }
}
