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
use crate::region::{Op, Region, RegionError, Shape};
use crate::skeleton::round_bf16;

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
    /// One value a row `m` (a row's scale).
    Row,
    /// One value for the whole tile (a layer's scale).
    One,
}

impl Input {
    /// The shape its parameter has in the region: a tile row's elements or
    /// one value a tile row.
    #[must_use]
    pub const fn shape(self) -> Shape {
        match self {
            Input::Tile | Input::Column => Shape::Row,
            Input::Row | Input::One => Shape::Scalar,
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

/// A contraction with its prologue and epilogue.
#[derive(Clone, Debug, PartialEq)]
pub struct Anchored {
    pub contraction: Contraction,
    pub prologue: Option<Pointwise>,
    pub epilogue: Option<Pointwise>,
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
    /// A prologue that lands other than one operand, or an epilogue that
    /// lands nothing.
    Results { space: Space, results: usize },
    /// A result that is not an f32 element of the tile.
    Result { space: Space, result: u32 },
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
            AnchorError::Results { space, results } => {
                write!(f, "the {space} region lands {results} result(s)")
            }
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
                // A sum across the row or its heads, and a gather within it,
                // read other columns than their own.
                Op::Reduce(..) | Op::ReduceHeads(..) | Op::Peer(..) => false,
            };
            if !pointwise {
                return Err(AnchorError::Across { space, op: at });
            }
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

impl Anchored {
    /// The plain contraction: no prologue, the products stored as they are.
    #[must_use]
    pub const fn plain(contraction: Contraction) -> Anchored {
        Anchored {
            contraction,
            prologue: None,
            epilogue: None,
        }
    }

    /// Whether this is an anchored region: a contraction of some extent,
    /// each region pointwise over its tile ([`Space`]) and of the
    /// parameters its tile and inputs give.
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
        Ok(())
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
) -> Result<Vec<Arg>, EvalError> {
    if given.len() != inputs.len() {
        let input = u32::try_from(given.len().min(inputs.len())).unwrap_or(u32::MAX);
        return Err(EvalError::Given { space, input });
    }
    (0u32..)
        .zip(inputs.iter().zip(given))
        .map(|(input, (kind, values))| match (kind, values) {
            (Input::Tile, Given::Tile(v)) if v.len() == rows * inner => {
                Ok(Arg::Row(v[row * inner..(row + 1) * inner].to_vec()))
            }
            (Input::Column, Given::Column(v)) if v.len() == inner => Ok(Arg::Row(v.clone())),
            (Input::Row, Given::Row(v)) if v.len() == rows => Ok(Arg::Scalar(v[row])),
            (Input::One, Given::One(x)) => Ok(Arg::Scalar(*x)),
            _ => Err(EvalError::Given { space, input }),
        })
        .collect()
}

/// `anchored` over `m` rows: `a` the operand (`m x k`), `w` the weight
/// (`parts · n x k`), each row-major, and each region's inputs. Its results
/// come back each `m x n`, row-major, as the declared roundings land them.
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
            args.extend(row_args(Space::Operand, &pro.inputs, prologue, m, k, row)?);
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
        return Ok(products);
    };
    let mut out = vec![vec![0f32; m * n]; epi.region.results.len()];
    for row in 0..m {
        let mut args: Vec<Arg> = products
            .iter()
            .map(|plane| Arg::Row(plane[row * n..(row + 1) * n].to_vec()))
            .collect();
        args.extend(row_args(Space::Product, &epi.inputs, epilogue, m, n, row)?);
        let landed = interp::eval(&epi.region, &args).map_err(EvalError::Interp)?;
        for (plane, values) in out.iter_mut().zip(landed) {
            plane[row * n..(row + 1) * n].copy_from_slice(&values);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::{Binary, Literal, Reduce, Unary};

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
        let n = 16;
        // y / Σₙ y: a softmax's denominator, a function of the whole row.
        let region = Region {
            width: n,
            params: vec![Shape::Row],
            ops: vec![
                Op::Reduce(Reduce::Sum, 0),    // 1
                Op::Binary(Binary::Div, 0, 1), // 2
            ],
            results: vec![2],
            rounds: vec![],
        };
        let anchored = Anchored {
            epilogue: Some(Pointwise {
                region,
                inputs: vec![],
            }),
            ..Anchored::plain(contraction(n, 32, Layout::Dense))
        };
        assert_eq!(
            anchored.verify(),
            Err(AnchorError::Across {
                space: Space::Product,
                op: 0
            })
        );
        let why = eval(&anchored, 1, &[0.0; 32], &[0.0; 16 * 32], &[], &[]);
        assert!(matches!(
            why,
            Err(EvalError::Anchor(AnchorError::Across { .. }))
        ));
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
