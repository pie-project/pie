//! What a cut is, as data: every staged array (`<name>.npy`) the plan's bindings place, and how
//! each element of it follows from the model's params (named as the trace names them). A family
//! states its layout here ([`super::super::family::Family::tiles`]); [`super::cut`] alone turns
//! the statement into arrays, so no family cuts anything itself.
//!
//! The forms are few: a weight PE grid's tiles (a matrix's rows or columns gathered a column, its
//! hidden dims split a row), gathered vectors, constants, and the tables a block computes from
//! params rather than copies (RoPE's cos / sin, the head's vocabulary tiles, Gemma's per-layer
//! inputs, MXFP4 experts as E5M2 pairs). An index of `-1` is a zero.

/// A cut: its arrays, made in order.
#[derive(Clone, Debug, Default)]
pub struct Spec {
    pub files: Vec<File>,
}

/// One staged array, zero where no fill writes.
#[derive(Clone, Debug)]
pub struct File {
    pub name: String,
    pub shape: Vec<usize>,
    pub elem: Elem,
    pub fills: Vec<Fill>,
}

/// An array's element type: a value is rounded to it when it is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    /// IEEE half bits, rounded to nearest even (numpy's `astype(float16)`).
    F16,
    /// bfloat16 bits, truncated (a bf16 param's values come back as they were stored).
    Bf16,
    F32,
    /// Bits as a fill makes them (the E5M2 pairs).
    U16,
}

/// Params stacked along their first axis (each `[rows][width]`, the same width).
pub type Src = Vec<String>;

/// How a gathered value is applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Map {
    Id,
    /// `1 + w` (a norm's unit-offset gamma).
    PlusOne,
    /// `w / d` in f32.
    Div(f32),
    /// `(1 + w) / d` in f32.
    PlusOneDiv(f32),
}

#[derive(Clone, Debug)]
pub enum Fill {
    Grid(Grid),
    Gather(Gather),
    /// Every value of `src`, in order, from element `at`.
    Copy {
        at: usize,
        src: String,
        map: Map,
    },
    /// `value` at each of `at`.
    Const {
        at: Vec<usize>,
        value: f32,
    },
    Rope(Rope),
    Head(Head),
    Ple(Ple),
    Experts(Experts),
    ExpertScales(ExpertScales),
}

/// A weight PE grid, the array `[rows][cols][width]`: PE (r, c) holds its parts' tiles one after
/// another. Row r holds the hidden dims `rows[r].0 ..` (`rows[r].1` of them; a dim past `hidden`
/// is zero).
#[derive(Clone, Debug)]
pub struct Grid {
    pub hidden: usize,
    pub rows: Vec<(usize, usize)>,
    pub parts: Vec<Part>,
}

#[derive(Clone, Debug)]
pub enum Part {
    /// `W [out][hidden]`: PE (r, c)'s tile `[dim][out]`, the outputs `outs[c]`.
    In { src: Src, outs: Vec<Vec<i64>> },
    /// `W [hidden][in]`: PE (r, c)'s tile `[in][dim]`, the inputs `ins[c]`.
    Out { src: Src, ins: Vec<Vec<i64>> },
}

/// `src`'s values at `idx` (flat), from element `at`.
#[derive(Clone, Debug)]
pub struct Gather {
    pub at: usize,
    pub src: String,
    pub idx: Vec<i64>,
    pub map: Map,
}

/// RoPE's tables, `[col][kind][level][digit][cos | sin][pair]`: a position is `levels` digits of
/// base `radix`, digit `d` of level `lv` turning pair `i` by `d radix^(levels - 1 - lv) f_i`
/// (`freqs[col][kind][i]`; 0: no turn); level 0's cos and sin carry `scale0` (YaRN's attention
/// factor).
#[derive(Clone, Debug)]
pub struct Rope {
    pub freqs: Vec<Vec<Vec<f64>>>,
    pub levels: usize,
    pub radix: usize,
    pub scale0: f64,
}

/// The head's tiles of `src` (`[vocab][hidden]`): vocabulary group g holds the ids
/// `g per .. (g + 1) per`, dims group j the dims `dims[j]`; past the vocabulary a row is the last
/// real row (`copy_last`) or zero. The array is `[group][dims group][..]` (`[dims group][group]`
/// where `dims_major`), a tile `[dim][row]` (`[row][dim]` where `row_major`), each value times
/// `scale`.
#[derive(Clone, Debug)]
pub struct Head {
    pub src: String,
    pub vocab: usize,
    pub groups: usize,
    pub per: usize,
    pub dims: Vec<Vec<i64>>,
    pub dims_major: bool,
    pub row_major: bool,
    pub copy_last: bool,
    pub scale: f32,
}

/// Gemma's per-layer inputs, for every token in `tokens` (a storage column's rows `k`, `nt` tokens
/// a row): `(RMSNorm(W e) n + T_l[t] sqrt(P)) / sqrt(2)` of each layer in `layers` (`W e` the
/// layer's `P` rows of the projection of `e`'s embedding), computed in f64 and rounded once to
/// f16. The array is `[row][P][nt layers.len()]`: row k's token `k nt + j`, layer `layers[i]` at
/// `j layers.len() + i`.
#[derive(Clone, Debug)]
pub struct Ple {
    pub embed: String,
    /// `[layers P][hidden]`.
    pub proj: String,
    pub norm: String,
    /// Each layer's table, `[vocab][P]`.
    pub tables: Vec<String>,
    pub width: usize,
    pub vocab: usize,
    pub rows: (usize, usize),
    pub nt: usize,
    pub layers: Vec<usize>,
    pub eps: f32,
}

/// One MXFP4 expert bank's legs: gate | up rows `[2 I][G][16]` codes, down `[H][G][16]`, with
/// their E8M0 scales.
#[derive(Clone, Debug)]
pub struct Bank {
    pub gate_up: String,
    pub gate_up_scales: String,
    pub down: String,
    pub down_scales: String,
}

/// A unit's experts as E5M2 pairs, `[rows][cols][layers experts (gate|up, down)]`: each expert's
/// weights times `2^-P` (P its leg's largest scale) must be E5M2 exactly, or the cut fails. PE
/// (r, c): the gate | up tile `[Hc / 2][2 Ic]` (dim i's byte high, `i + Hc / 2`'s low), then down's
/// `[Ic / 2][Hc]` (row j high, `j + Ic / 2` low).
#[derive(Clone, Debug)]
pub struct Experts {
    pub banks: Vec<Bank>,
    pub experts: usize,
    pub hidden: usize,
    pub inter: usize,
    pub rows: usize,
    pub cols: usize,
}

/// The experts' `2^P`s (gate | up, down) a layer and expert, repeated on each of `cols` rows.
#[derive(Clone, Debug)]
pub struct ExpertScales {
    pub banks: Vec<Bank>,
    pub experts: usize,
    pub cols: usize,
}

impl File {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, elem: Elem) -> File {
        File {
            name: name.into(),
            shape,
            elem,
            fills: Vec::new(),
        }
    }

    pub fn with(mut self, fill: Fill) -> File {
        self.fills.push(fill);
        self
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Part {
    /// Words a PE holds of this part, its row `dims` dims.
    pub fn width(&self, dims: usize) -> usize {
        match self {
            Part::In { outs: idx, .. } | Part::Out { ins: idx, .. } => {
                dims * idx.first().map_or(0, Vec::len)
            }
        }
    }
}

impl Spec {
    pub fn push(&mut self, file: File) {
        self.files.push(file);
    }
}

/// `idx` padded with zeros (`-1`) to `n`.
pub fn padded(idx: impl IntoIterator<Item = i64>, n: usize) -> Vec<i64> {
    let mut v: Vec<i64> = idx.into_iter().collect();
    v.resize(n.max(v.len()), -1);
    v
}

/// `0 .. n` as indices.
pub fn range(lo: usize, n: usize) -> Vec<i64> {
    (lo..lo + n).map(|i| i as i64).collect()
}
