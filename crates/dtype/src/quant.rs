//! QNF — a quantization format as a TERM, not as a name.
//!
//! `Dtype` names an element. It cannot name what GPTQ, MLX, Q4_K and MXFP4
//! actually are, because those are not elements: each is a small ALGEBRA over
//! elements — some codes, a group width, a gain that is itself stored in some
//! element, sometimes an offset that is itself stored in some element, and in
//! the k-quants a gain that is *itself quantized*. The tree learned this the
//! expensive way. `Dtype::MlxU4`, `MlxU8` and `MlxU4G32` are three variants
//! that differ in two integers; `Mxfp4` hides a group width, a scale element
//! and a packing inside one word; and every consumer that wanted a number back
//! out of them — a plane's bytes, a group's stride, whether a kernel can serve
//! it — re-derived that number from a table that only it could get wrong.
//!
//! **A FORMAT IS A VALUE HERE.** Write the algebra down and the numbers stop
//! being lore: bits per weight falls out of the term, the plane widths fall out
//! of the term, the legal k-split falls out of the term, and dispatch is a
//! match on a `Copy` value rather than a string compare against a vendor name.
//!
//! Three faces of the same thing, in the order you meet them:
//!
//! * [`Term`] is the truth — recursive, boxed, behind the `alloc` feature,
//!   because it is the canonicalizer's working type and a `no_std` guest has
//!   no business allocating one.
//! * [`Sig`] is the projection to depth two — `Copy`, `Eq`, `Hash`, const-
//!   patternable, and enough to spell every format anyone has shipped. This is
//!   what dispatch tables key on and what goes on the wire.
//! * The MANGLED SPELLING is the name — `"g64_u4_bf16_b_bf16"`, a fixed-arity
//!   prefix walk of the term with `_` between tokens. It is the only name.
//!
//! The ruling behind that last point is worth stating plainly, because it is
//! what keeps this module from growing a synonym table: **vendor names are not
//! identifiers.** GPTQ, AWQ, HQQ and MLX name PIPELINES, not formats — AWQ and
//! GPTQ ship the identical bytes, and "MLX 4-bit" is two formats depending on
//! which tensor you are looking at. Those words are legal in a doc comment and
//! legal as input to a canonicalizer that reads someone else's config; they
//! are not legal as a `const` name. The consts below are named by their own
//! spelling, uppercased, and the only synonyms are the handful of names that
//! belong to a PUBLISHED SPEC rather than to a toolchain: [`MXFP4`], [`NVFP4`],
//! [`Q4_0`], [`Q8_0`], [`Q4_K`], [`Q6_K`].
//!
//! # The grammar
//!
//! ```text
//! term   := leaf | group '_' leaf '_' term '_' offset
//! offset := 'n' | 'z' '_' term | 'b' '_' term
//! ```
//!
//! Fixed arity, so no parentheses are needed and the walk is unambiguous. The
//! elem slot is always a leaf: codes are stored, never computed. `z` is the
//! integer zero-point family (`gain · (codes − offset)`) and `b` is the real
//! bias family (`gain · codes + offset`); `n` is no offset at all.
//!
//! # Two normalizations that make the spelling unique
//!
//! **Signed codes are EXCESS-BINARY.** [`Leaf::I`] means a stored code `c`
//! decodes as `c − 2^(b−1)` — ggml's `Q4_0` convention and `Int4B8`'s. A
//! two's-complement source is the same information with the sign bit relabeled,
//! which a repack does bit-exactly, so one spelling suffices and the dispatch
//! table does not carry two arms that differ by a XOR.
//!
//! **Offsets always ADD.** A stored MINIMUM (`Q4_1`, the k-quant mins, MLX's
//! `biases` when a converter writes them negated) is turned into an addition at
//! repack by negating the float factor. That is bit-exact for every float type
//! in the tree, and it means [`Offset::Post`] never needs a sign flag.
//!
//! Together with "elem is a leaf" and "there are no constant offsets" (a
//! constant zero-point folds into [`Leaf::I`], which is why `Term` has no
//! constant node at all), those are the canonical rules. Three of the four are
//! enforced by the types; the fourth is enforced at repack, upstream of here.

use core::fmt;

// ─────────────────────────────────────────────────────────────────────────
// Leaves
// ─────────────────────────────────────────────────────────────────────────

/// A stored element — the only thing at the bottom of a term.
///
/// A leaf is what a plane's bytes literally hold. Its [`rate`](Leaf::rate) is a
/// PAIR and not a bit count, because two of these are not whole bits per
/// element: [`T3`](Leaf::T3) packs five ternary digits into a byte, and a
/// future codebook could pack anything. Carrying `(bits, elems)` and dividing
/// once at the end is what keeps `bpw` honest instead of rounding twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Leaf {
    /// Unsigned code of `b` bits: the stored code IS the value.
    U(u8),
    /// Signed code of `b` bits in **excess-binary**: a stored code `c` decodes
    /// as `c − 2^(b−1)`. This is ggml's convention (`Q4_0`'s nibble `8` is
    /// zero) and `Int4B8`'s. Two's-complement sources are relabeled at repack.
    I(u8),
    /// A float with `e` exponent bits and `m` mantissa bits.
    ///
    /// `m == 0` is the EXPONENT-ONLY family and carries no sign bit, so
    /// `e8m0` is eight bits and not nine — OCP defines the microscaling scale
    /// type as an unsigned power of two, and a sign on a scale that only ever
    /// multiplies a signed code would be a second spelling of the same value.
    /// Every other `e{e}m{m}` is `1 + e + m` bits, sign included: `e2m1` is
    /// four, `e4m3` is eight.
    E {
        /// Exponent bits.
        e: u8,
        /// Mantissa bits; `0` means exponent-only, hence unsigned.
        m: u8,
    },
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// bfloat16.
    Bf16,
    /// bitsandbytes' NF4: a fixed sixteen-entry normal-float table, four bits
    /// per code. A codebook whose table is a constant of the format, which is
    /// why it is a leaf of its own rather than a [`Cb`](Leaf::Cb).
    Nf4,
    /// A ternary digit at a fixed rate of eight bits per five elements —
    /// `3^5 = 243 ≤ 256`, so five trits pack into a byte with room to spare.
    /// BitNet-class weights.
    T3,
    /// A codebook code of registry index `n`: `n` says which table, and the
    /// table says how wide a code is and what it decodes to.
    ///
    /// The registry does not exist yet, so [`rate`](Leaf::rate) returns `None`
    /// here and every width answer downstream is `None` with it. That is
    /// deliberate: a codebook's rate is a fact about a TABLE, not about this
    /// enum, and a guessed number here would be exactly the silent-wrong-number
    /// class of bug this module was built to end.
    Cb(u16),
}

impl Leaf {
    /// Storage rate as `(bits, elements)` — this many bits carry that many
    /// elements. `U(4)` is `(4, 1)`, `T3` is `(8, 5)`, `F16` is `(16, 1)`,
    /// `E { e, m }` is `(1 + e + m, 1)` unless `m == 0`, which is `(e, 1)`.
    ///
    /// `None` for [`Cb`](Leaf::Cb): see its docs.
    #[must_use]
    pub const fn rate(self) -> Option<(u32, u32)> {
        match self {
            Self::U(b) | Self::I(b) => Some((b as u32, 1)),
            Self::E { e, m } => Some((
                if m == 0 { e as u32 } else { 1 + e as u32 + m as u32 },
                1,
            )),
            Self::F32 => Some((32, 1)),
            Self::F16 | Self::Bf16 => Some((16, 1)),
            Self::Nf4 => Some((4, 1)),
            Self::T3 => Some((8, 5)),
            Self::Cb(_) => None,
        }
    }
}

impl fmt::Display for Leaf {
    /// The leaf's mangling token.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::U(b) => write!(f, "u{b}"),
            Self::I(b) => write!(f, "i{b}"),
            Self::E { e, m } => write!(f, "e{e}m{m}"),
            Self::F32 => f.write_str("f32"),
            Self::F16 => f.write_str("f16"),
            Self::Bf16 => f.write_str("bf16"),
            Self::Nf4 => f.write_str("nf4"),
            Self::T3 => f.write_str("t3"),
            Self::Cb(n) => write!(f, "cb{n}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Groups
// ─────────────────────────────────────────────────────────────────────────

/// How many elements share one factor.
///
/// **GROUPS RUN ALONG THE REDUCTION AXIS.** A weight is `[n, k]` with `k` the
/// contracted dimension, and a group is a run of `k`-neighbours, because that
/// is the axis a dot product walks: a kernel accumulates a group, applies its
/// factor once, and moves on. A group that ran along `n` would have to be
/// applied to a partial sum that never exists.
///
/// [`Tile`](Group::Tile) is the one that touches both axes, and its pair reads
/// `(rows, cols)` over the `[n, k]` rectangle — `g128x128` is DeepSeek's block
/// scaling, one factor per 128 rows by 128 columns. Its `cols` is the k extent
/// and therefore the only half that a per-row width calculation sees; its
/// `rows` says how many weight rows SHARE that factor, which is what makes a
/// tile cheaper per row than an `N` of the same k extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    /// A run of `n` elements along k. Token `g{n}`.
    N(u32),
    /// A `(rows, cols)` block of the `[n, k]` rectangle. Token `g{r}x{c}`.
    Tile(u32, u32),
    /// One factor per output row — the whole k axis. Token `gr`.
    Row,
    /// One factor for the whole tensor. Token `gt`.
    Tensor,
}

impl Group {
    /// How many elements of the k axis this group spans, given the ambient
    /// count at this level of the term. [`Row`](Group::Row) and
    /// [`Tensor`](Group::Tensor) span all of it.
    const fn k_extent(self, ambient: u64) -> u64 {
        match self {
            Self::N(n) => n as u64,
            Self::Tile(_, c) => c as u64,
            Self::Row | Self::Tensor => ambient,
        }
    }

    /// The group's own k extent, or `None` when it spans whatever it is given
    /// — which is what makes a row unsplittable. See [`Sig::quantum`].
    const fn k_span(self) -> Option<u32> {
        match self {
            Self::N(n) => Some(n),
            Self::Tile(_, c) => Some(c),
            Self::Row | Self::Tensor => None,
        }
    }

    /// How many copies of a factor at this group a single ROW pays for, given
    /// the ambient count. A tile is divided by both extents because its `rows`
    /// weight rows share one factor; a tensor-wide factor is amortized over a
    /// row count this function does not know and is charged as zero — see
    /// [`Sig::bpw`].
    fn shares(self, ambient: f64) -> f64 {
        match self {
            Self::N(n) => ambient / f64::from(n),
            Self::Tile(r, c) => ambient / f64::from(c) / f64::from(r),
            Self::Row => 1.0,
            Self::Tensor => 0.0,
        }
    }
}

impl fmt::Display for Group {
    /// The group's mangling token.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::N(n) => write!(f, "g{n}"),
            Self::Tile(r, c) => write!(f, "g{r}x{c}"),
            Self::Row => f.write_str("gr"),
            Self::Tensor => f.write_str("gt"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Sig — the depth-two projection
// ─────────────────────────────────────────────────────────────────────────

/// A factor at the projected depth: a leaf, or one more level of quantization
/// whose own factors are leaves.
///
/// This is where the depth-two ceiling lives. `Q1` is the k-quant shape — the
/// scales are themselves grouped and scaled, and that is where every shipped
/// format stops. `Q1z` is the same with an integer zero-point on the inner
/// codes, for a scale plane that is itself affine-quantized.
///
/// [`Nil`](Sub::Nil) is the ABSENT factor. It has no spelling and the parser
/// never produces one; it exists so construction code can leave a slot empty
/// and so `OffSub::Pre(Nil)` has a name to be rejected under. A `Sig` carrying
/// one is not canonical — see [`Sig::is_canonical`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sub {
    /// No factor here. Non-canonical inside a `Sig`; has no mangled spelling.
    Nil,
    /// A stored factor.
    L(Leaf),
    /// A quantized factor: `elem` codes grouped by `Group`, scaled by a leaf,
    /// no offset. Token shape `g_elem_gain_n`.
    Q1(Group, Leaf, Leaf),
    /// A quantized factor with an integer zero-point: `g_elem_gain_z_zero`.
    Q1z(Group, Leaf, Leaf, Leaf),
}

/// The offset slot of a [`Sig`], carrying the Pre/Post distinction at the
/// projected depth.
///
/// The distinction is NOT cosmetic and cannot be normalized away.
/// `Pre` subtracts in the CODE domain — `gain · (codes − z)` — where `z` is an
/// integer of the same width as the codes for GPTQ/AWQ, but a real number for
/// HQQ, which is exactly why the two cannot be collapsed: HQQ's real
/// pre-scale zero breaks any inference that a `Pre` offset shares the elem's
/// dtype. `Post` adds in the VALUE domain — `gain · codes + b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OffSub {
    /// No offset: `gain · codes`. Token `n`.
    Nil,
    /// Subtracted from the codes before the gain applies. Token `z_…`.
    Pre(Sub),
    /// Added to the value after the gain applies. Token `b_…`.
    Post(Sub),
}

impl OffSub {
    /// The factor inside, if there is one.
    #[must_use]
    pub const fn factor(self) -> Option<Sub> {
        match self {
            Self::Nil => None,
            Self::Pre(s) | Self::Post(s) => Some(s),
        }
    }
}

/// A quantization format, projected to depth two: `Copy`, `Eq`, `Hash`, and
/// usable as a `const` pattern.
///
/// This is the dispatch key and the wire form. Everything shipped by anyone
/// fits — the deepest real format is a k-quant, which is exactly depth two —
/// and the ceiling is what buys the `Copy`: a `Sig` is a handful of bytes with
/// no indirection, so a kernel table can `match` it and a plan can carry it
/// without touching an allocator.
///
/// Build one from its spelling with [`sig`], which is a `const fn`: a wrong
/// spelling is a BUILD ERROR, not a runtime `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sig {
    /// Not quantized: one stored element per weight.
    Plain(Leaf),
    /// Grouped codes with a gain and an optional offset.
    Q {
        /// How many codes share one gain (and one offset, if any).
        g: Group,
        /// What a code is. Always a leaf: codes are stored, never computed.
        elem: Leaf,
        /// The multiplicative factor.
        gain: Sub,
        /// The additive or subtractive factor, and which of the two.
        offset: OffSub,
    },
}

/// The minimal unit a row may be split into along k, from [`Sig::quantum`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quantum {
    /// A split is legal at any multiple of this many elements.
    Elems(u32),
    /// The row does not split: some factor on the tree spans all of k.
    WholeRow,
}

impl Quantum {
    /// The quantum in elements, resolving [`WholeRow`](Quantum::WholeRow)
    /// against a known row width.
    #[must_use]
    pub const fn elems(self, k: u32) -> u32 {
        match self {
            Self::Elems(n) => n,
            Self::WholeRow => k,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Mangling — writing
// ─────────────────────────────────────────────────────────────────────────

impl fmt::Display for Sub {
    /// The factor's mangled spelling. [`Sub::Nil`] has none and fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Nil => Err(fmt::Error),
            Self::L(l) => write!(f, "{l}"),
            Self::Q1(g, e, gain) => write!(f, "{g}_{e}_{gain}_n"),
            Self::Q1z(g, e, gain, z) => write!(f, "{g}_{e}_{gain}_z_{z}"),
        }
    }
}

impl fmt::Display for OffSub {
    /// The offset slot's mangled spelling.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Nil => f.write_str("n"),
            Self::Pre(s) => write!(f, "z_{s}"),
            Self::Post(s) => write!(f, "b_{s}"),
        }
    }
}

impl fmt::Display for Sig {
    /// The mangled spelling — the format's only name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Plain(l) => write!(f, "{l}"),
            Self::Q {
                g,
                elem,
                gain,
                offset,
            } => write!(f, "{g}_{elem}_{gain}_{offset}"),
        }
    }
}

/// A mangled spelling in a fixed buffer — `no_std`'s answer to `String`.
///
/// The capacity is not a guess. The widest `Sig` the types admit is a `Q`
/// whose group is `g4294967295x4294967295` (22 bytes), whose elem is an
/// `e255m255` or `cb65535` (8), and whose gain and offset are both `Q1z` of
/// the same worst case (51 each, plus `b_`): 137 bytes. The buffer is rounded
/// up from there, and [`fmt::Write`] on it returns an error rather than
/// truncating, so an overflow can never become a silently wrong name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mangled {
    bytes: [u8; Self::CAPACITY],
    len: usize,
}

impl Mangled {
    /// Bytes the buffer holds. See the type's docs for where 160 comes from.
    pub const CAPACITY: usize = 160;

    /// An empty buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes: [0; Self::CAPACITY],
            len: 0,
        }
    }

    /// What has been written, as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.bytes[..self.len]).expect("only &str was written")
    }
}

impl Default for Mangled {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Write for Mangled {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len + s.len();
        if end > Self::CAPACITY {
            return Err(fmt::Error);
        }
        self.bytes[self.len..end].copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

impl fmt::Display for Mangled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::ops::Deref for Mangled {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Mangling — parsing
// ─────────────────────────────────────────────────────────────────────────

/// Why a spelling is not a format.
///
/// Every variant is a distinct `const` panic message in [`sig`], which is the
/// point of splitting them this finely: a typo in a `const` declaration should
/// say which token was wrong at compile time, and a `const` panic cannot
/// format, so the discrimination has to live in the enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SigError {
    /// The spelling was empty.
    Empty,
    /// A token ended where another was expected — a trailing or missing `_`.
    Truncated,
    /// Tokens remained after a complete term.
    Trailing,
    /// A token that should have been a leaf was not one.
    UnknownLeaf,
    /// A token that should have been a group was not one.
    UnknownGroup,
    /// A group size of zero.
    ZeroGroup,
    /// A number was expected and the token had no digits.
    MissingNumber,
    /// A number token held something that is not a digit.
    BadNumber,
    /// A number did not fit its field.
    NumberTooLarge,
    /// A code width outside `1..=64`, or an exponent width of zero.
    BadWidth,
    /// An exponent token with no `m` — `e4` rather than `e4m3`.
    BadExpToken,
    /// The offset slot held something other than `n`, `z` or `b`.
    BadOffsetMark,
    /// The term is deeper than the depth-two projection. Parse it as a
    /// [`Term`] instead.
    TooDeep,
    /// A POST offset nested inside a factor. [`Sub`] spells the pre-family
    /// only at depth two — see [`Sub::Q1z`] — so this term, though it is
    /// depth two, has no `Sig`. Parse it as a [`Term`].
    NestedPost,
}

impl SigError {
    /// A one-line reason, for a runtime caller that wants to log one.
    #[must_use]
    pub const fn why(self) -> &'static str {
        match self {
            Self::Empty => "empty spelling",
            Self::Truncated => "a token is missing",
            Self::Trailing => "tokens after a complete term",
            Self::UnknownLeaf => "not a leaf token",
            Self::UnknownGroup => "not a group token",
            Self::ZeroGroup => "a group of zero elements",
            Self::MissingNumber => "a number token with no digits",
            Self::BadNumber => "a number token with a non-digit",
            Self::NumberTooLarge => "a number too large for its field",
            Self::BadWidth => "a code width outside 1..=64",
            Self::BadExpToken => "an exponent token with no m",
            Self::BadOffsetMark => "the offset slot is not n, z or b",
            Self::TooDeep => "deeper than the depth-two projection",
            Self::NestedPost => "a post offset nested inside a factor",
        }
    }
}

impl fmt::Display for SigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.why())
    }
}

/// `?` for `const fn`, which has no `Try`.
macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Err(e),
        }
    };
}

/// Index just past the token starting at `i` — the next `_`, or the end.
const fn tok_end(b: &[u8], i: usize) -> usize {
    let mut j = i;
    while j < b.len() && b[j] != b'_' {
        j += 1;
    }
    j
}

/// Step over the `_` that must follow a token.
const fn sep(b: &[u8], i: usize) -> Result<usize, SigError> {
    if i < b.len() && b[i] == b'_' {
        Ok(i + 1)
    } else {
        Err(SigError::Truncated)
    }
}

/// Is `b[i..j]` exactly `lit`?
const fn is(b: &[u8], i: usize, j: usize, lit: &[u8]) -> bool {
    if j - i != lit.len() {
        return false;
    }
    let mut k = 0;
    while k < lit.len() {
        if b[i + k] != lit[k] {
            return false;
        }
        k += 1;
    }
    true
}

/// A decimal number spanning `b[i..j]`.
const fn digits(b: &[u8], i: usize, j: usize) -> Result<u32, SigError> {
    if i >= j {
        return Err(SigError::MissingNumber);
    }
    let mut v: u64 = 0;
    let mut k = i;
    while k < j {
        let c = b[k];
        if c < b'0' || c > b'9' {
            return Err(SigError::BadNumber);
        }
        v = v * 10 + (c - b'0') as u64;
        if v > u32::MAX as u64 {
            return Err(SigError::NumberTooLarge);
        }
        k += 1;
    }
    Ok(v as u32)
}

/// The leaf spelled by `b[i..j]`.
const fn leaf_of(b: &[u8], i: usize, j: usize) -> Result<Leaf, SigError> {
    if is(b, i, j, b"f32") {
        return Ok(Leaf::F32);
    }
    if is(b, i, j, b"f16") {
        return Ok(Leaf::F16);
    }
    if is(b, i, j, b"bf16") {
        return Ok(Leaf::Bf16);
    }
    if is(b, i, j, b"nf4") {
        return Ok(Leaf::Nf4);
    }
    if is(b, i, j, b"t3") {
        return Ok(Leaf::T3);
    }
    if i >= j {
        return Err(SigError::UnknownLeaf);
    }
    let head = b[i];
    if head == b'u' || head == b'i' {
        let w = tri!(digits(b, i + 1, j));
        if w == 0 || w > 64 {
            return Err(SigError::BadWidth);
        }
        return Ok(if head == b'u' {
            Leaf::U(w as u8)
        } else {
            Leaf::I(w as u8)
        });
    }
    if head == b'e' {
        let mut k = i + 1;
        while k < j && b[k] != b'm' {
            k += 1;
        }
        if k >= j {
            return Err(SigError::BadExpToken);
        }
        let e = tri!(digits(b, i + 1, k));
        let m = tri!(digits(b, k + 1, j));
        if e == 0 || e > 255 || m > 255 {
            return Err(SigError::BadWidth);
        }
        return Ok(Leaf::E {
            e: e as u8,
            m: m as u8,
        });
    }
    if j - i > 2 && head == b'c' && b[i + 1] == b'b' {
        let n = tri!(digits(b, i + 2, j));
        if n > u16::MAX as u32 {
            return Err(SigError::NumberTooLarge);
        }
        return Ok(Leaf::Cb(n as u16));
    }
    Err(SigError::UnknownLeaf)
}

/// The group spelled by `b[i..j]`.
const fn group_of(b: &[u8], i: usize, j: usize) -> Result<Group, SigError> {
    if is(b, i, j, b"gr") {
        return Ok(Group::Row);
    }
    if is(b, i, j, b"gt") {
        return Ok(Group::Tensor);
    }
    if i >= j || b[i] != b'g' {
        return Err(SigError::UnknownGroup);
    }
    let mut k = i + 1;
    while k < j && b[k] != b'x' {
        k += 1;
    }
    if k < j {
        let r = tri!(digits(b, i + 1, k));
        let c = tri!(digits(b, k + 1, j));
        if r == 0 || c == 0 {
            return Err(SigError::ZeroGroup);
        }
        return Ok(Group::Tile(r, c));
    }
    let n = tri!(digits(b, i + 1, j));
    if n == 0 {
        return Err(SigError::ZeroGroup);
    }
    Ok(Group::N(n))
}

/// Does the token at `i` open a group? Groups are the only tokens starting
/// with `g`, which is what makes the fixed-arity walk unambiguous.
const fn opens_group(b: &[u8], i: usize, j: usize) -> bool {
    j > i && b[i] == b'g'
}

/// A factor at the projected depth, and where it ends.
const fn sub_at(b: &[u8], i: usize) -> Result<(Sub, usize), SigError> {
    let j = tok_end(b, i);
    if !opens_group(b, i, j) {
        return Ok((Sub::L(tri!(leaf_of(b, i, j))), j));
    }
    let g = tri!(group_of(b, i, j));
    let i = tri!(sep(b, j));
    let j = tok_end(b, i);
    let elem = tri!(leaf_of(b, i, j));
    let i = tri!(sep(b, j));
    let j = tok_end(b, i);
    if opens_group(b, i, j) {
        return Err(SigError::TooDeep);
    }
    let gain = tri!(leaf_of(b, i, j));
    let i = tri!(sep(b, j));
    let j = tok_end(b, i);
    if is(b, i, j, b"n") {
        return Ok((Sub::Q1(g, elem, gain), j));
    }
    if is(b, i, j, b"b") {
        return Err(SigError::NestedPost);
    }
    if !is(b, i, j, b"z") {
        return Err(SigError::BadOffsetMark);
    }
    let i = tri!(sep(b, j));
    let j = tok_end(b, i);
    if opens_group(b, i, j) {
        return Err(SigError::TooDeep);
    }
    let z = tri!(leaf_of(b, i, j));
    Ok((Sub::Q1z(g, elem, gain, z), j))
}

/// The offset slot, and where it ends.
const fn off_at(b: &[u8], i: usize) -> Result<(OffSub, usize), SigError> {
    let j = tok_end(b, i);
    if is(b, i, j, b"n") {
        return Ok((OffSub::Nil, j));
    }
    let pre = if is(b, i, j, b"z") {
        true
    } else if is(b, i, j, b"b") {
        false
    } else {
        return Err(SigError::BadOffsetMark);
    };
    let i = tri!(sep(b, j));
    let (s, e) = tri!(sub_at(b, i));
    Ok((if pre { OffSub::Pre(s) } else { OffSub::Post(s) }, e))
}

impl Sig {
    /// Parse a mangled spelling, fallibly. `const`, so [`sig`] can build on it.
    ///
    /// # Errors
    ///
    /// [`SigError`] names which token was wrong and how.
    pub const fn parse(mangled: &str) -> Result<Self, SigError> {
        let b = mangled.as_bytes();
        if b.is_empty() {
            return Err(SigError::Empty);
        }
        let j = tok_end(b, 0);
        if !opens_group(b, 0, j) {
            let l = tri!(leaf_of(b, 0, j));
            if j != b.len() {
                return Err(SigError::Trailing);
            }
            return Ok(Self::Plain(l));
        }
        let g = tri!(group_of(b, 0, j));
        let i = tri!(sep(b, j));
        let j = tok_end(b, i);
        let elem = tri!(leaf_of(b, i, j));
        let i = tri!(sep(b, j));
        let (gain, e) = tri!(sub_at(b, i));
        let i = tri!(sep(b, e));
        let (offset, e) = tri!(off_at(b, i));
        if e != b.len() {
            return Err(SigError::Trailing);
        }
        Ok(Self::Q {
            g,
            elem,
            gain,
            offset,
        })
    }

    /// The mangled spelling, in a fixed buffer.
    ///
    /// # Panics
    ///
    /// If the `Sig` is not canonical — a [`Sub::Nil`] in a factor slot has no
    /// spelling. Check with [`is_canonical`](Sig::is_canonical) if the value
    /// did not come from a parser.
    #[must_use]
    pub fn mangle(self) -> Mangled {
        use fmt::Write as _;
        let mut out = Mangled::new();
        write!(out, "{self}").expect("a canonical Sig fits its own buffer");
        out
    }

    /// Whether every factor slot holds a factor. The other canonical rules —
    /// elem is a leaf, offsets add, there are no constant offsets — are
    /// enforced by the types, so this is the only one left to ask about.
    #[must_use]
    pub const fn is_canonical(self) -> bool {
        match self {
            Self::Plain(_) => true,
            Self::Q { gain, offset, .. } => {
                if matches!(gain, Sub::Nil) {
                    return false;
                }
                match offset {
                    OffSub::Nil => true,
                    OffSub::Pre(s) | OffSub::Post(s) => !matches!(s, Sub::Nil),
                }
            }
        }
    }
}

/// Build a [`Sig`] from its mangled spelling, at compile time.
///
/// **THIS IS THE LOAD-BEARING CONSTRUCTOR.** In a `const` context a bad
/// spelling is a build error — `const G64_U4: Sig = sig("g64_u4")` does not
/// compile, and the panic says which token was wrong. That is the whole
/// argument for a string-shaped constructor in a typed language: the string is
/// the format's NAME, it is what a checkpoint's config says, and the only way
/// a name earns the right to be a name is if writing it wrong stops the build.
///
/// # Panics
///
/// On any spelling that is not a canonical depth-two term.
#[must_use]
pub const fn sig(mangled: &str) -> Sig {
    match Sig::parse(mangled) {
        Ok(v) => v,
        Err(SigError::Empty) => panic!("qnf sig: the spelling is empty"),
        Err(SigError::Truncated) => panic!("qnf sig: a token is missing"),
        Err(SigError::Trailing) => panic!("qnf sig: tokens after a complete term"),
        Err(SigError::UnknownLeaf) => panic!("qnf sig: not a leaf token"),
        Err(SigError::UnknownGroup) => panic!("qnf sig: not a group token"),
        Err(SigError::ZeroGroup) => panic!("qnf sig: a group of zero elements"),
        Err(SigError::MissingNumber) => panic!("qnf sig: a number token with no digits"),
        Err(SigError::BadNumber) => panic!("qnf sig: a number token with a non-digit"),
        Err(SigError::NumberTooLarge) => panic!("qnf sig: a number too large for its field"),
        Err(SigError::BadWidth) => panic!("qnf sig: a code width outside 1..=64"),
        Err(SigError::BadExpToken) => panic!("qnf sig: an exponent token with no m"),
        Err(SigError::BadOffsetMark) => panic!("qnf sig: the offset slot is not n, z or b"),
        Err(SigError::TooDeep) => panic!("qnf sig: deeper than the depth-two projection"),
        Err(SigError::NestedPost) => panic!("qnf sig: a post offset nested inside a factor"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Derived widths
// ─────────────────────────────────────────────────────────────────────────

/// The most planes a depth-two `Sig` can name: the codes, plus up to three
/// leaves down the gain path (`Q1z`), plus up to three down the offset path.
///
/// The shipped formats top out at five — `Q4_K`, whose gain and offset are
/// both `Q1` — but the TYPE admits seven, and a buffer sized to the formats
/// rather than to the type is a bug waiting for the eighth format.
pub const MAX_PLANES: usize = 7;

/// Bytes per row of each plane of a leaf-per-plane container, in tree order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneWidths {
    bytes: [u32; MAX_PLANES],
    len: usize,
}

impl PlaneWidths {
    /// The widths, in tree order: the codes plane first, then the gain path's
    /// leaves, then the offset path's.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.bytes[..self.len]
    }

    /// How many planes the container has.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no planes at all. There always are; this is here so
    /// `len` does not read as a lie.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    const fn new() -> Self {
        Self {
            bytes: [0; MAX_PLANES],
            len: 0,
        }
    }

    fn push(&mut self, w: u32) {
        self.bytes[self.len] = w;
        self.len += 1;
    }
}

/// Bytes a plane of `n` elements of `l` occupies, rounded up once at the end.
fn leaf_bytes(l: Leaf, n: u64) -> Option<u32> {
    let (num, den) = l.rate()?;
    let bits = (n * u64::from(num)).div_ceil(u64::from(den));
    u32::try_from(bits.div_ceil(8)).ok()
}

/// Bits per row a factor at this level costs, given how many copies of it a
/// row pays for.
fn sub_bits(s: Sub, copies: f64) -> Option<f64> {
    let rate = |l: Leaf| l.rate().map(|(n, d)| f64::from(n) / f64::from(d));
    Some(match s {
        Sub::Nil => return None,
        Sub::L(l) => copies * rate(l)?,
        Sub::Q1(g, e, gain) => copies * rate(e)? + g.shares(copies) * rate(gain)?,
        Sub::Q1z(g, e, gain, z) => {
            copies * rate(e)? + g.shares(copies) * (rate(gain)? + rate(z)?)
        }
    })
}

/// Planes a factor at this level contributes, given its element count.
fn sub_planes(s: Sub, elems: u64, out: &mut PlaneWidths) -> Option<()> {
    match s {
        Sub::Nil => return None,
        Sub::L(l) => out.push(leaf_bytes(l, elems)?),
        Sub::Q1(g, e, gain) => {
            out.push(leaf_bytes(e, elems)?);
            out.push(leaf_bytes(gain, elems.div_ceil(g.k_extent(elems)))?);
        }
        Sub::Q1z(g, e, gain, z) => {
            out.push(leaf_bytes(e, elems)?);
            let inner = elems.div_ceil(g.k_extent(elems));
            out.push(leaf_bytes(gain, inner)?);
            out.push(leaf_bytes(z, inner)?);
        }
    }
    Some(())
}

/// Every group on the tree, in tree order, for [`Sig::quantum`].
fn groups_of(s: Sub) -> Option<Group> {
    match s {
        Sub::Nil | Sub::L(_) => None,
        Sub::Q1(g, ..) | Sub::Q1z(g, ..) => Some(g),
    }
}

impl Sig {
    /// Bits per weight for a `k`-wide row.
    ///
    /// A rate, not a container size: this is exact rational arithmetic and it
    /// does not round to the byte, because rounding belongs to
    /// [`plane_widths`](Sig::plane_widths). `k` is an argument rather than a
    /// property because [`Group::Row`] and [`Group::Tensor`] have no width of
    /// their own — a per-row scale costs `bits / k` per weight and there is no
    /// answer without `k` — while an `N` group closes over it and gives the
    /// same number at every `k` that it divides.
    ///
    /// **A TENSOR-WIDE FACTOR IS CHARGED AS ZERO.** It is one number for the
    /// whole tensor, so its share of a row depends on the ROW COUNT, which is
    /// not an argument here; charging it to a row would make NVFP4 report
    /// 4.5078 at `k = 4096` and a different number at every shape. Zero is the
    /// asymptote and the number every published table means.
    ///
    /// `None` if a leaf has no rate — see [`Leaf::Cb`] — or if `k` is zero.
    #[must_use]
    pub fn bpw(self, k: u32) -> Option<f64> {
        if k == 0 {
            return None;
        }
        let k = f64::from(k);
        let rate = |l: Leaf| l.rate().map(|(n, d)| f64::from(n) / f64::from(d));
        let bits = match self {
            Self::Plain(l) => k * rate(l)?,
            Self::Q {
                g,
                elem,
                gain,
                offset,
            } => {
                let copies = g.shares(k);
                let mut total = k * rate(elem)? + sub_bits(gain, copies)?;
                if let Some(s) = offset.factor() {
                    total += sub_bits(s, copies)?;
                }
                total
            }
        };
        Some(bits / k)
    }

    /// Bytes per row of every plane of the leaf-per-plane container, in tree
    /// order: the codes first, then the gain path's leaves, then the offset
    /// path's.
    ///
    /// A factor leaf under groups `g₁·g₂…` covers `k / (g₁·g₂…)` elements per
    /// row, rounded UP, so a factor whose group spans the row or the tensor
    /// reports one element — which is what its plane holds per row of itself.
    /// A [`Group::Tile`]'s plane has `⌈n / rows⌉` rows of its own; only its
    /// `cols` enters a per-row width.
    ///
    /// `None` if a leaf has no rate, or a width overflows `u32`.
    #[must_use]
    pub fn plane_widths(self, k: u32) -> Option<PlaneWidths> {
        let mut out = PlaneWidths::new();
        match self {
            Self::Plain(l) => out.push(leaf_bytes(l, u64::from(k))?),
            Self::Q {
                g,
                elem,
                gain,
                offset,
            } => {
                let k = u64::from(k);
                out.push(leaf_bytes(elem, k)?);
                let factors = k.div_ceil(g.k_extent(k));
                sub_planes(gain, factors, &mut out)?;
                if let Some(s) = offset.factor() {
                    sub_planes(s, factors, &mut out)?;
                }
            }
        }
        Some(out)
    }

    /// The minimal unit a row splits into along k.
    ///
    /// A split that landed mid-group would need a factor it does not own, so
    /// the answer is the lcm of every group's extent IN ELEMENTS.
    ///
    /// **IN ELEMENTS IS THE WHOLE SUBTLETY.** A nested group counts its parent
    /// in: `Q4_K`'s inner `g8` groups eight SCALES, and each scale already
    /// covers 32 weights, so its extent is 256 and not 8 — which is exactly
    /// the super-block size ggml publishes, and the number a container must
    /// split on. Reading the inner group as 8 would bless a 32-element split
    /// that cuts a super-scale in half.
    ///
    /// A group that spans the row or the tensor has no smaller unit to offer,
    /// and the answer is [`Quantum::WholeRow`].
    ///
    /// **WORD PACKING IS NOT IN HERE.** That eight `u4` codes want to land in
    /// a `u32`, or that five trits want to land in a byte, is a fact about the
    /// CONTAINER and is layered on top by whoever writes one; the rates are on
    /// [`Leaf`] for exactly that consumer. This function answers only which
    /// splits the ALGEBRA permits.
    #[must_use]
    pub fn quantum(self) -> Quantum {
        let Self::Q {
            g, gain, offset, ..
        } = self
        else {
            return Quantum::Elems(1);
        };
        let Some(outer) = g.k_span() else {
            return Quantum::WholeRow;
        };
        let mut acc = u64::from(outer);
        let inners = [groups_of(gain), offset.factor().and_then(groups_of)];
        for inner in inners.into_iter().flatten() {
            match inner.k_span() {
                Some(n) => acc = lcm(acc, u64::from(outer) * u64::from(n)),
                None => return Quantum::WholeRow,
            }
        }
        match u32::try_from(acc) {
            Ok(n) => Quantum::Elems(n),
            Err(_) => Quantum::WholeRow,
        }
    }
}

/// Least common multiple.
fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

/// Greatest common divisor.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

// ─────────────────────────────────────────────────────────────────────────
// Term — the truth, behind `alloc`
// ─────────────────────────────────────────────────────────────────────────

/// Where an offset applies, and therefore what it means.
///
/// The two families are not the same arithmetic and no normalization joins
/// them, so the term says which one it is and dispatch reads it off.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Offset {
    /// `gain · (codes − offset)` — the integer zero-point family. The offset
    /// lives in the CODE domain, but it is not necessarily an integer: HQQ's
    /// zero is a real number in the same position, which is the reason this is
    /// a term and not a code width.
    Pre(alloc::boxed::Box<Term>),
    /// `gain · codes + offset` — the real bias family. Always an ADDITION: a
    /// stored minimum is negated into the gain at repack.
    Post(alloc::boxed::Box<Term>),
}

/// A quantization format, in full.
///
/// The recursion is the point — a gain can be quantized, and its gain can be
/// quantized, and nothing in the algebra says stop. [`Sig`] is where the tree
/// stops for DISPATCH; this is where it does not stop for CANONICALIZATION,
/// which is why it is the canonicalizer's working type and why it is boxed and
/// behind a feature.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// A stored element, unquantized.
    Leaf(Leaf),
    /// Grouped codes with a gain and an optional offset.
    Q {
        /// How many codes share one gain.
        g: Group,
        /// What a code is — a leaf, always: codes are stored, never computed,
        /// and the type is what enforces that canonical rule.
        elem: Leaf,
        /// The multiplicative factor, itself a term.
        gain: alloc::boxed::Box<Term>,
        /// The additive or subtractive factor, and which of the two. There is
        /// no constant node in this grammar, so a constant zero-point cannot
        /// be spelled here — it folds into [`Leaf::I`] at repack.
        offset: Option<Offset>,
    },
}

/// How deep [`Term::parse`] will recurse before it calls a spelling hostile.
/// Nothing real is past three; the cap is here so a parser fed a megabyte of
/// `g2_` cannot walk the stack off its end.
#[cfg(feature = "alloc")]
pub const MAX_DEPTH: u32 = 8;

#[cfg(feature = "alloc")]
impl Term {
    /// Parse a mangled spelling at any depth.
    ///
    /// # Errors
    ///
    /// [`SigError`] names which token was wrong; [`SigError::TooDeep`] also
    /// covers a spelling past [`MAX_DEPTH`].
    pub fn parse(mangled: &str) -> Result<Self, SigError> {
        let b = mangled.as_bytes();
        if b.is_empty() {
            return Err(SigError::Empty);
        }
        let (t, e) = Self::at(b, 0, 0)?;
        if e != b.len() {
            return Err(SigError::Trailing);
        }
        Ok(t)
    }

    fn at(b: &[u8], i: usize, depth: u32) -> Result<(Self, usize), SigError> {
        if depth > MAX_DEPTH {
            return Err(SigError::TooDeep);
        }
        let j = tok_end(b, i);
        if !opens_group(b, i, j) {
            return Ok((Self::Leaf(leaf_of(b, i, j)?), j));
        }
        let g = group_of(b, i, j)?;
        let i = sep(b, j)?;
        let j = tok_end(b, i);
        let elem = leaf_of(b, i, j)?;
        let i = sep(b, j)?;
        let (gain, e) = Self::at(b, i, depth + 1)?;
        let gain = alloc::boxed::Box::new(gain);
        let i = sep(b, e)?;
        let j = tok_end(b, i);
        if is(b, i, j, b"n") {
            let q = Self::Q {
                g,
                elem,
                gain,
                offset: None,
            };
            return Ok((q, j));
        }
        let pre = if is(b, i, j, b"z") {
            true
        } else if is(b, i, j, b"b") {
            false
        } else {
            return Err(SigError::BadOffsetMark);
        };
        let i = sep(b, j)?;
        let (off, e) = Self::at(b, i, depth + 1)?;
        let off = alloc::boxed::Box::new(off);
        let offset = Some(if pre {
            Offset::Pre(off)
        } else {
            Offset::Post(off)
        });
        Ok((
            Self::Q {
                g,
                elem,
                gain,
                offset,
            },
            e,
        ))
    }

    /// The mangled spelling.
    #[must_use]
    pub fn mangle(&self) -> alloc::string::String {
        use alloc::string::ToString as _;
        self.to_string()
    }

    /// Project to the depth-two [`Sig`], or `None` when the term does not fit
    /// it — deeper than two, or carrying a POST offset inside a factor, which
    /// [`Sub`] spells only in its pre form.
    #[must_use]
    pub fn sig(&self) -> Option<Sig> {
        match self {
            Self::Leaf(l) => Some(Sig::Plain(*l)),
            Self::Q {
                g,
                elem,
                gain,
                offset,
            } => Some(Sig::Q {
                g: *g,
                elem: *elem,
                gain: sub_of(gain)?,
                offset: match offset {
                    None => OffSub::Nil,
                    Some(Offset::Pre(t)) => OffSub::Pre(sub_of(t)?),
                    Some(Offset::Post(t)) => OffSub::Post(sub_of(t)?),
                },
            }),
        }
    }
}

/// A term one level down, as a [`Sub`], or `None` if it does not fit.
#[cfg(feature = "alloc")]
fn sub_of(t: &Term) -> Option<Sub> {
    match t {
        Term::Leaf(l) => Some(Sub::L(*l)),
        Term::Q {
            g,
            elem,
            gain,
            offset,
        } => {
            let Term::Leaf(gain) = gain.as_ref() else {
                return None;
            };
            match offset {
                None => Some(Sub::Q1(*g, *elem, *gain)),
                Some(Offset::Pre(z)) => match z.as_ref() {
                    Term::Leaf(z) => Some(Sub::Q1z(*g, *elem, *gain, *z)),
                    _ => None,
                },
                Some(Offset::Post(_)) => None,
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl fmt::Display for Term {
    /// The mangled spelling.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leaf(l) => write!(f, "{l}"),
            Self::Q {
                g,
                elem,
                gain,
                offset,
            } => {
                write!(f, "{g}_{elem}_{gain}_")?;
                match offset {
                    None => f.write_str("n"),
                    Some(Offset::Pre(t)) => write!(f, "z_{t}"),
                    Some(Offset::Post(t)) => write!(f, "b_{t}"),
                }
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl From<Sub> for Option<Term> {
    fn from(s: Sub) -> Self {
        use alloc::boxed::Box;
        Some(match s {
            Sub::Nil => return None,
            Sub::L(l) => Term::Leaf(l),
            Sub::Q1(g, elem, gain) => Term::Q {
                g,
                elem,
                gain: Box::new(Term::Leaf(gain)),
                offset: None,
            },
            Sub::Q1z(g, elem, gain, z) => Term::Q {
                g,
                elem,
                gain: Box::new(Term::Leaf(gain)),
                offset: Some(Offset::Pre(Box::new(Term::Leaf(z)))),
            },
        })
    }
}

#[cfg(feature = "alloc")]
impl TryFrom<Sig> for Term {
    type Error = SigError;

    /// Expand a projection back into a term. Fails only on a non-canonical
    /// `Sig` — a [`Sub::Nil`] in a factor slot, which no parser produces.
    fn try_from(s: Sig) -> Result<Self, SigError> {
        use alloc::boxed::Box;
        let (g, elem, gain, offset) = match s {
            Sig::Plain(l) => return Ok(Self::Leaf(l)),
            Sig::Q {
                g,
                elem,
                gain,
                offset,
            } => (g, elem, gain, offset),
        };
        let gain: Option<Self> = gain.into();
        let gain = Box::new(gain.ok_or(SigError::UnknownLeaf)?);
        let offset = match offset {
            OffSub::Nil => None,
            OffSub::Pre(s) => {
                let t: Option<Self> = s.into();
                Some(Offset::Pre(Box::new(t.ok_or(SigError::UnknownLeaf)?)))
            }
            OffSub::Post(s) => {
                let t: Option<Self> = s.into();
                Some(Offset::Post(Box::new(t.ok_or(SigError::UnknownLeaf)?)))
            }
        };
        Ok(Self::Q {
            g,
            elem,
            gain,
            offset,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The registered rows
// ─────────────────────────────────────────────────────────────────────────
//
// Each name IS its spelling, uppercased. That rule is mechanical and a test
// checks it, which is what makes the list auditable: a reader who knows the
// grammar can read every constant below without a table, and a new row cannot
// be added under a name that describes it wrongly.

/// Affine 4-bit codes in groups of 64, bf16 gain, bf16 bias.
///
/// What `mlx_lm.convert` writes at its default settings, and the stack most of
/// the tree's MLX checkpoints are in.
pub const G64_U4_BF16_B_BF16: Sig = sig("g64_u4_bf16_b_bf16");

/// [`G64_U4_BF16_B_BF16`] at eight bits — the same scheme at twice the code
/// width. A converted checkpoint mixes the two on purpose: `mlx_lm`'s
/// `quant_predicate` lifts MoE router gates to eight bits, because four bits
/// of a router picks a different set of experts and the model that results is
/// not the model.
pub const G64_U8_BF16_B_BF16: Sig = sig("g64_u8_bf16_b_bf16");

/// [`G64_U4_BF16_B_BF16`] grouped by 32, for rows too narrow to group by 64 —
/// MLX requires the group to divide the last axis, and qwen4's 160-wide PLE
/// n-gram table does not admit 64.
pub const G32_U4_BF16_B_BF16: Sig = sig("g32_u4_bf16_b_bf16");

/// 4-bit codes in groups of 128 with an f16 gain and an integer zero point of
/// the same width — the byte layout GPTQ, AWQ and compressed-tensors all
/// publish. That three pipelines share one row is the argument for naming the
/// row and not the pipeline.
pub const G128_U4_F16_Z_U4: Sig = sig("g128_u4_f16_z_u4");

/// 4-bit codes in groups of 64 with an f16 gain and a REAL pre-scale zero —
/// HQQ's shape, and the reason [`OffSub::Pre`] cannot assume the offset shares
/// the code's dtype.
pub const G64_U4_F16_Z_F16: Sig = sig("g64_u4_f16_z_f16");

/// Excess-binary 4-bit codes in blocks of 32 with an f16 scale and no offset.
pub const G32_I4_F16_N: Sig = sig("g32_i4_f16_n");

/// Excess-binary 8-bit codes in blocks of 32 with an f16 scale.
pub const G32_I8_F16_N: Sig = sig("g32_i8_f16_n");

/// OCP Microscaling FP4: `e2m1` codes in blocks of 32 under an `e8m0` scale.
pub const G32_E2M1_E8M0_N: Sig = sig("g32_e2m1_e8m0_n");

/// FP8 weights with one f32 scale per output row.
pub const GR_E4M3_F32_N: Sig = sig("gr_e4m3_f32_n");

/// FP8 weights with one f32 scale per 128x128 block of the `[n, k]`
/// rectangle — DeepSeek's block scaling.
pub const G128X128_E4M3_F32_N: Sig = sig("g128x128_e4m3_f32_n");

/// NVFP4: `e2m1` codes in blocks of 16 under an `e4m3` scale, and those scales
/// themselves under one f32 for the whole tensor. Depth two, and the first row
/// here whose gain is not a leaf.
pub const G16_E2M1_GT_E4M3_F32_N_N: Sig = sig("g16_e2m1_gt_e4m3_f32_n_n");

/// The K-quant at four bits: 4-bit codes in sub-blocks of 32, whose scales AND
/// whose mins are themselves 6-bit codes grouped eight to a super-block under
/// an f16 each. The mins are stored to be SUBTRACTED and appear here as an
/// addition, negated into their own gain at repack.
pub const G32_U4_G8_U6_F16_N_B_G8_U6_F16_N: Sig = sig("g32_u4_g8_u6_f16_n_b_g8_u6_f16_n");

/// The K-quant at six bits: excess-binary 6-bit codes in sub-blocks of 16,
/// whose 8-bit scales are grouped sixteen to a super-block under an f16. No
/// min, so no offset anywhere on the tree.
pub const G16_I6_G16_I8_F16_N_N: Sig = sig("g16_i6_g16_i8_f16_n_n");

/// Ternary weights with one f16 scale for the whole tensor — BitNet-class, at
/// eight bits per five weights.
pub const GT_T3_F16_N: Sig = sig("gt_t3_f16_n");

// ── spec synonyms ───────────────────────────────────────────────────────
//
// The ONLY aliases in this module, and each one is a name that belongs to a
// published SPECIFICATION rather than to a toolchain that happens to emit the
// row. Every alias is an alias — a test asserts that each equals a registered
// row, so the two spellings can never drift.

/// OCP Microscaling FP4, by its spec name. See [`G32_E2M1_E8M0_N`].
pub const MXFP4: Sig = G32_E2M1_E8M0_N;

/// NVIDIA's NVFP4, by its spec name. See [`G16_E2M1_GT_E4M3_F32_N_N`].
pub const NVFP4: Sig = G16_E2M1_GT_E4M3_F32_N_N;

/// ggml's `Q4_0`, by its spec name. See [`G32_I4_F16_N`].
pub const Q4_0: Sig = G32_I4_F16_N;

/// ggml's `Q8_0`, by its spec name. See [`G32_I8_F16_N`].
pub const Q8_0: Sig = G32_I8_F16_N;

/// ggml's `Q4_K`, by its spec name. See [`G32_U4_G8_U6_F16_N_B_G8_U6_F16_N`].
pub const Q4_K: Sig = G32_U4_G8_U6_F16_N_B_G8_U6_F16_N;

/// ggml's `Q6_K`, by its spec name. See [`G16_I6_G16_I8_F16_N_N`].
pub const Q6_K: Sig = G16_I6_G16_I8_F16_N_N;

#[cfg(test)]
mod tests {
    use super::*;
    use std::string::{String, ToString};
    use std::vec::Vec;

    /// The registered rows, each paired with the TEXT of its own identifier —
    /// `stringify!` is what keeps the table from drifting away from the
    /// consts it lists.
    macro_rules! rows {
        ($($n:ident),* $(,)?) => { &[$((stringify!($n), $n)),*] };
    }

    const ROWS: &[(&str, Sig)] = rows![
        G64_U4_BF16_B_BF16,
        G64_U8_BF16_B_BF16,
        G32_U4_BF16_B_BF16,
        G128_U4_F16_Z_U4,
        G64_U4_F16_Z_F16,
        G32_I4_F16_N,
        G32_I8_F16_N,
        G32_E2M1_E8M0_N,
        GR_E4M3_F32_N,
        G128X128_E4M3_F32_N,
        G16_E2M1_GT_E4M3_F32_N_N,
        G32_U4_G8_U6_F16_N_B_G8_U6_F16_N,
        G16_I6_G16_I8_F16_N_N,
        GT_T3_F16_N,
    ];

    const SYNONYMS: &[(&str, Sig)] = rows![MXFP4, NVFP4, Q4_0, Q8_0, Q4_K, Q6_K];

    /// The spellings the design round settled, verbatim.
    #[test]
    fn every_worked_example_spells_what_the_design_round_said() {
        let want: &[(Sig, &str)] = &[
            (G64_U4_BF16_B_BF16, "g64_u4_bf16_b_bf16"),
            (G64_U8_BF16_B_BF16, "g64_u8_bf16_b_bf16"),
            (G32_U4_BF16_B_BF16, "g32_u4_bf16_b_bf16"),
            (G128_U4_F16_Z_U4, "g128_u4_f16_z_u4"),
            (G64_U4_F16_Z_F16, "g64_u4_f16_z_f16"),
            (G32_I4_F16_N, "g32_i4_f16_n"),
            (G32_I8_F16_N, "g32_i8_f16_n"),
            (G32_E2M1_E8M0_N, "g32_e2m1_e8m0_n"),
            (GR_E4M3_F32_N, "gr_e4m3_f32_n"),
            (G128X128_E4M3_F32_N, "g128x128_e4m3_f32_n"),
            (G16_E2M1_GT_E4M3_F32_N_N, "g16_e2m1_gt_e4m3_f32_n_n"),
            (
                G32_U4_G8_U6_F16_N_B_G8_U6_F16_N,
                "g32_u4_g8_u6_f16_n_b_g8_u6_f16_n",
            ),
            (G16_I6_G16_I8_F16_N_N, "g16_i6_g16_i8_f16_n_n"),
            (GT_T3_F16_N, "gt_t3_f16_n"),
        ];
        for (s, text) in want {
            assert_eq!(s.mangle().as_str(), *text);
            assert_eq!(Sig::parse(text), Ok(*s));
        }
    }

    /// Structure the spelling actually carries, spot-checked where the tree
    /// order is easiest to get backwards.
    #[test]
    fn the_worked_examples_parse_to_the_structure_they_describe() {
        assert_eq!(
            Q4_0,
            Sig::Q {
                g: Group::N(32),
                elem: Leaf::I(4),
                gain: Sub::L(Leaf::F16),
                offset: OffSub::Nil,
            }
        );
        assert_eq!(
            NVFP4,
            Sig::Q {
                g: Group::N(16),
                elem: Leaf::E { e: 2, m: 1 },
                gain: Sub::Q1(Group::Tensor, Leaf::E { e: 4, m: 3 }, Leaf::F32),
                offset: OffSub::Nil,
            }
        );
        assert_eq!(
            Q4_K,
            Sig::Q {
                g: Group::N(32),
                elem: Leaf::U(4),
                gain: Sub::Q1(Group::N(8), Leaf::U(6), Leaf::F16),
                offset: OffSub::Post(Sub::Q1(Group::N(8), Leaf::U(6), Leaf::F16)),
            }
        );
        assert_eq!(
            G128X128_E4M3_F32_N,
            Sig::Q {
                g: Group::Tile(128, 128),
                elem: Leaf::E { e: 4, m: 3 },
                gain: Sub::L(Leaf::F32),
                offset: OffSub::Nil,
            }
        );
    }

    /// A structured sweep of the whole grammar, not just the shipped rows.
    fn sweep() -> Vec<Sig> {
        let leaves = [
            Leaf::U(4),
            Leaf::U(8),
            Leaf::I(4),
            Leaf::I(6),
            Leaf::E { e: 2, m: 1 },
            Leaf::E { e: 4, m: 3 },
            Leaf::E { e: 8, m: 0 },
            Leaf::F32,
            Leaf::F16,
            Leaf::Bf16,
            Leaf::Nf4,
            Leaf::T3,
            Leaf::Cb(7),
        ];
        let groups = [
            Group::N(16),
            Group::N(32),
            Group::N(64),
            Group::N(128),
            Group::Tile(128, 128),
            Group::Tile(1, 64),
            Group::Row,
            Group::Tensor,
        ];
        let gains = [
            Sub::L(Leaf::F16),
            Sub::L(Leaf::Bf16),
            Sub::Q1(Group::N(8), Leaf::U(6), Leaf::F16),
            Sub::Q1z(Group::N(8), Leaf::U(6), Leaf::F16, Leaf::U(6)),
        ];
        let offsets = [
            OffSub::Nil,
            OffSub::Pre(Sub::L(Leaf::U(4))),
            OffSub::Pre(Sub::L(Leaf::F16)),
            OffSub::Post(Sub::L(Leaf::Bf16)),
            OffSub::Post(Sub::Q1(Group::N(8), Leaf::U(6), Leaf::F16)),
            OffSub::Pre(Sub::Q1z(Group::N(4), Leaf::U(4), Leaf::F16, Leaf::U(4))),
        ];
        let mut out = Vec::new();
        for l in leaves {
            out.push(Sig::Plain(l));
        }
        for g in groups {
            for elem in leaves {
                for gain in gains {
                    for offset in offsets {
                        out.push(Sig::Q {
                            g,
                            elem,
                            gain,
                            offset,
                        });
                    }
                }
            }
        }
        out
    }

    #[test]
    fn parse_inverts_mangle_over_the_whole_grammar() {
        let all = sweep();
        assert!(all.len() > 2000, "the sweep should be broad: {}", all.len());
        for s in all {
            let text = s.mangle();
            assert_eq!(Sig::parse(text.as_str()), Ok(s), "{text}");
        }
        for (_, s) in ROWS.iter().chain(SYNONYMS) {
            assert_eq!(Sig::parse(s.mangle().as_str()), Ok(*s));
        }
    }

    #[test]
    fn no_two_registered_rows_share_a_spelling() {
        let mut seen: Vec<String> = Vec::new();
        for (_, s) in ROWS {
            let text = s.mangle().as_str().to_string();
            assert!(!seen.contains(&text), "two rows spell {text}");
            seen.push(text);
        }
        assert_eq!(seen.len(), ROWS.len());
    }

    /// The naming rule, mechanically: a row's identifier IS its spelling.
    #[test]
    fn every_row_is_named_by_its_own_spelling_uppercased() {
        for (name, s) in ROWS {
            assert_eq!(*name, s.mangle().as_str().to_uppercase());
        }
    }

    /// A spec synonym is an alias and never a fourteenth row.
    #[test]
    fn the_spec_synonyms_alias_a_registered_row() {
        for (name, s) in SYNONYMS {
            assert!(
                ROWS.iter().any(|(_, row)| row == s),
                "{name} is not any registered row"
            );
        }
    }

    /// The bits-per-weight numbers the design doc pins.
    #[test]
    fn bpw_matches_the_published_tables() {
        let k = 4096;
        assert_eq!(MXFP4.bpw(k), Some(4.25));
        assert_eq!(NVFP4.bpw(k), Some(4.5));
        assert_eq!(G128_U4_F16_Z_U4.bpw(k), Some(4.15625));
        assert_eq!(Q4_K.bpw(k), Some(4.5));
        assert_eq!(Q6_K.bpw(k), Some(6.5625));
        assert_eq!(Q4_0.bpw(k), Some(4.5));
        assert_eq!(Q8_0.bpw(k), Some(8.5));
        // 4 code bits, plus a bf16 scale AND a bf16 bias every 64 codes.
        assert_eq!(G64_U4_BF16_B_BF16.bpw(k), Some(4.0 + 32.0 / 64.0));
        assert_eq!(G64_U8_BF16_B_BF16.bpw(k), Some(8.5));
        assert_eq!(G32_U4_BF16_B_BF16.bpw(k), Some(5.0));
        // A per-row scale is the one that moves with k.
        assert_eq!(GR_E4M3_F32_N.bpw(4096), Some(8.0 + 32.0 / 4096.0));
        assert_eq!(GR_E4M3_F32_N.bpw(1024), Some(8.0 + 32.0 / 1024.0));
        // Eight bits per five weights, and a tensor-wide scale costs nothing.
        assert!((GT_T3_F16_N.bpw(k).unwrap() - 1.6).abs() < 1e-12);
        assert_eq!(Sig::Plain(Leaf::Bf16).bpw(k), Some(16.0));
        // No rate, no answer.
        assert_eq!(Sig::Plain(Leaf::Cb(3)).bpw(k), None);
        assert_eq!(MXFP4.bpw(0), None);
    }

    /// The byte rectangle, which is where a wrong number actually cost us:
    /// `engine-cuda`'s `weights::packed` handed plane widths through in
    /// ELEMENTS, the qwen4 PLE gather recovered group 80 from a five-factor
    /// row, and the first light ran with mis-scaled n-gram embeddings and
    /// still said Paris. These three numbers are that bank, done right.
    #[test]
    fn plane_widths_are_the_byte_rectangle_the_gather_needed() {
        let w = G64_U4_BF16_B_BF16.plane_widths(10240).expect("has rates");
        assert_eq!(w.as_slice(), &[5120, 320, 320]);
        assert_eq!(w.len(), 3);
        assert!(!w.is_empty());

        // Tree order: codes, then the gain path, then the offset path.
        assert_eq!(
            Q4_K.plane_widths(4096).unwrap().as_slice(),
            &[2048, 96, 32, 96, 32]
        );
        assert_eq!(Q6_K.plane_widths(4096).unwrap().as_slice(), &[3072, 256, 32]);
        assert_eq!(MXFP4.plane_widths(4096).unwrap().as_slice(), &[2048, 128]);
        // The tensor-wide f32 still occupies a plane, of one element.
        assert_eq!(NVFP4.plane_widths(4096).unwrap().as_slice(), &[2048, 256, 4]);
        // A row-wide factor is one element per row; a tile's is per k extent.
        assert_eq!(GR_E4M3_F32_N.plane_widths(4096).unwrap().as_slice(), &[4096, 4]);
        assert_eq!(
            G128X128_E4M3_F32_N.plane_widths(4096).unwrap().as_slice(),
            &[4096, 128]
        );
        // Five trits to the byte, rounded up once.
        assert_eq!(GT_T3_F16_N.plane_widths(4096).unwrap().as_slice(), &[820, 2]);
        assert_eq!(Sig::Plain(Leaf::Cb(3)).plane_widths(64), None);
    }

    /// Every sweep case names at most `MAX_PLANES` planes.
    #[test]
    fn no_sig_names_more_planes_than_the_buffer_holds() {
        let mut widest = 0;
        for s in sweep() {
            if let Some(w) = s.plane_widths(4096) {
                widest = widest.max(w.len());
            }
        }
        assert_eq!(widest, MAX_PLANES);
    }

    #[test]
    fn the_quantum_is_the_super_block_and_not_the_sub_block() {
        assert_eq!(MXFP4.quantum(), Quantum::Elems(32));
        assert_eq!(Q4_0.quantum(), Quantum::Elems(32));
        // A nested group counts its parent in: 8 scales x 32 weights.
        assert_eq!(Q4_K.quantum(), Quantum::Elems(256));
        assert_eq!(Q6_K.quantum(), Quantum::Elems(256));
        // A tensor-wide gain leaves nothing smaller than the row.
        assert_eq!(NVFP4.quantum(), Quantum::WholeRow);
        assert_eq!(GR_E4M3_F32_N.quantum(), Quantum::WholeRow);
        assert_eq!(G128X128_E4M3_F32_N.quantum(), Quantum::Elems(128));
        // An unquantized row splits anywhere.
        assert_eq!(Sig::Plain(Leaf::F16).quantum(), Quantum::Elems(1));
        assert_eq!(Quantum::WholeRow.elems(4096), 4096);
        assert_eq!(Quantum::Elems(32).elems(4096), 32);
    }

    /// `sig` is a `const fn`: this line is the proof, and a typo in it would
    /// be a build error rather than a test failure.
    #[test]
    fn sig_builds_in_a_const_context() {
        const PROOF: Sig = sig("g64_u4_bf16_b_bf16");
        const PLAIN: Sig = sig("bf16");
        assert_eq!(PROOF, G64_U4_BF16_B_BF16);
        assert_eq!(PLAIN, Sig::Plain(Leaf::Bf16));
        // And it is usable as a const pattern, which is the other half of why
        // `Sig` is `Copy` and structural.
        assert!(matches!(G64_U4_BF16_B_BF16, PROOF));
    }

    #[test]
    fn the_grammar_says_no_to_what_is_not_a_format() {
        let bad: &[(&str, SigError)] = &[
            ("", SigError::Empty),
            ("g64_u4_bf16", SigError::Truncated),
            ("gt_t3_f16", SigError::Truncated),
            ("g64_u4_bf16_b_bf16_f16", SigError::Trailing),
            ("g0_u4_f16_n", SigError::ZeroGroup),
            ("g32x0_u4_f16_n", SigError::ZeroGroup),
            ("u65", SigError::BadWidth),
            ("u0", SigError::BadWidth),
            ("g32_q4_f16_n", SigError::UnknownLeaf),
            ("g32_e4_f16_n", SigError::BadExpToken),
            ("gx_u4_f16_n", SigError::MissingNumber),
            ("g3a_u4_f16_n", SigError::BadNumber),
            ("cb70000", SigError::NumberTooLarge),
            ("g32_u4_f16_q", SigError::BadOffsetMark),
            ("g32_u4_g8_u6_g4_f16_n_n_n", SigError::TooDeep),
            ("g32_u4_g8_u6_f16_b_f16_n", SigError::NestedPost),
        ];
        for (text, why) in bad {
            assert_eq!(Sig::parse(text), Err(*why), "{text}");
            assert!(!why.why().is_empty());
            assert_eq!(why.to_string(), why.why());
        }
    }

    /// The one canonical rule the types cannot carry.
    #[test]
    fn a_nil_factor_is_not_canonical_and_has_no_spelling() {
        let nil_gain = Sig::Q {
            g: Group::N(32),
            elem: Leaf::U(4),
            gain: Sub::Nil,
            offset: OffSub::Nil,
        };
        let nil_offset = Sig::Q {
            g: Group::N(32),
            elem: Leaf::U(4),
            gain: Sub::L(Leaf::F16),
            offset: OffSub::Pre(Sub::Nil),
        };
        assert!(!nil_gain.is_canonical());
        assert!(!nil_offset.is_canonical());
        assert!(G64_U4_BF16_B_BF16.is_canonical());
        assert!(Sig::Plain(Leaf::F16).is_canonical());
        // The parser cannot produce one, which is the point of checking here.
        for s in sweep() {
            assert!(s.is_canonical());
        }
    }

    // ── Term, the unprojected truth ────────────────────────────────────

    /// Every registered row survives the trip out to a term and back.
    #[cfg(feature = "alloc")]
    #[test]
    fn every_row_expands_to_a_term_and_projects_back() {
        for (name, s) in ROWS.iter().chain(SYNONYMS) {
            let term = Term::try_from(*s).expect("canonical");
            assert_eq!(term.mangle(), s.mangle().as_str(), "{name}");
            assert_eq!(term.sig(), Some(*s), "{name}");
            assert_eq!(Term::parse(&term.mangle()), Ok(term), "{name}");
        }
    }

    /// And so does the whole sweep, which is where the shapes get strange.
    #[cfg(feature = "alloc")]
    #[test]
    fn parse_inverts_mangle_for_terms_too() {
        for s in sweep() {
            let term = Term::try_from(s).expect("canonical");
            let text = term.mangle();
            assert_eq!(Term::parse(&text), Ok(term.clone()), "{text}");
            assert_eq!(term.sig(), Some(s), "{text}");
            assert_eq!(text, s.mangle().as_str());
        }
    }

    /// The projection is partial, and these are the two ways it fails.
    #[cfg(feature = "alloc")]
    #[test]
    fn a_term_past_the_projection_has_no_sig() {
        use alloc::boxed::Box;

        // Depth three: a gain whose gain is itself quantized.
        let deep = Term::parse("g32_u4_g8_u6_g4_u8_f16_n_n_n").expect("a legal term");
        assert_eq!(deep.sig(), None);
        assert_eq!(Sig::parse("g32_u4_g8_u6_g4_u8_f16_n_n_n"), Err(SigError::TooDeep));

        // Depth two, but with a POST offset nested inside the gain: `Sub`
        // spells the pre family only, so this term is legal and unprojected.
        let nested = Term::parse("g32_u4_g8_u6_f16_b_f16_n").expect("a legal term");
        assert_eq!(nested.sig(), None);
        assert_eq!(
            Sig::parse("g32_u4_g8_u6_f16_b_f16_n"),
            Err(SigError::NestedPost)
        );

        // A leaf projects to a Plain, always.
        assert_eq!(Term::Leaf(Leaf::Bf16).sig(), Some(Sig::Plain(Leaf::Bf16)));

        // The offset families do not collapse into each other.
        let pre = Term::Q {
            g: Group::N(64),
            elem: Leaf::U(4),
            gain: Box::new(Term::Leaf(Leaf::F16)),
            offset: Some(Offset::Pre(Box::new(Term::Leaf(Leaf::F16)))),
        };
        let post = Term::Q {
            g: Group::N(64),
            elem: Leaf::U(4),
            gain: Box::new(Term::Leaf(Leaf::F16)),
            offset: Some(Offset::Post(Box::new(Term::Leaf(Leaf::F16)))),
        };
        assert_ne!(pre, post);
        assert_ne!(pre.mangle(), post.mangle());
        assert_eq!(pre.mangle(), "g64_u4_f16_z_f16");
        assert_eq!(post.mangle(), "g64_u4_f16_b_f16");
    }

    /// A hostile spelling stops at the depth cap instead of at the stack.
    #[cfg(feature = "alloc")]
    #[test]
    fn a_runaway_spelling_stops_at_the_depth_cap() {
        let mut text = std::string::String::new();
        for _ in 0..(MAX_DEPTH + 4) {
            text.push_str("g2_u4_");
        }
        text.push_str("f16");
        for _ in 0..(MAX_DEPTH + 4) {
            text.push_str("_n");
        }
        assert_eq!(Term::parse(&text), Err(SigError::TooDeep));
        assert_eq!(Term::parse(""), Err(SigError::Empty));
        assert_eq!(Term::parse("f16_f16"), Err(SigError::Trailing));
    }

    /// Rates, against the widths the formats publish.
    #[test]
    fn leaf_rates_are_the_widths_the_formats_publish() {
        assert_eq!(Leaf::U(4).rate(), Some((4, 1)));
        assert_eq!(Leaf::I(6).rate(), Some((6, 1)));
        assert_eq!(Leaf::F16.rate(), Some((16, 1)));
        assert_eq!(Leaf::Bf16.rate(), Some((16, 1)));
        assert_eq!(Leaf::F32.rate(), Some((32, 1)));
        assert_eq!(Leaf::Nf4.rate(), Some((4, 1)));
        assert_eq!(Leaf::T3.rate(), Some((8, 5)));
        assert_eq!(Leaf::E { e: 2, m: 1 }.rate(), Some((4, 1)));
        assert_eq!(Leaf::E { e: 5, m: 2 }.rate(), Some((8, 1)));
        assert_eq!(Leaf::E { e: 4, m: 3 }.rate(), Some((8, 1)));
        // Exponent-only carries no sign: e8m0 is eight bits, not nine.
        assert_eq!(Leaf::E { e: 8, m: 0 }.rate(), Some((8, 1)));
        assert_eq!(Leaf::Cb(0).rate(), None);
    }
}
