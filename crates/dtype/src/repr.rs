use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Elem {
    U(u8),
    I(u8),
    E { e: u8, m: u8 },
    F32,
    F16,
    Bf16,
    Bool,
    Nf4,
    T3,
    Cb(u16),
}

impl Elem {
    #[must_use]
    pub const fn rate(self) -> Option<(u32, u32)> {
        match self {
            Self::U(b) | Self::I(b) => Some((b as u32, 1)),
            Self::E { e, m } => Some((
                if m == 0 {
                    e as u32
                } else {
                    1 + e as u32 + m as u32
                },
                1,
            )),
            Self::F32 => Some((32, 1)),
            Self::F16 | Self::Bf16 => Some((16, 1)),
            Self::Bool => Some((8, 1)),
            Self::Nf4 => Some((4, 1)),
            Self::T3 => Some((8, 5)),
            Self::Cb(_) => None,
        }
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::U(b) => write!(f, "u{b}"),
            Self::I(b) => write!(f, "i{b}"),
            Self::E { e, m } => write!(f, "e{e}m{m}"),
            Self::F32 => f.write_str("f32"),
            Self::F16 => f.write_str("f16"),
            Self::Bf16 => f.write_str("bf16"),
            Self::Bool => f.write_str("bool"),
            Self::Nf4 => f.write_str("nf4"),
            Self::T3 => f.write_str("t3"),
            Self::Cb(n) => write!(f, "cb{n}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    N(u32),
    Tile(u32, u32),
    Row,
    Tensor,
}

impl Group {
    const fn k_extent(self, ambient: u64) -> u64 {
        match self {
            Self::N(n) => n as u64,
            Self::Tile(_, c) => c as u64,
            Self::Row | Self::Tensor => ambient,
        }
    }

    const fn k_span(self) -> Option<u32> {
        match self {
            Self::N(n) => Some(n),
            Self::Tile(_, c) => Some(c),
            Self::Row | Self::Tensor => None,
        }
    }

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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::N(n) => write!(f, "g{n}"),
            Self::Tile(r, c) => write!(f, "g{r}x{c}"),
            Self::Row => f.write_str("gr"),
            Self::Tensor => f.write_str("gt"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Off<'a> {
    Pre(&'a Fmt<'a>),
    Post(&'a Fmt<'a>),
}

impl<'a> Off<'a> {
    #[must_use]
    pub const fn factor(self) -> &'a Fmt<'a> {
        match self {
            Self::Pre(t) | Self::Post(t) => t,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fmt<'a> {
    Elem(Elem),
    Q {
        g: Group,
        elem: Elem,
        gain: &'a Fmt<'a>,
        offset: Option<Off<'a>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quantum {
    Elems(u32),
    WholeRow,
}

impl Quantum {
    #[must_use]
    pub const fn elems(self, k: u32) -> u32 {
        match self {
            Self::Elems(n) => n,
            Self::WholeRow => k,
        }
    }
}

impl fmt::Display for Fmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Elem(e) => write!(f, "{e}"),
            Self::Q {
                g,
                elem,
                gain,
                offset,
            } => {
                write!(f, "{g}_{elem}_{gain}_")?;
                match offset {
                    None => f.write_str("n"),
                    Some(Off::Pre(t)) => write!(f, "z_{t}"),
                    Some(Off::Post(t)) => write!(f, "b_{t}"),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mangled {
    bytes: [u8; Self::CAPACITY],
    len: usize,
}

impl Mangled {
    pub const CAPACITY: usize = 160;

    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes: [0; Self::CAPACITY],
            len: 0,
        }
    }

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

impl Fmt<'_> {
    #[must_use]
    pub fn mangle(&self) -> Mangled {
        use fmt::Write as _;
        let mut out = Mangled::new();
        write!(out, "{self}").expect("a shipped format fits its own buffer");
        out
    }
}

macro_rules! opt {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return None,
        }
    };
}

const fn tok_end(b: &[u8], i: usize) -> usize {
    let mut j = i;
    while j < b.len() && b[j] != b'_' {
        j += 1;
    }
    j
}

const fn sep(b: &[u8], i: usize) -> Option<usize> {
    if i < b.len() && b[i] == b'_' {
        Some(i + 1)
    } else {
        None
    }
}

const fn is(b: &[u8], i: usize, j: usize, lit: &[u8]) -> bool {
    if j < i || j - i != lit.len() {
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

const fn digits(b: &[u8], i: usize, j: usize) -> Option<u64> {
    if i >= j {
        return None;
    }
    let mut v: u64 = 0;
    let mut k = i;
    while k < j {
        let c = b[k];
        if c < b'0' || c > b'9' {
            return None;
        }
        v = v * 10 + (c - b'0') as u64;
        if v > u32::MAX as u64 {
            return None;
        }
        k += 1;
    }
    Some(v)
}

const fn elem_at(e: Elem, b: &[u8], i: usize) -> Option<usize> {
    let j = tok_end(b, i);
    let ok = match e {
        Elem::F32 => is(b, i, j, b"f32"),
        Elem::F16 => is(b, i, j, b"f16"),
        Elem::Bf16 => is(b, i, j, b"bf16"),
        Elem::Bool => is(b, i, j, b"bool"),
        Elem::Nf4 => is(b, i, j, b"nf4"),
        Elem::T3 => is(b, i, j, b"t3"),
        Elem::U(w) => {
            i < j && b[i] == b'u' && matches!(digits(b, i + 1, j), Some(n) if n == w as u64)
        }
        Elem::I(w) => {
            i < j && b[i] == b'i' && matches!(digits(b, i + 1, j), Some(n) if n == w as u64)
        }
        Elem::E { e, m } => 'e_token: {
            if i >= j || b[i] != b'e' {
                break 'e_token false;
            }
            let mut k = i + 1;
            while k < j && b[k] != b'm' {
                k += 1;
            }
            k < j
                && matches!(digits(b, i + 1, k), Some(n) if n == e as u64)
                && matches!(digits(b, k + 1, j), Some(n) if n == m as u64)
        }
        Elem::Cb(n) => {
            j - i > 2
                && b[i] == b'c'
                && b[i + 1] == b'b'
                && matches!(digits(b, i + 2, j), Some(v) if v == n as u64)
        }
    };
    if ok { Some(j) } else { None }
}

const fn group_at(g: Group, b: &[u8], i: usize) -> Option<usize> {
    let j = tok_end(b, i);
    let ok = match g {
        Group::Row => is(b, i, j, b"gr"),
        Group::Tensor => is(b, i, j, b"gt"),
        Group::N(n) => {
            i < j && b[i] == b'g' && matches!(digits(b, i + 1, j), Some(v) if v == n as u64)
        }
        Group::Tile(r, c) => 'tile: {
            if i >= j || b[i] != b'g' {
                break 'tile false;
            }
            let mut k = i + 1;
            while k < j && b[k] != b'x' {
                k += 1;
            }
            k < j
                && matches!(digits(b, i + 1, k), Some(v) if v == r as u64)
                && matches!(digits(b, k + 1, j), Some(v) if v == c as u64)
        }
    };
    if ok { Some(j) } else { None }
}

const fn walk(f: &Fmt<'_>, b: &[u8], i: usize) -> Option<usize> {
    match *f {
        Fmt::Elem(e) => elem_at(e, b, i),
        Fmt::Q {
            g,
            elem,
            gain,
            offset,
        } => {
            let i = opt!(sep(b, opt!(group_at(g, b, i))));
            let i = opt!(sep(b, opt!(elem_at(elem, b, i))));
            let i = opt!(sep(b, opt!(walk(gain, b, i))));
            let j = tok_end(b, i);
            match offset {
                None => {
                    if is(b, i, j, b"n") {
                        Some(j)
                    } else {
                        None
                    }
                }
                Some(Off::Pre(t)) => {
                    if !is(b, i, j, b"z") {
                        return None;
                    }
                    walk(t, b, opt!(sep(b, j)))
                }
                Some(Off::Post(t)) => {
                    if !is(b, i, j, b"b") {
                        return None;
                    }
                    walk(t, b, opt!(sep(b, j)))
                }
            }
        }
    }
}

#[must_use]
pub const fn spells(f: &Fmt<'_>, spelling: &str) -> bool {
    let b = spelling.as_bytes();
    matches!(walk(f, b, 0), Some(end) if end == b.len())
}

pub const MAX_PLANES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneWidths {
    bytes: [u32; MAX_PLANES],
    len: usize,
}

impl PlaneWidths {
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.bytes[..self.len]
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

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

    fn push(&mut self, w: u32) -> Option<()> {
        if self.len == MAX_PLANES {
            return None;
        }
        self.bytes[self.len] = w;
        self.len += 1;
        Some(())
    }
}

fn elem_bytes(e: Elem, n: u64) -> Option<u32> {
    let (num, den) = e.rate()?;
    let bits = (n * u64::from(num)).div_ceil(u64::from(den));
    u32::try_from(bits.div_ceil(8)).ok()
}

fn cost(f: &Fmt<'_>, copies: f64) -> Option<f64> {
    let rate = |e: Elem| e.rate().map(|(n, d)| f64::from(n) / f64::from(d));
    Some(match *f {
        Fmt::Elem(e) => copies * rate(e)?,
        Fmt::Q {
            g,
            elem,
            gain,
            offset,
        } => {
            let inner = g.shares(copies);
            let mut total = copies * rate(elem)? + cost(gain, inner)?;
            if let Some(off) = offset {
                total += cost(off.factor(), inner)?;
            }
            total
        }
    })
}

fn planes(f: &Fmt<'_>, elems: u64, out: &mut PlaneWidths) -> Option<()> {
    match *f {
        Fmt::Elem(e) => out.push(elem_bytes(e, elems)?),
        Fmt::Q {
            g,
            elem,
            gain,
            offset,
        } => {
            out.push(elem_bytes(elem, elems)?)?;
            let inner = elems.div_ceil(g.k_extent(elems));
            planes(gain, inner, out)?;
            if let Some(off) = offset {
                planes(off.factor(), inner, out)?;
            }
            Some(())
        }
    }
}

fn quantum_walk(f: &Fmt<'_>, parent: u64, acc: &mut u64) -> Option<()> {
    let Fmt::Q {
        g, gain, offset, ..
    } = *f
    else {
        return Some(());
    };
    let span = u64::from(g.k_span()?) * parent;
    *acc = lcm(*acc, span);
    quantum_walk(gain, span, acc)?;
    if let Some(off) = offset {
        quantum_walk(off.factor(), span, acc)?;
    }
    Some(())
}

impl Fmt<'_> {
    #[must_use]
    pub const fn code(&self) -> Elem {
        match *self {
            Self::Elem(e) => e,
            Self::Q { elem, .. } => elem,
        }
    }

    #[must_use]
    pub fn bpw(&self, k: u32) -> Option<f64> {
        if k == 0 {
            return None;
        }
        let k = f64::from(k);
        Some(cost(self, k)? / k)
    }

    #[must_use]
    pub fn plane_widths(&self, k: u32) -> Option<PlaneWidths> {
        let mut out = PlaneWidths::new();
        planes(self, u64::from(k), &mut out)?;
        Some(out)
    }

    #[must_use]
    pub fn row_bytes(&self, k: u32) -> Option<u64> {
        if k == 0 || !k.is_multiple_of(self.quantum().elems(k)) {
            return None;
        }
        let widths = self.plane_widths(k)?;
        let mut total: u64 = 0;
        for bytes in widths.as_slice() {
            total += u64::from(*bytes);
        }
        Some(total)
    }

    #[must_use]
    pub fn quantum(&self) -> Quantum {
        if matches!(self, Fmt::Elem(_)) {
            return Quantum::Elems(1);
        }
        let mut acc = 1u64;
        if quantum_walk(self, 1, &mut acc).is_none() {
            return Quantum::WholeRow;
        }
        match u32::try_from(acc) {
            Ok(n) => Quantum::Elems(n),
            Err(_) => Quantum::WholeRow,
        }
    }
}

fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

pub const G128_U4_F16_Z_U4: Fmt<'static> = Fmt::Q {
    g: Group::N(128),
    elem: Elem::U(4),
    gain: &Fmt::Elem(Elem::F16),
    offset: Some(Off::Pre(&Fmt::Elem(Elem::U(4)))),
};
const _: () = assert!(spells(&G128_U4_F16_Z_U4, "g128_u4_f16_z_u4"));

pub const G64_U4_F16_Z_F16: Fmt<'static> = Fmt::Q {
    g: Group::N(64),
    elem: Elem::U(4),
    gain: &Fmt::Elem(Elem::F16),
    offset: Some(Off::Pre(&Fmt::Elem(Elem::F16))),
};
const _: () = assert!(spells(&G64_U4_F16_Z_F16, "g64_u4_f16_z_f16"));

pub const GT_T3_F16_N: Fmt<'static> = Fmt::Q {
    g: Group::Tensor,
    elem: Elem::T3,
    gain: &Fmt::Elem(Elem::F16),
    offset: None,
};
const _: () = assert!(spells(&GT_T3_F16_N, "gt_t3_f16_n"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_every_case() {
        display_and_the_walker_agree_over_a_sweep();
        bpw_matches_the_published_tables();
    }

    fn display_and_the_walker_agree_over_a_sweep() {
        const GAINS: &[Fmt<'static>] = &[
            Fmt::Elem(Elem::F16),
            Fmt::Elem(Elem::Bf16),
            Fmt::Q {
                g: Group::N(8),
                elem: Elem::U(6),
                gain: &Fmt::Elem(Elem::F16),
                offset: None,
            },
            Fmt::Q {
                g: Group::Tensor,
                elem: Elem::E { e: 4, m: 3 },
                gain: &Fmt::Elem(Elem::F32),
                offset: None,
            },
        ];
        const OFFS: &[Option<Off<'static>>] = &[
            None,
            Some(Off::Pre(&Fmt::Elem(Elem::U(4)))),
            Some(Off::Post(&Fmt::Elem(Elem::Bf16))),
            Some(Off::Post(&Fmt::Q {
                g: Group::N(8),
                elem: Elem::U(6),
                gain: &Fmt::Elem(Elem::F16),
                offset: None,
            })),
        ];
        let elems = [
            Elem::U(4),
            Elem::I(6),
            Elem::E { e: 2, m: 1 },
            Elem::T3,
            Elem::Nf4,
        ];
        let groups = [
            Group::N(32),
            Group::Tile(128, 128),
            Group::Row,
            Group::Tensor,
        ];
        let mut count = 0;
        for g in groups {
            for elem in elems {
                for gain in GAINS {
                    for offset in OFFS {
                        let f = Fmt::Q {
                            g,
                            elem,
                            gain,
                            offset: *offset,
                        };
                        let text = f.mangle();
                        assert!(spells(&f, text.as_str()), "{text}");
                        count += 1;
                    }
                }
            }
        }
        assert!(count > 200, "the sweep should be broad: {count}");
    }

    fn bpw_matches_the_published_tables() {
        let k = 4096;
        assert_eq!(G128_U4_F16_Z_U4.bpw(k), Some(4.15625));
        assert!((GT_T3_F16_N.bpw(k).unwrap() - 1.6).abs() < 1e-12);
        assert_eq!(Fmt::Elem(Elem::Bf16).bpw(k), Some(16.0));
        assert_eq!(Fmt::Elem(Elem::Cb(3)).bpw(k), None);
        assert_eq!(G128_U4_F16_Z_U4.bpw(0), None);
    }
}
