//! A grid program's CSL ([`crate::compile::stream::grid::GridProgram`]): `layout.csl`, a kernel per PE
//! kind, each printed region once, and what the host writes and reads ([`GridArray`]). Every
//! kernel runs the passes in order, each a sweep over tail rows with the head pulled to the
//! row its gather needs, so every PE interleaves head and tail rows alike.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use poem_ir::ValueId;

use crate::compile::emit::placed::Placed;
use crate::compile::emit::stream::{COMMON, Tiles, buf, cfg_decl, dedup, placed_of};
use crate::compile::fabric::Endpoint;
use crate::compile::floor::Coord;
use crate::compile::floorplan::stream::{
    self as fs, COL_CHAIN, KV_CHAIN, KV_RECV, KV_SEND, ROW_CHAIN,
};
use crate::compile::stream::grid::{GridProgram, Kind, MvIn, RegIn, Res, Step};
use crate::compile::stream::host::{BF16, FLAT, HALF, NONE, Span, UNIT, Word};
use crate::compile::stream::unit::Side;
use crate::compile::wiring::Slot;

type R<T> = Result<T, String>;

/// What the host writes into (or reads from) one PE's array.
#[derive(Clone, Debug, PartialEq)]
pub enum GridFill {
    /// The program's own words (a PE's config words).
    Words(Vec<u32>),
    /// Runs of the trace's values' elements (a tile, a root's residents): the checkpoint's.
    Spans(Vec<Span>),
    /// A step's values' elements (the host's rows): written each step.
    Entry(Vec<Word>),
    /// Read back: each of `rows` (the value's rows) the `parts` (value, from, len) side by side.
    Back {
        rows: Vec<usize>,
        parts: Vec<(u32, u32, u32)>,
    },
}

/// One PE array the host fills or reads.
#[derive(Clone, Debug, PartialEq)]
pub struct GridArray {
    pub at: Coord,
    pub sym: String,
    pub u16: bool,
    pub len: u32,
    pub fill: GridFill,
    /// The weight group whose launch it is written before (a group's tiles, its pass range);
    /// `None`: once a load, or each step's.
    pub group: Option<usize>,
}

/// The program's files and arrays.
#[derive(Clone, Debug)]
pub struct Emitted {
    pub files: Vec<(String, String)>,
    /// Its placement (`layout.csl`'s): what the static check reads.
    pub placed: super::placed::Placed,
    pub arrays: Vec<GridArray>,
    pub w: u32,
    pub h: u32,
}

fn ep(g: &GridProgram, c: Coord, slot: Slot) -> Option<Endpoint> {
    g.fabric.io(c).slots.get(&slot).copied()
}

fn colour_of(g: &GridProgram, name: &str) -> u32 {
    g.palette
        .map
        .iter()
        .find(|(k, _)| k.name == name && k.index == 0)
        .map_or(0xff, |(_, c)| u32::from(c.0))
}

/// `(send, recv, role)` of `c`'s chain `slot` (0xff where none).
fn chain_cols(g: &GridProgram, c: Coord, slot: Slot) -> (u32, u32, u32) {
    match ep(g, c, slot) {
        Some(Endpoint::Member(m)) => (
            u32::from(m.send.0),
            m.recv1.map_or(0xff, |c| u32::from(c.0)),
            m.role,
        ),
        Some(Endpoint::Root(r)) => (0xff, r.lo.or(r.hi).map_or(0xff, |c| u32::from(c.0)), 6),
        _ => (0xff, 0xff, 0),
    }
}

/// `0xff` (none) params filled with colours no route at `c` takes (parks for unused queues).
fn parked(g: &GridProgram, c: Coord, mut v: Vec<(String, u32)>) -> Vec<(String, u32)> {
    let routed: Vec<u8> = g.fabric.routes(c).iter().map(|r| r.colour.0).collect();
    let taken: Vec<u8> = v
        .iter()
        .filter(|(k, x)| k.starts_with("C_") && *x != 0xff)
        .map(|(_, x)| *x as u8)
        .collect();
    let mut free =
        (0..crate::compile::fabric::COLOURS).filter(|k| !routed.contains(k) && !taken.contains(k));
    for (k, x) in &mut v {
        if k.starts_with("C_") && *x == 0xff {
            *x = u32::from(free.next().unwrap_or(20));
        }
    }
    v
}

/// The root regions' modules: one a distinct printed text.
struct Mods {
    names: Vec<(String, String)>,
    of_reg: Vec<usize>,
}

fn mods(g: &GridProgram) -> Mods {
    let mut names: Vec<(String, String)> = Vec::new();
    let mut of_reg = Vec::new();
    for r in &g.regs {
        let text = r.printed.as_ref().map_or(String::new(), |p| p.csl.clone());
        let i = match names.iter().position(|(_, t)| *t == text) {
            Some(i) => i,
            None => {
                names.push((format!("rgg_{}.csl", names.len()), text));
                names.len() - 1
            }
        };
        of_reg.push(i);
    }
    Mods { names, of_reg }
}

/// A word of a kernel's program: a number, or a slot of its memory (a root's `mem`: its
/// residents, then its values, then its sums and norms' scales), placed once every slot is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wd {
    N(u32),
    Res(u32),
    Val(u32),
    Aux(u32),
}

/// The opcodes of the kernels' programs (`ROOT_VM`, `WEIGHT_VM`).
mod op {
    pub const RECV: u32 = 1;
    pub const CAST: u32 = 2;
    pub const CASTGB: u32 = 3;
    pub const REG: u32 = 4;
    pub const STATS: u32 = 5;
    pub const RING: u32 = 6;
    pub const GATHER: u32 = 7;
    pub const KS: u32 = 8;
    pub const KR: u32 = 9;
    pub const GNF: u32 = 10;
    pub const FETCH: u32 = 1;
    pub const HOLD: u32 = 2;
    pub const OUT: u32 = 3;
    pub const MV: u32 = 4;
}

/// One kernel's program as it is generated.
struct Gen<'g> {
    g: &'g GridProgram,
    who: Who,
    mods: &'g Mods,
    prog: Vec<Wd>,
    consts: Vec<f32>,
    used_mods: Vec<usize>,
    vals: BTreeMap<u32, u32>,
    vwords: u32,
    aux: BTreeMap<(u32, u8), u32>,
    awords: u32,
    res: ResAt,
    h16: u32,
    kv: bool,
    /// The most words a fetch, a cast drained, a tile's rows and columns.
    fb: u32,
    xh: u32,
    y: u32,
    st: u32,
    /// Strides the shuffles read and the unshuffles write with.
    shuf: Vec<(u32, u32)>,
    unsh: Vec<(u32, u32)>,
}

/// Which kernel a program is generated for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Who {
    Root(Side, u32),
    Weight,
}

/// A root's residents: (region, param) and group-norm (gamma, beta) words in its `mem`.
#[derive(Default)]
struct ResAt {
    reg: BTreeMap<(usize, usize), u32>,
    gn: BTreeMap<usize, (u32, u32)>,
    words: u32,
}

impl<'g> Gen<'g> {
    fn new(g: &'g GridProgram, who: Who, mods: &'g Mods) -> Gen<'g> {
        Gen {
            g,
            who,
            mods,
            prog: Vec::new(),
            consts: Vec::new(),
            used_mods: Vec::new(),
            vals: BTreeMap::new(),
            vwords: 0,
            aux: BTreeMap::new(),
            awords: 0,
            res: ResAt::default(),
            h16: 1,
            kv: false,
            fb: 1,
            xh: 1,
            y: 1,
            st: 1,
            shuf: Vec::new(),
            unsh: Vec::new(),
        }
    }

    /// Whether this root holds value `v` (its side's; a head value at root 0 alone).
    fn mine(&self, v: ValueId) -> bool {
        match self.who {
            Who::Root(s, k) => {
                let x = &self.g.vals[&v];
                x.side == s && (!x.head || k == 0)
            }
            Who::Weight => false,
        }
    }

    fn val(&mut self, v: ValueId) -> Wd {
        if let Some(&o) = self.vals.get(&v.0) {
            return Wd::Val(o);
        }
        let o = self.vwords;
        self.vals.insert(v.0, o);
        self.vwords += self.g.chunk(v);
        Wd::Val(o)
    }

    fn aux(&mut self, id: u32, what: u8, n: u32) -> Wd {
        if let Some(&o) = self.aux.get(&(id, what)) {
            return Wd::Aux(o);
        }
        let o = self.awords;
        self.aux.insert((id, what), o);
        self.awords += n;
        Wd::Aux(o)
    }

    fn res_of(&mut self, ri: usize, pk: usize, c: u32) -> Wd {
        if let Some(&o) = self.res.reg.get(&(ri, pk)) {
            return Wd::Res(o);
        }
        let o = self.res.words;
        self.res.reg.insert((ri, pk), o);
        self.res.words += c;
        Wd::Res(o)
    }

    fn gn_res(&mut self, gi: usize, c: u32) -> (Wd, Wd) {
        if let Some(&(a, b)) = self.res.gn.get(&gi) {
            return (Wd::Res(a), Wd::Res(b));
        }
        let o = (self.res.words, self.res.words + c);
        self.res.gn.insert(gi, o);
        self.res.words += 2 * c;
        (Wd::Res(o.0), Wd::Res(o.1))
    }

    fn put(&mut self, w: &[Wd]) {
        self.prog.extend_from_slice(w);
    }

    /// A root's op words for step `s` (pass `pi`); the count of ops.
    #[allow(clippy::too_many_lines)]
    fn root_step(&mut self, pi: usize, s: &Step) -> R<u32> {
        use Wd::N;
        let g = self.g;
        let Who::Root(side, k) = self.who else {
            return Err("a root step off a root".into());
        };
        match s {
            Step::Fetch(v) if self.mine(*v) => {
                let c = g.chunk(*v);
                let d = self.val(*v);
                self.put(&[N(op::RECV), d, N(c), N(0)]);
            }
            Step::Region(ri) => {
                let r = &g.regs[*ri];
                if r.side != side || (r.head && k != 0) {
                    return Ok(0);
                }
                let m = self.mods.of_reg[*ri];
                if !self.used_mods.contains(&m) {
                    self.used_mods.push(m);
                }
                let c = g.chunk(r.outs[0]);
                let mut w = vec![N(op::REG), N(m as u32)];
                for (pk, input) in r.ins.iter().enumerate() {
                    w.push(match input {
                        RegIn::Row(v) => self.val(*v),
                        RegIn::Res(Res::Value { .. }) => self.res_of(*ri, pk, c),
                        RegIn::Res(Res::GnScale(gi)) => self.aux(*gi as u32, 3, c),
                        RegIn::Res(Res::GnShift(gi)) => self.aux(*gi as u32, 4, c),
                    });
                }
                for out in &r.outs {
                    w.push(self.val(*out));
                }
                self.put(&w);
            }
            Step::Matvec(mi) => {
                let mv = &g.mvs[*mi];
                if mv.from == side {
                    let piece = g.piece(*mi, k).len() as u32;
                    if piece == 0 {
                        return Ok(0);
                    }
                    match mv.input {
                        MvIn::Value(x) => {
                            let d = self.val(x);
                            self.put(&[N(op::CAST), d, N(piece)]);
                        }
                        MvIn::Gathered { .. } => self.put(&[N(op::CASTGB), N(piece)]),
                    }
                } else {
                    let n = g.slice(*mi, k).len() as u32;
                    if n == 0 {
                        return Ok(0);
                    }
                    let d = self.val(mv.output);
                    let rnd = u32::from(g.vals[&mv.output].round);
                    self.h16 = self.h16.max(n);
                    self.put(&[N(op::RECV), d, N(n), N(rnd)]);
                }
            }
            Step::Hold(v) if self.mine(*v) => {
                let c = g.chunk(*v);
                let d = self.val(*v);
                self.put(&[N(op::CAST), d, N(c)]);
            }
            Step::Out(v) if self.mine(*v) => {
                let n = g.real(*v, k);
                if n == 0 {
                    return Ok(0);
                }
                let d = self.val(*v);
                self.put(&[N(op::CAST), d, N(n)]);
            }
            Step::Stats(v) if self.mine(*v) => {
                let c = g.chunk(*v);
                self.st = self.st.max(c);
                let d = self.val(*v);
                let (x0, sd, sq) = (
                    self.aux(v.0, 0, c),
                    self.aux(v.0, 1, c),
                    self.aux(v.0, 2, c),
                );
                self.put(&[N(op::STATS), d, N(c), x0, sd, sq]);
            }
            Step::Ring(v) if self.mine(*v) => {
                let gt = g.passes[pi]
                    .gather
                    .as_ref()
                    .ok_or("a ring with no gather")?;
                let c = g.chunk(*v);
                let d = self.val(*v);
                self.put(&[N(op::RING), d, N(c), N(gt.ring - 1)]);
            }
            Step::Gather => {
                let gt = g.passes[pi]
                    .gather
                    .as_ref()
                    .ok_or("a gather with no gather")?;
                if gt.side != side {
                    return Ok(0);
                }
                self.put(&[N(op::GATHER)]);
            }
            Step::Store { q, k: kk, v } if side == Side::Col && k == 0 => {
                for x in [q, kk, v] {
                    let n = g.vals[x].width;
                    let d = self.val(*x);
                    self.put(&[N(op::KS), d, N(n)]);
                }
                self.kv = true;
                return Ok(3);
            }
            Step::Attend(v) if side == Side::Col && k == 0 => {
                let n = g.vals[v].width;
                let d = self.val(*v);
                let rnd = u32::from(g.vals[v].round);
                self.h16 = self.h16.max(n);
                self.put(&[N(op::KR), d, N(n), N(rnd)]);
                self.kv = true;
            }
            _ => return Ok(0),
        }
        Ok(1)
    }

    /// A weight PE's op words for step `s`; the count of ops.
    fn weight_step(&mut self, s: &Step) -> u32 {
        use Wd::N;
        let g = self.g;
        let side = |v: ValueId| u32::from(g.vals[&v].side == Side::Col);
        match s {
            Step::Fetch(v) => {
                let c = g.chunk(*v);
                self.fb = self.fb.max(c);
                let (slot, hin) = match g.slots.hin.get(&v.0) {
                    Some(&o) => (o, 1),
                    None => (g.slots.store[&v.0], 0),
                };
                self.put(&[N(op::FETCH), N(side(*v)), N(slot), N(c), N(hin)]);
            }
            Step::Hold(v) => {
                let c = g.chunk(*v);
                self.fb = self.fb.max(c);
                self.put(&[N(op::HOLD), N(side(*v)), N(g.slots.store[&v.0]), N(c)]);
            }
            Step::Out(v) => {
                let c = g.chunk(*v);
                self.fb = self.fb.max(c);
                let n = g.outs.iter().position(|o| o == v).unwrap_or(0) as u32;
                self.put(&[N(op::OUT), N(side(*v)), N(g.slots.hout[&v.0]), N(n)]);
            }
            Step::Matvec(mi) => {
                let mv = &g.mvs[*mi];
                let (off, kc, nc) = g.tiles[*mi];
                self.xh = self.xh.max(kc);
                self.fb = self.fb.max(kc);
                self.y = self.y.max(nc);
                let head = match mv.from {
                    Side::Row => g.vals[&mv.output].head,
                    Side::Col => matches!(mv.input, MvIn::Value(x) if g.vals[&x].head),
                };
                let dir = u32::from(mv.from == Side::Col);
                self.put(&[N(op::MV), N(dir), N(off), N(kc), N(nc), N(u32::from(head))]);
            }
            _ => return 0,
        }
        1
    }

    /// A group norm's finalize op words at the end of pass `pi` (its input on this root); the
    /// count of ops.
    fn finals(&mut self, pi: usize) -> u32 {
        use Wd::N;
        let g = self.g;
        let mut n = 0;
        for &gi in &g.passes[pi].finals {
            let gn = &g.gns[gi];
            if !self.mine(gn.x) {
                continue;
            }
            let c = g.chunk(gn.x);
            let cpg = g.vals[&gn.x].width / gn.groups.max(1);
            let rows = g.vals[&gn.x].rows();
            let (ga, be) = self.gn_res(gi, c);
            let id = gn.x.0;
            let (x0, sd, sq) = (self.aux(id, 0, c), self.aux(id, 1, c), self.aux(id, 2, c));
            let (gs, gb) = (self.aux(gi as u32, 3, c), self.aux(gi as u32, 4, c));
            let kidx = self.consts.len() as u32;
            self.consts
                .extend([1.0 / rows as f32, 1.0 / cpg as f32, gn.eps]);
            self.put(&[
                N(op::GNF),
                N(c),
                N(cpg),
                x0,
                sd,
                sq,
                ga,
                be,
                gs,
                gb,
                N(kidx),
            ]);
            n += 1;
        }
        n
    }

    /// Pass `pi`'s header words (its `at`s and counts patched once its ops are generated).
    fn header(&mut self, pi: usize) -> usize {
        use Wd::N;
        let g = self.g;
        let p = &g.passes[pi];
        let at = self.prog.len();
        let mut h = vec![N(p.rows()), N(p.grid[2]), N(p.src_rows())];
        let mut gp: Vec<Wd> = Vec::new();
        match p.gather.as_ref() {
            None => h.push(N(0)),
            Some(gt) => {
                let cs = g.chunk(gt.src);
                let ws = gt.src_grid[2];
                let rm = gt.ring - 1;
                match gt.kind {
                    Kind::Conv { k, s, p, up } => {
                        h.push(N(1));
                        let [_, hi, wi] = gt.in_grid;
                        gp.extend(
                            [
                                k[0],
                                k[1],
                                s[0],
                                s[1],
                                p[0],
                                p[1],
                                hi,
                                wi,
                                up.trailing_zeros(),
                                ws,
                                rm,
                                cs,
                            ]
                            .map(N),
                        );
                    }
                    Kind::Up(f) => {
                        h.push(N(2));
                        let d = if self.who == Who::Weight {
                            N(0)
                        } else {
                            self.val(gt.out)
                        };
                        gp.extend([
                            N(f.trailing_zeros()),
                            N(ws),
                            N(rm),
                            N(cs),
                            d,
                            N(g.chunk(gt.out)),
                        ]);
                    }
                    Kind::Shuffle(r) => {
                        h.push(N(3));
                        let d = if self.who == Who::Weight {
                            N(0)
                        } else {
                            self.val(gt.out)
                        };
                        let c = g.chunk(gt.out);
                        if !self.shuf.contains(&(r * r, c)) {
                            self.shuf.push((r * r, c));
                        }
                        gp.extend([
                            N(r.trailing_zeros()),
                            N(ws),
                            N(rm),
                            N(cs),
                            d,
                            N(c),
                            N(r * r),
                        ]);
                    }
                    Kind::Unshuffle(r) => {
                        h.push(N(4));
                        let d = if self.who == Who::Weight {
                            N(0)
                        } else {
                            self.val(gt.out)
                        };
                        if !self.unsh.contains(&(r * r, cs)) {
                            self.unsh.push((r * r, cs));
                        }
                        gp.extend([N(r), N(ws), N(rm), N(cs), d, N(r * r)]);
                    }
                }
            }
        }
        gp.resize(12, N(0));
        h.extend(gp);
        h.extend([N(0); 7]);
        self.put(&h);
        at
    }

    /// The program: every pass's header and ops.
    fn passes(&mut self) -> R<()> {
        for pi in 0..self.g.passes.len() {
            let at = self.header(pi);
            let p = &self.g.passes[pi];
            let (head, tail) = (p.head.clone(), p.tail.clone());
            let h_at = self.prog.len();
            let mut nh = 0;
            for s in &head {
                nh += match self.who {
                    Who::Weight => self.weight_step(s),
                    Who::Root(..) => self.root_step(pi, s)?,
                };
            }
            let t_at = self.prog.len();
            let mut nt = 0;
            for s in &tail {
                nt += match self.who {
                    Who::Weight => self.weight_step(s),
                    Who::Root(..) => self.root_step(pi, s)?,
                };
            }
            let f_at = self.prog.len();
            let nf = match self.who {
                Who::Weight => 0,
                Who::Root(..) => self.finals(pi),
            };
            let next = self.prog.len();
            for (o, v) in [
                h_at,
                nh as usize,
                t_at,
                nt as usize,
                f_at,
                nf as usize,
                next,
            ]
            .into_iter()
            .enumerate()
            {
                self.prog[at + 16 + o] = Wd::N(v as u32);
            }
        }
        Ok(())
    }

    /// The program's words, every slot placed: `mem` is [residents | values | sums].
    fn words(&self) -> R<Vec<u32>> {
        let (rw, vw) = (self.res.words, self.vwords);
        self.prog
            .iter()
            .map(|w| {
                let x = match *w {
                    Wd::N(n) => n,
                    Wd::Res(o) => o,
                    Wd::Val(o) => rw + o,
                    Wd::Aux(o) => rw + vw + o,
                };
                if x > u32::from(u16::MAX) {
                    return Err(format!("a program word {x} past 16 bits"));
                }
                Ok(x)
            })
            .collect()
    }

    fn mem_words(&self) -> u32 {
        (self.res.words + self.vwords + self.awords).max(1)
    }
}

/// The interpreter every root runs over its program (`prog`): each pass's sweep, its ops a row
/// (`ROOT_OPS`), its finals (a group norm's scale and shift from its input's sums).
const SWEEP: &str = "fn shr(v: u16, n: u16) u16 { if (n == 0) { return v; } var q = v; var l: u16 = 0; while (l < n) : (l += 1) { q = q >> 1; } return q; }
fn nconv(y: u16, x: u16, kh: u16, kw: u16, sh: u16, sw: u16, ph: u16, pw: u16, hi: u16, wi: u16, lu: u16, ws: u16) u16 {
  var iy: i16 = @as(i16, y * sh + kh - 1) - @as(i16, ph);
  if (iy > @as(i16, hi) - 1) { iy = @as(i16, hi) - 1; }
  var ix: i16 = @as(i16, x * sw + kw - 1) - @as(i16, pw);
  if (ix > @as(i16, wi) - 1) { ix = @as(i16, wi) - 1; }
  if (iy < 0 or ix < 0) { return 0; }
  return shr(@as(u16, iy), lu) * ws + shr(@as(u16, ix), lu) + 1;
}
fn need(a: u16, y: u16, x: u16) u16 {
  const gk = prog[a + 3];
  if (gk == 1) { return nconv(y, x, prog[a + 4], prog[a + 5], prog[a + 6], prog[a + 7], prog[a + 8], prog[a + 9], prog[a + 10], prog[a + 11], prog[a + 12], prog[a + 13]); }
  if (gk == 4) { const r = prog[a + 4]; return (y * r + r - 1) * prog[a + 5] + x * r + r; }
  return shr(y, prog[a + 4]) * prog[a + 5] + shr(x, prog[a + 4]) + 1;
}
fn sweep(a: u16) void {
  const rows = prog[a]; const w = prog[a + 1]; const srows = prog[a + 2]; const gk = prog[a + 3];
  var y: u16 = 0; var x: u16 = 0; var s: u16 = 0; var r: u16 = 0;
  while (r < rows) : (r += 1) {
    if (gk != 0) { const nd = need(a, y, x); while (s < nd) : (s += 1) { ops(a, prog[a + 16], prog[a + 17], s, y, x); } }
    ops(a, prog[a + 18], prog[a + 19], r, y, x);
    x += 1;
    if (x == w) { x = 0; y += 1; }
  }
  if (gk != 0) { while (s < srows) : (s += 1) { ops(a, prog[a + 16], prog[a + 17], s, y, x); } }
  finals(prog[a + 20], prog[a + 21]);
}
fn go_all() void {
  var a: u16 = 0; var p: u16 = 0;
  while (p < NPASS) : (p += 1) { sweep(a); a = prog[a + 22]; }
}
";

/// [`SWEEP`] of a program in weight groups: a launch runs the passes `grp` names (first,
/// count), the host having written their tiles; `grp` exported for it.
fn sweep_of(g: &GridProgram) -> String {
    if g.groups.is_empty() {
        return SWEEP.to_string();
    }
    let all = "fn go_all() void {\n  var a: u16 = 0; var p: u16 = 0;\n  while (p < NPASS) : (p += 1) { sweep(a); a = prog[a + 22]; }\n}\n";
    let some = "var grp = @zeros([2]f32);\nvar grp_p: [*]f32 = &grp;\nfn go_all() void {\n  var a: u16 = 0; var p: u16 = 0;\n  const p0 = @as(u16, grp[0]); const pn = @as(u16, grp[1]);\n  while (p < p0) : (p += 1) { a = prog[a + 22]; }\n  p = 0;\n  while (p < pn) : (p += 1) { sweep(a); a = prog[a + 22]; }\n}\n";
    SWEEP.replace(all, some)
}

/// A grouped program's pass range at `c`, a group's before its launch (none in one group).
fn grp_arrays(g: &GridProgram, c: Coord, arrays: &mut Vec<GridArray>) {
    for (k, &(f, n)) in g.groups.iter().enumerate() {
        arrays.push(GridArray {
            at: c,
            sym: "grp".into(),
            u16: false,
            len: 2,
            // (as f32s: every exported array a root or weight PE takes is a word of one)
            fill: GridFill::Words(vec![(f as f32).to_bits(), (n as f32).to_bits()]),
            group: Some(k),
        });
    }
}

/// The export of a grouped program's pass range.
fn grp_export(g: &GridProgram) -> &'static str {
    if g.groups.is_empty() {
        ""
    } else {
        "  @export_symbol(grp_p, \"grp\");\n"
    }
}

/// A root's ops over `mem` (`M(o, n)`: n words from o), the line buffer and the gather.
const ROOT_OPS: &str = "fn rnd(o: u16, n: u16) void {
  const d = @set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, o), f32), n);
  @fs2h(@set_dsd_length(H16, n), d); @fh2s(d, @set_dsd_length(H16, n));
}
fn gconv(a: u16, y: u16, x: u16) void {
  const kh = prog[a + 4]; const kw = prog[a + 5];
  const hi: i16 = @as(i16, prog[a + 10]); const wi: i16 = @as(i16, prog[a + 11]);
  const lu = prog[a + 12]; const ws = prog[a + 13]; const rm = prog[a + 14]; const cs = prog[a + 15];
  const iy0: i16 = @as(i16, y * prog[a + 6]) - @as(i16, prog[a + 8]);
  const ix0: i16 = @as(i16, x * prog[a + 7]) - @as(i16, prog[a + 9]);
  var dst = @set_dsd_length(D_gb, cs);
  var i: u16 = 0;
  while (i < kh) : (i += 1) {
    const iy: i16 = iy0 + @as(i16, i);
    const rowin = iy >= 0 and iy < hi;
    const sy: u16 = if (rowin) shr(@as(u16, iy), lu) * ws else 0;
    var j: u16 = 0;
    while (j < kw) : (j += 1) {
      const ix: i16 = ix0 + @as(i16, j);
      if (rowin and ix >= 0 and ix < wi) {
        const srow: u16 = sy + shr(@as(u16, ix), lu);
        @fmovs(dst, @set_dsd_length(@increment_dsd_offset(D_ring, @as(i16, (srow & rm) * cs), f32), cs));
      } else {
        @fmovs(dst, 0.0);
      }
      dst = @increment_dsd_offset(dst, @as(i16, cs), f32);
    }
  }
}
fn stats(i: u16, t: u16) void {
  const c = prog[i + 2];
  const d = @set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, prog[i + 1]), f32), c);
  const x0 = @set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, prog[i + 3]), f32), c);
  const sd = @set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, prog[i + 4]), f32), c);
  const sq = @set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, prog[i + 5]), f32), c);
  const st = @set_dsd_length(D_st, c);
  if (t == 0) { @fmovs(x0, d); @fmovs(sd, 0.0); @fmovs(sq, 0.0); }
  @fsubs(st, d, x0); @fadds(sd, sd, st); @fmuls(st, st, st); @fadds(sq, sq, st);
}
fn finals(at: u16, n: u16) void {
  var i = at; var k: u16 = 0;
  while (k < n) : (k += 1) {
    const c = prog[i + 1]; const cpg = prog[i + 2];
    const x0 = prog[i + 3]; const sd = prog[i + 4]; const sq = prog[i + 5];
    const ga = prog[i + 6]; const be = prog[i + 7]; const gs = prog[i + 8]; const gb = prog[i + 9];
    const kc = prog[i + 10];
    const inv = fk[kc]; const icpg = fk[kc + 1]; const eps = fk[kc + 2];
    var e: u16 = 0;
    while (e < c) : (e += 1) { const m = mem[sd + e] * inv; gmc[e] = mem[x0 + e] + m; gvc[e] = mem[sq + e] * inv - m * m; }
    var q: u16 = 0;
    while (q < c) : (q += cpg) {
      var mg: f32 = 0.0; e = 0;
      while (e < cpg) : (e += 1) { mg += gmc[q + e]; }
      mg = mg * icpg;
      var vg: f32 = 0.0; e = 0;
      while (e < cpg) : (e += 1) { const dd = gmc[q + e] - mg; vg += gvc[q + e] + dd * dd; }
      const rs = math.rsqrt(vg * icpg + eps);
      e = 0;
      while (e < cpg) : (e += 1) { const av = rs * mem[ga + q + e]; mem[gs + q + e] = av; mem[gb + q + e] = mem[be + q + e] - mg * av; }
    }
    i += 11;
  }
}
";

/// The weight PE's ops: a held row fetched up its chain, a row kept off its cast, an output
/// kept for the host, a tile matvec into its chain.
const WEIGHT_OPS: &str = "fn ops(a: u16, at: u16, n: u16, t: u16, y: u16, x: u16) void {
  var i = at; var k: u16 = 0;
  while (k < n) : (k += 1) {
    const op = prog[i];
    if (op == 1) {
      const sd = prog[i + 1]; const slot = prog[i + 2]; const c = prog[i + 3];
      const fbc = @set_dsd_length(D_fb, c);
      const mine = if (sd == 0) ((t & PM) == ci) else ((t & PM) == rj);
      if (mine) {
        if (prog[i + 4] != 0) { @fmovs(fbc, @set_dsd_length(@increment_dsd_offset(D_hin, @as(i16, slot + (t >> PSH) * c), f32), c)); }
        else { @fh2s(fbc, @set_dsd_length(@increment_dsd_offset(D_ss, @as(i16, slot + (t >> PSH) * c), @fp16()), c)); }
      } else { @fmovs(fbc, 0.0); }
      if (sd == 0) { rw.role = rrole; rw.send(fbc, c); } else { rs.role = crole; rs.send(fbc, c); }
      i += 5;
    } else if (op == 2) {
      const sd = prog[i + 1]; const slot = prog[i + 2]; const c = prog[i + 3];
      const dst = @set_dsd_length(@increment_dsd_offset(D_ss, @as(i16, slot + (t >> PSH) * c), @fp16()), c);
      if (sd == 0) {
        if ((t & PM) == ci) { @fs2h(dst, fab.fin(q_xr, c)); } else { @fmovs(@set_dsd_length(D_fb, c), fab.fin(q_xr, c)); }
      } else {
        if ((t & PM) == rj) { @fs2h(dst, fab.fin(q_xc, c)); } else { @fmovs(@set_dsd_length(D_fb, c), fab.fin(q_xc, c)); }
      }
      i += 4;
    } else if (op == 3) {
      const sd = prog[i + 1]; const slot = prog[i + 2]; const c = ro[prog[i + 3]];
      if (c > 0) {
        const dst = @set_dsd_length(@increment_dsd_offset(D_hout, @as(i16, slot + (t >> PSH) * c), f32), c);
        if (sd == 0) {
          if ((t & PM) == ci) { @fmovs(dst, fab.fin(q_xr, c)); } else { @fmovs(@set_dsd_length(D_fb, c), fab.fin(q_xr, c)); }
        } else {
          if ((t & PM) == rj) { @fmovs(dst, fab.fin(q_xc, c)); } else { @fmovs(@set_dsd_length(D_fb, c), fab.fin(q_xc, c)); }
        }
      }
      i += 4;
    } else if (op == 4) {
      const dir = prog[i + 1]; const off = prog[i + 2]; const kc = prog[i + 3]; const nc = prog[i + 4]; const hd = prog[i + 5];
      const xv = @set_dsd_length(D_xh, kc); const yv = @set_dsd_length(D_y, nc);
      const wv = @set_dsd_length(@increment_dsd_offset(D_w, @as(i16, off), u16), nc);
      if (dir == 0) {
        if (hd == 0 or ci == 0) {
          @fs2h(xv, fab.fin(q_xr, kc)); matvec.tile32(yv, wv, xv);
          rs.role = crole; rs.send(yv, nc);
        } else { @fmovs(@set_dsd_length(D_fb, kc), fab.fin(q_xr, kc)); }
      } else {
        if (hd == 0 or ci == 0) {
          @fs2h(xv, fab.fin(q_xc, kc)); matvec.tile32(yv, wv, xv);
          rw.role = if (hd == 0) rrole else 0; rw.send(yv, nc);
        }
      }
      i += 6;
    } else { i += 1; }
  }
}
fn finals(at: u16, n: u16) void { }
";

/// The region dispatcher of a root (each module its arm: its parameters copied in from `mem`,
/// `row()`, its results out), and the shuffles' strided reads.
fn reg_fn(g: &GridProgram, mods: &Mods, used: &[usize]) -> String {
    let mut o = String::from("fn reg(m: u16, a: u16) void {\n");
    for &m in used {
        let (ri, _) = mods
            .of_reg
            .iter()
            .enumerate()
            .find(|(_, x)| **x == m)
            .expect("a module's region");
        let r = &g.regs[ri];
        let c = g.chunk(r.outs[0]);
        let _ = writeln!(o, "  if (m == {m}) {{");
        for k in 0..r.ins.len() {
            let _ = writeln!(
                o,
                "    @fmovs(rm{m}.IN{k}, @set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, prog[a + {k}]), f32), {c}));"
            );
        }
        let _ = writeln!(o, "    rm{m}.row();");
        for (oi, _) in r.outs.iter().enumerate() {
            let _ = writeln!(
                o,
                "    @fmovs(@set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, prog[a + {}]), f32), {c}), rm{m}.OUT{oi});",
                r.ins.len() + oi
            );
        }
        o.push_str("    return;\n  }\n");
    }
    o.push_str("}\nfn regn(m: u16) u16 {\n");
    for &m in used {
        let (ri, _) = mods
            .of_reg
            .iter()
            .enumerate()
            .find(|(_, x)| **x == m)
            .expect("a module's region");
        let r = &g.regs[ri];
        let _ = writeln!(
            o,
            "  if (m == {m}) {{ return {}; }}",
            r.ins.len() + r.outs.len()
        );
    }
    o.push_str("  return 0;\n}\n");
    o
}

/// The root's op loop and gather (its cast queue `q_x`, chain in `q_ci`).
fn root_ops(kg: &Gen) -> String {
    let mut o = String::from(
        "fn gather(a: u16, y: u16, x: u16) void {\n  const gk = prog[a + 3];\n  if (gk == 1) { gconv(a, y, x); return; }\n  \
         const l = prog[a + 4]; const ws = prog[a + 5]; const rm = prog[a + 6]; const cs = prog[a + 7]; const dst = prog[a + 8];\n",
    );
    o.push_str(
        "  if (gk == 2) { const srow = shr(y, l) * ws + shr(x, l); const c = prog[a + 9];\n    \
         @fmovs(@set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, dst), f32), c), @set_dsd_length(@increment_dsd_offset(D_ring, @as(i16, (srow & rm) * cs), f32), c)); return; }\n",
    );
    for &(rr, c) in &kg.shuf {
        let _ = writeln!(
            o,
            "  if (gk == 3 and prog[a + 10] == {rr} and prog[a + 9] == {c}) {{ const m = (@as(u16, 1) << l) - 1; const srow = shr(y, l) * ws + shr(x, l); const kk = ((y & m) << l) + (x & m);\n    \
             @fmovs(@set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, dst), f32), {c}), @increment_dsd_offset(D_rs{rr}_{c}, @as(i16, (srow & rm) * cs + kk), f32)); return; }}"
        );
    }
    for &(rr, cs) in &kg.unsh {
        let _ = writeln!(
            o,
            "  if (gk == 4 and prog[a + 9] == {rr} and prog[a + 7] == {cs}) {{ const r = prog[a + 4]; var i: u16 = 0;\n    \
             while (i < r) : (i += 1) {{ var j: u16 = 0; while (j < r) : (j += 1) {{ const srow = (y * r + i) * ws + x * r + j;\n      \
             @fmovs(@increment_dsd_offset(D_us{rr}_{cs}, @as(i16, dst + i * r + j), f32), @set_dsd_length(@increment_dsd_offset(D_ring, @as(i16, (srow & rm) * {cs}), f32), {cs})); }} }} return; }}"
        );
    }
    o.push_str("}\n");
    o.push_str(
        "fn ops(a: u16, at: u16, n: u16, t: u16, y: u16, x: u16) void {\n  var i = at; var k: u16 = 0;\n  while (k < n) : (k += 1) {\n    const op = prog[i];\n    \
         if (op == 1) { const o = prog[i + 1]; const c = prog[i + 2]; @fmovs(@set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, o), f32), c), fab.fin(q_ci, c)); if (prog[i + 3] != 0) { rnd(o, c); } i += 4; }\n    \
         else if (op == 2) { const c = prog[i + 2]; @fmovs(fab.fout(q_x, c), @set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, prog[i + 1]), f32), c)); i += 3; }\n    \
         else if (op == 3) { const c = prog[i + 1]; @fmovs(fab.fout(q_x, c), @set_dsd_length(D_gb, c)); i += 2; }\n    \
         else if (op == 4) { reg(prog[i + 1], i + 2); i += 2 + regn(prog[i + 1]); }\n    \
         else if (op == 5) { stats(i, t); i += 6; }\n    \
         else if (op == 6) { const c = prog[i + 2]; @fmovs(@set_dsd_length(@increment_dsd_offset(D_ring, @as(i16, (t & prog[i + 3]) * c), f32), c), @set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, prog[i + 1]), f32), c)); i += 4; }\n    \
         else if (op == 7) { gather(a, y, x); i += 1; }\n",
    );
    if kg.kv {
        o.push_str(
            "    else if (op == 8) { const c = prog[i + 2]; @fmovs(fab.fout(q_ks, c), @set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, prog[i + 1]), f32), c)); i += 3; }\n    \
             else if (op == 9) { const o = prog[i + 1]; const c = prog[i + 2]; @fmovs(@set_dsd_length(@increment_dsd_offset(D_mem, @as(i16, o), f32), c), fab.fin(q_kr, c)); if (prog[i + 3] != 0) { rnd(o, c); } i += 4; }\n",
        );
    }
    o.push_str("    else { i += 1; }\n  }\n}\n");
    o
}

/// The program's words as a CSL array, and the finals' constants.
fn tables(words: &[u32], consts: &[f32], npass: usize) -> String {
    let mut o = format!(
        "const NPASS: u16 = {npass};\nconst prog = [{}]u16 {{ ",
        words.len().max(1)
    );
    if words.is_empty() {
        o.push('0');
    }
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        let _ = write!(o, "{w}");
    }
    o.push_str(" };\n");
    let _ = write!(o, "const fk = [{}]f32 {{ ", consts.len().max(1));
    if consts.is_empty() {
        o.push_str("0.0");
    }
    for (i, c) in consts.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        let _ = write!(o, "{c:?}");
    }
    o.push_str(" };\n");
    o
}

/// A root kernel (side `side`, root `k`): its file and its residents' places.
fn root_file(g: &GridProgram, mods: &Mods, side: Side, k: u32) -> R<(String, String, ResAt)> {
    let mut kg = Gen::new(g, Who::Root(side, k), mods);
    kg.passes()?;
    let words = kg.words()?;
    let name = match side {
        Side::Row => format!("gx_{k}.csl"),
        Side::Col => format!("gy_{k}.csl"),
    };
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// {}: {} root {k} (kernels-cerebras emit::grid): the passes' program over its memory.",
        g.reading,
        match side {
            Side::Row => "CX",
            Side::Col => "CY",
        }
    );
    t.push_str("param memcpy_params;\nparam C_CI: u16;\nparam C_X: u16;\n");
    if kg.kv {
        t.push_str("param C_KS: u16;\nparam C_KR: u16;\n");
    }
    t.push_str(COMMON);
    t.push_str("const q_ci = @get_input_queue(2);\nconst q_x = @get_output_queue(2);\n");
    if kg.kv {
        t.push_str("const q_kr = @get_input_queue(3);\nconst q_ks = @get_output_queue(3);\n");
    }
    for &m in &kg.used_mods {
        let _ = writeln!(
            t,
            "const rm{m} = @import_module(\"{}\", .{{ .math = math, .seg = math }});",
            mods.names[m].0
        );
    }
    t.push_str(&tables(&words, &kg.consts, g.passes.len()));
    let ring = g
        .passes
        .iter()
        .filter_map(|p| {
            p.gather
                .as_ref()
                .filter(|gt| gt.side == side)
                .map(|gt| gt.ring * g.chunk(gt.src))
        })
        .max()
        .unwrap_or(1);
    t.push_str(&buf("ring", ring));
    for &(rr, c) in &kg.shuf {
        let _ = writeln!(
            t,
            "const D_rs{rr}_{c} = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{c}}} -> ring[i * {rr}] }});"
        );
    }
    let gb = g
        .mvs
        .iter()
        .filter(|m| m.from == side)
        .map(|m| match m.input {
            MvIn::Gathered { taps, c_in } => taps * c_in / g.p,
            MvIn::Value(_) => 0,
        })
        .max()
        .unwrap_or(1)
        .max(1);
    t.push_str(&buf("gb", gb));
    t.push_str(&buf("st", kg.st));
    let gm = g.gns.iter().map(|gn| g.chunk(gn.x)).max().unwrap_or(1);
    let _ = writeln!(
        t,
        "var gmc = @zeros([{gm}]f32);\nvar gvc = @zeros([{gm}]f32);"
    );
    let memw = kg.mem_words();
    let _ = writeln!(t, "{}var mem_p: [*]f32 = &mem;", buf("mem", memw));
    for &(rr, cs) in &kg.unsh {
        let _ = writeln!(
            t,
            "const D_us{rr}_{cs} = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{cs}}} -> mem[i * {rr}] }});"
        );
    }
    let _ = writeln!(
        t,
        "var h16 = @zeros([{0}]@fp16());\nconst H16 = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{0}}} -> h16[i] }});",
        kg.h16
    );
    t.push_str(&sweep_of(g));
    t.push_str(ROOT_OPS);
    t.push_str(&reg_fn(g, mods, &kg.used_mods));
    t.push_str(&root_ops(&kg));
    let mut body = String::from("  stamp(0);\n");
    for &m in &kg.used_mods {
        let _ = writeln!(body, "  rm{m}.init();");
    }
    body.push_str("  go_all();\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n");
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(mem_p, \"res\");\n");
    t.push_str(grp_export(g));
    t.push_str("  @initialize_queue(q_ci, .{ .color = @get_color(C_CI) });\n  @initialize_queue(q_x, .{ .color = @get_color(C_X) });\n");
    if kg.kv {
        t.push_str("  @initialize_queue(q_kr, .{ .color = @get_color(C_KR) });\n  @initialize_queue(q_ks, .{ .color = @get_color(C_KS) });\n");
    }
    t.push_str("}\n");
    let mut res = std::mem::take(&mut kg.res);
    res.words = memw;
    Ok((name, t, res))
}

/// The weight PEs' kernel (one file; a PE's place and roles its config words).
fn weight_file(g: &GridProgram, mods: &Mods) -> R<(String, String, u32)> {
    let mut kg = Gen::new(g, Who::Weight, mods);
    kg.passes()?;
    let words = kg.words()?;
    let name = "gw.csl".to_string();
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// {}: the unit's weight PEs (kernels-cerebras emit::grid): every matvec's tile, the held rows'\n// store, the host's rows in and out; the passes' program.",
        g.reading
    );
    t.push_str("param memcpy_params;\nparam C_XR: u16;\nparam C_XC: u16;\nparam C_CS: u16;\nparam C_CR: u16;\nparam C_RS: u16;\nparam C_RR: u16;\n");
    t.push_str(COMMON);
    t.push_str("const q_xr = @get_input_queue(2);\nconst q_xc = @get_input_queue(3);\nconst q_cr = @get_input_queue(4);\nconst q_rr = @get_input_queue(5);\nconst q_cs = @get_output_queue(2);\nconst q_rs = @get_output_queue(3);\n");
    t.push_str("const rs = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q_cr, .OUT = q_cs });\n");
    t.push_str("const rw = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q_rr, .OUT = q_rs });\n");
    t.push_str("const matvec = @import_module(\"ops/mv.csl\");\n");
    let _ = writeln!(
        t,
        "const PM: u16 = {};\nconst PSH: u16 = {};",
        g.p - 1,
        g.p.trailing_zeros()
    );
    t.push_str(&tables(&words, &[], g.passes.len()));
    let wlen = g
        .tiles
        .iter()
        .map(|(o, k, n)| o + k * n)
        .max()
        .unwrap_or(1)
        .max(1);
    let _ = writeln!(
        t,
        "var w = @zeros([{wlen}]u16);\nvar w_p: [*]u16 = &w;\nconst D_w = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{wlen}}} -> w[i] }});"
    );
    let ss = g.slots.store_words.max(1);
    let _ = writeln!(
        t,
        "var ss = @zeros([{ss}]@fp16());\nconst D_ss = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{ss}}} -> ss[i] }});"
    );
    let hin = g.slots.hin_words.max(1);
    let hout = g.slots.hout_words.max(1);
    let _ = writeln!(
        t,
        "{}var hin_p: [*]f32 = &hin;\n{}var hout_p: [*]f32 = &hout;",
        buf("hin", hin),
        buf("hout", hout)
    );
    let _ = writeln!(
        t,
        "var xh = @zeros([{0}]@fp16());\nconst D_xh = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{0}}} -> xh[i] }});",
        kg.xh
    );
    t.push_str(&buf("y", kg.y));
    t.push_str(&buf("fb", kg.fb));
    let nout = g.outs.len() as u32;
    // pcfg: column, row, the row chain's role, the column chain's, each output's real columns
    t.push_str(&cfg_decl(&vec![0; 4 + nout as usize]));
    let _ = writeln!(
        t,
        "var ci: u16 = 0;\nvar rj: u16 = 0;\nvar rrole: u16 = 0;\nvar crole: u16 = 0;\nvar ro = @zeros([{}]u16);",
        nout.max(1)
    );
    t.push_str(&sweep_of(g));
    t.push_str(WEIGHT_OPS);
    let mut body = String::from(
        "  stamp(0);\n  ci = @as(u16, pcfg[0]);\n  rj = @as(u16, pcfg[1]);\n  rrole = @as(u16, pcfg[2]);\n  crole = @as(u16, pcfg[3]);\n",
    );
    for n in 0..nout {
        let _ = writeln!(body, "  ro[{n}] = @as(u16, pcfg[{}]);", 4 + n);
    }
    body.push_str("  go_all();\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n");
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n  @export_symbol(w_p, \"w\");\n  @export_symbol(hin_p, \"hin\");\n  @export_symbol(hout_p, \"hout\");\n");
    t.push_str(grp_export(g));
    t.push_str("  @initialize_queue(q_xr, .{ .color = @get_color(C_XR) });\n  @initialize_queue(q_xc, .{ .color = @get_color(C_XC) });\n  @initialize_queue(q_cr, .{ .color = @get_color(C_CR) });\n  @initialize_queue(q_rr, .{ .color = @get_color(C_RR) });\n  @initialize_queue(q_cs, .{ .color = @get_color(C_CS) });\n  @initialize_queue(q_rs, .{ .color = @get_color(C_RS) });\n}\n");
    Ok((name, t, wlen))
}

/// The K / V column's root (ops/cy_satt.csl): the store pass's [q | k | v] down the column, then
/// each query's output back to the unit's head root.
fn kv_root_file(g: &GridProgram) -> (String, String) {
    let a = g.attn.as_ref().expect("an attention");
    let (dc, qd) = (a.width, a.width);
    let (lo, hi) = if g.round16 {
        ("1.0e-30", "1.0e30")
    } else {
        ("0.00390625", "32768.0")
    };
    let mut t = String::from(
        "// The K / V column's root (kernels-cerebras emit::grid; ops/cy_satt.csl).\nparam memcpy_params;\nparam C_I0: u16;\nparam C_O0: u16;\nparam C_AR: u16;\nparam C_AM: u16;\n",
    );
    t.push_str(COMMON);
    t.push_str("const q_i0 = @get_input_queue(2);\nconst q_o0 = @get_output_queue(3);\nconst q_ar = @get_input_queue(3);\nconst q_am = @get_output_queue(2);\n");
    let _ = writeln!(
        t,
        "const cy = @import_module(\"ops/cy_satt.csl\", .{{ .fab = fab, .math = math, .YO = q_am, .IN = q_ar, .Qh = 1, .Dc = {dc}, .LO = {lo}, .HI = {hi} }});"
    );
    let sw = qd + 2 * dc;
    t.push_str(&buf("sx", sw));
    t.push_str(&buf("ob", qd));
    let rows = a.rows;
    let (gate, grp) = kv_gate(g);
    t.push_str(grp);
    let _ = write!(
        t,
        "task go() void {{\n{gate}  stamp(0);\n  cy.passes = 0;\n  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ @fmovs(D_sx, fab.fin(q_i0, {sw})); @fmovs(fab.fout(q_am, {sw}), D_sx); }} }}\n  \
         {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ cy.query_q(D_ob); @fmovs(fab.fout(q_o0, {qd}), D_ob); }} }}\n  stamp(1);\n  tsc[2] = cy.passes;\n  sys_mod.unblock_cmd_stream();\n}}\n"
    );
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n");
    t.push_str(grp_export(g));
    t.push_str("  @initialize_queue(q_i0, .{ .color = @get_color(C_I0) });\n  @initialize_queue(q_o0, .{ .color = @get_color(C_O0) });\n  @initialize_queue(q_ar, .{ .color = @get_color(C_AR) });\n  @initialize_queue(q_am, .{ .color = @get_color(C_AM) });\n}\n");
    ("gkr.csl".into(), t)
}

/// The K / V column's attention PEs (ops/satt.csl, the queries held on the column).
fn kv_pe_file(g: &GridProgram) -> (String, String) {
    let a = g.attn.as_ref().expect("an attention");
    let pya = g.stream.kv[0].pya;
    let ne = a.rows.div_ceil(pya);
    let pmax = if g.round16 { "1.0e30" } else { "65504.0" };
    let mut t = String::from(
        "// The K / V column's attention PEs (kernels-cerebras emit::grid; ops/satt.csl).\nparam memcpy_params;\nparam C_AM: u16;\nparam C_S: u16;\nparam C_R: u16;\n",
    );
    t.push_str(COMMON);
    t.push_str(&cfg_decl(&[0, 0]));
    t.push_str("const q_am = @get_input_queue(2);\nconst q_r = @get_input_queue(3);\nconst q_s = @get_output_queue(2);\n");
    t.push_str("const red = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q_r, .OUT = q_s });\n");
    let _ = writeln!(
        t,
        "const att = @import_module(\"ops/satt.csl\", .{{ .fab = fab, .math = math, .red = red, .YM = q_am, .Qh = 1, .Dc = {}, .NE = {ne}, .PYA = {pya}, .SC = {:?}, .PMAX = {pmax}, .QH = true }});",
        a.width, a.scale
    );
    let rows = a.rows;
    let (gate, grp) = kv_gate(g);
    t.push_str(grp);
    let _ = write!(
        t,
        "task go() void {{\n{gate}  stamp(0);\n  att.init(@as(u16, pcfg[0]));\n  red.role = @as(u16, pcfg[1]);\n  \
         {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ att.store_q(t); }} }}\n  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ att.query_q(t); }} }}\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n}}\n"
    );
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n");
    t.push_str(grp_export(g));
    t.push_str("  @initialize_queue(q_am, .{ .color = @get_color(C_AM) });\n  @initialize_queue(q_r, .{ .color = @get_color(C_R) });\n  @initialize_queue(q_s, .{ .color = @get_color(C_S) });\n}\n");
    ("gkp.csl".into(), t)
}

/// A K / V kernel's gate in a grouped program (it runs in the launch of the group holding the
/// attention's stores alone) and its `grp`; empty in one.
fn kv_gate(g: &GridProgram) -> (String, &'static str) {
    let Some(&(f, n)) = g.groups.iter().find(|&&(f, n)| {
        (f..f + n).any(|pi| {
            g.passes[pi]
                .tail
                .iter()
                .chain(&g.passes[pi].head)
                .any(|s| matches!(s, Step::Store { .. }))
        })
    }) else {
        return (String::new(), "");
    };
    let _ = n;
    (
        format!(
            "  if (@as(u16, grp[0]) != {f}) {{ stamp(0); stamp(1); sys_mod.unblock_cmd_stream(); return; }}\n"
        ),
        "var grp = @zeros([2]f32);\nvar grp_p: [*]f32 = &grp;\n",
    )
}

fn filler_file() -> String {
    let mut t = String::from(
        "// An idle PE (kernels-cerebras emit::grid): routes only.\nparam memcpy_params;\n",
    );
    t.push_str(COMMON);
    t.push_str("task go() void { sys_mod.unblock_cmd_stream(); }\ncomptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n}\n");
    t
}

/// The 16-bit word flags of a tile in this build.
fn cast16(g: &GridProgram) -> u32 {
    if g.round16 { BF16 } else { HALF }
}

/// Matvec `m`'s tile at weight PE (`i`, `j`) as spans (`len` words from its offset).
pub fn tile_spans(g: &GridProgram, m: usize, i: u32, j: u32) -> Vec<Span> {
    let mv = &g.mvs[m];
    let (_, kc, nc) = g.tiles[m];
    let (ins, outs) = match mv.from {
        Side::Row => (g.piece(m, j), g.slice(m, i)),
        Side::Col => (g.piece(m, i), g.slice(m, j)),
    };
    let none = |n: u32| [NONE, 0, 0, n, 0, 0, 1, 0];
    let mut s: Vec<Span> = Vec::new();
    let flags = FLAT | cast16(g);
    if mv.ident {
        // the identity: one where the input is the output
        for r in 0..kc {
            let ki = ins.get(r as usize).copied();
            for c in 0..nc {
                let o = outs.get(c as usize).copied().flatten();
                s.push(match (ki, o) {
                    (Some(a), Some(b)) if a == b && a < mv.k => [UNIT, 0, 0, 1, 0, flags, 1, 0],
                    _ => none(1),
                });
            }
        }
        return s;
    }
    for r in 0..kc {
        let ki = ins.get(r as usize).copied().filter(|&k| k < mv.k);
        let mut c = 0u32;
        while c < nc {
            let o = outs.get(c as usize).copied().flatten();
            match (ki, o) {
                (Some(ki), Some(o0)) => {
                    let mut n = 1;
                    while c + n < nc
                        && outs.get((c + n) as usize).copied().flatten() == Some(o0 + n)
                    {
                        n += 1;
                    }
                    s.push([mv.weight.0, 0, o0 * mv.k + ki, n, mv.k, flags, 1, 0]);
                    c += n;
                }
                _ => {
                    s.push(none(1));
                    c += 1;
                }
            }
        }
    }
    s
}

/// `n` zero words.
fn zeros(n: u32) -> Span {
    [NONE, 0, 0, n, 0, 0, 1, 0]
}

/// Merges runs of zero spans.
fn merged(s: Vec<Span>) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    for x in s {
        if let Some(l) = out.last_mut()
            && l[0] == NONE
            && x[0] == NONE
            && l[6] == 1
            && x[6] == 1
        {
            l[3] += x[3];
            continue;
        }
        out.push(x);
    }
    out
}

/// The emitted program.
#[allow(clippy::too_many_lines)]
pub fn emit(g: &GridProgram) -> R<Emitted> {
    let s = &g.stream;
    let u = &s.units[0];
    let p = g.p;
    let mods = mods(g);
    let mut files: Vec<(String, String)> = Vec::new();
    let mut arrays: Vec<GridArray> = Vec::new();
    let mut tiles: Tiles = BTreeMap::new();
    for (n, text) in &mods.names {
        files.push((n.clone(), text.clone()));
    }
    // the weight PEs
    let (wname, wtext, wlen) = weight_file(g, &mods)?;
    files.push((wname.clone(), wtext));
    let outs: Vec<ValueId> = g.outs.clone();
    for i in 0..p {
        for j in 0..p {
            let c = u.w(i, j);
            let (cs, cr, crole) = chain_cols(g, c, COL_CHAIN);
            let (rs, rr, rrole) = chain_cols(g, c, ROW_CHAIN);
            let params = parked(
                g,
                c,
                vec![
                    ("C_XR".into(), colour_of(g, "XR0")),
                    ("C_XC".into(), colour_of(g, "XC0")),
                    ("C_CS".into(), cs),
                    ("C_CR".into(), cr),
                    ("C_RS".into(), rs),
                    ("C_RR".into(), rr),
                ],
            );
            tiles.insert(c, (wname.clone(), params));
            let mut words = vec![i, j, rrole, crole];
            for &v in &outs {
                let (_, chunk_of) = g.keeper(v, i, j);
                words.push(g.real(v, chunk_of));
            }
            arrays.push(GridArray {
                at: c,
                sym: "pcfg".into(),
                u16: false,
                len: words.len() as u32,
                fill: GridFill::Words(words),
                group: None,
            });
            // the tiles (in a grouped program, each group's its own, written before its launch)
            let runs: Vec<(Option<usize>, (usize, usize))> = if g.groups.is_empty() {
                vec![(None, (0, g.passes.len()))]
            } else {
                g.groups
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(k, r)| (Some(k), r))
                    .collect()
            };
            for (group, (f, n)) in runs {
                let in_run = |m: usize| {
                    group.is_none()
                        || (f..f + n).any(|pi| {
                            g.passes[pi]
                                .head
                                .iter()
                                .chain(&g.passes[pi].tail)
                                .any(|s| *s == Step::Matvec(m))
                        })
                };
                let mut sp: Vec<Span> = vec![zeros(1)];
                for (m, (off, kc, nc)) in g.tiles.iter().enumerate() {
                    if !in_run(m) {
                        continue;
                    }
                    let used: u32 = sp.iter().map(|x| x[3] * x[6]).sum();
                    if used < *off {
                        sp.push(zeros(off - used));
                    }
                    if kc * nc > 0 {
                        sp.extend(tile_spans(g, m, i, j));
                    }
                }
                let used: u32 = sp.iter().map(|x| x[3] * x[6]).sum();
                if used < wlen {
                    sp.push(zeros(wlen - used));
                }
                arrays.push(GridArray {
                    at: c,
                    sym: "w".into(),
                    u16: true,
                    len: wlen,
                    fill: GridFill::Spans(merged(sp)),
                    group,
                });
            }
            grp_arrays(g, c, &mut arrays);
            // the host's rows: rows t = i (mod P) of the root's chunk j
            let hin = g.slots.hin_words.max(1);
            let mut ent: Vec<Word> = vec![[NONE, 0, 0, 0]; hin as usize];
            for &v in &g.hostv {
                let slot = g.slots.hin[&v.0];
                let ch = g.chunk(v);
                let width = g.vals[&v].width;
                let rows = g.vals[&v].rows();
                let mut tt = i;
                while tt < rows {
                    for e in 0..ch {
                        let col = j * ch + e;
                        let w = (slot + (tt >> p.trailing_zeros()) * ch + e) as usize;
                        if col < width {
                            ent[w] = [v.0, tt, col, 0];
                        }
                    }
                    tt += p;
                }
            }
            if !g.hostv.is_empty() {
                arrays.push(GridArray {
                    at: c,
                    sym: "hin".into(),
                    u16: false,
                    len: hin,
                    fill: GridFill::Entry(ent),
                    group: None,
                });
            }
            // the outputs, read back
            for &v in &outs {
                let (keeper, chunk_of) = g.keeper(v, i, j);
                let real = g.real(v, chunk_of);
                if real == 0 {
                    continue;
                }
                let rows = g.vals[&v].rows();
                let slot = g.slots.hout[&v.0];
                let kept: Vec<usize> = (keeper..rows)
                    .step_by(p as usize)
                    .map(|r| r as usize)
                    .collect();
                let len = slot + kept.len() as u32 * real;
                arrays.push(GridArray {
                    at: c,
                    sym: "hout".into(),
                    u16: false,
                    len,
                    fill: GridFill::Back {
                        rows: kept,
                        parts: vec![(v.0, chunk_of * g.chunk(v), real)],
                    },
                    group: None,
                });
                if slot != 0 {
                    return Err("two outputs a weight PE: read back one at a time".into());
                }
            }
        }
    }
    // the roots
    for side in [Side::Row, Side::Col] {
        for k in 0..p {
            let (name, text, res) = root_file(g, &mods, side, k)?;
            let kv_root = text.contains("param C_KS");
            files.push((name.clone(), text));
            let c = match side {
                Side::Row => u.cx(k),
                Side::Col => u.cy(k),
            };
            let (slot, x) = match side {
                Side::Row => (ROW_CHAIN, "XR0"),
                Side::Col => (COL_CHAIN, "XC0"),
            };
            let (_, ci, _) = chain_cols(g, c, slot);
            let mut params = vec![
                ("C_CI".to_string(), ci),
                ("C_X".to_string(), colour_of(g, x)),
            ];
            if kv_root {
                let send = match ep(g, c, KV_SEND) {
                    Some(Endpoint::Send(x)) => u32::from(x.0),
                    _ => return Err("CY root 0 sends no K / V link".into()),
                };
                let recv = match ep(g, c, KV_RECV) {
                    Some(Endpoint::Recv(x)) => u32::from(x.0),
                    _ => return Err("CY root 0 receives no K / V link".into()),
                };
                params.push(("C_KS".into(), send));
                params.push(("C_KR".into(), recv));
            }
            tiles.insert(c, (name, parked(g, c, params)));
            grp_arrays(g, c, &mut arrays);
            // its residents: each region parameter's chunk, each group norm's gamma and beta
            let mut sp: Vec<(u32, Span)> = Vec::new();
            for (&(ri, pk), &off) in &res.reg {
                let r = &g.regs[ri];
                let RegIn::Res(Res::Value { v, from }) = r.ins[pk] else {
                    continue;
                };
                let ch = g.chunk(r.outs[0]);
                let real = g.real(r.outs[0], k);
                let base = if r.head { 0 } else { k * ch };
                if real > 0 {
                    sp.push((off, [v.0, 0, from + base, real, 1, FLAT, 1, 0]));
                }
                if real < ch {
                    sp.push((off + real, zeros(ch - real)));
                }
            }
            for (&gi, &(ga, be)) in &res.gn {
                let gn = &g.gns[gi];
                let ch = g.chunk(gn.x);
                sp.push((ga, [gn.gamma.0, 0, k * ch, ch, 1, FLAT, 1, 0]));
                sp.push((be, [gn.beta.0, 0, k * ch, ch, 1, FLAT, 1, 0]));
            }
            sp.sort_by_key(|x| x.0);
            let mut spans: Vec<Span> = Vec::new();
            let mut used = 0u32;
            for (off, x) in sp {
                if used < off {
                    spans.push(zeros(off - used));
                }
                used = off + x[3] * x[6];
                spans.push(x);
            }
            let resw = res.words.max(1);
            if used < resw {
                spans.push(zeros(resw - used));
            }
            arrays.push(GridArray {
                at: c,
                sym: "res".into(),
                u16: false,
                len: resw,
                fill: GridFill::Spans(merged(spans)),
                group: None,
            });
        }
    }
    // the K / V column
    if g.attn.is_some() {
        let kc = &s.kv[0];
        let (rname, rtext) = kv_root_file(g);
        files.push((rname.clone(), rtext));
        let root = kc.root(s);
        let i0 = match ep(g, root, fs::kv_in(0)) {
            Some(Endpoint::Recv(c)) => u32::from(c.0),
            _ => 0xff,
        };
        let o0 = match ep(g, root, fs::kv_out(0)) {
            Some(Endpoint::Send(c)) => u32::from(c.0),
            _ => 0xff,
        };
        let (_, ar, _) = chain_cols(g, root, KV_CHAIN);
        let am = colour_of(g, &format!("AM{}_{}", kc.block, kc.head));
        grp_arrays(g, root, &mut arrays);
        tiles.insert(
            root,
            (
                rname,
                parked(
                    g,
                    root,
                    vec![
                        ("C_I0".into(), i0),
                        ("C_O0".into(), o0),
                        ("C_AR".into(), ar),
                        ("C_AM".into(), am),
                    ],
                ),
            ),
        );
        let (pname, ptext) = kv_pe_file(g);
        files.push((pname.clone(), ptext));
        for i in 0..kc.pya {
            let c = kc.pe(s, i);
            let (sn, rv, role) = chain_cols(g, c, KV_CHAIN);
            tiles.insert(
                c,
                (
                    pname.clone(),
                    parked(
                        g,
                        c,
                        vec![("C_AM".into(), am), ("C_S".into(), sn), ("C_R".into(), rv)],
                    ),
                ),
            );
            arrays.push(GridArray {
                at: c,
                sym: "pcfg".into(),
                u16: false,
                len: 2,
                fill: GridFill::Words(vec![i, role]),
                group: None,
            });
            grp_arrays(g, c, &mut arrays);
        }
    }
    files.push(("filler.csl".into(), filler_file()));
    dedup(&mut files, &mut tiles);
    let pl = layout_placed(g, &tiles, &arrays);
    files.push(("layout.csl".into(), pl.csl()));
    Ok(Emitted {
        files,
        placed: pl,
        arrays,
        w: s.w,
        h: s.h,
    })
}

/// The placement: every PE's code, params and routes, every exported name.
fn layout_placed(g: &GridProgram, tiles: &Tiles, arrays: &[GridArray]) -> Placed {
    let s = &g.stream;
    let mut names: BTreeMap<String, &str> = BTreeMap::new();
    for (n, ty) in [
        ("tsc", "u32"),
        ("pcfg", "u32"),
        ("w", "u16"),
        ("hin", "f32"),
        ("hout", "f32"),
        ("res", "f32"),
    ] {
        names.insert(n.into(), ty);
    }
    for a in arrays {
        let ty = if a.sym == "pcfg" {
            "u32"
        } else if a.u16 {
            "u16"
        } else {
            "f32"
        };
        names.insert(a.sym.clone(), ty);
    }
    let header = format!(
        "// A grid reading's wafer program (kernels-cerebras emit::grid): {} x {} PEs.",
        s.w, s.h
    );
    placed_of(header, (s.w, s.h), tiles, &g.fabric, &names)
}
