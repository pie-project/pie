//! The wafer's eta guest VM (guest4: `spatial/eta/CONTRACT.md` and
//! `payload/guest4` in the bring-up tree): an epilogue stage as a guest
//! table. Ported from Fork 3's `spatial/eta/etagen/src/translate.rs`; the ISA
//! in [`isa`] is the VM's (`payload/guest4/pe.csl`), so the two move
//! together: a change to the VM's instructions lands there first.
//!
//! An epilogue stage (eta's launch ops, in the order eta-exec runs them) as a
//! guest4 table. Every value lives in a wide register (a vocabulary row) or
//! in the pool (anything narrower), typed by its eta dtype; channels the
//! program takes, reads or carries have a home (a wide register or pool
//! words) the host seeds once and the table updates in place; channels the
//! host takes come back through `out` slots. The table is two sections, each
//! ended by HALT: the ops that do not depend on the logits (`pre`: constants,
//! random draws, masks from carried state), which can run as soon as the
//! previous step has ended, and the rest (`main`), then the carries.

use std::collections::{BTreeMap, BTreeSet};

use eta_compiler::codegen::launch::{LaunchOp, LaunchPackage, ValueOrigin};
use eta_exec::ExecPlan;
use eta_ir::Dtype;
use eta_ir::op::tags;
use eta_ir::registry::Stage;
use eta_ir::types::RngKind;
use serde_json::{Value as Json, json};

pub mod isa;
pub mod lane;
pub mod wafer;
use isa::*;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Wide(u32),
    Pool(u32),
    /// Keyed draws over the vocabulary not kept in a register: made again
    /// (WRNG from the state at this pool offset; fl, kind) where read.
    Remat(u32, u32, u32),
    /// A multi-row value (`[rows, vocab]`): its rows, one a register, in
    /// `Tr::groups`.
    Group(usize),
}

#[derive(Clone, Copy, Debug)]
struct Loc {
    kind: Kind,
    /// Words (pool) or 1 a vocabulary row (wide).
    n: u32,
    dtype: Dtype,
    sid: usize,
    /// Rows of a multi-row value (several lanes' logits, `[rows, vocab]`):
    /// registers kind..kind+rows.
    rows: u32,
}

struct Storage {
    kind: Kind,
    n: u32,
    pinned: bool,
    refs: usize,
    freed: bool,
}

pub struct Home {
    pub chan: u32,
    pub kind: &'static str,
    pub at: u32,
    pub n: u32,
    pub dtype: Dtype,
}

pub struct Table {
    pub ins: Vec<[u32; 6]>,
    pub pre_len: usize,
    pub homes: Vec<Home>,
    /// (channel, first slot, words, dtype)
    pub outputs: Vec<(u32, u32, u32, Dtype)>,
    /// Host outputs a vocabulary row (or rows) wide: (channel, registers,
    /// dtype); the host reads those registers after the step.
    pub wide_outputs: Vec<(u32, Vec<u32>, Dtype)>,
    /// Homes the host writes before every step: channels the stage takes or
    /// reads that the host writes and nobody seeds (an amateur's logits, a
    /// grammar mask).
    pub per_step: Vec<u32>,
    /// The pool word holding the token after `run`: the home of the channel
    /// bound to `Port::EmbedTokens` (the next pass embeds it).
    pub tok_home: Option<u32>,
    pub nreg: u32,
    pub pool_words: u32,
    /// The guest4 op groups the table uses (pe.csl's F_* bits).
    pub feat: u32,
    /// Whether keyed draws are re-made at their readers (register pressure).
    pub remat: bool,
}

pub fn tcode(d: Dtype) -> u32 {
    match d {
        Dtype::F32 => T_F32,
        Dtype::I32 => T_I32,
        Dtype::Bool => T_BOOL,
        _ => T_U32,
    }
}

struct Tr<'a> {
    pkg: &'a LaunchPackage,
    vocab: u32,
    max_reg: u32,
    max_pool: u32,
    ins: Vec<[u32; 6]>,
    st: Vec<Storage>,
    loc: BTreeMap<u32, Loc>,
    free_regs: Vec<u32>,
    next_reg: u32,
    /// Free pool blocks (offset, words).
    free_pool: Vec<(u32, u32)>,
    pool_top: u32,
    nreg_used: u32,
    remat: bool,
    /// Integer values known at translation time (constants, iotas, sums).
    known: BTreeMap<u32, Vec<u32>>,
    groups: Vec<Vec<Loc>>,
}

impl Tr<'_> {
    fn numel(&self, id: u32) -> u32 {
        self.pkg.values[id as usize].shape.iter().product::<u32>().max(1)
    }
    fn dtype(&self, id: u32) -> Dtype {
        self.pkg.values[id as usize].dtype
    }
    fn emit(&mut self, op: u32, fl: u32, d: u32, a: u32, b: u32, c: u32) {
        self.ins.push([op, fl, d, a, b, c]);
    }
    fn new_storage(&mut self, kind: Kind, n: u32, pinned: bool) -> usize {
        self.st.push(Storage { kind, n, pinned, refs: 0, freed: false });
        self.st.len() - 1
    }
    fn wide(&mut self, dtype: Dtype) -> Result<Loc, String> {
        let r = match self.free_regs.pop() {
            Some(r) => r,
            None => {
                if self.next_reg >= self.max_reg {
                    return Err(format!("more than {} live wide values", self.max_reg));
                }
                self.next_reg += 1;
                self.next_reg - 1
            }
        };
        self.nreg_used = self.nreg_used.max(r + 1);
        let sid = self.new_storage(Kind::Wide(r), 1, false);
        Ok(Loc { kind: Kind::Wide(r), n: 1, dtype, sid, rows: 1 })
    }
    fn pool(&mut self, n: u32, dtype: Dtype) -> Result<Loc, String> {
        let at = if let Some(i) = self.free_pool.iter().position(|&(_, w)| w >= n) {
            let (o, w) = self.free_pool[i];
            if w == n {
                self.free_pool.remove(i);
            } else {
                self.free_pool[i] = (o + n, w - n);
            }
            o
        } else {
            if self.pool_top + n > self.max_pool {
                return Err(format!("more than {} pool words", self.max_pool));
            }
            self.pool_top += n;
            self.pool_top - n
        };
        let sid = self.new_storage(Kind::Pool(at), n, false);
        Ok(Loc { kind: Kind::Pool(at), n, dtype, sid, rows: 1 })
    }
    fn release(&mut self, sid: usize) {
        let s = &mut self.st[sid];
        if s.pinned || s.freed || s.refs > 0 {
            return;
        }
        s.freed = true;
        if let Kind::Group(g) = s.kind {
            for m in self.groups[g].clone() {
                self.st[m.sid].refs = self.st[m.sid].refs.saturating_sub(1);
                self.release(m.sid);
            }
            return;
        }
        match s.kind {
            Kind::Wide(r) => self.free_regs.push(r),
            Kind::Pool(o) => {
                let n = s.n;
                self.free_pool_range(o, n);
            }
            Kind::Remat(..) | Kind::Group(_) => {}
        }
    }
    /// Return pool words [o, o + n): merged with free neighbours, and given
    /// back to the top when they end there (the image's pool is the high-water
    /// mark, so fragments would cost PE memory).
    fn free_pool_range(&mut self, o: u32, n: u32) {
        let (mut lo, mut hi) = (o, o + n);
        loop {
            if let Some(i) = self.free_pool.iter().position(|&(fo, fw)| fo + fw == lo) {
                lo = self.free_pool.remove(i).0;
            } else if let Some(i) = self.free_pool.iter().position(|&(fo, _)| fo == hi) {
                hi += self.free_pool.remove(i).1;
            } else {
                break;
            }
        }
        if hi == self.pool_top {
            self.pool_top = lo;
        } else {
            self.free_pool.push((lo, hi - lo));
        }
    }
    fn at(&self, l: &Loc) -> u32 {
        match l.kind {
            Kind::Wide(r) | Kind::Pool(r) => r,
            Kind::Remat(..) => panic!("a re-made value read before it is made"),
            Kind::Group(_) => panic!("a multi-row value read as one row"),
        }
    }
    fn is_wide(l: &Loc) -> bool {
        matches!(l.kind, Kind::Wide(_) | Kind::Remat(..) | Kind::Group(_))
    }
    /// A multi-row value of `rows` (each holding one ref on its storage).
    fn group(&mut self, rows: Vec<Loc>, dtype: Dtype) -> Loc {
        for r in &rows {
            self.st[r.sid].refs += 1;
        }
        let n = rows.len() as u32;
        self.groups.push(rows);
        let sid = self.new_storage(Kind::Group(self.groups.len() - 1), 1, false);
        Loc { kind: Kind::Group(self.groups.len() - 1), n: 1, dtype, sid, rows: n }
    }
    /// Row `r` of a value read beside multi-row ones (itself when it is not).
    fn row(&self, l: &Loc, r: usize) -> Loc {
        match l.kind {
            Kind::Group(g) => self.groups[g][r],
            _ => *l,
        }
    }
    /// A re-made value in a fresh register (released after its reader).
    fn make(&mut self, l: Loc) -> Result<Loc, String> {
        let Kind::Remat(st, fl, kind) = l.kind else {
            return Ok(l);
        };
        let d = self.wide(Dtype::F32)?;
        let dd = self.at(&d);
        self.emit(WRNG, fl, dd, st, 0, kind);
        Ok(Loc { dtype: l.dtype, ..d })
    }
    /// A wide operand: a register, or a pool scalar flagged as broadcast.
    /// A wide operand: a register, or a pool scalar flagged as broadcast. Wide
    /// ops read pool scalars as f32: an integer or bool is cast first (its
    /// bits would read as a denormal).
    fn wide_operand(&mut self, l: &Loc) -> Result<(u32, bool), String> {
        match l.kind {
            Kind::Wide(r) => Ok((r, false)),
            Kind::Pool(_) if l.n == 1 => {
                let f = self.pool_f32(*l)?;
                Ok((self.at(&f), true))
            }
            _ => Err(format!("a {}-word value against a vocabulary row", l.n)),
        }
    }
    /// A pool scalar's offset as f32 (for WFILL).
    fn fill_src(&mut self, l: Loc) -> Result<u32, String> {
        let f = self.pool_f32(l)?;
        Ok(self.at(&f))
    }
    fn fl(t: u32, n: u32, ab: bool, bb: bool, cb: bool) -> u32 {
        t | (u32::from(ab) << 2) | (u32::from(bb) << 3) | (u32::from(cb) << 4) | (n << 8)
    }
    /// `l` as f32 words in the pool (a cast when it holds integers).
    fn pool_f32(&mut self, l: Loc) -> Result<Loc, String> {
        if l.dtype == Dtype::F32 {
            return Ok(l);
        }
        let d = self.pool(l.n, Dtype::F32)?;
        let (dd, la) = (self.at(&d), self.at(&l));
        self.emit(QUN, Self::fl(tcode(l.dtype), l.n, false, false, false), dd, la, 0, U_CAST + T_F32);
        Ok(d)
    }
}

/// Eta's epilogue stage of `plan` as a guest4 table. `host_takes`: the
/// channels the host takes each step.
pub fn translate(
    plan: &ExecPlan,
    vocab: u32,
    host_takes: &[u32],
    max_reg: u32,
    max_pool: u32,
) -> Result<Table, String> {
    match translate_with(plan, vocab, host_takes, max_reg, max_pool, false) {
        Err(e) if e.contains("live wide values") => {
            translate_with(plan, vocab, host_takes, max_reg, max_pool, true)
                .map_err(|e2| format!("{e2} (also with draws re-made at their readers)"))
        }
        other => other,
    }
}

fn translate_with(
    plan: &ExecPlan,
    vocab: u32,
    host_takes: &[u32],
    max_reg: u32,
    max_pool: u32,
    remat: bool,
) -> Result<Table, String> {
    let pkg = &plan.package;
    let sp = plan
        .stages
        .iter()
        .find(|sp| pkg.stages[sp.stage_index].stage == Stage::Epilogue)
        .ok_or("no epilogue")?;
    let stage = &pkg.stages[sp.stage_index];
    let ops: Vec<&LaunchOp> = sp
        .value_ids
        .iter()
        .filter_map(|id| sp.op_by_result.get(id).map(|&i| &stage.ops[i]))
        .collect();
    // Roots: every value an op reads that no op makes.
    let made: BTreeSet<u32> = ops.iter().flat_map(|o| (0..o.result_count.max(1) as u32).map(|k| o.result_id + k)).collect();
    // Logit dependence: main ops read the logits (directly or through others).
    let mut main: BTreeSet<u32> = BTreeSet::new();
    for v in &pkg.values {
        if v.source == ValueOrigin::Output {
            main.insert(v.id);
        }
    }
    // The prefix holds what is logits-free and either narrow or a random
    // draw (the costly part): vocabulary-wide masks from carried state stay
    // in main, so their registers do not live across the logits' arrival.
    let wide_of = |id: u32| pkg.values[id as usize].shape.iter().product::<u32>().max(1) == vocab;
    let mut order_pre = Vec::new();
    let mut order_main = Vec::new();
    for op in &ops {
        let dep = op.reads().any(|a| main.contains(&a))
            || (wide_of(op.result_id) && op.tag != tags::RNG_KEYED);
        if dep {
            for k in 0..op.result_count.max(1) as u32 {
                main.insert(op.result_id + k);
            }
            order_main.push(*op);
        } else {
            order_pre.push(*op);
        }
    }
    let ordered: Vec<&LaunchOp> = order_pre.iter().chain(order_main.iter()).copied().collect();
    // Last use of every value, in emission order; puts read at the end.
    let end = ordered.len();
    let mut last: BTreeMap<u32, usize> = BTreeMap::new();
    for (i, op) in ordered.iter().enumerate() {
        for a in op.reads() {
            last.insert(a, i);
        }
    }
    for p in &stage.puts {
        last.insert(p.value, end);
    }

    let mut t = Tr {
        pkg,
        vocab,
        max_reg,
        max_pool,
        ins: Vec::new(),
        st: Vec::new(),
        loc: BTreeMap::new(),
        free_regs: Vec::new(),
        next_reg: 1,
        free_pool: Vec::new(),
        pool_top: 0,
        nreg_used: 1,
        remat,
        known: BTreeMap::new(),
        groups: Vec::new(),
    };
    let mut homes: Vec<Home> = Vec::new();
    // The logits: W0.
    let w0 = t.new_storage(Kind::Wide(0), 1, true);
    // A multi-row pass's logits take W0..W(rows-1).
    let out_rows = pkg
        .values
        .iter()
        .filter(|v| v.source == ValueOrigin::Output)
        .map(|v| v.shape.iter().product::<u32>().max(1) / vocab)
        .max()
        .unwrap_or(1);
    t.next_reg = out_rows.max(1);
    t.nreg_used = t.next_reg;
    // Channel homes: every channel the stage takes, reads or puts (but the
    // host's outputs).
    let mut home_of: BTreeMap<u32, Loc> = BTreeMap::new();
    let chans: BTreeSet<u32> = stage
        .takes
        .iter()
        .chain(stage.reads.iter())
        .copied()
        .chain(stage.puts.iter().map(|p| p.channel))
        .filter(|c| !host_takes.contains(c))
        .collect();
    for &c in &chans {
        let ch = &pkg.channels[c as usize];
        let n = ch.shape.iter().product::<u32>().max(1);
        let dtype = match ch.dtype {
            eta_ir::container::ChanDType::Concrete(d) => d,
            _ => return Err(format!("channel {c}: a generic dtype")),
        };
        let l = if n == vocab { t.wide(dtype)? } else { t.pool(n, dtype)? };
        t.st[l.sid].pinned = true;
        homes.push(Home {
            chan: c,
            kind: if n == vocab { "wide" } else { "pool" },
            at: t.at(&l),
            n,
            dtype,
        });
        home_of.insert(c, l);
    }
    // Roots.
    let mut consts: Vec<(u32, Loc)> = Vec::new();
    for v in &pkg.values {
        if made.contains(&v.id) || !last.contains_key(&v.id) {
            continue;
        }
        let l = match v.source {
            ValueOrigin::Output => {
                let rows = v.shape.iter().product::<u32>().max(1) / vocab;
                if rows > 1 {
                    let members: Vec<Loc> = (0..rows)
                        .map(|r| {
                            let sid = t.new_storage(Kind::Wide(r), 1, true);
                            Loc { kind: Kind::Wide(r), n: 1, dtype: Dtype::F32, sid, rows: 1 }
                        })
                        .collect();
                    t.group(members, Dtype::F32)
                } else {
                    Loc { kind: Kind::Wide(0), n: 1, dtype: Dtype::F32, sid: w0, rows: 1 }
                }
            }
            ValueOrigin::ChannelTake | ValueOrigin::ChannelRead => *home_of
                .get(&v.channel)
                .ok_or(format!("value {} reads channel {} with no home", v.id, v.channel))?,
            ValueOrigin::Const => {
                let n = v.shape.iter().product::<u32>().max(1);
                let l = t.pool(n, v.dtype)?;
                consts.push((v.literal_bits, l));
                if v.dtype != Dtype::F32 {
                    t.known.insert(v.id, vec![v.literal_bits; n as usize]);
                }
                l
            }
            ValueOrigin::OpResult => continue,
        };
        t.st[l.sid].refs += 1;
        t.loc.insert(v.id, Loc { dtype: v.dtype, ..l });
    }
    for (bits, l) in consts {
        let at = t.at(&l);
        t.emit(QCONST, Tr::fl(tcode(l.dtype), l.n, false, false, false), at, 0, 0, bits);
    }
    let mut pre_len = 0;
    for (i, op) in ordered.iter().enumerate() {
        if i == order_pre.len() {
            t.emit(HALT, 0, 0, 0, 0, 0);
            pre_len = t.ins.len();
        }
        let results = one(&mut t, op).map_err(|e| format!("{} ({}, {:?}): {e}", tag_name(op.tag), op.result_id, pkg.values[op.result_id as usize].shape))?;
        for (k, l) in results.into_iter().enumerate() {
            let id = op.result_id + k as u32;
            t.st[l.sid].refs += 1;
            t.loc.insert(id, l);
            if !last.contains_key(&id) {
                t.st[l.sid].refs -= 1;
                t.release(l.sid);
            }
        }
        // Values whose last reader this was.
        let dead: Vec<u32> = op.reads().filter(|a| last.get(a) == Some(&i)).collect::<BTreeSet<_>>().into_iter().collect();
        for a in dead {
            if let Some(l) = t.loc.get(&a).copied() {
                t.st[l.sid].refs = t.st[l.sid].refs.saturating_sub(1);
                t.release(l.sid);
            }
        }
    }
    if order_pre.len() == ordered.len() {
        t.emit(HALT, 0, 0, 0, 0, 0);
        pre_len = t.ins.len();
    }
    // Puts: the host's outputs into slots, the rest into their homes.
    let mut outputs = Vec::new();
    let mut wide_outputs = Vec::new();
    let mut slot = 0u32;
    for p in &stage.puts {
        let l = *t.loc.get(&p.value).ok_or(format!("put of untranslated value {}", p.value))?;
        if host_takes.contains(&p.channel) {
            if Tr::is_wide(&l) {
                let regs: Vec<u32> = match l.kind {
                    Kind::Group(g) => t.groups[g].iter().map(|m| t.at(m)).collect(),
                    Kind::Remat(..) => {
                        let m = t.make(l)?;
                        vec![t.at(&m)]
                    }
                    _ => vec![t.at(&l)],
                };
                wide_outputs.push((p.channel, regs, l.dtype));
                continue;
            }
            if slot + l.n > 16 {
                return Err("more than 16 output words".into());
            }
            let at = t.at(&l);
            t.emit(OUT, Tr::fl(0, l.n, false, false, false), slot, at, 0, 0);
            outputs.push((p.channel, slot, l.n, l.dtype));
            slot += l.n;
        } else {
            let h = home_of[&p.channel];
            if let Kind::Group(_) = l.kind {
                return Err("a multi-row value put into a channel the program keeps".into());
            }
            if let (Kind::Wide(hr), Kind::Remat(st, fl, kind)) = (h.kind, l.kind) {
                t.emit(WRNG, fl, hr, st, 0, kind);
                continue;
            }
            if t.at(&h) == t.at(&l) && Tr::is_wide(&h) == Tr::is_wide(&l) {
                continue;
            }
            match (h.kind, l.kind) {
                (Kind::Wide(hr), Kind::Wide(r)) => t.emit(WMOV, 0, hr, r, 0, 0),
                (Kind::Wide(hr), Kind::Pool(_)) if l.n == 1 => {
                    let o = t.fill_src(l)?;
                    t.emit(WFILL, 0, hr, o, 0, 0)
                }
                (Kind::Pool(ho), Kind::Pool(o)) => {
                    let ab = l.n == 1 && h.n > 1;
                    t.emit(QMOV, Tr::fl(0, h.n, ab, false, false), ho, o, 0, 0)
                }
                _ => return Err("a pool home put from a vocabulary row".into()),
            }
        }
    }
    t.emit(HALT, 0, 0, 0, 0, 0);
    scalar_operands(&mut t.ins);
    let feat = t.ins.iter().fold(0, |f, i| f | feat_of(i[0]));
    let tok_home = pkg
        .ports
        .iter()
        .find(|p| !p.is_const && p.port == eta_ir::registry::Port::EmbedTokens)
        .and_then(|p| homes.iter().find(|h| h.chan == p.channel && h.kind == "pool"))
        .map(|h| h.at);
    let per_step = homes
        .iter()
        .filter(|h| {
            let ch = &pkg.channels[h.chan as usize];
            ch.host_role == eta_ir::container::HostRole::Writer && !ch.seeded
        })
        .map(|h| h.chan)
        .collect();
    Ok(Table {
        ins: t.ins,
        pre_len,
        homes,
        outputs,
        wide_outputs,
        per_step,
        tok_home,
        nreg: t.nreg_used,
        pool_words: t.pool_top,
        feat,
        remat,
    })
}

/// A WBIN right after the WFILL of one of its operands takes the filled pool
/// scalar instead (the VM's register-by-scalar paths are vector ops, the
/// divide included); the fill stays for later readers.
fn scalar_operands(ins: &mut [[u32; 6]]) {
    for i in 1..ins.len() {
        let [fop, _, r, p, _, _] = ins[i - 1];
        let [op, fl, _, a, b, c] = ins[i];
        if fop != WFILL || op != WBIN || fl & 12 != 0 || c > 3 || (a == r && b == r) {
            continue;
        }
        if b == r {
            ins[i][1] = fl | 8;
            ins[i][4] = p;
        } else if a == r && (c == 0 || c == 2) {
            ins[i][1] = fl | 8;
            ins[i][3] = b;
            ins[i][4] = p;
        }
    }
}

/// pe.csl's op groups: F_RNG 1, F_RED 2, F_GS 4, F_LIST 8, F_RANK 16,
/// F_QARITH 32, F_QMEM 64, F_QRED 128, F_WMATH 256.
pub fn feat_of(op: u32) -> u32 {
    match op {
        WRNG | QRNG => 1,
        RRED => 2,
        WGATHER | WSCATTER => 4,
        TOPK | PIVMASS => 8,
        PIVRANK => 16,
        QBIN | QCMP | QUN | QSEL => 32,
        QIOTA | QCONST | QMOV | QGATHER | QSCATTER | QCUM => 64,
        QRED => 128,
        WUN | WCMP => 256,
        _ => 0,
    }
}

pub fn tag_name(tag: u8) -> String {
    eta_ir::op::OP_TABLE
        .iter()
        .find(|s| s.tag == tag)
        .map_or_else(|| format!("tag{tag}"), |s| s.name.to_string())
}

fn binop(tag: u8) -> Option<u32> {
    Some(match tag {
        tags::ADD => B_ADD,
        tags::SUB => B_SUB,
        tags::MUL => B_MUL,
        tags::DIV => B_DIV,
        tags::MAX_ELEM => B_MAX,
        tags::MIN_ELEM => B_MIN,
        tags::REM => B_REM,
        _ => return None,
    })
}
fn cmpop(tag: u8) -> Option<u32> {
    Some(match tag {
        tags::GT => 0,
        tags::GE => 1,
        tags::LT => 2,
        tags::LE => 3,
        tags::EQ => 4,
        tags::NE => 5,
        _ => return None,
    })
}
fn unop(tag: u8) -> Option<u32> {
    Some(match tag {
        tags::NEG => U_NEG,
        tags::ABS => U_ABS,
        tags::EXP => U_EXP,
        tags::LOG => U_LOG,
        tags::SQRT => U_SQRT,
        tags::NOT => U_NOT,
        tags::RECIP => U_RECIP,
        tags::SIGN => U_SIGN,
        tags::RSQRT => U_RSQRT,
        _ => return None,
    })
}

/// One launch op's results.
fn one(t: &mut Tr, op: &LaunchOp) -> Result<Vec<Loc>, String> {
    let raw: Vec<Loc> = op
        .args
        .iter()
        .map(|a| t.loc.get(a).copied().ok_or(format!("value {a} untranslated")))
        .collect::<Result<_, _>>()?;
    // Re-made draws read here: made into temporaries, freed after the op.
    let mut args = Vec::new();
    let mut temps = Vec::new();
    for l in raw {
        if matches!(l.kind, Kind::Remat(..)) && op.tag != tags::RESHAPE && op.tag != tags::BROADCAST {
            let m = t.make(l)?;
            temps.push(m.sid);
            args.push(m);
        } else {
            args.push(l);
        }
    }
    let out = one_args(t, op, &args);
    for sid in temps {
        t.release(sid);
    }
    // What is known of the result.
    let k = |id: u32| t.known.get(&id).cloned();
    let known = match op.tag {
        tags::IOTA if t.numel(op.result_id) != t.vocab => Some((0..op.imm).collect()),
        tags::RESHAPE | tags::CAST | tags::BROADCAST if t.dtype(op.result_id) != Dtype::F32 => {
            k(op.args[0]).map(|v| {
                let n = t.numel(op.result_id) as usize;
                if v.len() == 1 { vec![v[0]; n] } else { v }
            })
        }
        tags::ADD => match (k(op.args[0]), k(op.args[1])) {
            (Some(a), Some(b)) => {
                let n = a.len().max(b.len());
                Some((0..n).map(|i| a[if a.len() == 1 { 0 } else { i }].wrapping_add(b[if b.len() == 1 { 0 } else { i }])).collect())
            }
            _ => None,
        },
        _ => None,
    };
    if let Some(v) = known {
        t.known.insert(op.result_id, v);
    }
    out
}

fn one_args(t: &mut Tr, op: &LaunchOp, args: &[Loc]) -> Result<Vec<Loc>, String> {
    let rows_in = args.iter().map(|a| a.rows).max().unwrap_or(1);
    let rn = t.numel(op.result_id);
    let rows_out = if rn > t.vocab && rn % t.vocab == 0 { rn / t.vocab } else { 1 };
    if rows_in > 1 || rows_out > 1 {
        return multi(t, op, args, rows_in.max(rows_out));
    }
    single(t, op, args.to_vec(), t.dtype(op.result_id), rn)
}

/// An op over multi-row values (`[rows, vocab]`, a row a register): row by
/// row where it is elementwise, per-row reductions into `[rows]`, a whole
/// value's max / argmax from its rows' (order-free, so exact).
fn multi(t: &mut Tr, op: &LaunchOp, args: &[Loc], rows: u32) -> Result<Vec<Loc>, String> {
    let rid = op.result_id;
    let rdt = t.dtype(rid);
    let rn = t.numel(rid);
    let v = t.vocab;
    let tag = op.tag;
    let a = args[0];
    match tag {
        tags::RESHAPE => return Ok(vec![Loc { dtype: rdt, ..a }]),
        tags::GATHER if a.rows > 1 => {
            let idx = t.known.get(&op.args[1]).cloned().ok_or("a row gather by a value not known when translating")?;
            if idx.iter().any(|&i| i >= a.rows) {
                return Err("a row gather out of range".into());
            }
            let picked: Vec<Loc> = idx.iter().map(|&i| t.row(&a, i as usize)).collect();
            if picked.len() == 1 {
                return Ok(picked);
            }
            return Ok(vec![t.group(picked, rdt)]);
        }
        tags::BROADCAST => {
            // [rows] or a scalar (pool) or a row (wide) over [rows, vocab].
            let mut out = Vec::new();
            for r in 0..rows as usize {
                if Tr::is_wide(&a) {
                    out.push(t.row(&a, r));
                    continue;
                }
                if a.n != 1 && a.n != rows {
                    return Err(format!("broadcast of {} values to {rows} rows", a.n));
                }
                let af = t.pool_f32(a)?;
                let off = t.at(&af) + if a.n == rows { r as u32 } else { 0 };
                let d = t.wide(rdt)?;
                let dd = t.at(&d);
                t.emit(WFILL, 0, dd, off, 0, 0);
                out.push(d);
            }
            return Ok(vec![t.group(out, rdt)]);
        }
        tags::RNG_KEYED => {
            let s = a;
            let kind = match op.rng_kind {
                RngKind::Uniform => 0,
                RngKind::Gumbel => 1,
                RngKind::Normal => return Err("normal draws (cos)".into()),
            };
            let sa = t.at(&s);
            let mut out = Vec::new();
            for r in 0..rows {
                let d = t.wide(Dtype::F32)?;
                let dd = t.at(&d);
                t.emit(WRNG, Tr::fl(0, 0, false, false, s.n == 1), dd, sa, r * v, kind);
                out.push(d);
            }
            return Ok(vec![t.group(out, Dtype::F32)]);
        }
        tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN | tags::REDUCE_ARGMAX => {
            let c = match tag {
                tags::REDUCE_SUM => 0,
                tags::REDUCE_MAX => 1,
                tags::REDUCE_MIN => 2,
                _ => 3,
            };
            let fdt = if c == 3 { Dtype::I32 } else { Dtype::F32 };
            if rn == rows {
                // A row each: [rows].
                let p = t.pool(rows, fdt)?;
                let pa = t.at(&p);
                for r in 0..rows as usize {
                    let ra = t.row(&a, r);
                    let rr = t.at(&ra);
                    t.emit(RRED, 0, pa + r as u32, rr, 0, c);
                }
                if c == 3 || rdt == Dtype::F32 {
                    return Ok(vec![Loc { dtype: rdt, ..p }]);
                }
                let d = t.pool(rows, rdt)?;
                let dd = t.at(&d);
                t.emit(QUN, Tr::fl(T_F32, rows, false, false, false), dd, pa, 0, U_CAST + tcode(rdt));
                return Ok(vec![d]);
            }
            if rn != 1 || c == 0 {
                return Err("a sum over several rows as one".into());
            }
            // The whole value: its rows' max (min), then for argmax the first
            // row holding it, at that row's argmax.
            let m = t.pool(rows, Dtype::F32)?;
            let ma = t.at(&m);
            for r in 0..rows as usize {
                let ra = t.row(&a, r);
                let rr = t.at(&ra);
                t.emit(RRED, 0, ma + r as u32, rr, 0, if c == 3 { 1 } else { c });
            }
            let g = t.pool(1, Dtype::F32)?;
            let ga = t.at(&g);
            t.emit(QRED, Tr::fl(T_F32, rows, false, false, false), ga, ma, 0, if c == 3 { 1 } else { c });
            if c != 3 {
                return Ok(vec![Loc { dtype: rdt, ..g }]);
            }
            let ix = t.pool(rows, Dtype::I32)?;
            let ia = t.at(&ix);
            for r in 0..rows as usize {
                let ra = t.row(&a, r);
                let rr = t.at(&ra);
                t.emit(RRED, 0, ia + r as u32, rr, 0, 3);
            }
            let eq = t.pool(rows, Dtype::Bool)?;
            let ea = t.at(&eq);
            t.emit(QCMP, Tr::fl(T_F32, rows, false, true, false), ea, ma, ga, 4);
            let first = t.pool(1, Dtype::U32)?;
            let fa = t.at(&first);
            t.emit(QRED, Tr::fl(T_U32, rows, false, false, false), fa, ea, 0, 3);
            let at_row = t.pool(1, Dtype::I32)?;
            let aa = t.at(&at_row);
            t.emit(QGATHER, Tr::fl(T_U32, 1, false, false, false), aa, ia, fa, rows);
            let vv = t.pool(1, Dtype::U32)?;
            let va = t.at(&vv);
            t.emit(QCONST, Tr::fl(T_U32, 1, false, false, false), va, 0, 0, v);
            let base = t.pool(1, Dtype::U32)?;
            let ba = t.at(&base);
            t.emit(QBIN, Tr::fl(T_U32, 1, false, false, false), ba, fa, va, B_MUL);
            let d = t.pool(1, Dtype::I32)?;
            let dd = t.at(&d);
            t.emit(QBIN, Tr::fl(T_U32, 1, false, false, false), dd, ba, aa, B_ADD);
            return Ok(vec![d]);
        }
        tags::TOP_K if rn == op.imm && op.imm > 0 => {
            // Top-k of the whole value: each row's top k, then k picks of the
            // first max among them (rows in order, a row's list sorted: the
            // first max is the lowest index among equal values).
            let k = op.imm;
            let cv = t.pool(rows * k, Dtype::F32)?;
            let ci = t.pool(rows * k, Dtype::U32)?;
            let (cva, cia) = (t.at(&cv), t.at(&ci));
            let vv = t.pool(1, Dtype::U32)?;
            let va = t.at(&vv);
            for r in 0..rows {
                let ra = t.row(&a, r as usize);
                let rr = t.at(&ra);
                t.emit(TOPK, 0, cva + r * k, rr, cia + r * k, k);
                if r > 0 {
                    t.emit(QCONST, Tr::fl(T_U32, 1, false, false, false), va, 0, 0, r * v);
                    t.emit(QBIN, Tr::fl(T_U32, k, false, true, false), cia + r * k, cia + r * k, va, B_ADD);
                }
            }
            let ov = t.pool(k, Dtype::F32)?;
            let oi = t.pool(k, Dtype::U32)?;
            let (ova, oia) = (t.at(&ov), t.at(&oi));
            let at = t.pool(1, Dtype::U32)?;
            let aa = t.at(&at);
            let ninf = t.pool(1, Dtype::F32)?;
            let na = t.at(&ninf);
            t.emit(QCONST, Tr::fl(T_F32, 1, false, false, false), na, 0, 0, f32::NEG_INFINITY.to_bits());
            for j in 0..k {
                t.emit(QRED, Tr::fl(T_F32, rows * k, false, false, false), aa, cva, 0, 3);
                t.emit(QGATHER, Tr::fl(T_U32, 1, false, false, false), ova + j, cva, aa, rows * k);
                t.emit(QGATHER, Tr::fl(T_U32, 1, false, false, false), oia + j, cia, aa, rows * k);
                t.emit(QSCATTER, Tr::fl(T_U32, 1, true, false, false), cva, aa, na, rows * k);
            }
            return Ok(vec![ov, oi]);
        }
        _ => {}
    }
    // Elementwise: row by row.
    let elementwise = binop(tag).is_some()
        || cmpop(tag).is_some()
        || unop(tag).is_some()
        || matches!(tag, tags::AND | tags::OR | tags::SELECT | tags::CAST);
    if elementwise && rn == rows * v {
        let mut out = Vec::new();
        for r in 0..rows as usize {
            let row_args: Vec<Loc> = args.iter().map(|l| t.row(l, r)).collect();
            if row_args.iter().any(|l| !Tr::is_wide(l) && l.n != 1) {
                return Err("a per-row value against rows".into());
            }
            let mut res = single(t, op, row_args, rdt, v)?;
            out.push(res.remove(0));
        }
        return Ok(vec![t.group(out, rdt)]);
    }
    if tag == tags::PIVOT_THRESHOLD && rn == rows * v {
        let p = *t.loc.get(&op.pred_payload).ok_or("the pivot's payload is untranslated")?;
        let mut out = Vec::new();
        for r in 0..rows as usize {
            let ra = t.row(&a, r);
            let pr = if p.n == rows && rows > 1 { Loc { kind: Kind::Pool(t.at(&p) + r as u32), n: 1, ..p } } else { p };
            let d = t.wide(Dtype::Bool)?;
            let (dd, aa, pp) = (t.at(&d), t.at(&ra), t.at(&pr));
            match op.pred_tag {
                0 => t.emit(PIVRANK, Tr::fl(tcode(pr.dtype), 0, false, false, false), dd, aa, pp, 0),
                1 => t.emit(PIVMASS, 0, dd, aa, pp, 0),
                _ => t.emit(WCMP, Tr::fl(0, 0, false, true, false), dd, aa, pp, 1),
            }
            out.push(d);
        }
        return Ok(vec![t.group(out, Dtype::Bool)]);
    }
    Err("an op over several rows".into())
}

fn single(t: &mut Tr, op: &LaunchOp, args: Vec<Loc>, rdt: Dtype, rn: u32) -> Result<Vec<Loc>, String> {
    let wide_result = rn == t.vocab;
    let tag = op.tag;
    // Aliases.
    if tag == tags::RESHAPE {
        return Ok(vec![Loc { dtype: rdt, ..args[0] }]);
    }
    if tag == tags::BROADCAST {
        let a = args[0];
        let an = if Tr::is_wide(&a) { t.vocab } else { a.n };
        if an == rn {
            return Ok(vec![Loc { dtype: rdt, ..a }]);
        }
        if !Tr::is_wide(&a) && a.n > 1 && rn % a.n == 0 {
            // A short value tiled (eta's broadcast of [n] to [m, n]).
            let d = t.pool(rn, rdt)?;
            let (dd, at) = (t.at(&d), t.at(&a));
            for k in 0..rn / a.n {
                t.emit(QMOV, Tr::fl(0, a.n, false, false, false), dd + k * a.n, at, 0, 0);
            }
            return Ok(vec![d]);
        }
        if a.n != 1 || Tr::is_wide(&a) {
            return Err(format!("broadcast of {an} values to {rn}"));
        }
        let at = t.at(&a);
        if wide_result {
            let at = t.fill_src(a)?;
            let d = t.wide(rdt)?;
            let dd = t.at(&d);
            t.emit(WFILL, 0, dd, at, 0, 0);
            return Ok(vec![d]);
        }
        let d = t.pool(rn, rdt)?;
        let dd = t.at(&d);
        t.emit(QMOV, Tr::fl(0, rn, true, false, false), dd, at, 0, 0);
        return Ok(vec![d]);
    }
    if tag == tags::CAST {
        let a = args[0];
        let from = a.dtype;
        if from == rdt {
            return Ok(vec![a]);
        }
        if Tr::is_wide(&a) {
            // Wide storage is f32: integers and bools are already exact.
            let to_bool = rdt == Dtype::Bool && from != Dtype::Bool;
            let to_int = matches!(rdt, Dtype::I32 | Dtype::U32) && from == Dtype::F32;
            if !to_bool && !to_int {
                return Ok(vec![Loc { dtype: rdt, ..a }]);
            }
            let d = t.wide(rdt)?;
            let (dd, aa) = (t.at(&d), t.at(&a));
            t.emit(WUN, 0, dd, aa, 0, if to_bool { U_NONZERO } else { U_TRUNC });
            return Ok(vec![d]);
        }
        let ints = |d: Dtype| matches!(d, Dtype::I32 | Dtype::U32);
        if ints(from) && ints(rdt) {
            return Ok(vec![Loc { dtype: rdt, ..a }]);
        }
        let d = t.pool(a.n, rdt)?;
        let (dd, aa) = (t.at(&d), t.at(&a));
        t.emit(QUN, Tr::fl(tcode(from), a.n, false, false, false), dd, aa, 0, U_CAST + tcode(rdt));
        return Ok(vec![d]);
    }
    if let Some(b) = binop(tag).or(match tag {
        tags::AND => Some(B_MIN),
        tags::OR => Some(B_MAX),
        _ => None,
    }) {
        let (x, y) = (args[0], args[1]);
        let adt = x.dtype;
        if wide_result {
            if adt != Dtype::F32 && adt != Dtype::Bool && matches!(b, B_DIV | B_REM) {
                return Err("integer division over the vocabulary".into());
            }
            let (xa, xs) = t.wide_operand(&x)?;
            let (ya, ys) = t.wide_operand(&y)?;
            let d = t.wide(rdt)?;
            let dd = t.at(&d);
            t.emit(WBIN, Tr::fl(0, 0, xs, ys, false), dd, xa, ya, b);
            return Ok(vec![d]);
        }
        let d = t.pool(rn, rdt)?;
        let (dd, xa, ya) = (t.at(&d), t.at(&x), t.at(&y));
        let tc = if adt == Dtype::Bool { T_U32 } else { tcode(adt) };
        t.emit(QBIN, Tr::fl(tc, rn, x.n == 1 && rn > 1, y.n == 1 && rn > 1, false), dd, xa, ya, b);
        return Ok(vec![d]);
    }
    if let Some(c) = cmpop(tag) {
        let (x, y) = (args[0], args[1]);
        if wide_result {
            let (xa, xs) = t.wide_operand(&x)?;
            let (ya, ys) = t.wide_operand(&y)?;
            let d = t.wide(Dtype::Bool)?;
            let dd = t.at(&d);
            t.emit(WCMP, Tr::fl(0, 0, xs, ys, false), dd, xa, ya, c);
            return Ok(vec![d]);
        }
        let d = t.pool(rn, Dtype::Bool)?;
        let (dd, xa, ya) = (t.at(&d), t.at(&x), t.at(&y));
        let tc = if x.dtype == Dtype::Bool { T_U32 } else { tcode(x.dtype) };
        t.emit(QCMP, Tr::fl(tc, rn, x.n == 1 && rn > 1, y.n == 1 && rn > 1, false), dd, xa, ya, c);
        return Ok(vec![d]);
    }
    if let Some(u) = unop(tag) {
        let a = args[0];
        if Tr::is_wide(&a) {
            if a.dtype != Dtype::F32 && !matches!(u, U_NOT | U_NEG | U_ABS) {
                return Err("integer math over the vocabulary".into());
            }
            let d = t.wide(rdt)?;
            let (dd, aa) = (t.at(&d), t.at(&a));
            t.emit(WUN, 0, dd, aa, 0, u);
            return Ok(vec![d]);
        }
        let d = t.pool(a.n, rdt)?;
        let (dd, aa) = (t.at(&d), t.at(&a));
        t.emit(QUN, Tr::fl(tcode(a.dtype), a.n, false, false, false), dd, aa, 0, u);
        return Ok(vec![d]);
    }
    match tag {
        tags::SELECT => {
            let (c, x, y) = (args[0], args[1], args[2]);
            if wide_result {
                let cw = if Tr::is_wide(&c) {
                    t.at(&c)
                } else {
                    let cc = t.fill_src(c)?;
                    let w = t.wide(Dtype::Bool)?;
                    let ww = t.at(&w);
                    t.emit(WFILL, 0, ww, cc, 0, 0);
                    ww
                };
                let (xa, xs) = t.wide_operand(&x)?;
                let (ya, ys) = t.wide_operand(&y)?;
                let d = t.wide(rdt)?;
                let dd = t.at(&d);
                t.emit(WSEL, Tr::fl(0, 0, xs, ys, false), dd, xa, ya, cw);
                return Ok(vec![d]);
            }
            let d = t.pool(rn, rdt)?;
            let (dd, xa, ya, ca) = (t.at(&d), t.at(&x), t.at(&y), t.at(&c));
            t.emit(QSEL, Tr::fl(0, rn, x.n == 1 && rn > 1, y.n == 1 && rn > 1, c.n == 1 && rn > 1), dd, xa, ya, ca);
            Ok(vec![d])
        }
        tags::IOTA => {
            if wide_result {
                let d = t.wide(rdt)?;
                let dd = t.at(&d);
                t.emit(WIOTA, 0, dd, 0, 0, 0);
                return Ok(vec![d]);
            }
            let d = t.pool(rn, rdt)?;
            let dd = t.at(&d);
            t.emit(QIOTA, Tr::fl(T_U32, rn, false, false, false), dd, 0, 0, 0);
            Ok(vec![d])
        }
        tags::RNG_KEYED => {
            let s = args[0];
            if Tr::is_wide(&s) || s.n > 2 {
                return Err("a keyed draw's state is not [key, counter]".into());
            }
            let kind = match op.rng_kind {
                RngKind::Uniform => 0,
                RngKind::Gumbel => 1,
                RngKind::Normal => return Err("normal draws (cos)".into()),
            };
            let sa = t.at(&s);
            if wide_result && t.remat {
                // Kept as its state: the state's storage lives as long.
                return Ok(vec![Loc { kind: Kind::Remat(sa, Tr::fl(0, 0, false, false, s.n == 1), kind), n: 1, dtype: Dtype::F32, sid: s.sid, rows: 1 }]);
            }
            if wide_result {
                let d = t.wide(Dtype::F32)?;
                let dd = t.at(&d);
                t.emit(WRNG, Tr::fl(0, 0, false, false, s.n == 1), dd, sa, 0, kind);
                return Ok(vec![d]);
            }
            let d = t.pool(rn, Dtype::F32)?;
            let dd = t.at(&d);
            t.emit(QRNG, Tr::fl(0, rn, false, false, s.n == 1), dd, sa, 0, kind);
            Ok(vec![d])
        }
        tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN | tags::REDUCE_ARGMAX => {
            let a = args[0];
            if rn != 1 {
                return Err("a multi-row reduction".into());
            }
            let c = match tag {
                tags::REDUCE_SUM => 0,
                tags::REDUCE_MAX => 1,
                tags::REDUCE_MIN => 2,
                _ => 3,
            };
            if Tr::is_wide(&a) {
                let fdt = if c == 3 { Dtype::I32 } else { Dtype::F32 };
                let f = t.pool(1, fdt)?;
                let (ff, aa) = (t.at(&f), t.at(&a));
                t.emit(RRED, 0, ff, aa, 0, c);
                if c == 3 || rdt == Dtype::F32 {
                    return Ok(vec![Loc { dtype: rdt, ..f }]);
                }
                // An integer reduction over the vocabulary: exact in f32.
                let d = t.pool(1, rdt)?;
                let dd = t.at(&d);
                t.emit(QUN, Tr::fl(T_F32, 1, false, false, false), dd, ff, 0, U_CAST + tcode(rdt));
                return Ok(vec![d]);
            }
            let d = t.pool(1, rdt)?;
            let (dd, aa) = (t.at(&d), t.at(&a));
            let tc = if a.dtype == Dtype::Bool { T_U32 } else { tcode(a.dtype) };
            t.emit(QRED, Tr::fl(tc, a.n, false, false, false), dd, aa, 0, c);
            Ok(vec![d])
        }
        tags::GATHER => {
            let (src, idx) = (args[0], args[1]);
            if Tr::is_wide(&idx) {
                return Err("a vocabulary-wide gather index".into());
            }
            let rest = t.pkg.values[op.args[0] as usize].shape.iter().skip(1).product::<u32>().max(1);
            if rest != 1 && !Tr::is_wide(&src) {
                // Rows of a short table: flat index idx[j]·rest + r for
                // entry j·rest + r (out of range stays out of range: 0).
                let m = idx.n * rest;
                let u = |t: &mut Tr, n: u32| t.pool(n, Dtype::U32);
                let io = u(t, m)?;
                let (ioa, ia) = (t.at(&io), t.at(&idx));
                t.emit(QIOTA, Tr::fl(T_U32, m, false, false, false), ioa, 0, 0, 0);
                let rs = u(t, 1)?;
                let rsa = t.at(&rs);
                t.emit(QCONST, Tr::fl(T_U32, 1, false, false, false), rsa, 0, 0, rest);
                let jj = u(t, m)?;
                let ja = t.at(&jj);
                t.emit(QBIN, Tr::fl(T_U32, m, false, true, false), ja, ioa, rsa, B_DIV);
                let rr = u(t, m)?;
                let ra = t.at(&rr);
                t.emit(QBIN, Tr::fl(T_U32, m, false, true, false), ra, ioa, rsa, B_REM);
                let rep = u(t, m)?;
                let rpa = t.at(&rep);
                t.emit(QGATHER, Tr::fl(T_U32, m, false, false, false), rpa, ia, ja, idx.n);
                t.emit(QBIN, Tr::fl(T_U32, m, false, true, false), rpa, rpa, rsa, B_MUL);
                t.emit(QBIN, Tr::fl(T_U32, m, false, false, false), rpa, rpa, ra, B_ADD);
                let d = t.pool(m, rdt)?;
                let (dd, sa) = (t.at(&d), t.at(&src));
                t.emit(QGATHER, Tr::fl(T_U32, m, false, false, false), dd, sa, rpa, src.n);
                for x in [io, rs, jj, rr, rep] {
                    t.release(x.sid);
                }
                return Ok(vec![d]);
            }
            if rest != 1 {
                return Err("a gather of rows".into());
            }
            let (ia, it) = (t.at(&idx), tcode(idx.dtype));
            if Tr::is_wide(&src) {
                if idx.n > 128 {
                    return Err("a gather of more than 128 entries".into());
                }
                let f = t.pool(idx.n, Dtype::F32)?;
                let (ff, sa) = (t.at(&f), t.at(&src));
                t.emit(WGATHER, Tr::fl(it, idx.n, false, false, false), ff, sa, ia, 0);
                if src.dtype == Dtype::F32 {
                    return Ok(vec![f]);
                }
                let d = t.pool(idx.n, rdt)?;
                let dd = t.at(&d);
                t.emit(QUN, Tr::fl(T_F32, idx.n, false, false, false), dd, ff, 0, U_CAST + tcode(rdt));
                return Ok(vec![d]);
            }
            let d = t.pool(idx.n, rdt)?;
            let (dd, sa) = (t.at(&d), t.at(&src));
            t.emit(QGATHER, Tr::fl(it, idx.n, false, false, false), dd, sa, ia, src.n);
            Ok(vec![d])
        }
        tags::SCATTER_ADD | tags::SCATTER_SET => {
            let (dst, idx, val) = (args[0], args[1], args[2]);
            let add = tag == tags::SCATTER_ADD;
            if Tr::is_wide(&idx) || Tr::is_wide(&val) {
                return Err("a vocabulary-wide scatter index or value".into());
            }
            let rest = t.pkg.values[op.args[0] as usize].shape.iter().skip(1).product::<u32>().max(1);
            if rest != 1 {
                return Err("a scatter of rows".into());
            }
            let scalar_val = val.n == 1 && idx.n != 1;
            let (ia, it) = (t.at(&idx), tcode(idx.dtype));
            if Tr::is_wide(&dst) {
                let v = t.pool_f32(val)?;
                let d = t.wide(rdt)?;
                let (dd, sa, va) = (t.at(&d), t.at(&dst), t.at(&v));
                t.emit(WMOV, 0, dd, sa, 0, 0);
                t.emit(WSCATTER, Tr::fl(it, idx.n, scalar_val, add, false), dd, ia, va, 0);
                return Ok(vec![d]);
            }
            let d = t.pool(dst.n, rdt)?;
            let (dd, sa, va) = (t.at(&d), t.at(&dst), t.at(&val));
            t.emit(QMOV, Tr::fl(0, dst.n, false, false, false), dd, sa, 0, 0);
            t.emit(QSCATTER, Tr::fl(it, idx.n, scalar_val, add, rdt == Dtype::F32), dd, ia, va, dst.n);
            Ok(vec![d])
        }
        tags::TOP_K => {
            let a = args[0];
            if !Tr::is_wide(&a) {
                return Err("top-k of a short value".into());
            }
            let k = op.imm;
            let v = t.pool(k, Dtype::F32)?;
            let i = t.pool(k, Dtype::U32)?;
            let (vv, ii, aa) = (t.at(&v), t.at(&i), t.at(&a));
            t.emit(TOPK, 0, vv, aa, ii, k);
            Ok(vec![v, i])
        }
        tags::PIVOT_THRESHOLD => {
            let a = args[0];
            if !Tr::is_wide(&a) {
                return Err("a pivot over a short value".into());
            }
            let p = *t.loc.get(&op.pred_payload).ok_or("the pivot's payload is untranslated")?;
            if Tr::is_wide(&p) || p.n != 1 {
                return Err("a per-row pivot payload".into());
            }
            let d = t.wide(Dtype::Bool)?;
            let (dd, aa, pp) = (t.at(&d), t.at(&a), t.at(&p));
            match op.pred_tag {
                0 => t.emit(PIVRANK, Tr::fl(tcode(p.dtype), 0, false, false, false), dd, aa, pp, 0),
                1 => t.emit(PIVMASS, 0, dd, aa, pp, 0),
                _ => t.emit(WCMP, Tr::fl(0, 0, false, true, false), dd, aa, pp, 1),
            }
            Ok(vec![d])
        }
        tags::CUMSUM | tags::CUMPROD => {
            let a = args[0];
            if Tr::is_wide(&a) {
                return Err("a cumulative sum over the vocabulary".into());
            }
            let a = t.pool_f32(a)?;
            let d = t.pool(a.n, Dtype::F32)?;
            let (dd, aa) = (t.at(&d), t.at(&a));
            t.emit(QCUM, Tr::fl(T_F32, a.n, false, false, false), dd, aa, 0, u32::from(tag == tags::CUMPROD));
            Ok(vec![d])
        }
        _ => Err("not in the guest ISA".into()),
    }
}

pub fn json_of(t: &Table) -> Json {
    json!({
        "table": t.ins,
        "pre_len": t.pre_len,
        "nreg": t.nreg,
        "pool_words": t.pool_words,
        "feat": t.feat,
        "tok_home": t.tok_home,
        "remat": t.remat,
        "homes": t.homes.iter().map(|h| json!({"chan": h.chan, "kind": h.kind, "at": h.at, "n": h.n, "dtype": format!("{:?}", h.dtype)})).collect::<Vec<_>>(),
        "outputs": t.outputs.iter().map(|(c, s, n, d)| json!({"chan": c, "slot": s, "n": n, "dtype": format!("{d:?}")})).collect::<Vec<_>>(),
        "wide_outputs": t.wide_outputs.iter().map(|(c, r, d)| json!({"chan": c, "regs": r, "dtype": format!("{d:?}")})).collect::<Vec<_>>(),
    })
}
