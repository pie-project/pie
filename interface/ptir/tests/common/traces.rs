//! Shared trace builders for the PTIR integration tests and golden vectors.
//! (`#[path]`-included by `ptir_examples.rs` and `ptir_golden.rs` — one
//! definition, so the golden files and the example tests cannot drift.)
#![allow(dead_code)]

use pie_ptir::interp::Value;
use pie_ptir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use pie_ptir::expand;
use pie_ptir::op::{IntrinsicId, Op};
use pie_ptir::registry::{ModelProfile, Port, Stage};
use pie_ptir::types::{DType, Literal, Shape};

/// Tiny SSA builder: push an op, get its first result id.
pub struct B {
    pub ops: Vec<Op>,
}
impl B {
    pub fn new() -> B {
        B { ops: Vec::new() }
    }
    pub fn p(&mut self, op: Op) -> u32 {
        let id = expand::next_id(&self.ops);
        self.ops.push(op);
        id
    }
    pub fn cu32(&mut self, v: u32) -> u32 {
        self.p(Op::Const(Literal::U32(v)))
    }
}

pub fn chan(shape: Shape, dtype: DType, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl { shape, dtype: ChanDType::Concrete(dtype), capacity: 1, host_role, seeded }
}

pub fn u32s(v: &[u32]) -> Value {
    Value::U32(v.to_vec())
}
pub fn i32s(v: &[i32]) -> Value {
    Value::I32(v.to_vec())
}
pub fn f32s(v: &[f32]) -> Value {
    Value::F32(v.to_vec())
}
pub fn bools(v: &[bool]) -> Value {
    Value::Bool(v.to_vec())
}

pub fn const_port(port: Port, dtype: DType, shape: Shape, words: &[u32]) -> PortBinding {
    PortBinding {
        port,
        source: PortSource::Const {
            dtype,
            shape,
            data: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Overview §3 — masked gumbel-greedy decode
// ═══════════════════════════════════════════════════════════════════════════

pub const VOCAB: u32 = 32;

/// Channels: 0 tok (loop), 1 out (host-read), 2 mask (host-fed bool[vocab]),
/// 3 len (in-graph counter), 4 rng (state [key,ctr]).
pub fn section3_trace() -> TraceContainer {
    let mut b = B::new();
    let logits2 = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(1, VOCAB),
        dtype: DType::F32,
    });
    let logits = b.p(Op::Reshape { value: logits2, shape: Shape::vector(VOCAB) });
    let r = b.p(Op::ChanTake(4)); // rng state [key, ctr]
    let m = b.p(Op::ChanTake(2)); // mask (readiness input)
    let g = expand::gumbel(&mut b.ops, r, Shape::vector(VOCAB));
    let masked = expand::mask_apply(&mut b.ops, logits, m);
    let sum = b.p(Op::Add(masked, g));
    let t = b.p(Op::ReduceArgmax(sum)); // scalar i32
    // rng.put(add(r, [0,1])) — counter advances in-graph (ping-pong)
    let ctr1 = b.p(Op::Iota { len: 2 }); // [0, 1] u32
    let r2 = b.p(Op::Add(r, ctr1));
    b.p(Op::ChanPut { chan: 4, value: r2 });
    let t1 = b.p(Op::Reshape { value: t, shape: Shape::vector(1) });
    b.p(Op::ChanPut { chan: 0, value: t1 }); // tok
    let l = b.p(Op::ChanTake(3));
    let one = b.cu32(1);
    let l2 = b.p(Op::Add(l, one));
    b.p(Op::ChanPut { chan: 3, value: l2 }); // len
    b.p(Op::ChanPut { chan: 1, value: t1 }); // out

    TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::vector(1), DType::I32, HostRole::None, true), // 0 tok
            chan(Shape::vector(1), DType::I32, HostRole::Reader, false), // 1 out
            chan(Shape::vector(VOCAB), DType::Bool, HostRole::Writer, false), // 2 mask
            chan(Shape::vector(1), DType::U32, HostRole::None, true), // 3 len
            chan(Shape::vector(2), DType::U32, HostRole::None, true), // 4 rng
        ],
        ports: vec![
            PortBinding { port: Port::EmbedTokens, source: PortSource::Channel(0) },
            const_port(Port::EmbedIndptr, DType::U32, Shape::vector(2), &[0, 1]),
            PortBinding { port: Port::KvLen, source: PortSource::Channel(3) },
        ],
        stages: vec![StageProgram { stage: Stage::Epilogue, ops: b.ops }],
        externs: Vec::new(),
    }
}

pub fn allow_all() -> Value {
    bools(&[true; VOCAB as usize])
}
pub fn allow_only(toks: &[usize]) -> Value {
    let mut m = [false; VOCAB as usize];
    for &t in toks {
        m[t] = true;
    }
    bools(&m)
}
pub fn flat_logits(fav: usize, x: f32) -> Value {
    let mut l = vec![0.0f32; VOCAB as usize];
    l[fav] = x;
    Value::F32(l)
}

// ═══════════════════════════════════════════════════════════════════════════
// Overview §6.2 — beam epilogue (reorder = gathers, divergence = freeze)
// ═══════════════════════════════════════════════════════════════════════════

pub const BB: u32 = 2; // beams
pub const V: u32 = 8; // vocab
pub const P: u32 = 3; // page slots per row
pub const PAGE: u32 = 4; // tokens per page

/// Channels (indices as in overview §6.2):
/// 0 pages [B,P] u32 · 1 lens [B,P] u32 · 2 klen [B] u32 · 3 kvm [B,P*page]
/// bool · 4 pos [B] u32 · 5 np [B] u32 · 6 tslot [B] u32 · 7 tfill [B] u32 ·
/// 8 w_slot [B] u32 · 9 w_off [B] u32 · 10 toks [B] i32 · 11 scores [B] f32 ·
/// 12 fresh [B] u32 (host-fed) · 13 out [B] i32 · 14 out_par [B] u32 ·
/// 15 out_scr [B] f32.
pub fn beam_trace() -> TraceContainer {
    let mut b = B::new();
    let scores = b.p(Op::ChanTake(11)); // [B]
    let logits = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(BB, V),
        dtype: DType::F32,
    });
    let lsm = expand::log_softmax(&mut b.ops, logits, Shape::matrix(BB, V));
    // broadcast [B] -> [B,V] must left-align as [B,1]; reshape first.
    let s1 = b.p(Op::Reshape { value: scores, shape: Shape::matrix(BB, 1) });
    let sbb = b.p(Op::Broadcast { value: s1, shape: Shape::matrix(BB, V) });
    let cand = b.p(Op::Add(sbb, lsm));
    let candf = b.p(Op::Reshape { value: cand, shape: Shape::vector(BB * V) });
    let s = b.p(Op::TopK { input: candf, k: BB }); // s, s+1 = i
    let i = s + 1;
    let vc = b.cu32(V);
    let parent = b.p(Op::Div(i, vc)); // integer division: flat id → row
    // reorder = row gathers
    let pg = {
        let t = b.p(Op::ChanTake(0));
        b.p(Op::Gather { src: t, idx: parent })
    };
    let pl = {
        let t = b.p(Op::ChanTake(1));
        b.p(Op::Gather { src: t, idx: parent })
    };
    let n = {
        let t = b.p(Op::ChanTake(5));
        b.p(Op::Gather { src: t, idx: parent })
    };
    let tf = {
        let t = b.p(Op::ChanTake(7));
        b.p(Op::Gather { src: t, idx: parent })
    };
    let lanes = b.p(Op::Iota { len: BB });
    // heir[p] = p's designated child (duplicate scatters: last wins)
    let heir = b.p(Op::ScatterSet { base: lanes, idx: parent, vals: lanes });
    let hp = b.p(Op::Gather { src: heir, idx: parent });
    let is_heir = b.p(Op::Eq(hp, lanes));
    let pagec = b.cu32(PAGE);
    let has_room = b.p(Op::Lt(tf, pagec));
    let cont = b.p(Op::And(is_heir, has_room));
    let tslot_t = b.p(Op::ChanTake(6));
    let tsp = b.p(Op::Gather { src: tslot_t, idx: parent });
    let fresh = b.p(Op::ChanTake(12));
    let slot = b.p(Op::Select { cond: cont, a: tsp, b: fresh });
    let zero = b.cu32(0);
    let off = b.p(Op::Select { cond: cont, a: tf, b: zero });
    let one = b.cu32(1);
    let n1 = b.p(Op::Add(n, one));
    let n2 = b.p(Op::Select { cond: cont, a: n, b: n1 });
    let pc = b.cu32(P);
    let n2m1 = b.p(Op::Sub(n2, one));
    let tcol = {
        let t = b.p(Op::Mul(lanes, pc));
        b.p(Op::Add(t, n2m1))
    };
    // pages
    let pgf = b.p(Op::Reshape { value: pg, shape: Shape::vector(BB * P) });
    let pg2 = b.p(Op::ScatterSet { base: pgf, idx: tcol, vals: slot });
    let pg3 = b.p(Op::Reshape { value: pg2, shape: Shape::matrix(BB, P) });
    b.p(Op::ChanPut { chan: 0, value: pg3 });
    // lens (the single source) + derivatives
    let off1 = b.p(Op::Add(off, one));
    let plf = b.p(Op::Reshape { value: pl, shape: Shape::vector(BB * P) });
    let pl2f = b.p(Op::ScatterSet { base: plf, idx: tcol, vals: off1 });
    let pl2 = b.p(Op::Reshape { value: pl2f, shape: Shape::matrix(BB, P) });
    b.p(Op::ChanPut { chan: 1, value: pl2 });
    let klen = {
        let t = b.p(Op::Mul(n2m1, pagec));
        b.p(Op::Add(t, off1))
    };
    // klen/kvm are pure derivatives of `lens`, recomputed each step: drain
    // the stale cell (take, value unused) before refilling — under §1's
    // capacity-1 full/empty bits a put without a drain would back-pressure
    // forever on step 2. (Overview §6.2 elides the drain; the trace can't.)
    b.p(Op::ChanTake(2));
    b.p(Op::ChanPut { chan: 2, value: klen });
    // kvm[b][j*page+o] = o < lens[b][j]
    let io = b.p(Op::Iota { len: PAGE });
    let io3 = b.p(Op::Reshape { value: io, shape: Shape::new(&[1, 1, PAGE]).unwrap() });
    let iob = b.p(Op::Broadcast { value: io3, shape: Shape::new(&[BB, P, PAGE]).unwrap() });
    let l3 = b.p(Op::Reshape { value: pl2, shape: Shape::new(&[BB, P, 1]).unwrap() });
    let lb = b.p(Op::Broadcast { value: l3, shape: Shape::new(&[BB, P, PAGE]).unwrap() });
    let kvm3 = b.p(Op::Lt(iob, lb));
    let kvm = b.p(Op::Reshape { value: kvm3, shape: Shape::matrix(BB, P * PAGE) });
    b.p(Op::ChanTake(3)); // drain (see klen note)
    b.p(Op::ChanPut { chan: 3, value: kvm });
    // pos (logical length, ping-pong)
    let pos = b.p(Op::ChanTake(4));
    let pos2 = b.p(Op::Add(pos, one));
    b.p(Op::ChanPut { chan: 4, value: pos2 });
    // bookkeeping
    b.p(Op::ChanPut { chan: 5, value: n2 });
    b.p(Op::ChanPut { chan: 6, value: slot });
    b.p(Op::ChanPut { chan: 7, value: off1 });
    b.p(Op::ChanPut { chan: 8, value: slot });
    b.p(Op::ChanPut { chan: 9, value: off });
    // tokens + scores + host-facing
    let tok_u = b.p(Op::Rem(i, vc));
    let tok = b.p(Op::Cast { value: tok_u, dtype: DType::I32 });
    b.p(Op::ChanPut { chan: 10, value: tok });
    b.p(Op::ChanPut { chan: 11, value: s });
    b.p(Op::ChanPut { chan: 13, value: tok });
    b.p(Op::ChanPut { chan: 14, value: parent });
    b.p(Op::ChanPut { chan: 15, value: s });

    let u32c = |shape: Shape, role, seeded| chan(shape, DType::U32, role, seeded);
    TraceContainer {
        names: vec![],
        channels: vec![
            u32c(Shape::matrix(BB, P), HostRole::None, true), // 0 pages
            u32c(Shape::matrix(BB, P), HostRole::None, true), // 1 lens
            u32c(Shape::vector(BB), HostRole::None, true),    // 2 klen
            chan(Shape::matrix(BB, P * PAGE), DType::Bool, HostRole::None, true), // 3 kvm
            u32c(Shape::vector(BB), HostRole::None, true),    // 4 pos
            u32c(Shape::vector(BB), HostRole::None, true),    // 5 np
            u32c(Shape::vector(BB), HostRole::None, true),    // 6 tslot
            u32c(Shape::vector(BB), HostRole::None, true),    // 7 tfill
            u32c(Shape::vector(BB), HostRole::None, true),    // 8 w_slot
            u32c(Shape::vector(BB), HostRole::None, true),    // 9 w_off
            chan(Shape::vector(BB), DType::I32, HostRole::None, true), // 10 toks
            chan(Shape::vector(BB), DType::F32, HostRole::None, true), // 11 scores
            u32c(Shape::vector(BB), HostRole::Writer, false), // 12 fresh
            chan(Shape::vector(BB), DType::I32, HostRole::Reader, false), // 13 out
            u32c(Shape::vector(BB), HostRole::Reader, false), // 14 out_par
            chan(Shape::vector(BB), DType::F32, HostRole::Reader, false), // 15 out_scr
        ],
        ports: vec![
            PortBinding { port: Port::EmbedTokens, source: PortSource::Channel(10) },
            const_port(Port::EmbedIndptr, DType::U32, Shape::vector(BB + 1), &[0, 1, 2]),
            PortBinding { port: Port::Positions, source: PortSource::Channel(4) },
            PortBinding { port: Port::Pages, source: PortSource::Channel(0) },
            const_port(Port::PageIndptr, DType::U32, Shape::vector(BB + 1), &[0, P, 2 * P]),
            PortBinding { port: Port::KvLen, source: PortSource::Channel(2) },
            PortBinding { port: Port::WSlot, source: PortSource::Channel(8) },
            PortBinding { port: Port::WOff, source: PortSource::Channel(9) },
            PortBinding { port: Port::AttnMask, source: PortSource::Channel(3) },
        ],
        stages: vec![StageProgram { stage: Stage::Epilogue, ops: b.ops }],
        externs: Vec::new(),
    }
}

pub fn beam_profile() -> ModelProfile {
    ModelProfile { vocab: V, page_size: PAGE, num_layers: 2, ..ModelProfile::dummy() }
}
