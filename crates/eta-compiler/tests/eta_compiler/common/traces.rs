#![allow(dead_code)]

use eta_compiler::eval::interp::Value;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::expand;
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Port, Stage};
use eta_ir::types::{Dtype, Literal, Shape};

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

pub fn chan(shape: Shape, dtype: Dtype, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    }
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

pub fn const_port(port: Port, dtype: Dtype, shape: Shape, words: &[u32]) -> PortBinding {
    PortBinding {
        port,
        source: PortSource::Const {
            dtype,
            shape,
            data: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        },
    }
}

pub const VOCAB: u32 = 32;

pub fn section3_trace() -> TraceContainer {
    let mut b = B::new();
    let logits2 = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(1, VOCAB),
        dtype: Dtype::F32,
    });
    let logits = b.p(Op::Reshape {
        value: logits2,
        shape: Shape::vector(VOCAB),
    });
    let r = b.p(Op::ChanTake(4));
    let m = b.p(Op::ChanTake(2));
    let g = expand::gumbel(&mut b.ops, r, Shape::vector(VOCAB));
    let masked = expand::mask_apply(&mut b.ops, logits, m);
    let sum = b.p(Op::Add(masked, g));
    let t = b.p(Op::ReduceArgmax(sum));
    let ctr1 = b.p(Op::Iota { len: 2 });
    let r2 = b.p(Op::Add(r, ctr1));
    b.p(Op::ChanPut { chan: 4, value: r2 });
    let t1 = b.p(Op::Reshape {
        value: t,
        shape: Shape::vector(1),
    });
    b.p(Op::ChanPut { chan: 0, value: t1 });
    let l = b.p(Op::ChanTake(3));
    let one = b.cu32(1);
    let l2 = b.p(Op::Add(l, one));
    b.p(Op::ChanPut { chan: 3, value: l2 });
    b.p(Op::ChanPut { chan: 1, value: t1 });

    TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::vector(1), Dtype::I32, HostRole::None, true),
            chan(Shape::vector(1), Dtype::I32, HostRole::Reader, false),
            chan(Shape::vector(VOCAB), Dtype::Bool, HostRole::Writer, false),
            chan(Shape::vector(1), Dtype::U32, HostRole::None, true),
            chan(Shape::vector(2), Dtype::U32, HostRole::None, true),
        ],
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(0),
            },
            const_port(Port::EmbedIndptr, Dtype::U32, Shape::vector(2), &[0, 1]),
            const_port(Port::Positions, Dtype::U32, Shape::vector(1), &[0]),
            const_port(Port::Pages, Dtype::U32, Shape::vector(1), &[0]),
            const_port(Port::PageIndptr, Dtype::U32, Shape::vector(2), &[0, 1]),
            PortBinding {
                port: Port::KvLen,
                source: PortSource::Channel(3),
            },
            const_port(Port::WSlot, Dtype::U32, Shape::vector(1), &[0]),
            const_port(Port::WOff, Dtype::U32, Shape::vector(1), &[0]),
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: b.ops,
        }],
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

pub const BB: u32 = 2;
pub const V: u32 = 8;
pub const P: u32 = 3;
pub const PAGE: u32 = 4;

pub fn beam_trace() -> TraceContainer {
    let mut b = B::new();
    let scores = b.p(Op::ChanTake(11));
    let logits = b.p(Op::IntrinsicVal {
        intr: IntrinsicId::Logits,
        shape: Shape::matrix(BB, V),
        dtype: Dtype::F32,
    });
    let lsm = expand::log_softmax(&mut b.ops, logits, Shape::matrix(BB, V));
    let s1 = b.p(Op::Reshape {
        value: scores,
        shape: Shape::matrix(BB, 1),
    });
    let sbb = b.p(Op::Broadcast {
        value: s1,
        shape: Shape::matrix(BB, V),
    });
    let cand = b.p(Op::Add(sbb, lsm));
    let candf = b.p(Op::Reshape {
        value: cand,
        shape: Shape::vector(BB * V),
    });
    let s = b.p(Op::TopK {
        input: candf,
        k: BB,
    });
    let i = s + 1;
    let vc = b.cu32(V);
    let parent = b.p(Op::Div(i, vc));
    let pg = {
        let t = b.p(Op::ChanTake(0));
        b.p(Op::Gather {
            src: t,
            idx: parent,
        })
    };
    let pl = {
        let t = b.p(Op::ChanTake(1));
        b.p(Op::Gather {
            src: t,
            idx: parent,
        })
    };
    let n = {
        let t = b.p(Op::ChanTake(5));
        b.p(Op::Gather {
            src: t,
            idx: parent,
        })
    };
    let tf = {
        let t = b.p(Op::ChanTake(7));
        b.p(Op::Gather {
            src: t,
            idx: parent,
        })
    };
    let lanes = b.p(Op::Iota { len: BB });
    let heir = b.p(Op::ScatterSet {
        base: lanes,
        idx: parent,
        vals: lanes,
    });
    let hp = b.p(Op::Gather {
        src: heir,
        idx: parent,
    });
    let is_heir = b.p(Op::Eq(hp, lanes));
    let pagec = b.cu32(PAGE);
    let has_room = b.p(Op::Lt(tf, pagec));
    let cont = b.p(Op::And(is_heir, has_room));
    let tslot_t = b.p(Op::ChanTake(6));
    let tsp = b.p(Op::Gather {
        src: tslot_t,
        idx: parent,
    });
    let fresh = b.p(Op::ChanTake(12));
    let slot = b.p(Op::Select {
        cond: cont,
        a: tsp,
        b: fresh,
    });
    let zero = b.cu32(0);
    let off = b.p(Op::Select {
        cond: cont,
        a: tf,
        b: zero,
    });
    let one = b.cu32(1);
    let n1 = b.p(Op::Add(n, one));
    let n2 = b.p(Op::Select {
        cond: cont,
        a: n,
        b: n1,
    });
    let pc = b.cu32(P);
    let n2m1 = b.p(Op::Sub(n2, one));
    let tcol = {
        let t = b.p(Op::Mul(lanes, pc));
        b.p(Op::Add(t, n2m1))
    };
    let pgf = b.p(Op::Reshape {
        value: pg,
        shape: Shape::vector(BB * P),
    });
    let pg2 = b.p(Op::ScatterSet {
        base: pgf,
        idx: tcol,
        vals: slot,
    });
    let pg3 = b.p(Op::Reshape {
        value: pg2,
        shape: Shape::matrix(BB, P),
    });
    b.p(Op::ChanPut {
        chan: 0,
        value: pg3,
    });
    let off1 = b.p(Op::Add(off, one));
    let plf = b.p(Op::Reshape {
        value: pl,
        shape: Shape::vector(BB * P),
    });
    let pl2f = b.p(Op::ScatterSet {
        base: plf,
        idx: tcol,
        vals: off1,
    });
    let pl2 = b.p(Op::Reshape {
        value: pl2f,
        shape: Shape::matrix(BB, P),
    });
    b.p(Op::ChanPut {
        chan: 1,
        value: pl2,
    });
    let klen = {
        let t = b.p(Op::Mul(n2m1, pagec));
        b.p(Op::Add(t, off1))
    };
    b.p(Op::ChanTake(2));
    b.p(Op::ChanPut {
        chan: 2,
        value: klen,
    });
    let io = b.p(Op::Iota { len: PAGE });
    let io3 = b.p(Op::Reshape {
        value: io,
        shape: Shape::new(&[1, 1, PAGE]).unwrap(),
    });
    let iob = b.p(Op::Broadcast {
        value: io3,
        shape: Shape::new(&[BB, P, PAGE]).unwrap(),
    });
    let l3 = b.p(Op::Reshape {
        value: pl2,
        shape: Shape::new(&[BB, P, 1]).unwrap(),
    });
    let lb = b.p(Op::Broadcast {
        value: l3,
        shape: Shape::new(&[BB, P, PAGE]).unwrap(),
    });
    let kvm3 = b.p(Op::Lt(iob, lb));
    let kvm = b.p(Op::Reshape {
        value: kvm3,
        shape: Shape::matrix(BB, P * PAGE),
    });
    b.p(Op::ChanTake(3));
    b.p(Op::ChanPut {
        chan: 3,
        value: kvm,
    });
    let pos = b.p(Op::ChanTake(4));
    let pos2 = b.p(Op::Add(pos, one));
    b.p(Op::ChanPut {
        chan: 4,
        value: pos2,
    });
    b.p(Op::ChanPut { chan: 5, value: n2 });
    b.p(Op::ChanPut {
        chan: 6,
        value: slot,
    });
    b.p(Op::ChanPut {
        chan: 7,
        value: off1,
    });
    b.p(Op::ChanPut {
        chan: 8,
        value: slot,
    });
    b.p(Op::ChanPut {
        chan: 9,
        value: off,
    });
    let tok_u = b.p(Op::Rem(i, vc));
    let tok = b.p(Op::Cast {
        value: tok_u,
        dtype: Dtype::I32,
    });
    b.p(Op::ChanPut {
        chan: 10,
        value: tok,
    });
    b.p(Op::ChanPut { chan: 11, value: s });
    b.p(Op::ChanPut {
        chan: 13,
        value: tok,
    });
    b.p(Op::ChanPut {
        chan: 14,
        value: parent,
    });
    b.p(Op::ChanPut { chan: 15, value: s });

    let u32c = |shape: Shape, role, seeded| chan(shape, Dtype::U32, role, seeded);
    TraceContainer {
        names: vec![],
        channels: vec![
            u32c(Shape::matrix(BB, P), HostRole::None, true),
            u32c(Shape::matrix(BB, P), HostRole::None, true),
            u32c(Shape::vector(BB), HostRole::None, true),
            chan(
                Shape::matrix(BB, P * PAGE),
                Dtype::Bool,
                HostRole::None,
                true,
            ),
            u32c(Shape::vector(BB), HostRole::None, true),
            u32c(Shape::vector(BB), HostRole::None, true),
            u32c(Shape::vector(BB), HostRole::None, true),
            u32c(Shape::vector(BB), HostRole::None, true),
            u32c(Shape::vector(BB), HostRole::None, true),
            u32c(Shape::vector(BB), HostRole::None, true),
            chan(Shape::vector(BB), Dtype::I32, HostRole::None, true),
            chan(Shape::vector(BB), Dtype::F32, HostRole::None, true),
            u32c(Shape::vector(BB), HostRole::Writer, false),
            chan(Shape::vector(BB), Dtype::I32, HostRole::Reader, false),
            u32c(Shape::vector(BB), HostRole::Reader, false),
            chan(Shape::vector(BB), Dtype::F32, HostRole::Reader, false),
        ],
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(10),
            },
            const_port(
                Port::EmbedIndptr,
                Dtype::U32,
                Shape::vector(BB + 1),
                &[0, 1, 2],
            ),
            PortBinding {
                port: Port::Positions,
                source: PortSource::Channel(4),
            },
            PortBinding {
                port: Port::Pages,
                source: PortSource::Channel(0),
            },
            const_port(
                Port::PageIndptr,
                Dtype::U32,
                Shape::vector(BB + 1),
                &[0, P, 2 * P],
            ),
            PortBinding {
                port: Port::KvLen,
                source: PortSource::Channel(2),
            },
            PortBinding {
                port: Port::WSlot,
                source: PortSource::Channel(8),
            },
            PortBinding {
                port: Port::WOff,
                source: PortSource::Channel(9),
            },
            PortBinding {
                port: Port::AttnMask,
                source: PortSource::Channel(3),
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: b.ops,
        }],
        externs: Vec::new(),
    }
}

pub fn beam_profile() -> ModelProfile {
    ModelProfile {
        vocab: V,
        page_size: PAGE,
        num_layers: 2,
        ..ModelProfile::dummy()
    }
}
