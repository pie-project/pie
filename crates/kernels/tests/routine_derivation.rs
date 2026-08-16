//! The routine machinery, exercised through a stand-in backend.
//!
//! What this proves is the one thing the rest of the refactor rests on: that a
//! table row DERIVES from a `fn`'s signature, in a `const` context, through a
//! macro that sees only the identifier. If this compiles, no routine's row can
//! disagree with its body, because there is one statement of it.

use kernels::Ty;
use kernels::routine::{Arg, Backend, Env, Provenance, Refusal, Routine};

/// The stand-in backend: an argument is one of three kinds, and the context
/// records what a body launched instead of launching it.
#[derive(Clone, Copy)]
struct Test;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    Ptr(usize),
    I32(i32),
    F32(f32),
}

#[derive(Default)]
struct Ctx {
    fired: std::cell::RefCell<Vec<&'static str>>,
}

impl Ctx {
    fn launch(&self, symbol: &'static str) -> Result<(), Refusal> {
        self.fired.borrow_mut().push(symbol);
        Ok(())
    }
}

impl Backend for Test {
    type Value = Value;
    type Ctx<'a> = Ctx;
}

/// The backend's own `routine!`, with its [`Backend`] filled in — the
/// three-line adapter every backend writes.
macro_rules! routine {
    ($body:ident $(, $($rest:tt)*)?) => {
        routine!(@go $body $(, $($rest)*)?)
    };
    (@go $($all:tt)*) => { ::kernels::routine!(Test, $($all)*) };
}

struct Bf16sMut(usize);
impl Arg<Test> for Bf16sMut {
    const TY: Ty = Ty::Bf16sMut;

    fn unpack(value: &Value, at: usize) -> Result<Self, Refusal> {
        match value {
            Value::Ptr(p) => Ok(Self(*p)),
            _ => Err(Refusal::Kind {
                at,
                want: <Self as Arg<Test>>::TY,
            }),
        }
    }
}

struct I32s(usize);
impl I32s {
    fn addr(&self) -> usize {
        self.0
    }
}
impl Arg<Test> for I32s {
    const TY: Ty = Ty::I32s;

    fn unpack(value: &Value, at: usize) -> Result<Self, Refusal> {
        match value {
            Value::Ptr(p) => Ok(Self(*p)),
            _ => Err(Refusal::Kind {
                at,
                want: <Self as Arg<Test>>::TY,
            }),
        }
    }
}

impl Arg<Test> for i32 {
    const TY: Ty = Ty::I32;

    fn unpack(value: &Value, at: usize) -> Result<Self, Refusal> {
        match value {
            Value::I32(v) => Ok(*v),
            _ => Err(Refusal::Kind {
                at,
                want: <Self as Arg<Test>>::TY,
            }),
        }
    }
}

impl Arg<Test> for f32 {
    const TY: Ty = Ty::F32;

    fn unpack(value: &Value, at: usize) -> Result<Self, Refusal> {
        match value {
            Value::F32(v) => Ok(*v),
            _ => Err(Refusal::Kind {
                at,
                want: <Self as Arg<Test>>::TY,
            }),
        }
    }
}

/// A routine, as one is actually written: an ordinary `fn`, refusing with a
/// value, choosing its symbol from a host-side fact.
fn rope_apply(
    ctx: &Ctx,
    q: Bf16sMut,
    positions: Env<I32s>,
    rows: i32,
    head_dim: i32,
    theta: f32,
) -> Result<(), Refusal> {
    let _ = theta;
    if q.0 == 0 {
        return Err(Refusal::Null { what: "q" });
    }
    // Through `Env`'s `Deref`, so a body reads an environment-supplied
    // argument exactly as it reads a stated one.
    if positions.addr() == 0 {
        return Err(Refusal::Null { what: "positions" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    match head_dim {
        64 => ctx.launch("rope::apply_bf16<64>"),
        128 => ctx.launch("rope::apply_bf16<128>"),
        d => Err(Refusal::Narrow {
            what: "head_dim",
            at: d.into(),
        }),
    }
}

/// A second routine at a different arity, so the marker really is doing the
/// disambiguating it exists for.
fn tanh_bf16(ctx: &Ctx, x: Bf16sMut, n: i32) -> Result<(), Refusal> {
    if x.0 == 0 {
        return Err(Refusal::Null { what: "x" });
    }
    if n <= 0 {
        return Err(Refusal::Empty { what: "n" });
    }
    ctx.launch("norm::tanh_bf16")
}

/// The table, in a `static` — which is the load-bearing claim: the rows are
/// `const`-promoted from generic associated consts, so nothing is built at
/// run time and nothing can be built inconsistently.
static ROUTINES: &[Routine<Test>] = &[
    routine!(rope_apply, in_place = &[(0, 0)]),
    routine!(tanh_bf16, whole, in_place = &[(0, 0)]),
];

fn find(name: &str) -> &'static Routine<Test> {
    ROUTINES
        .iter()
        .find(|r| r.name == name)
        .expect("a routine this test declares")
}

#[test]
fn a_row_is_its_fns_signature() {
    let rope = find("rope_apply");
    assert_eq!(
        rope.args,
        &[
            (Ty::Bf16sMut, Provenance::Trace),
            (Ty::I32s, Provenance::Env),
            (Ty::I32, Provenance::Trace),
            (Ty::I32, Provenance::Trace),
            (Ty::F32, Provenance::Trace),
        ],
        "the row is read off the parameter list, `Env` and all"
    );
    assert_eq!(
        find("tanh_bf16").args.len(),
        2,
        "and a different arity is a different row"
    );
}

#[test]
fn the_stated_facts_are_the_ones_stated() {
    let rope = find("rope_apply");
    assert!(!rope.whole, "unstated is false, not absent");
    assert_eq!(rope.in_place, &[(0, 0)]);

    let tanh = find("tanh_bf16");
    assert!(tanh.whole);
    assert!(!tanh.depth_prefix_plan);
}

#[test]
fn the_erased_body_is_the_typed_one() {
    let ctx = Ctx::default();
    let args = [
        Value::Ptr(0x1000),
        Value::Ptr(0x2000),
        Value::I32(4),
        Value::I32(64),
        Value::F32(1e4),
    ];
    (find("rope_apply").body)(&ctx, &args).expect("a live rectangle launches");
    assert_eq!(
        *ctx.fired.borrow(),
        ["rope::apply_bf16<64>"],
        "the symbol the body chose"
    );
}

#[test]
fn a_refusal_from_the_body_survives_the_erasure() {
    let ctx = Ctx::default();
    let empty = [
        Value::Ptr(0x1000),
        Value::Ptr(0x2000),
        Value::I32(0),
        Value::I32(64),
        Value::F32(1e4),
    ];
    assert_eq!(
        (find("rope_apply").body)(&ctx, &empty),
        Err(Refusal::Empty { what: "rows" }),
        "the body's own word for it, not a generic failure"
    );
    assert!(ctx.fired.borrow().is_empty(), "and nothing launched");
}

#[test]
fn a_list_that_does_not_fit_the_signature_is_refused() {
    let ctx = Ctx::default();
    assert_eq!(
        (find("rope_apply").body)(&ctx, &[Value::I32(1)]),
        Err(Refusal::Arity { want: 5, got: 1 })
    );
    let swapped = [
        Value::Ptr(0x1000),
        Value::Ptr(0x2000),
        Value::F32(4.0),
        Value::I32(64),
        Value::F32(1e4),
    ];
    assert_eq!(
        (find("rope_apply").body)(&ctx, &swapped),
        Err(Refusal::Kind {
            at: 2,
            want: Ty::I32
        }),
        "same width, different kind -- the position is named"
    );
}
