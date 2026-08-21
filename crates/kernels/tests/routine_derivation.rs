//! The routine machinery, exercised through a stand-in backend.
//!
//! What this proves is the one thing the rest of the refactor rests on: that a
//! table row DERIVES from a `fn`'s signature, in a `const` context, through a
//! macro that sees only the identifier. If this compiles, no routine's row can
//! disagree with its body, because there is one statement of it.
//!
//! # What it proves that it could not before
//!
//! That there are FOUR marks and every one of them is a quality. `Env` named a
//! supplier where the others named a direction, and it was the one mark that
//! was not positional; `Weight` was the one domain noun in a set of qualities.
//! Both are gone, so every parameter below claims a slot in one of the
//! statement's four runs — operands, results, weights, params — and the index
//! is its position among the marks rather than a number anyone wrote down.

use kernels::routine::{
    Answers, Arg, Asks, Backend, Const, Extent, In, InOut, Out, Refusal, Routine,
};
use kernels::{Kind, Source, Ty, keys};

/// The stand-in backend: an argument is one of three kinds, and the context
/// records what a body launched instead of launching it.
///
/// `Ptr(usize)` STOOD HERE and was the fourth. A bare address is what an
/// operand was before a statement placed a RECTANGLE at one, and nothing in
/// this fixture builds a bare address any more -- `Region` carries the rows
/// and the width a mark unpacks. The two arms that read it were spelled
/// `Value::Region { ptr: p, .. }`, so its half never ran.
#[derive(Clone, Copy)]
struct Test;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    I32(i32),
    F32(f32),
    /// An address the statement placed, WITH the shape it placed it at --
    /// what a mark unpacks a rectangle from.
    Region {
        ptr: usize,
        rows: i32,
        width: i32,
    },
}

// A PLANE WITH NO ABSENT VALUE, which is a legal answer and the one this
// fixture gives: `Option<In<..>>` unpacks as `Some` everywhere here, because
// nothing this backend binds can stand for "the statement placed nothing".
// The trait's defaults say exactly that, so the impl is empty on purpose.
impl kernels::routine::Absent for Value {}

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

// THIS FIRE'S ANSWERS, for a body that asks. A stand-in for what a driver
// lends: `keys::Rows` is the fire's token count and nothing else is answered,
// so a body that reaches for a fact this backend does not hold is refused with
// `Unstated` rather than handed a zero.
impl Answers<Test> for Ctx {
    fn resolve(&self, _ty: Ty, source: Source) -> Result<Value, Refusal> {
        match source {
            Source::Named(k) if k == <keys::Rows as keys::Fact>::KEY => Ok(Value::I32(4)),
            _ => Err(Refusal::Unstated {
                what: "a fact this backend does not answer",
            }),
        }
    }
}

impl Backend for Test {
    type Value = Value;
    type Ctx<'a> = Ctx;

    fn region(value: &Value) -> Result<Extent, Refusal> {
        match *value {
            Value::Region { rows, width, .. } => Ok(Extent { rows, width }),
            _ => Err(Refusal::Absent {
                what: "a region's shape",
            }),
        }
    }
}

/// WHAT `#[routine]` REGISTERS INTO, and the name it calls this backend by.
///
/// Both are the three-line adapter every plane writes: the attribute emits
/// `crate::Plane` for the backend and puts the row in `crate::ROUTINES`, which
/// the linker gathers. That is the last hand-written thing about a routine — a
/// membership line whose omission left the routine compiled, correct and
/// unreachable, with nothing to report it.
type Plane = Test;

// The declaration wears a name of its own and the ALIAS wears the one
// `#[routine]` emits, for the reason each plane's `lib.rs` states at length:
// `linkme` keys a slice on the STATIC's identifier, which is global across
// every crate linked into one binary.
#[linkme::distributed_slice]
static TEST_ROUTINES: [Routine<Test>];

use TEST_ROUTINES as ROUTINES;

/// The backend's own `routine!`, with its [`Backend`] filled in — the
/// three-line adapter every backend writes.
macro_rules! routine {
    ($body:ident $(, $($rest:tt)*)?) => {
        routine!(@go $body $(, $($rest)*)?)
    };
    (@go $($all:tt)*) => { ::kernels::routine!(Test, $($all)*) };
}

/// This backend's tensor constructor, and the whole of what makes `Tensor<E>`
/// per-plane: the innards differ, the written form does not.
struct Tensor<E>(core::marker::PhantomData<E>);

/// The activation element, as this backend spells it.
///
/// It carries the two bytes a bf16 actually is, and that is not decoration:
/// a unit struct is zero-sized, and `<*const Zst>::add` is a no-op that
/// `clippy::zst_offset` denies outright -- so a fixture standing in for a
/// real element has to have the size of one, or `advance_read` below is
/// deriving its answer from arithmetic that cannot move.
struct Bf16(
    #[expect(
        dead_code,
        reason = "a pointee is never constructed; \
    the field is here for its SIZE, which is what makes pointer arithmetic \
    on it mean anything"
    )]
    u16,
);

impl kernels::Elem for Tensor<Bf16> {
    type Read = *const Bf16;
    type Write = *mut Bf16;

    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
        unsafe { read.add(elems) }
    }

    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
        unsafe { write.add(elems) }
    }

    const CPP_CONST: &'static str = "const __nv_bfloat16*";
    const CPP_MUT: &'static str = "__nv_bfloat16*";
    const TY_CONST: Ty = Ty::Bf16s;
    const TY_MUT: Ty = Ty::Bf16sMut;
}

// A TENSOR IS THE WEIGHT RUN'S CARRIER, which is the half of `Const` that is
// not a scalar.
impl kernels::ConstRun for Tensor<Bf16> {
    const RUN: kernels::routine::Claim = kernels::routine::Claim::Weight;
    const TY: Ty = Ty::Bf16s;
    type Held = *const Bf16;
}

impl Arg<Test> for *const Bf16 {
    const TY: Ty = Ty::Bf16s;

    fn unpack(value: &Value, at: usize) -> Result<Self, Refusal> {
        match value {
            Value::Region { ptr: p, .. } => Ok(core::ptr::without_provenance(*p)),
            _ => Err(Refusal::Kind {
                at,
                want: <Self as Arg<Test>>::TY,
            }),
        }
    }
}

impl Arg<Test> for *mut Bf16 {
    const TY: Ty = Ty::Bf16sMut;

    fn unpack(value: &Value, at: usize) -> Result<Self, Refusal> {
        match value {
            Value::Region { ptr: p, .. } => Ok(core::ptr::without_provenance_mut(*p)),
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

/// A routine, as one is actually written after the migration.
///
/// Four parameters and every one of them positional: `q` is the operand the
/// statement placed and also the result it declared — one address in both runs
/// — and the three scalars are the checkpoint's, which the statement carries
/// in its params run. `positions` is the only thing this batch decided, so it
/// is the only thing asked for.
#[kernels_macros::routine]
fn rope_apply(
    ctx: &Ctx,
    q: InOut<Tensor<Bf16>>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    theta: Const<f32>,
) -> Result<(), Refusal> {
    let _ = (*rotary, *theta);
    if q.ptr.is_null() {
        return Err(Refusal::Null { what: "q" });
    }
    // THE ONE FACT A FIRE DECIDES, asked for rather than declared.
    let rows = ctx.ask::<i32, keys::Rows>()?;
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    match *head_dim {
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
#[kernels_macros::routine(whole)]
fn tanh_bf16(ctx: &Ctx, x: InOut<Tensor<Bf16>>, n: Const<i32>) -> Result<(), Refusal> {
    if x.ptr.is_null() {
        return Err(Refusal::Null { what: "x" });
    }
    if *n <= 0 {
        return Err(Refusal::Empty { what: "n" });
    }
    ctx.launch("norm::tanh_bf16")
}

/// A THIRD routine, whose destination arrives whole.
///
/// The rectangle is not a pair of extra parameters: it rides with the address,
/// and the body reads `y.rows` off the value the statement placed. `x` is the
/// statement's SECOND operand and nobody wrote the index down — `y` wears both
/// an operand slot and a result slot, so the counter has already moved.
#[kernels_macros::routine]
fn residual_add(ctx: &Ctx, y: InOut<Tensor<Bf16>>, x: In<Tensor<Bf16>>) -> Result<(), Refusal> {
    if y.ptr.is_null() || x.ptr.is_null() {
        return Err(Refusal::Null { what: "a region" });
    }
    if y.rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if y.width != x.width {
        return Err(Refusal::Narrow {
            what: "width",
            at: x.width.into(),
        });
    }
    ctx.launch("norm::residual_add_bf16")
}

/// A FOURTH routine, holding both halves of `Const` at once.
///
/// `w` is a weight — a tensor carrier, so it claims the weight run and
/// inherits the named-bank chain — and `eps` is a scalar the statement carries
/// in its params run. One mark, two runs, decided by the carrier.
#[kernels_macros::routine]
fn rmsnorm(
    ctx: &Ctx,
    y: Out<Tensor<Bf16>>,
    w: Const<Tensor<Bf16>>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    if y.ptr.is_null() || w.v.is_null() {
        return Err(Refusal::Null { what: "y or w" });
    }
    if *eps <= 0.0 {
        return Err(Refusal::Empty { what: "eps" });
    }
    ctx.launch("norm::rmsnorm_bf16")
}

/// The table, in a `static` — which is the load-bearing claim: the rows are
/// `const`-promoted from generic associated consts, so nothing is built at run
/// time and nothing can be built inconsistently.
static TABLE: &[Routine<Test>] = &[
    routine!(
        rope_apply,
        derived = <rope_apply as ::kernels::Derivation>::DERIVED
    ),
    routine!(
        tanh_bf16,
        whole,
        derived = <tanh_bf16 as ::kernels::Derivation>::DERIVED
    ),
    routine!(
        residual_add,
        derived = <residual_add as ::kernels::Derivation>::DERIVED
    ),
    routine!(
        rmsnorm,
        derived = <rmsnorm as ::kernels::Derivation>::DERIVED
    ),
];

fn find(name: &str) -> &'static Routine<Test> {
    TABLE
        .iter()
        .find(|r| r.name == name)
        .expect("a routine this test declares")
}

/// The row is the parameter list, in the parameter list's order.
///
/// It used to be a list of `(Ty, Provenance)` pairs, and the second half said
/// nothing: with `Env` out of the parameter list every parameter is the
/// statement's, so the column had one value at every row.
#[test]
fn a_row_is_its_fns_signature() {
    assert_eq!(
        find("rope_apply").args,
        &[Ty::Bf16sMut, Ty::I32, Ty::I32, Ty::F32],
        "the row is read off the parameter list, and nothing else is in it"
    );
    assert_eq!(
        find("tanh_bf16").args.len(),
        2,
        "and a different arity is a different row"
    );
}

/// EVERY MARK CLAIMS A SLOT, AND THE ORDER HANDS OUT THE INDEX.
///
/// The numbers used to be written by hand — `In<0, T>`, `Out<1, T>` — 1,271 of
/// them, and a wrong one COMPILES. Here nothing is written: `resolve` walks
/// the claims in signature order and the running counters ARE the indices.
#[test]
fn the_marks_hand_out_the_slots_in_signature_order() {
    assert_eq!(
        find("rope_apply").sources,
        &[
            // ONE ADDRESS IN TWO SLOTS. `InOut` moves both counters, and the
            // pair is what `in_place` used to state on the row.
            Some(Source::Alias(0, 0)),
            // THE PARAMS RUN, NUMBERED THE SAME WAY. `Const<f32>` takes the
            // float READING of the same channel, which is why the two kinds
            // share one counter.
            Some(Source::Slot(Kind::Param, 0)),
            Some(Source::Slot(Kind::Param, 1)),
            Some(Source::Slot(Kind::ParamF32, 2)),
        ],
    );
    assert_eq!(
        find("residual_add").sources,
        &[Some(Source::Alias(0, 0)), Some(Source::Slot(Kind::In, 1))],
        "the second operand's index is the counter's, not a number anyone wrote"
    );
    assert_eq!(
        find("residual_add").in_place(),
        &[(0, 0)],
        "and the aliasing derives from the same mark"
    );
}

/// A `Const` OVER A TENSOR IS A WEIGHT, AND KEEPS THE NAMED CHAIN.
///
/// An `OpKind::Launch` puts a weight in the operand list where it is
/// positional, while a semantic op carries only a NAME — so the weight slot is
/// a chain, *"the named bank first and the positional one after"*, and `Const`
/// inherits it unchanged from the mark it replaced.
#[test]
fn a_const_over_a_tensor_claims_the_weight_run() {
    let row = find("rmsnorm");
    assert_eq!(row.args, &[Ty::Bf16sMut, Ty::Bf16s, Ty::F32]);
    assert!(
        matches!(
            row.sources[1],
            Some(Source::Or(
                Source::Named("weight"),
                Source::Slot(Kind::Weight, 0)
            ))
        ),
        "the named bank first and the positional one after: {:?}",
        row.sources[1]
    );
    assert_eq!(
        row.sources[2],
        Some(Source::Slot(Kind::ParamF32, 0)),
        "and the scalar half of the same mark claims the params run instead"
    );
}

/// The stated facts are the ones stated.
#[test]
fn the_stated_facts_are_the_ones_stated() {
    assert!(!find("rope_apply").whole, "unstated is false, not absent");
    let tanh = find("tanh_bf16");
    assert!(tanh.whole);
    assert!(!tanh.depth_prefix_plan);
}

/// The facts the BODY asks for are enumerated, syntactically.
///
/// The derived column lost its `Env` half, and with it the check that walks a
/// row asking *"does this backend answer every fact its own kernels name"*.
/// `#[routine]` scans the body for `ask::<_, keys::X>` and emits the list
/// instead — same fidelity as the parameter run for a fact asked in the body,
/// and it cannot drift from the calls.
#[test]
fn the_facts_a_body_asks_for_are_enumerated() {
    assert_eq!(
        <rope_apply as kernels::Derivation>::ASKED,
        &[<keys::Rows as keys::Fact>::KEY],
    );
    assert_eq!(
        <residual_add as kernels::Derivation>::ASKED,
        &[] as &[&str],
        "a body that asks for nothing enumerates nothing"
    );
}

#[test]
fn the_erased_body_is_the_typed_one() {
    let ctx = Ctx::default();
    let args = [
        Value::Region {
            ptr: 0x1000,
            rows: 4,
            width: 64,
        },
        Value::I32(64),
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
    let null = [
        Value::Region {
            ptr: 0,
            rows: 4,
            width: 64,
        },
        Value::I32(64),
        Value::I32(64),
        Value::F32(1e4),
    ];
    assert_eq!(
        (find("rope_apply").body)(&ctx, &null),
        Err(Refusal::Null { what: "q" }),
        "the body's own word for it, not a generic failure"
    );
    assert!(ctx.fired.borrow().is_empty(), "and nothing launched");
}

/// A FACT THIS BACKEND DOES NOT ANSWER IS A REFUSAL, NOT A ZERO.
///
/// The cost `ask` carries, stated: it is a call and not a declaration, so
/// nothing checks at compile time that a backend answers it. What DOES hold is
/// that an unanswered fact refuses the fire rather than binding a default.
#[test]
fn an_unanswered_fact_refuses_the_fire() {
    struct Deaf;
    impl Answers<Test> for Deaf {
        fn resolve(&self, _ty: Ty, _source: Source) -> Result<Value, Refusal> {
            Err(Refusal::Unstated {
                what: "nothing at all",
            })
        }
    }
    // The `Ctx` a body launches through is this backend's own, so the deaf
    // resolver is exercised directly: what matters is the shape of the answer.
    assert_eq!(
        Asks::<Test>::ask::<i32, keys::Rows>(&Deaf),
        Err(Refusal::Unstated {
            what: "nothing at all"
        })
    );
}

#[test]
fn a_list_that_does_not_fit_the_signature_is_refused() {
    let ctx = Ctx::default();
    assert_eq!(
        (find("rope_apply").body)(&ctx, &[Value::I32(1)]),
        Err(Refusal::Arity { want: 4, got: 1 })
    );
    let swapped = [
        Value::Region {
            ptr: 0x1000,
            rows: 4,
            width: 64,
        },
        Value::F32(4.0),
        Value::I32(64),
        Value::F32(1e4),
    ];
    assert_eq!(
        (find("rope_apply").body)(&ctx, &swapped),
        Err(Refusal::Kind {
            at: 1,
            want: Ty::I32
        }),
        "same width, different kind -- the position is named"
    );
}

/// A REGION IS ONE ARGUMENT, and the body reads its shape off the value.
#[test]
fn a_region_is_one_argument() {
    let ctx = Ctx::default();
    let args = [
        Value::Region {
            ptr: 0x1000,
            rows: 4,
            width: 64,
        },
        Value::Region {
            ptr: 0x2000,
            rows: 4,
            width: 64,
        },
    ];
    (find("residual_add").body)(&ctx, &args).expect("matched widths launch");
    assert_eq!(*ctx.fired.borrow(), ["norm::residual_add_bf16"]);

    let mismatched = [
        Value::Region {
            ptr: 0x1000,
            rows: 4,
            width: 64,
        },
        Value::Region {
            ptr: 0x2000,
            rows: 4,
            width: 32,
        },
    ];
    assert_eq!(
        (find("residual_add").body)(&ctx, &mismatched),
        Err(Refusal::Narrow {
            what: "width",
            at: 32
        }),
        "the rectangle rides with the address, so the body can compare it"
    );
}
