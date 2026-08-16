//! The routine machinery, exercised through a stand-in backend.
//!
//! What this proves is the one thing the rest of the refactor rests on: that a
//! table row DERIVES from a `fn`'s signature, in a `const` context, through a
//! macro that sees only the identifier. If this compiles, no routine's row can
//! disagree with its body, because there is one statement of it.

use kernels::Ty;
use kernels::keys;
use kernels::routine::{Arg, Backend, Env, Extent, In, Out, Provenance, Refusal, Routine};

/// The stand-in backend: an argument is one of three kinds, and the context
/// records what a body launched instead of launching it.
#[derive(Clone, Copy)]
struct Test;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    Ptr(usize),
    I32(i32),
    F32(f32),
    /// An address the statement placed, with the shape it placed it at --
    /// what a fat `In<N, _>` or `Out<N, _>` unpacks from.
    Region { ptr: usize, rows: i32, width: i32 },
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

    fn region(value: &Value, _at: usize) -> Result<Extent, Refusal> {
        match *value {
            Value::Region { rows, width, .. } => Ok(Extent { rows, width }),
            _ => Err(Refusal::Absent { what: "a region's shape" }),
        }
    }
}

/// The backend's own `routine!`, with its [`Backend`] filled in — the
/// three-line adapter every backend writes.
macro_rules! routine {
    ($body:ident $(, $($rest:tt)*)?) => {
        routine!(@go $body $(, $($rest)*)?)
    };
    (@go $($all:tt)*) => { ::kernels::routine!(Test, $($all)*) };
}

struct Bf16;
impl kernels::Elem for Bf16 {
    const CPP_CONST: &'static str = "const __nv_bfloat16*";
    const CPP_MUT: &'static str = "__nv_bfloat16*";
    const TY_CONST: Ty = Ty::Bf16s;
    const TY_MUT: Ty = Ty::Bf16sMut;
}

impl Arg<Test> for *const Bf16 {
    const TY: Ty = Ty::Bf16s;

    fn unpack(value: &Value, at: usize) -> Result<Self, Refusal> {
        match value {
            Value::Ptr(p) | Value::Region { ptr: p, .. } => Ok(core::ptr::without_provenance(*p)),
            _ => Err(Refusal::Kind { at, want: <Self as Arg<Test>>::TY }),
        }
    }
}
impl Arg<Test> for *mut Bf16 {
    const TY: Ty = Ty::Bf16sMut;

    fn unpack(value: &Value, at: usize) -> Result<Self, Refusal> {
        match value {
            Value::Ptr(p) | Value::Region { ptr: p, .. } => {
                Ok(core::ptr::without_provenance_mut(*p))
            }
            _ => Err(Refusal::Kind { at, want: <Self as Arg<Test>>::TY }),
        }
    }
}

struct Bf16sMut(usize);
impl Arg<Test> for Bf16sMut {
    const TY: Ty = Ty::Bf16sMut;

    fn unpack(value: &Value, at: usize) -> Result<Self, Refusal> {
        match value {
            Value::Ptr(p) | Value::Region { ptr: p, .. } => Ok(Self(*p)),
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
            Value::Ptr(p) | Value::Region { ptr: p, .. } => Ok(Self(*p)),
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

/// A THIRD routine, whose destination arrives whole.
///
/// `.wiki/kilimanjaro2.md` §1.3's shape: the two above take an address and are
/// then TOLD their extents, in separate parameters that a binder resolves from
/// separate places. This one takes a region -- `Out<0, _>` -- and reads the
/// rectangle off the argument the statement placed. There is no `rows`
/// parameter and no `width` parameter, because there is nothing left for them
/// to say.
///
/// The `In<1, _>` is `residual_add`'s case in miniature (§3.3): the row
/// declares `in_place = &[(0, 0)]`, so its input is the statement's SECOND
/// operand and not its first. Counting derives `In(0)` and is wrong; the
/// index is written down, so nothing has to derive it.
#[kernels_macros::routine]
fn residual_add(ctx: &Ctx, y: Out<0, Bf16>, x: In<1, Bf16>) -> Result<(), Refusal> {
    if y.ptr.is_null() || x.ptr.is_null() {
        return Err(Refusal::Null { what: "a region" });
    }
    if y.rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if y.width != x.width {
        return Err(Refusal::Narrow { what: "width", at: x.width.into() });
    }
    ctx.launch("norm::residual_add_bf16")
}

/// A FOURTH routine, whose scalar names a fact instead of hoping its
/// parameter is spelled right.
///
/// `.wiki/kilimanjaro2.md` rule E3. The parameter is called `eps`, which is
/// what a reader of the signature wants to see, and the fact is
/// `keys::RmsEps`, which is what the binder wants to see. Today those two
/// have to be the same word: `kernels-macros`'s `fact_of` maps the NAME onto
/// a `Source`, so `eps` derives `RmsEps` only because someone remembered to
/// put `"eps"` in the alias list.
///
/// The interesting half is `theta`. `fact_of` maps `"theta" | "rope_theta" |
/// "rope_base"` onto ONE variant, and there are two thetas — the layer's and
/// the fire's — which differ on gemma-4. Here the parameter is called
/// `theta`, the alias table would say `Source::Named(<keys::Theta as keys::Fact>::KEY)`, and the signature says
/// `keys::RopeTheta`, which is the other one. **The name loses.** Compare
/// `rope_apply` above, whose `theta: f32` derives nothing at all.
#[kernels_macros::routine]
fn rmsnorm_named(
    ctx: &Ctx,
    y: Out<0, Bf16>,
    eps: Env<keys::RmsEps>,
    theta: Env<keys::RopeTheta>,
) -> Result<(), Refusal> {
    if y.ptr.is_null() {
        return Err(Refusal::Null { what: "y" });
    }
    // TWO `Deref` HOPS -- `Env` to the fact, the fact to its value -- so an
    // explicit dereference needs a second star. A METHOD CALL DOES NOT:
    // Rust's method probe walks the whole deref chain, so `eps.arg()`, which
    // is how every launcher in `kernels-cuda` actually spends its scalars,
    // is character-for-character unchanged. That asymmetry is the reason 171
    // sites can convert one at a time instead of in one commit, and the
    // reason a fact is a newtype rather than a unit struct.
    if **eps <= 0.0 || **theta <= 0.0 {
        return Err(Refusal::Empty { what: "eps" });
    }
    ctx.launch("norm::rmsnorm_named")
}

/// The table, in a `static` — which is the load-bearing claim: the rows are
/// `const`-promoted from generic associated consts, so nothing is built at
/// run time and nothing can be built inconsistently.
static ROUTINES: &[Routine<Test>] = &[
    routine!(rope_apply, in_place = &[(0, 0)]),
    routine!(tanh_bf16, whole, in_place = &[(0, 0)]),
    routine!(residual_add, in_place = &[(0, 0)], derived = <residual_add as ::kernels::Derivation>::DERIVED),
    routine!(rmsnorm_named, in_place = &[(0, 0)], derived = <rmsnorm_named as ::kernels::Derivation>::DERIVED),
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

/// A REGION IS ONE ARGUMENT, AND THE COLUMN SAYS WHICH SLOT.
///
/// Three claims, and each one is a thing the old spelling could not make.
///
/// The row has TWO entries where the equivalent thin signature had four --
/// the shape is not a pair of extra parameters any more, it rides with the
/// address. The `In` is `In(1)`, which no counting derives: this launcher
/// names one `*const`-shaped parameter, so inference reaches `In(0)` and
/// `alias()` exists in the CUDA binder to walk it past the in-place pair
/// afterwards. Written down, there is nothing to walk.
///
/// And the body reads `y.rows` off the value the statement placed, which is
/// what makes the other two more than notation.
#[test]
fn a_region_is_one_argument_and_the_column_says_which_slot() {
    let row = find("residual_add");
    assert_eq!(
        row.args,
        &[(Ty::Bf16sMut, Provenance::Trace), (Ty::Bf16s, Provenance::Trace)],
        "a region declares the pointee's `Ty`, and the WRAPPER picks const or mut"
    );
    assert_eq!(
        row.derived.iter().map(|d| d.source).collect::<Vec<_>>(),
        [
            Some(kernels::Source::Slot(kernels::Kind::Out, 0)),
            Some(kernels::Source::Slot(kernels::Kind::In, 1)),
        ],
        "the indices are read off the types, not counted off the pointers"
    );

    let ctx = Ctx::default();
    let args = [
        Value::Region { ptr: 0x1000, rows: 7, width: 4096 },
        Value::Region { ptr: 0x2000, rows: 7, width: 4096 },
    ];
    (row.body)(&ctx, &args).expect("a live rectangle launches");
    assert_eq!(*ctx.fired.borrow(), ["norm::residual_add_bf16"]);
}

/// A REGION ASKED FOR AND NOT SUPPLIED REFUSES, RATHER THAN LAUNCHING BLIND.
///
/// The other direction -- a thin `*const T` parameter handed a `Region` --
/// takes the address and drops the shape, which is what lets a family migrate
/// one launcher at a time. This direction must not be symmetric: a signature
/// that says `Out<0, _>` is claiming a width, and a binder that cannot supply
/// one has to say so rather than let the body divide by whatever `rows`
/// happened to default to.
#[test]
fn a_region_parameter_refuses_a_bare_address() {
    let ctx = Ctx::default();
    let args = [Value::Ptr(0x1000), Value::Ptr(0x2000)];
    let refusal = (find("residual_add").body)(&ctx, &args).expect_err("no shape, no launch");
    assert!(
        matches!(refusal, Refusal::Absent { .. }),
        "the address unpacked and the shape did not: {refusal:?}"
    );
    assert!(ctx.fired.borrow().is_empty(), "and nothing was launched");
}

/// A FACT IS A TYPE, AND THE TYPE WINS OVER THE NAME.
///
/// `.wiki/kilimanjaro2.md` rule E3, and the one case that proves it is worth
/// the churn: `theta`.
///
/// `kernels-macros`'s `fact_of` maps `"theta" | "rope_theta" | "rope_base"`
/// onto `Source::Named(<keys::Theta as keys::Fact>::KEY)`, and `bind/table.rs:1361` explains why that is a bug
/// waiting for a caller -- *"NOT `Theta`. `Cx::theta` resolves the layer and
/// falls back; this is the fire's field, and gemma-4 makes them differ."*
/// The parameter below IS called `theta`, so the alias table has its answer
/// ready. The signature says otherwise and the signature wins.
///
/// `kernels-cuda` has eleven parameters named `theta` and all eleven carry an
/// explicit `#[source(...)]` to get out of the alias's way. This is what
/// retires those eleven attributes.
#[test]
fn a_fact_is_a_type_and_the_type_beats_the_name() {
    let row = find("rmsnorm_named");
    assert_eq!(
        row.derived.iter().map(|d| d.source).collect::<Vec<_>>(),
        [
            Some(kernels::Source::Slot(kernels::Kind::Out, 0)),
            Some(kernels::Source::Named(<kernels::keys::RmsEps as kernels::keys::Fact>::KEY)),
            Some(kernels::Source::Named(<kernels::keys::RopeTheta as kernels::keys::Fact>::KEY)),
        ],
        "the third is `RopeTheta` though the parameter is spelled `theta`"
    );
    assert_eq!(
        row.derived.iter().map(|d| d.name).collect::<Vec<_>>(),
        ["y", "eps", "theta"],
        "and the names are untouched -- a signature still reads as a sentence"
    );

    // STATED, so nothing downstream may "correct" it. A CUDA-side `alias()`
    // walking indices past an in-place pair must leave a fact alone, and the
    // flag is how it knows.
    assert!(
        row.derived[1].stated && row.derived[2].stated,
        "a fact named by type is stated, not counted"
    );

    // The provenance still says the environment supplies it, which is what
    // the arity check reads. `Env<keys::_>` is `Env` first and a fact second.
    assert_eq!(
        row.args,
        &[
            (Ty::Bf16sMut, Provenance::Trace),
            (Ty::F32, Provenance::Env),
            (Ty::F32, Provenance::Env),
        ],
    );
}

/// AND THE FACT REACHES THE BODY AS ITS VALUE.
///
/// Two `Deref` hops. If this needed a `.get()` or a `.0.0`, converting 171
/// sites would mean touching 171 bodies, and the migration would have to
/// happen in one commit instead of one launcher at a time.
#[test]
fn a_fact_derefs_to_the_number() {
    let ctx = Ctx::default();
    let args = [
        Value::Region { ptr: 0x1000, rows: 4, width: 128 },
        Value::F32(1e-5),
        Value::F32(1e6),
    ];
    (find("rmsnorm_named").body)(&ctx, &args).expect("both facts arrived");
    assert_eq!(*ctx.fired.borrow(), ["norm::rmsnorm_named"]);

    let zero = [
        Value::Region { ptr: 0x1000, rows: 4, width: 128 },
        Value::F32(0.0),
        Value::F32(1e6),
    ];
    assert_eq!(
        (find("rmsnorm_named").body)(&ctx, &zero),
        Err(Refusal::Empty { what: "eps" }),
        "the body compared the unwrapped f32, not a wrapper"
    );
}
