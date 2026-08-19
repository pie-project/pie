//! One routine, written the way every ported family will be, checked for the
//! properties the rest of the port rests on.
//!
//! It is `rope::rope_standard_table` — a real kernel, its real geometry, its
//! real instantiation — declared as an ordinary `fn` and registered with
//! `routine!`. Nothing here launches: a table is readable without a GPU, and
//! that is half the point of the split.

use kernels::Ty;
use kernels::{Asks, Bind, Const, Fire, InOut, Out, Refusal};
use kernels::keys;
use kernels_cuda::jit::{ArgValue, Ctx, Launch, Root, Routine};
use kernels_cuda::jit::abi::{Tensor, bf16};

/// `rope.cu:82` — `constexpr int BLOCK = 256;`
const ROTATE_BLOCK: u32 = 256;

/// The cos/sin table `attn`'s fused prepare reads.
///
/// A SAFE `fn`, like every routine body: `KernelFn` is implemented over `Fn`,
/// which an `unsafe fn` does not implement. Nothing is lost -- a body cannot
/// check a device pointer either way, so the obligation is stated once at
/// `call()`, where the pointers come from, instead of restated by all 200 of
/// these.
///
/// `positions` used to be a parameter, `Env<*const i32>`; asking is what an
/// `Env` with no key beside it became, because there was no key beside it --
/// `keys::Positions` was always the fact, just not yet spelled. `table`
/// carries its own row count now, so `num_tokens` is gone with it: a region
/// mark's `.rows` is the count a bare pointer needed a second parameter to
/// say. This is `rope::rope_standard_table`'s own real signature, copied
/// rather than paralleled, so this file stays what it claims to be.
///
/// # Safety
///
/// `positions` addresses `table.rows` live `i32`s and `table` itself
/// `table.rows * head_dim` live floats; `stream` must be live across the
/// launch.
fn rope_standard_table(
    ctx: &Ctx<'_>,
    table: Out<Tensor<f32>>,
    head_dim: Const<i32>,
    theta: Const<f32>,
) -> Result<(), Refusal> {
    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    if *head_dim / 2 <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    ctx.fire(
        Fire::at("rope/rope.cuh", "::pie::rope::standard_table<::pie::i32>")
            .apply(Launch::per_row(table.rows.unsigned_abs(), ROTATE_BLOCK)),
        &[positions.arg(), table.arg(), head_dim.arg(), theta.arg()],
    )
}

/// A second routine, at a different arity and with a stated fact.
///
/// `in_place = &[(0, 0), (1, 1)]` was the row's own statement of this; it is
/// `q` and `k` wearing `InOut` now, one address in an operand slot and a
/// result slot each, and [`kernels::routine::aliased`] reads the same pairs
/// back off `SOURCES` rather than off a second, hand-kept list.
fn rope_bf16_stub(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    num_tokens: Const<i32>,
) -> Result<(), Refusal> {
    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    if *num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    ctx.fire(
        Fire::at("rope/rope.cuh", "::pie::rope::rotate<false, false>")
            .apply(Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK)),
        &[q.arg(), k.arg(), positions.arg(), num_tokens.arg()],
    )
}

/// Both routines' rows, built by hand rather than read off a distributed
/// slice.
///
/// `#[routine]` auto-registers into `crate::ROUTINES` — the crate that
/// compiled it, which for a kernel in `kernels-cuda/src` is the plane's own
/// `CUDA_ROUTINES`. An integration test is a SEPARATE crate with no such
/// slice of its own to register into, and no reason to invent one: two
/// routines a reader can see whole are the point of this file, and
/// `kernels::routine!` still builds a row from a plain `fn` without it.
static ROUTINES: &[Routine] = &[
    kernels::routine!(kernels_cuda::jit::Cuda, rope_standard_table),
    kernels::routine!(kernels_cuda::jit::Cuda, rope_bf16_stub),
];

fn find(name: &str) -> &'static Routine {
    ROUTINES.iter().find(|r| r.name == name).expect("a routine this test declares")
}

/// The row is the signature: the arguments a launch binds positionally, in
/// order.
///
/// `positions` is not among them for either routine — `keys::Positions` is a
/// fact only the fire answers, asked inside the body — so `Provenance`,
/// which used to mark an argument `Env` beside its `Ty`, has nothing left to
/// distinguish: every surviving argument is the statement's own, whether it
/// reads, writes, or is a `Const` the checkpoint fixed. `args` is a plain
/// `Ty` list now for exactly that reason.
#[test]
fn the_row_derives_from_the_signature() {
    assert_eq!(find("rope_standard_table").args, &[Ty::F32sMut, Ty::I32, Ty::F32]);
    assert_eq!(find("rope_bf16_stub").args.len(), 3);
}

/// The C++ spelling comes across too, per position, from the same impl the
/// marshalling tag does.
///
/// `rope_standard_table`'s row is three long, not five: `positions` carried
/// no spelling of its own to check, and does not need one, since asking it
/// is a call and not an entry in this list.
#[test]
fn the_spelling_comes_with_it() {
    let spelling = find("rope_standard_table").spelling;
    assert_eq!(spelling[0], "float*");
    assert_eq!(spelling[1], "int");
    assert_eq!(spelling[2], "float");
    assert_eq!(
        find("rope_bf16_stub").spelling[0],
        "::pie::bf16*",
        "and it is the DECLARED spelling, not `Ty::cpp`'s `void*`"
    );
}

/// A stated fact is stated; an unstated one is false rather than absent.
#[test]
fn the_stated_facts_are_the_ones_stated() {
    assert_eq!(find("rope_bf16_stub").in_place(), &[(0, 0), (1, 1)]);
    assert!(!find("rope_bf16_stub").whole);
    assert!(find("rope_standard_table").in_place().is_empty());
}

/// Answers every fact with a null pointer.
///
/// Enough to clear `ctx.ask::<_, keys::Positions>()`'s `Refusal::Unstated`
/// without ever dereferencing what it hands back -- nothing below reaches a
/// device, so nothing needs the address to be real, only present.
struct AnyFact;

impl kernels::Answers<kernels_cuda::jit::Cuda> for AnyFact {
    fn resolve(&self, _ty: Ty, _source: kernels::Source) -> Result<ArgValue, Refusal> {
        Ok(ArgValue::Ptr(std::ptr::null_mut()))
    }
}

/// A body's own refusal survives the erasure, and reaches no device on the
/// way — this test runs on a machine with no GPU.
///
/// `head_dim / 2 <= 0` is the one refusal left in this body: `num_tokens`'s
/// used to be a second, but the row count merged into `table.rows` once
/// `table` became a region mark, and a zero grid is [`Ctx::fire`]'s own
/// refusal now, common to every routine rather than typed by hand in each.
/// Reaching this check at all needs `positions` answered first, which is
/// what [`AnyFact`] is for.
#[test]
fn a_refusal_needs_no_device() {
    let env = AnyFact;
    // SAFETY: the stream is never used -- the refusal below fires first.
    let ctx = unsafe { Ctx::on(std::ptr::null_mut()) }.with_env(&env);
    let args = [
        ArgValue::Region { ptr: std::ptr::null_mut(), rows: 4, width: 8 },
        ArgValue::I32(0),
        ArgValue::F32(1e4),
    ];
    assert_eq!(
        (find("rope_standard_table").body)(&ctx, &args),
        Err(Refusal::Empty { what: "head_dim / 2" })
    );
}

/// The erased path checks the list against the signature, because the values
/// arrive dynamically and the signature is the only statement of the shape.
///
/// Arity and kind are both checked before the body runs, so neither case
/// here needs an environment that can answer `positions`.
#[test]
fn a_list_that_does_not_fit_is_refused() {
    // SAFETY: as above -- every case here refuses before a device is named.
    let ctx = unsafe { Ctx::on(std::ptr::null_mut()) };
    assert_eq!(
        (find("rope_standard_table").body)(&ctx, &[ArgValue::I32(1)]),
        Err(Refusal::Arity { want: 3, got: 1 })
    );
    let swapped = [
        ArgValue::Region { ptr: std::ptr::null_mut(), rows: 4, width: 8 },
        ArgValue::F32(1.0),
        ArgValue::F32(1e4),
    ];
    assert_eq!(
        (find("rope_standard_table").body)(&ctx, &swapped),
        Err(Refusal::Kind { at: 1, want: Ty::I32 })
    );
}

/// The cache key spans the instantiation, so two symbols out of one root are
/// two entries rather than one.
#[test]
fn two_instantiations_of_a_root_are_two_keys() {
    let rope = Root::new("rope/rope.cuh");
    let a = rope.key("pie::rope::rotate<false>", "sm_90");
    let b = rope.key("pie::rope::rotate<true>", "sm_90");
    assert_ne!(a, b);
    assert_eq!(a, rope.key("pie::rope::rotate<false>", "sm_90"));
}
