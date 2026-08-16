//! One routine, written the way every ported family will be, checked for the
//! properties the rest of the port rests on.
//!
//! It is `rope::rope_standard_table` — a real kernel, its real geometry, its
//! real instantiation — declared as an ordinary `fn` and registered with
//! `routine!`. Nothing here launches: a table is readable without a GPU, and
//! that is half the point of the split.

use kernels::Ty;
use kernels::routine::{Env, Provenance, Refusal};
use kernels_cuda::jit::{ArgValue, Ctx, Launch, Root, Routine};
use kernels_cuda::routine;
use kernels_cuda::jit::abi::bf16;


/// `rope.cu:82` — `constexpr int BLOCK = 256;`
const ROTATE_BLOCK: u32 = 256;

/// The cos/sin table `attn`'s fused prepare reads.
///
/// A SAFE `fn`, like every routine body: `KernelFn` is implemented over `Fn`,
/// which an `unsafe fn` does not implement. Nothing is lost -- a body cannot
/// check a device pointer either way, so the obligation is stated once at
/// `call()`, where the pointers come from, instead of restated by all 200 of
/// these.
#[kernels_macros::routine]
fn rope_standard_table(
    ctx: &Ctx,
    positions: Env<*const i32>,
    table: *mut f32,
    num_tokens: i32,
    head_dim: i32,
    theta: f32,
) -> Result<(), Refusal> {
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if head_dim / 2 <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound into this list
    // addresses live device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::standard_table<::pie::i32>",
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK),
            &[
                ArgValue::Ptr((*positions).cast_mut().cast()),
                ArgValue::Ptr(table.cast()),
                ArgValue::I32(head_dim),
                ArgValue::F32(theta),
            ],
        )
    }
}

/// A second routine, at a different arity and with a stated fact.
#[kernels_macros::routine]
fn rope_bf16_stub(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    positions: Env<*const i32>,
    num_tokens: i32,
) -> Result<(), Refusal> {
    let _ = (q, k, positions);
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    // SAFETY: as above.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::rotate<false, false>",
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK),
            &[],
        )
    }
}

static ROUTINES: &[Routine] =
    &[routine!(rope_standard_table), routine!(rope_bf16_stub, in_place = &[(0, 0), (1, 1)])];

fn find(name: &str) -> &'static Routine {
    ROUTINES.iter().find(|r| r.name == name).expect("a routine this test declares")
}

/// The row is the signature — including which arguments the environment
/// supplies, which is stated on the argument and nowhere else.
#[test]
fn the_row_derives_from_the_signature() {
    assert_eq!(
        find("rope_standard_table").args,
        &[
            (Ty::I32s, Provenance::Env),
            (Ty::F32sMut, Provenance::Trace),
            (Ty::I32, Provenance::Trace),
            (Ty::I32, Provenance::Trace),
            (Ty::F32, Provenance::Trace),
        ]
    );
    assert_eq!(find("rope_bf16_stub").args.len(), 4);
}

/// The C++ spelling comes across too, per position, from the same impl the
/// marshalling tag does.
#[test]
fn the_spelling_comes_with_it() {
    let spelling = find("rope_standard_table").spelling;
    assert_eq!(spelling[0], "const ::std::int32_t*");
    assert_eq!(spelling[1], "float*");
    assert_eq!(spelling[2], "int");
    assert_eq!(spelling[4], "float");
    assert_eq!(
        find("rope_bf16_stub").spelling[0],
        "::pie::bf16*",
        "and it is the DECLARED spelling, not `Ty::cpp`'s `void*`"
    );
}

/// A stated fact is stated; an unstated one is false rather than absent.
#[test]
fn the_stated_facts_are_the_ones_stated() {
    assert_eq!(find("rope_bf16_stub").in_place, &[(0, 0), (1, 1)]);
    assert!(!find("rope_bf16_stub").whole);
    assert!(find("rope_standard_table").in_place.is_empty());
}

/// A body's own refusal survives the erasure, and reaches no device on the
/// way — this test runs on a machine with no GPU.
#[test]
fn a_refusal_needs_no_device() {
    // SAFETY: the stream is never used -- every case here refuses first.
    let ctx = unsafe { Ctx::on(std::ptr::null_mut()) };
    let args = [
        ArgValue::Ptr(std::ptr::null_mut()),
        ArgValue::Ptr(std::ptr::null_mut()),
        ArgValue::I32(0),
        ArgValue::I32(128),
        ArgValue::F32(1e4),
    ];
    assert_eq!(
        (find("rope_standard_table").body)(&ctx, &args),
        Err(Refusal::Empty { what: "num_tokens" })
    );
}

/// The erased path checks the list against the signature, because the values
/// arrive dynamically and the signature is the only statement of the shape.
#[test]
fn a_list_that_does_not_fit_is_refused() {
    // SAFETY: as above.
    let ctx = unsafe { Ctx::on(std::ptr::null_mut()) };
    assert_eq!(
        (find("rope_standard_table").body)(&ctx, &[ArgValue::I32(1)]),
        Err(Refusal::Arity { want: 5, got: 1 })
    );
    let swapped = [
        ArgValue::Ptr(std::ptr::null_mut()),
        ArgValue::Ptr(std::ptr::null_mut()),
        ArgValue::F32(1.0),
        ArgValue::I32(128),
        ArgValue::F32(1e4),
    ];
    assert_eq!(
        (find("rope_standard_table").body)(&ctx, &swapped),
        Err(Refusal::Kind { at: 2, want: Ty::I32 })
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
