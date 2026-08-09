//! A row that names two kernels, and the fire that picks one.
//!
//! # What is on trial here
//!
//! `crates/kernels-cuda/csrc/src/norm/rmsnorm.cu` holds a host predicate,
//! `rmsnorm_vec8_ok`, and an `if` that consults it: three pointer alignments
//! and three strides decide between a scalar RMSNorm and its vectorised twin.
//! An ahead-of-time build had to write that `if`, because nvcc chose its
//! instantiations months before any pointer existed. A JIT does not: the fire
//! holds the addresses, so the row can name both kernels and the dispatcher
//! can choose — see [`kernels_cuda_new::device::Specialisation`].
//!
//! That buys speed and it costs the one property that made the table worth
//! having. An AOT `if` is auditable in one place; a `Select` that DISAGREES
//! with `rmsnorm_vec8_ok` produces a fast wrong answer, which is precisely
//! the failure this design says must be a refusal instead. So the gate is
//! four things, and three of them are about the disagreement:
//!
//! 1. **Both arms fire and agree** — the vectorised kernel's output against
//!    the scalar kernel's, on the same data, at the same width, measured to
//!    the bf16 bit.
//! 2. **The predicate agrees with the C++** — swept across every boundary the
//!    six clauses have, with the C++ text pinned so that an edit to it is a
//!    failing test rather than a silent divergence.
//! 3. **The negative controls** — a predicate doctored to disagree, an arm
//!    doctored to be ill-typed, and the wrong kernel fired on purpose. A gate
//!    that only ever sees the right answer has not been tested.
//! 4. **The cost** — of the choice, per fire, and of the variant, per
//!    compile. A specialisation that pays for itself is the claim; a number
//!    is what makes it one.
//!
//! # Why half of this runs with no CUDA at all
//!
//! The predicate is a function of values the host already has, which is the
//! whole reason it can be evaluated at fire time without a synchronisation.
//! The consequence is that it can also be evaluated with no GPU, and the
//! sweep below does — so the agreement between this table and `rmsnorm.cu` is
//! checked on every machine that builds the crate, not only on the one that
//! has a device. The fires are behind `_cuda` and skip with a stated reason.

use kernels_cuda_new::device::{self, Arm, Fact, Specialisation, Take, Term};

/// `norm::rmsnorm_strided_bf16` — the base row, and the only symbol anything
/// outside this file names.
const BASE: &str = "norm::rmsnorm_strided_bf16";

/// The variant row's symbol.
///
/// Spelled here because the negative control fires it deliberately. Nothing
/// else in the tree names it: `model-compiler` writes the base symbol, the
/// dispatcher matches the base symbol, and the specialisation reaches the
/// variant through a `&'static DeviceKernel` rather than through a string.
const VARIANT: &str = "norm::rmsnorm_strided_bf16#vec8";

// ---------------------------------------------------------------------------
// 2. The predicate agrees with the C++
// ---------------------------------------------------------------------------

/// `rmsnorm_vec8_ok`, transliterated.
///
/// The oracle the sweep compares against, and it is a HAND copy on purpose.
/// The alternative — deriving the oracle from the same table the sweep is
/// checking — would compare a thing to itself and pass forever. This function
/// is written from the C++ text, in the C++'s order, with the C++'s
/// operators; [`the_cpp_predicate_is_the_one_this_file_was_written_from`]
/// pins that text so the copy cannot go stale unnoticed.
fn rmsnorm_vec8_ok(x: u64, y: u64, weight: u64, hidden: i64, x_row_stride: i64, y_row_stride: i64) -> bool {
    let aligned = |p: u64| (p & 15) == 0;
    hidden % 8 == 0
        && x_row_stride % 8 == 0
        && y_row_stride % 8 == 0
        && aligned(x)
        && aligned(y)
        && aligned(weight)
}

/// The base row's operand order, as facts.
///
/// `x, weight, y, hidden, x_row_stride, y_row_stride, eps` — the order
/// `rmsnorm.cu`'s `<<<>>>` passes them in, which is the order the row states
/// them in, which is the order the terms index. `eps` is [`Fact::Opaque`]
/// because it is an `F32` and no term may read one: a float's bit pattern
/// divides by 8 perfectly happily and means nothing.
fn facts(x: u64, weight: u64, y: u64, hidden: i64, xs: i64, ys: i64) -> [Fact; 7] {
    [
        Fact::Address(x),
        Fact::Address(weight),
        Fact::Address(y),
        Fact::Int(hidden),
        Fact::Int(xs),
        Fact::Int(ys),
        Fact::Opaque,
    ]
}

/// A device address that is 256-byte aligned, which is what `cuMemAlloc`
/// returns and therefore what a real fire starts from.
const ARENA: u64 = 0x7f00_0000_0000;

/// Byte offsets swept over each of the three pointers.
///
/// The boundary is 16 bytes, so the interesting values are: on it, one BYTE
/// past it (which no bf16 pointer can actually be, and the predicate must
/// still answer), one ELEMENT past it (2 bytes — the offset a real slice of a
/// packed tensor produces), half of it, and one short of it. `24` is there
/// because 8-byte alignment is the trap: a `float2` would be happy and a
/// `float4` is not.
const OFFSETS: [u64; 8] = [0, 1, 2, 4, 8, 14, 16, 24];

/// Widths swept, including every residue of 8.
///
/// `hidden % 8` is one of the six clauses and the only one whose boundary is
/// not a power of two in an address, so it is swept exhaustively across a
/// residue class rather than at chosen points: 4096 is a real model width,
/// 4097..4103 are its seven wrong neighbours, and 2048/2816/5376 are the
/// three `rmsnorm.cu` published its sweep at.
const WIDTHS: [i64; 12] = [2048, 2816, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 5376, 5377];

/// Strides swept — the row's own width, the next multiple of 8, and the two
/// that are not.
const STRIDES: [i64; 4] = [0, 1, 7, 8];

/// The selection and `rmsnorm_vec8_ok` answer the same on every case.
///
/// **This is the test the whole design stands on.** The specialisation is
/// only defensible if it reproduces the decision the C++ makes; a `Select`
/// that agrees on the cases someone thought of and diverges on one they did
/// not is worse than no specialisation at all, because the divergence is a
/// finite, plausible number rather than a crash.
///
/// Swept: 8 offsets on each of three pointers × 12 widths × 4 stride offsets
/// on each of two strides — 8³ × 12 × 4² = 98 304 cases, of which the
/// interesting ones are the 1 536 where every pointer is aligned and the
/// clauses are decided by a stride.
#[test]
fn the_selection_agrees_with_rmsnorm_vec8_ok() {
    let spec = device::specialisation(BASE).expect("the base row is specialised");
    let mut cases = 0_u32;
    let mut chosen = 0_u32;
    let mut disagreed: Vec<String> = Vec::new();

    for xo in OFFSETS {
        for wo in OFFSETS {
            for yo in OFFSETS {
                for hidden in WIDTHS {
                    for xs in STRIDES {
                        for ys in STRIDES {
                            let (x, weight, y) = (ARENA + xo, ARENA + wo, ARENA + yo);
                            // A stride is an ELEMENT count, so the sweep
                            // perturbs it around the row's own width the way
                            // a padded tensor does.
                            let (xstride, ystride) = (hidden + xs, hidden + ys);
                            let want = rmsnorm_vec8_ok(x, y, weight, hidden, xstride, ystride);
                            let facts = facts(x, weight, y, hidden, xstride, ystride);
                            let got = spec.choose(&facts).expect("no term faults on a real row");
                            cases += 1;
                            if got.is_some() {
                                chosen += 1;
                            }
                            if got.is_some() != want {
                                disagreed.push(format!(
                                    "x+{xo} w+{wo} y+{yo} hidden {hidden} strides \
                                     {xstride}/{ystride}: C++ says {want}, the row says {}",
                                    got.is_some()
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "predicate agreement: {cases} cases, {chosen} took the vectorised arm, \
         {} disagreed",
        disagreed.len()
    );
    assert!(cases > 0);
    assert!(chosen > 0, "no case took the arm, so the sweep proved nothing");
    assert!(chosen < cases, "every case took the arm, so the sweep proved nothing");
    assert!(disagreed.is_empty(), "{} case(s) disagree:\n{}", disagreed.len(), disagreed.join("\n"));
}

/// The C++ predicate is still the six clauses this file was written from.
///
/// A pinned copy of `rmsnorm_vec8_ok`'s body, whitespace-normalised. The
/// oracle above is a hand transliteration and hand transliterations go stale;
/// this is what makes going stale LOUD. The day someone adds a seventh clause
/// to the C++ — a `num_rows` bound, a second alignment, a `hidden` maximum —
/// this test fails, names the table that has to change, and does so on a
/// machine with no GPU rather than in a model that answers fluently and
/// slightly wrong.
///
/// It reads the `.cu` through [`include_str!`], so the pin is against the
/// source in the tree rather than against a copy that could drift on its own.
///
/// **The path crosses crates on purpose.** The dependency inversion moved the
/// device headers to `kernels-cuda-new/csrc/src/norm/rmsnorm.cuh` — the JIT
/// compiles those — and left the host launchers in `kernels-cuda/csrc`, which
/// is where `rmsnorm_vec8_ok` lives, because a host `if` is exactly the thing
/// the JIT does not need. If this `include_str!` ever fails to resolve, the
/// launcher has moved too, and the right response is to follow it rather than
/// to drop the pin: without it the transliteration below is unwitnessed.
#[test]
fn the_cpp_predicate_is_the_one_this_file_was_written_from() {
    const CU: &str = include_str!("../../kernels-cuda/csrc/src/norm/rmsnorm.cu");

    let start = CU.find("inline bool rmsnorm_vec8_ok").expect("`rmsnorm_vec8_ok` is in rmsnorm.cu");
    let body = &CU[start..];
    let end = body.find("\n}\n").expect("the predicate has an end") + 3;
    let normalised: String = body[..end].split_whitespace().collect::<Vec<_>>().join(" ");

    const PINNED: &str = "inline bool rmsnorm_vec8_ok(const void* x, const void* y, const void* weight, \
         int hidden, int x_row_stride, int y_row_stride) { auto aligned = [](const void* p) { \
         return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }; return hidden % 8 == 0 && \
         x_row_stride % 8 == 0 && y_row_stride % 8 == 0 && aligned(x) && aligned(y) && \
         aligned(weight); }";

    assert_eq!(
        normalised, PINNED,
        "\n`rmsnorm_vec8_ok` has changed. `families::norm::RMSNORM_STRIDED_VEC8`'s terms and \
         this file's `rmsnorm_vec8_ok` transliteration were written from the previous text and \
         are now a DIFFERENT DECISION wearing the same name. Update all three together."
    );
}

/// The arm's terms are the C++'s six clauses, one for one.
///
/// The sweep proves the two agree on 98 304 points; this proves they agree
/// for the right reason. A `Select` that happened to coincide with
/// `rmsnorm_vec8_ok` over the swept space — testing `x` twice and `weight`
/// never, say, on a sweep where the two were always equal — would pass the
/// sweep and fail here.
#[test]
fn the_terms_are_the_clauses() {
    let spec = device::specialisation(BASE).expect("the base row is specialised");
    assert_eq!(spec.arms.len(), 1, "one arm, so first-match order is not a decision");
    let arm = &spec.arms[0];
    assert_eq!(
        arm.when,
        &[
            Term::Multiple { operand: 3, of: 8 },
            Term::Multiple { operand: 4, of: 8 },
            Term::Multiple { operand: 5, of: 8 },
            Term::Aligned { operand: 0, bytes: 16 },
            Term::Aligned { operand: 2, bytes: 16 },
            Term::Aligned { operand: 1, bytes: 16 },
        ],
        "the terms are `rmsnorm_vec8_ok`'s clauses in `rmsnorm_vec8_ok`'s order"
    );
    assert!(
        arm.because.contains("rmsnorm.cu") && arm.because.contains("rmsnorm_vec8_ok"),
        "an arm cites the host code it reproduces, or a reader has nothing to check it against"
    );
}

/// The one operand whose element type the row cannot state is never sourced.
///
/// **The gap this pins is real and is in [`kernels::Ty`], not in the row.**
/// `rmsnorm_vec8` takes `f16* y_fp16` where `x`, `weight` and `y` are
/// `bf16*`, and `Ty::BufMut` is "an opaque device buffer the launcher may
/// write through (`void*`)" — the vocabulary says in its own doc that it is
/// not describing what a buffer contains, which is why `q`, `k` and
/// `k_pages` are all one `Ty`. So the row states `BufMut` for both, the
/// generated facade takes `*mut c_void` for both, and `Args::bind` checks
/// pointer-versus-scalar and not width. Nothing between the caller and the
/// launch can tell the two apart.
///
/// Two things make that inert, and this test pins both rather than trusting
/// either: the reshape puts [`Take::Null`] in the slot, so no caller value
/// ever reaches it; and `elem` ends `false`, which is `EMIT_FP16`, so the
/// store is inside an `if constexpr` that is not compiled.
///
/// **What breaks without it.** An edit that changed slot 3 to `From(2)` —
/// plausible, since `y` is the buffer the result goes to — would pass
/// [`Specialisation::agrees`], because both operands are `BufMut` and that
/// is all the check can see; it would bind; and at `EMIT_FP16 = true` it
/// would write half-width data into a bf16 buffer at legal addresses, with
/// no fault and no diagnostic. That is the fast-wrong-answer failure this
/// whole file exists to make impossible, arriving through the reshape
/// instead of through the predicate.
#[test]
fn the_fp16_operand_is_nulled_and_compiled_away() {
    let arm = &device::specialisation(BASE).expect("specialised").arms[0];
    assert_eq!(
        arm.take[3],
        Take::Null,
        "`y_fp16` is `f16*` where the row can only say `BufMut`; the null is what makes \
         that safe, and sourcing it would be a silent half-width write"
    );
    assert_eq!(
        arm.take.iter().filter(|take| **take == Take::Null).count(),
        1,
        "exactly one invented value, so the reshape moves the caller's arguments and \
         does not manufacture them"
    );
    assert!(
        arm.row.elem.ends_with("false"),
        "`EMIT_FP16` is the last template argument and it is `false`, which is the \
         second guard: `{}`",
        arm.row.elem
    );
}

// ---------------------------------------------------------------------------
// The structural checks — what a machine can prove without a device
// ---------------------------------------------------------------------------

/// Every specialisation in the tree is well formed.
///
/// [`Specialisation::agrees`] is the check that keeps an arm from being a
/// second contract: same launch rule, same unit, same operand types through
/// the reshape, every term reading an operand that exists and is the kind it
/// tests. Run over [`device::specialisations`] so a family that specialises
/// tomorrow is covered without editing this file — which is the whole point
/// of registering per family rather than per row.
#[test]
fn every_specialisation_agrees_with_its_base() {
    let mut n = 0;
    for spec in device::specialisations() {
        spec.agrees().unwrap_or_else(|why| panic!("{why}"));
        n += 1;
    }
    assert!(n > 0, "the table came across empty, so this proved nothing");
    assert!(
        device::SPECIALISED.iter().all(|family| !family.is_empty()),
        "a family registered an empty slice; it either specialises or it does not appear"
    );
}

/// The base and the variant are two rows of one unit, so one compile serves
/// both.
///
/// The property that keeps the module cache a `OnceLock` per unit. If the
/// variant lived anywhere else, the first fire that happened to arrive with
/// aligned pointers would pay for a second NVRTC compile — a cold-start stall
/// whose timing depends on the data.
#[test]
fn the_arm_is_a_row_of_the_bases_own_unit() {
    use kernels_cuda_new::unit;
    let (_, base) = unit::unit_of(BASE).expect("the base is hosted");
    let (_, variant) = unit::unit_of(VARIANT).expect("the variant is hosted");
    assert_eq!(base.name, variant.name, "two units would be two cubins");
    assert_eq!(base.name, "norm/rmsnorm");
    assert!(
        base.instantiations().iter().any(|i| i.contains("rmsnorm_vec8")),
        "the unit's instantiation list is what an offline cubin cache keys on; \
         an arm missing from it is an arm a stale cubin does not carry"
    );
}

/// A malformed arm is refused, one malformation at a time.
///
/// **The negative control for [`Specialisation::agrees`].** The check is the
/// only thing standing between a specialisation and a launch that runs with
/// its arguments permuted, so a version of it that accepted everything would
/// pass every other test in this file. Each case below is a real way to get
/// it wrong: a variant at a different geometry, a reshape that swaps a width
/// for a pointer, a null on an operand that is not nullable, a term reading
/// past the row, a predicate that always holds, an alignment that is not a
/// power of two.
///
/// The doctored rows are leaked rather than declared as statics, and that is
/// deliberate: a `static KernelSig` here would be a SECOND copy of the
/// variant's contract, which is the thing this crate says two tables always
/// become. Leaking borrows the real one and changes the single field each
/// case is about. Leaks in a test process are freed by exit.
#[test]
fn agrees_refuses_every_way_of_getting_an_arm_wrong() {
    use kernels::{LaunchRule, kernel, operands};
    use kernels_cuda_new::device::DeviceKernel;

    let real = device::specialisation(BASE).expect("specialised");
    let row: &'static DeviceKernel = real.arms[0].row;

    /// One control: an arm built from the real row with one thing changed.
    fn spec(name: &'static str, when: &'static [Term], take: &'static [Take], row: &'static DeviceKernel) -> Specialisation {
        let arms: &'static [Arm] = Box::leak(Box::new([Arm {
            name,
            when,
            row,
            take,
            because: "a control",
        }]));
        Specialisation { base: BASE, arms }
    }

    // A variant at another geometry. `Rule::Rms` is one block per row at 256
    // threads; `Elementwise` is `rows * width / 256` blocks -- so this arm
    // would launch 16 blocks over a kernel that indexes its row by
    // `blockIdx.x` and norm 16 rows of a one-row tensor.
    let elsewhere = kernel!(wrong_rule "norm::rmsnorm_strided_bf16#vec8",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf, weight: Buf, y: BufMut, y_fp16: BufMut | null,
            hidden: I32, x_row_stride: I32, y_row_stride: I32, eps: F32,
        ]);
    let elsewhere_row: &'static DeviceKernel = Box::leak(Box::new(DeviceKernel {
        sig: Box::leak(Box::new(elsewhere)),
        template_path: row.template_path,
        elem: row.elem,
    }));

    // A row of no unit at all -- the arm that would need a second compile.
    let unhosted = kernel!(nowhere "norm::a_kernel_nobody_wrote",
        file = Some("norm/rmsnorm.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            x: Buf, weight: Buf, y: BufMut, y_fp16: BufMut | null,
            hidden: I32, x_row_stride: I32, y_row_stride: I32, eps: F32,
        ]);
    let unhosted_row: &'static DeviceKernel = Box::leak(Box::new(DeviceKernel {
        sig: Box::leak(Box::new(unhosted)),
        template_path: row.template_path,
        elem: row.elem,
    }));

    static SOME_TERMS: [Term; 1] = [Term::Multiple { operand: 3, of: 8 }];
    // Argument 0 is `x: Buf` and operand 3 is `hidden: I32`.
    static SWAPPED: [Take; 8] = [
        Take::From(3),
        Take::From(1),
        Take::From(2),
        Take::Null,
        Take::From(0),
        Take::From(4),
        Take::From(5),
        Take::From(6),
    ];
    // `y` is argument 2 and is not declared nullable.
    static NULLED_Y: [Take; 8] = [
        Take::From(0),
        Take::From(1),
        Take::Null,
        Take::Null,
        Take::From(3),
        Take::From(4),
        Take::From(5),
        Take::From(6),
    ];
    static OFF_THE_END: [Take; 8] = [
        Take::From(0),
        Take::From(1),
        Take::From(2),
        Take::Null,
        Take::From(3),
        Take::From(4),
        Take::From(5),
        Take::From(99),
    ];

    let controls: [(Specialisation, &str); 9] = [
        (Specialisation { base: BASE, arms: &[] }, "no arms"),
        (spec("no terms", &[], &TAKE_OK, row), "applies always"),
        (spec("short take", &SOME_TERMS, &[Take::From(0)], row), "arguments and"),
        (spec("swapped", &SOME_TERMS, &SWAPPED, row), "(Buf) from `hidden` (I32)"),
        (spec("nulled y", &SOME_TERMS, &NULLED_Y, row), "does not declare nullable"),
        (spec("take off the end", &SOME_TERMS, &OFF_THE_END, row), "operand 99 of a row with 7"),
        (
            spec("term past the end", &[Term::Multiple { operand: 99, of: 8 }], &TAKE_OK, row),
            "reads operand 99",
        ),
        (
            spec("alignment on a width", &[Term::Aligned { operand: 3, bytes: 16 }], &TAKE_OK, row),
            "tests the alignment of `hidden`",
        ),
        (
            spec("not a power of two", &[Term::Aligned { operand: 0, bytes: 24 }], &TAKE_OK, row),
            "MASK",
        ),
    ];
    for (control, expect) in &controls {
        let name = control.arms.first().map_or("no arms", |arm| arm.name);
        let why = control.agrees().expect_err("a malformed arm must be refused");
        assert!(why.contains(expect), "`{name}` was refused with `{why}`, wanted `{expect}`");
    }

    let why = spec("wrong rule", &SOME_TERMS, &TAKE_OK, elsewhere_row)
        .agrees()
        .expect_err("a variant at another geometry must be refused");
    assert!(why.contains("Elementwise") && why.contains("not a geometry"), "{why}");

    let why = spec("unhosted", &SOME_TERMS, &TAKE_OK, unhosted_row)
        .agrees()
        .expect_err("a variant no unit compiles must be refused");
    assert!(why.contains("second cubin"), "{why}");

    // And the real one is accepted, so the checks above are not simply
    // rejecting everything.
    real.agrees().expect("the real specialisation is well formed");
}

/// **A `Specialisation` may not change a `LaunchRule`, on the launcher that
/// would need it to.**
///
/// The case above proves the check with a DOCTORED row. This one proves it
/// with the two real ones, because the invariant is what refuses
/// `attn::qkv_decode_qk_norm_rope_write_kv_bf16` and a refusal resting on a
/// synthetic fixture is a refusal resting on a fixture.
///
/// `attn/qkv_fused.cu`'s decode launcher is four kernels behind one symbol at
/// TWO geometries:
///
/// ```text
/// :50-53   WARP_BLOCK = 256, total = num_requests * (num_q_heads + num_kv_heads),
///          warp_grid((total + 7) / 8)      -> WarpPackedHeads
/// :97-99   BLOCK = 128, dim3(num_requests, num_q_heads + num_kv_heads)
///                                          -> RowsPackedHeadsNarrow
/// ```
///
/// Both rules are ported, both were ported FROM this launcher, and both of
/// this file's rows for them live in the same unit — so every other clause of
/// `agrees` passes and the rule mismatch is the only thing standing. That is
/// what makes this the honest test of the invariant: nothing else is wrong
/// with the arm.
///
/// It also measures why the invariant is not bookkeeping. The two rules at
/// ONE shape give grids that are neither equal nor a scaling of each other,
/// and a specialisation that could swap them would have `runtime::fire`
/// computing one and running the other — `fire.rs:176-186` evaluates the
/// geometry from the BASE row before it consults the specialisation at all.
#[test]
fn agrees_refuses_an_arm_that_changes_the_launch_rule() {
    use kernels::LaunchRule;
    use kernels_cuda_new::unit;

    const DECODE: &str = "attn::qkv_decode_qk_norm_rope_write_kv";
    const WARP: &str = "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128";

    let (_, host) = unit::unit_of(DECODE).expect("the block decode row is hosted");
    assert_eq!(host.name, "attn/qkv_fused");
    let block = host.row(DECODE).expect("a row").sig;
    let warp = host.row(WARP).expect("a row").sig;

    // The premise: same unit, and two DIFFERENT rules, both cited to this
    // launcher. If either half stops holding, the test below is measuring
    // something else.
    assert!(host.hosts(WARP), "both rows are in one unit, so `agrees` gets past the cubin check");
    assert_eq!(block.launch, LaunchRule::RowsPackedHeadsNarrow);
    assert_eq!(warp.launch, LaunchRule::WarpPackedHeads);

    // A Qwen3-shaped decode: 4 requests, 32 query heads over 8 key heads.
    // `qkv_fused.cu:51` reads `num_requests * (num_q_heads + num_kv_heads)`
    // and `:98` reads the same two numbers as separate axes.
    //
    // `runtime` is behind `_cuda`, and the REFUSAL below is not — which is
    // the right split: the invariant is arithmetic over a table and holds on
    // a machine with no toolkit, while the measurement of what the two rules
    // give is the part that needs `eval`.
    #[cfg(feature = "_cuda")]
    {
        use kernels_cuda_new::runtime::{Dims, eval};
        let shape = Dims {
            rows: 4,
            width: 40 * 128,
            in_width: 40 * 128,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            stated_head_dim: 0,
            rotary_dims: 128,
            n_experts: 0,
            experts_per_token: 0,
            requests: 4,
            altup_streams: 0,
        };
        let narrow = eval(block.launch, shape).expect("ported");
        let packed = eval(warp.launch, shape).expect("ported");
        assert_eq!(narrow.grid, [4, 40, 1], "`:98` dim3(num_requests, q + kv)");
        assert_eq!(narrow.block, [128, 1, 1], "`:97` BLOCK = 128");
        assert_eq!(packed.grid, [20, 1, 1], "`:52` (4 * 40 + 7) / 8");
        assert_eq!(packed.block, [256, 1, 1], "`:50` WARP_BLOCK = 256");
        assert_ne!(
            narrow.grid, packed.grid,
            "if the two geometries agreed at this shape the refusal below would be free"
        );
    }

    // The arm the launcher would need, built from the REAL warp row.
    let arms: &'static [Arm] = Box::leak(Box::new([Arm {
        name: "d128",
        // `head_dim == 128` is not spellable, so this stands in for it with
        // the nearest clause that IS -- and the point of the test is that the
        // arm is refused before its predicate is ever reached.
        when: &[Term::Multiple { operand: 17, of: 64 }],
        row: host.rows.iter().find(|row| row.sig.symbol == WARP).expect("a row"),
        take: &TAKE_QKV,
        because: "qkv_fused.cu:85 `if (head_dim == 128)` -> the warp form at :101",
    }]));
    let cross = Specialisation { base: DECODE, arms };

    let why = cross.agrees().expect_err("an arm at another geometry must be refused");
    assert!(
        why.contains("WarpPackedHeads")
            && why.contains("RowsPackedHeadsNarrow")
            && why.contains("not a geometry"),
        "the refusal must name both rules and the reason, and says `{why}`"
    );

    // And the same arm pointed at a row that agrees on the rule is accepted,
    // so what refused it was the geometry and not the shape of the fixture.
    let same: &'static [Arm] = Box::leak(Box::new([Arm {
        name: "rope",
        when: &[Term::Present { operand: 7, value: true }],
        row: host
            .rows
            .iter()
            .find(|row| row.sig.symbol == "attn::qkv_decode_qk_norm_rope_write_kv#rope")
            .expect("a row"),
        take: &TAKE_QKV,
        because: "qkv_fused.cu:100 `rope_table != nullptr`",
    }]));
    Specialisation { base: DECODE, arms: same }
        .agrees()
        .expect("the same fixture at the same rule is well formed");
}

/// The identity over `qkv_decode_qk_norm_rope_write_kv`'s twenty-two
/// operands.
static TAKE_QKV: [Take; 22] = {
    let mut take = [Take::Null; 22];
    let mut i = 0;
    while i < 22 {
        take[i] = Take::From(i);
        i += 1;
    }
    take
};

/// A well-formed reshape, for controls that are wrong somewhere else.
static TAKE_OK: [Take; 8] = [
    Take::From(0),
    Take::From(1),
    Take::From(2),
    Take::Null,
    Take::From(3),
    Take::From(4),
    Take::From(5),
    Take::From(6),
];

/// A predicate doctored to disagree, and the sweep catching it.
///
/// **The negative control for the agreement sweep itself.** Dropping the
/// `weight` alignment clause is the realistic mistake — `weight` is the one
/// pointer of the three that is not an activation, so it is the one an author
/// forgets — and the resulting `Select` is correct on every case where the
/// weight happens to be aligned, which in a real model is nearly all of them.
/// The sweep has to catch it anyway, and this measures that it does.
///
/// Note what `agrees` says about this arm: nothing. It is structurally
/// perfect — same unit, same rule, well-typed reshape, five terms that read
/// operands that exist and are the kinds they test. **No check inside this
/// crate can catch it.** Only a sweep against the C++ can, which is why the
/// sweep is the gate and `agrees` is the floor.
#[test]
fn a_predicate_that_drops_a_clause_is_caught() {
    let real = device::specialisation(BASE).expect("specialised");
    static FIVE_OF_SIX: [Term; 5] = [
        Term::Multiple { operand: 3, of: 8 },
        Term::Multiple { operand: 4, of: 8 },
        Term::Multiple { operand: 5, of: 8 },
        Term::Aligned { operand: 0, bytes: 16 },
        Term::Aligned { operand: 2, bytes: 16 },
    ];
    let arms: &'static [Arm] = Box::leak(Box::new([Arm {
        name: "vec8 without the weight clause",
        when: &FIVE_OF_SIX,
        row: real.arms[0].row,
        take: &TAKE_OK,
        because: "a control",
    }]));
    let doctored = Specialisation { base: BASE, arms };
    doctored.agrees().expect("a dropped clause is not a STRUCTURAL defect");

    let mut caught = 0_u32;
    let mut cases = 0_u32;
    for xo in OFFSETS {
        for wo in OFFSETS {
            for yo in OFFSETS {
                for hidden in WIDTHS {
                    let (x, weight, y) = (ARENA + xo, ARENA + wo, ARENA + yo);
                    let want = rmsnorm_vec8_ok(x, y, weight, hidden, hidden, hidden);
                    let got = doctored
                        .choose(&facts(x, weight, y, hidden, hidden, hidden))
                        .expect("no fault")
                        .is_some();
                    cases += 1;
                    if got != want {
                        caught += 1;
                    }
                }
            }
        }
    }
    println!("dropped-clause control: {caught} of {cases} cases disagree with the C++");
    assert!(
        caught > 0,
        "the sweep did not notice a missing clause, so it would not notice a real one"
    );
}

/// A term that cannot read its operand faults rather than answering.
///
/// The distinction the design turns on: "the predicate was false" fires the
/// base row and is correct, while "the term could not be evaluated" means the
/// terms and the row have drifted and the specialisation is not the decision
/// it claims to be. The second is a refusal — see
/// [`kernels_cuda_new::runtime::Error::Specialise`].
#[test]
fn a_term_that_cannot_read_its_operand_faults() {
    let real = device::specialisation(BASE).expect("specialised");
    let one = |when: &'static [Term]| {
        let arms: &'static [Arm] = Box::leak(Box::new([Arm {
            name: "a control",
            when,
            row: real.arms[0].row,
            take: &TAKE_OK,
            because: "a control",
        }]));
        Specialisation { base: BASE, arms }
    };

    let Err(fault) = one(&[Term::Multiple { operand: 40, of: 8 }])
        .choose(&facts(ARENA, ARENA, ARENA, 4096, 4096, 4096))
    else {
        panic!("a term past the end must fault");
    };
    assert_eq!(fault, device::Fault::Range { operand: 40, arity: 7 });

    let Err(fault) = one(&[Term::Aligned { operand: 6, bytes: 16 }])
        .choose(&facts(ARENA, ARENA, ARENA, 4096, 4096, 4096))
    else {
        panic!("an alignment on `eps` must fault");
    };
    assert_eq!(fault, device::Fault::Kind { operand: 6, wanted: "an address" });

    // And the fault is not hidden behind an earlier `false`. A
    // short-circuiting `all` would be faster and would let a broken term
    // surface only on the inputs that reached it — which is to say in
    // production and not in this file. `hidden` 4097 makes the first clause
    // false; the second still faults.
    let Err(fault) = one(&[
        Term::Multiple { operand: 3, of: 8 },
        Term::Multiple { operand: 40, of: 8 },
    ])
    .choose(&facts(ARENA, ARENA, ARENA, 4097, 4097, 4097))
    else {
        panic!("a fault behind a false must still fault");
    };
    assert_eq!(fault, device::Fault::Range { operand: 40, arity: 7 });
}

// ---------------------------------------------------------------------------
// 5. The second shape: a flag, and the `template <bool>` it selects
// ---------------------------------------------------------------------------
//
// `rmsnorm_vec8_ok` is a predicate the host COMPUTES, out of pointers and
// strides. The nine rows blocked in `attn` and `rope` are the other shape: a
// flag the host was HANDED, choosing between two arms of a `template <bool>`.
// `kv_paged.cu:51` is the whole of it —
//
//     if (hnd_layout) { write_kv<true> <<<n, 256, 0, s>>>(k_curr, ..., first_token); }
//     else            { write_kv<false><<<n, 256, 0, s>>>(k_curr, ..., first_token); }
//
// The machinery is `Fact::Bool` and `Term::Is`. What follows is the argument
// for that pair over the shorter change — reusing `Fact::Int` and carrying the
// flag as 0 and 1 — made as tests rather than as a comment.

/// `moe::topk_sqrtsoftplus_bf16`, the only shipped base with a `Ty::Bool`.
///
/// Borrowed here because `agrees` resolves its base row out of
/// [`kernels_cuda_new::unit::UNITS`], so a base invented in this file has no
/// unit and is refused before any flag is looked at. `renormalize` is operand
/// 6 and is a host flag in exactly the sense the nine are: a deployment's
/// `norm_topk_prob`, read at load, constant for the process.
///
/// The kernel it names is not a `template <bool>` and nothing here fires it.
/// These four cases are about what `agrees` ACCEPTS and REFUSES, which is a
/// question about two operand lists and a predicate and has no device in it.
const FLAG_BASE: &str = "moe::topk_sqrtsoftplus_bf16";

/// The flag's index in [`FLAG_BASE`]'s operand list.
const FLAG: usize = 6;

/// Every bound value becomes the fact its kind allows, and no other.
///
/// The mapping is [`kernels_cuda_new::runtime::ArgValue::fact`] and it is the
/// only one — `runtime::fire::facts` calls it and nothing else constructs a
/// [`Fact`] on the launch path. Pinned variant by variant because this is the
/// point at which a predicate stops being able to see anything the host did
/// not already have: a pointer arrives as a NUMBER, and every kind no term may
/// read arrives as [`Fact::Opaque`] rather than as a bit pattern that happens
/// to divide by 8.
///
/// Gated on `_cuda` and NOT on a device: [`kernels_cuda_new::runtime`] is the
/// module the feature carries, so this cannot be compiled without it, but
/// nothing here allocates, launches or asks what card is present.
#[cfg(feature = "_cuda")]
#[test]
fn every_bound_value_becomes_the_fact_its_kind_allows() {
    use kernels_cuda_new::runtime::ArgValue;
    use std::ffi::c_void;

    assert_eq!(ArgValue::Ptr(ARENA as *mut c_void).fact(), Fact::Address(ARENA));
    assert_eq!(ArgValue::Ptr(std::ptr::null_mut()).fact(), Fact::Address(0));
    assert_eq!(ArgValue::I32(4096).fact(), Fact::Int(4096));
    assert_eq!(ArgValue::I32(-1).fact(), Fact::Int(-1), "the sign survives the widening");
    assert_eq!(ArgValue::U32(u32::MAX).fact(), Fact::Int(4_294_967_295), "and is not invented");
    assert_eq!(ArgValue::I64(i64::MIN).fact(), Fact::Int(i64::MIN));

    // The change this file is about.
    assert_eq!(ArgValue::Bool(true).fact(), Fact::Bool(true));
    assert_eq!(ArgValue::Bool(false).fact(), Fact::Bool(false));
    assert_ne!(ArgValue::Bool(false).fact(), Fact::Int(0), "a flag is not a zero");
    assert_ne!(ArgValue::Bool(true).fact(), Fact::Int(1), "a flag is not a one");

    // And the two kinds no term may read at all. `F32` is the one that would
    // be silently wrong: `1e-6` is `0x358637bd`, which is odd, and `1.0` is
    // `0x3f800000`, which divides by 8 — so a `Multiple` over an epsilon
    // would ANSWER, differently for different epsilons.
    assert_eq!(ArgValue::F32(1e-6).fact(), Fact::Opaque);
    assert_eq!(ArgValue::Usize(4096).fact(), Fact::Opaque);
}

/// A flag clause reads a flag, and an arithmetic clause cannot read one.
///
/// **The whole case for `Fact::Bool` over `Fact::Int`, in six assertions.**
/// Had the flag arrived as `Int(0 | 1)`, the last two lines would each be an
/// `Ok` instead of a `Fault`: `Multiple { of: 2 }` would be a well-formed
/// clause meaning *"the flag is false"*, and `Aligned { bytes: 1 }` a
/// well-formed clause that is always true. §18.4's finding is that the fatal
/// shape is a predicate that is well formed and WRONG — the kind no check
/// inside this crate can catch, because there is nothing structurally
/// defective about it. `Fact::Bool` makes that shape unspellable rather than
/// discouraged, which is the same move [`Fact`] already makes for a
/// device-side value one level out.
#[test]
fn a_flag_clause_reads_a_flag_and_arithmetic_cannot() {
    let selects_true = Term::Is { operand: 0, value: true };
    let selects_false = Term::Is { operand: 0, value: false };

    assert_eq!(selects_true.holds(&[Fact::Bool(true)]), Ok(true));
    assert_eq!(selects_true.holds(&[Fact::Bool(false)]), Ok(false));
    assert_eq!(selects_false.holds(&[Fact::Bool(false)]), Ok(true));
    assert_eq!(selects_false.holds(&[Fact::Bool(true)]), Ok(false));

    // An equality, and not a truth test with a negation around it: both arms
    // of `kv_paged.cu:51` are instantiations a fire must be able to NAME, and
    // an arm spelled as the negation of another reads as a fallback it is not.
    assert_eq!(selects_true.operand(), selects_false.operand());

    // A flag clause over anything else faults rather than guessing.
    for other in [Fact::Int(1), Fact::Int(0), Fact::Address(ARENA), Fact::Opaque] {
        assert_eq!(
            selects_true.holds(&[other]),
            Err(device::Fault::Kind { operand: 0, wanted: "a flag" }),
            "a flag clause read a {other:?}"
        );
    }
    assert_eq!(selects_true.holds(&[]), Err(device::Fault::Range { operand: 0, arity: 0 }));

    // And the closure the other way, which is the part `Fact::Int` would have
    // thrown away.
    assert_eq!(
        Term::Multiple { operand: 0, of: 2 }.holds(&[Fact::Bool(false)]),
        Err(device::Fault::Kind { operand: 0, wanted: "an integer" }),
        "dividing a flag by two must be a fault and not the answer `true`"
    );
    assert_eq!(
        Term::Aligned { operand: 0, bytes: 1 }.holds(&[Fact::Bool(true)]),
        Err(device::Fault::Kind { operand: 0, wanted: "an address" }),
        "aligning a flag must be a fault and not the answer `true`"
    );
}

/// One arm over [`FLAG_BASE`], built out of a doctored variant row.
///
/// The variant carries the base's SYMBOL, because `agrees` requires the unit
/// to host it and no unit hosts a symbol this file invents. That is not a
/// shortcut around the check — it is the shape a real flag row has, one
/// template compiled at two arguments, with the `#hnd`/`#nhd` suffixes a
/// family module would add and this test may not.
#[cfg(test)]
fn flag_spec(
    arms: &'static [(&'static str, bool)],
    drops_the_flag: bool,
) -> Specialisation {
    use kernels::{LaunchRule, kernel, operands};
    use kernels_cuda_new::device::DeviceKernel;

    // `moe::topk_sqrtsoftplus_bf16` minus its flag: seven operands, the shape
    // an instantiation that BAKED `renormalize` would declare.
    let without = kernel!(baked "moe::topk_sqrtsoftplus_bf16",
        file = Some("moe/dsv4_routing.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            logits: Buf, topk_idx: I32sMut, topk_w: F32sMut, correction_bias: F32s,
            num_experts: I32, top_k: I32, routed_scaling_factor: F32,
        ]);
    // The same list with the flag still on it — an instantiation that takes it
    // at run time, which is the shape that needs no coverage.
    let with = kernel!(taken "moe::topk_sqrtsoftplus_bf16",
        file = Some("moe/dsv4_routing.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            logits: Buf, topk_idx: I32sMut, topk_w: F32sMut, correction_bias: F32s,
            num_experts: I32, top_k: I32, renormalize: Bool, routed_scaling_factor: F32,
        ]);
    static DROPS: [Take; 7] = [
        Take::From(0),
        Take::From(1),
        Take::From(2),
        Take::From(3),
        Take::From(4),
        Take::From(5),
        Take::From(7),
    ];
    static FORWARDS: [Take; 8] = [
        Take::From(0),
        Take::From(1),
        Take::From(2),
        Take::From(3),
        Take::From(4),
        Take::From(5),
        Take::From(6),
        Take::From(7),
    ];
    let sig = if drops_the_flag { without } else { with };
    let row: &'static DeviceKernel = Box::leak(Box::new(DeviceKernel {
        sig: Box::leak(Box::new(sig)),
        template_path: "moe::device::topk_sqrtsoftplus",
        elem: "device::bf16",
    }));
    let built: Vec<Arm> = arms
        .iter()
        .map(|(name, value)| Arm {
            name,
            when: Box::leak(Box::new([Term::Is { operand: FLAG, value: *value }])),
            row,
            take: if drops_the_flag { &DROPS } else { &FORWARDS },
            because: "a control",
        })
        .collect();
    Specialisation { base: FLAG_BASE, arms: Box::leak(built.into_boxed_slice()) }
}

/// A flag no arm forwards must be covered both ways, and `agrees` says so.
///
/// **The refusal this change had to add, and the reason a flag is not just
/// another term.** `write_kv<HND_LAYOUT>` takes the same FIFTEEN parameters
/// whichever way it is instantiated; `hnd_layout` is the launcher's argument
/// and no kernel's. So the base row carries sixteen operands, each arm drops
/// the sixteenth through [`Take`], and if a fire's flag matches no arm it
/// falls through to a base row that binds SIXTEEN cells for a kernel
/// declaring fifteen.
///
/// `cuLaunchKernel` reads the parameter count out of the cubin. The sixteenth
/// cell is never read, the launch succeeds, and the result is a wrong kernel
/// reporting success — which is the one failure this whole design exists to
/// rule out. The rule that removes it is that the arms must be TOTAL over
/// such a flag, and totality is decided here by enumeration, with no device.
#[test]
fn a_flag_no_arm_forwards_must_be_covered_both_ways() {
    static ONE_WAY: [(&str, bool); 1] = [("renorm", true)];
    static BOTH_WAYS: [(&str, bool); 2] = [("renorm", true), ("plain", false)];

    let why = flag_spec(&ONE_WAY, true).agrees().expect_err("half a flag must be refused");
    assert!(why.contains("`renormalize` = false"), "{why}");
    assert!(why.contains("8 cells"), "the refusal counts the cells the base would bind: {why}");
    assert!(why.contains("State the other arm."), "{why}");

    flag_spec(&BOTH_WAYS, true).agrees().expect("both values stated is a total specialisation");

    // And the exemption, which is not a loophole: an arm that FORWARDS the
    // flag hands it to a kernel that declares it, so the base row and the
    // instantiation have the same arity and a fall-through binds exactly what
    // the kernel reads. `rope::device::rotate<kWriteKv, kHnd>` is that shape
    // for its `interleaved`.
    flag_spec(&ONE_WAY, false)
        .agrees()
        .expect("a flag the arm forwards needs no coverage — the kernel takes it");
}

/// An arm that also tests a pointer does not discharge a flag.
///
/// **The one place the coverage check is deliberately pessimistic, and the
/// reason it has to be.** Totality is decided with no fire in hand, so an
/// alignment clause is unknowable at the point of decision — and an arm that
/// might not fire cannot be counted as the one that catches
/// `renormalize = false`. The check therefore counts an arm towards coverage
/// only when EVERY clause it states is a [`Term::Is`].
///
/// Without that restriction the enumeration would read a mixed arm as
/// unconditional, call the flag covered, and admit exactly the table the
/// check exists to refuse: a fire whose pointer is misaligned AND whose flag
/// is `false` matches no arm, falls through to the base, and binds eight
/// cells for a seven-operand kernel. The refusal is louder than the bug, so
/// this is the right direction to be wrong in — a real mixed arm gets told to
/// state a total flag arm rather than gets silently trusted.
#[test]
fn an_arm_that_also_tests_a_pointer_does_not_cover_a_flag() {
    use kernels::{LaunchRule, kernel, operands};
    use kernels_cuda_new::device::DeviceKernel;

    let baked = kernel!(baked "moe::topk_sqrtsoftplus_bf16",
        file = Some("moe/dsv4_routing.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            logits: Buf, topk_idx: I32sMut, topk_w: F32sMut, correction_bias: F32s,
            num_experts: I32, top_k: I32, routed_scaling_factor: F32,
        ]);
    let row: &'static DeviceKernel = Box::leak(Box::new(DeviceKernel {
        sig: Box::leak(Box::new(baked)),
        template_path: "moe::device::topk_sqrtsoftplus",
        elem: "device::bf16",
    }));
    static DROPS: [Take; 7] = [
        Take::From(0),
        Take::From(1),
        Take::From(2),
        Take::From(3),
        Take::From(4),
        Take::From(5),
        Take::From(7),
    ];
    let arm = |name: &'static str, when: &'static [Term]| Arm {
        name,
        when,
        row,
        take: &DROPS,
        because: "a control",
    };

    // Both values of the flag are named, so a reader scanning the arms sees a
    // total specialisation. The `false` arm is the one that is not total.
    let arms: &'static [Arm] = Box::leak(Box::new([
        arm("renorm", &[Term::Is { operand: FLAG, value: true }]),
        arm(
            "plain",
            &[
                Term::Is { operand: FLAG, value: false },
                Term::Aligned { operand: 0, bytes: 16 },
            ],
        ),
    ]));
    let why = Specialisation { base: FLAG_BASE, arms }
        .agrees()
        .expect_err("an arm gated on an unknowable clause cannot be the one that covers");
    assert!(
        why.contains("`renormalize` = false"),
        "the refusal names the value that is only conditionally handled: {why}"
    );

    // And the control on the control: drop the alignment and the same two arms
    // are accepted, so the refusal above is about the extra clause and not
    // about the shape of the table.
    let arms: &'static [Arm] = Box::leak(Box::new([
        arm("renorm", &[Term::Is { operand: FLAG, value: true }]),
        arm("plain", &[Term::Is { operand: FLAG, value: false }]),
    ]));
    Specialisation { base: FLAG_BASE, arms }
        .agrees()
        .expect("two pure flag arms are total");
}

/// The three ways a flag clause can be written against the wrong operand.
///
/// All three are refused with no device, which is the floor the design asks
/// of `agrees`: a clause that could never be true, a clause that reads a flag
/// as a number, and a clause that reads a number as a flag.
#[test]
fn agrees_refuses_a_flag_clause_over_the_wrong_operand() {
    use kernels::{LaunchRule, kernel, operands};
    use kernels_cuda_new::device::DeviceKernel;

    let real = device::specialisation(BASE).expect("specialised");

    // (a) `Term::Is` over an operand that is not a Bool. `hidden` is an I32,
    // so a fire binds `Fact::Int` and the clause could never once be true —
    // the arm would be dead code that looked like a decision.
    let arms: &'static [Arm] = Box::leak(Box::new([Arm {
        name: "flag on a width",
        when: &[Term::Is { operand: 3, value: true }],
        row: real.arms[0].row,
        take: &TAKE_OK,
        because: "a control",
    }]));
    let why = Specialisation { base: BASE, arms }
        .agrees()
        .expect_err("a flag clause over an i32 must be refused");
    assert!(why.contains("selects on `hidden`") && why.contains("not a Bool"), "{why}");

    // (b) `Term::Multiple` over a Bool — the `Fact::Int` spelling's whole
    // liability, refused at build time as well as at fire time. It reads as a
    // parity test on a flag and it is not one.
    let taken = kernel!(taken "moe::topk_sqrtsoftplus_bf16",
        file = Some("moe/dsv4_routing.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            logits: Buf, topk_idx: I32sMut, topk_w: F32sMut, correction_bias: F32s,
            num_experts: I32, top_k: I32, renormalize: Bool, routed_scaling_factor: F32,
        ]);
    let row: &'static DeviceKernel = Box::leak(Box::new(DeviceKernel {
        sig: Box::leak(Box::new(taken)),
        template_path: "moe::device::topk_sqrtsoftplus",
        elem: "device::bf16",
    }));
    static FORWARDS: [Take; 8] = [
        Take::From(0),
        Take::From(1),
        Take::From(2),
        Take::From(3),
        Take::From(4),
        Take::From(5),
        Take::From(6),
        Take::From(7),
    ];
    let arms: &'static [Arm] = Box::leak(Box::new([Arm {
        name: "flag by parity",
        when: &[Term::Multiple { operand: FLAG, of: 2 }],
        row,
        take: &FORWARDS,
        because: "a control",
    }]));
    let why = Specialisation { base: FLAG_BASE, arms }
        .agrees()
        .expect_err("dividing a flag must be refused");
    assert!(why.contains("divides `renormalize`") && why.contains("Bool"), "{why}");

    // (c) `Term::Aligned` over a Bool — the same mistake with the other
    // arithmetic term.
    let arms: &'static [Arm] = Box::leak(Box::new([Arm {
        name: "flag by alignment",
        when: &[Term::Aligned { operand: FLAG, bytes: 1 }],
        row,
        take: &FORWARDS,
        because: "a control",
    }]));
    let why = Specialisation { base: FLAG_BASE, arms }
        .agrees()
        .expect_err("aligning a flag must be refused");
    assert!(why.contains("alignment of `renormalize`"), "{why}");
}

// ---------------------------------------------------------------------------
// 6. The third shape: a POINTER, and the `template <bool>` its presence
//    selects
// ---------------------------------------------------------------------------
//
// `Term::Is` unblocked five arms whose flag the host was HANDED as a `bool`.
// It did nothing for `attn::qkv_decode_qk_norm_rope_write_kv`, whose selector
// is `qkv_fused.cu:100`:
//
//     if (rope_table != nullptr) {
//         qkv_decode_qk_norm_rope_write_kv<BLOCK, true ><<<grid, BLOCK, 0, s>>>(...);  // :101
//     } else {
//         qkv_decode_qk_norm_rope_write_kv<BLOCK, false><<<grid, BLOCK, 0, s>>>(...);  // :126
//     }
//
// There is no `bool` operand to read. The predicate is a NULL TEST over an
// operand that is bound as an address, and the refusal that stood before
// `Term::Present` named the exact reason no existing term would do:
//
//     "`Term::Aligned` holds of address 0, so an alignment clause picks the
//      table arm for a fire with no table."
//
// The three tests below are that sentence as assertions, the mirror of
// `a_flag_clause_reads_a_flag_and_arithmetic_cannot` for the new term, and
// the two ways `agrees` refuses it.

/// `attn::qkv_decode_qk_norm_rope_write_kv`, whose selector is a pointer.
///
/// Twenty-two operands; `rope_table` is operand 7 and is `F32s | null`.
const NULL_BASE: &str = "attn::qkv_decode_qk_norm_rope_write_kv";

/// `rope_table`'s index in [`NULL_BASE`]'s operand list.
const TABLE: usize = 7;

/// A null clause reads a pointer, and every arithmetic clause over one lies.
///
/// **The whole case for `Term::Present` over the terms that already existed,
/// in the assertions that measure the alternative.** `Term::Aligned` is the
/// nearest spellable clause and it does not fault, it does not refuse, and it
/// does not answer `false`: it answers `Ok(true)` of address zero, for every
/// alignment, because `0 % n == 0`. A table written that way selects
/// `USE_ROPE_TABLE = true` for a fire that published no table, and
/// `qkv_fused.cuh:311` reads `rope_table[pos * head_dim + i]` off a null base.
///
/// `Term::Multiple` is refused a different way — it faults on an `Address`
/// rather than answering — but it is worth pinning too, because the fault is
/// what makes "read the pointer as a number and test it for divisibility by
/// one" unavailable as a spelling.
#[test]
fn a_null_clause_reads_a_pointer_and_arithmetic_cannot() {
    let has_table = Term::Present { operand: 0, value: true };
    let no_table = Term::Present { operand: 0, value: false };

    assert_eq!(has_table.holds(&[Fact::Address(ARENA)]), Ok(true));
    assert_eq!(has_table.holds(&[Fact::Address(0)]), Ok(false));
    assert_eq!(no_table.holds(&[Fact::Address(0)]), Ok(true));
    assert_eq!(no_table.holds(&[Fact::Address(ARENA)]), Ok(false));

    // The same argument `Term::Is` makes and for the same reason: both arms of
    // `qkv_fused.cu:100` are instantiations a fire must be able to NAME, so
    // the term carries a `value` and neither arm is spelled as the negation
    // of the other. An `Arm` whose `when` is "not the other one" has no
    // `because` of its own to cite.
    assert_eq!(has_table.operand(), no_table.operand());
    assert_ne!(has_table, no_table, "the two arms are distinct clauses, not one and a fallback");

    // ── the measurement the refusal was written from ─────────────────────
    //
    // Every alignment an operand of this kind could plausibly be given, over
    // a null. Not one of them is `false`.
    for bytes in [1u64, 4, 8, 16, 128, 256] {
        assert_eq!(
            Term::Aligned { operand: 0, bytes }.holds(&[Fact::Address(0)]),
            Ok(true),
            "`Aligned {{ bytes: {bytes} }}` over a null answered something other than true — \
             if this ever fails, the refusal `Term::Present` was added for has changed and \
             this term's doc needs rereading, not this assertion"
        );
    }
    // And the same clause over a real address is also true, at the alignments
    // an allocator gives — which is what makes it USELESS here rather than
    // merely wrong. It is the constant `true` function on this operand, and a
    // constant is not a decision.
    assert_eq!(Term::Aligned { operand: 0, bytes: 16 }.holds(&[Fact::Address(ARENA)]), Ok(true));

    // A null clause over anything that is not an address faults rather than
    // guessing — including over a `Bool`, which is the confusion in the other
    // direction: `Fact::Bool(false)` is not an absent pointer.
    for other in [Fact::Int(0), Fact::Int(1), Fact::Bool(false), Fact::Bool(true), Fact::Opaque] {
        assert_eq!(
            has_table.holds(&[other]),
            Err(device::Fault::Kind { operand: 0, wanted: "an address" }),
            "a null clause read a {other:?}"
        );
    }
    assert_eq!(has_table.holds(&[]), Err(device::Fault::Range { operand: 0, arity: 0 }));

    // The closure the other way: an address is not a flag either, so the two
    // selectors cannot be spelled with each other's term. Both directions
    // fault, which is what keeps `Fact` from needing a `Present(bool)`
    // variant — see `Term::Present`'s doc for why that variant would make
    // "the hidden size is present" a well-formed clause.
    assert_eq!(
        Term::Is { operand: 0, value: true }.holds(&[Fact::Address(0)]),
        Err(device::Fault::Kind { operand: 0, wanted: "a flag" }),
        "a null pointer must not read as `false`"
    );
    assert_eq!(
        Term::Multiple { operand: 0, of: 1 }.holds(&[Fact::Address(ARENA)]),
        Err(device::Fault::Kind { operand: 0, wanted: "an integer" }),
        "dividing a pointer must be a fault and not the answer `true`"
    );
}

/// The two ways a null clause can be written against the wrong operand.
///
/// Both are refused with no device, and both are refusals of an arm that
/// COMPILES and looks like a decision:
///
/// (a) over a scalar, the clause can never once be true, because a scalar
///     binds `Fact::Int`, `Fact::Bool` or `Fact::Opaque` and every one of
///     them faults. The arm is dead code that reads as a branch.
/// (b) over a pointer the row does not declare nullable, the clause is true
///     for every fire that reaches it — the binder refuses a null there — so
///     one of the two instantiations compiles, ships, and never runs. That
///     is not a launch bug; it is a table asserting a choice it does not
///     make, and §21.13's whole complaint about the previous round.
#[test]
fn agrees_refuses_a_null_clause_over_the_wrong_operand() {
    let real = device::specialisation(NULL_BASE).expect("specialised");
    let one = |when: &'static [Term]| {
        let arms: &'static [Arm] = Box::leak(Box::new([Arm {
            name: "a control",
            when,
            row: real.arms[0].row,
            take: real.arms[0].take,
            because: "a control",
        }]));
        Specialisation { base: NULL_BASE, arms }
    };

    // (a) `head_dim` is operand 17 and is an `I32`.
    let why = one(&[Term::Present { operand: 17, value: true }])
        .agrees()
        .expect_err("a null clause over an i32 must be refused");
    assert!(
        why.contains("tests `head_dim` for null") && why.contains("never supplies one"),
        "{why}"
    );

    // (b) `packed` is operand 0, a `Buf`, and is NOT nullable.
    let why = one(&[Term::Present { operand: 0, value: true }])
        .agrees()
        .expect_err("a null clause over a non-nullable pointer must be refused");
    assert!(
        why.contains("tests `packed` for null") && why.contains("does not declare it nullable"),
        "{why}"
    );

    // And the shipped pair, which states the same term over the operand that
    // IS nullable, is accepted — so the two refusals above are about the
    // operand and not about the term.
    real.agrees().expect("the shipped pair states `rope_table`, which is `F32s | null`");
    assert_eq!(real.arms.len(), 2);
    assert_eq!(real.arms[0].when, &[Term::Present { operand: TABLE, value: true }]);
    assert_eq!(real.arms[1].when, &[Term::Present { operand: TABLE, value: false }]);
}

/// `flags_are_covered` finds nothing over the shipped pair, and would find
/// something over a pair that baked the pointer away.
///
/// The exemption is real and it is narrow. `qkv_fused.cu:64` and `:77` pass
/// `rope_table` to `<BLOCK, true>` and `<BLOCK, false>` alike, so both arms
/// FORWARD operand 7, the retain in `flags_are_covered` drops it, and there is
/// nothing to enumerate. The hazard that check exists for — a base row binding
/// one cell more than the instantiation declares, which `cuLaunchKernel`
/// accepts and never reports — cannot arise when nothing is dropped.
///
/// The second half is why the enumeration was generalised to `Term::Present`
/// anyway: a kernel that baked its pointer's PRESENCE into the template and
/// stopped taking the pointer would have `write_kv<HND_LAYOUT>`'s exact shape
/// with an address in place of a flag, and would need the coverage check for
/// the identical reason. Built here rather than argued, because the argument
/// is the kind that is easy to get backwards.
#[test]
fn a_null_clause_that_reaches_no_kernel_must_be_covered_both_ways() {
    use kernels::{LaunchRule, kernel, operands};
    use kernels_cuda_new::device::DeviceKernel;

    let real = device::specialisation(NULL_BASE).expect("specialised");

    // The shipped pair: both arms forward operand 7, so the retain empties
    // the list before the enumeration runs.
    assert!(
        real.arms.iter().all(|arm| arm.take.contains(&Take::From(TABLE))),
        "both instantiations take `rope_table`; if that stops being true the exemption below \
         stops holding and this pair needs the same coverage `write_kv`'s does"
    );
    real.agrees().expect("nothing to enumerate");

    // A doctored row that BAKED the pointer's presence and stopped taking it:
    // `attn::qkv_decode_qk_norm_rope_write_kv` minus `rope_table`, twenty-one
    // operands. This is `write_kv<HND_LAYOUT>`'s shape with an address in
    // place of a flag, and it is the shape the enumeration was generalised
    // for.
    let baked = kernel!(baked "attn::qkv_decode_qk_norm_rope_write_kv",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::RowsPackedHeadsNarrow,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_q_heads: I32, num_kv_heads: I32, head_dim: I32, page_size: I32,
            hnd_layout: Bool, theta: F32, eps: F32,
        ]);
    let row: &'static DeviceKernel = Box::leak(Box::new(DeviceKernel {
        sig: Box::leak(Box::new(baked)),
        template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv",
        elem: "device::i32(128), true",
    }));

    // Twenty-one takes: everything but operand 7.
    let drops: &'static [Take] = Box::leak(
        (0..22usize)
            .filter(|i| *i != TABLE)
            .map(Take::From)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let doctored = |when: &'static [&'static [Term]]| {
        let arms: &'static [Arm] = Box::leak(
            when.iter()
                .enumerate()
                .map(|(i, when)| Arm {
                    name: if i == 0 { "rope" } else { "norope" },
                    when,
                    row,
                    take: drops,
                    because: "a control",
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        Specialisation { base: NULL_BASE, arms }
    };

    // One arm only: the `false` assignment falls through to a base that binds
    // twenty-two cells for a kernel declaring twenty-one.
    let why = doctored(&[&[Term::Present { operand: TABLE, value: true }]])
        .agrees()
        .expect_err("a dropped pointer covered on one value must be refused");
    assert!(
        why.contains("selects on a flag no arm forwards")
            && why.contains("`rope_table` = false")
            && why.contains("22 cells"),
        "{why}"
    );

    // Both arms: covered, and accepted. The check is a coverage check and not
    // a ban on dropping.
    doctored(&[
        &[Term::Present { operand: TABLE, value: true }],
        &[Term::Present { operand: TABLE, value: false }],
    ])
    .agrees()
    .expect("both values stated");

    // And an arm that also tests something the enumeration cannot discharge
    // does not count towards coverage — the same conservatism the flag case
    // has. `q_weight` is operand 4 and is an address, so the alignment is
    // well-formed and simply unknowable here.
    let why = doctored(&[
        &[Term::Present { operand: TABLE, value: true }],
        &[
            Term::Present { operand: TABLE, value: false },
            Term::Aligned { operand: 4, bytes: 16 },
        ],
    ])
    .agrees()
    .expect_err("an arm with an undischargeable clause cannot cover a value");
    assert!(why.contains("`rope_table` = false"), "{why}");
}

/// The pair chooses on the pointer, at fire-time facts, and the near-miss
/// chooses wrong on the same facts.
///
/// This is the mutation control for `Term::Present` itself: same base, same
/// two rows, same `take`, one clause changed to the nearest thing that
/// existed before — and the arm it selects for a fire with NO table is the
/// one that dereferences the table.
#[test]
fn the_qkv_decode_pair_chooses_on_the_pointer_and_the_near_miss_does_not() {
    let real = device::specialisation(NULL_BASE).expect("specialised");

    // Twenty-two facts in the base's operand order. Only operand 7 moves.
    let facts = |table: u64| {
        let mut facts = [Fact::Opaque; 22];
        for (i, f) in facts.iter_mut().enumerate() {
            *f = match i {
                17 => Fact::Int(128),
                19 => Fact::Bool(false),
                20 | 21 => Fact::Opaque,
                _ => Fact::Address(ARENA),
            };
        }
        facts[TABLE] = Fact::Address(table);
        facts
    };

    let with = real.choose(&facts(ARENA)).expect("no fault").expect("an arm");
    let without = real.choose(&facts(0)).expect("no fault").expect("an arm");
    assert_eq!(with.name, "rope");
    assert_eq!(without.name, "norope");
    assert_ne!(with.row.elem, without.row.elem, "the two arms name different instantiations");
    assert!(with.row.elem.ends_with("true"), "{}", with.row.elem);
    assert!(without.row.elem.ends_with("false"), "{}", without.row.elem);

    // The near-miss: `Term::Aligned` in place of `Term::Present`, which is
    // what a reader without this term would reach for. It selects `rope` for
    // BOTH fires, and the second has no table.
    let near: &'static [Arm] = Box::leak(Box::new([
        Arm {
            name: real.arms[0].name,
            when: &[Term::Aligned { operand: TABLE, bytes: 16 }],
            row: real.arms[0].row,
            take: real.arms[0].take,
            because: real.arms[0].because,
        },
        Arm {
            name: real.arms[1].name,
            when: real.arms[1].when,
            row: real.arms[1].row,
            take: real.arms[1].take,
            because: real.arms[1].because,
        },
    ]));
    let near = Specialisation { base: NULL_BASE, arms: near };
    let picked = near.choose(&facts(0)).expect("no fault").expect("an arm");
    assert_eq!(
        picked.name, "rope",
        "the alignment clause was supposed to pick the table arm for a fire with no table — \
         if it stops doing so, `Term::Present`'s justification has changed"
    );
    println!(
        "null-clause control: `Present` -> {} / {} for table={{addr, null}}; \
         `Aligned {{ bytes: 16 }}` -> {} / {} — the second is the constant function",
        with.name,
        without.name,
        near.choose(&facts(ARENA)).unwrap().unwrap().name,
        picked.name,
    );
}

// ---------------------------------------------------------------------------
// 1, 3 and 4 on a device
// ---------------------------------------------------------------------------

#[cfg(feature = "_cuda")]
mod fires {
    use super::{BASE, VARIANT};
    use cudarc::driver::sys as dr;
    use kernels_cuda_new::runtime::{self, ArgValue, Dims, Stream, cache};
    use std::ffi::c_void;

    /// `sm_XY` for the current device, or a stated reason there is none.
    fn arch_or_skip(what: &str) -> Option<&'static str> {
        match cache::arch() {
            Some(arch) => match cache::bind_context() {
                Ok(()) => Some(arch),
                Err(why) => {
                    eprintln!("SKIP {what}: no usable context ({why})");
                    None
                }
            },
            None => {
                eprintln!("SKIP {what}: no CUDA device is current");
                None
            }
        }
    }

    /// A device allocation, freed when it goes out of scope.
    struct Buffer {
        ptr: u64,
        bytes: usize,
    }

    impl Buffer {
        fn new(bytes: usize) -> Self {
            let mut ptr = 0u64;
            // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
            let code = unsafe { dr::cuMemAlloc_v2(&raw mut ptr, bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "allocating {bytes} bytes");
            assert_eq!(ptr % 256, 0, "cuMemAlloc is documented 256-byte aligned");
            Self { ptr, bytes }
        }

        fn upload(&self, from: &[u16]) {
            assert_eq!(from.len() * 2, self.bytes);
            // SAFETY: the allocation is `bytes` long and `from` is exactly that.
            let code = unsafe { dr::cuMemcpyHtoD_v2(self.ptr, from.as_ptr().cast(), self.bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "upload");
        }

        fn download(&self) -> Vec<u16> {
            let mut out = vec![0u16; self.bytes / 2];
            // SAFETY: same allocation, same length.
            let code =
                unsafe { dr::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, self.bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "download");
            out
        }

        fn fill(&self, byte: u8) {
            // SAFETY: the allocation is `bytes` long.
            let code = unsafe { dr::cuMemsetD8_v2(self.ptr, byte, self.bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "memset");
        }

        /// The address `offset` BYTES into this allocation, as an argument.
        fn at(&self, offset: u64) -> ArgValue {
            ArgValue::Ptr((self.ptr + offset) as *mut c_void)
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            // SAFETY: the handle came from `cuMemAlloc_v2` and is freed once.
            unsafe { dr::cuMemFree_v2(self.ptr) };
        }
    }

    fn synchronise(what: &str) {
        // SAFETY: no arguments, and the context is bound.
        let code = unsafe { dr::cuCtxSynchronize() };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "{what}");
    }

    /// bf16 test data with a controlled exponent range.
    ///
    /// The exponent is held in `[-4, +3]` and the mantissa is the noise, so
    /// the sum of squares over a few thousand of them stays far from both
    /// ends of fp32 and the reduction ORDER is the only thing that can move
    /// the answer. Data that straddled the exponent range would let a
    /// cancellation stand in for a reassociation, which is the measurement
    /// this file is trying not to make.
    fn sample(n: usize, seed: u64) -> Vec<u16> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let bits = (state >> 33) as u32;
                let sign = (bits & 1) << 15;
                let exponent = 123 + (bits >> 1) % 8;
                let mantissa = (bits >> 8) & 0x7f;
                u16::try_from(sign | (exponent << 7) | mantissa).expect("16 bits")
            })
            .collect()
    }

    /// The base row's argument list.
    fn values(x: ArgValue, w: ArgValue, y: ArgValue, hidden: i32, xs: i32, ys: i32) -> [ArgValue; 7] {
        [x, w, y, ArgValue::I32(hidden), ArgValue::I32(xs), ArgValue::I32(ys), ArgValue::F32(1e-6)]
    }

    fn dims(rows: u32, width: u32) -> Dims {
        Dims { rows, width, in_width: width, ..Default::default() }
    }

    /// How many bf16 words differ, and by how many representable steps.
    ///
    /// Counting ULPs on the STORED word rather than a relative error, because
    /// the question this file asks is whether the two kernels produce the
    /// same bytes — and if they do not, the useful number is how far apart
    /// the two bf16 values are in the format they were rounded into, not what
    /// that distance is as a fraction.
    fn compare(a: &[u16], b: &[u16]) -> (usize, u32) {
        let mut differ = 0;
        let mut worst = 0;
        for (&l, &r) in a.iter().zip(b) {
            if l != r {
                differ += 1;
                // Both are finite and same-signed by construction here, so
                // the ordered distance is the difference of the words.
                worst = worst.max(u32::from(l.abs_diff(r)));
            }
        }
        (differ, worst)
    }

    /// **Both arms fire, and they agree.**
    ///
    /// The vectorised kernel against the scalar one, on the same `x` and the
    /// same `weight`, at the same block width, through the same symbol —
    /// with the only difference being a two-byte offset on the OUTPUT
    /// pointer, which is a value the arithmetic never reads and the predicate
    /// always does. So one fire takes the arm and the other does not, and
    /// nothing else about the two is different.
    ///
    /// Measured on an L40S, NVRTC 13.0, `--fmad=false`.
    #[test]
    fn both_arms_fire_and_agree() {
        let Some(_) = arch_or_skip("both_arms_fire_and_agree") else { return };

        println!("\n{:>7} {:>5} {:>10} {:>9} {:>7}", "hidden", "rows", "values", "differing", "ulps");
        println!("{}", "-".repeat(43));
        let mut total_differ = 0usize;
        let mut total_values = 0usize;
        let mut worst_ulp = 0u32;

        for &(rows, hidden) in &[(1usize, 2048usize), (1, 2816), (1, 4096), (3, 5376), (7, 2048)] {
            let n = rows * hidden;
            let x = Buffer::new(n * 2);
            let w = Buffer::new(hidden * 2);
            // Two extra bytes so the scalar arm can be reached by offsetting
            // the output by one element without leaving the allocation.
            let vec_out = Buffer::new(n * 2 + 2);
            let scalar_out = Buffer::new(n * 2 + 2);
            x.upload(&sample(n, 0x5eed_0001 + hidden as u64));
            w.upload(&sample(hidden, 0xfeed_0002));
            vec_out.fill(0);
            scalar_out.fill(0);

            let (h, s) = (i32::try_from(hidden).unwrap(), i32::try_from(hidden).unwrap());
            let aligned = values(x.at(0), w.at(0), vec_out.at(0), h, s, s);
            let offset = values(x.at(0), w.at(0), scalar_out.at(2), h, s, s);

            assert_eq!(
                runtime::selects(BASE, &aligned).expect("no fault").map(|arm| arm.name),
                Some("vec8"),
                "aligned rows must take the vectorised arm"
            );
            assert_eq!(
                runtime::selects(BASE, &offset).expect("no fault").map(|arm| arm.name),
                None,
                "an output two bytes into an allocation is not 16-byte aligned"
            );

            let d = dims(u32::try_from(rows).unwrap(), u32::try_from(hidden).unwrap());
            // SAFETY: both pointers address live device memory of the extent
            // the row states, and the null stream is always live.
            unsafe { runtime::fire(BASE, d, &aligned, Stream::NULL) }.expect("the vectorised arm");
            // SAFETY: as above.
            unsafe { runtime::fire(BASE, d, &offset, Stream::NULL) }.expect("the scalar arm");
            synchronise("both arms");

            let got = vec_out.download();
            let want = scalar_out.download();
            // The scalar output starts one element in.
            let (differ, ulps) = compare(&got[..n], &want[1..=n]);
            println!("{hidden:>7} {rows:>5} {n:>10} {differ:>9} {ulps:>7}");
            total_differ += differ;
            total_values += n;
            worst_ulp = worst_ulp.max(ulps);

            // The output is not all zero -- a kernel that did nothing would
            // agree with another kernel that did nothing.
            assert!(got[..n].iter().any(|&v| v != 0), "the vectorised arm wrote nothing");
        }

        println!("{}", "-".repeat(43));
        println!(
            "{total_differ} of {total_values} bf16 values differ, worst {worst_ulp} ulp\n"
        );
        assert_eq!(
            total_differ, 0,
            "the two kernels reassociate their reduction and it SHOWS in the bf16 output: \
             {total_differ} of {total_values} values differ by up to {worst_ulp} ulp. That is a \
             finding, not a rounding detail -- a specialisation that changes the answer is a \
             second kernel wearing the first's name."
        );
    }

    /// **The negative control: the wrong arm, fired on purpose.**
    ///
    /// `hidden` is 4 095 and everything is aligned, so `rmsnorm_vec8_ok` says
    /// no and the specialisation agrees. The variant row is then fired
    /// DIRECTLY, by its own symbol, which is exactly what a `Select` that had
    /// dropped the `hidden % 8` clause would have done — and the failure it
    /// produces is the one this whole file exists to rule out: not a crash,
    /// not a driver error, but 4 095 finite plausible numbers, every one of
    /// them wrong.
    ///
    /// `rmsnorm_vec8` reads `nvec = hidden / 8`, so it sums 4 088 of the 4 095
    /// squares and still divides by 4 095. **What that measured is the point
    /// of the whole file: 7 of 4 095 values moved.** Not 4 095 — seven, a
    /// sixth of a per cent, and they are the tail the kernel never wrote.
    /// The 4 088 it did write are BIT-IDENTICAL to the scalar arm's, because
    /// dropping seven of 4 095 squares moves the norm by under a tenth of a
    /// per cent and bf16 has eight bits of mantissa to notice it with.
    ///
    /// So a wrong choice here does not look wrong. It looks like a tensor
    /// with a short zeroed tail, feeding sixty more layers, and no tolerance
    /// anyone would write for a reassociated reduction would flag it. That is
    /// why the bar in [`both_arms_fire_and_agree`] is zero differing values
    /// and not a relative error: at this width, a relative-error check with
    /// any tolerance loose enough to admit reassociation would also admit
    /// this.
    #[test]
    fn the_wrong_arm_is_caught() {
        let Some(_) = arch_or_skip("the_wrong_arm_is_caught") else { return };

        let hidden = 4095usize;
        let x = Buffer::new(hidden * 2);
        let w = Buffer::new(hidden * 2);
        let right = Buffer::new(hidden * 2);
        let wrong = Buffer::new(hidden * 2);
        x.upload(&sample(hidden, 0xbad_0000));
        w.upload(&sample(hidden, 0xbad_0001));
        right.fill(0);
        wrong.fill(0);

        let h = i32::try_from(hidden).unwrap();
        let correct = values(x.at(0), w.at(0), right.at(0), h, h, h);
        assert_eq!(
            runtime::selects(BASE, &correct).expect("no fault").map(|arm| arm.name),
            None,
            "hidden 4095 is not a whole number of vec8 vectors"
        );
        let d = dims(1, u32::try_from(hidden).unwrap());
        // SAFETY: live buffers of the stated extent.
        unsafe { runtime::fire(BASE, d, &correct, Stream::NULL) }.expect("the scalar arm");

        // The variant's own eight arguments, with the null `rmsnorm.cu`
        // passes for `y_fp16`. Firing it here is the mistake being staged.
        let forced = [
            x.at(0),
            w.at(0),
            wrong.at(0),
            ArgValue::Ptr(std::ptr::null_mut()),
            ArgValue::I32(h),
            ArgValue::I32(h),
            ArgValue::I32(h),
            ArgValue::F32(1e-6),
        ];
        // SAFETY: as above; the null is the operand the row declares nullable.
        unsafe { runtime::fire(VARIANT, d, &forced, Stream::NULL) }.expect("the forced arm ran");
        synchronise("the forced arm");

        let want = right.download();
        let got = wrong.download();
        let (differ, ulps) = compare(&got, &want);
        let tail = hidden - hidden % 8;
        let (body_differ, body_ulps) = compare(&got[..tail], &want[..tail]);
        println!(
            "negative control: forcing the vectorised arm at hidden 4095 moved {differ} of \
             {hidden} values, by up to {ulps} ulp -- of which {body_differ} are in the \
             {tail} it wrote (worst {body_ulps} ulp) and {} are the tail it skipped",
            differ - body_differ
        );
        assert!(
            differ > 0,
            "the wrong kernel produced the right answer, so the parity check proves nothing"
        );
        // The tail is the part it never touched: seven elements of the zeroed
        // buffer, which the scalar kernel wrote real values into.
        let tail = hidden - hidden % 8;
        assert!(
            got[tail..].iter().all(|&v| v == 0),
            "the vectorised kernel wrote past `nvec * 8`, which it must not"
        );
        assert!(
            want[tail..].iter().any(|&v| v != 0),
            "the scalar kernel left the tail unwritten too, so the control is not the control"
        );
    }

    /// The refusals a specialised row makes are the base row's refusals.
    ///
    /// Specialisation may change which kernel runs; it may not change which
    /// fires are legal. A wrong argument list is refused with the same error,
    /// naming the same symbol, whether the predicate would have held or not —
    /// which is what keeps a row a contract rather than a contract plus a
    /// decision.
    #[test]
    fn specialisation_does_not_change_which_fires_are_refused() {
        let Some(_) = arch_or_skip("specialisation_does_not_change_which_fires_are_refused")
        else {
            return;
        };
        let hidden = 4096usize;
        let buffer = Buffer::new(hidden * 2);
        let d = dims(1, u32::try_from(hidden).unwrap());
        let h = i32::try_from(hidden).unwrap();

        // Aligned -- the predicate would hold -- and one operand short.
        let short = [buffer.at(0), buffer.at(0), buffer.at(0), ArgValue::I32(h), ArgValue::I32(h)];
        // SAFETY: nothing launches; the list is refused before the driver is
        // reached.
        let refusal = unsafe { runtime::fire(BASE, d, &short, Stream::NULL) };
        let why = refusal.expect_err("a short list is refused").to_string();
        assert!(why.contains(BASE), "the refusal names the row the caller asked for: {why}");
        assert!(why.contains('5') && why.contains('7'), "{why}");

        // The same shape with a float where a stride belongs.
        let mistyped = [
            buffer.at(0),
            buffer.at(0),
            buffer.at(0),
            ArgValue::I32(h),
            ArgValue::F32(4096.0),
            ArgValue::I32(h),
            ArgValue::F32(1e-6),
        ];
        // SAFETY: as above.
        let refusal = unsafe { runtime::fire(BASE, d, &mistyped, Stream::NULL) };
        let why = refusal.expect_err("a mistyped list is refused").to_string();
        assert!(why.contains("x_row_stride"), "the refusal names the operand: {why}");
    }

    /// What the choice costs per fire, and what the variant costs per compile.
    ///
    /// Both halves of the bargain, because a specialisation is only worth
    /// having if the decision is cheaper than the kernel it saves. The
    /// per-fire number is the one that matters: it is paid once per kernel
    /// per layer per token, which at 60 layers and 100 tokens a second is
    /// 6 000 evaluations a second and has to disappear into the noise of a
    /// `cuLaunchKernel`.
    #[test]
    fn the_choice_is_cheaper_than_the_launch() {
        let Some(arch) = arch_or_skip("the_choice_is_cheaper_than_the_launch") else { return };
        use kernels_cuda_new::runtime::nvrtc;
        use kernels_cuda_new::unit;
        use std::time::Instant;

        let x = Buffer::new(1 << 22);
        let w = Buffer::new(1 << 15);
        let out = Buffer::new((1 << 22) + 2);
        x.upload(&sample(1 << 21, 0x711));
        w.upload(&sample(1 << 14, 0x712));
        let aligned = values(x.at(0), w.at(0), out.at(0), 4096, 4096, 4096);

        // The predicate alone, with no launch under it.
        const ROUNDS: u32 = 100_000;
        let started = Instant::now();
        let mut took = 0u32;
        for _ in 0..ROUNDS {
            if runtime::selects(BASE, &aligned).expect("no fault").is_some() {
                took += 1;
            }
        }
        let per_choice = started.elapsed().as_secs_f64() * 1e9 / f64::from(ROUNDS);
        assert_eq!(took, ROUNDS);
        println!("\nthe choice: {per_choice:.0} ns per fire, over {ROUNDS} evaluations");

        // The two arms, end to end, through the same symbol, over the shapes
        // a decode step and a prefill step actually present. The only
        // difference between the two lists is two bytes on the output
        // pointer, so the arms do identical arithmetic on identical bytes and
        // the difference in the column is the kernel and nothing else.
        println!("\n  rows  hidden   scalar us  vector us   ratio");
        println!("  ------------------------------------------------");
        for (rows, hidden) in [
            (1u32, 2048usize),
            (1, 4096),
            (1, 8192),
            (8, 4096),
            (64, 4096),
            (512, 4096),
            (1024, 2048),
        ] {
            let h = i32::try_from(hidden).unwrap();
            let d = dims(rows, u32::try_from(hidden).unwrap());
            let lists = [
                ("scalar", values(x.at(0), w.at(0), out.at(2), h, h, h)),
                ("vector", values(x.at(0), w.at(0), out.at(0), h, h, h)),
            ];
            let mut per_arm = [0f64; 2];
            for (slot, (what, list)) in lists.iter().enumerate() {
                assert_eq!(
                    runtime::selects(BASE, list).expect("no fault").map(|arm| arm.name).is_some(),
                    *what == "vector",
                    "the shape did not take the arm it was built for"
                );
                for _ in 0..50 {
                    // SAFETY: live buffers of the stated extent.
                    unsafe { runtime::fire(BASE, d, list, Stream::NULL) }.expect("warm");
                }
                synchronise("warm");
                // Minimum of five batches, not the mean of one. At decode's
                // shapes the two arms are within a few hundred nanoseconds
                // and a single batch put the ratio either side of 1.0 on
                // consecutive runs -- the mean was measuring the machine's
                // other tenants, and the minimum is the number that is about
                // the kernel.
                const LAUNCHES: u32 = 300;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let started = Instant::now();
                    for _ in 0..LAUNCHES {
                        // SAFETY: as above.
                        unsafe { runtime::fire(BASE, d, list, Stream::NULL) }.expect("timed");
                    }
                    synchronise("timed");
                    best = best.min(started.elapsed().as_secs_f64() * 1e6 / f64::from(LAUNCHES));
                }
                per_arm[slot] = best;
            }
            let [scalar, vector] = per_arm;
            println!(
                "  {rows:>4}  {hidden:>6}  {scalar:>10.2} {vector:>10.2}  {:>6.2}x",
                scalar / vector
            );
        }
        println!("  ------------------------------------------------");

        // The compile, with the variant and without it.
        let unit = unit::UNITS
            .iter()
            .find(|u| u.name == "norm/rmsnorm")
            .expect("the unit is in the table");
        let all: Vec<&kernels_cuda_new::device::DeviceKernel> = unit.rows.iter().collect();
        let scalar_only: Vec<&kernels_cuda_new::device::DeviceKernel> =
            unit.rows.iter().filter(|row| row.sig.symbol != VARIANT).collect();
        assert_eq!(scalar_only.len() + 1, all.len(), "exactly one row is the variant");
        let mut with = f64::MAX;
        let mut without = f64::MAX;
        for _ in 0..5 {
            let started = Instant::now();
            nvrtc::compile_rows(unit, arch, &all).expect("the unit compiles");
            with = with.min(started.elapsed().as_secs_f64() * 1e3);
            let started = Instant::now();
            nvrtc::compile_rows(unit, arch, &scalar_only).expect("the four scalar rows compile");
            without = without.min(started.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "norm/rmsnorm: {without:.1} ms for four rows, {with:.1} ms for five -- the variant \
             costs {:.1} ms ({:.0}%) on a compile that was already happening, against a second \
             {without:.1} ms compile deferred to whenever the data first asks for it\n",
            with - without,
            (with - without) / without * 100.0
        );

        assert!(
            per_choice < 1000.0,
            "the choice took {per_choice:.0} ns, which is not free next to a launch"
        );
    }
}


// ---------------------------------------------------------------------------
// 5 on a device: a flag, both arms of a real `template <bool>`, and the
// negative control
// ---------------------------------------------------------------------------

/// **`attn::device::write_kv<HND_LAYOUT>`, fired both ways, chosen by a
/// `bool`.**
///
/// The nine blocked rows turn on a flag selecting a `template <bool>` arm, and
/// everything above this line establishes that with no device: the fact, the
/// clause, the type check and the coverage rule. None of it says the ARM THE
/// PREDICATE NAMES IS THE KERNEL THAT RUNS, and that is the only claim a
/// wrong specialisation would falsify. So this compiles the real header,
/// resolves both instantiations, and fires each one on data whose correct
/// output is computable by hand.
///
/// # Why the rows are here and not in `families/attn.rs`
///
/// They cannot be there yet — the family modules belong to another change —
/// so this file states the three rows a real `attn::write_kv_bf16` needs and
/// drives them through the same four steps [`kernels_cuda_new::runtime::fire`]
/// does: facts out of [`kernels_cuda_new::runtime::ArgValue::fact`], an arm
/// out of [`Specialisation::choose`], a reshape through [`Take`], and
/// `Args::bind` + `KernelModule::fire`. Every one of those is the shipped
/// function; what this file supplies is the row the table does not have.
///
/// # Why `write_kv` is the right kernel to prove it on
///
/// It is a pure SCATTER — sixteen bf16 words per token, copied, with the
/// layout deciding only the destination index. No arithmetic, so there is no
/// reassociation to hide behind and no tolerance to argue about: the right
/// answer is a permutation of the input and the bar is bit equality over the
/// whole arena, including the pages neither kernel should touch.
#[cfg(feature = "_cuda")]
mod flag_arms {
    use cudarc::driver::sys as dr;
    use kernels::{KernelSig, LaunchRule, kernel, operands};
    use kernels_cuda_new::device::{Arm, DeviceKernel, Fact, Specialisation, Take, Term};
    use kernels_cuda_new::runtime::{self, ArgValue, Dims, KernelModule, Stream, cache, nvrtc};
    use kernels_cuda_new::unit;
    use std::ffi::c_void;

    /// The three rows `families/attn.rs` must write, spelled here because it
    /// cannot be edited by this change.
    ///
    /// `SIGS[0]` is the CONTRACT — `write_kv`'s fifteen parameters and
    /// `hnd_layout` as a sixteenth operand. The flag is the launcher's
    /// argument (`kv_paged.cu:660`) and no kernel's, which is exactly why it
    /// has to be an operand of the base and of nothing else: a fire has to be
    /// able to HAND it, and no instantiation can be handed it.
    ///
    /// `SIGS[1]` and `SIGS[2]` are the kernel's own fifteen, twice, under the
    /// `#hnd` and `#nhd` suffixes a variant row carries.
    #[rustfmt::skip]
    static SIGS: [KernelSig; 3] = [
        kernel!(contract "attn::write_kv_bf16",
            file = Some("attn/kv_paged.cuh"),
            launch = LaunchRule::PerRow,
            operands = operands![
                k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
                qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
                kv_last_page_lens: U32s, row_valid: U8s | null, win: U32s | null,
                r: I32, page_size: I32, h_kv: I32, d: I32, first_token: I32,
                hnd_layout: Bool,
            ]),
        kernel!(hnd "attn::write_kv_bf16#hnd",
            file = Some("attn/kv_paged.cuh"),
            launch = LaunchRule::PerRow,
            operands = operands![
                k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
                qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
                kv_last_page_lens: U32s, row_valid: U8s | null, win: U32s | null,
                r: I32, page_size: I32, h_kv: I32, d: I32, first_token: I32,
            ]),
        kernel!(nhd "attn::write_kv_bf16#nhd",
            file = Some("attn/kv_paged.cuh"),
            launch = LaunchRule::PerRow,
            operands = operands![
                k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
                qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
                kv_last_page_lens: U32s, row_valid: U8s | null, win: U32s | null,
                r: I32, page_size: I32, h_kv: I32, d: I32, first_token: I32,
            ]),
    ];

    /// One template, two arguments — which is the whole of what a
    /// specialisation may be.
    ///
    /// Three rows and not two: the CONTRACT needs one as well, because
    /// [`Specialisation::agrees`] resolves its base through
    /// [`kernels_cuda_new::unit::Unit::row`] and `runtime::fire` dispatches
    /// on it. It names the same template at one of the two arguments — `#nhd`
    /// is arbitrary; the base row is unreachable once
    /// [`Specialisation::agrees`] has proved the arms total, and
    /// [`both_flag_arms_fire_and_land_where_the_flag_says`] confirms NVRTC
    /// accepts the repeated name expression rather than rejecting it as a
    /// duplicate.
    ///
    /// `elem` is `device::true_type::value` and not `true`, and that is a
    /// measured constraint rather than a style: `write_kv` takes ONE template
    /// parameter and it is the flag, so the flag lands in the slot
    /// [`DeviceKernel::instantiation`] prefixes with
    /// `::pie_cuda_driver::kernels::`, and `::pie_cuda_driver::kernels::true`
    /// is `expected an identifier`. `examples/argform_probe.rs`'s last three
    /// rows are that measurement; the prelude ships the two tag types at
    /// `pie_device.cuh:485` for exactly this.
    static ROWS: [DeviceKernel; 3] = [
        DeviceKernel {
            sig: &SIGS[1],
            template_path: "attn::device::write_kv",
            elem: "device::true_type::value",
        },
        DeviceKernel {
            sig: &SIGS[2],
            template_path: "attn::device::write_kv",
            elem: "device::false_type::value",
        },
        DeviceKernel {
            sig: &SIGS[0],
            template_path: "attn::device::write_kv",
            elem: "device::false_type::value",
        },
    ];

    /// The base's first fifteen operands, in the kernel's order. Operand 15 —
    /// the flag — is forwarded by neither arm, which is what makes the
    /// coverage rule apply and both arms mandatory.
    static TAKE: [Take; 15] = [
        Take::From(0),
        Take::From(1),
        Take::From(2),
        Take::From(3),
        Take::From(4),
        Take::From(5),
        Take::From(6),
        Take::From(7),
        Take::From(8),
        Take::From(9),
        Take::From(10),
        Take::From(11),
        Take::From(12),
        Take::From(13),
        Take::From(14),
    ];

    /// `kv_paged.cu:51`, as data.
    static ARMS: [Arm; 2] = [
        Arm {
            name: "hnd",
            when: &[Term::Is { operand: 15, value: true }],
            row: &ROWS[0],
            take: &TAKE,
            because: "kv_paged.cu:51 `if (hnd_layout)` -> write_kv<true>",
        },
        Arm {
            name: "nhd",
            when: &[Term::Is { operand: 15, value: false }],
            row: &ROWS[1],
            take: &TAKE,
            because: "kv_paged.cu:51 `else` -> write_kv<false>",
        },
    ];

    static SPEC: Specialisation = Specialisation { base: "attn::write_kv_bf16", arms: &ARMS };

    /// A KV cache small enough that the correct output is a table.
    #[derive(Clone, Copy)]
    struct Shape {
        tokens: usize,
        page_size: usize,
        h_kv: usize,
        d: usize,
        /// The request's pages, in order — deliberately not `0, 1, 2`, so a
        /// kernel that ignored `kv_page_indices` would be caught.
        pages: &'static [u32],
        /// How many pages the arena holds. Larger than `pages`, so the ones
        /// nothing should write are checked to be untouched.
        arena: usize,
    }

    impl Shape {
        const fn row(self) -> usize {
            self.h_kv * self.d
        }
        const fn cells(self) -> usize {
            self.arena * self.page_size * self.row()
        }
    }

    /// The five shapes, spanning a decode step and a prefill.
    ///
    /// `h_kv` is never 1: at one KV head the two layouts compute the SAME
    /// destination for every element, so a shape with `h_kv = 1` would pass
    /// the negative control by accident.
    static SHAPES: [Shape; 5] = [
        Shape { tokens: 6, page_size: 4, h_kv: 2, d: 8, pages: &[3, 1], arena: 4 },
        Shape { tokens: 1, page_size: 4, h_kv: 8, d: 64, pages: &[1], arena: 2 },
        Shape { tokens: 17, page_size: 8, h_kv: 4, d: 16, pages: &[2, 0, 1], arena: 3 },
        Shape { tokens: 32, page_size: 16, h_kv: 8, d: 128, pages: &[1, 0], arena: 3 },
        Shape { tokens: 5, page_size: 2, h_kv: 4, d: 4, pages: &[2, 0, 4], arena: 5 },
    ];

    /// Where `write_kv` puts token `t`'s element `i`, transliterated from
    /// `kv_paged.cuh:186-196`.
    ///
    /// A HAND copy of the C++, in the C++'s order, for the same reason
    /// [`super::rmsnorm_vec8_ok`] is one: an oracle derived from the thing
    /// under test compares a value to itself. Every term the kernel computes
    /// — `pre_kv_len`, `abs_kv_pos`, `page_in_req` — is reproduced, not
    /// short-circuited, so the shapes below exercise the same arithmetic the
    /// device does.
    fn scatter(hnd: bool, s: &Shape, t: usize, i: usize) -> usize {
        let num_pages_r = s.pages.len();
        let last = s.tokens - (num_pages_r - 1) * s.page_size;
        let total_kv_after = (num_pages_r - 1) * s.page_size + last;
        let pre_kv_len = total_kv_after - s.tokens;
        let abs_kv_pos = pre_kv_len + t;
        let page_in_req = abs_kv_pos / s.page_size;
        let offset_in_page = abs_kv_pos % s.page_size;
        let page = s.pages[page_in_req] as usize;
        if hnd {
            let h = i / s.d;
            let j = i - h * s.d;
            ((page * s.h_kv + h) * s.page_size + offset_in_page) * s.d + j
        } else {
            (page * s.page_size + offset_in_page) * s.row() + i
        }
    }

    /// The whole arena as it must look after the fire: the scattered tokens,
    /// and zero everywhere else.
    fn expected(hnd: bool, s: &Shape, src: &[u16]) -> Vec<u16> {
        let mut out = vec![0u16; s.cells()];
        for t in 0..s.tokens {
            for i in 0..s.row() {
                out[scatter(hnd, s, t, i)] = src[t * s.row() + i];
            }
        }
        out
    }

    /// bf16 words that are never zero, so "did not write" and "wrote a zero"
    /// are different observations.
    fn sample(n: usize, seed: u64) -> Vec<u16> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bits = (state >> 33) as u32;
                let sign = (bits & 1) << 15;
                let exponent = 120 + (bits >> 1) % 12;
                let mantissa = ((bits >> 8) & 0x7e) | 1;
                u16::try_from(sign | (exponent << 7) | mantissa).expect("16 bits")
            })
            .collect()
    }

    /// A device allocation of `T`, freed on drop.
    struct Buffer {
        ptr: u64,
        bytes: usize,
    }

    impl Buffer {
        fn of<T: Copy>(from: &[T]) -> Self {
            let bytes = std::mem::size_of_val(from).max(1);
            let mut ptr = 0u64;
            // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
            let code = unsafe { dr::cuMemAlloc_v2(&raw mut ptr, bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "allocating {bytes} bytes");
            let me = Self { ptr, bytes };
            if !from.is_empty() {
                // SAFETY: the allocation is exactly `from`'s size.
                let code =
                    unsafe { dr::cuMemcpyHtoD_v2(ptr, from.as_ptr().cast(), me.bytes) };
                assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "upload");
            }
            me
        }

        fn zeroed(words: usize) -> Self {
            let me = Self::of(&vec![0u16; words]);
            me.clear();
            me
        }

        fn clear(&self) {
            // SAFETY: the allocation is `bytes` long.
            let code = unsafe { dr::cuMemsetD8_v2(self.ptr, 0, self.bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "memset");
        }

        fn words(&self) -> Vec<u16> {
            let mut out = vec![0u16; self.bytes / 2];
            // SAFETY: same allocation, same length.
            let code =
                unsafe { dr::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, self.bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "download");
            out
        }

        fn arg(&self) -> ArgValue {
            ArgValue::Ptr(self.ptr as *mut c_void)
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            // SAFETY: the handle came from `cuMemAlloc_v2` and is freed once.
            unsafe { dr::cuMemFree_v2(self.ptr) };
        }
    }

    fn synchronise(what: &str) {
        // SAFETY: no arguments, and the context is bound.
        let code = unsafe { dr::cuCtxSynchronize() };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "{what}");
    }

    /// The unit, compiled with the two instantiations the arms name, loaded.
    ///
    /// `nvrtc::compile_rows` against the SHIPPED `attn/kv_paged` unit — the
    /// same root, the same header set, the same options — so what is on trial
    /// is the two rows and nothing about the compile path.
    fn module(arch: &str) -> (KernelModule, Vec<(&'static str, String)>) {
        let unit = unit::UNITS
            .iter()
            .find(|u| u.name == "attn/kv_paged")
            .expect("`attn/kv_paged` is a unit");
        let rows: Vec<&DeviceKernel> = ROWS.iter().collect();
        let compiled = nvrtc::compile_rows(unit, arch, &rows).expect("all three rows compile");
        let table: Vec<&'static KernelSig> = vec![&SIGS[0], &SIGS[1], &SIGS[2]];
        let module = KernelModule::load_mangled(unit.name, &compiled.cubin, &table, &compiled.lowered)
            .expect("all three rows load");
        (module, compiled.lowered)
    }

    /// `sm_XY` for the current device, or a stated reason there is none.
    fn arch_or_skip(what: &str) -> Option<&'static str> {
        match cache::arch() {
            Some(arch) => match cache::bind_context() {
                Ok(()) => Some(arch),
                Err(why) => {
                    eprintln!("SKIP {what}: no usable context ({why})");
                    None
                }
            },
            None => {
                eprintln!("SKIP {what}: no CUDA device is current");
                None
            }
        }
    }

    /// The device buffers one fire needs, kept alive for its duration.
    struct Fire {
        k_curr: Buffer,
        v_curr: Buffer,
        k_pages: Buffer,
        v_pages: Buffer,
        qo_indptr: Buffer,
        page_indices: Buffer,
        page_indptr: Buffer,
        last_lens: Buffer,
        k_src: Vec<u16>,
        v_src: Vec<u16>,
    }

    impl Fire {
        fn new(s: &Shape, seed: u64) -> Self {
            let k_src = sample(s.tokens * s.row(), seed);
            let v_src = sample(s.tokens * s.row(), seed ^ 0xa5a5_a5a5);
            let last = (s.tokens - (s.pages.len() - 1) * s.page_size) as u32;
            Self {
                k_curr: Buffer::of(&k_src),
                v_curr: Buffer::of(&v_src),
                k_pages: Buffer::zeroed(s.cells()),
                v_pages: Buffer::zeroed(s.cells()),
                qo_indptr: Buffer::of(&[0u32, s.tokens as u32]),
                page_indices: Buffer::of(s.pages),
                page_indptr: Buffer::of(&[0u32, s.pages.len() as u32]),
                last_lens: Buffer::of(&[last]),
                k_src,
                v_src,
            }
        }

        /// The BASE row's sixteen values — fifteen the kernel takes and the
        /// flag it does not. Exactly what a caller would hand
        /// `runtime::fire("attn::write_kv_bf16", …)`.
        fn values(&self, s: &Shape, hnd: bool) -> [ArgValue; 16] {
            [
                self.k_curr.arg(),
                self.v_curr.arg(),
                self.k_pages.arg(),
                self.v_pages.arg(),
                self.qo_indptr.arg(),
                self.page_indices.arg(),
                self.page_indptr.arg(),
                self.last_lens.arg(),
                ArgValue::Ptr(std::ptr::null_mut()),
                ArgValue::Ptr(std::ptr::null_mut()),
                ArgValue::I32(1),
                ArgValue::I32(s.page_size as i32),
                ArgValue::I32(s.h_kv as i32),
                ArgValue::I32(s.d as i32),
                ArgValue::I32(0),
                ArgValue::Bool(hnd),
            ]
        }
    }

    /// The facts a fire would supply, through the shipped mapping.
    ///
    /// `runtime::fire::facts` is private, so this calls the function IT calls
    /// — [`ArgValue::fact`] — over the same list in the same order. If those
    /// two ever disagree the flag would arrive as something no clause can
    /// read, and every one of the assertions below would fail loudly.
    fn facts(values: &[ArgValue]) -> Vec<Fact> {
        values.iter().map(|v| v.fact()).collect()
    }

    /// The reshape `runtime::fire::fire_arm` performs, mirrored.
    fn reshape(arm: &Arm, values: &[ArgValue]) -> Vec<ArgValue> {
        arm.take
            .iter()
            .map(|take| match take {
                Take::From(index) => values[*index],
                Take::Null => ArgValue::Ptr(std::ptr::null_mut()),
            })
            .collect()
    }

    fn fire(module: &KernelModule, sig: &'static KernelSig, s: &Shape, list: &[ArgValue]) {
        let geometry = runtime::eval(sig.launch, Dims { rows: s.tokens as u32, ..Dims::default() })
            .expect("PerRow over a non-empty token count");
        assert_eq!(geometry.grid, [s.tokens as u32, 1, 1], "`<<<num_tokens, 256>>>`");
        assert_eq!(geometry.block, [256, 1, 1]);
        let mut args = kernels_cuda_new::runtime::Args::bind(sig, list).expect("the list binds");
        // SAFETY: every pointer addresses a live allocation of the extent the
        // row states, the two nulls are the operands the row declares
        // nullable, and the null stream is always live.
        module.fire(sig, geometry, &mut args, Stream::NULL).expect("the arm launches");
    }

    /// How many bf16 words differ.
    fn differing(a: &[u16], b: &[u16]) -> usize {
        a.iter().zip(b).filter(|(l, r)| l != r).count()
    }

    /// **Both arms of a real `template <bool>`, selected by a `bool`, at zero
    /// ulp.**
    ///
    /// Each shape is fired twice through the same base row, with `hnd_layout`
    /// the only difference between the two argument lists. For each fire the
    /// test states, in order:
    ///
    /// 1. the arm [`Specialisation::choose`] names, and that it is the one
    ///    `kv_paged.cu:51` would have picked;
    /// 2. that the arm's row spells the `elem` for that value, and that NVRTC
    ///    lowered it to a symbol carrying `Lb1` or `Lb0` — so the kernel in
    ///    the module is demonstrably the instantiation the predicate asked
    ///    for, not merely a kernel that agreed;
    /// 3. that the arena afterwards is bit-identical to the hand-computed
    ///    scatter, over EVERY cell including the pages nothing should touch.
    #[test]
    fn both_flag_arms_fire_and_land_where_the_flag_says() {
        let Some(arch) = arch_or_skip("both_flag_arms_fire_and_land_where_the_flag_says") else {
            return;
        };
        let (module, lowered) = module(arch);

        let mangled = |symbol: &str| {
            lowered
                .iter()
                .find(|(s, _)| *s == symbol)
                .map(|(_, m)| m.clone())
                .expect("the row was compiled")
        };
        let hnd_symbol = mangled("attn::write_kv_bf16#hnd");
        let nhd_symbol = mangled("attn::write_kv_bf16#nhd");
        let base_symbol = mangled("attn::write_kv_bf16");
        println!("\n  write_kv<true>  -> {hnd_symbol}");
        println!("  write_kv<false> -> {nhd_symbol}");
        assert!(hnd_symbol.contains("ILb1"), "`true` must mangle as `Lb1`: {hnd_symbol}");
        assert!(nhd_symbol.contains("ILb0"), "`false` must mangle as `Lb0`: {nhd_symbol}");
        assert_ne!(hnd_symbol, nhd_symbol, "two arms that share a symbol are one arm");
        // The contract row names one of the two, so the unit asked NVRTC for
        // the same name expression twice. That it resolves rather than
        // conflicting is what makes the three-row shape writable at all.
        assert_eq!(base_symbol, nhd_symbol, "the base row's instantiation is the `nhd` one");

        println!(
            "\n{:>6} {:>5} {:>5} {:>4} {:>6} {:>8} {:>10} {:>10}",
            "tokens", "pages", "h_kv", "d", "layout", "arm", "cells", "differing"
        );
        println!("{}", "-".repeat(70));
        let mut total_cells = 0usize;
        let mut total_differ = 0usize;

        for (at, s) in SHAPES.iter().enumerate() {
            for hnd in [true, false] {
                let f = Fire::new(s, 0x51de_0000 + at as u64);
                let values = f.values(s, hnd);

                // 1. The predicate names an arm, and it is the C++'s.
                let arm = SPEC
                    .choose(&facts(&values))
                    .expect("no clause faults on a flag the fire bound")
                    .expect("both values of the flag are covered");
                assert_eq!(arm.name, if hnd { "hnd" } else { "nhd" });

                // 2. The arm's row is the instantiation for that value.
                let want_elem =
                    if hnd { "device::true_type::value" } else { "device::false_type::value" };
                assert_eq!(arm.row.elem, want_elem);
                let want_mangled = if hnd { "ILb1" } else { "ILb0" };
                assert!(
                    mangled(arm.row.sig.symbol).contains(want_mangled),
                    "the arm the flag chose resolved to the wrong instantiation"
                );

                // 3. It writes what the layout says, everywhere.
                fire(&module, arm.row.sig, s, &reshape(arm, &values));
                synchronise("a flag arm");

                let want_k = expected(hnd, s, &f.k_src);
                let want_v = expected(hnd, s, &f.v_src);
                let got_k = f.k_pages.words();
                let got_v = f.v_pages.words();
                let differ = differing(&got_k, &want_k) + differing(&got_v, &want_v);
                let cells = want_k.len() + want_v.len();
                println!(
                    "{:>6} {:>5} {:>5} {:>4} {:>6} {:>8} {:>10} {:>10}",
                    s.tokens,
                    s.arena,
                    s.h_kv,
                    s.d,
                    if hnd { "HND" } else { "NHD" },
                    arm.name,
                    cells,
                    differ
                );
                total_cells += cells;
                total_differ += differ;

                assert!(got_k.iter().any(|&v| v != 0), "the arm wrote nothing at all");
                // The two layouts must be genuinely different at this shape,
                // or the comparison above would pass for the wrong reason.
                assert_ne!(
                    expected(true, s, &f.k_src),
                    expected(false, s, &f.k_src),
                    "at this shape HND and NHD scatter identically, so it proves nothing"
                );
            }
        }
        println!("{}", "-".repeat(70));
        println!("{total_differ} of {total_cells} bf16 cells differ from the hand-computed scatter\n");
        assert_eq!(
            total_differ, 0,
            "the arm the flag chose did not write what that layout says: {total_differ} of \
             {total_cells} cells differ. A KV cache in the other order is not an approximation \
             of the right one — it is a wrong read on the next decode."
        );
    }

    /// **The negative control: the arm the flag did NOT name, fired on
    /// purpose.**
    ///
    /// `hnd_layout` is `true`, the predicate says `hnd`, and the `nhd` arm is
    /// fired instead — which is precisely what a row that froze one
    /// instantiation would have done, and what `families/attn.rs` refused to
    /// write for exactly this reason.
    ///
    /// The failure it produces is silent in every way a check can be silent:
    /// no fault, no driver error, the same number of stores to the same
    /// arena, every value present and every value a real bf16 the model
    /// wrote. Only the ADDRESSES moved. There is no tolerance that flags a
    /// permutation, which is why the bar in
    /// [`both_flag_arms_fire_and_land_where_the_flag_says`] is bit equality
    /// over the whole arena rather than a norm over what was written.
    #[test]
    fn the_wrong_flag_arm_is_caught() {
        let Some(arch) = arch_or_skip("the_wrong_flag_arm_is_caught") else { return };
        let (module, _) = module(arch);

        println!(
            "\n{:>6} {:>5} {:>4} {:>10} {:>10} {:>9}",
            "tokens", "h_kv", "d", "cells", "differing", "written"
        );
        println!("{}", "-".repeat(52));
        let mut total_differ = 0usize;
        let mut total_cells = 0usize;
        for (at, s) in SHAPES.iter().enumerate() {
            let f = Fire::new(s, 0xdead_0000 + at as u64);
            let values = f.values(s, true);
            let arm = SPEC.choose(&facts(&values)).expect("no fault").expect("covered");
            assert_eq!(arm.name, "hnd", "the flag says HND");

            // The mistake: the other arm, on the same sixteen values.
            let wrong = &ARMS[1];
            assert_eq!(wrong.name, "nhd");
            fire(&module, wrong.row.sig, s, &reshape(wrong, &values));
            synchronise("the forced arm");

            let want = expected(true, s, &f.k_src);
            let got = f.k_pages.words();
            let differ = differing(&got, &want);
            let written = want.iter().filter(|&&v| v != 0).count();
            println!(
                "{:>6} {:>5} {:>4} {:>10} {:>10} {:>9}",
                s.tokens,
                s.h_kv,
                s.d,
                want.len(),
                differ,
                written
            );
            total_differ += differ;
            total_cells += want.len();

            // The forced arm wrote the same NUMBER of values — it is a
            // permutation of the right answer and not a truncation of it, so
            // nothing downstream sees a hole where it could notice.
            assert_eq!(
                got.iter().filter(|&&v| v != 0).count(),
                written,
                "the wrong arm must write as many values as the right one — that it does is \
                 what makes it undetectable by anything but an address check"
            );
            assert!(differ > 0, "the wrong arm produced the right cache, so this proves nothing");
        }
        println!("{}", "-".repeat(52));
        println!(
            "negative control: firing `write_kv<false>` where the flag says `true` moved \
             {total_differ} of {total_cells} bf16 cells, with the same count written\n"
        );
        assert!(total_differ > 0);
    }

    /// A fire cannot reach a kernel through a flag no clause can read.
    ///
    /// The other half of the control: not "the wrong arm ran" but "no arm can
    /// run". A base row whose flag operand were a `Ty::I32` binds
    /// [`Fact::Int`], every [`Term::Is`] over it faults, and
    /// [`Specialisation::choose`] returns [`kernels_cuda_new::device::Fault`]
    /// rather than falling through to the base — which for these rows would
    /// bind sixteen cells for a fifteen-parameter kernel and SUCCEED.
    #[test]
    fn a_flag_that_arrives_as_a_number_faults_rather_than_falling_through() {
        // The same sixteen values with the flag mistyped, which is what a row
        // that declared `hnd_layout: I32` would produce at every fire. No
        // device: `choose` reads facts, and a fault is decided before a
        // pointer is ever dereferenced.
        let mut mistyped = vec![ArgValue::Ptr(std::ptr::null_mut()); 15];
        mistyped.push(ArgValue::I32(1));
        assert_eq!(
            SPEC.choose(&facts(&mistyped)).map(|arm| arm.map(|a| a.name)),
            Err(kernels_cuda_new::device::Fault::Kind { operand: 15, wanted: "a flag" }),
            "an `Int(1)` must not be read as `true` — that coercion is the whole reason \
             `Fact::Bool` is a variant of its own"
        );
        // And one cell short is a fault too, not a silent `false`.
        assert_eq!(
            SPEC.choose(&facts(&mistyped[..15])).map(|arm| arm.map(|a| a.name)),
            Err(kernels_cuda_new::device::Fault::Range { operand: 15, arity: 15 })
        );
    }
}

/// **A `__global__` with no template parameter list, named by a row, fired
/// through the shipped path, and checked against the launcher it replaces.**
///
/// This module is the fire proof for [`kernels_cuda_new::device::DeviceKernel::PLAIN`].
/// Three agents independently reported that `instantiation()` could not name
/// a plain kernel; the fix was to teach it to, and the standing hazard with a
/// naming fix is that a row which RESOLVES gets mistaken for a row that
/// LAUNCHES. `cuModuleGetFunction` returning a handle proves the mangled name
/// was right. It proves nothing about the geometry the rule computes, nothing
/// about the order `Args::bind` packs operands in, and nothing about whether
/// the bytes that land are the bytes `pack_dense_mask.cu` would have landed.
/// So all three are measured here, against `attn/pack_dense_mask` — a unit
/// whose two rows are plain and whose device text this change did not touch.
///
/// # What is compared against what
///
/// `kernels-cuda-new` cannot depend on `kernels-cuda`: the edge runs the
/// other way (see `Cargo.toml`), so the ahead-of-time `pack_dense_mask(...)`
/// host function cannot be called from this test at all. What CAN be
/// compared is everything that function is, in the two pieces it is made of,
/// and both are compared here:
///
///  1. **The launch.** `pack_dense_mask.cu:94` is
///     `device::pack_dense_mask<<<B, BLOCK, 0, stream>>>` with
///     `constexpr int BLOCK = 128` at `:93`, and `<<<>>>` is `cuLaunchKernel`
///     with sugar. [`the_shipped_path_reproduces_the_launcher`] fires the
///     SAME entry point in the SAME module twice — once through
///     `runtime::eval` + `Args::bind` + `KernelModule::fire`, and once
///     through a raw `cuLaunchKernel` with the launcher's literal
///     `(B, 1, 1) x (128, 1, 1)` and a hand-built pointer array in the
///     kernel's declared order — and asserts the two output arenas are equal
///     byte for byte. That isolates exactly what the row adds over the
///     launcher: the rule, and the binding.
///  2. **The arithmetic.** [`a_plain_row_fires_and_packs_what_the_header_says`]
///     compares the whole `packed` arena against a HAND transliteration of
///     `pack_dense_mask.cuh`, in the header's own order, over every byte
///     including the ones no lane may touch. An oracle derived from the
///     kernel would compare a value to itself; this one is written from the
///     C++ by hand, the way `flag_arms::scatter` is.
///
/// # The negative control is a permutation
///
/// A control that writes FEWER bytes is caught by any count. §21.14 measured
/// an arm that moved 34 273 of 55 200 cells while writing the same number of
/// non-zero values, and no count, norm or tolerance would have flagged it. So
/// the control here is built to be exactly that: two lanes with the SAME byte
/// count and DIFFERENT contents, and a `mask_indptr` whose first two entries
/// are swapped. The kernel reads only `mask_indptr[b]`, so the swap sends
/// lane 0's bytes where lane 1's belong and back — the multiset of output
/// bytes is identical, the non-zero count is identical, and the arrangement
/// is wrong. [`the_wrong_offsets_are_caught_by_bytes_and_not_by_counts`]
/// asserts both halves of that: that the counts AGREE and the bytes DIFFER.
#[cfg(feature = "_cuda")]
mod plain_kernels {
    use cudarc::driver::sys as dr;
    use kernels::KernelSig;
    use kernels_cuda_new::device::DeviceKernel;
    use kernels_cuda_new::runtime::{self, ArgValue, Dims, KernelModule, Stream, cache, nvrtc};
    use kernels_cuda_new::unit;
    use std::ffi::c_void;

    /// The launcher's literal, from `pack_dense_mask.cu:93`.
    const BLOCK: u32 = 128;

    /// `sm_XY` for the current device, or a stated reason there is none.
    fn arch_or_skip(what: &str) -> Option<&'static str> {
        match cache::arch() {
            Some(arch) => match cache::bind_context() {
                Ok(()) => Some(arch),
                Err(why) => {
                    eprintln!("SKIP {what}: no usable context ({why})");
                    None
                }
            },
            None => {
                eprintln!("SKIP {what}: no CUDA device is current");
                None
            }
        }
    }

    fn synchronise(what: &str) {
        // SAFETY: no arguments, and the context is bound.
        let code = unsafe { dr::cuCtxSynchronize() };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "{what}");
    }

    /// A device allocation, freed on drop.
    struct Buffer {
        ptr: u64,
        bytes: usize,
    }

    impl Buffer {
        fn of<T: Copy>(from: &[T]) -> Self {
            let bytes = std::mem::size_of_val(from).max(1);
            let mut ptr = 0u64;
            // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
            let code = unsafe { dr::cuMemAlloc_v2(&raw mut ptr, bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "allocating {bytes} bytes");
            let me = Self { ptr, bytes };
            if !from.is_empty() {
                // SAFETY: the allocation is exactly `from`'s size.
                let code = unsafe { dr::cuMemcpyHtoD_v2(ptr, from.as_ptr().cast(), me.bytes) };
                assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "upload");
            }
            me
        }

        fn zeroed(bytes: usize) -> Self {
            let me = Self::of(&vec![0u8; bytes]);
            me.clear();
            me
        }

        fn clear(&self) {
            // SAFETY: the allocation is `bytes` long.
            let code = unsafe { dr::cuMemsetD8_v2(self.ptr, 0, self.bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "memset");
        }

        fn bytes(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.bytes];
            // SAFETY: same allocation, same length.
            let code =
                unsafe { dr::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, self.bytes) };
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "download");
            out
        }

        fn arg(&self) -> ArgValue {
            ArgValue::Ptr(self.ptr as *mut c_void)
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            // SAFETY: the handle came from `cuMemAlloc_v2` and is freed once.
            unsafe { dr::cuMemFree_v2(self.ptr) };
        }
    }

    /// One request's mask, small enough that the right answer is a table.
    #[derive(Clone, Copy)]
    struct Lane {
        /// Query rows in this lane.
        qo_len: usize,
        /// Physical key span — the mask's live column count.
        klen: usize,
    }

    impl Lane {
        const fn bits(self) -> usize {
            self.qo_len * self.klen
        }
        const fn packed_bytes(self) -> usize {
            self.bits().div_ceil(8)
        }
    }

    /// A whole batch: the lanes, the dense mask's row stride, and how many
    /// bytes the arena holds beyond what the lanes claim.
    struct Batch {
        lanes: &'static [Lane],
        /// `P_PAGE` — the dense `[TOTAL_Q, STRIDE]` row stride. Strictly
        /// greater than every `klen`, so a kernel that used `klen` as the
        /// stride would read the wrong columns and be caught.
        stride: usize,
        /// Trailing bytes in `packed` that no lane's offset reaches. Nothing
        /// may write them, and the comparison covers them.
        slack: usize,
    }

    impl Batch {
        fn qo_indptr(&self) -> Vec<u32> {
            let mut out = vec![0u32];
            for lane in self.lanes {
                out.push(out[out.len() - 1] + u32::try_from(lane.qo_len).expect("small"));
            }
            out
        }

        fn klen(&self) -> Vec<u32> {
            self.lanes.iter().map(|l| u32::try_from(l.klen).expect("small")).collect()
        }

        /// The per-lane BYTE offsets, prefix-summed the way the host builds
        /// them — `pack_dense_mask.cu:79-81`.
        fn mask_indptr(&self) -> Vec<i32> {
            let mut out = vec![0i32];
            for lane in self.lanes {
                out.push(out[out.len() - 1] + i32::try_from(lane.packed_bytes()).expect("small"));
            }
            out
        }

        fn total_q(&self) -> usize {
            self.lanes.iter().map(|l| l.qo_len).sum()
        }

        fn arena(&self) -> usize {
            usize::try_from(*self.mask_indptr().last().expect("a lane")).expect("non-negative")
                + self.slack
        }

        /// The dense `[TOTAL_Q, STRIDE]` mask, one byte per cell.
        ///
        /// Deliberately NOT all ones: a mask that admitted everything would
        /// make every output byte `0xff`, and a permutation of identical
        /// bytes is not a permutation anyone can see. The pattern below has
        /// no row equal to another and no column equal to another.
        fn dense(&self, seed: u64) -> Vec<u8> {
            let mut state = seed | 1;
            (0..self.total_q() * self.stride)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    u8::from((state >> 60) & 3 != 0)
                })
                .collect()
        }
    }

    /// `pack_dense_mask.cuh:127-160`, transliterated by hand.
    ///
    /// A HAND copy of the C++, in the C++'s order and with every term it
    /// computes reproduced rather than short-circuited. `total_bits`, the
    /// byte count, the per-bit `qi`/`col` split and the `qo_lo + qi` row base
    /// all appear here because they all appear there; the point of an oracle
    /// is that it was written from the source and not from a run of it.
    fn expected(batch: &Batch, dense: &[u8], offsets: &[i32]) -> Vec<u8> {
        let qo_indptr = batch.qo_indptr();
        let mut out = vec![0u8; batch.arena()];
        for (b, lane) in batch.lanes.iter().enumerate() {
            let kl = lane.klen;
            let qo_lo = qo_indptr[b] as usize;
            let qo_len = qo_indptr[b + 1] as usize - qo_lo;
            if kl == 0 || qo_len == 0 {
                continue;
            }
            let total_bits = qo_len * kl;
            let base = usize::try_from(offsets[b]).expect("non-negative");
            for byte in 0..total_bits.div_ceil(8) {
                let mut acc = 0u8;
                for bit in 0..8 {
                    let gbit = byte * 8 + bit;
                    if gbit < total_bits {
                        let qi = gbit / kl;
                        let col = gbit % kl;
                        if dense[(qo_lo + qi) * batch.stride + col] != 0 {
                            acc |= 1u8 << bit;
                        }
                    }
                }
                out[base + byte] = acc;
            }
        }
        out
    }

    /// The shipped unit, the shipped rows, compiled and loaded.
    ///
    /// `unit::UNITS` and nothing hand-spelled: what is on trial is the rows
    /// `families/attn.rs` declares, so a test that restated them would prove
    /// something about its own copy. `nvrtc::compile_rows` is the same
    /// function `tests/units.rs` walks every unit with.
    fn module(arch: &str) -> (KernelModule, Vec<(&'static str, String)>, &'static [DeviceKernel]) {
        let unit = unit::UNITS
            .iter()
            .find(|u| u.name == "attn/pack_dense_mask")
            .expect("`attn/pack_dense_mask` is a unit");
        let rows: Vec<&DeviceKernel> = unit.rows.iter().collect();
        assert!(
            rows.iter().all(|row| row.is_plain()),
            "this proof is about plain kernels; a templated row here means the unit changed"
        );
        let compiled = nvrtc::compile_rows(unit, arch, &rows).expect("both plain rows compile");
        let table: Vec<&'static KernelSig> = unit.rows.iter().map(|row| row.sig).collect();
        let module =
            KernelModule::load_mangled(unit.name, &compiled.cubin, &table, &compiled.lowered)
                .expect("both plain rows resolve");
        (module, compiled.lowered, unit.rows)
    }

    /// The row for `attn::pack_dense_mask`, and its sig.
    fn dense_row(rows: &'static [DeviceKernel]) -> &'static DeviceKernel {
        rows.iter().find(|row| row.sig.symbol == "attn::pack_dense_mask").expect("the first row")
    }

    /// The device buffers one fire needs, kept alive for its duration.
    struct Fire {
        dense: Buffer,
        klen: Buffer,
        qo_indptr: Buffer,
        mask_indptr: Buffer,
        packed: Buffer,
        b: i32,
        stride: i32,
    }

    impl Fire {
        fn new(batch: &Batch, dense: &[u8], offsets: &[i32]) -> Self {
            Self {
                dense: Buffer::of(dense),
                klen: Buffer::of(&batch.klen()),
                qo_indptr: Buffer::of(&batch.qo_indptr()),
                mask_indptr: Buffer::of(offsets),
                packed: Buffer::zeroed(batch.arena()),
                b: i32::try_from(batch.lanes.len()).expect("small"),
                stride: i32::try_from(batch.stride).expect("small"),
            }
        }

        /// The row's seven operands, in the kernel's order. Exactly what a
        /// caller would hand `runtime::fire("attn::pack_dense_mask", …)`.
        fn values(&self) -> [ArgValue; 7] {
            [
                self.dense.arg(),
                self.klen.arg(),
                self.qo_indptr.arg(),
                self.mask_indptr.arg(),
                self.packed.arg(),
                ArgValue::I32(self.b),
                ArgValue::I32(self.stride),
            ]
        }
    }

    /// Fire through the SHIPPED path: the rule computes the geometry,
    /// `Args::bind` packs the list, `KernelModule::fire` launches.
    fn shipped(module: &KernelModule, sig: &'static KernelSig, fire: &Fire) {
        let rows = u32::try_from(fire.b).expect("non-negative");
        let geometry = runtime::eval(sig.launch, Dims { rows, ..Dims::default() })
            .expect("PerRowNarrow over a non-empty lane count");
        assert_eq!(geometry.grid, [rows, 1, 1], "`<<<B, 128, 0, stream>>>` opens B blocks");
        assert_eq!(geometry.block, [BLOCK, 1, 1], "`constexpr int BLOCK = 128`");
        assert_eq!(geometry.smem, 0, "the launcher asks for no dynamic shared memory");
        let list = fire.values();
        let mut args = runtime::Args::bind(sig, &list).expect("the list binds");
        // SAFETY: every pointer addresses a live allocation of the extent the
        // row states, `packed` is `arena()` bytes and the largest offset any
        // lane writes is `arena() - slack`, and the null stream is live.
        module.fire(sig, geometry, &mut args, Stream::NULL).expect("the plain row launches");
        synchronise("the shipped fire");
    }

    /// Fire the way `pack_dense_mask.cu:94` does: `cuLaunchKernel` with the
    /// launcher's literal geometry and a hand-built pointer array, bypassing
    /// [`runtime::eval`] and [`runtime::Args`] entirely.
    ///
    /// `<<<B, BLOCK, 0, stream>>>` IS this call — the triple-chevron is sugar
    /// over `cudaLaunchKernel`, which is `cuLaunchKernel` under the runtime
    /// API. The same entry point in the same module is fired, so the only
    /// difference between this and [`shipped`] is the two pieces of the
    /// shipped path under test.
    fn as_the_launcher_would(entry: dr::CUfunction, fire: &Fire) {
        let mut dense = fire.dense.ptr;
        let mut klen = fire.klen.ptr;
        let mut qo_indptr = fire.qo_indptr.ptr;
        let mut mask_indptr = fire.mask_indptr.ptr;
        let mut packed = fire.packed.ptr;
        let mut b = fire.b;
        let mut stride = fire.stride;
        let mut raw: [*mut c_void; 7] = [
            (&raw mut dense).cast(),
            (&raw mut klen).cast(),
            (&raw mut qo_indptr).cast(),
            (&raw mut mask_indptr).cast(),
            (&raw mut packed).cast(),
            (&raw mut b).cast(),
            (&raw mut stride).cast(),
        ];
        // SAFETY: `entry` came from a loaded module that outlives the call,
        // the seven cells are live for its duration and are in the kernel's
        // declared order and widths, and the null stream is live.
        let code = unsafe {
            dr::cuLaunchKernel(
                entry,
                u32::try_from(fire.b).expect("non-negative"),
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                std::ptr::null_mut(),
                raw.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "the launcher's own launch");
        synchronise("the launcher's fire");
    }

    /// How many bytes differ, and how many are non-zero on each side.
    fn differing(a: &[u8], b: &[u8]) -> usize {
        assert_eq!(a.len(), b.len(), "two arenas of different sizes are not comparable");
        a.iter().zip(b).filter(|(l, r)| l != r).count()
    }

    fn written(a: &[u8]) -> usize {
        a.iter().filter(|byte| **byte != 0).count()
    }

    /// Five batches, spanning a decode step and a prefill.
    ///
    /// The second lane of each is never a copy of the first, and no batch has
    /// a bit count that is a multiple of 8 in every lane — the tail bits of
    /// the last byte are where a `<` written as a `<=` shows up, and they are
    /// the bits a kernel that packed whole bytes would get wrong.
    fn batches() -> Vec<Batch> {
        vec![
            Batch { lanes: &[Lane { qo_len: 3, klen: 5 }], stride: 8, slack: 3 },
            Batch {
                lanes: &[Lane { qo_len: 1, klen: 17 }, Lane { qo_len: 4, klen: 9 }],
                stride: 24,
                slack: 5,
            },
            Batch {
                lanes: &[
                    Lane { qo_len: 7, klen: 33 },
                    Lane { qo_len: 2, klen: 64 },
                    Lane { qo_len: 11, klen: 1 },
                ],
                stride: 72,
                slack: 9,
            },
            // A lane with nothing in it, next to lanes that have plenty. The
            // kernel returns early on `kl <= 0 || qo_len <= 0`, and the bytes
            // its offset points at must stay zero.
            Batch {
                lanes: &[
                    Lane { qo_len: 5, klen: 12 },
                    Lane { qo_len: 0, klen: 12 },
                    Lane { qo_len: 6, klen: 0 },
                    Lane { qo_len: 3, klen: 20 },
                ],
                stride: 32,
                slack: 4,
            },
            // Wider than one block of 128 threads' worth of bytes, so the
            // stride loop runs more than once per thread.
            Batch {
                lanes: &[Lane { qo_len: 64, klen: 200 }, Lane { qo_len: 33, klen: 129 }],
                stride: 256,
                slack: 7,
            },
        ]
    }

    /// **A row with no `elem` compiles, resolves, launches, and packs the
    /// bitmap `pack_dense_mask.cuh` describes — byte for byte.**
    ///
    /// The whole arena is compared, including the `slack` bytes past the last
    /// lane and the bytes belonging to lanes that return early: "did not
    /// write" and "wrote a zero" are different observations and both are
    /// checked. `written > 0` is asserted per batch, because an arena that
    /// stayed entirely zero would match an oracle that also computed zero and
    /// prove nothing at all.
    #[test]
    fn a_plain_row_fires_and_packs_what_the_header_says() {
        let Some(arch) = arch_or_skip("a_plain_row_fires_and_packs_what_the_header_says") else {
            return;
        };
        let (module, lowered, rows) = module(arch);
        let row = dense_row(rows);

        // The name NVRTC lowered is the bare path's, and it carries no
        // template arguments -- `I...E` is the Itanium ABI's template-argument
        // bracket, and its absence is the mangling saying "not a template".
        let (_, mangled) = lowered
            .iter()
            .find(|(symbol, _)| *symbol == "attn::pack_dense_mask")
            .expect("the row was lowered");
        assert_eq!(row.instantiation(), "::pie_cuda_driver::kernels::attn::device::pack_dense_mask");
        assert!(mangled.contains("pack_dense_mask"), "{mangled} does not name the kernel");
        assert!(
            !mangled.contains("pack_dense_maskI"),
            "{mangled} mangles template arguments onto a kernel that has none"
        );

        let mut total_bytes = 0usize;
        let mut total_written = 0usize;
        for (index, batch) in batches().iter().enumerate() {
            let dense = batch.dense(0x51ed_0000 + index as u64);
            let offsets = batch.mask_indptr();
            let fire = Fire::new(batch, &dense, &offsets);
            shipped(&module, row.sig, &fire);

            let got = fire.packed.bytes();
            let want = expected(batch, &dense, &offsets);
            assert_eq!(got.len(), batch.arena());
            assert_eq!(
                differing(&got, &want),
                0,
                "batch {index}: {} of {} bytes differ from the header's arithmetic",
                differing(&got, &want),
                got.len()
            );
            assert!(
                written(&got) > 0,
                "batch {index} wrote nothing, so an all-zero oracle would have passed"
            );
            total_bytes += got.len();
            total_written += written(&got);
        }
        assert!(total_written > 0);
        eprintln!(
            "attn::pack_dense_mask: {} batches, {total_bytes} bytes of packed bitmap compared, \
             {total_written} non-zero, 0 differing",
            batches().len()
        );
    }

    /// **The shipped path launches what the ahead-of-time launcher launches.**
    ///
    /// Same module, same entry point, same buffers, two launches: one through
    /// `runtime::eval` + `Args::bind` + `KernelModule::fire`, one through a
    /// raw `cuLaunchKernel` with `pack_dense_mask.cu:94`'s literal geometry
    /// and its own argument order. The arenas must be equal byte for byte.
    ///
    /// This is the half a hand oracle cannot cover: an oracle says the DEVICE
    /// text is right, and this says the HOST side of the row — the rule that
    /// turns `B` into a grid and the binding that turns seven `ArgValue`s
    /// into the array the driver reads — reproduces the launcher rather than
    /// merely producing something self-consistent.
    #[test]
    fn the_shipped_path_reproduces_the_launcher() {
        let Some(arch) = arch_or_skip("the_shipped_path_reproduces_the_launcher") else {
            return;
        };
        let (module, _, rows) = module(arch);
        let row = dense_row(rows);
        let entry = module.entry(row.sig.symbol).expect("the row resolved");

        let mut compared = 0usize;
        for (index, batch) in batches().iter().enumerate() {
            let dense = batch.dense(0xc0ff_ee00 + index as u64);
            let offsets = batch.mask_indptr();

            let fire = Fire::new(batch, &dense, &offsets);
            shipped(&module, row.sig, &fire);
            let through_the_row = fire.packed.bytes();

            fire.packed.clear();
            synchronise("clearing between the two launches");
            as_the_launcher_would(entry, &fire);
            let through_the_launcher = fire.packed.bytes();

            assert!(
                written(&through_the_launcher) > 0,
                "batch {index}: the launcher's own launch wrote nothing, so this \
                 comparison would hold for a kernel that did nothing"
            );
            assert_eq!(
                differing(&through_the_row, &through_the_launcher),
                0,
                "batch {index}: the row and the launcher disagree on {} of {} bytes",
                differing(&through_the_row, &through_the_launcher),
                through_the_row.len()
            );
            compared += through_the_row.len();
        }
        eprintln!(
            "attn::pack_dense_mask: {compared} bytes identical between the shipped path and \
             `<<<B, 128, 0, stream>>>`"
        );
    }

    /// **The negative control: same byte count, same non-zero count, wrong
    /// arrangement.**
    ///
    /// Two lanes are built with equal packed sizes (15 bits each, so 2 bytes
    /// each) and different contents, and the control swaps their two entries
    /// in `mask_indptr`. The kernel reads only `mask_indptr[b]`, so each lane
    /// writes the bytes it always wrote, at the other lane's offset: the
    /// output is a PERMUTATION of the right answer.
    ///
    /// The test asserts, in this order:
    ///
    ///  1. the control's non-zero byte count EQUALS the correct answer's — so
    ///     a count, a norm, a sum or a tolerance all pass it;
    ///  2. the control's bytes DIFFER from the correct answer's, and by how
    ///     many;
    ///  3. the correct answer is still the correct answer, so the difference
    ///     is the control's and not a fire that stopped working.
    #[test]
    fn the_wrong_offsets_are_caught_by_bytes_and_not_by_counts() {
        let Some(arch) = arch_or_skip("the_wrong_offsets_are_caught_by_bytes_and_not_by_counts")
        else {
            return;
        };
        let (module, _, rows) = module(arch);
        let row = dense_row(rows);

        // 3x5 and 5x3 are both 15 bits, so both lanes pack into 2 bytes and
        // swapping their offsets moves bytes without changing how many.
        let batch = Batch {
            lanes: &[Lane { qo_len: 3, klen: 5 }, Lane { qo_len: 5, klen: 3 }],
            stride: 8,
            slack: 2,
        };
        let dense = batch.dense(0xdeadbeef);
        let right = batch.mask_indptr();
        assert_eq!(right, vec![0, 2, 4], "both lanes pack into two bytes");
        let wrong = vec![right[1], right[0], right[2]];

        let good = Fire::new(&batch, &dense, &right);
        shipped(&module, row.sig, &good);
        let correct = good.packed.bytes();
        assert_eq!(differing(&correct, &expected(&batch, &dense, &right)), 0, "the fire is right");

        let bad = Fire::new(&batch, &dense, &wrong);
        shipped(&module, row.sig, &bad);
        let control = bad.packed.bytes();

        // The two lanes must actually differ, or the permutation is the
        // identity and this control controls for nothing.
        assert_ne!(&correct[0..2], &correct[2..4], "the two lanes packed the same bytes");

        assert_eq!(
            written(&control),
            written(&correct),
            "the control was meant to be a permutation, and a permutation writes the \
             same number of non-zero bytes"
        );
        assert_eq!(
            control.iter().map(|b| u32::from(*b)).sum::<u32>(),
            correct.iter().map(|b| u32::from(*b)).sum::<u32>(),
            "a permutation has the same sum, which is why a norm would not catch it"
        );
        let moved = differing(&control, &correct);
        assert!(
            moved > 0,
            "the control landed on the right answer, so it is not a control -- {moved} of {} \
             bytes differ",
            correct.len()
        );
        assert!(written(&correct) > 0, "an all-zero arena permutes to itself");
        eprintln!(
            "negative control: {moved} of {} bytes moved, {} non-zero on both sides, identical \
             sums -- caught only by comparing bytes",
            correct.len(),
            written(&correct)
        );
    }
}

/// EVERY SPECIALISATION'S ARM NAMES THE SYMBOL ITS BASE DOES.
///
/// `Specialisation` points at its arm by INDEX into a family's row list, and
/// an index is the one reference that can be silently wrong. It was:
/// `families::norm::RMSNORM_STRIDED_VEC8` said `RMSNORM_ROWS[4]`, five
/// `RowsPerHead` rows were appended, and `[4]` became `norm::rmsnorm_bf16` —
/// a different kernel with a different geometry.
///
/// `agrees()` caught it, which is the good half of the story: seven tests in
/// this file failed, each naming the mismatch — *"arm `vec8` states
/// RowsPerHead where the base states Rms; a specialisation chooses an
/// instantiation, not a geometry"*. That check is why the defect could not
/// ship.
///
/// But it caught it as a RULE mismatch, which is a coincidence of this case.
/// Two rows with the same `LaunchRule` would have swapped silently, and
/// `agrees()` would have had nothing to say — the arm would fire the wrong
/// kernel with the right geometry, and §21.14's measurements say what that
/// looks like: a wrong arm that is 99.83% of the right answer, or one that
/// moves 34,273 of 55,200 cells while writing the same number of non-zero
/// values. **A permutation, which no count or norm would flag.**
///
/// So this asserts the property `agrees()` cannot: an arm's row must name the
/// base's symbol, with only a `#suffix` between them. That is checkable
/// without knowing anything about geometry, and it holds however the list is
/// reordered.
#[test]
fn an_arm_names_a_variant_of_its_own_base() {
    use kernels_cuda_new::device;

    let mut checked = 0_u32;
    let mut wrong: Vec<String> = Vec::new();
    for spec in device::SPECIALISED.iter().flat_map(|f| f.iter()) {
        for arm in spec.arms {
            checked += 1;
            let named = arm.row.sig.symbol;
            // A `#` variant of the base, or the base itself.
            let stem = named.split('#').next().unwrap_or(named);
            if stem != spec.base {
                wrong.push(format!(
                    "`{}` arm `{}` points at `{named}`, whose stem `{stem}` is not the base -- \
                     an index into a row list moved under it",
                    spec.base, arm.name
                ));
            }
        }
    }
    assert!(
        checked > 0,
        "no specialisation has an arm, so this test verified the empty set"
    );
    assert!(
        wrong.is_empty(),
        "an arm names a kernel that is not a variant of its base:\n  {}",
        wrong.join("\n  ")
    );
    eprintln!("{checked} specialisation arm(s) name a variant of their own base");
}
