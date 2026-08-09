//! `mlp`'s JIT units — the gated activations, and AltUp's sparsity.
//!
//! Two units, twenty-one `__global__` templates, sixteen rows. The device
//! text is `kernels-cuda-new/csrc/src/mlp/swiglu.cuh` and
//! `.../mlp/gaussian_topk.cuh`, which the ahead-of-time `.cu` twins over in
//! `kernels-cuda/csrc/src` now include so the
//! ahead-of-time archive holds exactly ONE definition of each kernel. Two
//! copies that agree today drift tomorrow, each right for whichever half its
//! tests exercise; `norm/altup_aux` shipped exactly that for a release.
//!
//! # Nineteen rows became sixteen, and the three that went were duplicates
//!
//! `new-horizon.md` §28.4 measured that 40% of the table's unreached rows are
//! a second name for a job a reached row already does, and four of them were
//! this family's: `mlp::gpt_oss_glu_strided_bf16` beside
//! `mlp::gpt_oss_glu_bf16` (2 goldens), `mlp::chunked_swiglu_strided_bf16`
//! beside `mlp::chunked_swiglu_bf16` (17 goldens), and BOTH
//! `mlp::sigmoid_scalar_gate_add_bf16` and
//! `mlp::sigmoid_scalar_gate_strided_add_bf16` beside
//! `mlp::sigmoid_dot_scalar_gate_add_bf16` (2 goldens). Nothing named any of
//! the four — no model text, no lowering, no golden, no driver fire, no test
//! — and §28's cause is why: the DSL surface was generated FROM THE LAUNCHER
//! HEADERS, so a header declaring `gpt_oss_glu_strided_bf16` beside
//! `gpt_oss_glu_bf16` got two wrappers, two rows and two migration tickets.
//! They are gone from `table::mlp`, from `SWIGLU_SIGS` and from `dsl.rs`;
//! `mlp/swiglu.cuh` keeps their templates, because a `.cuh` template with no
//! row is uninstantiated text and not a second definition.
//!
//! # What the rules recovered
//!
//! The rows carry one operand fewer than their ahead-of-time twins for every
//! stream — a stream is `cuLaunchKernel`'s sixth PARAMETER, outside the
//! `void**` — plus the extents the rules recover: the `n`s the
//! `Elementwise`/`ElementwiseRows` grids compute. Three `gate_second`
//! booleans became names.
//!
//! What stayed — `I`, `H`, `cols`, `row_stride`, `stride` — is either a bound
//! the kernel TESTS or an address it computes, which is layout, which is the
//! kernel's. `n` on the flat activations stayed for the same reason: the
//! `Elementwise` grid rounds UP, so the last block runs threads the buffer
//! does not have and the kernel's guard is what stops them.
//!
//! # The three kernels with no row
//!
//! **Re-audited at `LaunchRule` 21 → 28.** All three refusals stand, and the
//! eight rules §21.13 added do not touch them. The nearest is `Slab`, which
//! divides an ELEMENT count by 8 and caps the quotient at 1024; this grid
//! divides a half-WIDTH by `BLOCK` and caps nothing, and the two agree at no
//! input. The host predicates below are untouched by any of them.
//!
//! `swiglu.cuh` carries three vectorised kernels and `swiglu.cu` still
//! launches them. Their grid is `ceil(((I + 1) / 2) / BLOCK)` — a HALF-WIDTH
//! extent no `LaunchRule` states — and the launcher picks between them and
//! their scalar twins on `I > 10000` and on the parity of `row_stride`, which
//! are predicates over an operand's VALUE rather than a place a `Source` can
//! name. Inventing a rule for three kernels would put a geometry in the
//! vocabulary that only those three mean, so they are carried as device text
//! and left unmigrated.
//!
//! # A row is a contract naming a SYMBOL, and that is why the ROW went
//!
//! `mlp::sigmoid_scalar_gate_add_bf16` was once left unwritten here on the
//! grounds that it named the same instantiation as
//! `mlp::sigmoid_scalar_gate_strided_add_bf16`, and that reading was
//! overturned: `norm_device`'s bijection is between that one table and its
//! kernels, nothing tree-wide forbids two rows naming one instantiation, and
//! `rope::device::rotate_partial<device::bf16>` is named by three
//! [`crate::families::rope::ROPE_ROWS`] entries. That argument still stands
//! and it is not what removed the symbol. What removed it is the answer to
//! the question one level up — **does anything write this symbol into a
//! trace?** — and §28.9 measured the answer as no, for both spellings, with
//! `dsl.rs`'s two wrappers at zero call sites and
//! `qwen_3_5/forward/mod.rs:65`'s doc naming a symbol `lower.rs:1571` does
//! not emit. A row is a contract naming a symbol; a symbol nothing names
//! needs no contract.
//!
//! # A second numeric format costs a row
//!
//! Every template here is `template <class T>` over `device::Elem<T>`, and
//! every row states `device::bf16` because bf16 is what this tree stores
//! activations in. An fp16 MLP is now sixteen lines, not sixteen
//! translation units — which is the measurement `norm/elementwise`'s
//! `residual_add_f16` made first and this family inherits.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Lit;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// `mlp`'s gated activations: eighteen rows over as many templates.
pub const SWIGLU: Unit = Unit {
    name: "mlp/swiglu",
    root: include_str!("../../csrc/src/mlp/swiglu.cuh"),
    rows: SWIGLU_ROWS,
    options: &[],
};

/// AltUp's activation sparsity, alone in its file because it always was.
pub const GAUSSIAN_TOPK: Unit = Unit {
    name: "mlp/gaussian_topk",
    root: include_str!("../../csrc/src/mlp/gaussian_topk.cuh"),
    rows: GAUSSIAN_TOPK_ROWS,
    options: &[],
};

/// The units `mlp` compiles.
pub static UNITS: &[Unit] = &[SWIGLU, GAUSSIAN_TOPK];

/// The instantiations `mlp/swiglu.cuh` is compiled for.
///
/// `chunked_swiglu` and `chunked_swiglu_gate_second` are two templates and
/// two rows for what C++ spelled `chunked_swiglu<GateSecond>`: the
/// instantiation a row states carries exactly ONE template argument, so the
/// packed layout's flag had to become part of a name. The dispatcher picks a
/// symbol, which is what it was already doing with a `bool` argument.
#[rustfmt::skip]
pub static SWIGLU_ROWS: &[DeviceKernel] = &[
    DeviceKernel { sig: &SWIGLU_SIGS[0],  template_path: "mlp::device::swiglu",                        elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[1],  template_path: "mlp::device::swiglu_clamp",                  elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[2],  template_path: "mlp::device::situ",                          elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[3],  template_path: "mlp::device::geglu_tanh",                    elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[4],  template_path: "mlp::device::relu2",                         elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[5],  template_path: "mlp::device::gpt_oss_glu",                   elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[6],  template_path: "mlp::device::sigmoid_gate_inplace",          elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[7],  template_path: "mlp::device::chunked_swiglu",                elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[8],  template_path: "mlp::device::chunked_swiglu_gate_second",    elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[9],  template_path: "mlp::device::chunked_swiglu_clamp",          elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[10], template_path: "mlp::device::chunked_situ",                  elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[11], template_path: "mlp::device::chunked_situ_gate_second",      elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[12], template_path: "mlp::device::chunked_geglu_tanh",            elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[13], template_path: "mlp::device::chunked_geglu_tanh_gate_second", elem: "device::bf16" },
    DeviceKernel { sig: &SWIGLU_SIGS[14], template_path: "mlp::device::sigmoid_dot_scalar_gate_add",   elem: "device::bf16" },
];

/// The contracts, in [`SWIGLU_ROWS`]' order.
///
/// Each is its ahead-of-time twin in [`crate::table::mlp`] minus the stream,
/// minus the extents the rule recovers, plus the `launch` the twin's C++
/// launcher held inside its `<<<>>>`. The `Source`s are the twins' own:
/// nothing about how an operand is SOURCED changes when a launcher becomes a
/// rule, only which operands there are.
#[rustfmt::skip]
static SWIGLU_SIGS: [KernelSig; 15] = [
    // `Elementwise` IS the launcher: `(n + 255) / 256` blocks of 256, and an
    // empty `n` refused rather than launched. `n` stays an operand because
    // the grid rounds UP — the last block runs threads the buffer does not
    // have, and the kernel's guard is what stops them.
    kernel!(swiglu "mlp::swiglu_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            gate: Buf <- Source::In(0),
            // TWO SPELLINGS OF THE UP PROJECTION. A trace that split the
            // packed projection states both halves; one that did not leaves
            // `up` to the join, which collected it as the statement's
            // foreign operand.
            up: Buf <- Source::Or(&Source::In(1), &Source::Aux(0)),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
        ]),
    kernel!(swiglu_clamp "mlp::swiglu_clamp_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            gate: Buf <- Source::In(0),
            up: Buf <- Source::Or(&Source::In(1), &Source::Aux(0)),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
            limit: F32 <- Source::Ctx("glu_limit"),
        ]),
    // SiTU is not a swiglu variant: the tanh saturates far enough out that a
    // bf16 intermediate loses the distinction the gate exists to make.
    kernel!(situ "mlp::situ_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            gate: Buf <- Source::In(0),
            up: Buf <- Source::Or(&Source::In(1), &Source::Aux(0)),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
            beta: F32 <- Source::Ctx("situ_beta"),
            linear_beta: F32 <- Source::Ctx("situ_linear_beta"),
        ]),
    // GeGLU-tanh is not a swiglu variant either: `gelu_pytorch_tanh` on the
    // gate is a different function. gemma-4's PLE gate states a `select` of
    // the per-layer relay as `up`, so the layer offset an arm used to add is
    // a placement the host makes — which is what let this row be stated at
    // all.
    kernel!(geglu_tanh "mlp::geglu_tanh_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            gate: Buf <- Source::In(0),
            up: Buf <- Source::In(1),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
        ]),
    kernel!(relu2 "mlp::relu2_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            x: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
        ]),
    // OPERAND ORDER MOVED, and only here. The C++ launcher took `y_fp16`
    // LAST because it was a defaulted argument, which is a fact about C++
    // call syntax and not about the kernel; the kernel takes it fourth,
    // beside the output it parallels. A row states the KERNEL's order,
    // because the `void**` it builds is the kernel's parameter list.
    //
    // The second output is an operand and not a format: the MXFP4 decode
    // GEMV reads fp16, and emitting it from the same fp32 the bf16 rounds
    // from is what deleted a cast launch. `Lit::Null` is how a row says
    // "absent" for a pointer the kernel tests.
    //
    // # `y_fp16` is `f16*` in the header and `bf16*` in this row
    //
    // Measured rather than reasoned: a function POINTER admits no parameter
    // conversion whatever, and initialising one from
    // `mlp::device::gpt_oss_glu<device::bf16>` compiled under nvcc 13.0
    // `-arch=sm_89` against `(const bf16*, const bf16*, bf16*, f16*, i32,
    // float, float)` and refused the same list with `bf16*` fourth. `BufMut`
    // takes its element from the row's `elem` and no `Ty` carries one of its
    // own, so `bf16*` is the only thing a row can say about this parameter
    // and it is not true.
    //
    // Inert, and only because of the line below it: `Lit::Null` is what
    // crosses, so nothing is allocated for the operand and nothing is stored
    // through it. Source it — a second output, the decode GEMV wanting its
    // fp16 without a cast launch — and the row's word becomes the
    // allocation's width: halves into bf16 slots, every address legal, no
    // fault and no wrong number until something reads the tensor.
    // `tests/units.rs` is structurally blind to this, because NVRTC proves
    // an instantiation EXISTS and never that a parameter list matches. The
    // repair is a `Ty` carrying its own element — `crates/kernels`' to add,
    // and not this row's to paper over.
    kernel!(gpt_oss_glu "mlp::gpt_oss_glu_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            gate: Buf <- Source::In(0),
            up: Buf <- Source::In(1),
            y: BufMut <- Source::Out(0),
            y_fp16: BufMut | null <- Source::Lit(Lit::Null),
            num_elements: I32 <- Source::OutElements(0),
            limit: F32 <- Source::ParamF32(0),
            // The one the arm let DEFAULT. A generated call passes every
            // operand, so the row spells what the header's default was —
            // which is the better place for it anyway: a default in a header
            // is a fact about the launcher that no caller can see it
            // relying on.
            alpha: F32 <- Source::Lit(Lit::F32(1.702)),
        ]),
    // The twin of this one lives in `crate::table::driver_internal`, not in
    // `crate::table::mlp`: the gate is EMITTED by the model rather than
    // stated by a trace. Same kernel, same operands.
    kernel!(sigmoid_gate_inplace "mlp::sigmoid_gate_inplace_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            gate: Buf <- Source::In(1),
            num_elements: I32 <- Source::OutElements(0),
        ]),
    // The twin says `OutRows` and not `Rows`, and the reason survives the
    // move: THREE callers share this kernel and one is the routed MoE leg,
    // whose rows are the padded block-major count rather than the fire's
    // tokens. `ElementwiseRows` reads the OUTPUT's rectangle, which is the
    // same number that row bound.
    kernel!(chunked_swiglu "mlp::chunked_swiglu_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::ElementwiseRows,
        in_place = &[(0, 1)],
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            i: I32 <- Source::OutWidth(0),
        ]),
    // The gate-second half of the same kernel, as a SYMBOL rather than as a
    // `bool` operand: `Source::Lit(Lit::Bool(false))` is what the twin
    // passes, and a literal that only ever takes two values is a name.
    kernel!(chunked_swiglu_gate_second "mlp::chunked_swiglu_gate_second_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::ElementwiseRows,
        in_place = &[(0, 1)],
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            i: I32 <- Source::OutWidth(0),
        ]),
    kernel!(chunked_swiglu_clamp "mlp::chunked_swiglu_clamp_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            i: I32 <- Source::OutWidth(0),
            limit: F32 <- Source::Ctx("glu_limit"),
        ]),
    kernel!(chunked_situ "mlp::chunked_situ_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            i: I32 <- Source::OutWidth(0),
            beta: F32 <- Source::Ctx("situ_beta"),
            linear_beta: F32 <- Source::Ctx("situ_linear_beta"),
        ]),
    // Where the twin bound `gate_second` from `Source::Ctx("gate_second")`,
    // the dispatcher now reads that key and picks a symbol. The branch
    // leaves the inner loop, which is what the `bool` template parameter was
    // for before an ahead-of-time build had to name its instantiations.
    kernel!(chunked_situ_gate_second "mlp::chunked_situ_gate_second_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            i: I32 <- Source::OutWidth(0),
            beta: F32 <- Source::Ctx("situ_beta"),
            linear_beta: F32 <- Source::Ctx("situ_linear_beta"),
        ]),
    kernel!(chunked_geglu_tanh "mlp::chunked_geglu_tanh_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            i: I32 <- Source::OutWidth(0),
        ]),
    kernel!(chunked_geglu_tanh_gate_second "mlp::chunked_geglu_tanh_gate_second_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            i: I32 <- Source::OutWidth(0),
        ]),
    // `Rms` to the byte: `<<<N, 256, (256 / 32) * sizeof(float)>>>`. The
    // dynamic shared memory is not incidental here — the kernel declares
    // `extern __shared__` and folds its dot product through exactly those
    // eight floats.
    kernel!(moe_shared_gate_dot "mlp::sigmoid_dot_scalar_gate_add_bf16",
        file = Some("mlp/swiglu.cuh"),
        launch = LaunchRule::Rms,
        in_place = &[(0, 1)],
        operands = operands![
            x: Buf <- Source::In(0),
            gate_w: Buf <- Source::Weight(0),
            out: BufMut <- Source::Out(0),
            // The ADDEND, operand 2, not the accumulator. The hand-written
            // arm carried a warning that reversing the last two lands the
            // gate on the wrong buffer and still compiles; a row states the
            // order once instead.
            y: Buf <- Source::In(2),
            h: I32 <- Source::OutWidth(0),
        ]),
];

/// The instantiation `mlp/gaussian_topk.cuh` is compiled for.
pub static GAUSSIAN_TOPK_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &GAUSSIAN_TOPK_SIGS[0],
    template_path: "mlp::device::gaussian_topk",
    elem: "device::bf16",
}];

/// The contract, in [`GAUSSIAN_TOPK_ROWS`]' order.
#[rustfmt::skip]
static GAUSSIAN_TOPK_SIGS: [KernelSig; 1] = [
    // `Rms` exactly, dynamic shared memory included: the launcher was
    // `<<<N, 256, (256 / 32) * sizeof(float)>>>` and the kernel folds two
    // block-wide reductions through those bytes. `n` is gone — the grid was
    // one block per row and the kernel reads `blockIdx.x` with no guard, so
    // the token count was pure geometry.
    kernel!(gaussian_topk "mlp::gaussian_topk_bf16",
        file = Some("mlp/gaussian_topk.cuh"),
        launch = LaunchRule::Rms,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            dim: I32 <- Source::OutWidth(0),
            // Per-layer, and the driver's own derivation: the config states
            // `activation_sparsity` and the kernel wants
            // `gaussian_inverse_cdf` of it.
            std_multiplier: F32 <- Source::CtxByLayer("altup_std_mult"),
        ]),
];
