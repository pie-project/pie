//! Feed-forward activations: the SwiGLU/GeGLU/SiTU spellings and their clamps.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::operands;
use kernels::Lit;
use kernels::Source;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // Two spellings of one arithmetic, and the BINDING picks: a packed
    // gate‖up bank feeds the chunked form, two narrow buffers the pair
    // form. A load-time fact, so the declaration states it.
    // The ALIGNED leg's spelling states a second operand — the staging
    // the pointer build named — and writes its result over it. The dense
    // and shared-expert spellings state one operand, and a pair outside
    // a statement's arity is not an error (`lower::Buffers`), so one row
    // serves all three.
    kernel!(chunked_swiglu "mlp::chunked_swiglu_bf16",
        in_place = &[(0, 1)],
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            // `OutRows`, not `Rows`: THREE callers share this kernel and
            // one of them is the routed MoE leg, whose rows are the
            // padded block-major count rather than the fire's tokens.
            // Binding the fire's would have activated the first N of the
            // padded rows and left the rest holding whatever the GEMM
            // wrote — which is what the arm computing `aligned_rows` by
            // hand was for, and what the row says instead.
            n: I32 <- Source::OutRows(0),
            i: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
            gate_second: Bool <- Source::Lit(Lit::Bool(false)),
        ]),
    kernel!(swiglu "mlp::swiglu_bf16",
        operands = operands![
            gate: Buf <- Source::In(0),
            up: Buf <- Source::In(1),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The stride is the PACKED value's own row width — a row holds the
    // gate and the up halves and only `i` of the result comes out of it,
    // which is the same shape `norm::rmsnorm_strided_bf16` states.
    kernel!(chunked_swiglu_strided "mlp::chunked_swiglu_strided_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            n: I32 <- Source::OutRows(0),
            i: I32 <- Source::OutWidth(0),
            row_stride: I32 <- Source::InWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(gpt_oss_glu_strided "mlp::gpt_oss_glu_strided_bf16",
        operands = operands![
            gate: Buf,
            up: Buf,
            y: BufMut,
            rows: I32,
            cols: I32,
            in_stride: I32,
            out_stride: I32,
            stream: Stream,
            limit: F32,
            alpha: F32,
        ]),
    kernel!(swiglu_clamp "mlp::swiglu_clamp_bf16",
        operands = operands![
            gate: Buf <- Source::In(0),
            up: Buf <- Source::In(1),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
            limit: F32 <- Source::Ctx("glu_limit"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(chunked_swiglu_clamp "mlp::chunked_swiglu_clamp_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            n: I32 <- Source::OutRows(0),
            i: I32 <- Source::OutWidth(0),
            limit: F32 <- Source::Ctx("glu_limit"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(relu2 "mlp::relu2_bf16",
        operands = operands![
            x: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // SiTU is not a swiglu variant: the tanh saturates far enough out that a
    // bf16 intermediate loses the distinction the gate exists to make.
    kernel!(situ "mlp::situ_bf16",
        operands = operands![
            gate: Buf <- Source::In(0),
            up: Buf <- Source::In(1),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
            beta: F32 <- Source::Ctx("situ_beta"),
            linear_beta: F32 <- Source::Ctx("situ_linear_beta"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(chunked_situ "mlp::chunked_situ_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            n: I32 <- Source::OutRows(0),
            i: I32 <- Source::OutWidth(0),
            beta: F32 <- Source::Ctx("situ_beta"),
            linear_beta: F32 <- Source::Ctx("situ_linear_beta"),
            gate_second: Bool <- Source::Ctx("gate_second"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(gaussian_topk "mlp::gaussian_topk_bf16",
        operands = operands![
            x: BufMut,
            n: I32,
            dim: I32,
            std_multiplier: F32,
            stream: Stream,
        ]),
    // GeGLU-tanh is not a swiglu variant: `gelu_pytorch_tanh` on the
    // gate is a different function. The packed/pair split is the same
    // binding question.
    // The PAIR form: `(gate, up, out)` with `out` over `gate`. gemma-4's
    // PLE gate is the same call with the relay slice as `up`.
    kernel!(geglu_tanh "mlp::geglu_tanh_bf16", in_place = &[(0, 0)],
        operands = operands![
            gate: Buf <- Source::In(0),
            // gemma-4's PLE gate states a `select` of the per-layer
            // relay here, so the layer offset the arm used to add is a
            // placement the host makes. That is what let this row be
            // stated at all: with the whole table as operand 1 there was
            // no expression for "plus l · N · ple_dim".
            up: Buf <- Source::In(1),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(chunked_geglu_tanh "mlp::chunked_geglu_tanh_bf16",
        operands = operands![
            packed: Buf <- Source::In(0),
            y: BufMut <- Source::Out(0),
            n: I32 <- Source::Rows,
            i: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
            gate_second: Bool <- Source::Lit(Lit::Bool(false)),
        ]),
    // SwiGLU with a clamp. `swiglu_limit` is a config constant, so this
    // is a different kernel and not a different argument.
    // `gate = glu(gate, up)` -- the gate half is the destination, which
    // is why the driver passes its pointer twice.
    kernel!(gpt_oss_glu "mlp::gpt_oss_glu_bf16", in_place = &[(0, 0)],
        operands = operands![
            gate: Buf <- Source::In(0),
            up: Buf <- Source::In(1),
            y: BufMut <- Source::Out(0),
            num_elements: I32 <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
            limit: F32 <- Source::ParamF32(0),
            // The two the arm let DEFAULT. A generated call passes every
            // operand, so the row spells what the header's defaults are
            // — which is the better place for them anyway: a default in
            // a header is a fact about the launcher that no caller can
            // see it relying on.
            alpha: F32 <- Source::Lit(Lit::F32(1.702)),
            y_fp16: BufMut <- Source::Lit(Lit::Null),
        ]),
    kernel!(sigmoid_scalar_gate_add "mlp::sigmoid_scalar_gate_add_bf16",
        operands = operands![
            out: BufMut,
            x: Buf,
            scalar_gate: Buf,
            n: I32,
            h: I32,
            stream: Stream,
        ]),
    kernel!(sigmoid_scalar_gate_strided_add "mlp::sigmoid_scalar_gate_strided_add_bf16",
        operands = operands![
            out: BufMut,
            x: Buf,
            scalar_gate: Buf,
            n: I32,
            h: I32,
            stride: I32,
            stream: Stream,
        ]),
    // The shared expert's landing: `out += sigmoid(x . gate) * y`, and
    // `out` IS the residual stream the statement takes as operand 1 --
    // the header calls it "in-place add destination" in as many words.
    kernel!(moe_shared_gate_dot "mlp::sigmoid_dot_scalar_gate_add_bf16",
        in_place = &[(0, 1)],
        operands = operands![
            x: Buf <- Source::In(0),
            gate_w: Buf <- Source::Weight(0),
            out: BufMut <- Source::Out(0),
            // The ADDEND, operand 2, not the accumulator. The hand arm
            // carried a warning that reversing the last two lands the
            // gate on the wrong buffer and still compiles; a row states
            // the order once instead.
            y: Buf <- Source::In(2),
            n: I32 <- Source::Rows,
            h: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
];
