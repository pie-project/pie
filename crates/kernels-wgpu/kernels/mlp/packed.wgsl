// The packed activations: one file because they are one BINDING CONTRACT.
//
// `mlp/gated.wgsl` beside this one opens by stating its own law -- "one file
// because they are one BINDING CONTRACT", `(gate, up, out)` at `@group(0)`
// bindings 0, 1 and 2 in every variant -- and these five do not fit under it.
// They take the ONE row a fused gate/up projection wrote and cut it
// themselves:
//
//   packed[n, 0 .. I)    the gate half
//   packed[n, I .. 2I)   the up half
//
// so the contract here is `(packed, out)` at 0 and 1, with `I` and whatever
// else the activation needs in the `@group(1)` uniform block. That is a
// DIFFERENT contract, not a sixth arithmetic under the existing one, and the
// two files are two for exactly that reason: a kernel from either bound
// against the other's operands does not fault -- it reads the gate half as a
// whole activation and returns a plausible wrong number.
//
// `kernels-metal/kernels/mlp/packed.metal` split for the same reason and says
// so at the same length; this file is its sibling and not its port.
//
//   packed_swiglu         y = silu(g) * u                       I
//   packed_swiglu_clamp   both halves clamped, then SwiGLU      I, limit
//   packed_gptoss_swiglu  gpt-oss's asymmetric clamp and alpha  I, limit, alpha
//   packed_geglu_tanh     y = gelu_tanh(g) * u                  I
//   packed_situ           SiTU's tanh-saturated gate            I, beta, up_cap
//
// THE GATE HALF IS FIRST IN ALL FIVE. `kernels-cuda/kernels/mlp/swiglu.cuh`
// carries a `GateSecond` template parameter and a `_gate_second` twin for
// three of these, because some checkpoints export `[up | gate]`. No point on
// the declaration floor states which order it holds, so a second entrypoint
// here would be a name nothing can ask for, reading like a choice.
//
// ## The grid is the extent, and the extent is in WORDS
//
// `dispatch_workgroups` counts WORKGROUPS, so an extent that is not a multiple
// of 256 leaves invocations past the end -- the fact `gated.wgsl` sets out and
// the reason nothing here takes a row count. The bound is `arrayLength(&out_)`,
// the bound storage range, which needs nothing from the caller and cannot
// drift from a scalar the statement would have to grow.
//
// And one invocation owns one WORD, which is TWO bf16 values, because WGSL has
// no 16-bit storage (`common/bf16.inc.wgsl`). So the launch's x extent is HALF
// the output's element count and the store is one plain write -- no atomic,
// no compare-exchange.
//
// THE TWO ELEMENTS OF A WORD NEED NOT BE THE SAME ROW, and that is what makes
// the plain store correct for every `I`. The output is densely `[rows, I]`, so
// a flat element index is all the addressing there is; word `w` owns elements
// `2w` and `2w + 1`, and each is decomposed into its OWN `(row, col)` below.
// An odd `I` therefore puts a row boundary inside a word and nothing has to
// know: both halves still have exactly one writer, which is this invocation.
// Deriving the row per ELEMENT rather than per INVOCATION is the whole of the
// difference from `gated.wgsl`'s strided arm, which cannot do it -- there the
// three operands have three pitches, so an output word's two halves are two
// different lanes' and the store has to be a compare-exchange.
//
// The last word's upper half is past the tensor when `rows * I` is odd. It is
// computed and stored anyway: `row` comes out as `rows`, the reads land past
// `packed`, and naga's `Restrict` bounds policy clamps them to a real word --
// a defined value written into padding no reader has. Guarding it would cost a
// stated row count to make the tail a special case of a bound the buffer
// already carries.
//
// `I` is bound all the same, and it is not a second bound: it is the stride
// from a row's gate half to its up half and the divisor that recovers the row.
// `swiglu.cuh` draws the line this file draws -- an extent the grid computes is
// geometry and belongs to the fire, while an address the kernel computes is
// layout and belongs to the kernel -- and `I` is on the layout side of it.
// `kernels_wgpu::mlp` refuses `I = 0` before the fire, which is what keeps the
// division below defined.

//#include "common/bf16.inc.wgsl"
//#include "common/math.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

// The uniform block, field for field in the order `kernels_wgpu::mlp` fires
// its marks. A block's layout on this backend is the order the BODY passed its
// scalars, so a field moved here is a number read at the wrong offset there
// and everything after it shifts by four bytes; nothing reports that, because
// the block is bytes and neither wgpu nor a validation layer knows what they
// were meant to be.
struct Params {
    intermediate: u32,
//#if defined(PIE_CLAMP) || defined(PIE_GPTOSS)
    limit: f32,
//#endif
//#if defined(PIE_GPTOSS)
    alpha: f32,
//#endif
//#if defined(PIE_SITU)
    beta: f32,
    up_cap: f32,
//#endif
}
@group(1) @binding(0) var<uniform> params: Params;

//#if defined(PIE_GEGLU)
// The TANH approximation of gelu, not the erf one: gemma's activation is
// specified as this closed form and the two differ by more than rounding. The
// cubic coefficient is the canonical 0.044715 that
// `torch.nn.functional.gelu(approximate="tanh")` uses, which is HF's
// `gelu_pytorch_tanh`.
//
// `pie_tanh` and not the builtin, for the reason `common/math.inc.wgsl` gives
// at length and measured on this family: `inner` crosses 44.36 at a GATE OF
// 10.5, an ordinary FFN activation, and WGSL's `tanh` is NaN past there.
fn gelu_tanh(x: f32) -> f32 {
    let k = 0.7978845608028654;  // sqrt(2/pi)
    let inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + pie_tanh(inner));
}
//#endif

// The activation, and the only place any of them is written.
//
// Every arm widens once, computes in f32 and rounds once -- which is what
// `swiglu.cuh` does on the sibling plane and what every packed form in this
// tree does. `gated.wgsl`'s `silu_mul` rounds its sigmoid to bf16 before the
// multiply, because the SPLIT kernel it serves was transcribed from MLX and a
// parity walk against `kernels-metal` has to make the same choice there; that
// is a fact about that entrypoint's provenance and not a convention this one
// inherits.
fn activate(g_in: f32, u_in: f32) -> f32 {
    var g = g_in;
    var u = u_in;
//#if defined(PIE_GEGLU)
    return gelu_tanh(g) * u;
//#elif defined(PIE_SITU)
    // SiTU: `beta * tanh(g / beta) * sigmoid(g)`, with an optional tanh
    // soft-cap on the up half. Not a SwiGLU variant -- the tanh saturates far
    // enough out that the gate is bounded by `beta` rather than by the logit,
    // which is the point of it, and it is why the whole computation stays in
    // f32: rounding the inner `g / beta` to bf16 first loses exactly the
    // distinction the saturation exists to make.
    //
    // `up_cap <= 0` means NO CAP, which is how a statement with no soft-cap
    // asks for the plain product without a second entrypoint. `beta` is never
    // zero: `kernels_wgpu::mlp` refuses that before the fire, because the
    // division is the gate.
    let s = params.beta * pie_tanh(g / params.beta) / (1.0 + exp(-g));
    if (params.up_cap > 0.0) {
        u = params.up_cap * pie_tanh(u / params.up_cap);
    }
    return s * u;
//#elif defined(PIE_GPTOSS)
    // gpt-oss's GLU. THE SAME ARITHMETIC AS `gated.wgsl`'s `gptoss_swiglu`,
    // SPELLED THE SAME WAY -- `(gc * sig) * (uc + 1.0)` and not
    // `(u + 1) * g * sig` -- because the only difference between the two entry
    // points is where the two halves came from, and a second entry into one
    // activation is worth having only if the two agree bit for bit.
    //
    // The transcription is the discipline and not what a comparison can reach:
    // at bf16 two spellings of one product usually round to the same eight
    // mantissa bits. What a comparison does catch is a symmetric clamp on the
    // gate, a dropped `alpha` or a swapped half, and keeping the spelling
    // identical is what leaves those as the only ways the two can drift.
    //
    // "USUALLY" IS MEASURED, and it is the one place this arm parts from
    // `pie::mlp::chunked_gpt_oss_glu` on an L40S. Over 917 elements of a
    // [-12, 12] packed row at `limit = 7`, `alpha = 1.702`, exactly ONE came
    // back a bf16 ulp apart: `g = -11.6875`, `u = 11.375`, where `exp(19.89)`
    // decides a product of -2.1467e-07 that sits within 1e-5 of the tie
    // between two bf16 codes. WGSL's `exp` is the device's approximate
    // exponential and cuda's `expf` is not, so the two land on opposite sides
    // of that tie -- and the correctly rounded answer is THIS one. Every other
    // element of all five activations, `expf` and `__expf` sites alike, agreed
    // bit for bit.
    g = min(g, params.limit);
    u = clamp(u, -params.limit, params.limit);
    let sig = 1.0 / (1.0 + exp(-params.alpha * g));
    return (g * sig) * (u + 1.0);
//#elif defined(PIE_CLAMP)
    // The gate is clamped ABOVE ONLY and the up half BOTH WAYS, which is not a
    // symmetry anyone should restore: a gate clamped from below saturates the
    // branch the activation exists to switch off, and the model still runs.
    g = min(g, params.limit);
    u = clamp(u, -params.limit, params.limit);
    return (g / (1.0 + exp(-g))) * u;
//#else
    // `silu(g) = g * sigmoid(g)`, spelled as the division
    // `g / (1 + exp(-g))` that `pie::mlp::chunked_swiglu` spells it as.
    return (g / (1.0 + exp(-g))) * u;
//#endif
}

// One output element, by its flat index over the dense `[rows, I]` result.
//
// The row is recovered per ELEMENT, which is what lets one invocation own a
// whole word whatever `I` is; see the header.
fn value_at(e: u32) -> f32 {
    let half = params.intermediate;
    let row = e / half;
    let col = e - row * half;
    // The packed row is `2I` wide, so a row's gate half begins at `2I * row`
    // and its up half `I` further on. Folding the two into one offset is the
    // classic way to read the gate half as a whole activation.
    let gate_at = row * 2u * half + col;
    let up_at = gate_at + half;
    return activate(
        pie_bf16_at(packed[gate_at >> 1u], gate_at),
        pie_bf16_at(packed[up_at >> 1u], up_at),
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    // The buffer's own length and not a stated count: `dispatch_workgroups`
    // rounds the group count up, so the last group runs past the data.
    if (w >= arrayLength(&out_)) { return; }
    out_[w] = pie_pack_bf16(value_at(w * 2u), value_at(w * 2u + 1u));
}

// pie:instantiate packed_swiglu_bfloat16
// pie:instantiate packed_swiglu_clamp_bfloat16 PIE_CLAMP=1
// pie:instantiate packed_gptoss_swiglu_bfloat16 PIE_GPTOSS=1
// pie:instantiate packed_geglu_tanh_bfloat16 PIE_GEGLU=1
// pie:instantiate packed_situ_bfloat16 PIE_SITU=1
