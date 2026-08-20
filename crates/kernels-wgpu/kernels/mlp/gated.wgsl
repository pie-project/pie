// The dense FFN gated activations: one file because they are one BINDING
// CONTRACT.
//
// `gate`, `up`, `out` at `@group(0)` bindings 0, 1, 2 in every variant, and a
// params struct at 3 for the TWO that still have one -- `silu_mul`, its
// strided form and the dense GeGLU state no params at all, which is why that
// buffer is declared inside the arms and not above them. Five entrypoints,
// three activations:
//
//   silu_mul       out = silu(gate) * up                    no params
//   geglu_tanh     out = gelu_tanh(gate) * up               no params
//   gptoss_swiglu  gpt-oss's clamped, alpha-scaled SwiGLU   GptOssSwiGluParams
//
// The third earns its model name: it bakes gpt-oss's asymmetric clamp, its
// `alpha` and its `(up + 1)` term, which is nobody else's SwiGLU, and dropping
// either produces a model that runs and is wrong.
//
// ## Five variants behind one contract is where a preprocessor slip lands
//
// A `//#if` arm that selected the wrong activation would compile, bind, launch
// and produce plausible numbers. So the arithmetic sits in ONE function,
// `activate`, whose arms are the three closed forms and nothing else, and the
// geometry sits outside it -- the strided pair differ from their flat siblings
// in ADDRESSING only, so they must not be able to differ in activation. The
// default arm is silu, which means a new variant that forgets its define
// silently becomes a SwiGLU: state the define on the `pie:instantiate` line.
//
// ## The grid is the extent, and the extent is in WORDS
//
// Metal launches these with `dispatchThreads`, an exact thread count, which is
// why `GegluParams::unused` was unused -- and, in the end, why the struct went
// away with the field and the dense GeGLU takes no params at all, which is the
// shape `silu_mul` has had from the start: there is no tail to guard on that
// backend. `dispatch_workgroups` counts WORKGROUPS, so the real extent here is
// `256 * ceil(n / 256)` and any width that is not a multiple of the workgroup
// leaves invocations past the end. The bound is `arrayLength(&out_)` -- the
// bound storage range, which needs nothing from the caller and cannot drift
// from a scalar the row would have to grow.
//
// And one invocation owns one WORD, which is TWO bf16 values: WGSL has no
// 16-bit storage, so the launch's x extent is HALF the element count. That is
// `norm/residual_add.wgsl`'s convention and it is the whole tree's. In the four
// flat variants that ownership is total and the store is one write. The strided
// GeGLU is the exception -- three pitches, so only the OUTPUT word is a unit of
// ownership and a ragged edge can be shared with the next row -- and it pays
// for that with an `atomic<u32>` output and a compare-exchange on the edge.

//#include "common/bf16.inc.wgsl"
//#include "common/math.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> gate: array<u32>;
@group(0) @binding(1) var<storage, read_write> up: array<u32>;
//#if defined(PIE_GEGLU_STRIDED)
// Atomic in this variant ALONE: it is the only one whose invocation can own
// half a word. See `store_half`. The host binds the same read_write storage
// buffer of 4-byte words either way, so the row's ABI does not move.
@group(0) @binding(2) var<storage, read_write> out_: array<atomic<u32>>;
//#else
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;
//#endif

// THE DENSE GEGLU DECLARES NO PARAMS BINDING AT ALL, which is why this chain
// opens on the strided arm rather than on `PIE_GEGLU`.
//
// `@group(0) @binding(3)` under `PIE_GEGLU` was `struct GegluParams { unused:
// u32 }` and a `read_write` storage buffer to bind it with: a whole descriptor
// and a whole staged word for a struct with ONE field that nothing in this
// file ever read. The field was dead on arrival -- it was a per-row element
// count read as `if (gid >= p.n) return;`, and "the grid is the extent" above
// is the whole of why that bound could not stay. What kept the BINDING alive
// after the field died was the row, which stated `params: Buf`; a shell built
// the bind group layout out of the row, so every backend declared a slot for
// it whether or not its body read one, and on this plane a declared slot must
// be filled or every buffer after it shifts by one.
//
// The row is retired and `mlp::geglu_tanh` no longer calls `ctx.params()`, so
// nothing states the buffer and nothing declares it. Deleting the DECLARATION
// rather than skipping the slot is what keeps `driver-wgpu`'s explicit layout
// and this body's three arguments the same number.
//
// The other two arms carried their numbers in a `binding(3)` storage struct for
// the same reason and no longer do. The strided form's five pitches and
// gpt-oss's `limit` and `alpha` ARE genuinely read -- they are numbers no grid
// can supply -- so unlike `GegluParams` they became `Const` marks rather than
// nothing, and the block they ride is the `@group(1)` uniform every other
// stated scalar in this tree uses. Same words, same order, no descriptor.
//#if defined(PIE_GEGLU_STRIDED)
// gemma4's per-layer-embedding GeGLU reads a NARROW gate out of a WIDE table:
// the PLE table is `[rows, n_layers * ple_dim]`, so layer L's slice is
// `ple_dim` wide with `n_layers * ple_dim` between rows, while the gate and the
// output are densely `[rows, ple_dim]`. A byte offset cannot express that, and
// the flat kernel reading one walks into the NEXT layers' slices after the
// first row -- not a crash and not even implausible numbers, since those slices
// are the same table.
// All five stated, in the order `mlp::geglu_tanh_strided` passes them. `width`
// and `rows` are the launch's own rectangle and the BODY also knows them --
// that is not a duplication introduced here, it is what the struct already
// held: the shader read the staged words while the grid came from the body,
// and both still do.
struct Params {
    width: u32,
    rows: u32,
    gate_pitch: u32,
    up_pitch: u32,
    out_pitch: u32,
}
@group(1) @binding(0) var<uniform> params: Params;
//#elif defined(PIE_GPTOSS)
// TWO WORDS, WHERE THE STRUCT HAD THREE.
//
// `GptOssSwiGluParams` opened with a per-row element count -- dead for the
// reason the note above the strided arm gives at length, the same number
// `GegluParams` held, and it outlived that struct only because `limit` and
// `alpha` beside it are read. A struct has to carry a dead field to keep the
// live ones at their offsets; a uniform packed from the marks the body PASSES
// does not, so the dead word no longer reaches the GPU at all.
//
// It is still stated: `gptoss_swiglu` declares a slot-holder mark for it,
// because `Const` slots are the statement's run counted in order and `limit`
// sits at word 1. That holder goes when the DSL stops stating the word.
struct Params {
    limit: f32,
    alpha: f32,
}
@group(1) @binding(0) var<uniform> params: Params;
//#endif

//#if defined(PIE_SILU_STRIDED)
// Vulkan sends this to a push block; WebGPU has none, so it is the one field of
// the uniform block. The placement follows `norm/residual_add.wgsl`, which is
// the same kernel shape with the same scalar, and `mlp::silu_mul_strided`'s
// signature is what states it.
struct Strided { row_pitch: i32 }
@group(1) @binding(0) var<uniform> strided: Strided;
//#endif

//#if !defined(PIE_GEGLU) && !defined(PIE_GEGLU_STRIDED) && !defined(PIE_GPTOSS)
// MLX's numerically stable sigmoid (`unary_ops.h` Sigmoid): the exponent is
// taken of `-|x|` so it cannot overflow, and the branch puts the reflection
// back.
fn sigmoid_mlx(x: f32) -> f32 {
    let y = 1.0 / (1.0 + exp(-abs(x)));
    return select(y, 1.0 - y, x < 0.0);
}
//#endif

//#if defined(PIE_GEGLU) || defined(PIE_GEGLU_STRIDED)
// The TANH approximation of gelu, not the erf one: gemma's activation is
// specified as this closed form and the two differ by more than rounding.
fn gelu_tanh(x: f32) -> f32 {
    let k = 0.7978845608028654;  // sqrt(2/pi)
    let inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + pie_tanh(inner));
}
//#endif

// The activation, and the only place any of them is written.
fn activate(g: f32, u: f32) -> f32 {
//#if defined(PIE_GEGLU) || defined(PIE_GEGLU_STRIDED)
    return gelu_tanh(g) * u;
//#elif defined(PIE_GPTOSS)
    // The gate is clamped ABOVE only; the linear branch is clamped both ways
    // and carries a `+1`. Both are gpt-oss's own.
    let gc = min(g, params.limit);
    let uc = clamp(u, -params.limit, params.limit);
    let sig = 1.0 / (1.0 + exp(-params.alpha * gc));
    return (gc * sig) * (uc + 1.0);
//#else
    // Metal rounds both intermediates to T, so this rounds through bf16 twice
    // as well: the sigmoid, and then the product with the gate. Doing the whole
    // thing in f32 and rounding once is a DIFFERENT number, and a parity walk
    // against the sibling has to make the same choice.
    let sg = pie_bf16_to_f32(pie_f32_to_bf16(sigmoid_mlx(g)));
    let sil = pie_bf16_to_f32(pie_f32_to_bf16(g * sg));
    return sil * u;
//#endif
}

//#if defined(PIE_GEGLU_STRIDED)

// The half-index split, one reader per binding. `pie_bf16_at` takes a WORD and
// not the buffer because core WGSL allows a pointer parameter only in the
// `function`, `private` and `workgroup` address spaces: a shared
// `load(&buffer, i)` PARSES and then fails validation, which is how the first
// draft of this tree shipped 478 unbuildable modules.
fn gate_at(i: u32) -> f32 {
    return pie_bf16_at(gate[i >> 1u], i);
}

fn up_at(i: u32) -> f32 {
    return pie_bf16_at(up[i >> 1u], i);
}

// One bf16 of a word this invocation does not own outright.
//
// Only an odd `out_pitch` reaches this, and it is the pitch of a table gemma4
// lays out as `[rows, n_layers * ple_dim]`, so the sharing partner is the NEXT
// ROW -- a different workgroup, since `gid.y` is the row. A read-modify-write
// keeps whichever landed second; the device-scoped compare-exchange keeps both
// and retries the spurious failure `...Weak` is permitted. Same pattern as
// `norm/rms.wgsl` and `kernels/quant/qmm_t.wgsl`.
fn store_half(i: u32, value: f32) {
    let at = i >> 1u;
    var old = atomicLoad(&out_[at]);
    loop {
        let res = atomicCompareExchangeWeak(&out_[at], old, pie_bf16_into(old, i, value));
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}

// 256x1 and flat, because that is what this row's rule says.
//
// # Why this was 16x16, and why that was wrong
//
// This body mirrored both siblings' `local_size` -- 16x16, reading `gid.y` as
// the row -- on the reasoning that PLE rows are NARROW (`ple_dim` is 256
// elements, 128 words) and a 256-wide workgroup leaves half of every row's
// lanes idle. The reasoning is fine and the shape was not: the ROW states
// `LaunchRule::Elementwise`, which is `[width * rows, 1, 1]`. One workgroup on
// y then covers 16 rows and every row past 15 is never dispatched. Measured at
// 21 rows on a 4090: row 16 came back holding the sentinel it was born with,
// and the dispatch SUCCEEDED. gemma's PLE reaches this with `rows` = the
// fire's token count, so any prefill longer than sixteen tokens was silently
// dropping most of its per-layer embeddings.
//
// The rule is shared with `kernels-metal`, where a threadgroup is sized at
// dispatch and `Elementwise` is correct, so changing the ROW would be a change
// to three tables and three drivers to fix one body. Changing the BODY to
// match the rule it already states is local, and this is that.
//
// `gid.x` is therefore a flat ELEMENT index over `rows * width`, not a word
// index. Word ownership is recovered below rather than assumed, so the fast
// whole-word store survives: exactly one lane writes each word.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.rows * params.width) { return; }
    let m = n / params.width;
    let k = n % params.width;

    // Three operands with three pitches, so only the OUTPUT's word is a unit of
    // ownership; the gate and the up rows are addressed per element and may sit
    // at any parity.
    //
    // Ownership is derived from the ABSOLUTE output offset rather than from
    // the lane number, because a row starts at `m * out_pitch` and an odd pitch
    // puts an odd row's first element in a word's UPPER half. So the three
    // cases below are: this lane owns the whole word, this lane owns one half
    // of a word it shares with a neighbouring ROW, or the lane before it
    // already wrote the word they share.
    //
    // Exactly one lane writes each word. The `k > 0` in the last case is what
    // makes that true: a lane on an odd absolute offset is the second half of a
    // pair the even lane already stored, UNLESS it is the row's first element,
    // in which case the even half belongs to the previous row.
    let base_o = m * params.out_pitch;
    let base_g = m * params.gate_pitch;
    let base_u = m * params.up_pitch;
    let at = base_o + k;
    let word = at >> 1u;
    let value = activate(gate_at(base_g + k), up_at(base_u + k));

    if ((at & 1u) == 0u) {
        if (k + 1u < params.width) {
            // Both halves are this row's and this lane's: one plain store, no
            // read-modify-write, nothing to race. The lane for `k + 1` sees an
            // odd offset with `k > 0` and returns.
            atomicStore(&out_[word], pie_pack_bf16(
                value,
                activate(gate_at(base_g + k + 1u), up_at(base_u + k + 1u)),
            ));
        } else {
            // The row's last element sits in a word's lower half. The upper
            // half is either padding -- nobody's, when `out_pitch > width` --
            // or the NEXT row's first element when the pitch is tight. The
            // compare-exchange is correct for both and does not have to know
            // which.
            store_half(at, value);
        }
    } else if (k == 0u) {
        // The row begins in this word's upper half, which only an odd
        // `out_pitch` produces. The lower half is the previous row's and that
        // row's lane is writing it concurrently, so this half goes through the
        // compare-exchange.
        store_half(at, value);
    }
}

//#else

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
//#if defined(PIE_SILU_STRIDED)
    // The pitch is in ELEMENTS and this invocation owns a WORD, so the pair it
    // writes is `2*i` and `2*i + 1` of the logical row -- the arithmetic
    // `norm/residual_add.wgsl` does, wanting an even pitch for the same reason:
    // an odd one puts two rows in one word, which no store granularity here can
    // make safe.
    let i = gid.y * u32(strided.row_pitch) / 2u + gid.x;
//#else
    let i = gid.x;
//#endif
    // The buffer's own length and not a stated count: `dispatch_workgroups`
    // rounds the group count up, so the last group runs past the data.
    if (i >= arrayLength(&out_)) { return; }

    // `gate`, `up` and `out` share one layout in these variants, so this word's
    // two channels are the two channels of the operands' words: one load each,
    // rather than four half-indexed ones.
    let g = gate[i];
    let u = up[i];
    out_[i] = pie_pack_bf16(
        activate(pie_bf16_to_f32(g & 0xffffu), pie_bf16_to_f32(u & 0xffffu)),
        activate(pie_bf16_to_f32(g >> 16u), pie_bf16_to_f32(u >> 16u)),
    );
}

//#endif

// pie:instantiate silu_mul_bfloat16
// pie:instantiate silu_mul_strided_bfloat16 PIE_SILU_STRIDED=1
// pie:instantiate geglu_tanh_bfloat16 PIE_GEGLU=1
// pie:instantiate geglu_tanh_strided_bfloat16 PIE_GEGLU_STRIDED=1
// pie:instantiate gptoss_swiglu_bfloat16 PIE_GPTOSS=1
