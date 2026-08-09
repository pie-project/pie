// The dense FFN gated activations: one file because they are one BINDING
// CONTRACT.
//
// `gate`, `up`, `out` at `@group(0)` bindings 0, 1, 2 in every variant, and a
// params struct at 3 for the three that have one -- `silu_mul` and its strided
// form state no params at all, which is why that buffer is declared inside the
// arms and not above them. Five entrypoints, three activations:
//
//   silu_mul       out = silu(gate) * up                    no params
//   geglu_tanh     out = gelu_tanh(gate) * up               GegluParams
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
// why `GegluParams::unused` is unused and why `silu_mul` takes no params at
// all: there is no tail to guard on that backend. `dispatch_workgroups` counts
// WORKGROUPS, so the real extent here is `256 * ceil(n / 256)` and any width
// that is not a multiple of the workgroup leaves invocations past the end. The
// bound is `arrayLength(&out_)` -- the bound storage range, which needs nothing
// from the caller and cannot drift from a scalar the row would have to grow.
//
// And one invocation owns one WORD, which is TWO bf16 values: WGSL has no
// 16-bit storage, so the launch's x extent is HALF the element count. That is
// `norm/residual_add.wgsl`'s convention and it is the whole tree's. In the four
// flat variants that ownership is total and the store is one write. The strided
// GeGLU is the exception -- three pitches, so only the OUTPUT word is a unit of
// ownership and a ragged edge can be shared with the next row -- and it pays
// for that with an `atomic<u32>` output and a compare-exchange on the edge.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read> gate: array<u32>;
@group(0) @binding(1) var<storage, read> up: array<u32>;
//#if defined(PIE_GEGLU_STRIDED)
// Atomic in this variant ALONE: it is the only one whose invocation can own
// half a word. See `store_half`. The host binds the same read_write storage
// buffer of 4-byte words either way, so the row's ABI does not move.
@group(0) @binding(2) var<storage, read_write> out_: array<atomic<u32>>;
//#else
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;
//#endif

//#if defined(PIE_GEGLU)
// Bound and never read, exactly as Metal's is -- see "the grid is the extent"
// above. It stays because the row states `params: Buf`, and the bind group
// layout a shell builds from the row is the same on all three backends.
struct GegluParams { unused: u32 }
@group(0) @binding(3) var<storage, read> params: GegluParams;
//#elif defined(PIE_GEGLU_STRIDED)
// gemma4's per-layer-embedding GeGLU reads a NARROW gate out of a WIDE table:
// the PLE table is `[rows, n_layers * ple_dim]`, so layer L's slice is
// `ple_dim` wide with `n_layers * ple_dim` between rows, while the gate and the
// output are densely `[rows, ple_dim]`. A byte offset cannot express that, and
// the flat kernel reading one walks into the NEXT layers' slices after the
// first row -- not a crash and not even implausible numbers, since those slices
// are the same table.
struct GegluStridedParams {
    width: u32,
    rows: u32,
    gate_pitch: u32,
    up_pitch: u32,
    out_pitch: u32,
}
@group(0) @binding(3) var<storage, read> params: GegluStridedParams;
//#elif defined(PIE_GPTOSS)
struct GptOssSwiGluParams {
    // Was a per-row element count. See `GegluParams`.
    unused: u32,
    limit: f32,
    alpha: f32,
}
@group(0) @binding(3) var<storage, read> params: GptOssSwiGluParams;
//#endif

//#if defined(PIE_SILU_STRIDED)
// Vulkan sends this to a push block; WebGPU has none, so it is the one field of
// the uniform block. The row is UNSTATED, so the placement follows
// `norm/residual_add.wgsl`, which is the same kernel shape with the same
// scalar.
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
    return 0.5 * x * (1.0 + tanh(inner));
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

// 16x16 and not 256x1, mirroring both siblings' launch: this is the one variant
// whose rows are NARROW -- `ple_dim` is 256 elements, so 128 words -- and a
// 256-wide workgroup would leave half of every row's lanes idle.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.y;
    if (m >= params.rows) { return; }

    // Three operands with three pitches, so only the OUTPUT's word is a unit of
    // ownership; the gate and the up rows are addressed per element and may sit
    // at any parity. `gid.x` counts words of the output row, so the host's x
    // extent is `ceil(width / 2)` and not `width`.
    let base_o = m * params.out_pitch;
    let base_g = m * params.gate_pitch;
    let base_u = m * params.up_pitch;
    let word = (base_o >> 1u) + gid.x;

    let lo = word * 2u;
    let hi = lo + 1u;
    let has_lo = lo >= base_o && lo < base_o + params.width;
    let has_hi = hi < base_o + params.width;
    if (has_lo && has_hi) {
        // Both elements are this row's, so this invocation owns the word: one
        // write, no read-modify-write, nothing to race.
        let k = lo - base_o;
        atomicStore(&out_[word], pie_pack_bf16(
            activate(gate_at(base_g + k), up_at(base_u + k)),
            activate(gate_at(base_g + k + 1u), up_at(base_u + k + 1u)),
        ));
    } else if (has_hi) {
        // The row begins in this word's UPPER half, which only an odd
        // `out_pitch` produces. The lower half is then the previous row's, and
        // that row's invocation is writing it concurrently, so this half goes
        // through the compare-exchange.
        let k = hi - base_o;
        store_half(hi, activate(gate_at(base_g + k), up_at(base_u + k)));
    } else if (has_lo) {
        // The row's last element is this word's LOWER half. The upper half is
        // either this row's PADDING -- nobody's, since `out_pitch > width` --
        // or the next row's first element when the pitch is odd and tight. The
        // CAS is correct for both and the branch does not have to know which.
        let k = lo - base_o;
        store_half(lo, activate(gate_at(base_g + k), up_at(base_u + k)));
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
