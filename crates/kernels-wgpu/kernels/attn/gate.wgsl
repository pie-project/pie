// qwen3.5's gated attention: the [query|gate] split, and the gate multiply.
//
// Both rows are UNSTATED — the table names no operands for `gate` or
// `q_gate_split` — so there is no `bindings()` answer to derive these numbers
// from, and the contract below is the one `kernels-vulkan`'s `attn/gate.comp`
// states, with its push block moved to `@group(1) @binding(0)` per this
// backend's ABI. A driver launching an unstated row follows the lowered plan's
// own argument order (`.wiki/new-driver/vulkan.md` §13); the ORDER is what has
// to match, and it is the order below.
//
// The gate multiply uses MLX's stable sigmoid in f32 and narrows once, at the
// store. Sigmoid of a large negative argument through the naive
// `1/(1+exp(-x))` overflows `exp` before it underflows the quotient.

//#include "common/bf16.inc.wgsl"

//#if defined(PIE_Q_GATE_SPLIT)

@group(0) @binding(0) var<storage, read> qg: array<u32>;
@group(0) @binding(1) var<storage, read_write> q_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> gate_out: array<u32>;

struct Params { head_dim: i32, qg_row_stride: i32, out_row_stride: i32 }
@group(1) @binding(0) var<uniform> params: Params;

// The bf16 half-index unpack. `pie_load_bf16(&qg, i)` is the shared answer and
// cannot be CALLED: its `ptr<storage, array<u32>, read>` parameter is WGSL's
// `unrestricted_pointer_parameters`, which naga does not implement, so a module
// that calls it parses and then fails `create_shader_module`. The CONVERSION
// keeps one definition in `common/bf16.inc.wgsl`.
fn qg_at(i: u32) -> f32 {
    let word = qg[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let h = gid.y;
    let row = gid.z;
    let hd = u32(params.head_dim);

    // One invocation per output WORD: a bf16 pair, owned outright. Storing one
    // half would be a read-modify-write of a word the next invocation is also
    // writing, and WGSL has no sub-word atomic to arbitrate it. Every point of
    // the head-dim axis is even, so a pair never straddles two heads.
    let d = gid.x * 2u;
    if (d >= hd) { return; }

    // The y extent is the query-head count, which nothing hands this kernel as
    // a scalar; the grid IS the statement. Same reading as the GLSL's
    // `gl_NumWorkGroups.y`.
    let n_q = groups.y;
    var out_row = row * n_q * hd;
    if (params.out_row_stride > 0) { out_row = row * u32(params.out_row_stride); }
    var qg_row = row * n_q * hd * 2u;
    if (params.qg_row_stride > 0) { qg_row = row * u32(params.qg_row_stride); }

    // `[query|gate]` interleaved per head: the head's 2*hd run holds the query
    // in its low half and the gate in its high half.
    let src = qg_row + h * 2u * hd + d;
    let dst = out_row + h * hd + d;

    let at = dst >> 1u;
    if (at < arrayLength(&q_out)) {
        q_out[at] = pie_pack_bf16(qg_at(src), qg_at(src + 1u));
    }
    if (at < arrayLength(&gate_out)) {
        gate_out[at] = pie_pack_bf16(qg_at(src + hd), qg_at(src + hd + 1u));
    }
}

//#else

@group(0) @binding(0) var<storage, read_write> attn: array<u32>;
@group(0) @binding(1) var<storage, read> gate: array<u32>;

struct Params { row_stride: i32 }
@group(1) @binding(0) var<uniform> params: Params;

// MLX's formulation, and the reason it is not `1/(1+exp(-x))`: `exp` of a large
// positive argument is an infinity long before the quotient underflows, so the
// naive form returns NaN for the very inputs a gate is meant to squash. Folding
// through `abs` keeps the exponent negative and the reflection restores the
// sign.
fn sigmoid_mlx(x: f32) -> f32 {
    let y = 1.0 / (1.0 + exp(-abs(x)));
    return select(y, 1.0 - y, x < 0.0);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let row = gid.y;
    // Indices here are WORDS, so a row pitch stated in ELEMENTS is halved --
    // the same reading `norm/residual_add.wgsl` gives its own pitch. The
    // fallback is the grid's own x extent, which is in words too because this
    // body handles a pair per invocation, so the expression is the GLSL's
    // unchanged and means the same thing.
    var row_base = row * groups.x * 256u;
    if (params.row_stride > 0) { row_base = row * u32(params.row_stride) / 2u; }
    let at = row_base + gid.x;

    // `dispatch_workgroups` rounds the group count up, so the last group runs
    // past the row. This is a read-modify-WRITE and an unguarded tail corrupts
    // whatever follows; the buffer's own length is the bound, and it needs
    // nothing from the caller -- which matters because this row states no
    // operands, so there is no scalar to add a count to even if one were
    // wanted.
    if (at >= arrayLength(&attn)) { return; }

    let a = attn[at];
    let g = gate[at];
    attn[at] = pie_pack_bf16(
        pie_bf16_to_f32(a & 0xffffu) * sigmoid_mlx(pie_bf16_to_f32(g & 0xffffu)),
        pie_bf16_to_f32(a >> 16u) * sigmoid_mlx(pie_bf16_to_f32(g >> 16u)),
    );
}

//#endif

// pie:instantiate gate_bfloat16
// pie:instantiate q_gate_split_bfloat16 PIE_Q_GATE_SPLIT=1
