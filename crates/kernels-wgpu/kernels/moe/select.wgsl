// The routed projection against a DENSE expert bank.
//
// `moe/qmv_routed.wgsl` beside this one already does this against a QUANTIZED
// bank, and it is where a gpt-oss or a Qwen3-MoE checkpoint lands: the bank is
// codes plus a scale plane, the dot product is a dequantise-and-accumulate,
// and the whole kernel is shaped around how many packs a lane can hold.
// `Moe::matmul_select` declares its bank `Const<Self::Tensor<T>>` -- ONE
// address at the element the activation is in -- so none of that applies and
// none of it should be paid for. This is the same routing over an ordinary
// bf16 stack, and a separate file because it is a separate binding contract:
// `(x, bank, routes, y)` where the quantized one has seven operands and no
// place to put a dense weight plane.
//
// The bank is `[E, N, K]` with the EXPERT ON AXIS 0, so expert `e`'s output
// row `r` begins at `(e * N + r) * K` and the expert axis needs no stride of
// its own. Folding the expert into the element offset instead is the classic
// way to read expert 0's weights for every expert, and it produces text.
//
// ## One workgroup per (output row, route), 32 lanes striding the reduction
//
// That is `qmv_routed.wgsl`'s shape without its output blocking, and it is
// chosen for the reason both are: at one row per route the projection reads a
// whole `[N, K]` slice per route and is a bandwidth problem before it is an
// arithmetic one, so what matters is that the 32 lanes read 32 contiguous
// elements and nothing else. `select.metal` runs a 128-thread threadgroup with
// four simdgroups in it; here the reduction is a workgroup tree rather than a
// `simd_sum`, so four output rows in one workgroup would be four trees over
// one barrier and the barrier is the thing being economised. One row, 32
// lanes, one tree.
//
// ## The unrouted slot is a FLAG and not a `return`
//
// `qmv_routed.wgsl` states this at length and it is the same trap: every lane
// reads the same `routes[route]`, so a `return` on `e < 0` really is
// workgroup-uniform -- but naga cannot know that, because a value loaded from
// a storage buffer is non-uniform to its analysis, and the `workgroupBarrier`
// after it would then sit in non-uniform control flow and the module would be
// REJECTED. So the flag guards the WORK, the reduction runs over zeros, and
// the row comes out as the zero `select.metal` writes explicitly. That is
// what makes the result DEFINED rather than whatever the arena held, which in
// bf16 can be inf and which the fold after this multiplies by a weight.
//
// ## `y` is `array<atomic<u32>>`
//
// A bf16 tensor is `array<u32>` with two values per word
// (`common/bf16.inc.wgsl`), and `y[route * N + out_row]` has its partner half
// at `out_row + 1` or -- at the end of a route's row -- at the next route's
// first column. Both are different WORKGROUPS here, since a workgroup is one
// output row, so a read-modify-write would keep one of the two and drop the
// other. The atomic AND/OR pair touches only the writer's sixteen bits.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> bank: array<u32>;
@group(0) @binding(2) var<storage, read_write> routes: array<i32>;
@group(0) @binding(3) var<storage, read_write> y: array<atomic<u32>>;

// The five scalars, in the order `kernels_wgpu::moe` fires them.
//
// `x_row_stride` and `x_slot_stride` are both here and are different numbers,
// and the two are how a caller says which tensor it handed over. A gate or up
// projection reads the ONE shared row a token's norm produced, so its slot
// stride is 0 and every slot of a token reads the same activation; a down
// projection reads the `[rows, k, I]` stack the activation before it wrote, so
// its slot stride is `I` and its row stride is `k * I`. Reading slot 0 for
// every expert is not a crash -- it is k copies of the first expert's
// activation, which survives all the way to a plausible wrong token.
struct Params {
    in_width: u32,
    out_width: u32,
    slots_per_row: u32,
    x_row_stride: u32,
    x_slot_stride: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

var<workgroup> partial: array<f32, 32>;

// The bf16 half-index split, per buffer. `pie_load_bf16(&x, i)` would say this
// once for both and cannot be called: naga 30 refuses a `ptr<storage, ...>`
// function parameter, so a module calling it would parse and then fail
// `create_shader_module` on every device. The widening -- the part that can be
// got wrong -- still goes through the fragment.
fn load_x(i: u32) -> f32 {
    let word = x[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn load_bank(i: u32) -> f32 {
    let word = bank[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

// See the header: two atomics rather than one read-modify-write, because the
// other half of this word is another workgroup's output row.
fn store_y(i: u32, v: f32) {
    let at = i >> 1u;
    let b = pie_f32_to_bf16(v);
    if ((i & 1u) == 1u) {
        atomicAnd(&y[at], 0x0000ffffu);
        atomicOr(&y[at], b << 16u);
    } else {
        atomicAnd(&y[at], 0xffff0000u);
        atomicOr(&y[at], b);
    }
}

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let lane = lid.x;
    // THE ROUTE IS THE OUTPUT ROW. `y` is one row per (token, slot) pair in the
    // order the router chose them, which is what `Moe::matmul_select` states
    // with `y = [per(routes), bank.axis(1)]`.
    let out_row = wid.x;
    let route = wid.y;

    let e = routes[route];
    let routed = e >= 0;
    let k = max(params.slots_per_row, 1u);
    let w_base = (u32(max(e, 0)) * params.out_width + out_row) * params.in_width;
    let x_base = (route / k) * params.x_row_stride + (route % k) * params.x_slot_stride;

    var acc = 0.0;
    if (routed) {
        // Lane-strided over the reduction dimension: 32 lanes, `in_width`
        // elements, so each lane owns every 32nd. `in_width` is not bounded by
        // the workgroup width and never was.
        for (var i = lane; i < params.in_width; i = i + 32u) {
            acc = acc + load_bank(w_base + i) * load_x(x_base + i);
        }
    }

    partial[lane] = acc;
    workgroupBarrier();
    // A halving tree over the 32 lanes. The bound is a const-expression and
    // the guard is on the ADD, so every invocation reaches every barrier --
    // which is the whole requirement.
    for (var step = 16u; step > 0u; step = step >> 1u) {
        if (lane < step) {
            partial[lane] = partial[lane] + partial[lane + step];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        store_y(route * params.out_width + out_row, partial[0]);
    }
}

// pie:instantiate select_gemv
