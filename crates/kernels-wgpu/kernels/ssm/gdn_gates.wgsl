// The gated-delta prologue's decay row: one packed operand in, one packed
// result out.
//
// ── THE PACKING IS THE KERNEL'S, AND IT IS NEVER RESTATED ──────────────────
//
// `ssm.gdn_prep` declares `ba` and `gates` as ONE rectangle each, and both are
// two half-rows glued together:
//
//     ba[t]    = [ b_0 .. b_{Vh-1} | a_0 .. a_{Vh-1} ]     bf16, 2*Vh wide
//     gates[t] = [ g_0 .. g_{Vh-1} | beta_0 .. beta_{Vh-1} ]  f32, 2*Vh wide
//
// `Vh` is READ OFF the operand -- half of `ba.width` -- by the claim body, and
// arrives here as the one number that says where the seam is. An executor
// never cuts either row: a shim that handed this kernel two compact halves
// while the kernel indexed a packed one made `prep -> chunked` wrong at every
// window longer than a single token, and the reason it survived a decode is
// that at one row the two layouts coincide.
//
// So this is ONE launch. Not two over two halves, not one per half-row: the
// operand it is given is the operand the point declares, and the seam is
// arithmetic inside the body.
//
// ── THE ARITHMETIC ────────────────────────────────────────────────────────
//
// `pie::ssm::qwen_gdn_ba_gates` in
// `kernels-cuda/kernels/ssm/gated_delta_net_prep.cuh`, which is where the
// numeric contract was measured:
//
//     z        = a[h] + dt_bias[h]
//     softplus = z > 20 ? z : log(1 + exp(z))
//     g_log[h] = -exp(A_log[h]) * softplus
//     beta[h]  = sigmoid(b[h])
//
// The `z > 20` branch is not an optimisation. `exp(z)` overflows f32 at about
// 88 and `log(1 + exp(z))` is within one ULP of `z` long before that, so the
// branch is what keeps a large decay finite rather than `inf`.
//
// cuda spells the small-`z` side `log1pf(expf(z))` where this says
// `log(1 + exp(z))`; WGSL has no `log1p`. The two differ only where `exp(z)`
// is far below the f32 epsilon of 1, and there both answers round to the same
// float in every bit that reaches `g_log`.
//
// ── WHY THE RESULT NEEDS NO WORD GAMES ────────────────────────────────────
//
// `gates` is f32, so an invocation owns ONE element and stores it directly.
// `ba` and `dt_bias` are bf16 and are only READ, and a read of one half of a
// word races with nothing. That is why this file needs neither the pair
// ownership `ssm/causal_conv1d.wgsl` uses nor the CAS `norm/rms.wgsl` uses.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> ba: array<u32>;
@group(0) @binding(1) var<storage, read_write> a_log: array<f32>;
@group(0) @binding(2) var<storage, read_write> dt_bias: array<u32>;
@group(0) @binding(3) var<storage, read_write> gates: array<f32>;

struct Params {
    v_heads: i32,
}
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let heads = u32(params.v_heads);
    let h = gid.x;
    if (h >= heads) { return; }
    let t = gid.y;

    // The seam: `b` in the first half of the row, `a` in the second. Reading
    // them the other way round is finite, plausible and wrong -- `b` becomes a
    // decay and `a` becomes a mixing rate.
    let row = t * 2u * heads;
    let b = pie_bf16_at(ba[(row + h) >> 1u], row + h);
    let a = pie_bf16_at(ba[(row + heads + h) >> 1u], row + heads + h);

    let z = a + pie_bf16_at(dt_bias[h >> 1u], h);
    var softplus = z;
    if (z <= 20.0) {
        softplus = log(1.0 + exp(z));
    }

    gates[row + h] = -exp(a_log[h]) * softplus;
    gates[row + heads + h] = 1.0 / (1.0 + exp(-b));
}

// pie:instantiate gdn_ba_gates_bfloat16
