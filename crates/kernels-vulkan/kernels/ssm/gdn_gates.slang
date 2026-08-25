// The gated-delta prologue's decay row: one packed operand in, one packed
// result out, in one launch.
//
// ── THE PACKING IS THE KERNEL'S, AND IT IS NEVER RESTATED ──────────────────
//
// `ssm.gdn_prep` declares `ba` and `gates` as ONE rectangle each, and each is
// two half-rows glued together:
//
//     ba[t]    = [ b_0 .. b_{Vh-1} | a_0 .. a_{Vh-1} ]       bf16, 2*Vh wide
//     gates[t] = [ g_0 .. g_{Vh-1} | beta_0 .. beta_{Vh-1} ]  f32, 2*Vh wide
//
// `Vh` is READ OFF the operand -- half of `ba.width` -- by the claim body and
// arrives here as the one number that says where the seam is. An executor
// never cuts either row. A shim that handed a kernel two compact halves while
// the kernel indexed a packed one made `prep -> gated_delta_chunked` wrong at
// every window longer than a single token, and the reason it survived every
// decode is that at one row `[first half | second half]` and `[half | half]`
// are the same bytes.
//
// So this is ONE dispatch over `(v_heads, rows)`, not two over two halves and
// not one per half-row: the operand it is given is the operand the point
// declares, and the seam is arithmetic inside the body.
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
// 88 and `log(1 + exp(z))` is within one ulp of `z` long before that, so the
// branch is what keeps a large decay finite instead of `inf` -- and an `inf`
// here does not fault, it multiplies into `g_log`, exponentiates to zero in
// the scan and silently erases the recurrent state.
//
// cuda spells the small-`z` side `log1pf(__expf(z))`; this says
// `log(1 + exp(z))`, because Slang's stdlib has no `log1p` that lowers to a
// SPIR-V instruction. The two differ only where `exp(z)` is far below the f32
// epsilon of 1, and there both answers round to the same float in every bit
// that reaches `g_log`.
//
// ── WHY THIS FILE NEEDS NO WORD GAMES AND NO SCRATCH ──────────────────────
//
// `gates` is f32, so an invocation owns ONE element and stores it directly.
// `ba` and `dt_bias` are bf16 and are only READ, and a read of one bf16 races
// with nothing. `A_log` is a weight and is read once per head.
//
// It is also why this is a separate file from `ssm/gdn_prep.slang` rather than
// an arm of it. That file's prep entrypoints compute this same decay row, but
// they write it into an fp32 SCRATCH plane laid out `[g, beta]` interleaved
// per head and packed at a `row_pitch` the host chose, alongside a normalised
// q, a normalised k and a widened v -- four planes no point declares and this
// plane's `Encode` has no door to hand a body. Nothing there can be reached by
// a `#[claims]` body, so the two files share an arithmetic and no code.

#include "common/bf16.slang"

#define PIE_GROUP_X 64

PIE_BUFFER_RO(0, ba);
[[vk::binding(1)]] StructuredBuffer<float> a_log;
PIE_BUFFER_RO(2, dt_bias);
[[vk::binding(3)]] RWStructuredBuffer<float> gates;

// The seam, and the only number this kernel is told.
struct Push {
    int v_heads;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const int heads = pc.v_heads;
    const int h = int(tid.x);
    if (h >= heads) return;
    const int t = int(tid.y);

    // The seam: `b` in the first half of the row, `a` in the second, which is
    // the order the import packs `[in_proj_b, in_proj_a]`. Reading them the
    // other way round is finite, plausible and wrong -- `b` becomes a decay
    // and `a` becomes a mixing rate, and every value stays in range.
    const int row = t * 2 * heads;
    const float b = PIE_LOAD(ba[row + h]);
    const float a = PIE_LOAD(ba[row + heads + h]);

    const float z = a + PIE_LOAD(dt_bias[h]);
    const float sp = z > 20.0 ? z : log(1.0 + exp(z));

    gates[row + h] = -exp(a_log[h]) * sp;
    gates[row + heads + h] = 1.0 / (1.0 + exp(-b));
}

// pie:instantiate gdn_ba_gates_bfloat16
