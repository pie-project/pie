// Dense FFN gated activations sharing one binding contract.
//
// Ported from Metal's combined `gated.metal`: buffers 0,1,2 are always
// `(gate, up, out)`, while the params buffer only exists for the variants whose
// arithmetic needs it. Bounds formerly derived from per-row params stay gone;
// the dispatch grid is the extent.

#include "common/bf16.slang"

#if defined(PIE_GEGLU_STRIDED)
// 256x1 and flat, because that is what this row's rule says.
//
// This was a 16x16 workgroup with the dispatch id's `y`
// read as the row, on the reasoning that PLE rows are NARROW (`ple_dim` is 256
// elements) and a 256-wide workgroup leaves half of every row's lanes idle. The
// reasoning is fine and the shape was not: the ROW states
// `LaunchRule::Elementwise`, which is `[width * rows, 1, 1]`. One workgroup on
// y then covers sixteen rows and every row past 15 is never dispatched --
// measured at 21 rows through the equivalent WGSL body on a 4090, where row 16
// came back holding the sentinel it was born with and the dispatch SUCCEEDED.
// `dsl::metal::geglu_strided` is gemma's PLE gate and `rows` is the fire's
// token count, so any prefill longer than sixteen tokens was silently dropping
// most of its per-layer embeddings.
//
// The rule is SHARED with `kernels-metal`, where a threadgroup is sized at
// dispatch and `Elementwise` is correct, so changing the row would be three
// tables and three drivers moved to fix one body. Changing the body to match
// the rule it already states is local, needs no table edit and needs no
// agreement from Metal. `kernels-wgpu` took the same change first.
#define PIE_NUMTHREADS [numthreads(256, 1, 1)]
#elif defined(PIE_SILU_STRIDED)
#define PIE_NUMTHREADS [numthreads(16, 16, 1)]
#else
#define PIE_NUMTHREADS [numthreads(256, 1, 1)]
#endif

PIE_BUFFER_RO(0, gate);
PIE_BUFFER_RO(1, up);
PIE_BUFFER_RW(2, out_);

struct GegluParams { uint unused; };
struct GegluStridedParams { uint width; uint rows; uint gate_pitch; uint up_pitch; uint out_pitch; };
struct GptOssSwiGluParams { uint unused; float limit; float alpha; };

#if defined(PIE_GEGLU)
PIE_PARAMS(3, GegluParams, p);
#elif defined(PIE_GEGLU_STRIDED)
PIE_PARAMS(3, GegluStridedParams, p);
#elif defined(PIE_GPTOSS)
PIE_PARAMS(3, GptOssSwiGluParams, p);
#endif

#if defined(PIE_SILU_STRIDED)
struct Push { int row_pitch; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#endif

float sigmoid_mlx(float x) {
    float y = 1.0 / (1.0 + exp(-abs(x)));
    return x < 0.0 ? 1.0 - y : y;
}

float gelu_tanh(float x) {
    const float k = 0.7978845608028654;
    float inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

// Metal launches these with `dispatchThreads`, an EXACT thread count, which is
// why `GegluParams::unused` is unused and why `silu_mul` takes no params at
// all -- there is no tail to guard on that backend. `vkCmdDispatch` takes
// WORKGROUPS, so the real extent here is always `256 * ceil(n / 256)` and any
// width that is not a multiple of 256 leaves up to 255 threads past the end.
// These bodies write unconditionally, so that tail is an out-of-range STORE,
// not a wasted read.
//
// The bound comes from the output buffer's own length rather than from a new
// scalar. A push constant would have to be added to the row, and the row is
// the ABI this crate shares with `kernels-metal`, so it would either diverge
// or drag a field Metal has deliberately retired back into both. `GetDimensions`
// on a runtime array is the descriptor range, which is exactly the
// region the store must stay inside, and it needs nothing from the caller.
bool past_the_end(uint gid) {
    uint length; uint stride;
    out_.GetDimensions(length, stride);
    return gid >= length;
}

PIE_NUMTHREADS
void main(uint3 tid : SV_DispatchThreadID) {
#if defined(PIE_GEGLU_STRIDED)
    // The dispatch id's `x` is a flat ELEMENT index over `rows * width`,
    // which is exactly what `Elementwise` hands this body. The row and the
    // column are recovered from it rather than read off two axes, one of which
    // was never dispatched.
    //
    // Simpler than the WGSL twin, and the difference is the storage type, not
    // the arithmetic: `PIE_ACT` is `uint16_t`, so a lane owns one ELEMENT and
    // stores it plainly. WGSL has no 16-bit storage, so there a lane owns a
    // 32-bit word holding two values and the ragged edge needs a
    // compare-exchange.
    uint n = tid.x;
    if (n >= p.rows * p.width) return;
    uint m = n / p.width;
    uint k = n % p.width;
    uint gi = m * p.gate_pitch + k;
    uint ui = m * p.up_pitch + k;
    uint oi = m * p.out_pitch + k;
    out_[oi] = PIE_STORE(gelu_tanh(PIE_LOAD(gate[gi])) * PIE_LOAD(up[ui]));
#elif defined(PIE_GEGLU)
    uint gid = tid.x;
    if (past_the_end(gid)) return;
    out_[gid] = PIE_STORE(gelu_tanh(PIE_LOAD(gate[gid])) * PIE_LOAD(up[gid]));
#elif defined(PIE_GPTOSS)
    uint gid = tid.x;
    if (past_the_end(gid)) return;
    float g = min(PIE_LOAD(gate[gid]), p.limit);
    float u = clamp(PIE_LOAD(up[gid]), -p.limit, p.limit);
    float sig = 1.0 / (1.0 + exp(-p.alpha * g));
    out_[gid] = PIE_STORE((g * sig) * (u + 1.0));
#else
    #if defined(PIE_SILU_STRIDED)
    uint gid = tid.y * uint(pc.row_pitch) + tid.x;
    #else
    uint gid = tid.x;
    #endif
    if (past_the_end(gid)) return;
    float g = PIE_LOAD(gate[gid]);
    float sg = bf16_to_f32(f32_to_bf16(sigmoid_mlx(g)));
    float sil = bf16_to_f32(f32_to_bf16(g * sg));
    out_[gid] = PIE_STORE(sil * PIE_LOAD(up[gid]));
#endif
}

// pie:instantiate geglu_tanh_bfloat16 PIE_GEGLU=1
// pie:instantiate geglu_tanh_strided_bfloat16 PIE_GEGLU_STRIDED=1
// pie:instantiate gptoss_swiglu_bfloat16 PIE_GPTOSS=1
// pie:instantiate silu_mul_bfloat16
// pie:instantiate silu_mul_strided_bfloat16 PIE_SILU_STRIDED=1
