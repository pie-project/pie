// Dense FFN gated activations sharing one binding contract.
//
// Ported from Metal's combined `gated.metal`: buffers 0,1,2 are always
// `(gate, up, out)`, while the params buffer only exists for the variants whose
// arithmetic needs it. Bounds formerly derived from per-row params stay gone;
// the dispatch grid is the extent.

#include "common/bf16.slang"

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
//
// EVERY ARM, and the chain that stood here is gone with the last exception:
// `PIE_SILU_STRIDED` was a 16x16 workgroup reading the row off `tid.y`, which
// is the very shape the paragraph above condemns. It was never fired -- the
// tree records the body as unwritten on this backend and on Metal -- so the
// bug was latent rather than paid for, and the fix is the one the geglu arm
// already took: one flat index over `rows * width`, recovered into a row and
// a column.
#define PIE_NUMTHREADS [numthreads(256, 1, 1)]

PIE_BUFFER_RO(0, gate);
PIE_BUFFER_RO(1, up);
PIE_BUFFER_RW(2, out_);



// THE DENSE GEGLU STATES NO PARAMS BLOCK, so this chain opens on the strided
// arm and `PIE_GEGLU` reaches `main` with buffers 0..2 and nothing else.
//
// `struct GegluParams { uint unused; }` and its `PIE_PARAMS(3, ...)` stood
// here. The field was a per-row element count read as a bound, which is the
// defect the comment above `past_the_end` records: the dispatch covers
// `width * rows` and a per-row number cannot bound it, so the bound went and
// the field stayed behind holding nothing. The BLOCK outlived the field only
// because the row said `params: Buf` and every backend built its bind layout
// from the row.
//
// On this plane it was already a fiction. `slangc` emits no binding
// decoration for a global the entrypoint never reads, so `PIE_GEGLU`'s module
// has never had a descriptor at 3 -- `mlp::geglu_tanh` was calling
// `ctx.params()` for a sentinel (`hold::params_block`) it then dropped on the
// floor. Deleting the declaration makes the SOURCE say what the compiled
// module has said all along.
//
// The two arms below carried theirs in a storage struct for the same reason and
// no longer do. Five pitches and gpt-oss's `limit`/`alpha` ARE numbers no grid
// supplies and are genuinely read, so unlike `GegluParams` they became `Const`
// marks rather than nothing -- and twenty bytes and eight bytes are both push
// ranges on every device this targets, well inside the 128 the limit
// guarantees.
//
// `GptOssSwiGluParams` opened with a dead per-row element count, held only so
// `limit` and `alpha` kept their offsets. A push block packed from the marks
// the BODY passes needs no such filler, so the dead word stops at the host.
#if defined(PIE_GEGLU_STRIDED) || defined(PIE_SILU_STRIDED)
struct Push { uint width; uint rows; uint gate_pitch; uint up_pitch; uint out_pitch; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#elif defined(PIE_GPTOSS)
struct Push { float limit; float alpha; };
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
// why `GegluParams::unused` was unused -- and, in the end, why that struct and
// its binding are gone and the dense GeGLU takes no params at all, the shape
// `silu_mul` has had from the start. There is no tail to guard on that
// backend. `vkCmdDispatch` takes WORKGROUPS, so the real extent here is always
// `256 * ceil(n / 256)` and any width that is not a multiple of 256 leaves up
// to 255 threads past the end.
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
    if (n >= pc.rows * pc.width) return;
    uint m = n / pc.width;
    uint k = n % pc.width;
    uint gi = m * pc.gate_pitch + k;
    uint ui = m * pc.up_pitch + k;
    uint oi = m * pc.out_pitch + k;
    out_[oi] = PIE_STORE(gelu_tanh(PIE_LOAD(gate[gi])) * PIE_LOAD(up[ui]));
#elif defined(PIE_GEGLU)
    uint gid = tid.x;
    if (past_the_end(gid)) return;
    out_[gid] = PIE_STORE(gelu_tanh(PIE_LOAD(gate[gid])) * PIE_LOAD(up[gid]));
#elif defined(PIE_GPTOSS)
    uint gid = tid.x;
    if (past_the_end(gid)) return;
    float g = min(PIE_LOAD(gate[gid]), pc.limit);
    float u = clamp(PIE_LOAD(up[gid]), -pc.limit, pc.limit);
    float sig = 1.0 / (1.0 + exp(-pc.alpha * g));
    out_[gid] = PIE_STORE((g * sig) * (u + 1.0));
#elif defined(PIE_SILU_STRIDED)
    uint n = tid.x;
    if (n >= pc.rows * pc.width) return;
    uint m = n / pc.width;
    uint k = n % pc.width;
    float g = PIE_LOAD(gate[m * pc.gate_pitch + k]);
    float sg = bf16_to_f32(f32_to_bf16(sigmoid_mlx(g)));
    float sil = bf16_to_f32(f32_to_bf16(g * sg));
    out_[m * pc.out_pitch + k] = PIE_STORE(sil * PIE_LOAD(up[m * pc.up_pitch + k]));
#else
    uint gid = tid.x;
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
