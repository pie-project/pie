#include "common/bf16.slang"

// qwen3.5 gated attention helpers. This keeps the Metal layout rules: split
// reads [query|gate] per head, and the gate multiply uses MLX's stable sigmoid
// in float before the single bf16 store.

#define PIE_GROUP 256

#if defined(PIE_Q_GATE_SPLIT)
PIE_BUFFER_RO(0, qg);
PIE_BUFFER_RW(1, q_out);
PIE_BUFFER_RW(2, gate_out);
struct Push { int head_dim; int qg_row_stride; int out_row_stride; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    uint h = gid.y;
    uint row = gid.z;
    uint hd = uint(pc.head_dim);
    if (i >= hd) return;
    uint n_q = gl_NumWorkGroups.y;
    uint out_row = pc.out_row_stride > 0 ? row * uint(pc.out_row_stride) : row * n_q * hd;
    uint qg_row = pc.qg_row_stride > 0 ? row * uint(pc.qg_row_stride) : row * n_q * hd * 2u;
    q_out[out_row + h * hd + i] = qg[qg_row + h * 2u * hd + i];
    gate_out[out_row + h * hd + i] = qg[qg_row + h * 2u * hd + hd + i];
}
#else
PIE_BUFFER_RW(0, attn);
PIE_BUFFER_RO(1, gate);
struct Push { int row_stride; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

float sigmoid_mlx(float x) {
    float y = 1.0 / (1.0 + exp(-abs(x)));
    return x < 0.0 ? (1.0 - y) : y;
}

[numthreads(PIE_GROUP, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint x = gid.x;
    uint row = gid.y;
    uint row_base = pc.row_stride > 0 ? row * uint(pc.row_stride) : row * gl_NumWorkGroups.x * uint(PIE_GROUP);
    uint at = row_base + x;
    // `vkCmdDispatch` is workgroup-granular, so the x extent is rounded up to
    // a multiple of the local size and the last group runs past the row. This
    // is a read-modify-WRITE, so an unguarded tail corrupts whatever follows.
    // The output's own length is the descriptor range and needs nothing from
    // the caller -- which matters here because this row states no operands, so
    // there is no scalar to add a bound to even if one were wanted. GLSL spelt
    // this `attn.length()`; Slang spells the same `OpArrayLength` as
    // `GetDimensions`, which returns the stride as well and is why there is a
    // second output nothing reads.
    uint length; uint stride;
    attn.GetDimensions(length, stride);
    if (at >= length) return;
    attn[at] = PIE_STORE(PIE_LOAD(attn[at]) * sigmoid_mlx(PIE_LOAD(gate[at])));
}
#endif

// pie:instantiate gate_bfloat16
// pie:instantiate q_gate_split_bfloat16 PIE_Q_GATE_SPLIT=1
