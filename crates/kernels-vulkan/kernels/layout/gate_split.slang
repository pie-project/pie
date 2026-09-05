#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

PIE_BUFFER_RO(0, qg);
PIE_BUFFER_RW(1, q_out);
PIE_BUFFER_RW(2, gate_out);

struct Push { int head_dim; int qg_width; int q_width; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    uint h = gid.y;
    uint row = gid.z;
    uint hd = uint(pc.head_dim);
    if (i >= hd) {
        return;
    }
    uint out_row = row * uint(pc.q_width);
    uint qg_row = row * uint(pc.qg_width);
    q_out[out_row + h * hd + i] = qg[qg_row + h * 2u * hd + i];
    gate_out[out_row + h * hd + i] = qg[qg_row + h * 2u * hd + hd + i];
}

// pie:instantiate q_gate_split_bf16 PIE_GROUP_X=256
