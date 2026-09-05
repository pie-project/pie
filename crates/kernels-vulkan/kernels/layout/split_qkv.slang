#include "common/bf16.slang"

PIE_BUFFER_RO(0, packed);
PIE_BUFFER_RW(1, q);
PIE_BUFFER_RW(2, k);
PIE_BUFFER_RW(3, v);

struct Push { uint q_width; uint kv_width; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint row = gid.y;
    uint q_width = pc.q_width;
    uint kv_width = pc.kv_width;
    uint packed_width = q_width + 2u * kv_width;
    if (c >= packed_width) {
        return;
    }
    PIE_ACT value = packed[row * packed_width + c];
    if (c < q_width) {
        q[row * q_width + c] = value;
    } else if (c < q_width + kv_width) {
        k[row * kv_width + (c - q_width)] = value;
    } else {
        v[row * kv_width + (c - q_width - kv_width)] = value;
    }
}

// pie:instantiate split_qkv_bf16
