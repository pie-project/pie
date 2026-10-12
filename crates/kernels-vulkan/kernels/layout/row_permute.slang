#include "common/bf16.slang"

// A row the table names as -1 is padding.

PIE_BUFFER_RO(0, src);
PIE_BUFFER_RW(1, dst);
[[vk::binding(2)]] StructuredBuffer<int> rows;

struct Push { uint width; uint count; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint i = gid.y;
    if (c >= pc.width || i >= pc.count) return;
    int named = rows[i];
    if (named < 0) return;
#ifdef PIE_SCATTER
    dst[uint(named) * pc.width + c] = src[i * pc.width + c];
#else
    dst[i * pc.width + c] = src[uint(named) * pc.width + c];
#endif
}

// pie:instantiate pack_rows_bf16
// pie:instantiate pack_rows_f32 PIE_ACT_F32=1
// pie:instantiate unpack_rows_bf16 PIE_SCATTER=1
// pie:instantiate unpack_rows_f32 PIE_SCATTER=1 PIE_ACT_F32=1
