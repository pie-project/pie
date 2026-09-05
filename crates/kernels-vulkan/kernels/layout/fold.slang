#include "common/bf16.slang"

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RW(1, y);
#if defined(PIE_MERGE)
struct Push { int merged; int rows; };
#else
struct Push { int width; int block; int rows; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint o = gid.y;
    if (o >= uint(pc.rows)) return;
#if defined(PIE_MERGE)
    if (c >= uint(pc.merged)) return;
    uint i = o * uint(pc.merged) + c;
    y[i] = x[i];
#else
    if (c >= uint(pc.width)) return;
    uint base = o * uint(pc.block) * uint(pc.width);
    float acc = 0.0;
    for (int r = 0; r < pc.block; r++) acc += PIE_LOAD(x[base + uint(r) * uint(pc.width) + c]);
    y[o * uint(pc.width) + c] = PIE_STORE(acc / float(pc.block));
#endif
}

// pie:instantiate pool_rows_bf16
// pie:instantiate merge_rows_bf16 PIE_MERGE=1
