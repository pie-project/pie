#include "common/bf16.slang"

// Vulkan port of Metal's split_qkv: same copy. The grid's rounded x tail must
// not fall through into the next row.
//
// THE TWO WIDTHS ARE PUSH CONSTANTS, NOT A STORAGE STRUCT. `SplitQkvParams {
// q_width, kv_width }` was Metal's buffer-4 layout carried over whole, and a
// storage block for eight live bytes cost this plane a descriptor, a staging
// slot and the sentinel `Holds::params_block` mints for a buffer with no
// address. `q_width: Const<u32>` and `kv_width: Const<u32>` are words 0 and 1
// of the same statement run the block was staged from, and eight bytes is a
// push range on every device this targets. The ORDER is the struct's, because
// it is the statement's: read swapped, both boundaries land inside a
// neighbouring projection rather than out of bounds, which is the failure this
// file's own test was written to catch.

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
    if (c >= packed_width) return;

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
