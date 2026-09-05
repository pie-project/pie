#include "common/bf16.slang"

#if defined(PIE_SELECT)
PIE_BUFFER_RO(0, table);
PIE_BUFFER_RW(1, out_);
struct Push { int stride; int offset; int width; };
#else
PIE_BUFFER_RO(0, src);
PIE_BUFFER_RW(1, left);
PIE_BUFFER_RW(2, right);
struct Push { int left_dim; int right_dim; };
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint row = gid.y;
#if defined(PIE_SELECT)
    uint width = uint(pc.width);
    if (c >= width) {
        return;
    }
    out_[row * width + c] = table[row * uint(pc.stride) + uint(pc.offset) + c];
#else
    uint left_dim = uint(pc.left_dim);
    uint total = left_dim + uint(pc.right_dim);
    if (c >= total) {
        return;
    }
    PIE_ACT value = src[row * total + c];
    if (c < left_dim) {
        left[row * left_dim + c] = value;
    } else {
        right[row * uint(pc.right_dim) + (c - left_dim)] = value;
    }
#endif
}

// pie:instantiate split_rows_bf16
// pie:instantiate select_slice_bf16 PIE_SELECT=1
