#include "common/bf16.slang"

// Raw RGB patch bytes (HWC) to normalized rows, channel-major (order 0) or
// pixel-major (order 1), the frame repeated width / (3 patch^2) times.
[[vk::binding(0)]] StructuredBuffer<uint8_t> x;
PIE_BUFFER_RW(1, y);
struct Push {
    int side_px; int width; int order;
    float mean0; float mean1; float mean2;
    float sd0; float sd1; float sd2;
    int rows;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c_out = gid.x;
    uint n = gid.y;
    if (c_out >= uint(pc.width) || n >= uint(pc.rows)) return;
    uint pixels = uint(pc.side_px * pc.side_px);
    uint in_width = 3u * pixels;
    uint temporal = uint(pc.width) / in_width;
    uint ch, at;
    if (pc.order == 0) {
        ch = c_out / (temporal * pixels);
        at = (c_out % pixels) * 3u + ch;
    } else {
        uint k = c_out % in_width;
        ch = k % 3u;
        at = k;
    }
    float mean = ch == 0u ? pc.mean0 : (ch == 1u ? pc.mean1 : pc.mean2);
    float sd = ch == 0u ? pc.sd0 : (ch == 1u ? pc.sd1 : pc.sd2);
    float v = float(uint(x[n * in_width + at])) / 255.0;
    y[n * uint(pc.width) + c_out] = PIE_STORE((v - mean) / sd);
}

// pie:instantiate pixels_bf16
