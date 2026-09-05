#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef N_READS
#define N_READS 4
#endif
#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, w);
PIE_BUFFER_RW(2, out_);

struct Push {
    float eps;
    int axis_size;
    uint w_stride;
    uint plus_one;
    float gain;
#if defined(PIE_GROUPED)

    uint groups;
#endif
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

float gain_at(uint i, uint plane) {
    float wv = PIE_LOAD(w[plane + pc.w_stride * i]);
    return pc.gain * (pc.plus_one != 0u ? (1.0 + wv) : wv);
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint axis_size = uint(pc.axis_size);
    uint base = group.x * axis_size;
#if defined(PIE_GROUPED)
    uint plane = (group.x % pc.groups) * axis_size * pc.w_stride;
#else
    uint plane = 0u;
#endif
    uint span = uint(PIE_GROUP_X) * uint(N_READS);
    uint lid = local.x;

    float acc = 0.0;
    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        [ForceUnroll] for (uint i = 0u; i < uint(N_READS); i++) {
            if (start + i < axis_size) {
                float xi = PIE_LOAD(x[base + start + i]);
                acc += xi * xi;
            }
        }
    }
    float inv = pie_inv_rms(acc, axis_size, pc.eps);

    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        [ForceUnroll] for (uint i = 0u; i < uint(N_READS); i++) {
            uint at = start + i;
            if (at < axis_size) {
                out_[base + at] = PIE_STORE(gain_at(at, plane) * (PIE_LOAD(x[base + at]) * inv));
            }
        }
    }
}

// pie:instantiate rms_single_row_bf16 PIE_GROUP_X=256 N_READS=4
// pie:instantiate rms_grouped_row_bf16 PIE_GROUP_X=256 N_READS=4 PIE_GROUPED=1
