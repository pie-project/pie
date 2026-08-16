// Weightless RMSNorm: `out = x / rms(x)`.
//
// This follows `norm/rms.slang`'s Vulkan reduction rather than Metal's fixed
// simdgroup width. The arithmetic and strided N_READS loop are the Metal port's:
// fp32 accumulate, one bf16 round on store, and one workgroup per row.

#include "common/bf16.slang"
#include "common/reduce.slang"
#include "norm/rms_params.slang"

#ifndef N_READS
#define N_READS 4
#endif

#define PIE_GROUP 256

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RW(1, out_);
PIE_PARAMS(2, VNormParams, p);

[numthreads(PIE_GROUP, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint axis_size = p.axis_size;
    uint base = group.x * axis_size;
    uint lid = local.x;
    uint span = uint(PIE_GROUP) * uint(N_READS);

    float acc = 0.0;
    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        [ForceUnroll] for (uint i = 0u; i < uint(N_READS); i++) {
            if (start + i < axis_size) {
                float xi = PIE_LOAD(x[base + start + i]);
                acc += xi * xi;
            }
        }
    }
    float inv = pie_inv_rms(acc, axis_size, p.eps);

    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        [ForceUnroll] for (uint i = 0u; i < uint(N_READS); i++) {
            uint at = start + i;
            if (at < axis_size) {
                out_[base + at] = PIE_STORE(PIE_LOAD(x[base + at]) * inv);
            }
        }
    }
}

// pie:instantiate vnorm_single_row_bfloat16 N_READS=4
