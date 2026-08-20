// Weightless RMSNorm: `out = x / rms(x)`.
//
// This follows `norm/rms.slang`'s Vulkan reduction rather than Metal's fixed
// simdgroup width. The arithmetic and strided N_READS loop are the Metal port's:
// fp32 accumulate, one bf16 round on store, and one workgroup per row.
//
// THE TWO SCALARS ARE A PUSH BLOCK, NOT A STORAGE STRUCT. `VNormParams { eps,
// axis_size }` was Metal's buffer-2 layout carried over whole, and a storage
// block for eight live bytes cost this plane a descriptor, a staging slot and
// the sentinel `Holds::params_block` mints for a buffer with no address.
// `vnorm_single_row` states `eps: Const<f32>` and `axis_size: Const<i32>`
// instead -- words 0 and 1 of the same statement run the struct was staged
// from, in the struct's own order because that order is the statement's -- and
// eight bytes is a push range on every device this targets.
//
// `Encoder::words` builds the block from the arguments the BODY passes, in the
// order it passes them, so the two fields below are `eps` then `axis_size` and
// nothing may go between them: `Device::dispatch` refuses a push run whose
// length is not exactly the pipeline's range, but it cannot see two four-byte
// words that swapped places.

#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef N_READS
#define N_READS 4
#endif

#define PIE_GROUP 256

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RW(1, out_);

struct Push { float eps; uint axis_size; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint axis_size = pc.axis_size;
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
    float inv = pie_inv_rms(acc, axis_size, pc.eps);

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
