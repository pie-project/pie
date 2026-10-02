#include "common/bf16.slang"

// A blockwise butterfly FWHT over the last dim, in place. The flat input is cut
// into contiguous N-element blocks (N a power of two, 64..1024). Each block is
// replaced by H . x_block, where H is the normalized Sylvester-Walsh-Hadamard
// matrix whose entries are (-1)^popcount(i & j) / sqrt(N). The pure transform
// is orthonormal and self-inverse (H . H = I).
//
// When a sign buffer is bound (signs_width != 0) each element is multiplied by
// its sign on load, before the butterfly -- this realizes the Randomized
// Hadamard H . S, which is not self-inverse. The sign vector repeats block-wise:
// the element at flat offset `p` reads signs[p % signs_width].
//
// Portable translation of the Metal `fwht_simd` / `fwht_tg` kernels: one
// workgroup owns one N-block, the block lives in groupshared, and the radix-2
// Sylvester stages (stride b = 1, 2, ..., N/2) run through groupshared with a
// barrier between them. No subgroup shuffle is used -- barriers are correct and
// portable (lavapipe). The butterfly convention is Metal's exactly: with
// `partner = block[g ^ b]`, `block[g] = ((g & b) == 0) ? partner + self
// : partner - self`.

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

// The largest block the groupshared scratch (and the per-thread register fan)
// must hold. N is a runtime push constant that never exceeds this.
#ifndef PIE_N_MAX
#define PIE_N_MAX 1024
#endif

#if defined(PIE_F32)
#define PIE_T float
#define PIE_C_LOAD(v) (v)
#define PIE_C_STORE(f) (f)
#else
#define PIE_T PIE_ACT
#define PIE_C_LOAD(v) PIE_LOAD(v)
#define PIE_C_STORE(f) PIE_STORE(f)
#endif

// Upper bound on the elements a single thread touches in one block.
#define PIE_VPT ((PIE_N_MAX + PIE_GROUP_X - 1) / PIE_GROUP_X)

[[vk::binding(0)]] RWStructuredBuffer<PIE_T> x;
[[vk::binding(1)]] StructuredBuffer<PIE_T> signs;

struct Push { uint block_size; uint signs_width; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pie_fwht_block[PIE_N_MAX];

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = pc.block_size;
    uint tid = local.x;
    uint base = group.x * n;
    float scale = 1.0 / sqrt(float(n));

    // Load the block into groupshared, folding in sign * scale before the
    // butterfly (sign on load, scale once), matching Metal's `float(x[p]) * s
    // * scale`.
    for (uint w = tid; w < n; w += uint(PIE_GROUP_X)) {
        uint p = base + w;
        float s = 1.0;
        if (pc.signs_width != 0u) {
            s = PIE_C_LOAD(signs[p % pc.signs_width]);
        }
        pie_fwht_block[w] = PIE_C_LOAD(x[p]) * s * scale;
    }

    // Radix-2 Sylvester stages. Each stage reads the whole block's pre-stage
    // values (first barrier makes the prior stage's writes visible), computes
    // every result into registers, then the second barrier gates the write-back
    // so no partner is clobbered before it is read.
    for (uint b = 1u; b < n; b <<= 1) {
        GroupMemoryBarrierWithGroupSync();
        float acc[PIE_VPT];
        uint idx = 0u;
        for (uint w = tid; w < n; w += uint(PIE_GROUP_X)) {
            float self_ = pie_fwht_block[w];
            float partner = pie_fwht_block[w ^ b];
            acc[idx] = ((w & b) == 0u) ? (partner + self_) : (partner - self_);
            idx += 1u;
        }
        GroupMemoryBarrierWithGroupSync();
        idx = 0u;
        for (uint w = tid; w < n; w += uint(PIE_GROUP_X)) {
            pie_fwht_block[w] = acc[idx];
            idx += 1u;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    for (uint w = tid; w < n; w += uint(PIE_GROUP_X)) {
        x[base + w] = PIE_C_STORE(pie_fwht_block[w]);
    }
}

// pie:instantiate hadamard_bf16 PIE_GROUP_X=256 PIE_N_MAX=1024
// pie:instantiate hadamard_f32 PIE_GROUP_X=256 PIE_N_MAX=1024 PIE_F32=1
