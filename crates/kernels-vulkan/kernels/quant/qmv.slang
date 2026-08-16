// Affine GEMV projections, ported from `quant/qmv.metal`.
//
// The port keeps MLX's little-endian affine packing, the four-rows-per-simdgroup
// launch shape, and the bounds-checked K tail whose absence broke Gemma-4-31B at
// K=5376. Vulkan does not promise a 32-wide subgroup, so the final reduction is
// through shared memory over the 32 local-x lanes rather than subgroup width.
//
// The Slang port changed spellings only. The two bodies used to read
// `gl_LocalInvocationID` and `gl_WorkGroupID` directly; they now take the two
// index vectors as parameters, which is the whole tree's rule -- a helper that
// reads a thread builtin is a helper that cannot be called from a second place
// without carrying its caller's grid with it.

#include "common/bf16.slang"
#include "common/affine.slang"

#ifndef PIE_VEC
#define PIE_VEC 1
#endif

[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
PIE_BUFFER_RO(3, x);
PIE_BUFFER_RW(4, y);
// Declared only by the variants whose ROW has a sixth buffer operand, and at 5
// rather than 7. Both halves of that are load-bearing.
//
// `kernels_vulkan::bindings()` numbers buffer-kinded operands densely from zero
// in the row's operand order, so `affine_qmv_fast_residual`'s residual is
// binding 5 -- while `affine_qmv_fast` has five operands and no sixth binding
// at all, which is why this cannot simply be declared unconditionally.
//
// This read binding 7 for every variant until an audit compiled each one and
// compared its `OpDecorate Binding` set against the table
// (`scripts/vulkan-kernel-audit.py --bindings`, since deleted). 7 is the Metal
// buffer index: Metal numbers scalars in the same run as buffers, Vulkan sends
// them to a push block, and the two numbers differ by however many scalars
// precede the operand. They must never be copied across, and nothing checks
// that they were not.
#if defined(PIE_RESIDUAL) || defined(PIE_BIAS)
PIE_BUFFER_RO(5, extra);
#endif

// The block is variant-shaped, and it has to be. A push block is not like a
// descriptor set, where declaring more than you read is free: the block a
// module declares must fit inside the pipeline layout's push range
// (`VUID-VkComputePipelineCreateInfo-layout-10069`), and that range is built
// from what the ROW states. `affine_qmv_fast` states two scalars, so a
// module that declares seven -- as this one did until a validation layer
// said so -- is asking the shell for twenty bytes no row will ever push.
//
// `row_stride` and `m` therefore appear only for the strided variant that
// reads them. Three more fields (`x_slot_stride`, `x_row_stride`,
// `slots_per_row`) were declared here and read by nothing at all; they were
// the routed row's shape, copied in and never removed.
struct Push {
    int in_vec_size;
    int out_vec_size;
#if defined(PIE_WIDE_STRIDED)
    int row_stride;
    int m;
#endif
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

// The K lanes. `quant/qmv.slang` splits a row's K across `PIE_LANES` threads
// and `qmv_grid` in `src/quant.rs` multiplies the batch extent by the same
// number -- the two are one decision written twice, and a change to either
// alone silently launches the wrong grid.
//
// It was 32, inherited from the Metal port's simdgroup width; the workgroup was
// therefore 64 threads, two warps. 64 lanes makes it 128 and doubles the
// threads a projection launches, without changing how many WORKGROUPS it
// launches -- `lanes` is divided by the module's own local size, so both
// numbers moving together leaves the group count fixed.
//
// Measured, it is a wash: 37.96 microseconds a dispatch at 32 lanes against
// 37.7 at 64, over 56541 fires of a real qwen3-0.6B decode. It is kept at 64
// because two warps per workgroup cannot use a barrier without half the group
// idling through it, not because it was faster. What this measurement DOES
// establish is that this kernel is not short of resident warps -- doubling
// them changed nothing -- which rules occupancy out as the remaining limit. It
// was read at the time as pointing at split-K; the `floor K` rows of
// `tests/qmv_bench.rs` have since ruled that out too, and the note above
// `dot_lane4` says how.
#define PIE_LANES 64

groupshared float partials[8 * PIE_LANES];

uint weight_words_per_row() { return uint(pc.in_vec_size) * uint(PIE_BITS) / 32u; }
uint groups_per_row() { return uint(pc.in_vec_size) / uint(PIE_GROUP); }

float affine_value(uint row, uint k) {
    uint word = w[row * weight_words_per_row() + k / uint(PIE_CODES_PER_WORD)];
    float code = pie_affine_code(word, k % uint(PIE_CODES_PER_WORD));
    uint g = k / uint(PIE_GROUP);
    return code * PIE_LOAD(scales[row * groups_per_row() + g])
         + PIE_LOAD(biases[row * groups_per_row() + g]);
}

float x_at(uint vec, uint k) {
#if defined(PIE_WIDE_STRIDED)
    return PIE_LOAD(x[vec * uint(pc.row_stride) + k]);
#else
    return PIE_LOAD(x[vec * uint(pc.in_vec_size) + k]);
#endif
}

// The chunk of activations a workgroup stages in shared memory at a time.
//
// 1024 floats is 4 KB, and it is a whole K of 1024 in one pass. Together with
// `partials` the group asks for 6 KB, well inside the 32 KB a workgroup may
// have anywhere this runs.
#define PIE_XCHUNK 1024

// Declared for every variant, including the wide strided one that does not
// call `dot_lane4` -- the whole file is compiled for every entry point, and a
// groupshared array no reachable code touches is one slangc drops.
groupshared float xs_sh[PIE_XCHUNK];


// One pass over K for all FOUR of a lane's rows, reading coalesced.
//
// Everything here was measured on a 4090 against a real 4-bit qwen3-0.6B
// decode of 452 dispatches, where this entry point and its residual twin are
// ninety per cent of the step. Four defects, in the order they were found --
// and only the first two paid, which is the more useful half of the record:
//
//  1. `affine_value` re-read `w[]` for every CODE, and a word holds eight of
//     them at four bits, so the dominant tensor was fetched eight times over
//     and its scales sixteen. `pie_affine_word_dot` is the primitive that
//     already existed for exactly this and that nothing called.
//     Decode step 25.2 ms -> 20.9 ms.
//  2. The four rows were four independent passes over K, so the activation
//     vector was read four times over and no two rows' weight loads were ever
//     in flight together. They are one pass now.  20.9 ms -> 11.0 ms.
//  3. The addresses were thread-contiguous rather than lane-contiguous: each
//     lane owned a RUN of K and walked it, so one warp-wide load instruction
//     touched thirty-two addresses 32 bytes apart -- eight cache lines to
//     serve 64 useful bytes. A warp now reads thirty-two CONSECUTIVE words per
//     instruction by striding the word index by the lane count.
//  4. The activations were read from global memory inside the inner loop as
//     16-bit scalars at a 16-element stride, the same defect in its worst
//     form. They are staged into `xs_sh` once per chunk by a coalesced
//     cooperative load and read from shared thereafter.
//
// 3 and 4 together moved a dispatch from 37.96 microseconds to 37.7, which is
// noise. They are kept because they are not slower and because the access
// pattern they establish is the one a split-K form will need -- but the honest
// reading is that this kernel's remaining cost is NOT its access pattern, and
// an L1 that holds a 6 KB activation vector was already absorbing the defect.
//
// # Split-K would buy nothing, and that is now measured rather than reasoned
//
// The paragraph above and the `PIE_LANES` note both end by pointing at
// split-K: occupancy was ruled out by doubling the warps and the access
// pattern by fixing it, so what was left looked like too little work per
// dispatch to fill the card. `tests/qmv_bench.rs` asked that directly and the
// answer is no.
//
// A barrier-separated dispatch of this kernel costs ~6.14 us on an RTX PRO
// 4500, and it costs the SAME 6.14 us at one workgroup as at 256 (`floor`),
// and the same again at K=64 as at K=1024 (`floor K`) -- sixteen times less
// work for an identical price. A cost that does not move when the work moves
// by 16x or the parallelism by 256x is not a work cost, and split-K only
// manufactures more workgroups.
//
// The same dispatches unbarriered cost 0.78 us (`overlap`), which is 640 GB/s
// -- so the kernel is close to what the card can do and ~5.4 us of every
// dispatch is the barrier in front of it. An execution-only dependency still
// costs ~4-6 us, so the drain is intrinsic and cannot be masked away either.
//
// What is left is the barrier COUNT. A decode records 311 of them, 91%
// read-after-write, which is what a forward pass is; at ~6 us each that is
// most of a 2.9 ms step. The way to a faster decode here is fewer serialized
// dispatches -- fusing a norm into the projection that reads it, a rope into
// the append that follows it -- and not a faster matvec.
//
// `sum(x)` does not depend on the row, so it is computed once per word and
// handed to all four rows -- see `pie_affine_word_dot_presummed`.
//
// A group boundary never falls inside a word -- `PIE_GROUP` is 32, 64 or 128
// and a word holds 4 or 8 codes -- so one scale and one bias serve the whole
// word, which is what makes the hoist legal rather than merely cheaper.
void dot_lane4(uint out0, uint vec, uint tid, uint lid, out float acc[4]) {
    uint cpw = uint(PIE_CODES_PER_WORD);
    uint n = uint(pc.in_vec_size);
    uint wpr = weight_words_per_row();
    uint gpr = groups_per_row();
    uint rows = uint(pc.out_vec_size);
    uint threads = uint(PIE_LANES) * 2u;
    uint words_per_chunk = uint(PIE_XCHUNK) / cpw;
    [ForceUnroll] for (uint r = 0u; r < 4u; r++) acc[r] = 0.0;

    for (uint base = 0u; base < n; base += uint(PIE_XCHUNK)) {
        // Before the write, not only after it: the previous chunk's readers
        // must be done with `xs_sh` before this one overwrites it.
        GroupMemoryBarrierWithGroupSync();
        for (uint i = tid; i < uint(PIE_XCHUNK); i += threads) {
            xs_sh[i] = base + i < n ? x_at(vec, base + i) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();

        uint w0 = base / cpw;
        for (uint j = lid; j < words_per_chunk; j += uint(PIE_LANES)) {
            uint kb = base + j * cpw;
            if (kb >= n) break;
            float xw[PIE_CODES_PER_WORD];
            float xsum = 0.0;
            [ForceUnroll] for (uint i = 0u; i < cpw; i++) {
                xw[i] = xs_sh[j * cpw + i];
                xsum += xw[i];
            }
            uint g = kb / uint(PIE_GROUP);
            uint at = w0 + j;
            [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
                uint row = out0 + r;
                if (row < rows) {
                    acc[r] += pie_affine_word_dot_presummed(
                        w[row * wpr + at], xw, xsum,
                        PIE_LOAD(scales[row * gpr + g]),
                        PIE_LOAD(biases[row * gpr + g]));
                }
            }
        }
    }
}

void reduce_store(bool tail, uint3 group, uint3 local) {
    uint lid = local.x;
    uint ly = local.y;
    uint out0 = group.y * 8u + ly * 4u;
    uint vec = group.x;

    float acc[4];
    dot_lane4(out0, vec, ly * uint(PIE_LANES) + lid, lid, acc);
    [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
        partials[(ly * 4u + r) * uint(PIE_LANES) + lid] = acc[r];
    }
    GroupMemoryBarrierWithGroupSync();

    // A tree, not a walk.
    //
    // This used to be `if (lid == 0) for i in 0..32` repeated for each of the
    // lane's four rows: 128 serial shared-memory adds with thirty-one of the
    // thirty-two lanes idle, at the end of every dispatch. Five halving rounds
    // do the same four sums in twenty adds with the lanes spread across them.
    // The barrier is outside the `if` on purpose -- a group-sync reached by
    // only some of the group is undefined, not merely slow.
    [ForceUnroll] for (uint s = uint(PIE_LANES) / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
                uint at = (ly * 4u + r) * uint(PIE_LANES);
                partials[at + lid] += partials[at + lid + s];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid == 0u) {
        for (uint r = 0u; r < 4u; r++) {
            uint row = out0 + r;
            if (row < uint(pc.out_vec_size)) {
                float sum = partials[(ly * 4u + r) * uint(PIE_LANES)];
#if defined(PIE_BIAS)
                if (tail) sum += PIE_LOAD(extra[row]);
#endif
#if defined(PIE_RESIDUAL)
                uint at = vec * uint(pc.out_vec_size) + row;
                PIE_ACT q = PIE_STORE(sum);
                y[at] = PIE_STORE(PIE_LOAD(q) + PIE_LOAD(extra[at]));
#else
                y[vec * uint(pc.out_vec_size) + row] = PIE_STORE(sum);
#endif
            }
        }
    }
}

// AN RMSNORM-FUSED MATVEC WAS ALSO WRITTEN HERE AND MEASURED AWAY, AND IT IS
// THE MORE INTERESTING OF THE TWO, BECAUSE IT DID EXACTLY WHAT IT SET OUT TO
// DO AND THE STEP GOT SLOWER ANYWAY.
//
// The argument was the strongest one in this driver, and it is written down
// in `hazards` in `driver-vulkan/src/device.rs`: patching every barrier out
// of a qwen3-0.6b decode takes the same 452 dispatches from 1.517 ms to
// 0.460, so ordering is two thirds of a step, the unit that costs is the
// STAGE rather than the dispatch, and a layer's eleven stages are ~0.14 ms
// each. `attn_norm` is read by q, k and v and by nothing else, so folding it
// into their prologue lets the forward text stop stating it: one stage off
// the layer, for a redundant pass over K that the traffic table below says is
// nearly free.
//
// It worked. `affine_qmv_fast_rmsnorm` was written on both backends, the
// plan fell from 452 launches to 424, and -- once `model-compiler`'s arena
// stopped handing `k_norm` the block `q_norm` had just freed, which is a
// write-after-read this driver has to order and which had been silently
// eating the win -- the barrier count fell from 311 to 283. Exactly one
// stage a layer, which is what was predicted.
//
// Three configurations, median device ms a step, same binary, same box,
// three runs each and the spreads do not overlap:
//
//   | configuration                | barriers | ms/step |
//   |------------------------------|----------|---------|
//   | as it ships                  |      311 |   2.757 |
//   | + arena cooling              |      311 |   2.807 |
//   | + arena cooling + norm fusion|      283 |   2.846 |
//
// A stage came off and the step got 0.09 ms SLOWER. Both halves cost:
// holding a dead block out of the arena for two ops buys nothing on its own
// (the count does not move without the fusion) and still costs 0.05 ms in
// bytes and locality, and the fused kernel costs another 0.04 ms because the
// norm is not a prologue at all once it is written -- the gain multiply and
// the bf16 round land on EVERY ELEMENT of every chunk in the inner loop, and
// the workgroup reduce that finds the inverse RMS puts a barrier inside every
// threadgroup. A projection's threadgroups are its output rows over eight, so
// q's 2048 rows recompute that reduction 256 times.
//
// WHAT IS REFUTED, MEASURED RATHER THAN GUESSED. The first suspicion was the
// pricing -- that "0.14 ms a stage" is an average over all 311 barriers and
// would not survive being applied to one. So the stage was priced on its own,
// with no kernel involved: drop one hazard in every N and time the step.
//
//   | barriers dropped | ms/step | delta | us/barrier |
//   |------------------|---------|-------|------------|
//   | 0 (all 311 kept) |   2.810 |    -- |         -- |
//   | 28 (one a layer) |   2.711 | 0.099 |        3.5 |
//   | 52               |   2.595 | 0.215 |        4.1 |
//   | 104              |   2.519 | 0.291 |        2.8 |
//   | 156              |   2.186 | 0.624 |        4.0 |
//
// Linear, and 3.7 us a barrier against the 3.4 the average predicted. The
// currency is SOUND and the stage this kernel removed was worth its 0.10 ms.
// What sank it is the side nothing had priced: the kernel added about 0.13 ms
// of arithmetic to buy that 0.10.
//
// So the lesson is a BUDGET, not a prohibition. A merge is worth ~0.10 ms a
// stage and has to add less work than that, and the way to lose is exactly
// what happened here -- a projection's threadgroups are its output rows over
// eight, so folding a reduction over K into its prologue makes every one of
// them redo it, 256 times over for q, on top of a gain multiply and a bf16
// round on every element of the inner loop. A merge that only touches
// elements the kernel already reads is free and this one was not. Of the
// three left in `hazards`, kv-append-into-rope and the qk-norms-into-rope are
// free by construction; swiglu into gate/up is not, and should be measured
// before it is written.
//
// A BATCHED MATVEC WAS WRITTEN HERE AND MEASURED AWAY. DO NOT WRITE IT AGAIN.
//
// The kernel is gone -- `tests/entrypoints.rs` holds that a built entrypoint
// is a dispatched entrypoint, and this one would never have been dispatched.
// What is kept is the reasoning that led to it and the measurement that
// killed it, because the reasoning is persuasive and wrong, and the next
// person to look at a decode batch will arrive at it again unaided.
//
// # Why it looked necessary
//
// `dot_lane4` above computes four output rows for ONE activation vector, and a
// batch is served by launching it once per vector along the grid's x extent.
// Every one of those launches re-reads the entire weight plane. `tests/
// qmv_bench.rs` measures what that costs: the ffn gate/up projection is
// 6.15 us at one token and 78.96 at thirty-two -- thirteen times the price for
// thirty-two times the work, where a form that read its weights once would
// have stayed near flat. A whole decode step is 2.75 ms of device at batch 1
// and 8.35 at batch 8 for the same reason: the fixed costs amortise and the
// projections, which are 70% of a step, do not.
//
// The tiled GEMM is not the answer to that and this is not a duplicate of it.
// `affine_qmm_t` has a floor of ~102 us against this kernel's 6.14 -- it
// amortises a large setup over hundreds of prefill rows and has nothing to
// offer at ten. What a decode batch wants is this kernel's launch cost with
// the GEMM's weight reuse, which is what the loop nest below is.
//
// # Why the premise is false
//
// The kernel was built -- `PIE_MROWS` activation vectors to a workgroup,
// each weight word fetched once and applied to all of them, the codes
// unpacked once -- and measured against the single form it was to replace,
// at a batch of thirty-two, walking the weight plane from well inside this
// card's 72 MB of L2 to well outside it:
//
//     plane      1.5 MB    8 MB    24 MB    64 MB   128 MB
//     batched     77.68  332.28   909.41  2587.83  5249.42
//     single      72.03  313.36   864.65  2724.79  5693.45
//
// The batched form loses at every plane that fits in L2 and wins by five to
// eight percent at the two that do not. Every projection
// in every model this tree targets is on the losing side of that line: the
// 0.6B's ffn gate/up plane is 1.5 MB, and a 4096-by-12288 ffn -- gemma4's
// shape -- is 24 MB.
//
// The reason is in the last two columns rather than the first three. If the
// re-read were reaching DRAM, sharing a word across two vectors would halve
// the traffic and roughly halve the time. It moves it by eight percent. At
// the 128 MB plane the single form moves 128 MB of weights in 5.69 ms, which
// is 24 GB/s of real DRAM traffic on a card that streams thirty times that.
// The thirty-two workgroups reading one plane are co-resident in time, so the
// plane streams through L2 once and all thirty-two rows consume it there no
// matter how large it is. The re-read is an L2 hit, and an L2 hit is not what
// the matvec is spending its time on.
//
// So the decode matvec is not weight-fetch-bound and never was. It is bound
// by the arithmetic and the L2-to-SM issue rate, both of which scale with the
// batch and neither of which a weight-sharing loop nest removes -- sharing
// the word saves the unpack, not the fused multiply-add. Holding the extra
// vectors costs occupancy, which is the five percent it loses by inside L2.
// `PIE_MROWS` was tried at four before two: four is worse again (26.63 us
// against 16.37 at sixteen tokens), which is the register pressure of
// `xw[4][8] + xsum[4] + acc[4][4]` showing up directly. Two was the widest
// that stayed within a few percent, and a few percent behind is still behind.
//
// The half of that table which does NOT depend on the deleted kernel is kept
// running in `qmv_bench.rs` under "what the matvec's weight traffic actually
// costs": the single form's real DRAM rate across the same planes. As long
// as that stays an order of magnitude under what the card can stream, no
// weight-sharing rewrite of this kernel can pay.
//
// What survives from this is a redirection, not a kernel. The projections
// really are 70% of a decode step and really do fail to amortise across a
// batch -- but the cost to attack is the per-launch and per-barrier one
// (a decode records 311 barriers, ~1.9 ms of a 2.9 ms step), which means
// FUSION: fewer stages, not cheaper reads inside a stage.

#if defined(PIE_WIDE_STRIDED)
void wide_strided(uint3 group, uint3 local) {
    uint row_slot = local.y * 4u + (local.x & 3u);
    if (local.x >= 4u) return;
    uint row = group.y * 8u + row_slot;
    uint vec0 = group.x * uint(PIE_VEC);
    if (row >= uint(pc.out_vec_size)) return;
    for (uint v = 0u; v < uint(PIE_VEC); v++) {
        uint vec = vec0 + v;
        if (vec < uint(pc.m)) {
            float acc = 0.0;
            for (uint k = 0u; k < uint(pc.in_vec_size); k++) {
                acc += x_at(vec, k) * affine_value(row, k);
            }
            y[vec * uint(pc.row_stride) + row] = PIE_STORE(acc);
        }
    }
}
#endif

[numthreads(PIE_LANES, 2, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
#if defined(PIE_WIDE_STRIDED)
    wide_strided(group, local);
#elif defined(PIE_TAIL)
    reduce_store(true, group, local);
#else
    reduce_store(false, group, local);
#endif
}

// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8 PIE_GROUP=64 PIE_BITS=4 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_8_v_4_kl_8 PIE_GROUP=64 PIE_BITS=8 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
