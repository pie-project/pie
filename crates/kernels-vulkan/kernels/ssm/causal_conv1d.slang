// The short causal depthwise convolution over the recurrent conv slab.
//
// Two entrypoints, one per point. `causal_conv1d` is the decode step -- one
// arriving row per request and the rest of the window already in the slab --
// and `causal_conv1d_chunked` is the prefill window, a CSR range of rows per
// request with only the leading taps off the slab. They are the same
// arithmetic written twice and they stay written twice: the step's whole
// window is in the slab and the chunk's is almost entirely in `x`, so folding
// them would put a branch on the token axis inside the innermost loop of both,
// which is the loop that runs `K` times per channel per token.
//
// ── THE SLAB IS `[K, C]`, AND `C` IS THE FAST AXIS ─────────────────────────
//
// A slot's conv state is `conv_width * channels` floats: K ROWS of C channels,
// oldest row first, addressed `state[k * C + c]`. The rectangle this file
// indexes is that one and no other.
//
// K rows is the RECTANGLE and not the live window, and the difference is one
// whole row. Between fires only `K - 1` of the rows carry taps the next token
// convolves over: the step reads rows `1 .. K-1`, appends the arriving column,
// shifts every row down one and lands the arriving column at row `K - 1`. Row
// 0 is where the shift's tail goes -- read once, by the token that arrives
// before it is overwritten, and never again. Declaring `K - 1` rows would be
// stating the live WINDOW, which is a different number from the rectangle, and
// they differ by exactly the row this paragraph exists to keep.
//
// The numeric contract is `pie::ssm::causal_conv1d_update_batched` and
// `pie::ssm::causal_conv1d_prefill_batched` in
// `kernels-cuda/kernels/ssm/causal_conv1d.cuh`, which is where it was
// measured:
//
//     y[t, c] = silu( sum_{k=0..K-1} W[c, k] * x[t - K + 1 + k, c] )
//
// with `x[t < 0, c]` read from the slab at row `K + t`. There is no bias term:
// `ssm.causal_conv1d` declares no bias operand and the cuda claim body passes
// a null one, so the accumulator opens at zero rather than at `bias[c]`.
//
// ── WHY THE SHIFTED WINDOW LANDS IN A SECOND PLANE ────────────────────────
//
// `conv_state` is read and `new_conv_state` is written, never the reverse and
// never alternating; `driver-vulkan`'s `baker::stage::Slab::{Conv, NewConv}`
// are the two planes and the carry-back after the fire retires is what makes
// the written one the next fire's read one.
//
// The cuda kernels shift in place, and they may: one BLOCK owns a channel for
// the whole launch there, so the shift moves values between rows that one
// block alone reads. A Vulkan dispatch makes no such promise -- workgroups of
// one dispatch are unordered and there is no device-wide barrier inside it --
// so a channel whose taps are being read by one invocation cannot be the
// channel a shift is landing on in another. The second plane is what removes
// the question rather than answering it.
//
// A request the fire does not name keeps whatever its `new_conv_state` rows
// held, which the previous carry-back made equal to its `conv_state` rows, so
// the copy is the identity for it. That invariant is why the chunked arm may
// leave an empty window alone rather than copying it forward by hand.
//
// ── ONE INVOCATION OWNS ONE CHANNEL, WHICH IS NOT TRUE ON EVERY PLANE ─────
//
// `PIE_ACT` is `uint16_t` and a Slang `StructuredBuffer<uint16_t>` addresses a
// single bf16 -- `storageBuffer16BitAccess`, which every device this plane
// opens offers and every bf16 entrypoint in this tree already depends on. So a
// lane writes its own `y[.., c]` and nothing else, and this file needs neither
// the channel-PAIR ownership a plane whose smallest storage element is four
// bytes must use, nor the even-channel refusal that ownership forces onto the
// claim body. The slab is `float` and would need none of it either way.

#include "common/bf16.slang"

#define PIE_GROUP_X 64

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, weight);
[[vk::binding(2)]] StructuredBuffer<float> conv_state;
[[vk::binding(3)]] RWStructuredBuffer<float> new_conv_state;
[[vk::binding(4)]] StructuredBuffer<uint> slots;
#if defined(PIE_CHUNKED)
[[vk::binding(5)]] StructuredBuffer<int> indptr;
PIE_BUFFER_RW(6, y);
#else
PIE_BUFFER_RW(5, y);
#endif

// Eight bytes against the 128 `Device::max_push` guarantees. Both numbers are
// the claim body's -- `channels` is the convolved row's width, read off the
// operand, and `conv_width` is the one number `ssm.causal_conv1d` states.
struct Push {
    int channels;
    int conv_width;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

// SiLU, in the one spelling every plane's conv shares: `pie::ssm::silu_f`.
float silu(float z) { return z / (1.0 + exp(-z)); }

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const int chans = pc.channels;
    const int taps = pc.conv_width;
    const int c = int(tid.x);
    // The grid is `channels` lanes rounded up to the workgroup, so the tail of
    // the last workgroup owns no channel. There is no barrier below, so this
    // is a plain early return rather than a guard on the stores.
    if (c >= chans) return;
    const int r = int(tid.y);
    const int w_base = c * taps;

#if defined(PIE_CHUNKED)
    const int begin = indptr[r];
    const int end = indptr[r + 1];
    // An empty window leaves BOTH planes alone; see the header.
    if (end <= begin) return;
    const int span = end - begin;

    // Every token of a request sits in the same seat, so the window's first
    // row names it.
    const int slab = int(slots[begin]) * taps * chans;

    for (int t = 0; t < span; ++t) {
        float acc = 0.0;
        for (int k = 0; k < taps; ++k) {
            const int src = t - (taps - 1) + k;
            const float v = src < 0
                ? conv_state[slab + (taps + src) * chans + c]
                : PIE_LOAD(x[(begin + src) * chans + c]);
            acc += v * PIE_LOAD(weight[w_base + k]);
        }
        y[(begin + t) * chans + c] = PIE_STORE(silu(acc));
    }

    // The trailing `K` rows of `[slab window | x window]`, oldest first, which
    // is where a follow-up step resumes from. A window shorter than `K` takes
    // its leading rows off the slab, which is the same expression the
    // convolution above reads them with.
    for (int s = 0; s < taps; ++s) {
        const int src = span - taps + s;
        new_conv_state[slab + s * chans + c] = src < 0
            ? conv_state[slab + (taps + src) * chans + c]
            : PIE_LOAD(x[(begin + src) * chans + c]);
    }
#else
    const int slab = int(slots[r]) * taps * chans;
    const int arriving = r * chans + c;
    const float fresh = PIE_LOAD(x[arriving]);

    // Rows `1 .. K-1` of the slab, then the arriving column at tap `K - 1`.
    float acc = fresh * PIE_LOAD(weight[w_base + taps - 1]);
    for (int k = 0; k + 1 < taps; ++k) {
        acc += conv_state[slab + (k + 1) * chans + c] * PIE_LOAD(weight[w_base + k]);
    }
    y[arriving] = PIE_STORE(silu(acc));

    // Shift every row down one and land the arriving column at row `K - 1`.
    for (int k = 0; k + 1 < taps; ++k) {
        new_conv_state[slab + k * chans + c] = conv_state[slab + (k + 1) * chans + c];
    }
    new_conv_state[slab + (taps - 1) * chans + c] = fresh;
#endif
}

// pie:instantiate causal_conv1d_bfloat16
// pie:instantiate causal_conv1d_chunked_bfloat16 PIE_CHUNKED=1
