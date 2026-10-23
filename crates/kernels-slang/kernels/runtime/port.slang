#include "prim/bf16.slang"
#include "prim/params.slang"

// A port fed from a device cell: the cell's `n` f32 values cast into the
// port's bf16 seat, which starts at element `at` of the input store. A seat
// may start and end mid-word, so the store binds whole from its start and a
// lane owns one word of it: a whole word is stored, an edge word's half
// outside the seat is left alone, through atomics, since another port's
// seat may hold it.

[[vk::binding(0)]] StructuredBuffer<float> x;
[[vk::binding(1)]] RWStructuredBuffer<Atomic<uint>> store;

struct Params {
    uint at;
    uint n;
};
PIE_PARAMS(2, Params, params);

void put_half(uint w, uint shift, float v) {
    store[w].and(~(0xffffu << shift));
    store[w].or(f32_to_bf16(v) << shift);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint end = params.at + params.n;
    uint w = (params.at >> 1) + gid.x;
    uint lo = 2u * w;
    if (lo >= end) {
        return;
    }
    bool has_lo = lo >= params.at;
    bool has_hi = lo + 1u < end;
    if (has_lo && has_hi) {
        store[w].store(pack_bf16(x[lo - params.at], x[lo + 1u - params.at]));
    } else if (has_lo) {
        put_half(w, 0u, x[lo - params.at]);
    } else {
        put_half(w, 16u, x[lo + 1u - params.at]);
    }
}

// pie:instantiate runtime_port_cast_bf16 PIE_GROUP_X=256
