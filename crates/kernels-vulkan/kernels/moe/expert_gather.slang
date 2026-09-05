[[vk::binding(0)]] StructuredBuffer<uint> src;
[[vk::binding(1)]] RWStructuredBuffer<uint> dst;
[[vk::binding(2)]] StructuredBuffer<int> routes;
[[vk::binding(3)]] RWStructuredBuffer<int> seat_routes;

struct Push {
    int words;
    int seats;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int seat = int(gid.y);
    int word = int(gid.x);
    if (seat >= pc.seats) {
        return;
    }
    if (word == 0) {
        seat_routes[seat] = seat;
    }
    int expert = routes[seat];
    if (expert < 0 || word >= pc.words) {
        return;
    }
    dst[uint(seat) * uint(pc.words) + uint(word)] = src[uint(expert) * uint(pc.words) + uint(word)];
}

// pie:instantiate expert_gather PIE_GROUP_X=256
