#define PIE_MAX_NGRAM 4
#define PIE_MAX_HEADS 32
#define PIE_GROUP_X 64

void ple_mask_window(inout int window[PIE_MAX_NGRAM], int ngram, int eos) {
    bool crossed = false;
    for (int p = 1; p < ngram; p++) {
        if (crossed) window[p] = eos;
        if (window[p] == eos) crossed = true;
    }
}

void ple_hash_row(StructuredBuffer<uint64_t> hash, int window[PIE_MAX_NGRAM], int ngram, int heads,
                  int heads_per_ngram, out int out_[PIE_MAX_HEADS]) {
    for (int k = 0; k < PIE_MAX_HEADS; k++) out_[k] = 0;
    for (int order = 2; order <= ngram; order++) {
        uint64_t mixed = uint64_t(int64_t(window[0])) * hash[0];
        for (int p = 1; p < order; p++) mixed ^= uint64_t(int64_t(window[p])) * hash[uint(p)];
        int base = (order - 2) * heads_per_ngram;
        for (int k = 0; k < heads_per_ngram; k++) {
            int head = base + k;
            uint64_t prime = hash[uint(ngram + head)];
            uint64_t offset = hash[uint(ngram + heads + head)];
            out_[head] = int(mixed % prime + offset);
        }
    }
}

int ple_cell(int cell, int eos) {
    return cell == 0 ? eos : cell - 1;
}

#if defined(PIE_PLE_UPDATE)

[[vk::binding(0)]] StructuredBuffer<int> ids;
[[vk::binding(1)]] RWStructuredBuffer<int> state;
[[vk::binding(2)]] StructuredBuffer<uint> slots;
[[vk::binding(3)]] StructuredBuffer<uint64_t> hash;
[[vk::binding(4)]] RWStructuredBuffer<int> ngram_ids;
struct Push { int ngram; int heads; int heads_per_ngram; int eos; int rows; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint r = gid.x;
    if (r >= uint(pc.rows)) return;
    int span = pc.ngram - 1;
    uint slab = slots[r] * uint(span);
    int window[PIE_MAX_NGRAM];
    for (int p = 0; p < PIE_MAX_NGRAM; p++) window[p] = 0;
    int fresh = ids[r];
    window[0] = fresh;
    for (int p = 1; p <= span; p++) window[p] = ple_cell(state[slab + uint(span - p)], pc.eos);
    ple_mask_window(window, pc.ngram, pc.eos);
    int out_[PIE_MAX_HEADS];
    ple_hash_row(hash, window, pc.ngram, pc.heads, pc.heads_per_ngram, out_);
    for (int k = 0; k < pc.heads; k++) ngram_ids[r * uint(pc.heads) + uint(k)] = out_[k];
    for (int p = 0; p + 1 < span; p++) state[slab + uint(p)] = state[slab + uint(p + 1)];
    state[slab + uint(span - 1)] = fresh + 1;
}

#elif defined(PIE_PLE_CHUNKED)

[[vk::binding(0)]] StructuredBuffer<int> ids;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] RWStructuredBuffer<int> state;
[[vk::binding(3)]] StructuredBuffer<uint> slots;
[[vk::binding(4)]] StructuredBuffer<uint64_t> hash;
[[vk::binding(5)]] RWStructuredBuffer<int> ngram_ids;
struct Push { int ngram; int heads; int heads_per_ngram; int eos; int lanes; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint r = gid.x;
    if (r >= uint(pc.lanes)) return;
    int begin = indptr[r];
    int end = indptr[r + 1u];
    if (end <= begin) return;
    int rows = end - begin;
    int span = pc.ngram - 1;
    uint slab = slots[uint(begin)] * uint(span);
    for (int t = 0; t < rows; t++) {
        int window[PIE_MAX_NGRAM];
        for (int p = 0; p < PIE_MAX_NGRAM; p++) window[p] = 0;
        window[0] = ids[uint(begin + t)];
        for (int p = 1; p <= span; p++) {
            window[p] = (t - p >= 0) ? ids[uint(begin + t - p)] : ple_cell(state[slab + uint(span - (p - t))], pc.eos);
        }
        ple_mask_window(window, pc.ngram, pc.eos);
        int out_[PIE_MAX_HEADS];
        ple_hash_row(hash, window, pc.ngram, pc.heads, pc.heads_per_ngram, out_);
        for (int k = 0; k < pc.heads; k++) ngram_ids[uint(begin + t) * uint(pc.heads) + uint(k)] = out_[k];
    }
    int next[PIE_MAX_NGRAM];
    for (int p = 0; p < PIE_MAX_NGRAM; p++) next[p] = 0;
    for (int p = 0; p < span; p++) {
        int src = rows - span + p;
        next[p] = src >= 0 ? ids[uint(begin + src)] + 1 : state[slab + uint(p + rows)];
    }
    for (int p = 0; p < span; p++) state[slab + uint(p)] = next[p];
}

#else

[[vk::binding(0)]] StructuredBuffer<int> ids;
[[vk::binding(1)]] StructuredBuffer<int> indptr;
[[vk::binding(2)]] StructuredBuffer<int> replay;
[[vk::binding(3)]] StructuredBuffer<int> commit;
[[vk::binding(4)]] StructuredBuffer<int> slots;
[[vk::binding(5)]] RWStructuredBuffer<int> state;
[[vk::binding(6)]] StructuredBuffer<uint64_t> hash;
[[vk::binding(7)]] RWStructuredBuffer<int> ngram_ids;
struct Push { int lane0; int ngram; int heads; int heads_per_ngram; int eos; int lanes; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint r = gid.x;
    if (r >= uint(pc.lanes)) return;
    int begin = indptr[r];
    for (uint j = 0u; j < r; j++) begin += replay[uint(pc.lane0) + j];
    int replayed = replay[uint(pc.lane0) + r];
    int rows = (indptr[r + 1u] - indptr[r]) + replayed;
    if (rows <= 0) return;
    int slot = slots[uint(pc.lane0) + r];
    if (slot < 0) return;
    int span = pc.ngram - 1;
    uint slab = uint(slot) * uint(span);
    for (int t = 0; t < rows; t++) {
        int window[PIE_MAX_NGRAM];
        for (int p = 0; p < PIE_MAX_NGRAM; p++) window[p] = 0;
        window[0] = ids[uint(begin + t)];
        for (int p = 1; p <= span; p++) {
            window[p] = (t - p >= 0) ? ids[uint(begin + t - p)] : ple_cell(state[slab + uint(span - (p - t))], pc.eos);
        }
        ple_mask_window(window, pc.ngram, pc.eos);
        if (t < replayed) continue;
        int out_[PIE_MAX_HEADS];
        ple_hash_row(hash, window, pc.ngram, pc.heads, pc.heads_per_ngram, out_);
        uint own = uint(indptr[r] + (t - replayed));
        for (int k = 0; k < pc.heads; k++) ngram_ids[own * uint(pc.heads) + uint(k)] = out_[k];
    }
    int keep = commit[uint(pc.lane0) + r];
    if (keep > rows) keep = rows;
    if (keep <= 0) return;
    int next[PIE_MAX_NGRAM];
    for (int p = 0; p < PIE_MAX_NGRAM; p++) next[p] = 0;
    for (int p = 0; p < span; p++) {
        int src = keep - span + p;
        next[p] = src >= 0 ? ids[uint(begin + src)] + 1 : state[slab + uint(p + keep)];
    }
    for (int p = 0; p < span; p++) state[slab + uint(p)] = next[p];
}
#endif

// pie:instantiate ple_ngram_ids_update PIE_PLE_UPDATE=1
// pie:instantiate ple_ngram_ids_chunked PIE_PLE_CHUNKED=1
// pie:instantiate ple_ngram_ids_committed
